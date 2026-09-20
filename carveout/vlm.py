# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""VLM interface for the optional stages (1.5 vocabulary, 4.5 verification).

Every model call goes through this module, and there is no hosted backend:
Carveout runs a LOCAL model, so nothing here reaches the network, there is
no API key, and a finished scene never tells a third party it exists.

`LocalBackend` runs the configured model through transformers on the local
GPU. It loads on first use and is released at the end of the stage that asked
for it (see `close()`): the web server drives every stage in one process, and
a resident 16-20 GB model would starve the render and detect stages that run
after it. `backend_available()` reports missing weights or a missing
dependency so a front-end can say WHY a button is dead instead of graying it
out silently.

Both stages are OFF by default; the deterministic core never imports this
module.
"""

import gc
import io
import json
import logging
import re
import time
from pathlib import Path

from .refusal import Refusal

log = logging.getLogger(__name__)


# A bf16 checkpoint's 4-bit NF4 footprint as a fraction of its size on disk,
# for a model the catalogue has not measured yet (the preflight and the
# creation dialog both say "estimated" then). Measured: 0.34 for the 27B
# (17.5 of 51.7 GiB); the same rule serves both places.
FOUR_BIT_OF_DISK = 0.35


class VLMFormatError(RuntimeError):
    """The VLM would not produce the required JSON after format-reminder
    retries. Callers absorb this per unit of work (verify:
    instance -> UNVERIFIED, label kept, flagged) instead of aborting a stage."""


FORMAT_REMINDER = (
    "Your previous reply was not the required JSON. Answer with ONLY the "
    "JSON object specified above — exact keys, no prose, no self-correction, "
    "no code fences.")

# A reply that ran to the token cap is a list that never closed, not a
# malformed one. It is re-asked from the ORIGINAL prompt with a short stub
# of the runaway in the assistant turn: feeding the whole runaway back as
# context primed the same runaway on every retry (the 8B on a workbench
# scene: three identical 55 s attempts, then the view skipped).
CAP_REMINDER = (
    "Your previous answer did not end: the list ran past the length limit. "
    "Answer again with ONLY the JSON object specified above, shorter — keep "
    "the most distinct entries, each 1-3 words, and close the JSON.")

PROPOSE_PROMPT = """\
These images are synthetic renders of a 3D Gaussian Splatting scene (a real space
that was 3D-scanned). Identify the distinct physical OBJECT types visible.

Rules:
- SHORT noun phrases only, 1-3 lowercase words (e.g. "office chair", "monitor").
  These feed an open-vocabulary segmentation model (SAM 3 promptable concept
  segmentation): common object categories, NOT referring expressions ("the chair
  next to the desk" is invalid).
- Only objects actually visible in the renders. Exclude building structure
  (walls, floor, ceiling, doors) — we segment movable/discrete objects.
- Merge duplicates into one category; prefer the most standard name.
- At most {max_objects} object types: the distinct ones, not every variant.
- Also propose {n_distractors} DISTRACTOR concepts: object types that would be
  plausible in this kind of space but are clearly ABSENT from every render
  (used as negative controls for score calibration).

Answer with ONLY this JSON, no other text:
{{"objects": ["...", "..."], "distractors": ["...", "..."]}}"""


VERIFY_PROMPT = """\
These {n} crop(s) show ONE object detected in a 3D-scanned scene (synthetic
renders; some blur/fog is normal). The segmentation pipeline labeled it:
"{label}".

Judge the LABEL, not the render quality:
- CONFIRM: the label correctly names the main object in the crop(s).
- RELABEL: there IS a clear object, but a different SHORT noun phrase
  (1-3 lowercase words, a common category name) names it better.
- REJECT: no such discrete object is present (mis-segmentation, wall/floor
  patch, unrecognizable fragment).

Answer with ONLY this JSON, no other text:
{{"verdict": "CONFIRM" | "REJECT" | "RELABEL", "label": "<new label if RELABEL, else repeat>", "rationale": "<one line>"}}"""


# Name-first verification: the label is NOT in the first
# question. Told "this is labeled X", the model confirmed the wrong label
# 19 times out of 20 on a test scene; asked for a name on the
# same crops it caught five. The label enters only when the name differs,
# in a choice with both names on equal footing.
NAME_PROMPT = """\
These {n} crop(s) show ONE thing from a 3D-scanned scene (synthetic
renders; some blur/fog is normal). The crop(s) are centred on it.{note}

What is it? Answer with ONE short noun phrase (1-3 lowercase words, a
common category name — the kind of name a segmentation vocabulary uses)
and one line of reason. If no discrete object is there (a wall/floor
patch, an unrecognizable fragment), say so in the name: "none".

Answer with ONLY this JSON, no other text:
{{"name": "<noun phrase or none>", "reason": "<one line>"}}"""


CHOICE_PROMPT = """\
These {n} crop(s) show ONE thing from a 3D-scanned scene (synthetic
renders; some blur/fog is normal).{note} Two names were given for it,
independently:
  A: "{label}"
  B: "{name}"

Which is right? Judge the NAME, not the render quality:
- "A": "{label}" names it best.
- "B": "{name}" names it best.
- "C": both name it equally well (synonyms, or the same object at two
  levels of detail with neither clearly better).
- "D": neither — say what it is in "label" (1-3 lowercase words), or
  put "none" if no discrete object is there.

Answer with ONLY this JSON, no other text:
{{"choice": "A" | "B" | "C" | "D", "label": "<D's name, none, or empty>", "rationale": "<one line>"}}"""


# Specific over generic: a name outside the
# vocabulary may stand only when it is a KIND of the detected label. A
# generic word is never a kind of a specific one, so the umbrella terms
# the vocabulary rule guards against cannot come back through this.
KIND_PROMPT = """\
Two names for the same thing in a 3D-scanned scene (the crops are shown
for context only): the detected label "{label}" and the proposed name
"{name}".

How does "{name}" relate to "{label}"?
- "specific": a {name} IS a kind of {label} — a more specific name for
  the same object (an onion is a kind of vegetable; an oven mitt is a
  kind of glove).
- "part_or_whole": a {name} is a part of a {label}, or contains one, or
  is what a {label} sits in or on — not the same object.
- "other": neither; a different thing, or a more general word.

Answer with ONLY this JSON, no other text:
{{"relation": "specific" | "part_or_whole" | "other", "rationale": "<one line>"}}"""


# The isolated render: a small object's own 3D
# reconstruction, rendered alone, joins the crops as one more image — the
# best witness under ~300 Gaussians in the isolated-render experiment, the worst above.
ISOLATED_NOTE = (" The LAST image is the same object's own 3D reconstruction "
                 "rendered alone on black — no surroundings; use it together "
                 "with the crop(s).")


# Relabel support check: the crop may contain SEVERAL objects —
# the question is about the MARKED one only, so a salient neighbor cannot
# lend its identity to the instance being judged (soda can relabeled "water
# bottle" because a bottle stood beside it in every crop).
SUPPORT_PROMPT = """\
This crop is from a 3D-scanned scene (synthetic render; blur/fog is normal).
The RED RECTANGLE outlines the projection of ONE specific 3D object instance.
Ignore everything outside the red rectangle.

Is the object INSIDE the red rectangle a "{label}"?
- CONFIRM: yes, the marked object itself is a {label}.
- REJECT: no — the marked object is something else (say what in the
  rationale), or the rectangle contains no clear discrete object.
Do NOT confirm because a {label} is visible elsewhere in the crop.

Answer with ONLY this JSON, no other text:
{{"verdict": "CONFIRM" | "REJECT", "rationale": "<one line>"}}"""


LENGTH_PROMPT = """\
These {n} images are synthetic renders of ONE 3D-scanned scene. A red
segment from A to B is drawn on the scene in each image (the same segment,
seen from different viewpoints). Judging by the objects around it — doors,
furniture, people-sized things, tiles, gravestones, cars, whatever you can
see — how long is the segment A–B, in metres?

Answer with ONLY a JSON object, no prose, no code fences:
{{"length_m": <number>, "what": "<what A–B spans: the height of a door, the
  width of a bench, ...>", "reason": "<one sentence>"}}

length_m: your best single estimate of the real-world length of A–B, in
metres (a decimal number, not a range).
what: what the segment spans, in a few words.
reason: one sentence naming the objects whose known size you judged by.
"""


def _prep_image(img, max_side: int = 768, quality: int = 88):
    """Accepts a Path or a PIL.Image (Stage 4.5 sends in-memory crops); returns
    a PIL.Image for the processor.

    The JPEG round trip is deliberate and is NOT vestigial API-upload thrift:
    every reference verdict in `work/*/stage45/verdicts.csv` was produced from
    an image that went through exactly this downsize-and-recompress, so keeping
    it is what makes the local model comparable with that bank. `max_side` is a
    knob worth testing on its own — small
    objects are the documented weak spot — but it is changed as a measured
    variable, never as a side effect of swapping the backend."""
    from PIL import Image

    im = (img if isinstance(img, Image.Image) else Image.open(img)).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class VLMBackend:
    """What a backend must provide. The four methods below are the entire
    contract; `_ask_json` is deliberately NOT part of it — a local backend
    with grammar-constrained decoding cannot produce malformed JSON and needs
    no repair loop, while the retry contract (raise VLMFormatError, which
    callers absorb per unit of work) stays the same either way."""

    def verify_label(self, images: list, label: str) -> dict:
        """-> {verdict: CONFIRM|REJECT|RELABEL, label, rationale}"""
        raise NotImplementedError

    def name_object(self, images: list, note: str = "") -> dict:
        """Name-first: the crops, NO label. -> {name, reason};
        name "none" when no discrete object is there. `note` says what
        an extra image is (the isolated render)."""
        raise NotImplementedError

    def choose_label(self, images: list, label: str, name: str,
                     note: str = "") -> dict:
        """The detected label and the name-first name on equal footing.
        -> {choice: A|B|C|D, label, rationale}"""
        raise NotImplementedError

    def kind_of(self, images: list, label: str, name: str) -> dict:
        """Specific over generic: is `name` a kind of `label`?
        -> {relation: specific|part_or_whole|other, rationale}"""
        raise NotImplementedError

    def support_label(self, image, label: str) -> dict:
        """Marked-box support check — the caller has drawn the red rectangle.
        -> {verdict: CONFIRM|REJECT, rationale}"""
        raise NotImplementedError

    def propose_vocabulary(self, image_paths: list[Path], cfg: dict) -> dict:
        """-> {objects: [...], distractors: [...]}"""
        raise NotImplementedError

    def read_length(self, image_paths: list[Path]) -> dict:
        """The second opinion on the ruler: how long the
        measured segment A–B, drawn in red into the probe renders, is in
        metres, judging by the objects around it. -> {length_m: float,
        what: str, reason: str}. The Check card states it beside the
        operator's measurement, flagged at a 2x disagreement; nothing
        applies it."""
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the backend holds. Stages call this in a finally;
        a backend that holds nothing needs no override."""


# Prose in unwrapped paragraphs: the panels wrap to their own width, and a
# hard break inside a sentence wraps twice. Only the commands keep lines.
MODEL_HELP = """\
No vision model installed{where}.

It is optional: without it the measurement gets no second opinion, the vocabulary is written by hand and labels ship unverified.

To add one (Apache-2.0, not gated, runs locally; README step 7), {choices}:

{downloads}"""

OTHER_MODEL_HELP = """\
{rel} holds no model, but {installed} is installed: pick it on the Vocabulary or Objects panel.
"""

DEPS_HELP = """\
ERROR: the local VLM needs transformers, accelerate and bitsandbytes, and
{what} is not importable in this environment ({err}).

  conda activate carveout
  pip install "transformers>=5.8,<6" "accelerate>=1.1" "bitsandbytes>=0.50"

These are pure additions: they do NOT re-resolve torch, which stays pinned at
2.7.1+cu128 (see environment.yml).
"""


def _resolve_model_dir(cfg: dict | None) -> Path:
    """Config carries a PATH, never a hub id, so a run can never silently
    reach the network for weights. Relative paths resolve against the repo
    root, the same way detect.py resolves the SAM 3 checkpoints."""
    if cfg is None:                      # settings dialog: no run loaded yet
        from .config import load_config
        cfg = load_config()
    d = Path(cfg["vlm"]["model_dir"])
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    return d


def _parse_json(text: str) -> dict:
    """The reply, stripped of everything a chat model wraps JSON in.

    A thinking model emits <think>...</think> first; every model sometimes
    reaches for a code fence despite the prompt. Both are recovered here
    rather than burned as a format-reminder retry."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j <= i:
        raise ValueError(f"no JSON object in reply: {t[:200]!r}")
    return json.loads(t[i:j + 1])


def _complete_entries(text: str, key: str) -> list[str]:
    """The complete string entries of the list under `key` in a reply that
    ran past the token cap: everything up to the last closed quote (or
    the closing bracket when the list did close). [] when the key is
    absent. What a runaway list still carries is real data — the objects
    the model saw before it stopped closing its brackets."""
    m = re.search(r'"%s"\s*:\s*\[' % re.escape(key), text)
    if not m:
        return []
    seg = text[m.end():]
    end = seg.find("]")
    if end != -1:
        seg = seg[:end]
    return [e for e in re.findall(r'"((?:[^"\\]|\\.)*)"', seg) if e.strip()]


def _cancel_criteria(should_cancel):
    """A generation stopping criterion that reads the job's cancel flag on
    every token, so Stop ends a 55 s generate call within one token
    instead of at the end of the stage. None when there is no probe."""
    if should_cancel is None:
        return None
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _CancelStop(StoppingCriteria):
        fired = False

        def __call__(self, input_ids, scores, **kwargs):
            if not self.fired and should_cancel():
                self.fired = True
            return torch.full((input_ids.shape[0],), self.fired,
                              dtype=torch.bool, device=input_ids.device)
    return StoppingCriteriaList([_CancelStop()])


class LocalBackend(VLMBackend):
    """The configured VLM, run locally through transformers.

    Weights load on FIRST CALL and are freed by `close()`, which the stages
    call in a finally. That is not tidiness: the web server runs every stage
    in one long-lived process, so a model left resident would hold 16-20 GB
    for the rest of the session and starve the render and detect stages that
    come after it — the same load-and-release discipline gsplat and SAM 3
    already follow."""

    def __init__(self, cfg: dict, should_cancel=None):
        self.cfg = cfg
        self.vcfg = cfg["vlm"]
        self.model_dir = _resolve_model_dir(cfg)
        # the job's cancel probe: read every generated token and between
        # retries, so Stop reaches a vision-model stage
        self._should_cancel = should_cancel
        self._model = None
        self._proc = None
        self._think_off: dict = {}
        self.quant: str | None = None     # 'bf16' | '4bit' | 'asis' once loaded
        self.calls = 0
        self.format_failures = 0

    # -- lifecycle ---------------------------------------------------------
    def _prequantized(self) -> bool:
        """A checkpoint that ships already quantised (AWQ/GPTQ/bnb) carries
        `quantization_config` in its config.json and loads as it is: no
        BitsAndBytes wrapper, and its on-disk size IS its GPU size."""
        try:
            cj = json.loads((self.model_dir / "config.json").read_text())
        except (OSError, ValueError):
            return False
        return bool(cj.get("quantization_config"))

    def _quantization(self) -> str:
        """'4bit', or 'asis' for a checkpoint that ships quantised — measured
        against the card actually present, AFTER giving the allocator's
        cache back, so the stage before this one (a render that reserved
        half the card) does not count as occupied.

        Every model loads COMPRESSED. The
        earlier rule took full precision whenever it happened to fit, which
        loaded the 8B at the same 17.5 GiB peak as the compressed 27B on a
        24 GB card — while the creation dialog had shown its 4-bit need. A
        smaller model exists to leave room on the card; one rule, the same
        number in the dialog and here. The need is the measured 4-bit peak
        when the catalogue carries one (`vlm.vram_4bit_gib`), else an
        estimate from the checkpoint's size, and the log says which."""
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
        on_disk = sum(f.stat().st_size
                      for f in self.model_dir.glob("*.safetensors")) / 2**30
        name = self.vcfg["model"]
        if self._prequantized():
            need = on_disk * 1.15
            log.info("VLM preflight: %s ships quantised, %.1f GiB on disk, "
                     "needs ~%.1f GiB; %.1f of %.1f GiB free",
                     name, on_disk, need, free, total)
            if need < free:
                return "asis"
            raise Refusal(
                f"ERROR: {name} needs about {need:.1f} GiB of GPU memory "
                f"free to load; {free:.1f} GiB is free. Close other GPU "
                f"programs (a browser tab showing this scene holds a copy "
                f"of it), or start the server with a smaller model: "
                f"carveout web --vlm-dir DIR.")
        measured = self.vcfg.get("vram_4bit_gib")
        need_4bit = (float(measured) if measured
                     else on_disk * FOUR_BIT_OF_DISK)
        how = "measured" if measured else "estimated from the checkpoint size"
        fits = need_4bit < free
        log.info("VLM preflight: %s, 4-bit needs ~%.1f GiB (%s); %.1f of "
                 "%.1f GiB free -> %s", name, need_4bit, how, free, total,
                 "4bit" if fits else "does not fit")
        if fits:
            return "4bit"
        raise Refusal(
            f"ERROR: {name} needs about {need_4bit:.1f} GiB of GPU memory "
            f"free to load 4-bit ({how}); {free:.1f} GiB is free. Close "
            f"other GPU programs (a browser tab showing this scene holds a "
            f"copy of it), or start the server with a smaller model: "
            f"carveout web --vlm-dir DIR.")

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise Refusal(DEPS_HELP.format(what="transformers", err=e))
        if not (self.model_dir / "config.json").exists():
            raise Refusal(_missing_weights(self.cfg))
        if not torch.cuda.is_available():
            raise Refusal(
                f"ERROR: no CUDA device, so {self.vcfg['model']} cannot run. "
                f"The vocabulary proposal and the label verification need "
                f"the GPU; the rest of the pipeline does not. Skip the verify "
                f"gate to ship labels unverified.")

        quant = self.quant = self._quantization()
        kw = dict(dtype=torch.bfloat16, device_map="cuda:0")
        if quant == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes   # import-time CUDA check
            except ImportError as e:
                raise Refusal(DEPS_HELP.format(what="bitsandbytes", err=e))
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)

        # Sampling as the model ships it: both candidates set do_sample=true
        # in generation_config.json with their own tuned temperature/top_p,
        # and Qwen documents greedy decoding as causing endless repetition.
        # It does — see the fix commit. Reproducibility comes from seeding
        # ONCE here rather than before every call: per-call seeding replays
        # the same RNG stream into each format-reminder retry, which is
        # exactly backwards for a repair loop.
        torch.manual_seed(self.vcfg.get("seed", 0))
        t0 = time.perf_counter()
        self._proc = AutoProcessor.from_pretrained(str(self.model_dir))
        self._model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_dir), **kw).eval()
        # Thinking models take an off switch in their chat template; the
        # Instruct ones warn on every call if handed a kwarg their template
        # never declared. Ask the template which it is, once.
        tmpl = (getattr(self._proc, "chat_template", None)
                or getattr(getattr(self._proc, "tokenizer", None),
                           "chat_template", "") or "")
        self._think_off = ({"enable_thinking": False}
                           if "enable_thinking" in tmpl else {})
        log.info("VLM: loaded %s (%s) from %s in %.1f s, %.2f GiB on GPU",
                 self.vcfg["model"], quant, self.model_dir,
                 time.perf_counter() - t0,
                 torch.cuda.memory_allocated() / 2**30)

    def close(self) -> None:
        """Release the weights. Safe to call when nothing was ever loaded."""
        if self._model is None:
            return
        import torch
        peak = torch.cuda.max_memory_allocated() / 2**30
        del self._model, self._proc
        self._model = self._proc = None
        gc.collect()
        torch.cuda.empty_cache()
        log.info("VLM: released after %d calls (%d format failures); "
                 "peak %.2f GiB", self.calls, self.format_failures, peak)

    # -- one exchange ------------------------------------------------------
    def _eos_ids(self) -> set[int]:
        e = getattr(self._model.generation_config, "eos_token_id", None)
        return set(e if isinstance(e, (list, tuple)) else
                   [] if e is None else [e])

    def _chat(self, messages: list, max_new: int, **gen) -> tuple[str, bool]:
        """One exchange: (reply, hit_cap). `hit_cap` is true when the model
        was still going at `max_new` tokens — a list that never closed
        reads differently from a malformed one, and is retried
        differently. `gen` are extra generate() kwargs for this call."""
        import torch
        from .sequencing import check_cancel

        self._load()
        inputs = self._proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            # A verdict is a one-line judgement: a <think> block costs seconds
            # per call and buys nothing. Only passed when the template
            # declares the switch; _parse_json strips a block that arrives
            # anyway.
            **self._think_off).to(self._model.device)
        n_in = inputs["input_ids"].shape[-1]
        stop = _cancel_criteria(self._should_cancel)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_new,
                                       stopping_criteria=stop, **gen)
        self.calls += 1
        check_cancel(self._should_cancel)   # the criterion ended the call
        ids = out[0, n_in:]
        hit_cap = (ids.shape[-1] >= max_new
                   and int(ids[-1]) not in self._eos_ids())
        return (self._proc.batch_decode(out[:, n_in:],
                                        skip_special_tokens=True)[0],
                hit_cap)

    def _ask_json(self, images: list, prompt: str, check,
                  max_new: int = 400, salvage=None, gen: dict | None = None
                  ) -> dict:
        """The format-repair loop, unchanged in contract from the API path:
        `check` raises ValueError on a bad shape, the bad reply plus
        FORMAT_REMINDER are re-asked `vlm.json_retries` times, and the final
        failure raises VLMFormatError — which callers absorb per unit of work
        rather than aborting a stage.

        A reply that ran to the token cap is the other failure: a list the
        model never closed. It is re-asked from the original prompt with
        CAP_REMINDER (not with the runaway as context), and when the LAST
        attempt still runs to the cap, `salvage(reply)` may turn the
        complete entries into an answer — returned loudly, never silently."""
        from .sequencing import check_cancel
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        base = [{"role": "user", "content": content}]
        messages = base
        last, reply, hit_cap = "", "", False
        for attempt in range(self.vcfg["json_retries"] + 1):
            if attempt:
                check_cancel(self._should_cancel)   # between retries
            reply, hit_cap = self._chat(messages, max_new, **(gen or {}))
            if hit_cap:
                n = len(_complete_entries(reply, "objects"))
                last = (f"reply ran to the {max_new}-token cap without "
                        f"closing the JSON ({n} complete entries): "
                        f"{reply[:120]!r}…")
                log.warning("VLM format failure, re-asking: %s", last)
                messages = base + [
                    {"role": "assistant",
                     "content": [{"type": "text", "text": reply[:300] + " …"}]},
                    {"role": "user",
                     "content": [{"type": "text", "text": CAP_REMINDER}]}]
                continue
            try:
                obj = _parse_json(reply)
                check(obj)
                return obj
            except (ValueError, json.JSONDecodeError) as e:
                last = f"{type(e).__name__}: {e}"
                log.warning("VLM format failure, re-asking: %s", last)
                messages = messages + [
                    {"role": "assistant",
                     "content": [{"type": "text", "text": reply}]},
                    {"role": "user",
                     "content": [{"type": "text", "text": FORMAT_REMINDER}]}]
        self.format_failures += 1
        if hit_cap and salvage is not None:
            obj = salvage(reply)
            if obj is not None:
                try:
                    check(obj)
                except ValueError:
                    obj = None
            if obj is not None:
                log.warning("VLM: every attempt ran to the token cap; kept "
                            "the %d complete entries of the last reply; "
                            "review them, they are what the model listed "
                            "before it stopped closing its brackets",
                            len(obj.get("objects") or []))
                return obj
        raise VLMFormatError(last)

    def _budget(self, stage: str, default: int) -> int:
        """Token budget per stage. A verdict is one line; a vocabulary is a
        list of 40-60 entries and needs an order of magnitude more. The old
        single max_tokens=1000 was inherited from the API path and is what
        made stage 1.5 return NOTHING: the reply was cut mid-array, failed
        every format retry, and the stage proposed no vocabulary at all."""
        v = self.vcfg.get("max_new_tokens")
        if isinstance(v, dict):
            return int(v.get(stage, default))
        return int(v) if v else default

    # -- the three methods -------------------------------------------------
    def verify_label(self, images: list, label: str) -> dict:
        def check(o):
            if o.get("verdict") not in ("CONFIRM", "REJECT", "RELABEL"):
                raise ValueError(f"verdict={o.get('verdict')!r}")
            if o["verdict"] == "RELABEL" and not str(o.get("label", "")).strip():
                raise ValueError("RELABEL without a label")
            o.setdefault("label", label)
            o.setdefault("rationale", "")

        ims = [_prep_image(i, self.vcfg.get("max_side", 768)) for i in images]
        return self._ask_json(
            ims, VERIFY_PROMPT.format(n=len(ims), label=label), check,
            max_new=self._budget("verify", 400))

    def name_object(self, images: list, note: str = "") -> dict:
        def check(o):
            if not str(o.get("name", "")).strip():
                raise ValueError("no name")
            o["name"] = " ".join(str(o["name"]).lower().split())
            o.setdefault("reason", "")

        ims = [_prep_image(i, self.vcfg.get("max_side", 768)) for i in images]
        return self._ask_json(ims, NAME_PROMPT.format(n=len(ims), note=note),
                              check, max_new=self._budget("verify", 400))

    def choose_label(self, images: list, label: str, name: str,
                     note: str = "") -> dict:
        def check(o):
            if o.get("choice") not in ("A", "B", "C", "D"):
                raise ValueError(f"choice={o.get('choice')!r}")
            o["label"] = " ".join(str(o.get("label") or "").lower().split())
            o.setdefault("rationale", "")

        ims = [_prep_image(i, self.vcfg.get("max_side", 768)) for i in images]
        return self._ask_json(
            ims, CHOICE_PROMPT.format(n=len(ims), label=label, name=name,
                                      note=note),
            check, max_new=self._budget("verify", 400))

    def kind_of(self, images: list, label: str, name: str) -> dict:
        def check(o):
            if o.get("relation") not in ("specific", "part_or_whole", "other"):
                raise ValueError(f"relation={o.get('relation')!r}")
            o.setdefault("rationale", "")

        ims = [_prep_image(i, self.vcfg.get("max_side", 768)) for i in images]
        return self._ask_json(
            ims, KIND_PROMPT.format(label=label, name=name), check,
            max_new=self._budget("verify", 400))

    def support_label(self, image, label: str) -> dict:
        def check(o):
            if o.get("verdict") not in ("CONFIRM", "REJECT"):
                raise ValueError(f"verdict={o.get('verdict')!r}")
            o.setdefault("rationale", "")

        ims = [_prep_image(image, self.vcfg.get("max_side", 768))]
        return self._ask_json(ims, SUPPORT_PROMPT.format(label=label), check,
                              max_new=self._budget("support", 400))

    def propose_vocabulary(self, image_paths: list[Path], cfg: dict) -> dict:
        def check(o):
            for k in ("objects", "distractors"):
                if not isinstance(o.get(k), list) or \
                        not all(isinstance(x, str) for x in o[k]):
                    raise ValueError(f"{k} is not a list of strings")
            if not o["objects"]:
                raise ValueError("no objects proposed")

        def salvage(reply):
            objects = _complete_entries(reply, "objects")
            return (dict(objects=objects,
                         distractors=_complete_entries(reply, "distractors"))
                    if objects else None)

        vcfg = cfg["vlm"]
        ims = [_prep_image(p, self.vcfg.get("max_side", 768))
               for p in image_paths]
        # The bound and the penalty are what stop a runaway list; the cap
        # reminder and the salvage are what a runaway still costs.
        gen = {}
        if vcfg.get("propose_repetition_penalty"):
            gen["repetition_penalty"] = float(vcfg["propose_repetition_penalty"])
        return self._ask_json(
            ims, PROPOSE_PROMPT.format(
                n_distractors=vcfg["distractors"],
                max_objects=vcfg.get("propose_max_objects", 40)),
            check, max_new=self._budget("propose", 2000),
            salvage=salvage, gen=gen)

    def read_length(self, image_paths: list[Path]) -> dict:
        def check(o):
            try:
                length = float(o.get("length_m"))
            except (TypeError, ValueError):
                raise ValueError(f"length_m={o.get('length_m')!r} is not a "
                                 f"number") from None
            if not length > 0:
                raise ValueError(f"length_m={length} is not positive")
            o["length_m"] = length
            o["what"] = str(o.get("what") or "").strip()
            o["reason"] = str(o.get("reason") or "").strip()

        ims = [_prep_image(p, self.vcfg.get("max_side", 768))
               for p in image_paths]
        # All the probes in ONE call: the question is about one segment
        # seen from several viewpoints, and the context views are what let
        # the model judge it against the objects around it.
        return self._ask_json(ims, LENGTH_PROMPT.format(n=len(ims)), check,
                              max_new=self._budget("read", 200))


def _missing_weights(cfg: dict | None) -> str:
    """The refusal when the configured directory holds no weights. Another
    model on disk is the remedy when there is one; otherwise a download
    line per catalogue model, the configured one first."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    rel = cfg["vlm"]["model_dir"]
    others = [m["name"] for m in list_models(cfg) if not m["current"]]
    if others:
        return OTHER_MODEL_HELP.format(rel=rel, installed=" or ".join(others))
    base = Path(rel).name
    cat = cfg["vlm"].get("catalogue") or {}
    order = ([base] if base in cat else []) + [k for k in cat if k != base]
    listed = [(k, cat[k]) for k in order if cat[k].get("hf_repo")]
    choices = " or ".join(
        f"{e.get('name', k)} ({e['vram_4bit_gib']:g} GB{' of VRAM' if i == 0 else ''})"
        if e.get("vram_4bit_gib") else e.get("name", k)
        for i, (k, e) in enumerate(listed))
    downloads = "".join(
        f"  hf download {e['hf_repo']} --local-dir models/{k}/\n"
        for k, e in listed)
    where = (f" under --vlm-dir {rel}"
             if cfg["vlm"].get("model_source") == "--vlm-dir" else "")
    return MODEL_HELP.format(where=where, choices=choices,
                             downloads=downloads)


def list_models(cfg: dict | None) -> list[dict]:
    """The vision models on this machine: every directory under `models/`
    holding a loadable checkpoint (config.json plus weights), plus the one
    the config or the server flag names if it lives elsewhere. Status
    only — Carveout never downloads; README step 7 does."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    from .config import model_entry
    root = Path(__file__).resolve().parent.parent
    dirs: list[Path] = []
    mroot = root / "models"
    if mroot.is_dir():
        dirs += sorted(d for d in mroot.iterdir() if d.is_dir())
    cur = _resolve_model_dir(cfg)
    if cur not in dirs:
        dirs.append(cur)
    out = []
    for d in dirs:
        weights = list(d.glob("*.safetensors"))
        if not (d / "config.json").is_file() or not weights:
            continue
        size = sum(f.stat().st_size for f in weights) / 2**30
        e = model_entry(cfg, d)
        try:
            pre = bool(json.loads((d / "config.json").read_text())
                       .get("quantization_config"))
        except (OSError, ValueError):
            pre = False
        if pre:
            need, how = size * 1.15, "estimated (ships quantised)"
        elif e["vram_4bit_gib"]:
            need, how = float(e["vram_4bit_gib"]), "measured"
        else:
            need, how = size * FOUR_BIT_OF_DISK, "estimated from the size"
        try:
            rel = str(d.relative_to(root))
        except ValueError:
            rel = str(d)
        out.append(dict(dir=rel, name=e["name"], size_gib=round(size, 1),
                        need_gib=round(need, 1), how=how,
                        current=(d == cur)))
    return out


def scene_gaussian_count(path: str | Path) -> int | None:
    """The Gaussian count from the container's header alone — no decode:
    `element vertex N` from a .ply header, `count` from a .sog's
    meta.json. None when the header does not say."""
    p = Path(path)
    try:
        if p.suffix.lower() == ".ply":
            with open(p, "rb") as f:
                for _ in range(200):
                    line = f.readline()
                    if not line or line.strip() == b"end_header":
                        break
                    if line.startswith(b"element vertex"):
                        return int(line.split()[2])
        elif p.suffix.lower() == ".sog":
            import zipfile
            with zipfile.ZipFile(p) as zf:
                return int(json.loads(zf.read("meta.json"))["count"])
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return None


# What the browser's copy of a scene costs on the card, per Gaussian: the
# splat renderer keeps packed positions, covariances and colours resident
# and builds sort buffers per frame — an order-of-magnitude figure, said as
# one ("about"), never a promise.
BROWSER_BYTES_PER_GAUSSIAN = 64


def estimate_fit(cfg: dict | None, scene_path: str | Path | None) -> dict:
    """The creation dialog's proposal: the card, what the browser's copy of
    this scene takes, and for each installed model whether its need fits
    in what is left. A proposal the operator sees and may change — never
    a choice Carveout makes."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    card = None
    try:
        import torch
        if torch.cuda.is_available():
            card = torch.cuda.mem_get_info()[1] / 2**30
    except Exception:              # pragma: no cover - no torch, no card
        card = None
    n = scene_gaussian_count(scene_path) if scene_path else None
    browser = (n * BROWSER_BYTES_PER_GAUSSIAN / 2**30) if n else None
    # a fixed allowance for the desktop, the CUDA context and the render's
    # own workspace that the job-boundary free does not hand back
    reserve = 2.0
    left = (card - reserve - (browser or 0)) if card else None
    models = list_models(cfg)
    for m in models:
        m["fits"] = (None if left is None else
                     "yes" if m["need_gib"] <= left - 1.0 else
                     "tight" if m["need_gib"] <= left else "no")
    # proposed: the largest model that fits, else the largest that is tight,
    # else the smallest; the current server default breaks ties
    order = sorted(models, key=lambda m: (-m["need_gib"], not m["current"]))
    proposed = (next((m for m in order if m["fits"] == "yes"), None)
                or next((m for m in order if m["fits"] == "tight"), None)
                or (order[-1] if order else None))
    return dict(gaussians=n, card_gib=round(card, 1) if card else None,
                browser_gib=round(browser, 1) if browser else None,
                reserve_gib=reserve, models=models,
                proposed=proposed["dir"] if proposed else None)


def describe_model(cfg: dict | None) -> dict:
    """What the front-ends show about the model in use: its name, the
    resolved directory, its size on disk and where the choice came from
    (the config, or `carveout web --vlm-dir`). Status only — never a key."""
    d = _resolve_model_dir(cfg)
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    on_disk = sum(f.stat().st_size for f in d.glob("*.safetensors")) / 2**30
    return dict(vlm_model=cfg["vlm"]["model"], vlm_model_dir=str(d),
                vlm_size_gib=round(on_disk, 1),
                vlm_source=cfg["vlm"].get("model_source") or "config")


def backend_available(cfg: dict | None) -> tuple[bool, str]:
    """(usable, reason). Front-ends call this to say WHY an LLM-stage action
    is unavailable rather than disabling it with no explanation: weights not
    downloaded yet, or the environment missing a dependency."""
    try:
        import transformers   # presence check only
    except ImportError as e:
        return False, DEPS_HELP.format(what="transformers", err=e)
    # config.json, not the directory: a half-finished `hf download` leaves the
    # directory behind, and "available" must mean loadable.
    if not (_resolve_model_dir(cfg) / "config.json").exists():
        return False, _missing_weights(cfg)
    return True, ""


def get_backend(cfg: dict, should_cancel=None) -> VLMBackend:
    ok, reason = backend_available(cfg)
    if not ok:
        # a refusal with a remedy, surfaced verbatim — never a traceback
        # that reads like a crash
        raise Refusal(reason)
    return LocalBackend(cfg, should_cancel)


def _merge_view_proposals(per_view: list, cfg: dict) -> dict:
    """Union of one proposal per view.

    Asking about six views in ONE call makes the model describe the room:
    it answers "workshop, desk, shelves" and never mentions the pen on the
    bench. Asking per view and unioning is what recovers small objects —
    measured on the Demo scene, 3 of its 14 confirmed prompts to 7.

    Dedupe is case- and whitespace-insensitive only, keeping the first
    surface form seen. Deliberately NOT stemmed: collapsing plurals mangles
    'cross' into 'cros' and 'glass' into 'glas', and these strings are fed
    verbatim to SAM 3 as concept prompts.

    Ordering is by how many views proposed a term. A term six views agree on
    is worth reviewing before one a single view guessed, and that ranking is
    free — the operator reads this list top-down.
    """
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    for _idx, prop in per_view:
        for o in prop.get("objects") or []:
            k = " ".join(str(o).strip().lower().split())
            if not k:
                continue
            surface.setdefault(k, str(o).strip())
            counts[k] = counts.get(k, 0) + 1
    # Fold a plural into its singular ONLY when the model proposed both —
    # "cable" and "cables" from different views are one concept, and SAM 3
    # would probe them twice. Never stem otherwise: guessing turns "cross"
    # into "cros" and these strings are sent to the detector verbatim.
    for k in sorted(counts, key=len, reverse=True):
        base = (k[:-2] if k.endswith("es") and k[:-2] in counts
                else k[:-1] if k.endswith("s") and k[:-1] in counts else None)
        if base:
            counts[base] += counts.pop(k)
            surface.pop(k, None)
    objects = sorted(surface, key=lambda k: (-counts[k], k))

    # Distractors are NEGATIVE CONTROLS: concepts absent from the scene. Per
    # view, "absent" only means absent from THAT view — a thing missing in
    # view 1 is often the subject of view 4 — so a candidate only survives if
    # NO view proposed it as an object. That is a stricter control than the
    # single-call version could give.
    seen, distractors = set(), []
    for _idx, prop in per_view:
        for d in prop.get("distractors") or []:
            k = " ".join(str(d).strip().lower().split())
            if k and k not in counts and k not in seen:
                seen.add(k)
                distractors.append(str(d).strip())
    if not distractors:
        log.warning("stage 1.5: every distractor proposed was also proposed "
                    "as a present object: none survived; the probe will run "
                    "without negative controls unless you add some")

    return {"objects": [surface[k] for k in objects],
            "distractors": distractors[:cfg["vlm"]["distractors"]],
            "view_counts": {surface[k]: counts[k] for k in objects}}


def run_propose_prompts(workdir: str, cfg: dict, should_cancel=None) -> dict:
    """Stage 1.5: send representative Stage 1 renders, save + print the proposal.

    Confirmation gate: detection NEVER runs on an unreviewed vocabulary —
    the vocabulary gate confirms an explicit prompt list the operator
    writes after reviewing/editing this proposal.
    """
    stage1 = Path(workdir) / "stage1"
    cams_path = stage1 / "cameras.json"
    if not cams_path.exists():
        raise Refusal(
            f"{cams_path} not found; render first (the render gate)",
            gate="render")
    frames = json.loads(cams_path.read_text())["frames"]
    n = min(cfg["vlm"]["max_views"], len(frames))
    # Evenly spaced over the kept set: positions are farthest-point sampled, so
    # this spreads the views across the room.
    picks = [frames[round(i * (len(frames) - 1) / max(n - 1, 1))] for i in range(n)]
    paths = [stage1 / f["file"] for f in picks]
    log.info("stage 1.5: sending %d views to %s: %s",
             n, cfg["vlm"]["model"], [p.name for p in paths])

    from .sequencing import check_cancel
    backend = get_backend(cfg, should_cancel)
    try:
        # ONE view per call, answers unioned: asking about six at once made
        # the model describe the room and miss the objects on the bench
        # (3/14 -> 7/14 of a confirmed vocabulary). The single-call
        # alternative was an A/B branch, removed.
        per_view, failed = [], 0
        for pth, fr in zip(paths, picks):
            check_cancel(should_cancel)   # between views
            try:
                per_view.append((fr["frame_idx"],
                                 backend.propose_vocabulary([pth], cfg)))
            except VLMFormatError as e:
                # A view that will not parse costs ONE view, not the
                # stage. Asking once for all six meant a single bad reply
                # returned no vocabulary at all.
                failed += 1
                log.warning("stage 1.5: view %d unparseable, skipped: %s",
                            fr["frame_idx"], e)
        if not per_view:
            raise VLMFormatError(
                f"no view produced a parseable vocabulary ({failed} tried)")
        proposal = _merge_view_proposals(per_view, cfg)
        proposal["views_failed"] = failed
    finally:
        backend.close()      # never leave the model resident past the stage
    proposal["views_used"] = [f["frame_idx"] for f in picks]
    proposal["model"] = cfg["vlm"]["model"]

    out = Path(workdir) / "stage15"
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposed_vocab.json").write_text(json.dumps(proposal, indent=1))

    print("\n=== Stage 1.5 proposed vocabulary (REVIEW REQUIRED) ===")
    counts = proposal.get("view_counts") or {}
    if counts:
        print("objects (views that saw each, of "
              f"{n - proposal.get('views_failed', 0)}):")
        print("    " + ", ".join(f"{o} [{counts.get(o, 0)}]"
                                 for o in proposal["objects"]))
    else:
        print("objects:    " + ", ".join(proposal["objects"]))
    print("distractors: " + ", ".join(proposal.get("distractors", [])))
    print(f"(from {n} views; saved to {out / 'proposed_vocab.json'})")
    print("Detection will only run on a vocabulary confirmed at the "
          "vocabulary gate.")
    return proposal
