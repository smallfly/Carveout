# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 4.5 (optional; the Verify panel's Run verification): LLM
verification of instance labels.

For each exported instance: crop its best 1-3 views (frames[] is sorted by
score desc; zero-confidence instances fall back to a projection-derived crop),
send to the vlm.py backend with the assigned label, and apply the structured
verdict: REJECT removes the object from interactions.json (kept with
rationale in stage45/rejected_instances.json).

RELABEL policy (KEEP BOTH — supersedes an earlier
label rewrite): the DETECTED label stays canonical in every legacy field
(objects[].label, frame_annotations[].label — schema compatibility); the
verifier's verdict goes under extended.objects[i].verified_label with
verify_rationale + verify_verdict (CONFIRM records verified_label == the
detected label). The viewer displays verified_label when present (the
label-source setting can show the detected term instead). RELABELs never auto-modify the vocabulary. All verdicts +
rationales also go to stage45/verdicts.csv alongside the presence scores so
this stage's FP reduction is measurable.

RELABEL bar (config-gated vlm.relabel_*): relabels are held to
a higher standard than confirms — the proposed label must be re-confirmed
on crops of the instance's OWN gaussian projections in >= 2 distinct views;
a single-crop verdict may CONFIRM but never RELABEL. Failures become
verdict RELABEL_UNSUPPORTED and keep the detected label.

Format robustness: a VLM reply that stays malformed after the
backend's format-reminder retries (vlm.json_retries) marks THAT instance
UNVERIFIED — label kept, flagged in verdicts.csv / extended.verification —
whether the failure hit the initial verdict or the relabel support check;
the stage never aborts on format noise.

HELD: an instance whose association evidence is
unresolved — export-time hold reasons (`ambiguous_det`, `refused_merge`,
`ambiguous_unassigned`) or the `projection_mismatch` crop path here — is
never CONFIRMED or RELABELED by the VLM alone: after the call, CONFIRM and
every RELABEL* verdict become HELD, with the original verdict, the proposed
label and the support counts preserved. REJECT stands (nothing matched even
in the wide crop) and UNVERIFIED stands (format failure) — both keep the
reasons as metadata and are counted as themselves. Held instances ship
unresolved, like the relabel holds, and the report says so; `verify_consent`
records consent to run, not acceptance of what came out.

The pre-verification interactions.json is preserved at
stage45/interactions.pre_verify.json; the updated file keeps `objects` /
`frame_annotations` positional identity consistent (object_idx remapped).

Checkpoint: every judged verdict is appended to
stage45/verdicts.partial.jsonl as it lands, under a header naming the
export and the parameters it was judged with. A run that stops — cancel,
crash, power — resumes from it: the same export under the same settings
continues where it stopped, and says so; anything else discards it. The
file goes when verdicts.csv is written.

Marked subset: an operator may mark instances for verification
(`extended.objects[i].verify_requested`, an explicit act on the export,
never a gate). When any are marked, only those are judged; every other
instance carries its verdict from the previous verification of THIS
export when there is one, else ships NOT_JUDGED — label kept as
detected, flagged, no verified_label. The marks are part of the stage's
fingerprint: a changed set is a fresh run, never a replay.
"""

import csv
import json
import re
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .exemplars import stage2_exemplar_record, stale_exemplars
from .sequencing import (fingerprint, stale_params, stale_upstream,
                         upstream_fingerprint)
from .manual_views import (frame_intrinsics,
                           stage1_manual_record, stale_manual_views)
from .ply_io import load_gaussian_ply
from .projection import project_visible
from .refusal import Refusal
from .units import scale_from_stage1, scene_units_cfg
from .vlm import ISOLATED_NOTE, VLMFormatError, get_backend

log = logging.getLogger(__name__)

# The hold policy and the verdict schema are INPUTS to the decision to run
# verification: both sit in the stage-4.5 fingerprint and in the
# verify_consent gate inputs, so a change demotes an existing consent.
HOLD_POLICY = "held_v1"     # HELD replaces CONFIRM/RELABEL* when reasons exist;
                            # REJECT and UNVERIFIED stand and keep the reasons
VERIFY_SCHEMA = 2

CHECKPOINT = "verdicts.partial.jsonl"   # the checkpoint
NOT_JUDGED = "NOT_JUDGED"               # not marked, not judged
FIELDNAMES = ["object_idx", "label", "verdict", "new_label", "rationale",
              "views", "source", "support_votes", "support_checked",
              "hold_reasons", "held_original_verdict", "held_proposed_label",
              "proj_iou", "proj_frame", "name_first", "choice", "kind_of",
              "isolated"]


def _row(v: dict) -> dict:
    """A verdict with every column of verdicts.csv (a row read back from
    an older CSV, or an auto verdict, may lack the newer ones)."""
    r = {k: v.get(k, "") for k in FIELDNAMES}
    r["object_idx"] = int(r["object_idx"])
    return r


def _not_judged(idx: int, o: dict, e0: dict) -> dict:
    return _row(dict(object_idx=idx, label=o["label"], verdict=NOT_JUDGED,
                     rationale="not marked for verification; the "
                               "operator's choice; the label ships as "
                               "detected, unjudged",
                     source="none",
                     hold_reasons=" ".join(e0.get("hold_reasons") or [])))


def requested_instances(workdir) -> list[int]:
    """Instance ids marked for verification on the current export,
    sorted; [] means every instance is judged."""
    ix_p = Path(workdir) / "stage4/interactions.json"
    if not ix_p.exists():
        return []
    ext = json.loads(ix_p.read_text()).get("extended", {}).get("objects", [])
    return sorted(int(e["instance_id"]) for e in ext
                  if e.get("verify_requested") and e.get("instance_id") is not None)


def _checkpoint_key(workdir, cfg: dict) -> dict:
    """What a checkpoint must have been written under to be resumed: this
    export, these manual views and exemplars, these parameters — the marks
    aside (a judged verdict serves any later run on the same export)."""
    p = {k: v for k, v in verify_params(cfg, workdir).items() if k != "requested"}
    return json.loads(json.dumps(dict(
        checkpoint=1,
        upstream_stage4=upstream_fingerprint(workdir, "stage4/manifest.json"),
        manual_views_hash=stage1_manual_record(workdir).get("hash", "absent"),
        exemplars_hash=stage2_exemplar_record(workdir).get("hash", "absent"),
        params=p)))


def read_checkpoint(workdir, cfg: dict, discard_stale: bool = False) -> dict[int, dict]:
    """The judged verdicts of a run that stopped, by object index, when the
    checkpoint was written under this export and these settings; {} else
    (the file is removed when `discard_stale`). A torn last line — the
    process died mid-write — is skipped."""
    path = Path(workdir) / "stage45" / CHECKPOINT
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    try:
        head = json.loads(lines[0]) if lines else None
    except ValueError:
        head = None
    if head != _checkpoint_key(workdir, cfg):
        if discard_stale:
            log.info("checkpoint %s is from another export or other "
                     "settings; discarded", path.name)
            path.unlink()
        return {}
    done: dict[int, dict] = {}
    for ln in lines[1:]:
        try:
            r = _row(json.loads(ln))
        except (ValueError, KeyError):
            continue
        done[r["object_idx"]] = r
    return done


def previous_verdicts(workdir) -> dict[int, dict]:
    """The verdicts of the previous verification of THIS export (its
    manifest names the stage-4 manifest now on disk), by object index of
    the pre-verification export; {} when there is none or it is another
    export's."""
    out = Path(workdir) / "stage45"
    mp, vp = out / "manifest.json", out / "verdicts.csv"
    if not (mp.exists() and vp.exists()):
        return {}
    kept = json.loads(mp.read_text())
    if kept.get("upstream_stage4") != upstream_fingerprint(
            workdir, "stage4/manifest.json"):
        return {}
    with open(vp, newline="") as f:
        return {int(r["object_idx"]): _row(r) for r in csv.DictReader(f)}


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _proj_box_one(pts: np.ndarray, cams: dict, fr: dict,
                  near: float) -> list[float] | None:
    """Robust 2D bounds of the instance's OWN gaussians in frame `fr`:
    every point in front of the near plane (scene units, the resolved
    `render.near_plane`) — deliberately NO occlusion test (`depth=None`),
    the crop frames the instance where it stands, occluded or not."""
    proj = project_visible(pts, fr, cams, None, near=near, min_points=20)
    if proj is None:
        return None
    uu, vv = proj.uf[proj.ok], proj.vf[proj.ok]
    return [float(np.percentile(uu, 3)), float(np.percentile(vv, 3)),
            float(np.percentile(uu, 97)), float(np.percentile(vv, 97))]


def _fold(s: str) -> list[str]:
    """Words of a label for comparison: lowercase, punctuation out, a
    trailing plural off words longer than three letters."""
    words = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]


def names_match(a: str, b: str) -> bool:
    """The name-first rule: equal after folding, the same head
    noun ("metal bowl" / "bowl"), or one a subset of the other's words
    ("bag of bread" / "bread"). A synonym the rule misses goes to the
    choice question, which is the safe side."""
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return False
    return fa == fb or fa[-1] == fb[-1] or set(fa) <= set(fb) or set(fb) <= set(fa)


def _isolated_crop(ply_path, fr: dict, cams: dict, cfg: dict,
                   pad_frac: float) -> Image.Image | None:
    """The instance's own Gaussians rendered alone through frame `fr`'s
    camera, cropped to where they land with the crop pad. None when
    too little lands in the image (behind the camera, or off it)."""
    import torch
    from .render import _render_batch
    scene = load_gaussian_ply(str(ply_path))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scene_t = dict(means=torch.as_tensor(scene.means_file, device=dev),
                   quats=torch.as_tensor(scene.quats_file, device=dev),
                   scales=torch.as_tensor(scene.scales, device=dev),
                   opacities=torch.as_tensor(scene.opacities, device=dev),
                   sh=torch.as_tensor(scene.sh, device=dev),
                   sh_degree=min(scene.sh_degree, cfg["render"]["sh_degree"]))
    fx, fy, cx, cy, w, h = frame_intrinsics(fr, cams)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    rgb, alpha, _, _ = _render_batch(
        scene_t, np.array(fr["c2w"], dtype=np.float32)[None], K, w, h, cfg)
    ys, xs = np.nonzero(alpha[0].numpy() > 0.3)
    if len(xs) < 20:
        return None
    box = [float(np.percentile(xs, 1)), float(np.percentile(ys, 1)),
           float(np.percentile(xs, 99)), float(np.percentile(ys, 99))]
    img = Image.fromarray((rgb[0].numpy() * 255).astype(np.uint8))
    return _crop(img, box, pad_frac)


def _judge(backend, crops: list, label: str, name_first: bool,
           extra: list | None = None, note: str = "") -> dict:
    """The verdict on an object's label. Name-first: the
    model names the crops with no label given; a match is CONFIRM; else
    both names on equal footing decide — A CONFIRM, B RELABEL, C CONFIRM
    with the name recorded as a proposal, D RELABEL the model's own name
    or REJECT when it sees no object. Off: the single labelled question
    (VERIFY_PROMPT), which confirmed 19 of 20 wrong labels on a test scene.
    Returns {verdict, label, rationale, name_first, choice}."""
    if not name_first:
        return dict(backend.verify_label(crops, label), name_first="", choice="")
    images = crops + (extra or [])
    nm = backend.name_object(images, note if extra else "")
    name = nm["name"]
    if name == "none" and extra:
        # a sparse object's own render can read as nothing (a 104-
        # Gaussian package rejected on a test scene). A REJECT is never
        # taken from the isolated image: ask again on the crops alone.
        images, note = crops, ""
        nm = backend.name_object(images)
        name = nm["name"]
    if name == "none":
        return dict(verdict="REJECT", label=label, name_first=name, choice="",
                    rationale=f"Asked with no label, the verifier saw no "
                              f"discrete object: {nm['reason']}")
    if names_match(name, label):
        return dict(verdict="CONFIRM", label=label, name_first=name, choice="",
                    rationale=f"Named it \"{name}\" unprompted: {nm['reason']}")
    ch = backend.choose_label(images, label, name, note)
    why = ch["rationale"]
    if ch["choice"] == "A":
        return dict(verdict="CONFIRM", label=label, name_first=name, choice="A",
                    rationale=f"Named it \"{name}\" unprompted, then chose "
                              f"the detected label over it: {why}")
    if ch["choice"] == "B":
        return dict(verdict="RELABEL", label=name, name_first=name, choice="B",
                    rationale=f"Named it \"{name}\" unprompted and kept it "
                              f"over the detected label: {why}")
    if ch["choice"] == "C":
        return dict(verdict="CONFIRM", label=label, name_first=name, choice="C",
                    rationale=f"Named it \"{name}\" unprompted; both names "
                              f"fit equally: {why}")
    other = ch["label"]
    if not other or other == "none":
        return dict(verdict="REJECT", label=label, name_first=name, choice="D",
                    rationale=f"Named it \"{name}\" unprompted, then found "
                              f"neither name fits and no discrete object: {why}")
    return dict(verdict="RELABEL", label=other, name_first=name, choice="D",
                rationale=f"Named it \"{name}\" unprompted, then found "
                          f"neither name fits and called it \"{other}\": {why}")


def _crop(img: Image.Image, box: list[float], pad_frac: float) -> Image.Image:
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    pw, ph = max(pw, 24), max(ph, 24)
    return img.crop((int(max(0, x1 - pw)), int(max(0, y1 - ph)),
                     int(min(img.width, x2 + pw)), int(min(img.height, y2 + ph))))


def _marked_crop(img: Image.Image, box: list[float],
                 pad_frac: float) -> Image.Image:
    """Padded crop with the instance's projected bounds drawn as a RED
    rectangle — the support check judges the marked object only, so an
    adjacent salient object cannot lend it its identity."""
    x1, y1, x2, y2 = box
    pw = max((x2 - x1) * pad_frac, 24)
    ph = max((y2 - y1) * pad_frac, 24)
    cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    im = img.crop((cx1, cy1, int(min(img.width, x2 + pw)),
                   int(min(img.height, y2 + ph))))
    ImageDraw.Draw(im).rectangle(
        [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
        outline=(255, 0, 0), width=3)
    return im


def _projection_boxes(pts: np.ndarray, cams: dict, frames_by_idx: dict,
                      near: float, k: int = 1) -> list[tuple[int, list[float]]]:
    """Projection crops from the instance's OWN gaussians: the top-k frames
    by in-image gaussian count, box = robust bounds of the projected
    centers (`_proj_box_one`: no occlusion test). k=1 is the
    zero-confidence fallback crop; k>1 feeds the relabel support check."""
    cand = []
    for fidx, fr in frames_by_idx.items():
        proj = project_visible(pts, fr, cams, None, near=near, min_points=20)
        if proj is None:
            continue
        uu, vv = proj.uf[proj.ok], proj.vf[proj.ok]
        cand.append((proj.n_ok, fidx,
                     [float(np.percentile(uu, 3)), float(np.percentile(vv, 3)),
                      float(np.percentile(uu, 97)), float(np.percentile(vv, 97))]))
    cand.sort(key=lambda t: -t[0])
    return [(fidx, box) for _, fidx, box in cand[:k]]


def verify_params(cfg: dict, workdir=None) -> dict:
    """Effective decision parameters of verification: the model and
    its sampling, the crop policy, the relabel bar, and the vocabulary the
    relabel constraint is checked against — and, given the workdir, the
    instances marked for verification: a changed set is a fresh
    run, never a replay."""
    V = cfg["vlm"]
    keys = ("model", "model_dir", "seed", "max_side",
            "max_new_tokens", "json_retries", "verify_views", "verify_pad",
            "verify_proj_iou_min", "verify_name_first",
            "relabel_specific_ok", "verify_isolated",
            "verify_isolated_max_gaussians", "relabel_support_check",
            "relabel_check_views", "relabel_min_agree_views",
            "relabel_support_pad", "relabel_vocab_only",
            "relabel_max_box_frac")
    p = {k: V.get(k) for k in keys}
    p["prompts"] = list(cfg["detect"].get("prompts") or [])
    p["negatives"] = list(cfg["detect"].get("negatives") or [])
    p["hold_policy"] = HOLD_POLICY
    p["verify_schema"] = VERIFY_SCHEMA
    if workdir is not None:
        p["requested"] = requested_instances(workdir)
    return p


def _carry_operator_labels(verified: dict, pre: dict) -> int:
    """A re-run judges the ORIGINAL export, but the operator's names
    were written on the verified one after the fact: carry them
    onto the pre-verification records by instance id (stable within one
    export; the verified file has the rejected objects removed, so the
    indices differ). Returns how many were carried."""
    keys = ("operator_label", "operator_label_at",
            "verify_requested", "verify_requested_at")   # the marks too
    names = {}
    for e in verified.get("extended", {}).get("objects", []):
        if e.get("instance_id") is not None and any(e.get(k) for k in keys):
            names[e["instance_id"]] = {k: e[k] for k in keys if k in e}
    n = 0
    for e in pre.get("extended", {}).get("objects", []):
        hit = names.get(e.get("instance_id"))
        if hit:
            e.update(hit)
            n += bool(hit.get("operator_label"))
    return n


def verify_cached(workdir: str, cfg: dict) -> dict | None:
    """The kept verification when it is still fresh — this export, this
    model, these parameters, the manual views and the exemplars as they
    were — else None. ONE decision for the run (which replays it) and the
    preflight (which says so before the button is pressed)."""
    out = Path(workdir) / "stage45"
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return None
    cached = json.loads(manifest_path.read_text())
    # fresh only against the stage-4 manifest now on disk and
    # these parameters — a re-export used to be followed by the
    # previous verification when the sequencer had not seen it change.
    if (not stale_manual_views(cached, workdir, "stage45")
            and not stale_exemplars(cached, workdir, "stage45")
            and not stale_upstream(cached, workdir, "stage45",
                                   "upstream_stage4",
                                   "stage4/manifest.json")
            and not stale_params(cached, verify_params(cfg, workdir), "stage45")):
        return cached
    return None


def run_verify(workdir: str, cfg: dict, force: bool = False,
               should_cancel=None) -> dict:
    from .sequencing import check_cancel
    out = Path(workdir) / "stage45"
    manifest_path = out / "manifest.json"
    params = verify_params(cfg, workdir)
    # The projection crops' near plane in scene units — the resolved
    # `render.near_plane` at the one factor stage 1 ran at, not a literal.
    s1_manifest_path = Path(workdir) / "stage1/manifest.json"
    near_units = scene_units_cfg(cfg, scale_from_stage1(
        cfg, json.loads(s1_manifest_path.read_text())
        if s1_manifest_path.exists() else None).value)[0]["render"]["near_plane"]

    # Manual-views staleness REFUSAL (not auto-invalidation: re-running verify
    # alone cannot fix a stale upstream): if manual_views.json changed after
    # this workdir's render, every stage output is stale — demand a re-run.
    rec = stage1_manual_record(workdir)
    if "path" in rec:
        live = fingerprint(rec["path"])
        if live != rec.get("hash", "absent"):
            raise Refusal(
                f"manual_views.json changed after this workdir's render "
                f"({rec.get('hash', 'absent')[:12]} -> {live[:12]} at "
                f"{rec['path']}); refusing to verify stale results; "
                f"re-render, then Run pipeline, then verify again",
                gate="render")

    # Exemplar staleness REFUSAL (same contract as manual views): a
    # changed exemplars.json means stage 2-4 outputs no longer reflect it.
    ex_rec = stage2_exemplar_record(workdir)
    if ex_rec.get("path"):
        live = fingerprint(ex_rec["path"])
        if live != ex_rec.get("hash", "absent"):
            raise Refusal(
                f"exemplars.json changed after this workdir's detect run "
                f"({ex_rec.get('hash', 'absent')[:12]} -> {live[:12]} at "
                f"{ex_rec['path']}); refusing to verify stale results; "
                f"re-run detect -> lift -> export first (detect re-runs "
                f"only the exemplar pass)")

    if not force:
        cached = verify_cached(workdir, cfg)
        if cached is not None:
            log.info("stage45 cached at %s (inputs unchanged)", out)
            # `cached` marks the replay for the sequencer's sentence
            #; the manifest on disk is untouched.
            return dict(cached, cached=True)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    stage1 = Path(workdir) / "stage1"
    stage4 = Path(workdir) / "stage4"
    inter_path = stage4 / "interactions.json"
    interactions = json.loads(inter_path.read_text())
    if "verification" in interactions.get("extended", {}):
        # idempotent re-verify: always judge the ORIGINAL export
        pre = out / "interactions.pre_verify.json"
        if not pre.exists():
            raise Refusal(
                "this export is already verified and its pre-verification "
                "copy is gone; run the pipeline first (it exports again), "
                "then verify again", gate="verify_consent")
        verified = interactions
        interactions = json.loads(pre.read_text())
        n = _carry_operator_labels(verified, interactions)
        log.info("re-verifying: restored pre-verification interactions"
                 + (f" ({n} operator name(s) carried over)" if n else ""))
    (out / "interactions.pre_verify.json").write_text(json.dumps(
        interactions, indent=1))
    cams = json.loads((stage1 / "cameras.json").read_text())
    frames_by_idx = {fr["frame_idx"]: fr for fr in cams["frames"]}
    vcfg = cfg["vlm"]
    backend = get_backend(cfg, should_cancel)

    objects = interactions["objects"]
    ext_objects = interactions["extended"]["objects"]
    verdicts, crop_dir = [], out / "crops"
    # the marked subset — judged; the rest carry their verdict from
    # the previous verification of this export, else ship NOT_JUDGED.
    requested = {i for i, e in enumerate(ext_objects) if e.get("verify_requested")}
    previous = previous_verdicts(workdir) if requested else {}
    if requested:
        log.info("verifying the %d marked instance(s) of %d; the other %d %s",
                 len(requested), len(objects), len(objects) - len(requested),
                 "carry their verdicts from the previous verification of "
                 "this export" if previous else
                 "ship NOT_JUDGED: not marked, the label kept as detected")
    # the checkpoint of a run that stopped — its verdicts stand when
    # the export and the settings are the same (the marks aside).
    done = read_checkpoint(workdir, cfg, discard_stale=True)
    if done:
        log.info("resuming: %d of %d judged before the stop; continuing "
                 "from there", len(done), len(objects))
    else:
        # a real run never shows the previous run's crops beside the
        # new ones — cleared HERE, after the cache check decided to run, so
        # a cached run keeps its crops (and a resumed one its own).
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
    crop_dir.mkdir(exist_ok=True)
    ck_path = out / CHECKPOINT
    if not done:
        ck_path.write_text(json.dumps(_checkpoint_key(workdir, cfg)) + "\n")
    ck = open(ck_path, "a")

    def keep(v: dict) -> None:
        """A judged verdict: into the run and onto the checkpoint at once."""
        v = _row(v)
        verdicts.append(v)
        ck.write(json.dumps(v) + "\n")
        ck.flush()
    frame_cache: dict[int, Image.Image] = {}
    n_judged = n_carried = 0

    def frame_img(fidx: int) -> Image.Image:
        if fidx not in frame_cache:
            frame_cache[fidx] = Image.open(
                stage1 / frames_by_idx[fidx]["file"]).convert("RGB")
        return frame_cache[fidx]

    # Relabel constraints (pipeline-generic).
    # Vocabulary = the scene's confirmed positives; free-text renames
    # re-imported the umbrella terms the vocabulary gate collapsed (both
    # exterior runs). Useful regression cases when changing this:
    # water-bottle/soda-can and donut/whiteboard.
    negs = set(cfg["detect"].get("negatives") or [])
    vocab = {p for p in (cfg["detect"].get("prompts") or []) if p not in negs}

    def frame_area(fidx: int) -> float:
        _, _, _, _, fw, fh = frame_intrinsics(frames_by_idx[fidx], cams)
        return float(fw * fh)

    try:
        for idx, o in enumerate(objects):
            e0 = ext_objects[idx] if idx < len(ext_objects) else {}
            if idx in done:                      # judged before the stop
                verdicts.append(done[idx])
                continue
            if requested and idx not in requested:
                n_carried += idx in previous
                verdicts.append(previous.get(idx) or _not_judged(idx, o, e0))
                continue
            check_cancel(should_cancel)   # between per-instance VLM calls
            n_judged += 1
            entries = o["frames"][:vcfg["verify_views"]]
            ply = sorted((stage4 / "instances").glob(f"{idx:03d}_*.ply"))
            pts = load_gaussian_ply(str(ply[0])).means if ply else None
            # §6: the association reasons written at export, plus
            # projection_mismatch when this stage takes that path.
            reasons = list(e0.get("hold_reasons") or [])
            proj_mismatch = None

            # A false positive's detection boxes frame whatever LOOKED like the
            # label, not the instance's own gaussians (desk-as-laptop: its best
            # box IS the real laptop). Verify the 3D INSTANCE: when its projected
            # bbox disagrees with the detection box, crop the projection instead.
            source, crops, used = "frames", [], []
            if entries and pts is not None:
                fr0 = frames_by_idx[entries[0]["frame_idx"]]
                pb0 = _proj_box_one(pts, cams, fr0, near_units)
                if pb0 is not None and \
                        _iou(pb0, entries[0]["box"]) < vcfg["verify_proj_iou_min"]:
                    source = "projection_mismatch"
                    proj_mismatch = dict(iou=round(_iou(pb0, entries[0]["box"]), 4),
                                         frame=int(entries[0]["frame_idx"]))
                    reasons.append("projection_mismatch")
                    entries = []
            box_fracs = []
            for en in entries:
                crops.append(_crop(frame_img(en["frame_idx"]), en["box"],
                                   vcfg["verify_pad"]))
                used.append(en["frame_idx"])
                b = en["box"]
                box_fracs.append((b[2] - b[0]) * (b[3] - b[1])
                                 / frame_area(en["frame_idx"]))
            if not crops and pts is not None:
                # projection-derived crop: zero-confidence instances (no matched
                # detection at all) and projection-mismatch cases above — these
                # are exactly the instances verification exists to judge.
                if source == "frames":
                    source = "projection"
                for fidx, box in _projection_boxes(pts, cams, frames_by_idx,
                                                   near_units):
                    crops.append(_crop(frame_img(fidx), box, vcfg["verify_pad"]))
                    used.append(fidx)
                    box_fracs.append((box[2] - box[0]) * (box[3] - box[1])
                                     / frame_area(fidx))
            if not crops:
                keep(dict(object_idx=idx, label=o["label"],
                                     verdict="REJECT", new_label="",
                                     rationale="no view evidence at all "
                                               "(auto, no LLM call)",
                                     views="", source="none",
                                     support_votes="", support_checked="",
                                     hold_reasons=" ".join(reasons),
                                     held_original_verdict="",
                                     held_proposed_label="",
                                     proj_iou="", proj_frame=""))
                continue
            for ci, cim in enumerate(crops):
                cim.save(crop_dir / f"{idx:03d}_{o['label'].replace(' ', '_')}"
                                    f"_{ci}.jpg", quality=90)
            support_votes, support_checked = "", ""
            name_first, choice, kind_of = "", "", ""
            # An operator option, off by default: under the
            # Gaussian threshold, the object's own render joins the crops
            # as one more image, said to be what it is — never alone,
            # never for a large object (absurd names there).
            isolated = ""
            # the export's key is `gaussian_count`; the ply's own length
            # when an older export lacks it
            n_g = int(e0.get("gaussian_count")
                      or (len(pts) if pts is not None else 0))
            if (vcfg.get("verify_isolated", False) and ply and used
                    and 0 < n_g <= int(vcfg.get("verify_isolated_max_gaussians", 300))):
                iso = _isolated_crop(ply[0], frames_by_idx[used[0]], cams, cfg,
                                     vcfg["verify_pad"])
                if iso is not None:
                    iso.save(crop_dir / f"{idx:03d}_isolated.jpg", quality=90)
                    isolated = "1"
            try:
                v = _judge(backend, crops, o["label"],
                           bool(vcfg.get("verify_name_first", True)),
                           extra=[iso] if isolated else None,
                           note=ISOLATED_NOTE)
                # kept apart: the constraint and hold branches below
                # rebuild `v` and would drop them from the audit row
                name_first, choice = v["name_first"], v["choice"]

                # Relabel constraints — applied BEFORE the support
                # check, so blocked proposals also skip its API calls:
                # (a) vocabulary constraint: the verifier may only relabel WITHIN
                #     the scene vocabulary; anything else keeps the detected label
                #     and is flagged as a REJECT-CANDIDATE for the human pass.
                # (b) large-footprint guard: crops of an instance whose bbox spans
                #     a large frame fraction show whatever stands INSIDE it (the
                #     11 m paved path relabeled 'tree trunk' by its own foreground
                #     trunks — and the support check INHERITED the defect, since
                #     every marked rectangle contained trunks too). Such instances
                #     are CONFIRM/REJECT only.
                if v["verdict"] == "RELABEL":
                    med_frac = float(np.median(box_fracs)) if box_fracs else 0.0
                    outside = bool(vcfg.get("relabel_vocab_only", True) and vocab
                                   and v["label"] not in vocab)
                    if outside and vcfg.get("relabel_specific_ok", False):
                        # An operator option, off by default: a name
                        # outside the vocabulary stands when the model says
                        # it is a KIND of the detected label — never a more
                        # general word, so the umbrella terms the vocabulary
                        # rule exists for cannot come back this way. The
                        # support check still follows.
                        kr = backend.kind_of(crops, o["label"], v["label"])
                        kind_of = kr["relation"]
                        if kind_of == "specific":
                            outside = False
                            v = dict(v, rationale=f"{v['rationale']}; "
                                     f"'{v['label']}' is outside the vocabulary "
                                     f"but a kind of '{o['label']}' "
                                     f"({kr['rationale']})")
                    if outside:
                        v = dict(verdict="RELABEL_OUTSIDE_VOCAB", label=v["label"],
                                 rationale=f"The verifier suggested "
                                           f"'{v['label']}', which is not in this "
                                           f"scene's vocabulary, so the detected "
                                           f"label was kept. Its reason: "
                                           f"{v['rationale']}")
                    elif med_frac > vcfg.get("relabel_max_box_frac", 0.5):
                        v = dict(verdict="RELABEL_TOOLARGE", label=v["label"],
                                 rationale=f"The verifier suggested "
                                           f"'{v['label']}', but the object fills "
                                           f"{med_frac:.0%} of its views, too much "
                                           f"to check a new name on, so the "
                                           f"detected label was kept. Its reason: "
                                           f"{v['rationale']}")

                # RELABEL support check (config-gated): a RELABEL
                # needs a HIGHER bar than a CONFIRM. The initial verdict names the
                # DOMINANT object in its crop, which is not necessarily the
                # instance (the regression case: whiteboard #14 relabeled "donut"
                # from one projection crop whose salient content was the donut on
                # the door knob below it). The proposed label must be re-confirmed
                # on crops of the instance's OWN gaussian projections in >=
                # relabel_min_agree_views DISTINCT views — a single-crop verdict
                # may CONFIRM but never RELABEL. Failures keep the detected label
                # (RELABEL_UNSUPPORTED).
                if (v["verdict"] == "RELABEL" and vcfg["relabel_support_check"]):
                    votes, checked = 0, 0
                    if pts is not None:
                        for ci, (fidx, box) in enumerate(_projection_boxes(
                                pts, cams, frames_by_idx, near_units,
                                vcfg["relabel_check_views"])):
                            sc = _marked_crop(frame_img(fidx), box,
                                              vcfg["relabel_support_pad"])
                            sc.save(crop_dir / f"{idx:03d}_support_{ci}.jpg",
                                    quality=90)
                            sv = backend.support_label(sc, v["label"])
                            checked += 1
                            if sv["verdict"] == "CONFIRM":
                                votes += 1
                    support_votes, support_checked = votes, checked
                    if votes < vcfg["relabel_min_agree_views"]:
                        v = dict(verdict="RELABEL_UNSUPPORTED", label=v["label"],
                                 rationale=f"The verifier suggested "
                                           f"'{v['label']}', but only {votes} of "
                                           f"{checked} views of this object agreed "
                                           f"(at least "
                                           f"{vcfg['relabel_min_agree_views']} "
                                           f"must), so the detected label was "
                                           f"kept. Its reason: {v['rationale']}")
            except VLMFormatError as e:
                # transient format noise must kill neither the stage nor the
                # label — the instance ships UNVERIFIED (kept, flagged in the
                # verdicts/report) whether the failure hit the initial verdict or
                # the relabel support check.
                v = dict(verdict="UNVERIFIED", label=o["label"],
                         rationale=f"VLM format failure after retries; label "
                                   f"kept, flagged: {e}")
            # §6 HELD — applied AFTER the call, replacing CONFIRM and every
            # RELABEL verdict: a supported relabel is evidence for a label,
            # not evidence that the instance is one object. REJECT and
            # UNVERIFIED stand and keep the reasons as metadata.
            held_orig, held_label = "", ""
            if reasons and v["verdict"] not in ("REJECT", "UNVERIFIED"):
                held_orig = v["verdict"]
                held_label = v["label"] if v["verdict"].startswith("RELABEL") else ""
                why = ", ".join(reasons)
                if proj_mismatch:
                    why += (f" (projection IoU {proj_mismatch['iou']} in "
                            f"frame {proj_mismatch['frame']})")
                v = dict(verdict="HELD", label=o["label"],
                         rationale=f"HELD [{why}]: verification could not "
                                   f"confirm this is one object; original "
                                   f"verdict {held_orig}"
                                   + (f" '{held_label}'" if held_label else "")
                                   + f": {v['rationale']}")
            keep(dict(object_idx=idx, label=o["label"],
                                 verdict=v["verdict"],
                                 new_label=v["label"]
                                 if v["verdict"].startswith("RELABEL") else "",
                                 rationale=v["rationale"],
                                 views=" ".join(map(str, used)), source=source,
                                 support_votes=support_votes,
                                 support_checked=support_checked,
                                 hold_reasons=" ".join(reasons),
                                 held_original_verdict=held_orig,
                                 held_proposed_label=held_label,
                                 proj_iou=proj_mismatch["iou"] if proj_mismatch else "",
                                 proj_frame=proj_mismatch["frame"] if proj_mismatch else "",
                                 # the unprompted name and the choice
                                 # that followed, so the run is auditable
                                 name_first=name_first, choice=choice,
                                 kind_of=kind_of, isolated=isolated))
            # k/N progress: minutes-long local-VLM stage; the log must show
            # liveness (and never suggest `conda run` for watching it — that
            # buffers all output until exit and healthy runs look hung).
            log.info("verify %d/%d %-18s -> %s%s (%s)", idx + 1, len(objects),
                     o["label"], v["verdict"],
                     f" '{v['label']}'" if v["verdict"].startswith("RELABEL")
                     else "", v["rationale"][:70])

    finally:
        # Release the weights the moment the last verdict is in: the web
        # server drives every stage in ONE process, so a model left
        # resident would hold its VRAM through the render and detect
        # stages of the next scene.
        backend.close()
        ck.close()

    # Zero instances is a STATE (export legitimately writes objects=[] when
    # everything dies at the thresholds), not a crash: write an honest empty
    # verdicts.csv with the standard header instead of indexing verdicts[0].
    fieldnames = FIELDNAMES
    # A re-run under another option (the model, name-first, specific-over-
    # generic) replaces the verdicts; the previous set is kept beside them
    # so the two can be compared without a copy made by hand.
    if (out / "verdicts.csv").exists():
        shutil.copyfile(out / "verdicts.csv", out / "verdicts.previous.csv")
    with open(out / "verdicts.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        wcsv.writerows(verdicts)
    ck_path.unlink(missing_ok=True)   # the run is whole: the checkpoint goes

    # Apply: REJECT removes (kept in rejected_instances.json). CONFIRM/RELABEL
    # keep the detected label canonical and record the verdict under extended
    # (KEEP BOTH policy — see module docstring).
    rejected, keep_map, new_objects, new_ext = [], {}, [], []
    for idx, (o, e, v) in enumerate(zip(objects, ext_objects, verdicts)):
        if v["verdict"] == "REJECT":
            rejected.append(dict(object=o, extended=e,
                                 rationale=v["rationale"]))
            continue
        # RELABEL_UNSUPPORTED keeps the detected label (the proposed label
        # failed the instance-support bar; it survives in the rationale).
        # HELD keeps the detected label too; the proposed label, the
        # original verdict and the support counts are preserved beside it.
        e = dict(e, verify_verdict=v["verdict"],
                 verify_rationale=v["rationale"],
                 hold_reasons=v["hold_reasons"].split())
        if v["verdict"] != NOT_JUDGED:    # unjudged claims nothing
            e["verified_label"] = (v["new_label"] if v["verdict"] == "RELABEL"
                                   else o["label"])
        # The name the verifier proposed, whatever became of it (applied,
        # outside the vocabulary, unsupported, too large, held) — a field,
        # not only a sentence, so the operator can take it.
        # ("both fit" keeps the label and records the unprompted
        # name as the proposal — a synonym is not a correction)
        proposed = (v["new_label"] or v["held_proposed_label"]
                    or (v["name_first"] if v["choice"] == "C" else ""))
        if proposed:
            e["proposed_label"] = proposed
        if v["verdict"] == "HELD":
            e.update(held_original_verdict=v["held_original_verdict"],
                     held_proposed_label=v["held_proposed_label"],
                     held_support=[v["support_votes"], v["support_checked"]])
        if v["proj_iou"] != "":
            e["projection_mismatch"] = dict(iou=v["proj_iou"],
                                            frame=v["proj_frame"])
        keep_map[idx] = len(new_objects)
        new_objects.append(o)
        new_ext.append(e)
    new_fa = {}
    for fidx, anns in interactions["frame_annotations"].items():
        kept = [dict(a, object_idx=keep_map[a["object_idx"]])
                for a in anns if a["object_idx"] in keep_map]
        if kept:
            new_fa[fidx] = kept
    interactions["objects"] = new_objects
    interactions["frame_annotations"] = new_fa
    interactions["extended"]["objects"] = new_ext
    interactions["extended"]["verification"] = dict(
        model=cfg["vlm"]["model"], model_dir=str(backend.model_dir),
        quantization=getattr(backend, "quant", None), rejected=len(rejected),
        relabeled=sum(1 for v in verdicts if v["verdict"] == "RELABEL"),
        relabels_unsupported=sum(1 for v in verdicts
                                 if v["verdict"] == "RELABEL_UNSUPPORTED"),
        relabels_outside_vocab=sum(1 for v in verdicts
                                   if v["verdict"] == "RELABEL_OUTSIDE_VOCAB"),
        relabels_too_large=sum(1 for v in verdicts
                               if v["verdict"] == "RELABEL_TOOLARGE"),
        unverified=sum(1 for v in verdicts if v["verdict"] == "UNVERIFIED"),
        confirmed=sum(1 for v in verdicts if v["verdict"] == "CONFIRM"),
        # what this run judged, resumed, carried, left
        not_judged=sum(1 for v in verdicts if v["verdict"] == NOT_JUDGED),
        judged=n_judged, resumed=len(done), carried=n_carried,
        requested=sorted(int(ext_objects[i]["instance_id"]) for i in requested
                         if ext_objects[i].get("instance_id") is not None),
        # §6: `held` counts FINAL HELD verdicts only, so the summary
        # reconciles with the exported count; the per-reason counts are
        # diagnostics and overlap verdicts by construction.
        held=sum(1 for v in verdicts if v["verdict"] == "HELD"),
        held_by_reason={r: sum(1 for v in verdicts
                               if r in v["hold_reasons"].split())
                        for r in ("ambiguous_det", "refused_merge",
                                  "ambiguous_unassigned",
                                  "projection_mismatch")},
        hold_policy=HOLD_POLICY, verify_schema=VERIFY_SCHEMA)
    inter_path.write_text(json.dumps(interactions, indent=1))
    (out / "rejected_instances.json").write_text(json.dumps(rejected, indent=1))

    manifest = dict(stage="verify", n_objects=len(objects),
                    manual_views_hash=rec.get("hash", "absent"),
                    exemplars_hash=ex_rec.get("hash", "absent"),
                    upstream_stage4=upstream_fingerprint(
                        workdir, "stage4/manifest.json"),
                    params=params,
                    **interactions["extended"]["verification"],
                    elapsed_s=round(time.perf_counter() - t0, 1))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log.info("stage45 done: %d confirmed / %d relabeled / %d rejected / "
             "%d HELD / %d UNVERIFIED%s -> %s (%.1fs)", manifest["confirmed"],
             manifest["relabeled"], manifest["rejected"], manifest["held"],
             manifest["unverified"],
             (f" / {manifest['not_judged']} NOT_JUDGED (not marked)"
              if manifest["not_judged"] else "")
             + (f"; {manifest['resumed']} resumed from the checkpoint"
                if manifest["resumed"] else "")
             + (f"; {manifest['carried']} carried from the previous run"
                if manifest["carried"] else ""),
             inter_path, manifest["elapsed_s"])
    return manifest
