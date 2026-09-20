# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared sequencing core of the gated pipeline.

Owns the gate rules and nothing else: the journal and its content-hash
cascade, gate-input hashing, profile writes, the stage force rules, the
factor-kill stop-and-look contract, and the report. Zero stage logic here
or in any front-end, by design.

RunCore is UI-free: everything an operator must see goes through say(),
everything an operator must decide goes through a named hook
(ack_stop_look, ask_verify_consent). The web backend implements those
against its event stream and the typed-ack endpoint.

The web backend's WebRunCore is the one implementation of these hooks.
The split between this UI-free core and the server is kept because it is
what keeps stage logic out of the server: a gate rule is written here
once, and the server can only call it.
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from . import config as _config
from .config import find_profile
from .refusal import Refusal

# The files the operator TEACHES the product about a scene — the region of
# interest and its density read and review plot, hand-captured views,
# exemplars, viewer settings — live in the WORKDIR, flat, beside
# run_journal.json. Before that they were
# written beside the capture under data/scenes, which coupled the product's
# state to the source file: a read-only capture could not be onboarded, a
# scratchpad symlink wrote into the real scene folder, and deleting a scene
# had to choose between the capture and the files learned about it. Now the
# capture's folder holds the capture and nothing else, Carveout never writes
# or deletes there, and deleting a scene is the profile and the workdir.
SCENE_FILES = ("volume.json", "density.json", "volume_review.png",
               "manual_views.json", "exemplars.json", "viewer_settings.json",
               "proposal.json", "probe")   # the Propose cards' analysis and
                                           # the scene read's probe renders


log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
GATES = ["volume", "render", "vocabulary", "exemplars", "verify_consent"]

def _deep_merge(dst: dict, src: dict) -> None:
    """Recursively merge src into dst (dicts merge; scalars/lists replace)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _leaf_paths(d: dict, prefix: tuple = ()):
    """Yield (path_tuple, value) for every non-dict leaf of a nested dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaf_paths(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _get_path(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def git_describe() -> str:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty",
                              "--tags"], cwd=REPO, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class StageCancelled(RuntimeError):
    """Raised inside a stage when the operator cancels. Stages check
    should_cancel() at loop boundaries and raise this BEFORE writing their
    manifest, so the manifest-written-last invariant leaves the stage cleanly
    re-runnable (no partial manifest, no cache false-hit)."""


def check_cancel(should_cancel) -> None:
    """Raise StageCancelled if the (optional) cancel probe fires. Cheap enough
    to call every loop iteration."""
    if should_cancel is not None and should_cancel():
        raise StageCancelled("stage cancelled by operator")


class WorkdirLocked(RuntimeError):
    """The workdir is held by another driver (single-writer rule)."""

    def __init__(self, rec: dict, stale: bool):
        self.rec = rec
        self.stale = stale
        state = ("no longer running" if stale else
                 f"pid {rec.get('pid')}, running")
        super().__init__(
            f"workdir held by {rec.get('holder')!r} since "
            f"{rec.get('since')} ({state})")


class WorkdirLock:
    """Single-writer rule: two app instances must never drive the same
    workdir concurrently — two writers corrupt the journal. The driving
    instance holds <workdir>/run_lock.json while a run is attached; another
    instance opens the scene read-only and every act that would write is
    refused, naming the holder. Stale-lock recovery (holder process gone)
    is an EXPLICIT operator act, never automatic."""

    def __init__(self, workdir, holder: str):
        self.path = Path(workdir) / "run_lock.json"
        self.holder = holder

    def read(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return dict(holder="unknown (corrupt lockfile)", pid=-1,
                        since="unknown")

    @staticmethod
    def alive(rec: dict) -> bool:
        import os
        pid = int(rec.get("pid") or -1)
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)   # localhost-only product: same-host pid probe
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def acquire(self) -> None:
        """Take the lock or raise WorkdirLocked (stale flag set when the
        holder process is gone — release stays the operator's call)."""
        import os
        rec = self.read()
        if rec is not None and int(rec.get("pid") or -1) != os.getpid():
            raise WorkdirLocked(rec, stale=not self.alive(rec))
        write_text_atomic(self.path, json.dumps(dict(
            holder=self.holder, pid=os.getpid(),
            since=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))

    def release(self) -> None:
        import os
        rec = self.read()
        if rec and int(rec.get("pid") or -1) == os.getpid():
            self.path.unlink(missing_ok=True)

    def break_stale(self) -> None:
        """Remove a dead holder's lock — call ONLY from an explicit
        operator act (the web release endpoint); refuses a
        live holder."""
        rec = self.read()
        if rec is None:
            return
        if self.alive(rec):
            raise WorkdirLocked(rec, stale=False)
        self.path.unlink(missing_ok=True)


def validate_vocabulary(prompts, negatives, *, allow_empty=False) -> None:
    """The vocabulary gate's two structural rules, defined once for every
    front-end: a confirmed vocabulary must have prompts, and every
    negative control must be one of them (a control that is never probed
    can calibrate nothing). A DRAFT may be empty (allow_empty) — a
    confirmed vocabulary may not."""
    if not prompts and not allow_empty:
        raise Refusal("no vocabulary: the vocabulary gate must confirm "
                      "detect.prompts", gate="vocabulary", status=400)
    unknown = sorted(set(negatives or []) - set(prompts or []))
    if unknown:
        raise Refusal(f"negatives not present in prompts: {unknown}",
                      gate="vocabulary", status=400)


def write_text_atomic(path, text: str) -> None:
    """Write a whole file so that no reader ever sees a torn one: the text
    goes to a temporary file beside the target, is flushed to disk, and
    takes the target's name in one rename. Every state file the product
    keeps (the journal, the profile, the volume, the views, the exemplars,
    the objects) is written through here — a plain write_text interrupted
    mid-way (Ctrl-C, a crash, the machine going down) left half a JSON
    file that the next start could not read."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fingerprint(path) -> str:
    """Content key for cache invalidation: sha256 hex, 'absent' if no file.

    The primitive the journal's content-hash cascade — and every stage
    manifest's cache key — is built on, so it lives with the journal."""
    p = Path(path)
    return (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists()
            else "absent")


# -- each stage's cache carries its own fresh/stale relation -----------------
# Before the fingerprint record every stage's cache check compared only the manual-views and
# exemplar records, so a re-run lift followed by an export served the old
# export over the new lift, and a pipeline interrupted between
# two stages did the same on its next run (the sequencer forces a downstream
# stage only from a fingerprint change it observed in the SAME invocation).
# Each stage manifest now records (a) the fingerprint of the upstream
# manifest it consumed and (b) its effective decision parameters, and its
# cache check compares both by VALUE. A manifest missing either is stale and
# the log says why (the lesson: a cache blind to a knob hits stale).

def upstream_fingerprint(workdir, upstream: str) -> str:
    """Fingerprint of an upstream manifest as it is NOW, e.g.
    upstream_fingerprint(wd, "stage3/manifest.json")."""
    return fingerprint(Path(workdir) / upstream)


def stale_upstream(cached: dict, workdir, stage: str, key: str,
                   upstream: str) -> bool:
    """True (with one unmissable log line) when this stage's cached manifest
    was produced from a different upstream manifest than the one on disk —
    or predates the record (`key` absent)."""
    cur = upstream_fingerprint(workdir, upstream)
    rec = cached.get(key)
    if rec is None:
        log.warning("%s cache INVALIDATED: %s absent from manifest (written "
                    "before the upstream record existed); re-running %s",
                    stage, key, stage)
        return True
    if rec != cur:
        log.warning("%s cache INVALIDATED: upstream %s changed (this stage "
                    "consumed %s, it is now %s); re-running %s",
                    stage, upstream, rec[:12], cur[:12], stage)
        return True
    return False


def stale_params(cached: dict, live: dict, stage: str) -> bool:
    """True (with one unmissable log line) when the stage's effective
    decision parameters differ BY VALUE from the ones the cached manifest
    was produced with. A manifest without the `params` block, or missing
    any live key, is stale — a cache blind to a knob hits stale."""
    live = json.loads(json.dumps(live))   # what the manifest would hold
    rec = cached.get("params")
    if not isinstance(rec, dict):
        log.warning("%s cache INVALIDATED: decision parameters absent from "
                    "manifest; re-running %s", stage, stage)
        return True
    missing = sorted(k for k in live if k not in rec)
    if missing:
        log.warning("%s cache INVALIDATED: parameter keys absent from "
                    "manifest: %s; re-running %s", stage, missing, stage)
        return True
    changed = {k: (rec[k], live[k]) for k in live if rec[k] != live[k]}
    if changed:
        log.warning("%s cache INVALIDATED: decision parameters changed: %s"
                    "; re-running %s", stage,
                    "; ".join(f"{k}: {a!r} -> {b!r}"
                              for k, (a, b) in sorted(changed.items())),
                    stage)
        return True
    return False


class Journal:
    """run_journal.json: gate approvals keyed by content hashes."""

    def __init__(self, workdir: Path):
        self.path = workdir / "run_journal.json"
        if not self.path.exists():
            self.data = dict(version=1, gates={})
            return
        try:
            self.data = json.loads(self.path.read_text())
        except json.JSONDecodeError as e:
            # Named, not a traceback: one unreadable journal used to 500
            # the whole library. The remedy is the operator's — nothing
            # here guesses which approvals the file held.
            raise Refusal(
                f"{self.path} is not valid JSON ({e}); the scene's gate "
                f"record cannot be read. Restore the file from a backup, "
                f"or delete it: the scene then starts with every gate "
                f"unapproved (renders, masks and objects on disk are kept "
                f"and their caches still apply).", status=500) from e
        if not isinstance(self.data, dict) or "gates" not in self.data:
            raise Refusal(
                f"{self.path} is not a gate record (no 'gates' block); "
                f"restore it from a backup, or delete it to start the "
                f"scene's gates over.", status=500)

    def save(self) -> None:
        write_text_atomic(self.path, json.dumps(self.data, indent=1))

    def record(self, gate: str, inputs: dict, **extra) -> None:
        """Approve `gate`; the backtrack cascade clears every later gate."""
        for g in GATES[GATES.index(gate) + 1:]:
            self.data["gates"].pop(g, None)
        self.data["gates"][gate] = dict(
            approved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            inputs=inputs, **extra)
        self.save()

    def demote(self, gate: str) -> None:
        """Drop `gate` and everything after it (inputs changed / re-opened)."""
        for g in GATES[GATES.index(gate):]:
            self.data["gates"].pop(g, None)
        self.save()

    def restamp_scene_yaml(self, new_hash: str) -> None:
        """A DRIVER-verified profile write (scaffold marker, confirmed
        vocabulary, exemplar thresholds) must not read as a hand-edit at the
        earlier gates — update their recorded scene-yaml hash in place."""
        for rec in self.data["gates"].values():
            rec["inputs"]["scene_yaml"] = new_hash
        self.save()


class RunCore:
    """Journal-backed sequencing state: the gate rules, behind the web
    backend."""

    def __init__(self, scene: str, workdir: str, scene_config: str | None,
                 cfg_loader):
        self.scene = Path(scene)
        self.workdir = Path(workdir)
        # Profile resolution: by NAME, in the one profile directory; a
        # scene with no profile yet scaffolds one there.
        self.name = scene_config or self.scene.parent.name.lower()
        self.profile_path = (find_profile(self.name)
                             or _config.PROFILES_DIR / f"{self.name}.yaml")
        # read through the module, never bound at import: `carveout web
        # --profiles-dir` rebinds it after this module is loaded, and a
        # copy taken at import would scaffold a new scene's profile into
        # the real directory
        self.cfg_loader = cfg_loader
        self.journal = Journal(self.workdir)
        self.entered: set[str] = set()    # gates NOT skipped via the journal
        self.export_changed = False
        self._cfg = None

    # -- front-end hooks --------------------------------------------------------
    def say(self, msg: str) -> None:
        """Operator-visible line (the web backend: the run event stream)."""
        raise NotImplementedError

    def ack_stop_look(self, kills: list[dict]) -> None:
        """Factor-cap kills (never-event): block until an explicit
        typed acknowledgment; raising aborts the run instead."""
        raise NotImplementedError

    def ask_verify_consent(self, n_objects: int, model: str,
                           available: bool) -> bool:
        """Explicit stage-4.5 RUN/SKIP choice: verification is optional and
        costs minutes of GPU time. It used to also be a privacy question —
        4.5 sent crops to an API — which is why the gate and this method are
        still named for consent; the model is local now, nothing leaves
        the machine, and what the operator is answering is whether to spend
        the time. The journalled field says `run`, which is what it means."""
        raise NotImplementedError

    def should_cancel(self) -> bool:
        """Cooperative-cancel probe: stages poll this at loop
        boundaries. Default False; the web backend overrides it against a
        per-session cancel Event."""
        return False

    # -- config ---------------------------------------------------------------
    def cfg(self) -> dict:
        if self._cfg is None:
            # the resolved path, not the name: a name would re-resolve
            # through the search path and could land on a different file
            # than the one this run's writes and gate hashes are bound to
            self._cfg = self.cfg_loader(str(self.profile_path))
        return self._cfg

    def _invalidate_cfg(self) -> None:
        self._cfg = None

    # -- journal plumbing -------------------------------------------------------
    def _gate_inputs(self, gate: str) -> dict:
        """Live content hashes a gate approval is pinned to. scene yaml +
        volume.json at EVERY gate: a hand-edit between
        sessions demotes the journal from the earliest affected gate."""
        base = dict(scene_yaml=fingerprint(self.profile_path),
                    volume=fingerprint(self.workdir / "volume.json"))
        s1 = fingerprint(self.workdir / "stage1/manifest.json")
        if gate == "volume":
            return base
        if gate == "render":
            from .manual_views import resolve_manual_views_path
            mv = resolve_manual_views_path(self.workdir, self.cfg())
            return dict(base, manual_views=fingerprint(mv), stage1=s1)
        if gate == "vocabulary":
            d = self.cfg()["detect"]
            return dict(base, stage1=s1, vocab=json.dumps(
                [d.get("prompts") or [], d.get("negatives") or []]))
        if gate == "exemplars":
            from .exemplars import resolve_exemplars_path
            ex = resolve_exemplars_path(self.workdir, self.cfg())
            return dict(base, stage1=s1, exemplars=fingerprint(ex))
        if gate == "verify_consent":
            # the hold policy and verdict schema are inputs to the decision
            # to run verification (§6): a change demotes an existing consent
            from .verify import HOLD_POLICY, VERIFY_SCHEMA
            return dict(base,
                        stage4=fingerprint(self.workdir / "stage4/manifest.json"),
                        hold_policy=HOLD_POLICY,
                        verify_schema=str(VERIFY_SCHEMA))
        raise ValueError(gate)

    # -- the gate acts (the public gate-state interface) --------------------------
    # These four operations plus gate_states() are everything a front-end
    # may do to gate state. The journal, the input hashing and the entered
    # bookkeeping are implementation — a front-end reaching past them can
    # manufacture a journal state the sequencing rules do not recognize.

    def approve(self, gate: str, **extra) -> None:
        """The approve act: journal the gate against its live input hashes
        (the backtrack cascade clears every later gate) and mark it
        entered for the stage force rules."""
        self.entered.add(gate)
        self.journal.record(gate, self._gate_inputs(gate), **extra)
        self.say(f"== gate {gate!r}: APPROVED (journaled)")

    def reopen(self, gate: str) -> list[str]:
        """The reopen act: demote `gate` and everything after it, mark it
        entered. Returns the gates that actually lost an approval."""
        demoted = [g for g in GATES[GATES.index(gate):]
                   if g in self.journal.data["gates"]]
        self.journal.demote(gate)
        self.entered.add(gate)
        return demoted

    def mark_entered(self, gate: str) -> None:
        """Record that the operator (re-)entered a gate's territory — the
        stage force rules key off this (the stage caches are blind to
        vocabulary and volume content)."""
        self.entered.add(gate)

    def gate_states(self) -> dict:
        """Per-gate journal state with staleness recomputed from live
        content hashes — run_journal.json semantics ARE the gate-state
        model; outside edits surface here as stale gates. None = never
        approved."""
        gates: dict = {}
        for g in GATES:
            rec = self.journal.data["gates"].get(g)
            if not rec:
                gates[g] = None
                continue
            try:
                live = self._gate_inputs(g)
                fresh = rec["inputs"] == live
                changed = ([] if fresh else sorted(
                    k for k in live if rec["inputs"].get(k) != live[k]))
            except Exception as e:   # missing profile/artifacts mid-setup
                fresh, changed = False, [f"unreadable: {e}"]
            # `run` is current; `consent` is what journals written before
            # the local-VLM swap carry. Reading both keeps recorded
            # approvals valid — dropping the legacy key would silently
            # re-read every existing journal as SKIP.
            gates[g] = dict(approved_at=rec["approved_at"],
                            state="approved" if fresh else "stale",
                            changed=changed,
                            run=rec.get("run", rec.get("consent")),
                            neg_ceiling_ack=rec.get("neg_ceiling_ack"))
        return gates

    # -- calibration ownership ---------------------------------------------------
    # Adjustable keys -> the earliest gate a change re-opens. Whitelisted so
    # the web endpoint can only touch structural/high-value calibration,
    # never inject arbitrary profile content. Lives beside _gate_inputs on
    # purpose: what a gate's approval covers and which knob re-opens it are
    # one concept — split across packages they drift.
    # The quality-filter thresholds are DERIVED from the discard reasons
    # (quality.GATE_QUALITY_KEYS), not listed here: the gate that shows a
    # refusal must be able to change every threshold that refusal names, and
    # two hand-maintained lists is exactly how that stops being true.
    @staticmethod
    def calibration_paths() -> dict:
        from .quality import GATE_QUALITY_KEYS
        return {
            ("scene", "floor"): "volume",
            ("scene", "up_axis"): "volume",
            ("scene", "floor_band_frac"): "volume",
            ("scene", "scale_m_per_unit"): "volume",
            # the ruler's record beside the factor it derived (a dict: the
            # whitelist matches it as a prefix; null clears it when a
            # factor is declared or typed over a measurement)
            ("scene", "measured"): "volume",
            # the levelling: a record like `measured` — the angles,
            # their source and the clicks; null = the file's axes
            ("scene", "alignment"): "volume",
            ("render", "path_mode"): "render",
            ("render", "interior", "cameras_in_volume"): "render",
            ("render", "interior", "focus_aim"): "render",
            ("render", "diversity", "min_pos_sep"): "render",
            # the view budget and the coverage repair it no longer caps
            #: the render gate's own levers against a thin render
            ("render", "num_views"): "render",
            ("render", "candidate_factor"): "render",
            ("render", "coverage", "target_frac"): "render",
            ("render", "coverage", "max_extra_rounds"): "render",
            ("render", "coverage", "round_views"): "render",
            ("lift", "instance_voxel"): "verify_consent",
            # the scene's vision model: its output is the
            # verification, whose decision parameters already carry the
            # directory; the proposal is a draft and the volume read
            # a check, so nothing earlier re-opens
            ("vlm", "model_dir"): "verify_consent",
            # the specific-over-generic option, a verification
            # decision parameter like the model
            ("vlm", "relabel_specific_ok"): "verify_consent",
            # the small-object render option, likewise
            ("vlm", "verify_isolated"): "verify_consent",
            # the instancing mode and its thresholds re-run the lift;
            # with the fingerprint record the export and the stage-4 fingerprint follow, and
            # the verify_consent gate is the one pinned to it
            ("lift", "instancing"): "verify_consent",
            ("lift", "track_affinity_min"): "verify_consent",
            ("lift", "track_min_support_ios"): "verify_consent",
            ("lift", "cannot_link_ios_max"): "verify_consent",
            ("lift", "support_min_frac"): "verify_consent",
            ("export", "hold_unassigned_frac"): "verify_consent",
            # a per-concept exemplar threshold changes which crops count:
            # the exemplars review re-opens (it used to restamp every
            # gate and re-open none)
            ("detect", "exemplar_threshold_overrides"): "exemplars",
            **{("render", "quality_filter", k): "render"
               for k in GATE_QUALITY_KEYS},
        }

    def calibration_gate(self, update: dict) -> str:
        """Validate a calibration update against the whitelist and name the
        earliest gate it re-opens; Refusal on a non-adjustable key."""
        leaves = list(_leaf_paths(update or {}))
        if not leaves:
            raise Refusal("no calibration keys supplied", status=400)
        paths = self.calibration_paths()
        gates = []
        for path, _ in leaves:
            # A whitelisted key may hold a record (scene.measured): its
            # leaves match through the longest whitelisted prefix.
            hit = next((path[:i] for i in range(len(path), 0, -1)
                        if path[:i] in paths), None)
            if hit is None:
                raise Refusal(f"not an adjustable calibration key: "
                              f"{'.'.join(path)}", status=400)
            gates.append(paths[hit])
        return min(gates, key=GATES.index)   # earliest re-opened gate wins

    # -- profile writes ---------------------------------------------------------
    # Every profile write goes through _write_profile. A profile Carveout
    # did not create is ADOPTED on its first write: the marker is added and
    # the file is regenerated from its values (comments do not survive; the
    # original is kept beside it, once). The paste-verify contract that used
    # to guard hand-authored profiles is retired: gate safety lives in
    # the journal's content-hash cascade, which demotes gates on ANY change
    # to the file, so who typed it never mattered.
    def _adopt(self, raw: dict) -> None:
        if raw.get("driver_scaffolded") or not self.profile_path.exists():
            return
        keep = self.profile_path.with_suffix(self.profile_path.suffix + ".orig")
        if not keep.exists():
            keep.write_text(self.profile_path.read_text())
        raw["driver_scaffolded"] = True
        self.say(f"profile {self.profile_path} was not created by Carveout; "
                 f"adopting it: driver_scaffolded added, the file is "
                 f"regenerated from its values (comments do not survive), "
                 f"the original is kept at {keep}")
        log.warning("profile adopted: %s (original kept at %s)",
                    self.profile_path, keep)

    def _write_profile(self, prof: dict) -> None:
        header = (f"# {self.name} scene profile, scaffolded by Carveout.\n"
                  f"# driver_scaffolded: gate confirmations are auto-written "
                  f"into this file.\n")
        write_text_atomic(
            self.profile_path,
            header + yaml.safe_dump(prof, sort_keys=False,
                                    default_flow_style=None))
        self._invalidate_cfg()
        self.journal.restamp_scene_yaml(fingerprint(self.profile_path))

    def write_calibration(self, update: dict, affected_gate: str,
                          what: str) -> None:
        """Operator calibration write: merge `update` — a nested partial
        profile, e.g. {"render": {"path_mode": "interior"}} — into the scene
        profile, then RE-OPEN the affected gate and everything after it.

        Journal cascade, in this order: demote(affected_gate) clears the
        affected gate + its successors and saves; then _write_profile
        writes the file and restamps the recorded scene_yaml on the gates
        that remain — so only the EARLIER gates keep a (restamped)
        approval. That is honest: a render-calibration change never
        touched what the volume gate reviewed, exactly as a
        confirmed-vocabulary write leaves volume/render untouched. The
        order matters: an interrupt between the two saves leaves either
        a demoted journal over the old file, or the earlier gates holding
        the OLD hash over the new file (they then read stale) — never an
        approval the cascade should have taken. Written the other way
        round (restamp first, then demote) the same interrupt left every
        gate approved under the new hash. A profile Carveout did not
        create is adopted first (see _adopt)."""
        raw = (yaml.safe_load(self.profile_path.read_text()) or {}
               if self.profile_path.exists() else {})
        self._adopt(raw)
        # A record is replaced whole, never merged: a levelling adopted
        # from the proposal must not keep the clicks of the one it replaces.
        for rec in (("scene", "alignment"), ("scene", "measured"),
                    ("detect", "exemplar_threshold_overrides")):
            if isinstance((update.get(rec[0]) or {}), dict) and rec[1] in (update.get(rec[0]) or {}):
                (raw.get(rec[0]) or {}).pop(rec[1], None)
        _deep_merge(raw, update)
        self.journal.demote(affected_gate)   # FIRST: gate + successors re-open
        self._write_profile(raw)   # then the file, restamping what remains
        self.say(f"{what} written to {self.profile_path}")
        self.say(f"== calibration ({what}): gate {affected_gate!r} onward "
                 f"re-opened (earlier approvals kept)")

    def write_detect_update(self, update: dict, what: str) -> None:
        """Set keys under detect: (dict-valued keys are REPLACED, not merged:
        a threshold-override write that drops an old override must land as
        written). A profile Carveout did not create is adopted first."""
        raw = (yaml.safe_load(self.profile_path.read_text()) or {}
               if self.profile_path.exists() else {})
        self._adopt(raw)
        det = raw.setdefault("detect", {})
        det.update(update)
        self._write_profile(raw)
        self.say(f"{what} written to {self.profile_path}")

    # -- scaffold ---------------------------------------------------------------
    def scaffold(self, extra: dict | None = None) -> None:
        """Create the scene profile from the defaults. `extra` is the
        creation dialog's answers: `scene.scale_m_per_unit` (1.0 for a
        metric capture, the number when the operator knows it, null when
        they will measure one thing on the canvas — the null is written on
        purpose: "not yet" is the recorded answer, not a missing one) and
        `render.path_mode` (how the scene was filmed: interior / orbit /
        ground, always asked) — the only calibration a new profile
        carries, because the operator gave it."""
        def relpath(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(REPO))
            except ValueError:
                return str(p)
        if self.profile_path.exists():
            raw = yaml.safe_load(self.profile_path.read_text()) or {}
            for key, val in (("scene_path", relpath(self.scene)),
                             ("workdir", relpath(self.workdir))):
                if raw.get(key) and raw[key] != val:
                    log.warning("profile %s carries %s=%r but this run says "
                                "%r; the run's value is used",
                                self.profile_path.name, key, raw[key], val)
            return
        # A new profile starts from the DEFAULTS, never from another scene's
        # calibration. Copying a reference profile by hand is the documented
        # way to reuse one (USAGE, "Reusing a scene's calibration") — it is a
        # file copy plus deleting `detect.prompts`/`negatives`, and doing it
        # by hand means the operator sees exactly which numbers they adopted
        # instead of inheriting them invisibly.
        prof: dict = {}
        prof["driver_scaffolded"] = True
        prof["scene_path"] = relpath(self.scene)
        prof["workdir"] = relpath(self.workdir)
        if extra:
            _deep_merge(prof, extra)
        self._write_profile(prof)
        self.say(f"scene profile scaffolded -> {self.profile_path}")

    # -- stage-2 cache classifier -------------------------------------------------
    def _detect_state(self, prompts: list[str], negatives: list[str]) -> str:
        """'force' (prompts/frames mismatch — the stage2 cache is blind to
        both), 'fold' (only exemplars.json changed — detect re-runs just the
        exemplar pass on its own), or 'current'."""
        mp = self.workdir / "stage2/probe_manifest.json"
        if not mp.exists():
            return "force"
        mm = json.loads(mp.read_text())
        cams = self.workdir / "stage1/cameras.json"
        n_frames = (len(json.loads(cams.read_text())["frames"])
                    if cams.exists() else -1)
        if (set(mm.get("prompts") or []) != set(prompts)
                or set(mm.get("negatives") or []) != set(negatives)
                or len(mm.get("frames") or []) != n_frames):
            return "force"
        from .exemplars import resolve_exemplars_path
        ex = resolve_exemplars_path(self.workdir, self.cfg())
        if mm.get("exemplars_hash", "absent") != fingerprint(ex):
            return "fold"
        return "current"

    # -- gate: vocabulary (shared confirm tail) -------------------------------------
    def negative_ceiling(self) -> dict | None:
        """Negative-control health of the current probe; None = no probe.
        The limit is the SCENE's presence_threshold (generality principle —
        never a constant)."""
        import csv
        path = self.workdir / "stage2/score_summary.csv"
        if not path.exists():
            return None
        limit = float(self.cfg()["detect"]["presence_threshold"])
        neg_max, outlier = 0.0, None
        with open(path) as f:
            for r in csv.DictReader(f):
                if (r["negative_control"] == "True"
                        and float(r["max_score"]) >= neg_max):
                    neg_max = float(r["max_score"])
                    outlier = r["concept"]
        frame = None
        if outlier is not None:
            sp = self.workdir / "stage2/scores.csv"
            best = -1.0
            if sp.exists():
                with open(sp) as f:
                    for r in csv.DictReader(f):
                        if (r["concept"] == outlier
                                and float(r["score"]) > best):
                            best = float(r["score"])
                            frame = int(r["frame_idx"])
        return dict(value=neg_max, limit=limit, violated=neg_max >= limit,
                    outlier_concept=outlier, outlier_frame=frame)

    def confirm_vocabulary(self, prompts: list[str],
                           negatives: list[str]) -> None:
        """The confirm act of the vocabulary gate: negative-ceiling
        warning when VIOLATED (recorded on the
        approval), profile write, full-pass finalize, journal approval.

        A VIOLATED ceiling is REPORTED, not blocking. It used to demand a
        typed acknowledgment; that was dropped because negative controls
        cannot reach the output — `lift` drops every detection whose concept
        is a negative before anything is lifted, so no label, export or
        viewer scene can carry one. What a violation actually says is that
        the control is not absent from the scene (the incident that built
        the ack: '3d printer' at 0.613 turned out to be a real machine on a
        back shelf), or that the threshold is low for this capture. That is
        a calibration observation, and a typed confirmation that fires on
        ~40% of runs stops being read — which is the failure mode a typed
        confirmation exists to prevent. It stays loud and it stays on the
        journal record; the ONE typed ack left is the area-factor kill,
        which does reach the output."""
        from .detect import run_detect
        validate_vocabulary(prompts, negatives)
        nc = self.negative_ceiling()
        extra = {}
        if nc and nc["violated"]:
            where = ("" if nc["outlier_frame"] is None
                     else f" (frame {nc['outlier_frame']:04d})")
            self.say(
                f"\nNEGATIVE-CONTROL CEILING VIOLATED: {nc['value']:.3f} >= "
                f"presence_threshold {nc['limit']}; outlier "
                f"{nc['outlier_concept']!r}{where}.\n"
                f"Negatives never export, so this cannot corrupt the run: it "
                f"means the control is not absent from the scene, or the "
                f"threshold is low for this capture. Crop-check it in "
                f"{self.workdir / 'stage2/overlays'} when the labels look "
                f"off. Recorded on this approval.")
            extra["neg_ceiling_ack"] = dict(
                value=nc["value"], limit=nc["limit"],
                outlier=nc["outlier_concept"])
        self.write_detect_update(
            dict(prompts=prompts, negatives=negatives), "vocabulary")
        # finalize: the confirmed vocabulary must exist as a stage-2 run
        # over exactly these lists
        state = self._detect_state(prompts, negatives)
        if state != "current":
            self.say(f"running the full stage-2 pass on the confirmed "
                     f"vocabulary ({state})...")
            run_detect(str(self.workdir), self.cfg(), prompts,
                       negatives=negatives, force=(state == "force"),
                       should_cancel=self.should_cancel)
        self.approve("vocabulary", **extra)

    # -- stages: detect / lift / export -------------------------------------------
    def factor_kills(self) -> list[dict]:
        """Area-factor cap kills from the lift audit (never-event)."""
        audit = json.loads(
            (self.workdir / "stage3/instance_audit.json").read_text())
        return ((audit.get("_mask_suppression", {}).get("factor_cap") or
                 {}).get("killed") or [])

    def _factor_kill_check(self) -> None:
        kills = self.factor_kills()
        if kills:
            self.ack_stop_look(kills)

    def stage_pipeline(self) -> None:
        from .detect import run_detect
        from .export import run_export
        from .lift import run_lift
        cfg = self.cfg()
        prompts = list(cfg["detect"].get("prompts") or [])
        negatives = list(cfg["detect"].get("negatives") or [])
        validate_vocabulary(prompts, negatives)
        # The stage caches are blind to vocabulary and volume content; the
        # journal knows which gates were (re)entered and forces accordingly.
        force_all = bool({"volume", "render"} & self.entered)
        self.say("\n== stages: detect -> lift -> export (unattended)")
        s2 = self.workdir / "stage2/probe_manifest.json"
        before = fingerprint(s2)
        state = self._detect_state(prompts, negatives)
        # detect decides from content — its manifest names the
        # views (the stage-1 manifest), the volume, the manual views and
        # the exemplars it was run over — not from which gates this
        # session entered: the vocabulary probe run after the render
        # approval stands, and the first pipeline press no longer pays
        # it twice (5 min on a 73-concept, 40-view scene). The lift and
        # the export keep the session rule: they read the volume directly
        # and their caches do not name it.
        run_detect(str(self.workdir), cfg, prompts, negatives=negatives,
                   force=state == "force",
                   should_cancel=self.should_cancel)
        det_changed = fingerprint(s2) != before

        s3 = self.workdir / "stage3/manifest.json"
        before = fingerprint(s3)
        run_lift(str(self.scene), str(self.workdir), cfg,
                 force=force_all or det_changed
                 or "vocabulary" in self.entered,
                 should_cancel=self.should_cancel)
        lift_changed = fingerprint(s3) != before
        self._factor_kill_check()

        s4 = self.workdir / "stage4/manifest.json"
        before = fingerprint(s4)
        run_export(str(self.scene), str(self.workdir), cfg,
                   force=force_all or lift_changed
                   or "vocabulary" in self.entered,
                   should_cancel=self.should_cancel)
        self.export_changed = fingerprint(s4) != before
        # The pipeline CONSUMED these re-entry signals — everything they
        # force just re-ran to completion. Left in place they were a latch:
        # one volume reopen forced detect/lift/export on every later
        # pipeline run for the server's whole life, cache never honored
        # again. A failed run keeps them (this line is
        # only reached on success), so the force survives a retry.
        self.entered.difference_update({"volume", "render", "vocabulary"})

    # -- gate: verify consent + stage 4.5 ------------------------------------------
    def gate_verify(self) -> None:
        n = len(json.loads(
            (self.workdir / "stage4/interactions.json").read_text())["objects"])
        # ONE freshness computation for the whole product: gate_states()
        # (it already reads the legacy `consent` key alongside `run` —
        # dropping that would silently re-read every existing journal as
        # SKIP and ship those scenes unverified).
        st = self.gate_states().get("verify_consent")
        if st and st["state"] == "approved":
            consent = bool(st["run"])
            self.say(f"\n== gate 'verify_consent': recorded "
                     f"{'YES' if consent else 'NO'} ({st['approved_at']}), "
                     f"inputs unchanged")
        else:
            if st:
                self.journal.demote("verify_consent")
            self.entered.add("verify_consent")
            cfg = self.cfg()
            from .vlm import backend_available
            available, why = backend_available(cfg)
            from .verify import requested_instances, previous_verdicts
            marked = len(requested_instances(self.workdir))
            self.say(f"\n== gate 'verify_consent': stage 4.5 judges "
                     + (f"the {marked} marked of {n} instance labels; the "
                        f"other {n - marked} "
                        + ("keep their verdicts from the previous "
                           "verification of this export"
                           if previous_verdicts(self.workdir) else
                           "ship not judged (marked by nobody)")
                        if marked else f"~{n} instance labels")
                     + f" (plus support checks for relabels) "
                     f"with {cfg['vlm']['model']}, on this machine. It costs "
                     f"minutes of GPU time and is the only pass that catches "
                     f"a wrong label; run it or skip it"
                     + ("" if available else f"\n   UNAVAILABLE: {why}"))
            consent = self.ask_verify_consent(n, cfg["vlm"]["model"],
                                              available)
            self.approve("verify_consent", run=consent)
        if not consent:
            self.say("verification SKIPPED: labels ship unverified")
            return
        from .verify import run_verify
        m = run_verify(str(self.workdir), self.cfg(),
                       force=self.export_changed,
                       should_cancel=self.should_cancel)
        if m.get("cached"):
            # a replay looked like a run that finished at once — the
            # summary below is the kept result, and the operator who
            # reopened the gate to judge again must hear that nothing did.
            self.say(f"verification already done on this export with "
                     f"{m.get('model', 'the vision model')}; nothing "
                     f"changed since (same objects, same model, same "
                     f"settings), so the kept result is replayed, not "
                     f"judged again; below is that result. To judge again: "
                     f"export again (Run pipeline after a change) or choose "
                     f"another model.")
        # Reject-candidates: relabels the vocab-only policy HELD (label kept,
        # flagged) — omitting them made the summary not reconcile with the
        # exported count (observed: 54+0+3+0=57, not 78; the 24
        # outside-vocab holds were invisible here though the report carried
        # them).
        held = (m.get("relabels_outside_vocab", 0)
                + m.get("relabels_unsupported", 0)
                + m.get("relabels_too_large", 0))
        self.say(f"verify: {m.get('confirmed', 0)} confirmed / "
                 f"{m.get('relabeled', 0)} relabeled / "
                 f"{m.get('rejected', 0)} rejected / "
                 f"{held} reject-candidate(s) (relabel held) / "
                 f"{m.get('held', 0)} HELD / "
                 f"{m.get('unverified', 0)} UNVERIFIED"
                 + (f" / {m['not_judged']} not judged (not marked)"
                    if m.get("not_judged") else "")
                 + (f"; {m['resumed']} taken from the checkpoint of the "
                    f"run that stopped" if m.get("resumed") else "")
                 + (f"; {m['carried']} carried from the previous "
                    f"verification" if m.get("carried") else ""))
        if m.get("held"):
            self.say(f"{m['held']} instance(s) ship HELD: verification "
                     f"could not confirm them (unresolved association "
                     f"evidence; see the report)")

    # -- report ---------------------------------------------------------------------
    def write_report(self) -> None:
        from .report import (build_exemplar_provenance, build_funnel,
                             build_pipeline_summary, build_provenance,
                             describe_pipeline, format_funnel)
        cfg = self.cfg()
        wd = self.workdir
        ix = json.loads((wd / "stage4/interactions.json").read_text())
        pipeline = build_pipeline_summary(str(wd))
        L = [f"# carveout report: {self.name}", "",
             f"- scene: {self.scene}",
             f"- workdir: {wd}",
             f"- code: {git_describe()}",
             f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             *describe_pipeline(pipeline),
             "", "## gates"]
        for g in GATES:
            rec = self.journal.data["gates"].get(g)
            if rec:
                ran = rec.get("run", rec.get("consent"))
                extra = ("" if ran is None else
                         f" ({'ran' if ran else 'skipped'})")
                L.append(f"- {g}: approved {rec['approved_at']}{extra}")
            else:
                L.append(f"- {g}: not approved")

        ver = ix["extended"].get("verification")
        exp = ix["extended"].get("export") or {}
        pairs = list(enumerate(zip(ix["objects"], ix["extended"]["objects"])))

        def _hold_line(i, o, e) -> str:
            why = ", ".join(e.get("hold_reasons") or [])
            pm = e.get("projection_mismatch")
            if pm:
                why += f" (projection IoU {pm['iou']} in frame {pm['frame']})"
            orig = e.get("held_original_verdict")
            if orig:
                why += f"; original verdict {orig}"
                if e.get("held_proposed_label"):
                    why += f" '{e['held_proposed_label']}'"
            return f"  - #{i} {o['label']}: {why}"

        L += ["", f"## instances: {len(ix['objects'])} exported"]
        if ver:
            L.append(f"- verify: {ver}")
            if ver.get("held"):
                L.append(f"- {ver['held']} instance(s) ship HELD: "
                         f"verification could not confirm them (per-reason "
                         f"counts are diagnostics and overlap: "
                         f"{ver.get('held_by_reason')})")
                L.extend(_hold_line(i, o, e) for i, (o, e) in pairs
                         if e.get("verify_verdict") == "HELD")
            unv = [f"  - #{i} {o['label']}: UNVERIFIED (format failure "
 f"after retries)"
                   + (f"; holds {', '.join(e['hold_reasons'])}"
                      if e.get("hold_reasons") else "")
                   for i, (o, e) in pairs
                   if e.get("verify_verdict") == "UNVERIFIED"]
            if unv:
                L.append(f"- UNVERIFIED instances ({len(unv)}):")
                L.extend(unv)
            nj = [i for i, (o, e) in pairs
                  if e.get("verify_verdict") == "NOT_JUDGED"]
            if nj:
                L.append(f"- {len(nj)} instance(s) NOT JUDGED: not marked "
                         f"for verification (the operator's choice); their "
                         f"labels ship as detected: "
                         + ", ".join(f"#{i}" for i in nj))
        else:
            L.append("- verification: NOT RUN, labels unverified")
            n_held = exp.get("held_association", 0)
            if n_held:
                L.append(f"- {n_held} instance(s) held by association "
                         f"evidence before verification (reasons "
                         f"{exp.get('held_by_reason')}); a projection "
                         f"mismatch cannot be known without running verify")
                L.extend(_hold_line(i, o, e) for i, (o, e) in pairs
                         if e.get("hold_reasons"))
        # Disappearance is reported by stage 3, not inferred: tracks that
        # emptied (all detections ambiguous) or fell under the floor, next
        # to the export-floor drops, whether or not verification ran.
        dropped_tracks = exp.get("dropped_tracks") or []
        floor_dropped = (ix["extended"].get("export_floor") or {}).get("dropped") or []
        if dropped_tracks or floor_dropped:
            L += ["", "## dropped before export"]
            by_reason: dict = {}
            for d in dropped_tracks:
                key = (d["class"], d["reason"])
                by_reason[key] = by_reason.get(key, 0) + 1
            for (c, reason), n in sorted(by_reason.items()):
                L.append(f"- stage 3 dropped {n} {c!r} track(s): {reason}")
            for d in floor_dropped:
                L.append(f"- export floor dropped {d['label']!r} instance "
                         f"{d['instance_id']} ({d['gaussians']} gaussians, "
                         f"{d['matched_frames']} matched frames)")

        kills = self.factor_kills()
        L += ["", "## area-factor kills"]
        if kills:
            L += [f"- {k['concept']!r} ({k['channel']}) frame "
                  f"{k['frame']:04d}: area {k['area_frac']:.4f} = "
                  f"{k['ratio']}x median {k['median_area']} "
                  f"(score {k['score']})" for k in kills]
        else:
            L.append("- none")

        L += ["", "## 2D->3D funnel (per class)"]
        rows, flags = build_funnel(str(wd), cfg)
        L.append("```\n" + format_funnel(rows, flags) + "\n```")
        provenance = {}
        for title, fn in (("manual view provenance", build_provenance),
                          ("exemplar provenance", build_exemplar_provenance)):
            L += ["", f"## {title}"]
            prov = fn(str(wd), cfg)
            provenance[title] = prov
            L.append("```\n" + prov + "\n```" if prov else "- none this run")

        text = "\n".join(L) + "\n"
        out = wd / "run_report.md"
        write_text_atomic(out, text)
        # report.json sidecar: the same report machine-readable, for
        # the web report drawer's native tables + deep links. A run-scoped
        # snapshot like the .md — always written together, never divergent.
        unverified = [i for i, e in enumerate(ix["extended"]["objects"])
                      if e.get("verify_verdict") == "UNVERIFIED"] if ver else []
        held = [i for i, e in enumerate(ix["extended"]["objects"])
                if e.get("verify_verdict") == "HELD"] if ver else []
        not_judged = [i for i, e in enumerate(ix["extended"]["objects"])
                      if e.get("verify_verdict") == "NOT_JUDGED"] if ver else []
        held_assoc = [i for i, e in enumerate(ix["extended"]["objects"])
                      if e.get("hold_reasons")]
        write_text_atomic(wd / "report.json", json.dumps(dict(
            # version 3: the `deltas` block (compare-to reference runs)
            # went with the command-line driver that produced it.
            version=3, name=self.name, scene=str(self.scene),
            workdir=str(wd), code=git_describe(),
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            gates={g: self.journal.data["gates"].get(g) for g in GATES},
            instances=dict(count=len(ix["objects"]),
                           verification=ver or None,
                           unverified_indices=unverified,
                           not_judged_indices=not_judged,
                           held_indices=held,
                           held_association=exp.get("held_association", 0),
                           held_association_indices=held_assoc,
                           held_by_reason=exp.get("held_by_reason")),
            dropped=dict(tracks=dropped_tracks, export_floor=floor_dropped),
            factor_kills=kills,
            funnel=dict(rows=rows, flags=flags),
            provenance=provenance,
            pipeline=pipeline), indent=1))
        self.say(f"\n{'=' * 72}\n{text}\nreport -> {out}")
