# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""`carveout web` HTTP server: REST + SSE + static bundle, stdlib only.

One ThreadingHTTPServer on 127.0.0.1 (localhost, single user; no
auth, no telemetry). REST speaks run_journal.json semantics through the
shared sequencing core; ONE SSE stream per scene carries liveness
(stage/log/journal/refusal/stop_look events, Last-Event-ID replay — the
re-attach contract). The frontend bundle is OPERATOR-BUILT
(webui/dist, Node at build time only); a missing dist fails with the
build command, never a blank page.

Cache discipline (inherited from view.py): small JSON/
image responses are no-store — per-scene files change identity under one
URL; the scene file keeps no-cache so its 304 saves the re-transfer.

No secrets to leak and no outbound calls: the LLM stages run a local model
so endpoints expose model AVAILABILITY, never a key. Streamed logs are
the pipeline's own lines.
"""

import json
import logging
import os
import re
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .. import config as _config
from ..config import find_profile
from ..staging import (discard_staged_render, publish_staged_render,
                       staged_render)
from ..manual_views import load_manual_views
from ..quality import GATE_QUALITY_KEYS, judged_stat
from ..refusal import Refusal
from ..scene_io import SPLAT_SUFFIXES, resolve_scene_path
from ..sequencing import (GATES, REPO, WorkdirLocked, fingerprint,
                          write_text_atomic,
                          validate_vocabulary)
from .runs import RunManager

# "How was it filmed?" — the three answers the creation dialog offers, as
# `render.path_mode` records them (the tile wording lives in the dialog).
FILMING_ANSWERS = ("interior", "orbit", "ground")

# The rule: no stage runs, and no gate is approved, while a
# gate BEFORE it is unapproved — the routes refuse to the earliest such gate,
# naming the act; the panels disable the same buttons with the same
# sentence and stay reachable for inspection. Calibration writes stay
# allowed (they re-open gates; they run nothing).
GATE_ACT = {
    "volume": "review the volume and approve it",
    "render": "review the rendered views and approve them",
    "vocabulary": "confirm the vocabulary",
    "exemplars": "review the exemplars and press Continue",
    "verify_consent": "answer the verification consent",
}
# the gate each stage belongs to; the gates before it must be approved
STAGE_GATE = {"render": "render", "probe": "vocabulary",
              "reprobe_exemplars": "exemplars", "pipeline": "verify_consent"}


def require_gates_before(core, gate: str) -> None:
    """Refuse, to the earliest unapproved gate before `gate`, when one is."""
    states = core.gate_states()
    for g in GATES[:GATES.index(gate)]:
        st = states.get(g)
        if not (st and st.get("state") == "approved"):
            raise Refusal(f"the {g} gate is not approved; {GATE_ACT[g]} "
                          f"first.", gate=g)

log = logging.getLogger(__name__)

DIST = REPO / "webui" / "dist"
BUILD_CMD = "npm --prefix webui ci && npm --prefix webui run build"

# Binding 127.0.0.1 keeps the network out, but not the operator's own browser:
# a hostile page they visit can still POST here cross-origin. No CORS headers
# are ever sent, so cross-origin READS were already blocked — this closes the
# writes (start runs, rewrite the volume/vocabulary, spend API budget) and the
# DNS-rebinding path that would turn reads back on. Any localhost-family
# ORIGIN on any port is accepted so the Vite dev server's proxy keeps working
# (it forwards the browser's :5173 origin); the HOST must be localhost-family,
# which is what rebinding cannot forge.
LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

_MISSING = object()


def _splat_count(p: Path) -> int | None:
    """The Gaussian count of a splat file from its header: a .ply's
    `element vertex N` line, a .sog's meta.json `count`. None when neither
    can be read; never the payload."""
    try:
        if p.suffix.lower() == ".ply":
            with open(p, "rb") as f:
                head = f.read(65536)
            for line in head.split(b"\n"):
                if line.startswith(b"element vertex"):
                    return int(line.split()[2])
                if line.strip() == b"end_header":
                    break
            return None
        import zipfile
        with zipfile.ZipFile(p) as zf:
            return int(json.loads(zf.read("meta.json"))["count"])
    except Exception:
        return None


def _read_json(path: Path, missing=_MISSING):
    """Read a JSON artifact, failing loudly but USEFULLY: a corrupt file
    raises a Refusal naming it, instead of a bare-traceback 500 that (at
    worst) takes a whole listing down with it. `missing` is returned when
    the file does not exist — pass it wherever absence is a state, not an
    error; without it, absence raises the same Refusal (callers that
    already guard with .exists() keep their own, more specific message)."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        if missing is not _MISSING:
            return missing
        raise Refusal(f"missing artifact: {path}; re-run the stage that "
                      f"writes it", status=500)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise Refusal(f"unreadable artifact {path}: {e}; re-run the stage "
                      f"that writes it, or restore the file", status=500)


def _bytes_on_disk(p: Path) -> int:
    """Size of a file or tree, symlinks counted as nothing (the viewer
    stages the scene by symlink; a link is not the megabytes it points at)."""
    if p.is_symlink() or p.is_file():
        return 0 if p.is_symlink() else p.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(p, onerror=lambda _e: None):
        for f in files:
            fp = Path(root) / f
            try:
                total += 0 if fp.is_symlink() else fp.stat().st_size
            except OSError:
                pass
    return total


def _under(p: Path, roots) -> bool:
    """True when p resolves STRICTLY inside one of roots — the guard that
    keeps a delete from walking out of the repository via a relative
    scene_path. A root itself is not under itself: a workdir of `.`
    resolved to the repository, and the repository is not a workdir."""
    rp = p.resolve()
    return any(r.resolve() in rp.parents for r in roots)


# Trees under the repository that hold what the product did not create and
# must never remove as somebody's workdir: the captures, the weights, the
# profiles, the source, the version control. A workdir at or above any of
# these is refused at creation and at deletion.
_PROTECTED = ("data", "models", "configs", "carveout", "webui", "viewer",
              "docs", "scripts", "tests", "release", "reference", ".git",
              ".github")


def _protected_reason(p: Path) -> str | None:
    """Why `p` may not be created or removed as a workdir, or None."""
    rp = p.resolve()
    repo = REPO.resolve()
    if rp == repo or rp in repo.parents:
        return "it is the repository itself (or above it)"
    for name in _PROTECTED:
        t = repo / name
        if rp == t or rp in t.parents:
            return f"it contains {name}/, which the product never removes"
        if t in rp.parents:
            return (f"it is inside {name}/, which the product never writes "
                    f"to or removes from")
    return None


def _deletion_target(kind: str, path: Path, roots, note: str = "") -> dict:
    """One row of the confirm dialog: what it is, how big, and whether this
    server is willing to remove it."""
    exists = path.exists() or path.is_symlink()
    why = None if _under(path, roots) else "outside the repository"
    if why is None and kind == "workdir":
        why = _protected_reason(path)
    removable = exists and why is None
    return dict(
        kind=kind, path=str(path), exists=exists, removable=removable,
        bytes=_bytes_on_disk(path) if exists else 0,
        note=note if removable or not exists else
             f"{why}: not removed; remove what you mean to yourself")


_ROUTES: list[tuple[str, re.Pattern, str]] = []
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def route(method: str, pattern: str):
    def deco(fn):
        _ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__))
        return fn
    return deco


class _Match:
    """A route match whose captures are percent-DECODED. The browser encodes
    a space in a path segment as %20 (fetch does it unasked), and the routes
    matched the raw path: an exemplar concept 'clipper guard' reached its
    DELETE handler as 'clipper%20guard' and was refused as unknown
    at the exemplar gate. Handlers only ever call
    group(n) with one index, so that is all this wraps."""

    def __init__(self, m: re.Match):
        self._m = m

    def group(self, i: int = 0):
        g = self._m.group(i)
        return unquote(g) if isinstance(g, str) else g


class Api:
    """Endpoint implementations; one instance shared by all requests."""

    def __init__(self, manager: RunManager):
        self.mgr = manager

    # -- scenes -----------------------------------------------------------------
    @route("GET", r"/api/scenes")
    def scenes(self, h, m, body):
        out = []
        for name, info in self.mgr.profiles().items():
            try:
                s = self.mgr.session(name)
            except Refusal as e:
                # a profile that does not parse, or a journal that does
                # not: the row says so, the rest of the library stands
                raw = info.get("raw") or {}
                out.append(dict(
                    name=name, scene=str(raw.get("scene_path") or ""),
                    workdir=str(raw.get("workdir") or ""), lock=None,
                    gaussians=None, instances=None, thumb=False,
                    last_report=None, error=str(e),
                    journal=dict(error=str(e))))
                continue
            wd = s.core.workdir
            entry = dict(name=name, scene=str(s.core.scene),
                         workdir=str(wd), lock=s.lock_info(),
                         gaussians=None, instances=None, thumb=False,
                         last_report=None)
            # One scene's corrupt artifact degrades ITS row (with the error
            # shown), never the whole library — same contract the journal
            # read below already honors.
            try:
                man = wd / "stage1/manifest.json"
                if man.exists():
                    mj = _read_json(man)
                    entry["gaussians"] = mj.get("num_gaussians")
                    entry["thumb"] = True
                ix = wd / "stage4/interactions.json"
                if ix.exists():
                    entry["instances"] = len(_read_json(ix)["objects"])
            except (Refusal, KeyError, TypeError) as e:
                entry["error"] = str(e)
            rep = wd / "run_report.md"
            if rep.exists():
                entry["last_report"] = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(rep.stat().st_mtime))
            try:
                entry["journal"] = s.journal_state()
            except Exception as e:
                entry["journal"] = dict(error=str(e))
            out.append(entry)
        return out

    @route("POST", r"/api/scenes")
    def scene_new(self, h, m, body):
        raw = body.get("scene") or ""
        given = (REPO / raw) if not os.path.isabs(raw) else Path(raw)
        if not given.exists():
            raise Refusal(f"scene file not found: {given}", status=400)
        try:
            # a directory holding exactly one scene file resolves; one
            # holding a .ply AND its .sog is refused, not guessed
            scene_path = resolve_scene_path(given)
        except (FileNotFoundError, ValueError) as e:
            raise Refusal(str(e), status=400) from e
        if scene_path.suffix.lower() not in SPLAT_SUFFIXES:
            raise Refusal(
                f"{scene_path.name}: Carveout reads "
                f"{' and '.join(SPLAT_SUFFIXES)} scenes", status=400)
        name = (body.get("scene_config")
                or scene_path.parent.name.lower())
        workdir = body.get("workdir") or f"work/{scene_path.parent.name}"
        # The name becomes the profile's file name and the workdir a tree
        # the product will write to and may later delete: neither is a
        # path the dialog gets to compose. Letters, digits, `_ . -`; the
        # workdir relative, inside the repository, under none of the trees
        # the product does not own, and nobody else's.
        if not _NAME_RE.match(str(name)):
            raise Refusal(
                f"profile name {name!r}: letters, digits, '_', '.' and '-' "
                f"only, starting with a letter or digit", status=400)
        wd = Path(str(workdir))
        if wd.is_absolute() or any(part in ("..", "") for part in wd.parts) \
                or not wd.parts:
            raise Refusal(
                f"workdir {workdir!r}: a relative path inside the "
                f"repository, like work/{scene_path.parent.name}",
                status=400)
        wd_full = REPO / wd
        if not _under(wd_full, [REPO]):
            raise Refusal(f"workdir {workdir!r} resolves outside the "
                          f"repository", status=400)
        why = _protected_reason(wd_full)
        if why:
            raise Refusal(f"workdir {workdir!r}: {why}", status=400)
        for other, info in self.mgr.profiles().items():
            raw = info.get("raw") or {}
            if raw.get("workdir") and \
                    (REPO / raw["workdir"]).resolve() == wd_full.resolve():
                raise Refusal(
                    f"workdir {workdir!r} is already the workdir of scene "
                    f"{other!r}; two scenes cannot share one", status=409)
        if find_profile(name) is not None:
            raise Refusal(
                f"profile {name!r} already exists; open it from the "
                f"library, or pick another name", status=409)
        from ..sequencing import RunCore
        said: list[str] = []

        class _ScaffoldCore(RunCore):
            def say(self, msg):   # no session yet: collect for the response
                said.append(msg)
        core = _ScaffoldCore(str(scene_path), str(REPO / workdir), name,
                             self.mgr.cfg_loader_factory)
        # The creation dialog's two questions: is the capture
        # metric — a factor (1.0 for metric, or the number the operator
        # knows) or null, which means "I will measure one thing on the
        # canvas" (the ruler at the volume gate; Propose refuses until then)
        # — and how it was filmed, which is REQUIRED: inside a space /
        # around a subject / outdoors on the ground is `render.path_mode`,
        # and the profile records the operator's answer, never a guess.
        factor = body.get("scale_m_per_unit")
        if factor is not None:
            try:
                factor = float(factor)
            except (TypeError, ValueError):
                raise Refusal(f"scale_m_per_unit={factor!r} is not a "
                              f"number (metres per scene unit)", status=400)
            if not factor > 0:
                raise Refusal("scale_m_per_unit must be greater than zero",
                              status=400)
        path_mode = body.get("path_mode")
        if path_mode not in FILMING_ANSWERS:
            raise Refusal(
                f"path_mode={path_mode!r}: say how the scene was filmed, "
                f"one of {', '.join(FILMING_ANSWERS)} (inside a space / "
                f"around a subject / outdoors on the ground)", status=400)
        extra = {"scene": {"scale_m_per_unit": factor},
                 "render": {"path_mode": path_mode}}
        # The third answer: which vision model, per
        # scene, from the ones installed — proposed by the dialog from the
        # card and the scene, chosen by the operator. Optional: absent
        # means the server's default.
        vdir = body.get("vlm_model_dir")
        if vdir:
            from ..vlm import list_models
            known = {m["dir"] for m in list_models(None)}
            if vdir not in known:
                raise Refusal(f"vlm_model_dir={vdir!r} is not an installed "
                              f"model; installed: {sorted(known)}", status=400)
            extra["vlm"] = {"model_dir": vdir}
        core.scaffold(extra=extra)
        return dict(name=name, workdir=workdir,
                    profile=str(core.profile_path), messages=said), 201

    # -- settings (read-only: where profiles live, is the model on disk) -----
    @route("GET", r"/api/settings")
    def settings(self, h, m, body):
        # Model STATUS only, never a key (there is none). The profile
        # directory is fixed for the process (`carveout web --profiles-dir`).
        from ..vlm import backend_available, describe_model, list_models
        ok, why = backend_available(None)
        return dict(profiles_dir=str(_config.PROFILES_DIR),
                    vlm_available=ok, vlm_reason=why, **describe_model(None),
                    vlm_models=list_models(None))

    # The creation dialog's proposal: which installed model fits this card
    # beside this scene. A read of the header and the card; nothing loads.
    @route("POST", r"/api/scenes/estimate")
    def scene_estimate(self, h, m, body):
        from ..vlm import estimate_fit
        raw = body.get("scene") or ""
        p = (REPO / raw) if raw and not os.path.isabs(raw) else Path(raw)
        try:
            p = resolve_scene_path(p) if raw and p.exists() else None
        except (FileNotFoundError, ValueError):
            p = None
        return estimate_fit(None, p)

    # -- scene removal ---------------------------------------------------------
    # Deleting is an explicit operator act, like every other act here: the
    # dialog is shown what will go and how big it is, the name is typed back,
    # and the scene's own files are a separate opt-in — Carveout did not
    # create the capture and does not assume it may destroy it.

    def _deletion_plan(self, name: str) -> tuple:
        s = self.mgr.session(name)
        prof = s.core.profile_path
        roots = [REPO, _config.PROFILES_DIR]
        # The capture under data/scenes is NOT a target: Carveout never
        # writes there and never deletes there. Everything the
        # product made or learned about the scene is in these two.
        return s, [
            _deletion_target("profile", prof, roots,
                             "the scene disappears from the library"),
            _deletion_target("workdir", s.core.workdir, [REPO],
                             "renders, masks, instances, and the volume, "
                             "views, exemplars and viewer settings learned "
                             "about the scene"),
        ]

    @route("GET", r"/api/scenes/([^/]+)/deletion")
    def scene_deletion(self, h, m, body):
        """What deleting this scene would remove — sizes and blockers, so the
        dialog states the consequence instead of implying it."""
        name = m.group(1)
        s, targets = self._deletion_plan(name)
        job = s.job_state()
        return dict(name=name, targets=targets, lock=s.lock_info(),
                    running=job["kind"] if job and job["state"] == "running"
                    else None)

    @route("DELETE", r"/api/scenes/([^/]+)")
    def scene_delete(self, h, m, body):
        name = m.group(1)
        if body.get("confirm") != name:
            raise Refusal(
                f"deletion not confirmed; the dialog must send the scene "
                f"name back verbatim (expected {name!r})", status=400)
        s, targets = self._deletion_plan(name)
        job = s.job_state()
        if job and job["state"] == "running":
            raise Refusal(
                f"{name!r} is running {job['kind']}: cancel it first; "
                f"deleting a workdir under a live stage corrupts the run",
                status=409)
        lock = s.lock_info()
        if lock and lock.get("alive") and not lock.get("mine"):
            raise WorkdirLocked(s.wlock.read(), stale=False)

        removed, kept = [], []
        for t in targets:
            if not t["removable"]:
                kept.append(t["path"])
                continue
            path = Path(t["path"])
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            removed.append(dict(kind=t["kind"], path=t["path"],
                                bytes=t["bytes"]))
            log.warning("deleted %s for scene %r: %s", t["kind"], name,
                        t["path"])
        if lock and lock.get("mine"):
            s.wlock.release()
        self.mgr.forget(name)
        return dict(deleted=name, removed=removed, kept=kept,
                    bytes=sum(r["bytes"] for r in removed))

    # -- picking a scene file ---------------------------------------------------
    # A browser cannot hand a server a path — a file input yields CONTENT, so
    # an OS dialog would mean uploading gigabytes to a server sharing the same
    # disk. These two read-only endpoints are the picker instead: the
    # documented convention, and a directory walk for scenes kept elsewhere.
    # Never contents, and only directories and splat files are ever named.

    @route("GET", r"/api/fs/scenes")
    def fs_scenes(self, h, m, body):
        """Every splat file under the documented data/scenes/<name>/ layout,
        each flagged with the profile already using it."""
        taken = {str((REPO / (info["raw"]["scene_path"])).resolve()): name
                 for name, info in self.mgr.profiles().items()}
        out = []
        root = REPO / "data" / "scenes"
        for f in sorted(root.glob("*/*")) if root.is_dir() else []:
            if f.suffix.lower() not in SPLAT_SUFFIXES:
                continue
            out.append(dict(path=str(f.relative_to(REPO)),
                            name=f.name, scene_dir=f.parent.name,
                            bytes=f.stat().st_size,
                            used_by=taken.get(str(f.resolve()))))
        return dict(root=str(root), scenes=out)

    @route("GET", r"/api/fs/splat_info")
    def splat_info(self, h, m, body):
        """What a scene file costs before it is a scene: its
        Gaussian count from the header alone (a .ply's `element vertex`,
        a .sog's meta.json — never the payload), and, from that and the
        card's size, how many classes of vocabulary the lift's class pass
        fits. Read-only; the New scene dialog shows the line."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(h.path).query)
        raw = (q.get("path") or [""])[0]
        if not raw:
            raise Refusal("path is required", status=400)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = REPO / p
        if not p.is_file() or p.suffix.lower() not in SPLAT_SUFFIXES:
            raise Refusal(f"not a scene file: {raw}", status=404)
        n = _splat_count(p)
        out = dict(path=str(p), bytes=p.stat().st_size, gaussians=n)
        if n is None:
            return dict(out, note="the file's count could not be read from "
                                  "its header")
        from ..lift import scene_capacity
        try:
            import torch
            card = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else None
        except Exception:                          # no usable card here
            card = None
        if card is None:
            return dict(out, note="no CUDA card found on this machine")
        return dict(out, **scene_capacity(n, card))

    @route("GET", r"/api/fs")
    def fs_browse(self, h, m, body):
        """One directory: its subdirectories and any splat files in it.

        Read-only and content-free by construction — it answers with names
        and sizes, never bytes of a file, and lists nothing that is not a
        directory or a scene Carveout can open."""
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(h.path).query)
        raw = (q.get("path") or [str(REPO)])[0]
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = REPO / target
        try:
            target = target.resolve(strict=True)
        except (OSError, RuntimeError):
            raise Refusal(f"no such directory: {raw}", status=404) from None
        if not target.is_dir():
            raise Refusal(f"not a directory: {target}", status=400)
        dirs, files = [], []
        try:
            for e in sorted(target.iterdir(), key=lambda x: x.name.lower()):
                if e.name.startswith("."):
                    continue
                if e.is_dir():
                    dirs.append(dict(name=e.name, path=str(e)))
                elif e.suffix.lower() in SPLAT_SUFFIXES:
                    try:
                        files.append(dict(name=e.name, path=str(e),
                                          bytes=e.stat().st_size))
                    except OSError:
                        pass
        except PermissionError:
            raise Refusal(f"not readable: {target}", status=403) from None
        return dict(path=str(target),
                    parent=str(target.parent) if target.parent != target
                    else None,
                    home=str(Path.home()), repo=str(REPO),
                    dirs=dirs, files=files)

    @route("GET", r"/api/scenes/([^/]+)/thumb.png")
    def scene_thumb(self, h, m, body):
        s = self.mgr.session(m.group(1))
        cams = s.core.workdir / "stage1/cameras.json"
        if not cams.exists():
            raise Refusal("no renders yet", status=404)
        frames = _read_json(cams)["frames"]
        best = max(frames, key=lambda f: f.get("sharpness", 0))
        return ("file", s.core.workdir / "stage1" / best["file"], "image/png")

    @route("POST", r"/api/scenes/([^/]+)/lock/release")
    def lock_release(self, h, m, body):
        s = self.mgr.session(m.group(1))
        rec = s.wlock.read()
        if rec is None:
            return dict(released=False, reason="no lock present")
        # Explicit operator act; refuses a live holder (never automatic).
        s.wlock.break_stale()
        return dict(released=True, was=rec)

    # -- journal / gates ------------------------------------------------------------
    @route("GET", r"/api/runs/([^/]+)/journal")
    def journal(self, h, m, body):
        return self.mgr.session(m.group(1)).journal_state()

    @route("POST", r"/api/runs/([^/]+)/gates/([a-z_]+)/reopen")
    def gate_reopen(self, h, m, body):
        s, gate = self.mgr.session(m.group(1)), m.group(2)
        if gate not in GATES:
            raise Refusal(f"unknown gate {gate!r}", status=404)
        demoted, journal = s.reopen_gate(gate)
        return dict(demoted=demoted, journal=journal)

    @route("POST", r"/api/runs/([^/]+)/gates/([a-z_]+)/approve")
    def gate_approve(self, h, m, body):
        s, gate = self.mgr.session(m.group(1)), m.group(2)
        core = s.core
        if gate in GATES:
            require_gates_before(core, gate)   # gates approve in order

        def precheck():
            if gate == "volume":
                from ..volume import WHOLE_SCENE, stale_volume_reason
                vol = core.workdir / "volume.json"
                data = _read_json(vol, missing={})
                if not data.get("boxes") and data.get("scope") != WHOLE_SCENE:
                    raise Refusal("no boxes; a volume is required (it "
                                  "scopes coverage, aim targets and "
                                  "lift/export); draw one, or take the "
                                  "whole-scene act", gate="volume")
                stale = stale_volume_reason(vol, core.cfg())
                if stale:
                    raise Refusal(stale, gate="volume")
            elif gate == "render":
                if not (core.workdir / "stage1/manifest.json").exists():
                    raise Refusal("no stage-1 render to approve; run the "
                                  "render stage first", gate="render")
                # Approving means "I reviewed these views". Verify they still
                # exist: a manifest describing frames that are gone would pin
                # the journal to a review of nothing, and the stages
                # downstream glob the directory the approval claims to have
                # covered.
                missing = self._missing_frames(s)
                if missing:
                    raise Refusal(
                        f"{len(missing)} of the recorded frames are missing "
                        f"from stage1/frames; the render on disk no longer "
                        f"matches what was reviewed; re-render before "
                        f"approving",
                        gate="render", remedy="re-render this scene")
                # ... and that the views file is the one this render drew
                #: a delete or an edit after the render leaves the
                # frames on disk as they were, and approving would pin the
                # NEW hash over a render of the OLD views — the cascade
                # stops at this gate, and verify was the first to notice.
                from ..manual_views import (stage1_manual_record,
                                            resolve_manual_views_path)
                from ..sequencing import fingerprint
                rec = stage1_manual_record(core.workdir)
                live = fingerprint(resolve_manual_views_path(core.workdir,
                                                             core.cfg()))
                if rec.get("hash", "absent") != live:
                    raise Refusal(
                        "the manual views changed since this render (a view "
                        "added, moved or deleted); re-render before "
                        "approving, so the frames match the views",
                        gate="render", remedy="re-render this scene")
            elif gate == "exemplars":
                # An empty set is a legitimate approval; crops are not
                # until the probe has seen them: Continue used to
                # take unprobed crops straight into detection.
                probed, n = self._exemplars_probed(s)
                if not probed:
                    raise Refusal(
                        f"{n} crop{'s' if n != 1 else ''} not yet probed; "
                        f"Re-probe, review the track rows, then continue",
                        gate="exemplars", remedy="re-probe the crops")
            else:
                raise Refusal(f"gate {gate!r} is not approved via this "
                              f"endpoint (vocabulary -> /vocabulary/confirm, "
                              f"verify_consent -> /verify/consent)",
                              status=405)
        return s.approve_gate(gate, precheck)

    @route("GET", r"/api/runs/([^/]+)/events")
    def events(self, h, m, body):
        return ("sse", self.mgr.session(m.group(1)))

    @route("POST", r"/api/runs/([^/]+)/acks")
    def acks(self, h, m, body):
        s = self.mgr.session(m.group(1))
        kind = body.get("kind")
        # area_factor_kill is the only TYPED ack left: it halts a run over a
        # never-event that reaches the output. The negative-control ceiling
        # used to be one too and is now reported rather than acknowledged —
        # negatives never export, and an ack that fired on ~40% of runs was
        # being typed unread.
        if kind != "area_factor_kill":
            raise Refusal(f"unknown ack kind {kind!r}", status=400)
        # The word is the class in the kill row: it cannot be typed
        # without reading the row.
        expected = s.expected_ack(kind)
        if s.ack_word(body.get("typed")) != s.ack_word(expected):
            raise Refusal(f"the acknowledgment must be typed: send "
                          f"{{typed: {expected!r}}}, the class named in "
                          f"the kill row, so the row cannot be acknowledged "
                          f"unread", status=400)
        block_id = body.get("id")
        if not block_id:
            raise Refusal("the acknowledgment must name the block it "
                          "answers: send {id: '<id from the event>'}",
                          status=400)
        s.supply_ack(kind, block_id)
        return dict(accepted=True, kind=kind, id=block_id)

    @route("POST", r"/api/runs/([^/]+)/jobs/cancel")
    def jobs_cancel(self, h, m, body):
        # cooperative cancel — sets the session's cancel flag; the stage
        # stops at its next loop boundary and stays re-runnable. Idempotent.
        s = self.mgr.session(m.group(1))
        return dict(cancelling=s.cancel_job())

    # -- stages -----------------------------------------------------------------------
    @route("POST", r"/api/runs/([^/]+)/stages/([a-z_]+)/start")
    def stage_start(self, h, m, body):
        s, stage = self.mgr.session(m.group(1)), m.group(2)
        core = s.core
        body = body or {}
        if stage in STAGE_GATE:
            require_gates_before(core, STAGE_GATE[stage])   # gates in order
        if stage == "propose_volume":
            from ..render import run_propose_volume
            s.start_job("propose_volume",
                        lambda: run_propose_volume(
                            str(core.scene), str(core.workdir), core.cfg(),
                            should_cancel=core.should_cancel))
        elif stage == "render":
            from ..render import run_render
            force = bool(body.get("force")) or "volume" in core.entered
            core.mark_entered("render")
            s.start_job("render", lambda: run_render(
                str(core.scene), str(core.workdir), core.cfg(), force=force,
                should_cancel=core.should_cancel))
        elif stage == "probe":
            draft = self._draft(s)
            if not draft["prompts"]:
                raise Refusal("no vocabulary yet; edit or propose first",
                              gate="vocabulary")
            from ..detect import run_detect
            prompts, negatives = list(draft["prompts"]), list(draft["negatives"])

            def probe():
                run_detect(str(core.workdir), core.cfg(), prompts,
                           negatives=negatives, force=True,
                           should_cancel=core.should_cancel)
                draft["probed"] = [prompts, negatives]
                self._save_draft(s)
            core.mark_entered("vocabulary")
            s.start_job("probe", probe)
        elif stage == "reprobe_exemplars":
            from ..detect import run_detect
            d = core.cfg()["detect"]
            core.mark_entered("exemplars")
            s.start_job("reprobe_exemplars", lambda: run_detect(
                str(core.workdir), core.cfg(),
                list(d.get("prompts") or []),
                negatives=list(d.get("negatives") or []),
                force=False,   # fold: only the exemplar pass re-runs
                should_cancel=core.should_cancel))
        elif stage == "pipeline":
            # detect -> lift -> export, unattended; factor-cap kills halt
            # the job on the server until the typed ack round-trips.
            s.start_job("pipeline", core.stage_pipeline)
        else:
            raise Refusal(f"unknown stage {stage!r}", status=404)
        return dict(started=stage), 202

    # -- volume -----------------------------------------------------------------------
    @route("GET", r"/api/runs/([^/]+)/volume")
    def volume_get(self, h, m, body):
        s = self.mgr.session(m.group(1))
        vol = s.core.workdir / "volume.json"
        data = _read_json(vol, missing=None)
        frame = None
        man = s.core.workdir / "stage1/manifest.json"
        dens = s.core.workdir / "density.json"
        dj = _read_json(dens, missing=None)
        if man.exists():
            frame = _read_json(man).get("scene_frame")
        elif dj is not None:
            frame = dict(up_axis="xyz"[dj["up_axis"]],
                         up_sign=dj["up_sign"], floor=dj["floor"])
        if frame is not None:
            # the levelling the frame was read under; null = the
            # file's axes (a manifest from before levelling carries none)
            frame.setdefault("alignment", (dj or {}).get("alignment"))
            # Effective floor: a scene.floor override wins over the
            # stored/detected value, so the 3D floor plane reflects an
            # operator change immediately, before any re-propose.
            override = (s.core.cfg().get("scene") or {}).get("floor")
            frame["floor_auto"] = (dj or {}).get("floor_auto",
                                                 frame.get("floor"))
            frame["floor_override"] = override
            if override is not None:
                frame["floor"] = float(override)
        # Scale read: the stage-1 manifest is authoritative once it exists,
        # the volume sidecar carries it before any render — which is the
        # point, since this gate is the first place a scene that is not to
        # scale can be caught. Re-derived against the CURRENT configured
        # factor so the panel responds to an edit immediately, the same way
        # the floor override does above. No default: with nothing recorded
        # or measured the read says so (`source: none`, the refusal's own
        # words) and the panel offers the ruler.
        from ..units import resolve_scale
        extent = ((frame or {}).get("notes", {}).get("extent")
                  or (dj or {}).get("extent"))
        scale = None
        if extent:
            from ..cameras import scale_read
            up = {"x": 0, "y": 1, "z": 2}.get(
                str((frame or {}).get("up_axis", "y")), 1)
            ext = np.asarray(extent, dtype=float)
            try:
                rec = resolve_scale(s.core.cfg())
            except Refusal as e:
                scale = dict(source="none", scale=None, scale_m_per_unit=None,
                             recorded=None, measured=None, height_m=None,
                             plausible=None, band=None, note=str(e),
                             extent_m=None)
            else:
                scale = scale_read(ext, up, s.core.cfg(), rec)
                scale["extent_m"] = [round(float(v) * rec.value, 3)
                                     for v in extent]
            # significant digits, not decimals: a kilometre-unit scene is
            # 0.0022 units across and must not read as 0.00
            scale["extent_units"] = [float(f"{float(v):.6g}") for v in extent]
        return dict(volume=data, frame=frame, scale=scale,
                    density_sidecar=dens.exists())

    @route("GET", r"/api/runs/([^/]+)/proposal")
    def proposal_get(self, h, m, body):
        """The Check cards' analysis (`<workdir>/proposal.json` v2, written
        last by the propose job): the geometry's check of the operator's
        two answers, the second opinion on the measurement, and the two
        measured settings proposed with their reasons. `null` with 200
        before any proposal — the panels read it through useRunData, where
        a 404 would log as an error. Adopting the settings is the existing
        `PUT /calibration`."""
        s = self.mgr.session(m.group(1))
        return _read_json(s.core.workdir / "proposal.json", missing=None)

    @route("PUT", r"/api/runs/([^/]+)/volume")
    def volume_put(self, h, m, body):
        s = self.mgr.session(m.group(1))
        from ..volume import WHOLE_SCENE
        if body.get("scope") == WHOLE_SCENE:
            # The whole-scene act: a volume.json that says so and holds no
            # boxes, so the journal hashes the decision like any other
            # volume — every Gaussian in scope, placement over the scene's
            # robust bounds, coverage over the scene.
            def save_whole():
                s.core.workdir.mkdir(parents=True, exist_ok=True)
                write_text_atomic(s.core.workdir / "volume.json", json.dumps(
                    dict(scene=s.core.scene.name, **self._volume_coords(s),
                         scope=WHOLE_SCENE, boxes=[]), indent=1))
                return dict(saved=0, scope=WHOLE_SCENE)
            return s.write(save_whole, gate="volume")
        boxes = body.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            raise Refusal("body must carry boxes: [{name,min,max}] (at least "
                          "one), or scope: whole_scene", status=400)
        # an edit of boxes that describe another frame would stamp their
        # old numbers as the current levelling's
        from ..volume import stale_volume_reason
        stale = stale_volume_reason(s.core.workdir / "volume.json", s.core.cfg())
        if stale:
            raise Refusal(stale, gate="volume")
        for b in boxes:
            if (not isinstance(b.get("name"), str)
                    or len(b.get("min", [])) != 3
                    or len(b.get("max", [])) != 3):
                raise Refusal(f"malformed box: {b}", status=400)
            b["min"] = [round(float(x), 3) for x in b["min"]]
            b["max"] = [round(float(x), 3) for x in b["max"]]

        def save():
            # volume.json is written VERBATIM in raw ply_world coords — the
            # journal hashes THIS file (read-back decision on the 03
            # contract). The mid-job guard matters here: a running
            # propose_volume job writes this same file.
            out = dict(scene=s.core.scene.name, **self._volume_coords(s),
                       boxes=boxes)
            s.core.workdir.mkdir(parents=True, exist_ok=True)
            write_text_atomic(s.core.workdir / "volume.json",
                              json.dumps(out, indent=1))
            return dict(saved=len(boxes))
        return s.write(save, gate="volume")

    @staticmethod
    def _volume_coords(s) -> dict:
        """Which frame a volume.json written from the canvas is in:
        the levelled one when the profile holds an alignment, with the
        angles beside the boxes; the file's otherwise, exactly as before."""
        from ..alignment import alignment_block
        try:
            blk = alignment_block((s.core.cfg().get("scene") or {}).get("alignment"))
        except ValueError as e:
            raise Refusal(f"the scene profile's alignment is malformed: {e}",
                          gate="volume")
        return dict(coords="scene", alignment=blk) if blk else dict(coords="ply_world")

    @staticmethod
    def _pinned_up(core, cfg, act: str) -> tuple[int, float]:
        """The up axis (index, sign) the level and square acts turn about:
        the profile's when pinned, else the last proposal's; a refusal
        naming the act when neither exists."""
        up_cfg = str((cfg.get("scene") or {}).get("up_axis", "-y"))
        if len(up_cfg) != 2 or up_cfg[1] not in "xyz" or up_cfg[0] not in "+-":
            dj = _read_json(core.workdir / "density.json", missing=None)
            if dj is None:
                raise Refusal("the scene's up axis is not pinned yet; pick "
                              f"it above, then {act}", gate="volume")
            return int(dj["up_axis"]), float(dj["up_sign"])
        return "xyz".index(up_cfg[1]), (1.0 if up_cfg[0] == "+" else -1.0)

    @route("POST", r"/api/runs/([^/]+)/alignment/level")
    def alignment_level(self, h, m, body):
        """Level the scene from the operator's clicks: three points
        on the floor, optionally two along a wall, FILE frame. The angles
        are computed here (no scene load) and written to the profile as a
        calibration write — the volume gate and everything after re-open."""
        from ..alignment import alignment_from_points
        s = self.mgr.session(m.group(1))
        core = s.core
        body = body or {}
        floor, wall = body.get("floor"), body.get("wall")
        cfg = core.cfg()
        up_axis, up_sign = self._pinned_up(core, cfg, "level")
        try:
            angles = alignment_from_points(floor, wall, up_axis, up_sign)
        except (ValueError, TypeError) as e:
            raise Refusal(f"could not level from these points: {e}",
                          gate="volume", status=400)
        block = dict(angles, source="levelled",
                     points=dict(floor=[[float(x) for x in p] for p in floor],
                                 wall=([[float(x) for x in p] for p in wall]
                                       if wall else None)))
        update = {"scene": {"alignment": block}}
        affected = core.calibration_gate(update)
        what = "calibration: scene.alignment (levelled from your clicks)"
        s.start_job("calibration",
                    lambda: core.write_calibration(update, affected, what),
                    gate=affected)
        return dict(started="calibration", affected_gate=affected,
                    alignment=block), 202

    @route("POST", r"/api/runs/([^/]+)/alignment/square")
    def alignment_square(self, h, m, body):
        """Square the scene to an edge: two points along one
        straight edge, FILE frame. The yaw alone changes — the recorded
        tilt stays (the proposal's, the operator's, or none), and so do
        the floor points it came from. The same calibration write as
        levelling: the volume gate and everything after re-open."""
        from ..alignment import validate_alignment, yaw_from_edge
        s = self.mgr.session(m.group(1))
        core = s.core
        edge = (body or {}).get("edge")
        cfg = core.cfg()
        up_axis, _ = self._pinned_up(core, cfg, "square")
        try:
            current = validate_alignment((cfg.get("scene") or {}).get("alignment")) or {}
            tilt = current.get("tilt_deg") or [0.0, 0.0]
            yaw = yaw_from_edge(edge, tilt, up_axis)
        except (ValueError, TypeError) as e:
            raise Refusal(f"could not square from these points: {e}",
                          gate="volume", status=400)
        floor = (current.get("points") or {}).get("floor")
        block = dict(tilt_deg=[round(float(tilt[0]), 4), round(float(tilt[1]), 4)],
                     yaw_deg=round(yaw, 4), source="levelled",
                     points=dict(floor=floor,
                                 wall=[[float(x) for x in p] for p in edge]))
        update = {"scene": {"alignment": block}}
        affected = core.calibration_gate(update)
        what = "calibration: scene.alignment (squared to the edge you clicked)"
        s.start_job("calibration",
                    lambda: core.write_calibration(update, affected, what),
                    gate=affected)
        return dict(started="calibration", affected_gate=affected,
                    alignment=block), 202

    # -- calibration --------------------------------------------------------------
    # The adjustable-key -> re-opened-gate whitelist lives with the gate
    # rules (RunCore.calibration_paths), beside _gate_inputs.

    @route("GET", r"/api/runs/([^/]+)/calibration")
    def calibration_get(self, h, m, body):
        s = self.mgr.session(m.group(1))
        cfg = s.core.cfg()
        sc = cfg.get("scene") or {}
        rn = cfg.get("render") or {}
        it = rn.get("interior") or {}
        qf = rn.get("quality_filter") or {}
        dv = rn.get("diversity") or {}
        floor_auto = None
        dens = s.core.workdir / "density.json"
        if dens.exists():
            dj = _read_json(dens)
            floor_auto = dj.get("floor_auto", dj.get("floor"))
        return dict(
            scene=dict(floor=sc.get("floor"), floor_auto=floor_auto,
                       up_axis=sc.get("up_axis"),
                       floor_band_frac=sc.get("floor_band_frac"),
                       # the floor rule's bin, read-only: the start pose
                       # estimates a floor by the same rule before a
                       # proposal has detected one
                       hist_bin_m=sc.get("hist_bin_m"),
                       # no default: null IS the answer "not recorded"
                       scale_m_per_unit=sc.get("scale_m_per_unit"),
                       # the ruler's record when the factor was measured
                       # on the canvas; null after a declared or typed one
                       measured=sc.get("measured"),
                       # the levelling: the angles, their source and
                       # the clicks; null = the file's axes
                       alignment=sc.get("alignment")),
            render=dict(path_mode=rn.get("path_mode"),
                        cameras_in_volume=it.get("cameras_in_volume"),
                        focus_aim=it.get("focus_aim"),
                        min_pos_sep=dv.get("min_pos_sep"),
                        num_views=rn.get("num_views"),
                        candidate_factor=rn.get("candidate_factor"),
                        # the eye height the path mode places cameras at:
                        # the start pose stands there too (read-only here)
                        eye_height=((rn.get("ground") or {}).get("eye_height")
                                    if rn.get("path_mode") == "ground"
                                    else it.get("eye_height")),
                        coverage={k: (rn.get("coverage") or {}).get(k)
                                  for k in ("target_frac", "max_extra_rounds",
                                            "round_views")},
                        # Whatever the gate is allowed to write, it can read —
                        # one derived set drives both, so the panel shows
                        # exactly the thresholds a refusal can name.
                        quality_filter={k: qf.get(k)
                                        for k in sorted(GATE_QUALITY_KEYS)},
                        # the thresholds that may hold "auto" (resolved
                        # from the scene's width at render): the panel
                        # offers the act only for these
                        quality_auto=sorted((qf.get("auto_frac") or {}).keys())),
            # the scene's vision model: chosen at creation or on the
            # vocabulary / verify panels; a verify decision parameter
            vlm=dict(model_dir=(cfg.get("vlm") or {}).get("model_dir"),
                     model=(cfg.get("vlm") or {}).get("model"),
                     source=(cfg.get("vlm") or {}).get("model_source"),
                     # the operator's option, off by default
                     relabel_specific_ok=bool((cfg.get("vlm") or {}).get(
                         "relabel_specific_ok", False)),
                     # the small-object render option and its cap
                     verify_isolated=bool((cfg.get("vlm") or {}).get(
                         "verify_isolated", False)),
                     verify_isolated_max_gaussians=int((cfg.get("vlm") or {}).get(
                         "verify_isolated_max_gaussians", 300))),
            lift=dict(instance_voxel=(cfg.get("lift") or {}).get(
                          "instance_voxel"),
                      # the instancing algorithm: tracks | connectivity —
                      # readable here (it was writable, never readable,
                      # so no panel could show it)
                      instancing=(cfg.get("lift") or {}).get(
                          "instancing", "tracks")))

    @route("PUT", r"/api/runs/([^/]+)/calibration")
    def calibration_put(self, h, m, body):
        from ..sequencing import _leaf_paths
        s = self.mgr.session(m.group(1))
        core = s.core
        update = body or {}
        affected = core.calibration_gate(update)   # whitelist lives there
        what = "calibration: " + ", ".join(
            ".".join(p) for p, _ in _leaf_paths(update))
        # Runs as a job (parity with the exemplar-threshold write); serialises
        # against any live stage (won't rewrite the profile mid-render).
        s.start_job("calibration",
                    lambda: core.write_calibration(update, affected, what),
                    gate=affected)
        return dict(started="calibration", affected_gate=affected), 202

    # -- cameras / views ------------------------------------------------------------
    @staticmethod
    def _refused_attempt(s, man: dict) -> dict | None:
        """The most recent render attempt, when it was REFUSED after the render
        on disk was made — `stage1/discard_report.json` exists only then (the
        stage unlinks it when a render publishes). A refusal leaves the earlier
        render standing, by design, and the gate went on showing it — contact
        sheet, frustums, Approve — under a banner saying every view was
        discarded. This is the signal the panel needs
        to say which render it is showing, and how the profile has moved
        since that render was made."""
        out = s.core.workdir / "stage1"
        rep_p, man_p = out / "discard_report.json", out / "manifest.json"
        if not (rep_p.exists() and man_p.exists()):
            return None
        try:
            rep = json.loads(rep_p.read_text())
        except json.JSONDecodeError:
            return None
        if rep_p.stat().st_mtime < man_p.stat().st_mtime:
            return None      # an older report a later publish did not clear
        # The knobs the render on disk ran under (the manifest echoes the
        # profile as it was) against the profile now — in the profile's own
        # numbers, exactly as the calibration panel shows them.
        was = (man.get("config") or {}).get("quality_filter") or {}
        now = (s.core.cfg().get("render") or {}).get("quality_filter") or {}
        changed = {k: dict(rendered=was.get(k), now=now.get(k))
                   for k in sorted(GATE_QUALITY_KEYS)
                   if k in was and was.get(k) != now.get(k)}
        stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(t))
        return dict(refused_at=rep.get("generated"),
                    views_attempted=rep.get("views_attempted"),
                    views_kept=rep.get("views_kept", 0),
                    rendered_at=stamp(man_p.stat().st_mtime),
                    thresholds_changed=changed)

    @staticmethod
    def _missing_frames(s) -> list[str]:
        """Frames cameras.json records that are not on disk. Non-empty means
        the workdir is describing a render it no longer has."""
        p = s.core.workdir / "stage1/cameras.json"
        if not p.exists():
            return []
        try:
            frames = json.loads(p.read_text()).get("frames", [])
        except json.JSONDecodeError:
            return []
        return [f["file"] for f in frames
                if not (s.core.workdir / "stage1" / f["file"]).exists()]

    @route("GET", r"/api/runs/([^/]+)/cameras")
    def cameras(self, h, m, body):
        s = self.mgr.session(m.group(1))
        cams_p = s.core.workdir / "stage1/cameras.json"
        if not cams_p.exists():
            raise Refusal("no stage-1 render yet", status=404)
        cams = _read_json(cams_p)
        man = _read_json(s.core.workdir / "stage1/manifest.json")
        # Manual poses saved but NOT in this render. views_manual above comes
        # from the stage-1 manifest, so a just-saved capture moved nothing on
        # screen and the operator could not tell it had been stored (it only
        # exists after a re-render). Rendered frames carry
        # provenance="manual:<id>" and the saved views carry those same ids,
        # so this is an exact set difference — no pose matching needed.
        # ... and EDITED poses, which the id test alone cannot see: a moved
        # camera keeps its id and is already in the render, so it read as 0
        # pending while the gate went stale underneath. Compare the saved pose
        # against the rendered c2w through load_manual_views rather than
        # redoing the quaternion conversion here — one implementation of that
        # contract, in the module that documents it.
        mv_path, mv = self._manual_views_file(s)
        rendered = {f["provenance"].split(":", 1)[1]: f
                    for f in cams["frames"]
                    if str(f.get("provenance", "")).startswith("manual:")}
        try:
            saved = {v["id"]: v for v in load_manual_views(
                mv_path, s.core.cfg())}
        except Exception:
            saved = {}          # malformed file: the render itself will refuse
        pending = []
        for v in mv.get("views", []):
            fr = rendered.get(v["id"])
            if fr is None:
                pending.append(dict(id=v["id"], label=v.get("label"),
                                    why="new"))
                continue
            cur = saved.get(v["id"])
            if cur is not None and not np.allclose(
                    cur["c2w"], np.asarray(fr["c2w"], dtype=float),
                    rtol=0, atol=1e-6):
                pending.append(dict(id=v["id"], label=v.get("label"),
                                    why="edited"))
        # ... and DELETED views: in the render, gone from the file.
        # Reported with the frame they still occupy, so the sheet and the
        # frustums can strike them until the re-render drops them — a
        # delete that changed nothing on screen read as "did it work?".
        live_ids = {v["id"] for v in mv.get("views", [])}
        for vid, fr in rendered.items():
            if vid not in live_ids:
                pending.append(dict(id=vid, label=fr.get("label"),
                                    why="deleted", frame_idx=fr["frame_idx"]))
        # Each discard names the stat it failed on, in the profile's numbers
        # and (for distances) in scene units — the table showed sharpness
        # for every reason, which is no help choosing the knob.
        qf_m = (man.get("config") or {}).get("quality_filter") or {}
        rec = man.get("scale") or {}
        factor = rec.get("scale_m_per_unit")
        factor = float(factor) if isinstance(factor, (int, float)) else 1.0
        discards = [dict(d, judged=judged_stat(d, qf_m, factor,
                                               man.get("auto_quality")))
                    for d in man.get("discards", [])]
        return dict(
            intrinsics={k: cams[k] for k in
                        ("width", "height", "fx", "fy", "cx", "cy")},
            frames=cams["frames"], discards=discards,
            # the factor THIS render ran at and where it came from: the
            # discard table speaks metres only under a recorded one
            scale=dict(scale_m_per_unit=factor,
                       source=rec.get("source") or "recorded"),
            coverage=man.get("coverage"),
            views_kept=man.get("views_kept"),
            views_manual=man.get("views_manual"),
            manual_pending=pending,
            frames_missing=len(self._missing_frames(s)),
            # what "auto" resolved to on this render, per knob, in scene
            # units and (under a factor) metres: the panel shows the number
            # beside a field the profile holds at "auto"
            auto_quality=man.get("auto_quality"),
            # A re-render that came out worse than this one was staged rather
            # than published; the gate offers the two acts. None = normal.
            staged=staged_render(str(s.core.workdir)),
            # The last attempt was refused and THIS payload is the render
            # before it; None = the render below is the latest attempt.
            refused=self._refused_attempt(s, man),
            health=man.get("health"))

    # -- staged render (an unhealthy re-render, held back from publishing) ----------
    @route("POST", r"/api/runs/([^/]+)/render/staged/publish")
    def publish_staged(self, h, m, body):
        # EXPLICIT operator act, the counterpart of the automatic swap that
        # used to happen here. Renames only — no GPU, nothing recomputed. It
        # rewrites cameras.json and manifest.json, so the render gate's pinned
        # hash changes and it, with every gate after it, demotes. The mid-job
        # guard matters here: a running render writes those same files.
        s = self.mgr.session(m.group(1))

        def publish():
            try:
                return publish_staged_render(str(s.core.workdir))
            except RuntimeError as e:
                raise Refusal(str(e), gate="render", status=409) from None
        man = s.write(publish, gate="render")
        return dict(published=True, views_kept=man.get("views_kept"),
                    journal=s.journal_state())

    @route("POST", r"/api/runs/([^/]+)/render/staged/discard")
    def discard_staged(self, h, m, body):
        s = self.mgr.session(m.group(1))

        def discard():
            if not discard_staged_render(str(s.core.workdir)):
                raise Refusal("nothing staged to discard", gate="render",
                              status=409)
            return dict(discarded=True)
        return s.write(discard, gate="render")

    @route("GET", r"/api/runs/([^/]+)/frames/(\d+).png")
    def frame_png(self, h, m, body):
        s = self.mgr.session(m.group(1))
        p = s.core.workdir / "stage1/frames" / f"frame_{int(m.group(2)):04d}.png"
        return ("file", p, "image/png")

    def _manual_views_file(self, s):
        from ..manual_views import resolve_manual_views_path
        p = resolve_manual_views_path(s.core.workdir, s.core.cfg())
        data = _read_json(p, missing=dict(version=1, space="ply", views=[]))
        return p, data

    @route("GET", r"/api/runs/([^/]+)/views")
    def views_get(self, h, m, body):
        _, data = self._manual_views_file(self.mgr.session(m.group(1)))
        return data

    @staticmethod
    def _next_view_label(views) -> str:
        """The next "view N": one past the highest N in use, not the count
        plus one — after a delete the count has a hole and "view 36" was
        handed out twice."""
        import re
        n = 0
        for v in views:
            mm = re.fullmatch(r"view (\d+)", str(v.get("label") or ""))
            if mm:
                n = max(n, int(mm.group(1)))
        return f"view {n + 1}"

    @route("POST", r"/api/runs/([^/]+)/views")
    def views_post(self, h, m, body):
        import uuid
        s = self.mgr.session(m.group(1))

        def save():
            p, data = self._manual_views_file(s)
            view = {k: body[k] for k in
                    ("position", "quaternion", "target", "fov_deg", "aspect")}
            view.update(id=str(uuid.uuid4()),
                        label=body.get("label")
                        or self._next_view_label(data["views"]),
                        target_source=body.get("target_source", "unknown"),
                        resolution=body.get("resolution"))
            data["views"].append(view)
            write_text_atomic(p, json.dumps(data, indent=1))
            return dict(saved=view["id"], n=len(data["views"]))
        return s.write(save, gate="render"), 201

    @route("PATCH", r"/api/runs/([^/]+)/views/([^/]+)")
    def views_patch(self, h, m, body):
        # reclassified at read-back: content-hash cascade already
        # handles an edited pose; this is the existing write path, granular.
        s = self.mgr.session(m.group(1))

        def patch():
            p, data = self._manual_views_file(s)
            for v in data["views"]:
                if v["id"] == m.group(2):
                    for k in ("position", "quaternion", "target", "fov_deg",
                              "aspect", "label", "target_source"):
                        if k in body:
                            v[k] = body[k]
                    write_text_atomic(p, json.dumps(data, indent=1))
                    return dict(updated=v["id"])
            raise Refusal(f"manual view {m.group(2)!r} not found", status=404)
        return s.write(patch, gate="render")

    @route("DELETE", r"/api/runs/([^/]+)/views/([^/]+)")
    def views_delete(self, h, m, body):
        s = self.mgr.session(m.group(1))

        def delete():
            p, data = self._manual_views_file(s)
            n0 = len(data["views"])
            data["views"] = [v for v in data["views"] if v["id"] != m.group(2)]
            if len(data["views"]) == n0:
                raise Refusal(f"manual view {m.group(2)!r} not found",
                              status=404)
            write_text_atomic(p, json.dumps(data, indent=1))
            return dict(deleted=m.group(2), n=len(data["views"]))
        return s.write(delete, gate="render")

    # -- fully-manual mode --------------------------------------------------------
    # Two actions, one idea: who owns the camera set. Auto views are DERIVED —
    # regenerated every render, then pruned and filtered — so a pose edited in
    # place would be silently recomputed away. Converting the whole set to
    # manual views and switching render.path_mode to "manual" makes the render
    # reproduce exactly what is on disk, which is what lets an adjusted camera
    # stay adjusted. `reset` is the way back: it starts the render step over.
    def _views_state(self, s) -> tuple:
        p, data = self._manual_views_file(s)
        cams_p = s.core.workdir / "stage1/cameras.json"
        cams = _read_json(cams_p, missing=None)
        mode = ((s.core.cfg().get("render") or {}).get("path_mode"))
        return p, data, cams, mode

    @route("POST", r"/api/runs/([^/]+)/views/convert")
    def views_convert(self, h, m, body):
        from ..manual_views import pose_from_frame
        import uuid
        s = self.mgr.session(m.group(1))
        # Slot check BEFORE the views are appended: a refusal after the write
        # would leave converted views on disk that a retry appends again.
        s.ensure_job_slot()
        p, data, cams, mode = self._views_state(s)
        if mode == "manual":
            raise Refusal("this scene is already in fully-manual mode; its "
                          "views are all editable", status=409)
        if not cams:
            raise Refusal("nothing to convert: this scene has no render yet",
                          status=409)
        auto = [f for f in cams["frames"]
                if not str(f.get("provenance", "")).startswith("manual:")]
        if not auto:
            raise Refusal("nothing to convert: every view in this render is "
                          "already a manual one", status=409)
        for fr in auto:
            v = pose_from_frame(fr, cams)
            # `resolution` is metadata only (the pipeline re-derives size from
            # fov/aspect at render height), but it should say where the pose
            # came from rather than carry a capture canvas it never had.
            v.update(id=str(uuid.uuid4()),
                     label=self._next_view_label(data["views"]),
                     resolution=[fr.get("width", cams["width"]),
                                 fr.get("height", cams["height"])])
            data["views"].append(v)
        write_text_atomic(p, json.dumps(data, indent=1))
        # The mode goes through the ordinary calibration write so the journal
        # cascade fires once, coherently — and remembers what to go back to,
        # since "auto" is not necessarily what this scene was on. The
        # re-render is chained into the SAME job (one GPU job at a time):
        # the converted views become the render — and adjustable in 3D —
        # only when stage 1 reproduces them, and leaving that step to the
        # operator made every convert a two-act ritual with a dead gizmo in
        # between. Job kind "render", so progress and refusals land exactly
        # where a hand-started re-render's would.
        core = s.core
        update = {"render": {"path_mode": "manual",
                             "path_mode_before_manual": mode}}
        from ..render import run_render

        def convert_and_render():
            core.write_calibration(update, "render", "fully-manual views")
            # cfg() re-read AFTER the write: the render must see path_mode
            # "manual", or it would regenerate the auto path one last time.
            run_render(str(core.scene), str(core.workdir), core.cfg(),
                       force=True, should_cancel=core.should_cancel)
        core.mark_entered("render")
        s.start_job("render", convert_and_render)
        s.buffer.publish("journal", s.journal_state())
        return dict(converted=len(auto), views=len(data["views"]),
                    started="render"), 202

    @route("GET", r"/api/runs/([^/]+)/views/reset")
    def views_reset_plan(self, h, m, body):
        """What `reset` would remove — the confirm dialog's numbers, computed
        where the deletion is, so the two cannot disagree."""
        s = self.mgr.session(m.group(1))
        _, data, cams, mode = self._views_state(s)
        rcfg = s.core.cfg().get("render") or {}
        return dict(views=len(data.get("views", [])),
                    frames=len(cams["frames"]) if cams else 0,
                    path_mode=mode,
                    restores_to=rcfg.get("path_mode_before_manual") or "auto",
                    stage1=str(s.core.workdir / "stage1"))

    @route("POST", r"/api/runs/([^/]+)/views/reset")
    def views_reset(self, h, m, body):
        """Start the render step over: every manual view goes, the rendered
        frames go, and the path mode returns to what it was before manual.

        Everything, deliberately — including hand-captured views. A reset that
        left some behind would be a partial undo the operator has to remember,
        and nothing in the 3D view distinguishes a converted view from a
        captured one. Removing ONE view is a different job, already served by
        the frame inspector's delete.
        """
        s = self.mgr.session(m.group(1))
        # Slot check BEFORE anything is deleted: refusing after the rmtree
        # would destroy the views and stage1 — possibly under a live render
        # writing into it — and leave path_mode unrestored.
        s.ensure_job_slot()
        p, data, cams, mode = self._views_state(s)
        rcfg = s.core.cfg().get("render") or {}
        restore = rcfg.get("path_mode_before_manual") or "auto"
        n_views = len(data.get("views", []))
        n_frames = len(cams["frames"]) if cams else 0
        if not n_views and not n_frames:
            raise Refusal("nothing to reset: this scene has no views and no "
                          "render", status=409)
        write_text_atomic(p, json.dumps(dict(version=1, space="ply", views=[]),
                                        indent=1))
        shutil.rmtree(s.core.workdir / "stage1", ignore_errors=True)
        core = s.core
        update = {"render": {"path_mode": restore,
                             "path_mode_before_manual": None}}
        s.start_job("calibration", lambda: core.write_calibration(
            update, "render", "render step reset"), gate="render")
        s.buffer.publish("journal", s.journal_state())
        return dict(views_deleted=n_views, frames_deleted=n_frames,
                    path_mode=restore), 202

    # -- detections: clear ------------------------------------------------------------
    _DETECTION_STAGES = ("stage2", "stage3", "stage4", "stage45")

    @route("GET", r"/api/runs/([^/]+)/detections/reset")
    def detections_reset_plan(self, h, m, body):
        """What clearing the detections would remove — computed where the
        deletion is, so the dialog and the act cannot disagree."""
        s = self.mgr.session(m.group(1))
        dirs = [s.core.workdir / d for d in self._DETECTION_STAGES]
        ix = s.core.workdir / "stage4/interactions.json"
        n_obj = len(_read_json(ix).get("objects", [])) if ix.exists() else 0
        return dict(stages=[str(d) for d in dirs if d.exists()],
                    objects=n_obj,
                    gate="verify_consent")

    @route("POST", r"/api/runs/([^/]+)/detections/reset")
    def detections_reset(self, h, m, body):
        """Clear every detection artefact — the probe, the lift, the export
        and the verification — and re-open the verify-consent gate. An
        explicit act: after a vocabulary change the operator wants
        the canvas and the disk clean, not the previous run's labels
        lingering under the new ones. The render and the gates before it
        stand."""
        s = self.mgr.session(m.group(1))
        s.ensure_job_slot()
        dirs = [s.core.workdir / d for d in self._DETECTION_STAGES]
        if not any(d.exists() for d in dirs):
            raise Refusal("nothing to clear: this scene has no detections",
                          status=409)
        removed = []
        for d in dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))
        demoted, journal = s.reopen_gate("verify_consent")
        log.warning("detections cleared for %r: %s", m.group(1), removed)
        s.buffer.publish("journal", journal)
        return dict(removed=removed, demoted=demoted), 200

    # -- vocabulary / probe ------------------------------------------------------------
    def _draft(self, s) -> dict:
        if s.vocab_draft is None:
            # The draft lived only in server memory, so a restart silently
            # discarded unconfirmed edits and probed-state. It now
            # persists beside the run; the sidecar wins over
            # the profile because it IS the newer operator intent — every
            # mutation (edit, probe, confirm) rewrites it. A corrupt sidecar
            # refuses loudly, naming the file; deleting it re-seeds from the
            # profile.
            saved = _read_json(s.core.workdir / "vocab_draft.json",
                               missing=None)
            d = s.core.cfg()["detect"]
            base = [list(d.get("prompts") or []),
                    list(d.get("negatives") or [])]
            # The sidecar wins ONLY while the profile still matches what it
            # was written against: a draft must not shadow a hand- or
            # terminal-edited profile (content-hash ethos). A moved profile
            # re-seeds, discarding the older draft loudly in the log.
            if (isinstance(saved, dict)
                    and isinstance(saved.get("prompts"), list)
                    and isinstance(saved.get("negatives"), list)
                    and saved.get("profile_detect") == base):
                s.vocab_draft = dict(prompts=saved["prompts"],
                                     negatives=saved["negatives"],
                                     probed=saved.get("probed"))
            else:
                if isinstance(saved, dict):
                    log.info("vocab draft predates a profile change; "
                             "re-seeding from the profile")
                s.vocab_draft = dict(prompts=base[0], negatives=base[1],
                                     probed=None)
        return s.vocab_draft

    @staticmethod
    def _save_draft(s) -> None:
        d = s.core.cfg()["detect"]
        payload = dict(s.vocab_draft,
                       profile_detect=[list(d.get("prompts") or []),
                                       list(d.get("negatives") or [])])
        s.core.workdir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(s.core.workdir / "vocab_draft.json",
                          json.dumps(payload, indent=1))

    @route("GET", r"/api/runs/([^/]+)/vocabulary")
    def vocab_get(self, h, m, body):
        s = self.mgr.session(m.group(1))
        draft = self._draft(s)
        prop = s.core.workdir / "stage15/proposed_vocab.json"
        from ..vlm import backend_available
        _vlm_ok, _vlm_why = backend_available(s.core.cfg())
        return dict(draft,
                    probed_current=(draft["probed"] ==
                                    [draft["prompts"], draft["negatives"]]),
                    llm_proposal=_read_json(prop, missing=None),
                    vlm_available=_vlm_ok, vlm_reason=_vlm_why,
                    cost=self._vocab_cost(s))

    def _vocab_cost(self, s) -> dict | None:
        """The rates the Vocabulary panel prices the list with:
        the views and Gaussians of the render, the seconds one prompt
        costs per view on this card (the last probe's, else a default),
        and the vocabulary the lift's class pass fits. The panel does the
        arithmetic live as the list is edited; None before a render."""
        s1 = _read_json(s.core.workdir / "stage1/manifest.json", missing=None)
        if not s1:
            return None
        views = int(s1.get("views_kept") or 0)
        n = int(s1.get("num_gaussians") or 0)
        pm = _read_json(s.core.workdir / "stage2/probe_manifest.json",
                        missing=None)
        rate, measured = 0.1, False
        # a probe over a handful of prompts is mostly per-frame overhead
        # and would overstate the per-prompt rate: measured from five up
        if (pm and pm.get("inference_s") and pm.get("frames")
                and len(pm.get("prompts") or []) >= 5):
            rate = float(pm["inference_s"]) / (len(pm["frames"]) * len(pm["prompts"]))
            measured = True
        out = dict(views=views, gaussians=n, s_per_prompt_view=round(rate, 4),
                   measured=measured)
        try:
            import torch
            card = torch.cuda.mem_get_info()[1] if torch.cuda.is_available() else None
        except Exception:
            card = None
        if card and n:
            from ..lift import CLASS_PASS_BYTES, CLASS_PASS_FIXED, scene_capacity
            out.update(scene_capacity(n, card),
                       class_pass_bytes=CLASS_PASS_BYTES,
                       class_pass_fixed=CLASS_PASS_FIXED)
        return out

    @route("PUT", r"/api/runs/([^/]+)/vocabulary")
    def vocab_put(self, h, m, body):
        s = self.mgr.session(m.group(1))
        draft = self._draft(s)
        prompts = [str(p).strip() for p in body.get("prompts", [])
                   if str(p).strip()]
        negatives = [str(n).strip() for n in body.get("negatives", [])
                     if str(n).strip()]
        # allow_empty: a DRAFT may be cleared; confirm enforces non-empty
        validate_vocabulary(prompts, negatives, allow_empty=True)
        draft.update(prompts=prompts, negatives=negatives)
        # during_job="allow": a probe job snapshots the draft lists at
        # start, so an edit cannot race it — and blocking would freeze the
        # vocabulary editor for a whole probe.
        s.write(lambda: self._save_draft(s), during_job="allow")
        return dict(draft)

    @route("GET", r"/api/runs/([^/]+)/vocabulary/reset")
    def vocab_reset_plan(self, h, m, body):
        """What resetting the vocabulary would clear — the dialog's numbers,
        computed where the clearing is (the views reset's pattern)."""
        s = self.mgr.session(m.group(1))
        draft = self._draft(s)
        prop = s.core.workdir / "stage15/proposed_vocab.json"
        return dict(prompts=len(draft["prompts"]),
                    negatives=len(draft["negatives"]),
                    proposal=prop.exists(),
                    probe=(s.core.workdir / "stage2").exists())

    @route("POST", r"/api/runs/([^/]+)/vocabulary/reset")
    def vocab_reset(self, h, m, body):
        """Start the vocabulary step over: the draft lists go, the
        vision model's proposal goes, and the gate re-opens. The probe's
        artefacts stay (Clear detections is the act for them); the
        gates before this one stand."""
        s = self.mgr.session(m.group(1))
        s.ensure_job_slot()
        draft = self._draft(s)
        prop = s.core.workdir / "stage15/proposed_vocab.json"
        if not draft["prompts"] and not draft["negatives"] and not prop.exists():
            raise Refusal("nothing to reset: the lists are empty and there "
                          "is no proposal", status=409)
        s.vocab_draft = dict(prompts=[], negatives=[], probed=None)
        self._save_draft(s)
        prop.unlink(missing_ok=True)
        demoted, journal = s.reopen_gate("vocabulary")
        log.warning("vocabulary reset for %r (demoted %s)", m.group(1), demoted)
        s.buffer.publish("journal", journal)
        return dict(demoted=demoted), 200

    @route("POST", r"/api/runs/([^/]+)/vocabulary/propose-llm")
    def vocab_propose(self, h, m, body):
        # One local model call; nothing leaves the machine, so there is
        # no consent flag to carry.
        s = self.mgr.session(m.group(1))
        from ..vlm import run_propose_prompts
        core = s.core
        require_gates_before(core, "vocabulary")   # it reads stage-1 frames
        s.start_job("propose_prompts",
                    lambda: run_propose_prompts(str(core.workdir),
                                                core.cfg(),
                                                core.should_cancel))
        return dict(started="propose_prompts"), 202

    @route("GET", r"/api/runs/([^/]+)/probe/results")
    def probe_results(self, h, m, body):
        import csv
        s = self.mgr.session(m.group(1))
        path = s.core.workdir / "stage2/score_summary.csv"
        if not path.exists():
            raise Refusal("no probe yet", status=404)
        d = s.core.cfg()["detect"]
        with open(path) as f:
            rows = [dict(concept=r["concept"],
                         negative=r["negative_control"] == "True",
                         n_det=int(r["n_det"]), n_frames=int(r["n_frames"]),
                         max=float(r["max_score"]),
                         mean=float(r["mean_score"]))
                    for r in csv.DictReader(f)]
        overlays = sorted(p.name for p in
                          (s.core.workdir / "stage2/overlays").glob("*.png")
                          ) if (s.core.workdir / "stage2/overlays").exists() \
            else []
        # exemplar frame cap — the configured value + what the last probe
        # actually used + its measured peak VRAM, so the exemplar gate can
        # surface the VRAM knob (config-only; not a UI-editable value).
        excap = dict(configured=d.get("exemplar_max_frames"),
                     profile=s.core.cfg().get("profile"))
        manp = s.core.workdir / "stage2/probe_manifest.json"
        if manp.exists():
            man = _read_json(manp)
            excap.update(
                used=man.get("exemplar_frames_used"),
                total=man.get("exemplar_frames_total"),
                peak_alloc_gb=man.get("exemplar_peak_alloc_gb"))
        return dict(
            concepts=rows,
            exemplar_frame_cap=excap,
            negative_ceiling=s.core.negative_ceiling(),
            thresholds=dict(
                presence=d["presence_threshold"],
                exemplar=d.get("exemplar_threshold"),
                presence_overrides=d.get("presence_threshold_overrides")
                or {},
                exemplar_overrides=d.get("exemplar_threshold_overrides")
                or {}),
            overlays=overlays,
            distribution_png=(s.core.workdir /
                              "stage2/score_distribution.png").exists())

    @route("POST", r"/api/runs/([^/]+)/vocabulary/confirm")
    def vocab_confirm(self, h, m, body):
        s = self.mgr.session(m.group(1))
        require_gates_before(s.core, "vocabulary")   # gates in order
        draft = self._draft(s)
        # fail fast, before the job slot is taken — confirm_vocabulary
        # enforces the same rules again inside the job
        validate_vocabulary(draft["prompts"], draft["negatives"])
        # No way around the probe: the gate IS the
        # review of the scores, and the "confirm anyway" escape recorded
        # nothing. Detection would have run at confirmation regardless.
        if draft["probed"] != [draft["prompts"], draft["negatives"]]:
            raise Refusal("these exact lists were not probed; Run probe, "
                          "review the scores, then confirm",
                          gate="vocabulary", remedy="run the probe")
        prompts = list(draft["prompts"])
        negatives = list(draft["negatives"])
        core = s.core

        def confirm():
            core.mark_entered("vocabulary")
            core.confirm_vocabulary(prompts, negatives)
            self._save_draft(s)   # sidecar tracks the confirmed state
        s.start_job("confirm_vocabulary", confirm)
        return dict(started="confirm_vocabulary"), 202

    # -- exemplars ----------------------------------------------------------------------
    def _exemplars_file(self, s):
        from ..exemplars import resolve_exemplars_path
        p = resolve_exemplars_path(s.core.workdir, s.core.cfg())
        data = _read_json(p, missing=dict(version=1, exemplars=[]))
        return p, data

    def _exemplars_probed(self, s) -> tuple[bool, int]:
        """(the crops on disk are the ones the probe saw, their count):
        the stage-2 manifest's exemplars hash against the live file — the
        same key detect.py folds on. No crops: nothing to probe, True."""
        from ..sequencing import fingerprint
        p, data = self._exemplars_file(s)
        n = sum(len(e.get("crops") or []) for e in data.get("exemplars", []))
        if not n:
            return True, 0
        man = _read_json(s.core.workdir / "stage2/probe_manifest.json",
                         missing=None) or {}
        return man.get("exemplars_hash", "absent") == fingerprint(p), n

    @route("GET", r"/api/runs/([^/]+)/exemplars")
    def exemplars_get(self, h, m, body):
        s = self.mgr.session(m.group(1))
        _, data = self._exemplars_file(s)
        d = s.core.cfg()["detect"]
        probed, _n = self._exemplars_probed(s)
        return dict(data,
                    threshold=d.get("exemplar_threshold"),
                    overrides=d.get("exemplar_threshold_overrides") or {},
                    probed_current=probed)

    @route("POST", r"/api/runs/([^/]+)/exemplars/crops")
    def exemplars_crops(self, h, m, body):
        # The crop-drawing save path (the only one): the SERVER stamps
        # frame_sha256 from the
        # stage-1 file (detect refuses stale crops after a re-render; the
        # client never hashes).
        s = self.mgr.session(m.group(1))
        concept = (body.get("concept") or "").strip()
        crops = body.get("crops") or []
        if not concept or not crops:
            raise Refusal("need concept + crops[{frame_idx, box_xyxy}]",
                          status=400)

        def save():
            cams = _read_json(s.core.workdir / "stage1/cameras.json")
            frames = {f["frame_idx"]: f for f in cams["frames"]}
            p, data = self._exemplars_file(s)
            entry = next((e for e in data["exemplars"]
                          if e["concept"] == concept), None)
            if entry is None:
                entry = dict(concept=concept, crops=[])
                data["exemplars"].append(entry)
            if body.get("replace"):
                entry["crops"] = []
            for c in crops:
                fr = frames.get(int(c["frame_idx"]))
                if fr is None:
                    raise Refusal(f"frame {c['frame_idx']} not in stage-1 "
                                  f"set", status=400)
                box = [int(x) for x in c["box_xyxy"]]
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    raise Refusal(f"malformed box_xyxy: {c['box_xyxy']}",
                                  status=400)
                entry["crops"].append(dict(
                    frame_idx=int(c["frame_idx"]), box_xyxy=box,
                    frame_sha256=fingerprint(
                        s.core.workdir / "stage1" / fr["file"]),
                    note=str(fr.get("label",
                                    fr.get("provenance", "auto")))))
            p.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(p, json.dumps(data, indent=1))
            s.core.mark_entered("exemplars")
            return dict(concept=concept, n_crops=len(entry["crops"]))
        return s.write(save, gate="exemplars"), 201

    @route("DELETE", r"/api/runs/([^/]+)/exemplars/([^/]+)(?:/crops/(\d+))?")
    def exemplars_delete(self, h, m, body):
        s = self.mgr.session(m.group(1))
        concept, crop_idx = m.group(2), m.group(3)

        def delete():
            p, data = self._exemplars_file(s)
            entry = next((e for e in data["exemplars"]
                          if e["concept"] == concept), None)
            if entry is None:
                raise Refusal(f"no exemplar concept {concept!r}", status=404)
            if crop_idx is None:
                data["exemplars"].remove(entry)
            else:
                i = int(crop_idx)
                if i >= len(entry["crops"]):
                    raise Refusal(f"crop {i} out of range", status=404)
                del entry["crops"][i]
                if not entry["crops"]:
                    data["exemplars"].remove(entry)
            write_text_atomic(p, json.dumps(data, indent=1))
            return dict(deleted=concept if crop_idx is None
                        else f"{concept}/crops/{crop_idx}")
        return s.write(delete, gate="exemplars")

    @route("PUT", r"/api/runs/([^/]+)/exemplars/thresholds")
    def exemplars_thresholds(self, h, m, body):
        s = self.mgr.session(m.group(1))
        core = s.core
        over = dict(core.cfg()["detect"].get(
            "exemplar_threshold_overrides") or {})
        for k, v in (body or {}).items():
            try:
                over[str(k)] = float(v)
            except (TypeError, ValueError):
                raise Refusal(f"threshold for {k!r} is not a number: {v!r}",
                              status=400)
        # A calibration write like any other: the profile is written as a
        # job (serialised against any live stage) and the exemplars gate
        # and everything after it re-open — it used to restamp every gate
        # and re-open none, so the review stood over rows that changed.
        update = dict(detect=dict(exemplar_threshold_overrides=over))
        affected = core.calibration_gate(update)
        s.start_job("threshold_override",
                    lambda: core.write_calibration(
                        update, affected, "exemplar threshold override"),
                    gate=affected)
        return dict(started="threshold_override", affected_gate=affected), 202

    # -- verify / report / results --------------------------------------------------------
    @route("GET", r"/api/runs/([^/]+)/verify/preflight")
    def verify_preflight(self, h, m, body):
        s = self.mgr.session(m.group(1))
        ix = s.core.workdir / "stage4/interactions.json"
        if not ix.exists():
            raise Refusal("no stage-4 export yet; run the pipeline first",
                          gate="verify_consent", status=404)
        n = len(_read_json(ix)["objects"])
        from ..vlm import backend_available, describe_model
        from ..verify import (previous_verdicts, read_checkpoint,
                              requested_instances, verify_cached)
        ok, why = backend_available(s.core.cfg())
        wd, cfg = str(s.core.workdir), s.core.cfg()
        # the kept verification, when still fresh — Run verification
        # would replay it at once; the panel says so before the press.
        kept = verify_cached(wd, cfg)
        marked = len(requested_instances(wd))                # the marks
        done = 0 if kept else len(read_checkpoint(wd, cfg))  # the checkpoint
        return dict(instances=n, est_calls=max((marked or n) - done, 0),
                    marked=marked,
                    # the unmarked carry these when any are marked
                    previous_verdicts=bool(previous_verdicts(wd)),
                    # judged before the run stopped; the next press
                    # continues from there
                    checkpoint=dict(judged=done) if done else None,
                    model=cfg["vlm"]["model"],
                    vlm_available=ok, vlm_reason=why,
                    already_verified=(dict(model=kept.get("model"),
                                           elapsed_s=kept.get("elapsed_s"))
                                      if kept else None),
                    **describe_model(s.core.cfg()))

    @route("POST", r"/api/runs/([^/]+)/verify/consent")
    def verify_consent(self, h, m, body):
        s = self.mgr.session(m.group(1))
        if not isinstance(body.get("consent"), bool):
            raise Refusal("body must carry consent: true|false", status=400)
        require_gates_before(s.core, "verify_consent")   # gates in order
        core = s.core
        # An answer already recorded and still fresh stands: the job acts
        # on the journal's `run`, so a different answer pressed now would
        # be silently overridden by the recorded one (Skip recorded, Run
        # pressed: the labels shipped unverified with no refusal). The
        # same answer replays the kept result; the other needs the
        # explicit act that re-opens the gate.
        st = core.gate_states().get("verify_consent")
        if st and st.get("state") == "approved" \
                and bool(st.get("run")) != body["consent"]:
            raise Refusal(
                f"the verify gate already recorded "
                f"{'RUN' if st.get('run') else 'SKIP'} "
                f"({st.get('approved_at')}) and nothing changed since; "
                f"reopen the verify gate to answer the other way.",
                gate="verify_consent")
        s.pending_consent = body["consent"]

        def verify_and_report():
            try:
                core.gate_verify()
                core.write_report()
            finally:
                s.pending_consent = None
        s.start_job("verify", verify_and_report)
        return dict(started="verify"), 202

    @route("GET", r"/api/runs/([^/]+)/report")
    def report(self, h, m, body):
        s = self.mgr.session(m.group(1))
        rj = s.core.workdir / "report.json"
        rm = s.core.workdir / "run_report.md"
        return dict(
            report=_read_json(rj, missing=None),
            markdown=rm.read_text() if rm.exists() else None)

    @route("GET", r"/api/runs/([^/]+)/instances")
    def instances(self, h, m, body):
        # (the helper _proposed_label is module-level, below the class)
        # Derived on GET from interactions.json — no stage change:
        # the artifact already carries the data.
        s = self.mgr.session(m.group(1))
        ix_p = s.core.workdir / "stage4/interactions.json"
        if not ix_p.exists():
            raise Refusal("no stage-4 export yet", status=404)
        ix = _read_json(ix_p)
        ext = ix.get("extended", {}).get("objects", [])
        out = []
        for i, o in enumerate(ix["objects"]):
            e = ext[i] if i < len(ext) else {}
            srcs = list((e.get("support_by_source") or {}).keys())
            out.append(dict(
                idx=i, label=o["label"],
                verified_label=e.get("verified_label"),
                verdict=e.get("verify_verdict"),
                # the verifier's own sentence (a relabel it could not apply
                # names the label it proposed) — the object card shows it
                verify_rationale=e.get("verify_rationale"),
                conf=e.get("aggregate_confidence", 0),
                scale=("exemplar" if srcs and all(
                    x.startswith("exemplar:") for x in srcs) else "text"),
                position=o["position"], scale_xyz=o["scale"],
                obb=e.get("obb"), instance_id=e.get("instance_id"),
                # the export writes `gaussian_count` (this read the wrong
                # key since the route was written; the card never showed it)
                gaussians=e.get("gaussian_count"),
                support_by_source=e.get("support_by_source") or {},
                # §6: the drawer badges holds from the export-time reasons,
                # not from the verdict, so a skipped verification never
                # hides them
                hold_reasons=e.get("hold_reasons") or [],
                held_original_verdict=e.get("held_original_verdict"),
                # the operator's own name for it, beside the
                # detected and the verified ones, never over them
                operator_label=e.get("operator_label"),
                # the name the verifier proposed, applied or not
                proposed_label=_proposed_label(e),
                # marked for verification
                verify_requested=bool(e.get("verify_requested"))))
        return dict(objects=out,
                    verification=ix.get("extended", {}).get("verification"))

    @route("PUT", r"/api/runs/([^/]+)/instances/(\d+)/label")
    def instance_label(self, h, m, body):
        """The operator names an object: `extended.objects[i]
        .operator_label` in interactions.json, with when. The detected
        label stays canonical (classes, colours, the ply's name until the
        next export) and the verifier's stays its own; the viewer and the
        report show the operator's first. An empty label takes it back.
        An explicit act on a stage output — never inferred, never a
        gate."""
        s = self.mgr.session(m.group(1))
        idx = int(m.group(2))
        label = " ".join(str((body or {}).get("label") or "").split())
        ix_p = s.core.workdir / "stage4/interactions.json"

        def relabel():
            if not ix_p.exists():
                raise Refusal("no stage-4 export yet", status=404)
            ix = _read_json(ix_p)
            objs = ix.get("objects") or []
            if not 0 <= idx < len(objs):
                raise Refusal(f"no object {idx} (the export has "
                              f"{len(objs)})", status=404)
            ext = ix.setdefault("extended", {}).setdefault("objects", [])
            while len(ext) <= idx:
                ext.append({})
            if label:
                ext[idx]["operator_label"] = label
                ext[idx]["operator_label_at"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S")
            else:
                ext[idx].pop("operator_label", None)
                ext[idx].pop("operator_label_at", None)
            write_text_atomic(ix_p, json.dumps(ix, indent=1))
            log.info("object %d %s: %r (detected %r)", idx,
                     "named" if label else "name taken back", label,
                     objs[idx].get("label"))
            return dict(idx=idx, label=objs[idx].get("label"),
                        operator_label=label or None)
        return s.write(relabel, gate="verify_consent")

    @route("PUT", r"/api/runs/([^/]+)/instances/(\d+)/verify_mark")
    def instance_verify_mark(self, h, m, body):
        """The operator marks an object for verification:
        `extended.objects[i].verify_requested` in interactions.json, with
        when; `marked: false` takes it back. While any object is marked,
        Run verification judges the marked ones only and the rest carry
        their previous verdicts on this export, or ship not judged. An
        explicit act on a stage output — never inferred, never a gate; the
        marks sit in the stage's fingerprint, so a changed set is a fresh
        run, never a replay."""
        s = self.mgr.session(m.group(1))
        idx = int(m.group(2))
        marked = bool((body or {}).get("marked"))
        ix_p = s.core.workdir / "stage4/interactions.json"

        def mark():
            if not ix_p.exists():
                raise Refusal("no stage-4 export yet", status=404)
            ix = _read_json(ix_p)
            objs = ix.get("objects") or []
            if not 0 <= idx < len(objs):
                raise Refusal(f"no object {idx} (the export has "
                              f"{len(objs)})", status=404)
            ext = ix.setdefault("extended", {}).setdefault("objects", [])
            while len(ext) <= idx:
                ext.append({})
            if marked:
                ext[idx]["verify_requested"] = True
                ext[idx]["verify_requested_at"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S")
            else:
                ext[idx].pop("verify_requested", None)
                ext[idx].pop("verify_requested_at", None)
            write_text_atomic(ix_p, json.dumps(ix, indent=1))
            n = sum(1 for e in ext if e.get("verify_requested"))
            log.info("object %d %s for verification (%d marked)", idx,
                     "marked" if marked else "unmarked", n)
            return dict(idx=idx, verify_requested=marked, marked=n)
        return s.write(mark, gate="verify_consent")

    @route("DELETE", r"/api/runs/([^/]+)/verify_marks")
    def clear_verify_marks(self, h, m, body):
        """Every mark taken back at once: Run verification judges
        every instance again."""
        s = self.mgr.session(m.group(1))
        ix_p = s.core.workdir / "stage4/interactions.json"

        def clear():
            if not ix_p.exists():
                raise Refusal("no stage-4 export yet", status=404)
            ix = _read_json(ix_p)
            ext = ix.get("extended", {}).get("objects", [])
            n = 0
            for e in ext:
                n += bool(e.pop("verify_requested", None))
                e.pop("verify_requested_at", None)
            write_text_atomic(ix_p, json.dumps(ix, indent=1))
            log.info("verification marks cleared (%d)", n)
            return dict(cleared=n)
        return s.write(clear, gate="verify_consent")

    @route("GET", r"/api/runs/([^/]+)/interactions.json")
    def interactions(self, h, m, body):
        s = self.mgr.session(m.group(1))
        return ("file", s.core.workdir / "stage4/interactions.json",
                "application/json")

    @route("GET", r"/api/runs/([^/]+)/instance_ids.bin")
    def instance_ids(self, h, m, body):
        s = self.mgr.session(m.group(1))
        npy = s.core.workdir / "stage3/instances.npy"
        if not npy.exists():
            raise Refusal("pre-lift: no instance ids yet", status=404)
        inst = np.load(npy)
        if inst.max() >= 65535:
            raise Refusal("more than 65534 instances; widen "
                          "instance_ids.bin", status=500)
        return ("bytes", np.where(inst < 0, 65535, inst)
                .astype("<u2").tobytes(), "application/octet-stream")

    @route("GET", r"/api/runs/([^/]+)/scene_source")
    def scene_source(self, h, m, body):
        """Which container this scene is in — the workbench canvas needs it
        to pick Spark's loader, exactly as the static viewer reads it from
        the staged data/scene_source.json."""
        s = self.mgr.session(m.group(1))
        suffix = s.core.scene.suffix.lower()
        return dict(file=f"scene{suffix}", format=suffix.lstrip("."))

    @route("GET", r"/api/runs/([^/]+)/scene\.([a-z0-9]+)")
    def scene_file(self, h, m, body):
        s = self.mgr.session(m.group(1))
        if f".{m.group(2)}" != s.core.scene.suffix.lower():
            raise Refusal(
                f"scene {m.group(1)!r} is {s.core.scene.name}, not "
                f"scene.{m.group(2)}; ask /scene_source which to fetch",
                status=404)
        return ("file", s.core.scene, "application/octet-stream")

    @route("GET", r"/api/runs/([^/]+)/density.json")
    def density(self, h, m, body):
        s = self.mgr.session(m.group(1))
        return ("file", s.core.workdir / "density.json",
                "application/json")

    @route("GET", r"/api/runs/([^/]+)/viewer_settings")
    def viewer_settings_get(self, h, m, body):
        s = self.mgr.session(m.group(1))
        p = s.core.workdir / "viewer_settings.json"
        return _read_json(p, missing={})

    @route("PUT", r"/api/runs/([^/]+)/viewer_settings")
    def viewer_settings_put(self, h, m, body):
        s = self.mgr.session(m.group(1))
        # Cosmetic file no stage reads — during_job="allow", because the
        # settings panel saves while renders run and no job can race it.
        def save():
            s.core.workdir.mkdir(parents=True, exist_ok=True)
            write_text_atomic(s.core.workdir / "viewer_settings.json",
                              json.dumps(body, indent=1))
            return dict(saved=True)
        return s.write(save, during_job="allow")

    @route("GET", r"/api/runs/([^/]+)/artifacts/(.+)")
    def artifacts(self, h, m, body):
        s = self.mgr.session(m.group(1))
        rel = m.group(2)
        for root in (s.core.workdir,):   # the scene's files live here too
            p = (root / rel).resolve()
            if not str(p).startswith(str(root.resolve()) + os.sep):
                raise Refusal("path traversal refused", status=400)
            if p.exists() and p.is_file():
                ctype = ("image/png" if p.suffix == ".png" else
                         "application/json" if p.suffix == ".json" else
                         "text/plain")
                return ("file", p, ctype)
        raise Refusal(f"artifact not found: {rel}", status=404)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    api: Api = None   # set by serve()

    def log_message(self, fmt, *args):
        log.debug("%s " + fmt, self.address_string(), *args)

    # -- request-origin guard ------------------------------------------------------
    @staticmethod
    def _hostname(value: str) -> str:
        """Strip the port from a Host header value (IPv6 literals keep their
        brackets: '[::1]:8090' -> '[::1]')."""
        if value.startswith("["):
            return value.split("]", 1)[0] + "]"
        return value.rsplit(":", 1)[0] if ":" in value else value

    def _local_request(self) -> str | None:
        """None when the request may proceed, else the reason it may not.

        Both headers are checked only when PRESENT: curl and scripts send no
        Origin and carry no ambient credentials, so they are not the threat
        this guards against — a browser is."""
        host = self.headers.get("Host")
        if host and self._hostname(host) not in LOCAL_HOSTNAMES:
            return (f"Host {host!r} is not localhost; `carveout web` serves "
                    f"127.0.0.1 only (DNS rebinding refused)")
        origin = self.headers.get("Origin")
        if origin and origin != "null":
            from urllib.parse import urlparse
            if (urlparse(origin).hostname or "") not in LOCAL_HOSTNAMES:
                return (f"cross-origin request from {origin!r} refused; this "
                        f"server is driven by its own UI on localhost")
        return None

    # -- plumbing ---------------------------------------------------------------
    def _json_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n == 0:
            return {}
        if n > 5_000_000:
            raise Refusal("body too large", status=413)
        # A form post is the one cross-origin write a browser can make without
        # a CORS preflight, and it cannot set this content type.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if ctype.lower() != "application/json":
            raise Refusal(f"Content-Type must be application/json (got "
                          f"{ctype or 'none'!r})", status=415)
        try:
            return json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            raise Refusal("body is not valid JSON", status=400) from None

    def _send_json(self, obj, status=200):
        blob = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def _send_file(self, path: Path, ctype: str):
        path = Path(path)
        if not path.exists():
            self._send_json(dict(error=f"not found: {path.name}"), 404)
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        # cache lessons learned the hard way: identities change under one URL
        self.send_header("Cache-Control",
                         "no-cache" if path.suffix.lower() in SPLAT_SUFFIXES
                         else "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, length=1024 * 1024)

    def _send_sse(self, session):
        last = 0
        try:
            last = int(self.headers.get("Last-Event-ID") or 0)
        except ValueError:
            pass
        q, replay = session.buffer.subscribe(last)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def emit(seq, etype, payload):
                self.wfile.write(
                    (f"id: {seq}\nevent: {etype}\n"
                     f"data: {json.dumps(payload)}\n\n").encode())
            # re-attach: current state first, then the missed events (a
            # reconnect) or the trimmed replay (a fresh page) — EventBuffer
            emit(last, "journal", session.journal_state())
            for seq, etype, payload in replay:
                emit(seq, etype, payload)
            self.wfile.flush()
            import queue as _q
            while True:
                try:
                    seq, etype, payload = q.get(timeout=15)
                    emit(seq, etype, payload)
                except _q.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            session.buffer.unsubscribe(q)

    def _dispatch(self, method: str):
        path = self.path.split("?", 1)[0]
        refused = self._local_request()
        if refused:
            log.warning("refused %s %s: %s", method, path, refused)
            self._send_json(dict(error=refused, gate=None, remedy=None), 403)
            return
        try:
            for meth, pattern, fname in _ROUTES:
                if meth != method:
                    continue
                m = pattern.match(path)
                if not m:
                    continue
                body = (self._json_body()
                        if method in ("POST", "PUT", "PATCH", "DELETE")
                        else {})
                result = getattr(self.api, fname)(self, _Match(m), body)
                status = 200
                if isinstance(result, tuple) and isinstance(result[0], str):
                    kind = result[0]
                    if kind == "file":
                        self._send_file(result[1], result[2])
                        return
                    if kind == "bytes":
                        blob = result[1]
                        self.send_response(200)
                        self.send_header("Content-Type", result[2])
                        self.send_header("Content-Length", str(len(blob)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(blob)
                        return
                    if kind == "sse":
                        self._send_sse(result[1])
                        return
                if isinstance(result, tuple):
                    result, status = result
                self._send_json(result, status)
                return
            if method == "GET" and not path.startswith("/api/"):
                self._send_static(path)
                return
            self._send_json(dict(error=f"no route: {method} {path}"), 404)
        except WorkdirLocked as e:
            rec = e.rec
            self._send_json(dict(
                error=str(e), holder=rec.get("holder"), pid=rec.get("pid"),
                since=rec.get("since"), stale=e.stale), 423)
        except Refusal as e:
            self._send_json(dict(error=str(e), gate=e.gate,
                                 remedy=e.remedy), e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log.exception("%s %s failed", method, path)
            self._send_json(dict(error=f"{type(e).__name__}: {e}"), 500)

    def _send_static(self, path: str):
        if not DIST.exists():
            msg = (f"webui bundle not built.\n\nRun:\n  {BUILD_CMD}\n\n"
                   f"then reload. (Node is a BUILD-TIME dependency only; "
                   f"`carveout web` itself needs no Node.)")
            blob = msg.encode()
            self.send_response(503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)
            return
        rel = path.lstrip("/") or "index.html"
        p = (DIST / rel).resolve()
        if not str(p).startswith(str(DIST.resolve()) + os.sep) and p != DIST.resolve():
            self._send_json(dict(error="path traversal refused"), 400)
            return
        if not p.is_file():
            p = DIST / "index.html"    # SPA fallback
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript", ".css": "text/css",
                 ".svg": "image/svg+xml", ".png": "image/png",
                 ".woff2": "font/woff2",
                 ".json": "application/json"}.get(p.suffix,
                                                  "application/octet-stream")
        self._send_file(p, ctype)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")


def serve(port: int, cfg_loader_factory) -> None:
    """Entry point for `carveout web` — binds 127.0.0.1 only."""
    manager = RunManager(REPO, cfg_loader_factory)
    Handler.api = Api(manager)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    with Server(("127.0.0.1", port), Handler) as httpd:
        dist = "" if DIST.exists() else (
            f"\n  frontend bundle NOT BUILT; run: {BUILD_CMD}")
        print(f"\nCarveout web: http://127.0.0.1:{port}/  "
              f"(Ctrl-C to stop){dist}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop_all()
            manager.release_all()


_LEGACY_PROPOSED = re.compile(r"The verifier suggested '([^']+)'")


def _proposed_label(e: dict) -> str | None:
    """The name the verifier proposed for an object, whatever became of it
    `proposed_label` since the field was added; before that, the held
    records carried it as `held_proposed_label` and the plain
    outside-vocabulary / unsupported / too-large ones only inside the
    verifier's own sentence — read back from that sentence, ours, for the
    exports written before the field existed. None when there is none."""
    p = e.get("proposed_label") or e.get("held_proposed_label")
    if p:
        return p
    m = _LEGACY_PROPOSED.search(e.get("verify_rationale") or "")
    return m.group(1) if m else None
