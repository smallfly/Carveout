# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Staged renders: the small on-disk protocol that lets a re-render be held
back from publishing, and published later as an explicit operator act.

Deliberately free of torch and of every heavy import, like `quality`: the web
server reads this on the scene-open path, and `carveout web` starts with no
CUDA stack loaded — torch is imported when a stage runs. Everything here is
renames and small JSON files.
"""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Staged renders. A render is written to frames.partial/ and swapped over the
# live frames at the end. The swap used to fire whenever ANY view survived —
# but manual views bypass the quality filter by design, and every real scene
# has some, so the survivor count is never zero and the "nothing survived"
# guard was unreachable: a threshold experiment that killed every AUTO view
# still published its 3 manual survivors over a good 24-view render. When the
# new render is unhealthy and the one it would replace was not, the whole
# render is left staged instead, and publishing it becomes an explicit
# operator act — 5 renames, no GPU, nothing recomputed.
STAGED_DESCRIPTOR = "staged_render.json"
CAMERAS_PARTIAL = "cameras.partial.json"
MANIFEST_PARTIAL = "manifest.partial.json"
SHEET_PARTIAL = "review_gate.partial.png"
_STAGED_SIDECARS = (SHEET_PARTIAL, CAMERAS_PARTIAL, MANIFEST_PARTIAL,
                    STAGED_DESCRIPTOR)


def render_health(manifest: dict | None, cfg: dict) -> dict:
    """Is a finished render healthy? Defined ONCE, here, from the two numbers
    the render gate already shows: views kept and final coverage. Absent
    coverage (no volume, so the metric never ran) is not a failure — the gate
    treats it the same way."""
    h = cfg["render"]["health"]
    kept = (manifest or {}).get("views_kept")
    cov = ((manifest or {}).get("coverage") or {}).get("frac_final")
    ok = (kept is not None and kept >= h["min_views"]
          and (cov is None or cov >= h["min_coverage"]))
    return dict(ok=bool(ok), views_kept=kept, coverage=cov,
                min_views=h["min_views"], min_coverage=h["min_coverage"])


def _health_line(h: dict) -> str:
    cov = "n/a" if h["coverage"] is None else f"{100 * h['coverage']:.1f}%"
    return f"{h['views_kept']} views kept, coverage {cov}"


def staged_render(workdir: str) -> dict | None:
    """The staged-but-unpublished render's descriptor, or None. Read-only:
    every caller that decides anything uses this rather than probing dirs."""
    p = Path(workdir) / "stage1" / STAGED_DESCRIPTOR
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def publish_staged_render(workdir: str) -> dict:
    """Publish the staged render over the live one: pure renames, no GPU.

    Refuses rather than half-publishing if any piece is missing. Rewrites
    cameras.json and manifest.json, so the render gate's pinned hash changes
    and it — with every gate after it — demotes on the next evaluation. That
    is the point: this is a new render being published, not a resume."""
    out = Path(workdir) / "stage1"
    if staged_render(workdir) is None:
        raise RuntimeError(
            f"nothing to publish: no staged render at {out / STAGED_DESCRIPTOR}")
    need = [out / "frames.partial", out / "depth.partial",
            *(out / n for n in _STAGED_SIDECARS)]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise RuntimeError(
            "staged render is incomplete, refusing to publish a partial "
            "swap; missing: " + ", ".join(missing)
            + ". Discard it and re-render.")
    for live, stg in ((out / "frames", out / "frames.partial"),
                      (out / "depth", out / "depth.partial")):
        if live.exists():
            shutil.rmtree(live)
        stg.rename(live)
    (out / SHEET_PARTIAL).rename(out / "review_gate.png")
    (out / CAMERAS_PARTIAL).rename(out / "cameras.json")
    (out / MANIFEST_PARTIAL).rename(out / "manifest.json")
    (out / STAGED_DESCRIPTOR).unlink()
    manifest = json.loads((out / "manifest.json").read_text())
    log.warning("staged render PUBLISHED: %d views kept; the render gate's "
                "inputs changed, so it and every gate after it demote",
                manifest.get("views_kept", -1))
    return manifest


def discard_staged_render(workdir: str, quiet: bool = False) -> bool:
    """Drop the staged render and its sidecars. The published render is never
    touched. Returns whether anything was actually staged."""
    out = Path(workdir) / "stage1"
    had = staged_render(workdir) is not None
    for d in (out / "frames.partial", out / "depth.partial"):
        if d.exists():
            shutil.rmtree(d)
    for n in _STAGED_SIDECARS:
        (out / n).unlink(missing_ok=True)
    if had and not quiet:
        log.warning("staged render DISCARDED; the published render at %s "
                    "is untouched", out / "frames")
    return had
