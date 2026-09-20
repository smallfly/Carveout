# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Volume of interest: axis-aligned boxes bounding the labelable scene region.

Format (data/scenes/<name>/volume.json, raw .ply world coordinates):
{
  "scene":  "my-scene.ply",
  "coords": "ply_world",
  "boxes":  [{"name": "main_room", "min": [x,y,z], "max": [x,y,z]}]
}

Auto-proposal derives boxes from the interior region of the occupancy grid
(cells enclosed by walls); it is a starting point meant to be hand-corrected.
Camera placement, the coverage metric, and Stage 3/4 labeling scope are all
restricted to in-volume Gaussians.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from .cameras import SceneFrame, check_grid_budget, footprint_dims
from .refusal import Refusal
from .sequencing import check_cancel, write_text_atomic


def resolve_volume_path(workdir, cfg: dict) -> Path:
    """render.volume override, else <workdir>/volume.json (the scene's files
    live in the workdir — sequencing.SCENE_FILES)."""
    return Path(cfg["render"].get("volume") or Path(workdir) / "volume.json")

log = logging.getLogger(__name__)


WHOLE_SCENE = "whole_scene"


def volume_scope(path: str | Path) -> str | None:
    """What volume.json decides: "boxes", "whole_scene" (the explicit act —
    no boxes, every Gaussian in scope, placement over the scene's robust
    bounds), or None when there is no file, which is the unrun state and
    what the render warns about."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return WHOLE_SCENE if data.get("scope") == WHOLE_SCENE else "boxes"


def load_volume(path: str | Path) -> list[dict] | None:
    """The boxes, or None for the whole-scene scope. A file with neither a
    box nor the scope is refused: an empty box list used to load as "no
    Gaussian anywhere" and silently empty every stage."""
    data = json.loads(Path(path).read_text())
    # `scene`: the boxes are in the levelled frame, which is the
    # frame every geometric decision uses; the alignment they were made
    # under rides beside them and the profile's is what applies now.
    if data.get("coords", "ply_world") not in ("ply_world", "scene"):
        raise ValueError(f"volume {path}: unsupported coords {data.get('coords')!r}")
    if data.get("scope") == WHOLE_SCENE:
        log.info("volume scope: the whole scene (an explicit act), from %s", path)
        return None
    boxes = data["boxes"]
    if not boxes:
        raise ValueError(
            f"volume {path} has no boxes and no scope; draw a box at the "
            f"volume gate, or take the whole-scene act there")
    for b in boxes:
        b["min"], b["max"] = np.asarray(b["min"], float), np.asarray(b["max"], float)
        if not (b["min"] < b["max"]).all():
            raise ValueError(f"volume box {b.get('name')}: min must be < max per axis")
    log.info("volume loaded: %d box(es) from %s", len(boxes), path)
    return boxes


def volume_alignment(path: str | Path) -> dict | None:
    """The levelling a volume.json's numbers were written under;
    None for the file's axes (or no file)."""
    p = Path(path)
    if not p.exists():
        return None
    return (json.loads(p.read_text()) or {}).get("alignment")


STALE_VOLUME = ("these boxes were proposed before the scene was levelled the "
                "way it is now, so their numbers describe another frame; "
                "press Propose fresh at the volume gate (or take the "
                "whole-scene act), then approve")


def stale_volume_reason(path: str | Path, cfg: dict) -> str | None:
    """The sentence when the volume's recorded levelling is not the
    profile's: a box drawn on one frame re-read in another cuts the
    room wrong — the garage's floor fell outside its box, a bench won the
    floor vote and every camera stood in the ceiling. None
    when they agree, or there is no volume."""
    from .alignment import alignment_block, same_block
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text()) or {}
    if data.get("scope") == WHOLE_SCENE:
        return None          # no numbers: the scene's own bounds, any frame
    current = alignment_block((cfg.get("scene") or {}).get("alignment"))
    # The frame the numbers were MADE in is the proposal's record
    # (density.json, written with the boxes); volume.json's own stamp is
    # what the last write put there — a box edit from the canvas stamps
    # the levelling of the moment over numbers it did not re-derive.
    dens = p.parent / "density.json"
    made_in = data.get("alignment")
    if dens.exists():
        try:
            made_in = (json.loads(dens.read_text()) or {}).get("alignment")
        except json.JSONDecodeError:
            pass
    return None if same_block(made_in, current) else STALE_VOLUME


def in_volume(points: np.ndarray, boxes: list[dict],
              margin: float = 0.0) -> np.ndarray:
    """Membership in the union of boxes, optionally dilated by `margin` (m).

    The margin separates the volume's two meanings:
    camera placement / coverage use the volume as drawn (margin 0), while
    Gaussian label scoping uses a soft boundary so wall-mounted objects that
    sit ON the trimmed boundary are not truncated.
    """
    mask = np.zeros(len(points), dtype=bool)
    for b in boxes:
        mask |= ((points >= b["min"] - margin) &
                 (points <= b["max"] + margin)).all(axis=1)
    return mask


def footprint_density(means: np.ndarray, opacities: np.ndarray,
                      frame: SceneFrame, vox: float,
                      ) -> tuple[np.ndarray, float, float]:
    """Top-down opacity-density grid over the robust footprint."""
    a0, a1 = frame.ground_axes
    n0, n1, lo0, lo1 = footprint_dims(frame, vox)
    i0 = np.clip(((means[:, a0] - lo0) / vox).astype(int), 0, n0 - 1)
    i1 = np.clip(((means[:, a1] - lo1) / vox).astype(int), 0, n1 - 1)
    density2d = np.zeros((n0, n1))
    np.add.at(density2d, (i0, i1), opacities)
    return density2d, lo0, lo1


def _components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4-neighbour flood fill; returns (labels 1..K, sizes[K])."""
    n0, n1 = mask.shape
    labels = np.zeros((n0, n1), dtype=int)
    comp = 0
    for a in range(n0):
        for b in range(n1):
            if mask[a, b] and labels[a, b] == 0:
                comp += 1
                stack = [(a, b)]
                labels[a, b] = comp
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = x + dx, y + dy
                        if (0 <= u < n0 and 0 <= v < n1 and mask[u, v]
                                and labels[u, v] == 0):
                            labels[u, v] = comp
                            stack.append((u, v))
    return labels, np.bincount(labels.ravel())[1:]


def _otsu(vals: np.ndarray, bins: int = 64) -> float:
    """Parameter-free bimodal threshold (used on log-density: a well-scanned
    core vs sparse periphery separates cleanly in log space)."""
    hist, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    p = hist.astype(float) / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    between = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-12)
    return float(centers[int(np.argmax(between[:-1]))])


def box_decimals(vox: float) -> int:
    """Decimals a proposed box is written with: at least three (every scene
    so far — a 0.2-unit voxel writes millimetres, as it always did), and
    two orders of magnitude below the voxel when the voxel is smaller than
    0.1 units, so a box is never rounded to nothing on a small-unit scene."""
    return max(3, int(np.ceil(-np.log10(vox))) + 2)


def propose_volume(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
                   cfg: dict, scene_name: str, should_cancel=None) -> dict:
    """Auto-proposal, strategy-dispatched (volume_proposal.strategy):

    - `enclosed` (rooms): components of the wall-enclosed interior region.
    - `density_core` (exteriors): components of the well-scanned core — cells
      above an Otsu threshold on log density; no enclosure assumption.
    - `auto`: enclosed, falling back to density_core when nothing is enclosed
      (an exterior-shaped scene) or when the enclosure scan would be over
      the footprint budget — logged either way. The result is a
      STARTING POINT for the volume gate, never the gate itself.
    """
    icfg = cfg["render"]["interior"]
    pcfg = cfg["render"]["volume_proposal"]
    strategy = pcfg.get("strategy", "auto")
    vox = icfg["voxel_size"]
    a0, a1, up = frame.ground_axes[0], frame.ground_axes[1], frame.up_axis
    density2d, lo0, lo1 = footprint_density(means, opacities, frame, vox)
    n0, n1 = density2d.shape

    labels, sizes, used, reason = None, np.array([]), strategy, None
    limit = int(cfg["render"]["grid_budget"]["footprint_cells"])
    if strategy == "auto" and n0 * n1 > limit:
        # Over the footprint budget under `auto`: no refusal — the scan is
        # what the budget guards, and density_core needs none. One log
        # line; the proposal records the fallback.
        log.warning("enclosure scan over budget (%d x %d = %s cells at %g "
                    "units/cell, render.grid_budget.footprint_cells = %s); "
                    "density_core", n0, n1, f"{n0 * n1:,}", vox, f"{limit:,}")
        used, reason = "density_core", "over_budget"
    elif strategy in ("auto", "enclosed"):
        # Budget BEFORE the scan: the enclosure test below is a
        # pure-Python 8-ray walk per cell, and on a scene 100x off scale
        # the footprint is 10^4 the cells and the walk ~10^6 the work. Only
        # this branch walks, so `density_core` is a real escape from it —
        # and under `auto` the fallback above takes it by itself; an
        # explicit `enclosed` still refuses here.
        check_grid_budget(
            "volume proposal", (n0, n1), vox, frame, limit,
            "footprint_cells", gate="volume",
            escape=", knowing the scan logs its progress and Stop lands "
                   "within one grid row, or set volume_proposal.strategy: "
                   "auto (falls back to density_core by itself over the "
                   "budget) or density_core, which needs no enclosure scan.")
        occupied2d = density2d >= icfg["occupied_density"]
        # Interior test: enclosed by occupancy in >= N of 8 ray directions.
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        interior = np.zeros_like(occupied2d)
        log.info("enclosure scan: %d x %d cells at %g units/cell", n0, n1, vox)
        last = time.monotonic()
        for a in range(n0):
            check_cancel(should_cancel)   # once per grid row: Stop lands here
            if time.monotonic() - last > 2.0:   # a room finishes in silence
                log.info("enclosure scan: row %d/%d (%d%%)", a, n0, 100 * a // n0)
                last = time.monotonic()
            for b in range(n1):
                hits = 0
                for da, db in dirs:
                    x, y = a + da, b + db
                    while 0 <= x < n0 and 0 <= y < n1:
                        if occupied2d[x, y]:
                            hits += 1
                            break
                        x += da
                        y += db
                if hits >= icfg["insideness_min_hits"]:
                    interior[a, b] = True
        labels, sizes = _components(interior)
        used = "enclosed"
        if len(sizes) == 0:
            if strategy == "enclosed":
                raise Refusal(
                    "volume proposal: no enclosed interior region found; an "
                    "exterior-shaped scene. Draw the volume on the canvas, "
                    "use the whole-scene act, or set "
                    "render.volume_proposal.strategy to auto (falls back by "
                    "itself) or density_core in the scene profile.",
                    gate="volume")
            log.warning("volume proposal: no enclosed interior region; "
                        "exterior-shaped scene, falling back to density_core")
            labels, reason = None, "nothing_enclosed"

    if labels is None:  # density_core, requested or fallen back to
        nz = density2d[density2d > 0]
        th = np.expm1(_otsu(np.log1p(nz)))
        core = density2d >= th
        labels, sizes = _components(core)
        used = "density_core"
        log.info("density_core: otsu threshold %.1f -> %d core cells, %d component(s)",
                 th, int(core.sum()), len(sizes))
        if len(sizes) == 0:
            raise Refusal("volume proposal: density_core found no core cells "
                          "(nothing dense enough to propose from). Draw the "
                          "volume on the canvas, or use the whole-scene act.",
                          gate="volume")

    keep = [c + 1 for c, s in enumerate(sizes)
            if s >= pcfg["min_component_frac"] * sizes.max()]

    # Heights: explicit override first; else enclosed uses floor..ceiling,
    # density_core uses the FOOTPRINT-CONSTRAINED height percentiles of the
    # core cells' own points (never global-bounds inheritance).
    hr = pcfg.get("height_range")
    if hr:
        h_lo, h_hi = min(hr), max(hr)
        log.info("volume proposal: height_range override [%.2f, %.2f]", h_lo, h_hi)
    elif used == "enclosed":
        h_lo = frame.floor - pcfg["floor_margin"]
        h_hi = max(frame.lo[up] * frame.up_sign, frame.hi[up] * frame.up_sign)
    else:
        i0 = np.clip(((means[:, a0] - lo0) / vox).astype(int), 0, n0 - 1)
        i1 = np.clip(((means[:, a1] - lo1) / vox).astype(int), 0, n1 - 1)
        in_core = np.isin(labels[i0, i1], keep)
        h = means[in_core, up] * frame.up_sign
        h_lo = float(np.percentile(h, 1)) - pcfg["floor_margin"]
        h_hi = float(np.percentile(h, 99))
        log.info("density_core heights from core-cell points: [%.2f, %.2f]; "
                 "review at the volume gate (canopy/sky policy is a per-scene "
                 "decision; override via volume_proposal.height_range)",
                 h_lo, h_hi)

    # Boxes are written to a resolution that follows the voxel, never a
    # fixed number of decimals: three decimals turned a valid box into all
    # zeros on a scene whose unit is 10^5 metres (a 100,000x shrink).
    decimals = box_decimals(vox)
    boxes = []
    for name_idx, c in enumerate(keep):
        cells = np.argwhere(labels == c)
        g0_min, g1_min = cells.min(axis=0) * vox + [lo0, lo1]
        g0_max, g1_max = (cells.max(axis=0) + 1) * vox + [lo0, lo1]
        bmin, bmax = np.zeros(3), np.zeros(3)
        bmin[a0], bmax[a0] = g0_min, g0_max
        bmin[a1], bmax[a1] = g1_min, g1_max
        # signed-height interval -> world coordinate on the up axis
        u_a, u_b = h_lo * frame.up_sign, h_hi * frame.up_sign
        bmin[up], bmax[up] = min(u_a, u_b), max(u_a, u_b)
        boxes.append(dict(name=f"region_{name_idx}",
                          min=bmin.round(decimals).tolist(),
                          max=bmax.round(decimals).tolist()))
    log.info("volume proposal (%s): %d box(es) from %d component(s)",
             used, len(boxes), len(sizes))
    # `_used` / `_reason`: which branch produced the boxes and why it fell
    # back (over_budget | nothing_enclosed | None) — the Propose analysis
    # records them; underscore keys never reach volume.json.
    return dict(scene=scene_name, coords="ply_world", boxes=boxes,
                _density2d=density2d, _axes=(a0, a1), _origin=(lo0, lo1), _vox=vox,
                _used=used, _reason=reason)


def save_volume_review(proposal: dict, out_png: str | Path) -> None:
    """Top-down log-density plot with proposed boxes, for hand-correction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    density = proposal["_density2d"]
    a0, a1 = proposal["_axes"]
    lo0, lo1 = proposal["_origin"]
    vox = proposal["_vox"]
    fig, ax = plt.subplots(figsize=(10, 8))
    extent = (lo1, lo1 + density.shape[1] * vox, lo0, lo0 + density.shape[0] * vox)
    ax.imshow(np.log1p(density), origin="lower", extent=extent, cmap="viridis",
              aspect="equal")
    for b in proposal["boxes"]:
        bmin, bmax = np.asarray(b["min"]), np.asarray(b["max"])
        ax.add_patch(Rectangle((bmin[a1], bmin[a0]), bmax[a1] - bmin[a1],
                               bmax[a0] - bmin[a0], fill=False, color="red", lw=2))
        ax.text(bmin[a1], bmax[a0], b["name"], color="red", fontsize=10, va="bottom")
    ax.set_xlabel(f"axis {'xyz'[a1]} (m)")
    ax.set_ylabel(f"axis {'xyz'[a0]} (m)")
    ax.set_title("top-down opacity density (log) with proposed volume boxes")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    log.info("volume review plot -> %s", out_png)


def write_volume(proposal: dict, path: str | Path,
                 frame: SceneFrame | None = None) -> None:
    clean = {k: v for k, v in proposal.items() if not k.startswith("_")}
    block = frame.alignment_block if frame is not None else None
    if block is not None:
        clean["coords"] = "scene"
        clean["alignment"] = block
    write_text_atomic(path, json.dumps(clean, indent=1))
    log.info("volume.json -> %s", path)


def write_frame_sidecar(frame: SceneFrame, means: np.ndarray,
                        opacities: np.ndarray, vox: float,
                        path: str | Path) -> None:
    """density.json with the scene-frame fields and NO grid (`shape`
    [0, 0], `density` []), written when the proposal is refused before its
    scan. The volume gate reads the extent and the scale read from
    this file, so it can state the scene's dimensions and take the factor
    even though nothing was proposed — without it the gate shows nothing
    and the operator cannot act on the refusal from the app."""
    a0, a1 = frame.ground_axes
    _, _, lo0, lo1 = footprint_dims(frame, vox)
    out = _sidecar_dict(frame, means, opacities, boxes=[], a0=a0, a1=a1,
                        lo0=lo0, lo1=lo1, vox=vox,
                        density=np.zeros((0, 0)))
    write_text_atomic(path, json.dumps(out))
    log.info("density sidecar -> %s (frame only, no grid: proposal refused)",
             path)


def write_density_sidecar(proposal: dict, frame: SceneFrame,
                          means: np.ndarray, opacities: np.ndarray,
                          path: str | Path) -> None:
    """density.json next to volume_review.png : the same
    top-down density grid machine-readable, plus a FOOTPRINT-CONSTRAINED
 height histogram (the rule — heights read from the footprint, never
    global bounds) for the web volume gate's plan view + elevation strip.
    Written wherever the review plot is refreshed; changes no stage output.
    """
    density = proposal["_density2d"]
    a0, a1 = proposal["_axes"]
    lo0, lo1 = proposal["_origin"]
    vox = float(proposal["_vox"])
    # Bound the JSON: sum-pool the grid to <= ~360 cells per side (an
    # exterior at fine voxels would otherwise ship megabytes to the browser).
    k = max(int(np.ceil(max(density.shape) / 360)), 1)
    if k > 1:
        n0, n1 = density.shape
        p0, p1 = (-n0) % k, (-n1) % k
        d = np.pad(density, ((0, p0), (0, p1)))
        density = d.reshape(d.shape[0] // k, k, d.shape[1] // k, k).sum((1, 3))
        vox *= k

    out = _sidecar_dict(frame, means, opacities, boxes=proposal["boxes"],
                        a0=a0, a1=a1, lo0=lo0, lo1=lo1, vox=vox,
                        density=density)
    write_text_atomic(path, json.dumps(out))
    log.info("density sidecar -> %s (grid %s, pool %dx)", path,
             tuple(density.shape), k)


def _sig(v, n: int = 6) -> float:
    """Round to significant digits, never to fixed decimals: a fixed three
    decimals turned a floor of 0.000889 units (a scene whose unit is a
    kilometre) into 0.001 and its extents into 0.00 — the A04 class of
    precision loss, in the sidecar the volume gate and the viewer read."""
    return float(f"{float(v):.{n}g}")


def _sidecar_dict(frame: SceneFrame, means: np.ndarray, opacities: np.ndarray,
                  *, boxes: list, a0: int, a1: int, lo0: float, lo1: float,
                  vox: float, density: np.ndarray) -> dict:
    up = frame.up_axis
    if boxes:
        sel = np.zeros(len(means), dtype=bool)
        for b in boxes:
            bmin, bmax = np.asarray(b["min"]), np.asarray(b["max"])
            sel |= ((means[:, a0] >= bmin[a0]) & (means[:, a0] <= bmax[a0])
                    & (means[:, a1] >= bmin[a1]) & (means[:, a1] <= bmax[a1]))
    else:
        sel = np.ones(len(means), dtype=bool)
    h = means[sel][:, up]
    w = opacities[sel]
    h_lo, h_hi = (float(np.percentile(h, 0.2)), float(np.percentile(h, 99.8))) \
        if len(h) else (0.0, 1.0)
    counts, edges = np.histogram(h, bins=120, range=(h_lo, h_hi), weights=w)

    block = frame.alignment_block
    return dict(
        version=1, coords="scene" if block else "ply_world",
        # the levelling the frame was read under; null = the file's axes
        alignment=block,
        ground_axes=[int(a0), int(a1)],
        up_axis=int(up), up_sign=int(frame.up_sign),
        floor=_sig(frame.floor),
        # floor_auto = the histogram vote: surfaced so the web volume
        # gate can show "override vs auto-detected" even when an override is
        # active (frame.floor then == the override). Falls back to floor.
        floor_auto=_sig(frame.notes.get("floor_auto", frame.floor)),
        # Robust extents + the scale read, so the volume gate can state the
        # scene's DIMENSIONS before any render has run. Nothing else records
        # them this early, and the gate is the first place a scene that is
        # not to scale can be caught — every later stage spends work on it.
        extent=[_sig(v) for v in (frame.hi - frame.lo)],
        scale=frame.notes.get("scale"),
        origin=[_sig(lo0), _sig(lo1)],
        voxel=_sig(vox),
        shape=list(density.shape),
        density=np.round(density, 2).tolist(),
        height_hist=dict(edges=[_sig(e) for e in edges],
                         counts=np.round(counts, 2).tolist(),
                         footprint="boxes" if boxes else "scene"),
    )
