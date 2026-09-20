# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Camera path generation for Stage 1.

Everything operates in the raw .ply coordinate frame. The up axis is detected
(or configured) rather than assumed: splat exports rarely document one.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from .refusal import Refusal
from .sequencing import check_cancel
from .units import ScaleRecord, resolve_scale

log = logging.getLogger(__name__)

_AXES = {"x": 0, "y": 1, "z": 2}
_UP_AXES = tuple(f"{s}{a}" for a in "xyz" for s in "+-")


def propose_up_axis(means: np.ndarray, opacities: np.ndarray, cfg: dict,
                    lo: np.ndarray, hi: np.ndarray) -> str:
    """A GUESS at the up axis, for an error message to offer — never applied.

    Rooms are wider than tall, so the smallest robust extent is usually the
    height; the floor then carries more opacity-weighted mass than the
    ceiling, which fixes the sign. Both halves fail on rooms that are not
    wider than tall, which is why nothing calls this to decide anything.
    """
    extent = hi - lo
    axis = int(np.argmin(extent))
    # hist_bin_m is metres; the factor for this guess is the same one the
    # frame detector would use (recorded or measured — never a guess).
    bin_units = cfg["scene"]["hist_bin_m"] / resolve_scale(cfg).value
    nbins = max(int(extent[axis] / bin_units), 10)
    hist, edges = np.histogram(means[:, axis], bins=nbins,
                               range=(lo[axis], hi[axis]), weights=opacities)
    peak = int(np.argmax(hist))
    peak_pos = (edges[peak] + edges[peak + 1]) / 2
    sign = "+" if peak_pos < (lo[axis] + hi[axis]) / 2 else "-"
    return f"{sign}{'xyz'[axis]}"


def scale_read(extent: np.ndarray, up_axis: int, cfg: dict,
               rec: ScaleRecord) -> dict:
    """The scale as the volume gate shows it: the factor's record plus a
    READ of it, never a decision.

    The note says what a unit is and where the number came from: "1 unit
    ≈ 0.93 m (measured: a door, 2 m)" after the ruler, "1 unit = 1 m
    (recorded)" after a declared or typed factor. The plausibility read
    uses the robust UP-AXIS extent rather than floor-to-ceiling on
    purpose: the floor estimate is a room-assuming heuristic that has
    elected a workbench top at ~0.89 m on three scenes, and a check must
    not inherit that failure. The band is deliberately wide — the
    question is "is this metric at all", an order-of-magnitude question —
    and it is asked only of a scene filmed INSIDE A SPACE
    (`render.path_mode: interior`): an exterior or an orbited subject has
    no ceiling to judge by, so `plausible` is None there.
    """
    span = float(extent[up_axis])
    lo_m, hi_m = cfg["scene"]["plausible_height_m"]
    base = dict(rec.record(), scale=float(rec.value), band=[lo_m, hi_m])
    metres = span * rec.value
    if rec.source == "measured":
        what = f"1 unit ≈ {rec.value:.3g} m (measured: {rec.measured_as()})"
    else:
        what = f"1 unit = {rec.value:g} m (recorded)"
    inside = str((cfg.get("render") or {}).get("path_mode")) == "interior"
    if not inside:
        return dict(base, height_m=round(metres, 3), plausible=None,
                    note=what)
    ok = lo_m <= metres <= hi_m
    return dict(base, height_m=round(metres, 3), plausible=bool(ok),
                note=(what if ok else
                      f"{what}: at this scale the scene reads {metres:.3g} m "
                      f"tall, outside the {lo_m}-{hi_m} m band for a space "
                      f"filmed from inside; check the two points you "
                      f"measured (or the number you typed), or the filming "
                      f"answer on the render panel if this is not an interior"))


@dataclass
class SceneFrame:
    """Detected geometry conventions of a scene."""

    up_axis: int              # 0/1/2
    up_sign: float            # +1 if "up" is +axis
    ground_axes: tuple[int, int]
    lo: np.ndarray            # robust bounds (3,)
    hi: np.ndarray
    floor: float | None = None   # up-coordinate of the floor plane (signed axis value)
    scale: float = 1.0          # metres per scene unit (a convention the file
                                # does not record): recorded, or measured
                                # with the ruler — always the operator's
    scale_source: str = "recorded"   # "recorded" | "measured"
    notes: dict = field(default_factory=dict)
    # The profile's alignment, scene = R · file; None = identity.
    # Everything on this frame is in the scene frame; camera poses are
    # taken to the file frame (alignment.to_file_c2w) before rendering.
    alignment: np.ndarray | None = None

    @property
    def alignment_block(self) -> dict | None:
        from .alignment import block_of_matrix
        return block_of_matrix(self.alignment, self.up_axis)

    @property
    def up_vec(self) -> np.ndarray:
        v = np.zeros(3)
        v[self.up_axis] = self.up_sign
        return v


def detect_scene_frame(means: np.ndarray, opacities: np.ndarray, cfg: dict,
                       floor_mask: np.ndarray | None = None) -> SceneFrame:
    """Robust bounds, up-axis detection, and floor estimate.

    The scale factor rides along as `frame.scale`: the recorded or the
    measured one — with neither, `resolve_scale` refuses to the volume gate
    naming the ruler (nothing here guesses a scene's size). The read of it
    lands in `notes["scale"]`, for the volume gate and the manifests.

    Up-axis heuristic: interiors are wider than tall — the up axis has the
    smallest robust extent. Sign: the floor carries more opacity-weighted mass
    than the ceiling, so "down" points toward the dominant density peak.

    floor_mask restricts the FLOOR histogram only (e.g. to in-volume Gaussians).
    Up-axis extents always use the full scene: a trimmed volume footprint can be
    narrower than the ceiling height, which flips the smallest-extent heuristic
    (observed on a small lab interior).
    """
    pct_lo, pct_hi = cfg["scene"]["bounds_pct_lo"], cfg["scene"]["bounds_pct_hi"]
    lo = np.percentile(means, pct_lo, axis=0)
    hi = np.percentile(means, pct_hi, axis=0)
    extent = hi - lo

    up_cfg = str(cfg["scene"]["up_axis"])
    if up_cfg == "auto":
        # Carveout does NOT infer the up axis: it is a render convention the
        # scene must arrive in. "auto" therefore PROPOSES and refuses, the
        # way every other machine suggestion in this pipeline waits for an
        # explicit operator act. Proceeding on the guess
        # is the failure mode — it is silently wrong on any room narrower than
        # it is tall, and by then a whole render has been spent on it.
        raise ValueError(
            f"scene.up_axis is 'auto', and Carveout does not infer the up "
            f"axis; no splat container records it, so it cannot be recovered "
            f"from the file.\n"
            f"  Carveout renders -Y up. Orient the scene that way at ingest "
            f"(the tool that wrote it is the place to bake the rotation), or "
            f"pin the axis in the scene profile:\n"
            f"      scene: {{up_axis: \"{propose_up_axis(means, opacities, cfg, lo, hi)}\"}}\n"
            f"  That proposal is a GUESS from the scene's proportions "
            f"(robust extents {np.round(extent, 2).tolist()}, smallest wins) "
            f"and is wrong whenever the room is narrower than it is tall. "
            f"Confirm it against the floor plane on the 3D canvas at the "
            f"volume gate before you trust it.")
    if up_cfg not in _UP_AXES:
        raise ValueError(
            f"scene.up_axis={up_cfg!r} is not a direction; expected one of "
            f"{', '.join(_UP_AXES)} (or 'auto' to be told what the scene's "
            f"proportions suggest). Carveout renders -Y up.")
    up_axis = _AXES[up_cfg[1]]
    up_sign = +1.0 if up_cfg[0] == "+" else -1.0

    # Scale: a convention the file does not record, like the up axis, and
    # like it an operator's answer — declared, typed, or measured with the
    # ruler. None recorded refuses here, to the volume gate.
    rec = resolve_scale(cfg)

    ground_axes = tuple(i for i in range(3) if i != up_axis)
    frame = SceneFrame(up_axis=up_axis, up_sign=up_sign, ground_axes=ground_axes,
                       lo=lo, hi=hi, scale=rec.value, scale_source=rec.source)

    # A read, not a gate: an implausible size is the one signal that a scene
    # is not to scale, and it is worth saying at the earliest point rather
    # than after eight stages have run on unitless numbers.
    read = scale_read(extent, up_axis, cfg, rec)
    if read["plausible"] is False:
        log.warning("scene scale: %s", read["note"])
    else:
        log.info("scene scale: %s", read["note"])

    # Floor: dominant peak in the lowest floor_band_frac of the (signed) height
    # range — confined to the bottom so a cluttered desk plane can't win.
    # (Room-assuming heuristic, config-gated per the generality amendment.)
    if floor_mask is not None:
        means, opacities = means[floor_mask], opacities[floor_mask]
    h = means[:, up_axis] * up_sign
    h_lo = min(lo[up_axis] * up_sign, hi[up_axis] * up_sign)
    h_hi = max(lo[up_axis] * up_sign, hi[up_axis] * up_sign)
    band_hi = h_lo + cfg["scene"]["floor_band_frac"] * (h_hi - h_lo)
    sel = h <= band_hi
    # hist_bin_m is a length; this function receives the profile's metres
    # (it is what establishes the factor), so it divides the one key it
    # needs here itself.
    bin_units = cfg["scene"]["hist_bin_m"] / frame.scale
    nbins = max(int((band_hi - h_lo) / bin_units), 10)
    hist, edges = np.histogram(h[sel], bins=nbins, range=(h_lo, band_hi),
                               weights=opacities[sel])
    peak = int(np.argmax(hist))
    floor_auto = (edges[peak] + edges[peak + 1]) / 2
    # Operator override: the band heuristic can elect a cluttered surface
    # (a workbench top once won the floor vote — signed 0.888 vs
    # true ~0.0, camera eye-height then ~0.5 m too high). scene.floor pins the
    # floor directly in SIGNED convention (same as frame.floor / density.json);
    # null = auto. Mirrors the scene.up_axis override above.
    override = cfg["scene"].get("floor")
    frame.floor = float(override) if override is not None else float(floor_auto)
    frame.notes.update(extent=extent.tolist(), floor_signed=float(frame.floor),
                       floor_auto=float(floor_auto),
                       floor_override=(None if override is None
                                       else float(override)),
                       # Recorded so every artefact downstream states which
                       # units its numbers are in, instead of leaving a reader
                       # to assume metres the way the code used to.
                       scale=read)
    if override is not None:
        log.info("floor plane at signed height %.3f (OPERATOR OVERRIDE; "
                 "auto-detect was %.3f, range %.2f..%.2f)",
                 frame.floor, floor_auto, h_lo, h_hi)
    else:
        log.info("floor plane at signed height %.3f (range %.2f..%.2f)",
                 frame.floor, h_lo, h_hi)
    return frame


# ---------------------------------------------------------------------------
# Grid budgets
# ---------------------------------------------------------------------------
def footprint_dims(frame: "SceneFrame", vox: float) -> tuple[int, int, float, float]:
    """Cell counts and origin of the 2D ground grid every footprint scan
    rasterises: the proposer's enclosure test and the interior path's
    free-space search build the same grid from the same knob."""
    a0, a1 = frame.ground_axes
    lo0, lo1 = float(frame.lo[a0]), float(frame.lo[a1])
    n0 = max(int((frame.hi[a0] - lo0) / vox) + 1, 4)
    n1 = max(int((frame.hi[a1] - lo1) / vox) + 1, 4)
    return n0, n1, lo0, lo1


def _fmt_cells(n: int) -> str:
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{n:,}"


def check_grid_budget(what: str, dims, vox: float, frame: "SceneFrame",
                      limit: int, key: str, gate: str, escape: str) -> None:
    """Refuse BEFORE a scene-sized grid is allocated or scanned, when its
    cell count exceeds the configured budget (`render.grid_budget.<key>`).

    The grids are sized in metres (`voxel_size`, `cell_size`) over the
    scene's robust bounds in scene units, so on a scene that is not to
    scale their size is off by the square or cube of the factor — a 100x
    export of a room is ten thousand times the cells, and the pure-Python
    enclosure scan a million times the work. The refusal states the
    dimensions and names the two causes in order, with what the operator
    can do about each today. `escape` is the caller's own sentence
    on what raising the budget costs and what else gets past THIS check —
    the proposer's `density_core` is no escape from the interior path's
    free-space search, which used to be recommended for both."""
    n = int(np.prod([int(d) for d in dims]))
    if n <= limit:
        return
    extent = np.asarray(frame.hi - frame.lo, dtype=float)
    scale = float(getattr(frame, "scale", 1.0))
    units = " x ".join(f"{v:.4g}" for v in extent)
    src = getattr(frame, "scale_source", "recorded")
    in_metres = (f"{' x '.join(f'{v * scale:.4g}' for v in extent)} m at "
                 f"scene.scale_m_per_unit {scale:.4g} ({src})")
    shape = " x ".join(str(int(d)) for d in dims)
    raise Refusal(
        f"{what} refused before it started: the grid would be {shape} cells "
        f"= {_fmt_cells(n)} at {vox:g} scene units per cell, over "
        f"render.grid_budget.{key} = {_fmt_cells(limit)}.\n"
        f"  This scene's robust extents are {units} scene units, "
        f"{in_metres}.\n"
        f"  Two causes, in order of likelihood:\n"
        f"    1. the measurement is wrong, or the scene is not to scale (a "
        f"photogrammetry .ply carries an arbitrary unit; an export scaled in "
        f"an editor does this). At the volume gate the scene's dimensions "
        f"are shown beside the scale: measure again on the canvas, or "
        f"correct the factor, and run this again; every metre-sized grid "
        f"follows it.\n"
        f"    2. it is a legitimately large metric scene. Raise "
        f"render.grid_budget.{key} in the scene profile" + escape,
        gate=gate)


# ---------------------------------------------------------------------------
# Pose construction (OpenCV convention: x right, y down, z forward)
# ---------------------------------------------------------------------------

def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up/down
        right = np.cross(forward, up + np.array([0.017, 0.013, 0.011]))
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    return c2w


def _to_world(frame: SceneFrame, g0: float, g1: float, h_signed: float) -> np.ndarray:
    p = np.zeros(3)
    p[frame.ground_axes[0]] = g0
    p[frame.ground_axes[1]] = g1
    p[frame.up_axis] = h_signed * frame.up_sign
    return p


def detect_ceiling(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
                   cfg: dict, mask: np.ndarray | None = None) -> float | None:
    """The ceiling as the floor is found (`detect_scene_frame`): the
    dominant opacity-weighted density peak, here within the HIGHEST
    `floor_band_frac` of the signed height range, over `mask` (the volume)
    when given. Signed height, or None when no peak holds a real share of
    the mass (an open scene, a capture with no ceiling). `cfg` is in scene
    units (`hist_bin_m` already divided)."""
    if mask is not None:
        means, opacities = means[mask], opacities[mask]
    if len(means) < 1000:
        return None
    h = means[:, frame.up_axis] * frame.up_sign
    h_lo, h_hi = float(np.percentile(h, 1)), float(np.percentile(h, 99))
    band_lo = h_hi - float(cfg["scene"]["floor_band_frac"]) * (h_hi - h_lo)
    sel = h >= band_lo
    nbins = max(int((h_hi - band_lo) / float(cfg["scene"]["hist_bin_m"])), 10)
    hist, edges = np.histogram(h[sel], bins=nbins, range=(band_lo, h_hi),
                               weights=opacities[sel])
    total = float(opacities.sum())
    if total <= 0 or hist.max() / total < 0.02:
        return None
    peak = int(np.argmax(hist))
    return float((edges[peak] + edges[peak + 1]) / 2)


# ---------------------------------------------------------------------------
# Interior path
# ---------------------------------------------------------------------------

def _in_boxes(points: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Membership in the union of boxes (a local copy of volume.in_volume:
    volume imports this module)."""
    mask = np.zeros(len(points), dtype=bool)
    for b in boxes:
        mask |= ((points >= np.asarray(b["min"])) & (points <= np.asarray(b["max"]))).all(axis=1)
    return mask


def interior_path(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
                  cfg: dict, n_views: int, seed: int = 42,
                  volume_boxes: list[dict] | None = None,
                  should_cancel=None) -> tuple[list[np.ndarray], np.ndarray]:
    """Eye-height cameras in free space, inside the room, aimed along sightlines.

    Returns (candidate c2w poses, all free-cell world positions at eye height,
    the number of near-field poses, which lead the list) — the free positions
    are reused for coverage-driven extra candidates; the near-field count lets
    the render keep those views whatever the budget, as it keeps manual ones.
    `should_cancel` (the web Stop) is polled once per grid row of the
    free-space search, the one pure-Python walk on this path.
    """
    icfg = cfg["render"]["interior"]
    vox = icfg["voxel_size"]
    eye_h = frame.floor + icfg["eye_height"]

    # 2D obstacle map over the ground plane: opacity mass in the collision
    # slab — ankle to head height. In a low room the slab's top would reach
    # the ceiling and count it as an obstacle over the whole floor (a 2.1 m
    # garage: two free cells by the door, every camera in one corner),
    # so the slab stops `ceiling_margin` under the ceiling
    # when one is read; a tall room is untouched.
    h = means[:, frame.up_axis] * frame.up_sign
    slab_top = float(icfg["slab_hi"])
    ceiling = detect_ceiling(means, opacities, frame, cfg,
                             mask=(_in_boxes(means, volume_boxes)
                                   if volume_boxes else None))
    if ceiling is not None:
        room_h = ceiling - frame.floor
        capped = room_h - float(icfg["ceiling_margin"])
        # A ceiling is over a standing person's head: the slab never stops
        # below eye height + 0.2 m, so a shelf top or a mezzanine that won
        # the vote in a sparse-ceilinged room can only ever cost the cap,
        # never the head-height clutter the slab exists for.
        if capped < slab_top and capped >= float(icfg["eye_height"]) + 0.2 / frame.scale:
            log.info("interior path: the ceiling reads %.2f m over the floor, "
                     "so the collision slab stops %.2f m under it at %.2f m "
                     "(render.interior.slab_hi %.2f m would count the ceiling "
                     "as an obstacle over the whole floor)",
                     room_h * frame.scale, float(icfg["ceiling_margin"]) * frame.scale,
                     capped * frame.scale, slab_top * frame.scale)
            slab_top = capped
        frame.notes["ceiling_signed"] = float(ceiling)
        frame.notes["room_height_m"] = float(room_h * frame.scale)
    frame.notes["slab_top_m"] = float(slab_top * frame.scale)
    slab = (h >= frame.floor + icfg["slab_lo"]) & (h <= frame.floor + slab_top)
    g0 = means[slab, frame.ground_axes[0]]
    g1 = means[slab, frame.ground_axes[1]]
    w = opacities[slab]
    n0, n1, lo0, lo1 = footprint_dims(frame, vox)
    # The free-space search below is the proposer's 8-ray enclosure walk
    # again, per clearance level — the same budget applies.
    check_grid_budget("interior camera placement", (n0, n1), vox, frame,
                      int(cfg["render"]["grid_budget"]["footprint_cells"]),
                      "footprint_cells", gate="render",
                      escape=", knowing the free-space search walks 8 rays "
                             "per cell at up to three clearance levels and "
                             "Stop lands within one grid row.")
    i0 = np.clip(((g0 - lo0) / vox).astype(int), 0, n0 - 1)
    i1 = np.clip(((g1 - lo1) / vox).astype(int), 0, n1 - 1)
    density2d = np.zeros((n0, n1))
    np.add.at(density2d, (i0, i1), w)
    occupied2d = density2d >= icfg["occupied_density"]

    # Near-field aim map: furniture below eye height is what a standing
    # person looks over, opacity in the slab below `near_field_top` with
    # nothing at or above it; a wall, a door or a tall cabinet has mass up
    # there. The mean height of that low mass per cell is where the
    # near-field view looks (a desk with monitors on it reads about 0.9 m,
    # a bare counter its top).
    hs = h[slab]
    lowm = hs <= frame.floor + float(icfg["near_field_top"])
    low2d = np.zeros((n0, n1))
    high2d = np.zeros((n0, n1))
    lowh2d = np.zeros((n0, n1))
    np.add.at(low2d, (i0[lowm], i1[lowm]), w[lowm])
    np.add.at(high2d, (i0[~lowm], i1[~lowm]), w[~lowm])
    np.add.at(lowh2d, (i0[lowm], i1[lowm]), w[lowm] * hs[lowm])
    low_only2d = (low2d >= icfg["occupied_density"]) & (high2d < icfg["occupied_density"])

    # Free cells: unoccupied, surrounded by occupancy in most of 8 directions
    # (= inside the room, not beyond the walls).
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def free_cells(clearance: int) -> list[tuple[int, int]]:
        cells = []
        for a in range(clearance, n0 - clearance):
            check_cancel(should_cancel)   # once per grid row: Stop lands here
            for b in range(clearance, n1 - clearance):
                if occupied2d[a - clearance:a + clearance + 1,
                              b - clearance:b + clearance + 1].any():
                    continue
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
                    cells.append((a, b))
        return cells

    # Prefer generous clearance (cameras away from walls/furniture); relax it
    # progressively so cluttered rooms still yield positions.
    n_pos = max(int(np.ceil(n_views / icfg["views_per_position"])), 1)

    # Camera confinement to the volume is config-gated: correct for
    # room-sized volumes (scoping cameras out through an open door), fatal for FOCUS
    # volumes whose footprint is fully occupied by the objects of interest —
    # cameras then roam the room's free space while the volume keeps scoping
    # coverage, aim targets, and lift/export.
    confine = icfg["cameras_in_volume"] and volume_boxes

    def in_volume_cell(a: int, b: int) -> bool:
        if not confine:
            return True
        p = _to_world(frame, lo0 + (a + 0.5) * vox, lo1 + (b + 0.5) * vox, eye_h)
        return any(((p >= bx["min"]) & (p <= bx["max"])).all() for bx in volume_boxes)

    free: list[tuple[int, int]] = []
    for clearance in range(max(icfg["clearance_vox"], 3), 0, -1):
        free = [c for c in free_cells(clearance) if in_volume_cell(*c)]
        if len(free) >= n_pos:
            log.info("interior path: clearance %d voxels -> %d free cells%s",
                     clearance, len(free),
                     " (confined to volume)" if confine else "")
            break
    if not free:
        hint = (
            "cameras are confined to the volume boxes and none of their "
            "footprint is free at eye height, which a tight focus volume does; "
            "set render.interior.cameras_in_volume: false in the scene profile"
        ) if confine else (
            "the room's free space is not found at eye height; switch the "
            "path mode on the render panel (orbit rings the volume; ground "
            "walks a local ground map), or turn 'tight focus volume' on for a "
            "bench-top volume; the occupancy keys "
            "(render.interior.voxel_size / occupied_density / "
            "insideness_min_hits) are profile-only"
        )
        raise Refusal(f"interior path: no free-space cells found; {hint}",
                      gate="render")

    # Focus aiming (found reviewing a bench-top scene's renders): when a FOCUS
    # volume is the subject, generic sightline aiming produces room views that
    # never frame it. Prefer positions in a stand-off band around the volume
    # footprint and aim poses directly at spread points ON the volume.
    focus = bool(icfg["focus_aim"]) and bool(volume_boxes)

    def dist_to_boxes_2d(cell) -> float:
        p = _to_world(frame, lo0 + (cell[0] + 0.5) * vox,
                      lo1 + (cell[1] + 0.5) * vox, eye_h)
        g = np.array([p[frame.ground_axes[0]], p[frame.ground_axes[1]]])
        best = np.inf
        for bx in volume_boxes:
            bmin = np.array([bx["min"][frame.ground_axes[0]], bx["min"][frame.ground_axes[1]]])
            bmax = np.array([bx["max"][frame.ground_axes[0]], bx["max"][frame.ground_axes[1]]])
            best = min(best, float(np.linalg.norm(g - np.clip(g, bmin, bmax))))
        return best

    # Farthest-point sampling for coverage.
    rng = np.random.default_rng(seed)
    pool = list(free)
    if focus:
        so_lo, so_hi = icfg["focus_standoff"]
        banded = [c for c in pool if so_lo <= dist_to_boxes_2d(c) <= so_hi]
        # The band wins if it can host a meaningful share of the achievable
        # positions (n_pos may exceed the room's total free cells).
        need = min(n_pos, len(pool))
        if len(banded) >= max(4, need // 2):
            pool = banded
            log.info("focus aim: %d/%d free cells in the %.2g-%.2g unit stand-off band",
                     len(banded), len(free), so_lo, so_hi)
        else:
            log.info("focus aim: only %d cells in the stand-off band (< %d needed)"
                     "; using all free cells", len(banded), max(4, need // 2))
    def fps(cands: list, k: int) -> list:
        pts = np.array(cands, dtype=float)
        out = [pts[rng.integers(len(pts))]]
        dd = np.full(len(pts), np.inf)
        while len(out) < min(k, len(pts)):
            dd = np.minimum(dd, ((pts - out[-1]) ** 2).sum(axis=1))
            out.append(pts[int(np.argmax(dd))])
        return out

    chosen = fps(pool, n_pos)

    # Macro tier (small-object rescue): a few extra positions very
    # close to the volume, aimed at its lowest aim band — small bench-top
    # objects only become detectable to SAM 3 at close stand-off.
    macro_chosen: list = []
    if focus and icfg["focus_macro_standoff"]:
        m_lo, m_hi = icfg["focus_macro_standoff"]
        macro_pool = [c for c in free if m_lo <= dist_to_boxes_2d(c) <= m_hi]
        if macro_pool:
            macro_chosen = fps(macro_pool, icfg["focus_macro_positions"])
            log.info("focus aim: %d macro positions from %d cells in the "
                     "%.2g-%.2g unit band", len(macro_chosen), len(macro_pool),
                     m_lo, m_hi)
        else:
            log.info("focus aim: no free cells in the macro band %.2g-%.2g units",
                     m_lo, m_hi)

    def sightline_vox(a: float, b: float, yaw: float) -> int:
        """2D ray-march until occupancy/bounds; length in voxels."""
        da, db = np.cos(yaw), np.sin(yaw)
        steps = 0
        x, y = a, b
        while True:
            x += da
            y += db
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < n0 and 0 <= yi < n1) or occupied2d[xi, yi]:
                return steps
            steps += 1

    # Aim along deep free-space sightlines: a long ray before hitting occupancy
    # means the view crosses the room instead of staring into the nearest wall.
    # (Pure density aiming failed on a small single-room test scene — the
    # nearest wall wins every azimuth histogram, distance-weighted or not.)
    poses = []
    near: list[np.ndarray] = []
    up = frame.up_vec
    if focus:
        # Aim points spread over each box: horizontal center + offsets along
        # the longer horizontal extent, at focus_aim_height_fracs of the box's
        # vertical span. Exact look_at pitch — no fixed tiers. Camera heights
        # from focus_heights (default: the single eye_height).
        heights = icfg["focus_heights"] or [icfg["eye_height"]]
        lowest_frac = min(icfg["focus_aim_height_fracs"])
        aims, low_mask = [], []
        for bx in volume_boxes:
            bmin, bmax = np.asarray(bx["min"], float), np.asarray(bx["max"], float)
            c = 0.5 * (bmin + bmax)
            a0g, a1g = frame.ground_axes
            long_ax = a0g if (bmax - bmin)[a0g] >= (bmax - bmin)[a1g] else a1g
            for frac in icfg["focus_aim_height_fracs"]:
                for off in (-0.25, 0.0, 0.25):
                    t = c.copy()
                    t[long_ax] += off * (bmax - bmin)[long_ax]
                    t[frame.up_axis] = (bmin[frame.up_axis]
                                        + frac * (bmax - bmin)[frame.up_axis])
                    aims.append(t)
                    low_mask.append(frac == lowest_frac)
        aims = np.array(aims)
        # Macro positions use only the lowest aim band (bench-top objects) at
        # the first (lowest) camera height.
        aims_low = aims[np.array(low_mask)]
        for cell, cell_heights, cell_aims in (
                [(c, heights, aims) for c in chosen]
                + [(c, heights[:1], aims_low) for c in macro_chosen]):
            p0 = lo0 + (cell[0] + 0.5) * vox
            p1 = lo1 + (cell[1] + 0.5) * vox
            for h_cam in cell_heights:
                pos = _to_world(frame, p0, p1, frame.floor + h_cam)
                order = np.argsort(np.linalg.norm(cell_aims - pos, axis=1))
                for t in cell_aims[order[:icfg["views_per_position"]]]:
                    poses.append(look_at(pos, t, up))
    else:
        min_sep = np.deg2rad(icfg["min_yaw_sep_deg"])
        min_sight = icfg["min_sightline"] / vox
        for cell in chosen:
            sights = [(sightline_vox(cell[0], cell[1], yaw), yaw)
                      for yaw in np.linspace(-np.pi, np.pi, 36, endpoint=False)]
            sights.sort(key=lambda s: s[0], reverse=True)
            yaws: list[float] = []
            for length, yaw in sights:
                if length < min_sight and yaws:
                    break
                if all(min(abs(yaw - y), 2 * np.pi - abs(yaw - y)) >= min_sep for y in yaws):
                    yaws.append(yaw)
                if len(yaws) >= icfg["views_per_position"]:
                    break
            p0 = lo0 + (cell[0] + 0.5) * vox
            p1 = lo1 + (cell[1] + 0.5) * vox
            pos = _to_world(frame, p0, p1, eye_h)
            for yaw in yaws:
                for pitch in icfg["pitches_deg"]:
                    t0 = p0 + np.cos(yaw) * 3.0
                    t1 = p1 + np.sin(yaw) * 3.0
                    t_h = eye_h + 3.0 * np.tan(np.deg2rad(pitch))
                    target = _to_world(frame, t0, t1, t_h)
                    poses.append(look_at(pos, target, up))
        # Near-field views: the sightline rule aims across the room, so a
        # desk, a counter or a bench is occupancy every camera turns away
        # from; the operator then captured those views by hand, one per
        # desk, standing back a little and looking down at the top. That
        # is what these are: views per PIECE of furniture below eye height
        # (not per standing position: a position takes what is nearest,
        # and the main desk went without when the nearest position stood
        # too close to it and the next one faced another desk), spaced
        # along its length, each from a free cell a stand-off away, aimed
        # by geometry at the top's own mean height, no pitch tier.
        near, nf = _near_field_views(frame, icfg, cfg["render"]["quality_filter"],
                                     occupied2d, low_only2d, low2d, lowh2d,
                                     free, vox, lo0, lo1, eye_h, up)
        # first in the list: the diversity prune keeps the first of two
        # near-duplicates, and a dedicated framing outranks a pitch tier
        poses[:0] = near
    if focus:
        log.info("interior path: %d candidate poses from %d positions (focus aim)",
                 len(poses), len(chosen))
    else:
        log.info("interior path: %d candidate poses from %d positions; %d "
                 "near-field views at %d pieces of furniture below eye height "
                 "(%d aim points %.1f m apart, %d with no free cell %.1f-%.1f m "
                 "back, beyond the %.2f m fog horizon, with a clear line to them)",
                 len(poses), len(chosen), len(near), nf["pieces"], nf["aims"],
                 float(icfg["near_field_spacing"]) * frame.scale, nf["unreachable"],
                 nf["standoff"][0] * frame.scale, nf["standoff"][1] * frame.scale,
                 nf["fog"] * frame.scale)

    free_world = np.stack([
        _to_world(frame, lo0 + (a + 0.5) * vox, lo1 + (b + 0.5) * vox, eye_h)
        for a, b in free
    ])
    return poses, free_world, len(near)


def _near_field_views(frame: SceneFrame, icfg: dict, qcfg: dict,
                      occupied2d: np.ndarray, low_only2d: np.ndarray,
                      low2d: np.ndarray, lowh2d: np.ndarray,
                      free: list[tuple[int, int]], vox: float,
                      lo0: float, lo1: float, eye_h: float, up: np.ndarray
                      ) -> tuple[list[np.ndarray], dict]:
    """Poses looking down at the furniture below eye height, one per aim
    point, at most `near_field_views` in all.

    A piece is an 8-connected run of at least `near_field_min_cells`
    low-only cells (opacity in the slab below `near_field_top` and none at
    or above it: what a standing person looks over, a desk with its
    monitors, a counter, a bench, a chair; a wall, a door or a tall
    cabinet has mass up there and is never one; a smaller run is a floater
    cluster or a lone small object the room views cover). Its aim points
    are its heaviest cell, then the cell
    farthest from every chosen one while that is at least
    `near_field_spacing` away, so a long counter gets a view per stretch.
    Heavier pieces come first, and every piece's first view before any
    piece's second, so the cap falls on the least. The camera stands on
    the free eye-height cell nearest the aim point within
    `near_field_standoff` (horizontal) whose line to the aim crosses no
    wall (low furniture in between is looked over) and whose SLANT
    distance to the piece's nearest cell, at the piece's top height, is
    beyond the quality filter's near-field fog horizon: the pass that
    reads close geometry as fog must not see the desk's own edge. Standing
    1.6 m up, that horizon is cleared a good deal closer than its own
    length, which is what makes these close-ups. The target is the aim
    cell at the opacity-weighted mean height of its piece around it."""
    n_max = int(icfg["near_field_views"] or 0)
    so_lo, so_hi = (float(x) for x in icfg["near_field_standoff"])
    fog = qcfg.get("near_field_depth")
    fog = float(fog) if isinstance(fog, (int, float)) else 0.0
    stats = dict(pieces=0, aims=0, unreachable=0, standoff=(so_lo, so_hi), fog=fog)
    if n_max <= 0 or not free:
        return [], stats
    n0, n1 = occupied2d.shape
    blocking = occupied2d & ~low_only2d

    # pieces: 8-connected components of the low-only map
    label = np.zeros((n0, n1), dtype=int)
    pieces: list[list[tuple[int, int]]] = []
    for a in range(n0):
        for b in range(n1):
            if not low_only2d[a, b] or label[a, b]:
                continue
            k = len(pieces) + 1
            label[a, b] = k
            stack, cells = [(a, b)], []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        xx, yy = x + dx, y + dy
                        if (0 <= xx < n0 and 0 <= yy < n1
                                and low_only2d[xx, yy] and not label[xx, yy]):
                            label[xx, yy] = k
                            stack.append((xx, yy))
            if len(cells) >= int(icfg["near_field_min_cells"]):
                pieces.append(cells)
            else:
                label[label == k] = 0     # a floater cluster or a lone small object
    stats["pieces"] = len(pieces)

    spacing = float(icfg["near_field_spacing"]) / vox
    aims: list[tuple[int, float, tuple[int, int]]] = []   # (rank in piece, -mass, cell)
    for cells in pieces:
        pts = np.array(cells, dtype=float)
        mass = np.array([low2d[c] for c in cells])
        chosen = [int(np.argmax(mass))]
        dd = np.linalg.norm(pts - pts[chosen[0]], axis=1)
        while True:
            j = int(np.argmax(dd))
            if dd[j] < spacing:
                break
            chosen.append(j)
            dd = np.minimum(dd, np.linalg.norm(pts - pts[j], axis=1))
        for rank, j in enumerate(chosen):
            aims.append((rank, -float(mass.sum()), cells[j]))
    aims.sort()
    stats["aims"] = len(aims)

    def clear(stand: tuple[int, int], aim: tuple[int, int]) -> bool:
        """No wall-like cell on the straight line from stand to aim."""
        x0, y0 = stand
        x1, y1 = aim
        n = max(abs(x1 - x0), abs(y1 - y0)) * 2
        for t in range(1, n):
            x = int(round(x0 + (x1 - x0) * t / n))
            y = int(round(y0 + (y1 - y0) * t / n))
            if (x, y) == aim:
                break
            if blocking[x, y]:
                return False
        return True

    free_pts = np.array(free, dtype=float)
    poses: list[np.ndarray] = []
    for _, _, (a, b) in aims:
        if len(poses) >= n_max:
            break
        k = label[a, b]
        aa, bb = np.meshgrid(np.arange(max(a - 1, 0), min(a + 2, n0)),
                             np.arange(max(b - 1, 0), min(b + 2, n1)), indexing="ij")
        near_cells = label[aa, bb] == k
        mass = float(low2d[aa, bb][near_cells].sum())
        height = float(lowh2d[aa, bb][near_cells].sum()) / mass
        drop = eye_h - height                      # the camera over the top
        piece_pts = np.array(pieces[k - 1], dtype=float)
        d_aim = np.linalg.norm(free_pts - (a, b), axis=1) * vox
        d_edge = np.array([np.linalg.norm(piece_pts - f, axis=1).min()
                           for f in free_pts]) * vox
        slant = np.sqrt(d_edge ** 2 + drop ** 2)
        stand = None
        for j in np.argsort(d_aim):                # the closest allowed
            if d_aim[j] < so_lo or slant[j] <= fog:
                continue
            if d_aim[j] > so_hi:
                break
            if clear(free[j], (a, b)):
                stand = free[j]
                break
        if stand is None:
            stats["unreachable"] += 1
            continue
        pos = _to_world(frame, lo0 + (stand[0] + 0.5) * vox,
                        lo1 + (stand[1] + 0.5) * vox, eye_h)
        target = _to_world(frame, lo0 + (a + 0.5) * vox, lo1 + (b + 0.5) * vox, height)
        poses.append(look_at(pos, target, up))
    return poses, stats


def aimed_cameras(targets: np.ndarray, free_positions: np.ndarray,
                  frame: SceneFrame, cfg: dict,
                  min_dist: float | None = None) -> list[np.ndarray]:
    """Cameras aimed at specific 3D targets (coverage-gap repair): for each
    target pick the free position nearest the preferred standoff distance."""
    if min_dist is None:
        min_dist = max(cfg["render"]["interior"]["min_sightline"],
                       cfg["render"]["coverage"]["min_aim_dist"])
    # The quality filter re-validates whatever standoff we pick.
    prefer = cfg["render"]["coverage"]["standoff"]
    up = frame.up_vec
    poses = []
    for t in targets:
        d = np.linalg.norm(free_positions - t, axis=1)
        ok = d > min_dist
        if not ok.any():
            continue
        cand = np.flatnonzero(ok)
        pos = free_positions[cand[np.argmin(np.abs(d[cand] - prefer))]]
        poses.append(look_at(pos, t, up))
    return poses


# ---------------------------------------------------------------------------
# Ground path (open exteriors)
# ---------------------------------------------------------------------------

_DIRS8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(radius):
        p = np.pad(m, 1, constant_values=False)
        n0, n1 = mask.shape
        acc = np.zeros_like(m)
        for da, db in _DIRS8:
            acc |= p[1 + da:1 + da + n0, 1 + db:1 + db + n1]
        m = m | acc
    return m


def ground_path(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
                cfg: dict, n_views: int, seed: int = 42,
                volume_boxes: list[dict] | None = None,
                ) -> tuple[list[np.ndarray], np.ndarray]:
    """Person-height cameras over a LOCAL ground map (open exteriors).

    The ground is NOT assumed planar (the first exterior tested had a ~4 m
    slope across the core — a global floor plane puts cameras underground at
    one end and airborne at the other). Per 2D cell: ground = robust low
    envelope of Gaussian heights; free space = low obstacle mass in a slab
    ABOVE LOCAL GROUND; cameras stand at ground + eye_height, aimed toward
    obstacle mass (the monuments/objects), not along empty sightlines.

    No insideness test — openness is legitimate outdoors; the volume's 2D
    FOOTPRINT scopes placement instead (the volume's height range scopes
    labeling/coverage, not where a person can stand: a sloped scene's eye
    height may exceed a trunk-height volume cap).

    The free-space threshold is auto-derived from the scene's own positive
 obstacle-mass percentiles (closure, built into this
    mode from the start); the choice is logged into frame.notes -> manifest.

    Returns (candidate c2w poses, free-cell world positions at eye height).
    """
    from .scene_read import ground_map   # lazy: scene_read imports this module
    gcfg = cfg["render"]["ground"]
    # The map — footprint, budget, ground envelope, obstacle mass, footing
    # — is the mode-independent half, shared with the Propose analysis.
    gm = ground_map(means, opacities, frame, cfg, volume_boxes)
    cell, (lo0, lo1), (n0, n1) = gm.cell, gm.origin, gm.dims
    ground, mass, valid, standable = gm.ground, gm.mass, gm.valid, gm.standable

    # Free-space threshold: auto-derived from the positive obstacle-mass
    # distribution. Strictest percentile that still yields enough
    # standable free cells wins, preferring generous clearance first.
    pos_mass = mass[standable & (mass > 0)]
    if pos_mass.size == 0:
        raise Refusal(
            "ground path: no obstacle mass found over standable ground; the "
            "volume footprint holds nothing to film. Redraw the volume at the "
            "volume gate (Propose fresh, or the whole-scene act), or switch "
            "the path mode on the render panel.", gate="render")
    n_pos = max(int(np.ceil(n_views / gcfg["views_per_position"])), 1)
    cands = [(p, float(np.percentile(pos_mass, p)))
             for p in gcfg["free_mass_percentiles"]]
    free = None
    chosen_th, chosen_pct, chosen_clear = None, None, None
    for clearance in range(max(gcfg["clearance_cells"], 0), -1, -1):
        for pct, th in cands:
            cand = standable & (mass <= th)
            if clearance:
                cand &= ~_dilate(mass > th, clearance)
            if int(cand.sum()) >= n_pos:
                free, chosen_th, chosen_pct, chosen_clear = cand, th, pct, clearance
                break
        if free is not None:
            break
    if free is None:
        raise Refusal(
            "ground path: no free standable cells at any auto-derived threshold "
            f"(candidates {cands}); the ground map finds no place to stand. "
            "Widen the volume at the volume gate, or switch the path mode to "
            "orbit on the render panel; render.ground.cell_size / slab_lo / "
            "slab_hi are profile-only.", gate="render")
    log.info("ground path: free-space threshold auto-derived: p%s = %.1f "
             "(clearance %d cells) -> %d free cells (need >= %d)",
             chosen_pct, chosen_th, chosen_clear, int(free.sum()), n_pos)
    rel_lo, rel_hi = gm.relief_p5_p95
    frame.notes.update(ground_free_threshold=round(chosen_th, 2),
                       ground_free_percentile=chosen_pct,
                       ground_clearance_cells=chosen_clear,
                       ground_free_cells=int(free.sum()),
                       ground_relief_p5_p95=[round(float(rel_lo), 2),
                                             round(float(rel_hi), 2)])

    # Farthest-point sampling over free cells.
    rng = np.random.default_rng(seed)
    cells = np.argwhere(free).astype(float)
    chosen = [cells[rng.integers(len(cells))]]
    d2 = np.full(len(cells), np.inf)
    while len(chosen) < min(n_pos, len(cells)):
        d2 = np.minimum(d2, ((cells - chosen[-1]) ** 2).sum(axis=1))
        chosen.append(cells[int(np.argmax(d2))])

    # Aim TOWARD obstacle mass: interior mode chases deep sightlines (crossing
    # the room); outdoors the longest ray points away from every monument into
    # open lawn. Score each yaw by distance-discounted obstacle mass along the
    # ray; reject yaws that face-plant into mass closer than min_standoff.
    eye = gcfg["eye_height"]
    max_steps = max(int(gcfg["aim_range"] / cell), 1)
    standoff_cells = gcfg["min_standoff"] / cell
    aim_floor = gcfg["aim_dist_floor"]   # scene units, like every length here

    def yaw_score(a: float, b: float, yaw: float) -> float:
        da, db = np.cos(yaw), np.sin(yaw)
        x, y = a, b
        score = 0.0
        for step in range(1, max_steps + 1):
            x += da
            y += db
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < n0 and 0 <= yi < n1):
                break
            m = mass[xi, yi] if valid[xi, yi] else 0.0
            if m > chosen_th and step < standoff_cells:
                return -1.0
            score += m / max(step * cell, aim_floor)
        return score

    poses = []
    up = frame.up_vec
    min_sep = np.deg2rad(gcfg["min_yaw_sep_deg"])
    for c in chosen:
        a, b = c
        scored = sorted(((yaw_score(a, b, yaw), yaw)
                         for yaw in np.linspace(-np.pi, np.pi, 36, endpoint=False)),
                        reverse=True)
        yaws: list[float] = []
        for s, yaw in scored:
            if s < 0:
                break
            if all(min(abs(yaw - y), 2 * np.pi - abs(yaw - y)) >= min_sep for y in yaws):
                yaws.append(yaw)
            if len(yaws) >= gcfg["views_per_position"]:
                break
        p0 = lo0 + (a + 0.5) * cell
        p1 = lo1 + (b + 0.5) * cell
        eye_h = ground[int(a), int(b)] + eye
        pos = _to_world(frame, p0, p1, eye_h)
        for yaw in yaws:
            for pitch in gcfg["pitches_deg"]:
                t0 = p0 + np.cos(yaw) * 3.0
                t1 = p1 + np.sin(yaw) * 3.0
                t_h = eye_h + 3.0 * np.tan(np.deg2rad(pitch))
                poses.append(look_at(pos, _to_world(frame, t0, t1, t_h), up))
    log.info("ground path: %d candidate poses from %d positions", len(poses), len(chosen))

    free_idx = np.argwhere(free)
    free_world = np.stack([
        _to_world(frame, lo0 + (a + 0.5) * cell, lo1 + (b + 0.5) * cell,
                  ground[a, b] + eye)
        for a, b in free_idx
    ])
    return poses, free_world


# ---------------------------------------------------------------------------
# Orbit path
# ---------------------------------------------------------------------------

def orbit_path(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
               cfg: dict, n_views: int,
               volume_boxes: list | None = None) -> list[np.ndarray]:
    """Ring the VOLUME when one exists, the whole scene otherwise.

    Orbit was the one path mode that never received `volume_boxes`, so on a
    focus scene it ringed the opacity-weighted scene centroid and framed the
    room instead of the subject — and none of the `focus_*` knobs could reach
    it, they are read by `interior_path` alone. Centring the ring on the
    volume makes `auto` landing on orbit a degraded outcome rather than a
    silent one. No volume -> the previous behaviour, unchanged.
    """
    ocfg = cfg["render"]["orbit"]
    if volume_boxes:
        lo = np.min([b["min"] for b in volume_boxes], axis=0)
        hi = np.max([b["max"] for b in volume_boxes], axis=0)
        center = (lo + hi) / 2.0
        # Half-diagonal of the volume: the smallest sphere that contains it,
        # then the same radius_scale stand-off the scene-wide path uses.
        radius = float(np.linalg.norm(hi - lo)) / 2.0
        log.info("orbit path: ringing the volume (half-diagonal %.2f units) "
                 "rather than the scene centroid", radius)
    else:
        center = np.average(means, axis=0, weights=opacities)
        radius = float(np.percentile(np.linalg.norm(means - center, axis=1), 90))
    radius *= ocfg["radius_scale"]
    up = frame.up_vec
    elevations = ocfg["elevations_deg"]
    per_ring = int(np.ceil(n_views / len(elevations)))
    poses = []
    for elev in elevations:
        el = np.deg2rad(elev)
        for k in range(per_ring):
            az = 2 * np.pi * k / per_ring
            g0 = center[frame.ground_axes[0]] + radius * np.cos(el) * np.cos(az)
            g1 = center[frame.ground_axes[1]] + radius * np.cos(el) * np.sin(az)
            h = center[frame.up_axis] * frame.up_sign + radius * np.sin(el)
            poses.append(look_at(_to_world(frame, g0, g1, h), center, up))
    return poses
