# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scene analysis behind the two Check cards (ask the operator, do
not guess the scene).

Pure numpy over `(means, opacities, frame, cfg, volume_boxes)`: no torch, no
file writes. Nothing here classifies a scene. The operator answered two
questions — what a unit is (the ruler) and how the scene was filmed (the
tile) — and this module CHECKS those answers against the geometry
(`check_answers`: stated on the cards, never applied) and proposes the two
measured settings that remain (`propose_settings`: tight focus, the
small-scene thresholds), adopted by an explicit act.

`ground_map` is the mode-independent half of the ground path (`cameras.
ground_path`): the local ground envelope, the obstacle mass above it and
where a person could stand. The ground path consumes it to place cameras;
the analysis reads its relief and standable fraction to tell an open site
from an object. One computation, two readers — the ground path's output
must not move when this module changes (the byte-identity check on the
an outdoor site, done when the ground read was added).
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from .cameras import SceneFrame, check_grid_budget

log = logging.getLogger(__name__)

_DIRS8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _neighbor_stack(arr: np.ndarray) -> np.ndarray:
    """(8, n0, n1) stack of the 8-neighbor values, NaN beyond the border."""
    p = np.pad(arr, 1, constant_values=np.nan)
    n0, n1 = arr.shape
    return np.stack([p[1 + da:1 + da + n0, 1 + db:1 + db + n1] for da, db in _DIRS8])


@dataclass
class GroundMap:
    """A local ground map over a 2D footprint, in scene units.

    `ground` is the per-cell low envelope of Gaussian heights (signed,
    up-positive; NaN where too few points), `mass` the opacity in the
    obstacle slab above it, `valid` where a ground estimate exists,
    `standable` where footing is trustworthy (valid in the full
    8-neighbourhood and, with a volume, inside a box footprint)."""

    ground: np.ndarray
    mass: np.ndarray
    valid: np.ndarray
    standable: np.ndarray
    cell: float                       # scene units per cell
    origin: tuple[float, float]       # (lo0, lo1) of the footprint
    dims: tuple[int, int]
    relief_p5_p95: tuple[float, float]   # 5th/95th percentile of the ground
    standable_frac: float             # standable cells / valid cells
    populated_frac: float             # valid cells / all cells

    def record(self) -> dict:
        return dict(relief_p5_p95=[round(float(self.relief_p5_p95[0]), 4),
                                   round(float(self.relief_p5_p95[1]), 4)],
                    standable_frac=round(self.standable_frac, 4),
                    populated_frac=round(self.populated_frac, 4),
                    cell_units=float(self.cell),
                    dims=[int(self.dims[0]), int(self.dims[1])])


def ground_map(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
               cfg: dict, volume_boxes: list[dict] | None,
               cell: float | None = None, budget: bool = True) -> GroundMap:
    """The local ground map the ground path places cameras on.

    The ground is NOT assumed planar (the first exterior tested had a ~4 m
    slope across the core — a global floor plane puts cameras underground
    at one end and airborne at the other). Per 2D cell: ground = robust low
    envelope of Gaussian heights; obstacle mass = opacity in a slab ABOVE
    LOCAL GROUND. No insideness test — openness is legitimate outdoors; the
    volume's 2D FOOTPRINT scopes the map instead (the volume's height range
    scopes labeling/coverage, not where a person can stand: a sloped
    scene's eye height may exceed a trunk-height volume cap).

    `cell` overrides `render.ground.cell_size` (the analysis coarsens it to
    fit the footprint budget); `budget=True` refuses over the budget the way
    the ground path always did, to the render gate.
    """
    gcfg = cfg["render"]["ground"]
    cell = float(gcfg["cell_size"] if cell is None else cell)
    a0, a1 = frame.ground_axes

    # Footprint: volume union bbox when present, else robust scene bounds.
    if volume_boxes:
        lo0 = min(b["min"][a0] for b in volume_boxes)
        hi0 = max(b["max"][a0] for b in volume_boxes)
        lo1 = min(b["min"][a1] for b in volume_boxes)
        hi1 = max(b["max"][a1] for b in volume_boxes)
    else:
        lo0, hi0 = frame.lo[a0], frame.hi[a0]
        lo1, hi1 = frame.lo[a1], frame.hi[a1]
    n0 = max(int((hi0 - lo0) / cell) + 1, 4)
    n1 = max(int((hi1 - lo1) / cell) + 1, 4)
    if budget:
        # The same footprint budget as the interior path: the ground
        # map is several full-footprint arrays and one percentile per
        # populated cell, over the volume's footprint when there is one,
        # else the scene's.
        check_grid_budget("ground camera placement", (n0, n1), cell, frame,
                          int(cfg["render"]["grid_budget"]["footprint_cells"]),
                          "footprint_cells", gate="render",
                          escape=", knowing the ground map is a handful of "
                                 "8-byte-per-cell arrays and one height "
                                 "percentile per populated cell (vectorised, "
                                 "no ray walk).")

    # Points feeding the ground/obstacle maps: opaque enough, in-volume when
    # a volume exists (its height cap keeps canopy out of the ground
    # estimate).
    sel = opacities >= gcfg["min_opacity"]
    if volume_boxes:
        inv = np.zeros(len(means), dtype=bool)
        for b in volume_boxes:
            inv |= ((means >= b["min"]) & (means <= b["max"])).all(axis=1)
        sel &= inv
    pts = means[sel]
    w = opacities[sel]
    h = pts[:, frame.up_axis] * frame.up_sign  # signed height, up-positive

    i0 = ((pts[:, a0] - lo0) / cell).astype(int)
    i1 = ((pts[:, a1] - lo1) / cell).astype(int)
    inb = (i0 >= 0) & (i0 < n0) & (i1 >= 0) & (i1 < n1)
    i0, i1, h, w = i0[inb], i1[inb], h[inb], w[inb]
    flat = i0 * n1 + i1

    # Local ground: per-cell low percentile of heights, min-count gated.
    order = np.argsort(flat, kind="stable")
    flat_s, h_s = flat[order], h[order]
    starts = np.searchsorted(flat_s, np.arange(n0 * n1))
    ends = np.searchsorted(flat_s, np.arange(n0 * n1) + 1)
    ground = np.full(n0 * n1, np.nan)
    enough = np.flatnonzero(ends - starts >= gcfg["min_cell_points"])
    for fid in enough:
        ground[fid] = np.percentile(h_s[starts[fid]:ends[fid]], gcfg["ground_pct"])
    ground = ground.reshape(n0, n1)

    # Fill small holes (paths/lawns with sparse splats) from neighbors, then
    # median-smooth so single-cell spikes don't fake terrain steps.
    # (nanmedian warns on an all-NaN column — an empty cell far from any
    # populated one, expected on any open map — so the warning is muted.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(gcfg["fill_iters"]):
            nb = _neighbor_stack(ground)
            have = (~np.isnan(nb)).sum(axis=0)
            fill = np.isnan(ground) & (have >= 3)
            if not fill.any():
                break
            ground[fill] = np.nanmedian(nb, axis=0)[fill]
        nb = _neighbor_stack(ground)
        valid = ~np.isnan(ground)
        smooth = np.nanmedian(np.concatenate([nb, ground[None]], axis=0), axis=0)
        ground[valid] = smooth[valid]

    # Obstacle mass: opacity in the slab ABOVE LOCAL GROUND, per cell.
    g_at_pt = ground[i0, i1]
    rel = h - g_at_pt
    slab = ~np.isnan(g_at_pt) & (rel >= gcfg["slab_lo"]) & (rel <= gcfg["slab_hi"])
    mass = np.bincount(flat[slab], weights=w[slab], minlength=n0 * n1).reshape(n0, n1)

    # Placement requires trustworthy footing: a valid ground estimate in the
    # full 8-neighborhood (rules out canopy-fringe and map-edge cells) and,
    # with a volume, a cell center inside some box footprint.
    nb_valid = (~np.isnan(_neighbor_stack(ground))).sum(axis=0) == 8
    standable = valid & nb_valid
    if volume_boxes:
        c0 = lo0 + (np.arange(n0) + 0.5) * cell
        c1 = lo1 + (np.arange(n1) + 0.5) * cell
        cc0, cc1 = np.meshgrid(c0, c1, indexing="ij")
        infoot = np.zeros((n0, n1), dtype=bool)
        for b in volume_boxes:
            infoot |= ((cc0 >= b["min"][a0]) & (cc0 <= b["max"][a0])
                       & (cc1 >= b["min"][a1]) & (cc1 <= b["max"][a1]))
        standable &= infoot

    n_valid = int(valid.sum())
    if n_valid:
        rel_lo, rel_hi = np.nanpercentile(ground[valid], [5, 95])
    else:
        rel_lo = rel_hi = float("nan")
    return GroundMap(ground=ground, mass=mass, valid=valid, standable=standable,
                     cell=cell, origin=(float(lo0), float(lo1)), dims=(n0, n1),
                     relief_p5_p95=(float(rel_lo), float(rel_hi)),
                     standable_frac=(int(standable.sum()) / n_valid
                                     if n_valid else 0.0),
                     populated_frac=n_valid / float(n0 * n1))


# ---------------------------------------------------------------------------
# The analysis, the checks and the settings proposal
# ---------------------------------------------------------------------------

def _footprint_frac(frame: SceneFrame, boxes: list[dict] | None) -> float:
    """The volume's footprint area over the scene's robust footprint area;
    1.0 without boxes. Boxes are summed and capped at 1 (the proposer's
    components do not overlap; hand-drawn ones rarely do)."""
    if not boxes:
        return 1.0
    a0, a1 = frame.ground_axes
    scene_area = float((frame.hi[a0] - frame.lo[a0]) * (frame.hi[a1] - frame.lo[a1]))
    if scene_area <= 0:
        return 1.0
    area = sum(float((np.asarray(b["max"])[a0] - np.asarray(b["min"])[a0])
                     * (np.asarray(b["max"])[a1] - np.asarray(b["min"])[a1]))
               for b in boxes)
    return float(min(area / scene_area, 1.0))


def analyse(means: np.ndarray, opacities: np.ndarray, frame: SceneFrame,
            cfg: dict, boxes: list[dict] | None, proposal_used: str,
            proposal_reason: str | None = None) -> dict:
    """Everything the two cards read, in scene units. `cfg` is the run's
    config in SCENE UNITS (`scene_units_cfg` applied); `proposal_used` is
    which branch the proposer took (`enclosed` | `density_core`) and
    `proposal_reason` why, when it fell back (`over_budget` |
    `nothing_enclosed`) — under the *around a subject* and *outdoors*
    answers the density core runs directly and nothing was scanned for
    enclosure (reason `not_scanned`)."""
    up = frame.up_axis
    a0, a1 = frame.ground_axes
    extent = (frame.hi - frame.lo).astype(float)
    span = float(extent[up])
    footprint = [float(extent[a0]), float(extent[a1])]
    # Geometric mean of the two horizontal extents over the height: a site
    # is wide in both directions relative to its height; a street (long,
    # narrow) still counts as wide; an object or a building seen from
    # outside is about as tall as it is wide.
    fo_h = float(np.sqrt(footprint[0] * footprint[1]) / span) if span > 0 else 0.0

    # The ground map over the proposal's boxes, its cell coarsened to the
    # footprint budget: a read never refuses.
    gcfg = cfg["render"]["ground"]
    limit = int(cfg["render"]["grid_budget"]["footprint_cells"])
    cell = float(gcfg["cell_size"])
    if boxes:
        w0 = max(b["max"][a0] for b in boxes) - min(b["min"][a0] for b in boxes)
        w1 = max(b["max"][a1] for b in boxes) - min(b["min"][a1] for b in boxes)
    else:
        w0, w1 = footprint
    coarsened = 0
    while (int(w0 / cell) + 1) * (int(w1 / cell) + 1) > limit:
        cell *= 2
        coarsened += 1
    gm = ground_map(means, opacities, frame, cfg, boxes, cell=cell, budget=False)
    ground = dict(gm.record(), coarsened=coarsened)

    return dict(
        up_axis=int(up),
        extent_units=[round(float(v), 6) for v in extent],
        span_units=round(span, 6),
        floor_signed=(None if frame.floor is None else round(float(frame.floor), 6)),
        footprint_units=[round(v, 6) for v in footprint],
        footprint_over_height=round(fo_h, 4),
        volume_footprint_frac=round(_footprint_frac(frame, boxes), 4),
        enclosure=dict(used=proposal_used, reason=proposal_reason),
        ground=ground,
        scale=frame.notes.get("scale"),
    )


def fmt_m(metres: float) -> str:
    """A length in metres to three significant digits: the cards speak
    metres, since every scene that reaches a card has a factor."""
    return f"{float(f'{metres:.3g}'):g} m"


FILMED = {"interior": "inside a space", "orbit": "around a subject",
          "ground": "outdoors on the ground", "manual": "manual views"}


def check_answers(analysis: dict, cfg_m: dict, read: dict | None = None) -> dict:
    """The geometry's opinion of the operator's two answers — and the
    model's of the measurement — STATED, never applied. `cfg_m` is the
    PROFILE's config (metres; `render.path_mode` is the filming answer);
    `read` the second opinion's record (`{length_m, what, reason, …}`,
    `{error}` or None).

    -> {path_mode: {said, fits, reason}, scale: {fits, reason} | None,
        opinion: {fits, reason} | None}. `fits` is True (agrees), False
    (flagged) or None (nothing to judge by, with the reason saying why).
    """
    said = str(cfg_m["render"]["path_mode"])
    enc = analysis["enclosure"]
    scanned = enc["used"] == "enclosed" or enc.get("reason") == "nothing_enclosed"
    enclosed = enc["used"] == "enclosed"
    g = analysis["ground"]
    fo_h = float(analysis["footprint_over_height"])
    standable = float(g["standable_frac"])
    scale = float((analysis.get("scale") or {}).get("scale") or 1.0)
    fp = analysis["footprint_units"]
    walk_m = float(np.sqrt(fp[0] * fp[1])) * scale
    span_m = float(analysis["span_units"]) * scale

    # --- the filming answer against the geometry ---------------------------
    # twice the room path's minimum sightline: under it the room path
    # stands its cameras in the furniture (the Demo nook, 2.0 x 2.6 m: all
    # 18 candidates mush): 3 m, stated as a check
    walk_min_m = 2.0 * float(cfg_m["render"]["interior"]["min_sightline"])
    if said == "interior":
        if enclosed and walk_m < walk_min_m:
            pm = dict(fits=False, reason=f"an enclosed space {fmt_m(walk_m)} "
                                         f"across, too small to walk (under "
                                         f"{fmt_m(walk_min_m)}): around a "
                                         f"subject may film it better")
        elif enclosed:
            pm = dict(fits=True, reason="an enclosed region was found, walls "
                                        "around a floor")
        elif enc.get("reason") == "over_budget":
            pm = dict(fits=None, reason="the enclosure scan was over budget, "
                                        "so nothing is known about the walls")
        else:
            pm = dict(fits=False, reason="nothing enclosed was found; if this "
                                         "is outdoors, pick outdoors on the "
                                         "render panel")
    elif said == "ground":
        if enclosed:
            pm = dict(fits=False, reason="an enclosed region was found: inside "
                                         "a space may film it better")
        elif standable < 0.2 or fo_h < 2.0:
            why = []
            if standable < 0.2:
                why.append(f"little standable ground ({100 * standable:.0f} % "
                           f"of the map)")
            if fo_h < 2.0:
                why.append(f"not wide (the footprint is {fo_h:.1f}x the height)")
            pm = dict(fits=False, reason=" / ".join(why) + ": around a subject "
                                         "may fit better")
        else:
            pm = dict(fits=True, reason=f"a person could stand on "
                                        f"{100 * standable:.0f} % of the mapped "
                                        f"ground and the footprint is "
                                        f"{fo_h:.1f}x the height")
    elif said == "orbit":
        if enclosed and walk_m > 3.0:
            pm = dict(fits=False, reason=f"an enclosed space big enough to walk "
                                         f"({fmt_m(walk_m)} across): inside a "
                                         f"space may film it better")
        elif not scanned and fo_h >= 3.0 and standable >= 0.2 and walk_m > 3.0:
            pm = dict(fits=False, reason=f"wide, standable ground ({fmt_m(walk_m)} "
                                         f"across, {fo_h:.1f}x the height, "
                                         f"standable on {100 * standable:.0f} %): "
                                         f"outdoors on the ground may film it "
                                         f"better")
        else:
            pm = dict(fits=True, reason=f"{fmt_m(walk_m)} across, "
                                        f"{fo_h:.1f}x the height: a ring around "
                                        f"it fits")
    elif said == "manual":
        pm = dict(fits=True, reason="manual views: the cameras are yours")
    else:
        pm = dict(fits=False, reason="no filming answer yet; pick how it was "
                                     "filmed on the render panel")
    pm["said"] = said

    # --- the measurement against the room's height (inside only) ---------
    sc = None
    if said == "interior":
        lo_m, hi_m = cfg_m["scene"]["plausible_height_m"]
        how = "measurement" if (analysis.get("scale") or {}).get(
            "source") == "measured" else "scale"
        if lo_m <= span_m <= hi_m:
            sc = dict(fits=True, reason=f"the room reads {fmt_m(span_m)} tall")
        else:
            # Two things can be wrong and the geometry cannot tell which:
            # the measurement (a wrong length, a wrong point) or the filming
            # answer (a wooded site said "inside a space" reads 16.4 m tall
            # because the enclosure scan takes trees and stones for walls).
            # Name both; the operator knows which.
            what = ("check the two points" if how == "measurement"
                    else "check the factor")
            sc = dict(fits=False, reason=f"at this {how} the room reads "
                                         f"{fmt_m(span_m)} tall: {what}, or "
                                         f"the filming answer if this is not "
                                         f"an interior")

    # --- the second opinion against the typed length ------------------------
    op = None
    if read:
        typed = ((analysis.get("scale") or {}).get("measured") or {}).get("length_m")
        if read.get("error"):
            op = dict(fits=None, reason=str(read["error"]))
        elif typed:
            model_m = float(read["length_m"])
            what = f" ({read['what']})" if read.get("what") else ""
            ratio = max(model_m / float(typed), float(typed) / model_m)
            if ratio <= 2.0:
                op = dict(fits=True, reason=f"the model reads A–B as about "
                                            f"{fmt_m(model_m)}{what}, which agrees "
                                            f"with the {fmt_m(float(typed))} "
                                            f"you typed")
            else:
                op = dict(fits=False, reason=f"the model reads A–B as about "
                                             f"{fmt_m(model_m)}{what}, but you "
                                             f"typed {fmt_m(float(typed))}: "
                                             f"check the two points")
        else:
            op = dict(fits=None, reason="the second opinion ran, but the "
                                        "measurement's length is missing")
    return dict(path_mode=pm, scale=sc, opinion=op)


def propose_settings(analysis: dict, cfg_m: dict) -> dict:
    """The two MEASURED proposals that remain: tight focus when the volume
    is under a quarter of the scene's footprint on a scene filmed inside a
    space (cameras roam the room, aimed at the volume); each `auto_frac`
    distance threshold at `frac x width_units x scale` in the profile's
    metres, proposed only when at least 5 % BELOW the profile's current
    value (the fractions are the defaults over a 6.1 m lab, so a room
    resolves to 0.999 against 1.0 — noise). Never a larger threshold.
    `cfg_m` is the PROFILE's config: the numbers the operator would type."""
    said = str(cfg_m["render"]["path_mode"])
    scale = float((analysis.get("scale") or {}).get("scale") or 1.0)
    reasons = []
    settings = dict(cameras_in_volume=True, focus_aim=False)
    vff = analysis["volume_footprint_frac"]
    if said == "interior" and vff < 0.25:
        settings = dict(cameras_in_volume=False, focus_aim=True)
        reasons.append(f"the volume is {100 * vff:.0f} % of the scene's "
                       f"footprint, a subject inside a room: cameras roam "
                       f"the room and aim at the volume (tight focus)")

    thresholds: dict = {}
    qf = cfg_m["render"]["quality_filter"]
    fracs = qf.get("auto_frac") or {}
    # the smallest HORIZONTAL extent, as `resolve_auto_quality` uses
    ref_units = min(v for i, v in enumerate(analysis["extent_units"])
                    if i != analysis["up_axis"])
    for key, frac in fracs.items():
        current = qf.get(key)
        if not isinstance(current, (int, float)):
            continue   # already "auto" (resolves itself) or unset
        units = float(frac) * ref_units
        proposed = round(units * scale, 4)
        if proposed < 0.95 * float(current):
            thresholds[key] = dict(proposed=proposed, current=float(current),
                                   proposed_units=round(units, 4))
    if thresholds:
        reasons.append(f"a small scene ({fmt_m(ref_units * scale)} at its "
                       f"narrowest): the distance thresholds lowered to a "
                       f"fraction of its width, below the profile's values")
    return dict(settings=settings, thresholds=thresholds, reasons=reasons)
