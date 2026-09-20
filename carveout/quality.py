# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The view-quality filter's discard reasons: which knob each answers to,
the keys the render gate must be able to change, and the diagnoses the
gate and the refusal text are built from.

Deliberately free of torch and of every heavy import: the web server reads
these on the scene-open path and at every render-gate request, and
`carveout web` must start without loading a CUDA stack. The filter itself
(`render._quality`) and the rasteriser stay in render.py.
"""

import logging

log = logging.getLogger(__name__)


# Which knob each discard reason answers to: the stat the filter judged, the
# threshold it was judged against, and whether passing means going ABOVE that
# threshold (min_*) or BELOW it (max_*).
_DISCARD_KNOBS = {
    "low_coverage": ("coverage", "min_coverage", "above",
                     "too little scene in frame; check the volume and the "
                     "camera path before loosening this"),
    "inside_geometry": ("median_depth", "min_median_depth", "above",
                        "cameras are sitting inside geometry; try "
                        "render.interior.cameras_in_volume: false with "
                        "focus_aim: true before lowering this"),
    "uniform_frame": ("rgb_std", "min_rgb_std", "above",
                      "frames are near-uniform (wall stares)"),
    "near_field_fog": ("near_alpha", "max_near_alpha", "below",
                       "near-camera junk; on a scene smaller than a few "
                       "metres the fog horizon (near_field_depth) can exceed "
                       "the room, which reads normal geometry as fog"),
    "blurry_mush": ("sharpness", "min_sharpness", "above",
                    "frames lack high-frequency detail"),
}


# Quality-filter knobs the RENDER GATE must be able to change.
#
# The invariant: every threshold a discard reason can NAME is adjustable from
# the gate that shows the refusal. A refusal reading "adjust ONE threshold in
# the render gate's calibration panel" beside a panel that cannot reach that
# threshold is a dead end — the operator is handed the number, the knob and
# the instruction, and no way to act. DERIVED from _DISCARD_KNOBS rather than
# listed again, so a new discard reason widens the panel with it and the two
# cannot drift apart.
GATE_QUALITY_KEYS = {thr for _, thr, _, _ in _DISCARD_KNOBS.values()} | {
    # Named by the near_field_fog NOTE rather than as a threshold in its own
    # right: it sets the horizon that makes near_alpha mean anything, and on a
    # scene a few metres across a 1.2 m horizon reads ordinary room geometry as
    # fog. On such a scene it is the knob that actually moves, so lowering
    # max_near_alpha alone cannot fix what it diagnoses.
    "near_field_depth",
}


def resolve_auto_quality(cfg: dict, frame) -> tuple[dict, dict]:
    """Resolve any "auto" quality threshold against the scene's own size.

    Both auto-able knobs are DISTANCES, and a distance calibrated in a 6 m
    room means nothing in a 2 m one. The reference length is the smallest
    HORIZONTAL extent — horizontal because these filters ask how far the
    camera is from the things in front of it, and a room's height says
    nothing about that. The resolution is in scene units and needs no
    factor; the metre figure is recorded only under a RECORDED factor.

    Returns (cfg with the quality filter resolved, a record for the manifest).
    The caller's cfg is NOT mutated — the run rebinds its own copy, so a
    cached config object cannot accumulate one scene's resolution.
    """
    qf = dict(cfg["render"]["quality_filter"])
    fracs = qf.pop("auto_frac", None) or {}
    up = frame.up_axis
    scale = float(getattr(frame, "scale", 1.0))
    metric = getattr(frame, "scale_source", "recorded") == "recorded"
    extent_units = frame.hi - frame.lo
    # The resolved threshold is compared against rendered depths in SCENE
    # UNITS, so it is stated in units — frac x width_units — and is the
    # same number whatever the factor. The metre figure beside it exists
    # only when a factor is recorded: no metres are spoken for a scene
    # sized by the estimate.
    ref_units = float(min(v for i, v in enumerate(extent_units) if i != up))
    ref = ref_units * scale if metric else None
    record: dict = {}
    for key, frac in fracs.items():
        if str(qf.get(key)) != "auto":
            continue
        qf[key] = round(frac * ref_units, 4)
        record[key] = qf[key]
        record[key + "_m"] = round(frac * ref, 4) if ref is not None else None
        log.info("quality filter: %s auto -> %.4g scene units (%.3g x the "
                 "scene's %.4g-unit width%s), a DISCOVERED calibration, "
                 "not a default", key, qf[key], frac, ref_units,
                 f" = {record[key + '_m']:.4g} m" if ref is not None
                 else ", no factor recorded")
    # A leftover "auto" is a knob nobody taught to resolve. Fail loudly rather
    # than compare a float against the string later, deep in the filter.
    for key, val in qf.items():
        if isinstance(val, str) and val == "auto":
            raise ValueError(
                f"render.quality_filter.{key} is 'auto', but only "
                f"{sorted(fracs) or 'nothing'} can be resolved automatically. "
                f"Set it to a number, or add its fraction under "
                f"render.quality_filter.auto_frac.")
    if record:
        record["reference_width_units"] = round(ref_units, 4)
        record["reference_width_m"] = round(ref, 4) if ref is not None else None
    cfg = {**cfg, "render": {**cfg["render"], "quality_filter": qf}}
    return cfg, record


# The judged stats that are DISTANCES. The filter measures them in scene
# units (the config was converted before any view was rendered), but the
# operator's knob for them is the profile's number — metres under a recorded
# factor, the estimate's metres otherwise — which is what the render gate's
# panel shows. A diagnosis that quotes units against a panel in metres sends
# the operator to type 78.5 into a field reading 1 (seen 78x apart on a
# scene sized by an estimate; on a metric scene the two coincide, which is
# why it went unseen).
_DISTANCE_STATS = {"median_depth"}


def discard_diagnosis(discarded: list[dict], qcfg: dict,
                      scale: float = 1.0, source: str = "recorded",
                      measured: dict | None = None) -> list[dict]:
    """Per-reason breakdown of a discard log: how many, what the judged stat
    actually measured, and how far the CLOSEST frame was from passing.

    The closest value is the operator's number — it says what a threshold
    would have to become to admit at least one frame, which a bare count
    never tells you. `scale` (metres per scene unit, the run's one factor)
    carries the distance rows back to the profile's own numbers:
    `*_profile` fields beside the unit values; `source` / `measured` say
    whether the factor was recorded or measured with the ruler, so the
    text can mark the metres "≈" after a measurement."""
    out = []
    for reason in sorted({d.get("reason") for d in discarded} - {None},
                         key=lambda r: -sum(d.get("reason") == r
                                            for d in discarded)):
        rows = [d for d in discarded if d.get("reason") == reason]
        knob = _DISCARD_KNOBS.get(reason)
        entry = dict(reason=reason, count=len(rows))
        if knob:
            stat_key, thr_key, sense, note = knob
            vals = sorted(v for v in (r.get(stat_key) for r in rows)
                          if isinstance(v, (int, float)))
            if vals:
                mid = vals[len(vals) // 2]
                # "closest to passing" depends on which way the test runs
                closest = vals[-1] if sense == "above" else vals[0]
                entry.update(stat=stat_key, threshold_key=thr_key,
                             threshold=qcfg.get(thr_key), sense=sense,
                             median=mid, closest=closest, note=note)
                thr = qcfg.get(thr_key)
                if (stat_key in _DISTANCE_STATS and scale != 1.0
                        and isinstance(thr, (int, float))):
                    entry.update(
                        scale_m_per_unit=scale, scale_source=source,
                        measured=measured,
                        threshold_profile=float(thr) * scale,
                        median_profile=float(mid) * scale,
                        closest_profile=float(closest) * scale)
        out.append(entry)
    return out


def judged_stat(d: dict, qcfg_profile: dict, scale: float = 1.0,
                auto: dict | None = None) -> dict | None:
    """What ONE discarded view failed on: the stat the filter judged, its
    value, the threshold, and which way passing lies — for the render gate's
    discard table, which showed sharpness for every row whatever the reason.
    `qcfg_profile` is the quality filter as the profile writes it (the
    manifest's echo); `auto` the resolved "auto" knobs in scene units.
    Distance rows carry both the profile's numbers and the scene units, so
    the panel can speak whichever the factor's source allows."""
    knob = _DISCARD_KNOBS.get(d.get("reason"))
    if not knob:
        return None
    stat, thr_key, sense, _ = knob
    v = d.get(stat)
    t = qcfg_profile.get(thr_key)
    dist = stat in _DISTANCE_STATS
    if str(t) == "auto":
        t_units = (auto or {}).get(thr_key)
        t = t_units * scale if isinstance(t_units, (int, float)) else None
    if not isinstance(t, (int, float)):
        t = None
    out = dict(stat=stat, threshold_key=thr_key, sense=sense, distance=dist,
               value=v, threshold=t)
    if dist:
        out.update(value_units=v, threshold_units=(
            t / scale if t is not None and scale else None))
        if isinstance(v, (int, float)):
            out["value"] = v * scale
    return out


def format_discard_diagnosis(rows: list[dict], attempted: int) -> str:
    """The refusal text: what rejected the views, and which way to move."""
    lines = [f"all {attempted} candidate view(s) were discarded by the "
             f"quality filter; there is nothing to render.",
             "what rejected them (most common first):"]
    for r in rows:
        if "stat" not in r:
            lines.append(f"  {r['count']:>4}x {r['reason']}")
            continue
        limit = "min" if r["sense"] == "above" else "max"
        move = "lower" if r["sense"] == "above" else "raise"
        if "threshold_profile" in r:
            # A distance under a factor other than 1: the knob's numbers
            # first (what the panel and the profile show), the measured
            # scene units after, so the two never read as one scale.
            lines.append(
                f"  {r['count']:>4}x {r['reason']}: {r['stat']} median "
                f"{r['median_profile']:.4g}, closest to passing "
                f"{r['closest_profile']:.4g} ({limit} "
                f"{r['threshold_profile']:.4g}); {move} "
                f"render.quality_filter.{r['threshold_key']} past "
                f"{r['closest_profile']:.4g} to admit that frame")
            at = (f"at {r['scale_m_per_unit']:.4g} m per scene unit "
                  f"({r.get('scale_source', 'recorded')})")
            lines.append(
                f"        (the profile's numbers, {at}; measured in "
                f"scene units: median {r['median']:.4g}, closest "
                f"{r['closest']:.4g}, {limit} {r['threshold']:.4g})")
        else:
            lines.append(
                f"  {r['count']:>4}x {r['reason']}: {r['stat']} median "
                f"{r['median']:.4g}, closest to passing {r['closest']:.4g} "
                f"({limit} {r['threshold']}); {move} "
                f"render.quality_filter.{r['threshold_key']} past "
                f"{r['closest']:.4g} to admit that frame")
        lines.append(f"        {r['note']}")
    return "\n".join(lines)
