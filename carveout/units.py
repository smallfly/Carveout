# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Metre knobs -> scene units, at use time, by ONE factor.

Every placement, filter and lift distance in the config is a METRE value
(`eye_height 1.6`, the 0.3-2.2 obstacle slab, `instance_voxel 0.15`). The
scenes that calibrated them were human-scale throughout: over 13 metric
runs from 2 m to 56 m across, 21 of the 28 keys never moved from their
default — they encode the size of the objects being detected, not the
size of the scene. The Gaussians are never rescaled (that would
invalidate every cached stage and every volume already drawn); the knobs
are converted instead, once, at the entry of each stage:
`scene_units_cfg` divides every LENGTH_KEYS value by the factor and
records what it did for the manifest.

The factor is ONE number per run (`resolve_scale`), and it is always the
operator's (nothing guesses a scene's size):

    recorded   scene.scale_m_per_unit is a number and scene.measured is
               null — the scene was declared metric or given a factor at
               creation or at the volume gate. Exact at 1.0 (`x / 1.0`).
    measured   scene.scale_m_per_unit is a number derived by the RULER at
               the volume gate: the operator clicked two points on
               something whose real length they know and typed it;
               scene.measured records the act (the length in metres, the
               distance in units, the label and unit they typed). The
               panels then speak metres marked "≈".

No factor recorded and nothing measured -> the run REFUSES, to the
volume gate: the grids and stand-offs are sized in metres and cannot be
sized without a factor, and a default factor would be a guess about the
scene. The earlier `estimated` source (a size word over the vertical
extent) is gone; a profile still carrying
`scene.size_class` refuses loudly rather than mapping it to anything.

Downstream stages read the record from the stage-1 manifest
(`scale_from_stage1`) rather than recomputing it, so one run uses one
factor throughout. Coordinates (`scene.floor`, volume boxes,
`volume_proposal.height_range`) are NOT converted: they are positions in
scene units already.
"""

import copy
import math
from dataclasses import dataclass

from .refusal import Refusal

# (path, kind): "n" a number, "l" a list of numbers, "ln" a list or null.
# A string value ("auto") is left alone — `resolve_auto_quality` resolves
# it in scene units itself.
LENGTH_KEYS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("scene", "hist_bin_m"), "n"),
    (("render", "near_plane"), "n"),
    (("render", "far_plane"), "n"),
    (("render", "volume_proposal", "floor_margin"), "n"),
    (("render", "interior", "eye_height"), "n"),
    (("render", "interior", "voxel_size"), "n"),
    (("render", "interior", "slab_lo"), "n"),
    (("render", "interior", "slab_hi"), "n"),
    (("render", "interior", "ceiling_margin"), "n"),
    (("render", "interior", "min_sightline"), "n"),
    (("render", "interior", "near_field_top"), "n"),
    (("render", "interior", "near_field_spacing"), "n"),
    (("render", "interior", "near_field_standoff"), "l"),
    (("render", "interior", "focus_standoff"), "l"),
    (("render", "interior", "focus_macro_standoff"), "ln"),
    (("render", "interior", "focus_heights"), "ln"),
    (("render", "ground", "eye_height"), "n"),
    (("render", "ground", "cell_size"), "n"),
    (("render", "ground", "slab_lo"), "n"),
    (("render", "ground", "slab_hi"), "n"),
    (("render", "ground", "min_standoff"), "n"),
    (("render", "ground", "aim_range"), "n"),
    (("render", "ground", "aim_dist_floor"), "n"),
    (("render", "coverage", "depth_tol"), "n"),
    (("render", "coverage", "gap_voxel"), "n"),
    (("render", "coverage", "standoff"), "n"),
    (("render", "coverage", "min_aim_dist"), "n"),
    (("render", "diversity", "min_pos_sep"), "n"),
    (("render", "quality_filter", "min_median_depth"), "n"),
    (("render", "quality_filter", "near_field_depth"), "n"),
    (("lift", "instance_voxel"), "n"),
    (("lift", "scope_margin"), "n"),
)

# The one sentence every "no scale" refusal ends with: the act that
# records one, and the other answer.
RULER_HINT = ("measure one thing you know on the canvas (the ruler at the "
              "volume gate: click two points on something whose real length "
              "you know and type it), or declare the scene metric")


def scale_of(cfg: dict) -> float | None:
    """The recorded factor as a number, or None when none is recorded
    (`null`, absent, or the older `auto` spelling). Refuses, to the volume
    gate, only on a value that is neither: a non-number or a non-positive
    one."""
    raw = cfg["scene"].get("scale_m_per_unit")
    if raw is None or str(raw) in ("auto", "null", "none", "unknown"):
        return None
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        raise Refusal(
            f"scene.scale_m_per_unit={raw!r} is not a number; expected "
            f"metres per scene unit (1.0 = the scene is already metric), or "
            f"null while the scale is not yet recorded.", gate="volume") from None
    if not scale > 0:
        raise Refusal(f"scene.scale_m_per_unit={scale} must be greater than "
                      f"zero.", gate="volume")
    return scale


def measurement_of(cfg: dict) -> dict | None:
    """The ruler's record (`scene.measured`) as a dict, or None when the
    factor was declared or typed. Refuses, to the volume gate, on a record
    that is not one: the panel writes it and the profile is a text file."""
    raw = cfg["scene"].get("measured")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise Refusal(f"scene.measured={raw!r} is not the ruler's record "
                      f"(length_m, units, label, unit); {RULER_HINT}.",
                      gate="volume")
    for key in ("length_m", "units"):
        try:
            v = float(raw.get(key))
        except (TypeError, ValueError):
            raise Refusal(f"scene.measured.{key}={raw.get(key)!r} is not a "
                          f"number; measure again on the canvas.",
                          gate="volume") from None
        if not v > 0:
            raise Refusal(f"scene.measured.{key}={v} must be greater than "
                          f"zero; measure again on the canvas.",
                          gate="volume")
    return dict(raw)


@dataclass
class ScaleRecord:
    """The one factor a run uses, and where it came from."""

    value: float
    source: str                        # "recorded" | "measured"
    recorded: float | None = None      # the profile's number (both sources)
    measured: dict | None = None       # the ruler's record, when measured

    def record(self) -> dict:
        return dict(scale_m_per_unit=float(self.value), source=self.source,
                    recorded=self.recorded, measured=self.measured)

    @classmethod
    def from_record(cls, rec: dict) -> "ScaleRecord":
        return cls(value=float(rec["scale_m_per_unit"]),
                   source=str(rec.get("source", "recorded")),
                   recorded=rec.get("recorded"), measured=rec.get("measured"))

    def measured_as(self) -> str:
        """'a door, 2 m' — what the operator measured, for the panels and
        the manifests; '' when the factor was declared or typed."""
        m = self.measured or {}
        if not m:
            return ""
        label = str(m.get("label") or "").strip()
        length = f"{float(m['length_m']):g} m"
        return f"{label}, {length}" if label else length

    def describe(self) -> str:
        if self.source == "measured":
            m = self.measured or {}
            return (f"scale_m_per_unit {self.value:.4g} (measured: "
                    f"{self.measured_as()} over {float(m.get('units', 0)):.4g} "
                    f"units)")
        return f"scale_m_per_unit {self.value:g} ({self.source})"


def resolve_scale(cfg: dict) -> ScaleRecord:
    """The one factor for a run: the profile's number, recorded or measured.
    Refuses, to the volume gate, when none is recorded — no default factor
    exists anywhere — and when the profile still carries the removed size
    word."""
    if cfg["scene"].get("size_class") is not None:
        raise Refusal(
            f"scene.size_class={cfg['scene'].get('size_class')!r} is no "
            f"longer read: Carveout does not size a scene from a word. "
            f"Delete the key from the profile and {RULER_HINT}.",
            gate="volume")
    recorded = scale_of(cfg)
    measured = measurement_of(cfg)
    if recorded is None:
        raise Refusal(
            f"no scale is recorded for this scene; {RULER_HINT}. Every "
            f"grid, stand-off and voxel is sized in metres, and assuming a "
            f"scale would be guessing the scene.", gate="volume")
    if measured is not None:
        return ScaleRecord(value=recorded, source="measured",
                           recorded=recorded, measured=measured)
    return ScaleRecord(value=recorded, source="recorded", recorded=recorded)


def scale_from_stage1(cfg: dict, manifest: dict | None) -> ScaleRecord:
    """The factor stage 1 ran at, for the stages after it — read, not
    recomputed, so one run uses one number. Refuses (gate `render`) when the
    profile would give a different factor now (through the app a factor
    edit demotes the volume gate and the render is redone; a manifest from
    before the measured source records `estimated` and is stale)."""
    rec = (manifest or {}).get("scale")
    if not rec:
        return resolve_scale(cfg)   # refuses to the volume gate when none
    ran = ScaleRecord.from_record(rec)
    now = resolve_scale(cfg)
    if now.source != ran.source or not math.isclose(now.value, ran.value,
                                                    rel_tol=1e-9):
        raise Refusal(
            f"stage 1 was rendered at {ran.describe()}; the profile now gives "
            f"{now.describe()}; re-render stage 1 so every stage uses one "
            f"scale.", gate="render")
    return ran


def scene_units_cfg(cfg: dict, scale: float) -> tuple[dict, dict]:
    """A copy of `cfg` with every LENGTH_KEYS value divided by `scale`, and
    a record {scale_m_per_unit, converted: {path: {m, units}}} for the
    manifest. The caller's cfg is not touched: the profile stays in metres
    and a cached config object never accumulates one scene's conversion."""
    out = copy.deepcopy(cfg)
    converted: dict = {}
    for path, kind in LENGTH_KEYS:
        node = out
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict) or path[-1] not in node:
            continue
        val = node[path[-1]]
        if val is None or isinstance(val, str):
            continue
        if kind == "n":
            new = val / scale
        else:
            new = [x / scale for x in val]
        node[path[-1]] = new
        converted[".".join(path)] = dict(m=val, units=new)
    return out, dict(scale_m_per_unit=scale, converted=converted)
