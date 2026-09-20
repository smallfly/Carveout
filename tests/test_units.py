# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seconds-fast unit test for `carveout.units` (pure logic, no models, no
scenes): one
factor per run, recorded or measured with the ruler — never guessed; the
division; the stage-1 record; the refusals, which name the ruler.

    python -m tests.test_units
"""

import numpy as np

from carveout.cameras import scale_read
from carveout.config import load_config
from carveout.refusal import Refusal
from carveout.units import (LENGTH_KEYS, ScaleRecord, resolve_scale,
                            scale_from_stage1, scale_of, scene_units_cfg)

MEASURED = dict(length_m=1.8, units=1.964, label="the bench edge", unit="m",
                points=[[0, 0, 0], [1.964, 0, 0]])


def _cfg(scale, measured=None):
    cfg = load_config()
    cfg["scene"]["scale_m_per_unit"] = scale
    cfg["scene"]["measured"] = measured
    return cfg


def _refuses(fn, gate, *needles):
    try:
        fn()
    except Refusal as e:
        assert e.gate == gate, (e.gate, str(e))
        for n in needles:
            assert n in str(e), (n, str(e))
    else:
        raise AssertionError("expected a Refusal")


def test_scale_of():
    assert scale_of(_cfg(1.0)) == 1.0
    assert scale_of(_cfg(0.01)) == 0.01
    assert scale_of(_cfg(None)) is None
    assert scale_of(_cfg("auto")) is None
    for bad in ("metres", 0, -2):
        _refuses(lambda: scale_of(_cfg(bad)), "volume")


def test_resolve_scale():
    r = resolve_scale(_cfg(0.01))
    assert (r.value, r.source, r.recorded, r.measured) == (0.01, "recorded", 0.01, None)
    assert "(recorded)" in r.describe()
    # the ruler: the factor is the profile's number, the act beside it
    factor = 1.8 / 1.964
    r = resolve_scale(_cfg(factor, MEASURED))
    assert r.source == "measured" and abs(r.value - factor) < 1e-12
    assert r.measured == MEASURED and r.measured_as() == "the bench edge, 1.8 m"
    assert "measured: the bench edge, 1.8 m over 1.964 units" in r.describe()
    assert ScaleRecord.from_record(r.record()).measured == MEASURED
    # no factor, nothing measured -> the refusal names the ruler
    _refuses(lambda: resolve_scale(_cfg(None)), "volume", "no scale is recorded",
             "ruler", "two points", "metric")
    # the removed size word refuses loudly, naming the ruler
    cfg = _cfg(None)
    cfg["scene"]["size_class"] = "room"
    _refuses(lambda: resolve_scale(cfg), "volume", "size_class", "no longer read",
             "ruler")
    cfg = _cfg(1.0)
    cfg["scene"]["size_class"] = "site"
    _refuses(lambda: resolve_scale(cfg), "volume", "size_class")
    # a broken record
    _refuses(lambda: resolve_scale(_cfg(1.0, "a door")), "volume", "ruler")
    _refuses(lambda: resolve_scale(_cfg(1.0, dict(length_m=0, units=2))),
             "volume", "length_m", "measure again")
    _refuses(lambda: resolve_scale(_cfg(1.0, dict(length_m=2, units="x"))),
             "volume", "units")


def test_scene_units_cfg():
    cfg = _cfg(0.01)
    out, rec = scene_units_cfg(cfg, 0.01)
    assert out["render"]["interior"]["eye_height"] == 1.6 / 0.01
    assert out["render"]["interior"]["focus_standoff"] == [0.5 / 0.01, 2.5 / 0.01]
    assert out["lift"]["instance_voxel"] == 0.15 / 0.01
    assert out["render"]["interior"]["focus_heights"] is None, "null stays null"
    assert cfg["render"]["interior"]["eye_height"] == 1.6, "caller's cfg untouched"
    assert rec["scale_m_per_unit"] == 0.01
    assert rec["converted"]["render.interior.eye_height"] == dict(m=1.6, units=160.0)
    cfg["render"]["quality_filter"]["min_median_depth"] = "auto"
    out, _ = scene_units_cfg(cfg, 0.01)
    assert out["render"]["quality_filter"]["min_median_depth"] == "auto"
    at_one, _ = scene_units_cfg(_cfg(1.0), 1.0)
    for path, kind in LENGTH_KEYS:
        a, b = at_one, _cfg(1.0)
        for k in path:
            a, b = a[k], b[k]
        assert a == b, path     # exact at 1.0


def test_scale_from_stage1():
    factor = 1.8 / 1.964
    cfg = _cfg(factor, MEASURED)
    ran = resolve_scale(cfg)
    m = dict(scale=ran.record())
    assert scale_from_stage1(cfg, m).value == ran.value
    # the factor moved (typed over the measurement) -> re-render
    _refuses(lambda: scale_from_stage1(_cfg(1.0), m), "render", "re-render")
    # the same number declared instead of measured is a different source
    _refuses(lambda: scale_from_stage1(_cfg(factor), m), "render", "re-render")
    # a manifest from before the measured source (the old estimate) is stale
    old = dict(scale=dict(scale_m_per_unit=factor, source="estimated",
                          size_class="room", span_units=2.7))
    _refuses(lambda: scale_from_stage1(cfg, old), "render", "estimated")
    assert scale_from_stage1(_cfg(1.0), None).value == 1.0
    assert scale_from_stage1(_cfg(0.5), dict(scale_m_per_unit=0.5)).recorded == 0.5
    _refuses(lambda: scale_from_stage1(_cfg(None), None), "volume", "ruler")


def test_scale_read():
    ext = np.array([4.0, 3.0, 6.0])
    factor = 1.8 / 1.964
    cfg = _cfg(factor, MEASURED)
    cfg["render"]["path_mode"] = "interior"
    rec = resolve_scale(cfg)
    read = scale_read(ext, 1, cfg, rec)
    assert read["source"] == "measured" and read["measured"] == MEASURED
    assert read["scale"] == rec.value == read["scale_m_per_unit"]
    assert abs(read["height_m"] - 3.0 * factor) < 1e-3 and read["plausible"]
    assert read["note"].startswith("1 unit ≈ 0.916 m (measured: the bench edge, 1.8 m)")
    assert ScaleRecord.from_record(read).value == rec.value
    # a wrong measurement on an interior: the band says so, naming the points
    read = scale_read(np.array([4.0, 0.5, 6.0]), 1, cfg, rec)
    assert read["plausible"] is False and "check the two points" in read["note"]
    # the same read outdoors: no ceiling to judge by
    cfg["render"]["path_mode"] = "ground"
    read = scale_read(np.array([4.0, 0.5, 6.0]), 1, cfg, rec)
    assert read["plausible"] is None and "check" not in read["note"]
    cfg = _cfg(1.0)
    cfg["render"]["path_mode"] = "interior"
    read = scale_read(ext, 1, cfg, resolve_scale(cfg))
    assert read["source"] == "recorded" and read["height_m"] == 3.0 and read["plausible"]
    assert read["note"] == "1 unit = 1 m (recorded)"
    for word in ("estimated", "span", "anchor", "size class", "factor"):
        assert word not in read["note"], word


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print(f"all passed in {time.perf_counter() - t0:.2f} s")
