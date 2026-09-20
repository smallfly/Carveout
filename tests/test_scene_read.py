# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seconds-fast unit test for `carveout.scene_read` (pure logic):
the ground map, the analysis, the checks of the operator's two answers and
the settings proposal on synthetic inputs — no models, no scenes.

    python -m tests.test_scene_read
"""

import numpy as np

from carveout.cameras import detect_scene_frame
from carveout.config import load_config
from carveout.scene_read import (analyse, check_answers, ground_map,
                                 propose_settings)
from carveout.units import scene_units_cfg


def _cfg(scale=1.0, measured=None, path_mode="interior"):
    cfg = load_config()
    cfg["scene"]["scale_m_per_unit"] = scale
    cfg["scene"]["measured"] = measured
    cfg["scene"]["up_axis"] = "+y"
    cfg["render"]["path_mode"] = path_mode
    return cfg


def _room(w=7.0, d=8.0, h=2.7, n=60000, seed=0):
    """A hollow room: opaque points on the four walls, the floor and the
    ceiling, nothing inside."""
    rng = np.random.default_rng(seed)
    pts = []
    per = n // 6
    for face in range(6):
        u, v = rng.uniform(0, 1, (2, per))
        if face == 0:   pts.append(np.c_[u * w, np.zeros(per), v * d])
        elif face == 1: pts.append(np.c_[u * w, np.full(per, h), v * d])
        elif face == 2: pts.append(np.c_[np.zeros(per), u * h, v * d])
        elif face == 3: pts.append(np.c_[np.full(per, w), u * h, v * d])
        elif face == 4: pts.append(np.c_[u * w, v * h, np.zeros(per)])
        else:           pts.append(np.c_[u * w, v * h, np.full(per, d)])
    means = np.concatenate(pts).astype(np.float32)
    return means, np.full(len(means), 0.9, dtype=np.float32)


def _site(w=40.0, d=30.0, slope=0.1, n=300000, seed=1):
    """An open sloped ground with a few pillars (stones) on it."""
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0, w, n), rng.uniform(0, d, n)
    y = slope * x + rng.normal(0, 0.02, n)
    pts = [np.c_[x, y, z]]
    for cx, cz in [(8, 8), (20, 15), (32, 22), (12, 24)]:
        k = 3000
        px, pz = rng.uniform(cx - 0.4, cx + 0.4, k), rng.uniform(cz - 0.4, cz + 0.4, k)
        pts.append(np.c_[px, slope * px + rng.uniform(0, 1.5, k), pz])
    means = np.concatenate(pts).astype(np.float32)
    return means, np.full(len(means), 0.9, dtype=np.float32)


def _frame_and_cfg(means, ops, **kw):
    cfg_m = _cfg(**kw)
    frame = detect_scene_frame(means, ops, cfg_m)
    cfg, _ = scene_units_cfg(cfg_m, frame.scale)
    return frame, cfg, cfg_m


def _said(cfg_m, mode):
    out = dict(cfg_m)
    out["render"] = dict(cfg_m["render"], path_mode=mode)
    return out


def test_ground_map_slope_and_footing():
    means, ops = _site()
    frame, cfg, _ = _frame_and_cfg(means, ops)
    gm = ground_map(means, ops, frame, cfg, None)
    lo, hi = gm.relief_p5_p95
    assert 2.5 < hi - lo < 4.5, gm.relief_p5_p95   # 0.1 x 40 = 4 of slope
    assert gm.standable_frac > 0.8 and gm.populated_frac > 0.8
    assert gm.mass.max() > 0 and gm.dims == (79, 59)   # robust bounds
    # a coarser cell fits a smaller budget, same relief
    gm2 = ground_map(means, ops, frame, cfg, None, cell=2.0, budget=False)
    assert gm2.dims == (20, 15)
    assert abs((gm2.relief_p5_p95[1] - gm2.relief_p5_p95[0]) - (hi - lo)) < 0.6
    # the volume's footprint scopes footing
    box = [dict(name="b", min=np.array([0.0, -1, 0.0]), max=np.array([10.0, 5, 10.0]))]
    gm3 = ground_map(means, ops, frame, cfg, box)
    assert gm3.dims == (21, 21) and gm3.standable.sum() < gm.standable.sum()


def test_analyse():
    means, ops = _room()
    frame, cfg, cfg_m = _frame_and_cfg(means, ops)
    a = analyse(means, ops, frame, cfg, None, "enclosed")
    assert a["up_axis"] == 1 and abs(a["span_units"] - 2.7) < 0.1
    assert a["volume_footprint_frac"] == 1.0
    assert a["enclosure"] == dict(used="enclosed", reason=None)
    assert "hollow" not in a and a["scale"]["source"] == "recorded"
    box = [dict(name="b", min=[1.0, 0.0, 1.0], max=[1.8, 1.0, 1.9])]
    assert analyse(means, ops, frame, cfg, box, "enclosed")["volume_footprint_frac"] < 0.25


def test_check_answers_room():
    means, ops = _room()                       # 7 x 8 x 2.7 m, enclosed
    frame, cfg, cfg_m = _frame_and_cfg(means, ops)
    a = analyse(means, ops, frame, cfg, None, "enclosed")
    c = check_answers(a, _said(cfg_m, "interior"))
    assert c["path_mode"] == dict(said="interior", fits=True,
                                  reason="an enclosed region was found — walls around a floor")
    assert c["scale"]["fits"] and "2.7 m tall" in c["scale"]["reason"]
    assert c["opinion"] is None
    # said outdoors on the room -> flagged (an enclosed region was found)
    c = check_answers(a, _said(cfg_m, "ground"))
    assert c["path_mode"]["fits"] is False and "inside a space" in c["path_mode"]["reason"]
    assert c["scale"] is None                  # inside only
    # said around a subject on a 7 m room -> flagged
    c = check_answers(a, _said(cfg_m, "orbit"))
    assert c["path_mode"]["fits"] is False and "big enough to walk" in c["path_mode"]["reason"]
    # no answer (a hand-written profile) -> flagged, naming the render panel
    c = check_answers(a, _said(cfg_m, "auto"))
    assert c["path_mode"]["fits"] is False and "render panel" in c["path_mode"]["reason"]
    assert check_answers(a, _said(cfg_m, "manual"))["path_mode"]["fits"] is True
    # a 0.4 m room (a wrong measurement): the scale line
    a_wrong = dict(a, scale=dict(a["scale"], scale=0.15, source="measured",
                                 measured=dict(length_m=1.0, units=6.7)))
    c = check_answers(a_wrong, _said(cfg_m, "interior"))
    assert c["scale"]["fits"] is False
    assert c["scale"]["reason"] == ("at this measurement the room reads 0.405 m tall: "
                                    "check the two points, or the filming answer if "
                                    "this is not an interior")
    # the same under a typed factor names the factor
    a_wrong2 = dict(a, scale=dict(a["scale"], scale=0.15, source="recorded"))
    assert "check the factor" in check_answers(a_wrong2, _said(cfg_m, "interior"))["scale"]["reason"]


def test_check_answers_site_and_nook():
    means, ops = _site()                       # 40 x 30 m, sloped, open
    frame, cfg, cfg_m = _frame_and_cfg(means, ops)
    a = analyse(means, ops, frame, cfg, None, "density_core", "nothing_enclosed")
    assert a["footprint_over_height"] > 2
    c = check_answers(a, _said(cfg_m, "interior"))
    assert c["path_mode"]["fits"] is False and "nothing enclosed" in c["path_mode"]["reason"]
    c = check_answers(a, _said(cfg_m, "ground"))
    assert c["path_mode"]["fits"] is True, c
    # said around a subject on the open site, never scanned -> flagged
    a_ns = dict(a, enclosure=dict(used="density_core", reason="not_scanned"))
    c = check_answers(a_ns, _said(cfg_m, "orbit"))
    assert c["path_mode"]["fits"] is False and "outdoors on the ground" in c["path_mode"]["reason"]
    # little standable ground -> outdoors flagged
    a5 = dict(a, ground=dict(a["ground"], standable_frac=0.1))
    c = check_answers(a5, _said(cfg_m, "ground"))
    assert c["path_mode"]["fits"] is False and "little standable" in c["path_mode"]["reason"]
    # the enclosure scan over budget under inside: unknown, not flagged
    a_ob = dict(a, enclosure=dict(used="density_core", reason="over_budget"))
    assert check_answers(a_ob, _said(cfg_m, "interior"))["path_mode"]["fits"] is None

    means, ops = _room(w=2.0, d=2.4, h=2.6)   # a nook, ringed
    frame, cfg, cfg_m = _frame_and_cfg(means, ops)
    a = analyse(means, ops, frame, cfg, None, "density_core", "not_scanned")
    c = check_answers(a, _said(cfg_m, "orbit"))
    assert c["path_mode"]["fits"] is True, c   # not flagged: 2.2 m across
    # the same nook said inside: too small to walk (under 3 m) -> flagged
    a_in = analyse(means, ops, frame, cfg, None, "enclosed")
    c = check_answers(a_in, _said(cfg_m, "interior"))
    assert c["path_mode"]["fits"] is False and "too small to walk" in c["path_mode"]["reason"]
    assert "under 3 m" in c["path_mode"]["reason"]


def test_second_opinion():
    means, ops = _room()
    frame, cfg, cfg_m = _frame_and_cfg(means, ops, scale=0.5, measured=dict(
        length_m=2.0, units=4.0, label="a door", unit="m"))
    a = analyse(means, ops, frame, cfg, None, "enclosed")
    assert a["scale"]["source"] == "measured"
    ok = dict(length_m=1.9, what="the height of a door", reason="x")
    c = check_answers(a, _said(cfg_m, "interior"), ok)
    assert c["opinion"]["fits"] is True
    assert c["opinion"]["reason"] == ("the model reads A–B as about 1.9 m (the height "
                                      "of a door), which agrees with the 2 m you typed")
    c = check_answers(a, _said(cfg_m, "interior"), dict(length_m=0.9, what="", reason=""))
    assert c["opinion"]["fits"] is False
    assert c["opinion"]["reason"] == ("the model reads A–B as about 0.9 m, but you "
                                      "typed 2 m: check the two points")
    assert check_answers(a, cfg_m, dict(length_m=4.0))["opinion"]["fits"] is True   # 2x exactly
    assert check_answers(a, cfg_m, dict(length_m=4.1))["opinion"]["fits"] is False
    c = check_answers(a, cfg_m, dict(error="weights missing at models/x"))
    assert c["opinion"] == dict(fits=None, reason="weights missing at models/x")


def test_propose_settings():
    means, ops = _room()
    frame, cfg, cfg_m = _frame_and_cfg(means, ops)
    a = analyse(means, ops, frame, cfg, None, "enclosed")
    s = propose_settings(a, _said(cfg_m, "interior"))
    assert s["settings"] == dict(cameras_in_volume=True, focus_aim=False)
    assert s["thresholds"] == {} and s["reasons"] == []       # 7 m: not small
    # a bench-top volume inside the room -> tight focus (inside only)
    box = [dict(name="b", min=[1.0, 0.0, 1.0], max=[1.8, 1.0, 1.9])]
    a2 = analyse(means, ops, frame, cfg, box, "enclosed")
    assert propose_settings(a2, _said(cfg_m, "interior"))["settings"] == dict(
        cameras_in_volume=False, focus_aim=True)
    assert propose_settings(a2, _said(cfg_m, "orbit"))["settings"]["focus_aim"] is False

    means, ops = _room(w=2.0, d=2.4, h=2.6)   # a nook
    frame, cfg, cfg_m = _frame_and_cfg(means, ops)
    a = analyse(means, ops, frame, cfg, None, "density_core", "not_scanned")
    th = propose_settings(a, _said(cfg_m, "orbit"))["thresholds"]
    assert set(th) == {"min_median_depth", "near_field_depth"}, th
    assert abs(th["min_median_depth"]["proposed"] - 0.163 * 2.0) < 0.02
    assert all(v["proposed"] < v["current"] for v in th.values())
    # never a larger threshold: profile already below -> nothing proposed
    cfg_m["render"]["quality_filter"]["min_median_depth"] = 0.2
    cfg_m["render"]["quality_filter"]["near_field_depth"] = 0.3
    assert propose_settings(a, cfg_m)["thresholds"] == {}
    # "auto" already resolves itself -> not proposed
    cfg_m["render"]["quality_filter"]["min_median_depth"] = "auto"
    cfg_m["render"]["quality_filter"]["near_field_depth"] = 1.2
    assert set(propose_settings(a, cfg_m)["thresholds"]) == {"near_field_depth"}
    # after a measurement the number is metres: units x factor
    a_m = dict(a, scale=dict(scale=0.5, source="measured"))
    th = propose_settings(a_m, cfg_m)["thresholds"]["near_field_depth"]
    assert abs(th["proposed"] - th["proposed_units"] * 0.5) < 1e-6


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print(f"all passed in {time.perf_counter() - t0:.2f} s")
