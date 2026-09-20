# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seconds-fast unit test for `carveout.projection.project_visible` (pure
geometry, no models, no scenes).

    python -m tests.test_projection

The one property that matters: the test is scale-free. Scale the points,
the camera position, the depth map and every length argument by the same
factor and the same points are visible — the four copies this helper
replaced each had a `z > 0.05` and a `* 1.05 + 0.2` that made the answer
depend on the scene's unit (defects A01, A02).
"""

import numpy as np

from carveout.projection import project_visible


def _camera(pos, w=64, h=48, f=40.0):
    """A camera at `pos` looking down +z with the identity rotation."""
    c2w = np.eye(4)
    c2w[:3, 3] = pos
    fr = dict(frame_idx=0, c2w=c2w.tolist())
    cams = dict(fx=f, fy=f, cx=w / 2, cy=h / 2, width=w, height=h)
    return fr, cams


def test_near_plane_and_image_bounds():
    fr, cams = _camera([0, 0, 0])
    pts = np.array([[0, 0, 1.0],       # centre pixel, in front
                    [0, 0, 0.01],      # behind the near plane
                    [0, 0, -1.0],      # behind the camera
                    [10, 0, 1.0],      # far outside the image
                    [0, 0, 0.0]],      # on the camera: no division blow-up
                   dtype=np.float32)
    proj = project_visible(pts, fr, cams, None, near=0.05)
    assert proj.ok.tolist() == [True, False, False, False, False]
    assert (proj.u[0], proj.v[0]) == (32, 24)
    assert proj.vis is proj.ok, "no depth -> no occlusion test"


def test_depth_occlusion_with_slack():
    fr, cams = _camera([0, 0, 0])
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    pts = np.array([[0, 0, 1.0],      # in front of the surface
                    [0, 0, 2.0],      # on it
                    [0, 0, 2.29],     # behind, inside 2.0 * 1.05 + 0.2 = 2.3
                    [0, 0, 2.31]],    # behind, outside the slack
                   dtype=np.float32)
    proj = project_visible(pts, fr, cams, depth, near=0.05,
                           tol_rel=0.05, tol_abs=0.2)
    assert proj.ok.tolist() == [True] * 4
    assert proj.vis.tolist() == [True, True, True, False]


def test_min_points_and_lazy_depth():
    fr, cams = _camera([0, 0, 0])
    pts = np.array([[0, 0, 1.0], [0, 0, -1.0]], dtype=np.float32)
    loads = []

    def depth():
        loads.append(1)
        return np.full((48, 64), 2.0, dtype=np.float32)

    assert project_visible(pts, fr, cams, depth, near=0.05, min_points=2) is None
    assert not loads, "depth is not read when too few points land"
    proj = project_visible(pts, fr, cams, depth, near=0.05, min_points=1)
    assert proj is not None and loads == [1]
    # 2 in front but only 1 visible: the depth test also honours min_points
    pts = np.array([[0, 0, 1.0], [0, 0, 5.0]], dtype=np.float32)
    assert project_visible(pts, fr, cams, depth, near=0.05, min_points=2) is None


def test_scale_free():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(2000, 3)).astype(np.float32) * [1, 1, 0.3] + [0, 0, 1.5]
    fr, cams = _camera([0.1, -0.2, -0.5])
    depth = rng.uniform(1.0, 2.5, size=(48, 64)).astype(np.float32)
    ref = project_visible(pts, fr, cams, depth, near=0.05,
                          tol_rel=0.05, tol_abs=0.2)
    assert 0 < ref.n_vis < ref.n_ok < len(pts), "the test must exercise every branch"
    for k in (0.001, 100.0, 1e5):
        frk, _ = _camera(np.array([0.1, -0.2, -0.5]) * k)
        got = project_visible((pts * k).astype(np.float32), frk, cams,
                              depth * k, near=0.05 * k,
                              tol_rel=0.05, tol_abs=0.2 * k)
        # float32 scaling moves a handful of points across pixel edges
        assert np.mean(got.ok == ref.ok) > 0.995, k
        assert np.mean(got.vis == ref.vis) > 0.995, k


def test_torch_matches_numpy():
    try:
        import torch
    except ImportError:   # the env always has it; the test stays honest
        return
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(500, 3)).astype(np.float32) * [1, 1, 0.3] + [0, 0, 1.5]
    fr, cams = _camera([0.1, -0.2, -0.5])
    depth = rng.uniform(1.0, 2.5, size=(48, 64)).astype(np.float32)
    a = project_visible(pts, fr, cams, depth, near=0.05, tol_rel=0.05, tol_abs=0.2)
    b = project_visible(torch.as_tensor(pts), fr, cams, torch.as_tensor(depth),
                        near=0.05, tol_rel=0.05, tol_abs=0.2)
    assert np.array_equal(a.ok, b.ok.numpy()) and np.array_equal(a.vis, b.vis.numpy())
    assert np.array_equal(a.u[a.ok], b.u.numpy()[a.ok])
    assert np.array_equal(a.v[a.ok], b.v.numpy()[a.ok])


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print(f"all passed in {time.perf_counter() - t0:.2f} s")
