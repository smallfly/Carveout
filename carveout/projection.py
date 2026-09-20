# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""One projection for every stage.

Four copies of the same fifteen lines used to live in the render (coverage
counts), the lift (connectivity visibility), the export (connectivity
association) and verify (projection crops), each with its own near plane
(`z > 0.05`) and three of them with a depth slack of `* 1.05 + 0.2` — raw
scene-unit literals that the scale factor never reached, so a scene
100x off scale tested visibility against a 2 mm slack and a 0.5 mm near
plane. They are one function now, and the near plane and the slack are
arguments the caller takes from the RESOLVED config (`render.near_plane`,
`render.coverage.depth_tol` / `depth_tol_rel`), which follow the scene's
units like every other length.

Works on numpy arrays and torch tensors alike — the render keeps its points
on the GPU, the other stages are numpy on the CPU — with the same
arithmetic and the same truncation to pixel indices as the copies it
replaces, so no caller's numbers move.
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .manual_views import frame_intrinsics


@dataclass
class Projection:
    """Per-point projection of one point set into one frame. Every field is
    an array over ALL the input points (numpy or torch, matching `pts`);
    the pixel fields are only meaningful where `ok` holds."""

    u: Any        # pixel column, truncated toward zero (index into depth/masks)
    v: Any        # pixel row, truncated
    uf: Any       # pixel column, float (sub-pixel bounds for crops)
    vf: Any       # pixel row, float
    z: Any        # camera-space depth, scene units
    ok: Any       # in front of the near plane AND inside the image
    vis: Any      # `ok` and not occluded by the rendered depth
                  # (identical to `ok` when no depth was given)

    @property
    def n_ok(self) -> int:
        return int(self.ok.sum())

    @property
    def n_vis(self) -> int:
        return int(self.vis.sum())


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def project_visible(pts, fr: dict, cams: dict,
                    depth: Any | Callable[[], Any] | None = None, *,
                    near: float, tol_rel: float = 0.0, tol_abs: float = 0.0,
                    min_points: int = 0) -> Projection | None:
    """Project `pts` (N, 3 — scene units, as the .ply) into cameras.json
    frame `fr` and depth-test them against that frame's rendered depth.

    `near`: the near plane in scene units — the resolved `render.near_plane`.
    `tol_rel`, `tol_abs`: the visibility slack, `z <= depth * (1 + tol_rel)
    + tol_abs` — the resolved `render.coverage.depth_tol_rel` / `depth_tol`.
    `depth`: the frame's (H, W) rendered depth, a zero-argument callable
    returning it (read only once enough points land in the image — the
    export loads one file per instance and frame), or None for NO occlusion
    test: verify's projection crops are the one caller that wants every
    point in front of the camera, occluded or not, and say so by passing
    None.
    `min_points`: return None when fewer points than this pass either test
    (the callers' "fewer than 50 land" early exits).
    """
    fx, fy, cx, cy, w, h = frame_intrinsics(fr, cams)
    if _is_torch(pts):
        import torch
        w2c = torch.linalg.inv(torch.as_tensor(
            np.array(fr["c2w"]), dtype=pts.dtype, device=pts.device))
        p = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2]
        zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        uf = fx * p[:, 0] / zs + cx
        vf = fy * p[:, 1] / zs + cy
        u, v = uf.long(), vf.long()
    else:
        w2c = np.linalg.inv(np.array(fr["c2w"]))
        p = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        uf = fx * p[:, 0] / zs + cx
        vf = fy * p[:, 1] / zs + cy
        u, v = uf.astype(int), vf.astype(int)
    ok = (z > near) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if int(ok.sum()) < min_points:
        return None
    if depth is None:
        vis = ok
    else:
        if callable(depth):
            depth = depth()
        vis = ok.clone() if _is_torch(ok) else ok.copy()
        vis[ok] = z[ok] <= depth[v[ok], u[ok]] * (1 + tol_rel) + tol_abs
        if int(vis.sum()) < min_points:
            return None
    return Projection(u=u, v=v, uf=uf, vf=vf, z=z, ok=ok, vis=vis)
