# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""3DGS `.ply` reading/writing.

One of the two containers behind `scene_io.load_scene`; the GaussianScene it
fills lives there. Writing is ply-only and stays here: per-object exports
are meant to open in SuperSplat.

Standard 3DGS layout: positions x,y,z; f_dc_0..2 (SH DC per channel);
f_rest_{i} with i = channel*K + coeff (channel-major; K = 0, 3, 8 or 15
for SH degree 0 to 3 — the count present decides the degree); opacity
(pre-sigmoid); scale_0..2 (pre-exp, log-scale); rot_0..3 (quaternion, w-first,
unnormalized).
"""

import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .scene_io import GaussianScene

log = logging.getLogger(__name__)


def load_gaussian_ply(path: str | Path) -> GaussianScene:
    path = Path(path)
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = set(v.dtype.names)

    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = required - names
    if missing:
        raise ValueError(
            f"{path} is not a supported 3DGS ply; missing fields: {sorted(missing)}"
        )

    n = len(v)
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1)).astype(np.float32)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    opacities = (1.0 / (1.0 + np.exp(-v["opacity"]))).astype(np.float32)

    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)  # (N,3)
    rest_names = sorted(
        (nm for nm in names if nm.startswith("f_rest_")),
        key=lambda nm: int(nm.split("_")[-1]),
    )
    if rest_names:
        n_coeffs = len(rest_names) // 3
        rest = np.stack([v[nm] for nm in rest_names], axis=1).astype(np.float32)
        # stored channel-major (3, n_coeffs) flattened -> (N, n_coeffs, 3)
        rest = rest.reshape(n, 3, n_coeffs).transpose(0, 2, 1)
    else:
        rest = np.zeros((n, 0, 3), dtype=np.float32)
    sh = np.concatenate([dc[:, None, :], rest], axis=1)

    log.info(
        "loaded %s: %d gaussians, SH degree %d",
        path.name, n, int(np.sqrt(sh.shape[1])) - 1,
    )
    return GaussianScene(
        means=means, scales=scales, quats=quats, opacities=opacities, sh=sh,
        source_path=str(path),
    )


def write_gaussian_ply(scene: GaussianScene, path: str | Path,
                    mask: np.ndarray | None = None) -> None:
    """Write (a subset of) a scene back to the 3DGS layout (inverse activations)."""
    idx = np.flatnonzero(mask) if mask is not None else np.arange(scene.num_gaussians)
    n = len(idx)
    k = scene.sh.shape[1]

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    fields += [(f"f_dc_{i}", "f4") for i in range(3)]
    fields += [(f"f_rest_{i}", "f4") for i in range(3 * (k - 1))]
    fields += [("opacity", "f4")]
    fields += [(f"scale_{i}", "f4") for i in range(3)]
    fields += [(f"rot_{i}", "f4") for i in range(4)]

    out = np.zeros(n, dtype=fields)
    m = scene.means_file[idx]          # the file's own frame
    out["x"], out["y"], out["z"] = m[:, 0], m[:, 1], m[:, 2]
    for i in range(3):
        out[f"f_dc_{i}"] = scene.sh[idx, 0, i]
    rest = scene.sh[idx, 1:, :].transpose(0, 2, 1).reshape(n, -1)  # channel-major
    for i in range(rest.shape[1]):
        out[f"f_rest_{i}"] = rest[:, i]
    op = np.clip(scene.opacities[idx], 1e-6, 1 - 1e-6)
    out["opacity"] = np.log(op / (1 - op))
    logs = np.log(np.maximum(scene.scales[idx], 1e-12))
    for i in range(3):
        out[f"scale_{i}"] = logs[:, i]
    for i in range(4):
        out[f"rot_{i}"] = scene.quats_file[idx][:, i]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyElement.describe(out, "vertex")
    PlyData([PlyElement.describe(out, "vertex")]).write(str(path))
    log.info("wrote %d gaussians -> %s", n, path)
