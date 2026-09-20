# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scene type every stage works on, and the reader that picks its codec.

Carveout reads a trained 3DGS scene out of one of two containers: the
standard 3DGS `.ply` (`ply_io`) and a PlayCanvas SOG v2 bundle, `.sog`
(`sog_io`). Nothing downstream of `load_scene` learns which one it came
from — GaussianScene is the boundary.

Writing stays ply-only (`ply_io.write_gaussian_ply`): the per-object exports
exist to be opened in SuperSplat, and SOG's quantisation is a shipping
format, not a working one.

The two containers do NOT share a Gaussian ordering — a `.sog` is written
spatially sorted, so index i of a scene's `.ply` and index i of the `.sog`
written from it are different Gaussians. Everything Carveout indexes
per-Gaussian (stage3/instances.npy, the viewer's instance_ids.bin) is
therefore bound to the file it was lifted from. Swapping the file in a
scene profile changes the profile's content hash, which demotes every gate
and forces the re-render — the correct outcome, and not a coincidence.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .refusal import Refusal

log = logging.getLogger(__name__)

SPLAT_SUFFIXES = (".ply", ".sog")


@dataclass
class GaussianScene:
    """Raw (activated) Gaussian parameters, index-aligned to the source file."""

    means: np.ndarray       # (N, 3) float32
    scales: np.ndarray      # (N, 3) float32, exp-activated (world units)
    quats: np.ndarray       # (N, 4) float32, wxyz, normalized
    opacities: np.ndarray   # (N,)  float32, sigmoid-activated [0, 1]
    sh: np.ndarray          # (N, K, 3) float32, K = (deg+1)^2, DC at index 0
    source_path: str = ""
    # `means`/`quats` are in the SCENE frame (the file rotated by the
    # profile's alignment) — every geometric decision reads them;
    # `means_file`/`quats_file` are the file's own, what renders and what
    # the per-object .ply cuts are written from. With no alignment they
    # are the SAME arrays (no copy, no cost). `alignment` is R or None.
    means_file: np.ndarray = None
    quats_file: np.ndarray = None
    alignment: np.ndarray = None

    def __post_init__(self):
        if self.means_file is None:
            self.means_file = self.means
        if self.quats_file is None:
            self.quats_file = self.quats

    @property
    def num_gaussians(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return int(np.sqrt(self.sh.shape[1])) - 1


def resolve_scene_path(path: str | Path) -> Path:
    """Accept either a scene file or the directory holding exactly one.

    A directory carrying two splat files (a `.ply` and the `.sog` written
    from it — the usual state while converting) is REFUSED, not guessed:
    the two carry different Gaussian orderings, so picking one silently
    would decide, behind the operator's back, which file a run's
    per-Gaussian ids belong to.
    """
    p = Path(path)
    if not p.is_dir():
        return p
    found = sorted(f for f in p.iterdir()
                   if f.suffix.lower() in SPLAT_SUFFIXES)
    if not found:
        raise FileNotFoundError(
            f"{p} holds no splat scene; expected one file ending in "
            f"{' or '.join(SPLAT_SUFFIXES)}")
    if len(found) > 1:
        raise ValueError(
            f"{p} holds {len(found)} splat files "
            f"({', '.join(f.name for f in found)}); they are different "
            f"Gaussian orderings, so name the one to use (pick the file, "
            f"not the folder, in the New scene dialog), e.g. {found[0]}")
    return found[0]


def scene_alignment(cfg: dict | None) -> np.ndarray | None:
    """The profile's alignment as R (scene = R · file), None for identity.
    Needs a pinned up axis: the angles are about it."""
    from .alignment import alignment_matrix, is_identity, validate_alignment
    if not cfg:
        return None
    sc = cfg.get("scene") or {}
    try:
        block = validate_alignment(sc.get("alignment"))
    except ValueError as e:
        raise Refusal(f"the scene profile's alignment is malformed: {e}",
                      gate="volume") from None
    if block is None or is_identity(block):
        return None
    up = str(sc.get("up_axis", "-y"))
    if len(up) != 2 or up[1] not in "xyz" or up[0] not in "+-":
        raise Refusal("the scene is levelled, but its up axis is not pinned "
                      "(scene.up_axis is 'auto'); pin the up axis at the "
                      "volume gate, then level", gate="volume")
    return alignment_matrix(block, "xyz".index(up[1]))


def apply_alignment(scene: GaussianScene, cfg: dict | None) -> GaussianScene:
    """Rotate a freshly loaded scene into the scene frame. No-op —
    the same arrays — when the profile holds no alignment."""
    from .alignment import quat_of_matrix, rotate_quats
    R = scene_alignment(cfg)
    if R is None:
        return scene
    scene.means_file, scene.quats_file = scene.means, scene.quats
    scene.means = (scene.means @ R.T.astype(scene.means.dtype))
    scene.quats = rotate_quats(quat_of_matrix(R).astype(scene.quats.dtype),
                               scene.quats)
    scene.alignment = R
    log.info("scene levelled by the profile's alignment (%s)",
             cfg["scene"]["alignment"])
    return scene


def load_scene(path: str | Path, cfg: dict | None = None) -> GaussianScene:
    """Read a scene from whichever supported container it is stored in.
    With `cfg`, the profile's alignment puts `means`/`quats` in the scene
    frame (`means_file`/`quats_file` stay the file's)."""
    p = resolve_scene_path(path)
    if not p.exists():
        raise FileNotFoundError(f"scene not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".ply":
        from .ply_io import load_gaussian_ply
        scene = load_gaussian_ply(p)
    elif suffix == ".sog":
        from .sog_io import load_gaussian_sog
        scene = load_gaussian_sog(p)
    else:
        raise ValueError(
            f"{p}: unsupported scene format {p.suffix!r}; Carveout reads "
            f"{' and '.join(SPLAT_SUFFIXES)}")
    return apply_alignment(scene, cfg)
