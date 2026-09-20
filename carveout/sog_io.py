# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""PlayCanvas SOG v2 (`.sog`) reading.

A `.sog` is a ZIP of `meta.json` plus lossless WebP planes, around a
sixteenth the size of the `.ply` it was written from — which is what makes
a scene shippable. Every field is quantised: 16-bit log-encoded means,
8-bit codebook indices for scales and the SH DC term, 8-bit quaternion
components, and rest coefficients drawn from a k-means palette.

The decode below follows Spark's own `unpackPcSogs`
(@sparkjsdev/spark 2.1.0), deliberately so: the viewer renders the same
file with that code, and the per-Gaussian instance ids Carveout lifts are
nothing but positions in it. Both sides walk the planes in raster order —
splat i at pixel i, nothing culled, nothing reordered — which is what keeps
`instance_ids.bin` aligned with what the browser draws. The one departure
is the SH palette stride: Spark 2.1.0 hardcodes 15 pixels per entry, which
is only right for three bands; the writer's layout (one pixel per rest
coefficient) is what is decoded here, so one- and two-band files load.

SOG v1 (the unzipped directory form, carrying mins/maxs where v2 carries
codebooks) is refused rather than half-supported.
"""

import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .scene_io import GaussianScene

log = logging.getLogger(__name__)

# Rest coefficients (beyond the DC term) carried by each SH band count.
_REST_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}
# The centroid table packs one pixel per rest coefficient, 64 palette
# entries per row: the writer (splat-transform write-sog.ts) sizes it
# 64 * _REST_COEFFS[bands] wide, so the stride is the band's count — 15
# only for three bands.
_PALETTE_PER_ROW = 64


def _plane(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """One WebP plane as an (H, W, 4) uint8 array."""
    return np.asarray(Image.open(io.BytesIO(zf.read(name))).convert("RGBA"))


def _rows(plane: np.ndarray, n: int, name: str) -> np.ndarray:
    """(H, W, 4) -> (n, 4): splat i lives at pixel i in raster order."""
    flat = plane.reshape(-1, 4)
    if len(flat) < n:
        raise ValueError(
            f"SOG plane {name} holds {len(flat)} pixels, fewer than the "
            f"{n} splats meta.json declares")
    return flat[:n]


def _decode_quats(plane: np.ndarray, path: Path) -> np.ndarray:
    """Largest-component packing -> (N, 4) wxyz.

    RGB carry the three components that were kept, in wxyz order skipping
    the one left out, rescaled from [0, 255] onto [-1/sqrt2, 1/sqrt2];
    alpha is 252 + the index of the omitted component, which the unit norm
    recovers.
    """
    largest = plane[:, 3].astype(np.int32) - 252
    if largest.min() < 0 or largest.max() > 3:
        raise ValueError(
            f"{path}: quaternion plane alpha outside 252..255, not the "
            f"quaternion_packed encoding Carveout reads")
    comp = (plane[:, :3].astype(np.float32) / 255.0 - 0.5) * np.sqrt(2.0)
    omitted = np.sqrt(np.maximum(0.0, 1.0 - (comp ** 2).sum(axis=1)))
    quats = np.empty((len(plane), 4), dtype=np.float32)
    for k in range(4):                      # k indexes wxyz
        sel = largest == k
        for slot, col in enumerate(c for c in range(4) if c != k):
            quats[sel, col] = comp[sel, slot]
        quats[sel, k] = omitted[sel]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    return quats


def _decode_sh(zf: zipfile.ZipFile, meta: dict, dc: np.ndarray,
               n: int) -> np.ndarray:
    """(N, K, 3) with the DC term at index 0 — the layout the raster wants.

    Rest coefficients are palettised: a 16-bit label per splat picks a
    centroid, whose 15 pixels' channels index the shared codebook.
    """
    shn = meta.get("shN")
    bands = int(shn["bands"]) if shn else 0
    if bands not in _REST_COEFFS:
        raise ValueError(f"SOG shN.bands={bands!r}; expected 0 to 3")
    k_rest = _REST_COEFFS[bands]
    if not k_rest:
        return dc[:, None, :]

    centroids = _plane(zf, shn["files"][0])
    width = centroids.shape[1]
    if width != _PALETTE_PER_ROW * k_rest:
        raise ValueError(
            f"SOG shN centroid table is {width} pixels wide; expected "
            f"{_PALETTE_PER_ROW * k_rest} for {bands} SH band(s)")
    labels_plane = _rows(_plane(zf, shn["files"][1]), n, shn["files"][1])
    labels = (labels_plane[:, 0].astype(np.int32)
              | (labels_plane[:, 1].astype(np.int32) << 8))
    offset = ((labels // _PALETTE_PER_ROW) * width
              + (labels % _PALETTE_PER_ROW) * k_rest)
    coeff = offset[:, None] + np.arange(k_rest, dtype=np.int32)[None, :]
    flat = centroids.reshape(-1, 4)[:, :3]
    if coeff.max() >= len(flat):
        raise ValueError(
            f"SOG shN labels index past the {len(flat)}-pixel centroid table")
    codebook = np.asarray(shn["codebook"], dtype=np.float32)
    rest = codebook[flat[coeff]]                       # (N, k_rest, 3)
    return np.concatenate([dc[:, None, :], rest], axis=1)


def load_gaussian_sog(path: str | Path) -> GaussianScene:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "meta.json" not in names:
            raise ValueError(f"{path} is not a SOG bundle: no meta.json inside")
        meta = json.loads(zf.read("meta.json"))
        version = meta.get("version")
        if version != 2:
            raise ValueError(
                f"{path}: SOG version {version!r}; Carveout reads v2 "
                f"bundles only. Write one with "
                f"`splat-transform scene.ply scene.sog`.")
        n = int(meta["count"])
        for key in ("means", "scales", "quats", "sh0"):
            if key not in meta:
                raise ValueError(f"{path}: SOG meta.json has no {key!r} block")
            for f in meta[key]["files"]:
                if f not in names:
                    raise ValueError(
                        f"{path}: meta.json names {f}, which the bundle "
                        f"does not contain")

        # means: two 8-bit planes per axis make a 16-bit unit value, lerped
        # into the declared range, then taken back out of the writer's log
        # encoding.
        lo = _rows(_plane(zf, meta["means"]["files"][0]), n, "means_l")[:, :3]
        hi = _rows(_plane(zf, meta["means"]["files"][1]), n, "means_u")[:, :3]
        unit = ((hi.astype(np.uint32) << 8) | lo) / 65535.0
        mins = np.asarray(meta["means"]["mins"], dtype=np.float64)
        maxs = np.asarray(meta["means"]["maxs"], dtype=np.float64)
        logv = mins + unit * (maxs - mins)
        means = (np.sign(logv) * np.expm1(np.abs(logv))).astype(np.float32)

        # scales: a codebook index per axis, the codebook itself in log space
        scale_cb = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
        scale_idx = _rows(_plane(zf, meta["scales"]["files"][0]), n,
                          "scales")[:, :3]
        scales = np.exp(scale_cb[scale_idx])

        quats = _decode_quats(
            _rows(_plane(zf, meta["quats"]["files"][0]), n, "quats"), path)

        # sh0: RGB index the DC codebook; alpha IS the opacity, already
        # sigmoid-activated (where the .ply stores the pre-sigmoid logit).
        sh0 = _rows(_plane(zf, meta["sh0"]["files"][0]), n, "sh0")
        dc = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)[sh0[:, :3]]
        opacities = (sh0[:, 3] / 255.0).astype(np.float32)

        sh = _decode_sh(zf, meta, dc, n)

    log.info(
        "loaded %s: %d gaussians, SH degree %d (SOG v2, %s)",
        path.name, n, int(np.sqrt(sh.shape[1])) - 1,
        (meta.get("asset") or {}).get("generator", "unknown generator"),
    )
    return GaussianScene(
        means=means, scales=scales, quats=quats, opacities=opacities, sh=sh,
        source_path=str(path),
    )
