# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene alignment: the small rotation that levels a scene.

No splat container records which way is up, and a capture can arrive
tilted: a garage capture leaned 12.75° along its length, so an
axis-aligned box had to be 3.5 m tall for a 2.34 m room and the floor
histogram elected a band holding 6 % of the mass where the levelled floor
holds 42 %. The alignment is the operator's answer to that — proposed
from the scene's own density and adopted by an explicit act, or levelled
from three points clicked on the floor — recorded in the profile under
`scene.alignment` beside the up axis it refines.

Two frames, named:
  file  — the .ply/.sog coordinates (`ply_world`): what renders, what
          indexes Gaussians, the camera poses on disk, the exports.
  scene — the file rotated: p_scene = R · p_file. Every geometric
          decision (the frame, the floor, the volume, the placement, the
          upright boxes) happens here.

R = Rot(up, yaw) · Rot(a1, tilt[1]) · Rot(a0, tilt[0]): the tilt about
the two ground axes a0 < a1 first, then the yaw about the up axis;
Rot(k, θ) is the right-handed rotation about +axis k. Identity when no
alignment is recorded — every scene from before levelling existed is unchanged.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_AXES = {"x": 0, "y": 1, "z": 2}


def _rot(axis: int, deg: float) -> np.ndarray:
    """Right-handed rotation about +axis by `deg` degrees."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    i, j = [k for k in range(3) if k != axis]
    # (i, j) is the plane perpendicular to `axis`, ordered so the rotation
    # is right-handed: about x the plane is (y, z), about y it is (z, x),
    # about z it is (x, y).
    if axis == 1:
        i, j = j, i
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def ground_axes(up_axis: int) -> tuple[int, int]:
    a = [k for k in range(3) if k != up_axis]
    return a[0], a[1]


def is_identity(alignment: dict | None) -> bool:
    if not alignment:
        return True
    t = alignment.get("tilt_deg") or [0.0, 0.0]
    y = alignment.get("yaw_deg") or 0.0
    return abs(float(t[0])) < 1e-9 and abs(float(t[1])) < 1e-9 and abs(float(y)) < 1e-9


def same_block(a: dict | None, b: dict | None, tol_deg: float = 0.01) -> bool:
    """Two alignment blocks name the same levelling (identity counts as a
    block of zeros), within `tol_deg` on every angle."""
    def angles(x):
        if not x:
            return (0.0, 0.0, 0.0)
        t = x.get("tilt_deg") or [0.0, 0.0]
        return (float(t[0]), float(t[1]), float(x.get("yaw_deg") or 0.0))
    return all(abs(p - q) <= tol_deg for p, q in zip(angles(a), angles(b)))


def validate_alignment(alignment: dict | None) -> dict | None:
    """The profile block, checked; None for identity. Fails loudly naming
    the key — a malformed alignment must not level a scene by accident."""
    if alignment is None:
        return None
    if not isinstance(alignment, dict):
        raise ValueError("scene.alignment must be a mapping with tilt_deg "
                         "[a0, a1] and yaw_deg, or null")
    t = alignment.get("tilt_deg", [0.0, 0.0])
    y = alignment.get("yaw_deg", 0.0)
    try:
        t = [float(t[0]), float(t[1])]
        y = float(y)
    except (TypeError, ValueError, IndexError):
        raise ValueError("scene.alignment: tilt_deg must be two numbers "
                         "(degrees about the two ground axes) and yaw_deg "
                         "one number (degrees about the up axis)") from None
    if not (np.isfinite(t).all() and np.isfinite(y)):
        raise ValueError("scene.alignment: angles must be finite")
    if max(abs(t[0]), abs(t[1])) > 90 or abs(y) > 180:
        raise ValueError("scene.alignment: a tilt beyond 90° or a yaw beyond "
                         "180° is not a levelling: pin scene.up_axis first")
    out = dict(alignment)
    out["tilt_deg"], out["yaw_deg"] = t, y
    return out


def alignment_matrix(alignment: dict | None, up_axis: int) -> np.ndarray:
    """R (3x3) for the profile block; identity for None / all zeros."""
    a = validate_alignment(alignment)
    if a is None or is_identity(a):
        return np.eye(3)
    a0, a1 = ground_axes(int(up_axis))
    t0, t1 = a["tilt_deg"]
    return _rot(int(up_axis), a["yaw_deg"]) @ _rot(a1, t1) @ _rot(a0, t0)


def alignment_block(alignment: dict | None) -> dict | None:
    """The two angles alone (for sidecars and manifests); None for identity."""
    a = validate_alignment(alignment)
    if a is None or is_identity(a):
        return None
    return dict(tilt_deg=[round(a["tilt_deg"][0], 4), round(a["tilt_deg"][1], 4)],
                yaw_deg=round(a["yaw_deg"], 4))


def block_of_matrix(R: np.ndarray | None, up_axis: int) -> dict | None:
    """The angles of a matrix (for sidecars and manifests); None for identity."""
    if R is None or np.allclose(R, np.eye(3), atol=1e-12):
        return None
    t, y = angles_of(R, int(up_axis))
    return dict(tilt_deg=[round(t[0], 4), round(t[1], 4)], yaw_deg=round(y, 4))


def quat_of_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    m = R
    tr = float(np.trace(m))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def rotate_quats(qR: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """qR ⊗ quats for an (N, 4) array of (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = qR
    w2, x2, y2, z2 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    out = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)
    return out.astype(quats.dtype)


def to_file_c2w(c2w: np.ndarray, R: np.ndarray | None) -> np.ndarray:
    """A camera pose planned in the scene frame, in the file frame (what
    renders): c2w_file = blockdiag(Rᵀ, 1) · c2w_scene."""
    if R is None:
        return c2w
    T = np.eye(4)
    T[:3, :3] = R.T
    return T @ c2w


def _signed_angle(a: np.ndarray, b: np.ndarray, axis_vec: np.ndarray) -> float:
    """Degrees from a to b about axis_vec (right-handed)."""
    return float(np.degrees(np.arctan2(np.dot(np.cross(a, b), axis_vec),
                                       np.dot(a, b))))


def _tilt_to(u: np.ndarray, target: np.ndarray, up_axis: int) -> tuple[float, float]:
    """The tilt (t0, t1) with Rot(a1, t1)·Rot(a0, t0)·u = target, `target`
    being ±e_up. Rot(a0, t0) zeroes u's a1 component, Rot(a1, t1) then
    zeroes its a0 component; each angle has two solutions and the one that
    lands on `target` (not its opposite) is kept."""
    a0, a1 = ground_axes(up_axis)
    u = np.asarray(u, float) / np.linalg.norm(u)

    def solve(axis: int, v: np.ndarray, comp: int) -> tuple[float, np.ndarray]:
        A = (_rot(axis, 0.0) @ v)[comp]
        B = (_rot(axis, 90.0) @ v)[comp]
        th = float(np.degrees(np.arctan2(-A, B)))
        best = None
        for cand in (th, th + 180.0):
            cand = (cand + 180.0) % 360.0 - 180.0
            w = _rot(axis, cand) @ v
            score = float(w @ target)
            if best is None or score > best[0]:
                best = (score, cand, w)
        return best[1], best[2]

    t0, v = solve(a0, u, a1)
    t1, w = solve(a1, v, a0)
    return t0, t1


def angles_of(R: np.ndarray, up_axis: int) -> tuple[list[float], float]:
    """(tilt_deg [t0, t1], yaw_deg) with alignment_matrix(...) == R."""
    up = int(up_axis)
    e_up = np.eye(3)[up]
    u = R.T @ e_up                       # the file-frame vector R sends to +up
    t0, t1 = _tilt_to(u, e_up, up)
    a0, a1 = ground_axes(up)
    Rt = _rot(a1, t1) @ _rot(a0, t0)
    Q = R @ Rt.T                         # what is left: Rot(up, yaw)
    e0 = np.eye(3)[a0]
    yaw = _signed_angle(e0, Q @ e0, e_up)
    return [round(t0, 6), round(t1, 6)], round(yaw, 6)


def _yaw_to_axis(direction: np.ndarray, up_axis: int) -> float:
    """The yaw in (-45, 45] that turns a ground direction onto the nearest
    ground axis (a box is symmetric, so 90° is the period)."""
    a0, a1 = ground_axes(int(up_axis))
    e_up = np.eye(3)[int(up_axis)]
    d = np.asarray(direction, float).copy()
    d -= e_up * float(d @ e_up)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return 0.0
    d /= n
    best = None
    for axis_vec in (np.eye(3)[a0], -np.eye(3)[a0], np.eye(3)[a1], -np.eye(3)[a1]):
        yaw = _signed_angle(d, axis_vec, e_up)
        if best is None or abs(yaw) < abs(best):
            best = yaw
    return float(best)


def yaw_from_edge(edge_pts, tilt_deg, up_axis: int) -> float:
    """The yaw that squares the levelled ground to two points along one
    straight edge (FILE frame) — a wall, a counter's long side — given the
    tilt already recorded: the Square to an edge act, and the wall
    half of alignment_from_points."""
    w = np.asarray(edge_pts, float)
    if w.shape != (2, 3) or not np.isfinite(w).all():
        raise ValueError("squaring needs exactly two edge points [x, y, z]")
    up = int(up_axis)
    a0, a1 = ground_axes(up)
    Rt = _rot(a1, float(tilt_deg[1])) @ _rot(a0, float(tilt_deg[0]))
    d = Rt @ (w[1] - w[0])
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("the two edge points coincide")
    return float(_yaw_to_axis(d, up))


def alignment_from_points(floor_pts, wall_pts, up_axis: int, up_sign: float) -> dict:
    """The angles from the operator's clicks (FILE frame): three points on
    the floor give the plane's normal, oriented up by `up_sign`; two points
    along a wall give the yaw on the levelled ground (0 without them)."""
    up = int(up_axis)
    f = np.asarray(floor_pts, float)
    if f.shape != (3, 3) or not np.isfinite(f).all():
        raise ValueError("levelling needs exactly three floor points [x, y, z]")
    n = np.cross(f[1] - f[0], f[2] - f[0])
    if np.linalg.norm(n) < 1e-9 * max(np.linalg.norm(f[1] - f[0]),
                                      np.linalg.norm(f[2] - f[0]), 1e-9):
        raise ValueError("the three floor points are on one line; click "
                         "three points far apart on the floor")
    n /= np.linalg.norm(n)
    target = np.eye(3)[up] * float(up_sign)
    if n @ target < 0:
        n = -n
    t0, t1 = _tilt_to(n, target, up)
    yaw = yaw_from_edge(wall_pts, (t0, t1), up) if wall_pts is not None else 0.0
    return dict(tilt_deg=[round(float(t0), 4), round(float(t1), 4)],
                yaw_deg=round(float(yaw), 4))


# --- the reading ---------------------------------------------------------------

def _floor_share(h: np.ndarray, w: np.ndarray, band_frac: float, bin_units: float) -> float:
    """The floor rule's peak, as a CONTRAST: the largest drop from one bin
    of the opacity-weighted height histogram to the bin above it, among
    the bins in the lowest `band_frac` of the robust range, as a share of
    the whole sample's mass. A floor has air above it, so its bin towers
    over the next; a slab cut through a solid body by a tilted direction
    has as much mass in the bin above and scores nothing — the band's top
    edge cutting a bench read 0.52 of the band on a level scene before
    this. The histogram runs over the whole range so the bin above
    the band's last bin exists."""
    lo, hi = np.percentile(h, [1, 99])
    if not hi > lo:
        return 0.0
    nb = max(int((hi - lo) / bin_units) if bin_units > 0 else 0, 20)
    sel = (h >= lo) & (h <= hi)
    hist, edges = np.histogram(h[sel], bins=nb, range=(lo, hi), weights=w[sel])
    band_hi = lo + band_frac * (hi - lo)
    centres = (edges[:-1] + edges[1:]) / 2
    k_max = int(np.searchsorted(centres, band_hi))
    if k_max < 1:
        return 0.0
    above = np.append(hist[1:], 0.0)
    contrast = (hist - above)[:k_max]
    return float(max(contrast.max(), 0.0) / max(w.sum(), 1e-12))


def _wall_score(x: np.ndarray, w: np.ndarray, bin_units: float) -> float:
    lo, hi = np.percentile(x, [1, 99])
    if not hi > lo:
        return 0.0
    nb = max(int((hi - lo) / bin_units) if bin_units > 0 else 0, 10)
    hist, _ = np.histogram(x, bins=nb, range=(lo, hi), weights=w)
    return float(np.sort(hist)[-2:].sum() / max(w.sum(), 1e-12))


def propose_alignment(means_file: np.ndarray, opacities: np.ndarray,
                      up_axis: int, up_sign: float, cfg: dict, path_mode: str,
                      boxes: list | None, current_R: np.ndarray | None = None,
                      frame=None) -> dict:
    """The reading, on FILE-frame means so the angles are
    absolute. `cfg` is in scene units (`hist_bin_m` already divided).

    inside a space / around a subject: the up direction along which the
    floor (or the subject's support) is the sharpest density peak, then the
    yaw that makes the walls sharpest (a room) or the subject's principal
    axes (an object). outdoors: a robust plane fit to the ground map; no
    yaw. Stated, never applied."""
    rcfg = cfg["scene"].get("alignment_read") or {}
    max_tilt = float(rcfg.get("max_tilt_deg", 30))
    tol = float(rcfg.get("level_tol_deg", 1.0))
    gain_min = float(rcfg.get("share_gain_min", 1.5))
    margin_min = float(rcfg.get("yaw_margin_min", 1.2))
    n_sample = int(rcfg.get("sample", 300000))
    band_frac = float(cfg["scene"]["floor_band_frac"])
    bin_units = float(cfg["scene"]["hist_bin_m"])
    up = int(up_axis)
    a0, a1 = ground_axes(up)
    target = np.eye(3)[up] * float(up_sign)     # the levelled up, file frame

    # The tilt is read from the WHOLE scene: the floor is a property of
    # the capture, and a volume drawn around a subject leaves little of
    # it inside (a bench's box read a 37° "floor" through the bench).
    # The same sample every time, so a read after adopting
    # lands on the same angles. The yaw is read inside the volume: the
    # walls of the room, the axes of the subject.
    sel = opacities > 0.3
    idx = np.flatnonzero(sel)
    empty = dict(reading=None, found=False, tilt_deg=None, yaw_deg=None,
                 tilt_total_deg=None, floor_share_as_is=None,
                 floor_share_levelled=None, yaw_margin=None,
                 off_level=False, confident_yaw=False)
    if len(idx) < 1000:
        log.warning("alignment read: %d usable Gaussians, nothing to read", len(idx))
        return empty
    rng = np.random.default_rng(0)
    if len(idx) > n_sample:
        idx = rng.choice(idx, n_sample, replace=False)
    p = means_file[idx].astype(np.float64)
    w = opacities[idx].astype(np.float64)
    if boxes:
        from .volume import in_volume
        pts = means_file if current_R is None else means_file @ current_R.T
        vsel = sel & in_volume(pts, boxes)
        vidx = np.flatnonzero(vsel)
        if len(vidx) > n_sample:
            vidx = rng.choice(vidx, n_sample, replace=False)
        pv, wv = means_file[vidx].astype(np.float64), opacities[vidx].astype(np.float64)
    else:
        pv, wv = p, w

    if path_mode == "ground":
        if frame is None:
            raise ValueError("the ground reading needs the scene frame")
        from .scene_read import ground_map
        gm = ground_map(means_file, opacities, frame, cfg, boxes, budget=False)
        valid = np.isfinite(gm.ground) & gm.valid
        if valid.sum() < 30:
            return dict(empty, reading="ground_map")
        aa, bb = np.nonzero(valid)
        g0 = gm.origin[0] + (aa + 0.5) * gm.cell
        g1 = gm.origin[1] + (bb + 0.5) * gm.cell
        hh = gm.ground[aa, bb]
        A = np.stack([g0, g1, np.ones_like(g0)], 1)
        keep = np.ones(len(hh), bool)
        coef = None
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(A[keep], hh[keep], rcond=None)
            res = hh - A @ coef
            mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-9
            keep = np.abs(res) < 3.0 * 1.4826 * mad
            if keep.sum() < 30:
                break
        p0, p1, _ = coef
        # the plane's upward normal in the file frame: h is the signed
        # height (coordinate x up_sign) over the ground axes
        n_file = np.zeros(3)
        n_file[a0], n_file[a1], n_file[up] = -p0, -p1, float(up_sign)
        n_file /= np.linalg.norm(n_file)
        t0, t1 = _tilt_to(n_file, target, up)
        total = float(np.degrees(np.arccos(np.clip(n_file @ target, -1, 1))))
        u_cur = target if current_R is None else current_R.T @ target
        residual = float(np.degrees(np.arccos(np.clip(n_file @ u_cur, -1, 1))))
        return dict(reading="ground_map", found=True,
                    tilt_deg=[round(t0, 4), round(t1, 4)], yaw_deg=None,
                    tilt_total_deg=round(total, 3),
                    residual_deg=round(residual, 3), residual_yaw_deg=None,
                    floor_share_as_is=None, floor_share_levelled=None,
                    yaw_margin=None, off_level=bool(residual >= tol),
                    off_square=False, confident_yaw=False,
                    ground_cells=int(keep.sum()))

    # --- inside a space / around a subject: the floor's sharpness ------------
    def up_of(t0: float, t1: float) -> np.ndarray:
        return (_rot(a1, t1) @ _rot(a0, t0)).T @ target

    def score(t0: float, t1: float) -> float:
        return _floor_share(p @ up_of(t0, t1), w, band_frac, bin_units)

    # "as-is" is how the scene reads NOW — under the current alignment,
    # the file's axes when there is none; the search itself is absolute
    cur_t = angles_of(current_R, up)[0] if current_R is not None else [0.0, 0.0]
    as_is = score(float(cur_t[0]), float(cur_t[1]))
    best = (score(0.0, 0.0), 0.0, 0.0)
    for step, span, ctr in ((5.0, max_tilt, (0.0, 0.0)), (1.0, 6.0, None), (0.25, 1.5, None)):
        if ctr is None:
            ctr = (best[1], best[2])
        for t0 in np.arange(ctr[0] - span, ctr[0] + span + 1e-9, step):
            for t1 in np.arange(ctr[1] - span, ctr[1] + span + 1e-9, step):
                if max(abs(t0), abs(t1)) > max_tilt + 1e-9:
                    continue
                s = score(float(t0), float(t1))
                if s > best[0]:
                    best = (s, float(t0), float(t1))
    share, t0, t1 = best
    u = up_of(t0, t1)
    total = float(np.degrees(np.arccos(np.clip(u @ target, -1, 1))))
    u_cur = up_of(float(cur_t[0]), float(cur_t[1]))
    residual = float(np.degrees(np.arccos(np.clip(u @ u_cur, -1, 1))))
    found = share >= 0.02          # a floor holds a real share of the scene
    off = bool(found and residual >= tol and share / max(as_is, 1e-9) >= gain_min)

    # --- the yaw, on the levelled ground ----------------------------------------
    # Inside the volume when the frame is level (the walls of the room of
    # interest, the axes of the subject); on the whole scene while it is
    # off level — a box drawn on a leaning frame is a slab through the
    # room, and the walls inside a slab read as square (the garage's first
    # read said yaw 0 with confidence).
    Rt = _rot(a1, t1) @ _rot(a0, t0)
    whole = off or not boxes
    q = (p if whole else pv) @ Rt.T
    g = q[:, [a0, a1]]
    w_prev, w = w, (w if whole else wv)
    if path_mode == "orbit":
        reading = "support+axes"
        c = np.average(g, axis=0, weights=w)
        d = (g - c) * np.sqrt(w)[:, None]
        cov = d.T @ d / w.sum()
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, 1]
        margin = float(np.sqrt(vals[1] / max(vals[0], 1e-12)))
        direction = np.zeros(3)
        direction[a0], direction[a1] = major[0], major[1]
    else:
        reading = "floor+walls"
        phis = np.arange(0.0, 90.0, 0.5)
        scores = np.array([_wall_score(g @ np.array([np.cos(np.deg2rad(ph)),
                                                     np.sin(np.deg2rad(ph))]),
                                       w, bin_units) for ph in phis])
        k = int(np.argmax(scores))
        far = np.abs((phis - phis[k] + 45.0) % 90.0 - 45.0) >= 15.0
        runner = float(scores[far].max()) if far.any() else 0.0
        margin = float(scores[k] / max(runner, 1e-9))
        direction = np.zeros(3)
        direction[a0], direction[a1] = np.cos(np.deg2rad(phis[k])), np.sin(np.deg2rad(phis[k]))
    yaw = _yaw_to_axis(direction, up)
    confident = bool(found and margin >= margin_min)
    cur_yaw = angles_of(current_R, up)[1] if current_R is not None else 0.0
    res_yaw = (yaw - cur_yaw + 45.0) % 90.0 - 45.0
    off_square = bool(found and confident and abs(res_yaw) >= tol)
    return dict(reading=reading, found=bool(found), off_square=off_square,
                tilt_deg=[round(t0, 4), round(t1, 4)] if found else None,
                yaw_deg=(round(yaw, 4) if confident else None),
                yaw_candidate_deg=round(yaw, 4),
                tilt_total_deg=round(total, 3) if found else None,
                residual_deg=round(residual, 3) if found else None,
                residual_yaw_deg=round(res_yaw, 3) if confident else None,
                floor_share_as_is=round(as_is, 4),
                floor_share_levelled=round(share, 4),
                yaw_margin=round(margin, 3), off_level=off,
                confident_yaw=confident)


def describe_level(read: dict, current: dict | None) -> dict:
    """The Check card's line: {fits, reason} in the operator's words."""
    if not read or not read.get("found"):
        return dict(fits=None, reason="no floor found to read the level from; "
                                      "keep the file's axes, or level from three "
                                      "points on a flat surface")
    total = float(read["tilt_total_deg"])
    resid = float(read.get("residual_deg", total))
    ground = read.get("reading") == "ground_map"
    what_reads = "the ground's overall slope" if ground else "the floor"
    levelled = (f"levelled by {float(current['tilt_deg'][0]):.1f}° and "
                f"{float(current['tilt_deg'][1]):.1f}° about the ground axes"
                + (f", {float(current['yaw_deg']):.1f}° about the up axis"
                   if current.get("yaw_deg") else "") + "; "
                if current else "")
    ry = read.get("residual_yaw_deg")
    square = (f"the walls sit {abs(float(ry)):.1f}° off square"
              if read.get("reading") == "floor+walls" else
              f"the subject sits {abs(float(ry)):.1f}° off the axes") \
        if read.get("off_square") and ry is not None else None
    if current and current.get("source") == "levelled":
        # The operator's clicks are the authority. The detector's own
        # read is stated against them — on a capture with no floor in it
        # the surface it elected is the wrong one — never as an instruction
        # to redo what was just done.
        turn = (f" and {abs(float(ry)):.1f}° in turn"
                if read.get("off_square") and ry is not None else "")
        return dict(fits=True, reason=(
            f"{levelled.rstrip('; ')} from the points you clicked; "
            f"{what_reads} as the detector reads it differs by "
            f"{resid:.1f}° in tilt{turn}; your clicks stand"))
    if not read["off_level"]:
        if square:
            return dict(fits=False, reason=f"level ({levelled}{what_reads} reads "
                                           f"{resid:.1f}° off), but {square}: "
                                           f"square the scene for a box that hugs it")
        return dict(fits=True, reason=f"level: {levelled}{what_reads} reads "
                                      f"{resid:.1f}° off")
    if ground:
        return dict(fits=False, reason=(
            f"{levelled}the ground's overall slope reads {resid:.1f}°; level "
            f"the scene only if the capture is tilted, not the site; a real "
            f"hillside is meant to slope"))
    parts = [f"the floor tilts {resid:.1f}° from the up axis"]
    if square:
        parts.append(square)
    what = " and ".join(parts)
    return dict(fits=False, reason=f"{levelled}{what}; a straight box cannot "
                                   f"hug this scene; level the scene, then "
                                   f"propose again")
