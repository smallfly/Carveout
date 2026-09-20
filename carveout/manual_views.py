# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Manual detection viewpoints: ingestion of viewer-captured manual_views.json.

The viewer captures hand-picked detection viewpoints into
data/scenes/<name>/manual_views.json. Poses are stored in RAW
.ply space with the three.js/OpenGL camera convention: quaternion [x, y, z, w],
camera looks down -Z, up +Y. "ply space" names the scene's own untransformed
coordinate frame and is the on-disk `space` literal — it stays that word for
a .sog scene too, which carries the same frame; renaming it would invalidate
every manual_views.json already written. The pipeline's camera model is an OpenCV c2w
(x right, y down, z forward — cameras.look_at), so conversion flips the
camera-frame Y and Z axes: R_cv = R_gl @ diag(1, -1, -1).

AUTHORITY RULE (viewer contract): the quaternion is the authoritative
orientation. `target` is advisory — the ROI surface point when target_source
is "raycast", an arbitrary point along the view axis when "fallback" — and is
used only for a sanity warning; orientation is never re-derived from it.

Manual views render at the pipeline's detection resolution, H-anchored:
height = the profile's render height, width = height x captured aspect
(clamped to render.manual_max_aspect, warned), focal from the captured
VERTICAL fov_deg. The captured canvas `resolution` is metadata only.

manual_views.json CONTENT is part of the stage1 cache key (sequencing's
`fingerprint`) and cascades through every downstream stage manifest, so
adding/editing/removing a view invalidates exactly the affected stages on
the next run (verify REFUSES instead — re-running it alone cannot fix a
stale upstream).
"""

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def resolve_manual_views_path(workdir, cfg: dict) -> Path:
    """render.manual_views override, else <workdir>/manual_views.json (the
    scene's files live in the workdir — sequencing.SCENE_FILES)."""
    return Path(cfg["render"].get("manual_views")
                or Path(workdir) / "manual_views.json")


def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_manual_views(path, cfg: dict) -> list[dict]:
    """Parse + convert manual views; [] when the file does not exist.

    Returns dicts: id, label, target_source, fov_deg, c2w (4x4 OpenCV c2w in
    raw .ply space), and width/height/fx/fy/cx/cy at detection resolution.
    Unknown version/space and malformed views are rejected with clear errors.
    """
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if data.get("version") != 1:
        raise ValueError(
            f"{p}: unsupported manual_views version {data.get('version')!r} "
            f"(this pipeline reads version 1)")
    if data.get("space") != "ply":
        raise ValueError(
            f"{p}: unsupported space {data.get('space')!r}; the pipeline "
            f"consumes raw .ply-space poses only (\"ply\")")

    std_h = int(cfg["render"]["height"])
    max_aspect = float(cfg["render"]["manual_max_aspect"])
    views = []
    for i, v in enumerate(data.get("views", [])):
        vid = str(v.get("id") or f"index{i}")
        label = str(v.get("label") or vid[:8])
        where = f"{p} view {i} ({label!r})"
        pos = np.asarray(v["position"], dtype=float)
        q = np.asarray(v["quaternion"], dtype=float)
        tgt = np.asarray(v["target"], dtype=float)
        fov = float(v["fov_deg"])
        aspect = float(v["aspect"])
        if (pos.shape != (3,) or q.shape != (4,) or tgt.shape != (3,)
                or not (np.isfinite(pos).all() and np.isfinite(q).all()
                        and np.isfinite(tgt).all()
                        and np.isfinite(fov) and np.isfinite(aspect))):
            raise ValueError(f"{where}: malformed position/quaternion/target/"
                             f"fov_deg/aspect")
        qn = float(np.linalg.norm(q))
        if qn < 1e-6:
            raise ValueError(f"{where}: zero-norm quaternion")
        if not 5.0 <= fov <= 175.0 or aspect <= 0:
            raise ValueError(f"{where}: implausible fov_deg={fov} "
                             f"aspect={aspect}")
        if aspect > max_aspect:
            log.warning("manual view %r: aspect %.2f clamped to "
                        "render.manual_max_aspect %.2f (sides cropped, "
                        "vertical fov preserved)", label, aspect, max_aspect)
            aspect = max_aspect

        # three.js/OpenGL camera (looks down -Z, up +Y) -> OpenCV c2w.
        c2w = np.eye(4)
        c2w[:3, :3] = _quat_to_rot(*(q / qn)) @ np.diag([1.0, -1.0, -1.0])
        c2w[:3, 3] = pos

        # Sanity cross-check ONLY (authority rule): the target sits on the
        # view axis in both raycast and fallback modes, so the quaternion
        # forward should point at it — a convention bug shows up here loudly.
        to_t = tgt - pos
        tn = float(np.linalg.norm(to_t))
        if tn > 1e-6:
            ang = float(np.degrees(np.arccos(
                np.clip(np.dot(c2w[:3, 2], to_t / tn), -1.0, 1.0))))
            if ang > 2.0:
                log.warning(
                    "manual view %r (%s): quaternion forward is %.1f deg off "
                    "the position->target axis (target_source=%s); "
                    "orientation follows the QUATERNION; check the capture",
                    label, vid, ang, v.get("target_source"))

        w = int(round(std_h * aspect))
        f = 0.5 * std_h / np.tan(0.5 * np.deg2rad(fov))
        views.append(dict(
            id=vid, label=label,
            target_source=v.get("target_source", "unknown"),
            fov_deg=fov, c2w=c2w,
            width=w, height=std_h, fx=f, fy=f, cx=w / 2, cy=std_h / 2))
    log.info("loaded %d manual view(s) from %s", len(views), p)
    return views


def pose_from_frame(fr: dict, cams: dict) -> dict:
    """One cameras.json frame -> a manual_views.json view body.

    The inverse of what `load_manual_views` does, and the ONE place the two
    camera representations meet in this direction. Used to convert an
    auto-sampled view into a hand-held one, so the operator can adjust a pose
    the sampler proposed.

    Two conversions, both exact:
      * intrinsics -> `fov_deg` + `aspect`. Manual views are H-anchored (the
        focal comes from the captured VERTICAL fov at render height), so
        `fov_deg = 2*atan(h / 2*fy)` and `aspect = w / h` reproduce fx/fy on
        the way back in. Auto frames inherit the global block; manual ones
        carry their own — `frame_intrinsics` already encodes that rule.
      * OpenCV c2w -> the three.js/OpenGL quaternion the file stores. `c2w`
        is `R_gl @ diag(1, -1, -1)`; that matrix is its own inverse, so the
        same multiply undoes it.

    `target` is advisory (see the AUTHORITY RULE above) but is checked against
    the quaternion at load, so it is placed one metre down the view axis
    rather than left absent — target_source says it was derived, not picked.
    """
    fx, fy, cx, cy, w, h = frame_intrinsics(fr, cams)
    c2w = np.asarray(fr["c2w"], dtype=float)
    R_gl = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    pos = c2w[:3, 3]
    return dict(
        position=[float(v) for v in pos],
        quaternion=[float(v) for v in _rot_to_quat(R_gl)],
        target=[float(v) for v in (pos + c2w[:3, 2])],
        target_source="derived",
        fov_deg=float(np.degrees(2 * np.arctan(h / (2 * fy)))),
        aspect=float(w) / float(h))


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [x, y, z, w] (the file's storage order).

    Shepperd's method: pivot on the largest diagonal term so the divisor is
    never near zero — the naive trace formula loses precision at 180 degrees,
    which a camera looking straight back along an axis reaches exactly.
    """
    m, t = R, float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def frame_intrinsics(fr: dict, cams: dict) -> tuple:
    """(fx, fy, cx, cy, width, height) for one cameras.json frame entry.

    Auto-sampled frames share the global intrinsics block; manual frames
    carry their own (captured fov/aspect at detection resolution).
    """
    return (fr.get("fx", cams["fx"]), fr.get("fy", cams["fy"]),
            fr.get("cx", cams["cx"]), fr.get("cy", cams["cy"]),
            fr.get("width", cams["width"]), fr.get("height", cams["height"]))


def stage1_manual_record(workdir) -> dict:
    """stage1 manifest's manual-views record ({path, hash, n_views}).

    {'hash': 'absent'} for a missing stage1 or a pre-feature manifest.
    """
    mp = Path(workdir) / "stage1/manifest.json"
    if not mp.exists():
        return {"hash": "absent"}
    return json.loads(mp.read_text()).get("manual_views") or {"hash": "absent"}


def stale_manual_views(cached_manifest: dict, workdir, stage: str) -> bool:
    """True (with one unmissable log line) when this stage's cached manifest
    was produced against different manual-views content than stage1's."""
    cur = stage1_manual_record(workdir).get("hash", "absent")
    rec = cached_manifest.get("manual_views_hash", "absent")
    if cur == rec:
        return False
    log.warning(
        "%s cache INVALIDATED: manual views changed upstream (this stage saw "
        "%s, stage1 now has %s); re-running %s from scratch",
        stage, rec[:12], cur[:12], stage)
    return True
