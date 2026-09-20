# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 4: per-instance localization + splat_analyzer-compatible export.

interactions.json carries EXACTLY the baseline schema:
objects[] {label, position, rotation(identity), scale(AABB full extents),
frames[] sorted by score desc}; frame_annotations{frame_idx: [...]}. Everything
Carveout adds lives under `extended` (per-object: instance id, aggregate
confidence, OBB, gaussian count; top-level: volume-scoping outcomes, audit).

Also: per-instance .ply subsets (SuperSplat-inspectable) and debug overlays with
projected 3D boxes.
"""

import json
import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .detect import PALETTE
from .exemplars import stage2_exemplar_record, stale_exemplars
from .lift import (_effective_threshold, _load_detections, _mask_path,
                   detection_params)
from .manual_views import (frame_intrinsics, stage1_manual_record,
                           stale_manual_views)
from .ply_io import write_gaussian_ply
from .projection import project_visible
from .refusal import Refusal
from .scene_io import load_scene
from .units import scale_from_stage1, scene_units_cfg

log = logging.getLogger(__name__)


def _mat2quat(R: np.ndarray) -> list[float]:
    """Rotation matrix -> quaternion (x, y, z, w), splat_analyzer field order."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, \
            (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    return [float(x), float(y), float(z), float(w)]


# Association hold reasons written at export. Part of
# the stage-4 fingerprint: a vocabulary change changes what a hold means.
HOLD_REASONS = ("ambiguous_det", "refused_merge", "ambiguous_unassigned")


def export_params(cfg: dict) -> dict:
    """Effective decision parameters of the export: the floors, the
    matching and confidence knobs, the OBB choice, the association-hold
    fraction and vocabulary, and the detect-side thresholds that
    select the detections it matches against."""
    E = cfg["export"]
    keys = ("min_matched_frames", "det_match_min_overlap",
            "top_k_confidence", "visibility_min_points",
            "obb_planar_ratio", "obb_planar_min_updot",
            "hold_unassigned_frac")
    p = {k: E.get(k) for k in keys}
    p["hold_reasons"] = list(HOLD_REASONS)
    p["detect"] = detection_params(cfg)
    return p


def _mask_bbox(stage2: Path, frame_idx: int, concept: str,
               det_idx: int) -> list[float] | None:
    m = np.asarray(Image.open(_mask_path(stage2, frame_idx, concept,
                                         det_idx))) > 127
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _track_frames(track: dict, stage2: Path) -> tuple[list[dict], int]:
    """§3.7: the track's own detections ARE the supporting frames — one
    entry per view (the highest-scoring NON-ambiguous detection in it).
    Ambiguous detections are excluded from frames[] and from confidence;
    their count is returned so the extended entry can say so."""
    best: dict[int, dict] = {}
    excluded = 0
    for d in track["dets"]:
        if d["ambiguous"]:
            excluded += 1
            continue
        cur = best.get(d["frame_idx"])
        if cur is None or d["score"] > cur["score"]:
            best[d["frame_idx"]] = d
    entries = []
    for fidx, d in best.items():
        box = _mask_bbox(stage2, fidx, d["concept"], d["det_idx"])
        if box is None:
            continue
        entries.append(dict(frame_idx=fidx, box=box,
                            score=round(d["score"], 4), source=d["source"]))
    return entries, excluded


def _hold_reasons(track: dict, frac_min: float) -> tuple[list[str], dict]:
    """Association hold reasons for one track, plus the readout the report
    and the operator see next to the verdict fields."""
    reasons = []
    if track["ambiguous_dets"]:
        reasons.append("ambiguous_det")
    if track["bridge_refused"]:
        reasons.append("refused_merge")
    if track["ambiguous_unassigned_frac"] >= frac_min:
        reasons.append("ambiguous_unassigned")
    readout = dict(
        margin=track["margin"], margin_low_frac=track["margin_low_frac"],
        ambiguous_dets=track["ambiguous_dets"],
        ambiguous_unassigned=track["ambiguous_unassigned"],
        ambiguous_unassigned_frac=track["ambiguous_unassigned_frac"],
        bridge_refused=[dict(edge=r["edge"], pair=r["pair"],
                             affinity=r["affinity"], third=r["third"],
                             third_side=r["third_side"])
                        for r in track["bridge_refused"]],
        conflicting_support=track["conflicting_support"],
        track_components=track["track_components"]["count"],
        views=track["views"])
    return reasons, readout


def run_export(scene_path: str, workdir: str, cfg: dict, force: bool = False,
               should_cancel=None) -> dict:
    from .sequencing import (check_cancel, fingerprint, stale_params,
                             stale_upstream, upstream_fingerprint)   # lazy
    from .volume import resolve_volume_path
    out = Path(workdir) / "stage4"
    manifest_path = out / "manifest.json"
    params = export_params(cfg)
    # the export's volume-scoping outcomes follow the volume on disk: its
    # content is a decision parameter, as in the lift
    params["volume"] = fingerprint(resolve_volume_path(workdir, cfg))
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        # fresh only against the stage-3 manifest now on disk and
        # these parameters — a re-run lift followed by an export used to
        # serve the old export over the new lift.
        if (not stale_manual_views(cached, workdir, "stage4")
                and not stale_exemplars(cached, workdir, "stage4")
                and not stale_upstream(cached, workdir, "stage4",
                                       "upstream_stage3",
                                       "stage3/manifest.json")
                and not stale_params(cached, params, "stage4")):
            log.info("stage4 cached at %s (inputs unchanged)", out)
            return cached
    # A re-export with different instance labels must not leave stale .ply
    # files behind — verify resolves each instance by globbing {idx:03d}_*.ply
    # and takes the first match (verify once judged the previous bank's
    # 004_screwdriver.ply while verifying the new run's
    # workbench #4). Same policy as render's frames/depth cleanup.
    for d in (out / "instances", out / "debug"):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    stage1, stage2, stage3 = (Path(workdir) / s for s in ("stage1", "stage2", "stage3"))
    cams = json.loads((stage1 / "cameras.json").read_text())
    classes = json.loads((stage3 / "classes.json").read_text())
    labels = np.load(stage3 / "labels.npy")
    instances = np.load(stage3 / "instances.npy")
    audit = json.loads((stage3 / "instance_audit.json").read_text())
    lift_manifest = json.loads((stage3 / "manifest.json").read_text())
    scene = load_scene(scene_path, cfg)
    ecfg = cfg["export"]
    R_align = scene.alignment          # None = the file's axes
    # with tracks, the instance's detections are in stage3/tracks.json
    # and export needs no mask-overlap match. Without it (a connectivity-
    # mode lift), the previous depth-tested match stays in force.
    tracks_path = stage3 / "tracks.json"
    tracks = None
    if lift_manifest.get("instancing", "connectivity") == "tracks":
        if not tracks_path.exists():
            raise Refusal(f"stage3 says instancing=tracks but {tracks_path} "
                          f"is missing; Run pipeline again (it re-runs the "
                          f"lift)", gate="verify_consent")
        tracks = json.loads(tracks_path.read_text())
        if tracks.get("instancing_schema") != lift_manifest.get(
                "instancing_schema"):
            raise Refusal(f"stage3/tracks.json schema "
                          f"{tracks.get('instancing_schema')} does not match "
                          f"the lift manifest; Run pipeline again (it "
                          f"re-runs the lift)", gate="verify_consent")
    per_view = None if tracks else _load_detections(stage2, cfg)[1]
    hold_frac = float(ecfg["hold_unassigned_frac"])
    stage1_manifest = json.loads((stage1 / "manifest.json").read_text())
    # The projection's lengths in scene units: the near plane for the
    # debug frames and, on the connectivity path, the visibility slack the
    # association depth-tests with — the same numbers the render counted
    # coverage with, at the one factor stage 1 ran at, never a literal.
    scale_rec = scale_from_stage1(cfg, stage1_manifest)
    cfg_units, _ = scene_units_cfg(cfg, scale_rec.value)
    near_units = cfg_units["render"]["near_plane"]
    ccfg = cfg_units["render"]["coverage"]
    up_axis = stage1_manifest["scene_frame"]["up_axis"]
    up_sign = stage1_manifest["scene_frame"]["up_sign"]
    # What the numbers mean: every position, AABB `scale` and OBB below is
    # in SCENE UNITS, as the .ply is, and the export says so. It names the
    # one factor the run used (recorded, or measured with the ruler) so a
    # consumer can multiply; it carries no metre geometry itself.
    scene_extent = (stage1_manifest["scene_frame"].get("notes") or {}
                    ).get("extent")
    units = dict(
        coords="scene units (ply_world), as the source .ply",
        scene_extent_units=([round(float(v), 4) for v in scene_extent]
                            if scene_extent else None),
        scale_m_per_unit=scale_rec.value, scale_source=scale_rec.source,
        measured=scale_rec.measured,
        note="objects[].position / scale / extended.obb are scene units; "
             "multiply by scale_m_per_unit for metres; exact when "
             "scale_source is 'recorded', approximate when 'measured' (the "
             "operator measured one known length on the canvas; the record "
             "is under `measured`)")

    inst_ids = sorted(set(instances[instances >= 0].tolist()))
    objects, extended_objects = [], []
    frame_annotations: dict[str, list] = defaultdict(list)
    dropped_no_evidence = []   # export floor, config-gated

    for inst in inst_ids:
        # cooperative cancel between instances — export was the one stage a
        # cancel had to wait out in full (per-instance depth + mask reads,
        # minutes on large scenes)
        check_cancel(should_cancel)
        gsel = instances == inst
        cls = classes[labels[gsel][0]]
        # position and AABB in the FILE frame (what interactions.json
        # describes); the OBB is found upright in the SCENE frame and
        # rotated back — with no alignment the two are one array
        pts_file = scene.means_file[gsel]
        pts = scene.means[gsel]
        ops = scene.opacities[gsel]
        centroid_file = np.average(pts_file, axis=0, weights=ops)
        centroid = np.average(pts, axis=0, weights=ops)

        aabb_min, aabb_max = pts_file.min(axis=0), pts_file.max(axis=0)
        scale = aabb_max - aabb_min

        # OBB. Unconstrained PCA follows mass distribution, not gravity — its
        # tilted boxes read as "wrong" on grounded room objects — but gravity yaw-only boxes cannot fit tilted planar objects
        # (open door, ceiling-tilted acoustic panels).
        # Decided PER INSTANCE: full PCA iff the instance is strongly planar
        # AND its plane normal is far from horizontal; gravity-aligned
        # otherwise. (The global `obb_mode: gravity|pca` override was an
        # A/B branch, removed.)
        d = (pts - centroid) * np.sqrt(ops)[:, None]
        cov = d.T @ d / ops.sum()
        vals, vecs = np.linalg.eigh(cov)   # ascending; vecs[:, 0] = plane normal
        up_i = "xyz".index(up_axis)
        thin = np.sqrt(vals[0] / max(vals[1], 1e-12)) < ecfg["obb_planar_ratio"]
        tilted = abs(vecs[up_i, 0]) > ecfg["obb_planar_min_updot"]
        mode = "pca" if (thin and tilted) else "gravity"
        if mode == "gravity":
            g = [i for i in range(3) if i != up_i]
            cov2 = d[:, g].T @ d[:, g] / ops.sum()
            _, v2 = np.linalg.eigh(cov2)
            e1 = np.zeros(3)
            e1[g[0]], e1[g[1]] = v2[0, 1], v2[1, 1]   # major ground axis
            upv = np.zeros(3)
            upv[up_i] = up_sign
            e2 = np.cross(upv, e1)
            R = np.stack([e1, e2, upv], axis=1)
        else:
            R = vecs[:, ::-1]  # major axis first
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        proj = (pts - centroid) @ R
        obb_extents = proj.max(axis=0) - proj.min(axis=0)
        obb_center = centroid + R @ ((proj.max(axis=0) + proj.min(axis=0)) / 2)
        if R_align is not None:            # back to the file frame
            obb_center = R_align.T @ obb_center
            R = R_align.T @ R
        centroid = centroid_file

        # Supporting frames. Tracks: the track's own non-ambiguous
        # detections, one per view (§3.7). Connectivity: project instance
        # gaussians into each view (depth-tested) and match against that
        # view's detections of the same class.
        frames_entries = []
        hold_reasons, association, amb_excluded = [], None, 0
        track = None
        if tracks is not None:
            ti = tracks["instance_to_track"].get(str(int(inst)))
            if ti is None:
                raise Refusal(f"instance {inst} ({cls}) has no track in "
                              f"stage3/tracks.json: stage3 is inconsistent; "
                              f"Run pipeline again (it re-runs the lift)",
                              gate="verify_consent")
            track = tracks["tracks"][ti]
            if track["class"] != cls:
                raise Refusal(f"instance {inst}: labels say {cls!r}, "
                              f"tracks.json says {track['class']!r}; "
                              f"Run pipeline again (it re-runs the lift)",
                              gate="verify_consent")
            frames_entries, amb_excluded = _track_frames(track, stage2)
            hold_reasons, association = _hold_reasons(track, hold_frac)
        for fr in (cams["frames"] if tracks is None else []):
            proj = project_visible(
                pts, fr, cams,
                lambda: np.load(stage1 / f"depth/depth_{fr['frame_idx']:04d}.npy"
                                ).astype(np.float32),
                near=near_units, tol_rel=ccfg["depth_tol_rel"],
                tol_abs=ccfg["depth_tol"],
                min_points=ecfg["visibility_min_points"])
            if proj is None:
                continue
            u, v, vis = proj.u, proj.v, proj.vis
            best = None
            for det in per_view.get(fr["frame_idx"], []):
                if det["concept"] != cls:
                    continue
                mp = _mask_path(stage2, fr["frame_idx"], cls, det["det_idx"])
                m = np.asarray(Image.open(mp)) > 127
                overlap = float(m[v[vis], u[vis]].mean())
                if overlap >= ecfg["det_match_min_overlap"] and (
                        best is None or det["score"] > best[0]["score"]):
                    ys, xs = np.nonzero(m)
                    best = (det, [float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())])
            if best:
                frames_entries.append(dict(frame_idx=fr["frame_idx"],
                                           box=best[1],
                                           score=round(best[0]["score"], 4),
                                           source=best[0].get("source", "text")))
        frames_entries.sort(key=lambda e: -e["score"])
        # Detection-source attribution lives under `extended` (the baseline
        # objects[].frames schema stays exact); the provenance report reads it
        # to tell exemplar-only rescues from mixed support.
        support_by_source: dict[str, int] = {}
        for e in frames_entries:
            src = e.pop("source")
            support_by_source[src] = support_by_source.get(src, 0) + 1
        top = [e["score"] for e in frames_entries[:ecfg["top_k_confidence"]]]
        aggregate_conf = round(float(np.mean(top)), 4) if top else 0.0

        # Export floor (config-gated, default OFF): instances
        # whose depth-tested projections matched NO detection in any view are
        # the zero-confidence sliver artifact (lift shards just above
        # min_instance_gaussians with no attributable 2D evidence). With
        # min_matched_frames >= 1 they are dropped here and recorded.
        if len(frames_entries) < ecfg["min_matched_frames"]:
            dropped_no_evidence.append(dict(
                instance_id=int(inst), label=cls, gaussians=int(gsel.sum()),
                matched_frames=len(frames_entries)))
            log.info("export floor: dropped %s (%d gaussians, %d matched "
                     "frames < %d)", cls, int(gsel.sum()),
                     len(frames_entries), ecfg["min_matched_frames"])
            continue
        obj_idx = len(objects)

        objects.append(dict(
            label=cls,
            position={k: float(x) for k, x in zip("xyz", centroid)},
            rotation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            scale={k: float(x) for k, x in zip("xyz", scale)},
            frames=frames_entries,
        ))
        ext = dict(
            instance_id=int(inst), gaussian_count=int(gsel.sum()),
            aggregate_confidence=aggregate_conf,
            support_by_source=support_by_source,
            mean_opacity=round(float(ops.mean()), 4),
            obb=dict(center=[float(x) for x in obb_center],
                     rotation_xyzw=_mat2quat(R),
                     extents=[float(x) for x in obb_extents],
                     mode=mode),
            # §3.7/§6: the association readout and the hold reasons live
            # next to the verdict fields; confidence stays detection
            # confidence and is never blended with them.
            hold_reasons=hold_reasons,
        )
        if track is not None:
            ext["ambiguous_dets_excluded"] = amb_excluded
            ext["association"] = association
        extended_objects.append(ext)
        for e in frames_entries:
            frame_annotations[str(e["frame_idx"])].append(dict(
                label=cls, object_idx=obj_idx, box=e["box"], score=e["score"]))

        write_gaussian_ply(scene, out / "instances" /
                        f"{obj_idx:03d}_{cls.replace(' ', '_')}.ply", mask=gsel)

    from .alignment import block_of_matrix
    interactions = dict(
        objects=objects,
        frame_annotations=dict(frame_annotations),
        # the frame these numbers are in (the file's) and the levelling
        # the run worked under, so a consumer can level them
        scene_frame=dict(coords="ply_world", up_axis=up_axis, up_sign=up_sign,
                         alignment=block_of_matrix(R_align, "xyz".index(up_axis))),
        extended=dict(
            generator="carveout",
            units=units,
            objects=extended_objects,
            classes=classes,
            volume_scoping=dict(
                out_of_volume_by_class=lift_manifest.get(
                    "out_of_volume_by_class", {}),
                note="classes detected in 2D whose gaussians fell outside the "
                     "volume produce no objects (through-door test)"),
            export_floor=dict(min_matched_frames=ecfg["min_matched_frames"],
                              dropped=dropped_no_evidence),
            # PRE-verification association holds (§3.7): set here, from
            # tracks.json and the audit, so a skipped verification cannot
            # hide them. `held_association` is reported as such and never
            # added to the verdict total.
            export=dict(
                instancing=("tracks" if tracks is not None else "connectivity"),
                hold_reasons_vocabulary=list(HOLD_REASONS),
                hold_unassigned_frac=hold_frac,
                held_association=sum(1 for e in extended_objects
                                     if e.get("hold_reasons")),
                held_by_reason={r: sum(1 for e in extended_objects
                                       if r in e.get("hold_reasons", []))
                                for r in HOLD_REASONS},
                ambiguous_dets_excluded=sum(
                    e.get("ambiguous_dets_excluded", 0)
                    for e in extended_objects),
                dropped_tracks=(tracks or {}).get("dropped_tracks", [])),
            instance_audit=audit,
        ),
    )
    (out / "interactions.json").write_text(json.dumps(interactions, indent=1))

    # Debug overlays: projected AABBs of supporting instances per frame.
    corners_unit = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    edges = [(a, b) for a in range(8) for b in range(8)
             if a < b and bin(a ^ b).count("1") == 1]
    for fr in cams["frames"]:
        entries = frame_annotations.get(str(fr["frame_idx"]), [])
        if not entries:
            continue
        img = Image.open(stage1 / fr["file"]).convert("RGB")
        dr = ImageDraw.Draw(img)
        fx, fy, cx, cy, w, h = frame_intrinsics(fr, cams)
        w2c = np.linalg.inv(np.array(fr["c2w"]))
        for e in entries:
            o = objects[e["object_idx"]]
            color = PALETTE[e["object_idx"] % len(PALETTE)]
            pos = np.array([o["position"][k] for k in "xyz"])
            sc = np.array([o["scale"][k] for k in "xyz"])
            corners = pos - sc / 2 + corners_unit * sc
            p = corners @ w2c[:3, :3].T + w2c[:3, 3]
            z = np.maximum(p[:, 2], near_units)   # edges behind it are not drawn
            uu = fx * p[:, 0] / z + cx
            vv = fy * p[:, 1] / z + cy
            for a, b in edges:
                if p[a, 2] > near_units and p[b, 2] > near_units:
                    dr.line([(uu[a], vv[a]), (uu[b], vv[b])], fill=color, width=2)
            dr.text((float(np.clip(uu.min(), 2, w - 80)),
                     float(np.clip(vv.min() - 12, 2, h - 12))),
                    f"{obj_label(e, objects)} {e['score']:.2f}", fill=color)
        img.save(out / "debug" / f"frame_{fr['frame_idx']:04d}.png")

    manifest = dict(stage="export", n_objects=len(objects),
                    instancing=interactions["extended"]["export"]["instancing"],
                    held_association=interactions["extended"]["export"][
                        "held_association"],
                    manual_views_hash=stage1_manual_record(workdir)
                    .get("hash", "absent"),
                    exemplars_hash=stage2_exemplar_record(workdir)
                    .get("hash", "absent"),
                    upstream_stage3=upstream_fingerprint(
                        workdir, "stage3/manifest.json"),
                    params=params,
                    per_class={c: sum(1 for o in objects if o["label"] == c)
                               for c in classes if any(
                                   o["label"] == c for o in objects)},
                    elapsed_s=round(time.perf_counter() - t0, 1))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log.info("stage4 done: %d objects -> %s (%.1fs)",
             len(objects), out / "interactions.json", manifest["elapsed_s"])
    return manifest


def obj_label(entry: dict, objects: list[dict]) -> str:
    return objects[entry["object_idx"]]["label"]
