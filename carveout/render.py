# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 1: render synthetic views of a trained 3DGS scene with gsplat.

Outputs under <workdir>/stage1/: frames/frame_XXXX.png, depth/depth_XXXX.npy,
cameras.json (intrinsics + per-frame c2w + stats), manifest.json (config echo,
discard log, coverage report). Cached: re-runs are skipped when manifest.json
exists and nothing it depends on changed (the web re-runs it when the volume
or render gate was re-entered).
"""

import gc
import json
import math
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .cameras import (aimed_cameras, detect_scene_frame, ground_path,
                      interior_path, orbit_path)
from .sequencing import fingerprint
from .alignment import to_file_c2w
from .manual_views import load_manual_views, resolve_manual_views_path
from .refusal import Refusal, missing_dependency
from .scene_io import load_scene
from .projection import project_visible
from .units import resolve_scale, scene_units_cfg
from .quality import (_DISCARD_KNOBS, GATE_QUALITY_KEYS, discard_diagnosis,
                      format_discard_diagnosis, judged_stat,
                      resolve_auto_quality)
from .staging import (CAMERAS_PARTIAL, MANIFEST_PARTIAL, SHEET_PARTIAL,
                      STAGED_DESCRIPTOR, _health_line,
                      discard_staged_render, render_health)
from .volume import (WHOLE_SCENE, in_volume, load_volume,
                     resolve_volume_path, volume_scope, stale_volume_reason)

log = logging.getLogger(__name__)


def _intrinsics(cfg: dict) -> tuple[float, float, float, float, int, int]:
    w, h = cfg["render"]["width"], cfg["render"]["height"]
    f = 0.5 * w / np.tan(0.5 * np.deg2rad(cfg["render"]["fov_deg"]))
    return f, f, w / 2, h / 2, w, h


@torch.no_grad()
def _render_batch(scene_t: dict, c2ws: np.ndarray, K: np.ndarray, w: int, h: int,
                  cfg: dict):
    """Render a chunk of views; returns (rgb, alpha, depth, near_alpha) on CPU."""
    try:
        import gsplat
    except ImportError as e:
        raise missing_dependency("gsplat", e) from e

    device = scene_t["means"].device
    viewmats = torch.linalg.inv(
        torch.as_tensor(c2ws, dtype=torch.float32, device=device))
    Ks = torch.as_tensor(K, dtype=torch.float32, device=device).expand(len(c2ws), 3, 3)
    renders, alphas, _ = gsplat.rasterization(
        means=scene_t["means"], quats=scene_t["quats"], scales=scene_t["scales"],
        opacities=scene_t["opacities"], colors=scene_t["sh"],
        viewmats=viewmats, Ks=Ks, width=w, height=h,
        sh_degree=scene_t["sh_degree"],
        near_plane=cfg["render"]["near_plane"], far_plane=cfg["render"]["far_plane"],
        render_mode="RGB+ED",
    )
    rgb = renders[..., :3].clamp(0, 1).cpu()
    depth = renders[..., 3].cpu()

    # Near-field pass: alpha from Gaussians within near_field_depth only.
    # Expected depth is alpha-weighted and sees through low-opacity floater fog;
    # this pass measures it directly.
    _, near_alpha, _ = gsplat.rasterization(
        means=scene_t["means"], quats=scene_t["quats"], scales=scene_t["scales"],
        opacities=scene_t["opacities"], colors=scene_t["sh"][:, :1],
        viewmats=viewmats, Ks=Ks, width=w, height=h, sh_degree=0,
        near_plane=cfg["render"]["near_plane"],
        far_plane=cfg["render"]["quality_filter"]["near_field_depth"],
    )
    return rgb, alphas[..., 0].cpu(), depth, near_alpha[..., 0].cpu()


def _quality(rgb: torch.Tensor, alpha: torch.Tensor, depth: torch.Tensor,
             near_alpha: torch.Tensor, qcfg: dict) -> tuple[bool, dict]:
    """Cheap degenerate-view heuristics (the stage-1 view-quality filter)."""
    coverage = float((alpha > qcfg["alpha_thresh"]).float().mean())
    valid = alpha > qcfg["alpha_thresh"]
    median_depth = float(depth[valid].median()) if valid.any() else 0.0
    rgb_std = float(rgb.std())
    near = float(near_alpha.mean())
    gray = rgb.mean(dim=-1)
    sharpness = float(
        gray.diff(n=2, dim=0).abs().mean() + gray.diff(n=2, dim=1).abs().mean())
    stats = dict(coverage=round(coverage, 4), median_depth=round(median_depth, 3),
                 rgb_std=round(rgb_std, 4), near_alpha=round(near, 4),
                 sharpness=round(sharpness, 5))
    if coverage < qcfg["min_coverage"]:
        return False, {**stats, "reason": "low_coverage"}
    if median_depth < qcfg["min_median_depth"]:
        return False, {**stats, "reason": "inside_geometry"}
    if rgb_std < qcfg["min_rgb_std"]:
        return False, {**stats, "reason": "uniform_frame"}
    if near > qcfg["max_near_alpha"]:
        return False, {**stats, "reason": "near_field_fog"}
    if sharpness < qcfg["min_sharpness"]:
        return False, {**stats, "reason": "blurry_mush"}
    return True, stats


def _pose_similar(c2w_a: np.ndarray, c2w_b: np.ndarray, min_pos: float,
                  cos_min: float) -> bool:
    if np.dot(c2w_a[:3, 2], c2w_b[:3, 2]) < cos_min:
        return False
    return bool(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]) < min_pos)


def _diversity_prune(c2ws: list[np.ndarray], accepted: list[np.ndarray],
                     dcfg: dict) -> list[np.ndarray]:
    """Drop candidates too close in pose to an accepted view or to each other —
    near-duplicate frames add cost without adding coverage (gate feedback)."""
    cos_min = np.cos(np.deg2rad(dcfg["min_angle_deg"]))
    out = []
    for c in c2ws:
        pool = accepted + out
        if not any(_pose_similar(c, k, dcfg["min_pos_sep"], cos_min) for k in pool):
            out.append(c)
    return out


@torch.no_grad()
def _visibility_counts(scene_t: dict, vol_mask_t: torch.Tensor, kept: list[dict],
                       depth_dir: Path, K: np.ndarray, w: int, h: int,
                       near: float, tol: float, tol_rel: float) -> torch.Tensor:
    """Per-Gaussian count of kept frames that observe it (depth-tested).
    `near`, `tol` are scene units — the resolved config's."""
    device = scene_t["means"].device
    pts = scene_t["means"][vol_mask_t]
    counts = torch.zeros(len(pts), dtype=torch.int16, device=device)
    glob = dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                cx=float(K[0, 2]), cy=float(K[1, 2]), width=w, height=h)
    for fr in kept:
        depth = torch.as_tensor(
            np.load(depth_dir / f"depth_{fr['frame_idx']:04d}.npy").astype(np.float32),
            device=device)
        proj = project_visible(pts, fr, glob, depth, near=near,
                               tol_rel=tol_rel, tol_abs=tol)
        counts[proj.vis] += 1
    return counts


@torch.no_grad()
def _select_views(scene_t: dict, vol_mask_t: torch.Tensor, kept: list[dict],
                  n_exempt: int, n_views: int, frames_dir: Path,
                  depth_dir: Path, K: np.ndarray, w: int, h: int,
                  near: float, tol: float, tol_rel: float, k: int) -> int:
    """Greedy set cover over the rendered pool: each pick is the
    candidate that newly covers the most of the volume at K views given
    what is already chosen (sharpness breaks ties), until `n_views` are
    chosen. Order-dependent on purpose — a static score would keep ten
    near-duplicates of the best-seen corner together. The first `n_exempt`
    entries (the manual views, then the near-field views) are exempt and
    stay first. Renumbers the chosen frames in pick order (the
    contact sheet then reads best first), deletes the rest from staging,
    records `rank` and `gain` (the fraction of the volume the pick newly
    covered) on each entry, and returns how many rendered candidates were
    not selected. Mutates `kept` in place."""
    auto = kept[n_exempt:]
    if len(auto) <= n_views:
        return 0
    device = scene_t["means"].device
    pts = scene_t["means"][vol_mask_t]
    # bound the pool x points matrix (uint8) to ~400 MB by subsampling
    stride = max(1, int(math.ceil(len(auto) * len(pts) / 4e8)))
    pts = pts[::stride]
    glob = dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                cx=float(K[0, 2]), cy=float(K[1, 2]), width=w, height=h)
    masks = torch.zeros((len(auto), len(pts)), dtype=torch.uint8, device=device)
    for i, fr in enumerate(auto):
        depth = torch.as_tensor(
            np.load(depth_dir / f"depth_{fr['frame_idx']:04d}.npy").astype(np.float32),
            device=device)
        proj = project_visible(pts, fr, glob, depth, near=near,
                               tol_rel=tol_rel, tol_abs=tol)
        masks[i, proj.vis] = 1
    sharp = torch.tensor([float(fr.get("sharpness") or 0) for fr in auto],
                         device=device)
    sharp = sharp / (sharp.max() + 1e-9) * 1e-3     # tie-break only
    seen = masks.sum(1).float() / max(len(pts), 1)   # each view's own share
    counts = torch.zeros(len(pts), dtype=torch.int16, device=device)
    remaining = torch.ones(len(auto), dtype=torch.bool, device=device)
    order: list[tuple[int, float]] = []
    n = float(len(pts))
    for _ in range(min(n_views, len(auto))):
        need = (counts == k - 1).to(torch.uint8)      # one view from covered
        below = (counts < k - 1).to(torch.uint8)      # further from covered
        gain_k = (masks * need).sum(1).float()
        gain = gain_k + 0.5 * (masks * below).sum(1).float() + sharp
        gain[~remaining] = -1
        i = int(torch.argmax(gain))
        order.append((i, float(gain_k[i]) / n))
        remaining[i] = False
        counts += masks[i].to(torch.int16)
    del masks
    # renumber in pick order; the unchosen leave staging
    chosen = {i for i, _ in order}
    for fr in auto:
        o = fr["frame_idx"]
        (frames_dir / f"frame_{o:04d}.png").rename(frames_dir / f"tmp_{o:04d}.png")
        (depth_dir / f"depth_{o:04d}.npy").rename(depth_dir / f"tmp_{o:04d}.npy")
    new_auto = []
    for rank, (i, g) in enumerate(order):
        fr = dict(auto[i]); o = fr["frame_idx"]; fidx = n_exempt + rank
        (frames_dir / f"tmp_{o:04d}.png").rename(frames_dir / f"frame_{fidx:04d}.png")
        (depth_dir / f"tmp_{o:04d}.npy").rename(depth_dir / f"depth_{fidx:04d}.npy")
        fr.update(frame_idx=fidx, file=f"frames/frame_{fidx:04d}.png",
                  rank=rank, gain=round(g, 4), seen=round(float(seen[i]), 4))
        new_auto.append(fr)
    for i, fr in enumerate(auto):
        if i not in chosen:
            o = fr["frame_idx"]
            (frames_dir / f"tmp_{o:04d}.png").unlink(missing_ok=True)
            (depth_dir / f"tmp_{o:04d}.npy").unlink(missing_ok=True)
    not_selected = len(auto) - len(new_auto)
    kept[n_exempt:] = new_auto
    log.info("ranked %d candidates by coverage gain: kept %d, %d not "
             "selected; first pick covers %.1f%% of the volume, last %.2f%%",
             len(auto), len(new_auto), not_selected,
             100 * order[0][1], 100 * order[-1][1])
    return not_selected


def run_render(scene_path: str, workdir: str, cfg: dict,
               force: bool = False, should_cancel=None) -> dict:
    from .sequencing import check_cancel
    out = Path(workdir) / "stage1"
    manifest_path = out / "manifest.json"
    mv_path = resolve_manual_views_path(workdir, cfg)
    mv_hash = fingerprint(mv_path)
    # The volume places and scores every camera, so it is part of the
    # cache key like the manual views are: a render made under another
    # volume is stale, whatever this session remembers about which gates
    # were re-entered (a server restart remembers nothing).
    vol_path = resolve_volume_path(workdir, cfg)
    vol_hash = fingerprint(vol_path)
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        old_hash = (cached.get("manual_views") or {}).get("hash", "absent")
        # The factor shapes every placement distance, so a manifest rendered
        # at another factor is stale: the record says what it ran at,
        # and the profile (recorded or measured) says what it would run at
        # now. A manifest without the record predates it.
        old = cached.get("scale")
        stale_scale = None
        if not old:
            stale_scale = "the render predates the scale record"
        else:
            now = resolve_scale(cfg)
            if (now.source != old.get("source")
                    or not math.isclose(now.value, float(old["scale_m_per_unit"]),
                                        rel_tol=1e-9)):
                stale_scale = (f"the scale changed ({old.get('source')} "
                               f"{float(old['scale_m_per_unit']):.4g} -> "
                               f"{now.describe()})")
        # The levelling moves every placement too: a manifest made
        # under another alignment is stale.
        from .alignment import alignment_block
        try:
            now_align = alignment_block(cfg["scene"].get("alignment"))
        except ValueError:
            now_align = "malformed"
        was_align = (cached.get("scene_frame") or {}).get("alignment")
        if stale_scale is None and now_align != was_align:
            stale_scale = "the scene's levelling changed"
        old_vol = cached.get("volume_hash")
        if stale_scale is None and old_vol is None:
            stale_scale = "the render predates the volume record"
        elif stale_scale is None and old_vol != vol_hash:
            stale_scale = "the volume changed"
        if old_hash == mv_hash and stale_scale is None:
            log.info("stage1 cached at %s (inputs unchanged; Re-render on "
                     "the render panel forces a new one)", out)
            return cached
        if old_hash != mv_hash:
            log.warning(
                "stage1 cache INVALIDATED: manual_views.json content changed "
                "(%s -> %s); re-rendering stage1 from scratch",
                old_hash[:12], mv_hash[:12])
        else:
            log.warning(
                "stage1 cache INVALIDATED: %s, which moves every placement "
                "distance; re-rendering stage1 from scratch", stale_scale)

    t_start = time.perf_counter()
    manual = load_manual_views(mv_path, cfg)   # fail fast on a malformed file
    if cfg["render"]["path_mode"] == "manual" and not manual:
        # Nothing to render: the manual set IS the camera set. Said before
        # the scene loads, in the words of the act that fixes it.
        raise Refusal(
            "no viewpoints saved yet; with 'manual' the render draws exactly "
            "the viewpoints you saved, and there are none. On the render panel, "
            "Capture view saves the canvas camera as a viewpoint: save at least "
            "one (several, from different sides, for each object you want), "
            "then render.", gate="render")
    scene = load_scene(scene_path, cfg)
    fx, fy, cx, cy, w, h = _intrinsics(cfg)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # --- volume of interest ---------------------------------------------------
    boxes = None
    scope = volume_scope(vol_path)
    if scope == "boxes":
        stale = stale_volume_reason(vol_path, cfg)
        if stale:
            raise Refusal(stale, gate="volume")
        boxes = load_volume(vol_path)
    elif scope == WHOLE_SCENE:
        log.info("volume scope: the whole scene; cameras over the scene's "
                 "robust bounds, coverage over every opaque Gaussian in them")
    else:
        log.warning("no volume.json at %s: placement and coverage are unbounded; "
                    "press Propose at the volume gate to create one", vol_path)

    # Floor prior restricts to in-volume Gaussians when a volume exists; up-axis extents stay full-scene — a trimmed footprint
    # narrower than the ceiling height flips the heuristic.
    floor_mask = in_volume(scene.means, boxes) if boxes is not None else None
    frame = detect_scene_frame(scene.means, scene.opacities, cfg,
                               floor_mask=floor_mask)
    frame.alignment = scene.alignment   # placement in the scene frame
    # Length knobs -> scene units, once, HERE: the frame (and so the one
    # factor, recorded or measured) is known and nothing has read a
    # distance yet. `cfg_m` stays the profile's metres for the manifest.
    cfg_m = cfg
    cfg, units = scene_units_cfg(cfg, frame.scale)
    # Resolve "auto" distance thresholds HERE too: the scene's size is known
    # for the first time, and nothing has rasterised yet — the near-field
    # pass reads near_field_depth as its far plane, so a late resolution
    # would be a silent no-op on the very knob it was asked to set.
    cfg, auto_quality = resolve_auto_quality(cfg, frame)

    # --- camera poses ---------------------------------------------------------
    n_views = cfg["render"]["num_views"]
    free_world = None
    if cfg["render"]["path_mode"] == "manual":
        # Fully manual: the camera set IS manual_views.json, and NOTHING is
        # sampled. The manual views are ingested below on the path they always
        # took (additive, before the candidate pass), so an empty candidate
        # list is all this mode has to say — no free-space map, no diversity
        # prune, no quality attrition. The render is then exactly the views on
        # disk, which is what makes a hand-adjusted camera stay where it was
        # put. The occupancy grid is deliberately not built: nothing samples.
        path_mode = "manual"
        candidates = []
        n_near = 0
    else:
        path_mode = cfg["render"]["path_mode"]
        if path_mode == "auto":
            # No filming answer (only a hand-written profile carries
            # `auto`: the creation dialog requires the tile). Carveout
            # never chooses the camera path — one sentence, to the render
            # panel where the answer lives.
            raise Refusal("no camera path chosen for this scene; pick how "
                          "it was filmed on the render panel (inside a space "
                          "/ around a subject / outdoors on the ground).",
                          gate="render")
        n_candidates = int(n_views * cfg["render"]["oversample"])
        n_near = 0   # near-field poses lead the interior list; exempt below
        if path_mode == "interior":
            candidates, free_world, n_near = interior_path(
                scene.means, scene.opacities, frame, cfg, n_candidates,
                volume_boxes=boxes, should_cancel=should_cancel)
        elif path_mode == "ground":
            candidates, free_world = ground_path(
                scene.means, scene.opacities, frame, cfg, n_candidates,
                volume_boxes=boxes)
        elif path_mode == "orbit":
            candidates = orbit_path(scene.means, scene.opacities, frame, cfg,
                                    n_candidates, volume_boxes=boxes)
        else:
            raise ValueError(f"unknown path_mode {path_mode!r}")
        # planned in the scene frame; rendered, compared and recorded in
        # the file's — one conversion, here
        candidates = [to_file_c2w(c, scene.alignment) for c in candidates]

    # the prune keeps order and drops entries, so the near-field poses that
    # survive it are still the first ones: count them by identity
    near_ids = {id(c) for c in candidates[:n_near]}
    candidates = _diversity_prune(candidates, [], cfg["render"]["diversity"])
    n_near = sum(id(c) in near_ids for c in candidates)
    log.info("%d diverse candidates after pose dedup (%d near-field)",
             len(candidates), n_near)

    # --- render + filter ------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering (gsplat)")
    device = "cuda"
    scene_t = dict(
        # the file's own arrays: what renders, under file-frame poses
        means=torch.as_tensor(scene.means_file, device=device),
        quats=torch.as_tensor(scene.quats_file, device=device),
        scales=torch.as_tensor(scene.scales, device=device),
        opacities=torch.as_tensor(scene.opacities, device=device),
        sh=torch.as_tensor(scene.sh, device=device),
        sh_degree=min(scene.sh_degree, cfg["render"]["sh_degree"]),
    )

    # Render into STAGING dirs and swap them in only once this render is known
    # good. Wiping the live ones up-front (what this used to do) destroyed the
    # previous render before knowing the new one would produce anything: a
    # threshold experiment that rejected every view left zero frames on disk
    # while cameras.json and manifest.json — written further down, so never
    # updated on that path — still described the OLD render. The contact sheet
    # then listed frames that no longer existed and the gate could still be
    # approved. Same hazard on cancellation. A re-render with fewer kept views
    # must still not leave stale frames behind, which the swap guarantees.
    live_frames, live_depth = out / "frames", out / "depth"
    frames_dir, depth_dir = out / "frames.partial", out / "depth.partial"
    # A staged render from an earlier attempt is superseded the moment this one
    # starts writing frames, so it goes FIRST — before the dirs below exist, or
    # it would delete them. STAGED_DESCRIPTOR must never outlive the frames it
    # describes. Also clears leftovers from an aborted attempt.
    discard_staged_render(workdir, quiet=True)
    for d in (frames_dir, depth_dir):
        if d.exists():
            shutil.rmtree(d)   # leftovers from an aborted earlier attempt
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    def commit_frames() -> None:
        """Swap staging over the live frames — the point of no return, called
        only once this render has views and is about to publish cameras.json."""
        for live, staged in ((live_frames, frames_dir),
                             (live_depth, depth_dir)):
            if live.exists():
                shutil.rmtree(live)
            staged.rename(live)

    chunk = max(int(cfg["render"].get("views_in_flight") or 2), 1)
    kept: list[dict] = []
    kept_poses: list[np.ndarray] = []
    discarded: list[dict] = []

    # --- manual views (viewer-captured; ADDITIVE, hand-picked views bypass the
    # quality filter and diversity prune — user judgment overrides heuristics;
    # a would-have-failed warning is logged). Per-view intrinsics: captured
    # vertical fov/aspect at detection resolution (H-anchored).
    for mv in manual:
        K_mv = np.array([[mv["fx"], 0, mv["cx"]],
                         [0, mv["fy"], mv["cy"]], [0, 0, 1]])
        rgb, alpha, depth, near_alpha = _render_batch(
            scene_t, mv["c2w"][None], K_mv, mv["width"], mv["height"], cfg)
        ok, stats = _quality(rgb[0], alpha[0], depth[0], near_alpha[0],
                             cfg["render"]["quality_filter"])
        if not ok:
            # the numbers and what they mean, not the reason's name
            # alone — a close-up standing inside the near-field horizon
            # reads real geometry as fog, and the operator should know it
            # was the frame's distance, not its quality, that tripped it.
            reason = stats.pop("reason")
            stat, knob, side, _ = _DISCARD_KNOBS.get(reason, (None,) * 4)
            qcfg = cfg["render"]["quality_filter"]
            detail = ""
            if stat:
                detail = (f": {stat} {stats.get(stat)} against {knob} "
                          f"{qcfg.get(knob)} ({'the floor' if side == 'above' else 'the cap'})")
            if reason == "near_field_fog":
                hz = float(qcfg["near_field_depth"])
                md = stats.get("median_depth")
                metres = f" ≈ {hz * frame.scale:.2f} m" if frame.scale else ""
                detail += (f"; the near-field horizon is {hz:.3g} units{metres} "
                           f"and this view's median depth {md} units")
                if md is not None and md < hz * 1.5:
                    detail += (": a close-up, with real geometry inside the "
                               "horizon; distance tripped it, not quality")
            log.warning("manual view %r would have FAILED the quality filter "
                        "(%s%s); kept anyway (manual views bypass it)",
                        mv["label"], reason, detail)
        fidx = len(kept)
        fname = f"frame_{fidx:04d}.png"
        Image.fromarray(
            (rgb[0].numpy() * 255).astype(np.uint8)).save(frames_dir / fname)
        np.save(depth_dir / f"depth_{fidx:04d}.npy",
                depth[0].numpy().astype(np.float32))
        kept.append({"frame_idx": fidx, "file": f"frames/{fname}",
                     "round": "manual", "provenance": f"manual:{mv['id']}",
                     "label": mv["label"],
                     "fx": mv["fx"], "fy": mv["fy"],
                     "cx": mv["cx"], "cy": mv["cy"],
                     "width": mv["width"], "height": mv["height"],
                     "c2w": mv["c2w"].tolist(), **stats})
        kept_poses.append(mv["c2w"])
    n_manual = len(kept)

    def render_and_keep(c2w_list: list[np.ndarray], round_tag: str,
                        budget: int) -> int:
        """Render candidates in order and keep the ones that pass, at most
        `budget` of them — the budget is PER ROUND: `num_views` bounds the
        initial round and `coverage.round_views` each repair round, so a
        render that filled its budget can still be repaired (the old
        global cap made repair unreachable exactly when the initial round
        succeeded). Manual views are additive and never count. Returns the
        number kept by this call."""
        # An empty round is legitimate, not a caller error: fully-manual mode
        # produces no candidates at all, and a coverage round can come back
        # with nothing viable. np.stack refuses an empty list, so the guard
        # belongs here rather than at each call site.
        if not c2w_list or budget <= 0:
            return 0
        added = 0
        c2ws = np.stack(c2w_list)
        for i0 in range(0, len(c2ws), chunk):
            check_cancel(should_cancel)   # between GPU batches
            rgb, alpha, depth, near_alpha = _render_batch(
                scene_t, c2ws[i0:i0 + chunk], K, w, h, cfg)
            for j in range(len(rgb)):
                if added >= budget:
                    return added
                ok, stats = _quality(rgb[j], alpha[j], depth[j], near_alpha[j],
                                     cfg["render"]["quality_filter"])
                if not ok:
                    discarded.append({"round": round_tag, **stats})
                    continue
                fidx = len(kept)
                fname = f"frame_{fidx:04d}.png"
                Image.fromarray(
                    (rgb[j].numpy() * 255).astype(np.uint8)).save(frames_dir / fname)
                np.save(depth_dir / f"depth_{fidx:04d}.npy",
                        depth[j].numpy().astype(np.float32))
                kept.append({"frame_idx": fidx, "file": f"frames/{fname}",
                             "round": round_tag, "provenance": "auto",
                             "c2w": c2ws[i0 + j].tolist(),
                             **stats})
                kept_poses.append(c2ws[i0 + j])
                added += 1
        return added

    ccfg = cfg["render"]["coverage"]
    # The coverage region: the boxes, or under the whole-scene act the
    # scene's robust bounds (the same region placement used). Needed
    # before the first round now: the selection below ranks by it.
    cover_boxes = boxes if boxes is not None else (
        [dict(name="scene", min=frame.lo, max=frame.hi)]
        if scope == WHOLE_SCENE else None)
    vol_mask = vol_mask_t = None
    if cover_boxes is not None:
        vol_mask = in_volume(scene.means, cover_boxes) & (
            scene.opacities >= ccfg["min_opacity"])
        vol_mask_t = torch.as_tensor(vol_mask, device=device)

    # The first round renders a POOL of `candidate_factor` x the budget
    # (every candidate that passes the filter, first-come), then keeps the
    # budget by greedy coverage gain. Without a coverage region there
    # is nothing to rank by, so the pool is the budget, as before.
    cand_factor = float(cfg["render"].get("candidate_factor") or 1)
    pool = (int(round(n_views * cand_factor))
            if vol_mask_t is not None and cand_factor > 1 else n_views)
    # The near-field views first, under their own round, outside the pool:
    # like the manual views they are placed for what they frame, which the
    # coverage gain below does not measure (a desktop is a sliver of a
    # room's Gaussians, so they ranked last and survived only when the
    # budget had room for the tail). Kept whatever the budget, never
    # ranked, first on the sheet after the manual views.
    n_near_kept = render_and_keep(candidates[:n_near], "near_field", len(candidates))
    render_and_keep(candidates[n_near:], "initial", pool)
    n_exempt = n_manual + n_near_kept
    log.info("initial round: %d rendered (%d manual, %d near-field) / %d discarded",
             len(kept), n_manual, n_near_kept, len(discarded))
    not_selected = 0
    if vol_mask_t is not None and len(kept) - n_exempt > n_views:
        check_cancel(should_cancel)
        not_selected = _select_views(
            scene_t, vol_mask_t, kept, n_exempt, n_views, frames_dir,
            depth_dir, K, w, h, cfg["render"]["near_plane"],
            ccfg["depth_tol"], ccfg["depth_tol_rel"], ccfg["k"])
        kept_poses[:] = [np.asarray(fr["c2w"]) for fr in kept]

    # --- coverage metric + gap-driven extra rounds -----------------------------
    # Two capabilities, two conditions. The METRIC needs only the VOLUME: it
    # asks how much of it the kept views actually see. The gap-driven REPAIR
    # rounds additionally need the FREE-SPACE map, to put a repair camera
    # somewhere legal. They were once a single `and`, so a path mode that
    # builds no free-space map (orbit) reported no coverage at all —
    # and `render_health` treats absent coverage as "no volume", so the render
    # gate silently degraded to a raw view count on exactly the scenes whose
    # framing is hardest to judge by eye.
    coverage_report = None
    if cover_boxes is not None:

        def coverage_frac() -> tuple[float, torch.Tensor]:
            counts = _visibility_counts(scene_t, vol_mask_t, kept, depth_dir,
                                        K, w, h, cfg["render"]["near_plane"],
                                        ccfg["depth_tol"], ccfg["depth_tol_rel"])
            return float((counts >= ccfg["k"]).float().mean()), counts

        frac, counts = coverage_frac()
        frac_initial = frac
        rounds = 0
        repaired = 0
        round_views = int(ccfg.get("round_views") or 12)
        if free_world is None:
            log.info("coverage: measured, but gap repair is unavailable on the "
                     "%s path (it builds no free-space map); a low number "
                     "here is read, not repaired", path_mode)
        # No view-count clause here: the initial round's budget is not a
        # ceiling on repair (each repair round has its own, `round_views`).
        while (free_world is not None
               and frac < ccfg["target_frac"] and rounds < ccfg["max_extra_rounds"]):
            check_cancel(should_cancel)   # between coverage rounds
            rounds += 1
            uncovered = scene_t["means"][vol_mask_t][counts < ccfg["k"]].cpu().numpy()
            if scene.alignment is not None:
                uncovered = uncovered @ scene.alignment.T   # file -> scene frame
            # Cluster gaps into coarse voxels; aim new candidates at the largest.
            vox = ccfg["gap_voxel"]
            keys, inv = np.unique((uncovered / vox).astype(int), axis=0,
                                  return_inverse=True)
            sizes = np.bincount(inv)
            order = np.argsort(sizes)[::-1][:ccfg["targets_per_round"]]
            targets = keys[order] * vox + vox / 2
            extra = aimed_cameras(
                targets, free_world, frame, cfg,
                min_dist=(max(cfg["render"]["ground"]["min_standoff"],
                              cfg["render"]["coverage"]["min_aim_dist"])
                          if path_mode == "ground" else None))
            extra = [to_file_c2w(c, scene.alignment) for c in extra]
            extra = _diversity_prune(extra, kept_poses, cfg["render"]["diversity"])
            if not extra:
                log.warning("coverage round %d: no viable extra candidates", rounds)
                break
            log.info("coverage round %d: %.1f%% covered, aiming %d extra candidates",
                     rounds, 100 * frac, len(extra))
            before = frac
            got = render_and_keep(extra, f"coverage_{rounds}", round_views)
            repaired += got
            frac, counts = coverage_frac()
            log.info("coverage round %d: %.1f%% -> %.1f%% (+%d views)",
                     rounds, 100 * before, 100 * frac, got)

        coverage_report = dict(
            k=ccfg["k"], min_opacity=ccfg["min_opacity"],
            in_volume_gaussians=int(vol_mask.sum()),
            frac_initial=round(frac_initial, 4), frac_final=round(frac, 4),
            extra_rounds=rounds, views_repaired=repaired,
            frac_by_k={str(kk): round(float(
                (counts >= kk).float().mean()), 4) for kk in (1, 2, 3)},
        )
        log.info("coverage (K=%d): %.1f%% of %d in-volume gaussians (started %.1f%%)",
                 ccfg["k"], 100 * frac, int(vol_mask.sum()), 100 * frac_initial)

    # The last GPU pass is behind us: release the splat tensors NOW, before
    # the contact sheet, the staging decision and any refusal below, the way
    # the probe renderer does (`_render_probes`). Python drops the objects
    # either way; what this buys is the allocator giving the blocks back, so
    # the stage after this one (the VLM at the vocabulary gate) measures a
    # free card instead of a reserved pool it cannot see.
    del scene_t, vol_mask_t
    gc.collect()
    torch.cuda.empty_cache()

    # Not in manual mode: zero auto views is the POINT there, and reporting it
    # as an honest ceiling reads as a failure the operator should act on.
    if path_mode != "manual" and len(kept) - n_exempt < n_views:
        log.warning("only %d/%d auto views survived (honest ceiling for this "
                    "scene at current thresholds)", len(kept) - n_exempt, n_views)
    if not kept:
        # The discard log is the only evidence for what to change, and it
        # used to die with the exception: manifest.json is written further
        # down, so on this path it either did not exist or still described
        # an EARLIER attempt — the operator was tuning against numbers from
        # a different render. Persist it under its own name; manifest.json
        # stays the "stage completed" marker, so writing one here would be a
        # cache false-hit on the next run.
        qcfg = cfg["render"]["quality_filter"]
        rows = discard_diagnosis(
            discarded, qcfg, scale=frame.scale, source=frame.scale_source,
            measured=(frame.notes.get("scale") or {}).get("measured"))
        report = out / "discard_report.json"
        out.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(frames_dir, ignore_errors=True)   # nothing published:
        shutil.rmtree(depth_dir, ignore_errors=True)    # the old render stands
        report.write_text(json.dumps(dict(
            version=1, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            views_attempted=len(discarded), views_kept=0,
            # `thresholds` are what the filter compared against — scene
            # units, like every per-view stat below; `thresholds_profile`
            # are the same knobs as the profile (and the panel) write them.
            thresholds=dict(qcfg),
            thresholds_profile=dict(cfg_m["render"]["quality_filter"]),
            scale_m_per_unit=frame.scale, scale_source=frame.scale_source,
            summary=rows, discards=discarded),
            indent=1))
        # A distance threshold larger than the whole scene is not a
        # threshold to tune: the factor is wrong, and the cause belongs to
        # the volume gate (a 72x scene recorded as 0.5 m tall put
        # min_median_depth at 393 units on a 207-unit scene, and the render
        # gate's hint sent the operator after camera placement).
        span = float(np.max(frame.hi - frame.lo))
        too_big = [(k, float(qcfg[k])) for k in
                   ("min_median_depth", "near_field_depth")
                   if isinstance(qcfg.get(k), (int, float)) and qcfg[k] > span]
        if too_big:
            src = getattr(frame, "scale_source", "recorded")
            named = ", ".join(f"render.quality_filter.{k} = {v:.4g} scene "
                              f"units" for k, v in too_big)
            rec = frame.notes.get("scale") or {}
            how = (f"measured: {rec.get('measured', {}).get('label') or 'one length'}, "
                   f"{float(rec.get('measured', {}).get('length_m', 0)):g} m"
                   if src == "measured" else "recorded")
            reads = (f"At {frame.scale:.4g} m per unit ({how}) the scene "
                     f"reads {span * frame.scale:.3g} m across, which no "
                     f"capture of a room, an object or a site is.")
            fix = ("measure again on the canvas (the two points, or the "
                   "length you typed), or correct the factor"
                   if src == "measured" else
                   "correct the factor, or measure one thing you know on "
                   "the canvas")
            raise Refusal(
                f"all {len(discarded)} candidate view(s) were discarded, and "
                f"the cause is the SCENE'S SCALE, not a threshold: {named}, "
                f"larger than the whole scene ({span:.4g} units across). "
                f"{reads}\n  At the volume gate, {fix}, then render again."
                f"\nfull per-view stats: {report}",
                gate="volume")
        # Refusal, not RuntimeError: stage refusals are surfaced verbatim
        # to the operator, where an unexpected exception prints a traceback
        # that reads like a crash and stops the run for good.
        raise Refusal(
            format_discard_diagnosis(rows, len(discarded))
            + f"\nfull per-view stats: {report}"
            + "\nadjust ONE threshold in the render gate's calibration "
              "panel, then re-render; the run resumes here.")

    # Render-review artifact (NOT optional): labeled
    # contact sheet of every kept frame — previously built ad hoc per scene,
    # now part of the stage. Yellow index labels match the earlier gates.
    # Built from the STAGED frames, BEFORE any swap: the same code then serves
    # both outcomes, and on the staged path there is no live dir to read from.
    from PIL import ImageDraw
    tile, cols = 224, 7
    rows_n = (len(kept) + cols - 1) // cols
    sheet = Image.new("RGB", (tile * cols, tile * rows_n), (13, 17, 23))
    dr = ImageDraw.Draw(sheet)
    for i, fr in enumerate(kept):
        im = Image.open(frames_dir / Path(fr["file"]).name).resize((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * tile
        sheet.paste(im, (x, y))
        tag = f"{fr['frame_idx']:04d}" + (
            f" M {fr['label']}" if fr["round"] == "manual"
            else " N" if fr["round"] == "near_field" else "")
        dr.text((x + 4, y + 2), tag, fill=(255, 230, 0))
    sheet.save(out / SHEET_PARTIAL)

    cameras = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, frames=kept)

    manifest = dict(
        stage="render", scene=str(scene_path), num_gaussians=scene.num_gaussians,
        path_mode=path_mode,
        volume=str(vol_path) if scope is not None else None,
        volume_hash=vol_hash,
        volume_scope=scope,
        scene_frame=dict(up_axis="xyz"[frame.up_axis],
                         up_sign=frame.up_sign, floor=frame.floor,
                         alignment=frame.alignment_block,
                         notes=frame.notes),
        views_kept=len(kept), views_manual=n_manual,
        views_near_field=n_near_kept,
        views_discarded=len(discarded),
        views_not_selected=not_selected, candidate_factor=cand_factor,
        # Thresholds the scene's size decided rather than the config: recorded
        # so a later reader can tell a discovered number from a set one.
        auto_quality=auto_quality or None,
        manual_views=dict(path=str(mv_path), hash=mv_hash, n_views=n_manual),
        coverage=coverage_report,
        discards=discarded,
        elapsed_s=round(time.perf_counter() - t_start, 2),
        # `config` is the profile's METRES; `units` is what this render
        # actually used, per key, at the one factor `scale` records —
        # recorded, or measured with the ruler (the stages after this one
        # read it from here). The discard stats above are in scene units.
        config=cfg_m["render"],
        scale=frame.notes["scale"],
        units=units,
    )
    manifest["health"] = render_health(manifest, cfg)

    # --- publish, or stage for an explicit operator act ------------------------
    # manifest_path still holds the PREVIOUS render's manifest at this point —
    # it is written below, after the swap — so the comparison needs no extra
    # bookkeeping: the render about to be destroyed is still describing itself.
    # REGRESSION, not a quality bar: staged only when this render is unhealthy
    # AND the one it would replace was healthy. A first render, or a scene where
    # both are unhealthy, publishes exactly as before.
    new_h = manifest["health"]
    prev_h = (render_health(json.loads(manifest_path.read_text()), cfg)
              if manifest_path.exists() else None)
    if prev_h and prev_h["ok"] and not new_h["ok"]:
        (out / CAMERAS_PARTIAL).write_text(json.dumps(cameras, indent=1))
        (out / MANIFEST_PARTIAL).write_text(json.dumps(manifest, indent=1))
        (out / STAGED_DESCRIPTOR).write_text(json.dumps(dict(
            version=1, staged_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            reason="worse_than_published", staged=new_h, published=prev_h),
            indent=1))
        log.warning("stage1 STAGED, not published: staged %s vs published %s",
                    _health_line(new_h), _health_line(prev_h))
        # Refusal for the same reason the discard branch uses it: this is a
        # refusal to publish, not a crash.
        raise Refusal(
            "render STAGED, not published: it came out worse than the render "
            "already on disk.\n"
            f"  staged    : {_health_line(new_h)}\n"
            f"  published : {_health_line(prev_h)}\n"
            f"  healthy   : >= {new_h['min_views']} views kept and coverage "
            f">= {100 * new_h['min_coverage']:.0f}%\n"
            "Your published render is UNTOUCHED. Either:\n"
            "  - keep it: adjust ONE threshold in the render gate's "
            "calibration panel and re-render (which discards the staged one), "
            "or\n"
            "  - publish the staged render anyway: the render gate's "
            "'publish staged' action.\n"
            f"staged frames: {frames_dir}")

    commit_frames()   # this render wins; the previous one is superseded now
    # A refusal's diagnosis describes the attempt that produced it, so once a
    # render publishes it is answering a question nobody is asking any more —
    # under thresholds that have since moved. Nothing reads the file, so the
    # cost is interpretive and lands on whoever opens a SUCCESSFUL workdir and
    # finds a "discard report" in it. Unlinking here makes its presence mean
    # one thing: the most recent render attempt was refused. The STAGED path
    # deliberately keeps its report — that render did not publish.
    (out / "discard_report.json").unlink(missing_ok=True)
    (out / SHEET_PARTIAL).rename(out / "review_gate.png")
    log.info("render-gate contact sheet -> %s", out / "review_gate.png")
    (out / "cameras.json").write_text(json.dumps(cameras, indent=1))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log.info("stage1 done: %d views kept, %d discarded, %.1fs -> %s",
             len(kept), len(discarded), manifest["elapsed_s"], out)
    return manifest


def _aimed_pose(frame, boxes, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The one probe aimed at the measured segment: the camera in the
    horizontal plane (the orbit's up-vector plane) through the segment's
    midpoint, at a stand-off of three times the segment's length, looking
    at the midpoint — the segment then fills about a third of the frame.
    The viewing direction is perpendicular to the segment (a door's height
    seen broadside, not end-on) and on the volume's side of it, so a
    segment on a wall is seen from inside the room. The stand-off is
    clamped to the scene's robust bounds (a deviation from the design's
    flat 3x, stated in the commit): three times a bench in a nook put the
    camera outside the capture, looking at the back of the splat, and the
    frame was mostly black."""
    from .cameras import look_at
    up = frame.up_vec
    mid = (a + b) / 2.0
    d = b - a
    length = float(np.linalg.norm(d)) or 1e-6
    if boxes:
        lo = np.min([np.asarray(bx["min"], float) for bx in boxes], axis=0)
        hi = np.max([np.asarray(bx["max"], float) for bx in boxes], axis=0)
        centre = (lo + hi) / 2.0
    else:
        centre = (frame.lo + frame.hi) / 2.0
    towards = centre - mid
    towards -= up * float(towards @ up)          # horizontal
    dh = d - up * float(d @ up)                  # the segment, horizontal part
    n = np.cross(dh, up)
    if np.linalg.norm(n) < 1e-6 * length:        # a vertical segment
        n = towards if np.linalg.norm(towards) > 1e-6 else np.cross(
            up, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(n) < 1e-6:
            n = np.cross(up, np.array([0.0, 0.0, 1.0]))
    n = n / (np.linalg.norm(n) + 1e-12)
    if float(n @ towards) < 0:
        n = -n                                   # stand on the volume's side
    standoff = 3.0 * length
    # How far back along n the camera can go before leaving the scene's
    # robust bounds (per-axis slab test); at least one segment length,
    # which at 70 deg still holds the segment in frame.
    t_max = standoff
    for ax in range(3):
        if abs(n[ax]) < 1e-9:
            continue
        lim = (frame.hi[ax] if n[ax] > 0 else frame.lo[ax]) - mid[ax]
        t_max = min(t_max, max(lim / n[ax], 0.0))
    standoff = max(min(standoff, 0.9 * t_max), length)
    return look_at(mid + n * standoff, mid, up)


def _draw_segment(img: Image.Image, uv_a, uv_b) -> None:
    """The measured segment into a probe: a 4 px red line with end caps and
    a small A and B — what the model is asked about."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    red = (255, 40, 30)
    dr.line([tuple(uv_a), tuple(uv_b)], fill=red, width=4)
    for (u, v), tag in ((uv_a, "A"), (uv_b, "B")):
        dr.ellipse([u - 7, v - 7, u + 7, v + 7], fill=red, outline=(255, 255, 255))
        dr.rectangle([u + 9, v - 18, u + 24, v - 2], fill=red)
        dr.text((u + 12, v - 17), tag, fill=(255, 255, 255))


def _render_probes(scene, frame, cfg: dict, boxes, out_dir: Path,
                   segment: np.ndarray, should_cancel=None) -> tuple[list[Path], str]:
    """The probes the second opinion looks at: `render.propose.
    probe_views` scale-free orbit poses around the proposal's boxes for
    context, plus ONE view aimed at the measured segment. The segment (the
    two ruler points, in the file frame) is drawn into every view where both
    points project in-frame, in front of the camera and not behind the
    rendered depth (the export's crop test); views with it hidden are
    dropped, and the aimed view — in-frame by construction — always has
    it. Files `out_dir/probe_00.png` …, the aimed view first. Returns the
    paths and the aimed file's name. The splat tensors are freed before
    this returns — the model that reads the probes needs the VRAM."""
    from .cameras import orbit_path
    from .sequencing import check_cancel
    if out_dir.exists():
        shutil.rmtree(out_dir)   # an earlier proposal's probes
    out_dir.mkdir(parents=True)
    n = int(cfg["render"]["propose"]["probe_views"])
    poses = orbit_path(scene.means, scene.opacities, frame, cfg, n,
                       volume_boxes=boxes)
    idx = np.unique(np.linspace(0, len(poses) - 1, n).round().astype(int))
    a, b = np.asarray(segment[0], float), np.asarray(segment[1], float)
    # the segment is a file-frame record (the ruler's clicks): aimed at in
    # the scene frame, drawn and rendered in the file's
    R = scene.alignment
    a_s, b_s = (a, b) if R is None else (R @ a, R @ b)
    c2ws = np.stack([to_file_c2w(p, R) for p in
                     [_aimed_pose(frame, boxes, a_s, b_s)] + [poses[i] for i in idx]])
    fx, fy, cx, cy, w, h = _intrinsics(cfg)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cams = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    near = float(cfg["render"]["near_plane"])
    ccfg = cfg["render"]["coverage"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering (gsplat)")
    scene_t = dict(
        means=torch.as_tensor(scene.means_file, device="cuda"),
        quats=torch.as_tensor(scene.quats_file, device="cuda"),
        scales=torch.as_tensor(scene.scales, device="cuda"),
        opacities=torch.as_tensor(scene.opacities, device="cuda"),
        sh=torch.as_tensor(scene.sh, device="cuda"),
        sh_degree=min(scene.sh_degree, cfg["render"]["sh_degree"]))
    paths: list[Path] = []
    dropped = 0
    try:
        step = max(int(cfg["render"]["views_in_flight"]), 1)
        for start in range(0, len(c2ws), step):
            check_cancel(should_cancel)
            rgb, _, depth, _ = _render_batch(scene_t, c2ws[start:start + step],
                                             K, w, h, cfg)
            for j in range(rgb.shape[0]):
                k = start + j
                aimed = k == 0
                pr = project_visible(
                    np.stack([a, b]), dict(c2w=c2ws[k].tolist()), cams,
                    None if aimed else depth[j].numpy(), near=near,
                    tol_rel=float(ccfg.get("depth_tol_rel", 0.0)),
                    tol_abs=float(ccfg["depth_tol"]))
                if pr is None or not bool(pr.vis.all()):
                    dropped += 1
                    continue
                img = Image.fromarray((rgb[j].numpy() * 255).astype(np.uint8))
                _draw_segment(img, (float(pr.uf[0]), float(pr.vf[0])),
                              (float(pr.uf[1]), float(pr.vf[1])))
                pth = out_dir / f"probe_{len(paths):02d}.png"
                img.save(pth)
                paths.append(pth)
    finally:
        del scene_t
        torch.cuda.empty_cache()
    log.info("propose: %d probe(s) with the measured segment drawn (1 aimed "
             "at it + %d of %d orbit views; %d without the segment in view "
             "dropped) at %dx%d -> %s", len(paths), len(paths) - 1, len(idx),
             dropped, w, h, out_dir)
    return paths, paths[0].name


def _read_probes(cfg: dict, paths: list[Path], aimed: str,
                 should_cancel=None) -> dict:
    """The second opinion: one ask over the probes — how long is the drawn
    segment. A refusal from the backend (weights, VRAM, dependencies) or a
    format failure is RECORDED, not raised — the Check card shows the
    error verbatim; the proposal itself stands."""
    from .vlm import VLMFormatError, get_backend
    t0 = time.perf_counter()
    backend = None
    try:
        backend = get_backend(cfg, should_cancel)
        read = backend.read_length(paths)
    except (VLMFormatError, Refusal) as e:
        log.warning("the second opinion did not run: %s", e)
        return dict(error=str(e))
    finally:
        if backend is not None:
            backend.close()
    elapsed = round(time.perf_counter() - t0, 1)
    log.info("second opinion by %s in %.1f s: A-B reads about %.3g m (%s); %s",
             cfg["vlm"]["model"], elapsed, read["length_m"], read["what"],
             read["reason"])
    return dict(length_m=float(read["length_m"]), what=read["what"],
                reason=read["reason"], probes=[p.name for p in paths],
                aimed=aimed, model=cfg["vlm"]["model"], elapsed_s=elapsed)


def run_propose_volume(scene_path: str, workdir: str, cfg: dict,
                       should_cancel=None) -> None:
    """Propose at the volume gate: write volume.json, the review plot and
    the density sidecar into the workdir (never beside the capture —
    sequencing.SCENE_FILES), then the analysis behind the two Check cards
    — `proposal.json`, written last.

    Refuses FIRST, to the volume gate, when no scale is recorded or
    measured: the grids are sized in metres, and a
    default factor is the blind guess this step removed. The filming
    answer (`render.path_mode`, asked at creation) picks the proposer's
    strategy: `interior` runs the configured one (enclosed-first); `orbit`
    and `ground` take the density core directly. On a MEASURED scene with
    `render.propose.vlm_read` and the model available, the probes are
    rendered with the measured segment drawn in and the local VLM is
    asked how long that segment is — the second opinion the Check card
    states beside the measurement, never applies. Nothing here
    writes a profile.

    `should_cancel` is the web Stop: polled once per grid row inside the
    enclosure scan and between the steps below; a cancel raises
    StageCancelled, and `proposal.json` is removed the moment a new
    volume.json is written, so a stop leaves the volume proposal as it
    always did and no analysis describing a different one.
    """
    from .scene_read import analyse, check_answers, propose_settings
    from .sequencing import check_cancel
    from .vlm import backend_available
    from .volume import (propose_volume, save_volume_review,
                         write_density_sidecar, write_frame_sidecar,
                         write_volume)

    t_start = time.perf_counter()
    # R2: before the scene is even loaded — no factor, no proposal.
    resolve_scale(cfg)
    scene = load_scene(scene_path, cfg)
    wd = Path(workdir)                 # the scene's files live in the workdir
    wd.mkdir(parents=True, exist_ok=True)
    frame = detect_scene_frame(scene.means, scene.opacities, cfg)
    frame.alignment = scene.alignment   # the levelling
    # Length knobs (voxel_size, floor_margin, the probes' clip planes) ->
    # scene units, by the frame's one factor. `cfg_m` stays the profile's
    # metres: the proposed thresholds are numbers the operator types.
    cfg_m = cfg
    cfg, _ = scene_units_cfg(cfg, frame.scale)
    cfg, _ = resolve_auto_quality(cfg, frame)   # the probes' near-field pass
    # The filming answer picks the strategy: inside a space -> as
    # configured (enclosed-first, with L77a's over-budget fallback); around
    # a subject / outdoors -> the density core, which the read-driven
    # re-proposal of step 4 reached by a detour. `auto` (no answer, only in
    # a hand-written profile) and `manual` run the configured strategy.
    said = str(cfg["render"]["path_mode"])
    if said in ("orbit", "ground"):
        cfg = {**cfg, "render": {**cfg["render"], "volume_proposal": {
            **cfg["render"]["volume_proposal"], "strategy": "density_core"}}}
        log.info("propose: filmed %s; the density core, no enclosure scan",
                 "around a subject" if said == "orbit" else "outdoors")
    try:
        proposal = propose_volume(scene.means, scene.opacities, frame, cfg,
                                  Path(scene_path).name,
                                  should_cancel=should_cancel)
    except Refusal:
        # Refused before the scan (grid budget). The frame read still
        # reaches the sidecar so the volume gate can show the dimensions
        # and take the scale; nothing else is written.
        write_frame_sidecar(frame, scene.means, scene.opacities,
                            cfg["render"]["interior"]["voxel_size"],
                            wd / "density.json")
        raise

    def publish(prop: dict) -> None:
        check_cancel(should_cancel)   # the last poll before anything is written
        (wd / "proposal.json").unlink(missing_ok=True)   # describes old boxes
        write_volume(prop, wd / "volume.json", frame)
        save_volume_review(prop, wd / "volume_review.png")
        write_density_sidecar(prop, frame, scene.means, scene.opacities,
                              wd / "density.json")
    publish(proposal)

    # --- the second opinion, on a measured scene ---------------------------
    probe_dir = wd / "probe"
    read = None
    measured = frame.scale_source == "measured"
    if measured and cfg["render"]["propose"]["vlm_read"]:
        ok, why = backend_available(cfg)
        segment = (frame.notes.get("scale") or {}).get("measured", {}).get("points")
        if not ok:
            log.warning("second opinion skipped, the model is not available: %s",
                        why.strip().splitlines()[0])
            read = dict(error=why)
        elif not segment or len(segment) != 2:
            read = dict(error="the measurement carries no points to draw; "
                              "measure again on the canvas")
            log.warning("second opinion skipped: %s", read["error"])
        else:
            check_cancel(should_cancel)
            paths, aimed = _render_probes(scene, frame, cfg, proposal["boxes"],
                                          probe_dir, np.asarray(segment, float),
                                          should_cancel=should_cancel)
            check_cancel(should_cancel)
            read = _read_probes(cfg, paths, aimed, should_cancel)
            check_cancel(should_cancel)
    elif probe_dir.exists():
        shutil.rmtree(probe_dir)   # a declared factor asks nothing; no stale probes

    boxes = [dict(name=b["name"], min=np.asarray(b["min"], float),
                  max=np.asarray(b["max"], float)) for b in proposal["boxes"]]
    analysis = analyse(scene.means, scene.opacities, frame, cfg, boxes,
                       proposal["_used"],
                       proposal["_reason"] or ("not_scanned" if said in
                                               ("orbit", "ground") else None))
    # The level read: on the file's own means, so the angles are
    # absolute; stated on the Check card, adopted only by the operator.
    from .alignment import describe_level, propose_alignment
    check_cancel(should_cancel)
    t_al = time.perf_counter()
    level = propose_alignment(scene.means_file, scene.opacities, frame.up_axis,
                              frame.up_sign, cfg, said, boxes,
                              current_R=scene.alignment, frame=frame)
    level["current"] = frame.alignment_block
    level["elapsed_s"] = round(time.perf_counter() - t_al, 1)
    analysis["alignment"] = level
    log.info("propose: level read (%s, %.1f s): %s%s", level.get("reading"),
             level["elapsed_s"],
             "off level" if level["off_level"] else "level",
             (f": tilt {level['tilt_deg']} yaw {level['yaw_deg']} "
              f"(floor share {level['floor_share_as_is']} -> "
              f"{level['floor_share_levelled']}, yaw margin {level['yaw_margin']})"
              if level.get("found") else " (no floor found)"))
    check = check_answers(analysis, cfg_m, read)
    # the profile's block, not the angles alone: its `source` says whether
    # the operator levelled by hand, which the line defers to
    check["level"] = describe_level(level, (cfg.get("scene") or {}).get("alignment"))
    settings = propose_settings(analysis, cfg_m)
    log.info("propose: filmed %s: %s%s; proposal %s%s",
             check["path_mode"]["said"],
             {True: "fits", False: "FLAGGED", None: "unknown"}[check["path_mode"]["fits"]],
             (f"; the measurement {'fits' if check['scale']['fits'] else 'FLAGGED'}: "
              f"{check['scale']['reason']}" if check.get("scale") else ""),
             settings["settings"],
             (f"; thresholds {settings['thresholds']}" if settings["thresholds"] else ""))
    if check.get("opinion"):
        log.info("propose: second opinion %s: %s",
                 {True: "agrees", False: "FLAGGED", None: "did not run"}[check["opinion"]["fits"]],
                 check["opinion"]["reason"].strip().splitlines()[0])
    check_cancel(should_cancel)
    out = dict(version=2, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               scene_frame=dict(up_axis="xyz"[frame.up_axis],
                                up_sign=frame.up_sign, floor=frame.floor,
                                alignment=frame.alignment_block),
               analysis=analysis, read=read, check=check, proposal=settings,
               elapsed_s=round(time.perf_counter() - t_start, 1))
    (wd / "proposal.json").write_text(json.dumps(out, indent=1))   # LAST
    log.info("proposal.json -> %s (%.1f s)", wd / "proposal.json",
             out["elapsed_s"])
