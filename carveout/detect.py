# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 2: SAM 3 Promptable Concept Segmentation on Stage 1 renders.

The probe: per-image PCS over every stage-1 frame,
logging ALL raw presence scores >= detect.log_threshold to CSV for empirical
calibration (photo-domain thresholds are a hypothesis, not a given).
The pseudo-video tracking mode for cross-view instance IDs is attempted at the
full Stage 2 run, not in the probe.

Outputs under <workdir>/stage2/: scores.csv (every detection), score_summary.csv
(per-concept aggregates), score_distribution.png, overlays/frame_XXXX.png,
masks/frame_XXXX/<concept>_<k>.png, probe_manifest.json.

Exemplar pass : per-scene exemplars.json crops run through
the SAM 3.1 video-multiplex predictor — one tracking session per crop over ALL
stage-1 frames; detections enter scores.csv additively with
source=exemplar:<concept> (text rows carry source=text) and are gated
downstream by detect.exemplar_threshold, never presence_threshold. A changed
exemplars.json re-runs ONLY this pass (text results kept and folded).
"""

import contextlib
import csv
import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .exemplars import (check_crop_frames, load_exemplars,
                        resolve_exemplars_path)
from .refusal import Refusal, missing_dependency
from .sequencing import fingerprint, stale_upstream, upstream_fingerprint
from .manual_views import (frame_intrinsics,
                           stage1_manual_record, stale_manual_views)


@contextlib.contextmanager
def _quiet_sam3_import():
    """SAM 3's package imports emit two third-party deprecation notices
    (its own `pkg_resources` use, and timm's old layers path) that mean
    nothing to an operator reading the stage log. Muted for the import
    only, by category and by the module that raises each — every other
    warning, and everything after the import, is left alone."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"sam3\.")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\.")
        yield

log = logging.getLogger(__name__)

GATED_HELP = """\
ERROR: SAM 3 checkpoint missing: {path}

The checkpoints are GATED on Hugging Face:
  1. Request access at https://huggingface.co/facebook/sam3
  2. Authenticate: conda activate carveout && hf auth login
  3. Download into the project (never a global cache):
       hf download facebook/sam3 sam3.pt config.json --local-dir models/sam3/
"""

PALETTE = [(47, 129, 247), (63, 185, 80), (219, 109, 40), (163, 113, 247),
           (247, 120, 186), (227, 179, 65), (86, 212, 221), (248, 81, 73)]

CSV_FIELDS = ["frame_idx", "concept", "det_idx", "score", "area_frac",
              "x1", "y1", "x2", "y2", "negative_control", "source"]


def _read_rows(csv_path) -> list[dict]:
    """scores.csv rows with types restored; pre-feature rows get source=text."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["frame_idx"] = int(r["frame_idx"])
            r["det_idx"] = int(r["det_idx"])
            r["score"] = float(r["score"])
            r["area_frac"] = float(r["area_frac"])
            r["negative_control"] = r["negative_control"] == "True"
            r["source"] = r.get("source") or "text"
            rows.append(r)
    return rows


def _mask_file(out: Path, r: dict) -> Path:
    return (out / "masks" / f"frame_{r['frame_idx']:04d}" /
            f"{r['concept'].replace(' ', '_')}_{r['det_idx']}.png")


def _drop_exemplar_artifacts(out: Path, rows: list[dict]) -> list[dict]:
    """Remove exemplar rows and their masks/overlays from a previous pass;
    returns the surviving text rows (stale-artifact policy, same reason as
    export's instances/ cleanup)."""
    for r in rows:
        if r["source"] != "text":
            _mask_file(out, r).unlink(missing_ok=True)
    if (out / "overlays").exists():
        for f in (out / "overlays").glob("exemplar_frame_*.png"):
            f.unlink()
    return [r for r in rows if r["source"] == "text"]


def _overlay(img: Image.Image, dets: list[dict], thresh: float) -> Image.Image:
    out = img.convert("RGBA")
    draw_dets = [d for d in dets if d["score"] >= thresh]
    for i, d in enumerate(draw_dets):
        color = PALETTE[i % len(PALETTE)]
        layer = np.zeros((*d["_mask"].shape, 4), dtype=np.uint8)
        layer[d["_mask"]] = (*color, 110)
        out = Image.alpha_composite(out, Image.fromarray(layer))
    dr = ImageDraw.Draw(out)
    for i, d in enumerate(draw_dets):
        color = PALETTE[i % len(PALETTE)]
        x1, y1 = d["x1"], d["y1"]
        dr.rectangle([d["x1"], d["y1"], d["x2"], d["y2"]], outline=color, width=2)
        dr.text((x1 + 3, max(y1 - 12, 1)), f"{d['concept']} {d['score']:.2f}",
                fill=(255, 255, 255))
    return out.convert("RGB")


def run_detect(workdir: str, cfg: dict, prompts: list[str],
               negatives: list[str] | None = None, force: bool = False,
               should_cancel=None) -> dict:
    """The stage-2 probe over the whole kept frame set: the text pass, then
    the exemplar pass when exemplars.json has crops. A changed exemplars.json
    re-runs ONLY the exemplar pass (fold); `force` re-runs both."""
    stage1 = Path(workdir) / "stage1"
    out = Path(workdir) / "stage2"
    manifest_path = out / "probe_manifest.json"

    ex_path = resolve_exemplars_path(workdir, cfg)
    exemplars = load_exemplars(ex_path)
    ex_hash = fingerprint(ex_path)

    # Cache: a text-side change (manual views, the views themselves, the
    # volume) invalidates everything; a changed exemplars.json re-runs
    # ONLY the exemplar pass (fold). The probe is current when it
    # was run over THESE views (the stage-1 manifest) and THIS volume —
    # content, not the session's memory of which gates were entered — so
    # a probe run after the render approval stands at the pipeline's
    # first press instead of running twice.
    from .volume import resolve_volume_path
    vol_hash = fingerprint(resolve_volume_path(workdir, cfg))
    exemplar_fold = False
    cached: dict | None = None
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        same_volume = cached.get("volume_hash") == vol_hash
        if not same_volume:
            log.warning("stage2 cache INVALIDATED: the volume changed since "
                        "the probe (%s -> %s); re-running stage2",
                        str(cached.get("volume_hash", "absent"))[:12],
                        vol_hash[:12])
        if (same_volume
                and not stale_manual_views(cached, workdir, "stage2")
                and not stale_upstream(cached, workdir, "stage2",
                                       "upstream_stage1",
                                       "stage1/manifest.json")):
            if cached.get("exemplars_hash", "absent") == ex_hash:
                log.info("stage2 probe cached at %s: the probe of %s over "
                         "these views, this volume and this vocabulary "
                         "stands; not run again", out,
                         cached.get("finished_at", "an earlier run"))
                return cached
            log.warning(
                "stage2 cache INVALIDATED (exemplar section only): "
                "exemplars.json changed (%s -> %s); re-running the "
                "EXEMPLAR pass; text results are kept",
                cached.get("exemplars_hash", "absent")[:12], ex_hash[:12])
            exemplar_fold = True

    cams_path = stage1 / "cameras.json"
    if not cams_path.exists():
        raise Refusal(f"{cams_path} not found; render first (the render "
                      f"gate)", gate="render")
    cams = json.loads(cams_path.read_text())
    negatives = negatives or []
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    if exemplar_fold:
        rows = _drop_exemplar_artifacts(out, _read_rows(out / "scores.csv"))
        prompts = cached["prompts"]
        negatives = cached.get("negatives", [])
        frames_rec = cached["frames"]
        text_meta = {k: cached[k] for k in
                     ("model_checkpoint", "model_load_s", "inference_s",
                      "s_per_frame") if k in cached}
    else:
        rows, frames_rec, text_meta = _run_text_pass(
            stage1, out, cfg, cams, prompts, negatives,
            should_cancel=should_cancel)

    # --- exemplar pass -----------------------------------------
    run_pass = bool(exemplars)
    ex_meta: dict = {}
    if run_pass:
        ex_rows, ex_meta = _run_exemplar_pass(stage1, out, cfg, exemplars,
                                              cams, rows,
                                              should_cancel=should_cancel)
        rows = rows + ex_rows

    exemplar_concepts = [e["concept"] for e in exemplars]
    with open(out / "scores.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wtr.writeheader()
        wtr.writerows(rows)

    # Per-concept summary (40+ concepts must be reviewable at a glance).
    # Exemplar rows aggregate under "exemplar:<concept>" — their scores live
    # on a different scale than text confidence and must never share a bin.
    by_c = defaultdict(list)
    for r in rows:
        key = r["concept"] if r["source"] == "text" else r["source"]
        by_c[key].append(r)
    summary = []
    for concept in list(prompts) + [f"exemplar:{c}" for c in exemplar_concepts]:
        rs = by_c.get(concept, [])
        scores = [r["score"] for r in rs]
        summary.append(dict(
            concept=concept,
            negative_control=concept in negatives,
            n_det=len(rs),
            n_frames=len({r["frame_idx"] for r in rs}),
            max_score=max(scores) if scores else 0.0,
            mean_score=round(float(np.mean(scores)), 4) if scores else 0.0,
            mean_area=round(float(np.mean([r["area_frac"] for r in rs])), 5) if rs else 0.0,
        ))
    summary.sort(key=lambda s: -s["max_score"])
    with open(out / "score_summary.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wtr.writeheader()
        wtr.writerows(summary)

    _distribution_plot(summary, by_c, out / "score_distribution.png")

    n_ex = sum(1 for r in rows if r["source"] != "text")
    manifest = dict(stage="detect_probe", prompts=prompts, negatives=negatives,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    # The views and the volume this probe was run
                    # over — the cache check reads these, not the session
                    upstream_stage1=upstream_fingerprint(
                        workdir, "stage1/manifest.json"),
                    volume_hash=vol_hash,
                    manual_views_hash=stage1_manual_record(workdir)
                    .get("hash", "absent"),
                    exemplars_path=str(ex_path) if ex_path else None,
                    exemplars_hash=ex_hash,
                    exemplar_concepts=exemplar_concepts,
                    frames=frames_rec,
                    log_threshold=cfg["detect"]["log_threshold"],
                    n_detections=len(rows), n_exemplar_detections=n_ex,
                    **text_meta, **ex_meta)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log.info("stage2 probe done: %d detections (%d exemplar) -> %s",
             len(rows), n_ex, out)
    return manifest


def _mask_floor(concept: str, cfg: dict) -> float:
    """The score at and above which a text detection's mask is kept and
    written: the overlay threshold, or the concept's effective presence
    threshold when a profile sets that lower (the lift reads every mask
    above the presence threshold, so a floor above it loses detections)."""
    d = cfg["detect"]
    presence = (d.get("presence_threshold_overrides") or {}).get(
        concept, d["presence_threshold"])
    return min(d["overlay_threshold"], presence)


def _run_text_pass(stage1: Path, out: Path, cfg: dict, cams: dict,
                   prompts: list[str], negatives: list[str],
                   should_cancel=None):
    """Per-image text PCS over every stage-1 frame.
    Returns (rows, probed frame indices, timing/checkpoint manifest fields)."""
    import torch
    from .sequencing import check_cancel

    ckpt = Path(cfg["detect"]["checkpoint_image"])
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parent.parent / ckpt
    if not ckpt.exists():
        raise Refusal(GATED_HELP.format(path=ckpt))

    frames = cams["frames"]
    log.info("stage2 probe: %d frames x %d concepts (%d negative controls)",
             len(frames), len(prompts), len(negatives))

    try:
        with _quiet_sam3_import():
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
    except ImportError as e:
        raise missing_dependency("sam3", e) from e

    t0 = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(ckpt), load_from_HF=False)
    # Log everything above log_threshold; the production presence_threshold is
    # applied downstream after calibration (docs/CALIBRATION.md).
    proc = Sam3Processor(model, confidence_threshold=cfg["detect"]["log_threshold"])
    t_load = time.perf_counter() - t0
    log.info("SAM 3 image model loaded in %.1f s", t_load)

    rows = []
    t_infer = time.perf_counter()
    for f_i, fr in enumerate(frames):
        check_cancel(should_cancel)   # between per-frame SAM3 inferences
        img = Image.open(stage1 / fr["file"]).convert("RGB")
        fdets = []
        # bf16 autocast is required.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Pass the PIL image, NOT np.asarray: with an (H,W,3) array the
            # processor reads original_width from the channel dim (=3) and
            # produces 768x3 masks.
            state = proc.set_image(img)
            for concept in prompts:
                output = proc.set_text_prompt(prompt=concept, state=state)
                masks = output.get("masks")
                scores = output.get("scores")
                if masks is None or scores is None or len(scores) == 0:
                    proc.reset_all_prompts(state)
                    continue
                masks = masks.squeeze(1) if masks.ndim == 4 else masks
                # SAM 3 masks come back at its processing resolution (1008);
                # resize to the render so bbox/area are in frame pixels.
                if masks.shape[-2:] != (img.height, img.width):
                    masks = torch.nn.functional.interpolate(
                        masks[None].float(), size=(img.height, img.width),
                        mode="nearest")[0]
                # Box and area for EVERY mask in one batched reduction on
                # the GPU: the per-detection copy to the CPU and
                # np.nonzero over 1024² cost ~5 ms each, 650 of the
                # Counter's 768 inference seconds over 129,746 detections.
                # The values are the same integers as before; only the
                # masks above the overlay threshold (the ones written and
                # drawn, ~10 %) come to the CPU at all.
                mb = masks.float() > 0.5                       # [N, H, W]
                H, W = mb.shape[-2:]
                area = mb.flatten(1).sum(1).tolist()
                rows_any = mb.any(2).float()                   # [N, H]
                cols_any = mb.any(1).float()                   # [N, W]
                y1s = rows_any.argmax(1).tolist()
                y2s = (H - 1 - rows_any.flip(1).argmax(1)).tolist()
                x1s = cols_any.argmax(1).tolist()
                x2s = (W - 1 - cols_any.flip(1).argmax(1)).tolist()
                # Masks must exist for every det that can clear the
                # concept's presence threshold — which a profile may set
                # BELOW overlay_threshold (a rescue at 0.25 vs overlay
                # 0.30), so the floor follows both; the lift refuses when
                # a mask it needs is missing. The exemplar pass has the
                # same rule against its own thresholds.
                keep_mask = _mask_floor(concept, cfg)
                for k in range(len(scores)):
                    if not area[k]:
                        continue
                    score = round(float(scores[k]), 4)
                    det = dict(
                        frame_idx=fr["frame_idx"], concept=concept, det_idx=k,
                        score=score,
                        area_frac=round(area[k] / (H * W), 5),
                        x1=x1s[k], y1=y1s[k], x2=x2s[k], y2=y2s[k],
                        negative_control=concept in negatives,
                        source="text",
                        _mask=(mb[k].cpu().numpy() if score >= keep_mask
                               else None),
                    )
                    fdets.append(det)
                proc.reset_all_prompts(state)
        mask_dir = out / "masks" / f"frame_{fr['frame_idx']:04d}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for d in fdets:
            if d["score"] >= _mask_floor(d["concept"], cfg):
                Image.fromarray((d["_mask"] * 255).astype(np.uint8)).save(
                    mask_dir / f"{d['concept'].replace(' ', '_')}_{d['det_idx']}.png")
        ov = _overlay(img, sorted(fdets, key=lambda d: -d["score"]),
                      cfg["detect"]["overlay_threshold"])
        ov.save(out / "overlays" / f"frame_{fr['frame_idx']:04d}.png")
        rows.extend({k: v for k, v in d.items() if k != "_mask"} for d in fdets)
        log.info("frame %04d (%d/%d): %d detections >= %.2f",
                 fr["frame_idx"], f_i + 1, len(frames), len(fdets),
                 cfg["detect"]["log_threshold"])
    t_infer = time.perf_counter() - t_infer
    log.info("text pass done: %d detections, %.1f s (%.1f s/frame)",
             len(rows), t_infer, t_infer / len(frames))
    # Free the image model before the (possible) exemplar pass loads the
    # video model — the two must never be resident together (24 GB profile).
    del proc, model
    torch.cuda.empty_cache()

    text_meta = dict(model_checkpoint=str(ckpt), model_load_s=round(t_load, 1),
                     inference_s=round(t_infer, 1),
                     s_per_frame=round(t_infer / len(frames), 2))
    return rows, [f["frame_idx"] for f in frames], text_meta


def _fold_duplicate(existing: list[dict], det: dict, iou_t: float = 0.6) -> bool:
    """True when det duplicates a same-concept exemplar det already found on
    this frame by ANOTHER crop's session (mask IoU >= iou_t) — the higher
    score stays. Cross-SOURCE duplicates (text vs exemplar) are NOT folded
    here; they collapse at lift's mask dedup where the decision is audited."""
    for e in existing:
        if e["source"] != det["source"]:
            continue
        inter = int((e["_mask"] & det["_mask"]).sum())
        if not inter:
            continue
        if inter / int((e["_mask"] | det["_mask"]).sum()) >= iou_t:
            if det["score"] > e["score"]:
                e.update(det)
            return True
    return False


def _run_exemplar_pass(stage1: Path, out: Path, cfg: dict,
                       exemplars: list[dict], cams: dict,
                       prior_rows: list[dict],
                       should_cancel=None) -> tuple[list[dict], dict]:
    """Video-multiplex visual-prompt pass .

    One tracking session per crop over ALL stage-1 frames (never the probe
    subset — the tracker wants frame density): the detector finds every
    instance matching the crop on its own frame, the tracker propagates them
    to the rest. Scores are exemplar-conditioned (sigmoid(logits) x presence
    against the visual prompt) — a DIFFERENT scale from text confidence,
    gated downstream by detect.exemplar_threshold. Masks/overlays are written
    like the text pass (masks above overlay_threshold; overlays prefixed
    exemplar_, source crops drawn in white). det_idx continues after
    prior_rows' numbering per (frame, concept) so mask files never collide.
    """
    import torch
    try:
        with _quiet_sam3_import():
            from sam3.model_builder import build_sam3_multiplex_video_predictor
    except ImportError as e:
        raise missing_dependency("sam3", e) from e

    from .sequencing import check_cancel
    check_crop_frames(exemplars, stage1)
    ckpt = Path(cfg["detect"]["checkpoint_video"])
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parent.parent / ckpt
    if not ckpt.exists():
        raise Refusal(GATED_HELP.format(path=ckpt))

    frames_by_idx = {fr["frame_idx"]: fr for fr in cams["frames"]}
    log_thr = cfg["detect"]["log_threshold"]
    n_crops = sum(len(e["crops"]) for e in exemplars)
    log.info("exemplar pass: %d concept(s), %d crop(s), %d frames",
             len(exemplars), n_crops, len(frames_by_idx))

    # bound the tracking session's frame count for 24 GB VRAM headroom.
    # The multiplex session loads EVERY frame under resource_path and its
    # memory bank grows with frame count, so peak VRAM tracks the scene's
    # frame count (47 frames OOM'd a 4090 mid-propagation). Cap it: force-keep
    # every exemplar CROP frame (they carry the prompts), evenly sample the
    # rest up to the cap, and run the session over a symlinked subset dir with
    # a session-position <-> real-frame-idx remap. Frame files are named by
    # index and the stock path relies on position == frame_idx, which a subset
    # breaks — the maps below restore it. Off (null / cap >= n) = identity.
    all_idx = sorted(frames_by_idx)
    crop_idx = sorted({c["frame_idx"] for e in exemplars for c in e["crops"]})
    cap = cfg["detect"].get("exemplar_max_frames")
    frames_dir = stage1 / "frames"
    pos2idx = {i: i for i in all_idx}   # session position -> real frame_idx
    idx2pos = {i: i for i in all_idx}
    frame_tmp = None
    if cap and cap < len(all_idx):
        keep = set(crop_idx)
        rest = [i for i in all_idx if i not in keep]
        room = cap - len(keep)
        if room > 0 and rest:
            picks = np.unique(np.linspace(
                0, len(rest) - 1, num=min(room, len(rest))).round().astype(int))
            keep.update(rest[p] for p in picks)
        elif room < 0:
            log.warning("exemplar frame cap %d < %d crop frame(s): all crop "
                        "frames kept, cap not reached", cap, len(crop_idx))
        sel = sorted(keep)
        import tempfile
        frame_tmp = tempfile.TemporaryDirectory(prefix="carveout_excap_")
        frames_dir = Path(frame_tmp.name)
        pos2idx, idx2pos = {}, {}
        for pos, fi in enumerate(sel):
            (frames_dir / f"frame_{pos:04d}.png").symlink_to(
                (stage1 / "frames" / f"frame_{fi:04d}.png").resolve())
            pos2idx[pos], idx2pos[fi] = fi, pos
        log.info("exemplar frame cap: %d -> %d frames "
                 "(cap=%d, %d crop frame(s) force-kept)",
                 len(all_idx), len(sel), cap, len(crop_idx))

    # measure the pass's peak VRAM so the frame cap can be tuned from
    # data, not a flush-to-the-limit guess (this pass is the pipeline's VRAM
    # ceiling on a 24 GB card). Reset here so the peak covers model load +
    # session + propagation — the full exemplar-pass footprint.
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    # use_fa3=False: FlashAttention 3 is not installed (Hopper-only kernels).
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(ckpt), use_fa3=False, async_loading_frames=False)
    t_load = time.perf_counter() - t0
    log.info("SAM 3.1 multiplex video model loaded in %.1f s", t_load)

    # The demo model's long-video heuristics assume the detector re-finds
    # every object on every frame. A pure visual prompt cannot provide that:
    # on non-prompted frames the detector runs against a dummy "visual" text,
    # so tracks are never re-matched — hotstart retro-REMOVED the masklets
    # and masklet confirmation (3 consecutive re-detections) hid the rest
    # (measured: 1 detection from 8 crops with both on). Disable
    # both for exemplar sessions; detection quality is still gated by the
    # exemplar threshold + the full downstream funnel.
    predictor.model.hotstart_delay = 0
    predictor.model.masklet_confirmation_enable = False
    try:
        sid = predictor.start_session(
            resource_path=str(frames_dir))["session_id"]
    except TypeError:
        # Upstream signature skew (sam3 March-2026 release): the base
        # predictor's start_session always passes offload_state_to_cpu, which
        # the multiplex init_state does not accept. Reproduce its
        # bookkeeping minus the kwarg.
        import uuid
        state = predictor.model.init_state(
            resource_path=str(frames_dir), async_loading_frames=False)
        sid = str(uuid.uuid4())
        predictor._all_inference_states[sid] = {
            "state": state, "session_id": sid,
            "start_time": time.time(), "last_use_time": time.time()}
        log.info("start_session fallback engaged (upstream "
                 "offload_state_to_cpu signature skew)")

    dets_by_frame: dict[int, list[dict]] = defaultdict(list)
    t_infer = time.perf_counter()
    for ex in exemplars:
        check_cancel(should_cancel)   # between per-exemplar tracking runs
        concept = ex["concept"]
        for ci, crop in enumerate(ex["crops"]):
            fr = frames_by_idx.get(crop["frame_idx"])
            if fr is None:
                raise Refusal(
                    f"exemplar crop for {concept!r} references frame "
                    f"{crop['frame_idx']} which is not in stage1 cameras.json")
            _, _, _, _, w, h = frame_intrinsics(fr, cams)
            x1, y1, x2, y2 = crop["box_xyxy"]
            tc = time.perf_counter()
            # bf16 autocast is required.
            with torch.inference_mode(), \
                    torch.autocast("cuda", dtype=torch.bfloat16):
                # Prompted frame: spawn masklets down to the logging floor
                # (the stock spawn gate, new_det_thresh 0.65, sits far above
                # exemplar-similarity scores on splat renders — run 3).
                predictor.model.score_threshold_detection = log_thr
                predictor.model.new_det_thresh = log_thr
                resp = predictor.add_prompt(
                    session_id=sid, frame_idx=idx2pos[crop["frame_idx"]],
                    bounding_boxes=[[x1 / w, y1 / h,
                                     (x2 - x1) / w, (y2 - y1) / h]],
                    bounding_box_labels=[1], output_prob_thresh=log_thr)
                outs = {resp["frame_index"]: resp["outputs"]}
                # Propagation: pure tracking. The per-frame detector runs on
                # a dummy "visual" text there — raising the gates above 1
                # stops its noise from spawning objects or reconditioning
                # our masks.
                predictor.model.score_threshold_detection = 1.1
                predictor.model.new_det_thresh = 1.1
                for r in predictor.propagate_in_video(
                        session_id=sid, propagation_direction="both",
                        output_prob_thresh=log_thr):
                    outs[r["frame_index"]] = r["outputs"]
            n_new = 0
            for pos, o in outs.items():
                fidx = pos2idx.get(pos)   # session position -> real frame_idx
                if fidx is None:
                    continue
                tfr = frames_by_idx.get(fidx)
                if tfr is None:
                    continue
                _, _, _, _, wf, hf = frame_intrinsics(tfr, cams)
                probs = np.asarray(o["out_probs"])
                masks = np.asarray(o["out_binary_masks"])
                for k in range(len(probs)):
                    if float(probs[k]) < log_thr:
                        continue
                    m = masks[k]
                    # session masks come back at ONE global size; resize to
                    # this frame's true pixels (manual frames differ)
                    if m.shape != (hf, wf):
                        m = np.asarray(
                            Image.fromarray(m.astype(np.uint8) * 255)
                            .resize((wf, hf), Image.NEAREST)) > 127
                    m = m.astype(bool)
                    ys, xs = np.nonzero(m)
                    if len(xs) == 0:
                        continue
                    det = dict(
                        frame_idx=fidx, concept=concept, det_idx=-1,
                        score=round(float(probs[k]), 4),
                        area_frac=round(float(m.mean()), 5),
                        x1=int(xs.min()), y1=int(ys.min()),
                        x2=int(xs.max()), y2=int(ys.max()),
                        negative_control=False,
                        source=f"exemplar:{concept}", _mask=m)
                    if not _fold_duplicate(dets_by_frame[fidx], det):
                        dets_by_frame[fidx].append(det)
                        n_new += 1
            log.info("exemplar %r crop %d/%d (frame %04d): %d detection(s) "
                     "kept across %d frame(s), %.1f s",
                     concept, ci + 1, len(ex["crops"]), crop["frame_idx"],
                     n_new, len(outs), time.perf_counter() - tc)
            # inference_mode required: the session tensors were created under
            # init_state's @torch.inference_mode, and reset_state updates
            # them in place (it crashed otherwise).
            with torch.inference_mode():
                predictor.reset_session(sid)
    # peak torch-allocated VRAM for this pass (before empty_cache; the
    # peak stat survives it). Non-torch/context memory sits ON TOP of this,
    # so headroom to the card total is smaller than (total - peak_alloc).
    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("exemplar pass peak VRAM: %.2f GB allocated / %.2f GB reserved "
             "of %.1f GB (%d frames, cap=%s)", peak_alloc_gb,
             peak_reserved_gb, total_gb, len(pos2idx), cap)
    with torch.inference_mode():
        predictor.close_session(sid)
    del predictor
    torch.cuda.empty_cache()
    if frame_tmp is not None:   # drop the symlinked subset dir
        frame_tmp.cleanup()
    t_infer = time.perf_counter() - t_infer

    # det_idx continues after existing numbering per (frame, concept)
    next_idx: dict[tuple, int] = defaultdict(int)
    for r in prior_rows:
        key = (r["frame_idx"], r["concept"])
        next_idx[key] = max(next_idx[key], r["det_idx"] + 1)

    crop_boxes: dict[int, list] = defaultdict(list)
    for ex in exemplars:
        for c in ex["crops"]:
            crop_boxes[c["frame_idx"]].append(c["box_xyxy"])

    rows = []
    for fidx in sorted(set(dets_by_frame) | set(crop_boxes)):
        dets = dets_by_frame.get(fidx, [])
        for d in dets:
            key = (fidx, d["concept"])
            d["det_idx"] = next_idx[key]
            next_idx[key] += 1
        mask_dir = out / "masks" / f"frame_{fidx:04d}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for d in dets:
            # Masks must exist for every det that can clear the concept's
            # exemplar threshold — which scene profiles may set BELOW
            # overlay_threshold (an ink bottle rescued at
            # 0.212-0.299 vs overlay 0.30), so the floor follows both.
            thr = cfg["detect"]["exemplar_threshold_overrides"].get(
                d["concept"], cfg["detect"]["exemplar_threshold"])
            if d["score"] >= min(cfg["detect"]["overlay_threshold"], thr):
                Image.fromarray((d["_mask"] * 255).astype(np.uint8)).save(
                    _mask_file(out, d))
        img = Image.open(stage1 / frames_by_idx[fidx]["file"]).convert("RGB")
        ov = _overlay(img, sorted(dets, key=lambda d: -d["score"]),
                      cfg["detect"]["overlay_threshold"])
        dr = ImageDraw.Draw(ov)
        for b in crop_boxes.get(fidx, []):   # the drawn reference crops
            dr.rectangle(b, outline=(255, 255, 255), width=3)
            dr.text((b[0] + 3, min(b[3] + 3, ov.height - 12)),
                    "exemplar crop", fill=(255, 255, 255))
        ov.save(out / "overlays" / f"exemplar_frame_{fidx:04d}.png")
        rows.extend({k: v for k, v in d.items() if k != "_mask"} for d in dets)

    log.info("exemplar pass done: %d detections, %.1f s (%.1f s/crop)",
             len(rows), t_infer, t_infer / max(n_crops, 1))
    return rows, dict(exemplar_model_load_s=round(t_load, 1),
                      exemplar_s=round(t_infer, 1),
                      exemplar_frames_total=len(all_idx),
                      exemplar_frames_used=len(pos2idx),
                      exemplar_max_frames=cap,
                      exemplar_peak_alloc_gb=round(peak_alloc_gb, 2),
                      exemplar_peak_reserved_gb=round(peak_reserved_gb, 2))


def _distribution_plot(summary: list[dict], by_c: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = [s["concept"] for s in summary]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(concepts) + 2))
    for y, s in enumerate(summary):
        xs = [r["score"] for r in by_c.get(s["concept"], [])]
        color = "crimson" if s["negative_control"] else "steelblue"
        ax.scatter(xs, [y] * len(xs), s=14, alpha=0.6, color=color)
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels([("[NEG] " if s["negative_control"] else "") + s["concept"]
                        for s in summary], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SAM 3 presence-gated score (all raw detections)")
    ax.set_title("Stage 2 probe: per-concept score distribution")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
