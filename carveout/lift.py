# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage 3: lift 2D class masks to per-Gaussian labels (FlashSplat-style).

Rendered mask value is linear in per-Gaussian colors under alpha compositing, so
each Gaussian's accumulated contribution to in-mask pixels equals the gradient of
S = sum(render * mask) w.r.t. its color. One backward pass per view with C+1
channels (one per class present + one all-ones total) yields every class
accumulator at once — no rasterizer surgery needed (route decision vs vendoring
FlashSplat: gsplat's autograd exposes exactly the weights we need; documented in
development notes).

Assignment: winner = argmax_c A_c, assigned iff A_win > gamma * (T - A_win).
Detections are containment-deduped first (mutual = synonym crossfire, keep the
higher score; consistent asymmetric = part-of, whole keeps its parts).
Instances: per class, voxel connected components + per-view 2D
count sanity; when 3D undercounts, splits are guided by per-DETECTION
contribution profiles (which 2D instance each Gaussian rendered into), falling
back to coordinate k-means; every decision logged to instance_audit.json —
the audit trail for 3D association. Splits must SEPARATE the 2D evidence (children
pairwise-distinct) or they are refused, and overlapping same-class instances
that are not pairwise-distinct are merged back — the split/merge
counterweight. Label scoping uses volume + lift.scope_margin
(the trimmed volume itself answers "where cameras go").

Outputs under <workdir>/stage3/: labels.npy (class id per Gaussian, -1 = bg),
instances.npy (instance id, -1 = none), classes.json, instance_audit.json,
manifest.json.
"""

import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .exemplars import stage2_exemplar_record, stale_exemplars
from .projection import project_visible
from .refusal import Refusal, missing_dependency
from .manual_views import (frame_intrinsics, stage1_manual_record,
                           stale_manual_views)
from .scene_io import load_scene
from .volume import in_volume, load_volume, resolve_volume_path

log = logging.getLogger(__name__)

# VRAM budget of the detection-channel pass. Per Gaussian and per
# channel of a chunk the pass holds the gradient and the in-mask fraction
# in float32 and four boolean masks — 12 bytes that live until the chunk's
# rows are extracted — and the calibration loop slices a class's channels
# on top (the fraction copy, two masks, the histogram gathers): a chunk
# of one class doubles the footprint. The scene is rendered whole (the
# volume scopes eligibility, not the rasterisation), so the number is
# the file's Gaussian count, not the box's. Measured on a test scene
# (2.08 M Gaussians, 32gb profile): ~12 B measured on mixed-class chunks.
DET_CHANNEL_BYTES = 24
# Left free beside the chunk: gsplat's own per-view buffers, the desktop
# and the browser's swings on a shared card.
DET_CHANNEL_RESERVE_GB = 1.0
# The chunk sizes offered: gsplat renders c and c + 1 channels (the first
# chunk carries the all-ones total) without padding for these; any other
# count is padded up to the next power of two and costs as much.
DET_CHANNEL_LADDER = (256, 128, 64, 32, 16, 8, 4, 2, 1)

# The class pass: per Gaussian and per class, the class matrix
# (float32) and, per view, the render channels and their gradient
# (float32 each) — 12 bytes; plus the total channel and the all-ones
# channel's pair. It is not chunked: the whole vocabulary is rasterised
# at once, so the vocabulary is what this pass scales with.
CLASS_PASS_BYTES = 12
CLASS_PASS_FIXED = 4 + 8      # T, and the all-ones channel with its gradient
# What the server and the scene's own tensors hold on the card before a
# lift starts — the dialog's estimate is made before any of it exists.
CARD_OVERHEAD_GB = 2.5


def class_capacity(n: int, free_bytes: float) -> int:
    """How many classes the class pass fits for `n` Gaussians with this
    much of the card free — the number the New scene dialog states
    and the Vocabulary gate can be held against."""
    usable = free_bytes - DET_CHANNEL_RESERVE_GB * 2**30 - n * CLASS_PASS_FIXED
    return max(int(usable // (n * CLASS_PASS_BYTES)), 0)


def scene_capacity(n: int, card_bytes: float) -> dict:
    """The dialog's line, from the file's Gaussian count and the card's
    size alone: the classes the lift fits, and what one class costs."""
    free = card_bytes - CARD_OVERHEAD_GB * 2**30
    return dict(gaussians=n, card_gb=round(card_bytes / 2**30, 1),
                classes_fit=class_capacity(n, free),
                per_class_mb=round(n * CLASS_PASS_BYTES / 2**20, 1),
                scene_gb=round(n * 44 / 2**30, 2))   # means, quats, scales, opacity


def class_pass_budget(n: int, n_classes: int) -> dict:
    """The class pass's estimate against the card's free memory at the
    moment of the call; a Refusal before the pass when it cannot fit
   , naming how many classes would."""
    free_dev, total = torch.cuda.mem_get_info()
    free = free_dev + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    need = n * (CLASS_PASS_BYTES * n_classes + CLASS_PASS_FIXED)
    usable = free - DET_CHANNEL_RESERVE_GB * 2**30
    rec = dict(n_gaussians=n, classes=n_classes,
               bytes_per_class=CLASS_PASS_BYTES,
               per_class_mb=round(n * CLASS_PASS_BYTES / 2**20, 1),
               estimate_gb=round(need / 2**30, 2),
               free_gb=round(free / 2**30, 2), card_gb=round(total / 2**30, 2),
               classes_fit=class_capacity(n, free))
    if need > usable:
        raise Refusal(
            f"The lift's class pass cannot fit on the card: {n:,} Gaussians "
            f"x {n_classes} classes need {rec['estimate_gb']:.1f} GB (the "
            f"class matrix and one view's channels), and {rec['free_gb']:.1f} "
            f"GB of {rec['card_gb']:.1f} is free. This card fits about "
            f"{rec['classes_fit']} classes for this scene: shorten the "
            f"vocabulary at the Vocabulary gate (each class costs "
            f"{rec['per_class_mb']:.0f} MB here), close other programs using "
            f"the GPU, or use a card with more memory. The lift renders the "
            f"whole file, so a smaller volume does not lower this.",
            gate="vocabulary")
    return rec


def det_channel_budget(setting, n: int, max_dets_view: int) -> dict:
    """The chunk of detection channels this pass may rasterise at once,
    from the card's free memory at the moment of the call — after the
    class pass, so the class matrices already hold their share.
    `setting` is lift.det_channel_chunk: "auto" (the largest rung of the
    ladder whose estimate fits) or a number that pins it. Either way the
    record says what the estimate was; a chunk that cannot fit is a
    Refusal before the pass, not an out-of-memory in its middle."""
    free_dev, total = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    free = free_dev + cached            # the allocator's cache is ours too
    usable = free - DET_CHANNEL_RESERVE_GB * 2**30
    per_channel = n * DET_CHANNEL_BYTES
    fits = [c for c in DET_CHANNEL_LADDER if per_channel * (c + 1) <= usable]
    rec = dict(setting=setting, n_gaussians=n,
               alloc_before_gb=round(torch.cuda.memory_allocated() / 2**30, 2),
               bytes_per_channel=DET_CHANNEL_BYTES,
               per_channel_mb=round(per_channel / 2**20, 1),
               card_gb=round(total / 2**30, 2), free_gb=round(free / 2**30, 2),
               reserve_gb=DET_CHANNEL_RESERVE_GB,
               usable_gb=round(max(usable, 0) / 2**30, 2),
               max_dets_per_view=max_dets_view)
    auto = setting in (None, "auto")
    chunk = fits[0] if (auto and fits) else (None if auto else int(setting))
    if chunk is not None:
        rec["chunk"] = chunk
        rec["estimate_gb"] = round(per_channel * (chunk + 1) / 2**30, 2)
        rec["source"] = "auto" if auto else "pinned"
    if chunk is None or per_channel * (chunk + 1) > usable:
        need = per_channel * ((chunk or 1) + 1) / 2**30
        what = (f"a chunk of {chunk} detection channels (lift.det_channel_chunk "
                f"pins it)" if chunk else "even one detection channel")
        hint = (f" The largest chunk that fits is {fits[0]}; \"auto\" would "
                f"choose it." if (fits and chunk) else
                " Close other programs using the GPU, or run this scene "
                "on a card with more memory; the lift renders the whole "
                "file, so a smaller volume does not lower this.")
        raise Refusal(
            f"The lift cannot fit {what} on the card: {n:,} Gaussians need "
            f"{rec['per_channel_mb']:.0f} MB per channel, {need:.1f} GB for "
            f"the pass, and {rec['free_gb']:.1f} GB of {rec['card_gb']:.1f} "
            f"is free ({DET_CHANNEL_RESERVE_GB:.0f} GB of it kept for the "
            f"pass's own buffers).{hint}")
    return rec


def _effective_threshold(concept: str, cfg: dict, source: str = "text") -> float:
    """Per-class gate. Exemplar detections gate against their OWN
    calibrated threshold — exemplar-similarity is a different scale and is
    NEVER compared with text presence confidence."""
    d = cfg["detect"]
    if source.startswith("exemplar:"):
        return d["exemplar_threshold_overrides"].get(
            concept, d["exemplar_threshold"])
    return d["presence_threshold_overrides"].get(
        concept, d["presence_threshold"])


def _effective_area_cap(concept: str, cfg: dict) -> float | None:
    """Per-class mask-area ceiling (frame fraction): SAM 3
    sometimes masks a whole furniture piece as a small-object class
    (observed: a rolling workbench masked as "cardboard box"), absorbing
    the real instances that sit on it. None = uncapped. A class listed here
    is EXEMPT from the max_area_factor rule — the absolute cap replaces it."""
    return cfg["detect"]["max_area_overrides"].get(concept)


def _channel(source: str) -> str:
    """Median grouping for the factor cap: text and exemplar masks never
 share a median — propagated tracker masks run ~2x looser (Rule C)."""
    return "exemplar" if source.startswith("exemplar:") else "text"


def _factor_medians(rows: list[dict], cfg: dict) -> dict:
    """Per-(class, channel) median mask area among above-threshold dets, for
    detect.max_area_factor (mechanism 1). Classes with fewer than
    max_area_min_dets dets have no meaningful median and get no entry (the
    cap does not apply); classes in max_area_overrides are handled by the
    absolute cap instead and are skipped here."""
    if not cfg["detect"].get("max_area_factor"):
        return {}
    by_cc: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if _effective_area_cap(r["concept"], cfg) is not None:
            continue
        by_cc[(r["concept"], _channel(r["source"]))].append(r["area_frac"])
    return {k: float(np.median(v)) for k, v in by_cc.items()
            if len(v) >= cfg["detect"]["max_area_min_dets"]}


# The dedup's masks are kept PACKED (one bit a pixel, np.packbits) and an
# intersection is a popcount over the AND of two packed rows — ~300x
# cheaper than the boolean AND at full resolution (0.7 ms a pair on 1024²)
# that made the step seven minutes on a test scene, and 8x less RAM
# for a view's mask list. numpy 1.26 has no bitwise_count; a 256-entry
# table does it.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint32)


def _packed_inter(a: np.ndarray, b: np.ndarray) -> int:
    """Pixels in both masks, the masks packed."""
    return int(_POPCOUNT[a & b].sum())


def _load_detections(stage2: Path, cfg: dict) -> tuple[list[str], dict, dict]:
    """Detections above the per-class effective threshold, grouped per view.

    With lift.mask_dedup on, same-view CROSS-CLASS mask containment is
    resolved before anything downstream sees the detections (the
    containment mechanism; thresholds calibrated on a lab-interior containment audit):
      Rule A (mutual, min IoS >= mask_dedup_mutual_ios): two class names on
        one mask = one object; keep the higher-scoring det, drop the other
        (synonym crossfire: workbench/shelf unit, tripod/light stand).
      Rule B (part-of): a class pair showing asymmetric containment
        (IoS >= partof_ios, not mutual) in >= partof_min_views views has its
        contained dets suppressed — the whole object keeps its parts (laptop
        keeps screen-as-monitor and its keyboard). Pair consistency across
        views is required so one loose mask cannot delete a real object.
    All decisions are returned in the suppression dict for audit.
    """
    manifest = json.loads((stage2 / "probe_manifest.json").read_text())
    negatives = set(manifest.get("negatives", []))
    classes = [p for p in manifest["prompts"] if p not in negatives]
    # Exemplar concepts are extra classes, additively.
    classes += [c for c in manifest.get("exemplar_concepts", [])
                if c not in classes]
    kept_rows: list[dict] = []
    with open(stage2 / "scores.csv") as f:
        for r in csv.DictReader(f):
            c = r["concept"]
            src = r.get("source") or "text"
            if c in negatives or float(r["score"]) < _effective_threshold(
                    c, cfg, src):
                continue
            kept_rows.append(dict(
                frame_idx=int(r["frame_idx"]), concept=c,
                det_idx=int(r["det_idx"]), score=float(r["score"]),
                area_frac=float(r["area_frac"]), source=src))

    # Area gates (mechanism 1): a class in max_area_overrides is
    # bounded by that ABSOLUTE frame fraction; every other class is bounded
    # by max_area_factor x its (class, channel) median area — the absurdity
    # guard for whole-furniture-as-small-object masks. Factor kills should
    # be a NEVER-EVENT on a calibrated scene: each one is logged loudly and
    # audited so the driver can surface them at the next gate.
    factor = cfg["detect"].get("max_area_factor")
    medians = _factor_medians(kept_rows, cfg)
    per_view: dict[int, list[dict]] = defaultdict(list)
    area_capped: dict[str, int] = defaultdict(int)
    factor_killed: list[dict] = []
    for r in kept_rows:
        c = r["concept"]
        cap = _effective_area_cap(c, cfg)
        if cap is not None:
            if r["area_frac"] > cap:
                area_capped[c] += 1
                continue
        elif factor:
            med = medians.get((c, _channel(r["source"])))
            if med is not None and r["area_frac"] > factor * med:
                factor_killed.append(dict(
                    concept=c, channel=_channel(r["source"]),
                    frame=r["frame_idx"], score=r["score"],
                    area_frac=r["area_frac"], median_area=round(med, 5),
                    ratio=round(r["area_frac"] / med, 1)))
                # Said in the operator's terms: what was dropped,
                # how far off it was, and that the run PAUSES for the
                # typed acknowledgment once the lift is done — the old
                # line read as the last thing before a hang.
                log.warning(
                    "AREA-FACTOR KILL: %r on frame %04d dropped: its mask "
                    "covers %.1f%% of the frame, %.0fx the class's usual "
                    "%.2f%% (cap %sx, score %.3f). A never-event on a "
                    "calibrated scene: the run pauses for your "
                    "acknowledgment after the lift.",
                    c, r["frame_idx"], 100 * r["area_frac"],
                    r["area_frac"] / med, 100 * med, factor, r["score"])
                continue
        per_view[r["frame_idx"]].append(
            dict(concept=c, det_idx=r["det_idx"], score=r["score"],
                 source=r["source"]))
    for c, n in area_capped.items():
        log.info("area cap (%s <= %.2f): dropped %d oversized det(s)",
                 c, _effective_area_cap(c, cfg), n)

    lcfg = cfg["lift"]
    suppression: dict = dict(enabled=bool(lcfg.get("mask_dedup", False)),
                             area_capped_by_class=dict(area_capped),
                             factor_cap=dict(
                                 factor=factor,
                                 min_dets=cfg["detect"]["max_area_min_dets"],
                                 killed=factor_killed),
                             mutual=[], partof_pairs=[], partof_dropped=0,
                             partof_dropped_by_class={},
                             carved=0, carved_by_class={},
                             cross_source=[])
    if not suppression["enabled"]:
        return classes, per_view, suppression

    # Specific-over-generic policy (found on an exterior scene), generalized to a
    # PRIORITY ORDERING: `specific` is an ORDERED
    # list — earlier = higher priority. In any containment pair between two
    # RANKED classes the higher-priority det survives; generic classes rank
    # below every specific one (so the original behavior is unchanged), and
    # specific-vs-specific pairs now resolve by list order instead of dying
    # to the standard rules (the tall-cross case: cross element inside
    # 'grave pedestal' whole-monument masks, 11-view quorum). Unlisted
    # classes have no rank — standard score/containment rules apply.
    sog = lcfg.get("specific_over_generic") or {}
    generic = set(sog.get("generic") or [])
    spec_list = list(sog.get("specific") or [])

    def _rank(c: str) -> int | None:
        if c in generic:
            return -1
        if c in spec_list:
            return len(spec_list) - spec_list.index(c)
        return None

    mutual_t = lcfg["mask_dedup_mutual_ios"]
    partof_t = lcfg["partof_ios"]
    relations = []   # (frame_idx, dets, mutual-drop set, [(part_i, whole_j)])
    # Said before and during: this is the lift's slowest step —
    # every kept mask of a view loaded, every pair compared at full
    # resolution — and it ran seven silent minutes on a test scene (37
    # views, ~6,900 detections above the presence thresholds, ~680,000
    # pairs). One line per view in the "k/n" form the header chip reads.
    n_views = len(per_view)
    log.info("mask dedup: %d views, %d masks; comparing every pair within "
             "each view", n_views, sum(len(d) for d in per_view.values()))
    for vi, (fidx, dets) in enumerate(per_view.items()):
        masks, areas, boxes = [], [], []
        for d in dets:
            mp = _mask_path(stage2, fidx, d["concept"], d["det_idx"])
            if not mp.exists():
                raise Refusal(
                    f"mask missing: {mp}; stage2 must be run with "
                    f"overlay_threshold <= the lowest effective presence threshold")
            m = np.asarray(Image.open(mp)) > 127
            masks.append(np.packbits(m))
            areas.append(int(m.sum()))
            # the mask's own box (not the probe's row: one source of truth
            # for the test below) — x1, y1, x2, y2, exclusive
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            boxes.append((int(cols[0]), int(rows[0]), int(cols[-1]) + 1,
                          int(rows[-1]) + 1) if rows.size else (0, 0, 0, 0))
        drop: set[int] = set()
        rel_f: list[tuple[int, int]] = []
        for i, a in enumerate(dets):
            if not areas[i]:
                continue
            ax1, ay1, ax2, ay2 = boxes[i]
            for j in range(i + 1, len(dets)):
                b = dets[j]
                if not areas[j]:
                    continue
                # Disjoint boxes: an empty intersection, which every test
                # below reads as "nothing" — so the answer is exact and the
                # 87 % of pairs that never touch (a test scene) cost nothing.
                bx1, by1, bx2, by2 = boxes[j]
                if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                    continue
                if a["concept"] == b["concept"]:
                    # Rule C: the SAME class arriving via both the text
                    # and the exemplar channel is one object seen twice.
                    # One-sided containment triggers it — propagated tracker
                    # masks run ~2x looser than detector masks, so the mutual
                    # test never fires (validation: text IoS 1.00 inside the
                    # exemplar mask at exemplar IoS 0.4-0.5). The TEXT det is
                    # kept: above its own calibrated gate its detector mask
                    # is the sharper one; exemplar rescues matter where text
                    # is silent.
                    sa = a.get("source", "text")
                    sb = b.get("source", "text")
                    if sa == sb:
                        continue
                    inter = _packed_inter(masks[i], masks[j])
                    if inter and max(inter / areas[i],
                                     inter / areas[j]) >= mutual_t:
                        lose = i if sa != "text" else j
                        drop.add(lose)
                        suppression["cross_source"].append(dict(
                            frame=fidx, concept=a["concept"],
                            kept="text",
                            ios=round(max(inter / areas[i],
                                          inter / areas[j]), 3)))
                    continue
                inter = _packed_inter(masks[i], masks[j])
                if not inter:
                    continue
                ios_ij, ios_ji = inter / areas[i], inter / areas[j]
                if min(ios_ij, ios_ji) >= mutual_t:
                    ra, rb = _rank(a["concept"]), _rank(b["concept"])
                    if ra is not None and rb is not None and ra != rb:
                        lose, policy = (j if ra > rb else i), True
                    else:
                        lose, policy = (i if a["score"] <= b["score"] else j), False
                    drop.add(lose)
                    suppression["mutual"].append(dict(
                        frame=fidx, dropped=dets[lose]["concept"],
                        kept=dets[i + j - lose]["concept"],
                        ios=round(min(ios_ij, ios_ji), 3),
                        **({"policy": "specific_over_generic"} if policy else {})))
                else:
                    if ios_ij >= partof_t:
                        rel_f.append((i, j))
                    if ios_ji >= partof_t:
                        rel_f.append((j, i))
        relations.append((fidx, dets, drop, rel_f))

    pair_frames: dict[tuple[str, str], set] = defaultdict(set)
    for fidx, dets, drop, rel_f in relations:
        for i, j in rel_f:
            if i not in drop and j not in drop:
                pair_frames[(dets[i]["concept"], dets[j]["concept"])].add(fidx)
    adopted = {p for p, fs in pair_frames.items()
               if len(fs) >= lcfg["partof_min_views"]}
    suppression["partof_pairs"] = [
        dict(part=p[0], whole=p[1], views=len(pair_frames[p]))
        for p in sorted(adopted)]

    for fidx, dets, drop, rel_f in relations:
        for i, j in rel_f:
            if ((dets[i]["concept"], dets[j]["concept"]) in adopted
                    and i not in drop and j not in drop):
                rp, rw = _rank(dets[i]["concept"]), _rank(dets[j]["concept"])
                if rp is not None and rw is not None and rp > rw:
                    # Keep the higher-priority part; CARVE its pixels from
                    # the lower-priority whole at accumulation —
                    # double-claimed pixels destabilize the per-gaussian
                    # argmax, and disjoint body+element is the semantically
                    # correct export. (Covers specific-in-generic AND
                    # specific-vs-specific ordering.)
                    dets[j].setdefault("carve", []).append(
                        (dets[i]["concept"], dets[i]["det_idx"]))
                    suppression["carved"] += 1
                    c = dets[i]["concept"]
                    suppression["carved_by_class"][c] = \
                        suppression["carved_by_class"].get(c, 0) + 1
                    continue
                drop.add(i)
                suppression["partof_dropped"] += 1
                c = dets[i]["concept"]   # per-class funnel accounting
                suppression["partof_dropped_by_class"][c] = \
                    suppression["partof_dropped_by_class"].get(c, 0) + 1
        if drop:
            per_view[fidx] = [d for k, d in enumerate(dets) if k not in drop]
        log.info("mask dedup view %d/%d (frame %04d): %d masks, %d dropped "
                 "so far", vi + 1, n_views, fidx, len(dets),
                 len(suppression["mutual"]) + suppression["partof_dropped"]
                 + suppression["carved"] + len(suppression["cross_source"]))
    if (suppression["mutual"] or suppression["partof_dropped"]
            or suppression["carved"] or suppression["cross_source"]):
        log.info("mask dedup: %d mutual drops, %d part-of drops (%d pairs), "
                 "%d specific-over-generic carves, %d cross-source collapses",
                 len(suppression["mutual"]), suppression["partof_dropped"],
                 len(adopted), suppression["carved"],
                 len(suppression["cross_source"]))
    return classes, per_view, suppression


def _mask_path(stage2: Path, frame_idx: int, concept: str, det_idx: int) -> Path:
    return (stage2 / "masks" / f"frame_{frame_idx:04d}" /
            f"{concept.replace(' ', '_')}_{det_idx}.png")


def _kmeans(x: torch.Tensor, k: int, iters: int = 25) -> torch.Tensor:
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:k]
    centers = x[idx].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)
        for j in range(k):
            sel = assign == j
            if sel.any():
                centers[j] = x[sel].mean(dim=0)
    return assign


def detection_params(cfg: dict) -> dict:
    """The detect-side values that decide WHICH stage-2 detections a stage
    retains (`_load_detections`): thresholds and area gates. Shared by the
    lift and export fingerprints — both stages read the detections
    through the same function, so both caches depend on the same keys."""
    d = cfg["detect"]
    return dict(
        presence_threshold=d["presence_threshold"],
        presence_threshold_overrides=d.get("presence_threshold_overrides") or {},
        exemplar_threshold=d["exemplar_threshold"],
        exemplar_threshold_overrides=d.get("exemplar_threshold_overrides") or {},
        max_area_overrides=d.get("max_area_overrides") or {},
        max_area_factor=d.get("max_area_factor"),
        max_area_min_dets=d.get("max_area_min_dets"))


def lift_params(cfg: dict) -> dict:
    """Effective decision parameters of the lift — every value that shapes
    stage-3 output — recorded in the manifest and compared BY VALUE by the
    cache check (`instance_voxel` was the only one fingerprinted;
    the floors, the scope margin, the gammas and the split/merge knobs all
    shaped the lift blind). `refine` and `track_affinity_weight` left the
    record with their branches — every earlier stage-3 manifest reads
    stale once, by design."""
    L = cfg["lift"]
    keys = ("background_gamma", "background_gamma_overrides", "min_opacity",
            "instance_voxel", "min_instance_gaussians",
            "count_stability_views", "scope_margin", "split_mode",
            "split_views", "split_correspondence", "corr_min_frac",
            "corr_mask_iou_max", "split_require_distinct",
            "same_class_merge", "merge_ios", "merge_dist_frac",
            "specific_over_generic", "mask_dedup", "mask_dedup_mutual_ios",
            "partof_ios", "partof_min_views")
    p = {k: L.get(k) for k in keys}
    # the instancing mode, its schema and every threshold that shapes
    # the tracks (det_channel_chunk changes nothing numerically; a chunk
    # boundary that changes the result is a bug the fingerprint exposes)
    from .instancing import INSTANCING_SCHEMA
    p["instancing"] = L.get("instancing", "tracks")
    p["instancing_schema"] = INSTANCING_SCHEMA
    for k in ("support_min_frac", "track_affinity_min",
              "track_min_support_ios", "cannot_link_ios_max",
              "det_channel_chunk"):
        p[k] = L.get(k)
    p["near_plane"] = cfg["render"]["near_plane"]
    p["far_plane"] = cfg["render"]["far_plane"]
    p["detect"] = detection_params(cfg)
    return p


def run_lift(scene_path: str, workdir: str, cfg: dict,
             force: bool = False, should_cancel=None,
             dump_dir: str | None = None) -> dict:
    """dump_dir: debug-only — write the raw support rows and per-class
    affinity matrices there (the calibration bank's instrument, not a
    product knob)."""
    from .sequencing import (check_cancel, fingerprint, stale_params,
                             stale_upstream, upstream_fingerprint)
    from .units import scale_from_stage1, scene_units_cfg
    out = Path(workdir) / "stage3"
    manifest_path = out / "manifest.json"
    params = lift_params(cfg)                  # the profile's METRES
    # the volume scopes the labels, read straight from disk: its content
    # is a decision parameter (the cache used to be blind to it and relied
    # on the session remembering that the volume gate was re-entered)
    params["volume"] = fingerprint(resolve_volume_path(workdir, cfg))
    # The factor is a decision parameter: the effective instance_voxel and
    # scope_margin depend on it. It is the one stage 1 ran at —
    # recorded, or measured with the ruler — read from its manifest
    # so the run uses one number; recorded beside the metre values.
    s1_manifest_path = Path(workdir) / "stage1" / "manifest.json"
    scale_rec = scale_from_stage1(
        cfg, json.loads(s1_manifest_path.read_text())
        if s1_manifest_path.exists() else None)
    params["scale_m_per_unit"] = scale_rec.value
    params["scale_source"] = scale_rec.source
    if manifest_path.exists() and not force:
        cached = json.loads(manifest_path.read_text())
        # the cache is fresh only if it was produced from the stage-2
        # manifest now on disk AND with these decision parameters (by
        # value; instance_voxel used to be the only one compared, after
        # an earlier fix — the rest shaped the lift blind).
        if (not stale_manual_views(cached, workdir, "stage3")
                and not stale_exemplars(cached, workdir, "stage3")
                and not stale_upstream(cached, workdir, "stage3",
                                       "upstream_stage2",
                                       "stage2/probe_manifest.json")
                and not stale_params(cached, params, "stage3")):
            log.info("stage3 cached at %s (inputs unchanged)", out)
            return cached
    out.mkdir(parents=True, exist_ok=True)
    mode = cfg["lift"].get("instancing", "tracks")
    if mode not in ("tracks", "connectivity"):
        raise Refusal(f"lift.instancing must be 'tracks' or 'connectivity', "
                      f"got {mode!r}")

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    stage1 = Path(workdir) / "stage1"
    stage2 = Path(workdir) / "stage2"
    cams = json.loads((stage1 / "cameras.json").read_text())
    classes, per_view, suppression = _load_detections(stage2, cfg)
    cls_id = {c: i for i, c in enumerate(classes)}

    # Per-frame intrinsics (manual views carry their own fov/aspect; auto
    # frames fall back to the global block).
    def frame_K(fr: dict) -> tuple:
        fxf, fyf, cxf, cyf, wf, hf = frame_intrinsics(fr, cams)
        Kt = torch.tensor([[fxf, 0, cxf], [0, fyf, cyf], [0, 0, 1.0]],
                          device="cuda")
        return Kt, wf, hf

    scene = load_scene(scene_path, cfg)
    n = scene.num_gaussians
    device = "cuda"
    # the file's own arrays render under the file-frame poses of
    # cameras.json; the volume scoping below reads the scene frame
    means = torch.as_tensor(scene.means_file, device=device)
    quats = torch.as_tensor(scene.quats_file, device=device)
    scales = torch.as_tensor(scene.scales, device=device)
    opacities = torch.as_tensor(scene.opacities, device=device)

    vol_path = resolve_volume_path(workdir, cfg)
    boxes = load_volume(vol_path) if vol_path.exists() else None
    cfg, units = scene_units_cfg(cfg, scale_rec.value)   # knobs -> units, once
    # Soft boundary for label scoping: the trimmed
    # volume answers "where cameras go"; wall-mounted objects sit ON that
    # boundary and must not be truncated by it.
    margin = cfg["lift"]["scope_margin"]
    vol_mask = (torch.as_tensor(in_volume(scene.means, boxes, margin=margin),
                                device=device)
                if boxes is not None else torch.ones(n, dtype=torch.bool, device=device))

    # --- contribution accumulation (one backward per view) --------------------
    try:
        import gsplat
    except ImportError as e:
        raise missing_dependency("gsplat", e) from e

    # the pass scales with the vocabulary; said, and refused before
    # it starts when it cannot fit
    class_budget = class_pass_budget(n, len(classes))
    log.info("class pass: %s Gaussians x %d classes = %.1f GB estimated "
             "(%.0f MB per class); %.1f GB free of %.1f on the card; this "
             "card fits about %d classes for this scene", f"{n:,}",
             len(classes), class_budget["estimate_gb"],
             class_budget["per_class_mb"], class_budget["free_gb"],
             class_budget["card_gb"], class_budget["classes_fit"])
    A = torch.zeros(n, len(classes), device=device)   # in-mask contribution
    T = torch.zeros(n, device=device)                 # total contribution
    n_masks_used = 0
    t_accum = time.perf_counter()
    for fr in cams["frames"]:
        check_cancel(should_cancel)   # between per-frame accumulation
        dets = per_view.get(fr["frame_idx"], [])
        present = sorted({d["concept"] for d in dets})
        if not present:
            continue
        K, w, h = frame_K(fr)
        stack = np.zeros((h, w, len(present) + 1), dtype=np.float32)
        stack[..., -1] = 1.0
        for d in dets:
            mp = _mask_path(stage2, fr["frame_idx"], d["concept"], d["det_idx"])
            if not mp.exists():
                raise Refusal(
                    f"mask missing: {mp}; stage2 must be run with "
                    f"overlay_threshold <= the lowest effective presence threshold")
            m = np.asarray(Image.open(mp), dtype=np.float32) / 255.0
            # specific-over-generic carve: the kept part's
            # pixels leave the generic whole so no pixel is double-claimed.
            # Split/correspondence machinery still sees raw per-class masks.
            for pc, pdi in d.get("carve", []):
                pm = np.asarray(Image.open(_mask_path(
                    stage2, fr["frame_idx"], pc, pdi)), dtype=np.float32) / 255.0
                m = m * (1.0 - pm)
            stack[..., present.index(d["concept"])] = np.maximum(
                stack[..., present.index(d["concept"])], m)
            n_masks_used += 1
        target = torch.as_tensor(stack, device=device)

        colors = torch.ones(n, len(present) + 1, device=device, requires_grad=True)
        viewmat = torch.linalg.inv(torch.tensor(
            np.array(fr["c2w"]), dtype=torch.float32, device=device))
        render, _, _ = gsplat.rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat[None], Ks=K[None], width=w, height=h,
            near_plane=cfg["render"]["near_plane"],
            far_plane=cfg["render"]["far_plane"])
        (render[0] * target).sum().backward()
        grad = colors.grad
        for j, c in enumerate(present):
            A[:, cls_id[c]] += grad[:, j]
        T += grad[:, -1]
    t_accum = time.perf_counter() - t_accum

    # --- closed-form assignment ------------------------------------------------
    # Per-class gamma overrides: a class detected in few
    # views accumulates T from every view but A only where masked — the
    # global bias then rejects gaussians whose argmax is clearly that class
    # (workbench body: median needed-gamma 0.41 vs global 1.0, audited).
    # Same precedent as detect.presence_threshold_overrides: scene-profile
    # values calibrated from the audit, empty by default.
    gamma = cfg["lift"]["background_gamma"]
    overrides = cfg["lift"].get("background_gamma_overrides") or {}
    gam = torch.full((len(classes),), float(gamma), device=device)
    for cname, gv in overrides.items():
        if cname in cls_id:
            gam[cls_id[cname]] = float(gv)
    best_val, best_cls = A.max(dim=1)
    assigned = best_val > gam[best_cls] * (T - best_val)
    labels = torch.where(assigned, best_cls, torch.full_like(best_cls, -1))

    # Through-door / volume-scoping accounting BEFORE scoping:
    oov = {c: int(((labels == cls_id[c]) & ~vol_mask).sum()) for c in classes}
    labels[~vol_mask] = -1

    # --- instance separation with 2D-count sanity ------------------------------
    lcfg = cfg["lift"]
    instances = torch.full((n,), -1, dtype=torch.int32, device=device)
    audit: dict[str, dict] = {"_mask_suppression": suppression,
                              "_scope_margin": margin}
    next_inst = 0
    means_np = scene.means_file      # projections against file-frame poses

    # Per-detection contribution features for mask-guided splits: when the 2D-count check demands a split, partition by WHICH
    # 2D instance each Gaussian's rendering contribution came from — the same
    # gradient trick as the class accumulation, with one channel per DETECTION
    # of the class in views that see >= 2 of them. Gaussians of one physical
    # object co-activate the same det in every view, so k-means on the
    # concatenated per-view det profiles splits along evidence boundaries, not
    # geometry (bare coordinate k-means cut merged panels arbitrarily).
    feat_cache: dict[str, tuple] = {}

    def _perdet_features(c: str):
        cand = []
        for fr in cams["frames"]:
            ds = [d for d in per_view.get(fr["frame_idx"], [])
                  if d["concept"] == c]
            if len(ds) >= 2:
                cand.append((fr, ds))
        cand.sort(key=lambda t: (-len(t[1]), -sum(d["score"] for d in t[1])))
        cand = cand[:lcfg["split_views"]]
        if not cand:
            return None, []
        feats, used = [], []
        for fr, ds in cand:
            K, w, h = frame_K(fr)
            stack = np.zeros((h, w, len(ds)), dtype=np.float32)
            for j, d in enumerate(ds):
                mp = _mask_path(stage2, fr["frame_idx"], d["concept"], d["det_idx"])
                stack[..., j] = np.asarray(Image.open(mp), np.float32) / 255.0
            target = torch.as_tensor(stack, device=device)
            colors = torch.ones(n, len(ds), device=device, requires_grad=True)
            viewmat = torch.linalg.inv(torch.tensor(
                np.array(fr["c2w"]), dtype=torch.float32, device=device))
            render, _, _ = gsplat.rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmat[None], Ks=K[None],
                width=w, height=h, near_plane=cfg["render"]["near_plane"],
                far_plane=cfg["render"]["far_plane"])
            (render[0] * target).sum().backward()
            g = colors.grad.clamp_min(0)
            g = g / g.sum(dim=1, keepdim=True).clamp_min(1e-12)  # det profile
            feats.append(g.cpu().numpy())
            used.append(fr["frame_idx"])
        return np.concatenate(feats, axis=1), used

    def _nearest_fill(pts: np.ndarray, seed_assign: np.ndarray,
                      has_seed: np.ndarray) -> np.ndarray:
        """Assign seedless points to their nearest seeded point's cluster."""
        full = np.empty(len(pts), dtype=np.int64)
        full[has_seed] = seed_assign
        rest = np.flatnonzero(~has_seed)
        if len(rest):
            src = torch.as_tensor(pts[has_seed], device=device)
            lab = torch.as_tensor(seed_assign, device=device)
            q = torch.as_tensor(pts[rest], device=device)
            for s in range(0, len(rest), 8192):
                dm = torch.cdist(q[s:s + 8192], src)
                full[rest[s:s + 8192]] = lab[dm.argmin(dim=1)].cpu().numpy()
        return full

    def _feature_bisect(c: str, g: np.ndarray):
        """Two-way split of gaussian set g along per-det evidence boundaries."""
        if c not in feat_cache:
            feat_cache[c] = _perdet_features(c)
        F, fviews = feat_cache[c]
        if F is None:
            return None, fviews
        Fb = F[g]
        nz = Fb.sum(axis=1) > 1e-6
        if nz.sum() < 2:
            return None, fviews
        a_nz = _kmeans(torch.as_tensor(Fb[nz]), 2).numpy()
        if len(np.unique(a_nz)) < 2:
            return None, fviews
        assign = _nearest_fill(means_np[g], a_nz, nz)
        return [g[assign == j] for j in (0, 1)], fviews

    depth_cache: dict[int, np.ndarray] = {}

    def _depth(fidx: int) -> np.ndarray:
        if fidx not in depth_cache:
            depth_cache[fidx] = np.load(
                stage1 / f"depth/depth_{fidx:04d}.npy").astype(np.float32)
        return depth_cache[fidx]

    # The visibility test's lengths, in scene units (the resolved config):
    # the near plane and the coverage slack the render counted with.
    near_units = cfg["render"]["near_plane"]
    ccfg = cfg["render"]["coverage"]

    def _project_visible(pts: np.ndarray, fr: dict):
        """Depth-tested pixel coords of pts in view fr, or None if < 50 land."""
        proj = project_visible(pts, fr, cams, lambda: _depth(fr["frame_idx"]),
                               near=near_units, tol_rel=ccfg["depth_tol_rel"],
                               tol_abs=ccfg["depth_tol"], min_points=50)
        if proj is None:
            return None
        return proj.u[proj.vis], proj.v[proj.vis]

    def _class_multi(c: str):
        """Multi-detection views of class c + their masks (evidence views)."""
        multi = []
        for fr in cams["frames"]:
            ds = [d for d in per_view.get(fr["frame_idx"], [])
                  if d["concept"] == c]
            if len(ds) >= 2:
                multi.append((fr, ds))
        masks_v = {fr["frame_idx"]: [
            np.asarray(Image.open(_mask_path(
                stage2, fr["frame_idx"], d["concept"], d["det_idx"]))) > 127
            for d in ds] for fr, ds in multi}
        return multi, masks_v

    def _pairwise_distinct(gA: np.ndarray, gB: np.ndarray,
                           multi: list, masks_v: dict) -> tuple[int, int]:
        """Votes that gA and gB are two 2D objects: views where each set's
        visible projection lands dominantly (>= corr_min_frac) in a DIFFERENT
        near-disjoint mask. Returns (votes, views_checked)."""
        votes = checked = 0
        for fr, ds in multi:
            fidx = fr["frame_idx"]
            pa = _project_visible(means_np[gA], fr)
            pb = _project_visible(means_np[gB], fr)
            if pa is None or pb is None:
                continue
            checked += 1
            fa = [float(m[pa[1], pa[0]].mean()) for m in masks_v[fidx]]
            fb = [float(m[pb[1], pb[0]].mean()) for m in masks_v[fidx]]
            ia, ib = int(np.argmax(fa)), int(np.argmax(fb))
            if (ia == ib or fa[ia] < lcfg["corr_min_frac"]
                    or fb[ib] < lcfg["corr_min_frac"]):
                continue
            ma, mb = masks_v[fidx][ia], masks_v[fidx][ib]
            iou = (ma & mb).sum() / max((ma | mb).sum(), 1)
            if iou <= lcfg["corr_mask_iou_max"]:
                votes += 1
        return votes, checked

    def _is_distinct(votes: int, checked: int) -> bool:
        """Quorum OR majority: co-visibility is scarce for well-separated
        objects (wall panels 2 m apart co-appear in ONE view), so 1/1 distinct
        views must count — while true stacked duplicates score 0-1 of MANY
        (plants: 0-1 of 14-16 checked; towel duplicate: 0 of 14)."""
        return (votes >= lcfg["count_stability_views"]
                or (checked > 0 and 2 * votes >= checked))

    # 2D-CORRESPONDENCE sanity (extends the 2D-count check):
    # the count check only catches GLOBAL undercounts — a merged pair hides
    # whenever other singletons keep the class total >= expected (the two
    # side-by-side desk monitors re-merged exactly this way once the
    # laptop-screen det was suppressed). Instead, test each 3D component
    # directly: if its depth-tested projection spans >= 2 near-disjoint
    # same-class masks (each holding >= corr_min_frac of visible points) in
    # >= count_stability_views views, it is two objects — split it along the
    # same per-det evidence boundaries as forced splits.
    def _merge_candidate(gA: np.ndarray, gB: np.ndarray) -> tuple:
        """Geometric stacked-duplicate signature: AABB overlap high
        relative to the smaller set, or centroids closer than a fraction of
        the smaller diagonal — AND centroids within that fraction of the
        LARGER diagonal as a hard backstop (thin coplanar slabs, e.g. wall
        panels, reach high AABB IoS with centroids metres apart).
        Returns (is_candidate, ios, dist)."""
        lo_a, hi_a = means_np[gA].min(0), means_np[gA].max(0)
        lo_b, hi_b = means_np[gB].min(0), means_np[gB].max(0)
        inter = np.prod(np.clip(np.minimum(hi_a, hi_b)
                                - np.maximum(lo_a, lo_b), 0, None))
        vmin = max(min(np.prod(hi_a - lo_a), np.prod(hi_b - lo_b)), 1e-9)
        ios = float(inter / vmin)
        dist = float(np.linalg.norm(means_np[gA].mean(0) - means_np[gB].mean(0)))
        dg_a = float(np.linalg.norm(hi_a - lo_a))
        dg_b = float(np.linalg.norm(hi_b - lo_b))
        frac = lcfg["merge_dist_frac"]
        cand = ((ios >= lcfg["merge_ios"] or dist <= frac * min(dg_a, dg_b))
                and dist <= frac * max(dg_a, dg_b))
        return cand, ios, dist

    # A split is KEPT unless the merge pass would undo it —
    # children that carry the stacked-duplicate signature (geometric merge
    # candidates) must be pairwise 2D-DISTINCT, else the bisection failed to
    # separate the evidence and recursing on such noise is what manufactured
    # 11 stacked plant fragments (18 exported vs 2D-expected 7) on a
    # mirror-heavy interior. Spatially separated children are legitimate
    # without a distinctness quorum: distinctness needs co-visibility, which
    # objects on different walls (that scene's 5 mirror surfaces) rarely have.
    def _split_effective(parts: list, multi: list, masks_v: dict,
                         entry: dict, mech: str, votes=None) -> bool:
        if not lcfg["split_require_distinct"]:
            return True
        cand, ios, dist = _merge_candidate(parts[0], parts[1])
        if not cand:
            return True
        dv, checked = _pairwise_distinct(parts[0], parts[1], multi, masks_v)
        if _is_distinct(dv, checked):
            return True
        rec = dict(result=f"{mech}_refused_ineffective", ios=round(ios, 3),
                   distinct_votes=dv, views_checked=checked)
        if votes is not None:
            rec["votes"] = votes
        entry["splits"].append(rec)
        return False

    def _correspondence_split(c: str, comp_list: list, entry: dict,
                              multi: list, masks_v: dict) -> list:
        if not multi:
            return comp_list
        queue, done = list(comp_list), []
        while queue:
            g = queue.pop()
            if len(g) < 2 * lcfg["min_instance_gaussians"]:
                done.append(g)
                continue
            votes = 0
            for fr, ds in multi:
                fidx = fr["frame_idx"]
                pv = _project_visible(means_np[g], fr)
                if pv is None:
                    continue
                u_vis, v_vis = pv
                fracs = sorted(
                    ((float(m[v_vis, u_vis].mean()), mi)
                     for mi, m in enumerate(masks_v[fidx])), reverse=True)
                (f1, m1), (f2, m2) = fracs[0], fracs[1]
                if f1 >= lcfg["corr_min_frac"] and f2 >= lcfg["corr_min_frac"]:
                    ma, mb = masks_v[fidx][m1], masks_v[fidx][m2]
                    iou = (ma & mb).sum() / max((ma | mb).sum(), 1)
                    if iou <= lcfg["corr_mask_iou_max"]:
                        votes += 1
            if votes < lcfg["count_stability_views"]:
                done.append(g)
                continue
            parts, fviews = _feature_bisect(c, g)
            if parts is None or min(len(p) for p in parts) < \
                    lcfg["min_instance_gaussians"]:
                entry["splits"].append(dict(result="corr_refused", votes=votes))
                done.append(g)
            elif not _split_effective(parts, multi, masks_v, entry,
                                      "corr", votes=votes):
                done.append(g)
            else:
                entry["splits"].append(dict(
                    split_gaussians=[len(p) for p in parts],
                    mechanism="correspondence", votes=votes, views=fviews))
                queue.extend(parts)
        return done

    # SAME-CLASS INSTANCE MERGE — the counterweight
    # the split-only sanity lacked ("3D count 18 > 2D-expected 7 (kept)").
    # Same-class components with high mutual overlap (box IoS, or centroid
    # distance small vs size — the stacked-duplicate signature) are merged
    # back UNLESS the pair is 2D-DISTINCT under _pairwise_distinct — the
    # same evidence standard the splits use, so split and merge cannot
    # disagree. Geometry alone CANNOT make this call: on one scene the
    # two real side-by-side display-case picture frames overlap at IoS 0.51
    # while plant duplicates sit at IoS 1.00 — no threshold separates them;
    # the 2D masks do.
    def _same_class_merge(comp_list: list, entry: dict,
                          multi: list, masks_v: dict) -> list:
        if len(comp_list) < 2 or not lcfg["same_class_merge"]:
            return comp_list
        entry["merges"] = []
        groups = {i: g for i, g in enumerate(comp_list)}
        parent = list(range(len(comp_list)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        cand = []
        for i in range(len(comp_list)):
            for j in range(i + 1, len(comp_list)):
                is_cand, ios, dist = _merge_candidate(comp_list[i], comp_list[j])
                if is_cand:
                    cand.append((-ios, dist, i, j))
        # strongest overlaps first; each pair tested once against the CURRENT
        # (possibly already-unioned) groups, so evidence sees merged geometry
        for neg_ios, dist, i, j in sorted(cand):
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            votes, checked = _pairwise_distinct(
                groups[ri], groups[rj], multi, masks_v)
            rec = dict(pair_gaussians=[len(groups[ri]), len(groups[rj])],
                       ios=round(-neg_ios, 3), dist=round(dist, 3),
                       distinct_votes=votes, views_checked=checked)
            if _is_distinct(votes, checked):
                rec["result"] = "kept_distinct"
            else:
                rec["result"] = "merged"
                parent[rj] = ri
                groups[ri] = np.concatenate([groups[ri], groups[rj]])
                del groups[rj]
            entry["merges"].append(rec)
        return list(groups.values())
    def _make_diag(c: str):
        """§3.2 step 3 diagnostic: the vote _pairwise_distinct would cast
        on two supports — recorded on refused merges, decides nothing."""
        cache: dict = {}

        def diag(gA: np.ndarray, gB: np.ndarray):
            if "m" not in cache:
                cache["m"] = _class_multi(c)
            multi, masks_v = cache["m"]
            return _pairwise_distinct(gA, gB, multi, masks_v)
        return diag

    # 2D-count sanity, shared by both modes: expected = max k seen in >=
    # count_stability_views views (the readout; unchanged).
    def _expected_from_2d(c: str) -> int:
        view_counts = [sum(1 for d in dets if d["concept"] == c)
                       for dets in per_view.values()]
        expected = 0
        for k in range(1, (max(view_counts) if view_counts else 0) + 1):
            if sum(1 for vc in view_counts if vc >= k) >= lcfg["count_stability_views"]:
                expected = k
        return expected

    det_stats: dict = {}
    if mode == "tracks":
        # ===== instances from 2D detection correspondence ===================
        # §3.1 — a SECOND pass over the views rasterises one channel per
        # retained detection (carve applied exactly as to the class
        # channels) plus the all-ones total, in chunks of det_channel_chunk.
        # It runs after the class assignment on purpose: a detection's
        # SUPPORT is restricted to gaussians carrying its class, so the
        # rows can be filtered at extraction and the class loop above stays
        # byte-identical (the design's "same call when they fit" was an
        # efficiency suggestion; eligibility needs the labels first).
        from .instancing import INSTANCING_SCHEMA, instance_class
        t_det = time.perf_counter()
        floor_f = float(lcfg["support_min_frac"])
        cl_max = float(lcfg["cannot_link_ios_max"])
        op_ok = opacities >= lcfg["min_opacity"]
        elig = torch.zeros(n, len(classes), dtype=torch.bool, device=device)
        for c in classes:
            elig[:, cls_id[c]] = (labels == cls_id[c]) & vol_mask & op_ok
        max_dets_view = max((len(v) for v in per_view.values()), default=0)
        budget = det_channel_budget(lcfg.get("det_channel_chunk", "auto"),
                                    n, max_dets_view)
        chunk = int(budget["chunk"])
        log.info("detection channels: %.1f GB free of %.1f on the card, "
                 "%.1f GB usable beside the %.0f GB reserve; %s Gaussians x "
                 "%d B = %.0f MB per channel -> chunk %d (%s, %.1f GB "
                 "estimated; the busiest view has %d detections)",
                 budget["free_gb"], budget["card_gb"], budget["usable_gb"],
                 budget["reserve_gb"], f"{n:,}", budget["bytes_per_channel"],
                 budget["per_channel_mb"], chunk, budget["source"],
                 budget["estimate_gb"], max_dets_view)
        det_info: list[dict] = []
        per_view_gids: dict[int, list[int]] = {}
        rows = dict(gidx=[], det=[], g=[], f=[], tv=[])
        f_hist = {c: dict(same=np.zeros(50, np.int64), other=np.zeros(50, np.int64))
                  for c in classes}
        tv_hist = {c: dict(support=np.zeros(70, np.int64), nonzero=np.zeros(70, np.int64))
                   for c in classes}
        cannot_link: dict[str, list] = defaultdict(list)
        pair_ios: dict[str, list] = defaultdict(list)
        n_raster = nonzero_total = 0
        for fr in cams["frames"]:
            check_cancel(should_cancel)   # between views AND between chunks
            fidx = fr["frame_idx"]
            dets = per_view.get(fidx, [])
            if not dets:
                continue
            K, w, h = frame_K(fr)
            masks_raw, masks_carved, gids = [], [], []
            for d in dets:
                gid = len(det_info)
                det_info.append(dict(frame_idx=fidx, concept=d["concept"],
                                     det_idx=d["det_idx"], score=d["score"],
                                     source=d.get("source", "text")))
                gids.append(gid)
                raw = np.asarray(Image.open(_mask_path(
                    stage2, fidx, d["concept"], d["det_idx"])))
                masks_raw.append(raw > 127)
                m = raw.astype(np.float32) / 255.0
                for pc, pdi in d.get("carve", []):
                    pm = np.asarray(Image.open(_mask_path(
                        stage2, fidx, pc, pdi)), dtype=np.float32) / 255.0
                    m = m * (1.0 - pm)
                masks_carved.append(m)
            per_view_gids[fidx] = gids
            # §3.2 cannot-link pairs: same view, same class, near-disjoint
            # by the SMALLER mask — from the masks alone, never revised.
            areas = [int(x.sum()) for x in masks_raw]
            for i in range(len(dets)):
                if not areas[i]:
                    continue
                for j in range(i + 1, len(dets)):
                    if dets[i]["concept"] != dets[j]["concept"] or not areas[j]:
                        continue
                    inter = int((masks_raw[i] & masks_raw[j]).sum())
                    v = inter / min(areas[i], areas[j])
                    c = dets[i]["concept"]
                    pair_ios[c].append([fidx, gids[i], gids[j], round(v, 4)])
                    if v <= cl_max:
                        cannot_link[c].append((gids[i], gids[j], v))
            viewmat = torch.linalg.inv(torch.tensor(
                np.array(fr["c2w"]), dtype=torch.float32, device=device))
            Tv = None
            for k0 in range(0, len(dets), chunk):
                check_cancel(should_cancel)
                k1 = min(len(dets), k0 + chunk)
                nk = k1 - k0
                nch = nk + (1 if Tv is None else 0)
                stack = np.zeros((h, w, nch), dtype=np.float32)
                for k in range(k0, k1):
                    stack[..., k - k0] = masks_carved[k]
                if Tv is None:
                    stack[..., -1] = 1.0
                target = torch.as_tensor(stack, device=device)
                colors = torch.ones(n, nch, device=device, requires_grad=True)
                render, _, _ = gsplat.rasterization(
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors, viewmats=viewmat[None],
                    Ks=K[None], width=w, height=h,
                    near_plane=cfg["render"]["near_plane"],
                    far_plane=cfg["render"]["far_plane"])
                (render[0] * target).sum().backward()
                n_raster += 1
                grad = colors.grad.clamp_(min=0)
                del render, target, colors
                if Tv is None:
                    Tv = grad[:, -1].clone()
                gd = grad[:, :nk]
                det_cls = torch.tensor([cls_id[dets[k]["concept"]]
                                        for k in range(k0, k1)], device=device)
                tv_ok = Tv > 0
                nz = gd > 0
                nonzero_total += int(nz.sum())
                # in-mask fraction f = g / T_v; T_v = 0 is NO support
                f = (gd / Tv.clamp_min(1e-30)[:, None]).clamp_(max=1.0)
                f[~tv_ok] = 0.0
                keep = elig[:, det_cls] & nz & tv_ok[:, None]
                # calibration readouts BEFORE the floor: f by class verdict
                lab_match = labels[:, None] == det_cls[None, :]
                nz_ok = nz & tv_ok[:, None]
                for cid in torch.unique(det_cls).tolist():
                    cm = det_cls == cid
                    fs = f[:, cm]
                    same = nz_ok[:, cm] & lab_match[:, cm]
                    other = nz_ok[:, cm] & ~lab_match[:, cm]
                    cname = classes[cid]
                    f_hist[cname]["same"] += torch.histc(
                        fs[same], 50, 0, 1).long().cpu().numpy()
                    f_hist[cname]["other"] += torch.histc(
                        fs[other], 50, 0, 1).long().cpu().numpy()
                    tvl = torch.log10(Tv.clamp_min(1e-12))
                    tv_hist[cname]["nonzero"] += torch.histc(
                        tvl[:, None].expand(-1, int(cm.sum()))[nz_ok[:, cm]],
                        70, -6, 1).long().cpu().numpy()
                    ks = keep[:, cm] & (fs >= floor_f)
                    tv_hist[cname]["support"] += torch.histc(
                        tvl[:, None].expand(-1, int(cm.sum()))[ks],
                        70, -6, 1).long().cpu().numpy()
                    del fs, same, other, ks
                keep &= f >= floor_f
                idx = keep.nonzero()
                gi, dj = idx[:, 0], idx[:, 1]
                rows["gidx"].append(gi.to(torch.int32).cpu().numpy())
                rows["det"].append(np.asarray(gids, dtype=np.int32)[
                    (dj + k0).cpu().numpy()])
                rows["g"].append(gd[gi, dj].cpu().numpy().astype(np.float32))
                rows["f"].append(f[gi, dj].cpu().numpy().astype(np.float16))
                rows["tv"].append(Tv[gi].cpu().numpy().astype(np.float16))
                del grad, gd, f, keep, nz, nz_ok, lab_match, idx, gi, dj
            del Tv
        rows = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int32))
                for k, v in rows.items()}
        rows_bytes = int(sum(v.nbytes for v in rows.values()))
        t_det = time.perf_counter() - t_det
        budget["peak_alloc_gb"] = round(
            torch.cuda.max_memory_allocated() / 2**30, 2)
        log.info("detection channels: %d dets, %d rasterisations (chunk %d, "
                 "max %d dets/view), %d nonzero rows, %d support rows held "
                 "(%.2f GB CPU), peak %.2f GB allocated on the card "
                 "(%.2f GB before the pass, %.1f GB estimated for it), %.1fs",
                 len(det_info), n_raster, chunk, max_dets_view, nonzero_total,
                 len(rows["gidx"]), rows_bytes / 1e9, budget["peak_alloc_gb"],
                 budget["alloc_before_gb"], budget["estimate_gb"], t_det)
        det_stats = dict(dets=len(det_info), rasterisations=n_raster,
                         det_channel_chunk=chunk, det_channel_budget=budget,
                         max_dets_per_view=max_dets_view,
                         nonzero_rows=nonzero_total,
                         support_rows=int(len(rows["gidx"])),
                         support_rows_bytes=rows_bytes,
                         det_pass_s=round(t_det, 1))

        # §3.2–3.4 per class
        det_frame = np.array([d["frame_idx"] for d in det_info], dtype=np.int64)
        det_score = np.array([d["score"] for d in det_info], dtype=np.float64)
        det_source = [d["source"] for d in det_info]
        det_didx = np.array([d["det_idx"] for d in det_info], dtype=np.int64)
        det_cls_all = np.array([cls_id[d["concept"]] for d in det_info],
                               dtype=np.int64)
        rows_cls = det_cls_all[rows["det"]] if len(rows["det"]) else rows["det"]
        A_pos = (A > 0).cpu().numpy()
        elig_np = elig.cpu().numpy()
        tracks_doc: list[dict] = []
        dropped_doc: list[dict] = []
        calib_classes: dict = {}
        debug_classes: dict = {}
        t_cls = time.perf_counter()
        for c in classes:
            check_cancel(should_cancel)
            cid = cls_id[c]
            class_gauss = np.flatnonzero(elig_np[:, cid])
            expected = _expected_from_2d(c)
            entry = dict(gaussians=int(len(class_gauss)), components=0,
                         splits=[], dropped=[], out_of_volume_gaussians=oov[c],
                         expected_from_2d=expected, merges=[],
                         instancing="tracks")
            audit[c] = entry
            if len(class_gauss) < lcfg["min_instance_gaussians"]:
                if len(class_gauss):
                    entry["dropped"].append(dict(
                        reason="class_below_min_gaussians",
                        gaussians=int(len(class_gauss))))
                continue
            dg = np.flatnonzero(det_cls_all == cid)
            sel = rows_cls == cid
            R = instance_class(
                c=c, det_gids=dg, det_frame=det_frame, det_score=det_score,
                det_source=det_source, det_didx=det_didx,
                rows_gidx=rows["gidx"][sel].astype(np.int64),
                rows_det=rows["det"][sel].astype(np.int64),
                rows_g=rows["g"][sel], rows_f=rows["f"][sel],
                cannot_link=cannot_link.get(c, []),
                class_gauss=class_gauss, touched=A_pos[class_gauss, cid],
                means_np=means_np, lcfg=lcfg,
                distinct_diag=_make_diag(c), device=device,
                want_debug=dump_dir is not None)
            entry.update(R.audit)
            for r in R.dropped_tracks:
                entry["dropped"].append(dict(reason=r["reason"],
                                             gaussians=r["gaussians"]))
                dropped_doc.append(dict(r, **{"class": c}))
            calib_classes[c] = R.calib
            if R.debug:
                debug_classes[c] = R.debug
            base = next_inst
            assigned = R.inst_local >= 0
            if assigned.any():
                instances[torch.as_tensor(R.gauss[assigned], device=device)] = \
                    torch.as_tensor(base + R.inst_local[assigned],
                                    device=device, dtype=torch.int32)
            for tr in R.tracks:
                inst_id = base + tr["track"]
                tr["instance_id"] = inst_id
                tracks_doc.append(dict(
                    instance_id=inst_id, **{"class": c},
                    class_track=tr["class_track"], gaussians=tr["gaussians"],
                    support_gaussians=tr["support_gaussians"],
                    views=tr["views"], dets=tr.pop("det_list"),
                    ambiguous_dets=tr["ambiguous_dets"],
                    ambiguous_unassigned=tr["ambiguous_unassigned"],
                    ambiguous_unassigned_frac=tr["ambiguous_unassigned_frac"],
                    bridge_refused=[entry["bridge_refused"][k]
                                    for k in tr["bridge_refused"]],
                    conflicting_support=tr["conflicting_support"],
                    track_components=tr["track_components"],
                    margin=tr["margin"], margin_low_frac=tr["margin_low_frac"]))
            entry["track_records"] = [dict(r, instance_id=base + r["track"])
                                      for r in entry["track_records"]]
            next_inst += len(R.tracks)
            if entry["components"] > max(expected, 0):
                entry["note"] = (f"3D count {entry['components']} > "
                                 f"2D-expected {expected} (kept)")
        det_stats["class_pass_s"] = round(time.perf_counter() - t_cls, 1)
        (out / "tracks.json").write_text(json.dumps(dict(
            instancing_schema=INSTANCING_SCHEMA, instancing="tracks",
            support_min_frac=floor_f,
            track_affinity_min=lcfg["track_affinity_min"],
            track_min_support_ios=lcfg["track_min_support_ios"],
            cannot_link_ios_max=cl_max,
            tracks=tracks_doc,
            instance_to_track={t["instance_id"]: k
                               for k, t in enumerate(tracks_doc)},
            dropped_tracks=dropped_doc), indent=1))
        (out / "instancing_calibration.json").write_text(json.dumps(dict(
            instancing_schema=INSTANCING_SCHEMA,
            support_min_frac=floor_f, cannot_link_ios_max=cl_max,
            track_affinity_min=lcfg["track_affinity_min"],
            track_min_support_ios=lcfg["track_min_support_ios"],
            f_hist=dict(bins=50, range=[0, 1],
                        by_class={c: dict(same=h["same"].tolist(),
                                          other=h["other"].tolist())
                                  for c, h in f_hist.items()}),
            tv_hist=dict(bins=70, range_log10=[-6, 1],
                         by_class={c: dict(support=h["support"].tolist(),
                                           nonzero=h["nonzero"].tolist())
                                   for c, h in tv_hist.items()}),
            mask_pair_ios={c: v for c, v in pair_ios.items()},
            cannot_link_pairs={c: len(v) for c, v in cannot_link.items()},
            by_class=calib_classes, det_stats=det_stats), indent=1))
        if dump_dir is not None:
            dd = Path(dump_dir)
            dd.mkdir(parents=True, exist_ok=True)
            np.savez(dd / "support_rows.npz", **rows)
            (dd / "det_info.json").write_text(json.dumps(det_info))
            for c, dbg in debug_classes.items():
                np.savez(dd / f"class_{c.replace(' ', '_')}.npz", **dbg)
        del rows, elig, A_pos, elig_np
    else:
        for c in classes:
            sel = ((labels == cls_id[c]) &
                   (opacities >= lcfg["min_opacity"])).cpu().numpy()
            idx = np.flatnonzero(sel)
            entry = dict(gaussians=int(len(idx)), components=0, splits=[], dropped=[],
                         out_of_volume_gaussians=oov[c])
            audit[c] = entry
            if len(idx) < lcfg["min_instance_gaussians"]:
                if len(idx):
                    entry["dropped"].append(
                        dict(reason="class_below_min_gaussians", gaussians=int(len(idx))))
                continue

            # voxel connected components (26-connectivity via voxel-key BFS)
            vox = lcfg["instance_voxel"]
            keys = np.floor(means_np[idx] / vox).astype(np.int64)
            key_set = {}
            for i, kk in enumerate(map(tuple, keys)):
                key_set.setdefault(kk, []).append(i)
            comp_of_voxel = {}
            comp = 0
            for start_key in key_set:
                if start_key in comp_of_voxel:
                    continue
                stack = [start_key]
                comp_of_voxel[start_key] = comp
                while stack:
                    ck = stack.pop()
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                nk = (ck[0] + dx, ck[1] + dy, ck[2] + dz)
                                if nk in key_set and nk not in comp_of_voxel:
                                    comp_of_voxel[nk] = comp
                                    stack.append(nk)
                comp += 1
            comp_ids = np.empty(len(idx), dtype=np.int64)
            for kk, members in key_set.items():
                comp_ids[members] = comp_of_voxel[kk]

            # drop fragments
            comps, counts = np.unique(comp_ids, return_counts=True)
            keep = comps[counts >= lcfg["min_instance_gaussians"]]
            for cc, cnt in zip(comps, counts):
                if cc not in keep:
                    entry["dropped"].append(dict(reason="fragment", gaussians=int(cnt)))
            comp_list = [idx[comp_ids == cc] for cc in keep]

            # 2D-count sanity: expected = max k seen in >= count_stability_views views
            view_counts = [sum(1 for d in dets if d["concept"] == c)
                           for dets in per_view.values()]
            expected = 0
            for k in range(1, (max(view_counts) if view_counts else 0) + 1):
                if sum(1 for vc in view_counts if vc >= k) >= lcfg["count_stability_views"]:
                    expected = k
            entry["expected_from_2d"] = expected

            multi, masks_v = _class_multi(c)

            set_aside: list = []   # comps whose bisection failed effectiveness
            while comp_list and len(comp_list) + len(set_aside) < expected:
                big_i = int(np.argmax([len(g) for g in comp_list]))
                big = comp_list.pop(big_i)
                parts, mech, fviews = None, "kmeans_coords", []
                if lcfg["split_mode"] == "mask_guided":
                    parts, fviews = _feature_bisect(c, big)
                    if parts is not None:
                        mech = "mask_guided"
                if parts is None:
                    assign = _kmeans(torch.as_tensor(means_np[big]), 2).numpy()
                    parts = [big[assign == j] for j in (0, 1)]
                if min(len(p) for p in parts) < lcfg["min_instance_gaussians"]:
                    comp_list.append(big)  # split would create a fragment — stop
                    entry["splits"].append(dict(result="refused_fragment",
                                                mechanism=mech))
                    break
                if not _split_effective(parts, multi, masks_v, entry, mech):
                    set_aside.append(big)  # try the next-largest comp instead
                    continue
                split_rec = dict(split_gaussians=[len(p) for p in parts],
                                 mechanism=mech)
                if mech == "mask_guided":
                    split_rec["views"] = fviews
                entry["splits"].append(split_rec)
                comp_list.extend(parts)
            comp_list.extend(set_aside)

            if lcfg["split_correspondence"]:
                comp_list = _correspondence_split(c, comp_list, entry,
                                                  multi, masks_v)
            comp_list = _same_class_merge(comp_list, entry, multi, masks_v)

            entry["components"] = len(comp_list)
            if len(comp_list) > max(expected, 0):
                entry["note"] = f"3D count {len(comp_list)} > 2D-expected {expected} (kept)"
            for g in comp_list:
                instances[torch.as_tensor(g, device=device)] = next_inst
                next_inst += 1

    np.save(out / "labels.npy", labels.cpu().numpy().astype(np.int32))
    np.save(out / "instances.npy", instances.cpu().numpy())
    (out / "classes.json").write_text(json.dumps(classes, indent=1))
    (out / "instance_audit.json").write_text(json.dumps(audit, indent=1))

    n_labeled = int((labels >= 0).sum())
    import resource
    from .instancing import INSTANCING_SCHEMA
    manifest = dict(
        stage="lift", scene=str(scene_path), classes=classes,
        instancing=mode, instancing_schema=INSTANCING_SCHEMA,
        peak_gpu_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        peak_gpu_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
        peak_cpu_rss_gb=round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        det_stats=det_stats, class_pass_budget=class_budget,
        manual_views_hash=stage1_manual_record(workdir).get("hash", "absent"),
        exemplars_hash=stage2_exemplar_record(workdir).get("hash", "absent"),
        instance_voxel=params["instance_voxel"],   # metres, as the profile says
        # The one factor this lift ran at, and every knob it converted.
        scale=scale_rec.record(), units=units,
        gamma=gamma, n_gaussians=n, n_labeled=n_labeled,
        upstream_stage2=upstream_fingerprint(workdir,
                                            "stage2/probe_manifest.json"),
        params=params,
        n_instances=next_inst, masks_used=n_masks_used,
        out_of_volume_by_class={c: v for c, v in oov.items() if v},
        accum_s=round(t_accum, 1), elapsed_s=round(time.perf_counter() - t0, 1))
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log.info("stage3 done: %d/%d gaussians labeled, %d instances, %.1fs (accum %.1fs)",
             n_labeled, n, next_inst, manifest["elapsed_s"], t_accum)
    return manifest
