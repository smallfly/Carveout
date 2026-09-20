# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stage-3 instance identity from 2D detection correspondence.

The class channels of the lift say WHICH class a Gaussian is; they discard
which detection it rendered into. This module rebuilds instances from that
discarded identity: one contribution channel per DETECTION (rasterised by
the lift in a second pass, `lift.py`), a sparse SUPPORT per detection (the
eligible Gaussians whose in-mask fraction `f = g / T_v` reaches
`lift.support_min_frac`), and then, per class:

1. detection TRACKS — affinity (cosine of the support-restricted
   contribution vectors, any two views, the same view included) proposes
   edges above `track_affinity_min` with support IoS above
   `track_min_support_ios`; edges merge in descending affinity UNLESS any
   member of one group and any member of the other are a CANNOT-LINK pair
   (two same-class masks SAM 3 drew apart in one view, near-disjoint by the
   smaller mask: `cannot_link_ios_max`). A refused merge is recorded
   (`bridge_refused`) and both resulting instances carry it as a hold
   reason at export;
2. AMBIGUOUS detections — a detection whose support overlaps a detection
   of its own track AND a cannot-linked detection of another track sits
   across two objects; it keeps its track for provenance but casts no vote,
   and leaves confidence and crops at export;
3. ASSIGNMENT — each Gaussian goes to the track with the largest
   per-distinct-view sum of its own-support contribution; Gaussians left
   without a vote stay unassigned and are COUNTED (ambiguity-created,
   below the support floor; a labelled gaussian no detection touched is
   reported, never filled);
4. 3D connectivity is RECORDED per instance (`track_components`) and
   decides nothing.

Every decision lands in the per-class audit and the per-track provenance
(`stage3/tracks.json`); the lift writes both. Nothing here touches the
class labels.
"""

import logging
import time
from collections import defaultdict

import numpy as np
import torch

log = logging.getLogger(__name__)

INSTANCING_SCHEMA = 1
DIAG_MAX_PER_CLASS = 50        # projection diagnostics per class on refusals
_BLOCK_BYTES = 512 << 20       # dense (dets x gaussians) block budget on GPU


def voxel_components(pts: np.ndarray, vox: float) -> np.ndarray:
    """26-connected voxel components of `pts` at `vox` (component id per
    point). The same BFS the connectivity instancer uses; here it only
    RECORDS."""
    keys = np.floor(pts / vox).astype(np.int64)
    key_set: dict = {}
    for i, kk in enumerate(map(tuple, keys)):
        key_set.setdefault(kk, []).append(i)
    comp_of: dict = {}
    comp = 0
    for start in key_set:
        if start in comp_of:
            continue
        stack = [start]
        comp_of[start] = comp
        while stack:
            ck = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (ck[0] + dx, ck[1] + dy, ck[2] + dz)
                        if nk in key_set and nk not in comp_of:
                            comp_of[nk] = comp
                            stack.append(nk)
        comp += 1
    out = np.empty(len(pts), dtype=np.int64)
    for kk, members in key_set.items():
        out[members] = comp_of[kk]
    return out


def _dense_block(rows_det, rows_col, rows_g, n_dets, c0, c1, device):
    """Dense (n_dets x (c1-c0)) block of the support rows for columns
    [c0, c1); rows must be sorted by column."""
    lo, hi = np.searchsorted(rows_col, [c0, c1])
    blk = torch.zeros(n_dets, c1 - c0, device=device)
    if hi > lo:
        d = torch.as_tensor(rows_det[lo:hi].astype(np.int64), device=device)
        cc = torch.as_tensor((rows_col[lo:hi] - c0).astype(np.int64),
                             device=device)
        blk[d, cc] = torch.as_tensor(rows_g[lo:hi], device=device)
    return blk


def _hist(x: np.ndarray, bins: int, lo: float, hi: float) -> list[int]:
    h, _ = np.histogram(x, bins=bins, range=(lo, hi))
    return h.astype(int).tolist()


class ClassResult:
    """What one class hands back to the lift."""

    def __init__(self):
        self.gauss = np.zeros(0, dtype=np.int64)     # global gaussian idx
        self.inst_local = np.zeros(0, dtype=np.int64)  # track id or -1
        self.tracks: list[dict] = []                 # surviving, in id order
        self.dropped_tracks: list[dict] = []
        self.audit: dict = {}
        self.calib: dict = {}
        self.debug: dict = {}


def instance_class(*, c: str, det_gids: np.ndarray, det_frame: np.ndarray,
                   det_score: np.ndarray, det_source: list, det_didx: np.ndarray,
                   rows_gidx: np.ndarray, rows_det: np.ndarray, rows_g: np.ndarray,
                   rows_f: np.ndarray,
                   cannot_link: list[tuple[int, int, float]],
                   class_gauss: np.ndarray, touched: np.ndarray,
                   means_np: np.ndarray, lcfg: dict,
                   distinct_diag=None, device: str = "cuda",
                   want_debug: bool = False) -> ClassResult:
    """Build the tracks and the assignment of ONE class.

    det_gids: global ids of every retained detection of class c (support
      or not); det_frame/score/source/didx are indexed by GLOBAL det id.
    rows_*: the class's support rows (global gaussian idx, global det id,
      raw weight g, in-mask fraction f), any order. Affinity uses f (the
      bank's finding: g scales with a gaussian's projected area and
      collapses the cosine between two views of one large or flat object;
      f is view-invariant for a gaussian inside the mask; the `raw`
      alternative was measured worse and removed).
      Assignment votes always use g.
    cannot_link: (p, q, ios) global det ids — same view, near-disjoint.
    class_gauss: global idx of the ELIGIBLE gaussians of class c (label,
      volume+margin, opacity); touched: per class_gauss, whether any
      detection of c left weight on it (A[:, c] > 0) — its complement
      (labelled with no detection weight) is reported, never filled.
    """
    t0 = time.perf_counter()
    R = ClassResult()
    a_min = lcfg["track_affinity_min"]
    s_min = lcfg["track_min_support_ios"]
    floor = lcfg["min_instance_gaussians"]
    n_dets = len(det_gids)
    local = {int(g): j for j, g in enumerate(det_gids)}
    ent = R.audit
    ent.update(tracks=0, no_support=0, bridge_refused=[], ambiguous_dets=[],
               ambiguous_unassigned=0, below_support_unassigned=0,
               dropped_tracks=[], refine_dropped=0, ties=0,
               conflicting_support_flags=0, track_records=[])

    # -- supports -------------------------------------------------------------
    cols_all, col_of_row = np.unique(rows_gidx, return_inverse=True)
    m = len(cols_all)
    rd = np.array([local[int(d)] for d in rows_det], dtype=np.int64) \
        if len(rows_det) else np.zeros(0, dtype=np.int64)
    order = np.argsort(col_of_row, kind="stable")
    rd, rc, rg = rd[order], col_of_row[order], rows_g[order].astype(np.float32)
    rw = rows_f[order].astype(np.float32)   # in-mask fraction, view-invariant
    size = np.bincount(rd, minlength=n_dets).astype(np.int64)
    norm = np.sqrt(np.bincount(rd, weights=rw.astype(np.float64) ** 2,
                               minlength=n_dets))
    has = size > 0
    ent["no_support"] = int((~has).sum())
    ent["no_support_dets"] = [int(det_gids[j]) for j in np.flatnonzero(~has)]
    # per-det support (sorted global gaussian idx) for the set operations
    by_det_order = np.argsort(rd, kind="stable")
    bounds = np.searchsorted(rd[by_det_order], np.arange(n_dets + 1))
    def support_cols(j: int) -> np.ndarray:
        return rc[by_det_order[bounds[j]:bounds[j + 1]]]

    if m == 0 or not has.any():
        R.gauss = class_gauss
        R.inst_local = np.full(len(class_gauss), -1, dtype=np.int64)
        ent["below_support_unassigned"] = int(touched.sum())
        return R

    # -- affinity + support intersections, dense blocks over columns ----------
    width = max(1024, int(_BLOCK_BYTES // (4 * max(n_dets, 1))))
    aff = torch.zeros(n_dets, n_dets, device=device)
    inter = torch.zeros(n_dets, n_dets, device=device)
    for c0 in range(0, m, width):
        c1 = min(m, c0 + width)
        blk = _dense_block(rd, rc, rw, n_dets, c0, c1, device)
        aff += blk @ blk.T
        b = (blk > 0).float()
        inter += b @ b.T
        del blk, b
    nrm = torch.as_tensor(norm, device=device, dtype=torch.float32)
    aff = aff / (nrm[:, None] * nrm[None, :]).clamp_min(1e-12)
    aff.fill_diagonal_(1.0)
    sz = torch.as_tensor(size, device=device, dtype=torch.float32)
    ios = inter / torch.minimum(sz[:, None], sz[None, :]).clamp_min(1.0)
    aff_np, ios_np = aff.cpu().numpy(), ios.cpu().numpy()
    del aff, inter, ios

    # -- cannot-link pairs (local) ------------------------------------------------
    cl_adj: dict[int, set] = defaultdict(set)
    cl_pairs: list[tuple[int, int, float]] = []
    for p, q, v in cannot_link:
        if p in local and q in local:
            a, b = local[p], local[q]
            cl_adj[a].add(b)
            cl_adj[b].add(a)
            cl_pairs.append((a, b, v))

    # -- candidate edges, descending affinity, ties by the lower id pair --------
    iu = np.triu_indices(n_dets, k=1)
    ok = (has[iu[0]] & has[iu[1]] & (aff_np[iu] >= a_min)
          & (ios_np[iu] >= s_min))
    ea, eb, ev = iu[0][ok], iu[1][ok], aff_np[iu][ok]
    ga, gb = det_gids[ea], det_gids[eb]
    lo_id, hi_id = np.minimum(ga, gb), np.maximum(ga, gb)
    e_order = np.lexsort((hi_id, lo_id, -ev))

    # -- clustering: affinity proposes, cannot-link vetoes ---------------------
    parent = list(range(n_dets))
    members: dict[int, set] = {j: {j} for j in range(n_dets) if has[j]}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    refused: list[dict] = []
    n_edges = 0
    for k in e_order:
        a, b, v = int(ea[k]), int(eb[k]), float(ev[k])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        n_edges += 1
        ma, mb = members[ra], members[rb]
        small, big = (ma, mb) if len(ma) <= len(mb) else (mb, ma)
        veto = None
        for mm in small:
            hit = cl_adj.get(mm)
            if hit:
                h = hit & big
                if h:
                    veto = (mm, min(h))
                    break
        if veto is not None:
            p, q = veto
            third = [x for x in (a, b) if x not in (p, q)]
            side = {}
            for x in third:
                side[x] = p if x in (ma if p in ma else mb) else q
            refused.append(dict(
                edge=[int(det_gids[a]), int(det_gids[b])],
                affinity=round(v, 4),
                support_ios=round(float(ios_np[a, b]), 4),
                pair=[int(det_gids[p]), int(det_gids[q])],
                pair_mask_ios=round(float(next(
                    vv for (x, y, vv) in cl_pairs
                    if {x, y} == {p, q})), 4),
                third=[int(det_gids[x]) for x in third],
                third_side=[int(det_gids[side[x]]) for x in third],
                groups=[len(ma), len(mb)],
                _roots=(ra, rb)))
            continue
        parent[rb] = ra
        members[ra] = ma | mb
        del members[rb]

    # -- tracks, ordered by their lowest global det id ---------------------------
    groups = sorted(members.values(), key=lambda s: min(det_gids[j] for j in s))
    track_of = np.full(n_dets, -1, dtype=np.int64)
    for t, g in enumerate(groups):
        for j in g:
            track_of[j] = t
    n_tracks = len(groups)
    ent["tracks"] = n_tracks

    # -- ambiguous detections -----------------------------------------------------
    ambiguous = np.zeros(n_dets, dtype=bool)
    amb_records = []
    for a, b, v in cl_pairs:
        ta, tb = track_of[a], track_of[b]
        if ta < 0 or tb < 0 or ta == tb:
            continue
        for p, q, tp in ((a, b, ta), (b, a, tb)):
            for d in groups[tp]:
                if d == p:
                    continue
                op, oq = float(ios_np[d, p]), float(ios_np[d, q])
                if op >= s_min and oq >= s_min:
                    if not ambiguous[d]:
                        ambiguous[d] = True
                    amb_records.append(dict(
                        det=int(det_gids[d]), track=int(tp),
                        pair=[int(det_gids[p]), int(det_gids[q])],
                        overlap_p=round(op, 3), overlap_q=round(oq, 3)))
    ent["ambiguous_dets"] = amb_records

    # -- diagnostics on refusals (the old projection signal) ---------------------
    if distinct_diag is not None and refused:
        for i, rec in enumerate(refused):
            if i >= DIAG_MAX_PER_CLASS:
                rec["diag"] = f"skipped (cap {DIAG_MAX_PER_CLASS} per class)"
                continue
            ra, rb = rec["_roots"]
            # the groups as they were at refusal are gone; use the FINAL
            # tracks of the two endpoints (members never leave a group)
            ta, tb = track_of[local[rec["edge"][0]]], track_of[local[rec["edge"][1]]]
            sa = np.unique(np.concatenate([support_cols(j) for j in groups[ta]]))
            sb = np.unique(np.concatenate([support_cols(j) for j in groups[tb]]))
            votes, checked = distinct_diag(cols_all[sa], cols_all[sb])
            rec["diag_distinct_votes"], rec["diag_views_checked"] = votes, checked
    for rec in refused:
        rec.pop("_roots", None)
    ent["bridge_refused"] = refused

    # -- assignment: per distinct view, max over non-ambiguous dets of the track
    frames_local = det_frame[det_gids]
    view_dets: dict[int, list[int]] = defaultdict(list)
    for j in range(n_dets):
        if has[j] and not ambiguous[j] and track_of[j] >= 0:
            view_dets[int(frames_local[j])].append(j)
    view_groups = []
    for f, js in view_dets.items():
        js = np.array(js, dtype=np.int64)
        view_groups.append((torch.as_tensor(js, device=device),
                            torch.as_tensor(track_of[js], device=device)))
    best_val = np.zeros(m, dtype=np.float32)
    best_trk = np.full(m, -1, dtype=np.int64)
    second = np.zeros(m, dtype=np.float32)
    for c0 in range(0, m, width):
        c1 = min(m, c0 + width)
        blk = _dense_block(rd, rc, rg, n_dets, c0, c1, device)
        score = torch.zeros(max(n_tracks, 1), c1 - c0, device=device)
        for js, ts in view_groups:
            sub = blk[js]
            per = torch.zeros_like(score).scatter_reduce_(
                0, ts[:, None].expand(len(js), c1 - c0), sub, reduce="amax",
                include_self=True)
            score += per
        bt = score.argmax(dim=0)
        bv = score.gather(0, bt[None]).squeeze(0)
        s2 = score.clone()
        s2[bt, torch.arange(c1 - c0, device=device)] = -1.0
        sv = s2.max(dim=0).values.clamp_min(0.0)
        best_val[c0:c1] = bv.cpu().numpy()
        best_trk[c0:c1] = bt.cpu().numpy()
        second[c0:c1] = sv.cpu().numpy()
        del blk, score, s2
    assigned = best_val > 0
    inst_col = np.where(assigned, best_trk, -1)
    ent["ties"] = int(((best_val == second) & assigned).sum())
    margin = np.where(assigned, best_val - second, 0.0)
    low_margin = assigned & (margin < 0.1 * best_val)

    # -- gaussians without a vote -------------------------------------------------
    # (a) in supports of ambiguous dets only — stays unassigned, counted per
    #     track; (b) below the floor on every det — never in a column;
    #     (c) labelled, untouched by any detection — reported below.
    amb_unassigned_by_track: dict[int, set] = defaultdict(set)
    for j in np.flatnonzero(ambiguous):
        cols = support_cols(j)
        lost = cols[~assigned[cols]]
        if len(lost):
            amb_unassigned_by_track[int(track_of[j])].update(lost.tolist())
    ent["ambiguous_unassigned"] = int(len(set().union(
        *amb_unassigned_by_track.values()))) if amb_unassigned_by_track else 0

    # map columns (support gaussians) -> positions in class_gauss
    pos_of = {int(g): i for i, g in enumerate(class_gauss)}
    col_pos = np.array([pos_of.get(int(g), -1) for g in cols_all], dtype=np.int64)
    in_cols = np.zeros(len(class_gauss), dtype=bool)
    in_cols[col_pos[col_pos >= 0]] = True
    inst = np.full(len(class_gauss), -1, dtype=np.int64)
    okc = col_pos >= 0
    inst[col_pos[okc]] = inst_col[okc]
    ent["below_support_unassigned"] = int((touched & ~in_cols).sum())
    ent["support_gaussians_outside_eligible"] = int((~okc).sum())

    # (c) label c with no detection weight at all: such gaussians cannot
    # exist (the gamma test needs A > 0) — report rather than silently
    # count them elsewhere. (The `--refine` voxel vote that used to fill
    # them was removed.)
    refine_created = np.flatnonzero(~touched)
    if len(refine_created):
        ent["refine_dropped"] = int(len(refine_created))

    # -- floor: small TRACKS drop (reason fragment); all-ambiguous tracks empty
    counts = np.bincount(inst[inst >= 0], minlength=max(n_tracks, 1))
    keep = np.zeros(n_tracks, dtype=bool)
    for t, g in enumerate(groups):
        n_amb = int(ambiguous[list(g)].sum())
        supp = np.unique(np.concatenate([support_cols(j) for j in g]))
        rec = dict(track=t, gaussians=int(counts[t]) if t < len(counts) else 0,
                   support_gaussians=int(len(supp)),
                   dets=len(g), ambiguous_dets=n_amb)
        if n_amb == len(g):
            rec["reason"] = "all_ambiguous"
        elif counts[t] < floor:
            rec["reason"] = "fragment"
        else:
            keep[t] = True
            continue
        R.dropped_tracks.append(rec)
    ent["dropped_tracks"] = R.dropped_tracks
    inst[(inst >= 0) & ~keep[np.maximum(inst, 0)]] = -1

    # -- surviving tracks: records, connectivity (recorded), purity flag --------
    new_id = {t: i for i, t in enumerate(np.flatnonzero(keep))}
    inst = np.where(inst >= 0, np.vectorize(lambda t: new_id.get(t, -1))(inst)
                    if len(new_id) else -1, -1)
    ent["components"] = len(new_id)
    col_inst = np.full(m, -1, dtype=np.int64)
    col_inst[okc] = inst[col_pos[okc]]
    # §5 conflicting-support FLAG (inputs, not decisions): an instance whose
    # gaussians sit in the supports of BOTH members of a cannot-link pair,
    # each side at least min_instance_gaussians strong. A flag is not proof
    # of a merge (one object tiled into two masks flags too); the bank
    # checks every flag against the reviewed objects.
    flags_by_inst: dict[int, list] = defaultdict(list)
    n_new = len(new_id)
    for a, b, v in cl_pairs:
        ca, cb = col_inst[support_cols(a)], col_inst[support_cols(b)]
        na = np.bincount(ca[ca >= 0], minlength=max(n_new, 1))
        nb = np.bincount(cb[cb >= 0], minlength=max(n_new, 1))
        for i in np.flatnonzero((na >= floor) & (nb >= floor)):
            flags_by_inst[int(i)].append(dict(
                pair=[int(det_gids[a]), int(det_gids[b])],
                gaussians=[int(na[i]), int(nb[i])]))
    for t, g in enumerate(groups):
        if t not in new_id:
            continue
        i_new = new_id[t]
        g_sorted = sorted(g, key=lambda j: det_gids[j])
        sel = inst == i_new
        n_g = int(sel.sum())
        pts = means_np[class_gauss[sel]]
        comp = voxel_components(pts, lcfg["instance_voxel"]) if n_g else np.zeros(0)
        comp_sizes = np.bincount(comp).tolist() if n_g else []
        cents = [pts[comp == k].mean(0) for k in range(len(comp_sizes))]
        maxd = (max(float(np.linalg.norm(x - y)) for i, x in enumerate(cents)
                    for y in cents[i + 1:]) if len(cents) > 1 else 0.0)
        pair_aff = [float(aff_np[x, y]) for i, x in enumerate(g_sorted)
                    for y in g_sorted[i + 1:]]
        cols_t = np.flatnonzero(col_inst == i_new)
        amb_lost = len(amb_unassigned_by_track.get(t, ()))
        supp = np.unique(np.concatenate([support_cols(j) for j in g]))
        flags = flags_by_inst.get(i_new, [])
        ent["conflicting_support_flags"] += 1 if flags else 0
        rec = dict(
            track=i_new, class_track=t, dets=len(g),
            views=int(len({int(frames_local[j]) for j in g})),
            gaussians=n_g, support_gaussians=int(len(supp)),
            affinity_min=round(min(pair_aff), 4) if pair_aff else None,
            affinity_median=round(float(np.median(pair_aff)), 4) if pair_aff else None,
            margin=round(float(margin[cols_t].sum()), 3),
            margin_low_frac=round(float(low_margin[cols_t].mean()), 4) if len(cols_t) else 0.0,
            ambiguous_dets=int(ambiguous[g_sorted].sum()),
            ambiguous_unassigned=int(amb_lost),
            ambiguous_unassigned_frac=round(amb_lost / max(len(supp), 1), 4),
            track_components=dict(count=len(comp_sizes), sizes=comp_sizes,
                                  max_centroid_dist=round(maxd, 3)),
            conflicting_support=flags,
            bridge_refused=[k for k, r in enumerate(refused)
                            if track_of[local[r["edge"][0]]] == t
                            or track_of[local[r["edge"][1]]] == t],
            det_list=[dict(frame_idx=int(det_frame[det_gids[j]]),
                           concept=c, det_idx=int(det_didx[det_gids[j]]),
                           score=round(float(det_score[det_gids[j]]), 4),
                           source=det_source[det_gids[j]],
                           ambiguous=bool(ambiguous[j]),
                           gid=int(det_gids[j]))
                      for j in g_sorted])
        R.tracks.append(rec)
    ent["track_records"] = [{k: v for k, v in r.items() if k != "det_list"}
                            for r in R.tracks]
    R.gauss, R.inst_local = class_gauss, inst

    # -- calibration readouts (histograms; the raw matrices only on request)
    tri = np.triu_indices(n_dets, k=1)
    both = has[tri[0]] & has[tri[1]]
    is_cl = np.zeros((n_dets, n_dets), dtype=bool)
    for a, b, v in cl_pairs:
        is_cl[a, b] = is_cl[b, a] = True
    clm = is_cl[tri] & both
    same_view = (frames_local[tri[0]] == frames_local[tri[1]]) & both
    R.calib = dict(
        dets=n_dets, dets_with_support=int(has.sum()),
        cannot_link_pairs=len(cl_pairs),
        aff_hist_bins=50,
        aff_cannot_link=_hist(aff_np[tri][clm], 50, 0, 1),
        aff_same_view_other=_hist(aff_np[tri][same_view & ~clm], 50, 0, 1),
        aff_cross_view=_hist(aff_np[tri][both & ~same_view], 50, 0, 1),
        ios_cannot_link=_hist(ios_np[tri][clm], 50, 0, 1),
        ios_same_view_other=_hist(ios_np[tri][same_view & ~clm], 50, 0, 1),
        ios_cross_view=_hist(ios_np[tri][both & ~same_view], 50, 0, 1),
        candidate_edges=int(ok.sum()), merges_attempted=n_edges,
        elapsed_s=round(time.perf_counter() - t0, 2))
    if want_debug:
        R.debug = dict(det_gids=det_gids, aff=aff_np, ios=ios_np, size=size,
                       cannot_link=np.array(cl_pairs, dtype=np.float64).reshape(-1, 3),
                       track_of=track_of, ambiguous=ambiguous,
                       cols=cols_all, col_inst=col_inst,
                       best_val=best_val, second=second)
    return R
