# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The per-scene funnel/provenance report builders.

The FUNNEL REPORT quantifies, per class, where 2D evidence dies on its way
to 3D: raw logged detections -> above presence threshold -> surviving mask
dedup (mutual synonym / part-of suppression) -> gaussians labeled in scope
vs volume-scoped away -> 3D components -> exported instances (after the
zero-evidence export floor). It flags any class where more than half the
above-threshold evidence dies at a single filter — the per-scene health
view.

One caller: the sequencing core's run report (RunCore.write_report).
"""

import json
from collections import defaultdict
from pathlib import Path


def _effective_threshold(concept: str, cfg: dict, source: str = "text") -> float:
    """Lift's gate, verbatim — imported lazily because lift's module pulls
    in torch, too heavy for report/batch import time. ONE definition on
    purpose: this used to be a hand-maintained mirror, which is exactly how
    the funnel report and the lift would have come to disagree quietly."""
    from .lift import _effective_threshold as eff
    return eff(concept, cfg, source)


def build_funnel(workdir: str, cfg: dict) -> tuple[list[dict], list[str]]:
    """Per-class funnel rows + flag lines, from the stage artifacts on disk."""
    import csv

    wd = Path(workdir)
    manifest = json.loads((wd / "stage2/probe_manifest.json").read_text())
    negatives = set(manifest.get("negatives", []))
    classes = [p for p in manifest["prompts"] if p not in negatives]
    classes += [c for c in manifest.get("exemplar_concepts", [])
                if c not in classes]

    logged = defaultdict(int)
    above = defaultdict(int)
    overlay_ct = defaultdict(int)  # dedup operates on >= overlay_threshold dets
    ov = cfg["detect"]["overlay_threshold"]
    with open(wd / "stage2/scores.csv") as f:
        for r in csv.DictReader(f):
            c = r["concept"]
            if c in negatives:
                continue
            logged[c] += 1
            if float(r["score"]) >= _effective_threshold(
                    c, cfg, r.get("source") or "text"):
                above[c] += 1
            if float(r["score"]) >= ov:
                overlay_ct[c] += 1

    audit = json.loads((wd / "stage3/instance_audit.json").read_text())
    sup = audit.get("_mask_suppression", {})
    dedup = defaultdict(int)
    for m in sup.get("mutual", []):
        dedup[m["dropped"]] += 1
    for c, n in sup.get("partof_dropped_by_class", {}).items():
        dedup[c] += n
    for e in sup.get("cross_source", []):   # text+exemplar same-object
        dedup[e["concept"]] += 1
    carved = sup.get("carved_by_class", {})  # kept-via-policy, informational

    interactions = json.loads((wd / "stage4/interactions.json").read_text())
    exported = defaultdict(int)
    for o in interactions["objects"]:
        exported[o["label"]] += 1
    floor = interactions["extended"].get("export_floor", {})
    floor_dropped = defaultdict(int)
    for d in floor.get("dropped", []):
        floor_dropped[d["label"]] += 1

    rows, flags = [], []
    for c in classes:
        a = audit.get(c, {})
        g_in = a.get("gaussians", 0)
        g_out = a.get("out_of_volume_gaussians", 0)
        comps = a.get("components", 0)
        expect = a.get("expected_from_2d", 0)
        row = dict(cls=c, logged=logged[c], above=above[c], dedup=dedup[c],
                   g_in=g_in, g_out=g_out, expect=expect, comps=comps,
                   floor=floor_dropped[c], exported=exported[c])
        rows.append(row)
        # Flag: >50% of the evidence dies at ONE filter. The dedup machinery
        # operates on >= overlay_threshold dets, so its flag uses THAT
        # denominator — mixing it with the >=thr column produced ">100%"
        # reads like "81/57" (report bug).
        if above[c] > 0:
            if overlay_ct[c] > 0 and dedup[c] / overlay_ct[c] > 0.5:
                flags.append(f"{c}: {dedup[c]}/{overlay_ct[c]} "
                             f"overlay-threshold(>={ov}) dets die at MASK "
                             f"DEDUP (part-of/mutual)")
            if g_in + g_out > 0 and g_out / (g_in + g_out) > 0.5:
                flags.append(f"{c}: {g_out}/{g_in + g_out} labeled gaussians "
                             f"die at VOLUME SCOPING (outside volume+margin)")
            if g_in > 0 and comps == 0:
                flags.append(f"{c}: all {g_in} in-scope gaussians die at the "
                             f"3D SUPPORT FLOOR (min_instance_gaussians)")
        if comps > 0 and floor_dropped[c] / comps > 0.5:
            flags.append(f"{c}: {floor_dropped[c]}/{comps} components die at "
                         f"the EXPORT FLOOR (no matched detections)")
        if carved.get(c):
            flags.append(f"{c}: {carved[c]} part dets KEPT via "
                         f"specific-over-generic policy (carved from the "
                         f"generic mask)")
        # 3D UNDERCOUNT: the class delivered far fewer instances than its own
        # 2D evidence supports — the merged-blob shape (an umbrella concept
        # masking whole regions, or voxel connectivity welding neighbours).
        # The split loop already tries and correctly REFUSES to bisect such a
        # blob (a two-way cut of an N-object cloud shares 2D masks, so
        # _split_effective's distinctness guard rejects it — that guard is
        # what prevents the opposite regression). So this direction is
        # invisible unless it is reported: the funnel's other flags all
        # describe evidence DYING at a filter, and here the evidence survives
        # and lands in one instance. Bar at a third: calibrated on two
        # scene banks, it fires on the four real blobs (tool 2/28,
        # cables 2/14, books 4/26, box 5/24) and on nothing in a healthy run.
        if expect >= 3 and comps > 0 and comps * 3 <= expect:
            flags.append(f"{c}: {comps} instance(s) for {expect} expected "
                         f"from 2D ({g_in} labeled gaussians): 3D UNDERCOUNT "
                         f"(merged blob or umbrella term; see "
                         f"lift.specific_over_generic, "
                         f"detect.max_area_overrides, lift.instance_voxel)")
    return rows, flags


def build_provenance(workdir: str, cfg: dict) -> str | None:
    """Manual-view provenance section (None when the run had no manual views):
    per manual view (by label), the detections it produced; per surviving
    instance, whether manual views are its ONLY support — these "manual
    rescues" are the point of the feature."""
    import csv

    wd = Path(workdir)
    cams = json.loads((wd / "stage1/cameras.json").read_text())
    manual = {fr["frame_idx"]: fr.get("label", fr["provenance"])
              for fr in cams["frames"]
              if str(fr.get("provenance", "auto")).startswith("manual:")}
    if not manual:
        return None

    manifest = json.loads((wd / "stage2/probe_manifest.json").read_text())
    negatives = set(manifest.get("negatives", []))
    raw = defaultdict(int)
    above = defaultdict(int)
    with open(wd / "stage2/scores.csv") as f:
        for r in csv.DictReader(f):
            fidx = int(r["frame_idx"])
            if fidx not in manual or r["concept"] in negatives:
                continue
            raw[fidx] += 1
            if float(r["score"]) >= _effective_threshold(
                    r["concept"], cfg, r.get("source") or "text"):
                above[fidx] += 1

    ix = json.loads((wd / "stage4/interactions.json").read_text())
    lines = [f"manual views: {len(manual)} of {len(cams['frames'])} kept frames"]
    for fidx in sorted(manual):
        lines.append(f"  {manual[fidx]!r} (frame {fidx:04d}): "
                     f"{raw[fidx]} dets logged, {above[fidx]} >= threshold")
    rescues, mixed = [], 0
    for i, o in enumerate(ix["objects"]):
        sup = [e["frame_idx"] for e in o["frames"]]
        msup = sorted({f for f in sup if f in manual})
        if not msup:
            continue
        if len(msup) == len(set(sup)):
            rescues.append(f"  #{i} {o['label']}: via "
                           + ", ".join(repr(manual[f]) for f in msup))
        else:
            mixed += 1
    lines.append(f"manual rescues (instances supported ONLY by manual views): "
                 f"{len(rescues)}")
    lines.extend(rescues)
    lines.append(f"instances with mixed (manual + auto) support: {mixed}")
    return "\n".join(lines)


def build_exemplar_provenance(workdir: str, cfg: dict) -> str | None:
    """Exemplar provenance block (None when the run had no exemplars):
    per exemplar concept, the detections it produced; per surviving instance
    with exemplar support, exemplar-only rescue vs mixed — plus the
    cross-source dedup collapses (same object via both channels)."""
    import csv

    wd = Path(workdir)
    manifest = json.loads((wd / "stage2/probe_manifest.json").read_text())
    concepts = manifest.get("exemplar_concepts", [])
    if not concepts:
        return None

    raw = defaultdict(int)
    above = defaultdict(int)
    with open(wd / "stage2/scores.csv") as f:
        for r in csv.DictReader(f):
            src = r.get("source") or "text"
            if not src.startswith("exemplar:"):
                continue
            c = r["concept"]
            raw[c] += 1
            if float(r["score"]) >= _effective_threshold(c, cfg, src):
                above[c] += 1

    lines = [f"exemplar concepts: {len(concepts)} (exemplars.json "
             f"{manifest.get('exemplars_hash', 'absent')[:12]})"]
    for c in concepts:
        t = cfg["detect"]["exemplar_threshold_overrides"].get(
            c, cfg["detect"]["exemplar_threshold"])
        lines.append(f"  {c!r}: {raw[c]} dets logged, {above[c]} >= "
                     f"exemplar_threshold({t})")

    ix = json.loads((wd / "stage4/interactions.json").read_text())
    rescues, mixed = [], 0
    for i, o in enumerate(ix["objects"]):
        sup = ix["extended"]["objects"][i].get("support_by_source", {})
        ex_n = sum(n for s, n in sup.items() if s.startswith("exemplar:"))
        if not ex_n:
            continue
        if ex_n == sum(sup.values()):
            rescues.append(f"  #{i} {o['label']}: {ex_n} exemplar-matched "
                           f"frame(s)")
        else:
            mixed += 1
    audit = json.loads((wd / "stage3/instance_audit.json").read_text())
    n_collapse = len(audit.get("_mask_suppression", {})
                     .get("cross_source", []))
    lines.append(f"exemplar rescues (instances supported ONLY by exemplar "
                 f"detections): {len(rescues)}")
    lines.extend(rescues)
    lines.append(f"instances with mixed (text + exemplar) support: {mixed}")
    lines.append(f"cross-source dedup collapses (same object via both "
                 f"channels): {n_collapse}")
    return "\n".join(lines)


def format_funnel(rows: list[dict], flags: list[str]) -> str:
    head = (f"{'class':24s} {'dets':>5} {'>=thr':>5} {'dedup-':>6} "
            f"{'gauss-in':>9} {'gauss-out':>9} {'expect':>6} {'comps':>5} "
            f"{'floor-':>6} {'exported':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['cls']:24s} {r['logged']:5d} {r['above']:5d} {r['dedup']:6d} "
            f"{r['g_in']:9d} {r['g_out']:9d} {r.get('expect', 0):6d} "
            f"{r['comps']:5d} {r['floor']:6d} {r['exported']:8d}")
    lines.append("")
    lines.append("flags (>50% of a class's evidence dies at one filter, or "
                 "its 3D count falls far below the 2D evidence):")
    lines.extend(f"  ! {f}" for f in flags) if flags else lines.append("  none")
    return "\n".join(lines)


def build_pipeline_summary(workdir: str) -> dict:
    """What produced the export, for the report header: the
    instancing algorithm the lift ran (stage-3 manifest) and the ONE
    factor the run used with its source and, under an estimate, the size
    word (stage-1 manifest). Missing manifests read as None rather than
    refusing — a report may be rebuilt over an older workdir."""
    wd = Path(workdir)
    out: dict = dict(instancing=None, scale=None)
    m3 = wd / "stage3/manifest.json"
    if m3.exists():
        out["instancing"] = json.loads(m3.read_text()).get("instancing")
    m1 = wd / "stage1/manifest.json"
    if m1.exists():
        rec = json.loads(m1.read_text()).get("scale")
        if rec:
            out["scale"] = dict(source=rec.get("source", "recorded"),
                                scale_m_per_unit=rec.get("scale_m_per_unit"),
                                measured=rec.get("measured"))
    return out


def describe_pipeline(summary: dict) -> list[str]:
    """The header lines: the factor and where it came from — recorded, or
    measured with the ruler (then "≈", with what was measured)."""
    lines = [f"- instancing: {summary.get('instancing') or 'unknown'}"]
    sc = summary.get("scale")
    if not sc:
        lines.append("- scale: unknown (no stage-1 scale record)")
    elif sc["source"] == "measured":
        m = sc.get("measured") or {}
        what = (f"{m.get('label')}, " if m.get("label") else "") + (
            f"{float(m['length_m']):g} m" if m.get("length_m") else "one length")
        lines.append(f"- scale: 1 unit ≈ {float(sc['scale_m_per_unit']):.4g} m "
                     f"(measured: {what})")
    else:
        lines.append(f"- scale: {sc['scale_m_per_unit']:g} m per scene unit "
                     f"(recorded)")
    return lines
