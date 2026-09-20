# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-scene exemplar prompts : visual concept references.

An exemplar backs a concept with one or more reference crops drawn on stage-1
frames at the Exemplars gate, stored in <workdir>/exemplars.json. Stage 2
feeds each crop to SAM 3's
video-multiplex predictor as a visual concept prompt: the detector finds every
instance matching the crop on its frame, the tracker propagates them across
all frames. Detections enter scores.csv ADDITIVELY with
source=exemplar:<concept> and are gated downstream by
detect.exemplar_threshold — an exemplar-similarity scale, NEVER compared
against text presence thresholds.

Cache contract (mirrors manual_views): exemplars.json CONTENT is recorded in
the stage-2 manifest (exemplars_hash) and cascades downstream. A changed file
re-runs ONLY the exemplar pass (text results are kept and folded), announced
loudly; text-side changes still invalidate everything. Each crop additionally
records frame_sha256 of the stage-1 frame it was drawn on: if a re-render
changed that frame's pixels, detect REFUSES with a re-draw message instead of
silently prompting against different content.
"""

import json
import logging
from pathlib import Path

from .refusal import Refusal
from .sequencing import fingerprint

log = logging.getLogger(__name__)


def resolve_exemplars_path(workdir, cfg: dict) -> Path:
    """detect.exemplars override, else <workdir>/exemplars.json (the scene's
    files live in the workdir — sequencing.SCENE_FILES). A missing file is
    zero exemplars, the silent default."""
    override = cfg["detect"].get("exemplars")
    return Path(override) if override else Path(workdir) / "exemplars.json"


def load_exemplars(path) -> list[dict]:
    """Parse exemplars.json; [] when path is None or the file does not exist.

    Returns dicts: concept, crops[{frame_idx, box_xyxy, frame_sha256, note}].
    Unknown version and malformed entries are rejected with clear errors.
    """
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if data.get("version") != 1:
        raise ValueError(
            f"{p}: unsupported exemplars version {data.get('version')!r} "
            f"(this pipeline reads version 1)")
    exemplars = []
    for i, e in enumerate(data.get("exemplars", [])):
        concept = str(e.get("concept") or "").strip()
        if not concept:
            raise ValueError(f"{p} exemplar {i}: empty concept")
        crops = e.get("crops") or []
        if not crops:
            raise ValueError(f"{p} exemplar {i} ({concept!r}): no crops")
        for j, c in enumerate(crops):
            where = f"{p} exemplar {i} ({concept!r}) crop {j}"
            box = c.get("box_xyxy")
            if (not isinstance(c.get("frame_idx"), int)
                    or c["frame_idx"] < 0
                    or not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(v, (int, float)) for v in box)
                    or not (box[0] < box[2] and box[1] < box[3])):
                raise ValueError(f"{where}: malformed frame_idx/box_xyxy")
            if not c.get("frame_sha256"):
                raise ValueError(f"{where}: missing frame_sha256 (crops must "
                                 f"be drawn at the Exemplars gate, which "
                                 f"stamps it)")
        exemplars.append(dict(concept=concept, crops=crops))
    if exemplars:
        log.info("loaded %d exemplar concept(s) from %s (%d crop(s))",
                 len(exemplars), p, sum(len(e["crops"]) for e in exemplars))
    return exemplars


def stage2_exemplar_record(workdir) -> dict:
    """stage2 manifest's exemplar record ({path, hash, concepts}).

    {'hash': 'absent'} for a missing stage2 or a pre-feature manifest.
    """
    mp = Path(workdir) / "stage2/probe_manifest.json"
    if not mp.exists():
        return {"hash": "absent"}
    m = json.loads(mp.read_text())
    return {"path": m.get("exemplars_path"),
            "hash": m.get("exemplars_hash", "absent"),
            "concepts": m.get("exemplar_concepts", [])}


def stale_exemplars(cached_manifest: dict, workdir, stage: str) -> bool:
    """True (with one unmissable log line) when this stage's cached manifest
    was produced against different exemplar content than stage2's."""
    cur = stage2_exemplar_record(workdir).get("hash", "absent")
    rec = cached_manifest.get("exemplars_hash", "absent")
    if cur == rec:
        return False
    log.warning(
        "%s cache INVALIDATED: exemplars changed upstream (this stage saw "
        "%s, stage2 now has %s); re-running %s from scratch",
        stage, rec[:12], cur[:12], stage)
    return True


def check_crop_frames(exemplars: list[dict], stage1: Path) -> None:
    """REFUSE when any crop's source frame no longer has the pixels it was
    drawn on (a re-render changed stage1): the box would be encoded against
    different content. The remedy is re-drawing, not silence."""
    for e in exemplars:
        for c in e["crops"]:
            fp = stage1 / "frames" / f"frame_{c['frame_idx']:04d}.png"
            live = fingerprint(fp)
            if live != c["frame_sha256"]:
                raise Refusal(
                    f"exemplar crop for {e['concept']!r} was drawn on "
                    f"frame {c['frame_idx']:04d} whose content has changed "
                    f"({c['frame_sha256'][:12]} -> {live[:12]}); stage1 was "
                    f"re-rendered since the crop was drawn. Re-draw it at "
                    f"the Exemplars gate ({e['concept']!r}, frame "
                    f"{c['frame_idx']:04d}).", gate="exemplars")
