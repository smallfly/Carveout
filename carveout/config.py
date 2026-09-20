# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""YAML config loading: the defaults, a VRAM profile, a scene profile.

Scene profiles live in ONE directory, `configs/scenes/` (gitignored — a
profile carries a scene's calibration and vocabulary, and none ships).
The web server may point the process at another directory with
`carveout web --profiles-dir` — the hook a spare server uses to run
against scratchpad copies without touching the operator's profiles. One
process, one directory; there is no search path and no machine-local
override file: the env var and `configs/local.yaml` that once resolved a
search path were removed, so a profile is always where the server says.
"""

import copy
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
PROFILES_DIR: Path = DEFAULT_CONFIG_PATH.parent / "scenes"
# The vision model directory for THIS process (`carveout web --vlm-dir`),
# None = the config's `vlm.model_dir`. Applied by load_config on every
# load, so the stages, the settings dialog and the stage-4.5 fingerprint
# (verify_params carries model_dir) all see the same choice.
VLM_DIR: Path | None = None


def set_profiles_dir(path: str | Path) -> Path:
    """Point this process at another profile directory (`carveout web
    --profiles-dir`). Refuses a path that is not a directory: a typo here
    would route every profile write to the wrong tree."""
    global PROFILES_DIR
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"profiles directory {p} does not exist; create it first")
    PROFILES_DIR = p
    log.info("profiles dir: %s (--profiles-dir)", p)
    return p


def set_vlm_dir(path: str | Path) -> Path:
    """Point this process at another vision model (`carveout web
    --vlm-dir`): a smaller checkpoint for a card the default does not fit
    beside a large scene. A local directory only — never a hub id, so the
    flag cannot make a run download anything — and it must already hold a
    `config.json`, so a typo refuses at start rather than at the gate."""
    global VLM_DIR
    p = Path(path).expanduser().resolve()
    if not (p / "config.json").is_file():
        raise FileNotFoundError(
            f"vision model directory {p} has no config.json; download the "
            f"model into it first (README step 7), or start without --vlm-dir")
    VLM_DIR = p
    log.info("vision model: %s (--vlm-dir)", p)
    return p


def model_entry(cfg: dict, model_dir: str | Path) -> dict:
    """What the catalogue knows about a model directory: its display name,
    download repo and measured 4-bit peak — by the directory's basename;
    an unknown directory is shown by that basename with nothing measured."""
    base = Path(model_dir).name
    known = (cfg.get("vlm") or {}).get("catalogue") or {}
    e = dict(known.get(base) or {})
    e.setdefault("name", base)
    e.setdefault("hf_repo", None)
    e.setdefault("vram_4bit_gib", None)
    return e


def apply_model_choice(cfg: dict, config_dir: str) -> None:
    """One resolved model per config: `vlm.model_dir` may come from the
    defaults, the server flag or the scene's profile (that order of
    precedence, last wins); the display name, the download hint and the
    measured peak follow the DIRECTORY through the catalogue, never a
    stale value from the defaults."""
    v = cfg["vlm"]
    v.setdefault("model_source", "config")
    e = model_entry(cfg, v["model_dir"])
    v["model"] = e["name"]
    v["hf_repo"] = e["hf_repo"]
    v["vram_4bit_gib"] = e["vram_4bit_gib"]


def find_profile(name: str) -> Path | None:
    """The profile file for a scene name, or None when the scene has none."""
    p = PROFILES_DIR / f"{name}.yaml"
    return p if p.exists() else None


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(profile: str | None = None,
                scene_config: str | None = None) -> dict:
    """Load configs/default.yaml, apply the VRAM profile, then the scene
    profile.

    Precedence: defaults < VRAM profile < scene profile. `scene_config` is
    a scene NAME resolved in the profile directory, or a PATH to a profile
    file — the web core loads every scene by the resolved path, so a run's
    writes and gate hashes are bound to one file.
    """
    path = DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    profiles = cfg.pop("profiles", {})
    name = profile or cfg.get("profile", "24gb")
    if name not in profiles:
        raise ValueError(
            f"unknown profile {name!r}; available: {sorted(profiles)} (config: {path})"
        )
    cfg["profile"] = name
    cfg = _deep_merge(cfg, profiles[name])
    config_dir = cfg["vlm"]["model_dir"]
    if VLM_DIR is not None:
        # the server's default for scenes that did not choose
        cfg["vlm"]["model_dir"] = str(VLM_DIR)
        cfg["vlm"]["model_source"] = "--vlm-dir"

    if scene_config:
        sp = Path(scene_config)
        if not sp.exists():
            sp = find_profile(scene_config)
        if sp is None:
            raise FileNotFoundError(
                f"scene profile {scene_config!r} not found in {PROFILES_DIR}")
        with open(sp) as f:
            raw = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, raw)
        if (raw.get("vlm") or {}).get("model_dir"):
            cfg["vlm"]["model_source"] = "scene"
        cfg["scene_config"] = str(sp)
        log.info("scene profile applied: %s", sp)
    apply_model_choice(cfg, config_dir)

    log.info("config loaded from %s (profile %s)", path, name)
    return cfg
