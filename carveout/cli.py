# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""`carveout` — the command line is one command: `carveout web`.

Carveout is a web tool. Every stage is a function in its module
(`run_render`, `run_detect`, `run_lift`, `run_export`, `run_verify`,
`run_propose_prompts`) that the web server imports directly, and there
is no other way to run one: one front-end means one set of gate rules,
and the web app is where the review happens.
"""

# default the CUDA allocator to expandable segments BEFORE torch is
# imported anywhere (it reads this at CUDA-context init). Recovers the
# reserved-but-unallocated fragmentation that tips the 24 GB exemplar video
# pass over on a 4090. setdefault so an explicit env override still wins.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys

from .config import load_config, set_profiles_dir, set_vlm_dir
from .refusal import Refusal


def main(argv: list[str] | None = None) -> int:
    # The CLI edge of the refusal contract: `serve` raises the core Refusal
    # (a non-localhost host, a missing frontend bundle); here it becomes a
    # clean exit with the message, never a traceback. (The web edge converts
    # the same type to a routed JSON response.)
    try:
        return _main(argv)
    except Refusal as e:
        raise SystemExit(str(e)) from None


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(prog="carveout")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "web", help="the Carveout app: one browser front-end over the "
                    "sequencing core: library, all five gates, live stage "
                    "progress, report, integrated 3D viewer. Localhost only "
                    "(127.0.0.1); the frontend bundle is operator-built: "
                    "`npm --prefix webui ci && npm --prefix webui run build` "
                    "(Node is a build-time dependency only).")
    p.add_argument("--port", type=int, default=8090,
                   help="port on 127.0.0.1 (default 8090)")
    p.add_argument("--profile", default=None, choices=["24gb", "32gb"],
                   help="VRAM profile (default: configs/default.yaml "
                        "`profile`, the 4090-safe 24gb block; 32gb on a "
                        "5090)")
    p.add_argument("--profiles-dir", default=None, metavar="DIR",
                   help="scene profiles directory for THIS server (default "
                        "configs/scenes/). The hook for a spare server "
                        "over scratchpad copies; the operator's server "
                        "never needs it.")
    p.add_argument("--vlm-dir", default=None, metavar="DIR",
                   help="the vision model this server uses for scenes that "
                        "did not choose one (default: configs/default.yaml "
                        "`vlm.model_dir`, Qwen3.8-27B). Each scene chooses "
                        "its own at creation and on the vocabulary / verify "
                        "panels, from the models installed under models/.")

    args = ap.parse_args(argv)

    if args.cmd == "web":
        try:
            if args.profiles_dir:
                set_profiles_dir(args.profiles_dir)
            if args.vlm_dir:
                set_vlm_dir(args.vlm_dir)
        except FileNotFoundError as e:
            # a start-time refusal: the message is the remedy, no traceback
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        # The web server loads scene profiles lazily, per scene, by name.
        from .web.server import serve
        serve(args.port,
              lambda name: load_config(args.profile, scene_config=name))
        return 0

    raise AssertionError(f"unhandled subcommand {args.cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
