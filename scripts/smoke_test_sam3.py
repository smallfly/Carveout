# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke test: SAM 3 image PCS runs from the project-local checkpoint.

Run on each rig: python scripts/smoke_test_sam3.py [image_path] [prompt]
Defaults to the sam3 repo's truck test image with prompt "truck".
"""

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CKPT = PROJECT / "models" / "sam3" / "sam3.pt"
CKPT_31 = PROJECT / "models" / "sam3" / "sam3.1_multiplex.pt"

GATED_HELP = f"""
ERROR: SAM 3 checkpoint missing: {{path}}

The checkpoints are GATED on Hugging Face:
  1. Request access at https://huggingface.co/facebook/sam3
     (and https://huggingface.co/facebook/sam3.1 for the video/tracking model)
  2. Authenticate: conda activate carveout && hf auth login
  3. Download into the project (never a global cache):
       hf download facebook/sam3 sam3.pt config.json --local-dir {PROJECT}/models/sam3/
       hf download facebook/sam3.1 sam3.1_multiplex.pt config.json --local-dir {PROJECT}/models/sam3/
"""


def main() -> None:
    for path in (CKPT, CKPT_31):
        if not path.exists():
            sys.exit(GATED_HELP.format(path=path))

    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(PROJECT / "reference" / "sam3" / "assets" / "images" / "truck.jpg")
    )
    prompt = sys.argv[2] if len(sys.argv) > 2 else "truck"

    t0 = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=str(CKPT), load_from_HF=False)
    processor = Sam3Processor(model)
    t_load = time.perf_counter() - t0
    print(f"model loaded from {CKPT.name} in {t_load:.1f} s")

    image = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    # bf16 autocast is required: the upstream example notebooks enter a global
    # torch.autocast("cuda", bfloat16) before inference; without it the image
    # model fails with a BFloat16/Float dtype mismatch in its linear layers.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"PCS '{prompt}' on {Path(image_path).name}: {len(scores)} instance(s), "
        f"scores={[round(float(s), 3) for s in scores]} | "
        f"{t_infer:.2f} s | peak VRAM {peak_gb:.1f} GB"
    )
    assert len(scores) > 0, "expected at least one detection on the reference image"
    print("OK")


if __name__ == "__main__":
    main()
