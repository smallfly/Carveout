#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: post-create configuration for the `carveout` conda env.
# Run AFTER `conda env create -f environment.yml`, from anywhere:
#   bash scripts/setup_env.sh
# Idempotent. Applies:
#   - env-scoped vars (TORCH_CUDA_ARCH_LIST, HF_HOME/TORCH_HOME → <repo>/models/)
#   - guarded libstdc++ LD_PRELOAD activate.d/deactivate.d scripts (resolve
#     $CONDA_PREFIX at activation; disabled until the CXXABI_1.3.15 error is
#     actually observed — see README.md)
# NOTE: the CUDA 12.8 toolkit is a SYSTEM dependency; point the env at it with
#   conda env config vars set -n carveout CUDA_HOME=/usr/local/cuda-12.8
# (see README "CUDA toolkit"). setup_env.sh does not set CUDA_HOME (rig-specific path).
set -euo pipefail

# The env name is the user's choice, not the project's: anyone who already
# has a `carveout` env, or who keeps one env per checkout, would otherwise
# have to edit this file. Worse, running it from a SECOND checkout would
# silently re-point the FIRST checkout's env, since it looks the name up
# and finds the existing one. Override with ENV_NAME=...
ENV_NAME="${ENV_NAME:-carveout}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')"
if [ -z "$ENV_PREFIX" ] || [ ! -d "$ENV_PREFIX" ]; then
    echo "ERROR: conda env '$ENV_NAME' not found. Create it first:" >&2
    echo "  conda env create -f $PROJECT_DIR/environment.yml" >&2
    exit 1
fi

# TORCH_EXTENSIONS_DIR: gsplat JIT-compiles its CUDA kernels on first use; the
# default cache is ~/.cache/torch_extensions, which violates the "nothing in
# ~/.cache" rule — keep build artifacts in-project alongside the model caches.
# PYTORCH_KERNEL_CACHE_PATH: torch's fused-kernel cache defaults to
# ~/.cache/torch/kernels — redirect it in-project for the same reason.
conda env config vars set -n "$ENV_NAME" \
    TORCH_CUDA_ARCH_LIST="8.9;12.0" \
    HF_HOME="$PROJECT_DIR/models/hf" \
    TORCH_HOME="$PROJECT_DIR/models/torch" \
    TORCH_EXTENSIONS_DIR="$PROJECT_DIR/models/torch/extensions" \
    PYTORCH_KERNEL_CACHE_PATH="$PROJECT_DIR/models/torch/kernels"

mkdir -p "$ENV_PREFIX/etc/conda/activate.d" "$ENV_PREFIX/etc/conda/deactivate.d"
cp "$PROJECT_DIR"/scripts/conda/activate.d/*.sh \
   "$ENV_PREFIX/etc/conda/activate.d/"
cp "$PROJECT_DIR"/scripts/conda/deactivate.d/*.sh \
   "$ENV_PREFIX/etc/conda/deactivate.d/"

echo "OK: env vars set and activate.d scripts installed for '$ENV_NAME'."
echo "Re-activate the env to pick up changes: conda activate $ENV_NAME"
