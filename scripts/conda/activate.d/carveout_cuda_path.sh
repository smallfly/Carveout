#!/bin/sh
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: put the env-pinned CUDA toolkit first on PATH so the interactive `nvcc`
# matches the toolkit gsplat's JIT build actually uses. torch's cpp_extension resolves
# nvcc via $CUDA_HOME (set per-env with `conda env config vars set ... CUDA_HOME=...`),
# but the bare `nvcc` otherwise resolves to whatever system CUDA is on PATH (e.g. a
# 12.1 default), which is confusing. This aligns them. Resolves $CUDA_HOME at
# activation time so the env stays portable across rigs.
# GUARDED: no-ops unless $CUDA_HOME is set and holds an executable bin/nvcc, so a setup
# without a valid CUDA_HOME (or a system-toolkit-on-PATH setup) is left untouched.
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    export CARVEOUT_SAVED_PATH="${PATH:-}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
