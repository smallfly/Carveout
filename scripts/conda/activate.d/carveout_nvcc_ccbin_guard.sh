#!/bin/sh
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: neutralize a leaked NVCC_CCBIN that points at a missing host compiler.
# gsplat's first-use JIT build invokes nvcc, which obeys $NVCC_CCBIN as its host
# compiler. If an unrelated shell/project exported e.g.
# NVCC_CCBIN=x86_64-conda-linux-gnu-c++ (a conda toolchain not installed here), nvcc
# dies with "nvcc fatal : Failed to preprocess host compiler properties" and the whole
# build fails — reproducibility must not depend on ambient shell state. When the target
# isn't resolvable to an executable, unset it for THIS env (nvcc then falls back to the
# working system g++) and warn loudly. A VALID NVCC_CCBIN is left untouched (deliberate
# choice). Restored on deactivate so the ambient shell is unchanged outside carveout.
if [ -n "${NVCC_CCBIN:-}" ] && ! command -v "$NVCC_CCBIN" >/dev/null 2>&1; then
    export CARVEOUT_SAVED_NVCC_CCBIN="$NVCC_CCBIN"
    unset NVCC_CCBIN
    echo "carveout: NVCC_CCBIN='$CARVEOUT_SAVED_NVCC_CCBIN' points at a host compiler that is not on PATH; unset for this env so gsplat's nvcc build uses the system default (g++). Restored on deactivate." >&2
fi
