#!/bin/sh
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: CXXABI_1.3.15 workaround (see the README's installation notes).
# Resolves $CONDA_PREFIX at activation time so the env stays portable across rigs.
# GUARDED: no-ops unless the marker file below exists. Enable only if/when the
# CXXABI error actually appears:
#   touch "$CONDA_PREFIX/etc/conda/enable_libstdcxx_preload"
# (and install conda-forge libstdcxx-ng>=13 into the env first).
if [ -f "$CONDA_PREFIX/etc/conda/enable_libstdcxx_preload" ]; then
    export CARVEOUT_SAVED_LD_PRELOAD="${LD_PRELOAD:-}"
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
