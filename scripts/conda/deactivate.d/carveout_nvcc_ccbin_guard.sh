#!/bin/sh
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: restore NVCC_CCBIN on deactivation (pairs with the activate.d guard).
if [ -n "${CARVEOUT_SAVED_NVCC_CCBIN+x}" ]; then
    export NVCC_CCBIN="$CARVEOUT_SAVED_NVCC_CCBIN"
    unset CARVEOUT_SAVED_NVCC_CCBIN
fi
