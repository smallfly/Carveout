#!/bin/sh
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: restore PATH on deactivation (pairs with the activate.d CUDA-PATH script).
if [ -n "${CARVEOUT_SAVED_PATH+x}" ]; then
    export PATH="$CARVEOUT_SAVED_PATH"
    unset CARVEOUT_SAVED_PATH
fi
