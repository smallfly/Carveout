#!/bin/sh
# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

# Carveout: restore LD_PRELOAD on deactivation (pairs with activate.d script).
if [ -n "${CARVEOUT_SAVED_LD_PRELOAD+x}" ]; then
    if [ -n "$CARVEOUT_SAVED_LD_PRELOAD" ]; then
        export LD_PRELOAD="$CARVEOUT_SAVED_LD_PRELOAD"
    else
        unset LD_PRELOAD
    fi
    unset CARVEOUT_SAVED_LD_PRELOAD
fi
