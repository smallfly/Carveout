# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""`carveout web`: the browser front-end server. Localhost, single user,
127.0.0.1 only, no auth, no telemetry. Sequencing semantics come verbatim
from carveout.sequencing (RunCore) — zero stage logic here: a gate rule
is written there once and the server can only call it.
"""
