# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ONE refusal type of the gated pipeline.

A refusal is an operator-visible "no" with the remedy in the message:
surfaced verbatim, routed to the gate that owns it, never a traceback
that reads like a crash (driver contract). Stages and the sequencing
core raise it directly; the two front-end edges convert it — the CLI to
a clean SystemExit, the web dispatcher to a routed JSON response. Before
this type existed the contract was a convention spread over four
exception types (SystemExit, RuntimeError, FileNotFoundError, and a
web-only Refusal), and each catch-site re-decided what was a refusal.

`gate` may stay None at a raise site that cannot know it (run_detect
serves probe, re-probe and confirm jobs owned by different gates); the
web job runner falls back to the running job's owning gate.
"""


DEPENDENCY_HELP = """\
ERROR: {what} is not importable in this environment ({err}).

The GPU stages need the pinned set from environment.yml: torch 2.7.x for
CUDA 12.8, gsplat, and Meta's sam3 package, installed by:

  conda env create -f environment.yml    # or: conda env update -f environment.yml
  conda activate carveout

scripts/smoke_test_torch.py, smoke_test_gsplat.py and smoke_test_sam3.py
check each one on its own.
"""


def missing_dependency(what: str, err: BaseException) -> "Refusal":
    """The refusal for a heavy dependency that does not import: the message
    is the remedy, never a bare ModuleNotFoundError from inside a stage."""
    return Refusal(DEPENDENCY_HELP.format(what=what, err=err))


class Refusal(RuntimeError):
    """Operator-visible refusal: message shown verbatim, routed to a gate.

    `status` is the HTTP status the web edge answers with; the CLI edge
    ignores it and exits with the message."""

    def __init__(self, message: str, gate: str | None = None,
                 remedy: str | None = None, status: int = 409):
        super().__init__(message)
        self.gate = gate
        self.remedy = remedy
        self.status = status
