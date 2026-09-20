// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The one place a panel decides how to label a LENGTH knob. The field holds
// the profile's number — a metre setting the stage divides by the scene's
// one factor. The label says "(m)" under a recorded factor and "(≈m)"
// after a measurement; with no scale yet the bare label.
// The hint gives what the number resolves to in scene units, which is
// what the stage actually compares.

import React from "react";

export const fmtUnits = (v: number) => Number(v.toPrecision(4)).toString();

/** Label + hint for a length knob at the scene's factor: `factor` is metres
 *  per unit (null = no scale yet), `approx` whether it was measured with
 *  the ruler (one known length: the metres are "≈"). */
export function lengthLabel(base: string, value: number | null | undefined,
                            factor: number | null, approx = false) {
  const label = factor ? `${base} (${approx ? "≈" : ""}m)` : base;
  const hint = factor && typeof value === "number" && factor !== 1
    ? React.createElement("span",
        { className: "text-t4 text-[0.6875rem] font-mono whitespace-nowrap" },
        `= ${fmtUnits(value / factor)} units`)
    : undefined;
  return { label, hint };
}
