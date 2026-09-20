// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Gate-state selectors: what the UI derives from the journal snapshot,
// defined ONCE. Every surface (rail, panels, overlays, dialogs) used to
// re-derive approved/stale/read-only and the demotion cascade by hand,
// which is how the reopen dialog's wording and the actual demotion could
// have drifted apart. The server (RunCore.gate_states) computes the
// states; this module only names the derivations.

import { GATES, type Gate, type Journal } from "./types";

/** "approved" (fresh) / "stale" / null (never approved). */
export const gateState = (j: Journal | null, g: Gate) =>
  j?.gates[g]?.state ?? null;

/** Another live driver holds the workdir — everything renders read-only. */
export const sceneReadOnly = (j: Journal | null) =>
  !!(j?.lock && !j.lock.mine && j.lock.alive);

export const staleGates = (j: Journal | null): Gate[] =>
  j ? GATES.filter((g) => j.gates[g]?.state === "stale") : [];

/** The cascade, named once: `g` and everything after it. */
export const downstream = (g: Gate): Gate[] => GATES.slice(GATES.indexOf(g));

/** What reopening `g` would actually demote — mirrors RunCore.reopen
 *  (downstream gates that currently hold an approval). */
export const demotedBy = (j: Journal | null, g: Gate): Gate[] =>
  j ? downstream(g).filter((d) => !!j.gates[d]) : [];

/** the gates BEFORE `g` that are not approved, in order — a stage
 *  of `g`'s territory, and `g`'s own approval, wait for them. Mirrors the
 *  server's require_gates_before, which refuses the same way. */
export const missingBefore = (j: Journal | null, g: Gate): Gate[] =>
  GATES.slice(0, GATES.indexOf(g)).filter(
    (e) => !j || gateState(j, e) !== "approved");

/** The act that clears the first missing gate, in the server's words. */
export const GATE_ACT: Record<Gate, string> = {
  volume: "review the volume and approve it",
  render: "review the rendered views and approve them",
  vocabulary: "confirm the vocabulary",
  exemplars: "review the exemplars and press Continue",
  verify_consent: "answer the verification consent",
};

/** "the volume gate is not approved — review the volume and approve it
 *  first." — the same sentence the route refuses with; null when nothing
 *  is missing. */
export const waitingOn = (j: Journal | null, g: Gate): string | null => {
  const m = missingBefore(j, g);
  return m.length ? `the ${m[0]} gate is not approved; ${GATE_ACT[m[0]]} first.`
                  : null;
};

/** The review gates the unattended pipeline needs approved first. */
export const reviewGates: Gate[] =
  GATES.filter((g) => g !== "verify_consent");

/** Which gate owns each 3D overlay: its overlay grays out when the
 *  approval no longer covers what is on screen. */
export const overlayStale = (j: Journal | null) => ({
  volume: gateState(j, "volume") === "stale",
  cameras: gateState(j, "render") === "stale",
  crops: gateState(j, "exemplars") === "stale",
});
