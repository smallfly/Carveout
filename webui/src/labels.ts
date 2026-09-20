// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Display names for the gates. PRESENTATION ONLY — the journal, the API
// and every payload keep the internal identifiers (`verify_consent`);
// these are what the rail, the library and the header show for them.
// The fifth point is named for what lands there — the objects segmented
// out of the scene — not for its last act:
// the extraction runs from that panel, and the verification decides it.

import type { Gate } from "./types";

export const GATE_LABEL: Record<Gate, string> = {
  volume: "Volume", render: "Render", vocabulary: "Vocabulary",
  exemplars: "Exemplars", verify_consent: "Objects",
};

/** Compact form for tight spots (library chips); the full name rides the
 *  accessible name. */
export const GATE_SHORT: Record<Gate, string> = {
  volume: "VOL", render: "REN", vocabulary: "VOC", exemplars: "EXE",
  verify_consent: "OBJ",
};

/** The gate waiting on the operator, in words. Authoritative input only:
 *  the journal's active_gate (null once every gate is approved). The
 *  verify gate waits for a decision (run or skip), the review gates for an
 *  approval. */
export function describeAwaiting(j: { active_gate: Gate | null } | null)
    : string | null {
  if (!j) return null;
  const g = j.active_gate;
  if (!g) return "All gates approved";
  return g === "verify_consent" ? "Awaiting verification decision"
                                : `Awaiting ${GATE_LABEL[g]} approval`;
}

/** The jobs the server runs, in words. PRESENTATION ONLY, like
 *  GATE_LABEL: the journal's `running.kind` and the stage events keep the
 *  internal ids (`propose_prompts`), and the log drawer still shows them
 *  — the log is what those ids index. An unknown kind falls through
 *  unchanged rather than hiding. */
export const JOB_LABEL: Record<string, string> = {
  propose_volume: "Proposing volume",
  render: "Rendering views",
  calibration: "Writing calibration",
  probe: "Probing vocabulary",
  propose_prompts: "Proposing vocabulary",
  confirm_vocabulary: "Confirming vocabulary",
  reprobe_exemplars: "Re-probing exemplars",
  threshold_override: "Applying the threshold",
  pipeline: "Detecting, lifting, exporting",
  verify: "Verifying labels",
};
export const jobLabel = (kind: string | null | undefined): string =>
  (kind && JOB_LABEL[kind]) || kind || "";

/** A job's state in words — the same phrase wherever it appears. `stage`
 *  is the live stage event (running | done | cancelled | failed); null =
 *  nothing has run in this session. */
export function describeJob(stage: { stage: string; state: string } | null)
    : string {
  if (!stage) return "No job running";
  const word = stage.state === "running" ? "running"
    : stage.state === "done" ? "finished"
    : stage.state === "failed" ? "failed"
    : stage.state === "cancelled" ? "cancelled" : stage.state;
  return `${jobLabel(stage.stage)} ${word}`;
}
