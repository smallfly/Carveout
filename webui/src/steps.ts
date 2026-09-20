// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The ordered acts of each gate — what the panel's Steps list shows and
// what a dead primary says it waits for. Defined ONCE, like journal.ts:
// the button's reason and the route's refusal must not drift apart
// (the volume gate's Propose refuses without a scale, carveout/units.py
// RULER_HINT — the sentence here is the same shape).

import type { Step } from "./ui";

/** The panel's current step — the first not done, the one the Steps list
 *  marks — so the block that owns it can wear the cue (ui.tsx Cued /
 *  Section cue) from the same truth the list shows; null when all done. */
export const currentStep = (steps: Step[]): string | null =>
  steps.find((s) => !s.done)?.id ?? null;

/** Why Propose (and Whole scene) is dead on the volume gate, in words;
 *  null when a scale is recorded. */
export const SCALE_ACT =
  "measure one thing you know on the canvas, or press Metric (1.0)";
export const scaleWaiting = (factor: number | null): string | null =>
  factor == null
    ? `no scale is recorded for this scene; ${SCALE_ACT} first.` : null;

/** One list for both cases: with a known scale the first step renders
 *  ticked, which still teaches the order. "Reviewed" has no flag of its
 *  own — inventing one would be a fake gate — so step 3 completes with
 *  the approval it leads to. */
/** The level step: Propose reads the tilt, so the step sits after
 *  it — done once a proposal exists and the scene reads level, or the
 *  adopted alignment is the one it proposed; `reason` is the check's
 *  sentence, shown while the step is current. */
export interface LevelState { known: boolean; offLevel: boolean;
                              adopted: boolean; reason: string | null }

export function volumeSteps(o: { factor: number | null; hasVolume: boolean;
                                 approved: boolean;
                                 level?: LevelState | null;
                                 /** the boxes were proposed under another
                                  *  levelling than the one recorded now */
                                 volumeStale?: boolean }): Step[] {
  const lv = o.level;
  return [
    { id: "scale", label: "Record the scale", done: o.factor != null,
      note: SCALE_ACT },
    { id: "propose", label: "Propose a volume", done: o.hasVolume && !o.volumeStale,
      note: o.volumeStale
        ? "these boxes were proposed before the levelling; press Propose fresh again"
        : "Propose fresh builds boxes from the scene's density" },
    { id: "level", label: "Level the scene",
      done: !!lv && lv.known && (!lv.offLevel || lv.adopted),
      note: lv?.known ? (lv.reason ?? "level as proposed, or from three points on the floor")
                      : "Propose reads the tilt" },
    { id: "review", label: "Check the floor, the up axis and the boxes",
      done: o.approved, note: "drag the handles, or fix the frame above" },
    { id: "approve", label: "Approve the volume", done: o.approved },
  ];
}

/** The exemplar gate: crops are optional, a probe of them is
 *  not — with none, the probe step reads done and Continue is the act. */
export function exemplarSteps(o: { crops: number; probedCurrent: boolean;
                                   approved: boolean }): Step[] {
  return [
    { id: "crops", label: "Add example crops, if a concept is blocked",
      done: o.crops > 0 || o.approved,
      note: "only for concepts the text probe proved unreachable; with none, continue" },
    { id: "reprobe", label: "Re-probe the crops", done: o.probedCurrent,
      note: "the example-image pass; its rows join the probe results" },
    { id: "continue", label: "Continue", done: o.approved },
  ];
}

export function vocabSteps(o: { hasPrompts: boolean; probedCurrent: boolean;
                                approved: boolean }): Step[] {
  return [
    { id: "list", label: "Build the word list", done: o.hasPrompts,
      note: "add terms, or propose with the vision model" },
    { id: "probe", label: "Probe the words", done: o.probedCurrent,
      note: "Run probe scores every term on the rendered views" },
    { id: "confirm", label: "Confirm the vocabulary", done: o.approved,
      note: "confirming starts detection on these words" },
  ];
}

/** The Objects point: the extraction (detect → lift → export) runs from
 *  the panel once the four review points are approved; the decision that
 *  approves it is the verification — run or skip, both recorded. */
export function objectsSteps(o: { gatesReady: boolean; missing: string[];
                                  hasExport: boolean; decided: boolean }): Step[] {
  return [
    { id: "extract", label: "Extract the objects", done: o.hasExport,
      note: o.gatesReady
        ? "Run detect → lift → export segments every object out of the scene"
        : `once ${o.missing.join(", ")} ${o.missing.length === 1 ? "is" : "are"} approved` },
    { id: "verify", label: "Verify their labels, or skip", done: o.decided,
      note: "the vision model re-examines each object; either choice is recorded" },
  ];
}

/** `manual`: the operator's own viewpoints are the camera set —
 *  the first step is to capture them, and the render draws exactly those;
 *  `saved` is how many are on disk, rendered or not. */
export function renderSteps(o: { hasCams: boolean; pending: number;
                                 approved: boolean; manual?: boolean;
                                 saved?: number }): Step[] {
  const capture: Step[] = o.manual ? [
    { id: "capture", label: "Capture your viewpoints", done: (o.saved ?? 0) > 0,
      note: "Capture view saves the canvas camera; save several, from different sides" },
  ] : [];
  return [
    ...capture,
    { id: "render", label: o.manual ? "Render them" : "Render the views",
      done: o.hasCams && o.pending === 0,
      note: o.manual ? "the render draws exactly the viewpoints you saved"
          : o.pending > 0 ? "Re-render applies the views you added, moved or deleted"
                          : "Carveout picks the views" },
    { id: "review", label: "Review the sheet and the discards",
      done: o.approved, note: "check coverage and what was thrown out" },
    { id: "approve", label: "Approve the render", done: o.approved },
  ];
}
