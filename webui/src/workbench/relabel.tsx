// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The operator's relabel, one control in two places: the report
// drawer's instance rows and the selected-object card on the canvas.
// Click the pencil, type, Enter; Escape or blur leaves it. The detected
// label stays beside the new name; an empty entry takes the name back.
// PUT /instances/<i>/label; the journal snapshot it publishes refreshes
// every /instances reader, the caller's own reload makes it immediate.

import { useState } from "react";
import { useSceneAct } from "../store";
import type { InstanceRow } from "../types";
import { INPUT, cx } from "../ui";

/** The name the display rule shows: the operator's, else the verifier's,
 *  else the detected one. */
export const shownName = (o: InstanceRow) =>
  o.operator_label ?? o.verified_label ?? o.label;

/** The verifier's proposal for an object when it is not already the
 *  name shown: the button that takes it has something to do. */
export const proposalOf = (o: InstanceRow) =>
  o.proposed_label && o.proposed_label !== shownName(o) ? o.proposed_label : null;

export function useRelabel(onSaved?: () => void) {
  const { act, busy } = useSceneAct();
  // One click takes the verifier's proposal as the operator's name — the
  // same write as typing it.
  const take = async (o: InstanceRow) => {
    const p = proposalOf(o);
    if (!p) return;
    const r = await act(`/instances/${o.idx}/label`, { label: p },
                        { method: "put" });
    if (r) onSaved?.();
  };
  const [editing, setEditing] =
    useState<{ idx: number; text: string } | null>(null);
  const commit = async () => {
    if (!editing) return;
    const { idx, text } = editing;
    setEditing(null);
    const r = await act(`/instances/${idx}/label`, { label: text },
                        { method: "put" });
    if (r) onSaved?.();
  };
  const isEditing = (o: InstanceRow) => editing?.idx === o.idx;
  const input = (o: InstanceRow, className = "") => (
    <input autoFocus className={cx(INPUT, "h-6", className)}
           aria-label={`name for object ${o.idx}`}
           placeholder={`detected: ${o.label}; empty takes the name back`}
           value={editing?.text ?? ""}
           onChange={(e) => setEditing({ idx: o.idx, text: e.target.value })}
           onKeyDown={(e) => {
             if (e.key === "Enter") commit();
             if (e.key === "Escape") setEditing(null);
           }}
           onBlur={() => setEditing(null)} />
  );
  const pencil = (o: InstanceRow) => (
    <button type="button" className="text-t4 hover:text-t1 px-1"
            aria-label={`rename object ${o.idx}`}
            title="name this object yourself (the detected label is kept beside it)"
            disabled={busy}
            onClick={() => setEditing({ idx: o.idx, text: o.operator_label ?? "" })}>
      ✎</button>
  );
  const takeButton = (o: InstanceRow, className = "") => {
    const p = proposalOf(o);
    return p ? (
      <button type="button"
              className={cx("text-primary hover:underline whitespace-nowrap", className)}
              aria-label={`use the verifier's name for object ${o.idx}: ${p}`}
              title={`the verifier proposed "${p}"; one click makes it your name for this object`}
              disabled={busy} onClick={() => take(o)}>
        use “{p}”</button>
    ) : null;
  };
  return { isEditing, input, pencil, take, takeButton, busy };
}

// The mark for verification, the same one control in the same two
// places: while any object is marked, Run verification judges the marked
// ones only; the rest keep their previous verdicts on this export, or
// ship not judged. PUT /instances/<i>/verify_mark; the Objects panel says
// how many are marked and offers to clear them.
export function useVerifyMark(onSaved?: () => void) {
  const { act, busy } = useSceneAct();
  const toggle = async (o: InstanceRow) => {
    const r = await act(`/instances/${o.idx}/verify_mark`,
                        { marked: !o.verify_requested }, { method: "put" });
    if (r) onSaved?.();
  };
  const button = (o: InstanceRow, className = "") => (
    <button type="button"
            className={cx("whitespace-nowrap px-1 rounded-sm border",
                          o.verify_requested
                            ? "border-primary text-primary"
                            : "border-line2 text-t3 hover:text-t1 hover:border-t3",
                          className)}
            aria-label={`${o.verify_requested ? "unmark" : "mark"} object ${o.idx} for verification`}
            title={o.verify_requested
                     ? "marked for verification; click to take the mark back"
                     : "mark this object for verification: while any object is "
                       + "marked, Run verification judges the marked ones only"}
            disabled={busy} onClick={() => toggle(o)}>
      {o.verify_requested ? "✓ to verify" : "verify this"}</button>
  );
  return { button, toggle, busy };
}
