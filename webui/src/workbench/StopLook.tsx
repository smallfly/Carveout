// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// STOP-AND-LOOK takeover — the ONLY modal takeover in the app. A
// detection was dropped for a mask far larger than its class's usual
// size; the run halts server-side until the operator types the class
// named in the row (proof the row was read) and resumes it.

import { useState } from "react";
import { runUrl } from "../api";
import { useRun, useSceneAct } from "../store";
import { Badge, Btn, INPUT, cx } from "../ui";

const word = (v: string | null | undefined) =>
  (v ?? "").split(/\s+/).filter(Boolean).join(" ").toLowerCase();

export default function StopLookTakeover() {
  const { state, dispatch } = useRun();
  const [typed, setTyped] = useState("");
  const { act, busy } = useSceneAct();
  const sl = state.stopLook!;
  const k = sl.kills[0] ?? {};
  const n = sl.kills.length;
  const cls: string = k.concept ?? "";
  const expected = cls || "ack";
  const matches = word(typed) === word(expected);
  const wrong = typed.trim().length > 0 && !matches;

  const ack = async () => {
    // the id names the block this ack answers; the server refuses one
    // that is no longer waiting, so a replayed ghost cannot be re-acked —
    // and that refusal must SURFACE (act routes it)
    const r = await act("/acks", { kind: sl.kind, id: sl.id, typed });
    if (r) dispatch({ type: "clear_stoplook" });
  };
  const pct = (v: number | undefined) =>
    typeof v === "number" ? `${(100 * v).toFixed(2)}% of the frame` : "";

  return (
    <div className="fixed inset-0 z-50 bg-app/85 flex items-center
                    justify-center p-4">
      {/* data-kbd-modal: a safety takeover scopes keyboard chips to itself
        * (it carries none) — a stray R must not start a render behind it */}
      <div className="w-[720px] max-w-full max-h-[85vh] overflow-y-auto
                      bg-panel border-2 border-fail rounded-md p-5"
           data-kbd-modal role="alertdialog" aria-modal="true"
           aria-label="Stop and look: a detection was dropped">
        <div className="flex items-center gap-3 mb-3">
          <Badge tone="fail">Stop and look</Badge>
          <span className="text-[0.9375rem] font-semibold text-fail">
            {n > 1 ? `${n} detections were dropped`
                   : "One detection was dropped"}: look before continuing
          </span>
        </div>
        <p className="text-[0.8125rem] text-t2 leading-5 mb-2">
          {n > 1
            ? <>{n} detections were dropped for being far larger than their
                class's usual size; the biggest is a <b>{cls}</b> at{" "}
                <b>{k.ratio}× the usual size</b>.</>
            : <>Carveout found a <b>{cls}</b> mask far larger than every
                other {cls} it has found in this scene, <b>{k.ratio}× the
                usual size</b>, and threw that one detection away.</>}
        </p>
        <p className="text-[0.8125rem] text-t2 leading-5 mb-2">
          Look at the frame below. If the outline covers a whole wall, floor
          or room that a small-object class has claimed, the guard did its
          job and you can carry on. If a {cls} in this scene really is that
          big, the size limit for this class is wrong and nothing this run
          produces should be trusted: stop here.
        </p>
        <p className="ui-help mb-3">
          The dropped detection is gone from the output either way;
          continuing only lets the rest of the run finish.
        </p>
        <table className="w-full ui-mono mb-3 text-[0.75rem]">
          <thead>
            <tr className="text-t3 border-b border-line">
              {["Class", "Found by", "View", "Mask size", "Usual for this class",
                "Times larger", "Score"].map((h) => (
                <th key={h} className="text-left font-normal py-1 pr-2">{h}</th>))}
            </tr>
          </thead>
          <tbody>
            {sl.kills.map((r: any, i: number) => (
              <tr key={i} className="border-b border-line">
                <td className="py-1 pr-2 text-t1">{r.concept}</td>
                <td className="py-1 pr-2 text-t2">
                  {r.channel === "exemplar" ? "example image" : "text prompt"}</td>
                <td className="py-1 pr-2 text-t2">frame {r.frame}</td>
                <td className="py-1 pr-2 text-t2">{pct(r.area_frac)}</td>
                <td className="py-1 pr-2 text-t2">{pct(r.median_area)}</td>
                <td className="py-1 pr-2 text-t2">{r.ratio}×</td>
                <td className="py-1 text-t2">{r.score}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex gap-2 flex-wrap mb-1">
          {sl.overlays.map((o) => (
            <img key={o}
                 src={runUrl(state.scene, `/artifacts/stage2/overlays/${o}`)}
                 className="h-44 border border-line rounded"
                 alt={`the view with the dropped ${cls} mask outlined`} />
          ))}
        </div>
        <p className="ui-help mb-4">The dropped mask is outlined in this view.</p>
        <div className="flex items-center gap-2 flex-wrap">
          <label className="text-[0.8125rem] text-t1" htmlFor="stoplook-ack">
            Type the class name shown above to continue:</label>
          <input id="stoplook-ack"
            className={cx(INPUT, "w-56")}
            aria-label="type the class name shown in the row above"
            placeholder={expected}
            value={typed}
            onChange={(e) => setTyped(e.target.value)} />
          <Btn variant="destructive" disabled={!matches || busy} onClick={ack}>
            I looked, resume the run
          </Btn>
        </div>
        {wrong && (
          <p className="text-[0.75rem] text-warn mt-1 mb-0" role="status">
            that is not the class in the row; type “{expected}”.</p>
        )}
        <p className="ui-help mt-2 mb-0">
          The run is halted on the server until you answer; it will wait
          as long as you need.
        </p>
      </div>
    </div>
  );
}
