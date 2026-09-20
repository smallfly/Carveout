// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Stage strip (bottom of canvas): idle = last journal event; running =
// job + progress hint (parsed from the pipeline's own "k/N" log lines —
// the stages already print per-item progress) + latest line. Click
// expands the log drawer.

import { useEffect, useRef, useState } from "react";
import { useJobProgress, useRun, useSceneAct } from "../store";
import { Badge, ProgressBar, cx } from "../ui";
import { describeJob, jobLabel } from "../labels";

/** `open` is the Workbench's: the header's job chip opens the same drawer. */
export default function StageStrip({ open, onOpen }:
    { open: boolean; onOpen: (open: boolean) => void }) {
  const { state } = useRun();
  const setOpen = onOpen;
  const [follow, setFollow] = useState(true);
  const [cancelling, setCancelling] = useState(false);
  const tail = useRef<HTMLDivElement>(null);

  const running = state.stage?.state === "running";
  // reset the STOP affordance once the job actually leaves running (the
  // cancelled/done/failed stage event lands)
  useEffect(() => { if (!running) setCancelling(false); }, [running]);
  const { act } = useSceneAct();
  const stop = () => {
    setCancelling(true);
    // a failed cancel surfaces its refusal (usually "no job running",
    // but a lock refusal must not vanish) and resets the affordance
    act("/jobs/cancel").then((r) => { if (!r) setCancelling(false); });
  };
  const last = state.log[state.log.length - 1];
  const pct = useJobProgress()?.pct ?? null;

  useEffect(() => {
    if (open && follow && tail.current)
      tail.current.scrollTop = tail.current.scrollHeight;
  }, [state.log, open, follow]);

  return (
    <div className="absolute bottom-0 inset-x-0 z-10">
      {open && (
        <div className="bg-panel/95 border-t border-line h-60 flex
                        flex-col">
          <div className="flex items-center gap-2 px-3 h-8 border-b
                          border-line">
            <span className="text-[0.75rem] font-semibold text-t1">Log</span>
            {state.stage && (
              <Badge tone={state.stage.state === "failed" ? "fail"
                : state.stage.state === "cancelled" ? "warn"
                : state.stage.state === "running" ? "accent" : "pass"}>
                {state.stage.stage} · {state.stage.state}
              </Badge>
            )}
            <div className="flex-1" />
            <button onClick={() => setFollow(!follow)}
                    aria-pressed={follow}
                    className={cx("text-[0.75rem] px-1.5 h-6 rounded-sm",
                                  follow ? "bg-sel text-t1 font-medium"
                                         : "text-t3 hover:text-t1")}>
              Follow</button>
            <button className="text-[0.75rem] text-t3 hover:text-t1 px-1.5 h-6
                               rounded-sm"
                    onClick={() => setOpen(false)}>Collapse ▾</button>
          </div>
          <div ref={tail} className="flex-1 overflow-y-auto px-3 py-1
                                     font-mono text-[0.75rem] leading-4">
            {state.log.map((l, i) => (
              <div key={i}
                   className={cx(l.level === "WARNING" && "text-warn",
                                 l.level === "ERROR" && "text-fail",
                                 l.level === "INFO" && "text-t2")}>
                <span className="text-t4">{l.ts} {l.source} </span>
                {l.line}
              </div>
            ))}
            {state.log.length === 0 && (
              <div className="text-t4">no output yet</div>
            )}
          </div>
        </div>
      )}
      <div className="w-full h-8 bg-panel/90 border-t border-line px-3 flex
                      items-center gap-3 text-[0.78rem]">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-3 flex-1 min-w-0 text-left">
          {running ? (
            <>
              <span className="font-medium text-t1 whitespace-nowrap">
                {jobLabel(state.stage!.stage)}</span>
              <ProgressBar pct={pct} className="w-40" />
              <span className="font-mono text-t3 truncate flex-1">
                {last?.line ?? ""}</span>
            </>
          ) : (
            <span className="text-t3 truncate">
              <span className={cx(
                state.stage?.state === "failed" && "text-fail",
                state.stage?.state === "cancelled" && "text-warn")}>
                {describeJob(state.stage)}</span>
              {last && <span className="font-mono">: {last.line}</span>}
            </span>
          )}
        </button>
        {running && (
          <button
            onClick={stop}
            disabled={cancelling}
            className="shrink-0 text-[0.75rem] px-2 h-6 border border-fail/60
                       text-fail rounded bg-inset
                       hover:bg-fail/10 disabled:opacity-50">
            {cancelling ? "Stopping…" : "Stop"}
          </button>
        )}
        <button onClick={() => setOpen(!open)}
                aria-label={open ? "collapse the log" : "expand the log"}
                className="shrink-0 text-t3 hover:text-t1 px-1">
          {open ? "▾" : "▴"}</button>
      </div>
    </div>
  );
}
