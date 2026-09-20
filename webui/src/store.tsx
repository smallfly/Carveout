// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Per-scene run store: journal + event stream state. The SERVER owns the
// run; this store is a spectator fed by SSE (re-attach is the server's
// replay, not client persistence).

import React, { createContext, useContext, useEffect, useReducer,
                useRef } from "react";
import { api, openEvents, runUrl } from "./api";
import type { CalibrationResp, Gate, Journal, WsEvent } from "./types";

export interface LogLine {
  ts: string; source: string; level: string; line: string;
}

export interface RunState {
  scene: string;
  journal: Journal | null;
  connected: boolean;
  log: LogLine[];
  stage: { stage: string; state: string; started?: string } | null;
  refusal: { message: string; gate: string | null;
             remedy: string | null } | null;
  stopLook: { kind: string; id: string; kills: Record<string, any>[];
              overlays: string[] } | null;
  dataEpoch: number;   // bumped on journal events -> panels refetch
}

type Action =
  | { type: "event"; ev: WsEvent }
  | { type: "connected"; on: boolean }
  | { type: "journal"; journal: Journal }
  | { type: "clear_refusal" }
  | { type: "clear_stoplook" };

const MAX_LOG = 4000;

// A halted job's open blocks ride the journal snapshot, so a reload can
// restore the stop-look dialog even after the SSE replay buffer evicted
// the original event. SET-only: clearing stays with the ack/stage events,
// so a snapshot racing a fresh block never kills a live dialog.
function withBlocks(s: RunState, j: Journal): RunState {
  const blocks = j.blocks ?? [];
  const sl = blocks.find((b) => b.kind === "area_factor_kill");
  return {
    ...s, journal: j, dataEpoch: s.dataEpoch + 1,
    stopLook: sl && s.stopLook?.id !== sl.id ? (sl as any) : s.stopLook,
  };
}

function reduce(s: RunState, a: Action): RunState {
  switch (a.type) {
    case "connected":
      return { ...s, connected: a.on };
    case "journal":
      return withBlocks(s, a.journal);
    case "clear_refusal":
      return { ...s, refusal: null };
    case "clear_stoplook":
      return { ...s, stopLook: null };
    case "event": {
      const ev = a.ev;
      switch (ev.type) {
        case "journal":
          return withBlocks(s, ev.data);
        case "log": {
          const log = s.log.length >= MAX_LOG
            ? [...s.log.slice(-MAX_LOG + 1), ev.data]
            : [...s.log, ev.data];
          return { ...s, log };
        }
        // A new attempt supersedes the previous failure banner, and a job
        // that reached an end has nothing blocking: these clear on the
        // events that follow them. (A fresh page no longer replays the
        // whole buffer — the server sends the latest stage, the refusal
        // only while it is current, and a log tail — but a reconnect
        // still replays its missed events, so the rules stay.)
        case "stage": {
          const st = ev.data as RunState["stage"];
          return { ...s, stage: st,
                   // a new attempt supersedes the previous failure banner
                   refusal: st?.state === "running" ? null : s.refusal,
                   // nothing can be blocking once the job reached an end
                   stopLook: st && st.state !== "running" ? null : s.stopLook };
        }
        case "refusal":
          return { ...s, refusal: ev.data };
        case "stop_look":
          return { ...s, stopLook: ev.data };
        case "ack":
          return { ...s,
                   stopLook: s.stopLook && s.stopLook.id === ev.data.id
                     ? null : s.stopLook };
        default:
          return s;
      }
    }
  }
}

const RunCtx = createContext<{ state: RunState;
                               dispatch: React.Dispatch<Action> } | null>(
  null);

export function RunProvider({ scene, children }:
    { scene: string; children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reduce, {
    scene, journal: null, connected: false, log: [], stage: null,
    refusal: null, stopLook: null, dataEpoch: 0,
  } satisfies RunState);
  const closer = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.get<Journal>(runUrl(scene, "/journal"))
      .then((j) => dispatch({ type: "journal", journal: j }))
      .catch(() => undefined);
    closer.current = openEvents(
      scene,
      (ev) => dispatch({ type: "event", ev }),
      (on) => dispatch({ type: "connected", on }));
    return () => closer.current?.();
  }, [scene]);

  return (
    <RunCtx.Provider value={{ state, dispatch }}>{children}</RunCtx.Provider>
  );
}

/** The ONE way a caught API error becomes a refusal event. Seven components
 * carried their own copy of this dispatch — the product's core contract
 * (refusals surfaced verbatim, routed to their gate) deserves a single
 * implementation that cannot drift. */
export function useRefusal() {
  const { dispatch } = useRun();
  return React.useCallback(
    (e: any) => dispatch({ type: "event", ev: {
      type: "refusal", data: { message: e.message, gate: e.gate ?? null,
                               remedy: e.remedy ?? null } } }),
    [dispatch]);
}

export function useRun() {
  const ctx = useContext(RunCtx);
  if (!ctx) throw new Error("useRun outside RunProvider");
  return ctx;
}

/** The scene-action hook: every MUTATING call to the run's API goes
 * through it. act() sends the request bound to this run, routes any
 * refusal through the ONE dispatch, and resolves null instead of
 * throwing — so a caller chains success-only work off the result and can
 * never forget the catch (the stop-and-look takeover shipped exactly
 * that bug: a try/finally with no catch silently swallowed a refused
 * ack). `busy` is true while any act from this hook instance is in
 * flight; approveGate/reopenGate/startStage are the three verbs behind
 * almost every panel button. */
export function useSceneAct() {
  const { state } = useRun();
  const refuse = useRefusal();
  const [busy, setBusy] = React.useState(false);
  const inflight = React.useRef(0);
  const act = React.useCallback(async function act<T = unknown>(
      path: string, body?: unknown,
      opts?: { method?: "post" | "put" | "patch" | "del" }):
      Promise<T | null> {
    inflight.current += 1;
    setBusy(true);
    try {
      return await api[opts?.method ?? "post"]<T>(
        runUrl(state.scene, path), body);
    } catch (e: any) {
      refuse(e);
      return null;
    } finally {
      inflight.current -= 1;
      if (inflight.current === 0) setBusy(false);
    }
  }, [state.scene, refuse]);
  const approveGate = React.useCallback(
    (g: Gate) => act(`/gates/${g}/approve`), [act]);
  const reopenGate = React.useCallback(
    (g: Gate) => act(`/gates/${g}/reopen`), [act]);
  const startStage = React.useCallback(
    (s: string, body?: unknown) => act(`/stages/${s}/start`, body), [act]);
  return { act, approveGate, reopenGate, startStage, busy };
}

/** True while a job is live on the server. The journal snapshot only
 * re-publishes at job BOUNDARIES, so journal.running alone leaves every
 * panel button enabled for the whole duration of a live GPU stage (found
 * on the first real browser-driven probe); stage events
 * stream at job start, so they carry the live signal. */
export function useJobRunning() {
  const { state } = useRun();
  return state.journal?.running?.state === "running"
    || state.stage?.state === "running";
}

const PROGRESS_RE = /(\d+)\s*\/\s*(\d+)/;

/** The live job, for anything that shows it: its kind (the journal's
 * `running.kind`, else the stage event's name), when it started, and a
 * progress fraction parsed from the pipeline's own "k/N" log lines (the
 * stages print per-item progress; nothing publishes a fraction). ONE
 * reading shared by the stage strip and the header chip, so the two can
 * never disagree. null when nothing runs. */
export function useJobProgress():
    { kind: string; started: string | null; pct: number | null;
      k: number | null; n: number | null } | null {
  const { state } = useRun();
  const running = useJobRunning();
  return React.useMemo(() => {
    if (!running) return null;
    // The live stage event names the job: the journal snapshot is
    // published at job END, so its `running` is the LAST job until this
    // one finishes — the chip read "Re-probing exemplars" through a whole
    // pipeline run. `started` below already prefers the live event.
    const kind = (state.stage?.state === "running" ? state.stage.stage : null)
      ?? state.journal?.running?.kind ?? "";
    const started = state.stage?.started ?? state.journal?.running?.started
      ?? null;
    for (let i = state.log.length - 1;
         i >= Math.max(0, state.log.length - 20); i--) {
      const m = state.log[i].line.match(PROGRESS_RE);
      if (m && +m[2] > 0 && +m[1] <= +m[2])
        return { kind, started, pct: +m[1] / +m[2], k: +m[1], n: +m[2] };
    }
    return { kind, started, pct: null, k: null, n: null };
  }, [running, state.log, state.journal, state.stage]);
}

/** The installed vision models, read once per mount (a server fact). */
export function useVlmModels() {
  const [models, setModels] = React.useState<import("./types").VlmModel[]>([]);
  useEffect(() => {
    let dead = false;
    api.get<{ vlm_models: import("./types").VlmModel[] }>("/api/settings")
      .then((r) => { if (!dead) setModels(r.vlm_models ?? []); })
      .catch(() => { /* the row stays absent */ });
    return () => { dead = true; };
  }, []);
  return models;
}

/** Per-scene calibration: the effective knob values + a writer that
 * PUTs a nested partial profile. The write runs server-side as a job and
 * re-opens the affected gate; the journal event it publishes bumps dataEpoch,
 * so the calib fetch (and every panel) refreshes on its own — no manual
 * reload (which would race the async profile write). */
/** The ONE derivation of the scene's scale factor in the browser: metres
 *  per scene unit when the profile records one, null when not known. The
 *  canvas turns null into a nominal display scale; the panels show metres
 *  only when this is a number. */
export function sceneFactor(calib: CalibrationResp | null | undefined):
    number | null {
  const v = calib?.scene.scale_m_per_unit;
  return typeof v === "number" && v > 0 ? v : null;
}

export function useCalibration() {
  const calib = useRunData<CalibrationResp>("/calibration");
  const { act } = useSceneAct();
  const write = (update: unknown) =>
    act("/calibration", update, { method: "put" });
  return { calib: calib.data, write };
}

/** Fetch helper bound to the run's dataEpoch: refetches when the journal
 * moves (a gate approval/demotion invalidates panel data). */
export function useRunData<T>(path: string, deps: unknown[] = []):
    { data: T | null; error: string | null; reload: () => void } {
  const { state } = useRun();
  const [data, setData] = React.useState<T | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [tick, setTick] = React.useState(0);
  useEffect(() => {
    let live = true;
    api.get<T>(runUrl(state.scene, path))
      .then((d) => live && (setData(d), setError(null)))
      // Clear the data too: a panel that keeps rendering the LAST good
      // payload beside an error is claiming something is there when it is
      // not. Resetting the render step deletes stage1, so /cameras 404s —
      // and the 3D view and contact sheet went on showing the deleted
      // frames until the browser was refreshed.
      .catch((e) => live && (setData(null), setError(e.message)));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.scene, state.dataEpoch, tick, path, ...deps]);
  return { data, error, reload: () => setTick((t) => t + 1) };
}
