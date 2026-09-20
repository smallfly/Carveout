// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The Workbench (design 01/02): ONE persistent full-bleed 3D canvas; gate
// chrome floats over it as HUD panels — never displacing the canvas. The
// journal decides which gate is active; the rail makes it visible.

import * as THREE from "three";
import { useEffect, useMemo, useRef, useState } from "react";
import { isTypingTarget } from "@viewer/stage.js";
import { api, runUrl } from "../api";
import { demotedBy, downstream, gateState, overlayStale, sceneReadOnly,
         staleGates } from "../journal";
import { sceneFactor, useCalibration, useJobRunning, useRefusal, useRun, useRunData,
         useSceneAct } from "../store";
import { GATES, type AlignmentBlock, type CamerasResp, type ExemplarsResp,
         type FrameEntry, type Gate, type InstanceRow, type VolumeBox,
         type VolumeResp }
  from "../types";
import { Badge, Btn, Confirm, FooterSlot, Segmented, Slider, cx, JobNote } from "../ui";
import { GATE_LABEL, describeAwaiting, describeJob } from "../labels";
import { sameAlignment, alignmentMatrix } from "./alignment";
import { frameFromUpAxis } from "./overlays";
import SplatCanvas, { type CanvasHandle } from "./SplatCanvas";
import GateRail from "./GateRail";
import StageStrip from "./StageStrip";
import { verdictInfo } from "@viewer/verdicts.js";
import { useVerifyMark } from "./relabel";
import JobChip from "./JobChip";
import SceneLoadChip from "./SceneLoadChip";
import type { SceneLoad } from "./SplatCanvas";
import ReportDrawer from "./ReportDrawer";
import { proposalOf, shownName, useRelabel } from "./relabel";
import StopLookTakeover from "./StopLook";
import SettingsPanel from "./panels/SettingsPanel";
import VolumePanel from "./panels/VolumePanel";
import RenderPanel from "./panels/RenderPanel";
import VocabPanel from "./panels/VocabPanel";
import ExemplarPanel from "./panels/ExemplarPanel";
import PipelinePanel from "./panels/PipelinePanel";

export default function Workbench({ onExit }: { onExit: () => void }) {
  const { state, dispatch } = useRun();
  const refuse = useRefusal();
  const { act, reopenGate } = useSceneAct();
  const j = state.journal;
  const [railGate, setRailGate] = useState<Gate | "report" | "settings"
                                           | null>(null);
  const [reopenAsk, setReopenAsk] = useState<Gate | null>(null);
  const [logOpen, setLogOpen] = useState(false);   // the stage strip's drawer
  // the scene file on its way to the canvas: the header says so while it
  // is (a splat file can be gigabytes), and says if it never arrives
  const [sceneLoad, setSceneLoad] = useState<SceneLoad | null>(null);
  // a live job: the acts that would be refused server-side are dead here
  // too, and say so (JobNote) rather than answering with a red banner
  const running = useJobRunning();
  // `grid` is the reference grid, lying on the floor (the proposal's, or
  // the estimate before one); the detected floor plane rides the volume
  // layer, being volume-gate state.
  const [layers, setLayers] = useState({ volume: true, cameras: true,
                                         instances: true, crops: true,
                                         grid: true });
  const [frustumScale, setFrustumScale] = useState(() =>
    parseFloat(localStorage.getItem("carveout_frustum_scale") || "1"));
  const [selFrame, setSelFrame] = useState<number | null>(null);
  // the selected detected object: the label engine reports a
  // click (or the report drawer's fly-to) and -1 when it is put down
  const [selInst, setSelInst] = useState<number | null>(null);
  // An adjusted camera, held locally until the re-render catches up. Same
  // shape as volDraft: the edit is visible immediately, and the authoritative
  // copy arrives with the next render. Without it the frustum and the handles
  // both jumped back to the pose cameras.json still described, which reads as
  // "the move did not take".
  const [poseDraft, setPoseDraft] =
    useState<Record<number, number[][]>>({});
  const [selBox, setSelBox] = useState(-1);
  const [volDraft, setVolDraft] = useState<VolumeBox[] | null>(null);
  // the volume gate's preview (outside the box as points): a display
  // choice, never written anywhere. It stays on through the other stages
  // (the view does not change at a stage boundary) and hands over to the
  // Viewer panel's own focus volume when that panel opens (see the rail).
  const [volPreview, setVolPreview] = useState(false);
  // The ruler (the scale act): armed from the volume panel,
  // the next two clicks on the splat are its points; a third starts
  // over; Escape cancels. The draft lives here because the panel unmounts
  // with the rail while the canvas does not.
  const [rulerArmed, setRulerArmed] = useState(false);
  const [rulerDraft, setRulerDraft] = useState<number[][] | null>(null);
  const armRuler = (on: boolean) => {
    setRulerArmed(on);
    if (!on) setRulerDraft(null);
    if (on) { setSelBox(-1); setFocusSel(false); }
  };
  const rulerPick = (pt: number[]) =>
    setRulerDraft((d) => (!d || d.length >= 2) ? [pt] : [...d, pt]);
  // The level act: armed from the volume panel, the next clicks on
  // the splat are its points (three on the floor, then up to two along a
  // wall); Escape cancels. Held here like the ruler's draft — the canvas
  // outlives the panel. `levelPreview` is a proposed alignment the canvas
  // draws the grid and the floor plane in, for this panel visit only.
  // Two acts share the draft and the canvas mode: the level act (three
  // floor points, then two along an edge) and the square act (the
  // two edge points alone, the tilt stays).
  const [levelAct, setLevelAct] = useState<"level" | "square" | null>(null);
  const levelArmed = levelAct !== null;
  const [levelDraft, setLevelDraft] = useState<number[][]>([]);
  const [levelPreview, setLevelPreview] = useState<AlignmentBlock | null>(null);
  const armAct = (act: "level" | "square" | null) => {
    setLevelAct(act);
    setLevelDraft([]);
    if (act) { setRulerArmed(false); setRulerDraft(null); setSelBox(-1); setFocusSel(false); }
  };
  const armLevel = (on: boolean) => armAct(on ? "level" : null);
  const armSquare = (on: boolean) => armAct(on ? "square" : null);
  const levelPick = (pt: number[]) =>
    setLevelDraft((d) => d.length >= (levelAct === "square" ? 2 : 5) ? d : [...d, pt]);
  // One bad point (a click that landed on a floater) costs that point,
  // not the act: Undo last on the panel, Backspace on the canvas.
  const levelUndo = () => setLevelDraft((d) => d.slice(0, -1));
  // The floor act: one click on the floor sets scene.floor — the clicked
  // point (file frame) carried into the levelled frame, its height along
  // the up axis, signed; the same value the number field takes.
  const [floorArmed, setFloorArmed] = useState(false);
  const armFloor = (on: boolean) => {
    setFloorArmed(on);
    if (on) { setRulerArmed(false); setRulerDraft(null); setLevelAct(null);
              setLevelDraft([]); setSelBox(-1); setFocusSel(false); }
  };
  // Bumped by the focus manipulator so the settings panel's sliders follow the
  // hand. The panel's own poll only watches object IDENTITY, which never
  // changes while a drag mutates values in place.
  const [focusEpoch, setFocusEpoch] = useState(0);
  const bumpFocus = () => setFocusEpoch((n) => n + 1);
  // The two volumes are selected the same way and never both: a manipulator
  // on each at once would put two sets of handles on one canvas. Enforced
  // here, in the pair of setters, rather than trusted to the pick order.
  const [focusSel, setFocusSel] = useState(false);
  const selectBox = (i: number) => {
    setSelBox(i);
    if (i !== -1) setFocusSel(false);
  };
  const selectFocus = (on: boolean) => {
    setFocusSel(on);
    if (on) setSelBox(-1);
  };
  // chrome visibility: the gate panel collapses; "3D only"
  // hides ALL chrome except safety takeovers. Tab toggles (old viewer's
  // sidebar key, kept for parity).
  const [panelHidden, setPanelHidden] = useState(false);
  const [zen, setZen] = useState(false);
  const canvas = useRef<CanvasHandle>(null);
  // The inspector's fixed footer: panels portal their panel-level actions
  // into it (ui.tsx PanelFooter) so the buttons never sit under the fold.
  const [footerEl, setFooterEl] = useState<HTMLElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Escape" && rulerArmed) {
        armRuler(false);   // the ruler: Escape cancels the measurement
        return;
      }
      if (e.code === "Escape" && floorArmed) {
        armFloor(false);   // the floor act: Escape cancels it
        return;
      }
      if (e.code === "Escape" && levelArmed) {
        armAct(null);   // the level or square act: Escape cancels it
        return;
      }
      if (e.code === "Backspace" && levelArmed
          && !isTypingTarget(e.target as HTMLElement)) {
        e.preventDefault();   // the level act: Backspace drops the last point
        levelUndo();
        return;
      }
      if (e.code !== "Tab") return;
      if (isTypingTarget(e.target as HTMLElement)) return;
      e.preventDefault();
      setZen((z) => !z);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rulerArmed, levelArmed, floorArmed]);

  const vol = useRunData<VolumeResp>("/volume");
  const cams = useRunData<CamerasResp>("/cameras");
  const exemplars = useRunData<ExemplarsResp>("/exemplars");

  const active: Gate | "report" | "settings" | null =
    railGate ?? j?.active_gate ?? (j ? "report" : null);
  // An approval moves the journal's active gate on; a rail node the
  // operator had clicked used to pin the panel to the gate just approved
  // ("Continue does nothing"). When the active gate changes
  // and the pinned node is now approved, let the journal lead again.
  // Clicking an approved node to inspect it still pins it (no change of
  // active gate happens then).
  const prevActiveGate = useRef(j?.active_gate);
  useEffect(() => {
    if (j?.active_gate === prevActiveGate.current) return;
    prevActiveGate.current = j?.active_gate;
    if (railGate && railGate !== "report" && railGate !== "settings"
        && gateState(j, railGate) === "approved") setRailGate(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [j?.active_gate]);
  const readOnly = sceneReadOnly(j);
  // An approved-FRESH gate opens for INSPECTION: its data renders
  // read-only and redoing it is an explicit act (the reopen button below)
  // — viewing must never demote (operator feedback).
  const activeApproved = !!(active
    && active !== "report" && active !== "settings"
    && gateState(j, active) === "approved");
  const gateReadOnly = readOnly || activeApproved;

  const boxes = volDraft ?? vol.data?.volume?.boxes ?? null;
  // The frame's floor was detected under the alignment the volume was
  // made with: once the operator levels the scene it is stale until
  // the next proposal, so the start pose reads its own estimate instead.
  const { calib, write: writeCalib } = useCalibration();
  const frameCurrent = sameAlignment(vol.data?.frame?.alignment ?? null,
                                     calib?.scene.alignment ?? null, 0.01);
  const volumeFrame = useMemo(() => {
    const f = vol.data?.frame;
    if (!f) return null;
    const up = "xyz".indexOf(f.up_axis);
    const ground = [0, 1, 2].filter((a) => a !== up);
    return { up, upSign: f.up_sign, a0: ground[0], a1: ground[1] };
  }, [vol.data?.frame]);
  // The origin grid needs an up axis before any proposal has measured one:
  // the profile's convention, from the calibration the volume panel edits.
  // The scene's factor, derived ONCE (store.sceneFactor): the canvas
  // makes its display scale from it (or from the extents while it is not
  // known), the volume panel its metre readouts.
  const factor = sceneFactor(calib);
  const gridFrame = useMemo(
    () => volumeFrame ?? frameFromUpAxis(calib?.scene.up_axis),
    [volumeFrame, calib?.scene.up_axis]);
  const floorPick = (pt: number[]) => {
    const v = new THREE.Vector3(pt[0], pt[1], pt[2])
      .applyMatrix4(alignmentMatrix(calib?.scene.alignment ?? null, gridFrame.up));
    const signed = v.getComponent(gridFrame.up) * gridFrame.upSign;
    writeCalib({ scene: { floor: Number(signed.toPrecision(6)) } });
    setFloorArmed(false);
  };

  // Frames as the 3D view should draw them: the render's own poses, with any
  // un-rendered adjustment substituted in.
  const draftedFrames = useMemo(() => {
    const fs = cams.data?.frames ?? null;
    if (!fs || !Object.keys(poseDraft).length) return fs;
    return fs.map((f) => poseDraft[f.frame_idx]
      ? { ...f, c2w: poseDraft[f.frame_idx] } : f);
  }, [cams.data, poseDraft]);

  // Drop a draft once the render reproduces it — matching on the pose itself
  // rather than on a render-finished signal, because the journal also moves
  // when the edit demotes the gate, which would clear the draft far too early.
  useEffect(() => {
    const fs = cams.data?.frames;
    if (!fs || !Object.keys(poseDraft).length) return;
    const settled = Object.entries(poseDraft).filter(([idx, c2w]) => {
      const f = fs.find((x) => x.frame_idx === Number(idx));
      return f && (f.c2w as number[][]).flat().every(
        (v, i) => Math.abs(v - c2w.flat()[i]) < 1e-6);
    });
    if (!settled.length) return;
    setPoseDraft((d) => {
      const next = { ...d };
      for (const [idx] of settled) delete next[Number(idx)];
      return next;
    });
  }, [cams.data]);

  // Frames are numbered at render time (manual first, in file order), so
  // a re-render after a delete moves every later frame down one: the same
  // index is another camera. Index-keyed state — the selection, the pose
  // drafts — is dropped when its frame's identity changes.
  const provRef = useRef<Map<number, string>>(new Map());
  useEffect(() => {
    const fs = cams.data?.frames;
    if (!fs) return;
    const prev = provRef.current;
    const next = new Map(fs.map((f) =>
      [f.frame_idx, String(f.provenance ?? "auto")] as [number, string]));
    const changed = (idx: number) =>
      prev.has(idx) && prev.get(idx) !== next.get(idx);
    if (selFrame !== null && changed(selFrame)) setSelFrame(null);
    setPoseDraft((d) => {
      const keep = Object.fromEntries(
        Object.entries(d).filter(([k]) => !changed(Number(k))));
      return Object.keys(keep).length === Object.keys(d).length ? d : keep;
    });
    provRef.current = next;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cams.data]);

  // frames whose manual view was deleted since the render: the
  // server names them with the frame they still occupy
  const deletedFrames = useMemo(() =>
    (cams.data?.manual_pending ?? [])
      .filter((p) => p.why === "deleted" && p.frame_idx != null)
      .map((p) => p.frame_idx as number),
    [cams.data]);
  const framesByIdx = useMemo(() => {
    const m = new Map<number, FrameEntry>();
    cams.data?.frames.forEach((f) => m.set(f.frame_idx, f));
    return m;
  }, [cams.data]);

  const cropMarkers = useMemo(() => {
    if (!exemplars.data || !cams.data) return [];
    const out: { frame: FrameEntry; box_xyxy: number[] }[] = [];
    for (const e of exemplars.data.exemplars)
      for (const c of e.crops) {
        const fr = framesByIdx.get(c.frame_idx);
        if (fr) out.push({ frame: fr, box_xyxy: c.box_xyxy });
      }
    return out;
  }, [exemplars.data, cams.data, framesByIdx]);

  // The draft outlives the PUT: clearing it on the response and THEN
  // reloading left one round-trip in which `boxes` fell back to the copy
  // /volume last returned — the box jumped to the previous state on every
  // slider release and gizmo drag end, then to the new one. The draft
  // is dropped only when the reloaded volume lands.
  const volSaved = useRef(false);
  const saveVolume = async (b: VolumeBox[]) => {
    setVolDraft(b);
    if (await act("/volume", { boxes: b }, { method: "put" })) {
      volSaved.current = true;
      vol.reload();
    } else {
      // refused (the lock lost, a job started, the gate moved): the
      // canvas and the panel go back to the volume the server holds,
      // instead of showing boxes it never accepted until the next save
      setVolDraft(null);
    }
  };
  useEffect(() => {
    if (!volSaved.current) return;
    volSaved.current = false;
    setVolDraft(null);
  }, [vol.data]);
  // The whole-scene act: a volume.json that says so and holds no boxes,
  // hashed by the journal like any drawn volume.
  const useWholeScene = async () => {
    setVolDraft(null);
    setSelBox(-1);
    if (await act("/volume", { scope: "whole_scene" }, { method: "put" }))
      vol.reload();
  };


  // The selected frame's manual-view id, or null when it is an auto view.
  // Auto poses are regenerated by the sampler every render, so there is
  // nothing durable to write them to — the render gate's "fully manual"
  // conversion is what makes a frame adjustable.
  const selManualId = (() => {
    const fr = selFrame === null ? null : framesByIdx.get(selFrame);
    const pv = String(fr?.provenance ?? "");
    return pv.startsWith("manual:") ? pv.split(":", 2)[1] : null;
  })();

  // A camera was moved or turned and the handle released. The gizmo emits
  // ply-space position + the three.js quaternion, which is exactly what
  // manual_views.json stores. `target` is advisory but IS checked at load
  // (a >2 deg mismatch logs a convention warning every render), so it is
  // recomputed down the new view axis at the original distance rather than
  // left pointing where the camera used to look.
  const saveFramePose = async (v: { position: number[];
                                    quaternion: number[];
                                    c2w: number[][] }) => {
    if (!selManualId) return;
    let saved: any = null;
    try {
      saved = (await api.get<{ views: any[] }>(
        runUrl(state.scene, "/views"))).views.find(
          (x) => x.id === selManualId);
    } catch (e: any) {
      refuse(e);
      return;
    }
    const p0 = v.position;
    const q = new THREE.Quaternion(...(v.quaternion as
      [number, number, number, number]));
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
    const dist = saved
      ? new THREE.Vector3(...(saved.position as [number, number, number]))
          .distanceTo(new THREE.Vector3(
            ...(saved.target as [number, number, number]))) || 1
      : 1;
    if (selFrame !== null) setPoseDraft((d) => ({ ...d,
      [selFrame]: v.c2w }));
    const ok = await act(`/views/${selManualId}`, {
      position: p0,
      quaternion: v.quaternion,
      target: [p0[0] + fwd.x * dist, p0[1] + fwd.y * dist,
               p0[2] + fwd.z * dist],
      target_source: "gizmo",
    }, { method: "patch" });
    if (ok) {
      cams.reload();
    } else if (selFrame !== null) {
      // Roll the optimistic draft back: the settle effect only clears
      // drafts that MATCH cams.json, so a refused move otherwise kept the
      // frustum showing a pose the server never accepted.
      setPoseDraft((d) => {
        const { [selFrame]: _gone, ...rest } = d;
        return rest;
      });
    }
  };

  const reopen = async (g: Gate) => {
    setReopenAsk(null);
    if (await reopenGate(g)) setRailGate(g);
  };

  const stale = staleGates(j);

  return (
    <div className="h-full flex flex-col bg-app">
      {/* top bar */}
      {!zen &&
      <header className="h-11 shrink-0 border-b border-line bg-panel px-3
                         flex items-center gap-3">
        <button className="font-semibold text-[0.8125rem] tracking-[0.08em]
                           text-t1 rounded px-1 -ml-1"
                onClick={onExit}>
          CARVEOUT
        </button>
        <span className="text-t4" aria-hidden="true">/</span>
        <span className="font-medium text-t1 text-[0.8125rem] truncate
                         max-w-[28ch]">{state.scene}</span>
        {/* pipeline status, never the viewed panel: the approval count,
            then the running job if there is one, else the gate that is
            waiting on the operator (labels.ts formatters — the stage
            strip uses the same words) */}
        {j && (
          <span className="text-[0.75rem] text-t3 ui-num whitespace-nowrap">
            {GATES.filter((g) => j.gates[g]?.state === "approved").length}
            {" / 5 gates approved · "}
            {j.running?.state === "running"
              ? describeJob({ stage: j.running.kind, state: "running" })
              : describeAwaiting(j)}
          </span>
        )}
        <div className="flex-1" />
        <SceneLoadChip load={sceneLoad} />
        <JobChip onOpenLog={() => setLogOpen(true)} />
        <span className="flex items-center gap-1.5 text-[0.75rem] text-t3"
              title={state.connected ? "connected to the server"
                                     : "re-attaching to the server…"}>
          <span className={cx("w-2 h-2 rounded-full",
                              state.connected ? "bg-pass"
                                              : "bg-warn animate-pulse")} />
          <span className="sr-only">
            {state.connected ? "connected" : "re-attaching"}</span>
        </span>
        {/* the appearance switcher is mounted once, on the Library header:
            a per-browser preference, set once, that follows
            the operator here through localStorage */}
        <Btn variant="ghost" onClick={onExit}>Library</Btn>
      </header>}

      {/* refusal banner: verbatim message + remedy + routing */}
      {!zen && state.refusal && (
        <div role="alert"
             className="shrink-0 border-b border-fail/60 bg-fail/10 px-3 py-2
                        flex items-start gap-3">
          <Badge tone="fail" className="mt-0.5">Refusal</Badge>
          <div className="flex-1 font-mono text-[0.78rem] text-t1
                          whitespace-pre-wrap min-w-0">
            {state.refusal.message}
            {state.refusal.remedy && (
              <div className="text-t2 mt-1 font-sans text-[0.75rem]">
                {state.refusal.remedy}</div>
            )}
          </div>
          {state.refusal.gate && (
            <Btn variant="outline-primary"
                 onClick={() => { setRailGate(state.refusal!.gate as Gate);
                                  dispatch({ type: "clear_refusal" }); }}>
              Go to {GATE_LABEL[state.refusal.gate as Gate]
                     ?? state.refusal.gate}
            </Btn>
          )}
          <Btn variant="ghost" aria-label="dismiss refusal"
               onClick={() => dispatch({ type: "clear_refusal" })}>✕</Btn>
        </div>
      )}

      {/* held-by-other-driver banner */}
      {!zen && readOnly && j?.lock && (
        <div className="shrink-0 border-b border-warn/60 bg-warn/10 px-3 py-2
                        text-[0.8125rem] text-t1 flex items-center gap-3">
          <Badge tone="warn">Locked</Badge>
          this scene is open in another Carveout since {j.lock.since};
          close it there to work on it here. This view is read-only.
        </div>
      )}
      {!zen && j?.lock && !j.lock.mine && !j.lock.alive && (
        <div className="shrink-0 border-b border-warn/60 bg-warn/10 px-3 py-2
                        text-[0.8125rem] flex items-center gap-3">
          <Badge tone="warn">Stale lock</Badge>
          {j.lock.holder} is no longer running.
          <Btn variant="destructive"
               onClick={() => api.post(
                 `/api/scenes/${state.scene}/lock/release`)
                 .then(() => window.location.reload())}>
            Release stale lock…
          </Btn>
        </div>
      )}

      {/* cascade banner (1k) */}
      {!zen && stale.length > 0 && (
        <div className="shrink-0 border-b border-warn/60 bg-warn/10 px-3 py-1.5
                        text-[0.78rem] text-t1 flex items-center gap-2">
          <Badge tone="warn">Stale</Badge>
          <span className="font-medium">{stale.map((g) => GATE_LABEL[g]).join(", ")}
          </span>
          <span className="text-t2">inputs changed since approval;
          re-approve from the earliest stale gate. Overlays shown are from
          the previous approval.</span>
        </div>
      )}

      <div className="flex-1 relative flex min-h-0">
        {!zen && (
          <GateRail journal={j} active={active}
                    onSelect={(g) => {
                      // re-clicking the active node toggles the panel
                      if (g === active && !panelHidden) {
                        setPanelHidden(true);
                      } else {
                        // The Viewer panel owns the focus volume: the
                        // volume gate's preview retires here and its
                        // effect continues under the panel's own switch
                        // and adjustments, so the view does not blink.
                        if (g === "settings" && volPreview) {
                          setVolPreview(false);
                          canvas.current?.apply("focus.enabled", true);
                        }
                        setRailGate(g);
                        setPanelHidden(false);
                      }
                    }} />
        )}

        <div className="flex-1 relative min-w-0">
          <SplatCanvas
            ref={canvas}
            scene={state.scene}
            onSceneLoad={setSceneLoad}
            epoch={state.dataEpoch}
            layers={layers}
            volumeBoxes={layers.volume ? boxes : null}
            focusBoxes={vol.data?.volume?.boxes ?? null}
            focusPreview={volPreview && active !== "settings" && !zen}
            focusDraftBoxes={boxes}
            volumeFrame={volumeFrame}
            alignment={calib?.scene.alignment ?? null}
            upAxis={gridFrame.up}
            previewAlignment={active === "volume" && !zen ? levelPreview : null}
            levelArmed={levelArmed && active === "volume"}
            levelMode={levelAct ?? "level"}
            levelPoints={levelDraft}
            onLevelPick={levelPick}
            floorArmed={floorArmed && active === "volume"}
            onFloorPick={floorPick}
            pathMode={calib?.render.path_mode ?? null}
            eyeHeight={calib?.render.eye_height ?? null}
            floorBandFrac={calib?.scene.floor_band_frac ?? null}
            histBinM={calib?.scene.hist_bin_m ?? null}
            gridFrame={gridFrame}
            volumeEditable={active === "volume" && !gateReadOnly && !running && !zen}
            volumeSelected={selBox}
            onVolumeChange={saveVolume}
            onVolumeDraft={setVolDraft}
            onFocusChange={bumpFocus}
            focusEpoch={focusEpoch}
            zen={zen}
            focusEditable={active === "settings" && !zen}
            focusSelected={focusSel}
            onFocusSelect={selectFocus}
            onVolumeSelect={selectBox}
            floorSigned={calib?.scene.floor
                         ?? (frameCurrent ? vol.data?.frame?.floor ?? null : null)}
            rulerArmed={rulerArmed && active === "volume"}
            rulerPoints={rulerDraft
                         ?? (active === "volume"
                             ? calib?.scene.measured?.points ?? null : null)}
            onRulerPick={rulerPick}
            frames={draftedFrames}
            deletedFrames={deletedFrames}
            intrinsics={cams.data?.intrinsics ?? null}
            frustumScale={frustumScale}
            factor={factor}
            extent={vol.data?.scale?.extent_units ?? null}
            selectedFrame={selFrame}
            onSelectFrame={setSelFrame}
            onInstanceSelect={(idx) => setSelInst(idx >= 0 ? idx : null)}
            cameraEditable={active === "render" && !gateReadOnly && !zen
                            && selManualId !== null}
            onFramePose={saveFramePose}
            crops={cropMarkers}
            stale={overlayStale(j)}
          />

          {/* 3D-only exit affordance (Tab also toggles) */}
          {zen && (
            <button
              className="absolute bottom-2 right-2 z-10 text-[0.75rem]
                         px-2 h-7 border border-line2 rounded bg-panel/90
                         text-t2 hover:text-t1 hover:bg-hover"
              onClick={() => setZen(false)}>
              Exit 3D only <span className="text-t4">(Tab)</span>
            </button>
          )}

          {/* collapsed-panel reopen tab */}
          {!zen && panelHidden && (
            <button
              className="absolute top-1/2 right-0 z-10 -translate-y-1/2
                         text-[0.75rem] px-1.5 py-3 border
                         border-line2 border-r-0 rounded-l bg-panel/90
                         text-t2 hover:text-t1 hover:bg-hover"
              aria-label="show the gate panel"
              onClick={() => setPanelHidden(false)}>
              ⟨
            </button>
          )}

          {/* viewport toolbar: layer toggles, frustum size, 3D only — loose
              chips floating over the scene (no enclosing bar); pressed =
              filled neutral + weight, never colour alone */}
          {!zen &&
          <div role="toolbar" aria-label="Viewport"
               className="absolute top-2 left-2 z-10 flex items-center gap-1.5">
            {(Object.keys(layers) as (keyof typeof layers)[]).map((k) => (
              <button key={k} type="button"
                      aria-pressed={layers[k]}
                      onClick={() => setLayers({ ...layers,
                                                 [k]: !layers[k] })}
                      className={cx(
                        "ui-cap text-[0.75rem] px-2 h-7 rounded leading-none",
                        "whitespace-nowrap border",
                        layers[k]
                          ? "bg-sel/95 border-line2 text-t1 font-medium"
                          : "bg-panel/85 border-line text-t3 hover:text-t1 "
                            + "hover:bg-hover/95")}>
                {k === "crops" ? "crop source" : k}
              </button>
            ))}
            {layers.cameras && (
              <span className="flex items-center gap-1.5 px-2 h-7 rounded
                               border border-line bg-panel/85 ml-1">
                <span className="text-[0.75rem] text-t3 whitespace-nowrap">
                  Camera size</span>
                <Slider min={0.3} max={4} step={0.1}
                        value={frustumScale}
                        label="camera frustum display size"
                        className="w-20"
                        onChange={(v) => {
                          setFrustumScale(v);
                          localStorage.setItem("carveout_frustum_scale",
                                               String(v));
                        }} />
              </span>
            )}
            <button type="button"
                    className="text-[0.75rem] px-2 h-7 rounded leading-none
                               border border-line bg-panel/85 text-t3
                               hover:text-t1 hover:bg-hover/95
                               whitespace-nowrap ml-1"
                    onClick={() => setZen(true)}>
              3D only
            </button>
          </div>}

          {/* PiP frame panel (1d) */}
          {!zen && selFrame !== null && framesByIdx.get(selFrame) && (
            <FramePiP scene={state.scene}
                      frame={framesByIdx.get(selFrame)!}
                      deleted={deletedFrames.includes(selFrame)}
                      // The PiP's delete/re-frame mutate manual_views.json —
                      // render-gate inputs — so an approved-FRESH render gate
                      // makes it read-only exactly as it does the panel: a
                      // change needs the explicit reopen, whichever gate is
                      // active when the operator opens a frame.
                      readOnly={running || (readOnly
                                || gateState(j, "render") === "approved")}
                      canvas={canvas}
                      onClose={() => setSelFrame(null)}
                      onChanged={() => { cams.reload(); }} />
          )}

          {/* selected-object readout: what the export knows about the
              object under the cursor's last click — sizes in metres only
              under a recorded factor, scene units otherwise */}
          {!zen && selInst !== null && (
            <InstanceCard idx={selInst} factor={factor}
                          onClose={() => canvas.current?.deselectInstance()} />
          )}

          {!zen && <StageStrip open={logOpen} onOpen={setLogOpen} />}
        </div>

        {/* gate panel drawer (P2 recipe: .94 inset bg, 1px border-left) */}
        {!zen && !panelHidden && (
        <aside aria-label="Inspector"
               className="w-[340px] xl:w-[380px] min-[2200px]:w-[428px] shrink-0
                          border-l border-line bg-panel/95 z-10
                          flex flex-col min-h-0">
          <div className="flex items-center gap-2 border-b border-line pl-4
                          h-10 shrink-0">
            <h2 className="text-[0.9375rem] font-semibold text-t1 m-0 flex-1
                           truncate">
              {active && active !== "report"
                && active !== "settings" ? GATE_LABEL[active]
                : active === "report" ? "Report"
                : active === "settings" ? "Viewer settings" : ""}
            </h2>
            <button className="w-9 h-10 flex items-center justify-center
                               border-l border-line text-[0.8125rem]
                               text-t3 hover:text-t1 hover:bg-hover"
                    aria-label="hide the panel"
                    onClick={() => setPanelHidden(true)}>
              ⟩
            </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
          {activeApproved && (
            <div className="border-b border-line bg-pass/10 px-4 py-2
                            flex items-center gap-2 flex-wrap">
              <Badge tone="pass">Approved</Badge>
              <span className="text-[0.75rem] text-t1 flex-1 min-w-[12ch]
                               leading-4">
                {readOnly ? "Read-only" : "Read-only · Reopen to edit"}
                <span className="block text-[0.6875rem] text-t3 font-mono">
                  {j!.gates[active as Gate]!.approved_at}</span>
              </span>
              {!readOnly && !running && (
                <Btn variant="destructive"
                     onClick={() => setReopenAsk(active as Gate)}>
                  Reopen gate…</Btn>
              )}
              {!readOnly && running && <JobNote show />}
            </div>
          )}
          <FooterSlot.Provider value={footerEl}>
          {active === "settings" && (
            <SettingsPanel canvas={canvas} focusEpoch={focusEpoch}
                           focusSelected={focusSel}
                           onFocusEdited={bumpFocus}
                           instancesLayer={layers.instances}
                           onInstancesLayer={(on) =>
                             setLayers({ ...layers, instances: on })} />
          )}
          {active === "volume" && (
            <VolumePanel boxes={boxes} vol={vol.data} factor={factor}
                         ruler={{ armed: rulerArmed, draft: rulerDraft }}
                         onRulerArm={armRuler}
                         level={{ armed: levelArmed, act: levelAct,
                                  draft: levelDraft, preview: levelPreview }}
                         onLevelArm={armLevel} onSquareArm={armSquare}
                         onLevelUndo={levelUndo}
                         onLevelPreview={setLevelPreview}
                         floorArmed={floorArmed} onFloorArm={armFloor}
                         onWholeScene={useWholeScene}
                         selected={selBox}
                         onSelect={selectBox} onSave={saveVolume}
                         onDraft={setVolDraft} canvas={canvas}
                         preview={volPreview} onPreview={setVolPreview}
                         readOnly={gateReadOnly} />
          )}
          {active === "render" && (
            <RenderPanel cams={cams.data} camsError={cams.error}
                         readOnly={gateReadOnly} canvas={canvas}
                         selFrame={selFrame} onSelectFrame={setSelFrame}
                         onChanged={() => cams.reload()} />
          )}
          {active === "vocabulary" &&
            <VocabPanel readOnly={gateReadOnly} />}
          {active === "exemplars" && (
            <ExemplarPanel readOnly={gateReadOnly} cams={cams.data}
                           data={exemplars.data}
                           reload={() => exemplars.reload()}
                           selFrame={selFrame}
                           onSelectFrame={setSelFrame} />
          )}
          {active === "verify_consent" && (
            <PipelinePanel readOnly={gateReadOnly} />
          )}
          {active === "report" && (
            <ReportDrawer onFlyTo={(idx) =>
              canvas.current?.flyToInstance(idx)} />
          )}
          </FooterSlot.Provider>
          </div>
          {/* fixed action footer; hidden while no panel portals into it */}
          <div ref={setFooterEl}
               className="shrink-0 border-t border-line bg-panel px-4 py-3
                          empty:hidden" />
        </aside>
        )}
      </div>

      {state.stopLook && <StopLookTakeover />}

      {reopenAsk && j && (
        <Confirm
          title={`Reopen ${GATE_LABEL[reopenAsk]}?`}
          body={<>Reopening <b>{GATE_LABEL[reopenAsk]}</b> also re-opens{" "}
            {demotedBy(j, reopenAsk).filter((g) => g !== reopenAsk)
              .map((g) => GATE_LABEL[g]).join(", ") || "nothing else"}.
            What they show stays on screen until re-approved.</>}
          confirmLabel="Reopen gate"
          destructive
          onConfirm={() => reopen(reopenAsk)}
          onCancel={() => setReopenAsk(null)} />
      )}
    </div>
  );
}

// PiP frame viewer with overlay toggle + view actions (1d)
/** The selected object's numbers, from the export (`/instances`): the
 *  aligned extents, the oriented-box extents, the position, the Gaussian
 *  count. One rule for the unit: metres when the scene's factor is
 *  RECORDED (× factor; 1.0 for a metric scene), scene units otherwise —
 *  never metres from an estimate. */
function InstanceCard({ idx, factor, onClose }: {
  idx: number; factor: number | null; onClose: () => void;
}) {
  const inst = useRunData<{ objects: InstanceRow[] }>("/instances");
  // the operator's relabel, here as well as on the report drawer
  const relabel = useRelabel(() => inst.reload());
  // the mark for verification, the same two places
  const mark = useVerifyMark(() => inst.reload());
  const o = inst.data?.objects.find((r) => r.idx === idx);
  if (!o) return null;
  const k = factor ?? 1;
  const unit = factor == null ? "units" : "m";
  const num = (v: number) => {
    const x = v * k;
    return factor == null ? Number(x.toPrecision(4)).toString()
                          : (Math.abs(x) >= 100 ? x.toFixed(1) : x.toFixed(2));
  };
  const dims = (v: number[]) => v.map(num).join(" × ");
  const aabb = [o.scale_xyz.x, o.scale_xyz.y, o.scale_xyz.z];
  const obb = o.obb ? [...o.obb.extents].sort((a, b) => b - a) : null;
  const Row = ({ l, v }: { l: string; v: React.ReactNode }) => (
    <div className="flex items-baseline gap-2">
      <span className="ui-label w-[6.5rem] shrink-0">{l}</span>
      <span className="font-mono text-t1 text-[0.75rem]">{v}</span>
    </div>);
  return (
    <div className="absolute bottom-10 right-2 w-[300px] z-10 rounded
                    border border-line2 bg-panel/95 shadow px-3 py-2"
         role="region" aria-label="selected object">
      <div className="flex items-center gap-2 text-[0.75rem] mb-1">
        <span className="text-t4 font-mono">{String(o.idx).padStart(2, "0")}</span>
        {relabel.isEditing(o) ? relabel.input(o, "flex-1 min-w-0") : (
          <span className="text-t1 font-medium flex-1 truncate"
                title={shownName(o) !== o.label ? `detected: ${o.label}` : undefined}>
            {shownName(o)}</span>)}
        {!relabel.isEditing(o) && relabel.pencil(o)}
        {!relabel.isEditing(o) && mark.button(o, "text-[0.6875rem]")}
        {o.verdict === "UNVERIFIED" && <Badge tone="warn">unverified</Badge>}
        {o.verdict === "NOT_JUDGED" && <Badge tone="warn">not judged</Badge>}
        {(o.hold_reasons?.length > 0 || o.verdict === "HELD") &&
          <Badge tone="warn">held</Badge>}
        <span className="font-mono text-t3">{o.conf.toFixed(2)}</span>
        <button type="button" aria-label="close"
                className="text-t3 hover:text-t1 px-1" onClick={onClose}>×</button>
      </div>
      <Row l={`size (${unit})`} v={dims(aabb)} />
      {obb && <Row l="oriented box" v={dims(obb)} />}
      <Row l={`position (${unit})`}
           v={[o.position.x, o.position.y, o.position.z].map(num).join(", ")} />
      {o.gaussians != null && <Row l="gaussians" v={o.gaussians.toLocaleString()} />}
      {o.verdict && (
        <div className="mt-1 text-[0.75rem] leading-4">
          <span className={verdictInfo(o.verdict).tone === "warn"
                             ? "text-warn" : "text-t1"}>
            {verdictInfo(o.verdict).word}</span>
          {o.verify_rationale && (
            <div className="text-t3 mt-0.5">{o.verify_rationale}</div>)}
          {proposalOf(o) && (
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-t3">the verifier proposed</span>
              {relabel.takeButton(o)}
            </div>)}
        </div>)}
      <div className="ui-help mt-1 mb-0">
        {factor == null
          ? "scene units (no factor recorded; the volume gate takes one)"
          : `metres at ${factor} m / unit; size is the aligned box on the scene axes`}
      </div>
    </div>
  );
}

function FramePiP({ scene, frame, deleted, readOnly, canvas, onClose,
                    onChanged }: {
  scene: string; frame: FrameEntry; readOnly: boolean;
  /** its view was deleted since the render: no acts, a note */
  deleted: boolean;
  canvas: React.RefObject<CanvasHandle>;
  onClose: () => void; onChanged: () => void;
}) {
  const { state } = useRun();
  const { act } = useSceneAct();
  const [overlay, setOverlay] = useState<"off" | "text" | "exemplar">("off");
  const [confirm, setConfirm] = useState<null | "delete" | "reframe">(null);
  const [big, setBig] = useState(false);   // expand over the 3D view
  const manual = frame.provenance?.startsWith("manual:");
  const mvId = manual ? frame.provenance.split(":", 2)[1] : null;

  // Cache-bust: frame/overlay identities change under one URL across
  // re-renders (a lesson learned twice) — the server says no-store, but the
  // browser's in-memory image cache can serve a stale <img> anyway (seen
  // live: a PiP showed the previous render's frame). Frames are versioned
  // by their sharpness (changes with the pixels); overlays by the journal
  // epoch (bumps past every probe/stage completion).
  const src = overlay === "off"
    ? runUrl(scene, `/frames/${frame.frame_idx}.png?v=${
        frame.sharpness ?? 0}`)
    : runUrl(scene, `/artifacts/stage2/overlays/${
        overlay === "exemplar" ? "exemplar_" : ""}frame_${
        String(frame.frame_idx).padStart(4, "0")}.png?v=${state.dataEpoch}`);

  const doDelete = async () => {
    setConfirm(null);
    if (await act(`/views/${mvId}`, undefined, { method: "del" })) {
      onChanged();
      onClose();
    }
  };

  const doReframe = async () => {
    setConfirm(null);
    const pose = canvas.current!.capturePose(frame.label ?? "view");
    if (await act(`/views/${mvId}`, {
      position: pose.position, quaternion: pose.quaternion,
      target: pose.target, target_source: pose.target_source,
      fov_deg: pose.fov_deg, aspect: pose.aspect }, { method: "patch" })) {
      onChanged();
    }
  };

  return (
    <div className={cx(
      "z-20 bg-panel/95 border border-line2 rounded flex flex-col",
      big ? "absolute inset-4 bottom-10" // expanded: over the 3D view
          : "absolute bottom-10 left-2 w-[390px]")}
         role="region" aria-label="frame preview">
      <div className="px-2 h-8 flex items-center gap-2 text-[0.75rem]
                      text-t2 border-b border-line">
        <span className="font-mono text-t1">
          frame_{String(frame.frame_idx).padStart(4, "0")}</span>
        {manual && <Badge tone="pass">M {frame.label}</Badge>}
        <span className="ui-num">sharp{" "}
          <span className="font-mono text-t1">
            {frame.sharpness?.toFixed(3)}</span></span>
        <span className="ui-num">cov{" "}
          <span className="font-mono text-t1">
            {frame.coverage?.toFixed(2)}</span></span>
        <div className="flex-1" />
        <button className="text-t3 hover:text-t1 px-1 rounded-sm"
                aria-label={big ? "shrink to corner"
                                : "expand over the 3D view"}
                onClick={() => setBig(!big)}>{big ? "⤡" : "⤢"}</button>
        <button className="text-t3 hover:text-t1 px-1 rounded-sm"
                aria-label="close frame preview"
                onClick={onClose}>✕</button>
      </div>
      <div className={cx(big && "flex-1 min-h-0 flex items-center",
                         big && "justify-center bg-app/60")}>
        <img src={src}
             className={cx(big ? "max-w-full max-h-full object-contain"
                               : "w-full block")}
             onError={(e) => ((e.target as HTMLImageElement).style.opacity =
               "0.2")} />
      </div>
      <div className="px-2 py-1 flex items-center gap-1.5 border-t
                      border-line flex-wrap">
        <span className="text-[0.75rem] text-t3">Overlay</span>
        <Segmented value={overlay} options={["off", "text", "exemplar"]}
                   label="frame overlay"
                   onChange={(m) => setOverlay(m as typeof overlay)} />
        <Btn variant="ghost" className="!px-1.5"
             onClick={() => canvas.current?.flyToFrame(frame)}>
          Fly to</Btn>
        {manual && !readOnly && !deleted && (
          <Btn variant="ghost" className="!px-1.5"
               onClick={() => setConfirm("reframe")}>
            Re-frame</Btn>
        )}
        {manual && !readOnly && !deleted && (
          <Btn variant="destructive" className="!px-1.5 ml-auto"
               onClick={() => setConfirm("delete")}>
            Delete</Btn>
        )}
        {deleted && (
          <span className="text-[0.6875rem] text-warn ml-auto"
                title="the view is gone from the file; the frame stays until the next render">
            deleted; Re-render drops it</span>
        )}
        {!manual && !readOnly && (
          <Btn variant="ghost" className="!px-1.5 ml-auto" disabled
               title="auto views are regenerated at each render; Convert to fully manual on the Render panel to edit or delete them">
            Delete</Btn>
        )}
      </div>
      {confirm === "delete" && (
        <Confirm title="Delete manual view?"
                 body={<>Deleting <b>{frame.label}</b> re-opens the Render
                   gate and every gate after it; what they show stays on
                   screen until re-approved.</>}
                 confirmLabel="Delete view"
                 destructive onConfirm={doDelete}
                 onCancel={() => setConfirm(null)} />
      )}
      {confirm === "reframe" && (
        <Confirm title="Re-frame this view?"
                 body={<>The saved pose is replaced by the current view.
                   The Render gate and every gate after it re-open, as
                   with a delete and a new capture.</>}
                 confirmLabel="Save new pose"
                 onConfirm={doReframe} onCancel={() => setConfirm(null)} />
      )}
    </div>
  );
}
