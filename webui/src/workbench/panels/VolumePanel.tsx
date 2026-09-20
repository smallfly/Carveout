// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Volume gate (1b): boxes live in 3D with gizmo handles; heights are shown
// THREE ways: always-on 3D tags, the elevation strip
// (density silhouette + box spans), and numeric fields. The 2D density
// tab is the plan-view precision surface (sidecar; PNG fallback).
// After user testing: center/size fields with SLIDERS beside
// every numeric (numerics stay as the precision path),
// slider ranges from the density sidecar extent; MOVE/SCALE toggle for
// the 3D axis-cross drag.

import { useEffect, useMemo, useRef, useState } from "react";
import { runUrl } from "../../api";
import { useCalibration, useJobRunning, useRun, useRunData,
         useSceneAct, sceneFactor } from "../../store";
import type { AlignmentBlock, AlignmentRead, CalibrationResp, CheckLine,
              DensitySidecar, ProposalResp, ScaleRead, VolumeBox,
              VolumeResp } from "../../types";
import { describeAngles, sameAlignment } from "../alignment";
import type { CanvasHandle } from "../SplatCanvas";
import { LEVEL_COLOURS } from "../overlays/ruler";
import type { GizmoMode } from "../overlays";
import { Badge, Btn, Details, LabeledNum, NumInput, PanelFooter, Section,
         Segmented, Slider, Switch, cx, Cued, JobNote, Steps } from "../../ui";
import { currentStep, scaleWaiting, volumeSteps, type LevelState } from "../../steps";
import { gateState } from "../../journal";

const UP_AXES = ["auto", "+x", "-x", "+y", "-y", "+z", "-z"] as const;
/** Scene-unit readouts to four significant digits: a unit may be a
 *  kilometre or a millimetre, and two fixed decimals read a
 *  kilometre-unit scene as 0.00 (metre readouts keep toFixed(2)). */
const fmtUnits = (v: number) => Number(v.toPrecision(4)).toString();

export default function VolumePanel({ boxes, vol, factor, selected, onSelect,
                                      onSave, onDraft, onWholeScene, canvas,
                                      readOnly, ruler, onRulerArm,
                                      level, onLevelArm, onSquareArm, onLevelUndo,
                                      onLevelPreview, floorArmed, onFloorArm,
                                      preview, onPreview }: {
  /** the preview: everything outside the box being edited collapses to
   *  points and follows the handles — display only, nothing saved */
  preview: boolean;
  onPreview: (on: boolean) => void;
  /** the level act: armed, the points clicked so far, and the
   *  proposed alignment being previewed on the canvas (the Workbench
   *  holds them — the canvas outlives this panel) */
  level: { armed: boolean; act: "level" | "square" | null; draft: number[][];
           preview: AlignmentBlock | null };
  onLevelArm: (on: boolean) => void;
  /** the square act: two points along an edge, the yaw alone */
  onSquareArm: (on: boolean) => void;
  /** drops the last point clicked (Undo last, Backspace) */
  onLevelUndo: () => void;
  /** the floor act: one click on the floor sets its height */
  floorArmed: boolean;
  onFloorArm: (on: boolean) => void;
  onLevelPreview: (a: AlignmentBlock | null) => void;
  boxes: VolumeBox[] | null;
  vol: VolumeResp | null;
  /** metres per scene unit when recorded or measured, null = not yet
   *  (store.sceneFactor) */
  factor: number | null;
  /** the ruler: armed, and the points clicked so far (the Workbench holds
   *  them — the canvas outlives this panel) */
  ruler: { armed: boolean; draft: number[][] | null };
  onRulerArm: (on: boolean) => void;
  /** the whole-scene act: scope everything, no boxes (an explicit decision
   *  the journal hashes, never a default) */
  onWholeScene: () => void;
  selected: number;
  onSelect: (i: number) => void;
  onSave: (b: VolumeBox[]) => void;
  /** live slider preview: gizmo follows, nothing is written yet */
  onDraft: (b: VolumeBox[]) => void;
  canvas: React.RefObject<CanvasHandle>;
  readOnly: boolean;
}) {
  const { state } = useRun();
  const running = useJobRunning();
  const wholeScene = vol?.volume?.scope === "whole_scene";
  // the gate's steps, and why Propose is dead without a scale (the
  // server refuses the same way; saying it here spares the red banner)
  const approved = gateState(state.journal, "volume") === "approved";
  const hasVolume = !!vol?.volume && (wholeScene || !!vol.volume.boxes?.length);
  const scaleWait = scaleWaiting(factor);
  const density = useRunData<DensitySidecar>("/density.json");
  // the last proposal: what the Check card shows, and where the level read
  // comes from (one fetch, shared by the card, the steps and the level row)
  const proposal = useRunData<ProposalResp>("/proposal");
  const { calib } = useCalibration();
  const levelState = levelStateOf(proposal.data, calib?.scene.alignment ?? null);
  // the boxes on disk were made under the levelling density.json records;
  // once the operator levels again they are a proposal for another frame
  const volumeCurrent = sameAlignment(vol?.frame?.alignment ?? null,
                                      calib?.scene.alignment ?? null, 0.01);
  // the steps once, for the list and for the cue on the block that owns
  // the current one
  const steps = volumeSteps({ factor, hasVolume, approved, level: levelState,
                              volumeStale: hasVolume && !volumeCurrent });
  const cue = currentStep(steps);
  const readNow = proposal.data?.analysis?.alignment ?? null;
  const readStale = !!readNow && !readCurrent(readNow, calib?.scene.alignment ?? null);
  const [tab, setTab] = useState<"boxes" | "density">("boxes");
  // Which manipulator the selected box carries in 3D. Rotate is absent by
  // construction, not by omission: volume.json is min/max and in_volume()
  // reads it that way, so a rotated box could not be saved or honoured.
  const [gizmo, setGizmo] = useState<GizmoMode>("translate");
  const chooseGizmo = (m: GizmoMode) => {
    setGizmo(m);
    canvas.current?.setGizmoMode(m);
  };

  const frame = vol?.frame;
  const up = frame ? "xyz".indexOf(frame.up_axis) : 1;
  // Heights are shown SIGNED (up = positive), the convention the Floor
  // field and density.json use — box coordinates are raw world values, so
  // on a -y scene the raw numbers read upside down.
  const sign = frame?.up_sign ?? 1;
  const signedSpan = (b: VolumeBox): [number, number] => {
    const a = b.min[up] * sign, c = b.max[up] * sign;
    return [Math.min(a, c), Math.max(a, c)];
  };

  const { approveGate, startStage } = useSceneAct();

  const propose = () => startStage("propose_volume");
  const approve = () => approveGate("volume");

  // center/size <-> min/max: edits keep the other quantity fixed
  const withField = (i: number, kind: "center" | "size", a: number,
                     v: number): VolumeBox[] | null => {
    if (!boxes) return null;
    return boxes.map((b, k) => {
      if (k !== i) return b;
      const c = (b.min[a] + b.max[a]) / 2, s = b.max[a] - b.min[a];
      const nc = kind === "center" ? v : c;
      const ns = Math.max(kind === "size" ? v : s, 0.05);
      const min = [...b.min], max = [...b.max];
      min[a] = nc - ns / 2;
      max[a] = nc + ns / 2;
      return { ...b, min, max };
    });
  };

  // The last slider draft: pointerup commits it (one PUT per gesture,
  // same save path as a gizmo drag end).
  const draftRef = useRef<VolumeBox[] | null>(null);
  const draft = (b: VolumeBox[] | null) => {
    if (!b) return;
    draftRef.current = b;
    onDraft(b);
  };
  const commit = (b?: VolumeBox[] | null) => {
    const next = b ?? draftRef.current;
    draftRef.current = null;
    if (next) onSave(next);
  };

  // Slider ranges: the density sidecar's extent (ground grid + height
  // histogram); scenes without the sidecar fall back to the SAVED
  // boxes' union padded by its own span. Ranges derive from the saved
  // volume, never the live draft — a range recomputed per drag event
  // shifted min/max together with the value, so the thumb never moved
  // (seen in user testing). Numerics accept any value regardless.
  const savedBoxes = vol?.volume?.boxes;
  // A sidecar with no grid (shape [0, 0]) is the frame-only file a refused
  // proposal writes so this gate can show the scene's dimensions and take
  // the scale; it has the height histogram but nothing to plan on.
  const hasGrid = !!density.data && density.data.shape[0] > 0;
  const ranges = useMemo(() => {
    if (density.data && hasGrid) {
      const d = density.data;
      const [g0, g1] = d.ground_axes;
      const lo = [0, 0, 0], hi = [0, 0, 0];
      lo[g0] = d.origin[0];
      hi[g0] = d.origin[0] + d.shape[0] * d.voxel;
      lo[g1] = d.origin[1];
      hi[g1] = d.origin[1] + d.shape[1] * d.voxel;
      const e = d.height_hist.edges;
      lo[d.up_axis] = e[0] - 0.5;
      hi[d.up_axis] = e[e.length - 1] + 0.5;
      return { lo, hi };
    }
    if (savedBoxes?.length) {
      const lo = [Infinity, Infinity, Infinity];
      const hi = [-Infinity, -Infinity, -Infinity];
      for (const b of savedBoxes)
        for (let a = 0; a < 3; a++) {
          lo[a] = Math.min(lo[a], b.min[a]);
          hi[a] = Math.max(hi[a], b.max[a]);
        }
      const pad = Math.max(...[0, 1, 2].map((a) => hi[a] - lo[a])) || 1;
      return { lo: lo.map((v) => v - pad), hi: hi.map((v) => v + pad) };
    }
    return null;
  }, [density.data, hasGrid, savedBoxes]);

  // Metres per scene unit, for the readouts below — ONLY when a factor is
  // recorded (the prop, from the one derivation). Coordinates stay in
  // scene units (that is what volume.json holds); the panel says what they
  // mean beside them, and converts by nothing it does not know: with the
  // factor unknown the readouts are in units and say so.
  // scene units to significant digits — a unit may be a kilometre or a
  // millimetre, and fixed decimals read a kilometre-unit scene as 0.00
  const u = (units: number) => fmtUnits(units);
  // Metres are "≈" after a measurement (one known length, R1) and exact
  // after a declared or typed factor.
  const approx = calib?.scene.measured ? "≈" : "";
  const m = (units: number) => approx + (units * (factor ?? 1)).toFixed(2);
  // The tallest box in metres, under a RECORDED factor only: a box taller
  // than any room or object is the one visible sign that the factor, the
  // up axis or the floor is wrong, and the note says which to check.
  const tallestM = useMemo(() => {
    if (!boxes || !boxes.length || factor == null) return null;
    return Math.max(...boxes.map((b) => Math.abs(b.max[up] - b.min[up]))) * factor;
  }, [boxes, up, factor]);
  const heightsSane = tallestM == null || tallestM < 15;

  return (
    <div>
      <Section title="Volume gate" band>
        <p className="ui-help m-0">
          Review and correct the proposed volume; you never draw from
          scratch.
        </p>
        <Steps steps={steps} />
        <Details>
          Carveout proposed this volume from the scene's density. A healthy
          one has its floor near the real floor and hugs the subject.
        </Details>
        {!heightsSane && (
          <div className="mt-2 ui-note-warn">
            At this scale the tallest box is{" "}
            <b>{approx}{tallestM!.toFixed(1)} m</b>; no room or object is.
            One of three things is wrong; check them below in this order:
            the <b>scale</b> (a wrong measurement makes the whole scene
            read too large; measure again, or press Metric if the capture
            is metric), the <b>up axis</b> (a box that is "tall" along the
            wrong axis), the <b>floor</b> (a floor elected far from the
            real one). Do not approve until the height reads like the real
            scene.
          </div>
        )}
      </Section>

      <SceneFrameCalibration readOnly={readOnly || running} locked={readOnly}
                             cue={cue} scale={vol?.scale ?? null}
                             ruler={ruler} onRulerArm={onRulerArm}
                             read={readStale ? null : readNow}
                             level={level} onLevelArm={onLevelArm}
                             onSquareArm={onSquareArm} onLevelUndo={onLevelUndo}
                             onLevelPreview={onLevelPreview}
                             floorArmed={floorArmed} onFloorArm={onFloorArm} />

      <div className="flex border-b border-line px-4" role="tablist">
        {(["boxes", "density"] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t}
                  onClick={() => setTab(t)}
                  className={cx("flex-1 h-8 text-[0.8125rem] border-b-2 -mb-px",
                                tab === t
                                  ? "border-t1 text-t1 font-medium"
                                  : "border-transparent text-t3 hover:text-t1")}>
            {t === "boxes" ? "3D / boxes" : "2D density"}
          </button>
        ))}
      </div>

      {tab === "boxes" && (
        <>
          <CheckCard locked={readOnly} factor={factor} p={proposal.data} />
          <Section title="3D handles" locked={readOnly}
                   right={<Segmented options={["move", "size"]}
                     label="3D handle mode"
                     value={gizmo === "translate" ? "move" : "size"}
                     disabled={readOnly || running}
                     onChange={(v) =>
                       chooseGizmo(v === "move" ? "translate" : "scale")} />}>
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[0.75rem] text-t2">
                Preview: outside as points</span>
              <Switch checked={preview} disabled={!boxes?.length}
                      label="preview outside the box as points"
                      onChange={onPreview} />
            </div>
            <p className="ui-help m-0">
              Drag the handles on the selected box in the view; the numbers
              below follow the drag and save when you let go. With the
              preview on, everything outside the box collapses to points
              and follows the handles as you drag: display only, nothing
              the pipeline reads and nothing saved. It stays on through the
              other stages; opening the Viewer panel hands it to that
              panel's focus volume.
            </p>
            <Details>
              <b>Move</b> shifts the box's centre, <b>Size</b> grows or
              shrinks it about that centre. Boxes stay axis-aligned; there
              is no rotation to give them.
            </Details>
          </Section>

          <Section title={wholeScene ? "Scope: the whole scene"
                                     : `Boxes (${boxes?.length ?? 0})`}
                   locked={readOnly} cue={cue === "review"}
                   right={!readOnly && !running && !wholeScene && (
                     <Btn variant="ghost" onClick={() => {
                       if (!boxes) return;
                       const base = boxes[0] ?? {
                         name: "", min: [-1, -1, -1], max: [1, 1, 1] };
                       onSave([...boxes, {
                         name: `region_${boxes.length}`,
                         min: [...base.min], max: [...base.max] }]);
                     }}>+ Add box</Btn>)}>
            {wholeScene && (
              <p className="ui-help m-0 mb-2">
                Every Gaussian is in scope; cameras use the scene's robust
                bounds and coverage is measured over them. Press Propose
                fresh to go back to boxes.
              </p>
            )}
            {!boxes && <div className="ui-help">
              no volume yet; propose one</div>}
            {boxes?.map((b, i) => (
              <div key={i}
                   onClick={() => onSelect(i)}
                   aria-selected={i === selected}
                   className={cx("border rounded p-2 mb-2 cursor-pointer",
                                 i === selected
                                   ? "border-t2 bg-inset"
                                   : "border-line hover:border-line2")}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-[0.8125rem] text-t1
                                   font-medium">{b.name}</span>
                  {!readOnly && !running && boxes.length > 1 && (
                    <button className="text-t3 hover:text-fail text-[0.8125rem]
                                       px-1 rounded-sm"
                            aria-label={`remove ${b.name}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              onSave(boxes.filter((_, k) => k !== i));
                              onSelect(-1);
                            }}>✕</button>
                  )}
                </div>
                {[0, 1, 2].map((a) => (
                  <FieldRow key={`c${a}`} label={`center ${"xyz"[a]}`}
                            accent={a === up} disabled={readOnly || running}
                            value={(b.min[a] + b.max[a]) / 2}
                            min={ranges?.lo[a] ?? -10}
                            max={ranges?.hi[a] ?? 10}
                            onDraft={(v) => draft(withField(i, "center",
                                                           a, v))}
                            onCommit={(v) => commit(
                              v === undefined ? undefined
                                : withField(i, "center", a, v))} />
                ))}
                {[0, 1, 2].map((a) => (
                  <FieldRow key={`s${a}`} label={`size ${"xyz"[a]}`}
                            accent={a === up} disabled={readOnly || running}
                            value={b.max[a] - b.min[a]}
                            min={(ranges ? Math.max(
                              ranges.hi[a] - ranges.lo[a], 1e-9) : 20) * 1e-3}
                            max={ranges ? Math.max(
                              ranges.hi[a] - ranges.lo[a], 1e-9) : 20}
                            onDraft={(v) => draft(withField(i, "size",
                                                            a, v))}
                            onCommit={(v) => commit(
                              v === undefined ? undefined
                                : withField(i, "size", a, v))} />
                ))}
                <div className="mt-1 font-mono text-[0.75rem] text-t3">
                  {factor == null ? (
                    <>{u(b.max[0] - b.min[0])} × {u(b.max[1] - b.min[1])} ×{" "}
                      {u(b.max[2] - b.min[2])} units · height{" "}
                      {u(signedSpan(b)[0])} … {u(signedSpan(b)[1])}
                      <span className="text-t4"> (no metres recorded)</span></>
                  ) : (
                    <>{m(b.max[0] - b.min[0])} × {m(b.max[1] - b.min[1])} ×{" "}
                      {m(b.max[2] - b.min[2])} m · height{" "}
                      {m(signedSpan(b)[0])} … {m(signedSpan(b)[1])} m
                      {factor !== 1 && (
                        <span className="text-t4"> (fields above in scene
                          units, {factor} m each)</span>)}</>
                  )}
                </div>
              </div>
            ))}
          </Section>

          {density.data && boxes && (
            <Section title="Elevation (density silhouette + box heights)">
              <ElevationStrip d={density.data} boxes={boxes} up={up}
                              sign={sign} />
            </Section>
          )}
        </>
      )}

      {tab === "density" && (
        <Section title="Top-down density">
          {density.data && hasGrid && boxes ? (
            <DensityPlan d={density.data} boxes={boxes} selected={selected}
                         onSelect={onSelect} />
          ) : density.data && !hasGrid ? (
            <div className="ui-help">
              no density plan; the proposal was refused before it could
              scan; the scene's dimensions are in the calibration above</div>
          ) : (
            <div className="ui-help">no density data yet</div>
          )}
        </Section>
      )}

      <PanelFooter>
        <Btn onClick={propose} disabled={readOnly || running || !!scaleWait}
             variant={cue === "propose" ? "primary" : undefined}
             title={scaleWait ?? undefined} kbd="P">
          Propose fresh</Btn>
        {/* The whole-scene act: an explicit decision, not the absence of
          * one — the journal hashes the file that records it. */}
        <Btn onClick={onWholeScene}
             disabled={readOnly || running || wholeScene || !!scaleWait}
             title={scaleWait ?? "scope every Gaussian: no boxes; cameras over the scene's bounds"}>
          Whole scene</Btn>
        <JobNote show={running} />
        <div className="flex-1" />
        {scaleWait && !running && (
          <span className="text-[0.75rem] text-warn text-right leading-tight">
            {scaleWait}</span>)}
        <Btn variant="primary" kbd="⏎"
             disabled={readOnly || running || (!boxes?.length && !wholeScene)
                       || (hasVolume && !volumeCurrent)}
             title={hasVolume && !volumeCurrent
                    ? "these boxes were proposed before the levelling; press Propose fresh first"
                    : undefined}
             onClick={approve}>
          Approve volume</Btn>
      </PanelFooter>
    </div>
  );
}

/** The level step's state, from the last proposal's read and the
 *  alignment recorded now: known once a proposal carries a read; adopted
 *  when the recorded block is the one it proposed (within a degree), or
 *  the operator's own from their clicks (those are the authority,
 *  whatever the detector reads). */
function levelStateOf(p: ProposalResp | null,
                      current: AlignmentBlock | null): LevelState | null {
  const read = p?.analysis?.alignment;
  if (!p || read === undefined) return null;
  if (!read) return { known: true, offLevel: false, adopted: false,
                      reason: p.check.level?.reason ?? null };
  // a read made under another levelling than the one recorded now says
  // nothing about the scene as it stands: Propose reads it again
  if (!readCurrent(read, current))
    return { known: false, offLevel: false, adopted: false,
             reason: STALE_READ };
  const proposed = proposedBlock(read, current);
  return { known: true, offLevel: read.off_level || !!read.off_square,
           adopted: (!!proposed && sameAlignment(proposed, current))
                    || current?.source === "levelled",
           reason: p.check.level?.reason ?? null };
}

/** The sentence for a read (or a volume) made before the scene was
 *  levelled the way it is now. */
const STALE_READ = "the scene was levelled since this was read; press Propose "
  + "fresh to read it again";
const STALE_VOLUME = "these boxes were proposed before the levelling; press "
  + "Propose fresh again";

/** True when the read ran under the levelling recorded now (the file's
 *  axes count as a levelling of none). */
function readCurrent(read: AlignmentRead, current: AlignmentBlock | null): boolean {
  return sameAlignment(read.current ?? null, current, 0.01);
}

/** The block "Level as proposed" writes: the read's tilt, and its yaw
 *  only when the read is sure of a wall direction. */
function proposedBlock(read: AlignmentRead | null | undefined,
                       current: AlignmentBlock | null = null): AlignmentBlock | null {
  if (!read || !(read.off_level || read.off_square)) return null;
  // the tilt as read (absolute); the yaw as read when the walls are sure,
  // else whatever is recorded now — squaring alone keeps the levelling
  return { tilt_deg: [read.tilt_deg[0], read.tilt_deg[1]],
           yaw_deg: read.confident_yaw ? (read.yaw_deg ?? 0) : (current?.yaw_deg ?? 0),
           source: "proposed" };
}

/** The filming answers in the tiles' words. */
const FILMED: Record<string, string> = {
  interior: "inside a space", orbit: "around a subject",
  ground: "outdoors on the ground", manual: "manual views", auto: "no answer yet",
};

/** One check line: the geometry's (or the model's) opinion, stated. */
function CheckRow({ label, line }: { label: string; line: CheckLine | null }) {
  if (!line) return null;
  const tone = line.fits === false ? "text-warn" : line.fits ? "text-t2" : "text-t3";
  return (
    <>
      <span>{label}</span>
      <span className={tone}>
        {line.fits === false && <b>flagged: </b>}
        {line.reason.split("\n")[0]}
      </span>
    </>
  );
}

/** The Volume gate's Check card: what the last proposal READ
 *  about the operator's two answers — the measures in metres at the
 *  scene's factor (≈ after a measurement), the region the proposer took,
 *  the scale sanity line under "inside a space", the filming check, and
 *  the model's second opinion on the measured segment. Everything here is
 *  stated; nothing is applied and nothing is adopted from this card (the
 *  render panel's card holds the two measured proposals). */
function CheckCard({ locked, factor, p }: { locked: boolean; factor: number | null;
                                            p: ProposalResp | null }) {
  const { calib } = useCalibration();
  const a = p?.analysis;
  const enclosed = a?.enclosure.used === "enclosed";
  const read = p?.read;
  const readOk = read && "length_m" in read ? read : null;
  const readErr = read && "error" in read ? read.error : null;
  const approx = calib?.scene.measured ? "≈" : "";
  const f = factor ?? 1;
  const m = (units: number) => `${approx}${(units * f).toFixed(2)} m`;
  const relief = a ? a.ground.relief_p5_p95[1] - a.ground.relief_p5_p95[0] : 0;
  return (
    <Section title="Check" locked={locked}
             right={p && <span className="text-[0.6875rem] text-t4 font-mono">
               {p.generated}</span>}>
      {!p && (
        <p className="ui-help m-0">
          Press Propose fresh; what the geometry says about your answers
          appears here.
        </p>
      )}
      {p && a && (
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5
                        text-[0.75rem] font-mono text-t3 ui-num">
          <span>measures</span>
          <span className="text-t2">
            {m(a.footprint_units[0])} × {m(a.footprint_units[1])} across ×{" "}
            {m(a.span_units)} tall
          </span>
          <span>region</span>
          <span className="text-t2">
            {enclosed ? "enclosed: walls around a floor"
                      : "the density core"}
            {a.enclosure.reason === "not_scanned" && " (filmed around a subject or outdoors: no enclosure scan)"}
            {a.enclosure.reason === "nothing_enclosed" && " (nothing enclosed was found)"}
            {a.enclosure.reason === "over_budget" && " (the scene was too large to scan for an enclosure)"}
          </span>
          <span>ground</span>
          <span className="text-t2">
            relief {m(relief)} · standable on{" "}
            {Math.round(100 * a.ground.standable_frac)}% of the map
          </span>
          <CheckRow label="filmed" line={{
            fits: p.check.path_mode.fits,
            reason: `you said ${FILMED[p.check.path_mode.said] ?? p.check.path_mode.said}; ${p.check.path_mode.reason}`
              + (p.check.path_mode.fits === false ? "; change it on the render panel" : "") }} />
          <CheckRow label="height" line={p.check.scale} />
          <CheckRow label="level" line={
            p.analysis.alignment && !readCurrent(p.analysis.alignment, calib?.scene.alignment ?? null)
              ? { fits: null, reason: STALE_READ } : p.check.level ?? null} />
          <CheckRow label="measured" line={readOk ? p.check.opinion : null} />
        </div>
      )}
      {readErr && (
        <div className="mt-2 ui-note-warn">
          <div className="mb-1">The model's second opinion on your
            measurement did not run:</div>
          <pre className="m-0 whitespace-pre-wrap font-mono text-[0.6875rem]
                          leading-4 text-t2">{readErr.trim()}</pre>
        </div>
      )}
      {readOk && (
        <p className="ui-help mt-2 mb-0">
          Read by {readOk.model} in {readOk.elapsed_s} s from the probe
          renders with your segment drawn in ({readOk.probes.length} view
          {readOk.probes.length === 1 ? "" : "s"}): {readOk.reason}
        </p>
      )}
      {p && (
        <Details>
          Carveout does not decide what this scene is. You said how it was
          filmed and what a unit is; the geometry and the model only check
          those answers and say so here. A flagged line is a reason to look
          again: measure again, or change the filming answer on the render
          panel; never a change Carveout makes on its own.
        </Details>
      )}
    </Section>
  );
}

/** Under each armed block: the hover ring says where the click
 *  lands before it does. */
const PickHint = () => (
  <p className="ui-help m-0 mt-1">
    Rest the pointer: a ring shows where the click will land; amber means
    a stray splat, move a little.
  </p>
);

/** Scene-frame calibration: the detected floor (the 3D plane shows it;
 * type it) + up-axis override. Editing re-opens the volume gate onward — the
 * scene frame drives propose + camera placement. */
function SceneFrameCalibration({ readOnly, locked, cue, scale, ruler, onRulerArm,
                                 read, level, onLevelArm, onSquareArm, onLevelUndo,
                                 onLevelPreview, floorArmed, onFloorArm }:
                               { readOnly: boolean; locked: boolean;
                                 /** the gate's current step: the block that
                                  *  owns it wears the cue */
                                 cue: string | null;
                                 scale: ScaleRead | null;
                                 ruler: { armed: boolean; draft: number[][] | null };
                                 onRulerArm: (on: boolean) => void;
                                 read: AlignmentRead | null;
                                 level: { armed: boolean; act: "level" | "square" | null;
                                          draft: number[][];
                                          preview: AlignmentBlock | null };
                                 onLevelArm: (on: boolean) => void;
                                 onSquareArm: (on: boolean) => void;
                                 onLevelUndo: () => void;
                                 onLevelPreview: (a: AlignmentBlock | null) => void;
                                 floorArmed: boolean;
                                 onFloorArm: (on: boolean) => void }) {
  const { calib, write } = useCalibration();
  const sc = calib?.scene;
  if (!sc) return null;
  const effective = sc.floor ?? sc.floor_auto ?? 0;
  // The factor: a number when recorded or measured, null = not yet — the
  // scene then has no scale, Propose refuses, and the ruler is the act.
  // The floor is a coordinate in scene units; it is restated in metres
  // only with a factor.
  const factor = sceneFactor(calib);
  const measured = sc.measured;
  const overridden = sc.floor !== null && sc.floor !== undefined;
  return (
    <Section title="Scene frame calibration" locked={locked}>
      <p className="ui-help m-0">
        Editing re-opens the volume gate; the frame drives placement.
      </p>
      <Details className="mb-1.5">
        The floor is drawn in 3D as a translucent plane; set its height here.
        The up axis and the scale are conventions the file does not record;
        they are stated here so what the units mean is written down.
      </Details>
      <Cued on={cue === "review"}>
      <LabeledNum label="floor" value={Number(effective.toPrecision(6))}
                  disabled={readOnly}
                  onCommit={(v) => write({ scene: { floor: v } })}
                  hint={
                    <span className="font-mono text-[0.75rem] text-t4">
                      auto {sc.floor_auto == null ? "none" : fmtUnits(sc.floor_auto)}
                      {factor != null && factor !== 1 && (
                        <> · {measured ? "≈" : ""}{(effective * factor).toFixed(2)} m</>)}
                    </span>} />
      <div className="flex items-center gap-2 mt-1 min-h-[24px]">
        {overridden ? (
          <Badge tone="accent">Override active</Badge>
        ) : (
          <span className="text-[0.75rem] text-t4">auto-detected</span>
        )}
        <div className="flex-1" />
        {!floorArmed && !readOnly && (
          <Btn variant="outline"
               title="click one point on the floor in the 3D view; its height becomes the floor"
               onClick={() => onFloorArm(true)}>
            Set from a point</Btn>
        )}
        {overridden && !readOnly && !floorArmed && (
          <Btn variant="ghost"
               onClick={() => write({ scene: { floor: null } })}>
            Reset to auto</Btn>
        )}
      </div>
      {floorArmed && (
        <div className="mt-1 border border-line2 rounded p-2 bg-inset"
             role="group" aria-label="set the floor from a point">
          <p className="text-[0.8125rem] text-t1 m-0 leading-5">
            Click one point on the floor in the 3D view.{" "}
            <span className="text-t3">(the grid and the floor plane move there; Esc cancels)</span>
          </p>
          <PickHint />
          <div className="flex gap-2 mt-2">
            <Btn variant="ghost" onClick={() => onFloorArm(false)}>Cancel</Btn>
          </div>
        </div>
      )}
      <div className="flex items-center gap-2 mt-2 flex-wrap">
        <span className="ui-label flex-1">up axis</span>
        <Segmented value={sc.up_axis} options={UP_AXES} disabled={readOnly}
                   label="up axis"
                   onChange={(v) => write({ scene: { up_axis: v } })} />
      </div>
      </Cued>

      <Cued on={cue === "level"}>
      <LevelBlock readOnly={readOnly} current={sc.alignment ?? null} read={read}
                  level={level} onLevelArm={onLevelArm} onSquareArm={onSquareArm}
                  onLevelUndo={onLevelUndo}
                  onLevelPreview={onLevelPreview} write={write} />
      </Cued>

      <Cued on={cue === "scale"}>
      <ScaleBlock readOnly={readOnly} factor={factor} measured={measured}
                  scale={scale} ruler={ruler} onRulerArm={onRulerArm}
                  write={write} />
      </Cued>
    </Section>
  );
}

/** Level: is the scene's floor level, and the three acts that make
 *  it so. Like the up axis and the scale it is a CONVENTION the file does
 *  not record, but unlike them the scene has an opinion: the proposal reads
 *  the tilt and says so on the Check card, and the operator adopts it,
 *  levels by hand from points on the floor, or keeps the file's axes.
 *  Nothing is applied by itself; a change re-opens this gate onward. */
function LevelBlock({ readOnly, current, read, level, onLevelArm, onSquareArm,
                      onLevelUndo, onLevelPreview, write }: {
  readOnly: boolean;
  current: AlignmentBlock | null;
  read: AlignmentRead | null;
  level: { armed: boolean; act: "level" | "square" | null; draft: number[][];
           preview: AlignmentBlock | null };
  onLevelArm: (on: boolean) => void;
  onSquareArm: (on: boolean) => void;
  onLevelUndo: () => void;
  onLevelPreview: (a: AlignmentBlock | null) => void;
  write: (u: unknown) => void;
}) {
  const { act, busy } = useSceneAct();
  const proposed = proposedBlock(read, current);
  const canAdopt = !!proposed && !sameAlignment(proposed, current);
  // a recorded alignment retires any preview (the adopt landed, or a reset)
  const curKey = current ? `${current.tilt_deg.join(",")},${current.yaw_deg}` : "";
  useEffect(() => { onLevelPreview(null); }, [curKey]);   // eslint-disable-line react-hooks/exhaustive-deps
  const pts = level.draft;
  const floorDone = pts.length >= 3;
  const wallDone = pts.length >= 5;
  const adopt = () => {
    if (!proposed) return;
    write({ scene: { alignment: { tilt_deg: proposed.tilt_deg,
                                  yaw_deg: proposed.yaw_deg,
                                  source: "proposed" } } });
    onLevelPreview(null);
  };
  const reset = () => write({ scene: { alignment: null } });
  const squaring = level.act === "square";
  const edgeDone = pts.length >= 2;
  const done = async () => {
    if (squaring) {
      if (!edgeDone) return;
      const ok = await act("/alignment/square", { edge: pts.slice(0, 2) });
      if (ok) onSquareArm(false);
      return;
    }
    if (!floorDone) return;
    const ok = await act("/alignment/level", {
      floor: pts.slice(0, 3), wall: wallDone ? pts.slice(3, 5) : null });
    if (ok) onLevelArm(false);
  };
  const tilt = read ? read.tilt_total_deg : null;
  const words = current
    ? `levelled · ${describeAngles(current)} · ${
        current.source === "levelled" ? "from the points you clicked"
                                      : "proposed from the scene"}`
    : read
      ? (read.off_level ? `the file's axes · off by ${tilt!.toFixed(1)}°`
                        : `the file's axes · the floor reads ${tilt!.toFixed(1)}° off`)
      : "the file's axes";
  return (
    <div className="mt-2">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="ui-label shrink-0">level</span>
        <span className={cx("font-mono text-[0.75rem] text-right flex-1 min-w-0",
                            read?.off_level && !current ? "text-warn" : "text-t3")}>
          {words}
        </span>
      </div>
      {read?.off_level && !current && (
        <p className="text-[0.8125rem] text-t2 mt-1 mb-0 leading-5">
          This scene is not level, so a straight box cannot hug it. Level it
          as proposed, or from three points on the floor, then propose again.
        </p>
      )}
      {read && !read.off_level && !current && (
        <p className="ui-help mt-1 mb-0">
          This scene reads level on the file's own axes; nothing to do
          here. The acts below are for a capture that is not, or to turn the
          box to an edge.
        </p>
      )}
      {current?.source === "levelled" && !level.armed && (
        <p className="ui-help mt-1 mb-0">
          Levelling sets the tilt and the turn only; the floor's height is
          the Floor row's above; Set from a point if the detected one is
          not the floor.
        </p>
      )}
      {!level.armed ? (
        <>
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            {proposed && (
              <Btn variant={canAdopt ? "primary" : "outline"}
                   disabled={readOnly || !canAdopt}
                   title={canAdopt
                     ? `record ${describeAngles(proposed)}; re-opens this gate onward`
                     : "this is the alignment already recorded"}
                   onClick={adopt}>
                Level as proposed</Btn>
            )}
            <Btn variant="outline" disabled={readOnly}
                 title="click three points on the floor (or any surface parallel to it), then two along a straight edge if you want the box squared to it"
                 onClick={() => onLevelArm(true)}>
              Level by hand</Btn>
            <Btn variant="outline" disabled={readOnly}
                 title="click two points along one straight edge (a wall, a counter's long side); the box is squared to it and the tilt stays"
                 onClick={() => onSquareArm(true)}>
              Square to an edge</Btn>
            {current && (
              <Btn variant="ghost" disabled={readOnly}
                   title="forget the alignment; the scene is read on the file's own axes"
                   onClick={reset}>
                Reset to the file's axes</Btn>
            )}
          </div>
          {proposed && canAdopt && (
            <div className="flex items-center justify-between gap-2 mt-1.5">
              <span className="text-[0.75rem] text-t2">
                Preview: the grid and the floor plane as proposed</span>
              <Switch checked={!!level.preview}
                      label="preview the proposed alignment"
                      onChange={(on) => onLevelPreview(on ? proposed : null)} />
            </div>
          )}
        </>
      ) : (
        <div className="mt-2 border border-line2 rounded p-2 bg-inset"
             role="group" aria-label={squaring ? "square the scene to an edge"
                                              : "level the scene"}>
          {squaring ? (
            <p className="text-[0.8125rem] text-t1 m-0 leading-5">
              {edgeDone
                ? "Done squares the box to that edge; the tilt stays."
                : "Click two points along one straight edge: a wall, the counter's long side."}{" "}
              <span className="text-t3">({pts.length} of 2 · Esc cancels)</span>
            </p>
          ) : (
            /* the two acts in order, each with the colour its points
               wear on the canvas — the floor first, the edge after it and
               optional; the current one weighted, the done one ticked */
            <ol className="list-none m-0 p-0 flex flex-col gap-1"
                aria-label="level the scene: the floor, then an edge">
              <li aria-current={!floorDone ? "step" : undefined}
                  className={cx("flex items-baseline gap-2 text-[0.8125rem] leading-5",
                                floorDone ? "text-t3" : "text-t1 font-medium")}>
                <span className="w-3 shrink-0 inline-flex justify-center" aria-hidden="true">
                  {floorDone ? <span className="text-pass">✓</span>
                    : <span className="inline-block w-2 h-2 rounded-full"
                            style={{ background: LEVEL_COLOURS.floor }} />}
                </span>
                <span>
                  <span style={{ color: floorDone ? undefined : LEVEL_COLOURS.floor }}>
                    Floor</span>
                  {": three points on the floor, far apart, or on any flat surface parallel to it "}
                  <span className="text-t3 font-normal">
                    ({Math.min(pts.length, 3)} of 3)</span>
                </span>
              </li>
              <li aria-current={floorDone && !wallDone ? "step" : undefined}
                  className={cx("flex items-baseline gap-2 text-[0.8125rem] leading-5",
                                wallDone ? "text-t3" : floorDone ? "text-t1 font-medium" : "text-t4")}>
                <span className="w-3 shrink-0 inline-flex justify-center" aria-hidden="true">
                  {wallDone ? <span className="text-pass">✓</span>
                    : <span className="inline-block w-2 h-2 rounded-full"
                            style={floorDone ? { background: LEVEL_COLOURS.edge }
                                             : { border: `1px solid ${LEVEL_COLOURS.edge}` }} />}
                </span>
                <span>
                  <span style={{ color: wallDone || !floorDone ? undefined : LEVEL_COLOURS.edge }}>
                    Edge</span>
                  {", optional: two points along one straight edge, a wall or the counter's long side, to square the box to it "}
                  <span className="text-t3 font-normal">
                    ({Math.max(pts.length - 3, 0)} of 2)</span>
                  {floorDone && !wallDone && (
                    <span className="text-t3 font-normal"> · or Done now</span>)}
                </span>
              </li>
              <li className="text-[0.75rem] text-t3 pl-5">Esc cancels</li>
            </ol>
          )}
          <PickHint />
          <div className="flex items-center gap-2 mt-1.5">
            <span className="ui-help m-0 flex-1">
              {squaring
                ? "writes the turn and re-opens this gate onward"
                : floorDone
                  ? "writes the alignment and re-opens this gate onward"
                  : "the three points define the floor's plane"}
            </span>
            <Btn variant="ghost" disabled={!pts.length || busy}
                 title="drop the last point clicked (Backspace)"
                 onClick={onLevelUndo}>Undo last</Btn>
            <Btn variant="ghost"
                 onClick={() => (squaring ? onSquareArm : onLevelArm)(false)}>
              Cancel</Btn>
            <Btn variant="primary"
                 disabled={readOnly || busy || (squaring ? !edgeDone : !floorDone)}
                 onClick={done}>Done</Btn>
          </div>
        </div>
      )}
      <Details>
        The proposal reads the tilt from the scene itself (the floor and
        the walls of a room, the ground of a site, the surface a subject
        stands on) and says so on the Check card; nothing is applied until
        you level it. Levelled, the boxes, the cameras and the objects all
        follow a floor that is flat, and the file itself is never changed.
      </Details>
    </div>
  );
}

/** The units the operator may type a length in; converted to metres at the
 *  write, the profile records both. The last one used is remembered per
 *  browser. */
const RULER_UNITS = ["m", "cm", "ft", "in"] as const;
const TO_M: Record<string, number> = { m: 1, cm: 0.01, ft: 0.3048, in: 0.0254 };
const RULER_UNIT_KEY = "carveout_ruler_unit";

/** Scale: what a scene unit is, and the act that records it.
 *  Like the up axis it is a CONVENTION the file does not record, and like
 *  the up axis it is the operator's answer — never guessed from the scene.
 *  Not recorded: "no scale yet — measure one thing you know", and Propose
 *  refuses until then. Measured: "1 unit ≈ 0.916 m" and "you measured a door:
 *  2 m", with Measure again. Recorded (Metric, or a typed number): the
 *  number, exact. The ruler works on any capture the canvas can show;
 *  Metric and the typed number stay beside it. Nothing is rescaled; the
 *  number records what the units mean, and changing it re-opens this
 *  gate: the lengths follow it. */
function ScaleBlock({ readOnly, factor, measured, scale, ruler, onRulerArm,
                      write }: {
  readOnly: boolean; factor: number | null;
  measured: CalibrationResp["scene"]["measured"];
  scale: ScaleRead | null;
  ruler: { armed: boolean; draft: number[][] | null };
  onRulerArm: (on: boolean) => void;
  write: (u: unknown) => void;
}) {
  const [unit, setUnit] = useState<string>(() => {
    try { return localStorage.getItem(RULER_UNIT_KEY) || "m"; }
    catch { return "m"; }
  });
  const [typed, setTyped] = useState<number | null>(null);
  const [label, setLabel] = useState("");
  const pts = ruler.draft ?? [];
  const units = pts.length === 2
    ? Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1],
                 pts[0][2] - pts[1][2]) : null;
  const lengthM = typed != null && typed > 0 ? typed * TO_M[unit] : null;
  const chooseUnit = (u: string) => {
    setUnit(u);
    try { localStorage.setItem(RULER_UNIT_KEY, u); } catch { /* per-browser only */ }
  };
  const record = () => {
    if (units == null || lengthM == null || !(units > 0)) return;
    write({ scene: {
      scale_m_per_unit: lengthM / units,
      measured: { length_m: lengthM, units, label: label.trim(),
                  unit, typed, points: pts } } });
    onRulerArm(false);
    setTyped(null);
    setLabel("");
  };
  // A declared or typed factor replaces a measurement: the record clears
  // with it, so the panel never says "≈" for a number the operator gave.
  const declare = (v: number) =>
    write({ scene: { scale_m_per_unit: v, measured: null } });
  const none = factor == null;
  const approx = measured ? "≈" : "=";
  return (
    <div className="mt-3 pt-2 border-t border-line">
      <div className="flex items-center gap-2">
        <span className="ui-label flex-1">scale</span>
        <span className={cx("font-mono text-[0.75rem]",
                            none ? "text-warn" : "text-t3")}>
          {none ? "no scale yet"
                : `1 unit ${approx} ${Number(factor!.toPrecision(4))} m`}
        </span>
      </div>
      {measured ? (
        <p className="text-[0.8125rem] text-t2 mt-1 mb-0 leading-5">
          you measured{measured.label ? ` ${measured.label}` : ""}:{" "}
          <b>{Number(measured.typed.toPrecision(4))} {measured.unit}</b>
          {measured.unit !== "m" && ` (${Number(measured.length_m.toPrecision(4))} m)`}
          {" "}over {fmtUnits(measured.units)} units on the canvas
        </p>
      ) : none ? (
        <p className="text-[0.8125rem] text-t2 mt-1 mb-0 leading-5">
          measure one thing you know: click two points on the scene
          (a door, a floor tile, a monument, a wall you paced) and type
          its length.
        </p>
      ) : (
        <p className="ui-help mt-1 mb-0">
          recorded: declared metric, or the number you typed.
        </p>
      )}
      {scale && scale.extent_units && (
        <p className="font-mono text-[0.75rem] text-t3 mt-1 mb-0 leading-4">
          measures {scale.extent_units.map(fmtUnits).join(" × ")}
          {" "}units
          {scale.extent_m && (
            <> {approx} {scale.extent_m.map((v) => v.toFixed(2)).join(" × ")} m</>)}
        </p>
      )}
      {scale && scale.plausible === false && (
        <p className="text-[0.75rem] text-warn mt-1 mb-0 leading-4">
          {scale.note}</p>)}

      {/* The ruler row. Armed: the canvas takes the next two clicks; the
        * distance appears here with the length field. */}
      {!ruler.armed ? (
        <div className="grid grid-cols-2 gap-2 mt-2">
          <Btn variant={none ? "primary" : "outline"} disabled={readOnly}
               title="click two points on the scene, then type their real length"
               onClick={() => onRulerArm(true)}>
            {measured ? "Measure again" : "Measure"}</Btn>
          <Btn disabled={readOnly} title="the scene is already in metres"
               onClick={() => declare(1)}>
            Metric (1.0)</Btn>
        </div>
      ) : (
        <div className="mt-2 border border-line2 rounded p-2 bg-inset"
             role="group" aria-label="ruler">
          {units == null ? (
            <>
              <p className="text-[0.8125rem] text-t1 m-0 leading-5">
                {pts.length === 0
                  ? "Click the first point on the scene."
                  : "Now the second point."}{" "}
                <span className="text-t3">Esc cancels.</span>
              </p>
              <PickHint />
            </>
          ) : (
            <>
              <p className="text-[0.8125rem] text-t1 m-0 leading-5">
                <b className="font-mono">{fmtUnits(units)} units</b>: what
                did you measure?{" "}
                <span className="text-t3">(a third click starts over)</span>
              </p>
              <div className="flex items-center gap-2 mt-1.5">
                <NumInput className="w-20" value={typed} aria-label="length"
                          placeholder="length" onCommit={setTyped} />
                <Segmented value={unit} options={RULER_UNITS} label="unit"
                           onChange={chooseUnit} />
                <input className={cx("ui-input flex-1 min-w-0")} value={label}
                       aria-label="what you measured (optional)"
                       placeholder="what you measured (optional), e.g. a door"
                       onChange={(e) => setLabel(e.target.value)} />
              </div>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="ui-help m-0 flex-1">
                  {lengthM != null
                    ? `1 unit ≈ ${Number((lengthM / units).toPrecision(4))} m; writes the scale and re-opens this gate onward`
                    : "type the real length of what you clicked"}
                </span>
                <Btn variant="ghost" onClick={() => onRulerArm(false)}>Cancel</Btn>
                <Btn variant="primary" disabled={readOnly || lengthM == null}
                     onClick={record}>Record</Btn>
              </div>
            </>
          )}
        </div>
      )}
      <LabeledNum label="metres per unit"
                  value={factor}
                  disabled={readOnly}
                  onCommit={(v) => v > 0 && declare(v)}
                  hint={
                    <span className="text-[0.75rem] text-t4">
                      if you know the number</span>} />
      <Details>
        Three ways to record it. <b>Measure</b>: two clicks on the scene on
        something whose real length you know, in any unit; a vast exterior
        needs one known object, a room one door, a scene with nothing
        familiar can be paced. <b>Metric</b>: the scene is already in metres.
        <b> Metres per unit</b>: a number you already know. Nothing is
        moved; the number records what the units mean, and changing it
        re-opens this gate: every length follows it. Propose refuses until
        one of the three is done; the volume's grids are sized in metres.
      </Details>
    </div>
  );
}

/** Slider + numeric pair: the slider live-drafts
 * (gizmo follows) and commits one PUT on release; the numeric input is
 * the precision path and accepts values outside the slider range. */
function FieldRow({ label, value, min, max, onDraft, onCommit, disabled,
                    accent }: {
  label: string; value: number; min: number; max: number;
  onDraft: (v: number) => void;
  /** v given = numeric commit; undefined = commit the pending draft */
  onCommit: (v?: number) => void;
  disabled?: boolean; accent?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 min-h-[28px]"
         onClick={(e) => e.stopPropagation()}>
      <span className={cx("ui-label w-16 shrink-0",
                          accent ? "text-t1 font-medium" : "text-t3")}>
        {label}</span>
      <Slider min={min} max={max} step={(max - min) / 400} value={value}
              label={label}
              disabled={disabled} className="flex-1 disabled:opacity-40"
              onChange={onDraft} onCommit={() => onCommit()} />
      <NumInput className="w-[68px] shrink-0"
                value={value} fmt={fmtUnits}
                onCommit={onCommit} disabled={disabled} />
    </div>
  );
}

/** Side-view density silhouette with box height spans (the second
 * height surface). Histogram from the density sidecar. Drawn in a fixed
 * coordinate space and scaled to the panel's width by the viewBox. */
function ElevationStrip({ d, boxes, up, sign }: {
  d: DensitySidecar; boxes: VolumeBox[]; up: number; sign: number;
}) {
  const W = 380, H = 90;
  // The sidecar's histogram is in raw world units; the strip reads SIGNED
  // (up = positive, left to right) like every height in this panel.
  const edges = d.height_hist.edges.map((e) => e * sign);
  const { counts } = d.height_hist;
  const maxC = Math.max(...counts, 1);
  const lo = Math.min(edges[0], edges[edges.length - 1]);
  const hi = Math.max(edges[0], edges[edges.length - 1]);
  const x = (h: number) => ((h - lo) / (hi - lo)) * W;
  const pts = counts.map((c, i) => {
    const cx = x((edges[i] + edges[i + 1]) / 2);
    const cy = H - 8 - (c / maxC) * (H - 20);
    return `${cx.toFixed(1)},${cy.toFixed(1)}`;
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full h-auto"
         role="img" aria-label="elevation density silhouette">
      {/* density in the volume's own colour (scene data), text on the
          chrome tokens */}
      <polyline points={`0,${H - 8} ${pts.join(" ")} ${W},${H - 8}`}
                fill="rgba(102,197,212,0.18)" stroke="#66c5d4"
                strokeWidth="1" />
      {boxes.map((b, i) => {
        const s0 = Math.min(b.min[up] * sign, b.max[up] * sign);
        const s1 = Math.max(b.min[up] * sign, b.max[up] * sign);
        return (
        <g key={i}>
          <rect x={x(s0)} y={4}
                width={Math.max(x(s1) - x(s0), 2)}
                height={H - 12}
                className="fill-t1/[.06] stroke-t3"
                strokeDasharray="3 2" strokeWidth="1" />
          <text x={x(s0) + 3} y={12}
                className="fill-t2" fontSize="11" fontFamily="IBM Plex Mono">
            {b.name}</text>
        </g>
        );
      })}
      <text x={0} y={H} className="fill-t4" fontSize="11"
            fontFamily="IBM Plex Mono">{lo.toFixed(1)}</text>
      <text x={W - 24} y={H} className="fill-t4" fontSize="11"
            fontFamily="IBM Plex Mono">{hi.toFixed(1)}</text>
    </svg>
  );
}

/** Native top-down density plan (canvas from sidecar) with box outlines —
 * the plan-view precision surface. Fixed coordinate space, scaled to the
 * panel's width (the overlay's viewBox matches the canvas). */
function DensityPlan({ d, boxes, selected, onSelect }: {
  d: DensitySidecar; boxes: VolumeBox[]; selected: number;
  onSelect: (i: number) => void;
}) {
  const [n0, n1] = d.shape;
  const W = 380;
  const scale = W / (n1 * d.voxel);
  const H = Math.round(n0 * d.voxel * scale);
  const [a0, a1] = d.ground_axes;
  const canvasRef = (el: HTMLCanvasElement | null) => {
    if (!el) return;
    const ctx = el.getContext("2d")!;
    const img = ctx.createImageData(n1, n0);
    let max = 0;
    for (const row of d.density)
      for (const v of row) max = Math.max(max, Math.log1p(v));
    for (let i = 0; i < n0; i++)
      for (let j = 0; j < n1; j++) {
        const t = Math.log1p(d.density[i][j]) / (max || 1);
        const o = ((n0 - 1 - i) * n1 + j) * 4;   // origin="lower" parity
        img.data[o] = 20 + 60 * t;
        img.data[o + 1] = 30 + 170 * t;
        img.data[o + 2] = 40 + 160 * t;
        img.data[o + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
  };
  const toPx = (b: VolumeBox) => {
    const x = (b.min[a1] - d.origin[1]) / d.voxel / n1 * W;
    const w = (b.max[a1] - b.min[a1]) / d.voxel / n1 * W;
    const y = H - ((b.max[a0] - d.origin[0]) / d.voxel / n0 * H);
    const h = (b.max[a0] - b.min[a0]) / d.voxel / n0 * H;
    return { x, y, w, h };
  };
  return (
    <div className="relative w-full" style={{ aspectRatio: `${W} / ${H}` }}>
      <canvas ref={canvasRef} width={n1} height={n0}
              style={{ imageRendering: "pixelated" }}
              className="block w-full h-full border border-line rounded" />
      <svg className="absolute inset-0 w-full h-full"
           viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {boxes.map((b, i) => {
          const r = toPx(b);
          return (
            <rect key={i} x={r.x} y={r.y} width={r.w} height={r.h}
                  fill="none"
                  stroke={i === selected ? "#fff" : "#66c5d4"}
                  strokeWidth={i === selected ? 2 : 1}
                  className="cursor-pointer"
                  style={{ pointerEvents: "all" }}
                  onClick={() => onSelect(i)} />
          );
        })}
      </svg>
    </div>
  );
}
