// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The workbench canvas: the EXISTING viewer's splat stage + label engine
// (viewer/lib — integrated, not rebuilt) plus the pipeline's spatial
// overlays (frustums, volume gizmo, crop markers). Full-bleed and always
// mounted; gate chrome floats over it (design center).

import { forwardRef, useEffect, useImperativeHandle, useRef,
         useState } from "react";
import * as THREE from "three";
import { SplatStage, isTypingTarget } from "@viewer/stage.js";
import { LabelEngine } from "@viewer/labels.js";
import { createHighlighter } from "@viewer/highlight.js";
import { createFocusVolume, VOLUME_PREVIEW_FOCUS } from "@viewer/focus.js";
import { captureView } from "@viewer/capture.js";
import { applySetting, resetFocus } from "@viewer/apply.js";
import { DEFAULT_SETTINGS, mergeSettings } from "@viewer/settings.js";
import { runUrl } from "../api";
import type { FrameEntry, VolumeBox } from "../types";
import { COL, cssHex, CameraGizmo, CropMarkers, FloorPlane, FocusGizmo, FrustumSet,
         OriginGrid, Ruler,
         Manipulator, VolumeGizmo,
         type CameraMode, type FocusMode, type GizmoMode, type Pick,
         type VolumeFrame } from "./overlays";
import ViewGizmo from "./ViewGizmo";

/** Pointer position → NDC over the holder (the camera's aspect is the
 *  holder's, so this is the ray the render used). */
function ndcOf(el: HTMLElement, x: number, y: number): THREE.Vector2 {
  const r = el.getBoundingClientRect();
  return new THREE.Vector2(((x - r.left) / r.width) * 2 - 1,
                           -((y - r.top) / r.height) * 2 + 1);
}

/** The pick as the rule read it, on the drive handle (`__carveoutCanvas
 *  .lastPick` for a click, `.lastHover` for the ring: range, hits, solid,
 *  ms) — the pick constants are calibrated on a test scene from the
 *  console. */
function recordPick(key: "lastPick" | "lastHover", pk: Pick | null, t0: number) {
  (window as unknown as { __carveoutCanvas: Record<string, unknown> })
    .__carveoutCanvas[key] = pk && { ...pk, ms: Math.round(performance.now() - t0) };
}

/** How long the pointer rests before the hover ring's pick is cast. */
const HOVER_SETTLE_MS = 220;
import { placeCamera, type ViewAxis } from "./viewAxes";
import { boundsOfBoxes, floorOfSample, framingPose, standingPose } from "./startPose";
import { alignmentKey, alignmentQuaternion, alignmentRows, robustBounds,
         rotateSample, type AlignmentBlock } from "./alignment";

export interface CanvasHandle {
  stage: any;
  /** apply ONE viewer setting through the shared effects table
   *  (viewer/lib/apply.js): the value is written into settings and the
   *  right applier runs — the panel never re-encodes which applier
   *  follows which mutation (that is the table's job, for both apps) */
  apply(path: string, value?: unknown): void;
  /** reset the focus volume to the detection volume (shared helper) */
  resetFocus(): void;
  /** manual capture: the camera pose in the file frame */
  /** label omitted on purpose: the server names a capture from the saved
   *  views, the only count that moves before a re-render */
  capturePose(label?: string): Record<string, unknown>;
  flyToFrame(fr: FrameEntry): void;
  flyToInstance(idx: number): void;
  /** put the selected instance down (the card's close, Esc) */
  deselectInstance(): void;
  settings: any;
  /** focus-volume controller (viewer/lib/focus.js) — null until the
   * volume fetch lands; {active:false} when the scene has no boxes */
  focus: any;
  /** label engine — null until interactions load (pre-export scenes) */
  engine: any;
  /** which manipulator the selected volume box carries: move its centre, or
   *  change its size about that centre. Never rotate — boxes are AABBs. */
  setGizmoMode(mode: GizmoMode): void;
  /** the FOCUS volume's manipulator — translate/rotate/scale, and rotate is
   *  offered here because this turns a display effect only */
  setFocusGizmoMode(mode: FocusMode): void;
  /** the selected camera's manipulator — move it, or turn it. No scale: a
   *  camera has no size, and its analogue (fov) is a number, not a handle. */
  setCameraGizmoMode(mode: CameraMode): void;
}

/** Ground footprint of the volume boxes in a frame: centre and half-extent
 * (a generous default when there are no boxes yet). Drives the floor
 * plane's size and the origin grid's fade. */
function footprint(boxes: VolumeBox[] | null, f: VolumeFrame):
    { center: number[]; half: number } {
  const center = [0, 0, 0];
  if (!boxes || !boxes.length) return { center, half: 8 };
  const { a0, a1 } = f;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const b of boxes)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], b.min[a]);
      hi[a] = Math.max(hi[a], b.max[a]);
    }
  center[a0] = (lo[a0] + hi[a0]) / 2;
  center[a1] = (lo[a1] + hi[a1]) / 2;
  return { center,
           half: Math.max(hi[a0] - lo[a0], hi[a1] - lo[a1], 2) * 2 };
}

/** Where the scene file is: on its way (bytes of the file's length, when
 *  the server named it), being decoded, on the canvas, or not coming. */
export interface SceneLoad {
  state: "loading" | "preparing" | "ready" | "failed";
  loaded: number; total: number;
  error?: string;
}

interface Props {
  scene: string;
  /** the scene file's progress, for the header's chip */
  onSceneLoad?: (load: SceneLoad) => void;
  epoch: number;              // journal moves -> spatial data refetch
  layers: { volume: boolean; cameras: boolean; instances: boolean;
            crops: boolean; grid: boolean };
  volumeBoxes: VolumeBox[] | null;
  /** saved volume boxes for the focus display effect — NOT layer-gated:
   *  hiding the volume gizmo must not switch the effect off */
  focusBoxes: VolumeBox[] | null;
  /** the volume gate's preview: the effect follows the boxes being EDITED
   *  (the draft, every drag frame) with fixed parameters, and nothing is
   *  written to the settings */
  focusPreview?: boolean;
  focusDraftBoxes?: VolumeBox[] | null;
  volumeFrame: VolumeFrame | null;
  /** the scene alignment: the profile's block, null = the file's
   *  axes. The splats and every file-frame overlay sit under a group that
   *  carries it; the boxes, the floor and the grid draw in the scene
   *  frame above it. `upAxis` names the axis the block's yaw turns about. */
  alignment?: AlignmentBlock | null;
  upAxis: number;
  /** a proposed alignment to PREVIEW: the floor plane and the origin grid
   *  are drawn in that frame instead — display only, nothing saved */
  previewAlignment?: AlignmentBlock | null;
  /** the level act: while armed the next clicks on the splat are
   *  its points (three on the floor, then up to two along a wall), drawn
   *  by the ruler in its level mode; file frame, like the ruler's */
  levelArmed?: boolean;
  /** which act the level points belong to: the level act (up to five)
   *  or the square act (two along an edge) */
  levelMode?: "level" | "square";
  levelPoints?: number[][] | null;
  onLevelPick?: (point: number[]) => void;
  /** the floor act: while armed the next click on the splat is the
   *  floor's point (file frame, like the ruler's); one click completes it */
  floorArmed?: boolean;
  onFloorPick?: (point: number[]) => void;
  /** the filming answer and the eye height (m) the start pose follows when
   *  no pose is saved: standing in a walkable scene, framing an object */
  pathMode?: string | null;
  eyeHeight?: number | null;
  /** the floor rule's band and bin (calibration, read-only): the start
   *  pose estimates a floor from the splats by the pipeline's own rule
   *  until a proposal has detected one */
  floorBandFrac?: number | null;
  histBinM?: number | null;
  volumeEditable: boolean;
  volumeSelected: number;
  onVolumeChange?: (boxes: VolumeBox[]) => void;
  /** live during a handle drag — the panel's numerics follow the hand, the
   *  same draft/commit split the sliders use in the other direction */
  onVolumeDraft?: (boxes: VolumeBox[]) => void;
  /** the focus manipulator moved: the settings panel re-reads its sliders */
  onFocusChange?: () => void;
  /** 3D-only mode: the canvas carries no widgets at all in it */
  zen?: boolean;
  /** the focus volume can be picked — true only where its controls live */
  focusEditable?: boolean;
  focusSelected?: boolean;
  onFocusSelect?: (on: boolean) => void;
  /** bumped whenever the focus volume changes, from either surface — the
   *  handles are seeded from its values and must follow a slider too */
  focusEpoch?: number;
  onVolumeSelect?: (i: number) => void;
  floorSigned: number | null;   // detected/override floor, SIGNED convention
  /** the ruler (the scale act): while armed the next click on the splat
   *  is a measurement point, not a pick; `rulerPoints` is the line to draw
   *  (the draft while measuring, the recorded measurement otherwise) */
  rulerArmed?: boolean;
  rulerPoints?: number[][] | null;
  onRulerPick?: (point: number[]) => void;
  /** frame of the origin grid: the measured one once a proposal exists,
   *  else the profile's up-axis convention — never null, so the grid is
   *  drawn from the moment the scene opens */
  gridFrame: VolumeFrame;
  frames: FrameEntry[] | null;
  /** frames whose manual view was deleted since the render: struck
   *  in 3D until the re-render drops them */
  deletedFrames?: number[];
  intrinsics: { width: number; height: number; fx: number;
                fy: number } | null;
  /** camera frustum display size, display metres */
  frustumScale: number;
  /** metres per scene unit when recorded, null when not known — either way
   *  ONE uniform scale on the root puts the scene in display metres
   *  (displayScaleOf) */
  factor: number | null;
  /** the scene's robust extents in scene units (the volume gate's read),
   *  for the nominal display scale while the factor is not known; null
   *  before any proposal has measured the scene */
  extent: number[] | null;
  selectedFrame: number | null;
  onSelectFrame?: (idx: number | null) => void;
  /** the selected frame's manipulator is live only under the RENDER gate,
   *  and only for a MANUAL view — an auto pose would be regenerated away */
  cameraEditable: boolean;
  /** a camera pose was adjusted and released: ply-space position +
   *  three.js quaternion (exactly what manual_views.json stores), plus the
   *  equivalent OpenCV c2w so the caller can redraw the frustum at the new
   *  pose before the re-render lands */
  onFramePose?: (v: { position: number[]; quaternion: number[];
                      c2w: number[][] }) => void;
  crops: { frame: FrameEntry; box_xyxy: number[] }[];
  /** per-overlay gate staleness: each gate's overlay grays out when its
   *  approval no longer covers what is on screen — the cascade banner's
   *  "overlays shown are from the previous approval", made visible */
  stale: { volume: boolean; cameras: boolean; crops: boolean };
  onInstanceSelect?: (idx: number) => void;
}

/** ONE uniform scale on stage.root puts the scene in display metres: the
 *  recorded factor when the scene's scale is known (world space IS metres);
 *  otherwise the viewer's nominal extent (viewer_settings
 *  display.nominalExtent, ~5 m) over the scene's largest robust extent, so
 *  an unknown-scale scene is displayed about room-sized; 1 before anything
 *  has measured it. Every speed, clip plane, grid spacing, label distance
 *  and frustum size is then a display-metre figure, and nothing else in the
 *  viewer knows the scene's unit. */
export function displayScaleOf(factor: number | null, extent: number[] | null,
                               settings: any): number {
  if (factor != null && factor > 0) return factor;
  const L = extent?.length ? Math.max(...extent) : 0;
  const nominal = Number(settings?.display?.nominalExtent) || 5;
  return L > 0 ? nominal / L : 1;
}

/**
 * What a view should frame, in WORLD space.
 *
 * The volume boxes when the scene has them — they are the region of interest,
 * which is what someone asking for "top" wants centred. Falling back to the
 * splats' own bounds would frame the whole capture including everything the
 * volume exists to exclude. With no volume yet (the canvas mounts before gate
 * 1 has run) the origin and a room-sized default are the honest answer: the
 * placement maths clamps the span anyway.
 *
 * Boxes are in the SCENE frame under `root`, which carries the flip; the
 * splats' bounds are in the FILE frame under the file group, which carries
 * the alignment — so every corner goes through its own group's
 * localToWorld. Taking min/max in local space and transforming only the
 * centre would be wrong the moment either rotation is on.
 */
function worldBounds(s: { stage: any; gizmo: VolumeGizmo }) {
  let boxes: { min: number[]; max: number[] }[] = s.gizmo.boxes;
  let group: THREE.Object3D = s.stage.root;
  if (!boxes.length) {
    // no volume yet: the splats' own robust bounds, when they are in
    const sb = s.stage.splatBounds?.();
    if (sb) { boxes = [{ min: sb.lo, max: sb.hi }]; group = s.stage.fileGroup; }
  }
  if (!boxes.length) {
    return { center: [0, 0, 0] as [number, number, number],
             size: [6, 3, 6] as [number, number, number] };
  }
  const bb = new THREE.Box3();
  const v = new THREE.Vector3();
  group.updateMatrixWorld(true);
  for (const b of boxes) {
    for (let c = 0; c < 8; c++) {
      v.set(c & 1 ? b.max[0] : b.min[0],
            c & 2 ? b.max[1] : b.min[1],
            c & 4 ? b.max[2] : b.min[2]);
      bb.expandByPoint(group.localToWorld(v.clone()));
    }
  }
  const ctr = bb.getCenter(new THREE.Vector3());
  const size = bb.getSize(new THREE.Vector3());
  return { center: [ctr.x, ctr.y, ctr.z] as [number, number, number],
           size: [size.x, size.y, size.z] as [number, number, number] };
}

/** FNV-1a over a string: a cheap content key for "did this artifact
 *  change", where the string itself is too long to keep and compare. */
function hashStr(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0") + ":" + s.length.toString(16);
}

export default forwardRef<CanvasHandle, Props>(function SplatCanvas(
    props, ref) {
  // Which face the dial lights up. Set when a face is chosen and cleared the
  // moment the camera is driven by hand — the dial claims "this is where you
  // asked to stand", never "this is where you are", which it cannot know once
  // free-flight has moved the camera off the axis.
  const [viewAxis, setViewAxis] = useState<ViewAxis | null>(null);
  const holder = useRef<HTMLDivElement>(null);
  const labelLayer = useRef<HTMLDivElement>(null);
  const tagLayer = useRef<HTMLDivElement>(null);
  // Flipped once the stage exists; a dependency of every effect that drives
  // it, so they re-run when it lands rather than silently doing nothing.
  const [stageReady, setStageReady] = useState(false);
  // the loaded splats' own robust extents: the display scale's reference
  // until a proposal has measured the scene (stage.splatExtent)
  const [splatExtent, setSplatExtent] = useState<number[] | null>(null);
  const S = useRef<{ stage: any; settings: any; frustums: FrustumSet;
                     cameraGizmo: CameraGizmo;
                     focusGizmo?: FocusGizmo;
                     gizmo: VolumeGizmo; crops: CropMarkers;
                     floor: FloorPlane; grid: OriginGrid; ruler: Ruler;
                     /** the floor plane's and the grid's parent: the scene
                      *  frame, or a proposed alignment's frame under preview */
                     frameGroup: THREE.Group;
                     /** the splat sample carried into the scene frame, keyed
                      *  by the alignment it was rotated by */
                     sceneSample: { key: string; sample: any; bounds: any } | null;
                     engine: any | null; focus: any | null;
                     ixSig: string;
                     nameSig: string;    // the names alone (relabel)
                     /** the floor the start pose reads off the splats
                      *  before any proposal (scene frame, signed) */
                     floorGuess?: () => number | null }>();
  const propsRef = useRef(props);
  propsRef.current = props;

  // -- one-time stage ---------------------------------------------------------
  useEffect(() => {
    let dead = false;
    let ro: ResizeObserver | null = null;
    // Assigned inside the async body, torn down by the OUTER cleanup: the
    // inner IIFE's return value is a promise, not a cleanup, so anything
    // registered in there has to hand its remover out to be removed at all.
    let detachViewDial: (() => void) | null = null;
    let detachResize: (() => void) | null = null;
    const tellLoad = (load: SceneLoad) => {
      if (!dead) propsRef.current.onSceneLoad?.(load);
    };
    (async () => {
      // Both startup reads in ONE round-trip: everything below waits on the
      // stage, so time spent here is time the whole canvas is not ready.
      // scene_source has no default — the container (.ply / .sog) decides
      // which decoder Spark gets, and a wrong guess would load garbage, so a
      // failed fetch must surface as the canvas not coming up — in the
      // header, with the server's own words: a refused or unknown scene
      // used to load "/undefined" and show Spark's message, and a network
      // failure or a browser without WebGL showed a black canvas and nothing.
      const [settingsRes, srcRes] = await Promise.all([
        fetch(runUrl(props.scene, "/viewer_settings")).catch(() => null),
        fetch(runUrl(props.scene, "/scene_source")),
      ]);
      let settings: any = structuredClone(DEFAULT_SETTINGS);
      try {
        if (settingsRes?.ok)
          settings = mergeSettings(DEFAULT_SETTINGS, await settingsRes.json());
      } catch { /* defaults */ }
      if (!srcRes.ok) {
        let why = `${srcRes.status} ${srcRes.statusText}`;
        try { why = (await srcRes.json()).error ?? why; } catch { /* as is */ }
        tellLoad({ state: "failed", loaded: 0, total: 0, error: why });
        return;
      }
      const src = await srcRes.json();
      if (dead || !holder.current) return;
      tellLoad({ state: "loading", loaded: 0, total: 0 });
      const stage = new SplatStage({
        container: holder.current,
        sceneUrl: runUrl(props.scene, `/${src.file}`),
        sceneFormat: src.format,
        settings,
        onLoadProgress: ({ loaded, total }: { loaded: number; total: number }) =>
          tellLoad({ state: total > 0 && loaded >= total ? "preparing" : "loading",
                     loaded, total }),
      });
      stage.applyFlip();
      // The start pose when none is saved: from the filming answer,
      // the frame, the floor and the volume (or the splats' bounds before
      // one exists). Read through propsRef and re-applied by the effect
      // below whenever those arrive, as long as the camera is still at
      // the start pose; also what "Go to start pose" returns to.
      // The splats' sample in the SCENE frame (through the alignment),
      // with its robust bounds: what the start pose and its floor estimate
      // read before a proposal. Rotated once per alignment, then cached.
      const sceneSample = () => {
        const raw = stage.splatSample?.();
        if (!raw) return null;
        const p = propsRef.current;
        const key = alignmentKey(p.alignment) + `|${p.upAxis}`;
        const st = S.current;
        if (st?.sceneSample && st.sceneSample.key === key) return st.sceneSample;
        const sample = rotateSample(raw, p.alignment, p.upAxis);
        const entry = { key, sample, bounds: robustBounds(sample) };
        if (st) st.sceneSample = entry;
        return entry;
      };
      stage.defaultPose = () => {
        const p = propsRef.current;
        const ss = sceneSample();
        const bounds = boundsOfBoxes(p.focusBoxes) ?? ss?.bounds ?? null;
        if (!bounds) return null;
        stage.root.updateMatrixWorld(true);
        const scale = stage.displayScale || 1;
        const eyeUnits = (p.eyeHeight ?? 1.6) / scale;
        // before a proposal: the profile's frame (the grid's), and a floor
        // read off the splats the way the proposal will read it — in the
        // scene frame, where the floor is level once the scene is
        const frame = p.volumeFrame ?? p.gridFrame;
        const floorSigned = p.floorSigned ?? floorOfSample({
          sample: ss?.sample ?? null, frame, bounds,
          bandFrac: p.floorBandFrac ?? 0.4,
          binUnits: (p.histBinM ?? 0.05) / scale });
        const st = standingPose({ pathMode: p.pathMode, frame,
                                  floorSigned, bounds, eyeUnits });
        if (st) {
          return { to: stage.root.localToWorld(new THREE.Vector3(...st.position)),
                   target: stage.root.localToWorld(new THREE.Vector3(...st.target)) };
        }
        // an object: framed from a diagonal a little above it, as close
        // as the whole of it fits the view (not the dial's 45° overview)
        const s = S.current;
        const wb = s ? worldBounds(s) : { center: [0, 0, 0], size: [6, 3, 6] };
        const { position, target } = framingPose({
          center: wb.center, size: wb.size, fovDeg: stage.camera.fov,
          aspect: stage.camera.aspect || 1 });
        return { to: new THREE.Vector3(...position),
                 target: new THREE.Vector3(...target) };
      };
      stage.setDisplayScale(displayScaleOf(
        propsRef.current.factor, propsRef.current.extent, settings));
      stage.applyStartPose();
      // Headless-drive handle (the perf counters' precedent): a verification
      // script reads speeds, clip planes and overlay state from here.
      (window as unknown as { __carveoutStage: unknown }).__carveoutStage = stage;
      // ...and the overlay groups, so a drive can check which frame each
      // overlay is parented in (file frame vs scene frame).
      const expose = (o: Record<string, unknown>) => {
        (window as unknown as { __carveoutCanvas: Record<string, unknown> })
          .__carveoutCanvas = o;
      };
      // Once the splats are in, measure them: with no sidecar extent yet the
      // display scale falls back to this, so a scene of any unit is
      // room-sized on screen from the first frame, not from the first
      // proposal (the effect below re-applies the scale when it lands).
      const splats = stage.splats as { initialized?: Promise<unknown> };
      (splats.initialized ?? Promise.resolve()).then(() => {
        if (dead) return;
        setSplatExtent(stage.splatExtent());
        const { loaded, total } = stage.loadProgress ?? { loaded: 0, total: 0 };
        tellLoad({ state: "ready", loaded, total });
      }).catch((e: unknown) => {
        // the fetch failed, or was aborted (a headless drive): said in the
        // header, never swallowed
        const { loaded, total } = stage.loadProgress ?? { loaded: 0, total: 0 };
        tellLoad({ state: "failed", loaded, total,
                   error: e instanceof Error ? e.message : String(e) });
      });
      // Two frames. FILE frame, under the file group that carries the
      // alignment: the frustums and the camera manipulator (cameras.json,
      // manual_views.json), the crop markers, the ruler and its points,
      // the label engine's boxes (interactions.json). SCENE frame, under
      // the root: the volume boxes, the focus effect, the floor plane and
      // the origin grid — the last two under `frameGroup`, which is the
      // scene frame until a proposed alignment is previewed through it.
      const frustums = new FrustumSet(stage.fileGroup);
      const gizmo = new VolumeGizmo(stage.root, stage.camera,
                                    stage.renderer.domElement, stage.scene);
      const cropM = new CropMarkers(stage.fileGroup);
      const frameGroup = new THREE.Group();
      stage.root.add(frameGroup);
      const floorPlane = new FloorPlane(frameGroup);
      const ruler = new Ruler(stage.fileGroup, holder.current);
      stage.frameHooks.push(() => ruler.update(stage.camera, stage.viewW,
                                               stage.viewH));
      const originGrid = new OriginGrid(frameGroup);
      gizmo.onChange = (b) => propsRef.current.onVolumeChange?.(b);
      gizmo.onDraft = (b) => propsRef.current.onVolumeDraft?.(b);
      // A handle drag must not also fly the camera. SparkControls listens on
      // the same canvas, so this is suppressed at the stage rather than by
      // racing listener order.
      gizmo.onDragging = (on) => { stage.controlsEnabled = !on; };

      // The FOCUS manipulator. Detached by default; the settings panel turns
      // it on. It and the detection gizmo are never both attached — that one
      // is editable only while the VOL gate is the active panel.
      const fgz = new FocusGizmo(stage.root, stage.camera,
                                 stage.renderer.domElement, stage.scene);
      fgz.onDragging = (on) => { stage.controlsEnabled = !on; };
      fgz.onChange = (v) => {
        const st = S.current;
        if (!st?.focus?.active) return;
        st.settings.focus.offset[0] = v.offset[0];
        st.settings.focus.offset[1] = v.offset[1];
        st.settings.focus.offset[2] = v.offset[2];
        st.settings.focus.scale[0] = v.scale[0];
        st.settings.focus.scale[1] = v.scale[1];
        st.settings.focus.scale[2] = v.scale[2];
        st.settings.focus.rotation[0] = v.rotation[0];
        st.settings.focus.rotation[1] = v.rotation[1];
        st.settings.focus.rotation[2] = v.rotation[2];
        // No flash: the wires are already visible whenever the handles are.
        st.focus.applyFocus(false);
        propsRef.current.onFocusChange?.();
      };
      // The CAMERA manipulator: one frustum's pose, editable only under the
      // RENDER gate. Mutually exclusive with the detection gizmo by
      // construction — that one is editable only under VOL.
      const cgz = new CameraGizmo(stage.fileGroup, stage.camera,
                                  stage.renderer.domElement, stage.scene);
      cgz.onDragging = (on) => { stage.controlsEnabled = !on; };
      cgz.onCommit = (v) => propsRef.current.onFramePose?.(v);
      S.current = { stage, settings, frustums, gizmo, crops: cropM,
                    floor: floorPlane, grid: originGrid, ruler, frameGroup,
                    sceneSample: null,
                    engine: null, focus: null, ixSig: "", nameSig: "",
                    focusGizmo: fgz, cameraGizmo: cgz ,
        floorGuess: () => {
          const p = propsRef.current;
          const ss = sceneSample();
          if (!ss) return null;
          return floorOfSample({
            sample: ss.sample, frame: p.volumeFrame ?? p.gridFrame, bounds: ss.bounds,
            bandFrac: p.floorBandFrac ?? 0.4,
            binUnits: (p.histBinM ?? 0.05) / (stage.displayScale || 1) });
        } };
      expose({ frustums: frustums.group, gizmo: gizmo.group, crops: cropM.group,
               floor: floorPlane.group, grid: originGrid.group,
               ruler: ruler.group, frameGroup });
      // Every effect below reads S.current and gives up when it is null, so
      // each was relying on its own props changing AGAIN after the stage
      // happened to land. That is a race, and the labels lost it: dataEpoch
      // bumps once, when the journal arrives, and an idle scene never bumps
      // it again — so a stage that finished a round-trip later left the
      // LabelEngine unbuilt for good. This makes the stage a dependency
      // instead of a hope.
      setStageReady(true);

      // focus volume from the scene's volume; the controller is KEPT on
      // the handle — the settings drawer drives it (no boxes -> an
      // inactive stub).
      // Built from the PROP, and rebuilt by the effect below when it
      // changes: this canvas mounts before the volume gate has run, so a
      // one-shot read here left the effect permanently stubbed out on the
      // normal first-run order (VOL is gate 1).
      if (dead || !S.current) return;
      S.current.focus = createFocusVolume({
        splats: stage.splats, root: stage.root, camera: stage.camera,
        volume: propsRef.current.focusBoxes
          ? { boxes: propsRef.current.focusBoxes } : null,
        settings });
      S.current.focus.setAlignment?.(
        alignmentRows(propsRef.current.alignment, propsRef.current.upAxis));

      // per-frame: engine labels, near-camera frustum hiding, and the
      // HTML tag layer — IMPERATIVE div pool with transform-only writes
      // (the viewer's zero-layout discipline; the first React/left-top
      // version repainted at 10 Hz and visibly lagged the camera —
      // operator feedback, first user test).
      // the camera-inside bypass runs per frame
      stage.frameHooks.push(() => S.current?.focus?.updateFocusBypass());
      const tagPool: HTMLDivElement[] = [];
      const camLocal = new THREE.Vector3();
      stage.frameHooks.push(() => {
        S.current?.engine?.update();
        const cam = stage.camera;
        camLocal.copy(cam.position);
        stage.fileGroup.worldToLocal(camLocal);   // the apexes are file frame
        frustums.updateProximity(camLocal);
        // fat-line materials are screen-space — track the viewport
        frustums.setResolution(stage.viewW, stage.viewH);

        // Chips are pills (P3.5): dark bg + border from the .tag3d
        // class; selected flips to filled accent via .sel (text goes
        // dark inline), raycast hover brightens via .hot.
        type TagData = { x: number; y: number; text: string;
                         color: string; sel: boolean; hot: boolean;
                         frame: number | null };
        const out: TagData[] = [];
        if (gizmo.group.visible)
          for (const a of gizmo.anchors(cam, stage.viewW, stage.viewH))
            if (a.visible)
              out.push({ x: a.x, y: a.y, text: a.text,
                         color: a.selected ? "#fff" : "#8a9499",
                         sel: false, hot: false, frame: null });
        if (frustums.group.visible) {
          // Every camera gets a chip (index for auto, M for manual);
          // chips landing on the same spot — the aimed-coverage tangles —
          // fan out into a vertical column so each stays readable and
          // individually clickable (operator request).
          const buckets = new Map<string, number>();
          for (const a of frustums.anchors(cam, stage.viewW, stage.viewH)) {
            const key = `${Math.round(a.x / 30)}:${Math.round(a.y / 20)}`;
            const stack = buckets.get(key) ?? 0;
            buckets.set(key, stack + 1);
            out.push({
              x: a.x, y: a.y - stack * 17,
              text: a.selected
                ? (a.manual ? `M ${a.label ?? ""}`.trim()
                            : `frame ${a.frameIdx}`)
                : (a.manual ? `M${a.frameIdx}`
                            : String(a.frameIdx).padStart(2, "0")),
              color: cssHex(a.manual ? COL.manual : COL.auto),
              sel: a.selected, hot: a.hovered,
              frame: a.frameIdx });
          }
        }
        const layer = tagLayer.current;
        if (!layer) return;
        while (tagPool.length < out.length) {
          const d = document.createElement("div");
          d.className = "tag3d";
          layer.appendChild(d);
          tagPool.push(d);
        }
        for (let i = 0; i < tagPool.length; i++) {
          const el = tagPool[i], a = out[i];
          if (!a) {
            if (el.style.visibility !== "hidden")
              el.style.visibility = "hidden";
            continue;
          }
          if (el.style.visibility !== "")
            el.style.visibility = "";
          el.style.transform = `translate3d(${a.x.toFixed(1)}px, ` +
            `${(a.y - 8).toFixed(1)}px, 0) translate(-50%, -100%)`;
          if (el.textContent !== a.text) el.textContent = a.text;
          const color = a.sel ? "#0a0b0c" : a.color;
          if (el.style.color !== color) el.style.color = color;
          if (el.classList.contains("sel") !== a.sel)
            el.classList.toggle("sel", a.sel);
          if (el.classList.contains("hot") !== a.hot)
            el.classList.toggle("hot", a.hot);
          const f = a.frame === null ? "" : String(a.frame);
          if ((el.dataset.frame ?? "") !== f) {
            if (f === "") delete el.dataset.frame;
            else el.dataset.frame = f;
          }
        }
      });
      // chip clicks select their frame (delegated; the divs are managed
      // outside React)
      const onTagClick = (e: MouseEvent) => {
        const t = (e.target as HTMLElement).closest?.(".tag3d");
        const f = (t as HTMLElement | null)?.dataset.frame;
        if (f !== undefined && f !== "") {
          propsRef.current.onSelectFrame?.(
            +f === frustums.selected ? null : +f);
        }
      };
      tagLayer.current?.addEventListener("click", onTagClick);
      stage.start();

      // pointer: gizmo drag (capture phase beats SparkControls' canvas
      // listeners), click-pick for frustums/boxes
      const el = holder.current;
      const ndc = (e: PointerEvent) => ndcOf(el, e.clientX, e.clientY);
      // Any hand-driven camera move retires the dial's highlight. The same
      // three inputs the stage uses to cancel a fly, so the two agree.
      const offAxis = () => setViewAxis(null);
      const onMoveKey = (e: KeyboardEvent) => {
        // A "w" typed into a text field neither moves the camera (the stage
        // ignores keys while typing) nor should it retire the dial.
        if (isTypingTarget(e.target as HTMLElement)) return;
        if ("wasdqe".includes(e.key.toLowerCase())) offAxis();
      };
      el.addEventListener("pointerdown", offAxis);
      el.addEventListener("wheel", offAxis, { passive: true });
      window.addEventListener("keydown", onMoveKey);
      detachViewDial = () => {
        el.removeEventListener("pointerdown", offAxis);
        el.removeEventListener("wheel", offAxis);
        window.removeEventListener("keydown", onMoveKey);
      };

      let downAt: { x: number; y: number; t: number } | null = null;
      const onDown = (e: PointerEvent) => {
        if (e.button !== 0) return;
        // A handle under the pointer belongs entirely to its manipulator —
        // don't arm the click-pick (Manipulator.anyHandleHot covers every
        // live manipulator, so a new one is covered by construction).
        //
        // Returning WITHOUT stopPropagation is deliberate and load-bearing.
        // This listener is on the holder in the CAPTURE phase and
        // TransformControls listens on the canvas inside it, so stopping the
        // event here would mean the manipulator never receives the
        // pointerdown that starts its drag. The camera is kept still by
        // `onDragging` toggling stage.controlsEnabled instead, which is why
        // that flag exists.
        if (Manipulator.anyHandleHot()) {
          return;
        }
        downAt = { x: e.clientX, y: e.clientY, t: performance.now() };
      };
      const onUp = (e: PointerEvent) => {
        if (!downAt) return;
        const moved = Math.hypot(e.clientX - downAt.x,
                                 e.clientY - downAt.y);
        const dt = performance.now() - downAt.t;
        downAt = null;
        if (moved > 6 || dt > 400) return;   // a drag-look, not a click
        // chip clicks are owned by the tag layer's click handler — running
        // the pick here too selected-then-toggled-off isolated chips (the
        // M chips "did nothing")
        if ((e.target as HTMLElement).closest?.(".tag3d")) return;
        const r = el.getBoundingClientRect();
        const p = propsRef.current;
        const px = e.clientX - r.left, py = e.clientY - r.top;
        // The ruler owns every click while it is armed — for a measurement
        // or for the level act: the point lands on the splat itself
        // (Spark's raycast), or nowhere — a miss is not a pick of anything
        // else. The point is file-frame (the ruler sits under the file
        // group), which is what both records hold.
        if (p.rulerArmed || p.levelArmed || p.floorArmed) {
          const t0 = performance.now();
          const pk = ruler.pick(ndc(e), stage.camera, stage.splatArrays());
          recordPick("lastPick", pk, t0);
          if (pk) (p.floorArmed ? p.onFloorPick
                   : p.levelArmed ? p.onLevelPick : p.onRulerPick)?.(pk.point);
          return;
        }
        if (p.layers.cameras) {
          // pick() cycles through coincident apexes on repeated clicks
          // (aimed coverage rounds share positions); returning the
          // current selection again means "only one here" -> deselect.
          const fidx = frustums.pick(px, py, stage.camera, stage.viewW,
                                     stage.viewH);
          if (fidx !== null) {
            p.onSelectFrame?.(fidx === frustums.selected ? null : fidx);
            return;
          }
        }
        if (p.layers.crops) {
          const cidx = cropM.pick(px, py, stage.camera, stage.viewW,
                                  stage.viewH);
          if (cidx !== null) {   // yellow crop marker -> its source frame
            p.onSelectFrame?.(cidx);
            return;
          }
        }
        // Volumes last, and a miss DESELECTS. Each is pickable only where
        // its controls live — the detection box during the VOL gate, the
        // focus box in the settings panel — so they never compete for a
        // click and only one manipulator can ever be attached.
        const fgzS = S.current?.focusGizmo;
        if (p.focusEditable && fgzS && S.current?.focus?.active) {
          const wires = S.current.focus.focusBoxes.map((fb: any) => fb.wire);
          if (fgzS.hitTest(ndc(e), wires)) { p.onFocusSelect?.(true); return; }
        }
        if (p.layers.volume && p.volumeEditable) {
          const bi = gizmo.pickBox(ndc(e));
          if (bi !== null) { p.onVolumeSelect?.(bi); return; }
        }
        // Nothing under the pointer: clear whichever selection was showing
        // handles. Without this a box, once picked, could never be put down.
        // Frames too — picking a frustum toggles, but an empty-space click
        // must also put a selected camera down (its gizmo detaches with it).
        if (p.layers.cameras && p.selectedFrame !== null)
          p.onSelectFrame?.(null);
        if (p.focusEditable) p.onFocusSelect?.(false);
        if (p.volumeEditable) p.onVolumeSelect?.(-1);
      };
      // raycast hover (P3.6): nearest apex under the pointer — core
      // width bumps + pill brightens; setHover no-ops when unchanged
      const onHover = (e: PointerEvent) => {
        if (!propsRef.current.layers.cameras) {
          frustums.setHover(null);
          return;
        }
        const r = el.getBoundingClientRect();
        frustums.setHover(frustums.hoverAt(
          e.clientX - r.left, e.clientY - r.top,
          stage.camera, stage.viewW, stage.viewH));
      };
      el.addEventListener("pointermove", onHover);
      el.addEventListener("pointerdown", onDown, { capture: true });
      el.addEventListener("pointerup", onUp);
      const onResize = () => stage.onResize();
      // Removed by the OUTER cleanup, like detachViewDial above — the IIFE's
      // return value is a promise nobody consumes, so a cleanup returned
      // from here never runs (that was the leak: every scene switch left a
      // window listener calling setSize on a disposed renderer).
      window.addEventListener("resize", onResize);
      detachResize = () => window.removeEventListener("resize", onResize);
      // the canvas holder resizes without a window resize when chrome
      // hides (panel collapse / 3D-only mode) — track it directly
      ro = new ResizeObserver(onResize);
      ro.observe(el);
    })().catch((e: unknown) => {
      // anything above that threw — the fetch, an unknown container
      // format, a WebGL context that could not be made — is said in the
      // header instead of vanishing as an unhandled rejection
      tellLoad({ state: "failed", loaded: 0, total: 0,
                 error: e instanceof Error ? e.message : String(e) });
    });
    return () => {
      dead = true;
      detachViewDial?.();
      detachResize?.();
      ro?.disconnect();
      const s = S.current;
      if (s) {
        s.focusGizmo?.dispose();
        s.engine?.dispose();
        s.focus?.dispose?.();
        s.frustums.dispose();
        s.gizmo.dispose();
      s.cameraGizmo.dispose();
        s.crops.dispose();
        s.floor.dispose();
        s.grid.dispose();
        s.ruler.dispose();
        s.stage.dispose();
        S.current = undefined;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.scene]);

  // -- instances (LabelEngine rebuilt when interactions change) ------------------
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    let dead = false;
    (async () => {
      try {
        const r = await fetch(runUrl(props.scene, "/interactions.json"));
        if (!r.ok) {
          // No export (cleared, or none yet): the labels of the export that
          // WAS there must go with it, not linger until a reload.
          if (!dead && S.current && s.engine) {
            s.engine.dispose(); s.engine = null; s.ixSig = ""; s.nameSig = "";
          }
          return;
        }
        const ix = await r.json();
        // The verifier's proposal comes from /instances, which
        // derives it for the exports written before it was a field;
        // merged into the extended records the engine reads.
        let extObjects: Array<Record<string, unknown>> = ix.extended?.objects ?? [];
        try {
          const ri = await fetch(runUrl(props.scene, "/instances"));
          if (ri.ok) {
            const rows = (await ri.json()).objects as
              Array<{ proposed_label: string | null }>;
            extObjects = extObjects.map((e, i) =>
              rows[i]?.proposed_label ? { ...e, proposed_label: rows[i].proposed_label } : e);
          }
        } catch { /* the proposals degrade to what the file carries */ }
        // Keyed on content, not on the object count: a re-run that lands
        // on the same count used to keep showing the old labels — and a
        // key on the serialised LENGTH did the same whenever the labels
        // and positions happened to print at the same width, so the
        // content is hashed whole.
        const sig = `${ix.objects?.length}:${hashStr(JSON.stringify(
          ix.objects?.map((o: { label: string; position: { x: number;
            y: number; z: number } }) =>
            [o.label, o.position?.x, o.position?.y, o.position?.z]) ?? []))
          }:${ix.extended?.verification
          ? hashStr(JSON.stringify(ix.extended.verification)) : ""}`;
        if (dead || !S.current || ix.stub) return;
        // The names the display rule reads (the operator's, and the
        // verifier's) live in the extended records, which the signature
        // above does not see — and the server sends no Last-Modified. A
        // relabel changes only these: update the live engine in place.
        const nameSig = JSON.stringify(extObjects.map((e) =>
          [e.operator_label ?? null, e.verified_label ?? null,
           e.proposed_label ?? null]));
        if (sig === s.ixSig) {
          if (nameSig !== s.nameSig && s.engine) {
            s.engine.setExtObjects(extObjects);
            s.nameSig = nameSig;
          }
          return;
        }
        s.ixSig = sig;
        s.nameSig = nameSig;
        s.engine?.dispose();
        s.engine = new LabelEngine({
          stage: s.stage, settings: s.settings, layer: labelLayer.current!,
          objects: ix.objects, extObjects,
          onSelectionChange: (idx: number) =>
            propsRef.current.onInstanceSelect?.(idx),
        });
        // Settle-occlusion rays cost ~150ms/ray PER MILLION splats and
        // queue input behind them — a validated tradeoff for reading
        // labels in a finished scene, wrong as a default on 2.7M+ scenes
        // (operator feedback: "freezes, cannot move the camera"). Own
        // key, default OFF.
        s.engine.occlusionOn =
          localStorage.getItem("carveout_wb_occl") === "1";
        s.engine.applyLabelStyle();
        try {
          const rb = await fetch(runUrl(props.scene, "/instance_ids.bin"));
          if (rb.ok && S.current?.engine === s.engine)
            s.engine.highlighter = createHighlighter({
              splats: s.stage.splats,
              instanceIds: new Uint16Array(await rb.arrayBuffer()),
              extObjects,
              colorOf: (i: number) =>
                s.engine.classColor.get(s.engine.overlays[i].cls),
            });
        } catch { /* tint degrades */ }
      } catch { /* pre-export: no instances */ }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.scene, props.epoch, stageReady]);

  // Rebuild the focus controller when the scene's volume changes. It is
  // built from geometry baked into a shader block, so re-creating is the
  // way to re-aim it; the outgoing one disposes its wires and goes inert.
  // Keyed on the SAVED boxes, never the edit draft — a rebuild per drag
  // frame would recompile the splat pipeline continuously.
  // Under the volume gate's preview the source is the DRAFT, keyed on its
  // box COUNT: adding or removing a box rebuilds (the shader bakes one
  // pair of bounds per box), a drag does not — the effect below re-points
  // the existing uniforms instead.
  const previewBoxes = props.focusPreview
    ? (props.focusDraftBoxes ?? props.focusBoxes) : null;
  const previewCount = previewBoxes?.length ?? -1;
  useEffect(() => {
    const s = S.current;
    if (!s || !s.stage) return;
    const src = props.focusPreview ? previewBoxes : props.focusBoxes;
    s.focus?.dispose?.();
    s.focus = createFocusVolume({
      splats: s.stage.splats, root: s.stage.root, camera: s.stage.camera,
      volume: src ? { boxes: src } : null,
      settings: s.settings });
    s.focus.setDisplayScale?.(s.stage.displayScale);
    s.focus.setAlignment?.(alignmentRows(props.alignment, props.upAxis));
    s.focus.setOverride?.(props.focusPreview ? VOLUME_PREVIEW_FOCUS : null);
    s.focus.applyFocus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.focusBoxes, props.focusPreview, previewCount, stageReady]);

  // The scene alignment: the file group turns under the scene frame,
  // the focus effect carries each centre through it, and a camera still at
  // the start pose is re-placed over the levelled floor.
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    s.stage.setAlignment(alignmentQuaternion(props.alignment, props.upAxis));
    s.focus?.setAlignment?.(alignmentRows(props.alignment, props.upAxis));
  }, [props.alignment, props.upAxis, stageReady]);

  // A proposed alignment previewed: the floor plane and the origin grid are
  // drawn in THAT frame. Their parent sits in the current scene frame, so
  // it turns by R_current · R_proposedᵀ — a file point lands where the
  // proposed frame would put it. Display only.
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    const q = alignmentQuaternion(props.alignment, props.upAxis);
    if (props.previewAlignment) {
      q.multiply(alignmentQuaternion(props.previewAlignment, props.upAxis).invert());
      s.frameGroup.quaternion.copy(q);
    } else {
      s.frameGroup.quaternion.identity();
    }
    s.frameGroup.updateMatrixWorld(true);
  }, [props.previewAlignment, props.alignment, props.upAxis, stageReady]);

  // The drag path: same boxes, moved — uniform writes, no rebuild.
  useEffect(() => {
    const s = S.current;
    if (!s || !props.focusPreview || !props.focusDraftBoxes) return;
    s.focus?.setBoxSources?.(props.focusDraftBoxes);
  }, [props.focusDraftBoxes, props.focusPreview]);

  // -- prop-driven overlay sync ---------------------------------------------------
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    if (props.volumeFrame) s.gizmo.setFrame(props.volumeFrame);
    if (props.volumeBoxes) s.gizmo.setBoxes(props.volumeBoxes);
    s.gizmo.setEditable(props.volumeEditable);
    s.gizmo.select(props.volumeSelected);
  }, [props.volumeBoxes, props.volumeFrame, props.volumeEditable,
      props.volumeSelected, stageReady]);

  // The focus manipulator follows SELECTION, the same way the detection one
  // follows volumeSelected — handles appear on the box you picked and nowhere
  // else. Seeded from the live settings each time it attaches, so it always
  // starts where the volume actually is. `focusEpoch` is in the deps because
  // the sliders can move the volume while it is selected, and the handles
  // have to follow.
  useEffect(() => {
    const s = S.current;
    if (!s?.focusGizmo) return;
    const on = !!props.focusSelected && !props.zen && !!s.focus?.active;
    if (!on) { s.focusGizmo.detach(); return; }
    if (s.focusGizmo.isDragging()) return;   // never re-seat under the hand
    const f = s.settings.focus;
    s.focusGizmo.attach(s.focus.unionCenter, f.offset, f.scale, f.rotation);
  }, [props.focusSelected, props.zen, props.focusEpoch, stageReady]);

  // origin grid: at height 0 of the up axis, from first load (the frame
  // falls back to the profile's convention until a proposal measures one);
  // only its fade distance follows the volume footprint
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    s.grid.setFrame(props.gridFrame);
    s.grid.setFade(footprint(props.volumeBoxes, props.gridFrame).half);
  }, [props.gridFrame, props.volumeBoxes, stageReady]);

  // the grid lies on the floor: the proposal's when known, else the
  // estimate the start pose reads off the splats (the origin until the
  // splats are in) — one surface with the floor plane
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    s.grid.setHeight(props.floorSigned ?? s.floorGuess?.() ?? null);
  }, [props.floorSigned, props.gridFrame, props.volumeFrame, props.alignment,
      props.floorBandFrac, props.histBinM, splatExtent, stageReady]);

  // floor plane: sits at the scene-frame floor, sized to the volume footprint
  // (generous fallback when there are no boxes yet); needs a measured frame
  useEffect(() => {
    const s = S.current;
    if (!s || !props.volumeFrame) return;
    s.floor.setFrame(props.volumeFrame);
    if (props.floorSigned !== null) s.floor.setFloor(props.floorSigned);
    const { center, half } = footprint(props.volumeBoxes, props.volumeFrame);
    s.floor.setExtent(center, half);
  }, [props.volumeFrame, props.floorSigned, props.volumeBoxes, stageReady]);

  useEffect(() => {
    const s = S.current;
    if (!s) return;
    // No frames is a STATE, not a reason to skip: resetting the render step
    // deletes stage1, so /cameras stops answering and `frames` goes null —
    // and bailing here left the deleted render's frustums drawn in 3D until
    // the browser was reloaded. Clear them instead.
    if (!props.frames || !props.intrinsics) {
      s.frustums.clear();
      s.cameraGizmo.detach();
      return;
    }
    s.frustums.scale = props.frustumScale / s.stage.displayScale;
    s.frustums.setFrames(props.frames, props.intrinsics);
    s.frustums.select(props.selectedFrame);
  }, [props.frames, props.intrinsics, stageReady]);

  // The display scale: applied on change so a factor set at the volume
  // gate (or the first proposal's extents) takes effect without a reload.
  // Everything under the root that carries a DISPLAY size in scene units
  // (frustums, the grid's lines, the crop markers, the box-size floor, the
  // focus point size) is told the scale; everything measuring in world
  // space (speeds, clips, label distances) needs nothing.
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    const sc = displayScaleOf(props.factor, props.extent ?? splatExtent,
                              s.settings);
    s.stage.setDisplayScale(sc);
    s.grid.setScale(sc);
    s.gizmo.setUnits(props.factor, sc);
    s.crops.setScale(sc);
    s.frustums.setScale(props.frustumScale / sc);
    s.focus?.setDisplayScale?.(sc);
  }, [props.factor, props.extent, splatExtent, props.frustumScale, stageReady]);

  useEffect(() => {
    S.current?.frustums.select(props.selectedFrame);
  }, [props.selectedFrame, stageReady]);

  // The start pose follows what it is placed by, until the operator moves
  // the camera (pointerdown/wheel clear _poseIsStart in the stage).
  useEffect(() => {
    const s = S.current;
    if (!s || !s.stage._poseIsStart) return;
    s.stage.applyStartPose();
  }, [props.pathMode, props.eyeHeight, props.volumeFrame, props.gridFrame,
      props.floorSigned, props.floorBandFrac, props.histBinM, props.focusBoxes,
      props.alignment, props.upAxis, splatExtent, stageReady]);

  // The ruler — for a measurement, the level act or the floor act: while
  // any is armed the cursor says so, and the stage's splat arrays are
  // built (once a scene) so the first pick does not pay for them.
  //
  // The hover ring (stage 2 of the pick): ONE pick when the pointer rests — a
  // pick walks every splat (pick.ts), so never per move — and only if the
  // camera has not moved since the rest began (a fly, a held key); the
  // ruler retires the ring itself the frame the camera moves. A click at
  // the resting pointer is this pick.
  const picking = !!props.rulerArmed || !!props.levelArmed || !!props.floorArmed;
  useEffect(() => {
    const s = S.current;
    if (!s) return;
    const el = holder.current;
    if (el) el.style.cursor = picking ? "crosshair" : "";
    if (!picking || !el) return;
    const { stage, ruler } = s;
    const tb = performance.now();
    stage.splatArrays();
    (window as unknown as { __carveoutCanvas: Record<string, unknown> })
      .__carveoutCanvas.pickArraysMs = Math.round(performance.now() - tb);
    let timer: number | null = null;
    const cancel = () => {
      if (timer !== null) { clearTimeout(timer); timer = null; }
    };
    const onMove = (e: PointerEvent) => {
      cancel();
      ruler.clearHover();
      if (e.buttons) return;   // a drag-look rests nowhere
      const { clientX: x, clientY: y } = e;
      const cam = stage.camera.matrixWorld.clone();
      timer = window.setTimeout(() => {
        timer = null;
        if (!cam.equals(stage.camera.matrixWorld)) return;
        const ndc = ndcOf(el, x, y);
        const t0 = performance.now();
        const pk = ruler.pick(ndc, stage.camera, stage.splatArrays());
        recordPick("lastHover", pk, t0);
        ruler.setHover(pk, ndc, stage.camera);
      }, HOVER_SETTLE_MS);
    };
    const onLeave = () => { cancel(); ruler.clearHover(); };
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    return () => {
      cancel();
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
      ruler.clearHover();
    };
  }, [picking, stageReady]);
  useEffect(() => {
    if (props.levelArmed)
      S.current?.ruler.setPoints(props.levelPoints ?? null,
                                 props.levelMode ?? "level");
    else
      S.current?.ruler.setPoints(props.rulerPoints ?? null, "ruler");
  }, [props.rulerPoints, props.levelPoints, props.levelArmed, props.levelMode,
      stageReady]);

  // The camera manipulator follows frustum SELECTION, the same way the focus
  // one follows focusSelected. Seeded from the frame's own c2w each time it
  // attaches, so it always starts where the camera actually is — and never
  // re-seated mid-drag, or the handle would jump out from under the pointer
  // as the re-render lands.
  useEffect(() => {
    const s = S.current;
    if (!s?.cameraGizmo) return;
    s.cameraGizmo.setEditable(props.cameraEditable && !props.zen);
    const fr = props.frames?.find(
      (f) => f.frame_idx === props.selectedFrame);
    if (!fr || !props.cameraEditable || props.zen) {
      s.cameraGizmo.detach();
      return;
    }
    if (s.cameraGizmo.isDragging()) return;
    s.cameraGizmo.attach(fr.c2w as number[][]);
  }, [props.selectedFrame, props.cameraEditable, props.zen, props.frames,
      stageReady]);

  useEffect(() => {
    const s = S.current;
    if (!s || !props.intrinsics) return;
    s.crops.set(props.crops, props.intrinsics, props.stale.crops);
  }, [props.crops, props.intrinsics, props.stale.crops, stageReady]);

  useEffect(() => {
    S.current?.frustums.setStale(props.stale.cameras);
  }, [props.stale.cameras, props.frames, stageReady]);
  useEffect(() => {
    S.current?.frustums.setDeleted(props.deletedFrames ?? []);
  }, [props.deletedFrames, props.frames, stageReady]);

  useEffect(() => {
    S.current?.gizmo.setStale(props.stale.volume);
  }, [props.stale.volume, props.volumeBoxes, stageReady]);

  useEffect(() => {
    const s = S.current;
    if (!s) return;
    s.gizmo.setVisible(props.layers.volume);
    s.floor.setVisible(props.layers.volume);  // the floor is volume-gate state
    s.grid.setVisible(props.layers.grid);     // the origin grid is its own layer
    s.frustums.setVisible(props.layers.cameras);
    s.crops.setVisible(props.layers.crops);
    if (s.engine) s.settings.labels.show = props.layers.instances;
  }, [props.layers, stageReady]);

  useImperativeHandle(ref, () => ({
    get stage() { return S.current?.stage ?? null; },
    get settings() { return S.current?.settings ?? null; },
    get focus() { return S.current?.focus ?? null; },
    get engine() { return S.current?.engine ?? null; },
    apply(path: string, value?: unknown) {
      const s = S.current;
      if (!s) return;
      applySetting({ stage: s.stage, engine: s.engine, focus: s.focus,
                     settings: s.settings }, path, value);
    },
    resetFocus() {
      const s = S.current;
      if (!s) return;
      resetFocus({ stage: s.stage, engine: s.engine, focus: s.focus,
                   settings: s.settings });
    },
    setGizmoMode(mode: GizmoMode) { S.current?.gizmo.setMode(mode); },
    capturePose(label?: string) {
      const s = S.current!;
      return captureView({ stage: s.stage, settings: s.settings, label });
    },
    flyToFrame(fr: FrameEntry) {
      const s = S.current;
      if (!s) return;
      const m = new THREE.Matrix4().fromArray(
        (fr.c2w as number[][]).flat()).transpose();
      const apex = new THREE.Vector3().setFromMatrixPosition(m);
      const fwd = new THREE.Vector3(0, 0, 1).applyMatrix4(
        new THREE.Matrix4().extractRotation(m));
      // cameras.json is file frame: through the file group
      s.stage.fileGroup.updateMatrixWorld(true);
      const to = s.stage.fileGroup.localToWorld(apex.clone());
      const target = s.stage.fileGroup.localToWorld(
        apex.clone().addScaledVector(fwd, 1.5));
      s.stage.flyTo({ to, target });
    },
    flyToInstance(idx: number) {
      S.current?.engine?.select(idx);
    },
    deselectInstance() {
      const e = S.current?.engine;
      if (e && e.selected >= 0) e.select(e.selected);   // select toggles
    },
    setFocusGizmoMode(mode: FocusMode) {
      S.current?.focusGizmo?.setMode(mode);
    },
    setCameraGizmoMode(mode: CameraMode) {
      S.current?.cameraGizmo?.setMode(mode);
    },
  }), []);

  return (
    <div ref={holder}
         className="absolute inset-0 overflow-hidden [&>canvas]:block">
      <div id="wb-labels" ref={labelLayer} />
      {/* tag divs are managed imperatively by the frame hook — React
          never touches this subtree (zero-layout discipline) */}
      <div ref={tagLayer}
           className="absolute inset-0 pointer-events-none z-[6]" />
      {!props.zen && <ViewGizmo axis={viewAxis} onChoose={(a) => {
        const s = S.current;
        if (!s) return;
        const b = worldBounds(s);
        const { to, target } = placeCamera(a, b);
        s.stage.flyTo({ to: new THREE.Vector3(...to),
                        target: new THREE.Vector3(...target) });
        setViewAxis(a);
      }} />}
    </div>
  );
});
