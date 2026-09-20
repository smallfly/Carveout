// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The three manipulator ADAPTERS. Each one owns only its proxy <-> domain
// mapping (box min/max, focus offset/scale/rotation, OpenCV c2w flip);
// the TransformControls scaffolding — proxy, helper parenting, drag
// wiring, commit-on-release, handle-hot — is the shared Manipulator.
//
// The manipulator is three's own `TransformControls` — the same control
// `@react-three/drei` wraps, used directly because this canvas is raw
// three (viewer/lib has no React at all).
// Zero new dependencies: three ships it and vite already aliases
// `three/addons/`.

import * as THREE from "three";
import type { VolumeBox } from "../../types";
import { COL, DEG, type VolumeFrame } from "./common";
import { Manipulator } from "./manipulator";

export type GizmoMode = "translate" | "scale";

/**
 * Volume boxes in 3D, with a real manipulator.
 *
 * This replaces two world-sized sphere handles that only resized height. They
 * were what survived after "3D translate/resize was still not usable enough"
 * (user testing) — and the reason is visible in the old code: the spheres were
 * a fixed 0.035 m, so on a room-scale scene they were specks, with no hover
 * state and no feedback. A gizmo that scales itself to the screen and lights
 * up under the pointer is a different instrument.
 *
 * Boxes stay AXIS-ALIGNED — volume.json is min/max and `in_volume()` reads it
 * that way — so the control is never put in rotate mode. Translate moves the
 * centre, scale changes the size about that centre, which is exactly the pair
 * the panel's centre/size sliders already write.
 */
export class VolumeGizmo {
  /** metres per scene unit when recorded (the tags' unit), null = not known */
  private factor: number | null = null;
  /** display metres per scene unit — the root's uniform scale (size floors) */
  private displayScale = 1;
  setUnits(factor: number | null, displayScale: number) {
    this.factor = factor;
    this.displayScale = displayScale > 0 ? displayScale : 1;
  }
  group = new THREE.Group();
  boxes: VolumeBox[] = [];
  selected = -1;
  editable = false;
  stale = false;
  /** drag END — one save per gesture, the panel's commit path */
  onChange: ((boxes: VolumeBox[]) => void) | null = null;
  /** live during a drag, so the panel's numerics track the hand */
  onDraft: ((boxes: VolumeBox[]) => void) | null = null;
  /** true while a handle is held: the camera must stand still */
  onDragging: ((dragging: boolean) => void) | null = null;
  private wires: THREE.Line[] = [];
  private frame: VolumeFrame = { up: 1, upSign: 1, a0: 0, a1: 2 };
  private mode: GizmoMode = "translate";
  /** position = box centre, scale = box size — mapped back to min/max */
  private m: Manipulator;
  private ray = new THREE.Raycaster();

  constructor(private root: THREE.Group, private camera: THREE.Camera,
              domElement: HTMLElement, scene: THREE.Scene) {
    root.add(this.group);
    this.m = new Manipulator({ parent: this.group, camera, domElement,
                               scene, mode: this.mode });
    this.m.onDragging = (on) => this.onDragging?.(on);
    this.m.onRelease = () => this.commit();
    this.m.onChange = () => {
      this.readProxy();
      this.drawWires();
      this.onDraft?.(this.snapshot());
    };
  }

  setFrame(f: VolumeFrame) { this.frame = f; }

  setMode(mode: GizmoMode) {
    this.mode = mode;
    this.m.setMode(mode);
  }

  setBoxes(boxes: VolumeBox[]) {
    this.boxes = boxes.map((b) => ({ name: b.name, min: [...b.min],
                                     max: [...b.max] }));
    if (this.selected >= this.boxes.length) this.selected = -1;
    // A drag drafts through React and comes straight back here as new boxes.
    // The numbers are the ones this gizmo just produced, so the wires are
    // right — but re-attaching TransformControls mid-gesture would drop the
    // drag. Redraw only until the hand lets go.
    if (this.isDragging()) { this.drawWires(); return; }
    this.rebuild();
  }

  setVisible(on: boolean) { this.group.visible = on; this.syncControl(); }
  setEditable(on: boolean) { this.editable = on; this.syncControl(); }
  setStale(on: boolean) { this.stale = on; this.drawWires(); }

  select(i: number) {
    this.selected = i;
    this.rebuild();
  }

  private snapshot(): VolumeBox[] {
    return this.boxes.map((b) => ({ name: b.name, min: [...b.min],
                                    max: [...b.max] }));
  }

  private commit() { this.onChange?.(this.snapshot()); }

  /** proxy transform -> the selected box's min/max. A box is never allowed to
   *  invert or collapse: scale is clamped to 5 DISPLAY centimetres per axis
   *  (scene units: 0.05 / displayScale — a fixed 0.05 scene units was 50 m
   *  on a scene whose unit is a kilometre), the same floor the old height
   *  handles enforced. */
  private readProxy() {
    const b = this.boxes[this.selected];
    if (!b) return;
    const c = this.m.proxy.position, sc = this.m.proxy.scale;
    const floor = 0.05 / this.displayScale;
    for (let a = 0; a < 3; a++) {
      const size = Math.max(Math.abs(sc.getComponent(a)), floor);
      sc.setComponent(a, size);
      const ctr = c.getComponent(a);
      b.min[a] = ctr - size / 2;
      b.max[a] = ctr + size / 2;
    }
  }

  /** the selected box -> proxy transform (the other direction) */
  private writeProxy() {
    const b = this.boxes[this.selected];
    if (!b) return;
    this.m.proxy.position.set(...[0, 1, 2].map(
      (a) => (b.max[a] + b.min[a]) / 2) as [number, number, number]);
    this.m.proxy.scale.set(...[0, 1, 2].map(
      (a) => Math.max(b.max[a] - b.min[a], 0.05 / this.displayScale)) as
      [number, number, number]);
  }

  /** attach or detach the control — it shows only on a selected box in an
   *  editable, visible gizmo. */
  private syncControl() {
    const on = this.editable && this.group.visible
      && this.selected >= 0 && this.selected < this.boxes.length;
    if (on) {
      this.writeProxy();
      this.m.attach();
    } else {
      this.m.detach();
    }
  }

  private rebuild() {
    this.drawWires();
    this.syncControl();
  }

  private drawWires() {
    for (const w of this.wires) {
      w.geometry.dispose();
      (w.material as THREE.Material).dispose();
      this.group.remove(w);
    }
    this.wires = [];
    this.boxes.forEach((b, i) => {
      const size = [0, 1, 2].map((a) => b.max[a] - b.min[a]);
      const ctr = [0, 1, 2].map((a) => (b.max[a] + b.min[a]) / 2);
      const sel = i === this.selected;
      const wire = new THREE.LineSegments(
        new THREE.EdgesGeometry(
          new THREE.BoxGeometry(size[0], size[1], size[2])),
        new THREE.LineBasicMaterial({
          color: this.stale ? COL.stale : sel ? COL.volumeSel : COL.volume,
          transparent: true, opacity: this.stale ? 0.35 : sel ? 1 : 0.8 }));
      wire.position.set(ctr[0], ctr[1], ctr[2]);
      // After the splat pass (Spark's mesh: renderOrder 0), depth-tested
      // against the depth the splats write — without this the wire sorted
      // against the splat mesh by distance and its occlusion was luck.
      wire.renderOrder = 3;
      this.group.add(wire);
      this.wires.push(wire);
      // Translucent faces: the edges alone vanished against the splats
      // (operator feedback). Faint, no depth write, so the splats and
      // the handles stay readable through them; the selected box a shade
      // stronger.
      const fill = new THREE.Mesh(
        new THREE.BoxGeometry(size[0], size[1], size[2]),
        new THREE.MeshBasicMaterial({
          color: this.stale ? COL.stale : sel ? COL.volumeSel : COL.volume,
          transparent: true, opacity: this.stale ? 0.04 : sel ? 0.14 : 0.08,
          depthWrite: false, side: THREE.DoubleSide }));
      fill.position.set(ctr[0], ctr[1], ctr[2]);
      fill.renderOrder = 2;
      this.group.add(fill);
      this.wires.push(fill as unknown as THREE.LineSegments);
    });
  }

  /** Screen anchors for the always-on tags. Reads centre/size in the mode
   *  being manipulated, so the number under the hand is the one being changed. */
  anchors(camera: THREE.Camera, viewW: number, viewH: number) {
    const { up, upSign } = this.frame;
    const v = new THREE.Vector3();
    return this.boxes.map((b, i) => {
      const ctr = [0, 1, 2].map((a) => (b.max[a] + b.min[a]) / 2);
      // anchored at the box's TOP face in the signed sense — raw max[up]
      // is the bottom on a -y scene, where the pill sat on the floor
      // handle and hid it
      ctr[up] = upSign > 0 ? b.max[up] : b.min[up];
      v.set(ctr[0], ctr[1], ctr[2])
        .applyMatrix4(this.root.matrixWorld).project(camera);
      const vis = v.z < 1 && v.z > -1 && Math.abs(v.x) < 1 &&
        Math.abs(v.y) < 1;
      // Metres only with a recorded factor; scene units, and said so,
      // otherwise — the tag never converts by a number it does not have.
      const k = this.factor ?? 1;
      const unit = this.factor == null ? " units" : " m";
      // metres to centimetres; scene units to significant digits (a unit
      // may be a kilometre or a millimetre — fixed decimals read as 0.00)
      const fmt = (v: number) => this.factor == null
        ? Number(v.toPrecision(4)).toString() : v.toFixed(2);
      const size = [0, 1, 2].map((a) => (b.max[a] - b.min[a]) * k);
      const text = this.mode === "scale" && i === this.selected
        ? `${b.name} · ${size.map(fmt).join(" × ")}${unit}`
        // SIGNED heights (up = positive), the panel's and the Floor's ruler
        : `${b.name} · height ` +
          `${fmt(Math.min(b.min[up] * upSign, b.max[up] * upSign) * k)} … ` +
          `${fmt(Math.max(b.min[up] * upSign, b.max[up] * upSign) * k)}${unit}`;
      return {
        i, visible: vis && this.group.visible,
        x: (v.x * 0.5 + 0.5) * viewW,
        y: (-v.y * 0.5 + 0.5) * viewH,
        text,
        selected: i === this.selected,
      };
    });
  }

  /** Pick a box by ray for selection (clicking its wireframe volume). */
  pickBox(ndc: THREE.Vector2): number | null {
    if (!this.group.visible) return null;
    this.ray.setFromCamera(ndc, this.camera);
    const local = new THREE.Ray();
    const inv = this.root.matrixWorld.clone().invert();
    local.copy(this.ray.ray).applyMatrix4(inv);
    let best: number | null = null, bestT = Infinity;
    const bb = new THREE.Box3();
    const pt = new THREE.Vector3();
    this.boxes.forEach((b, i) => {
      bb.min.set(b.min[0], b.min[1], b.min[2]);
      bb.max.set(b.max[0], b.max[1], b.max[2]);
      if (local.intersectBox(bb, pt)) {
        const t = pt.distanceTo(local.origin);
        if (t < bestT) { bestT = t; best = i; }
      }
    });
    return best;
  }

  isDragging(): boolean { return this.m.isDragging(); }

  dispose() {
    this.m.dispose();
    this.setBoxes([]);
    this.root.remove(this.group);
  }
}

export type FocusMode = "translate" | "rotate" | "scale";

/**
 * The FOCUS volume's adapter — the dual-render display effect, not the
 * detection volume.
 *
 * The mapping is one-to-one, which is why this is worth having rather than a
 * bank of sliders: the focus volume IS an offset, a scale and a rotation
 * applied to the detection boxes about their union centre, so a
 * TransformControls attached to a proxy sitting at that centre drives exactly
 * those three settings and nothing is derived or approximated.
 *   position - unionCentre -> focus.offset
 *   scale                  -> focus.scale
 *   rotation (degrees)     -> focus.rotation
 *
 * Rotate IS offered here, unlike on the detection volume: this turns a
 * display effect, and nothing downstream reads it. The detection volume stays
 * axis-aligned because `in_volume()` reads min/max.
 *
 * Only one manipulator is ever attached: the detection gizmo is editable only
 * while the VOL gate is the active panel, and these controls live under the
 * settings panel, so the two are mutually exclusive by construction.
 */
export class FocusGizmo {
  private m: Manipulator;
  private pivot = new THREE.Vector3();
  private ray = new THREE.Raycaster();
  /** live during a drag: offset, scale and rotation (degrees) */
  onChange: ((v: { offset: number[]; scale: number[];
                   rotation: number[] }) => void) | null = null;
  onDragging: ((dragging: boolean) => void) | null = null;

  constructor(root: THREE.Group, private camera: THREE.Camera,
              domElement: HTMLElement, scene: THREE.Scene) {
    // proxy in ply space, same frame as the focus boxes
    this.m = new Manipulator({ parent: root, camera, domElement, scene });
    this.m.onDragging = (on) => this.onDragging?.(on);
    this.m.onChange = () => this.emit();
  }

  setMode(mode: FocusMode) { this.m.setMode(mode); }

  /** Show the handles, seeded from the current settings. `pivot` is the focus
   *  controller's own union centre — never recomputed here, or the two
   *  definitions drift the first time a box moves. */
  attach(pivot: THREE.Vector3, offset: number[], scale: number[],
         rotation: number[]) {
    this.pivot.copy(pivot);
    const p = this.m.proxy;
    p.position.copy(pivot).add(
      new THREE.Vector3(offset[0], offset[1], offset[2]));
    p.scale.set(scale[0], scale[1], scale[2]);
    p.rotation.set(rotation[0] * DEG, rotation[1] * DEG,
                   rotation[2] * DEG, "XYZ");
    this.m.attach();
  }

  detach() { this.m.detach(); }

  isDragging(): boolean { return this.m.isDragging(); }

  /** Is one of the focus wireframes under this pointer? The wires are unit
   *  boxes carrying their own position/rotation/scale, so each is tested in
   *  its OWN space — an axis-aligned test in ply space would pick the wrong
   *  thing the moment the volume is turned, which is the whole point of it
   *  being turnable. Takes the wires rather than owning them: focus.js builds
   *  and disposes them, and a second copy of that list is a second truth. */
  hitTest(ndc: THREE.Vector2, wires: THREE.Object3D[]): boolean {
    if (!wires.length) return false;
    this.ray.setFromCamera(ndc, this.camera);
    const local = new THREE.Ray();
    const unit = new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5),
                                new THREE.Vector3(0.5, 0.5, 0.5));
    const hit = new THREE.Vector3();
    for (const w of wires) {
      if (!w.visible) continue;
      w.updateMatrixWorld();
      local.copy(this.ray.ray).applyMatrix4(
        w.matrixWorld.clone().invert());
      if (local.intersectBox(unit, hit)) return true;
    }
    return false;
  }

  private emit() {
    const p = this.m.proxy;
    // Scale is clamped to the slider's own floor rather than to zero: a zero
    // scale collapses the box to a plane the effect can never be dragged back
    // out of, because every later scale multiplies it.
    const sc = [0, 1, 2].map((a) =>
      Math.max(Math.abs(p.scale.getComponent(a)), 0.1));
    for (let a = 0; a < 3; a++) p.scale.setComponent(a, sc[a]);
    this.onChange?.({
      offset: [p.position.x - this.pivot.x, p.position.y - this.pivot.y,
               p.position.z - this.pivot.z],
      scale: sc,
      rotation: [p.rotation.x / DEG, p.rotation.y / DEG, p.rotation.z / DEG],
    });
  }

  dispose() { this.m.dispose(); }
}

// ---------------------------------------------------------------------------
// camera manipulator (render gate) — move/turn one detection viewpoint
// ---------------------------------------------------------------------------

export type CameraMode = "translate" | "rotate";

/**
 * The adapter for ONE camera frustum, so a viewpoint can be adjusted in
 * place instead of deleted and re-flown.
 *
 * Only MANUAL views get one, and only in fully-manual mode: auto views are
 * regenerated by the path sampler every render, so a pose edited on one would
 * be silently recomputed away. `render.path_mode: "manual"` is what makes an
 * adjustment durable, and the render gate offers the conversion.
 *
 * No scale handle: a camera has no size. Its analogue is fov, which is a
 * number on the captured pose, not something to drag.
 *
 * The proxy lives in ply space, so `proxy.position` / `proxy.quaternion`
 * ARE the values manual_views.json stores — no conversion on the way out.
 * On the way IN, `c2w` is OpenCV (+z forward, +y down) while the file
 * stores the three.js/GL quaternion (looks down -Z, up +Y):
 * carveout/manual_views.py defines `R_cv = R_gl @ diag(1, -1, -1)`, and
 * that matrix is its own inverse, so the same multiply undoes it. `FLIP`
 * below is that matrix, named once.
 */
const FLIP = new THREE.Matrix4().makeScale(1, -1, -1);

export class CameraGizmo {
  private m: Manipulator;
  private editable = true;
  /** committed on release, not per-frame: one PATCH per adjustment, and the
   *  re-render is an explicit operator act either way */
  onCommit: ((v: { position: number[]; quaternion: number[];
                   c2w: number[][] }) => void) | null = null;
  onDragging: ((dragging: boolean) => void) | null = null;

  constructor(root: THREE.Group, camera: THREE.Camera,
              domElement: HTMLElement, scene: THREE.Scene) {
    // smaller than the volume's: frustums are small and often clustered
    this.m = new Manipulator({ parent: root, camera, domElement, scene,
                               size: 0.7 });
    this.m.onDragging = (on) => this.onDragging?.(on);
    this.m.onRelease = () => this.commit();
  }

  setMode(mode: CameraMode) { this.m.setMode(mode); }

  setEditable(on: boolean) {
    this.editable = on;
    if (!on) this.detach();
  }

  /** Show the handles on one frame's pose. `c2w` is the row-major 4x4 the
   *  cameras.json frame carries. */
  attach(c2w: number[][]) {
    if (!this.editable) return;
    const mtx = new THREE.Matrix4().fromArray(c2w.flat()).transpose();
    mtx.multiply(FLIP);            // OpenCV c2w -> GL camera basis
    this.m.proxy.position.setFromMatrixPosition(mtx);
    this.m.proxy.quaternion.setFromRotationMatrix(mtx);
    this.m.attach();
  }

  detach() { this.m.detach(); }

  isDragging(): boolean { return this.m.isDragging(); }

  private commit() {
    const p = this.m.proxy;
    // Also hand back an OpenCV c2w, so the caller can redraw the frustum at
    // the new pose immediately. Without it the only source of frustum
    // geometry is cameras.json, which still describes the PREVIOUS render —
    // the frustum and the handles both snapped back and the edit looked
    // like it had been discarded.
    const mtx = new THREE.Matrix4()
      .compose(p.position, p.quaternion, new THREE.Vector3(1, 1, 1))
      .multiply(FLIP);                    // GL camera basis -> OpenCV c2w
    const a = mtx.clone().transpose().toArray();   // -> row-major, as stored
    this.onCommit?.({
      position: [p.position.x, p.position.y, p.position.z],
      quaternion: [p.quaternion.x, p.quaternion.y, p.quaternion.z,
                   p.quaternion.w],
      c2w: [a.slice(0, 4), a.slice(4, 8), a.slice(8, 12), a.slice(12, 16)],
    });
  }

  dispose() { this.m.dispose(); }
}
