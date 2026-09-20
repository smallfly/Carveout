// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// SplatStage: renderer + scene + camera + root group + Spark splats +
// SparkControls + fly-to animation — the shared 3D canvas core, extracted
// from main.js unchanged (web-UI task: INTEGRATED, not rebuilt).
//
// Two groups: `root` carries the pi-about-X display flip and the
// display scale, and is the SCENE frame — where the volume boxes, the
// floor plane and the origin grid draw; `fileGroup` under it carries the
// scene alignment (a small rotation recorded in the profile) and holds
// the splats and every overlay in the FILE's own coordinates (frustums,
// the ruler, the labels). With no alignment the two coincide.
//
// Navigation (aligned with Spark's own viewer/
// editor examples): SparkControls — hold LMB +
// drag to look (no pointer lock), WASD + QE move, Shift fast — plus
// right-drag view-plane pan, middle-drag ground-plane pan, wheel dolly.

import * as THREE from "three";
import { SparkRenderer, SplatFileType, SplatMesh,
         SparkControls } from "@sparkjsdev/spark";

// Scene containers Carveout serves, mapped onto Spark's loaders. The format
// is passed in rather than sniffed from the URL: the workbench already
// knows it (from
// /api/runs/<scene>/scene_source), and an explicit type keeps the URL free
// to be anything the server wants to call it.
const SPLAT_FILE_TYPE = {
  ply: SplatFileType.PLY,
  sog: SplatFileType.PCSOGSZIP,   // PlayCanvas SOG v2, zipped
};

// Keyboard input must yield to a focused text field: a "w" typed into any
// text input walked the camera, hotkeys renamed a view to "flip" and
// flipped the scene — a bug that shipped twice because two callers each
// carried their own copy of this predicate. ONE definition, exported:
// the stage checks it per frame (focus can change between a keydown and
// the tick that consumes it), the keydown listeners check the event
// target. Defaults to the focused element.
export function isTypingTarget(el = document.activeElement) {
  return !!el && (el.closest?.("input, textarea, select") != null
                  || el.isContentEditable);
}

export class SplatStage {
  constructor({ container, sceneUrl, sceneFormat, settings, onLoadProgress }) {
    const fileType = SPLAT_FILE_TYPE[sceneFormat];
    if (!fileType) {
      throw new Error(`unsupported scene format: ${sceneFormat}`);
    }
    this.container = container;
    this.settings = settings;
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(settings.background);
    this.camera = new THREE.PerspectiveCamera(
      65, container.clientWidth / container.clientHeight, 0.05, 500);
    this.camera.rotation.order = "YXZ";

    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    // The file frame under the scene frame: its quaternion is the
    // alignment R (scene = R · file), identity until the canvas sets one.
    this.fileGroup = new THREE.Group();
    this.root.add(this.fileGroup);

    // The file's bytes as they arrive (the server names the length), for
    // whoever tells the operator the scene is on its way: a splat file
    // can be gigabytes, and a blank canvas says nothing.
    this.loadProgress = { loaded: 0, total: 0 };
    this.splats = new SplatMesh({
      url: sceneUrl, fileType,
      onProgress: (e) => {
        this.loadProgress = { loaded: e.loaded ?? 0,
                              total: e.lengthComputable ? e.total : 0 };
        onLoadProgress?.(this.loadProgress);
      },
    });
    this.fileGroup.add(this.splats);

    this.viewW = container.clientWidth;
    this.viewH = container.clientHeight;

    // Perf counters: exported by the app as
    // window.__carveoutPerf — readable from a headless drive.
    this.perf = { frames: 0, frameMs: 0, labelMs: 0, occlMs: 0, occlRays: 0,
                  shown: 0,
                  reset() { this.frames = this.frameMs = this.labelMs =
                            this.occlMs = this.occlRays = 0; } };

    this.controls = new SparkControls({ canvas: this.renderer.domElement });
    // Set false to freeze the camera without tearing the controls down: a
    // gizmo drag must not also fly the camera, and SparkControls listens on
    // the same canvas the gizmo does, so suppressing it here is the one place
    // that cannot be lost to listener ordering.
    this.controlsEnabled = true;
    // Every speed and clip distance below is a DISPLAY-METRE figure. The
    // scene is put into display metres by ONE uniform scale on the root
    // group (setDisplayScale): the recorded factor when the scene's scale
    // is known, else a nominal extent over the scene's own. Nothing here,
    // and no overlay that measures in world space, knows the scene's unit;
    // what lives under `root` (splats, boxes, frustums, the grid) is in
    // scene units and is scaled on the way to the screen.
    this.displayScale = 1;
    this.controls.fpsMovement.moveSpeed = 2.0;          // validated base speed, m/s
    this.controls.fpsMovement.shiftMultiplier = 3.0;    // Shift = 6 m/s
    this.controls.pointerControls.slideSpeed = 6e-3;    // Spark defaults
    this.controls.pointerControls.scrollSpeed = 0.0015;
    this.applyControlFeel();

    this.flyAnim = null;
    // The pose applyStartPose uses when none is saved: set by the canvas
    // (world-space {to, target} or null); null = the bare fallback.
    this.defaultPose = null;
    this._splatBounds = null;   // splatBounds() cache
    this._splatSample = null;   // splatSample() cache
    this._splatArrays = null;   // splatArrays() cache
    this.onFlyArrive = null;   // (fov|null) => void — GUI FOV sync hook
    this.preHooks = [];        // called with (now) before controls/fly
    this.frameHooks = [];      // called with (now) after camera update

    // Grabbing the view or scrolling cancels a fly-to, and means the
    // camera is the operator's now, not the start pose.
    this._poseIsStart = false;
    this.renderer.domElement.addEventListener(
      "pointerdown", () => { this.flyAnim = null; this._poseIsStart = false; });
    this.renderer.domElement.addEventListener(
      "wheel", () => { this.flyAnim = null; this._poseIsStart = false; });

    this._clock = new THREE.Clock();
    this._running = false;
  }

  get isFlying() { return this.flyAnim !== null; }

  /** Display metres per scene unit: ONE uniform scale on the root group.
   * The recorded factor when the scene's scale is known (world space is
   * then metres), else a nominal extent over the scene's own (the scene
   * is displayed about room-sized). Fly speed 2 m/s, the pan and dolly
   * gains, the 0.05..500 m clip planes, the grid's metre lines, the label
   * distances and the frustum size are all display-metre figures and need
   * no conversion — a 100x scene flies at the same metres per second. */
  setDisplayScale(s) {
    s = Number(s) > 0 ? Number(s) : 1;
    if (s === this.displayScale) return;
    // The camera keeps its place RELATIVE TO THE SCENE: its root-local
    // position is taken before the scale moves and put back after, so the
    // scene never shrinks or swells away from the operator (a proposal on
    // a 72x scene changed the scale and left the camera in a black
    // void). A camera still at the start pose is re-placed there.
    this.root.updateMatrixWorld(true);
    const local = this.root.worldToLocal(this.camera.position.clone());
    this.displayScale = s;
    this.root.scale.setScalar(s);
    this.root.updateMatrixWorld(true);
    if (this._poseIsStart) this.applyStartPose();
    else {
      this.flyAnim = null;
      this.camera.position.copy(this.root.localToWorld(local));
    }
  }

  /** Robust bounds of the loaded splats, in scene units, from a
   * subsample: the 1st..99th percentile low and high per axis, and the
   * span between them. The display scale needs an extent BEFORE any
   * proposal has measured the scene (until then a 72x scene flew 72x too
   * slowly), and the start pose needs a centre before any
   * volume exists; the sidecar's extent replaces the span when it exists.
   * Null until the splats are loaded; computed once, then cached. */
  splatBounds(step = 8) {
    if (this._splatBounds) return this._splatBounds;
    const sample = this.splatSample(step);
    if (!sample) return null;
    const pct = (src) => {
      const a = Float32Array.from(src).sort();
      return [a[Math.floor(a.length * 0.01)],
              a[Math.min(a.length - 1, Math.floor(a.length * 0.99))]];
    };
    const [x, y, z] = [pct(sample.x), pct(sample.y), pct(sample.z)];
    this._splatBounds = { lo: [x[0], y[0], z[0]], hi: [x[1], y[1], z[1]],
                          span: [x[1] - x[0], y[1] - y[0], z[1] - z[0]] };
    return this._splatBounds;
  }

  /** Every `step`-th splat's centre and opacity, in scene units and in the
   * FILE frame, as typed arrays {x, y, z, opacity}: what splatBounds()
   * measures, and what the start pose's floor estimate reads (through the
   * alignment) before any proposal has detected one. Null until the splats
   * are loaded; taken once, then cached. */
  splatSample(step = 8) {
    if (this._splatSample) return this._splatSample;
    const sp = this.splats;
    if (!sp || typeof sp.forEachSplat !== "function") return null;
    const xs = [], ys = [], zs = [], os = [];
    let i = 0;
    sp.forEachSplat((index, center, scales, quaternion, opacity) => {
      if ((i++ % step) !== 0) return;
      xs.push(center.x); ys.push(center.y); zs.push(center.z);
      os.push(Number.isFinite(opacity) ? opacity : 1);
    });
    if (xs.length < 100) return null;
    this._splatSample = { x: Float32Array.from(xs), y: Float32Array.from(ys),
                          z: Float32Array.from(zs), opacity: Float32Array.from(os) };
    return this._splatSample;
  }

  /** Every splat's centre, scales, rotation (x, y, z, w) and opacity as
   * flat typed arrays, in scene units and the FILE frame: what the pick
   * walks (webui `overlays/pick.ts` — the median depth along a click's
   * ray). 44 bytes a splat (2M splats: 88 MB), built once on the first
   * call and kept; the canvas asks for it when an act is armed, so the
   * build lands on the arm and not on the first hover. Null until the
   * splats are loaded. */
  splatArrays() {
    if (this._splatArrays) return this._splatArrays;
    const sp = this.splats;
    if (!sp || typeof sp.forEachSplat !== "function") return null;
    const n = sp.splats?.numSplats ?? sp.packedSplats?.numSplats ?? 0;
    if (!n) return null;
    const center = new Float32Array(n * 3), scale = new Float32Array(n * 3),
          quat = new Float32Array(n * 4), opacity = new Float32Array(n);
    let k = 0;
    sp.forEachSplat((index, c, s, q, o) => {
      if (k >= n) return;
      center[k * 3] = c.x; center[k * 3 + 1] = c.y; center[k * 3 + 2] = c.z;
      scale[k * 3] = s.x; scale[k * 3 + 1] = s.y; scale[k * 3 + 2] = s.z;
      quat[k * 4] = q.x; quat[k * 4 + 1] = q.y; quat[k * 4 + 2] = q.z;
      quat[k * 4 + 3] = q.w;
      opacity[k] = Number.isFinite(o) ? o : 1;
      k++;
    });
    if (!k) return null;
    this._splatArrays = { n: k, center, scale, quat, opacity };
    return this._splatArrays;
  }

  /** The 1st..99th percentile span per axis (splatBounds().span). */
  splatExtent(step = 8) {
    return this.splatBounds(step)?.span ?? null;
  }

  applyFlip() {
    const on = this.settings.flip;
    this.root.quaternion.set(on ? 1 : 0, 0, 0, on ? 0 : 1);
  }

  /** The scene alignment as a quaternion on the file group (null = none).
   * A camera still at the start pose is re-placed: the pose is made in
   * the scene frame, which has just moved under the file. */
  setAlignment(q) {
    const next = q ?? new THREE.Quaternion();
    if (this.fileGroup.quaternion.equals(next)) return;
    this.fileGroup.quaternion.copy(next);
    this.fileGroup.updateMatrixWorld(true);
    if (this._poseIsStart) this.applyStartPose();
  }

  applyStartPose() {
    const s = this.settings.camera;
    this.camera.fov = s.fov;
    this.camera.updateProjectionMatrix();
    if (s.position && s.quaternion) {
      const p = new THREE.Vector3().fromArray(s.position);
      const q = new THREE.Quaternion().fromArray(s.quaternion);
      if (s.space === "root") {
        // Stored ROOT-LOCAL (scene units, like manual views), so the pose
        // stays put whatever the display scale; the flip and the scale
        // live on the root and are applied here.
        this.root.updateMatrixWorld(true);
        this.root.localToWorld(p);
        q.premultiply(this.root.quaternion);
      }   // else: a file from before `space` — world space at scale 1
      this.camera.position.copy(p);
      this.camera.quaternion.copy(q);
    } else {
      // No saved pose: the canvas supplies one from the scene frame (eye
      // height over the floor inside a walkable scene, the boxes framed for
      // an object) through `defaultPose`, a hook returning world-space
      // {to, target} or null while it has nothing to place by. Only then
      // the bare fallback: 3, 2, 3 display metres, looking at the origin.
      this.flyAnim = null;
      const d = this.defaultPose ? this.defaultPose() : null;
      if (d) {
        this.camera.position.copy(d.to);
        this.camera.lookAt(d.target);
      } else {
        this.camera.position.set(3, 2, 3);
        this.camera.lookAt(0, 0, 0);
      }
    }
    this._poseIsStart = true;
  }

  /** The current camera as a start pose for the settings file: root-local
   * position and rotation (scene units, flip and scale removed), marked
   * `space: "root"` so applyStartPose knows to put them back. */
  captureStartPose() {
    this.root.updateMatrixWorld(true);
    return {
      position: this.root.worldToLocal(this.camera.position.clone()).toArray(),
      quaternion: this.root.quaternion.clone().invert()
        .multiply(this.camera.quaternion).toArray(),
      space: "root",
    };
  }

  applyControlFeel() {
    // inertia=false -> near-zero decay constant = hard stop. Not exactly 0:
    // a zero-length frame would make exp(-0/0) NaN and poison the camera.
    const k = this.settings.controls.inertia ? 0.15 : 1e-6; // Spark default 0.15
    this.controls.pointerControls.rotateInertia = k;
    this.controls.pointerControls.moveInertia = k;
    // NEGATED on purpose. Spark's unflagged direction moves the CAMERA with
    // the pointer, so the scene slides the opposite way to your hand. Every
    // control people arrive with does the reverse — grab the world and it
    // follows the cursor: three's own OrbitControls pans the camera LEFT for a
    // rightward drag (`_panLeft(+deltaX)` negates), and Maya, Blender, Unity
    // and every map do the same. So `reversePan: false` means the normal
    // thing, and ticking it opts into Spark's raw direction.
    this.controls.pointerControls.reverseSlide =
      !this.settings.controls.reversePan;
  }

  flyTo({ to, target, fov = null }) {
    // Kill controls inertia so residual velocity doesn't fight/outlive the fly.
    this.controls.pointerControls.moveVelocity.set(0, 0, 0);
    this.controls.pointerControls.rotateVelocity.set(0, 0, 0);
    this.flyAnim = { from: this.camera.position.clone(),
                     fromQuat: this.camera.quaternion.clone(), t: 0,
                     target, to, fov };
  }

  onResize() {
    this.viewW = this.container.clientWidth;
    this.viewH = this.container.clientHeight;
    this.camera.aspect = this.viewW / this.viewH;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.viewW, this.viewH);
  }

  _tmpQuat = new THREE.Quaternion();

  _tick = () => {
    if (!this._running) return;
    requestAnimationFrame(this._tick);
    const tf0 = performance.now();
    for (const h of this.preHooks) h(tf0);
    const dt = Math.min(this._clock.getDelta(), 0.05);
    const fly = this.flyAnim;
    if (fly) {
      fly.t = Math.min(fly.t + dt * 1.5, 1);
      const k = fly.t * fly.t * (3 - 2 * fly.t);
      this.camera.position.lerpVectors(fly.from, fly.to, k);
      // Ease rotation too: lookAt gives the exact facing pose at the eased
      // position; slerping from the click-time pose by the same k removes
      // the first-frame snap (rotation used to jump instantly).
      this.camera.lookAt(fly.target);
      this._tmpQuat.copy(this.camera.quaternion);
      this.camera.quaternion.slerpQuaternions(fly.fromQuat, this._tmpQuat, k);
      if (fly.t >= 1) {
        // Manual-view jumps restore the captured FOV on arrival, so the
        // frame is exactly what the pipeline renders. Applied THROUGH the
        // app's FOV control (onFlyArrive), so value, UI and camera agree.
        // Cancelling mid-flight skips it.
        if (fly.fov != null) this.onFlyArrive?.(fly.fov);
        this.flyAnim = null;
      }
    } else if (this.controlsEnabled) {
      // Pointer controls stay live (a drag blurs any field first anyway);
      // only the keyboard channel yields to a focused text field.
      this.controls.fpsMovement.enable = !isTypingTarget();
      this.controls.update(this.camera);
    }
    for (const h of this.frameHooks) h(tf0);
    // Overlay occlusion: the splat material WRITES depth — the
    // nearest drawn splat per pixel, since Spark draws back to front — but
    // never tests it (AlwaysDepth), so Spark's radial sort is untouched and
    // no splat is ever rejected. Overlays drawn after the splats with a
    // normal depth test are hidden where a splat is nearer and drawn where
    // they are in front. Spark resets depthWrite on its first frame, so this
    // is re-asserted per frame (a flag read; no cost).
    const sm = this.spark.material;
    if (!sm.depthWrite) {
      sm.depthWrite = true;
      sm.depthFunc = THREE.AlwaysDepth;
    }
    this.renderer.render(this.scene, this.camera);
    this.perf.frames++;
    this.perf.frameMs += performance.now() - tf0;
  };

  start() {
    if (this._running) return;
    this._running = true;
    this._tick();
  }

  stop() {   // the React unmount path
    this._running = false;
  }

  // Everything a scene open took, given back: the splat mesh's packed
  // textures, Spark's sort workers (their own copy of the centres), the
  // WebGL context itself (browsers cap live contexts at about sixteen, and
  // every context kept pinned the scene's textures with it), and the
  // controls' document listeners, which Spark never removes — disabled so
  // they act on nothing. Before this only the renderer was disposed, and
  // each scene opened from the library stayed resident until a reload.
  dispose() {
    this.stop();
    this.flyAnim = null;
    this.frameHooks.length = 0;
    this.preHooks.length = 0;
    this.controlsEnabled = false;
    if (this.controls?.fpsMovement) this.controls.fpsMovement.enable = false;
    if (this.controls?.pointerControls) this.controls.pointerControls.enable = false;
    try { this.splats.dispose?.(); } catch (e) { console.warn("splat dispose:", e); }
    try { this.spark.dispose?.(); } catch (e) { console.warn("spark dispose:", e); }
    this.renderer.dispose();
    this.renderer.forceContextLoss?.();
    this.renderer.domElement.remove();
    this._splatBounds = this._splatSample = this._splatArrays = null;
  }
}
