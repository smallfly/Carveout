// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import * as THREE from "three";
import { COL, type VolumeFrame } from "./common";

// ---------------------------------------------------------------------------
// origin grid — the "infinite" reference grid every 3D package draws at the
// world origin: 1 m minor lines, 10 m major lines, the two ground axes in
// their gizmo colours, fading with distance so the edge of the quad is never
// seen. It lies on the detected floor once one is known (the proposal's,
// or the estimate the start pose reads off the splats before any), and at
// height 0 of the up axis until then: the grid
// IS the floor of the 3D view, so the two are one surface. Its height is a CONVENTION
// (configs/default.yaml scene.up_axis) the file does not record — so it is
// drawn from the moment the scene opens, before any proposal, and a grid
// standing on its side is the operator's first sign that the scene is not
// oriented the way Carveout reads it. It never moves: the detected floor is
// a separate, editable overlay (floorPlane.ts), and the gap between the two
// is the floor calibration made visible.
// ---------------------------------------------------------------------------

const GRID_HALF = 300;      // quad half-size, m — see rebuild() for why not larger
const FADE_MAX = 240;       // m; must end inside the quad, whose edge is never drawn
const MINOR = 1;            // m
const MAJOR = 10;           // m
const AXIS_COL = [0xe5534b, 0x7ee787, 0x56d4dd];   // x, y, z — the view gizmo's

const VERT = /* glsl */ `
  uniform vec3 uOffset;     // the quad's scene-frame offset (it follows the camera)
  uniform vec3 uA0;         // unit vector of the first ground axis
  uniform vec3 uA1;         // ... second
  varying vec2 vG;
  varying vec3 vWorld;
  void main() {
    vec3 lp = position + uOffset;
    vG = vec2(dot(lp, uA0), dot(lp, uA1));
    vec4 w = modelMatrix * vec4(position, 1.0);
    vWorld = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;

const FRAG = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uColA0;
  uniform vec3 uColA1;
  uniform float uMinor;
  uniform float uMajor;
  uniform float uFade;
  varying vec2 vG;
  varying vec3 vWorld;
  // 1 on a grid line of spacing s (about 1 px wide, screen-space AA), 0 off it
  float line(vec2 p, float s) {
    vec2 r = p / s;
    vec2 g = abs(fract(r - 0.5) - 0.5) / fwidth(r);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }
  void main() {
    float a = max(line(vG, uMinor) * 0.28, line(vG, uMajor) * 0.6);
    vec3 col = uColor;
    // the ground axes: vG.y == 0 runs along a0, vG.x == 0 along a1
    vec2 ax = 1.0 - min(abs(vG) / (fwidth(vG) * 1.5), 1.0);
    col = mix(col, uColA0, ax.y);
    col = mix(col, uColA1, ax.x);
    a = max(a, max(ax.x, ax.y) * 0.85);
    float d = distance(vWorld, cameraPosition);
    a *= 1.0 - smoothstep(uFade * 0.35, uFade, d);
    if (a < 0.004) discard;
    gl_FragColor = vec4(col, a);
  }`;

export class OriginGrid {
  group = new THREE.Group();
  private mesh: THREE.Mesh | null = null;
  private material: THREE.ShaderMaterial;
  private frame: VolumeFrame = { up: 1, upSign: 1, a0: 0, a1: 2 };
  private scale = 1;         // display metres per scene unit (the root's)
  private unit = 1;          // scene units per display metre = 1 / scale
  private fadeHalf = 0;      // last footprint half-extent, scene units
  private height: number | null = null;   // SIGNED floor height, scene units

  constructor(private root: THREE.Group) {
    root.add(this.group);
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG,
      uniforms: {
        uColor: { value: new THREE.Color(COL.floor) },
        uColA0: { value: new THREE.Color(AXIS_COL[0]) },
        uColA1: { value: new THREE.Color(AXIS_COL[2]) },
        uMinor: { value: MINOR }, uMajor: { value: MAJOR },
        uFade: { value: 60 },
        uOffset: { value: new THREE.Vector3() },
        uA0: { value: new THREE.Vector3(1, 0, 0) },
        uA1: { value: new THREE.Vector3(0, 0, 1) },
      },
      transparent: true, depthWrite: false, side: THREE.DoubleSide });
    this.rebuild();
  }

  /** Only the up axis matters (the grid lies at height 0); upSign is
   * carried for interface symmetry with the floor plane. */
  setFrame(f: VolumeFrame) { this.frame = f; this.rebuild(); this.place(); }
  /** The grid's height along the up axis, SIGNED (the floor plane's
   *  convention); null = the origin. */
  setHeight(signed: number | null) { this.height = signed; this.place(); }
  private place(): void {
    const p = [0, 0, 0];
    p[this.frame.up] = (this.height ?? 0) * this.frame.upSign;
    this.group.position.set(p[0], p[1], p[2]);
  }
  /** Fade distance from the volume footprint's half-extent (scene units),
   * so a small scene's grid fades near it and a large one's reaches
   * further. The shader compares it against a WORLD-space distance, so it
   * is stated in world (display metres): the half-extent crosses the
   * root's scale, the floor and cap are display-metre figures. */
  setFade(half: number) {
    this.fadeHalf = half;
    this.material.uniforms.uFade.value = Math.min(
      Math.max(half * 6 * this.scale, 60), FADE_MAX);
  }
  /** Display metres per scene unit (the root's uniform scale): the quad and
   * the 1 m / 10 m line spacing are drawn in the group's LOCAL (scene-unit)
   * space, so they are divided by it to stay metre figures on screen. */
  setScale(s: number) {
    this.scale = s > 0 ? s : 1;
    this.unit = 1 / this.scale;
    this.material.uniforms.uMinor.value = MINOR * this.unit;
    this.material.uniforms.uMajor.value = MAJOR * this.unit;
    this.setFade(this.fadeHalf);
    this.rebuild();
  }
  setVisible(on: boolean) { this.group.visible = on; }

  private rebuild(): void {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.group.remove(this.mesh);
      this.mesh = null;
    }
    const { a0, a1 } = this.frame;
    // A modest quad that FOLLOWS the camera, not one huge quad: at a 10 km
    // half-size the two triangles lost interpolation precision and each
    // drifted into its own plane — two grids at first load (reproduced
    // headlessly; 200 m drew one). The fade ends inside
    // the edge, so it reads as infinite from anywhere.
    const corner = (d0: number, d1: number): number[] => {
      const p = [0, 0, 0];
      p[a0] = d0 * GRID_HALF * this.unit;
      p[a1] = d1 * GRID_HALF * this.unit;
      return p;
    };
    const c = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    const pos = [c[0], c[1], c[2], c[0], c[2], c[3]].flat();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const u = this.material.uniforms;
    u.uColA0.value.set(AXIS_COL[a0]);
    u.uColA1.value.set(AXIS_COL[a1]);
    u.uA0.value.set(0, 0, 0).setComponent(a0, 1);
    u.uA1.value.set(0, 0, 0).setComponent(a1, 1);
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    // After the splats, depth-tested: the splat pass writes depth (see
    // stage.js), so the grid is hidden where a splat is nearer and drawn
    // where it is in front. (Drawing it BEFORE the splats hid it under any
    // splat regardless of depth.)
    this.mesh.renderOrder = 2;
    const camLocal = new THREE.Vector3();
    this.mesh.onBeforeRender = (_r, _s, camera) => {
      // re-centre under the camera in the scene frame (the group sits at the
      // root's identity; the root carries the display flip)
      camera.getWorldPosition(camLocal);
      this.group.worldToLocal(camLocal);
      const m = this.mesh!;
      m.position.set(0, 0, 0);
      m.position.setComponent(a0, camLocal.getComponent(a0));
      m.position.setComponent(a1, camLocal.getComponent(a1));
      u.uOffset.value.copy(m.position);
      // onBeforeRender runs before the model matrix is read for this draw,
      // so refreshing it here keeps the pattern and the quad in the same
      // frame — otherwise the grid swims by one frame's camera motion.
      m.updateMatrixWorld(true);
    };
    this.group.add(this.mesh);
  }

  dispose(): void {
    if (this.mesh) this.mesh.geometry.dispose();
    this.material.dispose();
    this.root.remove(this.group);
  }
}
