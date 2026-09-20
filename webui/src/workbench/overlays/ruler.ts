// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import * as THREE from "three";
import { COL } from "./common";
import { pickSurface, type SplatArrays } from "./pick";

// ---------------------------------------------------------------------------
// The ruler: the scale act. The operator clicks two
// points ON THE SPLAT — something whose real length they know: a door, a
// floor tile, a gravestone, a wall they paced — and types that length in
// their unit on the volume panel; the factor is typed length / measured
// distance. Nothing about the scene is inferred from the two points, and
// the act works on any capture the canvas can show, which is every
// capture. Display only: this class draws two markers, the line between
// them and the distance in scene units; the panel owns the numbers and the
// write. Points live in raw .ply space — the FILE frame, under the stage's
// file group like every overlay that describes the file; the labels
// are DOM (screen space, no atlas), placed once per frame by `update`.
//
// The same instrument serves the LEVEL act: in "level" mode up to
// five points are drawn — three on the floor (a triangle) and two along a
// wall (a segment) — labelled floor 1..3 and edge 1, 2; no distance is shown.
// And the SQUARE act: in "square" mode two points along an edge, a
// segment labelled E1, E2.
//
// The clicks land on the scene itself — a splat, not a box or a plane, so
// the measured distance is the distance between two surface points the
// operator can see — through the pick (pick.ts): the median depth of
// what the pixel shows along the click's ray, walked over the stage's
// splat arrays. From the click viewpoint a wrong depth is invisible (the
// marker sits under the cursor either way) and shows only once the camera
// moves, or as a tilted floor; the pick's model is the renderer's, so the
// point is where the colour under the cursor comes from.
//
// The hover (stage 2): the canvas casts one pick when the pointer
// RESTS while armed, and the ruler draws it as a ring before the click —
// white on a surface, amber with the word "thin" on a lone splat. The
// hover keeps the camera pose it was cast from, so `update` retires it the
// frame the camera moves, and its NDC, so a click at the resting pointer
// IS that pick: the ring and the point agree by construction, and the
// click costs no second raycast.
// ---------------------------------------------------------------------------

const COLOUR = 0xff5a4c;
// in "level" mode the edge's two points and their segment wear a
// second colour, so the two acts read apart on the canvas as on the
// panel; the panel's list uses the same two.
const EDGE_COLOUR = 0x2bb5a0;
export const LEVEL_COLOURS = { floor: "#ff5a4c", edge: "#2bb5a0" } as const;

/** What a click on the splat found. */
export interface Pick {
  /** the point in ply (file-frame) space — what the records hold */
  point: number[];
  /** distance along the ray from the camera, world (display) units */
  range: number;
  /** accumulated opacity along the ray at the point */
  opacity: number;
  /** splats that contributed up to the point */
  hits: number;
  /** the accumulation reached the median (a surface); false = something
   *  faint was crossed and nothing behind it — a floater more often than
   *  not */
  solid: boolean;
}

interface Hover { pick: Pick | null; ndc: THREE.Vector2; cam: THREE.Matrix4 }

/** The ring's sprite: a white stroke on a transparent square, tinted by
 *  the material per tone. */
function ringTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  g.strokeStyle = "#fff";
  g.lineWidth = 6;
  g.beginPath();
  g.arc(32, 32, 24, 0, Math.PI * 2);
  g.stroke();
  return new THREE.CanvasTexture(c);
}

export type RulerMode = "ruler" | "level" | "square";

export class Ruler {
  group = new THREE.Group();
  private points: number[][] = [];
  private mode: RulerMode = "ruler";
  private marks: THREE.Points | null = null;
  private line: THREE.Line | null = null;
  private edgeMarks: THREE.Points | null = null;
  private edgeLine: THREE.Line | null = null;
  private labels: HTMLDivElement[] = [];
  private ray = new THREE.Raycaster();
  private hover: Hover | null = null;
  private ring: THREE.Points | null = null;
  private ringLabel: HTMLDivElement | null = null;

  constructor(private root: THREE.Group, private dom: HTMLElement) {
    root.add(this.group);
    this.group.renderOrder = 20;
  }

  /** The points in ply space: two for a measurement, up to five for the
   *  level act (three floor, two wall), two for the square act. */
  setPoints(pts: number[][] | null, mode: RulerMode = "ruler") {
    this.mode = mode;
    const n = mode === "level" ? 5 : 2;
    this.points = (pts ?? []).slice(0, n).map((p) => [p[0], p[1], p[2]]);
    this.rebuild();
  }

  /** Distance between the two points, scene units; null before both. */
  get units(): number | null {
    if (this.points.length < 2) return null;
    const [a, b] = this.points;
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }

  /** What the pixel at `ndc` shows (pick.ts), its point in ply (root-local)
   *  space, or null when the ray meets nothing. `arrays` is the stage's
   *  `splatArrays()` (null before the splats are in). */
  pick(ndc: THREE.Vector2, camera: THREE.Camera,
       arrays: SplatArrays | null): Pick | null {
    // the resting pointer's pick is the hover's: the ring IS the point
    const h = this.hover;
    if (h && h.ndc.equals(ndc) && h.cam.equals(camera.matrixWorld)) return h.pick;
    if (!arrays) return null;
    this.ray.setFromCamera(ndc, camera);
    // the ray into the arrays' frame (the root's local: file frame, under
    // the display scale and the alignment); `t` comes back in its units
    this.root.updateMatrixWorld(true);
    const inv = this.root.matrixWorld.clone().invert();
    const o = this.ray.ray.origin.clone().applyMatrix4(inv);
    const d = this.ray.ray.direction.clone().transformDirection(inv);
    const hit = pickSurface(o.toArray(), d.toArray(), arrays);
    if (!hit) return null;
    const local = o.addScaledVector(d, hit.t);
    const range = this.root.localToWorld(local.clone())
      .distanceTo(this.ray.ray.origin);
    return { point: [local.x, local.y, local.z], range, opacity: hit.opacity,
             hits: hit.hits, solid: hit.solid };
  }

  /** The pick at the resting pointer (`ndc`, cast from `camera`), null for
   *  nothing under it: drawn as the ring until the camera moves, a
   *  clearHover, or the next call. */
  setHover(pick: Pick | null, ndc: THREE.Vector2, camera: THREE.Camera) {
    this.hover = { pick, ndc: ndc.clone(), cam: camera.matrixWorld.clone() };
    this.drawHover();
  }

  clearHover() {
    if (!this.hover) return;
    this.hover = null;
    this.drawHover();
  }

  private drawHover() {
    const pk = this.hover?.pick ?? null;
    if (!pk) {
      if (this.ring) this.ring.visible = false;
      if (this.ringLabel) this.ringLabel.style.display = "none";
      return;
    }
    if (!this.ring) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0], 3));
      this.ring = new THREE.Points(g, new THREE.PointsMaterial({
        map: ringTexture(), size: 26, sizeAttenuation: false,
        depthTest: false, transparent: true }));
      this.ring.renderOrder = 22;
      this.ring.frustumCulled = false;
      this.group.add(this.ring);
      this.ringLabel = this.label("thin", false, {
        background: "rgba(227, 179, 65, 0.95)", color: "#1a1a1a" });
    }
    this.ring.position.set(pk.point[0], pk.point[1], pk.point[2]);
    (this.ring.material as THREE.PointsMaterial).color.setHex(
      pk.solid ? 0xffffff : COL.crop);
    this.ring.visible = true;
    // placed by update; shown there only while the group is
    this.ringLabel!.style.display = pk.solid ? "none" : "";
  }

  setVisible(on: boolean) {
    this.group.visible = on;
    for (const l of this.labels) l.style.display = on ? "" : "none";
    if (!on && this.ringLabel) this.ringLabel.style.display = "none";
  }

  /** Per frame: retire a hover the camera has moved away from, and place
   *  the DOM labels ("A", "B", the distance, "thin") over the projected
   *  points. */
  update(camera: THREE.Camera, w: number, h: number) {
    if (this.hover && !camera.matrixWorld.equals(this.hover.cam))
      this.clearHover();
    const hp = this.hover?.pick;
    if (!this.group.visible || (!this.points.length && !hp)) return;
    this.root.updateMatrixWorld(true);
    const place = (el: HTMLDivElement, p: number[], dx: number, dy: number) => {
      const v = this.root.localToWorld(new THREE.Vector3(p[0], p[1], p[2]))
        .project(camera);
      const behind = v.z > 1;
      el.style.display = behind ? "none" : "";
      el.style.left = `${(v.x + 1) / 2 * w + dx}px`;
      el.style.top = `${(1 - v.y) / 2 * h + dy}px`;
    };
    if (hp && !hp.solid && this.ringLabel) place(this.ringLabel, hp.point, 12, 8);
    if (!this.points.length) return;
    if (this.mode !== "ruler") {
      this.points.forEach((p, i) => place(this.labels[i], p, 8, -22));
      return;
    }
    place(this.labels[0], this.points[0], 8, -22);
    if (this.points.length > 1) {
      place(this.labels[1], this.points[1], 8, -22);
      const m = this.points[0].map((x, i) => (x + this.points[1][i]) / 2);
      place(this.labels[2], m, 10, 6);
    }
  }

  private label(text: string, big: boolean,
                style: Partial<CSSStyleDeclaration> = {}): HTMLDivElement {
    const d = document.createElement("div");
    d.textContent = text;
    Object.assign(d.style, {
      position: "absolute", pointerEvents: "none", zIndex: "7",
      font: `${big ? 600 : 700} ${big ? 13 : 11}px ui-monospace, monospace`,
      color: "#fff", background: "rgba(255, 90, 76, 0.92)",
      padding: big ? "2px 7px" : "1px 5px", borderRadius: "3px",
      whiteSpace: "nowrap", lineHeight: "1.3",
      ...style,
    } as Partial<CSSStyleDeclaration>);
    this.dom.appendChild(d);
    return d;
  }

  private clear() {
    if (this.marks) {
      this.marks.geometry.dispose();
      (this.marks.material as THREE.Material).dispose();
      this.group.remove(this.marks);
      this.marks = null;
    }
    if (this.line) {
      this.line.geometry.dispose();
      (this.line.material as THREE.Material).dispose();
      this.group.remove(this.line);
      this.line = null;
    }
    for (const o of [this.edgeMarks, this.edgeLine]) {
      if (!o) continue;
      o.geometry.dispose();
      (o.material as THREE.Material).dispose();
      this.group.remove(o);
    }
    this.edgeMarks = this.edgeLine = null;
    for (const l of this.labels) l.remove();
    this.labels = [];
  }

  private rebuild() {
    this.clear();
    if (!this.points.length) return;
    // level: the floor's three points here; the edge's two below, in
    // their own colour
    const own = this.mode === "level" ? this.points.slice(0, 3) : this.points;
    const flat = own.flat();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(flat, 3));
    // Screen-constant markers, drawn through the scene: a measurement must
    // stay visible when the camera moves behind a wall.
    this.marks = new THREE.Points(g, new THREE.PointsMaterial({
      color: COLOUR, size: 11, sizeAttenuation: false, depthTest: false,
      transparent: true }));
    this.marks.renderOrder = 21;
    this.group.add(this.marks);
    if (this.mode !== "ruler") {
      // level: the three floor points close into a triangle, the two edge
      // points are a segment; square: the two edge points are a segment
      const names = this.mode === "level"
        ? ["floor 1", "floor 2", "floor 3", "edge 1", "edge 2"]
        : ["edge 1", "edge 2"];
      const edgeStyle = { background: "rgba(43, 181, 160, 0.92)" };
      this.points.forEach((_, i) => this.labels.push(this.label(
        names[i], false, this.mode === "level" && i >= 3 ? edgeStyle : {})));
      const seg: number[] = [];
      const P = this.points;
      const add = (a: number, b: number) => seg.push(...P[a], ...P[b]);
      if (P.length >= 2) add(0, 1);
      if (this.mode === "level" && P.length >= 3) { add(1, 2); add(2, 0); }
      if (this.mode === "level" && P.length >= 4) {
        const eg = new THREE.BufferGeometry();
        eg.setAttribute("position",
                        new THREE.Float32BufferAttribute(P.slice(3, 5).flat(), 3));
        this.edgeMarks = new THREE.Points(eg, new THREE.PointsMaterial({
          color: EDGE_COLOUR, size: 11, sizeAttenuation: false,
          depthTest: false, transparent: true }));
        this.edgeMarks.renderOrder = 21;
        this.group.add(this.edgeMarks);
        if (P.length >= 5) {
          const lg = new THREE.BufferGeometry();
          lg.setAttribute("position",
                          new THREE.Float32BufferAttribute([...P[3], ...P[4]], 3));
          this.edgeLine = new THREE.Line(lg, new THREE.LineBasicMaterial({
            color: EDGE_COLOUR, depthTest: false, transparent: true }));
          this.edgeLine.renderOrder = 20;
          this.group.add(this.edgeLine);
        }
      }
      if (seg.length) {
        const lg = new THREE.BufferGeometry();
        lg.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
        this.line = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
          color: COLOUR, depthTest: false, transparent: true })) as unknown as THREE.Line;
        this.line.renderOrder = 20;
        this.group.add(this.line);
      }
      return;
    }
    this.labels.push(this.label("A", false));
    if (this.points.length > 1) {
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", new THREE.Float32BufferAttribute(flat, 3));
      this.line = new THREE.Line(lg, new THREE.LineBasicMaterial({
        color: COLOUR, depthTest: false, transparent: true }));
      this.line.renderOrder = 20;
      this.group.add(this.line);
      this.labels.push(this.label("B", false));
      this.labels.push(this.label(
        `${Number(this.units!.toPrecision(4))} units`, true));
    }
  }

  dispose() {
    this.clear();
    this.hover = null;
    if (this.ring) {
      const m = this.ring.material as THREE.PointsMaterial;
      m.map?.dispose();
      m.dispose();
      this.ring.geometry.dispose();
      this.ring = null;
    }
    this.ringLabel?.remove();
    this.ringLabel = null;
    this.root.remove(this.group);
  }
}
