// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Camera frustums (auto + manual, selectable).
//
// Visibility: LineBasicMaterial.linewidth
// is ignored on WebGL — every frustum was a 1px hairline that vanished
// against bright splats. Now each frustum is drawn with screen-space fat
// lines (Line2 family) in three passes sharing one geometry:
//   halo   — #0A0B0C under-stroke, width+2.5px, 55% (legible over bright
//            AND dark regions without glow: an outline, not a bloom)
//   core   — accent, 1.75px (2.5px selected/hovered), depth-tested
//   behind — the occluded portions at 25% via depthFunc GREATER — frusta
//            behind geometry stay readable but clearly behind
// plus a 4px screen-space apex dot (the wire pyramid reads badly
// edge-on; the dot always reads), ringed when selected.

import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from
  "three/addons/lines/LineSegmentsGeometry.js";
import type { FrameEntry } from "../../types";
import { COL } from "./common";

// Round sprite for screen-space apex dots (Points render square without
// a map); ring variant marks the selected camera.
function dotTexture(ring: boolean): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d")!;
  g.beginPath();
  g.arc(16, 16, ring ? 12 : 14, 0, Math.PI * 2);
  if (ring) {
    g.lineWidth = 5;
    g.strokeStyle = "#fff";
    g.stroke();
  } else {
    g.fillStyle = "#fff";
    g.fill();
  }
  return new THREE.CanvasTexture(c);
}

// ---------------------------------------------------------------------------
// frustums
// ---------------------------------------------------------------------------
export class FrustumSet {
  group = new THREE.Group();
  private items: { frame: FrameEntry; grp: THREE.Group;
                   apex: THREE.Vector3; halo: LineMaterial;
                   core: LineMaterial; behind: LineMaterial;
                   dot: THREE.PointsMaterial }[] = [];
  selected = -1;    // frame_idx
  hovered = -1;     // frame_idx (raycast hover — width/weight only)
  stale = false;
  /** frames whose view was deleted since the render: drawn as
   *  stale until the re-render drops them */
  deleted = new Set<number>();
  // Display size (m): small by default so dense capture sets stay
  // readable (operator feedback, first look); adjusted via the slider.
  // Line WIDTH is a legibility constant, not user-tweakable (P3.7).
  scale = 1;
  private baseDepth = 0.1;
  private lastFrames: FrameEntry[] = [];
  private lastIntr: { width: number; height: number; fx: number;
                      fy: number } | null = null;
  private res = new THREE.Vector2(800, 600);
  private dotTex = dotTexture(false);
  private ringTex = dotTexture(true);
  private ring: THREE.Points;

  constructor(private root: THREE.Group) {
    root.add(this.group);
    this.ring = new THREE.Points(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]),
      new THREE.PointsMaterial({ size: 11, sizeAttenuation: false,
                                 map: this.ringTex, transparent: true,
                                 alphaTest: 0.4, depthTest: false }));
    this.ring.renderOrder = 10;
    this.ring.visible = false;
    this.group.add(this.ring);
  }

  /** Screen-space line materials must know the viewport size. */
  setResolution(w: number, h: number) {
    if (this.res.x === w && this.res.y === h) return;
    this.res.set(w, h);
    for (const it of this.items) {
      it.halo.resolution.set(w, h);
      it.core.resolution.set(w, h);
      it.behind.resolution.set(w, h);
    }
  }

  setScale(s: number) {
    this.scale = s;
    if (this.lastIntr) this.setFrames(this.lastFrames, this.lastIntr);
  }

  setFrames(frames: FrameEntry[],
            intr: { width: number; height: number; fx: number;
                    fy: number }) {
    // clear() forgets the cached set, so the cache is written AFTER it.
    this.clear();
    this.lastFrames = frames;
    this.lastIntr = intr;
    for (const fr of frames) {
      const w = fr.width ?? intr.width, h = fr.height ?? intr.height;
      const fx = fr.fx ?? intr.fx, fy = fr.fy ?? intr.fy;
      const d = this.baseDepth * this.scale;  // display only
      const x = (d * w) / (2 * fx), y = (d * h) / (2 * fy);
      // OpenCV c2w: +z forward, +y down — corners in camera space
      const c = [new THREE.Vector3(-x, -y, d), new THREE.Vector3(x, -y, d),
                 new THREE.Vector3(x, y, d), new THREE.Vector3(-x, y, d)];
      const m = new THREE.Matrix4().fromArray(
        (fr.c2w as number[][]).flat()).transpose();
      const apex = new THREE.Vector3().setFromMatrixPosition(m);
      const pos: number[] = [];
      for (let i = 0; i < 4; i++) {
        const a = c[i].clone().applyMatrix4(m);
        const b = c[(i + 1) % 4].clone().applyMatrix4(m);
        pos.push(apex.x, apex.y, apex.z, a.x, a.y, a.z,
                 a.x, a.y, a.z, b.x, b.y, b.z);
      }
      const geo = new LineSegmentsGeometry().setPositions(pos);
      const mk = (opts: ConstructorParameters<typeof LineMaterial>[0]) => {
        const mat = new LineMaterial({ transparent: true,
                                       depthWrite: false,
                                       worldUnits: false, ...opts });
        mat.resolution.copy(this.res);
        return mat;
      };
      const halo = mk({ color: COL.halo, linewidth: 4.25, opacity: 0.55 });
      const core = mk({ color: COL.auto, linewidth: 1.75, opacity: 1 });
      const behind = mk({ color: COL.auto, linewidth: 1.75, opacity: 0.25,
                          depthFunc: THREE.GreaterDepth });
      const grp = new THREE.Group();
      for (const [mat, order] of [[behind, 6], [halo, 7], [core, 8]] as
           const) {
        const seg = new LineSegments2(geo, mat);
        seg.renderOrder = order;
        grp.add(seg);
      }
      const dot = new THREE.PointsMaterial({
        size: 4, sizeAttenuation: false, map: this.dotTex,
        transparent: true, alphaTest: 0.4, depthTest: false });
      const pts = new THREE.Points(
        new THREE.BufferGeometry().setFromPoints([apex]), dot);
      pts.renderOrder = 9;
      grp.add(pts);
      this.group.add(grp);
      this.items.push({ frame: fr, grp, apex, halo, core, behind, dot });
    }
    this.applyStyle();
  }

  applyStyle() {
    let selApex: THREE.Vector3 | null = null;
    let selVisible = false;
    for (const it of this.items) {
      const manual = it.frame.provenance?.startsWith("manual:");
      const sel = it.frame.frame_idx === this.selected;
      const hot = it.frame.frame_idx === this.hovered;
      const stale = this.stale || this.deleted.has(it.frame.frame_idx);
      const color = stale ? COL.stale
        : sel ? COL.selected : manual ? COL.manual : COL.auto;
      it.core.color.setHex(color);
      it.behind.color.setHex(color);
      it.dot.color.setHex(color);
      const w = sel || hot ? 2.5 : 1.75;
      it.core.linewidth = w;
      it.behind.linewidth = w;
      it.halo.linewidth = w + 2.5;
      it.core.opacity = stale ? 0.35 : 1;
      it.behind.opacity = stale ? 0.1 : 0.25;
      it.halo.opacity = stale ? 0.25 : 0.55;
      if (sel) {
        selApex = it.apex;
        selVisible = it.grp.visible;
      }
    }
    if (selApex) this.ring.position.copy(selApex);
    this.ring.visible = !!selApex && selVisible;
  }

  select(frameIdx: number | null) {
    this.selected = frameIdx ?? -1;
    this.applyStyle();
  }

  setHover(frameIdx: number | null) {
    const v = frameIdx ?? -1;
    if (v === this.hovered) return;
    this.hovered = v;
    this.applyStyle();
  }

  setVisible(on: boolean) { this.group.visible = on; }

  setStale(on: boolean) { this.stale = on; this.applyStyle(); }

  setDeleted(ids: Iterable<number>) {
    this.deleted = new Set(ids);
    this.applyStyle();
  }

  /** Frusta whose apex is at (or nearly at) the camera render as giant
   * lines across the whole view (you are standing INSIDE them — e.g.
   * right after fly-to-frame). Hidden per frame; camPos in ply space. */
  updateProximity(camLocal: THREE.Vector3, minDist = 0.35) {
    let selVisible = false;
    for (const it of this.items) {
      const vis = it.apex.distanceTo(camLocal) > minDist * this.scale;
      if (it.grp.visible !== vis) it.grp.visible = vis;
      if (it.frame.frame_idx === this.selected && vis) selVisible = true;
    }
    this.ring.visible = this.selected >= 0 && selVisible;
  }

  /** Apexes within px of a screen point, nearest first. Aimed coverage
   * rounds genuinely share one position with several orientations, so
   * repeated clicks CYCLE through the coincident candidates. */
  pick(px: number, py: number, camera: THREE.Camera, viewW: number,
       viewH: number, radius = 26): number | null {
    if (!this.group.visible) return null;
    const v = new THREE.Vector3();
    const hits: { idx: number; d: number }[] = [];
    for (const it of this.items) {
      if (!it.grp.visible) continue;
      v.copy(it.apex).applyMatrix4(this.root.matrixWorld).project(camera);
      if (v.z > 1 || v.z < -1) continue;
      const sx = (v.x * 0.5 + 0.5) * viewW;
      const sy = (-v.y * 0.5 + 0.5) * viewH;
      const d = Math.hypot(sx - px, sy - py);
      if (d < radius) hits.push({ idx: it.frame.frame_idx, d });
    }
    if (!hits.length) return null;
    hits.sort((a, b) => a.d - b.d);
    const cur = hits.findIndex((h) => h.idx === this.selected);
    return hits[(cur + 1) % hits.length].idx;   // cur=-1 -> nearest
  }

  /** Nearest apex within radius — the raycast HOVER (no cycling). */
  hoverAt(px: number, py: number, camera: THREE.Camera, viewW: number,
          viewH: number, radius = 22): number | null {
    if (!this.group.visible) return null;
    const v = new THREE.Vector3();
    let best: number | null = null, bestD = radius;
    for (const it of this.items) {
      if (!it.grp.visible) continue;
      v.copy(it.apex).applyMatrix4(this.root.matrixWorld).project(camera);
      if (v.z > 1 || v.z < -1) continue;
      const sx = (v.x * 0.5 + 0.5) * viewW;
      const sy = (-v.y * 0.5 + 0.5) * viewH;
      const d = Math.hypot(sx - px, sy - py);
      if (d < bestD) { bestD = d; best = it.frame.frame_idx; }
    }
    return best;
  }

  /** Screen anchors for the HTML tag layer — EVERY visible frustum gets a
   * chip (operator request: auto frusta need labels for reading AND
   * clicking; coincident tangles are de-overlapped by the tag layer).
   * Unselected chips stay terse ("M" / index); the selected one carries
   * the full label. Proximity-hidden frusta get no chip. */
  anchors(camera: THREE.Camera, viewW: number, viewH: number) {
    const v = new THREE.Vector3();
    const out: { frameIdx: number; x: number; y: number; manual: boolean;
                 selected: boolean; hovered: boolean;
                 label?: string }[] = [];
    for (const it of this.items) {
      if (!it.grp.visible) continue;
      const manual = !!it.frame.provenance?.startsWith("manual:");
      const selected = it.frame.frame_idx === this.selected;
      const hovered = it.frame.frame_idx === this.hovered;
      v.copy(it.apex).applyMatrix4(this.root.matrixWorld).project(camera);
      if (v.z > 1 || v.z < -1 || Math.abs(v.x) > 1 || Math.abs(v.y) > 1)
        continue;
      out.push({ frameIdx: it.frame.frame_idx,
                 x: (v.x * 0.5 + 0.5) * viewW,
                 y: (-v.y * 0.5 + 0.5) * viewH,
                 manual, selected, hovered, label: it.frame.label });
    }
    return out;
  }

  clear() {
    for (const it of this.items) {
      it.grp.traverse((o) => {
        const g = (o as THREE.Mesh).geometry as THREE.BufferGeometry;
        // the three LineSegments2 share one geometry — dispose is
        // idempotent, so a triple call is harmless
        g?.dispose?.();
        ((o as THREE.Mesh).material as THREE.Material)?.dispose?.();
      });
      this.group.remove(it.grp);
    }
    this.items = [];
    // Forget the cached set too: setScale() rebuilds from lastFrames, and
    // a clear() that kept it was undone by the next display-scale pass.
    // Resetting the render step cleared the frustums, then the journal
    // moved, /volume refetched, the canvas re-applied its scale — and the
    // deleted render's cameras came straight back. No frames means
    // nothing to rebuild from.
    this.lastFrames = [];
    this.lastIntr = null;
    // The selection ring is a child of the group, not of an item, so it
    // survived a clear() and left a marker dot floating over a scene with
    // no cameras in it. Nothing can be selected once there are no items.
    this.selected = -1;
    this.ring.visible = false;
  }

  dispose() {
    this.clear();
    this.dotTex.dispose();
    this.ringTex.dispose();
    (this.ring.material as THREE.Material).dispose();
    this.ring.geometry.dispose();
    this.root.remove(this.group);
  }
}
