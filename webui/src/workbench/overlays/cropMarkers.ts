// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import * as THREE from "three";
import type { FrameEntry } from "../../types";
import { COL } from "./common";

// ---------------------------------------------------------------------------
// exemplar crop source markers (gate 4 provenance made spatial)
// ---------------------------------------------------------------------------
export class CropMarkers {
  group = new THREE.Group();
  private objs: THREE.Object3D[] = [];
  private centers: { frameIdx: number; center: THREE.Vector3 }[] = [];

  constructor(private root: THREE.Group) {
    root.add(this.group);
  }

  /** Marker centers within px of a screen point -> the crop's SOURCE
   * frame (clicking a yellow rectangle selects the frame it was drawn
   * on — crop provenance is navigable, not just visible). */
  pick(px: number, py: number, camera: THREE.Camera, viewW: number,
       viewH: number, radius = 22): number | null {
    if (!this.group.visible) return null;
    const v = new THREE.Vector3();
    let best: number | null = null, bestD = radius;
    for (const c of this.centers) {
      v.copy(c.center).applyMatrix4(this.root.matrixWorld).project(camera);
      if (v.z > 1 || v.z < -1) continue;
      const sx = (v.x * 0.5 + 0.5) * viewW;
      const sy = (-v.y * 0.5 + 0.5) * viewH;
      const d = Math.hypot(sx - px, sy - py);
      if (d < bestD) { bestD = d; best = c.frameIdx; }
    }
    return best;
  }

  set(crops: { frame: FrameEntry; box_xyxy: number[] }[],
      intr: { width: number; height: number; fx: number; fy: number },
      stale = false) {
    this.clear();
    for (const { frame: fr, box_xyxy } of crops) {
      const w = fr.width ?? intr.width, h = fr.height ?? intr.height;
      const fx = fr.fx ?? intr.fx, fy = fr.fy ?? intr.fy;
      const d = 0.3 / this.displayScale;   // 0.3 display metres ahead of the camera
      const px = (u: number) => ((u - w / 2) * d) / fx;
      const py = (v: number) => ((v - h / 2) * d) / fy;
      const [x1, y1, x2, y2] = box_xyxy;
      const cs = [new THREE.Vector3(px(x1), py(y1), d),
                  new THREE.Vector3(px(x2), py(y1), d),
                  new THREE.Vector3(px(x2), py(y2), d),
                  new THREE.Vector3(px(x1), py(y2), d),
                  new THREE.Vector3(px(x1), py(y1), d)];
      const m = new THREE.Matrix4().fromArray(
        (fr.c2w as number[][]).flat()).transpose();
      const geo = new THREE.BufferGeometry().setFromPoints(
        cs.map((c) => c.applyMatrix4(m)));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: stale ? COL.stale : COL.crop,
        transparent: stale, opacity: stale ? 0.4 : 1 }));
      this.group.add(line);
      this.objs.push(line);
      const center = new THREE.Vector3();
      for (let i = 0; i < 4; i++) center.add(cs[i]);
      this.centers.push({ frameIdx: fr.frame_idx,
                          center: center.multiplyScalar(0.25) });
    }
  }

  setVisible(on: boolean) { this.group.visible = on; }
  /** display metres per scene unit (the root's scale): the markers sit a
   *  display-metre distance in front of their cameras, in scene units */
  private displayScale = 1;
  setScale(s: number) { this.displayScale = s > 0 ? s : 1; }

  clear() {
    for (const o of this.objs) {
      (o as THREE.Line).geometry.dispose();
      ((o as THREE.Line).material as THREE.Material).dispose();
      this.group.remove(o);
    }
    this.objs = [];
    this.centers = [];
  }

  dispose() { this.clear(); this.root.remove(this.group); }
}
