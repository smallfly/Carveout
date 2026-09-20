// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import * as THREE from "three";
import { COL, type VolumeFrame } from "./common";

// ---------------------------------------------------------------------------
// detected floor (calibration) — a translucent plane at the scene frame's
// floor height, clipped to the volume footprint. Display only: the value is
// edited in the volume panel's Floor field (auto readback, reset), and the
// plane follows the reload. The floor is a scene-frame VALUE (camera eye
// height and the obstacle slab hang off it; the proposal puts the box
// bottom a margin below it), not a face of the volume box: the two are
// drawn separately and drift apart freely once the box is edited. The
// floor mis-election (a workbench top winning the floor vote) is invisible
// as a number but obvious as a plane hovering mid-scene. History: a
// draggable sphere handle once rode this plane (a
// sphere is not a gizmo, and the field does the job — removed rather than
// upgraded); the infinite reference grid lives in originGrid.ts.
// ---------------------------------------------------------------------------

export class FloorPlane {
  group = new THREE.Group();
  private mesh: THREE.Mesh | null = null;
  private frame: VolumeFrame = { up: 1, upSign: 1, a0: 0, a1: 2 };
  private floor = 0;              // SIGNED up-axis height (matches frame.floor)
  private center = [0, 0, 0];    // world ground centre of the plane
  private half = 6;              // world half-size

  constructor(private root: THREE.Group) {
    root.add(this.group);
  }

  setFrame(f: VolumeFrame) { this.frame = f; this.rebuild(); }
  setFloor(signed: number) { this.floor = signed; this.rebuild(); }
  setExtent(center: number[], half: number) {
    this.center = [center[0], center[1], center[2]];
    this.half = Math.max(half, 1);
    this.rebuild();
  }
  setVisible(on: boolean) { this.group.visible = on; }

  private rebuild(): void {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.group.remove(this.mesh);
      this.mesh = null;
    }
    const { up, a0, a1 } = this.frame;
    const wf = this.floor * this.frame.upSign;
    const corner = (d0: number, d1: number): number[] => {
      const p = [0, 0, 0];
      p[up] = wf;
      p[a0] = this.center[a0] + d0 * this.half;
      p[a1] = this.center[a1] + d1 * this.half;
      return p;
    };
    const c = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    const pos = [c[0], c[1], c[2], c[0], c[2], c[3]].flat();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    this.mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: COL.floorFill, transparent: true, opacity: 0.14,
      depthWrite: false, side: THREE.DoubleSide }));
    // After the splats, depth-tested (the splat pass writes depth — see
    // stage.js): the plane shows where it is in front of the scene and is
    // hidden where a splat is nearer, so a floor that sits ON the real floor
    // reads as a faint tint and one hovering mid-room reads as a sheet.
    this.mesh.renderOrder = 2;
    this.group.add(this.mesh);
  }

  dispose(): void {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
    }
    this.root.remove(this.group);
  }
}
