// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";

/**
 * The ONE TransformControls wrapper. Every 3D manipulator (volume box,
 * focus volume, camera pose) is an adapter over this class that only
 * translates proxy <-> domain; the scaffolding they used to hand-copy —
 * each invariant learned through a shipped defect — lives here once:
 *
 * - The control manipulates a PROXY Object3D, parented where the adapter
 *   says (ply space under the stage root), never the domain object.
 * - The helper goes in the SCENE, not under the (flipped/transformed)
 *   parent: it places itself from the attached object's world matrix, so
 *   parenting it under a transformed group would apply that transform
 *   twice.
 * - dragging-changed drives onDragging (the canvas freezes its camera
 *   there) and fires onRelease when the hand lets go while attached —
 *   commit-on-release, one write per gesture.
 * - handleHot(): a handle under the pointer (hover or drag) owns it — a
 *   TAP on a handle (press and release without moving) must not fall
 *   through to the click-pick and deselect the very thing the handle
 *   belongs to. `anyHandleHot()` answers for every live manipulator, so
 *   the canvas never enumerates them and a new manipulator is covered
 *   the day it is constructed.
 */
export class Manipulator {
  /** what TransformControls actually manipulates; the adapter seeds it
   *  before attach() and reads it back in onChange/onRelease */
  readonly proxy = new THREE.Object3D();
  /** true while a handle is held: the camera must stand still */
  onDragging: ((on: boolean) => void) | null = null;
  /** the hand let go while attached — adapters commit here */
  onRelease: (() => void) | null = null;
  /** live during a drag, after every proxy change */
  onChange: (() => void) | null = null;
  private tc: TransformControls;
  private helper: THREE.Object3D;
  private attached = false;
  private static live = new Set<Manipulator>();

  constructor({ parent, camera, domElement, scene, size = 0.85,
                mode = "translate" }: {
    parent: THREE.Object3D; camera: THREE.Camera; domElement: HTMLElement;
    scene: THREE.Scene; size?: number; mode?: string;
  }) {
    parent.add(this.proxy);
    this.tc = new TransformControls(camera, domElement);
    this.tc.setMode(mode as never);
    this.tc.setSpace("local");
    this.tc.setSize(size);
    this.helper = this.tc.getHelper();
    this.helper.visible = false;
    scene.add(this.helper);
    this.tc.addEventListener("dragging-changed", (e: any) => {
      this.onDragging?.(!!e.value);
      if (!e.value && this.attached) this.onRelease?.();
    });
    this.tc.addEventListener("objectChange", () => this.onChange?.());
    Manipulator.live.add(this);
  }

  setMode(m: string) { this.tc.setMode(m as never); }

  /** Show the handles on the proxy (the adapter seeds it first). */
  attach() {
    this.proxy.updateMatrixWorld(true);
    this.tc.attach(this.proxy);
    this.helper.visible = true;
    this.attached = true;
  }

  detach() {
    if (!this.attached) return;
    this.tc.detach();
    this.helper.visible = false;
    this.attached = false;
  }

  isDragging(): boolean { return (this.tc as any).dragging === true; }

  /** Is a handle under the pointer right now? TransformControls sets
   *  `axis` from its own hover pass, so this is its opinion rather than
   *  a second hit-test. */
  isOverHandle(): boolean { return (this.tc as any).axis != null; }

  handleHot(): boolean { return this.isOverHandle() || this.isDragging(); }

  /** Does ANY live manipulator's handle own the pointer? */
  static anyHandleHot(): boolean {
    for (const m of Manipulator.live) if (m.handleHot()) return true;
    return false;
  }

  dispose() {
    Manipulator.live.delete(this);
    this.tc.detach();
    this.tc.dispose();
    this.helper.removeFromParent();
    this.proxy.removeFromParent();
  }
}
