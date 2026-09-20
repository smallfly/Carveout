// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Manual detection-view capture math. Poses are stored in the FILE
// frame (raw .ply coordinates):
// the pi-flip and the display scale sit on the root group and the scene
// alignment on the file group under it (stage.js), so world != ply space.
// Capture converts through the file group's world transform, jump-to
// converts back. The look-at target is the splat
// surface under the view axis (one Spark raycast, ~150 ms — a one-off
// hitch on a button click), falling back to flyDistance along the forward
// axis on a miss; target_source records which.

import * as THREE from "three";

const _ray = new THREE.Raycaster();

export function captureView({ stage, settings, label }) {
  const { camera, splats, viewW, viewH } = stage;
  const root = stage.fileGroup ?? stage.root;
  root.updateMatrixWorld(true);
  const rootQuat = root.getWorldQuaternion(new THREE.Quaternion());
  const fwd = camera.getWorldDirection(new THREE.Vector3());
  let dist = settings.camera.flyDistance;
  let targetSource = "fallback";
  if (splats.isInitialized) {
    try {
      _ray.set(camera.position, fwd);
      _ray.far = Infinity;
      const hits = _ray.intersectObject(splats, false);
      if (hits.length) { dist = hits[0].distance; targetSource = "raycast"; }
    } catch { /* fallback */ }
  }
  const targetWorld = camera.position.clone().addScaledVector(fwd, dist);
  return {
    id: crypto.randomUUID(),
    label,
    position: root.worldToLocal(camera.position.clone()).toArray(),
    quaternion: rootQuat.invert().multiply(camera.quaternion).toArray(),
    target: root.worldToLocal(targetWorld).toArray(),
    target_source: targetSource,
    fov_deg: camera.fov,
    aspect: camera.aspect,
    resolution: [viewW, viewH],
  };
}

// Jump-to reuses the stage fly animation: the controls pin roll, so its
// lookAt(target) reproduces the captured quaternion.
export function jumpToView(stage, v) {
  const root = stage.fileGroup ?? stage.root;
  const to = root.localToWorld(new THREE.Vector3().fromArray(v.position));
  const target = root.localToWorld(new THREE.Vector3().fromArray(v.target));
  stage.flyTo({ to, target, fov: v.fov_deg });
}
