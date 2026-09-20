// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Per-splat instance tint on selection.
// Degrades to no-op when instance_ids.bin is absent (pre-lift) or the
// packedSplats edit path throws.

import * as THREE from "three";

export function createHighlighter({ splats, instanceIds, extObjects,
                                    colorOf }) {
  let savedTint = null;
  let ids = instanceIds;

  function tint(instId, on) {
    if (!ids || !splats.isInitialized) return;
    try {
      if (savedTint) {
        for (const s of savedTint)
          splats.packedSplats.setSplat(s.index, s.center, s.scales,
                                       s.quaternion, s.opacity, s.color);
        savedTint = null;
      }
      if (on) {
        const extId = extObjects[instId]?.instance_id;
        const tintColor = new THREE.Color(colorOf(instId));
        const saved = [];
        splats.forEachSplat((index, center, scales, quaternion, opacity,
                             color) => {
          if (ids[index] !== extId) return;
          saved.push({ index, center: center.clone(),
                       scales: scales.clone(),
                       quaternion: quaternion.clone(), opacity,
                       color: color.clone() });
          splats.packedSplats.setSplat(index, center, scales, quaternion,
                                       opacity,
                                       color.clone().lerp(tintColor, 0.65));
        });
        savedTint = saved;
      }
      // Upload the edits (updateVersion was a wrong guess):
      splats.packedSplats.needsUpdate = true;
      splats.updateGenerator?.();
    } catch (e) {
      console.warn("per-splat tint unavailable:", e);
      ids = null;
    }
  }

  return { tint };
}
