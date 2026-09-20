// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Where a scene opens when no start pose was saved. Filmed inside or
// on the ground: a person standing in the scene — eye height over the
// floor, inside the volume's footprint, looking across it. Filmed around
// an object: the object framed as a person looks at a thing on a bench —
// from a diagonal, a little above it, as close as the whole of it fits in
// the view. Before a proposal exists the frame is the profile's up-axis
// convention and the floor an estimate from the splats themselves, so a
// walkable scene opens standing from its first frame. Nothing here is a
// measurement the pipeline uses; it is the first thing the operator sees,
// so it follows the answers they gave.

import type { VolumeBox } from "../types";
import type { VolumeFrame } from "./overlays/common";

/** Axis-aligned bounds in the SCENE frame (stage root local): the volume
 *  boxes, or the splats' sample turned through the alignment. */
export interface PlyBounds { lo: number[]; hi: number[] }

export const boundsOfBoxes = (boxes: VolumeBox[] | null): PlyBounds | null => {
  if (!boxes?.length) return null;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const b of boxes)
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], b.min[a]); hi[a] = Math.max(hi[a], b.max[a]);
    }
  return { lo, hi };
};

/** What the stage samples of the splats (stage.splatSample()). */
export interface SplatSample {
  x: Float32Array; y: Float32Array; z: Float32Array; opacity: Float32Array;
}

/**
 * The floor before any proposal has detected one, SIGNED (height x upSign),
 * by the pipeline's own rule (`detect_scene_frame`): the opacity-weighted
 * density peak within the lowest `bandFrac` of the robust height range, in
 * bins of `binUnits`. The proposal's number replaces it; this is only where
 * the camera stands until then. Null with nothing to read.
 */
export function floorOfSample(o: {
  sample: SplatSample | null; frame: VolumeFrame | null; bounds: PlyBounds | null;
  bandFrac: number; binUnits: number;
}): number | null {
  if (!o.sample || !o.frame || !o.bounds) return null;
  const { up, upSign } = o.frame;
  const h = up === 0 ? o.sample.x : up === 1 ? o.sample.y : o.sample.z;
  const w = o.sample.opacity;
  const hLo = Math.min(o.bounds.lo[up] * upSign, o.bounds.hi[up] * upSign);
  const hHi = Math.max(o.bounds.lo[up] * upSign, o.bounds.hi[up] * upSign);
  const bandHi = hLo + o.bandFrac * (hHi - hLo);
  const band = bandHi - hLo;
  if (!(band > 0)) return null;
  const nbins = Math.max(o.binUnits > 0 ? Math.floor(band / o.binUnits) : 0, 10);
  const hist = new Float64Array(nbins);
  for (let i = 0; i < h.length; i++) {
    const v = h[i] * upSign;
    if (v < hLo || v > bandHi) continue;
    const b = Math.min(nbins - 1, Math.floor((v - hLo) / band * nbins));
    hist[b] += w[i];
  }
  let peak = 0;
  for (let b = 1; b < nbins; b++) if (hist[b] > hist[peak]) peak = b;
  if (hist[peak] <= 0) return null;
  return hLo + (peak + 0.5) * band / nbins;
}

/** How far back from the footprint centre the eye stands, as a fraction
 *  of the longest ground span: inside the room, not at its wall. */
const PULL_BACK = 0.25;
/** A slight downward look reads as standing, not as staring at a wall. */
const PITCH_DEG = -5;

/** True for the answers where a person can stand in the scene. */
export const walkable = (pathMode: string | null | undefined): boolean =>
  pathMode === "interior" || pathMode === "ground" || pathMode === "manual";

/**
 * A standing pose in the scene frame (stage root local), or null when the
 * scene is filmed around
 * an object (frame the bounds instead) or nothing is known yet.
 *
 * `floorSigned` is the frame's signed height along the up axis (the floor
 * plane's convention: coordinate = height x upSign); `eyeUnits` is the
 * eye height already in scene units (metres over the display scale).
 */
export function standingPose(o: {
  pathMode: string | null | undefined; frame: VolumeFrame | null;
  floorSigned: number | null; bounds: PlyBounds | null; eyeUnits: number;
}): { position: number[]; target: number[] } | null {
  if (!walkable(o.pathMode) || !o.frame || !o.bounds) return null;
  const { up, upSign, a0, a1 } = o.frame;
  const { lo, hi } = o.bounds;
  // the floor: the frame's, else the bounds' low side along the up axis
  const floor = o.floorSigned ?? (upSign > 0 ? lo[up] : -hi[up]);
  const eye = (floor + o.eyeUnits) * upSign;
  const c = [0, 0, 0];
  c[a0] = (lo[a0] + hi[a0]) / 2;
  c[a1] = (lo[a1] + hi[a1]) / 2;
  const long = hi[a0] - lo[a0] >= hi[a1] - lo[a1] ? a0 : a1;
  const span = Math.max(hi[long] - lo[long], 1e-6);
  const position = [...c];
  position[long] = c[long] - PULL_BACK * span;
  position[up] = eye;
  const target = [...c];
  target[up] = eye - Math.tan(-PITCH_DEG * Math.PI / 180) * PULL_BACK * span
    * upSign;
  return { position, target };
}

/** Where the eye sits for an object, above the horizon: high enough to
 *  see its top, low enough to read as looking AT it rather than down on
 *  it (the view dial's iso, at 45°, is the overview — not a start). */
const OBJECT_ELEVATION_DEG = 25;

/**
 * A framing pose in the stage's WORLD frame (+Y up, the flip applied):
 * the eye on the diagonal, `OBJECT_ELEVATION_DEG` above the horizon
 * through the bounds' centre, at the distance where the bounds' sphere
 * fills the narrower of the camera's two fields of view. `fovDeg` is the
 * vertical field of view, `aspect` width over height.
 */
export function framingPose(o: {
  center: number[]; size: number[]; fovDeg: number; aspect: number;
}): { position: number[]; target: number[] } {
  const r = Math.max(Math.hypot(o.size[0], o.size[1], o.size[2]) / 2, 0.5);
  const vfov = o.fovDeg * Math.PI / 180;
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * Math.max(o.aspect, 0.1));
  const half = Math.min(vfov, hfov) / 2;
  const dist = r / Math.sin(half);
  const el = OBJECT_ELEVATION_DEG * Math.PI / 180;
  const d = [Math.cos(el) * Math.SQRT1_2, Math.sin(el), Math.cos(el) * Math.SQRT1_2];
  return { position: o.center.map((c, i) => c + d[i] * dist), target: [...o.center] };
}
