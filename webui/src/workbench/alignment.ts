// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The scene alignment: a small rotation recorded in the profile on
// top of the up-axis convention, so a capture whose floor is not level is
// levelled BEFORE anything reads it. Two frames, named:
//
//   file  — the .ply/.sog coordinates, what the canvas loads and what
//           cameras.json, manual_views.json and interactions.json hold;
//   scene — the file rotated: p_scene = R · p_file, where every geometric
//           decision lives (the frame, the floor, the volume boxes).
//
// R = Rot(up, yaw) · Rot(a1, tilt[1]) · Rot(a0, tilt[0]): the tilt about
// the first ground axis, then the second, then the yaw about the up axis;
// right-handed rotations about +axis, degrees. The pipeline defines the
// same matrix (carveout/alignment.py); the two agree on these vectors
// (up = 1, a0 = 0, a1 = 2):
//   tilt [10, 0], yaw 0   → R·(0,0,1) = ( 0,       -0.173648, 0.984808)
//   tilt [0, 0],  yaw 30  → R·(1,0,0) = ( 0.866025, 0,       -0.5     )
//   tilt [10, 20], yaw 30 → R·(1,0,0) = ( 0.813798, 0.342020, -0.469846)
//
// The viewer holds the splats (file frame) in a group under the root whose
// quaternion IS R: a child at p_file sits at R·p_file in the root, which is
// the scene frame every root-level overlay draws in.

import * as THREE from "three";
import type { AlignmentBlock, AlignmentRead } from "../types";
export type { AlignmentBlock, AlignmentRead };

const DEG = Math.PI / 180;

const AXIS = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0),
              new THREE.Vector3(0, 0, 1)];

/** The two ground-axis indices for an up axis, in ascending order. */
export const groundAxes = (up: number): [number, number] => {
  const g = [0, 1, 2].filter((a) => a !== up);
  return [g[0], g[1]];
};

/** True for no block, or one whose three angles are all zero. */
export function isIdentity(a: AlignmentBlock | null | undefined): boolean {
  if (!a) return true;
  return !(Math.abs(a.tilt_deg?.[0] ?? 0) > 1e-9
           || Math.abs(a.tilt_deg?.[1] ?? 0) > 1e-9
           || Math.abs(a.yaw_deg ?? 0) > 1e-9);
}

/** R as a 4×4 (rotation only), scene = R · file. */
export function alignmentMatrix(a: AlignmentBlock | null | undefined,
                                upAxis: number): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  if (isIdentity(a)) return m;
  const [a0, a1] = groundAxes(upAxis);
  const r0 = new THREE.Matrix4().makeRotationAxis(AXIS[a0], (a!.tilt_deg[0] ?? 0) * DEG);
  const r1 = new THREE.Matrix4().makeRotationAxis(AXIS[a1], (a!.tilt_deg[1] ?? 0) * DEG);
  const ry = new THREE.Matrix4().makeRotationAxis(AXIS[upAxis], (a!.yaw_deg ?? 0) * DEG);
  return m.copy(ry).multiply(r1).multiply(r0);
}

/** The same rotation as a quaternion — what the file group carries. */
export function alignmentQuaternion(a: AlignmentBlock | null | undefined,
                                    upAxis: number): THREE.Quaternion {
  return new THREE.Quaternion().setFromRotationMatrix(alignmentMatrix(a, upAxis));
}

/** R as nine numbers, row-major — the focus effect's uniforms. */
export function alignmentRows(a: AlignmentBlock | null | undefined,
                              upAxis: number): number[] {
  const e = alignmentMatrix(a, upAxis).elements;   // column-major
  return [e[0], e[4], e[8],
          e[1], e[5], e[9],
          e[2], e[6], e[10]];
}

/** A stable key for caching things derived from a block. */
export const alignmentKey = (a: AlignmentBlock | null | undefined): string =>
  isIdentity(a) ? "" : `${a!.tilt_deg[0]},${a!.tilt_deg[1]},${a!.yaw_deg}`;

/** Do two blocks name the same rotation, within `tol` degrees per angle? */
export function sameAlignment(a: AlignmentBlock | null | undefined,
                              b: AlignmentBlock | null | undefined,
                              tol = 1.0): boolean {
  const ta = a?.tilt_deg ?? [0, 0], tb = b?.tilt_deg ?? [0, 0];
  const ya = a?.yaw_deg ?? 0, yb = b?.yaw_deg ?? 0;
  return Math.abs(ta[0] - tb[0]) <= tol && Math.abs(ta[1] - tb[1]) <= tol
    && Math.abs(ya - yb) <= tol;
}

/** The angles in words: "12.8° and 1.8° about the ground axes, 5.0° about
 *  the up axis" (the yaw omitted when zero). */
export function describeAngles(a: AlignmentBlock | { tilt_deg: number[];
                                                     yaw_deg: number | null }): string {
  const t = a.tilt_deg;
  const f = (v: number) => `${Math.abs(v).toFixed(1)}°`;
  const yaw = a.yaw_deg ?? 0;
  const ground = `${f(t[0])} and ${f(t[1])} about the ground axes`;
  return Math.abs(yaw) > 0.05 ? `${ground}, ${f(yaw)} about the up axis` : ground;
}

/** A typed-array sample of splat centres (the stage's `splatSample()`). */
export interface Sample { x: Float32Array; y: Float32Array; z: Float32Array;
                          opacity: Float32Array }

/** The sample carried into the scene frame: every centre through R. The
 *  identity returns the same object (no copy). */
export function rotateSample(s: Sample, a: AlignmentBlock | null | undefined,
                             upAxis: number): Sample {
  if (isIdentity(a)) return s;
  const r = alignmentRows(a, upAxis);
  const n = s.x.length;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const px = s.x[i], py = s.y[i], pz = s.z[i];
    x[i] = r[0] * px + r[1] * py + r[2] * pz;
    y[i] = r[3] * px + r[4] * py + r[5] * pz;
    z[i] = r[6] * px + r[7] * py + r[8] * pz;
  }
  return { x, y, z, opacity: s.opacity };
}

/** Robust bounds of a sample: the 1st..99th percentile per axis (the
 *  stage's rule, applied to a rotated sample). */
export function robustBounds(s: Sample): { lo: number[]; hi: number[] } {
  const pct = (src: Float32Array) => {
    const a = Float32Array.from(src).sort();
    return [a[Math.floor(a.length * 0.01)],
            a[Math.min(a.length - 1, Math.floor(a.length * 0.99))]];
  };
  const [x, y, z] = [pct(s.x), pct(s.y), pct(s.z)];
  return { lo: [x[0], y[0], z[0]], hi: [x[1], y[1], z[1]] };
}
