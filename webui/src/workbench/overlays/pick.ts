// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// ---------------------------------------------------------------------------
// The pick: where a click on the splat lands. A Gaussian
// splat scene has no surface to hit; what the eye sees at a pixel is the
// depth where the splats its ray crosses have accumulated half of the
// pixel's opacity — the renderer's own model, applied to one ray. Spark's
// raycast lists the splats a ray's ellipsoid test crosses and nothing else
// (no opacity, no falloff), and every rule over that list is a count: the
// first attempt (the first cluster of three hits within 2 % of depth)
// called a clump of floaters in front of a test scene's egg carton a
// surface, white ring and all. So the pick walks the splats itself:
//   1. a prefilter over every splat — the centre's distance to the ray
//      against PICK_CUTOFF_SIGMA times its largest scale;
//   2. for the candidates, the ray's closest approach in the splat's own
//      frame (the Mahalanobis distance), alpha = opacity · exp(−m²/2),
//      sorted by depth and accumulated front to back; the point is the
//      splat that takes the accumulated opacity past PICK_MEDIAN_T.
// Solid: it got there. Thin: something was crossed (PICK_THIN_OPACITY or
// more) but never that much — the strongest crossing, so the ring can say
// so. Under that: nothing under the pointer. Numbers only, no three.js:
// the caller brings the ray in the arrays' frame (the file frame) and
// reads `t` back along it. The constants are docs/CALIBRATION.md's.
// ---------------------------------------------------------------------------

/** Every splat of the scene as flat arrays (the stage builds them once,
 *  `splatArrays()`): centres and scales ×3, quaternions ×4 (x, y, z, w),
 *  opacities 0..1. */
export interface SplatArrays {
  n: number;
  center: Float32Array;
  scale: Float32Array;
  quat: Float32Array;
  opacity: Float32Array;
}

/** Accumulated opacity along the ray at which the point is taken: the
 *  median depth of what the pixel shows. Lower it to pick through a
 *  translucent layer (a reflection the model painted in front). */
export const PICK_MEDIAN_T = 0.5;
/** A splat is a candidate when the ray passes its centre within this many
 *  of its largest scale (the rasterizer's own extent). */
export const PICK_CUTOFF_SIGMA = 3;
/** A crossing below this alpha adds nothing (the rasterizer skips it). */
export const PICK_MIN_ALPHA = 1 / 255;
/** A crossing is never fully opaque (the rasterizer's clamp). */
export const PICK_MAX_ALPHA = 0.99;
/** Accumulated opacity under which the ray is said to have hit nothing. */
export const PICK_THIN_OPACITY = 0.1;

export interface SurfaceHit {
  /** distance along the ray, in the arrays' units */
  t: number;
  /** accumulated opacity at the point */
  opacity: number;
  /** crossings that contributed up to the point */
  hits: number;
  /** the accumulation reached PICK_MEDIAN_T */
  solid: boolean;
}

// v rotated by the CONJUGATE of q (world → the splat's frame), into `out`
const out = new Float64Array(3);
function rotConj(qx: number, qy: number, qz: number, qw: number,
                 vx: number, vy: number, vz: number) {
  const ux = -qx, uy = -qy, uz = -qz;
  const tx = 2 * (uy * vz - uz * vy);
  const ty = 2 * (uz * vx - ux * vz);
  const tz = 2 * (ux * vy - uy * vx);
  out[0] = vx + qw * tx + (uy * tz - uz * ty);
  out[1] = vy + qw * ty + (uz * tx - ux * tz);
  out[2] = vz + qw * tz + (ux * ty - uy * tx);
}

/** The ray `o + t·d` (d unit) through the splats: see the header. */
export function pickSurface(o: number[], d: number[], a: SplatArrays): SurfaceHit | null {
  const [ox, oy, oz] = o, [dx, dy, dz] = d;
  const { n, center: c, scale: s, quat: q, opacity: op } = a;
  const ts: number[] = [], as: number[] = [];
  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    const vx = c[i3] - ox, vy = c[i3 + 1] - oy, vz = c[i3 + 2] - oz;
    const tc = vx * dx + vy * dy + vz * dz;
    const sx = s[i3], sy = s[i3 + 1], sz = s[i3 + 2];
    const r = PICK_CUTOFF_SIGMA * Math.max(sx, sy, sz);
    if (tc + r < 0) continue;                       // behind the camera
    const dist2 = vx * vx + vy * vy + vz * vz - tc * tc;   // |v × d|², d unit
    if (dist2 > r * r) continue;
    if (sx <= 0 || sy <= 0 || sz <= 0) continue;
    // the ray in the splat's unit frame: rotate by the conjugate, divide
    // by the scales; `t` keeps its meaning (d is not renormalised)
    const i4 = i * 4;
    const qx = q[i4], qy = q[i4 + 1], qz = q[i4 + 2], qw = q[i4 + 3];
    rotConj(qx, qy, qz, qw, -vx, -vy, -vz);
    const px = out[0] / sx, py = out[1] / sy, pz = out[2] / sz;
    rotConj(qx, qy, qz, qw, dx, dy, dz);
    const ex = out[0] / sx, ey = out[1] / sy, ez = out[2] / sz;
    const dd = ex * ex + ey * ey + ez * ez;
    if (dd === 0) continue;
    const t = -(px * ex + py * ey + pz * ez) / dd;
    if (t < 0) continue;
    const mx = px + t * ex, my = py + t * ey, mz = pz + t * ez;
    const alpha = Math.min(PICK_MAX_ALPHA,
                           op[i] * Math.exp(-0.5 * (mx * mx + my * my + mz * mz)));
    if (alpha < PICK_MIN_ALPHA) continue;
    ts.push(t);
    as.push(alpha);
  }
  if (!ts.length) return null;
  const order = ts.map((_, i) => i).sort((p, r) => ts[p] - ts[r]);
  let T = 1, best = order[0], hits = 0;
  for (const i of order) {
    hits++;
    if (as[i] > as[best]) best = i;
    T *= 1 - as[i];
    if (1 - T >= PICK_MEDIAN_T)
      return { t: ts[i], opacity: 1 - T, hits, solid: true };
  }
  if (1 - T < PICK_THIN_OPACITY) return null;
  return { t: ts[best], opacity: 1 - T, hits, solid: false };
}
