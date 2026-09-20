// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Shared vocabulary of the overlay modules: the palette, and the scene
// frame (which axis is up, which two span the ground).

export const DEG = Math.PI / 180;

export const COL = {
  // Capture types (lavender/amber did not work): medium blue and a
  // controlled green, quieter than the
  // original cyan/green against the neutral UI. Identical in both UI
  // themes — viewport colours never follow the theme.
  auto: 0x648fd1,
  manual: 0x69b88a,
  selected: 0xffffff,
  volume: 0x66c5d4,
  volumeSel: 0xffffff,
  handle: 0x66c5d4,
  crop: 0xe3b341,
  stale: 0x4e585d,
  halo: 0x0a0b0c,
  floor: 0x8a9499,       // origin grid lines
  floorFill: 0xffffff,   // detected-floor plane (translucent)
};

export interface VolumeFrame { up: number; upSign: number; a0: number;
                               a1: number }

/** The scene frame implied by a profile's `scene.up_axis` ("-y", "+x", …).
 * `auto` and anything unparsable fall back to the -Y render convention:
 * that is what Carveout assumes until the axis is pinned, and drawing it is
 * how a mis-oriented scene is noticed (USAGE §1). */
export function frameFromUpAxis(spec: string | null | undefined): VolumeFrame {
  const m = /^([+-]?)([xyz])$/.exec((spec ?? "").trim().toLowerCase());
  const up = m ? "xyz".indexOf(m[2]) : 1;
  const upSign = m && m[1] === "-" ? -1 : m ? 1 : -1;
  const ground = [0, 1, 2].filter((a) => a !== up);
  return { up, upSign, a0: ground[0], a1: ground[1] };
}

/** The same palette entry as a CSS colour, for DOM chips and badges. */
export const cssHex = (c: number) => "#" + c.toString(16).padStart(6, "0");
