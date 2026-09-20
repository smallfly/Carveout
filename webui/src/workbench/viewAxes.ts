// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Where the camera stands for each face of the view gizmo, and how that
// gizmo is drawn. Pure arithmetic — no three, no React — so a face can be
// checked against known numbers instead of by orbiting and looking.
//
// Everything here is in the stage's WORLD frame, where +Y is up. That is
// true because `applyFlip()` maps the scene's -Y-up ply convention onto
// three's +Y-up with a pi-about-X rotation on the root; the gizmo therefore
// follows the flip rather than second-guessing it. A scene whose flip is
// wrong looks rolled in every other overlay too, and the viewer's flip note
// is the place that says so.

/** Which way the camera faces. `iso` is the angled overview. */
export type ViewAxis = "iso" | "top" | "bottom" | "front" | "back"
                     | "left" | "right";

export interface Bounds {
  /** world-space centre of what the view should frame */
  center: [number, number, number];
  /** world-space extent (full width) on each axis */
  size: [number, number, number];
}

export interface Placement {
  to: [number, number, number];
  target: [number, number, number];
}

// No keyboard shortcuts yet: the free-flight keys already own WASD/QE and the
// numeric row is the obvious remaining home, so it is a deliberate choice
// rather than an oversight — add them here and in the dial's titles together.
export const VIEWS: readonly { axis: ViewAxis; label: string }[] = [
  { axis: "iso", label: "Iso" },
  { axis: "top", label: "Top" },
  { axis: "bottom", label: "Bottom" },
  { axis: "front", label: "Front" },
  { axis: "back", label: "Back" },
  { axis: "left", label: "Left" },
  { axis: "right", label: "Right" },
];

// Straight down would make the camera's up vector parallel to its view
// direction, which is the one orientation `lookAt` cannot resolve — the
// result spins on whatever rounding error is left. A hair of lean costs
// nothing visually and removes the singularity.
const TOP_TILT = 0.001;

/** Unit direction the camera sits in, FROM the target, per face. */
const DIRECTIONS: Record<ViewAxis, readonly [number, number, number]> = {
  // Off every axis, so a perspective view reads as one: square-on to a face
  // makes perspective look like badly-done orthographic. High enough to look
  // INTO the scene rather than across it — at a grazing angle the near
  // geometry occludes everything the view exists to show.
  iso: [0.7, 1, 0.7],
  top: [0, Math.cos(TOP_TILT), Math.sin(TOP_TILT)],
  // Underneath. Present because a splat scene has no floor to stand on —
  // looking up at a table or a shelf from below is a real inspection, not the
  // curiosity it would be in a plan-drawing tool. Same lean, same reason.
  bottom: [0, -Math.cos(TOP_TILT), Math.sin(TOP_TILT)],
  front: [0, 0, -1],
  back: [0, 0, 1],
  left: [-1, 0, 0],
  right: [1, 0, 0],
};

/** How far back the camera sits, relative to the largest span it must hold.
 *  Comfortably outside: a camera inside the bounds frames the inside of the
 *  nearest surface. */
const DISTANCE_FACTOR = 1.6;

/** Smallest span worth framing, in metres. A degenerate volume (one box
 *  collapsed, or a scene that has none) would otherwise put the camera on
 *  its own target — a zero-length direction that goes NaN on normalise. */
const MIN_SPAN = 1;

export function placeCamera(axis: ViewAxis, b: Bounds): Placement {
  const span = Math.max(...b.size, MIN_SPAN);
  const d = DIRECTIONS[axis];
  const dist = span * DISTANCE_FACTOR;
  return {
    target: [...b.center] as [number, number, number],
    to: [b.center[0] + d[0] * dist,
         b.center[1] + d[1] * dist,
         b.center[2] + d[2] * dist],
  };
}

// --- the dial's drawn geometry ---------------------------------------------

const ISO_COS = Math.cos(Math.PI / 6);   // 30 degrees, the isometric standard
const ISO_SIN = Math.sin(Math.PI / 6);

/** A direction projected onto the dial's 2D face. +X goes right and slightly
 *  down, +Z comes toward the viewer as left and slightly down; Y is the only
 *  one that moves straight up the face. */
function projectIso(d: readonly [number, number, number]) {
  return { x: (d[0] - d[2]) * ISO_COS, y: d[1] - (d[0] + d[2]) * ISO_SIN };
}

export interface DialArm {
  axis: ViewAxis; x: number; y: number;
  /** on the viewer's side of the origin — the near three are drawn solid */
  toward: boolean;
}

/** The six arms, sorted back to front so the overlap reads correctly. Drawing
 *  them in table order paints the far arms over the near ones, which inverts
 *  the whole picture. `iso` has no arm — it is the hub. */
export function dialArms(): DialArm[] {
  return (Object.keys(DIRECTIONS) as ViewAxis[])
    .filter((a) => a !== "iso")
    .map((axis) => {
      const d = DIRECTIONS[axis];
      const { x, y } = projectIso(d);
      const iso = DIRECTIONS.iso;
      const depth = d[0] * iso[0] + d[1] * iso[1] + d[2] * iso[2];
      return { axis, x, y, toward: depth > 0, depth };
    })
    .sort((a, b) => a.depth - b.depth)
    .map(({ axis, x, y, toward }) => ({ axis, x, y, toward }));
}
