// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The scene-view dial: which way the camera faces, and a click to change it.
//
// Modelled on the view gizmo every 3D editor has, because anyone who has met
// one already knows what it does — click an axis to look along it.
//
// Deliberately DOM + SVG rather than a three-space widget. A gizmo drawn in
// the scene has to be rendered, raycast and kept out of the splat pass; this
// one is seven buttons over a drawing, so it is focusable, labelled and
// keyboard-reachable for free, and it costs the render loop nothing. It is
// also not drei's GizmoHelper: that moves the camera imperatively from its
// own orbit radius, which would bypass the stage's fly animation and leave
// the camera somewhere the settings never recorded.
//
// No projection toggle. An orthographic camera has no meaning here: manual
// captures record a FOV that the pipeline renders as a pinhole camera, so a
// view you cannot capture from is a view this app should not offer.

import { dialArms, VIEWS, type ViewAxis } from "./viewAxes";

/** How far the arms reach, as a share of the dial's half-width. */
const ARM_REACH = 0.78;
/** Drawn size in pixels. Big enough to click, small enough to ignore. */
const DIAL_PX = 66;

// The X/Y/Z convention every 3D tool shares — red, green, blue — so the dial
// is readable without a legend. Front and back are one axis and share a
// colour; which of the pair you are looking at comes from the highlight and
// the label, not from a colour nobody could name.
const AXIS_COLOUR: Record<ViewAxis, string> = {
  right: "#e5534b", left: "#e5534b",
  top: "#7ee787", bottom: "#7ee787", iso: "#7ee787",
  front: "#56d4dd", back: "#56d4dd",
};

/** The short cap on the near three only: a letter on all six at this size is
 *  unreadable, and the far arms are the ones whose label matters least —
 *  their axis is named by the arm opposite. */
const AXIS_LETTER: Partial<Record<ViewAxis, string>> = {
  right: "X", top: "Y", back: "Z",
};

export default function ViewGizmo({ axis, onChoose }: {
  /** the face the camera is currently on, or null when free-flown */
  axis: ViewAxis | null;
  onChoose: (axis: ViewAxis) => void;
}) {
  const arms = dialArms();
  const view = (a: ViewAxis) => VIEWS.find((v) => v.axis === a);
  /** dial coordinates (-1..1, y up) as a CSS percentage from the top left */
  const pct = (v: number, invert = false) =>
    `${50 + ((invert ? -v : v) * ARM_REACH * 100) / 2}%`;

  return (
    <div className="absolute top-2 right-2 z-10 select-none"
         data-testid="view-gizmo">
      <div className="relative rounded-full bg-panel/80 backdrop-blur-sm
                      border border-line2"
           role="radiogroup" aria-label="Camera view"
           style={{ width: DIAL_PX, height: DIAL_PX }}>
        {/* The arms, drawn once beneath the buttons. Lines cannot be focused
            or labelled, so they are decoration — the buttons over them carry
            every affordance. */}
        <svg className="absolute inset-0 w-full h-full" viewBox="-1 -1 2 2"
             aria-hidden="true" focusable="false">
          {arms.map((arm) => (
            <line key={arm.axis} x1={0} y1={0}
                  x2={arm.x * ARM_REACH}
                  // SVG's y runs down the screen and the dial's runs up.
                  y2={-arm.y * ARM_REACH}
                  stroke={AXIS_COLOUR[arm.axis]}
                  // Width in PIXELS, held there by non-scaling-stroke. A width
                  // in the viewBox's own units comes back as a fifteenth of a
                  // pixel and the dial renders as six loose dots.
                  strokeWidth={2.5} vectorEffect="non-scaling-stroke"
                  strokeLinecap="round"
                  // The three pointing away are dimmer; without it the dial is
                  // a flat six-spoke asterisk with no front or back.
                  opacity={arm.toward ? 1 : 0.32} />
          ))}
        </svg>

        {/* Iso at the centre, because a view that looks at all three axes
            belongs where they meet — and it is the largest single target,
            which suits the view you return to most. */}
        <button type="button" role="radio" aria-checked={axis === "iso"}
                onClick={() => onChoose("iso")}
                className={`absolute left-1/2 top-1/2 w-4 h-4 -ml-2 -mt-2
                            rounded-full border transition-colors ${
                  axis === "iso" ? "bg-t1 border-t1"
                                 : "bg-inset border-line2 hover:border-t4"}`}>
          <span className="sr-only">{view("iso")?.label}</span>
        </button>

        {arms.map((arm) => (
          <button key={arm.axis} type="button" role="radio"
                  aria-checked={axis === arm.axis}
                  onClick={() => onChoose(arm.axis)}
                  style={{ left: pct(arm.x), top: pct(arm.y, true),
                           background: AXIS_COLOUR[arm.axis],
                           // Matching the arm it caps, so a far handle does
                           // not float in front of the geometry it is behind.
                           opacity: arm.toward ? 1 : 0.42 }}
                  className={`absolute w-[15px] h-[15px] -ml-[7.5px]
                              -mt-[7.5px] rounded-full grid place-items-center
                              font-mono text-[0.5rem] leading-none text-[#0a0b0c]
                              hover:brightness-125 ${
                    axis === arm.axis ? "ring-1 ring-t1" : ""}`}>
            <span aria-hidden="true">{AXIS_LETTER[arm.axis] ?? ""}</span>
            <span className="sr-only">{view(arm.axis)?.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
