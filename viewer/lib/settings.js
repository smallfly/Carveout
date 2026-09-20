// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Per-scene viewer settings: defaults + deep merge.
//
// Absent file = these defaults (the original hardcoded behavior: 18 m distance
// cap, nearest-18, Spark control feel).

export const DEFAULT_SETTINGS = {
  version: 1,
  camera: { fov: 65, position: null, quaternion: null,    // null = default framing
            space: "world",       // "root": position/quaternion are scene-unit,
                                  // root-local (saved by captureStartPose);
                                  // "world": a file from before the root scale
            flyDistance: 1.0 },                           // click-to-fly stand-off (m)
  // The display scale while the scene's scale is NOT known: the scene's
  // largest robust extent is shown as this many display metres (with a
  // recorded factor the display IS metres and this is unused).
  display: { nominalExtent: 5 },
  background: "#0d1117",                                  // pre-18 hardcoded color
  labels: { show: true, showBoxes: false, mode: "distance",
            maxDistance: 18, maxCount: 18, minConfidence: 0,
            // Exemplar-rescued instances score on the
            // exemplar-similarity scale — NEVER comparable with text
            // confidence, so they get their own floor (a 0.53
            // text slider once silently hid 0.39-0.43 exemplar rescues).
            exemplarMinConfidence: 0,
            // What a label shows: "verified" (the default: a name the
            // operator gave, else the verifier's applied verdict, else the
            // detected term), "proposed" (the same, but a name the verifier
            // proposed and could not apply shows where there is one, under
            // the operator's own) or "detected" (the term SAM detected, for
            // every object, the operator's names included).
            source: "verified",
            markerColorMode: "category", markerColor: "#ffffff",
            pillBackground: "#0d1117", showAccent: true, showScore: true,
            scaleMode: "screen", worldSize: 0.06 },
  focus: { enabled: false, offset: [0, 0, 0], scale: [1, 1, 1],
           rotation: [0, 0, 0],   // degrees, XYZ, about the union centre
           pointSize: 0.01, transition: 0, invert: false, bypass: false,
           showBox: false },
  controls: { inertia: true, reversePan: false },
  flip: true,
};

export const mergeSettings = (d, o) => (o && typeof o === "object" && !Array.isArray(o))
  ? Object.fromEntries(Object.keys(d).map(
      (k) => [k, k in o ? mergeSettings(d[k], o[k]) : d[k]]))
  : (o === undefined ? d : o);

// Fetch + merge a scene's settings file; defaults on 404/parse failure.
export async function loadSettings(url) {
  try {
    const r = await fetch(url);
    if (r.ok) return mergeSettings(DEFAULT_SETTINGS, legacyLabels(await r.json()));
  } catch { /* defaults */ }
  return structuredClone(DEFAULT_SETTINGS);
}

// A file saved before `labels.source` carried two flags; raw took
// precedence over proposed on screen, so it does here.
function legacyLabels(o) {
  const l = o && typeof o === "object" ? o.labels : null;
  if (!l || typeof l !== "object" || "source" in l) return o;
  if (l.showRaw) l.source = "detected";
  else if (l.showProposed) l.source = "proposed";
  return o;
}
