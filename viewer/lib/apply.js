// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The settings-effects table: which applier runs after which settings
// mutation, defined ONCE. The workbench settings panel is an adapter over
// this table, never a mirror of it: a new setting or a changed invariant
// is edited here and nowhere else, so a control and the frame loop cannot
// silently diverge — the exact failure mode viewer/lib exists to prevent,
// one layer up.
//
// Each entry encodes a validated lesson (the flash semantics, the
// screen/world re-bake rule, the style/box appliers); keys with no entry
// are read live by the frame loop and need no applier. `engine` and
// `focus` may be null/inactive while the workbench canvas is still
// assembling — every effect tolerates that.

const style = (c) => c.engine?.applyLabelStyle();
const flash = (c) => c.focus?.applyFocus?.(true);
const feel = (c) => c.stage.applyControlFeel();

const EFFECTS = {
  "camera.fov": (c) => {
    c.stage.camera.fov = c.settings.camera.fov;
    c.stage.camera.updateProjectionMatrix();
  },
  "background": (c) => c.stage.scene.background.set(c.settings.background),
  "flip": (c) => c.stage.applyFlip(),
  "labels.showBoxes": (c) => c.engine?.boxVisibility(),
  "labels.source": (c) => c.engine?.refreshLabelTexts(),
  // screen mode's hot path assumes unbaked labels; world re-bakes on settle
  "labels.scaleMode": (c) => {
    if (!c.engine) return;
    if (c.settings.labels.scaleMode === "screen") c.engine.bakeAll(1);
    else c.engine.rebakeExact = true;
  },
  "labels.worldSize": (c) => {   // on COMMIT, not per slider tick
    if (c.engine) c.engine.rebakeExact = true;
  },
  "labels.markerColorMode": style,
  "labels.markerColor": style,
  "labels.pillBackground": style,
  "labels.showAccent": style,
  "labels.showScore": style,
  "focus.enabled": flash,
  "focus.offset.*": flash,
  "focus.scale.*": flash,
  "focus.rotation.*": flash,
  "focus.pointSize": flash,
  "focus.transition": flash,
  "focus.invert": flash,
  // showBox: no flash — the wires are already visible when the handles are
  "focus.showBox": (c) => c.focus?.applyFocus?.(),
  "controls.inertia": feel,
  "controls.reversePan": feel,
};

/**
 * Apply one setting. `ctx` is { stage, engine, focus, settings }; `path`
 * is dotted ("labels.scaleMode", "focus.offset.1"). When `value` is
 * given it is written into settings first; a caller whose field has
 * already written it omits it. Keys without
 * an effect just store — the frame loop reads them live.
 */
export function applySetting(ctx, path, value) {
  if (value !== undefined) {
    const keys = path.split(".");
    let node = ctx.settings;
    for (const k of keys.slice(0, -1)) node = node[k];
    node[keys[keys.length - 1]] = value;
  }
  const fx = EFFECTS[path] ?? EFFECTS[path.replace(/\.\d+$/, ".*")];
  fx?.(ctx);
}

/** The camera-inside bypass composes with the DEFAULT mode only:
 *  inverted, it would disable the effect exactly in the region chosen to
 *  be dots — so its control locks while invert is on. */
export function bypassLocked(settings) {
  return !!settings.focus.invert;
}

/** Reset the focus volume to the detection volume it started from.
 *  Mutates the arrays IN PLACE — the settings panel's controls are bound
 *  to these array objects. */
export function resetFocus(ctx) {
  ctx.settings.focus.offset.fill(0);
  ctx.settings.focus.scale.fill(1);
  ctx.settings.focus.rotation.fill(0);
  flash(ctx);
}
