// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// LabelEngine: screen-space HTML annotations (dot + leader + pill) +
// OBB wireframes + settle-based occlusion. The algorithms and
// constants here carry several rounds of validated perf work:
//
// - Zero-layout discipline: view size is CACHED,
//   positions go through compositor-only translate3d, show/hide through
//   `visibility`, nothing allocates per frame.
// - Settle-based occlusion: Spark's WASM raycast
//   iterates ALL splats per ray (~150 ms/ray at 1.69M measured) — rays
//   fire only >= SETTLE_MS after the camera stops, one per frame, aborted
//   on any movement.
// - World-mode text crispness: re-bake font-size on settle so the
//   compositor never scales a cached raster far from 1.

import * as THREE from "three";
import { verdictInfo } from "./verdicts.js";

export const PALETTE = ["#2f81f7", "#3fb950", "#db6d28", "#a371f7",
                        "#f778ba", "#e3b341", "#56d4dd", "#f85149",
                        "#7ee787", "#ffa657"];
const SETTLE_MS = 150;        // occlusion rays only this long after movement stops
const SETTLE_RAYS_PER_FRAME = 1;  // ~150 ms/ray at 1.69M splats — never stack
// "world" label scale mode: worldSize (m) is the pill's world-space height;
// LABEL_NATURAL_PX is the pill's natural CSS height (12px text + padding +
// borders) that worldSize maps onto. Calibration only.
const LABEL_NATURAL_PX = 22;
const LABEL_FONT_PX = 12;      // .ann base font-size (style.css) — bake anchor

export class LabelEngine {
  /**
   * @param {object} o
   *   stage: SplatStage · settings: viewer settings object ·
   *   layer: the #labels HTML element · objects/extObjects: interactions ·
   *   onSelectionChange(idx, prev): app hook (sidebar sync)
   */
  constructor({ stage, settings, layer, objects, extObjects,
                onSelectionChange = null }) {
    this.stage = stage;
    this.settings = settings;
    this.layer = layer;
    this.onSelectionChange = onSelectionChange;
    this.highlighter = null;    // optional {tint(idx, on)} — per-splat tint
    this.classColor = new Map();
    this.overlays = [];
    this.selected = -1;
    this.occlusionOn = localStorage.getItem("carveout_occl") !== "0";

    objects.forEach((o, i) => {
      if (!this.classColor.has(o.label))
        this.classColor.set(o.label,
                            PALETTE[this.classColor.size % PALETTE.length]);
      const color = this.classColor.get(o.label);
      const e = extObjects[i] ?? {};
      const obb = e.obb;

      const geo = new THREE.BoxGeometry(...(obb ? obb.extents
        : ["x", "y", "z"].map((k) => o.scale[k])));
      // Transparent (at full opacity) and after the splat pass: the
      // splats write depth and are themselves a transparent material,
      // and three.js draws every opaque object before any transparent
      // one, whatever its render order — an opaque wire was painted
      // first and the splats covered it, so no box ever showed in an
      // enclosed scene. In the transparent pass at order 3 the wire is
      // depth-tested against the splats: hidden where one is nearer,
      // drawn where it is in front. The volume gizmo's wires do the same.
      const box = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color, transparent: true }));
      box.renderOrder = 3;
      if (obb) {
        box.position.set(...obb.center);
        box.quaternion.set(...obb.rotation_xyzw);
      } else {
        box.position.set(o.position.x, o.position.y, o.position.z);
      }
      box.visible = false;
      // interactions.json describes the FILE: the boxes and anchors live
      // under the file group (the alignment sits on it), not the root
      (stage.fileGroup ?? stage.root).add(box);

      const conf = e.aggregate_confidence ?? 0;
      // exemplar-only support = exemplar-scale confidence (own slider)
      const srcs = Object.keys(e.support_by_source ?? {});
      const exOnly = srcs.length > 0 &&
        srcs.every((s) => s.startsWith("exemplar:"));
      this.overlays.push({
        obj: o, ext: e, box, cls: o.label, idx: i, conf, exOnly,
        anchor: new THREE.Vector3(o.position.x, o.position.y, o.position.z),
        el: this._makeAnnotation(i, o.label, conf,
                                 String(e.verify_verdict ?? "")),
        occluded: false, hovered: false, shown: false, _baked: 1,
      });
    });

    this.classOn = new Map([...this.classColor.keys()].map((c) => [c, true]));

    this._raycaster = new THREE.Raycaster();
    this._lastCamPose = new THREE.Matrix4();
    this._tmpWorld = new THREE.Vector3();
    this._tmpNdc = new THREE.Vector3();
    this._tmpDir = new THREE.Vector3();
    this._visibleBuf = [];
    this._lastMoveTime = performance.now();
    this._settleCursor = 0;
    this._settleDone = false;
    this.rebakeExact = false;  // GUI changes: next settled frame re-bakes exactly
    // The pills were made with the detected label; the display rule
    // (operator's name, else the verifier's, else the detected) applies
    // from the first frame, not only when "show raw" is toggled.
    this.refreshLabelTexts();
  }

  // The names changed under the same boxes (a relabel): take the
  // new extended records and re-apply the display rule — no rebuild of
  // the boxes, the highlighter or the occlusion state.
  setExtObjects(extObjects) {
    this.overlays.forEach((ov) => { ov.ext = extObjects[ov.idx] ?? {}; });
    this.refreshLabelTexts();
  }

  setOcclusion(on) {
    this.occlusionOn = on;
    localStorage.setItem("carveout_occl", on ? "1" : "0");
  }

  _makeAnnotation(idx, label, conf, verdict = "") {
    const el = document.createElement("div");
    el.className = "ann";
    // The verifier's doubt is shown where the label is, as a mark after
    // the name ("?" = the label is in doubt, "≈" = held: one object or
    // several?) with the words in the tooltip; the number beside it is
    // the DETECTOR's confidence, a different thing (verdicts.js).
    const vi = verdictInfo(verdict);
    const flagged = vi.mark !== "";
    el.innerHTML = `
      <div class="ann-pill${flagged ? " ann-flag" : ""}"${
        flagged ? ` data-mark="${vi.mark}" title="${vi.word}"` : ""}>
        <span class="ann-name">${label}</span>
        <span class="ann-conf">${conf.toFixed(2)}</span></div>
      <div class="ann-leader"></div>
      <div class="ann-dot"></div>`;
    el.style.visibility = "hidden";
    const pill = el.querySelector(".ann-pill");
    pill.addEventListener("mouseenter", () => this.setHover(idx, true));
    pill.addEventListener("mouseleave", () => this.setHover(idx, false));
    pill.addEventListener("click", () => this.select(idx));
    this.layer.appendChild(el);
    return el;
  }

  // KEEP BOTH relabel policy: detected label is canonical
  // (classes, colors, grouping); what gets DISPLAYED follows one
  // setting, labels.source. "detected": the term SAM detected, for every
  // object, over the operator's own names (a debugging view).
  // Otherwise the operator's own name (extended.operator_label) comes
  // first; then, under "proposed", the verifier's proposal
  // (extended.proposed_label) where there is one — a preview of what the
  // verifier would have called things, never over the operator's own
  // act; then the Stage 4.5 verdict under extended.verified_label.
  displayLabel(ov) {
    const src = this.settings.labels.source;
    if (src === "detected") return ov.cls;
    if (ov.ext.operator_label) return ov.ext.operator_label;
    if (src === "proposed" && ov.ext.proposed_label) return ov.ext.proposed_label;
    return ov.ext.verified_label ?? ov.cls;
  }

  refreshLabelTexts() {
    this.overlays.forEach((ov) => {
      ov.el.querySelector(".ann-name").textContent = this.displayLabel(ov);
    });
  }

  // Marker styling (single point): anchor dot, leader and pill accent bar
  // take either the category color or the custom marker color; accent/score
  // visibility toggles are classes on the label layer (see style.css). With
  // the accent hidden its inline color is cleared so the 1px class rule wins.
  applyLabelStyle() {
    const l = this.settings.labels;
    this.layer.classList.toggle("no-accent", !l.showAccent);
    this.layer.classList.toggle("no-score", !l.showScore);
    this.layer.style.setProperty("--pill-bg", l.pillBackground);
    for (const ov of this.overlays) {
      const color = l.markerColorMode === "custom" ? l.markerColor
                  : this.classColor.get(ov.cls);
      ov.el.querySelector(".ann-dot").style.background = color;
      ov.el.querySelector(".ann-leader").style.background = color;
      ov.el.querySelector(".ann-pill").style.borderLeftColor =
        l.showAccent ? color : "";
    }
  }

  toggleClass(cls, on) {
    this.classOn.set(cls, on);
    this.boxVisibility();
  }

  boxVisibility() {
    this.overlays.forEach((ov) => {
      ov.box.visible = this.classOn.get(ov.cls) &&
        (this.settings.labels.showBoxes || ov.hovered ||
         ov.idx === this.selected);
    });
  }

  setHover(idx, on) {
    this.overlays[idx].hovered = on;
    this.boxVisibility();
  }

  select(idx) {
    const prev = this.selected;
    if (prev >= 0) this.highlighter?.tint(prev, false);
    if (prev === idx) {
      this.selected = -1;
      this.boxVisibility();
      this.onSelectionChange?.(-1, prev);
      return;
    }
    this.selected = idx;
    this.highlighter?.tint(idx, true);
    this.boxVisibility();
    const target = (this.stage.fileGroup ?? this.stage.root).localToWorld(
      this.overlays[idx].anchor.clone());
    const dir = this.stage.camera.position.clone().sub(target).normalize();
    this.stage.flyTo({
      target,
      to: target.clone().add(
        dir.multiplyScalar(this.settings.camera.flyDistance)) });
    this.onSelectionChange?.(idx, prev);
  }

  _settleOcclusionStep(candidates, count) {
    const { camera, splats, perf } = this.stage;
    const root = this.stage.fileGroup ?? this.stage.root;
    const t0 = performance.now();
    const end = Math.min(this._settleCursor + SETTLE_RAYS_PER_FRAME, count);
    for (; this._settleCursor < end; this._settleCursor++) {
      const ov = candidates[this._settleCursor];
      this._tmpWorld.copy(ov.anchor).applyMatrix4(root.matrixWorld);
      const dist = camera.position.distanceTo(this._tmpWorld);
      this._tmpDir.copy(this._tmpWorld).sub(camera.position).normalize();
      this._raycaster.set(camera.position, this._tmpDir);
      this._raycaster.far = dist;
      try {
        const hits = this._raycaster.intersectObject(splats, false);
        // The anchor is the instance CENTROID — the ray always hits the
        // instance's own front surface first. Only count hits clearly in
        // front of the whole instance (its bounding radius), not self-hits.
        // The instance's extents are scene units; the ray runs in world
        // (display metres), so the radius crosses the root's scale.
        const o = ov.obj;
        const r = 0.5 * Math.hypot(o.scale.x, o.scale.y, o.scale.z)
          * this.stage.root.scale.x;
        ov.occluded = hits.length > 0 && hits[0].distance < dist - r - 0.25;
      } catch { ov.occluded = false; }
      perf.occlRays++;
    }
    if (this._settleCursor >= count) this._settleDone = true;
    const elapsed = performance.now() - t0;
    perf.occlMs += elapsed;
    return elapsed;
  }

  // World-mode text crispness: the compositor scales a CACHED raster, so a
  // transform-scaled pill blurs. Fix = re-bake: write the scale into the
  // REAL font-size (all .ann dimensions are em) so the label re-rasterizes
  // at the displayed size. Baking costs layout — callers run it on camera
  // settle or GUI changes only, never per frame while moving.
  bakeLabelScale(ov, s) {
    ov._baked = s;
    ov.el.style.fontSize =
      s === 1 ? "" : `${(LABEL_FONT_PX * s).toFixed(2)}px`;
  }

  bakeAll(s) {
    this.overlays.forEach((ov) => this.bakeLabelScale(ov, s));
  }

  _setShown(ov, on) {
    if (ov.shown !== on) {
      ov.shown = on;
      ov.el.style.visibility = on ? "visible" : "hidden";
    }
  }

  update() {
    const { camera, splats, perf, viewW, viewH } = this.stage;
    const root = this.stage.fileGroup ?? this.stage.root;
    const t0 = performance.now();
    const labelsOn = this.settings.labels.show;
    // "always" mode: every in-frustum label regardless of distance or count
    // (replaces the hardcoded 18 m / nearest-18).
    const distanceMode = this.settings.labels.mode === "distance";
    const maxDist = distanceMode ? this.settings.labels.maxDistance : Infinity;
    const maxCount = distanceMode ? this.settings.labels.maxCount : Infinity;
    const minConf = this.settings.labels.minConfidence;
    const minConfEx = this.settings.labels.exemplarMinConfidence ?? 0;
    const visibleBuf = this._visibleBuf;
    visibleBuf.length = 0;
    for (const ov of this.overlays) {
      if (!labelsOn || !this.classOn.get(ov.cls) ||
          ov.conf < (ov.exOnly ? minConfEx : minConf)) {
        this._setShown(ov, false);
        continue;
      }
      this._tmpWorld.copy(ov.anchor).applyMatrix4(root.matrixWorld);
      const dist = camera.position.distanceTo(this._tmpWorld);
      this._tmpNdc.copy(this._tmpWorld).project(camera);
      const inFrustum = this._tmpNdc.z < 1 && this._tmpNdc.z > -1 &&
        Math.abs(this._tmpNdc.x) < 1 && Math.abs(this._tmpNdc.y) < 1 &&
        dist < maxDist;
      if (!inFrustum) {
        this._setShown(ov, false);
        continue;
      }
      ov._x = (this._tmpNdc.x * 0.5 + 0.5) * viewW;
      ov._y = (-this._tmpNdc.y * 0.5 + 0.5) * viewH;
      ov._dist = dist;
      visibleBuf.push(ov);
    }
    visibleBuf.sort((a, b) => a._dist - b._dist);

    const now = performance.now();
    if (!this._lastCamPose.equals(camera.matrixWorld)) {
      this._lastCamPose.copy(camera.matrixWorld);
      this._lastMoveTime = now;
      this._settleCursor = 0;
      this._settleDone = false;
    }
    const settled = now - this._lastMoveTime >= SETTLE_MS;
    let occlElapsed = 0;
    if (labelsOn && this.occlusionOn && settled && !this._settleDone &&
        !this.stage.isFlying && splats.isInitialized) {
      occlElapsed = this._settleOcclusionStep(
        visibleBuf, Math.min(visibleBuf.length,
                             distanceMode ? maxCount * 2
                                          : visibleBuf.length));
    }

    // While moving (or with occlusion off), labels are optimistically visible.
    const hideOccluded = this.occlusionOn && settled;
    const fadeDist = distanceMode ? maxDist : 60;  // opacity falloff reference
    // "world" scale mode: the whole assembly scales about the anchor like a
    // fixed-world-size object — scale = worldSize · pxPerMeter /
    // LABEL_NATURAL_PX, pxPerMeter = viewH / (2·dist·tan(fov/2)). "screen"
    // (default) keeps the bare translate — the constant-pixel path.
    const worldMode = this.settings.labels.scaleMode === "world";
    const scaleK = worldMode
      ? this.settings.labels.worldSize * viewH /
        (2 * Math.tan(camera.fov * Math.PI / 360) * LABEL_NATURAL_PX)
      : 0;
    let shown = 0;
    for (const ov of visibleBuf) {
      if ((hideOccluded && ov.occluded) || shown >= maxCount) {
        this._setShown(ov, false);
        continue;
      }
      shown++;
      this._setShown(ov, true);
      if (worldMode && settled) {
        const s = scaleK / ov._dist;
        const drift = s / ov._baked;
        if (this.rebakeExact || drift > 1.1 || drift < 0.9)
          this.bakeLabelScale(ov, s);
      }
      ov.el.style.transform =
        `translate3d(${ov._x.toFixed(1)}px, ${ov._y.toFixed(1)}px, 0)` +
        (worldMode
          ? ` scale(${(scaleK / ov._dist / ov._baked).toFixed(3)})` : "");
      const op = Math.max(0.35, Math.min(1, 1.6 - ov._dist / fadeDist));
      if (Math.abs((ov._op ?? -1) - op) > 0.03) {
        ov._op = op;
        ov.el.style.opacity = op;
      }
    }
    if (settled) this.rebakeExact = false;
    perf.shown = shown;
    // labelMs = steady-state label bookkeeping only; settle rays are
    // deliberate idle-time work and tracked separately in occlMs.
    perf.labelMs += performance.now() - t0 - occlElapsed;
  }

  dispose() {   // React unmount path
    this.overlays.forEach((ov) => {
      ov.el.remove();
      ov.box.removeFromParent();
    });
    this.overlays.length = 0;
  }
}
