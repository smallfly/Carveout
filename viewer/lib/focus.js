// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Dual-render focus volume. Everything OUTSIDE the focus boxes renders
// as a point cloud
// (small isotropic gaussians), everything inside as normal 3DGS, via a dyno
// objectModifier on the SplatMesh — object space IS the file's space, the
// FILE frame (the pi-flip sits on the root, the scene alignment on the
// file group between them). volume.json is in the SCENE frame, so each
// centre is carried through the alignment R first (three dot products
// against its rows, identity by default — `setAlignment`), and the box
// test then needs no other transform. Box min/max, point size and
// on/off are dyno UNIFORMS: value writes + updateVersion() re-run the GPU
// generation pass with NO shader recompile — live volume adjustment stays
// interactive at 6.3M splats.
//
// Per-splat decision:
//   p   = the splat centre carried into the focus volume's own frame:
//         R^T (c - unionCentre) + unionCentre. The boxes stay axis-aligned in
//         that frame, so one inverse rotation buys an ORIENTED volume without
//         an oriented distance function.
//   sd  = signed distance to the focus union (sdBox min-composed; + outside)
//   t   = smoothstep of sd*dir over `transition` — the blend band sits on
//         the DOT side of the edge; transition=0 = the previous hard edge
//   dir = +1 default / -1 inverted (inside becomes the dot region)
//   t  *= on * strength; scales = mix(original, vec3(pointSize), t)
// strength = camera-inside bypass (default mode only): smoothstep of the
// CAMERA's signed distance over the same band, computed CPU-side per frame
// (spatial, stateless — chosen over a temporal fade).

import * as THREE from "three";
import { dyno } from "@sparkjsdev/spark";

/** The volume gate's preview: the effect with every adjustment at rest
 *  and the finest dot the settings slider offers (2 mm of displayed
 *  size). Fixed by design — the gate previews the box it is editing,
 *  nothing else — and never written to the settings. */
export const VOLUME_PREVIEW_FOCUS = Object.freeze({
  enabled: true, offset: [0, 0, 0], scale: [1, 1, 1], rotation: [0, 0, 0],
  pointSize: 0.002, transition: 0, invert: false, bypass: false,
  showBox: false,
});

export function createFocusVolume({ splats, root, camera, volume,
                                    settings }) {
  const focusOn = dyno.dynoFloat(0);         // bool-as-float uniform
  const focusPointSize = dyno.dynoFloat(settings.focus.pointSize);
  const focusDir = dyno.dynoFloat(1);        // +1 = outside collapses
  const focusTransition = dyno.dynoFloat(0); // blend band width (m)
  const focusStrength = dyno.dynoFloat(1);   // camera-inside bypass 0..1
  // Rotation, as the union centre plus the three COLUMNS of R — which are the
  // rows of R^T, so the inverse rotation is three dots and no transpose.
  // Passed as vec3s rather than a mat3 on purpose: vec3 is the uniform type
  // this file already drives, and the effect cannot be exercised without a
  // GPU, so this is not the place to take a new API on trust.
  // Always in the shader, never branched around: a zero rotation costs three
  // dot products and keeps enabling it a UNIFORM write, so it stays
  // interactive instead of forcing a pipeline recompile.
  const focusCenter = dyno.dynoVec3(new THREE.Vector3());
  const focusOffset = dyno.dynoVec3(new THREE.Vector3());
  // The scene alignment, as the three ROWS of R: component i of R·c is
  // dot(row_i, c). Identity until the canvas sets one.
  const alignRow = [dyno.dynoVec3(new THREE.Vector3(1, 0, 0)),
                    dyno.dynoVec3(new THREE.Vector3(0, 1, 0)),
                    dyno.dynoVec3(new THREE.Vector3(0, 0, 1))];
  const focusRotCol = [dyno.dynoVec3(new THREE.Vector3(1, 0, 0)),
                       dyno.dynoVec3(new THREE.Vector3(0, 1, 0)),
                       dyno.dynoVec3(new THREE.Vector3(0, 0, 1))];
  const rotM = new THREE.Matrix4();
  const rotEuler = new THREE.Euler();
  const focusBoxes = [];   // { src, min: dynoVec3, max: dynoVec3, wire }
  let focusRange = 5;
  let focusFlashTimer = null;
  // pointSize and transition are DISPLAY-METRE settings written into
  // object-space (scene-unit) gaussian scales and distances: divided by
  // the root's display scale, which the canvas keeps current here.
  let displayScale = 1;

  if (!volume?.boxes?.length) {
    return { active: false, focusBoxes, focusRange,
             applyFocus: () => {}, updateFocusBypass: () => {},
             setDisplayScale: () => {}, dispose: () => {},
             setBoxSources: () => false, setOverride: () => {},
             setAlignment: () => {} };
  }
  // A fixed-parameter block that shadows settings.focus while set (the
  // volume gate's preview): applyFocus reads it instead, settings stay
  // untouched, and setOverride(null) restores whatever the settings say.
  let override = null;

  const lo = new THREE.Vector3(+Infinity, +Infinity, +Infinity);
  const hi = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  for (const b of volume.boxes) {
    lo.min(new THREE.Vector3(...b.min));
    hi.max(new THREE.Vector3(...b.max));
  }
  const unionCenter = lo.clone().add(hi).multiplyScalar(0.5);
  focusRange = Math.max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 2);

  for (const b of volume.boxes) {
    const wire = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: "#e3b341" }));
    wire.visible = false;
    root.add(wire);   // root children are in ply coords, same as volume.json
    focusBoxes.push({ src: b, min: dyno.dynoVec3(new THREE.Vector3()),
                      max: dyno.dynoVec3(new THREE.Vector3()), wire });
  }

  const setWires = (on) => focusBoxes.forEach((fb) => (fb.wire.visible = on));

  // The composition, in order: scale about unionCentre, THEN turn about
  // unionCentre, THEN translate by offset.
  //
  //   world = R * ((src - U) * scale) + U + offset
  //
  // Offset last on purpose. Folded in before the rotation it would be turned
  // along with the box, so "offset X" would push along a different world
  // direction at every angle — unusable from a slider and impossible to map a
  // drag handle onto. Last, it is a plain world translation at any rotation,
  // which is exactly what a translate handle produces. Zero rotation makes
  // the two orders identical, so nothing already saved changes meaning.
  const DEG = Math.PI / 180;
  const applyFocus = (flash = false) => {
    const f = override ?? settings.focus;
    focusOn.value = f.enabled ? 1 : 0;
    focusPointSize.value = f.pointSize / displayScale;
    focusDir.value = f.invert ? -1 : 1;
    focusTransition.value = f.transition / displayScale;
    // Degrees in the settings file because that is what the sliders show and
    // what a hand-edit means; radians only ever exist in here.
    const r = f.rotation || [0, 0, 0];
    rotEuler.set(r[0] * DEG, r[1] * DEG, r[2] * DEG, "XYZ");
    rotM.makeRotationFromEuler(rotEuler);
    focusCenter.value.copy(unionCenter);
    for (let a = 0; a < 3; a++)
      focusRotCol[a].value.setFromMatrixColumn(rotM, a);
    focusOffset.value.set(f.offset[0], f.offset[1], f.offset[2]);
    for (const fb of focusBoxes) {
      for (let a = 0; a < 3; a++) {
        const c = unionCenter.getComponent(a);
        fb.min.value.setComponent(a, (fb.src.min[a] - c) * f.scale[a] + c);
        fb.max.value.setComponent(a, (fb.src.max[a] - c) * f.scale[a] + c);
      }
      // The wire is the one thing that carries the rotation FORWARD rather
      // than inverting it: the shader turns the splat into the box's frame,
      // the wire is the box drawn in the world.
      fb.wire.position.copy(fb.min.value).add(fb.max.value).multiplyScalar(0.5)
        .sub(unionCenter).applyEuler(rotEuler).add(unionCenter)
        .add(focusOffset.value);
      fb.wire.quaternion.setFromEuler(rotEuler);
      fb.wire.scale.copy(fb.max.value).sub(fb.min.value);
    }
    splats.updateVersion();   // uniforms changed -> regenerate (no recompile)
    // The wireframe flashes during adjustment even when "show volume box"
    // is off, so sliders have a visible target.
    // No yellow wire under an override: the gate already draws the box
    // it is editing, and a second outline would double it.
    setWires(!override && (f.showBox || flash));
    if (flash && !f.showBox && !override) {
      clearTimeout(focusFlashTimer);
      focusFlashTimer = setTimeout(
        () => setWires(settings.focus.showBox), 1200);
    }
  };

  splats.objectModifier = dyno.dynoBlock(
    { gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat },
    ({ gsplat }) => {
      const inTypes = { gsplat: dyno.Gsplat, on: "float", psize: "float",
                        dir: "float", trans: "float", strength: "float",
                        ucenter: "vec3", uoff: "vec3", rc0: "vec3",
                        rc1: "vec3", rc2: "vec3",
                        al0: "vec3", al1: "vec3", al2: "vec3" };
      const uniforms = { on: focusOn, psize: focusPointSize, dir: focusDir,
                         trans: focusTransition, strength: focusStrength,
                         ucenter: focusCenter, uoff: focusOffset,
                         rc0: focusRotCol[0], rc1: focusRotCol[1],
                         rc2: focusRotCol[2],
                         al0: alignRow[0], al1: alignRow[1], al2: alignRow[2] };
      focusBoxes.forEach((fb, i) => {
        inTypes[`bmin${i}`] = "vec3";
        inTypes[`bmax${i}`] = "vec3";
        uniforms[`bmin${i}`] = fb.min;
        uniforms[`bmax${i}`] = fb.max;
      });
      const d = new dyno.Dyno({
        inTypes,
        outTypes: { gsplat: dyno.Gsplat },
        globals: () => [dyno.unindent(`
          float sdBox(vec3 p, vec3 bmin, vec3 bmax) {
            vec3 q = abs(p - 0.5 * (bmin + bmax)) - 0.5 * (bmax - bmin);
            return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
          }
        `)],
        statements: ({ inputs, outputs }) => {
          // Union of boxes = min over per-box signed distances (+ outside).
          const sd = focusBoxes.map((_, i) =>
            `sdBox(p, ${inputs[`bmin${i}`]}, ${inputs[`bmax${i}`]})`)
            .reduce((a, b) => `min(${a}, ${b})`);
          return dyno.unindentLines(`
            ${outputs.gsplat} = ${inputs.gsplat};
            // the centre carried into the SCENE frame (the alignment R)
            vec3 c0 = ${inputs.gsplat}.center;
            vec3 c = vec3(dot(c0, ${inputs.al0}), dot(c0, ${inputs.al1}),
                          dot(c0, ${inputs.al2}));
            // Inverse of  world = R * (b - U) + U + offset, so the offset
            // comes off FIRST and the rotation second. rc* are the columns of
            // R: component i of R^T v is dot(column_i(R), v), transpose free.
            vec3 rel = c - ${inputs.ucenter} - ${inputs.uoff};
            vec3 p = ${inputs.ucenter} + vec3(dot(rel, ${inputs.rc0}),
                                              dot(rel, ${inputs.rc1}),
                                              dot(rel, ${inputs.rc2}));
            float sd = ${inputs.dir} * (${sd});
            float t = clamp(sd / max(${inputs.trans}, 1e-5), 0.0, 1.0);
            t = t * t * (3.0 - 2.0 * t) * ${inputs.on} * ${inputs.strength};
            ${outputs.gsplat}.scales =
              mix(${inputs.gsplat}.scales, vec3(${inputs.psize}), t);
          `);
        },
      });
      return { gsplat: d.apply({ gsplat, ...uniforms }).gsplat };
    });
  splats.updateGenerator();   // one-time pipeline recompile at attach
  applyFocus();

  // Camera-inside bypass (default mode only): strength = smoothstep of the
  // CAMERA's signed distance over the same transition band — spatial and
  // stateless, mirroring the GLSL. Runs every frame; regenerates only when
  // the value actually moves.
  const camLocal = new THREE.Vector3();
  const camRel = new THREE.Vector3();   // separate: the dots must read the
                                        // pre-write value, not a half-updated
                                        // camLocal mid-chain
  const sdBoxCpu = (p, lo2, hi2) => {
    const qx = Math.abs(p.x - 0.5 * (lo2.x + hi2.x)) - 0.5 * (hi2.x - lo2.x);
    const qy = Math.abs(p.y - 0.5 * (lo2.y + hi2.y)) - 0.5 * (hi2.y - lo2.y);
    const qz = Math.abs(p.z - 0.5 * (lo2.z + hi2.z)) - 0.5 * (hi2.z - lo2.z);
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0))
         + Math.min(Math.max(qx, qy, qz), 0);
  };
  const updateFocusBypass = () => {
    const f = settings.focus;
    let g = 1;
    if (f.enabled && f.bypass && !f.invert) {
      camLocal.copy(camera.position);
      root.worldToLocal(camLocal);   // the scene frame, same as the boxes
      // ...and then into the volume's own frame, exactly as the shader does.
      // Skipping this would make the bypass fade at the WRONG boundary as
      // soon as the volume is turned — the same value computed two ways.
      camRel.subVectors(camLocal, unionCenter).sub(focusOffset.value);
      camLocal.set(camRel.dot(focusRotCol[0].value),
                   camRel.dot(focusRotCol[1].value),
                   camRel.dot(focusRotCol[2].value)).add(unionCenter);
      let sd = Infinity;
      for (const fb of focusBoxes)
        sd = Math.min(sd, sdBoxCpu(camLocal, fb.min.value, fb.max.value));
      // sd is in scene units, so the band is the shader's (object-space)
      g = Math.min(Math.max(sd / Math.max(focusTransition.value, 1e-5), 0), 1);
      g = g * g * (3 - 2 * g);
    }
    if (focusStrength.value === g) return;
    focusStrength.value = g;
    splats.updateVersion();
  };

  // Release the wires and neutralise the effect. The caller re-creates the
  // controller when the volume changes (the workbench builds it before the
  // volume gate has run), and the object modifier is a single assignment —
  // so the successor replaces it. What does NOT replace itself is the wire
  // set added to root, and a stub successor installs no modifier at all:
  // hence focusOn = 0, which leaves any still-installed shader inert.
  const dispose = () => {
    clearTimeout(focusFlashTimer);
    focusOn.value = 0;
    for (const fb of focusBoxes) {
      root.remove(fb.wire);
      fb.wire.geometry.dispose();
      fb.wire.material.dispose();
    }
    focusBoxes.length = 0;
    splats.updateVersion();
  };

  // unionCenter is exported because it is the pivot every adjustment turns
  // on: offset is measured from it, scale and rotation happen about it. A
  // manipulator that wants to drive those three needs the same pivot, and
  // recomputing it from the boxes elsewhere is how two definitions drift.
  const setDisplayScale = (s) => {
    displayScale = Number(s) > 0 ? Number(s) : 1;
    applyFocus();
  };

  /** Re-point the per-box uniforms at new box objects — the drag path:
   *  the same count of boxes, moved or resized, needs no rebuild (uniform
   *  writes only). Returns false when the count differs; the caller
   *  rebuilds, because the shader bakes one bmin/bmax pair per box. */
  const setBoxSources = (boxes) => {
    if (!boxes || boxes.length !== focusBoxes.length) return false;
    focusBoxes.forEach((fb, i) => { fb.src = boxes[i]; });
    applyFocus();
    return true;
  };
  const setOverride = (o) => { override = o || null; applyFocus(); };
  /** The scene alignment as nine numbers, row-major (null = identity):
   *  the shader carries each centre through it before the box test. */
  const setAlignment = (rows) => {
    const r = rows ?? [1, 0, 0, 0, 1, 0, 0, 0, 1];
    alignRow[0].value.set(r[0], r[1], r[2]);
    alignRow[1].value.set(r[3], r[4], r[5]);
    alignRow[2].value.set(r[6], r[7], r[8]);
    splats.updateVersion();
  };

  return { active: true, focusBoxes, focusRange, applyFocus,
           updateFocusBypass, setDisplayScale, dispose,
           setBoxSources, setOverride, setAlignment,
           unionCenter: unionCenter.clone() };
}
