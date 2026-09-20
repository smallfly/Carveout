// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Viewer-settings drawer (a RAIL node, not a top-bar item): the canvas's
// display settings, over the workbench canvas; manual views are not here
// (capture lives in the render gate). The panel binds to the canvas's LIVE
// settings object; every edit goes through the settings-effects table
// (viewer/lib/apply.js, via canvas.apply) — which applier follows which
// mutation is defined there once, never mirrored here. SAVE writes
// viewer_settings.json in the workdir (the scene's own display settings);
// unsaved changes stay session-local.

import React, { useEffect, useReducer, useRef, useState } from "react";
import { bypassLocked } from "@viewer/apply.js";
import { api, runUrl } from "../../api";
import { sceneFactor, useCalibration, useRun } from "../../store";
import { Btn, Details, PanelFooter, Section, Segmented, Slider, Switch, cx }
  from "../../ui";
import type { CanvasHandle } from "../SplatCanvas";
import type { FocusMode } from "../overlays";

export default function SettingsPanel({ canvas, instancesLayer,
                                        onInstancesLayer, focusEpoch,
                                        focusSelected, onFocusEdited }: {
  canvas: React.RefObject<CanvasHandle>;
  instancesLayer: boolean;
  onInstancesLayer: (on: boolean) => void;
  /** bumped by the focus manipulator — the poll below watches object
   *  identity, which a drag mutating values in place never changes */
  focusEpoch: number;
  /** the focus box is picked, so its handles are on the canvas */
  focusSelected: boolean;
  /** a slider moved the volume: the handles have to follow */
  onFocusEdited: () => void;
}) {
  const { state } = useRun();
  // The canvas works in display metres: real ones at the scene's factor
  // (recorded, or "≈" after a measurement), a nominal room-sized scale
  // before any scale exists — said so, rather than a unit nobody has.
  const { calib } = useCalibration();
  const U = sceneFactor(calib) ? (calib?.scene.measured ? "≈m" : "m")
                               : "m, nominal";
  const [, bump] = useReducer((n: number) => n + 1, 0);
  const [status, setStatus] = useState<string | null>(null);
  const seen = useRef<{ s: any; f: any; e: any }>({ s: null, f: null,
                                                    e: null });

  // The canvas builds its pieces asynchronously (settings fetch, volume
  // fetch -> focus controller, interactions fetch -> label engine) — poll
  // their identities and re-render when one lands.
  useEffect(() => {
    const t = setInterval(() => {
      const c = canvas.current;
      const s = c?.settings ?? null, f = c?.focus ?? null,
            e = c?.engine ?? null;
      if (s !== seen.current.s || f !== seen.current.f
          || e !== seen.current.e) {
        seen.current = { s, f, e };
        bump();
      }
    }, 300);
    return () => clearInterval(t);
  }, [canvas]);

  useEffect(bump, [focusEpoch]);

  // The manipulator follows SELECTION, which lives in the workbench next to
  // the detection volume's — the canvas attaches it. Nothing imperative here.
  const [focusMode, setFocusMode] = useState<FocusMode>("translate");
  const chooseFocusMode = (m: FocusMode) => {
    setFocusMode(m);
    canvas.current?.setFocusGizmoMode(m);
  };

  const c = canvas.current;
  const s = c?.settings;
  if (!c || !s) {
    return <div className="p-4 ui-help animate-pulse">
      canvas loading…</div>;
  }
  const stage = c.stage;
  const eng = c.engine;
  const focus = c.focus;
  /** one settings edit: through the shared effects table, then re-render */
  const edit = (path: string, v?: unknown) => {
    c.apply(path, v);
    bump();
  };
  // Every focus edit also tells the canvas — whose handles are seeded from
  // these values and would otherwise sit where the volume USED to be after
  // a slider move.
  const editFocus = (path: string, v?: unknown) => {
    edit(path, v);
    onFocusEdited();
  };
  const occl = localStorage.getItem("carveout_wb_occl") === "1";
  const worldMode = s.labels.scaleMode === "world";

  const save = async () => {
    try {
      await api.put(runUrl(state.scene, "/viewer_settings"), s);
      setStatus("saved for this scene");
    } catch (e: any) {
      setStatus(`save failed: ${e.message}`);
    }
  };

  return (
    <div>
      {/* no repeated title: the inspector header already says "Viewer
          settings" — just the one sentence that scopes the panel */}
      <p className="ui-help m-0 px-4 py-3 border-b border-line">
        Display settings for this scene only; a new scene starts from
        the defaults. Nothing here touches the gates or the pipeline;
        unsaved changes last for this session.
      </p>

      <Section title="Camera">
        <SliderRow label="field of view" min={30} max={120} step={1}
                   value={s.camera.fov} fmt={(v) => `${v}°`}
                   onChange={(v) => edit("camera.fov", v)} />
        <SliderRow label={`fly-to distance (${U})`} min={0.2} max={5} step={0.05}
                   value={s.camera.flyDistance} fmt={(v) => v.toFixed(2)}
                   onChange={(v) => edit("camera.flyDistance", v)} />
        <div className="flex gap-2 mt-1.5 flex-wrap">
          <Btn onClick={() => {
            // root-local, like manual views: the pose survives a factor
            // or display-scale change (stage.captureStartPose)
            Object.assign(s.camera, stage.captureStartPose());
            setStatus("start pose captured from current view; save to "
                      + "persist");
          }}>Set start pose</Btn>
          <Btn variant="ghost" onClick={() => stage.applyStartPose()}>
            Go to start pose</Btn>
        </div>
      </Section>

      <Section title="Scene">
        <ColorRow label="background color" value={s.background}
                  onChange={(v) => edit("background", v)} />
      </Section>

      <Section title="Labels">
        <ToggleRow label="show labels" checked={instancesLayer}
                   onChange={onInstancesLayer} />
        <ToggleRow label="show all boxes" checked={s.labels.showBoxes}
                   onChange={(v) => edit("labels.showBoxes", v)} />
        <OptionRow label="visibility mode" value={s.labels.mode}
                   options={["distance", "always"]}
                   onChange={(v) => edit("labels.mode", v)} />
        <SliderRow label={`max distance (${U})`} min={2} max={60} step={1}
                   value={s.labels.maxDistance} fmt={(v) => `${v}`}
                   disabled={s.labels.mode !== "distance"}
                   onChange={(v) => edit("labels.maxDistance", v)} />
        <SliderRow label="max shown" min={4} max={100} step={1}
                   value={s.labels.maxCount} fmt={(v) => `${v}`}
                   disabled={s.labels.mode !== "distance"}
                   onChange={(v) => edit("labels.maxCount", v)} />
        <SliderRow label="min confidence" min={0} max={1} step={0.01}
                   value={s.labels.minConfidence} fmt={(v) => v.toFixed(2)}
                   onChange={(v) => edit("labels.minConfidence", v)} />
        {/* Exemplar-rescued instances score on the exemplar-similarity
 scale — never comparable with text confidence (:
            one slider silently hid the ink bottles). Own floor. */}
        <SliderRow label="min conf (exemplar)" min={0} max={1} step={0.01}
                   value={s.labels.exemplarMinConfidence}
                   fmt={(v) => v.toFixed(2)}
                   onChange={(v) => edit("labels.exemplarMinConfidence", v)} />
        {/* one control for what a label shows, so the precedence is on
            the panel: a name you gave comes first under "verified" and
            "proposed"; "detected" is the raw SAM term for every object */}
        <OptionRow label="label source" value={s.labels.source} wide
                   options={["detected", "verified", "proposed"]}
                   onChange={(v) => edit("labels.source", v)} />
        <p className="ui-help mt-0 mb-1">
          {s.labels.source === "detected"
            ? "the term each object was detected as, for every object; names you gave and the verifier's verdicts are not shown"
            : s.labels.source === "proposed"
            ? "a name you gave, else the name the verifier proposed but could not apply (outside the vocabulary, too few views, held), else its verdict; a preview that applies nothing"
            : "a name you gave, else the verifier's verdict, else the detected term"}
        </p>
        <OptionRow label="scale mode" value={s.labels.scaleMode}
                   options={["screen", "world"]}
                   onChange={(v) => edit("labels.scaleMode", v)} />
        {/* worldSize stores live per tick; the re-bake fires on COMMIT
            (the table's labels.worldSize entry) */}
        <SliderRow label={`label world size (${U})`} min={0.02} max={0.3}
                   step={0.005} value={s.labels.worldSize}
                   fmt={(v) => v.toFixed(3)} disabled={!worldMode}
                   onChange={(v) => { s.labels.worldSize = v; bump(); }}
                   onCommit={() => c.apply("labels.worldSize")} />
        <OptionRow label="marker color" value={s.labels.markerColorMode}
                   options={["category", "custom"]}
                   onChange={(v) => edit("labels.markerColorMode", v)} />
        <ColorRow label="custom marker color" value={s.labels.markerColor}
                  disabled={s.labels.markerColorMode !== "custom"}
                  onChange={(v) => edit("labels.markerColor", v)} />
        <ColorRow label="pill background" value={s.labels.pillBackground}
                  onChange={(v) => edit("labels.pillBackground", v)} />
        <ToggleRow label="accent line" checked={s.labels.showAccent}
                   onChange={(v) => edit("labels.showAccent", v)} />
        <ToggleRow label="confidence score" checked={s.labels.showScore}
                   onChange={(v) => edit("labels.showScore", v)} />
        {/* Workbench-scoped (own key): settle-occlusion rays queue input
            behind ~150ms/ray-per-million-splats scans — right for reading
            the labels of a finished scene, wrong as a workbench default
            (seen in user testing). */}
        <ToggleRow label="label occlusion (slower on large scenes)"
                   checked={occl}
                   onChange={(v) => {
                     localStorage.setItem("carveout_wb_occl", v ? "1" : "0");
                     if (eng) eng.occlusionOn = v;
                     bump();
                   }} />
      </Section>

      <Section title="Focus volume (display effect)">
        <p className="ui-help m-0">
          Display only: splats outside the focus box collapse to points.
          This is <b>not</b> the detection volume; nothing in the pipeline
          reads it. The Volume gate's preview hands over to this switch
          when this panel opens.
        </p>
        <Details className="mb-1.5">
          The cyan box in the view is the detection volume; this one is
          separate. Show the box, then click it in the view to select it;
          handles appear on the selection and go when you click away.
          Turning is offered here and not on the detection volume: nothing
          downstream reads this, so a subject sitting at an angle can be
          framed without changing what gets detected.
        </Details>
        {!focus?.active ? (
          <div className="ui-help">
            no volume defined yet; the focus effect starts from the
            detection volume, so propose one at the Volume gate first</div>
        ) : (
          <>
            <ToggleRow label="outside the box as points"
                       checked={s.focus.enabled}
                       onChange={(v) => editFocus("focus.enabled", v)} />
            {["X", "Y", "Z"].map((ax, a) => (
              <SliderRow key={`o${ax}`} label={`offset ${ax} (${U})`}
                         min={-focus.focusRange} max={focus.focusRange}
                         step={0.01} value={s.focus.offset[a]}
                         fmt={(v) => v.toFixed(2)}
                         onChange={(v) => editFocus(`focus.offset.${a}`, v)} />
            ))}
            {["X", "Y", "Z"].map((ax, a) => (
              <SliderRow key={`s${ax}`} label={`scale ${ax}`} min={0.1}
                         max={4} step={0.01} value={s.focus.scale[a]}
                         fmt={(v) => v.toFixed(2)}
                         onChange={(v) => editFocus(`focus.scale.${a}`, v)} />
            ))}
            {/* Degrees, about the union centre. This turns the DISPLAY
                effect only — the detection volume is axis-aligned and stays
                that way, so a subject sitting at an angle can be framed
                without changing what gets detected. */}
            {["X", "Y", "Z"].map((ax, a) => (
              <SliderRow key={`r${ax}`} label={`rotate ${ax} (deg)`}
                         min={-180} max={180} step={0.5}
                         value={s.focus.rotation[a]}
                         fmt={(v) => v.toFixed(1)}
                         onChange={(v) =>
                           editFocus(`focus.rotation.${a}`, v)} />
            ))}
            <SliderRow label={`point size (${U})`} min={0.002} max={0.05}
                       step={0.001} value={s.focus.pointSize}
                       fmt={(v) => v.toFixed(3)}
                       onChange={(v) => editFocus("focus.pointSize", v)} />
            <SliderRow label={`transition (${U})`} min={0} max={2} step={0.01}
                       value={s.focus.transition} fmt={(v) => v.toFixed(2)}
                       onChange={(v) => editFocus("focus.transition", v)} />
            <ToggleRow label="invert (inside = points)"
                       checked={s.focus.invert}
                       onChange={(v) => editFocus("focus.invert", v)} />
            {/* the bypass-locks-while-inverted rule is bypassLocked
                (viewer/lib/apply.js) */}
            <ToggleRow label="camera-inside bypass" checked={s.focus.bypass}
                       disabled={bypassLocked(s)}
                       onChange={(v) => edit("focus.bypass", v)} />
            <ToggleRow label="show focus box + 3D handles"
                       checked={s.focus.showBox}
                       onChange={(v) => editFocus("focus.showBox", v)} />
            {s.focus.showBox && (
              <div className="flex items-center justify-between py-1 gap-2">
                <span className="text-[0.75rem] text-t2">
                  {focusSelected ? "Handle mode"
                                 : "Click the yellow box to select it"}</span>
                {focusSelected && (
                  <Segmented options={["move", "turn", "size"]}
                             label="focus handle mode"
                             value={focusMode === "translate" ? "move"
                                  : focusMode === "rotate" ? "turn" : "size"}
                             onChange={(v) => chooseFocusMode(
                               v === "move" ? "translate"
                               : v === "turn" ? "rotate" : "scale")} />
                )}
              </div>
            )}
          </>
        )}
      </Section>

      <Section title="Controls feel">
        <ToggleRow label="glide / inertia" checked={s.controls.inertia}
                   onChange={(v) => edit("controls.inertia", v)} />
        {/* Off = right-drag grabs the scene and it follows the cursor, which
            is what OrbitControls, Maya, Blender and every map do. On = the
            camera moves with the cursor instead. */}
        <ToggleRow label="invert pan (drag moves the camera, not the scene)"
                   checked={s.controls.reversePan}
                   onChange={(v) => edit("controls.reversePan", v)} />
      </Section>

      {status && (
        <div className="px-4 py-2 text-[0.75rem] text-t2" role="status">
          {status}</div>
      )}

      {/* the panel's ONE primary, in the inspector's fixed footer */}
      <PanelFooter>
        {focus?.active && (
          <Btn onClick={() => {
            c.resetFocus();   // shared helper (viewer/lib/apply.js)
            bump();
            onFocusEdited();
          }}>Reset to detection volume</Btn>
        )}
        <div className="flex-1" />
        <Btn variant="primary" onClick={save}>
          Save for this scene</Btn>
      </PanelFooter>
    </div>
  );
}

// -- rows: 2-col grid `label 1fr / control 168px`, 28px min row height
// (hit target), label sans 0.75rem t2, value readout right-aligned mono t1
// (values are data — brightest tier). Disabled = 45% opacity on the WHOLE
// row.

function Row({ label, disabled, wide, children }: {
  label: string; disabled?: boolean; children: React.ReactNode;
  /** the control takes what the label leaves (a three-way segmented
   *  control overflows the fixed 168 px the sliders align on) */
  wide?: boolean;
}) {
  return (
    <div className={cx(
      "grid items-center gap-2 min-h-[28px]",
      wide ? "grid-cols-[auto_1fr]" : "grid-cols-[1fr_168px]",
      disabled && "opacity-45 pointer-events-none")}>
      <span className="ui-label">{label}</span>
      <div className="flex items-center justify-end gap-2">{children}</div>
    </div>
  );
}

function SliderRow({ label, min, max, step, value, onChange, onCommit,
                     disabled, fmt }: {
  label: string; min: number; max: number; step: number; value: number;
  onChange: (v: number) => void; onCommit?: () => void;
  disabled?: boolean; fmt: (v: number) => string;
}) {
  return (
    <Row label={label} disabled={disabled}>
      <Slider min={min} max={max} step={step} value={value} label={label}
              disabled={disabled} className="flex-1"
              onChange={onChange} onCommit={onCommit} />
      <span className="font-mono text-[0.78rem] text-t1 w-12 text-right ui-num">
        {fmt(value)}</span>
    </Row>
  );
}

function ToggleRow({ label, checked, onChange, disabled }: {
  label: string; checked: boolean; onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <Row label={label} disabled={disabled}>
      <Switch checked={checked} onChange={onChange} disabled={disabled}
              label={label} />
    </Row>
  );
}

function OptionRow({ label, value, options, onChange, wide }: {
  label: string; value: string; options: string[];
  onChange: (v: string) => void; wide?: boolean;
}) {
  return (
    <Row label={label} wide={wide}>
      <Segmented value={value} options={options} onChange={onChange}
                 label={label} />
    </Row>
  );
}

function ColorRow({ label, value, onChange, disabled }: {
  label: string; value: string; onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <Row label={label} disabled={disabled}>
      <span className="font-mono text-[0.78rem] text-t1">{value}</span>
      <input type="color" value={value} disabled={disabled}
             aria-label={label}
             className="w-8 h-6 bg-inset border border-line2 rounded
                        cursor-pointer p-0.5"
             onChange={(e) => onChange(e.target.value)} />
    </Row>
  );
}
