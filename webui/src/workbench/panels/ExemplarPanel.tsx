// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Exemplar gate (1f): concept list + exemplar score rows, the 2D crop
// drawing surface (the ONE-OBJECT rule pinned — the twice-validated
// crop-semantics lesson), threshold overrides, fold re-probe, continue.
// Crop provenance renders in 3D as frustum + frame-plane markers.

import { useRef, useState } from "react";
import { runUrl } from "../../api";
import { useJobRunning, useRun, useRunData,
         useSceneAct } from "../../store";
import { gateState, waitingOn } from "../../journal";
import { currentStep, exemplarSteps } from "../../steps";
import type { CamerasResp, ExemplarsResp, ProbeResults } from "../../types";
import { Badge, Btn, Details, INPUT, NumInput, PanelFooter, Section, cx,
         JobNote, Steps } from "../../ui";

export default function ExemplarPanel({ readOnly, cams, data, reload,
                                        selFrame, onSelectFrame }: {
  readOnly: boolean;
  cams: CamerasResp | null;
  data: ExemplarsResp | null;
  reload: () => void;
  selFrame: number | null;
  onSelectFrame: (i: number | null) => void;
}) {
  const { state } = useRun();
  const running = useJobRunning();
  const probe = useRunData<ProbeResults>("/probe/results");
  const [drawing, setDrawing] = useState<string | null>(null);  // concept
  const [concept, setConcept] = useState("");

  const { act, approveGate, startStage } = useSceneAct();

  const reprobe = () => startStage("reprobe_exemplars");
  const approve = () => approveGate("exemplars");
  // the fold and Continue wait for the gates before this one
  const waiting = waitingOn(state.journal, "exemplars");
  const setThreshold = (c: string, v: number) =>
    act("/exemplars/thresholds", { [c]: v }, { method: "put" });

  const exemplarRows = probe.data?.concepts.filter(
    (c) => c.concept.startsWith("exemplar:")) ?? [];
  const nCrops = data?.exemplars.reduce((n, e) => n + e.crops.length, 0) ?? 0;
  const probedCurrent = data?.probed_current ?? true;
  const steps = exemplarSteps({
    crops: nCrops, probedCurrent,
    approved: gateState(state.journal, "exemplars") === "approved" });
  const cue = currentStep(steps);

  return (
    <div>
      <Section title="Exemplar gate" band>
        <p className="ui-help m-0">
          Only for concepts the text probe proved unreachable; if nothing
          is blocked, continue.
        </p>
        <Steps steps={steps} />
        <Details>
          Re-probe is quick: the text results stay, only the example-image
          pass runs again and its rows join the probe results.
        </Details>
        {(() => {
          const cap = probe.data?.exemplar_frame_cap;
          if (!cap || (cap.configured == null &&
              !(cap.used != null && cap.total != null && cap.used < cap.total)))
            return null;
          return (
            <div className="mt-2 ui-note-muted font-mono">
              VRAM cap · exemplar tracking uses{" "}
              {cap.used != null && cap.total != null
                ? <b className="text-t1">{cap.used} of {cap.total}</b>
                : "all"} frames
              {cap.peak_alloc_gb != null && <> · peak {cap.peak_alloc_gb} GB</>}
              <Details className="font-sans">
                The frame cap follows the memory profile the server was
                started with{cap.profile && <> ({cap.profile})</>}; it is
                not edited here.
              </Details>
            </div>
          );
        })()}
      </Section>

      <Section title={`Concepts (${data?.exemplars.length ?? 0})`}
               locked={readOnly} cue={cue === "crops"}>
        {data?.exemplars.map((e) => {
          const thr = data.overrides[e.concept] ?? data.threshold;
          const row = exemplarRows.find(
            (r) => r.concept === `exemplar:${e.concept}`);
          return (
            <div key={e.concept}
                 className="border border-line rounded p-2 mb-2">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-[0.8125rem] text-t1 font-medium">
                  {e.concept}</span>
                <Badge>{e.crops.length} crop{e.crops.length !== 1 && "s"}
                </Badge>
                {row && (
                  <Badge tone={row.max >= thr ? "pass" : "warn"}>
                    track max {row.max.toFixed(3)}</Badge>
                )}
                <div className="flex-1" />
                {!readOnly && !running && (
                  <>
                    <Btn variant="ghost"
                         onClick={() => setDrawing(e.concept)}>+ Crop</Btn>
                    <button className="text-t3 hover:text-fail text-[0.8125rem]
                                       px-1 rounded-sm"
                            aria-label={`delete concept ${e.concept}`}
                            onClick={() =>
                              act(`/exemplars/${encodeURIComponent(e.concept)}`, undefined,
                                  { method: "del" })
                                .then((r) => r && reload())}>✕</button>
                  </>
                )}
              </div>
              <div className="flex gap-1 mt-1.5 flex-wrap">
                {e.crops.map((c, i) => (
                  <button key={i}
                          className="relative border border-line rounded
                                     overflow-hidden hover:border-t3"
                          onClick={() => onSelectFrame(c.frame_idx)}>
                    <CropThumb scene={state.scene} crop={c} />
                    {!readOnly && !running && (
                      <span
                        role="button"
                        aria-label="delete crop"
                        className="absolute top-0 right-0 bg-app/80
                                   text-t2 hover:text-fail px-1
                                   text-[0.6875rem] rounded-bl"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          act(`/exemplars/${encodeURIComponent(e.concept)}/crops/${i}`,
                              undefined, { method: "del" })
                            .then((r) => r && reload());
                        }}>✕</span>
                    )}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="ui-label">threshold</span>
                <NumInput
                  type="number" step={0.05} min={0} max={1}
                  value={thr}
                  aria-label={`${e.concept} threshold`}
                  disabled={readOnly || running}
                  className="w-[68px]"
                  onCommit={(v) => v !== thr &&
                    setThreshold(e.concept, v)} />
                {data.overrides[e.concept] !== undefined && (
                  <Badge tone="accent">Override</Badge>
                )}
              </div>
            </div>
          );
        })}
        {(!data || data.exemplars.length === 0) && (
          <div className="ui-help">
            none; draw one only for an object the text probe could not
            find</div>
        )}
        {!readOnly && !running && (
          <div className="flex gap-2 mt-2">
            <input
              className={cx(INPUT, "flex-1 min-w-0")}
              aria-label="new concept"
              placeholder="new concept (becomes the exported label)"
              value={concept}
              onChange={(e) => setConcept(e.target.value)} />
            <Btn disabled={!concept.trim()}
                 onClick={() => { setDrawing(concept.trim());
                                  setConcept(""); }}>Draw</Btn>
          </div>
        )}
      </Section>

      {drawing && cams && (
        <CropDrawer scene={state.scene} concept={drawing} cams={cams}
                    initialFrame={selFrame ?? cams.frames[0]?.frame_idx}
                    onDone={() => { setDrawing(null); reload(); }}
                    onCancel={() => setDrawing(null)} />
      )}

      <PanelFooter>
        <Btn disabled={readOnly || running || !data?.exemplars.length || !!waiting}
             variant={cue === "reprobe" ? "primary" : undefined}
             title={waiting ?? undefined}
             onClick={reprobe} kbd="R">Re-probe</Btn>
        <JobNote show={running} />
        <div className="flex-1" />
        {waiting && <span className="text-[0.75rem] text-warn text-right leading-tight">
          {waiting}</span>}
        <Btn variant="primary" kbd="⏎"
             disabled={readOnly || running || !!waiting || !probedCurrent}
             title={waiting ?? (!probedCurrent
                      ? "the crops changed since the last probe; Re-probe first"
                      : undefined)}
             onClick={approve}>Continue</Btn>
      </PanelFooter>
    </div>
  );
}

function CropThumb({ scene, crop }: {
  scene: string; crop: { frame_idx: number; box_xyxy: number[];
                         frame_sha256?: string };
}) {
  // draw the crop region only, via a canvas copy of the frame
  const ref = (el: HTMLCanvasElement | null) => {
    if (!el) return;
    const img = new Image();
    img.onload = () => {
      const [x1, y1, x2, y2] = crop.box_xyxy;
      const ctx = el.getContext("2d")!;
      ctx.drawImage(img, x1, y1, x2 - x1, y2 - y1, 0, 0, el.width,
                    el.height);
    };
    // cache-bust with the server-stamped content hash (stale-image quirk)
    img.src = runUrl(scene, `/frames/${crop.frame_idx}.png?v=${
      crop.frame_sha256?.slice(0, 12) ?? 0}`);
  };
  return <canvas ref={ref} width={56} height={56} className="block" />;
}

/** The crop drawing surface: rendered frame + drag box. THE RULE (pinned,
 * twice-validated): a crop is ONE object, tightly framed — never a
 * cluster, never the object plus its case or surroundings. */
function CropDrawer({ scene, concept, cams, initialFrame, onDone,
                      onCancel }: {
  scene: string; concept: string; cams: CamerasResp;
  initialFrame: number | undefined;
  onDone: () => void; onCancel: () => void;
}) {
  const { act } = useSceneAct();
  const [frameIdx, setFrameIdx] = useState(initialFrame ??
    cams.frames[0]?.frame_idx ?? 0);
  const [pending, setPending] = useState<
    { frame_idx: number; box_xyxy: number[] }[]>([]);
  const [drag, setDrag] = useState<{ x1: number; y1: number; x2: number;
                                     y2: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const frame = cams.frames.find((f) => f.frame_idx === frameIdx);
  const natW = frame?.width ?? cams.intrinsics.width;
  const natH = frame?.height ?? cams.intrinsics.height;

  const toNat = (e: React.PointerEvent) => {
    const r = imgRef.current!.getBoundingClientRect();
    return {
      x: Math.round(((e.clientX - r.left) / r.width) * natW),
      y: Math.round(((e.clientY - r.top) / r.height) * natH),
    };
  };

  const save = () =>
    act("/exemplars/crops", { concept, crops: pending })
      .then((r) => r && onDone());

  // Measured during render, so the first paint sees no img yet (scale 0 =
  // saved crops invisible); the img's onLoad bumps a re-render to fix it.
  const [, remeasure] = useState(0);
  const scale = imgRef.current
    ? imgRef.current.getBoundingClientRect().width / natW : 0;

  return (
    <Section title={`Draw crops · ${concept}`}>
      <div className="ui-note-warn mb-2">
        A crop is ONE object, tightly framed: never a cluster, never the
        object plus its case or surroundings.
      </div>
      <div className="flex items-center gap-2 mb-1.5">
        <select
          className={cx(INPUT, "flex-1 min-w-0")}
          aria-label="frame to draw on"
          value={frameIdx}
          onChange={(e) => setFrameIdx(+e.target.value)}>
          {cams.frames.map((f) => (
            <option key={f.frame_idx} value={f.frame_idx}>
              frame_{String(f.frame_idx).padStart(4, "0")}
              {f.provenance?.startsWith("manual:")
                ? ` · M ${f.label ?? ""}` : ""}
            </option>
          ))}
        </select>
        <span className="text-[0.75rem] text-t3 whitespace-nowrap">
          {pending.length} new</span>
      </div>
      <div className="relative select-none touch-none"
           onPointerDown={(e) => {
             const p = toNat(e);
             setDrag({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
             (e.target as HTMLElement).setPointerCapture(e.pointerId);
           }}
           onPointerMove={(e) => {
             if (!drag) return;
             const p = toNat(e);
             setDrag({ ...drag, x2: p.x, y2: p.y });
           }}
           onPointerUp={() => {
             if (!drag) return;
             const box = [Math.min(drag.x1, drag.x2),
                          Math.min(drag.y1, drag.y2),
                          Math.max(drag.x1, drag.x2),
                          Math.max(drag.y1, drag.y2)];
             if (box[2] - box[0] >= 4 && box[3] - box[1] >= 4)
               setPending([...pending, { frame_idx: frameIdx,
                                         box_xyxy: box }]);
             setDrag(null);
           }}>
        <img ref={imgRef}
             src={runUrl(scene, `/frames/${frameIdx}.png?v=${
               frame?.sharpness ?? 0}`)}
             onLoad={() => remeasure((n) => n + 1)}
             alt={`frame ${frameIdx}`}
             className="w-full block border border-line rounded
                        cursor-crosshair" draggable={false} />
        {scale > 0 && [...pending.filter((c) => c.frame_idx === frameIdx)
          .map((c) => c.box_xyxy), ...(drag
            ? [[Math.min(drag.x1, drag.x2), Math.min(drag.y1, drag.y2),
                Math.max(drag.x1, drag.x2), Math.max(drag.y1, drag.y2)]]
            : [])].map((b, i) => (
          <div key={i}
               className="absolute border-2 border-fail pointer-events-none"
               style={{ left: b[0] * scale, top: b[1] * scale,
                        width: (b[2] - b[0]) * scale,
                        height: (b[3] - b[1]) * scale }} />
        ))}
      </div>
      <div className="flex gap-2 mt-2 flex-wrap">
        <Btn variant="ghost"
             disabled={!pending.length}
             onClick={() => setPending(pending.slice(0, -1))}>
          Undo last</Btn>
        <div className="flex-1" />
        <Btn variant="ghost" onClick={onCancel}>Cancel</Btn>
        <Btn variant="primary" disabled={!pending.length} onClick={save}>
          Save {pending.length} crop{pending.length !== 1 && "s"}</Btn>
      </div>
    </Section>
  );
}
