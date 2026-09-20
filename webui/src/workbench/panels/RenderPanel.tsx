// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Render gate (1c/1d): stat row, native contact sheet (kept thumbs with M
// badges; discards collapsed with reason chips — served from cameras.json
// + the stage manifest, no sidecar needed), capture flow, re-render,
// approve. Selection is two-way with the 3D frustums.

import React, { useCallback, useMemo, useState } from "react";
import { api, runUrl } from "../../api";
import { sceneFactor, useCalibration, useJobRunning, useRefusal, useRun,
         useRunData, useSceneAct } from "../../store";
import { fmtUnits, lengthLabel } from "../../units";
import type { CamerasResp, Judged, ProposalResp, ResetPlan, VolumeResp }
  from "../../types";
import type { CanvasHandle } from "../SplatCanvas";
import { COL, cssHex } from "../overlays/common";
import { Badge, Btn, Confirm, Details, LabeledNum, PanelFooter, Section,
         Segmented, Spinner, Switch, cx, JobNote, Steps } from "../../ui";
import { currentStep, renderSteps } from "../../steps";
import { gateState, waitingOn } from "../../journal";

const PATH_MODES = ["interior", "orbit", "ground", "manual"] as const;
/** The camera strategies in plain words — the creation dialog's tiles:
 *  the selector shows one word each, the sentence under it says what the
 *  chosen one does, the profile keeps the config names. `auto` (no answer,
 *  only in a hand-written profile) is not offered: Carveout never proposes
 *  a path mode — the operator says how the scene was filmed. */
const MODE_WORDS: Record<string, string> = {
  interior: "room", orbit: "orbit", ground: "ground", manual: "manual",
};
const MODE_HELP: Record<string, string> = {
  interior: "Inside a space: eye-height cameras in the free space inside the "
    + "walls, aimed along the deep sightlines.",
  orbit: "Around a subject: a ring of cameras around the volume (the whole "
    + "scene without one) at three elevations: objects, benches, small "
    + "scenes.",
  ground: "Outdoors on the ground: person-height cameras over a local ground "
    + "map, aimed at obstacle mass: exteriors, sloped or open.",
  manual: "Manual views only: exactly the views you placed on the canvas; "
    + "nothing is sampled.",
  auto: "No filming answer yet: pick how the scene was filmed above; a "
    + "render refuses until you do.",
};
const modeOf = (word: string) =>
  (Object.keys(MODE_WORDS).find((k) => MODE_WORDS[k] === word) ?? word);
const CAMERA_MODES = ["translate", "rotate"] as const;

export default function RenderPanel({ cams, camsError, readOnly, canvas,
                                      selFrame, onSelectFrame, onChanged }: {
  cams: CamerasResp | null;
  camsError: string | null;
  readOnly: boolean;
  canvas: React.RefObject<CanvasHandle>;
  selFrame: number | null;
  onSelectFrame: (i: number | null) => void;
  onChanged: () => void;
}) {
  const { state } = useRun();
  const running = useJobRunning();
  const [showDiscards, setShowDiscards] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [camMode, setCamMode] = useState<"translate" | "rotate">("translate");
  const [resetAsk, setResetAsk] = useState<ResetPlan | null>(null);
  const { calib } = useCalibration();
  const manualMode = calib?.render?.path_mode === "manual";
  const selManual = selFrame !== null && String(
    cams?.frames.find((f) => f.frame_idx === selFrame)?.provenance ?? ""
  ).startsWith("manual:");

  const refuse = useRefusal();
  const { act, approveGate, startStage, busy } = useSceneAct();
  // the render and its approval wait for the volume gate; the panel
  // stays open for inspection, the buttons say what they wait for.
  const waiting = waitingOn(state.journal, "render");

  const render = (force = false) => startStage("render", { force });
  const approve = () => approveGate("render");

  const savePose = async () => {
    // no label: the server names it from the SAVED views. Naming it here
    // used the RENDERED count, which does not move until a re-render, so
    // every capture taken before one got the same name.
    const pose = canvas.current!.capturePose();
    // The section STAYS open: its own copy says "save as many as you
    // need first", and closing after each save contradicted it — one
    // re-click per pose. Done is the explicit way out.
    // The count in the Capture section reads the viewpoint file, which a
    // save changes without a journal event: reload it, or the count stood
    // still as the operator saved (the line above the section said the
    // number, but the section scrolls itself to the top and hides it).
    if (await act("/views", pose)) { onChanged(); views.reload(); }
  };

  // Auto views are regenerated every render, so adjusting one in place is not
  // a thing the pipeline can honour. Converting the whole set makes the render
  // reproduce exactly what is on disk, which is what makes an adjustment
  // stick. The server chains the re-render into the same job, so the frames
  // come back already manual — no second act needed here.
  const convert = () =>
    act("/views/convert").then((r) => r && onChanged());
  const askReset = () =>
    api.get<ResetPlan>(runUrl(state.scene, "/views/reset"))
      .then(setResetAsk).catch(refuse);
  const doReset = () => {
    setResetAsk(null);
    act("/views/reset").then((r) => {
      if (r) { onSelectFrame(null); onChanged(); }
    });
  };

  // Capture view opens its section ABOVE the contact sheet: with the sheet
  // scrolled down, the section mounted off-screen and the click looked dead.
  // Stable identity so the ref fires on mount only — never on the re-renders
  // each Save pose causes, which would yank the operator back mid-scroll.
  const revealCapture = useCallback((el: HTMLDivElement | null) => {
    el?.scrollIntoView({ block: "start" });
  }, []);

  const pending = cams?.manual_pending?.length ?? 0;
  // views deleted since the render: struck on the sheet by frame
  const deletedIdx = useMemo(() => new Set(
    (cams?.manual_pending ?? [])
      .filter((p) => p.why === "deleted" && p.frame_idx != null)
      .map((p) => p.frame_idx as number)), [cams]);
  // the viewpoints on disk, rendered or not: before any render exists
  // they are what the manual mode has to render
  const views = useRunData<{ views: { label?: string }[] }>("/views");
  const savedViews = views.data?.views ?? [];
  const saved = savedViews.length;
  const stepsNow = renderSteps({ hasCams: !!cams, pending, manual: manualMode,
    saved, approved: gateState(state.journal, "render") === "approved" });
  const cue = currentStep(stepsNow);

  const cov = cams?.coverage?.frac_final;
  // Thresholds come from render.health in the config, echoed on the payload,
  // so this banner and the decision that stages a render cannot drift apart.
  // Fallbacks match the previous literals for a manifest written before the
  // field existed.
  const minViews = cams?.health?.min_views ?? 15;
  const minCov = cams?.health?.min_coverage ?? 0.9;
  const lowCov = !!cams && cov !== undefined && cov !== null && cov < minCov;
  const lowViews = !!cams && cams.views_kept < minViews;
  const lowCoverage = lowCov || lowViews;

  // tight focus is for a bench-top subject: with a box that IS the room the
  // stand-off positions collapse to one spot (the garage: nine views in a
  // corner) — said in the thin-render note when Propose read
  // the box as the whole room
  const proposal = useRunData<ProposalResp>("/proposal");
  const focusOnRoom = calib?.render?.cameras_in_volume === false
    && calib?.render?.focus_aim === true
    && proposal.data?.proposal?.settings?.cameras_in_volume === true;
  const staged = cams?.staged ?? null;
  const stagedAct = (what: "publish" | "discard") =>
    act(`/render/staged/${what}`).then((r) => r && onChanged());
  const pct = (v: number | null) => v === null ? "n/a" : `${(100 * v).toFixed(0)}%`;

  return (
    <div>
      <Section title="Render gate" band>
        {staged && (
          <div className="mb-2 ui-note-warn">
            <div className="flex items-center gap-2 mb-1">
              <Badge tone="warn">Staged, not published</Badge>
              <span className="text-t3 font-mono">{staged.staged_at}</span>
            </div>
            <div>
              The last re-render came out worse than the render on disk, so it
              was <b>not published</b>; your render below is untouched.
            </div>
            <div className="mt-1 font-mono text-[0.75rem] text-t2 ui-num">
              <div>staged&nbsp;&nbsp;&nbsp;: {staged.staged.views_kept} views,
                coverage {pct(staged.staged.coverage)}</div>
              <div>published: {staged.published.views_kept} views,
                coverage {pct(staged.published.coverage)}</div>
              <div className="text-t3">healthy is ≥ {staged.staged.min_views}
                {" "}views and coverage ≥ {pct(staged.staged.min_coverage)}</div>
            </div>
            <div className="mt-2 flex gap-2 flex-wrap">
              <Btn variant="outline-primary" disabled={readOnly || running
                     || busy}
                   onClick={() => stagedAct("publish")}>
                Publish staged render</Btn>
              <Btn variant="outline" disabled={readOnly || running || busy}
                   onClick={() => stagedAct("discard")}>
                Discard staged</Btn>
            </div>
            <div className="mt-1 text-t3">
              Publishing replaces the render below and re-opens this gate and
              every gate after it. Re-rendering discards the staged one.
            </div>
          </div>
        )}
        {!cams && !camsError && <Spinner label="waiting for the render…" />}
        {camsError && (
          /* No render yet. The camera set comes from the filmed answer
           * below — Carveout's own placement, or the viewpoints you
           * capture on the canvas (manual), which needs no render first
           * (the panel used to wait for one). */
          <div className="ui-help">
            {manualMode
              ? (saved > 0
                  ? `no render yet; ${saved} viewpoint${saved === 1 ? "" : "s"} saved; render them.`
                  : "no render yet; capture your viewpoints on the canvas, then render them.")
              : "no render yet; run the render stage."}
            <Steps steps={stepsNow} />
            {waiting && <div className="ui-note-warn mt-2">{waiting}</div>}
            <div className="mt-2 flex gap-2 flex-wrap">
              {manualMode && (
                <Btn variant={saved > 0 ? "outline" : "primary"}
                     disabled={readOnly || running}
                     onClick={() => setCapturing(true)} kbd="C">Capture view</Btn>
              )}
              <Btn variant={manualMode && saved === 0 ? "outline" : "primary"}
                   disabled={readOnly || running || !!waiting
                             || (manualMode && saved === 0)}
                   title={waiting ?? (manualMode && saved === 0
                          ? "save at least one viewpoint first" : undefined)}
                   onClick={() => render(false)}>
                {manualMode ? "Render the viewpoints" : "Render views"}</Btn>
            </div>
          </div>
        )}
        {cams?.refused && (
          /* The refusal path's counterpart of the staged banner: a
           * re-render that discarded every view publishes nothing and the
           * earlier render stands — but the panel went on showing that
           * earlier render (sheet, frustums, Approve) under a banner saying
           * all views were discarded, with nothing to say which render was
           * which. */
          <div className="mb-2 ui-note-fail">
            <div className="flex items-center gap-2 mb-1">
              <Badge tone="fail">Last attempt refused</Badge>
              <span className="text-t3 font-mono">
                {cams.refused.refused_at}</span>
            </div>
            <div>
              The last render kept <b>{cams.refused.views_kept} of{" "}
              {cams.refused.views_attempted ?? "?"}</b> views under the
              current thresholds, so nothing was published. Everything
              below (the contact sheet, the cameras in 3D, the discards)
              is the <b>earlier render</b> ({cams.refused.rendered_at}),
              which stands untouched.
            </div>
            {Object.keys(cams.refused.thresholds_changed).length > 0 && (
              <div className="mt-1 font-mono text-[0.75rem] text-t2 ui-num">
                <div className="text-t3">that render was made with:</div>
                {Object.entries(cams.refused.thresholds_changed).map(
                  ([k, v]) => (
                    <div key={k}>{QUALITY_LABELS[k] ?? k}{" "}
                      {String(v.rendered)}
                      <span className="text-t3"> (now {String(v.now)})</span>
                    </div>))}
              </div>
            )}
            <div className="mt-1 text-t3">
              Approving ships the earlier render. To render under the
              current thresholds, change one and <b>Re-render</b>; the
              refusal above names the knob and the value that admits a view.
            </div>
          </div>
        )}
        {cams && (
          <>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5
                            text-[0.8125rem] mb-2 ui-num">
              <span className="text-t3">Views</span>
              <span className="text-t1 font-mono">
                {cams.views_kept} kept · {cams.views_manual} manual ·{" "}
                {cams.discards.length} discarded</span>
              {cov != null && (
                <>
                  <span className="text-t3">Coverage</span>
                  <span className="text-t1 font-mono">
                    {(cov * 100).toFixed(1)}%</span>
                </>
              )}
              {pending > 0 && (
                <>
                  <span className="text-t3">Pending</span>
                  <span className="text-warn">
                    {pending} added, moved or deleted since</span>
                </>
              )}
            </div>
            <p className="ui-help m-0">
              Review sharpness, coverage and framing before approving.
            </p>
            <Steps steps={stepsNow} />
            <Details>
              Look for crisp close-ups without fog or smear, and the subject
              framed from varied heights and sides. Coverage near its
              target with few discards is healthy.
            </Details>
            {cams.frames_missing > 0 && (
              <div className="mt-2 ui-note-fail">
                {cams.frames_missing} of these {cams.frames.length} frames are
                no longer on disk; the contact sheet below is describing a
                render that does not exist. Re-render before approving; the
                gate refuses a review of missing views.
              </div>
            )}
            {lowCoverage && cams && (
              <div className="mt-2 ui-note-warn">
                {lowViews && <>{cams.views_kept} views were kept, fewer
                  than {minViews}. </>}
                {lowCov && <>Coverage is {pct(cov ?? null)}, below the
                  {" "}{pct(minCov)} target. </>}
                Approving as is ships a thin render: detection sees only
                these views.
                {focusOnRoom && (
                  <div className="mt-1">
                    <b>Tight focus volume</b> is on, but Propose read this box
                    as the whole room: cameras then stand off the box and
                    crowd one spot. Switch it off below and re-render.
                  </div>
                )}
                <Details summary="What to do">
                  Open <b>Discarded</b> below to see what rejected the
                  views. If the scene's size, up axis or floor look wrong,
                  fix them at the Volume gate first; a wrong size discards
                  every view as "inside geometry". Otherwise capture the
                  missing views by hand with <b>Capture view</b>, allow more
                  views or repair rounds in the calibration below, or relax
                  the one threshold the discards name.
                </Details>
              </div>
            )}
          </>
        )}
      </Section>

      <ProposalCard readOnly={readOnly || running} locked={readOnly} />
      <CameraPathCalibration readOnly={readOnly || running} locked={readOnly}
                             autoQuality={cams?.auto_quality ?? null} />

      {/* A camera is adjustable only once the RENDER contains it, so say
        * that plainly rather than leaving the operator clicking a frustum
        * that will not respond. Convert re-renders on its own; this warning
        * is for poses captured or edited since the last render. */}
      {!readOnly && pending > 0 && (
        <div className="px-4 py-2 border-b border-line text-[0.75rem]
                        text-warn leading-4">
          {pending} view{pending === 1 ? " was" : "s were"} added, moved or
          deleted since this render. <b>Re-render</b> to apply{" "}
          {pending === 1 ? "it" : "them"}; a deleted view stays on the sheet
          and in 3D, struck, until then; a camera is adjustable in 3D only
          once the render contains it.
        </div>
      )}
      {/* Gated on the frame being MANUAL, exactly as the gizmo itself is —
        * not on the path mode. A hand-captured view is adjustable whatever
        * mode the scene is in, and the gizmo already treated it that way, so
        * gating this control on manualMode left the handles working with no
        * way to switch them from move to turn. */}
      {selManual && !readOnly && (
        <div className="px-4 py-2 border-b border-line flex items-center
                        gap-2 flex-wrap">
          <span className="text-[0.75rem] text-t2 flex-1 leading-4 min-w-[14ch]">
            Selected camera
            <span className="block text-t3">drag the handles in 3D; the
              gate re-opens on release</span>
          </span>
          <Segmented value={camMode} options={CAMERA_MODES}
                     label="camera handle mode"
                     onChange={(v) => {
                       setCamMode(v as "translate" | "rotate");
                       canvas.current?.setCameraGizmoMode(
                         v as "translate" | "rotate");
                     }} />
        </div>
      )}

      {capturing && (
        <div ref={revealCapture}>
        <Section title="Capture view">
          <p className="ui-help m-0 mb-2">
            Fly the canvas camera to frame the ROI; the live splat render IS
            the preview. No labels yet is expected before detect.
          </p>
          <p className="ui-help m-0 mb-2">
            Three steps: frame it, <b>Save pose</b>, then <b>Re-render</b>;
            a saved pose becomes a view only when the render runs again. Save
            as many as you need first; one re-render picks them all up.
          </p>
          {/* what is saved so far, where the operator is looking: the
            * running count and the names, and how many of them the
            * current render does not have yet */}
          <p className="ui-help m-0 mb-2">
            {saved === 0
              ? "No viewpoint saved yet."
              : <>
                  <b>{saved} viewpoint{saved === 1 ? "" : "s"} saved</b>
                  {cams && pending > 0 && <>, {pending} not in this render yet</>}
                  : {savedViews.map((v) => v.label ?? "unnamed").join(", ")}
                </>}
          </p>
          <div className="flex gap-2">
            <Btn variant="primary" onClick={savePose} kbd="⏎">
              Save pose</Btn>
            <Btn variant="ghost" onClick={() => setCapturing(false)}>
              Done</Btn>
          </div>
        </Section>
        </div>
      )}

      {cams && (
        <Section title={`Contact sheet (${cams.frames.length})`}
                 cue={cue === "review"}
                 right={
                   <button
                     className="text-[0.75rem] text-t3 hover:text-t1
                                underline underline-offset-2 whitespace-nowrap"
                     onClick={() => setShowDiscards(!showDiscards)}>
                     {showDiscards ? "Show kept" : `Discarded (${
                       cams.discards.length})`}
                   </button>}>
          {!showDiscards ? (
            <div className="grid grid-cols-3 gap-2">
              {cams.frames.map((f) => {
                const sel = selFrame === f.frame_idx;
                const manual = f.provenance?.startsWith("manual:");
                const gone = deletedIdx.has(f.frame_idx);
                return (
                  <button key={f.frame_idx}
                          aria-pressed={sel}
                          title={gone ? "deleted; Re-render drops it" : undefined}
                          onClick={() => onSelectFrame(
                            sel ? null : f.frame_idx)}
                          className={cx("rounded overflow-hidden border",
                                        "bg-inset text-left",
                                        gone && "opacity-40 line-through",
                                        sel ? "border-t1 ring-1 ring-t1"
                                            : "border-line hover:border-t3")}>
                    <img loading="lazy"
                         src={runUrl(state.scene,
                                     `/frames/${f.frame_idx}.png?v=${
                                       f.sharpness ?? 0}`)}
                         alt=""
                         className="w-full h-[72px] object-cover block" />
                    <span className="flex items-center gap-1 px-1.5 h-5
                                     font-mono text-[0.6875rem] text-t2">
                      {String(f.frame_idx).padStart(4, "0")}
                      {f.seen != null && (
                        <span className="text-t3"
                              title="share of the volume this view sees; views are in pick order, best first">
                          sees {(100 * f.seen).toFixed(0)}%</span>)}
                      {manual && (
                        <span className="truncate"
                              style={{ color: cssHex(COL.manual) }}>
                          M {f.label ?? ""}</span>
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <table className="w-full ui-mono text-[0.75rem]">
              <tbody>
                {cams.discards.map((d, i) => (
                  <tr key={i} className="border-b border-line">
                    <td className="py-1 text-t3">{d.round as string}</td>
                    <td className="py-1">
                      <Badge tone="warn">
                        {DISCARD_LABELS[d.reason as string] ?? d.reason}</Badge></td>
                    {/* the stat this view failed on, against its threshold
                      * — the knob to move is the one named by the reason;
                      * sharpness for every row said nothing about that */}
                    <td className="py-1 text-t3">
                      {describeJudged(d.judged ?? null, cams.scale.source)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      )}

      {resetAsk && (
        <Confirm title="Reset the render step" destructive
                 confirmLabel="Delete and start over"
                 onCancel={() => setResetAsk(null)} onConfirm={doReset}
                 body={
                   <>Deletes <b>all {resetAsk.views} manual view
                   {resetAsk.views === 1 ? "" : "s"}</b>, captured and
                   converted alike, and the {resetAsk.frames} rendered
                   frame{resetAsk.frames === 1 ? "" : "s"}. Path mode returns
                   to <code>{resetAsk.restores_to}</code>. To remove a single
                   view instead, open it and use Delete. This cannot be
                   undone.</>} />
      )}

      <PanelFooter>
        <Btn disabled={readOnly || running}
             variant={cue === "capture" ? "primary" : undefined}
             onClick={() => setCapturing(true)} kbd="C">Capture view</Btn>
        {!manualMode && (
          <Btn variant="outline" disabled={readOnly || running || !cams}
               onClick={convert}>
            Switch to fully manual</Btn>
        )}
        <Btn variant="ghost" disabled={readOnly || running}
             onClick={askReset}>Reset render step</Btn>
        <Btn variant={pending > 0 ? "primary" : "outline"}
             disabled={readOnly || running || !!waiting}
             title={waiting ?? undefined}
             onClick={() => render(true)} kbd="R">
          {pending > 0 ? `Re-render (${pending} to apply)` : "Re-render"}</Btn>
        <JobNote show={running} />
        <div className="flex-1" />
        {waiting && <span className="text-[0.75rem] text-warn text-right leading-tight">
          {waiting}</span>}
        <Btn variant="primary" kbd="⏎"
             disabled={readOnly || running || !cams || !!waiting}
             title={waiting ?? undefined}
             onClick={approve}>Approve render</Btn>
      </PanelFooter>
    </div>
  );
}

/** Display names for the quality thresholds. PRESENTATION ONLY — the server
 *  decides which knobs exist (derived from the discard reasons), so a key
 *  missing here still renders, under its raw name. A label cannot go stale
 *  into a knob the operator cannot reach, which is the failure this whole
 *  path exists to prevent. */
/** The discard reasons in words (the manifest keeps the ids). */
const DISCARD_LABELS: Record<string, string> = {
  blurry_mush: "blurry",
  inside_geometry: "inside geometry",
  near_field_fog: "fog in the near field",
  uniform_frame: "no detail (uniform frame)",
  low_coverage: "mostly empty",
};
const QUALITY_LABELS: Record<string, string> = {
  near_field_depth: "fog horizon",
  max_near_alpha: "max near-alpha (fog)",
  min_sharpness: "min sharpness",
  min_median_depth: "min median depth",
  min_rgb_std: "min RGB spread",
  min_coverage: "min frame coverage",
};
/** The knobs that are LENGTHS. Their field holds the profile's number — a
 *  metre setting the stage divides by the scene's one factor. Under a
 *  RECORDED factor the label says "(m)"; under the estimate no metres are
 *  spoken (non-metric scenes show units everywhere),
 *  and the hint beside the field gives what the number resolves to in
 *  scene units, which is what the filter actually compares. */
const LENGTH_KEYS = new Set(["near_field_depth", "min_median_depth",
                             "min_pos_sep"]);
/** The judged stats in words, for the discard table. */
const STAT_LABELS: Record<string, string> = {
  median_depth: "median depth", near_alpha: "near-alpha",
  sharpness: "sharpness", coverage: "frame coverage", rgb_std: "RGB spread",
};
const fmtStat = (v: number | null | undefined) =>
  v == null ? "?" : Number(v.toPrecision(3)).toString();
/** "median depth 0.727 m < min 1 m" — the failing value against its
 *  threshold, the comparison written the way the view failed it. Distances
 *  read in metres, "≈m" after a measurement. */
function describeJudged(j: Judged | null, source: "recorded" | "measured") {
  if (!j) return null;
  const name = STAT_LABELS[j.stat] ?? j.stat;
  const lim = j.sense === "above" ? "min" : "max";
  const op = j.sense === "above" ? "<" : ">";
  const v = j.value, t = j.threshold;
  const unit = j.distance ? (source === "measured" ? " ≈m" : " m") : "";
  return `${name} ${fmtStat(v)}${unit} ${op} ${lim} ${fmtStat(t)}${unit}`;
}

/** Refusal-first order: the two knobs that answer the discard reasons a
 *  close-quarters scene actually trips come first, the rarer wall-stare and
 *  coverage bars last. Server keys with no opinion here sort after, by name,
 *  so a newly added discard reason appears rather than vanishing. */
const QUALITY_ORDER = Object.keys(QUALITY_LABELS);
const qualityKeys = (qf: Record<string, number | "auto" | null>) =>
  Object.keys(qf).sort((a, b) => {
    const ia = QUALITY_ORDER.indexOf(a), ib = QUALITY_ORDER.indexOf(b);
    return (ia < 0 ? QUALITY_ORDER.length : ia)
         - (ib < 0 ? QUALITY_ORDER.length : ib) || a.localeCompare(b);
  });

/** The filming answers in the tiles' words, for the check line. */
const FILMED: Record<string, string> = {
  interior: "inside a space", orbit: "around a subject",
  ground: "outdoors on the ground", manual: "manual views", auto: "no answer yet",
};

/** The Render gate's "Check and proposal" card: the filming
 *  answer with the geometry's line under it — stated, never applied —
 *  then the two measured proposals that remain (tight focus on a subject
 *  inside a room; the lowered distance thresholds on a small scene, in
 *  metres) with ONE Adopt writing every key in one `PUT /calibration`;
 *  "adopted" when the profile already holds them. From `proposal.json`,
 *  written by the volume gate's Propose. The card writes nothing by
 *  itself, and it never proposes a path mode: the operator said how the
 *  scene was filmed. */
function ProposalCard({ readOnly, locked }:
                      { readOnly: boolean; locked: boolean }) {
  const proposal = useRunData<ProposalResp>("/proposal");
  const { calib, write } = useCalibration();
  const p = proposal.data;
  const rn = calib?.render;
  if (!p || !rn) {
    return (
      <Section title="Check and proposal" locked={locked}>
        <p className="ui-help m-0">
          {p ? "waiting for the calibration…"
             : "No proposal yet: press Propose fresh at the volume gate; the "
               + "check of your filming answer and the settings it proposes "
               + "appear here."}
        </p>
      </Section>
    );
  }
  const pr = p.proposal;
  const ck = p.check.path_mode;
  const factor = sceneFactor(calib);
  const approx = calib?.scene.measured ? "≈" : "";
  const focus = !pr.settings.cameras_in_volume && pr.settings.focus_aim;
  const thr = Object.entries(pr.thresholds);
  const matches = rn.cameras_in_volume === pr.settings.cameras_in_volume
    && rn.focus_aim === pr.settings.focus_aim
    && thr.every(([k, v]) => rn.quality_filter[k] === v.proposed);
  const nothing = !focus && !thr.length;
  const adopt = () => {
    const update: Record<string, unknown> = {
      interior: { cameras_in_volume: pr.settings.cameras_in_volume,
                  focus_aim: pr.settings.focus_aim },
    };
    if (thr.length)
      update.quality_filter = Object.fromEntries(
        thr.map(([k, v]) => [k, v.proposed]));
    write({ render: update });
  };
  // The check was made against the filming answer at propose time; a
  // changed answer since is said so, not judged.
  const stale = ck.said !== rn.path_mode;
  return (
    <Section title="Check and proposal" locked={locked}
             right={<span className="text-[0.6875rem] text-t4 font-mono">
               {p.generated}</span>}>
      <p className="m-0 text-[0.8125rem] text-t1 leading-5">
        filmed <b>{FILMED[rn.path_mode] ?? rn.path_mode}</b>
        <span className="text-t4 font-mono"> ({rn.path_mode})</span>
      </p>
      {stale ? (
        <p className="ui-help mt-1 mb-0">
          The last proposal checked "{FILMED[ck.said] ?? ck.said}"; press
          Propose fresh at the volume gate to check this answer.
        </p>
      ) : (
        <>
          <p className={cx("mt-1 mb-0 text-[0.75rem] leading-4",
                           ck.fits === false ? "text-warn" : "text-t3")}>
            {ck.fits === false && <b>flagged: </b>}{ck.reason}
            {ck.fits === false && " (change it below if you agree; Carveout never does)"}
          </p>
          {/* the room's height at this scale, under "inside a space": a
            * flag here is as often a wrong filming answer as a wrong
            * measurement, so the render panel shows it too */}
          {p.check.scale?.fits === false && (
            <p className="mt-1 mb-0 text-[0.75rem] leading-4 text-warn">
              <b>flagged: </b>{p.check.scale.reason}
            </p>
          )}
        </>
      )}
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 mt-2
                      text-[0.75rem] font-mono text-t3 ui-num">
        <span>tight focus</span>
        <span className="text-t2">
          {focus ? "on: cameras roam the room, aim at the volume"
                 : "off: cameras inside the volume"}</span>
        {thr.map(([k, v]) => (
          <React.Fragment key={k}>
            <span>{QUALITY_LABELS[k] ?? k}</span>
            <span className="text-t2">
              {v.current} → <b>{approx}{v.proposed}</b> m
              {factor && factor !== 1 && (
                <span className="text-t4"> = {fmtUnits(v.proposed_units)} units</span>)}
            </span>
          </React.Fragment>
        ))}
      </div>
      {pr.reasons.length > 0 && (
        <ul className="ui-help mt-2 mb-0 pl-4 list-disc">
          {pr.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}
      <div className="flex items-center gap-2 mt-2">
        <span className="ui-help m-0 flex-1">
          {matches ? (nothing ? "Nothing to change: the defaults fit this scene."
                              : "The profile holds this proposal.")
                   : "Adopt writes" + (focus ? " the focus keys" : " the focus defaults")
                     + (thr.length ? " and the thresholds" : "")
                     + " to the profile and re-opens this gate. It is a choice:"
                     + " without it, Render uses the profile as it stands."}
        </span>
        <Btn variant={matches ? "outline" : "primary"}
             disabled={readOnly || matches}
             title={matches ? "the profile already holds this proposal"
                            : "one write of every key above"}
             onClick={adopt}>{matches ? "adopted" : "Adopt"}</Btn>
      </div>
    </Section>
  );
}

/** Camera-path calibration: the knobs that decide WHERE cameras go and
 * which views survive — mis-set defaults (path_mode auto->orbit; a
 * room-scale placement on a tight focus volume) starved a render.
 * Editing re-opens the render gate. */
function CameraPathCalibration({ readOnly, locked, autoQuality }:
    { readOnly: boolean; locked: boolean;
      autoQuality: Record<string, number | null> | null }) {
  const { calib, write } = useCalibration();
  // The scene's factor for the length labels: recorded from the profile,
  // else the estimate the volume read carries (null before any proposal).
  const vol = useRunData<VolumeResp>("/volume");
  const rn = calib?.render;
  if (!rn) return null;
  const factor = sceneFactor(calib) ?? vol.data?.scale?.scale ?? null;
  const approx = !!calib?.scene.measured;
  const len = (k: string, v: number | null | undefined) =>
    lengthLabel(QUALITY_LABELS[k] ?? k, v, factor, approx);
  const focus = rn.cameras_in_volume === false && rn.focus_aim === true;
  return (
    <Section title="Camera path calibration" locked={locked}>
      <p className="ui-help m-0 mb-2">
        Where cameras go and which views survive. Editing re-opens the
        render gate.
      </p>
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className="ui-label flex-1">filmed</span>
        <Segmented value={MODE_WORDS[rn.path_mode] ?? rn.path_mode}
                   options={PATH_MODES.map((m) => MODE_WORDS[m])}
                   label="filmed"
                   disabled={readOnly}
                   onChange={(v) => write({ render: { path_mode: modeOf(v) } })} />
      </div>
      <p className="ui-help m-0 mb-2 -mt-1">
        {MODE_HELP[rn.path_mode] ?? `${rn.path_mode}: a path mode this panel has no words for.`}
        <span className="text-t4 font-mono"> ({rn.path_mode})</span>
      </p>
      <div className="flex items-center gap-2 mb-2">
        <span className="ui-label flex-1">
          tight focus volume
          <span className="block text-t3">cameras roam the room, aim at the
            box (off = the whole room)</span>
        </span>
        <Switch checked={focus} disabled={readOnly} label="tight focus volume"
                onChange={(on) => write({ render: { interior: {
                  cameras_in_volume: !on, focus_aim: on } } })} />
      </div>
      {qualityKeys(rn.quality_filter).map((k) => {
        const raw = rn.quality_filter[k];
        // "auto": resolved from the scene's width at render. The field
        // shows what it resolved to (the render's record, in the metres
        // the field speaks) so the number is there to read and to type
        // from; typing writes a number, the button writes "auto" back.
        // Without a render yet there is no number: the field is empty and
        // the hint says so.
        const isAuto = raw === "auto";
        const autoable = (rn.quality_auto ?? []).includes(k);
        const resolved = isAuto
          ? (autoQuality?.[`${k}_m`] ?? autoQuality?.[k] ?? null) : null;
        const v = isAuto ? resolved : (raw as number | null);
        const l = LENGTH_KEYS.has(k) ? len(k, v)
          : { label: QUALITY_LABELS[k] ?? k, hint: undefined };
        const hint = autoable ? (
          <>
            {isAuto && (
              <span className="text-t4 text-[0.6875rem] whitespace-nowrap">
                {resolved == null ? "auto, resolved at render" : "auto ="}
              </span>)}
            {l.hint}
            <Btn variant={isAuto ? "outline-primary" : "ghost"}
                 className="px-1.5 py-0 text-[0.6875rem]"
                 disabled={readOnly || isAuto}
                 title={isAuto ? "resolved from the scene's width at render"
                               : "resolve it from the scene's width at render"}
                 onClick={() => write({ render: {
                   quality_filter: { [k]: "auto" } } })}>auto</Btn>
          </>) : l.hint;
        return (
          <LabeledNum key={k} label={l.label} hint={hint} value={v}
                      disabled={readOnly}
                      onCommit={(nv) => write({ render: {
                        quality_filter: { [k]: nv } } })} />
        );
      })}
      <LabeledNum {...lengthLabel("min pose separation", rn.min_pos_sep,
                                  factor, approx)}
                  value={rn.min_pos_sep}
                  disabled={readOnly}
                  onCommit={(v) => write({ render: {
                    diversity: { min_pos_sep: v } } })} />
      <p className="ui-help m-0 mt-2 mb-1">
        The first round renders more candidates than the budget and keeps
        the ones that together see the most of the volume (each thumbnail
        shows its share); repair rounds then add views where the volume is
        still unseen, until the coverage target or the round limit.
      </p>
      <LabeledNum label="views (first round)" value={rn.num_views}
                  disabled={readOnly}
                  onCommit={(v) => write({ render: {
                    num_views: Math.max(1, Math.round(v)) } })} />
      <LabeledNum label="candidates per view kept" value={rn.candidate_factor}
                  hint={<span className="text-[0.75rem] text-t4">1 = first come</span>}
                  disabled={readOnly}
                  onCommit={(v) => write({ render: {
                    candidate_factor: Math.max(1, v) } })} />
      <LabeledNum label="coverage target (%)"
                  value={Math.round(100 * rn.coverage.target_frac)}
                  disabled={readOnly}
                  onCommit={(v) => write({ render: { coverage: {
                    target_frac: Math.min(1, Math.max(0, v / 100)) } } })} />
      <LabeledNum label="repair rounds (max)" value={rn.coverage.max_extra_rounds}
                  disabled={readOnly}
                  onCommit={(v) => write({ render: { coverage: {
                    max_extra_rounds: Math.max(0, Math.round(v)) } } })} />
      <LabeledNum label="views per repair round" value={rn.coverage.round_views}
                  disabled={readOnly}
                  onCommit={(v) => write({ render: { coverage: {
                    round_views: Math.max(1, Math.round(v)) } } })} />
    </Section>
  );
}
