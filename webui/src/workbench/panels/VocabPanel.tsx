// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Vocabulary gate (1e): list editor, LLM proposal behind
// explicit consent, probe loop, results table, negative-ceiling health
// line, which is REPORTED when VIOLATED rather than acknowledged
// (negatives never reach the output), confirm.

import { useState } from "react";
import { api, runUrl } from "../../api";
import { useJobRunning, useRefusal, useRun, useRunData,
         useSceneAct, useCalibration, useVlmModels } from "../../store";
import type { ProbeResults, VocabResp } from "../../types";
import { Badge, Btn, Confirm, Details, INPUT, PanelFooter, Section, cx,
         JobNote, Steps, ModelChoice } from "../../ui";
import { currentStep, vocabSteps } from "../../steps";
import { gateState, waitingOn } from "../../journal";

export default function VocabPanel({ readOnly }: { readOnly: boolean }) {
  const { state } = useRun();
  const running = useJobRunning();
  const vocab = useRunData<VocabResp>("/vocabulary");
  const probe = useRunData<ProbeResults>("/probe/results");
  const [showDistribution, setShowDistribution] = useState(false);
  // the two dialogs: start the step over; propose over a proposal
  const [resetAsk, setResetAsk] = useState<
    { prompts: number; negatives: number; proposal: boolean } | null>(null);
  const [proposeAsk, setProposeAsk] = useState(false);
  const refuse = useRefusal();

  const { act, startStage } = useSceneAct();
  const { calib, write } = useCalibration();
  const models = useVlmModels();

  const v = vocab.data;
  const nc = probe.data?.negative_ceiling ?? null;
  const violated = !!nc?.violated;

  const put = (prompts: string[], negatives: string[]) =>
    act("/vocabulary", { prompts, negatives }, { method: "put" })
      .then((r) => r && vocab.reload());

  // Adopting proposal terms is an EXPLICIT act (a click), so the
  // never-auto-adopt rule holds — these just spare the operator the
  // mouse-driven copy/paste of each term.
  const addPrompts = (terms: string[]) => {
    if (!v) return;
    put([...v.prompts, ...terms.filter((t) => !v.prompts.includes(t))],
        v.negatives);
  };
  const addNegatives = (terms: string[]) => {
    if (!v) return;
    // negatives must be a subset of prompts (server invariant)
    put([...v.prompts, ...terms.filter((t) => !v.prompts.includes(t))],
        [...v.negatives, ...terms.filter((t) => !v.negatives.includes(t))]);
  };
  // ... and the inverse in one click: add-all had no remove-all
  const removePrompts = (terms: string[]) => {
    if (!v) return;
    put(v.prompts.filter((p) => !terms.includes(p)),
        v.negatives.filter((n) => !terms.includes(n)));
  };
  const removeNegatives = (terms: string[]) => {
    if (!v) return;
    put(v.prompts, v.negatives.filter((n) => !terms.includes(n)));
  };

  const runProbe = () => startStage("probe");
  // the probe and the confirmation wait for the gates before them
  const waiting = waitingOn(state.journal, "vocabulary");

  const proposeLlm = () => act("/vocabulary/propose-llm", {});
  // a proposal on screen is replaced, not appended to: ask first
  const tryPropose = () => {
    if (v?.llm_proposal) setProposeAsk(true);
    else proposeLlm();
  };
  const askReset = () =>
    api.get<{ prompts: number; negatives: number; proposal: boolean }>(
      runUrl(state.scene, "/vocabulary/reset"))
      .then(setResetAsk).catch(refuse);
  const doReset = () => {
    setResetAsk(null);
    act("/vocabulary/reset").then((r) => { if (r) vocab.reload(); });
  };

  // the probe is the gate: no confirm without it
  const confirm = () => act("/vocabulary/confirm", {});

  const steps = vocabSteps({
    hasPrompts: !!v?.prompts.length, probedCurrent: !!v?.probed_current,
    approved: gateState(state.journal, "vocabulary") === "approved" });
  const cue = currentStep(steps);

  return (
    <div>
      <Section title="Vocabulary gate" band>
        <p className="ui-help m-0">
          Edit the list or propose one, probe it, then confirm. Detection
          never runs on an unreviewed vocabulary.
        </p>
        <Steps steps={steps} />
        <Details>
          Every scene starts fresh. Keep 2–3 deliberately-absent negatives:
          detected but never exported; they are the probe's health check.
        </Details>
      </Section>

      {v && (
        <>
          <ListEditor label={`Prompts (${v.prompts.length})`} cue={cue === "list"}
                      items={v.prompts} readOnly={readOnly} locked={readOnly}
                      onChange={(items) => put(items,
                        v.negatives.filter((n) => items.includes(n)))} />
          {/* what this list costs, live as it is edited — the probe's
              minutes on this card and the lift's class pass against the
              card; a warning when the list is more than the card fits */}
          {v.cost && v.prompts.length > 0 && (() => {
            const c = v.cost;
            const n = v.prompts.length;
            const minutes = c.views * n * c.s_per_prompt_view / 60;
            const passGb = c.class_pass_bytes != null
              ? c.gaussians * (c.class_pass_bytes * n + (c.class_pass_fixed ?? 0)) / 2 ** 30
              : null;
            const over = c.classes_fit != null && n > c.classes_fit;
            return (
              <p className={cx("mx-4 mt-1 mb-2", over ? "ui-note-warn" : "ui-help")}
                 data-testid="vocab-cost">
                <span className="font-mono">{n}</span> prompts over{" "}
                <span className="font-mono">{c.views}</span> views: about{" "}
                <span className="font-mono">
                  {minutes < 1 ? `${Math.max(Math.round(minutes * 60), 1)} s`
                               : `${Math.round(minutes)} min`}</span>
                {" "}of probing on this card
                {c.measured ? "" : " (a default rate; the first probe measures it)"}
                {passGb != null && c.card_gb != null && (
                  <>; about <span className="font-mono">{passGb.toFixed(1)} GB</span>
                    {" "}in the lift's class pass, of {c.card_gb} GB
                    {over && `, more than this card fits for this scene (about ${c.classes_fit} classes): shorten the list, or the lift refuses before that pass`}
                  </>)}.
              </p>);
          })()}
          <ListEditor label={`Negative controls (${v.negatives.length})`}
                      items={v.negatives} readOnly={readOnly} locked={readOnly}
                      onChange={(items) => put(v.prompts, items)}
                      addHint="must be a subset of prompts" />
          <Section title="Vision model proposal"
                   right={<Badge tone={v.vlm_available ? "pass" : "warn"}>
                     {v.vlm_available ? "Model ready"
                                      : "Model unavailable"}</Badge>}>
            <div className="flex items-center gap-2 flex-wrap">
              <Btn disabled={readOnly || running || !v.vlm_available
                             || !!waiting}
                   onClick={tryPropose}>
                {v.llm_proposal ? "Propose again" : "Propose with the vision model"}</Btn>
              {/* it reads the rendered views: the render must stand first */}
              {waiting && <span className="text-[0.75rem] text-warn leading-tight">
                {waiting}</span>}
            </div>
            <ModelChoice models={models} current={calib?.vlm?.model_dir}
                         disabled={readOnly || running}
                         onChoose={(dir) => write({ vlm: { model_dir: dir } })} />
            {/* a dead primary must say why */}
            {!v.vlm_available && (
              <div className="mt-2 ui-help whitespace-pre-wrap">
                {v.vlm_reason.trim()}</div>
            )}
            {v.llm_proposal && (
              <div className="mt-2 text-[0.75rem] leading-4">
                <ProposalRow
                  title="Proposed objects"
                  terms={v.llm_proposal.objects}
                  counts={v.llm_proposal.view_counts}
                  adopted={v.prompts}
                  allLabel="add all to prompts"
                  disabled={readOnly || running}
                  onAdd={addPrompts} onRemove={removePrompts} />
                {!!v.llm_proposal.distractors?.length && (
                  <ProposalRow
                    title="Proposed negative controls"
                    terms={v.llm_proposal.distractors}
                    adopted={v.negatives}
                    allLabel="add all as negatives"
                    disabled={readOnly || running}
                    onAdd={addNegatives} onRemove={removeNegatives} />
                )}
                <div className="text-t3 mt-1">
                  click a term to adopt it; the proposal never applies
                  itself.</div>
              </div>
            )}
          </Section>
        </>
      )}

      <Section title="Probe" cue={cue === "probe"}
               right={probe.data?.distribution_png ? (
                 <button className="text-[0.75rem] text-t3 hover:text-t1
                                    underline underline-offset-2"
                         onClick={() =>
                           setShowDistribution(!showDistribution)}>
                   {showDistribution ? "Show table" : "Show distribution"}
                 </button>) : undefined}>
        {!probe.data && (
          <div className="ui-help">no probe yet; run one on
            the current lists.</div>
        )}
        {probe.data && showDistribution && (
          <img src={runUrl(state.scene,
            "/artifacts/stage2/score_distribution.png")}
               alt="probe score distribution"
               className="w-full border border-line rounded" />
        )}
        {probe.data && !showDistribution && (
          <table className="w-full ui-mono text-[0.75rem]">
            <thead>
              <tr className="text-t3 text-left font-sans text-[0.6875rem]">
                <th className="py-1 font-medium">concept</th>
                <th className="text-right font-medium">detections</th>
                <th className="text-right font-medium">max</th>
                <th className="text-right font-medium">mean</th>
              </tr>
            </thead>
            <tbody>
              {probe.data.concepts.map((c) => (
                <tr key={c.concept} className="border-b border-line">
                  <td className={cx("py-0.5",
                                    c.negative ? "text-warn" : "text-t1")}>
                    {c.negative && "[NEG] "}{c.concept}</td>
                  <td className="text-right text-t2">{c.n_det}</td>
                  <td className={cx("text-right",
                                    c.max >= (probe.data!.thresholds
                                      .presence ?? 0.45)
                                      ? "text-t1 font-medium" : "text-t3")}>
                    {c.max.toFixed(3)}</td>
                  <td className="text-right text-t3">{c.mean.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {nc && (
          <div className={cx("mt-2", violated ? "ui-note-fail"
                                              : "ui-note-pass")}>
            <div className={cx("font-mono text-[0.78rem] font-medium",
                               violated ? "text-fail" : "text-pass")}>
              negative controls scored {nc.value.toFixed(3)}
              {violated ? ", above the " : ", within the "}{nc.limit} limit
              {violated ? ": look" : ""}
            </div>
            {violated && (
              <div className="mt-2 text-[0.75rem] text-t1 leading-4">
                outlier {nc.outlier_concept}
                {nc.outlier_frame !== null &&
                  <>, frame {nc.outlier_frame}</>}.
                Look at the outlier in its overlay before confirming;
                the image shows what the score cannot.
                {nc.outlier_frame !== null && (
                  <img src={runUrl(state.scene,
                    `/artifacts/stage2/overlays/frame_${
                      String(nc.outlier_frame).padStart(4, "0")}.png?v=${
                      state.dataEpoch}`)}
                       alt={`overlay of frame ${nc.outlier_frame}`}
                       className="w-full border border-line rounded mt-2" />
                )}
                <div className="mt-2 ui-help">
                  Not blocking; recorded on this approval either way.</div>
                <Details>
                  Negative controls never reach the output; lift drops them
                  before anything is labelled. A violation means the control
                  is not absent from the scene, or the threshold is low for
                  this capture. Crop-check the overlay above if the labels
                  come out wrong.
                </Details>
              </div>
            )}
          </div>
        )}
      </Section>

      <PanelFooter>
        <Btn disabled={readOnly || running || !v?.prompts.length || !!waiting}
             variant={cue === "probe" ? "primary" : undefined}
             title={waiting ?? undefined}
             onClick={runProbe} kbd="R">Run probe</Btn>
        <JobNote show={running} />
        <div className="flex-1" />
        {waiting && <span className="text-[0.75rem] text-warn text-right leading-tight">
          {waiting}</span>}
        {violated && !readOnly && (
          <span className="text-[0.75rem] text-warn text-right leading-tight">
            ceiling VIOLATED: recorded, not blocking</span>
        )}
        <Btn variant="ghost" disabled={readOnly || running}
             title="clear the lists and the proposal; start this step over"
             onClick={askReset}>Reset vocabulary</Btn>
        <Btn variant="primary" kbd="⏎"
             disabled={readOnly || running || !v?.prompts.length || !!waiting
                       || !v?.probed_current}
             title={waiting ?? (v && !v.probed_current
                      ? "run the probe on these exact lists first; the scores are what you confirm"
                      : undefined)}
             onClick={confirm}>Confirm vocabulary</Btn>
      </PanelFooter>

      {resetAsk && (
        <Confirm title="Reset the vocabulary step" destructive
                 confirmLabel="Clear and start over"
                 onCancel={() => setResetAsk(null)} onConfirm={doReset}
                 body={
                   <>Clears <b>{resetAsk.prompts} prompt
                   {resetAsk.prompts === 1 ? "" : "s"}</b> and{" "}
                   <b>{resetAsk.negatives} negative
                   {resetAsk.negatives === 1 ? "" : "s"}</b>
                   {resetAsk.proposal && <> and the vision model's proposal</>},
                   and re-opens this gate and every gate after it. The
                   probe's results stay on disk until you clear the
                   detections on the Objects panel.</>} />
      )}
      {proposeAsk && (
        <Confirm title="Propose again?"
                 confirmLabel="Propose again"
                 onCancel={() => setProposeAsk(false)}
                 onConfirm={() => { setProposeAsk(false); proposeLlm(); }}
                 body={<>A proposal is already on screen; proposing again
                   replaces it. The terms you adopted stay in your lists;
                   to start the step over, use <b>Reset vocabulary</b>
                   instead.</>} />
      )}
    </div>
  );
}

/** One proposal list as click-to-adopt chips + an add-all shortcut.
 *  Already-adopted terms render inert with a check — clicking a term or
 *  "add all" is the explicit operator act the adopt contract requires. */
/** Terms shown before the long tail folds: the model's list is
 *  ordered by how many views proposed each term, so the head is the
 *  agreed core and the tail the one-view variants. */
const PROPOSAL_HEAD = 15;

function ProposalRow({ title, terms, counts, adopted, allLabel, disabled,
                       onAdd, onRemove }: {
  title: string; terms: string[]; adopted: string[]; allLabel: string;
  /** views that proposed each term, when the proposal carries them */
  counts?: Record<string, number>;
  disabled: boolean; onAdd: (terms: string[]) => void;
  /** the inverse of add-all: drop the proposal's adopted terms */
  onRemove: (terms: string[]) => void;
}) {
  const remaining = terms.filter((t) => !adopted.includes(t));
  const added = terms.filter((t) => adopted.includes(t));
  const [all, setAll] = useState(false);
  const folded = terms.length > PROPOSAL_HEAD && !all;
  const shown = folded ? terms.slice(0, PROPOSAL_HEAD) : terms;
  return (
    <div className="mb-1.5">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[0.75rem] text-t2 font-medium flex-1">
          {title}{counts && <span className="text-t4 font-normal"> · by views agreeing</span>}</span>
        {!disabled && remaining.length > 0 && (
          <button className="text-[0.75rem] text-t2 underline
                             underline-offset-2 hover:text-t1"
                  onClick={() => onAdd(remaining)}>
            {allLabel} ({remaining.length})</button>
        )}
        {!disabled && added.length > 0 && (
          <button className="text-[0.75rem] text-t2 underline
                             underline-offset-2 hover:text-fail"
                  title="drop every term of this proposal from your lists"
                  onClick={() => onRemove(added)}>
            remove added ({added.length})</button>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {shown.map((t) => adopted.includes(t) ? (
          <span key={t}
                className="font-mono text-[0.75rem] px-1.5 h-6 inline-flex
                           items-center rounded bg-hover text-t3">
            ✓ {t}</span>
        ) : (
          <button key={t} disabled={disabled}
                  title={counts?.[t] != null
                    ? `proposed by ${counts[t]} view${counts[t] === 1 ? "" : "s"}`
                    : undefined}
                  className="font-mono text-[0.75rem] px-1.5 h-6 border
                             border-line2 rounded text-t1 bg-inset
                             hover:bg-hover hover:border-t3
                             disabled:opacity-45"
                  onClick={() => onAdd([t])}>
            + {t}{counts?.[t] != null && counts[t] > 1 && (
              <span className="text-t4"> ·{counts[t]}</span>)}</button>
        ))}
        {terms.length > PROPOSAL_HEAD && (
          <button className="text-[0.75rem] text-t3 underline
                             underline-offset-2 hover:text-t1 px-1 h-6"
                  onClick={() => setAll(!all)}>
            {folded ? `show all (${terms.length})`
                    : `show the first ${PROPOSAL_HEAD}`}</button>
        )}
      </div>
    </div>
  );
}

function ListEditor({ label, items, onChange, readOnly, locked, addHint, cue }: {
  label: string; items: string[]; onChange: (items: string[]) => void;
  readOnly: boolean; locked?: boolean; addHint?: string;
  /** this list owns the gate's current step */
  cue?: boolean;
}) {
  const [draft, setDraft] = useState("");
  const add = () => {
    const parts = draft.split(",").map((s) => s.trim()).filter(Boolean);
    if (parts.length) onChange([...items, ...parts.filter(
      (p) => !items.includes(p))]);
    setDraft("");
  };
  return (
    <Section title={label} locked={locked} cue={cue}
             right={!readOnly && items.length > 0 && (
               <button className="text-[0.75rem] text-t3 underline
                                  underline-offset-2 hover:text-fail"
                       title="remove every term"
                       onClick={() => onChange([])}>clear all</button>)}>
      <div className="flex flex-wrap gap-1 mb-2">
        {items.map((it) => (
          <span key={it}
                className="inline-flex items-center gap-1 font-mono
                           text-[0.75rem] pl-1.5 pr-1 h-6 rounded bg-hover
                           text-t1">
            {it}
            {!readOnly && (
              <button className="text-t3 hover:text-fail px-0.5 rounded-sm"
                      aria-label={`remove ${it}`}
                      onClick={() => onChange(items.filter(
                        (x) => x !== it))}>✕</button>
            )}
          </span>
        ))}
        {items.length === 0 && (
          <span className="text-[0.75rem] text-t4">(none)</span>
        )}
      </div>
      {!readOnly && (
        <input
          className={cx(INPUT, "w-full")}
          aria-label={`add ${label}`}
          placeholder={addHint ?? "add terms (comma-separated), Enter"}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add()} />
      )}
    </Section>
  );
}
