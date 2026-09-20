// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The Objects point: the unattended stages (1g) + the verify consent
// gate (1i). The pipeline job is
// detect -> lift -> export under the shared force rules; a factor-cap
// kill halts it server-side (stop-and-look takeover). Verify consent is
// an explicit act; the key never leaves the server — status only.

import { useState } from "react";
import { gateState, reviewGates } from "../../journal";
import { api, runUrl } from "../../api";
import { useCalibration, useJobRunning, useRefusal, useRun, useRunData,
         useSceneAct, sceneFactor, useVlmModels } from "../../store";
import { lengthLabel } from "../../units";
import type { VolumeResp } from "../../types";
import { Badge, Btn, Confirm, Details, LabeledNum, Section, Segmented,
         JobNote, ModelChoice, Steps, Switch } from "../../ui";
import { GATE_LABEL } from "../../labels";
import { currentStep, objectsSteps } from "../../steps";

interface Preflight {
  instances: number; est_calls: number; model: string;
  vlm_available: boolean; vlm_reason: string;
  /** the kept verification when still fresh: Run verification
   *  replays it at once instead of judging again */
  already_verified: { model: string | null; elapsed_s: number | null } | null;
  /** objects marked for verification: only these are judged */
  marked: number;
  /** a previous verification of this export exists: the unmarked carry
   *  their verdicts from it */
  previous_verdicts: boolean;
  /** verdicts of a run that stopped: the next press continues */
  checkpoint: { judged: number } | null;
}

export default function PipelinePanel({ readOnly }:
    { readOnly: boolean }) {
  const { state } = useRun();
  const j = state.journal;
  const running = useJobRunning();
  const preflight = useRunData<Preflight>("/verify/preflight");
  const refuse = useRefusal();
  const [clearAsk, setClearAsk] = useState<{ stages: string[];
                                             objects: number } | null>(null);

  const { act, startStage } = useSceneAct();
  const askClear = () =>
    api.get<{ stages: string[]; objects: number }>(
      runUrl(state.scene, "/detections/reset")).then(setClearAsk).catch(refuse);
  const doClear = () => {
    setClearAsk(null);
    act("/detections/reset");
  };

  // the review gates still standing between the operator and the run
  const missing = j ? reviewGates.filter(
    (g) => gateState(j, g) !== "approved") : reviewGates;
  const gatesReady = !!j && missing.length === 0;

  const startPipeline = () => startStage("pipeline");

  const consent = (yes: boolean) =>
    act("/verify/consent", { consent: yes });

  const vRec = j?.gates.verify_consent;
  // an answer recorded and still fresh stands: the server refuses the
  // other one until the gate is reopened, and the button says so first
  const decided: boolean | null =
    vRec && vRec.state === "approved" ? !!vRec.run : null;
  const { calib, write } = useCalibration();
  const models = useVlmModels();
  // The scene's factor for the voxel's label: recorded from the profile,
  // else the estimate the volume read carries (no metres spoken then).
  const vol = useRunData<VolumeResp>("/volume");
  const recorded = sceneFactor(calib);
  const steps = objectsSteps({
    gatesReady, missing: missing.map((g) => GATE_LABEL[g]),
    hasExport: !!preflight.data, decided: vRec?.state === "approved" });
  const cue = currentStep(steps);

  return (
    <div>
      <Section title="Objects" band>
        <p className="ui-help m-0">
          Segment every object out of the scene, then decide whether their
          labels are verified.
        </p>
        <Steps steps={steps} />
      </Section>
      <Section title="Extract the objects · detect → lift → export"
               cue={cue === "extract"}>
        <p className="ui-help m-0">
          Runs unattended; progress streams in the stage strip. A detection
          dropped for its size stops the run until you have looked at it.
        </p>
        <Details className="mb-2">
          Progress streams below and in the stage strip; closing the tab
          changes nothing; re-open to re-attach. A gate you redo makes the
          stages after it run again. If a detection is dropped for being far
          larger than its class's usual size, the run stops and shows you
          the frame until you continue.
        </Details>
        {calib?.lift && (
          <div className="border-b border-line pb-2 mb-2">
            <LabeledNum {...lengthLabel("instance voxel: lift granularity",
                                        calib.lift.instance_voxel,
                                        recorded ?? vol.data?.scale?.scale ?? null,
                                        !!calib.scene.measured)}
                        value={calib.lift.instance_voxel}
                        disabled={readOnly || running}
                        onCommit={(v) =>
                          write({ lift: { instance_voxel: v } })} />
            <p className="ui-help m-0">
              smaller = finer instance separation; a change re-opens verify
              consent and forces lift + export next run.
            </p>
            <div className="flex items-center gap-2 mt-2 flex-wrap">
              <span className="ui-label flex-1">instancing</span>
              <Segmented value={calib.lift.instancing === "connectivity"
                                  ? "Connectivity" : "Tracks"}
                         options={["Tracks", "Connectivity"]}
                         label="instancing" disabled={readOnly || running}
                         onChange={(v) => write({ lift: { instancing:
                           v === "Connectivity" ? "connectivity" : "tracks" } })} />
            </div>
            <p className="ui-help m-0 mt-1">
              <b>Tracks</b> (the default) follows each detection across
              views; <b>Connectivity</b> groups by 3D adjacency: try it
              when Tracks under-counts objects that touch or repeat.
            </p>
            <Details>
              Run once each and compare the object count in the report,
              which names the method that produced the export. A change
              re-opens the verify gate and runs the lift and the export
              again.
            </Details>
          </div>
        )}
        {!gatesReady && (
          <div className="ui-note-warn mb-2">
            Runs once the four review gates are approved. Still to approve:{" "}
            <b>{missing.map((g) => GATE_LABEL[g]).join(", ")}</b>.
          </div>
        )}
        <div className="flex items-center gap-2 flex-wrap">
          <Btn variant="primary"
               disabled={readOnly || running || !gatesReady}
               onClick={startPipeline}>
            Run detect → lift → export</Btn>
          <Btn variant="ghost" disabled={readOnly || running}
               title="delete the probe, the lift, the export and the verification; the render stays"
               onClick={askClear}>
            Clear detections</Btn>
          <JobNote show={running} />
        </div>
      </Section>

      <Section title="Verify the labels: run or skip"
               locked={readOnly} cue={cue === "verify"}>
        {vRec && (
          <div className="mb-2">
            <Badge tone={vRec.state === "approved" ? "pass" : "warn"}>
              Recorded {vRec.run ? "RAN" : "SKIPPED"} · {vRec.approved_at}
              {vRec.state === "stale" && " · STALE"}
            </Badge>
          </div>
        )}
        {preflight.data ? (
          <>
            <div className="text-[0.8125rem] text-t1 mb-1 ui-num">
              {preflight.data.marked > 0 && (
                <><span className="font-mono">{preflight.data.marked}</span>
                  {" "}marked of{" "}</>)}
              <span className="font-mono">{preflight.data.instances}</span>
              {" "}instances ≈{" "}
              <span className="font-mono">{preflight.data.est_calls}</span>
              {" "}judgements →{" "}
              <span className="font-mono">{preflight.data.model}</span>, locally
            </div>
            {/* the marked subset — said before the press, with what
                becomes of the rest; one control takes every mark back */}
            {preflight.data.marked > 0 && (
              <div className="mb-2 flex items-center gap-2 flex-wrap">
                <span className="ui-help mb-0 flex-1 min-w-0">
                  Only the marked objects are judged (mark one on its card
                  in the 3D view, or on its row in the report). The other{" "}
                  {preflight.data.instances - preflight.data.marked}{" "}
                  {preflight.data.previous_verdicts
                    ? "keep their verdicts from the previous verification of this export."
                    : "ship not judged; their labels stay as detected."}
                </span>
                <Btn variant="ghost" disabled={readOnly || running}
                     title="takes every mark back: Run verification judges every object again"
                     onClick={async () => {
                       if (await act("/verify_marks", {}, { method: "del" }))
                         preflight.reload();
                     }}>
                  Clear marks (verify all)</Btn>
              </div>)}
            {/* a run that stopped left its verdicts; the next press
                continues from them */}
            {preflight.data.checkpoint && (
              <p className="ui-help mt-0 mb-2" data-testid="verify-resume">
                <span className="font-mono">{preflight.data.checkpoint.judged}</span>
                {" "}judged before the run stopped; Run verification continues
                from there; those are not judged again.
              </p>)}
            <div className="mb-2 flex items-center gap-2 flex-wrap">
              <Badge tone={preflight.data.vlm_available ? "pass" : "warn"}>
                {preflight.data.vlm_available ? "Model ready"
                                              : "Model unavailable"}
              </Badge>
            </div>
            {!preflight.data.vlm_available && (
              <div className="ui-help whitespace-pre-wrap mb-2">
                {preflight.data.vlm_reason.trim()}</div>
            )}
            <ModelChoice models={models} current={calib?.vlm?.model_dir}
                         disabled={readOnly || running}
                         onChoose={(dir) => write({ vlm: { model_dir: dir } })} />
            {/* an option, off by default, recorded per scene; changing
                it re-opens verification as a real run (the parameter is in
                the stage's manifest). Verify first as it is, then again
                with it, and compare — verdicts.previous.csv keeps the
                earlier set. */}
            <div className="flex items-center gap-2 mt-2">
              <span className="ui-label flex-1">
                let a more specific name stand
                <span className="block text-t3">
                  a name outside the vocabulary is applied when the verifier
                  says it is a kind of the detected label (an onion for
                  "vegetables"); a more general word never is
                </span>
              </span>
              <Switch checked={!!calib?.vlm?.relabel_specific_ok}
                      disabled={readOnly || running || !calib}
                      label="let a more specific name stand"
                      onChange={(v) => write({ vlm: { relabel_specific_ok: v } })} />
            </div>
            {/* the second option, same shape — the object's own render
                for small objects, never for large ones */}
            <div className="flex items-center gap-2 mt-2">
              <span className="ui-label flex-1">
                show small objects on their own too
                <span className="block text-t3">
                  under {calib?.vlm?.verify_isolated_max_gaussians ?? 300} Gaussians,
                  the object's own 3D reconstruction rendered alone joins its
                  views as one more image, where a view cannot show a small
                  thing; large objects are never shown this way
                </span>
              </span>
              <Switch checked={!!calib?.vlm?.verify_isolated}
                      disabled={readOnly || running || !calib}
                      label="show small objects on their own too"
                      onChange={(v) => write({ vlm: { verify_isolated: v } })} />
            </div>
            {preflight.data.already_verified && (
              <p className="ui-help mt-0 mb-2" data-testid="verify-replay">
                Already verified on this export
                {preflight.data.already_verified.model
                  ? <> with <span className="font-mono">
                      {preflight.data.already_verified.model}</span></>
                  : null}
                ; nothing changed since, so Run verification replays the
                kept result at once, no GPU time. It judges again after a new
                export (Run pipeline after a change) or under another model.
              </p>
            )}
            <div className="flex gap-2 flex-wrap mt-3">
              <Btn variant="primary" kbd="Y"
                   disabled={readOnly || running || !gatesReady
                             || !preflight.data.vlm_available
                             || decided === false}
                   title={!gatesReady ? "the review gates are not all approved"
                          : preflight.data.already_verified
                            ? "replays the kept result; nothing changed since"
                            : undefined}
                   onClick={() => consent(true)}>
                {preflight.data.already_verified
                  ? "Run verification (replays the kept result)"
                  : "Run verification"}</Btn>
              <Btn variant="ghost" kbd="N"
                   disabled={readOnly || running || !gatesReady
                             || decided === true}
                   title={gatesReady ? undefined : "the review gates are not all approved"}
                   onClick={() => consent(false)}>
                Skip (labels ship unverified)</Btn>
            </div>
            {decided !== null && (
              <p className="ui-help mt-2 mb-0" data-testid="verify-decided">
                Recorded {decided ? "RAN" : "SKIPPED"}, and nothing changed
                since: {decided ? "Run verification replays the kept result; "
                                : ""}
                to answer the other way, reopen the verify gate first
                (Reopen gate… on the rail).
              </p>
            )}
            <p className="ui-help mt-2 mb-0">
              Either choice is recorded and writes the report. A label the
              verifier could not judge is kept and marked unverified.
            </p>
          </>
        ) : (
          <div className="ui-help">
            no objects yet; the decision opens when the extraction
            finishes.</div>
        )}
      </Section>
      {clearAsk && (
        <Confirm title="Clear detections?" destructive
                 confirmLabel="Clear"
                 body={<>Deletes the probe, the lift, the export
                   ({clearAsk.objects} object{clearAsk.objects === 1 ? "" : "s"})
                   and the verification, and re-opens the verify gate. The render and the gates before
                   it stay. The labels leave the canvas with the export.</>}
                 onConfirm={doClear} onCancel={() => setClearAsk(null)} />
      )}
    </div>
  );
}
