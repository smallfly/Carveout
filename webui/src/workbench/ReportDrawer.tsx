// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Report drawer (1j): native tables from report.json (gate timeline,
// verification, funnel + flags); the
// banked runs that predate the sidecar render their run_report.md.

import { useMemo } from "react";
import { runUrl } from "../api";
import { useRun, useRunData } from "../store";
import type { Gate, InstanceRow, ReportResp } from "../types";
import { Badge, Section, cx } from "../ui";
import { GATE_LABEL } from "../labels";
import { shownName, useRelabel, useVerifyMark } from "./relabel";

/** The verification counts, in words (the report keeps the keys). */
const VERIFICATION_WORDS: Record<string, string> = {
  confirmed: "confirmed", relabeled: "relabelled", rejected: "rejected",
  unverified: "unverified", held: "held",
  not_judged: "not judged (not marked)", judged: "judged this run",
  resumed: "taken from the checkpoint", carried: "carried from the previous run",
  relabels_unsupported: "label kept (too few views agreed)",
  relabels_outside_vocab: "label kept (name not in the vocabulary)",
  relabels_too_large: "label kept (too large to check)",
};

export default function ReportDrawer({ onFlyTo }:
    { onFlyTo: (idx: number) => void }) {
  const { state } = useRun();
  const rep = useRunData<ReportResp>("/report");
  const inst = useRunData<{ objects: InstanceRow[] }>("/instances");
  // The operator's relabel: the pencil on each row (relabel.tsx).
  const relabel = useRelabel(() => inst.reload());
  // The mark for verification: the same control on each row.
  const mark = useVerifyMark(() => inst.reload());

  const byLabel = useMemo(() => {
    const m = new Map<string, number>();
    inst.data?.objects.forEach((o) => {
      if (!m.has(o.label)) m.set(o.label, o.idx);
    });
    return m;
  }, [inst.data]);

  const r = rep.data?.report;
  return (
    <div>
      <Section title="Run report"
               right={rep.data?.markdown ? (
                 <a className="text-[0.75rem] text-t3 underline
                               underline-offset-2 hover:text-t1"
                    href={runUrl(state.scene, "/artifacts/run_report.md")}
                    target="_blank" rel="noreferrer">raw .md</a>
               ) : undefined}>
        {!rep.data?.markdown && !r && (
          <div className="ui-help">no report yet; it is
            written when the verify gate is answered.</div>
        )}
        {r && (
          <div className="font-mono text-[0.75rem] text-t3">
            code {r.code} · generated {r.generated}</div>
        )}
        {r?.pipeline && (
          <div className="font-mono text-[0.75rem] text-t3 mt-0.5">
            instancing {r.pipeline.instancing ?? "unknown"} ·{" "}
            {!r.pipeline.scale ? "scale unknown"
              : r.pipeline.scale.source === "measured"
                ? `1 unit ≈ ${Number(r.pipeline.scale.scale_m_per_unit.toPrecision(4))} m (measured${
                    r.pipeline.scale.measured?.label ? `: ${r.pipeline.scale.measured.label}` : ""})`
                : `scale ${r.pipeline.scale.scale_m_per_unit} m / unit (recorded)`}
          </div>
        )}
      </Section>

      {r && (
        <>
          <Section title="Gates">
            <table className="w-full ui-mono text-[0.75rem]">
              <tbody>
                {Object.entries(r.gates).map(([g, rec]) => (
                  <tr key={g} className="border-b border-line">
                    <td className="py-1 text-t1">{GATE_LABEL[g as Gate] ?? g}</td>
                    <td className="py-1 text-right text-t3">
                      {rec ? `approved ${rec.approved_at}` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>

          <Section title={`Instances · ${r.instances.count} exported`}>
            {r.instances.verification ? (
              <>
                <div className="flex gap-1 flex-wrap">
                  {Object.entries(r.instances.verification)
                    .filter(([k, v]) => typeof v === "number" && k !== "model"
                                        && k !== "verify_schema")
                    .map(([k, v]) => (
                      <Badge key={k}
                             tone={k === "confirmed" ? "pass"
                               : k === "rejected" || k === "unverified"
                                 ? (v ? "fail" : "muted")
                               : k === "held" ? (v ? "warn" : "muted")
                               : "muted"}>
                        {VERIFICATION_WORDS[k] ?? k} {v as number}</Badge>
                    ))}
                </div>
                {!!r.instances.verification.held && (
                  <div className="ui-help mt-1">
                    {r.instances.verification.held as number} instance(s)
                    ship HELD: verification could not confirm them
                    (unresolved association evidence; listed below).
                  </div>
                )}
              </>
            ) : (
              <>
                <Badge tone="warn">verification NOT RUN, labels
                  unverified</Badge>
                {!!r.instances.held_association && (
                  <div className="ui-help mt-1">
                    {r.instances.held_association} instance(s) are held;
                    each may be more than one object (marked below);
                    verification was not run, so their labels stand
                    unjudged.
                  </div>
                )}
              </>
            )}
          </Section>

          <Section title="Dropped for size (far larger than the class's usual mask)">
            {r.factor_kills.length === 0
              ? <Badge tone="pass">none</Badge>
              : r.factor_kills.map((k: any, i) => (
                  <div key={i} className="font-mono text-[0.75rem] text-fail">
                    {k.concept} · frame {k.frame} · {k.ratio}× the usual
                    size</div>
                ))}
          </Section>

          <Section title="From detections to objects, per class">
            <table className="w-full ui-mono text-[0.75rem]">
              <thead>
                <tr className="text-t3 text-left font-sans text-[0.6875rem]">
                  <th className="py-1">class</th>
                  <th className="text-right">detected</th>
                  <th className="text-right">above threshold</th>
                  <th className="text-right">distinct</th>
                  <th className="text-right">Gaussians</th>
                  <th className="text-right">groups</th>
                  <th className="text-right">exported</th>
                </tr>
              </thead>
              <tbody>
                {r.funnel.rows.map((row) => {
                  const flagged = r.funnel.flags.some(
                    (f) => f.startsWith(row.cls + ":"));
                  const idx = byLabel.get(row.cls);
                  return (
                    <tr key={row.cls}
                        className={cx("border-b border-line",
                                      idx !== undefined &&
                                        "cursor-pointer hover:bg-hover")}
                        onClick={() => idx !== undefined && onFlyTo(idx)}>
                      <td className={cx("py-0.5",
                                        flagged ? "text-warn" : "text-t1")}>
                        {row.cls}</td>
                      <td className="text-right text-t3">{row.logged}</td>
                      <td className="text-right text-t3">{row.above}</td>
                      <td className="text-right text-t3">{row.dedup}</td>
                      <td className="text-right text-t3">{row.g_in}</td>
                      <td className="text-right text-t3">{row.comps}</td>
                      <td className="text-right text-t1">{row.exported}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {r.funnel.flags.map((f) => (
              <div key={f} className="mt-1 font-mono text-[0.75rem] text-warn">
                ⚑ {f}</div>
            ))}
          </Section>
        </>
      )}

      {!r && rep.data?.markdown && (
        <Section title="Report (text)">
          <pre className="font-mono text-[0.75rem] text-t2 whitespace-pre-wrap
                          leading-4 overflow-x-auto">
            {rep.data.markdown}</pre>
        </Section>
      )}

      {inst.data && (
        <Section title={`Instances (${inst.data.objects.length})`}>
          {/* the whole list, in the panel's own scroll — no scroll of its own */}
          <div>
            {inst.data.objects.map((o) => {
              const shown = shownName(o);
              const renamed = shown !== o.label;
              return relabel.isEditing(o) ? (
                <div key={o.idx} className="flex items-center gap-2 py-0.5 px-1">
                  <span className="text-t4 w-6 ui-mono text-[0.75rem]">
                    {String(o.idx).padStart(2, "0")}</span>
                  {relabel.input(o, "flex-1")}
                </div>
              ) : (
                <div key={o.idx}
                     className="w-full flex items-center gap-2 py-0.5 px-1
                                ui-mono text-[0.75rem] hover:bg-hover rounded-sm">
                  <button className="flex items-center gap-2 flex-1 min-w-0 text-left"
                          onClick={() => onFlyTo(o.idx)}>
                    <span className="text-t4 w-6">
                      {String(o.idx).padStart(2, "0")}</span>
                    <span className="text-t1 flex-1 truncate"
                          title={renamed ? `detected: ${o.label}` : undefined}>
                      {shown}
                      {renamed && <span className="text-t4"> · was {o.label}</span>}
                    </span>
                  </button>
                  {relabel.takeButton(o, "text-[0.6875rem]")}
                  {relabel.pencil(o)}
                  {mark.button(o, "text-[0.6875rem]")}
                  {o.scale === "exemplar" && <Badge tone="accent">ex</Badge>}
                  {o.verdict === "UNVERIFIED" &&
                    <Badge tone="warn">unverified</Badge>}
                  {o.verdict === "NOT_JUDGED" &&
                    <Badge tone="warn">not judged</Badge>}
                  {(o.hold_reasons?.length > 0 || o.verdict === "HELD") &&
                    <Badge tone="warn">held</Badge>}
                  <span className="text-t3">{o.conf.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </Section>
      )}
    </div>
  );
}
