// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Scene Library: thumbnail + text rows (deliberately both),
// per-gate journal chips, state-derived primary action, lock chip
// for held scenes.

import { useEffect, useState } from "react";
import { api } from "../api";
import { GATES, type ConventionScene, type DeletionPlan, type FsListing,
         type Journal, type SceneEntry,
         type SettingsInfo, type FitEstimate } from "../types";
import { AppearanceMenu, Badge, Btn, DialogTitle, FieldLabel, INPUT, Modal,
         Segmented, Spinner, cx } from "../ui";
import { GATE_LABEL, GATE_SHORT, jobLabel } from "../labels";

/** The five gates as one compact strip: the letters name the gate, the
 *  tint and the mark carry its state (✓ approved, ! stale, filled = the
 *  active one, faint = not reached); the full name rides the tooltip
 *  and the accessible name. */
function GateChips({ journal }: { journal: Journal }) {
  return (
    <div className="inline-flex rounded border border-line overflow-hidden
                    ui-num"
         role="list" aria-label="gates">
      {GATES.map((g) => {
        const rec = journal.gates[g];
        const active = journal.active_gate === g;
        const state = rec?.state === "approved" ? "approved"
          : rec?.state === "stale" ? "stale" : active ? "active" : "idle";
        const word = state === "approved" ? "approved"
          : state === "stale" ? "stale" : state === "active" ? "active"
          : "not reached";
        return (
          <span key={g} role="listitem"
                aria-label={`${GATE_LABEL[g]} ${word}`}
                className={cx(
                  "font-mono text-[0.6875rem] px-1.5 h-5 inline-flex",
                  "items-center gap-0.5 border-r border-line last:border-r-0",
                  state === "approved" && "text-pass bg-pass/10",
                  state === "stale" && "text-warn bg-warn/10",
                  state === "active" && "bg-sel text-t1 font-medium",
                  state === "idle" && "text-t4 bg-inset")}>
            <span aria-hidden="true">{GATE_SHORT[g]}</span>
            {state === "approved" && <span aria-hidden="true">✓</span>}
            {state === "stale" && <span aria-hidden="true">!</span>}
          </span>
        );
      })}
    </div>
  );
}

function primaryAction(s: SceneEntry): { label: string; accent: boolean } {
  const j = s.journal as Journal;
  if (!j || "error" in (s.journal as object))
    return { label: "Open", accent: false };
  if (s.lock && !s.lock.mine)
    return { label: "View read-only", accent: false };
  if (j.running?.state === "running")
    return { label: `Re-attach · ${jobLabel(j.running.kind)}`, accent: true };
  if (j.active_gate)
    return { label: `Resume · ${GATE_LABEL[j.active_gate]} gate`,
             accent: true };
  return { label: "Open workbench", accent: false };
}

export default function Library({ onOpen }:
    { onOpen: (name: string) => void }) {
  const [scenes, setScenes] = useState<SceneEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dialog, setDialog] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Read-only server facts: the profile directory and the model status.
  const [settings, setSettings] = useState<SettingsInfo | null>(null);

  const load = () =>
    api.get<SceneEntry[]>("/api/scenes").then(setScenes)
      .catch((e) => setError(e.message));
  const loadSettings = () =>
    api.get<SettingsInfo>("/api/settings").then(setSettings)
      .catch((e) => setError(e.message));
  useEffect(() => { load(); loadSettings(); }, []);

  return (
    <div className="min-h-full bg-app">
      <header className="h-11 border-b border-line px-4 flex items-center
                         gap-3 bg-panel">
        <span className="font-semibold text-[0.8125rem] tracking-[0.08em]
                         text-t1">
          CARVEOUT</span>
        <span className="text-t4" aria-hidden="true">/</span>
        <h1 className="text-[0.8125rem] font-medium text-t2 m-0">
          Scene library</h1>
        <div className="flex-1" />
        <AppearanceMenu />
        <Btn variant="outline" onClick={() => setSettingsOpen(true)}>
          Settings
        </Btn>
        <Btn variant="primary" onClick={() => setDialog(true)}>
          + New scene
        </Btn>
      </header>

      {error && (
        <div className="m-4 ui-note-fail font-mono" role="alert">{error}</div>
      )}
      {!scenes && !error && <div className="p-6"><Spinner /></div>}
      {scenes && scenes.length === 0 && (
        <div className="flex flex-col items-center py-24 gap-3 text-t3">
          <div>no scenes yet; add one with New scene</div>
          <Btn variant="primary" onClick={() => setDialog(true)}>
            + New scene</Btn>
        </div>
      )}

      {/* One plain table in a centred rem-sized column (it scales with the
          root bump on 2.5K displays). Fixed layout: the scene column takes
          whatever is left and truncates, so the table can never outgrow the
          page — the only scrollbars are the browser's. */}
      {scenes && scenes.length > 0 && (
        <div className="max-w-[100rem] mx-auto w-full px-6 py-2">
        <table className="w-full border-collapse table-fixed">
          <colgroup>
            <col className="w-[4.5rem]" />
            <col />
            <col className="w-[15rem]" />
            <col className="w-[7.5rem]" />
            <col className="w-[6rem]" />
            <col className="w-[8.5rem]" />
            <col className="w-[17rem]" />
          </colgroup>
          <thead>
            <tr className="text-left text-[0.75rem] font-medium
                           text-t3 border-b border-line">
              <th className="pl-1 pr-2 py-2">
                <span className="sr-only">thumbnail</span></th>
              <th className="px-3 py-2 font-medium">Scene</th>
              <th className="px-3 py-2 font-medium">Gates</th>
              <th className="px-3 py-2 text-right font-medium">Gaussians</th>
              <th className="px-3 py-2 text-right font-medium">Instances</th>
              <th className="px-3 py-2 font-medium">Last report</th>
              <th className="pl-3 pr-1 py-2 text-right">
                <span className="sr-only">actions</span></th>
            </tr>
          </thead>
          <tbody>
            {scenes.map((s) => {
              const act = primaryAction(s);
              const j = s.journal as Journal;
              return (
                <tr key={s.name}
                    className="h-16 border-b border-line
                               hover:bg-hover cursor-pointer"
                    onClick={() => onOpen(s.name)}>
                  <td className="pl-1 pr-2 py-1.5">
                    {s.thumb ? (
                      <img src={`/api/scenes/${s.name}/thumb.png`}
                           alt=""
                           className="w-14 h-14 object-cover rounded
                                      border border-line bg-inset" />
                    ) : (
                      <div className="w-14 h-14 rounded border
                                      border-line bg-inset flex items-center
                                      justify-center text-center
                                      text-[0.6875rem] leading-tight
                                      text-t4">no
                        render</div>
                    )}
                  </td>
                  <td className="px-3 py-1.5 min-w-0">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-[0.875rem] font-semibold text-t1
                                       truncate">{s.name}</span>
                    </div>
                    <div className="text-[0.6875rem] text-t4 font-mono
                                    truncate">{s.scene}</div>
                  </td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      {j && !("error" in (s.journal as object))
                        ? <GateChips journal={j} /> : <Badge>?</Badge>}
                      {s.error && (
                        <span className="text-[0.75rem] text-warn
                                         whitespace-normal leading-tight">
                          {s.error}</span>
                      )}
                      {s.lock && !s.lock.mine && (
                        <Badge tone="warn" className="whitespace-normal">
                          Locked · {s.lock.holder} · since {s.lock.since}
                        </Badge>
                      )}
                    </div>
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono text-t2
                                 ui-num whitespace-nowrap">
                    {s.gaussians?.toLocaleString() ?? "—"}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-t2
                                 ui-num">
                    {s.instances ?? "—"}</td>
                  <td className="px-3 py-1.5 font-mono text-t3
                                 text-[0.75rem] ui-num">
                    {s.last_report ?? "—"}</td>
                  <td className="pl-3 pr-1 py-1.5 text-right">
                    <div className="flex gap-2 justify-end items-center">
                      <Btn variant={act.accent ? "primary" : "outline"}
                           onClick={(e) => { e.stopPropagation();
                                             onOpen(s.name); }}>
                        {act.label}
                      </Btn>
                      {/* Deliberately quiet and last: a destructive act
                          should be reachable, never the thing a hurried
                          click lands on. The dialog does the real gating. */}
                      <Btn variant="ghost"
                           aria-label={`delete ${s.name}`}
                           onClick={(e) => { e.stopPropagation();
                                             setDeleting(s.name); }}>
                        Delete
                      </Btn>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        </div>
      )}

      {deleting && <DeleteSceneDialog name={deleting}
                                      onClose={() => setDeleting(null)}
                                      onDeleted={() => { setDeleting(null);
                                                         load(); }} />}
      {dialog && <NewSceneDialog onClose={() => setDialog(false)}
                                 onCreated={(n) => { setDialog(false);
                                                     load(); onOpen(n); }} />}
      {settingsOpen && settings && (
        <SettingsDialog settings={settings}
                        onClose={() => setSettingsOpen(false)} />
      )}
    </div>
  );
}

function SettingsDialog({ settings, onClose }:
    { settings: SettingsInfo; onClose: () => void }) {
  // Read-only facts about this server: where profiles live (fixed for
  // the process — `carveout web --profiles-dir` for a spare server) and
  // whether the local model is on disk. Nothing here is edited.
  return (
    <Modal onClose={onClose} width={560} title="Settings">
      <DialogTitle>Settings</DialogTitle>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1
                      text-[0.75rem] items-center">
        <span className="ui-label">profiles</span>
        <span className="font-mono text-t2 break-all">
          {settings.profiles_dir}</span>
        <span className="ui-label">vision model</span>
        <span className="flex items-center gap-2 flex-wrap">
          <span className="text-t1">{settings.vlm_model}</span>
          {settings.vlm_size_gib > 0 && (
            <span className="text-t3">{settings.vlm_size_gib} GB on disk</span>)}
          <Badge tone={settings.vlm_available ? "pass" : "warn"}>
            {settings.vlm_available ? "Ready" : "Unavailable"}</Badge>
        </span>
        <span />
        <span className="font-mono text-t3 break-all">
          {settings.vlm_model_dir}</span>
      </div>
      {!settings.vlm_available && (
        <pre className="mt-2 ui-help whitespace-pre-wrap font-mono
                        text-[0.6875rem]">{settings.vlm_reason}</pre>
      )}
      <p className="ui-help mt-2">
        The vision model reads the measured length at the volume gate,
        proposes the vocabulary and verifies the labels, on this machine.
        Each scene chooses its own when it is created (proposed from the
        card and the scene) and can change it on the Vocabulary and Objects
        panels; this one is the server's default for scenes that did not
        choose{settings.vlm_source === "--vlm-dir" ? " (--vlm-dir)" : ""}.
        Installed: {settings.vlm_models.map((m) => m.name).join(", ")}.
        Every model loads compressed; a smaller one leaves room on the card
        beside a large scene:
      </p>
      <pre className="ui-help whitespace-pre-wrap font-mono text-[0.6875rem] m-0">
{`hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b/
carveout web --vlm-dir models/qwen3-vl-8b`}
      </pre>
      <p className="ui-help mt-1">
        Every verification records the model that produced it, and a scene
        verified with one model re-opens its verify gate under another.
        The choice is never made for you.
      </p>
      <div className="flex justify-end gap-2 mt-4">
        <Btn variant="ghost" onClick={onClose}>Close</Btn>
      </div>
    </Modal>
  );
}

interface SplatInfo {
  path: string; bytes: number; gaussians: number | null;
  card_gb?: number; classes_fit?: number; per_class_mb?: number;
  scene_gb?: number; note?: string;
}

const HUMAN_BYTES = (n: number) =>
  n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(1)} GB`
  : n >= 1 << 20 ? `${Math.round(n / (1 << 20))} MB`
  : n >= 1 << 10 ? `${Math.round(n / (1 << 10))} KB` : `${n} B`;

const KIND_LABEL: Record<string, string> = {
  profile: "profile", workdir: "workdir",
};

/** Deleting is irreversible and partly destroys data Carveout did not
 *  create, so the dialog states the consequence rather than implying it:
 *  every path with its size, the capture as a separate opt-in, and the name
 *  typed back before the button arms. */
/** Picking the scene file.
 *
 *  A browser cannot hand a server a path: a file input yields the file's
 *  CONTENT, so an OS dialog here would mean uploading gigabytes to a server
 *  sitting on the same disk. So the picker is served instead of native —
 *  the documented data/scenes/ layout as a list, and a directory walk for
 *  captures kept anywhere else. Typing a path still works; it is the one
 *  input that never breaks.
 */
function ScenePicker({ value, onPick }:
    { value: string; onPick: (p: string) => void }) {
  const [known, setKnown] = useState<ConventionScene[] | null>(null);
  const [browsing, setBrowsing] = useState(false);
  const [listing, setListing] = useState<FsListing | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.get<{ scenes: ConventionScene[] }>("/api/fs/scenes")
      .then((r) => setKnown(r.scenes)).catch(() => setKnown([]));
  }, []);

  const browse = async (path?: string) => {
    setErr(null);
    try {
      setListing(await api.get<FsListing>(
        `/api/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`));
      setBrowsing(true);
    } catch (e: any) { setErr(e.message); }
  };

  const free = known?.length === 0;
  return (
    <div>
      {known && known.length > 0 && (
        <div className="border border-line rounded divide-y divide-line
                        mb-2 max-h-44 overflow-y-auto bg-inset">
          {known.map((k) => (
            <button key={k.path} type="button"
                    disabled={!!k.used_by}
                    aria-pressed={value === k.path}
                    onClick={() => onPick(k.path)}
                    className={cx(
                      "w-full text-left px-3 py-1.5 flex gap-3 items-center",
                      k.used_by ? "opacity-45 cursor-not-allowed"
                                : "hover:bg-hover cursor-pointer",
                      value === k.path && "bg-sel")}>
              <span className="font-mono text-[0.78rem] text-t1 grow
                               truncate">{k.path}</span>
              {k.used_by && (
                <Badge tone="muted">in library · {k.used_by}</Badge>)}
              <span className="font-mono text-[0.6875rem] text-t3 shrink-0
                               ui-num">
                {HUMAN_BYTES(k.bytes)}</span>
            </button>
          ))}
        </div>
      )}
      <div className="flex gap-2 mb-2">
        <input className={cx(INPUT, "w-full")} value={value}
               aria-label="scene file path"
               placeholder={free ? "data/scenes/MyScene/my-scene.sog"
                                 : "…or type a path"}
               onChange={(e) => onPick(e.target.value)} />
        <Btn variant="outline"
             onClick={() => browsing ? setBrowsing(false) : browse()}>
          {browsing ? "Close" : "Browse…"}</Btn>
      </div>
      {browsing && listing && (
        <div className="border border-line rounded mb-2 bg-inset">
          <div className="px-3 py-1.5 border-b border-line font-mono
                          text-[0.6875rem] text-t3 flex gap-2 items-center">
            <span className="grow truncate">{listing.path}</span>
            <button type="button" className="hover:text-t1 underline
                                             underline-offset-2"
                    onClick={() => browse(listing.repo)}>repo</button>
            <button type="button" className="hover:text-t1 underline
                                             underline-offset-2"
                    onClick={() => browse(listing.home)}>home</button>
          </div>
          <div className="max-h-56 overflow-y-auto divide-y divide-line">
            {listing.parent && (
              <button type="button" onClick={() => browse(listing.parent!)}
                      className="w-full text-left px-3 py-1.5 font-mono
                                 text-[0.78rem] text-t2 hover:bg-hover">
                ../</button>
            )}
            {listing.dirs.map((d) => (
              <button key={d.path} type="button"
                      onClick={() => browse(d.path)}
                      className="w-full text-left px-3 py-1.5 font-mono
                                 text-[0.78rem] text-t1 hover:bg-hover">
                {d.name}/</button>
            ))}
            {listing.files.map((f) => (
              <button key={f.path} type="button"
                      onClick={() => { onPick(f.path); setBrowsing(false); }}
                      className="w-full text-left px-3 py-1.5 flex gap-3
                                 items-center hover:bg-hover">
                <span className="font-mono text-[0.78rem] text-t1 font-medium
                                 grow truncate">{f.name}</span>
                <span className="font-mono text-[0.6875rem] text-t3 ui-num">
                  {HUMAN_BYTES(f.bytes ?? 0)}</span>
              </button>
            ))}
            {!listing.dirs.length && !listing.files.length && (
              <div className="px-3 py-2 ui-help">
                no folders, and no .ply or .sog here</div>
            )}
          </div>
        </div>
      )}
      {err && <div className="ui-note-warn mb-2" role="alert">{err}</div>}
    </div>
  );
}


function DeleteSceneDialog({ name, onClose, onDeleted }:
    { name: string; onClose: () => void; onDeleted: () => void }) {
  const [plan, setPlan] = useState<DeletionPlan | null>(null);
  const [typed, setTyped] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get<DeletionPlan>(`/api/scenes/${encodeURIComponent(name)}/deletion`)
      .then(setPlan).catch((e) => setErr(e.message));
  }, [name]);

  const blocked = plan?.running
    ? `${plan.running} is running; cancel it before deleting`
    : plan?.lock && plan.lock.alive && !plan.lock.mine
      ? `held by ${plan.lock.holder} since ${plan.lock.since}`
      : null;
  const rows = plan?.targets ?? [];
  const total = rows.filter((t) => t.removable)
                    .reduce((a, t) => a + t.bytes, 0);

  const del = async () => {
    setBusy(true); setErr(null);
    try {
      await api.del(`/api/scenes/${encodeURIComponent(name)}`,
                    { confirm: name });
      onDeleted();
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  return (
    <Modal onClose={onClose} width={560} title={`Delete scene ${name}`}>
      <DialogTitle tone="warn">Delete scene “{name}”</DialogTitle>
      {!plan && !err && <Spinner label="reading what is on disk" />}
      {plan && (
        <>
          <div className="text-[0.8125rem] text-t2 mb-2">
            This removes:</div>
          <div className="border border-line rounded divide-y
                          divide-line mb-3 bg-inset">
            {rows.map((t) => (
              <div key={t.kind} className="px-3 py-2 flex gap-3 items-start">
                <div className="ui-label w-24 shrink-0 pt-0.5">
                  {KIND_LABEL[t.kind]}</div>
                <div className="min-w-0 grow">
                  <div className="font-mono text-[0.78rem] text-t1 break-all">
                    {t.path}</div>
                  <div className="ui-help">
                    {t.exists ? t.note : "does not exist; nothing to remove"}
                  </div>
                </div>
                <div className="font-mono text-[0.78rem] text-t2 shrink-0
                                ui-num">
                  {t.exists && t.removable ? HUMAN_BYTES(t.bytes) : "—"}</div>
              </div>
            ))}
          </div>
          <div className="ui-help mb-3">
            The capture itself stays where it is; Carveout never deletes
            under <span className="font-mono">data/scenes</span>. Create the
            scene again to start over from it.
          </div>
          {blocked && (
            <div className="ui-note-warn mb-3">{blocked}</div>
          )}
          <FieldLabel>type {name} to confirm</FieldLabel>
          <input className={cx(INPUT, "w-full")} value={typed} autoFocus
                 aria-label={`type ${name} to confirm`}
                 onChange={(e) => setTyped(e.target.value)} />
        </>
      )}
      {err && <div className="ui-note-warn mt-3" role="alert">{err}</div>}
      <div className="flex justify-end gap-2 mt-4">
        <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
        <Btn variant="destructive" onClick={del}
             disabled={busy || !plan || typed !== name || !!blocked}>
          {busy ? "deleting…"
                : `Delete${total ? ` · ${HUMAN_BYTES(total)}` : ""}`}
        </Btn>
      </div>
    </Modal>
  );
}


/** "How was it filmed?" — three tiles, one click, always asked: the answer
 *  is `render.path_mode`, and it also picks the volume proposer's strategy.
 *  Carveout never classifies a scene; the operator, who was there, says. */
const FILMING = [
  { mode: "interior", title: "inside a space",
    sub: "eye-height cameras inside the walls" },
  { mode: "orbit", title: "around a subject",
    sub: "a ring of cameras around the volume" },
  { mode: "ground", title: "outdoors on the ground",
    sub: "person-height cameras over the local ground" },
] as const;

function NewSceneDialog({ onClose, onCreated }:
    { onClose: () => void; onCreated: (name: string) => void }) {
  const [scene, setScene] = useState("");
  const [name, setName] = useState("");
  const [workdir, setWorkdir] = useState("");
  // what the file costs before it is a scene — its Gaussian count
  // from the header, and how many classes of vocabulary the lift's class
  // pass fits on this card. Said here, where the file is chosen.
  const [info, setInfo] = useState<SplatInfo | null>(null);
  useEffect(() => {
    setInfo(null);
    if (!scene.trim()) return;
    let live = true;
    api.get<SplatInfo>(`/api/fs/splat_info?path=${encodeURIComponent(scene.trim())}`)
      .then((r) => { if (live) setInfo(r); })
      .catch(() => { if (live) setInfo(null); });
    return () => { live = false; };
  }, [scene]);
  // "Is this capture metric?" — one of two answers before Create: Metric
  // (factor 1.0), or "no, I will measure one thing on the canvas" (null:
  // the ruler at the volume gate records the factor; Propose refuses
  // until then). The number field stays for the operator who has it.
  const [metric, setMetric] = useState<"metric" | "measure" | null>(null);
  const [factor, setFactor] = useState("");
  const factorNum = factor.trim() === "" ? null : Number(factor);
  const factorBad = factorNum !== null && !(factorNum > 0);
  // The factor to write: the typed number wins, else 1.0 for Metric, else
  // null for "I will measure".
  const scaleAnswer: number | null | undefined =
    factorNum !== null && !factorBad ? factorNum
      : metric === "metric" ? 1 : metric === "measure" ? null : undefined;
  const [mode, setMode] = useState<string | null>(null);
  // "Which vision model?" — the third answer: the installed
  // models with what each needs beside THIS scene on THIS card, the best
  // fit PROPOSED (pre-selected, labelled so) and changeable here or later
  // on the vocabulary / verify panels. Carveout proposes; you choose.
  const [fit, setFit] = useState<FitEstimate | null>(null);
  const [model, setModel] = useState<string | null>(null);
  useEffect(() => {
    let dead = false;
    api.post<FitEstimate>("/api/scenes/estimate", { scene })
      .then((f) => { if (dead) return; setFit(f);
                     setModel((m) => m ?? f.proposed); })
      .catch(() => { if (!dead) setFit(null); });
    return () => { dead = true; };
  }, [scene]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const create = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.post<{ name: string }>("/api/scenes", {
        scene, scene_config: name || undefined,
        workdir: workdir || undefined, scale_m_per_unit: scaleAnswer ?? null,
        path_mode: mode, vlm_model_dir: model || undefined,
      });
      onCreated(r.name);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const choice = (on: boolean) => cx(
    "text-left border rounded px-2.5 py-1.5 text-[0.8125rem] leading-4",
    on ? "border-t2 bg-sel text-t1" : "border-line2 bg-inset text-t2 hover:bg-hover hover:text-t1");

  return (
    <Modal onClose={onClose} width={560} title="New scene">
      <DialogTitle>New scene</DialogTitle>
      <FieldLabel>scene .ply or .sog</FieldLabel>
      <ScenePicker value={scene} onPick={setScene} />
      {info?.gaussians != null && (
        <p className={cx("mt-1.5 mb-0",
                         info.classes_fit != null && info.classes_fit < 40
                           ? "ui-note-warn" : "ui-help")}
           data-testid="scene-size">
          <span className="font-mono">{(info.gaussians / 1e6).toFixed(1)} M</span>
          {" "}Gaussians · {HUMAN_BYTES(info.bytes)}
          {info.classes_fit != null ? (
            <>. On this card ({info.card_gb} GB) the lift fits a vocabulary of
              about <span className="font-mono">{info.classes_fit}</span> classes
              for this scene ({info.per_class_mb} MB each); the render and the
              object pass size themselves to the card.
              {info.classes_fit < 40 && " A large file for this card: keep the vocabulary short, or use a card with more memory."}
            </>
          ) : info.note ? <>. {info.note}</> : null}
        </p>)}
      <div className="grid grid-cols-2 gap-3 mt-3">
        <div>
          <FieldLabel>profile name (optional)</FieldLabel>
          <input className={cx(INPUT, "w-full")} value={name}
                 aria-label="profile name" placeholder="myscene"
                 onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <FieldLabel>workdir (optional)</FieldLabel>
          <input className={cx(INPUT, "w-full")} value={workdir}
                 aria-label="workdir"
                 placeholder="work/MyScene"
                 onChange={(e) => setWorkdir(e.target.value)} />
        </div>
      </div>

      <div className="mt-4">
        <FieldLabel>Is this capture metric?</FieldLabel>
        <div role="radiogroup" aria-label="is this capture metric"
             className="grid grid-cols-2 gap-2 mt-1">
          <button type="button" role="radio" aria-checked={metric === "metric"}
                  className={choice(metric === "metric")}
                  onClick={() => setMetric("metric")}>
            <b>Metric</b>
            <span className="block text-[0.75rem] text-t3">the file is in
              metres already (LiDAR, a phone rig, a metric-posed scan)</span>
          </button>
          <button type="button" role="radio" aria-checked={metric === "measure"}
                  className={choice(metric === "measure")}
                  onClick={() => setMetric("measure")}>
            <b>No, I will measure one thing on the canvas</b>
            <span className="block text-[0.75rem] text-t3">the ruler at the
              volume gate: two clicks on something whose length you know</span>
          </button>
        </div>
        <div className="mt-2 flex items-end gap-2">
          <div className="flex-1">
            <FieldLabel>metres per scene unit, if you know the number</FieldLabel>
            <input className={cx(INPUT, "w-full")} value={factor}
                   aria-label="metres per scene unit" placeholder="optional"
                   inputMode="decimal"
                   onChange={(e) => setFactor(e.target.value)} />
          </div>
        </div>
        {factorBad && <div className="mt-1 ui-note-fail" role="alert">
          the number must be greater than zero, or empty</div>}
        <div className="ui-help mt-1">
          A photogrammetry reconstruction carries an arbitrary unit the
          file does not record; you measure one known length on the scene
          and Carveout sizes everything from it.
        </div>
      </div>

      <div className="mt-4">
        <FieldLabel>How was it filmed?</FieldLabel>
        <div role="radiogroup" aria-label="how was it filmed"
             className="grid grid-cols-3 gap-2 mt-1">
          {FILMING.map((f) => (
            <button key={f.mode} type="button" role="radio"
                    aria-checked={mode === f.mode}
                    className={choice(mode === f.mode)}
                    onClick={() => setMode(f.mode)}>
              <b>{f.title}</b>
              <span className="block text-[0.75rem] text-t3">{f.sub}</span>
            </button>
          ))}
        </div>
        <div className="ui-help mt-1">
          This chooses how the cameras are placed; you can change it on the
          render panel later.
        </div>
      </div>

      {fit && fit.models.length > 0 && (
        <div className="mt-4">
          <FieldLabel>Which vision model?</FieldLabel>
          <div role="radiogroup" aria-label="which vision model"
               className="grid grid-cols-2 gap-2 mt-1">
            {fit.models.map((mm) => (
              <button key={mm.dir} type="button" role="radio"
                      aria-checked={model === mm.dir}
                      className={choice(model === mm.dir)}
                      onClick={() => setModel(mm.dir)}>
                <b>{mm.name}</b>
                {fit.proposed === mm.dir && (
                  <span className="ml-1.5 text-[0.6875rem] text-pass">proposed</span>)}
                <span className="block text-[0.75rem] text-t3">
                  needs about {mm.need_gib} GB
                  {mm.fits === "yes" ? ": fits" : mm.fits === "tight"
                    ? ": tight" : mm.fits === "no" ? ": does not fit" : ""}
                </span>
              </button>
            ))}
          </div>
          <div className="ui-help mt-1">
            {fit.card_gib != null
              ? <>This card has {fit.card_gib} GB
                  {fit.browser_gib != null && fit.gaussians != null && <>;
                    the browser's copy of this scene
                    ({(fit.gaussians / 1e6).toFixed(1)} M Gaussians) takes
                    about {fit.browser_gib} GB</>}. </>
              : <>No GPU found on this server. </>}
            The larger model reads scenes better; the smaller is faster and
            fits beside a large scene. You can change it later on the
            Vocabulary and Objects panels.
          </div>
        </div>
      )}

      {err && <div className="mt-3 ui-note-fail font-mono" role="alert">
        {err}</div>}
      <div className="flex justify-end gap-2 mt-4">
        <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
        <Btn variant="primary"
             disabled={!scene || factorBad || scaleAnswer === undefined
                       || !mode || busy}
             onClick={create}>
          {busy ? "Creating…" : "Create scene"}</Btn>
      </div>
    </Modal>
  );
}
