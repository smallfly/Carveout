// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Minimal primitives, hand-rolled as Tailwind recipes over the semantic
// tokens in theme.css (no component library). Neutral chrome: the one
// primary Button per moment is a filled neutral, selection is a filled
// neutral surface with weight, and green/amber/red appear only as state.
// Sans for chrome, mono for paths/IDs/values, kbd hints as chips.

import React from "react";
import { createPortal } from "react-dom";
import { isTypingTarget } from "@viewer/stage.js";
import { APPEARANCES, APPEARANCE_LABEL, useAppearance } from "./theme";
import { useJobProgress } from "./store";
import { jobLabel } from "./labels";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

/** The one text-field recipe (theme.css .ui-input); extend with width. */
export const INPUT = "ui-input";

type BtnVariant = "primary" | "outline-primary" | "outline" | "destructive"
  | "ghost";

// -- keyboard chips -----------------------------------------------------------
// A `kbd` chip on a Btn IS its shortcut: mounting the button registers the
// key, and one document listener clicks the matching button. The chip cannot
// advertise a key that does nothing (every chip was dead once, for that reason).
// Rules: no modifiers, no key-repeat; ignored while a text field has focus;
// while a Modal is open only buttons inside it are eligible (Enter must
// never approve a gate behind a confirm dialog — and Confirm's buttons carry
// no kbd, so modals stay deliberately mouse-only); when several enabled
// buttons share a key, the LAST mounted wins — the most recently opened
// context (the capture section's Save pose over the gate's Approve).
const kbdRegistry = new Map<HTMLButtonElement, string>();
let kbdListening = false;
const KBD_KEY: Record<string, string> = { "⏎": "Enter" };

function onKbdKey(e: KeyboardEvent) {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  if (isTypingTarget(e.target as HTMLElement)) return;
  const inModal = document.querySelector("[data-kbd-modal]") !== null;
  let hit: HTMLButtonElement | null = null;
  for (const [el, key] of kbdRegistry) {     // Map preserves insertion order
    if ((KBD_KEY[key] ?? key).toLowerCase() !== e.key.toLowerCase()) continue;
    if (!el.isConnected || el.disabled || el.offsetParent === null) continue;
    if (inModal && !el.closest("[data-kbd-modal]")) continue;
    hit = el;
  }
  if (hit) {
    e.preventDefault();
    hit.click();
  }
}

function ensureKbdListener() {
  if (kbdListening) return;
  kbdListening = true;
  document.addEventListener("keydown", onKbdKey);
}

export function Btn({ variant = "outline", kbd, className, children,
                      ...rest }:
    React.ButtonHTMLAttributes<HTMLButtonElement> &
    { variant?: BtnVariant; kbd?: string }) {
  const v: Record<BtnVariant, string> = {
    primary: "bg-primary text-primary-fg border-primary font-medium "
      + "hover:opacity-90",
    // emphasised secondary: a stronger edge and primary text, no fill
    "outline-primary": "border-t3 text-t1 bg-inset hover:bg-hover",
    outline: "border-line2 text-t1 bg-inset hover:bg-hover hover:border-t4",
    destructive: "border-fail/60 text-fail bg-inset hover:bg-fail/10",
    ghost: "border-transparent text-t2 hover:text-t1 hover:bg-hover",
  };
  // Stable identity per `kbd` so registration order tracks MOUNT order (an
  // inline ref would re-register on every render and scramble last-wins).
  const held = React.useRef<HTMLButtonElement | null>(null);
  const refKbd = React.useCallback((el: HTMLButtonElement | null) => {
    if (held.current) kbdRegistry.delete(held.current);
    held.current = el;
    if (el && kbd) {
      kbdRegistry.set(el, kbd);
      ensureKbdListener();
    }
  }, [kbd]);
  return (
    <button
      ref={refKbd}
      className={cx(
        "inline-flex items-center gap-2 border rounded px-2.5 h-7",
        "text-[0.8125rem] leading-none whitespace-nowrap select-none",
        // no pointer-events-none: the disabled attribute already blocks
        // the click, and a disabled button that took no pointer events
        // could never show its title — the one place many buttons said
        // why they are dead
        "disabled:opacity-45 disabled:cursor-not-allowed",
        v[variant], className)}
      {...rest}>
      {children}
      {kbd && <Kbd>{kbd}</Kbd>}
    </button>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-[0.6875rem] px-1 py-px border
                     border-current/30 rounded-sm opacity-70 leading-none"
          aria-label={`shortcut ${children}`}>{children}</span>
  );
}

type BadgeTone = "pass" | "warn" | "fail" | "accent" | "muted";
/** Status chip: soft tinted fill, no outline; the word carries the state
 *  beside the colour. `accent` is the neutral "notable" tone. */
export function Badge({ tone = "muted", className, children }:
    { tone?: BadgeTone; className?: string; children: React.ReactNode }) {
  const t: Record<BadgeTone, string> = {
    pass: "bg-pass/15 text-pass",
    warn: "bg-warn/15 text-warn",
    fail: "bg-fail/15 text-fail",
    accent: "bg-sel text-t1",
    muted: "bg-hover text-t3",
  };
  return (
    <span
          className={cx("inline-block text-[0.6875rem] font-medium",
                        "px-1.5 py-px rounded-sm leading-4 whitespace-nowrap",
                        t[tone], className)}>{children}</span>
  );
}

/** A panel section: sans title, one subtle divider beneath the group.
 *  `locked` names the state of its controls (read-only gate or held
 *  scene) beside the title, so a dimmed field is never the only clue. */
/** The bar on the panel's left edge that marks the block owning the
 *  panel's current step (steps.ts currentStep) — the cue colour, the one
 *  colour in a neutral panel, which the step list's dot wears too, so the
 *  list's "you are here" points at the controls that do it (white read as
 *  just another line). `inset` for a block inside a
 *  Section (whose padding it sits within). */
function CueBar({ inset }: { inset?: boolean }) {
  return <span aria-hidden="true"
               className={cx("absolute top-0 bottom-0 w-0.5 rounded bg-cue",
                             inset ? "-left-4" : "left-0")} />;
}

/** A block inside a Section that owns the current step: the cue bar
 *  beside it while `on`, nothing otherwise. */
export function Cued({ on, children, className }:
    { on: boolean; children: React.ReactNode; className?: string }) {
  return (
    <div className={cx("relative", className)}>
      {on && <CueBar inset />}
      {children}
    </div>
  );
}

export function Section({ title, children, right, locked, cue, band }:
    { title: string; children: React.ReactNode;
      right?: React.ReactNode; locked?: boolean;
      /** this section owns the panel's current step */
      cue?: boolean;
      /** the panel's "where am I" block — the gate's name, its sentence
       *  and its steps — on a lifted surface so it stands apart from the
       *  acts below it */
      band?: boolean }) {
  return (
    <section className={cx("relative border-b px-4 py-3",
                           band ? "bg-inset border-line2" : "border-line")}>
      {cue && <CueBar />}
      <div className="flex items-center gap-2 mb-2 min-h-[20px]">
        <h3 className="ui-cap text-[0.8125rem] font-semibold text-t1 m-0
                       leading-5 truncate">{title}</h3>
        {locked && (
          <span className="inline-flex items-center gap-1 text-[0.6875rem]
                           text-t3 whitespace-nowrap">
            <Icon name="lock" className="w-3 h-3" />Read-only
          </span>
        )}
        <span className="flex-1" />
        {right}
      </div>
      {children}
    </section>
  );
}

/** Secondary explanation behind a native disclosure: keyboard-operable,
 *  its open state exposed by the element itself; toggling it touches no
 *  input and no state. The visible sentence above it must already say what
 *  the section is for — this holds the rest. */
export function Details({ summary = "Details", children, className }:
    { summary?: string; children: React.ReactNode; className?: string }) {
  return (
    <details className={cx("ui-details mt-1", className)}>
      <summary>{summary}</summary>
      <div className="ui-help mt-1 pl-3 border-l border-line">
        {children}
      </div>
    </details>
  );
}

/** The ordered acts of a gate, with the current one marked: a ✓ for the
 *  done ones, a filled dot and weight for the current, a hollow dot for
 *  the rest — never colour alone. `current` is the first step not done;
 *  nothing else decides it. Pure render: the lists live in steps.ts. */
export type Step = { id: string; label: string; done: boolean; note?: string };
export function Steps({ steps }: { steps: Step[] }) {
  const current = steps.findIndex((s) => !s.done);
  return (
    <ol aria-label="steps for this gate"
        className="list-none m-0 mt-2 p-0 flex flex-col gap-0.5">
      {steps.map((s, i) => {
        const isCurrent = i === current;
        const later = current !== -1 && i > current;
        return (
          <li key={s.id} aria-current={isCurrent ? "step" : undefined}
              className={cx("flex items-baseline gap-2 text-[0.75rem] leading-5",
                            s.done ? "text-t3" : isCurrent ? "text-t1 font-medium"
                                                           : "text-t4")}>
            <span className="w-3 shrink-0 inline-flex justify-center"
                  aria-hidden="true">
              {s.done ? <span className="text-pass">✓</span>
                : <span className={cx("inline-block w-1.5 h-1.5 rounded-full",
                                      isCurrent ? "bg-cue"
                                                : "border border-line2")} />}
            </span>
            <span>
              {s.label}
              {isCurrent && s.note && (
                <span className="text-t3 font-normal">: {s.note}</span>)}
              {later && <span className="sr-only"> (later)</span>}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/** The scene's vision model, chosen among the installed ones: the same
 *  row on the vocabulary and verify panels. A calibration write —
 *  `vlm.model_dir` — that re-opens the verify gate only. */
export function ModelChoice({ models, current, disabled, onChoose }: {
  models: { dir: string; name: string; need_gib: number; how: string }[];
  current: string | null | undefined; disabled?: boolean;
  onChoose: (dir: string) => void;
}) {
  if (models.length < 2) return null;
  const cur = models.find((m) => m.dir === current);
  return (
    <div className="flex items-center gap-2 mt-2 flex-wrap">
      <span className="ui-label flex-1">
        vision model
        <span className="block text-t3">
          {cur ? `needs about ${cur.need_gib} GB (${cur.how})` : ""}</span>
      </span>
      <Segmented value={cur?.name ?? (current ?? "")}
                 options={models.map((m) => m.name)} label="vision model"
                 disabled={disabled}
                 onChange={(v) => {
                   const m = models.find((x) => x.name === v);
                   if (m && m.dir !== current) onChoose(m.dir);
                 }} />
    </div>
  );
}

/** Where a panel's panel-level actions land: the inspector's fixed footer.
 *  The panel keeps owning the buttons (handlers, kbd chips, availability);
 *  only their DOM position moves, through a portal to the slot the
 *  Workbench provides. Without a slot they render in place. */
export const FooterSlot = React.createContext<HTMLElement | null>(null);
export function PanelFooter({ children }: { children: React.ReactNode }) {
  const slot = React.useContext(FooterSlot);
  const inner = (
    <div className="flex gap-2 flex-wrap items-center w-full">{children}</div>
  );
  return slot ? createPortal(inner, slot)
              : <div className="p-4 border-t border-line">{inner}</div>;
}

/** Styled range input: 4px track, neutral filled run, 14px thumb, 24px hit
 * height. The fill rides a CSS var; recipe in theme.css. */
export function Slider({ value, min, max, step, onChange, onCommit,
                         disabled, className, label }: {
  value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; onCommit?: () => void;
  disabled?: boolean; className?: string; label?: string;
}) {
  const fill = Math.max(0, Math.min(100,
    ((value - min) / (max - min || 1)) * 100));
  return (
    <input type="range" min={min} max={max} step={step} value={value}
           disabled={disabled} aria-label={label}
           className={cx("ui-slider min-w-0", className)}
           style={{ "--fill": `${fill}%` } as React.CSSProperties}
           onChange={(e) => onChange(+e.target.value)}
           onPointerUp={() => onCommit?.()}
           onKeyUp={() => onCommit?.()} />
  );
}

/** Small switch (32×18) — the knob's position is the state, the fill
 *  only underlines it. */
export function Switch({ checked, onChange, disabled, label }: {
  checked: boolean; onChange: (v: boolean) => void; disabled?: boolean;
  label?: string;
}) {
  return (
    <button type="button" role="switch" aria-checked={checked}
            aria-label={label}
            disabled={disabled}
            onClick={() => onChange(!checked)}
            className={cx(
              "relative w-8 h-[18px] rounded-full border shrink-0",
              "transition-colors",
              disabled ? "cursor-not-allowed"
                : checked ? "bg-primary border-primary"
                          : "bg-inset border-line2 hover:border-t4",
              disabled && (checked ? "bg-t3 border-t3"
                                   : "bg-hover border-line"))}>
      <span className={cx(
        "absolute top-[2px] left-[2px] w-3 h-3 rounded-full",
        "transition-transform",
        checked ? "translate-x-[14px]" : "",
        disabled ? (checked ? "bg-panel" : "bg-t4")
                 : checked ? "bg-primary-fg" : "bg-t3")} />
    </button>
  );
}

/** Numeric input with a commit-on-blur/Enter text buffer: type freely,
 * parse once on commit; while unfocused it re-renders the authoritative
 * value. The ONE implementation of this pattern. */
export function NumInput({ value, onCommit, fmt, className, disabled,
                           ...rest }:
    Omit<React.InputHTMLAttributes<HTMLInputElement>,
         "value" | "onChange"> & {
      value: number | null | undefined;
      onCommit: (v: number) => void;
      fmt?: (v: number) => string;
    }) {
  const [text, setText] = React.useState<string | null>(null);
  // What the field showed when focus arrived. A commit is a CHANGE: blur
  // used to commit whatever was in the field, so tabbing through a
  // calibration row wrote its unchanged value to the profile and re-opened
  // the gate (a no-op `min_sharpness: 0.008` landed in a profile that
  // way). Compared as numbers against the text shown at focus, not
  // against `value`: a `fmt` that rounds would otherwise read as a change.
  const start = React.useRef<string>("");
  const shown = text ?? (value === null || value === undefined
    ? "" : (fmt ? fmt(value) : String(value)));
  return (
    <input
      className={cx(INPUT, "text-right ui-num", className)}
      value={shown}
      disabled={disabled}
      onFocus={() => { start.current = shown; setText(shown); }}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        const v = text === null ? NaN : parseFloat(text);
        const was = parseFloat(start.current);
        if (!Number.isNaN(v) && (Number.isNaN(was) || v !== was))
          onCommit(v);
        setText(null);
      }}
      onKeyDown={(e) => e.key === "Enter" &&
        (e.target as HTMLInputElement).blur()}
      {...rest} />
  );
}

/** Labeled numeric input that commits on blur/Enter (calibration
 * rows). An optional right-aligned hint (e.g. "auto 0.888") sits before
 * the field. */
export function LabeledNum({ label, value, onCommit, disabled, hint }: {
  label: string; value: number | null | undefined;
  onCommit: (v: number) => void; disabled?: boolean; hint?: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-2 min-h-[28px]">
      <span className="ui-label flex-1">{label}</span>
      {hint}
      <NumInput className="w-20 shrink-0" value={value}
                onCommit={onCommit} disabled={disabled} />
    </label>
  );
}

/** Segmented control: one bordered group, the pressed option is a filled
 *  neutral with primary text and weight — never colour alone. */
export function Segmented({ value, options, onChange, disabled, label }: {
  value: string; options: readonly string[];
  onChange: (v: string) => void; disabled?: boolean; label?: string;
}) {
  return (
    <div role="radiogroup" aria-label={label}
         className={cx("inline-flex rounded border overflow-hidden",
                       disabled ? "border-line bg-hover/60 cursor-not-allowed"
                                : "border-line2 bg-inset")}>
      {options.map((o) => (
        <button key={o} type="button" role="radio"
                aria-checked={value === o}
                disabled={disabled}
                onClick={() => onChange(o)}
                className={cx(
                  "ui-cap text-[0.75rem] px-2 h-6 leading-none",
                  "border-r last:border-r-0",
                  disabled ? "border-line cursor-not-allowed" : "border-line2",
                  value === o
                    ? (disabled ? "bg-sel text-t2 font-medium"
                                : "bg-sel text-t1 font-medium")
                    : (disabled ? "text-t4"
                                : "text-t3 hover:text-t1 hover:bg-hover"))}>
          {o}
        </button>
      ))}
    </div>
  );
}

export function ProgressBar({ pct, className }:
    { pct: number | null; className?: string }) {
  return (
    <div className={cx("h-1.5 bg-line2 rounded-sm overflow-hidden",
                       className)}
         role="progressbar"
         aria-valuenow={pct === null ? undefined : Math.round(pct * 100)}>
      <div
        className={cx("h-full bg-t1 transition-[width]",
                      pct === null && "w-1/3 animate-pulse")}
        style={pct !== null ? { width: `${Math.round(pct * 100)}%` }
                            : undefined} />
    </div>
  );
}

export function Modal({ onClose, children, width = 480, title }:
    { onClose?: () => void; children: React.ReactNode; width?: number;
      title?: string }) {
  return (
    <div className="fixed inset-0 z-50 bg-app/70 flex items-center
                    justify-center p-4"
         onMouseDown={(e) => e.target === e.currentTarget && onClose?.()}>
      <div className="bg-panel border border-line2 rounded-md p-4 max-w-full
                      max-h-full overflow-y-auto shadow-[0_8px_32px_rgba(0,0,0,0.35)]"
           data-kbd-modal role="dialog" aria-modal="true"
           aria-label={title}
           style={{ width }}>
        {children}
      </div>
    </div>
  );
}

/** Dialog heading: sans, sentence case. */
export function DialogTitle({ children, tone }:
    { children: React.ReactNode; tone?: "warn" | "fail" }) {
  return (
    <h2 className={cx("text-[0.9375rem] font-semibold m-0 mb-3 leading-5",
                      tone === "warn" ? "text-warn"
                        : tone === "fail" ? "text-fail" : "text-t1")}>
      {children}</h2>
  );
}

/** Form label above a field. */
export function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label className="ui-label block mb-1">{children}</label>
  );
}

/** AlertDialog: cascade warnings, deletes — explicit confirm/cancel. */
export function Confirm({ title, body, confirmLabel, destructive, onConfirm,
                          onCancel }:
    { title: string; body: React.ReactNode; confirmLabel: string;
      destructive?: boolean; onConfirm: () => void; onCancel: () => void }) {
  return (
    <Modal onClose={onCancel} title={title}>
      <DialogTitle>{title}</DialogTitle>
      <div className="text-t2 text-[0.8125rem] leading-5 mb-4">{body}</div>
      <div className="flex justify-end gap-2">
        <Btn variant="ghost" onClick={onCancel}>Cancel</Btn>
        <Btn variant={destructive ? "destructive" : "primary"}
             onClick={onConfirm}>{confirmLabel}</Btn>
      </div>
    </Modal>
  );
}

/** A dead primary must say why: while a job runs, action rows name the
 * job — in words, in the warn colour, with a pulsing dot (the dot moves,
 * the words do not) — instead of graying out silently. */
export function JobNote({ show }: { show: boolean }) {
  const job = useJobProgress();
  if (!show) return null;
  const name = jobLabel(job?.kind) || "A job is running";
  return (
    <span className="inline-flex items-center gap-1.5 text-[0.75rem]
                     text-warn" role="status">
      <span className="w-1.5 h-1.5 rounded-full bg-warn animate-pulse
                       shrink-0" aria-hidden="true" />
      {name}: live progress in the stage strip
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="text-[0.78rem] text-t3 animate-pulse py-2">
      {label ?? "loading…"}
    </div>
  );
}

/** Dark / Light / System. Reachable from both screens' headers; the
 *  choice lives in a namespaced browser preference (theme.ts), never in
 *  pipeline configuration or scene files. */
export function AppearanceMenu() {
  const [pref, setPref] = useAppearance();
  return (
    <div role="radiogroup" aria-label="Appearance" data-testid="appearance"
         className="inline-flex rounded border border-line2 bg-inset
                    overflow-hidden">
      {APPEARANCES.map((a) => (
        <button key={a} type="button" role="radio" aria-checked={pref === a}
                onClick={() => setPref(a)}
                className={cx(
                  "text-[0.75rem] px-2 h-6 leading-none border-r border-line2",
                  "last:border-r-0",
                  pref === a ? "bg-sel text-t1 font-medium"
                             : "text-t3 hover:text-t1 hover:bg-hover")}>
          {APPEARANCE_LABEL[a]}
        </button>
      ))}
    </div>
  );
}

// -- small inline icons (16px grid, currentColor) -----------------------------
type IconName = "check" | "warn" | "circle" | "dot" | "gear" | "play"
  | "doc" | "spinner" | "lock";
export function Icon({ name, className }:
    { name: IconName; className?: string }) {
  const p = { width: 14, height: 14, viewBox: "0 0 16 16", fill: "none",
              stroke: "currentColor", strokeWidth: 1.6,
              strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
              "aria-hidden": true, className: cx("shrink-0", className) };
  switch (name) {
    case "check":
      return <svg {...p}><circle cx="8" cy="8" r="6.25" />
        <path d="M5.2 8.2l1.9 1.9 3.8-4" /></svg>;
    case "warn":
      return <svg {...p}><path d="M8 2.2L14.2 13H1.8z" />
        <path d="M8 6.3v3.2M8 11.6v.1" /></svg>;
    case "circle":
      return <svg {...p}><circle cx="8" cy="8" r="6.25" /></svg>;
    case "dot":
      return <svg {...p}><circle cx="8" cy="8" r="6.25" />
        <circle cx="8" cy="8" r="2.4" fill="currentColor" stroke="none" /></svg>;
    case "spinner":
      return <svg {...p} className={cx("shrink-0 animate-spin", className)}>
        <path d="M8 1.75a6.25 6.25 0 1 1-6.25 6.25" /></svg>;
    case "gear":
      return <svg {...p}><circle cx="8" cy="8" r="2.2" />
        <path d="M8 1.8v1.7M8 12.5v1.7M1.8 8h1.7M12.5 8h1.7M3.6 3.6l1.2 1.2M11.2 11.2l1.2 1.2M3.6 12.4l1.2-1.2M11.2 4.8l1.2-1.2" /></svg>;
    case "play":
      return <svg {...p}><path d="M4.5 2.8v10.4L13 8z" /></svg>;
    case "doc":
      return <svg {...p}><path d="M4 1.8h5.2L12.5 5v9.2H4z" />
        <path d="M9 1.8V5h3.5M6 8h4M6 10.5h4" /></svg>;
    case "lock":
      return <svg {...p}><rect x="3.5" y="7" width="9" height="7" rx="1.2" />
        <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" /></svg>;
  }
}
