// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Appearance (dark / light / system): one attribute on <html>, one
// namespaced localStorage key, one OS listener that exists only while
// "system" is selected. index.html paints the initial value inline before
// the bundle loads (no wrong-theme flash); this module re-asserts it and
// owns every later change. Pure chrome: nothing here touches the canvas,
// the scene's own background, viewer_settings.json or any pipeline state.

import { useSyncExternalStore } from "react";

export type Appearance = "dark" | "light" | "system";
export const APPEARANCES: readonly Appearance[] = ["dark", "light", "system"];
export const APPEARANCE_LABEL: Record<Appearance, string> = {
  dark: "Dark", light: "Light", system: "System",
};

const KEY = "carveout_ui_theme";
const MQ = "(prefers-color-scheme: light)";

function read(): Appearance {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "system" ? v : "dark";
  } catch {
    return "dark";           // storage unavailable: the default, never an error
  }
}

let pref: Appearance = read();
const subs = new Set<() => void>();
let mql: MediaQueryList | null = null;

export function resolveAppearance(p: Appearance): "dark" | "light" {
  if (p !== "system") return p;
  try {
    return window.matchMedia(MQ).matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function paint() {
  document.documentElement.dataset.theme = resolveAppearance(pref);
}

const onOsChange = () => { if (pref === "system") paint(); };

/** Attach the OS listener only while System is selected; drop it otherwise. */
function syncListener() {
  if (pref === "system" && !mql) {
    try {
      mql = window.matchMedia(MQ);
      mql.addEventListener("change", onOsChange);
    } catch {
      mql = null;
    }
  } else if (pref !== "system" && mql) {
    mql.removeEventListener("change", onOsChange);
    mql = null;
  }
}

export function setAppearance(p: Appearance) {
  pref = p;
  try {
    localStorage.setItem(KEY, p);
  } catch {
    /* storage unavailable: the choice still applies for this session */
  }
  paint();
  syncListener();
  subs.forEach((f) => f());
}

/** Called once at startup: (re)paint from storage and arm the listener. */
export function initAppearance() {
  paint();
  syncListener();
}

const subscribe = (f: () => void) => {
  subs.add(f);
  return () => { subs.delete(f); };
};

export function useAppearance(): [Appearance, (p: Appearance) => void] {
  const p = useSyncExternalStore(subscribe, () => pref, () => pref);
  return [p, setAppearance];
}
