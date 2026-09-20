// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

/** Semantic tokens only. Every colour is an `rgb(var(--c-*) / <alpha>)`
 * reference into src/theme.css, where the dark and light palettes live —
 * so `bg-panel/90` keeps working and the theme is one attribute on <html>.
 * Character rules: dense, sans for chrome, mono for paths/IDs/values,
 * colour only as state, no shadows/glows/entrance animations. */
const c = (name) => `rgb(var(--c-${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        app: c("app"),            // application background
        panel: c("panel"),        // rail, header, inspector, dialogs
        inset: c("inset"),        // inputs, wells, thumbnails' ground
        hover: c("hover"),        // hover surface
        sel: c("sel"),            // selected surface
        line: c("line"),          // structural divider
        line2: c("line2"),        // control edge (stronger)
        t1: c("t1"),              // primary text
        t2: c("t2"),              // secondary text
        t3: c("t3"),              // tertiary text (labels, helper prose)
        t4: c("t4"),              // faint text (placeholders, idle)
        primary: { DEFAULT: c("primary"), fg: c("primary-fg") },
        pass: c("pass"),
        warn: c("warn"),
        fail: c("fail"),
        focus: c("focus"),
        cue: c("cue"),            // the current step's bar and dot
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      borderRadius: { sm: "3px", DEFAULT: "4px", md: "5px" },
    },
  },
  plugins: [],
};
