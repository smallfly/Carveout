// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

import React from "react";
import ReactDOM from "react-dom/client";
// Inter for interface text (the operator's choice); IBM Plex Mono
// stays for paths, IDs and values. Bundled via @fontsource — no CDN.
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/600.css";
// label pills use Figtree (the label engine's face — theme.css #wb-labels);
// bundled like the Plex faces — no CDN at runtime
import "@fontsource/figtree/400.css";
import "@fontsource/figtree/600.css";
import "./theme.css";
import { initAppearance } from "./theme";
import App from "./App";

initAppearance();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
