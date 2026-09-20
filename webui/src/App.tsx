// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// Two-screen IA (design 01): / = Scene Library (the only non-3D screen),
// /scene/<name> = the Workbench. Tiny history-based router — no route per
// gate; the journal decides what is active.

import { useEffect, useState } from "react";
import Library from "./library/Library";
import Workbench from "./workbench/Workbench";
import { RunProvider } from "./store";

function parse(path: string): { scene: string | null } {
  const m = path.match(/^\/scene\/([^/]+)/);
  return { scene: m ? decodeURIComponent(m[1]) : null };
}

export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const nav = (to: string) => {
    window.history.pushState(null, "", to);
    setPath(to);
  };

  const { scene } = parse(path);
  if (scene) {
    return (
      <RunProvider scene={scene} key={scene}>
        <Workbench onExit={() => nav("/")} />
      </RunProvider>
    );
  }
  return <Library onOpen={(name) => nav(`/scene/${encodeURIComponent(name)}`)} />;
}
