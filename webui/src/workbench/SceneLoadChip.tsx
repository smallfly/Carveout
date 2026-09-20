// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The scene file on its way, in the header beside the job chip: "Loading
// the scene · 412 MB of 1.9 GB", then "Preparing the scene…" while the
// splats are decoded, then nothing. A file that never arrives stays said,
// in red, with the browser's reason on hover.

import type { SceneLoad } from "./SplatCanvas";

/** 412 MB, 1.9 GB — one decimal above a gigabyte, none below. */
export const fmtBytes = (n: number): string => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} kB`;
  return `${Math.round(n)} B`;
};

export default function SceneLoadChip({ load }: { load: SceneLoad | null }) {
  if (!load || load.state === "ready") return null;
  const failed = load.state === "failed";
  const got = fmtBytes(load.loaded), all = fmtBytes(load.total);
  // the last bytes decode as they land: once the two figures read the
  // same, "1.9 GB of 1.9 GB" says less than what is happening
  const preparing = load.state === "preparing" || (load.total > 0 && got === all);
  const text = failed ? "The scene file could not be loaded"
    : preparing ? "Preparing the scene…"
    : load.total > 0 ? `Loading the scene · ${got} of ${all}`
    : "Loading the scene…";
  return (
    <span role="status"
          title={failed ? (load.error || "the file did not arrive")
                        : "the 3D scene is on its way to the view"}
          className={`inline-flex items-center gap-1.5 h-6 px-2 rounded-full
                      text-[0.75rem] font-medium whitespace-nowrap ui-num
                      ${failed ? "bg-fail/15 text-fail" : "bg-primary/15 text-t1"}`}>
      <span className={`w-1.5 h-1.5 rounded-full
                        ${failed ? "bg-fail" : "bg-primary animate-pulse"}`}
            aria-hidden="true" />
      {text}
    </span>
  );
}
