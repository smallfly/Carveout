// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The running job, in the header: name · k/n · elapsed, as a button that
// opens the log drawer. A pointer to the stage strip, not a second source:
// the same useJobProgress reading feeds both.

import { useEffect, useState } from "react";
import { useJobProgress } from "../store";
import { jobLabel } from "../labels";

const elapsedOf = (started: string | null, now: number): string | null => {
  if (!started) return null;
  const t = new Date(started.replace(" ", "T")).getTime();
  if (Number.isNaN(t)) return null;
  const s = Math.max(0, Math.floor((now - t) / 1000));
  const m = Math.floor(s / 60), h = Math.floor(m / 60);
  const two = (v: number) => String(v).padStart(2, "0");
  return h > 0 ? `${h}:${two(m % 60)}:${two(s % 60)}` : `${m}:${two(s % 60)}`;
};

export default function JobChip({ onOpenLog }: { onOpenLog: () => void }) {
  const job = useJobProgress();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!job) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [job !== null]);
  if (!job) return null;
  const name = jobLabel(job.kind) || "A job is running";
  const elapsed = elapsedOf(job.started, now);
  const parts = [name];
  if (job.k !== null && job.n !== null) parts.push(`${job.k}/${job.n}`);
  if (elapsed) parts.push(elapsed);
  return (
    <button onClick={onOpenLog}
            aria-label={`${name}${job.n ? `, ${job.k} of ${job.n}` : ""}`
                        + `${elapsed ? `, running for ${elapsed}` : ""}`
                        + "; open the log"}
            title="open the log"
            className="inline-flex items-center gap-1.5 h-6 px-2 rounded-full
                       bg-warn/15 text-warn text-[0.75rem] font-medium
                       whitespace-nowrap hover:bg-warn/25">
      <span className="w-1.5 h-1.5 rounded-full bg-warn animate-pulse"
            aria-hidden="true" />
      {parts.join(" · ")}
    </button>
  );
}
