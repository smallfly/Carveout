// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// The gate rail (01): the journal made visible. States: approved / active
// / locked / invalid(STALE) / running. Selection (which panel is open) is
// the filled row; completion is the icon beside the name — an approved
// gate keeps its check when it is the one selected. Order and gate
// semantics are the journal's; the rail only names them.

import { gateState } from "../journal";
import { GATE_LABEL } from "../labels";
import { GATES, type Gate, type Journal } from "../types";
import { Icon, cx } from "../ui";

type RailKey = Gate | "report" | "settings";
type RailState = "approved" | "stale" | "running" | "idle";

export default function GateRail({ journal, active, onSelect }: {
  journal: Journal | null;
  active: RailKey | null;
  onSelect: (g: RailKey) => void;
}) {
  const node = (key: RailKey, label: string, state: RailState,
                icon?: "gear" | "doc") => {
    const selected = active === key;
    const stateIcon = icon ? icon
      : state === "approved" ? "check"
      : state === "stale" ? "warn"
      : state === "running" ? "spinner"
      : selected ? "dot" : "circle";
    return (
      <button key={key}
              aria-current={selected ? "page" : undefined}
              onClick={() => onSelect(key)}
              className={cx(
                "relative mx-2 h-8 rounded px-2 flex items-center gap-2",
                "text-[0.8125rem] leading-none text-left",
                selected ? "bg-sel text-t1 font-medium"
                         : "text-t2 hover:bg-hover hover:text-t1",
                state === "idle" && !selected && !icon && "text-t3")}>
        {selected && (
          <span aria-hidden="true"
                className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded
                           bg-t1" />
        )}
        <Icon name={stateIcon}
              className={cx(
                state === "approved" && "text-pass",
                state === "stale" && "text-warn",
                state === "running" && "text-t1",
                (state === "idle" || icon) && !selected && "text-t4")} />
        <span className="truncate">{label}</span>
        <span className="sr-only">
          {state === "approved" ? ", approved"
            : state === "stale" ? ", stale"
            : state === "running" ? ", running" : ""}
        </span>
      </button>
    );
  };

  return (
    <nav aria-label="Gates"
         className="w-[136px] min-[2200px]:w-[152px] shrink-0 border-r
                    border-line bg-panel flex
                    flex-col gap-1 py-2 z-10">
      {GATES.map((g) =>
        node(g, GATE_LABEL[g], (gateState(journal, g) ?? "idle") as RailState))}
      <div className="flex-1" />
      {/* The pipeline runner is the Objects point's own panel — a separate
          "run" node opened the same panel without the approved-gate
          read-only wrapper (a skip path, so it was removed). A running
          job shows in the header and the stage strip. */}
      {node("settings", "Viewer", "idle", "gear")}
      {node("report", "Report", "idle", "doc")}
    </nav>
  );
}
