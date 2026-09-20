// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// REST + SSE client. Refusals (409/423/4xx) surface as ApiError with the
// pipeline's verbatim message + the routing gate — never swallowed.

import type { WsEvent } from "./types";

export class ApiError extends Error {
  status: number;
  gate: string | null;
  remedy: string | null;
  holder?: string;
  stale?: boolean;

  constructor(status: number, body: Record<string, any>) {
    super(body.error || `HTTP ${status}`);
    this.status = status;
    this.gate = body.gate ?? null;
    this.remedy = body.remedy ?? null;
    this.holder = body.holder;
    this.stale = body.stale;
  }
}

async function req<T>(method: string, url: string, body?: unknown):
    Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body !== undefined
      ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const isJson = (r.headers.get("Content-Type") || "").includes("json");
  const data = isJson ? await r.json() : await r.text();
  if (!r.ok) throw new ApiError(r.status, isJson ? data : { error: data });
  return data as T;
}

export const api = {
  get: <T>(url: string) => req<T>("GET", url),
  post: <T>(url: string, body?: unknown) => req<T>("POST", url, body ?? {}),
  put: <T>(url: string, body: unknown) => req<T>("PUT", url, body),
  patch: <T>(url: string, body: unknown) => req<T>("PATCH", url, body),
  // DELETE carries a body: the typed confirmation, and which of the
  // scene's belongings the operator opted into removing.
  del: <T>(url: string, body?: unknown) => req<T>("DELETE", url, body),
};

export const runUrl = (scene: string, path: string) =>
  `/api/runs/${encodeURIComponent(scene)}${path}`;

/** One SSE stream per scene. The browser re-sends Last-Event-ID on its
 * automatic reconnects; the server replays the missed buffer
 * (re-attach — the run lives server-side). */
export function openEvents(scene: string,
                           onEvent: (ev: WsEvent) => void,
                           onState: (connected: boolean) => void) {
  const es = new EventSource(runUrl(scene, "/events"));
  const types = ["journal", "log", "stage", "refusal", "stop_look",
                 "ack"] as const;
  for (const t of types) {
    es.addEventListener(t, (e) => {
      try {
        onEvent({ type: t, data: JSON.parse((e as MessageEvent).data) } as
                WsEvent);
      } catch { /* malformed event — skip */ }
    });
  }
  es.onopen = () => onState(true);
  es.onerror = () => onState(false);   // EventSource auto-reconnects
  return () => es.close();
}
