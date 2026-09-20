# Carveout web front-end

The browser app served by `carveout web`: a React + TypeScript bundle built with Vite.
The Python server serves the built files statically from `webui/dist/` and exposes the
pipeline over REST and a server-sent-event stream. Node is a **build-time dependency
only**: once `dist/` exists, `carveout web` runs on the Python environment alone.

`dist/` is not committed, so a fresh clone must build it once:

```bash
npm --prefix webui ci && npm --prefix webui run build
```

If you start `carveout web` without building, every page request returns a 503 naming that
command.

## Development

```bash
carveout web                      # backend on 127.0.0.1:8090, in one terminal
npm --prefix webui run dev        # Vite dev server, in another
```

Open the URL Vite prints. `/api` is proxied to `127.0.0.1:8090` in dev only
(`vite.config.ts`), so the app talks to the real pipeline with hot reload in front of it.

`npm run build` runs `tsc -b && vite build`; type errors fail the build. The toolchain is
pinned in `package.json` (React 18.3, Vite 5.4, TypeScript 5.6, Tailwind 3.4); Node ≥ 20
is required (`engines`), and Vite 5 is the binding constraint, so use a currently
supported Node LTS.

## Layout

```
src/
├── main.tsx           entry
├── App.tsx            two screens: / = library, /scene/<name> = workbench
├── api.ts             REST + SSE client; ApiError carries the server's message and gate
├── store.tsx          per-scene run state, fed by the event stream
├── journal.ts         gate-state selectors over the run journal
├── types.ts           shared types, incl. the gate list
├── ui.tsx             shared primitives (buttons, badges, sliders, confirms)
├── labels.ts          gate display names + the shared status wording
├── theme.ts           dark / light / system appearance, persisted in localStorage
├── theme.css          Tailwind layers + the two palettes behind the design tokens
├── library/           scene list, thumbnails, lock state, new-scene form
└── workbench/         the run screen
    ├── Workbench.tsx  one persistent 3D canvas between the rail and the inspector
    ├── SplatCanvas.tsx    the canvas, built on the shared viewer core
    ├── GateRail.tsx   the review-point rail down the left edge
    ├── StageStrip.tsx / ReportDrawer.tsx / StopLook.tsx
    ├── overlays/      canvas gizmos: volume boxes, frusta, floor plane, crop markers
    └── panels/        one panel per review point (PipelinePanel is Verify's; it
                       holds the unattended run) plus SettingsPanel for the rail's
                       Viewer node
```

## The shared viewer core

The splat rendering, label engine, highlighting, focus volume, view capture and settings
defaults are **not** reimplemented here. They are the plain-JS modules in `../viewer/lib/`
(the canvas core, kept as plain JS with no build step of its own), imported through the
`@viewer` alias configured in `vite.config.ts`. They are consumed untyped on purpose
(`src/viewer-lib.d.ts`). Do not fork them into TypeScript: one copy of the canvas code
means one place where a rendering rule changes.

`three` and `@sparkjsdev/spark` are pinned here and deduped, so the bare imports in
`../viewer/lib/` and the workbench's own resolve to one copy of each.

## Conventions

- **The server owns the run.** The journal, the approvals and their invalidation live in
  the Python core; this app is a spectator over the event stream and a caller of endpoints.
  Do not reimplement approval rules, staleness or sequencing in the client.
- **Refusals are shown verbatim.** When the server refuses, `ApiError` carries its message,
  the review point it belongs to, and any remedy. Surface all three; never swallow one or
  replace it with a generic failure message.
- **Re-attach is server replay.** Reconnecting the event stream resumes from the last event
  id; client-side persistence of run state is not the mechanism.
- **Neutral, dense, no decoration.** Two appearances (dark, light, or follow the OS)
  over one set of semantic tokens in `theme.css`; colour carries state, not mood
  (green done, amber warning, red error/destructive), emphasis is a filled neutral or
  weight. The 3D canvas, its label pills and tags, and the overlay/axis colours are
  scene data and stay outside the theme. See the notes at the top of `theme.css`.
