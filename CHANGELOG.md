# Changelog

Carveout was developed privately from July 2026 to its first public release.
The repository starts at that release and carries no history before it. The
first section below describes what ships. The second retraces how the product
got there, one short chapter per turning point: the dense record of
iterations, test runs and working notes behind them stays private.

## 0.2.0 - unreleased (the first public release)

Carveout finds and labels objects in trained 3D Gaussian Splatting scenes: it
renders synthetic views of a scene, segments them with SAM 3 Promptable Concept
Segmentation, lifts the 2D masks to per-Gaussian labels, and exports labelled 3D
object instances, each with an oriented box and its own `.ply`. This is the
first public release. What it contains:

- **One front-end, `carveout web`**: a localhost browser app over a sequencing
  core; every stage is a function the app calls, every act a button on the
  panel that owns it, and every refusal names the act that answers it.
- **Five review points** (volume, render, vocabulary, exemplars, verify)
  journaled in `run_journal.json` with content-hash staleness: a changed input
  re-opens the affected gate and everything after it; nothing is approved by
  inference, and there is no skip path.
- **Cameras**: automatic paths for a room (`interior`, with a view at the
  furniture beside each standing position), a subject (`orbit`) and
  open ground (`ground`), chosen by the operator's answer to "how was it
  filmed?" at creation; hand-captured viewpoints adjustable in 3D; a fully
  manual mode; a view-quality filter and a coverage metric, both calibratable
  per scene from the render panel.
- **Scale stated, not assumed**: one factor per scene, recorded at creation or
  measured with the ruler on the canvas; nothing is rescaled and nothing
  guesses a scene's size.
- **An optional vision model, run locally**: Qwen3.8-27B or Qwen3-VL-8B,
  chosen per scene, for a second opinion on a measured length, a vocabulary
  proposal, and label verification. No API, no key; nothing reaches the
  network at run time.
- **Two containers**: 3DGS `.ply` and PlayCanvas SOG v2 `.sog`; exports are
  `.ply` and open in any splat viewer. `-Y` up is a stated convention, not
  something inferred.
- **Two VRAM profiles** (`24gb` default, `32gb` opt-in); RTX 4090 and RTX 5090
  tested. Scene profiles live in `configs/scenes/`, one per scene, ignored by
  git; `carveout web --profiles-dir` points a server elsewhere.
- **A scene library** where creating and deleting a scene are confirmed acts,
  and a run report that closes each run with the per-class funnel from raw
  detections to exported objects.

## Before the release: how Carveout got here

Each chapter is a turning point, placed in the month it landed in the private
tree; the bullets under it are what followed, in a line each.

### July 2026

#### The pipeline runs end to end

Render, segment, lift, export, on the first interior: one conda environment
for both cards, the weights confined to the repository.

- Interior and orbit camera paths, a view-quality filter, a coverage metric
- Thresholds calibrated from logged raw scores, not assumed
- A vocabulary proposed by a language model behind a swappable backend
- A 2D box editor for the volume of interest; a standalone Spark viewer
- A first label-verification stage: confirm, relabel or reject per object

#### Instances are formed in 3D

Pseudo-video tracking across views is retired; detections are associated in
world space instead.

- A salon full of mirrors brings the scope margin: reflections outside the volume are dropped
- Mask dedup across classes; evidence-guided splits and merges
- Unattended post-gate runs with a per-class funnel report

#### The first exterior

A ground path mode for open ground: a local ground map, free space found
automatically, the volume proposed from the density core.

- Sloped ground and tree cover shape the calibration
- Every step between two review points must drive itself; a hand-derived setting is a product gap
- Specific-over-generic dedup with a priority order; relabels held to the vocabulary
- Viewer: a focus volume, label pills, a screen or world label scale, click-to-fly

#### Hand-captured viewpoints and exemplar prompting

Camera poses captured in the viewer join the render, tagged by provenance;
reference crops drawn at a gate prompt SAM 3 for what a phrase cannot name.

- Every downstream cache invalidates when the viewpoints change
- The area-factor guard: a mask far larger than its class's usual size is dropped, and the run pauses
- A gated terminal driver sequencing the whole run (later replaced by the browser app)

#### The browser app

A sequencing core extracted from the driver, a standard-library server with
jobs and server-sent events, a React workbench over a shared splat canvas.

- A single-writer lock per working directory; JSON sidecars beside every artifact
- Approved gates open for inspection; reopening is an explicit act
- A floor-plane override at the volume gate; a Stop button with cooperative cancellation
- RTX 4090 bring-up: system CUDA 12.8, the libstdc++ preload, the exemplar pass capped for 24 GB

#### First release preparation

README, usage guide, GPL-3.0-or-later, citation metadata, SPDX headers; the
viewer bound to localhost and the server's origin guarded.

- A configurable profiles directory and a settings surface
- An acknowledgment must name the block it answers
- A rejected re-render no longer consumes the previous one

### August 2026

#### The canvas gets real handles

A manipulator for the volume boxes, a view dial for standing on a scene's
faces, a focus volume that turns.

- A re-render that comes out worse is staged, not published
- The viewer's typeface served locally
- A pre-push guard and a curated import build the public repository from the private one

#### `.sog` beside `.ply`; the up axis is a convention

PlayCanvas SOG bundles are read like `.ply`. Automatic up-axis detection
proposes and refuses instead of deciding.

- Scene deletion and a file picker in the library

#### The vision model runs locally

The remote API backend is removed entirely; a model on the card, loaded
compressed, proposes the vocabulary per view and verifies labels. Nothing
reaches the network at run time.

- The browser app becomes the only front-end: the terminal driver, frame exclusion and the adopt path go
- Scene scale as a stated convention, proposed at the volume gate (first version)
- A fully manual camera mode with a camera gizmo at the render gate
- The path mode's automatic choice proposes and refuses, like the up axis
- Refusals route to the gate that owns them; the vocabulary draft survives a restart
- Vision-model vocabulary terms adopt with a click
- An internal restructuring: one refusal type, one owner for gate state, one mutation path

### September 2026

#### The workbench restyled; the standalone viewer removed

Semantic colour tokens with dark, light and system appearance, a named rail,
a grouped toolbar. The workbench is now the results viewer too.

- The scene library as a plain table
- A floor grid and a floor plane as separate overlays; splats write depth for proper occlusion
- A camera overlay palette: blue automatic, green manual

#### Instances from 2D detection correspondence

Detections are followed across views into tracks, calibrated on five scenes,
as an alternative to 3D adjacency.

- A verdict that confirms a mismatched crop is held, not accepted
- Every stage cache carries its upstream fingerprint and its decision parameters
- Every profile is written by Carveout

#### Scale applied everywhere

Every metre-valued knob is divided by the scene's factor at use; the viewer,
the export and the grid follow it.

- The scale is optional through the app; whole-scene scope is an explicit act
- Stop works at the volume gate; scene-sized grids refuse over budget before scanning
- A readout card for the selected object: size, box, position, in metres or units
- The verifier's verdict on the card and a mark on the pill it doubted
- Clear detections as an explicit act; the rail follows the journal after an approval

#### The command line is `carveout web`

The batch runner, the matplotlib editors and the COLMAP loader go; every
refusal names a web act; scene files live in the working directory.

- The A/B experiment branches leave the stages
- One profile directory, pointed elsewhere with `--profiles-dir`
- An instancing selector on the panel: tracks or connectivity

#### Ask the operator, do not guess the scene

Creation asks two questions: is the capture metric, and how was it filmed.
The ruler measures one thing of known length on the splat.

- The model's second opinion is posed on the measured segment
- Check cards at the volume and render gates state what the scene read found against the answers
- A stage or an approval waits for every gate before it

#### The run says what it is doing

The running job named everywhere with elapsed time and a log; each gate lists
its steps with the current one marked.

- The vision model chosen per scene, changeable on its panels; every model loads compressed
- The stop-and-look dialog says in plain words what was dropped; the class name is the acknowledgment
- A scene opens standing in it, or framed for an object; the volume gate previews the box it edits
- The first render round renders a pool and keeps the views that together see the most of the volume
- Every panel, banner and refusal speaks to a user, not to the author

#### Levelling

A tilted capture is levelled by a rotation recorded in the profile, the file
never rotated: as proposed with a preview, or by hand from three floor clicks
and two wall clicks.

- Boxes proposed under another levelling refuse until Propose fresh
- The fifth review point becomes Objects
- The operator's own viewpoints can be rendered before any automatic render
- The floor set from one click

#### A click lands on the surface

A click on the splat takes the first surface along the ray, and a hover ring
shows where before the click.

- Undo last on the level act; Square to an edge
- A cue bar marks the gate's current step
- Approval refuses when the views changed since the render; no gate without its probe
- Mask dedup seven times faster; box and area for every mask in one GPU reduction

#### Naming

The operator names an object beside the detected and verified labels; the
verifier's proposal is taken in one click.

- Name-first verification: the model names the object before it judges the label
- Two options, off by default: a more specific name may stand; a small object's own render joins its views
- The previous run's verdicts kept beside the new ones

#### Sized to the card, and the audit before the release

The New scene dialog states the Gaussian count and the vocabulary the lift
fits; the Vocabulary panel prices the list live; the lift refuses before a
pass that would not fit.

- Verification checkpoints every verdict, resumes after a stop, and can judge the marked objects only
- The probe is current by content, not by session
- State files written whole or not at all; the cascade demotes before it writes
- The verify answer stands until the gate is reopened; the render cache names the volume
- Cancel reaches a halted run; a delete never takes the repository, the captures or the weights
- The server starts without torch and refuses a missing dependency with the remedy

#### Near-field views

The interior path adds views at the furniture below eye height beside each
standing position, outside the pool selection like the manual views.

- An auto button beside the two distance thresholds
- The Capture view section says what is saved so far
