// SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
// SPDX-License-Identifier: GPL-3.0-or-later

// API shapes — mirrors carveout/web/server.py (run_journal.json semantics
// ARE the gate-state model; the API serializes the journal, never a
// parallel state machine).

export const GATES = ["volume", "render", "vocabulary", "exemplars",
                      "verify_consent"] as const;
export type Gate = (typeof GATES)[number];

export interface GateRec {
  approved_at: string;
  state: "approved" | "stale";
  changed: string[];
  run: boolean | null;
  neg_ceiling_ack: { value: number; limit: number; outlier: string } | null;
}

export interface JobState {
  kind: string;
  started: string;
  state: "running" | "done" | "failed";
}

export interface Journal {
  gates: Record<Gate, GateRec | null>;
  active_gate: Gate | null;
  running: JobState | null;
  lock: LockInfo | null;
  /** blocks a halted job is waiting on (stop-and-look) —
   *  served with the snapshot so a reload can restore the dialog even
   *  after the SSE replay buffer evicted the original event */
  blocks?: { kind: string; id: string; [k: string]: unknown }[];
  scene: string;
  workdir: string;
}

export interface LockInfo {
  holder: string;
  pid: number;
  since: string;
  alive: boolean;
  mine: boolean;
}

export interface SettingsInfo {
  profiles_dir: string;         // where this server reads and writes profiles
  /** is a VLM backend usable, and if not, why — the LLM stages are local
   *  now, so this is "is the model on disk", never an API key */
  vlm_available: boolean;
  vlm_reason: string;
  /** the model in use: chosen at server start (`carveout web --vlm-dir`)
   *  or the config's default; status only, never a key */
  vlm_model: string;
  vlm_model_dir: string;
  vlm_size_gib: number;
  vlm_source: "config" | "--vlm-dir" | "scene";
  vlm_models: VlmModel[];
}

/** An installed vision model: a directory under models/ with a loadable
 *  checkpoint. `need_gib` is the measured 4-bit peak when the catalogue
 *  has one, else an estimate (and `how` says so). */
export interface VlmModel {
  dir: string; name: string; size_gib: number; need_gib: number;
  how: string; current: boolean; fits?: "yes" | "tight" | "no" | null;
}

/** The creation dialog's proposal: the card, the browser's copy of the
 *  scene, and which installed model fits beside it. */
export interface FitEstimate {
  gaussians: number | null; card_gib: number | null;
  browser_gib: number | null; reserve_gib: number;
  models: VlmModel[]; proposed: string | null;
}

export interface FsEntry { name: string; path: string; bytes?: number }

export interface FsListing {
  path: string;
  parent: string | null;
  home: string;
  repo: string;
  dirs: FsEntry[];
  files: FsEntry[];
}

/** A splat file under the documented data/scenes/<name>/ layout.
 *  used_by names the profile already pointing at it, so the picker can say
 *  "already in the library" instead of offering a duplicate. */
export interface ConventionScene {
  path: string;
  name: string;
  scene_dir: string;
  bytes: number;
  used_by: string | null;
}

/** One thing `DELETE /api/scenes/<name>` would remove. */
/** what deleting a scene removes: the profile and the workdir. The capture
 *  under data/scenes is never a target — Carveout neither writes nor
 *  deletes there */
export interface DeletionTarget {
  kind: "profile" | "workdir";
  path: string;
  exists: boolean;
  removable: boolean;
  bytes: number;
  note: string;
}

export interface DeletionPlan {
  name: string;
  targets: DeletionTarget[];
  lock: LockInfo | null;
  running: string | null;
}

export interface SceneEntry {
  name: string;
  scene: string;
  workdir: string;
  gaussians: number | null;
  instances: number | null;
  thumb: boolean;
  last_report: string | null;
  lock: LockInfo | null;
  journal: Journal | { error: string };
  /** the row could not be read (a profile or a journal that does not
   *  parse): the message, verbatim, in place of the gate chips */
  error?: string;
}

export interface FrameEntry {
  frame_idx: number;
  file: string;
  /** "manual", "near_field" (the furniture close-ups; both outside the
   *  pool selection and first), "initial", or "coverage_N" for a repair round */
  round: string;
  provenance: string;
  label?: string;
  c2w: number[][];
  fx?: number; fy?: number; cx?: number; cy?: number;
  width?: number; height?: number;
  coverage: number;
  sharpness: number;
  /** selection order and the share of the volume this view newly covered
   *  when picked (auto views of a ranked first round only) */
  rank?: number;
  gain?: number;
  /** the share of the volume this view sees on its own */
  seen?: number;
  median_depth: number;
  near_alpha: number;
}

export interface CamerasResp {
  intrinsics: { width: number; height: number; fx: number; fy: number;
                cx: number; cy: number };
  frames: FrameEntry[];
  discards: { round: string; reason: string; judged?: Judged | null;
              [k: string]: unknown }[];
  /** the factor this render ran at: recorded, or measured with the ruler
   *  (metres then read "≈") */
  scale: { scale_m_per_unit: number; source: "recorded" | "measured" };
  coverage: { frac_final: number; k: number } | null;
  views_kept: number;
  views_manual: number;
  /** manual views this render does not reflect: `new` (saved since the
   *  render), `edited` (pose moved since) or `deleted` (gone from the file,
   *  still drawn — `frame_idx` says where). All need a re-render; an
   *  edited one keeps its id, so the set difference alone cannot see it. */
  manual_pending: { id: string; label: string | null;
                    why?: "new" | "edited" | "deleted"; frame_idx?: number }[];
  /** frames cameras.json records but that are gone from disk — the contact
   *  sheet would otherwise show tiles for views that no longer exist */
  frames_missing: number;
  /** what "auto" resolved to on this render: per knob the scene-unit value
   *  under its key and the metres under `<key>_m` (null without a factor);
   *  null when no knob was "auto" */
  auto_quality?: Record<string, number | null> | null;
  /** what "healthy" means for this render, from render.health in the config
   *  — the panel shows these numbers rather than repeating the literals */
  health: RenderHealth | null;
  /** a re-render held back because it came out worse than the published one;
   *  null on the normal path */
  staged: {
    version: number; staged_at: string; reason: string;
    staged: RenderHealth; published: RenderHealth;
  } | null;
  /** the most recent render attempt was REFUSED (every view discarded) and
   *  this payload is the earlier render it left standing; null = this
   *  render is the latest attempt. `thresholds_changed` lists the quality
   *  knobs whose profile value differs from what this render ran under, in
   *  the profile's own numbers */
  refused: {
    refused_at: string | null; views_attempted: number | null;
    views_kept: number; rendered_at: string;
    thresholds_changed: Record<string, { rendered: unknown; now: unknown }>;
  } | null;
}

/** what one discarded view failed on: the judged stat, its value and the
 *  threshold in the profile's numbers; distance rows add the scene units */
export interface Judged {
  stat: string; threshold_key: string; sense: "above" | "below";
  distance: boolean; value: number | null; threshold: number | null;
  value_units?: number | null; threshold_units?: number | null;
}

export interface RenderHealth {
  ok: boolean;
  views_kept: number | null;
  coverage: number | null;
  min_views: number;
  min_coverage: number;
}

export interface VolumeBox { name: string; min: number[]; max: number[] }

/** What `POST /views/reset` would delete — read before the confirm dialog so
 *  the numbers it names come from where the deletion happens. */
export interface ResetPlan {
  views: number;
  frames: number;
  path_mode: string;
  restores_to: string;
  stage1: string;
}

/** The ruler's record (`scene.measured`): what the operator measured on
 *  the canvas — the length they typed (in metres, and in their unit), the
 *  distance it spanned in scene units, the label, and the two points so
 *  the line is redrawn on reopen. `scale_m_per_unit` = length_m / units. */
export interface Measured {
  length_m: number;
  units: number;
  label: string;
  unit: "m" | "cm" | "ft" | "in";
  typed: number;
  points: number[][];
}

/** The scene alignment: a small rotation on top of the up-axis
 *  convention — tilt about the two ground axes, yaw about the up axis,
 *  degrees — recorded in the profile; absent or null = the file's axes.
 *  `points` are the clicks of a by-hand levelling, file frame. */
export interface AlignmentBlock {
  tilt_deg: [number, number];
  yaw_deg: number;
  source?: "proposed" | "levelled" | string;
  /** the operator's clicks when levelled by hand: the three floor points
   *  (null when only squared to an edge), the two edge points or null */
  points?: { floor: number[][] | null; wall: number[][] | null } | null;
}

/** The proposal's read of the tilt (`analysis.alignment`): the angles
 *  that would level the file, how sure the read is, and which reading
 *  made it (a room's floor and walls, a site's ground map, a subject's
 *  support and axes). */
export interface AlignmentRead {
  reading: string;
  tilt_deg: [number, number];
  yaw_deg: number | null;
  tilt_total_deg: number;
  floor_share_as_is: number;
  floor_share_levelled: number;
  yaw_margin: number | null;
  off_level: boolean;
  confident_yaw: boolean;
  /** the walls (or the subject's axes) sit off square by a sure margin
   *  under the levelling recorded now */
  off_square?: boolean;
  residual_yaw_deg?: number | null;
  current?: { tilt_deg: [number, number]; yaw_deg: number } | null;
}

export interface CalibrationResp {
  scene: { floor: number | null; floor_auto: number | null;
           /** the scene alignment; absent on a server from before levelling */
           alignment?: AlignmentBlock | null;
           up_axis: string; floor_band_frac: number;
           /** the floor rule's histogram bin (m); read-only, the start
            *  pose's floor estimate uses it before a proposal exists */
           hist_bin_m: number | null;
           /** metres per scene unit; null = not yet recorded (the ruler at
            *  the volume gate records it; Propose refuses until then) */
           scale_m_per_unit: number | null;
           /** the ruler's record when the factor was measured; null after
            *  a declared or typed one */
           measured: Measured | null };
  /** quality_filter is server-driven: the thresholds the render gate may
   *  change are derived from the discard reasons, so the panel renders
   *  whatever arrives rather than naming them a second time here. */
  render: { path_mode: string; cameras_in_volume: boolean; focus_aim: boolean;
            min_pos_sep: number;
            /** the initial round's view budget, and the repair rounds that
             *  add to it (each with its own budget, round_views) */
            num_views: number;
            /** the first round renders this many times the budget and keeps
             *  the budget by coverage gain; 1 = first-come */
            candidate_factor: number;
            /** the path mode's camera eye height (m); the start pose stands
             *  there too. Read-only: not on the calibration whitelist. */
            eye_height: number | null;
            /** the quality thresholds that accept "auto" (resolved from the
             *  scene's width at render); the panel offers the act for these */
            quality_auto?: string[];
            coverage: { target_frac: number; max_extra_rounds: number;
                        round_views: number };
            /** a threshold in `quality_auto` may hold "auto" */
            quality_filter: Record<string, number | "auto" | null> };
  /** instancing: how labelled Gaussians become objects — `tracks` (each
   *  detection followed across views, grouped by identity; the default)
   *  or `connectivity` (3D adjacency, then splits and merges). */
  /** the scene's vision model; a verify decision parameter */
  vlm: { model_dir: string; model: string;
         source: "config" | "--vlm-dir" | "scene";
         /** a name outside the vocabulary may stand when it is a
          *  kind of the detected label — the operator's option */
         relabel_specific_ok: boolean;
         /** a small object's own render joins its crops; the cap in
          *  Gaussians is the profile's */
         verify_isolated: boolean; verify_isolated_max_gaussians: number };
  lift: { instance_voxel: number; instancing: "tracks" | "connectivity" };
}

/** The volume gate's scale read: the ONE factor the pipeline uses —
 *  recorded (declared or typed) or measured with the ruler — what the scene
 *  MEASURES, and (a scene filmed inside a space only) whether its height is
 *  plausible. `source: none` = nothing recorded yet: the note carries the
 *  refusal's words and the panel offers the ruler. A read, never a
 *  decision. */
export interface ScaleRead {
  scale: number | null;    // metres per scene unit the pipeline uses
  scale_m_per_unit: number | null;   // the same, as the manifests record it
  source: "recorded" | "measured" | "none";
  recorded: number | null; // the profile's number
  measured: Measured | null;  // the ruler's record, when measured
  height_m: number | null; // the vertical extent in metres
  plausible: boolean | null; // interior scenes only; null elsewhere
  band: number[] | null;
  note: string;
  extent_units: number[];
  extent_m: number[] | null;  // null until a factor exists
}

export interface VolumeResp {
  /** scope "whole_scene": the explicit act — no boxes, every Gaussian in
   *  scope, placement over the scene's robust bounds */
  volume: { boxes: VolumeBox[]; scope?: "whole_scene" } | null;
  frame: { up_axis: string; up_sign: number; floor: number;
           floor_auto?: number; floor_override?: number | null;
           /** the alignment the volume was made under (null = the file's axes) */
           alignment?: { tilt_deg: [number, number]; yaw_deg: number } | null } | null;
  scale: ScaleRead | null;
  density_sidecar: boolean;
}

/** One line of the geometry's (or the model's) opinion: `fits` true =
 *  agrees, false = flagged, null = nothing to judge by (the reason says
 *  why). Stated on the cards, never applied. */
export interface CheckLine { fits: boolean | null; reason: string }

/** `<workdir>/proposal.json` v2: what the two Check cards
 *  show. Written last by the propose job; null before any proposal. The
 *  analysis is in scene units; the panels restate it in metres at the
 *  scene's factor (every scene that reached a proposal has one). */
export interface ProposalResp {
  version: number;
  generated: string;
  scene_frame: { up_axis: string; up_sign: number; floor: number | null };
  analysis: {
    up_axis: number;
    extent_units: number[];
    span_units: number;
    floor_signed: number | null;
    footprint_units: number[];
    footprint_over_height: number;
    volume_footprint_frac: number;
    /** `not_scanned`: the density core ran directly (filmed around a
     *  subject / outdoors), so nothing was tested for enclosure */
    enclosure: { used: "enclosed" | "density_core"; reason: string | null };
    ground: { relief_p5_p95: number[]; standable_frac: number;
              populated_frac: number; cell_units: number; dims: number[];
              coarsened: number };
    scale: Record<string, unknown> | null;
    /** the tilt read; absent from a proposal written before it */
    alignment?: AlignmentRead | null;
  };
  /** the second opinion on the measured segment (measured scenes): the
   *  model's length and what it took A–B to span, or the error that
   *  stopped it (shown verbatim), or null (a declared factor asks nothing;
   *  or the read is switched off) */
  read: { length_m: number; what: string; reason: string; probes: string[];
          aimed: string; model: string; elapsed_s: number }
      | { error: string } | null;
  /** the geometry's opinion of the two answers and the model's of the
   *  measurement — never applied */
  check: {
    path_mode: CheckLine & { said: string };
    scale: CheckLine | null;      // inside a space only
    opinion: CheckLine | null;    // measured scenes only
    /** is the scene level?; absent from an older proposal */
    level?: CheckLine | null;
  };
  /** the two measured proposals that remain: tight focus, and the
   *  lowered distance thresholds on a small scene, in the profile's
   *  metres (what the render panel's fields hold) with the scene units
   *  they resolve to; empty when nothing is lower */
  proposal: {
    settings: { cameras_in_volume: boolean; focus_aim: boolean };
    thresholds: Record<string, { proposed: number; current: number;
                                 proposed_units: number }>;
    reasons: string[];
  };
  elapsed_s: number;
}

export interface DensitySidecar {
  ground_axes: number[];
  up_axis: number;
  up_sign: number;
  floor: number;
  origin: number[];
  voxel: number;
  shape: number[];
  density: number[][];
  height_hist: { edges: number[]; counts: number[]; footprint: string };
}

export interface ProbeConcept {
  concept: string; negative: boolean; n_det: number; n_frames: number;
  max: number; mean: number;
}

export interface NegativeCeiling {
  value: number; limit: number; violated: boolean;
  outlier_concept: string | null; outlier_frame: number | null;
}

export interface ProbeResults {
  concepts: ProbeConcept[];
  exemplar_frame_cap?: {   // VRAM knob for the exemplar video pass
    configured: number | null; profile?: string;
    used?: number; total?: number; peak_alloc_gb?: number;
  };
  negative_ceiling: NegativeCeiling | null;
  thresholds: { presence: number; exemplar: number;
                presence_overrides: Record<string, number>;
                exemplar_overrides: Record<string, number> };
  overlays: string[];
  distribution_png: boolean;
}

export interface VocabResp {
  prompts: string[]; negatives: string[];
  probed: [string[], string[]] | null;
  probed_current: boolean;
  /** the vision model's proposal, objects ordered by how many views
   *  proposed each (`view_counts`, the ordering's evidence) */
  llm_proposal: { objects: string[]; distractors?: string[];
                  view_counts?: Record<string, number> } | null;
  vlm_available: boolean;
  vlm_reason: string;
  /** the rates the list is priced with; null before a render */
  cost: { views: number; gaussians: number; s_per_prompt_view: number;
          measured: boolean; card_gb?: number; classes_fit?: number;
          per_class_mb?: number; class_pass_bytes?: number;
          class_pass_fixed?: number } | null;
}

export interface ExemplarCrop {
  frame_idx: number; box_xyxy: number[]; frame_sha256: string; note: string;
}
export interface ExemplarsResp {
  exemplars: { concept: string; crops: ExemplarCrop[] }[];
  threshold: number;
  overrides: Record<string, number>;
  /** the crops on disk are the ones the probe saw (true with no crops);
   *  Continue refuses otherwise */
  probed_current: boolean;
}

export interface InstanceRow {
  idx: number; label: string; verified_label: string | null;
  verdict: string | null; verify_rationale: string | null;
  conf: number; scale: "text" | "exemplar";
  position: { x: number; y: number; z: number };
  scale_xyz: { x: number; y: number; z: number };
  obb: { center: number[]; extents: number[];
         rotation_xyzw: number[] } | null;
  instance_id: number | null; gaussians: number | null;
  support_by_source: Record<string, number>;
  hold_reasons: string[]; held_original_verdict: string | null;
  /** the operator's own name for it; shown first, the detected
   *  label kept beside it */
  operator_label: string | null;
  /** the name the verifier proposed, applied or not */
  proposed_label: string | null;
  /** marked for verification: while any object is marked, Run
   *  verification judges the marked ones only */
  verify_requested: boolean;
}

export interface ReportResp {
  report: {
    gates: Record<string, { approved_at: string } | null>;
    instances: { count: number;
                 verification: Record<string, number | Record<string, number>
                                      | string> | null;
                 unverified_indices: number[];
                 held_indices?: number[]; held_association?: number;
                 held_association_indices?: number[];
                 held_by_reason?: Record<string, number> | null };
    dropped?: { tracks: Record<string, unknown>[];
                export_floor: Record<string, unknown>[] };
    factor_kills: Record<string, unknown>[];
    funnel: { rows: FunnelRow[]; flags: string[] };
    provenance: Record<string, string | null>;
    /** what produced the export (absent from older reports): the
     *  instancing algorithm from the stage-3 manifest and the one factor
     *  the run used from the stage-1 manifest */
    pipeline?: { instancing: string | null;
                 scale: { source: string; scale_m_per_unit: number;
                          measured: Measured | null } | null } | null;
    code: string; generated: string;
  } | null;
  markdown: string | null;
}
export interface FunnelRow {
  cls: string; logged: number; above: number; dedup: number;
  g_in: number; g_out: number; comps: number; floor: number;
  exported: number;
}

// SSE events
export type WsEvent =
  | { type: "journal"; data: Journal }
  | { type: "log"; data: { ts: string; source: string; level: string;
                           line: string } }
  | { type: "stage"; data: { stage: string; state: string;
                             started?: string } }
  | { type: "refusal"; data: { message: string; gate: string | null;
                               remedy: string | null } }
  // `id` identifies the blocking act each event belongs to: the ack must
  // name it, and the reducer clears the block it resolves.
  | { type: "stop_look"; data: { kind: string; id: string;
                                 kills: Record<string, any>[];
                                 overlays: string[] } }
  | { type: "ack"; data: { kind: string; id: string } };
