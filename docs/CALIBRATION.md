# Calibration guide

The measured evidence behind the defaults in `configs/default.yaml`, and how to
re-derive a threshold on a scene where the defaults do not hold. The defaults ship
calibrated: on the scenes they were set on (interiors from a 2 m nook to a workshop,
one outdoor site) a run needs none of this. Read it when a run refuses at a gate and
names a threshold, when the funnel table in the run report loses a class you can see,
or when a new capture rig or training pipeline changes what the renders look like.
Every number here was set empirically on splat renders (photo-domain defaults are
assumed NOT to transfer), and each section says what it was measured on, so it can
be re-checked rather than trusted. `docs/USAGE.md` is the operator's walkthrough;
this document is the reference behind its thresholds.

## Stage 1: view-quality filter (`configs/default.yaml [render.quality_filter]`)

Splat-specific degradations (floater fog, mush) are invisible to depth statistics
because gsplat's expected depth is alpha-weighted: it sees *through* low-opacity
junk. The two signals that work, and how to calibrate them:

1. Render once with permissive thresholds, build a contact sheet
   (`PIL` thumbnail grid over `work/<scene>/stage1/frames/`), and judge frames
   visually.
2. Every kept frame's stats are in `cameras.json`; discarded ones in
   `manifest.json`. Tabulate `near_alpha` and `sharpness` sorted descending,
   mark the visually-bad frames, and look for the separation gap.
3. Set thresholds inside the gap; **Re-render** on the render panel and re-review the sheet.

One calibration, worked (a lab/office interior, against a manual frame review): the
scene's fog required tightening two thresholds past the shipped defaults:

| threshold | shipped default | calibrated here | meaning / evidence |
|---|---|---|---|
| `near_field_depth` | 1.2 m | 1.2 m | horizon of the 2nd near-field alpha pass; fog walls sat 0.7–1.2 m out |
| `max_near_alpha` | 0.20 | 0.15 | flagged-garbage frames ≥ 0.152; crisp ≤ 0.115 |
| `min_sharpness` | 0.008 | 0.016 | flagged garbage ≤ 0.0157; crisp ≥ 0.0173 (Laplacian mean, grayscale) |
| `min_median_depth` | 1.0 m | 1.0 m | wall-stares and inside-geometry cameras |
| `min_coverage` | 0.55 | 0.55 | mostly-empty frames (alpha coverage) |

## Stage 1: coverage (`[render.coverage]`)

`manifest.json → coverage` reports the fraction of in-volume Gaussians
(opacity ≥ `min_opacity`) observed depth-tested in ≥ K kept frames, plus
`frac_by_k`. Stage 3 lifting depends on this number: an unobserved Gaussian
cannot be labeled. The first round renders `candidate_factor` × `num_views`
passing candidates and keeps `num_views` by greedy coverage gain (each pick
the view that newly covers the most of the volume given the picks before
it; sharpness breaks ties; `cameras.json` records `rank` and `gain` per
view and the contact sheet reads best first). `num_views` is the INITIAL round's budget only; each
gap-repair round may add up to `round_views` more, up to `max_extra_rounds`,
and the render panel exposes all four (`num_views`, `target_frac`,
`max_extra_rounds`, `round_views`). If `frac_final` is still well below
`target_frac` after the repair rounds, either the volume includes
unscanned space (trim it in `volume.json`) or the room needs more/lower
candidate positions.

## Scene frame: up axis and floor (`[scene]`)

Everything spatial is expressed in this frame, so it is the first thing to get
right and the cheapest thing to check: the volume gate draws the floor plane on
the 3D canvas *before* any render. A plane standing on its side means the frame
is wrong, and every camera placed from it will be too.

**Up axis is a convention, not a setting to tune.** Carveout renders `-Y` up and
the scene is expected to arrive that way; no splat container records an up axis,
so nothing downstream can recover it. Re-orient at ingest when a capture arrives
otherwise. `up_axis: auto` does not auto-detect: it proposes a direction from
the scene's proportions and refuses, because the proposal (smallest robust
extent = height) is right for rooms wider than they are tall and silently wrong
for a corridor. `render.path_mode` is the operator's answer to "how was it
filmed?" at creation (inside a space / around a subject / outdoors on the
ground); `auto` means no answer and a render under it refuses with one
sentence. Nothing classifies the scene: the Propose analysis CHECKS the
answer against the geometry (enclosure, the ground map, the footprint) and
states its opinion on both Check cards, never applying it.

**`floor_band_frac` (default 0.4) is the one that actually needs calibrating.**
The floor is the dominant density peak within the lowest `floor_band_frac` of
the height range. In a room whose work surfaces are near bench height, that band
reaches the bench and the bench wins the vote, a common failure. Drop it to
**0.2** there; it narrows the search below bench height without pinning a
number. `scene.floor` pins the signed height outright, and the volume gate's
draggable floor plane writes the same value; prefer either to living with a
floor a metre too high, which puts every camera eye-level with the ceiling.

**Levelling (`scene.alignment`, the read under `scene.alignment_read`).** The up
axis is the coarse convention; a capture that arrived tilted carries a small
rotation on top of it (two tilt angles about the ground axes and a yaw about
the up axis), recorded by the volume gate's Level acts (proposed or from three
floor clicks and two wall clicks), never applied on its own. The read runs in
the propose job on a 300k subsample: inside a space and around a subject, the
up direction along which the floor is the sharpest density contrast, searched
coarse-to-fine within `max_tilt_deg` (30); outdoors, a robust plane through
the ground map (a hillside is meant to slope; the read states it). A scene is
called off level when the residual tilt is at least `level_tol_deg` (1.0) and
the levelled contrast gains `share_gain_min` (1.5×); a yaw is proposed only
when the sharpest wall direction beats the runner-up by `yaw_margin_min`
(1.2×); around a subject the yaw follows the subject's own axes instead.
Geometry (the frame, the floor, the boxes, the placement, the OBBs) lives in
the levelled frame; the file is never rotated; cameras are taken back to the
file's axes for rendering, and every output stays in the file's coordinates
with the levelling recorded beside it.

## Lengths: one factor (`[scene]`, `carveout/units.py`)

Every length knob in the config is a metre value, divided by ONE factor at the entry of
each stage (`scene_units_cfg`; exact at 1.0). The factor is:

- **recorded**: `scene.scale_m_per_unit` is a number: the scene was declared metric or
  given a factor at creation or at the volume gate. The calibration bank (13 metric
  runs from a 2.2 m bench nook to a 56 m outdoor site on the same defaults) ran this
  way, and on it
  21 of the 28 length knobs never moved from their default (the seven that did moved at
  most 2.5x: `min_pos_sep` 0.5–1.0, `min_median_depth` 0.5–1.0, `near_field_depth`
  0.8–1.2, `ground.min_standoff` 1.0–2.5, `interior.min_sightline`, `focus_standoff`,
  `focus_heights`). The knobs encode the size of the objects being detected, not the
  size of the scene: no fraction of the scene's width, of the volume or of its height is
  constant across the bank (they spread 7x to 25x).
- **measured**: the ruler at the volume gate: two clicks on
  the splat on something whose real length the operator knows, the length typed in m /
  cm / ft / in; the factor is `typed length in metres / measured distance in units`,
  written to `scene.scale_m_per_unit` with the act under `scene.measured` (length_m,
  units, label, unit, the two points). Recorded in the frame, `density.json` and every
  manifest with `source: measured`; lift, export and verify read it from the stage-1
  manifest. The panels mark such metres "≈". Checked, never corrected: under *inside a
  space* the room's height against `plausible_height_m`, and on every measured scene
  the model's second opinion on the drawn segment (flagged past 2x). Two worked
  examples: a shelf span of 1.9667 units typed as 1.8 m → 0.9153, with the two distance
  thresholds proposed at 0.33 / 0.40 m; on an outdoor site a gravestone of 0.93 units
  typed as 1 m → 1.0755, read back as "about 1 m (height of a medium-sized gravestone)".
- No factor and nothing measured → the run REFUSES to the volume gate naming the ruler;
  no default factor exists anywhere.

### The pick: where a click on the splat lands (`webui/src/workbench/overlays/pick.ts`)

The ruler, the level act and the floor act all take their points from one pick, which
applies the renderer's own model to the click's ray: every splat the ray passes
contributes its alpha at its closest approach, and the point is the splat that takes
the accumulated opacity, front to back, past `PICK_MEDIAN_T` (0.5), the median depth
of what the pixel shows. A ray that never accumulates that much lands on its strongest
crossing instead and the ring turns amber with "thin"; a ray that crosses nothing
lands nowhere. The one knob that matters is `PICK_MEDIAN_T`: lower it to pick through
a translucent layer the model painted in front (a reflection on a glossy top), raise
it to sink into a surface. Each click writes its record to `__carveoutCanvas.lastPick`
(`range`, `opacity`, `hits`, `solid`); read it from the console when a point lands
wrong. The hover ring casts the same pick once the pointer has rested.

## Volume (`work/<name>/volume.json`)

**Propose** at the volume gate writes the proposal + `volume_review.png` (top-down
log-density with boxes). Adjust the boxes on the Volume panel against the plot: trim
them to the actually-scanned region; density holes inside a box depress the coverage
metric and attract useless gap-repair cameras.

Wall-mounted classes straddle a tightly-trimmed volume boundary; the lift scopes
labels to the volume plus `lift.scope_margin` (0.3 m) so they keep their gaussians.
If a wall object still comes out merged or displaced, widen that margin in the scene
profile rather than the box.

## Interiors: the collision slab and the ceiling (`[render.interior]`)

The free-space map counts opacity between `slab_lo` (0.3 m) and `slab_hi`
(2.2 m) over the floor: what a standing person collides with. A room
lower than that (a 2.1 m ceiling) would put its ceiling inside the slab
and count it as an obstacle over the whole floor, leaving a cell or two
free by the door. The path reads the ceiling as it reads the floor (the
density peak in the top of the height range, inside the volume) and stops
the slab `ceiling_margin` (0.2 m) under it, and never below eye height +
0.2 m: a ceiling is over a standing person's head, so a shelf top or a
mezzanine that wins the vote in a sparse-ceilinged room can only cost the
cap, never the head-height clutter the slab exists for; the stage log
states the room height and the slab top. A tall room never meets the cap; lower
`slab_hi` only for a room whose head-height clutter (hanging bikes,
low shelves) should not count.

The sightline rule aims each position across the room, so a desk, a counter or
a bench is occupancy every camera turns away from (on an office scene the
cameras a metre from the desk all looked at the whiteboard; the desktop was in
no auto view, and the operator captured it by hand, one view per desk, standing
back a little and looking down at the top). The near-field views do that per
PIECE of furniture, not per standing position: a position takes what is
nearest, and the main desk went without when the nearest position stood too
close to it and the next one faced another desk. Furniture below eye height is
what a standing person looks over: opacity in the slab below `near_field_top`
(1.6 m, eye height) with nothing at or above it, so a desk with its monitors
qualifies and a wall, a door or a tall cabinet never does (at 1.3 m the
boundary put the office's main desk, monitors and all, with the walls, and it
got no view). Each 8-connected piece of at least `near_field_min_cells` (3)
cells gets aim points along its length, its heaviest cell first and then every
`near_field_spacing` (0.6 m: a view a metre from its aim spans about 1.3 m at
the 65° field, and every stretch of a piece should sit near the centre of some
frame, not merely inside one; at 0.8 m a 1.2 m desk got a view at each end and
none at its middle), heavier pieces first and every piece's first view before
any piece's second; `near_field_views` (16) caps the total, `0` turns them off.
The camera stands on the closest free eye-height cell within
`near_field_standoff` (0.4–1.6 m, horizontal) whose line to the aim crosses no
wall and whose slant distance to the piece's edge, at the piece's top height,
clears the quality filter's `near_field_depth`: the fog pass must not read the
desk's own edge, and from 1.6 m up that horizon is cleared closer than its own
length, which is what makes these close-ups. The target is the aim cell at the
piece's mean height (a desk with monitors reads about 0.9 m), so the pitch is
the geometry's, 18–65° on the office, not a tier from `pitches_deg`. The near-
field poses go first in the candidate list, so the diversity prune drops a
pitch tier that duplicates one, never the reverse; and they are outside the
pool selection, as the manual views are: rendered under their own round
(`near_field`, "N" on the sheet), kept whatever `num_views`, never ranked by
coverage gain (a desktop is a sliver of a room's Gaussians, so they ranked last
and survived only when the budget had room for the tail), first on the sheet
after the manual views; the budget is spent on the sightline views alone.
Measured on the office (38 positions, 60 views asked, the scene's auto fog
horizon 0.90 m): 5 pieces, 16 views placed (the cap), 16 past the diversity
prune, 15 past the quality filter at median depths 1.1–1.9 m, among them the
main desk square on from the middle of its front (laptop and both monitors,
1.65 m) and each of its ends.

## Exteriors (`[render.ground]`, `lift.specific_over_generic`, `min_coverage`, the budgets)

The keys an open scene reaches that a room never does, calibrated on one outdoor site
(56 m across, 15 m tall with canopy, 7.9 M Gaussians):

- **`render.path_mode: ground`**: person-height cameras over a LOCAL ground map, the
  ground never assumed planar (the site slopes ~4 m across its core). The map is
  `scene_read.ground_map`, shared with the Propose analysis. Per cell of `ground.cell_size`
  (0.5 m): the ground is the `ground_pct` (5th) percentile of the heights of the
  Gaussians with opacity ≥ `ground.min_opacity` (0.3), in-volume when a volume exists (its
  height cap keeps canopy out), in cells with ≥ `min_cell_points` (20); holes are filled
  from neighbours `fill_iters` (2) times and the map median-smoothed. Obstacle mass is
  the opacity in the slab `slab_lo`..`slab_hi` (0.3–2.2 m) ABOVE local ground; a cell is
  standable when its ground is valid in the full 8-neighbourhood and inside a box
  footprint. The free-space threshold is DERIVED from the scene: the strictest of
  `free_mass_percentiles` (10, 25, 50, 75) of the positive per-cell mass that leaves
  enough standable cells after `clearance_cells` (2) of erosion (relaxed to 0 if it must);
  the manifest records `ground_free_threshold`, `ground_free_percentile`,
  `ground_clearance_cells`, `ground_free_cells`, `ground_relief_p5_p95`. Cameras stand
  at ground + `ground.eye_height` (1.6 m), yaw toward distance-discounted obstacle mass
  within `aim_range` (12 m; nearer than `aim_dist_floor` counts as at that distance),
  rejecting yaws with mass nearer than `min_standoff` (1.0 m; 2.5 m on that site's
  profile), `views_per_position` (4) yaws ≥ `min_yaw_sep_deg` (45°) apart, at
  `pitches_deg` (+5, −15). The site, metric: p10 = 2.96, 407 free cells, 60 of 61 views
  kept, coverage 0.777 in 19 s on the RTX 4090 (8.1 GB).
- **`lift.specific_over_generic`**: an ordered priority list for containment pairs
  between listed classes: on the site a `tree` mask absorbed the `gravestone`s
  standing under it, and no rename fixes a concept-level umbrella. `specific:
  [gravestone]`, `generic: [tree]` keeps the part and carves its pixels out of the
  whole's mask. Empty by default; the values live in the scene profile.
- **`render.quality_filter.min_coverage`** (0.55, the fraction of pixels with alpha)
  assumes something fills the frame above the horizon. Open sky is alpha ≈ 0, so a
  TREELESS exterior can lose honest views to it: lower it in the profile (0.35 was the
  site's documented intent; inert there, since the canopy and the far shell kept alpha
  coverage ≥ 0.83).
- **The grid budgets** (`render.grid_budget`). The grids are sized in metres over
  bounds in scene units, so the factor sets their cell count: the site measured on a
  gravestone (1.0755) is 328 × 301 = 99k footprint cells at 0.186 units per cell, under
  the 250k budget; a 10x wrong measurement would be 100x the cells. Under
  `volume_proposal.strategy: auto` an over-budget footprint falls back to the density
  core (one log line) rather than refusing; the interior and ground paths still refuse
  over it at the render gate. The filming answer picks the strategy: *inside a space*
  runs the enclosure scan first; *around a subject* and *outdoors on the ground* take
  the density core directly and scan nothing; the enclosure scan reads a wooded site
  as "enclosed" (trees and stones enclose a cell as walls do), which is why the
  operator's answer, not a scan, decides.

## Stage 3: mask containment dedup (`[lift]`)

`mask_dedup: true` resolves same-view cross-class containment before lifting:
mutual containment (min IoS ≥ `mask_dedup_mutual_ios`) = one object under two
names, keep the higher score; consistent asymmetric containment
(IoS ≥ `partof_ios` in ≥ `partof_min_views` views) = part-of, the contained
det is suppressed so the whole keeps its parts. Audit every decision in
`stage3/instance_audit.json` under `_mask_suppression`. To calibrate on a new
scene, print the containment table (pairs, IoS both directions, scores) and
check: real distinct objects that happen to sit inside another's loose mask
(e.g. an office chair mat under an office chair) must survive via their
uncontained detections elsewhere; verify their class's instance count after
the run.

## Stage 3: per-class background gamma (`[lift]`)

A class detected in few views accumulates total contribution T from every
view but in-mask evidence A only where detected; the global
`background_gamma` then rejects gaussians whose argmax IS that class. Audit
procedure (worked example, a workbench class): recompute A/T, select the
object's region, look at the distribution of A/(T−A) over gaussians whose argmax is
the class; set `background_gamma_overrides.<class>` just below the median
(workbench: median 0.41 → 0.30). Keep overrides in the scene profile.

## Stage 3: instancing by detection correspondence (`[lift]`)

`instancing: tracks` builds instances from cross-view detection
correspondence. Five thresholds, all calibrated on
a six-scene bank (two workshop interiors, A and B; a lab interior; a salon
interior; an outdoor site; a studio interior; RTX 4090) from the readouts
every tracks-mode lift writes to
`stage3/instancing_calibration.json`; re-run the same reading on a new scene
family before trusting the defaults. "Known-same" below means the cross-view
detection pairs of single-object classes (`expected_from_2d` 1, one instance
in both modes); "known-distinct" means same-view cannot-link pairs.

| threshold | default | reading that set it |
|---|---|---|
| `support_min_frac` | 0.5 | On truly disjoint same-view masks (IoS 0), a gaussian firmly in one mask (f ≥ 0.7) leaves f on the other of at most **0.298** (p50 0.11, p90 0.22, p99 0.29; workshop A, 1,659 pairs, floor-0.05 dump). 0.3 has no margin; 0.5 has 0.2, and two disjoint masks can then never share a support gaussian (f_p + f_q ≤ 1). Same-class rows lost between 0.3 and 0.5: 2.0–2.5 % per scene. |
| `cannot_link_ios_max` | 0.1 | Same-view same-class mask-pair IoS (by the smaller mask) is bimodal: 95.5 % ≤ 0.1, 1.6 % in (0.1, 0.5), 2.2 % ≥ 0.9 (workshop A, 1,865 pairs; the other scenes agree). The threshold sits in the empty middle; review the (0.1, 0.5) pairs by eye on a new scene. |
| `track_min_support_ios` | 0.3 | Cannot-link pairs' support IoS at floor 0.5: p99 0.064 / 0.037 / 0.009, known-same p10 0.71 / 0.35 / 0.25 (workshop A / workshop B / lab). It also drives the ambiguity test: at 0.2 the reviewed classes lose whole tracks to all-ambiguous drops (workshop B tape roll 3, books 1, tool 5); at 0.3 those drops vanish and the track counts of every reviewed class are unchanged; at 0.4 large objects start to fragment (books 27, panels 24). |
| `track_affinity_min` | 0.2 | Known-same cross-view affinity p10 is 0.54 / 0.16 / 0.07 / 0.87 (workshop A / workshop B / lab / salon) and cannot-link pairs never exceed 0.27 (p99 ≤ 0.065). Every single-object class stays one track at 0.2 in every scene; at 0.3 the workshop B cutting mat splits, at 0.4 the workbench and the tool pegboard. A cannot-link pair's own edge is always vetoed, so the floor guards against bridges through a third detection only together with the cannot-link veto. |
| `export.hold_unassigned_frac` | 0.1 | Per surviving track, the fraction of its support lost to ambiguity-created unassigned gaussians is 0 for 80 % of tracks; among the rest p50 0.04, p75 0.11, p90 0.17, p99 0.47 (three small scenes, 389 tracks). 0.1 holds 21 of them. |

Affinity is the cosine between two detections' in-mask fractions over the
gaussians they share, the design's own support measure, view-invariant for a
gaussian inside the mask. Weighting by the raw contribution instead let a large
or flat object's affinity be dominated by whichever gaussian faced the camera
(lab acoustic-panel tracks overlapping by 60–90 % of the smaller support reached
0.13–0.18 and never linked); the fraction separates the populations better on
every scene of the bank, and is the only definition.

**What the bank could not fix by threshold.** Large objects seen partly per
view (acoustic panels 8 expected → 18, trees, gravestones) stay over-split
under every setting because their cross-view supports overlap little
(support IoS 0.15–0.25), the "opposite views see disjoint surfaces" case, known and
open. Mirrors on a mirror-heavy interior lose their instances (7 → 2) because the
detection's evidence lies in reflected space outside the volume. Neither is a
threshold.

## Stage 4: OBB mode (`[export]`)

`obb_mode: auto` picks per instance: gravity-aligned (yaw-only) unless the
instance is strongly planar AND tilted, which gets full PCA. Tunables:
`obb_planar_ratio` (thin iff sqrt(smallest/next PCA eigenvalue) < ratio;
default 0.25) and `obb_planar_min_updot` (tilted iff |plane normal · up| >
this; default 0.25 ≈ 14° from vertical). Audit which instances chose what via
`extended.obb.mode` in interactions.json. A panel-like object that does NOT
trigger `pca` usually means the instance itself is not planar; inspect its
composition (merges, volume truncation) before touching the thresholds.

## Stage 2: presence threshold

Stage 2 logs ALL raw SAM 3 presence-gated scores to CSV so `presence_threshold`
is picked empirically per scene family. Probe workflow:

1. **Probe** at the vocabulary gate with the full vocabulary and the
   deliberately absent concepts as negatives; the probe runs at
   `log_threshold` (0.05) so everything is logged.
2. Read `stage2/score_summary.csv` (sorted by max score) and
   `score_distribution.png` (negatives in red).
3. Threshold in the gap between the negative-control ceiling and the weakest
   real class you care about. Per-class exceptions go in the scene config under
   `detect.presence_threshold_overrides`.

### Vocabulary design: the two cannibalization domains

Family/umbrella classes kill type-specific classes through two distinct
mechanisms; check for BOTH when composing a vocabulary:

1. **Score-domain**: synonym siblings split presence scores; visible at the
   PROBE as underscoring family members. Fix at the vocabulary review:
   collapse the family to one term.
2. **Mask-domain**: an umbrella concept masks WHOLE objects of many types, so
   part-of/mutual dedup suppresses the type classes, INVISIBLE at the probe
   (every class scores healthily); only the FUNNEL's dedup attribution
   reveals it. Signature: one class firing on every frame at very high raw
   counts while type classes die at mask dedup to it. Renaming does NOT help:
   the umbrella is concept-level in SAM 3. Remedy: the
   `lift.specific_over_generic` dedup policy (per class pair).

Heuristic: prefer type-specific nouns; treat a healthy-scoring,
every-frame, high-raw-count class as an umbrella suspect.

### Calibration health metric: the negative-to-strong gap

**gap = (lowest strong-class max score) − (highest negative-control score).**
A healthy probe shows a gap of roughly 0.25–0.5 with the threshold sitting
comfortably inside it. Recompute the gap on every new scene probe: if it
narrows substantially, the splat-render domain is degrading SAM 3's presence
calibration and per-class overrides (or render-quality work) are needed. In
practice the default threshold has transferred unchanged across several scene
families.

Lessons that recur when the gap is thin:

- **LLM-proposed distractors need a frame check** before they are trusted as
  negative controls; an "absent" concept can turn out present in the scene
  (promote it to a positive) or have an in-scene look-alike (swap the
  control). Re-shopping a control just to lower the ceiling is curve-fitting;
  a documented outlier can simply be excluded from the ceiling reading.
- **A class whose max score sits within ~0.01 of the threshold**, especially
  one resting on a single detection, deserves a crop check before shipping,
  even when it verifies clean.

### Close-quarters scenes: sharpness judges, the distance filters do not

In a ~2 m room every legal camera is close to geometry, so the distance
statistics (`median_depth`, `near_alpha`, `coverage`) overlap between a
camera correctly framing a bench 0.6 m away and one buried in a wall; no
threshold on them separates the two populations at that scale. `sharpness`
separates cleanly, because a camera inside geometry renders mush: a causal
relationship, not a coincidence.

Two consequences. **Scaling the distance thresholds is about not REJECTING
good work, never about catching bad**: at the fixed defaults they can
reject every view such a room permits. Set `min_median_depth` and
`near_field_depth` to `auto` to resolve them from the scene's own width
(that reproduces the defaults on a room-sized interior), and calibrate
`min_sharpness` as the knob that actually judges. And **check the path mode
before loosening any threshold**: an interior path confined to a tight
volume renders mush where an orbit ring or hand-captured close-ups of the
same subject are sharp; the verdict depends on where the cameras are, not
only on the thresholds.

### Capture-side colour grading is detection-safe

A visible temperature/contrast grade on the capture moved strong-class
presence scores by ≤ ±0.02 in a same-scene comparison: scores are robust to
grading where evidence is plentiful. Judge grading effects on strong classes
only; rare-class maxima are view-sampling noise and move regardless.

## Stage 2: exemplar prompts (`detect.exemplar_threshold`, `<workdir>/exemplars.json`)

For objects SAM 3 segments correctly but cannot NAME (a class stuck near-zero
text confidence whatever the phrase), back the concept with a visual
reference instead of another noun-phrase iteration:

1. **Draw crop(s)** at the Exemplars gate: page the stage-1 frames (manual
   views first), drag a box around ONE clean instance, save. Detection
   happens on the drawn frame (the tracker only propagates from there), so
   draw where the targets are actually visible; extra crops on other frames
   = more anchored views. The concept string is the exported label; the
   label must be correct, because it ships as-is.
2. **Re-probe** (the gate's act): only the exemplar pass runs (text results are kept and
   folded; the INVALIDATED line announces it). Exemplar rows land in
   scores.csv / score_summary.csv / score_distribution.png under
   `exemplar:<concept>`.
3. **Threshold**: exemplar-similarity scores live on their OWN scale; never
   compare them to `presence_threshold` or the negative-to-strong gap.
   Default 0.4. Calibrate exactly like the text probe: read the
   `exemplar:<concept>` distribution, crop-verify the detections around the
   candidate knee, then set `detect.exemplar_threshold` or a per-concept
   `exemplar_threshold_overrides` entry in the scene profile.
4. **Staleness**: a re-render changes the crop's source pixels; detect
   REFUSES and names the crop to re-draw (frame_sha256). Exemplar
   detections pass the full downstream funnel (mask dedup including the
   cross-source collapse, Stage 4.5 verification); no quality mechanism is
   bypassed.
