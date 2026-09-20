# Using Carveout

The complete walkthrough: what a run looks like from a trained `.ply` to labelled 3D
objects, what each step decides, and what lands on disk. Install first; see the
[README](../README.md).

Carveout runs interactively. It stops at five review points and asks you to approve what
it has done before it goes further, because every later stage inherits those decisions:
the region it looks at, the views it renders, the words it searches for, the visual
references it tracks, and whether the labels get verified. There is no way to skip them,
and a run resumes from the journal it keeps, so stopping and coming back later is normal.

---

## 1. Before you start

**Input.** One trained 3D Gaussian Splatting scene, in either container Carveout reads:

- **`.ply`** in the standard 3DGS layout: per-Gaussian position, opacity, scale,
  rotation and spherical-harmonic colour. The required fields are `x`, `y`, `z`,
  `opacity`, `scale_0..2`, `rot_0..3` and `f_dc_0..2`; the higher-order `f_rest_*` bands
  are optional. On load Carveout names any required field it cannot find.
- **`.sog`**, a PlayCanvas SOG v2 bundle, a compressed form of the same scene,
  written by `splat-transform my-scene.ply my-scene.sog`. Version 2 bundles only:
  the unzipped v1 directory form is refused by name rather than half-read.

Carveout does not train scenes, and reads no container other than these two.

**Orientation.** Carveout renders `-Y` up. No splat container records which way is
up, and the reconstruction that produced the scene chose its world frame
arbitrarily, so the information is not in the file to be recovered. Orient the
scene at ingest, in whatever tool wrote it. If a scene genuinely
cannot be re-oriented, pin `scene.up_axis` (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`) in its
profile. Setting it to `auto` does not auto-detect: Carveout proposes a direction from
the scene's proportions and then refuses to run, so the axis is always something you
chose.

Either way the Workbench is where you find out: from the moment a scene opens it draws
an infinite reference grid lying on the floor it is reading (the floor it estimates from
the scene's density before anything is proposed, the proposal's floor once one exists),
oriented by the up axis. A grid standing on its side means the scene is not oriented the
way Carveout is reading it; a grid cutting through a bench means the floor was elected
on the wrong surface, and the floor height in the Volume gate's calibration is the fix.

The two are **not interchangeable within a run**: they do not share a Gaussian
ordering, so per-Gaussian instance ids belong to whichever file produced them.
Repointing a scene at the other file changes the scene profile, which demotes every
gate and makes you re-render, deliberately, rather than silently mixing two orderings.

**Layout.** One directory per scene, the scene file inside it:

```
data/scenes/my-scene/my-scene.ply     # or my-scene.sog
```

The directory name (lowercased) becomes the scene's name, and Carveout writes a settings
file for it at `configs/scenes/<name>.yaml`. `data/` is git-ignored, so scenes never enter
version control.

The scene folder holds the capture and nothing else: Carveout never writes there and
never deletes there. Everything Carveout learns *about the scene* and everything it
*computes* go to the working directory, by default `work/<scene>/`. What it learns is
the region of interest (`volume.json`), hand-captured viewpoints (`manual_views.json`),
visual references (`exemplars.json`) and viewer preferences (`viewer_settings.json`).
To start a scene over, delete it from the library and create it again from the capture.

**Three questions when you create a scene.** The pipeline needs two things the file
does not carry, and you, who were there, can supply both; the third is which vision
model serves this scene (below). *Is this capture metric?* Press **Metric** for a
capture from a metric-posed device (LiDAR, a phone rig, a metric scan), type the
metres per unit if you know the number, or answer **No, I will measure one thing on
the canvas**: at the volume gate you click two points on something whose real length
you know (a door, a floor tile, a counter, a wall you paced) and type that length in
your unit: the ruler. *How was it filmed?* Three tiles, always asked: **inside a space** (eye-height cameras inside the walls), **around
a subject** (a ring of cameras around the volume), **outdoors on the ground**
(person-height cameras over the local ground). Carveout never classifies a scene; it
records your two answers in the profile, checks them against the geometry at the volume
gate, and says what it found. Both can be changed later (the ruler at the volume gate;
the filming answer on the render panel).

**Optional.** A local vision-language model, downloaded alongside the SAM 3 checkpoints
(README step 7), serves three uses: a second opinion on your measurement at the volume
gate (on every measured scene, when the model is installed), the vocabulary proposal
and label verification (each only when you press for it). It runs on this machine:
there is no API key and nothing is sent anywhere. Without it, everything else runs.

---

## 2. How you drive it

The **browser app** is the whole command line: `carveout web` serves
http://127.0.0.1:8090, and every act (proposing, rendering, probing, running the
pipeline, verifying) is a button on the panel that owns it. There are no per-stage
commands and no batch runner. The region of interest, the camera poses and the finished
objects share one 3D canvas.

```bash
conda activate carveout
carveout web                 # 127.0.0.1:8090; --port to change
```

It binds to localhost only, with no authentication and no telemetry. A 503 naming a
build command means the front-end bundle was never built: run
`npm --prefix webui ci && npm --prefix webui run build` and reload.

**Review points come in order.** A stage runs, and a point is approved, only once every
point before it is approved: the render waits for the volume, the probe for the render,
the pipeline for all four. The buttons say what they wait for, and the panels stay open
so you can look ahead. Calibration changes are never held back: they re-open review
points, they run nothing.

**One driver at a time.** The app holds `work/<scene>/run_lock.json` for every scene it
drives; a second instance opens the scene read-only, under a banner saying it is open in
another Carveout and since when, and refuses every act that would write, naming the
holder. If a driver died and left its lock, releasing it is an explicit act, refused
while the holder is alive.

**The Scene library** is where you land. **+ New scene** lists every `.ply` and `.sog`
under `data/scenes/<name>/` with sizes, marks the ones a profile already uses, and lets
you **Browse…** or type a path for a capture kept elsewhere. Once a file is chosen the
dialog states its Gaussian count and how many classes of vocabulary the lift fits for it
on this card, in a warning tone when the file is large for the card, better known here
than an hour into the run. Then the three questions of §1, and Create.

Each row carries **Delete**: a dialog lists what would go (the profile and the workdir,
with sizes) and you type the scene's name to arm it. The capture itself is never a
target, nothing outside the repository is ever removed, and deleting is refused while a
stage runs or another driver holds the lock.

**The Workbench** opens a scene: a 3D canvas between a rail of review points on the left
and the open point's panel on the right. The rail reads **Volume**, **Render**,
**Vocabulary**, **Exemplars**, **Objects** (which also holds the unattended detect →
lift → export run and its live log in the strip below the view), then **Viewer** and
**Report**. The mark beside each name says where it stands: a green check for approved,
an amber warning for stale, a spinner while it runs. Each panel's acts sit in a footer
pinned to its bottom. An approved point opens read-only with a **Reopen gate…** button;
reopening is the explicit act that demotes it and everything after it. While the scene
file loads the header says how much has arrived; a file that cannot be loaded is said
there in red. `Tab` hides the panels when you just want to look. **Dark / Light /
System** in the library's header sets the app's appearance and nothing else.

---

## 3. The run, step by step

### Step 1: Volume of interest (Volume)

**What it does.** Carveout proposes one or more axis-aligned boxes around the part of the
scene worth labelling, from the scene's own geometry: it finds the up axis and the floor
from the density, then the wall-enclosed interior or, for open scenes, a density core.
Every later stage depends on the box: cameras are placed with respect to it, coverage is
measured inside it, and only Gaussians inside it (plus a 0.3 m margin, so wall-mounted
objects survive) can become labelled objects. The proposal is a starting point; you
correct it and approve.

**What you do.** Open **Volume**. The boxes are drawn on the canvas with the detected floor
as a translucent plane. Drag the boxes to fit, add or remove boxes, and approve when the
region covers what you want labelled and little else.

The floor plane is where the cameras will stand, so check it first. A cluttered bench top
can win the vote over the real floor: press **Set from a point** in the Floor row, click
the real floor in the 3D view, and the grid and the plane move there; then **Propose
fresh** to re-propose on it. Typing the height or fixing the up axis does the same. Any of
these re-opens this step and everything after it.

**Scale: measure one thing you know.** Every placement length is a metre setting
(cameras stand 1.6 m above the floor) and nothing in the file says what a unit is, so
Propose refuses until the scene has a scale. The **Scale** block offers three ways to
record it:

- **Measure** arms the ruler. Click two points on something whose real length you know
  (a door, a floor tile, a car, a wall you paced), type the length in your unit (m, cm, ft
  or in), a label if you like, and press **Record**. Esc cancels; a third click starts
  over. The block then reads "1 unit ≈ 0.92 m" and "you measured a door: 2 m over 2.18 units".
- **Metric** declares the capture already in metres.
- **Metres per unit** takes a number you already know.

A declared or typed factor is exact and the panels say "m"; a measured one is marked "≈"
wherever metres appear. Nothing is rescaled: the number says what the units mean, and
every metre setting is divided by it when a stage starts. The block also states the
scene's measures, and for a scene filmed inside a space says when its height is
implausible for an interior (1.8–6 m), the sign of a wrong measurement. Changing the
scale re-opens this step and everything after it.

**Level: a capture that is not upright.** Some captures arrive tilted: the floor leans
against the file's axes and no straight box can hug the room. Propose reads the tilt and
says so on the Check card and in the steps list; the **Level** row then reads "the file's
axes · off by 12.7°". Three ways to level:

- **Level as proposed** adopts the angles Carveout read. Switch **Preview** on first to
  see the grid and the floor plane where they would lie.
- **Level by hand** lists its two steps: **Floor**, three clicks on the floor far apart
  (or on any flat surface parallel to it), then **Edge**, optional, two clicks along one
  straight edge to square the box to it, or Done. Each step shows its count and wears
  its colour on the canvas: red-orange for the floor, teal for the edge. **Undo last**
  or Backspace drops the last point.
- **Square to an edge** takes the two edge clicks alone and keeps the tilt, for when only
  the turn is off.

**Reset to the file's axes** undoes any of them. Levelling sets the tilt and the turn
only; the floor height stays the Floor row's, so set it from a point if the plane sits
wrong afterwards. Then press **Propose fresh**: the box hugs the levelled room and every
camera stands level. Outdoors, a hillside is meant to slope: the read states the slope
and leaves the choice to you. Nothing is applied on its own.

**Where a click lands.** A click lands on the first surface along your line of sight,
not on a stray splat in front of it, and you can see it before you click: rest the
pointer and a ring marks the spot: white on a surface, amber with "thin" on a stray
splat (move a little), none where there is nothing to hit.

**Whole scene** is an explicit act, not the absence of a volume: cameras then use the
scene's bounds, coverage is measured over every opaque Gaussian, and nothing is scoped
out of the lift or the export. Propose fresh brings boxes back.

**The Check card** (the boxes tab, above the box list) is what the last proposal read
about your two answers, stated and never applied. Your filming answer picks how the volume
is proposed: *inside a space* scans for walls around a floor and falls back to the
density core; *around a subject* and *outdoors* take the density core directly. The card
shows the measures in metres, the region taken, the ground relief and how much of it a
person could stand on, then three checks:

- **filmed**: the geometry's opinion of your answer: "you said inside a space; an
  enclosed region was found", or flagged: "nothing enclosed was found; if this is
  outdoors, pick outdoors on the render panel", "an enclosed space 2.3 m across, too
  small to walk: around a subject may film it better".
- **height**: under *inside a space*, the room's height at your scale: "the room reads
  2.5 m tall", or flagged "27.7 m tall: check the two points".
- **measured**: the model's second opinion, on a measured scene. Propose renders a few
  views with your segment drawn in red and asks the local vision model how long it
  looks, judging by the objects around it (about 15 s, 25 s the first time, on an
  RTX 4090 in September 2026): "the model reads A–B as about 1 m, which agrees with
  the 1 m you typed", or flagged past a 2x
  disagreement. A declared or typed factor asks nothing; without the model's weights
  the card shows the download command and everything else proceeds;
  `render.propose.vlm_read: false` turns it off.

A flagged line is a reason to look again (measure again, or change the filming answer
on the render panel), never a change Carveout makes on its own.

**A scene far off scale** is caught before any work is spent on it. The proposal's
grids are sized in metres, so on a scene 100x too large the enclosure scan would be a
million times the work. If the footprint exceeds `render.grid_budget.footprint_cells`,
the default `render.volume_proposal.strategy: auto` skips the scan and proposes the
`density_core` instead, which needs none. This is logged, and the volume panel says "the scene
was too large to scan for an enclosure". An explicit `enclosed` strategy refuses before
scanning, states the scene's dimensions in units and in metres, and names the two
causes: a wrong measurement (measure again) or a genuinely large metric scene (raise the
budget in the scene profile, or pick a strategy that needs no scan). The same budget
guards the render's camera placement.

**Stop** ends a running proposal; nothing is written on a stop.

**Units in the export.** `interactions.json` is in scene units and says so under
`extended.units`, with the factor and its source; a consumer that wants metres
multiplies by `scale_m_per_unit`.

**On disk:** `<workdir>/volume.json` (the boxes; on a levelled scene the levelling
recorded beside them), `volume_review.png` (top-down density plot with the boxes),
`density.json`, `proposal.json` (the analysis behind the Check cards), `probe/` (the
views the model read; measured scenes only).

### Step 2: Rendered views (Render)

**What it does.** Carveout places synthetic cameras and rasterises the scene with gsplat.
Where the cameras go follows how you said the scene was filmed; the **filmed** selector
on the panel names the four paths in the creation dialog's words, with the profile's
name beside each:

- *room* (`interior`): eye-height cameras walking the free space inside the volume,
  aimed along sightlines, plus close-ups of the furniture below eye height (a desk, a
  counter, a bench), one per stretch of it, each standing back just far enough; the
  close-ups are kept whatever the view budget, like your own viewpoints, and lead the
  contact sheet marked "N";
- *orbit*: a ring of cameras around the volume, or around the whole scene without one;
- *ground*: person-height cameras over a local ground map, for outdoor ground that is
  not flat;
- *manual*: only the viewpoints you placed yourself (below).

Carveout never chooses the path; a profile with no answer refuses and names the panel.
It over-samples candidates, drops near-duplicate poses, and discards degenerate frames:
views that stare at a wall, sit inside geometry, are near-uniform, blurry, or fogged by
floaters near the camera. Then it measures **coverage**, the fraction of in-volume
Gaussians seen in at least two kept views, depth-tested, and if that is below target
(90 %) it aims extra cameras at the largest uncovered clusters, up to four more rounds.

**What you do.** Open **Render**. The **Check and proposal** card restates your filming
answer with the geometry's opinion under it, flagged when they disagree ("nothing
enclosed was found; if this is outdoors, pick outdoors"), and holds two settings Propose
measured for you: tight focus when the volume is under a quarter of the scene's
footprint (a subject inside a room), and, on a small scene, the two distance thresholds
lowered to a fraction of the scene's width, each with its reason. **Adopt** writes them
in one act and re-opens this step. The card never proposes a path; change the selector
yourself if the check convinces you.

Start the render. When it finishes you get the kept frames, the discard reasons and the
coverage figure. Review the frames: this is what the detector will see, and nothing
outside it can be found. Then approve, or first add views of your own.

**Your own viewpoints.**

- **Capture a viewpoint.** Fly the canvas to something the sampler framed badly (a
  tight corner, a small object) and save the view. Captured views render in addition to
  the automatic ones, at the captured field of view, and bypass the quality filter: your
  judgement wins, and the log says what the filter would have judged.
- **Adjust a viewpoint.** Click a camera in the 3D view and drag its handles to move or
  turn it; the toggle beside the frame list switches between the two. Only your own
  viewpoints are adjustable; automatic ones are regenerated at every render. A
  captured camera becomes adjustable once a render contains it.
- **Your viewpoints from the start.** Pick *manual* on the selector, **Capture view**,
  fly to each place you want a view from and **Save pose** (several per object, from
  different sides), then **Render the viewpoints**. With nothing saved, Render refuses
  and says so. This is the way in when Carveout's own placement cannot serve a scene.
- **Switch to fully manual.** Converts every automatic view of the current render into
  an adjustable viewpoint of your own, stops generating new ones, and re-renders. The
  render then reproduces exactly the viewpoints on disk: no sampling, no pruning, the
  quality filter advisory only, coverage measured but no longer repaired.

**Removing and re-rendering.**

- **Delete** on a viewpoint's frame strikes it on the sheet and in 3D until the next
  render drops it; the render cannot be approved until then. Automatic viewpoints have
  no Delete; convert to fully manual first.
- **Reset the render step** deletes every viewpoint and every frame and restores the
  previous path, a clean start. It asks first and names what it will remove.
- **Re-render** after changing camera or quality settings; the button says how many
  viewpoints are new or edited and not yet rendered.

**"would have FAILED the quality filter" on a manual view.** Your views are kept
whatever the filter says; the line tells you what it would have judged, in numbers.
The near-field reason measures opacity within a horizon in front of the camera (0.2 of
the scene's smallest extent), and a close-up standing inside that horizon reads its own
subject as fog; the line then says "a close-up … distance tripped it, not quality". A
sharp, well-framed close-up is fine.

**On disk:** `work/<scene>/stage1/` holds `frames/frame_XXXX.png`, `depth/depth_XXXX.npy`,
`cameras.json` (intrinsics and per-frame camera-to-world matrices), `review_gate.png`
(contact sheet), `manifest.json` (settings, discard reasons, coverage). Captured
viewpoints go to `<workdir>/manual_views.json`.

### Step 3: Vocabulary

**What it does.** SAM 3 finds what you name. The vocabulary is a list of short noun
phrases ("office chair", "cardboard box") plus optional **negative controls**, phrases for
things you know are *not* in the scene. Probing runs the whole list over the rendered
views and records every detection above a low logging floor, so you see, per phrase, how
many detections it produced, in how many frames and at what scores. The negative controls
are the calibration check that matters: if an absent thing scores as high as the present
ones, the scores separate nothing and the threshold is wrong.

**What you do.**

1. Type the phrases you want, or ask the local vision model for a suggestion over the
   rendered views: a proposed list plus negative controls, each term a chip you adopt
   with a click (add-all covers a list). Nothing is adopted for you.
2. Probe. The cost line under the prompts says how long the list will take and what the
   lift will need for it.
3. Read the results: per-phrase counts and score ranges, the score-distribution plot,
   and the overlays showing what was masked on each frame. Drop phrases that found
   nothing or masked the wrong thing. **Look at the overlays**: a healthy score table
   can hide a phrase matching a completely different object.
4. Confirm. The lists are written into the scene's profile, and the probe you reviewed
   is the detection: the pipeline keeps it as long as the views, the volume and the
   lists are unchanged.

Aim for breadth: an extra phrase costs probe time and is cheap to drop here; a missing
phrase is an object that can never be found. The cost line never trims the list. A
negative control scoring at or above the threshold is said and recorded on the approval
but does not block, since negatives never reach the output; §7 says what the reading means.

**Starting over.** **Reset vocabulary** in the footer clears the lists and the proposal
and re-opens the gate, after asking; the probe's files stay until you clear the
detections on the Objects panel. Proposing again over a proposal asks too, and the terms
you adopted stay. Each proposal row has **remove added**, each list **clear all**.

**On disk:** `work/<scene>/stage2/` holds `scores.csv` (every detection), `score_summary.csv`
(per phrase), `score_distribution.png`, `overlays/frame_XXXX.png`,
`masks/frame_XXXX/<phrase>_<k>.png`, `probe_manifest.json`; the model's suggestion in
`stage15/proposed_vocab.json`; the confirmed lists in `configs/scenes/<name>.yaml`.

### Step 4: Visual references (Exemplars, optional)

**What it does.** Some objects cannot be named into existence: no phrase picks them out.
Draw a box around one instance on a rendered frame and Carveout feeds the crop to SAM 3's
video predictor as a visual prompt: it finds matching objects on that frame and tracks
them across the others. These detections are added to the text detections.

**What you do.** Only when a phrase has demonstrably failed at step 3. Pick the concept
name (it becomes the exported label), draw one or more crops on frames where the object
is clear, then **Re-probe**: only the visual pass runs, the text results are kept.
Visual-similarity scores live on their own scale, never compared with text confidence;
each concept has its own threshold here, calibrated the same way, by the overlays.
Changing a threshold is a calibration write: it re-opens this gate and everything
after it, like a threshold change on the render panel does.
Crops you add or change are not in the run until Re-probe has seen them; **Continue**
says so and waits. With no crops, Continue is the act: skipping this step is normal.

Each crop records a hash of its frame: after a re-render, detection refuses to run
against changed pixels and tells you to re-draw. On 24 GB cards the visual pass is capped
at `detect.exemplar_max_frames` frames (24 on the `24gb` profile, 48 on `32gb`), the
frames carrying your crops always kept; the panel shows the frames used and the peak VRAM.

**On disk:** `<workdir>/exemplars.json`; detections join `stage2/scores.csv` as
`source=exemplar:<concept>`; overlays as `overlays/exemplar_frame_XXXX.png`.

### Step 5: Detect, lift, export (in the Objects panel)

These three run unattended, in order, from the **Objects** panel; while any review point
above is still open, the panel lists what is left to approve instead of offering the run.

**Detect** runs the confirmed vocabulary over every rendered frame, one mask per
detection, or keeps the probe you reviewed when nothing has changed.

**Lift** turns masks into per-Gaussian labels. A rendered pixel is a linear combination
of the Gaussians behind it, so one backward pass per view says how much each Gaussian
contributed to each class's masked pixels; a Gaussian takes the class that dominates.
First, overlapping detections in a view are de-duplicated: two masks that contain each
other are one object under two names (the higher score wins), and a mask consistently
inside another is a part of it. Labelled Gaussians are then grouped into instances by the
**instancing** selector on the panel:

- **Tracks** (default) follows each detection across the views and groups by that
  identity.
- **Connectivity** groups by 3D adjacency and checks the count against the 2D evidence:
  one blob where the views consistently show two objects is split, but only if the split
  separates the evidence. Try it when Tracks under-counts objects that touch or repeat;
  the report names the algorithm that produced the export.

Changing it re-opens verify consent and re-runs lift and export. Both passes size their
memory to the card; the log says what they chose.

**Export** computes, per instance: centroid, axis-aligned extents, an oriented box
(gravity-aligned for ordinary objects, full PCA for thin tilted things like an open
door), the views that see it with each detection score, and an aggregate confidence. It
writes `interactions.json`, one `.ply` per object, and per-frame overlays with the
projected boxes. Instances that matched no detection in any view are dropped.

**What you do.** Start the run and watch the strip below the canvas (expand it for the
log). You can cancel (stages stop at a safe boundary and stay re-runnable) and close
the browser and re-attach without losing the stream.

**The one interruption.** A detection whose mask dwarfs the usual mask of its class is
dropped, and the run stops to show you the frame: usually SAM 3 masked a whole piece of
furniture as a small-object class and swallowed the real instances.
Look at the outlined view, type the class name shown in the row, and press "I looked,
resume the run". If an object of that class really is that big, give the class its own cap with
`detect.max_area_overrides` in the scene profile and run again.

**On disk:** `stage3/` holds `labels.npy` (class per Gaussian, -1 = none), `instances.npy`
(instance per Gaussian), `classes.json`, `instance_audit.json` (every split, merge and
suppression), `manifest.json`. `stage4/` holds `interactions.json` (the objects),
`instances/NNN_<label>.ply` (openable in any splat viewer), `debug/frame_XXXX.png`,
`manifest.json`.

### Step 6: Label verification (on the Objects panel, optional)

**What it does.** A local vision-language model is shown each exported object's best
views and asked what it is, with the label withheld; a matching answer confirms the
label, a different one is weighed against it on equal footing, and its unprompted name is
recorded beside the verdict. (Told the label up front it agreed with almost every wrong
one; `vlm.verify_name_first: false` restores that for comparison.) Verdicts: confirm,
relabel, reject. A relabel must be re-confirmed on the object's *own* Gaussians in at
least two views and stay within your vocabulary, else the detected label is kept; a
reject removes the object from `interactions.json` and keeps it, with the reason, in
`stage45/rejected_instances.json`. The detected label always stays canonical; the
verdict is recorded beside it, and the viewer shows the verified label when there is one.

**What you do.** It asks first (how many objects, which model, minutes of GPU time),
and nothing leaves your machine. No is a complete answer: labels ship unverified and the
report says so.

**If it stops.** A cancelled or crashed run keeps every verdict it reached; the panel
says "N judged before the run stopped" and the next press judges only the rest, as long
as the export, the model and the settings are the same.

**Judging a few objects.** Press **verify this** on an object's card in the 3D view, or
on its report row. While any object is marked, Run verification judges the marked ones
only; the others keep their verdicts from the previous verification of this export, or
ship **not judged** with a badge when there is none. **Clear marks (verify all)** takes
the marks back. Use it to re-judge a handful after a rename, another model or an option;
a first pass on a scene you do not know should judge everything.

**Two options**, off by default and per scene; either re-opens verification as a real
run, and `stage45/verdicts.previous.csv` keeps the earlier verdicts for comparison:

- **let a more specific name stand**: the verifier may normally apply only names from
  your vocabulary, so it cannot re-import terms you collapsed at step 3; this lets a name
  outside the list through when it is a kind of the detected label (an onion for
  "vegetables"), never a more general word.
- **show small objects on their own too**: for objects under 300 Gaussians, the
  object's own reconstruction rendered alone joins the crops as one more image, since a
  crop of a small object often shows its surroundings better than the object. Never
  alone, never for large objects.

**Held objects** ship **HELD** rather than confirmed or rejected: the verifier could not
establish that the object is *one* object. The reasons come from the lift and the export
(a detection spanning two objects the masks told apart, a merge the mask evidence
refused, a substantial part left unassigned) or from a projection mismatch found here
(the object's own projection disagrees with the box that scored it). A held object keeps
its label and score, with the original verdict and any proposed relabel beside it; the
report lists each with its reasons and the Workbench badges it `held`. The association
reasons show even when you skip verification.

**On disk:** `stage45/` holds `verdicts.csv` (verdict and rationale per object),
`verdicts.previous.csv`, `crops/` (exactly what was sent),
`interactions.pre_verify.json` (the untouched export), `rejected_instances.json`,
`manifest.json`; `verdicts.partial.jsonl` while a run is under way or stopped.

### Step 7: Report

Written at the end of a run: what was approved and when; how many objects were exported;
the verification summary (confirmed / relabelled / rejected / held / unverified / not
judged, which add up to the exported count; the per-reason hold counts overlap and are
diagnostics); the held objects with their reasons; the objects that never reached export;
any area-based kills; and the **per-class funnel**: for each phrase, how many raw
detections it produced and how many survived the threshold, de-duplication, region
scoping, instance forming and the export floor. The funnel is where you see *where* a
class died.

**On disk:** `work/<scene>/run_report.md` and `report.json` (the same, machine-readable).

---

## 4. Looking at the results

The Workbench is the results viewer. Open a finished scene: the canvas shows every
exported object with its label and confidence, and the **Report** panel lists them;
click one to fly to it and tint its Gaussians. The report also carries the gate timeline,
the verification summary and the per-class funnel (Step 7).

**The object card.** Selecting an object (a label on the canvas or a row of the
report's list) opens a card with what the export knows: size, oriented-box extents,
position and Gaussian count, in metres when the scene's factor is recorded and in scene
units otherwise (the card says which). After verification it also shows the verdict and
its one-line reason. Badges: `held` for an object whose association evidence is
unresolved, `unverified` for a format failure, `not judged` for one you did not mark; a
label the verifier doubted carries an amber `?`.

**Naming an object yourself.** The pencil on the card or on a report row lets you type
your own name, say an egg carton the detector called a box. Your name shows first, in 3D
and in the list, with the detected label kept beside it; an empty entry takes it back; a
re-run of verification keeps your names. The per-object files keep the detected name
until the next export.

**Taking the verifier's proposal.** Verification may propose a name it is not allowed to
apply: outside your vocabulary, supported by too few views, or on a held object. The
card says "the verifier proposed" with the name as a button; one click makes it your
name, the same as typing it. The **label source** control in the viewer settings
previews every proposal on the 3D labels at once (*proposed*); it applies nothing. The
same control shows the raw detected term for every object (*detected*), your names and
the verifier's verdicts included; *verified*, the default, shows your name, else the
verdict, else the detected term.

**Starting the detection half over** (after a vocabulary change, say): **Clear
detections** on the Objects panel deletes the probe, the lift, the export and the
verification and re-opens the verify gate; the render and the gates before it stay.

**Navigation.** Left-drag to look, right-drag to pan, wheel to dolly, `WASD`+`QE` to
move, `Shift` for fast. The toolbar over the canvas toggles the volume, camera, instance
and crop-source overlays and the origin grid; `Tab` hides the panels.

**The Viewer panel** holds the display settings: field of view and a per-scene start
pose, the scene background, label display (always or within a distance, how many, the
minimum confidence, with a separate one for visually-prompted objects, whose scores are on
their own scale), raw or verified labels, marker style, a focus mode that renders
everything outside a region as sparse points, and control feel. **Save for this scene**
writes them to `viewer_settings.json` in the workdir, so the scene looks the same
however it is opened; unsaved changes last for the session.

The canvas works in **display metres** whatever the scene's unit: the recorded factor
when there is one, else a nominal 5 m over the scene's largest extent, so a scene of
unknown scale is shown about room-sized and a kilometre or millimetre scene flies,
labels and grids like any other. Readouts say "units" until a factor is recorded.

To inspect an object outside Carveout, open its `.ply` from `stage4/instances/` in any
splat viewer; `interactions.json` carries the same objects with positions, extents,
oriented boxes, supporting views and confidences.

## 5. Re-running: which act does what

| You want to | Use |
|---|---|
| Continue or redo a run | open the scene; the app resumes from the journal, skipping unchanged steps |
| Re-render | **Re-render** on the Render panel (or **Reset the render step** to start the stage over) |
| Re-probe the vocabulary | **Run probe** on the Vocabulary panel; re-confirm |
| Redo detection, lift and export | **Clear detections**, then **Run detect → lift → export** on the Objects panel (a probe already run over the same views, volume and vocabulary is kept, and the log says so) |
| Re-verify labels on an existing export | **Run verification** on the Objects panel; with nothing changed since the last verification (same export, model and settings) it replays the kept result at once and says so; a new export or another model makes it judge again |
| Answer the verify gate the other way (ran → skip, or skip → run) | **Reopen gate…** on the verify gate first: a recorded answer stands while nothing changed, and the other button is refused with that reason |
| Continue a verification that stopped | **Run verification** again; it continues from the last verdict saved (the panel says how many) |
| Judge a few objects again | mark them (**verify this** on the object's card or report row), then **Run verification**; **Clear marks (verify all)** to judge everything again |

**What invalidates what.** Stages cache their outputs and re-use them; approvals are pinned
to the content of the things they depended on. Change a captured viewpoint, the region, the
vocabulary or a visual reference, and the affected step and everything after it re-open, and
the affected stages re-run. Two cases refuse instead of silently re-running: `verify` will
not run against an export whose inputs changed underneath it (re-run the pipeline first),
and detection refuses visual references drawn on frames that a re-render has since changed
(re-draw them).

---

## 6. Output reference

A completed working directory:

```
work/my-scene/
├── run_journal.json        approvals, pinned to the content they approved
├── run_lock.json           present only while a front-end is attached
├── run_report.md           the run report
├── report.json             the same report, machine-readable
├── stage1/                 rendered views
│   ├── frames/frame_XXXX.png
│   ├── depth/depth_XXXX.npy
│   ├── cameras.json        intrinsics + per-frame camera-to-world
│   ├── review_gate.png     contact sheet of every kept frame
│   └── manifest.json       settings, discard reasons, coverage
├── stage15/                LLM vocabulary suggestion (if used)
│   └── proposed_vocab.json
├── stage2/                 detections
│   ├── scores.csv          every detection: frame, phrase, score, box, area, source
│   ├── score_summary.csv   per-phrase aggregates
│   ├── score_distribution.png
│   ├── overlays/           what was masked, per frame
│   ├── masks/frame_XXXX/   one PNG per kept detection
│   └── probe_manifest.json
├── stage3/                 per-Gaussian labels and instances
│   ├── labels.npy          class id per Gaussian (-1 = unlabelled)
│   ├── instances.npy       instance id per Gaussian (-1 = none)
│   ├── classes.json
│   ├── instance_audit.json every split, merge and suppression decision
│   └── manifest.json
├── stage4/                 exported objects
│   ├── interactions.json   the objects: label, position, extents, box, views, confidence
│   ├── instances/NNN_label.ply
│   ├── debug/frame_XXXX.png
│   └── manifest.json
└── stage45/                verification (if run)
    ├── verdicts.csv
    ├── verdicts.previous.csv   the set before the last run
    ├── verdicts.partial.jsonl  checkpoint of a run under way or stopped
    ├── crops/
    ├── interactions.pre_verify.json
    ├── rejected_instances.json
    └── manifest.json
```

At the top of the workdir, beside `run_journal.json`: `volume.json`, `volume_review.png`,
`density.json`, `manual_views.json`, `exemplars.json`, `viewer_settings.json`. The scene
folder under `data/scenes` holds the capture only.

### Scene profiles

Every scene's profile is one file, `configs/scenes/<name>.yaml`, created by **New scene**
from the defaults in `configs/default.yaml` (which also holds the two VRAM profiles)
and written by the gates as you calibrate. The
directory is git-ignored: a profile carries a scene's calibration and its vocabulary, and
none ships. Paths inside it (`scene_path`, `workdir`) are relative to the repository root.

A new profile always starts from the defaults, never from another scene's calibration.
If you copy a profile by hand, or write one, the app **adopts it on its first write**: the
`driver_scaffolded` marker is added and the file is regenerated from its values, so
comments in it do not survive; the original is kept once, beside it, as
`<name>.yaml.orig`. Never carry `detect.prompts`, `detect.negatives` or the per-class
override maps between scenes: the vocabulary is reviewed per scene, at the gate. Nothing
about gate safety depends on who wrote the file: every approval is pinned to the file's
content, and any change re-opens the affected gates.

**Settings** in the Scene library header shows two read-only facts about the running
server: where profiles live, and the **vision model**: its name, its directory, and
`Ready` or `Unavailable`. Ready means the model's weights are in place (`config.json`
present in the model directory, so a half-finished download reads unavailable) and the
Python environment can load them. If it says unavailable, the Vocabulary and Objects panels
show the same badge with the reason under it (for missing weights, the exact download
command) and keep their action disabled; see README step 7 and §7 below.

**The vision model is chosen per scene.** It reads the measured length at the volume
gate, proposes the vocabulary and verifies the labels. Every model installed under
`models/` (README step 7: Qwen3.8-27B at ~17.5 GB of VRAM, and the smaller, faster
Qwen3-VL-8B at ~6 GB; every model loads compressed, 4-bit, so a smaller model is a
smaller footprint, not a faster full-precision one) is offered as the third question
of **New scene**: the dialog reads the scene's Gaussian count from the file's header
and your card's memory, says what each model needs beside that scene, and pre-selects
the largest that fits, labelled *proposed*; you may pick another. The Vocabulary and
Objects panels carry the same choice for later; changing it re-opens the verify gate
only, because the model is one of that gate's recorded inputs (the proposal is a draft
you adopt, the length read a check). `carveout web --vlm-dir DIR` makes a model the
server's default for scenes that did not choose. Every verification records which
model and quantization produced it.

---

## 7. Troubleshooting

Most of these quote the message verbatim; the rest name the symptom.

**`SAM 3 checkpoint missing`**: the checkpoint is not at `models/sam3/`. Request access on
Hugging Face, `hf auth login`, then download it (the message repeats the exact command).
`sam3.pt` is needed for every run; `sam3.1_multiplex.pt` only for the visual-reference pass.

**`CUDA is required for rendering (gsplat)`**: no usable GPU. Check
`python -c "import torch; print(torch.cuda.is_available())"`. If the first render dies while
compiling instead, gsplat's kernel build cannot find a CUDA 12.8 toolkit; verify
`$CUDA_HOME/bin/nvcc --version` and `$CUDA_HOME/include/cuda_runtime_api.h`.

**`CXXABI_1.3.15 not found`** on the first import: the Ubuntu 22.04 libstdc++ mismatch;
apply step 6 of the installation.

**`all N candidate view(s) were discarded by the quality filter`**: every generated view failed
as degenerate. The message names, per reason, the exact threshold that would admit the
closest frame, and every one of those thresholds is adjustable in the render gate's
calibration panel. `low_coverage` and `inside_geometry` usually mean the cameras are in the
wrong place (wrong path mode, or a region of interest whose footprint is filled by the
objects themselves; turn on *tight focus volume*), while `near_field_fog` and
`blurry_mush` mean the thresholds under `render.quality_filter` are too strict for this
scene's reconstruction quality.

On a small scene the two **distance** thresholds (`near_field_depth`,
`min_median_depth`) can reject every view the room permits; set them to `auto` (the
**auto** button beside each field on the render panel; the field then shows what it
resolved to) to resolve them from the scene's own width, and let `min_sharpness` do the
judging. The
message names the *first* filter each view hit, so fixing one can reveal another behind
it; read the per-view numbers in `discard_report.json`. Under a factor other than 1 the
message gives the knob's value as the profile writes it, then the measured scene units in
brackets; type the first. A refused re-render publishes nothing: the render you had stays
on disk and the render gate says so in a red **Last attempt refused** banner naming the
knobs that have moved since that render was made; approving ships that earlier render.
Full calibration procedure: [`CALIBRATION.md`](CALIBRATION.md), "Close-quarters scenes".

**`no volume.json ... placement and coverage are unbounded`**: you rendered without a
region of interest. Define one first; without it, cameras and coverage have nothing to
scope them.

**Fewer views kept than requested** is not a failure: the filter and the near-duplicate
rejection removed the rest. Capture viewpoints by hand for the parts you care
about, or relax the quality thresholds.

**`workdir held by '…' since …` / HTTP 423**: another app instance (or a dead one) holds
the lock. Close the other driver. If its process is genuinely gone, release the lock explicitly; a live
holder is never released.

**`webui bundle not built`**: run `npm --prefix webui ci && npm --prefix webui run build`.

**`no vocabulary: the vocabulary gate must confirm detect.prompts`**: detection was invoked before a
vocabulary was confirmed. This is deliberate: detection never runs on an unreviewed list,
and the lists cannot be confirmed until the probe has run on them exactly as they stand.

**ceiling VIOLATED: recorded, not blocking** (the vocabulary panel's footer, with the
score line "negative controls scored …, above the … limit: look"): an "absent" phrase
scored at or above the detection threshold. Not blocking, and it cannot corrupt a run (negatives never export);
it is recorded on the vocabulary approval. Open the named overlay when labels look wrong:
either the threshold needs calibrating, or the thing is not actually absent.

**Area-factor kill**: a detection's mask dwarfed the typical mask of its class and was
dropped. Inspect the named frame overlay: usually SAM 3 masked a whole structure as a
small-object class. The line appears the moment it happens, at the start of the lift's
slowest step; the run then pauses for your typed acknowledgment once the lift is done.
The lift's mask dedup can take minutes on a dense probe; it prints one line per view.

**`exemplar crop ... was drawn on frame NNNN whose content has changed`**: you re-rendered
after drawing visual references. Re-draw them on the new frames.

**`refusing to verify stale results`**: captured viewpoints or visual references changed
after the export. Re-run the pipeline; verification alone cannot repair a stale upstream.

**`No vision model on this machine`**: the local model is not downloaded; see README step 7
(the message repeats the exact command). The model is optional; everything else runs
without it.

**`needs about N GiB of GPU memory free to load`**: the vision model does not fit in
what the card has free. Close other GPU programs (a browser tab showing the scene holds a
copy of it on the card), or start the server with the smaller model (`--vlm-dir`, above).

**`The lift's class pass cannot fit on the card`**: the lift rasterises the whole
vocabulary at once, about 12 bytes per Gaussian per class, and the message names how many
classes this card fits for this scene. Shorten the vocabulary at the Vocabulary gate (the
panel's cost line shows the fit), close other GPU programs, or use a card with more
memory. The New scene dialog states this number when you pick a file, so it is rarely a
surprise here.

**`The lift cannot fit … detection channel(s) on the card`**: the lift's detection pass
holds, per Gaussian in the file and per detection it rasterises at once, about 24 bytes; it
takes the largest chunk of detections that fits in what the card has free (the log's
"detection channels: … GB free … -> chunk N" line, and `det_stats.det_channel_budget` in
`stage3/manifest.json`, say what it chose). This refusal means even one channel does not
fit: close other GPU programs (a browser tab showing the scene holds a copy of it on the
card), or use a card with more memory. A smaller volume does not help: the lift renders
the whole file and the volume only scopes which Gaussians may be labelled.

**The server prints nothing while a stage runs**: it was started through `conda run`,
which buffers output until the process exits. Start it with `conda activate carveout`
then `carveout web`; the stage log in the app is unaffected either way.

**More than 65534 instances**: the viewer's instance-id format cannot address them. This
means the scene produced an implausible number of objects; check the funnel in the report
before anything else.
