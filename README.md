# Carveout

Carveout finds and labels objects in trained 3D Gaussian Splatting scenes. You give it a
`.ply` or a `.sog`; it renders synthetic views of the scene, runs open-vocabulary segmentation on
those views with SAM 3 Promptable Concept Segmentation, lifts the resulting 2D masks back
onto individual Gaussians, and exports per-object 3D instances: label, position, oriented
bounding box, supporting views, and a per-object `.ply` subset. There is no
retraining and no per-scene model: the vocabulary is a list of noun phrases you review
before detection runs. You drive Carveout from a browser app (`carveout web`) that walks
you through every review point on a live 3D canvas and shows the finished results on the
same canvas.

![The Carveout Workbench on a finished run: a Gaussian-splat scene of a salon with every
detected object labelled and scored on the canvas, the review points down the left edge,
and the Render panel open on the right with its contact sheet of views.](docs/images/hero.png)

## Is this for you?

Carveout is for scenes you have already trained. It does **not** train Gaussian Splatting
scenes, and it does not run on point clouds, meshes, or NeRFs. You need a Linux box with a
recent NVIDIA GPU, and access to Meta's gated SAM 3 checkpoints (free, but you must
request it and accept Meta's licence yourself). A run is interactive by design: Carveout
stops and asks you to review the region of interest, the rendered views, and the
vocabulary before it commits to detection.

Carveout has been run on a selection of real captures: interiors (shops, offices, a
lab, a garage), an outdoor site on sloped ground, and captures focused on one area or
one object, some with a known metric scale and some without. That selection is not
exhaustive. A capture with different qualities or an unusual object may meet problems
Carveout has not seen, so treat every review point as a real check rather than a formality.

## Requirements

These are the **tested** configurations. Nothing else is supported or has been tried.
The measured figures here and under *Performance and sizing* are from September 2026,
on the RTX 5090 unless a line names the RTX 4090.

| | Tested |
|---|---|
| OS | Ubuntu 22.04, native (not WSL, not containerised) |
| GPU | NVIDIA RTX 4090 (Ada, sm_89, 24 GB) and RTX 5090 (Blackwell, sm_120, 32 GB) |
| Driver | NVIDIA driver supporting CUDA 12.8 |
| CUDA toolkit | 12.8 at the **system** level (`nvcc`), needed only to JIT-compile gsplat's kernels |
| Python | 3.12 (required by the `sam3` package) |
| PyTorch | 2.7.1 + cu128 wheels only; older wheels have no sm_120 kernels |
| RAM | 64 GB tested; the server process reached 50 GB resident on a 14 M-Gaussian scene |
| Disk | ~7 GB of SAM 3 weights, plus renders and masks per scene; the optional vision models add ~52 GB (27B) and ~16 GB (8B) |

VRAM is a config value, not a code path: `24gb` is the default, 4090-safe profile
(768×768 renders); `32gb` is opt-in (1024×1024). Select it once at launch with
`carveout web --profile 32gb` (or set `profile: 32gb` in `configs/default.yaml`).
A desktop session is required: the review steps run in a browser, and there is no
headless mode. The optional vision-language model (a second opinion at the volume
gate, the vocabulary suggestion, label verification) runs locally and is off unless you
opt in; nothing Carveout does reaches the network at run time.

### Performance and sizing

Measured in September 2026 on the tested cards, on two real captures: a small interior (2.1 M Gaussians,
39 views, ~70 prompts) and a large, dense one (13.9 M Gaussians, 40 views of which 10
hand-captured, 73 prompts). Times are for the RTX 5090 unless noted.

| Stage | What it scales with | Measured |
|---|---|---|
| Render, 1024² | views | 40 views in 31 s; peak ~7 GB |
| Probe (SAM 3) | views × prompts | 73 prompts × 40 views in 5.0 min (~0.1 s per prompt and view); 8 GB |
| Lift | Gaussians × classes (memory); detections (time) | large scene: 168 s, peak 15.9 GB; small scene: 114 s, 4.9 GB |
| Export | objects | ~30 s for 641 objects |
| Verification (27B, 4-bit) | objects | 3.5–7 s per object (RTX 4090 / RTX 5090 with support checks); 641 objects ≈ 70 min |
| Vision model load | disk speed | 27B: 60–100 s from a SATA SSD, ~20 s from NVMe; 8B: seconds. 27B needs ~17 GB of VRAM, 8B ~6 GB |

Keep the model weights on an NVMe drive: the 27B is read whole at every load.

**How big a scene can be.** Every pass sizes itself to the card at run time except the
lift's class pass, which needs about 12 bytes per Gaussian per class of vocabulary and is
the ceiling:

| Gaussians | 24 GB card | 32 GB card |
|---|---|---|
| 2 M | ~880 classes | ~1,200 classes |
| 14 M | ~130 classes | ~180 classes |
| 30 M | ~60 classes | ~85 classes |

The New scene dialog states the number for the file you pick, the Vocabulary panel
prices the list as you edit it, and the lift refuses before that pass, naming the
vocabulary that fits, rather than fail inside it. Density (Gaussians per cubic metre)
costs render time, not memory. Verification is the long pole on a scene with many
objects; it saves every verdict as it goes and can be stopped and resumed, or limited
to objects you mark.

## Installation

Copy-pasteable, in order, from a clone of this repository. Every step below is
reproducible from files in the repo except where noted.

```bash
# 1. System CUDA 12.8 toolkit. gsplat has no prebuilt wheels for torch 2.7/cu128 and
#    JIT-compiles its CUDA kernels on first use. Skip if the box already has 12.8.
#    Versioned install; coexists with other CUDA versions and does not touch the driver.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

# 2. Python environment (torch cu128, gsplat, sam3, and the carveout package itself)
conda env create -f environment.yml

# 3. Env-scoped configuration: TORCH_CUDA_ARCH_LIST="8.9;12.0", and HF_HOME/TORCH_HOME
#    pointed at ./models/ so no weights land in a global cache.
bash scripts/setup_env.sh

#    Another env name: `conda env create -f environment.yml -n <name>`, then
#    `ENV_NAME=<name> bash scripts/setup_env.sh`, substituting it below. (Without
#    ENV_NAME, a second checkout re-points the first's env at this ./models/.)

# 4. Point the environment at the toolkit (adjust the path to your install)
conda env config vars set -n carveout CUDA_HOME=/usr/local/cuda-12.8

# 5. Activate. Use `conda activate`, never `conda run`: `conda run` buffers all output
#    until the process exits, so a healthy 10-minute stage looks hung.
conda activate carveout
nvcc --version                      # expect release 12.8

# 6. Ubuntu 22.04 only: its system libstdc++ tops out at CXXABI_1.3.13, but the cu128 and
#    sam3 wheels are built against GCC 13. Skip on distributions shipping GCC 13+.
conda install -n carveout -c conda-forge "libstdcxx-ng>=13"
touch "$CONDA_PREFIX/etc/conda/enable_libstdcxx_preload"
conda activate carveout             # re-activate to pick up the preload

# 7. SAM 3 checkpoints. GATED: request access on the Hugging Face pages for
#    facebook/sam3 and facebook/sam3.1 first, and accept Meta's SAM License there.
#    `sam3.pt` drives detection; `sam3.1_multiplex.pt` serves only the optional
#    exemplar (visual-prompt) pass, but the step 9 smoke test expects both.
hf auth login
hf download facebook/sam3   sam3.pt             config.json --local-dir models/sam3/
hf download facebook/sam3.1 sam3.1_multiplex.pt config.json --local-dir models/sam3/

#    The local vision model for the OPTIONAL uses: a second opinion on your
#    measurement at the volume gate, the vocabulary proposal, and label
#    verification. Apache-2.0 and NOT gated: no access request, no
#    `hf auth login`, unlike the SAM 3 checkpoints above. ~52 GB on disk, loaded
#    4-bit at run time (~17.5 GB of VRAM). Skip it if you do not use those
#    stages: the core pipeline never touches it, and without the weights they
#    refuse with this download command.
hf download Qwen/Qwen3.8-27B --local-dir models/qwen3.8-27b/

#    OPTIONAL smaller model: Qwen3-VL-8B, Apache-2.0, ~16 GB on disk, faster,
#    and about a third of the VRAM: every model loads COMPRESSED (4-bit), so
#    the 27B needs ~17.5 GB of the card and the 8B ~6 GB, which is what
#    leaves room beside a large scene. Which model a scene uses is chosen
#    when you create it; see docs/USAGE.md §6.
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3-vl-8b/

# 8. Build the browser front-end (Node and npm are BUILD-time dependencies only;
#    `carveout web` itself runs on the Python environment and is launched later;
#    see the Quickstart below). Everything the bundle needs (three, Spark,
#    the fonts) is bundled at build time; the product makes no network calls
#    at run time.
npm --prefix webui ci && npm --prefix webui run build

# 9. Verify the install (the first gsplat run JIT-compiles kernels, ~2 minutes)
python scripts/smoke_test_torch.py
python scripts/smoke_test_gsplat.py
python scripts/smoke_test_sam3.py IMAGE.jpg "a prompt"
```

Steps that depend on something not in this repository:

- **The model weights** (step 7) come from Hugging Face. Carveout redistributes
  neither weights nor model code, and `configs/default.yaml` points at the local
  directories, never a hub id, so a run can never silently fetch a model.
- **Node and npm** (step 8) come from your system, not conda; `webui/package.json`
  requires Node ≥ 20, and the front-end is built with Vite 5.
- **`scripts/smoke_test_sam3.py`** defaults to a test image from the upstream sam3
  repository, not vendored here. Pass your own image and prompt (shown above), or
  `git clone --depth 1 https://github.com/facebookresearch/sam3.git reference/sam3`.
- **A trained scene.** Carveout does not train scenes; bring your own `.ply` or `.sog`.

## Quickstart

Put your scene where Carveout expects it: one directory per scene, with the scene file
inside. The directory name becomes the scene's name:

```
data/scenes/my-scene/my-scene.ply     # or my-scene.sog
```

Carveout reads two containers, and the whole pipeline works the same from either:

- **`.ply`**: a trained 3D Gaussian Splatting file in the standard 3DGS layout:
  per-Gaussian position, opacity, scale, rotation and spherical-harmonic colour, which is
  what the common 3DGS trainers write. Carveout tells you on load if a field is missing.
- **`.sog`**: a PlayCanvas SOG v2 bundle, a compressed form of the same scene
  (`splat-transform my-scene.ply my-scene.sog` writes one).

**Orientation is yours to set.** Carveout renders `-Y` up and no splat container
records which way is up, so bring the scene in oriented (re-orient in whatever wrote
the file) or set `scene.up_axis` in the scene's profile. And don't swap a scene between
its `.ply` and `.sog` mid-run: the Gaussian order differs, so every gate demotes.
Both are covered in depth in `docs/USAGE.md` §1.

### The browser app: a run in fifteen steps

```bash
conda activate carveout
carveout web                        # serves http://127.0.0.1:8090 (localhost only)
```

1. **New scene.** Pick the scene file. Check the size line under it (Gaussians, and the
   vocabulary the lift fits on your card). Answer the three questions: metric or not,
   how it was filmed, which vision model. **Create scene**, then open it.
2. **Volume: scale.** If the scene has no scale: **Measure**, click two points on
   something of known length, type the length, **Record**.
3. **Propose fresh.**
4. **Level** if the Check card says the scene is off level: **Level as proposed** (or
   **Level by hand**: three clicks on the floor), then **Propose fresh** again.
5. Check the box and the floor plane on the canvas; drag to fit, **Set from a point**
   if the floor is wrong. **Approve volume**.
6. **Render.** **Adopt** the card's proposal if it makes one. **Render views**.
7. Review the frames. Capture your own viewpoints where something is framed badly, and
   re-render. **Approve render**.
8. **Vocabulary.** **Propose with the vision model**, or type your phrases. Adopt the
   terms you want; keep two or three negative controls.
9. **Run probe.** Read the overlays; drop the phrases that found nothing or the wrong
   thing. **Confirm vocabulary**.
10. **Exemplars.** **Continue**. Draw a reference crop first only if a phrase failed.
11. **Objects.** **Run detect → lift → export**. If the run stops on an oversized mask,
    look at the frame, type the class name, resume.
12. **Run verification**, or **Skip**. To judge only some objects, mark them on their
    cards first.
13. **Report.** Read the funnel and the verdicts; click objects to fly to them.
14. Rename an object with the pencil on its card, or take the verifier's proposal with
    one click.
15. The results are `work/<scene>/interactions.json`, one `.ply` per object under
    `stage4/instances/`, and `run_report.md`.

Every step is explained in [`docs/USAGE.md`](docs/USAGE.md).

### Looking at results

The Workbench is also the results viewer: open a finished scene and its **Report**
panel lists every exported object. Click one to fly to it and tint its Gaussians.

Only one browser app may drive a working directory at a time: it holds a lock, and a
second app instance opens the scene read-only.

Note: Carveout runs from THIS repository checkout (editable install). The app
resolves `viewer/` and `webui/dist` relative to the repo tree, so a plain wheel
installed elsewhere serves neither. Install with `pip install -e .` inside the
checkout.

### Re-running

The app resumes from the journal: it skips review points whose inputs have not
changed, and re-opens a point (and everything after it) whose inputs have. Every
re-run is a button on the panel that owns it; see [`docs/USAGE.md`](docs/USAGE.md) §5
for which act does what.

## How it works

1. **Render.** Synthetic camera views are placed inside your volume of interest and
   rasterised with gsplat, with degenerate views filtered out and extra views generated to
   close coverage gaps.
2. **Detect.** SAM 3 Promptable Concept Segmentation runs your confirmed vocabulary over
   every view, logging raw scores so thresholds are calibrated from data rather than
   assumed. An optional visual-prompt pass handles what text cannot name.
3. **Lift.** Each 2D mask is attributed back to the Gaussians that rendered into it;
   per-Gaussian class labels become 3D instances by following each detection across the
   views (or, at your choice, by 3D adjacency cross-checked against the 2D count).
4. **Export.** Per-instance position, extents, oriented bounding box, supporting views
   and confidence, written as `interactions.json` plus one `.ply` per object.
5. **Verify** (optional, on the Objects panel). A local vision-language model names each
   instance from its views, then confirms, relabels, or rejects the label.

Full walkthrough with every option, output, and failure mode: [`docs/USAGE.md`](docs/USAGE.md).

## Models and acknowledgements

Carveout is a pipeline built on other people's work. It bundles none of it; everything
below is installed from its own upstream.

- **[SAM 3](https://github.com/facebookresearch/sam3)** (Meta): the segmentation model.
  Code and weights are distributed by Meta under its custom **SAM License**, and the
  checkpoints are gated: you request access and accept Meta's terms yourself, including
  its field-of-use restrictions. Carveout redistributes neither, and calls SAM 3 as an
  external package you install.
- **[gsplat](https://github.com/nerfstudio-project/gsplat)** (Apache-2.0): Gaussian
  Splatting rasterisation, used to render the synthetic views and to attribute masks back
  to Gaussians.
- **[three.js](https://github.com/mrdoob/three.js)** and
  **[SparkJS](https://github.com/sparkjs-dev/spark)** (both MIT): the 3D canvas. The
  browser app bundles them at build time; nothing is fetched at run time.
- **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** and
  **[Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)** (Alibaba, Apache
  2.0): optional, opt-in, for the measured-length read, vocabulary suggestions and
  label verification; one is chosen per scene. Run locally; the weights are
  downloaded by you and Carveout redistributes neither code nor weights.

## License

Carveout's own code is licensed under the **GNU General Public License v3.0 or later**
(`GPL-3.0-or-later`); see [`LICENSE`](LICENSE) for the text of version 3. Third-party
components keep their own terms, which are summarised in [`NOTICE`](NOTICE); the SAM 3
weights and code in particular are governed by Meta's SAM License, not by the GPL, and
are used at arm's length as a package you install and accept the terms for yourself.

## Citation

If you use Carveout in published work, please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff):

> Bruyère, H. (2026). *Carveout: open-vocabulary object segmentation and 3D localization
> for trained 3D Gaussian Splatting scenes* (Version 0.2.0) [Computer software].
> <https://github.com/smallfly/Carveout>

## Support boundary

**Supported:** Ubuntu 22.04 native, the two GPU configurations listed above, the workflow
documented in [`docs/USAGE.md`](docs/USAGE.md), and scenes supplied as 3DGS `.ply` files in
the standard layout or as PlayCanvas SOG v2 `.sog` bundles.

**Not supported:** any other OS, WSL, containers, multi-GPU or CPU-only execution, headless
or unattended operation of the review points, training or optimising Gaussian Splatting
scenes, and other scene formats. Carveout assumes a single local user: the server binds
to localhost only, there is no authentication, and no multi-user mode is planned. There is
no CI; the four unit tests under `tests/` cover the pure geometry, the scale rules and the
gate journal's cascade, and everything else is validated by running real scenes.

**Maintenance.** Carveout is a research project, released as it stands. It is not actively
maintained: issues and pull requests may go unanswered, and there is no support channel.
Fork it freely under the GPL. The repository starts at this release. The development
history before it was dense, made of iterations, test runs and personal working
records, and it was decided not to place it in the public repository. The evolution
of Carveout since its beginning is retraced in [`CHANGELOG.md`](CHANGELOG.md).
