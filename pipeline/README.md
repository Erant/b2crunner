# Native pipeline

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline. This
is the foundation for porting the node pack into a Dockerized, Gradio-fronted
tool that runs native PyTorch inference instead of routing through a ComfyUI
graph. It replaces `submit.py` + the ComfyUI API-format JSON graphs in
`workflows/api/` with plain Python: YAML workflows made of named `Step`s,
executed by a `Dispatcher` that hides *how* each step actually runs.

Status: the orchestration engine works end-to-end (verified: dataset
round-trip against real ComfyUI-written data, template resolution,
dispatcher caching, context wiring, and a real CLI run over `cyber_6f`).
Every node in the ComfyUI pack now has a native counterpart except
`DetectFaceLandmarks` — see "Coverage vs. the ComfyUI node pack" below.

What is verified varies a lot by step, and the distinction matters:
`rmbg`, `wan22_vace_denoise`, `sapiens2_lite`, `sam3d_body` and `seedvr2`
have run real inference on a GPU pod. `colmap_export`, `mask_splat`,
`inject_anchor` and the five `views` steps are verified against
`cyber_6f` — real recorded output of the ComfyUI stages they replace.
`brush` (a real short GPU training run), `render`'s pyrender/EGL path
(confirmed rendering on the actual GPU, not the Mesa software fallback),
and `render_splat`'s rasterisation — now the `brush-splat-render` binary,
not body2colmap's gsplat-based `SplatRenderer` (see
`pipeline/steps/splat.py`) — have all now run on a local RTX 4070 Ti, the
last one both as a raw binary call and through the actual pipeline step
against `cyber_6f`'s 81 real cameras. See `docs/docker-build-notes.md`.
Nothing is stubbed.

## Why this shape

Three requirements drove the design, all from direct conversation with the
project owner:

1. **Modularity** — a step should look identical to the dispatcher whether it
   runs in-process, in an isolated subprocess venv, against a warm HTTP
   service, or inside a Docker container. Swapping a step's execution
   mechanism is a one-line YAML edit, not a code change.
2. **Research-project flexibility** — workflows are human-edited YAML, not
   code. A workflow declares the form it wants: a `settings:` block of
   typed, labelled knobs and an `outputs:` block of deliverables, which is
   literally what the web UI draws. Those, plus a bare `globals:` block for
   plumbing with no control, share one namespace that steps read as
   `${globals.x}`; everything else is a per-step block overriding the
   defaults the Step class itself declares. Trying a new resolution or step
   count doesn't require touching Python — nor does promoting a step's knob
   to the UI — and two calls of the same step can be configured apart
   because a step's params are namespaced under its `id:`.
3. **In-memory by default** — datasets pass between steps as plain Python
   objects in a shared `Context`. Nothing touches disk unless a workflow
   explicitly includes a `save_dataset` step. This is a deliberate reversal
   from the ComfyUI-era pipeline, which persisted to disk at every stage
   boundary (see `workflows/pipeline/*.yaml` + `submit.py`).

## Module map

```
pipeline/
├── masks.py           normalize_mask()/mask_to_alpha_u8() — the one place
│                      the foreground=1 convention and its two ranges
│                      (float [0,1] from steps, uint8 [0,255] from disk)
│                      are reconciled. Three call sites had re-derived it
│                      inline; two of them wrongly.
├── dataset.py         Dataset — in-memory dataclass; to_disk()/from_disk()
│                      match the on-disk layout ComfyUI's Save/Load Dataset
│                      nodes already use (metadata.json, pointcloud.npz,
│                      frame_NNNNN_.png, reference.png, anchor.png, prompt.txt)
├── step.py            Step ABC: run(inputs, params) -> outputs, the Param
│                       declaration and the defaults merge, plus
│                      optional load()/unload() lifecycle hooks
├── registry.py         @register_step("name") / get_step_class("name")
├── context.py          Context: dotted-path get/set over a dict of objects
├── templating.py       "${a.b.c}" resolution against a workflow's globals
├── workflow.py         StepSpec / WorkflowSpec — the YAML schema, including
│                      the settings:/outputs: declarations the UI is drawn
│                      from; load_envs()
├── runner.py           WorkflowRunner — walks a WorkflowSpec, resolves &
│                      caches dispatchers, moves outputs back into Context
├── worker.py            Entry point run *inside* an isolated venv/container
│                      by SubprocessPythonDispatcher/DockerDispatcher
├── cli.py               `python -m pipeline.cli run <workflow.yaml> --dataset <dir>`
├── dispatch/
│   ├── base.py          Dispatcher ABC
│   ├── in_process.py    InProcessDispatcher
│   ├── subprocess_python.py  SubprocessPythonDispatcher (isolated venv)
│   ├── service.py       ServiceDispatcher (HTTP to a warm model server)
│   ├── docker.py        DockerDispatcher (container per call)
│   └── factory.py       build_dispatcher(dispatch, env_config) -> Dispatcher
├── steps/
│   ├── dataset_io.py    save_dataset / load_dataset — real, working
│   ├── rmbg.py          real, verified (single-image + batch paths)
│   ├── wan22_vace_denoise.py  real, verified against real inference
│   ├── sapiens2.py      real, verified (single-image + batch paths)
│   ├── brush.py         real, UNVERIFIED — dispatch: in_process (see
│                      docker/Dockerfile's comment on why brush is baked
│                      into the same image, targeting RunPod), never
│                      actually built/run
│   ├── sam3d_body.py    real, verified against real inference
│   ├── seedvr2.py       real, verified against real inference; also
│                      rescales the dataset's camera intrinsics to match
│                      the upscaled frames — the SeedVR2 upscale itself
│                      would otherwise silently invalidate them
│   ├── render.py        real, verified on GPU — camera-path + mesh/skeleton
│                      render; pyrender/EGL confirmed hitting real hardware
│                      (GL_VENDOR: NVIDIA) on a local RTX 4070 Ti
│   ├── face_landmarks.py detect_face_landmarks — MediaPipe (CPU-only),
│                      feeds render's face overlay; verified against
│                      cyber_6f's real photos
│   ├── views.py         drop_views / filter_fov / rotate_views /
│                      replace_views / merge_datasets — real, verified
│                      locally against cyber_6f's 81 real cameras
│   ├── mask_splat.py    real, verified against cyber_6f's recorded
│                      splatted/ -> masked_splatted/ stage output
│   ├── splat.py         load_splat / save_splat / render_splat — camera
│                      -path half verified locally against cyber_6f's
│                      recorded metadata; rasterisation shells out to the
│                      brush-splat-render binary (not gsplat — see the
│                      module docstring), verified end to end on GPU
│                      against all 81 of cyber_6f's real cameras
│   ├── colmap_export.py real, verified against cyber_6f's recorded
│                      colmap/ export (cameras.txt and points3D.txt
│                      byte-identical, images.txt to 2.4e-7)
│   ├── anchor_stub.py   generate_firstlast / inject_anchor — real,
│                      verified locally (pure numpy/cv2 logic, no GPU
│                      needed): affine + homography warp paths, anchor
│                      position-matching incl. duplicate cameras, no-anchor
│                      passthrough
│   └── reference_sheet.py split_reference_sheet — halves the two-panel
│                      front/back sheet the from-an-image path starts
│                      from: front to sam3d_body/generate_firstlast/rmbg,
│                      back over dataset.reference_image (all VACE
│                      conditions on). Pure numpy, verified locally
├── envs/
│   ├── envs.yaml        Per-machine registry: env name -> {python_bin |
│                      image | base_url}
│   ├── wan22/           requirements.txt + setup.sh (checkpoint/LoRA
│                      download) — see scripts/pod_bootstrap.sh
│   ├── rmbg/            requirements.txt (in_process — installed into
│                      the shared venv_main, not its own venv)
│   ├── sapiens2/        requirements.txt (in_process — same as rmbg)
│   ├── sam3dbody/       requirements.txt + setup.sh (gated checkpoint,
│                      pinned detectron2 build, numpy/cython +
│                      --no-build-isolation gotcha — see that
│                      requirements.txt's comment)
│   └── seedvr2/         requirements.txt + setup.sh (vendors
│                      numz/ComfyUI-SeedVR2_VideoUpscaler)
├── workflows/
│   ├── fast_helical_native.yaml a bootstrap prologue from a front/back
│                                sheet, then the full native port of the
│                                ComfyUI `fast helical` pipeline verbatim.
│                                `run_upscale: false` gates out the SeedVR2
│                                upscale (and the camera rescale that
│                                repairs what it invalidates) — the old
│                                fast_helical.yaml — for isolating the
│                                upscaler when output looks wrong. The
│                                bootstrap: split → sam3d_body →
│                                detect_face_landmarks → map_face_to_mesh →
│                                fit_head_to_face (the mesh head re-fitted
│                                to the photo's face, in the body model's
│                                own parameters) → a face branch
│                                (sapiens2_seg → crop_to_box → sapiens2_seg
│                                → sapiens2_lite → face_pointmap_splat)
│                                builds a Gaussian head from a crop of the
│                                sheet's front half → render draws an
│                                anchored circular orbit of
│                                outline+skeleton+splat frames, putting that
│                                face on every drawing within 60° of the
│                                source view → generate_firstlast +
│                                inject_anchor warp the photo onto the anchor
│                                frame. Since 2026-08-29 the face splat has
│                                replaced detect_face_landmarks; nothing else
│                                about this bootstrap changed, deliberately —
│                                it is the file for testing the face in
│                                isolation. rmbg/
│                                wan22_vace_denoise/sapiens2/sam3d_body
│                                verified on real hardware, render's own
│                                rasterisation not — see its STATUS note
│   └── fast_helical_shell.yaml  PARKED, selected by nothing. The same tail,
│                                but the whole bootstrap replaced by a
│                                photo-to-splat shell: rmbg + sapiens2_lite +
│                                pointmap_splat build a body-wide Gaussian
│                                shell → refine_pose_to_splat re-poses the
│                                fit to agree with it → a 380° helix (not a
│                                circle, and not anchored) → render_splat
│                                re-renders the shell along those same
│                                cameras → inject_shell_views swaps it into
│                                the frames within 15° of the source view and
│                                marks them as the batch's real-photograph
│                                frames, replacing generate_firstlast +
│                                inject_anchor. The pose fit replaces
│                                fix_head_angle (INCOMPATIBLE_STEPS). Carries
│                                the same face branch as the file above
└── tests/                 stdlib unittest, no pytest dependency. Run with
                           `python -m unittest discover -s tests -t .`.
                           Most tests are golden-output tests against
                           cyber_6f (real recorded ComfyUI stage output);
                           they skip cleanly when that directory is absent,
                           since it is gitignored local reference data.
```

Brush has no `pipeline/envs/brush/` directory — it's a Rust CLI baked
directly into `docker/Dockerfile`, not a Python env with a
`requirements.txt`/`setup.sh` of its own.

## Core concepts

### Dataset

`pipeline.dataset.Dataset` holds everything a workflow stage needs: `images`
(list of HxWx3/4 BGR(A) uint8 arrays — cv2 convention, matches the existing
render/export code), `image_names`, `cameras` (list of `body2colmap.Camera`),
`points_3d`, `resolution`, plus optional `masks`, `reference_image`,
`anchor_image`, `prompt`, `splat_path`, and a free-form `extras` dict for
anything else that needs to round-trip through a save/load (mirrors
`b2c_extras` in the ComfyUI node's `metadata.json`).

Convert to/from other representations (torch tensors, RGB) at step
boundaries — the shared type stays cv2/numpy so it doesn't force every step
to agree on a tensor framework.

### Step

One unit of work. Subclass `Step`, implement `run(inputs, params)`, decorate
with `@register_step("name")`, reference `"name"` from a workflow's `step:`
field. A `Step` must not know or care which `Dispatcher` will call it.

**A step declares the params it accepts**, as a `PARAMS` tuple of `Param`
(`pipeline/step.py`):

```python
@register_step("mask_splat")
class MaskSplatStep(Step):
    PARAMS = (
        Param("filter_size", int, 6, "Bilateral filter diameter", minimum=0),
        Param("dilation", int, 2, "Grow the kept region back out", minimum=0),
        Param("threshold", int, 16, "Opacity cutoff", minimum=1, maximum=255,
              advanced=True),
    )

    def run(self, inputs, params):
        filter_size = params["filter_size"]   # always present, already typed
```

That declaration is the single source of truth for the defaults.
`WorkflowRunner` merges a workflow's overrides onto them and coerces the
result (`Step.resolve_params`) before dispatch, so `run()` reads
`params["x"]` with no fallback and every dispatcher sees a complete dict.
`REQUIRED` as a default means the workflow must supply one; `None` means the
step computes it at runtime (`device`, resolved to cuda-if-available inside
`run()`). `advanced=True` folds a param away in the web UI — the rule is
that a knob which exists because the underlying library has one, rather than
because this pipeline tunes it, is advanced.

`python -m pipeline.cli params <workflow>` prints the workflow's settings and
outputs — the same form the web UI draws — then the effective value of every
step param, defaults included with `--all`. The fastest way to see what a run
will actually use.

`load(params)` / `unload()` are optional hooks for steps that hold expensive
state (GPU weights). They're only exercised by dispatchers that keep an
instance alive across calls (`InProcessDispatcher(keep_loaded=True)`,
`ServiceDispatcher`, `SubprocessPythonDispatcher` via `pipeline/worker.py`
which always calls `load()` then `unload()` around a single `run()`, since
each subprocess invocation is one-shot).

**Import discipline for real implementations**: a step module's own heavy
imports (torch, diffusers, detectron2, mmcv, ...) must be deferred to inside
`load()`/`run()`, not at module top level. `pipeline/steps/__init__.py`
imports every step module unconditionally so the registry is always
complete — including inside an isolated venv's worker process, which only
has *one* step's dependencies installed. A top-level `import diffusers` in
`wan22_vace_denoise`'s module would crash `pipeline.worker` when it's
running inside the `sam3dbody` venv looking up `sam3d_body`.

### Dispatcher

Four implementations, chosen per-step via a workflow YAML's `dispatch:`
field:

| `dispatch:`   | Class                        | Use for |
|---------------|-------------------------------|---------|
| `in_process`  | `InProcessDispatcher`         | No conflicting deps: dataset I/O, camera paths, rendering (pyrender/numpy), RMBG, Sapiens2-lite, and CLI binaries invoked as subprocesses (brush, brush-splat-render) |
| `subprocess`  | `SubprocessPythonDispatcher`  | Conflicting Python deps needing their own venv: SAM-3D-Body (pinned detectron2), Wan2.2 (diffusers), SeedVR2 (own torch/diffusers pins; no flash-attn/apex needed — see below) |
| `service`     | `ServiceDispatcher`           | A step run against a long-lived FastAPI microservice that keeps a model loaded across many workflow runs (batch processing) — not needed for a single interactive pass |
| `docker`      | `DockerDispatcher`            | OS-level isolation a venv can't give (different CUDA/cuDNN userspace, a non-Python runtime) |

`subprocess`/`docker` both use the same file-based IPC protocol:
`pipeline/worker.py` is invoked as `python -m pipeline.worker <step_name>
<inputs.pkl> <params.json> <outputs.pkl>`; inputs/outputs must be picklable
plain data (numpy arrays, dicts, strings) — no live GPU tensors or open
handles crossing the process boundary.

`dispatch:` says *what kind* of isolation; `env:` names an entry in
`envs/envs.yaml` that supplies the *where* (which `python_bin`, which
`image`, which `base_url`) — kept separate so workflow YAMLs stay portable
across machines and only `envs.yaml` needs to change per host.

`WorkflowRunner` caches one dispatcher instance per `(dispatch, env)` pair
for the lifetime of a `run()` call, and calls `.close()` on all of them in a
`finally` block afterward (releases subprocess-held state, HTTP sessions,
etc.).

### Context

Plain dict-of-objects with dotted-path access (`ctx.get("dataset.images")`,
`ctx.set("scene.vertices", value)`). Seeded with `{"dataset": <Dataset>}`
before the first step. A step's `inputs`/`outputs` in YAML are dotted paths
into this — nothing else. This is the entire mechanism that keeps data
in-memory between steps; disk only enters the picture via a `save_dataset`/
`load_dataset` step.

### Workflow YAML

```yaml
name: fast_helical_native

settings:                    # the knobs the web UI draws, in this order
  - name: resolution
    label: Resolution        # what the control is called; defaults to `name`
    type: list               # str | int | float | bool | list; inferred if omitted
    default: [720, 1280]
    choices: [[720, 1280], [600, 1040]]
    help: Frame size, width x height.
    # also: minimum / maximum (a slider), advanced (behind "More settings"),
    # group: outputs (drawn in the Outputs box instead of Settings)

outputs:                     # the deliverables, and the switch each one is
  - name: export_ply         # the global its export steps read via `when:`
    label: Trained .ply
    dir: ply                 # where it lands under output_root
    default: true
    help: A second, normal-supervised brush training.
  - name: export_colmap_preupscale
    label: Pre-upscale COLMAP dataset
    dir: colmap_preupscale
    default: false
    requires: run_upscale    # forced off, and its checkbox disabled, without it

globals:                     # plumbing with no control of its own
  output_root: output/fast_helical_native

steps:
  - id: denoise               # unique within the workflow; also the params namespace
    step: wan22_vace_denoise  # registered Step name (pipeline/registry.py)
    dispatch: subprocess       # in_process | subprocess | service | docker
    env: wan22                 # key into envs.yaml; ignored for in_process
    keep_loaded: false          # in_process only: reuse one Step instance + its load()ed state across calls
    inputs:                     # name -> dotted Context path (read before the call)
      control_video: dataset.images
      control_masks: dataset.masks
      reference_image: dataset.reference_image
      style_hint: scene.style?    # trailing ?: optional, None when nothing wrote it
    params:                      # OVERRIDES on this step's declared defaults
      width: ${globals.resolution.0}
      height: ${globals.resolution.1}
      steps: 6                    # a literal: this step's own knob, not shared
    outputs:                     # step's returned name -> dotted Context path (written after the call)
      images: dataset.images
    when: ${globals.export_ply}  # optional; skip this step when falsy
```

**One namespace, three declarations.** A `settings:` entry, an `outputs:`
entry and a bare `globals:` key all land in one flat namespace — the only
scope `${...}` resolves against, and the only thing `--param x=y` and the web
UI's overrides address. A name declared twice is refused at load. What
separates them is only what the UI is told:

| block | what it is | drawn as |
|---|---|---|
| `settings:` | a knob, declared as a `Param` (same vocabulary as a step param) | a control in the **Settings** box, or the **Outputs** box with `group: outputs`, or behind *More settings* with `advanced: true` |
| `outputs:` | a deliverable: its switch, its `dir:`, and an optional `requires:` | a checkbox in the **Outputs** box |
| `globals:` | plumbing (`output_root`) | nothing — undeclared means undrawable, which is what keeps `output_root`'s repoint safe |

A setting reaches the steps as `${globals.<name>}` written at each step that
reads it — the reference lives where the value is consumed, so it is
greppable from the step end, and `templating.global_ref` uses it to drop the
step-level duplicate from the per-step panel so the setting keeps one
editable home. `validate()` refuses a setting nothing reads, since that is a
control which silently does nothing.

Everything else belongs under the step that consumes it, where it overrides
the default that step's class declares — so a param absent from the file is not unset, it is
at its declared default. That split is what lets one workflow call the same
step twice and configure the two calls apart: `fast_helical_native.yaml` trains
`brush` twice (`train_splat`, `train_final_splat`) and denoises twice
(`denoise_pass1`, `denoise_pass2`), which under the old single flat `params:`
block meant hand-prefixed names like `brush_total_steps` and no way to tune
the two trainings independently at all.

A step override naming a param the step does not declare is refused when the
workflow is validated (`WorkflowSpec.validate`, called by the runner before
step one), not silently ignored.

**An optional read** is a path with a trailing `?`: if nothing has written
it, the step is handed `None` rather than the run failing at that step. It
exists for one shape — a `when:`-gated branch feeding a step that runs
either way — because a gated step's outputs are simply not in the Context
when it is off, and there is otherwise no way to say "take these if they
were built". The shipped case is the face splat's supporting views:
`face_support_views` writes them under `${globals.face_splat}`, and both
brush trainings read them with a `?` — trained without them when
`face_splat: false` turns the branch off. Everything else stays required,
which is what keeps a typo'd path a failure rather than a silently missing
input.

`when:` is what makes a workflow's tail optional. It resolves like any
param value, and a falsy result skips the step entirely — its inputs are
never read, its outputs never written, and `WorkflowSpec.enabled_steps()`
leaves it out, so the model prefetch does not block on a checkpoint only
that step needs. The runner still reports it, as `step_skipped` at its own
index, so a step list built from the YAML lines up with the run.

The case it exists for is the deliverables both `fast_helical` workflows end
with — the `outputs:` block above, whose switches are exactly these `when:`
conditions: the .ply is a full 30,000-iteration brush training, and starting
one you are going to discard is an hour of GPU. `false`, `no`, `off`, `0` and the empty string are all
falsy as *strings* too — a `when:` usually resolves through a param
somebody typed, and `bool("false")` is `True`.

See `pipeline/workflows/fast_helical_native.yaml` for a full multi-step,
from-an-image example.

### envs.yaml

```yaml
envs:
  wan22:
    kind: subprocess          # informational; dispatch: in the workflow is authoritative
    python_bin: envs/wan22/bin/python
  sam3dbody:
    kind: subprocess
    python_bin: envs/sam3dbody/bin/python
  seedvr2:
    kind: docker
    image: b2c/seedvr2:latest
  wan22_service:
    kind: service
    base_url: http://localhost:8001
```

Each isolated venv referenced here needs, out-of-band:
```
uv venv envs/<name>
uv pip install -r envs/<name>/requirements.txt   # that model's own pinned deps
uv pip install -e .                              # this pipeline package, so `pipeline.worker` imports
```

## Running today

```bash
python -m pipeline.cli run fast_helical_native \
    --reference-image path/to/sheet.png -v

# Validate a workflow references only real, registered steps (doesn't execute)
python -c "
from pipeline import steps
from pipeline.workflow import WorkflowSpec
spec = WorkflowSpec.from_yaml('pipeline/workflows/fast_helical_native.yaml')
print(spec.name, [s.step for s in spec.steps])
"
```

Requires `PyYAML` and `requests` (added to `requirements.txt`) plus whatever
`body2colmap`/`numpy`/`opencv-python` already provide `Camera` and image I/O.

## Current state — what's real vs. stubbed

**Real and tested:**
- `Dataset.to_disk()`/`from_disk()` — round-trips correctly, verified against
  a synthetic dataset in a throwaway venv.
- Full `WorkflowRunner` execution path — dispatcher resolution/caching,
  `${globals.x}` templating (including list indices like
  `${globals.resolution.0}`), the merge of a step's declared defaults with
  the workflow's overrides, `Context` get/set, `save_dataset` writing a
  real checkpoint to disk.
- `rmbg` — RMBG-2.0 background removal (`transformers`, in-process). Ran
  against `cyber_6f`'s reference image on an L40S pod, both single-image and
  batched (`inputs["images"]`) paths; mask shape/range correct on both.
- `wan22_vace_denoise` — Wan 2.2 VACE denoise, dual high/low-noise expert +
  VACE conditioning, 6-step/cfg=1/uni_pc-beta distilled schedule, fp8
  (torchao) via `diffusers.WanVACEPipeline`. Ran end-to-end on an L40S pod
  against all 81 frames of `cyber_6f`'s `initial/` dataset at strength=1.0;
  output confirmed correct by the project owner. See that module's
  docstring for what's verified vs. still a param to tune (LoRA strengths,
  attention backend, disk caching of the fused/quantized weights) and why
  frame count matters (a short test clip produces visibly worse output —
  not a bug, just an invalid test size for this model/LoRA pairing).
- `sapiens2_lite` — Sapiens2 surface-normal estimation via transformers'
  first-class support (`AutoModelForNormalEstimation`,
  `facebook/sapiens2-normal-0.8b`) — not the older facebookresearch/sapiens
  (v1) "lite" torchscript path this step's name originally referenced. Ran
  against `wan22_vace_denoise` output frames on an L40S pod, both
  single-image and batched paths; output correctly shaped and L2-normalized.
- every workflow YAML loads and its step names resolve against the
  registry.
- `sam3d_body` — SAM-3D-Body mesh/joint reconstruction. Ran real inference
  on an L40S pod against `cyber_6f`'s `anchor.png`: 18439 vertices, 36874
  faces, 127 joints, focal_length=1468.6px, all correctly shaped. Output
  schema confirmed against `PozzettiAndrea/ComfyUI-SAM3DBody`'s
  `process.py` (the node pack the project's actual ComfyUI flow uses), then
  confirmed directly by the real run. Four real, undocumented bugs found
  and fixed getting this to build and load — see that module's docstring
  and `pipeline/envs/sam3dbody/requirements.txt`'s comment for the full
  list (xtcocotools needing numpy pre-installed with
  `--no-build-isolation`, a CUDA-toolkit version pin detectron2's native
  build needs that a plain `pip install torch` doesn't give you, torch/
  torchvision needing reinstalling together, sam-3d-body needing to be
  vendored rather than pip-installed, `load_sam_3d_body`'s `checkpoint_path`
  needing to be the `.ckpt` file not its directory, and `mhr_path` being
  required despite defaulting to `""` upstream).
- `seedvr2` — one-step diffusion video upscaling. Ran real inference on an
  L40S pod against 5 frames of `cyber_6f`'s `initial/` dataset: a real
  720x1280 -> 1440x2560 upscale (`resolution` is the CLI's target output
  shortest edge, not a multiplier — asking for less than the input's
  shortest edge downscales, which is what the first smoke test
  accidentally did before this was caught and re-run correctly), visibly
  sharper output with no artifacts or color-cast drift. Needed
  `vae_encode_tiled`/`vae_decode_tiled` at this resolution — an
  untiled full-res VAE decode OOM'd a 44GB L40S outright, which is exactly
  what those CLI flags exist for. An earlier version of this module guessed
  the real API entirely wrong: it assumed `_process_frames_core` lived in
  `src.core.generation_utils` with ~3 fields, when it's actually a private
  function in `inference_cli.py` (the vendored repo's own CLI script, not
  a library module) taking ~30 `argparse.Namespace` fields covering every
  CLI flag. Fixed by reading `inference_cli.py` directly and vendoring +
  importing it as a module (confirmed safe: its only module-level side
  effects are `parse_known_args()`, `mp.set_start_method(...)`, and
  `os.environ.setdefault(...)`; `main()` is guarded by `if __name__ ==
  "__main__"`). Also corrected a bad guess in `requirements.txt`:
  flash-attn/apex are NOT required — they're optional accelerators behind
  non-default `attention_mode` values; the default `sdpa` is pure PyTorch.
- `generate_firstlast`/`inject_anchor` (`anchor_stub.py`) — warp the
  sheet's front half to the anchor camera and inject it into the frame batch,
  producing the per-frame reference/denoise mask `wan22_vace_denoise`
  consumes. Pure numpy/cv2 logic with no GPU/model dependency, so verified
  locally against synthetic data (no pod needed): the affine
  (identity-rotation) and full-homography warp paths both produce correctly
  shaped output; anchor position-matching correctly replaces every camera
  within tolerance including an exact-duplicate camera (the
  `overlap=1`-circular-path case where frame 0 and the last frame share a
  position); the no-`anchor_image` case passes inputs through with an
  all-1.0 mask instead of failing. Not yet run against a real `render`
  output (`render` itself is unverified — see below). Both are wired into
  `fast_helical_native.yaml`; the `fast_helical` files use `inject_anchor`
  only, since they start from a dataset that already has its anchor.

- `split_reference_sheet` (`reference_sheet.py`) — new step, no node
  behind it: in ComfyUI the cut lives in the interactive graph that
  produces an `initial/` directory, built from stock image nodes, and none
  of the checked-in API JSONs cover it. Halves the two-panel front/back
  sheet the from-an-image path starts from — front to `sam3d_body` /
  `generate_firstlast` / `rmbg`, back over `dataset.reference_image`, which
  is all `wan22_vace_denoise` conditions on. Pure numpy, verified locally
  (`tests/test_reference_sheet.py`), including over `cyber_6f`'s recorded
  sheet. Note that what that dataset does with its sheet is the older
  convention: `reference.png` there is both panels, joined, and
  `workflows/api/denoise.json` feeds the lot to `WanVaceToVideo`. The back
  half alone is what the current flow conditions on, because the anchor
  injection already puts the front view in the batch as a real photograph.

- `colmap_export` — verified against `cyber_6f/colmap`, the real COLMAP
  directory the ComfyUI stage produced from `cyber_6f/upscaled`.
  cameras.txt and points3D.txt come out byte-identical; images.txt agrees
  to 2.4e-7 per value (the poses have been through metadata.json's float
  round-trip). The exported PNGs are not compared — that stage runs RMBG
  first. Two layouts: `flat` (the default, and what that golden comparison
  is against — frames beside the .txt files, as the ComfyUI stage wrote
  them) and `brush` (`images/` and `normals/` subdirectories), which is
  what all three workflows use for the COLMAP dataset they hand back.
- `mask_splat` — new step, no single node behind it: it collapses the
  eight-node subgraph of `workflows/api/mask_splat.json`. Verified against
  `cyber_6f/splatted` -> `cyber_6f/masked_splatted` at the `fast helical`
  settings: mean absolute error ~0.25/255, max 15, and the two agree on
  which pixels survive to within sub-perceptual rounding at the mask edge.
  Two semantics were established by fitting against that recorded output
  rather than by reading node source, and both are load-bearing —
  `ToBinaryMask` compares strictly greater-than with no rounding (getting
  this wrong pushed max error to 140), and `ImpactDilateMask` uses a plain
  `dilation x dilation` kernel (200). See that module's docstring.
  **Update (2026-08-30):** superseded, and kept anyway. `render_splat`'s
  `confidence` mode makes the same fringe decision properly — once, in 3-D,
  from each Gaussian's multi-view evidence rather than from accumulated
  alpha per pixel per frame — so all three workflows now run this step as
  `mode: passthrough` behind a gated render, and running the old alpha cut
  on top of one would be wrong rather than merely redundant (it would
  re-composite the grey frames over black and smear the gate's soft edge).
  Passthrough is not a no-op: the step's *other* job, replacing the
  per-pixel splat alpha in `dataset.masks` with the per-frame all-1.0 VACE
  batch, is still what `denoise_pass2` reads and `inject_anchor` writes its
  0.0 into — which is also why the step stays, along with the ordering it
  anchors and the `mode: threshold` path that keeps the recorded run
  reproducible for an A/B. See `docs/spatial-reinforcement.md`.
- `views` (`drop_views`/`filter_fov`/`rotate_views`/`replace_views`/
  `merge_datasets`) — verified against `cyber_6f`'s real 81-camera helical
  orbit rather than synthetic data, which matters because the
  skeleton-relative azimuth convention can only be got wrong silently and
  a synthetic orbit built on the same assumption would agree with itself
  either way. The real data also exercises two cases synthetic data would
  have missed: a frame at *exactly* azimuth 0 (the anchor sits on the
  original camera), and exactly one duplicated camera position (the
  `overlap=1` twin, which `rotate_views` splits to first/last).
- `inject_anchor` — now verified against recorded output, not just
  synthetic data. `cyber_6f/initial` records `anchor_position` at the
  world origin, and frames 1 and 81 of that dataset are byte-identical to
  `anchor.png` — so the recorded data independently says which frames the
  ComfyUI flow injected into, and the port finds the same two from camera
  positions alone. Also verified to survive a `rotate_views` reordering,
  which is the whole reason position is the durable key and
  `anchor_frame_index` is informational. **No longer in the bootstrap**
  (2026-08-29): `inject_shell_views` does that job there, for a band of
  frames instead of one. The stage-3 call in the shared tail stays, and
  no-ops on a from-a-sheet run for want of an anchor image. Its partner
  `generate_firstlast` is now wired nowhere — the warp it does was only
  ever for that injection — so its synthetic-data-only verification status
  is frozen where it stands.
- `detect_face_landmarks` — MediaPipe face landmarks (CPU, no pod), plus
  the matching `face_landmarks` input and `face_mode`/`face_max_angle`
  params on `render` that consume them. Verified against `cyber_6f`'s real
  photos: 478 landmarks off the anchor photo through the crop-and-retry
  fallback (the face is a small part of a full-body frame, so the
  full-image stage finds nothing), and on the two-panel front/back
  reference sheet the frontality scoring correctly picks the front-facing
  subject in the left panel over the back of the head in the right. Note
  `face_mode` is the drawing style ("full" = points + connectivity lines |
  "points" | "none"); the view-angle gate is the separate
  `face_max_angle` (90 = full hemisphere, 45 = near-frontal).
- `map_face_to_mesh` + `fit_head_to_face` (`head_fit.py`) — the mesh head
  re-fitted to the photograph's face. The first renders the mesh head,
  shaded, through SAM-3D-Body's camera, runs MediaPipe on the render and
  snaps each of the 468 landmarks to the nearest visible vertex (main env;
  446/468 at 0.8 px on cyber2_6f). The second (sam3dbody env) replays the
  MHR body model with the neck/head rotations, the head joint's scale and
  the head-only shape components free, minimising the 2D distance between
  those vertices and MediaPipe's landmarks on the photo: 10.5 → 2.4 px rms
  on cyber2_6f, head pitched 7° down to where the photo looks, and it
  publishes updated `pose_params` (plus a `scale_offsets` vector) so the
  parameters keep describing the mesh. Replaces `head_angle_fix` in the
  native prologue — the auto nod turned that head 30° away from the photo
  (20.7 px). Why not a vertex deformation: fitted densely it either pitches
  the head 23° to hide a too-long mid-face or squashes the skull to fit it;
  see the module docstring. Geometry tests only; the fit needs the gated
  checkpoint.
- `head_angle_fix` (`head_angle.py`) — SUPERSEDED in the native prologue by
  `fit_head_to_face` (incompatible with it: this one deforms vertices
  without updating the pose parameters the fit replays). Stopgap for a systematic SAM-3D-Body
  failure: fitted from a frontal photo the head comes out craned forward
  (neck-to-head vector 30-45 deg off the torso axis). One weighted rigid nod
  about the inter-shoulder axis through the MHR70 neck joint, smoothstep-
  graded 0 at the neck to full at the crown. **Vertices and keypoints get
  the identical transform** (one `_bend` closure applied to both), so the
  skeleton overlay still tracks the silhouette — verified on a real fit:
  every joint stays inside its own mesh through the nod (median +24 -> +25
  mm behind the front surface, 0% outside). It does NOT update the MHR pose
  parameters, which is why it cannot be combined with
  `refine_pose_to_splat` — see `INCOMPATIBLE_STEPS` in `workflow.py`.
  Synthetic-skeleton tests only.
- `pointmap_splat` (`pointmap_splat.py`) — one photo into a feed-forward
  Gaussian shell, in SAM-3D-Body's own world: Sapiens2 pointmap, depth
  re-solved from `sapiens2_lite`'s normals, one oriented Gaussian per
  foreground pixel (matte from `rmbg`, not a seg head). Ported from
  ~/Projects/masktest, then PLACED in the mesh's camera: integrated in the
  pointmap's own fitted camera (masktest verbatim), rotated onto SAM-3D-Body's
  rays, and put on the ray through each source pixel with the relief scaled
  by the width ratio `s·f_pointmap/f_sam`, so the shell reprojects onto the
  source photo by construction AND keeps the network's shape. (Until
  2026-08-30 the pointmap camera was discarded outright; on a face crop that
  nodded the face 12.6° and stretched its relief 3.2× — the bug that got the
  whole line removed and reverted the same day.) Run end to end locally:
  480k Gaussians, and the anchor gate (best-fit shift against
  `generate_firstlast`'s warp of the photo) lands at (1, 0) px. Wired into
  `fast_helical_native.yaml`'s bootstrap, which is the thing it was written
  for: photographic reference views for VACE around the source view, and a
  non-degenerate helical first pass. Never run on a pod.
- `face_pointmap_splat` (`pointmap_splat.py`) — the same step against a
  head crop, and the reason `PointmapSplatStep` is now a base class with two
  registered specializations rather than one step. It differs in exactly two
  things: `_source_intrinsics`, which re-expresses SAM-3D-Body's camera on
  the crop's pixel grid so a crop pixel still unprojects onto the ray through
  the full-image pixel it was cut from, and two measured defaults
  (`splat_scale` 0.5, `fill_max_frac` 0.05 — a segmentation mask of a head
  fragments without hole filling where an RMBG matte of a body must never be
  filled). Fed by `sapiens2_seg` (a Face_Neck mask, `parts` configurable) and
  `crop_to_box`. Synthetic tests only; the gate is
  `tests/test_face_splat.py::TestCropIsANoOpOnTheRays`.
- The `...+splat` render modes (`render.py`) — alpha-composite that face
  onto the skeleton drawings on every frame within `splat_max_angle_deg` of
  the source view, which at the shipped 60° is roughly two thirds of the
  orbit against the shell band's 15. This is body2colmap's own
  `skeleton+splat` mode: the layer goes into the `render_composite` call
  this step already made, and `Renderer._composite_splat` blends it. The
  cull angle is that mode's constant widened, and passed explicitly for
  that reason — 45 is where body2colmap measured a Face_Neck shell still
  reading cleanly, 60 is where it is mostly open rim, and running to 60
  buys the views least like the photograph (the ones
  `select_support_views` supervises) at the price of that rim.

  **Until 2026-08-31 this was a pair of b2crunner steps** —
  `render_face_views` (a `render_splat`) plus `composite_splat_views` in
  `anchor_stub.py` — because body2colmap's mode rasterised through gsplat,
  which this project dropped to get rid of its runtime CUDA toolchain. The
  library now shells out to the same `brush-splat-render` binary, so the
  detour is gone; see `docs/revert-when-body2colmap-drops-gsplat.md` for
  what it was and what came out. The rasterisation is the library's too, as
  of the same day: `steps/splat.py`'s `_rasterize` drives
  `SplatRenderer.render_many` rather than carrying a parallel copy of it,
  and `render_splat_layers` is a thin wrapper over that. What stayed on this
  side is the crash report (`on_fault`), the throttled log relay
  (`on_output`) and the frame naming — the three things that are this
  project's rather than the renderer's. body2colmap grew those three seams
  plus `ply_path` for it; see its `body2colmap/CLAUDE.md`.
- `select_support_views` (`anchor_stub.py`) — the face splat's second route
  into a training, and the consumer of `brush`'s `support_*` inputs. It
  takes `render_face_support_views`' **cap** render (36 views sampled
  around the photograph's own view of the splat — a different view set
  from the drawings' own cameras), drops the ones that
  land on the training's denoising path, un-premultiplies the rest back to
  straight alpha, and hands them to the **stage-2** brush training as
  **masked** views — evidence that counts where the splat's own coverage
  says to and is ignored everywhere else. The composited copy goes through
  two diffusion passes and comes out as the denoiser's idea of the face;
  this copy does not.

  Two renders, because the two consumers want different views. The
  composite runs on the dataset's own cameras, one layer per drawing. The
  supporting views want the opposite: a view on the denoising path already
  has a denoised frame carrying the photograph, so a render of it
  supervises with a resampled copy of what the training has. The **outer**
  edge of the useful band is therefore drawn by the sampler
  (`cap_radius_deg`, 30° — body2colmap's measurement of where a Face_Neck
  shell still reads cleanly), and only the **inner** one is drawn here:
  `min_path_angle_deg`, 5°, measured to the nearest camera on
  `path_cameras`. Not a hole punched at the source view — a band swept
  along the whole path, because every frame on it is a denoised view in
  its own right, 140° round the orbit no less than at the anchor. On a
  circle that comes out as the elevation difference; on the shell file's
  helix it follows the sweep. Of a 30° cap of 36 against an 81-camera
  ring, 29 survive.

  There used to be a second way to draw the outer edge here — a
  `view_roles` input carrying `composite_splat_views`' per-frame verdict,
  and a `max_angle_deg` culling on the angle that step had measured. Both
  were already off in the shipped wiring because the cap drew the edge,
  and both left with that step on 2026-08-31.

  The **final** training takes no supporting views at all: by then the
  dataset is the helical re-render, denoised a second time and upscaled,
  and these renders are the bootstrap's. `train_splat` reads them through
  an optional `?` path, so `face_splat: false` trains exactly as before.
  `tests/test_support_views.py`.
- `refine_pose_to_splat` (`pose_refine.py`) — re-poses a SAM-3D-Body fit so
  its mesh agrees with that shell in novel views, which is what stops the
  skeleton overlay drifting off the subject a few degrees either side of the
  anchor. Adam over `body_pose_params`/`global_rot`/`cam_t` through the MHR
  body model's own differentiable forward, with shape/scale frozen so bone
  lengths are preserved by construction. Measured: skeleton pixels landing
  off the splat 5.1% -> 2.8% over +-19 deg (11.3% -> 5.2% at 28 deg), anchor
  drift 1.79 px median. Runs in the `sam3dbody` venv, since it needs that
  model. Tests cover the depth buffers, target self-consistency and the PLY
  reader; the optimisation itself needs the gated 2.8 GB checkpoint and is
  not exercised in CI.
- `inject_shell_views` (`anchor_stub.py`) — the step that spends the shell.
  Takes the mesh render and a `render_splat` of the shell along the *same*
  cameras (`pattern: ""`), and swaps the shell's frames in wherever the
  camera is within `replace_radius_deg` of the anchor's — an angle on the
  orbit sphere, so a helix's elevation counts like azimuth. It also writes
  the VACE mask batch: 1.0 for everything by default, since a shell render
  is not a photograph, with a second `reference_radius_deg` band available
  for marking the closest frames 0.0 ("keep it") once somebody has looked
  at them — which is what `fast_helical_native.yaml` now does, because
  there this step **replaced `generate_firstlast` + `inject_anchor`**: it
  puts photo-derived content on every frame near the source view rather
  than only the one exactly on it, so the render no longer bends its path
  (`override_cam_from_mesh: false`) and nothing warps the photo. At that
  source view the two agree to a best-fit shift of (1, 0) px. Synthetic
  camera paths in `tests/test_shell_views.py`, plus two golden ones against
  the smoke run's recorded manifest; no diffusion pass has yet been
  conditioned on a batch it produced.
- `load_splat`/`save_splat`/`render_splat` — `render_splat`'s camera-path
  resolution (which cameras, what focal length, which bounding box frames
  the orbit, point-cloud preservation, metadata pass-through) is verified
  against `cyber_6f`'s recorded metadata: the dataset's `focal_length_mm`
  reproduces its cameras' `fx` exactly, and an anchored override path
  rebuilt from its `orbit_target` lands a camera exactly on its recorded
  `anchor_position`. **Update (2026-08-23):** rasterisation shells out to
  the `brush-splat-render` binary, not gsplat (see
  `pipeline/steps/splat.py`) — now verified on a local RTX 4070 Ti, both as
  a raw binary call and through the step itself against all 81 of
  `cyber_6f`'s real cameras (real content, correct shapes/dtypes back). See
  `docs/docker-build-notes.md` for the full result.
  **Update (2026-08-30):** `bounds_source` picks which box sizes and aims a
  new orbit. The default, `dataset`, is the source render's `framing` box —
  what keeps a re-render lined up with the render it replaces, and what
  every shipped workflow uses. That is only right while the splat IS what
  the dataset was framed around, which the face splat is not: it is a
  head-only shell whose world centre sits well above a full-body orbit
  target, so orbiting the body's box aims at the chest and sizes a radius
  for a whole person, landing the head as a small smudge mid-frame.
  `bounds_source: splat` ignores `framing_bounds` and takes
  `scene.get_bounds()` instead — a plain min/max over the Gaussian means,
  so its centre is the same quantity `face_splat_stats.world_center`
  publishes. **Only the radius moves.** The intrinsics are deliberately
  untouched, because `ColmapExporter` writes a single camera line for a
  whole training set (`cameras[0]`, stamped `CAMERA_ID 1` on every image),
  so a per-view focal length would be silently discarded and those views
  trained at the wrong lens; getting closer is `compute_auto_orbit_radius`
  dollying in on the smaller box at the same lens, with `fill_ratio` the
  knob. It is an error rather than a warning to combine it with no
  `pattern` or with `override_cam_from_mesh` — neither of those branches
  computes a box at all, so the request would otherwise be a silent no-op,
  which is the exact failure the param exists to prevent.
  **Update (2026-08-30):** `confidence: true` gates the render on the
  per-Gaussian multi-view evidence `brush` now writes into the .ply
  (`export_evidence`), replacing the `mask_splat` stage that used to
  threshold rendered alpha afterwards. It changes the binary's output
  contract, which is the part to be careful with: the RGB is composited
  over `cull_color` (0.5 grey) rather than `bg_color`, `--background` is
  ignored and therefore not passed, and the alpha that comes back is the
  gate `smoothstep(gate_lo, gate_hi, C)` rather than accumulated opacity.
  Downstream nothing changes — foreground is still 1 — but a transparent
  pixel is grey, not black, so the mode must stay OFF for any render that
  feeds `select_support_views` (which enforces premultiplied-over-black and
  refuses it; the `+splat` compositing passes its own background and cannot
  be got wrong this way). `confidence_sidecar` keeps the raw per-pixel confidence
  under the log dir for tuning; `conf_args` passes `--conf-*` flags
  through verbatim. Argv-level tests only (`tests/test_splat.py`,
  `tests/test_workflows.py`) — the gating itself is the renderer's, and
  has not been through this pipeline on a pod.

**Real but UNVERIFIED:**
- `brush` — Gaussian-splat training via the `Erant/brush` CLI (COLMAP
  export, image/normal-map export, subprocess invocation — a close port of
  `nodes/brush_node.py`'s `Body2COLMAP_RunBrush`). `dispatch: in_process`
  (targeting RunPod, where a pod is a single container — no nested Docker
  daemon to split brush into its own `docker`-dispatched image the way an
  earlier version of this setup did; see `docker/Dockerfile`'s comment).
  **Update (2026-08-23):** verified with a real short GPU training run
  (200 iterations against `cyber_6f/colmap`'s 81-frame export, on a local
  RTX 4070 Ti) inside the actual shipped image — completed in ~3s and
  exported a valid `.ply` (12,463 splats). Getting Vulkan to reach the GPU
  in a container took a real fix (docker-ce + `libegl1`; see
  `docs/docker-build-notes.md`), not just the `NVIDIA_DRIVER_CAPABILITIES`
  bake this paragraph originally described — that variable made no
  measurable difference in the end. Whether RunPod's provisioning needs
  anything beyond what this image already does is still open.
  **Update (2026-08-25):** a non-zero exit is no longer trusted on its own.
  brush has been seen taking SIGSEGV (exit -11) during shutdown with the
  export already complete, so the step weighs a failed exit against the
  artefact: an export that exists, is non-empty, and was written by this
  run (mtime changed — a stale `.ply` in a reused `export_dir` does not
  count) means the training succeeded, and the whole failure is logged at
  WARNING instead. Anything else still raises. See
  `tests/test_brush_exit.py`.
  **Update (2026-08-30):** `export_evidence` (on by default) has brush
  measure each Gaussian's multi-view evidence after the last step and write
  it into the exported `.ply` as seven `ev_*` vertex properties. That is
  what `render_splat`'s `confidence` mode reads, and having it in the .ply
  is what lets that render need no second pass over the dataset. It costs
  seconds and every other .ply reader ignores the properties, so both
  trainings do it. `evidence_prune_inmask` (off) would drop under-supported
  splats from the export itself and stays off until it has been looked at
  on a real run; `evidence_normal_weight` (0) is untuned. Argv-level tests
  in `tests/test_brush_evidence.py`. Needs a `brush` built past the
  confidence work on `normal-map-supervision` — an older binary writes no
  `ev_*`, which the renderer reports as a warning and falls back from, so a
  mismatched image degrades loudly rather than silently.
- `render` — camera-path generation (circular/sinusoidal/helical,
  `override_cam_from_mesh` anchor mode) + mesh/depth/skeleton rendering +
  point-cloud sampling, ported from `nodes/render_node.py`. The geometry
  work is entirely `body2colmap`'s (`Renderer`, `Scene`, `OrbitPath`,
  `path`/`utils` helpers) — this module is a thin adapter, same as
  `sam3d_body.py`/`brush.py` are for their libraries. **Update
  (2026-08-23):** `pyrender`'s EGL path is confirmed hitting real hardware
  on a local RTX 4070 Ti — `GL_VENDOR: NVIDIA Corporation`, `GL_RENDERER:
  NVIDIA GeForce RTX 4070 Ti`, not the Mesa/OSMesa software fallback. The
  render call itself (mesh/depth/skeleton output against a real dataset)
  has not been separately exercised yet. **Load-bearing fix
  found while porting, not obvious from any docstring**: `sam3d_body.py`'s
  own `joints` output key (`pred_joint_coords`, 127 joints) is NOT what
  `body2colmap.Scene.from_sam3d_output` wants — it wants `keypoints_3d`
  (`pred_keypoints_3d`, 70 joints); confirmed by reading
  facebookresearch/sam-3d-body's `mhr_head.py` (`j3d = j3d[:, :70]`) and
  `Scene._infer_skeleton_format`'s joint-count-based format lookup
  (70 -> `"mhr70"`). Using `joints` here would either crash or silently
  mis-render. See `render.py`'s own docstring for the full story.
- `colmap_export` — standalone COLMAP sparse-reconstruction directory
  export (`cameras.txt`/`images.txt`/`points3D.txt` + frame PNGs +
  optional `normals/`), ported from `nodes/export_node.py`. Reuses the
  same image/mask/normal-map compositing logic `brush.py` already has
  inline (against its own tempdir) — this is that logic exposed as its own
  step, writing to a permanent directory instead. `body2colmap.
  ColmapExporter` itself is unmodified library code, verified with
  synthetic data locally (cameras.txt/images.txt/points3D.txt all written,
  correct camera count/PARAMS, RGBA + normal files land where expected);
  never run against a real `render` output.

**Stubbed (raise `NotImplementedError`):** none currently.

**Not ported at all:** nothing. Every node in the pack now has a native
counterpart or a documented reason it isn't needed — see the next section.

## Bugs found and fixed while verifying against real data

Four real defects, all of the "silently wrong" kind that a synthetic test
would not have surfaced. Recorded here because each one is a trap worth
not re-entering:

1. **`Dataset.to_disk()` dropped masks.** It wrote `self.images`
   unmodified, while `from_disk()` splits an RGBA frame into BGR + alpha-as
   -mask. So any `save_dataset` checkpoint of a dataset loaded from disk
   lost `dataset.masks` — which, before the denoise stage, is the per-frame
   reference/denoise flag `wan22_vace_denoise` consumes, not a throwaway.
   The ComfyUI save node composites the mask into alpha; this now does too
   (without ComfyUI's inversion, since this pipeline's convention is
   already foreground = 1).

2. **Two step modules imported PIL at module scope.** `sapiens2.py` and
   `wan22_vace_denoise.py` both had a top-level `from PIL import Image`,
   which breaks the invariant this project's own "Import discipline" note
   describes: `pipeline/steps/__init__.py` imports every step module
   unconditionally, so `python -m pipeline.worker sam3d_body` inside the
   `sam3dbody` venv would crash on a dependency that step does not use.
   Now guarded by a static check in `tests/test_import_discipline.py`.

3. **`colmap_export` binarised soft masks.** It computed alpha as
   `np.clip(m * 255.0, 0, 255)` regardless of the mask's range, so a uint8
   mask straight off disk saturated — every value >= 1 became 255,
   throwing away the soft edge. Both mask ranges are now reconciled in one
   place (`pipeline/masks.py`), which is also where bug 1's fix lives.

4. **`render` published four fewer metadata fields than the node it ports.**
   It dropped `orbit_target`, `forward_azimuth_deg`, `framing_bounds` and
   `initial_rotation`, and only published `focal_length_mm` in override
   mode. Not cosmetic: `filter_fov` and `rotate_views` hard-error without
   the first two, and `render_splat` reuses the rest to keep a re-render
   framed identically to the render it replaces. Found immediately on
   porting those steps, which is a decent argument for porting consumers
   and producers close together.

## Coverage vs. the ComfyUI node pack

Gap list between `~/Projects/ComfyUI-Body2COLMAP` (22 registered nodes +
`submit.py` + 7 API-format graphs in `workflows/api/` + 5 pipeline YAMLs)
and what exists here.

### Node-by-node

| ComfyUI node | Native equivalent | State |
|---|---|---|
| `CircularPath` / `SinusoidalPath` / `HelicalPath` | `render` / `render_splat` `pattern:` param | ported; path math verified against real cameras |
| `Body2COLMAP_Render` | `steps/render.py` | ported; **rasterisation untested** (needs headless GL) |
| `Body2COLMAP_ExportCOLMAP` | `steps/colmap_export.py` | ported, **verified against recorded output** |
| `Body2COLMAP_RunBrush` | `steps/brush.py` | ported, unverified |
| `Body2COLMAP_SaveDataset` / `LoadDataset` | `steps/dataset_io.py` | ported, verified against real ComfyUI-written data |
| `Body2COLMAP_GenerateFirstLast` | `steps/anchor_stub.py` | ported; verified on synthetic data only (see below) |
| `Body2COLMAP_InjectAnchor` | `steps/anchor_stub.py` | ported, **verified against recorded output** |
| `Body2COLMAP_LoadSplat` / `SaveSplat` | `steps/splat.py` | ported, verified (PLY round-trip) |
| `Body2COLMAP_RenderSplat` | `steps/splat.py` | ported; camera-path half verified, **rasterisation untested** |
| `Body2COLMAP_MergeDatasets` | `steps/views.py` | ported, verified |
| `Body2COLMAP_DropViews` | `steps/views.py` | ported, verified |
| `Body2COLMAP_FilterFoV` | `steps/views.py` | ported, verified |
| `Body2COLMAP_ReplaceViews` | `steps/views.py` | ported, verified |
| `Body2COLMAP_RotateViews` | `steps/views.py` | ported, verified |
| `Body2COLMAP_UnpackDataset` | n/a — `Context` dotted paths replace it | not needed |
| `Body2COLMAP_Placeholder` | n/a — a graph-rewiring hack `submit.py` targets | not needed |
| `Body2COLMAP_WorkflowComposer` | n/a — replaced by `workflow.py` + `cli.py` | not needed |
| `Body2COLMAP_DetectFaceLandmarks` | `steps/face_landmarks.py` | ported, **verified against real photos** |

Steps here with no single node behind them (they replace whole ComfyUI
subgraphs of third-party nodes): `rmbg` (`RMBG`), `wan22_vace_denoise`
(`WanVaceToVideo` + 2x `KSamplerAdvanced` + ... in `denoise.json`),
`sapiens2_lite` (`SapiensLoader`/`SapiensSampler`), `seedvr2` (the three
`SeedVR2*` nodes), and `mask_splat` (the eight generic image/mask nodes
`mask_splat.json` is built from).

### What is still missing

**Nothing, node-wise.** All 22 nodes now have a native counterpart or a
documented reason not to need one.

**The two rasterisers.** Both now confirmed on real GPU hardware
(2026-08-23, see above): `render` (pyrender/EGL) and `render_splat` — now
`brush-splat-render`, a Rust/wgpu binary, not gsplat/CUDA — the latter
verified both as a raw binary call and through the step itself against all
81 of `cyber_6f`'s real cameras. Everything around both — camera paths,
framing, anchoring, metadata, point clouds — was already verified.

**`generate_firstlast`'s warp.** Verified against synthetic data only. It
cannot be checked against `cyber_6f`: the image it warps is the front view
SAM-3D-Body ran on, and that image is not in the dataset. `reference.png`
there is the whole two-panel front/back sheet, fed to Wan-VACE as-is under
the older convention — a different image with different framing (the
subject's bounding box scales by 0.59 horizontally against 0.84
vertically, so no uniform warp maps one to the other). Current runs split
that sheet in `split_reference_sheet` and keep only the back half as
`reference.png`, so the front view it warps is not persisted there either.

### Orchestration differences (intentional)

`submit.py` does things `cli.py` still does not:

- **Batch datasets**: `submit.py pipeline.yaml ds1 ds2 ...` runs the whole
  pipeline over N datasets; `cli.py` takes a single `--dataset`.
- **Per-stage on-disk checkpoints**: every ComfyUI stage read and wrote a
  named subdirectory, so a crashed run resumed from the last completed
  stage and every intermediate was inspectable. The in-memory `Context`
  design deliberately drops this (see "Why this shape" #3); the equivalent
  is inserting `save_dataset` steps by hand, and there is no
  resume-from-stage mechanism at all. Worth revisiting if long pipelines
  start failing halfway on real hardware.
- **`web/merge_dataset.js`**: dynamic-input UI for `MergeDatasets`. The
  native `merge_datasets` takes a list of Context paths instead; no
  frontend equivalent is needed until the Gradio UI exists.

### Pipeline/stage coverage

| ComfyUI stage (`workflows/api/`) | Native | Notes |
|---|---|---|
| `denoise.json` | yes | `wan22_vace_denoise` (verified on a pod) |
| `upscale.json` | yes | `seedvr2` (verified on a pod) |
| `colmap.json` | yes | `rmbg` + `colmap_export` (export verified against recorded output) |
| `mask_splat.json` | yes | `mask_splat` (verified against recorded output) |
| `resplat_helical.json` | yes | `rmbg` + `sapiens2_lite` + `brush` + `load_splat` + `render_splat` + `inject_anchor` |
| `resplat_tiered.json` | yes | as above plus `merge_datasets` |
| `outline.json` | yes | adds `filter_fov` + `rotate_views` + `replace_views` |

Every stage of every ComfyUI pipeline YAML now has a native equivalent, and
`pipeline/workflows/fast_helical_native.yaml`'s tail (from `denoise_pass1`
on) is the full six-stage port of `workflows/pipeline/fast helical.yaml`.
**None of it has been executed
end-to-end** — that is the next milestone. `brush` and `render`'s
rasterisers are now individually verified on GPU (2026-08-23, a local RTX
4070 Ti — see `docs/docker-build-notes.md`); the remaining unexecuted
pieces are `render_splat`'s `brush-splat-render` call and the full
sequencing itself.

## Adding a new step (checklist)

1. Add a class to `pipeline/steps/` (new file or alongside related steps),
   subclass `Step`, implement `run()`. Defer heavy imports into
   `load()`/`run()`.
2. Declare a `PARAMS` tuple of `Param` for everything `run()`/`load()` reads
   out of `params`, with a default, a one-line `help=`, and `advanced=True`
   for the knobs nobody tunes. Read them as `params["x"]` — the runner has
   already merged the defaults in. `tests/test_step_params.py` checks that
   every registered step declares its params.
3. `@register_step("your_name")` on the class.
4. Import the new module from `pipeline/steps/__init__.py`.
5. Reference `"your_name"` from a workflow YAML's `step:` field, pick
   `dispatch:`, and if not `in_process`, add an `env:` entry to
   `envs/envs.yaml` and build that venv/image out-of-band. Only the params
   that differ from your defaults need to appear there.
6. If the step needs its own isolated venv, `uv pip install -e .` this
   `pipeline` package into it so `python -m pipeline.worker` is importable
   there.

## Suggested next steps

**Update (2026-08-23):** `docker/Dockerfile` now builds
(`docs/docker-build-notes.md`), and a local RTX 4070 Ti closed the
container-graphics question directly — `brush`, `render`, and now
`render_splat` (via `brush-splat-render`, replacing gsplat) are all
individually verified on real GPU hardware, the last one both as a raw
binary call and through the actual pipeline step against all 81 of
`cyber_6f`'s real cameras. That reframes the list below: it's no longer
"everything needs a GPU pod", RunPod-specific provisioning questions aside.

1. **Run `fast_helical_native.yaml` end to end.** Every stage now has a
   native step, each has run individually, but the full sequence never
   has. Expect to debug it stage by stage — insert `save_dataset` steps
   between stages while doing so, since there is no resume-from-stage
   mechanism. Along the way, unskip the PLY round-trip tests (now that
   `plyfile` is the only splat dependency this repo needs — see
   `requirements.txt`) and check `generate_firstlast` against a real
   `render`/`render_splat` output — its warp is the last piece still
   verified against synthetic data only (`cyber_6f` cannot cover it — see
   "What is still missing").
2. **Confirm whether RunPod's pod-creation path needs anything beyond what
   this image now does.** The local fix was `docker-ce` +
   `libegl1`(already in the image) — `NVIDIA_DRIVER_CAPABILITIES` turned
   out to make no measurable difference in any mode tested locally, which
   revises the open question from `docs/docker.md` down to "does RunPod's
   toolkit need the same driver-library completeness this box needed", not
   a capabilities-flag question.
3. **DONE — Wan weights: 81 GB read twice per run -> 47 GB read once.**
   `fast_helical_native` invokes `wan22_vace_denoise` twice
   (`denoise_pass1`, `denoise_pass2`), and the two cannot be merged: brush
   training, splat render, `inject_anchor` and `mask_splat` sit between
   them. Fixed from both ends:
   - **The bf16 path was deleted.** The step now loads the pre-quantized
     fp8 checkpoint only (17.58 GB x 2 in place of 69.36 GB of bf16
     transformers) and does no LoRA fuse and no quantize. `fused_cache_dir`
     and `quantize` are gone with it. `pipeline/models.py` moved in step —
     the wan22 patterns no longer pull `transformer*/` (keeping only
     `transformer/config.json`, which `wan_fp8.load_config` needs), and a
     `wan22_fp8` source was added: 82.2 GB -> 48.3 GB of prefetch.
     `pipeline/envs/wan22/setup.sh` now drives the registry rather than
     duplicating its pattern list, which is how it drifted in the first
     place.
   - **`pipeline.worker` can stay resident.** `keep_loaded: true` on both
     denoise steps means one worker process serves both, so the weights
     come off the network volume once. `build_dispatcher` used to drop
     `keep_loaded` on the floor for subprocess dispatch; that was the bug.
   - **Residency is DRAM, not VRAM.** `Step.release_vram()` (no-op by
     default) evicts the card while keeping weights in host RAM, and the
     resident worker calls it after *every* job — because brush needs the
     GPU between the two passes, and a worker squatting ~35 GB of VRAM
     there would be strictly worse than the reloading it replaced. Full
     eviction stays `unload()`, at shutdown or on a load-param change.
   What is NOT done: the fp8 path has never produced real frames end to
   end (verified at key-mapping, load, forward-pass and LoRA level only).
   **Look at the first pod run's output before trusting a long one** —
   there is no bf16 fallback behind it any more, by design.
4. **Verify the resident worker and the fp8 path on a real pod.** Both are
   green locally and unit-tested, but neither has run against real weights
   on a network volume. Specifically worth watching on the first run:
   that `denoise_pass2` logs no checkpoint download; that `brush` gets a
   clean card after `denoise_pass1` (the `release_vram()` contract — a
   `CUDA out of memory` in brush is the signature of that failing); that
   host RAM holds ~47 GB of resident weights across the middle of the run
   without swapping; and that the denoised frames actually look right,
   since the fp8 path has no bf16 fallback behind it.
5. **Decide what replaces `submit.py`'s batching and per-stage
   checkpointing** (see "Orchestration differences"). Deferred
   deliberately; revisit once long pipelines actually run.

   **When this happens, do the text-encoder work with it.** After the fp8
   switch the T5 encoder is the second-largest thing the denoise step
   pulls — `text_encoder/` is 11.36 GB of the remaining 47, a ~5.7B-param
   UMT5EncoderModel in bf16 — and all it does is turn a prompt into an
   embedding. It is loaded, used for a few hundred tokens, and then sits in
   memory for the whole run. `fast_helical_native` also sets `cfg: 1.0` (the
   Lightning distill LoRA is what makes 6-step cfg-1.0 sampling work), so
   there is no classifier-free guidance and the negative prompt is encoded
   and then contributes nothing at all.

   Batching is what makes the fix clean rather than fiddly. Each dataset
   carries its own prompt (`subject_desc: dataset.prompt`, substituted into
   `$SUBJECT_DESC$`), so a batch of N inputs is N prompts — encode all of
   them in one pass up front, **then discard the encoder before the
   transformers are loaded**, and its 11.36 GB is neither resident nor
   re-read for the rest of the batch. Doing this per-single-run instead
   would mean caching embeddings keyed by prompt, which is more machinery
   for less benefit; the batch case gets it almost for free.

   Note this interacts with the resident worker: `LOAD_PARAMS` on
   `Wan22VaceDenoiseStep` currently treats `prompt` as a per-call param
   (correct today — the two passes share one loaded pipeline). If prompt
   encoding moves ahead of the transformers, the split between "what
   load() builds" and "what run() takes" moves with it.
6. **The Gradio frontend last** — mechanical once the model steps behind
   it work.
