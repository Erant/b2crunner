# Native pipeline

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline. This
is the foundation for porting the node pack into a Dockerized, Gradio-fronted
tool that runs native PyTorch inference instead of routing through a ComfyUI
graph. It replaces `submit.py` + the ComfyUI API-format JSON graphs in
`workflows/api/` with plain Python: YAML workflows made of named `Step`s,
executed by a `Dispatcher` that hides *how* each step actually runs.

Status: the orchestration engine works end-to-end (verified: dataset
round-trip, template resolution, dispatcher caching, context wiring — see
"Current state" below). None of the actual ML models are ported yet; they
exist as documented stubs in `steps/stubs.py`.

## Why this shape

Three requirements drove the design, all from direct conversation with the
project owner:

1. **Modularity** — a step should look identical to the dispatcher whether it
   runs in-process, in an isolated subprocess venv, against a warm HTTP
   service, or inside a Docker container. Swapping a step's execution
   mechanism is a one-line YAML edit, not a code change.
2. **Research-project flexibility** — workflows are human-edited YAML, not
   code. Parameters (resolution, diffusion steps, cfg, seed, ...) are
   declared at the workflow level and templated (`${params.x}`) into
   individual step configs, so trying a new resolution or step count doesn't
   require touching Python.
3. **In-memory by default** — datasets pass between steps as plain Python
   objects in a shared `Context`. Nothing touches disk unless a workflow
   explicitly includes a `save_dataset` step. This is a deliberate reversal
   from the ComfyUI-era pipeline, which persisted to disk at every stage
   boundary (see `workflows/pipeline/*.yaml` + `submit.py`).

## Module map

```
pipeline/
├── dataset.py         Dataset — in-memory dataclass; to_disk()/from_disk()
│                      match the on-disk layout ComfyUI's Save/Load Dataset
│                      nodes already use (metadata.json, pointcloud.npz,
│                      frame_NNNNN_.png, reference.png, anchor.png, prompt.txt)
├── step.py            Step ABC: run(inputs, params) -> outputs, plus
│                      optional load()/unload() lifecycle hooks
├── registry.py         @register_step("name") / get_step_class("name")
├── context.py          Context: dotted-path get/set over a dict of objects
├── templating.py       "${a.b.c}" resolution against a workflow's params
├── workflow.py         StepSpec / WorkflowSpec — the YAML schema; load_envs()
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
│   └── stubs.py         rmbg / sam3d_body / sapiens2_lite /
│                      wan22_vace_denoise / seedvr2 — NotImplementedError,
│                      documented input/param/output contracts
├── envs/
│   └── envs.yaml        Per-machine registry: env name -> {python_bin |
│                      image | base_url}. Empty until real envs are built.
└── workflows/
    ├── roundtrip_example.yaml   in-process-only smoke test (no model deps)
    └── fast_helical_native.yaml target shape for the real pipeline; loads
                                 and validates today, stub steps not runnable
```

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
| `in_process`  | `InProcessDispatcher`         | No conflicting deps: dataset I/O, camera paths, rendering (pyrender/numpy), gsplat, RMBG, Sapiens2-lite |
| `subprocess`  | `SubprocessPythonDispatcher`  | Conflicting Python deps needing their own venv: SAM-3D-Body (pinned detectron2), Wan2.2 (diffusers), SeedVR2 (flash-attn/apex) if the ABI matches the host |
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
params:                      # workflow-level knobs, referenced via ${params.x}
  resolution: [512, 512]
  diffusion_steps: 6
  cfg: 1.0
  seed: 0

steps:
  - id: denoise               # unique within the workflow, used in error messages
    step: wan22_vace_denoise  # registered Step name (pipeline/registry.py)
    dispatch: subprocess       # in_process | subprocess | service | docker
    env: wan22                 # key into envs.yaml; ignored for in_process
    keep_loaded: false          # in_process only: reuse one Step instance + its load()ed state across calls
    inputs:                     # name -> dotted Context path (read before the call)
      control_video: dataset.images
      control_masks: dataset.masks
      reference_image: dataset.reference_image
    params:                      # step call params; may reference ${params.*}
      width: ${params.resolution.0}
      height: ${params.resolution.1}
      steps: ${params.diffusion_steps}
    outputs:                     # step's returned name -> dotted Context path (written after the call)
      images: dataset.images
```

See `pipeline/workflows/fast_helical_native.yaml` for a full multi-step
example and `pipeline/workflows/roundtrip_example.yaml` for the minimal
smoke-test shape.

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
# Smoke test — no model deps needed, proves the plumbing end-to-end
python -m pipeline.cli run pipeline/workflows/roundtrip_example.yaml \
    --dataset path/to/existing/b2c_dataset -v

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
  `${params.x}` templating (including list indices like
  `${params.resolution.0}`), `Context` get/set, `save_dataset` writing a
  real checkpoint to disk.
- `fast_helical_native.yaml` loads and its step names resolve against the
  registry (dataset_io + stubs all import cleanly).

**Stubbed (raise `NotImplementedError`, contract documented in
`steps/stubs.py`):**
- `rmbg` — RMBG-2.0 background removal. Plain PyTorch, no compiled
  extensions → should be `in_process`. Best first target: proves the
  in-process path needs no isolation machinery at all.
- `sam3d_body` — SAM-3D-Body mesh/joint reconstruction. Needs a pinned
  `detectron2` build (`--no-build-isolation --no-deps`) → `subprocess`, own
  venv. Gated HF model access — license must be accepted by a human before
  any automated download will work.
- `sapiens2_lite` — Sapiens2 normal-map estimation via its pure-PyTorch
  "lite" path (avoids the full mmcv/OpenMMLab install). Candidate for
  `in_process` once ported.
- `wan22_vace_denoise` — Wan 2.2 VACE denoise, dual high/low-noise expert +
  VACE conditioning, 6-step/cfg=1/uni_pc-beta distilled schedule. Maps to
  `diffusers.WanVACEPipeline(transformer, transformer_2, boundary_ratio=...)`.
  Target checkpoint: `linoyts/Wan2.2-VACE-Fun-14B-diffusers`; speed LoRAs:
  `lightx2v/Wan2.2-Lightning` (separate high/low-noise rank-64 LoRAs).
  **Open verification items before this can be trusted**: exact LoRA
  filename/variant match to the live ComfyUI graph, mask polarity convention,
  `boundary_ratio` tuning to replicate the step-2-of-6 expert handoff, and a
  VRAM/offload strategy for two 14B experts (~56GB bf16 combined) plus the
  UMT5 text encoder on a single 48GB card (fp8 quantization, sequential
  expert load/unload, or CPU offload — pick one deliberately, don't let it
  silently OOM).
- `seedvr2` — video upscaling. Needs `flash_attn==2.5.9.post1` + `apex`
  built from source pinned to torch 2.4.0 + CUDA 12.1/12.4 →
  `subprocess` if that ABI matches the host, `docker` otherwise.

Not yet started at all: RMBG/MediaPipe face-landmark porting (though
`nodes/face_landmarks_node.py` is already ComfyUI-independent and should
port almost as-is), pyrender-based rendering as a `Step`, COLMAP export as a
`Step`, Brush invocation as a `Step` (already a subprocess call internally in
`nodes/brush_node.py` — wrap, don't rewrite), gsplat re-render as a `Step`,
the actual `Dockerfile`s / `docker-compose.yml`, and the Gradio frontend.

## Adding a new step (checklist)

1. Add a class to `pipeline/steps/` (new file or alongside related steps),
   subclass `Step`, implement `run()`. Defer heavy imports into
   `load()`/`run()`.
2. `@register_step("your_name")` on the class.
3. Import the new module from `pipeline/steps/__init__.py`.
4. Reference `"your_name"` from a workflow YAML's `step:` field, pick
   `dispatch:`, and if not `in_process`, add an `env:` entry to
   `envs/envs.yaml` and build that venv/image out-of-band.
5. If the step needs its own isolated venv, `uv pip install -e .` this
   `pipeline` package into it so `python -m pipeline.worker` is importable
   there.

## Suggested next steps

1. Port `rmbg` for real (in-process) — cheapest way to validate a full
   real-model step against the `Dataset`/`Context` plumbing before touching
   anything that needs isolation.
2. Port `sam3d_body` next, since nearly everything downstream depends on its
   output shape (vertices/faces/joints) — needs the HF license accepted
   first (human step, can't be automated).
3. Wire up rendering + COLMAP export as `Step`s reusing the existing
   `core/`/`nodes/render_node.py` logic (already largely ComfyUI-independent
   numpy/pyrender code — mostly needs `folder_paths`/`comfy.*` calls
   stripped).
4. Only then tackle `wan22_vace_denoise` — highest-risk piece, resolve the
   four open verification items above against a real reference dataset
   before trusting its output.
5. `seedvr2` and the Docker/Compose/Gradio layer last — they're mechanical
   once the model steps behind them work.
