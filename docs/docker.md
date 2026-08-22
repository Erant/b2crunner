# Docker container guide

Real, buildable artifacts live in `docker/`:

- `docker/Dockerfile.main` — everything except brush: the orchestrator
  (`python -m pipeline.cli`) plus one venv per `subprocess`-dispatch step
  (`wan22`, `sam3dbody`, `seedvr2`) and one shared `venv_main` for
  `in_process`-dispatch steps (`rmbg`, `sapiens2`) — mirrors
  `scripts/pod_bootstrap.sh`'s venv layout, just inside a container instead
  of on a bare pod.
- `docker/Dockerfile.brush` — brush, alone, because it needs OS-level
  Vulkan/graphics capability no Python venv (in this image or any other)
  can provide — see "Why brush is a separate image" below.
- `docker/docker-compose.yml` — builds both; `brush`'s image is invoked
  per-call by `pipeline/dispatch/docker.py`, not run via `docker-compose
  up` (it has no long-running process), so it's tagged `profiles:
  [build-only]`.

This file is the narrative — *why* each piece looks the way it does, and
the gotchas that aren't obvious from reading the Dockerfiles alone. When
they disagree, the Dockerfiles are the source of truth; update this file to
match, not the other way around.

## Quickstart

```bash
export HF_TOKEN=hf_...
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm pipeline \
    run pipeline/workflows/fast_helical_native.yaml --dataset /data/some_dataset -v
```

Both images mount a shared `/data` volume (weights, HF cache, datasets you
bind-mount in). The `pipeline` service also mounts the host's Docker socket
so its `DockerDispatcher` calls can shell out to `docker run` for the brush
step — meaning the `pipeline` container and the Docker daemon building
`b2c/brush:latest` must be on the same host, not a remote Docker context.

**UNVERIFIED**: neither Dockerfile has actually been built end-to-end as of
this writing — see "What's confirmed vs. not" below for exactly which
pieces come from real pod-tested commands vs. are still best-guess.

## Why two images, not one

`pipeline/README.md`'s dispatch table draws the line: `subprocess`
(isolated venv, shared host CUDA/system userspace) for steps whose only
conflict is Python dependencies; `docker` (full container boundary) for
steps needing OS-level isolation a venv can't give. Every model step this
pipeline has today (`wan22`, `sam3dbody`, `seedvr2`, plus the in-process
`rmbg`/`sapiens2`) only ever conflicts at the Python-dependency level, so
they all fit in one image as separate venvs — that's `Dockerfile.main`.

`brush` is the one exception, and it's not a close call. Confirmed on a
real RunPod pod: the NVIDIA driver's Vulkan `.so` files
(`libGLX_nvidia.so.0`, `libnvidia-glcore.so.*`) and the ICD json can all be
present, and `vulkaninfo` still fails with `ERROR_INCOMPATIBLE_DRIVER` /
"Found no drivers!" if `NVIDIA_DRIVER_CAPABILITIES` doesn't include
`graphics` (and usually `display`) — that env var is read by
`nvidia-container-toolkit` at *container-creation time* to decide which
driver components to actually expose, and most ML host/pod templates only
set `compute,utility` since they're built for training/inference, not
rendering. Setting it inside an already-running container doesn't
retroactively fix anything. No Python venv, on any dispatch, changes what
capabilities the container was started with — only baking
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display` into the
image itself (`Dockerfile.brush`) fixes it, since that's what
`docker run --gpus all` (`pipeline/dispatch/docker.py`, which already names
brush as its motivating case) reads at container start regardless of the
host's own default.

## What's confirmed vs. not

Real, pod-verified (from this project's actual pipeline steps, run against
real inference on an L40S RunPod pod — see `pipeline/README.md`'s
"Current state"):
- The CUDA base image, Python 3.12 venv setup, and the exact pip failure
  modes below (Xet disk quota, missing peft/ftfy/timm, sageattention
  version) — all hit and fixed for real.
- `pipeline/envs/{wan22,rmbg,sapiens2}/requirements.txt` — every package in
  them was needed to make `wan22_vace_denoise`/`rmbg`/`sapiens2_lite`
  actually run, not guessed.
- `pipeline/envs/sam3dbody/requirements.txt` and the detectron2 pin
  (`@a1ce2f9`, `--no-build-isolation --no-deps`) — copied verbatim from
  facebookresearch/sam-3d-body's own INSTALL.md, but the *build itself*
  (detectron2 compiling cleanly, the full dependency list resolving) was
  still running on a pod as of this writing — check
  `pipeline/steps/sam3d_body.py`'s docstring / this repo's commit history
  for whether that finished clean.

Not yet verified — best-guess, flagged as such in the relevant Dockerfile
comments too:
- `Dockerfile.brush` in its entirety — brush has never actually been
  compiled in this project, on a pod or in a container. The Vulkan fix
  above is real (confirmed the underlying OS-level problem), but "does
  `cargo build --release` succeed with just `libvulkan1`/`vulkan-tools`, or
  does it need `vulkan-sdk` too" is still open.
- `pipeline/envs/seedvr2/requirements.txt` and the vendored
  `numz/ComfyUI-SeedVR2_VideoUpscaler` — `seedvr2`'s flash-attn/apex pins
  need checking against whatever CUDA/torch ABI `Dockerfile.main` actually
  produces (see that requirements.txt's own comments).
- The fp8 checkpoint disk cache (`wan22_vace_denoise.py`'s
  `fused_cache_dir` param) — confirmed broken in the diffusers/torchao
  version pairing this was tested against (`docs/fp8-quant-notes.md`), so
  don't bake a pre-fused checkpoint into an image layer expecting it to
  load correctly without checking that fix first.

## Gotchas worth knowing before you hit them again

**CUDA wheel index vs. actual torch version**: `pip install torch
--index-url https://download.pytorch.org/whl/cu124` resolved to a torch
build tagged `+cu130` during real testing, despite asking the `cu124`
index. Verify what you actually get (`python -c "import torch;
print(torch.__version__)"`) rather than trusting the index name — this
matters for `sageattention` and any other package that checks the installed
CUDA ABI at import/install time.

**HuggingFace's Xet backend ignores `HF_HOME`**: it caches to `$HOME`
regardless, which fills a container's small writable layer even when
`/data` (a large mounted volume) has plenty of room — downloads die with
"Disk quota exceeded" in a way that's confusing to debug from the error
alone. `Dockerfile.main` sets `HF_HUB_DISABLE_XET=1` to route around this
entirely.

**`briaai/RMBG-2.0` and `facebook/sam-3d-body-dinov3` are gated**: a human
must click "agree" on each model's HF page from an account before that
account's token can download them — not something a Dockerfile or
entrypoint script can do. `HF_TOKEN` should be injected at `docker run`
time (`-e HF_TOKEN=...` / this repo's `docker-compose.yml`), never baked
into an image layer.

**RMBG-2.0 needs `timm`, wan22_vace_denoise needs `peft`+`ftfy`+`kernels`,
sageattention needs `>=2.1.1`**: all four were missing-at-runtime
discoveries during real pod testing, not documented anywhere upstream as
required — they're in `pipeline/envs/{rmbg,wan22}/requirements.txt` now,
but if you're hand-rolling a requirements list instead of using those
files directly, you'll rediscover these the hard way.

## Building the sam3d_body checkpoint at container start, not build time

The gated checkpoint can't be baked into a shareable image layer (it needs
a real accepted-license HF account's token, and shouldn't be redistributed
inside the image anyway). Run the download against the mounted volume
instead, e.g. as part of your `docker-compose run` command or a small
entrypoint wrapper:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam-3d-body-dinov3')
"
```

(This is exactly what `pipeline/envs/sam3dbody/setup.sh` already does —
reuse it rather than duplicating the snippet.)
