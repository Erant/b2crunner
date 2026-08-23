# Docker container guide

Real, buildable artifacts live in `docker/`:

- `docker/Dockerfile` — one image, everything: the orchestrator
  (`python -m pipeline.cli`), one venv per `subprocess`-dispatch step
  (`wan22`, `sam3dbody`, `seedvr2`), one shared `venv_main` for
  `in_process`-dispatch steps (`rmbg`, `sapiens2`, `brush`) — mirrors
  `scripts/pod_bootstrap.sh`'s venv layout, just inside a container instead
  of on a bare pod — plus the `brush` Rust binary itself, built straight
  into the image.
- `docker/docker-compose.yml` — builds it, useful for local
  build/test before pushing as a RunPod pod template. RunPod itself doesn't
  consume this file directly.

This file is the narrative — *why* each piece looks the way it does, and
the gotchas that aren't obvious from reading the Dockerfile alone. When
they disagree, the Dockerfile is the source of truth; update this file to
match, not the other way around.

## Quickstart

```bash
export HF_TOKEN=hf_...
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm pipeline \
    run pipeline/workflows/fast_helical_native.yaml --dataset /data/some_dataset -v
```

The container mounts a shared `/data` volume (weights, HF cache, datasets
you bind-mount in).

**UNVERIFIED**: the Dockerfile hasn't actually been built end-to-end as of
this writing — see "What's confirmed vs. not" below for exactly which
pieces come from real pod-tested commands vs. are still best-guess.

## Why one image, not brush split out

`pipeline/README.md`'s dispatch table draws the line: `subprocess`
(isolated venv, shared host CUDA/system userspace) for steps whose only
conflict is Python dependencies; `docker` (full container boundary) for
steps needing OS-level isolation a venv can't give — brush is exactly that
case (see below), which argues for splitting it into its own image invoked
via `pipeline/dispatch/docker.py`'s `DockerDispatcher`. That's what an
earlier version of this setup did (`docker/Dockerfile.main` +
`docker/Dockerfile.brush`, see git history) — reverted in favor of one
image because **this project targets RunPod, and a RunPod pod is already a
single container with no nested Docker daemon to run a second image
inside** (confirmed on a real pod: no `/var/run/docker.sock`, no `docker`
binary at all). `DockerDispatcher`'s container-boundary trick only helps
when something *else* controls the outer container and grants it a Docker
socket — on RunPod, the pod's own image is the only container boundary
that exists, so brush's requirement has to be solved at that level or not
at all. `pipeline/steps/brush.py` runs `dispatch: in_process` accordingly.

Brush's actual requirement — confirmed on a real RunPod pod: the NVIDIA
driver's Vulkan `.so` files (`libGLX_nvidia.so.0`, `libnvidia-glcore.so.*`)
and the ICD json can all be present, and `vulkaninfo` still fails with
`ERROR_INCOMPATIBLE_DRIVER` / "Found no drivers!" if
`NVIDIA_DRIVER_CAPABILITIES` doesn't include `graphics` (and usually
`display`) — that env var is read by `nvidia-container-toolkit` at
*container-creation time* to decide which driver components to actually
expose, and the default RunPod PyTorch-template pod tested only set
`compute,utility`, built for training/inference, not rendering. Setting it
inside an already-running container doesn't retroactively fix anything.
`docker/Dockerfile` bakes
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display` into the
image itself — **necessary, but not yet confirmed sufficient**: whether
RunPod's pod-creation path actually honors an image-level `ENV` for this
the way a plain `docker run --gpus all` against a locally-built image does
is unresolved. If it turns out RunPod doesn't honor it, the fallback is
switching brush back to `dispatch: docker` against a target that does
expose a Docker daemon (a self-hosted box, a different provider) — the
dispatcher already supports this, it's a one-line change in
`fast_helical_native.yaml`, not a rewrite.

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
  facebookresearch/sam-3d-body's own INSTALL.md; the build itself, and real
  inference against `cyber_6f`'s `anchor.png`, are now confirmed clean on a
  bare pod venv — see `pipeline/steps/sam3d_body.py`'s docstring. What's
  NOT yet confirmed is this exact recipe working unmodified inside the
  Docker image build rather than a bare-pod venv (different base image,
  different starting package set) — worth a real build+run check before
  trusting it blind.
- `pipeline/envs/seedvr2/requirements.txt` and the vendored
  `numz/ComfyUI-SeedVR2_VideoUpscaler` — real inference (a genuine
  720x1280 -> 1440x2560 upscale) confirmed on a bare pod venv, see
  `pipeline/steps/seedvr2.py`'s docstring. flash-attn/apex are NOT actually
  needed (an earlier version of this doc/requirements.txt guessed they
  were) — the default `attention_mode: sdpa` is pure PyTorch. Same
  bare-pod-venv-vs-Docker-image caveat as sam3dbody above applies.

Not yet verified — best-guess, flagged as such in `docker/Dockerfile`'s
comments too:
- The brush build section in its entirety — brush has never actually been
  compiled in this project, on a pod or in a container. The Vulkan fix
  above is real (confirmed the underlying OS-level problem), but "does
  `cargo build --release` succeed with just `libvulkan1`/`vulkan-tools`, or
  does it need `vulkan-sdk` too" is still open. Also open: whether RunPod's
  pod-creation path honors the image-level `NVIDIA_DRIVER_CAPABILITIES` at
  all (see "Why one image" above).
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
alone. `docker/Dockerfile` sets `HF_HUB_DISABLE_XET=1` to route around this
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
