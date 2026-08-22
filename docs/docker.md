# Docker container guide

Working notes for building a container that has everything the pipeline
needs: the Python/CUDA stack for `wan22_vace_denoise`/`rmbg` (and eventually
`sam3d_body`/`seedvr2`), plus the [`Erant/brush`](https://github.com/Erant/brush)
Gaussian-splat trainer this pipeline shells out to (see `nodes/brush_node.py`
in the original `ComfyUI-Body2COLMAP` repo — `Body2COLMAP_RunBrush` invokes
it as a plain subprocess, never a Python binding, so it only needs to be a
binary on `PATH` inside the container).

This is written from what was actually verified running this pipeline on a
RunPod L40S pod (see the pod session that produced `pipeline/steps/*.py`) —
not a full multi-stage Dockerfile yet. Treat it as the checklist to turn into
one, and update it as steps get containerized for real.

## Base image

Start from an official CUDA devel image, not a `runtime` one — `brush`
compiles Rust code with GPU-backend build scripts (wgpu/naga), and the
Python side needs `nvcc` for nothing itself but some pip packages
(`torchao`, `flash-attn` if seedvr2 needs it) build native extensions against
the CUDA toolchain at install time.

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
```

Pin the CUDA minor version to whatever `torch`'s published wheel index
actually supports at build time (check
`https://download.pytorch.org/whl/cu124` — during the pod session this
resolved to a torch build tagged `+cu130` even though we asked the `cu124`
index, so verify what you actually get with `python -c "import torch;
print(torch.__version__)"` rather than trusting the index name).

## System packages

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    git curl build-essential pkg-config \
    libssl-dev \
    # Vulkan loader + NVIDIA ICD — brush runs on wgpu, which needs Vulkan
    # (not just the CUDA runtime) to find the GPU. nvidia-container-toolkit
    # mounts the driver's ICD json at runtime; the container still needs the
    # loader library itself.
    libvulkan1 vulkan-tools \
    ffmpeg libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
```

`vulkan-tools` gives you `vulkaninfo` to sanity-check the GPU is visible to
Vulkan inside the container before debugging a brush failure as something
else.

## Rust toolchain + brush

```dockerfile
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain 1.88.0
ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone https://github.com/Erant/brush.git /opt/brush \
    && cd /opt/brush \
    && cargo build --release \
    && cp target/release/brush /usr/local/bin/brush
```

Notes:
- Rust 1.88+ is what upstream/this fork documents as the minimum toolchain —
  pin exactly, don't float `stable`, since a fork this actively developed can
  rely on very recent language features.
- This fork (`Erant/brush`) adds normal-map supervision flags not in
  upstream `ArthurBrussee/brush` — `--normal-loss-weight`,
  `--normal-loss-start-iter`, `--alpha-mode`, `--export-name` (see
  `nodes/brush_node.py`'s `cmd = [...]` construction in the source repo for
  the full flag list actually used). Building the fork, not upstream, is the
  point — don't substitute `ArthurBrussee/brush` here.
- `cargo build --release` for this project builds against whatever GPU
  backend wgpu picks up at build time; if this ends up needing a specific
  Vulkan SDK version pinned (`vulkan-sdk` apt package vs. just the loader),
  that's unverified — the pod session never actually built brush, only
  researched it. First real container build should confirm this compiles
  clean with just `libvulkan1`/`vulkan-tools`, and add `vulkan-sdk` if not.
- Confirm the binary works headless before trusting it in a container with
  no display: `brush --help` should not require a window/X server. If it
  does, `--with-viewer` is opt-in per `nodes/brush_node.py` so the pipeline
  itself never needs one — just don't pass that flag.

## Python environment

Mirror `pipeline/envs/*/requirements.txt` rather than re-deriving versions
here — those files are the source of truth and get updated as bugs are
found running against real hardware. As of this pod session:

```dockerfile
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# torch first, from the CUDA wheel index — before the rest, which pin
# torch>=X but don't specify --index-url
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

COPY pipeline/envs/wan22/requirements.txt /tmp/wan22-requirements.txt
COPY pipeline/envs/rmbg/requirements.txt /tmp/rmbg-requirements.txt
RUN pip install --no-cache-dir -r /tmp/wan22-requirements.txt \
    && pip install --no-cache-dir -r /tmp/rmbg-requirements.txt

COPY . /workspace/b2c_runner
WORKDIR /workspace/b2c_runner
RUN pip install --no-cache-dir -e .
```

`body2colmap` (a git dependency in `pyproject.toml`) and the rest of the
pipeline's own deps come along with `pip install -e .`.

## HuggingFace auth + caching

```dockerfile
ENV HF_HOME=/data/hf_cache
# Xet caches to $HOME regardless of HF_HOME — on a container with a small
# writable layer and a large mounted volume for weights, this fills the
# small disk and downloads die with "Disk quota exceeded" even though the
# volume has room. Learned the hard way on the pod session before this
# guide existed.
ENV HF_HUB_DISABLE_XET=1
```

`HF_TOKEN` should be injected at `docker run` time (`-e HF_TOKEN=...`), not
baked into the image — it's a credential, and `briaai/RMBG-2.0` and
`facebook/sam-3d-body-dinov3` are both gated repos requiring a token from an
account that's clicked "agree" on each model page first (that's a one-time
human action per HF account, not something the container can do).

## The fp8 checkpoint cache

`wan22_vace_denoise.py`'s `load()` does LoRA-fuse + torchao fp8-quantize
once and can save the result to `params["fused_cache_dir"]` for instant
reload afterward (see that module's docstring). If this container bakes in
a pre-fused cache rather than an entrypoint that produces one on first run,
that's the artifact worth eventually publishing as a real fp8 diffusers VACE
checkpoint (none exists publicly as of this writing — see that module's
docstring for why). Verify a cache-hit load actually produces correct
output before trusting it that far; this was written but not yet verified
end-to-end as of this session.

## Volumes

Mount a persistent volume at `/data` (or wherever `HF_HOME`/`fused_cache_dir`
point) rather than baking multi-GB model weights into image layers — this
mirrors how the pod session used RunPod's `/workspace` network volume, and
keeps the image itself buildable/pushable without hauling ~80GB of weights
through a registry.

```dockerfile
VOLUME ["/data"]
```

## Open items before this is a real Dockerfile

- Never actually built `brush` in a container — the Vulkan/system-deps list
  above is researched from the fork's README, not verified by a build.
- `sam3d_body`'s env needs its own container stage (pinned `detectron2`
  build, per `pipeline/envs/sam3dbody/requirements.txt`'s comments) —
  probably a separate image, not bundled into this one, given the deferred
  heavy-import discipline the pipeline steps already follow for exactly this
  reason.
- `seedvr2` likewise — flash-attn/apex pinned to a specific torch/CUDA ABI;
  decide `subprocess` vs. `docker` dispatch (see pipeline README) once that
  ABI is checked against whatever CUDA image this ends up on.
