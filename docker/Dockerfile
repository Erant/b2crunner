# Single image for a RunPod deployment: every step's venv, plus brush (Rust
# CLI, invoked as a subprocess by pipeline/steps/brush.py — never a Python
# binding) baked into the same container. This trades the cleaner
# container-boundary isolation a separate brush image would give (see git
# history for that split — docker/Dockerfile.main + docker/Dockerfile.brush,
# reverted in favor of this) for matching how RunPod actually works: a pod
# IS a single container, and NVIDIA_DRIVER_CAPABILITIES is a property of
# that pod's own image, not something a second `docker run --gpus all`
# nested inside it could grant independently (RunPod pods don't expose a
# Docker daemon to nest containers in — confirmed on a real pod: no
# /var/run/docker.sock, no `docker` binary).
#
# Brush's Vulkan/graphics requirement is therefore RunPod's problem to
# solve via this image's own NVIDIA_DRIVER_CAPABILITIES (set below) and
# whatever RunPod's pod-creation path does with it — UNRESOLVED as of this
# writing (see docs/docker.md). Confirmed on a real pod that a *default*
# RunPod PyTorch-template pod's NVIDIA_DRIVER_CAPABILITIES was empty
# (compute,utility only) and vulkaninfo failed even with the driver's
# Vulkan .so files physically present; setting the env var in *this* image
# is necessary but not yet confirmed sufficient — RunPod's own container
# creation may or may not honor an image-level ENV for this the same way
# `docker run --gpus all` against a locally-built image does.
#
# Build: docker build -f docker/Dockerfile -t b2c/pipeline:latest .
# (build context must be the repo root, so `COPY . /workspace/b2c_runner`
# below picks up pipeline/, pyproject.toml, etc.)

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# See the file-level comment above — unresolved whether RunPod's pod
# creation actually honors this from a custom pod image the way a plain
# `docker run --gpus all` does.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    git curl build-essential pkg-config libssl-dev \
    # Vulkan loader + NVIDIA ICD — brush runs on wgpu, which needs Vulkan
    # (not just the CUDA runtime) to find the GPU. vulkan-tools gives you
    # `vulkaninfo` to sanity-check this before debugging a brush failure as
    # something else.
    libvulkan1 vulkan-tools \
    ffmpeg libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/b2c_runner
COPY . /workspace/b2c_runner

# huggingface_hub's Xet backend caches to $HOME regardless of HF_HOME — on
# a container with a small writable layer and a large mounted volume for
# weights, this fills the small disk and downloads die with "Disk quota
# exceeded" even though the volume has room. Learned the hard way on a real
# pod before this file existed (see docs/docker.md).
ENV HF_HOME=/data/hf_cache
ENV HF_HUB_DISABLE_XET=1

# --- venv_main: in_process steps (rmbg, sapiens2, brush) + orchestrator ---
RUN python3.12 -m venv /workspace/venv_main \
    && /workspace/venv_main/bin/pip install --no-cache-dir --upgrade pip \
    && /workspace/venv_main/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && /workspace/venv_main/bin/pip install --no-cache-dir \
        -r pipeline/envs/rmbg/requirements.txt \
        -r pipeline/envs/sapiens2/requirements.txt \
    && /workspace/venv_main/bin/pip install --no-cache-dir -e .

# --- venv_wan22: wan22_vace_denoise (subprocess dispatch) ---
RUN python3.12 -m venv /workspace/venv_wan22 \
    && /workspace/venv_wan22/bin/pip install --no-cache-dir --upgrade pip \
    && /workspace/venv_wan22/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && /workspace/venv_wan22/bin/pip install --no-cache-dir -r pipeline/envs/wan22/requirements.txt \
    && /workspace/venv_wan22/bin/pip install --no-cache-dir -e .

# --- venv_sam3dbody: sam3d_body (subprocess dispatch, gated checkpoint) ---
# detectron2 pinned per facebookresearch/sam-3d-body's own INSTALL.md.
# numpy/cython + --no-build-isolation: confirmed on a real pod that
# xtcocotools's legacy setup.py imports numpy at build time without
# declaring it as a PEP 517 build dependency, so pip's normal *isolated*
# per-package build env doesn't have numpy even when this venv already
# does — the whole bulk install has to run with --no-build-isolation
# instead (see pipeline/envs/sam3dbody/requirements.txt's comment).
RUN python3.12 -m venv /workspace/venv_sam3dbody \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir --upgrade pip \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir numpy cython \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir --no-build-isolation -r pipeline/envs/sam3dbody/requirements.txt \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir \
        'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir git+https://github.com/facebookresearch/sam-3d-body.git \
    && /workspace/venv_sam3dbody/bin/pip install --no-cache-dir -e .
# Checkpoint download (facebook/sam-3d-body-dinov3) is NOT baked into the
# image — it's gated, needs HF_TOKEN from an account that accepted the
# license, and shouldn't be baked into a shareable image layer anyway. Run
# pipeline/envs/sam3dbody/setup.sh's checkpoint-download step at container
# start against the mounted /data volume instead.

# --- venv_seedvr2: seedvr2 (subprocess dispatch) ---
# UNVERIFIED — this step is still a documented-contract stub as of this
# writing (pipeline/steps/seedvr2.py), and the vendored
# numz/ComfyUI-SeedVR2_VideoUpscaler repo's flash-attn/apex pins may not
# match this base image's CUDA/torch ABI. Confirm before trusting this
# layer in production.
RUN python3.12 -m venv /workspace/venv_seedvr2 \
    && /workspace/venv_seedvr2/bin/pip install --no-cache-dir --upgrade pip \
    && /workspace/venv_seedvr2/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && /workspace/venv_seedvr2/bin/pip install --no-cache-dir -r pipeline/envs/seedvr2/requirements.txt \
    && /workspace/venv_seedvr2/bin/pip install --no-cache-dir -e . \
    && git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler /workspace/venv_seedvr2_vendor
ENV PYTHONPATH="/workspace/venv_seedvr2_vendor:${PYTHONPATH}"

# --- brush: Rust CLI, no Python venv of its own (pipeline/steps/brush.py
# shells out to it) — Erant/brush specifically, this fork adds normal-map
# supervision flags not in upstream ArthurBrussee/brush
# (--normal-loss-weight, --normal-loss-start-iter, --alpha-mode,
# --export-name; see pipeline/steps/brush.py). UNVERIFIED build — never
# actually compiled in this session; confirm `cargo build --release`
# succeeds with just libvulkan1/vulkan-tools before trusting this layer,
# add vulkan-sdk if not.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain 1.88.0
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone https://github.com/Erant/brush.git /opt/brush \
    && cd /opt/brush \
    && cargo build --release \
    && cp target/release/brush /usr/local/bin/brush
# Sanity check the binary is at least invocable headless (no display) at
# build time — fails the build loudly instead of failing silently later.
# If NVIDIA_DRIVER_CAPABILITIES above doesn't actually take effect in this
# build environment (it's a build-time RUN, not a `docker run` with GPU
# access at all), this check may need to move to a container-start health
# check instead of a build-time RUN — unresolved, see file-level comment.
RUN brush --help

VOLUME ["/data"]

ENTRYPOINT ["/workspace/venv_main/bin/python", "-m", "pipeline.cli"]
