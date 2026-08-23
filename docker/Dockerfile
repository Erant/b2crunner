# Single image for a RunPod (or any GPU-rental) deployment: every step's
# venv plus the brush binary, in one container. A pod IS a single container
# with no nested Docker daemon (confirmed on a real pod: no
# /var/run/docker.sock, no `docker` binary), so everything has to live here
# — see docs/docker.md's "Why one image".
#
# Build (context must be the repo root):
#   docker build -f docker/Dockerfile -t b2c/pipeline:latest .
#
# READ docs/docker-build-notes.md BEFORE the first build. It has the CUDA/
# detectron2 test plan, the RTX 4070 Ti specifics, and what to do when a
# layer fails. Nothing in this file has been built yet.
#
# Structure:
#   stage 1  brush-builder — Rust toolchain, builds the brush binary only.
#            Nothing from this stage reaches the final image except the
#            binary itself, so the ~1.5GB Rust toolchain is not shipped.
#   stage 2  runtime — CUDA devel base, one shared venv_base holding the
#            heavy packages, and per-step child venvs that inherit from it.
#
# Venv sharing: children are created with `--system-site-packages` from
# venv_base's interpreter, so they see its torch/CUDA wheels without
# copying them (~3GB each, four times over, if they didn't). A child that
# genuinely needs a different version of something still installs it
# locally, shadowing the base — the isolation property is preserved, only
# the duplication is gone. Keep venv_base minimal for that reason: anything
# version-contentious (diffusers, peft, torchao) belongs in a child.


# =============================================================================
# Stage 1: brush
# =============================================================================
# Erant/brush specifically — this fork adds normal-map supervision flags
# upstream ArthurBrussee/brush doesn't have (--normal-loss-weight,
# --normal-loss-start-iter, --alpha-mode, --export-name; see
# pipeline/steps/brush.py).
#
# Built on bookworm (glibc 2.36) while the runtime is noble (glibc 2.39):
# a binary built against older glibc runs on newer, not the reverse, so
# this direction is the safe one.
#
# brush is wgpu/Vulkan, NOT CUDA — it needs no CUDA toolkit to build, and
# ash loads libvulkan.so.1 dynamically at runtime, so no Vulkan dev
# packages are needed here either. Only the loader in the runtime stage.
FROM rust:1.88-slim-bookworm AS brush-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates pkg-config libssl-dev cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# UNVERIFIED: brush has never been compiled in this project, on a pod or in
# a container. If cargo fails on a missing system library, add it above —
# that is the expected failure mode, not something deeper.
RUN git clone --depth 1 https://github.com/Erant/brush.git /src/brush
WORKDIR /src/brush
RUN cargo build --release && install -m 0755 target/release/brush /out-brush


# =============================================================================
# Stage 2: runtime
# =============================================================================
# CUDA 13.0 + Ubuntu 24.04 (python3.12 native — 22.04 has no python3.12
# package at all, which is what broke the previous version of this file).
#
# CUDA 13 needs host driver >= 580. Check `nvidia-smi` on the target
# machine BEFORE building; an older driver fails at container start, not
# at build, which is a confusing way to find out.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS runtime

# Read by nvidia-container-toolkit at CONTAINER-CREATION time to decide
# which driver components to expose. Without `graphics` the driver's Vulkan
# and EGL ICDs are not wired up even though the .so files are present —
# confirmed on a real RunPod pod, where vulkaninfo failed with
# ERROR_INCOMPATIBLE_DRIVER under the default compute,utility set.
#
# This gates BOTH brush (Vulkan) and render (EGL). Whether RunPod honours
# an image-level ENV for it is still unresolved; setting it in the pod
# template's own environment variables is the more reliable path, since
# that is applied at creation time. See docs/docker-build-notes.md.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# Which GPU architectures to compile native extensions (detectron2, gsplat)
# for. There is no GPU during `docker build`, so this cannot be
# autodetected and MUST be set explicitly or the builds guess badly.
#   8.9 = Ada: RTX 4070 Ti / 4090 / L40S  (the test box and the pod used so far)
# For a deployable image widen this, at the cost of build time:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ARG TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# detectron2's setup.py compiles its CUDA kernels only if
# `torch.cuda.is_available() or FORCE_CUDA=1`. A docker build has no GPU,
# so without this it builds CPU-only, installs cleanly, and then fails at
# inference — the single nastiest trap in this file.
ENV FORCE_CUDA=1

ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG CUDA_WHEEL_INDEX=https://download.pytorch.org/whl/cu130

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# huggingface_hub's Xet backend caches to $HOME regardless of HF_HOME,
# filling a container's small writable layer while /data sits empty —
# downloads then die with "Disk quota exceeded". Learned on a real pod.
ENV HF_HOME=/data/hf_cache
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        git curl ca-certificates \
        # Vulkan loader for brush. The ICD/driver comes from the host via
        # nvidia-container-toolkit; the image only supplies the loader.
        # vulkan-tools gives you `vulkaninfo` to check that separately from
        # debugging a brush failure.
        libvulkan1 vulkan-tools \
        # GL/EGL loaders for pyrender (the `render` step). Same split as
        # Vulkan: libglvnd provides libEGL.so.1, the driver ICD is injected
        # by the toolkit. The previous version of this file installed the
        # Vulkan loader but not this one, so `render` could never have
        # worked — it tries libEGL.so.1 first, then falls back to OSMesa,
        # and neither was present. libosmesa6 is that software fallback.
        libegl1 libgl1 libglvnd0 libgles2 libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=brush-builder /out-brush /usr/local/bin/brush

WORKDIR /workspace/b2c_runner

# -----------------------------------------------------------------------------
# venv_base — the one copy of every heavy shared package
# -----------------------------------------------------------------------------
# NOT pinned by any step: every requirements.txt in pipeline/envs/ says
# `torch>=2.4.0` or bare `torch`. The old cu128 pin on sam3dbody existed
# only to match a specific pod's pre-installed toolkit for detectron2's
# native build; here we own the toolkit, so one CUDA line serves all four.
#
# Verify what you actually got: asking the cu130 index has previously
# returned a differently-tagged build. The RUN below prints it and fails
# loudly on a mismatch rather than letting it surface as a detectron2
# compile error three layers later.
RUN python3 -m venv /opt/venv_base \
    && /opt/venv_base/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv_base/bin/pip install \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${CUDA_WHEEL_INDEX} \
    && /opt/venv_base/bin/python -c "\
import torch, sys; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
sys.exit(0 if torch.version.cuda and torch.version.cuda.startswith('13') else 1)"

# The rest of the shared set. transformers>=4.56 satisfies every step that
# asks for it (rmbg/wan22 want >=4.46, sapiens2 wants >=4.56).
# opencv-python (not -headless) to match pyproject.toml's declared
# dependency — pip treats the two as different distributions, so mixing
# them installs both.
RUN /opt/venv_base/bin/pip install \
        numpy \
        opencv-python \
        Pillow \
        transformers>=4.56.0 \
        huggingface_hub>=0.25.0 \
        hf_transfer \
        safetensors \
        einops \
        tqdm \
        psutil \
        PyYAML \
        requests \
        'body2colmap @ git+https://github.com/Erant/body2colmap.git'

# -----------------------------------------------------------------------------
# Child venvs — inherit venv_base, add only what is specific to them
# -----------------------------------------------------------------------------
# Note each is created by venv_base's OWN interpreter: that is what makes
# --system-site-packages resolve to venv_base's site-packages rather than
# the system python's.

# venv_main: runs the orchestrator itself and every in_process step
# (rmbg, sapiens2_lite, render, render_splat, views, mask_splat,
# colmap_export, face_landmarks, brush).
#
# InProcessDispatcher runs a step in the calling process, so these deps
# must live in whatever env runs `python -m pipeline.cli` — not in an
# isolated venv of their own.
RUN /opt/venv_base/bin/python -m venv --system-site-packages /opt/venv_main \
    && /opt/venv_main/bin/pip install \
        kornia timm \
        plyfile \
        mediapipe \
    # gsplat has no prebuilt wheel past torch 2.4/cu124, so this is an
    # sdist that JIT-compiles its CUDA kernels on first use. That needs
    # nvcc at RUNTIME, which is why this image stays on the `devel` base.
    # See docs/docker-build-notes.md for the alternatives.
    && /opt/venv_main/bin/pip install gsplat

# venv_wan22: wan22_vace_denoise (subprocess dispatch).
#
# sageattention is deliberately NOT installed: the requirements file asks
# for >=2.1.1, and PyPI's newest is 1.0.6 — 2.x was never published there,
# so that constraint is unsatisfiable in a clean build. It is also
# optional; wan22_vace_denoise.py wraps set_attention_backend in
# try/except and honours attention_backend: "none". To get the speedup
# back, build it from source here:
#   pip install 'git+https://github.com/thu-ml/SageAttention.git'
# (compiles CUDA kernels, slow, and needs TORCH_CUDA_ARCH_LIST set — which
# it is, above).
RUN /opt/venv_base/bin/python -m venv --system-site-packages /opt/venv_wan22 \
    && /opt/venv_wan22/bin/pip install \
        'diffusers>=0.36.0' \
        'accelerate>=1.0.0' \
        'peft>=0.13.0' \
        ftfy \
        'torchao>=0.16.0' \
        'kernels>=0.4.0'

# venv_seedvr2: seedvr2 (subprocess dispatch).
# Vendored numz/ComfyUI-SeedVR2_VideoUpscaler must be importable — the step
# imports that repo's inference_cli.py as a module (see steps/seedvr2.py).
# flash-attn/apex are NOT needed: they are optional accelerators behind
# non-default attention_mode values; the default sdpa is pure PyTorch.
RUN /opt/venv_base/bin/python -m venv --system-site-packages /opt/venv_seedvr2 \
    && /opt/venv_seedvr2/bin/pip install \
        'omegaconf>=2.3.0' \
        'diffusers>=0.33.1' \
        'peft>=0.17.0' \
        'rotary_embedding_torch>=0.5.3' \
        gguf \
        matplotlib \
    && git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler \
        /opt/vendor/seedvr2
ENV PYTHONPATH="/opt/vendor/seedvr2:${PYTHONPATH}"

# venv_sam3dbody: sam3d_body (subprocess dispatch).
#
# The ordering here is load-bearing and was established the hard way on a
# real pod — see pipeline/envs/sam3dbody/requirements.txt for the full
# story. In short:
#   - numpy/cython first, in their own command, and the bulk install with
#     --no-build-isolation: xtcocotools's legacy setup.py imports numpy at
#     build time without declaring it as a PEP 517 build dep, so pip's
#     per-package isolated build env doesn't have it even when the venv
#     does.
#   - detectron2 last, --no-build-isolation --no-deps, per
#     facebookresearch/sam-3d-body's own INSTALL.md.
#   - sam-3d-body itself is vendored, not pip-installed: confirmed it has
#     no setup.py/pyproject.toml at all.
#
# The old cu128 torch pin is gone: it matched a specific pod's toolkit, and
# here detectron2 compiles against venv_base's torch and this image's own
# CUDA 13 toolkit, which agree by construction. THIS IS THE LAYER MOST
# LIKELY TO FAIL — detectron2 @a1ce2f9 is from Aug 2025 and has not been
# built against torch 2.9/CUDA 13 by anyone here. See
# docs/docker-build-notes.md for the fallback ladder.
RUN /opt/venv_base/bin/python -m venv --system-site-packages /opt/venv_sam3dbody \
    && /opt/venv_sam3dbody/bin/pip install cython \
    && /opt/venv_sam3dbody/bin/pip install --no-build-isolation \
        -r pipeline/envs/sam3dbody/requirements.txt \
    && /opt/venv_sam3dbody/bin/pip install --no-build-isolation --no-deps \
        'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    && git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git \
        /opt/vendor/sam3dbody \
    && echo /opt/vendor/sam3dbody > \
        $(/opt/venv_sam3dbody/bin/python -c "import site; print(site.getsitepackages()[0])")/sam3dbody_vendor.pth \
    && /opt/venv_sam3dbody/bin/python -c "from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body" \
    && /opt/venv_sam3dbody/bin/python -c "\
from detectron2 import _C; print('detectron2 native extension OK')"

# -----------------------------------------------------------------------------
# The pipeline package itself
# -----------------------------------------------------------------------------
# Last, and in its own layer, so editing pipeline code doesn't invalidate
# any of the venv layers above. Every venv needs it: `python -m
# pipeline.worker` must be importable inside each isolated env, not just
# the orchestrator's.
COPY . /workspace/b2c_runner

# The container's own env registry, pointing at the venv paths this image
# actually creates. pipeline/envs/envs.yaml in the repo describes a bare
# pod (/workspace/venv_*) and is left alone for that use; cli.py defaults
# to that path, so the container copy overwrites it here rather than
# requiring every invocation to pass --envs.
COPY docker/envs.docker.yaml /workspace/b2c_runner/pipeline/envs/envs.yaml

RUN for v in main wan22 seedvr2 sam3dbody; do \
        /opt/venv_${v}/bin/pip install --no-deps -e . || exit 1; \
    done \
    && /opt/venv_main/bin/python -c "\
import pipeline.steps; from pipeline.registry import STEP_REGISTRY; \
print(len(STEP_REGISTRY), 'steps registered')"

# Gated checkpoints (briaai/RMBG-2.0, facebook/sam-3d-body-dinov3) are
# deliberately NOT baked in: a human must accept each licence, and they
# should not be redistributed inside a shareable image. Download them to
# the mounted volume at container start instead —
# pipeline/envs/sam3dbody/setup.sh already does exactly this.
VOLUME ["/data"]

ENTRYPOINT ["/opt/venv_main/bin/python", "-m", "pipeline.cli"]
