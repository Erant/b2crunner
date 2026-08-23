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
# Three stages:
#   brush-builder   Rust toolchain -> the brush binary, nothing else.
#   python-builder  CUDA *devel* base. Compiles detectron2's CUDA kernels,
#                   which is the only reason nvcc is needed at all.
#   runtime         CUDA *runtime* base. Receives the finished venvs and
#                   binaries; carries no compiler, no nvcc, no Rust.
#
# Venv sharing: /opt/venv_base holds one copy of torch and the other heavy
# packages. Each per-step venv gets a .pth file pointing at venv_base's
# site-packages, which is enough for both imports and pip's "already
# satisfied" resolution, so no child ever installs its own torch (~3GB
# saved per venv). A child that genuinely needs a different version still
# installs it locally, and its own site-packages is searched first, so it
# shadows the base — the isolation is preserved, only the duplication is
# gone.
#
# NOT `python -m venv --system-site-packages`: that resolves `home` to the
# real base interpreter, not to venv_base, so the child would inherit
# /usr/lib/python3's packages and see nothing of venv_base. Tested; it does
# not work. The .pth is the mechanism that does.


# =============================================================================
# Stage 1: brush
# =============================================================================
# Erant/brush specifically — this fork adds normal-map supervision flags
# upstream ArthurBrussee/brush doesn't have (--normal-loss-weight,
# --normal-loss-start-iter, --alpha-mode, --export-name; see
# pipeline/steps/brush.py).
#
# Built on bookworm (glibc 2.36) against a noble (glibc 2.39) runtime:
# older-to-newer is the safe direction for a dynamically linked binary.
#
# brush is wgpu/Vulkan, NOT CUDA — no CUDA toolkit needed to build it, and
# ash dlopen's libvulkan.so.1 at runtime, so no Vulkan dev packages here
# either. Only the loader, in the runtime stage.
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

# FUTURE: the splat *renderer* utility will be built here too, as a second
# binary linking brush-render, once it exists — see
# docs/brush-render-utility.md. That is what replaces gsplat (dropped
# below) as the backend for the render_splat step.


# =============================================================================
# Stage 2: python-builder (CUDA devel — nvcc lives here and only here)
# =============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS python-builder

# There is no GPU during `docker build`, so the arch list cannot be
# autodetected and MUST be set or detectron2 guesses badly.
#   8.9 = Ada: RTX 4070 Ti / 4090 / L40S  (the test box and the pod so far)
# Widen for a deployable image, at the cost of build time:
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

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/b2c_runner

# -----------------------------------------------------------------------------
# venv_base — the one copy of every heavy shared package
# -----------------------------------------------------------------------------
# NOT pinned by any step: every requirements.txt in pipeline/envs/ says
# `torch>=2.4.0` or bare `torch`. The old cu128 pin existed only to match a
# specific pod's pre-installed toolkit for detectron2's native build; here
# we own the toolkit, so one CUDA line serves all four venvs.
#
# The version check is deliberate: asking the cu130 index has previously
# returned a differently-tagged build, and the symptom would otherwise be a
# detectron2 compile error several layers later.
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

# Every child venv gets this: a plain venv plus a .pth naming venv_base's
# site-packages. See the header for why --system-site-packages is not it.
COPY docker/make-child-venv.sh /usr/local/bin/make-child-venv
RUN chmod +x /usr/local/bin/make-child-venv

# venv_main: runs the orchestrator and every in_process step (rmbg,
# sapiens2_lite, render, render_splat, views, mask_splat, colmap_export,
# face_landmarks, brush). InProcessDispatcher runs a step in the calling
# process, so these deps must live in whatever env runs
# `python -m pipeline.cli`.
#
# gsplat is deliberately NOT installed. It was the only reason this image
# needed nvcc at runtime: it publishes no wheel past torch 2.4/cu124, so on
# a modern stack it JIT-compiles its CUDA kernels on first use. Splat
# rendering moves to a brush-based utility instead (see
# docs/brush-render-utility.md), which is Vulkan — the capability this
# image already has to provide for brush itself. Until that utility exists
# and pipeline/steps/splat.py is switched over, `render_splat` will fail at
# import; every other step is unaffected.
#
# plyfile stays: load_splat/save_splat need it, and brush emits .ply.
RUN make-child-venv /opt/venv_main \
    && /opt/venv_main/bin/pip install \
        kornia timm \
        plyfile \
        mediapipe

# venv_wan22: wan22_vace_denoise (subprocess dispatch).
#
# sageattention is deliberately NOT installed: the requirements file asks
# for >=2.1.1 and PyPI's newest is 1.0.6 — 2.x was never published there,
# so that constraint is unsatisfiable in a clean build. It is also
# optional; wan22_vace_denoise.py wraps set_attention_backend in
# try/except and honours attention_backend: "none". To get the speedup
# back, build it from source here:
#   pip install 'git+https://github.com/thu-ml/SageAttention.git'
# (compiles CUDA kernels, slow, needs TORCH_CUDA_ARCH_LIST — set above).
RUN make-child-venv /opt/venv_wan22 \
    && /opt/venv_wan22/bin/pip install \
        'diffusers>=0.36.0' \
        'accelerate>=1.0.0' \
        'peft>=0.13.0' \
        ftfy \
        'torchao>=0.16.0' \
        'kernels>=0.4.0'

# venv_seedvr2: seedvr2 (subprocess dispatch).
# The vendored numz/ComfyUI-SeedVR2_VideoUpscaler must be importable — the
# step imports that repo's inference_cli.py as a module (see
# steps/seedvr2.py). flash-attn/apex are NOT needed: they are optional
# accelerators behind non-default attention_mode values; the default sdpa
# is pure PyTorch.
RUN make-child-venv /opt/venv_seedvr2 \
    && /opt/venv_seedvr2/bin/pip install \
        'omegaconf>=2.3.0' \
        'diffusers>=0.33.1' \
        'peft>=0.17.0' \
        'rotary_embedding_torch>=0.5.3' \
        gguf \
        matplotlib \
    && git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler \
        /opt/vendor/seedvr2

# venv_sam3dbody: sam3d_body (subprocess dispatch). THE LAYER MOST LIKELY
# TO FAIL — see docs/docker-build-notes.md for the fallback ladder.
#
# The ordering is load-bearing and was established the hard way on a real
# pod — see pipeline/envs/sam3dbody/requirements.txt. In short:
#   - cython first, and the bulk install with --no-build-isolation:
#     xtcocotools's legacy setup.py imports numpy at build time without
#     declaring it as a PEP 517 build dep, so pip's per-package isolated
#     build env lacks it even when the venv has it.
#   - detectron2 last, --no-build-isolation --no-deps, per
#     facebookresearch/sam-3d-body's own INSTALL.md.
#   - sam-3d-body itself is vendored, not pip-installed: it has no
#     setup.py/pyproject.toml at all.
#
# The old cu128 torch pin is gone: it matched a specific pod's toolkit,
# and here detectron2 compiles against venv_base's torch and this stage's
# CUDA 13 toolkit, which agree by construction.
RUN make-child-venv /opt/venv_sam3dbody \
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

# Strip build leftovers that would otherwise be copied into the runtime
# image: pip caches, and the compiled-object trees left by the source
# builds above.
RUN find /opt/venv_* -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/venv_* -type f -name '*.pyc' -delete \
    && rm -rf /root/.cache


# =============================================================================
# Stage 3: runtime (CUDA runtime — no nvcc, no compilers, no Rust)
# =============================================================================
# The torch wheels bundle their own CUDA libraries (the nvidia-* packages
# in venv_base), so almost nothing is needed from the base image itself;
# the cudnn-runtime variant is belt and braces for anything that dlopen's a
# system cudnn.
#
# The venvs are copied wholesale. That works because both stages are
# ubuntu24.04, so /usr/bin/python3 is 3.12 in both and the venvs' interpreter
# symlinks and pyvenv.cfg `home` entries stay valid.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 AS runtime

# Read by nvidia-container-toolkit at CONTAINER-CREATION time to decide
# which driver components to expose. Without `graphics` the driver's Vulkan
# and EGL ICDs are not wired up even though the .so files are present —
# confirmed on a real RunPod pod, where vulkaninfo failed with
# ERROR_INCOMPATIBLE_DRIVER under the default compute,utility set.
#
# This gates BOTH brush (Vulkan) and render (EGL). Whether RunPod honours
# an image-level ENV for it is unresolved; setting it in the pod template's
# own environment variables is the more reliable path, since that is
# applied at creation time. See docs/docker-build-notes.md.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# huggingface_hub's Xet backend caches to $HOME regardless of HF_HOME,
# filling a container's small writable layer while /data sits empty —
# downloads then die with "Disk quota exceeded". Learned on a real pod.
ENV HF_HOME=/data/hf_cache
ENV HF_HUB_DISABLE_XET=1

ENV PYTHONPATH="/opt/vendor/seedvr2:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        # python3 only: the venvs carry their own packages, and nothing is
        # built here. python3-venv/-dev and pip are build-stage concerns.
        python3 \
        ca-certificates \
        # Vulkan loader for brush. The ICD/driver is injected by
        # nvidia-container-toolkit; the image supplies only the loader.
        # vulkan-tools gives `vulkaninfo` for checking that separately from
        # debugging a brush failure.
        libvulkan1 vulkan-tools \
        # GL/EGL loaders for pyrender (the `render` step). Same split as
        # Vulkan: libglvnd provides libEGL.so.1, the driver ICD comes from
        # the toolkit. An earlier version of this file shipped the Vulkan
        # loader but not this one, so `render` could never have worked — it
        # tries libEGL.so.1, falls back to OSMesa, and neither was present.
        # libosmesa6 is that software fallback.
        libegl1 libgl1 libglvnd0 libgles2 libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=brush-builder /out-brush /usr/local/bin/brush
COPY --from=python-builder /opt/venv_base /opt/venv_base
COPY --from=python-builder /opt/venv_main /opt/venv_main
COPY --from=python-builder /opt/venv_wan22 /opt/venv_wan22
COPY --from=python-builder /opt/venv_seedvr2 /opt/venv_seedvr2
COPY --from=python-builder /opt/venv_sam3dbody /opt/venv_sam3dbody
COPY --from=python-builder /opt/vendor /opt/vendor

WORKDIR /workspace/b2c_runner

# Last, and in its own layer, so editing pipeline code invalidates neither
# the venv layers nor the brush build. Every venv needs the package:
# `python -m pipeline.worker` must be importable inside each isolated env,
# not just the orchestrator's.
COPY . /workspace/b2c_runner

# The container's own env registry, pointing at the venv paths this image
# actually creates. pipeline/envs/envs.yaml in the repo describes a bare
# pod (/workspace/venv_*) and is left alone for that use; cli.py defaults
# to that path, so the container copy overwrites it here rather than
# requiring every invocation to pass --envs.
COPY docker/envs.docker.yaml /workspace/b2c_runner/pipeline/envs/envs.yaml

# --no-deps: everything is already in venv_base, and there is no compiler
# here to build anything that isn't.
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
