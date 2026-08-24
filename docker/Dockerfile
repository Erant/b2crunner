# Single image for a RunPod (or any GPU-rental) deployment: every step's
# venv plus the brush binary, in one container. A pod IS a single container
# with no nested Docker daemon (confirmed on a real pod: no
# /var/run/docker.sock, no `docker` binary), so everything has to live here
# — see docs/docker.md's "Why one image".
#
# Build (context must be the repo root):
#   docker build -f docker/Dockerfile -t b2c/pipeline:latest .
#
# The container's default mode is the web UI (docker/entrypoint.sh) — a pod
# started with no arguments serves it on $B2C_PORT and stays alive. See
# docs/runpod.md for the pod template settings that go with it.
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
# rustc >= 1.95 is REQUIRED, and this is the floor the dependency tree sets,
# not a preference: the first build attempt used rust:1.88 and cargo refused
# to resolve, naming rerun@0.36.0 and sysinfo@0.39.6 as needing 1.95 (plus
# safe_arch/wide needing 1.89). 1.98 is current stable; bump rather than
# pin dependencies back.
#
# brush is wgpu/Vulkan, NOT CUDA — no CUDA toolkit needed to build it, and
# ash dlopen's libvulkan.so.1 at runtime, so no Vulkan dev packages here
# either. Only the loader, in the runtime stage.
FROM rust:1.98-slim-bookworm AS brush-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates pkg-config libssl-dev cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# The fork's work lives on the `normal-map-supervision` branch, NOT on
# `main`. Erant/brush's main merely tracks upstream ArthurBrussee/brush, so
# cloning the default branch silently builds a binary WITHOUT the two
# normal-supervision flags the step relies on (--normal-loss-weight,
# --normal-loss-start-iter). It compiles and runs; it just rejects that argv
# at runtime.
#
# Caught by diffing the built binary's `--help` against the argv
# pipeline/steps/brush.py constructs. Worth redoing whenever this pin moves:
#   docker run --rm --entrypoint /bin/sh b2c/brush-builder -c '/out-brush --help' > help.txt
#   grep -oE '"--[a-z-]+"' pipeline/steps/brush.py | sort -u | tr -d '"' \
#     | while read f; do grep -q -- "$f" help.txt || echo "MISSING $f"; done
#
# That diff also turned up a third flag, --total-steps, which is absent from
# BOTH branches: upstream renamed it --total-train-iters and the fork carries
# the rename. Fixed on the Python side in steps/brush.py, not here.
ARG BRUSH_BRANCH=normal-map-supervision
RUN git clone --depth 1 --branch ${BRUSH_BRANCH} \
        https://github.com/Erant/brush.git /src/brush
WORKDIR /src/brush

# Two binaries out of this one clone: `brush` (apps/brush-app, training +
# viewer) and `brush-splat-render` (crates/brush-splat-render, a standalone
# CLI that rasterises a trained .ply against an explicit camera list — see
# docs/splat-render.md in the brush repo). `brush-splat-render` is what
# replaces gsplat as render_splat's backend (see "gsplat: dropped" in
# docs/docker-build-notes.md): gsplat published no wheel past torch
# 2.4/cu124 and needed nvcc at runtime; this binary is wgpu/Vulkan, same as
# `brush` itself, so no CUDA toolchain is needed for either.
#
# `-p brush-app -p brush-splat-render` rather than a bare `cargo build`:
# the workspace's `default-members` is just `apps/brush-app`, so a bare
# build would silently skip brush-splat-render (it's a workspace member,
# not a default one). `-p` also avoids `--workspace`, which would pull in
# unrelated members this image doesn't need (brush-js, brush-c, the wasm
# bindings).
RUN cargo build --release -p brush-app -p brush-splat-render \
    && install -m 0755 target/release/brush /out-brush \
    && install -m 0755 target/release/brush-splat-render /out-brush-splat-render


# =============================================================================
# Stage 2: python-builder (CUDA devel — nvcc lives here and only here)
# =============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS python-builder

# There is no GPU during `docker build`, so the arch list cannot be
# autodetected and MUST be set or detectron2 guesses badly. This ONLY
# affects detectron2's own nvcc-compiled kernels (the one CUDA extension
# actually built from source in this image) — torch's own prebuilt cu130
# wheel already ships SASS for sm_75/80/86/90/100/120 plus compute_120 PTX
# regardless of this ARG (confirmed via torch.cuda.get_arch_list() in the
# built image), so nothing else here needs widening for a new GPU
# generation, only this.
#
# Default covers Ada and Blackwell, consumer and server, in one image —
# confirmed RunPod is the deploy target and the next verification round is
# explicitly "an Ada or Blackwell pod, consumer or server":
#   8.9  = Ada:       RTX 40-series (consumer) / L40, L40S (server) —
#          the 4070 Ti this was built and GPU-verified on
#   10.0 = Blackwell:  B100, B200, GB200 (server/datacenter)
#   12.0 = Blackwell:  RTX 50-series (consumer)
# Hopper (9.0 — H100/H200) isn't in the default since it wasn't asked for,
# but widens the same way if it comes up.
#
# Each additional arch is a real cost: detectron2's nvcc compile scales
# with the arch count (the 8.9-only build took ~2 min — see
# docs/docker-build-notes.md's detectron2 probe RESULT). Override for a
# single-arch build if compile time matters more than portability:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.9"
ARG TORCH_CUDA_ARCH_LIST="8.9;10.0;12.0"
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

# The last four are for `import cv2`, not for compiling anything. venv_base
# installs opencv-python (not -headless), whose cv2.abi3.so links against
# libGL/libglib/libgthread/libxcb; without them the sam3dbody layer's own
# `from sam_3d_body import ...` verification dies on
# `ImportError: libxcb.so.1` *after* detectron2 has already compiled
# successfully, which is a maximally misleading place to fail. This stage
# is not shipped, so the extra packages cost nothing in the final image.
# (The runtime stage gets libGL and glib explicitly and libxcb1 transitively
# via libxext6 -> libx11-6.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        git curl ca-certificates build-essential \
        libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/b2c_runner

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
# a modern stack it JIT-compiles its CUDA kernels on first use.
# render_splat now shells out to the brush-splat-render binary instead (see
# the brush-builder stage above), which is Vulkan — the capability this
# image already has to provide for brush itself.
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
#
# This stage never receives the repo (only the runtime stage does a
# `COPY .`), so the one repo file this layer reads has to be copied in
# explicitly. Copied on its own, not as part of the tree, so editing
# pipeline code does not invalidate this very expensive layer.
COPY pipeline/envs/sam3dbody/requirements.txt \
     /opt/b2c_runner/pipeline/envs/sam3dbody/requirements.txt

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

# gradio, for pipeline/webui.py — the web UI the container serves by
# default, and the reason the image can be built once and then driven
# entirely over HTTP.
#
# Installed HERE, in its own layer at the end of the stage, rather than up
# in the venv_main block where it belongs logically. Docker's layer cache is
# linear: venv_main is built before venv_wan22/venv_seedvr2/venv_sam3dbody,
# so touching its RUN invalidates all three, including the detectron2 layer
# that costs several minutes of nvcc. Putting a UI dependency at the front
# of that queue would mean every UI tweak pays for a detectron2 rebuild.
#
# In venv_main specifically because that is the venv the entrypoint runs the
# orchestrator with; the isolated step venvs have no use for it.
#
# Pinned to a major version: gradio's component API moves between majors
# (`gr.Timer`, `queue()` arguments and `Dataframe` datatypes have all
# changed), and this image is meant to be built once and left alone — a
# rebuild months from now silently picking up gradio 7 is exactly the class
# of surprise that costs a pod-hour.
RUN /opt/venv_main/bin/pip install 'gradio>=5.0,<7.0' \
    && /opt/venv_main/bin/python -c "import gradio; print('gradio', gradio.__version__)"

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

# Python block-buffers stdout whenever it is a pipe, which it always is
# under `docker logs` and under RunPod's log pane. Without this a step that
# runs for twenty minutes prints nothing and then prints everything, and a
# slow run is indistinguishable from a hung one. The single highest-value
# line in this file for debugging a remote run.
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Where the pipeline is allowed to write. pipeline/paths.py reads these; the
# entrypoint creates them. All of it has to be on the mounted volume: the
# container's writable layer is small, and an 81-frame batch of IPC pickles
# or a trained .ply will fill it. TMPDIR matters as much as the rest —
# SubprocessPythonDispatcher round-trips every step's inputs and outputs
# through it.
#
# The mkdir is not cosmetic and must come BEFORE the ENV: TMPDIR applies to
# the rest of the build too, and a TMPDIR that does not exist breaks any
# package whose postinst wants a temp file. Setting it without this line
# fails the very next apt layer with
#   dpkg: error processing package ca-certificates (--configure)
#   dpkg: error processing package openssh-server (--configure)
# and no mention of TMPDIR anywhere in the output. (The skeleton also means
# a container run with no volume at all still has somewhere to write, rather
# than dying on the first mkdir.)
RUN mkdir -p /data/output /data/logs /data/uploads /data/tmp /data/hf_cache /data/models
ENV B2C_DATA_DIR=/data
ENV B2C_OUTPUT_DIR=/data/output
ENV B2C_LOG_DIR=/data/logs
ENV B2C_UPLOAD_DIR=/data/uploads
# Weights whose downloader ignores HF_HOME — seedvr2's vendored one takes an
# explicit model_dir, and face_landmarks fetches MediaPipe's files straight
# from a URL. Both defaulted somewhere inside the container (a relative path
# and ~/.cache respectively), which on a pod means several GB on the small
# container disk, re-downloaded after every restart. See pipeline/paths.py.
ENV B2C_MODELS_DIR=/data/models
ENV TMPDIR=/data/tmp
ENV B2C_PORT=7860

# huggingface_hub's Xet backend caches to $HOME regardless of HF_HOME,
# filling a container's small writable layer while /data sits empty —
# downloads then die with "Disk quota exceeded". Learned on a real pod.
ENV HF_HOME=/data/hf_cache
ENV HF_HUB_DISABLE_XET=1
# Belt and braces for the same problem: if something ever re-enables Xet
# (a future huggingface_hub that ignores the flag above, or an explicit
# override at run time), this keeps its cache on the volume rather than in
# $HOME. Harmless while Xet is disabled.
ENV HF_XET_CACHE=/data/hf_cache/xet
#
# HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT set. It looks like free
# speed and is, in this stack, a no-op that costs clarity: the installed
# huggingface_hub prints a FutureWarning at every container start saying
# hf_transfer is no longer used and to enable Xet's high-performance mode
# instead — which is exactly what HF_HUB_DISABLE_XET above turns off, for
# the documented reason that Xet cached to $HOME regardless of HF_HOME and
# filled a pod's root disk. Slower downloads, correct ones.

# NOT "...:${PYTHONPATH}": nothing defines PYTHONPATH in the base image, so
# that expanded to a trailing colon, which BuildKit flags
# (UndefinedVar, the build's only warning) and which Python reads as an
# empty entry meaning the working directory:
#   PYTHONPATH="/opt/vendor/seedvr2:"  ->  ['/tmp/probe', '/opt/vendor/seedvr2',
#                                           '/tmp/probe', ...]
#   PYTHONPATH="/opt/vendor/seedvr2"   ->  ['/tmp/probe', '/opt/vendor/seedvr2',
#                                           '/usr/lib/python312.zip', ...]
# i.e. a duplicate CWD entry, not a shadowing-the-stdlib hazard — sys.path[0]
# is the script/CWD either way. Tidiness and a clean build, not a security fix.
ENV PYTHONPATH=/opt/vendor/seedvr2

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
        # ---------------------------------------------------------------
        # Debugging surface. This image is meant to be built once and then
        # left alone for the whole verification round, so anything you
        # might need *while diagnosing a failed run* has to already be
        # here — discovering that `git` is missing is itself a rebuild.
        #
        # git:      lets you `git pull` a fix into /opt/b2c_runner and
        #           re-run without rebuilding (the package is installed
        #           editable, so a pulled change takes effect immediately),
        #           and lets pip install from a git URL at all.
        # curl/wget/ca-certificates: fetching a checkpoint or poking an
        #           endpoint by hand.
        # procps/htop: `ps`, `top` — is the step running or wedged?
        # lsof:     which process is holding the GPU or a stuck file.
        # less/nano/vim-tiny: reading and patching a file in place.
        # rsync/openssh-client: getting results off the pod.
        # openssh-server: getting *into* the pod at all; a custom image
        #           has no sshd unless it ships one (see entrypoint.sh).
        # unzip/zip: the UI's dataset-archive path, by hand.
        # jq:       metadata.json is the pipeline's own manifest format.
        # tree:     seeing what a run actually wrote.
        # ---------------------------------------------------------------
        git curl wget \
        procps htop lsof \
        less nano vim-tiny \
        rsync openssh-client openssh-server \
        unzip zip jq tree \
    && rm -rf /var/lib/apt/lists/*

COPY --from=brush-builder /out-brush /usr/local/bin/brush
COPY --from=brush-builder /out-brush-splat-render /usr/local/bin/brush-splat-render
COPY --from=python-builder /opt/venv_base /opt/venv_base
COPY --from=python-builder /opt/venv_main /opt/venv_main
COPY --from=python-builder /opt/venv_wan22 /opt/venv_wan22
COPY --from=python-builder /opt/venv_seedvr2 /opt/venv_seedvr2
COPY --from=python-builder /opt/venv_sam3dbody /opt/venv_sam3dbody
COPY --from=python-builder /opt/vendor /opt/vendor

# NOT /workspace/b2c_runner, which is where this used to live. RunPod's
# pod template defaults its network volume's mount path to /workspace, and a
# volume mounted there shadows everything under it — including, in the old
# layout, the entire application. The symptom is the container exiting at
# boot with "No module named pipeline" on a pod that looks correctly
# configured. /opt is where the venvs already live and nothing mounts over
# it by default.
WORKDIR /opt/b2c_runner

# Last, and in its own layer, so editing pipeline code invalidates neither
# the venv layers nor the brush build. Every venv needs the package:
# `python -m pipeline.worker` must be importable inside each isolated env,
# not just the orchestrator's.
COPY . /opt/b2c_runner

# The container's own env registry, pointing at the venv paths this image
# actually creates. pipeline/envs/envs.yaml in the repo describes a bare
# pod (/workspace/venv_*) and is left alone for that use; cli.py defaults
# to that path, so the container copy overwrites it here rather than
# requiring every invocation to pass --envs.
COPY docker/envs.docker.yaml /opt/b2c_runner/pipeline/envs/envs.yaml

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

# 7860: the web UI. 22: sshd, started by the entrypoint only when RunPod (or
# you) supplies a PUBLIC_KEY. Both must also be listed in the pod template's
# exposed ports — EXPOSE alone documents them, it does not publish them.
EXPOSE 7860 22

# Answers "is the UI up" without a browser, and gives RunPod/compose
# something to restart on. start-period is generous because the first
# request imports torch.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${B2C_PORT}/" >/dev/null || exit 1

COPY docker/entrypoint.sh /usr/local/bin/b2c-entrypoint
RUN chmod +x /usr/local/bin/b2c-entrypoint

# CMD, not a hardcoded ENTRYPOINT argument, so RunPod's "container start
# command" (which replaces CMD) can select a different mode without having
# to override the entrypoint too. With no override the pod serves the UI and
# stays alive; `docker run IMAGE run <workflow> --dataset ...` still works
# exactly as before.
ENTRYPOINT ["/usr/local/bin/b2c-entrypoint"]
CMD ["ui"]
