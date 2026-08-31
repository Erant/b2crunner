# Single image for a RunPod (or any GPU-rental) deployment: every step's
# venv plus the brush binary, in one container. A pod IS a single container
# with no nested Docker daemon (confirmed on a real pod: no
# /var/run/docker.sock, no `docker` binary), so everything has to live here
# — see docs/docker.md's "Why one image".
#
# Build (context must be the repo root):
#   docker build -f docker/Dockerfile -t b2c/pipeline:latest \
#     --build-arg GIT_SHA="$(git rev-parse HEAD)" \
#     --build-arg GIT_DIRTY="$([ -n "$(git status --porcelain)" ] && echo -dirty)" \
#     --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
#     .
#
# The three --build-arg lines are optional (the build works without them) but
# stamp the commit onto the image as OCI labels — see "Build provenance" at
# the bottom of this file for why that is worth the typing.
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
#                   binaries; carries no nvcc and no Rust. It does carry
#                   gcc, which is NOT an oversight — Triton (and so
#                   every SageAttention backend) JIT-builds a CPython
#                   extension at runtime. See the runtime apt layer.
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
#
# BRUSH_REF is the commit that branch tip is expected to be, and it is not
# decoration: it is what makes a moved pin invalidate this stage's cache.
# `git clone --depth 1 --branch X` bakes no remote state into the RUN text,
# so Docker matches its old cache record however far the branch has moved —
# which is how `erantimus/b2crunner:c35b577` shipped a brush two commits
# behind and nothing caught it (docs/docker-build-notes.md, the 2026-08-25
# update). `--no-cache-filter brush-builder` does not fix that on its own:
# the stage re-executes, and the runtime stage's `COPY --from=brush-builder`
# still reports CACHED and keeps the old binary, because a COPY's cache key
# comes from the producing stage's *key*, not from what it produced. An ARG
# referenced in the RUN below is part of that key, so moving this line is
# what correctly invalidates both.
#
# It is also checked, which is the half `--no-cache-filter brush-builder,runtime`
# never gave: if the branch has moved past the pin, the build FAILS here
# rather than producing an image whose brush is a surprise. Moving the pin is
# a one-line edit; reconstructing which commit a shipped binary came from
# meant diffing its `--help` against the branch's history.
ARG BRUSH_BRANCH=normal-map-supervision
ARG BRUSH_REF=8eed7a61f48bbfb92a7326b52b5f86692d1d40ec
RUN git clone --depth 1 --branch ${BRUSH_BRANCH} \
        https://github.com/Erant/brush.git /src/brush \
    && cd /src/brush \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "${BRUSH_REF}" ]; then \
           echo "brush ${BRUSH_BRANCH} is at $head, not the pinned ${BRUSH_REF}." >&2; \
           echo "Either the branch moved (update BRUSH_REF, and check steps/brush.py" >&2; \
           echo "and steps/splat.py still match its CLI) or --build-arg BRUSH_REF is stale." >&2; \
           exit 1; \
       fi
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
#
# Both halves have to come from the SAME build, and from one past the
# confidence work on this branch (2026-08-30). `brush` writes per-Gaussian
# multi-view evidence into the exported .ply (--export-evidence) and
# `brush-splat-render --confidence` gates a render on it — that pair is
# what replaced the pipeline's mask_splat stage, so an image built from an
# older tip renders frames nothing downstream then filters. It degrades
# loudly rather than silently: an old renderer rejects --confidence
# outright, and a new renderer against an old training's .ply warns that it
# found no ev_* block and falls back to plain alpha.
#
# 2026-08-30: and past the MIXED ALPHA MODES commit (8eed7a61), which is
# what BRUSH_REF above pins. steps/brush.py stopped passing --alpha-mode so
# that a run can mix masked and transparent views, and passes
# --normalize-masked-loss when it does; both are that commit's. An older
# binary rejects the second flag outright and silently flattens the mix on
# the first, so the pin is the check that they arrive together.
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

ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
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

# The rest of the shared set. The transformers floor satisfies every step
# that asks for it (rmbg/wan22 want >=4.46, sapiens2 wants >=4.56).
#
# The <6 ceiling is the same defence as gradio's below, and it is not
# hypothetical here: this constraint was written as a bare `>=4.56.0` and
# had already floated to transformers **5.15.1** by the time anyone looked.
# It happens to work, so the pin records where it landed rather than
# dragging it back — but a rebuild months from now silently picking up
# transformers 6 is exactly the class of surprise that costs a pod-hour,
# and this image is meant to be built once and left alone. It is also the
# same failure mode that already bit torchao (see venv_wan22 below), where
# an unpinned floor floated onto a release built for a different torch and
# took `import diffusers` down with it.
#
# opencv-python (not -headless) to match pyproject.toml's declared
# dependency — pip treats the two as different distributions, so mixing
# them installs both.
RUN /opt/venv_base/bin/pip install \
        numpy \
        opencv-python \
        Pillow \
        'transformers>=4.56.0,<6' \
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
# torchao is PINNED, and the pin must move IN LOCKSTEP with
# TORCH_VERSION — this is a hard version coupling, not a preference. The
# failure mode is severe and non-obvious: a torchao built for a newer torch
# than the one installed dies on import with
#   ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'
# and transformers' modeling_utils imports quantizer_torchao
# unconditionally, so a torchao that is *installed but unimportable* takes
# `import diffusers` down with it. The whole wan22_vace_denoise step — the
# single most expensive thing in the pipeline — then cannot import at all.
# An absent torchao would be fine (the step guards for that); a broken one
# is not. This actually happened at torch 2.9.1 with an unpinned
# 'torchao>=0.16.0' floating up to 0.18.0.
#
# 0.18.0 pairs with torch 2.13.0. (For the record: 0.15.0 was the last
# release that worked on the old torch 2.9.1 floor, and 0.14.1 pairs with
# 2.9.0.) Re-check this pin whenever TORCH_VERSION moves.
#
# torchao >= 0.16 is also what resolves the "version triangle" that used to
# make the fp8 path unusable — LoRA onto quantized weights, and safetensors
# serialization of quantized weights, both of which diffusers gates on
# exactly that version. See docs/fp8-quant-notes.md.
#
# sageattention is NOT installed here — it is built from source in its own
# layer near the end of this stage (search for SAGE_REF). It cannot come
# from PyPI at all: the requirements ask for >=2.1.1 and PyPI stops at
# 1.0.6 (Nov 2024), because 2.x was never published there. It is also
# deliberately not in THIS layer, which sits upstream of the detectron2
# compile — see the sage layer's own comment for why.
RUN make-child-venv /opt/venv_wan22 \
    && /opt/venv_wan22/bin/pip install \
        'diffusers>=0.36.0' \
        'accelerate>=1.0.0' \
        'peft>=0.13.0' \
        ftfy \
        'torchao==0.18.0' \
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
#   - MoGe (sam3d_body's FOV / focal-length estimator, INSTALL.md step 5)
#     PINNED to b942f00 + --no-deps: `main` is now MoGe-3, whose deps drag
#     in an nvcc-built extension, gradio>=6, and a git package named
#     `pipeline` that would shadow this repo's own. Only moge.model.v2 is
#     on the inference path; it needs utils3d (also pinned) + torch/scipy.
#     steps/sam3d_body.py re-implements upstream's tools/build_fov_estimator
#     rather than importing it — detectron2 ships its own top-level `tools`
#     package here, which wins over the vendored repo's (only .pth-appended).
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
    && /opt/venv_sam3dbody/bin/pip install --no-deps \
        'moge @ git+https://github.com/microsoft/MoGe.git@b942f00' \
    && /opt/venv_sam3dbody/bin/pip install \
        'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183' \
    && git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git \
        /opt/vendor/sam3dbody \
    && echo /opt/vendor/sam3dbody > \
        $(/opt/venv_sam3dbody/bin/python -c "import site; print(site.getsitepackages()[0])")/sam3dbody_vendor.pth \
    && /opt/venv_sam3dbody/bin/python -c "from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body" \
    && /opt/venv_sam3dbody/bin/python -c "from moge.model.v2 import MoGeModel; print('moge v2 import OK')" \
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

# -----------------------------------------------------------------------------
# sageattention — built from source, because there is no other way to get it
# -----------------------------------------------------------------------------
# diffusers' attention dispatcher requires sageattention >= 2.1.1 for ANY
# sage backend variant, and PyPI's newest is 1.0.6: 2.x was never published
# there, only tagged on GitHub. So `pip install sageattention` can never
# satisfy the constraint, and the package was simply absent from the image —
# every sage backend raised, the step's try/except swallowed it, and the
# most expensive step in the pipeline quietly ran on plain SDPA.
#
# Measured on an RTX 4070 Ti (SM89), 40 heads x 128 dim, vs PyTorch SDPA:
#
#   seq    SDPA     triton int8+fp16      CUDA int8+fp8
#   1024   0.37ms   0.26ms (cos .99992)   0.29ms (cos .99930)
#   4096   4.80ms   2.98ms (cos .99991)   2.51ms (cos .99928)
#   9216  22.32ms  12.75ms (cos .99991)   9.40ms (cos .99926)
#
# i.e. ~1.75x on the Triton kernel the step selects for SM89, ~2.4x on the
# fp8 CUDA one. Both track SDPA closely. See pipeline/steps/
# wan22_vace_denoise.py's _select_sage_backend for which is chosen and why.
#
# LAST IN THE STAGE, like the gradio layer above and for the same reason:
# Docker's layer cache is linear, so putting a 4-minute nvcc build ahead of
# venv_sam3dbody would make every touch of it re-pay detectron2's compile.
#
# Arch list is NOT the stage-wide TORCH_CUDA_ARCH_LIST. SageAttention 2.2.0
# supports {8.0, 8.6, 8.9, 9.0, 12.0} and handles anything else by silently
# emitting no gencode for it (its SUPPORTED_ARCHS set is declared and then
# never used, so there is no error to catch — the kernels just aren't
# there). 10.0 — datacenter Blackwell, B100/B200/GB200 — is in this image's
# default arch list and is NOT supported, so it is dropped here explicitly
# rather than silently. On such a pod the sage backends have no kernel and
# the step falls back to SDPA, which is the pre-existing behaviour.
# Installed into BOTH venvs that can use it: wan22 dispatches sage through
# diffusers, and seedvr2's vendored inference_cli probes for it at import
# time ("SageAttention ❌" in its optimizations banner when absent). The
# wheel is compiled once and installed twice rather than built per venv.
ARG SAGE_REF=v2.2.0
ARG SAGE_CUDA_ARCH_LIST="8.9;12.0"
# NOTE the deliberate absence of `cd /tmp/SageAttention`: pip is given the
# source path instead. With the checkout as the working directory, a later
# `import sageattention` resolves to the SOURCE TREE rather than the
# installed package, and the source tree has no compiled `_fused`, so the
# verification below fails with a bogus "cannot import name '_fused' ...
# most likely due to a circular import" and the whole layer errors out on a
# perfectly good build. Cost an image build to find; keep pip path-based.
RUN git clone --depth 1 --branch ${SAGE_REF} \
        https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && TORCH_CUDA_ARCH_LIST="${SAGE_CUDA_ARCH_LIST}" MAX_JOBS="$(nproc)" \
       /opt/venv_wan22/bin/pip wheel --no-build-isolation --no-deps \
       -w /tmp/sagewheel /tmp/SageAttention \
    && /opt/venv_wan22/bin/pip install --no-deps /tmp/sagewheel/sageattention-*.whl \
    && /opt/venv_seedvr2/bin/pip install --no-deps /tmp/sagewheel/sageattention-*.whl \
    && /opt/venv_wan22/bin/python -c "\
from diffusers.models.attention_dispatch import _CAN_USE_SAGE_ATTN; \
import importlib.metadata as md; \
print('sageattention', md.version('sageattention'), '| diffusers accepts it:', _CAN_USE_SAGE_ATTN); \
assert _CAN_USE_SAGE_ATTN, 'diffusers still refuses the installed sageattention'" \
    && rm -rf /tmp/SageAttention /tmp/sagewheel

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

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# The /data skeleton. Stays HERE, above the apt layer, rather than moving
# down with the ENVs that name these paths: a container run with no volume
# mounted still needs somewhere to write, and this is a filesystem layer, not
# metadata — moving it would rebuild everything below for no benefit.
RUN mkdir -p /data/output /data/logs /data/uploads /data/tmp /data/hf_cache /data/models \
        /data/caches/triton /data/caches/nv /data/caches/torch /data/caches/xdg /data/caches/mpl
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
        python3 \
        # gcc + python3-dev: a deliberate exception to this stage's "no
        # compiler" rule, and it is load-bearing, not convenience. EVERY
        # SageAttention backend routes its per-block int8 quantization
        # through Triton, and Triton JIT-builds a small `cuda_utils` CPython
        # extension on first use. With no compiler that dies with
        #   RuntimeError: Failed to find C compiler
        # the step's try/except swallows it, and the most expensive step in
        # the pipeline silently runs on plain SDPA — i.e. the whole
        # sageattention layer in the builder stage buys exactly nothing.
        # Verified: all sage backends fail in this image without these and
        # work with them.
        #
        # Cost is ~276 MB (~5% of the image) against ~1.75x on attention in
        # wan22_vace_denoise. gcc alone is NOT enough: Triton's generated
        # source needs libc headers (fatal error: stdlib.h) and Python.h,
        # which is what python3-dev pulls in. There is no build-time
        # prewarm alternative — that compile needs a live GPU, which a
        # docker build does not have.
        gcc python3-dev \
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

COPY --from=python-builder /opt/venv_base /opt/venv_base
COPY --from=python-builder /opt/venv_main /opt/venv_main
COPY --from=python-builder /opt/venv_wan22 /opt/venv_wan22
COPY --from=python-builder /opt/venv_seedvr2 /opt/venv_seedvr2
COPY --from=python-builder /opt/venv_sam3dbody /opt/venv_sam3dbody
COPY --from=python-builder /opt/vendor /opt/vendor

# body2colmap is pip-installed unpinned in python-builder's venv_base, and
# BuildKit caches that layer: a rebuild that changes only application code
# ships whatever commit venv_base first resolved (the `COPY --from` keys on
# python-builder's cache key, not on what it produced — the same trap the
# brush notes describe). Pull the intended commit forward here in a cheap
# late layer so the image can track it without a full python-builder
# rebuild. The child venvs see it through their .pth into venv_base.
# --no-deps: body2colmap's deps (numpy, pyrender, trimesh, opencv-python)
# are unchanged and already present. Bump this SHA when the pipeline needs
# a newer body2colmap; drop the layer once the python-builder install is
# itself pinned.
#
# The pipeline needs 24fe424 or later, which replaced gsplat with
# brush-splat-render: steps/render.py's `...+splat` modes pass a
# `splat_layer` into `Renderer.render_composite`, which older commits do
# not accept. The two asserts below are that, and the reason for it — an
# older body2colmap fails the first, and the second names the backend: a
# gsplat-based SplatRenderer would drag a CUDA devel base back into this
# image, and has no BINARY_NAME at all.
#
# steps/splat.py's `_rasterize` drives SplatRenderer directly and needs the
# three seams added for it — `on_fault` (hands a crashed run's temp
# directory over before deleting it, which is what paths.crash_dir() feeds
# on), `on_output` (each line as it arrives, for proc.OutputRelay) and
# `ply_path`. 4bb9dd9 ("Add crash-report and progress hooks to
# SplatRenderer", 2026-08-31) is the commit that added them.
#
# Failing the build is deliberate: an image built against an older commit
# would install cleanly and then raise on the first render_splat of a real
# run, an hour into a pod booking. The third assert below is what stops it.
RUN /opt/venv_base/bin/pip install --force-reinstall --no-deps \
        'body2colmap @ git+https://github.com/Erant/body2colmap.git@4bb9dd9c68b4e0aeb2fab6f52b638867f7608f55' \
    && /opt/venv_base/bin/python -c "\
import inspect; \
import body2colmap.face as f; \
from body2colmap.renderer import Renderer; \
import body2colmap.splat_renderer as sr; \
assert f.EYE_STYLES == ('shape', 'dots'), f.EYE_STYLES; \
assert 'splat_layer' in inspect.signature(Renderer.render_composite).parameters; \
assert sr.BINARY_NAME == 'brush-splat-render', sr.BINARY_NAME; \
assert 'on_fault' in inspect.signature(sr.SplatRenderer.__init__).parameters, \
    'bump the body2colmap pin: steps/splat.py needs on_fault/on_output/ply_path'; \
print('body2colmap eye_style default:', f.DEFAULT_EYE_STYLE)"

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

# ---------------------------------------------------------------------------
# The brush binaries. LAST OF THE FILESYSTEM LAYERS ON PURPOSE — same
# reasoning as the ENV block below, and measured on the same kind of build.
# ---------------------------------------------------------------------------
# These used to sit at the top of the stage, immediately above the five
# `COPY --from=python-builder /opt/venv_*` lines. Invalidation cascades
# forward within a stage, so bumping BRUSH_REF — 321 MB of binaries — re-ran
# every venv copy, the body2colmap pull-forward, and the application copy
# behind it. What saved that build was the registry rather than the
# Dockerfile: a re-executed COPY of unchanged bytes produces the same blob
# digest, so the venvs all came back "Layer already exists" and only the
# binaries went up the wire. A re-executed RUN has no such property — its
# output is not byte-reproducible — which is exactly why the 2026-08-25
# `--no-cache-filter brush-builder,runtime` build pushed 7 GB while a
# BRUSH_REF bump pushes ~321 MB.
#
# So the dedup was luck of the layer types, and it runs out the moment a RUN
# lands between the brush COPY and the end of the stage. Down here nothing
# does: a brush bump now invalidates these two layers, the ENV/LABEL metadata
# below (no filesystem layers at all), and nothing else.
#
# Nothing earlier in the build needs the binaries — the step-registry smoke
# test above imports Python modules, and `pipeline/doctor.py` checks the
# binaries at container start, not at build time.
COPY --from=brush-builder /out-brush /usr/local/bin/brush
COPY --from=brush-builder /out-brush-splat-render /usr/local/bin/brush-splat-render

# Build provenance. Deliberately the LAST thing in the file: a LABEL makes a
# layer, and every layer after it would be invalidated by a new SHA. Here, a
# rebuild at a different commit redoes this metadata layer and nothing else.
#
# Without these, the only labels on the image are the ones inherited from the
# nvidia/cuda base (maintainer, cudnn.version, ...), every build looks
# identical to `docker inspect`, and identifying which image matches the
# working tree means hashing /opt/b2c_runner inside each one by hand.
#
# All three default to empty so a bare `docker build -f docker/Dockerfile .`
# still succeeds; the header's build command fills them in. GIT_DIRTY is the
# "-dirty" suffix `git describe` would add — an image built from an unclean
# tree does NOT correspond to its commit, and that has to be visible on the
# image, not just remembered.
#
# Read it back with:
#   docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' b2c/pipeline:latest

# ---------------------------------------------------------------------------
# Runtime configuration. LAST ON PURPOSE — same reasoning as the LABEL block
# below, and it is the more expensive of the two to get wrong.
# ---------------------------------------------------------------------------
# ENV does not create a filesystem layer, but it does change the image config
# that every subsequent RUN and COPY is cache-keyed on. This block used to sit
# near the top of the stage, above the apt layer and above the six
# `COPY --from=python-builder` venv copies. That meant editing one variable
# rebuilt all of it — and, worse, re-uploaded it: a re-executed COPY produces
# a fresh layer digest even when the source tree is byte-identical, so the
# registry saw new blobs rather than deduplicating.
#
# Measured on two consecutive builds of the same image. Editing
# HF_HUB_DISABLE_XET here (two lines) rebuilt 25 of 26 layers and pushed
# 7.97 GB. The next commit touched only `COPY . /opt/b2c_runner` near the end
# and pushed 26 MB. Same image, same size; the whole difference was where the
# edited line sat.
#
# So: anything read only by a RUNNING container belongs here. Anything the
# BUILD reads — DEBIAN_FRONTEND for apt, PIP_* for the editable installs,
# PYTHONPATH for the step-registry smoke test — has to stay above its
# consumer, and does.
#
# One behaviour change fell out of the move, and it is an improvement: TMPDIR
# no longer applies during the build. It used to, and the comment on the mkdir
# above records what that cost — a TMPDIR pointing at a directory that did not
# exist yet failed the next apt layer with an unrelated-looking dpkg error.
# The build now uses the default /tmp and cannot hit that at all.

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

ENV HF_HOME=/data/hf_cache
# Xet used to cache to $HOME regardless of HF_HOME, filling a container's
# small writable layer while /data sat empty — downloads died with "Disk
# quota exceeded", learned on a real pod. That is why this file used to set
# HF_HUB_DISABLE_XET=1.
#
# It no longer does, because the cost of that workaround turned out to be
# large and the reason for it has gone away:
#
#   * The bug is fixed upstream. huggingface_hub now derives the Xet cache
#     from HF_HOME (`default_xet_cache_path = os.path.join(HF_HOME, "xet")`
#     in huggingface_hub/constants.py), so it lands on the volume by
#     default. HF_XET_CACHE below pins it there explicitly anyway, which
#     makes this independent of which hub version the build resolves.
#   * Disabling Xet does not mean "download the old way". A Xet-backed repo
#     redirects a non-Xet client to the `xet-bridge-*` CDN endpoint, which
#     serves the file as ONE HTTP stream. wan22_vace_denoise's two fp8
#     experts are 17.58 GB each and live in a Xet-backed repo
#     (silveroxides/Wan_2.2-fp8_scaled_hybrid — a HEAD on either file
#     returns x-xet-hash and a rel="xet-auth" link), so that workaround was
#     costing ~35 GB of single-stream transfer on every cold pod, which is
#     most of the prefetch wall time.
ENV HF_XET_CACHE=/data/hf_cache/xet
# HF_XET_HIGH_PERFORMANCE is deliberately NOT set, and this is measured
# rather than assumed. Downloading one 17.58 GB fp8 expert with it on: RSS
# tracked bytes-downloaded 1:1 and NOTHING reached disk until the end
# (4.52 GB resident at 110s, i.e. a ~17.6 GB peak projected for the full
# file). With it off, the same download flushes progressively — resident
# set sawtooths between 0.7 and 2.4 GB while the blob grows on disk — at
# the same throughput. On a link fast enough to be the bottleneck the mode
# may still be worth it, but not on a pod where `doctor` already WARNs that
# host RAM is tight against wan22's ~47 GB resident set.
#
# ---------------------------------------------------------------------------
# Caches that default into $HOME, i.e. onto the pod's small ephemeral disk.
# ---------------------------------------------------------------------------
# Same class of bug as the Xet cache above, found by auditing what a run
# actually writes outside /data. None of these are large, so this is not a
# disk-quota fix — it is a "don't redo work every pod restart" fix, plus
# keeping the ephemeral layer free of anything worth keeping.
#
# TRITON_CACHE_DIR is the one that now matters most: every SageAttention
# backend goes through Triton, which JIT-compiles its kernels on first use
# and caches them (~1.2 MB) under ~/.triton. On the ephemeral disk that
# means re-JITting on every single pod start, in the middle of the most
# expensive step. CUDA_CACHE_PATH is the driver's own PTX JIT cache
# (~/.nv/ComputeCache), same story.
#
# TORCH_HOME, XDG_CACHE_HOME and MPLCONFIGDIR are belt and braces: nothing
# in the pipeline is known to use torch.hub today, but timm and
# matplotlib-using vendored code both write under $HOME by default, and a
# future step that does would silently land on the ephemeral disk.
ENV TRITON_CACHE_DIR=/data/caches/triton
ENV CUDA_CACHE_PATH=/data/caches/nv
ENV TORCH_HOME=/data/caches/torch
ENV XDG_CACHE_HOME=/data/caches/xdg
ENV MPLCONFIGDIR=/data/caches/mpl

# HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT set, and the `hf_transfer`
# package installed above is now dead weight. The installed huggingface_hub
# does not read that variable as a transfer backend at all — it only checks
# it in order to print a FutureWarning telling you to use
# HF_XET_HIGH_PERFORMANCE instead, which is what the Xet block above now
# sets. Left uninstalled-but-unremoved only because several child venvs'
# requirements.txt still list it; removing it there is a separate change.

ARG GIT_SHA=""
ARG GIT_DIRTY=""
ARG BUILD_DATE=""
LABEL org.opencontainers.image.title="b2c/pipeline" \
      org.opencontainers.image.source="https://github.com/Erant/b2crunner" \
      org.opencontainers.image.revision="${GIT_SHA}${GIT_DIRTY}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# CMD, not a hardcoded ENTRYPOINT argument, so RunPod's "container start
# command" (which replaces CMD) can select a different mode without having
# to override the entrypoint too. With no override the pod serves the UI and
# stays alive; `docker run IMAGE run <workflow> --dataset ...` still works
# exactly as before.
ENTRYPOINT ["/usr/local/bin/b2c-entrypoint"]
CMD ["ui"]
