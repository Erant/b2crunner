# The application image: the pipeline code, its in-house pins (body2colmap,
# b2cgltf, b2ctrain, b2cviewer, b2crig) and the runtime configuration, on top
# of the DEPENDENCY image docker/Dockerfile.base builds (CUDA runtime, apt,
# every step's venv, COLMAP). A pod IS a single container with no nested
# Docker daemon (confirmed on a real pod: no /var/run/docker.sock, no `docker`
# binary), so everything still ends up in this one image — see
# docs/docker.md's "Why one image". The split is in how it is BUILT: this
# file never runs a dependency step, so a code change pushes a few MB, and
# the ~9 GB of dependencies move only when BASE_IMAGE is bumped on purpose.
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
# Two stages:
#   b2ctrain-builder  CUDA *devel* base -> the b2ctrain binary, nothing
#                   else: the Gaussian-splat trainer AND, through its
#                   `render` subcommand, the rasteriser every splat render
#                   in this pipeline goes through.
#   runtime         BASE_IMAGE plus the in-house pins, the binaries and the
#                   application.

# The dependency image, by DIGEST: a tag can move, a digest cannot, so this
# line is the whole record of which dependencies an app build ships. Bump it
# only to a base built and pushed from docker/Dockerfile.base (its header
# has the steps). base-20261011 = :stable's (05723f8) dependency layers.
ARG BASE_IMAGE=erantimus/b2crunner:base-20261011@sha256:8f27df40bab8021d05b0a29391b042c4ef2ffb39d5b69475d5401fdc99c024ea

# The trainer's pin, declared before the first FROM so two stages read it:
# b2ctrain-builder clones and checks it, and the runtime stage bakes it as
# an ENV for a run's log to fall back on (see "Build provenance" at the
# bottom). A global ARG has no value inside a stage until the stage says
# `ARG B2CTRAIN_REF` again, which both do. Bump it here, once.
# 0d7e5a9 (2026-09-30): cage posing, the binding export and the pose containment
# loss, which the Rig splat output (b2crig) drives, on top of e0fbc38's visibility
# cull (the cage commit rebased onto it); additive for everything else.
ARG B2CTRAIN_REF=0d7e5a9db734e62b25f70317e6c14735a8485f61

# =============================================================================
# Stage 1: b2ctrain
# =============================================================================
# Erant/b2ctrain — a from-scratch C++/CUDA trainer with brush's CLI, dataset
# layout and .ply contract (docs/STATUS.md in that repo has the measured
# numbers: ~3.9x brush's wall time at equal or better PSNR and splat count on
# both example datasets, verified against the fork on an RTX 4070 Ti, and
# through this pipeline's own step class on both trainings' argv). It is
# what helical.yaml's two trainings run since 2026-09-07, and
# its `render` subcommand is what answers to `brush-splat-render` in the
# runtime stage below.
#
# It REPLACED the Rust brush rather than joining it: there is no
# brush-builder stage any more, no `brush` binary, no Vulkan rasteriser and
# no Vulkan loader in the runtime stage. `git log -S brush-builder -- \
# docker/Dockerfile` is where that stage went if it is ever wanted back;
# the fork itself is unchanged at Erant/brush's normal-map-supervision
# branch (`a9405881`, the pin this image shipped), and steps/brush.py still
# drives brush's argv, so restoring it is re-adding the stage.
#
# The same devel base as Dockerfile.base's colmap-builder and python-builder. The
# binary links the CUDA runtime statically, so the runtime stage needs only
# the driver (>= 580 for a CUDA 13 build) and nothing from this stage but
# the file. Built for sm_89 (Ada: L40S, RTX 40xx) and sm_120 (Blackwell:
# RTX 50xx, RTX PRO) plus compute_120 PTX for whatever comes after.
#
# Pinned the way brush was, and for the same reason: `git clone --branch`
# bakes no remote state into the RUN text, so without the check a moved
# branch would keep serving the cached binary — and a COPY out of this
# stage keys on the stage's cache key rather than on what it produced, so
# a bumped pin needs `--no-cache-filter b2ctrain-builder,runtime` to reach
# the runtime stage. Move B2CTRAIN_REF (the global ARG above the first
# FROM) when the trainer moves, and re-run
# `pipeline.cli doctor` — check_trainer_binaries greps its --help for the
# eight flags steps/brush.py's argv depends on.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS b2ctrain-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*

ARG B2CTRAIN_REPO=https://github.com/Erant/b2ctrain.git
ARG B2CTRAIN_BRANCH=main
ARG B2CTRAIN_REF
RUN git clone --depth 1 --branch ${B2CTRAIN_BRANCH} ${B2CTRAIN_REPO} /src/b2ctrain \
    && cd /src/b2ctrain \
    && head="$(git rev-parse HEAD)" \
    && if [ "$head" != "${B2CTRAIN_REF}" ]; then \
           echo "b2ctrain ${B2CTRAIN_BRANCH} is at $head, not the pinned ${B2CTRAIN_REF}." >&2; \
           echo "Either the branch moved (update B2CTRAIN_REF, and check steps/brush.py" >&2; \
           echo "and steps/splat.py still match its CLI) or --build-arg B2CTRAIN_REF is stale." >&2; \
           exit 1; \
       fi
WORKDIR /src/b2ctrain
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89-real;120-real;120-virtual" \
    && cmake --build build --target b2ctrain \
    && install -m 0755 build/b2ctrain /out-b2ctrain \
    && install -m 0755 docker/brush-splat-render /out-brush-splat-render-shim


# =============================================================================
# Stage 2: runtime (BASE_IMAGE + the application)
# =============================================================================
# Inherits from the base: DEBIAN_FRONTEND, PIP_*, PYTHONPATH, the /data
# skeleton, the venvs, /opt/vendor and /opt/colmap. Nothing below may change
# a dependency in place — a package the code needs goes into
# Dockerfile.base and ships with the next base bump; the exceptions are the
# in-house pins right here, which are small and move with the code.
FROM ${BASE_IMAGE} AS runtime

# body2colmap is pip-installed unpinned in the base's venv_base (python-builder
# in Dockerfile.base), so the base carries whatever commit it first resolved.
# Pull the intended commit forward here in a cheap layer so the image can
# track it without a base bump. The child venvs see it through their .pth into venv_base.
# --no-deps: body2colmap's deps (numpy, pyrender, trimesh, opencv-python)
# are unchanged and already present. Bump this SHA when the pipeline needs
# a newer body2colmap.
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
# 958fd3b ("Draw a world-fixed backdrop behind the render", 2026-09-01) adds
# `body2colmap.background`, which steps/backdrop.py builds both renderers'
# environment out of, and the `background=` seam on `Renderer` that carries
# it under a composite's base layer. Both step defaults are a grid cube, so
# an older commit is not a degraded run, it is an ImportError on the first
# render.
#
# d66b8fa ("Document the plain-target contract and the flat generator form",
# 2026-09-04) adds `body2colmap.fade` and the `fade=` seam on Background,
# which steps/backdrop.py's `build_fade` clears the room around the subject
# with. `background_fade` defaults ON for the `render` step, so an older
# commit is a TypeError out of `Background.create` on the first frame of
# stage 1.
#
# 637a6a6 ("Draw the skeleton the way DWPose draws one") adds
# `body2colmap.skeleton`'s get_bone_colors_dwpose / get_skeleton_bones_dwpose
# and the `style=` seam on Renderer.render_skeleton. steps/render.py's
# `skeleton_style` defaults to "dwpose", so an older commit is a TypeError on
# the first skeleton render rather than a degraded one.
#
# af48029 is where those two stopped being a choice. They were branches that
# both forked off 958fd3b, neither containing the other, and this image needs
# BOTH — the skeleton overlay draws in DWPose's convention on the same render
# that fades the room. body2colmap landed them together on MAIN (the DWPose
# work rebased on top of the fade, so d084702 became 637a6a6), which is why
# this pin finally names a commit on main rather than a topic branch.
#
# 60d169e ("SplatScene: carry extra vertex properties through the PLY round
# trip", 2026-09-07) keeps brush's ev_* evidence block across a
# SplatScene.from_ply/to_ply round trip. Any step that stages a scene in
# memory before a `confidence: true` render used to lose it, and the gate
# then degenerated to plain alpha behind a one-line warning. The workflow
# hands rerender_splat the trained file directly since 1547a4a, so an older
# commit is a degraded render rather than an error — which is exactly why
# the pin, not an assert, is what carries it.
#
# 81a0e1b ("Mark the splat as inactive in the frame's alpha", 2026-09-07)
# adds `InactiveMaskOptions`, the mask steps/render.py's `splat_inactive_mask`
# publishes: 0 over the face splat ("already real, keep it") against 1
# everywhere else. An older commit is an ImportError inside the render step,
# but only for a run that switched the flag on — it defaults off.
#
# 76a74bc ("Draw the outline from a supplied mask", 2026-09-08) adds
# `mask=` to `Renderer.render_outline` and forwards `modes["outline"]["mask"]`
# through `render_composite`. steps/render.py's `outline_masks` input — the
# re-outline branch of helical.yaml — passes a matte-derived
# silhouette per frame that way. An older commit ignores the key silently
# (`render_composite` reads its options with `.get`) and draws the mesh
# outline as before: a run that switched `re_outline` on would spend a third
# denoise and change nothing, with no error to say so. The assert below is
# what turns that into a failed build.
#
# Failing the build is deliberate: an image built against an older commit
# would install cleanly and then raise on the first render_splat of a real
# run, an hour into a pod booking. The last five asserts below are what stop it.
RUN /opt/venv_base/bin/pip install --force-reinstall --no-deps \
        'body2colmap @ git+https://github.com/Erant/body2colmap.git@4c98a532e45bc27cfc5ec4faaab621c2e34320e8' \
    && /opt/venv_base/bin/python -c "\
import inspect; \
import body2colmap.face as f; \
from body2colmap.renderer import Renderer; \
import body2colmap.splat_renderer as sr; \
from body2colmap.background import Background, TEXTURE_GENERATORS; \
assert f.EYE_STYLES == ('shape', 'dots'), f.EYE_STYLES; \
assert 'splat_layer' in inspect.signature(Renderer.render_composite).parameters; \
assert sr.BINARY_NAME == 'brush-splat-render', sr.BINARY_NAME; \
assert 'on_fault' in inspect.signature(sr.SplatRenderer.__init__).parameters, \
    'bump the body2colmap pin: steps/splat.py needs on_fault/on_output/ply_path'; \
assert 'sh_degree' in inspect.signature(sr.SplatRenderer.__init__).parameters, \
    'bump the body2colmap pin: render_splat needs SplatRenderer(sh_degree=...)'; \
assert 'grid' in TEXTURE_GENERATORS, sorted(TEXTURE_GENERATORS); \
assert 'background' in inspect.signature(Renderer.__init__).parameters, \
    'bump the body2colmap pin: steps/backdrop.py needs Renderer(background=...)'; \
assert 'style' in inspect.signature(Renderer.render_skeleton).parameters, \
    'bump the body2colmap pin: steps/render.py needs skeleton style=dwpose'; \
assert 'fade' in inspect.signature(Background.create).parameters, \
    'bump the body2colmap pin: steps/backdrop.py needs Background.create(fade=...)'; \
assert (sr.INACTIVE, sr.REACTIVE) == (0, 255) \
    and sr.InactiveMaskOptions().threshold == 0.9, \
    'bump the body2colmap pin: steps/render.py needs InactiveMaskOptions, and ' \
    'documents its polarity and 0.9 threshold as the ones it publishes'; \
assert 'mask' in inspect.signature(Renderer.render_outline).parameters, \
    'bump the body2colmap pin: steps/render.py needs render_outline(mask=...) ' \
    'for the outline_masks input, or the re-outline branch silently draws the mesh'; \
assert 'relief' in inspect.signature(Renderer.render_outline).parameters \
    and 'occlusion_tolerance' in inspect.signature(Renderer.render_skeleton).parameters, \
    'bump the body2colmap pin: steps/render.py needs render_outline(relief=...) and ' \
    'render_skeleton(occlusion_tolerance=...) for outline_relief / skeleton_occlusion_m'; \
print('body2colmap eye_style default:', f.DEFAULT_EYE_STYLE)"

# b2cgltf: the subject file (ply/scene.glb) the export_subject step writes, and
# tools/export_glb.py's writer (b2cgltf SPEC.md; Erant/b2cgltf, shared with
# b2crig). Into venv_base, so venv_main (the in_process step) and
# venv_sam3dbody (the tool) both see it through their .pth. A late layer here,
# not in the base, for the body2colmap layer's reason above: a pin bump
# then re-runs this one small layer and no venv. Pinned by commit, as
# b2ctrain is: the file format is a contract with b2crig. numpy is its only dependency, already in
# venv_base.
ARG B2CGLTF_REF=ad878fcb2b47a23bb49b070b9373accde3d2de34
RUN /opt/venv_base/bin/pip install --no-deps \
        "b2cgltf @ git+https://github.com/Erant/b2cgltf.git@${B2CGLTF_REF}" \
    && /opt/venv_base/bin/python -c "import b2cgltf.b2crunner.subject"

# Above the application copy too, and a COPY --chmod rather than COPY + RUN
# chmod: a RUN re-executed below `COPY . /opt/b2c_runner` makes a new blob on
# every code change, however small.
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/b2c-entrypoint

# ---------------------------------------------------------------------------
# The prebuilt pieces: the trainer binary, the viewer, the rig. BELOW every
# RUN of this stage and ABOVE the application copy — both halves measured.
# ---------------------------------------------------------------------------
# Invalidation cascades forward within a stage, and a re-executed layer is
# not a free one. That goes for RUNs (their output is not byte-reproducible,
# which is why the 2026-08-25 `--no-cache-filter brush-builder,runtime` build
# pushed 7 GB) AND, measured 2026-09-30, for `COPY --from` too: with these
# below `COPY . /opt/b2c_runner`, a one-line code change re-executed them and
# the COLMAP layer (in the base since 2026-10-11) came out a new 263 MB
# (compressed) blob, the trainer a new 6 MB one. The registry dedup these
# comments used to count on does not hold.
#
# So: below the RUNs, so bumping any of these re-runs no pip layer.
# Above the code, so a code-only image re-pushes the code copy, the envs.yaml
# and the editable install (~2 MB) and nothing else. Among themselves, the
# rarest-bumped first: a B2CTRAIN_REF bump re-pushes the trainer, the viewer,
# b2crig and the code; a B2CVIEWER_REF bump the viewer, b2crig and the code; a
# B2CRIG_REF bump b2crig and the code.
#
# Nothing in the build needs the binaries — the step-registry smoke test
# imports Python modules, and `pipeline/doctor.py` checks the binaries at
# container start, not at build time.

# b2ctrain (see its stage above): the trainer steps/brush.py runs, and —
# via the one-line shim installed under the name every renderer in this
# pipeline resolves (steps/splat.py's _RENDER_BINARY, body2colmap's
# BINARY_NAME, `pipeline.cli doctor`) — the rasteriser.
# `brush-splat-render` execs `b2ctrain render "$@"`: same argv, same
# cameras.json, same frame naming, same --confidence gate, output matching
# the Rust renderer it replaced to 1/255. Two names, one binary, and
# nothing left in the image that needs Vulkan.
#
# `--help` on both here is the cheapest possible smoke test: no GPU is
# needed for it, and it proves the binary loads in THIS stage's userspace
# (it links only libstdc++/libm/libc — the CUDA runtime is static) and
# that the shim finds b2ctrain on PATH. It is a RUN below the apt layers,
# which the note above warns about — harmless here only because it writes
# nothing: the layer it makes is empty, so a re-execution has no bytes to
# push.
COPY --from=b2ctrain-builder /out-b2ctrain /usr/local/bin/b2ctrain
COPY --from=b2ctrain-builder /out-brush-splat-render-shim /usr/local/bin/brush-splat-render
RUN b2ctrain --help > /dev/null && brush-splat-render --help > /dev/null

# b2cviewer (Erant/b2cviewer): the 3D viewer the web UI serves at /viewer/
# (pipeline/viewer.py; B2C_VIEWER_DIR defaults to this path). Static files
# plus a stdlib serve.py whose index() the UI imports — nothing to install.
# Last of the prebuilt pieces (the note above): a B2CVIEWER_REF bump re-runs
# this ~0.1 MB layer and the application layers, nothing else. An ADD of a git
# source rather than a RUN git clone: no git needed in this stage, and
# BuildKit caches it by the commit. Pinned by commit, as b2cgltf is: the
# page reads b2cgltf's files, so the two move together.
ARG B2CVIEWER_REF=82b22edfdf01752f9d9eac187f4355e85520d472
ADD --keep-git-dir=false https://github.com/Erant/b2cviewer.git#${B2CVIEWER_REF} /opt/b2cviewer

# b2crig (Erant/b2crig): the rig behind the "Rig splat" output
# (steps/rig_subject.py runs its tools/rig_subject.py; /opt/b2crig is the
# step's default checkout). Nothing to install: its tools run under venv_main's
# python, which already has what they import (torch through venv_base, scipy,
# plyfile, opencv, b2cgltf), and they find b2ctrain on PATH. Pinned by commit,
# as b2cgltf is, and bumped with it: b2crig writes its half of the subject file
# through that package. The doctor's b2crig check proves the imports at start.
ARG B2CRIG_REF=d1fcf73c11719b9b82ad86f9f76240a6567f7d56
ADD --keep-git-dir=false https://github.com/Erant/b2crig.git#${B2CRIG_REF} /opt/b2crig

# NOT /workspace/b2c_runner, which is where this used to live. RunPod's
# pod template defaults its network volume's mount path to /workspace, and a
# volume mounted there shadows everything under it — including, in the old
# layout, the entire application. The symptom is the container exiting at
# boot with "No module named pipeline" on a pod that looks correctly
# configured. /opt is where the venvs already live and nothing mounts over
# it by default.
WORKDIR /opt/b2c_runner

# Last, and in its own layer, so editing pipeline code invalidates neither
# the venv layers nor the trainer build. Every venv needs the package:
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
# here to build anything that isn't. The one RUN below the application copy,
# so it is re-pushed with every code change: PYTHONDONTWRITEBYTECODE and
# --no-compile keep it to the editable-install files (~0.1 MB). Without them
# pip's own modules and the smoke import's left ~22 MB of .pyc in the venvs.
RUN export PYTHONDONTWRITEBYTECODE=1; \
    for v in main wan22 seedvr2 sam3dbody; do \
        /opt/venv_${v}/bin/pip install --no-deps --no-compile -e . || exit 1; \
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
#
# A label is visible to `docker inspect` and to nothing inside the
# container, so the same revision is ALSO an ENV, B2C_GIT_REVISION, which is
# what pipeline/provenance.py reads when there is no .git to ask (the image
# is built from a `git archive`) — that is how the `versions:` line at the
# top of every run's log names the b2crunner commit. B2CTRAIN_REF rides
# along as the fallback for a trainer whose `--version` predates b49480f
# and does not name its own commit. Neither ENV makes a filesystem layer;
# both sit here, last, for the reason the LABEL does.

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
# This gates the pyrender `render` step (EGL) and nothing else since
# 2026-09-07: the trainer and the splat rasteriser are both b2ctrain, which
# is CUDA and wants only `compute`. A pod that exposes compute,utility
# alone therefore trains and renders splats fine, and fails at `render`
# instead of 40 minutes into a training. Whether RunPod honours
# an image-level ENV for it is unresolved; setting it in the pod template's
# own environment variables is the more reliable path, since that is
# applied at creation time. See docs/docker-build-notes.md.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# COLMAP, which lives under its own prefix so its bundled ONNX Runtime
# stays off the global loader path (see Dockerfile.base's colmap-builder
# stage and its COPY). This ENV is what puts it on PATH, where
# pipeline/steps/refine_cameras.py's default `colmap_path` and
# pipeline/doctor.py's colmap check both look for it. Here rather than as a
# symlink because an ENV makes no filesystem layer.
ENV PATH=/opt/colmap/bin:${PATH}

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
ARG B2CTRAIN_REF
LABEL org.opencontainers.image.title="b2c/pipeline" \
      org.opencontainers.image.source="https://github.com/Erant/b2crunner" \
      org.opencontainers.image.revision="${GIT_SHA}${GIT_DIRTY}" \
      org.opencontainers.image.created="${BUILD_DATE}"
ENV B2C_GIT_REVISION="${GIT_SHA}${GIT_DIRTY}" \
    B2CTRAIN_REF="${B2CTRAIN_REF}"

# CMD, not a hardcoded ENTRYPOINT argument, so RunPod's "container start
# command" (which replaces CMD) can select a different mode without having
# to override the entrypoint too. With no override the pod serves the UI and
# stays alive; `docker run IMAGE run <workflow> --dataset ...` still works
# exactly as before.
ENTRYPOINT ["/usr/local/bin/b2c-entrypoint"]
CMD ["ui"]