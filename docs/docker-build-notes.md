# Docker build notes — read before the first build

Written for a session that will run on a box with an **RTX 4070 Ti**, which
is the first machine this project has had with a local GPU. `docker/Dockerfile`
was rewritten against these notes but **has never been built**. Nothing
below is a verified recipe; it is a test plan plus the reasoning behind
every choice that could plausibly be wrong.

`docs/docker.md` remains the narrative for *why one image*; this file is
about *getting the thing to build*.

## The machine

| | |
|---|---|
| GPU | RTX 4070 Ti — Ada, **compute capability 8.9**, **12 GB VRAM** |
| Arch list | `TORCH_CUDA_ARCH_LIST=8.9` is the Dockerfile default, which is exactly right for this box (and for the L40S used on RunPod — also 8.9) |
| Driver | **CUDA 13 needs host driver >= 580.** Check `nvidia-smi` first. An old driver fails at *container start*, not at build, which is a confusing way to discover it |

12 GB is the thing to keep in mind. Every prior verification run happened
on an L40S with 44 GB. Two steps are unlikely to fit here:

- **`wan22_vace_denoise`** — a 14B dual-expert VACE model in fp8, verified
  at 81 frames of 720x1280 on 44 GB. On 12 GB, expect OOM. Testing it
  probably means dropping resolution and frame count, which the step's own
  docstring warns changes output quality (a short clip produces visibly
  worse results — not a bug, an invalid test size for this model/LoRA
  pairing). So a "does it run" test here is not a "does it produce good
  output" test.
- **`seedvr2`** — needed `vae_encode_tiled`/`vae_decode_tiled` at 1440 on
  44 GB, because an untiled full-res VAE decode OOM'd outright. On 12 GB,
  keep tiling on and expect to need a lower `resolution`.

What *is* comfortably testable on 12 GB: `sam3d_body`, `rmbg`,
`sapiens2_lite`, `render`, `brush`, and the whole in-memory orchestration
path. That includes most of what has never been executed at all, so this
box is well matched to the actual gap.

`render_splat` is the exception, and not for VRAM reasons: gsplat is gone
and its brush-based replacement does not exist yet, so that step will fail
at import until it does. See "gsplat: dropped" below.

## Build order, and what to check at each step

Do NOT start with a full `docker build`. The image is large and the risky
layers are near the end. Work up in three stages.

### 1. The capability probe (~10 min, settles the oldest open question)

`NVIDIA_DRIVER_CAPABILITIES` gates **both** brush (Vulkan) and `render`
(EGL). It is read by nvidia-container-toolkit at container-*creation*
time; setting it inside a running container does nothing. Locally you
control this directly, which is the whole reason a local GPU box is
valuable here — it decouples "does our graphics setup work at all" from
"does RunPod honour it".

```bash
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 bash -c '
    apt-get update -qq && apt-get install -y -qq \
        libvulkan1 vulkan-tools libegl1 libgl1 libglvnd0 >/dev/null
    echo "--- vulkaninfo ---"; vulkaninfo --summary 2>&1 | head -20
    echo "--- EGL ---"; python3 - <<PY
import ctypes
ctypes.CDLL("libEGL.so.1"); print("libEGL.so.1 loaded")
PY'
```

Then run it **again without** the `-e NVIDIA_DRIVER_CAPABILITIES` line. If
the first works and the second fails, that confirms the mechanism end to
end and the remaining question is purely "does RunPod let me set it",
which is a template-settings question, not an engineering one. My
expectation: setting it in the **pod template's environment variables** is
more reliable than baking `ENV` into the image, because the template value
is applied at creation time. Try both on RunPod; locally, image `ENV` is
enough.

### 2. The detectron2 probe (~15 min, the highest-risk unknown)

This is the layer most likely to fail, and it is cheap to isolate. Do not
discover it at the end of a 30 GB build.

```bash
cat > /tmp/d2probe.Dockerfile <<'EOF'
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV FORCE_CUDA=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/v && /opt/v/bin/pip install -U pip setuptools wheel \
    && /opt/v/bin/pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu130
RUN /opt/v/bin/python -c "import torch;print(torch.__version__, torch.version.cuda)"
RUN /opt/v/bin/pip install cython numpy \
    && /opt/v/bin/pip install --no-build-isolation --no-deps \
        'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9'
RUN /opt/v/bin/python -c "from detectron2 import _C; print('native ext OK')"
EOF
docker build -f /tmp/d2probe.Dockerfile -t d2probe /tmp
```

The final `from detectron2 import _C` is the real test — it is what proves
the CUDA kernels compiled and load. Without `FORCE_CUDA=1` detectron2's
`setup.py` silently builds CPU-only (it gates on
`torch.cuda.is_available()`, which is False during any docker build), so
the extension would import fine and then fail at inference. That trap is
why the check is spelled out rather than assumed.

**Fallback ladder if it fails**, in order of increasing deviation:

1. **Older torch, same CUDA 13.** `cu130` publishes torch 2.9.0 → 2.13.0.
   2.9.1 is already the conservative pick (closest to what an Aug-2025
   detectron2 commit was developed against). There is nothing older on
   cu130 to fall back to, so this rung is nearly empty — which is the
   argument for rung 2.
2. **Drop to CUDA 12.8 + torch 2.7/2.8.** `cu128` has torch 2.7.0 →
   2.11.0, an era detectron2 @a1ce2f9 definitely predates comfortably.
   Change `CUDA_WHEEL_INDEX`, `TORCH_VERSION`, `TORCHVISION_VERSION` and
   the base image to `12.8.1-cudnn-devel-ubuntu24.04` — all four are
   build args / one FROM line, so this is a cheap experiment. Note this
   also matches the toolkit the RunPod pod had, i.e. the combination
   sam3d_body was actually verified against.
3. **Newer detectron2 commit.** The repo is *actively maintained* (last
   commit was 4 days before these notes were written), so a recent commit
   is likely to build against modern torch. The cost is deviating from
   sam-3d-body's INSTALL.md, which pins `a1ce2f9` — check sam_3d_body
   still imports and runs afterwards, because that pin presumably exists
   for a reason.
4. **Give sam3dbody its own torch.** The venv-sharing scheme degrades
   gracefully: skip `make-child-venv` for that one venv (i.e. omit the
   `zz_shared_base.pth`), create a plain venv, and install its own torch at
   whatever CUDA version detectron2 tolerates. Costs ~3 GB of image and
   nothing else. The other three venvs keep sharing.

Record which rung worked, here, when you find out.

### 3. The full build

Only after 1 and 2 pass. Expect the long poles to be brush's `cargo build
--release` and the sam3dbody layer.

```bash
docker build -f docker/Dockerfile -t b2c/pipeline:latest .
docker run --rm --gpus all b2c/pipeline:latest --help
```

Then a real smoke test — the roundtrip workflow needs no GPU and no
weights, so it isolates plumbing from models:

```bash
docker run --rm --gpus all -v /path/to/cyber_6f:/data/cyber_6f b2c/pipeline:latest \
    run pipeline/workflows/roundtrip_example.yaml --dataset /data/cyber_6f/initial -v
```

## Why the CUDA version stopped being a constraint

Worth recording, because it looked like a hard pin and wasn't.

**No requirements file in `pipeline/envs/` pins torch.** They all say
`torch>=2.4.0` or bare `torch`. The `cu128` that appears in the old
Dockerfile and in `pod_bootstrap.sh` came from the *install instructions*,
and its comment explains the reason: the RunPod pod had CUDA 12.8
pre-installed, an unpinned `pip install torch` pulled a cu130 build, and
detectron2's native build failed on the mismatch. That is a constraint of
a machine we did not control.

In an image we choose the base, so the invariant "torch's CUDA == the
image's toolkit" is satisfiable at any version. sam-3d-body's own
INSTALL.md confirms there is no upstream pin — it says "install PyTorch
following the official instructions", and detectron2's `setup.py` asserts
only `torch >= 1.8` with no upper bound.

(INSTALL.md suggests Python **3.11**; the image uses 3.12, which is
Ubuntu 24.04's default. If something in the sam3dbody stack turns out to
be 3.11-specific, that is a base-image swap to `ubuntu22.04` — which has
python3.11 but *not* python3.12, the bug that broke the previous
Dockerfile.)

## Things that changed in the rewrite, and why

- **`python3.12` on Ubuntu 22.04 does not exist.** Confirmed against
  Launchpad: jammy publishes python3.10 and python3.11, zero python3.12
  sources. The old file's first `apt-get install` could never have
  succeeded. Now on Ubuntu 24.04, where python3 *is* 3.12.
- **`sageattention>=2.1.1` is unsatisfiable.** PyPI's newest is 1.0.6;
  2.x was never published there (it is a GitHub source build). It is also
  optional — the step wraps `set_attention_backend` in try/except and
  honours `attention_backend: "none"` — so it is dropped, with the git
  install line recorded in a comment. This would have failed the wan22
  layer of any clean build.
- **`FORCE_CUDA=1`** — see the detectron2 probe above.
- **EGL/GL loaders added.** The old file installed the Vulkan loader
  (correct) but no GL loader, so `render` could never have worked: it
  tries `libEGL.so.1`, falls back to OSMesa, and neither was present.
- **Rust toolchain no longer shipped.** brush builds in a `rust:1.88-slim`
  stage and only the binary is copied. Built on bookworm/glibc 2.36
  against a noble/glibc 2.39 runtime — older-to-newer is the safe
  direction.
- **Runtime base, not devel.** Three stages now: brush (Rust), a CUDA
  `devel` python-builder that exists only because detectron2 compiles CUDA
  kernels, and a CUDA `runtime` final stage that receives the finished
  venvs. No nvcc, no compilers, no Rust in the shipped image. This became
  possible only once gsplat was dropped — see below.
- **Venv sharing uses a `.pth`, not `--system-site-packages`.** The latter
  does not do what it looks like it does: `venv` resolves `home` to the
  real base interpreter, not to the venv you invoked it from, so a child
  created that way inherits `/usr/lib/python3`'s packages and sees nothing
  of `venv_base`. Verified directly. `docker/make-child-venv.sh` writes a
  `.pth` naming venv_base's site-packages instead, which gives all four
  properties needed: the child imports base packages, pip reports them
  already satisfied (so no child installs its own torch), a locally
  installed package still shadows the base copy, and installs in a child
  leave the base untouched. All four tested.
- **`libopencv-dev` dropped** — ~300 MB of C++ headers that nothing used;
  the opencv Python wheels bundle their own libraries.
- **`envs.yaml` paths fixed.** The old file created `/workspace/venv_*`
  while `envs.yaml` named `pipeline/envs/*/venv/bin/python`, so all three
  subprocess steps would have failed to find an interpreter.
  `docker/envs.docker.yaml` now describes the container and is copied over
  the default path during build.

## Venv sharing

Children are plain venvs carrying a **`.pth` file** that names venv_base's
site-packages — `docker/make-child-venv.sh` does it:

```
python3 -m venv /opt/venv_wan22
echo /opt/venv_base/lib/python3.12/site-packages \
    > /opt/venv_wan22/lib/python3.12/site-packages/zz_shared_base.pth
```

`--system-site-packages` is **not** the mechanism, whichever interpreter
invokes it: `venv` resolves the new venv's `home` to the real base
interpreter, so the child inherits `/usr/lib/python3`'s packages and sees
nothing of venv_base. That was tested directly; see the bullet under
"Things that changed in the rewrite".

The `.pth` gives what's needed: pip in a child sees base packages as
satisfied and skips them; if a child ever needs a different version it
installs locally, and its own site-packages is searched first, so it
shadows the base and the defensive isolation survives.

Keep venv_base minimal for that reason. Heavy and genuinely shared (torch,
torchvision, the `nvidia-*` CUDA wheels, transformers, numpy, opencv) goes
in base; anything version-contentious (diffusers, peft, torchao,
pytorch-lightning, detectron2) stays in a child.

Rough saving: torch plus its CUDA dependencies is ~3–3.5 GB installed.
Four copies would be ~13 GB; one is ~3.5 GB.

**Known small wart:** `pipeline/envs/rmbg/requirements.txt` asks for
`opencv-python-headless` while everything else (and `pyproject.toml`) uses
`opencv-python`. pip treats them as different distributions, so if that
file is ever fed to the image build verbatim, both get installed. The
Dockerfile currently sidesteps it by installing rmbg's two real extras
(`kornia`, `timm`) directly rather than via the requirements file. Worth
making the file consistent at some point.

## gsplat: dropped

`render_splat` was the only consumer, via body2colmap's `SplatRenderer`.
gsplat publishes no prebuilt wheel past torch 2.4 / cu124 (checked their
index: `pt20cu118` through `pt24cu124`, nothing since), so on this stack it
installs as an sdist and JIT-compiles its CUDA kernels on first use —
needing nvcc at *runtime*. That single dependency was what forced the whole
image onto a CUDA `devel` base.

It is now removed, and the image ships on `cuda:13.0.2-cudnn-runtime`.
detectron2 still needs nvcc, so it compiles in a `devel` builder stage and
only the finished venvs are copied forward.

**`render_splat` will therefore fail at import until its replacement
exists.** Everything else is unaffected — the step's camera-path half is
still verified, only the rasterisation call is missing a backend. The
replacement is a standalone Rust utility linking `brush-render`; see
`docs/brush-render-utility.md` (uncommitted design note).

One sequencing point from that note, worth repeating here because it is
easy to miss: **capture gsplat reference renders on the 4070 Ti before
moving on.** gsplat still installs fine in a scratch venv on a box with a
CUDA toolchain, and those images are the only oracle the brush renderer can
be validated against. Once that box is gone, so is the ability to produce
them.

## Still unverified after all of the above

Even with a green build, these remain untested:

- `brush` producing an actual splat (never run, on any machine)
- `render`'s pyrender/EGL rasterisation (never run)
- `render_splat`'s gsplat call (never run; the camera-path half around it
  is verified locally against cyber_6f)
- `generate_firstlast`'s warp against a real `render` output
- `fast_helical_full.yaml` end to end — every stage now has a native step
  and the file validates statically, but the sequence has never executed

The first three are exactly what a local GPU box is for.
