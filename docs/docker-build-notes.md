# Docker build notes — read before the first build

Written for a session that will run on a box with an **RTX 4070 Ti**, which
is the first machine this project has had with a local GPU. `docker/Dockerfile`
was rewritten against these notes and, as of **2026-08-23, now builds** —
see the RESULT blocks in each section below for what actually happened,
including the four fixes the first build needed, a later same-day fix for
the container Vulkan/EGL graphics problem (moved host Docker from snap to
`docker-ce`; the actual cause turned out to be a missing `libegl1` package,
not snap itself — see section 1's second RESULT block), and `brush`/`render`
both subsequently verified on real GPU hardware in the shipped image.

The original text is left in place as written, because the reasoning behind
each choice is still what you need when a layer breaks; the RESULT blocks
say which of those guesses survived contact.

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

#### RESULT (2026-08-23, RTX 4070 Ti box) — ran, but did NOT confirm the mechanism

Driver is **595.71.05 / CUDA 13.2**, comfortably past the >= 580 floor, so
that worry is closed.

Three things this box taught us, none of them the expected answer.

**`--gpus all` does not work here.** nvidia-container-toolkit is in **CDI**
mode, and the prestart hook refuses the flag outright:

```
invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the
docker --gpus flag) is not supported. Please use the NVIDIA Container
Runtime (e.g. specify the --runtime=nvidia flag) instead
```

Every command in this doc therefore needs `--runtime=nvidia -e
NVIDIA_VISIBLE_DEVICES=all` in place of `--gpus all`. The commands above
are left as written because that is the right form on RunPod; substitute
locally.

**`NVIDIA_DRIVER_CAPABILITIES` made no difference at all.** With the
variable and without it, the result was byte-identical:
`/etc/vulkan/icd.d/nvidia_icd.json` present either way, and
`libGLX_nvidia`/`libEGL_nvidia` on the ldconfig path either way. In CDI
mode the device spec is applied wholesale at creation time and the env var
is simply not consulted — it is a *legacy-hook-mode* control. So the A/B
test this section was designed around **cannot be run on this box**, and
the RunPod question it was meant to settle is still open. The `ENV` line in
the Dockerfile is harmless and still correct for a legacy-mode host; keep
it, and still set it in the pod template.

**Docker here is the snap package**, which confines what the daemon and CLI
can see. Two consequences that cost real time:

- Host driver libraries are injected at
  `/var/lib/snapd/hostfs/usr/lib/x86_64-linux-gnu/`, not
  `/usr/lib/x86_64-linux-gnu/`. They *are* on the ldconfig path, so this is
  cosmetic for loading — but it makes `ls /usr/lib/.../libnvidia*` look
  alarmingly empty.
- **Paths outside `$HOME`, and hidden directories inside it, are invisible.**
  `-v /tmp/...:/probe.sh` silently created `/probe.sh` as an empty
  *directory* rather than failing, and a build context under
  `~/.cache/` produced `transferring dockerfile: 2B` and "no such file or
  directory". Keep probe files in a non-hidden directory under `$HOME`
  (these used `~/b2c-probe`), or inline the script into `bash -c`.

**Vulkan does not reach the GPU on this host — brush would run on the CPU.**
This is the one finding that matters beyond local ergonomics. With the
runtime stage's full package set installed, the chain checks out right up to
the last step:

- `/etc/vulkan/icd.d/nvidia_icd.json` is injected (library_path
  `libGLX_nvidia.so.0`, api_version 1.4.329)
- `libGLX_nvidia.so.0` **dlopens successfully** and `ldd` reports no
  missing dependencies
- it **does export** `vk_icdGetInstanceProcAddr` (confirmed with `nm -D`
  on both the host copy and the container's resolved copy)

and yet the loader reports:

```
loader_scanned_icd_add: Could not get 'vkCreateInstance' via
'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
```

i.e. the ICD loads and is asked for `vkCreateInstance` and returns NULL.
The only device `vulkaninfo` then enumerates is **llvmpipe (Mesa 25.2.8,
`PHYSICAL_DEVICE_TYPE_CPU`)**. The GPU is otherwise fully present in the
container — `nvidia-smi` works, `/proc/driver/nvidia/gpus/0000:01:00.0`
and `/dev/dri/{card1,renderD128}` are all there — so this is not a missing
device node.

EGL is in the same state for the same reason: **no `10_nvidia.json` in
either `/usr/share/glvnd/egl_vendor.d/` or `/etc/glvnd/egl_vendor.d/`**,
only `50_mesa.json`. `eglQueryDevicesEXT` returns 2 devices, both Mesa
software. So `render`'s EGL path would get software rasterisation rather
than the GPU.

One trap worth recording separately, because it wasted a probe: the bare
CUDA image has no `libXext.so.6`, and `libGLX_nvidia.so.0` links against
it, so a probe that skips it fails with a *misleading* dlopen error. The
Dockerfile's runtime stage already installs `libxext6`, so the image is
fine; only ad-hoc probes need it added.

Leading hypothesis for the Vulkan/EGL failure is the snap confinement —
the CDI spec injects the libraries and the Vulkan ICD json but not the
glvnd EGL vendor json, and the driver declines to create an instance from
its hostfs-relocated libraries. **This is not proven.** The clean test is a
non-snap `docker-ce` install, which would also settle whether legacy-hook
mode (and hence a real `NVIDIA_DRIVER_CAPABILITIES` A/B) is available here.
Until then: **brush and `render` cannot be GPU-validated on this box**,
which unfortunately is exactly what the "still unverified" list at the
bottom was hoping this machine would close. The *build* is unaffected —
this is purely a container-runtime graphics issue.

#### RESULT (2026-08-23, same day): fixed. The snap hypothesis was wrong; the real cause was one missing package.

Switched the host from snap Docker to `docker-ce` (`sudo snap remove docker`,
then the standard `docker-ce`/`docker-ce-cli`/`containerd.io` apt install),
installed `nvidia-container-toolkit` from NVIDIA's apt repo (it turned out
not to be pulled in automatically by the docker-ce install — a separate
package), and ran `nvidia-ctk runtime configure --runtime=docker` +
`systemctl restart docker` to register the `nvidia` runtime.

**That alone did not fix it.** Vulkan still enumerated only `llvmpipe`
afterward, with the exact same `loader_scanned_icd_add: Could not get
'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'` error as under snap.
So before chasing the real cause, every remaining container-tooling
variable was tested and eliminated:

| Variable | Result |
|---|---|
| snap Docker → docker-ce | No change |
| `NVIDIA_DRIVER_CAPABILITIES` under legacy-hook mode (now actually available) | No change — made no difference, same as it made no difference under CDI |
| Legacy `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` vs modern CDI (`nvidia-ctk cdi generate` + `--device nvidia.com/gpu=all`) | Identical failure both ways |
| `--privileged` (rules out cgroup/seccomp device restriction) | No change |
| Library resolution (`ldd`, `nm -D` on `libGLX_nvidia.so.0`) | Clean in every case — no missing symbols, no missing deps |
| Vulkan on the **bare host**, no container at all (compiled a minimal `vkCreateInstance`/`vkEnumeratePhysicalDevices` C probe against `libvulkan.so.1`) | **Works.** Correctly lists the RTX 4070 Ti as `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` |

That last row is what reframed the problem: the driver and host Vulkan
stack were never broken. Every container-runtime knob (snap vs docker-ce,
legacy hook vs CDI, privileged vs not, capabilities env var) produced the
identical failure, which means none of them was the cause.

**The actual cause: `libegl1` was missing from the probe containers.**
NVIDIA ships one shared driver object, `libGLX_nvidia.so.0`, that backs
GLX, EGL, *and* the Vulkan ICD. During Vulkan instance creation it runs an
internal GLVND self-registration step that requires `libEGL.so.1` (the
glvnd EGL dispatch loader) to be resolvable — even though Vulkan itself has
no EGL dependency and `ash`/wgpu never call into EGL. Without that library
present, the check fails closed: `vk_icdGetInstanceProcAddr` silently
returns NULL for `vkCreateInstance` instead of erroring, and the loader
falls back to `llvmpipe`. This is exactly the kind of failure that reads
like a container/runtime problem (silent, no useful error message) but
isn't one.

Confirmed directly: `apt-get install libvulkan1 vulkan-tools libxext6` (no
`libegl1`) inside the container → `llvmpipe` only. Same container, same
run, adding `libegl1` → `vulkaninfo` reports:

```
GPU0:
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = NVIDIA GeForce RTX 4070 Ti
	driverID           = DRIVER_ID_NVIDIA_PROPRIETARY
```

Re-verified with the **exact package list from the Dockerfile's runtime
stage** (`nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` + the full
`apt-get install` line, not a minimal probe) — same result. **No Dockerfile
change was needed**: `libegl1` was already being installed there, for the
unrelated reason of giving `render`'s pyrender/EGL path a real loader. It
was load-bearing for brush's Vulkan path too, just never verified until
now.

Net effect: `brush` and `render` are unblocked on this box. Use either
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` or `--device
nvidia.com/gpu=all` (after `nvidia-ctk cdi generate`) — both now work
identically. The `NVIDIA_DRIVER_CAPABILITIES` question for RunPod is still
open (it made no measurable difference in either mode on this box), but
that was always a secondary question next to this one.

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
    && /opt/v/bin/pip install torch==2.13.0 torchvision==0.28.0 \
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
   The image now sits at the *top* of that range (2.13.0), so this rung has
   real room below it — 2.9.1 was verified working on 2026-08-23 and is the
   conservative pick. Below 2.9.0 there is nothing on cu130, which is the
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

### SageAttention: was missing entirely, now built from source (2026-08-24)

The image shipped with **no sageattention at all**, so every sage backend
raised, `wan22_vace_denoise`'s try/except swallowed it, and the most
expensive step in the pipeline quietly ran on plain PyTorch SDPA. Noticed
from a `SageAttention ❌` line in seedvr2's optimizations banner.

It cannot come from PyPI: the requirement is `>=2.1.1` (diffusers enforces
that for *any* sage backend variant) and PyPI stops at 1.0.6, because 2.x
was only ever tagged on GitHub. The fix is a source build, which turns out
to be cheap: **3m44s** with `MAX_JOBS=8`, **71 MB** installed, against torch
2.13.0 + CUDA 13.0 with no patches.

Measured on the RTX 4070 Ti (SM89), 40 heads x 128 dim, vs PyTorch SDPA:

| seq | SDPA | triton int8+fp16 | CUDA int8+fp8 |
|---|---|---|---|
| 1024 | 0.37 ms | 0.26 ms (cos .99992) | 0.29 ms (cos .99930) |
| 4096 | 4.80 ms | 2.98 ms (cos .99991) | 2.51 ms (cos .99928) |
| 9216 | 22.32 ms | **12.75 ms** (cos .99991) | **9.40 ms** (cos .99926) |

~1.75x on the Triton kernel the step selects for Ada, ~2.4x on the fp8 CUDA
one. Both track SDPA closely, and notably the fp8 CUDA kernel shows **no
sign of thu-ml/SageAttention#360** on this card at 2.2.0 — but the steering
still avoids it, because random tensors are not a 40-block diffusion run and
that bug would present as corrupted frames, not a bad cosine.

**Two real bugs found while doing this:**

1. **`sage_hub` is broken**, and it was what `_select_sage_backend()`
   returned for every GPU except Ada. diffusers 0.40.0 pins version 1 of
   `kernels-community/sage-attention`, and the revision that resolves to
   has no `build` on the Hub — every call ends in
   `RemoteEntryNotFoundError: 404`. **This is not a stale pin on our side:
   0.40.0 is the newest diffusers on PyPI** (2026-08-20), so there is
   nothing to upgrade to. `_select_sage_backend()` now returns the local
   `sage` instead.
2. **SageAttention 2.2.0 does not support sm_100** (datacenter Blackwell,
   B100/B200/GB200), which *is* in this image's default
   `TORCH_CUDA_ARCH_LIST`. Its `SUPPORTED_ARCHS` set is declared and then
   never used, so an unsupported arch emits no gencode and raises nothing —
   you get a package with no kernel for that card. The Dockerfile therefore
   passes its own `SAGE_CUDA_ARCH_LIST="8.9;12.0"` rather than inheriting
   the stage-wide list. On an sm_100 pod the sage backends have no kernel
   and the step falls back to SDPA.

#### RESULT (2026-08-24): **the torch floor moved to 2.13.0, and detectron2 did not care.**

The premise that detectron2 is what pins this image to torch 2.9.1 was
tested directly, and it is **false**. detectron2 @`a1ce2f9` compiles against
torch 2.13.0+cu130 with no source changes, no newer commit, and no separate
venv — rungs 3 and 4 of the ladder above were not needed and the fp8 work's
"it is not a free change" caveat turned out to be cheap.

Probed in a `--gpus all` container on the RTX 4070 Ti (sm_89), CUDA 13.0.2
devel base, `FORCE_CUDA=1`, `TORCH_CUDA_ARCH_LIST=8.9`:

```
torch 2.13.0+cu130  torchvision 0.28.0+cu130  cuda 13.0  available: True
detectron2 0.6
_C OK   has_cuda: True | built against CUDA 13.0 | compiler GCC 13.3
DeformConv  -> (2, 32, 32, 32) finite: True
ModulatedDC -> (2, 32, 32, 32) finite: True
deform_conv backward OK, grad finite: True
```

Forward **and backward** through the nvcc-compiled kernels, on the GPU —
not just `import _C`, for the CPU-only-build reason this section keeps
warning about. (Note this commit's `_C` exports deformable-conv and
COCOeval symbols only; rotated NMS, which the 2026-08-23 result above
exercised, now comes from torchvision. The deform-conv kernels are the
CUDA extension here.)

The full sam-3d-body stack was then installed on top — the whole of
`pipeline/envs/sam3dbody/requirements.txt`, `--no-build-isolation`, cython
first, exactly as the Dockerfile does it — and `from sam_3d_body import
SAM3DBodyEstimator, load_sam_3d_body` succeeds. `xtcocotools`, `pyrender`,
`pytorch-lightning` 2.6.5 and `networkx==3.2.1` all build/resolve fine
against torch 2.13 and numpy 2.5.2. The only resolver complaint is the
pre-existing `detectron2 0.6 requires iopath<0.1.10, but you have 0.1.10`,
which `setup.sh` already documents as harmless.

The other three venvs were verified on the same base, sharing torch through
the usual `zz_shared_base.pth`:

| venv | checked | result |
|---|---|---|
| `venv_main` | kornia 0.8.3, timm 1.0.28, plyfile, mediapipe 1.0.1 (+ `mediapipe.tasks`), gradio 6.25, cv2 5.0.0 | ✅ |
| `venv_wan22` | diffusers 0.40.0, torchao **0.18.0**, peft 0.20.0, accelerate 1.14.0 | ✅ |
| `venv_seedvr2` | omegaconf, gguf, rotary_embedding_torch, and the vendored `inference_cli` / `src.utils.*` | ✅ |

seedvr2's vendored code even recognises the new torch and applies its own
workaround: `Conv3d workaround active: PyTorch 2.13.0, cuDNN 92000`.

The full image was then rebuilt on it and re-verified on the GPU. In the
**shipped runtime image** (not the builder): all five venvs report
`2.13.0+cu130` with `cuda_is_available` True; `_C.has_cuda()` is True and
DeformConv runs fwd+bwd on the GPU; `sam_3d_body` imports; `torchao 0.18.0`
and `diffusers 0.40.0` import together and `wan22_vace_denoise` imports;
`torch.cuda.get_arch_list()` still covers `sm_75/80/86/90/100/120`, so Ada
and both Blackwell targets are intact. `pipeline.cli doctor` reports **11
ok, 0 failures** (the one warning is an unset HF_TOKEN, expected locally),
including Vulkan and EGL. 122 in-image tests pass.

Image size did not move: **5.66 GB content vs 5.67 GB** on torch 2.9.1.

**Why the bump was worth taking:** it resolves the fp8 "version triangle"
in `docs/fp8-quant-notes.md`. torchao 0.18 (which needs torch ≥ 2.11) is
what lets diffusers apply a LoRA to quantized weights and serialize them to
safetensors. Both were verified on GPU, not assumed. The torchao pin moved
to `0.18.0` in lockstep — **never move one without the other**, or `import
diffusers` breaks outright.

#### RESULT (2026-08-23): **rung 0 — no fallback needed.** Verified on the GPU.

The stack in the Dockerfile as written works. Nothing on the ladder was
used: CUDA 13.0.2 + torch 2.9.1+cu130 + detectron2 @`a1ce2f9` compiles and
runs.

```
torch: 2.9.1+cu130   torch.version.cuda: 13.0
cuda available: True | NVIDIA GeForce RTX 4070 Ti
_C.has_cuda(): True
_C.get_cuda_version(): CUDA 13.0
nms_rotated on CUDA -> [0]     (kernel actually executed on the GPU)
```

That last line is the one that matters. `from detectron2 import _C`
succeeding only proves the extension *loads*; the CPU-only trap this
section warns about would sail past it. So the probe was extended to (a)
assert `_C` exports CUDA symbols at build time, and (b) run a real rotated
NMS on a CUDA tensor in a `--runtime=nvidia` container. `FORCE_CUDA=1`
demonstrably took.

**The probe recipe printed above is wrong as written — fix it before reuse.**
It installs detectron2 with `--no-deps` and then immediately does
`from detectron2 import _C`, but that import runs `detectron2/__init__.py`,
which needs the runtime deps `--no-deps` just skipped. It fails with:

```
ModuleNotFoundError: No module named 'fvcore'
```

which reads like a detectron2 build failure and is nothing of the sort —
the CUDA wheel had already compiled fine. Add before the check:

```dockerfile
RUN /opt/v/bin/pip install fvcore omegaconf yacs termcolor tabulate         cloudpickle Pillow matplotlib tqdm
```

**The real image is not affected by that trap**, because
`pipeline/envs/sam3dbody/requirements.txt` already lists `fvcore` (and
`pycocotools`, `tensorboard`, `yacs`) and is installed *before* detectron2.
The ordering in the Dockerfile is correct as-is.

One cosmetic warning to expect and ignore in the full build — pip prints it
because `--no-deps` defers the dependency check to a later resolution:

```
detectron2 0.6 requires iopath<0.1.10,>=0.1.7, but you have iopath 0.1.10
```

Worth a glance if detectron2 ever misbehaves at runtime, but it did not
affect the CUDA extension.

Practical note: the probe build takes ~12 min from cold, and the detectron2
wheel compile alone is ~5 of that with `TORCH_CUDA_ARCH_LIST=8.9`. Widening
the arch list multiplies that.

#### The brush stage: three things were wrong (2026-08-23)

Not mentioned in the original plan because brush was expected to either
compile or fail on a missing system library. It did neither — it compiled
the *wrong source* into a binary that would have failed at runtime.

**1. `rust:1.88` is too old.** Cargo refused to resolve at all:

```
rerun@0.36.0 requires rustc 1.95
sysinfo@0.39.6 requires rustc 1.95
safe_arch@1.2.0 / wide@1.6.1 require rustc 1.89
```

Now `rust:1.98-slim-bookworm`. Compile takes ~6m20s.

**2. The clone was building upstream, not the fork.** `git clone --depth 1
https://github.com/Erant/brush.git` takes the *default* branch, and
`Erant/brush`'s `main` merely tracks upstream ArthurBrussee/brush — the
HEAD it built was `362dc39`, an upstream PR ("Compile kernels to MSL on
macOS #519"). The fork's actual work is on the **`normal-map-supervision`**
branch (`c228531`, "Expose normal-map supervision in the viewer
settings").

The build *succeeded* and produced a perfectly good 231 MB binary. The
defect was only visible by diffing `--help` against the argv
`pipeline/steps/brush.py` actually constructs, which is now worth doing
whenever the pin moves:

```bash
docker run --rm --entrypoint /bin/sh b2c/brush-builder -c '/out-brush --help' > help.txt
grep -oE '"--[a-z-]+"' pipeline/steps/brush.py | sort -u | tr -d '"' | while read f; do
    grep -q -- "$f" help.txt && echo "OK      $f" || echo "MISSING $f"; done
```

The Dockerfile now clones `--branch ${BRUSH_BRANCH}`, defaulting to
`normal-map-supervision`.

**3. `--total-steps` no longer exists — it is `--total-train-iters`.** This
one is a bug in *our* code, not the Dockerfile, and it survives the branch
fix. `pipeline/steps/brush.py` was written against an older brush; the fork
branch is rebased onto an upstream that renamed the flag. The binary
rejects it outright:

```
error: unexpected argument '--total-steps' found
  tip: a similar argument exists: '--total-train-iters'
```

Fixed in `pipeline/steps/brush.py`. Note the *pipeline param* is still
`total_steps` — `fast_helical_full.yaml` and `fast_helical_native.yaml`
pass `total_steps:` and that is unchanged; only the CLI flag string moved.

With the branch fix, all twelve flags the step passes are present.

**Bonus confirmation of the Vulkan finding above.** Running the fixed argv
gets past clap and dies where section 1 predicted:

```
No possible adapter available for backend. Falling back to first available.:
NotFound { active_backends: Backends(0x0), requested_backends: Backends(VULKAN),
supported_backends: Backends(VULKAN | GL) }
```

So brush itself independently corroborates that no Vulkan adapter is
reachable in a container on this box. brush is otherwise built and
functional; it is the host graphics stack that is the blocker.

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

#### RESULT (2026-08-23): **the image builds, and the smoke test passes.**

`docker build -f docker/Dockerfile -t b2c/pipeline:latest .` exits 0 with no
warnings. **15.7 GB on disk / 5.57 GB compressed.** Roughly 35 min cold,
dominated by the torch/CUDA wheel downloads, brush's ~6m20s compile and
detectron2's ~2m.

Four fixes were needed to get there — three in the Dockerfile, one in
pipeline code. They are described in the sections above; in build order:

| Stage | Fix | Failure mode without it |
|---|---|---|
| brush | `rust:1.88` -> `1.98` | hard build failure, cargo won't resolve |
| brush | clone `--branch normal-map-supervision` | *silent*: builds upstream, wrong CLI |
| python-builder | add `libgl1 libglib2.0-0 libxcb1` | build failure *after* detectron2 compiles |
| `steps/brush.py` | `--total-steps` -> `--total-train-iters` | runtime "unexpected argument" |

Verified in the built image:

- `22 steps registered`
- all four child venvs report `shares torch 2.13.0+cu130` — **the `.pth`
  venv-sharing scheme works as designed**, one torch for four venvs
  (originally verified at 2.9.1+cu130; it survived the bump untouched)
- `detectron2 native extension OK`, and `_C.has_cuda()` is True in the
  *shipped runtime image*, not just the builder
- all three subprocess envs in `envs.docker.yaml` resolve: each of
  `/opt/venv_{wan22,sam3dbody,seedvr2}/bin/python` exists, imports
  `pipeline.worker`, and reports `torch.cuda.is_available() == True`
- `sam_3d_body` and the vendored `inference_cli` both import
- `brush` is on PATH at `/usr/local/bin/brush`

The roundtrip smoke test round-trips **exactly**: 86 files in, 86 out; all
83 PNGs and `prompt.txt` byte-identical; `metadata.json` semantically equal
(key order only) and `pointcloud.npz` array-for-array equal (positions
(10000,3) float64, colors (10000,3) uint8) — the container bytes differ
only because npz is a zip and embeds timestamps.

Two things settled in passing that the requirements files flagged as open:

- **`chump` is correct, not a typo for `chumpy`.** `chump-1.6.0` installed
  fine as part of the bulk `--no-build-isolation` install. The note in
  `pipeline/envs/sam3dbody/requirements.txt` can be closed.
- **The whole sam3dbody ordering dance works as documented.** cython first,
  bulk install with `--no-build-isolation`, detectron2 last with
  `--no-deps` — `xtcocotools` 1.14.3 and all the rest built without
  complaint on Python 3.12.

`--gpus all` in the command above must be `--runtime=nvidia -e
NVIDIA_VISIBLE_DEVICES=all` on this box; see section 1.

One correctness fix beyond the build: `ENV PYTHONPATH="/opt/vendor/seedvr2:${PYTHONPATH}"`
expanded to a trailing colon (BuildKit's `UndefinedVar` warning — the only
warning the build emitted), which Python reads as an extra working-directory
entry on `sys.path`. Now set literally. It was a duplicate CWD entry rather
than a stdlib-shadowing hazard; `sys.path[0]` is the script dir either way.

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
- **`sageattention>=2.1.1` is unsatisfiable from PyPI.** PyPI's newest is
  1.0.6 (Nov 2024); 2.x was never published there, only tagged on GitHub.
  This would have failed the wan22 layer of any clean build, so it was
  dropped from that layer — but it is **now built from source** in its own
  late layer (`SAGE_REF`), because dropping it meant every sage backend
  raised and the step silently ran on plain SDPA. See "SageAttention: was
  missing entirely" below.
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

## gsplat: dropped, replaced by brush-splat-render

`render_splat` was the only consumer, via body2colmap's `SplatRenderer`.
gsplat publishes no prebuilt wheel past torch 2.4 / cu124 (checked their
index: `pt20cu118` through `pt24cu124`, nothing since), so on this stack it
installs as an sdist and JIT-compiles its CUDA kernels on first use —
needing nvcc at *runtime*. That single dependency was what forced the whole
image onto a CUDA `devel` base.

It is now removed, and the image ships on `cuda:13.0.2-cudnn-runtime`.
detectron2 still needs nvcc, so it compiles in a `devel` builder stage and
only the finished venvs are copied forward.

**Replacement: `brush-splat-render`, a standalone Rust CLI in the
Erant/brush fork** (`crates/brush-splat-render`, built in the brush-builder
stage alongside `brush` itself — see that stage's comments). It loads a
trained `.ply` and rasterises an explicit camera list through the same
wgpu/Vulkan renderer `brush` uses, so it needs nothing this image doesn't
already provide for `brush`. `pipeline/steps/splat.py`'s `RenderSplatStep`
now shells out to it directly (same pattern as `steps/brush.py` shelling
out to `brush`), bypassing body2colmap's `SplatRenderer`/gsplat entirely.
Full design rationale, the `cameras.json` schema, and the OpenGL/OpenCV
camera-convention derivation (verified against a real gsplat oracle: mean
abs error 0.0008-0.0015 on RGB) live in
`~/Projects/brush/docs/splat-render.md`.

**Built, shipped, and verified on GPU** (2026-08-23) — see "Still
unverified" below for the `render_splat` RESULT block: both the raw
`brush-splat-render` binary and the `RenderSplatStep` Python integration
ran a real render against real `cyber_6f` cameras in the shipped image.

The gsplat-reference-render errand from `docs/brush-render-utility.md` is
now historical: it was **done** as part of building `brush-splat-render` —
gsplat is CUDA, not Vulkan, so a scratch venv on this box produced the
oracle images the new renderer was validated against, per
`~/Projects/brush/docs/splat-render.md`.

## Still unverified after all of the above

#### UPDATE (2026-08-23, later the same day): the first two are now closed

Once the docker-ce + `libegl1` fix above landed, both graphics-blocked items
were re-tested directly against the real shipped image (`b2c/pipeline:latest`,
`--device nvidia.com/gpu=all`) and both pass on actual GPU hardware:

- **`brush` producing an actual splat.** A real (short) training run against
  `cyber_6f/colmap`'s 81-frame COLMAP export — 200 iterations, sh-degree 1,
  480px max resolution — completed in ~3s and exported a valid `test.ply`
  (12,463 splats after one refine pass). Full log confirms the COLMAP
  dataset loaded (81 training views, 10000 seed points) and training ran to
  completion with no errors.
- **`render`'s pyrender/EGL rasterisation.** Created a real
  `pyrender.OffscreenRenderer` inside the container and read back
  `GL_VENDOR`/`GL_RENDERER` directly from the GL context:
  `GL_VENDOR: NVIDIA Corporation`, `GL_RENDERER: NVIDIA GeForce RTX 4070
  Ti/PCIe/SSE2` — hardware rendering, not the Mesa/llvmpipe fallback.

#### UPDATE (2026-08-23, later still): `render_splat` closed out too

The Erant/brush commit adding `brush-splat-render` was pushed to
`erant/normal-map-supervision`, the image was rebuilt (`docker build
--no-cache-filter brush-builder ...` — a plain rebuild reuses the cached
`git clone` layer and silently keeps the pre-push checkout, since BuildKit
caches on the RUN command text, not remote git state; worth remembering
next time this stage's source moves upstream without the Dockerfile itself
changing), and both the raw binary and the actual pipeline step were
verified on GPU:

- `brush-splat-render` directly, against the `.ply` from the `brush` run
  above and 4 real cameras from `cyber_6f`: `wgpu_hal::vulkan::adapter`
  log confirms `backend: Vulkan` on `NVIDIA GeForce RTX 4070 Ti`; output
  PNGs have real content (RGB std ~32, ~23% alpha coverage — a
  person-shaped foreground in a portrait frame, not blank).
- `pipeline.steps.splat.RenderSplatStep` end to end, reusing all 81 of
  `cyber_6f/initial`'s real cameras: 81 images + 81 masks back, correct
  `(1280, 720, 3)` BGR uint8 images and `[0, 1]` float32 masks.

`render_splat` is no longer blocked by anything graphics- or
integration-related. What's left is quality, not plumbing: the smoke test
above trained for only 200 iterations at 480px max resolution, so the
render looks like a 200-iteration splat, not a finished one.

#### UPDATE (2026-08-23, later still): first full sequence run, and two real bugs it found

`fast_helical_local_smoke.yaml` — `fast_helical_full.yaml` minus
`wan22_vace_denoise` (both passes, too much VRAM for this 12GB card at
production settings) and `seedvr2`'s upscale — ran **end to end for the
first time**, against `cyber_6f/initial`, on GPU, at full production
settings otherwise (30000 brush iterations, sh-degree 3, 1920 max
resolution): `rmbg` -> `sapiens2_lite` -> `brush` (107,127 Gaussians) ->
`render_splat` (`brush-splat-render`, helical path, 81 cameras) ->
`inject_anchor` -> `mask_splat` -> `rmbg` -> `colmap_export`. Output: an 87-
file dataset and a valid 81-camera/81-image COLMAP directory (10,000 3D
points, correct PINHOLE intrinsics).

Getting there found two real, previously-undiscovered bugs — exactly what
"never executed as a sequence" (the caveat on every workflow YAML in this
repo) predicts, and neither was reachable by unit tests or static
validation:

1. **`Context.set` crashed on the first write into any not-yet-seeded
   namespace.** `cli.py` seeds the initial context as `{"dataset": dataset}`
   only; the first step to write to e.g. `scene.vertices` hit
   `self._data["scene"]` with a bare `KeyError`, since nothing had created
   `scene` yet. `test_workflows.py` never caught it because it only
   validates workflow structure statically — it has never actually run a
   `WorkflowRunner`. Fixed in `pipeline/context.py`: `Context.set` now
   auto-vivifies a missing top-level (or intermediate) namespace as an
   empty dict instead of assuming it already exists. This is a fix every
   workflow using a `scene.*`-style scratch namespace needed, not something
   specific to this run.
2. **`InProcessDispatcher` never freed GPU memory between one-shot steps.**
   `keep_loaded=False` (the default) makes a fresh `Step` instance per
   call and never tracks it, so `Dispatcher.close()`'s `unload()` loop —
   the only place already calling `torch.cuda.empty_cache()` — never runs
   for it. On this 12GB card, `rmbg` -> `sapiens2_lite` OOM'd because
   nothing released rmbg's model memory before sapiens2_lite tried to
   load its own. Fixed in `pipeline/dispatch/in_process.py`: releases the
   CUDA cache after every non-`keep_loaded` step. Only got the OOM down
   from ~3.24GB free to ~3.77GB free, though — not the whole story (see
   next).
3. **Not a bug, a real capacity limit:** even after (2), `sapiens2_lite`'s
   own default `batch_size=8` needed 4.53GB for one attention pass at
   720x1280, more than fit in the remaining headroom. `batch_size: 2` (the
   step already supported chunking, just wasn't being told to) fixed it
   with no code change. Left as an explicit override in
   `fast_helical_local_smoke.yaml`'s `normal_maps` step rather than
   changing the step's own default, since 8 is presumably fine on a
   bigger card.

What this run does **not** validate: `wan22_vace_denoise` and `seedvr2`
were skipped entirely (see above), so denoise quality and the upscale
stage remain unverified — the frames going into `brush` are `cyber_6f`'s
own already-denoised frames standing in for "assume denoise produced good
output," not a real denoise pass. What it does establish: the entire
*rest* of the pipeline — every step this project could not previously run
in combination — now runs cleanly together on real GPU hardware.

The remaining items below still need a run this session didn't attempt:

- `generate_firstlast`'s warp against a real `render` output (front-half
  wiring — see `pipeline/workflows/fast_helical_native.yaml`'s
  `render_initial_views`/`warp_reference_to_anchor` steps, added this
  session but not yet run: there's no CLI/Dataset bootstrap path for
  starting from a bare reference photo yet, only for an existing on-disk
  dataset — see that file's header comment).
- `fast_helical_full.yaml` end to end, with `wan22_vace_denoise` and
  `seedvr2` included — needs a bigger card (an L40S-class RunPod pod) to
  test at settings that mean anything quality-wise; on this box the two
  denoise-touching stages are the ones deliberately skipped above.

Unblocking the graphics items was the `docker-ce` + `libegl1` fix described
above, not a driver downgrade or a RunPod-specific workaround — worth
retesting the `NVIDIA_DRIVER_CAPABILITIES` question on RunPod with that in
mind, since it's now known to matter less than expected in every mode
tested here.

**Widened `TORCH_CUDA_ARCH_LIST` to `8.9;10.0;12.0`** (was `8.9` only), in
preparation for the next verification round moving to a RunPod Ada or
Blackwell pod (consumer or server variant of either). This only affects
detectron2's own nvcc-compiled kernels — torch's own cu130 wheel already
ships SASS for sm_75/80/86/90/100/120 plus compute_120 PTX regardless of
this ARG (checked via `torch.cuda.get_arch_list()` in the built image), so
nothing else needed widening. Real cost, not a free change: the
sam3dbody/detectron2 layer went from ~120s (single arch) to ~169s (three
arches) to compile. See `docker/Dockerfile`'s comment on this ARG for the
per-value breakdown (8.9 = Ada consumer+server, 10.0 = Blackwell server,
12.0 = Blackwell consumer).

Also worth recording since it came up chasing an unrelated OOM: this box's
own GPU (RTX 4070 Ti, cap 8.9) is **not** in torch's `get_arch_list()`
output at all, and no PTX below `compute_120` is embedded either — by the
letter of that list, this exact wheel should have no way to run a kernel
on this exact card. It does anyway (confirmed directly: a real
`torch.matmul` on `cuda:0` succeeded, and the whole smoke-test run above
did real `rmbg`/`sapiens2_lite` GPU inference throughout) — so whatever
compatibility path covers 8.9 here isn't reflected in that API's output.
Not chased further since it's empirically a non-issue, but worth knowing
`get_arch_list()` isn't a reliable yes/no for "will this run" if it ever
comes up again.

Note the gsplat-reference-render errand from `docs/brush-render-utility.md`
is separate from all of this: gsplat is CUDA, not Vulkan, so a scratch venv
on this box can still produce the oracle images the brush renderer was
validated against (see `~/Projects/brush/docs/splat-render.md`, which
records that comparison — mean abs error 0.0008-0.0015 on RGB).
