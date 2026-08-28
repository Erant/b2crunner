# Running the image on RunPod

Everything in this file is a *pod template* concern — settings that live
outside the image and cannot be baked into it. The image itself is meant to
be built once per verification round and then left alone; if you find
yourself wanting to rebuild it to debug something, check the
[debugging section](#debugging-without-rebuilding) first, because the tools
are probably already in there.

Companion documents: [docker.md](docker.md) for *why* the image is shaped
the way it is, and [docker-build-notes.md](docker-build-notes.md) for what
has and hasn't been verified on real hardware.

## Pod template settings

| Setting | Value | Why |
|---|---|---|
| Container image | your registry's `b2c/pipeline:latest` | |
| Container start command | **leave empty** | The image's `CMD` is `ui`, which serves the web UI and stays alive. Anything you put here replaces it. |
| Volume mount path | **`/data`** | Not the `/workspace` default. See below. |
| Volume size | 100 GB+ | HF checkpoints alone are ~60 GB, before any run output. |
| Container disk | 20 GB | Only the image's own writable layer; nothing the pipeline writes should land here. |
| **RAM** | **64 GB+** | Not optional for `fast_helical_full`. Its two denoise passes use `keep_loaded: true`, which holds ~47 GB of Wan weights resident in host RAM between them so they come off the volume once instead of twice. Too little and the worker is OOM-killed mid-run, which looks like a mysterious dead worker rather than a sizing mistake. `doctor` warns. Drop `keep_loaded` from the workflow if you must run smaller. |
| HTTP ports | `7860` | The web UI. RunPod proxies it at `https://<pod-id>-7860.proxy.runpod.net`. |
| TCP ports | `22` | SSH. Optional, but it is how you get a shell if the UI won't start. |

### Environment variables

| Variable | Value | Notes |
|---|---|---|
| `HF_TOKEN` | `hf_...` | **Required.** From an account that has accepted the licences for `briaai/RMBG-2.0` and `facebook/sam-3d-body-dinov3`; both are gated and a human has to click through each one. `doctor` reports whether the token can actually reach them. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,graphics,display` | **Set this here even though the image also sets it.** It is read by nvidia-container-toolkit at container-*creation* time, and whether RunPod honours an image-level `ENV` for it is unresolved. Without `graphics`, `vulkaninfo` finds no driver, and `brush` (Vulkan) and `render` (EGL) both fail — 40 minutes into a run, not at startup. |
| `PUBLIC_KEY` | your SSH public key | RunPod usually injects this. The entrypoint starts `sshd` only when it is present. |
| `B2C_PORT` | `7860` | Only if you need a different port. |
| `B2C_PREFETCH` | `0` | Only if you want the old lazy behaviour — each step downloading its own checkpoint mid-run. Default is to pull everything at start. |

### Why the volume mount path matters

RunPod's template defaults the volume mount path to `/workspace`. A volume
mounted there shadows everything beneath it. The application used to live at
`/workspace/b2c_runner`, so a pod left on the default would boot into
`No module named pipeline` while looking perfectly configured. The
application now lives at `/opt/b2c_runner` for exactly this reason, but
mount the volume at `/data` anyway — that is where every `B2C_*` path, the
HF cache and `TMPDIR` point.

If you must mount elsewhere, set `B2C_DATA_DIR` to match. Everything else
(`B2C_OUTPUT_DIR`, `B2C_LOG_DIR`, `B2C_UPLOAD_DIR`, `TMPDIR`) derives from
it unless overridden individually, and all of them have to be on the volume:
the container's writable layer is small, one 81-frame batch of dispatcher
IPC pickles is ~220 MB in each direction, and a trained splat is several GB.

## The first sixty seconds

The entrypoint runs `doctor --summary` before starting the UI, so the pod's
log answers "can this machine actually run the pipeline" before you submit
anything. Look for:

```
✓ OK    vulkan           deviceName = NVIDIA ...        <- brush can run
✓ OK    egl              NVIDIA ... /PCIe/SSE2          <- render can run
✓ OK    brush binaries   all 5 fork-specific flags present
✓ OK    step venvs       3 configured
✓ OK    attention        _sage_qk_int8_pv_fp16_triton   <- the denoise kernel
✓ OK    huggingface      accessible / accessible        <- both gated repos
```

A `FAIL` on `vulkan` almost always means `NVIDIA_DRIVER_CAPABILITIES` was
not set on the template. A `WARN` on `huggingface` means the token is
missing or has not accepted a licence — every model step will 401.

`attention` names the kernel `wan22_vace_denoise` will actually ask
diffusers for on this GPU, and what it found to back it (SageAttention,
Triton, diffusers versions), plus the upscaler's own backend. It is here
because that answer degrades silently: the step picks a backend from the
GPU's compute capability, and if `set_attention_backend` raises — a
SageAttention that did not compile for this arch, a missing Triton — it
logs a warning and runs on PyTorch native SDPA instead. Same output, an
hour longer, nothing obvious in the log. `none (PyTorch native SDPA)` is a
fine answer on a pre-Ampere card; a `WARN` here means a sage kernel was
selected and then could not be loaded, which is worth chasing.

Run it in full (`docker run IMAGE doctor`, or the UI's **Doctor** tab) for
the version lines behind each of those.

`doctor` is also a tab in the UI, and `docker run IMAGE doctor` (or
`python -m pipeline.cli doctor` over SSH) from anywhere.

## Submitting work

Open `https://<pod-id>-7860.proxy.runpod.net`. There is **one upload box**,
and what you put in it decides what runs — no input picker:

- **A dataset `.zip`** — any archive with a `metadata.json` in it, rooted at
  the dataset or one level above; the extractor finds it either way. One
  run. This is the path every verification run so far has used.
- **A single reference-sheet image** — the from-scratch path. One square
  image with the subject facing front on the left and seen from behind on
  the right, as a diffusion model generates it. Runs
  `fast_helical_native.yaml`, which splits it (front half to the
  SAM-3D-Body reconstruction and the anchor warp, back half to the denoise
  pass as its reference view), renders its own anchored views, and then
  runs `fast_helical_full`'s stages over that dataset — same `colmap/` and
  `ply/` deliverables, same Outputs box. **Least proven path**: its
  bootstrap prologue (`split_reference_sheet` → `render` →
  `generate_firstlast` → `inject_anchor`) has never executed end to end.
- **A `.zip` of image/prompt pairs** — `image1.jpg` + `image1.txt`,
  `image2.png` + `image2.txt`, ... Each image becomes its own
  reference-sheet run with its text file as the prompt, and the scheduler
  fans them across every GPU on the box — one upload instead of one
  submission per subject. A `.zip` of images with no `.txt` files works
  too: each is a reference sheet and the **Subject description** box is the
  prompt for all of them.

The upload's format also picks the pipeline — there is no workflow picker.
A dataset `.zip` runs `fast_helical_full`; an image or a zip of images runs
`fast_helical_native`. The read-only **Pipeline** field shows which.

The **Params** panel is generated from the workflow and the steps in it,
not typed as YAML. **Globals** at the top holds what the whole flow shares
(resolution, seed, the denoise prompts); below it is one collapsible section
per step, titled `<step id> (<step name>)`, holding that step's own params
with its declared defaults filled in — a dot marks the ones the workflow
sets. Knobs that exist because the underlying library has them, rather than
because this pipeline tunes them, sit behind each section's **Advanced**
fold.

The per-step sections are why `fast_helical_full`'s two brush trainings and
two denoise passes can be configured apart: `train_splat` and
`train_final_splat` get a section each. Only what you actually change is
submitted, so everything you leave alone stays owned by the workflow file
and the step defaults. `output_root` is deliberately not on the panel — it
is repointed automatically at this run's directory under `/data/output`,
and a control for it would let a run write somewhere else instead.

`python -m pipeline.cli params <workflow> --all` prints the same tree on
the command line.

**Outputs** picks what the run produces. These are real switches, not
filters on the result — an unchecked step does not run:

- **COLMAP dataset** / **Trained `.ply`** — one, the other, or both; at
  least one has to be selected. The `.ply` is a second full
  30,000-iteration brush training, so unchecking it is worth an hour.
- **Upscale dataset** (on by default) — runs the SeedVR2 upscale
  (720×1280 → 1080×1920) before the export. Off is the shorter pipeline
  that used to be the separate `fast_helical` workflow — the way to check
  whether the upscale is what degrades the output. Sets the `run_upscale`
  global.
- **Pre-upscale COLMAP dataset (debug)** — with the upscale on, also
  exports `colmap_preupscale/` from the frames as they are *before*
  SeedVR2, so you can train a splat on each and compare. Ignored with the
  upscale off (the frames are then the same, and it just guarantees the
  ordinary `colmap/`).

The **Results** tab has three things: the one `.zip` of the run's
deliverables (`colmap/` and/or `ply/`, and nothing else — the run
directory's own frames and the intermediate splat stay on the volume), the
final frames, and a **per-step contact sheet**: eight frames spaced evenly
through the batch, captured after every step, one row per step in run
order. That last one is for answering "which step broke it" by looking
rather than by reading the log; it fills in while the run is going.

The run happens in a background thread. **Closing the browser tab does not
stop it** — reopen the page and press *Attach / refresh* on the Progress
tab. Cancel takes effect at the next step boundary, not mid-step: a step is
one opaque call, often a subprocess holding the GPU, and tearing one down
mid-flight risks leaving the card in a state the next run inherits.

The same runs are available from the CLI, which is what you want for
anything long enough that you would rather have it survive in `tmux`:

```bash
python -m pipeline.cli run fast_helical_full --dataset /data/my_dataset

# a bare name is a workflow global (resolution is the only tunable one); a
# dotted one is that step's own param, which is how the two brush trainings
# and the two denoise passes are told apart
python -m pipeline.cli run fast_helical_full --dataset /data/my_dataset \
    --param 'resolution=[720, 1280]' --param denoise_pass1.steps=8 \
    --param train_final_splat.total_steps=15000

# what a run would actually use, defaults included
python -m pipeline.cli params fast_helical_full --all

# without the upscaler, to see whether that is what is degrading the output
python -m pipeline.cli run fast_helical_full --dataset /data/my_dataset \
    --param run_upscale=false

# the same output switches the UI's Outputs box drives
python -m pipeline.cli run fast_helical_full --dataset /data/my_dataset \
    --param export_ply=false --param export_colmap_preupscale=true

# from a front/back reference sheet instead
python -m pipeline.cli run fast_helical_native --reference-image /data/sheet.png \
    --prompt "a woman in a red jacket"
```

## When and where the models are downloaded

**Nothing is baked into the image** — two of these are gated (a human has to
accept the licence), they should not be redistributed inside a shareable
image, and baking ~60 GB of weights into one makes it painful to push and
pull.

Instead, **the pod pulls everything at start, in the background, and a run
blocks until the models its own workflow needs have arrived.** Three
properties, in order of how much they matter:

- **The UI comes up immediately.** The prefetch is backgrounded on purpose.
  Doing it in the foreground would leave the pod with no UI, no log and no
  healthcheck for the half hour it takes to pull ~60 GB — indistinguishable
  from a pod that failed to start, and on RunPod quite possibly killed as
  one. Watch it on the UI's **Models** tab or in `/data/logs/prefetch.log`.
- **A run never stalls mid-pipeline.** Submitting while the download is
  still going is fine: the run waits, says so on the Progress tab, and
  starts when its models are there. It will not get four stages in and then
  freeze for twenty minutes while Wan2.2 comes down — which is the failure
  this replaces, and the expensive one, because by then it has already
  burned GPU time on the stages before it.
- **Waiting is scoped to the workflow** — and, past that, to the steps a
  run's params actually select, since a `when:`-skipped step's checkpoint
  is not waited on either. `--param run_upscale=false` gates the upscale
  out, so that run does not block on SeedVR2's 6 GB:

  | Workflow | Blocks on | Total |
  |---|---|---|
  | `fast_helical_full` (`run_upscale=false`) | rmbg, sapiens2, wan22, wan22_fp8, wan22_lora | ~52.7 GB |
  | `fast_helical_full` | rmbg, sapiens2, wan22, wan22_fp8, wan22_lora, seedvr2 | ~58.7 GB |
  | `fast_helical_native` | rmbg, sapiens2, sam3dbody, wan22, wan22_fp8, wan22_lora, seedvr2, mediapipe | ~61.5 GB |

  `wan22` is now only 11.9 GB — the base repo's text_encoder, VAE,
  tokenizer and scheduler. The transformers come from `wan22_fp8` (35.2 GB
  of pre-quantized fp8) instead of 69 GB of bf16. **If your volume predates
  that change it still holds ~69 GB of `transformer/` and `transformer_2/`
  under `models--linoyts--Wan2.2-VACE-Fun-14B-diffusers` that nothing loads
  any more** — safe to delete, and worth doing before sizing the volume.

A single model failing does not abort the rest, and does not stop the pod
coming up: a pod whose token cannot reach the gated SAM-3D-Body repo still
pulls the other five, and only a run that actually needs SAM-3D-Body is
refused — at submit time, naming the licence you need to accept.

`B2C_PREFETCH=0` turns the whole thing off and restores lazy per-step
downloading; `--no-wait-for-models` does it for a single CLI run.

| Step | What | From | Lands in |
|---|---|---|---|
| `rmbg` | `briaai/RMBG-2.0` **(gated)** | `from_pretrained` | `$HF_HOME` = `/data/hf_cache` |
| `sapiens2_lite` | `facebook/sapiens2-normal-0.8b` | `from_pretrained` | `$HF_HOME` |
| `sam3d_body` | `facebook/sam-3d-body-dinov3` **(gated)** | `snapshot_download` | `$HF_HOME` |
| `wan22_vace_denoise` | `linoyts/Wan2.2-VACE-Fun-14B-diffusers` + `lightx2v/Wan2.2-Lightning` LoRAs | `from_pretrained` / `hf_hub_download` | `$HF_HOME` |
| `seedvr2` | `seedvr2_ema_3b_fp8_e4m3fn` + `ema_vae_fp16` | its vendored downloader | `$B2C_MODELS_DIR` = `/data/models/SEEDVR2` |
| `detect_face_landmarks` | MediaPipe `.task` / `.tflite` | Google Storage URL | `$B2C_MODELS_DIR/mediapipe` |
| `brush`, `render`, `render_splat`, `colmap_export`, … | nothing | — | — |

The last two rows are the ones that needed fixing. Both bypass
`huggingface_hub` and so ignore `HF_HOME`: seedvr2 hands its vendored
downloader an explicit `model_dir` which defaulted to the *relative* path
`models/SEEDVR2`, resolving against the worker subprocess's working
directory — several GB of DiT and VAE weights inside the container, on the
20 GB container disk, re-downloaded from scratch after every pod restart.
MediaPipe's did the same into `~/.cache`. Both now default under
`$B2C_MODELS_DIR`.

`doctor` reports both caches and their current size, so you can see what a
pod has already pulled:

```
✓ OK    model caches
          HF cache  /data/hf_cache: 56.8 GB
          models    /data/models: 4.2 GB
            briaai/RMBG-2.0 (0.9 GB)
            linoyts/Wan2.2-VACE-Fun-14B-diffusers (11.9 GB)
            silveroxides/Wan_2.2-fp8_scaled_hybrid (35.2 GB)
            ...
```

A `linoyts/...` line much larger than ~12 GB means the volume still holds
the bf16 transformers from before the fp8 switch. Nothing loads them.

It WARNs if `HF_HOME` is not on the volume at all, which is the shape of the
problem above rather than an instance of it.

### Is it already there?

Yes — checked, not assumed, and at two levels:

1. A marker file per model under `$B2C_MODELS_DIR/.ready/`, written after a
   successful download. Cheap, and the common case once a pod has run once.
2. Failing that, **the loader is asked directly.** Each model's probe is its
   own loading call restricted to local files — `snapshot_download(...,
   local_files_only=True)` for the huggingface_hub ones, a file check for
   the other two. So weights put on the volume by something *other than
   this prefetch* — an earlier pod, an older image, a hand-run
   `pipeline/envs/wan22/setup.sh` — are recognised, and the marker is
   written so the check is cheap from then on.

That second layer is the one that makes a reused network volume worth
having: **pull once, and every later pod starts warm and begins its first
run immediately.** Without it a warmed volume would look completely cold and
re-walk every repo.

Manual control, if you want it:

```bash
python -m pipeline.cli prefetch --status          # what's present, what isn't
python -m pipeline.cli prefetch                   # pull everything now
python -m pipeline.cli prefetch --only wan22,seedvr2
python -m pipeline.cli prefetch --force           # re-verify against the network
```

`--force` is the one to reach for if you suspect a cache was half-deleted:
it ignores the markers and re-checks every file.

**Budget the volume accordingly.** The Wan2.2 checkpoint alone is the bulk
of it; 100 GB is a reasonable floor once the whole set plus run output is on
there.

## Debugging without rebuilding

The image ships the tools for all of this. Nothing here needs a new build.

**Watch a run.** Every run writes `/data/logs/<workflow>-<timestamp>.log`
with timestamps, elapsed time, per-step timing and a VRAM summary after each
step. `tail -f` it, or read it from the UI's log pane. Subprocess steps
(`wan22_vace_denoise`, `seedvr2`, `sam3d_body`) and the external binaries
(`brush`, `brush-splat-render`) relay their output line by line as it
happens — they used to be silent until they finished.

**Get a shell.** SSH if `PUBLIC_KEY` was set; otherwise
`docker exec`-equivalent through RunPod's web terminal. From a local
container: `docker run --rm -it b2c/pipeline:latest bash`, or
`bash -c '...'` for one command.

**Patch the pipeline in place.** The package is installed editable, so an
edit takes effect on the next run with no rebuild:

```bash
cd /opt/b2c_runner
git pull                     # git is in the image
nano pipeline/steps/brush.py
python -m pipeline.cli run ...
```

**Run one step by hand.** Each isolated env's interpreter is a real
interpreter:

```bash
/opt/venv_sam3dbody/bin/python -c "from sam_3d_body import load_sam_3d_body; print('ok')"
/opt/venv_wan22/bin/python -c "import diffusers; print(diffusers.__version__)"
```

**Check the graphics stack separately from a failing step.**

```bash
vulkaninfo --summary          # brush's backend
brush --help                  # does this binary have the fork's flags?
nvidia-smi
```

**Raise the log level.** `B2C_DEBUG=1` sets DEBUG everywhere, including
inside the subprocess workers.

**Get results off the pod.** `/data/output/<run>/` holds the final dataset,
the splat and the COLMAP export. The UI's Results tab zips it for download;
`rsync`, `scp` and `runpodctl send` are all in the image.

## Cost safety

The image does **not** arm an auto-shutdown. `scripts/pod_bootstrap.sh` has
one (`runpodctl stop pod` after `AUTO_SHUTDOWN_HOURS`) that predates the
container, and it is worth copying onto the pod for an unattended run —
`fast_helical_full.yaml` at production settings is hours of GPU time, and a
step that hangs bills exactly like a step that works.
