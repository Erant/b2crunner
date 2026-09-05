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
| **RAM** | **64 GB+** | Not optional for `fast_helical_native`. Its two denoise passes use `keep_loaded: true`, which holds ~47 GB of Wan weights resident in host RAM between them so they come off the volume once instead of twice. Too little and the worker is OOM-killed mid-run, which looks like a mysterious dead worker rather than a sizing mistake. `doctor` warns. Drop `keep_loaded` from the workflow if you must run smaller. |
| HTTP ports | `7860` | The web UI *and* the HTTP API, on one port. RunPod proxies it at `https://<pod-id>-7860.proxy.runpod.net`. |
| TCP ports | `22` | SSH. Optional, but it is how you get a shell if the UI won't start. |

### Environment variables

| Variable | Value | Notes |
|---|---|---|
| `HF_TOKEN` | `hf_...` | **Required.** From an account that has accepted the licences for `briaai/RMBG-2.0` and `facebook/sam-3d-body-dinov3`; both are gated and a human has to click through each one. `doctor` reports whether the token can actually reach them. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,graphics,display` | **Set this here even though the image also sets it.** It is read by nvidia-container-toolkit at container-*creation* time, and whether RunPod honours an image-level `ENV` for it is unresolved. Without `graphics`, `vulkaninfo` finds no driver, and `brush` (Vulkan) and `render` (EGL) both fail — 40 minutes into a run, not at startup. |
| `B2C_API_TOKEN` | a long random string | **Set this.** It turns on the HTTP API at `/api/v1` (as a bearer token) *and* puts the web UI behind a login form that takes it as the password, username `b2c`. Without it the API is not served at all and the UI is open to anyone who can guess the pod id — the proxy URL below is public. Generate one with `openssl rand -hex 24`. |
| `PUBLIC_KEY` | your SSH public key | RunPod usually injects this. The entrypoint starts `sshd` only when it is present. |
| `B2C_SHUTDOWN_COMMAND` | `runpodctl stop pod $RUNPOD_POD_ID` | What to do about the *host* when `POST /api/v1/shutdown` stops the container. Optional, and empty by default: the container knows how to stop itself, but it cannot know whether that ends a bill — see [Stopping when you are done](#stopping-when-you-are-done). Needs `runpodctl` on the pod (**it is not in the image**) and `RUNPOD_API_KEY` / `RUNPOD_POD_ID` set. |
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
✓ OK    colmap           COLMAP 4.2.0 ... with CUDA     <- the pose refinement
✓ OK    step venvs       3 configured
✓ OK    attention        _sage_qk_int8_pv_fp16_triton   <- the denoise kernel
✓ OK    huggingface      accessible / accessible        <- both gated repos
```

A `FAIL` on `vulkan` almost always means `NVIDIA_DRIVER_CAPABILITIES` was
not set on the template. A `WARN` on `huggingface` means the token is
missing or has not accepted a licence — every model step will 401. A `WARN`
on `colmap` saying `built WITHOUT CUDA` is not a correctness problem: the
refinement still runs, on the ONNX CPU provider, at roughly three minutes
per training instead of seconds.

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

- **A single reference-sheet image** — the from-scratch path. One square
  image with the subject facing front on the left and seen from behind on
  the right, as a diffusion model generates it. Runs
  `fast_helical_native.yaml`, which splits it (front half to the
  SAM-3D-Body reconstruction and the anchor warp, back half to the denoise
  pass as its reference view), renders its own anchored views, and then
  runs the workflow's own six-stage tail over that dataset — same
  `colmap/` and `ply/` deliverables, same Outputs box. The prologue also
  builds a Gaussian cap of the face from the front half (the Sapiens2
  pointmap head, ~6.5 GB more prefetch).
- **A `.zip` of image/prompt pairs** — `image1.jpg` + `image1.txt`,
  `image2.png` + `image2.txt`, ... Each image becomes its own
  reference-sheet run with its text file as the prompt, and the scheduler
  fans them across every GPU on the box — one upload instead of one
  submission per subject. A `.zip` of images with no `.txt` files works
  too: each is a reference sheet and the **Subject description** box is the
  prompt for all of them.

Both shapes run `fast_helical_native` — there is no workflow picker. The
read-only **Pipeline** field just confirms it.

**Settings** holds what the pipeline chose to put in front of you, which for
`fast_helical_native` is four things: **Resolution**, **Framing**, **Seed**,
and — behind a *More settings* fold — **Face splat**. These are not typed as
YAML and they are not a fixed list in the UI's code: the workflow file
declares them, with their type, default, help text and choice list, in a
`settings:` block (see pipeline/README.md). Adding one is an edit to the
workflow, not to `webui.py`.

Everything the pipeline did *not* promote is still there and still editable,
behind the **Per-step settings** fold in the right-hand column: one
collapsible section per step, titled `<step id> (<step name>)`, holding that
step's own params with its declared defaults filled in — a dot marks the ones
the workflow sets. Knobs that exist because the underlying library has them,
rather than because this pipeline tunes them, sit behind each section's
**Advanced** fold. It is ~300 controls; on a normal run you open none of
them.

The per-step sections are why `fast_helical_native`'s two brush trainings and
two denoise passes can be configured apart: `train_splat` and
`train_final_splat` get a section each. A param the workflow wires to a
pipeline setting (`render`'s `resolution`, say) is not shown there at all —
its one editable home is the Settings box. Only what you actually change is
submitted, so everything you leave alone stays owned by the workflow file
and the step defaults. `output_root` has no control anywhere: it is a plain
`globals:` entry rather than a declared setting, and the form draws
declarations only — it is repointed automatically at this run's directory
under `/data/output`, and a control for it would let a run write somewhere
else instead.

`python -m pipeline.cli params <workflow>` prints the same settings and
outputs on the command line, `--all` adds every step param.

**Outputs** picks what the run produces, and is likewise the workflow's own
`outputs:` block. These are real switches, not filters on the result — an
unchecked step does not run:

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
  SeedVR2, so you can train a splat on each and compare. It declares
  `requires: run_upscale`, so with the upscale off the checkbox is greyed
  out: the frames would then be the same as `colmap/`'s, and it used to
  silently give you that instead, under a name that says otherwise.
- **Debug bundle** (on by default) — the `debug/` directory in the result
  `.zip`: refine_cameras' given-vs-refined camera models, the face splat's
  stats and depth visualisations, the face `.ply` files, and
  `intermediate_splat.ply` — the splat the helical re-render is built from,
  and therefore the first thing to look at when that re-render is wrong.
  Unlike every other switch here it skips no work: those dumps are a side
  effect of steps the run needs anyway, so they are written to the volume
  either way and this decides only whether they are packaged. Worth turning
  off for a run you are only going to look at, because the intermediate
  splat is hundreds of MB against a few hundred KB for the rest of that
  directory. It is not a deliverable on its own — a run with every other
  output off is still refused.

- **Intermediate COLMAP dataset (debug)** — also exports
  `colmap_intermediate/`: the dataset the **first** brush training is
  handed, i.e. the frames as `denoise_pass1` leaves them plus the RMBG
  mattes and the normal maps computed for that training. Every other export
  in a run describes frames from after the helical re-render; this is the
  only look at what the splat driving that re-render actually saw, which is
  the question behind a bad re-render. No interaction with the upscale, and
  it is a valid sole output. Sets `export_colmap_intermediate`.

The **Results** tab has three things: the one `.zip` of the run's
deliverables (`colmap/` and/or `ply/`, plus `debug/` and either debug export
you asked for, and nothing else — the run directory's own frames and the intermediate
splat stay on the volume), the
final frames, and a **per-step contact sheet**: eight frames spaced evenly
through the batch, captured after every step, one row per step in run
order. That last one is for answering "which step broke it" by looking
rather than by reading the log; it fills in while the run is going.

The UI asks for a password when `B2C_API_TOKEN` is set: username `b2c`, the
token as the password. With it unset there is no login and no HTTP API.

The run happens in a background thread. **Closing the browser tab does not
stop it** — reopen the page and press *Attach / refresh* on the Progress
tab. Cancel takes effect at the next step boundary, not mid-step: a step is
one opaque call, often a subprocess holding the GPU, and tearing one down
mid-flight risks leaving the card in a state the next run inherits.

The same runs are available from the CLI, which is what you want for
anything long enough that you would rather have it survive in `tmux`:

```bash
python -m pipeline.cli run fast_helical_native --reference-image /data/sheet.png \
    --prompt "a woman in a red jacket"

# a bare name is a workflow global (resolution is the only tunable one); a
# dotted one is that step's own param, which is how the two brush trainings
# and the two denoise passes are told apart
python -m pipeline.cli run fast_helical_native --reference-image /data/sheet.png \
    --param 'resolution=[720, 1280]' --param denoise_pass1.steps=8 \
    --param train_final_splat.total_steps=15000

# what a run would actually use, defaults included
python -m pipeline.cli params fast_helical_native --all

# without the upscaler, to see whether that is what is degrading the output
python -m pipeline.cli run fast_helical_native --reference-image /data/sheet.png \
    --param run_upscale=false

# the same output switches the UI's Outputs box drives
python -m pipeline.cli run fast_helical_native --reference-image /data/sheet.png \
    --param export_ply=false --param export_colmap_preupscale=true

# the debug COLMAP exports, which the UI's Outputs box also drives:
# colmap_intermediate/ is the dataset the FIRST brush training is handed
# (denoised frames + their mattes and normals), colmap_preupscale/ the same
# idea one stage before SeedVR2
python -m pipeline.cli run fast_helical_native --reference-image /data/sheet.png \
    --param export_colmap_intermediate=true
```

## Automating it

Everything the UI does, `/api/v1` does — same queue, same GPU slots, same
`.zip`. It is the same server process on the same port: the API is
registered on a FastAPI app and the Gradio UI is mounted underneath it, so
a run submitted by curl shows up in the browser's run picker and vice
versa. There is no second scheduler and no second set of GPU slots.

Every route needs `B2C_API_TOKEN` as a bearer token, and none of them is
served if that variable is unset. `--no-api` on `pipeline.cli ui` serves the
UI alone.

### From this repo

`pipeline.cli api` is a client for all of it. No gradio, no fastapi, no
torch and no GPU: a plain `pip install -r requirements.txt` is enough, so
it runs from a laptop that could not host the pipeline itself. (The
`pipeline.client` module underneath needs only `requests`, if you would
rather drive it from your own script than from the CLI.)
`--url`/`$B2C_API_URL` and `--token`/`$B2C_API_TOKEN` say where and as whom.

```bash
export B2C_API_URL=https://<pod-id>-7860.proxy.runpod.net
export B2C_API_TOKEN=...        # the value on the pod template

# the whole job in one command: submit, watch every stage finish with what
# it cost, then download the .zip. `--param` is spelled exactly as it is
# for a local `pipeline.cli run`
python -m pipeline.cli api run sheet.png --prompt 'a woman in a red jacket' \
    --param run_upscale=false --param train_final_splat.total_steps=15000 \
    -o results/
```
```
run fast_helical_native-20260905-101500-a1b2c3
  queued
  waiting for model download: wan22, seedvr2 (~48 GB)
  running
  [ 1/42] split_sheet                       1.9s
  [ 2/42] reconstruct_body                 44.2s
  [ 3/42] detect_face                       0.6s
  ...
  [33/42] upscale                        skipped
  ...
  [42/42] train_final_splat               1h04m

OK: complete — 81 frames in /data/output/...  (2h11m wall clock)
   50%  0.9 GB of 1.8 GB
  100%  1.8 GB of 1.8 GB
saved results/fast_helical_native-20260905-101500-a1b2c3-result.zip (1.8 GB)
```

A failed run prints the last 40 log lines instead of a download, which is
usually the whole diagnosis. Interrupting the watch stops watching, not the
run — it is a process on the pod.

The rest, one per route:

```bash
python -m pipeline.cli api health              # slots, queue depth, model status
python -m pipeline.cli api workflows fast_helical_native   # what --param accepts
python -m pipeline.cli api submit sheet.png    # queue it and return the name
python -m pipeline.cli api submit batch.zip    # one run per pair, fanned across GPUs
python -m pipeline.cli api submit /data/uploads/sheet.png --remote   # already there
python -m pipeline.cli api runs                # everything on the pod, in flight first
python -m pipeline.cli api status <run>
python -m pipeline.cli api follow <run>        # attach to one already going
python -m pipeline.cli api log <run> --tail 200
python -m pipeline.cli api result <run> -o results/
python -m pipeline.cli api cancel <run>
python -m pipeline.cli api shutdown [--force]  # stop the server and its container
python -m pipeline.cli api schema              # the OpenAPI document
```

`--remote` is the one worth remembering for a batch: put the sheets on the
volume with `rsync` or `runpodctl send` and name the path, rather than
pushing hundreds of MB back through the proxy to reach a disk they are
already on.

### With curl

```bash
POD=https://<pod-id>-7860.proxy.runpod.net
AUTH="Authorization: Bearer $B2C_API_TOKEN"

# is there capacity, and have the weights landed?
curl -sH "$AUTH" $POD/api/v1/health | jq '{gpus, queued}'

# the whole surface, machine-readable
curl -sH "$AUTH" $POD/api/v1/openapi.json | jq '.paths | keys'

# what can I set? — the workflow's own settings: and outputs: blocks,
# with types, defaults and choice lists. These names are exactly what
# `settings` below accepts; a name it does not list is refused, not ignored
curl -sH "$AUTH" $POD/api/v1/workflows/fast_helical_native | jq '.settings[].name'

# submit a reference sheet
NAME=$(curl -sH "$AUTH" \
  -F file=@sheet.png \
  -F prompt='a woman in a red jacket' \
  -F 'settings={"run_upscale": false, "export_ply": false}' \
  $POD/api/v1/runs | jq -r '.runs[0].name')

# ...or a .zip of image/prompt pairs: one run per pair, fanned across
# every GPU, exactly as the UI's upload box does it
curl -sH "$AUTH" -F file=@subjects.zip $POD/api/v1/runs | jq '.runs[].name'

# ...or a sheet already on the volume, which is the right way for anything
# large — rsync/runpodctl it in, then name the path
curl -sH "$AUTH" -H 'Content-Type: application/json' \
  -d '{"reference_image": "/data/uploads/sheet.png", "prompt": "..."}' \
  $POD/api/v1/runs

# watch it
curl -sH "$AUTH" $POD/api/v1/runs/$NAME | jq '{status, current, total, message}'
curl -sH "$AUTH" "$POD/api/v1/runs/$NAME/log?tail=100" | jq -r .log

# collect it: the same archive the Results tab builds, cached beside the
# run so asking twice does not re-zip a couple of gigabytes
curl -sH "$AUTH" -OJ $POD/api/v1/runs/$NAME/result

# stop one (at the next step boundary, not mid-step)
curl -sXPOST -H "$AUTH" $POD/api/v1/runs/$NAME/cancel

# stop the server, and with it the container — 409 while anything is still
# going, unless you mean it. See "Stopping when you are done" below
curl -sXPOST -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"force": false}' $POD/api/v1/shutdown
```

A poll loop is three lines, because `status` is terminal exactly when it
leaves `queued`/`running`:

```bash
until curl -sH "$AUTH" $POD/api/v1/runs/$NAME \
      | jq -e '.status | . != "queued" and . != "running"' >/dev/null; do
    sleep 30
done
```

Codes worth handling: **400** is a submission to fix and its `detail` says
what (an unknown setting, a `.txt` where a sheet should be, a run that
would export nothing); **409** on `/result` means the run is still going —
its exports are the last steps, so a `.zip` built now would look whole and
not be; **404** on `/result` means it finished without producing a
deliverable. `GET /api/v1/openapi.json` is the generated schema for the
rest — behind the same token, since FastAPI's own docs routes would
otherwise advertise every route of a guarded API on a public URL.

Two things only the pod can answer, so check them on a new template: that
the RunPod proxy forwards the `Authorization` header, and that a
multi-gigabyte `/result` download survives it. If either does not, the
`.zip` is still on the volume at `/data/output/<run>-result.zip` and
`runpodctl send` still works.

**What the API does not give you is a durable queue.** `GpuScheduler` holds
queued jobs in memory; if the container restarts, anything not yet started
is gone (and anything running is stopped). Treat a `queued` reply as live
state, not a receipt — a client submitting a batch should confirm each run
reached `running` or a terminal status rather than assuming the queue
survived.

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
  | `fast_helical_native` (`run_upscale=false`) | rmbg, sapiens2, sapiens2_pointmap, sapiens2_seg, sam3dbody, moge2, mediapipe, wan22, wan22_fp8, wan22_lora | ~72.5 GB |
  | `fast_helical_native` | rmbg, sapiens2, sapiens2_pointmap, sapiens2_seg, sam3dbody, moge2, mediapipe, wan22, wan22_fp8, wan22_lora, seedvr2 | ~78.5 GB |

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
| `sapiens2_lite` | `facebook/sapiens2-normal-1b` | `from_pretrained` | `$HF_HOME` |
| `sam3d_body` | `facebook/sam-3d-body-dinov3` **(gated)** + `Ruicheng/moge-2-vitl-normal` (FOV) | `snapshot_download` / `from_pretrained` | `$HF_HOME` |
| `sam3d_body`, `fit_head_to_face` | `facebookresearch/dinov3` SOURCE (~20 MB) — the backbone is a `torch.hub.load` of GitHub, not a checkpoint | `torch.hub` | `$TORCH_HOME/hub` = `/data/caches/torch/hub` |
| `face_pointmap_splat`, `pointmap_elevation_views` | `facebook/sapiens2-pointmap-1b` | `from_pretrained` | `$HF_HOME` |
| `sapiens2_seg` | `facebook/sapiens2-seg-1b` | `from_pretrained` | `$HF_HOME` |
| `wan22_vace_denoise` | `linoyts/Wan2.2-VACE-Fun-14B-diffusers` + `lightx2v/Wan2.2-Lightning` LoRAs | `from_pretrained` / `hf_hub_download` | `$HF_HOME` |
| `seedvr2` | `seedvr2_ema_3b_fp8_e4m3fn` + `ema_vae_fp16` | its vendored downloader | `$B2C_MODELS_DIR` = `/data/models/SEEDVR2` |
| `detect_face_landmarks` | MediaPipe `.task` / `.tflite` | Google Storage URL | `$B2C_MODELS_DIR/mediapipe` |
| `refine_cameras` | COLMAP `aliked-n32.onnx` + `aliked-lightglue.onnx` (~65 MB) | COLMAP's GitHub release | `$B2C_MODELS_DIR/colmap` |
| `brush`, `render`, `render_splat`, `colmap_export`, … | nothing | — | — |

The dinov3 row is the odd one: it is code, not weights, and `torch.hub`'s
cache test is `os.path.exists(repo_dir)` — so two workers racing it (one per GPU,
see `pipeline/gpu_scheduler.py`) can leave a directory that exists with files
missing, which torch then reuses forever. The prefetch's probe checks the tree
is whole rather than merely present, and its fetch forces a reload; that is
what heals a volume already carrying a damaged checkout.

The three rows before the last are the ones that needed fixing. All bypass
`huggingface_hub` and so ignore `HF_HOME`: seedvr2 hands its vendored
downloader an explicit `model_dir` which defaulted to the *relative* path
`models/SEEDVR2`, resolving against the worker subprocess's working
directory — several GB of DiT and VAE weights inside the container, on the
20 GB container disk, re-downloaded from scratch after every pod restart.
MediaPipe's did the same into `~/.cache`, and COLMAP's own model cache is
`$HOME/.cache/colmap` for the same reason — which is why `refine_cameras`
prefetches the two ONNX graphs itself and passes COLMAP explicit
`--...model_path` flags rather than letting it fetch them. All three now
default under `$B2C_MODELS_DIR`.

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

**Read a crashed binary.** `brush` and `brush-splat-render` both work
inside a temp directory that goes away with the step — which once left a
crash on a pod with nothing to look at but its exit code. Both now write
`/data/logs/crashes/<binary>-<timestamp>/` before that happens, and the run
log names the directory. Each holds a `report.txt` (argv, exit code, the
tail of the binary's own output, and the Vulkan/driver environment —
`NVIDIA_DRIVER_CAPABILITIES` and friends, the thing that decides whether
either binary can reach a GPU at all) plus what it was working on:

| | saved | not saved |
|---|---|---|
| `brush` | the COLMAP model's `.txt` files, a description of the export | the training frames — hundreds of MB, and still on the volume afterwards |
| `brush-splat-render` | `cameras.json`, a per-frame manifest, the last few frames written | the rest of the orbit's frames |

It triggers on any non-zero exit — including one the guard then tolerates —
and on a clean exit that produced nothing. Note that a crash which left the
work *complete* is not a failed step: brush's known shutdown SIGSEGV, and
the same thing in the renderer, are logged at WARNING, saved, and the
output is used.

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
the splat and the COLMAP export. The UI's Results tab zips it for download,
`GET /api/v1/runs/<run>/result` does the same over HTTP, and `rsync` and
`scp` are in the image. **`runpodctl` is not** — this page claimed it was
until 2026-09-05, and nothing in `docker/Dockerfile` installs it. If you
want it on a pod, fetch it there.

## Stopping when you are done

```bash
python -m pipeline.cli api shutdown
# or, for the whole unattended job:
python -m pipeline.cli api run sheet.png --prompt '...' -o results/ \
    --shutdown-when-done
```

`POST /api/v1/shutdown` stops the GPU workers, then the server, and the
container exits. It is refused with **409** while anything is queued or
running — a run in flight is hours of GPU already spent and its exports are
its last steps, so stopping then loses all of it rather than most of it.
`{"force": true}` (or `api shutdown --force`) overrides, and the reply names
what it abandoned. A run this server does not manage — a `pipeline.cli run`
someone started over SSH, or a status file a crashed server left at
`running` — is reported in `unmanaged_running` and warned about, but not
refused on: one stale file must never make a pod impossible to stop. `--shutdown-when-done` stops only after the result `.zip`
is on your disk; a download that failed leaves the pod up, because stopping
the container is what makes everything still on its volume unreachable.

**What the container can and cannot do.** It can stop itself, and that is
all it does by default — deliberately, because this image is not a RunPod
image. It runs under `docker-compose` too, and on a rented pod **a container
that exits is not a pod that stopped billing.** What a host should do about
a stopped container is a template concern, like everything else on this
page, so it comes from an environment variable rather than from a code path
in the image:

```
B2C_SHUTDOWN_COMMAND=runpodctl stop pod $RUNPOD_POD_ID   # a rented pod
B2C_SHUTDOWN_COMMAND=shutdown -h now                     # a machine you own
B2C_SHUTDOWN_COMMAND=curl -fsS -X POST https://…/done     # tell something else
```

It runs *after* the GPU workers are stopped and before the server exits —
that order matters, because a command that makes the machine disappear
seconds later would otherwise kill a `pipeline.run_worker` mid-write.

**That is a bound, not a guarantee.** A worker is given SIGTERM and ten
seconds; it stops at the next *step boundary*, and a step is one opaque
call — a 30,000-iteration brush training has no boundary for an hour. So a
worker inside a long step is still there when the wait expires, is named in
the log as such, and goes down with the container like anything else.
Shutting down between runs is clean; shutting down mid-training is not, and
that is why the endpoint refuses without `force`.

The command itself gets 30 seconds. One that fails, hangs or is missing is
logged and the container still stops, since a hook that did not fire means
the host was not told, not that this server should stay up and keep billing.
The entrypoint prints which of the two cases you are in at startup.

Queued runs are dropped rather than started: a terminated worker frees its
GPU slot, and a free slot would otherwise dispatch the next job — so
stopping the server used to *spawn* fresh workers that began pulling
checkpoints as it exited.

Note that `runpodctl` is **not in this image**. Install it on the pod if you
want the first line above.

## Cost safety

The shutdown endpoint above is the deliberate case. For the unattended one,
the image does **not** arm an auto-shutdown: `scripts/pod_bootstrap.sh` has
one (`runpodctl stop pod` after `AUTO_SHUTDOWN_HOURS`) that predates the
container, and it is worth copying onto the pod —
`fast_helical_native.yaml` at production settings is hours of GPU time, and a
step that hangs bills exactly like a step that works, and neither `/shutdown`
nor `--shutdown-when-done` fires for a run that never finishes.
