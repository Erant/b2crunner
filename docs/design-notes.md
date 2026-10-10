# Design notes

Why the pipeline is built the way it is: what each setting does and costs, the measurements behind the defaults, and the dead ends not to repeat. Older write-ups this replaces are in git history.

## Settings

The background behind the settings and outputs declared at the top of
`pipeline/workflows/helical.yaml`.
The UI shows a one-line help for each; this is the longer version: what
each one does, what it costs, and why the default is what it is.
`python -m pipeline.cli params helical --all` prints every effective value.

### Basics

**resolution** — frame size, width x height. Several steps have to agree on
it: the mesh render takes it whole, while `wan22_vace_denoise` and
`render_splat` split it into width and height. Only sizes the Wan denoise
model was trained for are offered.

**framing** (`full` / `torso` / `bust` / `head`) — how much of the body
fills the frame. Both renderers that see it use it: the mesh render
(`render_initial_views`) and the splat re-render (`render_subject`). The
splat is trained on cameras aimed for the preset, and the helical orbit is
rebuilt around it. The tighter presets come from the reconstructed
skeleton. When one cannot be computed, `render` falls back to `full` and
`render_subject` follows.

**input_layout** (`auto` / `sheet` / `single`):
- `sheet` is a two-panel front/back sheet, cut down the middle. The front
  half is the subject; the back half is the reference both denoise passes
  condition on.
- `single` is one frontal photo. Pass 1 runs on the injected photograph
  alone, and pass 2 conditions on pass 1's own rear-view frame
  (`pick_rear_view`).
- `auto` counts figures with a person detector. Two, one on each side of
  the centre, is a sheet; one is a photo; anything else stops the run and
  asks you to set this. It classified 27 of 27 test sheets and every single
  photo tried correctly (`steps/reference_sheet.py`).

**seed** — one seed for every stochastic stage: both denoise passes and the
SeedVR2 upscale.

**low_vram** — runs the Wan denoise the way ComfyUI does on a ~12 GB card:
- the eight VACE hints are computed one at a time, beside the block they
  feed;
- per-token work is chunked;
- weights are streamed a block at a time from the memory-mapped checkpoints;
- the VAE is tiled.

The transformer's arithmetic is bit-identical to the stock path. The tiled
VAE blends its seams, so frames are near-identical rather than identical.
It uses about 7.5 GiB of VRAM and ~3 GB of host RAM beyond the page cache;
on an RTX 4070 Ti at 720x1280x81 a denoise step takes ~105 s. Off is the
32 GB-card path.

### Denoise conditioning

**weak_skeleton** — shows pass 1 the DWPose skeleton on its first denoise
step only; the remaining steps get the same drawing without the sticks
(silhouette, relief and face splat stay). The first step is where the
skeleton sets the pose and the turn. On subjects whose costume has
coloured lines, every step that saw the sticks painted them in as glowing
piping. With the sticks on the first step only, the same seed gives the
same picture without that ink. It costs one extra composite per frame and
one extra VAE encode of the control video.

**outline_strength** — how dark the flat silhouette fill in the body-model
drawing is, as a percentage from the #7F7F7F ground (0, no silhouette) to
black (100). The body model's outline is wrong wherever hair and clothing
extend past it, so the fill is kept faint (6.25 ≈ #777777): a hint under the
skeleton rather than a shape to trace. With `re_outline` off, this drawing is
what pass 1 sees. With it on, this is what the 480p re-outline pass sees,
and pass 1 gets the re-outlined drawing at `reoutlined_strength`.

**re_outline** — draws pass 1's silhouette from a splat of the subject
instead of the body mesh:
1. The same control video goes through an extra Wan denoise at 480x832
   (pass 1's settings at four steps, 2 high / 2 low).
2. The result is matted with RMBG.
3. A splat is trained on those mattes, using the orbit's own cameras.
4. The orbit is re-rendered, with the splat's coverage as the outline.

This gives one silhouette that every frame agrees on, hair and clothing
included. The cost is a third 81-frame denoise (the resident experts are
reused) and a third splat training. With it on, `debug/reoutline/` holds
the 480p frames and mattes and `debug/reoutline_splat.ply` holds the
splat. See [below](#re-outline-the-silhouette-from-a-splat-not-the-mesh).

**reoutlined_strength** — the silhouette fill for the re-outlined drawing.
That outline can be trusted where the body model's cannot, so it is drawn
darker: 20 ≈ #666666.

**outline_relief** (`none` / `depth`) — what fills the silhouette in both
drawings. `none` is flat grey. `depth` fills it with the body model's depth
at the same mean grey: smoothed to body scale, cut to 16 levels over 80 cm
centred on the orbit target, near surfaces lighter and far ones darker.
The point: a flat silhouette under a 2-D skeleton looks the same from the
front and from behind, mirrored, so the denoise can turn the head without
the torso. The relief adds back which surface faces the camera. Where the
re-outlined silhouette is wider than the model, it takes the nearest model
depth.

**outline_relief_amplitude** — how far the relief's near and far ends sit
on either side of the fill strength, on the same percentage ramp. 6.25 is
8 grey levels each way. At `outline_strength` 6.25 the nearest level then
lands on the ground colour, which only a point 40 cm nearer than the
orbit target reaches.

**skeleton_occlusion_m** — hides every skeleton joint that the body model
hides from the camera, and every bone ending on one. For example, the far
arm behind the torso in profile is not drawn through it. That is DWPose's
own rule for undetected keypoints, and it stops the drawing from
supporting a mirrored reading of a profile. The value is how deep, in
metres, a limb joint may sit behind the model's surface and still count as
visible. Torso joints get 2–2.5x that depth and face landmarks half, per
body2colmap's table. 0.12 keeps every joint visible from every side and
drops the ones behind another body part. 0 draws every bone.

**specular_suppress** (0–1) — pulls highlights down in pass 1's output
before the intermediate splat trains on it. Pass 2 conditions on a
re-render of that splat, so pass 1's inconsistent highlights would
otherwise be baked in as SH sparkle. The specular excess is the per-pixel
minimum channel above the batch's `mean + eta * std`. It is subtracted
equally from all three channels, so colour and diffuse shading stay. The
photograph's own frame is untouched. See `pixel_ops`' `specular_eta` /
`specular_blur` for the shape of the cut.

### Face

**face_splat** — builds a Gaussian splat of the subject's real face and
composites it onto the skeleton drawings wherever the camera can see it
(±60°). Off leaves the drawings with no face at all; there is no landmark
fallback. It costs a third Sapiens2 1B head (segmentation, ~6.5 GB) on top
of the normal-map and pointmap heads.

**face_refine** — before the final training:
- fits the refit body's head to every frame that shows the face
  (neck/head pose plus the MHR expression blendshapes, through the frame's
  own camera);
- replaces each frame's eyes, which diffusion reinvents per frame, with
  eyeballs textured from the anchor photograph and rendered through that
  frame's fitted lids.

Measured on four subjects, eye error against the eye model fell from 51–82
to 28–56: closed slits and blue smears became open eyes with an iris. It
needs the body refit. See [below](#face-per-view-and-the-anchors-eyes).

**face_rig** — hands the trainer the per-view head fit as vertex
displacements of the rig's face core (`build_face_rig`), so the canonical
face follows each frame's expression. Off by default: the fit is
landmark-driven, and MediaPipe's landmarks on views 60–85° off-axis put the
face in the wrong place. Those views pull the canonical face apart into a
smear at 70–110°. Without the deltas, the profile is clean and the frontal
face and pasted eyes are unchanged (the eyes live in the frames).
`face_motion_cm` and `hold` on `build_face_rig` are its knobs.

**photo_priority** (0–1) — in the intermediate training, how far the
denoised frames yield to the photograph wherever its camera sees the body
head-on enough. The frame's loss weight there is 1 minus this value. The
step also adds masked copies of the photograph as supporting views. Off by
default, copies included, for two reasons: the frame at the anchor pose
turned out to be the denoiser's repaint rather than the photograph (17–20
dB from it), and the copies left a layer of opacity behind the front. See
[below](#photo-priority-off-by-default).

### Geometry and cameras

**refine_cameras** — bundle-adjusts the generated orbit against the frames
before each training, then restores the gauge so the splat keeps its
scale. It is worth +0.63 PSNR / +0.0045 SSIM on held-out views, against a
seed spread of ±0.003 ([below](#camera-refinement-refine_cameras-refine_cameras_final)).
It runs twice, because both camera paths are generated rather than
measured. At 81 frames, features and matching take ~27 s with CUDA COLMAP
(~209 s on CPU), plus a CPU-bound bundle adjustment. It never sees the
supporting views, which are rendered from these poses and carry no
independent evidence about them.

**refit_body** — after the first training, moves the SAM 3D Body mesh and
skeleton onto the splat, which by then sits centimetres off the starting
body. The final training's hollow loss then forbids weight behind a body
that is actually where the splat is. That lets its margin drop from 5 cm to
3 cm at the same PSNR, with a quarter of the weight behind the body. The
refit skeleton is also what rigging hangs from. It costs a probe render and
~90 s of fitting. See [below](#body-refit-to-the-intermediate-splat).

**body_rig_intermediate** — deforms the intermediate splat per training
view with the initial body's skeleton, as `body_rig` does for the final
training. The generated frames swing the arms by centimetres between
segments of the orbit. A single static body cannot explain all of them, so
without the rig the splat shows translucent double limbs, which pass 2
inherits. Measured: hands +27 % sharper, body +7 %, head +9 %, at no
extra time. Only the real frames are deformed; the face supporting views
stay undeformed, and that is what keeps the head gain.

### Orbit extension

**extend_orbit** — after pass 2, lengthens the orbit with two more 81-frame
VACE passes (`steps/extend_orbit.py`):
- The **before** pass ends on pass 2's first 40 frames, kept as inactive
  frames (mask 0), and paints 41 new frames ahead of them.
- The **after** pass starts on pass 2's last 41 frames and paints 40 after
  them.

The new cameras continue the helix past its ±30° band, ramped
`extend_tilt_deg` further out by the far end. Each pass is conditioned on
renders of a splat trained on pass 2's 81 frames. Whatever the passes paint
over the overlap is discarded; pass 2's frames stay. Everything downstream
then runs on 162 views instead of 81. The cost is two more denoises (no
model reload), a fourth splat training, and twice the frames in every later
step. `debug/extension_input/` holds the control videos and
`debug/extension_splat.ply` the guide splat.

**extend_tilt_deg** — how far past the ±30° band the outermost extension
frames reach. 0 carries the flat lead-in and lead-out on, which repeats
views: 32 of the 81 new frames land within 5° of an existing view. 10
leaves none within 5°. The default, 40 (ending at ±70°), is there because a
splat is only constrained up to its highest training view. Above that, the
view-dependent colour extrapolates: upward-facing surfaces seen from
+55°/+70° came out mottled at a tilt of 10 and clean at 40.

**extend_guide_sh** / **extend_novel_sh** — SH bands of the guide render on
the inactive frames and on the new frames, respectively. The inactive
frames look from where the guide was trained, so all 3 bands are supported
there. Uniform SH 1 washed out the colour context. The new frames are at
elevations the guide never saw, where the higher bands only extrapolate, so
1 keeps the base colour and its smooth view dependence. Setting the two
equal renders every frame alike.

**extend_overlap_before** / **extend_overlap_after** — how many of pass
2's first / last frames are reused as inactive frames. Each pass is 81
frames, so it paints 81 minus the overlap. Wan's temporal VAE encodes frame
0 alone and then every 4 frames. The mask therefore changes on a latent
edge only for these counts:
- **before:** the number of new frames is 4k+1, giving overlaps of 40, 44
  or 48;
- **after:** the overlap itself is 4k+1 (41, 45, 49), because there the
  inactive block starts the video.

Other values run, with a warning, with one latent straddling the boundary.

### Finishing

**run_upscale** — the SeedVR2 upscale (720x1280 → 1080x1920) before the
export. It also rescales the dataset's camera intrinsics in the same step.

**reupscale** — after the deliverable training, the pipeline:
1. renders the splat at 720x1280 from views spread evenly over ±70°
   elevation, as many views as there are training cameras;
2. upscales those renders with SeedVR2;
3. trains the splat again on them.

On one subject, band-limited sharpness rose 63 % (head 77 %) for ~2 dB of
fidelity to the real frames, 1.4 dB of which comes from retraining on
renders at all. Results are hit or miss, so neither quality profile turns
it on. The retraining keeps the alignment loop and body rig but drops the
normal maps. The first splat is kept as
`debug/reupscale/final_before_reupscale.ply`. It costs ~15 min of SeedVR2 on
a 4070 Ti plus a training.

**lighting_correction** (`prepass` / `off`) — takes the generator's
frame-to-frame relighting out of the frames before the deliverable
training. The passes are generated separately: exposure swings 10–23 % and
white balance 6–12 % across the final frames, and the splat's SH bands
absorb it. `prepass` does three things:
1. trains a 5k-iteration splat for geometry;
2. fits every frame's departure from the common lighting (27 numbers per
   frame);
3. divides that out of the frames that the COLMAP bundle and the final
   training get.

Measured on two subjects, the SH share on skin fell from 23 % to 15 % and
from 47 % to 31 %. The colour cast between passes went away and the fit
was unchanged. It costs ~10 s of training plus ~1 min of fitting.

**splat_labels** — segments every final frame with Sapiens2 (Goliath body
parts) and votes the classes onto the splats. `scene.ply` gains `seg_label`
(class id) and `seg_conf` (the winning class's share of the splat's
rendered weight); viewers ignore them. The same maps go into the COLMAP
bundle as a `labels/` sidecar. It costs ~20 s of GPU for 81 frames.
Measured: 96 % of rendered weight agrees with its splat's class, with a
median confidence of 0.99.

**extra_debug** — writes two COLMAP datasets that nothing else needs:
- `colmap_intermediate/`, what the first training is given, including the
  face supporting views;
- `colmap_preupscale/`, the frames before the upscale (only with the
  upscale on). This costs an extra RMBG pass.

`debug/` only goes into the result .zip when "Include debug/" is ticked.

### Outputs

**COLMAP dataset** (`colmap/`, always written) — `cameras.txt`,
`images.txt`, `points3D.txt` and `images/`, built from the final frames.

**Trained .ply** (`export_ply`, `ply/`) — the deliverable training on that
dataset: 30,000 iterations (roughly an hour of GPU), then four alignment
iterations (~5 min) that warp the frames onto the splat's own renders and
refit ([below](#the-deliverable-training-train_final_splat)).
Exported as `scene.ply` (the bare splat) and, with the body refit on,
`scene.glb` (splat, body, skeleton and cameras; b2cgltf SPEC.md).

**Rig splat** (`rig_splat`) — rigs the splat with b2crig so it can be
posed. `scene.glb` gains the cage (the refit MHR body plus garment, hair
and face layers) and the splat's binding to it (b2cgltf SPEC 5).
`rom_tour.clip.glb` is a range-of-motion clip for the viewer. Along the way
the splat is split at the finger joints and fine-tuned for about a minute,
so posed limbs don't throw splats outside the figure. The unrigged file
stays in `debug/rig/`. Needs the body refit.

## Video denoise (Wan 2.2 VACE)

`pipeline/steps/wan22_vace_denoise.py` (diffusers `WanVACEPipeline`, env `wan22`)
runs every Wan pass in `helical.yaml`: `reoutline_denoise` (480x832, 2|2 steps),
`denoise_pass1` (control = mesh drawing + skeleton), `denoise_pass2` (control =
confidence-gated render of the intermediate splat) and the two
`denoise_extension_*` passes (pass 2's settings). All declare `keep_loaded: true`
so one resident worker loads the experts once per run; `release_vram()` after each
job parks the weights in host RAM so the trainer finds an empty card.
`tests/test_workflows.py::TestTheDenoiseSettingsAreTheMeasuredOnes` pins the
workflow values below, because the step's own defaults (shift 8, `uni_pc`) are the
ones that measured worse.

### Weights: fp8 experts, unfused Lightning LoRA

- The experts are `silveroxides/Wan_2.2-fp8_scaled_hybrid`'s
  `wan2.2_fun_vace_{high,low}_noise_14B-fp8_scaled_original.safetensors`
  (17.6 GB each), loaded by `pipeline/wan_fp8.py` straight into torchao
  `Float8Tensor` params: no dequantize, no requantize. The rest of the pipeline
  (text encoder, VAE, scheduler) comes from `checkpoint`; diffusers never fetches
  the repo's bf16 `transformer/`/`transformer_2/` because both are passed in.
  ~47 GB per cold load.
- **The bf16 path (download bf16, fuse LoRA, `quantize_()`, `fused_cache_dir`) was
  deleted on purpose; do not restore it.** It cost 81 GB per load plus minutes of
  quantize work, and its on-disk cache never once saved successfully.
- ComfyUI-format -> diffusers keys: diffusers' own `convert_wan_transformer_to_diffusers`
  already covers VACE and renames each `X.scale_weight` with its `X.weight`. 7 keys
  are dropped (6 scales on BF16 layers + the `scaled_fp8` marker): 1338 -> the
  model's exact 1331. Per-tensor scale = torchao per-row scale with equal rows, so
  nothing is approximated.
- **Meta-device init is required**: normal init materialises the full bf16 14B
  before `load_state_dict(assign=True)` replaces it. Meta init leaves the
  non-persistent `rope.freqs_{cos,sin}` on meta, so `wan_fp8.py` rebuilds
  `WanRotaryPosEmbed`.
- Lightning: `lightx2v/Wan2.2-Lightning` T2V 4-step rank-64 LoRA (no VACE-specific
  one exists); works on the VACE backbone. Applied **unfused**
  (`_load_lora_unfused`): `fuse_lora` fails (`TypeError: can't multiply sequence by
  non-int of type 'float'`) because peft can only merge into a torchao subclass via
  an attached diffusers `TorchAoHfQuantizer`, and a model built from a state dict
  (or `quantize_()`) has none. Same root cause makes safetensors `save_pretrained`
  fail with "data pointer on an invalid python storage". Only a model quantized via
  `TorchAoConfig` at `from_pretrained` can fuse and save. Cost of unfused: one
  low-rank matmul per adapted Linear per step.
- Untaken idea: publish a diffusers-format fp8 checkpoint with the LoRA fused
  (quantize through `TorchAoConfig`, fuse, `safe_serialization=True`). None exists
  on the Hub (all fp8 quants are ComfyUI-format).

### Version pins

- `docker/Dockerfile`: torch 2.13.0 / torchvision 0.28.0, `venv_wan22` pins
  `torchao==0.18.0`, `diffusers>=0.38.0`. **Move the torchao pin in lockstep with
  `TORCH_VERSION`.** A torchao built for a newer torch dies on import
  (`cannot import name 'ScalingType'`), and since transformers imports
  `quantizer_torchao` unconditionally, an installed-but-broken torchao takes
  `import diffusers` (the whole step) down with it. An absent torchao is guarded; a
  broken one is not. torchao >= 0.16 is also what lets diffusers put a LoRA on
  quantized weights at all.
- The beta schedule needs diffusers' `set_timesteps(sigmas=...)` (present in 0.40,
  absent in 0.36); the wrapper refuses by signature rather than silently running
  linspace. Custom sigmas must be an ndarray, not a list.

### VRAM / offload

- Default `cpu_offload` = diffusers group offloading, unstreamed, 1 block per
  group. **`use_stream` must stay off**: with `Float8Tensor` weights the streamed
  path offloads nothing (the `.qdata`/`.scale` never leave the card; 23 GiB
  resident on the 5090) and the second forward raises `cannot pin
  'CUDAFloat8_e4m3fnType'`. Test any offload change with real `Float8Tensor`
  params and assert `weight.qdata.device`, not peak allocation. The step docstring
  has the measurements.
- `low_vram` (12 GB cards): VACE hints one at a time, chunked per-token work,
  mmap weight streaming, VAE tiling; transformer output bitwise equal.
- Resolution 720x1280 is the largest the model was trained for.

### Sampler: parity with the ComfyUI reference graph

The step's defaults reproduce the original ComfyUI graph
(`ComfyUI-Body2COLMAP/workflows/api/denoise.json`), which differs from the HF
scheduler config in several ways. Each is a param so the diffusers behaviour is
still sayable:

| param | default (= graph) | HF/diffusers behaviour |
|---|---|---|
| `sigma_schedule` | `beta` (comfy `beta_scheduler`, `_comfy_beta_sigmas`) | `linspace` |
| `sampler_shift` | 8.0 (Comfy's Wan default, no ModelSamplingSD3) | 3.0 |
| `solver_variant` | `bh1` | `bh2` |
| `solver_order` | 3 cap, per phase `min(3, steps-1)` | 2 flat |
| `handoff_reset` | on: each expert's sampler starts with empty history | one loop, history carried across |
| `reference_fit` | `crop` (comfy `common_upscale` center) | letterbox on white |

- Expert split is by count (`steps_high`/`steps_low`, 2|4), not by
  `boundary_ratio` 0.875, which at shift 8 would put 4 of 6 steps on the high expert.
- Traps: diffusers' bh1 is non-finite at a terminal sigma of 0, so bh1 uses
  `COMFY_TERMINAL_SIGMA` 0.001 as ComfyUI does; raising `solver_order` through the
  config resizes `model_outputs` but not `timestep_list` (IndexError), so the step
  sizes both.
- Remaining differences, left alone: fp8-scaled + live LoRA vs the graph's Q8 GGUF,
  RNG stream, integer timesteps.
- The workflow now overrides the sampler to `euler` on both experts of every pass
  (see below), so solver variant/order/hand-off are inert in production but keep
  the step's defaults honest.

### Pass 1: skeleton ink and the settled settings

Problem: the DWPose skeleton in pass 1's control was painted into the frames as
coloured sticks (worst: a red neck/shoulder "yoke" on back views), baked into the
intermediate splat and inherited by pass 2 and the final frames. Not an invocation
bug (checked against ComfyUI's `WanVaceToVideo`); a skeleton on a grey silhouette is
not the black-background pose map VACE learned.

Metric: `scripts/skeleton_leak.py --iou --by-hue <result>` reads
`debug/denoise_pass1_input` against `debug/colmap_intermediate` (units: 8-bit
chroma, ~15 visible, ~2 invisible, 0 clean; `--iou` = frame matte vs mesh
silhouette). `flow` = `debug/alignment/alignment.json` iteration-1 mean.

Measured (one subject, seed 0, 2|4 steps, beta):
- Strength per step (single-step halvings of `[1]*6`): step 2 (last high-noise step)
  is the lever (leak 16.6 -> 5.0, yoke 56 -> 10). Halving steps 5-6 costs silhouette
  and removes almost nothing. Along the strength axis leak and IoU trade tightly
  (Spearman 0.89).
- `[1,1,.5,.5,.5,.5]` at shift 2.5 euler: leak 1.74, but the yoke stays ~14.
- **Shift 5 alone removes all ink, yoke included** (leak -0.44 euler, -0.53 uni_pc).
- `uni_pc` on the high expert at shift 2.5: worst yoke ever (116). Likely UniPC's
  history carries the high steps' ink through the hand-off. At shift 5 the sampler
  stops mattering.
- `strength_layers` (8 VACE layers 0,5,...,35): dead axis. Shallow layers carry
  pose and ink together: deep-off paints the skeleton literally (leak 105),
  shallow-off ignores the control (IoU 0.57, flow 4.7, frontal figure in every
  view).
- `background: grid` behind the drawing: leak 1.7 -> 9.6, flow 0.99 -> 1.23. The
  render has no backdrop (`background: ""`).
- Cost of clean frames: flow +17-20 %, silhouette IoU -0.6..-1.4 pt. Pass 2 at a
  flat scale wins the flow back.

Shipped `denoise_pass1`: `sampler_shift: 5`, `strength: [1, 1, .5, .5, .5, .5]`,
`steps_high: 2`, `steps_low: 4`, `sampler_high/low: euler` (E2-equivalent; E4's
`uni_pc` high was later swapped to euler, equally clean). The re-outline pass uses
the same with 2|2 steps and `[1, 1, .5, .5]`. `splat_inactive_mask: true` on the
`+splat` render steps (face splat pixels marked real, VACE mask 0); every
reference run carried it. Unmeasured corners: the old taper
`[1,1,.75,.5,.25,0]` at shift 8, and `[1]*6` at shift 5 (would return the IoU the
taper costs). `sigma_schedule: simple` never run.

Noise floor: identical settings at seed 0 are not pixel-reproducible (sub-level
control differences become a new texture draw, frames 26-28 dB apart); structure
metrics reproduce. Head s1 sd ~1.2, body ~0.3; treat head deltas < 2.5 as noise.

### Pass 2: strength is a dial, the sampler axis is dead

Metric: `scripts/final_splat_quality.py <result>` (needs a local b2ctrain):
sharpness (s1) and PSNR of the final splat at its own cameras, plus a reconstructed
pass-2 control. A setting wins only when the final splat's head s1 AND PSNR rise
together; sharper frames with falling PSNR and rising flow is detail the fit
cannot keep.

Measured on pass 1 = E4 (base: flat 0.8, shift 2.5, euler, 2|4: splat head s1
29.5, 25.75 dB, flow 1.189):
- Strength 0.6 / 0.8 / 1.0: PSNR 24.77 / 25.75 / 26.29 dB, flow 1.389 / 1.189 /
  1.007, BA scale inflation 0.067 / 0.028 / 0.006; frame head s1 runs the other way
  (40.2 / 39.3 / 37.4). 0.6 invents (tile grids, ceiling lights, eyeshadow). A
  taper to 0 on the last step gives the sharpest frames and the worst splat (flow
  1.376).
- Shift 8 (the step default): splat head s1 -3.2 for +0.09 dB. `uni_pc` high: -2.8,
  -0.6 dB, flow +8 %, hallucinated tile grid; a splat render has no detail to hug.
  Shift 5 and a 3|3 split (each expert on its trained t-range): nothing gained.
- No pass-2 setting sharpened the head: the final head is ~0.75 of the frames' head
  sharpness and pass 2's head never exceeds its control's. Head detail is decided
  upstream of pass 2. (Masking the face cap out of pass 2's VACE mask was tried and
  backed out: detrimental to the splat.)

Shipped `denoise_pass2` and both extension passes: `sampler_shift: 2.5`,
`strength: [0.8]*6` (the texture compromise; `[1]*6` is the fidelity end, +0.5 dB,
flow -15 %), `sampler_high/low: euler`, 2|4. `sampler_low: euler` on pass 2 is
unmeasured. No taper: a scale-0 step on a splat render is a step spent inventing.

### Anchor drift (the body painted 5-10 px high)

Every pass-1 run paints the body 5-10 px higher than both the photograph and the
drawing; the injected anchor frames (VACE mask 0) pull neighbours back. The
transition is asymmetric because of the Wan VAE's causal chunking (frame 1 is its
own latent, so the step at frames 2-5 is sharp; frames 78-81 share a chunk with the
anchor and the descent smears over ~10 frames). Geometry was ruled out: at the
anchor camera the mesh silhouette matches the warped photo row by row through the
torso (1-3 px); the mesh scalp is 7 px under the hair and the mesh feet 22 px above
platform soles (why `re_outline` exists). The ComfyUI reference output does not
drift; cause unresolved (quant or control content). Downstream consequence: frames
near the photo are displaced re-drawings, not the photograph (see photo priority
below).

### Loose garments billow; the trainer cannot fix it

A loose coat in pass 1 is cloth in motion, not a rigid shape: strict visual hull of
81 mattes 62 l vs 61 l for the naked mesh, 7.5 % of each frame's subject outside the
hull (rigid subject: ~1 %), excess a smooth function of azimuth. Pass 1's control
has no channel for the garment (faint mesh fill + skeleton), and Wan's prior for a
mid-stride pose is walking. Pass 2 fills the gate's torn hem with moving cloth
again. Trainer-side consensus (IRLS down-weighting of disagreeing pixels, or a
hard envelope silhouette) only trims wisps: a splat cannot manufacture agreement
the frames do not contain. Any fix has to be upstream, in what pass 1 is told.
Unbuilt candidate: a rigid clothed envelope (naked mesh inflated per part by the
photo matte's excess, elliptical depth ratio K~0.25) drawn as pass 1's fill plus a
contour line through `render`'s `outline_masks`.

### Dead end: in-loop 3D sync / latent splatting

- Latent Gaussians carrying Wan latents (Splatent-style) fail on the Wan VAE: a
  latent frame is a steady-state code plus a chunk-specific temporal code (95 % of
  energy in the lowest radial band, the only band that is view-consistent); the
  encoder is equivariant only to whole-latent-pixel shifts; a Gaussian field on
  per-view codes fits training views and fails every held-out view (band-0 rel >
  1). Making latents into geometry needs a trained decoder (finetuning).
- Pixel-path x0 synchronisation (decode x0 at steps 2-4, fit a splat, re-render,
  replace x0's low band) was built, run A/B on a pod, and removed: it stripped
  6-10x of the frames' fine detail at every stage and the final splat rendered with
  9x less detail. Its fit-PSNR gain was mostly blur. x0 is already consistent
  enough at those steps that a consensus render is a downgrade. Any revival must
  be judged by the final splat's detail, never by fit PSNR.
- Trap kept from that work: b2ctrain `--lr-mean 0` writes NaN positions; freeze
  with 1e-12.

## Splat training and cameras

### Trainer and rasteriser

- Both trainings and every splat render are b2ctrain. The `brush` step id, `steps/brush.py`, and the `brush` / `brush-splat-render` binary names are the CLI contract only. Measurements credited to "brush" below were taken on the old Erant/brush fork. Re-check a brush-era number before you lean on it. The polish (below) is the case where the result did not carry over.
- The trainer refuses repeated flags (clap). Every invocation goes through `command()` in `steps/brush.py`. Overrides must come in through the param plumbing, never by appending to argv.
- The trainer scans a dataset directory recursively for `cameras.txt` and trains on the model with the most registered images. Never leave a second model or scratch output inside a dataset tree. Doing so once gave PSNR 4.45 with exit 0. A stray `masks/` sidecar also changes the alpha mode from transparent to masked.
- body2colmap's `render_many` drives the rasteriser. It always runs the binary on `--background 0,0,0` and composites `bg_color` in Python, which costs one extra 8-bit round trip (at most 1/255). `steps/splat.py::_rasterize` is a thin wrapper. It keeps only what belongs to this project: where crash reports go (`on_fault`), log relay (`on_output`), rendering an existing ply (`ply_path`), and frame naming (`frame_00001_.png`).

### The deliverable training (`train_final_splat`)

**Why the splat is softer than its frames.** The fit causes the softness, not the upscaler. Face Laplacian variance was 52.6 for the splat against 400-770 for the SeedVR2 frames it was trained on. Dense flow from each frame to the splat's render shows only ~0.5 px of whole-region shift, so the poses are fine. Per-pixel flow is 1.7-3.6 px mean (p90 up to 8 px): each generated view puts skin, hair and finger texture in a slightly different place, and the photometric loss averages them into a blur. Retraining on unsharp-masked renders of the splat itself (view-consistent by construction) took face sharpness from 91 to 228. Consistency between views is the limit, not capacity.

Metric: band-limited sharpness `s1` (Laplacian variance after a σ=1 blur), on Sapiens2 part crops, checked at novel (in-between) cameras as well. Raw Laplacian variance is dominated by resampling grain and cannot be compared across anything that resamples.

**Alignment loop** (`align_iters: 4`, the step default). Each iteration renders the splat at the training cameras, computes DIS optical flow from each original frame to its render, smooths the flow (σ 6 px), caps it at 6 px, masks it to alpha, Lanczos-warps the original, and warm-starts training with growth off for `align_steps` 3000.
- Measured face s1 21.1 → 22.3 → 22.9 → 23.4 → 23.8 over four iterations; iterations 5 and 6 add about +0.2 each. Novel views gain in the same ratio (~0.91). Fidelity rises with sharpness (27.64 → 28.41 dB), which is the signature of recovered detail. If sharpness rises while fidelity falls, something is wrong.
- Invariant: every iteration warps the **pristine originals**, never a warp of a warp (that drifts without bound). The accumulated gain lives in the splat. Frames aligned to the finished render carry about half of it (cold start on them: 22.5 vs 21.1).
- Interpolation must be Lanczos (`pipeline/align.py`, `INTER_LANCZOS4`). One 0.5 px bilinear resample takes a face crop from 738 to 134; Lanczos takes it to ~480. An early "alignment hurts" result was caused entirely by bilinear resampling.
- `align_steps` 1000 holds the fixed point exactly (47 s → 15 s per iteration), but every iteration that is still climbing was measured at 3000.
- Raising `align_flow_cap` from 6 to 12 alone changes nothing. The schedule worth trying is σ [6,6,3,3] with cap [6,6,12,12]. Hands keep double contours whatever the settings: the views disagree about hand *pose*, which no image warp fixes. Per-view loss weighting for this was tried and was negative.
- `align_backend: auto` uses b2ctrain's in-process loop when `--help` lists `--align-iters`: one invocation, about 2m50 for the step against 5m30 for the pipeline loop. `pipeline` (render + `align.py` + one re-invocation per iteration) still works and is the reference the settings were tuned on. The two agree on PSNR (32.24 vs 32.29 dB) and on trajectory shape. **Band-limited sharpness has not been measured on the in-trainer loop.**
- `export_evidence` after alignment measures against the warped set, which is what the splat was fitted to.

**Dense growth is off** (`growth_*` empty = trainer defaults). Growth 0.0012 / select 0.4 / stop 24000 gives face s1 23.9 alone, the same as alignment alone (23.8), at 1.68M splats / 424 MB against 356k / 84 MB. The two stack (26.3) when the flow is measured against a converged render. An early "does not stack" result came from aligning against a blurry cold-start render. Turning growth on buys +2.5 s1 for 4.6× the asset size.

**Normal supervision is on** (0.05 from step 5000, `normals_growth_grad_threshold` 0.0035, `random_background: true`). It was off for a while because normals cost −18 % s1 and −1.8 dB on the original measurement. Recomputing the normals from aligned frames did not help (22.0 vs 21.8), so the harm came from the loss term, not from bad normal maps. It came back for **geometry**: arm splats more than 2 cm off the refit body went from 20 % to 7 % and from 22 % to 10 % on two subjects, at ~3 % sharpness with a random background and an equal splat count. The three settings go together. The normal loss is a second gradient source, so growth at 0.0025 gives 553k splats against 389k; 0.0035 brings it back to 396k. Against a black background, normals make the dark silhouette streaks worse. In-trainer alignment refits run without the normal loss.

**No polish** (`polish_steps: 0` on both trainings). On brush, a 9000-step growth-off warm start was the biggest intermediate-splat lever (face 143 → 161; 39000 cold steps gave a third of that, so the gain came from the restart, not the extra steps). Re-measured on b2ctrain it buys nothing. The machinery stays.

**No supporting views from stage 1.** The face-cap renders belong to the circular bootstrap orbit at pre-upscale resolution. Wiring `scene.support_views.*` into `train_final_splat` would not fail, which is why the omission is deliberate and called out in the workflow.

### The intermediate training (`train_splat`)

This splat exists to drive the helical re-render. It was measured at the 81 novel helical cameras that the second denoise sees.
- Normals **on**: turning them off lowered s1 in every part and dropped the specular top's confidence (upper-body cull 8 % → 42 %). The final-training result does not transfer here.
- `match_alpha_weight: 0.5` (trainer default 0.1): 17 % fewer dark wedges at no sharpness cost, IoU −0.003. Dark wedges are dark splats in concave gaps (arm/torso, fingers) that a flat orbit never sees into. They open up at the ±30° the helix travels to.
- `align_iters: 0`. At stage 1 the warped arm (145/22.6) lost to the unwarped warm-start control (149/22.9). The applied flow was only 1.0 px, because one 720p VACE pass is far more consistent than upscaled frames. Alignment inline during growth gave 136/21.5.
- `export_evidence: true` is required because the re-render's gate reads the `ev_*` ply properties. Without them the gate warns and falls back to plain alpha.
- Measured and rejected here: dense growth (sharpest plain render, but 3× the time and the gate rejects a quarter of it), SH degree 1 (110/20.4 vs 132/21.8), `--evidence-prune-inmask 0.3` (prunes detail splats; face 132 → 117), training on a 0.5 grey background (halves the wedges but costs −8 % s1 everywhere). The real fix for the wedges is vertical parallax in the pass-1 path.

### The confidence-gated re-render (`render_subject` → `resplat_foreground_masks` → `mask_splat_fringes`)

- The trainer writes per-Gaussian evidence into the ply: `ev_w_in`, `ev_w_all`, `ev_err`, `ev_views`, `ev_dir_0..2`. The cost is seconds. `render_splat` with `confidence: true` forms a per-pixel confidence C and a gate `g = smoothstep(gate_lo, gate_hi, C)`. The output contract changes: alpha is the gate (not opacity), RGB is composited over `cull_color`, and `bg_color` is ignored.
- `conf_args: --conf-tau 0.3 --conf-angle-margin 45`. With the in-mask evidence fix (masked support views had counted outside-mask as background; loss weights scaled `w_in` but not `w_all`), these take the culled subject from 30 % to 6.4 % (face 69 → 8 %, specular top 67 → 3 %) and IoU from 0.70 to 0.92. Tau 0.08 is too tight for view-dependent materials. Margin 30 culled the face at the helix's ±30° extremes because the cap views sit in a narrow disc. Margin 15 cut patches out of clothing.
- `gate_lo`/`gate_hi` stay at 0.45/0.65 and are not the lever: 0.55/0.75 culls 43 % of the subject to remove a quarter of the fringe, and no lo/hi pair separates fringe from subject.
- `cull_color` is 0.5 grey, the same colour the frames are composited on. It was black for two days so that culled pixels would read as holes. In practice they showed up as black speckles on the cap rim and the specular top, which the second denoise faithfully kept. With one colour, partial coverage fades toward the final value and leaves no dark halo. (b2ctrain's effective `ev_views` count since removed most of those culls anyway.)
- The matte comes from **rmbg** over the re-render, not the gate's alpha. The gate says what the training constrained; rmbg says what is subject in this frame. Cutting with the gate would cut its holes out of the frame. `mask_splat` runs `mode: composite` over 0.5 grey. The legacy `threshold` mode (alpha ≥ 239/255, dilate, bilateral) must never run on a confidence render because it re-composites and smears the soft gate.
- `mask_splat` is also where `dataset.masks` changes meaning, from per-pixel alpha to the per-frame VACE flag (all 1.0 = regenerate). `reinject_anchor` must run **after** it, or it would overwrite the alpha and turn the masking into a no-op.
- Evidence view counts scale with training resolution. The defaults were tuned at a 720-1080 px short side.

### Supporting views and premultiplied renders

- `select_support_views` (`steps/anchor_stub.py`) un-premultiplies cap renders and refuses any render that is more than 8/255 bright where it is fully transparent (`_check_premultiplied`). **Never feed it a confidence render.** `tests/test_workflows.py::test_no_confidence_render_feeds_a_step_that_needs_black` guards the wiring.
- `min_alpha` 0.15 (was 1/255) plus a 5×5 grey closing on the alpha. The un-premultiplied outline is noisy (|rgb − median5| is 16.5 where alpha < 0.05), and there is a one-Gaussian-per-pixel checkerboard inside the mask. Measured neutral on the test subject; kept as hygiene against floaters on other subjects.
- `render`'s `*+skeleton+splat` modes composite a splat layer through body2colmap's `render_composite(splat_layer=...)`. The angular cull comes from body2colmap's shell measurement: clean to 30°, rim flares by 45°, mostly edge by 60°. `splat_max_angle_deg: 60` is passed **explicitly**. Leave it out and body2colmap's 45 is inherited silently. 60 is chosen because a flared rim feeds two denoise passes that can rewrite it, while a frame with no face cannot be recovered.

### Camera refinement (`refine_cameras`, `refine_cameras_final`)

- **Why.** Generated orbit poses are exactly ideal (height std 0, step 4.500°), so nothing ever observed the cameras. Refining against the frames gave **+0.63 PSNR / +0.0045 SSIM** on held-out views (three seeds per condition, baseline seed spread ±0.003; ahead at every checkpoint from 2.5k iterations on). The setting is on by default. Measured on one dataset only; re-measure with 3 seeds before assuming it holds elsewhere, because single runs are not trustworthy.
- **Where.** It runs twice, because each training dataset has its own generated poses. The stage-2 run comes before anything reads the poses: the face cap is rebuilt from it, then `train_splat`. The stage-6 run comes before `export_colmap`/`train_final_splat`. Supporting views are never part of the solve: they are renders made from the poses being corrected.
- **Recipe** (the step's docstring carries the full traps). Foreground-masked (alpha > 127) ALIKED_N32 + ALIKED_LIGHTGLUE exhaustive matching. Triangulate against the given poses, then 3 × (BA → retriangulate). Every intrinsic frozen. Sim(3) on camera centres back onto the given ones. Then the subject gauge.
- Foreground-only rather than all features: the 0.089 PSNR gap is within RANSAC nondeterminism (0.075). A backdrop that moves, or is generated inconsistently, poisons an unmasked solve silently.
- **Trap: BA scale inflation.** COLMAP's gauge hangs scale off one baseline, inflating the model by 15-26 % with no warning. The Sim(3) removes it exactly (0.006 % radius drift). The scale gate (`max_scale_drift`) is 1 %, not 0.1 %, because honest non-similar corrections move the mean radius by roughly the square of their size.
- **Trap: floating intrinsics** produce fx 1105 / fy 838 (24 % aspect distortion) for 0.017 px less reprojection error. Keep them frozen.
- **Trap: ring valleys.** For an inward-looking ring, "subject lower/forward/bigger" equals a small turn of each camera about its own centre, which a centre-fitted Sim(3) cannot see. One run pitched every camera +0.94° (19 px), which put the frames' face 19-21 px below the mesh and cap and made the splat train two faces. Fix: with `mesh_world` wired, project the mesh through the given cameras, triangulate with the refined ones, and move the cameras by the Umeyama similarity that puts it back on the mesh (`_align_to_structure`). One bundle had a 3.6 cm subject offset at 1 mm residual; the ring's radial jitter went from 33 mm to 9 mm. Without a mesh, `_remove_common_mode` takes out the mean rotation (gate 3°). `max_subject_shift` 0.1 of the scene radius.
- **Trap: COLMAP image-id order.** Writing the input model before the extractor aborted `point_triangulator` (SIGABRT) whenever thread completion order differed. That happened on 9 of 11 pod runs in the final refinement, which silently kept the given poses. The input model is now written after extraction, in database order (`_database_image_order`).
- Failing any check (scale, centre shift 3 %, common mode, subject shift) means `on_check_failure: keep_given`: an ERROR is logged and the given poses are published. Never publish poses that failed a check.
- **Ceiling.** Median reprojection bottoms out near 1.4 px (a photographic capture sits near 0.5) because the generated frames disagree with each other. Do this once; don't tune it, because the headroom is in the frames. "High reprojection" cannot tell bad poses from inconsistent frames, so refinement is a setting and not auto-detected.
- Image: COLMAP is built with CUDA, so ALIKED/LightGlue run on the ONNX CUDA provider (features + matching 209 s → 27 s; BA stays on CPU Ceres). COLMAP's `FETCH_ONNX` would fetch a cuda12 runtime that does not load against CUDA 13, so the build overrides it. The runtime stage needs `libglew2.2`, the builder needs `openimageio-tools`, and the ONNX weights are prefetched (`colmap_onnx` in `models.py`).

### Rebuild what was made from a moved camera

Order matters more than anything specific to the face: anything **built** from a camera that refinement moves is stale afterwards.
- The face splat is **rebuilt** (`face_splat_refined`), not rigidly carried. The carry (`rebase_cameras`, deleted) took the anchor's BA slide along its viewing ray into the splat's depth. That depth came from the mesh, which does not move, so the cap sat 50 mm inside the head (Janus). The rebuild hangs the photograph's rays on the given camera moved by T = refined ∘ given⁻¹, not on the refined camera's own `look_at`-turned rays. Hanging them on those rays put the cap 17 px / 28 mm high.
- `render_subject` moves its **whole helix** by the anchor's T (`given_anchor_camera`, `_carry_anchor_refinement` in `steps/splat.py`). The refined anchor sits 41-68 mm / 0.9-1.6° off the origin, which is 8-10 px at the subject. Rendering from the origin hands pass 2 a seam at the reinjected photograph. Moving only the anchor frame would kink the video. The step republishes `anchor_position`. `refine_cameras_final` deliberately publishes no `anchor_position`.

### Dead ends — do not repeat

- **Stage-1 body shells** (`pointmap_elevation_views`, ±20° renders of per-frame pointmap shells as support views). They were built against chalky zero-parallax streaks that no longer reproduce with the face cap and evidence gate. Re-measured, they were a net loss: less s1 in every part, 735k vs 284k splats, twice the training time. Removed from the workflow; the code is unwired.
- **Per-view diffusion refinement without registration** (Flux on training frames, renders, or close-up support views): always worse (support views 40-46 vs 91 without). Each view's denoise places fingers and lashes independently. Refined hands disagree with the splat by 6-7 px of flow, against 2.9 px for plain SeedVR2 frames.
- **Folding alignment into the loss** (learned warps, shift-tolerant losses): the published gain is ~0.1 dB, and gradient-based alignment has a ~±2 px capture basin against drift up to 8 px. Pre-warping the frames, which b2ctrain's loop does, is the form that works.
- **Phase-preserving-noise diffusion pass after alignment:** +6 s1 for −2 dB per iteration and runs away when iterated (visibly hallucinated faces). It works only on aligned views and needs a fidelity guard. Not shipped.
- **Inline alignment, longer cold runs instead of a restart, evidence-in-mask pruning, SH degree 1** — see the intermediate-training bullets.

## Body, face and silhouette

### Re-outline: the silhouette from a splat, not the mesh

Setting `re_outline` (default on). The first denoise conditions on a flat silhouette under the DWPose skeleton and the face splat. The SAM-3D-Body mesh is a naked body, so its outline stops short wherever hair, a coat or a platform sole extends past it (measured: scalp 7 px low, feet 22 px high). Eight steps between `reinject_anchor_initial` and `dump_denoise_input`, all gated on `re_outline`, replace that outline:

- `reoutline_downscale` (`resize_batch`, 720x1280 -> 480x832). `reoutline_denoise` then runs pass 1's control at 480x832, 2 high / 2 low steps, strength `[1, 1, 0.5, 0.5]`. Only the shape is kept; the texture is thrown away.
- `reoutline_matte` (rmbg). `reoutline_upscale` brings frames and mattes back to the render size. `reoutline_train_splat` uses the silhouette recipe only: `match_alpha_weight` 0.5, no align/polish, no support views, weights, normals, rig or hollow loss (the hollow loss would penalise exactly the hair and clothing the branch exists for), and `total_steps` 15000. Then `render_reoutline_splat` (alpha only, on `dataset.cameras` verbatim) -> `scene.outline_masks` -> `render_reoutlined_views` -> `reinject_anchor_reoutlined`.
- Why a splat and not the per-frame mattes: each 480p frame has its own idea of the hem and hair, so per-frame mattes jitter. A splat fitted to all 81 mattes outvotes single-frame bulges and keeps what they agree on.
- The batch is resized in b2crunner on purpose. diffusers' `WanVACEPipeline.preprocess_conditions` fits by area and floors to /16, so a 720x1280 batch asked for 480x832 runs at 464x832 with the reference letterboxed. The 0.667/0.65 anisotropy cancels because the same plain resize is inverted.
- `outline_mask_clean_px: 9` (opening then closing before the blur) removes horizontal "hairs" at the silhouette. These are low-opacity discs straddling the edge, drawn as streaks because every ray on a single-elevation ring is horizontal. Thin structure went 0.78% -> 0.00% and IoU against the mattes 0.9425 -> 0.9489. Dead ends: a trainer scale-ratio cap (8/4/2) and an export in-mask prune (0.8) both left the hairs, and the prune shrank the body. Raising `outline_mask_threshold` does nothing.
- Two fill strengths: `outline_strength` 6.25 (mesh outline, kept faint because it is wrong off the body) and `reoutlined_strength` 20 (`requires: re_outline`, drawn darker because it is the subject's agreed shape).
- No `refine_cameras` for this splat (the only brush training without one; tests/test_workflows.py exempts it by name). `dataset.cameras` must stay the ideal orbit. A uniform denoiser raise or offset survives into the outline anyway, because refinement removes the common mode. If the outline sits visibly high, shift the coverage in the drawing rather than correcting the poses.
- The Dockerfile asserts body2colmap has `render_outline(mask=...)`. An older body2colmap silently ignores the mask and draws the mesh outline.
- Reading a run: `debug/reoutline/frame_*.png` / `matte_*.png`, `debug/reoutline_splat.ply`, `debug/denoise_pass1_input/`.

### Face cap priority (`face_priority_weights`)

The face cap (renders of the photo-derived face splat) is the only face source that carries the photograph. At equal volume the denoised frames average it away. b2ctrain's `weights/` sidecar holds one greyscale map per view, multiplied into that view's loss. It reaches L1/SSIM, alpha-match, normals and the evidence pass, so a silenced region is not counted as evidence by the confidence gate. `face_priority` (gated on `face_splat`, for `train_splat` / `export_colmap_intermediate` only) writes

    weight = 1 - strength * g(theta) * feather(coverage)

- `strength` 0.9 (1.0 would stop the frames carving the silhouette there).
- `g(theta)`: 1 within `cap_radius_deg` 30 of the anchor view of the splat centre, fading over `fade_deg` 15. The cap is a 2.5-D shell, so beyond that the frames must keep their say. The anchor camera is read live from `dataset.cameras` because `refine_cameras` moves it.
- `feather_px` 4, no dilation.
- The coverage is rendered with the head's occlusion (`mesh_world`, `cull_margin` 15 mm). Without it, the far cheek showing through the temple at 40 degrees silenced a ring the cap never paints and left a void under the chin. `render_face_support_views` uses the same cull (`cull_mesh`) over a 60-degree cap. The refined cap is built with `depth_prior: mesh_surface`.
- `face_support_views.min_path_angle_deg` is 0, not 5: once a frame is silenced over the face, a cap view on the path is the only face evidence at that angle.
- `train_final_splat` takes no face weights (it has no supporting views).
- Dead end: masking the face cap out of pass 2's VACE mask (so pass 2 reproduces the splat's face) was detrimental to the trained splat. The reason was never established. Judge any retry on the trained splat, not on the re-render.

### Photo priority (off by default)

Setting `photo_priority` defaults to 0.0, and both `photo_priority` (intermediate) and `photo_priority_final` run with `copies: 0`. The steps are ungated, so with strength 0 they pass the face cap's weights through (all ones without the face branch) and publish empty support lists. The training is then identical to a run without them.

- Mechanism, kept for reference. The body mesh is rasterised per view, and each surface point is tested against the anchor camera: in frame, not occluded (`occlusion_margin` 15 mm), facing ramp `facing_lo` 0.2 -> `facing_hi` 0.5. The result is extended `extend_px` 24 past the body, clipped to the matte, feathered 4 px and windowed 45 + 45 degrees. Weight is `1 - strength * confidence`, multiplied into the face cap's weights (`scene.priority.weights`). `copies` adds masked copies of the anchor frame as `b_*` supporting views through `merge_support_views`.
- Why it once won: on a pod export, PSNR at the photograph's view went 17.8 -> 23.3-23.6 dB and sharpness x2.8. The copies did nearly all of it, the fade was a fidelity dial, and matte-masked copies beat confidence-masked ones.
- Why it is off: (1) the copies build a thin subsurface layer behind the front (2-2.5x see-through near the photo, +26% splats on the final) and store the photo detail in view-dependent SH, so it is gone by about 20 degrees. (2) The frame at `anchor_frame_index` is often a repaint, not the photograph (17-20 dB off `anchor.png` on recent runs, pod and local alike), so the copies amplify a repaint. The anchor frame stays a normal training view.
- Gotcha that still holds: only `anchor_frame_index` is ever treated as the photograph. The helix also ends on the anchor pose, but that last frame comes back as a repaint (14.5 dB off), so `anchor_tolerance_pct` (a pose-based rule) is 0. `copies_erode_px` 4 exists because the photo's anti-aliased edge made a front-only light halo.

### Body refit to the intermediate splat

`refit_body` (default on, also needs `export_ply`). The SAM-3D-Body fit is the run's frame of reference, but the trained splat drifts from it (camera refinement, limbs a few degrees off, torso 2-5 cm out). Everything downstream (the final hollow loss, both rigs, the head fit, the deliverable's skeleton) uses the refit body.

- `splat_surface` (`b2ctrain probe --depth`) takes the depth where accumulated alpha reaches `tau` 0.5, unprojected via the OpenGL c2w poses. It drops silhouette pixels (`edge_jump` 2 cm), uses stride 8, and caps at 300k points. Splat centres are deliberately not used, because they sit at every depth of a soft surface.
- `refit_body_to_splat` (sam3dbody env) runs the MHR forward in three stages: root + per-joint scales, then pose, then shape. Point-to-plane, one-sided: a point inside the mesh costs in full, and outside, the first `clothing_allowance` 1 cm is free and the rest weighs 0.1. A coverage term stops limbs shrinking. There is an L2 pull to SAM's fit, hands hardest and excluded. Root motion goes into MHR's `global_rot`/`global_trans` so `pose_params` replay the mesh exactly (checked < 1 mm).
- Measured: median surface-to-body 1.33 -> 0.73 cm, and points more than 1 cm inside 22.5% -> 7.6% (arms). The fit is not prior-limited: priors 10x weaker reach the same residual. The remainder is soft-surface thickness plus clothing. With the refit body, `train_final_splat` runs `hollow_margin: 0.03` (was 0.05).
- Nothing between the refit and the final training reads the body. The second `refine_cameras` keeps the gauge, so the refit frame carries through to the end.

### Per-view body rig (double limbs)

The generated frames move limbs by centimetres between orbit segments: forearms and hands move 7-8 px per 4.5-degree step, against 2-3 px for torso and legs. A canonical splat averages this into a double limb. `build_body_rig` (pipeline/body_rig.py) makes a rig from the body: a vertex subsample (`vertex_stride` 4), MHR skinning, the joint tree and the active joints. The trainer (`--body-rig`) learns one small rotation per view and active joint and exports the canonical model.

- Active joints are those skinned to at least `min_subtree` 30 vertices with a subtree no larger than `max_subtree_fraction` 0.5 of the body. That covers the limbs, hands, neck and head but not the root chain: per-view root rotations destroyed sharpness everywhere. Trainer params: `body_rig_start_iter` 1000, `_smooth` 0.05, `_zero` 0.02, `_lr` 0.002.
- Final training (refit body): double limb gone, hands +24%, body +6%, face +9%.
- Intermediate (`body_rig_intermediate`, default on, `build_body_rig_initial` on the initial body). Limb wobble here is about 6x the torso's, and it gets baked into pass 2's re-render. Novel-view sharpness: hands 84 -> 107, body 51 -> 55, head 43 -> 47, no time cost. The initial body works as well as a refit body would. The raw-to-world similarity is recovered from `mesh_raw` vs `mesh_world`, and the step refuses if they are not one rigid placement.
- Only the real frames are rigged. Giving the face supporting views rotations too erased the whole head gain. brush.py writes the rig for `image_names` only, and tests/test_body_rig.py pins this.
- Canonical PSNR against the training frames drops by construction (the limbs sit at the mean pose), so it is not the metric. Use novel-view sharpness.

### Face per view and the anchor's eyes

`face_refine` (default on; needs `refit_body`). It runs after the final camera refinement and before `export_colmap`, so the delivered dataset carries the pasted eyes. The frames' eyes are a per-frame invention (half shut, sideways, a blue smear), and the splat averages them into a slit. The anchor photo is the only consistent source.

- `detect_face_views`: MediaPipe on an upright 1.8x crop around the projected refit head. There is no detector, because a whole-frame detector missed the 80 px face in 63 of 81 frames. About 35 of 81 frames get landmarks (those within ~85 degrees of facing).
- `fit_head_per_view` (sam3dbody): one batched Adam run over the six neck/head rotations plus 72 expression coefficients per view. Huber loss at 4 px, features only. RMS error 6.4 -> 2.1 px. `pose_prior` 50 is needed: without it the head counter-rotates into a lateral shift of up to 61 mm. The anchor joins the batch through a PnP camera so its lids are fitted too. Views beyond `max_facing_deg` 85 are dropped.
- `paste_eyes`: a 15.5 mm icosphere at each eye joint (the model's eye surface fits r 15.8-16.4), textured from the anchor through its fitted lids. The eye follows its lid ring rigidly, uses the MHR-topology lid rings rather than per-run landmark snaps, is clipped in 3D to the lid cylinder (a 2D clip leaks at grazing angles), and is depth-tested at 1 mm against the head with back-face culling off. An eye is skipped if its visible fraction is < 0.5 (far eye past the nose) or > 1.05 (inconsistent fit, e.g. pasted on a cheek), or if its lid area has no dark pixels (a hand in front). A subject whose eyes cannot be modelled is left untouched rather than failing the run. Measured: eye MAE drops on all four subjects, face and hair sharpness stay within the metric's resolution, and the face's swim drops 3x.
- `build_face_rig` (`face_rig`, default off) writes the fitted head deltas into a v3 rig (`B2CRIG3`; brush.py falls back to v2 if the trainer lacks it), face core only (`face_motion_cm` 0.5, fade 3 cm), with `gap` 4 / `hold` 3. It is off because the 60-85 degree fits carry wrong MediaPipe landmarks and smear the profile view. The pasted eyes do not need it. Whole-head deltas also blurred hair 2-5%. Seeding the rig's head rotations from the fit sharpened nothing.
- Open: no loss boost on the eyes, gaze fixed to the anchor's, closed eyes not detected, and a dark hand in front of the face still gets an eye.

### MHR body record: frames and conventions

- `world = scale * raw @ R.T + t` (`world_from_raw`), where raw is SAM-3D-Body's OpenCV-flipped frame. `refit_body_to_splat`, `fit_head_to_face` and `sam3d_body` publish positions flipped (`* diag(1,-1,-1)`) but `global_rots` in MHR's own unflipped frame, which is SAM-3D-Body's convention. A world joint orientation consistent with the world joints is therefore `R @ FLIP @ rots`, not `R @ rots`. `ply_meta.body_comments` writes that (record version 2, test `test_header_rotations_agree_with_the_header_joints`). Version-1 headers hold `R @ rots`, and their rotations are wrong relative to their joints.
- The deliverable's body now lives in `ply/scene.glb` (`export_subject`, b2cgltf SPEC 4.3: skinned mesh plus `B2C_mhr` pose params, model and hash). `scene.ply` is the bare splat. The `comment b2c.mhr.<key>` header record (`ply_meta.parse_body_comments`) is still read for older runs by `tools/export_glb.py`, `export_mhr_subject` and `recover_body`. `SplatScene.to_ply` drops header comments, so never re-save a recorded .ply through body2colmap.
