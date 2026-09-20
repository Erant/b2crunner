# Latent splatting as in-loop consistency guidance for the Wan 2.2 VACE denoise

Research note, 2026-09-19. Sections 1–6 were written before any experiment; section 7 has the local measurements (E0–E2, run the same day) and what they settle. The question: without
finetuning anything, can the denoiser be told — while it is denoising —
how well its 81 views agree, using a 3D representation that lives in
the Wan 2.1 VAE's latent space (Splatent's idea), so that pass 1 comes
out spatially consistent instead of being made consistent afterwards by
the splat?

The short answer, argued below: **latent splatting in the strict sense
(Gaussians carrying 16-channel Wan latents, rasterised straight into
the denoiser's latent) is fighting two properties of the Wan VAE that
the Splatent authors did not have to face**, its temporal compression
(a latent frame is not a view) and its non-equivariance (the same for
every VAE, and the reason Splatent's own latent Gaussians come out
blurry). Both are measurable on this machine with the VAE alone before
a pod is booked. The mechanism that is worth building first does not
need latent Gaussians at all: **synchronise the x0 predictions through
a 3D canonical space at the three or four steps that decide structure,
and feed the consistent render back through VACE's own control branch
with a mask that says where the views disagreed.** Latent Gaussians
are then a speed optimisation of that loop, to be adopted only if
experiments E0–E2 say the latent is consistent enough to be a target.

## 1. Where we are

`helical.yaml` is already a 3D-consistency loop, but an *outer* one:
pass 1 (6 steps, Lightning LoRA, uni_pc/shift 5, strength
`[1,1,.5,.5,.5,.5]`) → `train_splat` → confidence-gated re-render →
pass 2 at 0.8 conditioning on that render through VACE. The splat is
the canonical space, the confidence gate is the view-agreement measure,
and VACE's control video + mask are the feedback channel. It runs once
per pass. The question here is what happens if the same loop closes
*inside* the sampler, between denoising steps.

Facts about the sampler that shape everything below
(`pipeline/steps/wan22_vace_denoise.py`):

- **Six steps, front-loaded.** The timesteps are ~1000, 988 (high-noise
  expert) then 955, 889, 753, 448 (low-noise expert), then straight to
  0. The x0 prediction at σ≈1 is a blur; structure is decided over
  steps 3–5 (σ .955→.448); fine detail is decided almost entirely by
  the last step, whose x0 *is* the output. So there are at most three
  useful synchronisation points, and the last step must be left alone
  or the result will be the projection's blur.
- **The hooks exist.** `register_forward_pre_hook(with_kwargs=True)` on
  both experts already rewrites `control_hidden_states_scale` per step
  (`_install_scale_hooks`); the same hook can swap
  `control_hidden_states` — the VACE conditioning latent — per step.
  `_euler_step` wraps `scheduler.step`, so the x0 estimate
  (`x_t − σ·v`) is exposed at every step under the euler sampler;
  uni_pc's multistep history makes x0 substitution messier, so the
  in-loop experiments run on `euler` (pass 2 already does).
- **diffusers' `callback_on_step_end`** may return a modified
  `latents` after each step (`_callback_tensor_inputs = ["latents",
  ...]`). That is the cheap injection point for latent blending.
- **The control latent is 96 channels**: `[inactive, reactive]`
  encodings (video·(1−mask), video·mask, each through the VAE) plus
  the mask interpolated `nearest-exact` to latent resolution, with the
  reference image prepended as one extra latent frame. The mask is a
  soft channel at latent res (60×104 at 480×832) — a per-latent-pixel
  "regenerate here" signal the model was trained on.
- **No backprop through the transformer**: 14B fp8, two experts. Any
  gradient guidance stops at the latent, or at most goes through the
  VAE decoder.

## 2. What the literature says

| Work | Model | Where in the loop | What is synchronised | Trained? | Take-away for us |
|---|---|---|---|---|---|
| **Splatent** (Dec 2025) | SD VAE f=8, d=4; SD-Turbo | after 3DGS, one refinement | latent 3DGS + multi-view attention from 3 reference latents in a grid | VAE frozen; SD-Turbo **fully finetuned** on 400 DL3DV scenes, 8×H100×24 h | Diagnosis: "VAE latent spaces fail to maintain equivariance under scaling and rotation; view-dependent high-frequency components exhibit the most severe 3D inconsistencies", so "during 3DGS optimization, inconsistent high frequencies average out, leaving only low-frequency components and causing blurry decoded images". The cure is a trained model. Their 3D-consistency metric is MEt3R (0.108→0.077). Limitation in their own words: where RGB-space splatting works, prefer it. |
| **LRF** (ICLR 2025) | SD VAE | before 3D | correspondence-aware regularisation of the *encoder*, then VAE↔RF alignment | VAE encoder finetuned | The latent can be *made* 3D-aware, but that changes the distribution the transformer was trained on — off the table until finetuning is. |
| **SyncTweedies** (NeurIPS 2024) | SD + depth ControlNet | every step | Tweedie x0 estimates, averaged in a canonical space (incl. 3DGS texture) and projected back; each instance keeps its own noise | no | Of five places to synchronise, only x0 (their Case 2) survives non-invertible projections such as rendering; synchronising ε or x_t "significantly declines"; denoising *in* the canonical space (Case 5) loses variance and blurs. For 3DGS they synchronise in **RGB**, not latent. |
| **MVEdit** (2024) | SD + Tile/Depth ControlNet | between steps | decoded x0 of all views → NeRF/mesh (96 Adam its/step) → rendered → **conditions the next step through ControlNet**; RGB blend only for t > 0.4T | no | "Appending a 3D NeRF… often leads to blurry results since NeRF averages the inconsistent multi-view inputs"; conditioning instead of replacing keeps the noisy→denoised information path intact. This is the argument for using VACE's control branch as the feedback channel. |
| **Scene-grounding guidance** (CVPR 2025) | camera-controlled I2V, 25 frames, 50 DDIM steps | every step | gradient of L1+LPIPS between **decoded** x0 and the 3DGS render, masked by transmittance (η=0.9), back through the VAE decoder into the latent | no | Works, but 50 steps and a decoder backprop per step; with six steps and 81 frames at 480×832 this is the expensive variant. |
| **VidSplat** (SIGGRAPH 2026) | **Wan 2.1 I2V** (also SVD, Hunyuan) | every step, three stages | rendered RGB + mask inverted to x_{t−1}, blended into the latent: strict early, fading (power ρ) in the middle, free at the end | no | Closest precedent on our model family. PSNR 15.7→25.8, FVD 262→114 with guidance; holding the mask for the whole run instead of fading it fails (FVD 241). Latent replacement of *known* regions, re-noised, with a schedule — the recipe for the steps that decide structure. |
| **FrameCrafter** (Apr 2026) | Wan 2.1 I2V 14B | — | encodes every view **as its own single-frame video** through the causal VAE's first-frame path | finetuned to "forget time" | The temporal-compression problem stated plainly: "potentially far away viewpoints can be merged into a shared encoding, losing fine-grained per-view information". Ablation: standard causal encoding instead of per-view drops PSNR 15.7→11.5, LPIPS .246→.676. We cannot adopt per-view latents (the transformer expects compressed ones without finetuning), but the loss it measures is the loss any latent-view representation has to model. |
| **Lyra** (NVIDIA 2025), **Wonderland**, **FlashWorld** (Wan-based) | video-model latents → 3DGS decoders | — | — | trained decoders / DMD | Evidence that video-model latents *carry* 3D structure well enough for a feed-forward 3DGS decoder — the latent is not geometry-blind, only not equivariant. Usable later as a trained module, not now. |
| **∫-noise / How I Warped Your Noise** (ICLR 2024) | image diffusion | before step 0 | the *noise* is warped along correspondences, Gaussianity preserved | no | Cheap, orthogonal lever: correlate the initial noise across the helix through the mesh's correspondences. Effect on a video model with its own temporal prior is unmeasured; SyncTweedies' result that noise-level synchronisation is the fragile one is a warning. |

Two facts to hold onto from this table. First, every training-free
method that works synchronises the **x0 estimate** (or a re-noised
version of a render of it), never the noise or the velocity, and every
one of them **fades the constraint out before the final steps**.
Second, nobody rasterises latents into a video model's latent without
a trained module in the loop; the two that go through the latent
(VidSplat, scene-grounding) encode a *pixel* render.

## 3. Why the Wan latent is harder than Splatent's

**Temporal compression.** The Wan 2.1 VAE (the one every Wan 2.2 14B
model, VACE-Fun included, uses — `z_dim 16`, spatial 8, temporal 4,
`temperal_downsample [F,T,T]`) is causal: frame 0 becomes latent frame
0 on its own, then every chunk of four frames becomes one latent frame.
81 frames → 21 latent frames (+1 for the reference). On the one-turn
helix a frame is 4.4° of orbit, so **one latent frame spans ~18° of
camera motion** and its content is whatever the encoder's causal 3D
convolutions make of four views plus the state carried from the chunks
before. A latent Gaussian rasteriser can put a camera anywhere, but no
single camera produces that latent. Splatent's per-image f=8 VAE has
no such term.

The approximation that makes latent splatting possible at all:
*treat latent frame k as the view from the chunk's central camera and
the encoder's temporal residual as noise the Gaussians will average
out.* Whether that residual is small on our slow orbit is experiment
E0 — and if it is not, the latent 3D representation is dead before it
starts, and the loop must go through pixels.

**Non-equivariance.** Splatent's diagnosis holds for any convolutional
VAE: a sub-8-px shift of the image is not a fractional shift of the
latent, and view-dependent high frequencies are encoded differently
from each view, so fitting a shared 3D field on them keeps only what
agrees — the low frequencies. In the *reconstruction* role (Splatent's)
that blur is the problem. In the *synchronisation* role it is
tolerable at the structural steps as long as the model's own
prediction keeps the high frequencies (SyncTweedies' Case 2 keeps each
view's variance; MVEdit conditions instead of replacing). The
measurement is E1: the fraction of latent energy per spatial frequency
that survives a fractional shift.

**Latent normalisation.** The transformer works on
`(z − latents_mean)·latents_std` per channel; any Gaussian fit and any
comparison must be done in that space (the per-channel stds range
1.1–3.3, so raw-space L2 weights the channels wrongly).

**Only three sync points.** At 50 DDIM steps a guidance term can be
gentle; at six Lightning steps with sigmas 1, .99, .96, .89, .75, .45,
any intervention is a large fraction of the whole trajectory. Steps
1–2 (σ ≥ .99) have nothing to synchronise; steps 3–5 are where it
acts; step 6 must be free.

## 4. Design space, ranked

### M1 — x0 synchronisation through a 3D canonical space (pixel path)

SyncTweedies Case 2 on our sampler. At the end of steps 3, 4 and 5
(euler):

1. x0 = x_t − σ·v from the model's output.
2. Decode x0 (81 frames, 480×832 — a few seconds on an L40S).
3. Lift to 3D and re-render at the 81 cameras. The canonical space is
   whatever is fast and forgiving of the mesh being ±5–10 px off the
   painted body: a warm-started b2ctrain splat (a few hundred
   iterations from the previous step's Gaussians, not a fresh
   training), or, for a first cut, the SAM-3D-Body mesh's UV atlas
   (`view_atlas.py` — closed-form bake, no optimisation, but locked to
   the mesh's geometry, which the anchor-raise finding says the
   painted body does not sit on).
4. Encode the re-render as a video (one VAE encode). Call it x0′.
5. Blend per latent pixel with the agreement mask: where the views
   agree, x0″ = x0′; where they disagree, keep x0. Take the euler
   step from x0″: v′ = (x_t − x0″)/σ, x_{t+1} = x_t + (σ_next − σ)·v′.
   Each frame keeps its own noise; only the clean estimate moved.
6. Step 6 runs untouched.

VidSplat's schedule (strict → fading → free) maps onto steps 3 → 4–5 →
6. Cost per sync: decode + fit + render + encode, of the order of a
minute with a warm-started splat, three times per pass.

### M2 — VACE self-conditioning with an agreement mask

MVEdit's argument transplanted: instead of overwriting x0, put the
consistent render where VACE expects a control video. At each sync
point, rebuild the control latent from x0′ — `inactive = x0′·(1−m)`,
`reactive = x0′·m`, mask m = *disagreement* (1 where the 3D projection
and the model's x0 differ by more than a threshold in normalised
latent space, 0 where they agree) — and swap it in through the
existing forward pre-hook (`control_hidden_states`, next to
`control_hidden_states_scale` which it already rewrites). This is
VACE's inpainting/extension mode, which it was trained on, driven by a
3D-consistency measurement: *keep what the views agree on, regenerate
what they do not*. It is the in-loop generalisation of what
`mask_splat` + the confidence gate do between the passes today, and
the closest reading of "guidance on how well all the views agree".

M1 and M2 compose: M2 alone leaves x_t untouched (the model is asked
to follow the control), M1+M2 both moves the estimate and asks. The
strength schedule (`strength`, per step) is the dial that already
exists for how hard the control pushes.

### M3 — latent Gaussians as the canonical space (Splatent proper)

Replace M1's decode → 3D → encode with Gaussians carrying a 16-channel
feature, fitted to the 21 latent frames of x0 at the chunk-centre
cameras (60×104 each — a tiny fit, seconds with gsplat's N-D feature
rasterisation; b2ctrain is RGB+SH only), rendered back at the same
cameras to give x0′ directly in latent space. Rendering into *latent*
frames is a matter of rendering at the chunk-centre camera; the
temporal residual is modelled as nothing. Gains: no VAE round-trips
(the sync becomes cheap enough to do at every step), a differentiable
projector in latent space (a gradient term ‖x0 − R(G(x0))‖² costs a
splat backward, not a decoder backward), and the consistent latent
can feed M2's control branch without an encode. Losses: whatever E0
and E1 say the latent throws away. A hybrid that avoids the temporal
problem: fit the Gaussians on **per-frame first-frame-path latents**
(each decoded x0 frame re-encoded alone, FrameCrafter's trick, but
only on our side) — 81 clean latent views — and use them only to
*measure* agreement (the mask of M2), while the control latent itself
still comes from a video encode of a pixel render.

### M4 — 3D-warped initial noise

Warp one noise field along the helix through the mesh's
correspondences (∫-noise at latent resolution, chunk-centre cameras),
so that a surface point sees correlated noise from every view. No
change to the sampler, a few lines before `pipe()`. Unknown effect on
a model with its own temporal prior; cheap enough to run as an arm of
the first pod batch and to drop.

### M5 — decoder-gradient guidance

Scene-grounding's loss (masked L1 + LPIPS between decoded x0 and the
render), gradient through the decoder to the latent, added to the
euler step. Strongest theoretical footing, weakest fit to a six-step
sampler with 81 frames at 480×832 (a decoder backward per step at that
size is heavy and gradient-checkpointing territory). Last.

## 5. Experiment ladder

E0–E2 need only the Wan VAE (`vae/` of
`linoyts/Wan2.2-VACE-Fun-14B-diffusers`, ~500 MB; the config is
already cached, the weights are not), diffusers in a local venv, and
`cyber_6f`'s 81 frames + cameras. They run on the 4070 Ti. E3 onward
need a pod.

**E0 — temporal mixing** (DONE, section 7)**.** For the 81 frames: z_video = E(all 81), and
z_1[k] = E(frame k alone) for every frame. Measure, in normalised
latent space, ‖z_video[j] − z_1[centre(j)]‖ / ‖z_video[j]‖ per latent
frame j, and per spatial frequency band; also decode z_video with
latent frame j replaced by z_1[centre(j)] and measure PSNR against the
frames. Repeat on a static clip (81 copies of one frame) to separate
the causal-state term from the motion term. *Decides whether a latent
frame can be treated as a view.*

**E1 — equivariance spectrum** (DONE, section 7)**.** Shift a frame by 0..8 px (and rotate
±2°, scale ±3%), encode, compare with the bilinearly-shifted latent of
the unshifted frame, per frequency band and per channel. *Gives the
frequency cut-off below which latent 3D consistency is even defined,
and whether the 16 channels differ (some may be usable as-is).*

**E2 — latent bake** (DONE, section 7)**.** Using the mesh and cameras: bake per-view
latents (both z_1 and z_video) into the UV atlas, re-render, decode,
compare with the frames (PSNR/LPIPS) and with the same bake done in
RGB and then encoded. Then the same with a gsplat fit of 16-channel
Gaussians on the chunk-centre views. *This is Splatent's Figure 3 on
our VAE and our subject; it says how much M3 loses relative to M1.*

**E3 — pod, M1 on helical pass 1.** Euler, shift 5, sync after steps
3/4/5 with the atlas first (no optimisation, worst geometry) then the
warm-started splat. Metrics we already have (the quality-sweep's PSNR
at the photograph, flow) plus MEt3R between adjacent frames and
between frames 90° apart. Repeats are owed: the denoise is not
pixel-deterministic at seed 0 (frames 27 dB apart, per the pass-2
sweep), so every arm runs twice.

**E4 — pod, M2** with the agreement mask, alone and on top of M1.
**E5 — pod, M4** as a cheap side arm.
**E6 — M3** only if E0's residual and E2's loss are small at the
structural frequencies.

## 6. Implementation notes for E3/E4

- Sampler hook: extend `_euler_step` (or wrap it) with an optional
  `sync(x0, step_index) → x0″` callable; uni_pc keeps its history of
  model outputs, so leave it out of the first cut.
- Control swap: the forward pre-hook already receives `kwargs`; add
  `control_hidden_states` replacement keyed by the same step index.
  The new control latent is built exactly as `prepare_video_latents`
  does it (inactive/reactive encode, mask `nearest-exact` to latent
  res, reference frame prepended), in normalised space.
- Cameras per latent frame: frame 0 → latent 0; frames 4j−3..4j →
  latent j, centre camera = frame 4j−1.5 (interpolate the helix).
- Agreement mask at latent res: |x0 − x0′| in normalised latent space,
  per-channel-std-weighted, thresholded and blurred; clamp to 1 outside
  the subject (rmbg matte at latent res) so the background is always
  regenerated.
- The decode inside the loop must use the tiled/chunked decoder path
  the pipeline uses at the end (`_PRE_LOOP_PHASES` has the timings of
  the encodes; the decode is inline in `__call__`).
- Everything logged: per-step ‖x0 − x0′‖, the mask's coverage, the
  time of each sync. The first run's job is to say whether the
  structure moved, not whether the result is pretty.

## 7. Results of E0–E2 (2026-09-19, local, Wan VAE only)

Scripts in `docs/latent-splat-experiments/` (masktest venv + diffusers,
the VAE from `linoyts/Wan2.2-VACE-Fun-14B-diffusers`, fp32, 480×832).
Latents are compared in the transformer's normalised space; "rel" is
‖a−b‖/‖b‖; "bands" are six radial sixths of the latent spectrum, low
to high (band 0 = structure coarser than ~6 latent px ≈ 48 image px).
Frames: `cyber_6f/circular` for E0/E1 (camera-free), `cyber_6f/colmap`
(the upscaled final frames, which the cameras belong to — the two sets
are numbered differently) for E2.

### E0 — a latent frame is not a view, but it is close to a *steady-state* code

| chunk latent `E(V)[j]` compared with… | rel | bands 0→5 |
|---|---|---|
| single-frame code `E_1(I_k)` of any frame in its chunk | 0.63 | .63 .60 .65 .70 .73 .74 |
| steady-state code `E_ss(I_{4j−1})` (3rd frame of the chunk) | **0.30** (cos .96) | **.27** .51 .68 .78 .84 .90 |
| the previous chunk latent `E(V)[j−1]` | 0.37 | .27 1.08 1.24 1.30 1.35 1.39 |

`E_ss(I)` is the last latent of a static clip of I: the causal encoder
drifts from the first-frame code to a code 2.2× its norm away (cos
0.86) within ~4 chunks (rel 0.038 after 17 frames, 0.020 after 49).
The first-frame path is a *different code family* — a stack of `E_1`
latents decodes through the video decoder with wrong colours and halos
(21 dB); a stack of `E_ss` codes decodes with the right colours (29.6 dB
at the matched frame of each chunk, 23.5 at the others, since the
decoder reads it as a still). `E_ss ≈ A·E_1 + b` with a 16×16 affine
map (rel 0.12, band 0 0.07) — the steady state is mostly a channel
recoding of the per-image code.

**95 % of a video latent's energy is in band 0**, and band 0 is the
only band where a chunk latent matches any per-view code (0.27) or
its own neighbour (0.27); bands 1–5 are chunk-specific and
uncorrelated between adjacent chunks (rel > 1) — the temporal code.

### E1 — equivariance: exact at 8 px, gone at 4 px above band 0

| image-plane transform | rel (E_1 / E_ss) | bands 0→5 (E_ss) |
|---|---|---|
| shift 8, 16, 24 px (whole latent px) | 0.008 / 0.004 | — (a stride-8 CNN: exact) |
| shift 1 px | 0.09 / 0.07 | .02 .15 .26 .33 .36 .43 |
| shift 4 px (half a latent px) | 0.22 / 0.17 | .05 .36 .63 .78 .85 .98 |
| rotation 1°, 2°, 5° | ~0.25 / ~0.21 | .08 .36 .63 .74 .81 .90 |
| scale 0.97, 1.03, 1.10 | 0.24–0.29 / 0.19–0.24 | .08 .34 .61 .73 .82 .93 |

A bilinear half-pixel shift of the latent decodes at 27 dB against
the shifted image (45 dB unshifted). Splatent's diagnosis, in
numbers, on the Wan VAE: **only band 0 transforms like an image**;
everything above it is re-encoded per view.

### E1b — what the decoder does with each band

Replace one part of `E(V)` (39.3 dB) with the same part of the
`E_ss` chunk stack (25.1 dB):

| latent | PSNR |
|---|---|
| band 0 from the stack, bands 1–5 from E(V) | 27.3 |
| band 0 from E(V), bands 1–5 from the stack | 25.4 |
| band 0 of E(V) shifted ½ latent px, rest intact | **34.3** |
| bands 1–5 shifted ½ px, band 0 intact | 27.0 |
| band 0 alone (bands 1–5 zeroed) | 23.6 |

The decoder tolerates a half-pixel error in band 0 (−5 dB) far
better than the same error in bands 1–5 (−12 dB). So a 3D projection
that moves band 0 and leaves the model's own bands 1–5 in place is
the cheap kind of intervention; one that touches the high bands is
the expensive kind.

### E2 — Gaussians carrying Wan latents vs the pixel path, with held-out views

Geometry from an RGB b2ctrain splat on the odd views (masks/ sidecar,
15 000 iterations, 40 s; subject-only PSNR 33.6 train / 22.1 held-out —
the 8.8° baseline is wide for this trainer, and the final frames are
not perfectly consistent). Latent path: six colour-only fits of 3
channels each on `E_ss` codes at 60×104 (SH 0, from the RGB geometry;
13 s each). Pixel path: render the RGB splat, encode. Everything
subject-masked.

| | train views rel (band 0) | held-out rel (band 0) | adjacent held-out change |
|---|---|---|---|
| truth `E_ss(V)` | — | — | 0.39 |
| **pixel path** `E_ss(render)` | 0.21 (0.17) | 0.77 (0.85) | 0.52 |
| **latent path**, geometry free | 0.23 (**0.11**) | 1.36 (**1.55**) | 0.81 |
| latent path, geometry frozen | 0.38 (0.25) | 1.23 (1.40) | — |
| latent path, geometry slow | 0.30 (0.16) | 1.34 (1.54) | — |

Decoded as chunk stacks at the held-out matched frames (subject
only): frames' own codes 22.7 dB, pixel path 23.1 dB, latent
Gaussians 20.8 dB.

The latent Gaussians fit the training views better than the pixel
path reproduces them (band 0: 0.11 vs 0.17) and **fail every held-out
view** (band 0 rel > 1: worse than predicting zero, whatever the
geometry is allowed to do), flickering between adjacent views at
twice the truth's rate. The pixel path degrades with the RGB splat's
own generalisation and no further; its per-view codes decode as well
as the frames' own. Splatent's Figure 3 reproduces on the Wan VAE:
the field keeps what agrees, and at band 0 too little agrees across
an 8.8° step for the field to say anything about a view it did not
see. In the sync role there are no unseen views (the fit and the
render share the 81 cameras), so a latent field would look fine on
its own training views and carry no consistency information — which
is the failure mode to avoid, not a feature.

Two things that were not the point but matter for M1: the NaN trap
(`--lr-mean 0` poisons b2ctrain's positions, 84 % NaN — freeze with
1e-12), and the cost: a 15 000-iteration RGB fit is 40 s on the
4070 Ti, a colour-only warm start 13 s, a VAE round trip of 81 frames
at 480×832 ~25 s (fp32, no tiling). Three syncs per pass are minutes,
not tens of minutes, even here.

### What this settles

- **M3 (latent Gaussians as the canonical space) is out**, on the
  measurements: a latent frame is a steady-state code plus a
  chunk-specific temporal code no view produces (E0), only band 0 of
  it is view-consistent (E1), and a Gaussian field fitted on the
  per-view codes does not generalise across views at all (E2). The
  Lyra/Wonderland route (a *trained* decoder from video latents to
  3DGS) is the only way latents become geometry, and that is
  finetuning.
- **M1 goes through pixels, and should synchronise band 0 only.** The
  x0 estimate is decoded, lifted (warm-started splat), re-rendered and
  encoded as a video; the low band of that latent replaces the low
  band of x0, the model's own bands 1–5 stay (E1b: −5 dB kind of
  intervention, not −12). The consistent render's *own* high bands
  are not used — they are the encoder's per-view re-encoding of the
  render, which is exactly what x0 already has.
- **M2 is unchanged**: VACE's control latent is built by the same
  encoder from the same render, and the mask comes from a band-0
  comparison in normalised latent space (E0's 0.27 is the noise floor
  of "agreement": a chunk latent is that far from the best per-view
  code even when nothing is wrong).
- The chunk-centre rule is now specific: latent frame j ↔ frame
  4j−1 (the third of its four), measured, not the middle.

## 8. The hook (built 2026-09-19; REMOVED 2026-09-20, see section 9)

M1 is in the tree as `pipeline/steps/wan22_sync.py` plus a seam in
`wan22_vace_denoise`, off unless asked for:

- **Settings** (both `helical.yaml` and `helical_shell.yaml`, mirrored):
  `sync_steps` (list, default `[]`), `sync_mix` (default `[1.0]`),
  `pass1_sampler_low` (default `uni_pc`; the sync needs `euler`). The
  step's own knobs: `sync_band` (1/6), `sync_iters` 6000 cold /
  `sync_warm_iters` 1500 warm, `sync_max_splats` 400k,
  `sync_mask_dilate_px` 24, `sync_debug_dir`.
- **Wiring**: a `sync_silhouettes` step (a `resize_batch` copy) keeps
  `render_initial_views`' mesh silhouettes before the anchor injection
  overwrites `dataset.masks` with VACE's flags; `denoise_pass1` (and
  the re-outline pass, which is pass 1 at 480p) reads `cameras`,
  `image_names`, `points_3d` and `sync_masks`.
- **The seam**: `_install_sampler`'s euler branch asks
  `LatentSync.velocity` for the velocity at a sync step; it decodes
  x0 = x_t − σ·v (the reference latent frames excluded), writes a
  COLMAP set, runs `b2ctrain` (masked views, warm-started from the
  previous sync's ply), renders over black at the same cameras,
  composites by the feathered silhouette, encodes, replaces the low
  radial band of x0 (raised-cosine mask at 1/6 of Nyquist; keeps 95.1 %
  of a real latent's energy) with `sync_mix` of the difference, and
  returns (x_t − x0″)/σ. UniPC steps never see it; `sync_steps` on a
  UniPC phase is refused by name.
- **Verified locally** (`e3_local_sync_check.py`, removed with the hook,
  E(V) of the colmap frames as x0, cyber_6f's cameras, the real VAE
  and b2ctrain on the 4070 Ti): 45 s per sync (decode 21, encode 12,
  splat 8 cold / 4 warm, render 4.4), x0 moved by 0.125 at mix 1 and
  0.06 at mix 0.5, the synchronised estimate decodes cleanly. Two
  compositing bugs found and fixed on the way: rendering over the
  renderer's default white whitened every soft edge, and compositing by
  the render's alpha on top of a black-background training counted the
  background twice — both decoded as a bright halo round the subject.
  The trainer must be given the card: `torch.cuda.empty_cache()` before
  it launches, or it fails on a busy card.
- **Unit tests**: `tests/test_wan22_sync.py` (the refusals, the seam's
  arithmetic, the band split where torch exists).

**Pass 2 too, and on by default** (later the same day, after the first
pod syncs fired as designed — 36 s each on an L40S, x0 moved 0.33 at
σ .93 then 0.20 at σ .83): `sync_steps_pass2` / `pass2_sampler_low`
wire the same seam into `denoise_pass2`, with the rmbg matte of the
re-render as the mask (`sync_silhouettes_pass2`) and the intermediate
splat (`dataset.splat_path`) as the first sync's warm start
(`sync_init_ply`). Both passes now default to `[2, 3, 4]` on `euler`;
`sync_steps: []` / `sync_steps_pass2: []` is the control arm, and
`sync_mix: [0]` measures without applying.

**First pod run, 2026-09-20 (`90db726`, pass 1 only):** the mechanics
held (36/32/31 s per sync, clean renders, no halos), x0 at σ .93 was
already sharp (Lightning), and the splat **memorised its views** —
render vs x0 23–29 dB per frame at 400k Gaussians / 6000 iterations,
each view rendered with its own walking pose (frame 0 feet together,
frames 20 and 60 mid-stride with different legs). What was applied
(0.33 → 0.20 → 0.15 of the low band) was the fit's softening, not a 3D
constraint: E2's train-view result, live. Answer (`32b8f1a`+): make the
splat unable to memorise — 50k Gaussians, 3000/800 iterations, the
hollow loss against the body mesh (`sync_mesh`), and the evidence gate
at render time (`sync_confidence`, render_splat's gate and conf args)
so what is fed back is what the views agree on and x0 is kept where
nothing does. Each sync now logs render-vs-x0 PSNR (memorisation if
high) and the gate's kept fraction of the subject. Locally, on the
consistent colmap frames: 25 dB, gate 97 %.

**Two artefacts of run c0514e's final splat, unrelated to the sync,
diagnosed by retraining its colmap_intermediate locally** (11 arms,
~2 min each on the 4070 Ti; scratch `halo/ab*.py`): the thin light rim
round the figure seen only head-on is the twelve photograph copies'
own outermost pixels (front-halo fraction 0.20 as-is, 0.13 without the
copies, 0.30 with their mask grown 4 px, 0.14 shrunk 4 px; compositing
the photo over the frame outside its mask changes nothing, so it is
not the backdrop) → `photo_priority_weights.copies_erode_px` 4. The
grey stripe from ear to chin is the face cap's rim along the jaw
contour (gone without the face supports, at the cost of the cap's
sharpness; gone with the supports' mask shrunk 30 px at their close-up
cameras, sharpness kept) → `select_support_views.erode_px`, 30 on the
face supports.

To run the experiment (E3), same seed, two arms:

    python -m pipeline.cli run helical --reference-image <photo>      # sync on (the defaults)
    python -m pipeline.cli run helical --reference-image <photo> \
        --param sync_steps=[] --param sync_steps_pass2=[]          # the control arm

The control arm keeps euler on the low expert, so the only difference
is the sync. `debug/sync_pass1/` holds every sync's decoded
and rendered frames (every ~10th) and `sync_stats.json` (per step: how
far the render moved x0, in full and in the low band; the timings).
Second arm of interest: `sync_mix=[1,1,0.5]`.

## 9. The A/B, and the verdict (2026-09-20)

Run b10542 (the defaults: re-outline `[2, 3, 4]`, pass 1 `[2, 3, 4]`,
pass 2 `[2, 3, 4]`, mix 1.0) against f267a0 (`sync_steps: []`,
`sync_steps_pass2: []`), both `c06777a`, same input. Each stage's shipped
splat rendered at its own cameras and scored inside the matte — the
frames' detail (energy above ~4 px), the splat's fit to its frames, the
detail the splat keeps, the rendered alpha outside the matte. Harness,
arms and sheets in `~/Projects/b2ctrain/out/sync_ab/`.

| stage | run | frame detail | fit PSNR | splat detail | alpha outside/inside |
|---|---|---|---|---|---|
| re-outline | ON | 17 | 31.0 | 13 | 5.3 % |
| | OFF | 127 | 27.5 | 75 | 6.1 % |
| pass 1 | ON | 39 | 20.4 | 39 | 6.0 % |
| | OFF | 227 | 17.4 | 144 | 5.7 % |
| pass 2 raw 720p | ON | 12 | | | |
| | OFF | 131 | | | |
| final splat | ON | 15 | 27.8 | 12 | 2.2 % |
| | OFF | 186 | 22.6 | 106 | 2.4 % |

- The sync strips 6–10x of the frames' fine detail at every stage it
  runs, and it compounds: the final splat renders with 9x less detail.
- The "grey cloud" the user saw on pass 2 is not alpha outside the
  matte (equal in both runs); it is lost detail and contrast inside the
  silhouette. It is born at step 2: x0 is already sharp there (pass 2 is
  conditioned at 0.8 on the intermediate splat, pass 1 on a strong
  control) and the 50k-Gaussian render that replaces 45 % of it is a
  blur with a grey rim (`debug/sync_pass2/step_02`).
- The fit gain is mostly blur. Same trainer recipe, no rig, on the pass-1
  frames: ON 27.5 dB, OFF 22.9, OFF blurred (σ 1.5) to ON's detail
  level 25.1. Half the gain is blur; ~2.4 dB is real low-band
  consistency — but the blurred control's splat keeps MORE detail (42)
  than ON's (32) from frames of equal detail, so what the sync leaves
  behind is less 3D-consistent than plain blur.
- Re-outline: hairs 0.71 % vs 0.81 %, silhouette IoU +0.008 — nothing
  the outline's alpha needs.

The premise was that x0 at steps 2–4 is inconsistent across views and a
consensus render fixes it; both passes' x0 is already consistent enough
that the render is a downgrade. Verdict: harmful at all three stages.
The hook, `wan22_sync.py`, its tests, the settings (`sync_steps`,
`sync_mix`, `sync_steps_pass2`, `reoutline_sync_steps`,
`pass1_sampler_low`, `pass2_sampler_low`), the `sync_silhouettes` steps
and the denoise inputs were removed from the tree the same day; the
samplers are back to what they were before 2026-09-19 (uni_pc on the
low expert of both passes). Any revival must be judged by the final
splat's detail, not by fit PSNR, which blur inflates.

## Sources

- Splatent: Splatting Diffusion Latents for Novel View Synthesis —
  https://arxiv.org/abs/2512.09923 (project: https://orhir.github.io/Splatent/)
- Latent Radiance Fields with 3D-aware 2D Representations (ICLR 2025) —
  https://arxiv.org/abs/2502.09613
- SyncTweedies: A General Generative Framework Based on Synchronized
  Diffusions (NeurIPS 2024) — https://arxiv.org/abs/2403.14370
- Generic 3D Diffusion Adapter Using Controlled Multi-View Editing
  (MVEdit) — https://arxiv.org/abs/2403.12032
- Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D
  Gaussian Splatting from Sparse Inputs (CVPR 2025) —
  https://arxiv.org/abs/2503.05082
- VidSplat: Gaussian Splatting Reconstruction with Geometry-Guided Video
  Diffusion Priors (SIGGRAPH 2026) — https://arxiv.org/abs/2605.11424
- Novel View Synthesis as Video Completion (FrameCrafter) —
  https://arxiv.org/abs/2604.08500
- Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model
  Self-Distillation — https://arxiv.org/abs/2509.19296
- FlashWorld: High-quality 3D Scene Generation within Seconds —
  https://arxiv.org/abs/2510.13678
- How I Warped Your Noise (ICLR 2024) — https://openreview.net/forum?id=pzElnMrgSD;
  Infinite-Resolution Integral Noise Warping — https://arxiv.org/abs/2411.01212
- diffusers `pipeline_wan_vace.py` (control latent construction, hooks) —
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py
- Wan VAE config: `~/.cache/huggingface/hub/models--linoyts--Wan2.2-VACE-Fun-14B-diffusers/.../vae/config.json`
