# The VACE denoise: skeleton bleed, the sampler, and the anchor drift

Findings from 2026-09-07/08, written so the next pod run starts from them.
Three things, in the order they were found. Sections 1 and 3 are
diagnoses; section 2 is a change already in the working tree. Section 4
is the one pod run that settles section 3's open question, and section 5
the minimal set that settles section 1's — section 1 stopped being
qualitative on 2026-09-08, when eleven runs made it a number.

## 1. The skeleton surviving the denoise as ink

**What was seen.** In `fd852e`'s `colmap_intermediate` (the pass-1 output)
the DWPose skeleton is painted into the frames: red neck-to-shoulder
sticks over the hair on back views, blue/purple sticks down the legs.
DWPose limb colours in DWPose places. Pass 2 keeps them: they are baked
into the intermediate splat, rerendered, and denoised at 0.8.

**The invocation is faithful.** `pipeline/steps/wan22_vace_denoise.py`
was read against ComfyUI's `WanVaceToVideo` (comfy_extras/nodes_wan.py)
and the `WAN21_Vace` wrapper: mask polarity, the inactive/reactive split,
the reference latent with its zero half, the mask padding, the latent
normalisation and the trim all match. Not the cause.

**The bleed predates the port.** `cyber_6f/circular` — the ComfyUI output
the port was verified against, strength 1.0 on every step — has it too,
faintly: a magenta line down the jacket's spine and pink hip lines on
frame 41. A skeleton over a grey silhouette is not the black-background
pose map VACE learned, and the model keeps part of it as content.

**The lever, first reading (eyeballed, superseded below).** The VACE scale
on the low-noise expert's FIRST step:

| run | pass-1 schedule | bleed |
|---|---|---|
| 9cc643 | high 1.0, low 0.5 on all four low steps | faint |
| fa59e5 | `[1, 1, 1, 0.5, 0.5, 0.5]` | heavy |
| 00ce5a, fd852e | `[1, 1, 0.8, 0.8, 0.8, 0.8]` | heavy |
| 5e2817 | `[1] * 6` | heavy |

fa59e5 differs from 9cc643 only at step 3, and the taper
`[1, 1, 0.75, 0.5, 0.25, 0]` also bleeds (0.75 at step 3). The bleed
turned visibly heavier on 2026-09-01, when the skeleton became the 7 px,
60%-dimmed DWPose style: more ink, darker against the grey ground.

### 1a. Measured, 2026-09-08: eleven runs, and the step that matters is step 2

`scripts/skeleton_leak.py` puts a number on the bleed, so a schedule can
be ranked instead of squinted at. It reads a run's `debug/denoise_pass1_input`
(the control it was handed) against its `colmap_intermediate/images` (what
came back), finds the stick pixels in the control, estimates what should
have been at those pixels from the output's own annulus just off each
stick, and projects the leftover onto that stick's colour direction. Units
are 8-bit chroma: **~15 is plainly visible ink, ~2 is invisible, 0 is
clean**, and the sign is meaningful. `--by-hue` splits it by limb, because
DWPose's palette makes hue and limb the same thing. `--iou` reports the
other half of the trade: how well the denoised frame's own alpha matte
still agrees with the control's mesh figure, which is what a run gives up
to score a low leak.

Eleven runs on 2026-09-08, one subject, `seed: 0` throughout,
`sigma_schedule: beta` throughout, `steps_high: 2` / `steps_low: 4`
throughout. The **control drawing is identical in all of them** — 12.17-12.19k
stick pixels at chroma 152 and luma 90 on frame 40 in every run, so
`outline_strength: 10` moved the silhouette fill by three levels (111 to
114) and left the skeleton alone. Everything below is therefore a
denoise-side effect, not a drawing-side one.

Pass-1 `strength`, single-step ablation off `[1]*6` at `sampler_shift: 3`,
`sampler_high: euler`. `flow` is `debug/alignment/alignment.json`'s
iteration-1 mean, the alignment loop's own measure of how far the views
disagree — an independent read, and lower is better:

| run | pass-1 `strength` | leak | yoke | cover | flow | IoU |
|---|---|---|---|---|---|---|
| 467c17 | `[1, 1, 1, 1, 1, 1]` | 16.62 | 55.6 | 0.58 | 1.039 | 0.8501 |
| b52659 | `[1, .5, 1, 1, 1, 1]` | **4.95** | **9.6** | 0.34 | 1.087 | 0.8388 |
| aaf842 | `[1, 1, .5, 1, 1, 1]` | 11.99 | 51.6 | 0.49 | 1.037 | 0.8471 |
| cbc61f | `[1, 1, 1, .5, 1, 1]` | 10.22 | 31.1 | 0.43 | 1.047 | 0.8424 |
| 19e3c5 | `[1, 1, 1, 1, .5, 1]` | 15.16 | 55.7 | 0.54 | 1.046 | 0.8457 |
| fb769a | `[1, 1, 1, 1, 1, .5]` | 16.52 | 58.3 | 0.58 | 1.046 | 0.8497 |
| ef13a7 | `[1, 1, 1, 1, .5, 0]`, pass 2 tapered too | 14.88 | 53.2 | 0.52 | 1.376 | 0.8435 |

**Step 2 is the lever, and step 2 is the LAST HIGH-NOISE step.** Halving it
alone takes 70% of the leak and 83% of the neck/shoulder yoke, for 1.1
points of silhouette IoU and 5% of the view flow. The first reading above
pointed one step too late: step 3, the low expert's first, halves the
overall leak by a third but leaves the yoke exactly where it was — it
governs the limbs, not the yoke.

**Steps 5 and 6 are worse than inert — they cost and do not pay.** Every
halved step gives up some silhouette IoU; what differs is what it buys.
Leak removed per 0.001 of IoU surrendered:

| step halved | 2 | 3 | 4 | 5 | 6 | ef13a7 |
|---|---|---|---|---|---|---|
| leak per 0.001 IoU | 1.03 | 1.54 | 0.83 | 0.33 | 0.25 | 0.26 |

Steps 2, 3 and 4 are the whole of the useful range and their ordering is
inside the noise on a single subject; steps 5 and 6 buy a quarter as much
per point given up. 19e3c5 is the clearest case: halving step 5 moved the
leak 16.62 -> 15.16, which is nothing, and gave up 0.0044 of IoU doing it.
ef13a7, with VACE fully OFF for the last step on both passes, is the
baseline on leak and the worst run in the group on flow (1.376).

The shipped default `strength: [1, 1, 0.75, 0.5, 0.25, 0]` spends its
entire taper on steps 3-6 — most of it on the two that were measured to be
a bad trade — and pays full scale on step 2, the one that is not.
**Move the taper earlier.**

Then the two runs that changed more than one thing, and the accident:

| run | pass 1 | leak | yoke | flow | IoU |
|---|---|---|---|---|---|
| 0281b9 | `[1]*6`, shift 2.5, euler | 17.32 | 62.5 | 1.047 | 0.867 |
| 31a008 | `[1, 1, .5, .5, .5, .5]`, shift 2.5, euler | 1.74 | 14.5 | **0.986** | 0.853 |
| cd7482 | 31a008 again, plus `background: grid` | 9.56 | 13.9 | 1.227 | — |
| 9e315f | `[1, 1, .5, .5, .5, .5]`, **shift 5, uni_pc** | **-0.47** | **-0.05** | — | 0.848 |

- **Tapering all four low-noise steps to 0.5 is nearly free.** 31a008 is a
  10x leak reduction over its own baseline 0281b9 for 1.4 points of
  silhouette IoU, and it has the BEST view agreement of any run measured
  (0.986 against 1.047). Nothing here supports the fear that a softer
  control costs consistency; over this range it buys some.
- **The yoke is the stubborn part.** Every taper leaves it near 14 while
  driving the limbs to zero. It is not "stick over dark hair": the yoke
  sits over the brightest content of the three bands measured (luma 106
  against the shins' 56) and leaks most, and leak correlates weakly
  NEGATIVELY with what is underneath (-0.38, -0.34, -0.14 for yoke, legs,
  shins). Something about that position, not that pixel.
- **`background: grid` costs both axes.** cd7482 is 31a008 with the grid
  turned back on: 1.74 -> 9.56 leak and 0.986 -> 1.227 flow. The shipped
  workflow already renders stage 1 on a flat wall; this is the measurement
  that says leave it there.
- **9e315f is the only run with no leak at all**, yoke included. It is
  also **confounded three ways**: `sampler_shift: 5` instead of 2.5,
  `sampler_high: uni_pc` instead of euler, and no `splat_inactive_mask`.
  It did not finish (no `ply/`, no `colmap/`, no `alignment.json`).
  Section 5 unpicks it.
- **The trade-off is real along the strength axis, and the sampler axis is
  off it.** Over the seven single-step ablation runs — same shift, same
  sampler, only `strength` moving — leak and silhouette IoU track each
  other tightly: Spearman 0.89, p = 0.007. Tapering the control buys a
  cleaner frame and pays for it in mesh agreement, every time, at the rate
  the table above prices.

  Over all ten measurable runs together that correlation collapses to 0.38
  (p = 0.28), because shift and the sampler move IoU on their own: 0281b9
  and 467c17 are the same `[1]*6` schedule at shift 2.5 and 3 and score
  0.867 against 0.850, and 9e315f reaches leak zero at 0.848 — better mesh
  agreement than b52659 manages at leak 4.95.

  That is the case for section 5's R2 and R3 in one sentence: **on the
  strength axis a clean frame has to be bought, and the other axes may not
  charge for it.** The whole measured range is 0.839 to 0.867, so compare
  within a sampler configuration and treat three points of IoU as the
  budget.

One trap found while reading these logs, now fixed: `pipeline/worker.py`'s
`_describe` summarised every list by its first element, so the
resolved-params block logged `[1, 0.5, 1, 1, 1, 1]` as `list[6] of 1` and
this whole sweep read back as six identical runs. Short scalar lists are
spelled out now. The per-run override line (`step params:`) was right all
along and is the record to trust for anything logged before the fix.

**Two real invocation differences from the reference graph** remained:
the sampler schedule (section 2) and the reference image — diffusers
LETTERBOXES it on white (768x1536 -> 640x1280 with 40 px white bars each
side, `preprocess_conditions` in pipeline_wan_vace.py) where ComfyUI's
`common_upscale(..., "center")` crops to fill at 0.9375. Untested whether
either matters for the bleed.

## 2. The sampler now matches the ComfyUI graph (working tree, untested on a pod)

The reference graph (`ComfyUI-Body2COLMAP/workflows/api/denoise.json`)
sampled `uni_pc` on ComfyUI's `beta` scheduler with no ModelSamplingSD3
node, which leaves a Wan 2.2 model at ComfyUI's default shift of 8.0
(comfy/supported_models.py, `WAN21_T2V.sampling_settings`). The HF
repo's scheduler config is UniPC too, but spaced `linspace` at
`flow_shift: 3.0`, and that is what the step ran until 2026-09-07:

    ComfyUI   beta, shift 8      t = 1000, 988 | 955, 889, 753, 448
    diffusers linspace, shift 3  t = 1000, 938 | 857, 750, 601, 376

`wan22_vace_denoise` gained two per-call params: `sigma_schedule`
(default `beta`) and `sampler_shift` (default 8.0). `linspace` + 3.0 is
the old run. `_comfy_beta_sigmas` reproduces comfy/samplers.py's
`beta_scheduler` with scipy (venv_wan22 sees venv_base's copy);
`_install_sigma_schedule` wraps the scheduler's `set_timesteps` on the
instance because `pipe()` calls it with a step count and nothing else;
`_configure_sampler` writes `flow_shift` through `register_to_config`.
Verified against a real diffusers UniPC scheduler built from the pinned
config (0.41 checkout): beta at shift 8 steps through 999, 987, 954, 889,
753, 447. The pod's diffusers 0.40.0 has the `sigmas` kwarg; 0.36.0 does
not, and the wrapper refuses by signature rather than running linspace
while logging beta. The custom sigmas must be an ndarray (diffusers does
arithmetic on them; a list raises).

Consequence: at shift 8 the checkpoint's `boundary_ratio` 0.875 would put
FOUR of six steps on the high-noise expert (1000, 988, 955, 889 >= 875);
the graph split by count at step 2, and so does `steps_high`/`steps_low`.

**2026-09-08: the solver, the hand-off and the reference image are matched
too**, by four more per-call params, all defaulting to the graph's
behaviour, all with the pre-change behaviour still sayable:

| param | default (= the graph) | the run before |
|---|---|---|
| `solver_variant` | `bh1` | `bh2` |
| `solver_order` | 3 (a cap, see below) | 2 |
| `handoff_reset` | on | off |
| `reference_fit` | `crop` | `letterbox` |

- **Solver.** ComfyUI's `uni_pc` sampler is UniPC's bh1 variant with
  `order = min(3, len(timesteps) - 2)` PER SAMPLER — and each
  `KSamplerAdvanced` owns only its slice of the schedule plus the boundary
  sigma, so the graph's 2-step high-noise sampler ran at order 1 and its
  4-step low-noise one at 3 (`_phase_order`). Per step the graph took
  orders 1, 1 | 1, 2, 2, 1; diffusers left alone takes 1, 2 | 3, 3, 2, 1.
  Both measured on the real scheduler. Two traps found there: raising
  `solver_order` through the config resizes `model_outputs` on the next
  `set_timesteps` but not `timestep_list`, and `step()` then indexes past
  it (IndexError) — the step sizes both itself; and diffusers' bh1 is
  NON-FINITE at the final step because the terminal sigma is 0 and bh1's
  B(h) = h is infinite there. ComfyUI's `sample_unipc` puts 0.001 in
  place of the terminal 0, and so does the step for bh1
  (`COMFY_TERMINAL_SIGMA`); bh2 keeps diffusers' 0 so the old run stays
  byte-identical.
- **Hand-off.** The graph's second KSampler is a NEW sampler: the
  low-noise expert starts with an empty multistep history (its first step
  first-order, no corrector for the step before it), and the first
  sampler ENDS, so its last step is first-order too (`lower_order_final`
  counts the steps its own sampler has left). diffusers runs one loop and
  carries the history across, so its step 3 was corrected by x0
  predictions the HIGH-noise expert made at t=1000 and 988. Two pre-hooks
  reproduce the graph: `_phase_end_hook` on `transformer` drops the order
  to 1 for the last high-noise step; `_handoff_hook` on `transformer_2`,
  at its first forward of a pass, sets the low-noise sampler's order and
  clears `model_outputs`, `timestep_list`, `lower_order_nums`,
  `last_sample` (what `set_timesteps` initialises), keeping `_step_index`.
  Verified on the real scheduler: orders 1, 1, 1, 2, 2, 1 and the corrector
  skipped at step 3, finite throughout.
- **Reference image.** `_fit_reference` is comfy/utils.py's
  `common_upscale(..., "center")`: crop to the frame's aspect about the
  centre (Python `round`), resize to fill with bilinear. A 768x1536 back
  panel loses 85 rows top and bottom; cyber_6f's 1440x1280 sheet loses 360
  columns each side. Compared numerically against ComfyUI's own function:
  identical when no scaling is involved, 0.27 levels mean difference at
  the 768->720 resample. diffusers then finds a reference that already
  fits and pads nothing.

Tests: `tests/test_wan22_conditioning.py` (`TestSamplerSchedule`,
`TestSamplerReachesTheScheduler`, `TestSolverAndHandoff`,
`TestReferenceFit`, `TestDeclaration`), full suite 1154 passed. Two
harness traps recorded there: scipy.stats first imported inside
`patch.dict(sys.modules)` is evicted on exit and refuses a re-import, and
scipy's array-API shim needs `torch.Tensor` on any stubbed torch.

What still differs from the graph, and is left alone: the fp8-scaled
experts with a live LoRA vs the Q8 GGUF quant, the RNG stream, integer
timesteps and the 1e-6 nudge on the first sigma.

Not committed. No image. No pod run.

## 3. The anchor drift on the circular stage — the denoiser, not the warp

**The observation.** In every b2crunner `colmap_intermediate`, a feature
near the frame centre (the trousers button, the top's hem) moves up from
frame 1 to 2 to 3 and then levels off; frames ~65-81 descend gradually
into the anchor at 81. cyber_6f does not do this. Template tracking,
fd852e, button row:

| frame | 81 | 80 | 79 | 78 | 77 | 75 | 72 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| row | 608 | 607 | 605 | 603 | 602 | 600 | 598 | 608 | 606 | 604 | 604 | 603 | 602 |

Every run drifts 5-10 px: 5e2817 (flat strength 1.0), 9cc643 (low 0.5),
the 08-30/31 bikini runs (before the DWPose skeleton and the room
backdrop), fa59e5. `cyber_6f/circular`: a centre patch holds row 588 on
every frame, both sides.

**What is ruled out.** The first 14 steps of the workflow were re-run
locally on `datasets/isolated/00.png` with an extra `save_dataset`
before `warp_reference_to_anchor`, giving the mesh render at the anchor
camera itself (recipe below). At that camera the mesh silhouette's width
matches the warped photo's ROW BY ROW through the torso (rows 450-650
within 1-3 px); the head/feet gaps are hair (mesh scalp 7 px below the
hair top) and the platform soles (mesh feet 22 px above them). The
render is flat from frame to frame (head, feet and skeleton constant over
frames 1-5). Given cameras have uniform height and pitch; `cameras.txt`
carries one model for all frames. Default vs MoGe-2 focal changes the
mesh height by one pixel. So the warp is right, the mesh placement is
right, and the anchor frame is the one correct frame.

**Mechanism.** The denoise pass paints the body 5-10 px higher than both
the photograph and the drawing; the injected anchor frames (VACE mask 0)
pull their neighbours back toward the truth. The transition is
asymmetric because of the Wan VAE's causal chunking: frame 1 is a latent
on its own, so frames 2-5 are the next chunk and the step is sharp;
frames 78-81 share one chunk with the photo at 81, so the descent smears
over the last ten frames.

**Suspects, since the geometry is right.** Everything in the invocation
that differed from the graph is now matched in the working tree (section
2: schedule, shift, solver, hand-off, reference fit) but untested on a
pod. What remains different: the Q8 GGUF vs fp8-scaled quant (rounding
level), and b2crunner's own control content — the outline silhouette, the
face splat composited in, MHR-derived joints drawn as DWPose. (The outline
has an alternative since 2026-09-08: docs/re-outline.md, a silhouette cut
from a matte of a 480p denoise of these very frames, gated off by default.)

**Earlier stage-2/3 finding, still true but a separate thing**: the
helical re-render anchors at the world origin while the intermediate
splat was trained on `refine_cameras`' moved anchor camera (fd852e: 46 mm,
1.1 deg), so frame 38 of `colmap_preupscale` sits ~5 px low against its
neighbours and the alignment loop's flow peaks there. Fix options: anchor
the helix at the live refined camera (`dataset.cameras[anchor_frame_index]`,
as `_cap_axis` does) and re-warp the anchor image by the residual
rotation, or pin frame 0 in `refine_cameras`. Not implemented.

## 4. The pod run that settles section 3

Run the CURRENT `wan22_vace_denoise` step — the section-2 defaults, so
the invocation matches the graph in everything but the quant and the RNG
— on `cyber_6f/initial`: 81 frames at 720x1280, the anchor photo at
frames 1 and 81 with alpha 0, every render at alpha 255, `reference.png`
(the 1440x1280 whole sheet, which `reference_fit: crop` takes to the
centre 720 columns exactly as ComfyUI did), `prompt.txt`. That is exactly
the control video ComfyUI received for `cyber_6f/circular`.

- Track a centre feature (the jacket's zipper/belt at about x=416,
  y=588 in frame 1) through frames 1-10 and 72-81. In
  `cyber_6f/circular` it holds row 588 on every frame.
- Drift absent and the bleed like `cyber_6f/circular` frame 41's -> the
  invocation is exonerated; it is the control content. Then A/B the
  render on this pipeline's own subject: skeleton on black without the
  silhouette (the canonical pose map), then the face splat off.
- Drift present -> what is left is the quant. Flip the four section-2
  params back one at a time to see which one it was NOT.
- Either way, read the same run for section 1: compare its skeleton bleed
  against `cyber_6f/circular` frame 41.

A single denoise pass is ~9 minutes on the 5090.

## 5. The six runs that settle section 1

One pod session, six runs, no conditionals: a pod's fixed launch cost
amortises over a batch, so every run here is readable on its own rather
than waiting on another run's answer. **Five of the six are a
single-parameter delta from a run already measured in 1a**, which is what
makes them interpretable without a fresh baseline in the batch.

The two references, both measured, both on this subject at `seed: 0`:

    A = 31a008   leak  1.74   yoke 14.54   flow 0.986   IoU 0.8525
    B = 467c17   leak 16.62   yoke 55.61   flow 1.039   IoU 0.8501

### The matrix

Every run is A or B with the cell below changed. Blank = unchanged.

| # | base | `denoise_pass1.strength` | `sampler_shift` | `sampler_high` | `strength_layers` | what it answers |
|---|---|---|---|---|---|---|
| E1 | A | `[1, .5, .5, .5, .5, .5]` | | | | do the two proven levers add? |
| E2 | A | | `5` | | | shift alone |
| E3 | A | | | `uni_pc` | | sampler alone |
| E4 | A | | `5` | `uni_pc` | | both — and 9e315f, completed |
| E5 | B | | | | `[1, 1, 1, .75, .5, .25, 0, 0]` | is the ink in the DEEP layers? |
| E6 | B | | | | `[0, 0, .25, .5, .75, 1, 1, 1]` | ... or the shallow ones? |

**E2/E3/E4 with A itself complete a 2x2 factorial** on the two knobs that
9e315f confounded, so they give both main effects and the interaction
rather than two thirds of an answer. That matters because 9e315f is the
only run that ever reached zero leak, it is unreplicated, and it never
finished (no `ply/`, no `alignment.json`) — E4 is that configuration run
to completion, so the batch contains a working pipeline even if the main
effects turn out to be nothing on their own.

**E5/E6 bracket the depth axis instead of guessing its direction.**
`strength_layers` tapers each step's scale across the eight VACE injection
layers (`vace_layers` = 0, 5, 10, 15, 20, 25, 30, 35), shallow to deep, and
it is `None` in all eleven runs of 1a. It is the only axis that could break
1a's trade-off curve, because pose and texture are coupled along time and
might not be along depth — but which end holds which is a guess, so run
both ends. If neither moves the leak, the depth axis is dead and never
needs another run. They sit on B, at `strength: [1]*6`, so the taper is
measured alone; keep the six-entry `strength` rather than `[1]`, because a
single-entry list bypasses `_vace_scale_hook` entirely and would change the
code path as well as the taper.

### Exact overrides

The six are checked in as settings sidecars under
**`docs/vace-leak-sweep/`** — one `.yaml` per run, each self-contained,
with that run's reasoning in its header. `docs/vace-leak-sweep/README.md`
has the zip recipe. That is the shape `pipeline/runs.py` was built for: the
same reference sheet copied under six stems, one sidecar beside each, is
six variants of one subject in a single submission.

Two things about the files. The **param panel stays empty** — a sidecar
merges into the submission-wide overrides and wins on conflict, but it
cannot remove one, and the families differ partly by what they leave unset
(`denoise_pass2.sampler_shift`, `render_initial_views.outline_strength`).
Every sidecar sets **`export_colmap_intermediate: true`**, which defaults
false and is the output the leak metric reads; without it the run is
wasted.

`seed` stays at the workflow default (0) — every run in 1a used it, and a
new seed would make all ten reference numbers unusable.

All six were dry-run through the real submission path
(`read_settings_sidecar` + `_refuse_unknown_overrides`, which is fatal on
an unknown step id or an undeclared param, so a typo would have failed on
the pod minutes in) and resolve to the intended values. Both layer tapers
were then put through `_scale_schedule` at `n_layers=8, n_steps=6` and come
out as 6 x 8 plans.

### Three things to get right

- **Pass 2 stays fixed in all six.** The leak is measured on
  `colmap_intermediate`, which is pass 1's output, so mirroring `sampler_shift`
  or `sampler_high` onto pass 2 would change the archive's *other* frames
  without changing the number being read and cost the delta its meaning.
  A-family pins pass 2 at shift 2.5, B-family leaves it at the default 8 —
  each matching its own reference run, which is why the two families' pass-2
  lines differ.
- **The render differs between the families and must stay that way.** A
  runs at the default `outline_strength` (12.6%, fill 111), B at 10 (fill
  114). 1a measured the skeleton itself to be pixel-for-pixel identical
  either way, so this changes nothing about the ink — but matching each
  reference exactly is what keeps five of the six single-variable.
- **`splat_inactive_mask` is the one thing 9e315f had that E4 will not.**
  It was OFF there — the param's own default — and ON in A. As of
  2026-09-08 it is also ON in the workflow, on both `+splat` render steps,
  so no run needs to remember it: 1a measured it in every reference run and
  a knob that works should not be a thing to forget. The step keeps
  declaring it off, because nothing downstream is *obliged* to consume the
  batch. If E4 fails to reproduce 9e315f's zero leak while E2 and E3 are
  also flat, that mask is the remaining suspect and is the seventh run —
  not a knob to fold into this batch, because it would stop E4 from being
  a clean 2x2 corner.

### Reading the batch

    scripts/skeleton_leak.py --iou --by-hue <result-dir> ...

plus `debug/alignment/alignment.json`'s iteration-1 mean. Against the
references above:

- **E1 at leak <1 with the yoke under 10 ends the sweep** — that is the
  new default, and E2-E6 become background on why.
- **Any of E2/E3/E4 at leak ~0 with IoU >= 0.85** beats E1 outright,
  because 1a showed the strength axis has to pay for a clean frame and
  these might not.
- **E5 or E6 moving the leak off B's 16.62 at all** is the most valuable
  result in the batch even if it is small, because it is the only evidence
  that ink and pose separate by depth. Compare them to each other, not just
  to B: opposite tapers that both reduce the leak would mean the layer sum
  is what matters, not where in the stack it sits.

### What none of these measure

Whether the frames still describe the subject's real shape. Silhouette IoU
compares them to the MESH, so a run that follows the mesh into a wrong pose
scores well; flow says the views agree with each other, not that they agree
with the person. Both are comparative and both are blind in the same
direction. The face is outside all of it too — the yoke leak sits at the
top of the torso, and the face cap covers what is above it.

### Deliberately not in the batch

- **`sigma_schedule: simple`** (still never run). Schedule and
  `sampler_shift` move the steps through sigma by the same mechanism, and
  E2/E4 probe that mechanism with an existing datapoint behind them. If
  shift proves out, `simple` is the gentler way to get it and becomes the
  obvious follow-up; if shift is flat, the schedule almost certainly is too.
- **`strength: [.5, 1, 1, 1, 1, 1]`** — step 1, the one cell of 1a's
  single-step grid never filled. It is the last strength lever on the yoke,
  but it is also the structure step and the most likely to cost the pose,
  and E1 already takes step 2 plus the tail. Worth a run only if E1 lands
  short of clean and E5/E6 are flat.

### Overlap with section 4

That run answers a different question (the anchor drift, on `cyber_6f`
rather than this subject) but its output is a `colmap_intermediate` like
any other, so read it with the same script. A bleed number for the
reference dataset is the one calibration point none of these runs provides,
and it is free if section 4 is in the same pod session.

## Local reproduction recipe (everything before the Wan denoise)

The first 14 steps run in ~90 s on the 4070 Ti inside
`b2c/pipeline:latest`; only the 14B denoise is out of reach here (29 GB
RAM, 12 GB VRAM). RMBG-2.0 over 81 frames OOMs at 12 GB.

    S=<scratch dir>; R=/home/tristan/Projects/b2crunner
    docker run --rm --gpus all --user $(id -u):$(id -g) \
      -v $HOME/.cache/huggingface:/data/hf_cache \
      -v $HOME/.cache/torch:/data/caches/torch \
      -v $S/data/models:/data/models -v $S/data/output:/data/output \
      -v $S/data/logs:/data/logs -v $S/data/tmp:/data/tmp -v $S/data/uploads:/data/uploads \
      -v $S/data/caches:/data/caches/nv -v $S/data/caches:/data/caches/xdg \
      -v $S/data/caches:/data/caches/triton -v $S/data/caches:/data/caches/mpl \
      -v $R/pipeline:/opt/b2c_runner/pipeline:ro \
      -v $R/docker/envs.docker.yaml:/opt/b2c_runner/pipeline/envs/envs.yaml:ro \
      -v /home/tristan/datasets/isolated:/data/in:ro -v $S:/data/probe \
      b2c/pipeline:latest /opt/venv_main/bin/python -m pipeline.cli run \
        /data/probe/anchor_probe2.yaml --reference-image /data/in/00.png \
        --prompt "$(cat /home/tristan/datasets/isolated/00.txt)" \
        --out /data/output/probe2 --run-name probe --no-wait-for-models

Two traps: mounting `pipeline/` over the image hides the image's env
registry, so `docker/envs.docker.yaml` must be mounted back over
`pipeline/envs/envs.yaml`; and the probe YAML is `fast_helical_native`
truncated after `reinject_anchor_initial` (a copy is at `docs/anchor_probe2.yaml`) with a `save_dataset` step
before the warp (`pre_inject/`) and one after (`initial/`), its
`settings` reduced to the globals those steps read and `outputs: []`.
`--param reconstruct_body.fov_estimator=` gives SAM-3D-Body's default
focal. The saved `pre_inject/frame_00001_.png` alpha is the mesh
coverage at the anchor camera; `initial/frame_00001_.png` is the warped
photo.
