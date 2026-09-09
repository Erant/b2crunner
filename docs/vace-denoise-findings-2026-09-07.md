# The VACE denoise: skeleton bleed, the sampler, and the anchor drift

Findings from 2026-09-07/08, written so the next pod run starts from them.
Three things, in the order they were found. Sections 1 and 3 are
diagnoses; section 2 is a change already in the working tree. Section 4
is the one pod run that settles section 3's open question, and section 5
the minimal set that settles section 1's — section 1 stopped being
qualitative on 2026-09-08, when eleven runs made it a number. Section 6,
added the same evening once section 5 had run, turns to pass 2 and to
the quality the clean frames gave up.

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
wasted. (As of 2026-09-08 that output is gone: those frames are part of the
debug bundle, `export_debug`, and the sidecars set that instead.)

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

### Results, 2026-09-08 evening — the sweep is settled

All six ran to completion (81 frames, `ply/`, `alignment.json`, both
debug datasets), and `log.txt` confirms each got exactly its sidecar's
values. Same script, same references, `flow` = alignment iteration 1:

| run | pass 1 | leak | yoke | cover | flow | IoU |
|---|---|---|---|---|---|---|
| A 31a008 | `[1, 1, .5, .5, .5, .5]`, shift 2.5, euler | 1.74 | 14.5 | 0.21 | 0.986 | 0.8525 |
| B 467c17 | `[1]*6`, shift 3, euler | 16.62 | 55.6 | 0.58 | 1.039 | 0.8501 |
| E1 | A, step 2 -> .5 | **-0.46** | **0.2** | 0.14 | 1.154 | 0.8392 |
| E2 | A, shift 5 | **-0.44** | **-0.3** | 0.15 | 1.165 | 0.8389 |
| E3 | A, `sampler_high: uni_pc` | 12.16 | **116.3** | 0.34 | 1.070 | 0.8642 |
| E4 | A, shift 5 + uni_pc | **-0.53** | **-0.1** | 0.15 | 1.189 | 0.8468 |
| E5 | B, deep layers off | **105.07** | 161.2 | 0.86 | 1.213 | 0.8196 |
| E6 | B, shallow layers off | 0.45 | 0.9 | 0.12 | **4.725** | **0.5721** |

`cover` bottoms out at 0.12-0.15 in every clean run, E6 included, so that
is the metric's floor and not residual ink. The contact sheets agree with
every row: E1/E2/E4 have no visible sticks on any of frames 1/21/41/61,
E3 has a bright red V across the shoulders on the back view and a red bar
over the face in profile, E5 has the whole skeleton painted as green,
yellow and red lines, and E6 shows the frontal reference photo from every
camera. The pass-2 output (`colmap_preupscale/`) inherits all of it: A, B
and E3 still carry red shoulder streaks in their final frames, E1/E2/E4
do not — the bake-in through the intermediate splat is confirmed.

**E1 clears the bar.** Leak -0.46, yoke 0.24: the two levers add, and
the yoke — 3-4x the rest in every earlier run — is gone with them. It is
the criterion set above and by that criterion the sweep is over.

**The 2x2: shift is the whole of 9e315f, and the sampler is a trap at low
shift.**
- Shift 5 alone (E2) reaches zero, yoke included. 9e315f's clean frames
  were the shift; `splat_inactive_mask` needs no seventh run.
- `uni_pc` on the high-noise expert at shift 2.5 (E3) is the worst yoke
  ever measured — 116, twice B's, on top of A's low-noise taper. So the
  taper that took A ten-fold below 0281b9 works under `euler` and barely
  at all under `uni_pc`: at low shift the sampler decides. The plausible
  mechanism is that UniPC's multistep history carries what the two
  high-noise steps committed straight through the hand-off, ink included,
  while Euler's memoryless steps let the low expert repaint it; that is a
  reading, not a measurement.
- At shift 5 the sampler stops mattering: E2 (euler) and E4 (uni_pc) are
  both clean. E4 keeps the most silhouette of the clean three (0.8468,
  0.6 pt under A) at the cost of the most flow (1.189).

**The depth axis separates — the wrong way round for what was wanted.**
The two tapers did NOT both lower the leak, so the layer sum is not what
matters; position in the stack is. But the shallow layers (0-10) carry
the pose AND the ink together: with them alone (E5) the pose survives
(IoU 0.82) and the skeleton arrives as literal paint on every limb (cover
0.86 — six times B's leak); with them gone (E6) the control is ignored
outright (IoU 0.57, flow 4.7, a frontal figure in every view). The deep
layers on their own hold no pose; their job, reading E5 against B, is to
make the hint be *understood* rather than copied. Neither end gives ink
reduction at pose authority, which is the only thing this axis was for.
Something like `[.5, .5, .75, 1, 1, 1, 1, 1]` is the one untested shape
left — E6's cliff at 0 says nothing about 0.5 — but it is a long shot and
the time axis already has three clean settings. Park it.

**What the clean frames cost.** 1a's reading that a softer control is
free on consistency does not hold at this level. All three clean runs sit
at flow 1.15-1.19 against A's 0.986 and B's 1.039 (+17-20%), and 0.6-1.4
points of silhouette IoU under A. Those are alignment-loop pixels of view
disagreement, still around one pixel mean. One confound on the flow
number: A, B and the other morning runs had `refine_cameras_final`
REFUSED (trap 5, the DB-order abort fixed in `d83499c`) and trained the
final splat on the given helix cameras, while E1-E6 ran on the fixed
image and got refined ones — so A-to-E is across an image change, and
within the E family E3 (1.070) < E1 < E2 < E4 (1.189) is the clean
ordering: flow rises as the control softens. What the final splat makes
of it is in section 6's baseline table: the clean runs' heads are
SHARPER (splat head s1 29.5-32.1 against A's 29.3) and their fidelity
0.8-1.4 dB LOWER (25.1-25.8 against 26.6 dB) — some of that the
refinement change, some the softer control.

**Recommendation: E4's settings as the new pass-1 default**
(`sampler_high: uni_pc`, `sampler_shift: 5`,
`strength: [1, 1, .5, .5, .5, .5]`), E1 as the runner-up. *Applied to
`denoise_pass1` (and the re-outline denoise, which mirrors it) on
2026-09-09; the two caveats below stand, and the missing corner is still
unrun.* E4 leaves the
structure steps at full scale, keeps the reference graph's sampler, holds
the best silhouette of the three, and — the deciding point — sits in the
regime where the sampler does not matter, while E1 at shift 2.5 is one
sampler flip away from E3's 116. Two caveats before it ships:
- **The shipped default has never been measured.** Every one of the 17
  runs deviates from `[1, 1, .75, .5, .25, 0]` at shift 8 / `uni_pc`;
  the bleed it was blamed for was seen on pre-2026-09-07 runs at
  `flow_shift` 3.0. If shift 5 erases ink, shift 8 may too, and the
  shipped default's only fault might be its taper on steps 5-6. One run of
  the workflow with NO overrides belongs beside E4 before the default
  moves — it is the missing corner.
- **Pass 2 was pinned at shift 2.5 / euler / flat 0.8 in all six**, on
  purpose. Moving pass 1 to shift 5 without deciding what pass 2 does is
  half a change; the pass-2 output above says pass 1's cleanliness is
  what pass 2 inherits, so pass 2 probably needs nothing, but that is
  inference.

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

## 6. The quality sweep: pass 2

Section 5 bought clean frames and paid in silhouette and view agreement.
This batch asks for quality back, and it asks it of the pass nothing has
ever varied on its own: `denoise_pass2`, the one that paints the frames
the deliverable is trained on. Six sidecars under **`docs/quality-sweep/`**
(README has the zip recipe), all dry-run through `read_settings_sidecar` +
`_refuse_unknown_overrides`.

### What pass 2 does, measured

A new reader, `scripts/final_splat_quality.py`, renders two things per
archive with the local b2ctrain rasteriser: the final `.ply` at its own
training cameras (sharpness and PSNR of the deliverable), and the
intermediate splat at the pre-upscale cameras with `rerender_splat`'s
confidence gate — a reconstruction of pass 2's control video, which the
archive does not carry. Over the 2026-09-08 archives:

- **Pass 2 at flat 0.8 / shift 2.5 adds body texture and leaves the head
  where the control had it.** On kept pixels, control vs output head s1 on
  E4 is 36.0 vs 34.8; body 28.1 -> 31.2. (Measured naively the control's
  head reads 44.5 — the gate's culled holes have hard grey edges that score
  as detail. The reader excludes them.)
- **Pass 2 is where the frames' agreement is decided, and letting go costs
  it.** ef13a7's pass 2 tapered to 0 on the last step: the sharpest frames
  in any archive (body s1 51.9 against ~31) and a final splat head NO
  sharper than anyone's (27.9 against 29.3-32.1) at the worst flow (1.376)
  and 25.2 dB. The alignment guide's thesis — the fit averages
  disagreeing texture into blur, consistency binds — seen on this
  pipeline, once. The shipped pass-2 default tapers to 0 the same way and
  has never been scored (every measured run overrode it).
- **The final splat is at roughly 0.75 of its frames' head sharpness**
  (29.5 / 39.3 on E4) and 25-27 dB fidelity, run to run. That ratio is the
  room pass 2 has to give back.
- Pass 2 has only ever run at shift 2.5 (A family) or 8 (B family,
  confounded by pass 1), flat 0.8 or a taper to 0, euler on the high
  expert. The shift, the strongest lever section 5 found, is untested on
  it.

Baselines for the batch are in `docs/quality-sweep/README.md`.

### The matrix

Base = E4 (02ff74). All six leave pass 1 alone, so their intermediate
splat and pass-2 control should be E4's frame for frame — a determinism
check for free, and what makes each pass-2 delta single-variable.

| # | pass 1 | `denoise_pass2` | what it answers |
|---|---|---|---|
| F1 | E4 | `strength: [1]*6` | hold the control tighter: flow down, and does the deliverable follow? |
| F2 | E4 | `strength: [.6]*6` | let go, mildly: 0.6 / 0.8 / 1.0 with E4 and F1 is the strength curve |
| F3 | E4 | `sampler_shift: 5` | shift on pass 2, the sweet spot pass 1 found |
| F4 | E4 | `sampler_shift: 8` | shift on pass 2, the step default; 2.5 / 5 / 8 is the shift curve |
| F5 | E4 | `sampler_high: uni_pc` | the sampler E3 showed hugs the control hardest, on a control with nothing to leak |
| F6 | E4 | `steps_high: 3`, `steps_low: 3` at shift 5 | the expert split, 2/4 since the port and never chosen; 3/3 at shift 5 is the one that matches the checkpoint's t = 875 boundary; reads against F3 |

### Reading it

    scripts/skeleton_leak.py --iou --by-hue <result-dir> ...
    scripts/final_splat_quality.py <result-dir> ...

- **A pass-2 setting wins when the final splat's head s1 and PSNR rise
  together** against E4's 29.5 / 25.75 dB. Sharper frames with falling
  PSNR and rising flow is ef13a7 again and loses, however good the frames
  look. Each reads against E4 on one axis: strength (F1, F2), shift (F3,
  F4), sampler (F5) — and F6 reads against F3, the split being the only
  difference between them. F6 is not at E4's shift 2.5 because a third
  high-expert step there would sit at t = 868, below the 875 the expert
  was trained down to; at shift 5 the six steps are 1000, 980, 929 | 833,
  655, 334. The same table says the 2/4 split every run has used hands
  the low expert a step above its boundary at shift 5 (929) and two at
  the graph's shift 8 (955, 889), so F6 is also the first run whose
  experts each see only the range they were trained on.
- **All six runs' pass-1 leak must equal E4's (-0.53) and their
  `colmap_intermediate` frames E4's.** If they do not, the runs are not
  deterministic at fixed seed and every single-variable delta in sections
  5 and 6 carries that noise; measure it before believing anything under
  a point of s1.

### Results, 2026-09-08 night — strength is the only lever, and it is a dial

All six ran to completion (81 frames, `ply/`, `alignment.json`, both
debug datasets, and `refine_cameras_final` ran on every one — no trap 5),
and `log.txt` resolves exactly the sidecar's pass-2 values with pass 1 =
E4 on all six. Same two scripts, `flow` = alignment iteration 1.

**The determinism check: settled structure, fresh texture.** Pass 1
reads as E4's on every metric — leak -0.52..-0.59, yoke -0.09..-0.17,
cover 0.152..0.155, silhouette IoU 0.8451..0.8471 — but the frames are
not E4's: `colmap_intermediate` differs from E4's at 26-28 dB (mean 7-8
levels, frame 1 at 31-34 dB), while the control drawings in
`denoise_pass1_input` differ at only 50-59 dB (under a level: the splat
trainer and rasteriser) and every log shows `seed = 0`. So the denoiser
turns a sub-level input difference into a different texture draw on the
same structure. The seven runs therefore share settings up to pass 2 and
are a free noise floor for the pass-2-independent columns: the
reconstructed control's head s1 spans 35.6-39.2 (mean 37.3, sd 1.2) and
its body s1 28.1-28.9 (sd 0.3). **A head-s1 delta under ~2.5 or a body
delta under ~0.7 is inside the noise.** No run has ever been repeated
whole, so PSNR and flow have no measured floor; where they matter below,
a three-point monotone curve is the evidence, not a single delta. This
also means section 5's flow spread among E1/E2/E4 (1.154-1.189) may be
noise; its leak deltas (10-100x) are not.

| run | `denoise_pass2` | frame head s1 | **splat head s1** | splat s1 | **PSNR** | p2 head / ctl head | p2 body / ctl body | p2 PSNR vs ctl | **flow** | BA infl. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E4 02ff74 | flat 0.8, shift 2.5, euler, 2/4 | 39.3 | 29.5 | 27.1 | 25.75 | 34.8 / 36.0 | 31.2 / 28.1 | 21.1 | 1.189 | 0.028 |
| F1 11ffc1 | `strength: [1]*6` | 37.4 | 28.7 | 26.5 | **26.29** | 35.2 / 37.9 | 28.7 / 28.6 | 21.9 | **1.007** | **0.006** |
| F2 0b6285 | `strength: [.6]*6` | **40.2** | 29.2 | **28.7** | **24.77** | 37.9 / 36.6 | 36.2 / 28.9 | 18.8 | **1.389** | 0.067 |
| F3 792cf0 | `sampler_shift: 5` | 38.2 | 28.6 | 28.0 | 25.64 | 37.2 / 37.3 | 33.6 / 28.7 | 20.8 | 1.215 | 0.072 |
| F4 5d3064 | `sampler_shift: 8` | 36.1 | **26.3** | 26.3 | 25.84 | 36.8 / 35.6 | 34.3 / 28.1 | 20.2 | 1.231 | 0.029 |
| F5 7b5a31 | `sampler_high: uni_pc` | 36.2 | **26.7** | 26.2 | 25.14 | 38.6 / 38.4 | 34.3 / 28.5 | 19.9 | 1.290 | 0.049 |
| F6 778eeb | 3/3 at shift 5 | 38.4 | 28.6 | 27.5 | 25.17 | 34.5 / 39.2 | 29.1 / 28.4 | 21.0 | 1.206 | 0.037 |

**Nobody clears the bar.** Every run's final head s1 is below E4's:
F1/F2/F3/F6 by 0.3-0.9 (noise), F4/F5 by 2.8-3.2 (real). No pass-2
setting made the deliverable's head sharper; two made it softer.

**Strength is a dial, and it turns the same way on three independent
instruments.** 0.6 / 0.8 / 1.0 (F2 / E4 / F1): PSNR 24.77 / 25.75 /
26.29 dB, flow 1.389 / 1.189 / 1.007, `ba_scale_inflation` 0.067 / 0.028
/ 0.006 — while the frames' head sharpness runs the other way, 40.2 /
39.3 / 37.4, and the pass-2 body against its control 36.2 / 31.2 / 28.7
over a control of ~28.5. Fidelity and agreement are bought with frame
texture, point for point.
- F1's flow 1.007 is A's 0.986: the 17-20 % view agreement that section
  5 said clean pass-1 frames cost is recovered entirely by holding pass
  2 at 1.0. And at 1.0 pass 2 adds nothing to the body (28.7 vs 28.6)
  and takes a little from the head (35.2 vs 37.9): it is a re-render
  with a small blur, which is exactly what a consistent set of frames is.
- F2 is ef13a7 in miniature: sharpest frames, sharpest body splat
  (28.7, the one body number outside the noise), worst PSNR and flow
  of the batch, and it invents — a tile grid and ceiling lights in the
  background, red eyeshadow on frame 41 — because 0.6 leaves the model
  room to paint what the control did not say.

**The sampler axis is dead or harmful on pass 2.**
- Shift 5 (F3): PSNR level, body +0.9 (noise edge), head level, and the
  highest BA inflation of the batch (0.072). Nothing bought.
- Shift 8 (F4), the step's default: head s1 -3.2 (outside noise) for
  +0.09 dB. The shipped pass-2 shift costs the head and buys nothing.
- `uni_pc` high (F5): head -2.8, -0.6 dB, flow +8 %, and the hardest
  hallucinated tile grid of the seven. The sampler that hugs the control
  hardest on pass 1 hugs a splat render on pass 2, and a splat render
  has no detail to hug; hugging it harder produced the least faithful
  deliverable. E3's lesson, in reverse.
- 3/3 at shift 5 (F6, against F3): head level, -0.47 dB, flow level.
  Each expert on its trained t-range buys nothing here; the 2/4 split
  stays.

**Where the head's detail actually is.** The final head sits at
0.75-0.78 of the frames' head sharpness in every run, and in every run
pass 2's head is at or below its control's (the one exception, F2's
+1.3, is noise plus invention). Pass 2 does not put detail into the head
because its control has none to give beyond what the intermediate splat
held, and that splat's head is the face cap's render. So the head
sharpness of the deliverable is decided upstream of pass 2 — and the
face-protecting mask below (keep the cap's pixels, re-synthesise the
body only) is the one pass-2 change left that can move it.

**Recommendation.** `denoise_pass2: strength: [1]*6` at euler / shift
2.5 / 2|4 (F1), on E4's pass 1: +0.5 dB, flow -15 %, BA inflation ÷ 5,
head within noise of E4, and the deliverable the fit can hold. If texture
in the frames is preferred over fidelity, 0.8 is the compromise and 0.6
is the wrong side of the curve. *Applied 2026-09-09 as the texture
compromise: `denoise_pass2` ships euler / shift 2.5 / 2|4 with a flat
`[0.8]*6` — E4's own pass 2, the corner every F run was measured
against — not F1's 1.0.*
The shipped pass 2 (uni_pc, shift 8, taper to 0) stacks the three
settings that each individually lost here or in ef13a7; it should not be
left as the default whatever is chosen.

**The one run this batch could not make: a repeat.** A second F1 (or
E4) under identical settings is the only way to put a floor under PSNR
and flow; one slot, and every future single-variable claim on those two
columns reads against it.

### Not in the batch, and why

- **The shipped pass 2 as-is** (uni_pc high, shift 8, `[0.8, 0.8, 0.6,
  0.4, 0.2, 0]`). Never scored, but never chosen against a measurement
  either; its three departures from E4 are each covered by an axis run
  here (F5, F4, and the taper-to-0 ef13a7 already paid for), so a run
  confirming their sum would say less than the parts.
- **A face-protecting VACE mask on pass 2** — `control_masks` at 0 over
  the face cap's footprint so the photograph's face is kept rather than
  re-synthesised at ~100 px. The most promising quality lever there is,
  and a code change (mask_splat writes all-1.0 masks by design), not a
  sidecar. Next after this batch if F1 shows the control's detail is worth
  holding.
- **Pass 1 at `[1]*6` under shift 5** — whether the low-noise taper is
  still needed once the shift erases the ink, which would return the 1.4
  points of silhouette the taper cost. Dropped from this batch to keep all
  six on pass 2; still the one pass-1 run worth making.
- **`steps_low: 8` on pass 2** — plausible, second-order, and it changes
  the LoRA's operating point. After the shift and strength curves exist.
- **Resolution** — already `[720, 1280]`, the largest the model was
  trained for; there is no larger to buy.

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
