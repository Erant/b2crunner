# The final splat: alignment loop and training settings

What to change in `train_final_splat` and why, from a measurement session on the
refinesplat rig (2026-09-04 → 2026-09-06, RTX 4070 Ti, 81 helical views at
1080×1920). Full evidence: `~/Downloads/refinesplat/FINDINGS.md`; running log
`research/ppd_log.md`; the "should this live inside brush" question is settled in
`research/fold_alignment_into_training.md` (answer: no).

Three changes, in order of leverage-per-effort. The first is one parameter. The
second is one parameter and a size trade. The third is new code, and it is the
one that matters.

---

## 0. The problem being solved

The final `.ply` is markedly softer than the frames it was trained on. Measured:
the trained splat's face scored 52.6 raw Laplacian variance against 400–770 for
the SeedVR2 frames it fit, and ~75–82 for the *un-upscaled* 720p frames. **The
fit is what destroys the detail, not the upscaler.** Swapping SeedVR2 for a
better model cannot fix this and was tried (it made things worse — see §5).

The mechanism is local texture disagreement between views. Dense optical flow
from each training frame to the splat's own render at that camera gives a
*whole-region* rigid shift of ~0.5 px — poses are fine — but a *per-pixel* flow
of 1.7–3.6 px mean, p90 up to 8 px. Each generated frame renders skin, hair and
finger micro-texture in a slightly different place. brush's photometric loss
averages those disagreeing textures into a blurred consensus.

That framing is the whole strategy: **the upscale is a texture-registration
problem, not a per-frame quality problem.**

The mechanism was proved directly. Unsharp-masking the splat's own renders and
retraining on them — perfectly view-consistent by construction, no new
information — moved face sharpness 91 → 228. brush absorbs *consistent* detail
almost completely. Consistency, not capacity, is the binding constraint.

### The numbers to beat

Band-limited sharpness (`s1`, Laplacian variance after a σ=1 blur — the
grain-insensitive metric; raw Laplacian variance is dominated by resampling
noise and should not be compared across anything that resamples). Face crop via
Sapiens2 seg. "novel" = interpolated in-between cameras, which is what catches
detail baked into per-view SH rather than geometry.

| configuration | face s1 | novel | splats | .ply |
|---|---:|---:|---:|---:|
| pipeline today | — | 16.4 | 363k | 92 MB |
| + normal loss off | 21.1 | 20.0 | 356k | 90 MB |
| + 4 alignment iterations | **23.8** | **22.6** | 356k | **84 MB** |
| + dense growth (instead of alignment) | 23.9 | — | 1.68M | 424 MB |
| + dense growth **and** alignment | **26.3** | **24.2** | 1.68M | 424 MB |
| + one PPD refinement pass (§5) | 32.5 | 29.4 | 1.68M | 424 MB |

For scale: the *source frames themselves* score 33.9. The pipeline today
delivers about half the sharpness present in its own training data.

---

## 1. Turn off normal-map supervision

**Change:** `train_final_splat.params.normal_loss_strength: 0.0`
(default is `0.05`, `pipeline/steps/brush.py:575`).

The cleanest control, run on identical aligned images:

| normal supervision | face s1 | novel | fidelity |
|---|---:|---:|---:|
| off | **26.5** | **24.4** | **29.3 dB** |
| pipeline's, weight 0.05 | 21.8 | 20.2 | 27.5 dB |
| recomputed from the aligned images | 22.0 | 20.4 | — |

It costs 4.7 s1 (−18%) **and** 1.8 dB of fidelity. Both, which is the tell: this
is not a sharpness-for-accuracy trade, it is a straight loss.

The third row matters for anyone tempted to fix it rather than remove it. The
suspicion was that Sapiens2 normals predicted on *generated* frames are
inconsistent between views, and that recomputing them from the aligned images
would rescue the term. It does not — 22.0 vs 21.8 is noise. **The harm is the
loss term itself, not the quality of the maps feeding it.** Verified visually:
no floaters, no degenerate geometry with it off.

Note the flag is only emitted when the `normal_maps` input is wired
(`brush.py:772`, `if normal_maps is not None`). `train_final_splat` does wire it
(`scene.export_normal_maps`), so setting the strength to 0 passes
`--normal-loss-weight 0` and is sufficient. Leaving the input wired keeps the
`normals/` sidecar available for `evidence_normal_weight`.

This says nothing about the *stage-2* training (`train_splat`), where the
workflow's own comment says normal supervision helps. Only the deliverable.

---

## 2. Dense growth — now optional

**Change:** the growth knobs are not currently exposed on `BrushStep`. Adding
them means three `Param`s and three lines in `command()` (`brush.py:733`):
`--growth-grad-threshold 0.0012 --growth-select-fraction 0.4
--growth-stop-iter 24000`.

This was the second-biggest lever found (+40% face on its own) and the original
recommendation was to adopt it. **The alignment loop has since made it
optional**, which is a better outcome than it sounds:

| | face s1 | splats | .ply |
|---|---:|---:|---:|
| dense growth alone | 23.9 | 1.68M | 424 MB |
| 4 alignment iterations alone | 23.8 | 356k | **84 MB** |
| both | 26.3 | 1.68M | 424 MB |

23.8 vs 23.9 is inside the ~0.3 s1 run-to-run noise floor. **Alignment alone
buys dense growth's entire quality gain at one fifth the asset size.** They do
stack, so dense growth is not wasted — but it is now a deliberate
quality-for-size purchase (4.6× the `.ply`) rather than a prerequisite.

Recommendation: **ship alignment first, without dense growth.** Add growth later
if the extra 2.5 s1 is worth 340 MB per asset.

One earlier finding here has been retracted, and it matters if you read FINDINGS
directly. That document reports that densification does *not* stack with
alignment (21.7 with growth vs 22.1 without, on warped images) and attributes it
to Lanczos-resampled frames having weaker gradients than sharp originals. **That
generalisation is wrong.** Re-tested against frames aligned to a *converged*
render rather than a blurry one:

| alignment target | no growth | dense growth |
|---|---:|---:|
| a blurry cold-start render | 22.1 | 21.7 (−0.4) |
| a converged, looped render | 22.5 | **24.7 (+2.2)** |

Growth stacks fine. The original negative was a property of the *target* the
flow was computed against, not of resampling: flow measured against a blurry
render is noisy, and densification amplifies that noise into splats. Align
against something converged and the gradients are clean enough to grow on.

Dense growth still belongs in the **cold start**, before any alignment
iteration, which is where the loop below puts it.

---

## 3. The alignment loop

The actual new capability. Per iteration:

```
render the current splat at the 81 training cameras     (~4 s, GPU)
  → DIS optical flow: each ORIGINAL frame → its render
  → smooth the flow (σ 6 px), cap it (6 px), mask to alpha
  → Lanczos-warp the ORIGINAL frames onto the render     (~60 s, CPU)
  → warm-start fine-tune from the current .ply, growth off
  → repeat
```

Measured trajectory, four iterations from the normals-off cold start:

| iteration | face s1 | Δ | novel | fidelity |
|---|---:|---:|---:|---:|
| start | 21.1 | — | 20.0 | — |
| 1 | 22.3 | +1.2 | 21.1 | 27.64 dB |
| 2 | 22.9 | +0.6 | 21.8 | 28.00 dB |
| 3 | 23.4 | +0.5 | 22.2 | 28.20 dB |
| 4 | **23.8** | +0.4 | **22.6** | **28.41 dB** |

Saturates by iteration 4; a fifth and sixth move it +0.2 and +0.2. **Four is the
right number.** The same loop on the dense splat runs 23.9 → 26.3 with the same
curve shape, so the behaviour is not specific to one splat size.

Two properties worth internalising before implementing:

- **Novel views gain as much as training views.** The ratio holds at ~0.91
  throughout. This is real geometry improving, not per-view SH memorisation.
- **Fidelity rises while sharpness rises** (27.64 → 28.41 dB). That is the
  signature of *recovered* detail. Any implementation where sharpness climbs and
  fidelity falls is doing something else — see §5.

### 3.1 The invariant that makes it safe

**Every iteration warps the pristine originals toward the current render. Never
warp a warp.**

This is a many-to-one contraction: all views are pulled onto one consensus, and
the consensus is anchored because the splat must still explain the mean image.
Iterating warps-of-warps is pairwise merging, which has unbounded drift. Keep
the originals; write each iteration's warped set to a fresh directory, or
restore the originals before each warp.

A corollary that surprises people: **the accumulated gain lives in the splat,
not in the images.** Each iteration's image set is only ever "originals warped
once". There is no frame set anywhere representing four iterations of
improvement.

That does not make the frames worthless as a standalone artifact — it caps what
they can carry. Measured directly, by aligning the originals to the *finished*
loop's render and then cold-starting from scratch on them (no warm start, no
loop):

| | face s1 | novel | fidelity |
|---|---:|---:|---:|
| cold start, unaligned frames | 21.1 | 19.8 | 26.73 dB |
| cold start, frames aligned to the finished render | **22.5** | **21.2** | **28.10 dB** |
| the loop itself | 23.8 | 22.6 | 28.41 dB |

**About half the loop's gain (+1.4 of +2.7 s1) ships in the frames**, along with
most of the fidelity improvement. So a consumer that cannot run the loop — a
downstream reconstructor, a different engine, an external client — can be handed
a corrected frame set and get half the benefit for free. The other half needs the
warm-start fit and cannot be exported.

If you do this, harvest the frames from the loop's **last** iteration: aligning
to a converged render beats aligning to a blurry one (22.5 vs 22.1), which is the
same effect as the growth result in §2.

### 3.2 Where it goes

**Recommended: inside `BrushStep`, as an `align_iters` parameter.**

The reason is `brush.py:672` — the COLMAP export is built into a
`TemporaryDirectory` and deleted on the way out of `run()`. A workflow-level
loop (render_splat → align → brush, four times over) would re-export the whole
dataset every iteration and needs a way to hand brush an `init.ply`, which
`BrushStep` has no input for. Inside `run()`, the directory is already there and
already correct.

Almost everything needed already exists, because `polish_steps` is the same
mechanism:

| need | already in the repo |
|---|---|
| warm start from a `.ply` | `_link_init_ply()`, `brush.py:456` |
| growth-off fine-tune argv | `command(total=…, refine=_NO_REFINE, growth_stop=0)`, `brush.py:733`, as used at `:823` |
| tolerant exit handling | `_run_brush()`, `brush.py:834` |
| render at the dataset's own cameras | `render_splat` with `pattern: ""` — "empty reuses the source dataset's cameras verbatim" (`steps/splat.py:236`) |

The only genuinely new code is the warp itself. Port
`~/Downloads/refinesplat/tools/warp_align.py` (~80 lines of OpenCV:
`DISOpticalFlow`, Gaussian-smooth the field, magnitude cap, alpha mask, remap).

Shape, slotting in between the main training and the existing polish block:

```python
self._run_brush(command(total=total_steps, refine=refine_every,
                        normal_start=normal_loss_step_start),
                ply_path, colmap_dir=colmap_dir)

for _ in range(align_iters):
    renders = _render_training_views(ply_path, colmap_dir)   # brush-splat-render
    _warp_toward(colmap_dir / "images", pristine_originals, renders,
                 sigma=align_flow_sigma, cap=align_flow_cap)
    _link_init_ply(colmap_dir, ply_path)
    self._run_brush(command(total=align_steps, refine=_NO_REFINE,
                            normal_start=0, growth_stop=0),
                    ply_path, colmap_dir=colmap_dir)

if polish_steps > 0:      # unchanged
    ...
```

New params:

| param | default | notes |
|---|---|---|
| `align_iters` | `0` | 0 = off. 4 is the measured saturation point. |
| `align_steps` | `3000` | Per-iteration fine-tune length. See §3.3. |
| `align_flow_sigma` | `6.0` | Flow smoothing, px. |
| `align_flow_cap` | `6.0` | Max applied displacement, px. |

### 3.3 Cost

Fine-tune time is linear in iterations with negligible fixed overhead — the
444 MB `.ply` load and 81 PNG decodes cost under ~5 s, measured. It scales with
*splat count*, not file count: ~47 ms/iteration at 1.68M splats, ~15 ms at 356k.

Per alignment iteration on a 4070 Ti, sparse splat: ~4 s render + ~60 s flow +
~47 s fine-tune ≈ **110 s**. Four iterations ≈ **7.5 minutes**, on top of the
~8–10 min cold start. On the dense splat it is ~14 minutes.

**The flow is the dominant cost and it is single-threaded.** DIS is
embarrassingly parallel across the 81 frames; a `multiprocessing.Pool` around
the per-frame body should take 60 s → ~10 s and cut the loop roughly in half.
This has not been done and is the cheapest available win.

`align_steps: 1000` instead of 3000 is measured to hold the fixed point exactly
(47 s → 15 s per iteration) — **but only at the fixed point.** Every iteration
that is still climbing was measured at 3000, and the one loop tested from an
under-trained start was still rising at iteration 4. Start at 3000; drop to 1000
only for iterations past saturation, or after measuring it on a climbing loop.

### 3.4 Sharp edges

1. **The polish path turns normals back on.** `brush.py:823` passes
   `normal_start=0`, i.e. normal supervision from step 0 of the polish run. That
   is deliberate for polish, and wrong for an alignment iteration. Setting
   `normal_loss_strength: 0.0` (§1) makes it moot, but if anyone re-enables
   normals for the final splat, the alignment iterations must still pass weight 0.

2. **clap refuses repeated flags.** This is why `command()` is a function and not
   a list to append overrides to (`brush.py:733` says so explicitly). Any new
   invocation must go through it.

3. **Interpolation must be Lanczos, never bilinear.** A single 0.5 px bilinear
   resample of undamaged face crops drops their score from 738 to 134; Lanczos
   drops it to ~480. An early "alignment makes it worse" result was entirely this
   bug. Use `cv2.INTER_LANCZOS4`.

4. **Decide what the evidence pass measures.** `export_evidence` runs after the
   last step of each invocation, against whatever is in `colmap_dir/images` — which
   after alignment is the *warped* set, not the originals. Probably correct (it
   matches what the splat was fitted to), but it should be a decision, not an
   accident, since `rerender_splat` gates on those `ev_*` properties.

5. **The brush binary must include `f80b23b8`.** Before that commit, a truncated
   or corrupt `.ply` made `brush-splat-render` spin at 95% CPU forever instead of
   erroring — a disk-full mid-export wedges the pipeline rather than failing it.
   Fixed on branch `normal-map-supervision`, pushed to `Erant/brush`. A loop that
   writes a `.ply` per iteration multiplies the exposure, so confirm the
   production image carries it.

6. **Hands will not improve much.** They gain in the same *ratio* as the face but
   keep visible double contours, and raising the flow cap from 6 to 12 px changes
   nothing. The residual is views disagreeing about hand *pose* — a multi-modal
   geometry problem no image warp addresses. Per-view loss weighting was tried
   and is negative (fidelity collapses where pixels are zeroed). This needs a
   pose-level fix or a body prior; it is out of scope for the loop.

---

## 4. Validation

Reproduce before trusting. In order:

1. **Cold start with normals off** should land near 21.1 face s1 / 20.0 novel. If
   not, the metric or the crop differs from ours before anything else is
   diagnosable.
2. **Sharpness must rise monotonically** across the four iterations, decelerating
   (+1.2, +0.6, +0.5, +0.4 is the measured shape).
3. **Fidelity must not fall.** Ours rose 27.64 → 28.41 dB. Falling fidelity with
   rising sharpness means the warp is running away from the data — that is the
   PPD signature (§5), not alignment's, and indicates a bug (most likely warping
   warps, §3.1).
4. **Check a novel view.** The gain must survive at interpolated cameras. If
   training views sharpen and novel views do not, detail is going into per-view
   SH rather than geometry.

Tooling for all four is in `~/Downloads/refinesplat/tools/` — `eval_splat.py`
(sharpness, both metrics, training and novel views), `fidelity.py` (PSNR/SSIM to
originals after drift compensation), `flow_measure.py` (the disagreement
measurement of §0).

---

## 5. What NOT to do

Recorded because each was tried and cost real time.

**Do not apply diffusion refinement per-view without registration.** Every
attempt was worse than doing nothing: Flux refinement of the training frames
(52.8 vs 52.6 baseline), of the splat's renders (42), and as refined close-up
support views (39.7–45.9 vs 91.2 for no support views at all). The refined crops
measure beautifully *in isolation* — 640 vs 52 — and transfer nothing, because
each view's denoising is an independent stochastic decision about where fingers
and eyelashes go. Measured: Flux-refined hands disagree with their own splat's
render by 6.0–6.8 px of flow versus 2.9 px for the plain SeedVR2 frames.
Per-view refinement is a *larger* source of cross-view disagreement than the
generation it was meant to improve.

**Do not fold the alignment into brush's optimiser.** Learned warp fields,
shift-tolerant losses and per-view 3D deformation were all assessed against the
literature and the source. The published precedent for the in-loss variant
(AligNeRF) measures +0.10 dB PSNR, gradient-based alignment has a ~±2 px capture
basin against measured drift of up to 8 px, and the GPU cost does not go away
when it moves inside the trainer. Full reasoning:
`research/fold_alignment_into_training.md`.

**Optional, later: one phase-preserving-noise pass.** After the loop, a single
structured-noise diffusion step (noise built from the render latent's own phase,
so it is content-locked across views) does survive the fit: 26.3 → 32.5 face s1,
24.2 → 29.4 novel, at a cost of ~2 dB fidelity. It is the only diffusion step
found that works, and it works *because* the views were aligned first. But it
costs ~9 min of GPU, must not be iterated without a fidelity guard (it is a
runaway — sharpness rises ~9 s1 and fidelity falls ~2.4 dB per iteration,
linearly, until the face is visibly hallucinated), and needs the luminance-only
frequency-split merge to avoid colour drift. Recipe in FINDINGS §9.2. **Ship §1
and §3 first.** This is a separate decision with a separate risk profile.
