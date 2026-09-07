# Improving the intermediate splat: findings and implementation guide

> **SUPERSEDED IN ONE PLACE, 2026-09-07: the polish is off.** Every number
> below was measured on the Erant/brush fork, which the pipeline no longer
> uses — b2ctrain replaced it (docs/docker.md). Re-measured there, the
> growth-off 9000-step warm start that §2 and §4 call the biggest lever
> does not improve quality, so `fast_helical_native` sets `polish_steps: 0`
> on both trainings. Everything else in this guide still describes what
> ships.

> **Landed 2026-09-05** (§6's list, all of it, plus the optional item 7).
> brush was already committed and pushed as `a9405881`; `BRUSH_REF` moved
> with it. Three deviations from §6, all deliberate and all asked for:
>
> * **the re-render is made on BLACK** (`cull_color: [0, 0, 0]`), not on
>   the 0.5 grey §4 weighed up. The measurement that grey halves the dark
>   wedges at 8% of the sharpness was about brush's TRAINING background,
>   which is untouched and still black; this is the render's cull colour,
>   and separating it from the colour the frames end on is what the next
>   two points do.
> * **the matte is rmbg's, not the gate's alpha.** A new
>   `resplat_foreground_masks` step runs rmbg over the re-render and
>   `mask_splat_fringes` (new `mode: composite`) lays the subject over 0.5
>   grey with it. The gate still decides what is missing — culled pixels
>   are black, a hole in the subject — and rmbg decides what is subject.
> * **the grid backdrop is off at stage 2 as well** (`background: ""` on
>   `rerender_splat`, matching `render_initial_views` since 09-04). Both
>   denoise prompts still describe the room; that contradiction is left
>   standing on purpose, as it was at stage 1.
>
> Not run on a pod. Every number below is still from the local retrain.

Measured 2026-09-05 on the run `fast_helical_native-20260904-224848-f10b4c` (its
`colmap_intermediate/` export, retrained locally on the RTX 4070 Ti with the pod's exact brush
flags; the local retrain reproduces the pod's sharpness within noise). Everything below is about
the FIRST brush training (`train_splat`) and the helical re-render (`rerender_splat`) that feeds
denoise pass 2. Nothing here changes the final training.

The question: what makes the **novel views** — the 81 helical cameras of
`colmap_preupscale/images.txt` (identical to `debug/refine_cameras_final/given`), spanning
-30..+30 degrees of elevation against a training orbit that spans -0.5..+0.9 — better as
conditioning for the second denoise?

## 1. Results, shortest form

| change | novel-view effect | verdict |
|---|---|---|
| **brush evidence fix** (two bugs in `w_all`) + `--conf-tau 0.3 --conf-angle-margin 45` | the gated re-render stops culling the face (69% -> 8% of face px), hair (59 -> 10%), the silver top (67 -> 3%); 30% of the subject culled -> 6%; silhouette IoU 0.70 -> 0.92 | **do it first; by far the biggest lever** |
| stage-1 shells OFF (`stage1_support_views: false`) | sharper everywhere (face 132/21.8 -> 143/21.8), more confident, 284k splats instead of 735k, half the training time; no chalky streaks at +-30 degrees | **do it; drops a Sapiens2 pointmap pass and 9 shell builds** |
| growth-off polish, one 9000-iter warm start after the 30k | on top of shells-off: face 143/21.8 -> 161/23.0, hands 49/9.0 -> 55/9.3, shoes 31/6.7 -> 36/7.0 | do it (~2 min on the 4070 Ti, less on the pod) |
| normal supervision OFF | worse: IoU 0.935 -> 0.927, band-limited sharpness down in every part, the specular top's confidence collapses (upper cull 8% -> 42%) | **keep normals at stage 1** (the final-training result of 2026-09-04 does not transfer) |
| **optical-flow alignment loop** (3 x align + 3000 warm iters) | face 145/22.6 — the no-align control reached 149/22.9 | **not worth implementing at stage 1** |
| alignment every 5000 iters during growth ("inline") | 136/21.5 — no better than no alignment | no |
| plain 39000-iter cold run | 137/22.1 — a third of the polish gain | no; the polish needs its own warm-start invocation |
| dense growth (0.0012 / 0.4 / stop 24000) | sharpest plain render (172/24.7) but 3x the time and the gate rejects a quarter of it (27% of its splats have in-mask < 0.5) | no |
| SH degree 1 | much worse (110/20.4) | no |
| `--background-color 0.5 0.5 0.5` (train on the cull colour) | dark wedges halved (0.037 -> 0.021 of the silhouette) but -8% band-limited sharpness in every part | trade-off, see 4 |
| `--match-alpha-weight 0.5` | wedges -17%, sharpness unchanged, IoU -0.003 | mild; optional |
| `--evidence-prune-inmask 0.3` | prunes the sub-pixel detail splats instead (face 132 -> 117), wedges unchanged | no |
| cleaned support masks (fringe + checkerboard) | metrics unchanged (130/21.5 vs 132/21.8, noise) | do it as hygiene, see 5 |

**Recommended candidate** (`S0polish9k`, all of the above "do it" items): face 161/23.0, hair
60/8.9, hands 55/9.3, upper 87/13.6, IoU 0.936, gated at the recommended settings 6.4% culled
(face 8%, hair 10%, hands 10%, top 3%), 284k splats, 5 + 2 minutes on the 4070 Ti. The pipeline
today: face 132/21.8, 30% culled (face 69%, top 67%), 735k splats, 10 minutes.
Side by side: `output/isplat/final_today_vs_recommended.png`.

Sharpness = Laplacian variance of the body-part crop of the plain render at the 81 novel cameras,
written `raw / s1` (s1 = after a sigma-1 blur: grain-free, the trustworthy one). Parts come from
Sapiens2 labels of the preupscale frames at the same cameras. "cull" = fraction of the part's
pixels the confidence gate removes. IoU = render silhouette vs the preupscale frame's matte.
Tools: `output/isplat/eval_novel.py`, `gate_sweep.py`; every number is in
`output/isplat/results.log`, the running log in `output/isplat/NOTES.md`.

## 2. The evidence bug (brush), and the gate

What `rerender_splat` hands the second denoise is the confidence-gated render. On this run it
culled 30% of the subject: the face, the hair and the whole silver top came back as 0.5 grey
(`output/isplat/runs/baseline/eval/sheet_gated.png`). The plain render at the same cameras was
fine. Decomposing the confidence terms with `brush-splat-render --conf-*` put it on the `inmask`
term (`w_in / w_all`), for two reasons in `crates/brush-train/src/evidence.rs`:

1. A **masked** support view (the face cap: a close-up whose mask is the face alone) counted
   every splat it could see outside its mask — hair, neck, shoulders, the top — as "drawing over
   background" (`w_all += vis`, `w_in += 0`). Outside a mask is *ignore*, not background.
2. A `weights/` sidecar (face_priority) scaled `w_in` but not `w_all`, so a silenced region
   voted in the denominator at full strength and in the numerator at 0.1.

Fix (in the brush working tree, `~/Projects/brush`, branch `normal-map-supervision`, **not yet
committed**): the third lane is `vis · k · w`, `k` = the mask for masked views (1 for
transparent ones), `w` = the loss weight. Transparent frames keep voting fringe and the dark
wedges out, so those stay culled. `docs/splat-confidence.md` and the bench test
(`crates/brush-bench-test/tests/evidence.rs`: mass conservation now asserted on a transparent
batch, plus a new masked-view test) are updated; `cargo test -p brush-bench-test --test evidence
--test loss_weights` passes 6/6. Re-measured on the very same baseline .ply
(`brush-splat-render --dataset ... --write-evidence`, `output/isplat/reev.sh`):

| gate | culled | IoU | face cull | hair | hands | upper |
|---|---:|---:|---:|---:|---:|---:|
| old evidence, defaults (the pipeline today) | 0.301 | 0.696 | 0.69 | 0.59 | 0.44 | 0.67 |
| fixed evidence, defaults | 0.233 | 0.763 | 0.28 | 0.25 | 0.42 | 0.56 |
| fixed, `--conf-tau 0.3` | 0.090 | 0.899 | 0.16 | 0.14 | 0.16 | 0.08 |
| fixed, tau 0.3, `--conf-angle-margin 45` (S0polish9k) | 0.064 | 0.922 | 0.08 | 0.10 | 0.10 | 0.03 |

What the fix leaves is the `agree` term culling the specular top (tau 0.08 is too tight for a
view-dependent material; 0.3 keeps it, 0.16 is a middle) and the `coverage` term culling the
face at the elevation extremes (the cap views sit in a 30-degree disc; margin 45 lets the +-30
helix through; 60 keeps a little more face and a little more fringe). `gate_lo/gate_hi` are NOT
the lever: 0.55/0.75 culls 43% of the subject to remove a quarter of the dark-fringe pixels, and
no lo/hi separates fringe from subject.

Every number in this document that involves confidence was re-measured with the fixed evidence
(`_ev` rows in results.log). Splats trained after the fix carry it in their `ev_*` block.

## 3. Shells and normals

- **Shells** were built against "chalky zero-parallax streaks at novel elevations" measured at an
  earlier pipeline state (docs/stage1-support-views-implementation.md). With the face cap and the
  evidence gate in place they no longer show: at the +-30 degree extremes the shells-off splat
  renders as cleanly as the baseline (`output/isplat/cmp_elev_baseline_S0.png`), with a higher
  IoU and confidence and 2.6x fewer splats. The one thing the shells still do is fill some of the
  concave gaps (dark wedges 0.037 of the silhouette with, 0.044 without), which the gate handles.
  Turning them off removes the `pointmap_elevation_views` Sapiens2 pointmap pass, 9 shell builds
  and 18 renders per run, and halves the training time.
- **Normals** help here: without them the specular top disagrees between views and loses
  confidence, and every part loses band-limited sharpness (`N0`, `N0S0` rows).

## 4. Alignment, polish, and the dark wedges

**Alignment.** The warm-start optical-flow loop from the final-splat work was rebuilt for stage 1
(`align.py`, `loop_posthoc.sh`, `loop_inline.sh`): render the current splat at the 81 training
cameras, DIS-flow the ORIGINAL frames toward it (sigma 6, cap 6 px, Lanczos), warm-start from
`init.ply` for 3000 iterations with growth off. Three iterations moved the face from 132/21.8 to
145/22.6. **The control — the same three warm starts on the unwarped frames — reached
149/22.9.** The applied flow was only 1.0 px mean (stage-1 frames are one 720p VACE pass, far
more consistent than the upscaled final frames were), and the resample costs slightly more than
the registration gains. Running it during growth (every 5000 iterations, schedules continued via
a `--start-iter` fix in brush) came out at 136/21.5, no better than nothing. So for the
intermediate splat the alignment is not worth implementing.

On "can it run inside training": brush decodes each image once into a packed batch cache and
never re-reads files; the training step already renders the sampled view, but there is no flow
estimator in brush (CPU or wgpu) and no way to write a warped GT back. The restart route (export,
align outside, resume with `init.ply` + `--start-iter`) is the practical form, and with the
schedule fix (train.rs `set_step_count`, also in the working tree) it trains exactly as the tail
of one run; the cost is a dataset reload (~5 s) and an evidence export (~10 s) per cycle. A
middle route — brush invalidating cache entries whose file mtime changed, with the aligner
running beside training against `--export-every` plys — is ~30 lines in `scene_loader.rs` and
was not built, since the measurement says the alignment itself is not the gain.

**Polish.** What the loop was measuring is extra growth-off iterations from a warm start at
full mean-LR. A single 9000-iteration warm start gives the whole effect (148/22.6 from the
baseline; 161/23.0 from shells-off) with no restarts; a plain 39000-iteration cold run gives a
third of it (137/22.1), so the restart matters, not the iteration count.

**Dark wedges** (the black fringes): dark splats in concave gaps (arm/torso, between fingers),
`output/isplat/runs/baseline/rim_zoom_ev.png`. They are not the fit painting background: the
stage-1 orbit is flat, so a splat in a gap is occluded from every training camera or peeks out
only along the silhouette, where it is matched against brush's training background colour —
which is **black** (`--background-color` default; steps/brush.py never sets it), not the 0.5 grey
`cull_color` the re-render composites over. From 30 degrees above, the gap opens. They are 3.7%
of the plain silhouette; the gate removes about half; what was tried at the source:

| | wedges (plain) | sharpness |
|---|---:|---|
| baseline | 0.037 | 132/21.8 |
| training bg 0.5 grey (`BG05`) | 0.021 | 122/20.2, -8% s1 in every part |
| training bg 0.5 grey, noise 0.3 (`BG05N`) | 0.021 | 121/19.9 — the same loss; it is the grey, not the noise |
| `--match-alpha-weight 0.5` (`AW05`) | 0.030 | 131/21.9, IoU -0.003 |
| `--evidence-prune-inmask 0.3` (`PRUNE03`) | 0.036 | 117/19.6 |

The real fix is vertical parallax in the stage-1 orbit (a helical or two-elevation pass-1 path),
a pipeline design change that was not tested here.

## 5. The support-mask fringe

`select_support_views` un-premultiplies the cap renders and writes their alpha as the mask. Two
defects in what brush receives (`output/isplat/fringe_zoom.png`,
`output/isplat/masks_clean/_zoom_before_after.png`):

- a staircase of un-premultiplied colour noise along the outline: |rgb - median5| is 16.5 where
  alpha < 0.05 and 3.1 at 0.05-0.15, against 0.4 in the core; these pixels sit INSIDE the mask
  at small but nonzero weight. The step's `min_alpha` is 0.004 (1/255); the measurement says
  **0.15**;
- a one-Gaussian-per-pixel checkerboard inside the mask (dips to 0.90) — a 5x5 grey closing
  fills it without moving the boundary (`output/isplat/mkmask.py`).

Training on cleaned masks (`M1`) measured identical to the baseline, so this is hygiene, not a
quality lever — but those noise pixels are exactly the kind of thing that becomes a floater on a
different subject.

## 6. Implementation guide (b2crunner)

1. **brush**: commit the working tree of `~/Projects/brush` (`crates/brush-train/src/evidence.rs`,
   `crates/brush-train/src/train.rs` + `crates/brush-process/src/train_stream.rs`,
   `docs/splat-confidence.md`, `crates/brush-bench-test/tests/evidence.rs`) on
   `normal-map-supervision`, push to `Erant/brush`, bump `BRUSH_REF` in `docker/Dockerfile`
   (currently `debff989`, already one commit behind the pushed `f80b23b8`), rebuild the image.
2. **`rerender_splat`** in `pipeline/workflows/fast_helical_native.yaml`: add
   `conf_args: ["--conf-tau", "0.3", "--conf-angle-margin", "45"]` (the `render_splat` step
   already passes `conf_args` verbatim). Leave `gate_lo`/`gate_hi` at 0.45/0.65. Amend the
   comment that the gate replaces `mask_splat`'s thresholding: it does, with the fixed evidence.
3. **Shells**: `stage1_support_views` default `false` in the `settings:` block (keep the steps,
   they are gated on it). Rewrite the setting's help: the streaks it cites were measured before
   the face cap and the evidence gate; on 2026-09-05 they did not reproduce and the shells
   measured as a net loss. The switch stays for a subject that does show streaks.
4. **Normals**: no change at `train_splat`.
5. **Polish**: a `polish_steps` param on `steps/brush.py` (default 9000 for `train_splat`, 0 for
   `train_final_splat` until measured there). After the main export it symlinks the exported
   .ply into the COLMAP dir as `init.ply` and runs brush once more with
   `--total-train-iters N --refine-every 1000000 --growth-stop-iter 0`, the same
   `--normalize-masked-loss`/normal/evidence flags, `--normal-loss-start-iter 0`, and
   `--export-every N`, exporting over the first .ply. `refine_every` and the normal-loss start
   must reach the command line through the existing param plumbing, not be appended: brush
   refuses a repeated flag. `--export-evidence` on the polish run is what the re-render reads.
6. **`select_support_views`** (`pipeline/steps/anchor_stub.py`): `min_alpha` default 0.004 ->
   0.15, and a 5x5 grey closing (`cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, ones(5,5))`) on the
   alpha before the cut. `_unpremultiply` already zeroes the RGB where the mask is zero.
7. **Optional**: `--match-alpha-weight 0.5` on `train_splat` (a `match_alpha_weight` param; mild
   wedge reduction, no sharpness cost, IoU -0.003).
8. **Do not add** an alignment step to stage 1.

## Appendix: every run

`output/isplat/results.log` has one line per evaluation (`_ev` = evidence re-measured with the
fixed brush, gate tau 0.3); `output/isplat/NOTES.md` is the running log. The scripts that
reproduce any row are in `output/isplat/` (`train.sh`, `finetune.sh`, `mkds.py`, `mkmask.py`,
`align.py`, `loop_posthoc.sh`, `loop_inline.sh`, `eval_novel.py`, `gate_sweep.py`, `reev.sh`,
`evalev.sh`); the data is `~/Downloads/refinesplat/splat/`. Note `output/` is gitignored — copy
what should survive.
