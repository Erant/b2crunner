# The face cap wins — `face_priority_weights`

## The problem

Three sources describe the face in the stage-2 training (`train_splat`),
and until 2026-09-02 brush heard them at the same volume:

| source | what it is | brush mode | weight over the face |
|---|---|---|---|
| the **face cap** | `render_face_support_views` + `face_support_views`: renders of the photo-derived face splat | masked supporting views | its own coverage (≈1) |
| the **denoised frames** | `denoise_pass1`'s output, the training views proper | transparent training views | 1, every pixel |
| the **stage-1 shells** | `pointmap_elevation_views` + `stage1_support_band`: Sapiens2 depth shells of every Nth denoised frame, from ±elevation | masked supporting views | their matte (≈1) |

> The shells were switched off on 2026-09-05 and removed from the workflow
> on 2026-09-06, taking `face_priority_shells` with them. Two sources
> describe the face now; everything below about the shells is a record of
> what the change did, not of what runs.

The cap is the only one carrying the photograph. The frames carry the
diffusion model's idea of the face, which is what the cap exists to
overrule, and the shells carry the frames' appearance again. Wherever two
of them cover the same surface the fit averages them, so the cap loses as
often as it wins, and the result is the fighting seen on every run since
the cap landed.

## The mechanism

brush weights a **masked** view by its mask and a **transparent** view not
at all — its alpha is a target (`alpha = 0` outside the subject), not a
weight, so there was no channel to turn the frames down over the face
without also stopping them carving the silhouette.

That channel is new on Erant/brush's `normal-map-supervision` branch: a
`weights/` sidecar, a greyscale map per view that brush multiplies into
that view's loss pixel by pixel on top of its alpha mode. It reaches the
L1/SSIM term, the alpha-match lane, the normal-supervision term (as a
weighted masked mean) and the end-of-training **evidence pass** — so a
region a view was told to be quiet over does not count as evidence, or
disagreement, in `rerender_splat`'s confidence gate. See brush's
`docs/loss-weights.md`.

`face_priority_weights` (steps/face_priority.py) produces the maps. For a
batch of cameras it renders the refined face splat's coverage from each
one (through the same `brush-splat-render` binary `render_splat` uses,
colour discarded) and writes

    weight = 1 − strength · g(θ) · feather(coverage)

- `strength` 0.9: the weight at full coverage is 0.1. 1.0 masks the face
  out of the other sources entirely — the blunt version.
- `g(θ)`: 1 for a view within `cap_radius_deg` (30) of the anchor camera's
  view of the splat, fading linearly to 0 over `fade_deg` (15) beyond. The
  face splat is a 2.5-D shell from one photograph and only means something
  within the cap; silencing a frame that sees a side of the head the cap
  has no evidence for would leave that surface constrained by nothing.
  The angle is measured about the splat's centre from the anchor camera,
  read live from `dataset.cameras` (the same rule as the cap's axis:
  `refine_cameras` moves it).
- `feather(coverage)`: the rendered alpha blurred by `feather_px` (4), so
  the weight ramps at the splat's edge instead of stepping.

## Where it sits

Two steps in the shared tail, both `face_priority_weights`:

- **`face_priority`** (after `face_support_views`, gated on `face_splat`):
  cameras `dataset.cameras`, splat `scene.face_splat_path` — the REFINED
  splat, rebuilt through the refined anchor by `face_splat_refined`.
  Publishes `scene.face_priority.weights`, which `train_splat` and
  `export_colmap_intermediate` read optionally as `weights`; steps/brush.py
  writes them as `weights/<stem>.png`. With `face_splat: false` nothing
  writes the path and brush trains on byte-identical data to before.
- **`face_priority_shells`** (removed 2026-09-06 with the shells; after
  `stage1_support_band`, gated on
  `stage1_support_views`): the shells' cameras and masks in, the same
  splat optionally (`?`). A shell is masked already, so the weight is
  folded INTO its mask (`scene.body_support_views.masks_deferring_to_face`),
  which is what `merge_support_views` now reads. With the face off the
  step passes the masks through unchanged, so the path has one writer
  whenever the shells exist at all.

`train_final_splat` takes neither: it has no supporting views, so there is
nothing for its frames to yield to.

## The second consumer: the second denoise's VACE mask

Added 2026-09-09, behind the `face_vace_mask` setting (on by default,
needs `face_splat`). The 2026-09-08 pass-2 sweep
(docs/vace-denoise-findings-2026-09-07.md, section 6) found that head
detail is decided upstream of `denoise_pass2`: in all six runs the final
head's sharpness sat at or below its control frame's, whatever the
strength, shift or sampler. The face in that control frame is the trained
splat's, which inside the cap is the photo-derived face the weights above
won for it — so the one pass-2 lever left is to stop the pass repainting
it.

A VACE control mask is per-pixel, and it answers the same question a loss
weight does: 0 means "conditioning region, reproduce what the control
frame shows", 1 means "generate" (diffusers' `pipeline_wan_vace.py`; the
convention the injected anchor photo's all-0 frame has always used). So
the workflow reuses the step:

- **`face_cap_vace_mask`** (after `mask_splat_fringes`, before
  `reinject_anchor`, gated on `face_splat` AND `face_vace_mask`): the
  helix's cameras (`dataset.cameras`, carrying the anchor's refinement
  rigidly since `rerender_splat`'s `given_anchor_camera`), the refined
  face splat, the anchor index and position `rerender_splat` published,
  and `dataset.masks` — the all-1.0 batch `mask_splat` just emitted —
  in; `dataset.masks` out with the weight folded in. Over the face of a
  frame within the cap the mask is 0; it ramps to 1 over `fade_deg` past
  the cap and over `feather_px` at the rim; everywhere else it is
  untouched. `reinject_anchor` then writes its 0.0 over the anchor frame
  as before.

Two settings differ from the training call, both forced by diffusers:

- **`strength: 1.0`**, not 0.9. `prepare_video_latents` splits the
  control video into inactive/reactive halves at `mask > 0.5` — a hard
  cut, so 0.1 and 0 are the same split — and only `prepare_masks`' mask
  channel, concatenated onto the conditioning latents, sees the soft
  value. A tenth left to the denoiser would reach it as an ambiguous
  hint, not a tenth of anything.
- **The feather stays** (4 px) for the same reason in reverse: the mask
  channel is carried to latent resolution with nothing spatial lost (an
  8x8 pixel block becomes 64 channels), so the ramp at the rim is exactly
  what the model sees between kept face and repainted hair.

Past the cap the mask is deliberately 1: there the splat's face is what
pass 1 painted, resampled through a training, and cleaning that up is the
pass's job.

The batch `denoise_pass2` is handed, mask in alpha, lands in the debug
bundle as `debug/denoise_pass2_input/` (`dump_denoise2_input`, gated on
`export_debug`), so the first run can answer whether the 0 sits on the
face and not a hairline off it.

Never run on a pod as of 2026-09-09, but run for real on a pod's data: the
step, wired exactly as above, on run F1's refined face .ply and its
`colmap_preupscale/` cameras (the helix at 720x1280, anchor at frame 37),
through the local b2ctrain rasteriser. 22 of 81 views are within the cap
and its fade, 17 carry a pixel below 0.5 — the anchor's neighbours 34-40
at a full 0, the other loop's pass at 0-5 and 70-73 partially — and on
frame 37 the 0 region is an 80x100 px box on the face (x 313-393,
y 185-286), eyes to chin, clear of the hair and the neck. About 0.7% of
the frame, 10x12 latent pixels. Three frames either side of the anchor
the mask is already above 0.5, which is the cap's 30 degrees at the
helix's 9 degrees per frame; the fade only reaches the mask channel.

`face_support_views`' `min_path_angle_deg` went from the step's default 5
to 0 in the same change. The default dropped the cap views within 5° of
the denoising path because those views "have a denoised frame of their own
already"; with that frame silenced over the face, a cap view on the path
is the only face evidence at that angle rather than a resampled copy of
something the training has.

## Status

Unit-tested end to end with the rasteriser stubbed (tests/test_face_priority.py,
tests/test_loss_weights.py, the wiring in tests/test_workflows.py). On the
brush side the sidecar is verified for real on the local 4070 Ti: the GPU
tests pin the arithmetic (a uniform weight of ½ halves a transparent
view's loss exactly, 0 silences it, it stacks with a mask), and a 300-step
training on `~/Projects/colmaptest`'s dataset with 20 weight maps logged
`Dataset loss weights: 20 view(s) carry a weights/ sidecar`, trained, ran
the evidence pass and exported. The pipeline steps have not run on a pod
yet, and `strength`, `fade_deg` and `feather_px` are untuned defaults.

Things to look at on the first run: whether the face's rim (where the cap's
coverage feathers out) shows a seam against the denoised frames, which
would argue for a larger `feather_px` or a lower `strength`; and whether
the confidence gate in `rerender_splat` still keeps the face, which the
evidence-pass weighting is there to guarantee.
