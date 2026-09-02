# The face cap wins — `face_priority_weights`

## The problem

Three sources describe the face in the stage-2 training (`train_splat`),
and until 2026-09-02 brush heard them at the same volume:

| source | what it is | brush mode | weight over the face |
|---|---|---|---|
| the **face cap** | `render_face_support_views` + `face_support_views`: renders of the photo-derived face splat | masked supporting views | its own coverage (≈1) |
| the **denoised frames** | `denoise_pass1`'s output, the training views proper | transparent training views | 1, every pixel |
| the **stage-1 shells** | `pointmap_elevation_views` + `stage1_support_band`: Sapiens2 depth shells of every Nth denoised frame, from ±elevation | masked supporting views | their matte (≈1) |

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
- **`face_priority_shells`** (after `stage1_support_band`, gated on
  `stage1_support_views`): the shells' cameras and masks in, the same
  splat optionally (`?`). A shell is masked already, so the weight is
  folded INTO its mask (`scene.body_support_views.masks_deferring_to_face`),
  which is what `merge_support_views` now reads. With the face off the
  step passes the masks through unchanged, so the path has one writer
  whenever the shells exist at all.

`train_final_splat` takes neither: it has no supporting views, so there is
nothing for its frames to yield to.

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
