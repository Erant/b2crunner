# Re-outline: the silhouette from a matte, not the mesh

*2026-09-08. Experimental, off by default (`re_outline` in
fast_helical_native.yaml). Never yet run on a pod.*

## The problem

The first denoise conditions on a drawing: a flat silhouette (`#6F6F6F` on
`#7F7F7F`, 4 px blur) under the DWPose skeleton and the face splat. The
silhouette is the SAM-3D-Body mesh's, rasterised by body2colmap's
`render_outline`, and the mesh is a naked body model. Wherever hair, a coat
or a platform sole leaves that model, the outline tells the denoiser the
subject stops short of where the photograph says it goes.
docs/vace-denoise-findings-2026-09-07.md measured it on cyber_6f: the mesh
scalp 7 px below the hair top, the mesh feet 22 px above the soles.

## The branch

Six steps between `reinject_anchor_initial` and `dump_denoise_input`, every
one `when: ${globals.re_outline}`:

| step | what |
|---|---|
| `reoutline_downscale` | `resize_batch`: the control video and its VACE flags, 720x1280 -> 480x832 |
| `reoutline_denoise` | `wan22_vace_denoise` at 480x832, `denoise_pass1`'s block character for character |
| `reoutline_matte` | `rmbg` over the denoised frames; `debug_dir` puts frames + mattes in `debug/reoutline/` |
| `reoutline_upscale_mattes` | `resize_batch`: the mattes back to the render size |
| `render_reoutlined_views` | `render`, `render_initial_views`' params exactly, plus `outline_masks` |
| `reinject_anchor_reoutlined` | `inject_anchor` over the fresh render |

The extra pass sees exactly what pass 1 sees — the same drawings, the
photograph at the anchor frame with its 0.0 VACE mask, the same reference,
seed and strength schedule — only smaller. Its output is thrown away except
for its silhouette: rmbg cuts the subject out, hair and all, and the orbit
is rendered again with that matte as the outline. The skeleton and the face
splat are re-drawn from the same mesh and .ply on the same cameras, so the
only thing that changes in the frames pass 1 is handed is the fill. With
the setting off no step runs and pass 1 conditions on the first render.

## Two things that are not obvious

**Why the batch is resized in b2crunner and not by the denoise step.**
diffusers' `WanVACEPipeline.preprocess_conditions` does not resize a
control video to `width` x `height`. It scales the video down
aspect-preserving until it fits under the target *area*, then floors each
side to a multiple of 16. A 720x1280 batch asked for 480x832 is therefore
denoised at **464x832**, and the reference image, which `_fit_reference`
cropped to the 480x832 the params named, is letterboxed into the 464-wide
frame between white bars. Handing it a batch that is already 480x832 leaves
diffusers nothing to fit, and the pass runs at the size its params say.

**Why the anisotropy does not matter.** 720x1280 -> 480x832 is 0.667 across
and 0.65 down: the denoiser sees a figure 2.6% squatter than the drawing.
The mattes go back through the same plain resize, inverted, which cancels
it exactly — frame i's matte lands on frame i's original pixel grid.
`render` refuses a matte of any other size rather than resampling it, so
the size is decided in one place.

## What changed where

- body2colmap `76a74bc`: `Renderer.render_outline(mask=...)`, a module-level
  `outline_from_mask`, and `modes["outline"]["mask"]` through
  `render_composite`. The drawing is byte-identical for the mesh path.
- `render`: an optional `outline_masks` input (one matte per frame, at the
  render size), `outline_mask_threshold` (0.5). The matte is cut hard at the
  threshold and then softened by `outline_blur` exactly as the mesh
  silhouette is.
- `resize_batch`: new step (steps/resize.py).
- `rmbg`: an optional `debug_dir`.
- docker/Dockerfile: the pin, and an assert that fails the build on a
  body2colmap without the seam — an older one ignores the key silently and
  the branch would spend a denoise and change nothing.

## Reading a run

`debug/reoutline/frame_NNNNN.png` is the 480p denoise output and
`matte_NNNNN.png` its matte; `debug/denoise_pass1_input/` is the drawing
pass 1 was handed, whose fill came from those mattes. Compare a matte
against the mesh outline of the same frame in a `re_outline: false` run at
the same seed: the hair top and the soles are where they should differ.
Then compare the two runs' pass-1 outputs.

Things worth knowing before believing a result: the 480p pass's matte at
the anchor frame is rmbg of the warped photograph, so that frame's outline
is the ground truth of the method; the seams to watch are the frames
farthest from it, where the denoiser is free to disagree with the mesh
about the body's width and the outline now follows the denoiser.
