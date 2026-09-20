# Re-outline: the silhouette from a splat of the subject, not the mesh

*2026-09-08; the splat since 2026-09-20. Experimental, off by default
(`re_outline` in helical.yaml). Never yet run on a pod.*

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

Eight steps between `reinject_anchor_initial` and `dump_denoise_input`,
every one `when: ${globals.re_outline}`:

| step | what |
|---|---|
| `reoutline_downscale` | `resize_batch`: the control video and its VACE flags, 720x1280 -> 480x832 |
| `reoutline_denoise` | `wan22_vace_denoise` at 480x832, `denoise_pass1`'s block at 2 high / 2 low steps (`strength` `[1, 1, 0.5, 0.5]`) |
| `reoutline_matte` | `rmbg` over the denoised frames; `debug_dir` puts frames + mattes in `debug/reoutline/` |
| `reoutline_upscale` | `resize_batch`: the frames AND their mattes back to the render size |
| `reoutline_train_splat` | `brush`: a splat fitted to those frames and mattes on `dataset.cameras`; `debug/reoutline_splat.ply` |
| `render_reoutline_splat` | `render_splat`, no `pattern` (the dataset's cameras verbatim); only its alpha is kept, as `scene.outline_masks` |
| `render_reoutlined_views` | `render`, `render_initial_views`' params exactly, plus `outline_masks` |
| `reinject_anchor_reoutlined` | `inject_anchor` over the fresh render |

The extra pass sees exactly what pass 1 sees — the same drawings, the
photograph at the anchor frame with its 0.0 VACE mask, the same reference
and seed — only smaller, and in four steps rather than six (2 high / 2 low;
the low expert's extra steps are texture, and the texture is thrown away).
Its output is thrown away except for its shape: rmbg cuts the subject out of each frame, hair and all, a
splat is trained to those mattes on the very cameras the frames were drawn
from, and the orbit is rendered again with the splat's coverage on each
camera as the outline. The skeleton and the face splat are re-drawn from
the same mesh and .ply on the same cameras, so the only thing that changes
in the frames pass 1 is handed is the fill. With the setting off no step
runs and pass 1 conditions on the first render.

**Why a splat and not the mattes.** The branch as first written
(2026-09-08) handed rmbg's mattes straight to the render, one per frame.
Those mattes are as consistent as the frames they are cut from, which is
to say not: a 480p denoise of 81 drawings paints the coat's hem, the hair
and the sleeve a little differently in every frame, and a matte of each
frame is that frame's own opinion of where the subject ends
(docs/garment-consistency-research-2026-09-19.md measured the same
per-frame billow on the coat in c0514e). Conditioning pass 1 on an outline
that jumps from frame to frame hands it the inconsistency the outline was
meant to remove. A splat fitted to all 81 mattes at once cannot follow any
one of them: where a frame's matte bulges on its own the other views
outvote it, and where they agree — the hair top, the hem, the soles, the
things the mesh gets wrong — the coverage follows them. Rendered back on
the same cameras it is one 3-D subject seen from 81 places, which is what
an outline for a camera orbit should be.

The training is the intermediate splat's silhouette recipe and nothing
else: `match_alpha_weight` 0.5 as `train_splat` has it, `align_iters` 0,
`polish_steps` 0, and none of the supporting views, loss weights, normals,
body rig or hollow loss the intermediate takes — this splat's colour is
never looked at, and the hollow loss penalises weight behind the body
model's surface on a training whose whole purpose is what the body model
leaves out. `total_steps` is 15000, half the intermediate's: the texture
is thrown away, and the back half of a training is where the texture
sharpens while the coverage has long settled. Unmeasured for this splat;
30000 was only ever the count the alpha knob happened to be measured at.

## The first run, and the hairs (helical-20260920-150953)

The branch's first pod run came back with the re-outlined drawing fringed
by horizontal hairs on every edge and pocked with small holes. Rendering
`debug/reoutline_splat.ply` on the run's own cameras locally and comparing
against the mattes it was trained to:

- The hairs are opaque needle Gaussians, not haze: the fill above alpha
  0.9 has as much thin structure as the fill above 0.5. Raising
  `outline_mask_threshold` does nothing; the evidence gate
  (`confidence: true` on the render) halves the hairs but leaves the gaps
  and drops 4% of the fill (the hands go first).
- 0.78% of the fill is thinner than 9 px, against 0.04% of the mattes —
  twenty times the subject's own thin structure — and 0.53% of it is gaps
  thinner than 9 px, against 0.11%.
- Mechanism: on a single-elevation orbit every camera ray is horizontal,
  so a Gaussian at the silhouette seen edge-on by one camera is free to
  stretch along that camera's ray, and every other camera on the ring sees
  the stretch as a horizontal streak. The per-frame disagreement of the
  480p frames (this run's subject walks) is what lets them survive the
  alpha loss.
- Fix, measured: `outline_mask_clean_px: 9` on `render_reoutlined_views`
  — an opening then closing of the cut silhouette with a 9 px disc, before
  the blur. Thin structure 0.78% -> 0.00%, IoU against the mattes 0.9425
  -> 0.9489, 0.04% of a matte's own structure lost (fingers survive at
  this resolution). The trainer has no anisotropy control to do it at the
  source — and, tried locally the same day, one would not help: a cap on
  each Gaussian's longest/middle scale ratio (8, 4, 2) leaves the hairs
  exactly where they are (0.76 -> 0.78 / 0.77 / 0.86% thin), and the
  export-time in-mask prune at 0.8 shrinks the body (IoU 0.9426 ->
  0.9346) before it clears them (0.69%). They are not needles and not
  out-of-hull floaters: by the ply's evidence they are low-opacity discs
  straddling the silhouette, inside the matte in most views and outside
  in a few — the consensus disagreeing with single frames, drawn as
  streaks because the disagreement lies along the ring's horizontal rays.
  The intermediate splat, checked at pass 2's helical cameras, has no
  such streaks (0.12% thin, no arm differs). b2ctrain/out/needle/README.md
  has the tables.

Two other things the same comparison showed, unfixed: the splat sits a
uniform 3.4 px to the RIGHT of every frame's matte all the way round the
orbit (a uniform image-space offset in the denoised frames — the lateral
cousin of the anchor raise — since no world offset can look the same from
every azimuth), and at the anchor frame the splat's feet are 30 px above
the photograph's while the head matches (the mesh's feet sit 22 px high,
docs/vace-denoise-findings-2026-09-07.md, and the denoise followed the
drawing).

## The two fill strengths

Two pipeline settings, since 2026-09-20, replace the render step's own
`outline_strength` (which is wired to them and so no longer drawn in the
per-step panel — a param written as `${globals.<name>}` has its one home
in the Settings box):

| setting | default | fill | read by | which is what... |
|---|---|---|---|---|
| `outline_strength` | 6.25 | #777777 | `render_initial_views` | pass 1 sees with the branch off; the 480p pass sees with it on |
| `reoutlined_strength` | 20 (`requires: re_outline`) | #666666 | `render_reoutlined_views` | pass 1 sees with the branch on |

The asymmetry is the point. The first drawing's silhouette is the body
model's, wrong wherever hair and clothing leave it, so it is kept faint —
a hint under the skeleton, not a shape to trace (the render step's own
default was 12.6%, #6F6F6F, until this). The re-outlined drawing's
silhouette is the splat's coverage, the subject's own shape that every
frame of the 480p pass had to agree on, and can be trusted, so it is
drawn three times darker. `reoutlined_strength` is greyed out in the UI
while `re_outline` is off, because nothing reads it then; `requires:` on
a setting is the same word and the same control an output's `requires:`
gives a deliverable.

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
The frames and their mattes go back through the same plain resize,
inverted, which cancels it exactly — frame i lands on frame i's original
pixel grid, which is the grid `dataset.cameras` describe and the size
brush writes into the COLMAP model it trains from. The upscaled frames are
soft, and it does not matter: the splat is kept for its coverage. `render`
refuses a matte of any other size rather than resampling it, and the
splat's render is asked for the render size explicitly, so the size is
decided in one place.

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
- 2026-09-20: `reoutline_upscale_mattes` became `reoutline_upscale`
  (frames too), and `reoutline_train_splat` + `render_reoutline_splat`
  sit between it and the re-render. No step code changed: `brush` and
  `render_splat` (no `pattern`, so the dataset's cameras verbatim) were
  already what the branch needed.
- docker/Dockerfile: the pin, and an assert that fails the build on a
  body2colmap without the seam — an older one ignores the key silently and
  the branch would spend a denoise and change nothing.

## Reading a run

`debug/reoutline/frame_NNNNN.png` is the 480p denoise output and
`matte_NNNNN.png` its matte; `debug/reoutline_splat.ply` is the splat
fitted to them; `debug/denoise_pass1_input/` is the drawing pass 1 was
handed, whose fill is that splat's coverage. Compare the fill against the
mesh outline of the same frame in a `re_outline: false` run at the same
seed: the hair top and the soles are where they should differ. Compare it
against `matte_NNNNN.png` (resized) too: where the two differ is where the
splat overruled that frame's matte, and the question is whether it was
right to. Then compare the two runs' pass-1 outputs.

Things worth knowing before believing a result: the 480p pass's matte at
the anchor frame is rmbg of the warped photograph, so the splat's coverage
at that camera against that matte is the method's own error bar (the
anchor frame itself is re-injected afterwards, so the drawing there is the
photograph regardless); the seams to watch are the frames farthest from
it, where the denoiser is free to disagree with the mesh about the body's
width and the outline now follows the consensus of the denoiser's frames.
The splat is trained on the orbit's ideal cameras, with no
`refine_cameras` of its own (the one brush training in the file without
one; tests/test_workflows.py exempts it by name). Two things follow. The
denoiser paints the body a few px above the drawing
(docs/vace-denoise-findings-2026-09-07.md, "anchor raise"), every frame
alike, and the splat absorbs that as a subject sitting slightly high in
the world — so the outline is a few px above the skeleton drawn from the
mesh. A refinement would not take that out: `refine_cameras` removes the
common mode of its own solution on purpose (Trap 4), so a uniform raise
stays in the frames and therefore in the splat. What a refinement would
correct is per-frame jitter, which is a sharpness matter for a texture
and mostly averages out of a coverage; and it could not publish to
`dataset.cameras`, which have to stay the ideal orbit the re-render and
the main solve read. If a run shows the outline visibly high
against the skeleton, the fix is in the drawing (shift the coverage down
by the measured raise before `render`), not in the poses; if it shows a
ragged edge, a refinement publishing to `scene.reoutline.cameras`, with
the splat still rendered on `dataset.cameras`, is the thing to try.
