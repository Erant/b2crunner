# The photograph wins over the surface it sees — `photo_priority_weights`

## The problem

The intermediate training (`train_splat`) fits the injected photograph and
80 denoised frames at the same volume. The ten or so frames either side of
the anchor are the denoiser's re-drawings of the front the photograph
already shows: re-lit, the costume redrawn, the body 5-10 px above the
photograph (docs/vace-denoise-findings-2026-09-07.md). The fit averages
them with the photograph, and the front of every intermediate splat came
out as the denoiser's consensus rather than the photograph's pixels. The
face cap fixes exactly this over the face (`face_priority_weights`,
docs/face-priority.md); nothing did it for the body.

The mesh path had the same problem in its atlas and solved it with
`photo_texture` (2026-09-16): every texel the photograph's camera sees is
the photograph's pixel, and klein never repaints them. Measured on 00307
it was that path's one unambiguous win (whole-subject sharpness 39.5 ->
51.5). This step is the same rule for the splat.

## The mechanism

`pipeline/steps/photo_priority.py`, wired after `face_priority` in
`helical.yaml`, ungated. For every training view:

1. the body mesh (`scene.mesh_world`, SAM-3D-Body's initial body in the
   world frame — the one the training's hollow loss uses) is rasterised
   from the view's camera (`pipeline/mesh_raster.py`), and every hit
   pixel becomes a surface point with a normal;
2. each point is tested against the photograph's camera (read live from
   `dataset.cameras[anchor_frame_index]`): inside its frame, not further
   than `occlusion_margin` behind the body's own depth from it, and facing
   it — the cosine to the anchor ramped from `facing_lo` 0.2 to
   `facing_hi` 0.5, photo_texture's skin/cloth ramp;
3. that confidence is carried `extend_px` (24) beyond the body's
   silhouette to the nearest on-mesh value (hair, the skirt's flare),
   clipped to the frame's own matte, feathered 4 px, and scaled by the
   same angular window the cap uses (`cap_radius_deg` 45 + `fade_deg` 45,
   wide because the per-pixel test already localises the yield);
4. the view's weight there is `1 - strength x attenuation x confidence`,
   multiplied into the face cap's weights, and `train_splat` and
   `export_colmap_intermediate` read the product (`scene.priority.weights`).

5. the step also hands the training `copies` (6) of the photograph's
   frame as masked supporting views at the anchor camera, masked by the
   photograph's own confidence field (steps 2-3 from its own camera),
   through `merge_support_views`' second triple: votes and supporting
   views at once (see the measurements).

The photograph's own frame keeps weight 1. **Only that frame**: see the
trap below. `strength` is the `photo_priority` setting (0..1; 0 leaves the
cap's weights untouched, the run as it was).

Cost: 81 numpy rasters of a 37k-triangle body, 3.6 s on the CPU.

## Trap: the helix ends on the anchor, and that frame is not the photograph

`inject_anchor` puts the photograph into every frame on the anchor's
pose — frames 0 and 80 of the helix — with the VACE mask at 0 there.
The denoiser hands frame 0 back as the photograph (Wan's first frame is
its own latent) but frame 80 comes back as a REPAINT: on the 2026-09-18
pod run the two are 14.5 dB apart, with a different face and the costume
cleaned. That repaint has trained at full weight at the photograph's own
pose in every run to date, so at that pose the splat is the average of
the two, and silencing the neighbouring frames alone does nothing (arm B2
below). The step therefore treats only `anchor_frame_index` as the
source; `anchor_tolerance_pct` (off) is the old pose rule, kept for a
dataset that has no index.

## Measured (local, 2026-09-19)

Substrate: the 2026-09-18 pod run's own intermediate export
(`debug/colmap_intermediate`: 81 frames + 36 cap views, the face cap's
weights), retrained on the 4070 Ti with the pod's argv minus the body rig
(not in the bundle), 30k iterations each. PSNR against the training
frame inside its matte; view 0 is the photograph; `cull` is the share of
the subject the pass-2 confidence render (`--confidence`, binary
defaults) greys out.

| arm | rule | PSNR @ photo | PSNR @ view 3 | cull @ photo |
|---|---|---|---|---|
| A | as shipped (face cap only) | 17.8 | 21.4 | 19 % |
| B | strength 0.8, frames 0 + 80 kept | 18.1 | 20.5 | 31 % |
| B2 | strength 1.0, frames 0 + 80 kept | 18.3 | 17.6 | 45 % |
| B4 | strength 1.0, frame 0 only | 20.6 | 17.5 | 45 % |
| **B5** | **strength 0.8, frame 0 only, + 6 masked copies of the photograph** | **23.5** | 18.9 | **14 %** |
| B6 | as B5 with 12 copies | 23.6 | 18.4 | 11 % |
| B7 | strength 0.9, 6 copies | 23.6 | 18.3 | 13 % |
| **B8** | **strength 0.5, 6 copies (the shipped default)** | **23.3** | **19.7** | 17 % |
| B9 | no fade, 6 copies | 23.1 | 20.5 | 18 % |
| B10 | only frame 80 faded, no copies | 18.2 | 21.5 | 17 % |
| B11 | frame 80 faded, 6 copies | 23.2 | 20.5 | 16 % |

Sharpness (Laplacian variance in the matte) at the photograph's view:
A 294, B5 837, the photograph itself 1126; at view 3: A 373, B5 531.

The copies are the mechanism (B9: the whole gain with no fade at all);
frame 80's repaint costs 0.35 dB on its own (B10); the fade is a fidelity
dial between the photograph and the frames beside it (B8 -> B5: +0.2 dB
at the photograph, -0.8 dB at view 3). Three more things the table shows. Fading the neighbours while frame 80 keeps
its weight buys nothing at the photograph's view (B, B2). With frame 80
faded too (B4) the photograph gains 3 dB but one supporting view is below
the evidence gate's `conf-min-views`, and strength 1.0 greys out 45 % of
the subject in the render pass 2 conditions on. The copies (B5) are the
answer to both: the photograph as several masked supporting views (masked
by its own confidence field) is more votes AND more `ev_views`, so the
front is the photograph's and the gate keeps it. Off-axis (views 3-10) no
ghosting; the splat shows the photograph's costume where it used to show
the denoiser's redrawing. PSNR against the denoised frames drops by
construction (they disagree with the photograph).

The `cull` column is at the binary's defaults; `render_subject` ships
with `--conf-tau 0.3 --conf-angle-margin 45`, at which A and B5 both cull
4 % — the copies' evidence advantage matters at the defaults, less there.

## Not done

* The frames 45-90 degrees round still see the front at a grazing angle
  and are only partly faded; a tighter window trades that against the
  sides' evidence.
* Off the body mesh beyond 24 px (a wide skirt, long hair) nothing yields.
* The final training never sees the photograph at all (no reinjection
  after pass 2); this step is intermediate-only by construction.
