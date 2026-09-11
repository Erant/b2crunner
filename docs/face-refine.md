# The face per view, and the anchor's eyes

Four steps before the final training (`face_refine`, on by default; needs
`refit_body`): fit the refit body's head to every frame that shows the
face, hand the trainer that per-view face geometry the way the body rig
hands it the arms, and replace the frames' eyes with eyeballs textured from
the anchor photograph. Code: `pipeline/steps/face_views.py`, the rig side
in `pipeline/body_rig.py` and `steps/body_rig.py` (`build_face_rig`);
measurements in b2ctrain's docs/STATUS.md, "The face per view, and the
anchor's eyes injected" (2026-09-11).

## Why

The body rig poses the arms per view; the face got nothing. And the eyes
of the helical frames are the diffusion's invention per frame — three
consecutive frames of one subject show three eye states (half closed, open
looking sideways, open with makeup); pass 2 re-denoises even the
reinjected anchor frame. A splat trained on that averages them into a dark
slit or a blue smear, and nothing multi-view can recover what was never
consistent. What IS consistent is the anchor photograph, and the MHR body
model has the geometry to carry its eyes into every frame: eye joints (no
eyeballs — the mesh is one closed surface, the lid ring 15-21 mm from the
joint), and 72 expression blendshapes that SAM-3D-Body predicts as zero
and that explain each frame's lids when fitted.

## What it does

1. **`detect_face_views`** (main env, MediaPipe, ~1 s). Landmarks the face
   per frame without a detector: the pipeline's BlazeFace short-range
   pass on a whole 1080x1920 frame misses the ~80 px face in 63 of 81
   helical frames and finds faces on the trousers in four. This projects
   the refit head into the frame (the vertices `map_face_to_mesh` mapped),
   cuts a 1.8x crop around it rolled upright by the projected head-up
   axis (MHR joints 113 -> 126), and runs the landmarker on the crop. 35-38
   of 81 landmarked: every view facing the camera within ~85 degrees; the
   rest are back views or hair-covered profiles. `debug/face_views/` has
   every crop with its landmarks and `meta.json` (facing angle, roll).
2. **`fit_head_per_view`** (sam3dbody env, ~20 s on a 4070 Ti). All
   fitted views in one batched Adam run on the MHR forward: per view the
   six neck/head rotations (`body_pose_params[18..23]`) and the 72
   expression coefficients, projected through the view's refined dataset
   camera against its landmarks (features only — no face oval, no irises,
   Huber 4 px). Landmark rms 6.4 -> 4.6 (pose) -> 2.1 px (pose + expression)
   on the first subject, 8.4 -> 4.6 -> 2.6 on the second. `pose_prior` 50
   (L2 on the rotations, rad^2 against pixels) matters: unregularised, the
   six DOF counter-rotate into a lateral shift of the head (61 mm on one
   frame) at no residual gain. The anchor photograph joins the batch — a
   PnP camera against the canonical head from `detect_face`'s landmarks
   (1.5-2.5 px), the camera free in the fit, the head pose fixed,
   expression free (-> 1.1-1.4 px) — so its lids are fitted too. Views
   facing beyond `max_facing_deg` (85) are dropped. `debug/head_fit_views/
   fit.json` has the per-view residuals and rotations.
3. **`paste_eyes`** (main env, pyrender, ~4 s). A 15.5 mm icosphere at
   each eye joint (the lid ring sits 15.4-20.6 mm out; a sphere fitted to
   the model's own eye surface is 15.8-16.4 mm), textured from the anchor
   through its fitted lids: the gaze axis is where the ray through the
   anchor's iris landmark hits the sphere; the vertices the anchor does not
   show — the ~30% of the iris under its upper lid — take a radial colour
   profile around that axis; beyond the iris, one median sclera shared by
   both eyes. Per fitted frame each eye moves rigidly with its lid ring
   (Kabsch, canonical -> fitted), the sphere is clipped **in 3D** to the
   lid contour's cylinder (a 2D polygon leaks at grazing angles),
   depth-tested at 1 mm against the head mesh with the eye-surface faces
   removed and back-face culling off (else there is no depth behind the
   hole), and composited over the frame. Two gates: an eye whose visible
   part is under half its lid polygon is skipped (the far eye peeking past
   the model's narrower nose is not that frame's eye), and an eye with
   nothing darker than the surrounding skin inside its lid polygon is
   skipped (a hand in front of the face; the head mesh cannot know). The
   pasted frames replace `dataset.images`, so `export_colmap` and
   `train_final_splat` both see them. `debug/eyes/panel.jpg` is the eye
   region before/after per frame; `eyes.json` says what was drawn and
   skipped and why.
4. **`build_face_rig`** (main env). The fit as the rig's per-view vertex
   displacements: fitted head minus canonical head at the rig's vertices,
   written as a **v3 rig** (`B2CRIG3`: the v2 file plus
   `[views][verts] float3`), which b2ctrain e8f43ac+ adds to a bound splat
   BEFORE the skinning blend, so it rides the learned joint rotations. This
   is what lets the canonical eye opening converge to the anchor's instead
   of the average of every frame's lids: with the eyes pasted but no
   deltas, the novel view between two half-closed-lid frames stays a dark
   slit. Frames the fit does not cover interpolate across gaps of up to
   `gap` (4) frames and otherwise **hold** the nearest fitted frame's
   deltas within `hold` (3) frames (holding beat fading to canonical: swim
   3.4 -> 2.4 px). The deltas are restricted to the **face core** —
   vertices the expression basis can move by `face_motion_cm` (0.5) or
   more, fading to zero `face_fade_cm` (3) away. Whole-head deltas
   (`face_motion_cm: 0`) blurred the hair 2-5% on three of four subjects:
   the frames' hair does not follow the face fit and the back views pin it
   canonical, so it was supervised in two places. `debug/face_rig/
   face_rig.json` says how each frame got its deltas. The brush step
   writes the file as v2 with a warning if the trainer's --help does not
   name `B2CRIG3`; `pipeline.cli doctor` checks for it.

## Measured (2026-09-11, four subjects, stage-5 argv, 4070 Ti)

s1 = band-limited sharpness at novel cameras, split into the face (the
expression-moved vertices) and the rest of the head; swim = std across
dense novel views of the rendered face landmarks' shift against the
projected canonical head; eye MAE = mean |render - eye model| inside the
eye model's mask at the novel cameras. base = v2 rig, original frames;
face-only = this integration's default.

| subject | face s1 base / face-only | hair s1 base / face-only | swim px base / face-only | eye MAE base / face-only |
|---|---|---|---|---|
| F3 | 28.95 / 28.22 | 22.90 / 22.42 | 2.47 / 2.02 | 66.6 / 55.7 |
| facerefine | 30.44 / 30.48 | 40.29 / 40.13 | 5.02 / 1.58 | 51.3 / 35.5 |
| 22 (bad face) | 21.48 / 21.94 | 19.29 / 19.22 | 3.03 / 3.09 | 65.6 / 28.3 |
| 19 (good face) | 35.32 / 34.96 | 25.76 / 26.42 | 1.79 / 2.13 | 81.5 / 51.5 |

The eyes are the headline on every subject: 22's blue smears and 19's
closed slits become open eyes with an iris at every novel view. Face
sharpness stays within +-2.5% of base (the metric's resolution); the hair
is untouched; the swim drops 3x where the subject swims. Whole-head deltas
give somewhat better eyes on the good-face subject (42 against 52) at 3%
of its hair — `face_motion_cm: 0` on `build_face_rig` is that option.

## Open

- The eye region gets no loss-weight boost (the `weights/` sidecar exists
  for that); gaze is fixed to the anchor's relative to the head; closed
  eyes are not detected; an eye pasted onto a hand in front of the face
  survives when the hand is dark (occlusion from the posed body would be
  the proper fix). The texture comes from a ~90 px-wide face (iris ~9 px),
  the same scale as the frames' eyes, so it is not the bottleneck.
- The intermediate training gets none of this: at stage 2 there is no
  refit body and the frames are about to be re-rendered anyway.
