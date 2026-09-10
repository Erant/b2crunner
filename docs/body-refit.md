# Body refit: moving the MHR body onto the trained splat

`pipeline/steps/body_refit.py` — two steps, `splat_surface` (main env) and
`refit_body_to_splat` (sam3dbody env). Written and wired into `fast_helical_native.yaml` 2026-09-09.

## Why

The body SAM-3D-Body fits to the one photograph is the pipeline's frame of
reference: the diffusion passes are conditioned on renders of it, the
cameras orbit it, and the trainer's hollow loss forbids splat weight behind
it. The trained splat follows the generated frames instead, and by the end
of the run it has drifted from the body: two rounds of camera refinement
move the camera set against the body frame, limbs land a few degrees off,
and the model's torso was 2-5 cm in front of one subject's skin. Everything
that leans on the body downstream — the hollow loss of the next training,
and later a binding that lets the body's skeleton drive the splat — should
lean on a body that is where the splat is.

## What it does

`splat_surface` runs `b2ctrain probe --depth` over the training cameras
(a b2ctrain build from 2026-09-09 or later), which writes per pixel the
depth at which the splat's accumulated alpha first reaches `tau` (0.5, the
median surface), unprojects the depths into world space with the OpenGL
camera-to-world poses the cameras carry, drops silhouette pixels (a depth
jump over `edge_jump` to a neighbour), and keeps an oriented, strided,
capped sample. Splat centres are deliberately not used: they sit at every
depth of a soft surface and carry every floater.

`refit_body_to_splat` re-runs the MHR body model's differentiable forward
(the same `mhr_forward` fit_head_to_face uses, loaded through
`head_fit.build_mhr_head`) with its parameters free, in three stages:
root rotation + translation + per-joint scales, then the body pose, then
the shape components. The objective is one-sided because the model is a
naked body and the splat is clothed: a surface point inside the mesh is
penalised in full; outside, the first `clothing_allowance` (1 cm) is free
and the rest counts at a tenth. A coverage term (vertex to nearest surface
point) keeps limbs from shrinking away. Distances are point-to-plane
against the mesh's vertex normals, correspondences re-found every step on
a 100k-point random subset. Every free parameter is L2-pulled toward
SAM-3D-Body's fit — hands hardest (and their vertices are left out of the
fit), scales and shape next; the root is free. The mesh's frame is
recovered from the (raw, world) pair by Procrustes and must fit to a
millimetre; the root's motion goes into the body model's own `global_rot`
and `global_trans`, so the updated `pose_params` regenerate the fitted mesh
exactly (replay checked to 0.001 mm).

Outputs: the refitted geometry in both frames, `pose_params` (now with
`global_trans` and `scale_offsets` entries a replay must pass),
`world_from_raw`, `body_refit_stats`, and `rig_binding` — the rig's joint
hierarchy, sparse skinning weights (~2.8 joints per vertex, summing to 1)
and inverse bind poses read off `mhr_model.pt`, for the binding step to
come.

## Measured (2026-09-09, 4070 Ti)

Local reproduction of the fast_helical_native run of 2026-09-09 (81 final
views at 1080x1920, the delivered `scene.ply`, SAM-3D-Body re-run on the
run's anchor and registered onto its points3D.txt). Surface: 300k points
from 403k sampled at stride 8, 7.6 s. Fit: 87 s.

| | SAM-3D-Body mesh | refit |
|---|---|---|
| surface-to-body, median abs | 1.33 cm | 0.73 cm |
| surface-to-body, p90 abs | 3.35 cm | 2.73 cm |
| surface points > 5 mm inside the body | 24.7% | 12.9% |
| surface points > 1 cm inside (arms, the worst region) | 22.5% | 7.6% |
| body-to-surface, median | 0.71 cm | 0.37 cm |

What moved: the whole body 3 cm up and 1.3 cm sideways with a 3 deg yaw
(root), the pose 0.2 deg rms (1.2 max), scales 0.002 rms, shape 0.02 sigma
rms. Priors ten times weaker (pose 1, shape 0.3, scale 1) reach the same
residual (0.77 cm median, 10.8% inside) with 0.6 deg rms of pose, so the
residual is not prior-limited; what remains is mostly 5-10 mm, the
thickness of a soft splat surface at tau 0.5, plus clothing. The overlay
(`out/refit/overlay.png` in the b2ctrain checkout) shows the SAM mesh
sitting low with its arms away from the real arms, and the refit on the
head, shoulders, arms and shoes.

Effect on the hollow loss: see b2ctrain's docs/STATUS.md ("Body refit").

## Wiring

In `fast_helical_native.yaml`, right after `load_trained_splat` (the
stage-2 splat) and before the second pass re-renders it: `splat_surface`
(splat_path, dataset.cameras) -> `refit_body_to_splat` (mesh_output: scene,
mesh_world, surface), which republishes scene.vertices / keypoints_3d /
joints / global_rots / pose_params / mesh_world plus scene.world_from_raw,
scene.rig_binding and scene.body_refit_stats. `train_final_splat` takes
the refit `scene.mesh_world` through its existing `mesh` input, at
`hollow_margin: 0.03` (was 0.05). Both steps are gated on the
`refit_body` setting (default on) and on `export_ply`. The debug bundle
gets `debug/splat_surface/surface.ply` and `debug/body_refit/` with
mesh_before.ply, mesh_refit.ply and stats.json. `pipeline.cli doctor` checks the trainer for
`probe --depth`. The sam3dbody env needs nothing new (torch, numpy,
huggingface_hub, sam_3d_body, and body2colmap through the child venv).

Nothing between the refit and the final training reads the body, and the
second camera refinement bundle-adjusts against the frames with the gauge
restored, so the refit frame carries through. A second refit after stage 5,
against the final splat, is where the splat-to-body binding will be
written.

## The body in the delivered .ply

`refit_body_to_splat` also publishes `scene.body_params`, and
`train_final_splat` takes it as `body_params` and writes it into
`ply/scene.ply`'s header after its last export (`pipeline/ply_meta.py`).
Header comments are the one place a record survives every reader —
b2ctrain refuses non-vertex elements and viewers parse every element they
find, but all of them skip comments. One line per key:

    comment b2c.mhr.<key> <shape> <values...>

| key | what |
|---|---|
| `version`, `model` | record version (1); the checkpoint repo and the `mhr_model.pt` the parameters replay through |
| `frame` | a one-line note of the conventions below |
| `world_from_raw.scale` / `.rotation` (3x3) / `.translation` (3) | `world = scale * raw @ rotation.T + translation`, SAM-3D-Body raw metres into the splat's world frame |
| `pose_params.<entry>` | every entry a replay must pass: `global_rot`, `body_pose_params`, `hand_pose_params`, `scale_params`, `shape_params`, `expr_params`, `global_trans` (metres), `scale_offsets` — raw frame |
| `joint_parents` (127) | the skeleton's parent index per joint (-1 at the root) |
| `joints` (127x3), `global_rots` (127x3x3) | the posed skeleton, joint positions and global joint rotations, already in the WORLD frame |

So a consumer that only wants to bind and pose the splats needs the last
three keys and the model's skinning weights (`rig_binding_data`, read from
`mhr_model.pt`); one that wants the mesh replays `pose_params` through the
model and applies `world_from_raw`. `ply_meta.parse_body_comments(
ply_meta.read_comments(path))` gives the record back as arrays. The record
is about 25 KB of header; it is written once, after the polish and the
alignment refits, and replaced rather than duplicated if written again.
`SplatScene.to_ply` does not carry comments, so a splat that is loaded and
re-saved through body2colmap loses it — nothing in the shipped workflow does
that to the deliverable (`load_splat` only reads; `Dataset.to_disk` copies
the file).


## Per-view joint rotations: the double limb

The generated frames move the limbs by centimetres between segments of
the orbit (measured between adjacent pristine frames against the refit
mesh's parallax: forearms and hands 7-8 px per 4.5-degree step, torso and
legs 2-3), and a canonical splat averages them into a **double limb** that
no image warp can fix. `build_body_rig` (pipeline/steps/body_rig.py,
pipeline/body_rig.py) turns the refit body into a rig — a subsample of its
vertices with the model's skinning, the joint tree and pivots, and the
active joints — and `train_final_splat` writes it as `body_rig.bin` beside
the COLMAP model and passes `--body-rig` (b2ctrain ee71363 or later; the
step probes the binary's --help and trains without it otherwise). The
trainer learns one small rotation per view and active joint, renders every
splat at its skinned position, and exports the canonical model; the
rotations land in `body_rig_omega.json` next to the .ply.

Active are all joints skinned to at least `min_subtree` (30) vertices
whose subtree is at most `max_subtree_fraction` (0.5) of the body: every
limb, the hands, the neck and head, but not the root chain (root, pelvis,
spine), where a rotation moves the whole body per view and was measured to
destroy sharpness everywhere. Settings on the brush step: `body_rig` (on),
`body_rig_start_iter` 1000, `body_rig_smooth` 0.05, `body_rig_zero` 0.02,
`body_rig_lr` 0.002. Measured on the 2026-09-09 bundle (b2ctrain
docs/STATUS.md, "Per-view arm rotations"): the double limb gone at novel
views, hand sharpness +24%, body +6%, face +9%, legs and torso unchanged,
the training loss the best of every run. The canonical model's PSNR
against the training frames drops by construction (its limbs sit at the
mean pose while each frame's are elsewhere); that number is not the
metric for this. `debug/body_rig/body_rig.json` lists the active joints
and subtree sizes; `pipeline.cli doctor` checks the trainer for
`--body-rig`.

## The same rig on the intermediate training

The intermediate splat has the problem too, and worse in relative terms:
measured on the 2026-09-09 bundle's `colmap_intermediate`, adjacent-frame
limb motion beyond the static body mesh is 8-13 mm at the arms and hands
against 1.4-1.9 mm at the torso and legs (~6x; at the final stage it is
~3x). It matters more than it looks, because the helical re-render that
pass 2 is built from is a render of this splat: a translucent double limb
here is baked into every frame the second denoise sees.

`build_body_rig_initial` (the `body_rig_intermediate` setting, on by
default) rigs the **initial** body for `train_splat` — SAM-3D-Body's fit
after `fit_head_to_face`, the same body `mesh_world` already carries for
the hollow loss. There is no splat to refit against at that point in the
run, and none is needed: the initial body rigs the intermediate splat as
well as the refit body does (hand sharpness 107.1 against 105.6, head 47.1
against 45.8). The step is the same `build_body_rig`; with no
`world_from_raw` to wire it recovers the raw-to-world similarity from
`mesh_raw` (scene.vertices) against `mesh_world`, and refuses if the two
are not the same mesh rigidly placed. `fit_head_to_face` publishes
`rig_binding` for it — pure model data (`rig_binding_data`), read where the
MHR model is already loaded.

Measured on that bundle (b2ctrain docs/STATUS.md, "The rig at the
INTERMEDIATE stage"), s1 at novel cameras against a two-seed baseline:

| | body | hands | head | subject |
|---|---|---|---|---|
| baseline | 51.3 | 84.4 | 43.1 | 51.6 |
| rig, initial body | 55.0 | 107.1 | 47.1 | 55.8 |
| rig, refit body | 55.3 | 105.6 | 45.8 | 55.9 |

at 1m32s either way (unchanged) and 1% fewer splats. Seed noise is under
1%. Canonical PSNR falls 38.07 to 33.34 for the usual reason, and the
split says exactly that: the 81 real frames go 33.86 to 27.20 while the 36
undeformed supporting views are untouched (47.91 to 47.85), with the
training loss unchanged.

**Only the real frames are rigged.** The 36 face supporting views are
renders of a fixed splat; giving them rotations too keeps the body and
hand gains but loses the whole head gain (novel/head 47.1 back to 43.1,
the baseline). They stay out by construction — `steps/brush.py` writes the
rig for `image_names`, which is the training frames, and appends the
supporting views to the COLMAP model separately — and
`tests/test_body_rig.py` holds that property down.

One thing worth knowing if you go measuring: the learned rotations here
are dominated by torso (1.8 deg), head (1.4) and legs (1.3) with the arms
at 0.3-0.8, the opposite of the final stage, where the trainer's alignment
loop has already absorbed the body-scale wobble. The two mechanisms are
close to orthogonal — the alignment loop alone gives body +9% and head
+12% and does nothing for the hands (+0.6%), the rig alone is what fixes
the hands. Running both on the intermediate training measured better still
(body +12%, head +15%, hands +19%); it is not wired, because
`train_splat`'s alignment is the pipeline's own `refine_cameras` and
turning the in-trainer loop on there is a separate change.
