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
