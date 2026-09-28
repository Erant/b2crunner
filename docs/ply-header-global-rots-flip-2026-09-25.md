# `b2c.mhr.global_rots` in the PLY header misses the FLIP

Found 2026-09-25 while building b2crig (animating the delivered splats). Not
yet fixed; pick up on the next b2crunner session.

## What is wrong

`pipeline/ply_meta.py:body_comments` writes

    joints      = scale * joints_raw @ R.T + t
    global_rots = R @ rots_raw

where `R`, `t`, `scale` are `world_from_raw`. The two inputs come from
`refit_body_to_splat` (`pipeline/steps/body_refit.py`), whose `forward()`
returns

    verts[0] * flip, keypoints * flip, joints[0] * flip, rots[0]

— positions are FLIPPED into SAM-3D-Body's raw (OpenCV) frame, the joint
rotations are NOT (they are `mhr_forward`'s own frame). So the header's
positions and rotations live in different frames: a joint's world
orientation consistent with the world vertices and joints is

    R @ diag(1, -1, -1) @ rots_mhr

and the header holds `R @ rots_mhr`.

## Evidence

On `helical-20260925-034304-416580`, replaying the header's pose params
through `mhr_model.pt` (b2crig `tools/export_mhr_subject.py`):

    header global_rots vs R @ rots_mhr              max |diff| 2.7e-7   (what is stored)
    header global_rots vs R @ FLIP @ rots_mhr       max |diff| 2.0      (what would be consistent)

Joints and vertices replay exactly (0.000 mm vs the header joints, 0.0006 mm
vs `debug/body_refit/mesh_refit.ply`), so only the rotations are affected.

## Consequence

Any consumer that uses `global_rots` together with `joints` (binding splats
to bones "without running the model at all", as the module docstring
promises) gets relative rotations conjugated by the wrong frame:
`R Q Q0^T R^T` instead of `M Q Q0^T M^T` with `M = R @ FLIP`. Nothing inside
b2crunner reads them back today as far as found; b2crig does not use them
(it re-derives world rotations from the model).

## Fix options

- In `body_refit.py`'s `forward()`, return `flip_matrix @ rots[0]` (with
  `flip_matrix = diag(1, -1, -1)`) so every returned quantity is in the raw
  frame, and keep `ply_meta` as is. Check the other consumers of the step's
  `global_rots` output first (`build_body_rig`, the face rig), since this
  changes what they receive.
- Or apply the flip in `ply_meta.body_comments` only (header-local fix).

Either way, bump `b2c.mhr.version` so readers can tell old headers apart,
and add a test that `global_rots` and `joints` from the header agree with a
replay's world-frame skeleton.
