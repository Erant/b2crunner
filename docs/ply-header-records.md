# PLY header records: `b2c.mhr.*` and `b2c.orbit.*`

**Legacy since 2026-09-29.** Runs now deliver `ply/scene.glb` (b2cgltf SPEC.md) with everything these records held,
beside a bare `scene.ply`, and the Results tab strips these lines from an older `scene.ply` it packages. This page
stays the contract for reading deliverables made before that date; `tools/export_glb.py` converts one into a subject
file.

The delivered splat (`ply/scene.ply`) carries two records as PLY header `comment` lines: the refitted MHR body and the
orbit the frames were made on. This page is the contract for readers outside b2crunner (b2crig reads both), so they
parse the header themselves instead of importing `pipeline.ply_meta` / `pipeline.orbit_record`. Those two modules are
the reference implementation (writers and readers); a change to either record's format changes this page too, and
bumps the record's `version`.

## Where the lines are

Every line is `comment <text>` inside the header, between `format ...` and `end_header`. Readers take the header as
ASCII lines from `ply` to `end_header`, keep the `comment ` lines in file order and strip that prefix. Other comments
(the exporter's own) may come before, after or between them; a reader ignores every line without its prefix. Each record
appears at most once: the writers replace an earlier record with the same prefix and leave everything else as it was.

## Line grammar (both records)

    <prefix><key> <shape> <values...>        an array or a scalar
    <prefix><key> json <json>                a JSON value (orbit record only)
    <prefix><key> <text>                     a string (only the keys listed as strings below)
    <prefix>version <n>                      the record's version, bare (no shape): the first line of each record

- `<shape>` is the array's shape as `d0xd1x...`, or `1` for a scalar (e.g. `3x3`, `127x3`, `1`).
- `<values>` are space-separated, row-major (C order). Integers are `%d`. Floats are `%.9g` in the body record (float32
  round-trips exactly) and `%.17g` in the orbit record (float64 cameras round-trip exactly). No NaN or inf is written.
- JSON is compact and ASCII-escaped (`ensure_ascii`), since the header is ASCII.
- A key with a dot is `<group>.<sub>`: readers nest it (`pose_params.shape_params` -> `record["pose_params"]["shape_params"]`).
- A scalar reads back as a number, anything else as an array of that shape.

Which keys are integers is fixed per record (below); everything else numeric is a float.

## `b2c.mhr.*`: the refitted body (`refit_body_to_splat`)

| key | kind | meaning |
|---|---|---|
| `version` | string | `2` (bare, no shape); `1` in headers written before 2026-09-29 |
| `frame` | string | a human-readable note on the frames (below) |
| `model` | string, optional | the MHR model file the parameters replay through |
| `world_from_raw.scale` | float, `1` | similarity from SAM-3D-Body's raw metres to the splat's world |
| `world_from_raw.rotation` | float, `3x3` | |
| `world_from_raw.translation` | float, `3` | `world = scale * raw @ rotation.T + translation` |
| `pose_params.<k>` | float32 | the MHR forward's inputs, in this order: `global_rot`, `body_pose_params`, `hand_pose_params`, `scale_params`, `shape_params`, `expr_params`, `global_trans` (raw, metres), `scale_offsets` |
| `joint_parents` | **int**, `J` | skeleton parents (-1 at the root) |
| `joints` | float, `Jx3` | posed joint positions, **world** frame |
| `global_rots` | float, `Jx3x3` | posed joint rotations, **world** frame (version 2; see the caveat for 1) |

`pose_params` are in SAM-3D-Body's raw frame: replaying them through the model and applying `world_from_raw` gives the
fitted mesh in the splat's world. `joints` are already in world.

In version 2 a joint's rotation and position agree: for a bone from parent `p` to child `c`,
`joints[c] - joints[p] = scale * global_rots[p] @ offset` with `offset` the bone in the parent's local frame.

**Version 1 caveat (`docs/ply-header-global-rots-flip-2026-09-25.md`):** version 1 headers wrote `global_rots` as
`rotation @ rots_mhr`, without the `diag(1, -1, -1)` flip that the positions went through, so they are not in the same
frame as `joints`. Convert them with `rotation @ diag(1, -1, -1) @ rotation.T @ global_rots` (the version 2 value), or
replay `pose_params`.

## `b2c.orbit.*`: the orbit record (`embed_orbit_record`)

| key | kind | meaning |
|---|---|---|
| `version` | **int**, bare (no shape) | `1` |
| `helix.n_frames`, `helix.n_loops` | **int** | the helical path (the EXTENDED helix if the orbit was extended) |
| `helix.amplitude_deg`, `helix.lead_in_deg`, `helix.lead_out_deg` | float | |
| `extension.before`, `.after`, `.overlap_before`, `.overlap_after` | **int** | what the orbit extension added (0 if nothing) |
| `extension.tilt_deg` | float | how far the new frames were ramped out of the elevation band |
| `pass_frames` | **int** | one denoise pass's length |
| `anchor_frame_index` | **int**, optional | the photograph's frame on the path |
| `orbit_cameras.rotation` | float, `Nx3x3` | camera-to-world, as the frames were denoised (before `refine_cameras_final`, render resolution) |
| `orbit_cameras.position` | float, `Nx3` | camera centres, splat world |
| `orbit_cameras.intrinsics` | float, `Nx4` | `fx fy cx cy` in pixels at `image_size` |
| `orbit_cameras.image_size` | **int**, `2` | `w h` |
| `final_cameras.*` | as above, optional | the cameras the splat was trained on (refined, deliverable resolution) |
| `extras` | json | the dataset extras that serialise: `orbit_target` (3), `original_focal_length`, `focal_length_mm`, `anchor_position`, ... |
| `prompt` | json | the subject description the denoise prompts substituted for `$SUBJECT_DESC$` |
| `settings` | json | seed, resolution and framing as the run had them |
| `images` | json | `{name: {"file": ..., "sha256": ...}}` for the images written beside the .ply (`reference.png`, `anchor.png`, `front.png`) |

Camera axes are body2colmap's: `rotation` columns are the camera's right, up and back axes in world (OpenGL
convention, the camera looks down -Z). A reader that uses the images resolves `file` beside the .ply and should check
`sha256`; one that only needs the cameras can ignore them.

Integers are exactly the keys marked **int**; a reader parses those with `int()` and every other array or scalar with
`float()`.
