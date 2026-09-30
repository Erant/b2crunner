# tools/: pipeline steps and models outside a workflow

`tools/` runs single steps and models on files, for callers that are not a workflow; b2crig is the main one (clip
generation, segmentation, SAM-3D-Body fits, the subject export). The callers hand over a directory or a file and get
files back, and never import `pipeline`: the file layouts below are the contract.

    python3 tools/run.py <tool> [args...]     # any python3: the launcher is standard library only
    python3 tools/run.py --list               # each tool, its environment and the interpreter it resolves to

`run.py` starts the tool in the environment it needs (`pipeline/envs/envs.yaml`'s `python_bin`), with this checkout on
`PYTHONPATH`. Per host, `B2CRUNNER_PYTHON_<ENV>` overrides an environment's interpreter and `B2CRUNNER_PATH_<ENV>`
prepends import paths (`os.pathsep`-separated). For example, a workstation without the sam3dbody venv but with a
SAM-3D-Body checkout and its own venv:

    export B2CRUNNER_PYTHON_SAM3DBODY=~/Projects/sam-3d-body/.venv/bin/python
    export B2CRUNNER_PATH_SAM3DBODY=~/Projects/sam-3d-body

Frames are `NNNN.png` (digits only; sidecars such as `NNNN.mask.png` are skipped). Images are 8-bit BGR as OpenCV
writes them.

## Clip tools (a clip directory)

| tool | env | reads | writes |
|---|---|---|---|
| `wan_clip CLIP` | wan22 | `control/NNNN.png` (RGBA; alpha composited on `background`), optional `mask/NNNN.png` (grey VACE masks: 0 keep the control pixel, 255 generate), `reference.png` (identity), `wan.json` | `wan/NNNN.png`, `wan/timing.json` (`load_s`, `run_s`, `frames`, `width`, `height`) |
| `seg_clip CLIP [SRC OUT]` | wan22 | `SRC/NNNN.png` (default `wan`) | `OUT/NNNN.png` (default `seg`): uint8 Sapiens2 Goliath class ids |
| `pointmap_clip CLIP [--src wan control]` | wan22 | `<src>/NNNN.png` per source | `pointmap_<src>.npy`: float16 `[T, H, W, 3]`, camera-frame points on the source pixel grid |
| `upscale_clip CLIP [--resolution 1080] [--seed 0]` | seedvr2 | `wan/NNNN.png` | `wan_<resolution>/NNNN.png`: SeedVR2 as the helical workflow upscales its frames (shortest edge = resolution; batch 1, tiled VAE encode and decode) |
| `video_fit CLIP [--frames wan]` | sam3dbody | `<frames>/NNNN.png`, `seg/NNNN.png` (the person box), `cameras.json` (intrinsics) | `video_fit.npz` (below) |

`wan.json`: `{"params": {...wan22_vace_denoise params...}, "background": [r, g, b] (0-1, default grey),
"subject_desc": optional}`. The params go through the step's `resolve_params`, so any omitted one takes its default.

`cameras.json` is b2ctrain's camera file: `{"width", "height", "cameras": [{"name", "fx", "fy", "cx", "cy",
"rotation", "position"}]}`; `video_fit` reads frame i's intrinsics from camera i (the last one repeats).

`video_fit.npz`, per frame `T` (SAM-3D-Body with the seg box and the camera's intrinsics, so neither the detector nor
MoGe runs):

| key | shape | |
|---|---|---|
| `model_params` | `[T, 204]` float32 | the MHR TorchScript row: global translation x10, global rotation, 130 body params with the hand PCA expanded, 68 scales (as `mhr.npz`) |
| `cam_t` | `[T, 3]` | SAM-3D-Body's camera translation |
| `joints` | `[T, 127, 3]` | camera frame (OpenCV), `pred_joint_coords + cam_t` |
| `global_rots` | `[T, 127, 3, 3]` | SAM-3D-Body's `pred_global_rots` (MHR's frame: SAM-3D-Body flips positions, not rotations) |
| `expr` | `[T, 72]` | expression params |
| `bbox` | `[T, 4]` | the person box used, `x0 y0 x1 y1` |

## Subject tools

`export_mhr_subject PLY OUT.npz [--device cuda]` (sam3dbody): replays a delivered splat's `b2c.mhr.*` body record
(docs/ply-header-records.md) through `head_fit.build_mhr_head` and writes `mhr.npz`; it refuses if the replay misses the
header's joints by 1 mm or more.

| key | | |
|---|---|---|
| `model_params` | float32 `[204]` | the TorchScript row (as `video_fit.npz`); a consumer edits its body slots and leaves `hand_idx` alone |
| `shape_params`, `expr_params` | float32 | |
| `hand_idx` | int64 | the model-param indices of the hand joints (left then right) |
| `wfr_scale`, `wfr_rotation` `[3, 3]`, `wfr_translation` `[3]` | float64 | the header's `world_from_raw` |
| `flip` | float64 `[3, 3]` | `diag(1, -1, -1)`, MHR's frame to SAM-3D-Body's raw frame |
| `verts_world` `[V, 3]`, `joints_world` `[J, 3]` | float32 | the replayed body in the splat's world |
| `global_rots_raw` | `[J, 3, 3]` | the replay's joint rotations, MHR's frame (unflipped) |
| `faces` `[F, 3]`, `joint_parents` `[J]` | int32 | |
| `skin_vertex`, `skin_joint`, `skin_weight` | | the model's sparse skinning weights (`rig_binding_data`) |
| `mhr_model` | str | the `mhr_model.pt` path the row drives |

b2crig reads the same fields straight from a subject file's `B2C_mhr` (b2crig `subject.mhr_npz`, used by the Rig splat output), so for runs since 2026-09-29 this tool is only needed without one.

`export_glb RUN OUT.glb [--ply PLY] [--device cuda] [--no-view-cameras]` (sam3dbody, needs `b2cgltf` importable,
e.g. `B2CRUNNER_PATH_SAM3DBODY=...:~/Projects/b2cgltf`): the run as a b2cgltf **subject file** (`b2cgltf/SPEC.md`
section 4). It holds the splat, the replayed MHR body skinned to its skeleton with `B2C_mhr`, and `B2C_orbit`
(cameras, record, embedded images), and replaces `scene.ply` + sidecars + `mhr.npz` as the hand-off. Runs since
2026-09-29 write `ply/scene.glb` themselves (`export_subject`); this tool is the migration path of SPEC 4.6 for
the ones before:
- a version-1 body record has its rotations converted (`sourceRecordVersion: 1`);
- a run without an orbit record gets `"migrated": true` and only the final cameras, from `colmap/`;
- final cameras are named from `colmap/images.txt`, which must hold the same cameras in the same order.

`face_reference IMAGE OUT.png [--size 768]` (wan22): the Sapiens2 face/hair box of a front view (a run's
`ply/front.png`), squared with a 1.35 margin and resized (Lanczos) to `size`: an identity reference for close-ups.

`recover_body RUN OUT.ply [--splat PLY] [--views 3] [--every 2] [--work DIR] [--trainer b2ctrain] [--compare]`
(sam3dbody): for a run whose `ply/scene.ply` lacks the body record (it died before embedding it). Probes the splat's
surface over the run's COLMAP cameras (`b2ctrain probe --depth`), fits SAM-3D-Body on the most frontal views, places
and rigidly ICPs each fit onto the surface, refits the best with `refit_body_to_splat`'s defaults, and writes `OUT.ply`
= the splat with the `b2c.mhr.*` record. `--compare` reports the distance to an existing record.
