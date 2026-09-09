"""splat_surface / refit_body_to_splat — re-fit the MHR body to a trained splat.

The body SAM-3D-Body fitted to the one photograph is what the pipeline is
built around: the diffusion passes are conditioned on renders of it, the
cameras orbit it, and the trainer's hollow loss forbids weight behind it.
The trained splat, though, follows the generated frames, and those drift
from the render they were conditioned on: a limb a few degrees off, a
torso slimmer than the model's (SAM-3D-Body's sat 2-5 cm in front of one
subject's skin, and the navel behind it was the first thing the hollow
loss erased). These two steps move the body model onto the splat, so that
everything downstream that leans on the body — the hollow loss of the
next training, and later the binding that lets the body's skeleton drive
the splat — leans on one that is where the splat actually is.

Two steps, split by environment, like map_face_to_mesh / fit_head_to_face:

  * **`splat_surface`** (main env) measures where the splat's surface is.
    It runs `b2ctrain probe --depth` over the training cameras, which
    writes, per pixel, the depth at which the splat's accumulated alpha
    first reaches `tau` (0.5 by default: the median surface, not the fuzz
    in front of it), unprojects those depths into world space and keeps
    an oriented, strided sample of them with the silhouette pixels
    dropped. Splat centres would be the obvious target and are the wrong
    one: they sit at every depth of a soft surface and carry every
    floater.
  * **`refit_body_to_splat`** (sam3dbody env: the MHR body model)
    re-runs the body model's differentiable forward with its parameters
    free — the root's rotation and translation, the body pose, the
    per-joint scales, the shape components, optionally the hands — and
    fits the mesh to those points.

The objective is one-sided
--------------------------
The model is a naked body and the splat is a clothed one, so the two
surfaces are not meant to coincide. A surface point that ends up INSIDE
the mesh means the model pokes out through the subject's skin, and is
penalised in full. A point outside the mesh is within `clothing_allowance`
free, and beyond that penalised at `outside_weight` (a tenth): the mesh
is pulled toward the surface but not through hair or a loose garment.
A coverage term (mesh vertex to nearest surface point) keeps a limb from
shrinking away from the surface, where the outside term alone is too
gentle. Distances are point-to-plane against the mesh's own vertex
normals, correspondences re-found every step on a random subset of the
points. Every free parameter is L2-regularised toward SAM-3D-Body's fit —
the frames were generated from that pose, so the truth is near it — with
the hands held hardest (the splat's fingers are not worth fitting to) and
the shape and scales next; the root's rigid motion is free.

Coordinate frames
-----------------
`mesh_output` is in SAM-3D-Body's raw output space, as fit_head_to_face
takes and returns it (`FLIP` there). `mesh_world` is the same mesh in the
scene's world frame, as steps/render.py publishes it; the rigid transform
between the two is recovered here by Procrustes rather than by replaying
render.py's chain (auto-orient, re-centre), and must fit to a millimetre.
The refit's free root rotation and translation are the body model's own
`global_rot` and `global_trans`, so the transform stays fixed and the
updated `pose_params` regenerate the fitted mesh in both frames: replaying
them is the only way the mesh is ever re-posed from here on.

What comes out
--------------
The refitted geometry in both frames, the updated `pose_params` (now with
`global_trans` and `scale_offsets` entries a replay must pass), the
world-from-raw transform, the fit statistics, and `rig_binding`: the
rig's own skeleton hierarchy, skinning weights and inverse bind poses,
read off the MHR model file — the data a later step needs to bind splats
to the body and re-pose them.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step
from .head_fit import FLIP, build_mhr_head

logger = logging.getLogger(__name__)

#: MHR70 keypoints of the two hands (21 each, wrists included), used to
#: find the hand vertices that are left out of the fit.
HAND_KEYPOINTS = tuple(range(21, 63))

#: The probe's depth PNGs are 16-bit millimetres, 0 = no surface.
DEPTH_UNIT = 1e-3


# -- geometry helpers (pure numpy, pinned by tests) ----------------------------

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """The similarity (scale, rotation, translation) with dst ~ s R src + t.

    Least squares over matched points (Umeyama 1991). `with_scale=False`
    fixes s = 1.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError(f"umeyama needs two matched Nx3 point sets, got {src.shape} and {dst.shape}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1.0
    rot = u @ d @ vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = float((sig * np.diag(d)).sum() / var_s) if with_scale else 1.0
    t = mu_d - scale * rot @ mu_s
    return scale, rot, t


def decode_depth_png(png: np.ndarray) -> np.ndarray:
    """A probe `*.zfirst.png` (uint16 millimetres, 0 = none) as float32 metres, NaN where none."""
    if png.ndim != 2 or png.dtype != np.uint16:
        raise ValueError(f"a probe depth PNG is a 2-D uint16 image, got {png.dtype} {png.shape}")
    depth = png.astype(np.float32) * DEPTH_UNIT
    depth[png == 0] = np.nan
    return depth


def unproject_depth(depth: np.ndarray, camera: Any, stride: int, edge_jump: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """World-space surface points and outward normals from one depth map.

    `depth` is camera-space z in metres (NaN = no surface) at the pixels of
    `camera` (a body2colmap Camera: OpenGL camera-to-world `rotation`,
    `position`, `fx/fy/cx/cy`). Every `stride`-th pixel in each direction
    is unprojected; its normal is the cross product of the neighbouring
    pixels' points, oriented toward the camera. A pixel whose four
    neighbours are not all covered, or whose depth jumps by more than
    `edge_jump` to one of them, is a silhouette or an occlusion boundary
    and is dropped: its normal would be meaningless and its position is
    the grazing skin of the splat, not the surface.
    """
    depth = np.asarray(depth, np.float32)
    height, width = depth.shape
    if (camera.width, camera.height) != (width, height):
        raise ValueError(f"depth map is {width}x{height} but the camera is {camera.width}x{camera.height}")
    stride = max(int(stride), 1)
    us = np.arange(stride // 2, width, stride)
    vs = np.arange(stride // 2, height, stride)
    us = us[(us >= 1) & (us <= width - 2)]
    vs = vs[(vs >= 1) & (vs <= height - 2)]
    uu, vv = np.meshgrid(us, vs)

    def points(u, v):
        z = depth[v, u]
        x = (u + 0.5 - camera.cx) / camera.fx * z
        y = (v + 0.5 - camera.cy) / camera.fy * z
        return np.stack([x, y, z], -1), z

    centre, zc = points(uu, vv)
    right, zr = points(uu + 1, vv)
    left, zl = points(uu - 1, vv)
    down, zd = points(uu, vv + 1)
    up, zu = points(uu, vv - 1)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(zc) & np.isfinite(zr) & np.isfinite(zl) & np.isfinite(zd) & np.isfinite(zu)
        jump = np.maximum.reduce([np.abs(zr - zc), np.abs(zl - zc), np.abs(zd - zc), np.abs(zu - zc)])
        ok &= jump <= edge_jump
    if not ok.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    centre, right, left, down, up = (a[ok] for a in (centre, right, left, down, up))
    normal = np.cross(right - left, down - up)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    good = length[:, 0] > 0
    centre, normal = centre[good], normal[good] / length[good]
    # Toward the camera, which sits at the origin of this (OpenCV) frame.
    facing = (normal * -centre).sum(1) < 0
    normal[facing] *= -1
    rot_cv = np.asarray(camera.rotation, np.float64) @ np.diag([1.0, -1.0, -1.0])
    world = centre.astype(np.float64) @ rot_cv.T + np.asarray(camera.position, np.float64)
    normal_world = normal.astype(np.float64) @ rot_cv.T
    return world.astype(np.float32), normal_world.astype(np.float32)


def hand_vertex_mask(vertices: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """True for every vertex whose nearest MHR70 keypoint is on a hand."""
    vertices = np.asarray(vertices, np.float32)
    keypoints = np.asarray(keypoints, np.float32)
    if len(keypoints) != 70:
        raise ValueError(f"hand_vertex_mask expects the 70 MHR keypoints, got {len(keypoints)}")
    nearest = np.empty(len(vertices), np.int64)
    for start in range(0, len(vertices), 65536):
        chunk = vertices[start:start + 65536]
        d = ((chunk[:, None, :] - keypoints[None, :, :]) ** 2).sum(-1)
        nearest[start:start + 65536] = d.argmin(1)
    return np.isin(nearest, HAND_KEYPOINTS)


def cameras_json(cameras: Sequence[Any]) -> Dict[str, Any]:
    """The trainer's cameras.json (OpenGL camera-to-world, as body2colmap's
    SplatRenderer writes it), one entry per camera, named by index."""
    if not cameras:
        raise ValueError("splat_surface: no cameras")
    width, height = int(cameras[0].width), int(cameras[0].height)
    entries = []
    for i, cam in enumerate(cameras):
        if (int(cam.width), int(cam.height)) != (width, height):
            raise ValueError(f"splat_surface: camera {i} is {cam.width}x{cam.height} but camera 0 is "
                             f"{width}x{height}; the probe renders one resolution")
        entries.append({
            "name": f"frame_{i:05d}.png",
            "fx": float(cam.fx), "fy": float(cam.fy), "cx": float(cam.cx), "cy": float(cam.cy),
            "position": [float(v) for v in cam.position],
            "rotation": [[float(v) for v in row] for row in cam.rotation],
        })
    return {"width": width, "height": height, "cameras": entries}


def write_points_ply(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    """A binary little-endian point-cloud .ply with normals, for viewers."""
    points = np.asarray(points, np.float32)
    normals = np.asarray(normals, np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.concatenate([points, normals], 1).astype("<f4").tobytes())


def rig_binding_data(mhr: Any) -> Dict[str, np.ndarray]:
    """The skeleton and skinning of an MHR model file, as numpy.

    `mhr` is the TorchScript module `mhr_model.pt` loads to (the head's
    `.mhr`). Its buffers hold everything a binding needs, none of it
    hidden: the joint hierarchy, the sparse linear-blend-skinning weights
    (about 2.8 joints per vertex) and each joint's inverse bind pose as
    translation + quaternion (xyzw) + scale. Units are the rig's own —
    centimetres, before the `/100` and the `FLIP` that `mhr_forward`'s
    callers apply — and are left that way so the numbers match a
    `skel_state` the rig returns.
    """
    buffers = dict(mhr.named_buffers())
    prefix = "character_torch."
    wanted = {
        "joint_parents": "skeleton.joint_parents",
        "joint_translation_offsets": "skeleton.joint_translation_offsets",
        "joint_prerotations": "skeleton.joint_prerotations",
        "inverse_bind_pose": "linear_blend_skinning.inverse_bind_pose",
        "skin_vertex": "linear_blend_skinning.vert_indices_flattened",
        "skin_joint": "linear_blend_skinning.skin_indices_flattened",
        "skin_weight": "linear_blend_skinning.skin_weights_flattened",
        "rest_vertices": "mesh.rest_vertices",
        "faces": "mesh.faces",
    }
    out = {}
    for key, name in wanted.items():
        full = prefix + name
        if full not in buffers:
            raise RuntimeError(f"rig_binding_data: the MHR model has no buffer {full!r}; its layout "
                               f"differs from the mhr_model.pt this was written against")
        out[key] = buffers[full].detach().cpu().numpy().copy()
    n_joints = len(out["joint_parents"])
    n_verts = len(out["rest_vertices"])
    if out["skin_joint"].max() >= n_joints or out["skin_vertex"].max() >= n_verts:
        raise RuntimeError("rig_binding_data: skinning indices are out of range for the rig's joints/vertices")
    return out


# -- splat_surface -------------------------------------------------------------

@register_step("splat_surface")
class SplatSurfaceStep(Step):
    """Sample the trained splat's surface as oriented world-space points.

    inputs:  {"splat_path": str — the trained .ply,
              "cameras": List[Camera] — the training cameras (body2colmap)}
    outputs: {"surface": {"points": Nx3 float32 world, "normals": Nx3 float32
              (outward), "view": N int32 camera index, "n_cameras", "tau"}}
    """

    PARAMS = (
        Param("trainer_path", str, "b2ctrain",
              "The trainer binary; its `probe --depth` renders the surface depth", advanced=True),
        Param("every", int, 1, "Probe every N-th training camera", minimum=1),
        Param("tau", float, 0.5,
              "Accumulated alpha that marks the surface. 0.5 is the median surface; 0.1 "
              "(the hollow loss's first surface) sits on the soft front of it",
              minimum=0.01, maximum=0.99),
        Param("stride", int, 8, "Keep every N-th pixel in each direction", minimum=1),
        Param("edge_jump", float, 0.02,
              "A pixel whose depth differs by more than this (metres) from a neighbour is a "
              "silhouette and is dropped", minimum=0.0),
        Param("max_points", int, 300000, "Random subsample cap over all cameras", minimum=1000),
        Param("device", int, 0, "CUDA device index for the probe", advanced=True),
        Param("keep_dir", str, "",
              "Keep the probe's output (depth maps, probe.json) here for inspection", advanced=True),
        Param("debug_dir", str, "",
              "Write the sampled surface as surface.ply here (the run's debug bundle)"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import tempfile

        import cv2

        splat_path = Path(str(inputs["splat_path"]))
        if not splat_path.is_file():
            raise FileNotFoundError(f"splat_surface: no splat at {splat_path}")
        cameras = list(inputs["cameras"])
        trainer = params["trainer_path"]
        if shutil.which(trainer) is None and not Path(trainer).is_file():
            raise RuntimeError(f"splat_surface: trainer binary {trainer!r} not found on PATH")
        help_text = subprocess.run([trainer, "probe", "--help"], capture_output=True, text=True,
                                   timeout=60).stdout
        if "--depth" not in help_text:
            raise RuntimeError(f"splat_surface: {trainer!r} has no `probe --depth`; a b2ctrain build "
                               f"from 2026-09-09 or later is needed")

        keep = Path(params["keep_dir"]) if params["keep_dir"] else None
        with tempfile.TemporaryDirectory(prefix="splat_surface_") as tmp:
            out_dir = keep or Path(tmp)
            out_dir.mkdir(parents=True, exist_ok=True)
            cams_path = out_dir / "cameras.json"
            cams_path.write_text(json.dumps(cameras_json(cameras)))
            cmd = [trainer, "probe", "--splat", str(splat_path), "--cameras", str(cams_path),
                   "--output-dir", str(out_dir), "--depth", "--tau", str(params["tau"]),
                   "--every", str(params["every"]), "--device", str(params["device"])]
            logger.info("splat_surface: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                tail = "\n".join((result.stderr or result.stdout).splitlines()[-15:])
                raise RuntimeError(f"splat_surface: probe exited {result.returncode}:\n{tail}")
            depth_files = sorted(out_dir.glob("frame_*.zfirst.png"))
            if not depth_files:
                raise RuntimeError(f"splat_surface: the probe wrote no depth maps into {out_dir}")

            points, normals, views = [], [], []
            for path in depth_files:
                index = int(path.name[len("frame_"):len("frame_") + 5])
                png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if png is None:
                    raise RuntimeError(f"splat_surface: could not read {path}")
                p, n = unproject_depth(decode_depth_png(png), cameras[index], params["stride"],
                                       params["edge_jump"])
                points.append(p)
                normals.append(n)
                views.append(np.full(len(p), index, np.int32))
        points = np.concatenate(points) if points else np.zeros((0, 3), np.float32)
        normals = np.concatenate(normals) if normals else np.zeros((0, 3), np.float32)
        views = np.concatenate(views) if views else np.zeros(0, np.int32)
        if len(points) < 1000:
            raise RuntimeError(f"splat_surface: only {len(points)} surface points from {len(depth_files)} "
                               f"cameras; is the splat empty, or the cameras looking elsewhere?")
        total = len(points)
        if total > params["max_points"]:
            keep_idx = np.random.RandomState(0).choice(total, params["max_points"], replace=False)
            keep_idx.sort()
            points, normals, views = points[keep_idx], normals[keep_idx], views[keep_idx]
        logger.info("splat_surface: %d surface points (of %d sampled) from %d of %d cameras at "
                    "tau %.2f, stride %d", len(points), total, len(depth_files), len(cameras),
                    params["tau"], params["stride"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            write_points_ply(debug / "surface.ply", points, normals)
        return {"surface": {"points": points, "normals": normals, "view": views,
                            "n_cameras": len(depth_files), "tau": float(params["tau"])}}


# -- refit_body_to_splat -------------------------------------------------------

def _check_mesh_world(mesh_world: Any) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(mesh_world, (tuple, list)) or len(mesh_world) != 2:
        raise ValueError("refit_body_to_splat: mesh_world must be the (vertices, faces) pair "
                         "steps/render.py publishes as `mesh` (scene.mesh_world)")
    vertices = np.asarray(mesh_world[0], np.float64)
    faces = np.asarray(mesh_world[1], np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"refit_body_to_splat: mesh_world has vertices {vertices.shape} and faces "
                         f"{faces.shape}; expected Nx3 and Mx3")
    return vertices, faces


def _check_surface(surface: Any) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(surface, dict) or "points" not in surface or "normals" not in surface:
        raise ValueError("refit_body_to_splat: `surface` must be splat_surface's output "
                         "(a dict with points and normals)")
    points = np.asarray(surface["points"], np.float32)
    normals = np.asarray(surface["normals"], np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError(f"refit_body_to_splat: surface points {points.shape} / normals {normals.shape}")
    if len(points) < 1000:
        raise ValueError(f"refit_body_to_splat: only {len(points)} surface points; nothing to fit to")
    return points, normals


@register_step("refit_body_to_splat")
class RefitBodyToSplatStep(Step):
    """Re-fit the MHR body parameters to the splat's surface.

    inputs:  {"mesh_output": dict — sam3d_body's (or fit_head_to_face's)
              outputs INCLUDING pose_params, in SAM-3D-Body's raw space,
              "mesh_world": (vertices, faces) — the same mesh in the world
              frame, as steps/render.py publishes it,
              "surface": splat_surface's output}
    outputs: {"vertices", "keypoints_3d", "joints", "global_rots" — the
              refitted geometry in the raw space,
              "mesh_world" — (vertices, faces) in the world frame,
              "pose_params" — updated, with `global_trans` and
              `scale_offsets` entries a replay must pass,
              "world_from_raw" — {"scale", "rotation", "translation"},
              "rig_binding" — the rig's skeleton and skinning (see
              `rig_binding_data`),
              "body_refit_stats"}
    """

    PARAMS = (
        Param("iterations", int, 300,
              "Adam steps of the pose stage and again of the shape stage; the rigid+scale "
              "warm-up runs half as many", minimum=1),
        Param("learning_rate", float, 0.01,
              "Adam step; rotations are in radians, so 0.01 is ~0.6 deg", advanced=True),
        Param("points_per_step", int, 100000,
              "Surface points re-sampled each step for the correspondences", minimum=1000),
        Param("clothing_allowance", float, 0.01,
              "Metres a surface point may sit outside the body before it pulls on it"),
        Param("inside_weight", float, 1.0,
              "Weight of the squared distance (cm) of surface points inside the body"),
        Param("outside_weight", float, 0.1,
              "Weight of the squared distance (cm) beyond the clothing allowance"),
        Param("coverage_weight", float, 0.1,
              "Weight of the squared distance (cm) from each body vertex to its nearest "
              "surface point, within coverage_radius"),
        Param("coverage_radius", float, 0.05,
              "Metres; a vertex farther than this from every surface point is unobserved "
              "and gets no coverage term"),
        Param("outlier_distance", float, 0.15,
              "Metres; surface points farther than this from SAM-3D-Body's mesh are not "
              "the body (hair, a garment, a floater) and are dropped before the fit"),
        Param("pose_regularisation", float, 10.0,
              "L2 pull on the body pose deltas (per rad^2, summed) against the fit loss in cm^2"),
        Param("hand_regularisation", float, 100.0, "L2 pull on the hand pose deltas"),
        Param("scale_regularisation", float, 10.0, "L2 pull on the per-joint scale offsets"),
        Param("shape_regularisation", float, 1.0, "L2 pull on the shape components, per sigma^2"),
        Param("fit_scale", bool, True, "Fit the per-joint scale offsets"),
        Param("fit_shape", bool, True, "Fit the shape components (last stage)"),
        Param("fit_hands", bool, False,
              "Fit the hand pose too. Off: the hands keep SAM-3D-Body's pose and their "
              "vertices take no part in the fit either way (see exclude_hands)"),
        Param("exclude_hands", bool, True,
              "Leave the hand vertices out of the fit: the splat's fingers are rarely "
              "worth fitting to"),
        Param("max_replay_drift", float, 1e-3,
              "Metres; replaying pose_params must reproduce mesh_output to this", advanced=True),
        Param("checkpoint_repo", str, "facebook/sam-3d-body-dinov3",
              "HF repo the checkpoint is pulled from", advanced=True),
        Param("checkpoint_dir", str, None,
              "A local snapshot directory to use instead of downloading", advanced=True),
        Param("mhr_path", str, None,
              "The mhr_model.pt to load; empty means assets/mhr_model.pt inside the "
              "checkpoint directory", advanced=True),
        Param("device", str, "cuda", "Torch device", advanced=True),
        Param("seed", int, 0, "Seed of the per-step point subsets", advanced=True),
        Param("debug_dir", str, "",
              "Write mesh_before.ply, mesh_refit.ply (world frame) and stats.json here "
              "(the run's debug bundle)"),
    )

    def __init__(self) -> None:
        self._head = None

    def load(self, params: Dict[str, Any]) -> None:
        self._head = build_mhr_head(params["checkpoint_repo"], params["checkpoint_dir"],
                                    params["mhr_path"], params["device"])

    def unload(self) -> None:
        self._head = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        mesh = inputs["mesh_output"]
        if not isinstance(mesh, dict) or mesh.get("pose_params") is None:
            raise KeyError(
                "refit_body_to_splat needs 'pose_params' inside mesh_output — wire "
                "sam3d_body's pose_params output into the scene. It re-runs the MHR "
                "body model's forward with those as the free variables."
            )
        pose_params = mesh["pose_params"]
        if hasattr(pose_params, "item") and not isinstance(pose_params, dict):
            pose_params = pose_params.item()      # an npz round-trip's object array
        world_vertices, world_faces = _check_mesh_world(inputs["mesh_world"])
        points_np, normals_np = _check_surface(inputs["surface"])
        raw_vertices = np.asarray(mesh["vertices"], np.float64)
        faces_np = np.asarray(mesh["faces"], np.int64)
        if raw_vertices.shape != world_vertices.shape:
            raise ValueError(f"refit_body_to_splat: mesh_output has {len(raw_vertices)} vertices and "
                             f"mesh_world {len(world_vertices)}; they must be the same mesh")

        # --- the frame: world = s R raw + t, recovered, not replayed ---------
        scale, rot, trans = umeyama(raw_vertices, world_vertices)
        fitted = scale * raw_vertices @ rot.T + trans
        frame_residual = float(np.abs(fitted - world_vertices).max())
        if frame_residual > 2e-3:
            raise ValueError(
                f"refit_body_to_splat: mesh_world is not a rigid placement of mesh_output "
                f"({frame_residual * 1000:.1f} mm off after the best similarity). Something "
                f"deformed one of them; the refit needs the same mesh in both frames.")
        if abs(scale - 1.0) > 0.02:
            logger.warning("refit_body_to_splat: mesh_world is scaled %.4f against mesh_output; "
                           "the pipeline's world frame is metric, is this the right mesh?", scale)

        import torch

        if self._head is None:
            self.load(params)
        head = self._head
        device = params["device"]
        gen = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
        flip = torch.tensor(FLIP, dtype=torch.float32, device=device)
        rot_t = torch.tensor(rot, dtype=torch.float32, device=device)
        trans_t = torch.tensor(trans, dtype=torch.float32, device=device)

        def tensor(value):
            out = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
            return out[None] if out.ndim == 1 else out

        g0 = tensor(pose_params["global_rot"])
        b0 = tensor(pose_params["body_pose_params"])
        h0 = tensor(pose_params["hand_pose_params"])
        sc0 = tensor(pose_params["scale_params"])
        sh0 = tensor(pose_params["shape_params"])
        ex0 = tensor(pose_params["expr_params"])
        n_scales = int(head.scale_mean.shape[0])
        so0 = torch.zeros(1, n_scales, device=device)
        if pose_params.get("scale_offsets") is not None:
            so0 = tensor(pose_params["scale_offsets"])
        gt0 = torch.zeros(1, 3, device=device)
        if pose_params.get("global_trans") is not None:
            gt0 = tensor(pose_params["global_trans"])
        n_keypoints = len(np.asarray(mesh["keypoints_3d"]))

        def forward(dg, dt, db, dh, dso, dsh):
            verts, keypoints, joints, rots = head.mhr_forward(
                global_trans=gt0 + dt, global_rot=g0 + dg, body_pose_params=b0 + db,
                hand_pose_params=h0 + dh, scale_params=sc0, shape_params=sh0 + dsh,
                expr_params=ex0, return_keypoints=True, return_joint_coords=True,
                return_joint_rotations=True, scale_offsets=so0 + dso,
            )
            return verts[0] * flip, keypoints[0, :n_keypoints] * flip, joints[0] * flip, rots[0]

        def to_world(raw):
            return scale * raw @ rot_t.T + trans_t

        zeros = {
            "dg": torch.zeros(1, 3, device=device), "dt": torch.zeros(1, 3, device=device),
            "db": torch.zeros_like(b0), "dh": torch.zeros_like(h0),
            "dso": torch.zeros_like(so0), "dsh": torch.zeros_like(sh0),
        }
        with torch.no_grad():
            verts0, keypoints0, joints0, _ = forward(**zeros)
        drift = float(np.abs(verts0.cpu().numpy() - raw_vertices).max())
        if drift > params["max_replay_drift"]:
            raise RuntimeError(
                f"refit_body_to_splat: replaying the pose parameters does not reproduce "
                f"the input mesh ({drift * 1000:.2f} mm off). Either something deformed the "
                f"vertices without updating the parameters, or mesh_output and pose_params "
                f"are from different fits.")

        # --- which vertices take part ---------------------------------------
        keep = np.ones(len(raw_vertices), bool)
        n_hand = 0
        if params["exclude_hands"]:
            keypoints_np = np.asarray(mesh["keypoints_3d"])
            if len(keypoints_np) == 70:
                hands = hand_vertex_mask(raw_vertices, keypoints_np)
                keep &= ~hands
                n_hand = int(hands.sum())
            else:
                logger.warning("refit_body_to_splat: keypoints_3d has %d entries, not the 70 of "
                               "MHR70; cannot tell the hands apart, fitting every vertex",
                               len(keypoints_np))
        fit_idx = torch.tensor(np.flatnonzero(keep), device=device)
        faces = torch.tensor(faces_np, device=device)

        def vertex_normals(verts):
            a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            fn = torch.cross(b - a, c - a, dim=1)
            vn = torch.zeros_like(verts)
            for k in range(3):
                vn.index_add_(0, faces[:, k], fn)
            return vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-12)

        # The winding of the rig's faces decides which way the normals point;
        # measure it instead of assuming: outward means away from the body.
        with torch.no_grad():
            w0 = to_world(verts0)
            n0 = vertex_normals(w0)
            outward = 1.0 if float(((w0 - w0.mean(0)) * n0).sum()) > 0 else -1.0

        # --- the targets ------------------------------------------------------
        points = torch.tensor(points_np, device=device)
        normals = torch.tensor(normals_np, device=device)

        def nearest(p, v, chunk=4096):
            """Per point the nearest vertex; per vertex the nearest of these points."""
            nn = torch.empty(len(p), dtype=torch.long, device=device)
            dmin = torch.full((len(v),), float("inf"), device=device)
            arg = torch.zeros(len(v), dtype=torch.long, device=device)
            for start in range(0, len(p), chunk):
                d = torch.cdist(p[start:start + chunk], v)
                nn[start:start + chunk] = d.argmin(1)
                m, a = d.min(0)
                better = m < dmin
                dmin[better] = m[better]
                arg[better] = a[better] + start
            return nn, dmin, arg

        with torch.no_grad():
            nn_all, _, _ = nearest(points, w0)
            far = (points - w0[nn_all]).norm(dim=1) > params["outlier_distance"]
        n_outliers = int(far.sum())
        points, normals = points[~far], normals[~far]
        if len(points) < 1000:
            raise ValueError(f"refit_body_to_splat: {n_outliers} of {len(points_np)} surface points are "
                             f"farther than {params['outlier_distance']} m from the body; the mesh and "
                             f"the splat are not in the same frame")

        allowance_cm = 100.0 * params["clothing_allowance"]
        radius_cm = 100.0 * params["coverage_radius"]
        w_in, w_out, w_cov = params["inside_weight"], params["outside_weight"], params["coverage_weight"]
        lam = {"db": params["pose_regularisation"], "dh": params["hand_regularisation"],
               "dso": params["scale_regularisation"], "dsh": params["shape_regularisation"]}

        def signed_cm(p, verts_world, vn, nn):
            return ((p - verts_world[nn]) * (outward * vn[nn])).sum(1) * 100.0

        def loss_terms(free, subset):
            verts_raw, _, _, _ = forward(**{**zeros, **free})
            verts_world = to_world(verts_raw)
            vn = vertex_normals(verts_world).detach()
            v_fit = verts_world[fit_idx]
            p = points[subset]
            with torch.no_grad():
                nn, dmin, arg = nearest(p, v_fit.detach())
            s = signed_cm(p, v_fit, vn[fit_idx], nn)
            inside = torch.relu(-s)
            outside = torch.relu(s - allowance_cm)
            data = w_in * (inside ** 2).mean() + w_out * (outside ** 2).mean()
            d_cov = (v_fit - p[arg]).norm(dim=1) * 100.0
            observed = dmin * 100.0 < radius_cm
            coverage = w_cov * (d_cov[observed] ** 2).mean() if bool(observed.any()) else data * 0
            prior = sum(lam[k] * (free[k] ** 2).sum() for k in lam if k in free)
            return data, coverage, prior

        def solve(free, iterations, learning_rate):
            optimiser = torch.optim.Adam([{"params": [t], "lr": learning_rate} for t in free.values()])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=iterations)
            n_sub = min(int(params["points_per_step"]), len(points))
            for _ in range(iterations):
                subset = torch.randperm(len(points), generator=gen)[:n_sub].to(device)
                optimiser.zero_grad()
                data, coverage, prior = loss_terms(free, subset)
                (data + coverage + prior).backward()
                optimiser.step()
                schedule.step()

        def stats(free):
            """Over every surface point and every fitted vertex, not the step's subset."""
            with torch.no_grad():
                verts_raw, keypoints, joints, rots = forward(**{**zeros, **free})
                verts_world = to_world(verts_raw)
                vn = vertex_normals(verts_world)
                v_fit = verts_world[fit_idx]
                nn, dmin, _ = nearest(points, v_fit)
                s = signed_cm(points, v_fit, vn[fit_idx], nn)
                a = s.abs()
                observed = dmin * 100.0 < radius_cm
                return {
                    "surface_to_body_cm": {
                        "median_abs": float(a.median()), "p90_abs": float(a.kthvalue(int(0.9 * len(a)) or 1).values),
                        "mean_signed": float(s.mean()),
                        "inside_fraction_5mm": float((s < -0.5).float().mean()),
                        "outside_fraction_3cm": float((s > 3.0).float().mean()),
                    },
                    "body_to_surface_cm": {
                        "median": float((dmin[observed] * 100.0).median()) if bool(observed.any()) else float("nan"),
                        "observed_fraction": float(observed.float().mean()),
                    },
                }, (verts_raw, keypoints, joints, rots, verts_world)

        before, _ = stats({})

        # --- the fit: rigid + scales, then the pose, then the shape ----------
        free = {"dg": zeros["dg"].clone().requires_grad_(True),
                "dt": zeros["dt"].clone().requires_grad_(True)}
        if params["fit_scale"]:
            free["dso"] = zeros["dso"].clone().requires_grad_(True)
        iterations = int(params["iterations"])
        lr = params["learning_rate"]
        solve(free, max(iterations // 2, 1), lr * 2)
        free["db"] = zeros["db"].clone().requires_grad_(True)
        if params["fit_hands"]:
            free["dh"] = zeros["dh"].clone().requires_grad_(True)
        solve(free, iterations, lr)
        if params["fit_shape"]:
            free["dsh"] = zeros["dsh"].clone().requires_grad_(True)
            solve(free, iterations, lr)
        final = {k: v.detach() for k, v in free.items()}
        after, (verts, keypoints, joints, rots, verts_world) = stats(final)

        # --- outputs ----------------------------------------------------------
        def delta(key):
            return final[key][0].cpu().numpy() if key in final else np.zeros(zeros[key].shape[1], np.float32)

        new_pose = {k: np.array(v, copy=True) for k, v in pose_params.items()}
        new_pose["global_rot"] = (g0[0] + final["dg"][0]).cpu().numpy().astype(np.float32)
        new_pose["global_trans"] = (gt0[0] + final["dt"][0]).cpu().numpy().astype(np.float32)
        new_pose["body_pose_params"] = (b0[0] + torch.as_tensor(delta("db"), device=device)).cpu().numpy().astype(np.float32)
        new_pose["hand_pose_params"] = (h0[0] + torch.as_tensor(delta("dh"), device=device)).cpu().numpy().astype(np.float32)
        new_pose["shape_params"] = (sh0[0] + torch.as_tensor(delta("dsh"), device=device)).cpu().numpy().astype(np.float32)
        new_pose["scale_offsets"] = (so0[0] + torch.as_tensor(delta("dso"), device=device)).cpu().numpy().astype(np.float32)

        verts_np = verts.cpu().numpy()
        joints_np = joints.cpu().numpy()
        joint_shift = np.linalg.norm(joints_np - joints0.cpu().numpy(), axis=1)
        vertex_shift = np.linalg.norm(verts_np - verts0.cpu().numpy(), axis=1)
        db = delta("db")[:130]
        stats_out = {
            "before": before, "after": after,
            "surface_points": {"used": int(len(points)), "outliers_dropped": n_outliers,
                               "tau": inputs["surface"].get("tau")},
            "vertices": {"fitted": int(keep.sum()), "hand_excluded": n_hand},
            "world_from_raw": {"scale": scale, "residual_mm": frame_residual * 1000},
            "root": {"rotation_delta_deg": [float(v) for v in np.degrees(delta("dg"))],
                     "translation_delta": [float(v) for v in delta("dt")]},
            "pose_delta_deg": {"rms": float(np.degrees(np.sqrt((db ** 2).mean()))),
                               "max": float(np.degrees(np.abs(db).max()))},
            "scale_offset_delta": {"rms": float(np.sqrt((delta("dso") ** 2).mean())),
                                   "max": float(np.abs(delta("dso")).max())},
            "shape_delta_sigma": {"rms": float(np.sqrt((delta("dsh") ** 2).mean())),
                                  "max": float(np.abs(delta("dsh")).max())},
            "joint_shift_mm": {"median": float(np.median(joint_shift) * 1000),
                               "max": float(joint_shift.max() * 1000)},
            "vertex_shift_mm": {"median": float(np.median(vertex_shift) * 1000),
                                "p90": float(np.percentile(vertex_shift, 90) * 1000),
                                "max": float(vertex_shift.max() * 1000)},
        }
        logger.info(
            "refit_body_to_splat: surface-to-body |d| median %.2f -> %.2f cm, p90 %.2f -> %.2f, "
            "inside>5mm %.1f%% -> %.1f%%; body-to-surface median %.2f -> %.2f cm; pose %.1f deg rms "
            "(max %.1f), scales %.3f rms, shape %.2f sigma rms, root moved %.1f mm; vertices moved "
            "%.1f mm median, %.1f max; %d points (%d outliers dropped), %d hand vertices excluded",
            before["surface_to_body_cm"]["median_abs"], after["surface_to_body_cm"]["median_abs"],
            before["surface_to_body_cm"]["p90_abs"], after["surface_to_body_cm"]["p90_abs"],
            100 * before["surface_to_body_cm"]["inside_fraction_5mm"],
            100 * after["surface_to_body_cm"]["inside_fraction_5mm"],
            before["body_to_surface_cm"]["median"], after["body_to_surface_cm"]["median"],
            stats_out["pose_delta_deg"]["rms"], stats_out["pose_delta_deg"]["max"],
            stats_out["scale_offset_delta"]["rms"], stats_out["shape_delta_sigma"]["rms"],
            float(np.linalg.norm(delta("dt"))) * 1000 * scale,
            stats_out["vertex_shift_mm"]["median"], stats_out["vertex_shift_mm"]["max"],
            len(points), n_outliers, n_hand,
        )
        mesh_world_out = (verts_world.cpu().numpy().astype(np.float32), np.asarray(world_faces, np.int32))
        if params["debug_dir"]:
            from .brush import _write_mesh_ply
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            _write_mesh_ply(debug / "mesh_before.ply", world_vertices.astype(np.float32), mesh_world_out[1])
            _write_mesh_ply(debug / "mesh_refit.ply", mesh_world_out[0], mesh_world_out[1])
            (debug / "stats.json").write_text(json.dumps(stats_out, indent=1))
        return {
            "vertices": verts_np.astype(np.asarray(mesh["vertices"]).dtype, copy=False),
            "keypoints_3d": keypoints.cpu().numpy().astype(np.asarray(mesh["keypoints_3d"]).dtype, copy=False),
            "joints": joints_np,
            "global_rots": rots.cpu().numpy(),
            "mesh_world": mesh_world_out,
            "pose_params": new_pose,
            "world_from_raw": {"scale": float(scale), "rotation": rot.astype(np.float64),
                               "translation": trans.astype(np.float64)},
            "rig_binding": rig_binding_data(head.mhr),
            "body_refit_stats": stats_out,
        }
