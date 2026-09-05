"""fit_head_to_face — re-fit SAM-3D-Body's head to the photograph's face.

Two steps, split by environment:

  * **`map_face_to_mesh`** (main env: pyrender + MediaPipe) builds a dense
    correspondence between MediaPipe's 468 face landmarks and the MHR mesh's
    vertices, by rendering THIS mesh's head, shaded, through SAM-3D-Body's
    own camera and running the landmarker on the render. Each landmark is
    snapped to the nearest visible vertex. The map is a property of the MHR
    topology and the mesh's own face; nothing about the photograph enters.
  * **`fit_head_to_face`** (sam3dbody env: the MHR body model) re-runs the
    body model's differentiable forward with the head's own parameters free
    — the neck/head joint rotations, the head joint's scale and the
    head-only shape components — and minimises the 2D distance between
    those mapped vertices and MediaPipe's landmarks on the PHOTO. The result
    is a new mesh, skeleton and an updated `pose_params` that regenerate it,
    so nothing downstream has to know the head was touched.

Why a fit and not a fixed nod
-----------------------------
The step this replaced (`head_angle_fix`, retired 2026-09-04) nodded the
head back to a fixed 10 deg of forward lean. It had no idea where the
photograph's face was looking, and measured on
cyber2_6f it nods the head 30 deg AWAY from it: the landmark residual of the
mesh's head goes 10.5 px (raw fit) -> 20.7 px (auto nod), where this step
reaches 2.4 px. The face splat, which sits on the photo's rays by
construction, then floats on a mesh head looking somewhere else, and the
outline the diffusion pass conditions on disagrees with the one real frame.

Why the body model's parameters and not a vertex deformation
------------------------------------------------------------
Three fits were measured on the same subject (output/face_mask_compare/
headfit_*.png in the run that produced this step):

  * a rigid nod + isotropic head scale, fitted densely, goes to -23 deg —
    it pitches the head down to foreshorten a mid-face the body model made
    too long for this subject, and barely reduces the residual (9.3 -> 8.5
    px). Pose fitted without shape absorbs the shape error;
  * a nod + anisotropic head scale (width x1.27, height x0.77) reaches 3.6
    px but squashes the skull — plausible from the front, flat in profile;
  * the body model's own head parameters reach 2.4 px with an intact head
    from every angle, and the pose parameters stay a faithful handle on the
    geometry — which is the property this step's own round-trip gate
    depends on.

What is fitted
--------------
Free variables, all discovered by perturbation against the MHR model that
ships with `facebook/sam-3d-body-dinov3` (each is checked at run time, see
`_check_handles`):

  * `body_pose_params[18..23]` — the six neck/head joint rotations: each
    moves >= 95% of its displaced vertices inside the head;
  * `scale_offsets[4]` — the head joint's own scale, applied through
    `mhr_forward(scale_offsets=...)`: the one handle that grows the head
    and nothing else (99% head);
  * `shape_params[20..44]` — the head-only shape components (each moves
    the head 0.7-3 mm per sigma and the body by 0.0), L2-regularised
    toward SAM-3D-Body's fit.

Everything else — global orientation, `cam_t`, the body's pose and shape,
the hands — is exactly what SAM-3D-Body fitted. The objective is a Huber
loss over the landmarks of the eyes, brows, nose, lips and cheeks; the face
oval (jaw and hairline contour) is scored but not fitted, because MediaPipe
places it on the visible silhouette and a shaded body-model render and a
photograph with hair disagree about where that is.

Measured on cyber2_6f (720x1280, head at 2.2 m): features rms 10.46 px ->
2.38 px at `shape_regularisation` 1.0 (shape moved 1.8 sigma rms, 4.3
max — the photograph's face is genuinely wide, W/H 1.20 against the fit's
0.82); the head pitched 7 deg down, to where the photograph looks. The
same fit at 0.3 reaches 2.05 px for 2.25 sigma rms and looks the same.

Coordinate frames
-----------------
All inputs and outputs are in SAM-3D-Body's raw output space (what
`pred_vertices` and `pred_keypoints_3d` share, `vertices + cam_t` being the
OpenCV camera frame). `mhr_head.forward` applies `diag(1,-1,-1)` to the
model's output AFTER `mhr_forward`; replaying a pose has to do the same or
the body lands ~2.8 m away (see `FLIP`).

The replay gate
---------------
Before anything is fitted the pose parameters are replayed and must
reproduce the input mesh to 1 mm. Anything that edits the vertices and
leaves the parameters describing the old geometry (the retired
`head_angle_fix` did exactly that) fails this gate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

#: OpenCV camera frame <-> what `mhr_forward` returns. See the docstring.
FLIP = np.array([1.0, -1.0, -1.0])

#: MediaPipe's face-oval contour: the jaw line and the hairline. Scored,
#: never fitted — see the module docstring.
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109)

#: MHR70 joints that define the head for the crop and the sanity checks.
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_NECK = 69
_HEAD_POINTS = (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR)

#: The handles, as shipped with facebook/sam-3d-body-dinov3. Params on the
#: fit step override them; `_check_handles` verifies whichever are used.
DEFAULT_POSE_INDICES = (18, 19, 20, 21, 22, 23)
DEFAULT_HEAD_SCALE_INDEX = 4
DEFAULT_HEAD_SHAPE_FROM = 20


def _project(points: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    z = np.clip(p[:, 2], 1e-6, None)
    return np.stack([focal * p[:, 0] / z + cx, focal * p[:, 1] / z + cy], 1)


def _head_vertices(vertices_cam: np.ndarray, joints_cam: np.ndarray) -> np.ndarray:
    """Vertices above the neck joint (the camera frame's y points down)."""
    return vertices_cam[:, 1] < joints_cam[_NECK, 1]


def head_crop_box(vertices_cam: np.ndarray, joints_cam: np.ndarray, focal: float,
                  width: int, height: int, margin: float = 0.25) -> Tuple[int, int, int, int]:
    """A frame-clamped box around the projected head, padded by `margin`."""
    head = vertices_cam[_head_vertices(vertices_cam, joints_cam)]
    if len(head) < 10:
        head = joints_cam[list(_HEAD_POINTS)]
    px = _project(head, focal, width / 2.0, height / 2.0)
    x0, y0 = px.min(0)
    x1, y1 = px.max(0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, x1 = max(0, int(np.floor(x0 - mx))), min(width, int(np.ceil(x1 + mx)))
    y0, y1 = max(0, int(np.floor(y0 - my))), min(height, int(np.ceil(y1 + my)))
    return x0, y0, x1, y1


def snap_landmarks_to_vertices(landmarks_px: np.ndarray, projected: np.ndarray,
                               visible: np.ndarray, snap_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest visible projected vertex per landmark, within `snap_px`.

    Returns (vertex_of_landmark, distance_px); unmapped landmarks carry -1
    and inf. Pure numpy/scipy so it can be tested without a renderer.
    """
    from scipy.spatial import cKDTree

    vis_idx = np.flatnonzero(visible)
    if vis_idx.size == 0:
        return np.full(len(landmarks_px), -1, np.int64), np.full(len(landmarks_px), np.inf)
    tree = cKDTree(projected[vis_idx])
    dist, idx = tree.query(landmarks_px, distance_upper_bound=snap_px)
    mapped = np.isfinite(dist)
    out = np.full(len(landmarks_px), -1, np.int64)
    out[mapped] = vis_idx[idx[mapped]]
    return out, dist


@register_step("map_face_to_mesh")
class MapFaceToMeshStep(Step):
    """MediaPipe landmark -> MHR vertex correspondence, from this mesh's own head.

    inputs:  {"mesh_output": dict — vertices, faces, keypoints_3d, cam_t,
                                     focal_length (sam3d_body's outputs),
              "image": HxWx3 — only its size is read: the camera the mesh
                                was fitted in is centred on this frame}
    outputs: {"face_correspondence": {"vertex_of_landmark": (N,) int, -1 where
                                       no visible vertex was within reach,
                                      "mapped": (N,) bool,
                                      "snap_px_mean": float,
                                      "render_box": [x0, y0, x1, y1],
                                      "upsample": int}}

    The render is a shaded, untextured head; MediaPipe finds a face on it
    reliably at the 3x head crop this renders (446/468 landmarks mapped at
    0.8 px mean snap distance on cyber2_6f). If it does not, the step fails
    loudly — a silent fallback would hand the fit a wrong correspondence.
    """

    PARAMS = (
        Param("upsample", int, 3,
              "Render the head crop at this many times the frame's pixel "
              "density, so MediaPipe sees a face-filling image and the snap "
              "distance is a fraction of a frame pixel", minimum=1, maximum=6),
        Param("snap_px", float, 2.0,
              "A landmark further than this (in FRAME pixels) from every visible "
              "vertex is left unmapped", minimum=0.1),
        Param("min_mapped", int, 200,
              "Refuse a correspondence with fewer landmarks mapped than this; "
              "the fit is dense or it is nothing", minimum=1),
        Param("min_detection_confidence", float, 0.3,
              "MediaPipe's detection floor on the render", minimum=0.0, maximum=1.0,
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        from . import render as _render_module  # noqa: F401 — sets PYOPENGL_PLATFORM
        from .face_landmarks import (
            DETECTOR_MODEL_NAME, DETECTOR_MODEL_URL, LANDMARKER_MODEL_NAME,
            LANDMARKER_MODEL_URL, _detect, _ensure_model, _model_path,
        )

        mesh = inputs["mesh_output"]
        image = np.asarray(inputs["image"])
        height, width = image.shape[:2]
        focal = float(mesh["focal_length"])
        cam_t = np.asarray(mesh["cam_t"], np.float64).reshape(3)
        vertices = np.asarray(mesh["vertices"], np.float64) + cam_t
        joints = np.asarray(mesh["keypoints_3d"], np.float64) + cam_t
        faces = np.asarray(mesh["faces"])

        x0, y0, x1, y1 = head_crop_box(vertices, joints, focal, width, height)
        k = int(params["upsample"])
        # Keep the render a sane size whatever the frame: MediaPipe wants a
        # face-filling image, not a large one.
        while k > 1 and max(x1 - x0, y1 - y0) * k > 1600:
            k -= 1
        rw, rh = (x1 - x0) * k, (y1 - y0) * k
        cx, cy = k * (width / 2.0 - x0), k * (height / 2.0 - y0)

        rgb, depth = self._render(vertices, faces, k * focal, cx, cy, rw, rh)

        # --- MediaPipe on the render -------------------------------------
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        landmarker_path = str(_ensure_model(LANDMARKER_MODEL_URL, _model_path(LANDMARKER_MODEL_NAME)))
        detector_path = str(_ensure_model(DETECTOR_MODEL_URL, _model_path(DETECTOR_MODEL_NAME)))
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=landmarker_path),
            min_face_detection_confidence=params["min_detection_confidence"],
            min_face_presence_confidence=params["min_detection_confidence"],
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        try:
            landmarks = _detect(
                rgb=np.ascontiguousarray(rgb), width=rw, height=rh, landmarker=landmarker,
                detector_path=detector_path, min_confidence=params["min_detection_confidence"],
                crop_padding=0.5, mp=mp, vision=vision, python=python,
            )
        finally:
            landmarker.close()
        if landmarks is None or len(landmarks) == 0:
            raise ValueError(
                "map_face_to_mesh: MediaPipe found no face on the shaded render of "
                "the mesh head. Either the head is not where keypoints_3d says "
                "(mesh_output and keypoints from different fits?) or the render "
                "is blank — check EGL (`python -m pipeline.cli doctor`)."
            )
        lm_px = np.asarray(landmarks, np.float64)[:, :2] * np.array([rw, rh])

        # --- visibility and the snap --------------------------------------
        projected = _project(vertices, k * focal, cx, cy)
        u = np.round(projected[:, 0]).astype(int)
        v = np.round(projected[:, 1]).astype(int)
        inside = (u >= 0) & (u < rw) & (v >= 0) & (v < rh)
        zbuf = np.full(len(vertices), np.inf)
        zbuf[inside] = depth[v[inside], u[inside]]
        visible = inside & (zbuf > 0) & (vertices[:, 2] <= zbuf + 0.004)

        vertex_of_landmark, dist = snap_landmarks_to_vertices(
            lm_px, projected, visible, params["snap_px"] * k)
        mapped = vertex_of_landmark >= 0
        if mapped.sum() < params["min_mapped"]:
            raise ValueError(
                f"map_face_to_mesh: only {int(mapped.sum())} of {len(mapped)} landmarks "
                f"landed within {params['snap_px']} px of a visible vertex "
                f"(min_mapped={params['min_mapped']}). The render and the landmarker "
                f"disagree about where the face is."
            )
        snap_mean = float(dist[mapped].mean() / k)
        logger.info(
            "map_face_to_mesh: %d/%d landmarks mapped to visible vertices (%d visible "
            "of %d), mean snap %.2f px, head box %s at %dx",
            int(mapped.sum()), len(mapped), int(visible.sum()), len(vertices), snap_mean,
            (x0, y0, x1, y1), k,
        )
        return {"face_correspondence": {
            "vertex_of_landmark": vertex_of_landmark,
            "mapped": mapped,
            "snap_px_mean": snap_mean,
            "render_box": [int(x0), int(y0), int(x1), int(y1)],
            "upsample": k,
        }}

    @staticmethod
    def _render(vertices_cam: np.ndarray, faces: np.ndarray, focal: float, cx: float,
                cy: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Shaded RGB + depth of the mesh through an (OpenCV) pinhole."""
        import pyrender
        import trimesh

        mesh = trimesh.Trimesh(vertices_cam * FLIP, faces, process=False)  # -> OpenGL
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.87, 0.68, 0.58, 1.0], metallicFactor=0.0, roughnessFactor=0.7)
        scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[0.5, 0.5, 0.5, 1.0])
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
        scene.add(pyrender.IntrinsicsCamera(focal, focal, cx, cy), pose=np.eye(4))
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=np.eye(4))
        fill = np.eye(4)
        fill[:3, 3] = [0.3, 0.5, 0.0]
        scene.add(pyrender.PointLight(intensity=2.0), pose=fill)
        renderer = pyrender.OffscreenRenderer(width, height)
        try:
            color, depth = renderer.render(scene)
        finally:
            renderer.delete()
        return np.asarray(color[:, :, :3]), np.asarray(depth)


@register_step("fit_head_to_face")
class FitHeadToFaceStep(Step):
    """Re-fit the MHR head parameters to the photograph's landmarks.

    inputs:  {"mesh_output": dict — sam3d_body's outputs INCLUDING pose_params,
              "face_landmarks": detect_face_landmarks' output on the photo,
              "face_correspondence": map_face_to_mesh's output,
              "image": HxWx3 — the photo; only its size is read}
    outputs: {"vertices", "keypoints_3d", "joints", "global_rots" — the
              refitted geometry, in SAM-3D-Body's raw output space,
              "pose_params" — updated, with a new "scale_offsets" entry
              (68 floats, zero except the head joint) that a replay must
              pass to `mhr_forward(scale_offsets=...)`,
              "head_fit_stats"}
    """

    PARAMS = (
        Param("shape_regularisation", float, 1.0,
              "L2 pull on the head shape components, per sigma squared, against "
              "the landmark loss in pixels. 1.0 leaves the shape 1.8 sigma rms "
              "from SAM-3D-Body's fit on a subject the model finds unusual; 0.3 "
              "buys 0.3 px for another half sigma", minimum=0.0),
        Param("fit_shape", bool, True,
              "Fit the head-only shape components. Off, the head is only turned "
              "and scaled — cheap, and enough when the body model's face already "
              "has the subject's proportions"),
        Param("fit_scale", bool, True, "Fit the head joint's scale"),
        Param("iterations", int, 600, "Adam steps of the joint pose+scale+shape "
              "stage; the pose+scale warm-up runs half as many", minimum=1),
        Param("learning_rate", float, 0.01,
              "Adam step; rotations are in radians, so 0.01 is ~0.6 deg", advanced=True),
        Param("huber_px", float, 4.0,
              "Landmark residuals beyond this many pixels count linearly, not "
              "quadratically — MediaPipe's occasional outlier does not steer the fit",
              minimum=0.1, advanced=True),
        Param("pose_indices", str, ",".join(str(i) for i in DEFAULT_POSE_INDICES),
              "body_pose_params indices of the neck/head joint rotations", advanced=True),
        Param("head_scale_index", int, DEFAULT_HEAD_SCALE_INDEX,
              "Index into mhr_forward's scale_offsets that scales the head joint",
              minimum=0, advanced=True),
        Param("head_shape_from", int, DEFAULT_HEAD_SHAPE_FROM,
              "shape_params[this:] are the head-only shape components", minimum=0,
              advanced=True),
        Param("checkpoint_repo", str, "facebook/sam-3d-body-dinov3",
              "HF repo the checkpoint is pulled from", advanced=True),
        Param("checkpoint_dir", str, None,
              "A local snapshot directory to use instead of downloading", advanced=True),
        Param("mhr_path", str, None,
              "The mhr_model.pt to load; empty means assets/mhr_model.pt inside the "
              "checkpoint directory", advanced=True),
        Param("device", str, "cuda", "Torch device", advanced=True),
    )

    def __init__(self) -> None:
        self._head = None

    def load(self, params: Dict[str, Any]) -> None:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from sam_3d_body import load_sam_3d_body

        checkpoint_dir = params["checkpoint_dir"] or snapshot_download(params["checkpoint_repo"])
        mhr_path = params["mhr_path"] or str(Path(checkpoint_dir) / "assets" / "mhr_model.pt")
        model, _ = load_sam_3d_body(str(Path(checkpoint_dir) / "model.ckpt"),
                                    device=params["device"], mhr_path=mhr_path)
        # Only the body model is needed; the image backbone never runs here.
        self._head = model.head_pose

    def unload(self) -> None:
        self._head = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _parse_indices(spec: str) -> List[int]:
        try:
            out = sorted({int(v) for v in str(spec).split(",") if v.strip()})
        except ValueError as exc:
            raise ValueError(f"fit_head_to_face: pose_indices must be comma-separated "
                             f"integers, got {spec!r}") from exc
        if not out:
            raise ValueError("fit_head_to_face: pose_indices is empty")
        return out

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        mesh = inputs["mesh_output"]
        if not isinstance(mesh, dict) or mesh.get("pose_params") is None:
            raise KeyError(
                "fit_head_to_face needs 'pose_params' inside mesh_output — wire "
                "sam3d_body's pose_params output into the scene. It re-runs the MHR "
                "body model's forward with those as the free variables."
            )
        import torch

        pose_params = mesh["pose_params"]
        if hasattr(pose_params, "item") and not isinstance(pose_params, dict):
            pose_params = pose_params.item()      # an npz round-trip's object array
        if self._head is None:
            self.load(params)

        image = np.asarray(inputs["image"])
        height, width = image.shape[:2]
        focal = float(mesh["focal_length"])
        cam_t_np = np.asarray(mesh["cam_t"], np.float64).reshape(3)
        device = params["device"]
        flip = torch.tensor(FLIP, dtype=torch.float32, device=device)

        # --- targets --------------------------------------------------------
        lm = inputs["face_landmarks"]
        landmarks = np.asarray(lm["landmarks"], np.float64)
        lw, lh = (lm.get("image_size") or (width, height))
        if (int(lw), int(lh)) != (width, height):
            raise ValueError(
                f"fit_head_to_face: face_landmarks were detected on a {lw}x{lh} image "
                f"but the photo is {width}x{height}; they must be the same frame."
            )
        photo_px = landmarks[:, :2] * np.array([width, height])
        corr = inputs["face_correspondence"]
        vert_of_lm = np.asarray(corr["vertex_of_landmark"], np.int64)
        mapped = np.asarray(corr["mapped"], bool)
        n = min(len(photo_px), len(vert_of_lm))
        photo_px, vert_of_lm, mapped = photo_px[:n], vert_of_lm[:n], mapped[:n]
        is_oval = np.zeros(n, bool)
        is_oval[[i for i in FACE_OVAL if i < n]] = True
        feature = mapped & ~is_oval
        oval = mapped & is_oval
        if feature.sum() < 50:
            raise ValueError(f"fit_head_to_face: only {int(feature.sum())} feature landmarks "
                             f"are mapped; nothing to fit")

        # --- the body model, replayed --------------------------------------
        def tensor(value):
            out = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
            return out[None] if out.ndim == 1 else out

        g0 = tensor(pose_params["global_rot"])
        b0 = tensor(pose_params["body_pose_params"])
        h0 = tensor(pose_params["hand_pose_params"])
        sc0 = tensor(pose_params["scale_params"])
        sh0 = tensor(pose_params["shape_params"])
        ex0 = tensor(pose_params["expr_params"])
        cam_t = tensor(cam_t_np)[0]
        n_scales = int(self._head.scale_mean.shape[0])
        so0 = torch.zeros(1, n_scales, device=device)
        if pose_params.get("scale_offsets") is not None:
            so0 = tensor(pose_params["scale_offsets"])
        n_keypoints = len(np.asarray(mesh["keypoints_3d"]))

        pose_idx = self._parse_indices(params["pose_indices"])
        if max(pose_idx) >= b0.shape[1]:
            raise ValueError(f"fit_head_to_face: pose index {max(pose_idx)} is outside "
                             f"body_pose_params ({b0.shape[1]})")
        scale_idx = int(params["head_scale_index"])
        shape_from = int(params["head_shape_from"])
        if scale_idx >= n_scales or shape_from >= sh0.shape[1]:
            raise ValueError("fit_head_to_face: head_scale_index/head_shape_from are "
                             "outside this body model's parameter ranges")
        pose_idx_t = torch.tensor(pose_idx, device=device)
        head = self._head

        def forward(dpose, dscale, dshape):
            body = b0.clone()
            body[0, pose_idx_t] = body[0, pose_idx_t] + dpose
            shape = sh0.clone()
            shape[0, shape_from:] = shape[0, shape_from:] + dshape
            so = so0.clone()
            so[0, scale_idx] = so[0, scale_idx] + dscale
            verts, keypoints, joints, rots = head.mhr_forward(
                global_trans=torch.zeros_like(g0), global_rot=g0, body_pose_params=body,
                hand_pose_params=h0, scale_params=sc0, shape_params=shape, expr_params=ex0,
                return_keypoints=True, return_joint_coords=True, return_joint_rotations=True,
                scale_offsets=so,
            )
            return verts[0] * flip, keypoints[0, :n_keypoints] * flip, joints[0] * flip, rots[0]

        def project(points):
            p = points + cam_t
            z = p[:, 2].clamp(min=1e-3)
            return torch.stack([focal * p[:, 0] / z + width / 2.0,
                                focal * p[:, 1] / z + height / 2.0], 1)

        zero_pose = torch.zeros(len(pose_idx), device=device)
        zero_scale = torch.zeros((), device=device)
        zero_shape = torch.zeros(sh0.shape[1] - shape_from, device=device)
        with torch.no_grad():
            verts0, keypoints0, _, _ = forward(zero_pose, zero_scale, zero_shape)
        drift = float(np.abs(verts0.cpu().numpy() - np.asarray(mesh["vertices"])).max())
        if drift > 1e-3:
            raise RuntimeError(
                f"fit_head_to_face: replaying the pose parameters does not reproduce "
                f"the input mesh ({drift * 1000:.2f} mm off). Either something "
                f"deformed the vertices without updating the parameters, or "
                f"mesh_output and pose_params are from different fits."
            )
        self._check_handles(forward, verts0, keypoints0, pose_idx, zero_pose, zero_scale,
                            zero_shape, params["fit_scale"])

        target_f = torch.tensor(photo_px[feature], dtype=torch.float32, device=device)
        target_o = torch.tensor(photo_px[oval], dtype=torch.float32, device=device)
        vf = torch.tensor(vert_of_lm[feature], device=device)
        vo = torch.tensor(vert_of_lm[oval], device=device)

        def residuals(verts):
            return target_f - project(verts[vf]), (target_o - project(verts[vo]) if oval.any() else None)

        def rms(r):
            return float(r.pow(2).sum(1).mean().sqrt()) if r is not None and len(r) else float("nan")

        with torch.no_grad():
            rf0, ro0 = residuals(verts0)
        before = {"features_rms_px": rms(rf0), "oval_rms_px": rms(ro0)}

        # --- the fit: pose+scale warm-up, then everything ------------------
        dpose = zero_pose.clone().requires_grad_(True)
        dscale = zero_scale.clone().requires_grad_(True)
        dshape = zero_shape.clone().requires_grad_(True)
        lam = params["shape_regularisation"]
        huber = params["huber_px"]

        def solve(iterations, learning_rate, with_shape):
            groups = [{"params": [dpose], "lr": learning_rate}]
            if params["fit_scale"]:
                groups.append({"params": [dscale], "lr": learning_rate})
            if with_shape:
                groups.append({"params": [dshape], "lr": learning_rate * 2})
            optimiser = torch.optim.Adam(groups)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=iterations)
            best = (float("inf"), None)
            for _ in range(iterations):
                optimiser.zero_grad()
                verts, _, _, _ = forward(dpose, dscale, dshape if with_shape else dshape * 0)
                r = target_f - project(verts[vf])
                loss = 2 * torch.nn.functional.huber_loss(
                    r, torch.zeros_like(r), delta=huber, reduction="mean")
                if with_shape:
                    loss = loss + lam * (dshape ** 2).mean()
                if loss.item() < best[0]:
                    best = (loss.item(), (dpose.detach().clone(), dscale.detach().clone(),
                                          dshape.detach().clone()))
                loss.backward()
                optimiser.step()
                schedule.step()
            with torch.no_grad():
                dpose.copy_(best[1][0])
                dscale.copy_(best[1][1])
                dshape.copy_(best[1][2] if with_shape else 0 * best[1][2])

        iterations = int(params["iterations"])
        solve(max(iterations // 2, 1), params["learning_rate"] * 2, with_shape=False)
        if params["fit_shape"]:
            solve(iterations, params["learning_rate"], with_shape=True)

        with torch.no_grad():
            verts, keypoints, joints, rots = forward(dpose, dscale, dshape)
            rf, ro = residuals(verts)
        after = {"features_rms_px": rms(rf), "oval_rms_px": rms(ro)}

        # --- outputs ----------------------------------------------------------
        new_pose = {k: np.array(v, copy=True) for k, v in pose_params.items()}
        body = np.asarray(new_pose["body_pose_params"], np.float32).copy()
        body[pose_idx] += dpose.detach().cpu().numpy()
        new_pose["body_pose_params"] = body
        shape = np.asarray(new_pose["shape_params"], np.float32).copy()
        shape[shape_from:] += dshape.detach().cpu().numpy()
        new_pose["shape_params"] = shape
        offsets = so0[0].detach().cpu().numpy().astype(np.float32).copy()
        offsets[scale_idx] += float(dscale)
        new_pose["scale_offsets"] = offsets

        k0 = keypoints0.cpu().numpy()
        k1 = keypoints.cpu().numpy()

        def nose_pitch(k):
            d = k[_NOSE] - 0.5 * (k[_L_EAR] + k[_R_EAR])
            return float(np.degrees(np.arctan2(d[1], -d[2])))

        stats = {
            "before": before, "after": after,
            "landmarks": {"features": int(feature.sum()), "oval": int(oval.sum())},
            "pose_delta_deg": [float(v) for v in np.degrees(dpose.detach().cpu().numpy())],
            "head_scale_offset": float(dscale),
            "shape_delta_sigma": {
                "rms": float(dshape.pow(2).mean().sqrt()), "max": float(dshape.abs().max()),
            } if params["fit_shape"] else None,
            "nose_direction_pitch_deg": {"before": nose_pitch(k0), "after": nose_pitch(k1)},
            "vertex_shift_mm_median": float(np.median(np.linalg.norm(
                verts.cpu().numpy() - verts0.cpu().numpy(), axis=1)) * 1000),
        }
        logger.info(
            "fit_head_to_face: landmarks %.2f -> %.2f px rms over %d features (oval %.2f -> "
            "%.2f); head pitched %+.1f deg, scale offset %+.3f, shape %s sigma rms; %d/%d "
            "landmarks mapped",
            before["features_rms_px"], after["features_rms_px"], int(feature.sum()),
            before["oval_rms_px"], after["oval_rms_px"],
            stats["nose_direction_pitch_deg"]["after"] - stats["nose_direction_pitch_deg"]["before"],
            float(dscale),
            f"{stats['shape_delta_sigma']['rms']:.2f}" if stats["shape_delta_sigma"] else "-",
            int(mapped.sum()), len(mapped),
        )
        return {
            "vertices": verts.cpu().numpy().astype(np.asarray(mesh["vertices"]).dtype, copy=False),
            "keypoints_3d": k1.astype(np.asarray(mesh["keypoints_3d"]).dtype, copy=False),
            "joints": joints.cpu().numpy(),
            "global_rots": rots.cpu().numpy(),
            "pose_params": new_pose,
            "head_fit_stats": stats,
        }

    @staticmethod
    def _check_handles(forward, verts0, keypoints0, pose_idx, zero_pose, zero_scale,
                       zero_shape, check_scale: bool) -> None:
        """Each fitted parameter must move the head and (almost) nothing else.

        The indices are facts about the MHR model that ships with the
        checkpoint, found by perturbation; this repeats the measurement so a
        changed model fails here instead of quietly re-posing a shoulder.
        """
        import torch

        joints_cam = keypoints0.cpu().numpy()
        head = torch.as_tensor(_head_vertices(verts0.cpu().numpy(), joints_cam), device=verts0.device)
        with torch.no_grad():
            for slot, i in enumerate(pose_idx):
                dpose = zero_pose.clone()
                dpose[slot] = 0.1
                verts, _, _, _ = forward(dpose, zero_scale, zero_shape)
                moved = (verts - verts0).norm(dim=1) > 1e-3
                if moved.sum() == 0 or (moved & head).sum().item() / moved.sum().item() < 0.9:
                    raise RuntimeError(
                        f"fit_head_to_face: body_pose_params[{i}] does not move only the "
                        f"head ({(moved & head).sum().item()} of {moved.sum().item()} moved "
                        f"vertices are above the neck). The body model's parameter layout "
                        f"differs from the one these defaults were measured on; set "
                        f"pose_indices to this model's neck/head joints.")
            if check_scale:
                verts, _, _, _ = forward(zero_pose, zero_scale + 0.1, zero_shape)
                moved = (verts - verts0).norm(dim=1) > 1e-3
                if moved.sum() == 0 or (moved & head).sum().item() / moved.sum().item() < 0.9:
                    raise RuntimeError(
                        "fit_head_to_face: head_scale_index does not scale only the head "
                        "on this body model; set it to the head joint's scale slot.")
