"""The face per view, and the anchor's eyes injected into the frames.

Three steps over the final training's frames, after the refit body and the
final camera refinement (b2ctrain docs/STATUS.md, "The face per view, and
the anchor's eyes injected", 2026-09-11):

  * **`detect_face_views`** (main env: MediaPipe) landmarks the face in
    every frame that shows one. The pipeline's own face detector (BlazeFace
    short-range on the whole 1080x1920 frame) misses the ~80 px face in 63
    of 81 helical frames and finds faces on the trousers in four; this step
    needs no detector. It projects the refit body's head into each frame,
    cuts a crop around it rolled upright (the projected head-up axis), and
    runs the landmarker on that crop — 35 of 81 landmarked, every view that
    faces the camera to about 85 degrees.
  * **`fit_head_per_view`** (sam3dbody env: the MHR body model) fits the
    head to each landmarked frame: the six neck/head rotations and the 72
    expression blendshapes SAM-3D-Body leaves at zero, all views in one
    batched Adam run through the views' own cameras. Landmark rms 6.4 ->
    4.6 (pose) -> 2.1 px (pose + expression). The rotations are held by an
    L2 prior: unregularised, the six DOF counter-rotate into a lateral head
    shift (61 mm on one frame) at no residual gain. The anchor photograph
    joins the batch through a PnP camera, expression free, so its lids are
    fitted too — the eye model below is textured through them.
  * **`paste_eyes`** (main env: pyrender) replaces the frames' eyes. The
    generated frames' eyes are the diffusion's invention per frame (three
    consecutive frames of one subject: half closed, open looking sideways,
    open with makeup), so nothing multi-view recovers them. The MHR has eye
    joints but no eyeballs; this builds a 15.5 mm sphere at each eye joint,
    textures it from the anchor photograph through the anchor's fitted
    lids (the ~30% of the iris under the anchor's upper lid is filled by a
    radial colour profile around the gaze axis, one sclera for both eyes),
    and renders it into every fitted frame through THAT frame's fitted
    lids: clipped in 3D to the lid contour's cylinder (a 2D polygon leaks
    at grazing angles), depth-tested at 1 mm against the head mesh with
    the eye-surface faces removed and back-face culling off, skipped when
    less than half its lid polygon is visible (the far eye peeking past
    the model's narrower nose) or when nothing darker than the skin sits
    inside the lid polygon (a hand in front of the face).

`build_face_rig` (steps/body_rig.py) turns the fit into the v3 rig's
per-view vertex displacements, restricted to the face core, which is what
lets the canonical face converge to the anchor's opening instead of the
average of every frame's lids.

Measured on four subjects: eye error against the eye model 51-82 -> 28-56,
face sharpness within +-2.5% of the baseline everywhere, hair untouched by
the face-only deltas (whole-head deltas cost it 2-5% on three of four).

Coordinate frames: everything here is in the dataset's world frame.
body2colmap's Camera holds an OpenGL camera-to-world rotation and a
position; `opencv_camera` turns it into the world-to-camera (R, t) and K
the projections below use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step
from .head_fit import FACE_OVAL, FLIP, DEFAULT_POSE_INDICES

logger = logging.getLogger(__name__)

#: MHR joints (facts about the model that ships with facebook/sam-3d-body;
#: `paste_eyes` checks the eye joints against the fitted lid rings).
MHR_HEAD_JOINT = 113
MHR_HEAD_TOP_JOINT = 126
MHR_EYE_JOINTS = (122, 124)   # eye centres; joint + 1 is the gaze child 1.7 cm ahead

#: MHR70 keypoints of the face plane (nose, left eye, right eye).
_KP_NOSE, _KP_L_EYE, _KP_R_EYE = 0, 1, 2

#: MediaPipe's lid contours (image-left and image-right eye) and iris centres.
LID_RING_IMAGE_LEFT = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LID_RING_IMAGE_RIGHT = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
IRIS_IMAGE_LEFT, IRIS_IMAGE_RIGHT = 468, 473
EYE_SIDES = (("image_left", LID_RING_IMAGE_LEFT, IRIS_IMAGE_LEFT),
             ("image_right", LID_RING_IMAGE_RIGHT, IRIS_IMAGE_RIGHT))


# -- geometry helpers (pure numpy, tested) -------------------------------------

def opencv_camera(camera: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, t, K): world -> OpenCV camera of a body2colmap Camera."""
    rot = np.asarray(camera.rotation, np.float64)
    pos = np.asarray(camera.position, np.float64).reshape(3)
    R = np.diag([1.0, -1.0, -1.0]) @ rot.T
    t = -R @ pos
    K = np.array([[float(camera.fx), 0.0, float(camera.cx)], [0.0, float(camera.fy), float(camera.cy)], [0.0, 0.0, 1.0]])
    return R, t, K


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates and camera depth of world points."""
    c = np.asarray(points, np.float64) @ R.T + t
    z = c[:, 2]
    uv = c[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def head_crop_affine(face_uv: np.ndarray, up_uv: np.ndarray, crop: int, padding: float = 1.8
                     ) -> Tuple[np.ndarray, float, float]:
    """The affine that maps the frame onto a `crop`x`crop` image of the head, rolled upright.

    `face_uv` are the projected face vertices, `up_uv` the projected head
    joint and head top (the head's up axis in the image). Returns the 2x3
    matrix (cv2.warpAffine), the crop's side in frame pixels and the roll
    in degrees that was removed.
    """
    x0, y0 = face_uv.min(0)
    x1, y1 = face_uv.max(0)
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    side = float(padding * max(x1 - x0, y1 - y0, 1e-6))
    dx, dy = up_uv[1] - up_uv[0]
    roll = float(np.degrees(np.arctan2(dx, -dy)))   # cv2: positive = counter-clockwise on screen
    M = cv2.getRotationMatrix2D(centre, roll, crop / side)
    M[0, 2] += crop / 2.0 - centre[0]
    M[1, 2] += crop / 2.0 - centre[1]
    return M, side, roll


def point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd test of 2-D points against a closed polygon (vertices in order)."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, np.float64).reshape(-1, 2)
    inside = np.zeros(len(points), bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        crosses = (y0 > points[:, 1]) != (y1 > points[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = x0 + (points[:, 1] - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (points[:, 0] < xs)
    return inside


def plane_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centroid and the two in-plane axes (3x2 -> rows) of a set of 3-D points."""
    P = np.asarray(points, np.float64)
    centre = P.mean(0)
    basis = np.linalg.svd(P - centre)[2][:2]
    return centre, basis


def inside_contour_cylinder(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """True where a 3-D point projects inside the closed 3-D contour, in the contour's own plane."""
    centre, basis = plane_basis(contour)
    return point_in_polygon((np.asarray(points, np.float64) - centre) @ basis.T, (contour - centre) @ basis.T)


def kabsch(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rigid (R, t) with B ~= A @ R.T + t."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def rodrigues_torch(rotvec):
    """Axis-angle vectors [B,3] -> rotation matrices [B,3,3] (torch, differentiable)."""
    import torch

    theta = rotvec.norm(dim=1, keepdim=True).clamp(min=1e-12)
    k = rotvec / theta
    Kx = torch.zeros(rotvec.shape[0], 3, 3, device=rotvec.device, dtype=rotvec.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)[None]
    return eye + s * Kx + (1 - c) * (Kx @ Kx)


def to_world(points: np.ndarray, world_from_raw: Dict[str, Any]) -> np.ndarray:
    scale = float(world_from_raw["scale"])
    rot = np.asarray(world_from_raw["rotation"], np.float64).reshape(3, 3)
    trans = np.asarray(world_from_raw["translation"], np.float64).reshape(3)
    return scale * np.asarray(points, np.float64) @ rot.T + trans


def feature_landmarks(face_correspondence: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(landmark indices, vertex indices) of the fitted features: mapped, not the face oval, not the irises."""
    vol = np.asarray(face_correspondence["vertex_of_landmark"], np.int64)
    feat = np.asarray(face_correspondence["mapped"], bool).copy()
    feat[[i for i in FACE_OVAL if i < len(feat)]] = False
    feat[468:] = False
    idx = np.flatnonzero(feat)
    return idx, vol[idx]


def _check_world_from_raw(inputs: Dict[str, Any], step: str) -> Dict[str, Any]:
    wfr = inputs.get("world_from_raw")
    if not isinstance(wfr, dict) or any(k not in wfr for k in ("scale", "rotation", "translation")):
        raise ValueError(f"{step} needs 'world_from_raw' — refit_body_to_splat's frame of the body")
    return wfr


# -- detect_face_views ----------------------------------------------------------

@register_step("detect_face_views")
class DetectFaceViewsStep(Step):
    """Face landmarks in every training frame that shows the face, from the projected head.

    inputs:  {"images": [HxWx3/4 BGR(A)], "image_names": [str], "cameras": [Camera],
              "mesh_world": (vertices, faces) — the refit body in the world frame,
              "joints": (J,3) raw, "keypoints_3d": (70,3) raw, "world_from_raw": {...},
              "face_correspondence": map_face_to_mesh's output (its vertices frame the crop)}
    outputs: {"face_views": {"names": [str] landmarked views in training order,
                             "landmarks_px": {name: (478,3) — frame pixels, z in frame-pixel units},
                             "facing_deg": {name: float} for EVERY view,
                             "cameras": {name: {"R","t","K"}} OpenCV world->camera, every view,
                             "image_size": (w, h), "stats": {...}}}
    """

    PARAMS = (
        Param("crop_size", int, 512, "Side of the upright head crop the landmarker sees", minimum=128),
        Param("crop_padding", float, 1.8, "Crop side as a multiple of the projected face's extent", minimum=1.0),
        Param("min_confidence", float, 0.2, "MediaPipe's presence/detection floor on the crop", minimum=0.0, maximum=1.0, advanced=True),
        Param("debug_dir", str, "", "Write the crops with their landmarks, and meta.json, here"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from .face_landmarks import LANDMARKER_MODEL_NAME, LANDMARKER_MODEL_URL, _ensure_model, _model_path

        images, names, cameras = list(inputs["images"]), list(inputs["image_names"]), list(inputs["cameras"])
        if not (len(images) == len(names) == len(cameras)):
            raise ValueError(f"detect_face_views: images ({len(images)}), image_names ({len(names)}) and cameras "
                             f"({len(cameras)}) disagree in length")
        mesh = inputs.get("mesh_world")
        if not isinstance(mesh, (tuple, list)) or len(mesh) != 2:
            raise ValueError("detect_face_views needs 'mesh_world' (vertices, faces) — refit_body_to_splat's")
        wfr = _check_world_from_raw(inputs, "detect_face_views")
        vertices = np.asarray(mesh[0], np.float64)
        joints = to_world(np.asarray(inputs["joints"], np.float64), wfr)
        keypoints = to_world(np.asarray(inputs["keypoints_3d"], np.float64), wfr)
        corr = inputs.get("face_correspondence")
        if not isinstance(corr, dict) or "vertex_of_landmark" not in corr:
            raise ValueError("detect_face_views needs 'face_correspondence' from map_face_to_mesh")
        vol = np.asarray(corr["vertex_of_landmark"], np.int64)
        face_idx = np.unique(vol[vol >= 0])
        if len(face_idx) < 50 or face_idx.max() >= len(vertices):
            raise ValueError("detect_face_views: face_correspondence does not index this mesh's face")
        if max(MHR_HEAD_JOINT, MHR_HEAD_TOP_JOINT) >= len(joints):
            raise ValueError(f"detect_face_views: the body has {len(joints)} joints, not the MHR's 127")
        head, head_top = joints[MHR_HEAD_JOINT], joints[MHR_HEAD_TOP_JOINT]
        if not 0.03 < np.linalg.norm(head_top - head) < 0.3:
            raise ValueError("detect_face_views: joints 113/126 are not a head and its top on this body")
        nose, l_eye, r_eye = keypoints[_KP_NOSE], keypoints[_KP_L_EYE], keypoints[_KP_R_EYE]
        normal = np.cross(l_eye - r_eye, nose - (l_eye + r_eye) / 2)
        normal /= max(np.linalg.norm(normal), 1e-12)
        if np.dot(normal, nose - head) < 0:
            normal = -normal

        crop = int(params["crop_size"])
        landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_model(LANDMARKER_MODEL_URL, _model_path(LANDMARKER_MODEL_NAME)))),
            min_face_detection_confidence=params["min_confidence"], min_face_presence_confidence=params["min_confidence"], num_faces=1))
        debug = Path(params["debug_dir"]) if params["debug_dir"] else None
        if debug:
            debug.mkdir(parents=True, exist_ok=True)
        landmarks_px: Dict[str, np.ndarray] = {}
        facing: Dict[str, float] = {}
        cams: Dict[str, Dict[str, np.ndarray]] = {}
        meta: Dict[str, Any] = {}
        height, width = images[0].shape[:2]
        try:
            for name, img, cam in zip(names, images, cameras):
                R, t, K = opencv_camera(cam)
                cams[name] = {"R": R, "t": t, "K": K}
                cam_dir = np.asarray(cam.position, np.float64) - head
                cam_dir /= max(np.linalg.norm(cam_dir), 1e-12)
                facing[name] = float(np.degrees(np.arccos(np.clip(np.dot(normal, cam_dir), -1.0, 1.0))))
                uv, z = project(vertices[face_idx], R, t, K)
                up_uv, _ = project(np.stack([head, head_top]), R, t, K)
                if (z <= 0.05).any():
                    meta[name] = {"facing_deg": facing[name], "ok": False, "reason": "head behind the camera"}
                    continue
                M, side, roll = head_crop_affine(uv, up_uv, crop, params["crop_padding"])
                bgr = img[:, :, :3]
                crop_img = cv2.warpAffine(bgr, M, (crop, crop))
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))))
                ok = bool(result.face_landmarks)
                if ok:
                    p = np.array([[l.x * crop, l.y * crop, l.z * crop] for l in result.face_landmarks[0]], np.float64)
                    Minv = cv2.invertAffineTransform(M)
                    full = p[:, :2] @ Minv[:, :2].T + Minv[:, 2]
                    landmarks_px[name] = np.concatenate([full, p[:, 2:] * side / crop], 1).astype(np.float32)
                meta[name] = {"facing_deg": facing[name], "ok": ok, "side_px": side, "roll_deg": roll}
                if debug:
                    if ok:
                        for q in p[:, :2].astype(int):
                            cv2.circle(crop_img, (int(q[0]), int(q[1])), 1, (0, 255, 0), -1)
                    cv2.putText(crop_img, f"{name} facing {facing[name]:.0f} {'ok' if ok else 'NO FACE'}", (5, 20), 0, 0.5, (0, 255, 255), 1)
                    cv2.imwrite(str(debug / f"crop_{Path(name).stem}.jpg"), crop_img)
        finally:
            landmarker.close()
        found = [n for n in names if n in landmarks_px]
        frontal = [n for n in names if facing[n] <= 85]
        stats = {"views": len(names), "landmarked": len(found), "facing_le_85": len(frontal),
                 "landmarked_of_frontal": sum(1 for n in frontal if n in landmarks_px),
                 "crop_size": crop, "crop_padding": params["crop_padding"]}
        logger.info("detect_face_views: %d of %d frames landmarked from the projected head (%d of the %d facing the camera "
                    "within 85 deg)", stats["landmarked"], stats["views"], stats["landmarked_of_frontal"], stats["facing_le_85"])
        if debug:
            (debug / "meta.json").write_text(json.dumps({"stats": stats, "views": meta}, indent=1))
        return {"face_views": {"names": found, "landmarks_px": landmarks_px, "facing_deg": facing, "cameras": cams,
                               "image_size": (int(width), int(height)), "stats": stats}}


# -- fit_head_per_view ----------------------------------------------------------

def anchor_camera_pnp(object_points: np.ndarray, image_points: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """The anchor photograph's camera against the canonical head: EPnP, refined; (R, t, rms px)."""
    obj = np.ascontiguousarray(object_points, np.float64)
    img = np.ascontiguousarray(image_points, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("fit_head_per_view: PnP of the anchor's landmarks against the head failed")
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt((np.linalg.norm(proj[:, 0] - img, axis=1) ** 2).mean()))
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), rms


@register_step("fit_head_per_view")
class FitHeadPerViewStep(Step):
    """The MHR head fitted to every landmarked frame: neck/head pose and expression, per view.

    inputs:  {"mesh_output": dict — the refit body (vertices raw, pose_params, focal_length, faces),
              "world_from_raw": {...}, "face_correspondence": map_face_to_mesh's output,
              "face_views": detect_face_views' output,
              "face_landmarks": detect_face_landmarks' output on the anchor photograph,
              "image": the anchor photograph (only its size is read)}
    outputs: {"head_fit_views": {"names": [str] fitted views in training order,
                                 "verts_world": (B,N,3), "joints_world": (B,J,3) — the fitted head per view,
                                 "verts0_world": (N,3), "joints0_world": (J,3) — the canonical (refit) body,
                                 "dpose": (B,6) rad, "dexpr": (B,72),
                                 "rms_px": {"raw", "pose", "pose_expr": (B,)},
                                 "anchor": {"R","t","K","image_size","verts_world","rms_px","pnp_rms_px"},
                                 "expression_motion": (N,) cm — how far the expression basis can move each vertex,
                                 "faces": (F,3), "stats": {...}}}
    """

    PARAMS = (
        Param("max_facing_deg", float, 85.0,
              "Fit only the views whose camera is within this angle of the face normal; beyond it the landmarks "
              "are hair and profile guesses", minimum=0.0, maximum=180.0),
        Param("pose_prior", float, 50.0,
              "L2 on the six neck/head rotations (rad^2, against a pixel loss). Unregularised they counter-rotate "
              "into a lateral head shift; 50 removes it at no residual cost", minimum=0.0),
        Param("expression_prior", float, 1.0, "L2 on the expression coefficients", minimum=0.0),
        Param("iterations", int, 600, "Adam steps of the pose+expression stage; the pose warm-up runs half as many", minimum=1),
        Param("learning_rate", float, 0.01, "Adam step (radians for the rotations)", advanced=True),
        Param("huber_px", float, 4.0, "Landmark residuals beyond this count linearly", minimum=0.1, advanced=True),
        Param("pose_indices", str, ",".join(str(i) for i in DEFAULT_POSE_INDICES),
              "body_pose_params indices of the neck/head joint rotations", advanced=True),
        Param("checkpoint_repo", str, "facebook/sam-3d-body-dinov3", "HF repo the checkpoint is pulled from", advanced=True),
        Param("checkpoint_dir", str, None, "A local snapshot directory to use instead of downloading", advanced=True),
        Param("mhr_path", str, None, "The mhr_model.pt to load; empty means assets/mhr_model.pt inside the checkpoint directory", advanced=True),
        Param("device", str, "cuda", "Torch device", advanced=True),
        Param("debug_dir", str, "", "Write fit.json (per-view residuals, rotations) here"),
    )

    def __init__(self) -> None:
        self._head = None

    def load(self, params: Dict[str, Any]) -> None:
        from .head_fit import build_mhr_head
        self._head = build_mhr_head(params["checkpoint_repo"], params["checkpoint_dir"], params["mhr_path"], params["device"])

    def unload(self) -> None:
        self._head = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        mesh = inputs["mesh_output"]
        if not isinstance(mesh, dict) or mesh.get("pose_params") is None:
            raise KeyError("fit_head_per_view needs 'pose_params' inside mesh_output (the refit body's)")
        pose_params = mesh["pose_params"]
        if hasattr(pose_params, "item") and not isinstance(pose_params, dict):
            pose_params = pose_params.item()
        wfr = _check_world_from_raw(inputs, "fit_head_per_view")
        fv = inputs.get("face_views")
        if not isinstance(fv, dict) or "landmarks_px" not in fv:
            raise ValueError("fit_head_per_view needs 'face_views' from detect_face_views")
        corr = inputs.get("face_correspondence")
        if not isinstance(corr, dict):
            raise ValueError("fit_head_per_view needs 'face_correspondence' from map_face_to_mesh")
        lm_idx, vf_np = feature_landmarks(corr)
        if len(lm_idx) < 50:
            raise ValueError(f"fit_head_per_view: only {len(lm_idx)} feature landmarks are mapped; nothing to fit")
        if self._head is None:
            self.load(params)
        head = self._head
        device = params["device"]

        # --- the anchor photograph: landmarks and a PnP camera ----------------
        anchor_img = np.asarray(inputs["image"])
        a_h, a_w = anchor_img.shape[:2]
        lm = inputs["face_landmarks"]
        a_lw, a_lh = (lm.get("image_size") or (a_w, a_h))
        if (int(a_lw), int(a_lh)) != (a_w, a_h):
            raise ValueError(f"fit_head_per_view: the anchor's face_landmarks were detected on a {a_lw}x{a_lh} image "
                             f"but the photograph is {a_w}x{a_h}")
        anchor_px = np.asarray(lm["landmarks"], np.float64)[:, :2] * np.array([a_w, a_h])
        focal = float(mesh["focal_length"])
        K_anchor = np.array([[focal, 0.0, a_w / 2.0], [0.0, focal, a_h / 2.0], [0.0, 0.0, 1.0]])

        # --- the views ----------------------------------------------------------
        max_facing = float(params["max_facing_deg"])
        names = [n for n in fv["names"] if fv["facing_deg"][n] <= max_facing]
        dropped = [n for n in fv["names"] if n not in names]
        if not names:
            raise ValueError(f"fit_head_per_view: none of the {len(fv['names'])} landmarked views faces the camera "
                             f"within {max_facing} deg")

        # --- the body model, replayed in the world frame -----------------------
        def T(v):
            return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)[None]

        g0, b0, h0 = T(pose_params["global_rot"]), T(pose_params["body_pose_params"]), T(pose_params["hand_pose_params"])
        sc0, sh0, ex0, gt0 = T(pose_params["scale_params"]), T(pose_params["shape_params"]), T(pose_params["expr_params"]), T(pose_params["global_trans"])
        n_scales = int(head.scale_mean.shape[0])
        so0 = T(pose_params["scale_offsets"]) if pose_params.get("scale_offsets") is not None else torch.zeros(1, n_scales, device=device)
        flip = torch.tensor(FLIP, dtype=torch.float32, device=device)
        sw = float(wfr["scale"])
        Rw = torch.as_tensor(np.asarray(wfr["rotation"], np.float64).reshape(3, 3), dtype=torch.float32, device=device)
        tw = torch.as_tensor(np.asarray(wfr["translation"], np.float64).reshape(3), dtype=torch.float32, device=device)
        pose_idx = sorted({int(v) for v in str(params["pose_indices"]).split(",") if v.strip()})
        if not pose_idx or max(pose_idx) >= b0.shape[1]:
            raise ValueError(f"fit_head_per_view: pose_indices {params['pose_indices']!r} are outside body_pose_params")
        pose_idx_t = torch.tensor(pose_idx, device=device)
        n_pose = len(pose_idx)
        n_expr = int(ex0.shape[1])

        def forward(dpose, dexpr):
            B = dpose.shape[0]
            body = b0.expand(B, -1).clone()
            body[:, pose_idx_t] = body[:, pose_idx_t] + dpose
            verts, joints = head.mhr_forward(
                global_trans=gt0.expand(B, -1), global_rot=g0.expand(B, -1), body_pose_params=body,
                hand_pose_params=h0.expand(B, -1), scale_params=sc0.expand(B, -1), shape_params=sh0.expand(B, -1),
                expr_params=ex0.expand(B, -1) + dexpr, return_joint_coords=True, scale_offsets=so0.expand(B, -1))
            return sw * (verts * flip) @ Rw.T + tw, sw * (joints * flip) @ Rw.T + tw

        with torch.no_grad():
            v0, j0 = forward(torch.zeros(1, n_pose, device=device), torch.zeros(1, n_expr, device=device))
            raw0 = ((v0[0] - tw) @ Rw) / sw
        drift = float(np.abs(raw0.cpu().numpy() - np.asarray(mesh["vertices"])).max())
        if drift > 1e-3:
            raise RuntimeError(f"fit_head_per_view: replaying the pose parameters does not reproduce the body "
                               f"({drift * 1000:.2f} mm off); mesh_output and pose_params are from different fits")
        V0 = v0[0].cpu().numpy().astype(np.float64)

        R_a, t_a, pnp_rms = anchor_camera_pnp(V0[vf_np], anchor_px[lm_idx], K_anchor)
        logger.info("fit_head_per_view: anchor camera by PnP against the canonical head, %.2f px rms over %d landmarks",
                    pnp_rms, len(lm_idx))

        # --- the batch: the views, then the anchor ----------------------------
        cams = fv["cameras"]
        B = len(names) + 1
        Rv = torch.stack([torch.as_tensor(cams[n]["R"], dtype=torch.float32) for n in names] + [torch.as_tensor(R_a, dtype=torch.float32)]).to(device)
        tv = torch.stack([torch.as_tensor(cams[n]["t"], dtype=torch.float32) for n in names] + [torch.as_tensor(t_a, dtype=torch.float32)]).to(device)
        Kv = torch.stack([torch.as_tensor(cams[n]["K"], dtype=torch.float32) for n in names] + [torch.as_tensor(K_anchor, dtype=torch.float32)]).to(device)
        target = torch.stack([torch.as_tensor(np.asarray(fv["landmarks_px"][n], np.float64)[lm_idx, :2], dtype=torch.float32) for n in names]
                             + [torch.as_tensor(anchor_px[lm_idx], dtype=torch.float32)]).to(device)
        vf = torch.tensor(vf_np, device=device)
        # The anchor's head pose is what the body was fitted to; its CAMERA is the free part (the PnP is 2.5 px,
        # the fit brings it to 1.4). The frames' cameras are the refined dataset cameras and stay put.
        cam_free = torch.zeros(B, 1, device=device); cam_free[-1] = 1.0
        pose_free = torch.ones(B, 1, device=device); pose_free[-1] = 0.0
        cam_rv = torch.zeros(B, 3, device=device, requires_grad=True)
        cam_dt = torch.zeros(B, 3, device=device, requires_grad=True)
        dpose = torch.zeros(B, n_pose, device=device, requires_grad=True)
        dexpr = torch.zeros(B, n_expr, device=device, requires_grad=True)
        fxy = torch.stack([Kv[:, 0, 0], Kv[:, 1, 1]], 1)[:, None]
        cxy = torch.stack([Kv[:, 0, 2], Kv[:, 1, 2]], 1)[:, None]

        def project_t(vw):
            Rc = rodrigues_torch(cam_rv * cam_free) @ Rv
            tc = tv + cam_dt * cam_free
            c = torch.einsum("bij,bmj->bmi", Rc, vw) + tc[:, None]
            return c[..., :2] / c[..., 2:3].clamp(min=1e-3) * fxy + cxy

        def rms(r):
            return r.pow(2).sum(-1).mean(-1).sqrt()

        lam_p, lam_e, huber = float(params["pose_prior"]), float(params["expression_prior"]), float(params["huber_px"])

        def solve(iterations, lr, with_expr):
            groups = [{"params": [dpose], "lr": lr}, {"params": [cam_rv], "lr": lr * 0.5}, {"params": [cam_dt], "lr": lr * 0.05}]
            if with_expr:
                groups.append({"params": [dexpr], "lr": lr * 2})
            opt = torch.optim.Adam(groups)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)
            for _ in range(iterations):
                opt.zero_grad()
                vw, _ = forward(dpose * pose_free, dexpr if with_expr else dexpr * 0)
                r = target - project_t(vw[:, vf])
                loss = 2 * torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=huber, reduction="mean")
                loss = loss + lam_p * (dpose ** 2).mean()
                if with_expr:
                    loss = loss + lam_e * (dexpr ** 2).mean()
                loss.backward()
                opt.step()
                sch.step()

        iterations = int(params["iterations"])
        lr = float(params["learning_rate"])
        with torch.no_grad():
            r0 = rms(target - project_t(forward(dpose, dexpr)[0][:, vf]))
        solve(max(iterations // 2, 1), lr * 2, with_expr=False)
        with torch.no_grad():
            r1 = rms(target - project_t(forward(dpose * pose_free, dexpr * 0)[0][:, vf]))
        solve(iterations, lr, with_expr=True)
        with torch.no_grad():
            dpose.mul_(pose_free)
            dpose, dexpr = dpose.detach(), dexpr.detach()
            vw, jw = forward(dpose, dexpr)
            r2 = rms(target - project_t(vw[:, vf]))
            R_anchor = (rodrigues_torch(cam_rv * cam_free) @ Rv)[-1].cpu().numpy().astype(np.float64)
            t_anchor = (tv + cam_dt * cam_free)[-1].cpu().numpy().astype(np.float64)
            head_shift = (jw[:-1, MHR_HEAD_JOINT] - j0[0, MHR_HEAD_JOINT]).norm(dim=1).cpu().numpy() * 1000
            E = head.mhr.face_expressions_model.shape_vectors            # [72, N, 3], centimetres
            expression_motion = E.abs().amax(dim=(0, 2)).cpu().numpy().astype(np.float32)

        r0n, r1n, r2n = (r.cpu().numpy() for r in (r0, r1, r2))
        stats = {
            "views": {"landmarked": len(fv["names"]), "fitted": len(names), "dropped_facing": dropped},
            "rms_px_mean": {"raw": float(r0n[:-1].mean()), "pose": float(r1n[:-1].mean()), "pose_expr": float(r2n[:-1].mean())},
            "anchor": {"pnp_rms_px": pnp_rms, "rms_px": float(r2n[-1])},
            "pose_deg": {"rms": float(np.degrees(np.sqrt((dpose[:-1] ** 2).mean().item()))),
                         "max": float(np.degrees(dpose[:-1].abs().max().item()))},
            "head_joint_shift_mm": {"median": float(np.median(head_shift)), "max": float(head_shift.max())},
            "expression": {"rms": float(dexpr.pow(2).mean().sqrt().item()), "max": float(dexpr.abs().max().item())},
            "pose_prior": lam_p, "expression_prior": lam_e, "iterations": iterations, "max_facing_deg": max_facing,
        }
        logger.info("fit_head_per_view: %d views fitted (%d landmarked, %d dropped beyond %.0f deg): landmarks %.2f -> %.2f "
                    "(pose) -> %.2f px rms (pose + expression); anchor %.2f -> %.2f px; head rotations %.1f deg rms, "
                    "max %.1f; head joint moved %.1f mm median, %.1f max",
                    len(names), len(fv["names"]), len(dropped), max_facing, stats["rms_px_mean"]["raw"],
                    stats["rms_px_mean"]["pose"], stats["rms_px_mean"]["pose_expr"], pnp_rms, stats["anchor"]["rms_px"],
                    stats["pose_deg"]["rms"], stats["pose_deg"]["max"],
                    stats["head_joint_shift_mm"]["median"], stats["head_joint_shift_mm"]["max"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            per_view = {n: {"facing_deg": fv["facing_deg"][n], "rms_px": [float(r0n[i]), float(r1n[i]), float(r2n[i])],
                            "pose_deg": [float(v) for v in np.degrees(dpose[i].cpu().numpy())],
                            "expression_rms": float(dexpr[i].pow(2).mean().sqrt().item())} for i, n in enumerate(names)}
            (debug / "fit.json").write_text(json.dumps({"stats": stats, "views": per_view}, indent=1))
        verts_world = vw.cpu().numpy().astype(np.float32)
        joints_world = jw.cpu().numpy().astype(np.float32)
        return {"head_fit_views": {
            "names": names,
            "verts_world": verts_world[:-1], "joints_world": joints_world[:-1],
            "verts0_world": V0.astype(np.float32), "joints0_world": j0[0].cpu().numpy().astype(np.float32),
            "dpose": dpose[:-1].cpu().numpy(), "dexpr": dexpr[:-1].cpu().numpy(),
            "rms_px": {"raw": r0n[:-1], "pose": r1n[:-1], "pose_expr": r2n[:-1]},
            "anchor": {"R": R_anchor, "t": t_anchor, "K": K_anchor, "image_size": (int(a_w), int(a_h)),
                       "verts_world": verts_world[-1], "rms_px": float(r2n[-1]), "pnp_rms_px": pnp_rms},
            "expression_motion": expression_motion,
            "faces": np.asarray(mesh["faces"], np.int32),
            "stats": stats,
        }}


# -- paste_eyes -----------------------------------------------------------------

class EyeModel:
    """Two textured eyeballs on the canonical head, rendered through fitted lids.

    Built from the canonical (refit) head `V0`, its joints `J0`, the mesh
    faces and the landmark->vertex map: per eye the lid ring vertices, the
    eye joint (the nearer of the MHR's two), the eye-surface faces removed
    from the occluder (centroid inside the lid contour, within 2 cm of the
    joint). The sphere sits at the joint with `radius`; per view it moves
    rigidly with its lid ring (Kabsch of the ring, canonical -> fitted).
    """

    def __init__(self, V0: np.ndarray, J0: np.ndarray, faces: np.ndarray, vertex_of_landmark: np.ndarray,
                 radius: float, supersample: int, depth_tolerance: float, sphere_subdivisions: int = 5) -> None:
        import trimesh

        self.V0, self.J0, self.F = np.asarray(V0, np.float64), np.asarray(J0, np.float64), np.asarray(faces, np.int64)
        self.radius, self.ss, self.depth_tol = float(radius), int(supersample), float(depth_tolerance)
        vol = np.asarray(vertex_of_landmark, np.int64)
        if max(MHR_EYE_JOINTS) + 1 >= len(self.J0):
            raise ValueError(f"paste_eyes: the body has {len(self.J0)} joints, not the MHR's 127")
        self.eyes: Dict[str, Dict[str, Any]] = {}
        centroids = self.V0[self.F].mean(1)
        removed = np.zeros(len(self.F), bool)
        for side, ring_lm, iris_lm in EYE_SIDES:
            ring = np.array([int(vol[i]) for i in ring_lm if vol[i] >= 0])
            if len(ring) < 8:
                raise ValueError(f"paste_eyes: only {len(ring)} of the {len(ring_lm)} {side} lid landmarks map to a vertex")
            ring_c = self.V0[ring].mean(0)
            j = min(MHR_EYE_JOINTS, key=lambda k: np.linalg.norm(self.J0[k] - ring_c))
            if np.linalg.norm(self.J0[j] - ring_c) > 0.03:
                raise ValueError(f"paste_eyes: MHR joint {j} is {np.linalg.norm(self.J0[j] - ring_c) * 100:.1f} cm from the "
                                 f"{side} lid ring; not an eye joint on this body")
            near = np.linalg.norm(centroids - self.J0[j], axis=1) < 0.02
            eye_faces = np.zeros(len(self.F), bool)
            eye_faces[near] = inside_contour_cylinder(centroids[near], self.V0[ring])
            removed |= eye_faces
            gaze = self.J0[j + 1] - self.J0[j]
            self.eyes[side] = {"joint": j, "ring": ring, "centre": self.J0[j].copy(), "iris_landmark": iris_lm,
                               "gaze": gaze / max(np.linalg.norm(gaze), 1e-12), "eye_faces": int(eye_faces.sum())}
        self.occluder_faces = self.F[~removed]
        sphere = trimesh.creation.icosphere(subdivisions=sphere_subdivisions, radius=self.radius)
        self.SV, self.SF = np.asarray(sphere.vertices, np.float64), np.asarray(sphere.faces, np.int64)
        self.colors: Dict[str, np.ndarray] = {}

    # -- rendering ------------------------------------------------------------------
    @staticmethod
    def _pose_gl(R, t):
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, t
        return np.linalg.inv(M) @ np.diag([1.0, -1.0, -1.0, 1.0])

    def _render(self, meshes, R, t, K, w, h, scale, off, nocull=False):
        import pyrender

        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        for m in meshes:
            scene.add(m)
        cam = pyrender.IntrinsicsCamera(K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] - off[0]) * scale, (K[1, 2] - off[1]) * scale, znear=0.05, zfar=20.0)
        scene.add(cam, pose=self._pose_gl(R, t))
        renderer = pyrender.OffscreenRenderer(w, h)
        try:
            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT | (pyrender.RenderFlags.SKIP_CULL_FACES if nocull else 0)
            color, depth = renderer.render(scene, flags=flags)
        finally:
            renderer.delete()
        return np.asarray(color), np.asarray(depth)

    def layer(self, verts: np.ndarray, img: np.ndarray, R, t, K, *, sample: bool = False,
              min_fraction: float = 0.5, dark_min: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Dict[str, Any]]:
        """Both eyeballs for one view (`verts` = that view's fitted head).

        Returns the premultiplied RGB layer and its coverage alpha (frame
        resolution), the layer's top-left corner, and per eye either the
        texture samples (`sample=True`: colour and validity per sphere
        vertex, the eye's rigid transform) or why it was skipped.
        """
        import pyrender
        import trimesh

        H, W = img.shape[:2]
        ss = self.ss
        rings = {s: verts[e["ring"]] for s, e in self.eyes.items()}
        ring_uv = {s: project(r, R, t, K)[0] for s, r in rings.items()}
        all_uv = np.concatenate(list(ring_uv.values()))
        zr = project(np.concatenate(list(rings.values())), R, t, K)[1]
        empty = (np.zeros((1, 1, 3), np.float32), np.zeros((1, 1), np.float32), (0, 0))
        if zr.min() <= 0.05:
            return (*empty, {s: {"skipped": "behind the camera"} for s in self.eyes})
        x0, y0 = np.maximum(np.floor(all_uv.min(0) - 40).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(all_uv.max(0) + 40).astype(int), [W, H])
        w, h = (x1 - x0) * ss, (y1 - y0) * ss
        if w <= 0 or h <= 0:
            return (*empty, {s: {"skipped": "outside the frame"} for s in self.eyes})
        off = np.array([x0, y0])
        layer = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros((h, w), np.float32)
        info: Dict[str, Any] = {}
        occluder = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, self.occluder_faces, process=False))
        _, dep_occl = self._render([occluder], R, t, K, w, h, ss, off, nocull=True)
        gray = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY).astype(np.float32) if not sample else None
        for side, E in self.eyes.items():
            Rr, tr = kabsch(self.V0[E["ring"]], rings[side])
            centre = Rr @ E["centre"] + tr
            sv = self.SV @ Rr.T + centre
            inside = inside_contour_cylinder(sv, rings[side])
            opening = self.SF[inside[self.SF].all(1)]
            if len(opening) == 0:
                info[side] = {"skipped": "lids closed"}
                continue
            col = self.colors.get(side)
            if col is None:
                col = np.full((len(self.SV), 3), 200, np.uint8)
            tm = trimesh.Trimesh(sv, opening, vertex_colors=np.c_[col, np.full(len(self.SV), 255)], process=False)
            rgba, dep = self._render([pyrender.Mesh.from_trimesh(tm)], R, t, K, w, h, ss, off)
            vis = (dep > 0) & (dep_occl > 0) & (dep < dep_occl + self.depth_tol)
            poly = np.zeros((h, w), np.uint8)
            cv2.fillPoly(poly, [((ring_uv[side] - off) * ss).astype(np.int32)], 255)
            fraction = float(vis.sum() / max((poly > 0).sum(), 1))
            if not sample:
                # The frame must show an eye there: something darker than the skin around the lid polygon. A hand
                # in front of the face does not, and the head mesh cannot know about the hand.
                pm = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(pm, [ring_uv[side].astype(np.int32)], 255)
                around = (cv2.dilate(pm, np.ones((15, 15), np.uint8)) > 0) & ~(pm > 0)
                dark = float((gray[pm > 0] < 0.6 * np.median(gray[around])).mean()) if pm.any() and around.any() else 0.0
                if dark < dark_min:
                    info[side] = {"skipped": "nothing dark inside the lids (occluded?)", "dark": dark, "fraction": fraction}
                    continue
                if fraction < min_fraction:
                    info[side] = {"skipped": "less than half the lid polygon visible", "dark": dark, "fraction": fraction}
                    continue
            layer[vis] = rgba[vis, :3]
            alpha[vis] = 1.0
            entry: Dict[str, Any] = {"fraction": fraction, "pixels": int(vis.sum() / (ss * ss))}
            if sample:
                # colour per sphere vertex from the image where the sphere is visible and front-facing
                uv, z = project(sv, R, t, K)
                cam_c = -R.T @ t
                front = ((sv - cam_c) * (sv - centre)).sum(1) < 0
                q = ((uv - off) * ss).round().astype(int)
                inb = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < w) & (q[:, 1] < h)
                ok = front & inb
                ok[ok] &= vis[q[ok, 1], q[ok, 0]] & (np.abs(dep[q[ok, 1], q[ok, 0]] - z[ok]) < 1.5 * self.depth_tol)
                colour = np.zeros((len(self.SV), 3), np.float32)
                colour[ok] = cv2.remap(np.ascontiguousarray(img[..., :3]).astype(np.float32), uv[ok, 0:1].astype(np.float32), uv[ok, 1:2].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
                entry.update({"sampled": ok, "colour": colour, "R": Rr, "t": tr})
            info[side] = entry
        layer_d = cv2.resize(layer * alpha[..., None], (w // ss, h // ss), interpolation=cv2.INTER_AREA)
        alpha_d = cv2.resize(alpha, (w // ss, h // ss), interpolation=cv2.INTER_AREA)
        return layer_d, alpha_d, (int(x0), int(y0)), info

    # -- texture ------------------------------------------------------------------
    def texture_from_anchor(self, anchor_verts: np.ndarray, anchor_img: np.ndarray, R, t, K,
                            anchor_landmarks_px: np.ndarray, iris_deg: float) -> Dict[str, Any]:
        """Colour every sphere vertex from the anchor photograph through its fitted lids.

        The gaze axis is where the ray through the anchor's iris landmark
        hits the sphere. The vertices the anchor does not show (under its
        upper lid, mostly the top ~30% of the iris) take a radial colour
        profile around that axis, 2-degree bins; beyond the iris they take
        the sclera, one median colour shared by both eyes.
        """
        _, _, _, samples = self.layer(anchor_verts, anchor_img, R, t, K, sample=True)
        stats: Dict[str, Any] = {}
        scleras: List[np.ndarray] = []
        profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for side, E in self.eyes.items():
            s = samples.get(side, {})
            if "sampled" not in s:
                raise RuntimeError(f"paste_eyes: the anchor photograph shows no {side} eye through the fitted lids "
                                   f"({s.get('skipped', 'no samples')})")
            ok, colour, Rr, tr = s["sampled"], s["colour"], s["R"], s["t"]
            centre = Rr @ E["centre"] + tr
            lm = anchor_landmarks_px[E["iris_landmark"]]
            d = R.T @ (np.linalg.inv(K) @ np.array([lm[0], lm[1], 1.0]))
            d /= np.linalg.norm(d)
            o = -R.T @ t
            oc = o - centre
            b = d @ oc
            disc = b * b - (oc @ oc - self.radius ** 2)
            hit = o + (-b - np.sqrt(disc)) * d if disc > 0 else centre + self.radius * (-R[2])
            axis = Rr.T @ (hit - centre)
            axis /= np.linalg.norm(axis)
            ang = np.degrees(np.arccos(np.clip((self.SV / self.radius) @ axis, -1.0, 1.0)))
            scl = ok & (ang > iris_deg + 6)
            sclera = np.median(colour[scl], 0) if scl.sum() > 20 else np.array([200.0, 200.0, 200.0])
            bins = np.arange(0, 91, 2)
            prof = np.zeros((len(bins) - 1, 3))
            have = np.zeros(len(bins) - 1, bool)
            for k in range(len(bins) - 1):
                m = ok & (ang >= bins[k]) & (ang < bins[k + 1])
                if m.sum() >= 3:
                    prof[k], have[k] = np.median(colour[m], 0), True
            last = int(np.flatnonzero(have).max()) if have.any() else -1
            for k in range(len(bins) - 1):
                if not have[k]:
                    prof[k] = prof[np.flatnonzero(have)[np.abs(np.flatnonzero(have) - k).argmin()]] if k <= last else sclera
            col = colour.copy()
            kk = np.clip((ang // 2).astype(int), 0, len(prof) - 1)
            col[~ok] = prof[kk][~ok]
            self.colors[side] = np.clip(col, 0, 255).astype(np.uint8)
            profiles[side] = (ok, ang)
            scleras.append(sclera)
            stats[side] = {"sampled": int(ok.sum()), "sphere_vertices": int(len(self.SV)), "iris_axis_hit": bool(disc > 0),
                           "iris_sampled": int((ok & (ang < iris_deg)).sum()), "iris_vertices": int((ang < iris_deg).sum()),
                           "sclera_bgr": [float(v) for v in sclera]}
        sclera = np.mean(scleras, 0)
        for side, (ok, ang) in profiles.items():
            self.colors[side][~ok & (ang > iris_deg + 6)] = sclera.astype(np.uint8)
        return stats


@register_step("paste_eyes")
class PasteEyesStep(Step):
    """The anchor's eyes, as textured eyeballs, rendered into every fitted frame through its lids.

    inputs:  {"images": [BGR(A)], "image_names": [str], "cameras": [Camera],
              "head_fit_views": fit_head_per_view's output, "face_correspondence": map_face_to_mesh's,
              "image": the anchor photograph, "face_landmarks": its landmarks (detect_face_landmarks)}
    outputs: {"images": the frames, the fitted ones with the eyes pasted (the rest are the same arrays),
              "eye_stats": {...}}
    """

    PARAMS = (
        Param("eye_radius_mm", float, 15.5,
              "Eyeball radius; the MHR lid ring sits 15.4-20.6 mm from the eye joint and a sphere fitted to its eye "
              "surface has r 15.8-16.4, so this sits just inside the lids", minimum=5.0, maximum=30.0),
        Param("min_visible_fraction", float, 0.5,
              "Skip an eye whose visible part covers less than this fraction of its lid polygon: the far eye "
              "peeking past the model's narrower nose is not the frame's eye", minimum=0.0, maximum=1.0),
        Param("dark_min", float, 0.3,
              "Skip an eye when less than this fraction of the pixels inside the lid polygon is darker than 0.6x "
              "the skin around it — a hand in front of the face has no eye to replace", minimum=0.0, maximum=1.0),
        Param("iris_deg", float, 26.0, "Angular radius of the iris about the gaze axis", minimum=5.0, maximum=60.0, advanced=True),
        Param("depth_tolerance_mm", float, 1.0, "The sphere shows where it is within this of the head mesh's depth", minimum=0.1, advanced=True),
        Param("supersample", int, 4, "Render the eye region at this multiple of the frame's pixels", minimum=1, maximum=8, advanced=True),
        Param("debug_dir", str, "", "Write eye_colors.npz, the per-view eye masks and a before/after panel here"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from . import render as _render_module  # noqa: F401 — sets PYOPENGL_PLATFORM before pyrender loads

        images, names, cameras = list(inputs["images"]), list(inputs["image_names"]), list(inputs["cameras"])
        if not (len(images) == len(names) == len(cameras)):
            raise ValueError("paste_eyes: images, image_names and cameras disagree in length")
        fit = inputs.get("head_fit_views")
        if not isinstance(fit, dict) or "verts_world" not in fit:
            raise ValueError("paste_eyes needs 'head_fit_views' from fit_head_per_view")
        corr = inputs.get("face_correspondence")
        if not isinstance(corr, dict) or "vertex_of_landmark" not in corr:
            raise ValueError("paste_eyes needs 'face_correspondence' from map_face_to_mesh")
        anchor_img = np.asarray(inputs["image"])
        lm = inputs["face_landmarks"]
        a_h, a_w = anchor_img.shape[:2]
        anchor_px = np.asarray(lm["landmarks"], np.float64)[:, :2] * np.array([a_w, a_h])
        anchor = fit["anchor"]
        if tuple(int(v) for v in anchor["image_size"]) != (a_w, a_h):
            raise ValueError(f"paste_eyes: the head fit's anchor camera is for a {anchor['image_size']} image, the photograph is {a_w}x{a_h}")

        model = EyeModel(fit["verts0_world"], fit["joints0_world"], fit["faces"], corr["vertex_of_landmark"],
                         radius=params["eye_radius_mm"] / 1000.0, supersample=params["supersample"],
                         depth_tolerance=params["depth_tolerance_mm"] / 1000.0)
        texture = model.texture_from_anchor(np.asarray(anchor["verts_world"], np.float64), anchor_img,
                                            np.asarray(anchor["R"], np.float64), np.asarray(anchor["t"], np.float64),
                                            np.asarray(anchor["K"], np.float64), anchor_px, params["iris_deg"])
        logger.info("paste_eyes: eye model textured from the anchor — %s", "; ".join(
            f"{s}: {v['sampled']}/{v['sphere_vertices']} sphere vertices, iris {v['iris_sampled']}/{v['iris_vertices']}"
            for s, v in texture.items()))

        debug = Path(params["debug_dir"]) if params["debug_dir"] else None
        if debug:
            debug.mkdir(parents=True, exist_ok=True)
            np.savez(debug / "eye_colors.npz", **model.colors, sphere_vertices=model.SV, sphere_faces=model.SF, radius=model.radius)
        index = {n: i for i, n in enumerate(names)}
        out = list(images)
        per_view: Dict[str, Any] = {}
        pasted = 0
        eyes_drawn = 0
        panel: List[np.ndarray] = []
        for k, name in enumerate(fit["names"]):
            i = index.get(name)
            if i is None:
                raise ValueError(f"paste_eyes: fitted view {name!r} is not among the training frames")
            img = images[i]
            R, t, K = opencv_camera(cameras[i])
            layer, a, (x0, y0), info = model.layer(np.asarray(fit["verts_world"][k], np.float64), img, R, t, K,
                                                   min_fraction=params["min_visible_fraction"], dark_min=params["dark_min"])
            per_view[name] = {s: {kk: v for kk, v in e.items() if kk in ("skipped", "fraction", "pixels", "dark")} for s, e in info.items()}
            drawn = sum(1 for e in info.values() if "skipped" not in e)
            if drawn == 0:
                continue
            h, w = a.shape
            res = img.astype(np.float32)
            roi = res[y0:y0 + h, x0:x0 + w, :3]
            res[y0:y0 + h, x0:x0 + w, :3] = roi * (1 - a[..., None]) + layer
            res = np.clip(res, 0, 255).astype(img.dtype)
            out[i] = res
            pasted += 1
            eyes_drawn += drawn
            if debug:
                m = np.zeros(img.shape[:2], np.uint8)
                m[y0:y0 + h, x0:x0 + w] = (a * 255).astype(np.uint8)
                cv2.imwrite(str(debug / f"mask_{Path(name).stem}.png"), m)
                if len(panel) < 40:
                    # the eye region before and after, side by side, at 4x
                    m = max(w, h) // 4
                    ys, xs = slice(max(y0 - m, 0), y0 + h + m), slice(max(x0 - m, 0), x0 + w + m)
                    before, after = img[ys, xs, :3], res[ys, xs, :3]
                    if before.size:
                        k = 96.0 / before.shape[0]
                        tile = cv2.resize(np.concatenate([before, after], 1), None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)[:96, :480]
                        tile = np.pad(tile, ((0, 0), (0, 480 - tile.shape[1]), (0, 0)))
                        cv2.putText(tile, Path(name).stem, (3, 12), 0, 0.4, (0, 255, 255), 1)
                        panel.append(tile)
        stats = {"fitted_views": len(fit["names"]), "views_pasted": pasted, "eyes_drawn": eyes_drawn,
                 "eye_radius_mm": params["eye_radius_mm"], "texture": texture, "views": per_view}
        skipped: Dict[str, int] = {}
        for v in per_view.values():
            for e in v.values():
                if "skipped" in e:
                    skipped[e["skipped"]] = skipped.get(e["skipped"], 0) + 1
        logger.info("paste_eyes: eyes pasted into %d of %d fitted frames (%d eyes drawn; skipped: %s)", pasted, len(fit["names"]), eyes_drawn,
                    ", ".join(f"{n} {reason}" for reason, n in skipped.items()) or "none")
        if debug:
            (debug / "eyes.json").write_text(json.dumps(stats, indent=1))
            if panel:
                rows = [np.concatenate(panel[r:r + 3] + [np.zeros((96, 480, 3), np.uint8)] * (3 - len(panel[r:r + 3])), 1) for r in range(0, len(panel), 3)]
                cv2.imwrite(str(debug / "panel.jpg"), np.concatenate(rows, 0))
        return {"images": out, "eye_stats": stats}
