"""The face detailer: short 480p denoise clips around the finished splat's head, trained back into it.

The old StableDiffusion FaceDetailer's idea on a splat (`face_detailer` / `hair_detailer`, 2026-10-18,
experiments/face_detailer, memory `face_detailer`). The deliverable splat is rendered along a path that keeps
the head filling a 480x832 frame, the render goes through Wan 2.2 VACE as a control video (the anchor
photograph warped into the first frame, inactive; the photograph's face as the reference), and the splat is
retrained, warm, with growth off, on its own frames plus the clip. Measured on two subjects (0256db, 9016a1),
on held-out face views: triangulated iris reprojection error 1.51 -> 1.17 px, mouth 1.60 -> 1.15, the face's
skin and tattoos as the photograph's, the splat count unchanged. The recipe, every part of which was measured
against its absence:

  * clip 1: a Lissajous figure, yaw +-45, pitch +-20 about the photograph's view, two vertical periods, at
    constant angular speed (3.2 deg/frame; the video model wants no acceleration and no reversal of one
    axis at a time, which a single arc or a back-and-forth sweep gives it);
  * clip 2: PROGRESSIVE — rendered from a splat already retrained on clip 1 — yaw +-90, pitch -40..+20
    (5.5 deg/frame). An independent second clip from the original splat softens the front (two
    hallucinations averaged); a progressive one does not, and it is what reaches the profiles and the
    views from below;
  * the pipeline's own `paste_eyes` on both clips, with the photograph's face crop upscaled 4x by SeedVR2 for
    the eye texture: Wan's eyes follow the camera (iris error 2.25 px on a single clip). Eyes the paste
    skips get zero loss weight (`face_detail_clip`);
  * the face rig v3 (`face_detail_rig`), constrained: the original views get no deltas, the clip frames'
    mean is removed (Wan moves the whole face 2-5 mm against the splat, a common offset the canonical face
    must not take), a +-3 frame temporal median, a 10 mm cap. The raw per-view fits are 8-16 mm noisy;
    constrained they are 1.5-2 mm;
  * the retraining: warm start from the deliverable, 6000 iterations, growth off, two alignment iterations
    (the largest single lever of the retraining: front sharpness 32 -> 42 and 76 -> 94);
  * every clip in 4 denoise steps (2 per expert, strength 0.8 then 0.5): against the measured 6 (2 + 4, 0.5 on
    the last four) on 9016a1, iris / mouth agreement 1.12 / 1.07 px against 1.06 / 1.23, photo-view PSNR 18.1
    against 17.1 dB, face sharpness 6-12 % lower, the three clips 550 s against 689 s on a 4070 Ti;
  * the hair clip (`hair_detailer`): one turn of a helix about the hair, elevation -20..+30, the sheet's back
    panel as the reference, rendered from the face-detailed splat, its frames supervising only the hair
    (loss weights = the original splat's hair class at those cameras). Judged by eye: the back of the head
    goes from clumps to fine strands matching the front.

Five steps here, the rest are the pipeline's own (wan22_vace_denoise, rmbg, seedvr2, detect_face_views,
fit_head_per_view, paste_eyes, brush):

  face_detail_views        the path's cameras, the splat rendered along it (the control video and its VACE
                           masks), the anchor frame, the reference, and for the hair clip its loss weights
  face_detail_photo_crop   the photograph's face, cut out square for one SeedVR2 frame
  face_detail_photo_merge  the photograph 4x with that crop feathered in: paste_eyes' eye texture
  face_detail_clip         one denoised clip as training views: names, matte, loss weights
  face_detail_dataset      the final training's views plus the clips, the sidecars per view
  face_detail_rig          the rig's per-view face deltas on the clips' frames, constrained

Coordinates: the dataset's world frame, y up (the orbit's). Cameras are body2colmap Cameras (OpenGL
camera-to-world rotation and a position); `_cv_c2w` / `_camera` convert.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import Param, Step
from .face_views import LID_RING_IMAGE_LEFT, LID_RING_IMAGE_RIGHT, MHR_HEAD_JOINT, to_world

logger = logging.getLogger(__name__)

_GL = np.diag([1.0, -1.0, -1.0])
_SH_C0 = 0.28209479177387814
#: Sapiens2's Hair class (steps/sapiens2.py SEG_CLASSES), as the trainer votes it onto the splats.
HAIR_LABEL = 4
#: The grey the clips stand on: the renders' background and the matted photograph's.
_GREY = 128


# -- the path -----------------------------------------------------------------------------------------------

def _directions(yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)], -1)


def path_offsets(path: str, frames: int, yaw_deg: float, pitch_deg: float, pitch_offset_deg: float,
                 ratio: float = 2.0, turns: float = 1.0, samples: int = 4001
                 ) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """(yaw, pitch) offsets in degrees for `frames` frames at constant angular speed, the frame that sits on
    the start pose (0, 0), and the speed in degrees per frame.

    lissajous  yaw = Y sin(2 pi t), pitch = P sin(2 pi ratio t + phase) + offset; the phase puts t = 0 at
               pitch 0 even with an offset, so the first frame is the start pose.
    helix      yaw = 360 turns (t - 1/2), pitch = P (2t - 1) + offset; the middle frame faces the start yaw.

    A dense curve re-sampled by arc length on the view sphere, split at the start pose so that it lands
    exactly on a frame."""
    tau = np.linspace(0.0, 1.0, samples)
    if path == "lissajous":
        phase = float(np.arcsin(np.clip(-pitch_offset_deg / pitch_deg, -1.0, 1.0))) if pitch_offset_deg and pitch_deg else 0.0
        u = yaw_deg * np.sin(2 * np.pi * tau)
        v = pitch_deg * np.sin(2 * np.pi * ratio * tau + phase) + pitch_offset_deg
        t_start = 0.0
    elif path == "helix":
        u = 360.0 * turns * (tau - 0.5)
        v = pitch_deg * (2 * tau - 1) + pitch_offset_deg
        t_start = 0.5
    else:
        raise ValueError(f"face_detail_views: unknown path {path!r} (lissajous, helix)")
    d = _directions(u, v)
    step = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(-1), -1.0, 1.0)))
    s = np.concatenate([[0.0], np.cumsum(step)])
    total = s[-1]
    s_start = float(np.interp(t_start, tau, s))
    k = int(round(s_start / total * (frames - 1)))
    q = np.concatenate([np.linspace(0.0, s_start, k + 1), np.linspace(s_start, total, frames - k)[1:]])
    return np.interp(q, s, u), np.interp(q, s, v), k, total / (frames - 1)


def _look_at_cv(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world (x right, y down, z forward), world y up."""
    z = target - position
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, position
    return m


def _cv_c2w(camera: Any) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = np.asarray(camera.rotation, np.float64) @ _GL
    m[:3, 3] = np.asarray(camera.position, np.float64).reshape(3)
    return m


def _camera(c2w_cv: np.ndarray, fx: float, width: int, height: int) -> Any:
    from body2colmap.camera import Camera

    return Camera(focal_length=(fx, fx), image_size=(width, height), principal_point=(width / 2.0, height / 2.0),
                  position=c2w_cv[:3, 3].astype(np.float32), rotation=(c2w_cv[:3, :3] @ _GL).astype(np.float32))


def _intrinsics(camera: Any) -> np.ndarray:
    return np.array([[float(camera.fx), 0.0, float(camera.cx)], [0.0, float(camera.fy), float(camera.cy)], [0.0, 0.0, 1.0]])


# -- the splat ----------------------------------------------------------------------------------------------

def _opacity(scene: Any) -> np.ndarray:
    o = np.asarray(scene.opacities, np.float64).reshape(-1)
    return 1.0 / (1.0 + np.exp(-o)) if (o.min() < 0.0 or o.max() > 1.0) else o


def _labels(scene: Any) -> Optional[np.ndarray]:
    extras = getattr(scene, "extras", None) or {}
    return np.asarray(extras["seg_label"]).astype(np.int64) if "seg_label" in extras else None


def hair_centre(scene: Any, below_m: float = 0.15) -> Optional[np.ndarray]:
    """The hair's bounding-box centre across, and halfway between the hair's top and `below_m` under its
    bottom in height — the head and shoulders, which a helix about it keeps in frame. None without labels."""
    labels = _labels(scene)
    if labels is None:
        return None
    hair = np.asarray(scene.means, np.float64)[(labels == HAIR_LABEL) & (_opacity(scene) > 0.5)]
    if len(hair) < 100:
        return None
    lo, hi = np.percentile(hair, 2, axis=0), np.percentile(hair, 98, axis=0)
    return np.array([(lo[0] + hi[0]) / 2, (hi[1] + lo[1] - below_m) / 2, (lo[2] + hi[2]) / 2])


def class_mask_scene(scene: Any, label: int) -> Any:
    """The splat with every Gaussian of class `label` white and every other black, base colour only: its
    render over black is that class's accumulated alpha, occluded by the rest."""
    from body2colmap.splat_scene import SplatScene

    labels = _labels(scene)
    if labels is None:
        raise ValueError("face_detail_views: the splat carries no seg_label (splat_labels off?)")
    sh = np.zeros((len(labels), 1, 3), np.float32)
    sh[:, 0, :] = np.where((labels == label)[:, None], 0.5 / _SH_C0, -0.5 / _SH_C0)
    return SplatScene(means=scene.means, scales=scene.scales, quats=scene.quats, opacities=scene.opacities,
                      sh_coeffs=sh, sh_degree=0)


# -- the photograph -----------------------------------------------------------------------------------------

def register_photo(front: np.ndarray, render: np.ndarray, min_inliers: int = 12) -> Optional[Tuple[np.ndarray, int, int]]:
    """The similarity (2x3) taking the front panel onto the splat's render at the photograph's camera, SIFT +
    RANSAC; (A, inliers, matches), or None when it does not register. A generated panel against a splat render
    matches sparsely (28 inliers of 145 on 9016a1, 72 of 264 on 0256db, both registered right), so the floor is
    low and the similarity itself is checked: the panel sits upright in the frame at a plausible scale."""
    sift = cv2.SIFT_create(20000, contrastThreshold=0.02)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(front[..., :3], cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(render[..., :3], cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return None
    good = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([k1[g.queryIdx].pt for g in good])
    dst = np.float32([k2[g.trainIdx].pt for g in good])
    A, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=6, maxIters=50000, confidence=0.999)
    if A is None or int(inliers.sum()) < min_inliers:
        return None
    scale, roll = float(np.hypot(*A[0, :2])), float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    if not 0.25 < scale < 4.0 or abs(roll) > 15.0:
        return None
    return A, int(inliers.sum()), len(good)


def back_crop(reference: np.ndarray, matte: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    """The head and shoulders of the matted back panel (the top 40 % of the figure, centred on the head),
    at the clip's size."""
    img = np.asarray(reference)[..., :3]
    if matte is not None:
        m = cv2.resize(normalize_mask(matte), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
    else:
        m = np.abs(img.astype(np.int16) - _GREY).max(2) > 10
    ys, xs = np.nonzero(m)
    if len(ys) < 100:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    top, bottom = ys.min(), ys.max()
    h = max(int(0.40 * (bottom - top)), 16)
    head = m[top:top + max(int(0.12 * (bottom - top)), 1)]
    cx = int(np.nonzero(head)[1].mean())
    w = int(h * width / height)
    x0, y0 = max(cx - w // 2, 0), max(top - int(0.04 * h), 0)
    crop = img[y0:y0 + h, x0:x0 + w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


# -- face_detail_views ----------------------------------------------------------------------------------------

@register_step("face_detail_views")
class FaceDetailViewsStep(Step):
    """The splat rendered along a constant-speed path about the head or the hair: a detailer clip's control.

    inputs:  {"splat_path": str — the splat the control video is rendered from,
              "cameras": [Camera] the dataset's, "anchor_frame_index": int — the photograph's among them,
              "joints": the refit body's (raw), "world_from_raw": its frame — the head is the face clip's centre,
              "front_image": the sheet's front panel, "front_matte": its matte — the anchor frame and the face
                             reference (`reference: photo`),
              "reference_image"?: the matted back panel (`reference: back`), "back_matte"?: its matte,
              "label_splat_path"?: a splat carrying seg_label, for the hair's framing and loss weights
                                   (default splat_path)}
    outputs: {"images": [BGR] the control video, "masks": [float HxW] its VACE masks (1 = paint, 0 at the
              anchor frame), "cameras": [Camera], "image_names": [str], "reference_image": BGR,
              "anchor_index": int or None, "weights": [float HxW] or None — the hair's loss weights
                                                                           (`hair_weights`),
              "splat_path": str — the splat as it was (`keep_copy`, else splat_path), "stats": {...}}

    The view sphere is centred on the head (`centre: head`, the MHR head joint) or the hair (`centre: hair`,
    `hair_centre`); a view is (yaw, pitch) about it at the photograph's distance, yaw from the photograph's
    direction, pitch from its elevation (`elevation: photo`) or from level (`elevation: level`). The focal
    length is `focal_per_m` times that distance, so the head is the same size in every subject's clip.
    """

    PARAMS = (
        Param("path", str, "lissajous", "View path: lissajous or helix", choices=("lissajous", "helix")),
        Param("frames", int, 81, "Frames in the clip (4k+1 for the video VAE)", minimum=5),
        Param("width", int, 480, "Clip width", minimum=16),
        Param("height", int, 832, "Clip height", minimum=16),
        Param("yaw_deg", float, 45.0, "Lissajous yaw amplitude, in degrees", minimum=0.0, maximum=180.0),
        Param("pitch_deg", float, 20.0, "Pitch amplitude (lissajous) or half-range (helix), in degrees", minimum=0.0, maximum=80.0),
        Param("pitch_offset_deg", float, 0.0, "Pitch the path is centred on, in degrees", minimum=-80.0, maximum=80.0),
        Param("ratio", float, 2.0, "Lissajous: vertical periods per horizontal one", minimum=0.5),
        Param("turns", float, 1.0, "Helix: turns about the subject", minimum=0.1),
        Param("elevation", str, "photo", "Pitch measured from the photograph's elevation or from level",
              choices=("photo", "level")),
        Param("centre", str, "head", "What the path circles: the head joint or the hair", choices=("head", "hair")),
        Param("focal_per_m", float, 1326.0, "Focal length in pixels per metre of camera distance (framing)", minimum=1.0),
        Param("anchor", bool, True, "Put the photograph, warped into the start pose, in the first frame (inactive)"),
        Param("reference", str, "photo", "VACE reference: the photograph's face at the start pose, or the back panel",
              choices=("photo", "back")),
        Param("hair_weights", bool, False, "Publish loss weights that keep this clip to the hair"),
        Param("hair_dilate_px", int, 4, "Dilate the hair weights by this many pixels", minimum=0),
        Param("name", str, "face", "Frame name prefix (unique per clip)"),
        Param("keep_copy", str, "", "Copy the splat here first and publish the copy as splat_path"),
        Param("debug_dir", str, "", "Write the path, the anchor registration and a contact sheet here"),
        Param("render_path", str, "brush-splat-render", "The rasteriser binary", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.splat_scene import SplatScene

        from .splat import _rasterize

        splat_path = Path(inputs["splat_path"])
        if params["keep_copy"]:
            dst = Path(params["keep_copy"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(splat_path, dst)
            logger.info("face_detail_views: kept the splat as it was at %s", dst)
            splat_path = dst
        cameras = list(inputs["cameras"])
        index = inputs.get("anchor_frame_index")
        if index is None or not 0 <= int(index) < len(cameras):
            raise ValueError(f"face_detail_views: anchor_frame_index {index!r} is not a frame of the {len(cameras)} cameras")
        photo_cam = cameras[int(index)]
        W, H, N = int(params["width"]), int(params["height"]), int(params["frames"])
        scene = SplatScene.from_ply(str(splat_path))
        label_path = inputs.get("label_splat_path")
        label_scene = SplatScene.from_ply(str(label_path)) if label_path and Path(label_path) != splat_path else scene

        joints = to_world(np.asarray(inputs["joints"], np.float64), inputs["world_from_raw"])
        target = joints[MHR_HEAD_JOINT]
        centre_from = "head joint"
        if params["centre"] == "hair":
            hc = hair_centre(label_scene)
            if hc is None:
                logger.warning("face_detail_views: no labelled hair in the splat; the path circles the head joint")
            else:
                target, centre_from = hc, "hair box"

        p0 = np.asarray(photo_cam.position, np.float64).reshape(3)
        v0 = p0 - target
        dist = float(np.linalg.norm(v0))
        yaw0 = float(np.degrees(np.arctan2(v0[0], v0[2])))
        pitch0 = float(np.degrees(np.arcsin(v0[1] / dist)))
        u, v, start, speed = path_offsets(params["path"], N, params["yaw_deg"], params["pitch_deg"],
                                          params["pitch_offset_deg"], params["ratio"], params["turns"])
        pitch = (pitch0 + v) if params["elevation"] == "photo" else v
        fx = params["focal_per_m"] * dist
        c2ws = [_look_at_cv(target + dist * _directions(np.array(yaw0 + a), np.array(b)), target) for a, b in zip(u, pitch)]
        clip_cams = [_camera(m, fx, W, H) for m in c2ws]
        names = [f"{params['name']}_{i + 1:05d}.png" for i in range(N)]
        images, _alphas = _rasterize(scene=scene, splat_path=str(splat_path), cameras=clip_cams, image_names=names,
                                     width=W, height=H, bg_color=(0.5, 0.5, 0.5), render_path=params["render_path"])
        images = [np.ascontiguousarray(im[..., :3]) for im in images]
        masks = [np.ones((H, W), np.float32) for _ in range(N)]
        stats: Dict[str, Any] = {"path": params["path"], "deg_per_frame": round(float(speed), 3), "centre": centre_from,
                                 "target": [round(float(x), 4) for x in target], "distance_m": round(dist, 4),
                                 "photo_yaw_deg": round(yaw0, 2), "photo_pitch_deg": round(pitch0, 2),
                                 "yaw_range": [round(float(u.min()), 1), round(float(u.max()), 1)],
                                 "pitch_range": [round(float(pitch.min()), 1), round(float(pitch.max()), 1)]}

        # The photograph at the start pose: the front panel onto the splat's own render at the photograph's
        # camera (a similarity), then from that camera into the start pose's — the same camera centre, so a
        # rotation homography.
        photo = photo_matte = None
        anchor_index: Optional[int] = None
        if params["anchor"] or params["reference"] == "photo":
            pw, ph = int(photo_cam.width), int(photo_cam.height)
            render, _ = _rasterize(scene=scene, splat_path=str(splat_path), cameras=[photo_cam], image_names=["photo"],
                                   width=pw, height=ph, bg_color=(0.5, 0.5, 0.5), render_path=params["render_path"])
            front = np.asarray(inputs["front_image"])[..., :3]
            reg = register_photo(front, render[0][..., :3])
            if reg is None:
                logger.warning("face_detail_views: the front panel does not register onto the splat's render at the "
                               "photograph's camera; no anchor frame, and the reference is the panel as it is")
                stats["anchor"] = "not registered"
            else:
                A, inliers, matches = reg
                stats["anchor"] = {"inliers": inliers, "matches": matches, "scale": round(float(np.hypot(*A[0, :2])), 4)}
                m0 = c2ws[start]
                Hs = (_intrinsics(clip_cams[start]) @ m0[:3, :3].T @ _cv_c2w(photo_cam)[:3, :3]
                      @ np.linalg.inv(_intrinsics(photo_cam)) @ np.vstack([A, [0.0, 0.0, 1.0]]))
                warped = cv2.warpPerspective(front, Hs, (W, H), flags=cv2.INTER_CUBIC, borderValue=(_GREY,) * 3)
                matte = cv2.resize(normalize_mask(inputs["front_matte"]), (front.shape[1], front.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
                photo_matte = cv2.warpPerspective(matte, Hs, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
                photo = (warped * photo_matte[..., None] + _GREY * (1 - photo_matte[..., None])).astype(np.uint8)
                if params["anchor"]:
                    if abs(u[start]) > 1e-6 or abs(v[start]) > 1e-6:
                        raise ValueError("face_detail_views: `anchor` needs a path through the photograph's pose "
                                         "(elevation: photo, pitch 0 at the start)")
                    anchor_index = start
                    images[start] = photo
                    masks[start] = np.zeros((H, W), np.float32)

        if params["reference"] == "photo":
            if photo is None:
                front = np.asarray(inputs["front_image"])[..., :3]
                matte = cv2.resize(normalize_mask(inputs["front_matte"]), (front.shape[1], front.shape[0]))
                reference = (front * matte[..., None] + _GREY * (1 - matte[..., None])).astype(np.uint8)
            else:
                reference = photo
        else:
            if inputs.get("reference_image") is None:
                raise ValueError("face_detail_views: reference: back needs 'reference_image'")
            reference = back_crop(inputs["reference_image"], inputs.get("back_matte"), W, H)

        weights = None
        if params["hair_weights"]:
            hair, _ = _rasterize(scene=class_mask_scene(label_scene, HAIR_LABEL), splat_path=None, cameras=clip_cams,
                                 image_names=names, width=W, height=H, bg_color=(0.0, 0.0, 0.0),
                                 render_path=params["render_path"])
            k = 2 * int(params["hair_dilate_px"]) + 1
            weights = []
            for im in hair:
                m = (cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY) > 64).astype(np.uint8) * 255
                m = cv2.GaussianBlur(cv2.dilate(m, np.ones((k, k), np.uint8)), (0, 0), 2)
                weights.append(m.astype(np.float32) / 255.0)
            stats["hair_coverage"] = round(float(np.mean([w.mean() for w in weights])), 4)

        logger.info("face_detail_views: %s %d frames about the %s at %.2f m, %.2f deg/frame, yaw %+.0f..%+.0f, "
                    "pitch %+.0f..%+.0f (photograph at %+.0f), anchor %s, reference %s", params["path"], N, centre_from,
                    dist, speed, u.min(), u.max(), pitch.min(), pitch.max(), pitch0,
                    "frame %d" % anchor_index if anchor_index is not None else "none", params["reference"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            (debug / f"{params['name']}_path.json").write_text(json.dumps(
                {**stats, "yaw": u.round(3).tolist(), "pitch": pitch.round(3).tolist()}, indent=1))
            picks = list(range(0, N, max(N // 8, 1)))[:9]
            sheet = np.concatenate([cv2.resize(images[i], (W // 2, H // 2)) for i in picks] + [cv2.resize(reference, (W // 2, H // 2))], 1)
            cv2.imwrite(str(debug / f"{params['name']}_control.jpg"), sheet)
        return {"images": images, "masks": masks, "cameras": clip_cams, "image_names": names,
                "reference_image": reference, "anchor_index": anchor_index, "weights": weights,
                "splat_path": str(splat_path), "stats": stats}


# -- the photograph's eyes, 4x ----------------------------------------------------------------------------------

def _landmark_px(face_landmarks: Dict[str, Any]) -> np.ndarray:
    w, h = face_landmarks["image_size"]
    return np.asarray(face_landmarks["landmarks"], np.float64)[:, :2] * [w, h]


@register_step("face_detail_photo_crop")
class FaceDetailPhotoCropStep(Step):
    """The photograph's face, square and padded, at the size one SeedVR2 frame takes.

    inputs:  {"image": the front panel (BGR), "face_landmarks": detect_face_landmarks' output on it}
    outputs: {"images": [the crop at `size`], "cameras": [a placeholder Camera of that size] (seedvr2 rescales
              cameras with its frames), "crop": {"x", "y", "side"} in the panel's pixels}
    """

    PARAMS = (
        Param("size", int, 320, "Side of the crop handed to the upscale, in pixels", minimum=16),
        Param("padding", float, 1.6, "Crop side as a multiple of the landmarks' extent", minimum=1.0),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.camera import Camera

        img = np.asarray(inputs["image"])[..., :3]
        lm = _landmark_px(inputs["face_landmarks"])
        (x0, y0), (x1, y1) = lm.min(0), lm.max(0)
        side = int(np.ceil(max(x1 - x0, y1 - y0) * params["padding"] / 8) * 8)
        bx, by = int(round((x0 + x1) / 2 - side / 2)), int(round((y0 + y1) / 2 - side / 2))
        crop = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_REFLECT)[by + side:by + 2 * side, bx + side:bx + 2 * side]
        size = int(params["size"])
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC if side < size else cv2.INTER_AREA)
        logger.info("face_detail_photo_crop: the face, %d px square at (%d, %d), handed to the upscale at %d px", side, bx, by, size)
        return {"images": [crop], "cameras": [Camera(focal_length=(1000.0, 1000.0), image_size=(size, size))],
                "crop": {"x": bx, "y": by, "side": side}}


@register_step("face_detail_photo_merge")
class FaceDetailPhotoMergeStep(Step):
    """The photograph at `scale` x with the upscaled face feathered in: the eye texture `paste_eyes` takes.

    inputs:  {"image", "face_landmarks", "focal_length": the photograph's (detect_face, reconstruct_body),
              "crop": face_detail_photo_crop's, "images": [the upscaled crop]}
    outputs: {"image": the photograph `scale` x, "face_landmarks": the same landmarks (they are normalised) at
              that image size, "focal_length": `scale` x — fit_head_per_view's anchor camera is then the same
              camera at `scale` x the pixels}
    """

    PARAMS = (
        Param("scale", int, 4, "How many times the photograph's pixels", minimum=1, maximum=8),
        Param("feather", float, 0.12, "Feather the crop's border over this fraction of its side", minimum=0.0, maximum=0.5),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        img = np.asarray(inputs["image"])[..., :3]
        h, w = img.shape[:2]
        k = int(params["scale"])
        crop = inputs["crop"]
        bx, by, side = int(crop["x"]), int(crop["y"]), int(crop["side"])
        sr = cv2.resize(np.asarray(inputs["images"][0])[..., :3], (side * k, side * k), interpolation=cv2.INTER_AREA)
        big = cv2.resize(img, (w * k, h * k), interpolation=cv2.INTER_CUBIC)
        fe = np.ones((side * k, side * k), np.float32)
        r = int(params["feather"] * side * k)
        for i in range(r):
            f = (i + 1) / (r + 1)
            fe[i, :] *= f
            fe[-1 - i, :] *= f
            fe[:, i] *= f
            fe[:, -1 - i] *= f
        X0, Y0 = bx * k, by * k
        xs0, ys0 = max(X0, 0), max(Y0, 0)
        xs1, ys1 = min(X0 + side * k, w * k), min(Y0 + side * k, h * k)
        sub = (slice(ys0 - Y0, ys1 - Y0), slice(xs0 - X0, xs1 - X0))
        f = fe[sub][..., None]
        big[ys0:ys1, xs0:xs1] = (big[ys0:ys1, xs0:xs1] * (1 - f) + sr[sub] * f).astype(np.uint8)
        landmarks = dict(inputs["face_landmarks"])
        landmarks["image_size"] = (w * k, h * k)
        logger.info("face_detail_photo_merge: the photograph %dx%d -> %dx%d with the upscaled face", w, h, w * k, h * k)
        return {"image": big, "face_landmarks": landmarks, "focal_length": float(inputs["focal_length"]) * k}


# -- face_detail_clip ---------------------------------------------------------------------------------------------

_LID_RINGS = {"image_left": LID_RING_IMAGE_LEFT, "image_right": LID_RING_IMAGE_RIGHT}


def eye_weights(shape: Tuple[int, int], landmarks: np.ndarray, sides: Sequence[str], dilate_px: int) -> np.ndarray:
    """1 everywhere but inside each named eye's lid contour, dilated by `dilate_px`."""
    w = np.ones(shape, np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    for side in sides:
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(landmarks)[list(_LID_RINGS[side]), :2] * 4).astype(np.int32)], 255, shift=2)
        w[cv2.dilate(m, k) > 0] = 0.0
    return w


@register_step("face_detail_clip")
class FaceDetailClipStep(Step):
    """A denoised detailer clip as training views.

    inputs:  {"images": [BGR] the clip (eyes pasted), "masks": [float] its matte (rmbg), "cameras", "image_names",
              "anchor_index"?: the photograph's frame, "face_views"?: detect_face_views' landmarks on the clip,
              "eye_stats"?: paste_eyes' per-view record, "weights"?: loss weights of the clip's own (the hair's)}
    outputs: {"clip": {"images", "masks", "cameras", "image_names", "weights": [float HxW or None]}}

    With `weights` wired they are the clip's loss weights (the hair clip). Otherwise every eye the paste did not
    draw gets zero weight, in every frame but the photograph's: those are the video model's eyes, which follow
    the camera, and the eyes the rest of the views agree on must not be pulled towards them.
    """

    PARAMS = (
        Param("eye_dilate_px", int, 4, "Grow the unpasted eyes' zero-weight region by this many pixels", minimum=0),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images, names = list(inputs["images"]), list(inputs["image_names"])
        masks, cameras = list(inputs["masks"]), list(inputs["cameras"])
        if not (len(images) == len(names) == len(masks) == len(cameras)):
            raise ValueError(f"face_detail_clip: images {len(images)}, image_names {len(names)}, masks {len(masks)}, "
                             f"cameras {len(cameras)} disagree")
        weights: List[Optional[np.ndarray]]
        if inputs.get("weights") is not None:
            weights = list(inputs["weights"])
            how = "own loss weights"
        else:
            anchor = inputs.get("anchor_index")
            landmarks = (inputs.get("face_views") or {}).get("landmarks_px", {})
            drawn = (inputs.get("eye_stats") or {}).get("views", {})
            if not isinstance(drawn, dict):
                drawn = {}
            weights, masked = [], 0
            for i, (img, name) in enumerate(zip(images, names)):
                lm = landmarks.get(name)
                if i == anchor or lm is None:
                    weights.append(None)
                    continue
                pasted = drawn.get(name, {})
                sides = [s for s in _LID_RINGS if not (isinstance(pasted.get(s), dict) and "skipped" not in pasted[s])]
                weights.append(eye_weights(img.shape[:2], lm, sides, int(params["eye_dilate_px"])) if sides else None)
                masked += len(sides)
            how = f"{masked} unpasted eyes at zero weight"
        logger.info("face_detail_clip: %d frames, %s", len(images), how)
        return {"clip": {"images": images, "masks": masks, "cameras": cameras, "image_names": names, "weights": weights}}


# -- face_detail_dataset ------------------------------------------------------------------------------------------

@register_step("face_detail_dataset")
class FaceDetailDatasetStep(Step):
    """The deliverable's training views with the detailer clips after them.

    inputs:  {"images", "image_names", "cameras", "masks": the dataset's, "labels"?: its class maps,
              "clip_a", "clip_b"?, "clip_c"?: face_detail_clip's}
    outputs: {"images", "image_names", "cameras", "masks", "weights": per view a map or None (no sidecar =
              weight 1), "labels": per view a map or None (the clips do not vote), or None without labels}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images, names = list(inputs["images"]), list(inputs["image_names"])
        cameras, masks = list(inputs["cameras"]), list(inputs["masks"])
        labels = inputs.get("labels")
        labels = list(labels) if labels is not None and len(labels) else None
        weights: List[Optional[np.ndarray]] = [None] * len(images)
        clips = [inputs[k] for k in ("clip_a", "clip_b", "clip_c") if inputs.get(k) is not None]
        for clip in clips:
            clash = set(clip["image_names"]) & set(names)
            if clash:
                raise ValueError(f"face_detail_dataset: frame names already taken: {sorted(clash)[:3]}")
            images += clip["images"]
            names += clip["image_names"]
            cameras += clip["cameras"]
            masks += clip["masks"]
            weights += list(clip.get("weights") or [None] * len(clip["images"]))
            if labels is not None:
                labels += [None] * len(clip["images"])
        logger.info("face_detail_dataset: %d views (%d clip frames from %d clips), %d with loss weights", len(images),
                    sum(len(c["images"]) for c in clips), len(clips), sum(w is not None for w in weights))
        return {"images": images, "image_names": names, "cameras": cameras, "masks": masks,
                "weights": weights if any(w is not None for w in weights) else None, "labels": labels}


# -- face_detail_rig ----------------------------------------------------------------------------------------------

@register_step("face_detail_rig")
class FaceDetailRigStep(Step):
    """The rig's per-view face deltas (rig v3) on the detailer clips' frames, constrained.

    inputs:  {"body_rig": build_body_rig's rig, "head_fit_a", "head_fit_b"?: fit_head_per_view on each clip,
              "image_names_a", "image_names_b"?: each clip's frames, in order}
    outputs: {"body_rig": the rig with "view_deltas" on the clip frames only, "face_rig_stats": {...}}

    Per clip as build_face_rig does it (fit minus canonical at the rig's vertices, gaps interpolated or held,
    the face core only), then: the mean over every clip frame with a delta removed, a temporal median over
    +-`smooth` frames within each clip, and each vertex capped at `cap_mm`. The dataset's own views keep none,
    including any build_face_rig gave them: the deliverable already is their canonical face, and the clips are
    what this rig is for.
    """

    PARAMS = (
        Param("face_motion_cm", float, 0.5, "Face core = vertices expressions move at least this many cm", minimum=0.01),
        Param("face_fade_cm", float, 3.0, "Fade the deltas to zero this far outside the face core", minimum=0.1),
        Param("gap", int, 4, "Interpolate across runs of at most this many unfitted frames", minimum=0),
        Param("hold", int, 3, "An unfitted frame within this many frames of a fitted one holds its deltas", minimum=0),
        Param("smooth", int, 3, "Temporal median over this many frames either side, within a clip", minimum=0),
        Param("cap_mm", float, 10.0, "Cap every vertex's delta at this many mm", minimum=0.1),
        Param("debug_dir", str, "", "Write face_detail_rig.json here"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from ..body_rig import assign_view_deltas, face_only_weights

        rig = inputs.get("body_rig")
        if not isinstance(rig, dict) or "verts" not in rig or rig.get("vertex_index") is None:
            raise ValueError("face_detail_rig needs 'body_rig' from build_body_rig (with vertex_index)")
        idx = np.asarray(rig["vertex_index"], np.int64)
        rig_verts = np.asarray(rig["verts"], np.float64)
        clips = [(inputs[f], list(inputs[n])) for f, n in (("head_fit_a", "image_names_a"), ("head_fit_b", "image_names_b"))
                 if inputs.get(f) is not None and inputs.get(n) is not None]
        if not clips:
            raise ValueError("face_detail_rig: no clip head fit")
        weights = None
        blocks, hows, spans = [], [], []
        for fit, names in clips:
            verts0 = np.asarray(fit["verts0_world"], np.float64)
            if idx.max() >= len(verts0) or np.abs(verts0[idx] - rig_verts).max() > 1e-3:
                raise ValueError("face_detail_rig: the rig and the clip's head fit are from different bodies")
            if weights is None:
                motion = np.asarray(fit["expression_motion"], np.float64)
                weights = face_only_weights(rig_verts, verts0[motion > params["face_motion_cm"]], params["face_fade_cm"] / 100.0)
            at = {n: i for i, n in enumerate(names)}
            fitted = {at[n]: (np.asarray(fit["verts_world"][k], np.float64)[idx] - rig_verts).astype(np.float32)
                      for k, n in enumerate(fit["names"]) if n in at}
            if fitted:
                d, how = assign_view_deltas(fitted, len(names), len(idx), gap=int(params["gap"]), hold=int(params["hold"]))
            else:
                d, how = np.zeros((len(names), len(idx), 3), np.float32), ["zero"] * len(names)
            spans.append((len(hows), len(hows) + len(names)))
            blocks.append(d * weights[None, :, None])
            hows += how
        D = np.concatenate(blocks)
        names_all = [n for _, names in clips for n in names]
        has = np.array([h != "zero" for h in hows])
        if not has.any():
            raise ValueError("face_detail_rig: the head fit covers no clip frame")
        mean_mm = float(np.linalg.norm(D[has].mean(0), axis=1).mean() * 1000)
        D[has] -= D[has].mean(0)
        S = np.zeros_like(D)
        r = int(params["smooth"])
        for lo, hi in spans:
            for i in range(lo, hi):
                if not has[i]:
                    continue
                win = [j for j in range(max(lo, i - r), min(hi, i + r + 1)) if has[j]]
                S[i] = np.median(D[win], 0)
        mag = np.linalg.norm(S, axis=2, keepdims=True) * 1000
        S = np.where(mag > params["cap_mm"], S * (params["cap_mm"] / np.maximum(mag, 1e-9)), S).astype(np.float32)
        per_view = np.linalg.norm(S, axis=2).mean(1) * 1000
        stats = {"clip_frames": len(names_all), "with_deltas": int(has.sum()),
                 "removed_mean_mm": round(mean_mm, 2),
                 "mean_delta_mm_p50": round(float(np.median(per_view[has])), 2),
                 "mean_delta_mm_p90": round(float(np.percentile(per_view[has], 90)), 2),
                 "cap_mm": params["cap_mm"], "smooth": r}
        logger.info("face_detail_rig: %d of %d clip frames carry face deltas (shared %.1f mm removed), mean displacement "
                    "p50 %.1f / p90 %.1f mm; the dataset's views none", stats["with_deltas"], len(names_all), mean_mm,
                    stats["mean_delta_mm_p50"], stats["mean_delta_mm_p90"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "face_detail_rig.json").write_text(json.dumps({**stats, "frames_how": dict(zip(names_all, hows))}, indent=1))
        out = {k: v for k, v in rig.items() if k != "view_deltas"}
        out["view_deltas"] = {n: S[i] for i, n in enumerate(names_all) if has[i]}
        return {"body_rig": out, "face_rig_stats": stats}
