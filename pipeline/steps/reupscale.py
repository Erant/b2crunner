"""reupscale_inputs — the deliverable splat's own renders, ready for a SeedVR2 pass and a retraining.

Why (measured 2026-10-02 on 9ddb62, plain trainer flags, scratchpad study in
memory `render_sr_retrain`): the trained splat holds about half the sharpness
of the frames it was fitted to (band-limited s1 7.7 against 13.9; the head
11.0 against 21.5). The frames' extra detail disagrees between views and
averages away. Rendering the splat back out gives frames that agree with
each other by construction; SeedVR2 then adds detail to clean, consistent
inputs, and a splat retrained on them kept it: s1 12.6, head 19.5 (+63 %,
+77 %), the same at cameras between the training views (12.7 / 19.0), for
0.6 dB of fidelity to the real frames on top of the 1.4 dB that retraining
on renders costs at all. Retraining on the renders bicubic-upscaled scored
6.7 (the control); 1080 renders carry little the 720 ones do not, so the
render is 720x1280 whatever the dataset's resolution.

This step: the final cameras rescaled to the render size (a pure intrinsics
scale; the poses do not move), the splat rendered over 0.5 grey at them, and
the frames' mattes (bilinear) and class maps (nearest) brought to the size
SeedVR2 will hand back, so the retraining's sidecars line up with its
frames. The splat it starts from is copied aside first: the retraining
exports over it.

Batched upscales (`inbetween` > 0, for a card that holds a SeedVR2 batch):
SeedVR2 treats a batch as video and expects little motion between its
frames, and the training cameras are a helix ~2 degrees apart. So
`inbetween` cameras are interpolated between every pair of neighbouring
ones (the frames are in path order across the passes: 41 + 81 + 40 along
one helix), the whole sequence is upscaled, and `reupscale_keep` keeps the
frames at the training cameras for the retraining. With `inbetween` equal
to the batch size minus 1, a batch spans one interval of the original path.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def interpolate_cameras(cameras: List[Any], inbetween: int) -> List[Any]:
    """`inbetween` cameras between every pair of neighbours (position linear, rotation slerp, intrinsics linear);
    the originals land at every (inbetween + 1)-th index."""
    from body2colmap.camera import Camera

    if inbetween <= 0 or len(cameras) < 2:
        return list(cameras)
    out = []
    for a, b in zip(cameras[:-1], cameras[1:]):
        out.append(a)
        qa = _quat_from_matrix(np.asarray(a.rotation, np.float64).reshape(3, 3))
        qb = _quat_from_matrix(np.asarray(b.rotation, np.float64).reshape(3, 3))
        for k in range(1, inbetween + 1):
            t = k / (inbetween + 1)
            lerp = lambda x, y: (1 - t) * float(x) + t * float(y)
            pos = (1 - t) * np.asarray(a.position, np.float64) + t * np.asarray(b.position, np.float64)
            out.append(Camera(focal_length=(lerp(a.fx, b.fx), lerp(a.fy, b.fy)), image_size=(int(a.width), int(a.height)),
                              principal_point=(lerp(a.cx, b.cx), lerp(a.cy, b.cy)), position=pos.astype(np.float32),
                              rotation=_matrix_from_quat(_slerp(qa, qb, t)).astype(np.float32)))
    out.append(cameras[-1])
    return out


def scale_cameras(cameras: List[Any], width: int, height: int) -> List[Any]:
    """The same views at another pixel size: fx, cx by the width ratio, fy, cy by the height ratio."""
    from body2colmap.camera import Camera

    out = []
    for cam in cameras:
        sx, sy = width / float(cam.width), height / float(cam.height)
        out.append(Camera(focal_length=(float(cam.fx) * sx, float(cam.fy) * sy), image_size=(width, height),
                          principal_point=(float(cam.cx) * sx, float(cam.cy) * sy),
                          position=np.asarray(cam.position, np.float32), rotation=np.asarray(cam.rotation, np.float32)))
    return out


@register_step("reupscale_inputs")
class ReupscaleInputsStep(Step):
    """Render the trained splat at its cameras for a SeedVR2 pass, with sidecars at the upscaled size.

    inputs: {"splat_path": str, "cameras": List[Camera],
             "masks": Optional[List[np.ndarray]] — the frames' mattes, any size,
             "labels": Optional[List[np.ndarray]] uint8 class ids, any size}
    outputs: {"images": List[np.ndarray] BGR uint8 at render_width x render_height (for seedvr2), the
                        training cameras and `inbetween` interpolated ones between each pair of them,
              "cameras": List[Camera] the same, at the render size (seedvr2 rescales them with its frames),
              "keep_every": int — the training cameras are every keep_every-th of them (reupscale_keep),
              "masks": List[np.ndarray] float32 at target_width x target_height, one per training camera,
              "labels": Optional[List[np.ndarray]] uint8 at the target size, one per training camera}
    """

    PARAMS = (
        Param("render_width", int, 720, "Width the splat is rendered at", minimum=1),
        Param("render_height", int, 1280, "Height the splat is rendered at", minimum=1),
        Param("target_width", int, 1080, "Width SeedVR2 hands back (its `resolution` is the short edge)", minimum=1),
        Param("target_height", int, 1920, "Height SeedVR2 hands back", minimum=1),
        Param("inbetween", int, 0,
              "Cameras interpolated between every pair of neighbouring ones, for a batched upscale that expects "
              "little motion between frames; 0 renders the training cameras alone", minimum=0),
        Param("keep_copy", str, "", "Copy the splat here before the retraining exports over it"),
        Param("render_path", str, "brush-splat-render", "The rasteriser binary", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        splat_path = Path(inputs["splat_path"])
        cameras = list(inputs["cameras"])
        masks: Optional[List[np.ndarray]] = inputs.get("masks")
        labels: Optional[List[np.ndarray]] = inputs.get("labels")
        if not cameras:
            raise ValueError("reupscale_inputs: no cameras")
        for name, maps in (("masks", masks), ("labels", labels)):
            if maps is not None and len(maps) != len(cameras):
                raise ValueError(f"reupscale_inputs: {len(maps)} {name} for {len(cameras)} cameras")
        rw, rh = int(params["render_width"]), int(params["render_height"])
        tw, th = int(params["target_width"]), int(params["target_height"])

        if params["keep_copy"]:
            dst = Path(params["keep_copy"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(splat_path, dst)
            logger.info("reupscale_inputs: kept the splat as it was at %s", dst)

        from body2colmap.splat_scene import SplatScene

        from .splat import _rasterize

        inbetween = int(params["inbetween"])
        render_cams = scale_cameras(interpolate_cameras(cameras, inbetween), rw, rh)
        images, alphas = _rasterize(
            scene=SplatScene.from_ply(str(splat_path)), splat_path=str(splat_path), cameras=render_cams,
            image_names=[f"frame_{i + 1:05d}_.png" for i in range(len(render_cams))],
            width=rw, height=rh, bg_color=(0.5, 0.5, 0.5), render_path=params["render_path"])

        # The mattes the retraining needs: the frames' own where wired (what the deliverable was fitted to),
        # else the render's alpha. Bilinear up to the size SeedVR2 hands back.
        source = masks if masks is not None else alphas
        out_masks = [np.clip(cv2.resize(normalize_mask(m), (tw, th), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
                     .astype(np.float32) for m in source]
        out_labels = None
        if labels is not None:
            out_labels = [cv2.resize(np.asarray(l, np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) for l in labels]
        logger.info("reupscale_inputs: %d views of %s (%d training cameras, %d between each pair) rendered at %dx%d "
                    "for the upscale; mattes%s at %dx%d", len(images), splat_path.name, len(cameras), inbetween, rw, rh,
                    " and class maps" if out_labels is not None else "", tw, th)
        return {"images": images, "cameras": render_cams, "keep_every": inbetween + 1, "masks": out_masks,
                "labels": out_labels}


@register_step("reupscale_keep")
class ReupscaleKeepStep(Step):
    """Keep every `keep_every`-th upscaled frame and its camera: the training cameras out of a densified path.

    inputs: {"images": List[np.ndarray], "cameras": List[Camera], "keep_every": int}
    outputs: {"images", "cameras"} — the kept ones, in order
    """

    PARAMS = ()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images, cameras = list(inputs["images"]), list(inputs["cameras"])
        every = int(inputs["keep_every"])
        if len(images) != len(cameras):
            raise ValueError(f"reupscale_keep: {len(images)} images for {len(cameras)} cameras")
        if every < 1 or (len(images) - 1) % every != 0:
            raise ValueError(f"reupscale_keep: {len(images)} frames are not training cameras every {every}")
        logger.info("reupscale_keep: %d of %d upscaled frames (every %d)", len(images[::every]), len(images), every)
        return {"images": images[::every], "cameras": cameras[::every]}
