"""MediaPipe face landmarks, and the face-only region they enclose.

Two steps. `detect_face_landmarks` finds the landmarks;
`face_landmark_mask` turns them into the box a face crop is cut to and the
matte the face splat is built on.

Native port of `nodes/face_landmarks_node.py`. Originally this fed
`steps/render.py`'s optional `face_landmarks` param, which draws face
keypoints on the skeleton render modes so a diffusion pass gets facial
structure to condition on, not just a body skeleton. **That is no longer
what it is for.** A Gaussian splat of the subject's real face replaced the
landmark dots (2026-08-29), and `render_initial_views` takes no
`face_landmarks` input any more — dots under the splat show through its
soft silhouette. The landmarks came back (2026-08-30) for the geometry
alone: Sapiens2's `parts: face` is Goliath class 3, `Face_Neck`, so the
face and the neck are one class and "just the face" cannot be selected,
only intersected. See `FaceLandmarkMaskStep`.

The detection is **crop-first**, and the crop comes from the body mesh.
MediaPipe's FaceLandmarker is trained on face-filling images; on this
project's inputs — full-body shots where the face is a few percent of the
frame — a whole-image pass either finds nothing or returns badly-placed
landmarks. So the step crops to the face and landmarks the crop, mapping
the crop-space landmarks back to full-image normalized coordinates.

Where the crop comes from changed on 2026-09-11. Until then a second
MediaPipe model, the `blaze_face_short_range` FaceDetector, located the
face boxes (the ComfyUI-Body2COLMAP reference node's approach, from
body2colmap's `tools/extract_face_landmarks.py`), and the most frontal of
several landmarked crops won. It was unreliable — spurious second faces on
a single subject, and misses — and by the time this step runs the
pipeline already knows where the head is: `sam3d_body` has fitted the
body, and its head keypoints projected through SAM-3D-Body's own camera
centre the crop (`mesh_head_box`). The raw fit's head misses the
photograph's by ~10 px on cyber2_6f, nothing against a crop padded by
three quarters of the head's extent on every side. `mesh_output` is
optional only so the step still runs without a mesh (a test, a
head-and-shoulders input); then, or when the crop yields no landmarks,
the fallback is FaceLandmarker on the whole frame.

Runs on CPU; no GPU or pod needed. The landmarker's `.task` file is
downloaded on first use into the models volume (see `_model_path`).

Output is the raw MediaPipe format — an (N, 3) array of normalized
coordinates, 468 or 478 points depending on whether iris landmarks are
present. Conversion to OpenPose Face 70 happens in `render`, via
body2colmap's `FaceLandmarkIngest.from_mediapipe`, because that is where
the image size needed to unnormalize them is known.

VERIFIED locally (CPU, no pod) on 2026-09-11 against the detector it
replaced: on cyber_6f's anchor photo and on a recorded SAM-3D-Body fit of
an 867x1552 portrait, the mesh-cropped landmarks agree with the
detector-cropped ones to well under a pixel (see the commit), and the
detector had reported two faces on the single-subject portrait. See
tests/test_face_landmarks.py.

One operational note: mediapipe 1.0.1 on macOS arm64 aborted the process
once (SIGABRT inside `TensorsToDetectionsCalculator::Open()` via
`DrishtiMetalHelper`, "Check failed: service_ Service is unavailable") on
the very first invocation after downloading the models, then worked on
every run since. It is an abort rather than an exception, so it cannot be
caught in-process; the end-to-end test therefore runs detection in a
subprocess so a recurrence degrades to a skip instead of taking the whole
test run down.
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_MODEL_NAME = "face_landmarker.task"


def _model_path(filename: str) -> Path:
    """Where a MediaPipe model file is cached.

    Resolved lazily rather than at import time: every step module is
    imported in every isolated venv (see pipeline/steps/__init__.py), and
    a module-level call would mkdir on the volume just to answer "which
    step class is this", in processes that will never touch MediaPipe.

    On the volume, not in ~/.cache — the latter is inside the container on
    a pod, so these re-download on every restart.
    """
    from ..paths import models_dir

    return models_dir() / "mediapipe" / filename


@register_step("detect_face_landmarks")
class DetectFaceLandmarksStep(Step):
    """Detect face landmarks in a single image.

    inputs:  {"image": np.ndarray BGR uint8 — the photograph,
              "mesh_output": dict, optional — sam3d_body's outputs for THIS
                             photograph (keypoints_3d, cam_t, focal_length;
                             image_size when published). Its head
                             keypoints, projected, centre the crop the
                             landmarker sees. None means the whole frame}
    params:  min_detection_confidence (float, default 0.3),
             crop_padding (float, default 0.75 — margin around the projected
             head keypoints, as a fraction of their extent, before
             landmarking the crop)
    outputs: {"face_landmarks": {"source": "mediapipe",
              "landmarks": np.ndarray (N, 3) normalized,
              "image_size": (width, height)}}

    Raises RuntimeError when neither the head crop nor the whole-image
    landmarker finds a face — a silent empty result would produce a
    skeleton render with no face and no indication why.
    """

    PARAMS = (
        Param("min_detection_confidence", float, 0.3,
              "MediaPipe's face detection / presence confidence floor",
              minimum=0.0, maximum=1.0, advanced=True),
        Param("crop_padding", float, 0.75,
              "How far to pad the mesh head's projected extent before landmarking "
              "the crop, as a fraction of that extent on each side",
              minimum=0.0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "mediapipe is required for face landmark detection. "
                "Install with: pip install mediapipe"
            ) from exc

        image = inputs["image"]
        min_confidence = params["min_detection_confidence"]

        # This pipeline speaks cv2 BGR; MediaPipe wants SRGB.
        bgr = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        height, width = rgb.shape[:2]

        logger.info(
            "detect_face_landmarks: %dx%d image, confidence=%.2f",
            width, height, min_confidence,
        )

        mesh = inputs.get("mesh_output")
        crop_box = None
        if mesh is not None:
            crop_box = mesh_head_box(mesh, width, height, params["crop_padding"])
        else:
            logger.info("detect_face_landmarks: no mesh_output; landmarking the whole frame")

        landmarker_path = str(_ensure_model(LANDMARKER_MODEL_URL, _model_path(LANDMARKER_MODEL_NAME)))
        lm_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=landmarker_path),
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(lm_options)
        try:
            landmarks = _detect(
                rgb=rgb, width=width, height=height, landmarker=landmarker,
                crop_box=crop_box, mp=mp,
            )
        finally:
            landmarker.close()

        logger.info("detect_face_landmarks: %d points", landmarks.shape[0])
        return {
            "face_landmarks": {
                "source": "mediapipe",
                "landmarks": landmarks,
                "image_size": (width, height),
            }
        }


#: MHR70's head keypoints: nose, left eye, right eye, left ear, right ear.
_MHR_HEAD_KEYPOINTS = (0, 1, 2, 3, 4)


def mesh_head_box(mesh: Dict[str, Any], width: int, height: int,
                  padding: float) -> Tuple[int, int, int, int]:
    """A square crop around the mesh head on this photograph.

    Centred on the projected head keypoints (nose, eyes, ears) through
    SAM-3D-Body's camera — `pred_keypoints_3d + pred_cam_t` with
    `focal_length` and the principal point at the frame's centre, which is
    the frame that focal is measured in — with a side of
    `(1 + 2 * padding)` times the keypoints' extent, ear to ear on a face
    seen from the front. A frontal face is ~1.3 ear-spans tall and the
    keypoints sit at its middle, so 0.75 leaves ~0.6 of a span around it:
    room for the raw fit's ~10 px miss and a head the model made 15% the
    wrong size, and still a face-filling crop, which is what the landmarker
    wants. Measured against the detector's crop on an 867x1552 portrait
    (111 px span): 0.5 / 0.75 / 1.0 agree to 0.9 / 0.6 / 0.7 px mean, the
    landmarker's own crop-to-crop jitter.

    Not `head_fit.head_crop_box`: that takes every vertex above the neck
    joint, which is the shoulders as well as the head — the right extent
    for a render nothing has to find a face in, and on an 867x1552
    portrait a 730x487 crop for a 110 px face.

    When the mesh publishes the size it was fitted on it must be this
    frame's: a resized copy would be self-consistent and silently wrong,
    the same check `face_pointmap_splat` makes.
    """
    try:
        focal = float(mesh["focal_length"])
        cam_t = np.asarray(mesh["cam_t"], np.float64).reshape(3)
        keypoints = np.asarray(mesh["keypoints_3d"], np.float64) + cam_t
        head = keypoints[list(_MHR_HEAD_KEYPOINTS)]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise KeyError(
            "detect_face_landmarks: 'mesh_output' must carry keypoints_3d "
            "(MHR70), cam_t and focal_length, as sam3d_body writes them"
        ) from exc
    fitted = mesh.get("image_size")
    if fitted is not None and tuple(int(v) for v in fitted) != (width, height):
        raise ValueError(
            f"detect_face_landmarks: the mesh was fitted on a "
            f"{fitted[0]}x{fitted[1]} frame and this image is {width}x{height}; "
            "the head box would land on the wrong pixels"
        )
    z = np.clip(head[:, 2], 1e-6, None)
    px = np.stack([focal * head[:, 0] / z + width / 2.0,
                   focal * head[:, 1] / z + height / 2.0], 1)
    lo, hi = px.min(0), px.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    if not np.isfinite(span) or span < 4.0:
        raise ValueError(
            f"detect_face_landmarks: the mesh head spans {span:.1f} px on a "
            f"{width}x{height} frame — is the mesh from this photograph?"
        )
    centre = 0.5 * (lo + hi)
    half = 0.5 * span * (1.0 + 2.0 * padding)
    box = (int(np.floor(centre[0] - half)), int(np.floor(centre[1] - half)),
           int(np.ceil(centre[0] + half)), int(np.ceil(centre[1] + half)))
    logger.info("detect_face_landmarks: mesh head spans %.0f px; crop %s at padding %.2f",
                span, box, padding)
    return box


@register_step("face_landmark_mask")
class FaceLandmarkMaskStep(Step):
    """The face on its own — the region MediaPipe's landmarks enclose.

    Sapiens2's `parts: face` is Goliath class 3, `Face_Neck`: the face and
    the neck are ONE class, so "just the face" cannot be asked for by
    naming a different class. It has to come from geometry, and the
    landmarks are the geometry already available — the same source
    body2colmap's `fit_face_to_skeleton` has always used, from back when
    the face reached the render as landmark dots rather than a splat.

    It intersects the hull with a segmentation matte already computed on the
    crop, and returns that matte with everything outside the face zeroed.
    `crop_info` is what maps the landmarks — which are in the FULL frame's
    normalized coordinates — onto the crop's pixel grid; it is the same
    relation `FacePointmapSplatStep._source_intrinsics` uses to move the
    camera the other way.

    **It does NOT size the crop, and must not.** The obvious companion
    change — cut the crop to the face too, since a Face_Neck box runs down
    the throat and a crop sized to it spends much of Sapiens2's 1024x768 on
    neck — was built, measured and reverted. It flattens the face. Measured
    on cyber2_6f, relief over the identical face pixels, with the nose's
    depth ahead of the face's outer edge in brackets:

        Face_Neck crop (261x348), Face_Neck matte    224.5 mm  (+99.2)
        Face_Neck crop,           hull matte         200.9 mm  (+99.5)
        face crop (208x277),      hull matte         144.2 mm  (+22.9)

    The matte costs nothing — the nose still stands 99.5 mm proud. The CROP
    costs four fifths of it. Sapiens2 upsamples whatever it is given to
    1024x768, so a 261 px crop is already being magnified 2.9x and a 208 px
    one 3.7x: tightening the box adds no real pixels of face, only
    interpolation, and the pointmap head answers a softer, more magnified
    input with a flatter face. The premise that a tighter crop buys
    resolution is simply wrong once the crop is smaller than the network's
    input, which it always is here.

    Both halves are wanted. The hull alone would take background with it
    wherever the head is turned and the convex boundary cuts past the
    cheek; the seg matte alone cannot tell a jaw from a throat. The
    intersection is the face, with the seg's soft silhouette kept where the
    two boundaries coincide.

    **The edge is feathered, not cut.** `pointmap_splat` keeps the matte's
    sub-threshold values as the Gaussians' opacity (see `soft_alpha`), so a
    hard-edged hull would hand it a rim of fully opaque primitives and the
    face would read as a sticker. `feather_frac` falls the hull off over a
    few pixels instead, and `soft_alpha` then treats that boundary exactly
    as it treats a matte's own.

    inputs:  {"face_landmarks": dict — detect_face_landmarks' output,
              "mask": HxW float32 [0,1] — the matte to intersect,
              "crop_info": dict, optional — crop_to_box's, when `mask` is
                           on the crop's grid rather than the full frame's}
    outputs: {"mask": HxW float32 [0,1] — the face region, feathered, and
                      multiplied into the input matte}

    Not wired into `render`: the landmarks are here for the mask and the
    box, and nothing else. `render_initial_views` deliberately takes no
    `face_landmarks` input any more — dots drawn under the splat show
    through its soft silhouette.
    """

    PARAMS = (
        Param("dilate_frac", float, 0.06,
              "Grow the landmark hull by this fraction of the face's larger "
              "side before intersecting. MediaPipe's outline sits on the skin "
              "at the jaw and the hairline; a little margin keeps the "
              "transition inside the support region rather than on its edge",
              minimum=0.0),
        Param("feather_frac", float, 0.03,
              "Fall the hull off to zero over this fraction of the face's "
              "larger side. 0 cuts hard, which hands pointmap_splat a rim of "
              "opaque Gaussians — see the class docstring", minimum=0.0),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from scipy import ndimage

        landmarks = inputs["face_landmarks"]
        source = landmarks.get("source")
        if source != "mediapipe":
            raise ValueError(
                f"face_landmark_mask: unsupported landmark source {source!r}. "
                f"Supported: 'mediapipe' (see DetectFaceLandmarksStep)."
            )
        points = np.asarray(landmarks["landmarks"], dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3:
            raise ValueError(
                f"face_landmark_mask: expected an (N, 3) landmark array with "
                f"N >= 3, got {points.shape}"
            )
        width_full, height_full = (float(v) for v in landmarks["image_size"])

        # Normalized (full-frame) -> full-frame pixels.
        full = np.stack([points[:, 0] * width_full, points[:, 1] * height_full], 1)

        if inputs.get("mask") is None:
            raise KeyError(
                "face_landmark_mask requires 'mask' — the segmentation matte "
                "to intersect the landmark hull with. The hull alone takes "
                "background with it wherever the head is turned and its "
                "convex boundary cuts past the cheek."
            )
        matte = np.asarray(inputs["mask"], dtype=np.float32)
        if matte.ndim != 2:
            raise ValueError(
                f"face_landmark_mask: 'mask' must be a single-channel matte, "
                f"got shape {matte.shape}"
            )
        shape = matte.shape
        info = inputs.get("crop_info")
        local = self._to_crop(full, info, shape) if info is not None else full

        region = self._hull(local, shape, params)
        covered = float((region * (matte >= 0.5)).sum())
        inside = float((matte >= 0.5).sum())
        logger.info(
            "face_landmark_mask: hull keeps %.0f%% of the %d px matte "
            "(the rest is neck, hair and ears)",
            100.0 * covered / max(inside, 1.0), int(inside),
        )
        region = region * matte

        if float(region.sum()) < 64.0:
            raise ValueError(
                "face_landmark_mask: the landmark hull and the matte barely "
                "overlap. Either they are not from the same photo, or "
                "`crop_info` does not describe the crop the matte was "
                "computed on."
            )
        logger.info("face_landmark_mask: face region %d px on a %dx%d grid",
                    int((region >= 0.5).sum()), shape[1], shape[0])
        return {"mask": region.astype(np.float32)}

    @staticmethod
    def _to_crop(full: np.ndarray, info: Dict[str, Any],
                 shape: Tuple[int, int]) -> np.ndarray:
        """Full-frame landmark pixels -> the crop's pixel grid.

        `u_full = x0 + r * u_c` with `r` the crop's resize factor, so
        `u_c = (u_full - x0) / r`. Exactly the relation
        `_source_intrinsics` inverts to carry the camera the other way; a
        crop emitted at native resolution (which is all crop_to_box makes)
        has r = 1 and this is a translation.
        """
        try:
            x0, y0, x1, y1 = (float(v) for v in info["box"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"face_landmark_mask: 'crop_info' must carry 'box' "
                f"(x0, y0, x1, y1 in full-image pixels), as crop_to_box "
                f"writes it. Got {info!r}"
            ) from exc
        height, width = shape
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"face_landmark_mask: degenerate crop box {info['box']}")
        ratio_x, ratio_y = (x1 - x0) / width, (y1 - y0) / height
        if abs(ratio_x - ratio_y) > 1e-3 * max(ratio_x, ratio_y):
            raise ValueError(
                f"face_landmark_mask: crop box {info['box']} against a "
                f"{width}x{height} matte implies a non-uniform resize "
                f"({ratio_x:.4f} vs {ratio_y:.4f})"
            )
        ratio = 0.5 * (ratio_x + ratio_y)
        return np.stack([(full[:, 0] - x0) / ratio, (full[:, 1] - y0) / ratio], 1)

    @staticmethod
    def _hull(points: np.ndarray, shape: Tuple[int, int],
              params: Dict[str, Any]) -> np.ndarray:
        """Convex hull of the landmarks, dilated then feathered, as [0,1].

        Convex rather than MediaPipe's own FACEMESH_FACE_OVAL contour: the
        oval's index list is a mediapipe-version detail, while a hull over
        whatever points arrived works for both the 468- and 478-point
        outputs the detector returns. The difference is a slight
        convexification at the temples, well inside `dilate_frac`.
        """
        from scipy import ndimage

        height, width = shape
        hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(filled, np.round(hull).astype(np.int32), 1)

        span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
        dilate_px = int(round(params["dilate_frac"] * span))
        if dilate_px > 0:
            filled = ndimage.binary_dilation(
                filled.astype(bool), iterations=dilate_px).astype(np.uint8)

        feather_px = params["feather_frac"] * span
        if feather_px <= 0.0:
            return filled.astype(np.float32)
        # Distance INTO the region, so the falloff eats inward from the
        # boundary and the interior stays at 1 — the same shape a matte has.
        distance = ndimage.distance_transform_edt(filled)
        return np.clip(distance / feather_px, 0.0, 1.0).astype(np.float32)


def _ensure_model(url: str, path: Path) -> Path:
    """Download a MediaPipe model file unless it is already cached.

    Into a uniquely-named sibling and then `replace`, for the reason
    refine_cameras' `ensure_onnx_model` does the same: the prefetch and a
    worker can be here at once, and a download written straight to the
    cached path is visible to the other one — and to every later run —
    while it is still half a file. `replace` is atomic, so what appears at
    `path` is either absent or whole.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("detect_face_landmarks: downloading %s", path.name)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".",
                                suffix=".partial")
    os.close(fd)
    scratch = Path(name)
    try:
        urllib.request.urlretrieve(url, str(scratch))
        scratch.replace(path)
    finally:
        scratch.unlink(missing_ok=True)
    return path


def _face_to_array(face_landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)


def _face_to_array_from_crop(
    face_landmarks, crop_w: int, crop_h: int, x1: int, y1: int,
    full_w: int, full_h: int,
) -> np.ndarray:
    """Map crop-normalized landmarks back to full-image normalized coords.

    z is left alone: MediaPipe's depth is relative and roughly on the same
    scale as x, so rescaling it against the crop would make it inconsistent
    with a full-image detection.
    """
    return np.array(
        [
            [(lm.x * crop_w + x1) / full_w, (lm.y * crop_h + y1) / full_h, lm.z]
            for lm in face_landmarks
        ],
        dtype=np.float32,
    )


def _detect(*, rgb, width, height, landmarker, crop_box, mp) -> np.ndarray:
    """Crop-first detection. Returns (N, 3) full-image normalized landmarks.

    `crop_box` is (x0, y0, x1, y1) in this frame's pixels — the mesh
    head, padded — and None landmarks the whole frame directly, which is
    what a render that already IS a head crop wants (`map_face_to_mesh`).
    When the crop yields nothing the whole frame is tried before giving up.
    """
    if crop_box is not None:
        x0, y0, x1, y1 = (int(v) for v in crop_box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(
                f"detect_face_landmarks: head crop {crop_box} is empty on a "
                f"{width}x{height} frame — is the mesh from this photograph?"
            )
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1, :])
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        )
        if result.face_landmarks:
            crop_h, crop_w = crop.shape[:2]
            logger.info(
                "detect_face_landmarks: from crop %dx%d at offset %d,%d",
                crop_w, crop_h, x0, y0,
            )
            return _face_to_array_from_crop(
                result.face_landmarks[0], crop_w, crop_h, x0, y0, width, height
            )
        logger.warning(
            "detect_face_landmarks: no face in the head crop %s; trying the "
            "whole frame", (x0, y0, x1, y1),
        )

    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if result.face_landmarks:
        return _face_to_array(result.face_landmarks[0])

    raise RuntimeError(
        f"No face detected in image ({width}x{height}) by the landmarker, "
        + ("in the head crop or " if crop_box is not None else "")
        + "on the whole frame. Ensure the image contains a visible face."
    )
