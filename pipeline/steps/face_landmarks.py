"""detect_face_landmarks — MediaPipe face landmarks for the skeleton overlay.

Native port of `nodes/face_landmarks_node.py`. Feeds `steps/render.py`'s
optional `face_landmarks` param, which draws face keypoints on the
skeleton render modes so a diffusion pass gets facial structure to
condition on, not just a body skeleton.

The detection pipeline is **crop-first**, which is the reason this is more
than a single library call. The ComfyUI-Body2COLMAP reference node
(nodes/face_landmarks_node.py, itself lifted from body2colmap's
`tools/extract_face_landmarks.py`) runs FaceLandmarker on the full image
first and only crops when that finds nothing — but for this project's
inputs, which are full-body shots where the face is a small fraction of
the frame, a whole-image FaceLandmarker pass either finds nothing or
returns badly-placed landmarks. MediaPipe's FaceLandmarker is trained on
face-filling images. So:

1. Run FaceDetector (blaze_face_short_range) to locate face bounding
   boxes, crop each with padding, and run FaceLandmarker on the crop.
   Crop-space landmarks are mapped back to full-image normalized coords.
2. Only if the detector finds nothing — or no crop yields landmarks — fall
   back to FaceLandmarker on the whole frame (a head-and-shoulders input,
   or an aspect ratio the short-range detector was not trained for).

With several faces found (a front/back reference sheet gives two), the
most frontal wins, scored by the z-component of the cross product of the
inter-eye and nose-to-chin vectors.

Runs on CPU; no GPU or pod needed. The two `.task`/`.tflite` model files
are downloaded on first use to ~/.cache/body2colmap, the same location and
filenames the ComfyUI node and body2colmap's own tool use, so an existing
cache is picked up rather than re-downloaded.

Output is the raw MediaPipe format — an (N, 3) array of normalized
coordinates, 468 or 478 points depending on whether iris landmarks are
present. Conversion to OpenPose Face 70 happens in `render`, via
body2colmap's `FaceLandmarkIngest.from_mediapipe`, because that is where
the image size needed to unnormalize them is known.

VERIFIED locally against cyber_6f's real reference photos (CPU, no pod):
478 landmarks from the single-subject anchor photo, and — on the
two-panel front/back reference sheet — the frontality scoring correctly
selects the front-facing subject in the left panel over the back-of-head
in the right one. Both resolve through the detector-and-crop path, since
the face is a small part of a full-body frame. See
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
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
LANDMARKER_MODEL_NAME = "face_landmarker.task"
DETECTOR_MODEL_NAME = "blaze_face_short_range.tflite"


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

# MediaPipe landmark indices used for frontality scoring.
_MP_RIGHT_EYE_OUTER = 33
_MP_LEFT_EYE_OUTER = 263
_MP_NOSE_BRIDGE = 168
_MP_CHIN = 152


@register_step("detect_face_landmarks")
class DetectFaceLandmarksStep(Step):
    """Detect face landmarks in a single image.

    inputs:  {"image": np.ndarray BGR uint8} — typically
             dataset.reference_image or dataset.anchor_image
    params:  min_detection_confidence (float, default 0.3),
             crop_padding (float, default 0.5 — padding around a detected
             face box, as a fraction of face size, before landmarking it)
    outputs: {"face_landmarks": {"source": "mediapipe",
              "landmarks": np.ndarray (N, 3) normalized,
              "image_size": (width, height)}}

    Raises RuntimeError when neither the short-range detector nor the
    whole-image landmarker finds a face — a silent empty result would
    produce a skeleton render with no face and no indication why.
    """

    PARAMS = (
        Param("min_detection_confidence", float, 0.3,
              "MediaPipe face-detector confidence floor", minimum=0.0, maximum=1.0,
              advanced=True),
        Param("crop_padding", float, 0.5,
              "How far to pad the detected face box before the second-stage crop, "
              "as a fraction of the box", minimum=0.0, advanced=True),
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
        crop_padding = params["crop_padding"]

        # This pipeline speaks cv2 BGR; MediaPipe wants SRGB.
        bgr = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        height, width = rgb.shape[:2]

        logger.info(
            "detect_face_landmarks: %dx%d image, confidence=%.2f",
            width, height, min_confidence,
        )

        landmarker_path = str(_ensure_model(LANDMARKER_MODEL_URL, _model_path(LANDMARKER_MODEL_NAME)))
        detector_path = str(_ensure_model(DETECTOR_MODEL_URL, _model_path(DETECTOR_MODEL_NAME)))

        lm_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=landmarker_path),
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            num_faces=10,
        )
        landmarker = vision.FaceLandmarker.create_from_options(lm_options)
        try:
            landmarks = _detect(
                rgb=rgb,
                width=width,
                height=height,
                landmarker=landmarker,
                detector_path=detector_path,
                min_confidence=min_confidence,
                crop_padding=crop_padding,
                mp=mp,
                vision=vision,
                python=python,
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


def _ensure_model(url: str, path: Path) -> Path:
    """Download a MediaPipe model file unless it is already cached."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("detect_face_landmarks: downloading %s", path.name)
    urllib.request.urlretrieve(url, str(path))
    return path


def _frontality_score(face_landmarks) -> float:
    """How frontal a face is: 0 = profile, 1 = straight on.

    The face normal is the cross product of the inter-eye vector and the
    nose-bridge-to-chin vector; its z-component (toward the camera),
    normalized, is the score.
    """
    def _xyz(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    eye_vec = _xyz(_MP_LEFT_EYE_OUTER) - _xyz(_MP_RIGHT_EYE_OUTER)
    vert_vec = _xyz(_MP_NOSE_BRIDGE) - _xyz(_MP_CHIN)
    normal = np.cross(eye_vec, vert_vec)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-10:
        return 0.0
    return abs(float(normal[2])) / norm


def _pick_best_face(face_landmarks_list) -> Tuple[Any, int]:
    """Most frontal face out of several, with its index."""
    if len(face_landmarks_list) == 1:
        return face_landmarks_list[0], 0

    best_score, best_idx = -1.0, 0
    for i, face in enumerate(face_landmarks_list):
        score = _frontality_score(face)
        if score > best_score:
            best_score, best_idx = score, i
    return face_landmarks_list[best_idx], best_idx


def _crop_to_face(rgb: np.ndarray, bbox, padding: float):
    """Crop to a face bounding box, padded by a fraction of its size."""
    height, width = rgb.shape[:2]
    x, y, w, h = bbox
    pad_x, pad_y = int(w * padding), int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return np.ascontiguousarray(rgb[y1:y2, x1:x2, :]), x1, y1


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


def _detect(
    *, rgb, width, height, landmarker, detector_path, min_confidence,
    crop_padding, mp, vision, python,
) -> np.ndarray:
    """Crop-first detection. Returns (N, 3) full-image normalized landmarks.

    Stage 1 is the short-range detector plus a padded crop per face box,
    landmarked individually — MediaPipe's FaceLandmarker needs a
    face-filling image and this project's inputs never are. Stage 2, only
    when stage 1 comes up empty, is FaceLandmarker on the whole frame.
    """
    # Stage 1: detect -> pad -> crop -> landmark each crop.
    det_options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=detector_path),
        min_detection_confidence=min_confidence,
    )
    detector = vision.FaceDetector.create_from_options(det_options)
    try:
        detections = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        bboxes = [
            (d.bounding_box.origin_x, d.bounding_box.origin_y,
             d.bounding_box.width, d.bounding_box.height)
            for d in detections.detections
        ]
    finally:
        detector.close()

    if bboxes:
        logger.info(
            "detect_face_landmarks: %d face box(es), cropping with padding=%.2f",
            len(bboxes), crop_padding,
        )
        candidates: List[Tuple[Any, np.ndarray, int, int]] = []
        for bbox in bboxes:
            crop, x1, y1 = _crop_to_face(rgb, bbox, padding=crop_padding)
            crop_result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            )
            if crop_result.face_landmarks:
                candidates.append((crop_result.face_landmarks[0], crop, x1, y1))

        if candidates:
            _, best_idx = _pick_best_face([c[0] for c in candidates])
            face, crop, x1, y1 = candidates[best_idx]
            crop_h, crop_w = crop.shape[:2]
            if len(candidates) > 1:
                logger.info(
                    "detect_face_landmarks: chose face #%d of %d (frontality %.2f)",
                    best_idx + 1, len(candidates), _frontality_score(face),
                )
            logger.info(
                "detect_face_landmarks: from crop %dx%d at offset %d,%d",
                crop_w, crop_h, x1, y1,
            )
            return _face_to_array_from_crop(
                face, crop_w, crop_h, x1, y1, width, height
            )

        logger.info(
            "detect_face_landmarks: detector found %d box(es) but no crop "
            "yielded landmarks; trying the whole frame", len(bboxes),
        )
    else:
        logger.info(
            "detect_face_landmarks: detector found no face; trying the whole frame"
        )

    # Stage 2 (fallback): the whole frame. A head-and-shoulders crop, or an
    # aspect ratio the short-range detector was not trained for.
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if result.face_landmarks:
        face, idx = _pick_best_face(result.face_landmarks)
        if len(result.face_landmarks) > 1:
            logger.info(
                "detect_face_landmarks: %d faces on the full frame, chose #%d "
                "(frontality %.2f)",
                len(result.face_landmarks), idx + 1, _frontality_score(face),
            )
        return _face_to_array(face)

    raise RuntimeError(
        f"No face detected in image ({width}x{height}) by the short-range "
        "detector or the whole-frame landmarker. Ensure the image contains a "
        "visible face."
    )
