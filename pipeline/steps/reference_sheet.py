"""split_reference_sheet — the front/back sheet cut into its two panels, or a
single frontal photo passed through whole.

The from-a-photo path usually does not start from a photo of the subject:
it starts from one square image a diffusion model generated, showing the
subject twice — facing front on the left, from behind on the right.
Everything downstream wants one panel or the other, never the sheet:

    left  (front) -> sam3d_body        reconstruct the body from a front view
                  -> generate_firstlast warp THIS photo onto the anchor camera
                  -> rmbg              foreground mask of the same photo
    right (back)  -> wan22_vace_denoise's `reference_image`

That last one is the part that is easy to get backwards, and it changed
recently. The back panel is what VACE conditions on now *because* the anchor
is injected: the front view already reaches the diffusion pass as a real
photograph at the anchor frame (steps/anchor_stub.py), so spending the
reference slot on the front as well tells the model nothing it cannot see,
while the back — the one view no input image and no mesh render can supply
truthfully — is exactly what the other 80 frames need.

**A single frontal photo** (2026-09-11) takes the other branch: the whole
image is the front, and there is no back — `back` comes out None, so pass 1
runs with no reference at all, on the injected photograph alone, and
`pick_rear_view` (steps/reference_view.py) fills the slot afterwards with
the pass-1 frame that looks at the subject from behind. Which branch an
upload takes is the `layout` param: `sheet`, `single`, or `auto`.

**`auto` counts the figures.** MediaPipe's ObjectDetector (EfficientDet-Lite0,
COCO, `person` only — the same `mediapipe` package `detect_face_landmarks`
runs, a 14 MB tflite file fetched into the same cache, ~15 ms on the CPU)
is run on the whole image; two figures on opposite sides of the centre
line are a sheet, one figure is a photo. Measured 2026-09-11 before it was
wired: 27/27 sheets in ~/datasets/cyberpunk2 gave exactly two boxes at
0.87-0.96 and NOTHING else above a 0.2 threshold; cyber_6f's own sheet 2
(0.91/0.90); four single photos (cyber_6f's warped anchor, girl_9_16, a
front panel alone, a BACK panel alone) 1 each at 0.90-0.95, with every
false positive at <= 0.52 and no taller than 6% of the frame. The two
figure tests in `classify_layout` (score and height) come from those
numbers. What was NOT good enough: MediaPipe's pose landmarker on the
whole image misses the back figure of a real sheet (1 of 2 on cyber_6f),
the blaze face detector finds two faces on one person (retired in
86c4f8e), SAM-3D-Body has no detector wired at all (steps/sam3d_body.py:
the whole image is the box), and rmbg gives a matte, not a count.

**cyber_6f is NOT a golden for this step.** Its recorded `reference.png` is
1440x1280, i.e. both 720x1280 panels still joined, from before the anchor
injection change; `workflows/api/denoise.json` wires that whole sheet into
`WanVaceToVideo.reference_image`. Anything in this repo that reasons from
those files is describing the older convention. The panels there are also
9:16 rather than the 1:2 that halving a square yields, so that sheet had
been reframed somewhere in the manual stage as well — the split here is a
plain cut down the middle, which is what the current flow does.

There is no ComfyUI node to port: the split lives in the interactive graph
that produces an `initial/` directory, built from stock ComfyUI image nodes,
and none of the checked-in API JSONs cover it (they all begin from an
`initial/` that already exists).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

# EfficientDet-Lite0, the smallest of MediaPipe's object detectors; the
# float32 build so the CPU delegate runs it as-is. Same cache directory and
# the same fetch-then-rename as face_landmarker.task (steps/face_landmarks.py
# `_ensure_model`), and models.py's `mediapipe` entry prefetches both.
DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
)
DETECTOR_MODEL_NAME = "efficientdet_lite0.tflite"

#: (score, x0, y0, x1, y1) — pixel coordinates, one per detected person.
Box = Tuple[float, float, float, float, float]

LAYOUTS = ("auto", "sheet", "single")

#: A person box shorter than this fraction of the frame is not a figure.
#: Every false positive measured (see the module docstring) was a sliver
#: at most 6% of the frame tall; every real figure was at least 83%.
MIN_FIGURE_HEIGHT = 0.4


def count_figures(image: np.ndarray, model_path: str, min_score: float) -> List[Box]:
    """Every `person` the detector finds in a BGR image, scoring `min_score` or more.

    Whole image, no crop: the panels of a sheet fill their halves and a
    photo fills its frame, which is the scale the detector was trained
    at. Sorted best first.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        score_threshold=float(min_score),
        category_allowlist=["person"],
        max_results=8,
    )
    rgb = cv2.cvtColor(np.ascontiguousarray(image[:, :, :3]), cv2.COLOR_BGR2RGB)
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector = vision.ObjectDetector.create_from_options(options)
    try:
        result = detector.detect(frame)
    finally:
        detector.close()
    boxes: List[Box] = []
    for detection in result.detections:
        box = detection.bounding_box
        boxes.append((
            float(detection.categories[0].score),
            float(box.origin_x), float(box.origin_y),
            float(box.origin_x + box.width), float(box.origin_y + box.height),
        ))
    boxes.sort(reverse=True)
    return boxes


def classify_layout(
    boxes: Sequence[Box], width: int, height: int, min_score: float,
) -> str:
    """`"sheet"` or `"single"` from the person boxes of a `width` x `height` image.

    A figure is a box scoring `min_score` or more and at least
    MIN_FIGURE_HEIGHT of the frame tall. Two or more figures whose two
    best sit on opposite sides of the centre line make a sheet; exactly
    one makes a single photo. Anything else — nobody, two figures on the
    same side, or a sheet in a portrait frame (two portrait panels side
    by side are never taller than wide) — is refused rather than guessed
    at, because the two branches send the pixels to different places and
    a wrong guess fits a mesh to half a person or to two.
    """
    figures = [
        b for b in boxes
        if b[0] >= min_score and (b[4] - b[2]) >= MIN_FIGURE_HEIGHT * height
    ]
    if not figures:
        raise ValueError(
            f"split_reference_sheet: no figure found in the {width}x{height} "
            f"input (person boxes: {_describe(boxes)}). Set the workflow's "
            f"`input_layout` to `sheet` or `single` to say which it is."
        )
    if len(figures) == 1:
        return "single"
    centre = width / 2.0
    first, second = figures[0], figures[1]
    mid_first = (first[1] + first[3]) / 2.0
    mid_second = (second[1] + second[3]) / 2.0
    if (mid_first < centre) == (mid_second < centre):
        raise ValueError(
            f"split_reference_sheet: {len(figures)} figures in the "
            f"{width}x{height} input, but the two clearest are on the same "
            f"side of the centre line (person boxes: {_describe(boxes)}) — "
            f"not a front/back sheet. Set `input_layout` explicitly."
        )
    if width < height:
        raise ValueError(
            f"split_reference_sheet: two figures side by side in a "
            f"{width}x{height} (portrait) input (person boxes: "
            f"{_describe(boxes)}). A front/back sheet is never taller than "
            f"it is wide; set `input_layout` explicitly."
        )
    return "sheet"


def _describe(boxes: Sequence[Box]) -> str:
    return ", ".join(
        f"{s:.2f}@[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]" for s, x0, y0, x1, y1 in boxes
    ) or "none"


@register_step("split_reference_sheet")
class SplitReferenceSheetStep(Step):
    """Halve a front/back sheet down the middle, or pass a photo through.

    inputs: {"sheet": np.ndarray HxWx3 BGR uint8 — the upload}
    outputs: {"front": np.ndarray, "back": np.ndarray | None,
              "layout": "sheet" | "single"}

    `sheet`: a sheet is always at least as wide as it is tall (two portrait
    panels side by side), so a portrait input is a single photo handed in
    where a sheet was promised — that raises rather than quietly cutting a
    person in half. `single`: the whole image is the front and `back` is
    None. `auto`: the module docstring's figure count decides.
    """

    PARAMS = (
        Param("layout", str, "auto",
              "What the upload is: `sheet` (front and back panels side by "
              "side), `single` (one frontal photo — no back view, so the "
              "first denoise runs without a reference and the second takes "
              "its rear view from the first's output), or `auto` (count "
              "the figures: two on opposite sides of the centre are a sheet)",
              choices=LAYOUTS),
        Param("front_side", str, "left", "Which half of the sheet holds the front view",
              choices=("left", "right")),
        Param("min_score", float, 0.6,
              "Detector confidence a `person` box needs to count as a figure "
              "under `layout: auto`. Real figures measured 0.87-0.96, the "
              "false positives at most 0.52 (and never tall enough anyway)",
              minimum=0.0, maximum=1.0, advanced=True),
    )

    # The detector, as a callable the tests replace: (image, model_path,
    # min_score) -> boxes. Looked up on the instance so a test can set it
    # without a model file, and so the model is fetched only under `auto`.
    count_figures: Callable[[np.ndarray, str, float], List[Box]] = staticmethod(count_figures)

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        sheet = np.asarray(inputs["sheet"])
        height, width = sheet.shape[:2]

        layout = params["layout"]
        if layout not in LAYOUTS:
            raise ValueError(
                f"split_reference_sheet: layout must be one of {LAYOUTS}, got {layout!r}"
            )
        front_side = params["front_side"]
        if front_side not in ("left", "right"):
            raise ValueError(
                f"split_reference_sheet: front_side must be 'left' or 'right', "
                f"got {front_side!r}"
            )

        if layout == "auto":
            layout = self._detect(sheet, params)

        if layout == "single":
            logger.info("single %dx%d photo: the whole image is the front, no back view", width, height)
            return {"front": np.ascontiguousarray(sheet), "back": None, "layout": "single"}

        if width < height:
            raise ValueError(
                f"split_reference_sheet got a {width}x{height} (portrait) image "
                f"under `layout: sheet`. It expects the two-panel front/back "
                f"sheet a diffusion model generates — the subject facing front "
                f"beside the same subject from behind — which is never taller "
                f"than it is wide. Halving this one would cut a single subject "
                f"down the middle; a single photo wants `layout: single`."
            )

        half = width // 2
        if width % 2:
            # The odd centre column belongs to neither panel; dropping it
            # keeps both halves the same size, which matters because the
            # front half's dimensions set the framing generate_firstlast
            # warps from.
            logger.debug("sheet width %d is odd; dropping the centre column", width)
        # Contiguous copies, not slice views: cv2 rejects a non-contiguous
        # array outright in several of the calls these halves go on to
        # (warpPerspective's src among them), and a view would also keep
        # the whole sheet alive behind each panel.
        left = np.ascontiguousarray(sheet[:, :half])
        right = np.ascontiguousarray(sheet[:, width - half:])

        front, back = (left, right) if front_side == "left" else (right, left)
        logger.info(
            "split %dx%d sheet into %dx%d front (%s) + back",
            width, height, front.shape[1], front.shape[0], front_side,
        )
        return {"front": front, "back": back, "layout": "sheet"}

    def _detect(self, image: np.ndarray, params: Dict[str, Any]) -> str:
        from .face_landmarks import _ensure_model, _model_path

        height, width = image.shape[:2]
        min_score = float(params["min_score"])
        model_path = str(_ensure_model(DETECTOR_MODEL_URL, _model_path(DETECTOR_MODEL_NAME)))
        # Asked for everything down to 0.2 so the log shows what the
        # threshold rejected, not only what it kept.
        boxes = self.count_figures(image, model_path, min(min_score, 0.2))
        layout = classify_layout(boxes, width, height, min_score)
        logger.info(
            "layout auto -> %s (%dx%d, person boxes: %s, min_score %.2f)",
            layout, width, height, _describe(boxes), min_score,
        )
        return layout
