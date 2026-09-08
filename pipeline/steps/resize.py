"""resize_batch — put a batch of frames, or their masks, on another pixel grid.

One small step, and it exists for one reason: **diffusers does not resize a
VACE control video to the `width` x `height` it is asked for.**
`WanVACEPipeline.preprocess_conditions` scales the control video down,
aspect-preserving, until it fits under the target AREA, then floors each
side to a multiple of 16 — so a 720x1280 batch handed to
`wan22_vace_denoise` at `width: 480, height: 832` is denoised at 464x832,
and the reference image, which `_fit_reference` cropped to the 480x832 the
params named, is then letterboxed into that 464-wide frame between white
bars. Resizing the batch to exactly 480x832 here first is what makes the
pass run at the size its params say, with nothing for diffusers to fit.

The re-outline branch of fast_helical_native.yaml uses it twice: once to
take the stage-1 control video down to 480x832 for the extra denoise, and
once to bring rmbg's mattes of that denoise's output back up to the render
size so `render` can draw the outline from them. Both resizes are PLAIN
(`cv2.resize` to the target, no letterbox, no crop), and that is the point:
720x1280 -> 480x832 is anisotropic by 2.6% (0.667 across, 0.65 down), and
the same plain resize back cancels it exactly, so frame i's matte lands on
frame i's original pixel grid. The denoiser sees a figure 2.6% squatter than
the drawing; the outline drawn from its matte does not.

Interpolation is `auto` unless asked otherwise: INTER_AREA when shrinking
(a box filter — the right thing for photographs and drawings going down),
INTER_LINEAR when enlarging (a soft matte stays soft). A per-frame VACE flag
mask — every pixel 0.0 or every pixel 1.0 — survives either exactly.

Masks come back as float32 in [0, 1] whatever form they arrived in
(pipeline/masks.py's `normalize_mask`), images as the uint8 BGR they were.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import REQUIRED, Param, Step

logger = logging.getLogger(__name__)

_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def _pick_interpolation(name: str, *, shrinking: bool) -> int:
    if name == "auto":
        return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return _INTERPOLATION[name]


def resize_frames(
    frames: List[np.ndarray], width: int, height: int, interpolation: str = "auto",
) -> List[np.ndarray]:
    """Plain-resize every frame to (width, height); same dtype out as in."""
    out = []
    for frame in frames:
        frame = np.asarray(frame)
        shrinking = frame.shape[0] * frame.shape[1] > width * height
        interp = _pick_interpolation(interpolation, shrinking=shrinking)
        if frame.shape[1] == width and frame.shape[0] == height:
            out.append(frame.copy())
        else:
            out.append(cv2.resize(frame, (width, height), interpolation=interp))
    return out


@register_step("resize_batch")
class ResizeBatchStep(Step):
    """Resize a batch of images and/or masks to a fixed size.

    inputs: {"images": Optional[List[np.ndarray]] (BGR uint8),
             "masks": Optional[List[np.ndarray]] (either mask form)} — at
             least one of them.
    outputs: the same keys, resized: images as BGR uint8, masks as float32
             [0, 1] foreground = 1.
    """

    PARAMS = (
        Param("width", int, REQUIRED, "Target width in pixels", minimum=1),
        Param("height", int, REQUIRED, "Target height in pixels", minimum=1),
        Param("interpolation", str, "auto",
              "cv2 filter. `auto` is INTER_AREA when shrinking and "
              "INTER_LINEAR when enlarging; the others force one. `nearest` "
              "keeps a hard mask hard at the cost of a jagged edge",
              choices=("auto", "area", "linear", "nearest"), advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        width, height = int(params["width"]), int(params["height"])
        interpolation = params["interpolation"]
        images: Optional[List[np.ndarray]] = inputs.get("images")
        masks: Optional[List[np.ndarray]] = inputs.get("masks")
        if images is None and masks is None:
            raise ValueError("resize_batch needs `images`, `masks`, or both")

        result: Dict[str, Any] = {}
        if images is not None:
            result["images"] = resize_frames(list(images), width, height, interpolation)
            logger.info(
                "resize_batch: %d images %s -> %dx%d",
                len(images), _describe(images), width, height,
            )
        if masks is not None:
            normalized = [normalize_mask(m) for m in masks]
            resized = resize_frames(normalized, width, height, interpolation)
            result["masks"] = [np.clip(m, 0.0, 1.0).astype(np.float32) for m in resized]
            logger.info(
                "resize_batch: %d masks %s -> %dx%d",
                len(masks), _describe(masks), width, height,
            )
        return result


def _describe(frames: List[np.ndarray]) -> str:
    sizes = {(np.asarray(f).shape[1], np.asarray(f).shape[0]) for f in frames}
    if len(sizes) == 1:
        w, h = next(iter(sizes))
        return f"{w}x{h}"
    return "of mixed sizes"
