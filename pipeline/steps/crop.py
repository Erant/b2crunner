"""crop_to_box — cut a region out of a frame, and say where it came from.

One small step between `sapiens2_seg`'s box and `face_pointmap_splat`.
Sapiens2 resizes whatever it is given to 1024x768, so a face occupying 6%
of a full-body frame reaches the network as roughly 40 px of head, and no
amount of normal integration recovers relief that was never sampled.
Cropping first is what puts the face in front of the model at the
resolution masktest measured its numbers on.

What makes this more than `image[y0:y1, x0:x1]` is the bookkeeping. A
Gaussian built from a crop pixel has to end up on the ray through the
FULL-image pixel it was cut from, or the splat does not land on the mesh
it is supposed to sit on. So this step publishes `crop_info` — the box it
actually used, after padding and clamping, plus the size of the frame that
box indexes into — and `FacePointmapSplatStep._source_intrinsics` turns
that into the camera the crop is unprojected through.

Two properties the consumer depends on, both enforced here:

  * **The emitted crop is native resolution.** Nothing is resized, so the
    resize factor is exactly 1 and a crop pixel is a frame pixel. The
    consumer supports a uniform resize anyway (it derives the factor from
    the box against the crop's own size), but not producing one is simpler
    and loses nothing: the whole point is to give the network more pixels
    of the subject, and upsampling adds none.
  * **The box stays inside the frame.** Padding is applied first and then
    clamped, so a subject near an edge yields a smaller crop rather than
    an out-of-bounds one — and `crop_info` reports the clamped box, not the
    one that was asked for.

`aspect` is why this is not simply "pad by 35%": Sapiens2 letterboxes or
squashes to its own aspect ratio, and the face is better served by a crop
already close to it than by a wide letterbox with the head in a band
across the middle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


@register_step("crop_to_box")
class CropToBoxStep(Step):
    """Pad a box, clamp it to the frame, cut it out, and record where it was.

    inputs:  {"image": HxWx3 BGR uint8,
              "box": (x0, y0, x1, y1) — in this image's pixels}
    outputs: {"image": the crop, native resolution,
              "crop_info": {"box": (x0, y0, x1, y1) as actually used,
                            "full_size": (width, height) of the input frame,
                            "crop_size": (width, height) of the crop}}
    """

    PARAMS = (
        Param("padding", float, 0.35,
              "Grow the box by this fraction of its larger side before "
              "cropping. A segmentation box of a face stops at the jaw and "
              "the hairline; the splat wants the whole head plus a margin of "
              "background for the silhouette's soft alpha to fall off into",
              minimum=0.0),
        Param("aspect", float, 0.75,
              "Width/height the padded box is grown to. 0 keeps the box's own "
              "shape. The default is Sapiens2's own 768/1024 — a crop already "
              "at the network's aspect ratio neither letterboxes nor squashes",
              minimum=0.0),
        Param("min_size", int, 64,
              "Refuse a crop smaller than this on either side. A box this "
              "small means the detection was wrong, and a 20 px face makes a "
              "splat of nothing", minimum=1),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        image = np.asarray(inputs["image"])
        height, width = image.shape[:2]
        x0, y0, x1, y1 = (float(v) for v in inputs["box"])
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"crop_to_box: degenerate box {inputs['box']}")

        box = _pad(x0, y0, x1, y1, params["padding"], params["aspect"])
        box = _clamp(box, width, height)
        x0, y0, x1, y1 = box

        if (x1 - x0) < params["min_size"] or (y1 - y0) < params["min_size"]:
            raise ValueError(
                f"crop_to_box: the padded box {box} is smaller than "
                f"min_size={params['min_size']} on at least one side. The "
                f"detection that produced it is almost certainly wrong."
            )

        crop = image[y0:y1, x0:x1]
        logger.info(
            "crop_to_box: %s of %dx%d -> %dx%d (%.1f%% of the frame's pixels)",
            box, width, height, x1 - x0, y1 - y0,
            100.0 * (x1 - x0) * (y1 - y0) / float(width * height),
        )
        return {
            "image": np.ascontiguousarray(crop),
            "crop_info": {
                "box": box,
                "full_size": (width, height),
                "crop_size": (x1 - x0, y1 - y0),
            },
        }


def _pad(x0: float, y0: float, x1: float, y1: float,
         padding: float, aspect: float) -> Tuple[float, float, float, float]:
    """Grow a box by `padding` of its larger side, then to `aspect`.

    Padding is proportional to the larger side rather than to each side
    separately, so a tall narrow face box does not come back with a wide
    thin margin on top and bottom and none at the sides.
    """
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half_w, half_h = 0.5 * (x1 - x0), 0.5 * (y1 - y0)

    margin = padding * max(half_w, half_h)
    half_w += margin
    half_h += margin

    if aspect > 0:
        # Grow the deficient axis only — never crop content back off.
        half_w = max(half_w, half_h * aspect)
        half_h = max(half_h, half_w / aspect)

    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _clamp(box: Tuple[float, float, float, float],
           width: int, height: int) -> Tuple[int, int, int, int]:
    """Slide the box inside the frame where it can, then clip.

    Sliding before clipping keeps the requested size whenever the frame is
    big enough to hold it, so a head near the top of a portrait crop does
    not silently lose a third of its margin.
    """
    x0, y0, x1, y1 = box
    box_w, box_h = min(x1 - x0, float(width)), min(y1 - y0, float(height))

    x0 = min(max(x0, 0.0), width - box_w)
    y0 = min(max(y0, 0.0), height - box_h)

    return (int(round(x0)), int(round(y0)),
            int(round(x0 + box_w)), int(round(y0 + box_h)))
