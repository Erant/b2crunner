"""crop_to_box — cut a region out of a frame, and say where it came from.

One small step between `sapiens2_seg`'s box and `face_pointmap_splat`.

What it cuts is not a close-up, and that is the whole story of `padding`.
At 0.35 this produced a crop that was mostly face, and the pointmap head
answers a face-filling image with a flat card — the flat face cap this was
raised to 3.5 to cure (2026-09-02). At 3.5 the face box comes back grown by
three and a half of its own half-sides, which frames the head against the
shoulders and the torso: a body, which is what Sapiens2 has something to
say about.

So the crop is no longer about resolution. Three things are left, and all
three still want a step here rather than the bare frame:

  * **Framing.** The box follows the head, so the subject sits near the
    middle of the network's field instead of wherever it happened to be in
    a full-body shot, and a frame far taller than 4:3 does not spend most
    of its input on empty floor.
  * **Aspect.** `aspect` grows the box to Sapiens2's own 768/1024 so the
    seg and pointmap configs — which squash anisotropically, with no
    letterbox — do not stretch the subject on the way in.
  * **Bookkeeping**, below, which is what lets the splat land on the right
    rays afterwards.

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
    and loses nothing.
  * **The box stays inside the frame, keeping its shape.** Padding is
    applied first and then clamped, so a subject near an edge yields a
    smaller crop rather than an out-of-bounds one — and `crop_info` reports
    the clamped box, not the one that was asked for. At `padding` 3.5 the
    padded box routinely wants to be bigger than the frame, so the clamp
    scales it down UNIFORMLY whenever an `aspect` was asked for, rather
    than clipping the two axes independently: clipping would quietly hand
    back a crop of some other shape, and the shape is the one thing
    `aspect` exists to control.

That the frame `crop_info` indexes into is the same photograph `sam3d_body`
was fitted to is not checked here — this step cannot know — but it is not
merely a convention either. A resized copy of that photograph would leave
this step's own arithmetic perfectly self-consistent (the box would still
be in "full-image" pixels, the resize factor would still be 1) and put
every Gaussian on a ray computed from a focal belonging to a different
pixel grid. `FacePointmapSplatStep._source_intrinsics` compares `full_size`
against the frame the focal was measured on, and the workflows are tested
to feed the whole face branch the one un-resized front half.
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
        Param("padding", float, 3.5,
              "Grow the box by this fraction of its larger HALF-side before "
              "cropping — 3.5 turns a face box into a head against a torso, "
              "not a head with a margin. Far more than the silhouette's soft "
              "alpha needs, deliberately: at 0.35 the crop is mostly face, "
              "which is out of anything a body model was trained to read, and "
              "the pointmap head answers it with a flat card. That is the "
              "flat face cap of 2026-09-02. The face keeps its own pixels "
              "either way (the crop is cut at native resolution); what it "
              "gives up is share of Sapiens2's 768x1024, and the trade came "
              "out in favour of the wider frame",
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
        box = _clamp(box, width, height, params["aspect"])
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
    """Grow a box by `padding` of its larger HALF-side, then to `aspect`.

    Padding is proportional to the larger half-side rather than to each
    side separately, so a tall narrow face box does not come back with a
    wide thin margin on top and bottom and none at the sides. Half-side,
    not side: `padding` 1.0 doubles the larger dimension, and the default
    3.5 multiplies it by 4.5.
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
           width: int, height: int, aspect: float) -> Tuple[int, int, int, int]:
    """Shrink the box to fit the frame, keeping its shape, then slide it in.

    Shrinking only ever happens when the padded box is bigger than the
    frame, which at `padding` 3.5 is the common case rather than the corner
    one — and how it shrinks is the part that matters. With an `aspect`
    asked for, both sides are scaled by the same factor, so the crop that
    comes out is the largest one of that shape the frame can hold. Clipping
    the two axes independently (what this did when the padding was 0.35 and
    the case was unreachable) would silently return a crop of some other
    shape, and Sapiens2's seg and pointmap configs squash anisotropically
    with no letterbox — so the subject would reach the network stretched by
    exactly the amount `aspect` exists to prevent. With `aspect` 0 there is
    no shape to keep and each axis is clipped on its own, as before.

    Sliding afterwards keeps the size whenever the frame is big enough to
    hold it, so a head near the top of a portrait crop does not lose a
    third of its margin instead of moving down the frame.
    """
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    box_w, box_h = x1 - x0, y1 - y0

    if aspect > 0:
        shrink = min(1.0, width / box_w, height / box_h)
        box_w, box_h = box_w * shrink, box_h * shrink
    else:
        box_w, box_h = min(box_w, float(width)), min(box_h, float(height))

    x0 = min(max(cx - 0.5 * box_w, 0.0), width - box_w)
    y0 = min(max(cy - 0.5 * box_h, 0.0), height - box_h)

    return (int(round(x0)), int(round(y0)),
            int(round(x0 + box_w)), int(round(y0 + box_h)))
