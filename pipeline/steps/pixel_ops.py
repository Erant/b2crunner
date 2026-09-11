"""pixel_ops — pixel-space operations on a batch of frames, in place in the
pipeline, between the step that made them and the step that reads them.

One step, a fixed order of operations, each one a knob that is OFF at its
default. A workflow switches an operation on by setting its knob, and a
new operation is a function here plus its `Param`s — not a new step, so
the workflow's wiring (which batch, which matte, which frame is the
photograph) is written once. The operations run in the order they are
listed below, on float32 BGR, and the frames go out as uint8 BGR the way
they came in; the matte and every other dataset field pass through
untouched. Nothing here changes a frame's size or count.

Operations, in order:

  * `specular_suppress` (2026-09-10). Exists because of double diffusion:
    the intermediate splat is trained on `denoise_pass1`'s output, so its
    re-render — the control video `denoise_pass2` conditions on — already
    carries one diffusion pass's shading and highlights, and the splat's
    SH has baked those in as view-dependent sparkle (pass 1's highlights
    are not consistent from frame to frame, and the fit reads that as
    view dependence). Pass 2 lights it all again, and each round blows the
    highlights out further: plasticky skin, cloth that reads as wet. The
    operation is the dichromatic model's specular-free image (Shen & Cai
    2009): a highlight is an additive, near-white component on top of the
    body colour, so it shows up as an elevated per-pixel `min(B,G,R)`;
    subtract the excess of that minimum over a threshold from all three
    channels equally and the whiteness goes while chroma and diffuse
    shading stay. The threshold is `mean + eta * std` of the minimum
    channel over the matte, measured over the WHOLE batch rather than per
    frame — a per-frame threshold drifts with how much skin versus
    clothing each camera sees, and a video model notices a subject whose
    skin dims and brightens along the orbit.

    It runs on pass 1's OUTPUT, before the splat is trained, and not on
    the re-render (where it first landed, on mask_splat): every pass-1 frame
    is a complete diffusion output with one lighting and no streaks or
    cull holes, so the statistics mean what they say, and treating the
    training data removes the SOURCE of the sparkle — the splat never
    learns it, and the re-render inherits the cut consistently across
    every view, the under-observed novel ones included, instead of being
    cut per pixel on whatever the rasteriser produced there. The
    photograph's frame (`anchor_frame_index`) is left out of the cut — its
    highlights are real, and face_priority leans on that frame for the
    face — but stays in the statistics, since it is the one frame whose
    skin is certainly skin.

See tests/test_pixel_ops.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


@register_step("pixel_ops")
class PixelOpsStep(Step):
    """Apply the switched-on pixel-space operations to a batch of frames.

    inputs:  {"images": List[np.ndarray] (BGR uint8, RGBA tolerated — the
              alpha is dropped),
              "masks": Optional[List[np.ndarray]] — the subject matte per
              frame, foreground = 1, either form. Operations that measure
              the subject (specular_suppress) need it and refuse to run
              without one, since measured over the frame whole they would
              be describing the background;
              "anchor_frame_index": Optional[int] — the photograph's frame,
              which the operations leave as it is}
    outputs: {"images": List[np.ndarray]} — BGR uint8, same size and count.

    With every knob at its default the frames come out byte-identical, and
    the step says so in the log rather than doing a silent round trip.
    """

    PARAMS = (
        Param("specular_suppress", float, 0.0,
              "How much of each highlight's specular excess to remove, 0 (off) "
              "to 1 (every highlight capped at the skin around it). The excess "
              "is the per-pixel min(B,G,R) above `mean + specular_eta * std` of "
              "that minimum over the whole batch's matte, subtracted from all "
              "three channels equally, so chroma and diffuse shading are "
              "untouched. Needs `masks`; skips the anchor frame",
              minimum=0.0, maximum=1.0),
        Param("specular_eta", float, 0.5,
              "How far above the batch's mean minimum-channel value a pixel has "
              "to sit, in standard deviations, before it counts as a highlight. "
              "Lower catches more of the skin's sheen, higher only the blown "
              "peaks", minimum=0.0, advanced=True),
        Param("specular_blur", float, 2.0,
              "Gaussian sigma in pixels applied to the excess map before it is "
              "subtracted, so a highlight's edge fades out rather than ringing; "
              "0 subtracts it as measured", minimum=0.0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = list(inputs["images"])
        masks = inputs.get("masks")
        anchor = inputs.get("anchor_frame_index")
        anchor = int(anchor) if anchor is not None else None

        frames = [_rgb(img) for img in images]
        mattes: Optional[List[np.ndarray]] = (
            [normalize_mask(m) for m in masks] if masks is not None else None
        )
        applied: List[str] = []

        amount = params["specular_suppress"]
        if amount > 0.0:
            if mattes is None:
                raise ValueError(
                    "pixel_ops: specular_suppress measures the subject's skin and "
                    "needs `masks` (the matte per frame) to know where it is. Wire "
                    "the step that produces it (rmbg) above this one."
                )
            eta, blur = params["specular_eta"], params["specular_blur"]
            threshold = _specular_threshold(frames, mattes, eta)
            frames = [
                bgr if i == anchor
                else _suppress_specular(bgr, fg, threshold, amount, blur)
                for i, (bgr, fg) in enumerate(zip(frames, mattes))
            ]
            applied.append(
                f"specular_suppress {amount:.2f} (min-channel threshold "
                f"{threshold:.1f}/255 over the batch's matte, eta {eta:.2f}, "
                f"excess blurred at sigma {blur:.1f} px"
                + (f", frame {anchor} left as the photograph)" if anchor is not None
                   else ", no anchor frame declared)")
            )

        if not applied:
            logger.info(
                "pixel_ops: every operation is off; %d frames pass through "
                "unchanged", len(frames),
            )
            return {"images": images}

        logger.info("pixel_ops: %d frames — %s", len(frames), "; ".join(applied))
        return {"images": [_to_u8(f) for f in frames]}


def _rgb(img: np.ndarray) -> np.ndarray:
    """The frame's colour channels, an RGBA frame's alpha dropped."""
    img = np.asarray(img)
    return img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img


def _to_u8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def _specular_threshold(frames: List[np.ndarray], mattes: List[np.ndarray],
                        eta: float) -> float:
    """`mean + eta * std` of the per-pixel minimum channel over every matte
    pixel in the batch, in 0-255 units.

    One number for the whole batch, accumulated rather than pooled: 81
    frames of minimum channel would be a spare copy of the batch, and the
    two moments need only three running sums. A matte pixel is one the
    matte puts at more than half foreground — the soft edge is left out of
    the statistics on both sides.
    """
    count = 0
    total = 0.0
    total_sq = 0.0
    for bgr, fg in zip(frames, mattes):
        i_min = bgr.astype(np.float32).min(axis=2)[fg > 0.5].astype(np.float64)
        count += i_min.size
        total += float(i_min.sum())
        total_sq += float(np.square(i_min).sum())
    if count == 0:
        # An empty matte has no skin to measure; a threshold above white
        # makes the suppression the no-op it should be.
        return 256.0
    mean = total / count
    var = max(total_sq / count - mean * mean, 0.0)
    return float(mean + eta * np.sqrt(var))


def _suppress_specular(bgr: np.ndarray, fg: np.ndarray, threshold: float,
                       amount: float, blur: float) -> np.ndarray:
    """The frame with its specular excess pulled down: the dichromatic
    model's specular-free image, scaled by `amount` and confined to the
    matte.

    The excess is the minimum channel's rise above `threshold`, the same
    value taken off all three channels — a highlight is body colour plus
    an (almost) white term, and removing an equal amount per channel
    removes only that term. Blurring the excess map first keeps a
    highlight's border from turning into a hard ring; the few units it
    spreads onto the skin around the peak are below what a denoise can
    tell apart. The matte weight keeps the subtraction off the background.
    """
    image = bgr.astype(np.float32)
    excess = np.clip(image.min(axis=2) - threshold, 0.0, None)
    if blur > 0.0:
        excess = cv2.GaussianBlur(excess, (0, 0), blur)
    excess *= amount * fg
    return np.clip(image - excess[:, :, None], 0.0, 255.0)
