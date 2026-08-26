"""split_reference_sheet — cut the front/back generation sheet into its two panels.

The from-a-photo path does not start from a photo of the subject: it starts
from one square image a diffusion model generated, showing the subject twice
— facing front on the left, from behind on the right. Everything downstream
wants one panel or the other, never the sheet:

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
from typing import Any, Dict

import numpy as np

from ..registry import register_step
from ..step import Step

logger = logging.getLogger(__name__)


@register_step("split_reference_sheet")
class SplitReferenceSheetStep(Step):
    """Halve a front/back sheet down the middle.

    inputs: {"sheet": np.ndarray HxWx3 BGR uint8 — the generated sheet}
    params: {"front_side": "left" (default) or "right" — which half holds
             the front view}
    outputs: {"front": np.ndarray, "back": np.ndarray}

    A sheet is always at least as wide as it is tall (two portrait panels
    side by side), so a portrait input is a single photo handed in by
    mistake — that raises rather than quietly cutting a person in half.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        sheet = np.asarray(inputs["sheet"])
        height, width = sheet.shape[:2]

        if width < height:
            raise ValueError(
                f"split_reference_sheet got a {width}x{height} (portrait) image. "
                f"It expects the two-panel front/back sheet a diffusion model "
                f"generates — the subject facing front beside the same subject "
                f"from behind — which is never taller than it is wide. Halving "
                f"this one would cut a single subject down the middle."
            )

        front_side = params.get("front_side", "left")
        if front_side not in ("left", "right"):
            raise ValueError(
                f"split_reference_sheet: front_side must be 'left' or 'right', "
                f"got {front_side!r}"
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
        return {"front": front, "back": back}
