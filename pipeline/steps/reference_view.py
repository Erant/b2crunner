"""pick_rear_view — the second denoise's reference, taken from the first's output.

For a front/back sheet the reference `wan22_vace_denoise` conditions on is
the sheet's back panel, and this step hands it through untouched. A single
frontal photo has no back panel (steps/reference_sheet.py, `layout:
single`): pass 1 runs on the injected photograph alone, and this step then
fills the slot with the pass-1 frame whose camera looks at the subject
from the far side of the orbit — the view furthest from the photograph,
and the one pass 2's 81 frames can least reconstruct from it.

"Most opposite" is measured about the orbit target, as render_initial_views
built the path: the frame whose direction from `orbit_target` has the
smallest dot product with the anchor camera's. The circular stage-1 orbit
has 81 frames over 360 degrees, so the pick is within 2.2 degrees of the
true rear view; measuring rather than counting frames keeps it right if
the pattern, the count or the anchor's position on the path ever change.

The frame goes out cut with its rmbg matte (`foreground_masks` has just
produced it) and laid over 0.5 grey, `matte: true` — the ground the warped
anchor photograph is bordered with (generate_firstlast) and the frames pass
2 sees are composited on (mask_splat_fringes), so the reference agrees with
the batch it sits beside rather than carrying whatever room pass 1 painted.
The sheet's back panel is not matted: it is the upload as given.

The `layout` input is the split step's verdict, and this step runs in
both modes because a `when:` resolves against globals before the run
(pipeline/runner.py), so a runtime detection cannot gate it. The runner
also requires every declared output to be returned, which is why the
sheet branch returns the reference it was handed rather than nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step
from .mask_splat import _composite_one

logger = logging.getLogger(__name__)


@register_step("pick_rear_view")
class PickRearViewStep(Step):
    """Publish the reference the second denoise conditions on.

    inputs:  {"images": List[np.ndarray] — the pass-1 batch,
              "masks": List[np.ndarray] — its rmbg mattes (foreground = 1),
              "cameras": List[Camera], "anchor_position": (3,),
              "orbit_target": (3,), "layout": "sheet" | "single",
              "reference_image": np.ndarray | None — the sheet's back panel,
                                 or None for a single photo}
    outputs: {"reference_image": np.ndarray, "rear_view_index": int | None}
    """

    PARAMS = (
        Param("matte", bool, True,
              "Cut the picked frame with its matte and lay it over `bg_color`, "
              "so the reference stands on the same ground as the batch"),
        Param("bg_color", list, [0.5, 0.5, 0.5],
              "RGB in 0..1 the matted frame is composited over; 0.5 grey is "
              "what the anchor border and mask_splat_fringes use"),
        Param("debug_dir", str, None,
              "When set, write the picked frame (rear_view.png) and which "
              "frame it was (rear_view.json) there. Nothing in sheet mode",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        layout = inputs["layout"]
        reference = inputs.get("reference_image")

        if layout == "sheet":
            if reference is None:
                raise ValueError(
                    "pick_rear_view: layout is 'sheet' but no reference_image "
                    "was handed in — split_reference_sheet should have written "
                    "the back panel to dataset.reference_image"
                )
            logger.info("sheet run: keeping the sheet's back panel as the reference")
            return {"reference_image": reference, "rear_view_index": None}
        if layout != "single":
            raise ValueError(f"pick_rear_view: unknown layout {layout!r}")

        images: List[np.ndarray] = inputs["images"]
        cameras = inputs["cameras"]
        if len(cameras) != len(images):
            raise ValueError(
                f"pick_rear_view: {len(images)} frames but {len(cameras)} cameras"
            )
        anchor = inputs.get("anchor_position")
        target = inputs.get("orbit_target")
        if anchor is None or target is None:
            raise ValueError(
                "pick_rear_view: a single-photo run needs anchor_position and "
                "orbit_target (render_initial_views publishes both under "
                "dataset.extras with override_cam_from_mesh on)"
            )

        index, angle_deg = _most_opposite(cameras, np.asarray(anchor), np.asarray(target))
        frame = images[index]
        if params["matte"]:
            masks = inputs.get("masks")
            if masks is None or len(masks) != len(images):
                raise ValueError(
                    "pick_rear_view: `matte` needs one mask per frame in "
                    "dataset.masks (foreground_masks writes them)"
                )
            frame = _composite_one(frame, masks[index], tuple(params["bg_color"]))
        else:
            frame = np.ascontiguousarray(frame[:, :, :3])

        logger.info(
            "single-photo run: frame %d is the rear view (%.1f deg from the "
            "anchor about the orbit target)%s", index, angle_deg,
            f", matted over {params['bg_color']}" if params["matte"] else "",
        )
        if params["debug_dir"]:
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug / "rear_view.png"), frame)
            (debug / "rear_view.json").write_text(json.dumps({
                "rear_view_index": index,
                "angle_from_anchor_deg": angle_deg,
                "matte": bool(params["matte"]),
            }, indent=1))
        return {"reference_image": frame, "rear_view_index": index}


def _most_opposite(cameras, anchor: np.ndarray, target: np.ndarray):
    """(index, angle in degrees) of the camera furthest round the orbit from `anchor`."""
    positions = np.stack([np.asarray(cam.position, dtype=np.float64) for cam in cameras])
    directions = positions - target.astype(np.float64)
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0):
        raise ValueError("pick_rear_view: a camera sits on the orbit target")
    directions /= norms[:, None]
    anchor_dir = anchor.astype(np.float64) - target.astype(np.float64)
    if not np.linalg.norm(anchor_dir):
        raise ValueError("pick_rear_view: the anchor sits on the orbit target")
    anchor_dir /= np.linalg.norm(anchor_dir)
    cosines = directions @ anchor_dir
    index = int(np.argmin(cosines))
    angle = float(np.degrees(np.arccos(np.clip(cosines[index], -1.0, 1.0))))
    return index, angle
