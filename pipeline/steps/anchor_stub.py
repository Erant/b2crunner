"""Anchor-frame warp/injection.

Ported from nodes/generate_firstlast_node.py and nodes/inject_anchor_node.py
in the original ComfyUI-Body2COLMAP repo:

- GenerateFirstLast homography-warps the original reference photo to align
  pixel-for-pixel with the rendered skeleton at the orbit's anchor camera
  (accounts for focal-length auto-framing and the look_at rotation
  correction; see body2colmap.utils.compute_warp_to_camera). The warp is
  reusable across a circular render and a later helical re-render that
  shares the same anchor camera — compute once, inject into both.
- InjectAnchor then overwrites every frame whose camera position matches
  the anchor (matched by position, not frame index, since more than one
  frame can land there — e.g. a circular path's default overlap=1
  duplicates the first camera as the last) with that warped image, so the
  diffusion pass gets one real conditioning frame instead of an all-render
  batch.

This is the mechanism behind cyber_6f's initial/ alpha convention
(pipeline/steps/wan22_vace_denoise.py's docstring): the injected anchor
frame is marked alpha=0 ("already real, don't denoise"), every synthetic
render frame is alpha=255 ("needs denoising"). Wiring these two steps in
is what would produce that mask correctly for a dataset built from scratch,
rather than relying on cyber_6f already having it baked in.

Both steps take render.py's output directly: GenerateFirstLastStep's
`camera`/`original_focal_length`/`render_size` inputs are exactly
render.py's `image_warp` dict's fields (override_cam_from_mesh=True run);
InjectAnchorStep's `cameras`/`anchor_position` are render.py's own
`cameras`/`anchor_position` outputs.

VERIFICATION STATUS differs between the two steps. `inject_anchor` is
verified against recorded output: `cyber_6f/initial` records an
anchor_position at the world origin, and frames 1 and 81 of that dataset
are byte-identical to its anchor.png — so the recorded data independently
says which frames the ComfyUI flow injected into, and this step finds the
same two from camera positions alone, including after a rotate_views
reordering. `generate_firstlast`'s warp is still verified on synthetic
data only, and cannot be checked against cyber_6f: the image it warps is
the front view SAM-3D-Body ran on, which that dataset does not keep — its
reference.png is the whole two-panel front/back sheet, framed differently.
(Current runs split that sheet up front and keep only the back half as
reference.png; the front half lives at `scene.front_image` for this step
and is likewise not persisted. See steps/reference_sheet.py.) It needs a
real render.py `image_warp` output, so it waits on a pod. See
tests/test_anchor.py.

Unlike the original ComfyUI nodes, this port works in cv2 BGR uint8
throughout (no RGB<->BGR/float<->uint8 tensor conversion needed — that
was purely for ComfyUI's IMAGE tensor convention) and masks are float32
[0,1] with foreground=1, matching rmbg.py/brush.py's convention rather
than ComfyUI's inverted MASK (1.0=background).

**At every call site of this step, the mask is the VACE mask.**
`dataset.masks` is overloaded across the pipeline as a whole — rmbg puts a
foreground silhouette there for brush, render_splat puts the splat's alpha
there for mask_splat — but this step only ever runs immediately before a
diffusion pass, where the field is the frame's alpha on disk and
`wan22_vace_denoise`'s `control_masks`: 1.0 "synthetic, denoise this", 0.0
"a real photograph, keep it". So the value written at an injected frame is
0.0, in every caller, because an injected frame is by definition the real
photo. (The ComfyUI node writes 0.0 too, but arrives
there backwards: its MASK is inverted and `SaveDataset` re-inverts on the
way to disk, `alpha = (1 - mask) * 255`. Two inversions, same number, and
reasoning from the graph alone gets the opposite answer — check the
recorded alpha instead: `cyber2_6f/masked_splatted/frame_00038_.png` is
byte-identical to that stage's `anchor.png` and carries a uniform alpha of
0, while all 80 other frames are 255.)

**This step must run after `mask_splat`, not before it.** That ordering is
load-bearing and was wrong, at the cost of a whole run's output quality.
`Body2COLMAP_InjectAnchor` takes `masks` as an *input* and clones it; this
port took no masks at all and manufactured an all-1.0 batch on every call.
Placed between `render_splat` and `mask_splat` — where the mask field is
carrying the splat render's per-*pixel* alpha, the thing `mask_splat`
exists to threshold — that wiped the alpha out. `mask_splat`'s keep-test
then passed everywhere: ~78% of each frame that the reference flow blacks
out survived into `denoise_pass2`, soft Gaussian fringes and all, and pass
2 hallucinated around them differently per frame. That view disagreement is
the ghosting in the final splat.

Ordering it after `mask_splat` also matches the recorded run frame for
frame, and dissolves the conflict rather than managing it: `mask_splat`
consumes the spatial alpha and emits the all-1.0 VACE mask, and this step
only ever sees and writes VACE masks. It reproduces
`cyber2_6f/masked_splatted` exactly — every frame masked and filtered at
alpha 255, except the anchor frame, which is `anchor.png` verbatim (not
composited, not filtered) at alpha 0.

`masks` stays an optional input regardless. When it is supplied it is
passed through untouched; a step handed somebody else's data should not
destroy it, which is the general form of the bug above.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step


@register_step("generate_firstlast")
class GenerateFirstLastStep(Step):
    """Warp a reference image to align with the rendered skeleton at the
    anchor frame.

    inputs: {"image": np.ndarray BGR uint8 (reference photo),
             "camera": Camera, "original_focal_length": float,
             "render_size": Tuple[int, int],
             "bg_color": Optional[Tuple[float, float, float]] RGB [0,1],
             defaults to white}
    outputs: {"warped_image": np.ndarray BGR uint8}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.utils import compute_warp_to_camera

        camera = inputs["camera"]
        original_focal_length = inputs["original_focal_length"]
        render_w, render_h = inputs["render_size"]
        img = inputs["image"]
        h_img, w_img = img.shape[:2]

        bg_rgb = inputs.get("bg_color", (1.0, 1.0, 1.0))
        # BGR order to match img's channel order (bg_rgb is RGB, cv2 border
        # color must be BGR for this image).
        border_color = (
            int(round(bg_rgb[2] * 255)),
            int(round(bg_rgb[1] * 255)),
            int(round(bg_rgb[0] * 255)),
        )

        is_identity_rotation = np.allclose(camera.rotation, np.eye(3), atol=1e-5)

        if is_identity_rotation:
            # Pure affine: scale + translate (fast path)
            s = float(camera.fx / original_focal_length)
            tx = camera.cx - s * w_img / 2.0
            ty = camera.cy - s * h_img / 2.0
            M = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float64)
            warped = cv2.warpAffine(
                img, M, (render_w, render_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )
        else:
            # Full homography: accounts for rotation + intrinsic change
            H = compute_warp_to_camera(
                original_focal_length=original_focal_length,
                original_image_size=(w_img, h_img),
                target_camera=camera,
            )
            warped = cv2.warpPerspective(
                img, H, (render_w, render_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )

        return {"warped_image": warped}


@register_step("inject_anchor")
class InjectAnchorStep(Step):
    """Overwrite every frame sitting at the anchor camera position with a
    given image (typically GenerateFirstLastStep's output).

    inputs: {"images": List[np.ndarray], "cameras": List[Camera],
             "anchor_position": Optional[np.ndarray],
             "anchor_image": Optional[np.ndarray],
             "masks": Optional[List[np.ndarray]] — the VACE mask batch to
             inject into, passed through untouched except at the matched
             frames. Omit it and an all-1.0 batch is manufactured instead
             ("everything is synthetic, denoise it"), which is what the
             pre-denoise callers want.}
    params: {"tolerance_pct": float, default 0.1 — match tolerance as a
             percentage of the camera bounding-box diagonal}
    outputs: {"images": List[np.ndarray], "masks": List[np.ndarray]}
             (masks: float32 [0,1] VACE masks — the supplied batch where
             there was one, all-1.0 otherwise, and 0.0 at the injected
             anchor frame(s): a real photograph, do not denoise it)

    With no anchor_image/anchor_position (a dataset with no anchor frame,
    or generate_firstlast simply not wired in), the inputs pass through
    rather than failing the workflow — masks included, untouched.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = inputs["images"]
        cameras = inputs["cameras"]
        anchor_position = inputs.get("anchor_position")
        anchor_image = inputs.get("anchor_image")
        in_masks = inputs.get("masks")

        # Supplied masks are somebody else's data: pass them through rather
        # than manufacturing a batch over the top of them. With none
        # supplied, all-1.0 is the right VACE mask for a batch of renders.
        if in_masks is None:
            masks = [np.ones(img.shape[:2], dtype=np.float32) for img in images]
        else:
            masks = [np.asarray(m, dtype=np.float32) for m in in_masks]

        if anchor_image is None or anchor_position is None:
            return {"images": images, "masks": masks}

        anchor_position = np.asarray(anchor_position, dtype=np.float32)
        positions = np.stack([cam.position for cam in cameras], axis=0)
        scale = _scene_scale(positions)
        tolerance_pct = params.get("tolerance_pct", 0.1)
        threshold = (tolerance_pct / 100.0) * scale

        distances = np.linalg.norm(positions - anchor_position, axis=1)
        matches = [int(i) for i in np.flatnonzero(distances <= threshold)]

        if not matches:
            return {"images": images, "masks": masks}

        if tuple(anchor_image.shape) != tuple(images[0].shape):
            raise ValueError(
                f"anchor_image shape {tuple(anchor_image.shape)} does not match the "
                f"image batch frame shape {tuple(images[0].shape)}. Render the anchor "
                f"at the same resolution as the orbit (GenerateFirstLastStep uses the "
                f"render_size from render.py's image_warp output)."
            )

        out_images = list(images)
        for idx in matches:
            out_images[idx] = anchor_image
            # 0.0 = "a real photograph, keep it" in VACE's control-mask
            # sense. Uniform over the frame: this is a per-frame flag, not a
            # silhouette. See the module docstring.
            masks[idx] = np.zeros(anchor_image.shape[:2], dtype=np.float32)

        return {"images": out_images, "masks": masks}


def _scene_scale(positions: np.ndarray) -> float:
    """Characteristic scale = diagonal of bounding box of camera positions."""
    if len(positions) < 2:
        return 1.0
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    return float(diag) if diag > 0 else 1.0
