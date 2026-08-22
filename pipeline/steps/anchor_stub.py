"""Anchor-frame warp/injection — not yet ported, and not exercised by the
current smoke tests (cyber_6f already has this baked in from the full
ComfyUI flow that produced it). A from-scratch run — SAM3D reconstruction
through render through wan22_vace_denoise, not starting from a pre-built
dataset — is missing this pair of steps.

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
"""

from __future__ import annotations

from typing import Any, Dict

from ..registry import register_step
from ..step import Step


@register_step("generate_firstlast")
class GenerateFirstLastStep(Step):
    """Warp a reference image to align with the rendered skeleton at the
    anchor frame.

    inputs: {"image": np.ndarray BGR (reference photo),
             "camera": Camera, "original_focal_length": float,
             "render_size": Tuple[int, int]}
    outputs: {"warped_image": np.ndarray BGR}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Anchor-frame warp not yet ported")


@register_step("inject_anchor")
class InjectAnchorStep(Step):
    """Overwrite every frame sitting at the anchor camera position with a
    given image (typically GenerateFirstLastStep's output).

    inputs: {"images": List[np.ndarray], "cameras": List[Camera],
             "anchor_position": np.ndarray, "anchor_image": np.ndarray}
    outputs: {"images": List[np.ndarray], "masks": List[np.ndarray]}
             (masks: alpha=0 at the injected anchor frame(s), alpha=255
             elsewhere — see wan22_vace_denoise.py's mask-semantics note)
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Anchor-frame injection not yet ported")
