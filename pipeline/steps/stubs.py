"""Placeholders for the heavy model steps, documenting the intended contract
for each so implementation can proceed independently per-model. None of
these run yet — they exist so the workflow YAMLs and the dispatcher wiring
can be built/tested end-to-end against the dataset_io steps while the model
integrations land one at a time.

Recommended dispatch per step (see prior scoping in this conversation):
  - rmbg              in_process   (plain PyTorch, no compiled extensions)
  - sam3d_body        subprocess   (pinned detectron2 build, own venv)
  - sapiens2_lite     in_process   (pure PyTorch "lite" inference path)
  - wan22_vace_denoise subprocess  (diffusers, own venv/CUDA-heavy deps)
  - seedvr2           subprocess or docker (flash-attn + apex pinned to
                       torch 2.4.0 / CUDA 12.1-12.4 — venv if that ABI
                       matches the host, docker if it doesn't)
"""

from __future__ import annotations

from typing import Any, Dict

from ..registry import register_step
from ..step import Step


@register_step("rmbg")
class RMBGStep(Step):
    """Background removal (RMBG-2.0).

    inputs: {"image": np.ndarray BGR}
    params: {} (model is fixed resolution per RMBG-2.0's own preprocessing)
    outputs: {"mask": np.ndarray HxW float32 [0,1]}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("RMBG-2.0 native inference not yet ported")


@register_step("sam3d_body")
class SAM3DBodyStep(Step):
    """SAM-3D-Body mesh/joint reconstruction from a single image.

    inputs: {"image": np.ndarray BGR}
    params: {} (checkpoint path baked into the isolated env's config)
    outputs: {"vertices": np.ndarray, "faces": np.ndarray, "joints": np.ndarray,
              "focal_length": float}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("SAM-3D-Body native inference not yet ported")


@register_step("sapiens2_lite")
class Sapiens2LiteStep(Step):
    """Sapiens2 normal-map estimation (pure-PyTorch lite path — avoids the
    full mmcv/OpenMMLab install).

    inputs: {"image": np.ndarray BGR}
    params: {"resolution": [w, h]}
    outputs: {"normal_map": np.ndarray HxWx3 float32}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Sapiens2-lite native inference not yet ported")


@register_step("wan22_vace_denoise")
class Wan22VaceDenoiseStep(Step):
    """Wan 2.2 VACE video denoise (dual-expert, few-step distilled schedule).

    Maps to diffusers.WanVACEPipeline(transformer, transformer_2,
    boundary_ratio=...); see prior scoping notes for the exact
    checkpoint/LoRA pairing and the open verification items (LoRA filename,
    mask polarity, boundary_ratio tuning, VRAM/offload strategy).

    inputs: {"control_video": List[np.ndarray], "control_masks": List[np.ndarray],
             "reference_image": np.ndarray}
    params: {"width": int, "height": int, "length": int, "steps": int,
             "cfg": float, "strength": float, "prompt": str, "negative_prompt": str,
             "seed": int}
    outputs: {"images": List[np.ndarray]}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Wan 2.2 VACE native inference not yet ported")


@register_step("seedvr2")
class SeedVR2Step(Step):
    """SeedVR2 video upscaling.

    inputs: {"images": List[np.ndarray]}
    params: {"scale": float}
    outputs: {"images": List[np.ndarray]}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("SeedVR2 native inference not yet ported")
