"""Sapiens2 surface-normal estimation.

Verified against real inference on an L40S pod: both the single-image
(`inputs["image"]`) and batched (`inputs["images"]`) paths run cleanly
against real wan22_vace_denoise output frames, producing correctly-shaped,
properly L2-normalized (min/max within [-1, 1]) normal maps. Model loads in
well under a second (0.4B checkpoint) — cheap enough to not bother with a
disk-cached/pre-warmed instance.

Uses transformers' first-class Sapiens2 support (added to the library
directly — see the model doc at
https://huggingface.co/docs/transformers/model_doc/sapiens2) rather than
the older facebookresearch/sapiens (v1) "lite" torchscript inference path
this step's name originally referenced. That older path needed its own
minimal-dependency install specifically to avoid the full mmcv/OpenMMLab
stack; transformers' AutoModel path achieves the same "no heavy CV
framework" goal for free, so the "lite" framing in this step's registered
name is about that outcome, not about using the old inference script.

Default checkpoint is the smallest (0.4B) normal-estimation variant —
facebook/sapiens2-normal-0.4b — for speed; pass params["checkpoint"] for a
larger one (0.4b/0.6b/1b/2b/2b are documented as the available sizes but
only 0.4b's exact repo id is confirmed from the model doc directly).

Output is raw (unnormalized) XYZ normals in camera space, L2-normalized to
[-1, 1] via the image processor's post_process_normal_estimation and
resized back to the input resolution — no background masking applied here;
pipeline/steps/brush.py combines this output with a separate foreground
mask (rmbg's) at the point the normal map actually gets written to disk,
matching nodes/brush_node.py's original design ("a normal map that carries
its own alpha keeps it; otherwise borrow the RGB frame's mask").
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step

DEFAULT_CHECKPOINT = "facebook/sapiens2-normal-0.4b"


@register_step("sapiens2_lite")
class Sapiens2LiteStep(Step):
    """Sapiens2 normal-map estimation.

    inputs: {"image": np.ndarray BGR} or {"images": List[np.ndarray] BGR}
    params: {"checkpoint": str, "device": str, "batch_size": int}
    outputs: {"normal_map": np.ndarray HxWx3 float32 in [-1,1]}
             or {"normal_maps": List[np.ndarray]} for the batched path
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None

    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForNormalEstimation

        checkpoint = params.get("checkpoint", DEFAULT_CHECKPOINT)
        self._device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = AutoModelForNormalEstimation.from_pretrained(checkpoint).to(self._device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        if self._model is None:
            self.load(params)

        if "images" in inputs:
            batch_size = int(params.get("batch_size", 8))
            images = inputs["images"]
            normals = []
            for i in range(0, len(images), batch_size):
                normals.extend(self._run_batch(images[i : i + batch_size]))
            return {"normal_maps": normals}

        return {"normal_map": self._run_batch([inputs["image"]])[0]}

    def _run_batch(self, images_bgr: List[np.ndarray]) -> List[np.ndarray]:
        import torch
        from PIL import Image

        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_bgr]
        sizes = [(img.shape[0], img.shape[1]) for img in images_bgr]

        inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)

        result = self._processor.post_process_normal_estimation(
            outputs, source_sizes=sizes, target_sizes=sizes
        )

        out = []
        for r in result:
            normals = r["normals"]  # (3, H, W), unit vectors in [-1, 1]
            normals_hwc = normals.permute(1, 2, 0).float().cpu().numpy()
            out.append(normals_hwc.astype(np.float32))
        return out
