"""RMBG-2.0 background removal.

Plain PyTorch (BiRefNet-based), no compiled extensions, no gated checkpoint
access needed -> in_process, cheapest step to verify once a pod is available.
Preprocessing follows the model card's example_inference.py: resize to
1024x1024, ImageNet normalize, sigmoid the last decoder output.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step

DEFAULT_CHECKPOINT = "briaai/RMBG-2.0"
_IMAGE_SIZE = (1024, 1024)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@register_step("rmbg")
class RMBGStep(Step):
    def __init__(self) -> None:
        self._model = None
        self._device = None

    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        checkpoint = params.get("checkpoint", DEFAULT_CHECKPOINT)
        self._device = params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForImageSegmentation.from_pretrained(checkpoint, trust_remote_code=True)
        model.eval().to(self._device)
        if self._device == "cuda":
            model.half()
        self._model = model

    def unload(self) -> None:
        self._model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if self._model is None:
            self.load(params)

        image_bgr = inputs["image"]
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, _IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.to(self._device)
        if self._device == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            preds = self._model(tensor)[-1].sigmoid().float().cpu()

        mask = preds[0, 0].numpy()
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return {"mask": mask.astype(np.float32)}
