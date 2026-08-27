"""RMBG-2.0 background removal.

Plain PyTorch (BiRefNet-based), no compiled extensions, no gated checkpoint
access needed -> in_process. Preprocessing follows the model card's
example_inference.py: resize to 1024x1024, ImageNet normalize, sigmoid the
last decoder output. Verified against real inference on an L40S pod
(single-image path) — see pipeline/README.md.

Batch support (inputs["images"], a List[np.ndarray]) was added afterward,
for the brush-training path (pipeline/steps/brush.py) which needs a mask
per training frame, not just the one reference image. Also verified on an
L40S pod: 5 wan22_vace_denoise output frames in, 5 correctly-shaped
float32 [0,1] masks out.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step

DEFAULT_CHECKPOINT = "briaai/RMBG-2.0"
_IMAGE_SIZE = (1024, 1024)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@register_step("rmbg")
class RMBGStep(Step):
    PARAMS = (
        Param("batch_size", int, 8, "Images per forward pass", minimum=1, advanced=True),
        Param("checkpoint", str, DEFAULT_CHECKPOINT, "HF repo for the segmentation model",
              advanced=True),
        Param("device", str, None, "Torch device; empty means cuda if available",
              advanced=True),
    )

    def __init__(self) -> None:
        self._model = None
        self._device = None

    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        checkpoint = params["checkpoint"]
        self._device = params["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
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
        if self._model is None:
            self.load(params)

        if "images" in inputs:
            batch_size = params["batch_size"]
            images = inputs["images"]
            masks = []
            for i in range(0, len(images), batch_size):
                masks.extend(self._run_batch(images[i : i + batch_size]))
            return {"masks": masks}

        return {"mask": self._run_batch([inputs["image"]])[0]}

    def _run_batch(self, images_bgr: List[np.ndarray]) -> List[np.ndarray]:
        import torch

        sizes = [img.shape[:2] for img in images_bgr]
        tensors = []
        for img in images_bgr:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            resized = cv2.resize(rgb, _IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(resized).permute(2, 0, 1))

        batch = torch.stack(tensors)
        mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_STD).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        batch = batch.to(self._device)
        if self._device == "cuda":
            batch = batch.half()

        with torch.no_grad():
            preds = self._model(batch)[-1].sigmoid().float().cpu()

        out = []
        for i, (h, w) in enumerate(sizes):
            mask = preds[i, 0].numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(mask.astype(np.float32))
        return out
