"""SAM-3D-Body mesh/joint reconstruction from a single image.

Gated checkpoint (facebook/sam-3d-body-dinov3 on HF — a human must accept
the license in the HF UI before any token can download it) and a pinned
detectron2 build -> dispatch: subprocess, own venv
(see envs/sam3dbody/requirements.txt), never import `sam_3d_body` at module
top level.

API surface below is inferred from facebookresearch/sam-3d-body's demo.py
(load_sam_3d_body() -> (model, model_cfg); SAM3DBodyEstimator.process_one_image()
-> outputs consumed by visualize_sample_together(img, outputs, estimator.faces)),
not from a documented return schema — the exact output dict keys/shapes are
UNVERIFIED and are the first thing to check once sam-3d-body is actually
importable on a pod (`python -c "help(SAM3DBodyEstimator.process_one_image)"`
beats guessing further from here).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step

DEFAULT_CHECKPOINT_REPO = "facebook/sam-3d-body-dinov3"


@register_step("sam3d_body")
class SAM3DBodyStep(Step):
    def __init__(self) -> None:
        self._estimator = None

    def load(self, params: Dict[str, Any]) -> None:
        from huggingface_hub import snapshot_download
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        checkpoint_dir = params.get("checkpoint_dir") or snapshot_download(
            params.get("checkpoint_repo", DEFAULT_CHECKPOINT_REPO)
        )
        device = params.get("device", "cuda")
        model, model_cfg = load_sam_3d_body(
            checkpoint_dir, device=device, mhr_path=params.get("mhr_path")
        )
        self._estimator = SAM3DBodyEstimator(model, model_cfg, device=device)

    def unload(self) -> None:
        self._estimator = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        if self._estimator is None:
            self.load(params)

        # process_one_image takes a file path in demo.py, not an array —
        # round-trip through a tempfile rather than assume an array overload
        # exists.
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            cv2.imwrite(str(image_path), inputs["image"])
            outputs = self._estimator.process_one_image(
                str(image_path),
                bbox_thr=params.get("bbox_thr", 0.8),
                use_mask=params.get("use_mask", False),
            )

        person = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return {
            "vertices": np.asarray(person["vertices"]),
            "faces": np.asarray(self._estimator.faces),
            "joints": np.asarray(person.get("joints", person.get("keypoints_3d"))),
            "focal_length": float(person.get("focal_length", person.get("focal", 0.0))),
        }
