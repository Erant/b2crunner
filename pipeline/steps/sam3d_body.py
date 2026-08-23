"""SAM-3D-Body mesh/joint reconstruction from a single image.

Gated checkpoint (facebook/sam-3d-body-dinov3 on HF — a human must accept
the license in the HF UI before any token can download it) and a pinned
detectron2 build -> dispatch: subprocess, own venv
(see envs/sam3dbody/requirements.txt), never import `sam_3d_body` at module
top level.

Output schema confirmed against PozzettiAndrea/ComfyUI-SAM3DBody's
process.py (the node pack the project's actual ComfyUI flow uses — see that
repo's SAM3DBodyProcess node, which does `output = outputs[0]` then reads
`pred_vertices`, `pred_keypoints_3d`, `pred_joint_coords`,
`pred_global_rots`, `pred_cam_t`, `focal_length`, `bbox`, plus pose-param
fields), not guessed from facebookresearch/sam-3d-body's demo.py/notebook
the way the first version of this file was — those never show the actual
dict keys, only that the whole `outputs` object gets forwarded to a
visualization helper. `faces` still comes from `estimator.faces` (both the
base repo's demo.py and the ComfyUI wrapper agree on this one).

The ComfyUI wrapper downloads a different checkpoint repo
(`apozz/sam-3d-body-safetensors`, a safetensors repackaging) and defers
actual model construction to an isolated worker process not shown in the
file that does the checkpoint download — the exact estimator-construction
call there is UNVERIFIED. This module instead calls the official
`facebookresearch/sam-3d-body` loading path directly
(`load_sam_3d_body(checkpoint_dir, device=device, mhr_path=...)` ->
`SAM3DBodyEstimator(model, model_cfg)`, per that repo's own demo.py) against
`facebook/sam-3d-body-dinov3` — same underlying model class producing the
same output schema, different checkpoint packaging/loader. Confirm
`SAM3DBodyEstimator`'s real constructor signature once sam_3d_body is
actually importable (`python -c "import inspect;
from sam_3d_body import SAM3DBodyEstimator;
print(inspect.signature(SAM3DBodyEstimator.__init__))"`) — this version
drops the `device=` kwarg the first version of this file guessed, since
neither source confirms the constructor accepts it directly.
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
        # load_sam_3d_body's checkpoint_path arg must be the .ckpt FILE
        # itself, not its containing directory — confirmed by reading its
        # source (sam_3d_body/build_models.py): it derives model_config.yaml's
        # location via os.path.dirname(checkpoint_path), so passing the
        # directory strips one level too many and looks for the config in
        # the *parent* of the actual snapshot dir. Not documented anywhere,
        # found by hitting the resulting FileNotFoundError on a real pod.
        checkpoint_path = Path(checkpoint_dir) / "model.ckpt"
        device = params.get("device", "cuda")
        model, model_cfg = load_sam_3d_body(
            str(checkpoint_path), device=device, mhr_path=params.get("mhr_path", "")
        )
        self._estimator = SAM3DBodyEstimator(model, model_cfg)

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

        if not outputs:
            raise RuntimeError("SAM-3D-Body detected no people in the input image")
        person = outputs[0]

        return {
            "vertices": np.asarray(person["pred_vertices"]),
            "faces": np.asarray(self._estimator.faces),
            "joints": np.asarray(person["pred_joint_coords"]),
            "keypoints_3d": np.asarray(person["pred_keypoints_3d"]),
            "global_rots": np.asarray(person["pred_global_rots"]),
            "cam_t": np.asarray(person["pred_cam_t"]),
            "focal_length": float(person["focal_length"]),
            "bbox": np.asarray(person["bbox"]),
        }
