"""SAM-3D-Body mesh/joint reconstruction from a single image.

Verified end to end on a real L40S pod: loaded facebook/sam-3d-body-dinov3
(gated checkpoint — a human must accept the license in the HF UI before any
token can download it) and ran real inference against cyber_6f's
anchor.png, producing sane-shaped output — 18439 vertices, 36874 faces, 127
joints, focal_length=1468.6px. dispatch: subprocess, own venv (see
envs/sam3dbody/requirements.txt), never import `sam_3d_body` at module top
level.

Output schema confirmed against PozzettiAndrea/ComfyUI-SAM3DBody's
process.py (the node pack the project's actual ComfyUI flow uses — see that
repo's SAM3DBodyProcess node, which does `output = outputs[0]` then reads
`pred_vertices`, `pred_keypoints_3d`, `pred_joint_coords`,
`pred_global_rots`, `pred_cam_t`, `focal_length`, `bbox`), then confirmed
directly by running this module's own `run()` against a real image and
checking the returned shapes match. `faces` comes from `estimator.faces`
(both the base repo's demo.py and the ComfyUI wrapper agree on this one).

The ComfyUI wrapper downloads a different checkpoint repo
(`apozz/sam-3d-body-safetensors`, a safetensors repackaging) and defers
actual model construction to an isolated worker process not shown in the
file that does the checkpoint download. This module instead calls the
official `facebookresearch/sam-3d-body` loading path directly against
`facebook/sam-3d-body-dinov3` — same underlying model class producing the
same output schema, different checkpoint packaging/loader. Two real,
undocumented bugs found and fixed getting this to actually load, neither
guessable from the public docs/notebook:

1. `load_sam_3d_body`'s `checkpoint_path` argument must be the `.ckpt`
   FILE itself, not its containing directory — it derives
   `model_config.yaml`'s location via `os.path.dirname(checkpoint_path)`
   internally (confirmed by reading `sam_3d_body/build_models.py`'s
   source), so passing the snapshot directory strips one path level too
   many.
2. `mhr_path` is NOT actually optional despite defaulting to `""` in
   `load_sam_3d_body`'s own signature — an empty string reaches
   `torch.jit.load("")` downstream and crashes. The checkpoint repo ships
   the real file at `assets/mhr_model.pt`; defaults to that path below
   unless overridden.
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
        # mhr_path is NOT optional despite the "" default in
        # load_sam_3d_body's own signature — an empty string reaches
        # torch.jit.load("") downstream and crashes with "The provided
        # filename  does not exist". The checkpoint repo ships the file at
        # assets/mhr_model.pt (confirmed present after snapshot_download on
        # a real pod); default to that unless overridden.
        mhr_path = params.get("mhr_path") or str(Path(checkpoint_dir) / "assets" / "mhr_model.pt")
        device = params.get("device", "cuda")
        model, model_cfg = load_sam_3d_body(
            str(checkpoint_path), device=device, mhr_path=mhr_path
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
