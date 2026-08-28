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

FOV / focal length: `SAM3DBodyEstimator` takes an optional `fov_estimator`.
Left unset — the state this module shipped in until 2026-08-28 — it prints
"No FOV estimator... Using the default FOV!" and `process_one_image` builds
camera intrinsics as `focal = sqrt(h**2 + w**2)`, principal point at the
image centre (see `sam_3d_body/data/utils/prepare_batch.py`): a guess from
the image dimensions alone, not the real lens, and the `focal_length` this
step returns is then just that guess round-tripped through the model. This
module now defaults to MoGe-2 (`Ruicheng/moge-2-vitl-normal`, loaded via
the vendored `tools.build_fov_estimator.FOVEstimator`), which predicts
intrinsics from the image the same way upstream's `demo.py` /
`tools/export.py` do. `fov_estimator=""` restores the old
`sqrt(h**2 + w**2)` behaviour. The `focal_length=1468.6px` figure above was
from a default-FOV run, before this was wired.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step

DEFAULT_CHECKPOINT_REPO = "facebook/sam-3d-body-dinov3"
#: MoGe-2 checkpoint the FOV estimator pulls when `fov_checkpoint` is unset.
#: `FOVEstimator("moge2")` -> `MoGeModel.from_pretrained(<this>)`, which
#: `hf_hub_download`s a single `model.pt` from it (see pipeline/models.py's
#: `_fetch_moge`). ViT-L "normal" variant, matching upstream's default.
DEFAULT_FOV_CHECKPOINT_REPO = "Ruicheng/moge-2-vitl-normal"


@register_step("sam3d_body")
class SAM3DBodyStep(Step):
    PARAMS = (
        Param("bbox_thr", float, 0.8,
              "Person-detection confidence floor", minimum=0.0, maximum=1.0),
        Param("use_mask", bool, False, "Let the estimator segment the subject first"),
        Param("fov_estimator", str, "moge2",
              "Monocular estimator run before mesh fitting to recover the camera "
              "focal length / intrinsics. Empty string falls back to the model's "
              "default of focal = sqrt(h**2 + w**2) — a guess from the image "
              "dimensions, not the real lens. Only 'moge2' is implemented upstream."),
        Param("fov_checkpoint", str, None,
              "Local dir or HF repo for the FOV estimator's weights; empty means "
              + DEFAULT_FOV_CHECKPOINT_REPO, advanced=True),
        Param("checkpoint_repo", str, DEFAULT_CHECKPOINT_REPO,
              "HF repo the checkpoint is pulled from", advanced=True),
        Param("checkpoint_dir", str, None,
              "A local snapshot directory to use instead of downloading", advanced=True),
        Param("mhr_path", str, None,
              "The mhr_model.pt to load; empty means assets/mhr_model.pt inside the "
              "checkpoint directory. Not genuinely optional — an empty string reaches "
              "torch.jit.load and crashes", advanced=True),
        Param("device", str, "cuda", "Torch device", advanced=True),
    )

    def __init__(self) -> None:
        self._estimator = None

    def load(self, params: Dict[str, Any]) -> None:
        from huggingface_hub import snapshot_download
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        checkpoint_dir = params["checkpoint_dir"] or snapshot_download(
            params["checkpoint_repo"]
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
        mhr_path = params["mhr_path"] or str(Path(checkpoint_dir) / "assets" / "mhr_model.pt")
        device = params["device"]
        model, model_cfg = load_sam_3d_body(
            str(checkpoint_path), device=device, mhr_path=mhr_path
        )
        self._estimator = SAM3DBodyEstimator(
            model, model_cfg, fov_estimator=self._load_fov_estimator(params, device)
        )

    def _load_fov_estimator(self, params: Dict[str, Any], device: str):
        """MoGe-2 FOV estimator for `SAM3DBodyEstimator`, or None if disabled.

        Without it the estimator prints "No FOV estimator... Using the
        default FOV!" and process_one_image assumes focal = sqrt(h**2 + w**2)
        with the principal point at the image centre — dimensions, not the
        lens. With it, MoGe-2 predicts real intrinsics from the image.
        """
        name = (params["fov_estimator"] or "").strip()
        if not name:
            return None
        # `tools` is the vendored facebookresearch/sam-3d-body checkout
        # (pipeline/envs/sam3dbody/setup.sh + the Dockerfile's venv_sam3dbody
        # layer add it via a .pth). FOVEstimator("moge2") imports `moge`,
        # pinned to the last MoGe-2 commit and installed --no-deps in that
        # same venv. A miss here means the venv is stale, not that we should
        # silently fall back to the worse focal guess.
        try:
            from tools.build_fov_estimator import FOVEstimator
        except ImportError as exc:
            raise RuntimeError(
                f"sam3d_body: fov_estimator={name!r} needs the vendored "
                "sam-3d-body 'tools' package and MoGe in this venv. Re-run "
                "pipeline/envs/sam3dbody/setup.sh, or set fov_estimator='' to "
                "assume focal = sqrt(h**2 + w**2)."
            ) from exc
        return FOVEstimator(
            name=name, device=device,
            path=params["fov_checkpoint"] or DEFAULT_FOV_CHECKPOINT_REPO,
        )

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
                bbox_thr=params["bbox_thr"],
                use_mask=params["use_mask"],
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
