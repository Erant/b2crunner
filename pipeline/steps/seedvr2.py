"""SeedVR2 one-step diffusion video upscaling.

Needs flash-attn/apex pinned to a specific torch/CUDA ABI -> dispatch:
subprocess if the pod's host ABI matches the pins in
envs/seedvr2/requirements.txt, docker otherwise.

API surface below is inferred from numz/ComfyUI-SeedVR2_VideoUpscaler's
inference_cli.py (prepare_runner() -> (runner, cache); frames processed via
_process_frames_core(frames_tensor, args, device_id, debug, runner_cache)).
Those are internal/private-looking names (leading underscore) from a
ComfyUI custom node repo, not a published library API — this module assumes
that repo is vendored onto sys.path inside the seedvr2 env. UNVERIFIED:
exact import path once vendored, exact `_Args` fields `_process_frames_core`
actually reads (only batch_size/resolution/max_resolution are referenced
here, inferred from the CLI script's argparse flags), and whether
`_process_frames_core` is still named that in whatever revision gets pinned.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step

DEFAULT_DIT_MODEL = "seedvr2_ema_7b_fp8_e4m3fn.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"


@register_step("seedvr2")
class SeedVR2Step(Step):
    def __init__(self) -> None:
        self._runner = None
        self._cache = None

    def load(self, params: Dict[str, Any]) -> None:
        from src.core.generation_utils import prepare_runner

        self._runner, self._cache = prepare_runner(
            dit_model=params.get("dit_model", DEFAULT_DIT_MODEL),
            vae_model=params.get("vae_model", DEFAULT_VAE_MODEL),
            model_dir=params.get("model_dir", "models/seedvr2"),
            dit_cache=True,
            vae_cache=True,
            attention_mode=params.get("attention_mode", "sdpa"),
        )

    def unload(self) -> None:
        self._runner = None
        self._cache = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from src.core.generation_utils import _process_frames_core

        if self._runner is None:
            self.load(params)

        frames = inputs["images"]
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(rgb)  # [T, H, W, C], matches documented input shape

        scale = params.get("scale", 2.0)
        target_res = params.get("resolution", int(round(frames[0].shape[0] * scale)))

        class _Args:
            batch_size = params.get("batch_size", _round_4n1(len(frames)))
            resolution = target_res
            max_resolution = params.get("max_resolution", 2160)

        result = _process_frames_core(
            frames_tensor=frames_tensor,
            args=_Args(),
            device_id=params.get("device_id", 0),
            debug=False,
            runner_cache=self._cache,
        )

        out = result.cpu().numpy() if hasattr(result, "cpu") else np.asarray(result)
        images = [
            cv2.cvtColor(np.clip(frame * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            for frame in out
        ]
        return {"images": images}


def _round_4n1(n: int) -> int:
    """SeedVR2 batch_size must follow 4n+1 (1, 5, 9, 13, ...)."""
    k = max(0, (n - 1) // 4)
    return 4 * k + 1
