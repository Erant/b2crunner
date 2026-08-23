"""SeedVR2 one-step diffusion video upscaling.

Needs flash-attn/apex pinned to a specific torch/CUDA ABI -> dispatch:
subprocess if the pod's host ABI matches the pins in
envs/seedvr2/requirements.txt, docker otherwise. UNVERIFIED on real
hardware as of this writing — implemented against numz/ComfyUI-
SeedVR2_VideoUpscaler's actual source (read directly, not guessed).

The previous version of this module assumed `_process_frames_core` lived
in `src.core.generation_utils` and took a handful of fields. Both were
wrong: `_process_frames_core` is a private function defined in
`inference_cli.py` itself (the standalone CLI script, not a library
module), and it takes an `argparse.Namespace` with ~30 fields covering
every CLI flag (offload devices, torch.compile, BlockSwap, VAE tiling,
etc) — not just batch_size/resolution/max_resolution. Confirmed by reading
inference_cli.py's `_process_frames_core` (~line 831) and its `main()`'s
argparse defaults (~line 1347) directly.

`inference_cli.py` is safe to import as a module despite being written as
a script: its module-level side effects are `parse_known_args()` (ignores
unrecognized argv, won't choke on ours), `mp.set_start_method('spawn',
force=True)`, and `os.environ.setdefault(...)` for CUDA allocator config —
and `main()` only runs under `if __name__ == "__main__"`. So this module
vendors the whole repo (see envs/seedvr2/setup.sh) and imports
`inference_cli` directly rather than reimplementing the 4-phase
encode/upscale/decode/postprocess pipeline itself.

`_process_frames_core` also does NOT auto-download weights (only
`main()` calls `download_weight` before invoking it) — `load()` below
calls it explicitly. Repo's actual default DiT checkpoint is the 3B fp8
model (`seedvr2_ema_3b_fp8_e4m3fn.safetensors`), not the 7B one this
module originally guessed — kept as the default here too since nothing
in this project's docs says otherwise; override via params.dit_model for
7B quality at ~2x the VRAM/time cost.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step


@register_step("seedvr2")
class SeedVR2Step(Step):
    def __init__(self) -> None:
        self._debug = None
        self._dit_model = None
        self._model_dir = None
        self._cache: Dict[str, Any] = {}

    def load(self, params: Dict[str, Any]) -> None:
        from src.utils.debug import Debug
        from src.utils.downloads import download_weight
        from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE

        self._debug = Debug(enabled=params.get("debug", False))
        self._dit_model = params.get("dit_model", DEFAULT_DIT)
        self._vae_model = params.get("vae_model", DEFAULT_VAE)
        self._model_dir = params.get("model_dir", "models/SEEDVR2")

        # main() calls this before ever touching _process_frames_core;
        # _process_frames_core itself assumes the weights are already on
        # disk and will error deep inside model loading if they aren't.
        if not download_weight(
            dit_model=self._dit_model,
            vae_model=self._vae_model,
            model_dir=self._model_dir,
            debug=self._debug,
        ):
            raise RuntimeError("SeedVR2 weight download/validation failed")

        # Fresh cache dict per load() — _process_frames_core populates
        # 'ctx' and 'runner' into whatever dict it's handed and reuses them
        # on subsequent calls when cache_dit/cache_vae are True (see its
        # source), so this is the load-once/run-many caching mechanism.
        self._cache = {}

    def unload(self) -> None:
        self._cache = {}
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import argparse

        import torch
        import inference_cli

        if self._dit_model is None:
            self.load(params)

        frames = inputs["images"]
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(rgb)  # [T, H, W, C], float32 [0,1]

        # Mirrors inference_cli.py's argparse defaults (~line 1347 on) so
        # this calls the exact same code path main() does, just without
        # going through argv. Only the fields _process_frames_core and the
        # calls it makes (prepare_runner, compute_generation_info, ...)
        # actually read are included.
        args = argparse.Namespace(
            dit_model=self._dit_model,
            model_dir=self._model_dir,
            resolution=params.get("resolution", 1080),
            max_resolution=params.get("max_resolution", 0),
            batch_size=params.get("batch_size", 5),
            uniform_batch_size=params.get("uniform_batch_size", False),
            seed=params.get("seed", 42),
            prepend_frames=params.get("prepend_frames", 0),
            temporal_overlap=params.get("temporal_overlap", 0),
            color_correction=params.get("color_correction", "lab"),
            input_noise_scale=params.get("input_noise_scale", 0.0),
            latent_noise_scale=params.get("latent_noise_scale", 0.0),
            dit_offload_device=params.get("dit_offload_device", "none"),
            vae_offload_device=params.get("vae_offload_device", "none"),
            tensor_offload_device=params.get("tensor_offload_device", "cpu"),
            blocks_to_swap=params.get("blocks_to_swap", 0),
            swap_io_components=params.get("swap_io_components", False),
            vae_encode_tiled=params.get("vae_encode_tiled", False),
            vae_encode_tile_size=params.get("vae_encode_tile_size", 1024),
            vae_encode_tile_overlap=params.get("vae_encode_tile_overlap", 128),
            vae_decode_tiled=params.get("vae_decode_tiled", False),
            vae_decode_tile_size=params.get("vae_decode_tile_size", 1024),
            vae_decode_tile_overlap=params.get("vae_decode_tile_overlap", 128),
            tile_debug=params.get("tile_debug", "false"),
            attention_mode=params.get("attention_mode", "sdpa"),
            compile_dit=params.get("compile_dit", False),
            compile_vae=params.get("compile_vae", False),
            compile_backend=params.get("compile_backend", "inductor"),
            compile_mode=params.get("compile_mode", "default"),
            compile_fullgraph=params.get("compile_fullgraph", False),
            compile_dynamic=params.get("compile_dynamic", False),
            compile_dynamo_cache_size_limit=params.get("compile_dynamo_cache_size_limit", 64),
            compile_dynamo_recompile_limit=params.get("compile_dynamo_recompile_limit", 128),
            # Caching on by default: this Step's load()/run() split is meant
            # to keep the runner alive across run() calls, mirroring
            # main()'s --cache_dit/--cache_vae batch-processing flags.
            cache_dit=params.get("cache_dit", True),
            cache_vae=params.get("cache_vae", True),
        )

        result = inference_cli._process_frames_core(
            frames_tensor=frames_tensor,
            args=args,
            device_id=params.get("device_id", "0"),
            debug=self._debug,
            runner_cache=self._cache,
        )

        result = result.cpu().numpy() if hasattr(result, "cpu") else np.asarray(result)
        images = [
            cv2.cvtColor(np.clip(frame * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            for frame in result
        ]
        return {"images": images}
