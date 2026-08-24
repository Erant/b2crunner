"""SeedVR2 one-step diffusion video upscaling.

Verified end to end on a real L40S pod: a genuine 720x1280 -> 1440x2560
upscale of 5 frames from cyber_6f's initial/ dataset (`resolution` is the
CLI's target output shortest edge, not a multiplier — passing something
smaller than the input's shortest edge downscales instead, which is what
the first, deliberately-cheap smoke test did before being corrected).
Output visibly sharper with no artifacts or color-cast drift. dispatch:
subprocess, own venv — flash-attn/apex are NOT required despite an earlier
version of this docstring assuming so (see below); the default
attention_mode ("sdpa") is pure PyTorch.

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

An untiled VAE decode at 1440px target resolution OOM'd a 44GB L40S
outright on just 5 frames — real memory pressure, not a bug. Pass
params.vae_encode_tiled / params.vae_decode_tiled (True) for any resolution
much above the input's native size; that's exactly what those flags exist
for (confirmed: identical call with them on succeeded immediately after).
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from ..paths import models_dir
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
        # NOT the relative "models/SEEDVR2" this used to default to: the
        # vendored download_weight() treats model_dir as its whole cache
        # root and never consults HF_HOME, so a relative path resolved
        # against the worker subprocess's cwd and put several GB of DiT and
        # VAE weights inside the container rather than on the volume — lost
        # on every pod restart, and enough to exhaust a default container
        # disk on its own.
        self._model_dir = params.get("model_dir") or str(models_dir() / "SEEDVR2")

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
