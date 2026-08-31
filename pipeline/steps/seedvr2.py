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

This step also rescales the dataset's camera intrinsics to match, which
used to be a separate `fit_cameras_to_images` step a workflow had to chain
right behind this one. SeedVR2 resamples the frames but has no way to
resize the cameras that describe them itself — it hands back plain arrays,
not `Camera` objects, and nothing else in the pipeline resizes images out
from under their cameras — so a dataset that skipped that follow-up step
held 1080x1920 images next to cameras still claiming 720x1280 (fx=1213.9):
`colmap_export` would write a cameras.txt that disagrees with its own
images, and `brush` would fit a splat at half the focal length the photos
were actually taken at (exactly the bug the recorded ComfyUI-era export in
`cyber_6f/colmap` has). Doing the rescale here instead keeps the dataset
congruent the moment this step returns, rather than for one step and then
not the next — every consumer of `cameras` is a `Dataset` field.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from ..paths import models_dir
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


def _fit_cameras_to_images(cameras: List[Any], images: List[np.ndarray]) -> tuple:
    """Rescale `cameras`' intrinsics to the size `images` actually are.

    Scaling is the whole operation: fx, fy, cx and cy all multiply by the
    same per-axis ratio, because an upscale is a pure resampling of the
    same view frustum. Poses are untouched — the camera did not move.

    Cameras that already match their image pass through unchanged (and say
    so in the log), so calling this on a dataset seedvr2 left alone (or
    that never upscaled at all) is a no-op.

    Returns (cameras, resolution) where resolution is the (width, height)
    of the first image, i.e. `Dataset.resolution`'s shape.
    """
    from body2colmap.camera import Camera

    if not images or not cameras:
        raise ValueError(
            "seedvr2 needs both images and cameras to keep them in sync; got "
            f"{len(images)} images and {len(cameras)} cameras."
        )

    rescaled = []
    changed = 0
    for index, camera in enumerate(cameras):
        image = images[min(index, len(images) - 1)]
        height, width = image.shape[:2]
        x_scale = width / float(camera.width)
        y_scale = height / float(camera.height)
        if x_scale == 1.0 and y_scale == 1.0:
            rescaled.append(camera)
            continue
        changed += 1
        rescaled.append(
            Camera(
                focal_length=(camera.fx * x_scale, camera.fy * y_scale),
                image_size=(width, height),
                principal_point=(camera.cx * x_scale, camera.cy * y_scale),
                position=camera.position,
                rotation=camera.rotation,
            )
        )

    first = images[0]
    resolution = (int(first.shape[1]), int(first.shape[0]))
    if changed:
        logger.info(
            "seedvr2: rescaled %d/%d cameras to %dx%d (was %dx%d, fx %.1f -> %.1f)",
            changed, len(rescaled), resolution[0], resolution[1],
            cameras[0].width, cameras[0].height, cameras[0].fx, rescaled[0].fx,
        )
    else:
        logger.info(
            "seedvr2: cameras already match the frames (%dx%d); nothing to do",
            resolution[0], resolution[1],
        )
    return rescaled, resolution


@register_step("seedvr2")
class SeedVR2Step(Step):
    # Almost all of these are inference_cli.py's own argparse knobs, mirrored
    # so this Step can reach the same code path main() does. That is exactly
    # what `advanced` is for: what this pipeline actually tunes is the top
    # group, and the rest exists because the upstream CLI has it.
    #
    # The two model names default to None rather than to
    # src.utils.model_registry's DEFAULT_DIT/DEFAULT_VAE because that module
    # only exists inside the seedvr2 venv — it is imported in load(), and
    # load() applies it as the fallback.
    PARAMS = (
        Param("resolution", int, 1080,
              "Target output SHORTEST edge, not a multiplier — asking for less "
              "than the input's shortest edge downscales", minimum=1),
        Param("batch_size", int, 5,
              "Frames per pass. The pipeline's own workflows set 1; raise it once "
              "a real VRAM budget for this step is known", minimum=1),
        Param("seed", int, 42, "Diffusion seed"),
        Param("vae_encode_tiled", bool, False, "Tile the VAE encode to save VRAM"),
        Param("vae_decode_tiled", bool, False, "Tile the VAE decode to save VRAM"),
        Param("color_correction", str, "lab",
              "How the output is matched back to the input's colour",
              choices=("lab", "none")),

        Param("max_resolution", int, 0, "Cap on the output's longest edge; 0 is no cap",
              minimum=0, advanced=True),
        Param("uniform_batch_size", bool, False, "Force every batch to the same size",
              advanced=True),
        Param("prepend_frames", int, 0, "Frames of lead-in context per batch",
              minimum=0, advanced=True),
        Param("temporal_overlap", int, 0, "Frames shared between consecutive batches",
              minimum=0, advanced=True),
        Param("input_noise_scale", float, 0.0, "Noise added to the input frames",
              advanced=True),
        Param("latent_noise_scale", float, 0.0, "Noise added in latent space",
              advanced=True),
        Param("dit_offload_device", str, "none", "Where the DiT parks between passes",
              advanced=True),
        Param("vae_offload_device", str, "none", "Where the VAE parks between passes",
              advanced=True),
        Param("tensor_offload_device", str, "cpu", "Where intermediate tensors park",
              advanced=True),
        Param("blocks_to_swap", int, 0, "DiT blocks swapped to host RAM per forward",
              minimum=0, advanced=True),
        Param("swap_io_components", bool, False, "Swap the DiT's IO layers too",
              advanced=True),
        Param("vae_encode_tile_size", int, 1024, "Encode tile size", advanced=True),
        Param("vae_encode_tile_overlap", int, 128, "Encode tile overlap", advanced=True),
        Param("vae_decode_tile_size", int, 1024, "Decode tile size", advanced=True),
        Param("vae_decode_tile_overlap", int, 128, "Decode tile overlap", advanced=True),
        Param("tile_debug", str, "false", "Upstream's tile debug switch", advanced=True),
        Param("attention_mode", str, "sdpa", "Attention implementation", advanced=True),
        Param("compile_dit", bool, False, "torch.compile the DiT", advanced=True),
        Param("compile_vae", bool, False, "torch.compile the VAE", advanced=True),
        Param("compile_backend", str, "inductor", "torch.compile backend", advanced=True),
        Param("compile_mode", str, "default", "torch.compile mode", advanced=True),
        Param("compile_fullgraph", bool, False, "torch.compile fullgraph", advanced=True),
        Param("compile_dynamic", bool, False, "torch.compile dynamic shapes", advanced=True),
        Param("compile_dynamo_cache_size_limit", int, 64, "Dynamo cache size limit",
              advanced=True),
        Param("compile_dynamo_recompile_limit", int, 128, "Dynamo recompile limit",
              advanced=True),
        # Caching on by default: this Step's load()/run() split is meant to
        # keep the runner alive across run() calls, mirroring main()'s
        # --cache_dit/--cache_vae batch-processing flags.
        Param("cache_dit", bool, True, "Keep the DiT runner alive between run() calls",
              advanced=True),
        Param("cache_vae", bool, True, "Keep the VAE alive between run() calls",
              advanced=True),
        Param("device_id", str, "0", "CUDA device index", advanced=True),
        Param("debug", bool, False, "Upstream's verbose debug output", advanced=True),
        Param("dit_model", str, None, "DiT checkpoint name; empty means upstream's default",
              advanced=True),
        Param("vae_model", str, None, "VAE checkpoint name; empty means upstream's default",
              advanced=True),
        Param("model_dir", str, None,
              "Weight cache root. Empty means B2C's models dir — do NOT make this a "
              "relative path: the vendored download_weight() treats it as its whole "
              "cache root and never consults HF_HOME, so a relative path resolves "
              "against the worker's cwd and puts several GB inside the container",
              advanced=True),
    )

    def __init__(self) -> None:
        self._debug = None
        self._dit_model = None
        self._model_dir = None
        self._cache: Dict[str, Any] = {}

    def load(self, params: Dict[str, Any]) -> None:
        from src.utils.debug import Debug
        from src.utils.downloads import download_weight
        from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE

        self._debug = Debug(enabled=params["debug"])
        self._dit_model = params["dit_model"] or DEFAULT_DIT
        self._vae_model = params["vae_model"] or DEFAULT_VAE
        # NOT the relative "models/SEEDVR2" this used to default to: the
        # vendored download_weight() treats model_dir as its whole cache
        # root and never consults HF_HOME, so a relative path resolved
        # against the worker subprocess's cwd and put several GB of DiT and
        # VAE weights inside the container rather than on the volume — lost
        # on every pod restart, and enough to exhaust a default container
        # disk on its own.
        self._model_dir = params["model_dir"] or str(models_dir() / "SEEDVR2")

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
            resolution=params["resolution"],
            max_resolution=params["max_resolution"],
            batch_size=params["batch_size"],
            uniform_batch_size=params["uniform_batch_size"],
            seed=params["seed"],
            prepend_frames=params["prepend_frames"],
            temporal_overlap=params["temporal_overlap"],
            color_correction=params["color_correction"],
            input_noise_scale=params["input_noise_scale"],
            latent_noise_scale=params["latent_noise_scale"],
            dit_offload_device=params["dit_offload_device"],
            vae_offload_device=params["vae_offload_device"],
            tensor_offload_device=params["tensor_offload_device"],
            blocks_to_swap=params["blocks_to_swap"],
            swap_io_components=params["swap_io_components"],
            vae_encode_tiled=params["vae_encode_tiled"],
            vae_encode_tile_size=params["vae_encode_tile_size"],
            vae_encode_tile_overlap=params["vae_encode_tile_overlap"],
            vae_decode_tiled=params["vae_decode_tiled"],
            vae_decode_tile_size=params["vae_decode_tile_size"],
            vae_decode_tile_overlap=params["vae_decode_tile_overlap"],
            tile_debug=params["tile_debug"],
            attention_mode=params["attention_mode"],
            compile_dit=params["compile_dit"],
            compile_vae=params["compile_vae"],
            compile_backend=params["compile_backend"],
            compile_mode=params["compile_mode"],
            compile_fullgraph=params["compile_fullgraph"],
            compile_dynamic=params["compile_dynamic"],
            compile_dynamo_cache_size_limit=params["compile_dynamo_cache_size_limit"],
            compile_dynamo_recompile_limit=params["compile_dynamo_recompile_limit"],
            # Caching on by default: this Step's load()/run() split is meant
            # to keep the runner alive across run() calls, mirroring
            # main()'s --cache_dit/--cache_vae batch-processing flags.
            cache_dit=params["cache_dit"],
            cache_vae=params["cache_vae"],
        )

        result = inference_cli._process_frames_core(
            frames_tensor=frames_tensor,
            args=args,
            device_id=params["device_id"],
            debug=self._debug,
            runner_cache=self._cache,
        )

        result = result.cpu().numpy() if hasattr(result, "cpu") else np.asarray(result)
        images = [
            cv2.cvtColor(np.clip(frame * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            for frame in result
        ]
        cameras, resolution = _fit_cameras_to_images(inputs["cameras"], images)
        return {"images": images, "cameras": cameras, "resolution": resolution}
