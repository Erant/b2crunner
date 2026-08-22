"""Wan 2.2 VACE-Fun video denoise — dual-expert (high/low noise), fp8.

Best-guess port of the live ComfyUI graph in workflows/api/denoise.json
(KSamplerAdvanced x2, uni_pc/beta, steps=6, cfg=1, split at step 2) onto
diffusers.WanVACEPipeline. Everything below is UNVERIFIED against real
inference — the whole point of the params below being overridable is so a
wrong guess is a workflow YAML edit, not a code change, once a pod is
available to check output against the cyber_6f reference dataset
(initial/ --[strength=1.0]--> circular/, masked_splatted/ --[strength=0.8]-->
helical/).

Base checkpoint: linoyts/Wan2.2-VACE-Fun-14B-diffusers (bf16 diffusers
weights; model_index.json confirms boundary_ratio=0.875 and
UniPCMultistepScheduler, matching the ComfyUI graph's uni_pc/beta sampler).
The user's ComfyUI setup used a GGUF quant of the *ComfyUI-format* checkpoint
(Kijai's WanVideoWrapper weights); GGUF isn't diffusers-loadable, so instead
we fp8-quantize the bf16 diffusers checkpoint at load time via torchao. This
should land close to the same VRAM footprint as the GGUF quant did in
ComfyUI, but the quantized weights themselves are not bit-identical — expect
to tune `quantize`/dtype once real hardware is available.

LoRA: lightx2v/Wan2.2-Lightning's 4-step distill LoRA
(Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/{high,low}_noise_model.safetensors).
This is a *T2V* lightning LoRA, not VACE-specific — no VACE-specific lightning
LoRA is published as of this writing. Community usage applies T2V lightning
LoRAs to VACE checkpoints since VACE reuses the T2V transformer backbone plus
added control conditioning; this is the single biggest correctness risk in
this file and must be checked against cyber_6f first.

Mask semantics: this `mask` is NOT a spatial subject/foreground cutout —
it's a per-*frame* flag distinguishing already-good reference frames from
frames that need denoising. Verified directly against `cyber_6f`'s
`initial/` frames: each frame's alpha channel is uniform across the whole
image (not a per-pixel silhouette), with `frame_00001` (the anchor/
reference view) at alpha=0 and every other frame at alpha=255. VACE's own
convention is white=generate, black=keep (per diffusers'
pipeline_wan_vace.py), which already matches reference=0/denoise=255
directly — so the dataset mask is passed straight through as `mask/255`,
*not* inverted, despite the ComfyUI graph running an `InvertMask` node
(199) before this same tensor reaches `WanVaceToVideo`. That graph-literal
inversion was tried first and produced visibly wrong output on a real pod
run (the reference frame got regenerated while the frames needing
denoising were left untouched) — whatever convention the live ComfyUI
mask tensor uses internally, it isn't the same as what ends up baked into
these PNGs' alpha channel on disk. Trust the verified per-frame alpha
values over the graph reading if the two ever conflict again.

RMBGStep's foreground-mask output is a *different* kind of mask (spatial,
per-pixel) and is not what belongs in `control_masks` here — don't wire
`rmbg`'s output into this step expecting frame-selection semantics.

Attention backend: defaults to SageAttention via diffusers' attention
dispatcher (params["attention_backend"] = "auto"), steered per-GPU-arch by
`_select_sage_backend()` below — see its docstring for the SM89/L40S
correctness caveat. Pass "none" to force PyTorch native SDPA, or an explicit
diffusers backend name (see `set_attention_backend` docs) to override.

Caching: pass params["fused_cache_dir"] to skip the LoRA-fuse + fp8-quantize
work (slow, CPU-bound) on every load() after the first — the fused
transformer/transformer_2 get saved there via save_pretrained() and reloaded
directly on subsequent runs. Also the natural artifact to eventually publish
as a real fp8 diffusers VACE checkpoint, since none currently exists (see
prior scoping conversation) — but verify a cache-hit load actually produces
correct output before trusting it that far.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ..registry import register_step
from ..step import Step

DEFAULT_CHECKPOINT = "linoyts/Wan2.2-VACE-Fun-14B-diffusers"
DEFAULT_LORA_REPO = "lightx2v/Wan2.2-Lightning"
DEFAULT_LORA_SUBFOLDER = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"
DEFAULT_LORA_HIGH = "high_noise_model.safetensors"
DEFAULT_LORA_LOW = "low_noise_model.safetensors"


@register_step("wan22_vace_denoise")
class Wan22VaceDenoiseStep(Step):
    def __init__(self) -> None:
        self._pipe = None

    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from diffusers import WanVACEPipeline, WanVACETransformer3DModel

        checkpoint = params.get("checkpoint", DEFAULT_CHECKPOINT)
        cache_dir = params.get("fused_cache_dir")
        cache_hit = bool(cache_dir) and Path(cache_dir, "transformer").is_dir() and Path(cache_dir, "transformer_2").is_dir()

        if cache_hit:
            # Skip base-checkpoint download of the transformers entirely —
            # load the already LoRA-fused, fp8-quantized weights saved by a
            # prior run below. VAE/text_encoder/tokenizer/scheduler still
            # come from the base checkpoint (never modified, so never cached
            # separately).
            transformer = WanVACETransformer3DModel.from_pretrained(
                cache_dir, subfolder="transformer", torch_dtype=torch.bfloat16
            )
            transformer_2 = WanVACETransformer3DModel.from_pretrained(
                cache_dir, subfolder="transformer_2", torch_dtype=torch.bfloat16
            )
            pipe = WanVACEPipeline.from_pretrained(
                checkpoint, transformer=transformer, transformer_2=transformer_2, torch_dtype=torch.bfloat16
            )
        else:
            pipe = WanVACEPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)

        if not cache_hit and params.get("use_lora", True):
            lora_repo = params.get("lora_repo", DEFAULT_LORA_REPO)
            lora_subfolder = params.get("lora_subfolder", DEFAULT_LORA_SUBFOLDER)
            pipe.load_lora_weights(
                lora_repo,
                subfolder=lora_subfolder,
                weight_name=params.get("lora_high", DEFAULT_LORA_HIGH),
                adapter_name="lightning_high",
            )
            pipe.load_lora_weights(
                lora_repo,
                subfolder=lora_subfolder,
                weight_name=params.get("lora_low", DEFAULT_LORA_LOW),
                adapter_name="lightning_low",
                load_into_transformer_2=True,
            )
            # Fuse-then-quantize: quantized linear layers can't have LoRA
            # adapters loaded onto them afterward (see huggingface/diffusers
            # discussion #12953), so LoRA must be fused into the base weights
            # before torchao quantization runs below.
            pipe.fuse_lora(
                components=["transformer"],
                lora_scale=params.get("lora_strength_high", 1.0),
                adapter_names=["lightning_high"],
            )
            pipe.fuse_lora(
                components=["transformer_2"],
                lora_scale=params.get("lora_strength_low", 1.0),
                adapter_names=["lightning_low"],
            )
            pipe.unload_lora_weights()

        if not cache_hit and params.get("quantize", True):
            from torchao.quantization import Float8WeightOnlyConfig, quantize_

            quant_config = Float8WeightOnlyConfig()
            device = params.get("device", "cuda")
            # One expert on GPU at a time rather than quantizing both in
            # place on CPU: torchao's fp8 conversion is a GPU-kernel
            # operation, so running it on CPU tensors is much slower, and
            # holding both 14B bf16 experts on GPU simultaneously risks OOM
            # on smaller cards. Move each expert over, quantize, move back
            # to host RAM, and clear the CUDA cache before the next one.
            for transformer in (pipe.transformer, pipe.transformer_2):
                transformer.to(device)
                quantize_(transformer, quant_config)
                transformer.to("cpu")
                torch.cuda.empty_cache()

        if not cache_hit and cache_dir:
            # Save the fused+quantized transformers so the next load() skips
            # straight to the cache_hit branch above. Best-effort: confirmed
            # broken on at least one diffusers/torchao/safetensors version
            # combo ("Attempted to access the data pointer on an invalid
            # python storage" — safetensors can't serialize the torchao
            # quantized tensor subclass's storage) — never let a caching
            # failure take down an otherwise-working run.
            try:
                pipe.transformer.save_pretrained(str(Path(cache_dir) / "transformer"))
                pipe.transformer_2.save_pretrained(str(Path(cache_dir) / "transformer_2"))
            except Exception as e:  # noqa: BLE001 - caching is an optimization, never fatal
                print(f"wan22_vace_denoise: failed to cache fused weights to {cache_dir!r}: {e!r}")

        attention_backend = params.get("attention_backend", "auto")
        if attention_backend and attention_backend != "none":
            backend_name = (
                _select_sage_backend() if attention_backend == "auto" else attention_backend
            )
            if backend_name:
                try:
                    pipe.transformer.set_attention_backend(backend_name)
                    pipe.transformer_2.set_attention_backend(backend_name)
                except Exception as e:  # noqa: BLE001 - best-effort optimization, never fatal
                    print(
                        f"wan22_vace_denoise: set_attention_backend({backend_name!r}) failed "
                        f"({e!r}), falling back to PyTorch native SDPA"
                    )

        if params.get("cpu_offload", True):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(params.get("device", "cuda"))

        self._pipe = pipe

    def unload(self) -> None:
        self._pipe = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if self._pipe is None:
            self.load(params)
        pipe = self._pipe

        # VideoProcessor.get_default_height_width() assumes a raw ndarray
        # frame already carries a leading batch dim (1, H, W, C) — a plain
        # per-frame (H, W, C) array gets its axes misread (shape[1]=W read
        # as height, shape[2]=channel-count read as width, which then
        # rounds down to 0 against vae_scale_factor). PIL images take the
        # correct .height/.width attribute path instead, so convert
        # everything frame-like to PIL rather than relying on the "accepts
        # numpy too" documentation, which doesn't hold for a list of
        # unbatched per-frame arrays.
        video = [Image.fromarray(_bgr_to_rgb(frame)) for frame in inputs["control_video"]]
        # Passed through as-is (not inverted — see module docstring) and
        # normalized to [0, 1] regardless of whether the source mask is
        # already float [0,1] (RMBGStep's contract) or raw uint8 [0,255]
        # (Dataset.from_disk's alpha-channel masks).
        masks = [_mask_to_pil(m) for m in inputs["control_masks"]]
        ref_img = inputs.get("reference_image")
        # check_inputs() requires reference_images to be PIL.Image (or nested
        # lists thereof) specifically — unlike video/mask, a raw ndarray is
        # rejected outright.
        reference_images = [Image.fromarray(_bgr_to_rgb(ref_img))] if ref_img is not None else None

        generator = torch.Generator(device=params.get("device", "cuda"))
        generator.manual_seed(int(params.get("seed", 0)))

        prompt = params.get("prompt", "")
        subject_desc = inputs.get("subject_desc")
        if subject_desc and "$SUBJECT_DESC$" in prompt:
            prompt = prompt.replace("$SUBJECT_DESC$", subject_desc)

        result = pipe(
            prompt=prompt,
            negative_prompt=params.get("negative_prompt", ""),
            video=video,
            mask=masks,
            reference_images=reference_images,
            conditioning_scale=params.get("strength", 1.0),
            height=params["height"],
            width=params["width"],
            num_frames=params.get("length", len(video)),
            num_inference_steps=params.get("steps", 6),
            guidance_scale=params.get("cfg", 1.0),
            generator=generator,
            output_type="np",
        )

        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        images = [_rgb_float_to_bgr_uint8(frame) for frame in frames]
        return {"images": images}


def _select_sage_backend() -> Optional[str]:
    """Pick a diffusers attention-dispatch backend name for SageAttention,
    or None to leave the default (PyTorch native SDPA) in place.

    SageAttention needs Ampere or newer (SM80+). Its v2 CUDA int8+fp8 kernel
    is reported to produce incorrect output on Ada Lovelace/SM89 (e.g. the
    L40S this was written against) — thu-ml/SageAttention#360 — so that
    architecture gets steered to the pure-Triton int8+fp16 kernel instead,
    which is confirmed to work there at reduced speedup. UNVERIFIED: this
    steering logic itself, since attention-backend selection wasn't
    exercised on the pod session that wrote this file — check the reloaded
    output isn't corrupted before trusting `sage_hub` on non-SM89 hardware
    either.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        return None
    if (major, minor) == (8, 9):
        return "_sage_qk_int8_pv_fp16_triton"
    return "sage_hub"


def _mask_to_pil(m: np.ndarray) -> Image.Image:
    """Normalize a control mask to uint8 [0,255] and wrap as PIL, without
    inverting — see module docstring for why. Handles both RMBGStep's
    float32 [0,1] contract and Dataset.from_disk's raw uint8 [0,255] alpha
    channel transparently."""
    arr = np.asarray(m, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_float_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
