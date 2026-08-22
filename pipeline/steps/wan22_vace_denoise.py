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

Mask polarity: the ComfyUI graph runs `InvertMask` (node 199) on the
dataset's mask channel before feeding it to `WanVaceToVideo` as
`control_masks` — i.e. VACE's mask semantics (white=regions to generate,
per diffusers' pipeline_wan_vace.py) are the *inverse* of the dataset's mask
convention (which follows RMBG's foreground=1 convention). Replicated below
as `1.0 - mask`; this one is read directly off the graph, not guessed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

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
        from diffusers import WanVACEPipeline

        checkpoint = params.get("checkpoint", DEFAULT_CHECKPOINT)
        pipe = WanVACEPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)

        if params.get("use_lora", True):
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

        if params.get("quantize", True):
            from torchao.quantization import Float8WeightOnlyConfig, quantize_

            quant_config = Float8WeightOnlyConfig()
            quantize_(pipe.transformer, quant_config)
            quantize_(pipe.transformer_2, quant_config)

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

        video = [_bgr_to_rgb(frame) for frame in inputs["control_video"]]
        masks = [1.0 - np.asarray(m, dtype=np.float32) for m in inputs["control_masks"]]
        ref_img = inputs.get("reference_image")
        reference_images = [_bgr_to_rgb(ref_img)] if ref_img is not None else None

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


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_float_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
