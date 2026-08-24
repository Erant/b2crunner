"""Wan 2.2 VACE-Fun video denoise — dual-expert (high/low noise), fp8.

Port of the live ComfyUI graph in workflows/api/denoise.json (KSamplerAdvanced
x2, uni_pc/beta, steps=6, cfg=1, split at step 2) onto
diffusers.WanVACEPipeline. VERIFIED against real inference on an L40S pod:
running the `initial/` --[strength=1.0]--> pass against all 81 frames of the
`cyber_6f` reference dataset (see below on why frame count matters) produces
output the project owner confirmed looks correct. Params below stay
overridable regardless — a future wrong guess (different dataset, different
LoRA) is still a workflow YAML edit, not a code change.

Frame count matters: the model/LoRA combination here is calibrated for
~81-frame clips (`denoise.json`'s `WanVaceToVideo` uses `length: 81`, matching
`LoadDataset`'s `batch_size: 81`). WAN's temporal VAE compresses 4 frames to
1 latent, so a short clip (e.g. 5 frames, used for early smoke tests here)
leaves the model almost no temporal context and produced visibly diverged,
lower-quality output relative to `cyber_6f`'s reference frames — not a code
bug, just an invalid test size. Always pass close to a full ~81-frame batch
when judging output quality, not a small trimmed subset.

Weights come from two repos, and the split is the point.

**The two transformers**: silveroxides/Wan_2.2-fp8_scaled_hybrid's VACE
files, `wan2.2_fun_vace_{high,low}_noise_14B-fp8_scaled_original.safetensors`,
17.58 GB each — a community fp8_scaled quant of exactly this model in
ComfyUI's format, loaded straight into WanVACETransformer3DModel by
pipeline/wan_fp8.py with no dequantize and no requantize anywhere (all 1331
tensors map; the fp8 bytes go from file to model unchanged). Confirmed
on-pod to be real WanVACETransformer3DModels on both experts
(`vace_in_channels: 96`, `vace_layers: [0,5,10,15,20,25,30,35]`, 235
vace-named submodules including full-width 5120x5120 Linear layers) — not a
plain T2V fallback silently ignoring `control_video`. The user's ComfyUI
setup used a GGUF quant of these same ComfyUI-format weights (Kijai's
WanVideoWrapper); GGUF isn't diffusers-loadable, fp8_scaled is, and it
lands in the same VRAM ballpark.

**Everything else**: linoyts/Wan2.2-VACE-Fun-14B-diffusers, for the VAE,
text encoder, tokenizer, scheduler and model_index.json (~11.89 GB all
told) plus the tiny `transformer/config.json` that describes the model
geometry wan_fp8.py instantiates. Its model_index.json confirms
boundary_ratio=0.875 and UniPCMultistepScheduler, matching the ComfyUI
graph's uni_pc/beta sampler. Its own `transformer/` and `transformer_2/`
(34.68 GB EACH) are deliberately never downloaded: diffusers skips fetching
any component handed to `from_pretrained` directly — it filters
`allow_patterns` against `passed_components`, see
diffusers/pipelines/pipeline_utils.py — and both transformers are passed in
from the fp8 files above. pipeline/models.py's prefetch patterns match, so
the two agree about what a cold pod pulls.

**The bf16 path was deleted deliberately; do not restore it.** This step
used to be able to download the bf16 diffusers transformers instead, fuse
the Lightning LoRA into them, fp8-quantize them with torchao at load time,
and cache the result under a `fused_cache_dir`. It worked, and it cost
81 GB per cold load against ~47 GB now, plus minutes of GPU-bound quantize
work and a cache that could only be a torch.save pickle (see
docs/fp8-quant-notes.md). The pre-quantized checkpoint is the same model at
half the download and no quantization at all, so there is no case left for
pulling bf16 weights. `fused_cache_dir` and `quantize` are gone with it —
nothing reads them, and a workflow still setting them is silently ignored,
so check for them if a run seems to be doing more work than it should.

LoRA: lightx2v/Wan2.2-Lightning's 4-step distill LoRA
(Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/{high,low}_noise_model.safetensors).
This is a *T2V* lightning LoRA, not VACE-specific — no VACE-specific lightning
LoRA is published as of this writing. Applying it to the VACE checkpoint
(since VACE reuses the T2V transformer backbone plus added control
conditioning) produces correct output per the verification above, so this
risk did not materialize in practice — kept as a param in case a different
dataset/prompt combination surfaces it.

$SUBJECT_DESC$ substitution: `prompt` must contain the literal substring
`$SUBJECT_DESC$` for `inputs["subject_desc"]` to get spliced in (see `run()`
below) — a caller that embeds this prompt through a shell heredoc or similar
must not let `$SUBJECT_DESC$` get backslash-escaped or shell-expanded before
it reaches Python, or the substitution silently no-ops and the model sees
the literal placeholder text instead of an actual subject description. Hit
exactly this in an ad-hoc bash test harness during verification — not a bug
in this module, but an easy trap for any script that constructs `params`
outside a plain Python/YAML path.

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

Where that per-frame flag actually comes from: `generate_firstlast`/
`inject_anchor` (pipeline/steps/anchor_stub.py, not yet ported) overwrite
the frame(s) at the anchor camera with a warped real photo and mark them
alpha=0; every synthetic render frame is alpha=255. `cyber_6f` already has
this baked in from the ComfyUI flow that produced it, which is why the
smoke test this was verified against never exercised the gap — a dataset
built from scratch needs those two steps wired in before this mask exists
at all.

Attention backend: defaults to SageAttention via diffusers' attention
dispatcher (params["attention_backend"] = "auto"), steered per-GPU-arch by
`_select_sage_backend()` below — see its docstring for the SM89/L40S
correctness caveat. Pass "none" to force PyTorch native SDPA, or an explicit
diffusers backend name (see `set_attention_backend` docs) to override.

Download budget, per cold load: 17.58 + 17.58 GB of fp8 experts plus
~11.89 GB of base-repo components (text_encoder 11.36, vae 0.51, the rest
kilobytes) — ~47 GB. Measured against the live repos; pipeline/models.py
carries the same numbers, which is what the pod prefetch reports before it
starts pulling. There is no on-disk cache of processed weights any more and
none is needed: nothing is processed. The files are used as downloaded, so
huggingface_hub's own cache in HF_HOME is the whole story, and a warm
volume makes the second load free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import Step

DEFAULT_CHECKPOINT = "linoyts/Wan2.2-VACE-Fun-14B-diffusers"
# The transformers, pre-quantized. Two separate files because the model is
# a dual-expert one: `transformer` denoises the high-noise steps and
# `transformer_2` the low-noise ones, split at boundary_ratio=0.875.
DEFAULT_FP8_REPO = "silveroxides/Wan_2.2-fp8_scaled_hybrid"
DEFAULT_FP8_HIGH = "wan2.2_fun_vace_high_noise_14B-fp8_scaled_original.safetensors"
DEFAULT_FP8_LOW = "wan2.2_fun_vace_low_noise_14B-fp8_scaled_original.safetensors"
DEFAULT_LORA_REPO = "lightx2v/Wan2.2-Lightning"
DEFAULT_LORA_SUBFOLDER = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"
DEFAULT_LORA_HIGH = "high_noise_model.safetensors"
DEFAULT_LORA_LOW = "low_noise_model.safetensors"


def resolve_fp8_checkpoint(
    value: str,
    repo: str = DEFAULT_FP8_REPO,
    local_files_only: bool = False,
) -> str:
    """A local path to one fp8 expert file, fetching it if it isn't here.

    `value` is either a path that already exists on disk (a hand-placed
    file, a bind-mounted volume, a prior download) or a filename inside
    `repo` — which is the default, and the normal case.

    Existence on disk is what decides, not the shape of the string. A
    "looks like a path" heuristic would send a mistyped local path to the
    Hub and surface it as a 404 against a repo the caller never named.

    `local_files_only=True` makes this a probe: it returns the cached path
    or raises without touching the network, which is how
    pipeline/models.py answers "is this already on the volume" using the
    exact call the step will make.
    """
    local = Path(value)
    if local.exists():
        return str(local)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, value, local_files_only=local_files_only)


@register_step("wan22_vace_denoise")
class Wan22VaceDenoiseStep(Step):
    # Which params load() actually reads. pipeline/worker.py's resident
    # worker reuses the loaded pipeline while these are unchanged and
    # rebuilds it when they are not — see load_signature() there.
    #
    # The per-call params are deliberately ABSENT: `strength`, `steps`,
    # `cfg`, `seed`, `prompt`, `negative_prompt`, `width`, `height`,
    # `subject_desc`. That is the whole point — fast_helical_full's two
    # passes differ only by `strength` (1.0 then 0.8), so listing it here
    # would rebuild the pipeline between them and buy nothing at all.
    LOAD_PARAMS = (
        "checkpoint", "fp8_repo", "fp8_checkpoint_high", "fp8_checkpoint_low",
        "fp8_config", "use_lora", "lora_repo", "lora_subfolder",
        "lora_high", "lora_low", "lora_strength_high", "lora_strength_low",
        "attention_backend", "cpu_offload", "device",
    )

    def __init__(self) -> None:
        self._pipe = None
        # Remembered from _finish_load: release_vram() has to undo whichever
        # placement load() chose, and they need opposite treatment.
        self._device = "cuda"
        self._cpu_offload = True

    def load(self, params: Dict[str, Any]) -> None:
        """Build the pipeline around the pre-quantized fp8 transformers.

        This is the only load path — see the module docstring on why the
        bf16 + fuse + torchao-quantize one was removed rather than kept as
        a fallback.

        Verified end to end against
        silveroxides/Wan_2.2-fp8_scaled_hybrid's VACE files: all 1331
        tensors map, and a full 40-block forward pass produces finite,
        correctly-shaped output. See pipeline/wan_fp8.py for how the key
        naming and the per-tensor scales are handled.
        """
        import torch
        from diffusers import WanVACEPipeline

        from ..wan_fp8 import load_config, load_fp8_transformer

        checkpoint = params.get("checkpoint", DEFAULT_CHECKPOINT)
        fp8_repo = params.get("fp8_repo", DEFAULT_FP8_REPO)
        fp8_high = resolve_fp8_checkpoint(
            params.get("fp8_checkpoint_high", DEFAULT_FP8_HIGH), fp8_repo
        )
        fp8_low = resolve_fp8_checkpoint(
            params.get("fp8_checkpoint_low", DEFAULT_FP8_LOW), fp8_repo
        )

        # Only the geometry, from the base repo's transformer/config.json —
        # kilobytes, and the reason `transformer/config.json` survives in
        # pipeline/models.py's allow_patterns while `transformer/*` does not.
        config = load_config(params.get("fp8_config", checkpoint))
        device = params.get("device", "cuda")
        transformer = load_fp8_transformer(fp8_high, config, device="cpu")
        transformer_2 = load_fp8_transformer(fp8_low, config, device="cpu")

        # VAE/text_encoder/tokenizer/scheduler still come from the base
        # repo; only the two transformers are replaced, and they are the
        # only things the fp8 files contain. Passing them here is also what
        # stops diffusers downloading the base repo's 34.68 GB bf16
        # transformers, which it would otherwise fetch and throw away.
        pipe = WanVACEPipeline.from_pretrained(
            checkpoint,
            transformer=transformer,
            transformer_2=transformer_2,
            torch_dtype=torch.bfloat16,
        )

        if params.get("use_lora", True):
            self._load_lora_unfused(pipe, params)

        self._finish_load(pipe, params, device)

    def _load_lora_unfused(self, pipe, params: Dict[str, Any]) -> None:
        """Apply the Lightning LoRA to already-quantized weights, unfused.

        The deleted bf16 path fused the LoRA into bf16 weights and
        quantized afterwards. Pre-quantized weights offer no such window,
        so the adapter stays live and its scale is set rather than baked
        in. That the LoRA can be applied at all is recent: the torch 2.13 /
        torchao 0.18 bump lifted diffusers' `torchao >= 0.16` gate on
        loading adapters onto quantized weights, which used to make fp8 and
        the Lightning LoRA mutually exclusive.

        Fusing specifically is NOT available on this path, and the reason is
        worth recording because the error message is opaque
        (`TypeError: can't multiply sequence by non-int of type 'float'`):
        peft's TorchaoLoraLinear can only merge when it can recover the
        requantization subclass, which it gets from a model that diffusers
        itself quantized via TorchAoConfig. wan_fp8.py builds the model
        directly from a state dict, so no quantizer is attached and merging
        has nothing to requantize with. Live adapters are unaffected —
        verified on GPU: forward output stays finite and the adapter
        measurably changes it.

        The cost of not fusing is a small per-step overhead (an extra
        low-rank matmul per adapted Linear), not a correctness difference.
        """
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
        # Set the scale per component rather than via pipe.set_adapters:
        # the two adapters live in different transformers (high-noise in
        # `transformer`, low-noise in `transformer_2`), and a pipeline-level
        # call has to guess which component each name belongs to.
        pipe.transformer.set_adapters(
            ["lightning_high"], [params.get("lora_strength_high", 1.0)]
        )
        pipe.transformer_2.set_adapters(
            ["lightning_low"], [params.get("lora_strength_low", 1.0)]
        )

    def _finish_load(self, pipe, params: Dict[str, Any], device: str) -> None:
        """Attention backend + offload, split out of load() for readability."""
        attention_backend = params.get("attention_backend", "auto")
        if attention_backend and attention_backend != "none":
            backend_name = (
                _select_sage_backend() if attention_backend == "auto" else attention_backend
            )
            if backend_name:
                try:
                    pipe.transformer.set_attention_backend(backend_name)
                    pipe.transformer_2.set_attention_backend(backend_name)
                except Exception as e:  # noqa: BLE001 - best-effort optimization
                    print(
                        f"wan22_vace_denoise: set_attention_backend({backend_name!r}) failed "
                        f"({e!r}), falling back to PyTorch native SDPA"
                    )

        self._device = device
        self._cpu_offload = bool(params.get("cpu_offload", True))
        if self._cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self._pipe = pipe

    def unload(self) -> None:
        self._pipe = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_vram(self) -> None:
        """Give the card back, keep the ~47 GB of weights in host RAM.

        Called by the resident worker after every job so `brush` — which
        runs on the GPU between this step's two passes in
        fast_helical_full — finds an empty card. Without this override the
        base-class no-op leaves the experts resident and brush OOMs; see
        Step.release_vram.

        The two placements load() can choose need opposite handling, which
        is why _finish_load records which one it used:

        * cpu_offload (the default): accelerate owns the placement. NOT
          `.to("cpu")` — that moves the modules out from under the hooks
          and desyncs them. `maybe_free_model_hooks()` is diffusers' own
          answer: it offloads every component and then re-applies the
          hooks, leaving the pipe ready for the next call, and is a silent
          no-op if offload was never enabled.
        * plain .to(device): no hooks to respect, so move it to CPU here
          and let run() put it back.
        """
        if self._pipe is None:
            return
        import torch

        if self._cpu_offload:
            self._pipe.maybe_free_model_hooks()
        else:
            self._pipe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        if self._pipe is None:
            self.load(params)
        pipe = self._pipe

        # Put the weights back if release_vram() moved them. Only the
        # non-offload placement needs this: with cpu_offload, accelerate's
        # hooks re-upload each module on demand.
        if not self._cpu_offload and torch.cuda.is_available():
            pipe.to(self._device)

        # VideoProcessor.get_default_height_width() assumes a raw ndarray
        # frame already carries a leading batch dim (1, H, W, C) — a plain
        # per-frame (H, W, C) array gets its axes misread (shape[1]=W read
        # as height, shape[2]=channel-count read as width, which then
        # rounds down to 0 against vae_scale_factor). PIL images take the
        # correct .height/.width attribute path instead, so convert
        # everything frame-like to PIL rather than relying on the "accepts
        # numpy too" documentation, which doesn't hold for a list of
        # unbatched per-frame arrays.
        from PIL import Image

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

    SageAttention needs Ampere or newer (SM80+), and the LOCALLY BUILT
    package — the image compiles it from source (see docker/Dockerfile's
    SAGE_REF layer) because 2.x has never been on PyPI and diffusers
    requires >= 2.1.1 for any sage backend.

    Not `sage_hub`, which this used to return for everything except SM89.
    That backend is broken with diffusers 0.40.0 (the newest release):
    diffusers pins version 1 of `kernels-community/sage-attention`, and the
    revision that resolves to has no `build` on the Hub, so every call ends
    in `RemoteEntryNotFoundError: 404`. Measured, not inferred. Since the
    caller wraps this in try/except, the old behaviour was a silent fall
    back to SDPA on every non-Ada GPU — i.e. "auto" bought nothing there.
    The local kernels work; prefer them.

    SM89 (Ada) gets the pure-Triton int8+fp16 kernel rather than the CUDA
    int8+fp8 one, because the latter was reported to produce incorrect
    output on Ada — thu-ml/SageAttention#360. Measured on an RTX 4070 Ti
    with SageAttention 2.2.0, 40 heads x 128 dim, against SDPA:

        seq    SDPA     triton int8+fp16      CUDA int8+fp8
        1024   0.37ms   0.26ms (cos .99992)   0.29ms (cos .99930)
        4096   4.80ms   2.98ms (cos .99991)   2.51ms (cos .99928)
        9216  22.32ms  12.75ms (cos .99991)   9.40ms (cos .99926)

    So on 2.2.0 the fp8 CUDA kernel shows no sign of #360 on this card, and
    is ~36% faster than the Triton one. It is still NOT the default: those
    are random tensors, not a 40-block diffusion run, and #360 is exactly
    the kind of bug that shows up as corrupted frames rather than a bad
    cosine similarity. Set `attention_backend:
    _sage_qk_int8_pv_fp8_cuda` explicitly to take it, after checking output.

    sm_100 (datacenter Blackwell) has no local kernel — SageAttention 2.2.0
    does not support it, so the image cannot build one (see the Dockerfile's
    SAGE_CUDA_ARCH_LIST). `sage` is still returned there; it raises, and the
    caller falls back to SDPA.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        return None
    if (major, minor) == (8, 9):
        return "_sage_qk_int8_pv_fp16_triton"
    return "sage"


def _mask_to_pil(m: np.ndarray) -> Image.Image:
    """Normalize a control mask to uint8 [0,255] and wrap as PIL, without
    inverting — see module docstring for why. Handles both RMBGStep's
    float32 [0,1] contract and Dataset.from_disk's raw uint8 [0,255] alpha
    channel transparently."""
    from PIL import Image

    return Image.fromarray(np.clip(normalize_mask(m) * 255.0, 0, 255).astype(np.uint8))


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_float_to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
