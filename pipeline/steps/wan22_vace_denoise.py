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

**How hard the control video pushes** is four params, not one.
`strength` is diffusers' `conditioning_scale` and is the whole of it for an
ordinary run. `strength_layers` tapers that across the eight VACE injection
layers, which diffusers already supports (`conditioning_scale` takes a list
as readily as a float) and this step merely types and validates. The other
two are the things diffusers cannot do, and they are two different axes of
the same rewrite: `strength_low` scales the LOW-NOISE expert alone, and
`strength_steps` scales each DENOISE STEP in turn — a schedule like
[1, 1, 0.75, 0.5, 0.25, 0] over a 6-step run, which lets the drawing set
the pose and then hands the frame back to the model. The pipeline builds
one scale tensor before the denoising loop and passes it to whichever
expert the timestep picks, so both overrides go in through forward
pre-hooks on the transformers rather than params — see `_vace_scale_hook`,
which explains why a pre-hook and not a `forward` wrapper, and
`_step_index` for how a hook knows which step it is on. All three default
to leaving `strength` exactly as it was, and none is a load param, so the
resident worker still serves both passes from one pipeline.

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

import contextlib
import functools
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import REQUIRED, Param, Step

logger = logging.getLogger(__name__)

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


def _conditioning_scale(
    strength: float, taper: Optional[list], n_layers: int
) -> list:
    """`strength` spread over the VACE layers, tapered by `strength_layers`.

    diffusers takes `conditioning_scale` as one float or as one value per
    entry in the transformer's `vace_layers` — [0, 5, 10, 15, 20, 25, 30, 35]
    here, the transformer layers the control latents are injected at,
    shallow to deep. It broadcasts a float to all of them, which is exactly
    what a `taper` of None keeps doing; a taper multiplies each layer's
    share of it, so a run that sets neither knob is unchanged.

    The length has to match the model, not a constant: `vace_layers` is read
    off the loaded transformer's config, so a checkpoint with a different
    injection pattern is a clear error here rather than a shape mismatch
    eight layers deep.
    """
    if taper is None:
        return [float(strength)] * n_layers
    if len(taper) != n_layers:
        raise ValueError(
            f"wan22_vace_denoise: strength_layers has {len(taper)} entries, "
            f"but this transformer injects at {n_layers} VACE layers"
        )
    return [float(strength) * float(multiplier) for multiplier in taper]


def _scale_schedule(
    strength: float,
    taper: Optional[list],
    schedule: Optional[list],
    n_layers: int,
    n_steps: int,
) -> list:
    """One per-layer scale list per denoise step: the whole run's plan.

    The second axis after `strength_layers`. That one says how hard the
    control video pushes at each DEPTH; this one says how hard it pushes at
    each TIME, and they multiply: entry [s][l] is `strength` times
    `strength_steps[s]` times `strength_layers[l]`.

    A `schedule` of None returns a single entry, not `n_steps` copies of
    one, and that shortness is load-bearing rather than an optimization —
    it is what tells the hook the scale is constant, so it never has to ask
    which step it is on and an unscheduled run stays exactly the run that
    came before this knob existed.

    Lengths are checked against the run, not a constant: `n_steps` is the
    step's own `steps` param, so a 6-entry schedule on a 4-step run is
    refused here, before any weights are touched, rather than silently
    applying the wrong step's value or running off the end.
    """
    base = _conditioning_scale(strength, taper, n_layers)
    if schedule is None:
        return [base]
    if len(schedule) != n_steps:
        raise ValueError(
            f"wan22_vace_denoise: strength_steps has {len(schedule)} entries, "
            f"but this run takes {n_steps} denoise steps"
        )
    return [
        [scale * float(multiplier) for scale in base] for multiplier in schedule
    ]


def _vace_scale_hook(step: "Wan22VaceDenoiseStep", expert: str):
    """A forward pre-hook that rewrites one expert's VACE scale per step.

    There is no per-expert knob to set, and no per-step one either.
    diffusers builds `conditioning_scale` ONCE, before the denoising loop,
    and hands that same tensor to whichever expert the timestep selects —
    `current_model(..., control_hidden_states_scale=conditioning_scale)` in
    pipeline_wan_vace.py, for both the cond and uncond calls. The call
    itself is the only seam, and this is it. One hook per expert, so
    `strength_low` is just the two of them being given different plans.

    A module pre-hook specifically, NOT a wrapper around the transformer's
    `.forward`: `enable_model_cpu_offload()` replaces `forward`, and
    `release_vram()` makes it do so again after every pass
    (`maybe_free_model_hooks()` removes accelerate's hooks and re-attaches
    them), so a wrapper installed at load time would be dropped somewhere
    between the two denoise passes and the override would silently stop
    applying. Pre-hooks run in `Module._call_impl`, ahead of whatever
    `forward` currently is, and nothing in that cycle touches them.

    The plan is read off the step per call rather than closed over, because
    the two passes share one resident pipeline and disagree about it.
    """

    def hook(module, args, kwargs):
        scales = step._scales[expert]
        if scales is None:
            return None
        incoming = kwargs.get("control_hidden_states_scale")
        if incoming is None:
            raise RuntimeError(
                "wan22_vace_denoise: strength_low or strength_steps is set, "
                "but the pipeline called the "
                f"{expert}-noise expert without control_hidden_states_scale. "
                "diffusers' WanVACEPipeline passes it on every call; a "
                "version that does not means this override no longer has a "
                "seam to work through, and refusing beats denoising at the "
                "wrong strength."
            )
        index = 0 if len(scales) == 1 else step._step_index(kwargs.get("timestep"))
        # new_tensor keeps the device and dtype diffusers already resolved
        # for the scale it built (execution device, transformer dtype).
        kwargs["control_hidden_states_scale"] = incoming.new_tensor(scales[index])
        return args, kwargs

    return hook


@register_step("wan22_vace_denoise")
class Wan22VaceDenoiseStep(Step):
    # The per-call knobs come first; everything from `checkpoint` down is
    # which weights to build the pipeline out of and how to place it, which
    # is set once for the machine rather than tuned per run — hence
    # `advanced`. Those are also exactly the LOAD_PARAMS list below, and the
    # two are meant to stay in step.
    PARAMS = (
        Param("width", int, REQUIRED, "Frame width the pipeline generates at", minimum=1),
        Param("height", int, REQUIRED, "Frame height the pipeline generates at", minimum=1),
        Param("steps", int, 6, "Diffusion steps", minimum=1),
        Param("cfg", float, 1.0, "Classifier-free guidance scale"),
        Param("seed", int, 0, "Diffusion seed"),
        Param("strength", float, 1.0,
              "VACE conditioning scale: 1.0 generates from the control video, lower "
              "values keep more of it", minimum=0.0, maximum=1.0),
        # No minimum/maximum on this one, unlike `strength` above, and the
        # reason is the UI rather than the range: a param declaring both
        # draws as a slider (webui.py's _control), a slider has no empty
        # position, and the value it would hand back for this param's None
        # default is its minimum — 0.0, which is VACE switched off on the
        # low-noise expert. A plain number box has an empty position and
        # returns None from it, which is what "same as `strength`" needs.
        Param("strength_low", float, None,
              "VACE conditioning scale for the LOW-noise expert alone; empty means "
              "whatever `strength` is. Lowering it keeps the pose lock the control "
              "video buys while the structure is being set, and gives the later "
              "steps room to paint over the drawing instead of copying it"),
        Param("strength_layers", list, None,
              "Per-layer multipliers on the two scales above, one for each VACE "
              "injection layer (8 of them, shallow to deep); empty means 1.0 at "
              "every layer, which is the plain scale"),
        Param("strength_steps", list, None,
              "Per-step multipliers on the scales above, one for each denoise step "
              "(`steps` of them, first to last); empty means 1.0 at every step. "
              "[1, 1, 0.75, 0.5, 0.25, 0] holds the control video at full scale "
              "while the pose is set and lets go of it before the last step, so "
              "the drawing steers the structure without being painted in"),
        Param("prompt", str, "",
              "Positive prompt. $SUBJECT_DESC$ in it is filled in from the "
              "`subject_desc` input (dataset.prompt)"),
        Param("negative_prompt", str, "", "Negative prompt"),
        Param("length", int, None,
              "Frames to generate; empty means as many as the control video has"),

        Param("checkpoint", str, DEFAULT_CHECKPOINT,
              "The diffusers repo the pipeline's non-transformer components come from",
              advanced=True),
        Param("fp8_repo", str, DEFAULT_FP8_REPO,
              "Repo holding the pre-quantized fp8 experts", advanced=True),
        Param("fp8_checkpoint_high", str, DEFAULT_FP8_HIGH,
              "High-noise expert: a filename in fp8_repo, or a local path",
              advanced=True),
        Param("fp8_checkpoint_low", str, DEFAULT_FP8_LOW,
              "Low-noise expert: a filename in fp8_repo, or a local path",
              advanced=True),
        Param("fp8_config", str, None,
              "Where the transformer config is read from; empty means `checkpoint`",
              advanced=True),
        Param("use_lora", bool, True, "Fuse the Lightning 4-step LoRAs", advanced=True),
        Param("lora_repo", str, DEFAULT_LORA_REPO, "LoRA repo", advanced=True),
        Param("lora_subfolder", str, DEFAULT_LORA_SUBFOLDER, "LoRA subfolder",
              advanced=True),
        Param("lora_high", str, DEFAULT_LORA_HIGH, "High-noise LoRA weights",
              advanced=True),
        Param("lora_low", str, DEFAULT_LORA_LOW, "Low-noise LoRA weights", advanced=True),
        Param("lora_strength_high", float, 1.0, "High-noise LoRA scale", advanced=True),
        Param("lora_strength_low", float, 1.0, "Low-noise LoRA scale", advanced=True),
        Param("attention_backend", str, "auto",
              "Attention implementation; auto picks per GPU architecture",
              advanced=True),
        Param("cpu_offload", bool, True,
              "Stream the two 17.58 GB transformers on and off the card per forward "
              "instead of resident-loading them", advanced=True),
        Param("device", str, "cuda", "Torch device", advanced=True),
    )

    # Which params load() actually reads. pipeline/worker.py's resident
    # worker reuses the loaded pipeline while these are unchanged and
    # rebuilds it when they are not — see load_signature() there.
    #
    # The per-call params are deliberately ABSENT: `strength`,
    # `strength_low`, `strength_layers`, `strength_steps`, `steps`, `cfg`,
    # `seed`, `prompt`, `negative_prompt`, `width`, `height`, `subject_desc`.
    # That is the whole point — fast_helical_native's two passes differ only
    # by `strength` (1.0 then 0.8), so listing it here would rebuild the
    # pipeline between them and buy nothing at all. The four strength knobs
    # reach the pipeline through the call and pre-hooks that read the step
    # per call, never through the loaded weights, so none of them needs a
    # rebuild either.
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
        # Each expert's VACE conditioning plan: a list of per-layer scale
        # lists, one entry per denoise step — or one entry in total when the
        # scale is constant, or None for "leave the scale diffusers built
        # alone". Written by run(), read per call by the pre-hooks
        # _finish_load installs — see _vace_scale_hook.
        self._scales = {"high": None, "low": None}
        # The run's timesteps, descending, cached on first use within a pass
        # and cleared at the top of the next one — see _step_index.
        self._timesteps = None

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

        checkpoint = params["checkpoint"]
        fp8_repo = params["fp8_repo"]
        fp8_high = resolve_fp8_checkpoint(
            params["fp8_checkpoint_high"], fp8_repo
        )
        fp8_low = resolve_fp8_checkpoint(
            params["fp8_checkpoint_low"], fp8_repo
        )

        # Only the geometry, from the base repo's transformer/config.json —
        # kilobytes, and the reason `transformer/config.json` survives in
        # pipeline/models.py's allow_patterns while `transformer/*` does not.
        config = load_config(params["fp8_config"] or checkpoint)
        device = params["device"]
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

        if params["use_lora"]:
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
        lora_repo = params["lora_repo"]
        lora_subfolder = params["lora_subfolder"]
        pipe.load_lora_weights(
            lora_repo,
            subfolder=lora_subfolder,
            weight_name=params["lora_high"],
            adapter_name="lightning_high",
        )
        pipe.load_lora_weights(
            lora_repo,
            subfolder=lora_subfolder,
            weight_name=params["lora_low"],
            adapter_name="lightning_low",
            load_into_transformer_2=True,
        )
        # Set the scale per component rather than via pipe.set_adapters:
        # the two adapters live in different transformers (high-noise in
        # `transformer`, low-noise in `transformer_2`), and a pipeline-level
        # call has to guess which component each name belongs to.
        pipe.transformer.set_adapters(
            ["lightning_high"], [params["lora_strength_high"]]
        )
        pipe.transformer_2.set_adapters(
            ["lightning_low"], [params["lora_strength_low"]]
        )

    def _finish_load(self, pipe, params: Dict[str, Any], device: str) -> None:
        """Attention backend + offload, split out of load() for readability."""
        attention_backend = params["attention_backend"]
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

        # Installed once, for the life of the pipeline, and inert until a
        # run sets `strength_low` or `strength_steps` — see _vace_scale_hook
        # for why they are pre-hooks and why they go on before the offload
        # hooks below. Both experts, because a per-step schedule spans the
        # whole run and the high-noise expert owns its opening steps; with
        # neither knob set each hook returns None and nothing is touched.
        for expert, transformer in (
            ("high", pipe.transformer),
            ("low", pipe.transformer_2),
        ):
            if transformer is not None:
                transformer.register_forward_pre_hook(
                    _vace_scale_hook(self, expert), with_kwargs=True
                )

        self._device = device
        self._cpu_offload = params["cpu_offload"]
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
        fast_helical_native — finds an empty card. Without this override the
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

    def _step_index(self, timestep) -> int:
        """Which denoise step the call the hook just intercepted belongs to.

        Read off the timestep the pipeline is passing the transformer rather
        than counted. The loop is `for i, t in enumerate(timesteps)` and `t`
        goes into the call as `timestep`, so the position of that value in
        `scheduler.timesteps` IS `i` — no bookkeeping to reset between the
        two passes a resident worker serves, no assumption about how many
        times an expert is called per step (classifier-free guidance calls
        it twice, at the same timestep, and both must get the same scale).

        The schedule is read from the live scheduler, which `pipe()` has by
        then called `set_timesteps` on, and cached for the rest of the pass;
        run() drops the cache before each one. A count that disagrees with
        the plan means the scheduler is not stepping the way `steps` says it
        is, and every index after the disagreement would be the wrong step's
        scale — so refuse, on the same grounds as the missing-kwarg guard.
        """
        if timestep is None:
            raise RuntimeError(
                "wan22_vace_denoise: a per-step VACE schedule is set, but the "
                "pipeline called the transformer without `timestep`, which is "
                "what says which step this is. Refusing beats denoising at "
                "the wrong strength."
            )
        if self._timesteps is None:
            timesteps = self._pipe.scheduler.timesteps
            self._timesteps = [float(value) for value in timesteps]
            planned = max(
                len(scales) for scales in self._scales.values() if scales is not None
            )
            if len(self._timesteps) != planned:
                raise RuntimeError(
                    "wan22_vace_denoise: strength_steps plans "
                    f"{planned} steps, but the scheduler is stepping "
                    f"{len(self._timesteps)} times"
                )
        # One value per frame in the batch, all of them `t` expanded.
        value = float(timestep.flatten()[0])
        # Equality is what actually holds — these are the scheduler's own
        # numbers coming back — with nearest as the tiebreak so a dtype cast
        # somewhere in diffusers cannot turn a scale into a lookup failure.
        return min(
            range(len(self._timesteps)),
            key=lambda index: abs(self._timesteps[index] - value),
        )

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

        converting = time.time()
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
        logger.info("  frame conversion: %.1fs", time.time() - converting)

        generator = torch.Generator(device=params["device"])
        generator.manual_seed(params["seed"])

        prompt = params["prompt"]
        subject_desc = inputs.get("subject_desc")
        if subject_desc and "$SUBJECT_DESC$" in prompt:
            prompt = prompt.replace("$SUBJECT_DESC$", subject_desc)

        # How hard the control video pushes, per VACE injection layer and
        # per expert. `vace_layers` is read off the model rather than
        # assumed: it is the same list on both experts (wan_fp8.py builds
        # them from one config), so either one answers.
        #
        # `strength_low` is what splits the two. With this checkpoint's
        # boundary_ratio of 0.875 and the scheduler's flow_shift of 3.0, a
        # 6-step run puts t=1000 and 937 on the high-noise expert and t=857,
        # 750, 600 and 375 on the low-noise one — four of the six steps, and
        # the four where detail is decided. A control frame here is a
        # drawing (a flat silhouette under a DWPose skeleton), so those late
        # steps at full scale are where its ink survives into the output as
        # ink instead of being read as pose.
        #
        # `strength_steps` is the other axis, and it spans the whole run
        # rather than one expert's share of it: a 6-entry schedule numbers
        # those same six steps, so its first two land on the high-noise
        # expert and its last four on the low-noise one — which is why BOTH
        # experts carry a hook, and why a schedule that fades to 0 is a
        # control video that sets the pose and then lets the model finish
        # the frame on its own.
        vace_layers = (
            pipe.transformer.config.vace_layers
            if pipe.transformer is not None
            else pipe.transformer_2.config.vace_layers
        )
        n_layers = len(vace_layers)
        n_steps = params["steps"]
        taper = params["strength_layers"]
        schedule = params["strength_steps"]
        strength_low = params["strength_low"]

        # What pipe() itself is given. With a schedule set every call is
        # rewritten and this is only the value the hooks replace, but it
        # stays the honest base so that a run setting neither per-expert nor
        # per-step knob never reaches a hook at all.
        conditioning_scale = _conditioning_scale(params["strength"], taper, n_layers)
        self._timesteps = None
        self._scales = {
            "high": (
                None
                if schedule is None
                else _scale_schedule(
                    params["strength"], taper, schedule, n_layers, n_steps
                )
            ),
            "low": (
                None
                if schedule is None and strength_low is None
                else _scale_schedule(
                    params["strength"] if strength_low is None else strength_low,
                    taper,
                    schedule,
                    n_layers,
                    n_steps,
                )
            ),
        }
        overridden = [name for name, scales in self._scales.items() if scales]
        if overridden or taper is not None:
            logger.info("  VACE scale: %s", conditioning_scale)
            for expert in overridden:
                scales = self._scales[expert]
                logger.info(
                    "    %s-noise expert: %s",
                    expert,
                    scales[0] if len(scales) == 1 else scales,
                )

        # Timings, not just a call. Everything up to the progress bar's
        # "0%" is silent otherwise, which on a resident worker's second
        # pass reads as a two-minute hang — see _PRE_LOOP_PHASES.
        started = time.time()
        with _timed_phases(pipe):
            result = pipe(
                prompt=prompt,
                negative_prompt=params["negative_prompt"],
                video=video,
                mask=masks,
                reference_images=reference_images,
                conditioning_scale=conditioning_scale,
                height=params["height"],
                width=params["width"],
                num_frames=params["length"] or len(video),
                num_inference_steps=params["steps"],
                guidance_scale=params["cfg"],
                generator=generator,
                output_type="np",
            )
        # Total minus the phases above minus the progress bar's own total
        # is the VAE *decode* of the finished latents, which is inline in
        # __call__ rather than a method and so cannot be wrapped.
        logger.info("  pipe() total: %.1fs", time.time() - started)

        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        images = [_rgb_float_to_bgr_uint8(frame) for frame in frames]
        return {"images": images}


# The phases of WanVACEPipeline.__call__ that run BEFORE the denoise loop's
# progress bar exists, in the order they run. Wrapped by _timed_phases()
# below so the two minutes between "reusing loaded" and "0%" stop being one
# opaque block. Named rather than derived because only these four can
# plausibly dominate:
#
#   encode_prompt         the T5 text encoder — 11.36 GB — coming back
#                         over PCIe under enable_model_cpu_offload before
#                         it runs a ~1s forward. release_vram() evicted it
#                         after the previous job, so every pass pays this.
#   preprocess_conditions diffusers resizing all 81 video frames AND all
#                         81 masks to width x height. CPU only, and the
#                         second time the frames get walked (run() already
#                         converted each one to PIL).
#   prepare_video_latents the expensive one. VACE splits the control video
#                         by the mask into `inactive` and `reactive` and
#                         encodes EACH through the VAE, and
#                         AutoencoderKLWan encodes in 1 + (frames - 1) // 4
#                         temporal chunks with tiling off by default — so
#                         81 frames at 720x1280 is 2 x 21 full-resolution
#                         encoder passes, plus one more for the reference
#                         image, before a single denoise step happens.
#   prepare_masks         an interpolate to latent resolution. Cheap; here
#                         to prove it is cheap.
#
# The two 17.58 GB transformers are deliberately NOT in this list: under
# cpu_offload they upload lazily on their first forward, which is inside
# the loop. If the gap between "0%" and "1/6" is the long one, that upload
# is what you are looking at, not anything timed here.
_PRE_LOOP_PHASES = (
    "encode_prompt",
    "preprocess_conditions",
    "prepare_video_latents",
    "prepare_masks",
)


def _sync() -> None:
    """Wait for the GPU before reading the clock.

    CUDA work is queued, not finished, when the Python call that submitted
    it returns. Without this the VAE encode would bill its time to
    whatever ran next, and the numbers would say prepare_video_latents is
    instant and the first denoise step takes a minute.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _timed_phases(pipe, names=_PRE_LOOP_PHASES):
    """Log how long each named phase of `pipe.__call__` took.

    Wrapping the bound methods for the duration of one call, rather than
    reimplementing __call__ here, keeps this indifferent to the diffusers
    version: a phase that has been renamed upstream is skipped (one fewer
    line in the log), and the run itself is byte-for-byte unaffected
    either way. The instance attribute shadows the class method and is
    deleted afterwards, uncovering the original.
    """
    patched = []
    for name in names:
        original = getattr(pipe, name, None)
        if original is None:
            continue

        def timed(*args, _name=name, _original=original, **kwargs):
            started = time.time()
            try:
                return _original(*args, **kwargs)
            finally:
                _sync()
                logger.info("  %s: %.1fs", _name, time.time() - started)

        setattr(pipe, name, functools.wraps(original)(timed))
        patched.append(name)
    try:
        yield
    finally:
        for name in patched:
            delattr(pipe, name)


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
