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
boundary_ratio=0.875 and UniPCMultistepScheduler — the same solver family
as the ComfyUI graph's uni_pc, though NOT the same step schedule; see "The
sampler's schedule" below. Its own `transformer/` and `transformer_2/`
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

**How hard the control video pushes** is one param: `strength`, a list
holding diffusers' `conditioning_scale` for each denoise step in turn —
[1, 1, 0.75, 0.5, 0.25, 0] over a 6-step run lets the drawing set the pose
and then hands the frame back to the model, and a single-entry [1.0] holds
it constant for the whole run the way a bare float used to. This was three
params once (`strength`, a per-expert `strength_low`, and a per-step
`strength_steps` multiplying it), which meant no single number said how
hard VACE was pushing at a given moment; they collapsed into this list on
2026-09-04. The per-EXPERT axis went with them and is not missed: the
experts split at a fixed step boundary, so anything `strength_low` could
say the schedule says too, by step index instead of by expert name.

diffusers builds one scale tensor before the denoising loop and hands it
to whichever expert the timestep picks, so a schedule can only go in
through forward pre-hooks on the transformers — see `_vace_scale_hook`,
which explains why a pre-hook and not a `forward` wrapper, and
`_step_index` for how a hook knows which step it is on. A constant
`strength` never reaches a hook at all: the value goes straight into the
call, exactly as it did before schedules existed.

`strength_layers` is the one knob beside it, and a different axis rather
than a second opinion on the same one: it tapers each step's scale across
the eight VACE injection layers, shallow to deep, which diffusers already
supports (`conditioning_scale` takes a list as readily as a float) and
this step merely types and validates. It defaults to leaving every layer
alone. Neither is a load param, so the resident worker still serves both
passes from one pipeline.

**The sampler's schedule** is two more per-call params, `sigma_schedule`
and `sampler_shift`, and their defaults reproduce the ComfyUI graph rather
than the HF repo. The graph sampled with `uni_pc` on ComfyUI's `beta`
scheduler and carried no ModelSamplingSD3 node, which leaves a Wan 2.2
model at ComfyUI's default shift of 8.0 (comfy/supported_models.py,
`WAN21_T2V.sampling_settings`). The repo's scheduler_config.json is UniPC
too, but spaced `linspace` at `flow_shift: 3.0`, and until 2026-09-07 that
is what this step ran, believing it matched. The two are not close:

    ComfyUI   beta, shift 8      t = 1000, 988 | 955, 889, 753, 448
    diffusers linspace, shift 3  t = 1000, 938 | 857, 750, 601, 376

`beta` is ComfyUI's arithmetic exactly — `beta_scheduler` in
comfy/samplers.py: Beta(0.6, 0.6) quantiles of the 1000-entry training
index, rounded, then the model's sigma shift — computed by
`_comfy_beta_sigmas` and handed to diffusers' own `set_timesteps` as custom
`sigmas`, which its flow-sigma branch shifts by `flow_shift` afterwards, the
same order of operations. It goes in through a wrapper on the scheduler's
`set_timesteps` (`_install_sigma_schedule`) because `pipe()` calls that
method itself with a step count and nothing else, so there is no argument
to pass a schedule through. `linspace` leaves the call alone and is the
pre-2026-09-07 run at `sampler_shift: 3.0`.

One consequence: the expert split is why `steps_high`/`steps_low` are
counts rather than the checkpoint's `boundary_ratio` of 0.875. At shift 8
that threshold would put FOUR of six steps on the high-noise expert (1000,
988, 955 and 889 are all >= 875), where the graph split at step 2
(`KSamplerAdvanced`'s end_at_step) — the count is what the graph said, so
the count is what is set.

**The solver and the hand-off** are matched too, by three more per-call
params, all defaulting to the graph's behaviour (2026-09-08):

  * `solver_variant` / `solver_order`: ComfyUI's `uni_pc` sampler is
    UniPC's bh1 variant; the repo's scheduler config is bh2 at order 2.
    Both are plain config values the scheduler reads inside `step()`, so
    `_configure_sampler` writes them with the shift. `solver_order` is a
    CAP: `sample_unipc` sets `order = min(3, len(timesteps) - 2)` per
    sampler, and each KSamplerAdvanced owns only its slice of the
    schedule, so the graph's 2-step high-noise sampler ran at order 1 and
    its 4-step low-noise one at 3 (`_phase_order`). Per step, the graph
    took orders 1, 1 | 1, 2, 2, 1; diffusers left alone takes 1, 2 | 3, 3,
    2, 1 — verified on the real scheduler, both. bh1 also needs ComfyUI's
    terminal-sigma substitution (`COMFY_TERMINAL_SIGMA`), applied by the
    `set_timesteps` wrapper: at a terminal sigma of 0 diffusers' bh1 last
    step is non-finite.
  * `handoff_reset`: the graph's second KSamplerAdvanced is a NEW sampler
    — it starts the low-noise expert with an empty multistep history, so
    step 3 is a first-order step and the order ramps up again over steps 4
    and 5, and the first sampler ENDS, so its last step is first-order
    too (`lower_order_final` counts the steps its own sampler has left).
    diffusers runs one loop and carries the history across, so without
    this the low-noise expert's first step is corrected by x0 predictions
    the HIGH-noise expert made at t=1000 and 988: a different network,
    trained on a different noise range, extrapolated into a step neither
    was trained for. Two pre-hooks do it: `_phase_end_hook` on
    `transformer` drops the order to 1 for the last high-noise step, and
    `_handoff_hook` on `transformer_2`, on its first forward of a pass,
    sets the low-noise sampler's order and clears what `set_timesteps`
    initialises (`model_outputs`, `timestep_list`, `lower_order_nums`,
    `last_sample`), leaving `_step_index` alone. That first low-noise step
    is also where the skeleton ink is committed (see the strength notes
    above), so this is not cosmetic. Off, the run is diffusers' continuous
    loop at the cap — the run before 2026-09-08.

**The reference image** is the last seam, `reference_fit`. diffusers'
`preprocess_conditions` LETTERBOXES a reference: scales it to fit inside
the frame and pads the rest with WHITE (a canvas of ones, in [-1, 1]
space) — a 768x1536 back view becomes 640x1280 between 40 px white bars.
ComfyUI's `WanVaceToVideo` runs `common_upscale(..., "center")` instead:
crop the reference to the frame's aspect about its centre, then resize it
to fill (comfy/utils.py), so the same back view loses 85 px top and
bottom and fills the frame. `crop` (the default) does ComfyUI's arithmetic
in `_fit_reference` before the image reaches diffusers, which then finds
nothing to scale or pad; `letterbox` hands it over untouched, the run
before 2026-09-08.

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

# ComfyUI's `beta` scheduler (comfy/samplers.py, `beta_scheduler`): the
# denoise steps are Beta(alpha, beta) quantiles of the training index.
COMFY_BETA_ALPHA = 0.6
COMFY_BETA_BETA = 0.6
# The sigma shift a Wan 2.2 model gets in ComfyUI when the graph carries no
# ModelSamplingSD3 node (comfy/supported_models.py, WAN21_T2V's
# `sampling_settings`) — and so what the reference graph sampled at. The HF
# repo's scheduler_config.json says 3.0.
COMFY_WAN_SHIFT = 8.0
# What ComfyUI's `sample_unipc` puts in place of a terminal sigma of 0
# (`timesteps[-1] = 0.001`, comfy/extra_samplers/uni_pc.py). Needed here
# for the bh1 variant: diffusers' bh1 update uses B(h) = h, and at sigma 0
# h is infinite, so the last step comes out non-finite — measured on the
# real scheduler for both hand-off modes. bh2 (B(h) = expm1(h)) stays
# finite at 0 and is left with diffusers' own terminal sigma, which keeps
# the pre-2026-09-08 run byte-identical.
COMFY_TERMINAL_SIGMA = 0.001


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


def _total_steps(params: Dict[str, Any]) -> int:
    """How many denoise steps this run takes: the two experts' shares, summed.

    There is no total to set. `steps_high` and `steps_low` are the whole of
    it, because the number that mattered was never the total: it was how
    many steps each expert got, and with a single `steps` param that was
    decided for you, by the scheduler's timesteps falling either side of the
    checkpoint's boundary_ratio. `steps: 6` happened to mean 2 high and 4
    low; `steps: 5` would have meant 1 and 4, and nothing said so.
    """
    high, low = int(params["steps_high"]), int(params["steps_low"])
    # `minimum` on a Param is a UI hint, not a check `resolve_params` runs,
    # so the floor is enforced here — a negative share would quietly shorten
    # the run rather than fail it.
    if high < 0 or low < 0:
        raise ValueError(
            f"wan22_vace_denoise: steps_high={high}, steps_low={low} — an "
            "expert cannot take a negative number of steps"
        )
    if high + low < 1:
        raise ValueError(
            "wan22_vace_denoise: steps_high and steps_low are both 0 — a run "
            "has to take at least one denoise step"
        )
    return high + low


def _expert_boundary_ratio(
    timesteps: list, steps_high: int, num_train_timesteps: int
) -> float:
    """The `boundary_ratio` that puts the first `steps_high` steps on the
    high-noise expert.

    diffusers picks the expert per step by comparing that step's timestep
    against `boundary_ratio * num_train_timesteps`: `t >= boundary` is the
    high-noise expert, below it the low-noise one (pipeline_wan_vace.py's
    denoising loop). So the split is not a count anywhere in diffusers — it
    is a threshold, and which count it produces depends on where the
    scheduler's timesteps happen to land, which depends on the step count
    and the flow_shift. Asking for a count and computing the threshold that
    delivers it turns that round the right way.

    The threshold goes MIDWAY between the last high step and the first low
    one, so it is the split furthest from either neighbour: a float cast
    somewhere in diffusers cannot move a step across it.

    The checkpoint's own 0.875 is what the old sampler (linspace, shift
    3.0: t = 1000, 937 | 857, 750, 600, 375) resolved to 2/4 under — the
    default 2/4 computes a boundary of 897 there and selects the same
    experts for the same steps. Under the sampler the step runs now (beta,
    shift 8: t = 1000, 988 | 955, 889, 753, 448) that same 0.875 would hand
    the high-noise expert FOUR steps, which is not what the reference graph
    did; it split by count, at step 2, and so does this.
    """
    if steps_high >= len(timesteps):
        return 0.0  # every t >= 0: the high-noise expert takes the run
    if steps_high <= 0:
        # Above the first (largest) timestep, so no step is ever >= it.
        return (timesteps[0] + 1.0) / num_train_timesteps
    last_high, first_low = timesteps[steps_high - 1], timesteps[steps_high]
    if not last_high > first_low:
        raise ValueError(
            f"wan22_vace_denoise: cannot split after step {steps_high} — the "
            f"scheduler's timesteps are {last_high} then {first_low}, which "
            "leaves nowhere to put a boundary between them"
        )
    return ((last_high + first_low) / 2.0) / num_train_timesteps


def _comfy_beta_sigmas(
    n_steps: int,
    num_train_timesteps: int = 1000,
    alpha: float = COMFY_BETA_ALPHA,
    beta: float = COMFY_BETA_BETA,
) -> list:
    """The sigmas ComfyUI's `beta` scheduler picks for an `n_steps` run,
    BEFORE the model's shift.

    comfy/samplers.py's `beta_scheduler`, line for line: `n_steps` quantile
    levels descending from 1 (`1 - linspace(0, 1, n, endpoint=False)`), each
    mapped through the Beta(alpha, beta) quantile function, scaled to the
    last index of the model's 1000-entry sigma table and rounded. ComfyUI
    then reads `model_sampling.sigmas[index]`, and for a Wan model that
    table is `shift((index + 1) / 1000)` (comfy/model_sampling.py,
    `ModelSamplingDiscreteFlow.set_parameters`) — so the unshifted value at
    an index is `(index + 1) / num_train_timesteps`, and the shift is left
    to the scheduler these are handed to: diffusers' flow-sigma branch
    applies `flow_shift` to custom sigmas, the same order ComfyUI does it in.

    scipy for the quantile function because ComfyUI uses scipy's, and
    matching its rounding to the index is the whole point. venv_wan22 sees
    venv_base's copy (docker/make-child-venv.sh).

    ComfyUI silently drops a repeated index, which shortens the run. That is
    refused here instead: everything else in this step plans against the
    step count it was given (`strength` has one entry per step, the expert
    split is placed by count), and a run of fewer steps than planned would
    misplace both.
    """
    try:
        from scipy.stats import beta as beta_distribution
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError(
            "wan22_vace_denoise: the `beta` sigma schedule needs scipy (it is "
            "ComfyUI's own quantile function); install it, or set "
            "sigma_schedule: linspace"
        ) from exc
    if n_steps < 1:
        raise ValueError("wan22_vace_denoise: a schedule needs at least one step")
    levels = 1.0 - np.linspace(0.0, 1.0, n_steps, endpoint=False)
    index = np.rint(
        beta_distribution.ppf(levels, alpha, beta) * (num_train_timesteps - 1)
    )
    if len(np.unique(index)) != len(index):
        raise ValueError(
            f"wan22_vace_denoise: {n_steps} steps on the beta schedule put two "
            "steps on the same timestep, which ComfyUI would silently drop; use "
            "fewer steps, or sigma_schedule: linspace"
        )
    return [float(value) for value in (index + 1.0) / num_train_timesteps]


def _install_sigma_schedule(step: "Wan22VaceDenoiseStep", scheduler) -> None:
    """Route the run's sigma schedule into the scheduler's `set_timesteps`.

    `pipe()` calls `self.scheduler.set_timesteps(num_inference_steps,
    device=device)` itself, and `_set_expert_split` calls it the same way
    before the pass; neither hands sigmas across. diffusers does accept
    them — `set_timesteps(..., sigmas=...)` is its documented way to run a
    custom schedule — so the one seam is the call, and this wraps it on the
    instance: a `beta` run computes ComfyUI's sigmas and passes them in, a
    `linspace` run passes nothing and diffusers spaces the steps itself.
    Both then go through diffusers' own shift, eps and final-sigma handling.

    The schedule is read off the step per call (`_sigma_schedule`), the way
    the scale hooks read `_scales`, so the two passes of one resident
    pipeline can differ without reinstalling anything. Installed once per
    scheduler object; `_configure_sampler` remembers which.

    `sigmas` on UniPC's `set_timesteps` is recent: diffusers 0.36.0 does
    not have it (checked against the release), the 0.41 development tree
    does. A scheduler without it is refused at the first `beta` call rather
    than run on the linspace schedule while the log says beta — the guard
    is by signature, so it is the installed diffusers that answers, not a
    version table here.
    """
    import inspect

    original = scheduler.set_timesteps
    accepts_sigmas = "sigmas" in inspect.signature(original).parameters

    @functools.wraps(original)
    def set_timesteps(num_inference_steps=None, device=None, sigmas=None, **kwargs):
        if sigmas is None and step._sigma_schedule == "beta":
            sigmas = _comfy_beta_sigmas(
                int(num_inference_steps), int(scheduler.config.num_train_timesteps)
            )
        if sigmas is not None:
            if not accepts_sigmas:
                raise RuntimeError(
                    "wan22_vace_denoise: this diffusers' UniPCMultistepScheduler."
                    "set_timesteps takes no `sigmas`, so the beta schedule cannot "
                    "reach it; upgrade diffusers (0.36.0 lacks it) or set "
                    "sigma_schedule: linspace, which only needs the shift"
                )
            # An array, whatever the signature says (`list[float]`): the
            # flow-sigma branch does `flow_shift * sigmas / (...)` on it
            # directly, and a list raises there. Verified against 0.41.
            kwargs["sigmas"] = np.asarray(sigmas, dtype=np.float64)
        result = original(num_inference_steps, device=device, **kwargs)
        # ComfyUI's terminal-sigma substitution, for the variant that
        # needs it — see COMFY_TERMINAL_SIGMA. The timesteps are untouched:
        # only where the last step lands changes, from 0 to 0.001.
        sigmas_out = getattr(scheduler, "sigmas", None)
        if (
            step._solver_variant == "bh1"
            and sigmas_out is not None
            and len(sigmas_out)
            and float(sigmas_out[-1]) == 0.0
        ):
            sigmas_out[-1] = COMFY_TERMINAL_SIGMA
        return result

    scheduler.set_timesteps = set_timesteps


def _history_slots(scheduler) -> int:
    """How many previous model outputs the scheduler keeps: its
    `solver_order`, read off the config so a change to the order made
    after construction is honoured."""
    order = getattr(getattr(scheduler, "config", None), "solver_order", None)
    if order is None:
        order = len(getattr(scheduler, "model_outputs", ()))
    return max(int(order), 1)


def _restart_multistep(scheduler) -> None:
    """Empty the scheduler's multistep history, as a new sampler starts.

    Exactly the fields UniPCMultistepScheduler's own `set_timesteps` (and
    `__init__`) initialise for the history: the retained model outputs and
    their timesteps, the warm-up counter that ramps the order, and the
    sample the corrector uses. `_step_index` is deliberately kept — the run
    is still on the same step of the same schedule; only what it may
    extrapolate from is gone. Attributes are only reset where they exist,
    so a scheduler without one field cannot be given a stray attribute.

    The lists are sized from the CONFIG's solver_order, not from their
    current length: `set_timesteps` rebuilds `model_outputs` from the
    config but leaves `timestep_list` at the length `__init__` gave it, so
    a solver_order raised after construction (which `_configure_sampler`
    does) otherwise leaves `step()` indexing past the shorter list.
    Measured on the real scheduler: order 3 over a 2-slot `timestep_list`
    raises IndexError on the first step.
    """
    slots = _history_slots(scheduler)
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * slots
    if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * slots
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None


def _phase_order(order_cap: int, phase_steps: int) -> int:
    """The UniPC order one of the graph's two samplers ran at.

    comfy/extra_samplers/uni_pc.py's `sample_unipc` sets
    `order = min(3, len(timesteps) - 2)`, and a KSamplerAdvanced hands it
    the slice of the schedule it owns plus the boundary sigma — so a
    sampler of `phase_steps` steps sees `phase_steps + 1` timesteps and
    runs at `min(cap, phase_steps - 1)`: ORDER 1 for the graph's 2-step
    high-noise sampler, 3 for its 4-step low-noise one. Floor 1, since a
    single step still has to be taken.
    """
    return max(1, min(int(order_cap), int(phase_steps) - 1))


def _nearest_step(scheduler, timestep) -> Optional[int]:
    """Which step of the live schedule `timestep` is, by nearest value —
    the lookup `_step_index` does, without its plan bookkeeping."""
    if timestep is None:
        return None
    timesteps = [float(value) for value in scheduler.timesteps]
    if not timesteps:
        return None
    value = float(timestep.flatten()[0])
    return min(range(len(timesteps)), key=lambda index: abs(timesteps[index] - value))


def _phase_end_hook(step: "Wan22VaceDenoiseStep"):
    """A forward pre-hook for the HIGH-noise expert: make its last step
    first-order, as the end of the graph's first sampler was.

    `lower_order_final` caps a step's order by how many steps REMAIN, and
    in ComfyUI that count belongs to the sampler at hand: the first
    KSamplerAdvanced's final step has one step remaining and runs at order
    1 whatever its cap. diffusers counts to the end of the whole run and
    would not cap it. Setting `solver_order` to 1 before that step's
    forward is enough — `step()` reads the config per call — and the
    hand-off hook restores the low-noise sampler's order right after.
    Fires only while the hand-off is pending, so it is inert once the
    low-noise expert has taken over, and inert when `handoff_reset` is off.
    """

    def hook(module, args, kwargs):
        if not step._handoff_pending:
            return None
        scheduler = step._pipe.scheduler
        index = _nearest_step(scheduler, kwargs.get("timestep"))
        if index is None or index != step._steps_high - 1:
            return None
        if getattr(scheduler.config, "solver_order", 1) != 1:
            scheduler.register_to_config(solver_order=1)
            logger.info("  hand-off: the high-noise sampler's final step is first-order")
        return None

    return hook


def _handoff_hook(step: "Wan22VaceDenoiseStep"):
    """A forward pre-hook for the LOW-noise expert: the first time it runs
    in a pass, give it the graph's second sampler — its own order, and an
    empty multistep history.

    A pre-hook rather than anything in the loop, for the same reason as
    `_vace_scale_hook`: the transformer's forward is the only seam the
    pipeline offers, and a module pre-hook survives the offload wrappers.
    Before the forward is early enough — the scheduler only reads its
    history and order in `step()`, after the forward returns.
    `_handoff_pending` is armed by `_configure_sampler` and disarmed on
    the first firing, so a run with classifier-free guidance (two forwards
    per step) resets once.
    """

    def hook(module, args, kwargs):
        if not step._handoff_pending:
            return None
        step._handoff_pending = False
        scheduler = step._pipe.scheduler
        order = _phase_order(step._solver_order, step._steps_low)
        scheduler.register_to_config(solver_order=order)
        _restart_multistep(scheduler)
        logger.info(
            "  hand-off: multistep history restarted for the low-noise expert, order %d",
            order,
        )
        return None

    return hook


def _fit_reference(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """ComfyUI's `common_upscale(image, width, height, "bilinear", "center")`.

    Crop to the target aspect about the centre — comfy/utils.py's own
    arithmetic, Python `round` included — then resize to exactly
    `width` x `height`. `cv2.INTER_LINEAR` is torch's `bilinear` without
    antialiasing, half-pixel centres on both sides. The result fits the
    frame exactly, so diffusers' `preprocess_conditions` finds nothing to
    scale or pad.
    """
    old_height, old_width = image.shape[:2]
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = image[y:old_height - y, x:old_width - x]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _scale_schedule(
    strength: list, taper: Optional[list], n_layers: int, n_steps: int
) -> list:
    """The whole run's plan: one per-layer scale list per denoise step.

    The two axes, resolved. `strength` says how hard the control video
    pushes at each TIME, `strength_layers` at each DEPTH, and they
    multiply: entry [s][l] is `strength[s]` times `strength_layers[l]`.

    A single-entry `strength` returns a single entry, not `n_steps` copies
    of one, and that shortness is load-bearing rather than an optimization
    — it is what tells run() the scale is constant, so the value can go
    straight into the call and no hook has to ask which step it is on.

    The length is checked against the run, not a constant: `n_steps` is the
    step's own `steps` param, so a 6-entry schedule on a 4-step run is
    refused here, before any weights are touched, rather than silently
    applying the wrong step's value or running off the end.
    """
    if len(strength) not in (1, n_steps):
        raise ValueError(
            f"wan22_vace_denoise: strength has {len(strength)} entries, but "
            f"this run takes {n_steps} denoise steps — give one scale per "
            "step, or a single one to hold it constant"
        )
    return [_conditioning_scale(scale, taper, n_layers) for scale in strength]


def _vace_scale_hook(step: "Wan22VaceDenoiseStep", expert: str):
    """A forward pre-hook that rewrites the VACE scale per denoise step.

    There is no per-step knob to set. diffusers builds `conditioning_scale`
    ONCE, before the denoising loop, and hands that same tensor to whichever
    expert the timestep selects —
    `current_model(..., control_hidden_states_scale=conditioning_scale)` in
    pipeline_wan_vace.py, for both the cond and uncond calls. The call
    itself is the only seam, and this is it. Both experts carry a hook
    because a schedule spans the whole run and they split it between them:
    at the default 2/4 the high-noise expert takes steps 1-2 and the
    low-noise one 3-6, and `steps_high`/`steps_low` move that. They read the
    same plan; `expert` is here to name which one went wrong in the guard
    below.

    A module pre-hook specifically, NOT a wrapper around the transformer's
    `.forward`: the offload machinery owns `forward`. diffusers' group
    offloading installs itself by wrapping it through a HookRegistry, and
    the model-level offload this replaced went further still — it replaced
    `forward` again on every `maybe_free_model_hooks()`, i.e. after every
    pass. Either way a wrapper installed at load time is wrapping something
    that is not the last word, and the override would silently stop
    applying. Pre-hooks run in `Module._call_impl`, ahead of whatever
    `forward` currently is, and nothing in that cycle touches them — which
    also means this hook runs before the group's weights are onloaded, and
    it must therefore not touch them. It doesn't: it rewrites one kwarg.

    The plan is read off the step per call rather than closed over, because
    the two passes share one resident pipeline and disagree about it.
    """

    def hook(module, args, kwargs):
        scales = step._scales
        if scales is None:
            return None
        incoming = kwargs.get("control_hidden_states_scale")
        if incoming is None:
            raise RuntimeError(
                "wan22_vace_denoise: strength schedules a scale per denoise "
                "step, but the pipeline called the "
                f"{expert}-noise expert without control_hidden_states_scale. "
                "diffusers' WanVACEPipeline passes it on every call; a "
                "version that does not means this override no longer has a "
                "seam to work through, and refusing beats denoising at the "
                "wrong strength."
            )
        index = step._step_index(kwargs.get("timestep"))
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
        Param("steps_high", int, 2,
              "Denoise steps the HIGH-noise expert takes — the opening steps, "
              "where the frame's structure is decided. The two experts' step "
              "counts are set separately because that split is the thing being "
              "chosen; their sum is the run's total (2 + 4 = the 6 steps this "
              "was calibrated at, and the same steps each expert had then)",
              minimum=0),
        Param("steps_low", int, 4,
              "Denoise steps the LOW-noise expert takes — the closing steps, "
              "where detail is decided, and the ones a `strength` taper is "
              "usually spent on",
              minimum=0),
        Param("sigma_schedule", str, "beta",
              "How the denoise steps are spaced before the shift. `beta` is "
              "ComfyUI's beta scheduler — Beta(0.6, 0.6) quantiles of the "
              "training index, what the reference graph sampled on; `linspace` "
              "is diffusers' own even spacing, what this step ran before "
              "2026-09-07",
              choices=("beta", "linspace")),
        Param("sampler_shift", float, COMFY_WAN_SHIFT,
              "Flow-matching sigma shift — how far the steps crowd toward the "
              "noisy end. 8.0 is what ComfyUI gives a Wan 2.2 model with no "
              "ModelSamplingSD3 node, and so what the reference graph ran at; "
              "the HF scheduler config's 3.0 is what this step ran before "
              "2026-09-07. With the beta schedule at 6 steps: 8 -> t = 1000, "
              "988 | 955, 889, 753, 448; 3 -> 1000, 968 | 888, 751, 534, 233",
              minimum=1.0),
        Param("solver_variant", str, "bh1",
              "UniPC's B(h) variant. ComfyUI's uni_pc sampler is bh1, which "
              "the reference graph ran; the HF scheduler config says bh2",
              choices=("bh1", "bh2")),
        Param("solver_order", int, 3,
              "UniPC's multistep order cap — how many previous model outputs a "
              "step may extrapolate from. With handoff_reset on, each expert's "
              "sampler runs at min(cap, its steps - 1) as ComfyUI's uni_pc does: "
              "1 for the 2-step high-noise sampler, 3 for the 4-step low-noise "
              "one. The HF scheduler config says 2, flat",
              minimum=1, maximum=3),
        Param("handoff_reset", bool, True,
              "Run the two experts as the reference graph's two KSamplerAdvanced "
              "nodes did: the high-noise sampler's last step is first-order, and "
              "the low-noise expert starts with an empty multistep history at "
              "its own order. Off, the run is one continuous loop at the cap and "
              "the low-noise expert's first step is corrected by the high-noise "
              "expert's x0 predictions — what this step did before 2026-09-08"),
        Param("reference_fit", str, "crop",
              "How the reference image is fitted to the frame. `crop` is "
              "ComfyUI's common_upscale center: crop to the frame's aspect, "
              "resize to fill; `letterbox` is diffusers' own: fit inside, pad "
              "with white — the run before 2026-09-08",
              choices=("crop", "letterbox")),
        Param("cfg", float, 1.0, "Classifier-free guidance scale"),
        Param("seed", int, 0, "Diffusion seed"),
        Param("strength", list, [1.0],
              "VACE conditioning scale, one entry per denoise step (`steps` of "
              "them, first to last); a single entry holds it constant for the "
              "whole run. 1.0 generates from the control video, lower values keep "
              "more of it. [1, 1, 0.75, 0.5, 0.25, 0] holds the control video at "
              "full scale while the pose is set and lets go of it before the last "
              "step, so the drawing steers the structure without being painted in"),
        Param("strength_layers", list, None,
              "Per-layer multipliers on the scale above, one for each VACE "
              "injection layer (8 of them, shallow to deep); empty means 1.0 at "
              "every layer, which is the plain scale"),
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
              "Stream the weights on and off the card a block at a time instead of "
              "resident-loading them; off needs a card that fits a whole expert "
              "plus the activations", advanced=True),
        Param("offload_blocks_per_group", int, 1,
              "How many transformer blocks ride onto the card together under "
              "cpu_offload. 1 holds the least and is the default; raising it "
              "trades VRAM for fewer, larger transfers", advanced=True,
              minimum=1),
        Param("device", str, "cuda", "Torch device", advanced=True),
    )

    # Which params load() actually reads. pipeline/worker.py's resident
    # worker reuses the loaded pipeline while these are unchanged and
    # rebuilds it when they are not — see load_signature() there.
    #
    # The per-call params are deliberately ABSENT: `strength`,
    # `strength_layers`, `steps_high`, `steps_low`, `sigma_schedule`,
    # `sampler_shift`, `solver_variant`, `solver_order`, `handoff_reset`,
    # `reference_fit`, `cfg`, `seed`, `prompt`, `negative_prompt`, `width`,
    # `height`, `subject_desc`. That is the whole point —
    # fast_helical_native's two passes differ only by `strength`, so listing
    # it here would rebuild the pipeline between them and buy nothing at
    # all. Both strength knobs reach the pipeline through the call and
    # pre-hooks that read the step per call, never through the loaded
    # weights, so neither needs a rebuild either.
    LOAD_PARAMS = (
        "checkpoint", "fp8_repo", "fp8_checkpoint_high", "fp8_checkpoint_low",
        "fp8_config", "use_lora", "lora_repo", "lora_subfolder",
        "lora_high", "lora_low", "lora_strength_high", "lora_strength_low",
        "attention_backend", "cpu_offload", "offload_blocks_per_group", "device",
    )

    def __init__(self) -> None:
        self._pipe = None
        # Remembered from _finish_load: release_vram() has to undo whichever
        # placement load() chose, and they need opposite treatment.
        self._device = "cuda"
        self._cpu_offload = True
        self._blocks_per_group = 1
        # The run's VACE conditioning plan: a list of per-layer scale lists,
        # one entry per denoise step — or None when the scale is constant,
        # which is "leave the scale diffusers built alone". Written by run(),
        # read per call by the pre-hooks _finish_load installs — see
        # _vace_scale_hook.
        self._scales = None
        # The run's timesteps, descending, cached on first use within a pass
        # and cleared at the top of the next one — see _step_index.
        self._timesteps = None
        # The run's step spacing, read per call by the wrapper
        # _install_sigma_schedule puts on the scheduler; `_scheduled` is the
        # scheduler object that wrapper is on, so it goes on once.
        self._sigma_schedule = "beta"
        self._scheduled = None
        # The hand-off plan: armed by _configure_sampler when
        # `handoff_reset` is on and both experts take steps, disarmed by the
        # low-noise expert's pre-hook the first time it fires — see
        # _phase_end_hook and _handoff_hook, which read the rest.
        self._handoff_pending = False
        self._steps_high = 0
        self._steps_low = 0
        self._solver_order = 3
        self._solver_variant = "bh1"

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
        # run gives `strength` more than one entry — see _vace_scale_hook
        # for why they are pre-hooks and why they go on before the offload
        # hooks below. Both experts, because a per-step schedule spans the
        # whole run and the high-noise expert owns its opening steps; with a
        # constant scale each hook returns None and nothing is touched.
        for expert, transformer in (
            ("high", pipe.transformer),
            ("low", pipe.transformer_2),
        ):
            if transformer is not None:
                transformer.register_forward_pre_hook(
                    _vace_scale_hook(self, expert), with_kwargs=True
                )
        # The hand-off: the high-noise expert's sampler ends (its last step
        # first-order), the low-noise expert's begins (its own order, no
        # history) — see _phase_end_hook and _handoff_hook.
        if pipe.transformer is not None:
            pipe.transformer.register_forward_pre_hook(
                _phase_end_hook(self), with_kwargs=True
            )
        if pipe.transformer_2 is not None:
            pipe.transformer_2.register_forward_pre_hook(
                _handoff_hook(self), with_kwargs=True
            )

        self._device = device
        self._cpu_offload = params["cpu_offload"]
        self._blocks_per_group = params["offload_blocks_per_group"]
        if self._cpu_offload:
            self._apply_group_offload(pipe, device)
        else:
            pipe.to(device)

        self._pipe = pipe

    def _apply_group_offload(self, pipe, device: str) -> None:
        """Place the pipeline a block at a time, not a model at a time.

        This used to be `pipe.enable_model_cpu_offload()`, which moves one
        whole component onto the card when its forward is called and leaves
        it there until the next component runs. That needs room for the
        largest component plus that component's activations at once, and on
        a 32 GB card this model does not have it. Measured, on a 5090, at
        720x1280x81 frames:

            one fp8 expert + its live LoRA        ~17.0 GiB
            the retained VACE hints                 6.3 GiB
            working set (hidden/control states,
            the fp32 upcasts, the FFN intermediate) ~4   GiB
            allocator fragmentation                 1.9 GiB
                                                  ---------
                                                  ~29.2 GiB  of 31.36

        and the next 1.51 GiB upcast in WanVACETransformerBlock is what
        actually raised OutOfMemoryError. Note the sequence is longer than
        the frame count suggests: `reference_images` buys a whole extra VAE
        temporal chunk (pipeline_wan_vace.py builds latents for
        `num_frames + num_reference_images * 4`), so 81 frames is 22 latent
        frames, 22 x 80 x 45 = 79,200 tokens, and an fp32 tensor over that
        at dim 5120 is exactly the 1.51 GiB that failed.

        The 6.3 GiB line is structural rather than incidental and is why
        trimming the working set would not have been enough:
        transformer_wan_vace.py builds `control_hidden_states_list`, one
        full-sequence tensor per VACE injection layer — eight of them, 811
        MiB each — before the main blocks run, and holds every one alive
        across all 40 of them.

        Group offloading takes the weights out of that sum instead. Only
        the blocks being executed are on the card, so what stays is
        activations plus a block or two, and the whole 17 GiB of weights
        stops competing with them.

        **`use_stream` must stay off, and that is not a tuning choice.**
        The first attempt at this streamed — one block ahead, group size
        forced to 1 — and shipped as `6597daa`, and the next 5090 run OOMed
        in the same place with the offload demonstrably active. Streaming
        offloads *nothing at all* when the weights are torchao tensors,
        which ours are: wan_fp8.py builds real `Float8Tensor` params, so
        `_is_torchao_tensor` is true and `_offload_to_memory` takes its
        `_restore_torchao_tensor` branch, which under a stream restores
        from a `cpu_param_dict` the onload's `swap_tensors` has already
        turned into CUDA data. `ModuleGroup.offload_` logs
        `['cuda:0'] -> ['cuda:0']`; the wrapper is a
        `_make_wrapper_subclass` with no storage of its own, so the
        `.qdata`/`.scale` holding every byte never leave the card. The pod
        ledger reads back exactly that: 23.05 GiB allocated, being one
        resident expert plus the 6.3 GiB of hints.

        It is not merely a sizing problem either — streaming cannot
        complete a run on any card. That same corrupted cache means the
        SECOND forward, denoise step 2, raises `cannot pin
        'CUDAFloat8_e4m3fnType'`. The 5090 only ran out of memory first.

        Measured in-image on a 4070 Ti, 8 blocks of Float8Tensor weights
        built the way wan_fp8.py builds them, 128 MiB of fp8 in total:

            blocks_per_group=1, use_stream=True   128.0 MiB LEFT ON GPU,
                                                  forward 2 raises
            blocks_per_group=1, no stream           0.0 MiB left, peak 185 MiB
            blocks_per_group=2, no stream           0.0 MiB left, peak 201 MiB
            blocks_per_group=4, no stream           0.0 MiB left, peak 233 MiB

        Unstreamed group size 1 is the best of them on both axes at once,
        which is why it is the default: it frees everything and it holds
        the least. Bigger groups only trade VRAM away for fewer, larger
        transfers.

        The placement itself was verified in-image on a 4070 Ti against a
        scaled-down WanVACETransformer3DModel (24 layers, 4 VACE layers,
        489M params — the real geometry) at 953 MiB resident vs 183 MiB
        grouped, bitwise equal output. That check used ORDINARY weights,
        which is precisely why it missed the above: it exercised the real
        shapes but not the real weight representation, and the torchao
        branch it depended on never ran. A stand-in for this model needs
        `Float8Tensor` params, and the assertion that matters is
        `weight.qdata.device` after a forward — not peak allocation, which
        a leak this shape flatters.

        **It has to go on every component.** diffusers refuses to mix this
        with the pipeline-level offload — `enable_model_cpu_offload()`
        opens with `_maybe_raise_error_if_group_offload_active(raise_error=
        True)` — so there is no arrangement where the transformers are
        group-offloaded and the text encoder keeps the old treatment. A
        component left out of this loop would stay wherever
        `from_pretrained` put it, which is the CPU, and fail on its first
        forward.

        The cost is that the weights now cross PCIe once per forward
        instead of once per pass: 16.4 GiB per step rather than per expert
        turn, and with no stream to hide it behind the compute, all of it
        exposed. At a warm x16 link that is order a second per step —
        against a denoise pass that takes minutes, and against a run that
        does not finish at all otherwise. It is affordable here
        specifically because the sequence is so long: a block is ~341 MiB
        of transfer against attention over 79,200 tokens at dim 5120,
        order 10^14 FLOPs. Do not carry that conclusion to a short clip;
        at a fraction of the sequence length the arithmetic inverts and
        this becomes the bottleneck rather than a rounding error.
        """
        import torch
        from diffusers.hooks import apply_group_offloading

        # No stream, ever — see the docstring. It is not a knob because
        # there is no setting of it that works: streaming is what makes
        # diffusers restore a torchao weight from a cache the onload has
        # already overwritten with device data, which offloads nothing and
        # then dies on the second forward. `offload_blocks_per_group` is
        # therefore a plain residency dial, low is good, and 1 is both the
        # least resident and the fastest to free.
        blocks = self._blocks_per_group
        onload = torch.device(device)
        offload = torch.device("cpu")
        # Walked rather than named: `pipe.components` is what diffusers
        # itself would have hooked, so a version that gains a component
        # gets placed too instead of being left behind on the CPU with
        # nothing to say so until its first forward. The non-modules in
        # there (tokenizer, scheduler) hold no weights and are skipped.
        for module in pipe.components.values():
            if not isinstance(module, torch.nn.Module):
                continue
            apply_group_offloading(
                module,
                onload_device=onload,
                offload_device=offload,
                offload_type="block_level",
                num_blocks_per_group=blocks,
                use_stream=False,
                # Inert with the stream off — diffusers builds the pinned
                # `cpu_param_dict` this guards only inside `if self.stream
                # is not None`, so an unstreamed onload copies from the
                # module's own pageable storage and caches nothing. Kept
                # because the constraint it states is permanent: the
                # default pre-pins, and for ~47 GB already resident in
                # host RAM that is a second, unswappable copy of all of
                # it. Anything that reintroduces a stream needs this flag
                # already set.
                low_cpu_mem_usage=True,
            )

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

        * cpu_offload (the default): there is nothing to undo. Group
          offloading offloads each group in its own post-forward, so by the
          time `pipe()` returns every weight is already back on the CPU and
          all that is left on the card is the caching allocator's blocks —
          which is what empty_cache() below is for. Emphatically NOT
          `.to("cpu")`, which would move the modules out from under the
          hooks and desync their bookkeeping.

          This used to call `maybe_free_model_hooks()`, which was the
          model-level offload's answer (offload everything, then re-apply
          accelerate's hooks so the pipe stays callable). Under group
          offloading it is a silent no-op — it returns early unless
          `_all_hooks` exists, and only `enable_model_cpu_offload()` sets
          that — so keeping the call would have been a comment pretending
          to be code.
        * plain .to(device): no hooks to respect, so move it to CPU here
          and let run() put it back.
        """
        if self._pipe is None:
            return
        import torch

        if not self._cpu_offload:
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
            planned = len(self._scales)
            if len(self._timesteps) != planned:
                raise RuntimeError(
                    f"wan22_vace_denoise: strength plans {planned} steps, but "
                    f"the scheduler is stepping {len(self._timesteps)} times"
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

    def _configure_sampler(self, pipe, params: Dict[str, Any]) -> None:
        """Point the scheduler at this run's step spacing and shift.

        Both are per-call. The shift is a config value diffusers reads
        inside `set_timesteps` (`flow_shift`, in its flow-sigma branch), so
        it is written to the config the way `_set_expert_split` writes the
        boundary; the spacing is read off the step by the wrapper
        `_install_sigma_schedule` puts on that method. Neither touches a
        weight, so the resident worker serves both passes from one pipeline,
        exactly as it does for `strength`.

        Refuses a scheduler that is not on flow sigmas: the shift is only
        read there, and custom sigmas are only accepted there, so on any
        other scheduler both knobs would be silently inert.
        """
        scheduler = pipe.scheduler
        if not getattr(scheduler.config, "use_flow_sigmas", False):
            raise RuntimeError(
                "wan22_vace_denoise: the scheduler is not on flow sigmas "
                "(`use_flow_sigmas` is off), so neither sampler_shift nor "
                "sigma_schedule would reach it — this is not the checkpoint's "
                "UniPC scheduler"
            )
        shift = float(params["sampler_shift"])
        variant, order = params["solver_variant"], int(params["solver_order"])
        high, low = int(params["steps_high"]), int(params["steps_low"])
        self._sigma_schedule = params["sigma_schedule"]
        self._solver_order = order
        self._solver_variant = variant
        self._steps_high, self._steps_low = high, low
        # The hand-off only exists when both experts take steps. With it
        # on, `solver_order` is the CAP and each sampler runs at its own
        # `_phase_order`; off, the run is one continuous loop at the cap,
        # which is what diffusers does by itself.
        self._handoff_pending = bool(params["handoff_reset"]) and high > 0 and low > 0
        if params["handoff_reset"]:
            opening = _phase_order(order, high if high > 0 else low)
        else:
            opening = order
        # The solver is config too: `step()` reads solver_type and
        # solver_order per call, and set_timesteps sizes the history from
        # solver_order, and both run after this.
        scheduler.register_to_config(
            flow_shift=shift, solver_type=variant, solver_order=opening
        )
        # Size the history to the order NOW: set_timesteps will rebuild
        # `model_outputs` from the config, but not `timestep_list` — see
        # _restart_multistep on what that does to step().
        _restart_multistep(scheduler)
        if self._scheduled is not scheduler:
            _install_sigma_schedule(self, scheduler)
            self._scheduled = scheduler
        logger.info(
            "  sampler: UniPC %s order %d (opening at %d), %s schedule, shift %.1f, hand-off %s",
            variant, order, opening, self._sigma_schedule, shift,
            "restarts the history" if self._handoff_pending else "carries it across",
        )

    def _set_expert_split(self, pipe, steps_high: int, n_steps: int) -> None:
        """Point the pipeline's expert boundary at the requested split, and
        log which steps each expert ends up with.

        The scheduler is asked for this run's timesteps here rather than
        after the fact because diffusers computes the boundary once, before
        the denoising loop, from `pipe.config.boundary_ratio` — so the only
        moment to place it is before the call. `set_timesteps` is a full
        reset of the scheduler's state (model_outputs, lower_order_nums,
        step index), and `pipe()` calls it again with the same count, so
        asking early costs the run nothing.

        The log line is half the point of this feature. `steps: 6` never
        said which steps the high-noise expert got, and a `strength`
        schedule is written against that split — the taper is spent on the
        low-noise steps because those are where detail is decided. Now the
        run's own log says it, per pass, in the units the schedule is
        written in.
        """
        scheduler = pipe.scheduler
        scheduler.set_timesteps(n_steps)
        timesteps = [float(value) for value in scheduler.timesteps]
        if len(timesteps) != n_steps:
            raise RuntimeError(
                f"wan22_vace_denoise: asked the scheduler for {n_steps} steps "
                f"and got {len(timesteps)} — the expert split cannot be placed "
                "against timesteps that are not the ones the run will take"
            )
        ratio = _expert_boundary_ratio(
            timesteps, steps_high, scheduler.config.num_train_timesteps
        )
        pipe.register_to_config(boundary_ratio=ratio)
        logger.info(
            "  experts: %d high-noise (t=%s) + %d low-noise (t=%s), boundary t=%.1f",
            steps_high, [round(t) for t in timesteps[:steps_high]],
            n_steps - steps_high, [round(t) for t in timesteps[steps_high:]],
            ratio * scheduler.config.num_train_timesteps,
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
        # Fitted to the frame HERE when `reference_fit` is `crop`, so that
        # diffusers' own letterboxing finds nothing to do — see the module
        # docstring on the two behaviours.
        if ref_img is not None and params["reference_fit"] == "crop":
            ref_img = _fit_reference(ref_img, int(params["width"]), int(params["height"]))
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

        # How hard the control video pushes, per denoise step and per VACE
        # injection layer. `vace_layers` is read off the model rather than
        # assumed: it is the same list on both experts (wan_fp8.py builds
        # them from one config), so either one answers.
        #
        # A schedule spans the whole run rather than one expert's share of
        # it, and the two experts split it at a fixed step: at the default
        # 2/4 on the default sampler (beta, shift 8) a 6-step run puts
        # t=1000 and 988 on the high-noise expert and t=955, 889, 753 and
        # 448 on the low-noise one — four of the six steps, and the four
        # where detail is decided. A control frame here
        # is a drawing (a flat silhouette under a DWPose skeleton), so those
        # late steps at full scale are where its ink survives into the output
        # as ink instead of being read as pose. That is why BOTH experts
        # carry a hook, and why a schedule that fades to 0 is a control video
        # that sets the pose and then lets the model finish the frame alone.
        vace_layers = (
            pipe.transformer.config.vace_layers
            if pipe.transformer is not None
            else pipe.transformer_2.config.vace_layers
        )
        n_layers = len(vace_layers)
        taper = params["strength_layers"]
        n_steps = _total_steps(params)
        schedule = _scale_schedule(
            params["strength"], taper, n_layers, n_steps
        )
        self._timesteps = None
        # A one-entry plan is a constant scale, and it goes into the call
        # itself rather than through the hooks — which is what keeps a run
        # that schedules nothing exactly the run that came before schedules
        # existed. Anything longer is rewritten per step by the hooks, and
        # what pipe() is handed is only their opening value.
        self._scales = None if len(schedule) == 1 else schedule
        conditioning_scale = schedule[0]
        if self._scales is not None or taper is not None:
            # Untapered, every entry is one number repeated at each layer, so
            # log the number — a 6-step schedule reads back as the six scales
            # the workflow wrote rather than as 48 of them.
            logger.info(
                "  VACE scale: %s",
                schedule if taper is not None else [entry[0] for entry in schedule],
            )

        # The sampler first: the split is placed against the timesteps the
        # scheduler will actually take, which the schedule and shift decide.
        self._configure_sampler(pipe, params)
        self._set_expert_split(pipe, params["steps_high"], n_steps)

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
                num_inference_steps=n_steps,
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
#                         over PCIe under cpu_offload before it runs a ~1s
#                         forward. It is offloaded again the moment that
#                         forward returns, so every pass pays this whole
#                         figure; nothing about group offloading makes it
#                         cheaper, because T5EncoderModel exposes no
#                         ModuleList child to group and lands in the
#                         all-or-nothing "unmatched" group.
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
# cpu_offload they upload lazily, a block at a time, inside the loop — and
# under group offloading they do it again on every step rather than once
# per expert turn, unstreamed and so fully exposed. If the gap between "0%"
# and "1/6" is the long one, those uploads are what you are looking at, not
# anything timed here — and expect that gap on every step, not just the
# first. `offload_blocks_per_group` is the only lever over it, and it buys
# fewer, larger transfers with VRAM.
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
