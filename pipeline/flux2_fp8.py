"""Load BFL's fp8 FLUX.2 klein checkpoint into a diffusers Flux2Transformer2DModel.

(Moved here from datasetgen/flux2_fp8.py for steps/refine_texture.py, 2026-09-16.)

`black-forest-labs/FLUX.2-klein-9b-fp8` is one 9.43 GB safetensors file in the
original BFL layout (`double_blocks.0.img_attn.qkv.weight`), and diffusers
cannot read it. It is not a missing feature — diffusers *has* a flux2 single-file
converter, `convert_flux2_transformer_checkpoint_to_diffusers`. It crashes:

    double_blocks.0.img_attn.qkv.weight_scale     F32, shape []

the converter's guard is `if ".weight" not in key and ".bias" not in key and
".scale" not in key: return`, and `".weight"` *is* a substring of
`".weight_scale"`. So a per-tensor scale reaches the fused-qkv branch and gets
`torch.chunk(scale, 3, dim=0)` applied to a 0-dim tensor.

This is the same gap `b2crunner/pipeline/wan_fp8.py` closed for ComfyUI-format
fp8 Wan checkpoints, and the same two facts close it:

**The scales are per-tensor; torchao's Float8Tensor carries per-row.** Per-tensor
is the degenerate case of per-row where every row shares a value, so the scalar
broadcasts and **nothing is dequantised and nothing is requantised** — the fp8
bytes go from the file into the model untouched. That is what makes this cheaper
than a bf16 load rather than merely different from it.

**`input_scale` is dropped.** Those are static activation scales for a serving
stack that quantises activations ahead of time. torchao quantises activations
dynamically, per call, from the actual tensor — strictly better calibrated than
a frozen scale, and it is what the rest of this stack expects. Keeping them
would mean carrying 112 tensors nothing reads.

**The renaming is not reimplemented here.** Duplicating diffusers' key map would
mean owning a copy that silently drifts. Instead the scales are carried through
*diffusers' own converter* as `[out_features, 1]` tensors sitting at the same key
as their weight: the converter renames them identically, and — the part that
actually matters — its `torch.chunk(fused, 3, dim=0)` splits a fused qkv scale
into three exactly where it splits the fused qkv weight into three. A column
rather than a full-shape stand-in because the weights are 9 B parameters and a
float32 copy of them would be 36 GB of host RAM.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_WEIGHT_SCALE = ".weight_scale"
_INPUT_SCALE = ".input_scale"
# The header marker BFL writes describing which layers were quantised. Not a
# weight; it would otherwise arrive as an unexpected key.
_QUANT_METADATA = "_quantization_metadata"


def _build_float8(qdata, scale_column):
    """A torchao Float8Tensor from (fp8 weight, per-row scale column).

    `scale_column` is [out_features, 1] — already the shape torchao wants, since
    it was carried through the converter that way. `block_size [1, in_features]`
    says "one scale per row, spanning the whole row", which is exactly what a
    per-tensor scale broadcast across rows means.

    **`act_quant_kwargs` is not optional here, and leaving it off is a trap.**
    A Float8Tensor without it is a *weight-only* quantised tensor, and torchao's
    matmul then takes this branch:

        # when input is not `Float8Tensor` ... this is float8 weight only
        out = torch.matmul(input_tensor, weight_tensor.dequantize())

    — i.e. it materialises a full bf16 copy of every weight, one layer at a
    time, on the way into each matmul. That is slower than a bf16 model and it
    OOMs: the 9B transformer died on a 128 MB transient dequantisation with
    123 MB free. Setting `act_quant_kwargs` makes torchao quantise the
    activations dynamically instead, at which point both operands are fp8 with
    per-row scales and it dispatches to `torch._scaled_mm`, which is the
    hardware path this card (sm89) has.

    PerRow to match the weight: the fast path asserts that a row-wise scaled
    weight meets a row-wise scaled input.

    `mm_config` is likewise mandatory rather than cosmetic — the same code path
    asserts it is not None before reaching `_scaled_mm`. Normally torchao's
    `quantize_()` fills both fields in; nothing fills them in when the tensors
    are built by hand from a checkpoint, which is what this module does.
    """
    import torch
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import Float8Tensor, PerRow
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    out_features, in_features = qdata.shape
    mm_config = Float8MMConfig(use_fast_accum=True)
    return Float8Tensor(
        qdata=qdata,
        scale=scale_column.to(torch.float32).contiguous(),
        block_size=[1, in_features],
        mm_config=mm_config,
        act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
            float8_dtype=torch.float8_e4m3fn,
            granularity=PerRow(),
            mm_config=mm_config,
        ),
        dtype=torch.bfloat16,
    )


def split_scales(state_dict: dict[str, Any]) -> tuple[dict, dict]:
    """Separate the checkpoint into (weights, per-row scale columns).

    Returns two dicts with *the same key names* for every quantised weight, so
    both can be run through diffusers' converter and lined back up afterwards.
    """
    import torch

    state_dict.pop(_QUANT_METADATA, None)

    scales: dict[str, Any] = {}
    for key in [k for k in state_dict if k.endswith(_WEIGHT_SCALE)]:
        scalar = state_dict.pop(key)
        weight_key = key[: -len(_WEIGHT_SCALE)] + ".weight"
        weight = state_dict.get(weight_key)
        if weight is None:
            # A scale pairing with nothing — BFL's quantiser writes one for
            # layers it deliberately left in bf16. Nothing to attach it to.
            logger.debug("dropping orphan scale %s", key)
            continue
        out_features = weight.shape[0]
        # repeat, not expand: expand shares storage, and these go on to be the
        # scale a saved Float8Tensor owns.
        scales[weight_key] = (
            scalar.reshape(1, 1).to(torch.float32).repeat(out_features, 1)
        )

    for key in [k for k in state_dict if k.endswith(_INPUT_SCALE)]:
        state_dict.pop(key)

    return state_dict, scales


def convert(state_dict: dict[str, Any]) -> dict[str, Any]:
    """BFL-layout fp8 checkpoint -> diffusers-layout state dict with Float8Tensors."""
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )

    weights, scales = split_scales(state_dict)
    n_quantised = len(scales)

    # Two passes of the *same* converter. It renames, and it splits fused qkv
    # into q/k/v, identically for both — which is the whole reason the scales
    # are shaped like weights rather than kept in a side table.
    converted_weights = convert_flux2_transformer_checkpoint_to_diffusers(weights)
    converted_scales = convert_flux2_transformer_checkpoint_to_diffusers(scales)

    for key, scale_column in converted_scales.items():
        weight = converted_weights.get(key)
        if weight is None:
            raise KeyError(
                f"scale for {key!r} survived the rename but its weight did not; "
                f"the checkpoint and diffusers' converter disagree"
            )
        converted_weights[key] = _build_float8(weight, scale_column)

    logger.info(
        "converted %d tensors, %d of them fp8 (%d after the qkv split)",
        len(converted_weights), n_quantised, len(converted_scales),
    )
    return converted_weights


def load_flux2_fp8_transformer(
    repo_id: str,
    filename: str,
    config_repo: str,
    config_subfolder: str = "transformer",
):
    """The whole path: download, convert, instantiate, load.

    `config_repo` is separate because the fp8 repos hold nothing but the weights
    file — no `config.json`, no `model_index.json` — so the architecture comes
    from the sibling bf16 repo.
    """
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info("loading fp8 transformer from %s", path)
    converted = convert(load_file(path))

    config = Flux2Transformer2DModel.load_config(
        config_repo, subfolder=config_subfolder
    )
    with init_empty_weights():
        model = Flux2Transformer2DModel.from_config(config)

    check_parity(model, converted)

    # assign=True, not the default copy: a copy would write the Float8Tensor's
    # values into a plain bf16 parameter and undo the entire point of loading an
    # fp8 checkpoint. assign puts the subclass instance in as the parameter.
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def check_parity(model, converted: dict[str, Any]) -> None:
    """Fail loudly on any key mismatch, before a partial load half-succeeds.

    `load_state_dict(strict=True)` would also catch this, but its message is a
    wall of several hundred key names. A converter that drifts against a new
    diffusers release is the expected failure here, so it is worth a report that
    says how many and which few.
    """
    expected = set(model.state_dict())
    got = set(converted)
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"fp8 conversion does not match Flux2Transformer2DModel: "
            f"{len(missing)} missing, {len(unexpected)} unexpected.\n"
            f"  missing (first 10): {missing[:10]}\n"
            f"  unexpected (first 10): {unexpected[:10]}\n"
            f"This usually means diffusers' flux2 key map changed."
        )
    logger.info("key parity OK: %d tensors", len(expected))
