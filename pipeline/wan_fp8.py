"""Load a ComfyUI-format fp8_scaled Wan checkpoint into a diffusers model.

The problem this solves: `wan22_vace_denoise` used to download 34.68 GB of
bf16 weights per expert and spend minutes of GPU time fusing the LoRA and
fp8-quantizing them, on every cold load. Meanwhile the community had already
published fp8 quants of exactly this model — half the download, none of the
quantization — but only in ComfyUI's format, which diffusers does not read.
This module is what closed that gap, and since it did, the bf16 path has
been deleted: this is now the only way the step gets its transformers.

Two gaps sit between that file and `WanVACETransformer3DModel`, and neither
turns out to need much code:

**Key naming.** The file uses the original Wan repo's names
(`blocks.0.cross_attn.k.weight`); diffusers uses its own
(`blocks.0.attn2.to_k.weight`). diffusers already ships the mapping, and it
already covers VACE — `vace_blocks.0.after_proj.bias` is the very key it
detects a VACE checkpoint by. Because the mapping is applied as a substring
rename, it renames each `X.scale_weight` in step with its `X.weight` without
being told anything about scales.

**The scales.** Each fp8 weight has a sibling `X.scale_weight` (f32, shape
[1]) and the true weight is `qdata * scale`. torchao's `Float8Tensor` stores
precisely those two things, and both halves of that correspondence were
checked directly rather than assumed: rebuilding a `Float8Tensor` from its
own `(qdata, scale)` reproduces it bit-for-bit, and its `dequantize()`
equals `qdata * scale` to within bf16 rounding. The only difference is
granularity — torchao carries a per-row scale (`[out, 1]`, block_size
`[1, in]`), the file a per-tensor one (`[1]`). Per-tensor is the special
case of per-row where every row shares a value, so the scale broadcasts and
nothing is approximated.

The consequence worth stating plainly: **no dequantization and no
requantization happens here.** The fp8 bytes go from the file into the
model unchanged. That is what makes this cheaper than the bf16 path rather
than merely different from it.

Verified against
`silveroxides/Wan_2.2-fp8_scaled_hybrid/wan2.2_fun_vace_high_noise_14B-fp8_scaled_original.safetensors`:
all 1331 tensors the model expects are matched, with no missing keys and no
shape mismatches.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ComfyUI writes a scale for every Linear, including the ones its quantizer
# deliberately left in bf16 (the text/time embedders, the time projection and
# the output head — small, precision-sensitive layers). Those scales pair
# with nothing and are dropped. `scaled_fp8` is a 2-element marker tensor
# identifying the file format. Neither is a weight; both would otherwise show
# up as unexpected keys.
_FORMAT_MARKER = "scaled_fp8"
_SCALE_SUFFIX = ".scale_weight"


def _build_float8(qdata, scale):
    """A torchao Float8Tensor from the file's (fp8 weight, per-tensor scale)."""
    import torch
    from torchao.quantization import Float8Tensor

    out_features, in_features = qdata.shape
    # repeat, not expand: expand shares storage, and a saved weight is
    # expected to own its scale tensor.
    per_row = scale.reshape(1, 1).to(torch.float32).repeat(out_features, 1)
    return Float8Tensor(
        qdata=qdata,
        scale=per_row,
        block_size=[1, in_features],
        dtype=torch.bfloat16,
    )


# fp8 x fp8 compute. The Float8Tensors `_build_float8` makes are weight-only:
# with no `act_quant_kwargs`, torchao's linear dequantizes the weight and runs a
# bf16 matmul, so the fp8 halves memory and transfer but buys no compute.
# Quantizing the activation per row as well lets `torch._scaled_mm` run on the
# fp8 tensor cores (SM89+), ~2x the bf16 GEMM rate. torchao can do that itself
# (act_quant_kwargs + mm_config), but its eager activation quantization is a
# chain of separate kernels that cost as much as the GEMM saves (5-13 ms per
# matmul at 16k tokens on the 4070 Ti, vs 0.6-1.6 ms compiled) — so the
# quantization is a small torch.compile'd function and the GEMM is called
# directly. Measured 2026-10-16 on run 1d1e93's pass 1 (720x1280x81, 4070 Ti,
# low_vram): 99.9 s -> 83.1 s per high-noise step, warm (1.20x); predicted x0
# 34 dB from the weight-only one.
_FP8_MAX = 448.0
_quantize_rows = None


def _quantize_rows_fn():
    global _quantize_rows
    if _quantize_rows is None:
        import torch

        # dynamic: the low_vram forward chunks tokens, so the row count varies.
        @torch.compile(dynamic=True)
        def quantize_rows(x):
            scale = (x.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX).clamp(min=1e-12)
            return (x.float() / scale).to(torch.float8_e4m3fn), scale

        _quantize_rows = quantize_rows
    return _quantize_rows


def _fp8_linear_forward(self, x):
    """nn.Linear.forward for a Float8Tensor weight: fp8 activations x fp8 weight.

    Reads `self.weight` per call, so the offload / weight-streaming hooks that
    swap a device copy in before the block runs are honoured.
    """
    import torch

    weight = self.weight
    shape = x.shape
    xq, x_scale = _quantize_rows_fn()(x.reshape(-1, shape[-1]))
    out = torch._scaled_mm(
        xq,
        weight.qdata.t(),
        scale_a=x_scale,
        scale_b=weight.scale.reshape(1, -1).float(),
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    if self.bias is not None:
        out = out + self.bias.to(out.dtype)
    return out.reshape(*shape[:-1], out.shape[-1]).to(x.dtype)


WAN_MATMUL_ENV = "B2C_WAN_MATMUL"
WAN_MATMUL_CHOICES = (8, 16)


def wan_matmul_bits() -> int:
    """The Wan experts' matmul precision for this process: 8 (fp8 x fp8, the
    default) or 16 (bf16 activations against the fp8 weights, the path before
    f1468e2). From $B2C_WAN_MATMUL, which `pipeline.cli run/ui --wan-matmul`
    and run.sh set; wan22_vace_denoise's `fp8_matmul` param, when given,
    overrides it per step."""
    raw = os.environ.get(WAN_MATMUL_ENV, "").strip()
    if not raw:
        return 8
    if raw not in {str(b) for b in WAN_MATMUL_CHOICES}:
        raise ValueError(f"{WAN_MATMUL_ENV}={raw!r}: expected one of {', '.join(map(str, WAN_MATMUL_CHOICES))}")
    return int(raw)


def enable_fp8_matmul(model) -> int:
    """Run every fp8-weight Linear in `model` as fp8 x fp8. Returns how many.

    Only the base Linears change; a peft LoRA wrapped around one keeps its
    adapter matmuls in bf16 on the unquantized input.
    """
    import types

    import torch
    from torchao.quantization import Float8Tensor

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and isinstance(module.weight, Float8Tensor):
            module.forward = types.MethodType(_fp8_linear_forward, module)
            count += 1
    return count


def fp8_matmul_supported(device) -> bool:
    """fp8 tensor cores: Ada (SM89), Hopper and later."""
    import torch

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= (8, 9)


def convert_fp8_state_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """ComfyUI-format fp8 state dict -> a diffusers-named one, still fp8.

    Consumes `raw` (diffusers' converter pops as it goes), so peak memory
    stays at roughly one copy of the checkpoint.
    """
    import torch
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    fp8_count = sum(1 for v in raw.values() if v.dtype == torch.float8_e4m3fn)
    logger.info("fp8 checkpoint: %d tensors, %d in fp8", len(raw), fp8_count)

    converted = convert_wan_transformer_to_diffusers(raw)

    quantized = {}
    for key in [k for k in converted if k.endswith(_SCALE_SUFFIX)]:
        base = key[: -len(_SCALE_SUFFIX)]
        weight = converted.get(f"{base}.weight")
        if weight is None or weight.dtype != torch.float8_e4m3fn:
            continue
        quantized[f"{base}.weight"] = _build_float8(weight, converted.pop(key))
    converted.update(quantized)

    dropped = [k for k in converted if k.endswith(_SCALE_SUFFIX)]
    if _FORMAT_MARKER in converted:
        dropped.append(_FORMAT_MARKER)
    for key in dropped:
        converted.pop(key, None)

    logger.info(
        "converted: %d tensors, %d fp8 linears, %d vestigial keys dropped",
        len(converted), len(quantized), len(dropped),
    )
    return converted


def load_fp8_transformer(
    path: str | Path,
    config: Dict[str, Any],
    max_blocks: Optional[int] = None,
    device: str = "cpu",
):
    """Build a WanVACETransformer3DModel from a ComfyUI fp8_scaled file.

    `max_blocks` truncates to the first N transformer blocks — enough to
    exercise this path on a machine that cannot hold the whole model, and
    not useful for anything else.
    """
    import torch
    from diffusers import WanVACETransformer3DModel
    from safetensors.torch import safe_open

    path = str(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if max_blocks is not None:
            def keep(key: str) -> bool:
                match = re.match(r"(?:vace_)?blocks\.(\d+)\.", key)
                return not match or int(match.group(1)) < max_blocks
            keys = [k for k in keys if keep(k)]
        raw = {k: handle.get_tensor(k) for k in keys}

    converted = convert_fp8_state_dict(raw)

    config = dict(config)
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    if max_blocks is not None:
        config["num_layers"] = max_blocks
        config["vace_layers"] = [l for l in config["vace_layers"] if l < max_blocks]

    # Instantiated on the meta device, which allocates nothing. A normal
    # instantiation would materialise the full bf16 model first — ~35 GB for
    # the 14B — and only then have its parameters replaced by the 17.6 GB of
    # fp8 we already hold, so the peak would exceed the memory of a machine
    # that can comfortably hold the result. `assign=True` then installs our
    # tensors in place of the meta ones rather than copying into them.
    with torch.device("meta"):
        model = WanVACETransformer3DModel(**config)

    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"fp8 checkpoint does not match {type(model).__name__}: "
            f"{len(missing)} missing (e.g. {list(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {list(unexpected)[:3]})"
        )

    # Meta-init leaves NON-PERSISTENT buffers behind: they are computed in
    # __init__ (the rotary position embedding's frequency table is the one
    # that matters here) and deliberately absent from any state dict, so
    # nothing above fills them in. Left on meta they raise at the first
    # forward pass, far from this function. Recompute them by building the
    # owning submodules for real — they are small, which is why they are
    # computed rather than stored.
    # There are exactly two, `rope.freqs_cos` and `rope.freqs_sin`, and both
    # belong to WanRotaryPosEmbed, whose constructor takes three values that
    # are all in the config. Rebuilding that one small module is the point:
    # instantiating a whole reference model to copy them from would allocate
    # the very ~35 GB the meta init exists to avoid.
    stale = [name for name, buf in model.named_buffers() if buf.is_meta]
    if stale:
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        logger.info("recomputing %d non-persistent buffer(s): %s", len(stale), stale)
        rope = WanRotaryPosEmbed(
            config["attention_head_dim"],
            tuple(config["patch_size"]),
            config["rope_max_seq_len"],
        )
        source = dict(rope.named_buffers())
        for name in stale:
            parent_path, _, leaf = name.rpartition(".")
            if not parent_path.endswith("rope") or leaf not in source:
                raise RuntimeError(
                    f"unexpected meta buffer {name!r}: this loader only knows how to "
                    f"rebuild WanRotaryPosEmbed's. diffusers may have added another."
                )
            setattr(model.get_submodule(parent_path), leaf, source[leaf].to(device))
        del rope

    remaining = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                 if t.is_meta]
    if remaining:
        raise RuntimeError(f"still on the meta device after load: {remaining[:5]}")

    return model.to(device)


def load_config(source: str) -> Dict[str, Any]:
    """The transformer config, from a local dir or an HF repo id."""
    local = Path(source)
    if local.is_file():
        return json.loads(local.read_text())
    if (local / "config.json").is_file():
        return json.loads((local / "config.json").read_text())

    from huggingface_hub import hf_hub_download

    return json.loads(
        Path(hf_hub_download(source, "config.json", subfolder="transformer")).read_text()
    )
