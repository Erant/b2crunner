"""The live Lightning LoRA's add folded into one GEMM.

The adapters stay unfused (wan22_vace_denoise._load_lora_unfused says why),
so every adapted Linear runs peft's vanilla path:

    result = result + lora_B(lora_A(x)) * scaling

which after the two low-rank matmuls is a multiply and an add over the
whole (tokens x out_features) output, each a full read and write of it in
bf16 — 2 x 8192 x 13824 x 2 bytes per FFN chunk. `addmm_` does the scale
and the add in lora_B's GEMM epilogue instead (beta = 1 into `result`).
Measured 2026-10-25 on a 4070 Ti, 720x1280x81, low_vram, fp8 matmul: a
warm high-noise step 60.4 -> 56.7 s. The value differs only in rounding
(one rounding of B(A(x)) * scaling + result in the GEMM's fp32 accumulator,
where peft rounds each of the three to bf16).

Per instance, like wan_fp8.enable_fp8_matmul: only the Linears of the
model it is given change, and anything other than the plain inference
case — adapters disabled or merged, several active adapters, a LoRA
variant (DoRA, ...), dropout, extra forward arguments — goes through
peft's own forward untouched.
"""
from __future__ import annotations

import types


def _addmm_forward(self, x, *args, **kwargs):
    import torch

    active = self.active_adapters
    if (args or kwargs or self.disable_adapters or self.merged or len(active) != 1):
        return type(self).forward(self, x, *args, **kwargs)
    name = active[0]
    if (name not in self.lora_A or name in self.lora_variant
            or not isinstance(self.lora_dropout[name], torch.nn.Identity)):
        return type(self).forward(self, x, *args, **kwargs)
    result = self.base_layer(x)
    a_weight = self.lora_A[name].weight
    b_weight = self.lora_B[name].weight
    low = torch.nn.functional.linear(x.to(a_weight.dtype), a_weight)
    if result.dtype != b_weight.dtype or not result.is_contiguous():
        # peft's arithmetic, without running the base layer a second time.
        return (result + self.lora_B[name](low) * self.scaling[name]).to(result.dtype)
    result.view(-1, result.shape[-1]).addmm_(
        low.reshape(-1, low.shape[-1]), b_weight.t(), alpha=self.scaling[name]
    )
    return result


def enable_lora_addmm(model) -> int:
    """Route every peft LoRA Linear in `model` through _addmm_forward. Returns how many."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    count = 0
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.forward = types.MethodType(_addmm_forward, module)
            count += 1
    return count
