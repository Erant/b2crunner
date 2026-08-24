# fp8 for Wan 2.2 VACE: what works, what doesn't, and why

How `pipeline/steps/wan22_vace_denoise.py` gets fp8 weights onto the GPU.
This file records what was actually tried and measured, so the next attempt
doesn't re-derive it — including the path that was tried and then deleted.

**Short version:** the step loads a pre-quantized community fp8 checkpoint
and applies the Lightning LoRA on top as a live (unfused) adapter, which
the torch 2.13 / torchao 0.18 bump is what made possible. It downloads
**47 GB per load instead of 81 GB**, and does no quantization at all. The
bf16 + fuse + quantize path, and its `fused_cache_dir`, have been removed —
see below before reinstating anything.

## The version triangle: RESOLVED (read this first)

There used to be three constraints that could not all hold at once. The
image pinned **torch 2.9.1 / torchao 0.15.0**, which satisfied only the
first:

| Want | Needs | On torch 2.9.1 | On torch 2.13.0 |
|---|---|---|---|
| `import torchao` at all | a torchao built for the installed torch (0.18 needs torch ≥ 2.11) | ✅ | ✅ |
| LoRA applied to quantized weights | diffusers requires torchao ≥ 0.16 | ❌ | ✅ |
| `save_pretrained` in safetensors | diffusers requires torchao ≥ 0.16 | ❌ | ✅ * |

The image now pins **torch 2.13.0 / torchvision 0.28.0 / torchao 0.18.0**,
and all three rows hold. Each was verified directly on an sm_89 GPU against
a real (if small) `WanVACETransformer3DModel` — not inferred from version
numbers.

\* Row 3 has a condition that is easy to trip over, and it is *not* a
version condition. See "Row 3's catch" below.

**Why the pin still matters.** torchao and torch are hard-coupled: a torchao
built for a newer torch than the one installed dies on import with

```
ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'
```

and because `transformers.modeling_utils` imports `quantizer_torchao`
unconditionally, an installed-but-unimportable torchao takes `import
diffusers` down with it. **The entire wan22_vace_denoise step could not
import in the shipped image.** An absent torchao is handled by the guard; a
broken one is not. This actually happened, at torch 2.9.1 with an unpinned
`torchao>=0.16.0` floating to 0.18.0. Move the torchao pin *with*
`TORCH_VERSION`, never independently.

### Row 3's catch: who did the quantizing

torchao ≥ 0.16 is necessary for safetensors serialization of quantized
weights but not sufficient. safetensors still cannot introspect torchao's
tensor subclass directly:

```
RuntimeError: Attempted to access the data pointer on an invalid python storage
```

diffusers works around this by *flattening* the subclass into plain
`(qdata, scale)` tensors plus metadata — but that flattening lives on the
attached `TorchAoHfQuantizer`, so it only happens for a model **diffusers
itself quantized** via `TorchAoConfig`. Verified both ways:

| How the model was quantized | LoRA adapter | `fuse_lora` | safetensors save |
|---|---|---|---|
| `quantize_()` directly (torchao API) | ✅ | ❌ | ❌ |
| `TorchAoConfig` at `from_pretrained` | ✅ | ✅ | ✅ |

This is why the earlier note's "a prior attempt on torch 2.13/torchao 0.18
still hit the storage-pointer error" was right about the symptom and wrong
about the cause: that attempt used `quantize_()` directly. The torchao
version gate is real — below 0.16 diffusers will not even try — but clearing
it is not on its own enough, and it was never what produced *that* error.

The `fuse_lora` failure in row 1 of that table surfaces as
`TypeError: can't multiply sequence by non-int of type 'float'`, which is
maximally unhelpful. peft says it plainly in a warning it emits earlier:
`TorchaoLoraLinear` can only `merge()` when it can recover the
requantization subclass, which it gets from the attached quantizer.

## The bf16 path: REMOVED (2026-08-24)

There used to be a default path that downloaded the bf16 diffusers
checkpoint, fused the Lightning LoRA into it, ran `quantize_(model,
Float8WeightOnlyConfig())`, and optionally cached the result under
`fused_cache_dir`. **It is gone**, along with `fused_cache_dir` and the
`quantize` param. Do not restore it. The numbers are the argument:

| | bf16 path (removed) | fp8 path (now the only one) |
|---|---|---|
| transformers | 69.36 GB (2 x 34.68) | **35.16 GB** (2 x 17.58) |
| text_encoder + vae + tokenizer + scheduler | 11.89 GB | 11.89 GB |
| **per load** | **81.24 GB** | **47.05 GB** |
| quantize step | minutes of GPU work, every load | none |

Measured from the live repos, not estimated. And `fast_helical_full` loads
the denoiser **twice**, so this is ~68 GB per run off a network volume.

Two things worth keeping from that path's history, because they are still
true and still cost time to rediscover:

**The `fused_cache_dir` cache never once succeeded**, and the recorded
reason was wrong. It saved via diffusers' default safetensors
serialization, which cannot introspect torchao's tensor subclass:

```
RuntimeError: Attempted to access the data pointer on an invalid python storage
```

An earlier version of this file guessed at a version mismatch. That was
wrong twice over: it reproduced identically on torchao 0.15.0, and it still
reproduces on 0.18.0. The real cause is "Row 3's catch" above — that path
called `quantize_()` directly, so no `TorchAoHfQuantizer` was attached and
diffusers never flattened the subclass. `safe_serialization=False`
(torch.save) did work, verified bit-identical on a real
`WanVACETransformer3DModel`.

**`cache_hit` tested for the directory, not the weight file**, so every
previously-failed save left an empty directory that registered as a hit.
If you ever build another on-disk cache here, check for the artifact.

## The fp8 path (the only path)

`pipeline/wan_fp8.py`. `fp8_checkpoint_high` / `fp8_checkpoint_low` now
default to the community checkpoint rather than gating an opt-in, and
`resolve_fp8_checkpoint()` treats a value that exists on disk as a path and
anything else as a filename inside `fp8_repo`. **Proven to work**, against
`silveroxides/Wan_2.2-fp8_scaled_hybrid`'s
`wan2.2_fun_vace_{high,low}_noise_14B-fp8_scaled_original.safetensors`
(17.6 GB each, vs ~35 GB of bf16):

- all **1331** tensors the model expects are matched — no missing keys, no
  unexpected keys, no shape mismatches;
- loads on CPU in **5 s at 0.8 GB RSS** (safetensors mmaps; the fp8 bytes
  are never copied);
- a full **40-block forward pass** produces finite, correctly-shaped output
  (`(1, 16, 9, 32, 32)`, mean −0.086, std 0.369, no NaNs), peak RSS 18.7 GB;
- **no dequantization and no requantization anywhere** — the fp8 bytes go
  from file to model unchanged.

An earlier version of this file called this "not a simple rename job" and
listed two gaps. Both turned out to be smaller than feared:

1. **Key naming.** diffusers already ships the mapping *and it already
   covers VACE* — `vace_blocks.0.after_proj.bias` is the very key
   `convert_wan_transformer_to_diffusers` detects a VACE checkpoint by.
   Because it renames by substring, it renames each `X.scale_weight` in step
   with its `X.weight` for free.
2. **The scale convention.** Each fp8 weight has a sibling `X.scale_weight`
   (f32, `[1]`) and the true weight is `qdata * scale`. torchao's
   `Float8Tensor` stores exactly those two things — verified directly, not
   assumed: rebuilding one from its own `(qdata, scale)` reproduces it
   bit-for-bit, and `dequantize()` equals `qdata * scale` to within bf16
   rounding. The only difference is granularity: torchao is per-row
   (`[out, 1]`, block_size `[1, in]`), the file per-tensor (`[1]`).
   Per-tensor is per-row with every row equal, so it broadcasts and nothing
   is approximated.

Two details worth keeping:

- **7 keys are dropped, not mapped**: 6 `.scale_weight` whose weight is BF16
  (the quantizer deliberately left the text/time embedders, time projection
  and output head in high precision, but ComfyUI writes a scale for every
  linear regardless) plus `scaled_fp8`, a 2-element format marker. Dropping
  them is what turns 1338 keys into the model's exact 1331.
- **Meta-device init is required**, not an optimization. Instantiating the
  model normally materialises the full bf16 14B — tens of GB — before
  `load_state_dict` replaces it with the 17.6 GB we already hold. Meta init
  plus `assign=True` avoids that, at the cost of leaving
  `rope.freqs_{cos,sin}` (non-persistent buffers, in no state dict) on meta;
  `wan_fp8.py` rebuilds just `WanRotaryPosEmbed` to fill them.

### The Lightning LoRA: now available here, unfused

This used to be the blocker, and the step raised if you asked for both.
diffusers refused a LoRA on torchao-quantized weights below torchao 0.16:

```
ImportError: Found an incompatible version of torchao.
Found version 0.15.0, but only versions above 0.16.0 are supported
```

torchao 0.18 clears that gate. `_load_lora_unfused` in the step loads both
Lightning adapters onto the pre-quantized transformers and sets their
scales, and the guard is gone.

**Unfused, though.** `fuse_lora` on this path fails with
`TypeError: can't multiply sequence by non-int of type 'float'`, because
peft can only merge into a torchao subclass when it can recover the
requantization config from an attached diffusers quantizer — and
`wan_fp8.py` builds the model straight from a state dict, so there is none.
See "Row 3's catch" above; it is the same root cause.

That is a performance detail, not a correctness one. Verified on GPU: with
the adapter live, the forward pass stays finite and the adapter measurably
changes the output. The cost is one extra low-rank matmul per adapted
Linear per step.

### What is still unproven about it

It has never produced real frames end to end. Everything above is verified —
key mapping, load, a full 40-block forward pass, LoRA application, GPU
output finite — but no complete `fast_helical_full` run has come out of it
and been looked at. The bf16 path it replaced had produced actual output.

That was a deliberate, stated trade: the bf16 path was removed on the
user's explicit instruction ("we should never be downloading bf16 weights
if the fp8 checkpoint is easily available"), with the download numbers
above as the justification. The way back is git history and the
`b2c/pipeline:torch291-rollback` image, not a hidden fallback in the step.

**So: eyeball the first pod run's frames before trusting a long one.**

## The follow-on that is now unblocked

Publishing a real diffusers-format fp8 VACE checkpoint — which still does
not exist publicly. Every fp8 quant of this model on the Hub is
ComfyUI-format (`Kijai/WanVideo_comfy_fp8_scaled` ships only the VACE
*module*, not the full transformer; `silveroxides` ships the full one), and
both diffusers conversions on the Hub (`linoyts`, `Pyros13`) are BF16.

The route, given "Row 3's catch": quantize through diffusers'
`TorchAoConfig` rather than `quantize_()`, which attaches the quantizer that
makes both `fuse_lora` and safetensors serialization work. Fuse the
Lightning LoRA once, offline, save with `safe_serialization=True`, publish.

That is now a smaller change than it was: the step no longer fuses the
LoRA at all (it cannot — the weights arrive pre-quantized), so the ordering
conflict that used to block it is gone. What remains is a one-off offline
script, not a change to the step: load the fp8 experts, apply and merge the
Lightning LoRA via a model diffusers quantized through `TorchAoConfig`, and
`save_pretrained(safe_serialization=True)`.

The prize is a checkpoint that needs no LoRA at run time, which would also
remove the per-step low-rank matmul the unfused adapter costs today.
