# fp8 for Wan 2.2 VACE: what works, what doesn't, and why

`pipeline/steps/wan22_vace_denoise.py` has two ways to get fp8 weights onto
the GPU. This file records what was actually tried and measured, so the next
attempt doesn't re-derive it.

**Short version:** the default path (bf16 + fuse LoRA + torchao quantize)
now caches correctly, so that work happens once per volume instead of once
per run. Loading a community fp8 checkpoint also works and is proven, but
cannot be combined with the Lightning LoRA until the image's torch moves.

## The version triangle (read this first)

Three constraints that cannot currently all hold:

| Want | Needs |
|---|---|
| `import torchao` at all | torchao ≤ 0.15 on torch 2.9.1 (0.18 needs torch ≥ 2.11) |
| LoRA applied to quantized weights | diffusers requires torchao **> 0.16** |
| `save_pretrained` in safetensors | diffusers requires torchao's safetensors support |

The image pins **torch 2.9.1 / torchao 0.15.0**. That satisfies the first
row and neither of the others.

This is not academic. Before it was pinned, `torchao>=0.16.0` floated to
0.18.0, which fails to import on torch 2.9.1:

```
ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'
```

and because `transformers.modeling_utils` imports `quantizer_torchao`
unconditionally, an installed-but-unimportable torchao takes `import
diffusers` down with it. **The entire wan22_vace_denoise step could not
import in the shipped image.** An absent torchao is handled by the guard; a
broken one is not. Re-check the pin whenever `TORCH_VERSION` moves.

## Path 1 (default): bf16 + fuse + quantize, cached

What the step does with no fp8 params set: download the bf16 diffusers
checkpoint, fuse the Lightning LoRA, `quantize_(model,
Float8WeightOnlyConfig())`, and — with `fused_cache_dir` set — save the
result so later loads skip all of it.

**That cache had never once succeeded.** It saved with diffusers' default
safetensors serialization, which cannot introspect the storage of torchao's
quantized tensor subclass:

```
RuntimeError: Attempted to access the data pointer on an invalid python storage
```

An earlier version of this file guessed at a version mismatch. It is not:
the failure reproduces identically on torchao 0.15.0, and diffusers states
the rule plainly — *"not serializable with safe serialization without
safetensors support from the installed torchao version"* — which is row 3 of
the triangle above.

`safe_serialization=False` (torch.save) works today. Verified on a real
`WanVACETransformer3DModel`: save, reload, and the dequantized weights are
**bit-identical**, with every quantized linear preserved. Three lines were
needed — the save, a matching `use_safetensors=False` on the load side, and
the `cache_hit` predicate, which tested for the *directory* and so treated
every previously-failed save's empty directory as a hit.

Cost: a pickle rather than safetensors. Fine for a cache a process writes to
its own volume; not what you would publish to HF.

## Path 2 (opt-in): load a community fp8 checkpoint

`pipeline/wan_fp8.py`, enabled with `fp8_checkpoint_high` /
`fp8_checkpoint_low`. **Proven to work**, against
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

### Why it isn't the default

**It cannot use the Lightning LoRA.** diffusers refuses to apply a LoRA to
torchao-quantized weights below torchao 0.16:

```
ImportError: Found an incompatible version of torchao.
Found version 0.15.0, but only versions above 0.16.0 are supported
```

and the existing path only works because it fuses the LoRA *before*
quantizing. With weights that arrive already quantized there is no such
window. Since 6-step sampling at cfg 1.0 depends on that distill LoRA, this
path currently means either accepting non-distilled sampling (many more
steps) or waiting on the torch bump below. The step raises if you set both
rather than letting you find out from the output.

## The unlock, if it's wanted

`torch 2.13.0+cu130` exists on the PyTorch index, and torchao 0.18 pairs
with it. That single bump resolves rows 2 and 3 of the triangle at once:
LoRA onto quantized weights, and possibly safetensors serialization (the
prior attempt on torch 2.13/torchao 0.18 still hit the storage-pointer
error, so verify rather than assume).

It is not a free change. torch 2.9.1 is what detectron2, the sam3dbody
stack and the whole CUDA-13 arrangement were verified against — see
docker-build-notes.md. Bumping means re-running that verification. Worth
doing deliberately, between pod rounds, not on the way into one.

If the bump lands, the follow-on is worth it: fuse the Lightning LoRA into
the fp8 weights **once**, offline, and publish the result as a real
diffusers fp8 VACE checkpoint — which still does not exist publicly. Every
fp8 quant of this model on the Hub is ComfyUI-format
(`Kijai/WanVideo_comfy_fp8_scaled` ships only the VACE *module*, not the
full transformer; `silveroxides` ships the full one), and both diffusers
conversions on the Hub (`linoyts`, `Pyros13`) are BF16.
