# Notes toward a real fp8 diffusers VACE checkpoint

`pipeline/steps/wan22_vace_denoise.py` currently quantizes to fp8 at every
`load()` (torchao, in-memory, on top of the bf16
`linoyts/Wan2.2-VACE-Fun-14B-diffusers` checkpoint) because no pre-quantized
diffusers-format WAN2.2 VACE checkpoint exists publicly. Publishing one to
HF would let future loads skip both the ~80GB bf16 download and the
fuse+quantize step. This file records what's been hit trying to get there,
so the next attempt doesn't re-discover the same walls.

## What exists publicly (and why none of it is directly usable)

fp8 quants of WAN2.2 VACE-Fun-14B are all **ComfyUI-native single-file
safetensors** — `Kijai/WanVideo_comfy_fp8_scaled`, `Comfy-Org`'s repackaged
files, `wangkanai/wan22-fp8-i2v`, `silveroxides/Wan_2.2-fp8_scaled_hybrid`
(`wan2.2_fun_vace_{high,low}_noise_14B-fp8_scaled_original.safetensors`).
None are diffusers-format (`WanVACETransformer3DModel`'s module/key layout).
Two separate gaps to close before one of these is usable from diffusers,
not one:

1. **Key naming.** ComfyUI/Kijai-style checkpoints use the original
   Wan-Video repo's key names, not diffusers' converted names. diffusers
   does have conversion logic for this (see
   `scripts/convert_wan_to_diffusers.py` in the diffusers repo), and
   `from_single_file()` is confirmed working for **GGUF**-quantized Wan
   checkpoints via this path.
2. **Quantization scheme.** The "fp8_scaled" files use Comfy-Org's own
   scaled-fp8 convention — separate per-tensor/per-block scale factors
   alongside fp8 weights — which is a different representation than
   torchao's `Float8WeightOnlyConfig` tensor subclass. `from_single_file()`
   being proven for GGUF does not mean it handles this scheme; that's
   unconfirmed, not ruled out, as of this writing (see conversation this
   file was written from — nobody has actually tried loading one of these
   through `from_single_file()` yet).

Converting one of these into a real diffusers artifact means writing (or
finding) a script that does both key remapping and scale-tensor handling —
not a simple rename job.

## What we tried instead, and what broke

Approach: load the bf16 diffusers checkpoint, fuse the Lightning LoRA in,
quantize with torchao (`quantize_(model, Float8WeightOnlyConfig())`), then
`save_pretrained()` the result so it only has to happen once.

**Broke**: `save_pretrained()` on the quantized `WanVACETransformer3DModel`
raises deep inside `safetensors`:

```
RuntimeError: Attempted to access the data pointer on an invalid python storage.
```

from `safetensors/torch.py`'s `_find_shared_tensors` → `storage_ptr()` →
`tensor.storage().data_ptr()`. This happens on this exact stack:
`torch==2.13.0+cu130`, `diffusers==0.40.0`, `torchao==0.18.0`. safetensors
can't introspect the storage of torchao's quantized tensor subclass to
serialize it — either a genuine incompatibility between these three
versions, or torchao quantized tensors just aren't safetensors-serializable
in general and need a different save path.

Also notable: loading the **plain bf16** checkpoint (no quantization
involved yet) already prints

```
Unable to import `torchao` Tensor objects. This may affect loading checkpoints serialized with `torchao`
```

on every `WanVACEPipeline.from_pretrained()` call in this environment —
diffusers is attempting to register torchao's tensor-subclass import at
import time and failing, in this same version combo. That warning firing on
a checkpoint that isn't even torchao-quantized is a hint the diffusers/
torchao pairing itself is off, not just a save-path bug — check for a
diffusers/torchao version combination that doesn't print this at all before
sinking more time into the save side specifically.

## Where to actually look next

- Check the `torchao` changelog / diffusers' `quantization/torchao.md` docs
  for the currently-recommended version pairing, rather than whatever
  floated in via `torchao>=0.16.0` — pin, don't float, once one is found
  that doesn't print the import warning above.
- Try `pipe.transformer.dequantize()` (if torchao/diffusers exposes one)
  before `save_pretrained()`, saving a plain bf16 checkpoint instead of a
  quantized one — defeats the point of caching a *quantized* checkpoint,
  but at least validates the fuse-LoRA half of the pipeline can round-trip,
  isolating whether the bug is specifically the quantized-tensor
  serialization or something broader.
- Or skip torchao's safetensors path entirely and use `torch.save` /
  `pickle` for the fused+quantized `state_dict()`, accepting the security/
  portability tradeoffs of pickle — safetensors is the thing that's broken
  here, not necessarily torch's own serialization of the tensor subclass.
- Whichever of the above actually works is also the template for producing
  something publishable to HF: the goal there needs a real diffusers
  `save_pretrained()` (safetensors, HF-hostable), not a `torch.save` blob, so
  the safetensors compatibility question has to get resolved either way if
  publishing is the end goal, not just local caching.

Current code (`wan22_vace_denoise.py`) treats this cache as strictly
best-effort — a failed save logs a warning and the run continues normally
using the in-memory quantized pipe. Don't remove that guard while iterating
on a fix; a caching bug should never be able to break inference itself.
