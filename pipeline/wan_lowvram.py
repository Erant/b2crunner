"""A low-VRAM forward for diffusers' WanVACETransformer3DModel.

The stock forward cannot run a 720x1280x81 VACE denoise on a 12 GB card even
with every weight streamed (group offloading), because its ACTIVATIONS do not
fit. Two things in it are the problem, and ComfyUI's Wan code does neither:

1. **It precomputes every VACE hint.** All eight VACE blocks run before the
   first main block and their eight full-sequence outputs (811 MiB each at
   79,200 tokens x 5120) stay alive across all 40 main blocks: 6.3 GiB held
   for the whole forward. A VACE block only reads the patch-embedded input
   (block 0's `proj_in`) and the previous VACE block's output — never a main
   block's — so hint k can be computed right before main block
   `vace_layers[k]` instead, holding one hint and one control stream at a
   time. ComfyUI's `VaceWanModel.forward_orig` interleaves them exactly this
   way.

2. **It materialises full-sequence fp32 temporaries.** Every norm, every
   modulation and every residual add upcasts the whole sequence to fp32
   (1.51 GiB each), rotary embedding does the same to q and k, and the FFN's
   intermediate is 13824 wide (2 GiB in bf16, twice over through the GELU).
   All of those are per-token operations, so they are run here over chunks
   of tokens and written into preallocated bf16 buffers. Only the two
   attention calls see the whole sequence, and they must: self-attention
   mixes every token with every other.

The arithmetic is the stock block's, op for op and in the same dtypes —
chunking a per-token op does not change any element's value, and the
attention calls get the same full-sequence q/k/v — so this is a memory
layout change, not an approximation. See tests/test_wan_lowvram.py.

The weights are streamed by `stream_weights` below, not by diffusers' group
offloading (see its docstring for why), though `install()` works under both.
Blocks are always entered through `module(...)`, never by reaching into
their submodules from outside, so the offload hooks fire for every block.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Iterator, Optional, Tuple

#: Tokens per chunk for the per-token work. 8192 x 13824 bf16 is the largest
#: temporary (the FFN intermediate) at 216 MiB; the attention buffers dwarf it.
DEFAULT_CHUNK = 8192


def _spans(length: int, chunk: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, length, chunk):
        yield start, min(start + chunk, length)


def _rope(x, cos, sin):
    """diffusers' WanAttnProcessor.apply_rotary_emb, verbatim."""
    import torch

    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = cos[..., 0::2]
    sin = sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


def _attend(attn, q, k, v):
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    out = dispatch_attention_fn(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
        backend=attn.processor._attention_backend, parallel_config=None,
    )
    return out.flatten(2, 3).type_as(q)


def _block_body(block, h, encoder_hidden_states, temb, rotary_emb, chunk: int):
    """WanTransformerBlock.forward (temb.ndim == 3), in place on `h`.

    The VACE block's body after its `proj_in` is the same code, so both
    block kinds come through here.
    """
    import torch

    if temb.ndim != 3:
        raise NotImplementedError("wan_lowvram: per-token temb (Wan 2.2 TI2V 5B) is not supported")
    attn1, attn2 = block.attn1, block.attn2
    if attn1.fused_projections or attn2.fused_projections or attn2.add_k_proj is not None:
        raise NotImplementedError("wan_lowvram: fused or image-conditioned attention is not supported")

    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        block.scale_shift_table.to(temb.device) + temb.float()
    ).chunk(6, dim=1)
    batch, length, _ = h.shape
    heads = attn1.heads
    cos, sin = rotary_emb

    # 1. Self-attention: q/k/v built a chunk at a time, attended whole.
    q = h.new_empty(batch, length, heads, h.shape[-1] // heads)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    for s, e in _spans(length, chunk):
        x = (block.norm1(h[:, s:e].float()) * (1 + scale_msa) + shift_msa).type_as(h)
        q[:, s:e] = _rope(attn1.norm_q(attn1.to_q(x)).unflatten(2, (heads, -1)), cos[:, s:e], sin[:, s:e])
        k[:, s:e] = _rope(attn1.norm_k(attn1.to_k(x)).unflatten(2, (heads, -1)), cos[:, s:e], sin[:, s:e])
        v[:, s:e] = attn1.to_v(x).unflatten(2, (heads, -1))
    del x
    a = _attend(attn1, q, k, v)
    del q, k, v
    for s, e in _spans(length, chunk):
        o = attn1.to_out[1](attn1.to_out[0](a[:, s:e]))
        h[:, s:e] = (h[:, s:e].float() + o * gate_msa).type_as(h)
    del a, o

    # 2. Cross-attention: the text side is 512 tokens, only the query side
    # is chunked.
    q = h.new_empty(batch, length, heads, h.shape[-1] // heads)
    for s, e in _spans(length, chunk):
        x = block.norm2(h[:, s:e].float()).type_as(h)
        q[:, s:e] = attn2.norm_q(attn2.to_q(x)).unflatten(2, (heads, -1))
    del x
    k = attn2.norm_k(attn2.to_k(encoder_hidden_states)).unflatten(2, (heads, -1))
    v = attn2.to_v(encoder_hidden_states).unflatten(2, (heads, -1))
    a = _attend(attn2, q, k, v)
    del q, k, v
    for s, e in _spans(length, chunk):
        h[:, s:e] = h[:, s:e] + attn2.to_out[1](attn2.to_out[0](a[:, s:e]))
    del a

    # 3. Feed-forward.
    for s, e in _spans(length, chunk):
        x = (block.norm3(h[:, s:e].float()) * (1 + c_scale_msa) + c_shift_msa).type_as(h)
        f = block.ffn(x)
        h[:, s:e] = (h[:, s:e].float() + f.float() * c_gate_msa).type_as(h)
    return h


def _main_block_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    return _block_body(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                       self._lowvram_chunk)


def _vace_block_forward(self, hidden_states, encoder_hidden_states, control_hidden_states,
                        temb, rotary_emb):
    """WanVACETransformerBlock.forward: `control_hidden_states` in place,
    returns (hint, control_hidden_states)."""
    chunk = self._lowvram_chunk
    c = control_hidden_states
    length = c.shape[1]
    if self.proj_in is not None:
        for s, e in _spans(length, chunk):
            c[:, s:e] = self.proj_in(c[:, s:e]) + hidden_states[:, s:e]
    _block_body(self, c, encoder_hidden_states, temb, rotary_emb, chunk)
    hint = None
    if self.proj_out is not None:
        hint = c.new_empty(c.shape)
        for s, e in _spans(length, chunk):
            hint[:, s:e] = self.proj_out(c[:, s:e])
    return hint, c


def _model_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    control_hidden_states=None,
    control_hidden_states_scale=None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
):
    """WanVACETransformer3DModel.forward with the VACE hints interleaved."""
    import torch
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if attention_kwargs and attention_kwargs.get("scale", 1.0) != 1.0:
        raise NotImplementedError("wan_lowvram: a LoRA scale through attention_kwargs is not supported")
    if encoder_hidden_states_image is not None:
        raise NotImplementedError("wan_lowvram: image conditioning is not supported")
    chunk = self._lowvram_chunk
    vace_layers = list(self.config.vace_layers)

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_f, post_h, post_w = num_frames // p_t, height // p_h, width // p_w

    if control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(vace_layers))
    scales = torch.unbind(control_hidden_states_scale)
    if len(scales) != len(vace_layers):
        raise ValueError(
            f"Length of `control_hidden_states_scale` {len(scales)} should be "
            f"equal to {len(vace_layers)}."
        )

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    control = self.vace_patch_embedding(control_hidden_states)
    control = control.flatten(2).transpose(1, 2)
    padding = control.new_zeros(batch_size, hidden_states.size(1) - control.size(1), control.size(2))
    control = torch.cat([control, padding], dim=1)
    del padding

    temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
        timestep, encoder_hidden_states, None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    length = hidden_states.shape[1]
    next_vace = 0
    for i, block in enumerate(self.blocks):
        hint = None
        if i in vace_layers:
            # Before main block i, not after: VACE block 0 reads the
            # patch-embedded input, which main block 0 is about to
            # overwrite in place. Later VACE blocks ignore hidden_states.
            hint, control = self.vace_blocks[next_vace](
                hidden_states, encoder_hidden_states, control, timestep_proj, rotary_emb
            )
            scale = scales[next_vace]
            next_vace += 1
        hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        if hint is not None:
            for s, e in _spans(length, chunk):
                hidden_states[:, s:e] = hidden_states[:, s:e] + hint[:, s:e] * scale
            del hint
    del control

    shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    out = None
    for s, e in _spans(length, chunk):
        x = (self.norm_out(hidden_states[:, s:e].float()) * (1 + scale) + shift).type_as(hidden_states)
        x = self.proj_out(x)
        if out is None:
            out = x.new_empty(batch_size, length, x.shape[-1])
        out[:, s:e] = x
    del hidden_states

    out = out.reshape(batch_size, post_f, post_h, post_w, p_t, p_h, p_w, -1)
    out = out.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = out.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _swap_in(module, device, recurse: bool):
    """Point `module`'s params/buffers at device copies; return the originals."""
    import torch

    saved = []
    subs = module.modules() if recurse else (module,)
    for sub in subs:
        for table in (sub._parameters, sub._buffers):
            for name, tensor in table.items():
                if tensor is None or tensor.device == device:
                    continue
                saved.append((table, name, tensor))
                moved = tensor.to(device, non_blocking=True)
                table[name] = (
                    torch.nn.Parameter(moved, requires_grad=False)
                    if isinstance(tensor, torch.nn.Parameter) else moved
                )
    return saved


def stream_weights(root, device) -> None:
    """Keep `root`'s weights where they are and copy them to `device` per use.

    ComfyUI's low-VRAM loading, and the reason this exists instead of
    diffusers' group offloading: an unstreamed group offload returns each
    group with `module.to("cpu")`, which allocates a NEW anonymous host copy
    of every weight on the first forward. The fp8 experts are mmap'd
    straight out of their safetensors files (page cache, reclaimable), so
    that turns 0.6 GB of RSS into 17.4 GB per expert — measured on the
    4070 Ti box — and two experts plus T5 into more RAM than it has. Here a
    module's device copies are simply dropped after its forward and the
    original (mmap'd) tensors put back, so nothing is ever copied to host.

    Units: every member of a ModuleList is one unit (its whole subtree moves
    together, before its forward); every other module that owns tensors
    directly gets a non-recursive unit of its own. That covers any model
    whose weights are only touched inside the forward of the module that
    owns them, which is true of Wan, the Wan VAE and T5.
    """
    import torch

    device = torch.device(device)

    def hook(module, recurse):
        def pre(mod, args, kwargs):
            mod._lowvram_saved = _swap_in(mod, device, recurse)

        def post(mod, args, kwargs, output):
            for table, name, tensor in getattr(mod, "_lowvram_saved", ()):
                table[name] = tensor
            mod._lowvram_saved = ()

        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.register_forward_hook(post, with_kwargs=True)

    seen = set()

    def add(module, recurse):
        # Once per module: T5's `shared` and `encoder.embed_tokens` are the
        # same Embedding, and a second pre-hook would overwrite the first's
        # saved originals with nothing, stranding the weights on the card.
        if id(module) in seen:
            return
        seen.update(id(m) for m in (module.modules() if recurse else (module,)))
        hook(module, recurse)

    def walk(mod):
        if any(t is not None for t in mod._parameters.values()) or any(
            t is not None for t in mod._buffers.values()
        ):
            add(mod, recurse=False)
        for child in mod.children():
            if isinstance(child, torch.nn.ModuleList):
                # The outermost list only: a block's own lists (to_out)
                # travel with the block.
                for unit in child:
                    add(unit, recurse=True)
            else:
                walk(child)

    walk(root)


def install(transformer, chunk: int = DEFAULT_CHUNK) -> None:
    """Swap the low-VRAM forwards onto `transformer` and its blocks.

    Under diffusers' group offloading this must run first: those hooks wrap
    whatever `forward` each module has when they are registered.
    `stream_weights` uses ordinary module hooks and does not care.
    """
    if 0 not in transformer.config.vace_layers:
        raise ValueError("wan_lowvram: VACE layer 0 is required")
    for module in (transformer, *transformer.blocks, *transformer.vace_blocks):
        module._lowvram_chunk = chunk
    transformer.forward = types.MethodType(_model_forward, transformer)
    for block in transformer.blocks:
        block.forward = types.MethodType(_main_block_forward, block)
    for block in transformer.vace_blocks:
        block.forward = types.MethodType(_vace_block_forward, block)
