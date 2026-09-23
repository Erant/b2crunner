"""pipeline/wan_lowvram.py computes what the stock VACE forward computes.

The low-VRAM forward reorders the VACE blocks and chunks the per-token work;
neither may change a single output value. Compared here against the stock
forward on a scaled-down WanVACETransformer3DModel with native SDPA, where
the only difference between the two is memory layout — so the outputs must
be bitwise equal, not merely close. Runs in venv_wan22 on a CUDA box (the
image); skips elsewhere.
"""

from __future__ import annotations

import copy
import unittest


def _stack_or_skip():
    try:
        import torch
        from diffusers import WanVACETransformer3DModel  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None
    return torch


def _model(torch):
    from diffusers import WanVACETransformer3DModel

    torch.manual_seed(0)
    model = WanVACETransformer3DModel(
        num_attention_heads=2, attention_head_dim=128, in_channels=16, out_channels=16,
        text_dim=64, freq_dim=256, ffn_dim=512, num_layers=7, vace_layers=[0, 3, 5],
        vace_in_channels=96,
    )
    # Randomise the zero-initialised tables so every path carries signal.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return model.to("cuda", torch.bfloat16).eval()


def _inputs(torch, frames: int = 3, height: int = 12, width: int = 10):
    g = torch.Generator("cuda").manual_seed(1)
    kw = dict(device="cuda", dtype=torch.bfloat16, generator=g)
    return dict(
        hidden_states=torch.randn(1, 16, frames, height, width, **kw),
        timestep=torch.tensor([700], device="cuda"),
        encoder_hidden_states=torch.randn(1, 9, 64, **kw),
        # One latent frame short, as a reference image makes it: the stock
        # forward zero-pads the control stream and so must this one.
        control_hidden_states=torch.randn(1, 96, frames - 1, height, width, **kw),
        control_hidden_states_scale=torch.tensor([1.0, 0.5, 0.25], device="cuda"),
        return_dict=False,
    )


class TestLowVramForwardIsTheStockForward(unittest.TestCase):
    def test_bitwise_equal_to_the_stock_forward(self):
        torch = _stack_or_skip()
        if torch is None:
            self.skipTest("needs torch + diffusers + CUDA")
        from pipeline import wan_lowvram

        stock = _model(torch)
        low = copy.deepcopy(stock)
        # 7 tokens a chunk: 3*6*5 = 90 tokens, so chunks straddle frames and
        # the last one is short.
        wan_lowvram.install(low, chunk=7)
        inputs = _inputs(torch)
        with torch.no_grad():
            want = stock(**inputs)[0]
            got = low(**inputs)[0]
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(torch.equal(got, want),
                        f"max |diff| {(got.float() - want.float()).abs().max().item()}")

    def test_group_offload_wraps_the_low_vram_forwards(self):
        """The offload hooks must wrap the swapped-in forwards, not the
        stock ones — install() first, then offload — and still give the
        same answer with every block streamed."""
        torch = _stack_or_skip()
        if torch is None:
            self.skipTest("needs torch + diffusers + CUDA")
        from diffusers.hooks import apply_group_offloading
        from pipeline import wan_lowvram

        stock = _model(torch)
        low = copy.deepcopy(stock).to("cpu")
        wan_lowvram.install(low, chunk=7)
        apply_group_offloading(
            low, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
            offload_type="block_level", num_blocks_per_group=1, use_stream=False,
        )
        inputs = _inputs(torch)
        with torch.no_grad():
            want = stock(**inputs)[0]
            got = low(**inputs)[0]
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(str(low.blocks[0].ffn.net[0].proj.weight.device), "cpu")


    def test_stream_weights_leaves_the_host_tensors_untouched(self):
        """stream_weights: same answer, and every host weight afterwards is
        the very tensor it was before — no host copy is ever made, which is
        what keeps mmap'd checkpoints out of anonymous RAM."""
        torch = _stack_or_skip()
        if torch is None:
            self.skipTest("needs torch + diffusers + CUDA")
        from pipeline import wan_lowvram

        stock = _model(torch)
        low = copy.deepcopy(stock).to("cpu")
        before = {n: t for n, t in list(low.named_parameters()) + list(low.named_buffers())}
        wan_lowvram.install(low, chunk=7)
        wan_lowvram.stream_weights(low, "cuda")
        inputs = _inputs(torch)
        with torch.no_grad():
            want = stock(**inputs)[0]
            got = low(**inputs)[0]
        self.assertTrue(torch.equal(got, want))
        after = dict(list(low.named_parameters()) + list(low.named_buffers()))
        self.assertEqual(set(after), set(before))
        for name, tensor in after.items():
            self.assertIs(tensor, before[name], name)


if __name__ == "__main__":
    unittest.main()
