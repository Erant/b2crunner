"""Loading a ComfyUI-format fp8_scaled Wan checkpoint into diffusers.

The expensive part of this — does every key line up, is anything missing,
do the shapes agree — needs no weights at all, only the checkpoint's
safetensors header. `tests/fixtures/wan22_vace_fp8_header.json.gz` is that
header, recorded from
`silveroxides/Wan_2.2-fp8_scaled_hybrid/wan2.2_fun_vace_high_noise_14B-fp8_scaled_original.safetensors`
via HTTP range request: 1827 tensor names with dtypes and shapes, 5 KB.
Meta tensors reconstructed from it exercise the whole mapping for free.

Skips where diffusers/torchao aren't installed, which is everywhere except
venv_wan22 — same convention as the cyber_6f golden-data tests.
"""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wan22_vace_fp8_header.json.gz"

# linoyts/Wan2.2-VACE-Fun-14B-diffusers' transformer/config.json. Inlined
# rather than fetched so the test needs no network.
TRANSFORMER_CONFIG = {
    "added_kv_proj_dim": None, "attention_head_dim": 128, "cross_attn_norm": True,
    "eps": 1e-06, "ffn_dim": 13824, "freq_dim": 256, "image_dim": None,
    "in_channels": 16, "num_attention_heads": 40, "num_layers": 40,
    "out_channels": 16, "patch_size": [1, 2, 2], "pos_embed_seq_len": None,
    "qk_norm": "rms_norm_across_heads", "rope_max_seq_len": 1024, "text_dim": 4096,
    "vace_in_channels": 96, "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
}


def _require_diffusers():
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"diffusers/torch not installed here: {exc}")


class TestFp8KeyMapping(unittest.TestCase):
    def setUp(self):
        _require_diffusers()
        import torch

        with gzip.open(FIXTURE, "rt") as f:
            self.header = json.load(f)
        self.dtypes = {
            "F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F32": torch.float32,
        }

    def _meta_state_dict(self):
        import torch

        with torch.device("meta"):
            return {
                name: torch.empty(shape, dtype=self.dtypes[dtype])
                for name, (dtype, shape) in self.header.items()
            }

    def test_the_fixture_looks_like_a_scaled_fp8_checkpoint(self):
        dtypes = {d for d, _ in self.header.values()}
        self.assertIn("F8_E4M3", dtypes)
        self.assertTrue(any(k.endswith(".scale_weight") for k in self.header))
        self.assertIn("scaled_fp8", self.header, "the ComfyUI format marker")
        self.assertTrue(any(k.startswith("vace_blocks.") for k in self.header),
                        "this must be a VACE checkpoint, not plain T2V")

    def test_conversion_matches_the_model_exactly(self):
        """No missing keys, no unexpected keys, no shape mismatches.

        This is the whole claim. If diffusers changes its Wan key mapping,
        or the published checkpoint's layout moves, this is what catches it
        — without downloading 17.6 GB to find out.
        """
        import torch
        from diffusers import WanVACETransformer3DModel

        from pipeline.wan_fp8 import convert_fp8_state_dict

        converted = convert_fp8_state_dict(self._meta_state_dict())
        with torch.device("meta"):
            model = WanVACETransformer3DModel(**TRANSFORMER_CONFIG)
        expected = model.state_dict()

        missing = sorted(set(expected) - set(converted))
        unexpected = sorted(set(converted) - set(expected))
        self.assertEqual(missing, [], f"{len(missing)} keys the model needs are absent")
        self.assertEqual(unexpected, [], f"{len(unexpected)} converted keys the model has no slot for")

        mismatched = [
            (k, tuple(expected[k].shape), tuple(converted[k].shape))
            for k in expected if tuple(expected[k].shape) != tuple(converted[k].shape)
        ]
        self.assertEqual(mismatched, [], "shape mismatches after conversion")

    def test_vestigial_keys_are_dropped_not_ignored(self):
        """The 6 bf16-layer scales and the format marker must go.

        They pair with nothing (their weights were deliberately left in
        bf16), and leaving them in is what turns an exact 1331-key match
        into a load error.
        """
        from pipeline.wan_fp8 import convert_fp8_state_dict

        converted = convert_fp8_state_dict(self._meta_state_dict())
        self.assertNotIn("scaled_fp8", converted)
        self.assertEqual([k for k in converted if k.endswith(".scale_weight")], [])


class TestFloat8Reconstruction(unittest.TestCase):
    """torchao's Float8Tensor really is (fp8 data, scale), and ours matches."""

    def setUp(self):
        _require_diffusers()
        try:
            import torchao  # noqa: F401
        except Exception as exc:  # broken installs raise, not just ImportError
            raise unittest.SkipTest(f"torchao unusable here: {exc}")
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("torchao float8 quantization needs a GPU")

    def test_a_rebuilt_tensor_dequantizes_identically(self):
        import torch
        from torchao.quantization import Float8WeightOnlyConfig, quantize_

        from pipeline.wan_fp8 import _build_float8

        model = torch.nn.Sequential(
            torch.nn.Linear(256, 128, bias=False).to(torch.bfloat16).cuda()
        )
        quantize_(model, Float8WeightOnlyConfig())
        weight = model[0].weight

        # torchao is per-row; a checkpoint's per-tensor scale is the case
        # where every row shares one value. Collapse to that and rebuild.
        single = weight.scale.flatten()[:1].clone()
        rebuilt = _build_float8(weight.qdata.clone(), single)

        self.assertEqual(rebuilt.qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(rebuilt.scale.shape), (128, 1))
        expected = weight.qdata.to(torch.float32) * single.to(torch.float32)
        torch.testing.assert_close(
            rebuilt.dequantize().float(), expected, rtol=0, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
