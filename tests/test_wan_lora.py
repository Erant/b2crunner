"""pipeline/wan_lora.py: the LoRA add as one addmm_, and peft's own forward
for everything that is not the plain inference case. CPU is enough."""

from __future__ import annotations

import unittest


def _peft_or_skip():
    try:
        import torch
        from peft import LoraConfig, inject_adapter_in_model  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return torch


def _model(torch, dropout: float = 0.0):
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 48), torch.nn.GELU(), torch.nn.Linear(48, 16))
    model = inject_adapter_in_model(
        LoraConfig(r=4, lora_alpha=8, lora_dropout=dropout, target_modules=["0", "2"]), model)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:   # zero-initialised: give the adapter signal
                p.normal_(0, 0.1)
    return model.to(torch.bfloat16).eval()


class TestLoraAddmm(unittest.TestCase):
    def test_same_answer_as_peft_within_bf16_rounding(self):
        torch = _peft_or_skip()
        if torch is None:
            self.skipTest("needs torch + peft")
        from pipeline.wan_lora import enable_lora_addmm

        model = _model(torch)
        x = torch.randn(3, 7, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            want = model(x).float()
            self.assertEqual(enable_lora_addmm(model), 2)
            got = model(x).float()
            base = model[2].base_layer(model[1](model[0].base_layer(x))).float()
        self.assertLess(((got - want).norm() / want.norm()).item(), 0.01)
        # ...and the adapter is really in it.
        self.assertGreater(((got - base).norm() / base.norm()).item(), 0.05)

    def test_disabled_adapters_go_through_peft(self):
        torch = _peft_or_skip()
        if torch is None:
            self.skipTest("needs torch + peft")
        from pipeline.wan_lora import enable_lora_addmm

        model = _model(torch)
        enable_lora_addmm(model)
        x = torch.randn(5, 32, dtype=torch.bfloat16)
        for layer in (model[0], model[2]):
            layer.enable_adapters(False)
        with torch.no_grad():
            got = model(x)
            want = model[2].base_layer(model[1](model[0].base_layer(x)))
        self.assertTrue(torch.equal(got, want))

    def test_dropout_goes_through_peft(self):
        """A non-Identity dropout is peft's path, so training-mode dropout
        would still apply; in eval both agree."""
        torch = _peft_or_skip()
        if torch is None:
            self.skipTest("needs torch + peft")
        from pipeline import wan_lora

        model = _model(torch, dropout=0.1)
        x = torch.randn(5, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            want = model(x)
            wan_lora.enable_lora_addmm(model)
            got = model(x)
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    unittest.main()
