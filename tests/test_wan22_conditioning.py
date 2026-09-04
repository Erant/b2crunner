"""`wan22_vace_denoise`'s VACE conditioning scale — per layer, per expert.

The knobs exist because the control video here is a DRAWING: a flat
silhouette under a DWPose skeleton, in a gridded room. At `strength: 1.0`
the skeleton can survive the denoise as ink rather than being read as pose,
and with this checkpoint's boundary_ratio (0.875) and the scheduler's
flow_shift (3.0) four of a 6-step run's six steps are the LOW-noise
expert's — so the useful cap is the one that leaves the high-noise steps,
where the pose is set, alone.

Two things are worth pinning. First, that a run setting neither knob is
byte-for-byte the run that came before them: `strength` alone must still
mean one flat scale at every layer. Second, that the override reaches the
low-noise expert at all — diffusers hands both experts the same tensor, so
this step goes in through a module pre-hook, and a pre-hook that quietly
stopped firing (accelerate's offload replaces `forward` on every pass) would
denoise at the wrong strength while logging nothing.

No torch and no diffusers here (neither is installed outside venv_wan22),
so the tensor the hook rewrites is a stub — same convention as
tests/test_wan22_residency.py. `new_tensor` is the only method the hook
uses, and that it uses THAT rather than building a tensor of its own is the
point: it is what carries the device and dtype diffusers already resolved.
"""

from __future__ import annotations

import unittest


class _FakeScale:
    """Stands in for the scale tensor diffusers built for the call."""

    def __init__(self, values, device="cuda:0", dtype="bfloat16"):
        self.values = list(values)
        self.device = device
        self.dtype = dtype

    def new_tensor(self, values):
        return _FakeScale(values, device=self.device, dtype=self.dtype)


def _conditioning_scale(*args):
    from pipeline.steps.wan22_vace_denoise import _conditioning_scale as fn

    return fn(*args)


class TestConditioningScale(unittest.TestCase):
    def test_no_taper_is_the_plain_scale_at_every_layer(self):
        self.assertEqual(_conditioning_scale(0.8, None, 8), [0.8] * 8)

    def test_a_taper_multiplies_each_layers_share(self):
        taper = [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3]
        self.assertEqual(
            _conditioning_scale(1.0, taper, 8),
            [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3],
        )
        self.assertEqual(_conditioning_scale(0.5, taper, 8)[-1], 0.15)

    def test_an_all_ones_taper_is_the_untapered_scale(self):
        """The declared default, spelled out: it must change nothing."""
        self.assertEqual(_conditioning_scale(0.8, [1.0] * 8, 8), [0.8] * 8)

    def test_a_taper_of_the_wrong_length_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            _conditioning_scale(1.0, [1.0, 0.5], 8)
        message = str(caught.exception)
        self.assertIn("strength_layers", message)
        self.assertIn("2", message)
        self.assertIn("8", message)

    def test_yaml_integers_are_taken_as_scales(self):
        """A workflow writing `[1, 1, 1, 1, 1, 1, 0, 0]` means floats."""
        self.assertEqual(
            _conditioning_scale(1.0, [1, 1, 1, 1, 1, 1, 0, 0], 8),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        )


class TestLowNoiseScaleHook(unittest.TestCase):
    def _hook(self, low_scale):
        from pipeline.steps.wan22_vace_denoise import (
            Wan22VaceDenoiseStep,
            _low_noise_scale_hook,
        )

        step = Wan22VaceDenoiseStep()
        step._low_scale = low_scale
        return step, _low_noise_scale_hook(step)

    def test_without_an_override_the_call_is_left_alone(self):
        """`strength_low` unset must be indistinguishable from no hook."""
        _, hook = self._hook(None)
        scale = _FakeScale([1.0] * 8)
        self.assertIsNone(hook(None, (), {"control_hidden_states_scale": scale}))

    def test_the_override_replaces_the_scale_the_pipeline_built(self):
        _, hook = self._hook([0.5] * 8)
        scale = _FakeScale([1.0] * 8)
        args, kwargs = hook(None, ("positional",), {"control_hidden_states_scale": scale})
        self.assertEqual(args, ("positional",))
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [0.5] * 8)

    def test_the_replacement_keeps_the_devices_dtype_and_placement(self):
        """new_tensor, not a fresh tensor: the alternative lands on the CPU."""
        _, hook = self._hook([0.5] * 8)
        scale = _FakeScale([1.0] * 8, device="cuda:0", dtype="bfloat16")
        _, kwargs = hook(None, (), {"control_hidden_states_scale": scale})
        replacement = kwargs["control_hidden_states_scale"]
        self.assertEqual(replacement.device, "cuda:0")
        self.assertEqual(replacement.dtype, "bfloat16")

    def test_the_step_is_read_per_call_not_closed_over(self):
        """One resident pipeline serves both passes, which disagree here."""
        step, hook = self._hook([0.5] * 8)
        step._low_scale = [0.25] * 8
        _, kwargs = hook(None, (), {"control_hidden_states_scale": _FakeScale([1.0] * 8)})
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [0.25] * 8)

    def test_a_pipeline_that_stopped_passing_the_scale_is_an_error(self):
        """Silently denoising at the wrong strength is the bad outcome."""
        _, hook = self._hook([0.5] * 8)
        with self.assertRaises(RuntimeError) as caught:
            hook(None, (), {})
        self.assertIn("strength_low", str(caught.exception))


class TestRunPassesTheScaleThrough(unittest.TestCase):
    """The wiring itself: params -> the kwarg diffusers reads.

    This is the seam that rots quietly. `conditioning_scale` accepting a
    list is a diffusers API detail, and the number of entries it wants comes
    off the loaded model's config rather than a constant here, so a stub
    pipeline carrying the real `vace_layers` is what pins both.
    """

    def _run(self, **params):
        import sys
        import types
        from unittest.mock import patch

        import numpy as np

        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        calls = {}

        class _Pipe:
            transformer = types.SimpleNamespace(
                config=types.SimpleNamespace(vace_layers=[0, 5, 10, 15, 20, 25, 30, 35])
            )
            transformer_2 = transformer

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return types.SimpleNamespace(
                    frames=[np.zeros((1, 4, 4, 3), dtype=np.float32)]
                )

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        class _Generator:
            def __init__(self, device=None):
                self.device = device

            def manual_seed(self, seed):
                return self

        torch.Generator = _Generator

        step = Wan22VaceDenoiseStep()
        step._pipe = _Pipe()
        step._cpu_offload = True
        resolved = step.resolve_params({"width": 16, "height": 16, "seed": 0, **params})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        inputs = {
            "control_video": [frame],
            "control_masks": [np.zeros((4, 4), dtype=np.uint8)],
        }
        with patch.dict(sys.modules, {"torch": torch}):
            step.run(inputs, resolved)
        return step, calls

    def test_a_plain_strength_still_reaches_every_layer(self):
        step, calls = self._run(strength=0.8)
        self.assertEqual(calls["conditioning_scale"], [0.8] * 8)
        self.assertIsNone(step._low_scale)

    def test_strength_low_is_left_for_the_hook_not_sent_to_the_pipeline(self):
        """diffusers has no per-expert param — sending one would be ignored."""
        step, calls = self._run(strength=1.0, strength_low=0.5)
        self.assertEqual(calls["conditioning_scale"], [1.0] * 8)
        self.assertEqual(step._low_scale, [0.5] * 8)
        self.assertNotIn("strength_low", calls)

    def test_the_taper_reaches_both_experts(self):
        taper = [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3]
        step, calls = self._run(strength=1.0, strength_low=0.5, strength_layers=taper)
        self.assertEqual(calls["conditioning_scale"], taper)
        self.assertEqual(step._low_scale, [0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.15])

    def test_the_layer_count_comes_off_the_loaded_model(self):
        with self.assertRaises(ValueError):
            self._run(strength=1.0, strength_layers=[1.0] * 4)


class TestDeclaration(unittest.TestCase):
    def test_neither_knob_forces_the_pipeline_to_be_rebuilt(self):
        """Both are per-call. In LOAD_PARAMS they would cost a 47 GB reload."""
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        for name in ("strength", "strength_low", "strength_layers"):
            self.assertNotIn(name, Wan22VaceDenoiseStep.LOAD_PARAMS)

    def test_the_declared_defaults_leave_strength_alone(self):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name: param for param in Wan22VaceDenoiseStep.PARAMS}
        self.assertIsNone(declared["strength_low"].default)
        self.assertIsNone(declared["strength_layers"].default)
        self.assertEqual(declared["strength"].default, 1.0)

    def test_strength_low_draws_as_a_box_that_can_be_left_empty(self):
        """A min AND a max would make webui draw a slider (see _control).

        A slider has no empty position: the value it hands back for a None
        default is its minimum, which here would be 0.0 — VACE switched off
        on the low-noise expert, from a run that set nothing.
        """
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name: param for param in Wan22VaceDenoiseStep.PARAMS}
        low = declared["strength_low"]
        self.assertIsNone(low.minimum)
        self.assertIsNone(low.maximum)


if __name__ == "__main__":
    unittest.main()
