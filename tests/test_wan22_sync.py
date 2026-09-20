"""`wan22_vace_denoise`'s in-loop 3D synchronisation (steps/wan22_sync.py).

What is pinned here is the wiring, not the 3D: that `sync_steps` is
refused unless the steps it names run euler and the dataset inputs are
wired; that at a sync step the euler step is taken from the
synchroniser's velocity and at every other step from the model's; that
the sigma handed to the synchroniser is the one the euler step then
reads; and that under UniPC the synchroniser is never consulted. The
band arithmetic is checked where torch is available (it is not in the
main venv — same convention as test_wan22_conditioning.py) and skipped
where it is not.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import scipy.stats  # noqa: F401  (see test_wan22_conditioning.py)

from tests.test_wan22_conditioning import _FakeScheduler


class _Camera:
    """The four numbers and two arrays the synchroniser reads off a camera."""

    def __init__(self, width=16, height=16):
        self.width, self.height = width, height
        self.fx = self.fy = float(width)
        self.cx, self.cy = width / 2.0, height / 2.0
        self.position = np.zeros(3, np.float32)
        self.rotation = np.eye(3, dtype=np.float32)


def _run(inputs_extra=None, **params):
    """test_wan22_conditioning's `_run`, with extra inputs and the pass's
    synchroniser captured while `pipe()` has it."""
    from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

    calls = {}

    class _Pipe:
        transformer = types.SimpleNamespace(
            config=types.SimpleNamespace(vace_layers=[0, 5, 10, 15, 20, 25, 30, 35])
        )
        transformer_2 = transformer

        def __init__(self):
            self.scheduler = _FakeScheduler()

        def register_to_config(self, **kwargs):
            calls.update(kwargs)

        def __call__(self, **kwargs):
            calls.update(kwargs)
            calls["sync"] = step._sync
            return types.SimpleNamespace(frames=[np.zeros((1, 4, 4, 3), dtype=np.float32)])

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.Tensor = type("Tensor", (), {})

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
    inputs = {"control_video": [frame], "control_masks": [np.zeros((4, 4), dtype=np.uint8)]}
    inputs.update(inputs_extra or {})
    with patch.dict(sys.modules, {"torch": torch}):
        step.run(inputs, resolved)
    return step, calls


_DATASET = {"cameras": [_Camera()], "image_names": ["frame_00001.png"]}


class TestBuildSync(unittest.TestCase):
    def test_no_sync_steps_builds_nothing(self):
        _, calls = _run()
        self.assertIsNone(calls["sync"])

    def test_a_sync_step_on_unipc_is_refused_by_the_samplers_name(self):
        with self.assertRaises(ValueError) as caught:
            _run(_DATASET, sync_steps=[3])
        self.assertIn("sampler_low: euler", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            _run(_DATASET, sync_steps=[1], sampler_low="euler")
        self.assertIn("sampler_high: euler", str(caught.exception))

    def test_a_sync_step_outside_the_run_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _run(_DATASET, sync_steps=[6], sampler_low="euler")
        self.assertIn("6-step", str(caught.exception))

    def test_the_dataset_inputs_are_required(self):
        with self.assertRaises(ValueError) as caught:
            _run(sync_steps=[3], sampler_low="euler")
        self.assertIn("dataset.cameras", str(caught.exception))

    def test_a_mix_of_the_wrong_length_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _run(_DATASET, sync_steps=[2, 3, 4], sync_mix=[1.0, 0.5], sampler_low="euler")
        self.assertIn("sync_mix", str(caught.exception))

    def test_one_mix_entry_covers_every_sync_step(self):
        _, calls = _run(_DATASET, sync_steps=[2, 3, 4], sync_mix=[0.5], sampler_low="euler")
        self.assertEqual(calls["sync"].mix, {2: 0.5, 3: 0.5, 4: 0.5})

    def test_the_synchroniser_is_built_for_the_pass_and_dropped_after_it(self):
        step, calls = _run(_DATASET, sync_steps=[2, 3, 4], sync_mix=[1, 1, 0.5], sampler_low="euler")
        sync = calls["sync"]
        self.assertIsNotNone(sync)
        self.assertEqual(sync.steps, [2, 3, 4])
        self.assertEqual(sync.mix[4], 0.5)
        self.assertEqual(sync.n_ref, 0)
        self.assertIsNone(step._sync)

    def test_a_reference_image_is_one_untouched_latent_frame(self):
        _, calls = _run({**_DATASET, "reference_image": np.zeros((4, 4, 3), np.uint8)},
                        sync_steps=[3], sampler_low="euler")
        self.assertEqual(calls["sync"].n_ref, 1)

    def test_cameras_are_rescaled_to_the_denoise_size(self):
        cameras = [_Camera(width=8, height=8)]
        _, calls = _run({"cameras": cameras, "image_names": ["a.png"]}, sync_steps=[3], sampler_low="euler")
        cam = calls["sync"].cameras[0]
        self.assertEqual((int(cam.width), int(cam.height)), (16, 16))
        self.assertAlmostEqual(float(cam.fx), 16.0)
        self.assertAlmostEqual(float(cam.cx), 8.0)

    def test_masks_are_dilated_and_kept_per_view(self):
        mask = np.zeros((16, 16), np.float32)
        mask[8, 8] = 1.0
        _, calls = _run({**_DATASET, "sync_masks": [mask]}, sync_steps=[3], sampler_low="euler", sync_mask_dilate_px=2)
        grown = calls["sync"].masks[0]
        self.assertEqual(grown.dtype, np.uint8)
        self.assertEqual(int(grown[8, 8]), 255)
        self.assertEqual(int(grown[8, 10]), 255)
        self.assertEqual(int(grown[8, 11]), 0)


class _FakeSync:
    def __init__(self, at, velocity=3.0):
        self.at = set(at)
        self.velocity_value = velocity
        self.calls = []

    def wants(self, index):
        return index in self.at

    def velocity(self, model_output, sample, sigma, index):
        self.calls.append((model_output, sample, sigma, index))
        return self.velocity_value


class TestTheSeam(unittest.TestCase):
    def _drive(self, step, sample=0.0, velocity=1.0):
        scheduler = step._pipe.scheduler
        samples = []
        for timestep in list(scheduler.timesteps):
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
            samples.append(sample)
        return samples

    def test_the_euler_step_is_taken_from_the_synchronisers_velocity_at_its_steps(self):
        step, _ = _run(sampler_high="euler", sampler_low="euler")
        sync = _FakeSync(at=[3], velocity=3.0)
        step._sync = sync
        scheduler = step._pipe.scheduler
        sigmas = list(scheduler.sigmas)
        samples = self._drive(step, sample=0.0, velocity=1.0)
        # Steps 0-2 and 4-5 on the model's velocity 1.0, step 3 on 3.0.
        expected = 0.0
        for i in range(6):
            expected += (sigmas[i + 1] - sigmas[i]) * (3.0 if i == 3 else 1.0)
            self.assertAlmostEqual(samples[i], expected)
        self.assertEqual(len(sync.calls), 1)
        model_output, sample, sigma, index = sync.calls[0]
        self.assertEqual(index, 3)
        self.assertEqual(model_output, 1.0)
        self.assertAlmostEqual(sample, samples[2])
        self.assertAlmostEqual(sigma, sigmas[3])

    def test_a_synchroniser_that_wants_no_step_changes_nothing(self):
        step, _ = _run(sampler_high="euler", sampler_low="euler")
        step._sync = _FakeSync(at=[])
        samples = self._drive(step, sample=0.0, velocity=1.0)
        self.assertAlmostEqual(samples[-1], -1.0)

    def test_unipc_steps_never_consult_the_synchroniser(self):
        step, _ = _run(sampler_high="euler")
        sync = _FakeSync(at=[1, 3])
        step._sync = sync
        self._drive(step)
        self.assertEqual([c[3] for c in sync.calls], [1])
        self.assertEqual(step._pipe.scheduler.stepped, [("uni_pc", i) for i in range(2, 6)])


def _torch():
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    return sys.modules["torch"]


@unittest.skipUnless(_torch() is not None and hasattr(_torch(), "fft"), "torch is not installed here")
class TestBands(unittest.TestCase):
    def test_the_lowpass_keeps_dc_and_drops_nyquist(self):
        from pipeline.steps.wan22_sync import radial_lowpass

        mask = radial_lowpass(32, 32, 1.0 / 6.0)
        self.assertEqual(tuple(mask.shape), (32, 32))
        self.assertAlmostEqual(float(mask[0, 0]), 1.0)
        self.assertAlmostEqual(float(mask[16, 16]), 0.0)

    def test_mix_one_takes_the_low_band_and_leaves_the_rest(self):
        import torch

        from pipeline.steps.wan22_sync import blend_bands, radial_lowpass

        torch.manual_seed(0)
        x0 = torch.randn(1, 2, 3, 32, 32)
        projected = torch.randn(1, 2, 3, 32, 32)
        mask = radial_lowpass(32, 32, 1.0 / 6.0, edge=0.0)
        out = blend_bands(x0, projected, mask, 1.0)
        f_out, f_x0, f_pr = (torch.fft.fft2(t) for t in (out, x0, projected))
        low = mask > 0.5
        self.assertTrue(torch.allclose(f_out[..., low], f_pr[..., low], atol=1e-4))
        self.assertTrue(torch.allclose(f_out[..., ~low], f_x0[..., ~low], atol=1e-4))
        self.assertTrue(torch.allclose(blend_bands(x0, projected, mask, 0.0), x0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
