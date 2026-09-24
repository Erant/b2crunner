"""`wan22_vace_denoise`'s Weak skeleton: the stick-free drawing on the steps after the first.

Measured 2026-09-24 (run 1127f0): every denoise step that sees the DWPose
sticks can paint them as costume trim, and the step that needs them is the
first. The step therefore takes a second control video, `control_video_alt`
— the same drawing without the skeleton — and swaps it into the call on
every step `skeleton_steps` does not name.

Three things are pinned. The swap touches the VIDEO half of
`control_hidden_states` and leaves the mask half alone (both controls share
one VACE mask). The stick-free video is encoded through exactly the main
video's preprocessing, and the generator comes back where the stock call
left it — the VAE encode draws from its posterior, and one extra draw would
make a weak-skeleton run a different sample from the baseline at the same
seed, which is the comparison the setting was measured by. And a run that
does not hand over the alternate control is the run it always was.

No torch here (see tests/test_wan22_conditioning.py), so tensors are stubs
that carry a list of channel labels: slicing, `torch.cat` along channels and
`.to()` are the only things the hook does to them.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class _Tensor:
    """A (batch, channels, ...) tensor reduced to its channel labels."""

    def __init__(self, channels, device="cuda:0", dtype="bfloat16"):
        self.channels = list(channels)
        self.device = device
        self.dtype = dtype

    @property
    def shape(self):
        return (1, len(self.channels))

    def __getitem__(self, index):
        _, chans = index
        return _Tensor(self.channels[chans], self.device, self.dtype)

    def to(self, device, dtype):
        return _Tensor(self.channels, device, dtype)


def _cat(tensors, dim):
    assert dim == 1
    first = tensors[0]
    return _Tensor([c for t in tensors for c in t.channels], first.device, first.dtype)


_TORCH = types.ModuleType("torch")
_TORCH.cat = _cat


class _Timestep:
    def __init__(self, value):
        self.value = value

    def flatten(self):
        return [self.value, self.value]


TIMESTEPS = (999.0, 980.0, 929.0, 833.0, 655.0, 336.0)


class TestAltControlHook(unittest.TestCase):
    def _step(self, alt=None, skeleton_steps=(1,)):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        step = Wan22VaceDenoiseStep()
        step._pipe = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(timesteps=list(TIMESTEPS))
        )
        step._alt_latents = alt
        step._skeleton_steps = set(skeleton_steps)
        return step

    def _call(self, step, timestep):
        from pipeline.steps.wan22_vace_denoise import _alt_control_hook

        incoming = _Tensor(["v0", "v1", "m0", "m1", "m2"], device="cuda:0", dtype="bf16")
        kwargs = {"control_hidden_states": incoming, "timestep": _Timestep(timestep)}
        with patch.dict(sys.modules, {"torch": _TORCH}):
            return _alt_control_hook(step)(None, (), kwargs)

    def test_without_an_alternate_control_the_call_is_left_alone(self):
        self.assertIsNone(self._call(self._step(alt=None), TIMESTEPS[3]))

    def test_the_skeleton_step_keeps_the_drawing_it_was_given(self):
        step = self._step(alt=_Tensor(["a0", "a1"], device="cpu", dtype="f32"))
        self.assertIsNone(self._call(step, TIMESTEPS[0]))

    def test_every_other_step_gets_the_stick_free_video_and_the_same_mask(self):
        step = self._step(alt=_Tensor(["a0", "a1"], device="cpu", dtype="f32"))
        for timestep in TIMESTEPS[1:]:
            _, kwargs = self._call(step, timestep)
            swapped = kwargs["control_hidden_states"]
            self.assertEqual(swapped.channels, ["a0", "a1", "m0", "m1", "m2"])
            # On the device and in the dtype diffusers resolved for the call.
            self.assertEqual((swapped.device, swapped.dtype), ("cuda:0", "bf16"))

    def test_the_step_is_read_off_the_timestep(self):
        """A cast timestep still names its step; step 3 here keeps the sticks."""
        step = self._step(alt=_Tensor(["a0", "a1"]), skeleton_steps=(1, 3))
        self.assertIsNone(self._call(step, TIMESTEPS[2] + 0.4))
        self.assertIsNotNone(self._call(step, TIMESTEPS[1]))

    def test_a_pipeline_that_stopped_passing_the_control_is_an_error(self):
        from pipeline.steps.wan22_vace_denoise import _alt_control_hook

        step = self._step(alt=_Tensor(["a0"]))
        with patch.dict(sys.modules, {"torch": _TORCH}), self.assertRaises(RuntimeError):
            _alt_control_hook(step)(None, (), {"timestep": _Timestep(TIMESTEPS[1])})


class _Generator:
    def __init__(self):
        self.state = 0

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class TestEncodeAltControl(unittest.TestCase):
    def _pipe(self, log):
        class _Pipe:
            def preprocess_conditions(self, video, mask, reference_images, *args, **kwargs):
                log.append(("pre", video, mask, reference_images, args))
                return (f"processed:{video}", f"processed:{mask}", reference_images)

            def prepare_video_latents(self, video, mask, reference_images, generator=None, device=None):
                log.append(("encode", video, mask, reference_images, device))
                generator.state += 1  # the posterior draw
                return f"latents:{video}"

        return _Pipe()

    def test_the_alternate_video_takes_the_main_videos_path(self):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep, _encode_alt_control

        log = []
        step, pipe = Wan22VaceDenoiseStep(), self._pipe(log)
        installed = _encode_alt_control(step, pipe, "alt")
        self.assertEqual(installed, ["preprocess_conditions", "prepare_video_latents"])

        video, mask, refs = pipe.preprocess_conditions("main", "m", "refs", 2, 81)
        generator = _Generator()
        main = pipe.prepare_video_latents(video, mask, refs, generator, "cuda")

        self.assertEqual(main, "latents:processed:main")
        self.assertEqual(step._alt_latents, "latents:processed:alt")
        self.assertEqual(log[1], ("pre", "alt", "m", "refs", (2, 81)))
        self.assertEqual(log[3], ("encode", "processed:alt", "processed:m", "refs", "cuda"))

    def test_the_generator_is_left_where_the_stock_call_leaves_it(self):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep, _encode_alt_control

        step, pipe = Wan22VaceDenoiseStep(), self._pipe([])
        _encode_alt_control(step, pipe, "alt")
        video, mask, refs = pipe.preprocess_conditions("main", "m", "refs")
        generator = _Generator()
        pipe.prepare_video_latents(video, mask, refs, generator, "cuda")
        self.assertEqual(generator.state, 1)


class TestRunWiresTheAlternateControl(unittest.TestCase):
    """`run()` against a stub pipeline that records what it was called with."""

    def _run(self, alt_frames=None, n_frames=3, **params):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        seen = {}

        class _Pipe:
            transformer = types.SimpleNamespace(
                config=types.SimpleNamespace(vace_layers=[0, 5, 10, 15, 20, 25, 30, 35])
            )
            transformer_2 = transformer
            scheduler = types.SimpleNamespace(
                config=types.SimpleNamespace(num_train_timesteps=1000),
                register_to_config=lambda **kwargs: None,
                set_timesteps=lambda *a, **k: None,
                timesteps=list(TIMESTEPS),
            )

            def register_to_config(self, **kwargs):
                pass

            def preprocess_conditions(self, video, mask, reference_images, *a, **k):
                return video, mask, reference_images

            def prepare_video_latents(self, video, mask, reference_images, generator=None, device=None):
                seen["encodes"] = seen.get("encodes", 0) + 1
                return "latents"

            def __call__(self, **kwargs):
                self.preprocess_conditions(kwargs["video"], kwargs["mask"], None)
                self.prepare_video_latents(kwargs["video"], kwargs["mask"], None, _Generator(), "cuda")
                seen["alt"] = step._alt_latents
                seen["skeleton_steps"] = step._skeleton_steps
                return types.SimpleNamespace(frames=[np.zeros((1, 4, 4, 3), dtype=np.float32)])

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.Tensor = type("Tensor", (), {})
        torch.Generator = lambda device=None: types.SimpleNamespace(manual_seed=lambda s: None)

        step = Wan22VaceDenoiseStep()
        step._pipe = _Pipe()
        step._configure_sampler = lambda pipe, params: None
        step._set_expert_split = lambda pipe, high, n: None
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        inputs = {
            "control_video": [frame] * n_frames,
            "control_masks": [np.zeros((4, 4), dtype=np.uint8)] * n_frames,
        }
        if alt_frames is not None:
            inputs["control_video_alt"] = alt_frames
        resolved = step.resolve_params({"width": 16, "height": 16, **params})
        with patch.dict(sys.modules, {"torch": torch}):
            step.run(inputs, resolved)
        return step, seen

    def test_without_the_input_the_control_is_encoded_once(self):
        step, seen = self._run(skeleton_steps=[1])
        self.assertEqual(seen["encodes"], 1)
        self.assertIsNone(seen["alt"])

    def test_with_it_the_stick_free_video_is_encoded_for_the_hooks(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        step, seen = self._run(alt_frames=[frame] * 3, skeleton_steps=[1])
        self.assertEqual(seen["encodes"], 2)
        self.assertEqual(seen["alt"], "latents")
        self.assertEqual(seen["skeleton_steps"], {1})

    def test_the_pass_leaves_nothing_behind_for_the_next(self):
        """A resident worker's pass 2 must not inherit pass 1's swap."""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        step, _ = self._run(alt_frames=[frame] * 3, skeleton_steps=[1])
        self.assertIsNone(step._alt_latents)
        self.assertNotIn("prepare_video_latents", step._pipe.__dict__)
        self.assertNotIn("preprocess_conditions", step._pipe.__dict__)

    def test_a_copy_of_a_different_length_is_refused(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self._run(alt_frames=[frame] * 2, skeleton_steps=[1])

    def test_a_skeleton_step_outside_the_run_is_refused(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self._run(alt_frames=[frame] * 3, skeleton_steps=[7])


if __name__ == "__main__":
    unittest.main()
