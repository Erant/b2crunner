"""`wan22_vace_denoise`'s VACE conditioning scale — per denoise step, per layer.

The knobs exist because the control video here is a DRAWING: a flat
silhouette under a DWPose skeleton, in a gridded room. At a scale of 1.0
the skeleton can survive the denoise as ink rather than being read as pose,
and at the default split four of a 6-step run's six steps are the LOW-noise
expert's — so the useful fade is the one that leaves the opening steps,
where the pose is set, alone.

That split is a setting rather than a consequence: `steps_high` and
`steps_low` are the two experts' step counts, and the step computes the
`boundary_ratio` that delivers them (`_expert_boundary_ratio`). The pinning
that matters most is that the declared 2/4 still selects exactly what the
checkpoint's own 0.875 selected — this feature is a control over something
that already happened, and a run that asks for nothing new must denoise
identically.

`strength` is the whole of it: one scale per denoise step, first to last.
It was three params until 2026-09-04 (`strength`, a per-expert
`strength_low`, and a `strength_steps` multiplying both), which meant no
single number said how hard VACE was pushing at a given moment.

Three things are worth pinning. First, that a run giving `strength` one
entry is byte-for-byte the run that came before schedules existed: one flat
scale at every layer, handed to the call, no hook fired. Second, that a
schedule reaches the low-noise expert at all — diffusers hands both experts
the same tensor, so this step goes in through a module pre-hook, and a
pre-hook that quietly stopped firing (accelerate's offload replaces
`forward` on every pass) would denoise at the wrong strength while logging
nothing. Third, that a per-step schedule lands on the right step: the index
is read off the timestep the call carries rather than counted, and an
off-by-one there would fade the control video out a step early on every run
without failing.

No torch and no diffusers here (neither is installed outside venv_wan22),
so the tensor the hook rewrites is a stub — same convention as
tests/test_wan22_residency.py. `new_tensor` is the only method the hook
uses, and that it uses THAT rather than building a tensor of its own is the
point: it is what carries the device and dtype diffusers already resolved.
The timestep the hook reads is stubbed the same way, by `flatten()[0]`,
which is the one thing it does to it.
"""

from __future__ import annotations

import types
import unittest

# Loaded here, outside any `patch.dict(sys.modules, ...)`: the beta
# schedule's quantile function is scipy's, and the run harness patches
# `torch` into sys.modules for the duration of `run()`. A scipy.stats first
# imported INSIDE that patch is removed with it on exit, and scipy's
# array-API shim refuses to be imported twice in one process — every later
# test would then find no scipy at all.
import scipy.stats  # noqa: F401


class _FakeScale:
    """Stands in for the scale tensor diffusers built for the call."""

    def __init__(self, values, device="cuda:0", dtype="bfloat16"):
        self.values = list(values)
        self.device = device
        self.dtype = dtype

    def new_tensor(self, values):
        return _FakeScale(values, device=self.device, dtype=self.dtype)


class _FakeTimestep:
    """`t.expand(batch)`: one value, repeated once per frame."""

    def __init__(self, value, frames=2):
        self.value = value
        self.frames = frames

    def flatten(self):
        return [self.value] * self.frames


class _FakeScheduler:
    """Carries the timesteps `pipe()` set before the loop, and nothing else.

    `set_timesteps` reproduces diffusers' flow-sigma branch rather than
    returning a canned list, because the expert split is placed against
    these numbers: sigmas linspace(1, 1/n, n) unless custom `sigmas` are
    handed in (which is how the step delivers ComfyUI's beta schedule),
    shifted by `shift * s / (1 + (shift - 1) * s)`, times
    num_train_timesteps. Built fresh at shift 3.0 — the HF config's value
    — the way the real one is; the step then writes its own shift through
    `register_to_config`, as it does to the real one. At n=6 and shift 3.0
    the linspace gives 1000, 937.5, 857.1, 750, 600, 375, the values the
    step's docstring records from the real scheduler, which is what makes
    this stub worth trusting for the boundary arithmetic.
    """

    def __init__(self, timesteps=(), shift=3.0):
        self.timesteps = list(timesteps)
        self.shift = shift
        self.config = types.SimpleNamespace(
            num_train_timesteps=1000, use_flow_sigmas=True,
            solver_type="bh2", solver_order=2,
        )
        # The multistep history, as UniPCMultistepScheduler keeps it.
        self.model_outputs = [None, None]
        self.timestep_list = [None, None]
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        # Which sampler took each step, in order — what a mixed run is read
        # off. The euler steps append themselves from `_euler_step`'s side
        # effect on `_step_index`, so this records only UniPC's own.
        self.stepped = []

    def register_to_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        if "flow_shift" in kwargs:
            self.shift = float(kwargs["flow_shift"])

    def set_timesteps(self, num_inference_steps, device=None, sigmas=None):
        n = num_inference_steps
        if sigmas is None:
            sigmas = [1.0 - (1.0 - 1.0 / n) * i / (n - 1) for i in range(n)] if n > 1 else [1.0]
        shifted = [self.shift * s / (1.0 + (self.shift - 1.0) * s) for s in sigmas]
        self.timesteps = [s * self.config.num_train_timesteps for s in shifted]
        # diffusers' `final_sigmas_type: zero`: the step sigmas plus a 0.
        self.sigmas = shifted + [0.0]
        # As UniPCMultistepScheduler.set_timesteps does: a new schedule has
        # taken no steps yet.
        self._step_index = None

    def _init_step_index(self, timestep):
        """UniPC's own: the position of this timestep in the schedule."""
        value = float(timestep)
        self._step_index = min(
            range(len(self.timesteps)), key=lambda i: abs(self.timesteps[i] - value)
        )

    def step(self, model_output, timestep, sample, return_dict=True):
        """UniPC's step as far as these tests read it: it advances the index
        and says it ran. The solver arithmetic is diffusers', not this."""
        if self._step_index is None:
            self._init_step_index(timestep)
        self.stepped.append(("uni_pc", self._step_index))
        self._step_index += 1
        return (sample,) if not return_dict else types.SimpleNamespace(prev_sample=sample)


def _conditioning_scale(*args):
    from pipeline.steps.wan22_vace_denoise import _conditioning_scale as fn

    return fn(*args)


def _scale_schedule(*args):
    from pipeline.steps.wan22_vace_denoise import _scale_schedule as fn

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


class TestScaleSchedule(unittest.TestCase):
    """`strength` itself: one scale per denoise step, or one for all of them."""

    def test_one_entry_is_one_constant_entry(self):
        """Shortness is the signal: one entry means "never ask the step"."""
        self.assertEqual(_scale_schedule([0.8], None, 8, 6), [[0.8] * 8])

    def test_the_schedule_fades_the_scale_across_the_run(self):
        plan = _scale_schedule([1, 1, 0.75, 0.5, 0.25, 0.0], None, 8, 6)
        self.assertEqual(len(plan), 6)
        self.assertEqual(
            [entry[0] for entry in plan], [1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
        )
        for entry in plan:
            self.assertEqual(len(set(entry)), 1)

    def test_each_entry_is_the_scale_itself_not_a_multiplier(self):
        """denoise_pass2 spells its own 0.8-scaled shape, in full."""
        plan = _scale_schedule([0.8, 0.8, 0.6, 0.4, 0.2, 0.0], None, 8, 6)
        for entry, expected in zip(plan, [0.8, 0.8, 0.6, 0.4, 0.2, 0.0]):
            self.assertAlmostEqual(entry[0], expected)

    def test_the_two_axes_multiply(self):
        taper = [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3]
        plan = _scale_schedule([1.0, 0.5], taper, 8, 2)
        self.assertEqual(plan[0], taper)
        self.assertEqual(plan[1], [value * 0.5 for value in taper])

    def test_a_schedule_of_the_wrong_length_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            _scale_schedule([1.0] * 6, None, 8, 4)
        message = str(caught.exception)
        self.assertIn("strength", message)
        self.assertIn("6", message)
        self.assertIn("4", message)

    def test_an_all_ones_schedule_is_the_unscheduled_scale(self):
        self.assertEqual(_scale_schedule([0.8] * 6, None, 8, 6), [[0.8] * 8] * 6)


class TestVaceScaleHook(unittest.TestCase):
    def _step(self, scales=None, timesteps=(1000.0, 937.0, 857.0)):
        import types

        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        step = Wan22VaceDenoiseStep()
        step._scales = scales
        step._pipe = types.SimpleNamespace(scheduler=_FakeScheduler(timesteps))
        return step

    def _hook(self, scales, expert="low"):
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        step = self._step(scales)
        return step, _vace_scale_hook(step, expert)

    def _call(self, hook, scale, timestep=None, args=()):
        kwargs = {"control_hidden_states_scale": scale}
        if timestep is not None:
            kwargs["timestep"] = timestep
        return hook(None, args, kwargs)

    def test_without_a_schedule_the_call_is_left_alone(self):
        """A constant `strength` must be indistinguishable from no hook."""
        _, hook = self._hook(None)
        self.assertIsNone(self._call(hook, _FakeScale([1.0] * 8)))

    def test_the_schedule_replaces_the_scale_the_pipeline_built(self):
        _, hook = self._hook([[0.5] * 8, [0.25] * 8, [0.0] * 8])
        args, kwargs = self._call(
            hook,
            _FakeScale([1.0] * 8),
            timestep=_FakeTimestep(1000.0),
            args=("positional",),
        )
        self.assertEqual(args, ("positional",))
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [0.5] * 8)

    def test_the_replacement_keeps_the_devices_dtype_and_placement(self):
        """new_tensor, not a fresh tensor: the alternative lands on the CPU."""
        _, hook = self._hook([[0.5] * 8, [0.25] * 8, [0.0] * 8])
        scale = _FakeScale([1.0] * 8, device="cuda:0", dtype="bfloat16")
        _, kwargs = self._call(hook, scale, timestep=_FakeTimestep(1000.0))
        replacement = kwargs["control_hidden_states_scale"]
        self.assertEqual(replacement.device, "cuda:0")
        self.assertEqual(replacement.dtype, "bfloat16")

    def test_the_step_is_read_per_call_not_closed_over(self):
        """One resident pipeline serves both passes, which disagree here."""
        step, hook = self._hook([[0.5] * 8, [0.25] * 8, [0.0] * 8])
        step._scales = [[0.1] * 8, [0.2] * 8, [0.3] * 8]
        _, kwargs = self._call(hook, _FakeScale([1.0] * 8), _FakeTimestep(1000.0))
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [0.1] * 8)

    def test_both_experts_read_the_one_plan(self):
        """A schedule spans the run; the experts merely split it in two."""
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        step = self._step([[1.0] * 8, [0.5] * 8, [0.0] * 8])
        high = _vace_scale_hook(step, "high")
        low = _vace_scale_hook(step, "low")
        _, kwargs = self._call(high, _FakeScale([9.0] * 8), _FakeTimestep(1000.0))
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [1.0] * 8)
        _, kwargs = self._call(low, _FakeScale([9.0] * 8), _FakeTimestep(857.0))
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [0.0] * 8)

    def test_a_schedule_picks_the_entry_the_timestep_names(self):
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        plan = [[1.0] * 8, [0.5] * 8, [0.0] * 8]
        step = self._step(plan, timesteps=(1000.0, 937.0, 857.0))
        hook = _vace_scale_hook(step, "low")
        for timestep, expected in ((1000.0, 1.0), (937.0, 0.5), (857.0, 0.0)):
            _, kwargs = self._call(
                hook, _FakeScale([9.0] * 8), timestep=_FakeTimestep(timestep)
            )
            self.assertEqual(
                kwargs["control_hidden_states_scale"].values, [expected] * 8
            )

    def test_the_step_is_not_counted_so_guidance_calls_twice_agree(self):
        """cond and uncond share a timestep and must share a scale."""
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        step = self._step([[1.0] * 8, [0.5] * 8, [0.0] * 8])
        hook = _vace_scale_hook(step, "low")
        first = self._call(hook, _FakeScale([9.0] * 8), timestep=_FakeTimestep(937.0))
        second = self._call(hook, _FakeScale([9.0] * 8), timestep=_FakeTimestep(937.0))
        self.assertEqual(
            first[1]["control_hidden_states_scale"].values,
            second[1]["control_hidden_states_scale"].values,
        )

    def test_a_timestep_cast_still_finds_its_step(self):
        """Nearest, not equality: a dtype round trip must not break lookup."""
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        step = self._step([[1.0] * 8, [0.5] * 8, [0.0] * 8])
        hook = _vace_scale_hook(step, "low")
        _, kwargs = self._call(
            hook, _FakeScale([9.0] * 8), timestep=_FakeTimestep(936.9375)
        )
        self.assertEqual(kwargs["control_hidden_states_scale"].values, [0.5] * 8)

    def test_a_scheduler_stepping_a_different_number_of_times_is_an_error(self):
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        step = self._step([[1.0] * 8] * 6, timesteps=(1000.0, 500.0))
        hook = _vace_scale_hook(step, "low")
        with self.assertRaises(RuntimeError) as caught:
            self._call(hook, _FakeScale([9.0] * 8), timestep=_FakeTimestep(1000.0))
        self.assertIn("strength", str(caught.exception))

    def test_a_pipeline_that_stopped_passing_the_timestep_is_an_error(self):
        """Guessing the step would fade the control video out at random."""
        from pipeline.steps.wan22_vace_denoise import _vace_scale_hook

        step = self._step([[1.0] * 8, [0.5] * 8, [0.0] * 8])
        hook = _vace_scale_hook(step, "low")
        with self.assertRaises(RuntimeError) as caught:
            self._call(hook, _FakeScale([9.0] * 8))
        self.assertIn("timestep", str(caught.exception))

    def test_a_pipeline_that_stopped_passing_the_scale_is_an_error(self):
        """Silently denoising at the wrong strength is the bad outcome."""
        _, hook = self._hook([[0.5] * 8, [0.25] * 8, [0.0] * 8])
        with self.assertRaises(RuntimeError) as caught:
            hook(None, (), {})
        self.assertIn("strength", str(caught.exception))


class _RunsTheStep:
    """A `run()` against a stub pipeline, returning `(step, calls)`.

    `calls` is everything the step handed diffusers: `pipe()`'s kwargs and
    whatever it wrote to the config. Shared by the two classes below rather
    than inherited from one to the other, which would re-run the first
    class's tests under the second's name.
    """

    def _run(self, reference=None, **params):
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

            def __init__(self):
                self.scheduler = _FakeScheduler()

            def register_to_config(self, **kwargs):
                # What diffusers' ConfigMixin does, reduced to the one key
                # the step writes: the boundary the denoise loop reads.
                calls.update(kwargs)

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return types.SimpleNamespace(
                    frames=[np.zeros((1, 4, 4, 3), dtype=np.float32)]
                )

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        # scipy's array-API shim asks any `torch` it finds in sys.modules for
        # `torch.Tensor` (array_api_compat's is_torch_array) — the beta
        # schedule's quantile function goes through it — and a stub without
        # one breaks scipy.stats' import for every test after this one.
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
        inputs = {
            "control_video": [frame],
            "control_masks": [np.zeros((4, 4), dtype=np.uint8)],
        }
        if reference is not None:
            inputs["reference_image"] = reference
        with patch.dict(sys.modules, {"torch": torch}):
            step.run(inputs, resolved)
        return step, calls


class TestRunPassesTheScaleThrough(_RunsTheStep, unittest.TestCase):
    """The wiring itself: params -> the kwarg diffusers reads.

    This is the seam that rots quietly. `conditioning_scale` accepting a
    list is a diffusers API detail, and the number of entries it wants comes
    off the loaded model's config rather than a constant here, so a stub
    pipeline carrying the real `vace_layers` is what pins both.
    """

    def test_a_constant_strength_still_reaches_every_layer(self):
        step, calls = self._run(strength=[0.8])
        self.assertEqual(calls["conditioning_scale"], [0.8] * 8)
        self.assertIsNone(step._scales)

    def test_the_declared_default_is_a_plain_full_scale_run(self):
        """Nothing set at all: full strength, every layer, no hook."""
        step, calls = self._run()
        self.assertEqual(calls["conditioning_scale"], [1.0] * 8)
        self.assertIsNone(step._scales)

    def test_the_taper_reaches_the_call(self):
        taper = [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3]
        step, calls = self._run(strength=[0.5], strength_layers=taper)
        self.assertEqual(
            calls["conditioning_scale"], [0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.15]
        )
        self.assertIsNone(step._scales)

    def test_a_schedule_is_planned_for_the_hooks_not_sent_to_the_pipeline(self):
        """diffusers has no per-step param — sending one would be ignored."""
        schedule = [1, 1, 0.75, 0.5, 0.25, 0.0]
        step, calls = self._run(strength=schedule, steps_high=2, steps_low=4)
        self.assertEqual(calls["conditioning_scale"], [1.0] * 8)
        self.assertEqual(list(calls["conditioning_scale"]), [1.0] * 8)
        self.assertEqual(
            [entry[0] for entry in step._scales], [1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
        )

    def test_a_schedule_and_a_taper_compose(self):
        taper = [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.3]
        step, _ = self._run(
            strength=[0.8, 0.8, 0.6, 0.4, 0.2, 0.0], steps_high=2, steps_low=4,
            strength_layers=taper
        )
        self.assertEqual([entry[0] for entry in step._scales],
                         [0.8, 0.8, 0.6, 0.4, 0.2, 0.0])
        self.assertAlmostEqual(step._scales[0][-1], 0.24)
        self.assertAlmostEqual(step._scales[3][-1], 0.12)

    def test_the_schedule_length_is_checked_against_this_runs_steps(self):
        with self.assertRaises(ValueError):
            self._run(strength=[1.0] * 6, steps_high=2, steps_low=2)

    def test_a_new_pass_drops_the_previous_ones_timesteps(self):
        """A resident worker's second pass must not index the first's."""
        step, _ = self._run(strength=[1.0] * 6, steps_high=2, steps_low=4)
        self.assertIsNone(step._timesteps)

    def test_the_layer_count_comes_off_the_loaded_model(self):
        with self.assertRaises(ValueError):
            self._run(strength=[1.0], strength_layers=[1.0] * 4)


class TestExpertSplit(unittest.TestCase):
    """`steps_high` / `steps_low`: which expert takes which denoise step.

    diffusers has no step counts. It has one threshold — a step runs on the
    high-noise expert when its timestep is `>= boundary_ratio *
    num_train_timesteps` — so the counts a run gets are a consequence of
    where the scheduler's timesteps happen to fall. These pin the arithmetic
    that turns that round: ask for a split, compute the threshold that
    delivers it. `_select` below is diffusers' own comparison, and the
    tests are read through it rather than against a raw ratio, because the
    ratio is an implementation detail and the selection is the behaviour.
    """

    def _select(self, ratio, timesteps, num_train=1000):
        """Which expert each step gets, the way pipeline_wan_vace.py picks."""
        boundary = ratio * num_train
        return ["high" if t >= boundary else "low" for t in timesteps]

    def _timesteps(self, n):
        scheduler = _FakeScheduler()
        scheduler.set_timesteps(n)
        return scheduler.timesteps

    def _ratio(self, n, steps_high):
        from pipeline.steps.wan22_vace_denoise import _expert_boundary_ratio

        return _expert_boundary_ratio(self._timesteps(n), steps_high, 1000)

    def test_the_default_split_is_what_the_checkpoints_own_ratio_chose(self):
        """The whole feature is a control over something already happening.

        2 high / 4 low is what `steps: 6` gave at boundary_ratio 0.875, so a
        run that asks for the declared default has to denoise identically —
        same expert on the same steps, not merely the same counts.
        """
        timesteps = self._timesteps(6)
        self.assertEqual(
            self._select(self._ratio(6, 2), timesteps),
            self._select(0.875, timesteps),
        )
        self.assertEqual(
            self._select(self._ratio(6, 2), timesteps),
            ["high", "high", "low", "low", "low", "low"],
        )

    def test_a_different_split_is_now_sayable(self):
        self.assertEqual(
            self._select(self._ratio(6, 3), self._timesteps(6)),
            ["high", "high", "high", "low", "low", "low"],
        )

    def test_the_boundary_sits_midway_between_the_two_steps_it_separates(self):
        """Furthest from either neighbour, so no float cast in diffusers can
        move a step across it."""
        timesteps = self._timesteps(6)
        boundary = self._ratio(6, 2) * 1000
        self.assertAlmostEqual(boundary, (timesteps[1] + timesteps[2]) / 2)

    def test_no_low_steps_gives_the_high_expert_the_whole_run(self):
        self.assertEqual(
            self._select(self._ratio(4, 4), self._timesteps(4)), ["high"] * 4
        )

    def test_no_high_steps_gives_the_low_expert_the_whole_run(self):
        self.assertEqual(
            self._select(self._ratio(4, 0), self._timesteps(4)), ["low"] * 4
        )

    def test_timesteps_that_leave_no_room_are_refused(self):
        from pipeline.steps.wan22_vace_denoise import _expert_boundary_ratio

        with self.assertRaises(ValueError):
            _expert_boundary_ratio([1000.0, 1000.0, 500.0], 1, 1000)

    def test_the_two_counts_are_the_runs_step_count(self):
        from pipeline.steps.wan22_vace_denoise import _total_steps

        self.assertEqual(_total_steps({"steps_high": 2, "steps_low": 4}), 6)
        self.assertEqual(_total_steps({"steps_high": 0, "steps_low": 3}), 3)

    def test_a_run_of_no_steps_at_all_is_refused(self):
        from pipeline.steps.wan22_vace_denoise import _total_steps

        with self.assertRaises(ValueError):
            _total_steps({"steps_high": 0, "steps_low": 0})

    def test_a_negative_share_is_refused(self):
        """`minimum` on a Param draws a widget; it does not check anything,
        so -1 would otherwise shorten the run instead of failing it."""
        from pipeline.steps.wan22_vace_denoise import _total_steps

        with self.assertRaises(ValueError):
            _total_steps({"steps_high": -1, "steps_low": 4})


class TestSplitReachesThePipeline(_RunsTheStep, unittest.TestCase):
    """The same wiring harness, asking what `pipe()` and the config got.

    Shares the stub pipeline rather than rebuilding it: what is being
    checked is that the two counts reach diffusers as one
    `num_inference_steps` and one `boundary_ratio`, which is the same seam
    the scale tests drive.
    """

    def test_the_sum_is_what_the_pipeline_is_asked_for(self):
        _, calls = self._run(steps_high=1, steps_low=5)
        self.assertEqual(calls["num_inference_steps"], 6)

    def test_the_boundary_reaches_the_config_before_the_call(self):
        """On the old sampler, so the expectation is the stub's own linspace."""
        _, calls = self._run(
            steps_high=3, steps_low=3, sigma_schedule="linspace", sampler_shift=3.0
        )
        timesteps = _FakeScheduler()
        timesteps.set_timesteps(6)
        self.assertAlmostEqual(
            calls["boundary_ratio"] * 1000,
            (timesteps.timesteps[2] + timesteps.timesteps[3]) / 2,
        )

    def test_the_boundary_is_placed_against_the_sampler_the_run_uses(self):
        """The default sampler is ComfyUI's, and its timesteps are not the
        HF config's — so the 2/4 split lands between 988 and 955, not
        between 937 and 857."""
        _, calls = self._run(steps_high=2, steps_low=4)
        self.assertAlmostEqual(calls["boundary_ratio"] * 1000, (987.641 + 954.733) / 2, places=1)

    def test_the_run_logs_which_steps_each_expert_took(self):
        """`steps: 6` never said, and a `strength` schedule is written
        against the split — the taper is spent on the low-noise steps."""
        with self.assertLogs("pipeline.steps.wan22_vace_denoise", level="INFO") as caught:
            self._run(steps_high=2, steps_low=4)
        line = next(m for m in caught.output if "experts" in m)
        self.assertIn("2 high-noise", line)
        self.assertIn("4 low-noise", line)
        self.assertIn("1000", line)

    def test_a_run_of_no_steps_is_refused_before_the_pipeline_is_called(self):
        with self.assertRaises(ValueError):
            self._run(steps_high=0, steps_low=0)


class TestSamplerSchedule(unittest.TestCase):
    """`sigma_schedule` / `sampler_shift`: the step spacing ComfyUI ran.

    The reference graph sampled uni_pc on ComfyUI's `beta` scheduler at the
    Wan default shift of 8; the HF scheduler config is linspace at 3.0, and
    the port ran that for two weeks believing it matched. These pin the
    arithmetic against ComfyUI's own (comfy/samplers.py `beta_scheduler`,
    evaluated with scipy at n=6: indices 999, 908, 724, 500, 275, 91) and
    the wiring that gets it into diffusers' `set_timesteps`.
    """

    # comfy/samplers.py at n=6: rint(beta.ppf(levels, 0.6, 0.6) * 999).
    COMFY_INDEX = [999, 908, 724, 500, 275, 91]

    def test_the_beta_schedule_is_comfyuis_index_rounding(self):
        from pipeline.steps.wan22_vace_denoise import _comfy_beta_sigmas

        sigmas = _comfy_beta_sigmas(6)
        self.assertEqual(len(sigmas), 6)
        for sigma, index in zip(sigmas, self.COMFY_INDEX):
            # ComfyUI reads sigmas[index] and the Wan table is shift((i+1)/1000).
            self.assertAlmostEqual(sigma, (index + 1) / 1000)

    def test_the_schedule_descends_from_one(self):
        from pipeline.steps.wan22_vace_denoise import _comfy_beta_sigmas

        for n in (1, 2, 4, 6, 8):
            sigmas = _comfy_beta_sigmas(n)
            self.assertEqual(sigmas[0], 1.0)
            self.assertEqual(sigmas, sorted(sigmas, reverse=True))

    def test_a_schedule_that_repeats_a_timestep_is_refused_not_shortened(self):
        """ComfyUI drops the duplicate and runs fewer steps; here `strength`
        and the split are planned per step, so the run is refused."""
        from pipeline.steps.wan22_vace_denoise import _comfy_beta_sigmas

        with self.assertRaises(ValueError):
            _comfy_beta_sigmas(6, num_train_timesteps=4)

    def test_no_steps_at_all_is_refused(self):
        from pipeline.steps.wan22_vace_denoise import _comfy_beta_sigmas

        with self.assertRaises(ValueError):
            _comfy_beta_sigmas(0)

    def test_the_simple_schedule_is_comfyuis_strided_table_walk(self):
        """comfy/samplers.py's `simple_scheduler`: sigmas[-(1 + int(x * ss))]
        with ss = len(sigmas) / steps, which at 6 strides 166 indices."""
        from pipeline.steps.wan22_vace_denoise import _comfy_simple_sigmas

        sigmas = _comfy_simple_sigmas(6)
        index = [999 - int(x * (1000 / 6)) for x in range(6)]
        self.assertEqual(index, [999, 833, 666, 499, 333, 166])
        for sigma, i in zip(sigmas, index):
            self.assertAlmostEqual(sigma, (i + 1) / 1000)

    def test_simple_is_linspace_to_within_the_stride_truncation(self):
        """The reason to reach for `linspace` instead: the two schedules
        agree exactly where the step count divides the table, and by under
        a timestep in 1000 where it does not."""
        from pipeline.steps.wan22_vace_denoise import _comfy_simple_sigmas

        for n in (4, 8, 20):  # divides 1000
            linspace = [1.0 - (1.0 - 1.0 / n) * i / (n - 1) for i in range(n)]
            for a, b in zip(_comfy_simple_sigmas(n), linspace):
                self.assertAlmostEqual(a, b)
        for n in (6, 12):  # does not
            linspace = [1.0 - (1.0 - 1.0 / n) * i / (n - 1) for i in range(n)]
            gaps = [abs(a - b) for a, b in zip(_comfy_simple_sigmas(n), linspace)]
            self.assertLess(max(gaps), 1 / 1000)
            self.assertGreater(max(gaps), 0.0)

    def test_the_simple_schedule_descends_from_one(self):
        from pipeline.steps.wan22_vace_denoise import _comfy_simple_sigmas

        for n in (1, 2, 4, 6, 8):
            sigmas = _comfy_simple_sigmas(n)
            self.assertEqual(len(sigmas), n)
            self.assertEqual(sigmas[0], 1.0)
            self.assertEqual(sigmas, sorted(sigmas, reverse=True))

    def test_a_simple_schedule_that_repeats_a_timestep_is_refused(self):
        from pipeline.steps.wan22_vace_denoise import _comfy_simple_sigmas

        with self.assertRaises(ValueError):
            _comfy_simple_sigmas(6, num_train_timesteps=4)

    def test_no_simple_steps_at_all_is_refused(self):
        from pipeline.steps.wan22_vace_denoise import _comfy_simple_sigmas

        with self.assertRaises(ValueError):
            _comfy_simple_sigmas(0)

    def test_shifted_by_eight_it_is_the_reference_graphs_timesteps(self):
        """The two numbers together, through the stub's flow-sigma branch:
        what the reference graph's KSamplers actually stepped through."""
        from pipeline.steps.wan22_vace_denoise import _comfy_beta_sigmas

        scheduler = _FakeScheduler(shift=8.0)
        scheduler.set_timesteps(6, sigmas=_comfy_beta_sigmas(6))
        self.assertEqual(
            [round(t) for t in scheduler.timesteps], [1000, 988, 955, 889, 753, 448]
        )


class TestSamplerReachesTheScheduler(_RunsTheStep, unittest.TestCase):
    """The wiring: the shift lands in the config, the schedule in the call.

    `pipe()` calls `set_timesteps` with a step count and nothing else, so
    the schedule can only arrive through the wrapper the step installs on
    that method — the same shape of seam as the scale hooks, and the same
    way it could rot: a wrapper that stopped firing would run the HF
    schedule while logging the ComfyUI one.
    """

    def test_the_default_run_samples_the_way_the_reference_graph_did(self):
        step, _ = self._run()
        scheduler = step._pipe.scheduler
        self.assertEqual(scheduler.shift, 8.0)
        self.assertEqual(
            [round(t) for t in scheduler.timesteps], [1000, 988, 955, 889, 753, 448]
        )

    def test_linspace_at_three_is_the_run_before_the_change(self):
        step, _ = self._run(sigma_schedule="linspace", sampler_shift=3.0)
        scheduler = step._pipe.scheduler
        self.assertEqual(scheduler.shift, 3.0)
        self.assertEqual(
            [round(t) for t in scheduler.timesteps], [1000, 938, 857, 750, 600, 375]
        )

    def test_simple_spreads_the_steps_the_beta_run_crowded(self):
        """`simple` through the same wrapper: an even walk, so the run keeps
        steps at the low-noise end instead of spending four above t=750."""
        step, _ = self._run(sigma_schedule="simple")
        scheduler = step._pipe.scheduler
        self.assertEqual(scheduler.shift, 8.0)
        self.assertEqual(
            [round(t, 1) for t in scheduler.timesteps],
            [1000.0, 975.7, 941.3, 888.9, 800.5, 616.0],
        )

    def test_the_schedule_is_read_per_call_so_two_passes_can_differ(self):
        """One resident pipeline, one wrapper, two passes — the second
        pass's schedule must win without reinstalling anything."""
        step, _ = self._run()
        scheduler = step._pipe.scheduler
        first = scheduler.set_timesteps
        step._sigma_schedule = "linspace"
        scheduler.set_timesteps(6)
        self.assertIs(scheduler.set_timesteps, first)
        self.assertEqual(round(scheduler.timesteps[1]), 976)  # linspace at shift 8

    def test_the_wrapper_goes_on_once_per_scheduler(self):
        """A second pass configures the same scheduler again and must not
        stack a second wrapper on the first."""
        step, _ = self._run()
        scheduler = step._pipe.scheduler
        installed = scheduler.set_timesteps
        step._configure_sampler(
            step._pipe, step.resolve_params({"width": 16, "height": 16, "seed": 0})
        )
        self.assertIs(scheduler.set_timesteps, installed)
        self.assertIs(step._scheduled, scheduler)

    def test_a_scheduler_off_flow_sigmas_is_refused(self):
        """Neither knob reaches such a scheduler, so the run must not
        pretend they did."""
        import types

        step, _ = self._run()
        step._pipe.scheduler.config = types.SimpleNamespace(
            num_train_timesteps=1000, use_flow_sigmas=False
        )
        with self.assertRaises(RuntimeError):
            step._configure_sampler(step._pipe, step.resolve_params(
                {"width": 16, "height": 16, "seed": 0}
            ))

    def test_the_run_logs_the_sampler(self):
        with self.assertLogs("pipeline.steps.wan22_vace_denoise", level="INFO") as caught:
            self._run()
        line = next(m for m in caught.output if "sampler" in m)
        self.assertIn("beta", line)
        self.assertIn("8.0", line)


class TestSolverAndHandoff(_RunsTheStep, unittest.TestCase):
    """`solver_variant` / `solver_order` / `handoff_reset`: the rest of the
    reference graph's sampler.

    ComfyUI's uni_pc is bh1 at order 3, and its second KSamplerAdvanced is
    a NEW sampler — the low-noise expert starts with an empty multistep
    history. diffusers runs one continuous loop, so the history has to be
    emptied by hand at the hand-off, from a pre-hook on the low-noise
    expert, the same seam the scale hooks use.
    """

    def _hook(self, step):
        from pipeline.steps.wan22_vace_denoise import _handoff_hook

        return _handoff_hook(step)

    def _dirty(self, scheduler):
        scheduler.model_outputs = ["a", "b"]
        scheduler.timestep_list = [1000.0, 988.0]
        scheduler.lower_order_nums = 2
        scheduler.last_sample = "x"
        scheduler._step_index = 2

    def test_each_sampler_runs_at_comfyuis_order_for_its_step_count(self):
        """`min(3, len(timesteps) - 2)`, with a sampler seeing its own steps
        plus the boundary sigma: 2 steps -> 1, 4 -> 3, 3 -> 2, 1 -> 1."""
        from pipeline.steps.wan22_vace_denoise import _phase_order

        self.assertEqual(_phase_order(3, 2), 1)
        self.assertEqual(_phase_order(3, 4), 3)
        self.assertEqual(_phase_order(3, 3), 2)
        self.assertEqual(_phase_order(3, 1), 1)
        self.assertEqual(_phase_order(2, 4), 2)

    def test_the_solver_reaches_the_scheduler_config(self):
        """bh1, opening at the HIGH-noise sampler's order: 1 for 2 steps."""
        step, _ = self._run()
        config = step._pipe.scheduler.config
        self.assertEqual(config.solver_type, "bh1")
        self.assertEqual(config.solver_order, 1)

    def test_the_hand_off_gives_the_low_noise_sampler_its_own_order(self):
        step, _ = self._run()
        self._hook(step)(None, (), {})
        self.assertEqual(step._pipe.scheduler.config.solver_order, 3)
        self.assertEqual(len(step._pipe.scheduler.timestep_list), 3)

    def test_a_three_three_split_opens_at_order_two(self):
        step, _ = self._run(steps_high=3, steps_low=3)
        self.assertEqual(step._pipe.scheduler.config.solver_order, 2)
        self._hook(step)(None, (), {})
        self.assertEqual(step._pipe.scheduler.config.solver_order, 2)

    def test_the_high_noise_samplers_last_step_is_first_order(self):
        """lower_order_final, counted within the first sampler: at a 3/3
        split step 3 drops to order 1; steps 1 and 2 are left alone."""
        from pipeline.steps.wan22_vace_denoise import _phase_end_hook

        step, _ = self._run(steps_high=3, steps_low=3)
        scheduler = step._pipe.scheduler
        hook = _phase_end_hook(step)
        hook(None, (), {"timestep": _FakeTimestep(scheduler.timesteps[1])})
        self.assertEqual(scheduler.config.solver_order, 2)
        hook(None, (), {"timestep": _FakeTimestep(scheduler.timesteps[2])})
        self.assertEqual(scheduler.config.solver_order, 1)
        # Once the low-noise expert has taken over, the hook is inert.
        self._hook(step)(None, (), {})
        hook(None, (), {"timestep": _FakeTimestep(scheduler.timesteps[2])})
        self.assertEqual(scheduler.config.solver_order, 2)

    def test_off_is_one_continuous_loop_at_the_cap(self):
        step, _ = self._run(handoff_reset=False)
        self.assertEqual(step._pipe.scheduler.config.solver_order, 3)
        self.assertFalse(step._handoff_pending)

    def test_the_old_solver_is_still_sayable(self):
        step, _ = self._run(solver_variant="bh2", solver_order=2, handoff_reset=False)
        config = step._pipe.scheduler.config
        self.assertEqual((config.solver_type, config.solver_order), ("bh2", 2))

    def test_no_hand_off_when_one_expert_takes_the_whole_run(self):
        step, _ = self._run(steps_high=0, steps_low=6)
        self.assertFalse(step._handoff_pending)
        self.assertEqual(step._pipe.scheduler.config.solver_order, 3)

    def test_the_restart_empties_exactly_what_a_new_sampler_starts_without(self):
        from pipeline.steps.wan22_vace_denoise import _restart_multistep

        scheduler = _FakeScheduler()
        self._dirty(scheduler)
        _restart_multistep(scheduler)
        self.assertEqual(scheduler.model_outputs, [None, None])
        self.assertEqual(scheduler.timestep_list, [None, None])
        self.assertEqual(scheduler.lower_order_nums, 0)
        self.assertIsNone(scheduler.last_sample)
        # Still on the same step of the same schedule.
        self.assertEqual(scheduler._step_index, 2)

    def test_the_history_is_sized_to_the_order_the_run_sets(self):
        """diffusers' set_timesteps resizes `model_outputs` to the config's
        order but leaves `timestep_list` as built; at order 3 over a 2-slot
        list the real scheduler's step() raises IndexError. The stub is
        built with 2 slots, as the real one is from the HF config."""
        step, _ = self._run(handoff_reset=False)
        scheduler = step._pipe.scheduler
        self.assertEqual(len(scheduler.timestep_list), 3)
        self.assertEqual(len(scheduler.model_outputs), 3)

    def test_the_restart_sizes_the_history_from_the_config(self):
        from pipeline.steps.wan22_vace_denoise import _restart_multistep

        scheduler = _FakeScheduler()
        scheduler.config.solver_order = 3
        _restart_multistep(scheduler)
        self.assertEqual(scheduler.timestep_list, [None, None, None])

    def test_the_restart_never_invents_an_attribute(self):
        from pipeline.steps.wan22_vace_denoise import _restart_multistep

        bare = types.SimpleNamespace()
        _restart_multistep(bare)
        self.assertEqual(vars(bare), {})

    def test_the_hand_off_fires_once_per_pass(self):
        """Two forwards per step under guidance, four low-noise steps: one
        restart, at the first."""
        step, _ = self._run()
        scheduler = step._pipe.scheduler
        hook = self._hook(step)
        self.assertTrue(step._handoff_pending)
        self._dirty(scheduler)
        hook(None, (), {})
        self.assertEqual(scheduler.lower_order_nums, 0)
        self.assertFalse(step._handoff_pending)
        self._dirty(scheduler)
        hook(None, (), {})
        self.assertEqual(scheduler.lower_order_nums, 2)  # left alone

    def test_the_next_pass_arms_it_again(self):
        step, _ = self._run()
        self._hook(step)(None, (), {})
        self.assertFalse(step._handoff_pending)
        step, _ = self._run()
        self.assertTrue(step._handoff_pending)

    def test_off_means_the_history_is_carried_across(self):
        step, _ = self._run(handoff_reset=False)
        scheduler = step._pipe.scheduler
        self.assertFalse(step._handoff_pending)
        self._dirty(scheduler)
        self._hook(step)(None, (), {})
        self.assertEqual(scheduler.lower_order_nums, 2)

    def test_bh1_lands_its_last_step_where_comfyui_does(self):
        """diffusers' bh1 is non-finite at a terminal sigma of 0 (measured);
        ComfyUI's sample_unipc puts 0.001 there. bh2 keeps diffusers' 0."""
        step, _ = self._run()
        self.assertEqual(step._pipe.scheduler.sigmas[-1], 0.001)
        step, _ = self._run(solver_variant="bh2")
        self.assertEqual(step._pipe.scheduler.sigmas[-1], 0.0)

    def test_the_run_logs_the_solver_and_the_hand_off(self):
        with self.assertLogs("pipeline.steps.wan22_vace_denoise", level="INFO") as caught:
            self._run()
        line = next(m for m in caught.output if "sampler" in m)
        self.assertIn("bh1", line)
        self.assertIn("order 3", line)
        self.assertIn("restarts", line)


class TestSamplerPerExpert(_RunsTheStep, unittest.TestCase):
    """`sampler_high` / `sampler_low`: which sampler takes each phase's steps.

    The scheduler the pipeline was loaded with is UniPC and stays UniPC —
    it owns the sigmas, the timesteps and the expert boundary — so `euler`
    is a substitution inside `step()`, installed by the same kind of
    instance wrapper the schedule uses. These pin the split (by step index,
    the same one the experts split on), the arithmetic, and the one place
    the choice changes something else: bh1's terminal sigma, which exists
    for UniPC's last step and not for Euler's.
    """

    def _drive(self, step, sample=1.0, velocity=1.0):
        """Take the run's steps the way the denoise loop does, and hand back
        the sample after each."""
        scheduler = step._pipe.scheduler
        samples = []
        for timestep in list(scheduler.timesteps):
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
            samples.append(sample)
        return samples

    def test_euler_is_the_flow_matching_step_on_the_schedulers_own_sigmas(self):
        from pipeline.steps.wan22_vace_denoise import _euler_step

        scheduler = _FakeScheduler(shift=8.0)
        scheduler.set_timesteps(6)
        sigmas = list(scheduler.sigmas)
        sample = _euler_step(scheduler, 2.0, scheduler.timesteps[0], 5.0, False)[0]
        self.assertAlmostEqual(sample, 5.0 + (sigmas[1] - sigmas[0]) * 2.0)
        self.assertEqual(scheduler._step_index, 1)

    def test_the_default_run_takes_every_step_on_unipc(self):
        step, _ = self._run()
        self._drive(step)
        self.assertEqual(
            step._pipe.scheduler.stepped, [("uni_pc", i) for i in range(6)]
        )

    def test_euler_on_the_high_expert_leaves_the_low_ones_on_unipc(self):
        """The split is `steps_high`, the same index the experts split on."""
        step, _ = self._run(sampler_high="euler")
        self._drive(step)
        self.assertEqual(
            step._pipe.scheduler.stepped, [("uni_pc", 2), ("uni_pc", 3),
                                           ("uni_pc", 4), ("uni_pc", 5)]
        )

    def test_euler_on_the_low_expert_leaves_the_high_ones_on_unipc(self):
        step, _ = self._run(sampler_low="euler")
        self._drive(step)
        self.assertEqual(
            step._pipe.scheduler.stepped, [("uni_pc", 0), ("uni_pc", 1)]
        )

    def test_euler_throughout_never_reaches_the_scheduler_step(self):
        step, _ = self._run(sampler_high="euler", sampler_low="euler")
        samples = self._drive(step, sample=0.0, velocity=1.0)
        self.assertEqual(step._pipe.scheduler.stepped, [])
        # Six first-order steps down the sigma ladder, ending at sigma 0:
        # x0 = x + sum(d sigma) * v, which from 0 is -sigmas[0] = -1.
        self.assertAlmostEqual(samples[-1], -1.0)

    def test_bh1s_terminal_sigma_is_left_off_when_euler_takes_the_last_step(self):
        """COMFY_TERMINAL_SIGMA is there because diffusers' bh1 update is
        non-finite at sigma 0. An Euler step is not, and 0.001 would leave
        the last of the noise in."""
        step, _ = self._run(sampler_low="euler")
        self.assertEqual(step._pipe.scheduler.sigmas[-1], 0.0)
        step, _ = self._run()
        self.assertEqual(step._pipe.scheduler.sigmas[-1], 0.001)

    def test_a_high_only_run_puts_its_last_step_on_the_high_sampler(self):
        """With no low-noise steps the high sampler takes the run, so it is
        the one the terminal sigma answers to."""
        step, _ = self._run(steps_high=6, steps_low=0, strength=[1.0],
                            sampler_high="euler")
        self.assertEqual(step._pipe.scheduler.sigmas[-1], 0.0)

    def test_the_wrapper_goes_on_once_per_scheduler(self):
        step, _ = self._run()
        scheduler = step._pipe.scheduler
        installed = scheduler.step
        step._configure_sampler(
            step._pipe, step.resolve_params({"width": 16, "height": 16, "seed": 0})
        )
        self.assertIs(scheduler.step, installed)
        self.assertIs(step._sampled, scheduler)

    def test_the_sampler_is_read_per_call_so_two_passes_can_differ(self):
        """One resident pipeline, one wrapper: the second pass's choice must
        win without reinstalling anything."""
        step, _ = self._run()
        scheduler = step._pipe.scheduler
        step._sampler_high = step._sampler_low = "euler"
        self._drive(step)
        self.assertEqual(scheduler.stepped, [])

    def test_the_run_logs_which_sampler_took_which_phase(self):
        with self.assertLogs("pipeline.steps.wan22_vace_denoise", level="INFO") as caught:
            self._run(sampler_high="euler")
        line = next(m for m in caught.output if "sampler:" in m)
        self.assertIn("Euler high", line)
        self.assertIn("UniPC bh1", line)


class TestReferenceFit(_RunsTheStep, unittest.TestCase):
    """`reference_fit`: ComfyUI's centre-crop-and-fill versus diffusers'
    letterbox-on-white.

    The arithmetic is comfy/utils.py's `common_upscale(..., "center")`:
    crop to the target aspect about the centre (Python `round`), then
    resize to fill. The numbers below are that formula evaluated for the
    two reference shapes this pipeline has actually seen — the 768x1536
    back panel of a split sheet, and cyber_6f's 1440x1280 whole sheet.
    """

    def _indexed(self, width, height, axis):
        """An image whose first channel carries its own row (or column)
        index / 8, so a crop's placement can be read back off the pixels."""
        import numpy as np

        image = np.zeros((height, width, 3), dtype=np.uint8)
        index = np.arange(height)[:, None] if axis == "row" else np.arange(width)[None, :]
        image[..., 0] = np.broadcast_to(index // 8, (height, width))
        return image

    def test_a_portrait_panel_loses_85_rows_top_and_bottom(self):
        from pipeline.steps.wan22_vace_denoise import _fit_reference

        out = _fit_reference(self._indexed(768, 1536, "row"), 720, 1280)
        self.assertEqual(out.shape, (1280, 720, 3))
        # round((1536 - 1536 * (0.5 / 0.5625)) / 2) = 85: rows 85..1450 survive.
        self.assertAlmostEqual(int(out[0, 360, 0]), 85 // 8, delta=1)
        self.assertAlmostEqual(int(out[-1, 360, 0]), 1450 // 8, delta=1)

    def test_a_landscape_sheet_loses_360_columns_each_side(self):
        from pipeline.steps.wan22_vace_denoise import _fit_reference

        out = _fit_reference(self._indexed(1440, 1280, "col"), 720, 1280)
        self.assertEqual(out.shape, (1280, 720, 3))
        # round((1440 - 1440 * (0.5625 / 1.125)) / 2) = 360: cols 360..1079.
        self.assertAlmostEqual(int(out[640, 0, 0]), 360 // 8, delta=1)
        self.assertAlmostEqual(int(out[640, -1, 0]), 1079 // 8, delta=1)

    def test_the_frames_own_aspect_is_a_plain_resize(self):
        import cv2

        from pipeline.steps.wan22_vace_denoise import _fit_reference

        image = self._indexed(360, 640, "row")
        out = _fit_reference(image, 720, 1280)
        expected = cv2.resize(image, (720, 1280), interpolation=cv2.INTER_LINEAR)
        self.assertTrue((out == expected).all())

    def test_crop_hands_diffusers_a_reference_that_already_fits(self):
        import numpy as np

        _, calls = self._run(reference=np.zeros((24, 8, 3), dtype=np.uint8))
        self.assertEqual(calls["reference_images"][0].size, (16, 16))

    def test_letterbox_hands_it_over_untouched(self):
        import numpy as np

        _, calls = self._run(
            reference=np.zeros((24, 8, 3), dtype=np.uint8), reference_fit="letterbox"
        )
        self.assertEqual(calls["reference_images"][0].size, (8, 24))


class TestDeclaration(unittest.TestCase):
    def test_neither_knob_forces_the_pipeline_to_be_rebuilt(self):
        """All per-call. In LOAD_PARAMS they would cost a 47 GB reload."""
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        for name in ("strength", "strength_layers", "sigma_schedule", "sampler_shift",
                     "sampler_high", "sampler_low", "solver_variant", "solver_order",
                     "handoff_reset", "reference_fit"):
            self.assertNotIn(name, Wan22VaceDenoiseStep.LOAD_PARAMS)

    def test_the_declared_sampler_is_the_whole_reference_graph(self):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name: param for param in Wan22VaceDenoiseStep.PARAMS}
        self.assertEqual(declared["sampler_high"].default, "uni_pc")
        self.assertEqual(declared["sampler_low"].default, "uni_pc")
        self.assertEqual(declared["solver_variant"].default, "bh1")
        self.assertEqual(declared["solver_order"].default, 3)
        self.assertTrue(declared["handoff_reset"].default)
        self.assertEqual(declared["reference_fit"].default, "crop")

    def test_the_declared_sampler_is_the_reference_graphs(self):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name: param for param in Wan22VaceDenoiseStep.PARAMS}
        self.assertEqual(declared["sigma_schedule"].default, "beta")
        self.assertEqual(declared["sampler_shift"].default, 8.0)

    def test_the_declared_defaults_are_a_plain_full_scale_run(self):
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name: param for param in Wan22VaceDenoiseStep.PARAMS}
        self.assertEqual(declared["strength"].default, [1.0])
        self.assertIsNone(declared["strength_layers"].default)

    def test_strength_is_the_only_scale_the_step_declares(self):
        """The unification, pinned: `strength_low`/`strength_steps` are gone.

        Not pedantry — a workflow still spelling either one gets a refusal
        from `resolve_params` naming what the step does accept, which is the
        error a stale YAML should produce rather than a silently ignored
        knob.
        """
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name for param in Wan22VaceDenoiseStep.PARAMS}
        scales = {
            name
            for name in declared
            if name.startswith("strength") and not name.startswith("lora_")
        }
        self.assertEqual(scales, {"strength", "strength_layers"})

    def test_strength_draws_as_a_yaml_list_box(self):
        """A min AND a max would make webui draw a slider (see _control).

        A slider cannot hold a schedule at all, and this param's whole point
        is that one number and six are the same knob.
        """
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        declared = {param.name: param for param in Wan22VaceDenoiseStep.PARAMS}
        strength = declared["strength"]
        self.assertIs(strength.type, list)
        self.assertIsNone(strength.minimum)
        self.assertIsNone(strength.maximum)


if __name__ == "__main__":
    unittest.main()
