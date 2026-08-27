"""The param declaration a Step carries, and the merge built on top of it.

Before this existed, a step's defaults lived inline as
`params.get("filter_size", 6)` at each call site: invisible to the UI,
unenumerable, and impossible to check a workflow against. `Step.PARAMS`
moved them into one declaration and `Step.resolve_params` is what turns a
workflow's overrides plus that declaration into the dict `run()` reads.

The properties worth pinning are the ones a mistake in either would make
silent: a typo'd override that does nothing, a REQUIRED param nobody
supplied, and "false" from a text box arriving as True.
"""

from __future__ import annotations

import unittest

from pipeline.registry import STEP_REGISTRY, get_step_class
from pipeline.step import REQUIRED, Param, ParamError, Step

import pipeline.steps  # noqa: F401  populates the registry


class Sample(Step):
    STEP_NAME = "_sample"
    PARAMS = (
        Param("count", int, 6, "how many"),
        Param("scale", float, 0.5),
        Param("enabled", bool, True),
        Param("mode", str, "fast", choices=("fast", "slow")),
        Param("size", list, [720, 1280]),
        Param("device", str, None, "computed at runtime when empty"),
        Param("destination", str, REQUIRED, "nowhere sensible to default to"),
    )

    def run(self, inputs, params):
        return dict(params)


class Undeclared(Step):
    STEP_NAME = "_undeclared"

    def run(self, inputs, params):
        return dict(params)


class TestResolveParams(unittest.TestCase):
    def test_defaults_fill_in_everything_not_overridden(self):
        params = Sample.resolve_params({"destination": "/tmp/x"})
        self.assertEqual(params["count"], 6)
        self.assertEqual(params["scale"], 0.5)
        self.assertEqual(params["mode"], "fast")
        self.assertEqual(params["size"], [720, 1280])
        self.assertEqual(sorted(params), sorted(Sample.declared_params()))

    def test_an_override_wins(self):
        params = Sample.resolve_params({"destination": "/tmp/x", "count": 12})
        self.assertEqual(params["count"], 12)

    def test_an_undeclared_name_is_refused_by_name(self):
        """The failure this prevents is a run that completes at the wrong
        settings and looks like it honoured you."""
        with self.assertRaises(ParamError) as ctx:
            Sample.resolve_params({"destination": "/tmp/x", "cont": 12})
        message = str(ctx.exception)
        self.assertIn("cont", message)
        self.assertIn("_sample", message)
        self.assertIn("count", message, "the message should list what IS accepted")

    def test_a_required_param_nobody_supplied_raises(self):
        with self.assertRaises(ParamError) as ctx:
            Sample.resolve_params({})
        self.assertIn("destination", str(ctx.exception))

    def test_none_is_a_default_and_is_not_required(self):
        """`None` means 'the step computes this at runtime' — rmbg's device,
        seedvr2's model_dir — and is a different thing from REQUIRED."""
        params = Sample.resolve_params({"destination": "/tmp/x"})
        self.assertIsNone(params["device"])

    def test_it_is_idempotent(self):
        """pipeline.worker re-merges what WorkflowRunner already merged, so
        a second pass over a complete dict has to be a no-op rather than a
        complaint about params it now contains."""
        once = Sample.resolve_params({"destination": "/tmp/x", "count": 12})
        self.assertEqual(Sample.resolve_params(once), once)

    def test_an_undeclared_step_passes_its_overrides_through(self):
        self.assertEqual(
            Undeclared.resolve_params({"anything": 1}), {"anything": 1}
        )


class TestCoercion(unittest.TestCase):
    """A value can arrive as a string from `--param` or a UI text box."""

    def test_the_string_false_is_false(self):
        for text in ("false", "False", "no", "off", "0", "", "  false  "):
            with self.subTest(value=text):
                params = Sample.resolve_params({"destination": "x", "enabled": text})
                self.assertIs(params["enabled"], False)

    def test_other_strings_are_true(self):
        for text in ("true", "yes", "1", "anything"):
            with self.subTest(value=text):
                params = Sample.resolve_params({"destination": "x", "enabled": text})
                self.assertIs(params["enabled"], True)

    def test_numbers_come_back_typed(self):
        params = Sample.resolve_params({"destination": "x", "count": "12", "scale": "0.25"})
        self.assertIsInstance(params["count"], int)
        self.assertEqual(params["count"], 12)
        self.assertIsInstance(params["scale"], float)
        self.assertEqual(params["scale"], 0.25)

    def test_a_fractional_int_is_reported_not_truncated(self):
        with self.assertRaises(ParamError) as ctx:
            Sample.resolve_params({"destination": "x", "count": 6.5})
        self.assertIn("count", str(ctx.exception))

    def test_a_non_number_names_the_param_and_the_type(self):
        with self.assertRaises(ParamError) as ctx:
            Sample.resolve_params({"destination": "x", "count": "many"})
        message = str(ctx.exception)
        self.assertIn("count", message)
        self.assertIn("int", message)

    def test_a_tuple_is_accepted_for_a_list(self):
        params = Sample.resolve_params({"destination": "x", "size": (512, 512)})
        self.assertEqual(params["size"], [512, 512])


class TestEveryRegisteredStep(unittest.TestCase):
    """Properties that have to hold across all of pipeline/steps/."""

    def test_every_step_declares_its_params(self):
        """A step with no declaration is invisible to the UI and unchecked
        by `WorkflowSpec.validate`. The three that legitimately take none
        are named here so adding a fourth is a deliberate act."""
        takes_none = {"fit_cameras_to_images", "generate_firstlast"}
        for name, step_class in sorted(STEP_REGISTRY.items()):
            if name.startswith("_test"):
                continue
            with self.subTest(step=name):
                if name in takes_none:
                    self.assertEqual(step_class.PARAMS, ())
                else:
                    self.assertTrue(
                        step_class.PARAMS, f"'{name}' declares no params"
                    )

    def test_register_step_names_the_class(self):
        """Error messages say 'step mask_splat', not 'step MaskSplatStep'."""
        self.assertEqual(get_step_class("mask_splat").STEP_NAME, "mask_splat")

    def test_defaults_survive_their_own_coercion(self):
        """A default that its own declared type rejects is a typo that would
        only surface for whoever left the param alone."""
        for name, step_class in sorted(STEP_REGISTRY.items()):
            for param in step_class.PARAMS:
                if param.default is REQUIRED or param.default is None:
                    continue
                with self.subTest(step=name, param=param.name):
                    single = type(step_class.__name__, (Step,), {
                        "PARAMS": (param,),
                        "run": lambda self, i, p: p,
                    })
                    self.assertEqual(
                        single.resolve_params({})[param.name], param.default
                    )

    def test_a_choice_default_is_one_of_the_choices(self):
        for name, step_class in sorted(STEP_REGISTRY.items()):
            for param in step_class.PARAMS:
                if not param.choices or param.default is REQUIRED:
                    continue
                with self.subTest(step=name, param=param.name):
                    self.assertIn(param.default, param.choices)


if __name__ == "__main__":
    unittest.main()
