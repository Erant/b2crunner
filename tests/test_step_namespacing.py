"""Two calls of the same step, configured apart — the reason for the split.

A workflow used to have one flat `params:` block, so anything a step needed
was hoisted to the top of the file and hand-prefixed to keep two calls of
that step from colliding (`brush_total_steps`, `mask_filter_size`,
`upscale_resolution`). fast_helical_full calls `brush` twice and
`wan22_vace_denoise` twice, and under that shape the two trainings could not
be given different settings at all.

Now a step's params live under its `id:`. These tests run a real
WorkflowRunner over trivial fakes and check that each call sees its own
values, that the step's declared defaults fill in the rest, and that the
merge happens before dispatch rather than inside any one step.
"""

from __future__ import annotations

import unittest

from pipeline.registry import register_step
from pipeline.runner import WorkflowRunner
from pipeline.step import REQUIRED, Param, ParamError, Step
from pipeline.workflow import StepSpec, WorkflowSpec

import tests.test_runner_events  # noqa: F401  registers _test_echo


@register_step("_test_recorder")
class RecorderStep(Step):
    """Returns the params it was handed, so a test can see the merge."""

    PARAMS = (
        Param("label", str, REQUIRED),
        Param("count", int, 100),
        Param("mode", str, "fast", choices=("fast", "slow")),
    )

    def run(self, inputs, params):
        return {"seen": dict(params)}


def _spec(step_dicts, globals_=None):
    return WorkflowSpec(
        name="t", globals=globals_ or {},
        steps=[StepSpec.from_dict(d) for d in step_dicts],
    )


class TestTwoCallsOfOneStep(unittest.TestCase):
    def _run(self):
        spec = _spec([
            {"id": "first", "step": "_test_recorder",
             "params": {"label": "one", "count": 5},
             "outputs": {"seen": "out.first"}},
            {"id": "second", "step": "_test_recorder",
             "params": {"label": "two", "mode": "slow"},
             "outputs": {"seen": "out.second"}},
        ])
        return WorkflowRunner(spec).run({"dataset": None})

    def test_each_call_sees_its_own_overrides(self):
        ctx = self._run()
        self.assertEqual(ctx.get("out.first")["label"], "one")
        self.assertEqual(ctx.get("out.second")["label"], "two")
        self.assertEqual(ctx.get("out.first")["count"], 5)
        self.assertEqual(ctx.get("out.second")["mode"], "slow")

    def test_one_call_s_override_does_not_leak_into_the_other(self):
        """The failure the old flat namespace made unavoidable."""
        ctx = self._run()
        self.assertEqual(ctx.get("out.second")["count"], 100, "count leaked from 'first'")
        self.assertEqual(ctx.get("out.first")["mode"], "fast", "mode leaked from 'second'")

    def test_a_shared_value_still_comes_from_a_global(self):
        """The other half: what genuinely must agree across calls is a
        global, referenced by template from both."""
        spec = _spec(
            [
                {"id": "first", "step": "_test_recorder",
                 "params": {"label": "one", "count": "${globals.shared}"},
                 "outputs": {"seen": "out.first"}},
                {"id": "second", "step": "_test_recorder",
                 "params": {"label": "two", "count": "${globals.shared}"},
                 "outputs": {"seen": "out.second"}},
            ],
            globals_={"shared": 42},
        )
        ctx = WorkflowRunner(spec).run({"dataset": None})
        self.assertEqual(ctx.get("out.first")["count"], 42)
        self.assertEqual(ctx.get("out.second")["count"], 42)


class TestTheMergeHappensInTheRunner(unittest.TestCase):
    def test_a_step_is_handed_every_declared_param(self):
        """`run()` reads params["x"] with no fallback, so a param the
        workflow never mentions still has to arrive."""
        spec = _spec([
            {"id": "only", "step": "_test_recorder",
             "params": {"label": "x"}, "outputs": {"seen": "out.seen"}},
        ])
        seen = WorkflowRunner(spec).run({"dataset": None}).get("out.seen")
        self.assertEqual(sorted(seen), ["count", "label", "mode"])

    def test_a_template_is_expanded_before_the_merge(self):
        """Order matters: coercing `${globals.n}` as an int would fail."""
        spec = _spec(
            [{"id": "only", "step": "_test_recorder",
              "params": {"label": "x", "count": "${globals.n}"},
              "outputs": {"seen": "out.seen"}}],
            globals_={"n": "7"},
        )
        seen = WorkflowRunner(spec).run({"dataset": None}).get("out.seen")
        self.assertEqual(seen["count"], 7)
        self.assertIsInstance(seen["count"], int)


class TestValidation(unittest.TestCase):
    def test_an_undeclared_override_stops_the_run_before_step_one(self):
        """The expensive failure this replaces: a typo noticed forty minutes
        in, when the step it belongs to is finally reached."""
        spec = _spec([
            {"id": "cheap", "step": "_test_echo",
             "params": {"value": 1}, "outputs": {"value": "out.one"}},
            {"id": "typo", "step": "_test_recorder", "params": {"label": "x", "coont": 1}},
        ])
        seen = []
        with self.assertRaises(ValueError) as ctx:
            WorkflowRunner(spec, on_event=seen.append).run({"dataset": None})
        self.assertIn("coont", str(ctx.exception))
        self.assertEqual(seen, [], "the cheap step ran before the typo was noticed")

    def test_validate_names_the_step_id_not_just_the_step(self):
        """With two calls of one step, the class name alone doesn't say
        which block to go and fix."""
        spec = _spec([
            {"id": "first", "step": "_test_recorder", "params": {"label": "x"}},
            {"id": "second", "step": "_test_recorder", "params": {"label": "y", "nope": 1}},
        ])
        with self.assertRaises(ValueError) as ctx:
            spec.validate()
        self.assertIn("second", str(ctx.exception))

    def test_a_missing_required_param_names_the_step(self):
        spec = _spec([{"id": "only", "step": "_test_recorder"}])
        with self.assertRaises(ParamError) as ctx:
            WorkflowRunner(spec).run({"dataset": None})
        self.assertIn("label", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
