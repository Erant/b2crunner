"""`when:` — the switch that lets a workflow declare optional outputs.

Runs a real WorkflowRunner over trivial fake steps (the ones
tests/test_runner_events.py registers), because the thing worth testing is
not the truthiness helper but the runner's behaviour around it: that a
skipped step's outputs never reach the Context, that it still occupies its
slot in the event stream so a UI's step table lines up with the YAML, and
that an unresolvable `when:` stops the run at the start rather than at the
step.

That last one is the reason `enabled` is computed up front in
WorkflowRunner.run: a typo in `when: ${globals.export_ply}` guarding the
final brush training would otherwise surface an hour into a run, having
already spent two denoise passes and a splat training.

The `?` suffix on an input path is the other half of the same feature, and
is tested here for that reason: a gated step's outputs are absent when it
is switched off, so a step downstream of one has no way to say "take these
if they were built" without it. That is what lets all three shipped
workflows share one tail while only two of them can build the face splat's
supporting views.
"""

from __future__ import annotations

import unittest

from pipeline.runner import WorkflowRunner
from pipeline.workflow import StepSpec, WorkflowSpec, step_enabled

import tests.test_runner_events  # noqa: F401  registers _test_echo


def _make_spec(step_dicts, globals_=None):
    return WorkflowSpec(
        name="t", globals=globals_ or {},
        steps=[StepSpec.from_dict(d) for d in step_dicts],
    )


class TestStepEnabled(unittest.TestCase):
    def test_default_is_on(self):
        step = StepSpec.from_dict({"id": "a", "step": "_test_echo"})
        self.assertTrue(step_enabled(step, {}))

    def test_literal_false_is_off(self):
        step = StepSpec.from_dict({"id": "a", "step": "_test_echo", "when": False})
        self.assertFalse(step_enabled(step, {}))

    def test_param_reference_follows_the_param(self):
        step = StepSpec.from_dict(
            {"id": "a", "step": "_test_echo", "when": "${globals.export_ply}"}
        )
        self.assertTrue(step_enabled(step, {"export_ply": True}))
        self.assertFalse(step_enabled(step, {"export_ply": False}))

    def test_the_string_false_is_false(self):
        """A `when:` almost always resolves through a param someone typed,
        and bool("false") is True — which would run exactly the step they
        switched off."""
        step = StepSpec.from_dict(
            {"id": "a", "step": "_test_echo", "when": "${globals.x}"}
        )
        for value in ("false", "False", "no", "off", "0", "", "  false  "):
            with self.subTest(value=value):
                self.assertFalse(step_enabled(step, {"x": value}))
        for value in ("true", "yes", "1", "anything"):
            with self.subTest(value=value):
                self.assertTrue(step_enabled(step, {"x": value}))

    def test_an_unresolvable_when_names_the_step(self):
        step = StepSpec.from_dict(
            {"id": "final_splat", "step": "_test_echo", "when": "${globals.typo}"}
        )
        with self.assertRaises(KeyError) as caught:
            step_enabled(step, {"export_ply": True})
        self.assertIn("final_splat", str(caught.exception))


class TestRunnerSkips(unittest.TestCase):
    def _spec(self, second_on: bool):
        return _make_spec(
            [
                {"id": "one", "step": "_test_echo",
                 "params": {"value": 1}, "outputs": {"value": "out.one"}},
                {"id": "two", "step": "_test_echo", "when": "${globals.want_two}",
                 "params": {"value": 2}, "outputs": {"value": "out.two"}},
                {"id": "three", "step": "_test_echo",
                 "params": {"value": 3}, "outputs": {"value": "out.three"}},
            ],
            globals_={"want_two": second_on},
        )

    def test_a_skipped_step_writes_nothing(self):
        ctx = WorkflowRunner(self._spec(False)).run({"dataset": None})
        self.assertEqual(ctx.get("out.one"), 1)
        self.assertEqual(ctx.get("out.three"), 3)
        with self.assertRaises(KeyError):
            ctx.get("out.two")

    def test_it_keeps_its_slot_in_the_event_stream(self):
        seen = []
        WorkflowRunner(self._spec(False), on_event=seen.append).run({"dataset": None})

        self.assertEqual(
            [e.kind for e in seen],
            ["workflow_start", "step_start", "step_end", "step_skipped",
             "step_start", "step_end", "workflow_end"],
        )
        skipped = next(e for e in seen if e.kind == "step_skipped")
        self.assertEqual(skipped.index, 2)
        self.assertEqual(skipped.step_id, "two")
        # Indices stay 1..3 against a total of 3 — nothing renumbers around
        # the hole, so a step table built from the YAML still lines up.
        self.assertEqual([e.index for e in seen if e.index], [1, 1, 2, 3, 3])
        self.assertTrue(all(e.total == 3 for e in seen))

    def test_switching_it_on_runs_it(self):
        ctx = WorkflowRunner(self._spec(True)).run({"dataset": None})
        self.assertEqual(ctx.get("out.two"), 2)

    def test_a_bad_when_fails_before_any_step_runs(self):
        spec = _make_spec(
            [
                {"id": "expensive", "step": "_test_echo",
                 "params": {"value": 1}, "outputs": {"value": "out.one"}},
                {"id": "guarded", "step": "_test_echo", "when": "${globals.nope}"},
            ],
            globals_={},
        )
        seen = []
        with self.assertRaises(KeyError):
            WorkflowRunner(spec, on_event=seen.append).run({"dataset": None})
        self.assertEqual(seen, [])

    def test_enabled_steps_reports_what_will_run(self):
        spec = self._spec(False)
        self.assertEqual([s.id for s in spec.enabled_steps()], ["one", "three"])
        spec.globals["want_two"] = True
        self.assertEqual([s.id for s in spec.enabled_steps()], ["one", "two", "three"])


if __name__ == "__main__":
    unittest.main()


class TestOptionalInputs(unittest.TestCase):
    """`path?` — absent resolves to None instead of failing the run."""

    def _run(self, steps, globals_=None):
        spec = _make_spec(steps, globals_)
        return WorkflowRunner(spec).run({"dataset": None})

    def test_a_required_input_still_fails(self):
        """The default, and what makes a typo'd path a failure at that step
        rather than an input that silently arrives as None."""
        with self.assertRaises(KeyError) as caught:
            self._run([{"id": "a", "step": "_test_echo",
                        "inputs": {"value": "nothing.here"},
                        "outputs": {"value": "out.a"}}])
        self.assertIn("nothing.here", str(caught.exception))

    def test_an_optional_input_resolves_to_none(self):
        ctx = self._run([{"id": "a", "step": "_test_echo",
                          "inputs": {"value": "nothing.here?"},
                          "outputs": {"value": "out.a"}}])
        self.assertIsNone(ctx.get("out.a"))

    def test_an_optional_input_that_is_there_is_read(self):
        """The marker says 'may be absent', not 'ignore it'."""
        ctx = self._run([
            {"id": "a", "step": "_test_echo", "params": {"value": 7},
             "outputs": {"value": "out.a"}},
            {"id": "b", "step": "_test_echo", "inputs": {"value": "out.a?"},
             "outputs": {"value": "out.b"}},
        ])
        self.assertEqual(ctx.get("out.b"), 7)

    def test_a_missing_required_output_still_fails(self):
        """The default, and what keeps a typo'd output name a failure."""
        with self.assertRaises(KeyError) as caught:
            self._run([{"id": "a", "step": "_test_echo", "params": {"value": 1},
                        "outputs": {"vlaue": "out.a"}}])
        self.assertIn("vlaue", str(caught.exception))

    def test_a_gated_producer_feeds_an_ungated_consumer(self):
        """The shape this exists for: the face branch writes the supporting
        views when it runs, and the training downstream takes them or
        trains without them."""
        steps = [
            {"id": "branch", "step": "_test_echo", "params": {"value": 3},
             "when": "${globals.face_splat}", "outputs": {"value": "scene.support"}},
            {"id": "train", "step": "_test_echo",
             "inputs": {"value": "scene.support?"}, "outputs": {"value": "out.train"}},
        ]
        self.assertEqual(
            self._run(steps, {"face_splat": True}).get("out.train"), 3
        )
        self.assertIsNone(
            self._run(steps, {"face_splat": False}).get("out.train")
        )
