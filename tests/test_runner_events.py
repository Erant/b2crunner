"""The runner's progress callback and its failure reporting.

These run a real WorkflowRunner — which, as docs/docker-build-notes.md
records, nothing in the suite did before: both bugs the first full GPU run
turned up were in code no test ever executed. The steps here are trivial
fakes, so the test needs no GPU, but the runner, context, dispatcher
factory and templating are all the real ones.
"""

from __future__ import annotations

import unittest

from pipeline.registry import register_step
from pipeline.runner import RunCancelled, RunEvent, WorkflowRunner
from pipeline.step import Step
from pipeline.workflow import WorkflowSpec


@register_step("_test_echo")
class EchoStep(Step):
    def run(self, inputs, params):
        return {"value": params.get("value", inputs.get("value"))}


@register_step("_test_boom")
class BoomStep(Step):
    def run(self, inputs, params):
        raise ValueError("this step always fails")


_RESIDENT_LOG = []


@register_step("_test_resident")
class ResidentStep(Step):
    def load(self, params):
        _RESIDENT_LOG.append("load")

    def unload(self):
        _RESIDENT_LOG.append("unload")

    def run(self, inputs, params):
        _RESIDENT_LOG.append("run")
        return {"value": params.get("value")}


@register_step("_test_log")
class LogStep(Step):
    def run(self, inputs, params):
        _RESIDENT_LOG.append("other")
        return {}


def _make_spec(step_dicts, globals_=None):
    from pipeline.workflow import StepSpec

    return WorkflowSpec(
        name="t", globals=globals_ or {},
        steps=[StepSpec.from_dict(d) for d in step_dicts],
    )


class TestRunnerEvents(unittest.TestCase):
    def test_events_are_emitted_in_order(self):
        spec = _make_spec([
            {"id": "one", "step": "_test_echo",
             "params": {"value": 1}, "outputs": {"value": "out.one"}},
            {"id": "two", "step": "_test_echo",
             "params": {"value": 2}, "outputs": {"value": "out.two"}},
        ])
        seen = []
        ctx = WorkflowRunner(spec, on_event=seen.append).run({"dataset": None})

        self.assertEqual(
            [e.kind for e in seen],
            ["workflow_start", "step_start", "step_end", "step_start", "step_end", "workflow_end"],
        )
        self.assertEqual([e.index for e in seen if e.kind == "step_start"], [1, 2])
        self.assertTrue(all(e.total == 2 for e in seen))
        self.assertEqual(ctx.get("out.one"), 1)
        self.assertEqual(ctx.get("out.two"), 2)

    def test_resident_dispatcher_closes_after_its_last_step(self):
        # helical's Wan worker stayed in host RAM until the run ended and the
        # upscale was OOM-killed beside it; it must go once nothing uses it.
        spec = _make_spec([
            {"id": "a", "step": "_test_resident", "keep_loaded": True,
             "params": {"value": 1}, "outputs": {"value": "out.a"}},
            {"id": "b", "step": "_test_resident", "keep_loaded": True,
             "params": {"value": 2}, "outputs": {"value": "out.b"}},
            {"id": "c", "step": "_test_log"},
        ])
        _RESIDENT_LOG.clear()
        WorkflowRunner(spec).run({"dataset": None})
        self.assertEqual(_RESIDENT_LOG, ["load", "run", "run", "unload", "other"])

    def test_step_error_event_then_reraise(self):
        spec = _make_spec([{"id": "kaboom", "step": "_test_boom"}])
        seen = []
        with self.assertRaises(ValueError):
            WorkflowRunner(spec, on_event=seen.append).run({"dataset": None})

        errors = [e for e in seen if e.kind == "step_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].step_id, "kaboom")
        self.assertIn("always fails", errors[0].error)

    def test_a_broken_observer_does_not_break_the_run(self):
        spec = _make_spec([
            {"id": "one", "step": "_test_echo",
             "params": {"value": 1}, "outputs": {"value": "out.one"}},
        ])

        def observer(event: RunEvent) -> None:
            raise RuntimeError("the UI exploded")

        ctx = WorkflowRunner(spec, on_event=observer).run({"dataset": None})
        self.assertEqual(ctx.get("out.one"), 1)

    def test_cancellation_propagates(self):
        """RunCancelled is the one exception an observer may raise to stop."""
        spec = _make_spec([
            {"id": "one", "step": "_test_echo",
             "params": {"value": 1}, "outputs": {"value": "out.one"}},
            {"id": "two", "step": "_test_echo",
             "params": {"value": 2}, "outputs": {"value": "out.two"}},
        ])

        def observer(event: RunEvent) -> None:
            if event.kind == "step_start" and event.index == 2:
                raise RunCancelled("stop")

        with self.assertRaises(RunCancelled):
            WorkflowRunner(spec, on_event=observer).run({"dataset": None})

    def test_missing_input_names_the_step_and_the_path(self):
        spec = _make_spec([
            {"id": "reader", "step": "_test_echo", "inputs": {"value": "nowhere.at.all"}},
        ])
        with self.assertRaises(KeyError) as caught:
            WorkflowRunner(spec).run({"dataset": None})
        message = str(caught.exception)
        self.assertIn("reader", message)
        self.assertIn("nowhere.at.all", message)

    def test_unknown_env_is_reported_before_dispatching(self):
        spec = _make_spec([
            {"id": "remote", "step": "_test_echo", "dispatch": "subprocess", "env": "ghost"},
        ])
        with self.assertRaises(ValueError) as caught:
            WorkflowRunner(spec, envs={"wan22": {}}).run({"dataset": None})
        self.assertIn("ghost", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
