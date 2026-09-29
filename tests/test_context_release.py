"""The runner releases context entries once no later step reads them.

On 2026-09-29 two helical runs on the 29 GB local box were OOM-killed the
moment the final training launched: the run worker was 23.8 GB resident,
because every intermediate any step had written (re-outline frames,
extension phases, support views, normal maps, ...) stayed in the context
until the run ended. `WorkflowRunner.run(..., keep=...)` now drops each
step output after its last reader. These tests run a real runner over
trivial fakes and check what is still there at each step.
"""

from __future__ import annotations

import unittest

from pipeline.context import Context
from pipeline.registry import register_step
from pipeline.runner import WorkflowRunner
from pipeline.step import Param, Step
from pipeline.workflow import StepSpec, WorkflowSpec


@register_step("_test_emit")
class EmitStep(Step):
    """Returns `value` as `out`."""

    PARAMS = (Param("value", str, "x"),)

    def run(self, inputs, params):
        return {"out": params["value"]}


@register_step("_test_peek")
class PeekStep(Step):
    """Returns its inputs, so a test sees what a step was handed."""

    PARAMS = ()

    def run(self, inputs, params):
        return {"out": dict(inputs)}


def _spec(step_dicts, globals_=None):
    return WorkflowSpec(
        name="t", globals=globals_ or {},
        steps=[StepSpec.from_dict(d) for d in step_dicts],
    )


def _emit(step_id, path, value="x", when=None):
    step = {"id": step_id, "step": "_test_emit", "params": {"value": value},
            "outputs": {"out": path}}
    if when is not None:
        step["when"] = when
    return step


def _peek(step_id, inputs, path, when=None):
    step = {"id": step_id, "step": "_test_peek", "inputs": inputs,
            "outputs": {"out": path}}
    if when is not None:
        step["when"] = when
    return step


class _Dataset:
    def __init__(self):
        self.splat_path = None


class TestRelease(unittest.TestCase):
    def _run(self, steps, keep=("dataset",), globals_=None):
        seen = []

        def snapshot(event):
            if event.kind == "step_end":
                seen.append((event.step_id, sorted(_leaves(event.context))))

        ctx = WorkflowRunner(_spec(steps, globals_), on_event=snapshot).run(
            {"dataset": _Dataset()}, keep=keep)
        return ctx, dict(seen)

    def test_an_output_goes_after_its_last_reader(self):
        ctx, seen = self._run([
            _emit("a", "scene.big"),
            _peek("b", {"x": "scene.big"}, "scene.b"),
            _peek("c", {"x": "scene.big"}, "scene.c"),
            _emit("d", "scene.d"),
        ])
        self.assertIn("scene.big", seen["b"])
        # c's own step_end fires before the release, so it still sees it.
        self.assertIn("scene.big", seen["c"])
        self.assertNotIn("scene.big", seen["d"])
        self.assertEqual(ctx.get("scene"), {})

    def test_a_whole_namespace_read_keeps_everything_under_it(self):
        """helical's `mesh_output: scene` reads every scene.* entry."""
        ctx, seen = self._run([
            _emit("a", "scene.one"),
            _emit("b", "scene.two"),
            _peek("c", {"mesh_output": "scene"}, "out.c"),
            _emit("d", "out.d"),
        ])
        self.assertIn("scene.one", seen["b"])
        self.assertNotIn("scene.one", seen["d"])

    def test_a_read_of_a_part_keeps_the_whole_output(self):
        ctx, seen = self._run([
            _emit("a", "scene.record"),
            _emit("b", "scene.other"),
            _peek("c", {"x": "scene.record.part?"}, "out.c"),
        ])
        self.assertIn("scene.record", seen["b"])

    def test_an_optional_read_counts(self):
        ctx, seen = self._run([
            _emit("a", "scene.maybe"),
            _emit("b", "scene.other"),
            _peek("c", {"x": "scene.maybe?"}, "out.c"),
        ])
        self.assertIn("scene.maybe", seen["b"])

    def test_a_skipped_reader_does_not_hold_an_entry(self):
        ctx, seen = self._run(
            [
                _emit("a", "scene.big"),
                _emit("b", "scene.other"),
                _peek("c", {"x": "scene.big"}, "out.c", when="${globals.on}"),
            ],
            globals_={"on": False},
        )
        self.assertNotIn("scene.big", seen["b"])

    def test_a_rewritten_path_goes_after_its_last_reader(self):
        ctx, seen = self._run([
            _emit("a", "scene.v", "1"),
            _peek("b", {"x": "scene.v"}, "out.b"),
            _emit("c", "scene.v", "2"),
            _peek("d", {"x": "scene.v"}, "out.d"),
            _emit("e", "out.e"),
        ])
        self.assertIn("scene.v", seen["c"])
        self.assertNotIn("scene.v", seen["e"])

    def test_a_reader_gets_the_value_written_last(self):
        steps = [
            _emit("a", "scene.v", "1"),
            _emit("b", "scene.v", "2"),
            _peek("c", {"x": "scene.v"}, "result.c"),
        ]
        ctx = WorkflowRunner(_spec(steps)).run({"dataset": _Dataset()}, keep=("result",))
        self.assertEqual(ctx.get("result.c"), {"x": "2"})

    def test_keep_protects_the_caller_s_paths(self):
        ctx, _ = self._run([
            _emit("a", "dataset.splat_path", "scene.ply"),
            _emit("b", "scene.unread"),
        ])
        self.assertEqual(ctx.get("dataset").splat_path, "scene.ply")
        self.assertEqual(ctx.get("scene"), {})

    def test_no_keep_releases_nothing(self):
        ctx, _ = self._run([_emit("a", "scene.unread")], keep=None)
        self.assertEqual(ctx.get("scene.unread"), "x")


class TestContextDelete(unittest.TestCase):
    def test_deletes_a_nested_entry_and_leaves_siblings(self):
        ctx = Context({})
        ctx.set("scene.a.b", 1)
        ctx.set("scene.a.c", 2)
        self.assertTrue(ctx.delete("scene.a.b"))
        self.assertEqual(ctx.get("scene.a"), {"c": 2})

    def test_a_missing_path_is_not_an_error(self):
        ctx = Context({})
        self.assertFalse(ctx.delete("scene.nothing"))
        ctx.set("scene.x", 1)
        self.assertFalse(ctx.delete("scene.x.y"))

    def test_an_object_attribute_is_left_alone(self):
        dataset = _Dataset()
        ctx = Context({"dataset": dataset})
        self.assertFalse(ctx.delete("dataset.splat_path"))
        self.assertTrue(hasattr(dataset, "splat_path"))


def _leaves(ctx, prefix=""):
    """Every dotted path to a non-dict value under the dict namespaces."""
    data = ctx.as_dict() if isinstance(ctx, Context) else ctx
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            yield from _leaves(value, path + ".")
        elif not isinstance(value, _Dataset):
            yield path


if __name__ == "__main__":
    unittest.main()
