"""InProcessDispatcher's GPU-memory release between one-shot steps.

Found by actually running a sequence of GPU steps in one process on a
VRAM-constrained box (12GB): rmbg -> sapiens2_lite OOM'd because nothing
ever called torch.cuda.empty_cache() between them. keep_loaded=False (the
default) creates a fresh Step instance per call and never tracks it, so
Dispatcher.close()'s unload() loop — the only place that already called
empty_cache() — never runs for it either. See pipeline/dispatch/in_process.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.dispatch.in_process import InProcessDispatcher
from pipeline.registry import register_step
from pipeline.step import Step


@register_step("_test_stub_step")
class _StubStep(Step):
    load_calls = 0
    unload_calls = 0
    release_vram_calls = 0

    def load(self, params):
        _StubStep.load_calls += 1

    def unload(self):
        _StubStep.unload_calls += 1

    def release_vram(self):
        _StubStep.release_vram_calls += 1

    def run(self, inputs, params):
        return {"echo": inputs.get("x")}


class TestInProcessDispatcherMemoryRelease(unittest.TestCase):
    def setUp(self):
        _StubStep.load_calls = 0
        _StubStep.unload_calls = 0
        _StubStep.release_vram_calls = 0

    @patch("pipeline.dispatch.in_process._release_cuda_cache")
    def test_releases_cache_after_a_one_shot_step(self, mock_release):
        dispatcher = InProcessDispatcher(keep_loaded=False)
        result = dispatcher.run("_test_stub_step", {"x": 1}, {})
        self.assertEqual(result, {"echo": 1})
        mock_release.assert_called_once()

    @patch("pipeline.dispatch.in_process._release_cuda_cache")
    def test_keep_loaded_frees_vram_but_keeps_the_instance(self, mock_release):
        """keep_loaded holds weights in DRAM between calls, not on the card.

        This assertion used to be the opposite — that keep_loaded must NOT
        touch the allocator between calls — and under the semantics of the
        time it was correct: with the model still on the GPU,
        empty_cache() frees only unused cached blocks, so calling it was
        pure churn for no reclaimed VRAM.

        Step.release_vram() is what changed that. Paired with it the call
        is meaningful, because the weights have actually moved to host RAM
        by the time it runs, so the driver gets the VRAM back. Same
        contract the resident subprocess worker gives, and for the same
        reason: fast_helical_native runs brush on the GPU between two
        invocations of a kept-loaded step.

        What must NOT change is the instance: it is still reused, which is
        what saves the checkpoint re-read that keep_loaded exists for.
        """
        dispatcher = InProcessDispatcher(keep_loaded=True)
        dispatcher.run("_test_stub_step", {"x": 1}, {})
        dispatcher.run("_test_stub_step", {"x": 2}, {})
        self.assertEqual(mock_release.call_count, 2)
        self.assertEqual(_StubStep.release_vram_calls, 2)
        self.assertEqual(_StubStep.load_calls, 1)  # instance reused: the point
        dispatcher.close()
        self.assertEqual(_StubStep.unload_calls, 1)

    def test_one_shot_steps_get_a_fresh_instance_each_call(self):
        dispatcher = InProcessDispatcher(keep_loaded=False)
        dispatcher.run("_test_stub_step", {"x": 1}, {})
        dispatcher.run("_test_stub_step", {"x": 2}, {})
        # load() is only ever called explicitly for keep_loaded instances;
        # one-shot steps load lazily inside their own run() (see rmbg.py),
        # so this dispatcher never calls it.
        self.assertEqual(_StubStep.load_calls, 0)


if __name__ == "__main__":
    unittest.main()
