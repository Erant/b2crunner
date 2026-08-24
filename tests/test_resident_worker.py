"""`keep_loaded: true` keeps a subprocess step's model resident between calls.

The measurement this exists for: `pipeline/workflows/fast_helical_full.yaml`
invokes `wan22_vace_denoise` twice — stage 1 and stage 4, with brush
training, a splat re-render, an anchor re-inject and a mask in between, so
the two calls cannot be merged. Under the one-process-per-call dispatch that
is ~47 GB of weights read off a pod's network volume twice. `StepSpec.
keep_loaded` already existed and was already passed to `build_dispatcher`,
but factory.py forwarded it only to InProcessDispatcher: on the subprocess
branch it was accepted and silently dropped, which is the worst shape a
config option can have.

These tests drive the real SubprocessPythonDispatcher against real child
processes — no mocked Popen — using the stub steps in tests/resident_stubs.py,
which the worker picks up via B2C_EXTRA_STEP_MODULES. No GPU, no network, no
weights: the child is this same interpreter.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from pipeline.dispatch.factory import build_dispatcher
from pipeline.dispatch.subprocess_python import SubprocessPythonDispatcher
from pipeline.worker import SERVE_MARKER, load_signature

REPO_ROOT = Path(__file__).resolve().parent.parent


class _CaptureLogs:
    """Attach a handler to the root logger at INFO for the duration.

    Records the arrival time of each message, not just its text: the
    difference between "streamed" and "captured and dumped at the end" is
    entirely a matter of when, and asserting only on content would pass
    against a regression back to capture_output.
    """

    def __enter__(self):
        self.records: list[tuple[float, str]] = []
        outer = self

        class _Handler(logging.Handler):
            def emit(self, record):
                outer.records.append((time.monotonic(), record.getMessage()))

        self.handler = _Handler()
        self.root = logging.getLogger()
        self.previous = self.root.level
        self.root.setLevel(logging.INFO)
        self.root.addHandler(self.handler)
        return self

    def __exit__(self, *exc):
        self.root.removeHandler(self.handler)
        self.root.setLevel(self.previous)
        return False

    @property
    def messages(self) -> list[str]:
        return [message for _, message in self.records]

    def first_time(self, needle: str) -> float:
        for when, message in self.records:
            if needle in message:
                return when
        raise AssertionError(f"{needle!r} never reached the log; saw: {self.messages}")


def _dispatcher(keep_loaded: bool, unload_log: Path | None = None) -> SubprocessPythonDispatcher:
    python_path = os.pathsep.join(
        part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part
    )
    env = {
        "B2C_EXTRA_STEP_MODULES": "tests.resident_stubs",
        "PYTHONPATH": python_path,
    }
    if unload_log is not None:
        env["B2C_STUB_UNLOAD_LOG"] = str(unload_log)
    return SubprocessPythonDispatcher(
        python_bin=sys.executable,
        cwd=str(REPO_ROOT),
        env=env,
        keep_loaded=keep_loaded,
    )


class TestResidency(unittest.TestCase):
    def test_keep_loaded_loads_once_across_two_calls(self):
        """The headline claim: two run()s, one load(), one process."""
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
            second = dispatcher.run("_resident_probe", {}, {"model": "wan"})
        finally:
            dispatcher.close()

        self.assertEqual(first["loads"], 1)
        self.assertEqual(second["loads"], 1, "the model was reloaded for the second call")
        self.assertEqual(first["pid"], second["pid"], "a second worker process was spawned")
        self.assertEqual(second["runs"], 2, "the same instance did not serve both jobs")

    def test_the_default_is_still_a_fresh_process_per_call(self):
        """keep_loaded=False must behave exactly as it did before this existed."""
        dispatcher = _dispatcher(keep_loaded=False)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
            second = dispatcher.run("_resident_probe", {}, {"model": "wan"})
        finally:
            dispatcher.close()

        self.assertEqual(first["loads"], 1)
        self.assertEqual(second["loads"], 1)
        self.assertEqual(first["runs"], 1)
        self.assertNotEqual(first["pid"], second["pid"], "the one-shot path reused a process")

    def test_run_params_that_do_not_affect_load_do_not_reload(self):
        """fast_helical_full's actual difference between the passes.

        denoise_pass1 and denoise_pass2 differ only in `strength` (1.0 vs
        0.8), which `Wan22VaceDenoiseStep.load()` never reads. A rule that
        reloaded on any param change would reload here — i.e. would do
        nothing for the one workflow this feature was built for.
        """
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan", "strength": 1.0})
            second = dispatcher.run("_resident_probe", {}, {"model": "wan", "strength": 0.8})
        finally:
            dispatcher.close()

        self.assertEqual(second["loads"], 1)
        self.assertEqual(first["pid"], second["pid"])

    def test_a_changed_load_param_reloads_in_the_same_process(self):
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
            second = dispatcher.run("_resident_probe", {}, {"model": "other"})
        finally:
            dispatcher.close()

        self.assertEqual(first["loads"], 1)
        self.assertEqual(second["loads"], 2, "a changed LOAD_PARAM did not force a reload")
        self.assertEqual(second["model"], "other")
        # Reload, not respawn: the process (and its warm page cache for the
        # weights it is about to re-read) is worth keeping either way.
        self.assertEqual(first["pid"], second["pid"])

    def test_a_step_declaring_no_load_params_is_reused_regardless(self):
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe_undeclared", {}, {"model": "wan"})
            second = dispatcher.run("_resident_probe_undeclared", {}, {"model": "other"})
        finally:
            dispatcher.close()

        self.assertEqual(second["loads"], 1)
        self.assertEqual(first["pid"], second["pid"])

    def test_a_different_step_on_the_same_worker_swaps_the_resident(self):
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
            other = dispatcher.run("_resident_probe_undeclared", {}, {"model": "wan"})
        finally:
            dispatcher.close()

        # Same process, but the first step's instance was unloaded to make
        # room — holding two 47 GB models because a workflow interleaved
        # them is not what keep_loaded promised.
        self.assertEqual(first["pid"], other["pid"])
        self.assertEqual(other["loads"], 1)


class TestVramEviction(unittest.TestCase):
    """Residency is DRAM residency. The card goes back after every job.

    `brush` — a GPU program — runs between fast_helical_full's two denoise
    passes. A resident worker that kept ~35 GB of Wan experts in VRAM
    across that gap would OOM it on any card, which is strictly worse than
    the reload-every-time behaviour keep_loaded replaces. What keep_loaded
    is buying is skipping the ~47 GB network read, and host RAM buys that
    on its own.
    """

    def test_vram_is_released_between_jobs_but_the_weights_are_not(self):
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
            second = dispatcher.run("_resident_probe", {}, {"model": "wan"})
        finally:
            dispatcher.close()

        # Partial eviction happened in between...
        self.assertEqual(first["releases"], 0, "released before the first job even ran")
        self.assertEqual(second["releases"], 1, "the card was not handed back after job 1")
        self.assertIs(second["was_on_gpu"], False, "job 2 started with the model still on GPU")

        # ...and cost nothing that lives in host RAM. Same process (pid),
        # so the id() comparison is meaningful, and it is the same object
        # still holding what load() put on it.
        self.assertEqual(first["pid"], second["pid"])
        self.assertEqual(first["instance"], second["instance"], "the instance was rebuilt")
        self.assertEqual(second["weights"], "weights-for-wan", "the weights were dropped too")
        self.assertEqual(second["loads"], 1)

    def test_the_card_is_free_before_run_returns_not_after(self):
        """The parent's contract: when run() returns, the GPU is yours.

        Releasing after the status line instead would be a race the runner
        cannot see — the next step it starts may want the card immediately.
        """
        dispatcher = _dispatcher(keep_loaded=True)
        with _CaptureLogs() as logs:
            try:
                dispatcher.run("_resident_probe", {}, {"model": "wan"})
            finally:
                dispatcher.close()

        released = logs.first_time("probe release_vram #1")
        finished = logs.first_time(": finished in")
        self.assertLess(
            released, finished,
            "the worker reported the job done before it released the GPU",
        )

    def test_an_explicit_release_keeps_the_worker_and_its_weights(self):
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
            dispatcher.release_vram()
            self.assertIsNotNone(dispatcher._resident, "the explicit release killed the worker")
            second = dispatcher.run("_resident_probe", {}, {"model": "wan"})
        finally:
            dispatcher.close()

        # One automatic release after job 1, plus the explicit one.
        self.assertEqual(second["releases"], 2)
        self.assertEqual(first["instance"], second["instance"])
        self.assertEqual(second["loads"], 1, "the explicit release cost a reload")

    def test_release_vram_is_a_noop_without_a_resident_worker(self):
        _dispatcher(keep_loaded=False).release_vram()  # must not raise
        _dispatcher(keep_loaded=True).release_vram()   # nothing started yet

    def test_close_is_the_full_eviction(self):
        with tempfile.TemporaryDirectory() as tmp:
            unload_log = Path(tmp) / "unloads.txt"
            dispatcher = _dispatcher(keep_loaded=True, unload_log=unload_log)
            with _CaptureLogs() as logs:
                dispatcher.run("_resident_probe", {}, {"model": "wan"})
                self.assertFalse(
                    unload_log.exists(),
                    "a partial eviction called unload(); that drops the host-RAM copy "
                    "and puts the 47 GB network read back",
                )
                dispatcher.close()

            self.assertIn("ResidentProbeStep", unload_log.read_text())
            self.assertIn("probe unload", logs.messages)

    def test_the_default_step_hook_is_a_safe_noop(self):
        """None of the ~22 existing steps override this; none may break."""
        from pipeline.step import Step

        class Plain(Step):
            def run(self, inputs, params):
                return {}

        step = Plain()
        step.release_vram()
        step.release_vram()  # idempotent: it is called after every job

    def test_emptying_the_cuda_cache_is_safe_with_or_without_torch(self):
        from pipeline.worker import _empty_cuda_cache

        _empty_cuda_cache()  # no torch, or no GPU, or a GPU: never raises

    def test_the_allocator_actually_gives_memory_back(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this environment")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device available")

        from pipeline.worker import _empty_cuda_cache

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_reserved()
        block = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.assertGreater(torch.cuda.memory_reserved(), baseline)
        del block
        _empty_cuda_cache()
        self.assertLessEqual(torch.cuda.memory_reserved(), baseline)


class TestResidentStreaming(unittest.TestCase):
    def test_output_reaches_the_log_before_the_step_returns(self):
        """A 40-minute step must print progress live, resident path included."""
        dispatcher = _dispatcher(keep_loaded=True)
        with _CaptureLogs() as logs:
            try:
                dispatcher.run(
                    "_resident_probe", {}, {"model": "wan", "mode": "slow", "pause": 0.6}
                )
                returned = time.monotonic()
            finally:
                dispatcher.close()

        halfway = logs.first_time("SENTINEL_HALFWAY")
        self.assertGreater(
            returned - halfway, 0.4,
            "the halfway line only reached the log when run() returned; output is "
            "being captured and dumped rather than streamed",
        )

    def test_both_jobs_output_is_relayed_and_the_protocol_line_is_not(self):
        dispatcher = _dispatcher(keep_loaded=True)
        with _CaptureLogs() as logs:
            try:
                dispatcher.run("_resident_probe", {}, {"model": "wan"})
                dispatcher.run("_resident_probe", {}, {"model": "wan"})
            finally:
                dispatcher.close()

        runs = [m for m in logs.messages if m.startswith("probe run ")]
        self.assertEqual(len(runs), 2, f"expected one 'probe run' line per job, got {runs}")
        self.assertFalse(
            [m for m in logs.messages if SERVE_MARKER in m],
            "the status marker was relayed into the log as if it were step output",
        )


class TestResidentFailures(unittest.TestCase):
    def test_a_raising_step_fails_the_run_with_the_output_tail(self):
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            with _CaptureLogs():
                with self.assertRaises(RuntimeError) as caught:
                    dispatcher.run("_resident_probe", {}, {"model": "wan", "mode": "raise"})
        finally:
            dispatcher.close()

        message = str(caught.exception)
        self.assertIn("_resident_probe", message)
        self.assertIn("the probe step was told to fail", message)
        self.assertIn("SENTINEL_LAST_WORDS", message)
        self.assertIn("lines of output", message)

    def test_a_failure_ends_the_worker_and_the_next_call_gets_a_new_one(self):
        """The documented choice: a raised job kills the resident child.

        See pipeline/worker.py's serve() for why — an exception leaves the
        Step mid-run() with unknown internal state, and the runner aborts
        the whole run anyway.
        """
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            with _CaptureLogs():
                first = dispatcher.run("_resident_probe", {}, {"model": "wan"})
                with self.assertRaises(RuntimeError):
                    dispatcher.run("_resident_probe", {}, {"model": "wan", "mode": "raise"})
                self.assertIsNone(dispatcher._resident, "the failed worker was left attached")
                recovered = dispatcher.run("_resident_probe", {}, {"model": "wan"})
        finally:
            dispatcher.close()

        self.assertNotEqual(first["pid"], recovered["pid"])
        self.assertEqual(recovered["loads"], 1, "the replacement worker started warm somehow")

    def test_a_child_that_dies_mid_job_raises_instead_of_hanging(self):
        """An OOM kill gives EOF with no status line; that must not block."""
        dispatcher = _dispatcher(keep_loaded=True)
        try:
            with _CaptureLogs():
                with self.assertRaises(RuntimeError) as caught:
                    dispatcher.run("_resident_probe", {}, {"model": "wan", "mode": "die"})
        finally:
            dispatcher.close()

        message = str(caught.exception)
        self.assertIn("died mid-job", message)
        self.assertIn("exit 9", message)
        self.assertIn("SENTINEL_LAST_WORDS", message)

    def test_missing_interpreter_names_the_envs_registry(self):
        dispatcher = SubprocessPythonDispatcher(
            python_bin="/nonexistent/bin/python", keep_loaded=True
        )
        with self.assertRaises(RuntimeError) as caught:
            dispatcher.run("save_dataset", {}, {})
        self.assertIn("envs.yaml", str(caught.exception))
        dispatcher.close()


class TestResidentShutdown(unittest.TestCase):
    def test_close_unloads_the_step_and_reaps_the_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            unload_log = Path(tmp) / "unloads.txt"
            dispatcher = _dispatcher(keep_loaded=True, unload_log=unload_log)
            dispatcher.run("_resident_probe", {}, {"model": "wan"})
            child = dispatcher._resident
            self.assertIsNotNone(child)
            self.assertIsNone(child.poll(), "the worker exited before close()")

            dispatcher.close()

            self.assertIsNotNone(child.poll(), "the worker was left running after close()")
            self.assertEqual(child.returncode, 0)
            self.assertIn("ResidentProbeStep", unload_log.read_text())

    def test_close_is_idempotent_and_safe_with_no_worker(self):
        dispatcher = _dispatcher(keep_loaded=True)
        dispatcher.close()
        dispatcher.close()
        dispatcher.run("_resident_probe", {}, {"model": "wan"})
        dispatcher.close()
        dispatcher.close()


class TestWiring(unittest.TestCase):
    def test_the_factory_forwards_keep_loaded_to_the_subprocess_dispatcher(self):
        """The actual bug this work started from: it used to be dropped here."""
        built = build_dispatcher(
            "subprocess", {"python_bin": sys.executable}, keep_loaded=True
        )
        self.assertTrue(built.keep_loaded)
        self.assertFalse(
            build_dispatcher("subprocess", {"python_bin": sys.executable}).keep_loaded
        )

    def test_the_runner_keys_its_dispatcher_cache_on_keep_loaded(self):
        """Otherwise residency is decided by YAML declaration order.

        Two steps sharing dispatch+env is exactly how they come to share a
        resident worker (fast_helical_full's two denoise passes). A third
        step on the same env that did not ask for residency must not
        inherit it, and must not deny it to the two that did.
        """
        from pipeline.runner import WorkflowRunner
        from pipeline.workflow import StepSpec, WorkflowSpec

        runner = WorkflowRunner(WorkflowSpec(name="probe"))
        warm_a = StepSpec(id="a", step="_resident_probe", keep_loaded=True)
        warm_b = StepSpec(id="b", step="_resident_probe", keep_loaded=True)
        cold = StepSpec(id="c", step="_resident_probe", keep_loaded=False)

        self.assertIs(runner._get_dispatcher(warm_a), runner._get_dispatcher(warm_b))
        self.assertIsNot(runner._get_dispatcher(warm_a), runner._get_dispatcher(cold))
        self.assertTrue(runner._get_dispatcher(warm_a).keep_loaded)
        self.assertFalse(runner._get_dispatcher(cold).keep_loaded)

    def test_load_signature_rule(self):
        class Declared:
            LOAD_PARAMS = ("checkpoint", "use_lora")

        class Undeclared:
            pass

        self.assertEqual(
            load_signature(Declared, {"checkpoint": "a", "use_lora": True, "strength": 0.8}),
            {"checkpoint": "a", "use_lora": True},
        )
        # A load param that isn't in params at all still compares equal
        # across calls, so an omitted-then-omitted key never forces a reload.
        self.assertEqual(load_signature(Declared, {}), {"checkpoint": None, "use_lora": None})
        self.assertIsNone(load_signature(Undeclared, {"anything": 1}))


if __name__ == "__main__":
    unittest.main()
