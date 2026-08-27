"""`GpuScheduler`: N GPU slots, automatic first-free assignment, a FIFO
queue beyond that.

The spawn function is stubbed throughout — no real `pipeline.run_worker`
process, no GPU — so what these pin down is the scheduling logic itself:
two submissions land on distinct slots, a third queues until one frees,
cancelling a queued job never touches a slot, and a status file the worker
writes atomically is picked up on the next poll. The real subprocess
boundary (`pipeline.run_worker` actually running, `CUDA_VISIBLE_DEVICES`
actually reaching a child) is exercised separately by
`tests/test_run_worker.py`.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.gpu_scheduler import GpuScheduler
from pipeline.run_state import RunJob, RunState


class _FakeProcess:
    """Stands in for `subprocess.Popen`: finishes when told to, not on a
    timer — so a test controls exactly when a slot frees rather than racing
    a sleep against the scheduler's own 1s poll."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self.returncode: Optional[int] = None
        self.terminated = False

    def poll(self) -> Optional[int]:
        return self.returncode if self._done.is_set() else None

    def finish(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._done.set()

    def terminate(self) -> None:
        self.terminated = True
        self.finish(returncode=-15)


class _FakeSpawner:
    """A stand-in `spawn` callable that records every call and lets the
    test control completion and status-file content per run."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, object]] = []
        self.processes: Dict[str, _FakeProcess] = {}

    def __call__(self, job: RunJob, gpu_index: int, job_path: Path, status_path: Path):
        process = _FakeProcess()
        self.processes[job.run_name] = process
        self.calls.append({
            "run_name": job.run_name, "gpu_index": gpu_index,
            "job_path": job_path, "status_path": status_path,
        })
        # A real worker's first act is publishing a "running" status; do the
        # same here so `_watch`'s first poll has something to read.
        status_path.write_text(json.dumps(
            RunState(name=job.run_name, workflow=job.workflow_name, status="running").to_dict()
        ))
        return process

    def complete(self, run_name: str, status: str = "done") -> None:
        """What a real worker does right before exiting: publish a
        terminal status, then the process itself ends."""
        process = self.processes[run_name]
        for call in self.calls:
            if call["run_name"] == run_name:
                status_path = call["status_path"]
                break
        state = RunState.from_dict(json.loads(status_path.read_text()))
        state.status = status
        status_path.write_text(json.dumps(state.to_dict()))
        process.finish(0 if status == "done" else 1)


def _job(name: str) -> RunJob:
    return RunJob(
        run_name=name, workflow_name="wf", workflow_path="/dev/null",
        output_dir=f"/tmp/{name}",
    )


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never became true")


class TestGpuScheduler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spawner = _FakeSpawner()
        self.scheduler = GpuScheduler(
            gpu_count=2, work_dir=Path(self.tmp.name), spawn=self.spawner
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_submissions_run_concurrently_on_distinct_slots(self):
        self.scheduler.submit(_job("a"))
        self.scheduler.submit(_job("b"))

        _wait_until(lambda: len(self.spawner.calls) == 2)
        gpu_indices = {call["gpu_index"] for call in self.spawner.calls}
        self.assertEqual(gpu_indices, {0, 1})
        self.assertEqual(self.scheduler.snapshot("a").status, "running")
        self.assertEqual(self.scheduler.snapshot("b").status, "running")
        self.assertEqual(self.scheduler.queued_count(), 0)

    def test_a_third_submission_queues_until_a_slot_frees(self):
        self.scheduler.submit(_job("a"))
        self.scheduler.submit(_job("b"))
        _wait_until(lambda: len(self.spawner.calls) == 2)

        self.scheduler.submit(_job("c"))
        self.assertEqual(self.scheduler.snapshot("c").status, "queued")
        self.assertEqual(self.scheduler.queued_count(), 1)
        self.assertEqual(len(self.spawner.calls), 2, "c must not have been spawned yet")

        self.spawner.complete("a", status="done")
        _wait_until(lambda: len(self.spawner.calls) == 3)
        self.assertEqual(self.scheduler.queued_count(), 0)
        _wait_until(lambda: self.scheduler.snapshot("c").status == "running")

    def test_cancelling_a_queued_job_never_touches_a_slot(self):
        self.scheduler.submit(_job("a"))
        self.scheduler.submit(_job("b"))
        _wait_until(lambda: len(self.spawner.calls) == 2)

        self.scheduler.submit(_job("c"))
        self.scheduler.cancel("c")

        self.assertEqual(self.scheduler.snapshot("c").status, "cancelled")
        self.assertEqual(self.scheduler.queued_count(), 0)
        # Completing "a" must not resurrect the cancelled "c".
        self.spawner.complete("a", status="done")
        time.sleep(0.2)
        self.assertEqual(len(self.spawner.calls), 2)

    def test_cancelling_a_running_job_terminates_its_process(self):
        self.scheduler.submit(_job("a"))
        _wait_until(lambda: len(self.spawner.calls) == 1)

        self.scheduler.cancel("a")

        self.assertTrue(self.spawner.processes["a"].terminated)

    def test_a_worker_that_dies_without_a_terminal_status_is_reported_failed(self):
        """The fallback the exit code exists for: an OOM kill leaves
        whatever was last published (`running`), not a terminal status."""
        self.scheduler.submit(_job("a"))
        _wait_until(lambda: len(self.spawner.calls) == 1)

        self.spawner.processes["a"].finish(returncode=-9)  # no status update first

        _wait_until(lambda: self.scheduler.snapshot("a").status == "failed")
        self.assertIn("-9", self.scheduler.snapshot("a").message)

    def test_gpu_status_reports_busy_slots_and_their_run(self):
        self.scheduler.submit(_job("a"))
        _wait_until(lambda: len(self.spawner.calls) == 1)

        slots = self.scheduler.gpu_status()
        busy = [s for s in slots if s["busy"]]
        self.assertEqual(len(busy), 1)
        self.assertEqual(busy[0]["run"], "a")

    def test_list_runs_preserves_submission_order(self):
        self.scheduler.submit(_job("a"))
        self.scheduler.submit(_job("b"))
        self.scheduler.submit(_job("c"))

        self.assertEqual([s.name for s in self.scheduler.list_runs()], ["a", "b", "c"])

    def test_gpu_count_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            GpuScheduler(gpu_count=0, work_dir=Path(self.tmp.name), spawn=self.spawner)

    def test_a_spawn_that_raises_fails_that_run_without_wedging_the_slot(self):
        def _bad_spawn(job, gpu_index, job_path, status_path):
            raise OSError("no such interpreter")

        scheduler = GpuScheduler(gpu_count=1, work_dir=Path(self.tmp.name), spawn=_bad_spawn)
        scheduler.submit(_job("a"))

        self.assertEqual(scheduler.snapshot("a").status, "failed")
        self.assertEqual(scheduler.gpu_status()[0]["busy"], False)


if __name__ == "__main__":
    unittest.main()
