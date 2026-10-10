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
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

from pipeline.gpu_scheduler import GpuScheduler, detect_gpu_count
from pipeline.run_state import RunJob, RunState, StepRecord


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

    def test_cancel_all_stops_running_and_queued_without_starting_the_queue(self):
        self.scheduler.submit(_job("a"))
        self.scheduler.submit(_job("b"))
        _wait_until(lambda: len(self.spawner.calls) == 2)
        self.scheduler.submit(_job("c"))

        names = self.scheduler.cancel_all()

        self.assertEqual(sorted(names), ["a", "b", "c"])
        self.assertTrue(self.spawner.processes["a"].terminated)
        self.assertTrue(self.spawner.processes["b"].terminated)
        self.assertEqual(self.scheduler.snapshot("c").status, "cancelled")
        # A freed slot must not pick up the cancelled "c".
        self.spawner.complete("a", status="cancelled")
        time.sleep(0.2)
        self.assertEqual(len(self.spawner.calls), 2)

    def test_a_worker_that_dies_without_a_terminal_status_is_reported_failed(self):
        """The fallback the exit code exists for: an OOM kill leaves
        whatever was last published (`running`), not a terminal status."""
        self.scheduler.submit(_job("a"))
        _wait_until(lambda: len(self.spawner.calls) == 1)

        self.spawner.processes["a"].finish(returncode=-9)  # no status update first

        _wait_until(lambda: self.scheduler.snapshot("a").status == "failed")
        self.assertIn("-9", self.scheduler.snapshot("a").message)
        # ...and on disk, which is all that is left after a restart.
        on_disk = RunState.from_dict(json.loads(self.spawner.calls[0]["status_path"].read_text()))
        self.assertEqual(on_disk.status, "failed")

    def test_a_restart_fails_the_runs_its_predecessor_left_live(self):
        """A status file still `running` when the server starts has no
        worker behind it; finished ones are left as they are."""
        work = Path(self.tmp.name)
        for name, status in (("lost", "running"), ("waiting", "queued"), ("over", "done")):
            (work / f"{name}.status.json").write_text(json.dumps(RunState(
                name=name, status=status, message="[3/9] train",
                steps=[StepRecord(1, "a", "a", "done"), StepRecord(2, "b", "b", "running")],
            ).to_dict()))

        GpuScheduler(gpu_count=1, work_dir=work, spawn=self.spawner)

        read = lambda n: RunState.from_dict(json.loads((work / f"{n}.status.json").read_text()))
        for name in ("lost", "waiting"):
            state = read(name)
            self.assertEqual(state.status, "failed")
            self.assertTrue(state.message.startswith("[3/9] train: "))
            self.assertGreater(state.finished, 0)
            self.assertEqual([s.status for s in state.steps], ["done", "failed"])
        self.assertEqual(read("over").status, "done")
        self.assertEqual(read("over").steps[1].status, "running")

    def test_a_queued_run_is_on_disk_the_moment_it_is_submitted(self):
        """Before a slot takes it, the files are the only copy that outlives
        the server — 2026-10-10 a host freeze lost two queued runs' prompts."""
        scheduler = GpuScheduler(gpu_count=1, work_dir=Path(self.tmp.name), spawn=self.spawner)
        scheduler.submit(_job("running"))
        job = _job("waiting")
        job.prompt = "a prompt nobody wrote down"
        scheduler.submit(job)

        work = Path(self.tmp.name)
        self.assertEqual(RunJob.from_dict(json.loads((work / "waiting.job.json").read_text())), job)
        state = RunState.from_dict(json.loads((work / "waiting.status.json").read_text()))
        self.assertEqual(state.status, "queued")

    def test_a_restart_queues_again_what_its_predecessor_left_waiting(self):
        """In submission order, and dispatched; the run that was running
        is failed, not started over."""
        work = Path(self.tmp.name)
        first = GpuScheduler(gpu_count=1, work_dir=work, spawn=self.spawner)
        for name in ("running", "w-2", "w-1", "w-3"):
            first.submit(_job(name))
        # The server dies here: no shutdown, nothing written.

        spawner = _FakeSpawner()
        second = GpuScheduler(gpu_count=1, work_dir=work, spawn=spawner)

        read = lambda n: RunState.from_dict(json.loads((work / f"{n}.status.json").read_text()))
        self.assertEqual(read("running").status, "failed")
        self.assertEqual([call["run_name"] for call in spawner.calls], ["w-2"])
        self.assertEqual([s.name for s in second.list_runs()], ["w-2", "w-1", "w-3"])
        self.assertEqual(second.queued_count(), 2)
        spawner.complete("w-2")
        _wait_until(lambda: [c["run_name"] for c in spawner.calls] == ["w-2", "w-1"])

    def test_a_restart_after_shutdown_queues_the_dropped_runs_again(self):
        work = Path(self.tmp.name)
        first = GpuScheduler(gpu_count=1, work_dir=work, spawn=self.spawner)
        first.submit(_job("running"))
        first.submit(_job("waiting"))
        first.shutdown()

        spawner = _FakeSpawner()
        GpuScheduler(gpu_count=1, work_dir=work, spawn=spawner)
        self.assertEqual([call["run_name"] for call in spawner.calls], ["waiting"])

    def test_a_cancelled_queued_run_stays_cancelled_after_a_restart(self):
        work = Path(self.tmp.name)
        first = GpuScheduler(gpu_count=1, work_dir=work, spawn=self.spawner)
        for name in ("running", "cancel-one", "cancel-all"):
            first.submit(_job(name))
        first.cancel("cancel-one")
        first.cancel_all()

        spawner = _FakeSpawner()
        GpuScheduler(gpu_count=1, work_dir=work, spawn=spawner)
        self.assertEqual(spawner.calls, [])
        for name in ("cancel-one", "cancel-all"):
            state = RunState.from_dict(json.loads((work / f"{name}.status.json").read_text()))
            self.assertEqual(state.status, "cancelled")

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

    def test_shutdown_does_not_start_the_next_queued_run(self):
        """Terminating a worker frees its slot, and a free slot dispatches.

        Which meant stopping the server *started* fresh workers: the dying
        one's watcher reached its `finally`, freed the slot and pulled the
        next job off the queue, so a shutdown spawned processes that began
        holding GPUs as the server exited — and outlived it on any host
        that does not tear the whole container down.
        """
        scheduler = GpuScheduler(
            gpu_count=1, work_dir=Path(self.tmp.name), spawn=self.spawner,
        )
        scheduler.submit(_job("running"))
        scheduler.submit(_job("queued-1"))
        scheduler.submit(_job("queued-2"))
        self.assertEqual(len(self.spawner.calls), 1)

        scheduler.shutdown()
        # The watcher thread notices the terminated child within its 1s
        # poll; give it room to do the wrong thing if it is going to.
        time.sleep(1.5)

        self.assertEqual(
            [call["run_name"] for call in self.spawner.calls], ["running"],
            "shutdown started a queued run",
        )
        self.assertEqual(scheduler.queued_count(), 0)

    def test_shutdown_marks_the_runs_it_dropped(self):
        scheduler = GpuScheduler(
            gpu_count=1, work_dir=Path(self.tmp.name), spawn=self.spawner,
        )
        scheduler.submit(_job("running"))
        scheduler.submit(_job("dropped"))
        scheduler.shutdown()
        self.assertEqual(scheduler.snapshot("dropped").status, "cancelled")
        self.assertIn("shutting down", scheduler.snapshot("dropped").message)

    def test_submitting_after_shutdown_is_refused(self):
        self.scheduler.shutdown()
        with self.assertRaises(RuntimeError):
            self.scheduler.submit(_job("late"))

    def test_shutdown_waits_for_a_worker_that_stops(self):
        # A worker between steps exits on SIGTERM, and the caller needs to
        # know that before it runs something that destroys the machine.
        self.scheduler.submit(_job("a"))
        self.assertEqual(self.scheduler.shutdown(timeout=5.0), [])
        self.assertTrue(self.spawner.processes["a"].terminated)

    def test_shutdown_names_a_worker_that_did_not_stop_in_time(self):
        """A step is one opaque call; a brush training has no boundary for
        an hour, so its worker is still there when the wait expires."""
        class _Stubborn(_FakeProcess):
            def terminate(self):
                self.terminated = True  # ignores it, like a worker mid-step

        stubborn = _Stubborn()
        scheduler = GpuScheduler(
            gpu_count=1, work_dir=Path(self.tmp.name), spawn=lambda *_: stubborn,
        )
        scheduler.submit(_job("long-step"))
        self.assertEqual(scheduler.shutdown(timeout=0.3), ["long-step"])

    def test_a_spawn_that_raises_fails_that_run_without_wedging_the_slot(self):
        def _bad_spawn(job, gpu_index, job_path, status_path):
            raise OSError("no such interpreter")

        scheduler = GpuScheduler(gpu_count=1, work_dir=Path(self.tmp.name), spawn=_bad_spawn)
        scheduler.submit(_job("a"))

        self.assertEqual(scheduler.snapshot("a").status, "failed")
        self.assertEqual(scheduler.gpu_status()[0]["busy"], False)


class DetectGpuCountTest(unittest.TestCase):
    """`nvidia-smi -L`, with `CUDA_VISIBLE_DEVICES` taking precedence the
    way it would for torch, and 1 whenever nothing can be counted."""

    def _run(self, stdout="", returncode=0, env=None, side_effect=None):
        result = subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")
        with mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch("pipeline.gpu_scheduler.subprocess.run",
                           return_value=result, side_effect=side_effect) as run:
            if env is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            return detect_gpu_count(), run

    def test_counts_the_gpu_lines(self):
        stdout = ("GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-a)\n"
                  "GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-b)\n")
        self.assertEqual(self._run(stdout)[0], 2)

    def test_mig_lines_are_not_counted_as_gpus(self):
        stdout = ("GPU 0: NVIDIA A100 (UUID: GPU-a)\n"
                  "  MIG 1g.10gb Device 0: (UUID: MIG-a)\n")
        self.assertEqual(self._run(stdout)[0], 1)

    def test_cuda_visible_devices_wins_and_skips_nvidia_smi(self):
        count, run = self._run(env={"CUDA_VISIBLE_DEVICES": "2,3,5"})
        self.assertEqual(count, 3)
        run.assert_not_called()

    def test_falls_back_to_one(self):
        self.assertEqual(self._run(returncode=9)[0], 1)
        self.assertEqual(self._run(side_effect=FileNotFoundError())[0], 1)
        self.assertEqual(self._run(env={"CUDA_VISIBLE_DEVICES": ""})[0], 1)


if __name__ == "__main__":
    unittest.main()
