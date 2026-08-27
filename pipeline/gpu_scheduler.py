"""Fans submitted runs out across every GPU on the box, one `pipeline.run_worker`
process per physical GPU at a time, first-free assignment, FIFO beyond that.

No queue file, socket, or database: this is in-memory state inside the
single Gradio process (see `pipeline.webui`), the same shape the old
single-run `RunManager` used — a lock, some state, a background thread —
just N slots wide instead of one. A run crosses into its own OS process
because CUDA context isolation between two GPUs is a property of the OS
process boundary (`CUDA_VISIBLE_DEVICES`), not something achievable by
juggling `torch.cuda.set_device` inside one interpreter across N cards and
N sets of resident-worker subprocesses.

`SubprocessPythonDispatcher._child_env` (pipeline/dispatch/subprocess_python.py)
already forwards the parent's environment — including `CUDA_VISIBLE_DEVICES`
— to every step subprocess and resident worker it spawns. So pinning it once
here, in the `Popen` that starts `pipeline.run_worker`, is enough to pin an
entire run's GPU usage; nothing downstream needs to know.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .run_state import RunJob, RunState, tail_lines

logger = logging.getLogger(__name__)

# (job, gpu_index, job_path, status_path) -> the started child process.
SpawnFn = Callable[[RunJob, int, Path, Path], "subprocess.Popen"]


def _default_spawn(job: RunJob, gpu_index: int, job_path: Path, status_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return subprocess.Popen(
        [
            sys.executable, "-m", "pipeline.run_worker",
            "--job", str(job_path), "--status", str(status_path),
        ],
        env=env,
    )


def detect_gpu_count() -> int:
    """`torch.cuda.device_count()`, falling back to 1 (today's behaviour)
    on a machine with no GPU visible at all — a dev laptop, most CI."""
    try:
        import torch
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except ImportError:
        pass
    return 1


@dataclass
class _Slot:
    gpu_index: int
    process: Optional["subprocess.Popen"] = None
    run_name: str = ""
    status_path: Optional[Path] = None

    @property
    def busy(self) -> bool:
        return self.process is not None and self.process.poll() is None


class GpuScheduler:
    """N GPU slots, a FIFO queue beyond that, automatic first-free assignment.

    `submit()` never blocks: it queues immediately and returns, whether or
    not a slot happens to be free. A per-slot watcher thread notices its
    child exit and dispatches the next queued job, so nothing polls for
    "is a slot free" beyond that.
    """

    def __init__(
        self,
        gpu_count: int,
        work_dir: Path,
        spawn: SpawnFn = _default_spawn,
    ) -> None:
        if gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        self._spawn = spawn
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._slots = [_Slot(gpu_index=i) for i in range(gpu_count)]
        self._queue: "deque[RunJob]" = deque()
        self._states: Dict[str, RunState] = {}
        self._order: List[str] = []  # submission order, for the UI's run picker

    @property
    def gpu_count(self) -> int:
        return len(self._slots)

    # -- submitting ---------------------------------------------------------

    def submit(self, job: RunJob) -> str:
        with self._lock:
            self._states[job.run_name] = RunState(
                name=job.run_name, workflow=job.workflow_name, status="queued",
                output_dir=Path(job.output_dir),
            )
            self._order.append(job.run_name)
            self._queue.append(job)
            self._dispatch_locked()
        return job.run_name

    def _dispatch_locked(self) -> None:
        """Assign queued jobs to free slots. Caller holds `self._lock`."""
        for slot in self._slots:
            if slot.busy or not self._queue:
                continue
            job = self._queue.popleft()
            job_path = self._work_dir / f"{job.run_name}.job.json"
            status_path = self._work_dir / f"{job.run_name}.status.json"
            job_path.write_text(json.dumps(job.to_dict()))

            try:
                process = self._spawn(job, slot.gpu_index, job_path, status_path)
            except Exception:
                logger.exception("failed to start a worker for run '%s'", job.run_name)
                state = self._states[job.run_name]
                state.status = "failed"
                state.finished = time.time()
                state.message = "failed to start a GPU worker process"
                continue

            slot.process = process
            slot.run_name = job.run_name
            slot.status_path = status_path

            state = self._states[job.run_name]
            state.status = "running"
            state.started = time.time()
            state.gpu_index = slot.gpu_index

            threading.Thread(
                target=self._watch, args=(slot,), daemon=True,
                name=f"gpu{slot.gpu_index}-{job.run_name}",
            ).start()

    # -- watching -------------------------------------------------------

    def _watch(self, slot: _Slot) -> None:
        run_name = slot.run_name
        status_path = slot.status_path
        process = slot.process
        assert process is not None and status_path is not None

        while process.poll() is None:
            self._refresh(run_name, status_path)
            time.sleep(1.0)
        self._refresh(run_name, status_path)

        with self._lock:
            state = self._states.get(run_name)
            if state is not None and state.status in ("queued", "running"):
                # The child exited without ever publishing a terminal
                # status — most likely an OOM kill.
                state.status = "failed"
                state.finished = time.time()
                state.message = f"worker exited unexpectedly (code {process.returncode})"

            if slot.process is process:
                slot.process = None
                slot.run_name = ""
                slot.status_path = None
            self._dispatch_locked()

    def _refresh(self, run_name: str, status_path: Path) -> None:
        try:
            raw = status_path.read_text()
        except OSError:
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return  # caught the writer mid-replace; next poll gets a clean copy
        with self._lock:
            previous = self._states.get(run_name)
            new_state = RunState.from_dict(data)
            new_state.gpu_index = previous.gpu_index if previous else None
            self._states[run_name] = new_state

    # -- reading --------------------------------------------------------

    def snapshot(self, run_name: str) -> RunState:
        with self._lock:
            return self._states.get(run_name, RunState())

    def log_text(self, run_name: str) -> str:
        state = self.snapshot(run_name)
        if not state.log_path:
            return ""
        return tail_lines(Path(state.log_path))

    def list_runs(self) -> List[RunState]:
        with self._lock:
            return [self._states[name] for name in self._order if name in self._states]

    def gpu_status(self) -> List[Dict[str, object]]:
        with self._lock:
            return [
                {"gpu": slot.gpu_index, "busy": slot.busy, "run": slot.run_name}
                for slot in self._slots
            ]

    def queued_count(self) -> int:
        with self._lock:
            return len(self._queue)

    # -- cancelling -------------------------------------------------------

    def cancel(self, run_name: str) -> None:
        with self._lock:
            for job in list(self._queue):
                if job.run_name == run_name:
                    self._queue.remove(job)
                    state = self._states.get(run_name)
                    if state is not None:
                        state.status = "cancelled"
                        state.message = "cancelled before it started"
                        state.finished = time.time()
                    return
            for slot in self._slots:
                if slot.run_name == run_name and slot.process is not None:
                    slot.process.terminate()
                    return

    # -- shutdown -----------------------------------------------------------

    def shutdown(self) -> None:
        """Best-effort SIGTERM to every running child, for a clean pod stop.

        The scheduler lives inside the Gradio process; nothing forwards a
        signal to children it spawned unless this does it explicitly.
        """
        with self._lock:
            processes = [slot.process for slot in self._slots if slot.process is not None]
        for process in processes:
            try:
                process.terminate()
            except OSError:
                pass
