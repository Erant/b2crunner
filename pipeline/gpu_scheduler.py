"""Fans submitted runs out across every GPU on the box, one `pipeline.run_worker`
process per physical GPU at a time, first-free assignment, FIFO beyond that.

No socket or database: this is in-memory state inside the single Gradio
process (see `pipeline.webui`), the same shape the old single-run
`RunManager` used — a lock, some state, a background thread — just N slots
wide instead of one. The queue is backed by files, though: a run's
`<name>.job.json` and a `queued` status are written the moment it is
submitted, and a restarted server picks those runs up again in submission
order (`_recover`). Before 2026-10-10 the job file was only written when a
slot took the run, so a crash or restart lost every queued run, prompt and
all. A run crosses into its own OS process
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

from .run_state import RunJob, RunState, run_owner_alive, tail_lines

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
    """The GPUs this process can see, falling back to 1 (today's behaviour)
    on a machine with no GPU visible at all — a dev laptop, most CI.

    `nvidia-smi -L` rather than `torch.cuda.device_count()`: importing torch
    only to count cards cost the UI ~12 s of its start in a fresh container.
    `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so a set one is counted
    here instead, the way torch would have.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return max(1, len([d for d in visible.split(",") if d.strip()]))
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1
    if result.returncode != 0:
        return 1
    return max(1, sum(line.startswith("GPU ") for line in result.stdout.splitlines()))


def _write_json(path: Path, data: Dict, run_name: str) -> None:
    """By atomic replace, as the worker writes its status."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    except OSError as error:
        logger.warning("could not write %s of run '%s': %s", path.name, run_name, error)


def _publish(status_path: Path, state: RunState) -> None:
    """Write a status the worker never got to. Steps it left `running`
    died with it."""
    for step in state.steps:
        if step.status == "running":
            step.status = "failed"
    _write_json(status_path, state.to_dict(), state.name)


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
        self._stopping = False
        self._slots = [_Slot(gpu_index=i) for i in range(gpu_count)]
        self._queue: "deque[RunJob]" = deque()
        self._states: Dict[str, RunState] = {}
        self._order: List[str] = []  # submission order, for the UI's run picker
        self._recover()

    def _paths(self, run_name: str):
        return (self._work_dir / f"{run_name}.job.json",
                self._work_dir / f"{run_name}.status.json")

    def _recover(self) -> None:
        """Take back the runs the previous server process left queued, and
        fail the ones it left running.

        This scheduler is the only thing that starts workers into
        `work_dir`, and it has started none yet. A `queued` status that
        never started, with its job file beside it, is a run that was
        waiting for a slot: it goes back on the queue, in the order the
        job files were written (submission order). A `running` status (or
        a `queued` one with no job file, from before job files were written
        at submit) belongs to a worker the previous server lost: killed
        with it, or killed before `_watch` wrote the file. Left alone, those
        show as in flight forever. A worker that did outlive its server
        republishes its whole state at the next step, so it corrects this
        itself.

        A `pipeline.cli` run is not any server's: it names its own process
        (`RunState.owner`), and is left alone unless that is known to be
        gone.
        """
        requeue = []
        for path in sorted(self._work_dir.glob("*.status.json")):
            try:
                state = RunState.from_dict(json.loads(path.read_text()))
                finished = path.stat().st_mtime
            except (OSError, ValueError):
                continue
            if state.status not in ("queued", "running"):
                continue
            if state.owner and run_owner_alive(state.owner, finished) is not False:
                continue
            job_path = self._work_dir / path.name.replace(".status.json", ".job.json")
            if state.status == "queued" and not state.started:
                try:
                    job = RunJob.from_dict(json.loads(job_path.read_text()))
                    requeue.append((job_path.stat().st_mtime_ns, job.run_name, job, state))
                    continue
                except (OSError, ValueError, KeyError) as error:
                    logger.warning("run '%s' was queued but its job file is unreadable: %s",
                                   state.name, error)
            state.status = "failed"
            state.finished = finished
            state.message = f"{state.message}: the worker was lost (server restarted)".lstrip(": ")
            _publish(path, state)
            logger.info("run '%s' had no worker left; marked failed", state.name)

        if not requeue:
            return
        requeue.sort(key=lambda item: item[:2])
        with self._lock:
            for _, name, job, state in requeue:
                self._states[name] = state
                self._order.append(name)
                self._queue.append(job)
            self._dispatch_locked()
        logger.info("re-queued %d run(s) the previous server left waiting: %s",
                    len(requeue), ", ".join(item[1] for item in requeue))

    @property
    def gpu_count(self) -> int:
        return len(self._slots)

    # -- submitting ---------------------------------------------------------

    def submit(self, job: RunJob) -> str:
        with self._lock:
            if self._stopping:
                raise RuntimeError(
                    "this scheduler is shutting down and is not accepting work"
                )
            state = RunState(
                name=job.run_name, workflow=job.workflow_name, status="queued",
                output_dir=Path(job.output_dir),
            )
            # On disk before anything else: until a slot takes it, this is
            # the only copy of the run that would survive the server.
            job_path, status_path = self._paths(job.run_name)
            _write_json(job_path, job.to_dict(), job.run_name)
            _write_json(status_path, state.to_dict(), job.run_name)
            self._states[job.run_name] = state
            self._order.append(job.run_name)
            self._queue.append(job)
            self._dispatch_locked()
        return job.run_name

    def _dispatch_locked(self) -> None:
        """Assign queued jobs to free slots. Caller holds `self._lock`.

        Nothing is started once `shutdown` has been called. Without that
        check, terminating a running worker *starts the next queued one*:
        the dying worker's watcher reaches its `finally`, frees the slot,
        and dispatches — so stopping the server would spawn fresh workers
        that begin pulling checkpoints and holding GPUs as it exits, and
        outlive it on any host that does not tear the whole container down.
        """
        if self._stopping:
            return
        for slot in self._slots:
            if slot.busy or not self._queue:
                continue
            job = self._queue.popleft()
            job_path, status_path = self._paths(job.run_name)
            _write_json(job_path, job.to_dict(), job.run_name)
            # `running` before the worker exists, so a crash between here
            # and its first status reads as a lost worker, not a queued run
            # to start over.
            state = self._states[job.run_name]
            state.status = "running"
            state.started = time.time()
            state.gpu_index = slot.gpu_index
            _write_json(status_path, state.to_dict(), job.run_name)

            try:
                process = self._spawn(job, slot.gpu_index, job_path, status_path)
            except Exception:
                logger.exception("failed to start a worker for run '%s'", job.run_name)
                state.status = "failed"
                state.finished = time.time()
                state.message = "failed to start a GPU worker process"
                _publish(status_path, state)
                continue

            slot.process = process
            slot.run_name = job.run_name
            slot.status_path = status_path

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
                # The file too: it is what `discover_runs` reads once this
                # process has forgotten the run.
                _publish(status_path, state)

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
                    self._cancel_queued_locked(run_name)
                    return
            for slot in self._slots:
                if slot.run_name == run_name and slot.process is not None:
                    slot.process.terminate()
                    return

    def cancel_all(self) -> List[str]:
        """Cancel every queued and running run; return their names.

        The queue is emptied before any worker is signalled, and under the
        same lock: a terminated worker's watcher dispatches the next queued
        job onto the slot it frees, so the other order would start a run
        just to cancel it.
        """
        with self._lock:
            names = [job.run_name for job in self._queue]
            self._queue.clear()
            for name in names:
                self._cancel_queued_locked(name)
            for slot in self._slots:
                if slot.process is not None:
                    slot.process.terminate()
                    names.append(slot.run_name)
        return names

    def _cancel_queued_locked(self, run_name: str) -> None:
        """Mark a run taken off the queue cancelled, in memory and on disk
        (or the next server start would queue it again)."""
        state = self._states.get(run_name)
        if state is None:
            return
        state.status = "cancelled"
        state.message = "cancelled before it started"
        state.finished = time.time()
        _publish(self._paths(run_name)[1], state)

    def forget(self, run_names: List[str]) -> None:
        """Drop these runs from the list, if they are over.

        A queued or running run is kept whatever the caller asked: its
        watcher still writes into `_states`, and the row is how it is seen.
        """
        with self._lock:
            for name in run_names:
                state = self._states.get(name)
                if state is not None and state.status in ("queued", "running"):
                    continue
                self._states.pop(name, None)
                if name in self._order:
                    self._order.remove(name)

    # -- shutdown -----------------------------------------------------------

    def shutdown(self, timeout: float = 0.0) -> List[str]:
        """SIGTERM every running child; return the ones still alive after.

        The scheduler lives inside the server process; nothing forwards a
        signal to children it spawned unless this does it explicitly. It
        also drops whatever is still queued and latches `_stopping`, so a
        terminated worker's watcher cannot start the next job on the slot
        it just freed. The dropped runs are cancelled in memory only: their
        files still say `queued`, so the next server start queues them
        again (`_recover`).

        `timeout` waits for the children to actually go. That matters when
        something is going to happen *after* this returns — a
        `B2C_SHUTDOWN_COMMAND` that destroys the machine, say. But it is a
        bound, not a guarantee: `pipeline.run_worker` checks its cancelled
        flag at step boundaries only, and a step is one opaque call that
        can be an hour of brush training, so a worker inside a long one
        will still be alive when this returns. The names it returns are
        exactly those, for the caller to report.
        """
        with self._lock:
            self._stopping = True
            dropped = [job.run_name for job in self._queue]
            self._queue.clear()
            for name in dropped:
                state = self._states.get(name)
                if state is not None and state.status in ("queued", "running"):
                    state.status = "cancelled"
                    state.message = "stopped: the server was shutting down (queued again at its next start)"
                    state.finished = time.time()
            slots = [
                (slot.run_name, slot.process)
                for slot in self._slots if slot.process is not None
            ]
        if dropped:
            logger.info("dropped %d queued run(s): %s", len(dropped), ", ".join(dropped))

        for _name, process in slots:
            try:
                process.terminate()
            except OSError:
                pass

        deadline = time.monotonic() + max(0.0, timeout)
        alive = []
        for name, process in slots:
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                alive.append(name)
        return alive
