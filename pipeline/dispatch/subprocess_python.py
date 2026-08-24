"""Runs a step inside a separate Python interpreter (its own venv/conda env).

For steps whose dependencies conflict with the main environment or with each
other — SAM-3D-Body's pinned detectron2 build, SeedVR2's own torch/diffusers
pins, Sapiens2, Wan2.2/diffusers. Each such
step gets its own venv under `envs/<name>/` (created out-of-band, e.g. via
`uv venv envs/sam3dbody && uv pip install -r envs/sam3dbody/requirements.txt`
plus `uv pip install -e .` for this `pipeline` package itself), and this
dispatcher just shells out to that venv's interpreter running
`pipeline.worker`.

IPC is file-based (pickle for data, JSON for params) rather than pipes/stdin
— simplest thing that works for research-project sized payloads (a batch of
frames), and it's trivial to inspect a stuck run by looking at the temp dir.

**Output is streamed, not captured.** This used to be a
`subprocess.run(capture_output=True)`, which meant the three steps that
dispatch this way — `wan22_vace_denoise`, `seedvr2`, `sam3d_body`, i.e.
exactly the long ones — produced literally nothing until they exited. A
40-minute denoise and a hung denoise looked identical from outside. Now
every line the child writes is relayed to this process's logger as it
arrives, so it lands in the console, the run's log file, and the web UI
alike. The tail is still buffered so a failure can quote it in the
exception, which is what the old behaviour was actually for.

**keep_loaded keeps the child alive.** Off by default, and with it off
nothing below changes: one `python -m pipeline.worker <argv>` per call,
which is the right shape for a step invoked once. It is wrong for
`pipeline/workflows/fast_helical_full.yaml`, which calls
`wan22_vace_denoise` at stage 1 and again at stage 4 with brush training, a
splat re-render, an anchor re-inject and a mask in between — the calls
cannot be merged, and each fresh process re-reads ~47 GB of weights off the
pod's network volume. With `keep_loaded: true` the dispatcher instead
starts ONE `pipeline.worker --serve` child for the step, feeds it one job
per call over stdin, and keeps the loaded Step (and its weights) alive
between them; `close()` — which WorkflowRunner already calls at the end of
a run — shuts it down.

The control channel is deliberately the smallest thing that works: one line
of JSON in on the child's stdin (`kind` plus, for a job, the same four
values one-shot argv carries), one SERVE_MARKER status line back on the
stdout stream we are already reading. Three kinds — "run", "release_vram",
"shutdown". Payloads still go through files, so a stuck resident worker is
inspected exactly the way a stuck one-shot worker is. No concurrency: the
parent writes a request and blocks on its status, so there are no request
ids to correlate and no queue to drain.

**Resident means resident in DRAM, not in VRAM.** The child hands the card
back after every job (Step.release_vram plus an empty_cache, before it
reports the job done) and keeps only the host-RAM copy. It has to: the
steps between fast_helical_full's two denoise passes include `brush`,
which trains a Gaussian splat on the same GPU, and a worker sitting on ~35
GB of Wan experts would OOM it — a regression over the reloading this
replaces, not an improvement. Skipping the network read is the win;
squatting on VRAM was never part of it.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from ..worker import SERVE_MARKER
from .base import Dispatcher

logger = logging.getLogger(__name__)

# How many lines of child output to quote in the exception when a step
# fails. The full output is already in the log file by then; this is just
# enough to make the traceback self-contained.
_ERROR_TAIL_LINES = 60

# How long to wait for a resident worker to exit after its stdin is closed
# and its stdout has run to EOF. By that point it has already finished
# unloading (the unload happens before the process can close stdout), so
# this is only covering an interpreter that is slow to tear down a CUDA
# context — generous, but bounded, because close() runs in a `finally` and
# must not be the thing that hangs a completed run.
_SHUTDOWN_TIMEOUT_S = 60.0
_KILL_TIMEOUT_S = 10.0


class SubprocessPythonDispatcher(Dispatcher):
    def __init__(
        self,
        python_bin: str,
        cwd: str | None = None,
        env: Dict[str, str] | None = None,
        keep_loaded: bool = False,
    ):
        """python_bin: path to the isolated venv's interpreter, e.g.
        'envs/sam3dbody/bin/python'.

        keep_loaded: reuse one long-lived `--serve` child across calls
        instead of spawning a fresh process each time. See the module
        docstring; the reload rule lives in `pipeline.worker.load_signature`.
        """
        self.python_bin = python_bin
        self.cwd = cwd
        self.env = env
        self.keep_loaded = keep_loaded
        self._resident: Optional[subprocess.Popen] = None

    def _child_env(self) -> Dict[str, str]:
        """This process's environment with the env-config overrides layered on.

        NOT `self.env` alone, which is what subprocess would use verbatim:
        an envs.yaml entry that sets one variable would otherwise drop
        PATH, HF_HOME, HF_TOKEN, TMPDIR and CUDA_VISIBLE_DEVICES along with
        everything else, and the resulting failure (a gated model 401ing,
        or a download filling the container's overlay) points nowhere near
        the config that caused it.
        """
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        return env

    def run(self, step_name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"b2c_{step_name}_") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "inputs.pkl"
            params_path = tmp_path / "params.json"
            output_path = tmp_path / "outputs.pkl"

            with open(input_path, "wb") as f:
                pickle.dump(inputs, f)
            with open(params_path, "w") as f:
                json.dump(params, f)

            logger.info(
                "%s: dispatching to %s%s (inputs %.1f MB via %s)",
                step_name,
                self.python_bin,
                " [resident]" if self.keep_loaded else "",
                input_path.stat().st_size / 1e6,
                tmp_path,
            )

            started = time.time()
            if self.keep_loaded:
                tail = self._run_on_resident(
                    step_name, str(input_path), str(params_path), str(output_path)
                )
            else:
                cmd = [
                    self.python_bin, "-m", "pipeline.worker",
                    step_name, str(input_path), str(params_path), str(output_path),
                ]
                returncode, tail = self._stream(cmd, step_name)
                if returncode != 0:
                    elapsed = time.time() - started
                    raise RuntimeError(
                        f"Step '{step_name}' failed after {elapsed:.1f}s in isolated env "
                        f"'{self.python_bin}' (exit {returncode}).\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
                    )
            elapsed = time.time() - started

            if not output_path.exists():
                raise RuntimeError(
                    f"Step '{step_name}' exited 0 but wrote no output pickle to "
                    f"{output_path}. This usually means the worker was killed "
                    f"(OOM) rather than raising.\n"
                    f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
                )

            with open(output_path, "rb") as f:
                outputs = pickle.load(f)

            logger.info("%s: finished in %.1fs", step_name, elapsed)
            return outputs

    # ---------------------------------------------------------------- one-shot

    def _stream(self, cmd: list[str], step_name: str) -> tuple[int, deque]:
        """Run `cmd`, relaying each output line to the logger as it arrives."""
        relay = logging.getLogger(f"step.{step_name}")
        tail: deque = deque(maxlen=_ERROR_TAIL_LINES)

        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                env=self._child_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise self._interpreter_missing(step_name, cmd[0]) from None

        # `with process` closes the stdout pipe; a bare wait() leaks one
        # file descriptor per dispatched step.
        assert process.stdout is not None
        with process:
            for line in process.stdout:
                tail.append(line)
                relay.info("%s", line.rstrip())

        return process.returncode, tail

    def _interpreter_missing(self, step_name: str, binary: str) -> RuntimeError:
        """Shared by both paths: a bad python_bin is a config mistake, and
        the message has to name the file that holds it rather than the argv
        that used it."""
        return RuntimeError(
            f"Interpreter not found for step '{step_name}': {binary}\n"
            f"Check the 'python_bin' for this step's env in the envs registry "
            f"(pipeline/envs/envs.yaml, or docker/envs.docker.yaml inside the image)."
        )

    # ---------------------------------------------------------------- resident

    def _ensure_resident(self, step_name: str) -> subprocess.Popen:
        """The live `--serve` child, started on first use and after a death.

        `poll()` rather than trusting the handle: a child can die between
        jobs (the OOM killer picking off the biggest RSS on the box while
        brush trains, say), and the next job should get a fresh worker
        rather than a BrokenPipeError from a write to a corpse.
        """
        if self._resident is not None and self._resident.poll() is None:
            return self._resident
        if self._resident is not None:
            logger.warning(
                "%s: resident worker exited between jobs (code %s); starting a new one",
                step_name, self._resident.returncode,
            )
            self._shutdown_resident(drain=False)

        cmd = [self.python_bin, "-m", "pipeline.worker", "--serve"]
        try:
            self._resident = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                env=self._child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise self._interpreter_missing(step_name, cmd[0]) from None

        logger.info(
            "%s: started resident worker pid %d (%s)",
            step_name, self._resident.pid, self.python_bin,
        )
        return self._resident

    @staticmethod
    def _send(process: subprocess.Popen, request: Dict[str, Any]) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

    def release_vram(self) -> None:
        """Ask a resident worker to give the card back, keeping its weights.

        The resident worker already does this after every job, so nothing
        in the current runner needs to call this — it exists because the
        job boundary is not the only moment the GPU might be wanted, and a
        control channel that can only say "run" would make adding that a
        protocol change rather than a call. No-op when there is no live
        child, which is every one-shot dispatcher.
        """
        process = self._resident
        if process is None or process.poll() is not None:
            return
        assert process.stdout is not None
        try:
            self._send(process, {"kind": "release_vram"})
            relay = logging.getLogger("step.worker")
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                if line.startswith(SERVE_MARKER):
                    return
                relay.info("%s", line.rstrip())
        except (BrokenPipeError, ValueError, OSError):
            pass
        # Fell out of the loop: the child died while releasing. Nothing to
        # raise about — the GPU is free either way, which is what was asked
        # — but the handle must not survive into the next job.
        self._shutdown_resident(drain=False)

    def _run_on_resident(
        self, step_name: str, input_path: str, params_path: str, output_path: str
    ) -> deque:
        """Hand one job to the resident child and block until it reports back.

        Returns the tail of that job's output; raises on any failure, having
        first shut the child down (see the comment in worker.serve for why a
        failed job ends the worker).
        """
        relay = logging.getLogger(f"step.{step_name}")
        tail: deque = deque(maxlen=_ERROR_TAIL_LINES)
        process = self._ensure_resident(step_name)
        request = {
            "kind": "run",
            "step": step_name,
            "inputs": input_path,
            "params": params_path,
            "outputs": output_path,
        }

        assert process.stdin is not None and process.stdout is not None
        try:
            self._send(process, request)
        except (BrokenPipeError, ValueError, OSError) as exc:
            returncode = self._shutdown_resident()
            raise RuntimeError(
                f"Step '{step_name}': the resident worker in '{self.python_bin}' was gone "
                f"before its job could be sent (exit {returncode}). {exc}"
            ) from exc

        # readline() rather than `for line in stdout`: this loop has to stop
        # at the status marker and then hand the *same* stream to the next
        # job, and an explicit readline leaves no question about what the
        # iterator did or didn't buffer past the break.
        status = None
        while True:
            line = process.stdout.readline()
            if not line:
                break  # EOF: the child is gone, see below
            if line.startswith(SERVE_MARKER):
                status = json.loads(line[len(SERVE_MARKER):].strip() or "{}")
                break
            tail.append(line)
            relay.info("%s", line.rstrip())

        if status is None:
            # EOF with no status line: the child died mid-job — OOM killer,
            # a segfault in a CUDA kernel, `sys.exit` somewhere in a step.
            # Reap it and say so, rather than blocking forever on a pipe
            # nobody will write to again.
            returncode = self._shutdown_resident(drain=False)
            raise RuntimeError(
                f"Step '{step_name}': the resident worker in '{self.python_bin}' died "
                f"mid-job without reporting (exit {returncode}). This usually means it "
                f"was killed (OOM) rather than raising.\n"
                f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
            )

        if not status.get("ok"):
            self._shutdown_resident()
            raise RuntimeError(
                f"Step '{step_name}' failed in the resident worker for isolated env "
                f"'{self.python_bin}': {status.get('error', 'no error reported')}\n"
                f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
            )

        return tail

    def _shutdown_resident(self, drain: bool = True) -> Optional[int]:
        """Stop the resident child and return its exit code (None if none ran).

        A "shutdown" request first, then stdin closed. Either alone would
        work — worker.serve()'s readline returns '' at EOF and takes the
        same exit — but sending it means the child's log says *why* it
        stopped, which is the difference between "the run ended" and "the
        parent died" when reading a pod's log after the fact. Shutdown is
        the full eviction: unload(), instance dropped, host RAM freed.

        Draining stdout to EOF before wait() is not optional — the child
        logs while unloading, and a parent that stopped reading would leave
        it blocked on a full pipe until the timeout below turned a clean
        shutdown into a SIGTERM.
        """
        process, self._resident = self._resident, None
        if process is None:
            return None

        if process.stdin is not None and not process.stdin.closed:
            if drain and process.poll() is None:
                try:
                    self._send(process, {"kind": "shutdown"})
                except (OSError, ValueError):
                    # Already gone (it exits by itself after a failed job,
                    # so this races on every failure path). Closing stdin
                    # below is the fallback and needs no cooperation.
                    pass
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

        relay = logging.getLogger("step.worker")
        try:
            if process.stdout is not None and not process.stdout.closed:
                if drain:
                    for line in process.stdout:
                        relay.info("%s", line.rstrip())
                process.stdout.close()
        except OSError:
            pass

        try:
            process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning(
                "resident worker pid %d did not exit within %.0fs; terminating",
                process.pid, _SHUTDOWN_TIMEOUT_S,
            )
            process.terminate()
            try:
                process.wait(timeout=_KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return process.returncode

    def close(self) -> None:
        # Called from WorkflowRunner's `finally`, including on the way out
        # of a failed run — so it must never raise, or it replaces the
        # exception that explains what actually went wrong.
        try:
            code = self._shutdown_resident()
        except Exception:
            logger.exception("shutting down the resident worker failed; continuing")
            return
        if code is not None:
            logger.info("resident worker for %s exited (code %s)", self.python_bin, code)
