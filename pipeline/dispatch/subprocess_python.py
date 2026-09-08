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
`pipeline/workflows/fast_helical_native.yaml`, which calls
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
steps between fast_helical_native's two denoise passes include `brush`,
which trains a Gaussian splat on the same GPU, and a worker sitting on ~35
GB of Wan experts would OOM it — a regression over the reloading this
replaces, not an improvement. Skipping the network read is the win;
squatting on VRAM was never part of it.

**A failed child fails its step, with one TEMPORARY exception.** See the
block above `_payload_complete`: a one-shot child that died on a signal
*after* writing a complete-looking payload has that payload accepted, and
says so loudly, because seedvr2 has been observed crashing on its way out
of a successful upscale.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import signal
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

# What a negative return code means, in the words of the thing that most
# often produces it. `Popen.returncode` is -N when the child died on signal
# N, and "exit -9" on its own has sent more than one debugging session
# looking for a Python bug that was never there.
_SIGNAL_HINTS = {
    "SIGKILL": (
        "no traceback is possible — nothing in the child ran after this. "
        "Almost always the host OOM killer: check `dmesg -T | grep -i -E "
        "'oom|killed process'` and what else was resident at the time"
    ),
    "SIGSEGV": "a native crash inside the child (torch/CUDA), not a Python exception",
    "SIGBUS": "a native crash inside the child (torch/CUDA), not a Python exception",
    "SIGABRT": "the child aborted — usually a C++ exception escaping a torch/CUDA call",
}


def _exit_description(returncode: Optional[int]) -> str:
    """How the child ended, named rather than numbered.

    A death by signal also explains a line that turns up at the very end of
    such a child's output and reads like the cause:

        UserWarning: resource_tracker: There appear to be 1 leaked
        semaphore objects to clean up at shutdown

    `multiprocessing.resource_tracker` is a *separate* helper process, forked
    the first time anything in the child makes a semaphore or a shared-memory
    block (seedvr2's `inference_cli` sets the spawn start method at import;
    torch's DataLoader and shared tensors do the same). It holds a pipe to
    the child, and stays quiet when the child exits normally, because a
    normal exit unregisters everything first. It reports leaks only when that
    pipe closed with objects still registered — i.e. when the child died
    without running a single line of cleanup. So the warning is written by a
    process that outlived the crash, and it is a symptom of the signal named
    below, never the cause of it.
    """
    if returncode is None:
        return "never started"
    if returncode >= 0:
        return f"exit {returncode}"
    number = -returncode
    try:
        name = signal.Signals(number).name
    except ValueError:
        return f"killed by signal {number}"
    hint = _SIGNAL_HINTS.get(name)
    return f"killed by {name} ({number})" + (f" — {hint}" if hint else "")


# ---------------------------------------------------------------------------
# TEMPORARY (added 2026-09-08) — remove once the seedvr2 teardown crash is
# understood.
#
# `seedvr2` has been seen dying on a signal *after* finishing its upscale and
# writing its output pickle: the log shows the step's own "finished in Ns"
# line and its output summary, and only then the child disappears somewhere
# in interpreter teardown. The run is thrown away at that point, which costs
# a whole pipeline for a process that had already done its work.
#
# So: a child that died by signal, but left behind a payload that loads and
# looks complete, has its payload accepted and the run continues. The check
# below is deliberately shallow — the frames are all there and none of them
# is empty — not a claim that the output is *correct*. Everything else still
# fails the way it did: a Python-level failure (a positive exit code) is
# never salvaged, a missing or truncated pickle is never salvaged, and both
# say so.
#
# When the teardown crash is fixed, delete `_payload_complete`,
# `_salvage_after_signal` and their call site in `run()`.


def _payload_complete(outputs: Dict[str, Any], inputs: Dict[str, Any]) -> tuple[bool, str]:
    """Shallow "did it write everything" check on a salvaged payload.

    Only list-valued outputs are examined, which for the step this exists
    for is the frames and their cameras: each must be non-empty, must not
    have lost entries against the same-named input list, and must contain no
    zero-size array. Returns (ok, reason-it-is-not).
    """
    for name, value in outputs.items():
        if not isinstance(value, list):
            continue
        if not value:
            return False, f"output '{name}' is empty"
        expected = inputs.get(name)
        if isinstance(expected, list) and len(value) != len(expected):
            return False, (
                f"output '{name}' has {len(value)} entries, not the "
                f"{len(expected)} that went in"
            )
        for index, item in enumerate(value):
            size = getattr(item, "size", None)
            if size is not None and size == 0:
                return False, f"output '{name}'[{index}] is empty"
    return True, ""


def _salvage_after_signal(
    step_name: str,
    returncode: int,
    output_path: Path,
    inputs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """The payload of a child that died on its way out, or None to raise.

    See the TEMPORARY block above.
    """
    if returncode >= 0:
        return None  # raised, exited, or SystemExit — not a teardown crash
    if not output_path.exists():
        return None

    try:
        with open(output_path, "rb") as f:
            outputs = pickle.load(f)
    except Exception as exc:  # noqa: BLE001 - anything here means "not salvageable"
        logger.warning(
            "%s: %s, and the output pickle it left behind will not load (%s)",
            step_name, _exit_description(returncode), exc,
        )
        return None

    if not isinstance(outputs, dict) or not outputs:
        return None

    complete, reason = _payload_complete(outputs, inputs)
    if not complete:
        logger.warning(
            "%s: %s, and the output pickle it left behind is incomplete (%s)",
            step_name, _exit_description(returncode), reason,
        )
        return None

    logger.warning(
        "%s: the child was %s AFTER writing what looks like a complete payload "
        "(%s). TEMPORARY: accepting it and continuing the run rather than "
        "throwing the whole run away. This is not a clean step — the crash is "
        "real and still needs fixing; see the TEMPORARY block in %s.",
        step_name,
        _exit_description(returncode),
        ", ".join(
            f"{name}: {len(value)}" if isinstance(value, list) else name
            for name, value in sorted(outputs.items())
        ),
        __name__,
    )
    return outputs


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
            outputs: Optional[Dict[str, Any]] = None
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
                    # TEMPORARY, see the block above _payload_complete: a
                    # child that crashed after writing a good payload keeps
                    # its run instead of costing one.
                    outputs = _salvage_after_signal(
                        step_name, returncode, output_path, inputs
                    )
                if returncode != 0 and outputs is None:
                    elapsed = time.time() - started
                    raise RuntimeError(
                        f"Step '{step_name}' failed after {elapsed:.1f}s in isolated env "
                        f"'{self.python_bin}' ({_exit_description(returncode)}).\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
                    )
            elapsed = time.time() - started

            if outputs is None:
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
                "%s: resident worker exited between jobs (%s); starting a new one",
                step_name, _exit_description(self._resident.returncode),
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
                f"before its job could be sent ({_exit_description(returncode)}). {exc}"
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
                f"mid-job without reporting ({_exit_description(returncode)}). This "
                f"usually means it was killed rather than raising.\n"
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
            logger.info(
                "resident worker for %s exited (%s)", self.python_bin, _exit_description(code)
            )
