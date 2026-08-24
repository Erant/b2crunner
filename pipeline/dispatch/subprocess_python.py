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
from typing import Any, Dict

from .base import Dispatcher

logger = logging.getLogger(__name__)

# How many lines of child output to quote in the exception when a step
# fails. The full output is already in the log file by then; this is just
# enough to make the traceback self-contained.
_ERROR_TAIL_LINES = 60


class SubprocessPythonDispatcher(Dispatcher):
    def __init__(self, python_bin: str, cwd: str | None = None, env: Dict[str, str] | None = None):
        """python_bin: path to the isolated venv's interpreter, e.g.
        'envs/sam3dbody/bin/python'."""
        self.python_bin = python_bin
        self.cwd = cwd
        self.env = env

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
                "%s: dispatching to %s (inputs %.1f MB via %s)",
                step_name,
                self.python_bin,
                input_path.stat().st_size / 1e6,
                tmp_path,
            )

            cmd = [
                self.python_bin, "-m", "pipeline.worker",
                step_name, str(input_path), str(params_path), str(output_path),
            ]
            started = time.time()
            returncode, tail = self._stream(cmd, step_name)
            elapsed = time.time() - started

            if returncode != 0:
                raise RuntimeError(
                    f"Step '{step_name}' failed after {elapsed:.1f}s in isolated env "
                    f"'{self.python_bin}' (exit {returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
                )

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
            raise RuntimeError(
                f"Interpreter not found for step '{step_name}': {cmd[0]}\n"
                f"Check the 'python_bin' for this step's env in the envs registry "
                f"(pipeline/envs/envs.yaml, or docker/envs.docker.yaml inside the image)."
            ) from None

        # `with process` closes the stdout pipe; a bare wait() leaks one
        # file descriptor per dispatched step.
        assert process.stdout is not None
        with process:
            for line in process.stdout:
                tail.append(line)
                relay.info("%s", line.rstrip())

        return process.returncode, tail
