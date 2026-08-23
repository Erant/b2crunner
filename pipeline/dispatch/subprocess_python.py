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
"""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from .base import Dispatcher


class SubprocessPythonDispatcher(Dispatcher):
    def __init__(self, python_bin: str, cwd: str | None = None, env: Dict[str, str] | None = None):
        """python_bin: path to the isolated venv's interpreter, e.g.
        'envs/sam3dbody/bin/python'."""
        self.python_bin = python_bin
        self.cwd = cwd
        self.env = env

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

            result = subprocess.run(
                [self.python_bin, "-m", "pipeline.worker", step_name, str(input_path), str(params_path), str(output_path)],
                cwd=self.cwd,
                env=self.env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Step '{step_name}' failed in isolated env '{self.python_bin}':\n"
                    f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
                )

            with open(output_path, "rb") as f:
                return pickle.load(f)
