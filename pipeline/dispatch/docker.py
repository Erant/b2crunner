"""Runs a step inside a fresh Docker container, one-shot per call.

Same file-based IPC as SubprocessPythonDispatcher, just crossing a container
boundary instead of a venv boundary — for steps that need OS-level isolation
beyond what a venv gives you (a pinned CUDA/cuDNN userspace, a non-Python
runtime like MediaPipe's bundled TFLite engine or Brush's Rust/wgpu binary
baked into the image). Heavier startup cost than a venv subprocess, so prefer
SubprocessPythonDispatcher unless you actually hit a conflict a venv can't
resolve (e.g. two steps needing different system CUDA toolkits, not just
different Python packages).
"""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from .base import Dispatcher


class DockerDispatcher(Dispatcher):
    def __init__(self, image: str, gpus: str = "all", extra_args: list[str] | None = None):
        self.image = image
        self.gpus = gpus
        self.extra_args = extra_args or []

    def run(self, step_name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"b2c_{step_name}_") as tmp:
            tmp_path = Path(tmp)
            with open(tmp_path / "inputs.pkl", "wb") as f:
                pickle.dump(inputs, f)
            with open(tmp_path / "params.json", "w") as f:
                json.dump(params, f)

            cmd = [
                "docker", "run", "--rm",
                *(["--gpus", self.gpus] if self.gpus else []),
                "-v", f"{tmp_path}:/data",
                *self.extra_args,
                self.image,
                "python", "-m", "pipeline.worker", step_name,
                "/data/inputs.pkl", "/data/params.json", "/data/outputs.pkl",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Step '{step_name}' failed in container '{self.image}':\n"
                    f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
                )

            with open(tmp_path / "outputs.pkl", "rb") as f:
                return pickle.load(f)
