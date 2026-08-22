"""Runs a step against a long-lived HTTP microservice.

Use when a model is expensive enough to load that you want it to stay
resident across many workflow runs (e.g. batch-processing many datasets
through the same Wan2.2 checkpoint) instead of paying the load cost every
single invocation the way SubprocessPythonDispatcher does. The service side
is expected to be a small FastAPI app exposing /load, /infer, /unload,
/health, running the same Step class via its own copy of this `pipeline`
package.

Not needed for a single interactive research pass over one dataset —
SubprocessPythonDispatcher already isolates deps fine there. Reach for this
once "reload the model every step" becomes the actual bottleneck.
"""

from __future__ import annotations

from typing import Any, Dict

import requests

from .base import Dispatcher


class ServiceDispatcher(Dispatcher):
    def __init__(self, base_url: str, timeout: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def run(self, step_name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # NOTE: inputs containing large numpy arrays should go over a format
        # more efficient than JSON in a real implementation (e.g. msgpack +
        # shared temp-file paths, mirroring SubprocessPythonDispatcher's
        # pickle files but over a mounted volume). Left as JSON here for the
        # scaffold; tighten this when a real service backend is built.
        response = self._session.post(
            f"{self.base_url}/infer",
            json={"step": step_name, "inputs": inputs, "params": params},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["outputs"]

    def close(self) -> None:
        self._session.close()
