"""Dispatcher interface.

Every dispatcher exposes the same surface: given a registered step name plus
resolved inputs/params, run it and return outputs. The WorkflowRunner never
imports a concrete dispatcher — it only sees this interface, so a workflow
YAML can send any step through any dispatch mechanism without the runner or
the step's own code changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Dispatcher(ABC):
    @abstractmethod
    def run(self, step_name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute `step_name` with `inputs`/`params`, return its outputs dict."""
        raise NotImplementedError

    def close(self) -> None:
        """Release anything held open (subprocess, HTTP session, container).

        Called once when the WorkflowRunner is done with this dispatcher
        instance. Safe to no-op.
        """
