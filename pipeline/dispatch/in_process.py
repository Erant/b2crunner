"""Runs a step as a plain Python call in the current process/venv.

Use this for steps with no conflicting dependencies: dataset I/O, camera path
generation, COLMAP export, rendering (pyrender/numpy), gsplat (has clean
wheels). Cheapest and fastest dispatcher — no IPC, no serialization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..registry import get_step_class
from ..step import Step
from .base import Dispatcher


class InProcessDispatcher(Dispatcher):
    def __init__(self, keep_loaded: bool = False):
        # keep_loaded=True reuses one Step instance (and its load()ed state,
        # e.g. GPU weights) across calls instead of instantiating fresh each
        # time. Off by default: most in-process steps are cheap, stateless
        # numpy/cv2 work where a fresh instance is simpler and safer.
        self.keep_loaded = keep_loaded
        self._instances: Dict[str, Step] = {}

    def run(self, step_name: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        step = self._get_instance(step_name, params)
        return step.run(inputs, params)

    def _get_instance(self, step_name: str, params: Dict[str, Any]) -> Step:
        if self.keep_loaded:
            instance = self._instances.get(step_name)
            if instance is None:
                instance = get_step_class(step_name)()
                instance.load(params)
                self._instances[step_name] = instance
            return instance
        return get_step_class(step_name)()

    def close(self) -> None:
        for instance in self._instances.values():
            instance.unload()
        self._instances.clear()
