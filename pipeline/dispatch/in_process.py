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


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        result = step.run(inputs, params)
        if self.keep_loaded:
            # Same contract the resident subprocess worker gives (see
            # pipeline/worker.py's serve loop): a kept-loaded step holds its
            # weights in host RAM between calls, not on the card, so the
            # next GPU step finds an empty one. The base Step.release_vram
            # is a no-op, so this costs nothing for the numpy/cv2 steps
            # that make up most of the in-process set.
            step.release_vram()
            _release_cuda_cache()
        if not self.keep_loaded:
            # A fresh instance per call (the keep_loaded=False default)
            # goes out of scope right after this, but PyTorch's caching
            # allocator doesn't hand memory back to the driver just because
            # Python freed the tensors that held it — the next in-process
            # step's own model load/forward pass sees it as still gone.
            # On a VRAM-constrained box running several GPU steps back to
            # back in one process (this dispatcher's whole point), that is
            # a real OOM, not a theoretical one: caught it directly running
            # rmbg -> sapiens2_lite in sequence on a 12GB card, where
            # sapiens2_lite's attention allocation failed with 3.24GB free
            # against an 11.59GB card despite rmbg's model already being
            # unreachable Python-side.
            step = None
            _release_cuda_cache()
        return result

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
