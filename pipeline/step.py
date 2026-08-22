"""Step interface — the unit of work a workflow YAML wires together.

A Step knows how to do one thing (denoise a video, detect landmarks, export
COLMAP). It does NOT know or care whether it's about to be called directly in
this process, spawned in an isolated venv's subprocess, or hit over HTTP on a
warm model server — that's entirely the Dispatcher's problem. This split is
what makes "run this step in isolation instead" a one-line YAML edit instead
of a code change.

Subclass this, decorate with @register_step("name"), and reference "name"
from a workflow YAML's `step:` field.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Step(ABC):
    @abstractmethod
    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step. Must be side-effect-free w.r.t. disk unless the
        step's whole purpose is I/O (e.g. SaveDatasetStep)."""
        raise NotImplementedError

    def load(self, params: Dict[str, Any]) -> None:
        """Optional: acquire expensive state (load model weights onto GPU).

        Only meaningful for dispatchers that keep a Step instance alive
        across multiple `run()` calls (e.g. ServiceDispatcher). Dispatchers
        that instantiate fresh per call can skip calling this.
        """

    def unload(self) -> None:
        """Optional: release state acquired in load() (free VRAM)."""
