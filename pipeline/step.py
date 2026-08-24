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

    def release_vram(self) -> None:
        """Optional: give the GPU back but KEEP the weights in host RAM.

        The middle ground between "still loaded" and `unload()`, and the
        only reason `keep_loaded` is safe on a shared card. Two evictions,
        two costs:

          * release_vram() — VRAM freed, weights stay in DRAM. The next
            run() re-uploads over PCIe: seconds.
          * unload() — both freed. The next run() re-reads the checkpoint:
            on a RunPod network volume, ~47 GB and tens of minutes for
            wan22_vace_denoise.

        `keep_loaded` exists to skip the *network* read, not to squat on
        the card. fast_helical_full.yaml runs `brush` — a GPU program —
        between its two `wan22_vace_denoise` passes, so a resident worker
        that held ~35 GB of Wan experts in VRAM across that gap would OOM
        brush on any card in existence: strictly worse than the
        reload-every-time behaviour it replaced. So a resident worker calls
        this after every job (see pipeline/worker.py's serve loop) and
        calls unload() only at the end of the run.

        Default is a no-op, which is correct for every step that holds no
        GPU state — most of them — and merely leaves the old behaviour for
        one that does. Override it if this step keeps torch modules alive
        between run() calls.

        Implementing it: move the modules to CPU, then
        `torch.cuda.empty_cache()` — the caching allocator does not hand
        memory back to the driver just because a tensor moved, so without
        the second half the first half buys nothing (see
        dispatch/in_process.py for where that was learned).

        **If the step used accelerate's offload hooks, do not `.to("cpu")`
        it.** `enable_model_cpu_offload()` installs hooks that move modules
        on and off the card per forward pass and track where each one
        belongs; a manual `.to()` underneath them desynchronises that
        bookkeeping. diffusers exposes `pipe.maybe_free_model_hooks()` for
        exactly this — it returns everything to CPU and leaves the hooks
        able to bring it back. A plain `pipe.to(device)` load has no hooks,
        so it *does* need the manual `.to("cpu")`, and its run() then has
        to put the pipeline back on the device before using it. Which of
        the two applies is step-specific knowledge, which is why this hook
        is here rather than being guessed at by the dispatcher.

        Must be idempotent: it is called after every job, including one
        that changed nothing, and may be called when nothing is loaded.
        """
