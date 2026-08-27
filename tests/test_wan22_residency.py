"""`wan22_vace_denoise`'s DRAM/VRAM eviction, which the resident worker drives.

These are the hooks that make `keep_loaded: true` safe on this step, and
they are easy to get wrong in a way nothing catches until a pod run OOMs:
`fast_helical_full` trains a Gaussian splat with `brush` — on the GPU —
between the two denoise passes, so the pipeline must be off the card by the
time `run()` returns while its ~47 GB of weights stay in host RAM.

No torch and no diffusers here (neither is installed outside venv_wan22), so
the pipe and torch are stubs. That is enough: what is being asserted is
*which* call the step makes for each placement, and picking the wrong one is
exactly the bug — `.to("cpu")` on an accelerate-offloaded pipeline moves the
modules out from under its hooks and desyncs them.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


class _FakePipe:
    def __init__(self):
        self.freed_hooks = 0
        self.moved_to = []

    def maybe_free_model_hooks(self):
        self.freed_hooks += 1

    def to(self, device):
        self.moved_to.append(str(device))
        return self


def _fake_torch(cuda_available: bool = True):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=lambda: torch.cuda.__dict__.__setitem__(
            "empty_cache_calls", torch.cuda.empty_cache_calls + 1
        ),
    )
    torch.cuda.empty_cache_calls = 0
    return torch


def _step():
    from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

    return Wan22VaceDenoiseStep()


class TestReleaseVram(unittest.TestCase):
    def test_offloaded_pipeline_uses_diffusers_own_hook_not_to_cpu(self):
        """With cpu_offload (the default), accelerate owns placement.

        `maybe_free_model_hooks()` offloads every component AND re-applies
        the hooks, so the pipe is left ready for the next call. Moving it
        with `.to("cpu")` instead would desync those hooks.
        """
        step = _step()
        pipe = _FakePipe()
        step._pipe = pipe
        step._cpu_offload = True
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            step.release_vram()
        self.assertEqual(pipe.freed_hooks, 1)
        self.assertEqual(pipe.moved_to, [], "must not .to() an offloaded pipeline")

    def test_plain_device_placement_moves_to_cpu(self):
        step = _step()
        pipe = _FakePipe()
        step._pipe = pipe
        step._cpu_offload = False
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            step.release_vram()
        self.assertEqual(pipe.moved_to, ["cpu"])
        self.assertEqual(pipe.freed_hooks, 0, "no hooks exist on this placement")

    def test_is_a_no_op_before_load(self):
        """The worker may release after a job that failed during load()."""
        step = _step()
        self.assertIsNone(step._pipe)
        step.release_vram()  # must not raise


class TestLoadParams(unittest.TestCase):
    def test_per_call_params_are_absent_so_the_two_passes_share_a_worker(self):
        """The whole point: pass1 and pass2 differ only by `strength`.

        If `strength` were listed, the resident worker would rebuild the
        pipeline between the two passes and `keep_loaded` would buy nothing.
        """
        from pipeline.steps.wan22_vace_denoise import Wan22VaceDenoiseStep

        load_params = set(Wan22VaceDenoiseStep.LOAD_PARAMS)
        for per_call in (
            "strength", "steps", "cfg", "seed", "prompt",
            "negative_prompt", "width", "height", "subject_desc",
        ):
            self.assertNotIn(per_call, load_params, f"{per_call} is a per-call param")

    def test_every_declared_param_is_one_load_actually_reads(self):
        """Guards the other direction: a stale name silently over-reloads.

        Checked against the step's `PARAMS` declaration rather than against
        an AST walk for `params.get("x")` — a name in LOAD_PARAMS that the
        step does not declare is now a genuine bug in its own right, since
        `load_signature` would compare `None` against `None` on every job
        and never notice the param it was asked to watch.
        """
        from pipeline.steps import wan22_vace_denoise as mod

        declared = set(mod.Wan22VaceDenoiseStep.declared_params())
        missing = set(mod.Wan22VaceDenoiseStep.LOAD_PARAMS) - declared
        self.assertEqual(
            missing, set(),
            f"LOAD_PARAMS names params the step does not declare: {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main()
