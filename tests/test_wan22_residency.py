"""`wan22_vace_denoise`'s DRAM/VRAM eviction, which the resident worker drives.

These are the hooks that make `keep_loaded: true` safe on this step, and
they are easy to get wrong in a way nothing catches until a pod run OOMs:
`fast_helical_native` trains a Gaussian splat with `brush` — on the GPU —
between the two denoise passes, so the pipeline must be off the card by the
time `run()` returns while its ~47 GB of weights stay in host RAM.

No torch and no diffusers here (neither is installed outside venv_wan22), so
the pipe and torch are stubs. That is enough: what is being asserted is
*which* call the step makes for each placement, and picking the wrong one is
exactly the bug — `.to("cpu")` on an offloaded pipeline moves the modules out
from under its hooks and desyncs them.
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
    def test_offloaded_pipeline_is_left_alone_and_only_the_cache_is_freed(self):
        """With cpu_offload (the default), group offloading owns placement.

        Each group is already back on the CPU by the end of its own
        post-forward, so the card holds nothing but the caching allocator's
        blocks and there is no placement left to undo. Moving the pipe with
        `.to("cpu")` would desync the offload hooks' bookkeeping, and
        `maybe_free_model_hooks()` — which the model-level offload needed —
        is a no-op here: it returns early unless `enable_model_cpu_offload()`
        set `_all_hooks`.
        """
        step = _step()
        pipe = _FakePipe()
        step._pipe = pipe
        step._cpu_offload = True
        fake_torch = _fake_torch()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            step.release_vram()
        self.assertEqual(pipe.moved_to, [], "must not .to() an offloaded pipeline")
        self.assertEqual(pipe.freed_hooks, 0, "no accelerate hooks on this placement")
        self.assertEqual(fake_torch.cuda.empty_cache_calls, 1)

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


class _FakeModule:
    """Stands in for torch.nn.Module — see _fake_torch_with_nn below."""


class _OffloadPipe:
    """A pipeline whose `components` mixes weights with the things that aren't.

    `tokenizer` and `scheduler` are what WanVACEPipeline actually puts in
    there beside the four models, and they have no `.parameters()` for an
    offload hook to place — walking them would raise rather than quietly do
    nothing, which is why the step filters on nn.Module.
    """

    def __init__(self):
        self.transformer = _FakeModule()
        self.transformer_2 = _FakeModule()
        self.text_encoder = _FakeModule()
        self.vae = _FakeModule()
        self.components = {
            "text_encoder": self.text_encoder,
            "tokenizer": object(),
            "transformer": self.transformer,
            "transformer_2": self.transformer_2,
            "vae": self.vae,
            "scheduler": object(),
        }


def _fake_torch_with_nn():
    torch = _fake_torch()
    torch.nn = types.SimpleNamespace(Module=_FakeModule)
    torch.device = lambda spec: f"device({spec})"
    return torch


def _record_group_offloading():
    """Stub `diffusers.hooks.apply_group_offloading`, capturing every call."""
    calls = []
    diffusers = types.ModuleType("diffusers")
    hooks = types.ModuleType("diffusers.hooks")
    hooks.apply_group_offloading = lambda module, **kwargs: calls.append((module, kwargs))
    diffusers.hooks = hooks
    return calls, {"diffusers": diffusers, "diffusers.hooks": hooks}


class TestGroupOffload(unittest.TestCase):
    """Which modules get placed, and on what terms.

    The failure this guards against is silent in unit tests and fatal on a
    pod: diffusers refuses to mix group offloading with the pipeline-level
    `enable_model_cpu_offload()`, so every component has to be placed by
    this one loop. One left out stays on the CPU — where `from_pretrained`
    put it — and dies on its first forward, several minutes into a run.
    """

    def _apply(self, blocks_per_group=1):
        step = _step()
        step._blocks_per_group = blocks_per_group
        pipe = _OffloadPipe()
        calls, modules = _record_group_offloading()
        modules["torch"] = _fake_torch_with_nn()
        with patch.dict(sys.modules, modules):
            step._apply_group_offload(pipe, "cuda")
        return pipe, calls

    def test_every_weight_bearing_component_is_placed(self):
        pipe, calls = self._apply()
        placed = {id(module) for module, _ in calls}
        self.assertEqual(
            placed,
            {id(m) for m in (pipe.text_encoder, pipe.transformer,
                             pipe.transformer_2, pipe.vae)},
            "a component left unplaced stays on the CPU and fails mid-run",
        )

    def test_one_block_per_group_streams_so_the_transfer_can_hide(self):
        _, calls = self._apply(blocks_per_group=1)
        for _, kwargs in calls:
            self.assertEqual(kwargs["num_blocks_per_group"], 1)
            self.assertTrue(kwargs["use_stream"])
            self.assertEqual(kwargs["offload_type"], "block_level")
            self.assertEqual(kwargs["onload_device"], "device(cuda)")
            self.assertEqual(kwargs["offload_device"], "device(cpu)")

    def test_bigger_groups_drop_the_stream_rather_than_be_overridden(self):
        """diffusers forces num_blocks_per_group back to 1 under a stream.

        `_apply_group_offloading_block_level` does it with a warning, not an
        error, so asking for both would silently get neither: the group size
        would be ignored and the log line lost among the others. The step
        picks one, and >1 means the caller wants blocks resident.
        """
        _, calls = self._apply(blocks_per_group=8)
        for _, kwargs in calls:
            self.assertEqual(kwargs["num_blocks_per_group"], 8)
            self.assertFalse(kwargs["use_stream"])

    def test_weights_are_pinned_on_the_fly_not_up_front(self):
        """Pre-pinning would ask the host for a second copy of ~47 GB."""
        _, calls = self._apply()
        for _, kwargs in calls:
            self.assertTrue(kwargs["low_cpu_mem_usage"])
