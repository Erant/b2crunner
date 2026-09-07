"""`wan22_vace_denoise`'s DRAM/VRAM eviction, which the resident worker drives.

These are the hooks that make `keep_loaded: true` safe on this step, and
they are easy to get wrong in a way nothing catches until a pod run OOMs:
`fast_helical_native` trains a Gaussian splat with `brush` — on the GPU —
between the two denoise passes, so the pipeline must be off the card by the
time `run()` returns while its ~47 GB of weights stay in host RAM.

Most of this stubs torch and diffusers, because neither is installed outside
venv_wan22, and for most of it that is enough: what is being asserted is
*which* call the step makes for each placement, and picking the wrong one is
exactly the bug — `.to("cpu")` on an offloaded pipeline moves the modules out
from under its hooks and desyncs them.

TestGroupOffloadFreesFp8Weights at the bottom is the exception and has to
be. Its subject is what diffusers does to a torchao `Float8Tensor`, which
no stub can model, and skipping it is how `6597daa` shipped an offload that
freed nothing: that change was verified against a stand-in with the real
geometry but ordinary weights, so the branch it depended on never ran. It
skips unless a real torch, torchao, diffusers and CUDA card are all present
— i.e. it runs in-image on a GPU box, which is the only place it means
anything.
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

    def test_one_block_per_group_is_the_least_resident_placement(self):
        _, calls = self._apply(blocks_per_group=1)
        for _, kwargs in calls:
            self.assertEqual(kwargs["num_blocks_per_group"], 1)
            self.assertEqual(kwargs["offload_type"], "block_level")
            self.assertEqual(kwargs["onload_device"], "device(cuda)")
            self.assertEqual(kwargs["offload_device"], "device(cpu)")

    def test_the_stream_is_off_at_every_group_size(self):
        """Streaming offloads NOTHING for the fp8 weights this step loads.

        wan_fp8.py builds torchao `Float8Tensor` params, and diffusers'
        streamed path restores them from a `cpu_param_dict` the onload's
        `swap_tensors` has already replaced with device data — so
        `ModuleGroup.offload_` runs `cuda:0 -> cuda:0` and the whole
        expert stays resident. It shipped that way in `6597daa` and the
        5090 OOMed again with the offload active.

        Not a tuning choice and not a size to avoid: it cannot complete a
        run on any card, because the same corrupted cache makes the second
        forward raise `cannot pin 'CUDAFloat8_e4m3fnType'`. The group size
        is now free to mean only what it says, so assert the stream is off
        on both sides of the old branch.
        """
        for blocks in (1, 8):
            _, calls = self._apply(blocks_per_group=blocks)
            self.assertTrue(calls)
            for _, kwargs in calls:
                self.assertEqual(kwargs["num_blocks_per_group"], blocks)
                self.assertFalse(
                    kwargs["use_stream"],
                    "a stream silently defeats the offload for torchao weights",
                )

    def test_the_host_is_never_asked_for_a_second_copy_of_the_weights(self):
        """~47 GB is already resident in DRAM; a pinned duplicate is fatal.

        Kept passing `low_cpu_mem_usage=True` even though it is inert with
        the stream off — diffusers only builds the pinned `cpu_param_dict`
        the flag guards inside `if self.stream is not None`, so unstreamed
        onloads copy straight from the module's own pageable storage and
        cache nothing. It stays because it is the flag that would matter
        again the moment anything reintroduces a stream, and because the
        assertion states the constraint whether or not today's code path
        happens to consult it.
        """
        _, calls = self._apply()
        for _, kwargs in calls:
            self.assertTrue(kwargs["low_cpu_mem_usage"])


def _real_stack_or_skip():
    """torch + torchao + diffusers + a CUDA card, or None.

    Everything above this point stubs torch, because neither torch nor
    diffusers is installed outside venv_wan22. This one cannot: what it
    asserts is what the real libraries do to a real tensor subclass, which
    is exactly the part a stub cannot model and exactly where `6597daa`
    went wrong. It runs in-image on a GPU box and skips everywhere else.
    """
    try:
        import torch
        from torchao.quantization import Float8Tensor
        import diffusers.hooks  # noqa: F401 - the step imports it lazily
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch, Float8Tensor


class TestGroupOffloadFreesFp8Weights(unittest.TestCase):
    """The offload has to move the bytes, not just the wrapper.

    A torchao `Float8Tensor` is a `_make_wrapper_subclass` with no storage
    of its own — every byte lives in its `.qdata`/`.scale` attributes. An
    offload that replaces the wrapper and leaves those behind reports
    success, frees nothing, and shows up hours later as an OOM on a pod.
    Peak allocation does not catch it either: the weights are resident for
    the whole forward, so the peak looks like a normal one. The device of
    `.qdata` after the forward is the assertion that catches it.
    """

    BLOCKS = 4
    DIM = 1024

    def _model(self, torch, Float8Tensor):
        dim = self.DIM

        def fp8_weight():
            qdata = torch.randint(
                0, 200, (dim, dim), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            scale = torch.full((dim, 1), 0.01, dtype=torch.float32)
            # The same construction as pipeline/wan_fp8.py's _build_float8.
            return Float8Tensor(qdata=qdata, scale=scale, block_size=[1, dim],
                                dtype=torch.bfloat16)

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(dim, dim, bias=False,
                                           dtype=torch.bfloat16)
                self.lin.weight = torch.nn.Parameter(fp8_weight(),
                                                     requires_grad=False)

            def forward(self, x):
                return self.lin(x)

        class Model(torch.nn.Module):
            def __init__(self, n):
                super().__init__()
                # A top-level ModuleList is what block_level offloading
                # groups on, and what the real transformer exposes.
                self.blocks = torch.nn.ModuleList([Block() for _ in range(n)])

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        return Model(self.BLOCKS)

    def _place(self, model):
        """Route the model through the step's own placement call.

        Not `apply_group_offloading` directly: the argument this is
        defending is the step's, so the step has to be the one that makes
        the call.
        """
        step = _step()
        step._blocks_per_group = 1
        step._apply_group_offload(
            types.SimpleNamespace(components={"transformer": model}), "cuda"
        )

    def test_every_fp8_weight_is_back_on_the_cpu_after_a_forward(self):
        stack = _real_stack_or_skip()
        if stack is None:
            self.skipTest("needs torch + torchao + diffusers + CUDA")
        torch, Float8Tensor = stack

        model = self._model(torch, Float8Tensor)
        self._place(model)

        x = torch.randn(1, 32, self.DIM, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()

        left = [str(b.lin.weight.qdata.device) for b in model.blocks]
        self.assertEqual(
            left, ["cpu"] * self.BLOCKS,
            "the offload moved the wrapper and left the weight on the card",
        )

    def test_a_second_forward_still_works(self):
        """Denoise step 2. The streamed path raised here, not at step 1."""
        stack = _real_stack_or_skip()
        if stack is None:
            self.skipTest("needs torch + torchao + diffusers + CUDA")
        torch, Float8Tensor = stack

        model = self._model(torch, Float8Tensor)
        self._place(model)

        x = torch.randn(1, 32, self.DIM, dtype=torch.bfloat16, device="cuda")
        for _ in range(2):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        self.assertEqual(
            [str(b.lin.weight.qdata.device) for b in model.blocks],
            ["cpu"] * self.BLOCKS,
        )
