"""Stub steps that report on their own residency, for the subprocess tests.

Kept in its own module rather than inside a test file because the point is
that a *different process* imports it: `pipeline.worker` picks it up via
`B2C_EXTRA_STEP_MODULES=tests.resident_stubs`, registers the steps below,
and then dispatches them exactly the way it dispatches `wan22_vace_denoise`.
Testing residency any other way (mocking Popen, calling the worker's
functions directly) would test the mock, not the pipe.

Stdlib only, and no GPU, network or model weights — these run in whatever
interpreter the test points `python_bin` at, which is the test runner's own.

The counters are CLASS attributes, not instance ones, because that is the
whole measurement: a resident worker builds one instance and reuses it, a
reloading one builds another instance in the same process, and a one-shot
worker starts a whole new process. `loads` distinguishes the first two and
`pid` distinguishes the third.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict

from pipeline.registry import register_step
from pipeline.step import Step


def _say(message: str) -> None:
    """Write a line the parent's relay will pick up, right now.

    `flush=True` matters: the worker's stdout is a pipe, so Python
    block-buffers it, and an unflushed print would arrive in a lump at exit
    — which is precisely the pre-streaming behaviour these tests exist to
    catch.
    """
    print(message, flush=True)


class _ProbeBase(Step):
    loads = 0
    runs = 0
    releases = 0

    def load(self, params: Dict[str, Any]) -> None:
        type(self).loads += 1
        # Stands in for the weights: something attached to the *instance*,
        # so a test can tell "same object, still holding its state" (a
        # partial unload) from "rebuilt" (a full one).
        self.weights = f"weights-for-{params.get('model')}"
        self.on_gpu = True
        _say(f"probe load #{type(self).loads} model={params.get('model')}")

    def release_vram(self) -> None:
        # The partial eviction: the card goes back, `self.weights` — the
        # expensive thing that came off the network volume — does not.
        type(self).releases += 1
        self.on_gpu = False
        _say(f"probe release_vram #{type(self).releases}")

    def unload(self) -> None:
        # Recorded to a file because the only observer of an unload during
        # shutdown is a process that is about to exit; a return value or a
        # log line the parent may or may not still be reading proves less.
        path = os.environ.get("B2C_STUB_UNLOAD_LOG")
        if path:
            with open(path, "a") as f:
                f.write(f"{type(self).__name__} {os.getpid()}\n")
        _say("probe unload")

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        type(self).runs += 1
        mode = params.get("mode", "ok")
        # Observed at the top of run(), i.e. what the *previous* job left
        # behind: a resident worker must arrive here with the weights still
        # in hand but the GPU already released.
        was_on_gpu = getattr(self, "on_gpu", None)
        self.on_gpu = True  # a real step would re-upload here
        _say(f"probe run mode={mode} model={params.get('model')}")

        if mode == "raise":
            _say("SENTINEL_LAST_WORDS before the raise")
            raise RuntimeError("the probe step was told to fail")

        if mode == "die":
            # Hard exit with no unwinding: what an OOM kill or a segfault
            # in a CUDA kernel looks like from the parent — a pipe that
            # goes to EOF with no status line.
            _say("SENTINEL_LAST_WORDS before the hard exit")
            sys.stdout.flush()
            os._exit(9)

        if mode == "slow":
            # Halfway marker, then a pause, so a test can prove the line
            # reached the parent's log *before* run() returned rather than
            # being dumped afterwards.
            _say("SENTINEL_HALFWAY")
            time.sleep(float(params.get("pause", 0.5)))

        return {
            "loads": type(self).loads,
            "runs": type(self).runs,
            "releases": type(self).releases,
            "pid": os.getpid(),
            "model": params.get("model"),
            "instance": id(self),
            "weights": getattr(self, "weights", None),
            "was_on_gpu": was_on_gpu,
        }


@register_step("_resident_probe")
class ResidentProbeStep(_ProbeBase):
    """Declares which params its load() reads, so the reload rule applies."""

    LOAD_PARAMS = ("model",)


@register_step("_resident_probe_undeclared")
class UndeclaredProbeStep(_ProbeBase):
    """Declares nothing — the default `keep_loaded` contract: reuse always."""
