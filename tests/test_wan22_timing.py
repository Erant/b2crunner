"""`wan22_vace_denoise`'s pre-loop phase timers.

The step's two-minute silence between the resident worker's "reusing
loaded" line and the denoise progress bar's "0%" is not a hang: it is
WanVACEPipeline.__call__'s preamble — the T5 text encoder coming back over
PCIe under cpu_offload, 81 frames resized twice, and the VAE encoding the
control video *twice* (VACE splits it into inactive/reactive by the mask).
`_timed_phases` wraps those methods so the log says which one it was.

Instrumentation earns a test for one reason: it monkeypatches the live
pipeline object. If it ever failed to put the pipe back — after an
exception, above all — the wrappers would stack up across a resident
worker's passes, and the thing meant to explain a slow run would be making
it slower. No torch or diffusers here (neither is installed outside
venv_wan22), so the pipe and torch are stubs, same convention as
tests/test_wan22_residency.py.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


class _FakePipe:
    """Two of the real pipeline's pre-loop phases, plus one it doesn't have."""

    def __init__(self):
        self.calls = []

    def encode_prompt(self, prompt, negative_prompt=None):
        self.calls.append(("encode_prompt", prompt, negative_prompt))
        return "embeds"

    def prepare_video_latents(self, video):
        self.calls.append(("prepare_video_latents", video))
        return "latents"


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.synchronized = 0

    def synchronize():
        torch.synchronized += 1

    torch.cuda = types.SimpleNamespace(is_available=lambda: True, synchronize=synchronize)
    return torch


def _timed_phases(*args, **kwargs):
    from pipeline.steps.wan22_vace_denoise import _timed_phases as ctx

    return ctx(*args, **kwargs)


class TestTimedPhases(unittest.TestCase):
    def test_logs_one_line_per_phase_and_passes_calls_through(self):
        pipe = _FakePipe()
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            with self.assertLogs("pipeline.steps.wan22_vace_denoise", "INFO") as logs:
                with _timed_phases(pipe):
                    self.assertEqual(pipe.encode_prompt("a prompt", negative_prompt="no"), "embeds")
                    self.assertEqual(pipe.prepare_video_latents(["frame"]), "latents")

        # Arguments and return values survive the wrapper untouched — this
        # is a measurement, not an interception.
        self.assertEqual(
            pipe.calls,
            [("encode_prompt", "a prompt", "no"), ("prepare_video_latents", ["frame"])],
        )
        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(len(messages), 2)
        self.assertIn("encode_prompt", messages[0])
        self.assertIn("prepare_video_latents", messages[1])

    def test_phases_the_installed_diffusers_does_not_have_are_skipped(self):
        """A renamed phase upstream costs a log line, never a crash."""
        pipe = _FakePipe()
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            with _timed_phases(pipe, ("encode_prompt", "no_such_phase")):
                pipe.encode_prompt("p")
                self.assertFalse(hasattr(pipe, "no_such_phase"))

    def test_the_gpu_is_waited_on_before_the_clock_is_read(self):
        """Otherwise the VAE encode bills its time to the first denoise step.

        CUDA work is queued, not finished, when the submitting call
        returns, so an unsynchronized timer would report
        prepare_video_latents as instant.
        """
        pipe = _FakePipe()
        torch = _fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            with _timed_phases(pipe):
                pipe.encode_prompt("p")
                pipe.prepare_video_latents(["frame"])
        self.assertEqual(torch.synchronized, 2)

    def test_the_pipe_is_left_exactly_as_it_was_found(self):
        pipe = _FakePipe()
        original = pipe.encode_prompt
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            with _timed_phases(pipe):
                self.assertIsNot(pipe.encode_prompt, original)
        self.assertNotIn("encode_prompt", vars(pipe))
        self.assertEqual(pipe.encode_prompt.__func__, original.__func__)

    def test_the_pipe_is_restored_even_when_the_call_raises(self):
        """The case that matters: a failed pass must not leave a wrapper on.

        The resident worker ends the process on a failed job today, but
        that is the worker's policy, not this helper's guarantee.
        """
        pipe = _FakePipe()
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            with self.assertRaises(RuntimeError):
                with _timed_phases(pipe):
                    raise RuntimeError("CUDA out of memory")
        self.assertNotIn("encode_prompt", vars(pipe))

    def test_a_phase_that_raises_is_still_timed(self):
        """Half a log is what tells you *which* phase blew up."""
        pipe = _FakePipe()

        def boom(*_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

        pipe.prepare_video_latents = boom
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            with self.assertLogs("pipeline.steps.wan22_vace_denoise", "INFO") as logs:
                with _timed_phases(pipe):
                    with self.assertRaises(RuntimeError):
                        pipe.prepare_video_latents(["frame"])
        self.assertIn("prepare_video_latents", logs.records[0].getMessage())


class TestPreLoopPhaseNames(unittest.TestCase):
    def test_covers_the_phases_that_can_actually_dominate(self):
        """Guards against someone trimming the list back to nothing useful.

        The expensive two are the text encoder's 11.36 GB round trip
        (encode_prompt) and the double VAE encode of the control video
        (prepare_video_latents); the transformers are deliberately absent
        because under cpu_offload they upload inside the loop, after "0%".
        """
        from pipeline.steps.wan22_vace_denoise import _PRE_LOOP_PHASES

        self.assertIn("encode_prompt", _PRE_LOOP_PHASES)
        self.assertIn("prepare_video_latents", _PRE_LOOP_PHASES)


if __name__ == "__main__":
    unittest.main()
