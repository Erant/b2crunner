"""TEMPORARY (2026-09-08) — a step that crashes AFTER writing its output.

`seedvr2` has been seen finishing an upscale, logging its "finished in Ns"
line and its output summary, and only then dying on a signal somewhere in
interpreter teardown. Under the old dispatcher that threw the whole run
away, which is a very expensive way to lose work that was already done.

`pipeline/dispatch/subprocess_python.py` now salvages such a payload if it
loads and looks complete. This file, the `crash_on_exit` mode in
tests/resident_stubs.py, and that salvage all go together — delete them
together once the teardown crash itself is fixed.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.dispatch.subprocess_python import (
    SubprocessPythonDispatcher,
    _payload_complete,
    _salvage_after_signal,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dispatcher() -> SubprocessPythonDispatcher:
    python_path = os.pathsep.join(
        part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part
    )
    return SubprocessPythonDispatcher(
        python_bin=sys.executable,
        cwd=str(REPO_ROOT),
        env={
            "B2C_EXTRA_STEP_MODULES": "tests.resident_stubs",
            "PYTHONPATH": python_path,
        },
    )


class TestPayloadCompleteness(unittest.TestCase):
    """The shallow check: everything is there and nothing is empty."""

    def test_a_full_payload_passes(self):
        frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
        ok, reason = _payload_complete({"images": frames}, {"images": frames})
        self.assertTrue(ok, reason)

    def test_a_short_payload_is_rejected_against_its_input(self):
        frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
        ok, reason = _payload_complete({"images": frames[:2]}, {"images": frames})
        self.assertFalse(ok)
        self.assertIn("2 entries", reason)
        self.assertIn("3", reason)

    def test_an_empty_frame_is_rejected(self):
        frames = [np.zeros((4, 4, 3), np.uint8), np.zeros((0, 4, 3), np.uint8)]
        ok, reason = _payload_complete({"images": frames}, {"images": frames})
        self.assertFalse(ok)
        self.assertIn("images", reason)

    def test_an_empty_list_is_rejected(self):
        ok, reason = _payload_complete({"images": []}, {})
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_non_list_outputs_are_not_examined(self):
        """`resolution` and `cameras` shapes must not trip the frame check."""
        ok, _ = _payload_complete({"resolution": (1080, 1920)}, {})
        self.assertTrue(ok)


class TestSalvageGuards(unittest.TestCase):
    """What is NOT salvaged, which is most of it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "outputs.pkl"
        self.addCleanup(self.tmp.cleanup)

    def _write(self, payload) -> None:
        with open(self.output, "wb") as f:
            pickle.dump(payload, f)

    def test_a_python_level_failure_is_never_salvaged(self):
        """A positive exit code means the step raised; a stale pickle from a
        previous call must not be mistaken for a result."""
        self._write({"images": [np.zeros((2, 2, 3), np.uint8)]})
        self.assertIsNone(_salvage_after_signal("probe", 1, self.output, {}))

    def test_no_pickle_is_never_salvaged(self):
        self.assertIsNone(_salvage_after_signal("probe", -9, self.output, {}))

    def test_a_truncated_pickle_is_never_salvaged(self):
        self._write({"images": [np.zeros((2, 2, 3), np.uint8)]})
        data = self.output.read_bytes()
        self.output.write_bytes(data[: len(data) // 2])
        self.assertIsNone(_salvage_after_signal("probe", -9, self.output, {}))

    def test_an_incomplete_payload_is_never_salvaged(self):
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(4)]
        self._write({"images": frames[:1]})
        self.assertIsNone(
            _salvage_after_signal("probe", -9, self.output, {"images": frames})
        )

    def test_a_complete_payload_survives_a_signal(self):
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(4)]
        self._write({"images": frames})
        salvaged = _salvage_after_signal("probe", -9, self.output, {"images": frames})
        self.assertIsNotNone(salvaged)
        self.assertEqual(len(salvaged["images"]), 4)


class TestSalvageEndToEnd(unittest.TestCase):
    """A real child, a real SIGKILL on its way out, a real pickle."""

    def test_the_run_continues_when_the_crash_is_on_the_exit_path(self):
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
        outputs = _dispatcher().run(
            "_resident_probe", {"images": frames}, {"mode": "crash_on_exit"}
        )
        self.assertEqual(len(outputs["images"]), 3)

    def test_the_same_crash_with_frames_missing_still_fails(self):
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
        with self.assertRaises(RuntimeError) as caught:
            _dispatcher().run(
                "_resident_probe",
                {"images": frames},
                {"mode": "crash_on_exit", "keep_images": 1},
            )
        message = str(caught.exception)
        self.assertIn("SIGKILL", message)
        self.assertIn("SENTINEL_LAST_WORDS", message)

    def test_a_step_that_raises_is_untouched_by_any_of_this(self):
        with self.assertRaises(RuntimeError) as caught:
            _dispatcher().run("_resident_probe", {}, {"mode": "raise"})
        self.assertIn("exit 1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
