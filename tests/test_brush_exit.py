"""brush's non-zero exits are judged against the export, not on their own.

brush has been seen taking SIGSEGV (exit -11) on shutdown, after the .ply
is already written and complete. Failing the whole run there throws away a
finished 30,000-iteration training, so the step accepts a failed exit that
left a real export behind — and only that. These drive `_run_brush`
directly with a stand-in binary, since the surrounding `run()` needs a
COLMAP export and a frame batch to say anything about exit codes.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.proc import ProcessFailed
from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401


def _fake_brush(ply_path: Path, *, write: str | None, exit_code: int = 0,
                segfault: bool = False):
    """argv for a process that optionally writes `ply_path`, then exits.

    `segfault=True` reproduces the case this whole mechanism exists for:
    the export is complete and the process then dies on SIGSEGV, which
    Popen reports as returncode -11 rather than as any exit status.
    """
    script = ""
    if write is not None:
        script += f"open({str(ply_path)!r}, 'w').write({write!r});"
    if segfault:
        script += "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"
    else:
        script += f"import sys; sys.exit({exit_code})"
    return [sys.executable, "-c", script]


class TestBrushExitCode(unittest.TestCase):
    def setUp(self):
        self.step = get_step_class("brush")()
        self.tmp = tempfile.TemporaryDirectory(prefix="b2c_brush_test_")
        self.addCleanup(self.tmp.cleanup)
        self.ply = Path(self.tmp.name) / "scene.ply"

    def test_a_failed_exit_with_a_written_export_is_accepted(self):
        with self.assertLogs("pipeline.steps.brush", level=logging.WARNING) as caught:
            self.step._run_brush(_fake_brush(self.ply, write="ply", exit_code=1), self.ply)
        self.assertIn(str(self.ply), "\n".join(caught.output))

    def test_a_clean_exit_says_nothing(self):
        with self.assertNoLogs("pipeline.steps.brush", level=logging.WARNING):
            self.step._run_brush(_fake_brush(self.ply, write="ply", exit_code=0), self.ply)

    def test_a_failed_exit_with_no_export_still_raises(self):
        with self.assertRaises(ProcessFailed):
            self.step._run_brush(_fake_brush(self.ply, write=None, exit_code=1), self.ply)

    def test_a_failed_exit_with_an_empty_export_still_raises(self):
        """A zero-byte .ply is a crash mid-write, not a finished training."""
        with self.assertRaises(ProcessFailed):
            self.step._run_brush(_fake_brush(self.ply, write="", exit_code=1), self.ply)

    def test_a_previous_run_s_export_cannot_stand_in(self):
        """`export_dir` is reused across runs (${output_root}/ply), so the
        .ply already sitting there belongs to the last training. Accepting
        it would hand back a stale splat as if this run had produced it."""
        self.ply.write_text("a previous training")
        with self.assertRaises(ProcessFailed):
            self.step._run_brush(_fake_brush(self.ply, write=None, exit_code=1), self.ply)
        self.assertEqual(self.ply.read_text(), "a previous training")

    def test_an_export_overwritten_by_this_run_is_accepted(self):
        """And the real shape of the failure: exported, then SIGSEGV."""
        self.ply.write_text("a previous training")
        with self.assertLogs("pipeline.steps.brush", level=logging.WARNING) as caught:
            self.step._run_brush(
                _fake_brush(self.ply, write="a new one", segfault=True), self.ply
            )
        self.assertEqual(self.ply.read_text(), "a new one")
        self.assertIn("-11", "\n".join(caught.output))

    def test_a_missing_binary_is_still_a_hard_failure(self):
        """Not a ProcessFailed at all — argv[0] never started, so there is
        no exit code to weigh against anything."""
        with self.assertRaises(RuntimeError) as caught:
            self.step._run_brush(["b2c-no-such-binary-exists"], self.ply)
        self.assertNotIsInstance(caught.exception, ProcessFailed)


if __name__ == "__main__":
    unittest.main()
