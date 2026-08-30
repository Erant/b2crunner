"""brush's non-zero exits are judged against the export, not on their own,
and every crash leaves something behind to read.

brush has been seen taking SIGSEGV (exit -11) on shutdown, after the .ply
is already written and complete. Failing the whole run there throws away a
finished 30,000-iteration training, so the step accepts a failed exit that
left a real export behind — and only that. These drive `_run_brush`
directly with a stand-in binary, since the surrounding `run()` needs a
COLMAP export and a frame batch to say anything about exit codes.

The crashlog half is the same mechanism `brush-splat-render` got in
tests/test_render_exit.py, for the same reason: the COLMAP export brush
trains from lives in a `TemporaryDirectory` that `run()` deletes on its way
out, exception or not, so a training that died on a pod left an exit code
and nothing else.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.proc import ProcessFailed
from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401

from tests.helpers import crash_dirs, redirect_crash_dir


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


def _colmap_export(root: Path) -> Path:
    """A stand-in for what ColmapExporter leaves in the temp directory.

    Shape rather than content: the three model .txt files at the root, and
    an images/ directory beside them standing for the several hundred MB
    the crashlog deliberately does not copy.
    """
    colmap_dir = root / "colmap"
    (colmap_dir / "images").mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (colmap_dir / name).write_text(f"# {name}\n")
    for i in range(3):
        (colmap_dir / "images" / f"frame_{i + 1:05d}_.png").write_bytes(b"x" * 16)
    return colmap_dir


class _BrushCase(unittest.TestCase):
    def setUp(self):
        self.step = get_step_class("brush")()
        self.tmp = tempfile.TemporaryDirectory(prefix="b2c_brush_test_")
        self.addCleanup(self.tmp.cleanup)
        self.ply = Path(self.tmp.name) / "scene.ply"
        self.crashes = redirect_crash_dir(self)

    def crash_dirs(self):
        return crash_dirs(self.crashes)


class TestBrushExitCode(_BrushCase):
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
        self.assertEqual(self.crash_dirs(), [], "nothing ran, so there is nothing to save")

    def test_a_clean_exit_that_exported_nothing_is_a_failure_here(self):
        """And it fails here, where the COLMAP export it was training on
        still exists, rather than back in run() one deleted temp directory
        later as 'Expected output PLY file not found'."""
        with self.assertRaises(RuntimeError) as caught:
            self.step._run_brush(_fake_brush(self.ply, write=None), self.ply)
        self.assertNotIsInstance(caught.exception, ProcessFailed)
        self.assertIn("MISSING", str(caught.exception))

    def test_a_clean_exit_that_left_a_previous_export_untouched_is_flagged(self):
        """Not fatal — brush said it succeeded and the only evidence
        against it is an unchanged mtime, which a coarse-timestamp
        filesystem could produce for a real overwrite. Said out loud all
        the same, because the alternative reading is that run() just handed
        back a stale splat."""
        self.ply.write_text("a previous training")
        with self.assertLogs("pipeline.steps.brush", level=logging.WARNING) as caught:
            self.step._run_brush(_fake_brush(self.ply, write=None, exit_code=0), self.ply)
        self.assertIn("same mtime", "\n".join(caught.output))


class TestBrushCrashlog(_BrushCase):
    """What a crashed training leaves behind once the temp directory is gone."""

    def _crash_after(self, cmd, expect_raises=True):
        colmap_dir = _colmap_export(Path(self.tmp.name))
        if expect_raises:
            with self.assertRaises(Exception):
                self.step._run_brush(cmd, self.ply, colmap_dir=colmap_dir)
        else:
            with self.assertLogs("pipeline.steps.brush", level=logging.WARNING):
                self.step._run_brush(cmd, self.ply, colmap_dir=colmap_dir)
        dirs = self.crash_dirs()
        self.assertEqual(len(dirs), 1, f"expected one crash directory, got {dirs}")
        return dirs[0]

    def test_a_hard_failure_saves_the_colmap_model_it_was_training_on(self):
        crash = self._crash_after(_fake_brush(self.ply, write=None, exit_code=1))
        saved = sorted(p.name for p in (crash / "colmap").iterdir())
        self.assertEqual(saved, ["cameras.txt", "images.txt", "points3D.txt"])

    def test_the_frames_are_described_but_not_copied(self):
        """Hundreds of MB, and unlike the model they are the dataset's —
        still on the volume after the temp directory is gone."""
        crash = self._crash_after(_fake_brush(self.ply, write=None, exit_code=1))
        self.assertEqual(
            sorted(p.name for p in crash.rglob("*.png")), [],
            "training frames must not be copied into a crash directory",
        )
        self.assertIn("images/ 3 files", (crash / "report.txt").read_text())

    def test_the_report_names_the_exit_code_the_argv_and_the_export(self):
        crash = self._crash_after(_fake_brush(self.ply, write=None, exit_code=3))
        report = (crash / "report.txt").read_text()
        self.assertIn("exit code 3", report)
        self.assertIn(sys.executable, report)
        self.assertIn("NOT written by this run", report)
        # The environment that decides whether this binary can reach a GPU
        # at all, recorded while the pod still exists to be asked.
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=", report)

    def test_a_tolerated_crash_is_still_saved(self):
        """The guard says the training is usable; it does not say the crash
        is uninteresting. This is the case the pod round needs read."""
        crash = self._crash_after(
            _fake_brush(self.ply, write="ply", segfault=True), expect_raises=False
        )
        report = (crash / "report.txt").read_text()
        self.assertIn("written by this run", report)
        self.assertIn("-11", report)

    def test_a_crashlog_that_cannot_be_written_does_not_mask_the_failure(self):
        """It runs on the failure path, so it must never replace the
        failure it was describing."""
        logs = self.crashes.parent
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "in-the-way").write_text("not a directory")
        os.environ["B2C_LOG_DIR"] = str(logs / "in-the-way")
        with self.assertRaises(ProcessFailed):
            self.step._run_brush(_fake_brush(self.ply, write=None, exit_code=1), self.ply)


if __name__ == "__main__":
    unittest.main()
