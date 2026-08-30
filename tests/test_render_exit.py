"""brush-splat-render's non-zero exits are judged against the frames, and
every crash leaves something behind to read.

The training half of this binary has a known shutdown SIGSEGV (exit -11)
that lands *after* the work is on disk — see tests/test_brush_exit.py — and
the rasteriser has now been seen failing on a pod in what may well be the
same thing. So it gets the same guard: a failed exit with every frame
written is a finished render, and only a missing frame is a real failure.

The second half of these is the reason that pod crash could not be
diagnosed at all. The render's cameras.json and its part-written frames
live in a `TemporaryDirectory` that is deleted on the way out of
`_rasterize`, exception or not, so the crash left an exit code and nothing
else. They are now copied to `paths.crash_dir()` first.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.proc import ProcessFailed
from pipeline.steps.splat import _run_render

from tests.helpers import crash_dirs, redirect_crash_dir

NAMES = ["frame_00001_.png", "frame_00002_.png", "frame_00003_.png"]


def _fake_render(output_dir: Path, *, writes, exit_code: int = 0,
                 segfault: bool = False, body: str = "png"):
    """argv for a process that writes `writes` into `output_dir`, then exits.

    `segfault=True` reproduces the case the guard exists for: every frame
    is on disk and the process then dies on SIGSEGV, which Popen reports as
    returncode -11 rather than as any exit status.
    """
    script = f"import os; d={str(output_dir)!r}; os.makedirs(d, exist_ok=True);"
    for name in writes:
        script += f"open(os.path.join(d, {name!r}), 'w').write({body!r});"
    if segfault:
        script += "import signal; os.kill(os.getpid(), signal.SIGSEGV)"
    else:
        script += f"import sys; sys.exit({exit_code})"
    return [sys.executable, "-c", script]


class _RenderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="b2c_render_test_")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.crashes = redirect_crash_dir(self)

        self.output_dir = root / "renders"
        self.cameras = root / "cameras.json"
        self.cameras.write_text('{"width": 4, "height": 4, "cameras": []}')

    def run_render(self, cmd):
        _run_render(
            cmd,
            output_dir=self.output_dir,
            cameras_path=self.cameras,
            image_names=list(NAMES),
        )

    def crash_dirs(self):
        return crash_dirs(self.crashes)


class TestRenderExitCode(_RenderCase):
    def test_a_failed_exit_with_every_frame_written_is_accepted(self):
        with self.assertLogs("pipeline.steps.splat", level=logging.WARNING) as caught:
            self.run_render(_fake_render(self.output_dir, writes=NAMES, exit_code=1))
        self.assertIn("treating the render as successful", "\n".join(caught.output))

    def test_a_clean_exit_says_nothing(self):
        with self.assertNoLogs("pipeline.steps.splat", level=logging.WARNING):
            self.run_render(_fake_render(self.output_dir, writes=NAMES))

    def test_the_real_shape_of_the_failure_rendered_then_segfaulted(self):
        with self.assertLogs("pipeline.steps.splat", level=logging.WARNING) as caught:
            self.run_render(_fake_render(self.output_dir, writes=NAMES, segfault=True))
        self.assertIn("-11", "\n".join(caught.output))

    def test_a_failed_exit_with_a_missing_frame_still_raises(self):
        with self.assertRaises(ProcessFailed):
            self.run_render(_fake_render(self.output_dir, writes=NAMES[:2], exit_code=1))

    def test_an_empty_frame_counts_as_missing(self):
        """A zero-byte PNG is a crash caught mid-write, not a frame."""
        with self.assertRaises(ProcessFailed):
            self.run_render(
                _fake_render(self.output_dir, writes=NAMES, exit_code=1, body="")
            )

    def test_a_clean_exit_that_wrote_nothing_is_still_a_failure(self):
        """And it fails here, where the evidence still exists, rather than
        several lines later as cv2.imread returning None on a temp
        directory that has since been deleted."""
        with self.assertRaises(RuntimeError) as caught:
            self.run_render(_fake_render(self.output_dir, writes=[]))
        self.assertNotIsInstance(caught.exception, ProcessFailed)
        self.assertIn("frame_00001_.png", str(caught.exception))

    def test_a_missing_binary_is_still_a_hard_failure(self):
        """Not a ProcessFailed at all — argv[0] never started, so there is
        no exit code to weigh against any frame."""
        with self.assertRaises(RuntimeError) as caught:
            self.run_render(["b2c-no-such-binary-exists"])
        self.assertNotIsInstance(caught.exception, ProcessFailed)
        self.assertEqual(self.crash_dirs(), [], "nothing ran, so there is nothing to save")


class TestRenderCrashlog(_RenderCase):
    """What a crash leaves behind once the temp directory is gone."""

    def _crash_after(self, cmd, expect_raises=True):
        if expect_raises:
            with self.assertRaises(Exception):
                self.run_render(cmd)
        else:
            with self.assertLogs("pipeline.steps.splat", level=logging.WARNING):
                self.run_render(cmd)
        dirs = self.crash_dirs()
        self.assertEqual(len(dirs), 1, f"expected one crash directory, got {dirs}")
        return dirs[0]

    def test_a_hard_failure_saves_the_views_it_was_rendering(self):
        crash = self._crash_after(
            _fake_render(self.output_dir, writes=NAMES[:1], exit_code=1)
        )
        self.assertEqual(
            (crash / "cameras.json").read_text(), self.cameras.read_text()
        )

    def test_the_report_names_the_exit_code_the_argv_and_every_frame(self):
        crash = self._crash_after(
            _fake_render(self.output_dir, writes=NAMES[:1], exit_code=3)
        )
        report = (crash / "report.txt").read_text()
        self.assertIn("exit code 3", report)
        self.assertIn(sys.executable, report)
        self.assertIn("1 of 3 written, first missing frame_00002_.png", report)
        for name in NAMES:
            self.assertIn(name, report)
        self.assertIn("frame_00002_.png  missing", report)
        # The environment that decides whether this binary can reach a GPU
        # at all, recorded while the pod still exists to be asked.
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=", report)

    def test_the_frames_that_did_land_are_copied_out(self):
        crash = self._crash_after(
            _fake_render(self.output_dir, writes=NAMES[:2], exit_code=1)
        )
        kept = sorted(p.name for p in (crash / "frames").iterdir())
        self.assertEqual(kept, NAMES[:2])

    def test_only_the_last_few_frames_are_kept(self):
        """A full orbit's renders are not something to copy onto the volume
        every time; the frames around the crash are what gets read."""
        from pipeline.steps import splat as splat_module

        many = [f"frame_{i + 1:05d}_.png" for i in range(12)]
        cmd = _fake_render(self.output_dir, writes=many[:10], exit_code=1)
        with self.assertRaises(ProcessFailed):
            _run_render(
                cmd, output_dir=self.output_dir, cameras_path=self.cameras,
                image_names=many,
            )
        crash = self.crash_dirs()[0]
        kept = sorted(p.name for p in (crash / "frames").iterdir())
        self.assertEqual(kept, many[10 - splat_module._CRASH_FRAMES_KEPT:10])

    def test_a_tolerated_crash_is_still_saved(self):
        """The guard says the render is usable; it does not say the crash
        is uninteresting. This is the case the pod round needs read."""
        crash = self._crash_after(
            _fake_render(self.output_dir, writes=NAMES, segfault=True),
            expect_raises=False,
        )
        self.assertIn("3 of 3 written (all present)", (crash / "report.txt").read_text())

    def test_a_crashlog_that_cannot_be_written_does_not_mask_the_failure(self):
        """It runs on the failure path, so it must never replace the
        failure it was describing."""
        logs = self.crashes.parent
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "in-the-way").write_text("not a directory")
        os.environ["B2C_LOG_DIR"] = str(logs / "in-the-way")
        with self.assertRaises(ProcessFailed):
            self.run_render(_fake_render(self.output_dir, writes=[], exit_code=1))


if __name__ == "__main__":
    unittest.main()
