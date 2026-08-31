"""Every brush-splat-render crash leaves something behind to read.

The rasterisation itself is body2colmap's since 2026-08-31 — `_rasterize`
drives its `SplatRenderer` rather than running the binary itself — and so
is the judgement that used to live here: a non-zero exit with every frame
written is a finished render, and only a missing frame is a real failure.
That is `render_many`'s now, and `tests/test_splat_renderer.py` in that
repo covers it. **What is still this project's is the crash report**, and
it is the half that matters most, because it is the reason a real pod crash
could not be diagnosed at all.

Everything the binary is handed lives in a temp directory `render_many`
deletes on the way out, exception or not, so that crash left an exit code
and nothing else. body2colmap now calls `on_fault` while the directory is
still there; `_save_render_crashlog` is what this project hangs off it, and
copies the cameras.json, a per-frame manifest and the last frames written
into `paths.crash_dir()` before it goes.

Driven against a stub binary, because the cases worth testing are crashes.
"""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import crash_dirs, redirect_crash_dir, stub_render_binary


def _cameras(count):
    from body2colmap.camera import Camera

    return [
        Camera(focal_length=(4.0, 4.0), image_size=(4, 4),
               principal_point=(2.0, 2.0),
               position=np.array([0.0, 0.0, float(i + 1)], dtype=np.float32),
               rotation=np.eye(3, dtype=np.float32))
        for i in range(count)
    ]


class _Scene:
    """Enough of a SplatScene for the renderer; never serialized, because
    `_rasterize` passes the .ply below as `ply_path`."""

    sh_degree = 0

    def to_ply(self, path):
        raise AssertionError("an existing .ply must be rendered where it lies")

    def __len__(self):
        return 1


class _RenderCase(unittest.TestCase):
    FRAMES = 3

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="b2c_render_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.crashes = redirect_crash_dir(self)
        self.ply = self.root / "scene.ply"
        self.ply.write_bytes(b"a splat, as far as the stub is concerned")

    def render(self, *, frames="all", damage="", segfault=False, count=None):
        from pipeline.steps import splat as splat_module

        count = self.FRAMES if count is None else count
        binary = stub_render_binary(self.root, frames=frames, damage=damage,
                                    segfault=segfault)
        return splat_module._rasterize(
            scene=_Scene(),
            splat_path=str(self.ply),
            cameras=_cameras(count),
            image_names=[f"frame_{i + 1:05d}_.png" for i in range(count)],
            width=4,
            height=4,
            bg_color=(0.0, 0.0, 0.0),
            render_path=binary,
            confidence=None,
        )

    def crash_dirs(self):
        return crash_dirs(self.crashes)


class TestTheVerdictStillHolds(_RenderCase):
    """body2colmap's judgement, checked from this side because it is what
    the pipeline depends on — not a second implementation of it."""

    def test_a_clean_render_returns_its_frames(self):
        images, masks = self.render()
        self.assertEqual(len(images), self.FRAMES)
        self.assertEqual(len(masks), self.FRAMES)
        self.assertEqual(self.crash_dirs(), [], "nothing went wrong")

    def test_a_crash_after_every_frame_was_written_is_accepted(self):
        with self.assertLogs("pipeline.steps.splat", level=logging.WARNING) as caught:
            images, _ = self.render(segfault=True)
        self.assertEqual(len(images), self.FRAMES)
        self.assertIn("treating the render as successful", "\n".join(caught.output))

    def test_a_crash_that_lost_a_frame_still_raises(self):
        with self.assertRaises(RuntimeError):
            self.render(frames="short", segfault=True)

    def test_an_empty_frame_counts_as_missing(self):
        """A zero-byte PNG is a crash caught mid-write, not a frame."""
        with self.assertRaises(RuntimeError):
            self.render(damage="empty", segfault=True)

    def test_a_truncated_frame_is_caught_too(self):
        """It is non-empty, so only the decode catches it."""
        with self.assertRaises(RuntimeError):
            self.render(damage="truncate", segfault=True)

    def test_a_missing_binary_is_a_hard_failure_with_nothing_to_save(self):
        from pipeline.steps import splat as splat_module

        with self.assertRaises(RuntimeError):
            splat_module._rasterize(
                scene=_Scene(), splat_path=str(self.ply), cameras=_cameras(1),
                image_names=["frame_00001_.png"], width=4, height=4,
                bg_color=(0.0, 0.0, 0.0),
                render_path="b2c-no-such-binary-exists", confidence=None,
            )
        self.assertEqual(self.crash_dirs(), [],
                         "nothing ran, so there is nothing to save")


class TestRenderCrashlog(_RenderCase):
    """What a crash leaves behind once the temp directory is gone."""

    def _crash(self, *, expect_raises=True, **kwargs):
        if expect_raises:
            with self.assertRaises(Exception):
                self.render(**kwargs)
        else:
            with self.assertLogs("pipeline.steps.splat", level=logging.WARNING):
                self.render(**kwargs)
        dirs = self.crash_dirs()
        self.assertEqual(len(dirs), 1, f"expected one crash directory, got {dirs}")
        return dirs[0]

    def test_it_saves_the_views_it_was_rendering(self):
        """The thing whose absence made the pod crash undiagnosable, and the
        only reason the fault hook exists at all."""
        crash = self._crash(frames="short", segfault=True)
        payload = (crash / "cameras.json").read_text()
        self.assertIn('"cameras"', payload)
        self.assertIn('"fx": 4.0', payload)

    def test_the_report_names_the_signal_the_argv_and_every_frame(self):
        crash = self._crash(frames="short", segfault=True)
        report = (crash / "report.txt").read_text()

        # Decoded, not left as "exit -11" for the reader to look up.
        self.assertIn("SIGSEGV", report)
        self.assertIn("stub-brush-splat-render.py", report)
        self.assertIn("2 of 3 written, first missing frame_00003_.png", report)
        # The manifest is in THIS project's frame names, not the renderer's
        # f00000.png, so the report reads in the run's own terms.
        for i in range(1, 4):
            self.assertIn(f"frame_{i:05d}_.png", report)
        self.assertIn("frame_00003_.png  missing", report)
        # What the binary was saying on its way down.
        self.assertIn("stub wrote 2 of 3 frames", report)
        # The environment that decides whether this binary can reach a GPU
        # at all, recorded while the pod still exists to be asked.
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES=", report)

    def test_the_frames_that_did_land_are_copied_out(self):
        crash = self._crash(frames="short", segfault=True)
        kept = sorted(p.name for p in (crash / "frames").iterdir())
        self.assertEqual(len(kept), 2)

    def test_only_the_last_few_frames_are_kept(self):
        """A full orbit's renders are not something to copy onto the volume
        every time; the frames around the crash are what gets read."""
        from pipeline.steps import splat as splat_module

        crash = self._crash(count=12, frames="short", segfault=True)
        kept = sorted(p.name for p in (crash / "frames").iterdir())
        self.assertEqual(len(kept), splat_module._CRASH_FRAMES_KEPT)

    def test_a_tolerated_crash_is_still_saved(self):
        """The guard says the render is usable; it does not say the crash is
        uninteresting. This is the case the pod round needs read."""
        crash = self._crash(segfault=True, expect_raises=False)
        self.assertIn("3 of 3 written (all present)",
                      (crash / "report.txt").read_text())

    def test_a_crashlog_that_cannot_be_written_does_not_mask_the_failure(self):
        """It runs on the failure path, so it must never replace the failure
        it was describing."""
        logs = self.crashes.parent
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "in-the-way").write_text("not a directory")
        os.environ["B2C_LOG_DIR"] = str(logs / "in-the-way")
        with self.assertRaises(RuntimeError) as caught:
            self.render(frames="short", segfault=True)
        self.assertIn("frame", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
