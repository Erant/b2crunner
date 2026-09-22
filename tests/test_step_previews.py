"""The per-step contact sheet: what gets sampled, and what it survives.

This exists to answer "which step broke the output" by looking rather than
by reading a log, so the properties that matter are the unglamorous ones:
the sampled frames are spread across the batch rather than bunched at the
front, a mask-only step's previews actually differ from the step before
it, and nothing in here can take down the run it is observing.

`preview_indices`/`write_previews` live in `pipeline.run_state`, which has
no `gradio` dependency; the status-publishing side (what used to be
`RunManager._capture_previews`/`_on_event`) now lives in
`pipeline.run_worker._StatusWriter`, run as its own OS process — see that
module's docstring. Both are exercised here directly; only the last test
(which also checks `pipeline.webui.preview_gallery`) needs the UI's own
dependency, hence the guarded import.
"""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np

from pipeline import run_state
from pipeline.run_state import RunState, StepRecord
from pipeline.runner import RunEvent

try:
    from pipeline import run_worker, webui
except ImportError as exc:  # pragma: no cover - depends on the local env
    raise unittest.SkipTest(f"the web UI's dependencies are not installed here: {exc}")


class TestPreviewIndices(unittest.TestCase):
    def test_an_81_frame_orbit_strides_by_ten(self):
        """The shape the workflows actually produce — frames 0, 10, ... 70."""
        self.assertEqual(
            run_state.preview_indices(81), [0, 10, 20, 30, 40, 50, 60, 70]
        )

    def test_it_spans_the_batch_rather_than_its_front(self):
        """A denoise pass that holds up at the front of the orbit and falls
        apart at the back looks perfect in the first eight frames."""
        indices = run_state.preview_indices(200)
        self.assertEqual(len(indices), 8)
        self.assertGreater(indices[-1], 150)

    def test_a_short_batch_gives_what_there_is(self):
        self.assertEqual(run_state.preview_indices(3), [0, 1, 2])
        self.assertEqual(run_state.preview_indices(8), list(range(8)))

    def test_nothing_to_sample(self):
        self.assertEqual(run_state.preview_indices(0), [])


class TestWritePreviews(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "03_denoise_pass1"
        self.images = [
            np.full((1280, 720, 3), i * 2, np.uint8) for i in range(81)
        ]
        self.names = [f"frame_{i + 1:05d}_.png" for i in range(81)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_writes_one_file_per_sampled_frame(self):
        written = run_state.write_previews(self.images, None, self.names, self.dest)

        self.assertEqual(len(written), 8)
        self.assertTrue(all(Path(p).exists() for p in written))
        self.assertTrue(all(p.endswith(".jpg") for p in written))

    def test_the_filenames_carry_the_frame_they_came_from(self):
        written = run_state.write_previews(self.images, None, self.names, self.dest)

        self.assertIn("frame_00001_", Path(written[0]).name)
        self.assertIn("frame_00011_", Path(written[1]).name)

    def test_frames_are_downscaled(self):
        written = run_state.write_previews(self.images, None, self.names, self.dest)
        preview = cv2.imread(written[0])

        self.assertEqual(preview.shape[1], run_state.PREVIEW_WIDTH)
        # Aspect preserved: 720x1280 in, so taller than it is wide out.
        self.assertGreater(preview.shape[0], preview.shape[1])

    def test_a_mask_only_step_produces_visibly_different_previews(self):
        """rmbg and mask_splat change nothing but the mask. Without
        compositing it in, their previews are identical to the step before
        them — which is precisely when someone is looking at this."""
        unmasked = run_state.write_previews(self.images, None, self.names, self.dest)

        masks = [np.zeros((1280, 720), np.float32) for _ in self.images]
        for mask in masks:
            mask[400:900, 200:500] = 1.0
        masked = run_state.write_previews(
            self.images, masks, self.names, self.dest.parent / "04_rmbg"
        )

        before, after = cv2.imread(unmasked[0]), cv2.imread(masked[0])
        self.assertEqual(before.shape, after.shape)
        self.assertFalse(np.array_equal(before, after))
        # Outside the mask the frame is drawn over the backdrop, not kept.
        corner = after[5, 5]
        self.assertTrue(np.all(np.abs(corner.astype(int) - run_state.PREVIEW_BACKDROP) < 12))

    def test_rgba_frames_use_their_own_alpha(self):
        rgba = []
        for image in self.images[:12]:
            alpha = np.zeros(image.shape[:2], np.uint8)
            alpha[400:900, 200:500] = 255
            rgba.append(np.dstack([image, alpha]))

        written = run_state.write_previews(rgba, None, self.names, self.dest)
        preview = cv2.imread(written[0])

        self.assertEqual(preview.shape[2], 3)  # JPEG, so alpha is composited
        self.assertTrue(
            np.all(np.abs(preview[5, 5].astype(int) - run_state.PREVIEW_BACKDROP) < 12)
        )

    def test_greyscale_frames_do_not_crash_it(self):
        grey = [np.full((640, 360), 90, np.uint8) for _ in range(10)]
        self.assertEqual(len(run_state.write_previews(grey, None, [], self.dest)), 8)


class _FakeContext:
    def __init__(self, data):
        self._data = data

    def get(self, path):
        return self._data[path]


class TestCaptureIsNeverFatal(unittest.TestCase):
    """A debugging aid must not be able to kill the run it is watching."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.writer = run_worker._StatusWriter(
            RunState(
                total=1,
                steps=[StepRecord(1, "denoise_pass1", "wan22_vace_denoise")],
                output_dir=Path(self.tmp.name),
            ),
            status_path=Path(self.tmp.name) / "status.json",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _event(self, context):
        return RunEvent(
            kind="step_end", workflow="t", index=1, total=1,
            step_id="denoise_pass1", step_name="wan22_vace_denoise",
            context=context,
        )

    def test_a_context_with_no_images_yet(self):
        """Every step before `render` in a from-a-photo workflow."""
        self.assertEqual(
            self.writer._capture_previews(self._event(_FakeContext({}))), ([], False)
        )

    def test_no_context_at_all(self):
        self.assertEqual(self.writer._capture_previews(self._event(None)), ([], False))

    def test_a_write_that_blows_up_is_swallowed(self):
        context = _FakeContext({
            "dataset.images": [np.zeros((8, 8, 3), np.uint8)],
            "dataset.masks": None,
            "dataset.image_names": ["frame_00001_.png"],
        })
        with unittest.mock.patch.object(
            run_worker, "write_previews", side_effect=OSError("disk full")
        ):
            self.assertEqual(
                self.writer._capture_previews(self._event(context)), ([], False)
            )

    def test_the_happy_path_records_against_the_step(self):
        context = _FakeContext({
            "dataset.images": [np.full((64, 32, 3), 200, np.uint8) for _ in range(20)],
            "dataset.masks": None,
            "dataset.image_names": [f"frame_{i + 1:05d}_.png" for i in range(20)],
        })
        self.writer(self._event(context))

        record = self.writer.state.steps[0]
        self.assertEqual(len(record.previews), 8)
        self.assertTrue(
            all(run_state.PREVIEW_DIRNAME in p for p in record.previews),
            "previews belong in their own directory, out of the result archive",
        )

        gallery = webui.preview_gallery(self.writer.state)
        self.assertEqual(len(gallery), 8)
        self.assertTrue(all(caption.startswith("01 denoise_pass1") for _, caption in gallery))
        self.assertEqual(
            webui.preview_step_choices(self.writer.state),
            [webui.PREVIEW_ALL, "01 denoise_pass1"],
        )


class TestUnchangedStepsHaveNoRow(unittest.TestCase):
    """Camera refinement, splat training, the fits: most of a run's steps
    leave the sampled frames exactly as they found them, and a row that
    repeats the one above it only pushes the step that did change
    something further down the sheet."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        ids = ["render", "refine_cameras", "denoise"]
        self.writer = run_worker._StatusWriter(
            RunState(
                total=3,
                steps=[StepRecord(i + 1, sid, sid) for i, sid in enumerate(ids)],
                output_dir=Path(self.tmp.name),
            ),
            status_path=Path(self.tmp.name) / "status.json",
        )
        self.ids = ids

    def tearDown(self):
        self.tmp.cleanup()

    def _end(self, index, value, names=None):
        context = _FakeContext({
            "dataset.images": [np.full((64, 32, 3), value, np.uint8) for _ in range(20)],
            "dataset.masks": None,
            "dataset.image_names": names or [f"frame_{i + 1:05d}_.png" for i in range(20)],
        })
        self.writer(RunEvent(
            kind="step_end", workflow="t", index=index, total=3,
            step_id=self.ids[index - 1], step_name=self.ids[index - 1], context=context,
        ))

    def test_a_step_that_changed_nothing_is_dropped_and_the_next_compares_to_the_last_kept(self):
        self._end(1, 100)
        self._end(2, 100)
        self._end(3, 180)

        first, second, third = self.writer.state.steps
        self.assertEqual(len(first.previews), 8)
        self.assertEqual(second.previews, [])
        self.assertTrue(second.unchanged)
        self.assertEqual(len(third.previews), 8)
        self.assertFalse(third.unchanged)
        # The dropped sheet does not linger on disk either.
        self.assertFalse((Path(self.tmp.name) / run_state.PREVIEW_DIRNAME / "02_refine_cameras").exists())

        gallery = webui.preview_gallery(self.writer.state)
        self.assertEqual(len(gallery), 16)
        self.assertFalse(any("refine_cameras" in caption for _, caption in gallery))
        self.assertNotIn("02 refine_cameras", webui.preview_step_choices(self.writer.state))

    def test_a_rename_alone_is_not_a_change(self):
        self._end(1, 100)
        self._end(2, 100, names=[f"view_{i:03d}.png" for i in range(20)])
        self.assertTrue(self.writer.state.steps[1].unchanged)

    def test_a_masked_step_keeps_its_row(self):
        """The compositing exists so that a mask-only step *is* a change."""
        self._end(1, 100)
        masks = [np.zeros((64, 32), np.float32) for _ in range(20)]
        for mask in masks:
            mask[10:50, 5:25] = 1.0
        context = _FakeContext({
            "dataset.images": [np.full((64, 32, 3), 100, np.uint8) for _ in range(20)],
            "dataset.masks": masks,
            "dataset.image_names": [f"frame_{i + 1:05d}_.png" for i in range(20)],
        })
        self.writer(RunEvent(
            kind="step_end", workflow="t", index=2, total=3,
            step_id="refine_cameras", step_name="refine_cameras", context=context,
        ))
        self.assertEqual(len(self.writer.state.steps[1].previews), 8)
        self.assertFalse(self.writer.state.steps[1].unchanged)

    def test_the_flag_survives_the_status_file(self):
        self._end(1, 100)
        self._end(2, 100)
        self.writer.publish()
        loaded = RunState.from_dict(
            __import__("json").loads((Path(self.tmp.name) / "status.json").read_text())
        )
        self.assertTrue(loaded.steps[1].unchanged)
        self.assertFalse(loaded.steps[0].unchanged)


if __name__ == "__main__":
    unittest.main()
