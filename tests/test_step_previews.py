"""The per-step contact sheet: what gets sampled, and what it survives.

This exists to answer "which step broke the output" by looking rather than
by reading a log, so the properties that matter are the unglamorous ones:
the sampled frames are spread across the batch rather than bunched at the
front, a mask-only step's previews actually differ from the step before
it, and nothing in here can take down the run it is observing.

`pipeline.webui` imports gradio, so this skips where the UI's own
dependency isn't installed.
"""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np

try:
    from pipeline import webui
except ImportError as exc:  # pragma: no cover - depends on the local env
    raise unittest.SkipTest(f"the web UI's dependencies are not installed here: {exc}")

from pipeline.runner import RunEvent


class TestPreviewIndices(unittest.TestCase):
    def test_an_81_frame_orbit_strides_by_ten(self):
        """The shape the workflows actually produce — frames 0, 10, ... 70."""
        self.assertEqual(
            webui.preview_indices(81), [0, 10, 20, 30, 40, 50, 60, 70]
        )

    def test_it_spans_the_batch_rather_than_its_front(self):
        """A denoise pass that holds up at the front of the orbit and falls
        apart at the back looks perfect in the first eight frames."""
        indices = webui.preview_indices(200)
        self.assertEqual(len(indices), 8)
        self.assertGreater(indices[-1], 150)

    def test_a_short_batch_gives_what_there_is(self):
        self.assertEqual(webui.preview_indices(3), [0, 1, 2])
        self.assertEqual(webui.preview_indices(8), list(range(8)))

    def test_nothing_to_sample(self):
        self.assertEqual(webui.preview_indices(0), [])


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
        written = webui.write_previews(self.images, None, self.names, self.dest)

        self.assertEqual(len(written), 8)
        self.assertTrue(all(Path(p).exists() for p in written))
        self.assertTrue(all(p.endswith(".jpg") for p in written))

    def test_the_filenames_carry_the_frame_they_came_from(self):
        written = webui.write_previews(self.images, None, self.names, self.dest)

        self.assertIn("frame_00001_", Path(written[0]).name)
        self.assertIn("frame_00011_", Path(written[1]).name)

    def test_frames_are_downscaled(self):
        written = webui.write_previews(self.images, None, self.names, self.dest)
        preview = cv2.imread(written[0])

        self.assertEqual(preview.shape[1], webui._PREVIEW_WIDTH)
        # Aspect preserved: 720x1280 in, so taller than it is wide out.
        self.assertGreater(preview.shape[0], preview.shape[1])

    def test_a_mask_only_step_produces_visibly_different_previews(self):
        """rmbg and mask_splat change nothing but the mask. Without
        compositing it in, their previews are identical to the step before
        them — which is precisely when someone is looking at this."""
        unmasked = webui.write_previews(self.images, None, self.names, self.dest)

        masks = [np.zeros((1280, 720), np.float32) for _ in self.images]
        for mask in masks:
            mask[400:900, 200:500] = 1.0
        masked = webui.write_previews(
            self.images, masks, self.names, self.dest.parent / "04_rmbg"
        )

        before, after = cv2.imread(unmasked[0]), cv2.imread(masked[0])
        self.assertEqual(before.shape, after.shape)
        self.assertFalse(np.array_equal(before, after))
        # Outside the mask the frame is drawn over the backdrop, not kept.
        corner = after[5, 5]
        self.assertTrue(np.all(np.abs(corner.astype(int) - webui._PREVIEW_BACKDROP) < 12))

    def test_rgba_frames_use_their_own_alpha(self):
        rgba = []
        for image in self.images[:12]:
            alpha = np.zeros(image.shape[:2], np.uint8)
            alpha[400:900, 200:500] = 255
            rgba.append(np.dstack([image, alpha]))

        written = webui.write_previews(rgba, None, self.names, self.dest)
        preview = cv2.imread(written[0])

        self.assertEqual(preview.shape[2], 3)  # JPEG, so alpha is composited
        self.assertTrue(
            np.all(np.abs(preview[5, 5].astype(int) - webui._PREVIEW_BACKDROP) < 12)
        )

    def test_greyscale_frames_do_not_crash_it(self):
        grey = [np.full((640, 360), 90, np.uint8) for _ in range(10)]
        self.assertEqual(len(webui.write_previews(grey, None, [], self.dest)), 8)


class _FakeContext:
    def __init__(self, data):
        self._data = data

    def get(self, path):
        return self._data[path]


class TestCaptureIsNeverFatal(unittest.TestCase):
    """A debugging aid must not be able to kill the run it is watching."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = webui.RunManager(envs_path="")
        self.manager.state = webui.RunState(
            total=1,
            steps=[webui.StepRecord(1, "denoise_pass1", "wan22_vace_denoise")],
            output_dir=Path(self.tmp.name),
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
        self.assertEqual(self.manager._capture_previews(self._event(_FakeContext({}))), [])

    def test_no_context_at_all(self):
        self.assertEqual(self.manager._capture_previews(self._event(None)), [])

    def test_a_write_that_blows_up_is_swallowed(self):
        context = _FakeContext({
            "dataset.images": [np.zeros((8, 8, 3), np.uint8)],
            "dataset.masks": None,
            "dataset.image_names": ["frame_00001_.png"],
        })
        with unittest.mock.patch.object(
            webui, "write_previews", side_effect=OSError("disk full")
        ):
            self.assertEqual(self.manager._capture_previews(self._event(context)), [])

    def test_the_happy_path_records_against_the_step(self):
        context = _FakeContext({
            "dataset.images": [np.full((64, 32, 3), 200, np.uint8) for _ in range(20)],
            "dataset.masks": None,
            "dataset.image_names": [f"frame_{i + 1:05d}_.png" for i in range(20)],
        })
        self.manager._on_event(self._event(context))

        record = self.manager.state.steps[0]
        self.assertEqual(len(record.previews), 8)
        self.assertTrue(
            all(webui._PREVIEW_DIRNAME in p for p in record.previews),
            "previews belong in their own directory, out of the result archive",
        )

        gallery = webui.preview_gallery(self.manager.state)
        self.assertEqual(len(gallery), 8)
        self.assertTrue(all(caption.startswith("01 denoise_pass1") for _, caption in gallery))
        self.assertEqual(
            webui.preview_step_choices(self.manager.state),
            [webui.PREVIEW_ALL, "01 denoise_pass1"],
        )


if __name__ == "__main__":
    unittest.main()
