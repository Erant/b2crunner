"""The two small steps the re-outline branch is built from.

`resize_batch` exists because diffusers fits a VACE control video under the
target AREA rather than resizing it to the size asked for (see the step's
module docstring), and its one property worth pinning is the round trip:
a plain resize down and the same plain resize back up must put a mask on
the pixel grid it came from, anisotropy and all. `rmbg`'s `debug_dir` is the
only place the 480p denoise output touches disk.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.registry import get_step_class
from pipeline.steps.resize import resize_frames


def _run(step, inputs, params):
    cls = get_step_class(step)
    return cls().run(inputs, cls.resolve_params(params))


class TestResizeBatch(unittest.TestCase):
    def test_images_and_masks_land_at_the_target_size(self):
        images = [np.full((1280, 720, 3), 90, dtype=np.uint8) for _ in range(2)]
        masks = [np.ones((1280, 720), dtype=np.float32), np.zeros((1280, 720), dtype=np.float32)]
        out = _run("resize_batch", {"images": images, "masks": masks},
                   {"width": 480, "height": 832})
        self.assertEqual([i.shape for i in out["images"]], [(832, 480, 3)] * 2)
        self.assertEqual([m.shape for m in out["masks"]], [(832, 480)] * 2)
        self.assertEqual(out["images"][0].dtype, np.uint8)
        self.assertEqual(out["masks"][0].dtype, np.float32)

    def test_a_per_frame_vace_flag_survives_exactly(self):
        """inject_anchor's masks are all-1.0 or all-0.0 per frame; a resize
        must not invent an edge in them."""
        masks = [np.ones((64, 48), dtype=np.float32), np.zeros((64, 48), dtype=np.float32)]
        out = _run("resize_batch", {"masks": masks}, {"width": 32, "height": 40})
        self.assertTrue(np.all(out["masks"][0] == 1.0))
        self.assertTrue(np.all(out["masks"][1] == 0.0))

    def test_only_what_was_given_comes_back(self):
        out = _run("resize_batch", {"masks": [np.zeros((8, 8), np.uint8)]},
                   {"width": 4, "height": 4})
        self.assertEqual(set(out), {"masks"})
        with self.assertRaises(ValueError):
            _run("resize_batch", {}, {"width": 4, "height": 4})

    def test_a_uint8_mask_is_normalised(self):
        out = _run("resize_batch", {"masks": [np.full((8, 8), 255, np.uint8)]},
                   {"width": 4, "height": 4})
        self.assertTrue(np.all(out["masks"][0] == 1.0))

    def test_the_anisotropic_round_trip_cancels(self):
        """720x1280 -> 480x832 -> 720x1280: a hard silhouette comes back
        within a pixel of where it started, although the two axes were
        scaled by different factors on the way down."""
        mask = np.zeros((1280, 720), dtype=np.float32)
        mask[200:1100, 180:540] = 1.0
        down = _run("resize_batch", {"masks": [mask]}, {"width": 480, "height": 832})["masks"]
        up = _run("resize_batch", {"masks": down}, {"width": 720, "height": 1280})["masks"][0]
        back = up >= 0.5
        original = mask >= 0.5
        disagreement = back ^ original
        # Only a one-pixel band around the edge may differ.
        eroded = cv2.erode(original.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        dilated = cv2.dilate(original.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.assertFalse(disagreement[eroded].any())
        self.assertFalse(disagreement[~dilated].any())

    def test_auto_picks_area_down_and_linear_up(self):
        """A box filter shrinking a checkerboard averages it to mid-grey;
        linear enlargement of a hard step stays monotone with a ramp."""
        checker = np.indices((64, 64)).sum(axis=0) % 2 * 255
        small = resize_frames([checker.astype(np.uint8)], 32, 32)[0]
        self.assertTrue(np.all(np.abs(small.astype(int) - 127) <= 1))
        step = np.zeros((4, 4), np.float32)
        step[:, 2:] = 1.0
        big = resize_frames([step], 16, 4)[0]
        self.assertTrue(np.all(np.diff(big[0]) >= 0))
        self.assertTrue(((big > 0.0) & (big < 1.0)).any())


class TestRmbgDebugDir(unittest.TestCase):
    def test_frames_and_mattes_are_written_one_pair_per_frame(self):
        cls = get_step_class("rmbg")
        step = cls()
        step._model = object()  # never called: _run_batch is stubbed below
        images = [np.full((16, 12, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
        step._run_batch = lambda batch: [
            np.full(img.shape[:2], 0.5, dtype=np.float32) for img in batch
        ]
        with tempfile.TemporaryDirectory() as tmp:
            debug = Path(tmp) / "debug" / "reoutline"
            out = step.run({"images": images}, cls.resolve_params({"debug_dir": str(debug)}))
            self.assertEqual(len(out["masks"]), 3)
            names = sorted(p.name for p in debug.iterdir())
            self.assertEqual(names, [
                "frame_00001.png", "frame_00002.png", "frame_00003.png",
                "matte_00001.png", "matte_00002.png", "matte_00003.png",
            ])
            self.assertEqual(int(cv2.imread(str(debug / "frame_00002.png"))[0, 0, 0]), 20)
            self.assertEqual(int(cv2.imread(str(debug / "matte_00001.png"), 0)[0, 0]), 127)

    def test_off_by_default_writes_nothing(self):
        declared = get_step_class("rmbg").declared_params()
        self.assertIsNone(declared["debug_dir"].default)


if __name__ == "__main__":
    unittest.main()
