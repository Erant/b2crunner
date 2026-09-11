"""pixel_ops — pixel-space operations on a batch, each off by default.

`specular_suppress` (2026-09-10) is the first: the frames the intermediate
splat is trained on are pass 1's output, highlights and all, and lit a
second time by pass 2 those blow out. The suppression is the dichromatic
model's specular-free image: the minimum channel's excess over
`mean + eta * std` of the batch's matte pixels, taken off all three
channels equally.

An 8x8 patch of one skin colour with a single brighter, whiter pixel in
the middle of it — body colour plus a white term — is the whole model.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import run_step

import pipeline.steps  # noqa: F401


SKIN = (120, 160, 220)          # BGR
HIGHLIGHT = (200, 220, 250)     # the same skin plus ~80 of white


def _frame(highlight_at=(4, 4)):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = SKIN
    image[highlight_at] = HIGHLIGHT
    return image


def _run(images, mask=None, anchor=None, **params):
    if mask is None:
        mask = np.ones((8, 8), dtype=np.float32)
    inputs = {"images": images, "masks": [mask] * len(images)}
    if anchor is not None:
        inputs["anchor_frame_index"] = anchor
    return run_step(
        "pixel_ops", inputs,
        {"specular_suppress": 1.0, "specular_blur": 0.0, **params},
    )["images"]


class TestPixelOpsOff(unittest.TestCase):

    def test_every_knob_off_is_the_frames_as_they_came(self):
        images = [_frame()]
        out = run_step("pixel_ops", {"images": images}, {})["images"]
        self.assertIs(out[0], images[0])

    def test_off_needs_no_matte(self):
        run_step("pixel_ops", {"images": [_frame()]}, {})


class TestSpecularSuppress(unittest.TestCase):

    def test_the_highlight_comes_down_by_the_same_amount_per_channel(self):
        """Body colour plus white, minus white: the pixel's chroma — the
        gaps between its channels — is what survives, not its brightness."""
        out = _run([_frame()])[0]
        centre = out[4, 4].astype(int)
        self.assertLess(centre[0], HIGHLIGHT[0])
        np.testing.assert_array_equal(np.diff(centre), np.diff(np.array(HIGHLIGHT)))

    def test_the_skin_around_it_is_untouched(self):
        out = _run([_frame()])[0]
        np.testing.assert_array_equal(out[0, 0], SKIN)
        np.testing.assert_array_equal(out[4, 3], SKIN)

    def test_the_output_is_uint8_bgr_of_the_same_shape(self):
        out = _run([_frame()])[0]
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (8, 8, 3))

    def test_an_rgba_frame_loses_its_alpha_and_nothing_else(self):
        rgba = np.dstack([_frame(), np.full((8, 8), 255, np.uint8)])
        out = _run([rgba])[0]
        self.assertEqual(out.shape, (8, 8, 3))
        np.testing.assert_array_equal(out[0, 0], SKIN)

    def test_amount_scales_the_cut(self):
        full = _run([_frame()])[0][4, 4].astype(float)
        half = _run([_frame()], specular_suppress=0.5)[0][4, 4].astype(float)
        expected = (np.array(HIGHLIGHT, dtype=float) + full) / 2.0
        np.testing.assert_allclose(half, expected, atol=1.0)

    def test_the_threshold_is_the_batch_s_not_the_frame_s(self):
        """One frame all highlight would, measured alone, see nothing to
        cut (every pixel is the mean). In a batch with a normal frame it is
        the outlier, and the cut is decided by the batch."""
        glossy = np.zeros((8, 8, 3), dtype=np.uint8)
        glossy[:] = HIGHLIGHT
        out = _run([_frame(), glossy])
        self.assertLess(int(out[1][0, 0, 0]), HIGHLIGHT[0])
        # And the same pixel value lands in both frames: one threshold.
        np.testing.assert_array_equal(out[0][4, 4], out[1][0, 0])

    def test_the_anchor_frame_is_left_as_the_photograph(self):
        """Its highlights are real. It still counts in the statistics: the
        other frame's cut is the same with or without it declared."""
        out = _run([_frame(), _frame()], anchor=1)
        np.testing.assert_array_equal(out[1], _frame())
        self.assertLess(int(out[0][4, 4, 0]), HIGHLIGHT[0])
        np.testing.assert_array_equal(out[0], _run([_frame(), _frame()])[0])

    def test_outside_the_matte_nothing_is_measured_or_subtracted(self):
        mask = np.ones((8, 8), dtype=np.float32)
        mask[0, :] = 0.0
        image = _frame()
        image[0, :] = (255, 255, 255)   # a bright row the matte excludes
        out = _run([image], mask=mask)[0]
        np.testing.assert_array_equal(out[0, 0], (255, 255, 255))
        # The highlight inside the matte still comes down, and the white
        # row did not pull the threshold up to meet it.
        self.assertLess(int(out[4, 4, 0]), HIGHLIGHT[0])

    def test_an_empty_matte_is_a_no_op(self):
        out = _run([_frame()], mask=np.zeros((8, 8), dtype=np.float32))[0]
        np.testing.assert_array_equal(out, _frame())

    def test_it_refuses_to_run_without_a_matte(self):
        with self.assertRaises(ValueError) as caught:
            run_step("pixel_ops", {"images": [_frame()]}, {"specular_suppress": 1.0})
        self.assertIn("masks", str(caught.exception))

    def test_blur_softens_the_edge_of_the_cut(self):
        """With a blurred excess map the highlight's neighbours take a
        little of the subtraction and the peak keeps a little more."""
        sharp = _run([_frame()])[0].astype(int)
        soft = _run([_frame()], specular_blur=1.0)[0].astype(int)
        self.assertLess(soft[4, 3, 0], SKIN[0])
        self.assertGreater(soft[4, 4, 0], sharp[4, 4, 0])


if __name__ == "__main__":
    unittest.main()
