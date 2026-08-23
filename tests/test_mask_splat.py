"""mask_splat against the recorded ComfyUI output of the same stage.

cyber_6f/splatted -> cyber_6f/masked_splatted is a real run of
workflows/api/mask_splat.json at the `fast helical` settings
(filter_size=6, dilation=2), which makes this a golden-output test rather
than a self-consistency one: the ported step is compared against frames
produced by the ComfyUI graph it replaces.

The port is not bit-exact (see pipeline/steps/mask_splat.py's docstring):
the surviving *mask* matches exactly, while the filtered pixel values differ
by a mean of ~0.25/255 with a max around 15, concentrated at mask edges.
The tolerances below are set just above the measured values so a real
regression — a wrong threshold comparison or dilation kernel, both of which
moved the max error into the hundreds while fitting this — fails loudly.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from tests.helpers import require_stage

import pipeline.steps  # noqa: F401


class TestMaskSplatGolden(unittest.TestCase):
    FRAMES = (1, 20, 41, 60, 81)

    @classmethod
    def setUpClass(cls):
        cls.src, cls.gold_dir = require_stage("splatted", "masked_splatted")
        cls.ds = Dataset.from_disk(cls.src)
        cls.out = get_step_class("mask_splat")().run(
            {"dataset": cls.ds}, {"filter_size": 6, "dilation": 2}
        )["dataset"]

    def _gold(self, n):
        img = cv2.imread(str(self.gold_dir / f"frame_{n:05d}_.png"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(img, f"missing golden frame {n}")
        return img

    def test_matches_recorded_output(self):
        for n in self.FRAMES:
            with self.subTest(frame=n):
                gold = self._gold(n)[:, :, :3].astype(np.int32)
                ours = self.out.images[n - 1].astype(np.int32)
                err = np.abs(ours - gold)
                self.assertLess(err.mean(), 0.5, "mean absolute error too high")
                self.assertLess(err.max(), 30, "max absolute error too high")

    def test_surviving_region_matches(self):
        """Which pixels survive is the decision this stage exists to make.

        Compared as "is this pixel black", the two agree on ~99.85% of
        pixels. Every disagreement is a near-black pixel: values of 1-3 the
        bilateral filter left just above or below zero, either hugging the
        mask boundary or sitting inside genuinely black image content. So
        the assertion is not an exact match but that no disagreement is
        anything but sub-perceptually dark — a wrong threshold comparison
        or dilation kernel breaks it immediately, since those move bands of
        real image content across the boundary (max error went to 140 and
        200 respectively while fitting this).
        """
        for n in self.FRAMES:
            with self.subTest(frame=n):
                gold = self._gold(n)[:, :, :3]
                ours = self.out.images[n - 1]
                gold_black = gold.max(axis=2) == 0
                ours_black = ours.max(axis=2) == 0
                disagree = np.logical_xor(gold_black, ours_black)

                self.assertLess(disagree.mean(), 0.005)
                self.assertLessEqual(int(ours[disagree & gold_black].max(initial=0)), 8)
                self.assertLessEqual(int(gold[disagree & ours_black].max(initial=0)), 8)

                # Away from the boundary, disagreements are pure rounding.
                kept = (~ours_black).astype(np.uint8) * 255
                near_edge = cv2.morphologyEx(
                    kept, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)
                ) > 0
                far = disagree & ~near_edge
                self.assertLessEqual(int(ours[far].max(initial=0)), 2)
                self.assertLessEqual(int(gold[far].max(initial=0)), 2)

    def test_output_is_fully_opaque(self):
        """The ComfyUI graph saves an all-zero MASK, i.e. alpha 255 — the
        recorded frames confirm it, and the port must match or the next
        denoise pass reads the blacked-out region as reference material."""
        for n in self.FRAMES:
            self.assertEqual(self._gold(n)[:, :, 3].min(), 255)
        for mask in self.out.masks:
            self.assertTrue(np.all(mask == 1.0))

    def test_dilation_zero_is_allowed(self):
        """`tiered`'s second mask_splat pass uses dilation=0."""
        out = get_step_class("mask_splat")().run(
            {"dataset": self.ds}, {"filter_size": 4, "dilation": 0}
        )["dataset"]
        self.assertEqual(len(out.images), len(self.ds.images))

    def test_requires_masks(self):
        no_masks = Dataset(
            images=self.ds.images[:1], image_names=self.ds.image_names[:1],
            cameras=self.ds.cameras[:1], points_3d=self.ds.points_3d,
            resolution=self.ds.resolution, masks=None,
        )
        with self.assertRaises(ValueError):
            get_step_class("mask_splat")().run({"dataset": no_masks}, {})


if __name__ == "__main__":
    unittest.main()
