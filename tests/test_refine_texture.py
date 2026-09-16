"""steps/refine_texture.py's pure helpers (the mask logic the loop rests on)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.steps.refine_texture import feather_repaint, fit_reference, repaint_mask, view_order


class TestRepaintMask(unittest.TestCase):
    def test_claims_where_the_view_beats_the_best_or_nothing_painted(self):
        alpha = np.full((4, 4), 255, np.uint8)
        alpha[0, :] = 0                       # background row
        cos = np.ones((4, 4), np.float32)
        dens = np.full((4, 4), 2.0, np.float32)
        best = np.zeros((4, 4), np.float32)
        best[1, :] = 10.0                     # painted better already
        best[2, :] = np.inf                   # protected
        best[3, 0] = 1.0                      # weight 2 > 1.5 * 1
        claim, painted = repaint_mask(alpha, cos, dens, best, power=4.0, gain=1.5, dilate=0)
        self.assertFalse(claim[0].any())
        self.assertFalse(claim[1].any())
        self.assertFalse(claim[2].any())
        self.assertTrue(claim[3].all())
        self.assertTrue((painted == claim).all())

    def test_dilation_stays_inside_the_subject(self):
        alpha = np.zeros((9, 9), np.uint8)
        alpha[2:7, 2:7] = 255
        cos = np.ones((9, 9), np.float32)
        dens = np.ones((9, 9), np.float32)
        best = np.full((9, 9), np.inf, np.float32)
        best[4, 4] = 0.0
        claim, painted = repaint_mask(alpha, cos, dens, best, 4.0, 1.5, dilate=3)
        self.assertEqual(int(claim.sum()), 1)
        self.assertEqual(int(painted.sum()), 25)  # the 7x7 ring clipped to the 5x5 subject
        self.assertFalse(painted[alpha == 0].any())


class TestFeather(unittest.TestCase):
    def test_outside_the_mask_the_render_is_kept_exactly(self):
        render = np.full((32, 32, 3), 100, np.uint8)
        repaint = np.full((32, 32, 3), 200, np.uint8)
        alpha = np.full((32, 32), 255, np.uint8)
        mask = np.zeros((32, 32), np.uint8)
        mask[8:24, 8:24] = 255
        out = feather_repaint(repaint, render, mask, alpha, feather=4)
        self.assertEqual(int(out[0, 0, 0]), 100)
        self.assertGreaterEqual(int(out[16, 16, 0]), 195)  # the blur reaches the centre of a 16 px mask faintly
        self.assertTrue(100 < int(out[9, 16, 0]) <= 200)  # the edge blends
        out0 = feather_repaint(repaint, render, mask, alpha, feather=0)
        self.assertEqual(int(out0[8, 8, 0]), 200)


class TestOrderAndRefs(unittest.TestCase):
    def test_head_first_by_default(self):
        order = view_order([0, 180], [0, 45], True)
        self.assertEqual(order, [("head", 0.0), ("head", 45.0), ("body", 0.0), ("body", 180.0)])
        self.assertEqual(view_order([0], [0], False)[0][0], "body")

    def test_reference_fits_the_pixel_budget_in_multiples_of_16(self):
        im = np.zeros((1536, 768, 3), np.uint8)
        out = fit_reference(im, 300000)
        self.assertLessEqual(out.shape[0] * out.shape[1], 300000)
        self.assertEqual(out.shape[0] % 16, 0)
        self.assertEqual(out.shape[1] % 16, 0)
        self.assertEqual(fit_reference(np.zeros((64, 48, 3), np.uint8), 300000).shape[:2], (64, 48))


if __name__ == "__main__":
    unittest.main()
