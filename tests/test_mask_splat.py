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
from tests.helpers import require_stage, run_step

import pipeline.steps  # noqa: F401


class TestMaskSplatGolden(unittest.TestCase):
    FRAMES = (1, 20, 41, 60, 81)

    @classmethod
    def setUpClass(cls):
        cls.src, cls.gold_dir = require_stage("splatted", "masked_splatted")
        cls.ds = Dataset.from_disk(cls.src)
        cls.out = run_step("mask_splat", 
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
        out = run_step("mask_splat", 
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
            run_step("mask_splat", {"dataset": no_masks}, {})

    def test_threshold_is_the_declared_default(self):
        """The golden comparison above runs at the step's default, so it is
        only a golden test while that default is `threshold`. Every shipped
        workflow overrides it to passthrough, which is exactly the way a
        default gets changed to match without anyone noticing."""
        default = get_step_class("mask_splat").declared_params()["mode"].default
        self.assertEqual(default, "threshold")


class TestMaskSplatPassthrough(unittest.TestCase):
    """`mode: passthrough` — the shipped mode, behind a confidence render.

    render_splat's `confidence` already decided which pixels survive, in 3-D
    and once, and composited them over the cull colour. Re-running the old
    alpha cut on top of that is wrong rather than redundant: it would
    composite grey frames over black and smear the gate's soft edge. What is
    left for this step is the half that is still needed — turning the
    per-pixel splat alpha in dataset.masks into the per-frame all-1.0 VACE
    batch that denoise_pass2 reads and inject_anchor writes its 0.0 into.
    """

    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("splatted"))

    def _run(self, dataset):
        return run_step("mask_splat", {"dataset": dataset}, {"mode": "passthrough"})["dataset"]

    def test_the_frames_come_through_untouched(self):
        out = self._run(self.ds)
        self.assertEqual(len(out.images), len(self.ds.images))
        for before, after in zip(self.ds.images, out.images):
            np.testing.assert_array_equal(after, before)

    def test_the_masks_are_still_the_vace_batch(self):
        """Not a pass-through of dataset.masks: that field changes meaning
        here — in as the splat's per-pixel alpha, out as the per-frame
        'synthetic, regenerate this' flag."""
        out = self._run(self.ds)
        h, w = self.ds.images[0].shape[:2]
        self.assertEqual(len(out.masks), len(out.images))
        for mask in out.masks:
            self.assertEqual(mask.shape, (h, w))
            self.assertEqual(mask.dtype, np.float32)
            self.assertTrue(np.all(mask == 1.0))

    def test_it_does_not_need_the_splat_alpha(self):
        """A confidence render's alpha IS the gate, already applied to the
        RGB by the rasteriser, so this mode never reads it — and must not
        refuse a dataset that arrived without one."""
        no_masks = Dataset(
            images=self.ds.images[:2], image_names=self.ds.image_names[:2],
            cameras=self.ds.cameras[:2], points_3d=self.ds.points_3d,
            resolution=self.ds.resolution, masks=None,
        )
        out = self._run(no_masks)
        self.assertEqual(len(out.masks), 2)

    def test_it_differs_from_the_threshold_path(self):
        """Guards against a passthrough that quietly still filters — the two
        modes have to disagree on a real splat render, or the frames
        denoise_pass2 sees are not the ones the gate produced."""
        thresholded = run_step(
            "mask_splat", {"dataset": self.ds}, {"filter_size": 6, "dilation": 2}
        )["dataset"]
        passed = self._run(self.ds)
        self.assertFalse(
            np.array_equal(thresholded.images[0], passed.images[0]),
            "passthrough produced the thresholded frame",
        )


class TestMaskSplatComposite(unittest.TestCase):
    """`mode: composite` — the shipped mode since 2026-09-05.

    The re-render is made on BLACK now and the matte comes from rmbg rather
    than from the render's own alpha, so the two halves of the old subgraph
    have separated: the confidence gate decides what is MISSING (culled
    pixels come through black, a hole in the subject the next denoise
    repaints) and this decides what is SUBJECT, laying it over the mid grey
    the rest of the batch — the warped anchor photo's border included —
    grounds on.

    Synthetic rather than golden: there is no recorded ComfyUI run of a
    stage that did not exist there.
    """

    def _dataset(self, mask):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:] = (20, 60, 200)
        return Dataset(
            images=[image], image_names=["frame_00001_.png"], cameras=[None],
            points_3d=None, resolution=(4, 4), masks=[mask],
        )

    def _run(self, mask, **params):
        dataset = self._dataset(mask)
        return run_step(
            "mask_splat", {"dataset": dataset}, {"mode": "composite", **params}
        )["dataset"]

    def test_the_background_becomes_the_flat_colour(self):
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[1:3, 1:3] = 1.0
        out = self._run(mask)
        np.testing.assert_array_equal(out.images[0][0, 0], (128, 128, 128))

    def test_the_subject_is_left_alone(self):
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[1:3, 1:3] = 1.0
        out = self._run(mask)
        np.testing.assert_array_equal(out.images[0][1, 1], (20, 60, 200))

    def test_a_soft_edge_blends_rather_than_cuts(self):
        """The whole reason for using a matte measured against the frames:
        its edge is already right, so there is nothing to threshold and
        nothing to bilateral-filter back into shape."""
        mask = np.full((4, 4), 0.5, dtype=np.float32)
        out = self._run(mask)
        np.testing.assert_array_equal(out.images[0][0, 0], (74, 94, 164))

    def test_the_colour_is_bgr_the_way_the_frames_are(self):
        """`bg_color` is RGB in [0,1], like every other step's, and the
        frames are cv2 BGR — a swap here would tint every background."""
        out = self._run(np.zeros((4, 4), dtype=np.float32), bg_color=[1.0, 0.0, 0.0])
        np.testing.assert_array_equal(out.images[0][0, 0], (0, 0, 255))

    def test_the_masks_are_still_the_vace_batch(self):
        """Same as every other mode: in as a matte, out as the per-frame
        'synthetic, regenerate this' flag denoise_pass2 reads."""
        out = self._run(np.zeros((4, 4), dtype=np.float32))
        self.assertTrue(np.all(out.masks[0] == 1.0))
        self.assertEqual(out.masks[0].shape, (4, 4))

    def test_it_refuses_a_dataset_with_no_matte(self):
        """Unlike passthrough, this mode has nothing to do without one, and
        silently compositing over an implicit all-1.0 would emit the frames
        unchanged — passthrough under another name."""
        dataset = self._dataset(np.zeros((4, 4), dtype=np.float32))
        dataset.masks = None
        with self.assertRaises(ValueError) as caught:
            run_step("mask_splat", {"dataset": dataset}, {"mode": "composite"})
        self.assertIn("composite", str(caught.exception))


class TestMaskSplatSpecularSuppress(unittest.TestCase):
    """`specular_suppress` on the composite mode — 2026-09-10.

    The frames denoise_pass2 sees are a re-render of a splat trained on
    pass 1's output, highlights and all; lit a second time those blow out.
    The suppression is the dichromatic model's specular-free image: the
    minimum channel's excess over `mean + eta * std` of the batch's matte
    pixels, taken off all three channels equally.

    A 8x8 patch of one skin colour with a single brighter, whiter pixel in
    the middle of it — body colour plus a white term — is the whole model.
    """

    SKIN = (120, 160, 220)          # BGR
    HIGHLIGHT = (200, 220, 250)     # the same skin plus ~80 of white

    def _frame(self, highlight_at=(4, 4)):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:] = self.SKIN
        image[highlight_at] = self.HIGHLIGHT
        return image

    def _dataset(self, images, mask=None):
        if mask is None:
            mask = np.ones((8, 8), dtype=np.float32)
        return Dataset(
            images=images, image_names=[f"frame_{i:05d}_.png" for i in range(len(images))],
            cameras=[None] * len(images), points_3d=None, resolution=(8, 8),
            masks=[mask] * len(images),
        )

    def _run(self, images, mask=None, **params):
        return run_step(
            "mask_splat", {"dataset": self._dataset(images, mask)},
            {"mode": "composite", "specular_suppress": 1.0, "specular_blur": 0.0, **params},
        )["dataset"].images

    def test_off_is_the_composite_as_it_was(self):
        out = run_step(
            "mask_splat", {"dataset": self._dataset([self._frame()])},
            {"mode": "composite"},
        )["dataset"].images[0]
        np.testing.assert_array_equal(out, self._frame())

    def test_the_highlight_comes_down_by_the_same_amount_per_channel(self):
        """Body colour plus white, minus white: the pixel's chroma — the
        gaps between its channels — is what survives, not its brightness."""
        out = self._run([self._frame()])[0]
        centre = out[4, 4].astype(int)
        self.assertLess(centre[0], self.HIGHLIGHT[0])
        gaps = np.diff(centre)
        np.testing.assert_array_equal(gaps, np.diff(np.array(self.HIGHLIGHT)))

    def test_the_skin_around_it_is_untouched(self):
        out = self._run([self._frame()])[0]
        np.testing.assert_array_equal(out[0, 0], self.SKIN)
        np.testing.assert_array_equal(out[4, 3], self.SKIN)

    def test_amount_scales_the_cut(self):
        full = self._run([self._frame()])[0][4, 4].astype(float)
        half = self._run([self._frame()], specular_suppress=0.5)[0][4, 4].astype(float)
        expected = (np.array(self.HIGHLIGHT, dtype=float) + full) / 2.0
        np.testing.assert_allclose(half, expected, atol=1.0)

    def test_the_threshold_is_the_batch_s_not_the_frame_s(self):
        """One frame all highlight would, measured alone, see nothing to
        cut (every pixel is the mean). In a batch with a normal frame it is
        the outlier, and the cut is decided by the batch."""
        plain = self._frame()
        glossy = np.zeros((8, 8, 3), dtype=np.uint8)
        glossy[:] = self.HIGHLIGHT
        out = self._run([plain, glossy])
        self.assertLess(int(out[1][0, 0, 0]), self.HIGHLIGHT[0])
        # And the same pixel value lands in both frames: one threshold.
        np.testing.assert_array_equal(out[0][4, 4], out[1][0, 0])

    def test_outside_the_matte_nothing_is_subtracted(self):
        """The statistics come from the matte and so does the cut: the
        black the re-render culls to and the grey the frame is about to
        be laid on are not skin."""
        mask = np.ones((8, 8), dtype=np.float32)
        mask[0, :] = 0.0
        image = self._frame()
        image[0, :] = (255, 255, 255)   # a bright row the matte excludes
        out = self._run([image], mask=mask)[0]
        np.testing.assert_array_equal(out[0, 0], (128, 128, 128))
        # The highlight inside the matte still comes down, and the white
        # row did not pull the threshold up to meet it.
        self.assertLess(int(out[4, 4, 0]), self.HIGHLIGHT[0])

    def test_an_empty_matte_is_a_no_op(self):
        mask = np.zeros((8, 8), dtype=np.float32)
        out = self._run([self._frame()], mask=mask)[0]
        np.testing.assert_array_equal(out[4, 4], (128, 128, 128))

    def test_blur_softens_the_edge_of_the_cut(self):
        """With a blurred excess map the highlight's neighbours take a
        little of the subtraction and the peak keeps a little more."""
        sharp = self._run([self._frame()])[0].astype(int)
        soft = self._run([self._frame()], specular_blur=1.0)[0].astype(int)
        self.assertLess(soft[4, 3, 0], self.SKIN[0])
        self.assertGreater(soft[4, 4, 0], sharp[4, 4, 0])


if __name__ == "__main__":
    unittest.main()
