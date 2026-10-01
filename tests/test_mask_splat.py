"""mask_splat's three modes, on synthetic splat renders.

`mode: threshold` is the port of workflows/api/mask_splat.json at the
`fast helical` settings (filter_size=6, dilation=2): keep only the
near-opaque part of the splat render's alpha, grow it back out a little,
composite over black and bilateral-filter. What is checked here is the
decision that stage exists to make — which pixels survive — on a frame
whose alpha has a fully opaque core, a just-opaque-enough band, a
translucent fringe and a transparent background, all over a texture that
is nowhere black. A wrong threshold comparison or dilation kernel moves
whole bands of that texture across the boundary, which is what fails.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.dataset import Dataset
from tests.helpers import run_step

import pipeline.steps  # noqa: F401

SIZE = 80
CORE = (30, 50)      # opaque core, rows and columns
FRINGE = (6, 74)     # translucent fringe, reaching well past the dilation


def _splat_render(seed=0):
    """A textured frame (40-216 everywhere, so black only ever means
    'masked') and a splat alpha: 1.0 on the left of the core, 0.95 on its
    right — above the default threshold's 1 - 16/255 = 0.937, so still
    kept — 0.9 on the fringe around it, just under, and 0 outside."""
    rng = np.random.default_rng(seed)
    image = rng.integers(40, 216, size=(SIZE, SIZE, 3), dtype=np.uint8)
    alpha = np.zeros((SIZE, SIZE), dtype=np.float32)
    lo, hi = FRINGE
    alpha[lo:hi, lo:hi] = 0.9
    lo, hi = CORE
    mid = (lo + hi) // 2
    alpha[lo:hi, lo:mid] = 1.0
    alpha[lo:hi, mid:hi] = 0.95
    return image, alpha


def _splatted(frames=3):
    renders = [_splat_render(seed) for seed in range(frames)]
    return Dataset(
        images=[image for image, _ in renders],
        image_names=[f"frame_{i + 1:05d}_.png" for i in range(frames)],
        cameras=[None] * frames, points_3d=None, resolution=(SIZE, SIZE),
        masks=[alpha for _, alpha in renders],
    )


def _core_region(grow):
    """The core, grown (positive) or shrunk (negative) by `grow` pixels."""
    region = np.zeros((SIZE, SIZE), dtype=bool)
    lo, hi = CORE
    region[max(lo - grow, 0):hi + grow, max(lo - grow, 0):hi + grow] = True
    return region


class TestMaskSplatThreshold(unittest.TestCase):
    PARAMS = {"filter_size": 6, "dilation": 2}

    @classmethod
    def setUpClass(cls):
        cls.ds = _splatted()
        cls.out = run_step("mask_splat", {"dataset": cls.ds}, dict(cls.PARAMS))["dataset"]

    def test_the_near_opaque_core_survives_and_nothing_else_does(self):
        """Which pixels survive is the decision this stage exists to make.

        Inside the core, clear of the filter's reach, every pixel keeps
        its texture — the 0.95 half included, since the keep-test is
        `alpha >= 1 - threshold/255`. Beyond the dilation and the filter's
        reach every pixel is black — the 0.9 fringe included, which a test
        against 0.5 or a rounded threshold would have kept.
        """
        reach = self.PARAMS["dilation"] + self.PARAMS["filter_size"]
        inside = _core_region(-self.PARAMS["filter_size"])
        outside = ~_core_region(reach)
        for i, image in enumerate(self.out.images):
            with self.subTest(frame=i + 1):
                self.assertTrue(np.all(image[inside].max(axis=1) > 0),
                                "part of the kept core came out black")
                self.assertEqual(int(image[outside].max()), 0,
                                 "something beyond the core survived")

    def test_the_kept_texture_is_only_filtered_not_replaced(self):
        """The bilateral filter smooths; it does not repaint. Deep inside
        the core the output stays close to the frame it was given."""
        inside = _core_region(-self.PARAMS["filter_size"])
        for before, after in zip(self.ds.images, self.out.images):
            err = np.abs(after[inside].astype(int) - before[inside].astype(int))
            self.assertLess(float(err.mean()), 40.0)

    def test_output_is_fully_opaque(self):
        """The ComfyUI graph saves an all-zero MASK, i.e. alpha 255, and the
        port must match or the next denoise pass reads the blacked-out
        region as reference material."""
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
        cls.ds = _splatted()

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
        modes have to disagree on a splat render with a transparent
        background, or the frames denoise_pass2 sees are not the ones the
        gate produced."""
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

    def test_over_a_backdrop_the_room_replaces_the_grey(self):
        """2026-10-01: pass 2's control stands in the studio, frame by frame."""
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[1:3, 1:3] = 1.0
        room = np.full((4, 4, 3), (0, 230, 0), dtype=np.uint8)
        out = run_step("mask_splat", {"dataset": self._dataset(mask), "backdrops": [room]},
                       {"mode": "composite"})["dataset"]
        np.testing.assert_array_equal(out.images[0][0, 0], (0, 230, 0))
        np.testing.assert_array_equal(out.images[0][1, 1], (20, 60, 200))
        with self.assertRaises(ValueError):
            run_step("mask_splat", {"dataset": self._dataset(mask), "backdrops": [room, room]},
                     {"mode": "composite"})

    def test_the_margin_freezes_the_room_past_the_matte(self):
        mask = np.zeros((12, 12), dtype=np.float32)
        mask[5:7, 5:7] = 1.0
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        dataset = Dataset(images=[image], image_names=["frame_00001_.png"], cameras=[None],
                          points_3d=None, resolution=(12, 12), masks=[mask])
        out = run_step("mask_splat", {"dataset": dataset},
                       {"mode": "composite", "inactive_margin_px": 2})["dataset"]
        vace = out.masks[0]
        self.assertEqual(vace[5, 3], 1.0)    # 2 px out: the band
        self.assertEqual(vace[5, 2], 0.0)    # 3 px out: the room
        self.assertEqual(vace[0, 0], 0.0)
        with self.assertRaises(ValueError):
            run_step("mask_splat", {"dataset": dataset},
                     {"mode": "passthrough", "inactive_margin_px": 2})

    def test_it_refuses_a_dataset_with_no_matte(self):
        """Unlike passthrough, this mode has nothing to do without one, and
        silently compositing over an implicit all-1.0 would emit the frames
        unchanged — passthrough under another name."""
        dataset = self._dataset(np.zeros((4, 4), dtype=np.float32))
        dataset.masks = None
        with self.assertRaises(ValueError) as caught:
            run_step("mask_splat", {"dataset": dataset}, {"mode": "composite"})
        self.assertIn("composite", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
