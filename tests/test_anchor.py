"""inject_anchor on an anchored, closed orbit.

`orbit_dataset` is an anchored circular render as body2colmap builds it:
extras["anchor_position"] is the world origin (override_cam_from_mesh puts
the original SAM-3D camera there), and anchor.png is the image to inject.
Crucially, frames 1 and 81 both sit on that position and both carry the
anchor image — the `overlap=1` case the module docstring describes, where
the orbit closes on itself and two cameras occupy the same position. The
port has to find both from the camera positions alone.

**generate_firstlast's warp is not covered here.** Its input is the single
photo SAM-3D-Body was run on, and what the warp does to it is verified on
synthetic data only (TestAnchorBorderColour below pins its border).
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.registry import get_step_class
from tests.helpers import orbit_dataset, run_step

import pipeline.steps  # noqa: F401


class TestInjectAnchorOnAClosedOrbit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = orbit_dataset()
        cls.anchor_position = np.asarray(cls.ds.extras["anchor_position"], dtype=np.float32)

    def _run(self, **overrides):
        inputs = {
            "images": self.ds.images,
            "cameras": self.ds.cameras,
            "anchor_position": self.anchor_position,
            "anchor_image": self.ds.anchor_image,
        }
        inputs.update(overrides)
        return run_step("inject_anchor", inputs, {})

    def test_the_orbit_really_has_a_duplicated_anchor_frame(self):
        """The premise of the test below: two cameras at the anchor, and both
        of those frames already carry the anchor image."""
        positions = np.stack([c.position for c in self.ds.cameras])
        at_anchor = np.flatnonzero(
            np.linalg.norm(positions - self.anchor_position, axis=1) < 1e-6
        )
        self.assertEqual(at_anchor.tolist(), [0, 80])
        for idx in at_anchor:
            np.testing.assert_array_equal(self.ds.images[idx], self.ds.anchor_image)

    def test_injects_into_every_frame_at_the_anchor(self):
        out = self._run()
        injected = [
            i for i, img in enumerate(out["images"])
            if img is self.ds.anchor_image
        ]
        self.assertEqual(injected, [0, 80])

    def test_injected_frames_are_masked_zero(self):
        """Injected frames are reference material, not something to denoise."""
        out = self._run()
        for i, mask in enumerate(out["masks"]):
            if i in (0, 80):
                self.assertTrue(np.all(mask == 0.0), f"frame {i} should be masked 0")
            else:
                self.assertTrue(np.all(mask == 1.0), f"frame {i} should be masked 1")

    def test_survives_reordering_by_matching_on_position(self):
        """The durable key is the position, not the recorded index — the
        whole reason anchor_frame_index is called informational. After a
        rotate_views the index is meaningless but injection must still land
        on the same two cameras."""
        rotated = run_step("rotate_views", 
            {"dataset": self.ds}, {"start_azimuth_deg": 137.0}
        )["dataset"]

        out = run_step("inject_anchor", 
            {
                "images": rotated.images,
                "cameras": rotated.cameras,
                "anchor_position": self.anchor_position,
                "anchor_image": self.ds.anchor_image,
            },
            {},
        )
        injected = [
            i for i, img in enumerate(out["images"]) if img is self.ds.anchor_image
        ]
        self.assertEqual(len(injected), 2)
        for i in injected:
            np.testing.assert_allclose(
                rotated.cameras[i].position, self.anchor_position, atol=1e-6
            )

    def test_no_anchor_passes_through(self):
        out = self._run(anchor_image=None)
        self.assertIs(out["images"], self.ds.images)
        self.assertTrue(all(np.all(m == 1.0) for m in out["masks"]))

        out = self._run(anchor_position=None)
        self.assertIs(out["images"], self.ds.images)

    def test_a_missed_anchor_is_logged_loudly(self):
        """The no-match case is the one that looks like success.

        The batch comes back intact and the run finishes; the only thing
        that changed is that nothing in it is marked as a real photograph.
        helical shipped in exactly that state — an unanchored
        helical re-render, whose nearest camera is 24x the tolerance from
        the anchor — and the port, unlike the ComfyUI node it was ported
        from, said nothing at all. So a WARNING naming the distance is part
        of the step's contract, not decoration.
        """
        far = self.anchor_position + np.array([1e3, 0.0, 0.0], dtype=np.float32)
        with self.assertLogs("pipeline.steps.anchor_stub", level="WARNING") as logged:
            out = self._run(anchor_position=far)
        self.assertIs(out["images"], self.ds.images)
        message = "\n".join(logged.output)
        self.assertIn("NO frame matched", message)
        self.assertIn("closest", message)

    def test_shape_mismatch_raises(self):
        wrong = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self._run(anchor_image=wrong)

    def test_supplied_masks_survive_the_step(self):
        """A step handed somebody else's masks must not manufacture over them.

        The general form of the bug that cost a run's output quality: this
        step took no `masks` input at all and built an all-1.0 batch on
        every call, so wherever it was placed it destroyed whatever the
        mask field was carrying.

        A gradient, not a constant: an all-1.0 mask would pass a test that
        only checked "masks came out non-manufactured".
        """
        h, w = self.ds.images[0].shape[:2]
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, 0)
        supplied = [ramp * (i / len(self.ds.images)) for i in range(len(self.ds.images))]

        out = self._run(masks=[m.copy() for m in supplied])
        injected = [
            i for i, img in enumerate(out["images"]) if img is self.ds.anchor_image
        ]
        self.assertEqual(len(injected), 2, "premise: two frames sit at the anchor")
        for i, mask in enumerate(out["masks"]):
            if i in injected:
                continue
            np.testing.assert_array_equal(
                mask, supplied[i], f"frame {i}'s mask was not passed through"
            )

    def test_an_injected_frame_is_marked_keep_not_denoise(self):
        """0.0, uniform, whether or not masks were supplied.

        `dataset.masks` is the VACE mask everywhere in this pipeline — 1.0
        "synthetic, denoise this", 0.0 "a real photograph, keep it" — and
        an injected frame is by definition the real photograph. Pinned
        because the ComfyUI graph reads the other way round (its MASK is
        inverted and SaveDataset re-inverts on the way to disk), so
        reasoning from the node source instead of the recorded alpha
        produces 1.0 here and tells VACE to regenerate the one real frame
        in the batch.
        """
        h, w = self.ds.images[0].shape[:2]
        for masks in (None, [np.full((h, w), 0.5, np.float32) for _ in self.ds.images]):
            out = self._run(**({} if masks is None else {"masks": masks}))
            injected = [
                i for i, img in enumerate(out["images"]) if img is self.ds.anchor_image
            ]
            self.assertTrue(injected)
            for i in injected:
                with self.subTest(supplied_masks=masks is not None, frame=i):
                    self.assertEqual(float(out["masks"][i].min()), 0.0)
                    self.assertEqual(float(out["masks"][i].max()), 0.0)


class TestMaskThenInject(unittest.TestCase):
    """The stage-2 -> stage-3 chain: mask_splat, then inject_anchor.

    What the chain has to produce: every re-rendered frame masked
    (composited over black and bilateral-filtered) at a uniform VACE alpha
    of 1.0, and the anchor frames the photograph verbatim at a uniform 0.0
    — not composited, not filtered.

    That requires mask_splat to run BEFORE inject_anchor. The other order
    was shipped, and it put inject_anchor where dataset.masks is carrying
    the splat render's per-pixel alpha, so the alpha mask_splat exists to
    threshold was overwritten with all-1.0 and the stage silently became a
    bilateral filter. `test_the_other_order_masks_nothing_and_blacks_out_the_photo`
    shows what that does; the YAML-level guard is in test_workflows.py.
    """

    def _dataset(self):
        """A splatted stage: every frame a render of a textured subject on a
        textured background, its alpha the splat's (1 on the subject, 0 off
        it) — including the two frames at the anchor, which the re-render
        drew like any other and inject_anchor has to replace."""
        ds = orbit_dataset()
        h, w = ds.images[0].shape[:2]
        rng = np.random.default_rng(3)
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0
        ds.images = [rng.integers(40, 216, size=(h, w, 3), dtype=np.uint8)
                     for _ in ds.images]
        ds.masks = [alpha.copy() for _ in ds.images]
        return ds, alpha

    def _anchor_frames(self, ds):
        positions = np.stack([c.position for c in ds.cameras])
        return np.flatnonzero(np.linalg.norm(
            positions - np.asarray(ds.extras["anchor_position"]), axis=1) < 1e-6).tolist()

    def _inject(self, ds, masks=None):
        inputs = {
            "images": ds.images,
            "cameras": ds.cameras,
            "anchor_position": ds.extras["anchor_position"],
            "anchor_image": ds.anchor_image,
        }
        if masks is not None:
            inputs["masks"] = masks
        return run_step("inject_anchor", inputs, {"tolerance_pct": 0.1})

    def _mask(self, ds):
        return run_step("mask_splat", {"dataset": ds},
                        {"filter_size": 6, "dilation": 2})["dataset"]

    def test_the_anchor_frame_is_the_photo_verbatim(self):
        """Byte-exact, at alpha 0 — the single check that pins both the
        ordering and the mask convention at once. Composite it over black
        or bilateral-filter it and the bytes stop matching; mark it 1.0 and
        denoise_pass2 regenerates the only real frame in the batch."""
        ds, _alpha = self._dataset()
        anchors = self._anchor_frames(ds)
        self.assertEqual(anchors, [0, 80], "premise: two frames sit at the anchor")

        masked = self._mask(ds)
        out = self._inject(masked, masks=masked.masks)
        for i, (image, mask) in enumerate(zip(out["images"], out["masks"])):
            with self.subTest(frame=i + 1):
                # Uniform per frame: a VACE flag, not a matte.
                self.assertEqual(float(mask.min()), float(mask.max()))
                if i in anchors:
                    np.testing.assert_array_equal(image, ds.anchor_image)
                    self.assertEqual(float(mask.max()), 0.0)
                else:
                    self.assertEqual(float(mask.max()), 1.0)
                    # ...and masked: the background around the subject is gone.
                    self.assertEqual(int(image[0, 0].max()), 0)

    def test_the_other_order_masks_nothing_and_blacks_out_the_photo(self):
        """The same two steps the wrong way round.

        inject_anchor first overwrites the splat alpha with an all-1.0
        batch, so mask_splat's keep-test passes on every pixel and nothing
        is ever blacked out — and the anchor frame, injected first with a
        uniform 0.0, fails the keep-test everywhere and comes out black.
        """
        ds, alpha = self._dataset()
        anchors = self._anchor_frames(ds)

        right = self._mask(ds)
        out = self._inject(ds)
        ds.images, ds.masks = out["images"], out["masks"]
        wrong = self._mask(ds)

        for i in range(len(wrong.images)):
            if i in anchors:
                continue
            with self.subTest(frame=i + 1):
                kept_right = float((right.images[i].max(axis=2) > 8).mean())
                kept_wrong = float((wrong.images[i].max(axis=2) > 8).mean())
                # Right: roughly the subject survives. Wrong: the whole frame.
                self.assertLess(kept_right, float(alpha.mean()) + 0.35)
                self.assertGreater(kept_wrong, 0.9)
        for i in anchors:
            self.assertEqual(int(wrong.images[i].max()), 0)


class TestAnchorBorderColour(unittest.TestCase):
    """The colour generate_firstlast fills around the warped photo.

    The warp maps a photo taken at one focal length into the orbit
    camera's framing, which leaves border. That border travels all the way
    to denoise_pass2 as part of the anchor frame, sitting among renders on
    a mid-grey background — so white, the step's old effective value in
    helical.yaml, is the largest possible disagreement with
    its neighbours.

    0.5 is pinned rather than a literal 127 or 128 because the reference
    ComfyUI run showed BOTH numbers and 0.5 is what produces them: the
    renderer truncates (`int(bg*255)` = 127) and this step rounds
    (`round(bg*255)` = 128). That run had its mesh frames at 127 and its
    anchor.png at 128, which is the evidence that 0.5 is the value the
    reference pipeline used.
    """

    def _border(self, bg_color):
        from body2colmap.camera import Camera

        camera = Camera(
            focal_length=(400.0, 400.0), image_size=(600, 900),
            principal_point=(300.0, 450.0),
            position=np.zeros(3, dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        out = run_step("generate_firstlast", 
            {
                "image": np.full((200, 200, 3), 200, np.uint8),
                "camera": camera,
                "original_focal_length": 800.0,
                "render_size": (600, 900),
                "bg_color": bg_color,
            },
            {},
        )["warped_image"]
        return [int(v) for v in out[0, 0]]

    def test_half_grey_paints_the_recorded_anchor_border(self):
        self.assertEqual(self._border((0.5, 0.5, 0.5)), [128, 128, 128])

    def test_the_step_still_defaults_to_white(self):
        """Unchanged: the default belongs to callers that render on white.
        helical.yaml overrides it via the render step's
        bg_color, which render.py publishes as image_warp["bg_color"]."""
        self.assertEqual(self._border((1.0, 1.0, 1.0)), [255, 255, 255])


class TestAnchorBorderFromTheRoom(unittest.TestCase):
    """2026-10-01: the warped photo's border filled with the studio the
    re-outlined render drew, instead of the flat bg_color."""

    def setUp(self):
        from body2colmap.camera import Camera

        self.camera = Camera(
            focal_length=(400.0, 400.0), image_size=(60, 90),
            principal_point=(30.0, 45.0),
            position=np.zeros(3, dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        self.warp = run_step("generate_firstlast", {
            "image": np.full((40, 40, 3), 200, np.uint8),
            "camera": self.camera,
            "original_focal_length": 800.0,
            "render_size": (60, 90),
            "bg_color": (0.5, 0.5, 0.5),
        }, {})

    def test_the_warp_reports_where_the_photo_reaches(self):
        coverage = self.warp["warped_coverage"]
        self.assertEqual(coverage.shape, (90, 60))
        self.assertEqual(coverage[0, 0], 0.0)
        self.assertEqual(coverage[45, 30], 1.0)
        self.assertEqual(tuple(self.warp["border_color"]), (128, 128, 128))

    def _inject(self, with_room=False, **extra):
        room = np.zeros((90, 60, 3), np.uint8)
        room[..., 1] = 230
        if with_room:
            extra.update(backdrops=[room, room],
                         anchor_coverage=self.warp["warped_coverage"],
                         border_color=self.warp["border_color"])
        return run_step("inject_anchor", {
            "images": [np.zeros((90, 60, 3), np.uint8)] * 2,
            "cameras": [self.camera, self.camera],
            "anchor_position": np.zeros(3, dtype=np.float32),
            "anchor_image": self.warp["warped_image"],
            **extra,
        }, {}), room

    def test_the_border_is_the_room_and_the_photo_is_untouched(self):
        out, _ = self._inject(with_room=True)
        frame = out["images"][0]
        self.assertEqual(tuple(frame[0, 0]), (0, 230, 0))
        self.assertEqual(tuple(frame[45, 30]), (200, 200, 200))
        # The resampled edge is a blend of photo and room — no grey left in it.
        edge = self.warp["warped_coverage"]
        partial = (edge > 0.0) & (edge < 1.0)
        if partial.any():
            c = edge[partial][:, None]
            np.testing.assert_allclose(
                frame[partial].astype(float),
                200 * c + np.array([0, 230, 0]) * (1 - c), atol=1.0)
        self.assertTrue(np.all(out["masks"][0] == 0.0))

    def test_without_the_room_the_photo_goes_in_whole(self):
        out, _ = self._inject(anchor_coverage=self.warp["warped_coverage"])
        self.assertIs(out["images"][0], self.warp["warped_image"])


if __name__ == "__main__":
    unittest.main()
