"""face_priority_weights: the face cap wins.

The step renders the face splat's coverage from a batch's cameras and turns
it into per-pixel loss weights that fade those views out over the face —
in full within the cap's radius of the anchor, fading beyond it, and not
at all for a view the cap has no evidence for. These drive it with the
rasteriser stubbed (a disc of coverage per camera) and pin the arithmetic,
the angular gate, the mask fold and the pass-through cases a workflow
relies on when the face branch is off.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from pipeline.steps import face_priority
from pipeline.steps.face_priority import angular_attenuation, weight_from_coverage
from tests.helpers import run_step

import pipeline.steps  # noqa: F401

W, H = 16, 12
RADIUS = 5.0


def _camera(angle_deg: float, *, size=(W, H)):
    """A camera on a ring of radius 5 about the origin, `angle_deg` from +z."""
    from body2colmap.camera import Camera

    theta = np.radians(angle_deg)
    return Camera(
        focal_length=(8.0, 8.0),
        image_size=size,
        principal_point=(size[0] / 2.0, size[1] / 2.0),
        position=np.array([RADIUS * np.sin(theta), 0.0, RADIUS * np.cos(theta)],
                          dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
    )


def _disc(height=H, width=W):
    """Coverage: a solid block in the middle of the frame, nothing outside."""
    cov = np.zeros((height, width), np.float32)
    cov[3:9, 5:11] = 1.0
    return cov


def _stub_coverage(calls):
    def fake(splat_path, cameras, *, width, height, render_path):
        calls.append({"splat_path": splat_path, "cameras": list(cameras),
                      "width": width, "height": height, "render_path": render_path})
        return [_disc(height, width) for _ in cameras]
    return fake


def _inputs(angles, **extra):
    cameras = [_camera(a) for a in angles]
    return {
        "cameras": cameras,
        "splat_path": "/nonexistent/face.ply",
        "anchor_cameras": [_camera(0.0)],
        "anchor_frame_index": 0,
        "splat_center": [0.0, 0.0, 0.0],
        **extra,
    }


class TestArithmetic(unittest.TestCase):
    def test_attenuation_is_full_inside_the_cap_and_fades_beyond_it(self):
        self.assertEqual(angular_attenuation(0.0, 30.0, 15.0), 1.0)
        self.assertEqual(angular_attenuation(30.0, 30.0, 15.0), 1.0)
        self.assertAlmostEqual(angular_attenuation(37.5, 30.0, 15.0), 0.5)
        self.assertEqual(angular_attenuation(45.0, 30.0, 15.0), 0.0)
        self.assertEqual(angular_attenuation(90.0, 30.0, 15.0), 0.0)

    def test_a_zero_fade_is_a_hard_edge(self):
        self.assertEqual(angular_attenuation(30.0, 30.0, 0.0), 1.0)
        self.assertEqual(angular_attenuation(30.1, 30.0, 0.0), 0.0)

    def test_the_weight_is_one_minus_strength_at_full_coverage(self):
        weight = weight_from_coverage(_disc(), strength=0.9, attenuation=1.0, feather_px=0.0)
        self.assertEqual(weight.dtype, np.float32)
        self.assertAlmostEqual(float(weight[5, 8]), 0.1, places=6)
        self.assertEqual(float(weight[0, 0]), 1.0)

    def test_attenuation_scales_the_strength(self):
        weight = weight_from_coverage(_disc(), strength=0.9, attenuation=0.5, feather_px=0.0)
        self.assertAlmostEqual(float(weight[5, 8]), 0.55, places=6)

    def test_a_uint8_coverage_is_read_as_a_mask(self):
        cov = (_disc() * 255).astype(np.uint8)
        weight = weight_from_coverage(cov, strength=1.0, attenuation=1.0, feather_px=0.0)
        self.assertEqual(float(weight[5, 8]), 0.0)

    def test_feathering_ramps_the_edge_both_ways(self):
        hard = weight_from_coverage(_disc(), strength=1.0, attenuation=1.0, feather_px=0.0)
        soft = weight_from_coverage(_disc(), strength=1.0, attenuation=1.0, feather_px=1.5)
        # Just inside the block the weight rises off zero; just outside it
        # dips below one. The far corner is untouched either way.
        self.assertEqual(float(hard[3, 5]), 0.0)
        self.assertGreater(float(soft[3, 5]), 0.0)
        self.assertEqual(float(hard[2, 5]), 1.0)
        self.assertLess(float(soft[2, 5]), 1.0)
        self.assertAlmostEqual(float(soft[0, 0]), 1.0, places=3)


class TestStep(unittest.TestCase):
    def setUp(self):
        self.calls = []
        patcher = mock.patch.object(face_priority, "_render_coverage",
                                    _stub_coverage(self.calls))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_views_inside_the_cap_yield_and_views_outside_do_not(self):
        out = run_step("face_priority_weights", _inputs([0.0, 20.0, 37.5, 60.0, 180.0]),
                       {"feather_px": 0.0})
        weights = out["weights"]
        self.assertEqual(len(weights), 5)
        self.assertNotIn("masks", out)
        centre = [float(w[5, 8]) for w in weights]
        self.assertAlmostEqual(centre[0], 0.1, places=5)   # the anchor's own view
        self.assertAlmostEqual(centre[1], 0.1, places=5)   # inside the 30 deg cap
        self.assertAlmostEqual(centre[2], 0.55, places=5)  # halfway through the fade
        self.assertEqual(centre[3], 1.0)                   # past the fade
        self.assertEqual(centre[4], 1.0)                   # the back of the head
        for w in weights:
            self.assertEqual(w.dtype, np.float32)
            self.assertEqual(w.shape, (H, W))
            self.assertEqual(float(w[0, 0]), 1.0, "outside the coverage nothing yields")

    def test_the_coverage_is_rendered_from_the_batch_s_cameras_at_their_size(self):
        inputs = _inputs([0.0, 45.0])
        run_step("face_priority_weights", inputs)
        self.assertEqual(len(self.calls), 1)
        call = self.calls[0]
        self.assertEqual(call["splat_path"], "/nonexistent/face.ply")
        self.assertEqual(len(call["cameras"]), 2)
        self.assertEqual((call["width"], call["height"]), (W, H))
        self.assertIsNone(call["render_path"])

    def test_the_anchor_is_read_live_from_the_camera_list(self):
        """The photograph's camera is `anchor_cameras[anchor_frame_index]`,
        not the recorded position: refine_cameras moves the list and only
        republishes the record afterwards."""
        inputs = _inputs([0.0, 90.0],
                         anchor_cameras=[_camera(0.0), _camera(90.0)],
                         anchor_frame_index=1,
                         anchor_position=[0.0, 0.0, RADIUS])
        out = run_step("face_priority_weights", inputs, {"feather_px": 0.0})
        # The cap is now about the 90-degree view, so THAT view yields and
        # the 0-degree one is 90 degrees off and does not.
        self.assertEqual(float(out["weights"][0][5, 8]), 1.0)
        self.assertAlmostEqual(float(out["weights"][1][5, 8]), 0.1, places=5)

    def test_the_recorded_position_is_the_fallback(self):
        inputs = _inputs([0.0, 90.0])
        del inputs["anchor_cameras"]
        del inputs["anchor_frame_index"]
        inputs["anchor_position"] = [RADIUS * 1.0, 0.0, 0.0]  # the 90-degree view
        out = run_step("face_priority_weights", inputs, {"feather_px": 0.0})
        self.assertEqual(float(out["weights"][0][5, 8]), 1.0)
        self.assertAlmostEqual(float(out["weights"][1][5, 8]), 0.1, places=5)

    def test_no_anchor_at_all_is_refused(self):
        inputs = _inputs([0.0])
        del inputs["anchor_cameras"]
        del inputs["anchor_frame_index"]
        with self.assertRaisesRegex(ValueError, "anchor"):
            run_step("face_priority_weights", inputs)

    def test_no_pivot_is_refused(self):
        inputs = _inputs([0.0])
        del inputs["splat_center"]
        with self.assertRaisesRegex(ValueError, "splat_center"):
            run_step("face_priority_weights", inputs)

    def test_the_orbit_target_is_the_pivot_s_fallback(self):
        inputs = _inputs([0.0])
        del inputs["splat_center"]
        inputs["orbit_target"] = np.zeros(3, np.float32)
        out = run_step("face_priority_weights", inputs, {"feather_px": 0.0})
        self.assertAlmostEqual(float(out["weights"][0][5, 8]), 0.1, places=5)

    def test_strength_one_masks_the_face_out_entirely(self):
        out = run_step("face_priority_weights", _inputs([0.0]),
                       {"strength": 1.0, "feather_px": 0.0})
        self.assertEqual(float(out["weights"][0][5, 8]), 0.0)

    def test_masks_come_back_folded(self):
        """A masked batch (the stage-1 shells): its mask is its per-pixel
        weight already, so the yield goes into the mask."""
        mask = np.full((H, W), 0.5, np.float32)
        out = run_step("face_priority_weights", _inputs([0.0], masks=[mask]),
                       {"feather_px": 0.0})
        folded = out["masks"][0]
        self.assertEqual(folded.dtype, np.float32)
        self.assertAlmostEqual(float(folded[5, 8]), 0.05, places=5)
        self.assertAlmostEqual(float(folded[0, 0]), 0.5, places=6)

    def test_a_uint8_mask_is_folded_in_unit_range(self):
        mask = np.full((H, W), 255, np.uint8)
        out = run_step("face_priority_weights", _inputs([0.0], masks=[mask]),
                       {"feather_px": 0.0})
        self.assertAlmostEqual(float(out["masks"][0][0, 0]), 1.0, places=6)

    def test_a_mask_count_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "masks"):
            run_step("face_priority_weights",
                     _inputs([0.0, 10.0], masks=[np.ones((H, W), np.float32)]))

    def test_cameras_disagreeing_on_size_are_refused(self):
        inputs = _inputs([0.0])
        inputs["cameras"].append(_camera(10.0, size=(W * 2, H)))
        with self.assertRaisesRegex(ValueError, "frame size"):
            run_step("face_priority_weights", inputs)


class TestPassThrough(unittest.TestCase):
    """The face branch off, or nothing to weight: every weight is 1, masks
    are untouched, and the rasteriser is never launched — which is what
    lets the shells' step run ungated on the face switch."""

    def setUp(self):
        self.calls = []
        patcher = mock.patch.object(face_priority, "_render_coverage",
                                    _stub_coverage(self.calls))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_splat_means_weights_of_one(self):
        inputs = _inputs([0.0, 90.0], splat_path=None)
        mask = np.full((H, W), 0.25, np.float32)
        inputs["masks"] = [mask, mask]
        del inputs["splat_center"]  # nothing to measure about either
        out = run_step("face_priority_weights", inputs)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(out["weights"]), 2)
        for w in out["weights"]:
            self.assertEqual(w.shape, (H, W))
            self.assertTrue((w == 1.0).all())
        for m in out["masks"]:
            np.testing.assert_allclose(m, 0.25)

    def test_a_zero_strength_renders_nothing(self):
        out = run_step("face_priority_weights", _inputs([0.0]), {"strength": 0.0})
        self.assertEqual(self.calls, [])
        self.assertTrue((out["weights"][0] == 1.0).all())

    def test_an_empty_batch_yields_empty_lists(self):
        out = run_step("face_priority_weights", _inputs([], masks=[]))
        self.assertEqual(out["weights"], [])
        self.assertEqual(out["masks"], [])
        self.assertEqual(self.calls, [])


class TestCoverageRender(unittest.TestCase):
    """`_render_coverage` goes through render_splat's `_rasterize`, on black
    with confidence off, and keeps only the alpha."""

    def test_it_rasterises_on_black_without_confidence(self):
        from pipeline.steps import splat as splat_module

        seen = {}

        def fake_rasterize(*, scene, splat_path, cameras, image_names, width, height,
                           bg_color, render_path, confidence=None):
            seen.update(splat_path=splat_path, n=len(cameras), names=list(image_names),
                        width=width, height=height, bg_color=bg_color,
                        render_path=render_path, confidence=confidence)
            return ([np.zeros((height, width, 3), np.uint8)] * len(cameras),
                    [_disc(height, width)] * len(cameras))

        class FakeScene:
            def __len__(self):
                return 7

        with mock.patch.object(splat_module, "_rasterize", fake_rasterize), \
                mock.patch("body2colmap.splat_scene.SplatScene.from_ply",
                           classmethod(lambda cls, path: FakeScene())):
            coverage = face_priority._render_coverage(
                "/x/face.ply", [_camera(0.0), _camera(5.0)], width=W, height=H,
                render_path=None,
            )
        self.assertEqual(len(coverage), 2)
        self.assertEqual(coverage[0].dtype, np.float32)
        self.assertEqual(seen["splat_path"], "/x/face.ply")
        self.assertEqual(seen["bg_color"], (0.0, 0.0, 0.0))
        self.assertIsNone(seen["confidence"])
        self.assertEqual(seen["render_path"], splat_module._RENDER_BINARY)
        self.assertEqual(len(set(seen["names"])), 2)


if __name__ == "__main__":
    unittest.main()
