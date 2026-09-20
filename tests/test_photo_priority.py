"""photo_priority_weights: the photograph wins over the surface it sees.

A unit cube of body in front of a ring of cameras, the photograph's camera
on +z: the views inside the window yield over the cube's front where the
anchor sees it, the anchor's own frames keep weight 1, the back views are
untouched, and the input weights (the face cap's) multiply through.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.steps.photo_priority import (anchor_frames, extend_off_mesh, photo_confidence, ramp, raster_view,
                                           surface_points, yield_weight)
from tests.helpers import run_step

import pipeline.steps  # noqa: F401

W, H = 32, 24
RADIUS = 5.0


def _camera(angle_deg: float, *, size=(W, H), look_at_origin: bool = True):
    """A camera on a ring of radius 5 about the origin, `angle_deg` from +z, looking at the origin."""
    from body2colmap.camera import Camera

    theta = np.radians(angle_deg)
    position = np.array([RADIUS * np.sin(theta), 0.0, RADIUS * np.cos(theta)], dtype=np.float64)
    # OpenGL camera axes: -z looks at the origin, y up.
    forward = -position / np.linalg.norm(position)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.stack([right, up, -forward], 1)
    return Camera(focal_length=(20.0, 20.0), image_size=size, principal_point=(size[0] / 2.0, size[1] / 2.0),
                  position=position.astype(np.float32), rotation=rotation.astype(np.float32))


def _cube(half: float = 1.0):
    """A closed cube of side 2*half about the origin, outward-facing triangles."""
    v = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], np.float64)
    faces = [
        (0, 1, 3), (0, 3, 2),   # -x
        (4, 6, 7), (4, 7, 5),   # +x
        (0, 4, 5), (0, 5, 1),   # -y
        (2, 3, 7), (2, 7, 6),   # +y
        (0, 2, 6), (0, 6, 4),   # -z
        (1, 5, 7), (1, 7, 3),   # +z
    ]
    return v, np.asarray(faces, np.int64)


def _inputs(angles, **extra):
    cameras = [_camera(a) for a in angles]
    return {"cameras": cameras, "mesh_world": _cube(), "anchor_cameras": cameras, "anchor_frame_index": 0,
            "splat_center": [0.0, 0.0, 0.0], **extra}


class TestPieces(unittest.TestCase):
    def test_ramp(self):
        x = np.array([0.0, 0.2, 0.35, 0.5, 1.0])
        np.testing.assert_allclose(ramp(x, 0.2, 0.5), [0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(ramp(x, 0.5, 0.5), [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_the_front_face_of_the_cube_is_seen_and_the_back_is_not(self):
        anchor = _camera(0.0)
        raster = raster_view(_cube(), anchor)
        self.assertTrue(raster.hit.any())
        points, normals, hit = surface_points(raster, anchor)
        # Every front-facing hit is the +z face, one unit from the origin, facing +z.
        np.testing.assert_allclose(points[:, 2], 1.0, atol=1e-6)
        np.testing.assert_allclose(normals[:, 2], 1.0, atol=1e-6)
        conf = photo_confidence(points, normals, anchor, raster.depth, margin=0.01, facing_lo=0.2, facing_hi=0.5)
        np.testing.assert_allclose(conf, 1.0)
        # Seen from behind, the same points face away from the anchor and are behind the body.
        back = _camera(180.0)
        raster_back = raster_view(_cube(), back)
        p_back, n_back, _ = surface_points(raster_back, back)
        np.testing.assert_allclose(p_back[:, 2], -1.0, atol=1e-6)
        conf_back = photo_confidence(p_back, n_back, anchor, raster.depth, margin=0.01, facing_lo=0.2, facing_hi=0.5)
        np.testing.assert_allclose(conf_back, 0.0)

    def test_a_grazing_face_ramps_with_its_cosine(self):
        anchor = _camera(0.0)
        anchor_depth = raster_view(_cube(), anchor).depth
        side = _camera(90.0)  # sees the +x face, whose normal is at 90 deg to the anchor's rays... nearly
        points, normals, _ = surface_points(raster_view(_cube(), side), side)
        conf = photo_confidence(points, normals, anchor, anchor_depth, margin=0.01, facing_lo=0.2, facing_hi=0.5)
        # The +x face at x=1 seen from a camera at z=5: cosine ~ 1/sqrt(26) = 0.2 -> at the ramp's foot.
        self.assertLess(float(conf.max()), 0.1)

    def test_extend_carries_the_nearest_value_out_and_stops(self):
        conf = np.zeros((10, 10), np.float32)
        hit = np.zeros((10, 10), bool)
        hit[4:6, 4:6] = True
        conf[4:6, 4:6] = 0.8
        out = extend_off_mesh(conf, hit, 2.0, None)
        self.assertAlmostEqual(float(out[4, 4]), 0.8)
        self.assertAlmostEqual(float(out[4, 6]), 0.8)   # one pixel out
        self.assertAlmostEqual(float(out[4, 7]), 0.8)   # two
        self.assertEqual(float(out[4, 8]), 0.0)          # three: beyond the reach
        alpha = np.zeros((10, 10), np.float32)
        alpha[4:6, 4:7] = 1.0
        clipped = extend_off_mesh(conf, hit, 2.0, alpha)
        self.assertAlmostEqual(float(clipped[4, 6]), 0.8)
        self.assertEqual(float(clipped[4, 7]), 0.0)

    def test_yield_weight_arithmetic(self):
        conf = np.ones((4, 4), np.float32)
        np.testing.assert_allclose(yield_weight(conf, strength=0.8, attenuation=1.0, feather_px=0.0), 0.2)
        np.testing.assert_allclose(yield_weight(conf, strength=0.8, attenuation=0.5, feather_px=0.0), 0.6)
        np.testing.assert_allclose(yield_weight(conf * 0, strength=0.8, attenuation=1.0, feather_px=0.0), 1.0)

    def test_anchor_frames_are_the_cameras_on_the_anchor(self):
        cameras = [_camera(a) for a in (0.0, 10.0, 90.0, 180.0, 0.0)]
        found = anchor_frames(cameras, np.asarray(cameras[0].position, np.float64), 0.5)
        self.assertEqual(found, [0, 4])


class TestStep(unittest.TestCase):
    def test_views_yield_over_the_front_and_the_anchor_keeps_its_weight(self):
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 30.0, 180.0, 0.0]),
                       {"feather_px": 0.0, "extend_px": 0.0, "strength": 0.8})
        weights = out["weights"]
        self.assertEqual(len(weights), 5)
        stats = out["photo_priority_stats"]
        self.assertEqual(stats["sources"], [0], "only the index is the photograph; the frame on the same pose is a repaint")
        np.testing.assert_allclose(weights[0], 1.0)
        self.assertLess(float(weights[4][H // 2, W // 2]), 1.0)
        # Opting the position rule in makes the repeated pose a source too.
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 30.0, 180.0, 0.0]),
                       {"feather_px": 0.0, "extend_px": 0.0, "anchor_tolerance_pct": 0.5})
        self.assertEqual(out["photo_priority_stats"]["sources"], [0, 4])
        # A view 15 deg round sees the cube's front, which the photograph owns: 1 - strength there.
        centre = weights[1][H // 2, W // 2]
        self.assertAlmostEqual(float(centre), 0.2, places=5)
        self.assertEqual(float(weights[1][0, 0]), 1.0, "off the body nothing yields")
        # The back view sees the -z face only: untouched.
        np.testing.assert_allclose(weights[3], 1.0)
        for w in weights:
            self.assertEqual(w.dtype, np.float32)
            self.assertEqual(w.shape, (H, W))

    def test_the_window_fades_a_view_past_the_cap(self):
        out = run_step("photo_priority_weights", _inputs([0.0, 60.0]),
                       {"feather_px": 0.0, "extend_px": 0.0, "cap_radius_deg": 45.0, "fade_deg": 30.0, "strength": 0.8})
        # 60 deg: halfway through the fade, so the attenuation is 0.5 and the yield 0.4 at most.
        w = out["weights"][1]
        owned = w < 1.0
        self.assertTrue(owned.any())
        self.assertGreaterEqual(float(w[owned].min()), 0.6 - 1e-5)

    def test_input_weights_multiply_through(self):
        base = [np.full((H, W), 0.5, np.float32) for _ in range(3)]
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 180.0], weights=base),
                       {"feather_px": 0.0, "extend_px": 0.0, "strength": 0.8})
        np.testing.assert_allclose(out["weights"][0], 0.5)
        np.testing.assert_allclose(out["weights"][2], 0.5)
        self.assertAlmostEqual(float(out["weights"][1][H // 2, W // 2]), 0.1, places=5)

    def test_no_mesh_or_zero_strength_pass_the_weights_through(self):
        base = [np.full((H, W), 0.5, np.float32) for _ in range(2)]
        inputs = _inputs([0.0, 15.0], weights=base)
        inputs["mesh_world"] = None
        out = run_step("photo_priority_weights", inputs)
        np.testing.assert_allclose(out["weights"][1], 0.5)
        self.assertEqual(out["photo_priority_stats"]["yielded"], 0)
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0]), {"strength": 0.0})
        np.testing.assert_allclose(out["weights"][1], 1.0)

    def test_no_anchor_is_refused_and_a_recorded_position_is_the_fallback(self):
        inputs = _inputs([0.0, 15.0])
        del inputs["anchor_cameras"]
        del inputs["anchor_frame_index"]
        with self.assertRaises(ValueError):
            run_step("photo_priority_weights", inputs)
        inputs["anchor_position"] = [0.0, 0.0, RADIUS]
        out = run_step("photo_priority_weights", inputs, {"feather_px": 0.0, "extend_px": 0.0, "strength": 0.8})
        self.assertEqual(out["photo_priority_stats"]["sources"], [0])
        self.assertAlmostEqual(float(out["weights"][1][H // 2, W // 2]), 0.2, places=5)

    def test_count_mismatches_are_refused(self):
        with self.assertRaises(ValueError):
            run_step("photo_priority_weights", _inputs([0.0, 15.0], weights=[np.ones((H, W), np.float32)]))
        with self.assertRaises(ValueError):
            run_step("photo_priority_weights", _inputs([0.0, 15.0], alphas=[np.ones((H, W), np.float32)]))

    def test_copies_of_the_photograph_are_masked_by_its_own_confidence(self):
        images = [np.full((H, W, 3), 200, np.uint8) for _ in range(3)]
        images[0][:] = 50  # the photograph's frame
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 180.0], images=images),
                       {"feather_px": 0.0, "extend_px": 0.0, "copies": 4})
        self.assertEqual(len(out["support_images"]), 4)
        self.assertEqual(len(out["support_masks"]), 4)
        self.assertEqual(len(out["support_cameras"]), 4)
        self.assertEqual(int(out["support_images"][0][H // 2, W // 2, 0]), 50)
        mask = out["support_masks"][0]
        self.assertEqual(mask.dtype, np.float32)
        self.assertAlmostEqual(float(mask[H // 2, W // 2]), 1.0, places=5)  # the cube's front faces its camera
        self.assertEqual(float(mask[0, 0]), 0.0)                            # off the body
        self.assertIs(out["support_cameras"][0], out["support_cameras"][3])
        self.assertEqual(out["photo_priority_stats"]["copies"], 4)
        # With an alpha to read, the default `matte` mask is the silhouette itself, cosine or not.
        alphas = [np.zeros((H, W), np.float32) for _ in range(3)]
        alphas[0][2:H - 2, 2:W - 2] = 1.0
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 180.0], images=images, alphas=alphas),
                       {"feather_px": 0.0, "extend_px": 0.0, "copies": 2, "copies_erode_px": 0})
        np.testing.assert_allclose(out["support_masks"][0], alphas[0])

    def test_the_copies_mask_is_shrunk_from_the_photographs_edge(self):
        """The photograph's outermost pixels are its backdrop-mixed edge, and
        twelve copies of them at one camera taught the splat a light rim
        seen only head-on (run c0514e). The default leaves that band to the
        frames."""
        images = [np.full((H, W, 3), 200, np.uint8) for _ in range(3)]
        alphas = [np.zeros((H, W), np.float32) for _ in range(3)]
        alphas[0][2:H - 2, 2:W - 2] = 1.0
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 180.0], images=images, alphas=alphas),
                       {"feather_px": 0.0, "extend_px": 0.0, "copies": 1, "copies_erode_px": 2})
        mask = out["support_masks"][0]
        self.assertEqual(float(mask[2, 2]), 0.0)                   # the edge band is out
        self.assertEqual(float(mask[3, 3]), 0.0)
        self.assertEqual(float(mask[H // 2, W // 2]), 1.0)         # the interior stays
        self.assertEqual(float(mask[4, 4]), 1.0)
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0, 180.0], images=images, alphas=alphas),
                       {"feather_px": 0.0, "extend_px": 0.0, "copies": 2, "copies_mask": "confidence"})
        self.assertEqual(float(out["support_masks"][0][2, 2]), 0.0)   # the cube's edge: no body there
        # No images wired: no copies, and a passthrough returns the empty triple too.
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0]), {"copies": 4})
        self.assertEqual(out["support_images"], [])
        out = run_step("photo_priority_weights", _inputs([0.0, 15.0], images=images[:2]), {"strength": 0.0})
        self.assertEqual(out["support_cameras"], [])

    def test_an_empty_batch_yields_empty_lists(self):
        out = run_step("photo_priority_weights", _inputs([]))
        self.assertEqual(out["weights"], [])


if __name__ == "__main__":
    unittest.main()
