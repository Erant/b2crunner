"""steps/photo_texture.py's pure helpers: the camera, the projection, the grid diffusion, the confidence and the levelling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from pipeline.steps.photo_texture import (confidence, grid_smooth, head_region, level_bake, photo_camera, project, ramp,
                                          smooth_normals, soft_masks)


def _camera(position, rotation=None):
    return SimpleNamespace(position=list(position), rotation=(np.eye(3) if rotation is None else np.asarray(rotation)).tolist())


class TestCamera(unittest.TestCase):
    def test_identity_pose_projects_with_the_half_pixel_shift(self):
        entry = photo_camera(_camera([0, 0, 0]), _camera([0, 0, 0]), 1000.0, 768, 1536)
        self.assertEqual((entry["cx"], entry["cy"]), (384.5, 768.5))
        # a point 2 m down the view (-z in OpenGL), 0.1 m to the right and 0.2 m up
        u, v, z = project(np.array([[0.1, 0.2, -2.0]]), entry)
        self.assertAlmostEqual(float(z[0]), 2.0)
        self.assertAlmostEqual(float(u[0]), 384.5 + 50.0)   # +x is right
        self.assertAlmostEqual(float(v[0]), 768.5 - 100.0)  # +y is up, image rows go down

    def test_the_refinement_delta_moves_the_photo_camera(self):
        # the given anchor sits at the origin; the refined one moved 3 cm along +x: so does the photograph's camera
        entry = photo_camera(_camera([0.03, 0, 0]), _camera([0, 0, 0]), 1000.0, 768, 1536)
        self.assertTrue(np.allclose(entry["position"], [0.03, 0, 0]))
        self.assertTrue(np.allclose(entry["rotation"], np.eye(3)))

    def test_without_a_given_camera_the_refined_pose_is_used(self):
        entry = photo_camera(_camera([1, 2, 3]), None, 500.0, 100, 100)
        self.assertEqual(entry["position"], [1.0, 2.0, 3.0])
        self.assertEqual(entry["fx"], 500.0)


class TestGrid(unittest.TestCase):
    def test_diffusion_averages_a_constant_and_keeps_it_constant(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 0.1, (5000, 3))
        values = np.full(5000, 7.0)
        out, mass = grid_smooth(p, values, np.ones(5000), 0.01, 3)
        self.assertTrue(np.allclose(out[mass > 1e-6], 7.0))
        self.assertTrue((mass > 0).all())

    def test_mass_dies_away_from_the_weighted_points(self):
        p = np.stack([np.linspace(0, 0.3, 301), np.zeros(301), np.zeros(301)], 1)
        w = (p[:, 0] < 0.05).astype(float)
        _, mass = grid_smooth(p, np.ones(301), w, 0.01, 4)
        self.assertGreater(mass[0], mass[100])
        self.assertLess(mass[300], 1e-9)

    def test_read_back_is_continuous_across_cells(self):
        # a linear field sampled on a line: a per-cell read-back would step, a trilinear one follows the line
        p = np.stack([np.linspace(0, 0.2, 2001), np.zeros(2001), np.zeros(2001)], 1)
        out, _ = grid_smooth(p, p[:, 0], np.ones(2001), 0.01, 0)
        steps = np.abs(np.diff(out[500:1500]))
        self.assertLess(steps.max(), 0.0005)

    def test_smoothed_normals_are_unit(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 0.05, (2000, 3))
        n = rng.normal(size=(2000, 3))
        out = smooth_normals(p, n, 0.01, 2)
        self.assertTrue(np.allclose(np.linalg.norm(out, axis=1), 1.0))


class TestMasksAndConfidence(unittest.TestCase):
    def test_soft_masks_ramp_inside_their_classes(self):
        labels = np.zeros((40, 40), np.int32)
        labels[5:35, 5:35] = 1        # apparel
        labels[10:20, 10:30] = 3      # face
        labels[20:30, 10:30] = 4      # hair
        fg, face, hair = soft_masks(labels, 3.0, 2.0)
        self.assertEqual(float(fg[0, 0]), 0.0)
        self.assertEqual(float(fg[20, 20]), 1.0)
        self.assertLess(float(fg[5, 20]), 0.5)   # the first foreground pixel is not trusted
        self.assertEqual(float(face[15, 20]), 1.0)
        self.assertEqual(float(face[25, 20]), 0.0)
        self.assertEqual(float(hair[25, 20]), 1.0)
        self.assertEqual(float(hair[15, 20]), 0.0)

    def test_confidence_hair_is_stricter_and_the_face_core_wins(self):
        facing = np.array([0.35, 0.35, 0.35, 0.9])
        ones = np.ones(4)
        hair = np.array([0.0, 1.0, 0.0, 0.0])
        face = np.array([0.0, 0.0, 1.0, 0.0])
        conf, core = confidence(facing, ones, ones, face, hair, ones, 0.2, 0.5, 0.45, 0.85, 0.1, 0.3)
        self.assertTrue(0.0 < conf[0] < 1.0)     # skin at a middling angle: partial
        self.assertEqual(conf[1], 0.0)           # hair at that angle: nothing
        self.assertEqual(conf[2], 1.0)           # the face core: the photograph's pixels
        self.assertEqual(core[2], 1.0)
        self.assertEqual(conf[3], 1.0)           # facing squarely: full

    def test_hidden_or_background_texels_get_nothing(self):
        facing = np.ones(3)
        conf, _ = confidence(facing, np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0]), np.zeros(3), np.zeros(3),
                             np.array([1.0, 1.0, 0.0]), 0.2, 0.5, 0.45, 0.85, 0.1, 0.3)
        self.assertTrue((conf == 0).all())

    def test_head_region_is_the_ball_and_the_neck(self):
        c = np.array([0.0, 1.6, 0.0])
        p = np.array([c, c + [0.0, 0.0, 0.10], c + [0.0, 0.0, 0.30], c - [0.0, 0.10, 0.0], c - [0.0, 0.30, 0.0], c - [0.20, 0.10, 0.0]])
        r = head_region(p, c, 0.16, 0.08, 0.18)
        self.assertEqual(r[0], 1.0)
        self.assertEqual(r[1], 1.0)
        self.assertEqual(r[2], 0.0)      # 30 cm in front: outside
        self.assertEqual(r[3], 1.0)      # 10 cm below, on the axis: the neck
        self.assertEqual(r[4], 0.0)      # 30 cm below: past the neck
        self.assertEqual(r[5], 0.0)      # 20 cm to the side: outside both


class TestLevelling(unittest.TestCase):
    def test_the_bake_is_shifted_toward_the_photo_near_the_reliable_texels_only(self):
        # a line of texels; the photograph owns the first 5 cm and is 40 brighter than the bake there
        n = 401
        p = np.stack([np.linspace(0, 0.4, n), np.zeros(n), np.zeros(n)], 1)
        bake = np.full((n, 3), 100.0)
        photo = np.full((n, 3), 140.0)
        conf = (p[:, 0] < 0.05).astype(np.float32)
        out, fade = level_bake(bake, photo, conf, np.zeros(n), p, np.ones(n), 0.01, 20, 0.6, 0.01, 0.08)
        self.assertTrue(np.allclose(out[:50], 140.0, atol=1.0))      # under the photograph: fully levelled
        self.assertGreater(out[70, 0], 120.0)                          # just beyond it: still lifted
        self.assertTrue(np.allclose(out[-50:], 100.0))                 # far away: untouched
        self.assertEqual(fade[-1], 0.0)

    def test_hair_and_skin_offsets_stay_apart(self):
        n = 400
        p = np.stack([np.linspace(0, 0.2, n), np.zeros(n), np.zeros(n)], 1)
        hair = (np.arange(n) >= 200).astype(np.float32)
        bake = np.where(hair[:, None] > 0, 40.0, 180.0) * np.ones((n, 3))
        photo = bake + np.where(hair[:, None] > 0, -20.0, +20.0)
        conf = np.ones(n, np.float32)
        out, _ = level_bake(bake, photo, conf, hair, p, np.ones(n), 0.01, 10, 0.6, 0.01, 0.08)
        self.assertTrue(np.allclose(out[:180], 200.0, atol=2.0))
        self.assertTrue(np.allclose(out[220:], 20.0, atol=2.0))

    def test_nothing_reliable_means_no_change(self):
        p = np.random.default_rng(0).uniform(0, 0.1, (100, 3))
        bake = np.full((100, 3), 50.0)
        out, fade = level_bake(bake, bake + 10, np.zeros(100), np.zeros(100), p, np.ones(100), 0.01, 5, 0.6, 0.01, 0.08)
        self.assertTrue(np.allclose(out, 50.0))
        self.assertTrue((fade == 0).all())


class TestRamp(unittest.TestCase):
    def test_ramp_endpoints(self):
        self.assertEqual(float(ramp(np.array(0.0), 0.0, 1.0)), 0.0)
        self.assertEqual(float(ramp(np.array(1.0), 0.0, 1.0)), 1.0)
        self.assertAlmostEqual(float(ramp(np.array(0.5), 0.0, 1.0)), 0.5)


if __name__ == "__main__":
    unittest.main()
