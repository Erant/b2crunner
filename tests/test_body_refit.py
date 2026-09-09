"""splat_surface / refit_body_to_splat — the parts that run without the
trainer binary or the body model.

The fit needs the MHR body model and its checkpoint (gated, 2.8 GB) and
the surface needs a trained splat and the trainer; what can be pinned
here is the geometry around them — the depth unprojection and its edge
rule, the similarity recovery, the hand mask, the cameras.json — and
the steps' refusals, which are what a mis-wired workflow hits first.
"""

from __future__ import annotations

import unittest

import numpy as np

import pipeline.steps  # noqa: F401  — registers the steps
from pipeline.registry import get_step_class
from pipeline.steps import body_refit


class _Camera:
    """The body2colmap Camera fields unproject_depth reads."""

    def __init__(self, width=64, height=48, fx=80.0, position=(0.0, 0.0, 0.0), rotation=None):
        self.width, self.height = width, height
        self.fx = self.fy = fx
        self.cx, self.cy = width / 2.0, height / 2.0
        self.position = np.asarray(position, np.float64)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, np.float64)


class TestUmeyama(unittest.TestCase):
    def test_recovers_a_known_similarity(self):
        rng = np.random.RandomState(1)
        src = rng.normal(size=(200, 3))
        angle = 0.7
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        dst = 1.3 * src @ rot.T + np.array([0.5, -2.0, 3.0])
        s, r, t = body_refit.umeyama(src, dst)
        self.assertAlmostEqual(s, 1.3, places=6)
        np.testing.assert_allclose(r, rot, atol=1e-6)
        np.testing.assert_allclose(t, [0.5, -2.0, 3.0], atol=1e-6)

    def test_without_scale_the_scale_is_one(self):
        rng = np.random.RandomState(2)
        src = rng.normal(size=(50, 3))
        s, r, t = body_refit.umeyama(src, 2.0 * src, with_scale=False)
        self.assertEqual(s, 1.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-6)


class TestDepthUnprojection(unittest.TestCase):
    def test_a_flat_wall_unprojects_onto_the_plane_facing_the_camera(self):
        cam = _Camera()
        depth = np.full((cam.height, cam.width), 2.0, np.float32)
        points, normals = body_refit.unproject_depth(depth, cam, stride=4, edge_jump=0.02)
        self.assertGreater(len(points), 100)
        # OpenGL camera-to-world identity: the camera looks down -Z, so a
        # wall 2 m in front of it is at z = -2 and its normal faces +Z.
        np.testing.assert_allclose(points[:, 2], -2.0, atol=1e-5)
        np.testing.assert_allclose(normals, np.tile([0, 0, 1.0], (len(normals), 1)), atol=1e-5)
        # The pixel grid spans the image: x from about -0.8 m to +0.8 m at 2 m for fx 80 / 64 px.
        self.assertLess(points[:, 0].min(), -0.6)
        self.assertGreater(points[:, 0].max(), 0.6)

    def test_the_camera_pose_is_applied(self):
        # Camera at (0, 0, 5) looking down -Z (identity rotation): the wall at depth 2 is at z = 3.
        cam = _Camera(position=(0.0, 0.0, 5.0))
        depth = np.full((cam.height, cam.width), 2.0, np.float32)
        points, _ = body_refit.unproject_depth(depth, cam, stride=8, edge_jump=0.02)
        np.testing.assert_allclose(points[:, 2], 3.0, atol=1e-5)
        # Turned 90 deg about Y, the camera looks down -X: the wall is at x = -2.
        rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        cam = _Camera(rotation=rot)
        points, normals = body_refit.unproject_depth(depth, cam, stride=8, edge_jump=0.02)
        np.testing.assert_allclose(points[:, 0], -2.0, atol=1e-5)
        np.testing.assert_allclose(normals[:, 0], 1.0, atol=1e-5)

    def test_silhouettes_and_holes_are_dropped(self):
        cam = _Camera()
        depth = np.full((cam.height, cam.width), 2.0, np.float32)
        depth[:, 32:] = 2.5                       # a 50 cm step down the middle
        depth[10:14, 10:14] = np.nan              # a hole
        points, _ = body_refit.unproject_depth(depth, cam, stride=1, edge_jump=0.02)
        # Nothing straddles the step: every point is on one of the two planes.
        z = -points[:, 2]
        self.assertTrue(np.all((np.abs(z - 2.0) < 1e-5) | (np.abs(z - 2.5) < 1e-5)))
        # And the columns at the step (31, 32) and the pixels around the hole are gone.
        us = np.round((points[:, 0] / z) * cam.fx + cam.cx - 0.5)
        self.assertFalse(np.any((us == 31) | (us == 32)))
        full = (cam.width - 2) * (cam.height - 2)
        self.assertLess(len(points), full)
        self.assertGreater(len(points), full - 2 * (cam.height - 2) - 6 * 6 - 1)

    def test_a_slanted_surface_gets_a_slanted_normal(self):
        cam = _Camera()
        u = np.arange(cam.width)[None, :] + 0.5
        # depth grows with x: z = 2 + 0.5 * x_world, so x_world = (u - cx)/fx * z  =>  z = 2 / (1 - 0.5 (u-cx)/fx)
        depth = np.broadcast_to(2.0 / (1.0 - 0.5 * (u - cam.cx) / cam.fx), (cam.height, cam.width)).astype(np.float32)
        points, normals = body_refit.unproject_depth(depth, cam, stride=4, edge_jump=0.5)
        # In world (OpenGL) the plane is z = -(2 + 0.5 x): normal ∝ (0.5, 0, 1), toward the camera.
        expected = np.array([0.5, 0.0, 1.0]) / np.sqrt(1.25)
        np.testing.assert_allclose(normals, np.tile(expected, (len(normals), 1)), atol=1e-3)
        np.testing.assert_allclose(points[:, 2], -(2.0 + 0.5 * points[:, 0]), atol=1e-4)

    def test_depth_png_decodes_millimetres(self):
        png = np.array([[0, 1500], [65535, 2]], np.uint16)
        depth = body_refit.decode_depth_png(png)
        self.assertTrue(np.isnan(depth[0, 0]))
        self.assertAlmostEqual(float(depth[0, 1]), 1.5, places=6)
        self.assertAlmostEqual(float(depth[1, 1]), 0.002, places=6)

    def test_a_camera_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            body_refit.unproject_depth(np.zeros((10, 10), np.float32), _Camera(width=12, height=10), 1, 0.02)


class TestHandMask(unittest.TestCase):
    def test_vertices_nearest_a_hand_keypoint_are_hands(self):
        keypoints = np.zeros((70, 3))
        keypoints[:, 0] = np.arange(70) * 0.1        # spread out along x
        vertices = np.array([[2.1, 0, 0], [6.2, 0, 0], [0.05, 0, 0], [6.9, 0, 0]])
        mask = body_refit.hand_vertex_mask(vertices, keypoints)
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_the_wrong_keypoint_count_is_refused(self):
        with self.assertRaises(ValueError):
            body_refit.hand_vertex_mask(np.zeros((3, 3)), np.zeros((127, 3)))


class TestCamerasJson(unittest.TestCase):
    def test_entries_are_named_by_index_and_carry_the_pose_untouched(self):
        rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        cams = [_Camera(position=(1, 2, 3), rotation=rot), _Camera()]
        payload = body_refit.cameras_json(cams)
        self.assertEqual((payload["width"], payload["height"]), (64, 48))
        self.assertEqual([c["name"] for c in payload["cameras"]], ["frame_00000.png", "frame_00001.png"])
        self.assertEqual(payload["cameras"][0]["rotation"], rot.tolist())
        self.assertEqual(payload["cameras"][0]["position"], [1.0, 2.0, 3.0])

    def test_mixed_resolutions_are_refused(self):
        with self.assertRaises(ValueError):
            body_refit.cameras_json([_Camera(), _Camera(width=32)])


class TestRefusals(unittest.TestCase):
    def test_splat_surface_needs_the_trainer(self):
        step = get_step_class("splat_surface")()
        params = get_step_class("splat_surface").resolve_params({"trainer_path": "/nonexistent/b2ctrain"})
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ply") as ply:
            with self.assertRaises(RuntimeError) as ctx:
                step.run({"splat_path": ply.name, "cameras": [_Camera()]}, params)
        self.assertIn("/nonexistent/b2ctrain", str(ctx.exception))

    def test_refit_needs_pose_params(self):
        step = get_step_class("refit_body_to_splat")()
        params = get_step_class("refit_body_to_splat").resolve_params({})
        with self.assertRaises(KeyError):
            step.run({"mesh_output": {"vertices": np.zeros((4, 3))}, "mesh_world": (np.zeros((4, 3)), np.zeros((1, 3))),
                      "surface": {"points": np.zeros((2000, 3)), "normals": np.zeros((2000, 3))}}, params)

    def test_refit_needs_the_world_mesh_pair(self):
        step = get_step_class("refit_body_to_splat")()
        params = get_step_class("refit_body_to_splat").resolve_params({})
        mesh = {"vertices": np.zeros((4, 3)), "faces": np.zeros((1, 3)), "pose_params": {}}
        with self.assertRaises(ValueError) as ctx:
            step.run({"mesh_output": mesh, "mesh_world": np.zeros((4, 3)),
                      "surface": {"points": np.zeros((2000, 3)), "normals": np.zeros((2000, 3))}}, params)
        self.assertIn("mesh_world", str(ctx.exception))

    def test_refit_needs_a_surface(self):
        step = get_step_class("refit_body_to_splat")()
        params = get_step_class("refit_body_to_splat").resolve_params({})
        mesh = {"vertices": np.zeros((4, 3)), "faces": np.zeros((1, 3)), "pose_params": {}}
        with self.assertRaises(ValueError) as ctx:
            step.run({"mesh_output": mesh, "mesh_world": (np.zeros((4, 3)), np.zeros((1, 3))),
                      "surface": {"points": np.zeros((10, 3)), "normals": np.zeros((10, 3))}}, params)
        self.assertIn("surface", str(ctx.exception))

    def test_refit_refuses_a_deformed_world_mesh(self):
        step = get_step_class("refit_body_to_splat")()
        params = get_step_class("refit_body_to_splat").resolve_params({})
        rng = np.random.RandomState(0)
        raw = rng.normal(size=(500, 3))
        world = raw + rng.normal(scale=0.01, size=raw.shape)     # 1 cm of non-rigid noise
        mesh = {"vertices": raw, "faces": np.zeros((1, 3), int), "pose_params": {}}
        with self.assertRaises(ValueError) as ctx:
            step.run({"mesh_output": mesh, "mesh_world": (world, np.zeros((1, 3), int)),
                      "surface": {"points": np.zeros((2000, 3)), "normals": np.zeros((2000, 3))}}, params)
        self.assertIn("rigid placement", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
