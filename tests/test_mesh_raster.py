"""pipeline/mesh_raster.py — the numpy z-buffer the face cap's surface
placement and the body cull rest on — against closed-form geometry."""

import unittest

import numpy as np

from pipeline import mesh_raster as mr

FX = 400.0
W, H = 160, 120
CX, CY = W / 2, H / 2


def _sphere(radius=0.3, centre=(0.0, 0.0, 2.0), subdivisions=4):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(m.vertices, np.float64) + np.asarray(centre), np.asarray(m.faces, np.int64)


class _Camera:
    """A body2colmap Camera's attributes: OpenGL camera-to-world rotation."""

    def __init__(self, position, rotation=None, fx=FX, width=W, height=H):
        self.position = np.asarray(position, np.float32)
        self.rotation = np.eye(3, dtype=np.float32) if rotation is None else np.asarray(rotation, np.float32)
        self.fx = self.fy = fx
        self.cx, self.cy = width / 2, height / 2
        self.width, self.height = width, height


class TestRasterize(unittest.TestCase):
    def test_sphere_depth_and_normals_match_the_closed_form(self):
        v, f = _sphere()
        r = mr.rasterize(v, f, fx=FX, fy=FX, cx=CX, cy=CY, width=W, height=H)
        vv, uu = np.mgrid[0:H, 0:W]
        d = np.stack([(uu - CX) / FX, (vv - CY) / FX, np.ones((H, W))], 2)
        dn = d / np.linalg.norm(d, axis=2, keepdims=True)
        c = np.array([0.0, 0.0, 2.0])
        b = (dn * c).sum(2)
        disc = b * b - (c @ c - 0.09)
        hit = disc > 0
        z_true = (b - np.sqrt(np.where(hit, disc, 0.0))) * dn[..., 2]
        both = hit & r.hit
        self.assertGreater(both.sum(), 0.99 * hit.sum())
        # inside the silhouette the polyhedron's depth is the sphere's to a
        # millimetre or so at this subdivision; the normal is the sphere's
        err = np.abs(r.depth[both] - z_true[both])
        self.assertLess(np.percentile(err, 95), 0.003)
        p = dn[both] * (r.depth[both] / dn[both][:, 2])[:, None]
        n_true = p - c
        n_true /= np.linalg.norm(n_true, axis=1, keepdims=True)
        cosine = (r.normal[both] * n_true).sum(1)
        self.assertGreater(np.percentile(cosine, 5), np.cos(np.radians(8)))
        self.assertTrue(r.facing[r.hit].all())
        self.assertTrue((r.normal[r.hit][:, 2] < 0).all())
        self.assertTrue(np.isinf(r.depth[~r.hit]).all())

    def test_a_back_face_seen_through_a_gap_is_not_facing(self):
        # One triangle, wound so its normal points AWAY from the camera
        # (+z in OpenCV: (v1 - v0) x (v2 - v0) = (0, 0, 4)).
        v = np.array([[-1, -1, 2.0], [1, -1, 2.0], [0, 1, 2.0]])
        r = mr.rasterize(v, np.array([[0, 1, 2]]), fx=FX, fy=FX, cx=CX, cy=CY, width=W, height=H)
        self.assertTrue(r.hit.any())
        self.assertFalse(r.facing[r.hit].any())
        # but the stored normal still faces the camera
        self.assertTrue((r.normal[r.hit][:, 2] < 0).all())

    def test_a_triangle_behind_the_camera_is_dropped(self):
        v = np.array([[-1, -1, -2.0], [1, -1, -2.0], [0, 1, -2.0]])
        r = mr.rasterize(v, np.array([[0, 1, 2]]), fx=FX, fy=FX, cx=CX, cy=CY, width=W, height=H)
        self.assertFalse(r.hit.any())

    def test_nearest_wins(self):
        near = np.array([[-1, -1, 1.5], [1, -1, 1.5], [0, 1, 1.5]])
        far = near + [0, 0, 1.0]
        v = np.vstack([far, near])
        r = mr.rasterize(v, np.array([[0, 1, 2], [3, 4, 5]]), fx=FX, fy=FX, cx=CX, cy=CY, width=W, height=H)
        self.assertEqual(set(np.unique(r.face[r.hit])), {1})
        self.assertAlmostEqual(float(r.depth[H // 2, W // 2]), 1.5, places=9)


class TestHitSurface(unittest.TestCase):
    def test_masked_pixels_off_the_mesh_take_the_nearest_hit(self):
        v, f = _sphere()
        r = mr.rasterize(v, f, fx=FX, fy=FX, cx=CX, cy=CY, width=W, height=H)
        # a mask 6 px wider than the silhouette: the hair rim
        import cv2
        mask = cv2.dilate(r.hit.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        z, n, on = mr.hit_surface(r, mask)
        self.assertTrue((on == (mask & r.hit)).all())
        self.assertTrue(np.isfinite(z[mask]).all())
        self.assertTrue((z[mask] > 0).all())
        rim = mask & ~on
        self.assertTrue(rim.any())
        # the rim's depth is a silhouette depth, not the background's
        self.assertLess(z[rim].max(), r.depth[r.hit].max() + 1e-9)
        self.assertGreater(np.linalg.norm(n[rim], axis=1).min(), 0.99)

    def test_no_hit_is_refused(self):
        r = mr.rasterize(np.zeros((0, 3)), np.zeros((0, 3), int), fx=FX, fy=FX, cx=CX, cy=CY, width=W, height=H)
        with self.assertRaises(ValueError):
            mr.hit_surface(r, np.ones((H, W), bool))


class TestCull(unittest.TestCase):
    def test_points_behind_the_body_are_hidden_and_the_rest_are_not(self):
        v, f = _sphere()
        # the sphere sits at world (0, 0, -2) for a camera at the origin
        # looking down -z (OpenGL identity): the OpenCV z of 2
        world = (v * mr._FLIP, f)
        cam = _Camera((0, 0, 0))
        pts = np.array([
            [0, 0, -1.7],     # in front of the sphere's near pole (z_cv 1.7 < 1.7)
            [0, 0, -2.0],     # inside: 0.3 behind the surface -> hidden
            [0, 0, -1.71],    # 1 cm behind the surface: within the margin
            [0.5, 0.5, -2.0],  # off the sphere's silhouette: visible
            [0, 0, 1.0],      # behind the camera: hidden
        ])
        hidden = mr.behind_mesh(pts, world, cam, margin=0.015)
        self.assertEqual(hidden.tolist(), [False, True, False, False, True])

    def test_cull_returns_a_scene_without_the_hidden_gaussians(self):
        from body2colmap.splat_scene import SplatScene

        v, f = _sphere()
        world = (v * mr._FLIP, f)
        means = np.array([[0, 0, -1.7], [0, 0, -2.0], [0.5, 0.5, -2.0]], np.float32)
        n = len(means)
        scene = SplatScene(means=means, scales=np.zeros((n, 3), np.float32), quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                           opacities=np.zeros(n, np.float32), sh_coeffs=np.zeros((n, 1, 3), np.float32), sh_degree=0)
        culled = mr.cull_behind_mesh(scene, world, _Camera((0, 0, 0)))
        self.assertEqual(len(culled), 2)
        np.testing.assert_array_equal(culled.means, means[[0, 2]])
        # nothing hidden: the same object comes back
        visible = SplatScene(means=means[[0, 2]], scales=np.zeros((2, 3), np.float32),
                             quats=np.tile([1, 0, 0, 0], (2, 1)).astype(np.float32), opacities=np.zeros(2, np.float32),
                             sh_coeffs=np.zeros((2, 1, 3), np.float32), sh_degree=0)
        self.assertIs(mr.cull_behind_mesh(visible, world, _Camera((0, 0, 0))), visible)


if __name__ == "__main__":
    unittest.main()
