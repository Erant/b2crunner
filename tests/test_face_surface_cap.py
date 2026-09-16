"""face_pointmap_splat with `depth_prior: mesh_surface` — the photograph's
pixels on the body model's surface (FACE_GUIDE.md's cap), with no
pointmap run at all — and the body cull the cap's renders get."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pipeline.steps import pointmap_splat as ps
from tests.test_pointmap_splat import FOCAL, CX, CY, WIDTH, HEIGHT, _quat_to_mat, _read_ply


class _NoNetwork(ps.PointmapSplatStep):
    def load(self, params):  # noqa: D102
        raise AssertionError("the mesh_surface prior must not load the pointmap head")

    def _pointmap(self, image_bgr, params):  # noqa: D102
        raise AssertionError("the mesh_surface prior must not run the pointmap head")


def _sphere_mesh(radius=0.3, centre=(0.0, 0.0, 2.0), subdivisions=5):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(m.vertices, np.float64) + np.asarray(centre), np.asarray(m.faces, np.int64)


class TestMeshSurfaceCap(unittest.TestCase):
    def test_every_gaussian_sits_on_the_mesh_with_its_normal(self):
        vertices, faces = _sphere_mesh()
        # SAM-3D-Body's camera at the origin: vertices_cam = vertices + cam_t
        cam_t = np.array([0.01, -0.02, 0.03])
        mesh_output = {"vertices": vertices - cam_t, "cam_t": cam_t, "faces": faces, "focal_length": FOCAL}
        # a matte a little wider than the sphere's silhouette: a hair rim
        vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        dx, dy = (uu - CX) / FOCAL, (vv - CY) / FOCAL
        a = dx ** 2 + dy ** 2 + 1.0
        disc = 4.0 * 2.0 ** 2 - 4 * a * (2.0 ** 2 - 0.3 ** 2)
        on_sphere = disc > 0
        import cv2
        matte = cv2.dilate(on_sphere.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        image[..., 1] = 200
        with tempfile.TemporaryDirectory() as tmp:
            params = ps.PointmapSplatStep.resolve_params(
                {"filepath": str(Path(tmp) / "cap.ply"), "depth_prior": "mesh_surface",
                 "fill_max_frac": 0.0, "alpha_erode": 0, "debug_dir": str(Path(tmp) / "dbg")})
            result = _NoNetwork().run(
                {"image": image, "mask": matte, "normal_map": np.zeros((HEIGHT, WIDTH, 3), np.float32),
                 "mesh_output": mesh_output},
                params,
            )
            _head, props, data = _read_ply(Path(tmp) / "cap.ply")
            self.assertTrue((Path(tmp) / "dbg" / "stats.json").is_file())
        stats = result["splat_stats"]
        self.assertEqual(stats["depth_prior"], "mesh_surface")
        self.assertGreater(stats["surface"]["on_mesh_pixels"], 0.9 * stats["mask_pixels"])
        self.assertGreater(stats["surface"]["off_mesh_pixels"], 0)   # the rim was filled, not dropped
        self.assertEqual(stats["n_splats"], stats["mask_pixels"])    # and nothing cliff-culled

        means = data[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)
        # world = camera * FLIP; the sphere's centre in world is (0, 0, -2)
        radius = np.linalg.norm(means - np.array([0.0, 0.0, -2.0]), axis=1)
        # the 90 % that hit are on the polyhedron: within ~1.5 mm of the sphere
        self.assertLess(np.percentile(np.abs(radius - 0.3), 90), 0.0015)
        # the filled rim took a silhouette depth, so nothing flew to the background
        self.assertLess(radius.max(), 0.36)
        # the thin axis (column 2 of the rotation) is the surface normal
        quats = data[:, [props.index(f"rot_{i}") for i in range(4)]]
        thin = _quat_to_mat(quats)[:, :, 2]
        outward = (means - np.array([0.0, 0.0, -2.0])) / radius[:, None]
        cosine = np.abs((thin * outward).sum(1))
        self.assertGreater(np.percentile(cosine, 10), np.cos(np.radians(6)))
        scales = np.exp(data[:, [props.index(f"scale_{i}") for i in range(3)]])
        self.assertTrue((scales[:, 2] < scales[:, 1]).all())
        # the colour is the photograph's, in SH DC
        dc = data[:, [props.index(f"f_dc_{i}") for i in range(3)]]
        rgb = dc * ps.SH_C0 + 0.5
        np.testing.assert_allclose(rgb[:, 1], 200 / 255.0, atol=1e-3)
        np.testing.assert_allclose(rgb[:, [0, 2]], 0.0, atol=1e-3)

    def test_needs_the_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            params = ps.PointmapSplatStep.resolve_params(
                {"filepath": str(Path(tmp) / "cap.ply"), "depth_prior": "mesh_surface"})
            with self.assertRaises(ValueError) as caught:
                _NoNetwork().run(
                    {"image": np.zeros((HEIGHT, WIDTH, 3), np.uint8), "mask": np.ones((HEIGHT, WIDTH), np.float32),
                     "normal_map": np.zeros((HEIGHT, WIDTH, 3), np.float32),
                     "mesh_output": {"vertices": np.zeros((3, 3)), "cam_t": np.zeros(3), "focal_length": FOCAL}},
                    params,
                )
        self.assertIn("faces", str(caught.exception))


class TestRenderSplatCull(unittest.TestCase):
    """With `cull_mesh` wired, render_splat renders view by view from the
    scene without the Gaussians the body hides from that view."""

    def test_one_render_per_view_from_the_culled_scene(self):
        from body2colmap.splat_scene import SplatScene

        from pipeline.steps import splat as splat_module

        vertices, faces = _sphere_mesh()
        world = (vertices * ps.FLIP, faces)
        # a Gaussian on the sphere's near pole (visible from the front, hidden
        # from behind) and one on its far pole
        means = np.array([[0, 0, -1.7], [0, 0, -2.3]], np.float32)
        scene = SplatScene(means=means, scales=np.zeros((2, 3), np.float32), quats=np.tile([1, 0, 0, 0], (2, 1)).astype(np.float32),
                           opacities=np.zeros(2, np.float32), sh_coeffs=np.zeros((2, 1, 3), np.float32), sh_degree=0)

        class Cam:
            def __init__(self, position, rotation):
                self.position = np.asarray(position, np.float32)
                self.rotation = np.asarray(rotation, np.float32)
                self.fx = self.fy = 500.0
                self.cx, self.cy, self.width, self.height = 100.0, 100.0, 200, 200

        front = Cam((0, 0, 0), np.eye(3))
        turn = np.diag([-1.0, 1.0, -1.0])          # 180 deg about y: looking down +z from behind
        back = Cam((0, 0, -4), turn)
        seen = []

        def fake_rasterize(*, scene, splat_path, cameras, image_names, width, height, bg_color, render_path,
                           confidence=None, sh_degree=None):
            seen.append((len(scene), [float(m[2]) for m in scene.means], splat_path, list(image_names)))
            return ([np.zeros((height, width, 3), np.uint8)] * len(cameras), [np.ones((height, width), np.float32)] * len(cameras))

        with mock.patch.object(splat_module, "_rasterize", fake_rasterize):
            images, masks = splat_module._rasterize_culled(
                scene=scene, cull_mesh=world, margin=0.015, cameras=[front, back], image_names=["a.png", "b.png"],
                width=200, height=200, bg_color=(0, 0, 0), render_path="x")
        self.assertEqual(len(images), 2)
        self.assertEqual(len(masks), 2)
        self.assertEqual([s[0] for s in seen], [1, 1])
        np.testing.assert_allclose(seen[0][1], [-1.7], atol=1e-6)   # the front sees the near pole
        np.testing.assert_allclose(seen[1][1], [-2.3], atol=1e-6)   # the back the far one
        self.assertEqual([s[2] for s in seen], [None, None])  # the culled scene is what is rendered, not the file
        self.assertEqual([s[3] for s in seen], [["a.png"], ["b.png"]])

    def test_coverage_goes_through_the_cull_when_the_mesh_is_wired(self):
        from pipeline.steps import face_priority, splat as splat_module

        class FakeScene:
            def __len__(self):
                return 7

        calls = []

        def fake_culled(*, scene, cull_mesh, margin, cameras, image_names, width, height, bg_color, render_path,
                        confidence=None, sh_degree=None):
            calls.append((margin, len(cameras), bg_color, render_path))
            return ([np.zeros((height, width, 3), np.uint8)] * len(cameras), [np.ones((height, width), np.float32)] * len(cameras))

        cams = [mock.Mock(width=64, height=48)]
        with mock.patch.object(splat_module, "_rasterize_culled", fake_culled), \
                mock.patch("body2colmap.splat_scene.SplatScene.from_ply", classmethod(lambda cls, path: FakeScene())):
            coverage = face_priority._render_coverage("/x/face.ply", cams, width=64, height=48, render_path=None,
                                                      mesh_world=(np.zeros((3, 3)), np.zeros((1, 3), int)), cull_margin=0.02)
        self.assertEqual(len(coverage), 1)
        self.assertEqual(calls, [(0.02, 1, (0.0, 0.0, 0.0), splat_module._RENDER_BINARY)])


if __name__ == "__main__":
    unittest.main()
