"""load_splat / save_splat, and render_splat's camera-path resolution.

What is covered here is everything in render_splat except the gsplat
rasterisation call itself: which cameras get built, at what focal length,
with which framing bounds, and what happens to the point cloud and the
inherited metadata. That is where all of this step's subtlety lives —
`SplatRenderer.render()` is one unmodified body2colmap call that needs a
CUDA device, and stays unverified until a pod run.

The anchored-override case is checked against cyber_6f's real recorded
metadata: rebuilding the path from the dataset's own orbit_target and
focal_length_mm has to reproduce the camera intrinsics and the anchor
position that dataset already records.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from pipeline.steps.splat import _resolve_cameras, _resolve_pointcloud
from tests.helpers import require_stage

import pipeline.steps  # noqa: F401


def _synthetic_scene(n=64, sh_degree=0):
    from body2colmap.splat_scene import SplatScene

    rng = np.random.default_rng(0)
    n_sh = (sh_degree + 1) ** 2
    return SplatScene(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        scales=np.abs(rng.normal(size=(n, 3))).astype(np.float32) * 0.01,
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=rng.uniform(size=(n,)).astype(np.float32),
        sh_coeffs=rng.normal(size=(n, n_sh, 3)).astype(np.float32),
        sh_degree=sh_degree,
    )


class TestSplatIO(unittest.TestCase):
    def setUp(self):
        # PLY I/O lives behind body2colmap's "splat" extra (plyfile), which
        # is not part of the default install — see requirements.txt.
        try:
            import plyfile  # noqa: F401
        except ImportError:
            self.skipTest("plyfile not installed (body2colmap[splat])")

    def test_save_load_roundtrip(self):
        scene = _synthetic_scene()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nested" / "splat.ply")
            out = get_step_class("save_splat")().run({"splat_scene": scene}, {"filepath": path})
            self.assertTrue(Path(out["splat_path"]).exists())

            back = get_step_class("load_splat")().run({}, {"filepath": path})["splat_scene"]

        self.assertEqual(len(back), len(scene))
        self.assertEqual(back.sh_degree, scene.sh_degree)
        np.testing.assert_allclose(back.means, scene.means, atol=1e-5)
        np.testing.assert_allclose(back.opacities, scene.opacities, atol=1e-5)
        np.testing.assert_allclose(back.scales, scene.scales, atol=1e-5)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            get_step_class("load_splat")().run({}, {"filepath": "/nonexistent/x.ply"})

    def test_load_requires_a_path(self):
        with self.assertRaises(ValueError):
            get_step_class("load_splat")().run({}, {})

    def test_save_requires_a_path(self):
        with self.assertRaises(ValueError):
            get_step_class("save_splat")().run({"splat_scene": _synthetic_scene()}, {})


class TestRenderSplatCameras(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("initial"))
        cls.scene = _synthetic_scene()

    def _resolve(self, params, dataset=None):
        return _resolve_cameras(
            scene=self.scene,
            dataset=self.ds if dataset is None else dataset,
            params=params,
            width=params.get("width", 720),
            height=params.get("height", 1280),
        )

    def test_reuses_dataset_cameras_without_a_pattern(self):
        cameras, _fl, mm, anchor = self._resolve({})
        self.assertEqual(len(cameras), len(self.ds.cameras))
        self.assertIs(cameras[0], self.ds.cameras[0])
        self.assertIsNone(anchor)
        # focal_length_mm inherited from the dataset, not recomputed
        self.assertAlmostEqual(mm, self.ds.extras["focal_length_mm"])

    def test_reuse_requires_a_dataset(self):
        with self.assertRaises(ValueError):
            _resolve_cameras(
                scene=self.scene, dataset=None, params={}, width=720, height=1280
            )

    def test_inherited_focal_length_reproduces_recorded_intrinsics(self):
        """The dataset records focal_length_mm=60.6958 and cameras with
        fx=1213.917px at 720 wide. Inheriting the former must reproduce the
        latter exactly, or a re-render silently reframes."""
        _cams, focal_px, _mm, _a = self._resolve({})
        self.assertAlmostEqual(focal_px, self.ds.cameras[0].fx, places=6)

    def test_new_path_builds_requested_frame_count(self):
        for pattern, extra in (
            ("circular", {"elevation_deg": 10.0}),
            ("sinusoidal", {"amplitude_deg": 30.0, "n_cycles": 2}),
            ("helical", {"n_loops": 1, "amplitude_deg": 30.0}),
        ):
            with self.subTest(pattern=pattern):
                params = {"pattern": pattern, "n_frames": 24, **extra}
                cameras, _fl, _mm, anchor = self._resolve(params)
                self.assertGreaterEqual(len(cameras), 24)
                self.assertIsNone(anchor)

    def test_unknown_pattern_raises(self):
        with self.assertRaises(ValueError):
            self._resolve({"pattern": "spiral", "n_frames": 8})

    def test_anchored_path_lands_on_the_recorded_anchor_position(self):
        """The whole point of override mode: rebuilt from the dataset's own
        orbit_target and focal length, one camera must sit exactly where the
        dataset says its anchor is — otherwise the warped reference image no
        longer matches the frame it is injected into."""
        params = {
            "pattern": "helical",
            "n_frames": 81,
            "n_loops": 1,
            "amplitude_deg": 30.0,
            "override_cam_from_mesh": True,
        }
        cameras, _fl, _mm, anchor = self._resolve(params)
        self.assertIsNotNone(anchor)

        recorded = np.asarray(self.ds.extras["anchor_position"], dtype=np.float32)
        np.testing.assert_allclose(cameras[anchor].position, recorded, atol=1e-5)

    def test_anchored_circular_anchors_frame_zero(self):
        params = {
            "pattern": "circular",
            "n_frames": 36,
            "override_cam_from_mesh": True,
        }
        cameras, _fl, _mm, anchor = self._resolve(params)
        self.assertEqual(anchor, 0)
        recorded = np.asarray(self.ds.extras["anchor_position"], dtype=np.float32)
        np.testing.assert_allclose(cameras[0].position, recorded, atol=1e-5)

    def test_override_rejects_camera_reuse(self):
        with self.assertRaises(ValueError) as ctx:
            self._resolve({"override_cam_from_mesh": True})
        self.assertIn("pattern", str(ctx.exception))

    def test_override_rejects_sinusoidal(self):
        with self.assertRaises(ValueError):
            self._resolve({
                "pattern": "sinusoidal", "n_frames": 8, "amplitude_deg": 30.0,
                "n_cycles": 2, "override_cam_from_mesh": True,
            })

    def test_override_rejects_a_non_override_dataset(self):
        """A normally-rendered dataset has been auto-oriented, so the original
        camera is not at the origin and anchoring would be silently wrong.
        original_focal_length is the marker for override mode."""
        import copy

        plain = copy.copy(self.ds)
        plain.extras = {
            k: v for k, v in self.ds.extras.items() if k != "original_focal_length"
        }
        with self.assertRaises(ValueError) as ctx:
            _resolve_cameras(
                scene=self.scene, dataset=plain,
                params={"pattern": "circular", "n_frames": 8,
                        "override_cam_from_mesh": True},
                width=720, height=1280,
            )
        self.assertIn("original_focal_length", str(ctx.exception))

    def test_override_without_any_focal_length_raises(self):
        import copy

        no_mm = copy.copy(self.ds)
        no_mm.extras = {k: v for k, v in self.ds.extras.items() if k != "focal_length_mm"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_cameras(
                scene=self.scene, dataset=no_mm,
                params={"pattern": "circular", "n_frames": 8,
                        "override_cam_from_mesh": True},
                width=720, height=1280,
            )
        self.assertIn("focal length", str(ctx.exception))

    def test_explicit_focal_length_wins_over_inherited(self):
        _cams, focal_px, mm, _a = self._resolve(
            {"pattern": "circular", "n_frames": 8, "focal_length_mm": 50.0}
        )
        self.assertAlmostEqual(mm, 50.0)
        self.assertAlmostEqual(focal_px, (50.0 / 36.0) * 720)

    def test_framing_bounds_from_metadata_are_used(self):
        """A re-render frames itself with the mesh render's bounds so it lines
        up with the render it replaces, rather than with the splat's own
        (noisier) extent."""
        import copy

        framed = copy.copy(self.ds)
        framed.extras = dict(self.ds.extras)
        tight = (np.array([-0.1, -0.1, -0.1], np.float32),
                 np.array([0.1, 0.1, 0.1], np.float32))
        framed.extras["framing_bounds"] = {"torso": tight}

        with_bounds, _, _, _ = _resolve_cameras(
            scene=self.scene, dataset=framed,
            params={"pattern": "circular", "n_frames": 8, "framing": "torso"},
            width=720, height=1280,
        )
        without, _, _, _ = _resolve_cameras(
            scene=self.scene, dataset=framed,
            params={"pattern": "circular", "n_frames": 8, "framing": "full"},
            width=720, height=1280,
        )
        # Different bounds -> different orbit centre/radius -> different cameras.
        self.assertFalse(
            np.allclose(with_bounds[0].position, without[0].position),
            "framing_bounds from metadata were ignored",
        )


class TestRenderSplatPointcloud(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("initial"))
        cls.scene = _synthetic_scene()

    def test_preserves_dataset_pointcloud_by_default(self):
        points = _resolve_pointcloud(self.scene, self.ds, {})
        np.testing.assert_array_equal(points[0], self.ds.points_3d[0])

    def test_override_samples_from_the_splat(self):
        points, colors = _resolve_pointcloud(
            self.scene, self.ds, {"override_pointcloud": True, "pointcloud_samples": 32}
        )
        self.assertEqual(len(points), 32)
        self.assertEqual(len(colors), 32)

    def test_samples_when_there_is_no_dataset(self):
        points, _ = _resolve_pointcloud(self.scene, None, {"pointcloud_samples": 16})
        self.assertEqual(len(points), 16)


class TestCamerasJson(unittest.TestCase):
    """_write_cameras_json's schema against brush-splat-render's expected
    input (~/Projects/brush/docs/splat-render.md): row-major rotation,
    body2colmap.Camera's OpenGL-convention c2w passed straight through with
    no conversion (that happens Rust-side).
    """

    def test_schema_matches_camera_fields(self):
        from body2colmap.camera import Camera

        from pipeline.steps.splat import _write_cameras_json

        camera = Camera(
            focal_length=(1213.917, 1213.917),
            image_size=(720, 1280),
            principal_point=(360.0, 640.0),
            position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            rotation=np.array(
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            _write_cameras_json([camera], ["frame_00001_.png"], 720, 1280, path)
            payload = json.loads(path.read_text())

        self.assertEqual(payload["width"], 720)
        self.assertEqual(payload["height"], 1280)
        self.assertEqual(len(payload["cameras"]), 1)

        entry = payload["cameras"][0]
        self.assertEqual(entry["name"], "frame_00001_.png")
        self.assertEqual(entry["fx"], 1213.917)
        self.assertEqual(entry["fy"], 1213.917)
        self.assertEqual(entry["cx"], 360.0)
        self.assertEqual(entry["cy"], 640.0)
        self.assertEqual(entry["position"], [1.0, 2.0, 3.0])
        # rotation[row][col], columns are local axes — unchanged from
        # camera.rotation, no axis flip on the Python side.
        self.assertEqual(entry["rotation"], camera.rotation.tolist())


if __name__ == "__main__":
    unittest.main()
