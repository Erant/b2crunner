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
import cv2
from unittest import mock

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from pipeline.steps.splat import (
    _resolve_cameras as _resolve_cameras_raw,
    _resolve_pointcloud as _resolve_pointcloud_raw,
)
from tests.helpers import require_stage, run_step

import pipeline.steps  # noqa: F401


def _rs_params(params):
    """render_splat's declared defaults with `params` on top.

    Both helpers below read `params["x"]` and expect the merge the runner
    does before dispatch (see pipeline/runner.py). Doing it here keeps every
    call site in this file passing only the values it is actually testing.
    """
    return get_step_class("render_splat").resolve_params(params)


def _resolve_cameras(*, scene, dataset, params, width, height):
    return _resolve_cameras_raw(
        scene=scene, dataset=dataset, params=_rs_params(params),
        width=width, height=height,
    )


def _resolve_pointcloud(scene, dataset, params):
    return _resolve_pointcloud_raw(scene, dataset, _rs_params(params))


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
            out = run_step("save_splat", {"splat_scene": scene}, {"filepath": path})
            self.assertTrue(Path(out["splat_path"]).exists())

            back = run_step("load_splat", {}, {"filepath": path})["splat_scene"]

        self.assertEqual(len(back), len(scene))
        self.assertEqual(back.sh_degree, scene.sh_degree)
        np.testing.assert_allclose(back.means, scene.means, atol=1e-5)
        np.testing.assert_allclose(back.opacities, scene.opacities, atol=1e-5)
        np.testing.assert_allclose(back.scales, scene.scales, atol=1e-5)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            run_step("load_splat", {}, {"filepath": "/nonexistent/x.ply"})

    def test_load_requires_a_path(self):
        with self.assertRaises(ValueError):
            run_step("load_splat", {}, {})

    def test_save_requires_a_path(self):
        with self.assertRaises(ValueError):
            run_step("save_splat", {"splat_scene": _synthetic_scene()}, {})


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
        # Different bounds -> different orbit centre -> different cameras.
        self.assertFalse(
            np.allclose(with_bounds[0].position, without[0].position),
            "framing_bounds from metadata were ignored",
        )

    def _framed(self, framing, bounds_by_preset):
        import copy

        framed = copy.copy(self.ds)
        framed.extras = dict(self.ds.extras)
        framed.extras["framing_bounds"] = bounds_by_preset
        return _resolve_cameras(
            scene=self.scene, dataset=framed,
            params={"pattern": "helical", "n_frames": 24, "framing": framing},
            width=720, height=1280,
        )

    def test_a_framing_preset_zooms_instead_of_dollying(self):
        """The regression: a preset used to recompute the orbit radius from its
        own (much smaller) bounds, walking the camera in towards the subject
        while leaving the intrinsics at the source render's full-body focal
        length. It must instead stay on the orbit the dataset was framed on and
        lengthen the focal length, as steps/render.py's override branch does —
        the splat is only valid from near the cameras it was trained on, and
        `inject_anchor` matches the anchor frame by camera position."""
        full = (np.array([-0.4, -0.9, -2.4], np.float32),
                np.array([0.4, 0.9, -1.7], np.float32))
        head = (np.array([-0.1, 0.6, -2.15], np.float32),
                np.array([0.1, 0.9, -1.95], np.float32))
        bounds = {"full": full, "head": head}

        full_cams, full_px, _, _ = self._framed("full", bounds)
        head_cams, head_px, _, _ = self._framed("head", bounds)

        def radius(cams, box):
            centre = (box[0] + box[1]) / 2.0
            return float(np.linalg.norm(cams[0].position - centre))

        self.assertAlmostEqual(radius(head_cams, head), radius(full_cams, full),
                               places=5, msg="a preset moved the camera off the orbit")
        self.assertGreater(head_px, full_px * 1.5,
                           "a preset did not lengthen the focal length")
        self.assertAlmostEqual(head_cams[0].fx, head_px, places=5)

    def test_full_framing_is_unchanged_by_the_zoom_path(self):
        """`full` must come out exactly as the radius-based path left it: the
        focal length stays the one inherited from the dataset, because
        `_focal_length_framing` inverts the `compute_auto_orbit_radius` call
        that produced the radius. This is what keeps the shipped workflows —
        all of which render at `full` — byte-for-byte unaffected."""
        full = (np.array([-0.4, -0.9, -2.4], np.float32),
                np.array([0.4, 0.9, -1.7], np.float32))
        _cams, focal_px, mm, _ = self._framed("full", {"full": full})

        inherited = self.ds.extras["focal_length_mm"]
        self.assertAlmostEqual(mm, inherited, places=6)
        self.assertAlmostEqual(focal_px, (inherited / 36.0) * 720, places=4)

    def test_an_explicit_focal_length_is_not_overridden_by_a_preset(self):
        """focal_length_mm is documented as winning over the inherited value;
        a preset must not silently re-frame on top of it."""
        head = (np.array([-0.1, 0.6, -2.15], np.float32),
                np.array([0.1, 0.9, -1.95], np.float32))
        import copy

        framed = copy.copy(self.ds)
        framed.extras = dict(self.ds.extras)
        framed.extras["framing_bounds"] = {"head": head}
        _cams, focal_px, mm, _ = _resolve_cameras(
            scene=self.scene, dataset=framed,
            params={"pattern": "helical", "n_frames": 24, "framing": "head",
                    "focal_length_mm": 50.0},
            width=720, height=1280,
        )
        self.assertAlmostEqual(mm, 50.0)
        self.assertAlmostEqual(focal_px, (50.0 / 36.0) * 720)

    def test_framing_bounds_survive_a_disk_round_trip(self):
        """Regression: Dataset.to_disk's JSON filter only flattened a
        top-level ndarray, so framing_bounds (nested {preset: (min, max)}
        arrays) was dropped entirely. A fast_helical_full run loads its input
        dataset from disk, so every non-'full' render_splat framing silently
        fell back to the whole splat — identical to 'full'."""
        import copy

        framed = copy.copy(self.ds)
        framed.extras = dict(self.ds.extras)
        framed.extras["framing_bounds"] = {
            "full": (np.array([-1.0, -1.0, -1.0], np.float32),
                     np.array([1.0, 1.0, 1.0], np.float32)),
            "torso": (np.array([-0.1, -0.1, -0.1], np.float32),
                      np.array([0.1, 0.1, 0.1], np.float32)),
        }

        with tempfile.TemporaryDirectory() as tmp:
            framed.to_disk(tmp)
            back = Dataset.from_disk(tmp)

        self.assertIn("framing_bounds", back.extras)
        self.assertIn("torso", back.extras["framing_bounds"])

        _torso, torso_px, _, _ = _resolve_cameras(
            scene=self.scene, dataset=back,
            params={"pattern": "helical", "n_frames": 8, "framing": "torso"},
            width=720, height=1280,
        )
        _full, full_px, _, _ = _resolve_cameras(
            scene=self.scene, dataset=back,
            params={"pattern": "helical", "n_frames": 8, "framing": "full"},
            width=720, height=1280,
        )
        # The preset zooms, so the focal length — not the camera position — is
        # what a working preset changes. See
        # test_a_framing_preset_zooms_instead_of_dollying.
        self.assertGreater(
            torso_px, full_px * 1.5,
            "framing preset had no effect after a to_disk/from_disk round-trip",
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


class TestRenderSplatBackground(unittest.TestCase):
    """The splat render's background colour, which is a real output decision.

    It was white, and white is the one value that cannot be right here:
    mask_splat composites this render over black one step later, so a white
    background maximises the gap between the colour a fringe pixel is
    rendered at and the colour it ends at. Those partial-alpha fringes are
    then bilateral-filtered across the silhouette before anything blacks
    them out, so a white halo survived into denoise_pass2's input. Pinned
    because it is a one-line default that is easy to regress and whose
    effect only shows up several steps downstream.
    """

    def _captured_argv(self, bg_color):
        """Build the brush-splat-render argv without running the binary."""
        from body2colmap.camera import Camera

        from pipeline.steps import splat as splat_module

        seen = {}

        def fake_render(cmd):
            seen["cmd"] = cmd
            out = Path(cmd[cmd.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "frame_00001_.png"), np.zeros((4, 4, 4), np.uint8))

        camera = Camera(
            focal_length=(4.0, 4.0),
            image_size=(4, 4),
            principal_point=(2.0, 2.0),
            position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        scene = _synthetic_scene()

        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "s.ply"
            run_step("save_splat", {"splat_scene": scene}, {"filepath": str(ply)})
            with mock.patch.object(splat_module, "_run_render", fake_render):
                splat_module._rasterize(
                    scene=scene,
                    splat_path=str(ply),
                    cameras=[camera],
                    image_names=["frame_00001_.png"],
                    width=4,
                    height=4,
                    bg_color=bg_color,
                    render_path="brush-splat-render",
                )
        return seen["cmd"]

    def test_the_shipped_default_is_black(self):
        """The default the step applies, read from the step rather than
        restated here — a test that hardcodes (0,0,0) on both sides passes
        even if the step's default goes back to white."""
        from pipeline.steps.splat import RenderSplatStep  # noqa: F401

        default = _render_splat_default_bg()
        self.assertEqual(tuple(default), (0.0, 0.0, 0.0))

        argv = self._captured_argv(default)
        self.assertEqual(argv[argv.index("--background") + 1], "0.000000,0.000000,0.000000")

    def test_an_explicit_background_still_wins(self):
        argv = self._captured_argv((0.25, 0.5, 1.0))
        self.assertEqual(argv[argv.index("--background") + 1], "0.250000,0.500000,1.000000")


def _render_splat_default_bg():
    """The bg_color render_splat falls back to, read off its declaration.

    This used to dig the literal out of a `params.get("bg_color", ...)` call
    inside run() with a regex, because that was the only place the default
    existed. It is now declared (pipeline/step.py's Param), so the test can
    simply ask — and still gets the shipped value rather than one restated
    here, which is the property that matters.
    """
    from pipeline.registry import get_step_class

    return tuple(get_step_class("render_splat").declared_params()["bg_color"].default)
