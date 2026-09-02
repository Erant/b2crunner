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
from pipeline.steps.splat import (
    _resolve_cameras as _resolve_cameras_raw,
    _resolve_pointcloud as _resolve_pointcloud_raw,
)
from pipeline.steps.splat import _confidence_options
from tests.helpers import (
    redirect_crash_dir,
    require_stage,
    run_step,
    stub_render_binary,
)

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


def _boxed_scene(center, half_extent, n=64):
    """A splat whose Gaussians fill one small box, far from the origin.

    Stands in for the face splat: a head-only shell sitting well above a
    full-body orbit target, and small enough that the two boxes size very
    different radii. `_synthetic_scene`'s normal cloud is centred on the
    origin and cannot show either difference.
    """
    from body2colmap.splat_scene import SplatScene

    rng = np.random.default_rng(1)
    center = np.asarray(center, np.float32)
    half = np.asarray(half_extent, np.float32)
    means = center + rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32) * half
    # Pin two corners so get_bounds() is exactly the box asked for, whatever
    # the sampler happened to draw.
    means[0] = center - half
    means[1] = center + half
    return SplatScene(
        means=means,
        scales=np.full((n, 3), 1e-4, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 0.5, np.float32),
        sh_coeffs=rng.normal(size=(n, 1, 3)).astype(np.float32),
        sh_degree=0,
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

    def test_a_cap_samples_a_disc_around_the_anchor_s_view(self):
        """The supporting-view pattern. Every camera looks at the splat's
        centre from within cap_radius_deg of the direction the photograph
        saw it from — a disc on the orbit sphere, not an orbit."""
        params = {"pattern": "cap", "n_frames": 36, "cap_radius_deg": 30.0,
                  "bounds_source": "splat"}
        cameras, _fl, _mm, anchor = self._resolve(params)
        self.assertEqual(len(cameras), 36)
        self.assertIsNone(anchor)

        target = self.scene.get_bounds()
        target = (target[0] + target[1]) / 2.0
        axis = np.asarray(self.ds.extras["anchor_position"], dtype=np.float64) - target
        axis /= np.linalg.norm(axis)

        angles = []
        for camera in cameras:
            offset = np.asarray(camera.position, dtype=np.float64) - target
            offset /= np.linalg.norm(offset)
            angles.append(np.degrees(np.arccos(np.clip(np.dot(offset, axis), -1, 1))))
        self.assertLessEqual(max(angles), 30.0 + 1e-6)
        # Spread through the disc rather than bunched at its centre or rim.
        self.assertGreater(max(angles), 25.0)
        self.assertLess(min(angles), 6.0)
        # One radius, so the whole set shares a scale: COLMAP export writes
        # a single camera line for all of them.
        radii = [np.linalg.norm(np.asarray(c.position, dtype=np.float64) - target)
                 for c in cameras]
        self.assertAlmostEqual(max(radii), min(radii), places=5)

    def test_a_cap_follows_the_live_anchor_camera_not_the_recorded_position(self):
        """`refine_cameras` rewrites dataset.cameras in place and leaves
        extras["anchor_position"] where the anchor USED to be. The cap is
        sampled around the photograph's view, so it has to read that view
        from the camera list — here the anchor camera is moved to the far
        side of the splat and the disc must go with it."""
        from body2colmap.camera import Camera

        moved = Dataset.from_disk(require_stage("initial"))
        recorded = np.asarray(moved.extras["anchor_position"], dtype=np.float64)
        index = int(np.argmin([np.linalg.norm(np.asarray(c.position, dtype=np.float64)
                                              - recorded) for c in moved.cameras]))
        moved.extras["anchor_frame_index"] = index
        bounds = self.scene.get_bounds()
        target = (bounds[0] + bounds[1]) / 2.0
        old = moved.cameras[index]
        opposite = target - (recorded - target)
        cameras = list(moved.cameras)
        cameras[index] = Camera(
            focal_length=(old.fx, old.fy), image_size=(old.width, old.height),
            principal_point=(old.cx, old.cy), position=opposite.astype(np.float32),
            rotation=np.asarray(old.rotation),
        )
        moved.cameras = cameras

        params = {"pattern": "cap", "n_frames": 16, "cap_radius_deg": 30.0,
                  "bounds_source": "splat"}
        cap, _fl, _mm, _a = self._resolve(params, dataset=moved)

        live_axis = opposite - target
        live_axis /= np.linalg.norm(live_axis)
        recorded_axis = recorded - target
        recorded_axis /= np.linalg.norm(recorded_axis)
        for camera in cap:
            offset = np.asarray(camera.position, dtype=np.float64) - target
            offset /= np.linalg.norm(offset)
            to_live = np.degrees(np.arccos(np.clip(np.dot(offset, live_axis), -1, 1)))
            to_recorded = np.degrees(np.arccos(np.clip(np.dot(offset, recorded_axis), -1, 1)))
            self.assertLessEqual(to_live, 30.0 + 1e-6)       # around the LIVE anchor
            self.assertGreaterEqual(to_recorded, 150.0 - 1e-6)  # nowhere near the recorded one

    def test_a_cap_needs_a_frame_count(self):
        with self.assertRaises(ValueError):
            self._resolve({"pattern": "cap", "bounds_source": "splat"})

    def test_a_cap_without_an_anchor_falls_back_to_the_start_azimuth(self):
        """No anchor means no photograph to sample around; the cap then
        points where the orbits start, rather than guessing. Both anchor
        keys go: the live camera at `anchor_frame_index` is what the cap
        reads first (see `_cap_axis`), and `_ANCHOR_KEYS` treats the pair as
        one fact anyway — a render that drops one drops the other."""
        stripped = Dataset.from_disk(require_stage("initial"))
        stripped.extras = {k: v for k, v in stripped.extras.items()
                           if k not in ("anchor_position", "anchor_frame_index")}
        params = {"pattern": "cap", "n_frames": 8, "cap_radius_deg": 20.0,
                  "bounds_source": "splat", "start_azimuth_deg": 90.0}
        cameras, _fl, _mm, _a = self._resolve(params, dataset=stripped)

        bounds = self.scene.get_bounds()
        target = (bounds[0] + bounds[1]) / 2.0
        axis = np.array([1.0, 0.0, 0.0])  # azimuth 90 deg, elevation 0, Y up
        for camera in cameras:
            offset = np.asarray(camera.position, dtype=np.float64) - target
            offset /= np.linalg.norm(offset)
            self.assertLessEqual(
                np.degrees(np.arccos(np.clip(np.dot(offset, axis), -1, 1))),
                20.0 + 1e-6)

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

    def _radius_about(self, cams, centre):
        d = np.linalg.norm([c.position - centre for c in cams], axis=1)
        return float(d.mean()), float(d.std())

    def test_a_framing_preset_reaims_and_tightens_the_orbit(self):
        """fast_helical_native threads one `framing` global through both the
        mesh `render` and this step, so a non-'full' preset is rendered onto
        cameras re-aimed at that preset and the splat is trained there. This
        step must rebuild its orbit around the SAME box — centred on it, with
        a radius sized from it (a tighter box -> a shorter radius at the
        inherited focal) — not stay on the full-body orbit. The focal length
        is inherited from that render and is not re-framed per preset."""
        full = (np.array([-0.4, -0.9, -2.4], np.float32),
                np.array([0.4, 0.9, -1.7], np.float32))
        head = (np.array([-0.15, 0.35, -2.2], np.float32),
                np.array([0.15, 0.75, -1.9], np.float32))
        bounds = {"full": full, "head": head}

        full_cams, full_px, full_mm, _ = self._framed("full", bounds)
        head_cams, head_px, head_mm, _ = self._framed("head", bounds)

        r_full, s_full = self._radius_about(full_cams, (full[0] + full[1]) / 2.0)
        r_head, s_head = self._radius_about(head_cams, (head[0] + head[1]) / 2.0)

        self.assertLess(s_full, 1e-3, "full orbit is not centred on the full box")
        self.assertLess(s_head, 1e-3, "head orbit is not centred on the head box")
        self.assertLess(r_head, r_full, "a tighter preset did not tighten the radius")

        self.assertAlmostEqual(head_px, full_px, places=5,
                               msg="a preset re-framed the focal length")
        self.assertAlmostEqual(head_mm, full_mm, places=6)
        self.assertAlmostEqual(head_mm, self.ds.extras["focal_length_mm"], places=6)

    def test_full_framing_inherits_the_focal_and_uses_full_bounds(self):
        """`full` is unchanged from before the `framing` global existed: the
        focal length is the one inherited from the dataset, and the orbit is
        sized from the full-body box. Every shipped fast_helical run renders
        at `full`, so this is the path that must not move."""
        full = (np.array([-0.4, -0.9, -2.4], np.float32),
                np.array([0.4, 0.9, -1.7], np.float32))
        _cams, focal_px, mm, _ = self._framed("full", {"full": full})

        inherited = self.ds.extras["focal_length_mm"]
        self.assertAlmostEqual(mm, inherited, places=6)
        self.assertAlmostEqual(focal_px, (inherited / 36.0) * 720, places=4)

    def test_an_explicit_focal_length_wins_over_the_inherited_one(self):
        """focal_length_mm is documented as winning over the inherited value;
        selecting a preset alongside it changes the orbit, not the focal."""
        head = (np.array([-0.15, 0.35, -2.2], np.float32),
                np.array([0.15, 0.75, -1.9], np.float32))
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
        arrays) was dropped entirely — every non-'full' render_splat framing
        then silently fell back to the splat's own bounds. `_json_safe` now
        recurses, so a preset still frames the orbit after a round-trip."""
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

        torso_cams, torso_px, _, _ = _resolve_cameras(
            scene=self.scene, dataset=back,
            params={"pattern": "helical", "n_frames": 8, "framing": "torso"},
            width=720, height=1280,
        )
        full_cams, full_px, _, _ = _resolve_cameras(
            scene=self.scene, dataset=back,
            params={"pattern": "helical", "n_frames": 8, "framing": "full"},
            width=720, height=1280,
        )
        # Both boxes are centred on the origin, so the radius is the tell: the
        # tiny torso box sizes a much shorter orbit. Without the round-trip
        # fix both fall back to the splat's own bounds and these are equal.
        r_torso = float(np.linalg.norm(torso_cams[0].position))
        r_full = float(np.linalg.norm(full_cams[0].position))
        self.assertLess(
            r_torso, r_full * 0.5,
            "framing preset had no effect after a to_disk/from_disk round-trip",
        )
        self.assertAlmostEqual(torso_px, full_px, places=5)


class TestCapDirections(unittest.TestCase):
    """The sampling itself, away from datasets and framing.

    "Uniformly within the circle" has to mean uniform by AREA on the
    sphere: spacing the polar angle evenly instead piles most of the views
    into the middle of the disc, which is the one place the photograph
    already covers.
    """

    AXIS = np.array([0.0, 0.0, 1.0])

    def _angles(self, n, radius_deg=30.0, axis=None):
        from pipeline.steps.splat import _cap_directions

        directions = _cap_directions(n, radius_deg,
                                    self.AXIS if axis is None else axis)
        axis_unit = (self.AXIS if axis is None else np.asarray(axis, float))
        axis_unit = axis_unit / np.linalg.norm(axis_unit)
        return np.degrees([
            np.arccos(np.clip(float(np.dot(d, axis_unit)), -1.0, 1.0))
            for d in directions
        ])

    def test_every_direction_is_a_unit_vector(self):
        from pipeline.steps.splat import _cap_directions

        for direction in _cap_directions(36, 30.0, self.AXIS):
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=9)

    def test_nothing_escapes_the_radius(self):
        self.assertLessEqual(self._angles(200).max(), 30.0 + 1e-9)

    def test_the_spread_is_uniform_by_area(self):
        """Half the samples inside the equal-area median radius, which for
        a small cap is close to R/sqrt(2). Checked against the exact
        median: cos t = 1 - (1 - cos R)/2."""
        radius = 30.0
        median = np.degrees(np.arccos(
            1.0 - (1.0 - np.cos(np.radians(radius))) / 2.0))
        angles = self._angles(400, radius)
        self.assertEqual(int((angles < median).sum()), 200)

    def test_it_does_not_sample_the_axis_itself(self):
        """The source view is the one view a photograph already covers, and
        the training already has a denoised frame there."""
        self.assertGreater(self._angles(36).min(), 0.0)

    def test_the_azimuths_spread_rather_than_stacking(self):
        """The golden angle is what keeps consecutive samples from landing
        on one meridian — spacing by 2*pi/n instead would put them all on a
        handful of spokes."""
        from pipeline.steps.splat import _cap_directions

        directions = _cap_directions(36, 30.0, self.AXIS)
        azimuths = np.degrees([np.arctan2(d[1], d[0]) for d in directions]) % 360.0
        # Every 90-degree quadrant gets its share.
        counts = np.histogram(azimuths, bins=4, range=(0.0, 360.0))[0]
        self.assertTrue((counts >= 36 // 4 - 2).all(), counts)

    def test_the_disc_follows_the_axis(self):
        angles = self._angles(36, 30.0, axis=np.array([1.0, 2.0, -3.0]))
        self.assertLessEqual(angles.max(), 30.0 + 1e-9)

    def test_a_zero_axis_is_refused(self):
        from pipeline.steps.splat import _cap_directions

        with self.assertRaises(ValueError):
            _cap_directions(4, 30.0, np.zeros(3))

    def test_no_frames_is_refused(self):
        from pipeline.steps.splat import _cap_directions

        with self.assertRaises(ValueError):
            _cap_directions(0, 30.0, self.AXIS)


class TestRenderSplatBoundsSource(unittest.TestCase):
    """`bounds_source: splat` — frame the orbit on the splat's own box.

    The default, `dataset`, orbits the source render's `framing` box, which
    is what keeps a re-render lined up with the render it replaces. That is
    only right while the splat IS what the dataset was framed around. The
    face splat is not: it is a head-only shell whose world centre sits a long
    way above a full-body orbit target, so orbiting the body's box points the
    camera at the chest and sizes a radius for a whole person — the head
    lands as a small smudge mid-frame. These check the opt-out aims at and
    sizes from the splat instead, and that it does so without touching the
    intrinsics, since every view in one brush training shares a single COLMAP
    camera line.
    """

    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("initial"))
        # A body-sized box at chest height, and a head-sized one well above it.
        cls.body = (np.array([-0.4, -0.9, -2.4], np.float32),
                    np.array([0.4, 0.9, -1.7], np.float32))
        cls.head_box = (np.array([-0.09, 0.70, -2.15], np.float32),
                        np.array([0.09, 0.88, -1.97], np.float32))
        cls.head_scene = _boxed_scene(
            (cls.head_box[0] + cls.head_box[1]) / 2.0,
            (cls.head_box[1] - cls.head_box[0]) / 2.0,
        )

    def _framed_ds(self):
        import copy

        framed = copy.copy(self.ds)
        framed.extras = dict(self.ds.extras)
        framed.extras["framing_bounds"] = {"full": self.body}
        return framed

    def _resolve(self, **overrides):
        # overlap=0 so no camera is duplicated and the positions are evenly
        # spaced — _orbit() recovers the centre as their mean, which the
        # default overlap=1 would drag toward the repeated first frame.
        params = {"pattern": "circular", "n_frames": 12, "framing": "full",
                  "overlap": 0}
        params.update(overrides)
        return _resolve_cameras(
            scene=self.head_scene, dataset=self._framed_ds(),
            params=params, width=720, height=1280,
        )

    def _orbit(self, cams):
        """(centre, radius) recovered from the cameras themselves."""
        positions = np.array([c.position for c in cams], np.float32)
        centre = positions.mean(axis=0)
        radius = float(np.linalg.norm(positions - centre, axis=1).mean())
        return centre, radius

    def test_splat_bounds_reaim_and_tighten_the_orbit(self):
        """The whole point: the head box, not the body box, decides where the
        camera looks and how close it gets."""
        on_body, _, _, _ = self._resolve()
        on_splat, _, _, _ = self._resolve(bounds_source="splat")

        body_centre = (self.body[0] + self.body[1]) / 2.0
        head_centre = (self.head_box[0] + self.head_box[1]) / 2.0

        got_body, r_body = self._orbit(on_body)
        got_head, r_head = self._orbit(on_splat)

        np.testing.assert_allclose(got_body, body_centre, atol=1e-3)
        np.testing.assert_allclose(got_head, head_centre, atol=1e-3)
        self.assertLess(
            r_head, r_body,
            "the splat's smaller box did not dolly the camera in",
        )

    def test_splat_bounds_leave_the_intrinsics_alone(self):
        """Only the radius moves. body2colmap's ColmapExporter writes ONE
        camera line for a whole training set (cameras[0], stamped CAMERA_ID 1
        on every image), so a focal length that differed per view would be
        silently discarded and the views trained at the wrong lens."""
        body_cams, body_px, body_mm, _ = self._resolve()
        splat_cams, splat_px, splat_mm, _ = self._resolve(bounds_source="splat")

        self.assertAlmostEqual(splat_px, body_px, places=6)
        self.assertAlmostEqual(splat_mm, body_mm, places=6)
        self.assertAlmostEqual(splat_mm, self.ds.extras["focal_length_mm"], places=6)
        self.assertAlmostEqual(splat_cams[0].fx, body_cams[0].fx, places=6)
        self.assertAlmostEqual(splat_cams[0].fy, body_cams[0].fy, places=6)
        self.assertEqual(
            (splat_cams[0].width, splat_cams[0].height),
            (body_cams[0].width, body_cams[0].height),
        )

    def test_framing_preset_is_ignored_rather_than_fallen_back_from(self):
        """A preset the dataset DOES carry is still not consulted — this mode
        is an opt-out, not the existing missing-preset fallback."""
        import copy

        framed = copy.copy(self.ds)
        framed.extras = dict(self.ds.extras)
        framed.extras["framing_bounds"] = {"full": self.body, "head": self.body}

        cams, _, _, _ = _resolve_cameras(
            scene=self.head_scene, dataset=framed,
            params={"pattern": "circular", "n_frames": 12, "framing": "head",
                    "bounds_source": "splat", "overlap": 0},
            width=720, height=1280,
        )
        centre, _ = self._orbit(cams)
        np.testing.assert_allclose(
            centre, (self.head_box[0] + self.head_box[1]) / 2.0, atol=1e-3
        )

    def test_splat_bounds_need_a_pattern(self):
        """With no pattern the dataset's cameras are reused verbatim and no
        box is consulted, so the request would be a silent no-op."""
        with self.assertRaises(ValueError) as caught:
            self._resolve(pattern="", bounds_source="splat")
        self.assertIn("pattern", str(caught.exception))

    def test_splat_bounds_refuse_the_anchored_path(self):
        """The anchored path takes its target and radius from the dataset's
        orbit_target and focal length and never looks at a box — accepting
        both would silently ignore one of them."""
        with self.assertRaises(ValueError) as caught:
            self._resolve(override_cam_from_mesh=True, bounds_source="splat")
        self.assertIn("override_cam_from_mesh", str(caught.exception))

    def test_default_still_frames_on_the_dataset(self):
        """The declared default must leave every shipped workflow's orbit
        exactly where it was."""
        explicit, _, _, _ = self._resolve(bounds_source="dataset")
        default, _, _, _ = self._resolve()
        for a, b in zip(explicit, default):
            np.testing.assert_allclose(a.position, b.position, atol=1e-6)


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
    """The cameras.json schema against brush-splat-render's expected input
    (~/Projects/brush/docs/splat-render.md): row-major rotation,
    body2colmap.Camera's OpenGL-convention c2w passed straight through with
    no conversion (that happens Rust-side).

    The writer is body2colmap's now rather than this project's, but the
    contract still binds every render here, so it is read off the file a
    stub binary was actually handed. A Python-side axis flip would cancel
    against the binary's own and produce a vertically mirrored render that
    looks almost right, which is the failure this exists to catch.
    """

    def test_schema_matches_camera_fields(self):
        payload = _drive_render(cameras=2)["cameras_json"]

        self.assertEqual(payload["width"], 4)
        self.assertEqual(payload["height"], 4)
        self.assertEqual(len(payload["cameras"]), 2)

        entry = payload["cameras"][0]
        self.assertEqual(entry["fx"], 4.0)
        self.assertEqual(entry["fy"], 4.0)
        self.assertEqual(entry["cx"], 2.0)
        self.assertEqual(entry["cy"], 2.0)
        self.assertEqual(entry["position"], [0.0, 0.0, 1.0])
        # rotation[row][col], columns are local axes — identity in, identity
        # out, no axis flip on the Python side.
        self.assertEqual(entry["rotation"],
                         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_the_cameras_are_in_render_order(self):
        """The frames come back indexed against this list, so an order the
        writer invented would pair every frame with the wrong camera."""
        payload = _drive_render(cameras=3)["cameras_json"]
        self.assertEqual([e["position"][2] for e in payload["cameras"]],
                         [1.0, 2.0, 3.0])


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

    def test_the_shipped_default_is_black(self):
        """The default the step applies, read from the step rather than
        restated here — a test that hardcodes (0,0,0) on both sides passes
        even if the step's default goes back to white."""
        from pipeline.steps.splat import RenderSplatStep  # noqa: F401

        default = _render_splat_default_bg()
        self.assertEqual(tuple(default), (0.0, 0.0, 0.0))

    def test_the_background_is_applied_here_rather_than_by_the_binary(self):
        """Where this is enforced moved with the rasterisation. The binary is
        always invoked with `--background 0,0,0` — body2colmap's decision, so
        its RGB comes back premultiplied and both alpha conventions derive
        from one intermediate — and `bg_color` is composited under it in
        Python. So the property to check is the OUTPUT, not the argv."""
        run = _drive_render(bg_color=(0.25, 0.5, 1.0))
        self.assertEqual(run["argv"][run["argv"].index("--background") + 1], "0,0,0")

        # The stub renders fully opaque, so a correct implementation
        # composites nothing: `rgb + bg*(1-a)` with a=1 leaves rgb alone.
        np.testing.assert_array_equal(run["masks"][0], 1.0)
        np.testing.assert_array_equal(run["images"][0], 100)

    def test_a_render_comes_back_as_bgr_plus_a_float_mask(self):
        """This pipeline's convention, and body2colmap's is RGBA — the
        conversion is `_rasterize`'s and is easy to drop silently, since a
        grey test frame looks identical either way."""
        run = _drive_render()
        image, mask = run["images"][0], run["masks"][0]
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, np.float32)


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


def _drive_render(*, bg_color=(0.0, 0.0, 0.0), confidence=None, cameras=1):
    """Run `_rasterize` against a stub binary and report what happened.

    The argv used to be captured by patching `_run_render`, an internal of
    this module. That function is gone: since 2026-08-31 `_rasterize` drives
    body2colmap's `SplatRenderer`, which builds the argv and runs the binary
    itself. So the observation point moved out to the binary — which is the
    better one anyway, being what the real renderer would be handed.

    Returns a dict of `argv`, `cameras_json`, `images` and `masks`.
    """
    from body2colmap.camera import Camera

    from pipeline.steps import splat as splat_module

    camera_list = [
        Camera(
            focal_length=(4.0, 4.0),
            image_size=(4, 4),
            principal_point=(2.0, 2.0),
            position=np.array([0.0, 0.0, float(i + 1)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(cameras)
    ]
    scene = _synthetic_scene()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ply = root / "s.ply"
        run_step("save_splat", {"splat_scene": scene}, {"filepath": str(ply)})
        record = root / "record"
        binary = stub_render_binary(root, record=record)

        images, masks = splat_module._rasterize(
            scene=scene,
            splat_path=str(ply),
            cameras=camera_list,
            image_names=[f"frame_{i + 1:05d}_.png" for i in range(cameras)],
            width=4,
            height=4,
            bg_color=bg_color,
            render_path=binary,
            confidence=confidence,
        )
        return {
            "argv": json.loads((record / "argv.json").read_text()),
            "cameras_json": json.loads((record / "cameras.json").read_text()),
            "images": images,
            "masks": masks,
        }


def _captured_render_argv(**kwargs):
    return _drive_render(**kwargs)["argv"]


def _confidence(**overrides):
    """The `ConfidenceOptions` render_splat would build for these params.

    Through `resolve_params` and the step's own reader, so the declared
    defaults are the ones under test rather than values restated here.
    """
    from pipeline.steps.splat import _confidence_options

    return _confidence_options(_rs_params({"confidence": True, **overrides}))


class TestRenderSplatConfidence(unittest.TestCase):
    """`confidence: true` — the mode that replaces mask_splat.

    brush writes each Gaussian's multi-view evidence into the .ply
    (`export_evidence`), and brush-splat-render gates the render on it,
    handing back the gate as the frame's alpha. The output contract changes
    with it: RGB is composited over the CULL colour, `--background` is
    ignored, and a transparent pixel is grey rather than black. What is
    checked here is the argv, which is the whole of this step's side of that
    seam — the gating itself is the renderer's.
    """

    def test_off_by_default_and_the_old_argv_is_untouched(self):
        """Every render that is not opted in — the face cap views that feed
        select_support_views above all — must still get exactly the
        premultiplied-over-black call it got before."""
        self.assertIs(_confidence_options(_rs_params({})), None)

        argv = _captured_render_argv(bg_color=(0.0, 0.0, 0.0))
        self.assertIn("--background", argv)
        self.assertNotIn("--confidence", argv)
        self.assertNotIn("--cull-color", argv)

    def test_the_gate_flags_carry_the_cull_colour(self):
        """`--cull-color` is the whole background of a gated render: the
        binary resolves culled pixels and the ground it composites over to
        one colour. `--background` is on the argv and ignored by the binary
        in this mode — body2colmap passes it unconditionally so its
        non-gated path has one contract, which is its call to make."""
        argv = _captured_render_argv(confidence=_confidence())

        self.assertIn("--confidence", argv)
        self.assertEqual(argv[argv.index("--cull-color") + 1], "0.5,0.5,0.5")
        self.assertEqual(argv[argv.index("--gate-lo") + 1], "0.45")
        self.assertEqual(argv[argv.index("--gate-hi") + 1], "0.65")

    def test_the_declared_defaults_are_the_shipped_ones(self):
        """Read off the declaration rather than restated, so a changed
        default fails here instead of silently changing every run."""
        declared = get_step_class("render_splat").declared_params()
        self.assertIs(declared["confidence"].default, False)
        self.assertEqual(tuple(declared["cull_color"].default), (0.5, 0.5, 0.5))
        self.assertEqual(declared["gate_lo"].default, 0.45)
        self.assertEqual(declared["gate_hi"].default, 0.65)

    def test_an_inverted_gate_is_refused(self):
        """The gate is a smoothstep from lo to hi. lo above hi does not mean
        'keep less', it is undefined — and it would run, silently."""
        with self.assertRaises(ValueError) as caught:
            _confidence(gate_lo=0.8, gate_hi=0.2)
        self.assertIn("gate_lo", str(caught.exception))

    def test_the_evidence_dataset_carries_the_training_alpha_mode(self):
        """For a .ply that predates export_evidence. The dataset options
        have to match what training saw, and since steps/brush.py stopped
        forcing an alpha mode — so a run can mix masked and transparent
        views — matching means passing none here either: brush then reads
        the mode per view from the dataset's own layout, which for a
        colmap_export directory (RGBA frames, no masks/ sidecar) is the
        transparent this used to spell out."""
        argv = _captured_render_argv(
            confidence=_confidence(evidence_dataset="/data/output/colmap_intermediate")
        )
        self.assertEqual(
            argv[argv.index("--dataset") + 1], "/data/output/colmap_intermediate"
        )
        self.assertNotIn("--alpha-mode", argv)

    def test_no_dataset_flag_without_one(self):
        """The .ply's own ev_* block is the normal source; passing an empty
        --dataset would make the renderer re-measure against nothing."""
        argv = _captured_render_argv(confidence=_confidence())
        self.assertNotIn("--dataset", argv)

    def test_conf_args_go_through_verbatim_and_last(self):
        """The tuning escape hatch for --conf-tau and friends."""
        argv = _captured_render_argv(
            confidence=_confidence(conf_args=["--conf-tau", "0.12"])
        )
        self.assertEqual(argv[-2:], ["--conf-tau", "0.12"])

    def test_the_sidecars_survive_the_temp_directory(self):
        """The binary writes them into a temp directory body2colmap deletes
        on the way out, handing them back only in memory
        (`last_confidence_maps`). This step writes them under the log dir,
        for the reason crash directories go there: whatever gets copied off
        a pod to read the run log brings them along. And they are named the
        way the rest of the run names its frames, not `f00000.conf.png`."""
        # redirect_crash_dir sets B2C_LOG_DIR, which is the whole of what
        # this needs: the sidecars land under the log dir for the reason
        # crash directories do, so redirecting one redirects both and the
        # suite writes nothing into the developer's real volume.
        logs = redirect_crash_dir(self).parent

        run = _drive_render(confidence=_confidence(confidence_sidecar=True),
                            cameras=2)
        self.assertIn("--confidence-sidecar", run["argv"])

        kept = sorted((logs / "confidence").rglob("*.conf.png"))
        self.assertEqual([p.name for p in kept],
                         ["frame_00001_.conf.png", "frame_00002_.conf.png"])

    def test_no_sidecar_flag_by_default(self):
        argv = _captured_render_argv(confidence=_confidence())
        self.assertNotIn("--confidence-sidecar", argv)

    def test_a_sidecar_is_not_mistaken_for_a_frame(self):
        """A `.conf.png` beside each frame must not turn into extra views."""
        redirect_crash_dir(self)
        run = _drive_render(confidence=_confidence(confidence_sidecar=True),
                            cameras=2)
        self.assertEqual(len(run["images"]), 2)
        self.assertEqual(len(run["masks"]), 2)


if __name__ == "__main__":
    unittest.main()
