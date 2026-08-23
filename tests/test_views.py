"""View-manipulation steps, exercised against cyber_6f's real helical orbit.

cyber_6f/initial/ is 81 real cameras with real orbit_target /
forward_azimuth_deg in b2c_extras, which is what makes these tests worth
more than synthetic ones: the azimuth convention (module docstring of
pipeline/steps/views.py) can only be got wrong silently, and a synthetic
orbit built with the same arctan2 assumption the code under test uses would
agree with itself either way.
"""

from __future__ import annotations

import copy
import unittest

import numpy as np

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from pipeline.steps.views import _relative_azimuths, parse_view_indices
from tests.helpers import require_stage

import pipeline.steps  # noqa: F401  (populates the registry)


def _run(step_name: str, inputs, params=None):
    return get_step_class(step_name)().run(inputs, params or {})


class ViewsTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stage = require_stage("initial")
        cls.ds = Dataset.from_disk(stage)


class TestParseViewIndices(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_view_indices("1,2,3"), {1, 2, 3})
        self.assertEqual(parse_view_indices("9-12"), {9, 10, 11, 12})
        self.assertEqual(parse_view_indices("1,3,9-11"), {1, 3, 9, 10, 11})
        self.assertEqual(parse_view_indices(""), set())
        self.assertEqual(parse_view_indices(" 2 , 4 "), {2, 4})

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            parse_view_indices("5-2")
        with self.assertRaises(ValueError):
            parse_view_indices("abc")


class TestDropViews(ViewsTestBase):
    def test_drops_and_keeps_names(self):
        out = _run("drop_views", {"dataset": self.ds}, {"views_to_drop": "1,2,3,9-40"})["dataset"]
        expected = 81 - (3 + 32)
        self.assertEqual(len(out.images), expected)
        self.assertEqual(len(out.cameras), expected)
        self.assertEqual(len(out.masks), expected)
        # Dropping keeps original names so a view stays traceable.
        self.assertEqual(out.image_names[0], "frame_00004_.png")

    def test_empty_spec_is_passthrough(self):
        out = _run("drop_views", {"dataset": self.ds}, {"views_to_drop": ""})["dataset"]
        self.assertIs(out, self.ds)

    def test_rejects_out_of_range_and_dropping_everything(self):
        with self.assertRaises(ValueError):
            _run("drop_views", {"dataset": self.ds}, {"views_to_drop": "82"})
        with self.assertRaises(ValueError):
            _run("drop_views", {"dataset": self.ds}, {"views_to_drop": "1-81"})

    def test_does_not_mutate_input(self):
        before = len(self.ds.images)
        _run("drop_views", {"dataset": self.ds}, {"views_to_drop": "1-40"})
        self.assertEqual(len(self.ds.images), before)


class TestFilterFoV(ViewsTestBase):
    def test_azimuths_cover_the_full_circle(self):
        """Sanity-check the convention against the real orbit before relying
        on it: a helical orbit of 81 frames over 1 loop should visit every
        azimuth, and exactly one frame should sit at the front (0°)."""
        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)
        fwd = float(self.ds.extras["forward_azimuth_deg"])
        az = _relative_azimuths(self.ds.cameras, target, fwd)
        self.assertLess(min(az), -170.0)
        self.assertGreater(max(az), 170.0)
        near_front = [a for a in az if abs(a) < 5.0]
        self.assertTrue(near_front, "no frame near the skeleton's front")

    def test_front_hemisphere_is_about_half(self):
        out = _run("filter_fov", {"dataset": self.ds}, {"azimuth_deg": 0.0, "fov_deg": 180.0})["dataset"]
        self.assertAlmostEqual(len(out.cameras), 41, delta=2)

    def test_narrow_wedge_and_back_are_disjoint(self):
        front = _run("filter_fov", {"dataset": self.ds}, {"azimuth_deg": 0.0, "fov_deg": 90.0})["dataset"]
        back = _run("filter_fov", {"dataset": self.ds}, {"azimuth_deg": 180.0, "fov_deg": 90.0})["dataset"]
        self.assertTrue(set(front.image_names).isdisjoint(set(back.image_names)))
        self.assertLess(len(front.cameras), len(self.ds.cameras))

    def test_wraparound_at_180(self):
        """A cone centred on the back must wrap across the ±180 discontinuity
        rather than keeping only one side of it."""
        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)
        fwd = float(self.ds.extras["forward_azimuth_deg"])
        az = dict(zip(self.ds.image_names, _relative_azimuths(self.ds.cameras, target, fwd)))
        back = _run("filter_fov", {"dataset": self.ds}, {"azimuth_deg": 180.0, "fov_deg": 60.0})["dataset"]
        kept = [az[n] for n in back.image_names]
        self.assertTrue(any(a > 0 for a in kept), "no views kept on the +180 side")
        self.assertTrue(any(a < 0 for a in kept), "no views kept on the -180 side")

    def test_requires_orbit_metadata(self):
        stripped = copy.copy(self.ds)
        stripped.extras = {}
        with self.assertRaises(ValueError) as ctx:
            _run("filter_fov", {"dataset": stripped}, {})
        self.assertIn("orbit_target", str(ctx.exception))

    def test_anchor_frame_sits_exactly_at_the_front(self):
        """forward_azimuth_deg is defined as the azimuth that puts the camera
        back at the original viewpoint, so in an override_cam_from_mesh
        dataset one frame must land at exactly 0.0° — a useful independent
        check that the convention is implemented the same way the render
        node computed the metadata."""
        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)
        fwd = float(self.ds.extras["forward_azimuth_deg"])
        az = _relative_azimuths(self.ds.cameras, target, fwd)
        self.assertAlmostEqual(min(abs(a) for a in az), 0.0, places=6)

    def test_empty_result_raises(self):
        """A cone aimed into the gap between two adjacent views keeps
        nothing, and must raise rather than return an empty dataset."""
        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)
        fwd = float(self.ds.extras["forward_azimuth_deg"])
        az = sorted(_relative_azimuths(self.ds.cameras, target, fwd))
        gap_center = (az[10] + az[11]) / 2.0
        with self.assertRaises(ValueError):
            _run("filter_fov", {"dataset": self.ds},
                 {"azimuth_deg": gap_center, "fov_deg": 0.01})


class TestRotateViews(ViewsTestBase):
    def _azimuth_of_first(self, ds):
        target = np.asarray(ds.extras["orbit_target"], dtype=np.float64)
        fwd = float(ds.extras["forward_azimuth_deg"])
        return _relative_azimuths(ds.cameras, target, fwd)[0]

    def test_puts_requested_azimuth_first(self):
        out = _run("rotate_views", {"dataset": self.ds}, {"start_azimuth_deg": 90.0})["dataset"]
        self.assertEqual(len(out.cameras), 81)
        self.assertAlmostEqual(self._azimuth_of_first(out), 90.0, delta=360.0 / 81)
        self.assertEqual(out.image_names[0], "frame_00001_.png")
        self.assertEqual(out.image_names[-1], "frame_00081_.png")

    def test_is_idempotent(self):
        once = _run("rotate_views", {"dataset": self.ds}, {"start_azimuth_deg": 120.0})["dataset"]
        twice = _run("rotate_views", {"dataset": once}, {"start_azimuth_deg": 120.0})["dataset"]
        for a, b in zip(once.cameras, twice.cameras):
            np.testing.assert_allclose(a.position, b.position, atol=1e-6)

    def test_camera_image_pairing_is_preserved(self):
        """The whole point of rotating: a view's camera must travel with its
        image. Compared as multisets of (position, image identity) pairs,
        because this orbit contains a duplicated camera position (the
        overlap=1 twin) that a position-keyed dict would silently collapse."""
        out = _run("rotate_views", {"dataset": self.ds}, {"start_azimuth_deg": 90.0})["dataset"]
        before = sorted(
            (tuple(np.round(c.position, 6)), id(img))
            for c, img in zip(self.ds.cameras, self.ds.images)
        )
        after = sorted(
            (tuple(np.round(c.position, 6)), id(img))
            for c, img in zip(out.cameras, out.images)
        )
        self.assertEqual(before, after)

    def test_twin_is_split_to_first_and_last(self):
        """This orbit closes on itself (overlap=1), so exactly one pair of
        cameras shares a position. Rotating onto that pair must put one at
        frame_00001 and the other at frame_N, which is what a FirstLast
        diffusion pass consumes."""
        import collections

        positions = [tuple(np.round(c.position, 6)) for c in self.ds.cameras]
        dupes = [p for p, n in collections.Counter(positions).items() if n > 1]
        self.assertEqual(len(dupes), 1, "expected exactly one duplicated camera position")
        twin_pos = dupes[0]
        twin_idx = positions.index(twin_pos)

        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)
        fwd = float(self.ds.extras["forward_azimuth_deg"])
        twin_az = _relative_azimuths(self.ds.cameras, target, fwd)[twin_idx]

        out = _run("rotate_views", {"dataset": self.ds}, {"start_azimuth_deg": twin_az})["dataset"]
        first = tuple(np.round(out.cameras[0].position, 6))
        last = tuple(np.round(out.cameras[-1].position, 6))
        self.assertEqual(first, twin_pos)
        self.assertEqual(last, twin_pos)

    def test_requires_orbit_metadata(self):
        stripped = copy.copy(self.ds)
        stripped.extras = {}
        with self.assertRaises(ValueError):
            _run("rotate_views", {"dataset": stripped}, {})


class TestReplaceViews(ViewsTestBase):
    def test_replaces_all_when_camera_sets_match(self):
        """cyber_6f/circular is a later stage of the same orbit, so every
        camera should match its counterpart and every image get swapped."""
        circular = Dataset.from_disk(require_stage("circular"))
        out = _run(
            "replace_views",
            {"dataset": self.ds, "replacement": circular},
            {"tolerance_pct": 0.1},
        )["dataset"]
        self.assertEqual(len(out.images), len(self.ds.images))
        replaced = sum(
            1 for a, b in zip(out.images, self.ds.images)
            if not np.array_equal(a, b)
        )
        self.assertGreater(replaced, 0)

    def test_no_match_returns_base_unchanged(self):
        far = copy.copy(self.ds)
        far.cameras = []
        for cam in self.ds.cameras:
            moved = copy.copy(cam)
            moved.position = np.asarray(cam.position, dtype=np.float32) + 1000.0
            far.cameras.append(moved)
        out = _run(
            "replace_views",
            {"dataset": self.ds, "replacement": far},
            {"tolerance_pct": 0.1},
        )["dataset"]
        self.assertIs(out, self.ds)

    def test_output_keeps_base_size_and_names(self):
        subset = _run("drop_views", {"dataset": self.ds}, {"views_to_drop": "10-81"})["dataset"]
        out = _run(
            "replace_views",
            {"dataset": self.ds, "replacement": subset},
            {"tolerance_pct": 0.1},
        )["dataset"]
        self.assertEqual(len(out.images), 81)
        self.assertEqual(out.image_names, self.ds.image_names)


class TestMergeDatasets(ViewsTestBase):
    def test_concatenates_and_renumbers(self):
        circular = Dataset.from_disk(require_stage("circular"))
        out = _run(
            "merge_datasets",
            {"datasets": [self.ds, circular]},
            {"pointcloud_mode": "first"},
        )["dataset"]
        self.assertEqual(len(out.images), len(self.ds.images) + len(circular.images))
        self.assertEqual(out.image_names[0], "frame_00001_.png")
        self.assertEqual(out.image_names[-1], f"frame_{len(out.images):05d}_.png")

    def test_numbered_inputs_form(self):
        circular = Dataset.from_disk(require_stage("circular"))
        out = _run(
            "merge_datasets",
            {"dataset_1": self.ds, "dataset_2": circular},
            {},
        )["dataset"]
        self.assertEqual(len(out.cameras), 162)

    def test_pointcloud_modes(self):
        circular = Dataset.from_disk(require_stage("circular"))
        n_first = len(self.ds.points_3d[0])
        n_second = len(circular.points_3d[0])

        first = _run("merge_datasets", {"datasets": [self.ds, circular]},
                     {"pointcloud_mode": "first"})["dataset"]
        self.assertEqual(len(first.points_3d[0]), n_first)

        merged = _run("merge_datasets", {"datasets": [self.ds, circular]},
                      {"pointcloud_mode": "merge"})["dataset"]
        self.assertEqual(len(merged.points_3d[0]), n_first + n_second)

        resampled = _run("merge_datasets", {"datasets": [self.ds, circular]},
                         {"pointcloud_mode": "resample", "pointcloud_samples": 500})["dataset"]
        self.assertEqual(len(resampled.points_3d[0]), 500)
        self.assertEqual(len(resampled.points_3d[1]), 500)

    def test_rejects_resolution_mismatch(self):
        other = copy.copy(self.ds)
        other.resolution = (512, 512)
        with self.assertRaises(ValueError):
            _run("merge_datasets", {"datasets": [self.ds, other]}, {})

    def test_drops_orbit_metadata_so_filter_refuses(self):
        """A merged dataset has no single orbit center, so the azimuth-based
        steps must refuse rather than compute nonsense."""
        circular = Dataset.from_disk(require_stage("circular"))
        merged = _run("merge_datasets", {"datasets": [self.ds, circular]}, {})["dataset"]
        self.assertNotIn("orbit_target", merged.extras)
        with self.assertRaises(ValueError):
            _run("filter_fov", {"dataset": merged}, {})
        with self.assertRaises(ValueError):
            _run("rotate_views", {"dataset": merged}, {})

    def test_requires_two(self):
        with self.assertRaises(ValueError):
            _run("merge_datasets", {"datasets": [self.ds]}, {})


if __name__ == "__main__":
    unittest.main()
