"""refine_cameras' arithmetic — the parts that do not need COLMAP.

The step is mostly a driver for four COLMAP binaries, which no test here
can run. What it does on its own is the half the measurement in
docs/camera-pose-refinement.md says is load-bearing:

  * reading COLMAP's world-to-camera poses back into body2colmap `Camera`s,
  * the Sim(3) that puts the gauge back (trap 1 — without it the model
    comes back 15-24% larger and nothing warns you),
  * and the checks that assert it happened.

So those are what is tested, against synthetic orbits where the right
answer is known exactly. `_align_to` is fed a rig that has been deliberately
scaled, rotated and translated, and has to undo precisely that.
"""

from __future__ import annotations

import unittest

import numpy as np
from body2colmap.camera import Camera

from pipeline.steps.refine_cameras import (
    RefineCamerasStep, _align_to, _check, _movement, _quaternion_to_rotation,
    _read_images_txt, onnx_options_for,
)

TARGET = np.array([0.0, 0.9, 0.0])


def _orbit(count=12, radius=1.8036):
    cameras = []
    for index in range(count):
        angle = np.radians(index * 360.0 / count)
        camera = Camera(
            focal_length=(1066.17, 1066.17), image_size=(720, 1280),
            position=np.array([radius * np.sin(angle), 0.0, radius * np.cos(angle)]),
        )
        camera.look_at(TARGET)
        cameras.append(camera)
    return cameras


def _transform(cameras, scale, degrees, translation):
    angle = np.radians(degrees)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    return [
        Camera(
            focal_length=(camera.fx, camera.fy),
            image_size=(camera.width, camera.height),
            position=scale * (rotation @ np.asarray(camera.position, dtype=np.float64))
            + np.asarray(translation, dtype=np.float64),
            rotation=rotation @ np.asarray(camera.rotation, dtype=np.float64),
        )
        for camera in cameras
    ]


class TestPoseRoundTrip(unittest.TestCase):
    """COLMAP's convention back to the world one body2colmap works in.

    `_read_poses` inverts `world_to_colmap_camera`; if the inversion were
    wrong the refined cameras would be mirrored or upside down, and the
    Sim(3) after it would happily fit a reflection.
    """

    def test_a_camera_survives_the_colmap_convention_and_back(self):
        gl_from_cv = np.diag([1.0, -1.0, -1.0])
        for camera in _orbit():
            quaternion, translation = camera.get_colmap_extrinsics()
            rotation_w2c = _quaternion_to_rotation(np.asarray(quaternion, dtype=np.float64))
            position = -rotation_w2c.T @ np.asarray(translation, dtype=np.float64)
            rotation = rotation_w2c.T @ gl_from_cv

            np.testing.assert_allclose(position, camera.position, atol=1e-5)
            np.testing.assert_allclose(rotation, camera.rotation, atol=1e-5)


class TestGauge(unittest.TestCase):
    def test_a_23_percent_inflation_is_undone_exactly(self):
        """Trap 1, at the top of the observed range (+15.1% to +23.5%)."""
        given = _orbit()
        drifted = _transform(given, scale=1.235, degrees=17.0, translation=[0.4, -0.2, 0.9])

        aligned, (scale, _, _) = _align_to(drifted, given)

        self.assertAlmostEqual(1.0 / scale - 1.0, 0.235, places=6)
        for camera, expected in zip(aligned, given):
            np.testing.assert_allclose(camera.position, expected.position, atol=1e-5)
            np.testing.assert_allclose(camera.rotation, expected.rotation, atol=1e-5)

    def test_a_real_correction_survives_the_alignment(self):
        """The gauge goes back; the per-camera correction must not.

        One camera is lifted 6.9 cm — the largest movement measured on the
        refined dataset — and the whole rig is then inflated. After the
        Sim(3) the lift has to still be there, or this step is a very
        expensive no-op.
        """
        given = _orbit()
        moved = [Camera(focal_length=(c.fx, c.fy), image_size=(c.width, c.height),
                        position=np.asarray(c.position, dtype=np.float64)
                        + (np.array([0.0, 0.069, 0.0]) if i == 3 else 0.0),
                        rotation=np.asarray(c.rotation, dtype=np.float64))
                 for i, c in enumerate(given)]
        drifted = _transform(moved, scale=1.2, degrees=-40.0, translation=[1.0, 2.0, 3.0])

        aligned, _ = _align_to(drifted, given)

        shift = np.linalg.norm(
            np.array([np.asarray(c.position) for c in aligned])
            - np.array([np.asarray(c.position) for c in given]), axis=1)
        # The fit spreads a single outlier over the rig, so frame 3 keeps
        # most of its lift rather than all of it — what matters is that it
        # is still by far the largest movement and the rest are small.
        self.assertEqual(int(np.argmax(shift)), 3)
        self.assertGreater(shift[3], 0.05)
        self.assertLess(np.median(shift), 0.01)


class TestChecks(unittest.TestCase):
    """The doc's check list, as the assertions it asked to become."""

    PARAMS = RefineCamerasStep.resolve_params({})

    def _stats(self, refined, given):
        aligned, transform = _align_to(refined, given)
        return aligned, _movement(aligned, given, transform)

    def test_a_refinement_that_only_moved_a_little_passes(self):
        given = _orbit()
        jitter = np.random.default_rng(7).normal(scale=0.008, size=(len(given), 3))
        refined = [Camera(focal_length=(c.fx, c.fy), image_size=(c.width, c.height),
                          position=np.asarray(c.position, dtype=np.float64) + d,
                          rotation=np.asarray(c.rotation, dtype=np.float64))
                   for c, d in zip(given, jitter)]

        _, stats = self._stats(_transform(refined, 1.19, 5.0, [0.1, 0.1, 0.1]), given)

        self.assertLess(stats["centre_shift"]["mean_frac"], 0.03)
        self.assertEqual(_check(stats, self.PARAMS), "")

    def test_a_re_solved_scene_is_refused(self):
        """Cameras a tenth of the scene radius from where they were is BA
        re-solving rather than refining — the third check in the doc.

        The displacement is TANGENTIAL — each camera slid along the orbit
        by a random angle — so every radius is preserved exactly and the
        scale check has nothing to say. That is the point: this check has
        to fire on its own, on a correction that trap 1's check calls
        perfect.
        """
        given = _orbit()
        rng = np.random.default_rng(11)
        refined = []
        for camera in given:
            angle = np.radians(rng.uniform(-5.0, 5.0))
            spin = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                             [0.0, 1.0, 0.0],
                             [-np.sin(angle), 0.0, np.cos(angle)]])
            refined.append(Camera(
                focal_length=(camera.fx, camera.fy),
                image_size=(camera.width, camera.height),
                position=spin @ np.asarray(camera.position, dtype=np.float64),
                rotation=spin @ np.asarray(camera.rotation, dtype=np.float64)))

        _, stats = self._stats(refined, given)

        self.assertLess(stats["radius_drift"], self.PARAMS["max_scale_drift"])
        self.assertGreater(stats["centre_shift"]["mean_frac"], 0.03)
        self.assertIn("re-solving", _check(stats, self.PARAMS))

    def test_an_unaligned_model_is_refused(self):
        """What trap 1 produces if step 5 is skipped: same shape, +19%.

        `_check` is fed the movement of the *un*aligned model, which is
        what the step would publish if `_align_to` were removed.
        """
        given = _orbit()
        drifted = _transform(given, scale=1.19, degrees=0.0, translation=[0.0, 0.0, 0.0])

        stats = _movement(drifted, given, (1.0, np.eye(3), np.zeros(3)))

        self.assertAlmostEqual(stats["radius_drift"], 0.19, places=5)
        self.assertIn("trap 1", _check(stats, self.PARAMS))

    def test_keep_given_publishes_the_poses_it_was_handed(self):
        given = _orbit()
        step = RefineCamerasStep()

        result = step._give_up(given, self.PARAMS, "refine_cameras", "a bad solve")

        self.assertIs(result["cameras"], given)
        self.assertFalse(result["stats"]["accepted"])
        self.assertEqual(result["stats"]["failure"], "a bad solve")

    def test_raise_stops_the_run_instead(self):
        params = RefineCamerasStep.resolve_params({"on_check_failure": "raise"})

        with self.assertRaises(RuntimeError) as caught:
            RefineCamerasStep()._give_up(_orbit(), params, "refine_cameras", "a bad solve")

        self.assertIn("a bad solve", str(caught.exception))


class TestImagesTxt(unittest.TestCase):
    def test_the_points2d_line_of_each_pair_is_skipped(self):
        """COLMAP writes two lines per image and the second is the
        observations, which this step has no use for — but miscounting it
        would silently read every other pose as an image name."""
        import tempfile
        from pathlib import Path

        text = (
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "1 1.0 0.0 0.0 0.0 0.1 0.2 0.3 1 frame_00000_.png\n"
            "10.5 20.5 -1\n"
            "2 0.0 1.0 0.0 0.0 0.4 0.5 0.6 1 frame_00001_.png\n"
            "\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "images.txt"
            path.write_text(text)
            poses = _read_images_txt(path)

        self.assertEqual(sorted(poses), ["frame_00000_.png", "frame_00001_.png"])
        np.testing.assert_allclose(poses["frame_00001_.png"][1], [0.4, 0.5, 0.6])


class TestOnnxModelSelection(unittest.TestCase):
    """Which weights a type needs — the table `pipeline/models.py` prefetches
    off, so a type whose model is missing from it would download mid-run."""

    def test_the_shipped_pair_names_both_of_its_models(self):
        self.assertEqual(
            onnx_options_for("ALIKED_N32", "ALIKED_LIGHTGLUE"),
            ["AlikedExtraction.n32_model_path", "AlikedMatching.lightglue_model_path"],
        )

    def test_sift_extraction_needs_no_onnx_model(self):
        self.assertEqual(onnx_options_for("SIFT", "SIFT"), [])


if __name__ == "__main__":
    unittest.main()
