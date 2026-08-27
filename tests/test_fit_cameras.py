"""fit_cameras_to_images — repairing the intrinsics an upscale invalidates.

The bug this closes is silent by construction: seedvr2 returns bigger
frames and no cameras (it runs in its own venv, and that IPC boundary
carries arrays, not Camera objects), so the dataset ends up self-
inconsistent with nothing raising. The recorded ComfyUI-era export has it
too — cyber_6f/colmap's cameras.txt says 720x1280 next to 1080x1920
frames — so the golden test in test_colmap_export.py can't catch it
either; it compares against output that has the same mistake.

Synthetic data on purpose: the numbers wanted here are exact ratios, and
the whole operation is four multiplications.
"""

from __future__ import annotations

import unittest

import numpy as np
from body2colmap.camera import Camera

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class

from tests.helpers import run_step

import pipeline.steps  # noqa: F401


def _dataset(camera_size, image_size, count=3):
    width, height = camera_size
    image_width, image_height = image_size
    cameras = [
        Camera(
            focal_length=(1213.916992, 1213.916992),
            image_size=(width, height),
            principal_point=(width / 2.0, height / 2.0),
            position=np.array([0.0, 1.0, float(i)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(count)
    ]
    return Dataset(
        images=[
            np.zeros((image_height, image_width, 3), dtype=np.uint8) for _ in range(count)
        ],
        image_names=[f"frame_{i + 1:05d}_.png" for i in range(count)],
        cameras=cameras,
        points_3d=(np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)),
        resolution=(width, height),
        extras={"orbit_target": [0.0, 1.0, 0.0]},
    )


def _run(dataset):
    return run_step("fit_cameras_to_images", {"dataset": dataset}, {})["dataset"]


class TestFitCamerasToImages(unittest.TestCase):
    def test_a_2x_upscale_scales_every_intrinsic(self):
        out = _run(_dataset((720, 1280), (1440, 2560)))

        for camera in out.cameras:
            self.assertEqual((camera.width, camera.height), (1440, 2560))
            self.assertAlmostEqual(camera.fx, 1213.916992 * 2, places=4)
            self.assertAlmostEqual(camera.fy, 1213.916992 * 2, places=4)
            self.assertAlmostEqual(camera.cx, 720.0)
            self.assertAlmostEqual(camera.cy, 1280.0)
        self.assertEqual(out.resolution, (1440, 2560))

    def test_the_recorded_mismatch_is_what_it_repairs(self):
        """cyber_6f/colmap's exact numbers: 720x1280 intrinsics, 1080x1920
        frames — a 1.5x upscale whose cameras never moved."""
        out = _run(_dataset((720, 1280), (1080, 1920)))

        self.assertEqual((out.cameras[0].width, out.cameras[0].height), (1080, 1920))
        self.assertAlmostEqual(out.cameras[0].fx, 1213.916992 * 1.5, places=4)

    def test_poses_are_untouched(self):
        """An upscale resamples the view; it does not move the camera."""
        before = _dataset((720, 1280), (1440, 2560))
        after = _run(before)

        for original, updated in zip(before.cameras, after.cameras):
            np.testing.assert_array_equal(original.position, updated.position)
            np.testing.assert_array_equal(original.rotation, updated.rotation)

    def test_matching_frames_pass_through_unchanged(self):
        """Safe to leave in a workflow that does not upscale."""
        before = _dataset((720, 1280), (720, 1280))
        after = _run(before)

        self.assertEqual(after.resolution, (720, 1280))
        for original, updated in zip(before.cameras, after.cameras):
            self.assertIs(original, updated)

    def test_non_uniform_scaling_is_per_axis(self):
        out = _run(_dataset((720, 1280), (1440, 1280)))

        self.assertAlmostEqual(out.cameras[0].fx, 1213.916992 * 2, places=4)
        self.assertAlmostEqual(out.cameras[0].fy, 1213.916992, places=4)

    def test_everything_else_survives(self):
        before = _dataset((720, 1280), (1440, 2560))
        before.prompt = "a woman in a red jacket"
        before.splat_path = "/data/out/ply/scene.ply"
        after = _run(before)

        self.assertEqual(after.image_names, before.image_names)
        self.assertEqual(after.prompt, before.prompt)
        self.assertEqual(after.splat_path, before.splat_path)
        self.assertEqual(after.extras["orbit_target"], [0.0, 1.0, 0.0])

    def test_an_empty_dataset_says_so(self):
        empty = _dataset((720, 1280), (720, 1280), count=0)
        with self.assertRaises(ValueError):
            _run(empty)


if __name__ == "__main__":
    unittest.main()
