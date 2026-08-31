"""seedvr2's camera rescale — repairing the intrinsics an upscale invalidates.

The bug this closes is silent by construction: seedvr2 resamples frames but
runs in its own venv, so the IPC boundary carries plain arrays, not Camera
objects. Without a rescale, the dataset ends up self-inconsistent with
nothing raising. The recorded ComfyUI-era export has it too — cyber_6f/
colmap's cameras.txt says 720x1280 next to 1080x1920 frames — so the golden
test in test_colmap_export.py can't catch it either; it compares against
output that has the same mistake.

`_fit_cameras_to_images` (pipeline/steps/seedvr2.py) is what steps/seedvr2.py
calls internally after upscaling, folding the repair into the same step
rather than leaving it to a separate workflow step.

Synthetic data on purpose: the numbers wanted here are exact ratios, and
the whole operation is four multiplications.
"""

from __future__ import annotations

import unittest

import numpy as np
from body2colmap.camera import Camera

from pipeline.steps.seedvr2 import _fit_cameras_to_images


def _cameras(camera_size, count=3):
    width, height = camera_size
    return [
        Camera(
            focal_length=(1213.916992, 1213.916992),
            image_size=(width, height),
            principal_point=(width / 2.0, height / 2.0),
            position=np.array([0.0, 1.0, float(i)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(count)
    ]


def _images(image_size, count=3):
    image_width, image_height = image_size
    return [np.zeros((image_height, image_width, 3), dtype=np.uint8) for _ in range(count)]


class TestFitCamerasToImages(unittest.TestCase):
    def test_a_2x_upscale_scales_every_intrinsic(self):
        cameras, resolution = _fit_cameras_to_images(
            _cameras((720, 1280)), _images((1440, 2560))
        )

        for camera in cameras:
            self.assertEqual((camera.width, camera.height), (1440, 2560))
            self.assertAlmostEqual(camera.fx, 1213.916992 * 2, places=4)
            self.assertAlmostEqual(camera.fy, 1213.916992 * 2, places=4)
            self.assertAlmostEqual(camera.cx, 720.0)
            self.assertAlmostEqual(camera.cy, 1280.0)
        self.assertEqual(resolution, (1440, 2560))

    def test_the_recorded_mismatch_is_what_it_repairs(self):
        """cyber_6f/colmap's exact numbers: 720x1280 intrinsics, 1080x1920
        frames — a 1.5x upscale whose cameras never moved."""
        cameras, _ = _fit_cameras_to_images(_cameras((720, 1280)), _images((1080, 1920)))

        self.assertEqual((cameras[0].width, cameras[0].height), (1080, 1920))
        self.assertAlmostEqual(cameras[0].fx, 1213.916992 * 1.5, places=4)

    def test_poses_are_untouched(self):
        """An upscale resamples the view; it does not move the camera."""
        before = _cameras((720, 1280))
        after, _ = _fit_cameras_to_images(before, _images((1440, 2560)))

        for original, updated in zip(before, after):
            np.testing.assert_array_equal(original.position, updated.position)
            np.testing.assert_array_equal(original.rotation, updated.rotation)

    def test_matching_frames_pass_through_unchanged(self):
        """Safe to call on a dataset that never upscaled."""
        before = _cameras((720, 1280))
        after, resolution = _fit_cameras_to_images(before, _images((720, 1280)))

        self.assertEqual(resolution, (720, 1280))
        for original, updated in zip(before, after):
            self.assertIs(original, updated)

    def test_non_uniform_scaling_is_per_axis(self):
        cameras, _ = _fit_cameras_to_images(_cameras((720, 1280)), _images((1440, 1280)))

        self.assertAlmostEqual(cameras[0].fx, 1213.916992 * 2, places=4)
        self.assertAlmostEqual(cameras[0].fy, 1213.916992, places=4)

    def test_an_empty_dataset_says_so(self):
        with self.assertRaises(ValueError):
            _fit_cameras_to_images(_cameras((720, 1280), count=0), _images((720, 1280), count=0))


if __name__ == "__main__":
    unittest.main()
