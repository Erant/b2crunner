"""face_detail_clip_upscale: a detailer clip swapped for its upscaled frames, matte and loss weights resized."""
import unittest

import numpy as np

import pipeline.steps  # noqa: F401
from pipeline.registry import get_step_class


def _clip(n=3, h=8, w=4):
    masks = [np.zeros((h, w), np.float32) for _ in range(n)]
    for m in masks:
        m[:, : w // 2] = 1.0
    return {"images": [np.zeros((h, w, 3), np.uint8)] * n, "image_names": [f"face1_{i:05d}.png" for i in range(n)],
            "cameras": ["small"] * n, "masks": masks, "weights": [None, np.full((h, w), 0.5, np.float32), None]}


class TestFaceDetailClipUpscale(unittest.TestCase):
    def setUp(self):
        self.step = get_step_class("face_detail_clip_upscale")()

    def test_frames_and_cameras_replaced_matte_and_weights_resized(self):
        big = [np.zeros((16, 8, 3), np.uint8)] * 3
        out = self.step.run({"clip": _clip(), "images": big, "cameras": ["big"] * 3}, self.step.resolve_params({}))["clip"]
        self.assertIs(out["images"][0], big[0])
        self.assertEqual(out["cameras"], ["big"] * 3)
        self.assertEqual(out["image_names"], _clip()["image_names"])
        self.assertEqual(out["masks"][0].shape, (16, 8))
        self.assertEqual(out["masks"][0][:, 0].min(), 1.0)
        self.assertEqual(out["masks"][0][:, -1].max(), 0.0)
        self.assertIsNone(out["weights"][0])
        np.testing.assert_allclose(out["weights"][1], 0.5)

    def test_a_count_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            self.step.run({"clip": _clip(), "images": [np.zeros((16, 8, 3), np.uint8)] * 2, "cameras": ["big"] * 2},
                          self.step.resolve_params({}))


if __name__ == "__main__":
    unittest.main()
