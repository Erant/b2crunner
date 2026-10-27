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


class TestFaceDetailRigViews(unittest.TestCase):
    def test_a_colmap_dataset_with_the_sidecars_the_views_have(self):
        import tempfile
        from pathlib import Path

        import cv2
        from body2colmap.camera import Camera

        cams = [Camera(focal_length=(f, f), image_size=(w, h), principal_point=(w / 2, h / 2),
                       position=np.array([0, 0, 2], np.float32), rotation=np.eye(3, dtype=np.float32))
                for f, w, h in ((100.0, 8, 16), (200.0, 16, 32))]
        mask = [np.ones((16, 8), np.float32), np.zeros((32, 16), np.float32)]
        step = get_step_class("face_detail_rig_views")()
        with tempfile.TemporaryDirectory() as d:
            out = Path(step.run({"images": [np.zeros((16, 8, 3), np.uint8), np.zeros((32, 16, 3), np.uint8)],
                                 "image_names": ["frame_00001_.png", "face1_00001.png"], "cameras": cams, "masks": mask,
                                 "weights": [None, np.full((32, 16), 0.5, np.float32)],
                                 "labels": [np.full((16, 8), 3, np.uint8), None],
                                 "points_3d": (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))},
                                step.resolve_params({"output_dir": d + "/views"}))["views_dir"])
            cam_lines = [l for l in (out / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")]
            self.assertEqual(len(cam_lines), 2)   # one per lens
            self.assertEqual(cv2.imread(str(out / "images" / "frame_00001_.png"), cv2.IMREAD_UNCHANGED)[..., 3].min(), 255)
            self.assertEqual(sorted(p.name for p in (out / "weights").iterdir()), ["face1_00001.png"])
            self.assertEqual(sorted(p.name for p in (out / "labels").iterdir()), ["frame_00001_.png"])


if __name__ == "__main__":
    unittest.main()
