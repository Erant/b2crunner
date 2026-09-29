"""Dataset.to_disk()/from_disk() against the ComfyUI on-disk layout.

The docstring in pipeline/dataset.py claims the on-disk layout is
interchangeable with what nodes/save_dataset_node.py writes and
nodes/load_dataset_node.py reads. A round trip through Dataset alone
cannot catch a divergence from that format — it would agree with itself
either way — so the dataset loaded here is written out by hand, file by
file, the way the save node lays it out: RGBA frames whose alpha is the
mask, metadata.json with per-camera intrinsics/extrinsics, pointcloud.npz,
reference.png, anchor.png and prompt.txt.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.dataset import Dataset

WIDTH, HEIGHT, FRAMES = 9, 16, 3


def _write_comfyui_dataset(root: Path) -> dict:
    """The save node's layout, written without going through Dataset."""
    rng = np.random.default_rng(0)
    cameras = []
    for i in range(FRAMES):
        name = f"frame_{i + 1:05d}_.png"
        rgba = rng.integers(0, 256, size=(HEIGHT, WIDTH, 4), dtype=np.uint8)
        # A soft alpha, not 0/255: the mask is a per-pixel value, and a
        # loader that binarised it would still pass on a hard one.
        rgba[..., 3] = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
        cv2.imwrite(str(root / name), rgba)
        angle = np.radians(40.0 * i)
        rotation = [[float(np.cos(angle)), 0.0, float(np.sin(angle))],
                    [0.0, 1.0, 0.0],
                    [float(-np.sin(angle)), 0.0, float(np.cos(angle))]]
        cameras.append({
            "image_name": name,
            "intrinsics": {"fx": 15.25 + i, "fy": 15.5 + i, "cx": 4.5, "cy": 8.0},
            "extrinsics": {"rotation": rotation,
                           "position": [0.1 * i, 0.25, -2.0 + 0.5 * i]},
        })
    metadata = {
        "version": "1.0",
        "resolution": [WIDTH, HEIGHT],
        "cameras": cameras,
        "b2c_extras": {"focal_length_mm": 60.7, "orbit_target": [0.0, 0.0, -2.0]},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    np.savez_compressed(
        root / "pointcloud.npz",
        positions=rng.normal(size=(50, 3)).astype(np.float32),
        colors=rng.integers(0, 256, size=(50, 3), dtype=np.uint8),
    )
    cv2.imwrite(str(root / "reference.png"), np.full((HEIGHT, 2 * WIDTH, 3), 90, np.uint8))
    cv2.imwrite(str(root / "anchor.png"), np.full((HEIGHT, WIDTH, 3), 200, np.uint8))
    (root / "prompt.txt").write_text("a figure in a silver jacket", encoding="utf-8")
    return metadata


class TestDatasetAgainstComfyUIOutput(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.stage = Path(tmp.name)
        self.metadata = _write_comfyui_dataset(self.stage)
        self.ds = Dataset.from_disk(self.stage)

    def test_loads_comfyui_written_dataset(self):
        self.assertEqual(len(self.ds.images), FRAMES)
        self.assertEqual(len(self.ds.cameras), FRAMES)
        self.assertEqual(len(self.ds.image_names), FRAMES)
        self.assertEqual(tuple(self.ds.resolution), (WIDTH, HEIGHT))
        # RGBA frames get split into BGR + alpha-as-mask on load.
        self.assertIsNotNone(self.ds.masks)
        self.assertEqual(self.ds.images[0].shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(self.ds.masks[0].shape, (HEIGHT, WIDTH))
        self.assertIsNotNone(self.ds.reference_image)
        self.assertIsNotNone(self.ds.anchor_image)
        self.assertTrue(self.ds.prompt)
        self.assertEqual(self.ds.extras["focal_length_mm"], 60.7)

    def test_camera_intrinsics_match_metadata(self):
        for cam, cam_meta in zip(self.ds.cameras, self.metadata["cameras"]):
            self.assertAlmostEqual(cam.fx, cam_meta["intrinsics"]["fx"], places=6)
            self.assertAlmostEqual(cam.fy, cam_meta["intrinsics"]["fy"], places=6)
            self.assertAlmostEqual(cam.cx, cam_meta["intrinsics"]["cx"], places=6)
            self.assertAlmostEqual(cam.cy, cam_meta["intrinsics"]["cy"], places=6)
            np.testing.assert_allclose(
                cam.position, cam_meta["extrinsics"]["position"], atol=1e-6)
            np.testing.assert_allclose(
                cam.rotation, cam_meta["extrinsics"]["rotation"], atol=1e-6)

    def test_roundtrip_preserves_masks(self):
        """Regression: to_disk() used to write self.images unmodified, so a
        dataset loaded from RGBA frames lost its masks on the next save —
        i.e. any save_dataset checkpoint dropped the per-frame
        reference/denoise flag wan22_vace_denoise reads."""
        with tempfile.TemporaryDirectory() as tmp:
            self.ds.to_disk(tmp)
            back = Dataset.from_disk(tmp)
        self.assertIsNotNone(back.masks, "masks lost on to_disk/from_disk round-trip")
        for a, b in zip(self.ds.masks, back.masks):
            np.testing.assert_array_equal(a, b)

    def test_roundtrip_preserves_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.ds.to_disk(tmp)
            back = Dataset.from_disk(tmp)

        self.assertEqual(back.image_names, self.ds.image_names)
        self.assertEqual(tuple(back.resolution), tuple(self.ds.resolution))
        self.assertEqual(back.prompt, self.ds.prompt)
        for a, b in zip(self.ds.images, back.images):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(self.ds.masks, back.masks):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(self.ds.cameras, back.cameras):
            np.testing.assert_allclose(a.position, b.position, atol=1e-6)
            np.testing.assert_allclose(a.rotation, b.rotation, atol=1e-6)
        for a, b in zip(self.ds.points_3d, back.points_3d):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
