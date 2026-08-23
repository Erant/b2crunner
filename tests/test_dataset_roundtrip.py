"""Dataset.to_disk()/from_disk() against real ComfyUI-written data.

The docstring in pipeline/dataset.py claims the on-disk layout is
interchangeable with what nodes/save_dataset_node.py writes and
nodes/load_dataset_node.py reads. cyber_6f/ was written by those nodes, so
loading it here is the actual compatibility check — previously this was only
verified against a synthetic dataset the runner itself produced, which cannot
catch a divergence from the ComfyUI format.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.dataset import Dataset
from tests.helpers import require_stage


class TestDatasetAgainstComfyUIOutput(unittest.TestCase):
    def setUp(self):
        self.stage = require_stage("initial")
        self.ds = Dataset.from_disk(self.stage)

    def test_loads_comfyui_written_dataset(self):
        self.assertEqual(len(self.ds.images), 81)
        self.assertEqual(len(self.ds.cameras), 81)
        self.assertEqual(len(self.ds.image_names), 81)
        self.assertEqual(tuple(self.ds.resolution), (720, 1280))
        # RGBA frames get split into BGR + alpha-as-mask on load.
        self.assertIsNotNone(self.ds.masks)
        self.assertEqual(self.ds.images[0].shape, (1280, 720, 3))
        self.assertEqual(self.ds.masks[0].shape, (1280, 720))
        self.assertIsNotNone(self.ds.reference_image)
        self.assertIsNotNone(self.ds.anchor_image)
        self.assertTrue(self.ds.prompt)

    def test_camera_intrinsics_match_metadata(self):
        import json

        meta = json.loads((self.stage / "metadata.json").read_text())
        for cam, cam_meta in zip(self.ds.cameras, meta["cameras"]):
            self.assertAlmostEqual(cam.fx, cam_meta["intrinsics"]["fx"], places=6)
            self.assertAlmostEqual(cam.fy, cam_meta["intrinsics"]["fy"], places=6)
            self.assertAlmostEqual(cam.cx, cam_meta["intrinsics"]["cx"], places=6)
            self.assertAlmostEqual(cam.cy, cam_meta["intrinsics"]["cy"], places=6)

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
