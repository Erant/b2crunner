"""colmap_export: which files it writes, where, and with what alpha.

The geometry files themselves (cameras.txt / images.txt / points3D.txt)
are body2colmap's ColmapExporter output; what this step adds, and what is
checked here, is the frames beside them — flat or in the brush layout,
with the mask carried in the alpha channel without being saturated.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import orbit_dataset, run_step

import pipeline.steps  # noqa: F401


class TestColmapExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = orbit_dataset(n_frames=3, n_points=50)
        # A frame with genuinely soft alpha, stored the way it comes off
        # disk (uint8, 0-255) — a splat render's uncertain fringes — which
        # is what the saturation regression below needs to be able to
        # detect.
        cls.soft_ds = orbit_dataset(n_frames=3, n_points=50)
        h, w = cls.soft_ds.images[0].shape[:2]
        ramp = np.linspace(0, 255, h * w).reshape(h, w).astype(np.uint8)
        cls.soft_ds.masks = [ramp.copy() for _ in cls.soft_ds.images]
        cls.masked_ds = cls.ds

    def test_writes_frames_when_images_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_step("colmap_export", 
                {
                    "cameras": self.masked_ds.cameras[:3],
                    "image_names": self.masked_ds.image_names[:3],
                    "points_3d": self.masked_ds.points_3d,
                    "images": self.masked_ds.images[:3],
                    "masks": self.masked_ds.masks[:3],
                },
                {"output_dir": tmp},
            )
            written = sorted(p.name for p in Path(tmp).glob("frame_*.png"))
            self.assertEqual(written, self.masked_ds.image_names[:3])

    def test_brush_layout_puts_frames_and_normals_in_their_own_dirs(self):
        """`layout: brush` is what the deliverable COLMAP dataset uses; the
        flat default is the ComfyUI graph's own layout."""
        frames = 2
        with tempfile.TemporaryDirectory() as tmp:
            run_step("colmap_export", 
                {
                    "cameras": self.masked_ds.cameras[:frames],
                    "image_names": self.masked_ds.image_names[:frames],
                    "points_3d": self.masked_ds.points_3d,
                    "images": self.masked_ds.images[:frames],
                    "masks": self.masked_ds.masks[:frames],
                    "normal_maps": [
                        np.zeros((*img.shape[:2], 3), dtype=np.float32)
                        for img in self.masked_ds.images[:frames]
                    ],
                },
                {"output_dir": tmp, "layout": "brush"},
            )
            root = Path(tmp)
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                self.assertTrue((root / name).exists(), f"{name} belongs at the root")
            self.assertEqual(
                sorted(p.name for p in (root / "images").glob("*.png")),
                self.masked_ds.image_names[:frames],
            )
            self.assertEqual(
                sorted(p.name for p in (root / "normals").glob("*.png")),
                self.masked_ds.image_names[:frames],
            )
            # Nothing left loose beside the .txt files — that is the whole
            # difference from the flat layout.
            self.assertEqual(sorted(root.glob("frame_*.png")), [])

    def test_an_unknown_layout_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_step("colmap_export", 
                    {
                        "cameras": self.ds.cameras[:1],
                        "image_names": self.ds.image_names[:1],
                        "points_3d": self.ds.points_3d,
                    },
                    {"output_dir": tmp, "layout": "sparse0"},
                )

    def test_masks_from_disk_keep_their_soft_edge(self):
        """Regression: alpha was computed as `mask * 255` regardless of the
        mask's range, so a uint8 mask straight off disk saturated to a hard
        binary alpha. See pipeline/masks.py."""
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            run_step("colmap_export", 
                {
                    "cameras": self.soft_ds.cameras[:1],
                    "image_names": self.soft_ds.image_names[:1],
                    "points_3d": self.soft_ds.points_3d,
                    "images": self.soft_ds.images[:1],
                    "masks": self.soft_ds.masks[:1],
                },
                {"output_dir": tmp},
            )
            written = cv2.imread(
                str(Path(tmp) / self.soft_ds.image_names[0]), cv2.IMREAD_UNCHANGED
            )
            self.assertEqual(written.shape[2], 4)
            np.testing.assert_array_equal(written[:, :, 3], self.soft_ds.masks[0])
            # Not a hard 0/255 binary — that is what the bug produced.
            self.assertGreater(len(np.unique(written[:, :, 3])), 50)


if __name__ == "__main__":
    unittest.main()
