"""colmap_export against the recorded COLMAP directory of the same stage.

cyber_6f/upscaled -> cyber_6f/colmap is a real run of
workflows/api/colmap.json, so this compares the ported step's output
against the files the ComfyUI graph actually produced, rather than against
synthetic data the runner made itself (which is all this step had before).

cameras.txt and points3D.txt come out byte-identical. images.txt agrees to
~2.4e-7 per value: the camera poses have been through metadata.json's
float round-trip, so the recovered quaternions differ in the last printed
digit. The tolerance below is set accordingly — tight enough that a genuine
pose error (a transposed rotation, a world-vs-camera frame mixup) cannot
hide under it.

The exported PNGs are deliberately not compared: the ComfyUI stage runs
RMBG over the frames before export, which needs a GPU model this test
cannot run. Only the geometry files are checked here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from tests.helpers import require_stage, run_step

import pipeline.steps  # noqa: F401


def _data_lines(path: Path):
    return [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


class TestColmapExportGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src, cls.gold = require_stage("upscaled", "colmap")
        cls.ds = Dataset.from_disk(cls.src)
        # upscaled/ frames are RGB (SeedVR2 output has no alpha), so the
        # mask-handling cases below need a stage that actually carries one.
        cls.masked_ds = Dataset.from_disk(require_stage("initial"))
        # splatted/ is the stage with genuinely soft alpha (the splat
        # render's uncertain fringes), which is what the saturation
        # regression below needs to be able to detect.
        cls.soft_ds = Dataset.from_disk(require_stage("splatted"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        run_step("colmap_export", 
            {
                "cameras": cls.ds.cameras,
                "image_names": cls.ds.image_names,
                "points_3d": cls.ds.points_3d,
            },
            {"output_dir": str(cls.out)},
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _compare(self, filename: str, tolerance: float):
        gold = _data_lines(self.gold / filename)
        ours = _data_lines(self.out / filename)
        self.assertEqual(len(gold), len(ours), f"{filename}: line count differs")

        for lineno, (g_line, o_line) in enumerate(zip(gold, ours), start=1):
            g_tokens, o_tokens = g_line.split(), o_line.split()
            self.assertEqual(
                len(g_tokens), len(o_tokens),
                f"{filename}:{lineno}: token count differs",
            )
            for g_tok, o_tok in zip(g_tokens, o_tokens):
                try:
                    g_val = float(g_tok)
                except ValueError:
                    self.assertEqual(g_tok, o_tok, f"{filename}:{lineno}")
                    continue
                self.assertAlmostEqual(
                    g_val, float(o_tok), delta=tolerance,
                    msg=f"{filename}:{lineno}: {g_tok} vs {o_tok}",
                )

    def test_cameras_txt_matches(self):
        """Intrinsics are printed from the same values, so this is exact."""
        self._compare("cameras.txt", tolerance=0.0)

    def test_images_txt_matches(self):
        """Poses agree to within the metadata.json float round-trip."""
        self._compare("images.txt", tolerance=1e-6)

    def test_points3d_txt_matches(self):
        """The point cloud comes straight from pointcloud.npz — exact."""
        self._compare("points3D.txt", tolerance=0.0)

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
        """`layout: brush` is what the deliverable COLMAP dataset uses —
        the flat default is only kept because the golden comparison above
        is against a flat directory."""
        import numpy as np

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
        import numpy as np

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
