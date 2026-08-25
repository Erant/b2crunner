"""Starting a run from something other than a complete on-disk dataset.

Two entry points the pipeline did not have before: a Dataset built from a
bare reference photo (the from-scratch path — no shipped workflow consumes
it now that fast_helical_native.yaml is gone, but it is the hard half of
putting one back), and locating the dataset root inside an uploaded
archive.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from pipeline.dataset import Dataset, find_dataset_root


class TestFromReferenceImage(unittest.TestCase):
    def test_from_an_array(self):
        photo = np.full((480, 640, 3), 128, np.uint8)
        dataset = Dataset.from_reference_image(photo, prompt="a person")

        self.assertEqual(dataset.resolution, (640, 480))  # (width, height)
        self.assertEqual(dataset.reference_image.shape, (480, 640, 3))
        self.assertEqual(dataset.prompt, "a person")
        self.assertEqual(dataset.images, [])
        self.assertEqual(dataset.cameras, [])

    def test_empty_point_cloud_has_a_shape_not_a_none(self):
        """A step reaching for points_3d too early should see (0, 3), not None."""
        dataset = Dataset.from_reference_image(np.zeros((8, 8, 3), np.uint8))
        positions, colors = dataset.points_3d
        self.assertEqual(positions.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))
        self.assertEqual(positions.dtype, np.float32)

    def test_from_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo.png"
            cv2.imwrite(str(path), np.full((100, 50, 3), 200, np.uint8))
            dataset = Dataset.from_reference_image(path)
        self.assertEqual(dataset.resolution, (50, 100))

    def test_alpha_and_greyscale_inputs_become_bgr(self):
        rgba = np.zeros((10, 10, 4), np.uint8)
        self.assertEqual(Dataset.from_reference_image(rgba).reference_image.shape, (10, 10, 3))
        grey = np.zeros((10, 10), np.uint8)
        self.assertEqual(Dataset.from_reference_image(grey).reference_image.shape, (10, 10, 3))

    def test_a_missing_file_says_so(self):
        with self.assertRaises(FileNotFoundError):
            Dataset.from_reference_image("/nonexistent/photo.png")


class TestFindDatasetRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _make_dataset(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text('{"resolution": [8, 8], "cameras": []}')
        return directory

    def test_the_directory_itself(self):
        root = self._make_dataset(self.tmp / "initial")
        self.assertEqual(find_dataset_root(root), root)

    def test_one_level_down(self):
        """A zip of `initial/` unpacks to `<extract>/initial/metadata.json`."""
        root = self._make_dataset(self.tmp / "extract" / "initial")
        self.assertEqual(find_dataset_root(self.tmp / "extract"), root)

    def test_macos_resource_forks_are_ignored(self):
        root = self._make_dataset(self.tmp / "extract" / "initial")
        junk = self.tmp / "extract" / "__MACOSX" / "initial"
        junk.mkdir(parents=True)
        (junk / "metadata.json").write_text("{}")
        self.assertEqual(find_dataset_root(self.tmp / "extract"), root)

    def test_the_shallowest_dataset_wins(self):
        """An output dir containing a checkpoint is still one dataset."""
        outer = self._make_dataset(self.tmp / "run")
        self._make_dataset(self.tmp / "run" / "checkpoint")
        self.assertEqual(find_dataset_root(self.tmp / "run"), outer)

    def test_no_dataset_anywhere_is_a_clear_error(self):
        (self.tmp / "junk").mkdir()
        (self.tmp / "junk" / "notes.txt").write_text("hello")
        with self.assertRaises(FileNotFoundError) as caught:
            find_dataset_root(self.tmp / "junk")
        self.assertIn("notes.txt", str(caught.exception))

    def test_against_a_real_zip_round_trip(self):
        self._make_dataset(self.tmp / "src" / "initial")
        archive = self.tmp / "dataset.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(self.tmp / "src" / "initial" / "metadata.json", "initial/metadata.json")

        extracted = self.tmp / "extracted"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        self.assertEqual(find_dataset_root(extracted), extracted / "initial")


if __name__ == "__main__":
    unittest.main()
