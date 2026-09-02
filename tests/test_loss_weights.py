"""Per-pixel loss weights: the `weights/` sidecar brush reads.

The third thing a training view can carry after its frame and its alpha,
and the only one not about the alpha: a float32 [0,1] map that brush
multiplies into that view's loss pixel by pixel, on top of its alpha mode
(Erant/brush docs/loss-weights.md). `face_priority_weights` is what
produces it. These pin what steps/brush.py and steps/colmap_export.py
write for it — the same greyscale sidecar, named by stem the way masks/
is — and what they refuse.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.helpers import run_step
from tests.test_brush_support_views import _Run, _inputs, _support

import pipeline.steps  # noqa: F401


def _weights(count: int = 2):
    """A different flat weight per view, so the pairing is checkable."""
    return [np.full((8, 8), 0.1 * (i + 1), dtype=np.float32) for i in range(count)]


class TestBrushWritesTheSidecar(unittest.TestCase):
    def test_no_weights_means_no_directory(self):
        """Absence is what "weight 1 everywhere" looks like to brush, and it
        keeps a run without the face branch byte-identical to before."""
        run = _Run(_inputs())
        self.assertFalse(any(name.startswith("weights/") for name in run.files))

    def test_an_empty_list_is_the_same_as_absent(self):
        run = _Run(_inputs(weights=[]))
        self.assertFalse(any(name.startswith("weights/") for name in run.files))

    def test_each_view_gets_a_greyscale_png_named_by_its_stem(self):
        run = _Run(_inputs(weights=_weights()))
        for i, name in enumerate(("frame_00001_", "frame_00002_")):
            img = run.image(f"weights/{name}.png")
            self.assertEqual(img.ndim, 2, "a weight map is one channel")
            self.assertEqual(img.dtype, np.uint8)
            # mask_to_alpha_u8 truncates, as it does for a matte.
            self.assertTrue((img == int(255 * 0.1 * (i + 1) + 1e-6)).all(), name)

    def test_the_training_views_keep_their_alpha_and_no_mask_sidecar(self):
        """A weight map is not a mask: the view stays transparent."""
        run = _Run(_inputs(weights=_weights()))
        frame = run.image("images/frame_00001_.png")
        self.assertEqual(frame.shape[-1], 4)
        self.assertFalse(any(name.startswith("masks/") for name in run.files))
        self.assertNotIn("--alpha-mode", run.argv)

    def test_a_uint8_map_is_written_as_is(self):
        weights = [np.full((8, 8), 200, dtype=np.uint8), np.zeros((8, 8), np.uint8)]
        run = _Run(_inputs(weights=weights))
        self.assertTrue((run.image("weights/frame_00001_.png") == 200).all())
        self.assertTrue((run.image("weights/frame_00002_.png") == 0).all())

    def test_a_count_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "weights has 1 entries"):
            _Run(_inputs(weights=_weights(1)))

    def test_supporting_views_get_no_weight_map(self):
        """The weights are the training views'; a supporting view's mask is
        its weight already (face_priority_weights folds into it)."""
        run = _Run(_inputs(weights=_weights(), **_support(1)))
        written = sorted(name for name in run.files if name.startswith("weights/"))
        self.assertEqual(written, ["weights/frame_00001_.png", "weights/frame_00002_.png"])

    def test_the_crash_report_names_the_directory(self):
        from pipeline.steps.brush import _describe_colmap_export

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "weights").mkdir()
            cv2.imwrite(str(Path(tmp) / "weights" / "a.png"), np.zeros((2, 2), np.uint8))
            self.assertIn("weights/ 1 files", _describe_colmap_export(Path(tmp)))


class TestColmapExportWritesTheSameSidecar(unittest.TestCase):
    """export_colmap_intermediate is a record of what brush saw, so the
    weights ride along in the brush layout and nowhere else."""

    def _export(self, layout: str, **extra):
        inputs = {k: v for k, v in _inputs(**extra).items()}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run_step("colmap_export", inputs, {"output_dir": tmp.name, "layout": layout})
        return Path(tmp.name)

    def test_the_brush_layout_writes_weights_beside_images(self):
        root = self._export("brush", weights=_weights())
        self.assertEqual(
            sorted(p.name for p in (root / "weights").glob("*.png")),
            ["frame_00001_.png", "frame_00002_.png"],
        )
        img = cv2.imread(str(root / "weights" / "frame_00002_.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(img.ndim, 2)
        self.assertTrue((img == int(255 * 0.2 + 1e-6)).all())

    def test_without_weights_no_directory_is_written(self):
        root = self._export("brush")
        self.assertFalse((root / "weights").exists())

    def test_the_flat_layout_refuses_them(self):
        with self.assertRaisesRegex(ValueError, "weights were wired in"):
            self._export("flat", weights=_weights())

    def test_a_count_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "weights has 1 entries"):
            self._export("brush", weights=_weights(1))


if __name__ == "__main__":
    unittest.main()
