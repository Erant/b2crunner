"""split_reference_sheet — the cut, and which half comes out where.

Synthetic for the arithmetic (a sheet whose two panels are distinguishable
by construction), plus one pass over cyber_6f's recorded reference.png,
which is a real generated sheet even though its *use* there is the older
convention (whole sheet into VACE — see steps/reference_sheet.py).
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipeline.registry import get_step_class
from tests.helpers import require_stage

import pipeline.steps  # noqa: F401


def _sheet(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1)


def _panel(value: int, height: int = 64, width: int = 32) -> np.ndarray:
    """A panel that is uniform except for a marked corner, so a half taken
    from the wrong side or flipped is not silently equal to the right one."""
    panel = np.full((height, width, 3), value, dtype=np.uint8)
    panel[:8, :8] = 255 - value
    return panel


def _run(sheet, **params):
    step = get_step_class("split_reference_sheet")()
    return step.run({"sheet": sheet}, params)


class TestSplitReferenceSheet(unittest.TestCase):
    def test_front_is_the_left_panel_by_default(self):
        front_in, back_in = _panel(40), _panel(200)
        out = _run(_sheet(front_in, back_in))
        np.testing.assert_array_equal(out["front"], front_in)
        np.testing.assert_array_equal(out["back"], back_in)

    def test_front_side_right_swaps_them(self):
        back_in, front_in = _panel(40), _panel(200)
        out = _run(_sheet(back_in, front_in), front_side="right")
        np.testing.assert_array_equal(out["front"], front_in)
        np.testing.assert_array_equal(out["back"], back_in)

    def test_an_odd_width_drops_the_centre_column(self):
        """Both halves have to stay the same size: the front half's
        dimensions are what generate_firstlast warps from, and a one-pixel
        difference there is a one-pixel error in the anchor frame."""
        front_in, back_in = _panel(40), _panel(200)
        seam = np.full((64, 1, 3), 7, dtype=np.uint8)
        sheet = np.concatenate([front_in, seam, back_in], axis=1)

        out = _run(sheet)
        self.assertEqual(out["front"].shape, out["back"].shape)
        np.testing.assert_array_equal(out["front"], front_in)
        np.testing.assert_array_equal(out["back"], back_in)

    def test_the_halves_are_contiguous_copies(self):
        """cv2 rejects a non-contiguous src outright in several of the calls
        these halves go on to (warpPerspective among them), and a view would
        keep the whole sheet alive behind each panel."""
        out = _run(_sheet(_panel(40), _panel(200)))
        for key in ("front", "back"):
            with self.subTest(half=key):
                self.assertTrue(out[key].flags["C_CONTIGUOUS"])
                self.assertFalse(np.shares_memory(out[key], out["front"] if key == "back" else out["back"]))

    def test_a_portrait_image_is_refused(self):
        """A single photo handed in where the sheet belongs. Two portrait
        panels side by side are never taller than wide, so this is
        detectable — and silent otherwise: sam3d_body would happily fit a
        mesh to half a person."""
        with self.assertRaises(ValueError) as caught:
            _run(np.zeros((128, 64, 3), dtype=np.uint8))
        self.assertIn("front/back sheet", str(caught.exception))

    def test_an_unknown_front_side_is_refused(self):
        with self.assertRaises(ValueError):
            _run(_sheet(_panel(40), _panel(200)), front_side="middle")


class TestSplitAgainstARecordedSheet(unittest.TestCase):
    def test_the_recorded_sheet_halves_into_two_frame_sized_panels(self):
        """cyber_6f/initial/reference.png is a real generated sheet: 1440x1280,
        two 720x1280 panels, front on the left. What it was *used* for there
        is the older convention (the whole sheet went to VACE), so this
        checks the cut, not the wiring."""
        stage = require_stage("initial")
        sheet = cv2.imread(str(stage / "reference.png"), cv2.IMREAD_COLOR)
        self.assertIsNotNone(sheet)

        out = _run(sheet)
        for key in ("front", "back"):
            with self.subTest(half=key):
                self.assertEqual(out[key].shape, (1280, 720, 3))
        np.testing.assert_array_equal(
            np.concatenate([out["front"], out["back"]], axis=1), sheet
        )


if __name__ == "__main__":
    unittest.main()
