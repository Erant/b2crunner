"""split_reference_sheet — the cut, which half comes out where, and which
layout an upload is.

Synthetic for the arithmetic (a sheet whose two panels are distinguishable
by construction, cut under `layout: sheet` because a uniform panel holds no
figure for `auto` to count), synthetic boxes for `classify_layout`, an
injected detector for the `auto` routing, plus real detections over
cyber_6f's recorded reference.png (a real generated sheet, even though its
*use* there is the older convention — whole sheet into VACE, see
steps/reference_sheet.py) and its anchor.png (a single photo). Those last
two need the 14 MB detector file and skip without it: the module never
downloads in a test.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipeline.registry import get_step_class
from pipeline.steps.reference_sheet import (
    DETECTOR_MODEL_NAME, MIN_FIGURE_HEIGHT, classify_layout, count_figures,
)
from tests.helpers import require_stage, run_step

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
    params.setdefault("layout", "sheet")
    return run_step("split_reference_sheet", {"sheet": sheet}, params)


def _run_auto(sheet, boxes, **params):
    """The step under `layout: auto` with the detector replaced by `boxes`."""
    step_class = get_step_class("split_reference_sheet")
    step = step_class()
    step.count_figures = lambda image, model_path, min_score: list(boxes)
    params["layout"] = "auto"
    return step.run({"sheet": sheet}, step_class.resolve_params(params))


def _figure(x0: float, x1: float, score: float = 0.9, height: int = 64) -> tuple:
    """A person box spanning columns x0..x1 and nearly the whole frame."""
    return (score, x0, 2.0, x1, height - 2.0)


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

    def test_a_portrait_image_is_refused_as_a_sheet(self):
        """A single photo handed in where a sheet was declared. Two portrait
        panels side by side are never taller than wide, so this is
        detectable — and silent otherwise: sam3d_body would happily fit a
        mesh to half a person."""
        with self.assertRaises(ValueError) as caught:
            _run(np.zeros((128, 64, 3), dtype=np.uint8), layout="sheet")
        self.assertIn("front/back sheet", str(caught.exception))
        self.assertIn("layout: single", str(caught.exception))

    def test_a_single_photo_is_the_front_whole_with_no_back(self):
        """`layout: single`: nothing is cut, and `back` is None so the first
        denoise runs without a reference. Landscape or portrait alike — a
        photo's shape is not evidence of anything under an explicit layout."""
        for shape in ((128, 64, 3), (64, 128, 3)):
            with self.subTest(shape=shape):
                photo = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
                out = _run(photo, layout="single")
                np.testing.assert_array_equal(out["front"], photo)
                self.assertTrue(out["front"].flags["C_CONTIGUOUS"])
                self.assertIsNone(out["back"])
                self.assertEqual(out["layout"], "single")

    def test_a_sheet_reports_its_layout(self):
        self.assertEqual(_run(_sheet(_panel(40), _panel(200)))["layout"], "sheet")

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaises(ValueError):
            _run(_sheet(_panel(40), _panel(200)), layout="double")

    def test_an_unknown_front_side_is_refused(self):
        with self.assertRaises(ValueError):
            _run(_sheet(_panel(40), _panel(200)), front_side="middle")


class TestClassifyLayout(unittest.TestCase):
    """The rule over the boxes, with the measured shapes: figures at
    0.87-0.96 and >= 83% of the frame tall, false positives <= 0.52 and
    slivers (steps/reference_sheet.py's module docstring)."""

    W, H = 128, 64

    def classify(self, boxes, width=W, height=H, min_score=0.6):
        return classify_layout(boxes, width, height, min_score)

    def test_two_figures_either_side_of_centre_are_a_sheet(self):
        self.assertEqual(self.classify([_figure(10, 55), _figure(70, 120, 0.87)]), "sheet")

    def test_one_figure_is_a_single_photo(self):
        self.assertEqual(self.classify([_figure(30, 100)]), "single")

    def test_slivers_and_weak_boxes_do_not_count(self):
        """girl_9_16's pattern: one real figure and four low-scoring
        slivers in the background. Both tests reject them — score alone
        would too, but the cyber_6f anchor's 0.38 sliver shows a border can
        score, and height is the cheaper certainty."""
        boxes = [
            _figure(20, 110, 0.91),
            (0.52, 18, 30, 23, 40),          # sliver, low score
            (0.95, 0, 20, 8, 20 + MIN_FIGURE_HEIGHT * self.H - 1),  # confident, too short
        ]
        self.assertEqual(self.classify(boxes), "single")

    def test_a_tall_confident_box_on_each_side_is_what_a_sheet_needs(self):
        """A second figure that scores but sits on the same side as the
        first is not a sheet — nothing says which half is which."""
        with self.assertRaises(ValueError) as caught:
            self.classify([_figure(5, 40), _figure(45, 63, 0.88)])
        self.assertIn("same side", str(caught.exception))
        self.assertIn("input_layout", str(caught.exception))

    def test_nobody_is_refused_not_guessed(self):
        with self.assertRaises(ValueError) as caught:
            self.classify([(0.4, 1, 1, 5, 5)])
        self.assertIn("no figure", str(caught.exception))
        self.assertIn("input_layout", str(caught.exception))

    def test_two_figures_in_a_portrait_frame_are_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.classify(
                [_figure(2, 30, height=128), _figure(34, 62, 0.9, height=128)],
                width=64, height=128,
            )
        self.assertIn("portrait", str(caught.exception))

    def test_min_score_is_the_step_s_threshold(self):
        boxes = [_figure(10, 55, 0.9), _figure(70, 120, 0.7)]
        self.assertEqual(self.classify(boxes, min_score=0.6), "sheet")
        self.assertEqual(self.classify(boxes, min_score=0.8), "single")


class TestAutoLayoutRoutes(unittest.TestCase):
    """`auto` with the detector injected: the verdict picks the branch."""

    def test_two_figures_cut_the_sheet(self):
        front_in, back_in = _panel(40), _panel(200)
        out = _run_auto(_sheet(front_in, back_in), [_figure(4, 28), _figure(36, 60)])
        self.assertEqual(out["layout"], "sheet")
        np.testing.assert_array_equal(out["front"], front_in)
        np.testing.assert_array_equal(out["back"], back_in)

    def test_one_figure_passes_the_photo_through(self):
        photo = _panel(40)
        out = _run_auto(photo, [_figure(4, 28)])
        self.assertEqual(out["layout"], "single")
        np.testing.assert_array_equal(out["front"], photo)
        self.assertIsNone(out["back"])

    def test_the_detector_is_asked_below_the_threshold_so_the_log_shows_the_rejects(self):
        seen = {}
        step_class = get_step_class("split_reference_sheet")
        step = step_class()

        def fake(image, model_path, min_score):
            seen["min_score"] = min_score
            seen["model"] = model_path
            return [_figure(4, 28)]

        step.count_figures = fake
        step.run({"sheet": _panel(40)}, step_class.resolve_params({"layout": "auto", "min_score": 0.6}))
        self.assertLessEqual(seen["min_score"], 0.2)
        self.assertTrue(seen["model"].endswith(DETECTOR_MODEL_NAME))

    def test_min_score_is_a_step_param(self):
        with self.assertRaises(ValueError):
            _run_auto(_panel(40), [_figure(4, 28, 0.7)], min_score=0.8)


def _detector_model():
    """The cached detector file, or a skip: tests do not download."""
    from pipeline.steps.face_landmarks import _model_path

    path = _model_path(DETECTOR_MODEL_NAME)
    if not path.exists():
        raise unittest.SkipTest(f"detector model missing: {path}")
    return str(path)


class TestRealDetectionsOnRecordedImages(unittest.TestCase):
    """The detector itself, on the two recorded images the evaluation used.
    The numbers in steps/reference_sheet.py's docstring came from these."""

    def test_the_recorded_sheet_is_two_figures(self):
        stage = require_stage("initial")
        model = _detector_model()
        sheet = cv2.imread(str(stage / "reference.png"), cv2.IMREAD_COLOR)
        boxes = count_figures(sheet, model, 0.2)
        self.assertEqual(classify_layout(boxes, sheet.shape[1], sheet.shape[0], 0.6), "sheet")
        strong = [b for b in boxes if b[0] >= 0.6]
        self.assertEqual(len(strong), 2, boxes)

    def test_the_recorded_anchor_is_one_figure(self):
        stage = require_stage("initial")
        model = _detector_model()
        photo = cv2.imread(str(stage / "anchor.png"), cv2.IMREAD_COLOR)
        boxes = count_figures(photo, model, 0.2)
        self.assertEqual(classify_layout(boxes, photo.shape[1], photo.shape[0], 0.6), "single")

    def test_auto_on_the_recorded_sheet_cuts_it(self):
        stage = require_stage("initial")
        _detector_model()
        sheet = cv2.imread(str(stage / "reference.png"), cv2.IMREAD_COLOR)
        out = run_step("split_reference_sheet", {"sheet": sheet}, {"layout": "auto"})
        self.assertEqual(out["layout"], "sheet")
        self.assertEqual(out["front"].shape, (1280, 720, 3))


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
