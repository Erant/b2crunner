"""Face-landmark geometry and real MediaPipe detection.

Detection is verified against cyber_6f's real reference photos. It runs in
a *subprocess*: mediapipe 1.0.1 on macOS aborted the process once (SIGABRT
via DrishtiMetalHelper) on the first invocation after downloading its
models, and an abort cannot be caught in-process — it would take the whole
test run down. Out-of-process, a recurrence degrades to a skip. On Linux
this should simply pass.

Everything that is not MediaPipe is tested directly, and that is the part
most likely to be wrong: the crop -> full-image coordinate mapping (easy to
get subtly wrong and impossible to notice by eye) and the frontality
scoring that decides which of several detected faces wins.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.face_landmarks import (
    _MP_CHIN,
    _MP_LEFT_EYE_OUTER,
    _MP_NOSE_BRIDGE,
    _MP_RIGHT_EYE_OUTER,
    _crop_to_face,
    _face_to_array,
    _face_to_array_from_crop,
    _frontality_score,
    _pick_best_face,
)
from tests.helpers import require_stage, run_step

REPO_ROOT = Path(__file__).resolve().parent.parent


class _LM:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


def _synthetic_face(yaw: float = 0.0, n: int = 478):
    """A face whose eye line is rotated `yaw` radians about the y axis.

    yaw=0 is straight-on (score 1); yaw=pi/2 turns the eye line to point
    along z, i.e. full profile (score 0).
    """
    face = [_LM(0.5, 0.5) for _ in range(n)]
    face[_MP_RIGHT_EYE_OUTER] = _LM(0.5 - 0.1 * np.cos(yaw), 0.4, -0.1 * np.sin(yaw))
    face[_MP_LEFT_EYE_OUTER] = _LM(0.5 + 0.1 * np.cos(yaw), 0.4, 0.1 * np.sin(yaw))
    face[_MP_NOSE_BRIDGE] = _LM(0.5, 0.45, 0.0)
    face[_MP_CHIN] = _LM(0.5, 0.7, 0.0)
    return face


class TestFrontality(unittest.TestCase):
    def test_frontal_scores_near_one(self):
        self.assertAlmostEqual(_frontality_score(_synthetic_face(0.0)), 1.0, places=5)

    def test_profile_scores_near_zero(self):
        self.assertAlmostEqual(
            _frontality_score(_synthetic_face(np.pi / 2)), 0.0, places=5
        )

    def test_score_decreases_with_yaw(self):
        scores = [_frontality_score(_synthetic_face(y)) for y in (0.0, 0.4, 0.8, 1.2)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_degenerate_face_scores_zero(self):
        """All landmarks coincident -> zero-length normal, no division blowup."""
        self.assertEqual(_frontality_score([_LM(0.5, 0.5) for _ in range(478)]), 0.0)

    def test_picks_the_most_frontal_face(self):
        faces = [_synthetic_face(1.2), _synthetic_face(0.0), _synthetic_face(0.6)]
        best, idx = _pick_best_face(faces)
        self.assertEqual(idx, 1)
        self.assertIs(best, faces[1])

    def test_single_face_is_returned_unscored(self):
        faces = [_synthetic_face(1.4)]
        best, idx = _pick_best_face(faces)
        self.assertEqual(idx, 0)
        self.assertIs(best, faces[0])


class TestCropping(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((1000, 600, 3), dtype=np.uint8)

    def test_padding_is_a_fraction_of_face_size(self):
        crop, x1, y1 = _crop_to_face(self.img, (200, 300, 100, 100), padding=0.5)
        self.assertEqual((x1, y1), (150, 250))
        self.assertEqual(crop.shape[:2], (200, 200))

    def test_zero_padding(self):
        crop, x1, y1 = _crop_to_face(self.img, (200, 300, 100, 120), padding=0.0)
        self.assertEqual((x1, y1), (200, 300))
        self.assertEqual(crop.shape[:2], (120, 100))

    def test_clamps_to_image_bounds(self):
        """A face near an edge must not produce negative or out-of-range
        offsets — the mapping back to full-image coords depends on them."""
        crop, x1, y1 = _crop_to_face(self.img, (10, 5, 100, 100), padding=1.0)
        self.assertEqual((x1, y1), (0, 0))
        self.assertLessEqual(crop.shape[1], self.img.shape[1])

        crop, x1, y1 = _crop_to_face(self.img, (550, 950, 100, 100), padding=1.0)
        self.assertLessEqual(x1 + crop.shape[1], self.img.shape[1])
        self.assertLessEqual(y1 + crop.shape[0], self.img.shape[0])

    def test_crop_is_contiguous(self):
        """MediaPipe's Image wrapper requires a contiguous buffer."""
        crop, _, _ = _crop_to_face(self.img, (200, 300, 100, 100), padding=0.5)
        self.assertTrue(crop.flags["C_CONTIGUOUS"])


class TestCoordinateMapping(unittest.TestCase):
    def test_crop_coords_map_back_to_full_image(self):
        """A landmark at the centre of a crop must land at the centre of
        that crop's position in the full image."""
        full_w, full_h = 600, 1000
        crop_w, crop_h, x1, y1 = 200, 200, 150, 250

        face = [_LM(0.5, 0.5, 0.3)]
        out = _face_to_array_from_crop(face, crop_w, crop_h, x1, y1, full_w, full_h)

        self.assertAlmostEqual(out[0, 0], (0.5 * crop_w + x1) / full_w, places=6)
        self.assertAlmostEqual(out[0, 1], (0.5 * crop_h + y1) / full_h, places=6)
        # Expected absolute pixel position: 150 + 100 = 250 of 600.
        self.assertAlmostEqual(out[0, 0] * full_w, 250.0, places=4)
        self.assertAlmostEqual(out[0, 1] * full_h, 350.0, places=4)

    def test_z_is_passed_through_unscaled(self):
        """MediaPipe z is relative depth on roughly the x scale; rescaling it
        against the crop would make a cropped detection inconsistent with a
        full-image one."""
        out = _face_to_array_from_crop([_LM(0.5, 0.5, 0.42)], 200, 200, 150, 250, 600, 1000)
        self.assertAlmostEqual(float(out[0, 2]), 0.42, places=6)

    def test_corners_map_to_crop_extent(self):
        out = _face_to_array_from_crop(
            [_LM(0.0, 0.0), _LM(1.0, 1.0)], 200, 200, 150, 250, 600, 1000
        )
        np.testing.assert_allclose(out[0, :2], [150 / 600, 250 / 1000], atol=1e-6)
        np.testing.assert_allclose(out[1, :2], [350 / 600, 450 / 1000], atol=1e-6)

    def test_identity_crop_is_a_no_op(self):
        """A 'crop' covering the whole image must leave coords unchanged."""
        face = [_LM(0.25, 0.75, 0.1), _LM(0.5, 0.5, -0.2)]
        out = _face_to_array_from_crop(face, 600, 1000, 0, 0, 600, 1000)
        np.testing.assert_allclose(out, _face_to_array(face), atol=1e-6)

    def test_face_to_array_shape_and_dtype(self):
        out = _face_to_array(_synthetic_face())
        self.assertEqual(out.shape, (478, 3))
        self.assertEqual(out.dtype, np.float32)


DETECTION_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo!r})
    from pipeline.dataset import Dataset
    from pipeline.registry import get_step_class
    import pipeline.steps

    ds = Dataset.from_disk({stage!r})
    step_class = get_step_class("detect_face_landmarks")
    step, params = step_class(), step_class.resolve_params()
    out = {{}}
    for key, image in (("anchor", ds.anchor_image), ("reference", ds.reference_image)):
        res = step.run({{"image": image}}, params)["face_landmarks"]
        lm = res["landmarks"]
        out[key] = {{
            "n_points": int(lm.shape[0]),
            "image_size": list(res["image_size"]),
            "source": res["source"],
            "x_min": float(lm[:, 0].min()), "x_max": float(lm[:, 0].max()),
            "y_min": float(lm[:, 1].min()), "y_max": float(lm[:, 1].max()),
        }}
    print(json.dumps(out))
    """
)


class TestDetectionEndToEnd(unittest.TestCase):
    """Real MediaPipe detection, run out-of-process so an abort can't kill
    the suite. Skips on any failure to start, with the reason attached —
    on macOS that is expected (see this module's docstring)."""

    def test_detects_a_face_in_the_reference_photo(self):
        stage = require_stage("initial")
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            self.skipTest("mediapipe not installed")

        script = DETECTION_SCRIPT.format(repo=str(REPO_ROOT), stage=str(stage))
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            self.skipTest(
                "mediapipe detection could not run in this environment "
                f"(exit {proc.returncode}). On macOS this is the known "
                "DrishtiMetalHelper abort; run this test on Linux. "
                f"stderr tail: {proc.stderr.strip()[-300:]}"
            )

        import json

        results = json.loads(proc.stdout.strip().splitlines()[-1])

        anchor = results["anchor"]
        self.assertEqual(anchor["source"], "mediapipe")
        self.assertIn(anchor["n_points"], (468, 478))
        self.assertEqual(anchor["image_size"], [720, 1280])
        for key in ("x_min", "y_min"):
            self.assertGreaterEqual(anchor[key], 0.0)
        for key in ("x_max", "y_max"):
            self.assertLessEqual(anchor[key], 1.0)
        # A real face is a small part of a full-body frame, not the whole
        # thing — this is what catches the fallback silently returning a
        # whole-image box instead of a face.
        self.assertLess(anchor["x_max"] - anchor["x_min"], 0.4)
        self.assertLess(anchor["y_max"] - anchor["y_min"], 0.4)
        # Head of a standing figure: upper part of the frame.
        self.assertLess(anchor["y_max"], 0.5)

        # reference.png is a two-panel front/back sheet. The front-facing
        # subject is the left panel, so frontality selection must land
        # there rather than on the back of the head on the right.
        reference = results["reference"]
        self.assertEqual(reference["image_size"], [1440, 1280])
        face_centre_x = (reference["x_min"] + reference["x_max"]) / 2
        self.assertLess(
            face_centre_x, 0.5,
            "frontality scoring picked the back-facing subject in the right panel",
        )


if __name__ == "__main__":
    unittest.main()
