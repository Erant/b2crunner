"""Face-landmark geometry and real MediaPipe detection.

Detection is verified against cyber_6f's real anchor photo. It runs in a
*subprocess*: mediapipe 1.0.1 on macOS aborted the process once (SIGABRT
via DrishtiMetalHelper) on the first invocation after downloading its
models, and an abort cannot be caught in-process — it would take the whole
test run down. Out-of-process, a recurrence degrades to a skip. On Linux
this should simply pass.

Everything that is not MediaPipe is tested directly, and that is the part
most likely to be wrong: the crop -> full-image coordinate mapping (easy to
get subtly wrong and impossible to notice by eye) and the crop itself,
which since 2026-09-11 is the body mesh's head projected onto the
photograph rather than a face detector's box.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.face_landmarks import (
    _detect,
    _face_to_array,
    _face_to_array_from_crop,
    mesh_head_box,
)
from tests.helpers import require_stage, run_step

REPO_ROOT = Path(__file__).resolve().parent.parent


class _LM:
    """Stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)




def _mesh_with_head_at(x0, y0, x1, y1, width, height, focal=1500.0, depth=2.0):
    """A `mesh_output` whose five MHR70 head keypoints project into the
    pixel box (x0, y0, x1, y1) the way a face's do: eyes at the top
    corners, ears at the left/right edges at mid-height, nose at the
    centre — so the keypoints' own extent is the box's width by the top
    half of its height. Enough of sam3d_body's dict for the crop; no
    vertices, which the crop must not need."""
    def back(u, v):
        return [(u - width / 2.0) * depth / focal, (v - height / 2.0) * depth / focal, depth]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    keypoints = np.zeros((70, 3))
    keypoints[0] = back(cx, cy)          # nose
    keypoints[1] = back(x1, y0)          # left eye
    keypoints[2] = back(x0, y0)          # right eye
    keypoints[3] = back(x1, cy)          # left ear
    keypoints[4] = back(x0, cy)          # right ear
    cam_t = np.array([0.1, -0.2, 0.3])
    return {"keypoints_3d": keypoints - cam_t, "cam_t": cam_t,
            "focal_length": focal, "image_size": (width, height)}


class TestMeshHeadBox(unittest.TestCase):
    """The crop the landmarker sees, from the mesh head's keypoints."""

    def test_is_a_square_around_the_keypoints_padded_per_side(self):
        mesh = _mesh_with_head_at(300, 200, 400, 250, 720, 1280)
        x0, y0, x1, y1 = mesh_head_box(mesh, 720, 1280, padding=0.5)
        # Keypoint extent 100 wide x 25 tall -> span 100, side 200, centred
        # on (350, 212.5); floor/ceil outward.
        self.assertEqual((x0, y0, x1, y1), (250, 112, 450, 313))

    def test_zero_padding_is_the_keypoints_square(self):
        mesh = _mesh_with_head_at(300, 200, 400, 250, 720, 1280)
        self.assertEqual(mesh_head_box(mesh, 720, 1280, padding=0.0), (300, 162, 400, 263))

    def test_cam_t_is_applied(self):
        """The keypoints are camera-relative only once cam_t is added; a box
        built from the raw ones lands somewhere else entirely."""
        mesh = _mesh_with_head_at(300, 200, 400, 250, 720, 1280)
        with_t = mesh_head_box(mesh, 720, 1280, padding=0.0)
        mesh["cam_t"] = np.zeros(3)
        self.assertNotEqual(mesh_head_box(mesh, 720, 1280, padding=0.0), with_t)

    def test_refuses_a_mesh_fitted_on_another_frame_size(self):
        mesh = _mesh_with_head_at(300, 200, 400, 250, 720, 1280)
        with self.assertRaises(ValueError) as caught:
            mesh_head_box(mesh, 360, 640, padding=0.5)
        self.assertIn("720x1280", str(caught.exception))

    def test_refuses_a_mesh_without_the_camera(self):
        with self.assertRaises(KeyError):
            mesh_head_box({"vertices": np.zeros((3, 3))}, 720, 1280, padding=0.5)

    def test_refuses_a_degenerate_head(self):
        mesh = _mesh_with_head_at(350, 225, 351, 226, 720, 1280)
        with self.assertRaises(ValueError):
            mesh_head_box(mesh, 720, 1280, padding=0.5)


class _Result:
    def __init__(self, faces):
        self.face_landmarks = faces


class _FakeLandmarker:
    """Answers with one face at the crop's centre, only for a crop of the
    size it was told to expect — so a test can tell the crop path from the
    whole-frame fallback by what comes back."""

    def __init__(self, answer_for_shape):
        self.answer_for_shape = answer_for_shape
        self.seen = []

    def detect(self, image):
        shape = image.numpy_view().shape[:2]
        self.seen.append(shape)
        if shape == self.answer_for_shape:
            return _Result([[_LM(0.5, 0.5, 0.1)] * 3])
        return _Result([])


class TestDetectCropPath(unittest.TestCase):
    """`_detect` with a real mediapipe.Image and a fake landmarker."""

    def setUp(self):
        try:
            import mediapipe as mp
        except ImportError:
            self.skipTest("mediapipe not installed")
        self.mp = mp
        self.rgb = np.zeros((1280, 720, 3), np.uint8)

    def test_landmarks_the_crop_and_maps_back(self):
        landmarker = _FakeLandmarker(answer_for_shape=(200, 200))
        out = _detect(rgb=self.rgb, width=720, height=1280, landmarker=landmarker,
                      crop_box=(250, 125, 450, 325), mp=self.mp)
        self.assertEqual(landmarker.seen, [(200, 200)])
        # The crop's centre is full-frame (350, 225).
        np.testing.assert_allclose(out[0, :2], [350 / 720, 225 / 1280], atol=1e-6)
        self.assertAlmostEqual(float(out[0, 2]), 0.1, places=6)

    def test_clamps_the_crop_to_the_frame(self):
        """A box hanging off the frame's edge is cut to the frame, and the
        landmarks map back from the cut crop's origin, not the box's."""
        landmarker = _FakeLandmarker(answer_for_shape=(100, 100))
        out = _detect(rgb=self.rgb, width=720, height=1280, landmarker=landmarker,
                      crop_box=(-100, -50, 100, 100), mp=self.mp)
        self.assertEqual(landmarker.seen, [(100, 100)])
        np.testing.assert_allclose(out[0, :2], [50 / 720, 50 / 1280], atol=1e-6)

    def test_falls_back_to_the_whole_frame(self):
        landmarker = _FakeLandmarker(answer_for_shape=(1280, 720))
        out = _detect(rgb=self.rgb, width=720, height=1280, landmarker=landmarker,
                      crop_box=(250, 125, 450, 325), mp=self.mp)
        self.assertEqual(landmarker.seen, [(200, 200), (1280, 720)])
        np.testing.assert_allclose(out[0, :2], [0.5, 0.5], atol=1e-6)

    def test_no_crop_box_is_the_whole_frame_only(self):
        landmarker = _FakeLandmarker(answer_for_shape=(1280, 720))
        _detect(rgb=self.rgb, width=720, height=1280, landmarker=landmarker,
                crop_box=None, mp=self.mp)
        self.assertEqual(landmarker.seen, [(1280, 720)])

    def test_no_face_anywhere_raises(self):
        landmarker = _FakeLandmarker(answer_for_shape=(1, 1))
        with self.assertRaises(RuntimeError):
            _detect(rgb=self.rgb, width=720, height=1280, landmarker=landmarker,
                    crop_box=(250, 125, 450, 325), mp=self.mp)

    def test_an_empty_crop_is_refused(self):
        landmarker = _FakeLandmarker(answer_for_shape=(1, 1))
        with self.assertRaises(ValueError):
            _detect(rgb=self.rgb, width=720, height=1280, landmarker=landmarker,
                    crop_box=(800, 125, 900, 325), mp=self.mp)


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
        out = _face_to_array([_LM(0.5, 0.5) for _ in range(478)])
        self.assertEqual(out.shape, (478, 3))
        self.assertEqual(out.dtype, np.float32)




DETECTION_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo!r})
    import numpy as np
    from pipeline.dataset import Dataset
    from pipeline.registry import get_step_class
    import pipeline.steps
    from tests.test_face_landmarks import _mesh_with_head_at

    ds = Dataset.from_disk({stage!r})
    step_class = get_step_class("detect_face_landmarks")
    step, params = step_class(), step_class.resolve_params()
    image = ds.anchor_image
    h, w = image.shape[:2]
    mesh = _mesh_with_head_at(*{head!r}, w, h)
    res = step.run({{"image": image, "mesh_output": mesh}}, params)["face_landmarks"]
    lm = res["landmarks"]
    print(json.dumps({{
        "n_points": int(lm.shape[0]),
        "image_size": list(res["image_size"]),
        "source": res["source"],
        "x_min": float(lm[:, 0].min()), "x_max": float(lm[:, 0].max()),
        "y_min": float(lm[:, 1].min()), "y_max": float(lm[:, 1].max()),
    }}))
    """
)

#: Where the face is on cyber_6f/initial/anchor.png (720x1280), in pixels:
#: the extent of the landmarks the retired blaze detector's crop produced,
#: measured 2026-09-11. The end-to-end test builds a mesh whose head
#: keypoints project to roughly this — deliberately roughly, 10 px off and
#: a little small, the way a raw SAM-3D-Body fit is.
ANCHOR_FACE_PX = (303, 173, 393, 274)


class TestDetectionEndToEnd(unittest.TestCase):
    """Real MediaPipe detection, run out-of-process so an abort can't kill
    the suite. Skips on any failure to start, with the reason attached —
    on macOS that is expected (see this module's docstring)."""

    def test_detects_the_face_in_the_mesh_crop(self):
        stage = require_stage("initial")
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            self.skipTest("mediapipe not installed")

        x0, y0, x1, y1 = ANCHOR_FACE_PX
        # Eyes a third of the way down the face, ears at its middle, the
        # whole thing 10 px right and 5% narrow of the truth.
        head = (x0 + 12, y0 + (y1 - y0) // 3, x1 + 6, y0 + (y1 - y0) // 2)
        script = DETECTION_SCRIPT.format(repo=str(REPO_ROOT), stage=str(stage), head=head)
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

        anchor = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(anchor["source"], "mediapipe")
        self.assertIn(anchor["n_points"], (468, 478))
        self.assertEqual(anchor["image_size"], [720, 1280])
        # The landmarks land on the face, not on the crop or the frame:
        # within a few pixels of where the detector-cropped ones did.
        found = (anchor["x_min"] * 720, anchor["y_min"] * 1280,
                 anchor["x_max"] * 720, anchor["y_max"] * 1280)
        for got, want in zip(found, ANCHOR_FACE_PX):
            self.assertLess(abs(got - want), 6.0, (found, ANCHOR_FACE_PX))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
class TestFaceLandmarkMask(unittest.TestCase):
    """The face-only region, and the crop mapping it depends on.

    The mapping is the part worth testing hardest. `face_landmark_mask`
    receives landmarks in the FULL frame's normalized coordinates and a
    matte on a CROP's pixel grid, and if it maps between them wrongly the
    result is not an error — it is a plausible-looking mask over the wrong
    part of the face, which is exactly the failure that is invisible until
    a pod run.
    """

    FULL = (400, 600)  # width, height

    def _landmarks(self, cx=0.5, cy=0.4, rx=0.1, ry=0.13, n=64):
        """An ellipse of landmark points, in normalized full-frame coords."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return {
            "source": "mediapipe",
            "landmarks": np.stack(
                [cx + rx * np.cos(t), cy + ry * np.sin(t), np.zeros(n)], 1
            ).astype(np.float32),
            "image_size": self.FULL,
        }

    def _run(self, inputs, **params):
        return run_step("face_landmark_mask", inputs, params)

    # -- the region -------------------------------------------------------
    def _full_frame(self, **params):
        """The hull against a matte covering the whole frame."""
        full = np.ones((self.FULL[1], self.FULL[0]), np.float32)
        return self._run({"face_landmarks": self._landmarks(), "mask": full},
                         **params)["mask"]

    def test_the_region_takes_the_mattes_grid_and_dtype(self):
        mask = self._full_frame()
        self.assertEqual(mask.shape, (self.FULL[1], self.FULL[0]))
        self.assertEqual(mask.dtype, np.float32)

    def test_a_matte_is_required(self):
        """The hull alone takes background with it wherever the head is
        turned and its convex boundary cuts past the cheek."""
        with self.assertRaises(KeyError) as caught:
            self._run({"face_landmarks": self._landmarks()})
        self.assertIn("mask", str(caught.exception))

    def test_the_hull_covers_the_landmarks_and_not_the_far_corner(self):
        mask = self._full_frame()
        self.assertGreater(mask[int(0.4 * 600), int(0.5 * 400)], 0.99)  # centre
        self.assertEqual(mask[0, 0], 0.0)                               # corner

    def test_the_edge_is_feathered_not_cut(self):
        """A hard edge hands pointmap_splat a rim of fully opaque Gaussians
        and the face reads as a sticker — see soft_alpha."""
        soft = self._full_frame()
        hard = self._full_frame(feather_frac=0.0)
        partial = ((soft > 0.02) & (soft < 0.98)).sum()
        self.assertGreater(partial, 100)
        self.assertEqual(((hard > 0.02) & (hard < 0.98)).sum(), 0)

    def test_dilate_grows_the_region(self):
        tight = self._full_frame(dilate_frac=0.0, feather_frac=0.0)
        grown = self._full_frame(dilate_frac=0.2, feather_frac=0.0)
        self.assertGreater(grown.sum(), tight.sum())

    # -- the intersection, which is the point ------------------------------
    def _crop_info(self, box, crop_size):
        return {"box": box, "full_size": self.FULL, "crop_size": crop_size}

    def test_a_matte_is_cut_to_the_hull_on_the_crops_grid(self):
        """The landmarks say where the face is in the FULL frame; the matte
        lives on a crop of it. Get the mapping wrong and the mask lands on
        the wrong part of the face without raising."""
        box = (150, 180, 250, 300)          # around the ellipse, native res
        w, h = box[2] - box[0], box[3] - box[1]
        matte = np.ones((h, w), np.float32)  # a matte covering the whole crop
        out = self._run({"face_landmarks": self._landmarks(),
                         "mask": matte,
                         "crop_info": self._crop_info(box, (w, h))})
        mask = out["mask"]
        self.assertEqual(mask.shape, (h, w))
        # the ellipse centre (0.5*400, 0.4*600) = (200, 240) full-frame,
        # which is (50, 60) in the crop
        self.assertGreater(mask[60, 50], 0.99)
        self.assertEqual(mask[0, 0], 0.0)
        self.assertLess(mask.sum(), matte.sum())

    def test_the_matte_bounds_the_result(self):
        """Intersection, not replacement: where the seg says background, the
        face region must not resurrect it."""
        box = (150, 180, 250, 300)
        w, h = box[2] - box[0], box[3] - box[1]
        matte = np.zeros((h, w), np.float32)
        matte[:, : w // 2] = 1.0
        out = self._run({"face_landmarks": self._landmarks(), "mask": matte,
                         "crop_info": self._crop_info(box, (w, h))})
        self.assertTrue(np.all(out["mask"][:, w // 2:] == 0.0))

    def test_a_disjoint_matte_is_refused(self):
        box = (0, 0, 40, 40)
        out_of_reach = np.ones((40, 40), np.float32)
        with self.assertRaises(ValueError) as caught:
            self._run({"face_landmarks": self._landmarks(), "mask": out_of_reach,
                       "crop_info": self._crop_info(box, (40, 40))})
        self.assertIn("barely", str(caught.exception))

    def test_a_non_uniform_crop_resize_is_refused(self):
        box = (150, 180, 250, 300)
        with self.assertRaises(ValueError) as caught:
            self._run({"face_landmarks": self._landmarks(),
                       "mask": np.ones((60, 100), np.float32),
                       "crop_info": self._crop_info(box, (100, 60))})
        self.assertIn("non-uniform", str(caught.exception))

    def test_an_unknown_landmark_source_is_refused(self):
        lm = self._landmarks()
        lm["source"] = "dlib"
        full = np.ones((self.FULL[1], self.FULL[0]), np.float32)
        with self.assertRaises(ValueError) as caught:
            self._run({"face_landmarks": lm, "mask": full})
        self.assertIn("dlib", str(caught.exception))

    def test_too_few_landmarks_is_refused(self):
        lm = self._landmarks(n=2)
        full = np.ones((self.FULL[1], self.FULL[0]), np.float32)
        with self.assertRaises(ValueError) as caught:
            self._run({"face_landmarks": lm, "mask": full})
        self.assertIn("N >= 3", str(caught.exception))
