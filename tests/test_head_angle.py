"""head_angle_fix — the craned-head correction the native path runs before
render_initial_views.

No recorded golden: this is a synthetic-skeleton test. What matters is that
the nod is (a) applied about the right axis and pivot, (b) graded so nothing
at or below the shoulders moves, (c) sized correctly in both modes, and
(d) reprojection-preserving — the head centre must come out on the pixel it
started on, which is what keeps the photo-derived face splat registered to
the mesh head it is composited onto.
"""

from __future__ import annotations

import unittest

import numpy as np

import pipeline.steps  # noqa: F401  — registers head_angle_fix
from tests.helpers import run_step


def _mhr70(lean_deg: float, neck_to_head: float = 0.22) -> np.ndarray:
    """A minimal MHR70 skeleton with the head leaned `lean_deg` forward (+Z)
    of the torso-up (+Y) axis, measured at the neck joint."""
    j = np.zeros((70, 3), dtype=np.float32)
    j[9] = [0.10, 0.00, 0.0]    # left hip
    j[10] = [-0.10, 0.00, 0.0]  # right hip
    j[5] = [0.18, 0.55, 0.0]    # left shoulder
    j[6] = [-0.18, 0.55, 0.0]   # right shoulder
    j[69] = [0.0, 0.60, 0.0]    # neck
    a = np.radians(lean_deg)
    head = j[69] + neck_to_head * np.array([0.0, np.cos(a), np.sin(a)])
    j[0] = head + [0.0, 0.00, 0.03]     # nose
    j[1] = head + [0.03, 0.02, 0.0]     # left eye
    j[2] = head + [-0.03, 0.02, 0.0]    # right eye
    j[3] = head + [0.06, 0.00, -0.02]   # left ear
    j[4] = head + [-0.06, 0.00, -0.02]  # right ear
    return j


#: SAM-3D-Body's `pred_cam_t` for these skeletons: the camera sits 2.5 m
#: back on +Z, with the subject's own +Y (torso up) pointing up the frame.
#: Any value works — the compensation is defined by the ray, not the pose —
#: but a realistic distance keeps the reported pixel numbers meaningful.
CAM_T = np.array([0.0, -0.9, 2.5], dtype=np.float32)
FOCAL = 1000.0


def _pixel(point: np.ndarray) -> np.ndarray:
    """Where a raw-space point lands, in pixels off the principal point."""
    p = np.asarray(point, dtype=np.float64) + CAM_T
    return FOCAL * p[:2] / p[2]


def _head_center(j: np.ndarray) -> np.ndarray:
    return j[[0, 1, 2, 3, 4]].mean(axis=0)


def _bare_nod(j: np.ndarray, deg: float) -> np.ndarray:
    """What the step did before the compensation: a rigid rotation of the
    head about the inter-shoulder axis through the neck, at fixed length."""
    a = np.radians(-deg)
    out = j.astype(np.float64).copy()
    rel = out[[0, 1, 2, 3, 4]] - out[69]
    rot = np.array([[1, 0, 0],
                    [0, np.cos(a), -np.sin(a)],
                    [0, np.sin(a), np.cos(a)]])
    out[[0, 1, 2, 3, 4]] = out[69] + rel @ rot.T
    return out


def _forward_offset(j: np.ndarray) -> float:
    """How far forward of the torso-up axis the head centre sits, in metres —
    the displacement the crane actually is."""
    up = j[69] - 0.5 * (j[9] + j[10])
    up = up / np.linalg.norm(up)
    fwd = np.cross(j[5] - j[6], up)
    fwd = fwd / np.linalg.norm(fwd)
    return abs(float((_head_center(j) - j[69]) @ fwd))


def _auto_correction(j: np.ndarray, target_lean_deg: float,
                     max_correction_deg: float = 35.0) -> float:
    """The nod auto mode picks, mirroring the step. Measured off the head
    CENTRE, which the synthetic skeleton's nose/eye/ear offsets put a
    degree or two shy of the angle `_mhr70` was asked for."""
    return min(max(_forward_lean(j) - target_lean_deg, 0.0), max_correction_deg)


def _expected_forward_offset(j: np.ndarray, correction_deg: float) -> float:
    """Where a nod of `correction_deg` leaves the head's forward offset.

    `_forward_lean` is atan2(forward, up) on the neck-to-head vector, so the
    forward component is |v| * sin(lean) and a nod of c takes it to
    |v| * sin(lean - c). The compensation slides the head along the camera's
    image plane only — no depth component — so it leaves this essentially
    untouched, which is why it is the quantity to assert on rather than the
    residual angle.
    """
    length = float(np.linalg.norm(_head_center(j) - j[69]))
    return length * float(np.sin(np.radians(_forward_lean(j) - correction_deg)))


def _forward_lean(j: np.ndarray) -> float:
    up = j[69] - 0.5 * (j[9] + j[10])
    up = up / np.linalg.norm(up)
    head_vec = j[[0, 1, 2, 3, 4]].mean(axis=0) - j[69]
    fwd = np.cross(j[5] - j[6], up)
    fwd = fwd / np.linalg.norm(fwd)
    return float(np.degrees(np.arctan2(abs(head_vec @ fwd), head_vec @ up)))


class TestHeadAngleFix(unittest.TestCase):
    def test_auto_straightens_to_the_target_forward_offset(self):
        """The crane is a forward DISPLACEMENT, and that is what auto mode
        removes: the head ends up `sin(target)` of a neck-length forward of
        the torso axis, exactly where the bare rotation put it.

        The residual *angle* comes out a couple of degrees wider, because the
        reprojection compensation shortens the neck under it — see
        `test_the_neck_shortens_rather_than_the_head_moving`.
        """
        j = _mhr70(38.0, neck_to_head=0.22)
        verts = _head_center(j) + np.random.default_rng(0).normal(
            0, 0.02, (300, 3)
        ).astype(np.float32)
        out = run_step(
            "head_angle_fix",
            {"vertices": verts, "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "auto", "target_lean_deg": 10.0},
        )
        self.assertAlmostEqual(
            _forward_offset(out["keypoints_3d"]),
            _expected_forward_offset(j, _auto_correction(j, 10.0)), delta=0.002,
        )
        # ...and the angle it leaves is a couple of degrees wider than the
        # target, on the shortened neck.
        self.assertTrue(10.0 < _forward_lean(out["keypoints_3d"]) < 14.0)

    def test_the_head_center_reprojects_where_it_started(self):
        """The invariant the whole compensation exists for.

        `face_pointmap_splat` puts every Gaussian on the ray through the
        pixel it came from, so the photo-derived face sits at the
        photograph's head position no matter what the mesh does. If the nod
        moves the mesh head off that pixel, the splat is composited onto a
        head that is no longer under it — and that step's depth solve, which
        compares mesh against pointmap under the face matte, starts
        comparing the wrong surfaces.
        """
        for lean in (25.0, 38.0, 55.0):
            with self.subTest(lean=lean):
                j = _mhr70(lean)
                out = run_step(
                    "head_angle_fix",
                    {"vertices": j.copy(), "keypoints_3d": j, "cam_t": CAM_T,
                     "focal_length": FOCAL},
                    {"mode": "auto", "target_lean_deg": 10.0},
                )
                moved = _pixel(_head_center(out["keypoints_3d"])) - _pixel(_head_center(j))
                self.assertLess(np.abs(moved).max(), 0.01, f"moved {moved} px")

    def test_without_the_compensation_the_head_would_have_moved(self):
        """Guards the test above against passing for the wrong reason: a nod
        that did nothing would also reproject perfectly."""
        j = _mhr70(38.0)
        out = run_step(
            "head_angle_fix",
            {"vertices": j.copy(), "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "auto", "target_lean_deg": 10.0},
        )
        # The rotation alone lifts the head by L*(cos10 - cos38) ~ 0.043 m,
        # which is what a bare _rodrigues would have left on the table.
        rotated_only = _bare_nod(j, 28.0)
        moved = _pixel(_head_center(rotated_only)) - _pixel(_head_center(j))
        self.assertGreater(np.abs(moved).max(), 10.0, f"only moved {moved} px")
        # ...and the head did genuinely go somewhere in 3D.
        self.assertGreater(
            np.abs(_head_center(out["keypoints_3d"]) - _head_center(j)).max(), 0.02
        )

    def test_the_neck_shortens_rather_than_the_head_moving(self):
        """How the compensation is expressed: graded by the same smoothstep
        as the nod, so it lands as a shorter neck and not as a head that has
        been detached and slid."""
        j = _mhr70(38.0, neck_to_head=0.22)
        out = run_step(
            "head_angle_fix",
            {"vertices": j.copy(), "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "auto", "target_lean_deg": 10.0},
        )
        nj = out["keypoints_3d"]
        before = np.linalg.norm(_head_center(j) - j[69])
        after = np.linalg.norm(_head_center(nj) - nj[69])
        # Order cos(38)/cos(10) = 0.80 — about a fifth off the neck. Not
        # exactly that: the head also ends up further from the camera, and
        # the ray it is held on rises as it recedes, which gives some of the
        # length back. Asserted as a band, since the precise figure depends
        # on where the camera is.
        self.assertTrue(0.75 < after / before < 0.90, f"neck scale {after / before}")
        # The head is still rigid: the eye-to-eye span is untouched.
        self.assertAlmostEqual(
            float(np.linalg.norm(nj[1] - nj[2])),
            float(np.linalg.norm(j[1] - j[2])), delta=1e-5,
        )

    def test_missing_cam_t_is_refused(self):
        j = _mhr70(38.0)
        with self.assertRaises(KeyError) as caught:
            run_step("head_angle_fix", {"vertices": j.copy(), "keypoints_3d": j})
        self.assertIn("cam_t", str(caught.exception))

    def test_auto_leaves_an_already_upright_head_alone(self):
        j = _mhr70(8.0)
        verts = j[[0]].repeat(50, axis=0)
        out = run_step(
            "head_angle_fix",
            {"vertices": verts, "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "auto", "target_lean_deg": 10.0},
        )
        np.testing.assert_array_equal(out["keypoints_3d"], j)
        np.testing.assert_array_equal(out["vertices"], verts)

    def test_auto_correction_is_capped(self):
        j = _mhr70(60.0)
        before = _forward_lean(j)
        out = run_step(
            "head_angle_fix",
            {"vertices": j[[0]].copy(), "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "auto", "target_lean_deg": 5.0, "max_correction_deg": 20.0},
        )
        # Far past the target: the cap applies, so ~20 deg of nod comes off,
        # not enough to reach the 5 deg target. Asserted on the forward
        # offset, not the angle — at this magnitude the compensation takes a
        # fifth off the neck, and the residual *angle* over a much shorter
        # neck is correspondingly wider (here ~48 deg, not ~38). The head is
        # nonetheless exactly as far forward as a bare 20 deg nod leaves it.
        self.assertAlmostEqual(
            _forward_offset(out["keypoints_3d"]),
            _expected_forward_offset(j, 20.0), delta=0.002,
        )
        self.assertLess(_forward_offset(out["keypoints_3d"]), _forward_offset(j))
        del before

    def test_fixed_mode_nods_by_exactly_pitch_deg(self):
        j = _mhr70(35.0)
        before = _forward_lean(j)
        out = run_step(
            "head_angle_fix",
            {"vertices": j[[0]].copy(), "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "fixed", "pitch_deg": 15.0},
        )
        self.assertAlmostEqual(
            _forward_offset(out["keypoints_3d"]),
            _expected_forward_offset(j, 15.0), delta=0.002,
        )
        self.assertLess(_forward_lean(out["keypoints_3d"]), before)

    def test_nothing_at_or_below_the_shoulders_moves(self):
        j = _mhr70(40.0)
        out = run_step(
            "head_angle_fix",
            {"vertices": j.copy(), "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "auto"},
        )
        nj = out["keypoints_3d"]
        for idx in (5, 6, 9, 10, 69):  # shoulders, hips, neck (the pivot)
            np.testing.assert_allclose(nj[idx], j[idx], atol=1e-6)
        # ...while the head joints clearly did move.
        self.assertGreater(np.abs(nj[0] - j[0]).max(), 0.02)

    def test_output_dtype_matches_input(self):
        j = _mhr70(30.0).astype(np.float32)
        out = run_step(
            "head_angle_fix",
            {"vertices": j.astype(np.float64), "keypoints_3d": j, "cam_t": CAM_T},
            {"mode": "fixed", "pitch_deg": 10.0},
        )
        self.assertEqual(out["vertices"].dtype, np.float64)
        self.assertEqual(out["keypoints_3d"].dtype, np.float32)

    def test_degenerate_skeleton_passes_through(self):
        j = np.zeros((70, 3), dtype=np.float32)  # every joint coincident
        verts = np.ones((10, 3), dtype=np.float32)
        out = run_step("head_angle_fix", {"vertices": verts, "keypoints_3d": j, "cam_t": CAM_T})
        np.testing.assert_array_equal(out["keypoints_3d"], j)
        np.testing.assert_array_equal(out["vertices"], verts)


if __name__ == "__main__":
    unittest.main()
