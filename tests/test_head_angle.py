"""head_angle_fix — the craned-head correction the native path runs before
render_initial_views.

No recorded golden: this is a synthetic-skeleton test. What matters is that
the nod is (a) applied about the right axis and pivot, (b) graded so nothing
at or below the shoulders moves, and (c) sized correctly in both modes.
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


def _forward_lean(j: np.ndarray) -> float:
    up = j[69] - 0.5 * (j[9] + j[10])
    up = up / np.linalg.norm(up)
    head_vec = j[[0, 1, 2, 3, 4]].mean(axis=0) - j[69]
    fwd = np.cross(j[5] - j[6], up)
    fwd = fwd / np.linalg.norm(fwd)
    return float(np.degrees(np.arctan2(abs(head_vec @ fwd), head_vec @ up)))


class TestHeadAngleFix(unittest.TestCase):
    def test_auto_straightens_to_the_target_lean(self):
        j = _mhr70(38.0)
        verts = j[[0, 1, 2, 3, 4]].mean(axis=0) + np.random.default_rng(0).normal(
            0, 0.02, (300, 3)
        ).astype(np.float32)
        out = run_step(
            "head_angle_fix",
            {"vertices": verts, "keypoints_3d": j},
            {"mode": "auto", "target_lean_deg": 10.0},
        )
        self.assertAlmostEqual(_forward_lean(out["keypoints_3d"]), 10.0, delta=1.0)

    def test_auto_leaves_an_already_upright_head_alone(self):
        j = _mhr70(8.0)
        verts = j[[0]].repeat(50, axis=0)
        out = run_step(
            "head_angle_fix",
            {"vertices": verts, "keypoints_3d": j},
            {"mode": "auto", "target_lean_deg": 10.0},
        )
        np.testing.assert_array_equal(out["keypoints_3d"], j)
        np.testing.assert_array_equal(out["vertices"], verts)

    def test_auto_correction_is_capped(self):
        j = _mhr70(60.0)
        before = _forward_lean(j)
        out = run_step(
            "head_angle_fix",
            {"vertices": j[[0]].copy(), "keypoints_3d": j},
            {"mode": "auto", "target_lean_deg": 5.0, "max_correction_deg": 20.0},
        )
        # Far past the target: the cap applies, so ~20 deg comes off, not
        # enough to reach the 5 deg target.
        self.assertAlmostEqual(
            _forward_lean(out["keypoints_3d"]), before - 20.0, delta=2.5
        )

    def test_fixed_mode_nods_by_exactly_pitch_deg(self):
        j = _mhr70(35.0)
        before = _forward_lean(j)
        out = run_step(
            "head_angle_fix",
            {"vertices": j[[0]].copy(), "keypoints_3d": j},
            {"mode": "fixed", "pitch_deg": 15.0},
        )
        self.assertAlmostEqual(
            _forward_lean(out["keypoints_3d"]), before - 15.0, delta=1.0
        )

    def test_nothing_at_or_below_the_shoulders_moves(self):
        j = _mhr70(40.0)
        out = run_step(
            "head_angle_fix",
            {"vertices": j.copy(), "keypoints_3d": j},
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
            {"vertices": j.astype(np.float64), "keypoints_3d": j},
            {"mode": "fixed", "pitch_deg": 10.0},
        )
        self.assertEqual(out["vertices"].dtype, np.float64)
        self.assertEqual(out["keypoints_3d"].dtype, np.float32)

    def test_degenerate_skeleton_passes_through(self):
        j = np.zeros((70, 3), dtype=np.float32)  # every joint coincident
        verts = np.ones((10, 3), dtype=np.float32)
        out = run_step("head_angle_fix", {"vertices": verts, "keypoints_3d": j})
        np.testing.assert_array_equal(out["keypoints_3d"], j)
        np.testing.assert_array_equal(out["vertices"], verts)


if __name__ == "__main__":
    unittest.main()
