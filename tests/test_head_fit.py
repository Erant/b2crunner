"""map_face_to_mesh / fit_head_to_face — the parts that run without a
renderer, a landmarker or the body model.

The fit itself needs the MHR body model and its checkpoint (gated, 2.8 GB);
what can be pinned here is the pure geometry around it — the head crop,
the landmark-to-vertex snap with visibility, the parameter parsing — and
the step's refusals, which are what a mis-wired workflow hits first.
"""

from __future__ import annotations

import unittest

import numpy as np

import pipeline.steps  # noqa: F401  — registers the steps
from pipeline.registry import get_step_class
from pipeline.steps import head_fit
from pipeline.workflow import INCOMPATIBLE_STEPS


def _skeleton_and_mesh(focal=1000.0, width=720, height=1280, depth=2.5):
    """A neck joint, five head joints above it, and a blob of vertices
    around the head plus a slab of body below the neck — camera frame."""
    joints = np.zeros((70, 3))
    joints[69] = [0.0, 0.1, depth]                     # neck
    head = np.array([0.0, -0.05, depth])
    joints[0] = head + [0.0, 0.02, -0.08]              # nose, proud of the face
    joints[1] = head + [0.03, -0.01, -0.05]
    joints[2] = head + [-0.03, -0.01, -0.05]
    joints[3] = head + [0.08, 0.0, 0.0]
    joints[4] = head + [-0.08, 0.0, 0.0]
    rng = np.random.RandomState(0)
    head_verts = head + rng.normal(scale=0.06, size=(400, 3))
    body_verts = np.array([0.0, 0.5, depth]) + rng.normal(scale=[0.2, 0.3, 0.1], size=(600, 3))
    return joints, np.concatenate([head_verts, body_verts]), focal, width, height


class TestHeadCrop(unittest.TestCase):
    def test_the_box_holds_the_head_and_not_the_body(self):
        joints, verts, focal, w, h = _skeleton_and_mesh()
        x0, y0, x1, y1 = head_fit.head_crop_box(verts, joints, focal, w, h)
        head_px = head_fit._project(verts[:400], focal, w / 2, h / 2)
        self.assertTrue((head_px[:, 0] >= x0).all() and (head_px[:, 0] <= x1).all())
        self.assertTrue((head_px[:, 1] >= y0).all() and (head_px[:, 1] <= y1).all())
        # ...and stops well above the body's centre.
        body_centre_v = focal * 0.5 / 2.5 + h / 2
        self.assertLess(y1, body_centre_v)

    def test_the_box_is_clamped_to_the_frame(self):
        joints, verts, focal, w, h = _skeleton_and_mesh()
        verts[:, 1] -= 1.0                                # push the head off the top
        joints[:, 1] -= 1.0
        x0, y0, x1, y1 = head_fit.head_crop_box(verts, joints, focal, w, h)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, w)
        self.assertLess(y0, y1)


class TestSnap(unittest.TestCase):
    def test_landmarks_take_the_nearest_visible_vertex_only(self):
        projected = np.array([[10.0, 10.0], [10.5, 10.5], [50.0, 50.0], [90.0, 90.0]])
        visible = np.array([True, False, True, True])
        landmarks = np.array([[10.4, 10.4],    # nearest is the hidden vertex 1 -> takes 0
                              [50.5, 50.0],    # vertex 2
                              [70.0, 70.0]])   # nothing within reach
        idx, dist = head_fit.snap_landmarks_to_vertices(landmarks, projected, visible, snap_px=2.0)
        self.assertEqual(idx.tolist(), [0, 2, -1])
        self.assertTrue(np.isinf(dist[2]))
        self.assertLess(dist[0], 1.0)

    def test_no_visible_vertices_maps_nothing(self):
        idx, dist = head_fit.snap_landmarks_to_vertices(
            np.zeros((3, 2)), np.zeros((5, 2)), np.zeros(5, bool), snap_px=2.0)
        self.assertEqual(idx.tolist(), [-1, -1, -1])
        self.assertTrue(np.isinf(dist).all())


class TestFitStepContract(unittest.TestCase):
    def test_pose_indices_are_parsed_and_validated(self):
        step = get_step_class("fit_head_to_face")
        self.assertEqual(step._parse_indices("18, 19,20"), [18, 19, 20])
        self.assertEqual(step._parse_indices("23,18"), [18, 23])
        with self.assertRaises(ValueError):
            step._parse_indices("")
        with self.assertRaises(ValueError):
            step._parse_indices("18,neck")

    def test_it_refuses_a_mesh_without_pose_params(self):
        step = get_step_class("fit_head_to_face")()
        params = get_step_class("fit_head_to_face").resolve_params({})
        with self.assertRaises(KeyError) as caught:
            step.run({"mesh_output": {"vertices": np.zeros((3, 3))},
                      "face_landmarks": {}, "face_correspondence": {},
                      "image": np.zeros((8, 8, 3), np.uint8)}, params)
        self.assertIn("pose_params", str(caught.exception))

    def test_the_face_oval_is_mediapipes_contour(self):
        self.assertEqual(len(head_fit.FACE_OVAL), 36)
        self.assertEqual(len(set(head_fit.FACE_OVAL)), 36)
        self.assertTrue(all(0 <= i < 468 for i in head_fit.FACE_OVAL))

    def test_it_cannot_be_combined_with_head_angle_fix(self):
        self.assertIn(frozenset({"head_angle_fix", "fit_head_to_face"}), INCOMPATIBLE_STEPS)

    def test_both_steps_are_registered_with_defaults_that_resolve(self):
        for name in ("map_face_to_mesh", "fit_head_to_face"):
            cls = get_step_class(name)
            params = cls.resolve_params({})
            self.assertTrue(params)
        fit = get_step_class("fit_head_to_face").resolve_params({})
        self.assertEqual(fit["pose_indices"], "18,19,20,21,22,23")
        self.assertEqual(fit["head_scale_index"], 4)
        self.assertEqual(fit["head_shape_from"], 20)


if __name__ == "__main__":
    unittest.main()
