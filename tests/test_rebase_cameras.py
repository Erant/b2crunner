"""rebase_cameras — carrying the face cap across a camera refinement.

`refine_cameras` moves the frames' poses. The face splat's supporting
views were rendered in the bootstrap, out of the reference photograph and
through the anchor camera's pose, so they are the one thing in the run
that is left describing where that camera USED to be. This step moves
them.

What the tests below pin is the geometry, in the two directions it can be
got wrong:

  * the delta must be the anchor's own `P_new @ P_old^-1` and not its
    inverse — the test for that unprojects a pixel through the old pose
    and checks the moved point lands back under the same pixel of the NEW
    one, which is the sentence the whole step is a transcription of;
  * and moving the cameras must leave the renders they came with valid,
    i.e. a supporting camera and the splat move together, so every pixel
    is unchanged. That one is what makes it legitimate to publish the
    images untouched.

Plus the pass-throughs, which are ordinary wiring rather than failures: a
run with the refinement switched off, and a refinement that refused its
own solve.
"""

from __future__ import annotations

import unittest

import numpy as np
from body2colmap.camera import Camera

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401

TARGET = np.array([0.0, 0.9, 0.0])


def _camera(position, target=TARGET):
    camera = Camera(
        focal_length=(1066.17, 1066.17), image_size=(720, 1280),
        position=np.asarray(position, dtype=np.float64),
    )
    camera.look_at(target)
    return camera


def _orbit(count=12, radius=1.8036):
    return [
        _camera([radius * np.sin(a), 0.0, radius * np.cos(a)])
        for a in np.radians(np.arange(count) * 360.0 / count)
    ]


def _nudged(cameras, index, shift, degrees):
    """One frame of an orbit moved, the way a refinement moves one.

    Everything else is left exactly where it was, so a step that reads the
    wrong frame gets an identity and the assertion it fails is obvious.
    """
    angle = np.radians(degrees)
    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(angle), 0.0, np.cos(angle)]])
    out = list(cameras)
    source = cameras[index]
    out[index] = Camera(
        focal_length=(source.fx, source.fy),
        image_size=(source.width, source.height),
        principal_point=(source.cx, source.cy),
        position=np.asarray(source.position, dtype=np.float64) + np.asarray(shift),
        rotation=rotation @ np.asarray(source.rotation, dtype=np.float64),
    )
    return out


def _unproject(camera, pixel, depth):
    """A pixel of `camera` at `depth`, as a world point.

    The inverse of `Camera.project`, which is the operation the face splat
    is built out of: the photograph's pixels, pushed out along the anchor
    camera's rays to the depth the mesh says.
    """
    x = (pixel[0] - camera.cx) / camera.fx
    y = (pixel[1] - camera.cy) / camera.fy
    # project()'s OpenCV -> OpenGL flip, undone.
    point_cam = np.array([x * depth, -y * depth, -depth], dtype=np.float64)
    return (np.asarray(camera.rotation, dtype=np.float64) @ point_cam
            + np.asarray(camera.position, dtype=np.float64))


def _run(inputs, **params):
    step_class = get_step_class("rebase_cameras")
    return step_class().run(inputs, step_class.resolve_params(params))


class TestTheCorrectionCarried(unittest.TestCase):
    def test_the_face_lands_on_the_refined_anchor_s_ray(self):
        """The delta is the anchor's, in the direction that moves content.

        A face Gaussian is a pixel of the reference photograph pushed out
        along the anchor camera's ray. The refinement says that photograph
        was taken from somewhere else, so the world point it depicts is
        the same pixel of the SAME image through the NEW pose — which is
        what `P_new @ P_old^-1` does to it, and what its inverse does not.
        """
        given = _orbit()
        refined = _nudged(given, 3, shift=[0.02, 0.01, -0.015], degrees=0.6)
        pixel = (301.0, 512.0)
        world_before = _unproject(given[3], pixel, depth=1.6)

        # A camera at the world point, so the step has something to move
        # that stands in for the splat rather than for a view of it.
        carried = _run({
            "cameras": [_camera(world_before)],
            "from_cameras": given, "to_cameras": refined,
            "reference_index": 3,
        })["cameras"]

        moved = np.asarray(carried[0].position, dtype=np.float64)
        np.testing.assert_allclose(
            refined[3].project(moved[None, :])[0], pixel, atol=1e-2)

    def test_the_wrong_frame_s_correction_is_not_the_one_applied(self):
        """reference_index names the frame the views were built from.

        Only frame 3 moved here, so reading any other one carries an
        identity — which is a silent no-op in production and the reason
        the index is wired from `dataset.extras.anchor_frame_index`.
        """
        given = _orbit()
        refined = _nudged(given, 3, shift=[0.02, 0.01, -0.015], degrees=0.6)
        views = [_camera([0.3, 1.2, 0.4])]

        at_three = _run({"cameras": views, "from_cameras": given,
                         "to_cameras": refined, "reference_index": 3})
        at_four = _run({"cameras": views, "from_cameras": given,
                        "to_cameras": refined, "reference_index": 4})

        self.assertTrue(at_three["stats"]["applied"])
        self.assertGreater(at_three["stats"]["rotation_deg"], 0.5)
        self.assertLess(at_four["stats"]["rotation_deg"], 0.05)
        self.assertFalse(np.allclose(at_three["cameras"][0].position,
                                     at_four["cameras"][0].position))

    def test_the_renders_stay_valid_because_camera_and_splat_move_together(self):
        """Which is what lets the images be published untouched.

        A render is a function of where its camera sits relative to the
        splat. Move both by the same rigid transform and every pixel is
        identical, so the step touches the cameras and nothing else.
        """
        given = _orbit()
        refined = _nudged(given, 0, shift=[-0.03, 0.02, 0.01], degrees=1.1)
        # A cap of supporting views around the head, and a handful of
        # Gaussians for them to look at.
        views = [_camera([0.6 * np.sin(a), 1.4, 0.6 * np.cos(a)], TARGET)
                 for a in np.radians([-30.0, -10.0, 10.0, 30.0])]
        splat = np.array([[0.0, 1.5, 0.1], [0.05, 1.45, 0.08],
                          [-0.04, 1.55, 0.12]], dtype=np.float64)

        result = _run({"cameras": views, "from_cameras": given,
                       "to_cameras": refined, "reference_index": 0})
        rotation = (np.asarray(refined[0].rotation, dtype=np.float64)
                    @ np.asarray(given[0].rotation, dtype=np.float64).T)
        translation = (np.asarray(refined[0].position, dtype=np.float64)
                       - rotation @ np.asarray(given[0].position, dtype=np.float64))
        moved_splat = (rotation @ splat.T).T + translation

        for before, after in zip(views, result["cameras"]):
            np.testing.assert_allclose(
                after.project(moved_splat), before.project(splat), atol=1e-2)


class TestNothingToDo(unittest.TestCase):
    def test_no_refinement_wired_in_passes_the_views_through(self):
        """`refine_cameras: false` leaves the optional read unwritten, and
        the views are already correct against the poses in use."""
        views = _orbit(3)
        result = _run({"cameras": views})

        self.assertEqual(result["cameras"], views)
        self.assertFalse(result["stats"]["applied"])
        self.assertIn("no refinement", result["stats"]["reason"])

    def test_a_refusal_upstream_measures_as_a_zero_correction(self):
        """`refine_cameras` publishes the given poses as both halves when
        it refuses its own solve, so this needs no second code path.

        "Zero" to within `Camera`'s float32 storage, which is a floor of
        about 0.02 deg on any angle recovered from a rotation matrix that
        has been through it — 0.4 mm at this orbit's radius, and the
        reason the assertion is a bound rather than an equality.
        """
        given = _orbit()
        result = _run({"cameras": [_camera([0.3, 1.2, 0.4])],
                       "from_cameras": given, "to_cameras": given,
                       "reference_index": 0})

        self.assertTrue(result["stats"]["applied"])
        self.assertLess(result["stats"]["rotation_deg"], 0.05)
        self.assertAlmostEqual(result["stats"]["reference_shift"], 0.0, places=6)

    def test_a_correction_larger_than_a_refinement_makes_is_refused(self):
        """The mis-wire detector: 20 degrees is not a bundle adjustment of
        this orbit, and applying it would move the face further than
        leaving it alone would."""
        given = _orbit()
        refined = _nudged(given, 0, shift=[0.5, 0.0, 0.0], degrees=20.0)
        views = [_camera([0.3, 1.2, 0.4])]

        result = _run({"cameras": views, "from_cameras": given,
                       "to_cameras": refined, "reference_index": 0})

        self.assertEqual(result["cameras"], views)
        self.assertFalse(result["stats"]["applied"])
        self.assertIn("max_delta_deg", result["stats"]["reason"])

    def test_the_guard_can_be_switched_off(self):
        given = _orbit()
        refined = _nudged(given, 0, shift=[0.5, 0.0, 0.0], degrees=20.0)
        result = _run({"cameras": [_camera([0.3, 1.2, 0.4])],
                       "from_cameras": given, "to_cameras": refined,
                       "reference_index": 0}, max_delta_deg=0.0)

        self.assertTrue(result["stats"]["applied"])


class TestWhatIsRefused(unittest.TestCase):
    def test_two_lists_that_are_not_the_same_path_raise(self):
        with self.assertRaisesRegex(ValueError, "same frames"):
            _run({"cameras": _orbit(2), "from_cameras": _orbit(12),
                  "to_cameras": _orbit(11), "reference_index": 0})

    def test_an_index_off_the_end_of_the_path_raises(self):
        given = _orbit()
        with self.assertRaisesRegex(ValueError, "reference_index"):
            _run({"cameras": _orbit(2), "from_cameras": given,
                  "to_cameras": given, "reference_index": 99})


if __name__ == "__main__":
    unittest.main()
