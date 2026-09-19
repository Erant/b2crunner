"""`render`'s `helix_anchor: start` — the helix that begins on the
photograph's camera (helical_shell.yaml, 2026-09-19).

An anchored helical render used to have one shape: body2colmap's
`compute_helical_anchor_params` bends the helix so the frame whose
elevation matches the anchor's lands on it, mid-ramp. `start` is the other
shape — frame 0 on the anchor, exactly as the circular orbit puts it, and
the climb from there — and this pins its geometry: where frame 0 is, what
the last frame is, how far the path climbs, and that `ramp` is unchanged.

Nothing rasterizes: the recorder test_backdrop.py installs stands in for
the Renderer, as for every render test.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_skeleton_style import _RenderStepCase


def _spherical(position, target):
    """(radius, azimuth_deg, elevation_deg) of `position` about `target`,
    body2colmap's own convention."""
    from body2colmap import coordinates

    offset = np.asarray(position, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    return coordinates.cartesian_to_spherical(offset)


class TestHelixStart(_RenderStepCase):
    N = 81

    def _helix(self, **params):
        base = dict(
            pattern="helical", n_frames=self.N, n_loops=1, amplitude_deg=5.0,
            lead_in_deg=0.0, lead_out_deg=4.5, override_cam_from_mesh=True,
            helix_anchor="start", render_mode="outline+skeleton",
        )
        return self._run(**{**base, **params})

    def test_the_default_is_the_mid_ramp_solve(self):
        from pipeline.registry import get_step_class

        declared = get_step_class("render").declared_params()
        self.assertEqual(declared["helix_anchor"].default, "ramp")
        self.assertEqual(set(declared["helix_anchor"].choices), {"ramp", "start"})

    def test_frame_zero_is_the_photographs_camera(self):
        """The origin: SAM-3D-Body's camera, which is where the circular
        orbit's anchor frame sits too — so the warp, the injection and every
        anchor extra downstream are the circle's."""
        result = self._helix()
        self.assertEqual(result["anchor_frame_index"], 0)
        np.testing.assert_allclose(result["cameras"][0].position, np.zeros(3), atol=1e-5)
        np.testing.assert_allclose(result["anchor_position"], np.zeros(3), atol=1e-5)
        self.assertIs(result["image_warp"]["camera"], result["cameras"][0])

    def test_the_climb_is_twice_the_amplitude_and_the_loop_closes(self):
        result = self._helix()
        target = result["orbit_target"]
        r0, az0, el0 = _spherical(result["cameras"][0].position, target)
        r1, az1, el1 = _spherical(result["cameras"][-1].position, target)
        self.assertAlmostEqual(r0, r1, places=5)
        # lead_out 4.5 = 360 / 80 puts frame 80 on frame 0's azimuth.
        self.assertAlmostEqual((az1 - az0 + 180.0) % 360.0 - 180.0, 0.0, places=3)
        self.assertAlmostEqual(el1 - el0, 10.0, places=3)
        # And it is a climb, monotone frame to frame.
        elevations = [_spherical(c.position, target)[2] for c in result["cameras"]]
        self.assertTrue(all(b >= a - 1e-6 for a, b in zip(elevations, elevations[1:])))

    def test_a_negative_amplitude_descends(self):
        result = self._helix(amplitude_deg=-5.0)
        target = result["orbit_target"]
        el0 = _spherical(result["cameras"][0].position, target)[2]
        el1 = _spherical(result["cameras"][-1].position, target)[2]
        np.testing.assert_allclose(result["cameras"][0].position, np.zeros(3), atol=1e-5)
        self.assertAlmostEqual(el1 - el0, -10.0, places=3)

    def test_a_zero_amplitude_is_refused(self):
        """That would be the circular orbit under another name."""
        with self.assertRaises(ValueError):
            self._helix(amplitude_deg=0.0)

    def test_ramp_still_lands_the_anchor_mid_sequence(self):
        """The existing shape, untouched: pass 2's helix in helical.yaml
        bends onto the anchor mid-ramp and starts on the far side."""
        result = self._run(
            pattern="helical", n_frames=self.N, n_loops=1, amplitude_deg=30.0,
            override_cam_from_mesh=True, render_mode="outline+skeleton",
        )
        index = result["anchor_frame_index"]
        self.assertGreater(index, 0)
        np.testing.assert_allclose(result["cameras"][index].position, np.zeros(3), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
