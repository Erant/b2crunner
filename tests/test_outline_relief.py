"""`render`'s two answers to a front/back-ambiguous drawing.

helical-20260920-202010 turned the head and not the torso: at azimuth 171
the control skeleton was a back view and the denoise painted a breastplate,
because a flat silhouette plus a 2-D skeleton is the same drawing from the
front and from behind, mirrored. `outline_relief: depth` fills the
silhouette with the body model's depth at outline strength — the bit the
drawing lacked — and `skeleton_occlusion_m` stops drawing the joints the
model hides, DWPose's own rule for a keypoint its detector did not find.

body2colmap draws both (its tests cover relief_fill and _joints_visible);
what this step owns is what it ASKS for, through the recorder
test_backdrop.py installs — nothing here rasterizes.
"""

import unittest

import numpy as np

from pipeline.registry import get_step_class
from pipeline.steps.render import _outline_grey
from tests.test_skeleton_style import _RenderStepCase


def _byte(grey):
    return int(grey[0] * 255)


class TestOutlineRelief(_RenderStepCase):
    def _outline_opts(self, **params):
        self._run(render_mode="outline+skeleton", **params)
        name, kwargs = self.recorder.calls[0]
        self.assertEqual(name, "render_composite")
        return kwargs["modes"]["outline"]

    def test_off_by_default_and_the_flat_fill_is_unchanged(self):
        opts = self._outline_opts()
        self.assertNotIn("relief", opts)
        self.assertEqual(_byte(opts["fg_color"]), 0x6F)

    def test_the_relief_straddles_the_flat_fill(self):
        """Near end lighter, far end darker, by the amplitude each way on
        the strength ramp — so the mean grey is the flat fill's and an A/B
        changes only the shading. 8 bytes each way at 6.25."""
        opts = self._outline_opts(outline_relief="depth", outline_strength=20.0,
                                  outline_relief_amplitude=6.25)
        relief = opts["relief"]
        self.assertEqual(_byte(opts["fg_color"]), 0x66)
        self.assertEqual(_byte(relief["near_color"]), 0x66 + 8)
        self.assertEqual(_byte(relief["far_color"]), 0x66 - 8)
        self.assertEqual(relief["levels"], 16)
        self.assertEqual(relief["depth_range"], 0.8)
        self.assertEqual(relief["smooth"], 12.0)

    def test_the_near_end_clips_at_the_ground(self):
        opts = self._outline_opts(outline_relief="depth", outline_strength=6.25,
                                  outline_relief_amplitude=6.25)
        self.assertEqual(_byte(opts["relief"]["near_color"]), 0x7F)
        self.assertEqual(_byte(opts["relief"]["near_color"]), _byte(opts["bg_color"]))

    def test_the_window_is_centred_on_the_orbit_target(self):
        """One centre for every frame, so a surface keeps its grey around
        the orbit; the step publishes the same point as orbit_target."""
        result = self._run(render_mode="outline+skeleton", outline_relief="depth")
        relief = self.recorder.calls[0][1]["modes"]["outline"]["relief"]
        self.assertEqual(relief["center"].shape, (3,))
        np.testing.assert_allclose(relief["center"],
                                   result["orbit_target"], atol=1e-6)
        # And the same object reaches every frame.
        for name, kwargs in self.recorder.calls:
            self.assertIs(kwargs["modes"]["outline"]["relief"], relief)

    def test_the_knobs_reach_the_renderer(self):
        opts = self._outline_opts(outline_relief="depth", outline_relief_levels=4,
                                  outline_relief_depth_m=0.5,
                                  outline_relief_smooth=0.0)
        self.assertEqual(opts["relief"]["levels"], 4)
        self.assertEqual(opts["relief"]["depth_range"], 0.5)
        self.assertEqual(opts["relief"]["smooth"], 0.0)

    def test_the_ablation_mode_takes_it_too(self):
        self._run(render_mode="outline", outline_relief="depth")
        self.assertIn("relief", self.recorder.calls[0][1]["modes"]["outline"])

    def test_only_the_two_fills_are_offered(self):
        param = next(p for p in get_step_class("render").PARAMS
                     if p.name == "outline_relief")
        self.assertEqual(param.choices, ("none", "depth"))


class TestSkeletonOcclusion(_RenderStepCase):
    def test_off_by_default(self):
        self.assertIsNone(self._skeleton_opts()["occlusion_tolerance"])

    def test_the_tolerance_reaches_the_overlay(self):
        self.assertEqual(self._skeleton_opts(skeleton_occlusion_m=0.12)["occlusion_tolerance"], 0.12)

    def test_zero_is_off_not_a_zero_tolerance(self):
        """A tolerance of 0 would hide every joint inside its own limb."""
        self.assertIsNone(self._skeleton_opts(skeleton_occlusion_m=0.0)["occlusion_tolerance"])

    def test_the_single_layer_skeleton_mode_carries_it_too(self):
        self._run(render_mode="skeleton", skeleton_occlusion_m=0.12)
        name, kwargs = self.recorder.calls[0]
        self.assertEqual(name, "render_skeleton")
        self.assertEqual(kwargs["occlusion_tolerance"], 0.12)


if __name__ == "__main__":
    unittest.main()
