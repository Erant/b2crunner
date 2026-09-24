"""`render`'s `skeleton_free_copy`: the control drawing again, without its sticks.

The Weak skeleton setting shows pass 1's denoise the skeleton on its first
step only and this copy on the rest. The comparison it was measured by —
same seed, same picture, minus the stick ink — holds only if the two
drawings differ in the skeleton and nowhere else, so what is pinned is that
the copy is the SAME composite call with the skeleton layer (and the face
overlay the skeleton draws) taken out: base, relief, backdrop and splat
layer as they were.
"""

from __future__ import annotations

import unittest

from pipeline.registry import get_step_class
from tests.test_skeleton_style import _RenderStepCase


class TestSkeletonFreeCopy(_RenderStepCase):
    def test_the_default_is_off(self):
        declared = get_step_class("render").declared_params()
        self.assertIs(declared["skeleton_free_copy"].default, False)

    def test_off_publishes_none_and_draws_nothing_extra(self):
        result = self._run(render_mode="outline+skeleton")
        self.assertIsNone(result["images_no_skeleton"])
        self.assertEqual(len(self.recorder.calls), 2)  # one composite per frame

    def test_the_key_is_published_even_when_off(self):
        """The runner refuses a mapped output a step did not return."""
        self.assertIn("images_no_skeleton", self._run(render_mode="outline"))

    def test_on_it_is_the_same_composite_without_the_skeleton(self):
        result = self._run(render_mode="outline+skeleton", skeleton_free_copy=True)
        composites = [kwargs for name, kwargs in self.recorder.calls if name == "render_composite"]
        self.assertEqual(len(composites), 4)  # two per frame
        for drawing, copy in zip(composites[0::2], composites[1::2]):
            self.assertIn("skeleton", drawing["modes"])
            self.assertEqual(
                copy["modes"],
                {k: v for k, v in drawing["modes"].items() if k not in ("skeleton", "face")},
            )
            self.assertIs(copy["camera"], drawing["camera"])
            self.assertIs(copy["splat_layer"], drawing["splat_layer"])
        self.assertEqual(len(result["images_no_skeleton"]), len(result["images"]))
        self.assertEqual(result["images_no_skeleton"][0].shape, result["images"][0].shape)

    def test_a_mode_without_a_skeleton_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._run(render_mode="outline", skeleton_free_copy=True)
        self.assertIn("skeleton_free_copy", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
