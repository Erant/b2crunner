"""`render`'s `outline_inactive_margin_px`: the backdrop frozen past the drawing.

The recorder below draws a 4x4 silhouette in the alpha channel and, when the
composite carries a skeleton, one stick pixel well outside it — the case that
matters, since a stick past the outline must stay reactive or the denoise is
told to keep the ink.
"""

import unittest

import numpy as np

from pipeline.registry import get_step_class
from tests.test_backdrop import _RecordingRenderer
from tests.test_skeleton_style import _RenderStepCase

STICK = (2, 2)


class _SilhouetteRenderer(_RecordingRenderer):
    def render_composite(self, **kwargs):
        self.calls.append(("render_composite", kwargs))
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        image[..., :3] = 127
        image[10:14, 10:14, 3] = 255
        image[10:14, 10:14, :3] = 102
        if "skeleton" in kwargs["modes"]:
            image[STICK][:3] = (255, 0, 0)
        return image


class TestOutlineInactiveMargin(_RenderStepCase):
    def setUp(self):
        super().setUp()
        self.recorder = _SilhouetteRenderer.install(self)

    def _run(self, **params):
        step = get_step_class("render")()
        return step.run(
            {"mesh_output": self.mesh_output},
            get_step_class("render").resolve_params(
                {"n_frames": 2, "resolution": [24, 24],
                 "render_mode": "outline+skeleton", **params}),
        )

    def test_off_by_default(self):
        declared = {p.name: p for p in get_step_class("render").PARAMS}
        self.assertIsNone(declared["outline_inactive_margin_px"].default)
        self.assertIsNone(self._run()["inactive_masks"])

    def test_the_silhouette_and_the_sticks_stay_reactive(self):
        masks = self._run(outline_inactive_margin_px=0)["inactive_masks"]
        self.assertEqual(len(masks), 2)
        expected = np.zeros((24, 24), dtype=np.float32)
        expected[10:14, 10:14] = 1.0
        expected[STICK] = 1.0
        for mask in masks:
            self.assertEqual(mask.dtype, np.float32)
            np.testing.assert_array_equal(mask, expected)

    def test_the_margin_dilates_the_reactive_region(self):
        mask = self._run(outline_inactive_margin_px=2)["inactive_masks"][0]
        self.assertEqual(mask[10, 8], 1.0)    # 2 px left of the silhouette
        self.assertEqual(mask[10, 7], 0.0)    # 3 px
        self.assertEqual(mask[4, 2], 1.0)     # 2 px below the stick
        self.assertEqual(mask[23, 23], 0.0)

    def test_without_a_skeleton_there_is_no_ink_to_keep_reactive(self):
        mask = self._run(render_mode="outline",
                         outline_inactive_margin_px=0)["inactive_masks"][0]
        self.assertEqual(mask[STICK], 0.0)
        self.assertEqual(mask[11, 11], 1.0)

    def test_the_stick_free_copy_is_still_only_published_on_request(self):
        result = self._run(outline_inactive_margin_px=0)
        self.assertIsNone(result["images_no_skeleton"])
        result = self._run(outline_inactive_margin_px=0, skeleton_free_copy=True)
        self.assertEqual(len(result["images_no_skeleton"]), 2)
        self.assertNotEqual(tuple(result["images_no_skeleton"][0][STICK]), (0, 0, 255))

    def test_a_mode_without_an_outline_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._run(render_mode="mesh+skeleton", outline_inactive_margin_px=8)
        self.assertIn("outline_inactive_margin_px", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
