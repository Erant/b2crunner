"""`render`'s `splat_inactive_mask`: the batch that says the face is real.

A `skeleton+splat` frame is a drawing with one region of photograph
composited into it. Nothing downstream knew that: `inject_anchor`
manufactures an all-1.0 VACE batch for the renders, so `denoise_pass1` has
always been told the whole frame is synthetic and free to repaint the face
the splat exists to carry. This flag publishes the mask that says otherwise.

The mask itself is body2colmap's (`InactiveMaskOptions`, 81a0e1b) and is
tested there; what is tested here is the wiring — that the flag is off by
default, that it does not disturb the coverage `masks` beside it, and that
the polarity reaching `wan22_vace_denoise` is the one that step reads.

Nothing rasterizes: the recorder test_backdrop.py installs stands in for the
Renderer, and the splat layers are handed in directly.
"""

import unittest

import numpy as np

from pipeline.registry import get_step_class
from tests.test_skeleton_style import _RenderStepCase


class _SplatLayerCase(_RenderStepCase):
    """A render step whose splat layers are supplied rather than rasterized.

    `_resolve_splat_layers` shells out to `brush-splat-render`; the mask is
    a pure function of what comes back from it, so what comes back is what
    this patches.
    """

    #: The frame the base class renders, from its `_run` defaults.
    SIZE = (8, 8)

    def layers(self, *, alpha):
        """One 8x8 RGBA layer whose left half carries `alpha`, and one None.

        The None is a frame the `splat_max_angle_deg` cull dropped, which
        body2colmap defines as wholly reactive rather than absent.
        """
        width, height = self.SIZE
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        layer[:, : width // 2, 3] = alpha
        return [layer, None]

    def install_layers(self, layers):
        import pipeline.steps.render as render_module

        original = render_module._resolve_splat_layers
        render_module._resolve_splat_layers = lambda *a, **k: list(layers)
        self.addCleanup(
            setattr, render_module, "_resolve_splat_layers", original
        )

    def _run_masked(self, *, alpha=255, **params):
        self.install_layers(self.layers(alpha=alpha))
        return self._run(
            render_mode="outline+skeleton+splat",
            # `_resolve_splat_layers` is patched, but the step only calls it
            # when something is wired to draw, and it reads this path itself.
            splat_inactive_mask=True,
            **params,
        )


class TestTheFlagIsOff(_RenderStepCase):
    def test_the_default_is_off(self):
        """Every run before this one had no mask, and turning it on changes
        what a denoise pass is allowed to repaint. That is an experiment,
        not a fix, until a run says otherwise."""
        declared = get_step_class("render").declared_params()
        self.assertIs(declared["splat_inactive_mask"].default, False)

    def test_off_publishes_none_rather_than_a_uniform_batch(self):
        """None is what makes the workflow's optional read a no-op:
        `inject_anchor` treats it as "not given" and manufactures its own
        all-1.0 batch, exactly as when nothing was wired at all."""
        result = self._run(render_mode="outline+skeleton+splat")
        self.assertIsNone(result["inactive_masks"])

    def test_the_key_is_published_even_when_off(self):
        """The runner refuses a declared output a step did not return, so
        the workflow could not wire this at all if it came and went."""
        self.assertIn("inactive_masks", self._run(render_mode="outline"))

    def test_a_mode_without_a_splat_is_refused(self):
        """body2colmap's own rule: a uniform mask is indistinguishable from
        no mask, so the flag fails rather than answering with one."""
        with self.assertRaises(ValueError) as caught:
            self._run(render_mode="outline+skeleton", splat_inactive_mask=True)
        self.assertIn("splat_inactive_mask", str(caught.exception))


class TestTheMask(_SplatLayerCase):
    def test_the_splat_is_inactive_and_the_rest_reactive(self):
        """0.0 "a real photograph, keep it" over the splat, 1.0 "synthetic,
        denoise it" everywhere else — `wan22_vace_denoise`'s `control_masks`
        polarity, not inverted on the way."""
        mask = self._run_masked()["inactive_masks"][0]
        width = self.SIZE[0]
        np.testing.assert_array_equal(mask[:, : width // 2], 0.0)
        np.testing.assert_array_equal(mask[:, width // 2:], 1.0)

    def test_a_culled_frame_is_wholly_reactive(self):
        """Not absent. A batch with holes in it no longer lines up with the
        frames it describes."""
        mask = self._run_masked()["inactive_masks"][1]
        np.testing.assert_array_equal(mask, 1.0)

    def test_there_is_one_mask_per_frame(self):
        result = self._run_masked()
        self.assertEqual(len(result["inactive_masks"]), len(result["images"]))

    def test_the_mask_matches_the_frame_and_the_mask_convention(self):
        """float32 [0,1] at the render resolution, which is what every other
        mask this pipeline passes between steps is."""
        mask = self._run_masked()["inactive_masks"][0]
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(mask.shape, self.SIZE[::-1])

    def test_partial_coverage_stays_reactive(self):
        """The threshold is body2colmap's 0.9 and it is high on purpose: a
        partly covered pixel is a blend of the splat with the drawing under
        it, and preserving it preserves the drawing too."""
        mask = self._run_masked(alpha=200)["inactive_masks"][0]
        np.testing.assert_array_equal(mask, 1.0)

    def test_the_coverage_masks_are_left_alone(self):
        """The two batches are independent here. body2colmap puts the mask
        in the frame's alpha and loses the silhouette doing it, because a
        PNG has nowhere else to put it; this pipeline keeps images and masks
        apart and pays nothing."""
        result = self._run_masked()
        for mask in result["masks"]:
            np.testing.assert_array_equal(mask, 1.0)
        self.assertTrue((result["inactive_masks"][0] == 0.0).any())


class TestTheDownstreamAgrees(_SplatLayerCase):
    """The one seam that matters: `inject_anchor` between here and denoise."""

    class _Cam:
        def __init__(self, position):
            self.position = np.asarray(position, dtype=np.float32)

    def _inject(self, masks, **inputs):
        step = get_step_class("inject_anchor")()
        return step.run(
            {
                "images": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in masks],
                "cameras": [self._Cam([0.0, 0.0, 1.0]),
                            self._Cam([0.0, 0.0, -1.0])],
                "masks": masks,
                **inputs,
            },
            get_step_class("inject_anchor").resolve_params({}),
        )

    def test_the_two_marks_compose(self):
        """`inject_anchor` overwrites only the frame it injects into, so the
        per-pixel mark from the render survives on every other frame while
        the injected photo still gets its uniform per-frame 0.0."""
        masks = self._run_masked()["inactive_masks"]
        out = self._inject(
            masks,
            anchor_position=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            anchor_image=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        # Frame 0 is the anchor: wholly real, so wholly inactive.
        np.testing.assert_array_equal(out["masks"][0], 0.0)
        # Frame 1 is a render and keeps whatever the mask said about it.
        np.testing.assert_array_equal(out["masks"][1], masks[1])

    def test_a_supplied_batch_is_never_manufactured_over(self):
        """The general form of the bug that cost a run: a step handed
        somebody else's masks must not replace them with its own."""
        masks = self._run_masked()["inactive_masks"]
        out = self._inject(masks)
        for produced, supplied in zip(out["masks"], masks):
            np.testing.assert_array_equal(produced, supplied)


if __name__ == "__main__":
    unittest.main()
