"""`render`'s skeleton drawing convention, and the ablation that removes it.

Two changes share this file because they were made for one question: is the
OpenPose skeleton the shipped workflow draws visible in the denoised frames
because of what it looks like, or because it is there at all?

`skeleton_style` answers the first — `dwpose` redraws the overlay in the
convention Wan 2.2 VACE's own pose maps are drawn in, which the older
`openpose` style agreed with only across the upper body (see
body2colmap.skeleton). The `outline` render mode answers the second: the same
frame with the overlay and nothing else removed.

Nothing here rasterizes. Pyrender needs a headless-GL setup this environment
has not got, so these assert what the step ASKS the Renderer to draw, via the
recorder test_backdrop.py already installs.
"""

import unittest

import numpy as np

from pipeline.registry import get_step_class
from pipeline.steps.render import _SKELETON_RADII
from tests.test_backdrop import _RecordingRenderer


class _RenderStepCase(unittest.TestCase):
    def setUp(self):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        rng = np.random.default_rng(3)
        self.mesh_output = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
            "cam_t": np.array([0.0, 0.0, 3.0], dtype=np.float32),
            "keypoints_3d": (rng.normal(size=(70, 3)) * 0.3).astype(np.float32),
            "focal_length": 1000.0,
        }
        self.recorder = _RecordingRenderer.install(self)

    def _run(self, **params):
        step = get_step_class("render")()
        return step.run(
            {"mesh_output": self.mesh_output},
            get_step_class("render").resolve_params(
                {"n_frames": 2, "resolution": [8, 8], **params}),
        )

    def _skeleton_opts(self, **params):
        """The `skeleton` block of the first composite the step asked for."""
        self._run(**params)
        name, kwargs = self.recorder.calls[0]
        self.assertEqual(name, "render_composite")
        return kwargs["modes"].get("skeleton")


class TestSkeletonStyle(_RenderStepCase):
    def test_the_default_is_dwpose(self):
        """The convention VACE conditions on, so the overlay reads as pose
        rather than as something painted on the subject."""
        declared = get_step_class("render").declared_params()
        self.assertEqual(declared["skeleton_style"].default, "dwpose")

    def test_every_style_has_default_radii(self):
        """`joint_radius`/`bone_radius` default to None and are filled in per
        style, so a style added without a size would raise a KeyError deep in
        a render rather than here."""
        declared = get_step_class("render").declared_params()
        self.assertEqual(
            set(declared["skeleton_style"].choices), set(_SKELETON_RADII)
        )

    def test_the_style_reaches_the_renderer(self):
        opts = self._skeleton_opts(render_mode="outline+skeleton")
        self.assertEqual(opts["style"], "dwpose")

    def test_the_older_style_is_still_selectable(self):
        """It is the "before" the ablation is measured against."""
        opts = self._skeleton_opts(
            render_mode="outline+skeleton", skeleton_style="openpose"
        )
        self.assertEqual(opts["style"], "openpose")

    def test_an_unset_radius_takes_the_style_default(self):
        for style, (joint, bone) in _SKELETON_RADII.items():
            with self.subTest(style=style):
                opts = self._skeleton_opts(
                    render_mode="outline+skeleton", skeleton_style=style
                )
                self.assertEqual(opts["joint_radius"], joint)
                self.assertEqual(opts["bone_radius"], bone)

    def test_the_styles_do_not_share_a_size(self):
        """A guard on the table, not on the numbers: DWPose draws a limb at
        twice the width the old style used and a dot exactly as wide as a
        limb, so a table that collapsed them would be a mistake."""
        self.assertNotEqual(_SKELETON_RADII["dwpose"], _SKELETON_RADII["openpose"])
        joint, bone = _SKELETON_RADII["dwpose"]
        self.assertEqual(joint, bone)

    def test_an_explicit_radius_overrides_the_style(self):
        opts = self._skeleton_opts(
            render_mode="outline+skeleton", bone_radius=0.01, joint_radius=0.02
        )
        self.assertEqual(opts["bone_radius"], 0.01)
        self.assertEqual(opts["joint_radius"], 0.02)

    def test_the_single_layer_skeleton_mode_carries_the_style_too(self):
        """`skeleton` alone does not go through render_composite, so it is a
        separate call site and a separate chance to forget."""
        self._run(render_mode="skeleton")
        name, kwargs = self.recorder.calls[0]
        self.assertEqual(name, "render_skeleton")
        self.assertEqual(kwargs["style"], "dwpose")
        self.assertEqual(kwargs["bone_radius"], _SKELETON_RADII["dwpose"][1])


class TestOutlineAblation(_RenderStepCase):
    """`outline` / `outline+splat`: the control frame minus the skeleton."""

    def test_the_ablation_draws_no_skeleton(self):
        self.assertIsNone(self._skeleton_opts(render_mode="outline"))

    def test_the_ablation_changes_nothing_else_about_the_base_layer(self):
        """The point of the ablation is that the silhouette is untouched —
        same fill, same blur — so the only difference from the shipped mode
        is the overlay."""
        with_skeleton = self._run(render_mode="outline+skeleton")
        ablated = self._run(render_mode="outline")
        del with_skeleton, ablated
        outlines = [
            kwargs["modes"]["outline"] for _, kwargs in self.recorder.calls
        ]
        self.assertEqual(len(set(map(repr, outlines))), 1)

    def test_the_ablation_keeps_the_backdrop(self):
        self._run(render_mode="outline")
        self.assertIsNotNone(self.recorder.background)

    def test_the_ablation_is_not_composited_twice(self):
        """It is a single-layer mode but it still goes through
        render_composite, which draws the backdrop under its own base layer.
        The step must not composite it a second time."""
        self._run(render_mode="outline")
        self.assertEqual(self.recorder.composited, 0)

    def test_the_ablation_survives_the_splat_spelling(self):
        """`outline+splat` is the useful one: everything the shipped mode
        draws except the overlay, face splat included."""
        self._run(render_mode="outline+splat")
        _, kwargs = self.recorder.calls[0]
        self.assertNotIn("skeleton", kwargs["modes"])
        self.assertIn("outline", kwargs["modes"])
        self.assertIn("splat_layer", kwargs)

    def test_both_ablation_spellings_are_offered(self):
        choices = get_step_class("render").declared_params()["render_mode"].choices
        self.assertIn("outline", choices)
        self.assertIn("outline+splat", choices)


if __name__ == "__main__":
    unittest.main()
