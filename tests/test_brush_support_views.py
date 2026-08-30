"""Supporting views: the masked half of a mixed brush training run.

A supporting view is one the training should fit where its mask says to
and *ignore* everywhere else — the confidence-gated splat re-renders,
whose background is the cull colour rather than emptiness. That is brush's
`masked` alpha mode, and the rendered training views are its `transparent`
one; brush decides which is which per view from the export's layout (a
`masks/` sidecar means masked), so the whole contract of this step is what
it writes on disk and the one flag it must NOT pass. See brush's
docs/mixed-alpha-modes.md and steps/brush.py's module docstring.

These check the argv and the export `run()` builds, with the training
itself stubbed out — the export lives in a TemporaryDirectory that `run()`
deletes on the way out, so the stand-in for brush snapshots its layout
while it exists.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401


def _cameras(count: int, start: int = 0):
    from body2colmap.camera import Camera

    return [
        Camera(
            focal_length=(8.0, 8.0),
            image_size=(8, 8),
            principal_point=(4.0, 4.0),
            position=np.array([0.0, 0.0, float(start + i + 1)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(count)
    ]


def _inputs(**extra):
    """Two 8x8 training views — the smallest batch ColmapExporter accepts."""
    return {
        "cameras": _cameras(2),
        "image_names": ["frame_00001_.png", "frame_00002_.png"],
        "points_3d": (
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.uint8),
        ),
        "images": [np.full((8, 8, 3), 40, dtype=np.uint8) for _ in range(2)],
        "masks": [np.ones((8, 8), dtype=np.float32) for _ in range(2)],
        **extra,
    }


def _support(count: int = 1, **extra):
    """One supporting view: a grey-backgrounded render plus its gate."""
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    return {
        "support_cameras": _cameras(count, start=10),
        "support_images": [np.full((8, 8, 3), 128, dtype=np.uint8) for _ in range(count)],
        "support_masks": [mask.copy() for _ in range(count)],
        **extra,
    }


class _Run:
    """One stubbed `run()`: the argv brush would have been given, and a
    snapshot of the COLMAP export it would have trained on."""

    def __init__(self, inputs, **overrides):
        step_class = get_step_class("brush")
        step = step_class()
        self.argv = None
        self.files = {}

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            self.argv = list(cmd)
            if colmap_dir is not None:
                for path in sorted(Path(colmap_dir).rglob("*")):
                    if path.is_file():
                        self.files[str(path.relative_to(colmap_dir))] = path.read_bytes()
                self.colmap_dir = Path(colmap_dir)
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"export_dir": tmp, **overrides})
            step.run(inputs, params)

    def image(self, name: str) -> np.ndarray:
        return cv2.imdecode(
            np.frombuffer(self.files[name], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )

    @property
    def images_txt(self) -> str:
        return self.files["images.txt"].decode()


class TestNoAlphaModeIsForced(unittest.TestCase):
    """--alpha-mode is a global force: passing it is what *prevents* a mix."""

    def test_the_flag_is_not_passed_by_default(self):
        self.assertNotIn("--alpha-mode", _Run(_inputs()).argv)

    def test_the_default_forces_nothing(self):
        self.assertEqual(
            get_step_class("brush").declared_params()["alpha_mode"].default, "auto"
        )

    def test_training_views_still_carry_their_alpha_and_no_sidecar(self):
        """Which is the transparent case — byte-identical to what the
        forced `--alpha-mode transparent` used to train on."""
        run = _Run(_inputs())
        frame = run.image("images/frame_00001_.png")
        self.assertEqual(frame.shape[-1], 4)
        self.assertTrue((frame[..., 3] == 255).all())
        self.assertFalse(any(name.startswith("masks/") for name in run.files))

    def test_an_explicit_mode_is_still_honoured(self):
        argv = _Run(_inputs(), alpha_mode="masked").argv
        self.assertEqual(argv[argv.index("--alpha-mode") + 1], "masked")

    def test_the_declared_choices_are_the_ones_brush_accepts(self):
        """`ignore` was declared for years and would have had clap reject
        the invocation; brush's enum is masked | transparent."""
        self.assertEqual(
            get_step_class("brush").declared_params()["alpha_mode"].choices,
            ("auto", "transparent", "masked"),
        )


class TestSupportViewsAreWrittenMasked(unittest.TestCase):
    def setUp(self):
        self.run = _Run(_inputs(**_support()))

    def test_the_frame_is_rgb_not_rgba(self):
        """An RGBA frame with no sidecar reads as transparency and is
        premultiplied at load, which destroys the RGB under the mask."""
        frame = self.run.image("images/support_00001.png")
        self.assertEqual(frame.shape[-1], 3)
        self.assertTrue((frame == 128).all())

    def test_the_mask_rides_in_a_sidecar(self):
        mask = self.run.image("masks/support_00001.png")
        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[4, 4], 255)

    def test_the_training_views_get_no_sidecar(self):
        self.assertEqual(
            sorted(n for n in self.run.files if n.startswith("masks/")),
            ["masks/support_00001.png"],
        )

    def test_both_kinds_are_in_one_colmap_model(self):
        text = self.run.images_txt
        self.assertIn("frame_00001_.png", text)
        self.assertIn("support_00001.png", text)

    def test_supplied_names_are_used(self):
        run = _Run(_inputs(**_support(support_image_names=["gated_00038_.png"])))
        self.assertIn("images/gated_00038_.png", run.files)
        self.assertIn("masks/gated_00038_.png", run.files)


class TestNormalizeMaskedLoss(unittest.TestCase):
    """The weighting only becomes a bias once both modes are in one run."""

    def test_auto_is_off_for_an_all_transparent_run(self):
        self.assertNotIn("--normalize-masked-loss", _Run(_inputs()).argv)

    def test_auto_is_on_for_a_mixed_run(self):
        self.assertIn("--normalize-masked-loss", _Run(_inputs(**_support())).argv)

    def test_auto_is_off_again_when_the_mix_is_flattened(self):
        """--alpha-mode forces every view to one mode, so there is no mix
        left to normalise — an argv nothing reads is how a run gets tuned
        on a flag that was never doing anything."""
        argv = _Run(_inputs(**_support()), alpha_mode="transparent").argv
        self.assertNotIn("--normalize-masked-loss", argv)

    def test_it_can_be_forced_on(self):
        self.assertIn(
            "--normalize-masked-loss", _Run(_inputs(), normalize_masked_loss="on").argv
        )

    def test_it_can_be_forced_off(self):
        argv = _Run(_inputs(**_support()), normalize_masked_loss="off").argv
        self.assertNotIn("--normalize-masked-loss", argv)

    def test_a_bad_setting_is_refused(self):
        with self.assertRaises(ValueError):
            _Run(_inputs(), normalize_masked_loss="yes")


class TestARetiredAlphaModeIsRefused(unittest.TestCase):
    def test_ignore_is_not_a_brush_mode(self):
        """It was this param's declared choice for a long time; clap would
        have rejected the invocation, ~forty minutes into nothing."""
        with self.assertRaises(ValueError) as caught:
            _Run(_inputs(), alpha_mode="ignore")
        self.assertIn("masked", str(caught.exception))


class TestSupportNormalMaps(unittest.TestCase):
    def test_they_land_beside_the_training_ones(self):
        normals = [np.zeros((8, 8, 3), dtype=np.float32) for _ in range(2)]
        run = _Run(_inputs(
            normal_maps=normals,
            **_support(support_normal_maps=[np.zeros((8, 8, 3), dtype=np.float32)]),
        ))
        self.assertIn("normals/frame_00001_.png", run.files)
        self.assertIn("normals/support_00001.png", run.files)
        self.assertEqual(run.image("normals/support_00001.png").shape[-1], 4)

    def test_a_count_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            _Run(_inputs(**_support(
                support_normal_maps=[np.zeros((8, 8, 3), dtype=np.float32)] * 2
            )))


class TestSupportViewValidation(unittest.TestCase):
    def test_a_view_without_a_mask_is_refused(self):
        """It would be a full-weight view fitting the cull colour as if it
        were the subject."""
        inputs = _inputs(**_support())
        del inputs["support_masks"]
        with self.assertRaises(ValueError) as caught:
            _Run(inputs)
        self.assertIn("support_masks", str(caught.exception))

    def test_a_view_without_a_camera_is_refused(self):
        inputs = _inputs(**_support())
        del inputs["support_cameras"]
        with self.assertRaises(ValueError) as caught:
            _Run(inputs)
        self.assertIn("support_cameras", str(caught.exception))

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            _Run(_inputs(**_support(support_cameras=_cameras(2, start=10))))

    def test_a_name_colliding_with_a_training_view_is_refused(self):
        """brush matches a sidecar by stem as well as by full name, so
        `masks/frame_00001_.png` would attach to the training frame too and
        silently stop it carving the silhouette."""
        with self.assertRaises(ValueError) as caught:
            _Run(_inputs(**_support(support_image_names=["frame_00001_.jpg"])))
        self.assertIn("frame_00001_", str(caught.exception))

    def test_no_support_views_is_the_untouched_case(self):
        """Nothing wired means the export this step always built."""
        run = _Run(_inputs())
        self.assertEqual(
            sorted(run.files),
            ["cameras.txt", "images.txt", "images/frame_00001_.png",
             "images/frame_00002_.png", "points3D.txt"],
        )


if __name__ == "__main__":
    unittest.main()
