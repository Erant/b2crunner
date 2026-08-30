"""brush's evidence export — the flags render_splat's confidence mode needs.

`--export-evidence` measures each Gaussian's multi-view evidence against
every training view and writes it into the exported .ply as `ev_*` vertex
properties. That block is what `brush-splat-render --confidence` reads to
gate a render in 3-D, which is what replaced `mask_splat`'s per-pixel alpha
cut (see docs/spatial-reinforcement.md). It is on by default because both
trainings want it: the cost is seconds, and a mismatch degrades loudly
rather than silently — the renderer warns and falls back to plain alpha.

These check the argv `run()` builds, with the training itself stubbed out.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401


def _inputs():
    """The smallest batch ColmapExporter will accept: two 8x8 views."""
    from body2colmap.camera import Camera

    cameras = [
        Camera(
            focal_length=(8.0, 8.0),
            image_size=(8, 8),
            principal_point=(4.0, 4.0),
            position=np.array([0.0, 0.0, float(i + 1)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(2)
    ]
    return {
        "cameras": cameras,
        "image_names": ["frame_00001_.png", "frame_00002_.png"],
        "points_3d": (
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.uint8),
        ),
        "images": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
        "masks": [np.ones((8, 8), dtype=np.float32) for _ in range(2)],
    }


class TestBrushEvidenceArgv(unittest.TestCase):
    def _argv(self, **overrides):
        """The argv `run()` hands `_run_brush`, without running brush."""
        step_class = get_step_class("brush")
        step = step_class()
        seen = {}

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            seen["cmd"] = list(cmd)
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"export_dir": tmp, **overrides})
            step.run(_inputs(), params)
        return seen["cmd"]

    def test_evidence_is_exported_by_default(self):
        """Both trainings want it, and the next-but-one step depends on it:
        without the ev_* block a confidence render warns and falls back to
        plain alpha — silently back to what mask_splat used to patch up."""
        self.assertIs(
            get_step_class("brush").declared_params()["export_evidence"].default, True
        )
        self.assertIn("--export-evidence", self._argv())

    def test_it_can_be_turned_off(self):
        self.assertNotIn("--export-evidence", self._argv(export_evidence=False))

    def test_pruning_is_off_unless_asked_for(self):
        """A splat dropped by --evidence-prune-inmask is gone from the
        deliverable .ply, not merely hidden in one render, and the threshold
        has not been looked at on a real run."""
        self.assertIs(
            get_step_class("brush").declared_params()["evidence_prune_inmask"].default,
            None,
        )
        self.assertNotIn("--evidence-prune-inmask", self._argv())

    def test_a_prune_threshold_goes_through(self):
        argv = self._argv(evidence_prune_inmask=0.2)
        self.assertEqual(argv[argv.index("--evidence-prune-inmask") + 1], "0.2")

    def test_a_zero_prune_threshold_is_not_the_same_as_off(self):
        """0.0 still drops the splats no view supported at all, which is a
        real request — `if x is not None`, not `if x`."""
        self.assertIn("--evidence-prune-inmask", self._argv(evidence_prune_inmask=0.0))

    def test_the_normal_weight_is_off_by_default_and_untuned(self):
        argv = self._argv()
        self.assertNotIn("--evidence-normal-weight", argv)
        argv = self._argv(evidence_normal_weight=0.5)
        self.assertEqual(argv[argv.index("--evidence-normal-weight") + 1], "0.5")


if __name__ == "__main__":
    unittest.main()
