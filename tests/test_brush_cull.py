"""The visibility cull and the final-export evidence cull on the brush step's argv.

`--cull-weight` prunes, at every refine, the splats whose rendered mass over
the refine window is below the threshold (the alignment refits included);
`--evidence-prune-wall` drops, at the final export, the splats no training
view renders. Both are b2ctrain flags (docs/unsupported-splats-2026-09-28.md
there); the deliverable training turns them on in helical.yaml. These check
the argv `run()` builds, with the training itself stubbed out.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401

from .test_brush_evidence import TestBrushEvidenceArgv

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestBrushCullArgv(TestBrushEvidenceArgv):
    def test_the_cull_is_off_by_default(self):
        """A splat culled here is gone from the deliverable .ply; a workflow
        opts in where it was measured (helical.yaml's train_final_splat)."""
        params = get_step_class("brush").declared_params()
        self.assertEqual(params["cull_weight"].default, 0.0)
        self.assertIsNone(params["evidence_prune_wall"].default)
        argv = self._argv()
        self.assertNotIn("--cull-weight", argv)
        self.assertNotIn("--evidence-prune-wall", argv)

    def test_the_thresholds_go_through(self):
        argv = self._argv(cull_weight=5, evidence_prune_wall=5)
        self.assertEqual(float(argv[argv.index("--cull-weight") + 1]), 5.0)
        self.assertEqual(float(argv[argv.index("--evidence-prune-wall") + 1]), 5.0)

    def test_the_cull_rides_every_invocation(self):
        """The polish is the same command with growth off; the cull is a
        regulariser like the hollow loss and stays on it."""
        step_class = get_step_class("brush")
        step = step_class()
        cmds = []

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            cmds.append(list(cmd))
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0, "polish_steps": 100, "cull_weight": 2})
            from .test_brush_evidence import _inputs
            step.run(_inputs(), params)
        self.assertEqual(len(cmds), 2)
        for cmd in cmds:
            self.assertEqual(float(cmd[cmd.index("--cull-weight") + 1]), 2.0)

    def test_the_deliverable_training_culls(self):
        wf = yaml.safe_load((REPO_ROOT / "pipeline" / "workflows" / "helical.yaml").read_text())
        steps = {s["id"]: s for s in wf["steps"]}
        params = steps["train_final_splat"]["params"]
        self.assertEqual(params["cull_weight"], 5)
        self.assertEqual(params["evidence_prune_wall"], 5)
        self.assertEqual(params["hollow_weight"], 0.5, "the cull goes with the hollow loss, not instead of it")


del TestBrushEvidenceArgv  # not re-run under this module's name


if __name__ == "__main__":
    unittest.main()
