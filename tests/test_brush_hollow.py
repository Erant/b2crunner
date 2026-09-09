"""The hollow loss on the brush step's argv: a wired `mesh` input is written
beside the COLMAP model as mesh.ply and, with `hollow_weight` > 0, the trainer
is pointed at it; without a mesh the loss is off however the weight is set."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401

from .test_brush_evidence import _inputs


def _mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


class TestBrushHollowArgv(unittest.TestCase):
    def _run(self, inputs, **overrides):
        step_class = get_step_class("brush")
        step = step_class()
        seen = {}

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            seen["cmd"] = list(cmd)
            mesh = Path(colmap_dir) / "mesh.ply" if colmap_dir else None
            seen["mesh"] = mesh.read_bytes() if mesh and mesh.exists() else None
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0, **overrides})
            step.run(inputs, params)
        return seen

    def test_off_by_default_and_without_a_mesh(self):
        self.assertEqual(get_step_class("brush").declared_params()["hollow_weight"].default, 0.0)
        seen = self._run(_inputs())
        self.assertNotIn("--hollow-weight", seen["cmd"])
        self.assertIsNone(seen["mesh"])
        # A weight without a mesh is a warning, not a flag the trainer would refuse to honour.
        seen = self._run(_inputs(), hollow_weight=0.5)
        self.assertNotIn("--hollow-weight", seen["cmd"])
        self.assertNotIn("--mesh", seen["cmd"])

    def test_mesh_is_written_and_passed(self):
        inputs = {**_inputs(), "mesh": _mesh()}
        seen = self._run(inputs, hollow_weight=0.5, hollow_margin=0.04, hollow_dilate=3)
        cmd = seen["cmd"]
        self.assertIn("--hollow-weight", cmd)
        self.assertEqual(cmd[cmd.index("--hollow-weight") + 1], "0.5")
        self.assertEqual(cmd[cmd.index("--hollow-margin") + 1], "0.04")
        self.assertEqual(cmd[cmd.index("--hollow-dilate") + 1], "3")
        self.assertTrue(cmd[cmd.index("--mesh") + 1].endswith("mesh.ply"))
        # The sidecar: a binary little-endian ply with 4 vertices and 4 triangles.
        head = seen["mesh"].split(b"end_header\n")[0].decode()
        self.assertIn("format binary_little_endian 1.0", head)
        self.assertIn("element vertex 4", head)
        self.assertIn("element face 4", head)
        body = seen["mesh"].split(b"end_header\n", 1)[1]
        self.assertEqual(len(body), 4 * 12 + 4 * (1 + 12))

    def test_mesh_written_even_with_the_loss_off(self):
        """One param away from on: the sidecar is there, the flags are not."""
        seen = self._run({**_inputs(), "mesh": _mesh()})
        self.assertIsNotNone(seen["mesh"])
        self.assertNotIn("--hollow-weight", seen["cmd"])
        self.assertNotIn("--mesh", seen["cmd"])


if __name__ == "__main__":
    unittest.main()
