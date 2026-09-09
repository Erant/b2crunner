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

    def test_off_by_default_and_points_fallback_without_a_mesh(self):
        self.assertEqual(get_step_class("brush").declared_params()["hollow_weight"].default, 0.0)
        seen = self._run(_inputs())
        self.assertNotIn("--hollow-weight", seen["cmd"])
        self.assertIsNone(seen["mesh"])
        # A weight without a mesh still reaches the trainer, which falls back to surfels on points3D.txt.
        seen = self._run(_inputs(), hollow_weight=0.5)
        cmd = seen["cmd"]
        self.assertIn("--hollow-weight", cmd)
        self.assertNotIn("--mesh", cmd)
        self.assertEqual(cmd[cmd.index("--hollow-proxy") + 1], "auto")

    def test_mesh_is_written_and_passed(self):
        inputs = {**_inputs(), "mesh": _mesh()}
        seen = self._run(inputs, hollow_weight=0.5, hollow_margin=0.04, hollow_dilate=3)
        cmd = seen["cmd"]
        self.assertIn("--hollow-weight", cmd)
        self.assertEqual(cmd[cmd.index("--hollow-weight") + 1], "0.5")
        self.assertEqual(cmd[cmd.index("--hollow-margin") + 1], "0.04")
        self.assertEqual(cmd[cmd.index("--hollow-dilate") + 1], "3")
        self.assertTrue(cmd[cmd.index("--mesh") + 1].endswith("mesh.ply"))
        self.assertEqual(cmd[cmd.index("--hollow-proxy") + 1], "auto")
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


class TestBrushBodyRecord(unittest.TestCase):
    """A wired `body_params` input ends up in the exported .ply's header."""

    def test_the_record_is_embedded_after_the_export(self):
        from pipeline import ply_meta
        from plyfile import PlyData, PlyElement
        step_class = get_step_class("brush")
        step = step_class()
        calls = []

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            calls.append(list(cmd))
            data = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            data["x"] = [1, 2, 3]
            PlyData([PlyElement.describe(data, "vertex")], comments=["Exported from Brush"]).write(str(ply_path))

        step._run_brush = fake_run_brush
        rng = np.random.RandomState(0)
        pose = {k: rng.randn(n).astype(np.float32) for k, n in
                zip(ply_meta.POSE_KEYS, (3, 133, 108, 28, 45, 72, 3, 68))}
        body = {
            "pose_params": pose,
            "world_from_raw": {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)},
            "joints": rng.randn(127, 3).astype(np.float32),
            "global_rots": np.tile(np.eye(3, dtype=np.float32), (127, 1, 1)),
            "joint_parents": np.arange(-1, 126, dtype=np.int32),
            "model": "facebook/sam-3d-body-dinov3 assets/mhr_model.pt",
        }
        inputs = {**_inputs(), "body_params": body}
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0, "polish_steps": 1000})
            out = step.run(inputs, params)
            ply = PlyData.read(out["splat_path"])
            np.testing.assert_array_equal(ply["vertex"]["x"], [1, 2, 3])
            self.assertEqual(ply.comments[0], "Exported from Brush")
            rec = ply_meta.parse_body_comments(ply_meta.read_comments(out["splat_path"]))
        self.assertEqual(len(calls), 2)  # cold run + polish: the record survives the last export
        np.testing.assert_array_equal(rec["pose_params"]["body_pose_params"], pose["body_pose_params"])
        np.testing.assert_allclose(rec["joints"], body["joints"], rtol=1e-6, atol=1e-6)
        self.assertEqual(rec["joint_parents"].shape, (127,))
        self.assertEqual(rec["model"], body["model"])

    def test_no_record_without_the_input(self):
        from pipeline import ply_meta
        from plyfile import PlyData, PlyElement
        step_class = get_step_class("brush")
        step = step_class()

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            data = np.zeros(1, dtype=[("x", "f4")])
            PlyData([PlyElement.describe(data, "vertex")]).write(str(ply_path))

        step._run_brush = fake_run_brush
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0})
            out = step.run(_inputs(), params)
            self.assertEqual(ply_meta.parse_body_comments(ply_meta.read_comments(out["splat_path"])), {})
