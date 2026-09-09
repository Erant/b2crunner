"""The per-view deformation rig: which joints are active, the file the trainer reads, and the brush argv."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline.steps  # noqa: F401
from pipeline import body_rig
from pipeline.registry import get_step_class
from pipeline.steps import brush as brush_mod

from .test_brush_evidence import _inputs


def _skeleton():
    """root(0) -> spine(1) -> {left arm 2 -> hand 3, head 4}; 100 vertices along y, skinned by height."""
    parents = np.array([-1, 0, 1, 2, 1], np.int32)
    joints = np.array([[0, 0, 0], [0, 1, 0], [0.5, 1.5, 0], [1.0, 1.5, 0], [0, 2, 0]], np.float64)
    n = 100
    verts = np.zeros((n, 3)); verts[:, 1] = np.linspace(0, 2.5, n)
    # dominant joint: 0-40 root, 40-60 spine, 60-70 arm, 70-80 hand, 80-100 head
    dom = np.concatenate([np.zeros(40), np.ones(20), np.full(10, 2), np.full(10, 3), np.full(20, 4)]).astype(int)
    sv = np.arange(n); sj = dom; sw = np.ones(n)
    return verts, joints, parents, sv, sj, sw


class TestBuild(unittest.TestCase):
    def test_root_chain_is_excluded_and_small_joints_kept(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, max_subtree_fraction=0.5, vertex_stride=1)
        # subtrees: root 100, spine 60, arm 20, hand 10, head 20 -> root and spine over half the body
        self.assertEqual(list(rig["active"]), [2, 3, 4])
        self.assertEqual(list(rig["excluded_root_chain"]), [0, 1])
        self.assertEqual(list(rig["subtree"]), [100, 60, 20, 10, 20])

    def test_min_subtree_drops_a_small_joint(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=15, max_subtree_fraction=0.5, vertex_stride=1)
        self.assertEqual(list(rig["active"]), [2, 4])  # the hand's 10 vertices are under the floor

    def test_stride_and_top4_weights(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=4)
        self.assertEqual(len(rig["verts"]), 25)
        self.assertEqual(rig["joints"].shape, (25, 4))
        np.testing.assert_allclose(rig["weights"].sum(1), 1.0, atol=1e-6)
        self.assertEqual(rig["joints"][0, 0], 0)  # the first vertex is root-dominated

    def test_refusals(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        with self.assertRaises(ValueError):
            body_rig.build_body_rig(verts, joints, np.array([1, 0, 1, 2, 1]), sv, sj, sw)  # not parent-first
        with self.assertRaises(ValueError):
            body_rig.build_body_rig(verts, joints[:3], parents, sv, sj, sw)
        with self.assertRaises(ValueError):
            body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=1000)


class TestFile(unittest.TestCase):
    def test_round_trip(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rig.bin"
            body_rig.write_body_rig(path, rig, ["frame_00001_.png", "frame_00002_.png"])
            back = body_rig.read_body_rig(path)
            self.assertEqual(path.read_bytes()[:8], b"B2CRIG2\0")
        self.assertEqual(back["names"], ["frame_00001_.png", "frame_00002_.png"])
        np.testing.assert_array_equal(back["verts"], rig["verts"])
        np.testing.assert_array_equal(back["joints"], rig["joints"])
        np.testing.assert_allclose(back["weights"], rig["weights"])
        np.testing.assert_array_equal(back["parents"], parents)
        np.testing.assert_allclose(back["joint_positions"], joints.astype(np.float32))
        np.testing.assert_array_equal(back["active"], rig["active"])

    def test_long_name_refused(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                body_rig.write_body_rig(Path(tmp) / "rig.bin", rig, ["x" * 64])


class TestStep(unittest.TestCase):
    def _inputs(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
        wfr = {"scale": 2.0, "rotation": rot, "translation": np.array([1.0, 0.0, 0.0])}
        world_verts = 2.0 * verts @ rot.T + wfr["translation"]
        raw_joints = joints  # the step maps them into the world frame
        return {"mesh_world": (world_verts, np.zeros((1, 3), np.int32)), "joints": raw_joints, "world_from_raw": wfr,
                "rig_binding": {"joint_parents": parents, "skin_vertex": sv, "skin_joint": sj, "skin_weight": sw}}

    def test_joints_land_in_the_world_frame(self):
        cls = get_step_class("build_body_rig"); step = cls()
        with tempfile.TemporaryDirectory() as tmp:
            out = step.run(self._inputs(), cls.resolve_params({"min_subtree": 5, "vertex_stride": 1, "debug_dir": tmp}))
            self.assertTrue((Path(tmp) / "body_rig.json").exists())
        rig = out["body_rig"]
        _, joints, *_ = _skeleton()
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
        np.testing.assert_allclose(rig["joint_positions"], (2.0 * joints @ rot.T + [1, 0, 0]).astype(np.float32), rtol=1e-6)
        self.assertEqual(out["body_rig_stats"]["active"], 3)
        self.assertEqual(out["body_rig_stats"]["excluded_root_chain"], [0, 1])

    def test_refusals(self):
        cls = get_step_class("build_body_rig"); step = cls(); params = cls.resolve_params({})
        inputs = self._inputs()
        for key in ("mesh_world", "joints", "world_from_raw", "rig_binding"):
            bad = dict(inputs); del bad[key]
            with self.assertRaises(ValueError):
                step.run(bad, params)


class TestBrushArgv(unittest.TestCase):
    def _run(self, inputs, supports, **overrides):
        step_class = get_step_class("brush"); step = step_class()
        seen = {}

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            seen.setdefault("cmds", []).append(list(cmd))
            rig = Path(colmap_dir) / "body_rig.bin" if colmap_dir else None
            seen["rig"] = body_rig.read_body_rig(rig) if rig and rig.exists() else None
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        brush_mod._HELP_PROBE["brush"] = "--align-iters --body-rig" if supports else "--align-iters"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0, "brush_path": "brush", **overrides})
                step.run(inputs, params)
        finally:
            brush_mod._HELP_PROBE.pop("brush", None)
        return seen

    def _rig_inputs(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=1)
        return {**_inputs(), "body_rig": rig}

    def test_rig_written_and_flags_on_every_invocation(self):
        seen = self._run(self._rig_inputs(), True, polish_steps=500)
        self.assertEqual(seen["rig"]["names"], ["frame_00001_.png", "frame_00002_.png"])
        self.assertEqual(list(seen["rig"]["active"]), [2, 3, 4])
        for cmd in seen["cmds"]:
            self.assertIn("--body-rig", cmd)
            i = cmd.index("--body-rig")
            self.assertTrue(cmd[i + 1].endswith("body_rig.bin"))
            for flag, value in (("--body-rig-start-iter", "1000"), ("--body-rig-smooth", "0.05"), ("--body-rig-zero", "0.02"), ("--body-rig-lr", "0.002")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertEqual(len(seen["cmds"]), 2)

    def test_no_flags_without_trainer_support(self):
        seen = self._run(self._rig_inputs(), False)
        self.assertIsNone(seen["rig"])
        self.assertNotIn("--body-rig", seen["cmds"][0])

    def test_param_off_or_input_absent(self):
        seen = self._run(self._rig_inputs(), True, body_rig=False)
        self.assertNotIn("--body-rig", seen["cmds"][0])
        seen = self._run(_inputs(), True)
        self.assertNotIn("--body-rig", seen["cmds"][0])
