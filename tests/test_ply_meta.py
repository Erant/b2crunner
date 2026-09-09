"""The body record in the delivered PLY's header: written without touching
the vertex data, read back exactly, replaced rather than duplicated."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import ply_meta


def _write_ply(path: Path, n: int = 5) -> np.ndarray:
    from plyfile import PlyData, PlyElement
    data = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    data["x"] = np.arange(n); data["y"] = np.arange(n) * 2; data["z"] = -np.arange(n); data["opacity"] = 0.5
    PlyData([PlyElement.describe(data, "vertex")], comments=["Exported from Brush", "Vertical axis: y"]).write(str(path))
    return data


def _body():
    rng = np.random.RandomState(3)
    pose = {k: rng.randn(n).astype(np.float32) for k, n in
            zip(ply_meta.POSE_KEYS, (3, 133, 108, 28, 45, 72, 3, 68))}
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
    wfr = {"scale": 0.9864, "rotation": rot, "translation": np.array([0.1, -0.2, 0.3])}
    joints = rng.randn(127, 3).astype(np.float32)
    rots = np.tile(np.eye(3, dtype=np.float32), (127, 1, 1))
    parents = np.array([-1] + list(range(126)), np.int32)
    return pose, wfr, joints, rots, parents


class TestEmbed(unittest.TestCase):
    def test_round_trip_keeps_the_vertices_and_the_record(self):
        from plyfile import PlyData
        pose, wfr, joints, rots, parents = _body()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.ply"
            data = _write_ply(path)
            lines = ply_meta.body_comments(pose, wfr, joints=joints, global_rots=rots, joint_parents=parents,
                                           model="facebook/sam-3d-body-dinov3 assets/mhr_model.pt")
            n = ply_meta.embed_comments(path, lines)
            self.assertEqual(n, 2 + len(lines))
            ply = PlyData.read(str(path))
            self.assertEqual(ply["vertex"].count, len(data))
            np.testing.assert_array_equal(ply["vertex"]["y"], data["y"])
            self.assertEqual(ply.comments[:2], ["Exported from Brush", "Vertical axis: y"])
            rec = ply_meta.parse_body_comments(ply_meta.read_comments(path))
            self.assertEqual(rec["version"], "1")
            self.assertEqual(rec["model"], "facebook/sam-3d-body-dinov3 assets/mhr_model.pt")
            for k in ply_meta.POSE_KEYS:
                np.testing.assert_array_equal(rec["pose_params"][k], pose[k])
            self.assertAlmostEqual(float(rec["world_from_raw"]["scale"]), 0.9864)
            np.testing.assert_allclose(rec["world_from_raw"]["rotation"], wfr["rotation"])
            np.testing.assert_array_equal(rec["joint_parents"], parents)
            # joints and rotations land in the world frame
            expect = 0.9864 * joints.astype(np.float64) @ wfr["rotation"].T + wfr["translation"]
            np.testing.assert_allclose(rec["joints"], expect, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(rec["global_rots"], np.tile(wfr["rotation"], (127, 1, 1)), atol=1e-7)

    def test_embedding_twice_replaces_the_record(self):
        pose, wfr, *_ = _body()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.ply"
            _write_ply(path)
            ply_meta.embed_comments(path, ply_meta.body_comments(pose, wfr))
            pose["global_trans"] = np.array([9, 9, 9], np.float32)
            ply_meta.embed_comments(path, ply_meta.body_comments(pose, wfr))
            comments = ply_meta.read_comments(path)
            self.assertEqual(sum(c.startswith(ply_meta.PREFIX + "pose_params.global_trans") for c in comments), 1)
            np.testing.assert_array_equal(ply_meta.parse_body_comments(comments)["pose_params"]["global_trans"], [9, 9, 9])
            self.assertEqual(comments[0], "Exported from Brush")

    def test_no_record_parses_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.ply"
            _write_ply(path)
            self.assertEqual(ply_meta.parse_body_comments(ply_meta.read_comments(path)), {})

    def test_a_missing_pose_entry_is_refused(self):
        pose, wfr, *_ = _body()
        del pose["scale_offsets"]
        with self.assertRaises(ValueError):
            ply_meta.body_comments(pose, wfr)

    def test_not_a_ply_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ply"
            path.write_bytes(b"hello\n")
            with self.assertRaises(ValueError):
                ply_meta.embed_comments(path, ["a"])
