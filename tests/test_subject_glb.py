"""export_subject (steps/subject_glb.py): the run's subject file from the scene, and the bare .ply beside it.

Synthetic scene values in the shapes refit_body_to_splat and snapshot_orbit publish; skips without b2cgltf (the
package the pod image installs, b2cgltf SPEC.md).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import ply_meta
from tests.helpers import orbit_dataset, run_step

import pipeline.steps  # noqa: F401

try:
    from b2cgltf import document, read
    from b2cgltf.b2crunner import convert
except ImportError:  # pragma: no cover - depends on the local env
    document = None

FIELDS = ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "opacity",
          "f_dc_0", "f_dc_1", "f_dc_2", "seg_label", "seg_conf")


def _trainer_ply(path: Path, n: int = 12) -> dict:
    rng = np.random.default_rng(0)
    fields = {k: rng.normal(size=n).astype(np.float32) for k in FIELDS}
    fields["seg_label"] = rng.integers(0, 28, n).astype(np.float32)
    rec = np.zeros(n, dtype=[(k, "<f4") for k in FIELDS])
    for k in FIELDS:
        rec[k] = fields[k]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {k}\n" for k in FIELDS) + "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode() + rec.tobytes())
    return fields


def _rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _scene(rng):
    parents = np.array([-1, 0, 1, 1])
    verts = rng.normal(size=(6, 3)).astype(np.float32)
    wfr = {"scale": 1.02, "rotation": _rotation(rng), "translation": np.array([0.1, -0.2, 0.3])}
    body_params = {
        "pose_params": {"global_rot": np.zeros(3, np.float32), "shape_params": np.ones(45, np.float32)},
        "world_from_raw": wfr, "joints": rng.normal(size=(4, 3)),
        "global_rots": np.stack([_rotation(rng) for _ in range(4)]), "joint_parents": parents,
        "model": "facebook/sam-3d-body-dinov3 assets/mhr_model.pt",
        "model_params": np.arange(204, dtype=np.float32), "hand_idx": np.array([2, 3]),
        "joint_names": ["body_world", "root", "l_upleg", "r_upleg"], "model_sha256": "ab" * 32,
    }
    rig = {"skin_vertex": np.array([0, 1, 2, 3, 4, 5, 5]), "skin_joint": np.array([0, 1, 2, 3, 1, 2, 3]),
           "skin_weight": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.75]), "joint_parents": parents,
           "rest_vertices": verts}
    return body_params, (verts, np.array([[0, 1, 2], [2, 3, 4], [3, 4, 5]], np.int32)), rig


@unittest.skipIf(document is None, "b2cgltf is not installed")
class TestExportSubject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ply = Path(self.tmp.name) / "ply" / "scene.ply"

    def _run(self, with_record: bool = True):
        rng = np.random.default_rng(1)
        fields = _trainer_ply(self.ply)
        body_params, mesh_world, rig = _scene(rng)
        dataset = orbit_dataset(n_frames=5)
        record = {
            "helix": {"n_frames": 5, "n_loops": 1, "amplitude_deg": 10.0, "lead_in_deg": 0.0, "lead_out_deg": 0.0},
            "extension": {"before": 0, "after": 0, "overlap_before": 0, "overlap_after": 0, "tilt_deg": 0.0},
            "pass_frames": 5, "anchor_frame_index": np.int64(0), "orbit_cameras": list(dataset.cameras),
            "extras": {"orbit_target": [0.0, 1.0, 0.0]}, "prompt": "a subject",
            "settings": {"seed": 1, "resolution": [720, 1280], "framing": "full"},
            "_images": {"front": np.full((8, 8, 3), 200, np.uint8)},
        }
        inputs = {"splat_path": str(self.ply), "cameras": list(dataset.cameras),
                  "image_names": [f"frame_{i:05d}_.png" for i in range(1, 6)], "body_params": body_params,
                  "mesh_world": mesh_world, "rig_binding": rig}
        if with_record:
            inputs["record"] = record
        out = run_step("export_subject", inputs)
        return fields, body_params, out

    def test_the_subject_file_carries_splat_body_and_capture(self):
        fields, body_params, out = self._run()
        path = Path(out["subject_path"])
        self.assertEqual(path, self.ply.parent / "scene.glb")
        doc = document.load(path)
        js = doc.json

        splat = read.to_trainer_ply_fields(read.splat(doc, read.current_splat(doc)))
        for k in ("x", "f_dc_2", "seg_label"):
            self.assertTrue(np.array_equal(np.asarray(splat[k], np.float32), fields[k]))

        # the skeleton is the refit's joints in the splat's world frame (ply_meta.body_world, record version 2)
        joints_w, rots_w = ply_meta.body_world(body_params["world_from_raw"], body_params["joints"],
                                               body_params["global_rots"])
        expected = convert.joint_nodes(joints_w, rots_w, body_params["joint_parents"])
        sk = read.skeleton(doc)
        self.assertEqual(sk.names, body_params["joint_names"])
        self.assertTrue(np.allclose(sk.rest_t, expected["translation"], atol=1e-6))

        body = next(n for n in js["nodes"] if n.get("name") == "b2c_body")
        mhr = body["extensions"]["B2C_mhr"]
        self.assertEqual(mhr["model"]["sha256"], "ab" * 32)
        self.assertEqual(mhr["sourceRecordVersion"], 2)

        orbit = js["extensions"]["B2C_orbit"]
        self.assertEqual(orbit["final_cameras"]["names"][0], "frame_00001_.png")
        self.assertEqual(orbit["final_cameras"]["dataset"], "../colmap/")
        self.assertEqual(orbit["anchor_frame_index"], 0)
        self.assertEqual(orbit["prompt"], "a subject")
        self.assertIn("orbit_cameras", orbit)
        self.assertEqual(js["images"][orbit["images"]["front"]]["mimeType"], "image/png")

        # the .ply beside it is left as the trainer wrote it: no b2c records
        self.assertFalse(any(c.startswith("b2c.") for c in ply_meta.read_comments(self.ply)))

    def test_without_an_orbit_record_only_the_final_cameras_go_in(self):
        _fields, _body, out = self._run(with_record=False)
        orbit = document.load(out["subject_path"]).json["extensions"]["B2C_orbit"]
        self.assertNotIn("orbit_cameras", orbit)
        self.assertEqual(len(orbit["final_cameras"]["names"]), 5)

    def test_a_refit_that_does_not_publish_the_model_row_is_refused(self):
        rng = np.random.default_rng(1)
        _trainer_ply(self.ply)
        body_params, mesh_world, rig = _scene(rng)
        del body_params["model_params"]
        dataset = orbit_dataset(n_frames=2)
        with self.assertRaisesRegex(ValueError, "model_params"):
            run_step("export_subject", {"splat_path": str(self.ply), "cameras": list(dataset.cameras),
                                        "image_names": ["a.png", "b.png"], "body_params": body_params,
                                        "mesh_world": mesh_world, "rig_binding": rig})


class TestTrainerFilesLeavePly(unittest.TestCase):
    def test_body_rig_omega_moves_out_of_the_export_directory(self):
        from pipeline.steps.brush import _move_trainer_files

        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "ply" / "scene.ply"
            ply.parent.mkdir()
            ply.write_bytes(b"ply")
            (ply.parent / "body_rig_omega.json").write_text("{}")
            _move_trainer_files(ply, Path(tmp) / "debug" / "final_splat")
            self.assertEqual(sorted(p.name for p in ply.parent.iterdir()), ["scene.ply"])
            self.assertTrue((Path(tmp) / "debug" / "final_splat" / "body_rig_omega.json").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
