"""steps/meshify.py's pure helpers, and the step's argv against a fake trainer."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.steps.meshify import (MeshifyStep, camera_facing, encode_normal_png, head_centre_from_cap, look_at_c2w,
                                    orbit_cameras, read_ply_xyz, subject_frame, write_mesh_ply)


def _camera(position, target=(0, 0, 0), fx=1000.0, width=720, height=1280):
    rot = look_at_c2w(np.asarray(position, float), np.asarray(target, float))
    return SimpleNamespace(position=list(position), rotation=rot.tolist(), fx=fx, fy=fx, cx=width / 2, cy=height / 2,
                           width=width, height=height)


class TestHelpers(unittest.TestCase):
    def test_look_at_is_a_rotation_looking_at_the_target(self):
        r = look_at_c2w(np.array([0.0, 0.5, 3.0]), np.array([0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(r @ r.T, np.eye(3), atol=1e-9))
        self.assertGreater(np.linalg.det(r), 0.99)
        # the camera looks down its -z: -column 2 points from the camera to the target
        view = -r[:, 2]
        self.assertTrue(np.allclose(view, -np.array([0.0, 0.5, 3.0]) / np.linalg.norm([0.0, 0.5, 3.0]), atol=1e-9))
        self.assertGreater(r[1, 1], 0.9)  # up stays up

    def test_orbit_cameras_ring_the_target_at_the_radius(self):
        cams = orbit_cameras([0.1, 0.2, -2.0], 2.5, 1200.0, 720, 1280, [-30, 30], 45)
        self.assertEqual((cams["width"], cams["height"]), (720, 1280))
        self.assertEqual(len(cams["cameras"]), 16)
        for cam in cams["cameras"]:
            p = np.asarray(cam["position"]) - np.array([0.1, 0.2, -2.0])
            self.assertAlmostEqual(np.linalg.norm(p), 2.5, places=9)
            self.assertEqual(cam["fx"], 1200.0)
        names = [c["name"] for c in cams["cameras"]]
        self.assertIn("orb_e-30_a000.png", names)
        self.assertIn("orb_e+30_a315.png", names)
        self.assertEqual(len(set(names)), 16)
        front = next(c for c in cams["cameras"] if c["name"] == "orb_e-30_a000.png")
        self.assertGreater(front["position"][2], -2.0)  # azimuth 0 is the +z side

    def test_subject_frame_uses_the_bounds_centre_and_mean_camera_distance(self):
        v = np.array([[-0.3, -0.8, -2.5], [0.3, 0.8, -1.9]])
        cams = [_camera((0, 0, 0.2)), _camera((2.0, 0, -2.2))]
        centre, radius, fx = subject_frame(v, cams)
        self.assertTrue(np.allclose(centre, [0, 0, -2.2]))
        self.assertAlmostEqual(radius, (2.4 + 2.0) / 2, places=9)
        self.assertEqual(fx, 1000.0)

    def test_head_centre_sits_behind_the_cap(self):
        cap = np.array([[0.0, 0.7, -1.9], [0.02, 0.72, -1.9], [-0.02, 0.68, -1.9]])
        c = head_centre_from_cap(cap, [0, 0, 1])
        self.assertTrue(np.allclose(c, [0, 0.7, -1.95]))

    def test_camera_facing_points_from_the_subject_to_the_camera(self):
        cam = _camera((0, 0, 3.0))
        self.assertTrue(np.allclose(camera_facing(cam), [0, 0, 1], atol=1e-9))

    def test_normal_png_encodes_half_up_with_the_mask_in_alpha(self):
        n = np.zeros((2, 2, 3), np.float32)
        n[..., 2] = 1.0
        png = encode_normal_png(n, np.array([[1.0, 0.0], [0.5, 1.0]]))
        self.assertEqual(png.shape, (2, 2, 4))
        self.assertEqual(tuple(int(v) for v in png[0, 0, :3]), (127, 127, 255))  # (n + 1) / 2 * 255, truncated as colmap_export does
        self.assertEqual(tuple(int(v) for v in png[:, :, 3].ravel()), (255, 0, 127, 255))
        self.assertEqual(int(encode_normal_png(n, None)[0, 0, 3]), 255)

    def test_ply_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.ply"
            v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
            write_mesh_ply(path, v, np.array([[0, 1, 2]]))
            self.assertTrue(np.allclose(read_ply_xyz(path), v))


class TestStepArgv(unittest.TestCase):
    """The step against a fake `b2ctrain` that records its argv and writes the files the next stage opens."""

    FAKE = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f: f.write(json.dumps(args) + "\n")
sub = args[0]
def out_after(flag):
    return args[args.index(flag) + 1]
if sub == "mesh-fuse" and "--help" in args: print("--prior --views"); sys.exit(0)
if sub == "probe":
    d = out_after("--output-dir"); os.makedirs(d, exist_ok=True); open(os.path.join(d, "probe.json"), "w").write("{}")
elif sub in ("mesh-fuse", "mesh-refine", "mesh-bake"):
    open(out_after("--output"), "wb").write(b"ply\n")
elif sub == "mesh-unwrap":
    d = out_after("--output"); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "atlas.json"), "w").write(json.dumps({"res": 4096, "tris": 300000, "uv_verts": 1, "covered": 2}))
sys.exit(0)
'''

    def test_chain_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake = tmp / "b2ctrain"
            fake.write_text(self.FAKE)
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            log = tmp / "argv.jsonl"
            os.environ["FAKE_LOG"] = str(log)
            splat = tmp / "scene.ply"
            splat.write_bytes(b"ply\n")
            cap = tmp / "cap.ply"
            write_mesh_ply(cap, np.array([[0, 0.7, -1.9]] * 3, np.float32), np.array([[0, 1, 2]]))
            cams = [_camera((0, 0, 0.5)), _camera((0.5, 0, -2.0))]
            body = (np.array([[-0.3, -0.8, -2.5], [0.3, 0.8, -1.9], [0, 0, -2.2]], np.float32), np.array([[0, 1, 2]]))
            normals = [np.zeros((1280, 720, 3), np.float32)] * 2
            params = MeshifyStep.resolve_params({"trainer_path": str(fake), "output_dir": str(tmp / "mesh"), "orbit_azimuth_step": 90.0,
                                                 "orbit_elevations": [0], "keep_probes": True})
            result = MeshifyStep().run({"splat_path": str(splat), "cameras": cams, "mesh_world": body, "cap_path": str(cap),
                                        "normal_maps": normals, "anchor_frame_index": 1}, params)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            subs = [c[0] for c in calls]
            self.assertEqual(subs, ["mesh-fuse", "probe", "probe", "mesh-fuse", "mesh-refine", "mesh-bake", "mesh-unwrap"])
            fuse = calls[3]
            self.assertEqual(fuse.count("--views"), 2)
            self.assertEqual(fuse.count("--carve"), 2)
            self.assertNotIn("--protect", fuse)  # the face's shape is in the splat (face_geometry splat)
            refine = calls[4]
            self.assertIn("--keep", refine)  # but the refine keeps the cap's footprint as fused
            bake = calls[5]
            self.assertEqual(bake.count("--views"), 1)  # the orbit only, not the training views
            self.assertEqual(bake[bake.index("--project") + 2], "1")  # the anchor camera
            unwrap = calls[6]
            self.assertIn("--protect-head", unwrap)
            self.assertIn("--cap", unwrap)
            self.assertTrue(result["mesh_dir"].endswith("atlas"))
            self.assertEqual(result["mesh_stats"]["anchor_frame_index"], 1)
            self.assertEqual(result["mesh_stats"]["cameras"]["orbit"], 4)
            self.assertTrue((tmp / "mesh" / "meshify.json").is_file())
            # the orbit went to the trainer in the cameras.json layout
            orbit = json.loads((tmp / "mesh" / "cams" / "orbit.json").read_text())
            self.assertEqual(len(orbit["cameras"]), 4)


if __name__ == "__main__":
    unittest.main()
