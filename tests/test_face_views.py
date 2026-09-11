"""The per-view face: rig v3 (deltas), the frame-to-frame delta assignment, and the geometry the eye paste stands on."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline.steps  # noqa: F401
from pipeline import body_rig
from pipeline.registry import get_step_class
from pipeline.steps import brush as brush_mod
from pipeline.steps import face_views as fv

from .test_body_rig import _skeleton
from .test_brush_evidence import _inputs


class TestAssignViewDeltas(unittest.TestCase):
    def _fitted(self, *views):
        return {v: np.full((2, 3), float(v), np.float32) for v in views}

    def test_fitted_views_keep_their_deltas_and_gaps_interpolate(self):
        deltas, how = body_rig.assign_view_deltas(self._fitted(0, 4), 5, 2, gap=4, hold=3)
        self.assertEqual(how, ["fit", "interp", "interp", "interp", "fit"])
        np.testing.assert_allclose(deltas[:, 0, 0], [0, 1, 2, 3, 4])

    def test_a_gap_wider_than_gap_is_held_from_the_nearer_side_then_zero(self):
        # fitted at 0 and 10: nine unfitted between them, more than gap+1
        deltas, how = body_rig.assign_view_deltas(self._fitted(0, 10), 11, 2, gap=4, hold=3)
        self.assertEqual(how, ["fit", "hold", "hold", "hold", "zero", "zero", "zero", "hold", "hold", "hold", "fit"])
        np.testing.assert_allclose(deltas[1:4, 0, 0], 0.0)
        np.testing.assert_allclose(deltas[7:10, 0, 0], 10.0)
        np.testing.assert_allclose(deltas[4:7], 0.0)

    def test_the_ends_hold_and_hold_zero_means_canonical(self):
        _, how = body_rig.assign_view_deltas(self._fitted(5), 8, 2, gap=4, hold=1)
        self.assertEqual(how, ["zero", "zero", "zero", "zero", "hold", "fit", "hold", "zero"])
        _, how = body_rig.assign_view_deltas(self._fitted(5), 8, 2, gap=4, hold=0)
        self.assertEqual(how.count("hold"), 0)


class TestFaceOnlyWeights(unittest.TestCase):
    def test_full_on_the_face_fading_to_zero(self):
        rig = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.05, 0, 0]], np.float64)
        face = np.array([[0, 0, 0]], np.float64)
        w = body_rig.face_only_weights(rig, face, fade=0.03)
        np.testing.assert_allclose(w, [1.0, 2 / 3, 1 / 3, 0.0], atol=1e-6)
        with self.assertRaises(ValueError):
            body_rig.face_only_weights(rig, face[:0], fade=0.03)


class TestRigV3File(unittest.TestCase):
    def test_deltas_round_trip_in_the_names_order_and_missing_views_are_zero(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=4)
        self.assertEqual(list(rig["vertex_index"]), list(range(0, 100, 4)))
        nv = len(rig["verts"])
        d1 = np.full((nv, 3), 0.25, np.float32)
        rig3 = {**rig, "view_deltas": {"b.png": d1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rig.bin"
            body_rig.write_body_rig(path, rig3, ["a.png", "b.png", "c.png"])
            self.assertEqual(path.read_bytes()[:8], b"B2CRIG3\0")
            back = body_rig.read_body_rig(path)
            # the v2 part is byte-identical to a v2 write of the same rig
            body_rig.write_body_rig(Path(tmp) / "rig2.bin", rig, ["a.png", "b.png", "c.png"])
            v2 = (Path(tmp) / "rig2.bin").read_bytes()
            self.assertEqual(path.read_bytes()[8:len(v2)], v2[8:])
        self.assertEqual(back["deltas"].shape, (3, nv, 3))
        np.testing.assert_array_equal(back["deltas"][0], 0)
        np.testing.assert_array_equal(back["deltas"][1], d1)
        np.testing.assert_array_equal(back["deltas"][2], 0)

    def test_a_delta_of_the_wrong_shape_is_refused(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=4)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                body_rig.write_body_rig(Path(tmp) / "rig.bin", {**rig, "view_deltas": {"a.png": np.zeros((3, 3))}}, ["a.png"])

    def test_an_empty_view_deltas_is_a_v2_file(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rig.bin"
            body_rig.write_body_rig(path, {**rig, "view_deltas": {}}, ["a.png"])
            self.assertEqual(path.read_bytes()[:8], b"B2CRIG2\0")
            self.assertNotIn("deltas", body_rig.read_body_rig(path))


class TestBuildFaceRigStep(unittest.TestCase):
    def _inputs(self):
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=1)
        names = [f"f{i}.png" for i in range(6)]
        # the fit covers frames 1 and 2 and moves the WHOLE body 1 cm in +x; the "face" is the top 20 vertices
        # (the head), so the face-only weights are what keeps the rest still
        motion = np.zeros(100, np.float32); motion[80:] = 1.0
        moved = verts.copy(); moved[:, 0] += 0.01
        fit = {"names": ["f1.png", "f2.png"], "verts_world": np.stack([moved, moved]).astype(np.float32),
               "verts0_world": verts.astype(np.float32), "expression_motion": motion}
        return {"body_rig": rig, "head_fit_views": fit, "image_names": names}, verts

    def test_deltas_land_on_the_fitted_frames_and_hold_beside_them(self):
        cls = get_step_class("build_face_rig"); step = cls()
        inputs, verts = self._inputs()
        with tempfile.TemporaryDirectory() as tmp:
            out = step.run(inputs, cls.resolve_params({"hold": 1, "face_motion_cm": 0.5, "face_fade_cm": 5.0, "debug_dir": tmp}))
            self.assertTrue((Path(tmp) / "face_rig.json").exists())
        vd = out["body_rig"]["view_deltas"]
        self.assertEqual(sorted(vd), ["f0.png", "f1.png", "f2.png", "f3.png"])   # 1, 2 fitted; 0 and 3 held
        stats = out["face_rig_stats"]
        self.assertEqual((stats["frames_fit"], stats["frames_hold"], stats["frames_zero"]), (2, 2, 2))
        # the head vertices move the full centimetre, the body not at all, the 5 cm fade in between (vertices
        # are 2.5 cm apart: the one below the head at half weight, the next at zero)
        d = vd["f1.png"]
        np.testing.assert_allclose(d[80:, 0], 0.01, atol=1e-6)
        np.testing.assert_allclose(d[:78], 0.0)
        np.testing.assert_allclose(d[79, 0], 0.005, atol=1e-4)
        # every rig key is kept
        for k in ("verts", "vertex_index", "joints", "weights", "parents", "joint_positions", "active"):
            self.assertIn(k, out["body_rig"])

    def test_whole_head_when_face_motion_is_zero(self):
        cls = get_step_class("build_face_rig"); step = cls()
        inputs, _ = self._inputs()
        out = step.run(inputs, cls.resolve_params({"face_motion_cm": 0.0}))
        self.assertEqual(out["face_rig_stats"]["face_only"]["faded"], 0)
        np.testing.assert_allclose(out["body_rig"]["view_deltas"]["f1.png"][:, 0], 0.01, atol=1e-6)

    def test_refusals(self):
        cls = get_step_class("build_face_rig"); step = cls(); params = cls.resolve_params({})
        inputs, verts = self._inputs()
        bad = dict(inputs); bad["head_fit_views"] = {**inputs["head_fit_views"], "names": ["nope.png", "f2.png"]}
        with self.assertRaises(ValueError):
            step.run(bad, params)
        bad = dict(inputs); bad["head_fit_views"] = {**inputs["head_fit_views"], "verts0_world": inputs["head_fit_views"]["verts0_world"] + 0.1}
        with self.assertRaises(ValueError):
            step.run(bad, params)   # the rig is not this body
        bad = dict(inputs); bad["body_rig"] = {k: v for k, v in inputs["body_rig"].items() if k != "vertex_index"}
        with self.assertRaises(ValueError):
            step.run(bad, params)


class TestBrushWritesV3(unittest.TestCase):
    def _run(self, help_text, **overrides):
        step_class = get_step_class("brush"); step = step_class()
        seen = {}

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            seen["cmd"] = list(cmd)
            rig = Path(colmap_dir) / "body_rig.bin"
            seen["magic"] = rig.read_bytes()[:8]
            seen["rig"] = body_rig.read_body_rig(rig)
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        brush_mod._HELP_PROBE["brush"] = help_text
        verts, joints, parents, sv, sj, sw = _skeleton()
        rig = body_rig.build_body_rig(verts, joints, parents, sv, sj, sw, min_subtree=5, vertex_stride=1)
        rig["view_deltas"] = {"frame_00002_.png": np.full((100, 3), 0.01, np.float32)}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0, "brush_path": "brush", **overrides})
                step.run({**_inputs(), "body_rig": rig}, params)
        finally:
            brush_mod._HELP_PROBE.pop("brush", None)
        return seen

    def test_a_v3_trainer_gets_the_deltas(self):
        seen = self._run("--align-iters --body-rig B2CRIG3")
        self.assertEqual(seen["magic"], b"B2CRIG3\0")
        self.assertEqual(seen["rig"]["names"], ["frame_00001_.png", "frame_00002_.png"])
        np.testing.assert_array_equal(seen["rig"]["deltas"][0], 0)
        np.testing.assert_allclose(seen["rig"]["deltas"][1], 0.01)
        self.assertIn("--body-rig", seen["cmd"])

    def test_an_older_trainer_gets_a_v2_file(self):
        seen = self._run("--align-iters --body-rig")
        self.assertEqual(seen["magic"], b"B2CRIG2\0")
        self.assertNotIn("deltas", seen["rig"])
        self.assertIn("--body-rig", seen["cmd"])


class TestGeometry(unittest.TestCase):
    def test_opencv_camera_projects_like_body2colmap(self):
        from body2colmap import Camera
        cam = Camera(focal_length=(800.0, 800.0), image_size=(640, 480), principal_point=(320.0, 240.0),
                     position=np.array([0.3, -0.2, 2.0], np.float32), rotation=np.eye(3, dtype=np.float32))
        cam.look_at(np.zeros(3, np.float32)) if hasattr(cam, "look_at") else None
        pts = np.random.default_rng(0).normal(size=(20, 3)) * 0.3
        R, t, K = fv.opencv_camera(cam)
        uv, z = fv.project(pts, R, t, K)
        np.testing.assert_allclose(uv, cam.project(pts.astype(np.float32)), atol=1e-2)
        self.assertTrue((z > 0).all())

    def test_head_crop_affine_puts_the_head_upright_and_inverts(self):
        import cv2
        face = np.array([[100, 100], [140, 100], [140, 160], [100, 160]], np.float64)
        up = np.array([[120, 160], [160, 120]], np.float64)   # head-up axis pointing up-right: 45 deg roll
        M, side, roll = fv.head_crop_affine(face, up, 256, padding=1.8)
        self.assertAlmostEqual(side, 1.8 * 60)
        self.assertAlmostEqual(abs(roll), 45.0)
        centre = np.array([120.0, 130.0])
        c = M[:, :2] @ centre + M[:, 2]
        np.testing.assert_allclose(c, [128, 128], atol=1e-6)
        # the up axis maps to straight up in the crop
        a, b = (M[:, :2] @ up.T).T + M[:, 2]
        self.assertAlmostEqual(b[0] - a[0], 0.0, places=6)
        self.assertLess(b[1], a[1])
        Minv = cv2.invertAffineTransform(M)
        np.testing.assert_allclose(Minv[:, :2] @ c + Minv[:, 2], centre, atol=1e-6)

    def test_point_in_polygon(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
        inside = fv.point_in_polygon(np.array([[0.5, 0.5], [1.5, 0.5], [0.5, -0.1], [0.99, 0.99]]), square)
        self.assertEqual(list(inside), [True, False, False, True])
        # a concave contour: the notch is outside
        notch = np.array([[0, 0], [2, 0], [2, 2], [1, 0.5], [0, 2]], np.float64)
        self.assertEqual(list(fv.point_in_polygon(np.array([[1.0, 1.5], [0.2, 1.0], [1.8, 1.0]]), notch)), [False, True, True])

    def test_inside_contour_cylinder_ignores_the_normal_direction(self):
        ring = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)])
        pts = np.array([[0.2, 0.1, 5.0], [0.2, 0.1, -3.0], [1.5, 0.0, 0.0]])
        self.assertEqual(list(fv.inside_contour_cylinder(pts, ring)), [True, True, False])

    def test_kabsch_recovers_a_rigid_motion(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(12, 3))
        ang = 0.7
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        t = np.array([0.1, -0.2, 0.3])
        Rk, tk = fv.kabsch(A, A @ R.T + t)
        np.testing.assert_allclose(Rk, R, atol=1e-9)
        np.testing.assert_allclose(tk, t, atol=1e-9)

    def test_feature_landmarks_drop_the_oval_the_irises_and_the_unmapped(self):
        vol = np.arange(478); vol[5] = -1
        mapped = vol >= 0
        idx, verts = fv.feature_landmarks({"vertex_of_landmark": vol, "mapped": mapped})
        self.assertNotIn(5, idx)
        self.assertNotIn(10, idx)       # face oval
        self.assertNotIn(468, idx)      # iris
        self.assertIn(33, idx)          # an eye corner
        np.testing.assert_array_equal(verts, vol[idx])

    def test_anchor_pnp_recovers_a_camera(self):
        rng = np.random.default_rng(2)
        obj = rng.normal(size=(60, 3)) * 0.1
        R_true = fv.kabsch(rng.normal(size=(4, 3)), rng.normal(size=(4, 3)))[0]
        t_true = np.array([0.05, -0.02, 2.0])
        K = np.array([[1500.0, 0, 384.0], [0, 1500.0, 768.0], [0, 0, 1]])
        img, _ = fv.project(obj, R_true, t_true, K)
        R, t, rms = fv.anchor_camera_pnp(obj, img, K)
        self.assertLess(rms, 1e-3)
        np.testing.assert_allclose(R, R_true, atol=1e-6)
        np.testing.assert_allclose(t, t_true, atol=1e-6)


class TestStepsRegistered(unittest.TestCase):
    def test_defaults_resolve(self):
        for name in ("detect_face_views", "fit_head_per_view", "paste_eyes", "build_face_rig"):
            cls = get_step_class(name)
            params = cls.resolve_params({})
            self.assertIsInstance(params, dict)
        self.assertEqual(get_step_class("build_face_rig").resolve_params({})["face_motion_cm"], 0.5)
        self.assertEqual(get_step_class("fit_head_per_view").resolve_params({})["pose_prior"], 50.0)

    def test_the_rodrigues_matches_cv2(self):
        import cv2
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not in this env (the fit runs in the sam3dbody env)")
        rv = np.array([[0.3, -0.2, 0.5], [0.0, 0.0, 0.0], [1e-9, 0, 0]])
        R = fv.rodrigues_torch(torch.tensor(rv, dtype=torch.float64)).numpy()
        for i in range(3):
            np.testing.assert_allclose(R[i], cv2.Rodrigues(rv[i])[0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
