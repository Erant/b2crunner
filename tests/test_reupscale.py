"""Re-upscale (steps/reupscale.py): camera scaling and interpolation, the keep step, and the wiring."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.reupscale import ReupscaleKeepStep, interpolate_cameras, scale_cameras
from pipeline.workflow import WorkflowSpec, when_truthy

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / "pipeline" / "workflows").glob("helical*.yaml"))


def _orbit(n=6):
    from body2colmap.camera import Camera

    cams = []
    for i in range(n):
        a = np.radians(10.0 * i)
        pos = np.array([2 * np.sin(a), 0.1 * i, 2 * np.cos(a)], np.float32)
        z = pos / np.linalg.norm(pos)                       # back axis (OpenGL), the camera looks down -z at the origin
        x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        cams.append(Camera(focal_length=(1658.0, 1658.0), image_size=(1080, 1920), principal_point=(540.0, 960.0),
                           position=pos, rotation=np.stack([x, y, z], 1).astype(np.float32)))
    return cams


class TestCameras(unittest.TestCase):
    def test_scaling_is_intrinsics_only(self):
        cams = _orbit()
        out = scale_cameras(cams, 720, 1280)
        for a, b in zip(cams, out):
            self.assertEqual((b.width, b.height), (720, 1280))
            self.assertAlmostEqual(b.fx, a.fx * 720 / 1080, places=3)
            self.assertAlmostEqual(b.cy, a.cy * 1280 / 1920, places=3)
            np.testing.assert_allclose(b.position, a.position)
            np.testing.assert_allclose(b.rotation, a.rotation)

    def test_interpolation_keeps_the_originals_every_nth(self):
        cams = _orbit()
        out = interpolate_cameras(cams, 4)
        self.assertEqual(len(out), (len(cams) - 1) * 5 + 1)
        for i, c in enumerate(cams):
            np.testing.assert_allclose(out[i * 5].position, c.position, atol=1e-6)
            np.testing.assert_allclose(out[i * 5].rotation, c.rotation, atol=1e-6)
        mid = out[2]   # halfway between cameras 0 and 1 (t = 0.4)
        R = np.asarray(mid.rotation, np.float64)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        expect = 0.6 * np.asarray(cams[0].position) + 0.4 * np.asarray(cams[1].position)
        np.testing.assert_allclose(mid.position, expect, atol=1e-6)
        # the view direction turns monotonically between its neighbours
        fwd = [-np.asarray(c.rotation)[:, 2] for c in out[:6]]
        ang = [np.degrees(np.arccos(np.clip(np.dot(fwd[0], f), -1, 1))) for f in fwd]
        self.assertTrue(all(b > a for a, b in zip(ang, ang[1:])))
        self.assertAlmostEqual(ang[5], 10.0, delta=0.5)

    def test_no_inbetween_is_the_identity(self):
        cams = _orbit()
        self.assertEqual(len(interpolate_cameras(cams, 0)), len(cams))


class TestKeep(unittest.TestCase):
    def test_keeps_the_training_cameras(self):
        cams = interpolate_cameras(_orbit(), 4)
        images = [np.full((2, 2, 3), i, np.uint8) for i in range(len(cams))]
        out = ReupscaleKeepStep().run({"images": images, "cameras": cams, "keep_every": 5}, {})
        self.assertEqual([int(im[0, 0, 0]) for im in out["images"]], list(range(0, len(cams), 5)))
        self.assertEqual(len(out["cameras"]), 6)

    def test_refuses_a_count_that_is_not_a_densified_path(self):
        cams = _orbit()
        with self.assertRaises(ValueError):
            ReupscaleKeepStep().run({"images": [np.zeros((2, 2, 3), np.uint8)] * 6, "cameras": cams, "keep_every": 4}, {})


class TestWiring(unittest.TestCase):
    def test_one_render_and_one_upscale_per_low_vram_setting(self):
        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            for low_vram in (True, False):
                with self.subTest(workflow=path.name, low_vram=low_vram):
                    g = {"export_ply": True, "reupscale": True, "low_vram": low_vram}
                    from pipeline.templating import resolve
                    live = [s.id for s in spec.steps if s.id.startswith("reupscale")
                            and when_truthy(resolve(s.when, {"globals": g}))]
                    render = "reupscale_render" if low_vram else "reupscale_render_batched"
                    sr = "reupscale_sr" if low_vram else "reupscale_sr_batched"
                    self.assertEqual(live, [render, sr, "reupscale_keep", "reupscale_train"])
                    batch = by_id[sr].params["batch_size"]
                    self.assertEqual(by_id[render].params["inbetween"] + 1, batch if not low_vram else 1)

    def test_the_retraining_replaces_the_deliverable_and_keeps_the_first(self):
        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            ids = [s.id for s in spec.steps]
            with self.subTest(workflow=path.name):
                final, train = by_id["train_final_splat"], by_id["reupscale_train"]
                self.assertLess(ids.index("train_final_splat"), ids.index("reupscale_render"))
                self.assertLess(ids.index("reupscale_keep"), ids.index("reupscale_train"))
                self.assertLess(ids.index("reupscale_train"), ids.index("export_subject"))
                self.assertEqual(train.params["export_dir"], final.params["export_dir"])
                self.assertEqual(train.params["export_name"], final.params["export_name"])
                self.assertEqual(train.outputs["splat_path"], "dataset.splat_path")
                self.assertIn("final_before_reupscale.ply", by_id["reupscale_render"].params["keep_copy"])
                # SeedVR2 upscales each frame (or batch) on its own: the loop and the rig stay on
                self.assertEqual(train.params["align_iters"], 4)
                self.assertEqual(train.inputs["body_rig"], "scene.body_rig?")
                self.assertNotIn("normal_maps", train.inputs)
                setting = next(p for p in spec.settings if p.name == "reupscale")
                self.assertIs(setting.default, False)


class TestShDegree(unittest.TestCase):
    def _render(self, params):
        from unittest import mock
        from pipeline.steps.reupscale import ReupscaleInputsStep

        seen = {}

        def fake(**kw):
            seen.update(kw)
            n = len(kw["cameras"])
            return [np.zeros((4, 4, 3), np.uint8)] * n, [np.ones((4, 4), np.float32)] * n

        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "s.ply"
            ply.write_bytes(b"")
            with mock.patch("pipeline.steps.splat._rasterize", fake), \
                    mock.patch("body2colmap.splat_scene.SplatScene.from_ply", lambda path: None):
                ReupscaleInputsStep().run({"splat_path": str(ply), "cameras": _orbit()},
                                          ReupscaleInputsStep.resolve_params(params))
        return seen["sh_degree"]

    def test_defaults_to_two(self):
        self.assertEqual(self._render({}), 2)

    def test_reaches_the_rasteriser(self):
        self.assertEqual(self._render({"sh_degree": 0}), 0)

    def test_every_instance_renders_at_two(self):
        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                if step.step == "reupscale_inputs":
                    with self.subTest(workflow=path.name, step=step.id):
                        self.assertEqual(step.params["sh_degree"], 2)


if __name__ == "__main__":
    unittest.main()
