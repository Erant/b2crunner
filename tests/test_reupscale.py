"""Re-upscale (steps/reupscale.py): camera scaling, the views and their sidecars, and the wiring."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.reupscale import ReupscaleInputsStep, scale_cameras
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



def _run_inputs(inputs, params, seen=None):
    """ReupscaleInputsStep.run with the rasteriser faked: view i renders as value i, alpha 0.25."""
    from unittest import mock

    def fake(**kw):
        if seen is not None:
            seen.update(kw)
        n = len(kw["cameras"])
        return ([np.full((4, 4, 3), i, np.uint8) for i in range(n)],
                [np.full((4, 4), 0.25, np.float32) for _ in range(n)])

    with tempfile.TemporaryDirectory() as tmp:
        ply = Path(tmp) / "s.ply"
        ply.write_bytes(b"")
        with mock.patch("pipeline.steps.splat._rasterize", fake), \
                mock.patch("body2colmap.splat_scene.SplatScene.from_ply", lambda path: None):
            return ReupscaleInputsStep().run({"splat_path": str(ply), **inputs},
                                             ReupscaleInputsStep.resolve_params(params))


class TestViews(unittest.TestCase):
    def _inputs(self, n=6, labels=True, masks=True):
        return {"cameras": _orbit(n),
                "masks": [np.ones((8, 8), np.float32)] * n if masks else None,
                "labels": [np.full((8, 8), 3, np.uint8)] * n if labels else None}

    def test_one_view_per_training_camera_with_its_matte_and_labels(self):
        out = _run_inputs(self._inputs(), {"target_width": 4, "target_height": 4})
        for key in ("images", "cameras", "masks", "labels"):
            self.assertEqual(len(out[key]), 6, key)
        for mask, label in zip(out["masks"], out["labels"]):
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(mask.shape, (4, 4))
            self.assertEqual(int(mask.max()), 255)
            self.assertEqual(int(label[0, 0]), 3)

    def test_without_mattes_the_render_alpha_is_the_matte(self):
        out = _run_inputs(self._inputs(labels=False, masks=False), {"target_width": 4, "target_height": 4})
        self.assertEqual([int(m.max()) for m in out["masks"]], [64] * 6)     # the render's alpha, 0.25
        self.assertIsNone(out["labels"])

    def test_refuses_mattes_that_do_not_match_the_cameras(self):
        inputs = self._inputs()
        inputs["masks"] = inputs["masks"][:-1]
        with self.assertRaises(ValueError):
            _run_inputs(inputs, {})


class TestWiring(unittest.TestCase):
    def test_one_render_and_one_upscale_at_batch_one(self):
        from pipeline.templating import resolve

        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            for low_vram in (True, False):
                with self.subTest(workflow=path.name, low_vram=low_vram):
                    g = {"export_ply": True, "reupscale": True, "low_vram": low_vram}
                    live = [s.id for s in spec.steps if s.id.startswith("reupscale")
                            and when_truthy(resolve(s.when, {"globals": g}))]
                    self.assertEqual(live, ["reupscale_render", "reupscale_sr", "reupscale_train"])
                    self.assertEqual(by_id["reupscale_sr"].params["batch_size"], 1)
                    self.assertEqual(by_id["reupscale_train"].inputs["image_names"], "dataset.image_names")

    def test_the_retraining_replaces_the_deliverable_and_keeps_the_first(self):
        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            ids = [s.id for s in spec.steps]
            with self.subTest(workflow=path.name):
                final, train = by_id["train_final_splat"], by_id["reupscale_train"]
                self.assertLess(ids.index("train_final_splat"), ids.index("reupscale_render"))
                self.assertLess(ids.index("reupscale_sr"), ids.index("reupscale_train"))
                self.assertLess(ids.index("reupscale_train"), ids.index("export_subject"))
                self.assertEqual(train.params["export_dir"], final.params["export_dir"])
                self.assertEqual(train.params["export_name"], final.params["export_name"])
                self.assertEqual(train.outputs["splat_path"], "dataset.splat_path")
                self.assertIn("final_before_reupscale.ply", by_id["reupscale_render"].params["keep_copy"])
                # SeedVR2 upscales each frame on its own: the loop and the rig stay on
                self.assertEqual(train.params["align_iters"], 4)
                self.assertEqual(train.inputs["body_rig"], "scene.body_rig?")
                self.assertNotIn("normal_maps", train.inputs)
                setting = next(p for p in spec.settings if p.name == "reupscale")
                self.assertIs(setting.default, False)


class TestShDegree(unittest.TestCase):
    def _render(self, params):
        seen = {}
        cams = _orbit()
        _run_inputs({"cameras": cams}, params, seen)
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
