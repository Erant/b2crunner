"""Re-upscale (steps/reupscale.py): the band of views, the renders and their mattes, and the wiring."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.reupscale import ReupscaleInputsStep, band_cameras
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
    def _band(self, n=81, lo=-70.0, hi=70.0):
        cams = _orbit()
        return cams, band_cameras(cams, n, lo, hi, 720, 1280)

    def _elevations(self, cams):
        return np.degrees(np.arcsin([c.position[1] / np.linalg.norm(c.position) for c in cams]))

    def test_count_size_and_lens(self):
        cams, band = self._band()
        self.assertEqual(len(band), 81)
        for c in band:
            self.assertEqual((c.width, c.height), (720, 1280))
            self.assertAlmostEqual(c.fx, cams[0].fx * 720 / 1080, places=3)
            self.assertAlmostEqual(c.cy, cams[0].cy * 1280 / 1920, places=3)

    def test_band_spans_minus_to_plus_seventy(self):
        cams, band = self._band()
        el = self._elevations(band)
        self.assertAlmostEqual(el.min(), -70.0, places=2)
        self.assertAlmostEqual(el.max(), 70.0, places=2)
        # every camera at the orbit's median radius, looking at its centre
        radius = np.median([np.linalg.norm(c.position) for c in cams])
        for c in band:
            self.assertAlmostEqual(float(np.linalg.norm(c.position)), radius, delta=0.02)
            np.testing.assert_allclose(c.get_forward_vector(), -c.position / np.linalg.norm(c.position), atol=1e-4)

    def test_evenly_spaced(self):
        _, band = self._band()
        d = np.array([c.position / np.linalg.norm(c.position) for c in band], np.float64)
        ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
        np.fill_diagonal(ang, 360)
        nearest = ang.min(1)
        self.assertLess(nearest.max() / nearest.min(), 1.6)     # no clumps, no gaps
        # consecutive views are neighbours on the sphere (the rig smooths between them)
        steps = [ang[i, i + 1] for i in range(len(band) - 1)]
        self.assertLess(np.median(steps), 1.2 * np.median(nearest))

    def test_more_views_at_the_equator_than_the_poles(self):
        _, band = self._band()
        el = np.round(self._elevations(band), 1)
        self.assertGreater(np.sum(el == 0.0), np.sum(el == 70.0))


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
    def test_one_view_per_training_camera_by_default_with_the_alpha_as_matte(self):
        out = _run_inputs({"cameras": _orbit(6)}, {"target_width": 4, "target_height": 4})
        for key in ("images", "cameras", "image_names", "masks"):
            self.assertEqual(len(out[key]), 6, key)
        self.assertEqual([int(m.max()) for m in out["masks"]], [64] * 6)     # the render's alpha, 0.25
        self.assertEqual(len(set(out["image_names"])), 6)

    def test_views_param_sets_the_count(self):
        out = _run_inputs({"cameras": _orbit(6)}, {"views": 20, "target_width": 4, "target_height": 4})
        self.assertEqual(len(out["images"]), 20)


class TestWiring(unittest.TestCase):
    def test_one_render_and_one_upscale_at_batch_one(self):
        from pipeline.templating import resolve

        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            by_id = {s.id: s for s in spec.steps}
            for low_vram in (True, False):
                with self.subTest(workflow=path.name, low_vram=low_vram):
                    g = {"export_ply": True, "reupscale": True, "low_vram": low_vram, "splat_labels": True}
                    live = [s.id for s in spec.steps if s.id.startswith("reupscale")
                            and when_truthy(resolve(s.when, {"globals": g}))]
                    self.assertEqual(live, ["reupscale_render", "reupscale_sr", "reupscale_segment", "reupscale_train"])
                    self.assertEqual(by_id["reupscale_sr"].params["batch_size"], 1)
                    self.assertEqual(by_id["reupscale_train"].inputs["image_names"], "scene.reupscale.image_names")
                    render = by_id["reupscale_render"].params
                    self.assertEqual((render["min_elevation_deg"], render["max_elevation_deg"]), (-70, 70))

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


class TestLabelsOnTheLastSplat(unittest.TestCase):
    def _live(self, step, **g):
        from pipeline.templating import resolve

        s = next(x for x in self.spec.steps if x.id == step)
        base = {"export_ply": True, "splat_labels": True, "reupscale": False, "lighting_correction": "prepass"}
        base.update(g)
        return when_truthy(resolve(s.when, {"globals": base})), resolve(s.params.get("defer_labels"), {"globals": base})

    def test_the_first_training_defers_its_vote_when_retrained(self):
        for path in WORKFLOWS:
            self.spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                self.assertIs(self._live("train_final_splat", reupscale=True)[1], True)
                self.assertIs(self._live("train_final_splat", reupscale=False)[1], False)

    def test_early_segmentation_only_where_something_reads_it(self):
        for path in WORKFLOWS:
            self.spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                self._early()

    def _early(self):
        self.assertTrue(self._live("segment_views")[0])
        self.assertTrue(self._live("segment_views", reupscale=True)[0])                  # relight's fit
        self.assertFalse(self._live("segment_views", reupscale=True, lighting_correction="off")[0])
        self.assertFalse(self._live("segment_views", splat_labels=False)[0])
        self.assertTrue(self._live("reupscale_segment", reupscale=True)[0])
        self.assertFalse(self._live("reupscale_segment", reupscale=True, splat_labels=False)[0])


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
