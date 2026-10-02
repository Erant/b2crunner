"""relight_frames (steps/relight.py): the per-frame lighting fit, the per-pixel correction and the wiring."""
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.relight import apply_correction, correction_map, fit_frame_lighting, sh9
from pipeline.workflow import WorkflowSpec

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / "pipeline" / "workflows").glob("helical*.yaml"))


def _synthetic(n_pts=3000, n_cams=24, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n_pts, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rho = rng.uniform(0.1, 0.6, size=(n_pts, 3))
    az = np.linspace(0, 2 * np.pi, n_cams, endpoint=False)
    views = np.stack([np.cos(az), np.zeros_like(az), np.sin(az)], 1)
    # each frame: a gain, a colour cast and a directional term, all mild
    beta = np.zeros((n_cams, 9, 3))
    beta[:, 0, :] = rng.uniform(0.8, 1.25, size=(n_cams, 1)) * rng.uniform(0.95, 1.05, size=(n_cams, 3))
    beta[:, 1:4, :] = rng.normal(scale=0.05, size=(n_cams, 3, 3))
    point, cam = [], []
    for c in range(n_cams):
        seen = np.flatnonzero(normals @ views[c] > 0.2)
        point.append(seen)
        cam.append(np.full(len(seen), c))
    point, cam = np.concatenate(point), np.concatenate(cam)
    k = np.einsum("ek,ekr->er", sh9(normals)[point], beta[cam])
    colour = rho[point] * k * (1 + noise * rng.normal(size=k.shape))
    return point, cam, colour, normals, beta, n_cams


class TestFit(unittest.TestCase):
    def test_recovers_each_frames_departure_from_the_common_lighting(self):
        point, cam, colour, normals, beta, n_cams = _synthetic()
        fitted, stats = fit_frame_lighting(point, cam, colour, normals, n_cams)
        self.assertEqual(stats["frames_fitted"], n_cams)
        self.assertGreater(stats["explained"], 0.97)
        # Only the departure is identifiable: compare frame to frame ratios at the same normals.
        X = sh9(normals)
        k_true = np.einsum("pk,ckr->cpr", X, beta)
        k_fit = np.einsum("pk,ckr->cpr", X, fitted)
        r_true = k_true / k_true.mean(0, keepdims=True)
        r_fit = k_fit / k_fit.mean(0, keepdims=True)
        self.assertLess(np.median(np.abs(r_true - r_fit)), 0.02)

    def test_noise_does_not_break_it(self):
        point, cam, colour, normals, beta, n_cams = _synthetic(noise=0.1, seed=1)
        fitted, stats = fit_frame_lighting(point, cam, colour, normals, n_cams)
        gains_true = beta[:, 0, :].mean(1) / beta[:, 0, :].mean()
        gains_fit = fitted[:, 0, :].mean(1) / fitted[:, 0, :].mean()
        self.assertGreater(np.corrcoef(gains_true, gains_fit)[0, 1], 0.95)

    def test_a_frame_with_too_few_samples_keeps_k_1(self):
        point, cam, colour, normals, beta, n_cams = _synthetic()
        drop = cam == 3
        fitted, stats = fit_frame_lighting(point[~drop], cam[~drop], colour[~drop], normals, n_cams)
        self.assertEqual(stats["frames_fitted"], n_cams - 1)
        np.testing.assert_allclose(fitted[3, 0], 1.0)
        np.testing.assert_allclose(fitted[3, 1:], 0.0)


class TestCorrection(unittest.TestCase):
    def test_k_1_is_the_identity_and_alpha_is_untouched(self):
        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, size=(16, 20, 4), dtype=np.uint8)
        out = apply_correction(image, np.ones((16, 20, 3)))
        np.testing.assert_array_equal(out, image)

    def test_divides_in_linear_light(self):
        image = np.full((2, 2, 3), 188, np.uint8)            # sRGB 188 ~ linear 0.5
        out = apply_correction(image, np.full((2, 2, 3), 2.0))
        self.assertTrue(np.all(np.abs(out.astype(int) - 137) <= 1))   # linear 0.25 ~ sRGB 137

    def test_the_correction_fades_to_1_off_the_splat(self):
        beta = np.zeros((9, 3))
        beta[0] = 1.5
        normal = np.full((4, 4, 3), 128, np.uint8)
        normal[..., 2] = 255
        alpha = np.zeros((4, 4))
        alpha[:2] = 1.0
        k = correction_map(normal, alpha, beta, kmin=0.5, kmax=2.0)
        np.testing.assert_allclose(k[:2], 1.5)
        np.testing.assert_allclose(k[2:], 1.0)


class TestWiring(unittest.TestCase):
    def test_the_frames_are_corrected_before_the_bundle_and_the_final_training(self):
        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            ids = [s.id for s in spec.steps]
            by_id = {s.id: s for s in spec.steps}
            with self.subTest(workflow=path.name):
                pre, fix = by_id["relight_pretrain"], by_id["relight_frames"]
                self.assertLess(ids.index("refine_cameras_final"), ids.index("relight_pretrain"))
                self.assertLess(ids.index("segment_views"), ids.index("relight_frames"))
                self.assertLess(ids.index("relight_pretrain"), ids.index("relight_frames"))
                self.assertLess(ids.index("relight_frames"), ids.index("export_colmap"))
                self.assertLess(ids.index("relight_frames"), ids.index("train_final_splat"))
                self.assertEqual(fix.outputs.get("images"), "dataset.images")
                self.assertEqual(fix.inputs.get("splat_path"), pre.outputs.get("splat_path"))
                self.assertEqual(fix.inputs.get("labels"), "scene.seg_labels?")
                for step in (pre, fix):
                    self.assertIn({"eq": ["${globals.lighting_correction}", "prepass"]}, step.when)
                setting = next(p for p in spec.settings if p.name == "lighting_correction")
                self.assertEqual(setting.default, "prepass")


if __name__ == "__main__":
    unittest.main()
