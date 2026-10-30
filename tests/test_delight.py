"""delight_references (steps/delight.py): the headlight fit, the lift, and Rig splat's De-light wiring."""
import unittest
from pathlib import Path

import numpy as np

from pipeline.steps.delight import _log_lum, delight
from pipeline.workflow import WorkflowSpec
from tests.helpers import run_step

WORKFLOW = Path(__file__).resolve().parent.parent / "pipeline" / "workflows" / "helical.yaml"
GATE = ["${globals.rig_splat}", "${globals.delight}"]


def _headlit(slope=1.0, size=96):
    """A grey disc on black, lit from the lens: log2 lum falls by `slope` per unit of n_z."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    inside = r < 0.95
    nz = np.sqrt(np.clip(1 - r ** 2, 0, 1))
    normals = np.zeros((size, size, 3), np.float32)
    normals[..., 2] = nz
    labels = inside.astype(np.int32)
    lin = 0.4 * np.exp2(slope * (nz - 1.0))
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    image = np.repeat((srgb * 255 + 0.5).astype(np.uint8)[..., None], 3, -1) * inside[..., None]
    return image.astype(np.uint8), labels, normals


class TestDelight(unittest.TestCase):
    def test_a_headlit_disc_comes_out_flatter_and_its_slope_is_found(self):
        image, labels, normals = _headlit(slope=1.0)
        out, fits = delight(image, labels, normals, max_lift=1.5, blur_sigma=0.5, min_px=100)
        self.assertAlmostEqual(fits[1], 1.0, delta=0.15)
        ring = (labels == 1) & (normals[..., 2] > 0.3)
        self.assertLess(np.std(_log_lum(out)[ring]), 0.5 * np.std(_log_lum(image)[ring]))
        # The background (label 0) is untouched.
        np.testing.assert_array_equal(out[labels == 0], image[labels == 0])

    def test_a_negative_slope_is_clamped_to_no_change(self):
        image, labels, normals = _headlit(slope=-0.5)
        out, fits = delight(image, labels, normals, max_lift=1.5, blur_sigma=0.5, min_px=100)
        self.assertEqual(fits[1], 0.0)
        self.assertLessEqual(np.abs(out.astype(int) - image.astype(int)).max(), 1)

    def test_a_single_photo_has_no_back_panel_to_delight(self):
        image, labels, normals = _headlit()
        out = run_step("delight_references", {
            "reference_image": None, "reference_labels": None, "reference_normals": None,
            "anchor_image": image, "anchor_labels": labels, "anchor_normals": normals,
        }, {"min_px": 100})
        self.assertIsNone(out["reference_image"])
        self.assertEqual(out["anchor_image"].shape, image.shape)
        self.assertEqual(set(out["slopes"]), {"anchor"})

    def test_maps_of_another_size_are_a_wiring_bug(self):
        image, labels, normals = _headlit()
        with self.assertRaises(ValueError):
            run_step("delight_references", {
                "anchor_image": image, "anchor_labels": labels[:10], "anchor_normals": normals,
            })


class TestDelightWiring(unittest.TestCase):
    def setUp(self):
        self.spec = WorkflowSpec.from_yaml(WORKFLOW)
        self.spec.validate()
        self.ids = [s.id for s in self.spec.steps]

    def test_de_light_is_off_by_default_behind_rig_splat(self):
        setting = next(p for p in self.spec.settings if p.name == "delight")
        self.assertIs(setting.default, False)
        self.assertEqual(setting.requires, "rig_splat")
        self.assertFalse(setting.advanced)

    def test_every_delight_step_runs_only_with_rig_splat_and_de_light(self):
        steps = [s for s in self.spec.steps if s.id.startswith("delight_")]
        self.assertEqual(len(steps), 5)
        for step in steps:
            self.assertEqual(step.when, GATE, step.id)

    def test_pass_one_reads_the_delit_images(self):
        self.assertLess(self.ids.index("delight_references"), self.ids.index("dump_denoise_input"))
        self.assertLess(self.ids.index("matte_reference"), self.ids.index("delight_reference_labels"))
        step = self.spec.steps[self.ids.index("delight_references")]
        self.assertEqual(step.outputs["reference_image"], "dataset.reference_image")
        self.assertEqual(step.outputs["anchor_image"], "dataset.anchor_image")


if __name__ == "__main__":
    unittest.main()
