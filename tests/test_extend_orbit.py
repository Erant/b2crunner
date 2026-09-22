"""The orbit extension's three steps (pipeline/steps/extend_orbit.py).

The path is checked against cyber_6f's real recorded metadata: rebuilt
from render_subject's own helix params, the extended path has to contain
the dataset's cameras verbatim in the middle, continue at the same angular
step either side (41 frames ahead and 40 after: an 81-frame pass less a
40- and a 41-frame overlap, each pass's mask changing on a latent edge),
and carry whatever rigid motion the source path was carried by.
The two control videos and the splice are checked on small synthetic
frames.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from pipeline.steps.extend_orbit import (
    extended_helix_params, helix_step_deg, latent_aligned, new_frames,
)
from pipeline.steps.splat import _resolve_cameras, _transform_camera
from tests.helpers import require_stage, run_step

import pipeline.steps  # noqa: F401

# render_subject's helix in helical.yaml, which is what the step's own
# defaults are (tests/test_workflows.py pins the two blocks agree).
HELIX = dict(n_frames=81, n_loops=2, amplitude_deg=30.0, lead_in_deg=30.0, lead_out_deg=90.0)


def _spherical(camera, target):
    from body2colmap import coordinates

    return coordinates.cartesian_to_spherical(
        np.asarray(camera.position, dtype=np.float64) - np.asarray(target, dtype=np.float64))


def _small_motion():
    import cv2

    rotation, _ = cv2.Rodrigues(np.radians(np.array([0.4, 1.2, -0.2])))
    return rotation.astype(np.float64), np.array([0.041, -0.034, 0.021])


class TestTheHelixArithmetic(unittest.TestCase):
    def test_the_step_is_total_over_n_not_n_minus_one(self):
        # OrbitPath.helical puts frame i at i / n_frames of the total.
        self.assertAlmostEqual(helix_step_deg(81, 2, 30.0, 90.0), 840.0 / 81)

    def test_a_pass_adds_its_length_less_the_overlap(self):
        self.assertEqual(new_frames(81, 40), 41)
        self.assertEqual(new_frames(81, 41), 40)
        with self.assertRaisesRegex(ValueError, "not 4k\\+1"):
            new_frames(80, 40)
        with self.assertRaisesRegex(ValueError, "one frame to keep and one to paint"):
            new_frames(81, 81)
        with self.assertRaisesRegex(ValueError, "one frame to keep and one to paint"):
            new_frames(81, 0)

    def test_a_mask_boundary_sits_on_a_latent_edge_at_4k_plus_1(self):
        """Frame 0 is a latent of its own, then every 4 frames: the edges
        fall after frames 1, 5, 9, ... — so BEFORE's boundary (81 - 40 =
        41) and AFTER's (41) are aligned, and 40 is not."""
        self.assertTrue(latent_aligned(41))
        self.assertTrue(latent_aligned(1))
        self.assertTrue(latent_aligned(45))
        self.assertFalse(latent_aligned(40))
        self.assertFalse(latent_aligned(42))

    def test_the_extension_grows_the_leads_by_whole_steps(self):
        params = extended_helix_params(HELIX, 41, 40)
        step = 840.0 / 81
        self.assertEqual(params["n_frames"], 162)
        self.assertEqual(params["n_loops"], 2)
        self.assertAlmostEqual(params["lead_in_deg"], 30.0 + 41 * step)
        self.assertAlmostEqual(params["lead_out_deg"], 90.0 + 40 * step)
        # ...which keeps the per-frame step exactly what it was.
        self.assertAlmostEqual(
            helix_step_deg(params["n_frames"], params["n_loops"],
                           params["lead_in_deg"], params["lead_out_deg"]), step)


class TestExtendHelicalPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("initial"))
        params = get_step_class("render_splat").resolve_params(
            dict(HELIX, pattern="helical", override_cam_from_mesh=True))
        cls.source, _, _, cls.anchor = _resolve_cameras(
            scene=None, dataset=cls.ds, params=params, width=720, height=1280)

    def _extend(self, cameras, **params):
        return run_step("extend_helical_path", {"cameras": cameras, "extras": self.ds.extras}, params)

    def test_the_source_cameras_are_the_middle_verbatim(self):
        out = self._extend(self.source)
        self.assertEqual((out["before"], out["after"]), (41, 40))
        self.assertEqual((out["overlap_before"], out["overlap_after"]), (40, 41))
        self.assertEqual(len(out["cameras"]), 162)
        self.assertEqual(len(out["image_names"]), 162)
        self.assertEqual(out["image_names"][0], "frame_00001_.png")
        self.assertEqual(out["image_names"][-1], "frame_00162_.png")
        for got, want in zip(out["cameras"][41:122], self.source):
            self.assertIs(got, want)

    def test_the_extension_continues_at_the_helix_s_own_step_on_the_lead_elevations(self):
        out = self._extend(self.source)
        target = self.ds.extras["orbit_target"]
        radius, azimuth, elevation = zip(*(_spherical(c, target) for c in out["cameras"]))
        self.assertLess(max(radius) - min(radius), 1e-5)
        steps = [(azimuth[i + 1] - azimuth[i] + 180.0) % 360.0 - 180.0 for i in range(161)]
        for step in steps:
            self.assertAlmostEqual(step, 840.0 / 81, places=4)
        # Flat at the source's first elevation ahead, at its last after —
        # the lead-in and lead-out rule continued.
        for value in elevation[:42]:
            self.assertAlmostEqual(value, elevation[41], places=4)
        for value in elevation[121:]:
            self.assertAlmostEqual(value, elevation[121], places=4)
        self.assertLess(elevation[41], elevation[121])

    def test_the_intrinsics_are_the_source_s(self):
        out = self._extend(self.source)
        first = self.source[0]
        for camera in out["cameras"]:
            self.assertEqual((camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height),
                             (first.fx, first.fy, first.cx, first.cy, first.width, first.height))

    def test_a_carried_source_path_carries_its_extension(self):
        """render_subject moves the whole path rigidly by the anchor's
        refinement; the new frames have to hang on the moved path, not the
        solver's, or they would sit off the frames by the same delta."""
        rotation, translation = _small_motion()
        moved = [_transform_camera(c, rotation, translation) for c in self.source]
        plain = self._extend(self.source)["cameras"]
        carried = self._extend(moved)["cameras"]
        for got, want in zip(carried[41:122], moved):
            self.assertIs(got, want)
        for before, after in list(zip(plain, carried))[:41] + list(zip(plain, carried))[122:]:
            np.testing.assert_allclose(after.position, rotation @ before.position + translation, atol=1e-5)
            np.testing.assert_allclose(after.rotation, rotation @ before.rotation, atol=1e-5)

    def test_another_overlap_or_pass_length(self):
        out = self._extend(self.source, overlap_before=44, overlap_after=45)
        self.assertEqual((out["before"], out["after"]), (37, 36))
        self.assertEqual(len(out["cameras"]), 154)
        for got, want in zip(out["cameras"][37:118], self.source):
            self.assertIs(got, want)
        out = self._extend(self.source, phase_frames=49, overlap_before=40, overlap_after=41)
        self.assertEqual((out["before"], out["after"]), (9, 8))
        with self.assertRaisesRegex(ValueError, "not 4k\\+1"):
            self._extend(self.source, phase_frames=80)
        with self.assertRaisesRegex(ValueError, "more frames than pass 2"):
            self._extend(self.source, phase_frames=101, overlap_before=90)

    def test_a_misaligned_overlap_runs_with_a_warning(self):
        with self.assertLogs("pipeline.steps.extend_orbit", level="WARNING") as logs:
            out = self._extend(self.source, overlap_after=40)
        self.assertEqual((out["before"], out["after"]), (41, 41))
        self.assertEqual(len(logs.output), 1)
        self.assertIn("AFTER pass's mask changes at frame 40", logs.output[0])
        with self.assertNoLogs("pipeline.steps.extend_orbit", level="WARNING"):
            self._extend(self.source)

    def test_it_refuses_a_path_that_is_not_the_helix_it_was_told(self):
        with self.assertRaisesRegex(ValueError, "not render_subject's helix"):
            self._extend(self.source, n_loops=1)
        with self.assertRaisesRegex(ValueError, "n_frames says"):
            self._extend(self.source[:-1])
        # A per-frame perturbation is not a rigid carry either.
        jittered = list(self.source)
        jittered[10] = _transform_camera(self.source[10], np.eye(3), np.array([0.0, 0.01, 0.0]))
        with self.assertRaisesRegex(ValueError, "not render_subject's helix"):
            self._extend(jittered)


def _frame(value: int, h: int = 8, w: int = 6) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _dataset(frames):
    from body2colmap.camera import Camera

    return Dataset(
        images=frames, image_names=[f"src_{i}.png" for i in range(len(frames))],
        cameras=[Camera(focal_length=(5.0, 5.0), image_size=(6, 8)) for _ in frames],
        points_3d=(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)),
        resolution=(6, 8), masks=[np.ones((8, 6), np.float32) for _ in frames],
    )


_COUNTS = {"before": 2, "overlap_before": 3, "after": 1, "overlap_after": 4}


class TestAssembleExtension(unittest.TestCase):
    """Five source frames; two new ahead over pass 2's first three, one new
    after over pass 2's last four: a 5-frame pass each way, the overlaps
    one apart as the shipped defaults are."""

    N = 5

    def _assemble(self, guide_alpha=None, debug_dir=None, **params):
        from body2colmap.camera import Camera

        source = [_frame(10 + i) for i in range(self.N)]
        total = _COUNTS["before"] + self.N + _COUNTS["after"]
        inputs = {
            "dataset": _dataset(source),
            "cameras": [Camera(focal_length=(5.0, 5.0), image_size=(6, 8)) for _ in range(total)],
            "image_names": [f"frame_{i + 1:05d}_.png" for i in range(total)],
            **_COUNTS,
        }
        if guide_alpha is not None:
            inputs["guide_images"] = [_frame(100 + i) for i in range(total)]
            inputs["guide_masks"] = [np.full((8, 6), guide_alpha, dtype=np.float32)] * total
        if debug_dir is not None:
            params["debug_dir"] = debug_dir
        return source, run_step("assemble_extension", inputs, params)

    def _flat(self, frames):
        return [int(f[0, 0, 0]) for f in frames]

    def test_unguided_the_new_frames_are_grey_and_the_overlap_is_pass_2s(self):
        source, out = self._assemble()
        # BEFORE: two new (grey, reactive) then pass 2's first three (inactive).
        self.assertEqual(self._flat(out["before_images"]), [127, 127, 10, 11, 12])
        self.assertEqual([float(m[0, 0]) for m in out["before_masks"]], [1, 1, 0, 0, 0])
        for got, want in zip(out["before_images"][2:], source[:3]):
            self.assertIs(got, want)
        # AFTER: pass 2's last four (inactive) then one new.
        self.assertEqual(self._flat(out["after_images"]), [11, 12, 13, 14, 127])
        self.assertEqual([float(m[0, 0]) for m in out["after_masks"]], [0, 0, 0, 0, 1])
        for got, want in zip(out["after_images"][:4], source[1:]):
            self.assertIs(got, want)
        for mask in out["before_masks"] + out["after_masks"]:
            self.assertEqual(mask.dtype, np.float32)
            self.assertEqual(mask.shape, (8, 6))

    def test_guided_every_frame_is_the_matted_render_over_grey(self):
        """The default `inactive_source: guide`: the whole pass is the
        render, cut from the extended path — frame i of the video is guide
        frame i — and the flags are unchanged."""
        _, out = self._assemble(guide_alpha=1.0, guide="retrained")
        self.assertEqual(self._flat(out["before_images"]), [100, 101, 102, 103, 104])
        self.assertEqual(self._flat(out["after_images"]), [103, 104, 105, 106, 107])
        self.assertEqual([float(m[0, 0]) for m in out["before_masks"]], [1, 1, 0, 0, 0])
        self.assertEqual([float(m[0, 0]) for m in out["after_masks"]], [0, 0, 0, 0, 1])
        # Half-transparent render: half way to the grey, to rounding.
        _, out = self._assemble(guide_alpha=0.5, guide="intermediate")
        self.assertTrue(abs(int(out["before_images"][0][0, 0, 0]) - (100 + 127) // 2) <= 1)

    def test_the_hybrid_keeps_pass_2s_frames_inactive_beside_the_render(self):
        source, out = self._assemble(guide_alpha=1.0, guide="retrained", inactive_source="frames")
        self.assertEqual(self._flat(out["before_images"]), [100, 101, 10, 11, 12])
        self.assertEqual(self._flat(out["after_images"]), [11, 12, 13, 14, 107])
        for got, want in zip(out["before_images"][2:], source[:3]):
            self.assertIs(got, want)

    def test_inactive_source_is_moot_without_a_guide(self):
        _, out = self._assemble(inactive_source="guide")
        self.assertEqual(self._flat(out["before_images"]), [127, 127, 10, 11, 12])

    def test_a_guide_that_was_expected_and_did_not_arrive_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no guide_images"):
            self._assemble(guide="retrained")

    def test_the_batch_has_to_be_the_one_the_path_was_built_for(self):
        with self.assertRaisesRegex(ValueError, "different batch"):
            run_step("assemble_extension", {
                "dataset": _dataset([_frame(1)] * 4), "cameras": [object()] * 8,
                "image_names": [""] * 8, **_COUNTS,
            }, {})

    def test_passes_of_two_lengths_are_refused(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            run_step("assemble_extension", {
                "dataset": _dataset([_frame(1)] * 5), "cameras": [object()] * 8,
                "image_names": [""] * 8, **dict(_COUNTS, overlap_after=3),
            }, {})

    def test_the_control_videos_are_dumped_as_datasets(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._assemble(debug_dir=tmp)
            for name in ("before", "after"):
                with self.subTest(pass_=name):
                    saved = Dataset.from_disk(f"{tmp}/{name}")
                    self.assertEqual(len(saved.images), 5)
                    self.assertEqual(len(saved.cameras), 5)
            saved = Dataset.from_disk(f"{tmp}/before")
            # The flag rides in the alpha: reactive 1, inactive 0.
            flags = [float(np.asarray(m, dtype=np.float32).max() > 0.5) for m in saved.masks]
            self.assertEqual(flags, [1, 1, 0, 0, 0])
            self.assertEqual(saved.image_names, [f"frame_{i + 1:05d}_.png" for i in range(5)])


class TestSpliceExtension(unittest.TestCase):
    N = 5

    def _splice(self, before_len=5, after_len=5, **extra):
        source = [_frame(10 + i) for i in range(self.N)]
        total = _COUNTS["before"] + self.N + _COUNTS["after"]
        before_out = [_frame(50 + i) for i in range(before_len)]
        after_out = [_frame(70 + i) for i in range(after_len)]
        inputs = {
            "dataset": _dataset(source),
            "before_denoised": before_out, "after_denoised": after_out,
            "cameras": [object()] * total,
            "image_names": [f"frame_{i + 1:05d}_.png" for i in range(total)],
            **_COUNTS, **extra,
        }
        return source, before_out, after_out, run_step("splice_extension", inputs, {})

    def test_the_new_frames_flank_pass_2s_and_the_overlap_returns_are_dropped(self):
        source, before_out, after_out, out = self._splice(anchor_frame_index=1)
        self.assertEqual([int(f[0, 0, 0]) for f in out["images"]],
                         [50, 51, 10, 11, 12, 13, 14, 74])
        for got, want in zip(out["images"][2:7], source):
            self.assertIs(got, want)
        self.assertIs(out["images"][0], before_out[0])
        self.assertIs(out["images"][-1], after_out[-1])
        self.assertEqual(len(out["masks"]), 8)
        self.assertTrue(all((m == 1.0).all() and m.dtype == np.float32 for m in out["masks"]))
        self.assertEqual(len(out["cameras"]), 8)
        self.assertEqual(out["image_names"][-1], "frame_00008_.png")
        self.assertEqual(out["anchor_frame_index"], 3)

    def test_no_anchor_index_publishes_none(self):
        _, _, _, out = self._splice()
        self.assertNotIn("anchor_frame_index", out)

    def test_a_pass_of_the_wrong_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, "did not return the batch"):
            self._splice(before_len=4)
        with self.assertRaisesRegex(ValueError, "did not return the batch"):
            self._splice(after_len=6)


if __name__ == "__main__":
    unittest.main()
