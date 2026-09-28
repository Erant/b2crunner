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
    extended_helix_params, helix_step_deg, latent_aligned, new_frames, pass_elevation_offsets,
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


class TestTheTilt(unittest.TestCase):
    def test_each_pass_is_one_ramp_through_its_inactive_frames(self):
        before, after = pass_elevation_offsets(41, 40, 41, 40, 10.0)
        self.assertEqual((len(before), len(after)), (81, 81))
        # The far ends at the tilt, 0 on pass 2's first / last camera.
        self.assertAlmostEqual(before[0], -10.0)
        self.assertAlmostEqual(before[41], 0.0)
        self.assertAlmostEqual(after[40], 0.0)
        self.assertAlmostEqual(after[-1], 10.0)
        # One constant rate end to end, no bend at the mask boundary.
        for ramp, rate in ((before, 10.0 / 41), (after, 10.0 / 40)):
            for a, b in zip(ramp, ramp[1:]):
                self.assertAlmostEqual(b - a, rate)

    def test_no_tilt_is_the_flat_leads(self):
        before, after = pass_elevation_offsets(41, 40, 41, 40, 0.0)
        self.assertEqual(set(before) | set(after), {0.0})


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

    def test_the_tilt_ramps_the_new_frames_out_of_the_band(self):
        out = self._extend(self.source, tilt_deg=10.0)
        flat = self._extend(self.source)
        self.assertEqual(out["tilt_deg"], 10.0)
        self.assertEqual(flat["tilt_deg"], 0.0)
        for got, want in zip(out["cameras"][41:122], self.source):
            self.assertIs(got, want)
        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)
        tilted = [_spherical(c, target) for c in out["cameras"]]
        level = [_spherical(c, target) for c in flat["cameras"]]
        offsets = [-10.0 * (41 - i) / 41 for i in range(41)] + [0.0] * 81 \
            + [10.0 * (k + 1) / 40 for k in range(40)]
        for (r, az, el), (r0, az0, el0), d in zip(tilted, level, offsets):
            self.assertAlmostEqual(r, r0, places=5)
            self.assertAlmostEqual((az - az0 + 180.0) % 360.0 - 180.0, 0.0, places=4)
            self.assertAlmostEqual(el, el0 + d, places=4)
        # The passes' own paths: their new frames are the dataset's, their
        # inactive frames continue the same ramp at pass 2's azimuths.
        passes = out["pass_cameras"]
        self.assertEqual(len(passes), 162)
        for got, want in zip(passes[:41] + passes[81 + 41:], out["cameras"][:41] + out["cameras"][122:]):
            np.testing.assert_allclose(got.position, want.position, atol=1e-6)
        ramp = [_spherical(c, target) for c in passes]
        lead_in, lead_out = level[0][2], level[-1][2]
        for i in range(81):
            self.assertAlmostEqual(ramp[i][2], lead_in + 10.0 * (i - 41) / 41, places=4)
            self.assertAlmostEqual(ramp[81 + i][2], lead_out + 10.0 * (i - 40) / 40, places=4)
        for i in range(40):
            self.assertAlmostEqual((ramp[41 + i][1] - level[41 + i][1] + 180.0) % 360.0 - 180.0,
                                   0.0, places=4)
        for i in range(41):
            self.assertAlmostEqual((ramp[81 + i][1] - level[81 + i][1] + 180.0) % 360.0 - 180.0,
                                   0.0, places=4)
        # Still turned onto the target: the optical axis through it.
        for camera in out["cameras"][:41] + out["cameras"][122:]:
            to_target = target - np.asarray(camera.position, dtype=np.float64)
            forward = np.asarray(camera.rotation, dtype=np.float64) @ np.array([0.0, 0.0, -1.0])
            cos = forward @ to_target / np.linalg.norm(to_target) / np.linalg.norm(forward)
            self.assertGreater(cos, 1.0 - 1e-6)

    def test_at_ten_degrees_no_new_frame_repeats_a_view(self):
        """The reason for the tilt: flat, 32 of the 81 new frames sit within
        5 deg of a view the orbit already has (half pass 2's spacing)."""
        target = np.asarray(self.ds.extras["orbit_target"], dtype=np.float64)

        def repeats(cameras):
            dirs = np.array([np.asarray(c.position, np.float64) - target for c in cameras])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            kept, count = list(range(41, 122)), 0
            for i in list(range(40, -1, -1)) + list(range(122, 162)):
                nearest = np.degrees(np.arccos(np.clip(dirs[kept] @ dirs[i], -1.0, 1.0))).min()
                count += nearest < 5.0
                kept.append(i)
            return count

        self.assertEqual(repeats(self._extend(self.source)["cameras"]), 32)
        self.assertEqual(repeats(self._extend(self.source, tilt_deg=10.0)["cameras"]), 0)

    def test_a_tilt_past_the_pole_is_refused(self):
        with self.assertRaisesRegex(ValueError, "tilt_deg"):
            self._extend(self.source, tilt_deg=55.0)
        with self.assertRaisesRegex(ValueError, "tilt_deg"):
            self._extend(self.source, tilt_deg=-1.0)

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
        plain = self._extend(self.source, tilt_deg=10.0)["cameras"]
        carried = self._extend(moved, tilt_deg=10.0)["cameras"]
        for got, want in zip(carried[41:122], moved):
            self.assertIs(got, want)
        plain_passes = self._extend(self.source, tilt_deg=10.0)["pass_cameras"]
        carried_passes = self._extend(moved, tilt_deg=10.0)["pass_cameras"]
        for before, after in zip(plain_passes, carried_passes):
            np.testing.assert_allclose(after.position, rotation @ before.position + translation, atol=1e-5)
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
    """Five source frames; two new frames ahead of three inactive, four
    inactive before one new after — the overlaps one apart as the shipped
    defaults are: a 5-frame pass each way, rendered at the passes' own 10
    cameras."""

    N = 5
    PHASE = 5

    def _inputs(self, guide_alpha=1.0):
        from body2colmap.camera import Camera

        source = [_frame(10 + i) for i in range(self.N)]
        total = _COUNTS["before"] + self.N + _COUNTS["after"]
        return source, {
            "dataset": _dataset(source),
            "pass_cameras": [Camera(focal_length=(5.0, 5.0), image_size=(6, 8))
                             for _ in range(2 * self.PHASE)],
            "image_names": [f"frame_{i + 1:05d}_.png" for i in range(total)],
            "guide_images": [_frame(100 + i) for i in range(2 * self.PHASE)],
            "guide_masks": [np.full((8, 6), guide_alpha, dtype=np.float32)] * (2 * self.PHASE),
            **_COUNTS,
        }

    def _flat(self, frames):
        return [int(f[0, 0, 0]) for f in frames]

    def test_every_frame_is_the_matted_render_of_its_pass_over_grey(self):
        """The BEFORE pass is the render's first PHASE frames, the AFTER
        pass its last PHASE; pass 2's own frames are in neither."""
        _, inputs = self._inputs()
        out = run_step("assemble_extension", inputs, {})
        self.assertEqual(self._flat(out["before_images"]), [100, 101, 102, 103, 104])
        self.assertEqual(self._flat(out["after_images"]), [105, 106, 107, 108, 109])
        self.assertEqual([float(m[0, 0]) for m in out["before_masks"]], [1, 1, 0, 0, 0])
        self.assertEqual([float(m[0, 0]) for m in out["after_masks"]], [0, 0, 0, 0, 1])
        for mask in out["before_masks"] + out["after_masks"]:
            self.assertEqual(mask.dtype, np.float32)
            self.assertEqual(mask.shape, (8, 6))
        # Half-transparent render: half way to the grey, to rounding.
        _, inputs = self._inputs(guide_alpha=0.5)
        out = run_step("assemble_extension", inputs, {})
        self.assertTrue(abs(int(out["before_images"][0][0, 0, 0]) - (100 + 127) // 2) <= 1)

    def test_a_render_of_another_path_is_refused(self):
        _, inputs = self._inputs()
        inputs["guide_images"] = inputs["guide_images"][:-1]
        with self.assertRaisesRegex(ValueError, "guide render has 9 frames"):
            run_step("assemble_extension", inputs, {})

    def test_the_batch_has_to_be_the_one_the_path_was_built_for(self):
        _, inputs = self._inputs()
        inputs["dataset"] = _dataset([_frame(1)] * 4)
        with self.assertRaisesRegex(ValueError, "different batch"):
            run_step("assemble_extension", inputs, {})

    def test_passes_of_two_lengths_are_refused(self):
        _, inputs = self._inputs()
        inputs["overlap_after"] = 3
        with self.assertRaisesRegex(ValueError, "differ in length"):
            run_step("assemble_extension", inputs, {})

    def test_the_control_videos_are_dumped_as_datasets(self):
        import tempfile

        _, inputs = self._inputs()
        with tempfile.TemporaryDirectory() as tmp:
            run_step("assemble_extension", inputs, {"debug_dir": tmp})
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
            saved = Dataset.from_disk(f"{tmp}/after")
            self.assertEqual(saved.image_names, [f"frame_{i + 1:05d}_.png" for i in range(3, 8)])


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
        return source, before_out, after_out, run_step("splice_extension", inputs,
                                                       {"colour_match": False})

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


def _subject(seed: int, size: int = 48):
    """A textured square on 0.5 grey and its matte: colours spread enough in
    L and chroma that contrast and saturation are measurable."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 127, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.float32)
    lo, hi = size // 6, size - size // 6
    image[lo:hi, lo:hi] = rng.integers(40, 216, size=(hi - lo, hi - lo, 3), dtype=np.uint8)
    mask[lo:hi, lo:hi] = 1.0
    return image, mask


def _graded(image, mask, l_scale, chroma_scale):
    """`image` with its subject's L spread and chroma scaled, as a pass adds
    them — apply_colour_match's own transform, about the subject's mean."""
    from pipeline.steps.extend_orbit import _lab, apply_colour_match

    l_mean = float(_lab(image)[mask > 0.5][:, 0].mean())
    return apply_colour_match(
        image, {"l_mean": l_mean, "l_target": l_mean, "l_scale": l_scale,
                "chroma_scale": chroma_scale}, mask)


def _ratios(frames, guides, masks):
    from pipeline.steps.extend_orbit import _colour_stats

    (_, f_std, f_chroma), (_, g_std, g_chroma) = _colour_stats(frames, guides, masks, 1)
    return f_std / g_std, f_chroma / g_chroma


class TestColourMatch(unittest.TestCase):
    """The pass's own cast (L spread x1.25, chroma x1.2 over the guide, where
    pass 2's frames match it) is measured and taken back out."""

    def setUp(self):
        subjects = [_subject(seed) for seed in range(6)]
        self.guides = [image for image, _ in subjects]
        self.masks = [mask for _, mask in subjects]
        self.new = [_graded(g, m, 1.25, 1.2) for g, m in zip(self.guides[:3], self.masks[:3])]

    def test_the_cast_is_measured_and_undone(self):
        from pipeline.steps.extend_orbit import apply_colour_match, fit_colour_match

        match = fit_colour_match(self.new, self.guides[:3], self.masks[:3],
                                 self.guides[3:], self.guides[3:], self.masks[3:], erode_px=1)
        self.assertAlmostEqual(match["l_scale"], 0.8, delta=0.03)
        self.assertAlmostEqual(match["chroma_scale"], 1 / 1.2, delta=0.03)
        fixed = [apply_colour_match(f, match, m) for f, m in zip(self.new, self.masks[:3])]
        l_ratio, chroma_ratio = _ratios(fixed, self.guides[:3], self.masks[:3])
        self.assertAlmostEqual(l_ratio, 1.0, delta=0.03)
        self.assertAlmostEqual(chroma_ratio, 1.0, delta=0.03)

    def test_the_grey_outside_the_weight_is_untouched(self):
        from pipeline.steps.extend_orbit import apply_colour_match

        match = {"l_mean": 30.0, "l_target": 40.0, "l_scale": 0.8, "chroma_scale": 0.8}
        out = apply_colour_match(self.new[0], match, self.masks[0])
        outside = self.masks[0] < 0.5
        np.testing.assert_array_equal(out[outside], self.new[0][outside])

    def test_nothing_to_measure_is_none(self):
        from pipeline.steps.extend_orbit import fit_colour_match

        empty = [np.zeros_like(m) for m in self.masks]
        self.assertIsNone(fit_colour_match(self.new, self.guides[:3], empty[:3],
                                           self.guides[3:], self.guides[3:], empty[3:]))


class TestSpliceColourMatch(unittest.TestCase):
    """The splice with a guide: the new frames corrected, pass 2's kept."""

    def _run(self, params=None, with_guide=True):
        # 2 new before, 5 of pass 2, 1 new after (_COUNTS); guide = the clean
        # subject at every camera, pass 2's frames = the guide, the passes'
        # returns = the guide with the cast. The guide render is laid out by
        # pass (the BEFORE pass's 5 cameras, then the AFTER pass's), each
        # inactive camera standing in at the pass 2 view it sits beside.
        subjects = [_subject(seed, size=96) for seed in range(8)]
        guides = [image for image, _ in subjects]
        masks = [mask for _, mask in subjects]
        source = guides[2:7]
        cast = [_graded(g, m, 1.25, 1.2) for g, m in zip(guides, masks)]
        before_out = cast[0:2] + source[:3]
        after_out = source[1:] + cast[7:8]
        inputs = {
            "dataset": _dataset(source),
            "before_denoised": before_out, "after_denoised": after_out,
            "cameras": [object()] * 8,
            "image_names": [f"frame_{i + 1:05d}_.png" for i in range(8)],
            **_COUNTS,
        }
        if with_guide:
            inputs.update(guide_images=guides[0:5] + guides[3:8],
                          guide_masks=masks[0:5] + masks[3:8])
        return guides, masks, source, cast, run_step("splice_extension", inputs, params or {})

    def test_the_new_frames_are_brought_to_pass_2s_colour(self):
        guides, masks, source, cast, out = self._run({"colour_grow_px": 1})
        for got, want in zip(out["images"][2:7], source):
            self.assertIs(got, want)
        new = [0, 1, 7]
        l_ratio, chroma_ratio = _ratios([out["images"][i] for i in new],
                                        [guides[i] for i in new], [masks[i] for i in new])
        self.assertAlmostEqual(l_ratio, 1.0, delta=0.05)
        self.assertAlmostEqual(chroma_ratio, 1.0, delta=0.05)

    def test_off_the_frames_go_in_as_returned(self):
        _, _, _, cast, out = self._run({"colour_match": False}, with_guide=False)
        self.assertIs(out["images"][0], cast[0])
        self.assertIs(out["images"][-1], cast[7])

    def test_on_without_a_guide_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "needs the guide render"):
            self._run(with_guide=False)


if __name__ == "__main__":
    unittest.main()
