"""The face branch: a crop, a splat built through it, and the composite.

Three things are new against tests/test_pointmap_splat.py, and the first is
the one everything else rests on:

  * **A crop must not move a Gaussian.** `face_pointmap_splat` re-expresses
    SAM-3D-Body's camera on the crop's pixel grid so that unprojecting a
    crop pixel lands on the ray through the full-image pixel it was cut
    from. Build the same scene twice — once whole, once cropped — and the
    Gaussians in the shared region have to come out at the same world
    coordinates. Get this wrong and the face floats off the skull, which no
    later step can detect or repair.
  * The crop bookkeeping itself (padding, aspect, clamping at an edge).
  * The composite's CULL: which frames are close enough to the photograph's
    view to take the splat layer at all, and what that angle is measured
    about. The blend itself is body2colmap's again as of 2026-08-31 (see
    docs/revert-when-body2colmap-drops-gsplat.md), so what is tested here
    is only the part `render` still owns.

Same discipline as the shell's tests — synthetic geometry with a known
answer, and the 1B forward pass stubbed out.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline.steps  # noqa: F401  — registers the face branch
from pipeline.registry import get_step_class
from pipeline.step import Param, ParamError, with_defaults
from pipeline.steps import pointmap_splat as ps
from pipeline.steps import render
from pipeline.steps import sapiens2
from tests.test_pointmap_splat import (
    CX, CY, FOCAL, HEIGHT, WIDTH, _StubbedStep, _read_ply, _sphere,
)


class _StubbedFaceStep(ps.FacePointmapSplatStep):
    """The face step with the pointmap head replaced by a known map."""

    def __init__(self, pointmap: np.ndarray) -> None:
        super().__init__()
        self._stub = pointmap

    def load(self, params):  # noqa: D102 — no model to load
        self._model = object()
        self._checkpoint = params["checkpoint"]

    def _pointmap(self, image_bgr, params):  # noqa: D102
        return self._stub


def _scene():
    """The sphere from the shell's tests, plus the mesh and photo around it."""
    z, normals, mask = _sphere()
    points = ps.backproject(z, FOCAL, CX, CY)
    cam_t = np.array([0.02, -0.03, 0.05])
    mesh_output = {
        "vertices": points[mask][::7] - cam_t,
        "cam_t": cam_t,
        "focal_length": FOCAL,
    }
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[..., 2] = 255
    return z, normals, mask, image, mesh_output


def _means(result) -> np.ndarray:
    _, _, data = _read_ply(Path(result["splat_path"]))
    return data[:, 0:3].astype(np.float64)


# ---------------------------------------------------------------------------
class TestCropIsANoOpOnTheRays(unittest.TestCase):
    """The gate. Everything else in the face branch assumes this holds."""

    BOX = (70, 90, 246, 324)      # a window well inside 320x440

    def _build(self, cropped: bool, tmp: str):
        z, normals, mask, image, mesh_output = _scene()
        pointmap = ps.backproject(z, FOCAL, CX, CY).astype(np.float32)
        inputs = {
            "image": image,
            "mask": mask.astype(np.float32),
            "normal_map": normals * ps.NORMAL_TO_CAMERA_FRAME,
            "mesh_output": mesh_output,
        }
        overrides = {
            "filepath": str(Path(tmp) / ("face.ply" if cropped else "full.ply")),
            # Off: the two runs must differ ONLY by the crop, and a scale
            # fitted over image bins is fitted over different bins.
            "align_depth": False,
            "cliff_k": 0.0,
            "fill_max_frac": 0.0,
        }
        if not cropped:
            step = _StubbedStep(pointmap)
            params = ps.PointmapSplatStep.resolve_params(
                dict(overrides, splat_scale=0.5))
            return step.run(inputs, params)

        x0, y0, x1, y1 = self.BOX
        cropped_inputs = {
            "image": np.ascontiguousarray(image[y0:y1, x0:x1]),
            "mask": np.ascontiguousarray(mask[y0:y1, x0:x1]).astype(np.float32),
            "normal_map": np.ascontiguousarray(
                (normals * ps.NORMAL_TO_CAMERA_FRAME)[y0:y1, x0:x1]),
            "mesh_output": mesh_output,
            "crop_info": {"box": self.BOX, "full_size": (WIDTH, HEIGHT),
                          "crop_size": (x1 - x0, y1 - y0)},
        }
        step = _StubbedFaceStep(np.ascontiguousarray(pointmap[y0:y1, x0:x1]))
        params = ps.FacePointmapSplatStep.resolve_params(overrides)
        return step.run(cropped_inputs, params)

    def test_the_same_surface_lands_in_the_same_place_either_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            whole = _means(self._build(False, tmp))
            crop = _means(self._build(True, tmp))

        self.assertGreater(len(crop), 1000, "the crop should still hold a face")
        self.assertLess(len(crop), len(whole), "and less than the whole frame")

        # Every Gaussian the crop produced must coincide with one the full
        # frame produced. Match on the two lateral axes, which is enough to
        # identify the source pixel, then compare in full.
        lookup = {(round(x, 6), round(y, 6)): (x, y, z) for x, y, z in whole}
        matched = 0
        worst = 0.0
        for x, y, z in crop:
            key = (round(x, 6), round(y, 6))
            if key not in lookup:
                continue
            matched += 1
            worst = max(worst, abs(z - lookup[key][2]))
        self.assertGreater(matched, 0.98 * len(crop),
                           "crop pixels landed on rays the full frame never used")
        self.assertLess(worst, 1e-5, "same ray, different depth")

    def test_a_crop_pixel_reprojects_onto_its_full_image_pixel(self):
        """The same gate stated the other way round, and the one that would
        catch a sign error in the principal-point shift."""
        with tempfile.TemporaryDirectory() as tmp:
            crop = _means(self._build(True, tmp))

        cam = crop * ps.FLIP                       # world -> camera frame
        u = FOCAL * cam[:, 0] / cam[:, 2] + CX     # the FULL image's pinhole
        v = FOCAL * cam[:, 1] / cam[:, 2] + CY

        self.assertLess(float(np.abs(u - np.round(u)).max()), 1e-3)
        self.assertLess(float(np.abs(v - np.round(v)).max()), 1e-3)
        x0, y0, x1, y1 = self.BOX
        self.assertTrue(((u >= x0 - 1) & (u <= x1) & (v >= y0 - 1) & (v <= y1)).all(),
                        "a Gaussian reprojected outside the box it was cut from")


class TestFaceIntrinsics(unittest.TestCase):
    def _intrinsics(self, box, crop_size, focal=FOCAL):
        step = ps.FacePointmapSplatStep()
        inputs = {"mesh_output": {"focal_length": focal},
                  "crop_info": {"box": box, "full_size": (WIDTH, HEIGHT),
                                "crop_size": crop_size}}
        return step._source_intrinsics(inputs, {}, crop_size[0], crop_size[1])

    def test_a_native_resolution_crop_only_shifts_the_principal_point(self):
        f, cx, cy = self._intrinsics((70, 90, 246, 324), (176, 234))
        self.assertAlmostEqual(f, FOCAL)
        self.assertAlmostEqual(cx, CX - 70)
        self.assertAlmostEqual(cy, CY - 90)

    def test_a_uniformly_resized_crop_scales_the_focal_too(self):
        """Half-resolution: the same rays, half as many pixels across them."""
        f, cx, cy = self._intrinsics((70, 90, 246, 324), (88, 117))
        self.assertAlmostEqual(f, FOCAL / 2.0)
        self.assertAlmostEqual(cx, (CX - 70) / 2.0)
        self.assertAlmostEqual(cy, (CY - 90) / 2.0)

    def test_a_non_uniform_resize_is_refused(self):
        """One camera carries one focal length; averaging the two ratios
        would silently shear the face."""
        with self.assertRaises(ValueError) as caught:
            self._intrinsics((70, 90, 246, 324), (176, 117))
        self.assertIn("non-uniform", str(caught.exception))

    def test_the_body_step_is_the_full_frame(self):
        step = ps.BodyPointmapSplatStep()
        f, cx, cy = step._source_intrinsics(
            {"mesh_output": {"focal_length": FOCAL}}, {}, WIDTH, HEIGHT)
        self.assertEqual((f, cx, cy), (FOCAL, WIDTH / 2.0, HEIGHT / 2.0))


class TestSpecializationDefaults(unittest.TestCase):
    def test_the_three_measured_defaults_differ_and_nothing_else_does(self):
        body = ps.BodyPointmapSplatStep.declared_params()
        face = ps.FacePointmapSplatStep.declared_params()
        self.assertEqual(sorted(body), sorted(face))
        differing = {name for name in body
                     if body[name].default != face[name].default}
        self.assertEqual(differing,
                         {"splat_scale", "fill_max_frac", "align_bin_px"})
        self.assertEqual(face["splat_scale"].default, 0.5)
        self.assertEqual(face["fill_max_frac"].default, 0.05)
        # 8, not the base's 32. A 32 px bin on a face crop reports the mesh's
        # front surface too near — the chin overhangs the throat — and
        # depth_scale_to_mesh pulls the splat 15 mm toward the camera by it.
        # Measured against a rasterised z-buffer; see the step's docstring.
        self.assertEqual(face["align_bin_px"].default, 8)
        self.assertEqual(body["align_bin_px"].default, 32)

    def test_with_defaults_refuses_a_name_the_base_does_not_declare(self):
        base = (Param("count", int, 6), Param("scale", float, 0.5))
        self.assertEqual(with_defaults(base, count=9)[0].default, 9)
        with self.assertRaises(ParamError):
            with_defaults(base, cont=9)


# ---------------------------------------------------------------------------
class TestCropToBox(unittest.TestCase):
    def _crop(self, box, image=None, **params):
        step = get_step_class("crop_to_box")()
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8) if image is None else image
        return step.run({"image": image, "box": box},
                        step.resolve_params(params))

    def test_crop_info_describes_the_cut_that_was_actually_made(self):
        result = self._crop((140, 180, 200, 260))
        x0, y0, x1, y1 = result["crop_info"]["box"]
        self.assertEqual(result["image"].shape[:2], (y1 - y0, x1 - x0))
        self.assertEqual(result["crop_info"]["full_size"], (WIDTH, HEIGHT))
        self.assertEqual(result["crop_info"]["crop_size"], (x1 - x0, y1 - y0))

    def test_padding_grows_the_box_and_aspect_squares_it_up(self):
        result = self._crop((140, 180, 200, 260), padding=0.5, aspect=0.75)
        x0, y0, x1, y1 = result["crop_info"]["box"]
        self.assertLess(x0, 140)
        self.assertGreater(y1, 260)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 0.75, delta=0.02)

    def test_a_box_against_an_edge_slides_in_rather_than_shrinking(self):
        """A head at the top of a portrait frame must not silently lose its
        margin — the crop slides down, keeping the size it asked for. A 60px
        box padded by 1.0 of its half-side is 120px, and all 120 survive."""
        result = self._crop((150, 0, 210, 60), padding=1.0, aspect=0.0)
        x0, y0, x1, y1 = result["crop_info"]["box"]
        self.assertEqual(y0, 0)
        self.assertEqual(y1 - y0, 120, "the padded height should survive")
        self.assertTrue(0 <= x0 < x1 <= WIDTH)

    def test_a_crop_never_leaves_the_frame(self):
        result = self._crop((280, 400, 320, 440), padding=2.0)
        x0, y0, x1, y1 = result["crop_info"]["box"]
        self.assertTrue(0 <= x0 < x1 <= WIDTH)
        self.assertTrue(0 <= y0 < y1 <= HEIGHT)

    def test_a_box_too_small_to_be_a_face_is_refused(self):
        with self.assertRaises(ValueError):
            self._crop((150, 200, 156, 206), padding=0.0, min_size=64)

    def test_a_degenerate_box_is_refused(self):
        with self.assertRaises(ValueError):
            self._crop((150, 200, 150, 260))


class TestSelectedBox(unittest.TestCase):
    """The locate pass's box. Its one non-obvious property is the component
    filter — without it a handful of stray face-classified pixels anywhere
    else in the frame stretches the box across the image, and the crop hands
    back exactly the resolution it was cut to gain."""

    def test_the_box_is_tight_around_the_selection(self):
        hard = np.zeros((HEIGHT, WIDTH), bool)
        hard[100:140, 60:90] = True
        self.assertEqual(sapiens2._selected_box(hard, WIDTH, HEIGHT),
                         (60, 100, 90, 140))

    def test_a_stray_blob_elsewhere_does_not_stretch_the_box(self):
        hard = np.zeros((HEIGHT, WIDTH), bool)
        hard[100:140, 60:90] = True      # the face
        hard[400:403, 300:303] = True    # something misclassified, far away
        self.assertEqual(sapiens2._selected_box(hard, WIDTH, HEIGHT),
                         (60, 100, 90, 140))

    def test_an_empty_selection_is_refused(self):
        with self.assertRaises(ValueError):
            sapiens2._selected_box(np.zeros((HEIGHT, WIDTH), bool), WIDTH, HEIGHT)


class TestParseParts(unittest.TestCase):
    def test_the_default_preset_is_masktests_face_neck(self):
        self.assertEqual(sapiens2.parse_parts("face"), {3})

    def test_head_adds_hair_glasses_and_lips(self):
        self.assertEqual(sapiens2.parse_parts("head"), {2, 3, 4, 24, 25})

    def test_teeth_and_tongue_are_never_foreground(self):
        for preset in ("all", "body"):
            self.assertFalse(sapiens2.parse_parts(preset) & {26, 27, 28})

    def test_an_explicit_list_wins(self):
        self.assertEqual(sapiens2.parse_parts(" 2, 3 ,4 "), {2, 3, 4})

    def test_nonsense_is_refused_by_name(self):
        with self.assertRaises(ValueError) as caught:
            sapiens2.parse_parts("faec")
        self.assertIn("faec", str(caught.exception))


# ---------------------------------------------------------------------------
class _Camera:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)


class TestSplatLayerCull(unittest.TestCase):
    """Which frames get the splat overlay, and what it is measured about.

    The blend and the refusals are gone from this project: the compositing
    is `Renderer.render_composite(splat_layer=...)`'s now, and the
    black-background requirement went with it, because
    `steps/splat.py`'s `render_splat_layers` passes the background itself.
    What stays on this side is the cull — `render` builds its own cameras
    and never touches `OrbitPipeline`, so `splat_view_angle_deg` is not
    reachable from it. See docs/revert-when-body2colmap-drops-gsplat.md.
    """

    CENTRE = np.array([0.0, 0.0, -2.0])   # the "head", 2 m down -Z
    SOURCE = np.zeros(3)                  # where the photograph was taken

    def _cameras(self, angles_deg):
        """Cameras on a circle about CENTRE, `angles_deg` off the source view."""
        out = []
        for angle in angles_deg:
            rad = np.radians(angle)
            radius = 2.0
            out.append(_Camera((radius * np.sin(rad),
                                0.0,
                                self.CENTRE[2] + radius * np.cos(rad))))
        return out

    def _angles(self, angles_deg, center=None):
        return render._splat_view_angles_deg(
            self._cameras(angles_deg),
            self.CENTRE if center is None else np.asarray(center, dtype=np.float64),
            self.SOURCE,
        )

    def test_the_angle_is_the_camera_s_own_offset_from_the_source_view(self):
        measured = self._angles([0.0, 30.0, 60.0, 120.0])
        np.testing.assert_allclose(measured, [0.0, 30.0, 60.0, 120.0], atol=1e-6)

    def test_the_default_band_runs_to_sixty_degrees(self):
        """Pinned because it is a deliberate widening of body2colmap's
        measured 45: out past it the shell is mostly open rim, and what
        makes that affordable is that these frames are inputs to two
        denoise passes that can rewrite a rim. A revert that adopted the
        render mode without passing this would inherit body2colmap's own
        45; the shipped workflows pass 60 explicitly for the same reason.
        select_support_views, whose frames nothing rewrites, bands at 30 out
        of its cap's radius and does not follow this number."""
        default = get_step_class("render")().resolve_params(
            {"n_frames": 1})["splat_max_angle_deg"]
        self.assertEqual(default, 60.0)
        measured = self._angles([50.0, 70.0])
        self.assertLessEqual(measured[0], default)
        self.assertGreater(measured[1], default)

    def test_the_pivot_is_the_splat_not_the_orbit_target(self):
        """A camera aimed at a body's centre is not aimed at the head, so
        measuring the view angle about the wrong pivot culls the wrong
        frames. The head here is 1 m above the orbit target, and this camera
        sits exactly on the line from the source view through the target:
        0.0 degrees off about the target, 13.8 about the head."""
        camera = [_Camera((0.0, -0.32918, -0.65836))]
        orbit_target = np.array([0.0, -1.0, -2.0])

        about_target = render._splat_view_angles_deg(camera, orbit_target, self.SOURCE)
        self.assertAlmostEqual(about_target[0], 0.0, places=3)

        about_splat = render._splat_view_angles_deg(camera, self.CENTRE, self.SOURCE)
        self.assertGreater(about_splat[0], 10.0)

    def test_a_source_view_on_the_splat_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            render._splat_view_angles_deg(self._cameras([0.0]), self.CENTRE,
                                          self.CENTRE)
        self.assertIn("source view direction", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
