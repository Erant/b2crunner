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
  * The composite: what it blends, what it culls, and the two ways it is
    asked to refuse rather than produce something quietly wrong.

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


class TestCompositeSplatViews(unittest.TestCase):
    """The blend, the cull, and the two refusals."""

    CENTRE = (0.0, 0.0, -2.0)      # the "head", 2 m down -Z from the camera

    def _frames(self, n=3, value=40):
        return [np.full((8, 6, 3), value, np.uint8) for _ in range(n)]

    def _cameras(self, angles_deg):
        """Cameras on a circle about CENTRE, `angles_deg` off the source view.

        The source view is the world origin looking at CENTRE, i.e. along -Z.
        """
        out = []
        for angle in angles_deg:
            rad = np.radians(angle)
            radius = 2.0
            out.append(_Camera((radius * np.sin(rad),
                                0.0,
                                self.CENTRE[2] + radius * np.cos(rad))))
        return out

    def _run(self, angles, layers, alphas, **params):
        step = get_step_class("composite_splat_views")()
        return step.run(
            {
                "images": self._frames(len(angles)),
                "splat_images": layers,
                "splat_masks": alphas,
                "cameras": self._cameras(angles),
                "splat_center": self.CENTRE,
            },
            step.resolve_params(params),
        )

    def test_an_opaque_layer_replaces_and_a_transparent_one_does_not(self):
        angles = [0.0, 0.0]
        layers = [np.full((8, 6, 3), 200, np.uint8), np.zeros((8, 6, 3), np.uint8)]
        alphas = [np.ones((8, 6), np.float32), np.zeros((8, 6), np.float32)]
        result = self._run(angles, layers, alphas)

        np.testing.assert_array_equal(result["images"][0], layers[0])
        np.testing.assert_array_equal(result["images"][1][:, :, 0], 40)

    def test_half_alpha_blends_premultiplied_colour(self):
        """The composite is out = base*(1-a) + rgb, because a splat rendered
        on black is already premultiplied. 200 rendered at alpha 0.5 arrives
        as 100, over a base of 40: 40*0.5 + 100 = 120."""
        layers = [np.full((8, 6, 3), 100, np.uint8)]
        alphas = [np.full((8, 6), 0.5, np.float32)]
        result = self._run([0.0], layers, alphas)
        np.testing.assert_array_equal(result["images"][0], 120)

    def test_a_frame_past_the_cull_keeps_the_drawing(self):
        angles = [0.0, 30.0, 60.0]
        layers = [np.full((8, 6, 3), 200, np.uint8)] * 3
        alphas = [np.ones((8, 6), np.float32)] * 3
        result = self._run(angles, layers, alphas, max_angle_deg=45.0)

        roles = [role["role"] for role in result["view_roles"]]
        self.assertEqual(roles, ["composited", "composited", "base"])
        np.testing.assert_array_equal(result["images"][2][:, :, 0], 40)

    def test_the_default_band_runs_to_sixty_degrees(self):
        """Pinned because it is a deliberate widening of body2colmap's
        measured 45: out past it the shell is mostly open rim, and what
        makes that affordable is that these frames are inputs to two
        denoise passes that can rewrite a rim. select_support_views, whose
        frames nothing rewrites, culls at 30 out of its own param and does
        not follow this number."""
        angles = [50.0, 70.0]
        layers = [np.full((8, 6, 3), 200, np.uint8)] * 2
        alphas = [np.ones((8, 6), np.float32)] * 2
        result = self._run(angles, layers, alphas)

        roles = [role["role"] for role in result["view_roles"]]
        self.assertEqual(roles, ["composited", "base"])

    def test_the_pivot_is_the_splat_not_the_orbit_target(self):
        """A camera aimed at a body's centre is not aimed at the head, so
        measuring the view angle about the wrong pivot culls the wrong
        frames. The head here is 1 m above the orbit target, and this camera
        sits exactly on the line from the source view through the target:
        0.0 degrees off about the target, 13.8 about the head."""
        step = get_step_class("composite_splat_views")()
        common = {
            "images": self._frames(1),
            "splat_images": [np.full((8, 6, 3), 200, np.uint8)],
            "splat_masks": [np.ones((8, 6), np.float32)],
            "cameras": [_Camera((0.0, -0.32918, -0.65836))],
            "orbit_target": np.array([0.0, -1.0, -2.0]),
        }
        params = step.resolve_params({"max_angle_deg": 10.0})

        about_target = step.run(dict(common), params)
        self.assertEqual(about_target["view_roles"][0]["role"], "composited")

        about_splat = step.run(dict(common, splat_center=self.CENTRE), params)
        self.assertEqual(about_splat["view_roles"][0]["role"], "base")

    def test_a_render_on_a_non_black_background_is_refused(self):
        """The failure this prevents is silent: the composite would blend
        toward the background a second time and every face would come back
        washed out."""
        layers = [np.full((8, 6, 3), 127, np.uint8)]
        alphas = [np.zeros((8, 6), np.float32)]
        with self.assertRaises(ValueError) as caught:
            self._run([0.0], layers, alphas)
        self.assertIn("non-black", str(caught.exception))

    def test_misaligned_camera_batches_are_refused(self):
        step = get_step_class("composite_splat_views")()
        with self.assertRaises(ValueError) as caught:
            step.run(
                {
                    "images": self._frames(2),
                    "splat_images": [np.zeros((8, 6, 3), np.uint8)] * 2,
                    "splat_masks": [np.zeros((8, 6), np.float32)] * 2,
                    "cameras": self._cameras([0.0, 10.0]),
                    "splat_cameras": self._cameras([0.0, 40.0]),
                    "splat_center": self.CENTRE,
                },
                step.resolve_params({}),
            )
        self.assertIn("cameras", str(caught.exception))

    def test_the_splat_is_unioned_into_the_mask_when_one_is_wired(self):
        step = get_step_class("composite_splat_views")()
        result = step.run(
            {
                "images": self._frames(1),
                "masks": [np.zeros((8, 6), np.float32)],
                "splat_images": [np.full((8, 6, 3), 200, np.uint8)],
                "splat_masks": [np.full((8, 6), 0.75, np.float32)],
                "cameras": self._cameras([0.0]),
                "splat_center": self.CENTRE,
            },
            step.resolve_params({}),
        )
        np.testing.assert_allclose(result["masks"][0], 0.75)

    def test_no_mask_is_invented_when_none_was_wired(self):
        result = self._run([0.0], [np.zeros((8, 6, 3), np.uint8)],
                           [np.zeros((8, 6), np.float32)])
        self.assertNotIn("masks", result)


if __name__ == "__main__":
    unittest.main()
