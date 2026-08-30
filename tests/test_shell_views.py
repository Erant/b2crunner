"""inject_shell_views: which frames take the shell's render, and what the
VACE mask says about them.

Synthetic throughout, and it can be: the step is pure geometry over a
camera path plus a batch swap. What it cannot check is the thing that
matters most — whether a shell render at 15 degrees off the anchor is
actually good enough to condition a diffusion pass on. That needs a pod.

The camera paths below are built by hand rather than by `render`, so the
geometry under test is stated in the test rather than borrowed from the
thing being tested: cameras sit on a unit sphere about the origin at known
azimuths and elevations, and the anchor is one of them.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from body2colmap.camera import Camera

from tests.helpers import run_step

import pipeline.steps  # noqa: F401


HEIGHT, WIDTH = 8, 6


def camera_at(azimuth_deg: float, elevation_deg: float, radius: float = 2.0) -> Camera:
    """A camera on the orbit sphere about the world origin.

    Only `position` is read by the step; the rest is filled in so the object
    is a real Camera rather than a stub, in case that ever stops being true.
    """
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    position = radius * np.array([
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
        np.cos(elevation) * np.cos(azimuth),
    ], dtype=np.float32)
    return Camera(
        focal_length=(500.0, 500.0), image_size=(WIDTH, HEIGHT),
        principal_point=(WIDTH / 2.0, HEIGHT / 2.0),
        position=position, rotation=np.eye(3, dtype=np.float32),
    )


def frames(count: int, value: int) -> list:
    """A batch whose every pixel carries the frame's index — so a test can
    say which batch a returned frame came from."""
    return [np.full((HEIGHT, WIDTH, 3), value + i, dtype=np.uint8)
            for i in range(count)]


class ShellViewsCase(unittest.TestCase):
    def build(self, azimuths, elevations=None, anchor_index=0):
        elevations = elevations if elevations is not None else [0.0] * len(azimuths)
        cameras = [camera_at(a, e) for a, e in zip(azimuths, elevations)]
        self.mesh_images = frames(len(cameras), 0)
        self.shell_images = frames(len(cameras), 100)
        return {
            "images": self.mesh_images,
            "shell_images": self.shell_images,
            "cameras": cameras,
            "orbit_target": np.zeros(3, dtype=np.float64),
            "anchor_position": np.asarray(cameras[anchor_index].position),
        }

    def sources(self, result):
        """Which batch each returned frame came from, by its pixel value."""
        return ["shell" if int(img[0, 0, 0]) >= 100 else "mesh"
                for img in result["images"]]


class TestTheBand(ShellViewsCase):
    def test_only_frames_inside_the_radius_take_the_shell(self):
        inputs = self.build([0.0, 10.0, 20.0, 90.0, 180.0, -10.0])
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 15.0})
        self.assertEqual(
            self.sources(result),
            ["shell", "shell", "mesh", "mesh", "mesh", "shell"],
        )

    def test_the_radius_is_an_angle_on_the_sphere_not_an_azimuth(self):
        """A helix moves in elevation too, so 15 degrees of azimuth, 15 of
        elevation and 10.6 of both have to be the same distance from the
        anchor. An azimuth-only test would pass on all three of these with
        the wrong implementation."""
        inputs = self.build(
            azimuths=[0.0, 14.0, 0.0, 10.0, 12.0],
            elevations=[0.0, 0.0, 14.0, 10.0, 12.0],
        )
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 15.0})
        # 14 deg of azimuth and 14 of elevation are both inside; 10+10 is
        # 14.1 deg on the sphere and inside; 12+12 is 16.9 and outside.
        self.assertEqual(
            self.sources(result), ["shell", "shell", "shell", "shell", "mesh"],
        )
        angles = [v["angle_from_anchor_deg"] for v in result["view_roles"]]
        self.assertAlmostEqual(angles[1], 14.0, places=3)
        self.assertAlmostEqual(angles[2], 14.0, places=3)
        self.assertAlmostEqual(angles[3], 14.107, places=2)

    def test_a_zero_radius_substitutes_nothing(self):
        inputs = self.build([0.0, 5.0, 10.0])
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 0.0})
        self.assertEqual(self.sources(result), ["mesh", "mesh", "mesh"])
        self.assertEqual([v["role"] for v in result["view_roles"]], ["mesh"] * 3)
        for mask in result["masks"]:
            self.assertTrue(np.all(mask == 1.0))

    def test_the_anchor_need_not_be_the_first_frame(self):
        """`override_cam_from_mesh` puts the anchor wherever on the path the
        original camera falls — frame 44 of 81 on the smoke run's helix."""
        inputs = self.build([0.0, 30.0, 60.0, 90.0, 120.0], anchor_index=2)
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 31.0})
        self.assertEqual(
            self.sources(result), ["mesh", "shell", "shell", "shell", "mesh"],
        )

    def test_the_input_batch_is_not_mutated(self):
        inputs = self.build([0.0, 5.0, 90.0])
        before = [img.copy() for img in self.mesh_images]
        run_step("inject_shell_views", inputs, {"replace_radius_deg": 15.0})
        for original, kept in zip(before, inputs["images"]):
            np.testing.assert_array_equal(original, kept)


class TestTheVaceMask(ShellViewsCase):
    def test_substituted_frames_are_still_marked_for_denoising_by_default(self):
        """The default substitutes without trusting: a shell render is not a
        photograph, and a 0.0 mask tells the diffusion pass to keep its holes
        and fringes verbatim."""
        inputs = self.build([0.0, 5.0, 90.0])
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 15.0})
        self.assertEqual(self.sources(result), ["shell", "shell", "mesh"])
        for mask in result["masks"]:
            self.assertEqual(mask.shape, (HEIGHT, WIDTH))
            self.assertEqual(mask.dtype, np.float32)
            self.assertTrue(np.all(mask == 1.0))

    def test_the_reference_band_is_marked_keep_it(self):
        inputs = self.build([0.0, 5.0, 12.0, 90.0])
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 15.0,
                           "reference_radius_deg": 6.0})
        self.assertEqual([v["role"] for v in result["view_roles"]],
                         ["reference", "reference", "replace_diffuse", "mesh"])
        values = [float(m.max()) for m in result["masks"]]
        self.assertEqual(values, [0.0, 0.0, 1.0, 1.0])
        # Every reference frame is also a substituted frame — marking a mesh
        # drawing "keep it" is the one combination that cannot be right.
        for role, source in zip([v["role"] for v in result["view_roles"]],
                                self.sources(result)):
            if role == "reference":
                self.assertEqual(source, "shell")

    def test_a_reference_band_wider_than_the_replaced_one_is_refused(self):
        inputs = self.build([0.0, 5.0])
        with self.assertRaises(ValueError) as caught:
            run_step("inject_shell_views", inputs,
                     {"replace_radius_deg": 5.0, "reference_radius_deg": 10.0})
        self.assertIn("reference_radius_deg", str(caught.exception))

    def test_the_anchor_injection_after_it_preserves_the_band(self):
        """The wiring the workflow depends on: `inject_anchor` manufactures
        an all-1.0 batch when no `masks` input is wired, which would erase
        this step's roles. With it wired, the band survives and the anchor
        frame itself becomes 0.0."""
        inputs = self.build([0.0, 5.0, 90.0])
        banded = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 15.0,
                           "reference_radius_deg": 15.0})
        anchor_image = np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8)

        without = run_step("inject_anchor", {
            "images": banded["images"], "cameras": inputs["cameras"],
            "anchor_position": inputs["anchor_position"],
            "anchor_image": anchor_image,
        })
        self.assertEqual([float(m.max()) for m in without["masks"]],
                         [0.0, 1.0, 1.0])

        with_masks = run_step("inject_anchor", {
            "images": banded["images"], "cameras": inputs["cameras"],
            "anchor_position": inputs["anchor_position"],
            "anchor_image": anchor_image, "masks": banded["masks"],
        })
        self.assertEqual([float(m.max()) for m in with_masks["masks"]],
                         [0.0, 0.0, 1.0])


class TestTheWiringItRefuses(ShellViewsCase):
    def test_a_shell_batch_of_the_wrong_length(self):
        inputs = self.build([0.0, 5.0, 90.0])
        inputs["shell_images"] = inputs["shell_images"][:2]
        with self.assertRaises(ValueError) as caught:
            run_step("inject_shell_views", inputs, {})
        self.assertIn("matched by index", str(caught.exception))

    def test_shell_cameras_that_are_not_the_frame_batch_s(self):
        """The failure this catches is a `pattern:` left on the shell's
        render_splat, which silently renders a different path — same frame
        count, different views, and the substitution would put frame 40's
        content on frame 3."""
        inputs = self.build([0.0, 5.0, 90.0])
        inputs["shell_cameras"] = [camera_at(a, 0.0) for a in (0.0, 45.0, 90.0)]
        with self.assertRaises(ValueError) as caught:
            run_step("inject_shell_views", inputs, {})
        self.assertIn("pattern", str(caught.exception))

    def test_matching_shell_cameras_are_accepted(self):
        inputs = self.build([0.0, 5.0, 90.0])
        inputs["shell_cameras"] = list(inputs["cameras"])
        result = run_step("inject_shell_views", inputs,
                          {"replace_radius_deg": 15.0})
        self.assertEqual(self.sources(result), ["shell", "shell", "mesh"])

    def test_a_shell_render_at_the_wrong_resolution(self):
        inputs = self.build([0.0, 5.0])
        inputs["shell_images"] = [np.zeros((HEIGHT * 2, WIDTH, 3), np.uint8)] * 2
        with self.assertRaises(ValueError) as caught:
            run_step("inject_shell_views", inputs, {"replace_radius_deg": 15.0})
        self.assertIn("same resolution", str(caught.exception))

    def test_an_anchor_sitting_on_the_orbit_target(self):
        """Which is what an unanchored render leaves behind: no direction to
        measure a band around, so the whole idea is undefined."""
        inputs = self.build([0.0, 5.0])
        inputs["anchor_position"] = np.zeros(3, dtype=np.float64)
        with self.assertRaises(ValueError) as caught:
            run_step("inject_shell_views", inputs, {"replace_radius_deg": 15.0})
        self.assertIn("orbit target", str(caught.exception))


class TestAgainstTheSmokeRun(unittest.TestCase):
    """The one non-synthetic check available: the local smoke run that
    produced the shell in the first place (output/smoke_shell_helical/,
    gitignored, so this skips when it is not there).

    Its `smoke.py` did the substitution inline, before this step existed,
    and wrote a manifest recording every view's angle from the anchor and
    which batch its frame came from. The step has to reproduce that
    composition exactly — the same 7 of 81 frames, at the same angles —
    from the camera geometry alone. Manifest views carry azimuth and
    elevation but not the orbit radius; the radius cancels out of an angle
    between two directions, so a unit sphere reproduces it.
    """

    MANIFEST = (Path(__file__).resolve().parent.parent
                / "output" / "smoke_shell_helical" / "manifest.json")

    @classmethod
    def setUpClass(cls):
        if not cls.MANIFEST.exists():
            raise unittest.SkipTest(f"smoke run missing: {cls.MANIFEST}")
        cls.manifest = json.loads(cls.MANIFEST.read_text())

    def test_it_substitutes_exactly_the_frames_the_smoke_run_did(self):
        views = self.manifest["views"]
        cameras = [camera_at(v["azimuth_deg"], v["elevation_deg"], radius=1.0)
                   for v in views]
        anchor = cameras[self.manifest["anchor_frame_index"]]
        result = run_step("inject_shell_views", {
            "images": frames(len(views), 0),
            "shell_images": frames(len(views), 100),
            "cameras": cameras,
            "orbit_target": np.zeros(3),
            "anchor_position": np.asarray(anchor.position),
        }, {"replace_radius_deg": self.manifest["radii_deg"]["replace"],
            "reference_radius_deg": self.manifest["radii_deg"]["reference"]})

        recorded = [v["source"] for v in views]
        reproduced = ["splat" if r["source"] == "shell" else "mesh"
                      for r in result["view_roles"]]
        self.assertEqual(reproduced, recorded)
        self.assertEqual(sum(1 for r in reproduced if r == "splat"),
                         self.manifest["counts"]["reference"]
                         + self.manifest["counts"]["replace_diffuse"])

        for view, role in zip(views, result["view_roles"]):
            with self.subTest(frame=view["name"]):
                self.assertAlmostEqual(role["angle_from_anchor_deg"],
                                       view["angle_from_anchor_deg"], places=2)

    def test_the_vace_masks_match_what_that_run_planned(self):
        """smoke.py recorded a `planned_vace_mask` per frame for "the step
        after this one" — which is this step."""
        views = self.manifest["views"]
        cameras = [camera_at(v["azimuth_deg"], v["elevation_deg"], radius=1.0)
                   for v in views]
        result = run_step("inject_shell_views", {
            "images": frames(len(views), 0),
            "shell_images": frames(len(views), 100),
            "cameras": cameras,
            "orbit_target": np.zeros(3),
            "anchor_position": np.asarray(
                cameras[self.manifest["anchor_frame_index"]].position),
        }, {"replace_radius_deg": self.manifest["radii_deg"]["replace"],
            "reference_radius_deg": self.manifest["radii_deg"]["reference"]})
        self.assertEqual([float(m.max()) for m in result["masks"]],
                         [v["planned_vace_mask"] for v in views])


if __name__ == "__main__":
    unittest.main()
