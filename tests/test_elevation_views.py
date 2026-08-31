"""pointmap_elevation_views — a shell per orbit frame, and the pose seam.

Same discipline as tests/test_pointmap_splat.py: synthetic geometry with a
closed-form answer, the 1B forward pass stubbed, and the render binary
stubbed. What is new here is that the shell is no longer built at the
world origin, so three claims need pinning that the photo path never had
to make:

  * **`pose=None` is not `pose=(I, 0)`.** The two agree to the last bit of
    value, but only the `None` branch skips the transform arithmetic
    entirely, and that skip is what makes `pointmap_splat` and
    `face_pointmap_splat` byte-identical to what they were before the
    argument existed (an identity matmul turns some +0.0 into -0.0).
    Both halves of that are checked.
  * **The mesh's route into a camera has two FLIPs**, and they cancel at
    the identity pose — so forgetting the inner one is invisible to every
    identity-pose test and only shows up as nonsense depth on a pod.
    `TestTheMeshRoute` is that test, taken at a camera that is *not* the
    origin.
  * **The pair is an elevation change and nothing else.** Same radius,
    same azimuth, same intrinsics, measured from the source camera's OWN
    elevation rather than from the equator, because this orbit's ring
    sits at the photographer's height.

The diagnostics get tests of their own for the same reason they exist: a
number that is quietly always zero would let a bad batch through looking
healthy.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline.steps  # noqa: F401  — registers pointmap_elevation_views
from pipeline.registry import get_step_class
from pipeline.steps import elevation_views as ev
from pipeline.steps import pointmap_splat as ps
from pipeline.steps import splat as splat_step
from tests.test_pointmap_splat import _quat_to_mat, _sphere

#: A deliberately small frame, unlike tests/test_pointmap_splat.py's. This
#: module builds a whole RING of shells rather than one, and every one of
#: them runs the same conjugate-gradient depth solve — at that module's
#: 320x440 the agreement test alone would be most of a minute.
FOCAL, CX, CY = 200.0, 48.0, 64.0
WIDTH, HEIGHT = 96, 128

#: The synthetic orbit: a sphere at the pivot, cameras on a ring around it.
#: The radius is chosen so the sphere's silhouette fits inside the frame —
#: a clipped silhouette makes `height_ratio` measure the frame rather than
#: the mesh.
RADIUS = 2.0
SPHERE_RADIUS = 0.35
RING_ELEVATION = 12.0          # the photographer's height, not the equator
TARGET = np.array([0.0, 0.0, 0.0])


def _surface():
    """The sphere at the pivot, as this module's camera sees it."""
    return _sphere(width=WIDTH, height=HEIGHT, f=FOCAL, cx=CX, cy=CY,
                   centre_z=RADIUS, radius=SPHERE_RADIUS)


def _camera(azimuth_deg: float, elevation_deg: float = RING_ELEVATION):
    from body2colmap import coordinates
    from body2colmap.camera import Camera

    camera = Camera(
        focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
        principal_point=(CX, CY),
        position=TARGET + coordinates.spherical_to_cartesian(
            RADIUS, azimuth_deg, elevation_deg),
    )
    camera.look_at(TARGET, coordinates.WorldCoordinates.UP_AXIS)
    return camera


def _ring(count: int, overlap: bool = True):
    """`count` cameras round a circle, optionally with `render`'s duplicate.

    `render`'s `overlap` defaults to 1, so a circular `OrbitPath` appends
    `cameras[0]` again as the last camera — two different denoised frames
    through one camera. The dedupe in `select_indices` exists for it.
    """
    cameras = [_camera(360.0 * i / count) for i in range(count)]
    if overlap:
        cameras.append(_camera(0.0))
    return cameras


def _mesh_output():
    """A sphere of vertices at the pivot, as SAM-3D-Body would publish it.

    `mesh_in_world` computes `FLIP * (vertices + cam_t)`, and FLIP is its
    own inverse, so pre-flipping the world points is how a chosen world
    mesh is expressed in SAM-3D-Body's frame.
    """
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(4000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    world = TARGET + SPHERE_RADIUS * directions
    return {"vertices": world * ps.FLIP, "cam_t": np.zeros(3), "focal_length": FOCAL}


class _StubbedStep(ev.PointmapElevationViewsStep):
    """The step with the pointmap head and the render binary replaced.

    The pointmap is the same map for every frame, which is exactly right
    for this scene: a sphere at the pivot looks identical from every camera
    on the ring, so a correct implementation must place seventeen identical
    pointmaps at seventeen different poses and have them land on one
    sphere.
    """

    def __init__(self, pointmap: np.ndarray, fail: tuple = ()) -> None:
        super().__init__()
        self._stub = pointmap
        self._fail = set(fail)
        self._frame = None
        self.rendered = []

    def load(self, params):  # noqa: D102 — no model to load
        self._model = object()
        self._checkpoint = params["checkpoint"]

    def unload(self):  # noqa: D102 — nothing to free, and no torch in this env
        self._model = None

    def _build_shell(self, image, matte, normal_map, **kwargs):  # noqa: D102
        label = kwargs["label"]
        index = int(label[label.index("[") + 1:label.index("]")])
        if index in self._fail:
            raise ValueError(f"stub: frame {index} was told to fail")
        return super()._build_shell(image, matte, normal_map, **kwargs)

    def _pointmap(self, image_bgr, params):  # noqa: D102
        return self._stub


def _stub_rasterize(monkey_store, colour=180):
    """Replace `_rasterize` with a recorder that returns a plain disc."""
    def fake(*, scene, splat_path, cameras, image_names, width, height,
             bg_color, render_path, confidence=None):
        monkey_store.append({"splat_path": splat_path, "cameras": list(cameras),
                             "names": list(image_names), "bg_color": bg_color,
                             "render_path": render_path})
        images, masks = [], []
        for _ in cameras:
            alpha = np.zeros((height, width), np.float32)
            alpha[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = 1.0
            images.append((np.full((height, width, 3), colour, np.uint8)
                           * alpha[..., None].astype(np.uint8)))
            masks.append(alpha)
        return images, masks
    return fake


def _scene(scale_the_pointmap_by: float = 2.5):
    """The frames, the pointmap the stub returns, and the mesh."""
    z, normals, mask = _surface()
    ripple = 1.0 + 0.02 * np.sin(np.mgrid[0:HEIGHT, 0:WIDTH][1] / 8.0)
    pointmap = ps.backproject(np.where(mask, z * ripple, 0.0) / scale_the_pointmap_by,
                              FOCAL, CX, CY).astype(np.float32)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    image[..., 2] = 255
    return z, normals, mask, image, pointmap


def _inputs(count: int, overlap: bool = True):
    z, normals, mask, image, pointmap = _scene()
    cameras = _ring(count, overlap=overlap)
    n = len(cameras)
    return pointmap, {
        "images": [image] * n,
        "masks": [mask.astype(np.float32)] * n,
        "normal_maps": [normals * ps.NORMAL_TO_CAMERA_FRAME] * n,
        "cameras": cameras,
        "mesh_output": _mesh_output(),
        "orbit_target": TARGET,
        "anchor_frame_index": 0,
    }


# ---------------------------------------------------------------------------
class TestThePoseSeam(unittest.TestCase):
    """`build_gaussians(pose=...)` — the one change inside the shared numerics."""

    def _gaussians(self, **kwargs):
        z, normals, mask = _surface()
        xyz = ps.backproject(z, FOCAL, CX, CY)
        rgb = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
        return ps.build_gaussians(xyz, normals, rgb, mask.astype(np.float64),
                                  mask, FOCAL, **kwargs)

    def test_none_and_the_identity_pose_agree_on_every_value(self):
        plain = self._gaussians()
        identity = self._gaussians(pose=(np.eye(3), np.zeros(3)))
        for key in plain:
            np.testing.assert_array_equal(plain[key], identity[key], err_msg=key)

    def test_only_the_none_branch_is_byte_identical(self):
        """Why `pose=None` skips the arithmetic instead of passing I through it.

        A matmul with the identity is value-preserving and not
        bit-preserving: it turns +0.0 into -0.0 wherever a row sums to
        zero. `pointmap_splat` and `face_pointmap_splat` are pinned to
        byte-identity against their pre-pose output, so they must take a
        branch that does no arithmetic at all — this documents that the
        difference is real rather than theoretical.
        """
        plain = self._gaussians()["means"]
        identity = self._gaussians(pose=(np.eye(3), np.zeros(3)))["means"]
        np.testing.assert_array_equal(plain, identity)
        self.assertTrue((np.signbit(plain) != np.signbit(identity)).any(),
                        "no signed zero appeared, so this test no longer says "
                        "anything about why the None branch exists")

    def test_a_pose_rigidly_carries_means_normals_and_frames(self):
        camera = _camera(37.0)
        rotation, position = ev.camera_pose(camera)
        plain = self._gaussians()
        posed = self._gaussians(pose=(rotation, position))

        np.testing.assert_allclose(posed["means"],
                                   plain["means"] @ rotation.T + position, atol=1e-12)
        np.testing.assert_allclose(posed["normals"],
                                   plain["normals"] @ rotation.T, atol=1e-12)
        # Scales are invariant under a rigid transform, and colours and
        # opacities know nothing about where the shell is.
        for key in ("log_scales", "sh_dc", "opacity"):
            np.testing.assert_array_equal(posed[key], plain[key])
        # The orthonormal frame is rotated BEFORE the quaternion
        # conversion, so the quats must come out as R times the originals'
        # matrices — not as some renormalised approximation of it.
        np.testing.assert_allclose(_quat_to_mat(posed["quats"]),
                                   rotation @ _quat_to_mat(plain["quats"]),
                                   atol=1e-9)

    def test_a_posed_shell_reprojects_onto_its_own_pixels(self):
        """The gate, moved off the origin.

        `pointmap_splat`'s own gate is that every Gaussian lands back on
        the pixel it came from through SAM-3D-Body's camera. The same claim
        has to survive the pose, through the source camera this time, or
        the whole design ("lateral position is exact by construction")
        is false the moment a shell leaves the origin.
        """
        camera = _camera(-63.0)
        posed = self._gaussians(pose=ev.camera_pose(camera))
        uv, z = ev.project_cv(posed["means"], camera)
        self.assertTrue((z > 0).all())
        self.assertLess(float(np.abs(uv - np.round(uv)).max()), 1e-6)


class TestTheMeshRoute(unittest.TestCase):
    def test_the_two_flips_cancel_at_the_origin(self):
        mesh = _mesh_output()
        origin = _camera(0.0, 0.0)
        origin.position = np.zeros(3, dtype=np.float32)
        origin.rotation = np.eye(3, dtype=np.float32)
        expected = np.asarray(mesh["vertices"]) + np.asarray(mesh["cam_t"])
        np.testing.assert_allclose(
            ev.vertices_in_camera(ev.mesh_in_world(mesh), origin), expected, atol=1e-12)

    def test_at_a_real_orbit_camera_the_mesh_is_in_front_and_upright(self):
        """The case the identity pose cannot see.

        With the inner FLIP dropped the mesh comes back mirrored — still in
        front of the camera, still the right size, so nothing raises. What
        it stops doing is agreeing with the frame: the sphere here is
        centred on the pivot, so the check that bites is that its centre
        lands at (0, 0, radius) in the camera's own OpenCV frame.
        """
        mesh_world = ev.mesh_in_world(_mesh_output())
        for azimuth in (0.0, 90.0, 187.0, -45.0):
            with self.subTest(azimuth=azimuth):
                in_camera = ev.vertices_in_camera(mesh_world, _camera(azimuth))
                centre = in_camera.mean(axis=0)
                np.testing.assert_allclose(centre, [0.0, 0.0, RADIUS], atol=0.05)
                self.assertGreater(in_camera[:, 2].min(), 0.0)

    def test_a_mirrored_mesh_is_caught_by_the_silhouette_ratio(self):
        """`silhouette.height_ratio` is the run-time number that catches it.

        Not a unit-testable one: it is published per shell so a pod run's
        log says whether the mesh landed in the right camera. Here it is
        checked at the extreme — an upside-down mesh against an upright
        matte — so the number is known to move at all.
        """
        mesh_world = ev.mesh_in_world(_mesh_output())
        camera = _camera(24.0)
        _, _, mask = _surface()
        good = ps.silhouette_agreement(
            mask, ev.vertices_in_camera(mesh_world, camera), FOCAL, CX, CY)
        # A body-shaped mesh mirrored top to bottom about its own centre.
        stretched = mesh_world.copy()
        stretched[:, 1] = (stretched[:, 1] - TARGET[1]) * 3.0 + TARGET[1]
        bad = ps.silhouette_agreement(
            mask, ev.vertices_in_camera(stretched, camera), FOCAL, CX, CY)
        self.assertAlmostEqual(good["height_ratio"], 1.0, delta=0.05)
        self.assertGreater(abs(bad["height_ratio"] - 1.0), 0.5)


class TestSelection(unittest.TestCase):
    def test_the_anchor_frame_is_always_selected(self):
        cameras = _ring(20)
        for every_n in (1, 3, 5, 7, 10):
            for anchor in (0, 4, 13):
                with self.subTest(every_n=every_n, anchor=anchor):
                    kept, _ = ev.select_indices(len(cameras), every_n, anchor, cameras)
                    self.assertIn(anchor, kept)

    def test_the_overlap_duplicate_is_dropped(self):
        cameras = _ring(20)                      # 21 cameras, the last a repeat
        kept, dropped = ev.select_indices(len(cameras), 4, 0, cameras)
        self.assertEqual(dropped, [20])
        self.assertEqual(kept, [0, 4, 8, 12, 16])

    def test_without_the_duplicate_nothing_is_dropped(self):
        cameras = _ring(20, overlap=False)
        kept, dropped = ev.select_indices(len(cameras), 4, 0, cameras)
        self.assertEqual(dropped, [])
        self.assertEqual(kept, [0, 4, 8, 12, 16])

    def test_every_n_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            ev.select_indices(10, 0, 0, _ring(10))


class TestTheElevationPair(unittest.TestCase):
    def _pair(self, source, delta=20.0):
        from body2colmap import coordinates
        from body2colmap.camera import Camera

        radius, azimuth, elevation = coordinates.cartesian_to_spherical(
            np.asarray(source.position, np.float64) - TARGET)
        step = ev.PointmapElevationViewsStep()
        return step._elevation_pair(source, TARGET, radius, azimuth, elevation,
                                    delta, Camera, coordinates, "test", 0), \
            (radius, azimuth, elevation)

    def test_it_changes_the_elevation_and_nothing_else(self):
        from body2colmap import coordinates

        source = _camera(123.0)
        (up, down), (radius, azimuth, elevation) = self._pair(source, 20.0)
        for camera, expected in ((up, elevation + 20.0), (down, elevation - 20.0)):
            r, a, e = coordinates.cartesian_to_spherical(
                np.asarray(camera.position, np.float64) - TARGET)
            self.assertAlmostEqual(r, radius, places=5)
            self.assertAlmostEqual(a, azimuth, places=4)
            self.assertAlmostEqual(e, expected, places=4)
            # Same intrinsics as the source view: these views share a
            # COLMAP camera with the 81 training frames.
            self.assertEqual((camera.fx, camera.fy, camera.cx, camera.cy),
                             (source.fx, source.fy, source.cx, source.cy))
            self.assertEqual((camera.width, camera.height),
                             (source.width, source.height))

    def test_the_offset_is_from_the_rings_own_elevation(self):
        """Not from the equator. `override_cam_from_mesh` puts frame 0 on the
        photograph's camera, so the ring sits at the photographer's height."""
        from body2colmap import coordinates

        (up, down), _ = self._pair(_camera(0.0), 10.0)
        elevations = [coordinates.cartesian_to_spherical(
            np.asarray(c.position, np.float64) - TARGET)[2] for c in (up, down)]
        self.assertAlmostEqual(elevations[0], RING_ELEVATION + 10.0, places=4)
        self.assertAlmostEqual(elevations[1], RING_ELEVATION - 10.0, places=4)

    def test_the_pole_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._pair(_camera(0.0, 80.0), 20.0)
        self.assertIn("look_at", str(caught.exception))


class TestDiagnostics(unittest.TestCase):
    def test_pair_residual_is_zero_for_two_shells_of_one_surface(self):
        z, _, mask = _surface()
        camera_i, camera_j = _camera(0.0), _camera(20.0)
        # Shell i's means: its own depth map, carried into world.
        rotation, position = ev.camera_pose(camera_i)
        means = (ps.backproject(z, FOCAL, CX, CY)[mask] * ps.FLIP) @ rotation.T + position
        result = ev.pair_residual(means, camera_j, z, mask)
        self.assertGreater(result["overlap_px"], 1000)
        self.assertLess(abs(result["signed_median_mm"]), 1.0)

    def test_pair_residual_reports_a_shell_pushed_toward_its_own_camera(self):
        """The failure mode §6 of the plan names: each shell's depth ruler is
        read with a bias that varies with azimuth, so neighbours disagree."""
        z, _, mask = _surface()
        camera_i, camera_j = _camera(0.0), _camera(20.0)
        rotation, position = ev.camera_pose(camera_i)
        shifted = np.where(mask, z - 0.05, 0.0)          # 50 mm toward camera i
        means = (ps.backproject(shifted, FOCAL, CX, CY)[mask] * ps.FLIP)
        means = means @ rotation.T + position
        result = ev.pair_residual(means, camera_j, z, mask)
        self.assertGreater(abs(result["signed_median_mm"]), 20.0)

    def test_mesh_residual_moves_millimetre_for_millimetre(self):
        """What the number has to do is track the shell's placement.

        Not sit at zero: `mesh_front_depth` is the NEAREST projected vertex
        per bin, which over a curved surface is systematically closer than
        the median depth of the pixels in that bin — a real offset, present
        on a pod too, and the reason the plan reads the residual's sign and
        its consistency across shells rather than its absolute value.
        """
        z, _, mask = _surface()
        front = ps.mesh_front_depth(
            ev.vertices_in_camera(ev.mesh_in_world(_mesh_output()), _camera(0.0)),
            FOCAL, CX, CY, (HEIGHT, WIDTH), 16)
        on_the_mesh = ev.mesh_residual(z, mask, front, 16)
        in_front = ev.mesh_residual(np.where(mask, z - 0.010, 0.0), mask, front, 16)
        behind = ev.mesh_residual(np.where(mask, z + 0.010, 0.0), mask, front, 16)
        self.assertGreater(on_the_mesh["bins"], 10)
        self.assertAlmostEqual(in_front["median_mm"],
                               on_the_mesh["median_mm"] - 10.0, places=6)
        self.assertAlmostEqual(behind["median_mm"],
                               on_the_mesh["median_mm"] + 10.0, places=6)

    def test_eroding_the_alpha_pulls_the_silhouette_in(self):
        alpha = np.zeros((32, 32), np.float32)
        alpha[8:24, 8:24] = 1.0
        self.assertIs(ev._erode_alpha(alpha, 0), alpha)
        eroded = ev._erode_alpha(alpha, 2)
        self.assertLess(float(eroded.sum()), float(alpha.sum()))
        self.assertTrue((eroded <= alpha).all())


class TestRun(unittest.TestCase):
    """The whole step, with the head and the render binary stubbed."""

    def _run(self, count=12, fail=(), **params):
        pointmap, inputs = _inputs(count)
        step = _StubbedStep(pointmap, fail=fail)
        calls = []
        original = splat_step._rasterize
        splat_step._rasterize = _stub_rasterize(calls)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                merged = dict({"output_dir": tmp, "every_n": 4,
                               "elevation_deg": 10.0}, **params)
                result = step.run(
                    inputs, ev.PointmapElevationViewsStep.resolve_params(merged))
                result["_plys"] = sorted(p.name for p in Path(tmp).glob("*.ply"))
        finally:
            splat_step._rasterize = original
        return result, calls, inputs

    def test_it_emits_two_views_per_shell_and_keeps_the_plys(self):
        result, calls, _ = self._run(count=12)
        stats = result["stats"]
        self.assertEqual(stats["selected"], [0, 4, 8])
        self.assertEqual(stats["dropped_duplicate_cameras"], [12])
        self.assertEqual(len(result["images"]), 6)
        self.assertEqual(len(result["masks"]), 6)
        self.assertEqual(len(result["cameras"]), 6)
        self.assertEqual(stats["views"], 6)
        self.assertEqual(result["_plys"],
                         ["view_00000.ply", "view_00004.ply", "view_00008.ply"])
        # One binary launch per shell, both its cameras in the one call.
        self.assertEqual([len(call["cameras"]) for call in calls], [2, 2, 2])
        self.assertEqual(calls[0]["bg_color"], (0.0, 0.0, 0.0))

    def test_the_views_sit_above_and_below_the_ring(self):
        from body2colmap import coordinates

        result, _, _ = self._run(count=12, elevation_deg=10.0)
        elevations = sorted(
            round(coordinates.cartesian_to_spherical(
                np.asarray(c.position, np.float64) - TARGET)[2], 3)
            for c in result["cameras"])
        self.assertEqual(set(elevations),
                         {round(RING_ELEVATION - 10.0, 3), round(RING_ELEVATION + 10.0, 3)})

    def test_the_shells_agree_with_each_other_and_with_the_mesh(self):
        """The scene is one sphere seen from every camera, so identical
        pointmaps placed at different poses must land on one surface. This
        is the end-to-end version of the pose seam's reprojection test:
        it goes through the depth solve, the mesh scale fit and the
        placement rather than only through `build_gaussians`."""
        # A DENSE ring, deliberately: the pair residual is a claim about
        # NEIGHBOURING shells, and three shells spread round a circle are
        # 120 degrees apart, where a 2.5-D shell of a sphere is mostly the
        # far side of it. The real orbit selects every 5th or 10th of 81
        # frames, i.e. 20-45 degrees.
        result, _, _ = self._run(count=18, every_n=1)
        stats = result["stats"]
        self.assertEqual(len(stats["shells"]), 18)
        for shell in stats["shells"]:
            self.assertAlmostEqual(shell["silhouette"]["height_ratio"], 1.0, delta=0.1)
            self.assertLess(abs(shell["mesh_residual"]["median_mm"]), 25.0)
        for pair in stats["pairs"]:
            self.assertLess(pair["abs_median_mm"], 25.0)
        self.assertLess(stats["pair_residual_median_mm"], 25.0)

    def test_one_bad_frame_is_skipped_rather_than_fatal(self):
        """A run forty minutes into a pod must not die because the denoiser
        lost the subject in one frame."""
        result, calls, _ = self._run(count=12, fail=(4,))
        self.assertEqual(sorted(result["stats"]["failed"]), ["4"])
        self.assertEqual(len(result["images"]), 4)
        self.assertEqual(result["_plys"], ["view_00000.ply", "view_00008.ply"])

    def test_every_frame_failing_raises(self):
        with self.assertRaises(ValueError) as caught:
            self._run(count=12, fail=(0, 4, 8))
        self.assertIn("every one", str(caught.exception))

    def test_a_camera_that_disagrees_with_its_frame_is_refused(self):
        pointmap, inputs = _inputs(8)
        inputs["cameras"][0].width = WIDTH + 1
        step = _StubbedStep(pointmap)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                step.run(inputs, ev.PointmapElevationViewsStep.resolve_params(
                    {"output_dir": tmp, "every_n": 4}))
        self.assertIn("same view", str(caught.exception))

    def test_parallel_inputs_of_different_lengths_are_refused(self):
        pointmap, inputs = _inputs(8)
        inputs["masks"] = inputs["masks"][:-1]
        step = _StubbedStep(pointmap)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                step.run(inputs, ev.PointmapElevationViewsStep.resolve_params(
                    {"output_dir": tmp}))
        self.assertIn("disagree", str(caught.exception))

    def test_it_declares_no_filepath(self):
        """The base's `filepath` is REQUIRED and names one ply; this step
        names one per shell under `output_dir`, so the knob is gone rather
        than left in the UI doing nothing."""
        declared = ev.PointmapElevationViewsStep.declared_params()
        self.assertNotIn("filepath", declared)
        self.assertIn("output_dir", declared)
        # The measured departure from the base's tuning, and the only one.
        self.assertEqual(declared["align_bin_px"].default, 16)
        self.assertEqual(ps.PointmapSplatStep.declared_params()["align_bin_px"].default, 32)


class TestMergeSupportViews(unittest.TestCase):
    def _run(self, inputs, **params):
        step = get_step_class("merge_support_views")()
        merged = get_step_class("merge_support_views").resolve_params(params)
        return step.run(inputs, merged)

    def test_both_branches_are_concatenated_in_order(self):
        result = self._run({
            "a_images": ["fa"], "a_masks": ["ma"], "a_cameras": ["ca"],
            "b_images": ["fb1", "fb2"], "b_masks": ["mb1", "mb2"],
            "b_cameras": ["cb1", "cb2"],
        })
        self.assertEqual(result["images"], ["fa", "fb1", "fb2"])
        self.assertEqual(result["masks"], ["ma", "mb1", "mb2"])
        self.assertEqual(result["cameras"], ["ca", "cb1", "cb2"])

    def test_a_missing_branch_contributes_nothing(self):
        result = self._run({"b_images": ["f"], "b_masks": ["m"], "b_cameras": ["c"]})
        self.assertEqual(result["images"], ["f"])

    def test_both_branches_off_is_three_empty_lists(self):
        """Not an error: brush then trains exactly as it did before either
        branch existed, which is what makes this step safe to run ungated."""
        self.assertEqual(self._run({}),
                         {"images": [], "masks": [], "cameras": []})

    def test_require_any_turns_that_into_a_failure(self):
        with self.assertRaises(ValueError):
            self._run({}, require_any=True)

    def test_a_ragged_triple_names_which_one(self):
        with self.assertRaises(ValueError) as caught:
            self._run({"b_images": ["f"], "b_masks": [], "b_cameras": ["c"]})
        self.assertIn("'b'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
