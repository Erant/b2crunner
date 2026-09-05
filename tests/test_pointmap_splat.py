"""pointmap_splat — the feed-forward photo-to-splat shell.

No recorded golden: the Sapiens2 pointmap head is not runnable in a test,
so everything here is synthetic geometry where the right answer is known in
closed form. Two groups:

  * the numerics ported from masktest (intrinsics fit, normal integration,
    Gaussian construction, PLY layout) — checked against analytic surfaces
    and against the conventions brush's importer actually reads;
  * the part that is NOT masktest's and has no upstream to inherit
    confidence from: placing the shell in SAM-3D-Body's camera. The load-
    bearing claim there is that every Gaussian sits on the ray through the
    pixel it came from, so the shell reprojects onto the photo it was built
    from. `test_run_puts_every_gaussian_back_on_its_own_pixel` is that gate.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline.steps  # noqa: F401  — registers pointmap_splat
from pipeline.steps import pointmap_splat as ps

FOCAL, CX, CY = 900.0, 160.0, 220.0
WIDTH, HEIGHT = 320, 440


def _sphere(width=WIDTH, height=HEIGHT, f=FOCAL, cx=CX, cy=CY,
            centre_z=2.0, radius=0.55):
    """Front hemisphere of a sphere seen by a pinhole at the origin.

    Returns (z, normals_camera_frame, mask). The normals are exact, which
    is what makes this a fair test of the integration: any depth error is
    the solver's, not the data's.
    """
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = (uu - cx) / f, (vv - cy) / f
    # |t*(dx,dy,1) - (0,0,cz)| = r  ->  quadratic in t (which is the depth).
    a = dx ** 2 + dy ** 2 + 1.0
    b = -2.0 * centre_z
    c = centre_z ** 2 - radius ** 2
    disc = b ** 2 - 4 * a * c
    mask = disc > 0
    z = np.zeros_like(disc)
    z[mask] = (-b - np.sqrt(disc[mask])) / (2 * a[mask])   # near intersection

    points = np.stack([dx * z, dy * z, z], axis=2)
    normals = points - np.array([0.0, 0.0, centre_z])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-12
    return z, normals, mask


def _quat_to_mat(quats: np.ndarray) -> np.ndarray:
    """(N,4) (w,x,y,z) -> (N,3,3), the column convention brush applies."""
    w, x, y, z = quats.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
    ], axis=1)


def _read_ply(path: Path):
    raw = path.read_bytes()
    marker = b"end_header\n"
    head = raw[: raw.index(marker)].decode("ascii").splitlines()
    body = raw[raw.index(marker) + len(marker):]
    props = [line.split()[-1] for line in head if line.startswith("property float")]
    count = int(next(line for line in head if line.startswith("element vertex")).split()[-1])
    data = np.frombuffer(body, np.float32).reshape(count, len(props))
    return head, props, data


class _StubbedStep(ps.PointmapSplatStep):
    """The step with the 1B forward pass replaced by a known pointmap."""

    def __init__(self, pointmap: np.ndarray) -> None:
        super().__init__()
        self._stub = pointmap

    def load(self, params):  # noqa: D102 — no model to load
        self._model = object()
        self._checkpoint = params["checkpoint"]

    def _pointmap(self, image_bgr, params):  # noqa: D102
        return self._stub


# ---------------------------------------------------------------------------
class TestCameraMath(unittest.TestCase):
    def test_fit_intrinsics_recovers_a_synthetic_pinhole(self):
        z, _, mask = _sphere()
        xyz = ps.backproject(z, FOCAL, CX, CY)
        f, cx, cy, rms = ps.fit_intrinsics(xyz, mask)
        self.assertAlmostEqual(f, FOCAL, places=3)
        self.assertAlmostEqual(cx, CX, places=3)
        self.assertAlmostEqual(cy, CY, places=3)
        self.assertLess(rms, 1e-6)

    def test_backproject_puts_the_point_on_its_own_ray(self):
        z, _, mask = _sphere()
        xyz = ps.backproject(z, FOCAL, CX, CY)
        vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        u_back = FOCAL * xyz[..., 0] / np.where(z > 0, z, 1) + CX
        v_back = FOCAL * xyz[..., 1] / np.where(z > 0, z, 1) + CY
        self.assertLess(np.abs(u_back - uu)[mask].max(), 1e-9)
        self.assertLess(np.abs(v_back - vv)[mask].max(), 1e-9)

    def test_camera_frame_normals_flip_the_frame_and_face_the_camera(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(HEIGHT, WIDTH, 3))
        raw /= np.linalg.norm(raw, axis=2, keepdims=True)
        n = ps.camera_frame_normals(raw, FOCAL, CX, CY)

        self.assertLess(abs(np.linalg.norm(n, axis=2) - 1).max(), 1e-9)
        # Every normal faces the camera: n . (ray from the origin) < 0.
        vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        ray = np.stack([(uu - CX) / FOCAL, (vv - CY) / FOCAL, np.ones_like(uu)], 2)
        self.assertLessEqual((n * ray).sum(2).max(), 0.0)
        # ...and it is the [1,-1,-1] mapping, up to that sign flip.
        expected = raw * ps.NORMAL_TO_CAMERA_FRAME
        agree = np.abs((n * expected).sum(2))
        self.assertLess(abs(agree - 1).max(), 1e-9)


class TestIntegration(unittest.TestCase):
    def test_it_scrubs_high_frequency_depth_error_at_the_default_lambda(self):
        """The regime the default lambda is tuned for: the pointmap's low
        frequencies are right and its fine structure is not."""
        z, normals, mask = _sphere()
        vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        ripple = 1.0 + 0.02 * np.sin(2 * np.pi * uu / 8) * np.sin(2 * np.pi * vv / 8)
        corrupted = np.where(mask, z * ripple, 0.0)

        before = ps.normal_angle_error(ps.backproject(corrupted, FOCAL, CX, CY), normals, mask)
        solved = ps.integrate_depth(corrupted, normals, mask, FOCAL, CX, CY, lam=0.01)
        after = ps.normal_angle_error(ps.backproject(solved.astype(np.float64), FOCAL, CX, CY),
                                      normals, mask)

        self.assertGreater(float(np.median(before)), 50.0)
        self.assertLess(float(np.median(after)), 5.0)
        self.assertLess(float(np.median(np.abs(solved[mask] - z[mask]) / z[mask])), 1e-3)

    def test_a_lower_lambda_hands_the_global_shape_to_the_normals_too(self):
        """The other end of the trade-off, and why 0.01 is a choice rather
        than a limit: at 0.01 a *low*-frequency error survives, because that
        is exactly what the data term is there to hold on to. Turning the
        data term down lets the normals rebuild the whole sphere."""
        z, normals, mask = _sphere()
        flattened = np.where(mask, z.mean() + 0.2 * (z - z.mean()), 0.0)

        loose = ps.integrate_depth(flattened, normals, mask, FOCAL, CX, CY, lam=1e-4)
        tight = ps.integrate_depth(flattened, normals, mask, FOCAL, CX, CY, lam=0.01)

        self.assertLess(float(np.median(np.abs(loose[mask] - z[mask]) / z[mask])), 0.01)
        self.assertGreater(float(np.median(np.abs(tight[mask] - z[mask]) / z[mask])), 0.01)

    def test_a_scale_on_the_data_term_passes_straight_through(self):
        """The claim the whole placement order rests on: fitting the mesh
        scale before integrating and after it are the same thing, because
        only differences of w = log z enter the gradient equations."""
        z, normals, mask = _sphere()
        flattened = np.where(mask, z.mean() + 0.2 * (z - z.mean()), 0.0)

        plain = ps.integrate_depth(flattened, normals, mask, FOCAL, CX, CY, lam=0.01)
        scaled = ps.integrate_depth(flattened * 3.0, normals, mask, FOCAL, CX, CY, lam=0.01)

        self.assertLess(float(np.abs(scaled[mask] / plain[mask] - 3.0).max()), 1e-3)

    def test_a_frontoparallel_plane_solves_to_one_depth(self):
        mask = np.ones((40, 50), bool)
        normals = np.zeros((40, 50, 3))
        normals[..., 2] = -1.0                      # facing the camera
        rng = np.random.default_rng(1)
        z0 = 2.0 * np.exp(rng.normal(scale=0.05, size=(40, 50)))

        solved = ps.integrate_depth(z0, normals, mask, FOCAL, CX, CY, lam=1e-5)

        self.assertLess(float(np.ptp(solved)), 1e-3)
        # With every gradient equation reading "no change", the data term
        # alone decides, and it is quadratic in log z: the geometric mean.
        self.assertAlmostEqual(float(solved.mean()), float(np.exp(np.log(z0).mean())), places=5)


class TestPlacementAgainstTheMesh(unittest.TestCase):
    def test_front_depth_keeps_the_nearest_vertex_in_each_bin(self):
        vertices = np.array([
            [0.0, 0.0, 3.0],   # both land in the bin at the principal point
            [0.0, 0.0, 2.0],
            [0.4, 0.0, 5.0],
        ])
        front = ps.mesh_front_depth(vertices, FOCAL, CX, CY, (HEIGHT, WIDTH), 32)
        self.assertEqual(front[int(CY) // 32, int(CX) // 32], 2.0)
        self.assertEqual(np.isfinite(front).sum(), 2)

    def test_depth_scale_recovers_a_known_factor(self):
        z, _, mask = _sphere()
        points = ps.backproject(z, FOCAL, CX, CY)
        vertices = points[mask][::37]              # the "mesh", at true depth
        front = ps.mesh_front_depth(vertices, FOCAL, CX, CY, (HEIGHT, WIDTH), 32)

        scale, stats = ps.depth_scale_to_mesh(z / 2.5, mask, front, 32)

        self.assertAlmostEqual(scale, 2.5, delta=0.05)
        self.assertGreater(stats["bins_compared"], 20)

    def test_the_median_survives_bins_the_mesh_gets_wrong(self):
        z, _, mask = _sphere()
        points = ps.backproject(z, FOCAL, CX, CY)
        vertices = points[mask][::37]
        front = ps.mesh_front_depth(vertices, FOCAL, CX, CY, (HEIGHT, WIDTH), 32)
        # A third of the bins report nonsense — hair and clothing the body
        # model does not have, or a limb the fit misplaced.
        finite = np.flatnonzero(np.isfinite(front.reshape(-1)))
        flat = front.reshape(-1)
        flat[finite[::3]] *= 4.0

        scale, _ = ps.depth_scale_to_mesh(z / 2.5, mask, front.reshape(front.shape), 32)
        self.assertAlmostEqual(scale, 2.5, delta=0.1)

    def test_it_refuses_a_mesh_that_does_not_overlap_the_matte(self):
        _, _, mask = _sphere()
        far_away = np.array([[10.0, 10.0, 3.0]])
        front = ps.mesh_front_depth(far_away, FOCAL, CX, CY, (HEIGHT, WIDTH), 32)
        with self.assertRaises(ValueError):
            ps.depth_scale_to_mesh(np.ones((HEIGHT, WIDTH)), mask, front, 32)


class TestGaussians(unittest.TestCase):
    def test_quaternions_round_trip_including_a_trace_of_minus_one(self):
        rng = np.random.default_rng(2)
        matrices = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(200)]
        matrices = [m if np.linalg.det(m) > 0 else m[:, ::-1] for m in matrices]
        # The case the naive trace-only formula loses: 180 deg about X,
        # which is exactly what FLIP does to a camera-facing normal.
        matrices.append(np.diag([1.0, -1.0, -1.0]))
        matrices.append(np.diag([-1.0, 1.0, -1.0]))
        rotations = np.stack(matrices)

        back = _quat_to_mat(ps.mat_to_quat_wxyz(rotations))
        self.assertLess(float(np.abs(back - rotations).max()), 1e-12)

    def test_the_normal_is_column_two_and_scale_two_is_the_smallest(self):
        z, normals, mask = _sphere()
        xyz = ps.backproject(z, FOCAL, CX, CY)
        rgb = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
        alpha = mask.astype(np.float32)

        g = ps.build_gaussians(xyz, normals, rgb, alpha, mask, FOCAL, cliff_k=0.0)

        frames = _quat_to_mat(g["quats"])
        # Column 2 is the surface normal, in world (flipped) coordinates.
        self.assertGreater(float(np.abs((frames[:, :, 2] * g["normals"]).sum(1)).min()), 1 - 1e-6)
        self.assertTrue(np.all(g["log_scales"][:, 2] < g["log_scales"][:, 0]))
        self.assertTrue(np.all(g["log_scales"][:, 2] < g["log_scales"][:, 1]))
        # Discs, not slivers: the thin axis is splat_thickness of the base.
        self.assertLess(float(np.abs(np.exp(g["log_scales"][:, 2] - g["log_scales"][:, 1])
                                     - 0.15).max()), 1e-9)

    def test_the_world_frame_flips_y_and_z_and_is_not_recentred(self):
        """masktest subtracts the centroid; this pipeline must not — the
        world origin has to stay at SAM-3D-Body's camera."""
        mask = np.zeros((HEIGHT, WIDTH), bool)
        mask[int(CY) - 2:int(CY) + 3, int(CX) - 2:int(CX) + 3] = True
        z = np.where(mask, 2.0, 0.0)
        normals = np.zeros((HEIGHT, WIDTH, 3))
        normals[..., 2] = -1.0
        xyz = ps.backproject(z, FOCAL, CX, CY)
        rgb = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)

        g = ps.build_gaussians(xyz, normals, rgb, mask.astype(np.float32), mask,
                               FOCAL, cliff_k=0.0)

        centre = g["means"][len(g["means"]) // 2]
        self.assertAlmostEqual(float(centre[0]), 0.0, places=9)
        self.assertAlmostEqual(float(centre[1]), 0.0, places=9)
        self.assertAlmostEqual(float(centre[2]), -2.0, places=9)   # Z flipped
        self.assertGreater(abs(float(g["means"][:, 2].mean())), 1.0)

    def test_the_cliff_cull_drops_a_depth_discontinuity(self):
        z, normals, mask = _sphere()
        xyz = ps.backproject(z, FOCAL, CX, CY)
        rgb = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
        alpha = mask.astype(np.float32)

        kept = len(ps.build_gaussians(xyz, normals, rgb, alpha, mask, FOCAL,
                                      cliff_k=0.0)["means"])
        # Push one column of the sphere half a metre back: every pixel along
        # the two seams it creates is now a sheet through empty space.
        torn = xyz.copy()
        torn[:, WIDTH // 2:, 2] += 0.5
        culled = len(ps.build_gaussians(torn, normals, rgb, alpha, mask, FOCAL,
                                        cliff_k=8.0)["means"])
        self.assertLess(culled, kept)
        self.assertGreater(culled, 0.9 * kept)   # a seam, not the subject


class TestMasks(unittest.TestCase):
    def test_nothing_is_filled_by_default(self):
        """An RMBG matte of a whole body has no holes to fill, so every one
        left in it is real background — an arm raised over the head encloses
        a loop of it, and filling that hangs a slab of Gaussians behind the
        subject in every novel view."""
        matte = np.zeros((200, 200), np.float32)
        matte[20:180, 20:180] = 1.0
        # 20x20, wider than close_iters can bridge, so this is the hole
        # filling's decision and not the morphology's.
        matte[90:110, 135:155] = 0.0       # ~2% of the subject
        matte[40:160, 60:120] = 0.0        # ~28%: an arm-to-torso gap

        mask = ps.clean_mask(matte)

        self.assertFalse(mask[100, 145])
        self.assertFalse(mask[100, 90])

    def test_filling_can_be_turned_back_on_for_a_seg_derived_matte(self):
        """Where the original argument does apply — a head crop whose eyes
        and lips segment out separately — the knob still bounds it: the
        small hole fills, the limb-sized gap does not."""
        matte = np.zeros((200, 200), np.float32)
        matte[20:180, 20:180] = 1.0
        matte[90:110, 135:155] = 0.0
        matte[40:160, 60:120] = 0.0

        mask = ps.clean_mask(matte, fill_max_frac=0.05)

        self.assertTrue(mask[100, 145])
        self.assertFalse(mask[100, 90])

    def test_a_detached_component_survives_if_it_is_big_enough(self):
        matte = np.zeros((200, 200), np.float32)
        matte[20:180, 20:120] = 1.0        # torso
        matte[60:110, 150:190] = 1.0       # a hand, disconnected
        matte[5:8, 5:8] = 1.0              # speckle

        mask = ps.clean_mask(matte)

        self.assertTrue(mask[80, 170])
        self.assertFalse(mask[6, 6])

    def test_soft_alpha_pins_the_interior_and_keeps_the_edge_soft(self):
        matte = np.zeros((120, 120), np.float32)
        matte[20:100, 20:100] = 0.7        # a legitimately dim interior
        mask = matte >= 0.5

        alpha = ps.soft_alpha(matte, mask, erode=3)

        self.assertAlmostEqual(float(alpha[60, 60]), 1.0)
        self.assertAlmostEqual(float(alpha[20, 60]), 0.7, places=6)
        self.assertEqual(float(alpha[0, 0]), 0.0)


class TestPly(unittest.TestCase):
    def test_the_written_ply_is_the_layout_brush_reads(self):
        rng = np.random.default_rng(3)
        gaussians = {
            "means": rng.normal(size=(7, 3)),
            "sh_dc": rng.normal(size=(7, 3)),
            "opacity": rng.normal(size=7),
            "log_scales": rng.normal(size=(7, 3)),
            "quats": np.tile([1.0, 0.0, 0.0, 0.0], (7, 1)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "shell.ply"
            ps.write_ply(path, gaussians)
            head, props, data = _read_ply(path)

        self.assertIn("format binary_little_endian 1.0", head)
        self.assertIn("comment SH degree: 0", head)
        self.assertIn("comment Vertical axis: y", head)
        self.assertEqual(props, list(ps.PLY_PROPS))
        self.assertEqual(data.shape, (7, 14))
        np.testing.assert_allclose(data[:, 0:3], gaussians["means"], rtol=1e-6)
        np.testing.assert_allclose(data[:, 7:10], gaussians["log_scales"], rtol=1e-6)
        np.testing.assert_allclose(data[:, 10], 1.0)


class TestStepRun(unittest.TestCase):
    """The whole step, with only the 1B forward pass stubbed out."""

    def _run(self, tmp: str, scale_the_pointmap_by: float = 2.5,
             pointmap_camera=(FOCAL, CX, CY), camera=None, given_camera=None,
             **params):
        z, normals, mask = _sphere()
        # The "mesh": SAM-3D-Body vertices sampled off the true surface,
        # with cam_t already folded out so vertices + cam_t is the truth.
        # With `camera`, the truth is that same surface as seen from THAT
        # camera, expressed in SAM-3D-Body's frame the way `mesh_in_world`
        # undoes it — the posed route's world is body2colmap's.
        points = ps.backproject(z, FOCAL, CX, CY)
        cam_t = np.array([0.02, -0.03, 0.05])
        surface = points[mask][::29]
        if camera is not None:
            # With a given camera too, the truth sits where the PHOTOGRAPH's
            # camera ends up after the refinement's delta — not where the
            # dataset camera is; that is the whole point of the input.
            rotation, position = (ps.refined_photo_pose(camera, given_camera)
                                  if given_camera is not None else ps.camera_pose(camera))
            surface = ((surface * ps.FLIP) @ rotation.T + position) * ps.FLIP
        mesh_output = {
            "vertices": surface - cam_t,
            "cam_t": cam_t,
            "focal_length": FOCAL,
        }
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        image[..., 2] = 255                              # BGR: a red subject
        # The pointmap the stub returns is at a different metric scale and
        # carries a fine ripple its own normals contradict — the two things
        # the step has to fix.
        vv, uu = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        ripple = 1.0 + 0.02 * np.sin(2 * np.pi * uu / 8) * np.sin(2 * np.pi * vv / 8)
        z_stub = np.where(mask, z * ripple, 0.0) / scale_the_pointmap_by
        # Through the network's own camera. By default the same pinhole
        # the normals were made in, so the surface the "network" reports is
        # the sphere and the integration can be judged; the placement test
        # hands in a different one.
        pointmap = ps.backproject(z_stub, *pointmap_camera)

        step = _StubbedStep(pointmap.astype(np.float32))
        merged = dict({"filepath": str(Path(tmp) / "shell.ply")}, **params)
        inputs = {
            "image": image,
            "mask": mask.astype(np.float32),
            "normal_map": normals * ps.NORMAL_TO_CAMERA_FRAME,   # OpenGL in
            "mesh_output": mesh_output,
        }
        if camera is not None:
            inputs.update({"cameras": [camera], "anchor_frame_index": 0})
        if given_camera is not None:
            inputs["given_camera"] = given_camera
        return step.run(inputs, ps.PointmapSplatStep.resolve_params(merged)), (z, mask)

    def test_debug_dir_gets_the_stats_beside_the_pictures(self):
        """`stats.json` next to the mask/depth visualisations: the numbers a
        placement question is answered from — which camera the shell was
        built through, at what depth — as the record, not the log's summary
        of it. They ride into the result .zip under debug/."""
        import json

        with tempfile.TemporaryDirectory() as tmp:
            debug = Path(tmp) / "debug" / "face_splat"
            result, _ = self._run(tmp, debug_dir=str(debug))

            for name in ("mask.png", "depth.png", "normal.png", "stats.json"):
                self.assertTrue((debug / name).is_file(), name)
            with open(debug / "stats.json", encoding="utf-8") as handle:
                dumped = json.load(handle)
        self.assertEqual(dumped["n_splats"], result["splat_stats"]["n_splats"])
        self.assertEqual(dumped["source_camera"]["frame_index"], None)
        self.assertIn("depth_m", dumped["placement"])

    def test_run_puts_every_gaussian_back_on_its_own_pixel(self):
        """The gate: the shell must reproject onto the photo it came from,
        through SAM-3D-Body's camera, not the pointmap's."""
        with tempfile.TemporaryDirectory() as tmp:
            result, (z_true, mask) = self._run(tmp)

            _, _, data = _read_ply(Path(result["splat_path"]))
            means_world = data[:, 0:3].astype(np.float64)
            cam = means_world * ps.FLIP                  # back to the camera frame
            u = FOCAL * cam[:, 0] / cam[:, 2] + CX
            v = FOCAL * cam[:, 1] / cam[:, 2] + CY

            # Every Gaussian lands on an integer pixel of the masked region.
            self.assertLess(float(np.abs(u - np.round(u)).max()), 1e-3)
            self.assertLess(float(np.abs(v - np.round(v)).max()), 1e-3)
            self.assertTrue(mask[np.round(v).astype(int), np.round(u).astype(int)].all())

    def test_run_through_a_dataset_camera_lands_on_that_cameras_pixels(self):
        """The posed route, which exists for `face_splat_refined`: handed
        `cameras` and an index, the shell is the photograph's pixels on THAT
        camera's rays at the mesh's depth. So the same gate as above, read
        through the posed camera — and the depth scale still recovered from
        the mesh, which is the half a rigid carry of the origin's shell got
        wrong (the 50 mm of 2026-09-02)."""
        from body2colmap import coordinates
        from body2colmap.camera import Camera

        target = np.array([0.4, -1.1, -2.0])
        camera = Camera(
            focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
            principal_point=(CX, CY),
            position=target + coordinates.spherical_to_cartesian(2.2, 37.0, 9.0),
        )
        camera.look_at(target, coordinates.WorldCoordinates.UP_AXIS)
        # The given anchor here is the photograph's own camera, so the
        # refinement delta IS the dataset camera and the shell sits on its rays.
        plain = Camera(focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
                       principal_point=(CX, CY), position=np.zeros(3))

        with tempfile.TemporaryDirectory() as tmp:
            result, (z_true, mask) = self._run(tmp, camera=camera, given_camera=plain)

            _, _, data = _read_ply(Path(result["splat_path"]))
            means_world = data[:, 0:3].astype(np.float64)
            rotation, position = ps.camera_pose(camera)
            cam = ((means_world - position) @ rotation) * ps.FLIP
            u = FOCAL * cam[:, 0] / cam[:, 2] + CX
            v = FOCAL * cam[:, 1] / cam[:, 2] + CY
            self.assertLess(float(np.abs(u - np.round(u)).max()), 1e-3)
            self.assertLess(float(np.abs(v - np.round(v)).max()), 1e-3)
            self.assertTrue(mask[np.round(v).astype(int), np.round(u).astype(int)].all())
            # ...and NOT on the origin camera's: the shell moved with the pose.
            origin = means_world * ps.FLIP
            u0 = FOCAL * origin[:, 0] / np.where(origin[:, 2] > 0, origin[:, 2], 1) + CX
            self.assertGreater(float(np.median(np.abs(u0 - u))), 10.0)

            stats = result["splat_stats"]
            self.assertAlmostEqual(stats["depth_alignment"]["scale"], 2.5, delta=0.1)
            self.assertAlmostEqual(stats["placement"]["depth_m"],
                                   float(np.median(z_true[mask])), delta=0.03)
            self.assertEqual(stats["source_camera"]["frame_index"], 0)
            np.testing.assert_allclose(stats["source_camera"]["position"], position, atol=1e-6)

    @staticmethod
    def _anchor_pair(tilt_target, delta_deg, delta_mm):
        """A look_at-turned anchor at the origin, and it after a refinement.

        `given` is what render's override mode builds: SAM-3D-Body's
        position, turned onto an orbit target off the optical axis.
        `refined` is that camera moved by a small rigid delta, the way a
        bundle adjustment leaves it.
        """
        from body2colmap import coordinates
        from body2colmap.camera import Camera

        given = Camera(focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
                       principal_point=(CX, CY), position=np.zeros(3))
        given.look_at(np.asarray(tilt_target, np.float32), coordinates.WorldCoordinates.UP_AXIS)
        angle = np.radians(delta_deg)
        axis = np.array([0.6, 0.8, 0.0])
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        delta_r = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
        r_g, p_g = ps.camera_pose(given)
        refined = Camera(focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
                         principal_point=(CX, CY),
                         position=(p_g + np.array(delta_mm) / 1000.0).astype(np.float32))
        refined.rotation = (delta_r @ r_g).astype(np.float32)
        return given, refined

    def test_refined_photo_pose_is_the_delta_applied_to_the_identity(self):
        """T = refined o given^-1 on the pose the photograph was taken from:
        when the given anchor is at the origin, the result's position is the
        refined one and its rotation is the delta alone — the look_at tilt
        drops out."""
        given, refined = self._anchor_pair([0.03, 0.1, -2.1], 1.3, [-12.0, 16.0, -24.0])
        rotation, position = ps.refined_photo_pose(refined, given)

        np.testing.assert_allclose(position, ps.camera_pose(refined)[1], atol=1e-6)
        self.assertAlmostEqual(ps.rotation_angle_deg(rotation), 1.3, places=3)
        # The tilt is real (it is what the fix subtracts)...
        self.assertGreater(ps.rotation_angle_deg(ps.camera_pose(given)[0]), 2.0)
        # ...and an unturned given anchor makes the pose the refined camera.
        from body2colmap.camera import Camera
        plain = Camera(focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
                       principal_point=(CX, CY), position=np.zeros(3))
        rotation, position = ps.refined_photo_pose(refined, plain)
        np.testing.assert_allclose(rotation, ps.camera_pose(refined)[0], atol=1e-6)

    def test_run_with_the_given_anchor_ignores_its_lookat_tilt(self):
        """The 5e2817 bug (2026-09-04). The path's anchor is look_at-turned
        onto an orbit target off the photograph's axis; hung on that camera
        the shell turns with it — 0.83 deg, 28 mm at the face, in the run.
        With the given anchor handed in, the Gaussians land on the
        photograph's pixels through the photograph's camera moved by the
        refinement's delta, and NOT through the refined dataset camera."""
        given, refined = self._anchor_pair([0.03, 0.1, -2.1], 0.6, [-12.0, 16.0, -24.0])

        with tempfile.TemporaryDirectory() as tmp:
            result, (z_true, mask) = self._run(tmp, camera=refined, given_camera=given)

            _, _, data = _read_ply(Path(result["splat_path"]))
            means_world = data[:, 0:3].astype(np.float64)
            rotation, position = ps.refined_photo_pose(refined, given)
            cam = ((means_world - position) @ rotation) * ps.FLIP
            u = FOCAL * cam[:, 0] / cam[:, 2] + CX
            v = FOCAL * cam[:, 1] / cam[:, 2] + CY
            self.assertLess(float(np.abs(u - np.round(u)).max()), 1e-3)
            self.assertLess(float(np.abs(v - np.round(v)).max()), 1e-3)
            self.assertTrue(mask[np.round(v).astype(int), np.round(u).astype(int)].all())

            # Through the refined dataset camera itself the same pixels are
            # off by the tilt — the error the run measured.
            r_d, p_d = ps.camera_pose(refined)
            cam_d = ((means_world - p_d) @ r_d) * ps.FLIP
            u_d = FOCAL * cam_d[:, 0] / cam_d[:, 2] + CX
            v_d = FOCAL * cam_d[:, 1] / cam_d[:, 2] + CY
            tilt_px = np.radians(ps.rotation_angle_deg(ps.camera_pose(given)[0])) * FOCAL
            self.assertGreater(float(np.median(np.hypot(u_d - u, v_d - v))), 0.5 * tilt_px)

            # The depth is still the mesh's, read through the photograph's pose.
            self.assertAlmostEqual(result["splat_stats"]["placement"]["depth_m"],
                                   float(np.median(z_true[mask])), delta=0.03)
            source = result["splat_stats"]["source_camera"]
            self.assertAlmostEqual(source["refinement_delta_deg"], 0.6, places=2)
            self.assertGreater(source["given_lookat_tilt_deg"], 2.0)
            np.testing.assert_allclose(source["position"], position, atol=1e-6)

    def test_run_refuses_an_anchor_index_off_the_path(self):
        from body2colmap.camera import Camera

        camera = Camera(focal_length=(FOCAL, FOCAL), image_size=(WIDTH, HEIGHT),
                        principal_point=(CX, CY), position=np.zeros(3))
        step = ps.PointmapSplatStep()
        with self.assertRaises(ValueError):
            step._source_camera({"cameras": [camera], "anchor_frame_index": 3}, "t")
        self.assertEqual(step._source_camera({"cameras": []}, "t"), (None, None, None))
        # A given camera on its own is nothing to hang rays on.
        self.assertEqual(step._source_camera({"given_camera": camera}, "t"), (None, None, None))
        self.assertIs(step._source_camera({"cameras": [camera], "given_camera": camera},
                                          "t")[2], camera)
        self.assertEqual(step._source_camera({}, "t"), (None, None, None))
        # A dataset camera without the anchor it was refined from is refused:
        # it carries the path's look_at tilt, the 5e2817 bug.
        with self.assertRaises(ValueError) as caught:
            step._source_camera({"cameras": [camera]}, "t")
        self.assertIn("given_camera", str(caught.exception))

    def test_run_scales_the_shell_onto_the_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, (z_true, mask) = self._run(tmp, scale_the_pointmap_by=2.5)

            stats = result["splat_stats"]
            self.assertAlmostEqual(stats["depth_alignment"]["scale"], 2.5, delta=0.1)
            # ...and the relief the flattened pointmap lost is back.
            self.assertLess(stats["normal_agreement_deg"]["after_median"], 5.0)
            self.assertGreater(stats["normal_agreement_deg"]["before_median"], 50.0)

            _, _, data = _read_ply(Path(result["splat_path"]))
            depth = -data[:, 2].astype(np.float64)
            self.assertAlmostEqual(float(np.median(depth)),
                                   float(np.median(z_true[mask])), delta=0.02)

    def test_run_reports_the_two_cameras_and_the_silhouette(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run(tmp, pointmap_camera=(1400.0, CX + 40, CY - 30))

            stats = result["splat_stats"]
            # The pointmap's own camera is measured (1400 px, off-centre
            # principal point); the mesh's is the one the Gaussians are
            # placed through, and both are reported.
            self.assertAlmostEqual(stats["pointmap_intrinsics"]["f"], 1400.0, delta=1.0)
            self.assertEqual(stats["sam3d_intrinsics"]["f"], FOCAL)
            self.assertAlmostEqual(stats["focal_ratio_sam3d_over_pointmap"],
                                   FOCAL / 1400.0, places=3)
            offset = stats["silhouette"]["centroid_offset_px"]
            self.assertLess(max(abs(offset[0]), abs(offset[1])), 5.0)
            self.assertEqual(result["splat_scene"].sh_degree, 0)
            self.assertEqual(len(result["splat_scene"].means), stats["n_splats"])

    def test_run_keeps_the_networks_shape_through_a_different_camera(self):
        """A crop's pointmap camera has its principal point at the crop's
        centre and its own focal. The Gaussians must still land on the
        pixels' rays in SAM-3D-Body's camera (tested above) — and the
        surface between those rays must keep the shape the network saw,
        not the shape re-reading its depth through the other camera makes:
        the relief scales with the width (k), not with the distance."""
        # A few degrees off axis, like a head near the top of a portrait
        # frame (not more: the relief of an object seen obliquely genuinely
        # changes with the axis it is measured along, and this asserts on
        # the relief the network reported).
        f_net, cx_net, cy_net = 500.0, CX + 20, CY - 45
        with tempfile.TemporaryDirectory() as tmp:
            result, (z_true, mask) = self._run(
                tmp, scale_the_pointmap_by=2.5, pointmap_camera=(f_net, cx_net, cy_net),
                integration_lambda=0.0)          # keep the stub's own shape
            stats = result["splat_stats"]
            placement = stats["placement"]

            # The rotation is the angle between the two principal rays.
            expected = np.degrees(np.arctan(np.hypot(cx_net - CX, cy_net - CY) / f_net))
            self.assertAlmostEqual(placement["rotation_deg"], expected, delta=2.0)
            self.assertGreater(placement["ray_residual_deg"]["median"], 0.0)

            # k = scale * f_net / f_sam, and the depth is the mesh's.
            scale = stats["depth_alignment"]["scale"]
            self.assertAlmostEqual(placement["width_ratio_k"], scale * f_net / FOCAL, places=6)
            self.assertAlmostEqual(placement["depth_m"], float(np.median(z_true[mask])), delta=0.03)

            # Relief-to-width is the network's own. The network's surface is
            # the stub depth backprojected through ITS camera: measure its
            # relief and width there, and compare with the placed splat's.
            z_net = np.where(mask, z_true, 0.0) / 2.5
            net_relief = float(np.ptp(np.percentile(z_net[mask], [2, 98])))
            net_width = float(np.ptp(np.nonzero(mask)[1])) * float(np.median(z_net[mask])) / f_net
            _, _, data = _read_ply(Path(result["splat_path"]))
            cam = data[:, 0:3].astype(np.float64) * ps.FLIP
            placed_relief = float(np.ptp(np.percentile(cam[:, 2], [2, 98])))
            placed_width = float(np.ptp(np.percentile(cam[:, 0], [0.5, 99.5])))
            self.assertAlmostEqual(placed_relief / placed_width, net_relief / net_width,
                                   delta=0.15 * net_relief / net_width)
            # ...whereas scaling the relief by the distance alone would have
            # stretched it by f_sam / f_net = 1.8x.
            self.assertLess(placed_relief / placed_width,
                            1.3 * net_relief / net_width)

    def test_run_refuses_inputs_from_a_different_photo(self):
        """A size mismatch is the shape "these came from different images"
        takes in practice, and the placement argument collapses if it is
        broadcast away instead of refused."""
        step = _StubbedStep(np.zeros((HEIGHT, WIDTH, 3), np.float32))
        with self.assertRaises(ValueError) as caught:
            step.run(
                {
                    "image": np.zeros((HEIGHT, WIDTH, 3), np.uint8),
                    "mask": np.ones((HEIGHT // 2, WIDTH), np.float32),
                    "normal_map": np.zeros((HEIGHT, WIDTH, 3), np.float32),
                    "mesh_output": {"vertices": np.zeros((3, 3)),
                                    "cam_t": np.zeros(3), "focal_length": FOCAL},
                },
                ps.PointmapSplatStep.resolve_params({"filepath": "/dev/null"}),
            )
        self.assertIn("same photo", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
