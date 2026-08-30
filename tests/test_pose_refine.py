"""refine_pose_to_splat — the geometry around the pose optimisation.

The optimisation itself needs the MHR body model (2.8 GB, gated, and only
importable inside the sam3dbody venv), so it is not exercised here. What IS
tested is everything that decides *what the optimisation is asked to do*,
which is where a silent error would do real damage: the depth buffers, the
anatomical offsets that form the target, and the PLY reader. Those are pure
numpy and are the parts that must not drift.

The load-bearing property is self-consistency: handed a shell that already
agrees with the mesh, the target must ask for no correction at all.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline.steps  # noqa: F401  — registers refine_pose_to_splat
from pipeline.registry import get_step_class
from pipeline.steps import pose_refine as pr

FOCAL, WIDTH, HEIGHT = 900.0, 320, 440
CX, CY = WIDTH / 2.0, HEIGHT / 2.0


def _slab(z=2.0, half=0.35, n=240):
    """A frontal plane of points at depth `z`, centred on the axis."""
    grid = np.linspace(-half, half, n)
    x, y = np.meshgrid(grid, grid)
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1)


def _write_ply(path: Path, means: np.ndarray) -> None:
    """A minimal 14-property binary 3DGS PLY, the layout pointmap_splat writes."""
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(means)}\n"
              + "".join(f"property float {p}\n" for p in
                        ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                         "scale_0", "scale_1", "scale_2",
                         "rot_0", "rot_1", "rot_2", "rot_3"))
              + "end_header\n")
    body = np.zeros((len(means), 14), np.float32)
    body[:, 0:3] = means
    path.write_bytes(header.encode("ascii") + body.tobytes())


class TestZBuffer(unittest.TestCase):
    def test_it_keeps_the_nearest_point_per_pixel(self):
        points = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0], [0.1, 0.0, 5.0]])
        buffer = pr.z_buffer(points, FOCAL, CX, CY, (HEIGHT, WIDTH), dilate=0)
        self.assertAlmostEqual(float(buffer[int(CY), int(CX)]), 2.0, places=5)
        self.assertEqual(int((buffer < 1e5).sum()), 2)

    def test_dilation_closes_gaps_without_moving_the_surface(self):
        # Deliberately sparse — this is the gap-closing behaviour, which is
        # what an 18k-vertex mesh needs and a 480k-Gaussian shell does not.
        points = _slab(z=2.0, n=40)
        tight = pr.z_buffer(points, FOCAL, CX, CY, (HEIGHT, WIDTH), dilate=0)
        loose = pr.z_buffer(points, FOCAL, CX, CY, (HEIGHT, WIDTH), dilate=6)
        self.assertGreater((loose < 1e5).sum(), 3 * (tight < 1e5).sum())
        # A min-filter can only pull the surface nearer, never further, and
        # on a flat slab it must not move it at all.
        covered = loose < 1e5
        self.assertAlmostEqual(float(loose[covered].max()), 2.0, places=5)
        self.assertAlmostEqual(float(loose[covered].min()), 2.0, places=5)

    def test_points_behind_the_camera_are_dropped(self):
        points = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]])
        buffer = pr.z_buffer(points, FOCAL, CX, CY, (HEIGHT, WIDTH), dilate=0)
        self.assertEqual(int((buffer < 1e5).sum()), 1)


class TestTarget(unittest.TestCase):
    """The target must be self-consistent: a shell that already agrees with
    the mesh has to produce no correction."""

    def test_an_agreeing_shell_asks_for_no_change(self):
        surface = _slab(z=2.0)
        mesh = pr.z_buffer(surface, FOCAL, CX, CY, (HEIGHT, WIDTH), 6)
        shell = pr.z_buffer(surface, FOCAL, CX, CY, (HEIGHT, WIDTH), 2)
        joints = np.array([[0.0, 0.0, 2.10], [0.05, 0.05, 2.05]])

        offsets, valid = pr.anatomical_offsets(joints, mesh, shell, FOCAL, CX, CY)

        self.assertTrue(valid.all())
        # offset = how far behind the mesh; wanted = shell + offset. With
        # shell == mesh that is the joint's own depth, i.e. no correction.
        wanted = shell[
            np.round(pr.project(joints, FOCAL, CX, CY)[:, 1]).astype(int),
            np.round(pr.project(joints, FOCAL, CX, CY)[:, 0]).astype(int),
        ] + offsets
        np.testing.assert_allclose(wanted, joints[:, 2], atol=1e-5)

    def test_a_shell_that_is_further_away_pulls_the_joint_back(self):
        mesh = pr.z_buffer(_slab(z=2.0), FOCAL, CX, CY, (HEIGHT, WIDTH), 6)
        shell = pr.z_buffer(_slab(z=2.2), FOCAL, CX, CY, (HEIGHT, WIDTH), 2)
        joints = np.array([[0.0, 0.0, 2.10]])          # 100 mm behind the mesh

        offsets, valid = pr.anatomical_offsets(joints, mesh, shell, FOCAL, CX, CY)

        self.assertTrue(valid.all())
        self.assertAlmostEqual(float(offsets[0]), 0.10, places=4)
        pixel = np.round(pr.project(joints, FOCAL, CX, CY)[0]).astype(int)
        wanted = float(shell[pixel[1], pixel[0]] + offsets[0])
        # The joint keeps its 100 mm of half-thickness, now behind the shell.
        self.assertAlmostEqual(wanted, 2.30, places=4)

    def test_joints_off_the_image_or_off_the_surface_are_excluded(self):
        surface = _slab(z=2.0, half=0.05, n=120)       # a small central patch
        mesh = pr.z_buffer(surface, FOCAL, CX, CY, (HEIGHT, WIDTH), 6)
        shell = pr.z_buffer(surface, FOCAL, CX, CY, (HEIGHT, WIDTH), 2)
        joints = np.array([
            [0.0, 0.0, 2.05],       # on the patch
            [5.0, 0.0, 2.05],       # projects outside the image
            [0.0, -0.30, 2.05],     # inside the image, no surface under it
            [0.0, 0.0, -1.0],       # behind the camera
        ])

        _, valid = pr.anatomical_offsets(joints, mesh, shell, FOCAL, CX, CY)

        self.assertTrue(valid[0])
        self.assertFalse(valid[1:].any())


class TestPlyReader(unittest.TestCase):
    def test_it_reads_centres_back_into_the_camera_frame(self):
        rng = np.random.default_rng(0)
        world = rng.normal(size=(32, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shell.ply"
            _write_ply(path, world.astype(np.float32))
            means = pr.read_splat_means(str(path))
        # The reader must undo the world flip pointmap_splat applied.
        np.testing.assert_allclose(means, world * pr.FLIP, rtol=1e-6)


class TestStepContract(unittest.TestCase):
    def test_it_refuses_a_fit_with_no_pose_parameters(self):
        step = get_step_class("refine_pose_to_splat")()
        step._head = object()          # skip the 2.8 GB load
        with self.assertRaises(ValueError) as caught:
            step.run(
                {"mesh_output": {"vertices": np.zeros((3, 3)), "cam_t": np.zeros(3),
                                 "keypoints_3d": np.zeros((70, 3)),
                                 "focal_length": FOCAL},
                 "splat_path": "/dev/null",
                 "image": np.zeros((HEIGHT, WIDTH, 3), np.uint8)},
                get_step_class("refine_pose_to_splat").resolve_params({}),
            )
        self.assertIn("pose_params", str(caught.exception))

    def test_the_defaults_are_the_measured_ones(self):
        params = get_step_class("refine_pose_to_splat").resolve_params({})
        # 2e-3 diverged at step 500 on the real landscape; 6e-4 converged.
        self.assertEqual(params["learning_rate"], 6e-4)
        self.assertEqual(params["mesh_dilate_px"], 6)


if __name__ == "__main__":
    unittest.main()
