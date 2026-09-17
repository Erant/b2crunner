"""pipeline/view_atlas.py — the character-sheet layout and its transfer.

The pure helpers run anywhere; the rasteriser needs torch (the wan22 env the
step runs in), so those tests skip where it is absent.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import view_atlas as va


def cube_atlas(root: Path, res: int = 256):
    """A unit cube with a six-square atlas (no overlaps), a flat grey texture and a protect mask on one face."""
    import cv2

    V = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], float)
    V[:, 1] += 1.0  # the "head" is the top face
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces, uvs = [], []
    for q, (a, b, c, d) in enumerate(quads):
        cx, cy = (q % 3) / 3, (q // 3) / 2
        sq = [(cx + .02, cy + .02), (cx + .31, cy + .02), (cx + .31, cy + .48), (cx + .02, cy + .48)]
        faces += [(a, b, c), (a, c, d)]
        uvs += [[sq[0], sq[1], sq[2]], [sq[0], sq[2], sq[3]]]
    F = np.asarray(faces)
    # outward winding
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flip = np.einsum("ij,ij->i", n, t.mean(1) - V.mean(0)) < 0
    F[flip] = F[flip][:, ::-1]
    uv = np.asarray(uvs, float)
    uv[flip] = uv[flip][:, ::-1]
    root.mkdir(parents=True, exist_ok=True)
    va.write_obj(root / "mesh_uv.obj", V, F, uv, "mesh_uv.mtl", [("tex", np.ones(len(F), bool))])
    tex = np.full((res, res, 3), 120, np.uint8)
    tex[:, :, 1] = (np.arange(res)[None, :] * 255 // res).astype(np.uint8)  # a gradient so resampling shows
    cv2.imwrite(str(root / "texture.png"), tex)
    mask = np.zeros((res, res), np.uint8)
    for corners in uv:
        cv2.fillPoly(mask, [np.rint(np.c_[corners[:, 0] * res, (1 - corners[:, 1]) * res]).astype(np.int32)], 255)
    cv2.imwrite(str(root / "mask.png"), mask)
    prot = np.zeros((res, res), np.uint8)
    prot[: res // 2, : res // 3] = 255  # the first chart
    cv2.imwrite(str(root / "protect_cap.png"), prot)
    return V, F, uv


class TestHelpers(unittest.TestCase):
    def test_obj_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            V, F, uv = cube_atlas(Path(tmp))
            V2, F2, uv2 = va.read_obj(Path(tmp) / "mesh_uv.obj")
            np.testing.assert_allclose(V2, V)
            np.testing.assert_array_equal(F2, F)
            np.testing.assert_allclose(uv2, uv, atol=1e-7)

    def test_grid_boxes_tile_the_region(self):
        boxes = va.grid_boxes(6, 2, (100, 0, 700, 400))
        self.assertEqual(len(boxes), 6)
        self.assertEqual(boxes[0], (100, 0, 300, 200))
        self.assertEqual(boxes[5], (500, 200, 700, 400))
        self.assertEqual(va.grid_boxes(0, 2, (0, 0, 10, 10)), [])

    def test_panel_basis_is_orthonormal_and_upright(self):
        front, up = np.array([0, 0, 1.]), np.array([0, 1., 0])
        for d in (np.array([0.6, -0.5, 0.62]), np.array([0, 0.999, 0.04]), np.array([0, -1., 0])):
            d = d / np.linalg.norm(d)
            r, u = va.panel_basis(d, front, up)
            self.assertAlmostEqual(r @ u, 0, places=6)
            self.assertAlmostEqual(r @ d, 0, places=6)
            self.assertAlmostEqual(np.linalg.norm(r), 1, places=6)
            self.assertAlmostEqual(np.linalg.norm(u), 1, places=6)
        r, u = va.panel_basis(np.array([0.6, -0.5, 0.62]) / np.linalg.norm([0.6, -0.5, 0.62]), front, up)
        self.assertGreater(u @ up, 0.8, "a steep view still hangs the body upright")

    def test_sphere_directions_are_unit_and_spread(self):
        d = va.sphere_directions(64)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1, atol=1e-9)
        self.assertLess(np.abs(d.mean(0)).max(), 0.1)

    def test_overlap_pairs_finds_crossing_triangles_and_allows_shared_edges(self):
        tri = np.array([[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]], [[.2, .2], [.8, .2], [.2, .8]]], float)
        pairs = va.overlap_pairs(tri)
        self.assertEqual(pairs.tolist(), [[0, 2]])


class TestLayoutAndTransfer(unittest.TestCase):
    """The rasteriser and the sheets on a cube, on the CPU (the step runs the same code on CUDA)."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in this environment (the step runs in the wan22 env)")

    def test_raster_ids_and_barycentrics(self):
        F = np.array([[0, 1, 2]])
        P = np.array([[1., 1.], [9., 1.], [1., 9.]])
        r = va.Raster(F, "cpu").run(P, np.zeros(3), 10, 10)
        tri = r["tri"].numpy()
        self.assertEqual(tri[2, 2], 0)
        self.assertEqual(tri[8, 8], -1)
        b = r["bary"].numpy()[2, 2]
        self.assertAlmostEqual(b.sum(), 1, places=5)
        self.assertTrue((b >= -1e-5).all())

    def test_layout_owns_every_visible_face_and_transfer_is_exact(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            atlas = Path(tmp) / "atlas"
            V, F, uv = cube_atlas(atlas)
            out = Path(tmp) / "sheets"
            manifest = va.build_layout(atlas, out, res=512, extra=2, head=1, candidates=8, pad=4, device="cpu", vis_res=256, head_height=0.5)
            kinds = [s["kind"] for s in manifest["sheets"]]
            self.assertEqual(kinds[0], "main")
            owners = [np.load(Path(s["dir"]) / "face_panel.npy") for s in manifest["sheets"]]
            owned = np.any([o >= 0 for o in owners], axis=0)
            self.assertTrue(owned.all(), "every cube face is visible from some direction and must be owned")
            self.assertEqual(sum(int((o >= 0).sum()) for o in owners), len(F), "each face on exactly one sheet")
            for s in manifest["sheets"]:
                d = Path(s["dir"])
                for name in ("diffusion_texture.png", "edit_mask.png", "context.png", "atlas.json", "face_uv.npy", "protect_cap.png"):
                    self.assertTrue((d / name).exists(), name)
            self.assertTrue((out / "mesh_diffusion.obj").exists())
            # The unedited sheets transfer nothing in delta form, and something with a real edit.
            source = cv2.imread(str(atlas / "texture.png"))
            mask = cv2.imread(str(atlas / "mask.png"), cv2.IMREAD_GRAYSCALE)
            _, _, olduv = va.read_obj(atlas / "mesh_uv.obj")
            cur = source.copy()
            for s in manifest["sheets"]:
                d = Path(s["dir"])
                sheet = cv2.imread(str(d / "diffusion_texture.png"))
                cur, changed, metrics = va.apply_sheet(d, sheet, cur, sheet, mask, olduv, "cpu")
                self.assertEqual(metrics["edited_texels"], 0)
            np.testing.assert_array_equal(cur, source)
            d = Path(manifest["sheets"][0]["dir"])
            sheet = cv2.imread(str(d / "diffusion_texture.png"))
            edited = sheet.copy()
            edited[..., 2] = np.clip(edited[..., 2].astype(int) + 50, 0, 255)
            result, changed, metrics = va.apply_sheet(d, edited, source, sheet, mask, olduv, "cpu")
            self.assertGreater(metrics["edited_texels"], 0)
            self.assertEqual(metrics["unowned_surface_max_error"], 0.0)
            diff = result.astype(int) - source.astype(int)
            self.assertTrue((diff[changed > 0][:, 2] > 40).all(), "the delta lands on the owned texels")
            self.assertTrue((diff[changed > 0][:, :2] == 0).all(), "and only in the edited channel")


if __name__ == "__main__":
    unittest.main()
