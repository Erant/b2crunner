"""steps/refine_texture.py's pure helpers (the mask logic the loop rests on)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.steps.refine_texture import feather_repaint, fit_reference, repaint_mask, view_order


class TestRepaintMask(unittest.TestCase):
    def test_claims_where_the_view_beats_the_best_or_nothing_painted(self):
        alpha = np.full((4, 4), 255, np.uint8)
        alpha[0, :] = 0                       # background row
        cos = np.ones((4, 4), np.float32)
        dens = np.full((4, 4), 2.0, np.float32)
        best = np.zeros((4, 4), np.float32)
        best[1, :] = 10.0                     # painted better already
        best[2, :] = np.inf                   # protected
        best[3, 0] = 1.0                      # weight 2 > 1.5 * 1
        claim, painted = repaint_mask(alpha, cos, dens, best, power=4.0, gain=1.5, dilate=0)
        self.assertFalse(claim[0].any())
        self.assertFalse(claim[1].any())
        self.assertFalse(claim[2].any())
        self.assertTrue(claim[3].all())
        self.assertTrue((painted == claim).all())

    def test_dilation_stays_inside_the_subject(self):
        alpha = np.zeros((9, 9), np.uint8)
        alpha[2:7, 2:7] = 255
        cos = np.ones((9, 9), np.float32)
        dens = np.ones((9, 9), np.float32)
        best = np.full((9, 9), np.inf, np.float32)
        best[4, 4] = 0.0
        claim, painted = repaint_mask(alpha, cos, dens, best, 4.0, 1.5, dilate=3)
        self.assertEqual(int(claim.sum()), 1)
        self.assertEqual(int(painted.sum()), 25)  # the 7x7 ring clipped to the 5x5 subject
        self.assertFalse(painted[alpha == 0].any())


class TestFeather(unittest.TestCase):
    def test_outside_the_mask_the_render_is_kept_exactly(self):
        render = np.full((32, 32, 3), 100, np.uint8)
        repaint = np.full((32, 32, 3), 200, np.uint8)
        alpha = np.full((32, 32), 255, np.uint8)
        mask = np.zeros((32, 32), np.uint8)
        mask[8:24, 8:24] = 255
        out = feather_repaint(repaint, render, mask, alpha, feather=4)
        self.assertEqual(int(out[0, 0, 0]), 100)
        self.assertGreaterEqual(int(out[16, 16, 0]), 195)  # the blur reaches the centre of a 16 px mask faintly
        self.assertTrue(100 < int(out[9, 16, 0]) <= 200)  # the edge blends
        out0 = feather_repaint(repaint, render, mask, alpha, feather=0)
        self.assertEqual(int(out0[8, 8, 0]), 200)


class TestOrderAndRefs(unittest.TestCase):
    def test_head_first_by_default(self):
        order = view_order([0, 180], [0, 45], True)
        self.assertEqual(order, [("head", 0.0), ("head", 45.0), ("body", 0.0), ("body", 180.0)])
        self.assertEqual(view_order([0], [0], False)[0][0], "body")

    def test_reference_fits_the_pixel_budget_in_multiples_of_16(self):
        im = np.zeros((1536, 768, 3), np.uint8)
        out = fit_reference(im, 300000)
        self.assertLessEqual(out.shape[0] * out.shape[1], 300000)
        self.assertEqual(out.shape[0] % 16, 0)
        self.assertEqual(out.shape[1] % 16, 0)
        self.assertEqual(fit_reference(np.zeros((64, 48, 3), np.uint8), 300000).shape[:2], (64, 48))


if __name__ == "__main__":
    unittest.main()


class TestStep(unittest.TestCase):
    """The loop against a fake trainer and a fake klein: what is protected, what is kept, what is cleaned up."""

    FAKE = r'''#!/usr/bin/env python3
import sys, json, os, shutil
import numpy as np
argv = sys.argv[1:]
def after(flag): return argv[argv.index(flag) + 1]
sub = argv[0]
if sub == "mesh-render" and "--make-cams" in argv:
    kind = "head" if "--head" in argv else "body"
    w, h = int(after("--width")), int(after("--height"))
    cams = [{"name": f"{kind}_e+00_a{int(a):03d}.png", "rotation": [[1,0,0],[0,1,0],[0,0,1]], "position": [0,0,0],
             "fx": 1.0, "fy": 1.0, "cx": w/2, "cy": h/2} for a in after("--azims").split(",")]
    json.dump({"width": w, "height": h, "cameras": cams}, open(after("--output"), "w"))
elif sub == "mesh-render":
    cam = json.load(open(after("--cameras")))
    w, h = cam["width"], cam["height"]
    out = after("--output"); os.makedirs(out, exist_ok=True)
    stem = cam["cameras"][0]["name"][:-4]
    import cv2
    rgba = np.zeros((h, w, 4), np.uint8); rgba[..., :3] = 100; rgba[h//4:3*h//4, w//4:3*w//4, 3] = 255
    cv2.imwrite(os.path.join(out, stem + ".png"), rgba)
    np.ones((h, w), np.float32).tofile(os.path.join(out, stem + ".cos.f32"))
    np.ones((h, w), np.float32).tofile(os.path.join(out, stem + ".dens.f32"))
    np.ones((h, w), np.float32).tofile(os.path.join(out, stem + ".depth.f32"))
    best = np.fromfile(after("--aux"), np.float32)
    aux = np.zeros((h, w), np.float32)
    # the top-left quarter of the subject reads the protected texels
    aux[h//4:h//2, w//4:w//2] = best.max()
    aux.tofile(os.path.join(out, stem + ".aux.f32"))
elif sub == "mesh-backproject":
    if after("--texture") != after("--output"): shutil.copy(after("--texture"), after("--output"))
    open(os.environ["FAKE_LOG"], "a").write(json.dumps(argv) + "\n")
else:
    sys.exit("unexpected " + sub)
'''

    def test_loop_outputs_and_cleanup(self):
        import json
        import os
        import stat
        import sys
        import tempfile
        from pathlib import Path
        from unittest import mock

        import cv2

        from pipeline.steps import refine_texture as rt

        class FakeKlein:
            calls = []

            def __init__(self, *a, **k):
                pass

            def encode(self, prompt):
                return prompt

            def set_references(self, refs):
                FakeKlein.refs = len(refs)

            def repaint(self, render_rgb, mask, prompt, strength, steps, guidance, seed):
                FakeKlein.calls.append((prompt, int((mask > 0).sum())))
                return np.full_like(render_rgb, 200)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rt, "Klein", FakeKlein):
            tmp = Path(tmp)
            fake = tmp / "b2ctrain"
            # the fake renders with numpy / cv2: it runs on this interpreter
            fake.write_text(self.FAKE.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            os.environ["FAKE_LOG"] = str(tmp / "bp.jsonl")
            atlas = tmp / "atlas"
            atlas.mkdir()
            (atlas / "mesh_uv.obj").write_text("mtllib mesh_uv.mtl\nv 0 0 0\n")
            cv2.imwrite(str(atlas / "texture.png"), np.full((64, 64, 3), 50, np.uint8))
            cv2.imwrite(str(atlas / "mask.png"), np.full((64, 64), 255, np.uint8))
            prot = np.zeros((64, 64), np.uint8)
            prot[:16, :16] = 255
            cv2.imwrite(str(atlas / "protect_cap.png"), prot)
            params = rt.RefineTextureStep.resolve_params({"trainer_path": str(fake), "output_dir": str(tmp / "mesh"), "debug_dir": str(tmp / "dbg"),
                                                          "body_size": [64, 128], "head_size": 64, "body_azimuths": [0, 180], "head_azimuths": [0],
                                                          "min_repaint": 1})
            front = np.zeros((32, 32, 3), np.uint8)
            result = rt.RefineTextureStep().run({"mesh_dir": str(atlas), "mesh_stats": {"head_centre": [0, 0.6, -2]}, "front_image": front,
                                                 "back_image": front, "caption": "a tester"}, params)
            out = tmp / "mesh"
            stats = result["refine_texture_stats"]
            self.assertEqual(stats["protected_texels"], 256)
            self.assertEqual(stats["face_policy"], "protect_cap")
            self.assertEqual([v["kind"] for v in stats["views"]], ["head", "body", "body"])  # head first
            self.assertEqual(FakeKlein.refs, 2)
            self.assertEqual(len(FakeKlein.calls), 3)
            self.assertIn("a tester", FakeKlein.calls[0][0])
            # the protected quarter of the subject is never in a repaint mask: the head view's
            # subject is 32x32, its protected quarter 16x16 (dilation 8 px reaches into it)
            head = stats["views"][0]
            self.assertLess(head["claimed_px"], head["subject_px"])
            self.assertEqual(head["claimed_px"], 32 * 32 - 16 * 16)
            # outputs, and the working files gone
            self.assertTrue((out / "texture_final.png").is_file())
            self.assertEqual((out / "mesh_klein.mtl").read_text().split("\n")[2], "map_Kd texture_final.png")
            self.assertTrue((out / "mesh_klein.obj").read_text().startswith("mtllib mesh_klein.mtl"))
            self.assertTrue((out / "refine_texture.json").is_file())
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["mesh_klein.mtl", "mesh_klein.obj", "refine_texture.json", "texture_final.png"])
            # the debug directory has each view's PNGs and nothing else
            views = sorted(p.name for p in (tmp / "dbg").iterdir())
            self.assertEqual(views, ["00_head_e+00_a000", "01_body_e+00_a000", "02_body_e+00_a180"])
            self.assertEqual(sorted(p.name for p in (tmp / "dbg" / views[0]).iterdir()), ["refined.png", "render.png", "repaint_mask.png"])
            # every backproject was told the bake mask and the best map
            bp = [json.loads(line) for line in (tmp / "bp.jsonl").read_text().splitlines()]
            self.assertEqual(len(bp), 3)
            self.assertTrue(all("--mask-dir" in c and "--best" in c for c in bp))
