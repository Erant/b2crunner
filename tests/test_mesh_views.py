"""steps/mesh_views.py — the textured mesh rendered at the helix as pass 2's frames."""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class TestHelpers(unittest.TestCase):
    def test_composite_over_grey(self):
        from pipeline.steps.mesh_views import composite_over_grey

        rgba = np.zeros((2, 2, 4), np.uint8)
        rgba[0, 0] = (10, 20, 30, 255)
        rgba[0, 1] = (10, 20, 30, 0)
        rgba[1, 0] = (10, 20, 30, 128)
        out = composite_over_grey(rgba, 0.5)
        self.assertEqual(out[0, 0].tolist(), [10, 20, 30])
        self.assertEqual(out[0, 1].tolist(), [128, 128, 128])
        self.assertTrue(np.all(np.abs(out[1, 0].astype(int) - np.array([69, 74, 79])) <= 1))

    def test_pick_texture_prefers_klein_then_photo_then_bake(self):
        from pipeline.steps.mesh_views import pick_texture

        with tempfile.TemporaryDirectory() as tmp:
            atlas = Path(tmp)
            (atlas / "texture.png").write_bytes(b"x")
            self.assertEqual(pick_texture(atlas, None, None), atlas / "texture.png")
            photo = atlas / "texture_photo.png"
            photo.write_bytes(b"x")
            self.assertEqual(pick_texture(atlas, str(atlas / "missing.png"), str(photo)), photo)
            klein = atlas / "texture_final.png"
            klein.write_bytes(b"x")
            self.assertEqual(pick_texture(atlas, str(klein), str(photo)), klein)


class TestStep(unittest.TestCase):
    FAKE = r'''#!/usr/bin/env python3
import sys, json, os
import numpy as np, cv2
argv = sys.argv[1:]
def after(flag): return argv[argv.index(flag) + 1]
assert argv[0] == "mesh-render", argv
cams = json.load(open(after("--cameras")))
out = after("--output"); os.makedirs(out, exist_ok=True)
w, h = cams["width"], cams["height"]
json.dump(dict(texture=after("--texture"), bg=after("--bg"), ss=after("--ss")), open(os.environ["FAKE_LOG"], "w"))
for i, c in enumerate(cams["cameras"]):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = (10 * i, 100, 200)
    rgba[h // 4: 3 * h // 4, w // 4: 3 * w // 4, 3] = 255
    cv2.imwrite(os.path.join(out, c["name"]), rgba)
'''

    def test_frames_and_masks_from_the_render(self):
        from unittest import mock

        from pipeline.steps import mesh_views as mv

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake = tmp / "b2ctrain"
            fake.write_text(self.FAKE.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            os.environ["FAKE_LOG"] = str(tmp / "log.json")
            atlas = tmp / "atlas"
            atlas.mkdir()
            (atlas / "mesh_uv.obj").write_text("v 0 0 0\n")
            (atlas / "texture.png").write_bytes(b"x")
            klein = tmp / "texture_final.png"
            klein.write_bytes(b"x")
            cams = [SimpleNamespace(fx=100.0, fy=100.0, cx=16.0, cy=24.0, width=32, height=48, position=[0, 0, float(i)],
                                    rotation=np.eye(3).tolist()) for i in range(3)]
            names = [f"frame_{i + 1:05d}_.png" for i in range(3)]
            params = mv.RenderMeshViewsStep.resolve_params({"trainer_path": str(fake), "output_dir": str(tmp / "views")})
            out = mv.RenderMeshViewsStep().run({"mesh_dir": str(atlas), "cameras": cams, "image_names": names, "texture_path": str(klein)}, params)
            self.assertEqual(len(out["images"]), 3)
            self.assertEqual(out["images"][0].shape, (48, 32, 3))
            self.assertEqual(out["images"][2][24, 16].tolist(), [20, 100, 200], "inside the silhouette: the render's colour")
            self.assertEqual(out["images"][2][0, 0].tolist(), [128, 128, 128], "outside: the grey")
            self.assertEqual(out["masks"][0].dtype, np.float32)
            self.assertEqual(float(out["masks"][0][24, 16]), 1.0)
            self.assertEqual(float(out["masks"][0][0, 0]), 0.0)
            log = json.loads((tmp / "log.json").read_text())
            self.assertEqual(log["texture"], str(klein), "klein's texture is the one rendered")
            self.assertEqual(log["bg"], "0.5")
            cams_json = json.loads((tmp / "views" / "cams_helix.json").read_text())
            self.assertEqual([c["name"] for c in cams_json["cameras"]], names, "the renders carry the frames' names")
            self.assertEqual(out["mesh_views_stats"]["frames"], 3)
            self.assertFalse((tmp / "views" / "renders").exists(), "renders are not kept by default")


if __name__ == "__main__":
    unittest.main()
