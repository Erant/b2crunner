"""steps/mesh_views.py — render_subject: the textured mesh as pass 2's frames, at the splat path's cameras."""

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
    def test_composite_over(self):
        from pipeline.steps.mesh_views import composite_over

        rgba = np.zeros((2, 2, 4), np.uint8)
        rgba[0, 0] = (10, 20, 30, 255)
        rgba[0, 1] = (10, 20, 30, 0)
        rgba[1, 0] = (10, 20, 30, 128)
        out = composite_over(rgba, (0.5, 0.5, 0.5))
        self.assertEqual(out[0, 0].tolist(), [10, 20, 30])
        self.assertEqual(out[0, 1].tolist(), [128, 128, 128])
        self.assertTrue(np.all(np.abs(out[1, 0].astype(int) - np.array([69, 74, 79])) <= 1))

    def test_blur_within_stays_inside_the_alpha(self):
        """The colour is smoothed inside the silhouette, the surround never bleeds in, and the alpha is untouched."""
        from pipeline.steps.mesh_views import blur_within

        rgba = np.zeros((60, 60, 4), np.uint8)
        rgba[10:50, 10:50, 3] = 255
        rgba[10:50, 10:50, :3] = 200
        rgba[25:35, 25:35, :3] = 0  # a dark square inside the subject; outside the alpha the colour is black too
        out = blur_within(rgba, 3.0)
        self.assertTrue(np.array_equal(out[..., 3], rgba[..., 3]), "the alpha is the rasteriser's")
        self.assertEqual(int(out[11, 11, 0]), 200, "a subject pixel by the silhouette does not darken toward the transparent surround")
        self.assertTrue(0 < int(out[30, 30, 0]) < 200 and 0 < int(out[25, 25, 0]) < 200, "the dark square is smoothed")
        self.assertTrue(np.array_equal(blur_within(rgba, 0.0), rgba), "0 = untouched")

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


class TestSubjectFromMesh(unittest.TestCase):
    """render_subject's mesh path against a fake mesh-render: the frames, the masks, the names, the texture, the colour."""

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

    def test_render_frames_from_the_mesh(self):
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
            params = mv.RenderSubjectStep.resolve_params({"from_mesh": True, "trainer_path": str(fake), "mesh_render_dir": str(tmp / "views"), "confidence": True})
            step = mv.RenderSubjectStep()
            confidence = SimpleNamespace(cull_color=(0.5, 0.5, 0.5))
            images, masks = step._render_frames({"mesh_dir": str(atlas), "mesh_texture_path": str(klein)}, params, scene=None, splat_path=None, cameras=cams,
                                                image_names=names, width=32, height=48, bg_color=(0.0, 0.0, 0.0), render_path="unused", confidence=confidence, sh_degree=2)
            self.assertEqual(len(images), 3)
            self.assertEqual(images[0].shape, (48, 32, 3))
            self.assertEqual(images[2][24, 16].tolist(), [20, 100, 200], "inside the silhouette: the render's colour")
            self.assertEqual(images[2][0, 0].tolist(), [128, 128, 128], "outside: the confidence mode's cull grey")
            self.assertEqual(masks[0].dtype, np.float32)
            self.assertEqual(float(masks[0][24, 16]), 1.0)
            self.assertEqual(float(masks[0][0, 0]), 0.0)
            log = json.loads((tmp / "log.json").read_text())
            self.assertEqual(log["texture"], str(klein), "klein's texture is the one rendered")
            self.assertEqual(log["bg"], "0.5000")
            cams_json = json.loads((tmp / "views" / "cams.json").read_text())
            self.assertEqual([c["name"] for c in cams_json["cameras"]], names, "the renders carry the frames' names")
            self.assertFalse((tmp / "views" / "renders").exists(), "renders are not kept by default")

    def test_off_it_is_render_splat(self):
        from pipeline.steps import mesh_views as mv
        from pipeline.steps.splat import RenderSplatStep

        self.assertTrue(issubclass(mv.RenderSubjectStep, RenderSplatStep))
        names = {p.name for p in mv.RenderSubjectStep.PARAMS}
        self.assertTrue({"from_mesh", "pattern", "override_cam_from_mesh", "sh_degree", "confidence"} <= names)
        self.assertFalse(mv.RenderSubjectStep.resolve_params({})["from_mesh"], "a bare render_subject is a splat render")
        self.assertEqual(mv.RenderSubjectStep.resolve_params({})["mesh_blur_px"], 0.0, "the step itself does not blur; the workflow asks for it")

    def test_the_workflow_blurs_the_mesh_frames_instead_of_klein(self):
        import yaml

        wf = yaml.safe_load((Path(__file__).resolve().parents[1] / "pipeline" / "workflows" / "helical.yaml").read_text())
        settings = {g["name"]: g for g in wf["settings"]}
        self.assertFalse(settings["refine_texture"]["default"], "the klein pass is off by default")
        self.assertEqual(settings["mesh_blur"]["default"], 2.0)
        render = next(s for s in wf["steps"] if s.get("id") == "render_subject")
        self.assertEqual(render["params"]["mesh_blur_px"], "${globals.mesh_blur}")


if __name__ == "__main__":
    unittest.main()
