"""render_mesh_views — the textured mesh rendered at the helix, as pass 2's frames.

The second denoise conditions on frames along the helical path. Until
2026-09-17 those were the intermediate splat re-rendered (`rerender_splat`);
with `pass2_mesh` on (the default) they are the meshify mesh with its klein
texture, rendered by `b2ctrain mesh-render` at EXACTLY the cameras
`rerender_splat` produced — so the path, the anchor's position and the frame
count are what they were, and `mask_splat` and `reinject_anchor` downstream
see a dataset of the same shape. The A/B this ports is b2ctrain
out/mesh/ab2/ds_mesh_b2c (`tools/ab_arm_b2c.py`, the pod pass-2 A/B's fourth
arm): every frame composited over 0.5 grey through the render's alpha.

The frames replace `dataset.images`; the masks are the render's alpha
(float32 [0,1], rmbg's convention), which is the exact silhouette a matte
could only approximate. The texture is the klein-refined one when
`refine_texture` ran, the photograph-projected one when only `photo_texture`
did, the bake otherwise.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..proc import ProcessFailed, stream_command
from ..registry import register_step
from ..step import Param, Step
from .body_refit import cameras_json

logger = logging.getLogger(__name__)


def composite_over_grey(rgba: np.ndarray, grey: float) -> np.ndarray:
    """A BGRA render composited over a flat grey through its own alpha -> BGR uint8."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3].astype(np.float32) * alpha + 255.0 * grey * (1.0 - alpha) + 0.5, 0, 255).astype(np.uint8)


def pick_texture(atlas: Path, refined: Optional[str], photo: Optional[str]) -> Path:
    """The best texture on hand: klein's, else the photograph's, else the bake."""
    for candidate in (refined, photo):
        if candidate and Path(str(candidate)).is_file():
            return Path(str(candidate))
    return atlas / "texture.png"


@register_step("render_mesh_views")
class RenderMeshViewsStep(Step):
    """The textured mesh at the given cameras, composited over grey, with its alpha as the mask.

    inputs:  {"mesh_dir": str — meshify's atlas directory (mesh_uv.obj, texture.png),
              "cameras": List[Camera] — the frames' cameras (rerender_splat's helix),
              "image_names": List[str] — the frames' names, kept for the renders,
              "texture_path"?: str — refine_texture's texture_final.png,
              "photo_texture_path"?: str — photo_texture's texture_photo.png}
    outputs: {"images": List[HxWx3 uint8 BGR], "masks": List[HxW float32 in [0,1]], "mesh_views_stats": dict}
    """

    PARAMS = (
        Param("trainer_path", str, "b2ctrain", "The b2ctrain binary (mesh-render)", advanced=True),
        Param("output_dir", str, help="Where the renders go (kept when keep_renders, else removed after the read)"),
        Param("device", int, 0, "CUDA device index", advanced=True),
        Param("bg_grey", float, 0.5, "The grey the frames are composited over (the splat path's cull_color)", minimum=0.0, maximum=1.0, advanced=True),
        Param("supersample", int, 2, "mesh-render supersampling", minimum=1, maximum=4, advanced=True),
        Param("keep_renders", bool, False, "Keep the RGBA renders on disk under output_dir"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        trainer = params["trainer_path"]
        if shutil.which(trainer) is None and not Path(trainer).is_file():
            raise RuntimeError(f"render_mesh_views: trainer binary {trainer!r} not found on PATH")
        atlas = Path(str(inputs["mesh_dir"]))
        if not (atlas / "mesh_uv.obj").is_file():
            raise FileNotFoundError(f"render_mesh_views: {atlas} is not a meshify atlas (no mesh_uv.obj)")
        cameras = list(inputs["cameras"])
        names = list(inputs.get("image_names") or [])
        if names and len(names) != len(cameras):
            raise ValueError(f"render_mesh_views: {len(names)} image names for {len(cameras)} cameras")
        texture = pick_texture(atlas, inputs.get("texture_path"), inputs.get("photo_texture_path"))
        out = Path(params["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cams = cameras_json(cameras)
        if names:
            for entry, name in zip(cams["cameras"], names):
                entry["name"] = name if name.lower().endswith(".png") else name + ".png"
        cams_json = out / "cams_helix.json"
        cams_json.write_text(json.dumps(cams))
        render_dir = out / "renders"
        if render_dir.exists():
            shutil.rmtree(render_dir)
        render_dir.mkdir(parents=True)
        try:
            stream_command([trainer, "mesh-render", "--atlas", str(atlas), "--texture", str(texture), "--cameras", str(cams_json), "--output", str(render_dir),
                            "--bg", str(params["bg_grey"]), "--ss", str(int(params["supersample"])), "--device", str(int(params["device"]))],
                           log_name="render_mesh_views.render", throttle=True)
        except ProcessFailed as exc:
            raise RuntimeError(f"render_mesh_views: `b2ctrain mesh-render` failed: {exc}") from exc
        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        coverage = []
        for entry in cams["cameras"]:
            path = render_dir / entry["name"]
            rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                raise RuntimeError(f"render_mesh_views: {path} is not the RGBA render mesh-render writes")
            if rgba.shape[:2] != (cams["height"], cams["width"]):
                raise RuntimeError(f"render_mesh_views: {path} is {rgba.shape[1]}x{rgba.shape[0]}, the cameras {cams['width']}x{cams['height']}")
            images.append(composite_over_grey(rgba, float(params["bg_grey"])))
            alpha = rgba[..., 3].astype(np.float32) / 255.0
            masks.append(alpha)
            coverage.append(float((alpha > 0.5).mean()))
        if not params["keep_renders"]:
            shutil.rmtree(render_dir, ignore_errors=True)
        stats = {"frames": len(images), "texture": str(texture), "width": cams["width"], "height": cams["height"],
                 "subject_coverage_mean": round(float(np.mean(coverage)), 4) if coverage else 0.0, "seconds": round(time.time() - t0, 1)}
        (out / "render_mesh_views.json").write_text(json.dumps(stats, indent=1))
        logger.info("render_mesh_views: %d frames of %s at %dx%d in %.0fs (subject %.0f%% of the frame)", len(images), texture.name,
                    cams["width"], cams["height"], stats["seconds"], 100 * stats["subject_coverage_mean"])
        return {"images": images, "masks": masks, "mesh_views_stats": stats}
