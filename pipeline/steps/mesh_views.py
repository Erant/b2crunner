"""render_subject — the frames pass 2 conditions on: the textured mesh, or the splat.

`render_splat` (steps/splat.py) decides the helical path: the 81 cameras at
the resolution and framing, `override_cam_from_mesh`'s placement, the anchor
camera carried rigidly by the refinement's delta, and the `anchor_position` /
`anchor_frame_index` extras `reinject_anchor` reads. `render_subject` is that
step with one more decision — `from_mesh` — made where the frames are drawn:
on, the meshify mesh with its klein texture is rendered by `b2ctrain
mesh-render` at those cameras and composited over the same flat colour the
splat would have been, the render's alpha as the matte (the exact silhouette
a matte could only approximate); off, the splat is rasterised as before.
Nothing else in the step or downstream changes: `mask_splat_fringes`,
`reinject_anchor` and `denoise_pass2` see a dataset of the same shape.

The A/B this ports is b2ctrain out/mesh/ab2/ds_mesh_b2c (`tools/ab_arm_b2c.py`,
the pod pass-2 A/B's fourth arm). The texture is klein's when
`refine_texture` ran, the photograph's when only `photo_texture` did, the
bake otherwise.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..proc import ProcessFailed, stream_command
from ..registry import register_step
from ..step import Param
from .body_refit import cameras_json
from .splat import RenderSplatStep

logger = logging.getLogger(__name__)


def composite_over(rgba: np.ndarray, flat_bgr: Sequence[float]) -> np.ndarray:
    """A BGRA render composited over a flat colour (BGR in [0,1]) through its own alpha -> BGR uint8."""
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    flat = np.asarray(flat_bgr, np.float32).reshape(1, 1, 3) * 255.0
    return np.clip(rgba[..., :3].astype(np.float32) * alpha + flat * (1.0 - alpha) + 0.5, 0, 255).astype(np.uint8)


def pick_texture(atlas: Path, refined: Optional[str], photo: Optional[str]) -> Path:
    """The best texture on hand: klein's, else the photograph's, else the bake."""
    for candidate in (refined, photo):
        if candidate and Path(str(candidate)).is_file():
            return Path(str(candidate))
    return atlas / "texture.png"


def render_mesh_frames(trainer: str, atlas: Path, texture: Path, cameras: Sequence[Any], image_names: Sequence[str], flat_bgr: Sequence[float],
                       supersample: int, device: int, out: Path, keep: bool) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """`b2ctrain mesh-render` at the cameras: (BGR frames over `flat_bgr`, float32 alpha masks, stats)."""
    import cv2

    if shutil.which(trainer) is None and not Path(trainer).is_file():
        raise RuntimeError(f"render_subject: trainer binary {trainer!r} not found on PATH")
    if not (atlas / "mesh_uv.obj").is_file():
        raise FileNotFoundError(f"render_subject: {atlas} is not a meshify atlas (no mesh_uv.obj)")
    if len(image_names) != len(cameras):
        raise ValueError(f"render_subject: {len(image_names)} image names for {len(cameras)} cameras")
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    cams = cameras_json(cameras)
    for entry, name in zip(cams["cameras"], image_names):
        entry["name"] = name if name.lower().endswith(".png") else name + ".png"
    cams_json = out / "cams.json"
    cams_json.write_text(json.dumps(cams))
    render_dir = out / "renders"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)
    grey = float(np.mean(flat_bgr))  # mesh-render draws one grey; the composite below uses the exact colour
    try:
        stream_command([trainer, "mesh-render", "--atlas", str(atlas), "--texture", str(texture), "--cameras", str(cams_json), "--output", str(render_dir),
                        "--bg", f"{grey:.4f}", "--ss", str(int(supersample)), "--device", str(int(device))],
                       log_name="render_subject.mesh", throttle=True)
    except ProcessFailed as exc:
        raise RuntimeError(f"render_subject: `b2ctrain mesh-render` failed: {exc}") from exc
    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    coverage = []
    for entry in cams["cameras"]:
        path = render_dir / entry["name"]
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError(f"render_subject: {path} is not the RGBA render mesh-render writes")
        if rgba.shape[:2] != (cams["height"], cams["width"]):
            raise RuntimeError(f"render_subject: {path} is {rgba.shape[1]}x{rgba.shape[0]}, the cameras {cams['width']}x{cams['height']}")
        images.append(composite_over(rgba, flat_bgr))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        masks.append(alpha)
        coverage.append(float((alpha > 0.5).mean()))
    if not keep:
        shutil.rmtree(render_dir, ignore_errors=True)
    stats = {"frames": len(images), "texture": str(texture), "width": cams["width"], "height": cams["height"],
             "subject_coverage_mean": round(float(np.mean(coverage)), 4) if coverage else 0.0, "seconds": round(time.time() - t0, 1)}
    (out / "render_subject.json").write_text(json.dumps(stats, indent=1))
    logger.info("render_subject: %d frames of the mesh (%s) at %dx%d in %.0fs (subject %.0f%% of the frame)", len(images), texture.name,
                cams["width"], cams["height"], stats["seconds"], 100 * stats["subject_coverage_mean"])
    return images, masks, stats


@register_step("render_subject")
class RenderSubjectStep(RenderSplatStep):
    """`render_splat` with the source of the frames decided by `from_mesh` (see the module docstring).

    extra inputs (mesh mode): {"mesh_dir": str — meshify's atlas directory,
                               "mesh_texture_path"?: str — refine_texture's texture,
                               "mesh_photo_texture_path"?: str — photo_texture's texture}
    outputs: render_splat's, the frames and masks from the mesh when `from_mesh`.
    """

    PARAMS = RenderSplatStep.PARAMS + (
        Param("from_mesh", bool, False, "Render the textured mesh (mesh_dir + the best texture on hand) at the resolved cameras instead of the splat"),
        Param("trainer_path", str, "b2ctrain", "The b2ctrain binary (mesh-render)", advanced=True),
        Param("mesh_supersample", int, 2, "mesh-render supersampling", minimum=1, maximum=4, advanced=True),
        Param("mesh_render_dir", str, "", "Where the mesh renders and their camera file go (a temporary directory when empty)", advanced=True),
        Param("keep_mesh_renders", bool, False, "Keep the RGBA mesh renders under mesh_render_dir"),
        Param("device", int, 0, "CUDA device index for mesh-render", advanced=True),
    )

    def _render_frames(self, inputs: Dict[str, Any], params: Dict[str, Any], *, scene, splat_path, cameras, image_names,
                       width: int, height: int, bg_color, render_path, confidence, sh_degree: int):
        if not params["from_mesh"]:
            return super()._render_frames(inputs, params, scene=scene, splat_path=splat_path, cameras=cameras, image_names=image_names,
                                          width=width, height=height, bg_color=bg_color, render_path=render_path, confidence=confidence, sh_degree=sh_degree)
        mesh_dir = inputs.get("mesh_dir")
        if not mesh_dir:
            raise ValueError("render_subject: from_mesh is on but no mesh_dir was wired (meshify did not run?)")
        atlas = Path(str(mesh_dir))
        texture = pick_texture(atlas, inputs.get("mesh_texture_path"), inputs.get("mesh_photo_texture_path"))
        # The flat colour the splat frames would have had under them: cull_color in confidence mode (where
        # bg_color never reaches the rasteriser), bg_color otherwise. The backdrop step composites over it later.
        flat = tuple(float(c) for c in (bg_color if confidence is None else confidence.cull_color))
        out = Path(params["mesh_render_dir"]) if params["mesh_render_dir"] else Path(tempfile.mkdtemp(prefix="b2c_render_subject_"))
        images, masks, stats = render_mesh_frames(params["trainer_path"], atlas, texture, cameras, image_names, flat, int(params["mesh_supersample"]),
                                                  int(params["device"]), out, bool(params["keep_mesh_renders"]))
        if not params["mesh_render_dir"]:
            shutil.rmtree(out, ignore_errors=True)
        self._mesh_stats = stats
        return images, masks
