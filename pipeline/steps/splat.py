"""Gaussian-splat steps: load_splat / save_splat / render_splat.

Native ports of `nodes/splat_loader_node.py`, `nodes/save_splat_node.py`
and `nodes/splat_render_node.py`.

`render_splat` is the missing half of this project's core loop. `brush`
trains a splat from a set of views; this step renders that trained splat
back out along a *new* camera path, so the next denoise pass has fresh
views to clean up:

    denoise -> brush (train) -> render_splat (re-render) -> denoise -> ...

Every ComfyUI pipeline YAML is built on that loop —
`workflows/api/resplat_helical.json`, `resplat_tiered.json` and
`outline.json` all call `Body2COLMAP_RenderSplat` — so without this step
`brush`'s output is a dead end.

**Rasterisation is `brush-splat-render`, not gsplat.** gsplat publishes no
wheel past torch 2.4/cu124, so on a modern stack it JIT-compiles its CUDA
kernels on first use and needs nvcc at *runtime* — the reason the image had
to ship a CUDA devel base at all. `brush-splat-render` is a standalone Rust
CLI (`crates/brush-splat-render` in the Erant/brush fork, built into
docker/Dockerfile right alongside the `brush` training binary) that loads a
trained `.ply` and rasterises an explicit camera list via the same
wgpu/Vulkan renderer `brush` itself uses. This step shells out to it the
same way `steps/brush.py` shells out to `brush`. See
`~/Projects/brush/docs/splat-render.md` for the CLI, the `cameras.json`
schema, and — the part that is easy to get wrong silently — the coordinate
conversion between body2colmap's OpenGL-convention `Camera` and brush's
OpenCV-convention one, verified there against a real gsplat oracle (mean
abs error 0.0008-0.0015 on RGB, comfortably under 1/255).

Everything *except* rasterisation still belongs to body2colmap
(`SplatScene`, `OrbitPath`, the `path`/`utils` helpers); `SplatRenderer`
(the gsplat wrapper) is no longer used here at all.

**Verification status.** The camera-path half — which is where all the
subtle behaviour lives — is verified locally against `cyber_6f`'s real
recorded metadata (see tests/test_splat.py), including that the anchored
override path puts a camera exactly on the recorded `anchor_position`.
PLY load/save is verified by round-trip. `brush-splat-render` itself is
verified against gsplat as described above, but that verification ran
outside this pipeline (a manual comparison against local data); running
*through* this step, end to end, on this box's GPU is still pending — see
docs/docker-build-notes.md.

**Focal-length inheritance** is the one piece here that is easy to get
wrong and silent when wrong. The resolution order is: the step's own
`focal_length_mm` param, else the source dataset's recorded
`focal_length_mm`, else auto. In override mode a framed focal length is
*required* — the anchor frame has to match the framing the mesh render
used, or the reference image warped during that render no longer lines up
with the frame it gets injected into. Passing an explicit
`focal_length_mm` that disagrees with the dataset's is therefore warned
about rather than silently honoured.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..proc import stream_command
from ..registry import register_step
from ..step import Step

logger = logging.getLogger(__name__)

_FULL_FRAME_SENSOR_WIDTH_MM = 36.0

# Rebuilt by every render; never inherited from the source dataset's extras.
_REBUILT_KEYS = {
    "cameras", "image_names", "points_3d", "resolution",
    "focal_length_mm", "splat_path",
}
# A new path invalidates an inherited anchor: the orbit is different, so the
# recorded position need not be on it. Reused cameras keep theirs, which is
# exactly why the position (not the frame index) is the durable key.
_ANCHOR_KEYS = {"anchor_frame_index", "anchor_position"}


def _mm_to_pixels(focal_length_mm: float, image_width: int) -> float:
    return (focal_length_mm / _FULL_FRAME_SENSOR_WIDTH_MM) * image_width


@register_step("load_splat")
class LoadSplatStep(Step):
    """Load a trained Gaussian splat from a PLY file.

    inputs:  {} (or {"splat_path": str} to override the param)
    params:  filepath (str)
    outputs: {"splat_scene": SplatScene, "splat_path": str}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.splat_scene import SplatScene

        filepath = inputs.get("splat_path") or params.get("filepath")
        if not filepath:
            raise ValueError("load_splat needs a 'filepath' param or 'splat_path' input")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Splat PLY not found: {path}")

        scene = SplatScene.from_ply(str(path))
        logger.info(
            "load_splat: %s (%d Gaussians, SH degree %d)",
            path, len(scene), scene.sh_degree,
        )
        return {"splat_scene": scene, "splat_path": str(path)}


@register_step("save_splat")
class SaveSplatStep(Step):
    """Write a Gaussian splat scene to a PLY file.

    inputs:  {"splat_scene": SplatScene}
    params:  filepath (str)
    outputs: {"splat_path": str}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        scene = inputs["splat_scene"]
        filepath = params.get("filepath")
        if not filepath:
            raise ValueError("save_splat needs a 'filepath' param")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene.to_ply(str(path))
        logger.info("save_splat: wrote %d Gaussians to %s", len(scene), path)
        return {"splat_path": str(path)}


@register_step("render_splat")
class RenderSplatStep(Step):
    """Render a trained Gaussian splat along a camera path.

    inputs:  {"splat_scene": SplatScene} or {"splat_path": str},
             plus optionally {"dataset": Dataset} — the source dataset,
             used for framing bounds, focal-length inheritance, camera
             reuse, point-cloud preservation and extras pass-through
    params:  pattern ("circular" | "sinusoidal" | "helical", omit to reuse
             the dataset's cameras verbatim), n_frames and the
             pattern-specific params (as steps/render.py), width, height,
             framing, focal_length_mm, fill_ratio, bg_color,
             override_cam_from_mesh, override_pointcloud,
             pointcloud_samples, render_path
    outputs: same shape as steps/render.py — {"images", "masks",
             "cameras", "image_names", "points_3d", "resolution",
             "focal_length_mm"} plus "anchor_position" /
             "anchor_frame_index" when the path was anchored, and any
             non-rebuilt extras carried through from the source dataset

    With no `pattern`, the source dataset's cameras are reused verbatim —
    that is how `outline.json` re-renders the exact same views from a
    trained splat so `replace_views` can swap them back in.

    Rasterisation shells out to the `brush-splat-render` binary (a Rust CLI
    on `PATH`, built into docker/Dockerfile — see the module docstring).
    `render_path` overrides the binary name/path, same convention as
    `steps/brush.py`'s `brush_path`.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.splat_scene import SplatScene

        scene = inputs.get("splat_scene")
        splat_path = inputs.get("splat_path") or params.get("splat_path")
        if scene is None:
            if not splat_path:
                raise ValueError(
                    "render_splat needs a 'splat_scene' input, or a 'splat_path' "
                    "to load one from"
                )
            scene = SplatScene.from_ply(str(splat_path))

        dataset = inputs.get("dataset")
        width = int(params.get("width", 720))
        height = int(params.get("height", 1280))

        cameras, focal_length, effective_mm, anchor_frame_index = _resolve_cameras(
            scene=scene, dataset=dataset, params=params, width=width, height=height
        )

        bg_color = tuple(params.get("bg_color", (1.0, 1.0, 1.0)))
        render_path = params.get("render_path", "brush-splat-render")
        image_names = [f"frame_{i + 1:05d}_.png" for i in range(len(cameras))]

        logger.info(
            "render_splat: %d Gaussians (SH degree %d), %d frames at %dx%d via %s",
            len(scene), scene.sh_degree, len(cameras), width, height, render_path,
        )

        images, masks = _rasterize(
            scene=scene,
            splat_path=splat_path,
            cameras=cameras,
            image_names=image_names,
            width=width,
            height=height,
            bg_color=bg_color,
            render_path=render_path,
        )

        points_3d = _resolve_pointcloud(scene, dataset, params)

        result: Dict[str, Any] = {
            "images": images,
            "masks": masks,
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": points_3d,
            "resolution": (width, height),
            "focal_length_mm": effective_mm,
        }

        # Carry through everything the source dataset knew that this render
        # does not rebuild, so downstream steps keep working without an
        # allowlist for every key.
        if dataset is not None:
            skip = _REBUILT_KEYS | _ANCHOR_KEYS if params.get("pattern") else _REBUILT_KEYS
            for key, value in dataset.extras.items():
                if key not in skip:
                    result[key] = value

        if anchor_frame_index is not None:
            result["anchor_frame_index"] = int(anchor_frame_index)
            result["anchor_position"] = np.asarray(
                cameras[anchor_frame_index].position, dtype=np.float32
            )

        return result


def _rasterize(
    *, scene, splat_path, cameras, image_names, width, height, bg_color, render_path,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Render `cameras` against `scene` via the `brush-splat-render` CLI.

    Returns (images, masks): BGR uint8 images and float32 [0,1] foreground
    masks, one per camera, in `cameras` order.
    """
    with tempfile.TemporaryDirectory(prefix="b2c_render_splat_") as tmp:
        tmp_dir = Path(tmp)

        # Reuse an existing PLY on disk rather than re-serializing through
        # SplatScene when we already have one (the common case: this step
        # usually follows `brush` training, which writes a PLY directly).
        ply_path = Path(splat_path) if splat_path and Path(splat_path).exists() else None
        if ply_path is None:
            ply_path = tmp_dir / "scene.ply"
            scene.to_ply(str(ply_path))

        cameras_path = tmp_dir / "cameras.json"
        _write_cameras_json(cameras, image_names, width, height, cameras_path)

        output_dir = tmp_dir / "renders"
        cmd = [
            render_path,
            "--splat", str(ply_path),
            "--cameras", str(cameras_path),
            "--output-dir", str(output_dir),
            "--background", ",".join(f"{c:.6f}" for c in bg_color),
        ]
        _run_render(cmd)

        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for name in image_names:
            png_path = output_dir / name
            rgba = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                raise RuntimeError(
                    f"render_splat: brush-splat-render did not produce {png_path}\n"
                    f"Command: {' '.join(cmd)}"
                )
            images.append(rgba[..., :3])
            masks.append(rgba[..., 3].astype(np.float32) / 255.0)

    return images, masks


def _write_cameras_json(cameras, image_names, width: int, height: int, path: Path) -> None:
    """Write brush-splat-render's cameras.json — see
    ~/Projects/brush/docs/splat-render.md for the schema. `camera.rotation`
    is already in the exact row-major, OpenGL-convention c2w form the
    binary expects; the OpenGL->OpenCV axis conversion happens Rust-side.
    """
    payload = {
        "width": width,
        "height": height,
        "cameras": [
            {
                "name": name,
                "fx": float(camera.fx),
                "fy": float(camera.fy),
                "cx": float(camera.cx),
                "cy": float(camera.cy),
                "position": [float(v) for v in camera.position],
                "rotation": [[float(v) for v in row] for row in camera.rotation],
            }
            for camera, name in zip(cameras, image_names)
        ],
    }
    path.write_text(json.dumps(payload))


def _run_render(cmd: List[str]) -> None:
    # Same live relay as the brush training step — see pipeline/proc.py.
    stream_command(
        cmd,
        log_name="brush-splat-render",
        not_found_hint=(
            "It is built into the image at /usr/local/bin/brush-splat-render, "
            "alongside `brush`, from the same Erant/brush clone."
        ),
    )


def _resolve_cameras(
    *, scene, dataset, params: Dict[str, Any], width: int, height: int
) -> Tuple[list, float, float, Optional[int]]:
    """Work out which cameras to render from, and at what focal length.

    Returns (cameras, focal_length_px, effective_focal_length_mm,
    anchor_frame_index). Split out of run() so the whole decision — which
    is where every subtlety in this step lives — is testable without gsplat
    or a GPU.
    """
    from body2colmap.camera import Camera
    from body2colmap.path import (
        OrbitPath,
        compute_helical_anchor_params,
        compute_original_camera_orbit_params,
    )
    from body2colmap.utils import compute_auto_orbit_radius, compute_default_focal_length

    pattern = params.get("pattern")
    override_cam_from_mesh = bool(params.get("override_cam_from_mesh", False))
    extras = dataset.extras if dataset is not None else {}

    # Focal length: explicit param > inherited from the dataset > auto.
    focal_length_mm = float(params.get("focal_length_mm", 0.0) or 0.0)
    effective_mm = focal_length_mm
    if effective_mm <= 0:
        effective_mm = float(extras.get("focal_length_mm", 0.0) or 0.0)

    inherited_mm = float(extras.get("focal_length_mm", 0.0) or 0.0)
    if (
        override_cam_from_mesh
        and focal_length_mm > 0
        and inherited_mm > 0
        and not np.isclose(focal_length_mm, inherited_mm)
    ):
        logger.warning(
            "render_splat: focal_length_mm=%.2f overrides the %.2fmm the source "
            "dataset was framed at. The anchor frame will no longer match the "
            "reference image warped during that render. Set focal_length_mm to 0 "
            "to inherit it.",
            focal_length_mm, inherited_mm,
        )

    focal_length = (
        _mm_to_pixels(effective_mm, width) if effective_mm > 0
        else compute_default_focal_length(width)
    )

    # No pattern: reuse the dataset's cameras verbatim. Its anchor keys then
    # still describe this render, so they pass through untouched.
    if pattern is None:
        if dataset is None or not dataset.cameras:
            raise ValueError(
                "render_splat needs either a 'pattern' param to build a new "
                "camera path, or a 'dataset' input whose cameras it can reuse."
            )
        if override_cam_from_mesh:
            raise ValueError(
                "override_cam_from_mesh requires a 'pattern' — there is no path to "
                "anchor when cameras are reused verbatim. Either set a pattern, or "
                "drop the option (reused cameras already carry the original "
                "render's anchor)."
            )
        logger.info("render_splat: reusing %d cameras from the dataset", len(dataset.cameras))
        return list(dataset.cameras), focal_length, effective_mm, None

    camera_template = Camera(
        focal_length=(focal_length, focal_length), image_size=(width, height)
    )

    if override_cam_from_mesh:
        cameras, anchor_frame_index = _anchored_path(
            pattern=pattern,
            params=params,
            extras=extras,
            camera_template=camera_template,
            effective_mm=effective_mm,
            orbit_path_cls=OrbitPath,
            circular_solver=compute_original_camera_orbit_params,
            helical_solver=compute_helical_anchor_params,
        )
        return cameras, focal_length, effective_mm, anchor_frame_index

    # Framing bounds from the source render, falling back to the splat's own
    # bounds. Using the mesh render's bounds is what keeps a re-render framed
    # identically to the render it is replacing.
    framing = params.get("framing", "full")
    bounds = None
    framing_bounds = extras.get("framing_bounds")
    if framing_bounds and framing in framing_bounds:
        bounds = framing_bounds[framing]
        logger.info("render_splat: using '%s' framing bounds from the dataset", framing)
    elif framing != "full":
        logger.warning(
            "render_splat: framing preset '%s' not available in the dataset's "
            "metadata; falling back to the splat scene's own bounds.", framing,
        )
    if bounds is None:
        bounds = scene.get_bounds()

    orbit_center = (bounds[0] + bounds[1]) / 2.0
    radius = params.get("radius")
    if radius is None:
        radius = compute_auto_orbit_radius(
            bounds=bounds,
            render_size=(width, height),
            focal_length=focal_length,
            fill_ratio=float(params.get("fill_ratio", 0.8)),
        )

    path_gen = OrbitPath(target=orbit_center, radius=radius)
    cameras = _generate_path(path_gen, pattern, params, camera_template)
    logger.info(
        "render_splat: %s path, %d cameras, radius=%.3f", pattern, len(cameras), radius
    )
    return cameras, focal_length, effective_mm, None


def _generate_path(path_gen, pattern: str, params: Dict[str, Any], camera_template):
    if pattern == "circular":
        return path_gen.circular(
            n_frames=params["n_frames"],
            elevation_deg=params.get("elevation_deg", 0.0),
            start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
            overlap=params.get("overlap", 1),
            camera_template=camera_template,
        )
    if pattern == "sinusoidal":
        return path_gen.sinusoidal(
            n_frames=params["n_frames"],
            amplitude_deg=params["amplitude_deg"],
            n_cycles=params["n_cycles"],
            start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
            camera_template=camera_template,
        )
    if pattern == "helical":
        return path_gen.helical(
            n_frames=params["n_frames"],
            n_loops=params["n_loops"],
            amplitude_deg=params["amplitude_deg"],
            lead_in_deg=params.get("lead_in_deg", 45.0),
            lead_out_deg=params.get("lead_out_deg", 45.0),
            start_azimuth_deg=params.get("start_azimuth_deg", 0.0),
            camera_template=camera_template,
        )
    raise ValueError(f"Unknown path pattern: {pattern!r}")


def _anchored_path(
    *, pattern, params, extras, camera_template, effective_mm,
    orbit_path_cls, circular_solver, helical_solver,
):
    """Build a path anchored to the original SAM-3D-Body camera.

    Mirrors steps/render.py's override mode, but takes the orbit target and
    framed focal length from the source dataset's metadata rather than
    recomputing them from geometry — that is what keeps this render's anchor
    frame identical to the mesh render's, so the reference image warped
    during that render stays valid here without being re-warped.

    Anchoring assumes the original camera sits at the world origin, which
    only holds for a dataset rendered in override mode (a normal render
    auto-orients the scene and breaks it). `original_focal_length` is
    written only in override mode, so its presence is the marker.
    """
    if pattern not in ("circular", "helical"):
        raise ValueError(
            f"override_cam_from_mesh only works with a circular or helical "
            f"pattern, got {pattern!r}"
        )
    if effective_mm <= 0:
        raise ValueError(
            "override_cam_from_mesh needs a framed focal length, but neither the "
            "focal_length_mm param nor the dataset's metadata provides one. "
            "Re-render the source dataset with a render step that records the "
            "framed focal length it actually used."
        )
    if "orbit_target" not in extras:
        raise ValueError(
            "override_cam_from_mesh needs extras['orbit_target'] from the source "
            "render's metadata."
        )
    if "original_focal_length" not in extras:
        raise ValueError(
            "override_cam_from_mesh needs a dataset rendered with "
            "override_cam_from_mesh itself — the marker for that is "
            "extras['original_focal_length'], which this dataset lacks. A "
            "normally-rendered dataset has been auto-oriented, so the original "
            "camera is no longer at the world origin and the anchor would be wrong."
        )

    # Save -> Load turns the ndarray into a plain list; normalise.
    orbit_center = np.asarray(extras["orbit_target"], dtype=np.float32)

    if pattern == "circular":
        orbit_params = circular_solver(orbit_center)
        radius = float(orbit_params["radius"])
        path_gen = orbit_path_cls(target=orbit_center, radius=radius)
        cameras = path_gen.circular(
            n_frames=params["n_frames"],
            elevation_deg=orbit_params["elevation_deg"],
            start_azimuth_deg=orbit_params["start_azimuth_deg"],
            overlap=params.get("overlap", 1),
            camera_template=camera_template,
        )
        return cameras, 0

    helix_params = dict(
        n_frames=params["n_frames"],
        n_loops=params["n_loops"],
        amplitude_deg=params["amplitude_deg"],
        lead_in_deg=params.get("lead_in_deg", 45.0),
        lead_out_deg=params.get("lead_out_deg", 45.0),
    )
    anchor_info = helical_solver(target=orbit_center, **helix_params)
    path_gen = orbit_path_cls(target=orbit_center, radius=float(anchor_info["radius"]))
    cameras = path_gen.helical(
        start_azimuth_deg=anchor_info["start_azimuth_deg"],
        elevation_offset_deg=anchor_info["elevation_offset_deg"],
        camera_template=camera_template,
        **helix_params,
    )
    return cameras, int(anchor_info["anchor_frame_index"])


def _resolve_pointcloud(scene, dataset, params: Dict[str, Any]):
    """Sample a fresh point cloud, or keep the dataset's existing one.

    Preserving it is the default: the mesh render's point cloud describes
    the actual subject geometry, while one sampled from a trained splat
    inherits whatever noise that training left behind.
    """
    override = bool(params.get("override_pointcloud", False))
    if not override and dataset is not None and dataset.points_3d is not None:
        logger.info("render_splat: preserving the dataset's point cloud")
        return dataset.points_3d

    n_samples = int(params.get("pointcloud_samples", 10000))
    logger.info("render_splat: sampling %d points from the splat scene", n_samples)
    return scene.get_point_cloud(n_samples=n_samples)
