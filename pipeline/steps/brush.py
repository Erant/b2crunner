"""Gaussian-splat training via the Erant/brush CLI (a Rust binary on `PATH`,
never a Python binding — built into docker/Dockerfile).

Dispatch: `in_process`, not `docker` — targeting RunPod specifically, where
a pod is a single container with no nested Docker daemon to run a separate
brush image in (confirmed on a real pod: no /var/run/docker.sock, no
`docker` binary at all). This Step's own Python code has no conflicting
dependencies either way — subprocess.Popen'ing a CLI binary doesn't need
venv isolation — so `in_process` is fine for the Python side; what's
*unresolved* is brush's OS-level Vulkan/graphics requirement, since a
default RunPod pod's `NVIDIA_DRIVER_CAPABILITIES` only exposed
`compute,utility` on a real pod tested this session (`vulkaninfo` failed
with `ERROR_INCOMPATIBLE_DRIVER` even with the driver's Vulkan libraries
physically present). docker/Dockerfile bakes
`compute,utility,graphics,display` into the image itself, which is
necessary but not yet confirmed sufficient for how RunPod provisions a pod
from a custom image — see that file's comment and docs/docker.md. If this
never gets resolved for RunPod, `dispatch: docker` (this pipeline still
supports it — `pipeline/dispatch/docker.py`'s own docstring names brush as
its motivating case) is the fallback for any other target that does expose
a Docker daemon.

Port of nodes/brush_node.py's Body2COLMAP_RunBrush, minus everything that
was only there to unwrap ComfyUI's list-batched inputs (this pipeline's
Steps already take plain lists). UNVERIFIED end to end: brush itself was
never actually built or run in this session (see docs/docker.md's "Open
items" — the Vulkan/system-deps list there is researched, not confirmed by
a build), so treat this module the same way as the pod-untested steps
(sam3d_body, seedvr2) even though its logic is a close port of code that
does run in production via ComfyUI.

Normal-map supervision: per the original node's behavior, a normal map
that already carries an alpha channel keeps it; otherwise the RGB frame's
own foreground mask (rmbg's output) is reused as the normal map's alpha,
since that mask is what the loss should be restricted to. Brush
auto-detects a `normals/` directory beside `images/` in the COLMAP export
— its absence (masks/normal_maps not passed) just leaves normal
supervision inactive, not an error.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..registry import register_step
from ..step import Step


@register_step("brush")
class BrushStep(Step):
    """Train a 3D Gaussian Splat using the brush CLI tool.

    inputs: {"cameras": List[Camera], "image_names": List[str],
             "points_3d": Tuple[np.ndarray, np.ndarray],
             "images": List[np.ndarray] BGR(A),
             "masks": Optional[List[np.ndarray]] float32 [0,1], foreground=1,
             "normal_maps": Optional[List[np.ndarray]] HxWx3 float32 [-1,1]}
    params: {"brush_path": str, "total_steps": int, "sh_degree": int,
             "max_resolution": int, "max_splats": int, "refine_every": int,
             "alpha_mode": str, "normal_loss_strength": float,
             "normal_loss_step_start": int, "with_viewer": bool,
             "output_dir": str}
    outputs: {"splat_path": str}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.exporter import ColmapExporter

        cameras = inputs["cameras"]
        image_names = inputs["image_names"]
        points_3d = inputs.get("points_3d")
        images = inputs["images"]
        masks = inputs.get("masks")
        normal_maps = inputs.get("normal_maps")

        if len(images) != len(image_names):
            raise ValueError(f"images ({len(images)}) and image_names ({len(image_names)}) length mismatch")
        if normal_maps is not None and len(normal_maps) != len(images):
            raise ValueError(
                f"Normal map count ({len(normal_maps)}) does not match image count "
                f"({len(images)}). Every training view needs a matching normal map."
            )

        brush_path = params.get("brush_path", "brush")
        total_steps = int(params.get("total_steps", 30000))
        sh_degree = int(params.get("sh_degree", 3))
        max_resolution = int(params.get("max_resolution", 1920))
        max_splats = int(params.get("max_splats", 10_000_000))
        refine_every = int(params.get("refine_every", 200))
        alpha_mode = params.get("alpha_mode", "transparent")
        normal_loss_strength = float(params.get("normal_loss_strength", 0.05))
        normal_loss_step_start = int(params.get("normal_loss_step_start", 5000))
        with_viewer = bool(params.get("with_viewer", False))

        timestamp = int(time.time() * 1000)
        out_root = Path(params.get("output_dir", tempfile.gettempdir())) / "brush" / f"training_{timestamp}"
        out_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="b2c_colmap_") as temp_dir:
            colmap_dir = Path(temp_dir)

            ColmapExporter(cameras=cameras, image_names=image_names, points_3d=points_3d).export(
                output_dir=colmap_dir
            )

            images_dir = colmap_dir / "images"
            images_dir.mkdir(exist_ok=True)

            alpha_channel = None
            if masks is not None:
                alpha_channel = [np.clip(m * 255.0, 0, 255).astype(np.uint8) for m in masks]

            for i, (img, filename) in enumerate(zip(images, image_names)):
                if alpha_channel is not None:
                    alpha = alpha_channel[i]
                    if img.shape[-1] == 4:
                        rgba = img.copy()
                        rgba[..., 3] = alpha
                    elif img.shape[-1] == 3:
                        rgba = np.dstack([img, alpha])
                    else:
                        raise ValueError(f"Unexpected image channels: {img.shape[-1]} (expected 3 or 4)")
                    cv2.imwrite(str(images_dir / filename), rgba)
                else:
                    cv2.imwrite(str(images_dir / filename), img)

            if normal_maps is not None:
                normals_dir = colmap_dir / "normals"
                normals_dir.mkdir(exist_ok=True)
                for i, (normal, filename) in enumerate(zip(normal_maps, image_names)):
                    # normal is HxWx3 float32 in [-1, 1] (sapiens2's output convention) ->
                    # BGR uint8 [0, 255] for disk, matching how images are stored here.
                    normal_bgr = np.clip((normal[..., ::-1] + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                    if alpha_channel is not None:
                        out = np.dstack([normal_bgr, alpha_channel[i]])
                    else:
                        out = normal_bgr
                    normal_path = normals_dir / Path(filename).with_suffix(".png").name
                    cv2.imwrite(str(normal_path), out)

            ply_output_name = "export.ply"
            cmd = [
                brush_path,
                str(colmap_dir),
                "--total-steps", str(total_steps),
                "--sh-degree", str(sh_degree),
                "--export-path", str(out_root.absolute()),
                "--export-name", ply_output_name,
                "--export-every", str(total_steps),
                "--max-resolution", str(max_resolution),
                "--max-splats", str(max_splats),
                "--refine-every", str(refine_every),
            ]
            if with_viewer:
                cmd.append("--with-viewer")
            if masks is not None:
                cmd.extend(["--alpha-mode", alpha_mode])
            if normal_maps is not None:
                cmd.extend([
                    "--normal-loss-weight", str(normal_loss_strength),
                    "--normal-loss-start-iter", str(normal_loss_step_start),
                ])

            self._run_brush(cmd)

        ply_path = out_root / ply_output_name
        if not ply_path.exists():
            raise RuntimeError(
                f"Expected output PLY file not found: {ply_path}\nBrush may not have exported successfully."
            )

        return {"splat_path": str(ply_path.absolute())}

    def _run_brush(self, cmd: List[str]) -> None:
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Brush executable not found at: {cmd[0]}\n"
                f"Please ensure brush is installed and the path is correct."
            )

        output_lines: List[str] = []

        def read_output() -> None:
            for line in process.stdout:
                output_lines.append(line)

        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        return_code = process.wait()
        output_thread.join(timeout=5.0)

        if return_code != 0:
            raise RuntimeError(
                f"Brush training failed with exit code {return_code}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{''.join(output_lines[-50:])}"
            )
