"""In-memory dataset representation.

Mirrors the on-disk layout already produced by nodes/save_dataset_node.py and
consumed by nodes/load_dataset_node.py (metadata.json + pointcloud.npz +
frame_NNNNN_.png + optional reference/anchor.png + prompt.txt), so datasets
stay interchangeable with the existing ComfyUI graphs during the transition.

Steps pass this object around by reference in the WorkflowContext. Nothing
touches disk unless a step explicitly asks to (see steps/dataset_io.py) —
that's the whole point of keeping it a plain in-memory dataclass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from body2colmap.camera import Camera

_SPECIAL_KEYS = {"cameras", "image_names", "points_3d", "resolution", "splat_path"}


@dataclass
class Dataset:
    """A batch of rendered views plus the metadata needed to export/reload them.

    images are HxWx3 (BGR) or HxWx4 (BGRA) uint8 arrays — cv2 convention,
    since that's what the render/export code already speaks. Convert at the
    edges (e.g. when handing frames to a torch-based model step) rather than
    changing this representation.
    """

    images: List[np.ndarray]
    image_names: List[str]
    cameras: List[Camera]
    points_3d: Tuple[np.ndarray, np.ndarray]
    resolution: Tuple[int, int]
    masks: Optional[List[np.ndarray]] = None
    reference_image: Optional[np.ndarray] = None
    anchor_image: Optional[np.ndarray] = None
    prompt: Optional[str] = None
    splat_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_disk(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        for img, name in zip(self.images, self.image_names):
            cv2.imwrite(str(out / name), img)

        if self.reference_image is not None:
            cv2.imwrite(str(out / "reference.png"), self.reference_image)
        if self.anchor_image is not None:
            cv2.imwrite(str(out / "anchor.png"), self.anchor_image)
        if self.prompt:
            (out / "prompt.txt").write_text(self.prompt, encoding="utf-8")

        metadata: Dict[str, Any] = {
            "version": "1.0",
            "resolution": list(self.resolution),
            "cameras": [
                {"image_name": name, **_serialize_camera(cam)}
                for name, cam in zip(self.image_names, self.cameras)
            ],
        }

        json_safe_extras = {}
        for key, value in self.extras.items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            try:
                json.dumps(value)
                json_safe_extras[key] = value
            except (TypeError, ValueError):
                continue
        if json_safe_extras:
            metadata["b2c_extras"] = json_safe_extras

        if self.splat_path and Path(self.splat_path).exists():
            import shutil

            shutil.copy(self.splat_path, out / "splat.ply")
            metadata["splat_filename"] = "splat.ply"

        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

        positions, colors = self.points_3d
        np.savez_compressed(out / "pointcloud.npz", positions=positions, colors=colors)

        return out

    @classmethod
    def from_disk(cls, directory: str | Path) -> "Dataset":
        src = Path(directory)
        metadata = json.loads((src / "metadata.json").read_text())
        resolution = tuple(metadata["resolution"])

        cameras, image_names, images = [], [], []
        for cam_data in metadata["cameras"]:
            name = cam_data["image_name"]
            image_names.append(name)
            cameras.append(_deserialize_camera(cam_data, resolution))
            img = cv2.imread(str(src / name), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Missing frame: {src / name}")
            images.append(img)

        masks = None
        if images and images[0].ndim == 3 and images[0].shape[2] == 4:
            masks = [img[:, :, 3] for img in images]
            images = [img[:, :, :3] for img in images]

        pointcloud = np.load(src / "pointcloud.npz")
        points_3d = (pointcloud["positions"], pointcloud["colors"])

        reference_path = src / "reference.png"
        reference_image = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED) if reference_path.exists() else None

        anchor_path = src / "anchor.png"
        anchor_image = cv2.imread(str(anchor_path), cv2.IMREAD_UNCHANGED) if anchor_path.exists() else None

        prompt_path = src / "prompt.txt"
        prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None

        splat_path = None
        if metadata.get("splat_filename") and (src / metadata["splat_filename"]).exists():
            splat_path = str(src / metadata["splat_filename"])

        return cls(
            images=images,
            image_names=image_names,
            cameras=cameras,
            points_3d=points_3d,
            resolution=resolution,
            masks=masks,
            reference_image=reference_image,
            anchor_image=anchor_image,
            prompt=prompt,
            splat_path=splat_path,
            extras=metadata.get("b2c_extras", {}),
        )


def _serialize_camera(camera: Camera) -> dict:
    return {
        "intrinsics": {
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
        },
        "extrinsics": {
            "rotation": camera.rotation.tolist(),
            "position": camera.position.tolist(),
        },
    }


def _deserialize_camera(camera_data: dict, resolution: Tuple[int, int]) -> Camera:
    intrinsics = camera_data["intrinsics"]
    extrinsics = camera_data["extrinsics"]
    return Camera(
        focal_length=(intrinsics["fx"], intrinsics["fy"]),
        image_size=resolution,
        principal_point=(intrinsics["cx"], intrinsics["cy"]),
        position=np.array(extrinsics["position"], dtype=np.float32),
        rotation=np.array(extrinsics["rotation"], dtype=np.float32),
    )
