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

from .masks import mask_to_alpha_u8

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
    # The view wan22_vace_denoise conditions on, saved as reference.png.
    # In a from-a-sheet run this is the BACK half of the input sheet, put
    # there by `split_reference_sheet`: the front already reaches the
    # diffusion pass as the injected anchor frame, so the reference slot
    # carries the one view nothing else in the batch can supply. Between
    # `from_reference_image` and that first step it briefly holds the
    # whole two-panel sheet instead.
    reference_image: Optional[np.ndarray] = None
    anchor_image: Optional[np.ndarray] = None
    prompt: Optional[str] = None
    splat_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_disk(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        # Masks ride along in the alpha channel, matching what
        # nodes/save_dataset_node.py writes and what from_disk() below
        # reads back. Without this a save_dataset checkpoint silently
        # drops dataset.masks — which for a pre-denoise dataset is the
        # per-frame reference/denoise flag wan22_vace_denoise consumes,
        # not a throwaway.
        #
        # No inversion here, unlike the ComfyUI node: ComfyUI's MASK
        # convention is inverted (1.0 = background) so that node writes
        # alpha = 1 - mask, whereas this pipeline's convention is
        # foreground = 1 throughout (see steps/rmbg.py), which is already
        # what alpha means.
        for i, (img, name) in enumerate(zip(self.images, self.image_names)):
            frame = img
            if self.masks is not None and i < len(self.masks):
                frame = _attach_alpha(img, self.masks[i])
            cv2.imwrite(str(out / name), frame)

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
    def from_reference_image(
        cls,
        image: "str | Path | np.ndarray",
        prompt: Optional[str] = None,
    ) -> "Dataset":
        """A Dataset carrying nothing but the one image the pipeline starts from.

        That image is the two-panel front/back sheet, not a photo of the
        subject — `fast_helical_native.yaml`'s first step splits it and
        overwrites `reference_image` with the back half (see
        steps/reference_sheet.py). This constructor stays deliberately
        ignorant of that: it validates nothing about the panels, so the
        error for a single portrait photo comes from the step that can
        name the problem rather than from here.

        `fast_helical_native.yaml` builds everything else itself — sam3d_body
        reconstructs a mesh from the sheet's front half, and `render` populates
        images/cameras/points_3d/resolution from that mesh. But the dataclass
        requires all four up front, and `from_disk` is the only constructor
        there was, so "run the pipeline from a single photo" had no entry
        point at all: you had to hand-build a Context and call
        WorkflowRunner directly. That gap is called out in
        fast_helical_native.yaml's own header and is what this closes.

        Which fields it leaves empty is load-bearing beyond this file: the
        web UI decides whether a workflow can start from a photo at all by
        asking whether it reads one of them before writing it (see
        `webui._NEEDS_A_REAL_DATASET`).

        The empty fields are genuinely empty, not placeholder-shaped:
        `points_3d` is a (0, 3) float32 pair rather than None so that a step
        which reaches for it before `render` has run fails on a shape it can
        report, not on `NoneType`.

        `resolution` starts as the photo's own (width, height). `render`
        overwrites it with the render size; nothing reads it in between.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if loaded is None:
                raise FileNotFoundError(f"Could not read reference image: {path}")
        else:
            loaded = image
            if loaded.ndim == 3 and loaded.shape[2] == 4:
                loaded = cv2.cvtColor(loaded, cv2.COLOR_BGRA2BGR)
            elif loaded.ndim == 2:
                loaded = cv2.cvtColor(loaded, cv2.COLOR_GRAY2BGR)

        height, width = loaded.shape[:2]
        return cls(
            images=[],
            image_names=[],
            cameras=[],
            points_3d=(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            ),
            resolution=(width, height),
            reference_image=loaded,
            prompt=prompt,
        )

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


def _attach_alpha(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite a mask into an image's alpha channel as uint8 [0,255]."""
    alpha = mask_to_alpha_u8(mask)

    bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if alpha.shape != bgr.shape[:2]:
        alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.dstack([bgr, alpha])


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


def find_dataset_root(directory: str | Path) -> Path:
    """Locate the directory holding metadata.json, at or below `directory`.

    Uploaded archives are almost never rooted the way you'd like — a zip of
    `initial/` unpacks to `initial/metadata.json`, one made on macOS adds
    `__MACOSX/`, and one made by selecting the files rather than the folder
    unpacks flat. Rather than making the user get it right, look for the one
    file that identifies a b2c dataset and use whatever directory holds it.

    Raises FileNotFoundError naming what was actually found, since "no
    metadata.json anywhere in this archive" is otherwise indistinguishable
    from "the upload failed".
    """
    root = Path(directory)
    if (root / "metadata.json").exists():
        return root

    candidates = sorted(
        p.parent for p in root.rglob("metadata.json")
        if "__MACOSX" not in p.parts
    )
    if not candidates:
        listing = sorted(p.name for p in root.iterdir())[:20] if root.is_dir() else []
        raise FileNotFoundError(
            f"No metadata.json found in {root} or any subdirectory — this does not "
            f"look like a b2c dataset. Top level contains: {listing or 'nothing'}"
        )
    # Shallowest wins: a dataset directory can legitimately contain a nested
    # one (a checkpoint written inside an output dir), and the outer one is
    # what was uploaded.
    return min(candidates, key=lambda p: len(p.parts))
