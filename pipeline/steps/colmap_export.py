"""Standalone COLMAP sparse-reconstruction export — the on-disk format
brush and other 3DGS tools consume directly (cameras.txt/images.txt/
points3D.txt + frame_NNNNN_.png, optionally RGBA with masks as alpha, plus
an optional normals/ directory).

Two layouts, via `params["layout"]`:

    flat    frames beside the .txt files. What the ComfyUI colmap.json
            stage wrote (cyber_6f/colmap), and the default, because that
            is what the golden comparison below is against.
    brush   frames in images/, normals in normals/, .txt files at the
            root — what steps/brush.py builds in its tempdir before
            invoking the trainer, and what the `fast_helical` workflows
            use for the COLMAP dataset a run hands back.

Supporting views (`support_*`) ride along in the `brush` layout, written
exactly the way steps/brush.py writes them into its own tempdir — RGB in
`images/` with the matte beside it in `masks/`, and their cameras appended
to the same `images.txt`. That is the whole point of sharing brush's
`_SupportViews`: what this exports is not "a dataset plus some extra
pictures", it is the model brush was handed, so a supporting view landing
in the wrong place in the trained splat is visible in the export rather
than only in the result. `flat` has no `masks/` convention to write them
into and refuses them.

Note what neither layout does: rescale intrinsics. The cameras written
here are whatever the dataset holds, so a dataset whose frames were
resized without its cameras being updated exports a cameras.txt that
disagrees with its own images — see steps/seedvr2.py, the one step that
resizes frames, which rescales the cameras that describe them to match as
part of the same step.

Ported from nodes/export_node.py in the original ComfyUI-Body2COLMAP repo,
minus the ComfyUI-specific parts (folder_paths output-dir resolution,
INPUT_IS_LIST batch merging, tensor conversion). The actual export is
body2colmap's `ColmapExporter` — this step is a thin wrapper around it,
same image/normal-map/alpha-combination logic pipeline/steps/brush.py
already uses internally (against its own tempdir) — this step is that
same logic exposed as its own step, writing to a permanent directory
instead, for producing a COLMAP dataset without going through brush.
VERIFIED against recorded output: `cyber_6f/upscaled` -> `cyber_6f/colmap`
is a real run of `workflows/api/colmap.json`, and this step reproduces its
cameras.txt and points3D.txt byte-for-byte and its images.txt to 2.4e-7
per value (the poses have been through metadata.json's float round-trip).
The exported PNGs are not compared — that ComfyUI stage runs RMBG over the
frames first. See tests/test_colmap_export.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from ..masks import mask_to_alpha_u8
from ..registry import register_step
from ..step import REQUIRED, Param, Step


LAYOUTS = ("flat", "brush")


@register_step("colmap_export")
class ColmapExportStep(Step):
    """Write a COLMAP sparse reconstruction directory.

    inputs: {"cameras": List[Camera], "image_names": List[str],
             "points_3d": Tuple[np.ndarray, np.ndarray],
             "images": Optional[List[np.ndarray]] BGR(A),
             "masks": Optional[List[np.ndarray]] float32 [0,1], foreground=1,
             "normal_maps": Optional[List[np.ndarray]] HxWx3 float32 [-1,1],
             "weights": Optional[List[np.ndarray]] float32 [0,1] per training
                        view, written as brush's weights/ sidecar (brush layout),
             "support_cameras": Optional[List[Camera]],
             "support_images": Optional[List[np.ndarray]] BGR(A),
             "support_masks": Optional[List[np.ndarray]] float32 [0,1],
             "support_image_names": Optional[List[str]],
             "support_normal_maps": Optional[List[np.ndarray]]}
    outputs: {"output_path": str}

    The `support_*` inputs are steps/brush.py's, read by the same
    `_SupportViews` and written by the same code, so wiring this step and
    `brush` to the same context paths exports the model brush trains on
    rather than a near-miss of it. They need `layout: brush`. Absent (the
    default), this builds byte-identical output to before they existed.
    """

    PARAMS = (
        Param("output_dir", str, REQUIRED, "Directory to write the COLMAP dataset into"),
        Param("layout", str, "flat",
              "flat: frames beside the .txt files (the ComfyUI stage's shape, and what "
              "the golden test compares against). brush: images/ and normals/ "
              "subdirectories, which is what a dataset handed to somebody wants",
              choices=("flat", "brush")),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.exporter import ColmapExporter

        from .brush import _SupportViews, _loss_weights, write_loss_weights

        cameras = inputs["cameras"]
        image_names = inputs["image_names"]
        points_3d = inputs["points_3d"]
        images = inputs.get("images")
        masks = inputs.get("masks")
        normal_maps = inputs.get("normal_maps")

        if images is not None and len(images) != len(image_names):
            raise ValueError(f"images ({len(images)}) and image_names ({len(image_names)}) length mismatch")
        if normal_maps is not None and images is not None and len(normal_maps) != len(images):
            raise ValueError(
                f"Normal map count ({len(normal_maps)}) does not match image count "
                f"({len(images)}). Every training view needs a matching normal map."
            )

        layout = params["layout"]
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}; expected one of {LAYOUTS}.")

        weights = _loss_weights(inputs, len(image_names))
        if weights is not None and layout != "brush":
            raise ValueError(
                f"weights were wired in but layout is {layout!r}. A loss-weight map "
                f"is a weights/ sidecar, and only the brush layout has a directory "
                f"to put it in — flat has nowhere to record it, which is a dataset "
                f"that trains at a different weighting from the one that was run."
            )

        # Same reader, same validation, same error messages as the training
        # step's — including the stem-collision check against the training
        # names, which matters here for the same reason it matters there.
        support = _SupportViews.from_inputs(inputs, image_names)
        if support and layout != "brush":
            raise ValueError(
                f"support_* views were wired in but layout is {layout!r}. A "
                f"supporting view is an RGB frame plus a matte in a masks/ "
                f"sidecar, and only the brush layout has a masks/ directory to "
                f"put it in — flat would write the frames beside the .txt files "
                f"with their mattes nowhere, which is a dataset that says the "
                f"opposite of what it means."
            )

        output_path = Path(params["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)

        # `flat` is what the ComfyUI colmap.json stage wrote (frames sitting
        # beside the .txt files — see cyber_6f/colmap), and stays the default
        # so the golden comparison in tests/test_colmap_export.py keeps
        # meaning what it says. `brush` is the layout a dataset meant to be
        # handed to somebody wants: images and normals in their own
        # directories, which is also exactly what steps/brush.py builds in
        # its tempdir before invoking the trainer.
        images_dir = output_path / "images" if layout == "brush" else output_path
        images_dir.mkdir(parents=True, exist_ok=True)

        support.check_intrinsics(cameras)
        # One model covering both, exactly as steps/brush.py builds it: the
        # supporting views are extra images.txt entries against the same
        # cameras.txt, and differ only in how their frames are written.
        ColmapExporter(
            cameras=list(cameras) + support.cameras,
            image_names=list(image_names) + support.image_names,
            points_3d=points_3d,
        ).export(output_dir=output_path)

        alpha_channel = None
        if images is not None:
            if masks is not None:
                # Handles both mask ranges in circulation — a mask loaded
                # from disk is uint8, not float [0,1]. See pipeline/masks.py.
                alpha_channel = [mask_to_alpha_u8(m) for m in masks]

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

        if normal_maps is not None and images is not None:
            normals_dir = output_path / "normals"
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

        write_loss_weights(output_path, image_names, weights)
        support.write(output_path)

        return {"output_path": str(output_path.absolute())}
