"""delight_references — take the camera's headlight out of pass 1's two real images.

Why (the lighting study, 2026-10-27/28, /data/experiments/lighting): Wan pass
1 copies the lighting of the images it conditions on, and both real ones, the
sheet's back panel and the warped front photo, are lit from their own camera.
That headlight (bright where the surface faces the lens, darker toward the
silhouette) is baked into every frame and so into the splat, where it reads as
painted-on shading once the rig is posed and lit by something else. The DL arm
of the study, these two images delit before pass 1, won every lighting metric
on the final splat (headlight slope hl45 +0.51 -> +0.05, cam_b z +0.79 ->
+0.23) against the shipped baseline and the re-prompted sheets.

The model: each image is lit from its own camera, so per segmentation label

    log2 lum = a + b * n_z            (n_z = Sapiens2 normal toward the lens)

fitted robustly over that label's pixels (b clamped >= 0), and the image
divided by 2^(b * (n_z - 1)) in linear light: facing pixels unchanged, grazing
ones lifted, the lift capped at `max_lift` stops. n_z is blurred first so
normal noise does not become texture. The background (label 0) is left alone.

inputs:  {"reference_image": BGR uint8 | None — the matted back panel (None on a single photo),
          "reference_labels": HxW int | None, "reference_normals": HxWx3 | None — its Sapiens2 maps,
          "anchor_image": BGR uint8 — the warped front photo,
          "anchor_labels": HxW int, "anchor_normals": HxWx3}
outputs: {"reference_image": BGR uint8 | None, "anchor_image": BGR uint8,
          "slopes": {"reference": {label: b}, "anchor": {label: b}}}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step
from .relight import linear_to_srgb, srgb_to_linear

logger = logging.getLogger(__name__)

_EPS = 1e-3


def _log_lum(bgr: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(bgr[..., :3].astype(np.float32) / 255.0)
    return np.log2(0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2] + _EPS)


def delight(image: np.ndarray, labels: np.ndarray, normals: np.ndarray, max_lift: float,
            blur_sigma: float, min_px: int) -> Tuple[np.ndarray, Dict[int, float]]:
    """One image delit; returns it and the headlight slope b fitted per label."""
    import cv2

    nz = cv2.GaussianBlur(normals[..., 2].astype(np.float32), (0, 0), blur_sigma)
    L = _log_lum(image)
    stops = np.zeros(image.shape[:2], np.float32)
    fits: Dict[int, float] = {}
    fg = labels > 0
    for k in np.unique(labels[fg]):
        m = (labels == k) & (nz > 0.05)
        if m.sum() < min_px:
            continue
        A = np.column_stack([np.ones(m.sum()), nz[m]])
        w = np.ones(m.sum())
        for _ in range(5):  # IRLS (Huber-ish) so speculars and print do not drive the slope
            coef = np.linalg.lstsq(A * w[:, None], L[m] * w, rcond=None)[0]
            r = L[m] - A @ coef
            s = 1.4826 * np.median(np.abs(r)) + 1e-6
            w = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-6)))
        b = max(float(coef[1]), 0.0)
        fits[int(k)] = round(b, 3)
        sel = labels == k
        stops[sel] = np.clip(-b * (nz[sel] - 1.0), 0.0, max_lift)
    stops = cv2.GaussianBlur(stops, (0, 0), 1.0) * fg
    lin = srgb_to_linear(image[..., :3].astype(np.float32) / 255.0) * np.exp2(stops)[..., None]
    out = (linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
    return out, fits


@register_step("delight_references")
class DelightReferencesStep(Step):
    PARAMS = (
        Param("max_lift", float, 1.5, "Most a grazing pixel is brightened, in stops", minimum=0.0),
        Param("blur_sigma", float, 3.0, "Blur of the normals' z before the fit, in pixels",
              minimum=0.0, advanced=True),
        Param("min_px", int, 2000, "Fewest pixels a label needs to get a fit of its own",
              minimum=1, advanced=True),
        Param("debug_dir", str, None, "Write a before/after preview here", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"reference_image": None}
        slopes: Dict[str, Dict[int, float]] = {}
        before_after = []
        for name in ("reference", "anchor"):
            image = inputs.get(f"{name}_image")
            if image is None:
                continue
            labels, normals = inputs.get(f"{name}_labels"), inputs.get(f"{name}_normals")
            if labels is None or normals is None:
                raise ValueError(f"delight_references: the {name} image came without its labels or normals")
            if labels.shape[:2] != image.shape[:2] or normals.shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"delight_references: {name} is {image.shape[:2]}, its labels {labels.shape[:2]} "
                    f"and normals {normals.shape[:2]}: they must be the image's own maps"
                )
            delit, slopes[name] = delight(image, np.asarray(labels), np.asarray(normals),
                                          params["max_lift"], params["blur_sigma"], params["min_px"])
            out[f"{name}_image"] = delit
            before_after.append((image, delit))
            logger.info("delight_references: %s headlight slope per label %s", name, slopes[name])
        if "anchor_image" not in out:
            raise ValueError("delight_references: no anchor image")
        if params["debug_dir"]:
            import cv2

            h = 960
            tiles = [cv2.resize(x[..., :3], (round(x.shape[1] * h / x.shape[0]), h))
                     for pair in before_after for x in pair]
            Path(params["debug_dir"]).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(params["debug_dir"]) / "delight_preview.jpg"), np.hstack(tiles))
        out["slopes"] = slopes
        return out
