"""reupscale_inputs — the deliverable splat's own renders, ready for a SeedVR2 pass and a retraining.

Why (measured 2026-10-02 on 9ddb62, plain trainer flags, scratchpad study in
memory `render_sr_retrain`): the trained splat holds about half the sharpness
of the frames it was fitted to (band-limited s1 7.7 against 13.9; the head
11.0 against 21.5). The frames' extra detail disagrees between views and
averages away. Rendering the splat back out gives frames that agree with
each other by construction; SeedVR2 then adds detail to clean, consistent
inputs, and a splat retrained on them kept it: s1 12.6, head 19.5 (+63 %,
+77 %), the same at cameras between the training views (12.7 / 19.0), for
0.6 dB of fidelity to the real frames on top of the 1.4 dB that retraining
on renders costs at all. Retraining on the renders bicubic-upscaled scored
6.7 (the control); 1080 renders carry little the 720 ones do not, so the
render is 720x1280 whatever the dataset's resolution.

This step: a fresh set of views, not the orbit's. By now the splat exists
and nothing is denoised as a video, so the cameras need not follow a smooth
path: they sit on rings of even elevation from `min_elevation_deg` to
`max_elevation_deg` (default -70..+70), each ring holding a share of the
views proportional to its circumference, so neighbouring views are about the
same angle apart everywhere on the band. Rings alternate direction (and
offset by half a step), so consecutive views stay neighbours on the sphere —
the trainer's per-view body-rig rotations are smoothed between consecutive
views. Centre, radius and intrinsics come from the training cameras (the
point they look at, their median distance from it, the first camera's lens
rescaled to the render size). The splat is rendered over 0.5 grey and its
alpha, bilinear to the size SeedVR2 hands back, is the retraining's matte:
the frames' own mattes and class maps belong to the orbit's views (the class
maps are made again on the upscaled renders, `reupscale_segment`). The
splat it starts from is copied aside first: the retraining exports over it.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


def band_cameras(cameras: List[Any], n_views: int, min_elevation_deg: float, max_elevation_deg: float,
                 width: int, height: int) -> List[Any]:
    """`n_views` cameras spread evenly over the elevation band, looking at where `cameras` look.

    Rings at evenly spaced elevations, as many as make the ring spacing match
    the spacing along a ring; each ring's count proportional to cos(elevation)
    (largest remainder, so the counts sum to `n_views`). Azimuth starts at
    the first camera's, odd rings run backwards and half a step offset.
    """
    from body2colmap import coordinates
    from body2colmap.camera import Camera

    from .backdrop import orbit_frame

    if n_views < 1:
        raise ValueError("band_cameras: n_views must be at least 1")
    lo, hi = float(min_elevation_deg), float(max_elevation_deg)
    if not -90.0 < lo <= hi < 90.0:
        raise ValueError(f"band_cameras: elevations {lo}..{hi} must lie inside (-90, 90), low first")
    center = orbit_frame(cameras)[0].astype(np.float64)
    radius = float(np.median([np.linalg.norm(np.asarray(c.position, np.float64).reshape(3) - center)
                              for c in cameras]))
    up = np.asarray(coordinates.WorldCoordinates.UP_AXIS, np.float64)

    # Rings: band area / n_views is the area per view; its square root the spacing.
    lo_r, hi_r = np.radians(lo), np.radians(hi)
    spacing = np.sqrt(2.0 * np.pi * (np.sin(hi_r) - np.sin(lo_r)) / n_views) if hi > lo else 0.0
    n_rings = 1 if spacing <= 0.0 else max(1, min(n_views, int(round((hi_r - lo_r) / spacing)) + 1))
    elevations = np.linspace(lo, hi, n_rings) if n_rings > 1 else np.array([(lo + hi) / 2.0])
    share = np.cos(np.radians(elevations))
    share = share / share.sum() * n_views
    counts = np.floor(share).astype(int)
    for i in np.argsort(-(share - counts))[:n_views - int(counts.sum())]:
        counts[i] += 1

    # Azimuth of the first training camera around the up axis (body2colmap: 0 = +Z, 90 = +X).
    offset = np.asarray(cameras[0].position, np.float64).reshape(3) - center
    start = float(np.degrees(np.arctan2(offset[0], offset[2])))

    ref = cameras[0]
    sx, sy = width / float(ref.width), height / float(ref.height)
    out = []
    for ring, (elevation, count) in enumerate(zip(elevations, counts)):
        if count == 0:
            continue
        step = 360.0 / count
        sign = -1.0 if ring % 2 else 1.0
        for k in range(count):
            azimuth = start + sign * (k + (0.5 if ring % 2 else 0.0)) * step
            position = center + np.asarray(coordinates.spherical_to_cartesian(radius, azimuth, float(elevation)),
                                           np.float64)
            cam = Camera(focal_length=(float(ref.fx) * sx, float(ref.fy) * sy), image_size=(width, height),
                         principal_point=(float(ref.cx) * sx, float(ref.cy) * sy),
                         position=position.astype(np.float32), rotation=np.eye(3, dtype=np.float32))
            cam.look_at(center.astype(np.float32), up.astype(np.float32))
            out.append(cam)
    return out


@register_step("reupscale_inputs")
class ReupscaleInputsStep(Step):
    """Render the trained splat over an even band of views for a SeedVR2 pass, mattes at the upscaled size.

    inputs: {"splat_path": str, "cameras": List[Camera] — the training cameras, for centre, radius and lens}
    outputs: {"images": List[np.ndarray] BGR uint8 at render_width x render_height (for seedvr2), one per view,
              "cameras": List[Camera] the band's views, at the render size (seedvr2 rescales them with its frames),
              "image_names": List[str] one per view,
              "masks": List[np.ndarray] uint8 at target_width x target_height — the render's alpha, one per view}
    """

    PARAMS = (
        Param("render_width", int, 720, "Width the splat is rendered at", minimum=1),
        Param("render_height", int, 1280, "Height the splat is rendered at", minimum=1),
        Param("target_width", int, 1080, "Width SeedVR2 hands back (its `resolution` is the short edge)", minimum=1),
        Param("target_height", int, 1920, "Height SeedVR2 hands back", minimum=1),
        Param("views", int, 0, "Views on the band; 0 = as many as there are training cameras", minimum=0),
        Param("min_elevation_deg", float, -70.0, "Lowest view elevation in degrees (negative is below eye level)",
              minimum=-89.0, maximum=89.0),
        Param("max_elevation_deg", float, 70.0, "Highest view elevation in degrees", minimum=-89.0, maximum=89.0),
        # render_splat's `sh_degree`: clamped to the splat's own degree. 2 drops
        # band 3, the most view-dependent, from what SeedVR2 sees and the
        # retraining fits.
        Param("sh_degree", int, 2,
              "Highest spherical-harmonic band rendered, 0-3 (0 = base colour only)", minimum=0, maximum=3),
        Param("keep_copy", str, "", "Copy the splat here before the retraining exports over it"),
        Param("render_path", str, "brush-splat-render", "The rasteriser binary", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        splat_path = Path(inputs["splat_path"])
        cameras = list(inputs["cameras"])
        if not cameras:
            raise ValueError("reupscale_inputs: no cameras")
        rw, rh = int(params["render_width"]), int(params["render_height"])
        tw, th = int(params["target_width"]), int(params["target_height"])

        if params["keep_copy"]:
            dst = Path(params["keep_copy"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(splat_path, dst)
            logger.info("reupscale_inputs: kept the splat as it was at %s", dst)

        from body2colmap.splat_scene import SplatScene

        from .splat import _rasterize

        n_views = int(params["views"]) or len(cameras)
        lo, hi = float(params["min_elevation_deg"]), float(params["max_elevation_deg"])
        render_cams = band_cameras(cameras, n_views, lo, hi, rw, rh)
        names = [f"reupscale_{i + 1:05d}.png" for i in range(len(render_cams))]
        images, alphas = _rasterize(
            scene=SplatScene.from_ply(str(splat_path)), splat_path=str(splat_path), cameras=render_cams,
            image_names=names, width=rw, height=rh, bg_color=(0.5, 0.5, 0.5), render_path=params["render_path"],
            sh_degree=int(params["sh_degree"]))

        # The render's own alpha is the matte: these views have no frame of their own. Bilinear up to the size
        # SeedVR2 hands back.
        out_masks = [np.clip(cv2.resize(normalize_mask(m), (tw, th), interpolation=cv2.INTER_LINEAR) * 255.0 + 0.5,
                             0, 255).astype(np.uint8) for m in alphas]
        logger.info("reupscale_inputs: %d views of %s on rings from %.0f to %.0f deg elevation, rendered at %dx%d "
                    "with SH bands 0..%d for the upscale and the retraining; mattes at %dx%d", len(images),
                    splat_path.name, lo, hi, rw, rh, int(params["sh_degree"]), tw, th)
        return {"images": images, "cameras": render_cams, "image_names": names, "masks": out_masks}
