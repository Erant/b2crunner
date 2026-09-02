"""face_priority_weights — the face cap wins.

Three things describe the face in the stage-2 training, and until this
step they all spoke at the same volume:

  * the **face cap** — `render_face_support_views` + `face_support_views`,
    renders of the photo-derived face splat handed to `brush` as masked
    supporting views. This is the one that carries the photograph.
  * the **denoised frames** — `denoise_pass1`'s output, the training views
    proper. Transparent-mode RGBA: they carve the silhouette, and inside it
    they carry the diffusion model's idea of the face, which is what the
    cap exists to overrule.
  * the **stage-1 shells** — `pointmap_elevation_views` + `stage1_support_band`,
    Sapiens2 depth shells of every Nth denoised frame rendered from an
    elevation offset. Masked supporting views whose appearance is the
    denoised frame's, face included.

brush weights a masked view by its mask and a transparent view not at all,
so wherever the cap's renders and a denoised frame both cover the face the
fit averages them — and the cap loses as often as it wins. This step tips
that: for each view of a batch it renders the face splat's coverage from
that view's camera and turns it into a per-pixel **loss weight** that fades
the view out over the face, `1 - strength` at full coverage. The weights
reach brush as a `weights/` sidecar (see steps/brush.py and brush's
docs/loss-weights.md), which is the channel a transparent view otherwise
lacks; for the shells, which are masked already, the same weight is folded
into their mask (`masks` in, `masks` out), since a masked view's mask *is*
its per-pixel weight.

`strength` is the whole dial: 1.0 masks the face out of the other sources
entirely, which is the blunt version; the default leaves them a tenth of
their say, enough to keep the cap's rim tied to the frames around it.

**Only within the cap.** The face splat is a 2.5-D shell unprojected from
one photograph, and its renders only mean something within `cap_radius_deg`
of the photograph's own view — the same 30 degrees the cap itself is
sampled within (body2colmap's c65a7f7 measurement). A view further round
sees a side of the head the cap has no evidence for, and silencing the
denoised frame there would leave that surface constrained by nothing. So
the attenuation is full inside the cap's radius, fades linearly to nothing
over `fade_deg` beyond it, and a view past that is untouched. The angle is
measured about the splat's own centre, between the view and the anchor
camera — read LIVE from the camera list, as `render_splat`'s cap does,
because `refine_cameras` moves it.

**Coverage is the splat's own alpha**, rendered through the same binary
`render_splat` uses, feathered by a few pixels so the weight ramps rather
than steps at the splat's edge. Nothing here depends on the render's
colour; it is thrown away.

With no `splat_path` (the face branch is off) every weight is 1 and any
masks pass through unchanged, so a workflow can wire this ungated in front
of `merge_support_views` without a second switch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..masks import normalize_mask
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


def angular_attenuation(angle_deg: float, cap_radius_deg: float,
                        fade_deg: float) -> float:
    """How much of `strength` applies to a view `angle_deg` from the anchor.

    1 inside the cap, a linear ramp to 0 over `fade_deg` past its edge, 0
    beyond. A zero fade is a hard edge at the radius.
    """
    if angle_deg <= cap_radius_deg:
        return 1.0
    if fade_deg <= 0.0 or angle_deg >= cap_radius_deg + fade_deg:
        return 0.0
    return 1.0 - (angle_deg - cap_radius_deg) / fade_deg


def weight_from_coverage(coverage: np.ndarray, *, strength: float,
                         attenuation: float, feather_px: float) -> np.ndarray:
    """`1 - strength * attenuation * feathered(coverage)`, float32 in [0, 1]."""
    cov = normalize_mask(coverage)
    if feather_px > 0.0 and cov.size:
        cov = cv2.GaussianBlur(cov, (0, 0), sigmaX=float(feather_px),
                               sigmaY=float(feather_px))
    weight = 1.0 - strength * attenuation * cov
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


@register_step("face_priority_weights")
class FacePriorityWeightsStep(Step):
    """Per-pixel loss weights that fade a batch of views out over the face.

    inputs: {"cameras": List[Camera] — the views to weight,
             "splat_path": Optional[str] — the (refined) face splat .ply;
             None means the face branch is off and every weight is 1,
             "masks": Optional[List[np.ndarray]] — a masked batch's masks,
             returned multiplied by the weights,
             "anchor_cameras": Optional[List[Camera]] and
             "anchor_frame_index": Optional[int] — the camera list holding
             the photograph's view and its index; read live, because
             refine_cameras moves it,
             "anchor_position": Optional[Sequence[float]] — fallback when
             the index is not wired,
             "splat_center": Optional[Sequence[float]] — the pivot the
             angle is measured about; falls back to "orbit_target"}
    outputs: {"weights": List[np.ndarray] float32 HxW in [0, 1],
              "masks": List[np.ndarray] — only when masks were given}

    The batch's frame size comes off its cameras, which must agree — one
    COLMAP camera line covers a whole training, so they always do here.
    """

    PARAMS = (
        Param("strength", float, 0.9,
              "How far the other sources yield to the face cap where it covers "
              "them: the weight at full coverage is 1 - strength. 1.0 masks the "
              "face out of them entirely; the default leaves a tenth, which keeps "
              "the cap's rim tied to the frames around it",
              minimum=0.0, maximum=1.0),
        Param("cap_radius_deg", float, 30.0,
              "Views within this angle of the anchor camera's view of the splat "
              "yield in full. Match render_face_support_views' cap_radius_deg: it "
              "is where the 2.5-D face shell still reads cleanly, and past it "
              "the shell has nothing to say that could replace what is silenced",
              minimum=0.0, maximum=180.0),
        Param("fade_deg", float, 15.0,
              "Past cap_radius_deg the attenuation ramps linearly to nothing over "
              "this many degrees; 0 is a hard edge", minimum=0.0, maximum=180.0),
        Param("feather_px", float, 4.0,
              "Gaussian sigma, in pixels, applied to the splat's coverage before it "
              "becomes a weight, so the weight ramps at the splat's edge instead "
              "of stepping; 0 uses the coverage as rendered", minimum=0.0),
        Param("render_path", str, None,
              "The rasteriser binary; empty uses render_splat's default",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cameras = list(inputs["cameras"])
        masks = inputs.get("masks")
        splat_path = inputs.get("splat_path")
        strength = float(params["strength"])

        if masks is not None and len(masks) != len(cameras):
            raise ValueError(
                f"face_priority_weights: {len(masks)} masks for {len(cameras)} "
                f"cameras. The masks are the views' own and have to arrive with them."
            )

        if not cameras:
            return self._passthrough([], masks, why="no views")
        if not splat_path:
            return self._passthrough(
                cameras, masks,
                why="no face splat was wired in (the face branch is off)",
            )
        if strength <= 0.0:
            return self._passthrough(cameras, masks, why="strength is 0")

        width, height = _frame_size(cameras)
        coverage = _render_coverage(
            splat_path, cameras, width=width, height=height,
            render_path=params["render_path"],
        )

        pivot = _pivot(inputs)
        anchor = _anchor_position(inputs)
        anchor_dir = _unit(anchor - pivot)
        cap_radius = float(params["cap_radius_deg"])
        fade = float(params["fade_deg"])
        feather = float(params["feather_px"])

        weights: List[np.ndarray] = []
        attenuated = 0
        covered_weights: List[float] = []
        for camera, cov in zip(cameras, coverage):
            view_dir = _unit(np.asarray(camera.position, dtype=np.float64).reshape(3) - pivot)
            angle = float(np.degrees(np.arccos(np.clip(np.dot(view_dir, anchor_dir), -1.0, 1.0))))
            attenuation = angular_attenuation(angle, cap_radius, fade)
            weight = weight_from_coverage(
                cov, strength=strength, attenuation=attenuation, feather_px=feather,
            )
            weights.append(weight)
            inside = normalize_mask(cov) > 0.5
            if attenuation > 0.0 and inside.any():
                attenuated += 1
                covered_weights.append(float(weight[inside].mean()))

        logger.info(
            "face_priority_weights: %d/%d views yield to the face cap (within "
            "%.0f+%.0f deg of the anchor and covered by the splat), mean weight "
            "over the face %s; strength %.2f",
            attenuated, len(cameras), cap_radius, fade,
            f"{np.mean(covered_weights):.2f}" if covered_weights else "n/a",
            strength,
        )
        if attenuated == 0:
            logger.warning(
                "face_priority_weights: no view is both within the cap and covered "
                "by the face splat, so nothing yields. Either the batch sits outside "
                "the cap (fine for a batch that never sees the face) or the splat "
                "is not where these cameras look."
            )

        result: Dict[str, Any] = {"weights": weights}
        if masks is not None:
            result["masks"] = [
                (normalize_mask(mask) * weight).astype(np.float32)
                for mask, weight in zip(masks, weights)
            ]
        return result

    @staticmethod
    def _passthrough(cameras, masks, *, why: str) -> Dict[str, Any]:
        logger.info("face_priority_weights: %s; every weight is 1", why)
        if cameras:
            width, height = _frame_size(cameras)
            weights = [np.ones((height, width), dtype=np.float32) for _ in cameras]
        else:
            weights = []
        result: Dict[str, Any] = {"weights": weights}
        if masks is not None:
            result["masks"] = [normalize_mask(m) for m in masks]
        return result


def _frame_size(cameras) -> tuple:
    sizes = {(int(c.width), int(c.height)) for c in cameras}
    if len(sizes) != 1:
        raise ValueError(
            f"face_priority_weights: the cameras disagree on the frame size "
            f"({sorted(sizes)}); one batch renders at one size."
        )
    (width, height), = sizes
    return width, height


def _render_coverage(splat_path: str, cameras, *, width: int, height: int,
                     render_path: Optional[str]) -> List[np.ndarray]:
    """The face splat's alpha from every camera, float32 HxW in [0, 1]."""
    from body2colmap.splat_scene import SplatScene

    from .splat import _RENDER_BINARY, _rasterize

    scene = SplatScene.from_ply(str(splat_path))
    image_names = [f"coverage_{i + 1:05d}_.png" for i in range(len(cameras))]
    logger.info(
        "face_priority_weights: rendering the face splat's coverage (%d Gaussians) "
        "from %d cameras at %dx%d", len(scene), len(cameras), width, height,
    )
    _images, masks = _rasterize(
        scene=scene,
        splat_path=str(splat_path),
        cameras=cameras,
        image_names=image_names,
        width=width,
        height=height,
        bg_color=(0.0, 0.0, 0.0),
        render_path=render_path or _RENDER_BINARY,
        confidence=None,
    )
    return [np.asarray(m, dtype=np.float32) for m in masks]


def _pivot(inputs: Dict[str, Any]) -> np.ndarray:
    from .anchor_stub import _composite_pivot

    return _composite_pivot(inputs, where="face_priority_weights")


def _anchor_position(inputs: Dict[str, Any]) -> np.ndarray:
    """The photograph's camera position: live from the list, else recorded."""
    cameras = inputs.get("anchor_cameras")
    index = inputs.get("anchor_frame_index")
    if cameras and index is not None and 0 <= int(index) < len(cameras):
        return np.asarray(cameras[int(index)].position, dtype=np.float64).reshape(3)
    position = inputs.get("anchor_position")
    if position is not None:
        return np.asarray(position, dtype=np.float64).reshape(3)
    raise ValueError(
        "face_priority_weights: wire 'anchor_cameras' + 'anchor_frame_index' (the "
        "camera list and the photograph's index in it) or 'anchor_position'. The "
        "cap is measured from the photograph's view of the splat, and there is "
        "no view to measure from."
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(
            "face_priority_weights: a camera sits on the splat's centre, so it has "
            "no view direction."
        )
    return vector / norm
