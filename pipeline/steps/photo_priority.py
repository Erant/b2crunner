"""photo_priority_weights — the photograph wins over the whole surface it sees.

`face_priority_weights` lets the face cap overrule the denoised frames over
the face. Everything else the photograph shows — the front of the jacket,
the hands, the skirt's front — is trained from the photograph AND the ten
or so denoised frames either side of it, which are the diffusion model's
re-lit, slightly displaced re-drawings of the same surface (5-10 px above
the photograph in every run, see docs/vace-denoise-findings-2026-09-07.md),
and the fit averages them into a softer, drifted front. The mesh path's one
unambiguous win was the opposite rule (`photo_texture`, 2026-09-16: every
texel the photograph's camera sees is the photograph's pixel; whole-subject
sharpness 39.5 -> 51.5 on 00307). This step is that rule for the splat.

For each training view the body mesh (`mesh_world`, SAM-3D-Body's, the
frame the splat is trained in) is rasterised from the view's camera; each
hit pixel's surface point is tested against the photograph's camera — in
its frame, not behind the body from it, and facing it with a cosine that
ramps from `facing_lo` to `facing_hi` (photo_texture's skin/cloth ramp) —
and that confidence, feathered, becomes the yield: the view's weight there
is `1 - strength x attenuation x confidence`. The photograph's own frame
(`anchor_frame_index`) keeps weight 1: it is the source. Only that one: the
helix's LAST frame sits on the same pose and is the denoiser's repaint of
the photograph (measured 2026-09-19 on a pod run: 14.5 dB from it, another
face); fading it is worth 0.35 dB at the photograph's view — the copies
below are the larger part.
The angular window (`cap_radius_deg` + `fade_deg`, about the subject's centre, measured from the anchor camera read live from
the list) is the same shape as the cap's; here it is wide, because the
per-pixel visibility and facing already localise the yield, and it only
keeps a view round the back from being touched at all.

Off the body mesh — the hair, the skirt's flare, a loose sleeve — there is
no surface to test. Those pixels take the nearest on-mesh pixel's
confidence within `extend_px` (the same rule `mesh_raster.hit_surface`
uses for the cap's hair rim) and nothing beyond it, clipped to the frame's
own alpha when `alphas` is wired. The rest of the frame keeps weight 1.

Fading the neighbours is only half of it. Measured on the 2026-09-18 pod
run's own intermediate export (docs/photo-priority.md): with the frames
faded to nothing where the photograph sees the body, the photograph still
reached only 20.6 dB at its own view, and the confidence render greyed out
45 % of the front — one frame is one vote, and one supporting view is
below the evidence gate's `conf-min-views`. So the step also hands the
training `copies` (6) of the photograph's frame as MASKED supporting views
at the anchor camera, each masked by the photograph's own confidence field
(its facing cosine from its own camera, extended off the body, clipped to
its matte): 23.1 dB at the photograph's view from the copies alone (23.3
at strength 0.5, 23.5 at 0.8), sharpness there x2.8, and the cull no worse
than the untouched run's, because every copy is a view the evidence pass
counts. The fade is the smaller half and a fidelity dial: 0.8 costs 1.5 dB
against the frames beside the photograph for +0.4 dB at it. `images` (the
training frames) has to be wired for the copies; without it none are made.

The weights multiply whatever `weights` were wired in (the face cap's), so
the two yields stack: over the face the cap's rule already silences the
frames; this one adds the body. With `strength` 0 or no mesh the input
weights pass through unchanged (weights of 1 when there were none), so the
workflow wires this ungated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..masks import normalize_mask
from ..mesh_raster import _FLIP, Raster, rasterize, vertices_in_camera
from ..registry import register_step
from ..step import Param, Step
from .face_priority import angular_attenuation

logger = logging.getLogger(__name__)


def ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 at or below `lo`, 1 at or above `hi`, linear between."""
    if hi <= lo:
        return (x >= hi).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def raster_view(mesh_world: Tuple[np.ndarray, np.ndarray], camera: Any) -> Raster:
    """The body from `camera`, at the camera's own size."""
    vertices, faces = mesh_world
    return rasterize(vertices_in_camera(vertices, camera), faces,
                     fx=float(camera.fx), fy=float(camera.fy), cx=float(camera.cx), cy=float(camera.cy),
                     width=int(camera.width), height=int(camera.height))


def surface_points(raster: Raster, camera: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(the hit pixels' world points (n,3), their world normals toward the camera (n,3), the hit mask HxW).

    The inverse of `mesh_raster.vertices_in_camera`: a pixel (v, u) at depth z
    is `((u - cx) / fx, (v - cy) / fy, 1) * z` in the OpenCV camera frame,
    and world = position + (p * FLIP) @ R^T for a camera-to-world `rotation`.
    """
    hit = raster.hit & raster.facing
    rows, cols = np.nonzero(hit)
    z = raster.depth[rows, cols]
    p = np.stack([(cols - float(camera.cx)) / float(camera.fx) * z,
                  (rows - float(camera.cy)) / float(camera.fy) * z, z], 1)
    rotation = np.asarray(camera.rotation, np.float64).reshape(3, 3)
    position = np.asarray(camera.position, np.float64).reshape(3)
    points = (p * _FLIP) @ rotation.T + position
    normals = (raster.normal[rows, cols] * _FLIP) @ rotation.T
    return points, normals, hit


def photo_confidence(points: np.ndarray, normals: np.ndarray, anchor: Any, anchor_depth: np.ndarray, *,
                     margin: float, facing_lo: float, facing_hi: float) -> np.ndarray:
    """How much the photograph owns each surface point, in [0, 1].

    Seen by the photograph's camera (in front of it, inside its frame, and
    not further than `margin` behind the body's own depth there) and facing
    it: the cosine between the surface normal and the direction to the
    camera, ramped from `facing_lo` to `facing_hi`.
    """
    rotation = np.asarray(anchor.rotation, np.float64).reshape(3, 3)
    position = np.asarray(anchor.position, np.float64).reshape(3)
    p = ((points - position) @ rotation) * _FLIP
    z = p[:, 2]
    ahead = z > 1e-6
    zs = np.where(ahead, z, 1.0)
    u = np.rint(float(anchor.fx) * p[:, 0] / zs + float(anchor.cx)).astype(np.int64)
    v = np.rint(float(anchor.fy) * p[:, 1] / zs + float(anchor.cy)).astype(np.int64)
    width, height = int(anchor.width), int(anchor.height)
    inside = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    visible = np.zeros(len(points), bool)
    if inside.any():
        body = anchor_depth[v[inside], u[inside]]
        visible[inside] = ~(np.isfinite(body) & (z[inside] > body + margin))
    to_camera = position[None, :] - points
    to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-12)
    cosine = np.einsum("ij,ij->i", normals, to_camera)
    return ramp(cosine, facing_lo, facing_hi) * visible


def extend_off_mesh(conf: np.ndarray, hit: np.ndarray, extend_px: float, alpha: Optional[np.ndarray]) -> np.ndarray:
    """`conf` (HxW, meaningful where `hit`) carried to the pixels within `extend_px` of the
    body's silhouette (the nearest hit pixel's value — the hair rim, the skirt's flare), clipped
    to `alpha` where one is given. Everything further from the body is 0."""
    if not hit.any():
        return np.zeros_like(conf, dtype=np.float32)
    out = np.where(hit, conf, 0.0).astype(np.float32)
    if extend_px > 0:
        dist, labels = cv2.distanceTransformWithLabels((~hit).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(int(labels.max()) + 1, np.int64)
        lut[labels[hit]] = np.flatnonzero(hit)
        near = (~hit) & (dist <= extend_px)
        out[near] = out.reshape(-1)[lut[labels[near]]]
    if alpha is not None:
        out *= normalize_mask(alpha)
    return out


def yield_weight(conf: np.ndarray, *, strength: float, attenuation: float, feather_px: float) -> np.ndarray:
    """`1 - strength * attenuation * feathered(conf)`, float32 in [0, 1]."""
    c = np.asarray(conf, np.float32)
    if feather_px > 0.0 and c.size:
        c = cv2.GaussianBlur(c, (0, 0), sigmaX=float(feather_px), sigmaY=float(feather_px))
    return np.clip(1.0 - strength * attenuation * c, 0.0, 1.0).astype(np.float32)


def anchor_frames(cameras: Sequence[Any], anchor_position: np.ndarray, tolerance_pct: float) -> List[int]:
    """The views sitting on the photograph's camera (the injected anchor, both ends of a
    helix that starts and ends there): within `tolerance_pct` of the camera bounding-box
    diagonal, the rule `inject_anchor` matches by."""
    positions = np.stack([np.asarray(c.position, np.float64).reshape(3) for c in cameras])
    scale = float(np.linalg.norm(positions.max(0) - positions.min(0)))
    if scale <= 0.0:
        scale = 1.0
    threshold = tolerance_pct / 100.0 * scale
    distances = np.linalg.norm(positions - anchor_position[None, :], axis=1)
    return [int(i) for i in np.flatnonzero(distances <= threshold)]


@register_step("photo_priority_weights")
class PhotoPriorityWeightsStep(Step):
    """Per-pixel loss weights that fade a batch of views out wherever the photograph sees the body.

    inputs: {"cameras": List[Camera] — the views to weight,
             "mesh_world": Optional[(vertices, faces)] — the body in the world
             frame; None means every weight passes through,
             "weights": Optional[List[np.ndarray]] — weights to multiply
             (face_priority's); None means 1,
             "alphas": Optional[List[np.ndarray]] — the views' own mattes,
             clipping the yield off the body,
             "anchor_cameras": Optional[List[Camera]] and
             "anchor_frame_index": Optional[int] — the camera list holding the
             photograph's view and its index; read live, because
             refine_cameras moves it,
             "anchor_position": Optional[Sequence[float]] — fallback,
             "splat_center": Optional[Sequence[float]] — the pivot the angle is
             measured about; the body's centre when not given,
             "images": Optional[List[np.ndarray]] — the training frames, so the
             photograph's own frame can be copied as supporting views}
    outputs: {"weights": List[np.ndarray] float32 HxW in [0, 1],
              "support_images": List[np.ndarray] BGR — `copies` of the photograph,
              "support_masks": List[np.ndarray] float32 — its confidence field,
              "support_cameras": List[Camera] — the anchor camera, repeated,
              "photo_priority_stats": dict}
    """

    PARAMS = (
        Param("strength", float, 0.5,
              "How far the views yield to the photograph where it sees the body: the weight "
              "at full confidence is 1 - strength. 0 is off (the input weights pass through)",
              minimum=0.0, maximum=1.0),
        Param("facing_lo", float, 0.2,
              "The photograph's confidence ramps in from this cosine between the surface and "
              "its camera (photo_texture's skin/cloth ramp) ...", minimum=-1.0, maximum=1.0),
        Param("facing_hi", float, 0.5, "... to full at this one", minimum=-1.0, maximum=1.0),
        Param("cap_radius_deg", float, 45.0,
              "Views within this angle of the anchor camera (about the subject's centre) yield in "
              "full; the per-pixel visibility already localises the yield, this only keeps the "
              "views round the back untouched", minimum=0.0, maximum=180.0),
        Param("fade_deg", float, 45.0,
              "Past cap_radius_deg the yield ramps linearly to nothing over this many degrees",
              minimum=0.0, maximum=180.0),
        Param("feather_px", float, 4.0,
              "Gaussian sigma, in pixels, over the confidence before it becomes a weight",
              minimum=0.0),
        Param("extend_px", float, 24.0,
              "Pixels beyond the body mesh's silhouette that take the nearest on-mesh confidence "
              "(hair, a skirt's flare); clipped to the view's alpha when wired", minimum=0.0),
        Param("occlusion_margin", float, 0.015,
              "A surface point further than this (metres) behind the body's depth from the "
              "photograph's camera is hidden from it", minimum=0.0, advanced=True),
        Param("anchor_tolerance_pct", float, 0.0,
              "Also treat every view within this percentage of the camera bounding-box diagonal of the "
              "anchor position as the photograph (weight 1). OFF by default, and for a reason: a helix "
              "starts AND ends on the anchor, and the last frame comes back from the denoiser as a "
              "repaint (measured 2026-09-19: 14.5 dB from the photograph, a different face), which "
              "at full weight outvotes the photograph one to one at its own pose. Only "
              "`anchor_frame_index` is the photograph", minimum=0.0, advanced=True),
        Param("copies", int, 6,
              "How many masked copies of the photograph's frame go to the training as supporting views at the "
              "anchor camera (votes AND supporting views for the evidence gate; 6 measured, 12 barely better). "
              "0 makes none. Needs `images`", minimum=0),
        Param("debug_dir", str, "", "Write each view's confidence and weight here, and stats.json"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cameras = list(inputs["cameras"])
        base = inputs.get("weights")
        alphas = inputs.get("alphas")
        mesh_world = inputs.get("mesh_world")
        strength = float(params["strength"])

        if base is not None and len(base) != len(cameras):
            raise ValueError(f"photo_priority_weights: {len(base)} weights for {len(cameras)} cameras")
        if alphas is not None and len(alphas) != len(cameras):
            raise ValueError(f"photo_priority_weights: {len(alphas)} alphas for {len(cameras)} cameras")

        if not cameras:
            return self._passthrough(cameras, base, why="no views")
        if mesh_world is None:
            return self._passthrough(cameras, base, why="no body mesh was wired in")
        if strength <= 0.0:
            return self._passthrough(cameras, base, why="strength is 0")

        width, height = _frame_size(cameras)
        mesh = (np.asarray(mesh_world[0], np.float64).reshape(-1, 3), np.asarray(mesh_world[1], np.int64).reshape(-1, 3))
        anchor_index, anchor_camera = _anchor(inputs, cameras)
        anchor_position = np.asarray(anchor_camera.position, np.float64).reshape(3)
        pivot = _pivot(inputs, mesh)
        anchor_dir = _unit(anchor_position - pivot)
        sources = set()
        tolerance = float(params["anchor_tolerance_pct"])
        if anchor_index is None:
            # Only a recorded position: the photograph's frame has to be found by pose (inject_anchor's own rule).
            tolerance = max(tolerance, 0.1)
        if tolerance > 0.0:
            sources.update(anchor_frames(cameras, anchor_position, tolerance))
        if anchor_index is not None:
            sources.add(int(anchor_index))
        # The photograph's camera may not be one of this batch's; render its depth once regardless.
        anchor_depth = raster_view(mesh, anchor_camera).depth

        debug = Path(params["debug_dir"]) if params["debug_dir"] else None
        if debug is not None:
            debug.mkdir(parents=True, exist_ok=True)
        weights: List[np.ndarray] = []
        per_view: List[Dict[str, Any]] = []
        yielded = 0
        for i, camera in enumerate(cameras):
            base_w = np.ones((height, width), np.float32) if base is None else normalize_mask(base[i])
            view_dir = _unit(np.asarray(camera.position, np.float64).reshape(3) - pivot)
            angle = float(np.degrees(np.arccos(np.clip(np.dot(view_dir, anchor_dir), -1.0, 1.0))))
            attenuation = 0.0 if i in sources else angular_attenuation(angle, float(params["cap_radius_deg"]), float(params["fade_deg"]))
            conf = np.zeros((height, width), np.float32)
            if attenuation > 0.0:
                raster = raster_view(mesh, camera)
                points, normals, hit = surface_points(raster, camera)
                if hit.any():
                    conf[hit] = photo_confidence(points, normals, anchor_camera, anchor_depth, margin=float(params["occlusion_margin"]),
                                                 facing_lo=float(params["facing_lo"]), facing_hi=float(params["facing_hi"]))
                conf = extend_off_mesh(conf, hit, float(params["extend_px"]), None if alphas is None else alphas[i])
            weight = base_w * yield_weight(conf, strength=strength, attenuation=attenuation, feather_px=float(params["feather_px"]))
            weights.append(weight.astype(np.float32))
            owned = conf > 0.5
            stat = {"view": i, "angle_deg": round(angle, 2), "attenuation": round(attenuation, 3), "source": i in sources,
                    "owned_px": int(owned.sum()), "mean_weight_owned": round(float(weight[owned].mean()), 3) if owned.any() else None}
            per_view.append(stat)
            if attenuation > 0.0 and owned.any():
                yielded += 1
            if debug is not None:
                cv2.imwrite(str(debug / f"conf_{i:03d}.png"), np.clip(conf * 255, 0, 255).astype(np.uint8))
                cv2.imwrite(str(debug / f"weight_{i:03d}.png"), np.clip(weight * 255, 0, 255).astype(np.uint8))

        support_images, support_masks, support_cameras = self._copies(
            inputs, params, anchor_index, anchor_camera, mesh, width, height,
            None if alphas is None or anchor_index is None else alphas[int(anchor_index)])

        stats = {"views": len(cameras), "yielded": yielded, "sources": sorted(sources), "strength": strength, "copies": len(support_images),
                 "cap_radius_deg": float(params["cap_radius_deg"]), "fade_deg": float(params["fade_deg"]), "per_view": per_view}
        if debug is not None:
            (debug / "stats.json").write_text(json.dumps(stats, indent=1))
        logger.info(
            "photo_priority_weights: %d/%d views yield to the photograph where it sees the body (within %.0f+%.0f deg of the "
            "anchor); %d view(s) are the photograph itself and keep weight 1; strength %.2f",
            yielded, len(cameras), float(params["cap_radius_deg"]), float(params["fade_deg"]), len(sources), strength)
        if yielded == 0:
            logger.warning("photo_priority_weights: no view yields — either every view is the photograph's or the body is not "
                           "where these cameras look")
        return {"weights": weights, "support_images": support_images, "support_masks": support_masks,
                "support_cameras": support_cameras, "photo_priority_stats": stats}

    @staticmethod
    def _copies(inputs, params, anchor_index, anchor_camera, mesh, width, height, alpha):
        """`copies` of the photograph's frame, masked by its own confidence field, at its camera."""
        copies = int(params["copies"])
        images = inputs.get("images")
        if copies <= 0:
            return [], [], []
        if images is None or anchor_index is None or not (0 <= int(anchor_index) < len(images)):
            logger.warning("photo_priority_weights: copies=%d but no `images` / anchor index to copy the photograph from; none made", copies)
            return [], [], []
        photo = np.asarray(images[int(anchor_index)])
        if photo.shape[:2] != (height, width):
            raise ValueError(f"photo_priority_weights: the photograph's frame is {photo.shape[1]}x{photo.shape[0]}, the cameras {width}x{height}")
        if alpha is None and photo.ndim == 3 and photo.shape[2] == 4:
            alpha = photo[..., 3]
        raster = raster_view(mesh, anchor_camera)
        points, normals, hit = surface_points(raster, anchor_camera)
        conf = np.zeros((height, width), np.float32)
        if hit.any():
            conf[hit] = photo_confidence(points, normals, anchor_camera, raster.depth, margin=float(params["occlusion_margin"]),
                                         facing_lo=float(params["facing_lo"]), facing_hi=float(params["facing_hi"]))
        conf = extend_off_mesh(conf, hit, float(params["extend_px"]), alpha)
        if float(params["feather_px"]) > 0:
            conf = cv2.GaussianBlur(conf, (0, 0), float(params["feather_px"]))
        if alpha is not None:
            conf = conf * (normalize_mask(alpha) > 0.5)
        bgr = photo[..., :3] if photo.ndim == 3 else np.repeat(photo[..., None], 3, 2)
        logger.info("photo_priority_weights: %d masked copies of the photograph's frame as supporting views (mask mean over the "
                    "subject %.2f)", copies, float(conf[normalize_mask(alpha) > 0.5].mean()) if alpha is not None and (normalize_mask(alpha) > 0.5).any() else float(conf.mean()))
        return [bgr.copy() for _ in range(copies)], [conf.astype(np.float32) for _ in range(copies)], [anchor_camera] * copies

    @staticmethod
    def _passthrough(cameras, base, *, why: str) -> Dict[str, Any]:
        logger.info("photo_priority_weights: %s; the weights pass through", why)
        if base is not None:
            weights = [normalize_mask(w).astype(np.float32) for w in base]
        elif cameras:
            width, height = _frame_size(cameras)
            weights = [np.ones((height, width), np.float32) for _ in cameras]
        else:
            weights = []
        return {"weights": weights, "support_images": [], "support_masks": [], "support_cameras": [],
                "photo_priority_stats": {"views": len(cameras), "yielded": 0, "passthrough": why}}


def _frame_size(cameras) -> Tuple[int, int]:
    sizes = {(int(c.width), int(c.height)) for c in cameras}
    if len(sizes) != 1:
        raise ValueError(f"photo_priority_weights: the cameras disagree on the frame size ({sorted(sizes)}); one batch renders at one size.")
    (width, height), = sizes
    return width, height


def _anchor(inputs: Dict[str, Any], cameras: Sequence[Any]) -> Tuple[Optional[int], Any]:
    """(the photograph's index in `cameras` if known, its camera): live from `anchor_cameras`,
    else the recorded position with the batch's first camera's intrinsics and orientation."""
    anchor_cameras = inputs.get("anchor_cameras")
    index = inputs.get("anchor_frame_index")
    if anchor_cameras and index is not None and 0 <= int(index) < len(anchor_cameras):
        camera = anchor_cameras[int(index)]
        in_batch = int(index) if anchor_cameras is cameras or (len(anchor_cameras) == len(cameras) and cameras[int(index)] is camera) else None
        return in_batch, camera
    position = inputs.get("anchor_position")
    if position is not None:
        from body2colmap.camera import Camera

        first = cameras[0]
        camera = Camera(focal_length=(first.fx, first.fy), image_size=(first.width, first.height),
                        principal_point=(first.cx, first.cy), position=np.asarray(position, np.float32),
                        rotation=np.asarray(first.rotation, np.float32))
        logger.warning("photo_priority_weights: the anchor camera is a recorded position with the first view's orientation; "
                       "wire anchor_cameras + anchor_frame_index for the photograph's real pose")
        return None, camera
    raise ValueError("photo_priority_weights: wire 'anchor_cameras' + 'anchor_frame_index' (the camera list and the "
                     "photograph's index in it) or 'anchor_position'. The yield is measured from the photograph's view of "
                     "the body, and there is no view to measure from.")


def _pivot(inputs: Dict[str, Any], mesh: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    for key in ("splat_center", "orbit_target"):
        value = inputs.get(key)
        if value is not None:
            return np.asarray(value, np.float64).reshape(3)
    vertices = mesh[0]
    return (vertices.min(0) + vertices.max(0)) / 2.0


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("photo_priority_weights: a camera sits on the pivot, so it has no view direction.")
    return vector / norm
