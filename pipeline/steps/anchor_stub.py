"""Anchor-frame warp/injection, and the anchor *band*.

Ported from nodes/generate_firstlast_node.py and nodes/inject_anchor_node.py
in the original ComfyUI-Body2COLMAP repo:

- GenerateFirstLast homography-warps the original reference photo to align
  pixel-for-pixel with the rendered skeleton at the orbit's anchor camera
  (accounts for focal-length auto-framing and the look_at rotation
  correction; see body2colmap.utils.compute_warp_to_camera). The warp is
  reusable across a circular render and a later helical re-render that
  shares the same anchor camera — compute once, inject into both.
- InjectAnchor then overwrites every frame whose camera position matches
  the anchor (matched by position, not frame index, since more than one
  frame can land there — e.g. a circular path's default overlap=1
  duplicates the first camera as the last) with that warped image, so the
  diffusion pass gets one real conditioning frame instead of an all-render
  batch.

- CompositeSplatViews is the fourth, and the only one that does not
  replace a frame: it alpha-composites a splat render ON TOP of the mesh
  drawing, keeping the drawing everywhere the splat is transparent. That is
  what puts a real face on a skeleton rig, and it is why it survives much
  further off the source view than a substitution does — a face-only splat
  read cleanly out to about 30 degrees and still holds at 45, against the
  15 the body shell's substitution band is set to. See its docstring for
  the black-background requirement, which is load-bearing.

- SelectSupportViews is the fifth, and does not touch a frame at all: it
  picks the frames of a splat render that a composite kept, un-premultiplies
  them, and hands them to `brush` as *supporting views* — training evidence
  that counts only where its mask says to. It is the face splat's second
  route into the training, the first being composited into the drawings
  before the diffusion passes get their hands on it. See steps/brush.py's
  `support_*` inputs.

The alpha convention is what the first three write. This is the mechanism behind
cyber_6f's initial/ alpha convention (pipeline/steps/wan22_vace_denoise.py's
docstring): the injected anchor frame is marked alpha=0 ("already real,
don't denoise"), every synthetic render frame is alpha=255 ("needs
denoising"). Wiring these steps in is what would produce that mask
correctly for a dataset built from scratch, rather than relying on
cyber_6f already having it baked in.

All three take render.py's output directly: GenerateFirstLastStep's
`camera`/`original_focal_length`/`render_size` inputs are exactly
render.py's `image_warp` dict's fields (override_cam_from_mesh=True run);
InjectAnchorStep's `cameras`/`anchor_position` are render.py's own
`cameras`/`anchor_position` outputs.

VERIFICATION STATUS differs between the steps. `inject_anchor` is
verified against recorded output: `cyber_6f/initial` records an
anchor_position at the world origin, and frames 1 and 81 of that dataset
are byte-identical to its anchor.png — so the recorded data independently
says which frames the ComfyUI flow injected into, and this step finds the
same two from camera positions alone, including after a rotate_views
reordering. `generate_firstlast`'s warp is still verified on synthetic
data only, and cannot be checked against cyber_6f: the image it warps is
the front view SAM-3D-Body ran on, which that dataset does not keep — its
reference.png is the whole two-panel front/back sheet, framed differently.
(Current runs split that sheet up front and keep only the back half as
reference.png; the front half lives at `scene.front_image` for this step
and is likewise not persisted. See steps/reference_sheet.py.) It needs a
real render.py `image_warp` output, so it waits on a pod. See
tests/test_anchor.py.

`InjectShellViews` used to sit here too — the same idea one step wider,
substituting photoreal renders off a photo-derived body shell across the
band of frames near the source view. It went with the rest of the
photo-to-splat work on 2026-08-30 and lives on the
`pointmap-splat-integration` branch.

Unlike the original ComfyUI nodes, this port works in cv2 BGR uint8
throughout (no RGB<->BGR/float<->uint8 tensor conversion needed — that
was purely for ComfyUI's IMAGE tensor convention) and masks are float32
[0,1] with foreground=1, matching rmbg.py/brush.py's convention rather
than ComfyUI's inverted MASK (1.0=background).

**At every call site of this step, the mask is the VACE mask.**
`dataset.masks` is overloaded across the pipeline as a whole — rmbg puts a
foreground silhouette there for brush, render_splat puts the splat's alpha
there for mask_splat — but this step only ever runs immediately before a
diffusion pass, where the field is the frame's alpha on disk and
`wan22_vace_denoise`'s `control_masks`: 1.0 "synthetic, denoise this", 0.0
"a real photograph, keep it". So the value written at an injected frame is
0.0, in every caller, because an injected frame is by definition the real
photo. (The ComfyUI node writes 0.0 too, but arrives
there backwards: its MASK is inverted and `SaveDataset` re-inverts on the
way to disk, `alpha = (1 - mask) * 255`. Two inversions, same number, and
reasoning from the graph alone gets the opposite answer — check the
recorded alpha instead: `cyber2_6f/masked_splatted/frame_00038_.png` is
byte-identical to that stage's `anchor.png` and carries a uniform alpha of
0, while all 80 other frames are 255.)

**This step must run after `mask_splat`, not before it.** That ordering is
load-bearing and was wrong, at the cost of a whole run's output quality.
`Body2COLMAP_InjectAnchor` takes `masks` as an *input* and clones it; this
port took no masks at all and manufactured an all-1.0 batch on every call.
Placed between `render_splat` and `mask_splat` — where the mask field is
carrying the splat render's per-*pixel* alpha, the thing `mask_splat`
exists to threshold — that wiped the alpha out. `mask_splat`'s keep-test
then passed everywhere: ~78% of each frame that the reference flow blacks
out survived into `denoise_pass2`, soft Gaussian fringes and all, and pass
2 hallucinated around them differently per frame. That view disagreement is
the ghosting in the final splat.

Ordering it after `mask_splat` also matches the recorded run frame for
frame, and dissolves the conflict rather than managing it: `mask_splat`
consumes the spatial alpha and emits the all-1.0 VACE mask, and this step
only ever sees and writes VACE masks. It reproduces
`cyber2_6f/masked_splatted` exactly — every frame masked and filtered at
alpha 255, except the anchor frame, which is `anchor.png` verbatim (not
composited, not filtered) at alpha 0.

`masks` stays an optional input regardless. When it is supplied it is
passed through untouched; a step handed somebody else's data should not
destroy it, which is the general form of the bug above.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


@register_step("generate_firstlast")
class GenerateFirstLastStep(Step):
    """Warp a reference image to align with the rendered skeleton at the
    anchor frame.

    inputs: {"image": np.ndarray BGR uint8 (reference photo),
             "camera": Camera, "original_focal_length": float,
             "render_size": Tuple[int, int],
             "bg_color": Optional[Tuple[float, float, float]] RGB [0,1],
             defaults to white}
    outputs: {"warped_image": np.ndarray BGR uint8}

    Takes no params: everything it needs (including the border colour) comes
    from the render step upstream, via `image_warp`.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.utils import compute_warp_to_camera

        camera = inputs["camera"]
        original_focal_length = inputs["original_focal_length"]
        render_w, render_h = inputs["render_size"]
        img = inputs["image"]
        h_img, w_img = img.shape[:2]

        bg_rgb = inputs.get("bg_color", (1.0, 1.0, 1.0))
        # BGR order to match img's channel order (bg_rgb is RGB, cv2 border
        # color must be BGR for this image).
        border_color = (
            int(round(bg_rgb[2] * 255)),
            int(round(bg_rgb[1] * 255)),
            int(round(bg_rgb[0] * 255)),
        )

        is_identity_rotation = np.allclose(camera.rotation, np.eye(3), atol=1e-5)

        if is_identity_rotation:
            # Pure affine: scale + translate (fast path)
            s = float(camera.fx / original_focal_length)
            tx = camera.cx - s * w_img / 2.0
            ty = camera.cy - s * h_img / 2.0
            M = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float64)
            warped = cv2.warpAffine(
                img, M, (render_w, render_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )
        else:
            # Full homography: accounts for rotation + intrinsic change
            H = compute_warp_to_camera(
                original_focal_length=original_focal_length,
                original_image_size=(w_img, h_img),
                target_camera=camera,
            )
            warped = cv2.warpPerspective(
                img, H, (render_w, render_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )

        return {"warped_image": warped}


@register_step("inject_anchor")
class InjectAnchorStep(Step):
    """Overwrite every frame sitting at the anchor camera position with a
    given image (typically GenerateFirstLastStep's output).

    inputs: {"images": List[np.ndarray], "cameras": List[Camera],
             "anchor_position": Optional[np.ndarray],
             "anchor_image": Optional[np.ndarray],
             "masks": Optional[List[np.ndarray]] — the VACE mask batch to
             inject into, passed through untouched except at the matched
             frames. Omit it and an all-1.0 batch is manufactured instead
             ("everything is synthetic, denoise it"), which is what the
             pre-denoise callers want.}
    outputs: {"images": List[np.ndarray], "masks": List[np.ndarray]}
             (masks: float32 [0,1] VACE masks — the supplied batch where
             there was one, all-1.0 otherwise, and 0.0 at the injected
             anchor frame(s): a real photograph, do not denoise it)

    With no anchor_image/anchor_position (a dataset with no anchor frame,
    or generate_firstlast simply not wired in), the inputs pass through
    rather than failing the workflow — masks included, untouched.
    """

    PARAMS = (
        Param("tolerance_pct", float, 0.1,
              "How close a camera has to be to the anchor position to count as "
              "sitting on it, as a percentage of the camera bounding-box diagonal"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = inputs["images"]
        cameras = inputs["cameras"]
        anchor_position = inputs.get("anchor_position")
        anchor_image = inputs.get("anchor_image")
        in_masks = inputs.get("masks")

        # Supplied masks are somebody else's data: pass them through rather
        # than manufacturing a batch over the top of them. With none
        # supplied, all-1.0 is the right VACE mask for a batch of renders.
        if in_masks is None:
            masks = [np.ones(img.shape[:2], dtype=np.float32) for img in images]
        else:
            masks = [np.asarray(m, dtype=np.float32) for m in in_masks]

        if anchor_image is None or anchor_position is None:
            return {"images": images, "masks": masks}

        anchor_position = np.asarray(anchor_position, dtype=np.float32)
        positions = np.stack([cam.position for cam in cameras], axis=0)
        scale = _scene_scale(positions)
        tolerance_pct = params["tolerance_pct"]
        threshold = (tolerance_pct / 100.0) * scale

        distances = np.linalg.norm(positions - anchor_position, axis=1)
        matches = [int(i) for i in np.flatnonzero(distances <= threshold)]

        if not matches:
            return {"images": images, "masks": masks}

        if tuple(anchor_image.shape) != tuple(images[0].shape):
            raise ValueError(
                f"anchor_image shape {tuple(anchor_image.shape)} does not match the "
                f"image batch frame shape {tuple(images[0].shape)}. Render the anchor "
                f"at the same resolution as the orbit (GenerateFirstLastStep uses the "
                f"render_size from render.py's image_warp output)."
            )

        out_images = list(images)
        for idx in matches:
            out_images[idx] = anchor_image
            # 0.0 = "a real photograph, keep it" in VACE's control-mask
            # sense. Uniform over the frame: this is a per-frame flag, not a
            # silhouette. See the module docstring.
            masks[idx] = np.zeros(anchor_image.shape[:2], dtype=np.float32)

        return {"images": out_images, "masks": masks}


def _scene_scale(positions: np.ndarray) -> float:
    """Characteristic scale = diagonal of bounding box of camera positions."""
    if len(positions) < 2:
        return 1.0
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    return float(diag) if diag > 0 else 1.0


@register_step("composite_splat_views")
class CompositeSplatViewsStep(Step):
    """Alpha-composite a splat render over every frame close enough to see it.

    inputs: {"images": List[np.ndarray] — the mesh render, one per frame,
             "splat_images": List[np.ndarray] — `render_splat` over the SAME
             cameras, in the same order (`pattern: ""` is what gives that),
             "splat_masks": List[np.ndarray] float32 [0,1] — that render's
             per-pixel alpha, which `render_splat` publishes as its `masks`,
             "cameras": List[Camera],
             "splat_center": Optional[Sequence[float]] — the splat's own
             centre in world coordinates, which is the pivot the view angle
             is measured about. A splat builder publishes it as
             `splat_stats.world_center`. Falls back to `orbit_target`,
             "orbit_target": Optional[np.ndarray] (3,),
             "anchor_position": Optional[np.ndarray] (3,) — where the photo
             was taken from. Omit it for the world origin, which is where it
             is: `sam3d_to_world` does not recentre, so the SAM-3D-Body
             camera IS the world origin,
             "masks": Optional[List[np.ndarray]] — passed through with the
             splat's coverage unioned in,
             "splat_cameras": Optional[List[Camera]] — wire it and the two
             batches are checked to be the same views rather than assumed}
    outputs: {"images": List[np.ndarray],
              "masks": List[np.ndarray] — only when `masks` was wired in,
              "view_roles": List[dict] — per frame: index, angle, role}

    This is the b2crunner side of body2colmap's `skeleton+splat` composite
    mode (`c65a7f7`). The geometry, the compositing rule and the 45-degree
    default are all that commit's; what differs is where the layer comes
    from. body2colmap rasterises it with gsplat inside
    `OrbitPipeline.render_splat_layer`, and this project deliberately has no
    gsplat — `render_splat` shells out to `brush-splat-render` instead, which
    is what removed its last runtime CUDA toolchain (see pyproject.toml).
    So the layer is rendered by a separate step and composited here.
    **See docs/revert-when-body2colmap-drops-gsplat.md**: when body2colmap
    moves to the brush renderer, this step and its wiring come out in favour
    of the render mode.

    **The splat render must be on a BLACK background.** `brush-splat-render`
    writes `rgb = colour*alpha + bg*(1-alpha)`, so with `bg = 0` its output
    is premultiplied by alpha and the composite is exactly

        out = base * (1 - alpha) + splat_rgb

    On any other background the same expression blends toward that
    background a second time and the splat comes back washed out — the
    identical trap body2colmap's commit fixed by adding straight-alpha
    output (`bg_color=None`) to its own renderer. `render_splat`'s default
    `bg_color` is already black; this step re-checks what it can (an opaque
    pixel whose colour is far from the base's cannot prove much, but a
    fully transparent pixel that is not black is proof of the bug) and
    refuses rather than quietly producing grey ghosts.

    **Why the cull.** The splat is a 2.5-D shell of the side of the subject
    one photograph saw. Turn the camera away from that view and the open rim
    swings into frame as a flare of grazing-incidence Gaussians. body2colmap
    measured the boundary on a Face_Neck head splat: reads cleanly to about
    30 degrees, the rim starts flaring by 45, and by 60 the shell is mostly
    edge. 45 is that measurement, not a guess.

    **The pivot is the splat's centre, not the orbit target.** A head sits
    well above a full-body orbit target, so a camera 30 degrees up from the
    equator views the *body* at 30 degrees and the *head* at rather less.
    body2colmap's `splat_view_angle_deg` measures from the splat's own bbox
    centre for exactly this reason, and this follows it.
    """

    PARAMS = (
        Param("max_angle_deg", float, 45.0,
              "Composite the splat on every frame whose view of it is within "
              "this angle of the photograph's. Past it the layer is dropped "
              "entirely rather than faded: what appears out there is the "
              "shell's open rim, and a half-transparent rim is still a rim. "
              "0 disables the compositing", minimum=0.0, maximum=180.0),
        Param("min_alpha", float, 0.004,
              "Treat alpha below this as fully transparent. Splat renders "
              "have a long tail of near-zero alpha that would otherwise tint "
              "the whole frame by a level or two", minimum=0.0, maximum=1.0,
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = list(inputs["images"])
        splat_images: List[np.ndarray] = list(inputs["splat_images"])
        splat_masks: List[np.ndarray] = list(inputs["splat_masks"])
        cameras = inputs["cameras"]
        max_angle = params["max_angle_deg"]

        if not (len(splat_images) == len(splat_masks) == len(images)):
            raise ValueError(
                f"composite_splat_views: {len(splat_images)} splat renders and "
                f"{len(splat_masks)} alphas against {len(images)} frames. The "
                f"batches are matched by index, which holds because "
                f"render_splat with `pattern: \"\"` reuses the source dataset's "
                f"cameras verbatim and in order — a pattern on that step "
                f"breaks this."
            )

        # The same cheap proof of index alignment inject_anchor makes.
        splat_cameras = inputs.get("splat_cameras")
        if splat_cameras is not None and cameras:
            drift = max(
                float(np.linalg.norm(np.asarray(a.position, dtype=np.float64)
                                     - np.asarray(b.position, dtype=np.float64)))
                for a, b in zip(cameras, splat_cameras)
            )
            if drift > 1e-4:
                raise ValueError(
                    f"composite_splat_views: the splat render's cameras are "
                    f"not the frame batch's (worst position drift "
                    f"{drift:.6f}). Render it with `pattern: \"\"` and the "
                    f"dataset wired, so it reuses these cameras verbatim."
                )

        masks = inputs.get("masks")
        masks = None if masks is None else [np.asarray(m).copy() for m in masks]

        if max_angle <= 0.0:
            logger.info("composite_splat_views: max_angle_deg is 0, nothing "
                        "composited")
            return _composite_result(images, masks, [
                {"index": i, "angle_from_anchor_deg": None, "role": "base"}
                for i in range(len(images))
            ])

        pivot = _composite_pivot(inputs)
        supplied = inputs.get("anchor_position")
        anchor = (np.zeros(3) if supplied is None
                  else np.asarray(supplied, dtype=np.float64).reshape(3))
        source_dir = pivot - anchor
        if float(np.linalg.norm(source_dir)) < 1e-9:
            raise ValueError(
                "composite_splat_views: the camera the photo was taken from "
                "sits on the splat's centre, so there is no source view "
                "direction to measure against."
            )

        view_roles = []
        for index, camera in enumerate(cameras):
            direction = pivot - np.asarray(camera.position, dtype=np.float64)
            angle = _angle_between_deg(direction, source_dir)
            if angle > max_angle:
                view_roles.append({"index": index,
                                   "angle_from_anchor_deg": round(angle, 3),
                                   "role": "base"})
                continue

            layer = splat_images[index]
            alpha = np.asarray(splat_masks[index], dtype=np.float32)
            if tuple(layer.shape) != tuple(images[index].shape):
                raise ValueError(
                    f"composite_splat_views: splat render {index} has shape "
                    f"{tuple(layer.shape)} against the frame's "
                    f"{tuple(images[index].shape)}. Render the splat at the "
                    f"same resolution as the orbit."
                )
            _check_premultiplied(layer, alpha, index, params["min_alpha"])

            alpha = np.where(alpha < params["min_alpha"], 0.0, alpha)[..., None]
            images[index] = np.clip(
                images[index].astype(np.float32) * (1.0 - alpha)
                + layer.astype(np.float32),
                0, 255,
            ).astype(np.uint8)
            if masks is not None:
                # The splat is real subject coverage, exactly as the mesh
                # silhouette is, so a mask derived from the result has to
                # include it — body2colmap's `_composite_splat` unions alpha
                # for the same reason, and excludes the skeleton, which is
                # an annotation rather than geometry.
                masks[index] = np.maximum(masks[index], alpha[..., 0])

            view_roles.append({"index": index,
                               "angle_from_anchor_deg": round(angle, 3),
                               "role": "composited"})

        composited = sum(1 for role in view_roles if role["role"] == "composited")
        logger.info(
            "composite_splat_views: %d/%d frames within %.1f deg of the source "
            "view take the splat layer",
            composited, len(images), max_angle,
        )
        if composited == 0:
            logger.warning(
                "composite_splat_views: no frame is within %.1f deg of the "
                "source view, so the splat was rendered and then discarded. "
                "Either anchor_position and the cameras did not come from the "
                "same render, or the radius is smaller than the path's "
                "angular step.", max_angle,
            )
        return _composite_result(images, masks, view_roles)


@register_step("select_support_views")
class SelectSupportViewsStep(Step):
    """Turn a splat render into supporting views for a brush training.

    inputs: {"images": List[np.ndarray] — a `render_splat` batch on a BLACK
             background, i.e. premultiplied colour,
             "masks": List[np.ndarray] float32 [0,1] — that render's alpha,
             "cameras": List[Camera] — the cameras it was rendered from,
             "view_roles": Optional[List[dict]] — `composite_splat_views`'
             own per-frame verdict; frames whose role is not `role` are
             dropped}
    outputs: {"images": List[np.ndarray], "masks": List[np.ndarray],
              "cameras": List[Camera]}

    The other half of `brush`'s `support_*` inputs (see steps/brush.py):
    views the training fits where their mask says to and ignores
    everywhere else. The face splat is the case this exists for. Its
    renders carry the subject's actual face, and by the time the second
    denoise pass and the upscale have been over the batch, the frames
    brush trains on carry a diffusion model's idea of it — so the render
    goes into the training a second time, as evidence, weighted by the
    splat's own coverage.

    **Masked, not transparent.** Outside the face the render is black
    background, and that is not a statement that nothing is there: the
    body, the hair and the rest of the frame are all outside a Face_Neck
    matte. Training on it as transparent would ask the model to carve away
    the whole subject in the name of one small render, which is what the
    `masks/` sidecar in steps/brush.py prevents.

    **Which frames.** The same ones `composite_splat_views` kept. Past its
    cull angle what a 2.5-D shell shows is its own open rim, and a rim is
    no more supervision than it is a face — so rather than re-deriving the
    angle here, this reads that step's `view_roles` and keeps the frames it
    called `composited`. Wire no `view_roles` and every frame is kept,
    which is right for a splat that is not a shell.

    **Un-premultiplying** is what makes the render straight-alpha, which is
    what brush's masked mode expects: it does not premultiply masked ground
    truth, so a `colour*a` frame would ask the model to be dark and
    semi-transparent along the silhouette instead of opaque and the right
    colour. Only the soft rim differs — inside the matte alpha is 1 and the
    two are identical — but the rim is exactly where a face splat's
    silhouette is decided.
    """

    PARAMS = (
        Param("role", str, "composited",
              "Keep the frames `composite_splat_views` gave this role, when a "
              "`view_roles` input is wired in. `composited` is the band within its "
              "cull angle of the source view — the frames the splat can speak for. "
              "Empty keeps every frame",
              choices=("composited", "base", "")),
        Param("unpremultiply", bool, True,
              "Divide the colour back out by alpha, turning a render made on black "
              "into the straight-alpha frame brush's masked mode expects. Off leaves "
              "the render as it came, which darkens the soft silhouette",
              advanced=True),
        Param("min_alpha", float, 0.004,
              "Alpha at or below this is treated as fully transparent: the colour "
              "there is not recoverable by dividing and the mask weights it at zero "
              "anyway. Same default as composite_splat_views' own",
              minimum=0.0, maximum=1.0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = list(inputs["images"])
        masks: List[np.ndarray] = list(inputs["masks"])
        cameras = list(inputs["cameras"])
        view_roles = inputs.get("view_roles")
        role = params["role"]
        min_alpha = params["min_alpha"]

        if not (len(images) == len(masks) == len(cameras)):
            raise ValueError(
                f"select_support_views: {len(images)} frames, {len(masks)} alphas "
                f"and {len(cameras)} cameras. The three describe the same views and "
                f"have to arrive together."
            )

        keep = list(range(len(images)))
        if view_roles is not None and role:
            if len(view_roles) != len(images):
                raise ValueError(
                    f"select_support_views: {len(view_roles)} view roles against "
                    f"{len(images)} frames. Wire the `view_roles` of the "
                    f"composite_splat_views that took THIS render."
                )
            keep = [i for i, entry in enumerate(view_roles) if entry.get("role") == role]

        out_images, out_masks, out_cameras = [], [], []
        for index in keep:
            alpha = np.asarray(masks[index], dtype=np.float32)
            layer = images[index]
            # The same requirement composite_splat_views has, for a
            # different reason: dividing by alpha only recovers the
            # straight colour if the render was premultiplied over black.
            _check_premultiplied(
                layer, alpha, index, min_alpha, where="select_support_views",
                because="Un-premultiplying (rgb / a) only recovers the straight "
                        "colour these views are supposed to carry for",
            )
            out_images.append(
                _unpremultiply(layer, alpha, min_alpha)
                if params["unpremultiply"] else layer
            )
            out_masks.append(np.where(alpha < min_alpha, 0.0, alpha))
            out_cameras.append(cameras[index])

        logger.info(
            "select_support_views: %d/%d frames kept%s as supporting views",
            len(keep), len(images),
            f" (role={role})" if view_roles is not None and role else "",
        )
        if not keep:
            # Not an error: brush takes no supporting views and trains
            # exactly as it did before. Worth saying out loud, because the
            # run then silently loses the thing this wiring exists for.
            logger.warning(
                "select_support_views: no frame has role %r, so the training will "
                "get no supporting views. Either the render and the roles came from "
                "different steps, or nothing was within the composite's cull angle.",
                role,
            )
        return {"images": out_images, "masks": out_masks, "cameras": out_cameras}


def _unpremultiply(layer: np.ndarray, alpha: np.ndarray, min_alpha: float) -> np.ndarray:
    """`colour*a` back to `colour`, black where there is no colour to recover.

    Below `min_alpha` the division is both unstable and meaningless — a
    value of 1/255 divided by an alpha of 0.002 is noise amplified 500x —
    and the mask hands those pixels a weight of zero regardless, so they
    are left at black rather than reconstructed.
    """
    rgb = layer[..., :3].astype(np.float32)
    safe = alpha >= min_alpha
    divisor = np.where(safe, alpha, 1.0)[..., None]
    straight = np.clip(rgb / divisor, 0, 255)
    return np.where(safe[..., None], straight, 0.0).astype(np.uint8)


def _composite_result(images, masks, view_roles) -> Dict[str, Any]:
    """Only publish `masks` when one was wired in — a step that returns an
    output a workflow did not ask for is harmless, but one that invents an
    empty mask batch downstream steps then threshold is not."""
    result: Dict[str, Any] = {"images": images, "view_roles": view_roles}
    if masks is not None:
        result["masks"] = masks
    return result


def _composite_pivot(inputs: Dict[str, Any]) -> np.ndarray:
    """Where to measure the view angle about: the splat's centre if known.

    See CompositeSplatViewsStep's docstring — for a head on a full-body
    orbit the splat's centre and the orbit target are not interchangeable.
    """
    center = inputs.get("splat_center")
    if center is None:
        center = inputs.get("orbit_target")
    if center is None:
        raise ValueError(
            "composite_splat_views: wire either 'splat_center' (the splat's "
            "own world centre — a splat builder publishes it as "
            "splat_stats.world_center) or 'orbit_target'. The view angle has "
            "to be measured about something."
        )
    return np.asarray(center, dtype=np.float64).reshape(3)


def _check_premultiplied(layer: np.ndarray, alpha: np.ndarray, index: int,
                         min_alpha: float, where: str = "composite_splat_views",
                         because: str = "This step composites premultiplied "
                                        "colour (out = base*(1-a) + rgb), which only "
                                        "holds for") -> None:
    """Refuse a splat render that was not made on a black background.

    Only one direction of this is provable from the images alone: where
    alpha is zero the render must be black, because `colour*0 + bg*1` is
    the background and nothing else. A non-black value there is proof the
    caller passed a `bg_color`, and the composite that follows would blend
    toward it twice.

    `where`/`because` only name the caller in the message: both steps that
    read a splat render depend on it being premultiplied over black, for
    reasons that differ in the second sentence and not in the check.
    """
    transparent = alpha < min_alpha
    if not transparent.any():
        return
    worst = int(layer[transparent].max())
    if worst > 8:
        raise ValueError(
            f"{where}: splat render {index} is {worst}/255 "
            f"bright where it is fully transparent, so it was rendered on a "
            f"non-black background. {because} "
            f"bg_color [0, 0, 0] — anything else blends toward the "
            f"background twice. Set `bg_color: [0.0, 0.0, 0.0]` on the "
            f"render_splat that feeds this step."
        )
