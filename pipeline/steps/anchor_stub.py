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

- CompositeSplatViews used to be the third: it alpha-composited a splat
  render ON TOP of the mesh drawing, which is what puts a real face on a
  skeleton rig. It is gone, and its job is body2colmap's again. That step
  only ever existed because body2colmap's own `skeleton+splat` mode
  rasterised through gsplat, which this project will not carry; now that
  the library rasterises through `brush-splat-render` too, the mesh
  `render` step draws the composite itself — a `...+splat` render_mode, a
  `splat_path` input and `splat_max_angle_deg`, blended by
  `Renderer._composite_splat`. See docs/revert-when-body2colmap-drops-gsplat.md
  for what came out and steps/render.py for what replaced it.

- SelectSupportViews is the third, and does not touch a frame at all: it
  picks the frames of a splat render taken as a `cap` around the source
  view, un-premultiplies them, and hands them to `brush` as *supporting
  views* — training evidence that counts only where its mask says to. It is
  the face splat's second route into the training, the first being
  composited into the drawings by `render` before the diffusion passes get
  their hands on it. See steps/brush.py's `support_*` inputs.

- MergeSupportViews is the fourth, and exists because there are now two
  producers of that kind of evidence and `brush` takes one set: the face
  cap above, and the stage-1 body shells
  (steps/elevation_views.py's `pointmap_elevation_views`, a shell per Nth
  denoised frame rendered from +/- an elevation). It concatenates two
  optional triples, runs ungated so it is always the writer of
  `scene.support_views.*`, and publishes three empty lists when both
  branches are off.

The alpha convention is what the first two write. This is the mechanism behind
cyber_6f's initial/ alpha convention (pipeline/steps/wan22_vace_denoise.py's
docstring): the injected anchor frame is marked alpha=0 ("already real,
don't denoise"), every synthetic render frame is alpha=255 ("needs
denoising"). Wiring these steps in is what would produce that mask
correctly for a dataset built from scratch, rather than relying on
cyber_6f already having it baked in.

Both take render.py's output directly: GenerateFirstLastStep's
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

**The path the batch was rendered along has to touch the anchor.** Matching
on position is only durable against reordering, not against a re-render:
where the frames came from a NEW camera path, that path must have been
anchored (`render_splat`'s `override_cam_from_mesh`), and the
`anchor_position` handed here must be the one that render published for it.
fast_helical_native shipped with neither until 2026-09-04 — an unanchored
helical re-render, matched against an anchor `refine_cameras` had since
moved — and the two failures compound: the nearest camera was 24x the
tolerance away, so the stage-3 injection matched nothing, returned the
batch untouched, and `denoise_pass2` conditioned on a batch with no real
photograph in it and no 0.0 in its VACE mask. Nothing failed; the run just
quietly lost its one piece of ground truth. That is why the no-match case
logs a WARNING naming the distance, and why tests/test_workflows.py refuses
the wiring outright.
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
            logger.info(
                "inject_anchor: nothing to inject (anchor_image=%s, "
                "anchor_position=%s); passing %d frames through unchanged",
                "set" if anchor_image is not None else "unset",
                "set" if anchor_position is not None else "unset",
                len(images),
            )
            return {"images": images, "masks": masks}

        anchor_position = np.asarray(anchor_position, dtype=np.float32)
        positions = np.stack([cam.position for cam in cameras], axis=0)
        scale = _scene_scale(positions)
        tolerance_pct = params["tolerance_pct"]
        threshold = (tolerance_pct / 100.0) * scale

        distances = np.linalg.norm(positions - anchor_position, axis=1)
        matches = [int(i) for i in np.flatnonzero(distances <= threshold)]

        if not matches:
            # Loud, because this is the failure that looks like success: the
            # batch comes out intact, the run finishes, and the diffusion
            # pass downstream simply never sees a real photograph. It means
            # the path this batch was rendered along does not pass through
            # the anchor at all — the usual cause is a re-render that built
            # a fresh orbit without anchoring it (render_splat's
            # `override_cam_from_mesh`), or an `anchor_position` that has
            # been republished since the path was built.
            logger.warning(
                "inject_anchor: NO frame matched the anchor within %.3f%% "
                "(threshold=%.6f, scene scale=%.6f, closest=%.6f at frame %d "
                "of %d). Returning the batch unchanged — nothing in it is "
                "marked as a real photograph.",
                tolerance_pct, threshold, scale, float(distances.min()),
                int(distances.argmin()), len(distances),
            )
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

        logger.info(
            "inject_anchor: injected the anchor into %d/%d frames %s "
            "(tolerance=%.3f%%, threshold=%.6f)",
            len(matches), len(images),
            ", ".join(f"{i} (d={distances[i]:.8f})" for i in matches),
            tolerance_pct, threshold,
        )
        return {"images": out_images, "masks": masks}


def _scene_scale(positions: np.ndarray) -> float:
    """Characteristic scale = diagonal of bounding box of camera positions."""
    if len(positions) < 2:
        return 1.0
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    return float(diag) if diag > 0 else 1.0


@register_step("select_support_views")
class SelectSupportViewsStep(Step):
    """Turn a splat render into supporting views for a brush training.

    inputs: {"images": List[np.ndarray] — a `render_splat` batch on a BLACK
             background, i.e. premultiplied colour,
             "masks": List[np.ndarray] float32 [0,1] — that render's alpha,
             "cameras": List[Camera] — the cameras it was rendered from,
             "path_cameras": Optional[List[Camera]] — the cameras of the
             batch this training is fitted to, i.e. the denoising path.
             Any view within `min_path_angle_deg` of it is dropped,
             "splat_center": Optional[Sequence[float]] — the splat's own
             world centre, the pivot that distance is measured about.
             Falls back to `orbit_target`. Required with `path_cameras`,
             "orbit_target": Optional[np.ndarray] (3,)}
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

    **Which frames.** A band, with an edge at each end, and the two are
    measured against different things because they are avoiding different
    things.

    **Which render.** Its own: `render_splat` with `pattern: cap`, a disc
    of views sampled around the photograph's own view of the splat
    (steps/splat.py's `cap_directions`). NOT the mesh `render`'s own
    cameras — every one of those is a view the training already has a
    denoised frame for, and the `+splat` overlay is composited onto those
    drawings there rather than handed here. The cap's radius is the outer
    edge of the band, and it belongs there rather than here: culling views
    after rendering them keeps the ones nearest the photograph, which are
    the ones worth least, while a sampler spends all 36 renders where
    supervision is actually wanted.

    **The inner edge is the denoising path** (`min_path_angle_deg`, against
    `path_cameras`) — a band swept along that path, not a hole punched
    around the source view. The training these supervise is fitted to a
    batch of denoised frames covering a whole orbit: for `render`'s
    `pattern: circular`, one elevation and 360 degrees of azimuth. Every
    frame on it is a denoised view in its own right, so a supporting view
    sitting on the path competes with one wherever it sits — 140 degrees
    round the orbit from the photograph no less than at the anchor. Out of
    distribution means *off the path*, and for a circle that means a
    different elevation; the distance measured here is to the nearest path
    camera, so one threshold would mean the same thing on a helix.
    Against a 30-degree cap of 36 views
    on an 81-camera ring, the default 5 degrees drops the 7 innermost.

    **A render taken along the path's own cameras keeps nothing at all** —
    `render_splat` with `pattern: ""` reuses the dataset's cameras, so
    every view is zero degrees from the path. That is the rule working
    rather than failing, and it is why the supporting views get a `cap`
    render of their own.

    **There is only one way to draw the outer edge, and it is the cap.**
    There used to be a second — a `view_roles` input carrying
    `composite_splat_views`' per-frame verdict, and a `max_angle_deg` that
    culled on the angle that step had measured. Both were off in the
    shipped wiring, because the cap already drew the edge, and both went
    with the step: the compositing is `render`'s `...+splat` mode now
    (see steps/render.py), which publishes no roles because nothing needs
    them. A render this is pointed at that was *not* sampled as a cap
    therefore gets no outer edge at all.

    **Un-premultiplying** is what makes the render straight-alpha, which is
    what brush's masked mode expects: it does not premultiply masked ground
    truth, so a `colour*a` frame would ask the model to be dark and
    semi-transparent along the silhouette instead of opaque and the right
    colour. Only the soft rim differs — inside the matte alpha is 1 and the
    two are identical — but the rim is exactly where a face splat's
    silhouette is decided.
    """

    PARAMS = (
        Param("min_path_angle_deg", float, 5.0,
              "Drop the frames within this angle of the DENOISING PATH — the "
              "cameras of the batch brush is training on, wired in as "
              "`path_cameras`. Those views have a denoised frame of their own "
              "already, carrying the photograph at full resolution, and a render "
              "of the same view would only compete with it. Measured to the "
              "nearest path camera, so on a circular orbit it is the elevation "
              "difference and on a helix it follows the sweep. 0 keeps the frames "
              "on the path", minimum=0.0, maximum=180.0),
        Param("unpremultiply", bool, True,
              "Divide the colour back out by alpha, turning a render made on black "
              "into the straight-alpha frame brush's masked mode expects. Off leaves "
              "the render as it came, which darkens the soft silhouette",
              advanced=True),
        Param("min_alpha", float, 0.004,
              "Alpha at or below this is treated as fully transparent: the colour "
              "there is not recoverable by dividing and the mask weights it at zero "
              "anyway. The same default the compositing uses",
              minimum=0.0, maximum=1.0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = list(inputs["images"])
        masks: List[np.ndarray] = list(inputs["masks"])
        cameras = list(inputs["cameras"])
        path_cameras = inputs.get("path_cameras")
        min_path_angle = params["min_path_angle_deg"]
        min_alpha = params["min_alpha"]

        if not (len(images) == len(masks) == len(cameras)):
            raise ValueError(
                f"select_support_views: {len(images)} frames, {len(masks)} alphas "
                f"and {len(cameras)} cameras. The three describe the same views and "
                f"have to arrive together."
            )

        # The outer edge is the cap's own radius, drawn by the sampler that
        # rendered these views rather than by a cull here — see the
        # docstring's "Which render".
        keep = list(range(len(images)))

        # The inner edge: the band swept along the denoising path.
        on_path: List[int] = []
        if path_cameras and min_path_angle > 0.0:
            pivot = _composite_pivot(inputs, where="select_support_views")
            path_dirs = [_direction(pivot, camera) for camera in path_cameras]
            on_path = [i for i in keep
                       if _nearest_path_angle_deg(
                           _direction(pivot, cameras[i]), path_dirs) < min_path_angle]
            dropped = set(on_path)
            keep = [i for i in keep if i not in dropped]
        elif min_path_angle > 0.0:
            # Not a warning: a splat that is not fed by a denoised orbit has
            # no path to be off, and the default would then make every such
            # run shout.
            logger.info(
                "select_support_views: min_path_angle_deg is %.1f but no "
                "path_cameras were wired in, so there is no denoising path to "
                "measure against and every frame is kept — including any sitting "
                "on it.", min_path_angle,
            )

        out_images, out_masks, out_cameras = [], [], []
        for index in keep:
            alpha = np.asarray(masks[index], dtype=np.float32)
            layer = images[index]
            # The same requirement steps/render.py's `+splat` compositing
            # has, for a different reason: dividing by alpha only recovers
            # the straight colour if the render was premultiplied over
            # black. There the background is passed by the code, so it
            # cannot be got wrong; here it comes off a workflow's wiring.
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
            "select_support_views: %d/%d frames kept as supporting views%s",
            len(keep), len(images),
            f", {len(on_path)} dropped for sitting within {min_path_angle:.1f} deg "
            f"of the denoising path" if on_path else "",
        )
        if not keep:
            # Not an error: brush takes no supporting views and trains
            # exactly as it did before. Worth saying out loud, because the
            # run then silently loses the thing this wiring exists for.
            logger.warning(
                "select_support_views: nothing is left after the band (at least "
                "%.1f deg off the denoising path), so the training will get no "
                "supporting views. The usual cause is a render taken along the "
                "denoising path's own cameras — `render_splat` with "
                "`pattern: \"\"` — every frame of which is ON the path; the "
                "supporting views want a `cap` render of their own.",
                min_path_angle,
            )
        return {"images": out_images, "masks": out_masks, "cameras": out_cameras}


@register_step("merge_support_views")
class MergeSupportViewsStep(Step):
    """Two sets of supporting views into the one set `brush` reads.

    inputs: {"a_images"/"a_masks"/"a_cameras": Optional[List], one triple,
             "b_images"/"b_masks"/"b_cameras": Optional[List], the other}
    outputs: {"images": List, "masks": List, "cameras": List}

    `brush` takes a single `support_*` triple (steps/brush.py), and this
    pipeline now has two producers of one: the face cap
    (`select_support_views` over `render_face_support_views`) and the
    stage-1 body shells (`select_support_views` over
    `pointmap_elevation_views`). Concatenating them is all this does.

    **It runs ungated, so it is always the writer of
    `scene.support_views.*`.** With both branches switched off it emits
    three empty lists and `brush` trains exactly as it did before either
    existed — `_SupportViews.from_inputs` already treats an empty list as
    absent. That is why the inputs are optional reads (`?`): a
    `when:`-skipped branch simply never writes its path, and there is no
    way to conditionally wire an input.

    A missing or `None` triple contributes nothing. Lengths *within* a
    triple must agree, and the error names which triple — the two arrive
    from different branches of the workflow, so "3 images and 4 masks" is
    only actionable if you know whose.

    Order is a's views then b's. Nothing downstream depends on it: brush
    weights a supporting view by its own mask, not by its position in the
    list.
    """

    PARAMS = (
        Param("require_any", bool, False,
              "Raise instead of publishing three empty lists. Off by default "
              "because empty is a legitimate configuration — both branches "
              "switched off — and brush handles it. Turn it on in a workflow "
              "whose whole point is the supporting views, where getting none "
              "means a branch silently did not run"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        cameras: List[Any] = []

        for prefix in ("a", "b"):
            triple = [inputs.get(f"{prefix}_{name}")
                      for name in ("images", "masks", "cameras")]
            if all(part is None for part in triple):
                continue
            parts = [list(part or []) for part in triple]
            if len({len(part) for part in parts}) != 1:
                raise ValueError(
                    f"merge_support_views: triple '{prefix}' has "
                    f"{len(parts[0])} images, {len(parts[1])} masks and "
                    f"{len(parts[2])} cameras. The three describe the same "
                    f"views and have to arrive together."
                )
            images += parts[0]
            masks += parts[1]
            cameras += parts[2]

        if not images:
            if params["require_any"]:
                raise ValueError(
                    "merge_support_views: neither branch produced a supporting "
                    "view and require_any is set. Check that the branch you "
                    "expected is switched on and that its select_support_views "
                    "kept anything."
                )
            logger.info(
                "merge_support_views: no supporting views from either branch; "
                "the training will run without them, as it did before they existed"
            )
        else:
            logger.info("merge_support_views: %d supporting views", len(images))
        return {"images": images, "masks": masks, "cameras": cameras}


def _direction(pivot: np.ndarray, camera: Any) -> np.ndarray:
    """Which way the splat is seen from, as a unit vector.

    Positions alone would make the metric depend on how far down a ray a
    camera sits; what matters to both edges is the direction the subject is
    seen from, which is the orbit sphere's own coordinate.
    """
    offset = np.asarray(camera.position, dtype=np.float64).reshape(3) - pivot
    norm = float(np.linalg.norm(offset))
    return offset if norm < 1e-9 else offset / norm


def _nearest_path_angle_deg(direction: np.ndarray,
                            path_dirs: List[np.ndarray]) -> float:
    """Angle from one view direction to the closest one on the path.

    The path is a set of cameras rather than a curve, so this is the
    distance to the nearest sample of it. That is the right measure for
    what it is used for — a supporting view competes with an actual
    denoised frame, and the frames are the samples — and it makes the
    threshold behave the same on a circle, where it comes out as the
    elevation difference, and on a helix, where it follows the sweep.
    """
    cosines = np.clip([float(np.dot(direction, other)) for other in path_dirs],
                      -1.0, 1.0)
    return float(np.degrees(np.arccos(cosines.max())))


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


def _composite_pivot(inputs: Dict[str, Any],
                     where: str = "select_support_views") -> np.ndarray:
    """Where to measure the view angle about: the splat's centre if known.

    For a head on a full-body orbit the splat's centre and the orbit target
    are not interchangeable: a camera 30 degrees up from the equator views
    the *body* at 30 degrees and the *head* at rather less. body2colmap's
    `OrbitPipeline.splat_view_angle_deg` measures from the splat's own bbox
    centre for that reason, `steps/render.py` reads it straight off the
    scene (`SplatScene.get_bbox_center()`) when it culls the `+splat`
    overlay, and select_support_views measures its distance to the denoising
    path about the same point.
    """
    center = inputs.get("splat_center")
    if center is None:
        center = inputs.get("orbit_target")
    if center is None:
        raise ValueError(
            f"{where}: wire either 'splat_center' (the splat's "
            f"own world centre — pointmap_splat publishes it as "
            f"splat_stats.world_center) or 'orbit_target'. The view angle has "
            f"to be measured about something."
        )
    return np.asarray(center, dtype=np.float64).reshape(3)


def _check_premultiplied(layer: np.ndarray, alpha: np.ndarray, index: int,
                         min_alpha: float, *, where: str, because: str) -> None:
    """Refuse a splat render that was not made on a black background.

    Only one direction of this is provable from the images alone: where
    alpha is zero the render must be black, because `colour*0 + bg*1` is
    the background and nothing else. A non-black value there is proof the
    caller passed a `bg_color`, and what follows would blend toward it
    twice.

    `where`/`because` only name the caller in the message. There used to be
    two callers with two different second sentences; the compositing one is
    now `steps/render.py`'s `+splat` modes, which need no check at all —
    `render_splat_layers` passes the background itself, so a render it did
    not make on black cannot exist. This one survives because
    select_support_views takes a render a *workflow* wired to it.
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
