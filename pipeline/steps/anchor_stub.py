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

- InjectShellViews is the same idea one step wider, and the only one of
  the three with no ComfyUI ancestor: where inject_anchor puts a real
  photograph on the one frame that sits exactly at the anchor camera,
  inject_shell_views puts novel views rendered off `pointmap_splat`'s
  photo-derived shell on every frame within a few degrees of it. That band
  is the part of the orbit the shell actually knows about, and filling it
  with photoreal renders instead of outline drawings is what the shell was
  built for. See that class's docstring for the two radii and why the
  default only substitutes, never trusts.

- CompositeSplatViews is the fourth, and the only one that does not
  replace a frame: it alpha-composites a splat render ON TOP of the mesh
  drawing, keeping the drawing everywhere the splat is transparent. That is
  what puts a real face on a skeleton rig, and it is why it survives much
  further off the source view than a substitution does — a face-only splat
  read cleanly out to about 30 degrees and still holds at 45, against the
  15 the body shell's substitution band is set to. The cull is set wider
  still, at 60: see its docstring for what is being bought out there and
  what it costs. See its docstring for
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
tests/test_anchor.py. `inject_shell_views` is the newest and the least
proven: its geometry is covered on synthetic camera paths
(tests/test_shell_views.py) and its inputs came out of one local end-to-end
smoke run of the shell, but no diffusion pass has ever been conditioned on
a batch it produced.

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
from typing import Any, Dict, List, Optional

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


@register_step("inject_shell_views")
class InjectShellViewsStep(Step):
    """Swap novel views rendered off the photo-derived shell into the frames
    that sit near the anchor camera, and mark each frame's VACE role.

    inputs: {"images": List[np.ndarray] — the mesh render, one per frame,
             "shell_images": List[np.ndarray] — `render_splat` over the SAME
             cameras, in the same order (that is what `pattern: ""` gives),
             "cameras": List[Camera], "orbit_target": np.ndarray (3,),
             "anchor_position": Optional[np.ndarray] (3,) — where the photo
             was taken from, in world coordinates. Omit it and the world
             origin is used, which is where it is: `sam3d_to_world` does not
             recentre, so the SAM-3D-Body camera IS the world origin (and
             `cyber_6f/initial` records exactly that as its anchor_position).
             An anchored render publishes it explicitly; an unanchored one
             does not, and does not need to,
             "shell_cameras": Optional[List[Camera]] — wire it and the two
             batches are checked to be the same views rather than assumed}
    outputs: {"images": List[np.ndarray],
              "masks": List[np.ndarray] — the VACE mask batch (see below),
              "anchor_position": np.ndarray (3,) — the position the band was
              measured from, echoed so a workflow with no anchored render can
              still publish it as `dataset.extras.anchor_position`,
              "view_roles": List[dict] — per frame: index, angle from the
              anchor, role, source. Diagnostics; nothing downstream reads it}

    `pointmap_splat` builds a 2.5-D shell of the side of the subject the
    reference photo saw. It reprojects onto that photo exactly, so at the
    anchor camera it *is* the photo, and it stays usable for about ±19
    degrees around it before the missing back half and the grazing-incidence
    fringes show. This step is what spends that band: within it the frame the
    diffusion pass conditions on becomes a photoreal render of the subject
    instead of a flat outline+skeleton drawing of the mesh; outside it the
    mesh render is still the only thing that knows what is there.

    **It supersedes `inject_anchor` in a workflow that has a shell.** They
    do the same thing: put content that came from the reference photograph
    into the batch, and mark those frames "keep it" for the diffusion pass.
    inject_anchor does it for the one frame sitting exactly at the source
    camera, by warping the photo onto it; this does it for every frame the
    shell can speak for, by rendering the shell — and at the source camera
    the two agree to a best-fit shift of (1, 0) px (the smoke run's anchor
    gate), because the shell reprojects onto the photo it was built from.
    A workflow that runs this one therefore does not need `render`'s
    `override_cam_from_mesh`, `generate_firstlast` or `inject_anchor` at
    all: the path does not have to be bent to put a camera exactly on the
    photo's, and nothing has to warp the photo onto it.

    Two radii, because "substituted" and "trusted" are different questions:

      * `reference_radius_deg` — substituted AND marked VACE 0.0, "a real
        photograph, keep it". Zero, the default, means no such band at all,
        because a shell render is not a photograph: it has holes where the
        photo saw nothing and soft edges where the Gaussians are uncertain,
        and telling the diffusion pass to keep such a frame verbatim writes
        those artefacts into the batch every later stage is built on. Raise
        it — to the replace radius, which makes the whole band the batch's
        reference material — in a workflow where this step *replaces* the
        anchor injection, because then nothing else marks any frame as real
        and the diffusion pass has no photographic content to hold on to
        beyond the reference image. That is the trade: shell artefacts kept
        verbatim on a handful of frames, against a first pass with no
        anchored appearance anywhere.
      * `replace_radius_deg` — substituted but still marked 1.0, "synthetic,
        denoise it". This is the useful band: better conditioning content,
        no claim that it is ground truth. 15 degrees is what the smoke run
        measured as comfortably inside the shell's reliable range.

    **This step manufactures the VACE mask batch**, exactly as the
    pre-denoise `inject_anchor` call does and for the same reason: at this
    point in a bootstrap `dataset.masks` still holds the mesh render's
    per-pixel silhouettes, which nothing downstream reads, and the field's
    next reader is `wan22_vace_denoise`'s `control_masks`. So it is written
    here rather than passed through — 1.0 everywhere, 0.0 inside the
    reference band. If an `inject_anchor` does still follow it, wire
    `masks: dataset.masks` into that step, or it will manufacture an
    all-1.0 batch over the top and the reference band will be silently
    lost.

    The band is measured as the angle between two directions from the orbit
    target — the frame's camera and the anchor camera — not as a difference
    of azimuths. A helical path sweeps elevation as well, and 15 degrees of
    azimuth, 15 of elevation and 10.6 of both should all be the same
    distance from the anchor, which is what a single radius has to mean.
    """

    PARAMS = (
        Param("replace_radius_deg", float, 15.0,
              "Substitute the shell's render for the mesh render on every "
              "frame whose camera is within this angle of the anchor's. 0 "
              "disables the substitution entirely", minimum=0.0, maximum=180.0),
        Param("reference_radius_deg", float, 0.0,
              "Of those, the ones this close to the anchor are additionally "
              "marked as VACE references (mask 0.0, 'keep it'). A shell render "
              "is not a photograph — see the class docstring before raising it",
              minimum=0.0, maximum=180.0),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        images: List[np.ndarray] = list(inputs["images"])
        shell_images: List[np.ndarray] = list(inputs["shell_images"])
        cameras = inputs["cameras"]
        replace_radius = params["replace_radius_deg"]
        reference_radius = params["reference_radius_deg"]

        if reference_radius > replace_radius:
            raise ValueError(
                f"inject_shell_views: reference_radius_deg "
                f"({reference_radius}) exceeds replace_radius_deg "
                f"({replace_radius}), so it would mark frames as references "
                f"that were never substituted — i.e. tell the diffusion pass "
                f"to keep a mesh render verbatim."
            )
        if len(shell_images) != len(images):
            raise ValueError(
                f"inject_shell_views: {len(shell_images)} shell renders "
                f"against {len(images)} frames. The two batches are matched by "
                f"index, which holds because render_splat with `pattern: \"\"` "
                f"reuses the source dataset's cameras verbatim and in order — "
                f"a pattern on that step breaks this."
            )

        # Cheap proof of that index alignment when the cameras are wired.
        shell_cameras = inputs.get("shell_cameras")
        if shell_cameras is not None:
            drift = max(
                float(np.linalg.norm(np.asarray(a.position, dtype=np.float64)
                                     - np.asarray(b.position, dtype=np.float64)))
                for a, b in zip(cameras, shell_cameras)
            ) if cameras else 0.0
            if drift > 1e-4:
                raise ValueError(
                    f"inject_shell_views: the shell render's cameras are not "
                    f"the frame batch's (worst position drift {drift:.6f}). "
                    f"Render the shell with `pattern: \"\"` and the dataset "
                    f"wired, so it reuses these cameras verbatim."
                )

        masks = [np.ones(img.shape[:2], dtype=np.float32) for img in images]

        # Where the photograph was taken from. An anchored render publishes
        # it; an unanchored one does not, and does not have to — the world
        # origin is that camera by construction (see the class docstring).
        supplied = inputs.get("anchor_position")
        anchor = (np.zeros(3) if supplied is None
                  else np.asarray(supplied, dtype=np.float64).reshape(3))

        if replace_radius <= 0.0:
            logger.info("inject_shell_views: replace_radius_deg is 0, nothing "
                        "substituted; emitting an all-synthetic VACE mask batch")
            return {"images": images, "masks": masks, "anchor_position": anchor,
                    "view_roles": [{"index": i, "angle_from_anchor_deg": None,
                                    "role": "mesh", "source": "mesh"}
                                   for i in range(len(images))]}

        target = np.asarray(inputs["orbit_target"], dtype=np.float64).reshape(3)
        anchor_dir = anchor - target
        if float(np.linalg.norm(anchor_dir)) < 1e-9:
            raise ValueError(
                "inject_shell_views: the camera the photo was taken from sits "
                "on the orbit target, so there is no direction to measure a "
                "band around. Either anchor_position and orbit_target did not "
                "come from the same render, or an unanchored run put the orbit "
                "target on the world origin, where this step assumes the "
                "SAM-3D-Body camera is."
            )

        view_roles = []
        for index, camera in enumerate(cameras):
            direction = np.asarray(camera.position, dtype=np.float64) - target
            angle = _angle_between_deg(direction, anchor_dir)
            # `reference_radius > 0` and not just `angle <= radius`: 0 has
            # to mean "no reference band at all", and the anchor frame sits
            # at exactly 0 degrees. Its role here is moot either way — the
            # inject_anchor after this one overwrites that frame with the
            # real warped photograph and marks it 0.0 itself.
            if reference_radius > 0.0 and angle <= reference_radius:
                role, source = "reference", "shell"
            elif angle <= replace_radius:
                role, source = "replace_diffuse", "shell"
            else:
                role, source = "mesh", "mesh"

            if source == "shell":
                shell = shell_images[index]
                if tuple(shell.shape) != tuple(images[index].shape):
                    raise ValueError(
                        f"inject_shell_views: shell render {index} has shape "
                        f"{tuple(shell.shape)} against the frame's "
                        f"{tuple(images[index].shape)}. Render the shell at the "
                        f"same resolution as the orbit."
                    )
                images[index] = shell
                if role == "reference":
                    # 0.0 = "a real photograph, keep it", uniform over the
                    # frame: a per-frame flag, not a silhouette. Same
                    # convention as inject_anchor — see the module docstring.
                    masks[index] = np.zeros(shell.shape[:2], dtype=np.float32)

            view_roles.append({
                "index": index,
                "angle_from_anchor_deg": round(angle, 3),
                "role": role,
                "source": source,
            })

        counts = {role: sum(1 for v in view_roles if v["role"] == role)
                  for role in ("reference", "replace_diffuse", "mesh")}
        logger.info(
            "inject_shell_views: %d/%d frames within %.1f deg of the anchor take "
            "the shell's render (%d of them marked as VACE references within "
            "%.1f deg); %d keep the mesh render",
            counts["reference"] + counts["replace_diffuse"], len(images),
            replace_radius, counts["reference"], reference_radius, counts["mesh"],
        )
        if counts["reference"] + counts["replace_diffuse"] == 0:
            logger.warning(
                "inject_shell_views: no frame is within %.1f deg of the anchor, "
                "so the shell was rendered and then discarded. Either the orbit "
                "is not anchored (override_cam_from_mesh) or the radius is "
                "smaller than the path's angular step.", replace_radius,
            )

        return {"images": images, "masks": masks, "anchor_position": anchor,
                "view_roles": view_roles}


def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions, in degrees.

    The orbit sphere's own metric: this is what makes one radius mean the
    same thing whether the camera moved in azimuth, in elevation, or in
    both — which a helical path does.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 180.0
    cosine = float(np.dot(a, b)) / (norm_a * norm_b)
    return float(np.degrees(np.arccos(min(max(cosine, -1.0), 1.0))))


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
             is measured about. `pointmap_splat` publishes it as
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
    mode (`c65a7f7`). The geometry and the compositing rule are that
    commit's, and the cull angle started as its 45 before this project
    widened it (see **Why the cull**); what differs is where the layer comes
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
    edge. Those are properties of a 2.5-D shell and hold here.

    **Why 60 anyway.** The default is the far end of that measurement rather
    than the middle of it, which is a deliberate trade and not a reading of
    it. What sits between 45 and 60 is the part of the orbit furthest from
    the photograph that the splat saw anything of at all, and out there the
    alternative is a skeleton drawing with no face on it at all. What makes
    the trade affordable is what happens next to these frames: they are
    inputs to two denoise passes, which are free to rewrite a flared rim,
    and a rim that reads as a jaw edge is a better prompt than nothing.
    `select_support_views` — evidence that no diffusion pass stands in front
    of — makes the opposite trade and culls at 30. Pull this back to 45 for
    body2colmap's measured clean band.

    **The pivot is the splat's centre, not the orbit target.** A head sits
    well above a full-body orbit target, so a camera 30 degrees up from the
    equator views the *body* at 30 degrees and the *head* at rather less.
    body2colmap's `splat_view_angle_deg` measures from the splat's own bbox
    centre for exactly this reason, and this follows it.
    """

    PARAMS = (
        Param("max_angle_deg", float, 60.0,
              "Composite the splat on every frame whose view of it is within "
              "this angle of the photograph's. Past it the layer is dropped "
              "entirely rather than faded: what appears out there is the "
              "shell's open rim, and a half-transparent rim is still a rim. "
              "60 runs to the far end of body2colmap's measured band, where the "
              "shell is mostly edge, to reach the views the photograph is "
              "furthest from; 45 is that measurement's clean limit, and what a "
              "frame nobody denoises afterwards should be held to (see "
              "select_support_views' own, tighter cull). 0 disables the "
              "compositing", minimum=0.0, maximum=180.0),
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

        # The same cheap proof of index alignment inject_shell_views makes.
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
             dropped, and its `angle_from_anchor_deg` is what the outer
             edge of the band is measured on,
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
    (steps/splat.py's `cap_directions`). NOT the batch
    `composite_splat_views` composited — that one rides the dataset's
    cameras because its frames have to land on those exact drawings, and
    every one of them is a view the training already has a denoised frame
    for. The cap's radius is the outer edge of the band, and it belongs
    there rather than here: culling views after rendering them keeps the
    ones nearest the photograph, which are the ones worth least, while a
    sampler spends all 36 renders where supervision is actually wanted.

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
    camera, so one threshold means the same thing for the helix
    fast_helical_shell.yaml renders on. Against a 30-degree cap of 36 views
    on an 81-camera ring, the default 5 degrees drops the 7 innermost.

    **A render taken along the path's own cameras keeps nothing at all** —
    `render_splat` with `pattern: ""` reuses the dataset's cameras, so
    every view is zero degrees from the path. That is the rule working
    rather than failing, and it is why the supporting views get a `cap`
    render of their own.

    **`view_roles` and `max_angle_deg` are the other way to draw the outer
    edge**, for a render that was not sampled as a cap: they cull on the
    angle `composite_splat_views` measured, and are off in the shipped
    wiring (`role: ""`, `max_angle_deg: 180`) because the cap already drew
    it. Keep them for a batch that shares the composite's cameras — the
    roles are indexed against ITS frames, so they must never be wired
    against a different view set. An edge with nothing to measure against
    is skipped, with a line in the log saying so.

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
        Param("min_path_angle_deg", float, 5.0,
              "Drop the frames within this angle of the DENOISING PATH — the "
              "cameras of the batch brush is training on, wired in as "
              "`path_cameras`. Those views have a denoised frame of their own "
              "already, carrying the photograph at full resolution, and a render "
              "of the same view would only compete with it. Measured to the "
              "nearest path camera, so on a circular orbit it is the elevation "
              "difference and on a helix it follows the sweep. 0 keeps the frames "
              "on the path", minimum=0.0, maximum=180.0),
        Param("max_angle_deg", float, 30.0,
              "Drop the frames whose view of the splat is FURTHER than this from "
              "the photograph's, when a `view_roles` input is wired in. Tighter "
              "than composite_splat_views' cull on purpose: a composited frame is "
              "an input to diffusion, which can rewrite the shell's rim, but a "
              "supporting view is fitted straight into the splat, where the rim "
              "would be reconstructed instead. 30 is where body2colmap measured a "
              "Face_Neck shell still reading cleanly. 180 leaves the outer edge to "
              "the `role` filter alone", minimum=0.0, maximum=180.0),
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
        path_cameras = inputs.get("path_cameras")
        role = params["role"]
        min_path_angle = params["min_path_angle_deg"]
        max_angle = params["max_angle_deg"]
        min_alpha = params["min_alpha"]

        if not (len(images) == len(masks) == len(cameras)):
            raise ValueError(
                f"select_support_views: {len(images)} frames, {len(masks)} alphas "
                f"and {len(cameras)} cameras. The three describe the same views and "
                f"have to arrive together."
            )

        keep = list(range(len(images)))

        # The outer edge, and the composite's own verdict on which frames
        # its splat can speak for. Both read from view_roles rather than
        # re-derived: the pivot, the source view and the sphere metric are
        # measured once, in composite_splat_views.
        far: List[int] = []
        if view_roles is not None:
            if len(view_roles) != len(images):
                raise ValueError(
                    f"select_support_views: {len(view_roles)} view roles against "
                    f"{len(images)} frames. Wire the `view_roles` of the "
                    f"composite_splat_views that took THIS render."
                )
            if role:
                keep = [i for i in keep if view_roles[i].get("role") == role]
            angles = {i: _angle_of(view_roles[i], "angle_from_anchor_deg")
                      for i in keep}
            far = [i for i in keep
                   if angles[i] is not None and angles[i] > max_angle]
            dropped = set(far)
            keep = [i for i in keep if i not in dropped]
        elif max_angle < 180.0:
            logger.info(
                "select_support_views: max_angle_deg is %.1f but no view_roles were "
                "wired in, so there is no view angle to measure and no outer edge.",
                max_angle,
            )

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
            "select_support_views: %d/%d frames kept%s as supporting views%s%s",
            len(keep), len(images),
            f" (role={role})" if view_roles is not None and role else "",
            f", {len(on_path)} dropped for sitting within {min_path_angle:.1f} deg "
            f"of the denoising path" if on_path else "",
            f", {len(far)} dropped beyond {max_angle:.1f} deg" if far else "",
        )
        if not keep:
            # Not an error: brush takes no supporting views and trains
            # exactly as it did before. Worth saying out loud, because the
            # run then silently loses the thing this wiring exists for.
            logger.warning(
                "select_support_views: nothing is left after the band (role %r, "
                "at least %.1f deg off the denoising path, within %.1f deg of the "
                "photograph's view), so the training will get no supporting views. "
                "The usual cause is a render taken along the denoising path's own "
                "cameras — `render_splat` with `pattern: \"\"` — every frame of "
                "which is ON the path; the supporting views want a `cap` render "
                "of their own.",
                role, min_path_angle, max_angle,
            )
        return {"images": out_images, "masks": out_masks, "cameras": out_cameras}


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


def _angle_of(entry: Dict[str, Any], key: str) -> Optional[float]:
    """One of a view role's angles, or None when it has none.

    `composite_splat_views` publishes None on every frame when its own
    compositing is switched off (`max_angle_deg: 0`). There is then nothing
    to measure an edge against, and the frame is kept rather than guessed
    at.
    """
    angle = entry.get(key)
    return None if angle is None else float(angle)


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


def _composite_pivot(inputs: Dict[str, Any],
                     where: str = "composite_splat_views") -> np.ndarray:
    """Where to measure the view angle about: the splat's centre if known.

    See CompositeSplatViewsStep's docstring — for a head on a full-body
    orbit the splat's centre and the orbit target are not interchangeable.
    Shared with select_support_views, which measures its distance to the
    denoising path about the same point for the same reason.
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
