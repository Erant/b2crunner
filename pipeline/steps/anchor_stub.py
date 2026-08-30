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

The alpha convention is what all three write. This is the mechanism behind
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
