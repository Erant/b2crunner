"""pointmap_splat — one photograph into a Gaussian-splat shell, placed in
the SAM-3D-Body camera's world.

A feed-forward alternative to `brush`: no optimisation, no training loop,
one forward pass of Sapiens2's *pointmap* head over the reference photo,
one oriented Gaussian per foreground pixel. What comes out is a 2.5-D
shell — the side of the subject the camera can see, and nothing behind it
— which is useless as a deliverable and valuable as a *source of views*.
The two consumers it was built for, both of them now wired up in
`workflows/fast_helical_native.yaml`'s bootstrap:

  * extra reference views for the VACE conditioning batch, rendered off
    the shell at angles the single photo does not cover;
  * a shallow 360-degree helical camera path for the first denoise pass,
    in place of the degenerate circular one. A circular path re-photographs
    the same great circle 81 times, which is as close to zero parallax as a
    camera path gets and therefore the worst possible input to a splat
    trainer. A helix that also sweeps elevation gives brush something to
    triangulate. Novel views off this shell can stand in for the frames of
    that path the mesh render cannot supply.

Ported from ~/Projects/masktest (`face_to_splat.py`, `normal_integration.py`,
`splat.py`, and the measurements in its SPEC.md/NOTES.md), with three
deliberate departures — see "What is different from masktest" below.

Pipeline
--------
    rmbg            -> foreground matte           (inputs["mask"])
    sapiens2_lite   -> surface normals            (inputs["normal_map"])
    sam3d_body      -> mesh + camera focal length (inputs["mesh_output"])
    THIS STEP       -> Sapiens2 pointmap -> depth -> oriented Gaussians -> .ply

Only the pointmap head runs here. Segmentation is *not* run: the Sapiens2
seg head would work (transformers ships `Sapiens2ForSemanticSegmentation`,
and masktest uses it), but on whole-body inputs RMBG-2.0 — already in this
pipeline, already loaded for every run — produces the better matte, and it
comes out soft, which is exactly what the silhouette alpha wants. Normals
come from the existing `sapiens2_lite` step rather than being re-inferred
here.

Two specializations
-------------------
`PointmapSplatStep` is the base and is not registered. What a workflow
names is one of:

  * **`pointmap_splat`** — the whole subject from the whole photograph.
    The body shell described everywhere below; unchanged.
  * **`face_pointmap_splat`** — a head, from a crop of that same
    photograph, with a segmentation mask instead of RMBG's matte. The
    crop frames the head against the torso and centres it in Sapiens2's
    768x1024 (`crop_to_box`, `padding` 3.5); it is emphatically NOT a
    close-up, which the pointmap head answers with a flat card.

They differ in exactly two things: `_source_intrinsics`, which says where
the image sits in SAM-3D-Body's camera (the full frame, or a crop of it),
and two measured defaults (`splat_scale`, `fill_max_frac`). Everything
between — the mask morphology, the intrinsics fit, the depth solve, the
scale fit to the mesh, the Gaussian construction, the PLY layout — is
shared verbatim, which is the point of the split.

How it is wired
---------------
`workflows/fast_helical_native.yaml` is that wiring; read it for the whole
bootstrap. Everything this step needs already existed on that prologue, and
the three feeds must all be the same photo — `scene.front_image`:

    - id: front_matte
      step: rmbg
      dispatch: in_process
      inputs: {image: scene.front_image}
      outputs: {mask: scene.front_matte}

    - id: front_normals
      step: sapiens2_lite
      dispatch: in_process
      inputs: {image: scene.front_image}
      outputs: {normal_map: scene.front_normals}

    - id: shell_splat
      step: pointmap_splat
      dispatch: in_process
      inputs:
        image: scene.front_image
        mask: scene.front_matte
        normal_map: scene.front_normals
        mesh_output: scene            # sam3d_body's own outputs
      params:
        filepath: ${globals.output_root}/shell/shell.ply
      outputs:
        splat_path: scene.shell_splat_path
        splat_stats: scene.shell_splat_stats

`render_splat` takes that `splat_path` directly — with `pattern: ""` it
re-renders the shell along the mesh render's own cameras, which is what
lets `inject_shell_views` swap those frames into the band around the source
view and mark them as the batch's real-photograph frames. That is the job
`generate_firstlast` + `inject_anchor` used to do for one anchored frame,
which is why neither is in that bootstrap any more. If
`fix_head_angle` is ever put back in place of the pose fit, it belongs
BEFORE this step: this one reads `mesh_output["vertices"]`, and the head
correction rewrites them.

`refine_pose_to_splat` follows it there, for the same reason it was
written — it re-poses the body so the mesh agrees with the shell in novel
views, which is what stops the rendered skeleton drifting off the subject a
few degrees either side of the anchor. Note it is mutually exclusive with
`fix_head_angle` (see `INCOMPATIBLE_STEPS` in pipeline/workflow.py), so a
workflow picks one; and re-running this step afterwards, on the re-posed
mesh, is the untried second iteration of the loop.

Coordinate frames
-----------------
Two camera frames are in play and they are the same physical camera, from
two different monocular estimates of it:

  * the *pointmap* frame — OpenCV (X right, Y down, Z into the scene),
    camera at the origin, with whatever focal length AND principal point
    the Sapiens2 network implicitly committed to for the image it was
    given. For a crop that principal point is the CROP's centre, so the
    network's optical axis is turned toward the crop by atan(offset / f)
    — 12.6 deg for a head at the top of cyber2_6f's portrait frame;
  * the *SAM-3D-Body* frame — also OpenCV, also camera at the origin, with
    `mesh_output["focal_length"]` (MoGe-2's estimate, see steps/sam3d_body.py)
    and the principal point at the FULL frame's centre. `pred_vertices +
    pred_cam_t` lives here; `body2colmap.coordinates.sam3d_to_world` turns
    it into the pipeline's world frame with a 180-degree rotation about X,
    i.e. an elementwise `diag(1, -1, -1)`.

Every camera downstream — the mesh render's anchor, the orbit path, the
COLMAP export, `render_splat` — is built on the second, so the Gaussians
must end up on ITS rays; but the surface's SHAPE is only coherent in the
first, because that is the camera the network's depth is depth along and
the camera its normals are expressed in. So the step does both, in order:

    1. fit (f_p, cx_p, cy_p) to the pointmap; take the normals into that
       frame; integrate the depth there — masktest, verbatim;
    2. rotate the result so the pointmap camera's rays line up with
       SAM-3D-Body's through the same pixels (`rays_rotation`, a Wahba
       fit over the mask). Depth taken after this is depth along the axis
       everything downstream shares;
    3. put every Gaussian on SAM-3D-Body's ray through its own pixel, at

           z = z_mesh + k * (z_net - z_net_ref),   k = s * f_p / f_s

       where `s` is the mesh-depth scale from `depth_scale_to_mesh` and
       `k` — the WIDTH ratio, how much wider the surface is when its pixels
       are re-read through the narrower camera at the mesh's distance — is
       what the relief is scaled by, so relief-to-width stays what the
       network said.

Steps 1 and 3 together are what make the anchor frame exact (every
Gaussian reprojects onto its source pixel by construction — measured on a
real 867x1552 body shot at 2e-5 px) without deforming the face.

**What this replaced, and why it mattered** (2026-08-30). Until then this
step discarded the pointmap camera outright: it read the network's depth
as depth along SAM-3D-Body's axis, integrated in that camera, and scaled
the depth by `s` alone. On a full-frame body shell the principal points
coincide and the only damage is a relief stretched by f_s/f_p (1.57 on
the smoke-test shot — the "over-rotated novel views" the old table below
blamed on a focal disagreement). On a face crop it was fatal twice over:
the depth was assigned along an axis 12.6 deg off the one it was measured
along, so the face nodded back by that angle; and the relief was scaled by
f_s/f_p = 3.19 on top of the width, so a 178 mm wide face came out 338 mm
deep, with the integration making the normals agree LESS (36.5 -> 40.3
deg) because it was reading them in the wrong frame. Corrected: 178 x
155 mm, the face looking at the camera the way the photo does, no neck
sheet. The measurement and the figure are in output/face_mask_compare/
(`diag_fixed_placement.py`, `diag_shipped_vs_fixed.png`) of the run that
fixed it. The old focal-ratio table is kept below for the record of what
the stretch cost; the ratio itself is now a diagnostic, not a price.

What that cost, measured on the smoke-test body shot by sweeping the focal
the step was told SAM-3D-Body found, UNDER THE OLD PLACEMENT (the pointmap
fits itself 1099 px there; SAM-3D-Body's own dimensions-only default,
`sqrt(h**2 + w**2)`, is 1778):

    f_s / f_pointmap   normal agreement after integration
    1.00               19.5 deg -> 6.3 deg
    1.18               21.6 deg -> 7.5 deg
    1.62               26.8 deg -> 12.5 deg

Normal agreement is now measured in the pointmap camera, where the
integration happens, and does not depend on the ratio.

Scale — the remaining degree of freedom, since depth and metric size trade
off exactly under a fixed projection — is fitted to the mesh:
`depth_scale_to_mesh` compares the (rotated) pointmap depth to the mesh's
front surface over a coarse grid of image bins and takes the median ratio.
That puts the shell at the mesh's distance, hence at the mesh's height,
which is what makes the orbit radius computed from the mesh frame it
correctly. The integration is invariant to it (a scale on z is a constant
offset on `w = log z`, and only differences of `w` enter the gradient
equations), so integrating first and scaling afterwards is exact.

What is different from masktest
-------------------------------
1. No segmentation head; the matte is an input (RMBG-2.0), and the soft
   silhouette alpha is that matte rather than a sum of seg class
   probabilities.
2. The Gaussians are placed in SAM-3D-Body's camera (rotated onto its
   rays, relief by the width ratio — see "Coordinate frames"), and the
   world is not recentred on the splat's own centroid. The integration
   itself is masktest's, in the pointmap's own camera.
3. Inference goes through transformers' first-class Sapiens2 support
   (`AutoModelForPointmapEstimation`) rather than the mmengine/`init_model`
   path against a vendored `sapiens2` checkout. Same weights — the two
   files a Sapiens2 repo ships, `model.safetensors` and
   `sapiens2_1b_<task>.safetensors`, are byte-identical blobs — and the
   image processor's `post_process_pointmap_estimation` does exactly what
   masktest's hand-written unpad/resize/`/scale` did. This is also what
   `sapiens2_lite` already does for the normal head, verified on a pod.

**fp32 for the pointmap head is not optional** (masktest measured it):
bf16's ulp near the ~0.7 m a head sits at is ~3.9 mm against a face
spanning ~128 mm of depth, and the median angle between pointmap-derived
and predicted normals goes 23.0 deg (bf16) -> 5.1 deg (fp32). The
segmentation and normal heads are unaffected — argmax and unit vectors.
`dtype` exists to force bf16 anyway on a card that cannot hold fp32; the
run falls back to it by itself on OOM.

Dependencies: scipy (`ndimage` for the mask morphology, `sparse`/`cg` for
the integration) is not a declared dependency of this repo but is present
in every venv the image builds, via body2colmap -> pyrender -> scipy. It is
imported inside functions, as the import-discipline test requires.

Verification status
-------------------
The numerics (intrinsics fit, integration, Gaussian construction, PLY
layout) are a port of code verified in masktest against a real 3DGS
renderer, and are covered here by tests/test_pointmap_splat.py on synthetic
geometry where the answer is known in closed form.

Two things were checked against real data before this shipped, on
masktest's 867x1552 full-body shot:

  * **The transformers path is the mmengine path.** Run head to head
    against masktest's cached `out_body/raw.npz`, the two pointmaps differ
    by a single global scale of 0.991 and essentially nothing else: after
    removing it the median depth difference is 0.94 mm (p99 4.6 mm) on a
    ~1.27 m subject, against 11.4 mm before. The implied focal is 1099 px
    against 1086, and the *shape* is identical — normal agreement of the
    raw pointmap 10.84 deg against 10.91 deg. A global scale is exactly
    what `align_depth` fits out anyway. (The likely cause is the resize
    filter: the processor notes upstream Sapiens2 uses INTER_AREA/INTER_CUBIC
    where it uses bilinear.)
  * **The step runs end to end** at full resolution — 485k unknowns through
    the CG solve, 480k Gaussians, a 27 MB PLY — with the reprojection and
    focal-ratio numbers quoted above.

  * **The placement holds against a real SAM-3D-Body fit.** A full local
    run — sam3d_body (MoGe-2 FOV) -> render (81-frame helix,
    override_cam_from_mesh, outline+skeleton) -> rmbg -> sapiens2_lite ->
    this step -> render_splat over the same 81 cameras — put the shell
    exactly where `generate_firstlast`'s warp of the source photo puts the
    subject at the anchor camera: **best-fit integer shift (1, 0) px** over
    a +-8 px search (MAE 5.53 against 5.86 at (0,0)), which is masktest's
    own gate for "a wrong focal, a wrong principal point or a flipped
    world". Projected mesh height came out within 3% of the matte's. Novel
    views held together visibly out to about +-19 degrees off the anchor.
    (`output/smoke_shell_helical/` on that box, if it is still there.)

The one number that run did NOT like is the focal ratio: MoGe-2 gave
SAM-3D-Body 1731 px on that image where the pointmap implies 1099, a ratio
of 1.57 — the bad end of the table above, and the residual normal
agreement came out at 12.1 deg to match. The shell is correspondingly
stretched along the view axis. Which of the two estimates is wrong for
that image is open, and is the first thing to chase if novel views look
over-rotated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..registry import register_step
from ..step import REQUIRED, Param, Step, with_defaults

logger = logging.getLogger(__name__)

#: The 1B pointmap head. masktest measured every default in this module
#: against this checkpoint; the family also has 0.4b/0.8b variants.
DEFAULT_CHECKPOINT = "facebook/sapiens2-pointmap-1b"

#: brush: crates/brush-render/src/shaders.rs
SH_C0 = 0.2820947917738781

#: OpenCV camera frame (X right, Y down, Z into the scene) -> body2colmap
#: world (X right, Y up, Z toward the viewer). Identical to the rotation
#: `body2colmap.coordinates.sam3d_to_world` applies, and a rotation rather
#: than a reflection (det +1), so it carries an orthonormal frame across
#: unchanged and never flips handedness.
FLIP = np.array([1.0, -1.0, -1.0], np.float64)

#: Sapiens2 normals are OpenGL-convention (+Y up, +Z toward the viewer);
#: the pointmap is OpenCV. Calibrated in the brush fork over all 8
#: axis-sign combinations by masked mean cosine similarity — 0.980 for this
#: mapping against 0.762 for the runner-up (brush/docs/normal-supervision.md).
NORMAL_TO_CAMERA_FRAME = np.array([1.0, -1.0, -1.0], np.float64)

PLY_PROPS = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    """Fill interior holes up to `max_frac` of the subject area; keep bigger.

    **Off by default here, and that is a change from masktest.** Hole
    filling exists because a *segmentation* mask of a head fragments: eyes,
    nostrils and lips are separate seg classes, so the Face_Neck mask is
    riddled with holes and without filling them it falls apart. An RMBG-2.0
    matte of a whole body has no such holes — it is one solid silhouette —
    so every "hole" left in it is real background, and filling one stretches
    a sheet of Gaussians across it.

    Measured on the smoke-test photo (a subject with one arm raised over her
    head): filling at masktest's 0.05 pulled 9741 background pixels into the
    mask — the arm-over-head loop and the hood-to-shoulder gap, each a
    closed region well under 5% of the subject — and those became a slab of
    convention-hall background hanging behind her head in every novel view.
    Size does not separate the two cases once the matte is RMBG's; nothing
    does, because there is nothing legitimate left to fill.

    Kept as a knob rather than deleted, for a seg-derived or head-crop matte
    where the original argument does apply.
    """
    from scipy import ndimage

    if max_frac <= 0:
        return mask
    inverse, count = ndimage.label(~mask)
    if count == 0:
        return mask
    # Anything touching the image border is true background, never a hole.
    border = np.unique(np.concatenate([
        inverse[0], inverse[-1], inverse[:, 0], inverse[:, -1],
    ]))
    sizes = ndimage.sum(np.ones_like(inverse), inverse, index=np.arange(1, count + 1))
    ids = np.arange(1, count + 1)
    fill = ids[(~np.isin(ids, border)) & (sizes <= max_frac * float(mask.sum()))]
    return mask | np.isin(inverse, fill) if fill.size else mask


def clean_mask(matte: np.ndarray, threshold: float = 0.5, fill_max_frac: float = 0.0,
               min_component_frac: float = 0.02, close_iters: int = 4) -> np.ndarray:
    """RMBG matte -> the hard support region the Gaussians are built on.

    Every component at least `min_component_frac` of the largest is kept, so
    a hand or a foot the matte leaves disconnected from the torso survives
    instead of being discarded with the noise.
    """
    from scipy import ndimage

    mask = np.asarray(matte) >= threshold
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=close_iters)

    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
        keep = np.arange(1, count + 1)[sizes >= min_component_frac * sizes.max()]
        mask = np.isin(labels, keep)

    mask = fill_small_holes(mask, fill_max_frac)
    return ndimage.binary_erosion(mask, iterations=1)  # shed 1 px of edge bleed


def soft_alpha(matte: np.ndarray, mask: np.ndarray, erode: int = 3) -> np.ndarray:
    """Per-pixel opacity: the matte at the silhouette, saturated inside.

    A Gaussian's whole advantage over a triangle at the silhouette is that
    it can be partially transparent, so the boundary keeps RMBG's soft
    matte rather than the hard threshold. The interior is pinned to 1
    regardless — a matte legitimately dips over dark hair and clothing
    seams, which must stay opaque.
    """
    from scipy import ndimage

    alpha = np.clip(np.asarray(matte, dtype=np.float32), 0.0, 1.0)
    alpha[ndimage.binary_erosion(mask, iterations=erode)] = 1.0
    return alpha * mask


# ---------------------------------------------------------------------------
# camera / geometry helpers
# ---------------------------------------------------------------------------
def backproject(z: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """Depth map -> HxWx3 points in the camera frame (OpenCV, camera at 0)."""
    height, width = z.shape
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    return np.stack([(uu - cx) * z / f, (vv - cy) * z / f, z], axis=2)


def fit_intrinsics(xyz: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Least-squares fit of (f, cx, cy) to the pointmap's own projection.

    Purely diagnostic here — the pipeline uses SAM-3D-Body's camera, not
    this one — but a valuable diagnostic twice over. The RMS residual is a
    regression guard on the pointmap output (masktest: 0.47 px on a tight
    head crop, 3.40 px over a full body, where a pinhole stops describing
    the network's implicit projection), and the ratio of this focal to
    SAM-3D-Body's says how much the two monocular estimates of the same
    lens disagree.

    Do NOT read the focal off the config instead: `canonical_focal_length`
    is a *training-target* normalisation applied when a ground-truth K is
    known; at inference the network commits to its own focal from image
    content. On masktest's 401-square head crop the config implies f = 401
    and the fit says 667.7, at 0.47 px RMS against 12.0/25.4 px.

    Returns (f, cx, cy, rms_px).
    """
    v, u = np.nonzero(mask)
    x, y, z = (xyz[..., i][mask].astype(np.float64) for i in range(3))

    # Solve u = f*(X/Z) + cx and v = f*(Y/Z) + cy jointly for a single f.
    zero, one = np.zeros(x.size), np.ones(x.size)
    a = np.concatenate([np.stack([x / z, one, zero], 1),
                        np.stack([y / z, zero, one], 1)])
    b = np.concatenate([u, v]).astype(np.float64)
    (f, cx, cy), *_ = np.linalg.lstsq(a, b, rcond=None)
    rms = float(np.sqrt((((a @ [f, cx, cy]) - b) ** 2).mean()))
    return float(f), float(cx), float(cy), rms


def camera_frame_normals(normal_map: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """Sapiens2 normals -> camera frame, unit length, forced camera-facing.

    Unlike masktest's version this decides the sign against the *ray*
    through each pixel rather than against a reconstructed point, so it
    needs no depth map and is therefore usable before the depth exists.
    Identical result: the point is on the ray.
    """
    height, width = normal_map.shape[:2]
    n = np.asarray(normal_map, dtype=np.float64) * NORMAL_TO_CAMERA_FRAME
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12

    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    ray = np.stack([(uu - cx) / f, (vv - cy) / f, np.ones_like(uu)], axis=2)
    # The camera is at the origin looking down +Z, so a surface facing it
    # has n . ray < 0.
    n[(n * ray).sum(2) > 0] *= -1.0
    return n


def rays_rotation(mask: np.ndarray, f_from: float, cx_from: float, cy_from: float,
                  f_to: float, cx_to: float, cy_to: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The rotation carrying one pinhole's rays onto another's, over `mask`.

    Two cameras with the same centre see the same pixel grid through
    different rays when their intrinsics differ. For a crop, the network's
    fitted principal point sits at the crop's centre while SAM-3D-Body's
    sits at the FULL frame's — hundreds of pixels away — so the network's
    optical axis is rotated toward the crop by atan(offset / f). This is
    that rotation, solved as a Wahba problem (Kabsch on unit rays) so it
    also absorbs any small yaw and the sign conventions come out right.

    A rotation cannot make the two ray fields coincide when the focals
    differ (a face subtends 18 deg in the pointmap camera of cyber2_6f's
    crop and 5.7 deg in SAM-3D-Body's); the residual is reported and is
    what the placement in `PointmapSplatStep.run` spends on the shape
    rather than on the framing. Returns (R, stats) with R applied as
    `R @ ray_from ~ ray_to`.
    """
    height, width = mask.shape
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    a = np.stack([(uu - cx_from) / f_from, (vv - cy_from) / f_from, np.ones_like(uu)], 2)[mask]
    b = np.stack([(uu - cx_to) / f_to, (vv - cy_to) / f_to, np.ones_like(uu)], 2)[mask]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))
    residual = np.degrees(np.arccos(np.clip(((a @ rotation.T) * b).sum(1), -1.0, 1.0)))
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    norm = float(np.linalg.norm(axis))
    stats = {
        "rotation_deg": angle,
        "rotation_axis": [float(v) for v in (axis / norm if norm > 1e-12 else axis)],
        "ray_residual_deg": {"median": float(np.median(residual)), "max": float(residual.max())},
    }
    return rotation, stats


def normal_angle_error(xyz: np.ndarray, n_cam: np.ndarray, mask: np.ndarray,
                       erode: int = 3) -> np.ndarray:
    """Per-pixel angle (deg) between finite-difference and predicted normals.

    The headline quality metric, reported before and after integration:
    masktest measured a median of 46 deg on a raw pointmap, 4.2 deg after
    integrating at lambda=0.01. `n_cam` must already be in the camera frame.
    """
    from scipy import ndimage

    gu = np.gradient(xyz.astype(np.float64), axis=1)
    gv = np.gradient(xyz.astype(np.float64), axis=0)
    gn = np.cross(gu, gv)
    gn /= np.linalg.norm(gn, axis=2, keepdims=True) + 1e-12
    # cross(dP/du, dP/dv) points away from the camera in this frame.
    gn[(gn * xyz).sum(2) > 0] *= -1.0

    inner = ndimage.binary_erosion(mask, iterations=erode) if erode else mask
    if not inner.any():
        return np.zeros(0)
    dot = np.clip((gn[inner] * n_cam[inner]).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def integrate_depth(z0: np.ndarray, n_cam: np.ndarray, mask: np.ndarray,
                    f: float, cx: float, cy: float, lam: float = 0.01,
                    grazing_floor: float = 0.05) -> np.ndarray:
    """Re-solve depth from the normals, keeping the pointmap's low frequencies.

    The pointmap head gives good low-frequency shape but badly
    under-predicts relief (masktest: nose protrudes ~8 mm past the forehead
    where reality is 20-25 mm), and finite-difference normals taken from the
    raw pointmap disagree with the *predicted* normals by a median of 46
    degrees. The normal head is the far sharper signal, so: keep the
    pointmap's low frequencies, take the relief from the normals.

    For a pinhole surface P = ((u-cx) z/f, (v-cy) z/f, z), requiring the
    normal to be orthogonal to both tangents gives, with w = log z:

        D   = n_x*xbar + n_y*ybar + n_z*f          (= f * (n . P) / z)
        w_u = -n_x / D          w_v = -n_y / D

    and then a linear least-squares problem in w over the masked pixels:

        min_w  sum_edges weight*(w_j - w_i - g)^2  +  lam * sum_i (w_i - w0_i)^2

    The gradient term sets the relief; the data term pins the low-frequency
    shape and the absolute scale. lambda=0.01 is masktest's measured
    optimum: 4.2 deg of normal agreement at 3 mm of RMS drift from the
    pointmap (0.1 gives 6.3 deg / 1.7 mm, 0.001 gives 3.3 deg / 4.6 mm).

    Because only *differences* of w enter the gradient term and the data
    term is pinned to `z0`, this is equivariant under a global scale on
    `z0`: integrating then scaling and scaling then integrating give the
    same answer. That is what lets the caller fit the mesh scale once, up
    front, and never revisit it.

    `grazing_floor` clamps |D| to that fraction of f: at grazing incidence D
    goes to zero and the implied log-depth gradient diverges, so those edges
    get both a clamped gradient and a proportionally reduced weight rather
    than being dropped outright.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    height, width = mask.shape
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    xbar, ybar = uu - cx, vv - cy
    d = n_cam[..., 0] * xbar + n_cam[..., 1] * ybar + n_cam[..., 2] * f

    # Clamp |D| away from zero, keeping its sign, and record how far we had to.
    floor = grazing_floor * f
    trust = np.clip(np.abs(d) / floor, 0.0, 1.0)   # 1 = face-on, ->0 = grazing
    d_safe = np.where(np.abs(d) < floor, np.sign(d) * floor + (d == 0) * floor, d)
    w_u = -n_cam[..., 0] / d_safe                  # d(log z)/du
    w_v = -n_cam[..., 1] / d_safe

    index = np.full((height, width), -1, np.int64)
    n_unknowns = int(mask.sum())
    index[mask] = np.arange(n_unknowns)

    rows, cols, vals, rhs = [], [], [], []
    n_equations = 0

    def add_edges(i_from, i_to, gradient, weight):
        """One equation per interior edge: w_to - w_from = gradient."""
        nonlocal n_equations
        k = np.arange(n_equations, n_equations + i_from.size)
        rows.extend([k, k])
        cols.extend([i_from, i_to])
        vals.extend([-weight, weight])
        rhs.append(gradient * weight)
        n_equations += i_from.size

    # Horizontal edges (u -> u+1): trapezoid rule on w_u across the edge.
    horizontal = mask[:, :-1] & mask[:, 1:]
    if horizontal.any():
        add_edges(index[:, :-1][horizontal], index[:, 1:][horizontal],
                  0.5 * (w_u[:, :-1][horizontal] + w_u[:, 1:][horizontal]),
                  np.minimum(trust[:, :-1][horizontal], trust[:, 1:][horizontal]))

    # Vertical edges (v -> v+1).
    vertical = mask[:-1] & mask[1:]
    if vertical.any():
        add_edges(index[:-1][vertical], index[1:][vertical],
                  0.5 * (w_v[:-1][vertical] + w_v[1:][vertical]),
                  np.minimum(trust[:-1][vertical], trust[1:][vertical]))

    w0 = np.log(np.clip(z0[mask].astype(np.float64), 1e-6, None))
    if n_equations == 0:
        out = np.zeros_like(z0, dtype=np.float32)
        out[mask] = np.exp(w0)
        return out

    a = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_equations, n_unknowns)).tocsr()
    b = np.concatenate(rhs)

    ata = (a.T @ a) + diags(np.full(n_unknowns, lam))
    atb = (a.T @ b) + lam * w0
    w, info = cg(ata, atb, x0=w0, rtol=1e-8, maxiter=5000)
    if info != 0:
        logger.warning("pointmap_splat: conjugate gradient returned info=%s", info)

    out = np.zeros_like(z0, dtype=np.float32)
    out[mask] = np.exp(w)
    logger.info(
        "pointmap_splat: integrated %d unknowns / %d gradient equations "
        "(lambda=%g), depth p1..p99 %s -> %s m",
        n_unknowns, n_equations, lam,
        np.percentile(z0[mask], [1, 99]).round(4),
        np.percentile(out[mask], [1, 99]).round(4),
    )
    return out


# ---------------------------------------------------------------------------
# placement against the SAM-3D-Body mesh
# ---------------------------------------------------------------------------
def mesh_front_depth(vertices_cam: np.ndarray, f: float, cx: float, cy: float,
                     shape: Tuple[int, int], bin_px: int) -> np.ndarray:
    """Coarse per-bin depth of the mesh's front surface, in the camera frame.

    A real z-buffer would need pyrender and a working EGL, which this
    pipeline cannot count on (see docs/docker-build-notes.md on Blackwell).
    It does not need one: an MHR mesh is ~18k vertices, and over image bins
    coarse enough to hold dozens of them the nearest projected vertex per
    bin is a perfectly good stand-in for the visible surface at the scale a
    depth *scale* fit cares about.

    Returns a (ceil(H/bin), ceil(W/bin)) array, +inf where no vertex landed.
    """
    height, width = shape
    x, y, z = np.asarray(vertices_cam, dtype=np.float64).T
    ahead = z > 1e-6
    u = f * x[ahead] / z[ahead] + cx
    v = f * y[ahead] / z[ahead] + cy
    z = z[ahead]

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    bins_y, bins_x = -(-height // bin_px), -(-width // bin_px)
    front = np.full(bins_y * bins_x, np.inf)
    if inside.any():
        flat = (v[inside] // bin_px).astype(np.int64) * bins_x \
            + (u[inside] // bin_px).astype(np.int64)
        np.minimum.at(front, flat, z[inside])
    return front.reshape(bins_y, bins_x)


def depth_scale_to_mesh(z: np.ndarray, mask: np.ndarray, front: np.ndarray,
                        bin_px: int, min_pixels: int = 16) -> Tuple[float, Dict[str, Any]]:
    """Robust scalar s such that `s * z` sits at the mesh's depth.

    Under a fixed projection, distance and metric size are the same degree
    of freedom: put the shell at the mesh's distance and it comes out at the
    mesh's height, which is what makes an orbit radius computed from the
    mesh frame the shell correctly too.

    Compared per image bin (the mesh's nearest projected vertex against the
    pointmap's median masked depth) and reduced by the median of the
    per-bin ratios, so a bin where the two disagree — hair and clothing the
    body model does not have, a limb the fit put in the wrong place — moves
    the answer by nothing.
    """
    bins_y, bins_x = front.shape
    rows, cols = np.nonzero(mask)
    flat = (rows // bin_px) * bins_x + (cols // bin_px)
    depths = z[mask].astype(np.float64)

    order = np.argsort(flat, kind="stable")
    flat_sorted, depths_sorted = flat[order], depths[order]

    candidates = np.flatnonzero(np.isfinite(front.reshape(-1)))
    starts = np.searchsorted(flat_sorted, candidates, side="left")
    ends = np.searchsorted(flat_sorted, candidates, side="right")

    ratios = []
    for candidate, start, end in zip(candidates, starts, ends):
        if end - start < min_pixels:
            continue
        median = float(np.median(depths_sorted[start:end]))
        if median > 1e-6:
            ratios.append(front.reshape(-1)[candidate] / median)

    stats: Dict[str, Any] = {
        "bins_compared": len(ratios),
        "bins_with_mesh": int(candidates.size),
    }
    if not ratios:
        raise ValueError(
            "pointmap_splat: the SAM-3D-Body mesh and the foreground matte do "
            "not overlap in a single image bin — they cannot be from the same "
            "photo, or mesh_output/image were wired to different images"
        )
    ratios = np.asarray(ratios)
    scale = float(np.median(ratios))
    stats["scale_ratio_p10_p90"] = [float(v) for v in np.percentile(ratios, [10, 90])]
    return scale, stats


def mesh_depth_prior(z: np.ndarray, mask: np.ndarray, front: np.ndarray,
                     bin_px: int) -> np.ndarray:
    """Low frequencies from the SAM-3D-Body mesh, fine structure from the
    pointmap — the fusion `depth_prior="mesh"` selects.

    **EXPERIMENTAL, and not currently recommended.** Read this before
    turning it on.

    *What prompted it.* Novel views a few degrees off the anchor put the
    hands in the wrong place: the rendered skeleton tracks the shell at the
    anchor and separates from it by ~15 degrees out. Chasing that turned up
    a real inconsistency between the shell and the body fit — projected into
    the anchor camera, the nose joint sits 29 mm and the raised right
    wrist 74 mm *in front of* the shell's own visible surface, which is
    impossible (a joint is inside the body), while the left wrist sits
    116 mm behind it. Shell coverage at those pixels is 100%, so it is not
    a sampling artefact and not the shell's missing back half.

    *What is NOT established.* Which of the two is wrong. An earlier version
    of this docstring claimed the pointmap was 2.5x too flat; that number
    compared the shell's *front surface* depth range against the range of
    *joint* depths, which is not like for like — a joint can be occluded and
    the shell has nothing behind the skin. Redone front-surface against
    front-surface, at pixel level against the mesh's own z-buffer, the shell
    is somewhat flatter and no more: depth spread 226-231 mm against the
    mesh's 307-397 mm depending on the sampling radius, with regression
    slopes anywhere from 0.31 (ordinary least squares, attenuated because
    the mesh side is noisy) to 0.89 (total least squares, which
    over-corrects). The two surfaces correlate only ~0.55, because one is a
    clothed person with hair and a baggy coat and the other is a naked body
    model — which is exactly why this comparison cannot settle the question.

    A single photograph cannot separate "the body fit put the hand too
    near" from "the depth network put the surface too far". A second real
    view can, and the native workflow has one: the back panel of the
    reference sheet.

    *What this option does anyway.* Takes the coarse depth from the mesh and
    adds back only the pointmap's fine structure:

        w0 = log(mesh_coarse) + [log(z_pointmap) - lowpass(log(z_pointmap))]

    Both terms are lowpassed on the same bin grid, so what is added back is
    strictly the structure the mesh's bins cannot represent, with no seam
    and no double-counting. Bins the mesh does not reach (hair above the
    crown, clothing past the silhouette) are filled from the nearest bin it
    does.

    *Measured effects, good and bad.* It removes every sign violation
    (right hand -112 -> +30 mm, left hand +120 -> +36 mm, all 68 joints
    p10 -115 -> +7 mm) and deepens the shell (p2..p98 spread 0.255 ->
    0.324 m), with the anchor gate still at a (1, 0) shift. But: it removes
    them partly **by construction** — it makes the shell copy the mesh,
    then the mesh's own skeleton agrees with it, which is circular. It makes
    the one independent metric here *worse*: normal agreement after
    integration goes 11.7 -> 18.4 degrees, i.e. the surface it produces fits
    the predicted normals less well than the pointmap's does. And
    `align_bin_px` (32 px) is far too coarse for a hand: fingers get dragged
    toward the depth of whatever shares their bin, and visibly deform.

    So: useful as an experiment, wrong as a default — and superseded.
    `refine_pose_to_splat` addresses the same disagreement from the other
    end, moving the *body* onto the shell instead of the shell onto the
    body, which keeps the mesh a valid body and so cannot deform a finger
    at all. Prefer that. This option is kept because it is the only lever
    that acts when no body fit is available to re-pose, and because the
    comparison is what established which direction to push.

    Returns a depth map, defined inside `mask`.
    """
    height, width = mask.shape
    coarse = front.copy()
    missing = ~np.isfinite(coarse)
    if missing.all():
        raise ValueError("pointmap_splat: the mesh covers no image bin at all")
    if missing.any():
        # Nearest defined bin, via a distance transform on the coarse grid.
        _, indices = cv2.distanceTransformWithLabels(
            missing.astype(np.uint8), cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL)
        defined = np.flatnonzero(~missing.reshape(-1))
        lookup = np.zeros(indices.max() + 1, dtype=np.int64)
        lookup[indices[~missing]] = defined
        coarse[missing] = coarse.reshape(-1)[lookup[indices[missing]]]

    def upsample(bins: np.ndarray) -> np.ndarray:
        return cv2.resize(bins.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_LINEAR)

    # The pointmap's own low frequencies, measured on the identical grid so
    # the subtraction below removes exactly what the mesh replaces.
    log_z = np.zeros_like(z, dtype=np.float64)
    log_z[mask] = np.log(np.clip(z[mask], 1e-6, None))
    bins_y, bins_x = front.shape
    rows, cols = np.nonzero(mask)
    flat = (rows // bin_px) * bins_x + (cols // bin_px)
    total = np.bincount(flat, weights=log_z[mask], minlength=bins_y * bins_x)
    count = np.bincount(flat, minlength=bins_y * bins_x)
    pointmap_coarse = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    pointmap_coarse = pointmap_coarse.reshape(bins_y, bins_x)
    empty = ~np.isfinite(pointmap_coarse)
    if empty.any():
        pointmap_coarse[empty] = np.nanmean(pointmap_coarse)

    detail = log_z - upsample(pointmap_coarse)
    prior = np.exp(upsample(np.log(coarse)) + detail)
    return np.where(mask, prior, 0.0)


def silhouette_agreement(mask: np.ndarray, vertices_cam: np.ndarray,
                         f: float, cx: float, cy: float) -> Dict[str, float]:
    """How far the mesh's projection and the matte disagree, in pixels.

    Nothing depends on this — it is the check that the assumed camera is
    the right one. The principal point is taken to be the image centre
    here, exactly as body2colmap does everywhere; if SAM-3D-Body's FOV
    estimator ever hands back an off-centre one, a systematic offset in
    these numbers is how it will show.

    **Read the height ratio, not the centroid.** On real data the centroid
    offset is dominated by something that is not misplacement: an MHR mesh
    has far more vertices per unit volume in the head and hands than in the
    legs, so its *vertex* centroid sits well above the *area* centroid of a
    matte, and the subject's clothing only widens the gap. A real run
    measured 182 px of vertical centroid offset with the projected height
    within 3% of the matte's — the height was right and the centroid was
    never going to be. The x offset is still worth reading (6 px there),
    since nothing biases it.
    """
    height, width = mask.shape
    x, y, z = np.asarray(vertices_cam, dtype=np.float64).T
    ahead = z > 1e-6
    u = f * x[ahead] / z[ahead] + cx
    v = f * y[ahead] / z[ahead] + cy

    rows, cols = np.nonzero(mask)
    return {
        "mask_centroid_px": [float(cols.mean()), float(rows.mean())],
        "mesh_centroid_px": [float(u.mean()), float(v.mean())],
        "centroid_offset_px": [float(u.mean() - cols.mean()), float(v.mean() - rows.mean())],
        "mesh_height_px": float(v.max() - v.min()),
        "mask_height_px": float(rows.max() - rows.min()),
        # The number to actually read; 1.0 means the mesh and the matte
        # agree about how big the subject is under the assumed camera.
        "height_ratio": float((v.max() - v.min()) / max(rows.max() - rows.min(), 1)),
        "image_size": [int(width), int(height)],
    }


# ---------------------------------------------------------------------------
# Gaussians
# ---------------------------------------------------------------------------
def cliff_ratio(xyz: np.ndarray, mask: np.ndarray, f: float) -> np.ndarray:
    """Per-pixel max 3D distance to a masked 4-neighbour, in pixel footprints.

    About 1.2 on a surface facing the camera, growing with incidence, and
    exploding across a genuine depth discontinuity (the chin/neck
    self-occlusion) where the pointmap interpolates a sheet of Gaussians
    through empty space. The splat analogue of a mesh edge-length cull, but
    gentler: it only has to remove primitives, never to keep a surface
    connected.
    """
    footprint = np.maximum(xyz[..., 2] / f, 1e-9)
    worst = np.zeros(mask.shape)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbour = np.roll(np.roll(xyz, dy, 0), dx, 1)
        neighbour_mask = np.roll(np.roll(mask, dy, 0), dx, 1)
        distance = np.linalg.norm(xyz - neighbour, axis=2)
        worst = np.maximum(worst, np.where(mask & neighbour_mask, distance, 0.0))
    return np.where(mask, worst / footprint, 0.0)


def mat_to_quat_wxyz(rotations: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,4) unit quaternions as (w, x, y, z).

    Branchless Shepperd: build all four candidate quaternions and pick, per
    matrix, the one whose denominator is largest. The naive w-only formula
    loses precision and sign as the trace approaches -1, which happens
    constantly here — many surface normals point almost straight back down
    -Z in world space.
    """
    m00, m01, m02 = rotations[:, 0, 0], rotations[:, 0, 1], rotations[:, 0, 2]
    m10, m11, m12 = rotations[:, 1, 0], rotations[:, 1, 1], rotations[:, 1, 2]
    m20, m21, m22 = rotations[:, 2, 0], rotations[:, 2, 1], rotations[:, 2, 2]

    # Squared components x4, one per branch; the largest is always >= 1.
    candidates = np.stack([1.0 + m00 + m11 + m22,    # 4w^2
                           1.0 + m00 - m11 - m22,    # 4x^2
                           1.0 - m00 + m11 - m22,    # 4y^2
                           1.0 - m00 - m11 + m22],   # 4z^2
                          axis=1)
    branch = np.argmax(candidates, axis=1)
    s = 2.0 * np.sqrt(np.maximum(candidates[np.arange(len(rotations)), branch], 1e-12))

    quats = np.empty((len(rotations), 4))
    for b in range(4):
        k = branch == b
        if not k.any():
            continue
        sk = s[k]
        if b == 0:
            quats[k] = np.stack([0.25 * sk, (m21[k] - m12[k]) / sk,
                                 (m02[k] - m20[k]) / sk, (m10[k] - m01[k]) / sk], 1)
        elif b == 1:
            quats[k] = np.stack([(m21[k] - m12[k]) / sk, 0.25 * sk,
                                 (m01[k] + m10[k]) / sk, (m02[k] + m20[k]) / sk], 1)
        elif b == 2:
            quats[k] = np.stack([(m02[k] - m20[k]) / sk, (m01[k] + m10[k]) / sk,
                                 0.25 * sk, (m12[k] + m21[k]) / sk], 1)
        else:
            quats[k] = np.stack([(m10[k] - m01[k]) / sk, (m02[k] + m20[k]) / sk,
                                 (m12[k] + m21[k]) / sk, 0.25 * sk], 1)

    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def _upsample(array: np.ndarray, factor: int, nearest: bool = False) -> np.ndarray:
    if factor == 1:
        return array
    height, width = array.shape[:2]
    return cv2.resize(array, (width * factor, height * factor),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def build_gaussians(xyz: np.ndarray, n_cam: np.ndarray, rgb: np.ndarray,
                    alpha: np.ndarray, mask: np.ndarray, f: float,
                    k_tangent: float = 0.4, k_normal: float = 0.15,
                    max_stretch: float = 3.0, cliff_k: float = 8.0,
                    supersample: int = 1, *,
                    pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    ) -> Dict[str, np.ndarray]:
    """Masked camera-frame points + normals -> 3DGS parameters in world coords.

    One oriented Gaussian per selected pixel: at the point, flattened into
    the tangent plane of the predicted normal, sized to cover about one
    input pixel from the source view, coloured by the photo.

    Conventions, verified in masktest against the brush fork
    (crates/brush-serde, crates/brush-cube/src/lib.rs) rather than assumed:
    `rot_*` is (w,x,y,z); `scale_*` is *log* scale, exp()'d and applied as
    `quat.to_mat3().mul_diag(scale)`, so scale index i pairs with COLUMN i
    of the rotation matrix. The normal goes in column 2, making `scale_2`
    the smallest — which is also exactly what 3DGS normal-supervision code
    assumes (shortest-scale axis = surface normal), so these files drop
    straight into such a trainer.

    SH degree 0 only: one view constrains nothing view-dependent.

    Unlike masktest this does NOT recentre on the splat's own centroid. The
    world origin has to stay where SAM-3D-Body's camera is, or the shell no
    longer lines up with the mesh, the orbit, or the anchor.

    **`pose` is the one thing that moves that origin**, and it exists for
    `pointmap_elevation_views`, which builds a shell per orbit frame and
    every one of those cameras is somewhere other than the world origin.
    It is a `(R_c2w, position)` pair in body2colmap's camera-local
    convention — `Camera.rotation` and `Camera.position` — and it is
    applied at the very END of the function, deliberately:

        means   = means @ R.T + position       # after the existing * FLIP
        frames  = R @ frames                   # BEFORE mat_to_quat_wxyz
        normals = normals @ R.T

    Everything above it — `base = k_tangent * z / f`, `cliff_ratio`, the
    incidence stretch, `view = p_cam / |p_cam|` — genuinely needs a camera
    AT THE ORIGIN looking down its own axis, so this cannot be a post-hoc
    fix-up of the returned arrays. Rotating the orthonormal frame before
    the quaternion conversion avoids quaternion composition entirely, and
    lengths are invariant under a rigid transform, so `log_scales` needs
    nothing.

    `None` — both registered steps, whose camera IS the origin — skips the
    arithmetic entirely rather than passing an identity through it. That is
    not tidiness: an identity matmul turns ~1e-4 of the floats into -0.0,
    so only the skipped branch is byte-identical to what this function
    returned before the argument existed, and that is what
    tests/test_elevation_views.py pins.
    """
    if supersample > 1:
        xyz = _upsample(xyz.astype(np.float32), supersample)
        n_cam = _upsample(n_cam.astype(np.float32), supersample)
        rgb = _upsample(rgb, supersample)
        alpha = _upsample(alpha.astype(np.float32), supersample)
        mask = _upsample(mask.astype(np.uint8), supersample, nearest=True) > 0
        f = f * supersample

    selected = mask & (alpha > 1e-3)
    if cliff_k > 0:
        selected = selected & (cliff_ratio(xyz.astype(np.float64), mask, f) <= cliff_k)
    if not selected.any():
        raise ValueError("pointmap_splat: no pixel survived selection — empty mask?")

    p_cam = xyz[selected].astype(np.float64)
    n = n_cam[selected].astype(np.float64)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    # Base radius: the world-space footprint of one input pixel at this
    # depth, so neighbours overlap by construction *from the source view*.
    z = np.clip(p_cam[:, 2], 1e-6, None)
    base = k_tangent * z / f

    # From any other view that is not enough: where the surface turns away
    # from the source camera, consecutive pixels are far apart in depth
    # while their screen footprint stays one pixel, so the discs separate
    # and the silhouette combs into stripes. Widen along the foreshortened
    # in-plane direction only, by 1/cos(incidence), capped — at a true
    # silhouette that diverges. masktest measured the cost at ~0.7 dB of
    # source-view PSNR.
    view = p_cam / np.linalg.norm(p_cam, axis=1, keepdims=True)   # camera -> point
    cos_incidence = np.abs((n * view).sum(1))
    stretch = np.clip(1.0 / np.maximum(cos_incidence, 1e-6), 1.0, max_stretch)

    # In-plane direction of that foreshortening; degenerate facing the
    # camera head-on, where the stretch is 1 and the direction is arbitrary.
    t1 = view - (view * n).sum(1, keepdims=True) * n
    fallback = np.zeros_like(t1)
    fallback[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    fallback -= (fallback * n).sum(1, keepdims=True) * n
    t1 = np.where(np.linalg.norm(t1, axis=1, keepdims=True) > 1e-6, t1, fallback)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(n, t1)

    means = p_cam * FLIP
    frames = np.stack([t1 * FLIP, t2 * FLIP, n * FLIP], axis=2)   # columns
    normals = n * FLIP
    log_scales = np.log(np.stack([base * stretch, base, k_normal * base], 1))

    if pose is not None:
        rotation, position = pose
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        position = np.asarray(position, dtype=np.float64).reshape(3)
        means = means @ rotation.T + position
        frames = rotation @ frames
        normals = normals @ rotation.T

    dropped = int((mask & (alpha > 1e-3)).sum() - selected.sum())
    logger.info(
        "pointmap_splat: %d gaussians (%d dropped at depth cliffs, cliff_k=%g), "
        "radius %.2f-%.2f mm (median %.2f), stretch median %.2f max %.2f",
        len(means), dropped, cliff_k, base.min() * 1000, base.max() * 1000,
        np.median(base) * 1000, np.median(stretch), stretch.max(),
    )

    opacity = np.clip(alpha[selected].astype(np.float64), 1e-3, 0.995)
    return {
        "means": means,
        "sh_dc": (rgb[selected].astype(np.float64) / 255.0 - 0.5) / SH_C0,
        "opacity": np.log(opacity / (1.0 - opacity)),
        "log_scales": log_scales,
        "quats": mat_to_quat_wxyz(frames),
        "normals": normals,
    }


@dataclass
class ShellResult:
    """What `PointmapSplatStep._build_shell` returns: one placed shell.

    `gaussians` and `stats` are what `run()` publishes. The rest are the
    intermediates, kept for two reasons that would otherwise each need
    their own return value:

      * `_write_debug` draws `mask`, `alpha`, `z_aligned` (the raw
        pointmap depth, placed) and `z_refined` (the integrated one,
        placed) and `n_cam`, and it stays in `run()`;
      * `pointmap_elevation_views` measures shells against each other with
        `z_refined` + `mask` (the pair residual: shell i's world means
        projected into camera j against shell j's own depth there) and
        against the mesh with `front`.

    Depths are along the source camera's OpenCV axis and are 0 outside the
    mask; `front` is `mesh_front_depth`'s per-bin grid, +inf where no
    vertex landed.
    """

    gaussians: Dict[str, np.ndarray]
    stats: Dict[str, Any]
    mask: np.ndarray
    alpha: np.ndarray
    z_aligned: np.ndarray
    z_refined: np.ndarray
    n_cam: np.ndarray
    front: np.ndarray


def write_ply(path: Path, gaussians: Dict[str, np.ndarray]) -> None:
    """Binary little-endian 3DGS PLY, in the layout brush's importer expects."""
    count = len(gaussians["means"])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Generated by b2crunner pointmap_splat\n"
        "comment SH degree: 0\n"
        "comment Vertical axis: y\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in PLY_PROPS)
        + "end_header\n"
    )
    body = np.empty((count, len(PLY_PROPS)), np.float32)
    body[:, 0:3] = gaussians["means"]
    body[:, 3:6] = gaussians["sh_dc"]
    body[:, 6] = gaussians["opacity"]
    body[:, 7:10] = gaussians["log_scales"]
    body[:, 10:14] = gaussians["quats"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(body.tobytes())


def _write_debug(directory: Path, image: np.ndarray, mask: np.ndarray,
                 alpha: np.ndarray, z_raw: np.ndarray, z_refined: np.ndarray,
                 n_cam: np.ndarray) -> None:
    """The five images that make a bad reconstruction obvious at a glance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "mask.png"), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(directory / "alpha.png"), (alpha * 255).astype(np.uint8))

    for name, depth in (("depth_raw.png", z_raw), ("depth.png", z_refined)):
        foreground = depth[mask]
        if not foreground.size:
            continue
        low, high = np.percentile(foreground, [2, 98])
        normalised = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
        coloured = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        coloured[~mask] = 40
        cv2.imwrite(str(directory / name), coloured)

    # n_cam is OpenCV-frame; visualise it the way the normal maps elsewhere
    # in this pipeline are drawn (BGR, +1 -> 255).
    visual = ((n_cam + 1) * 127.5).astype(np.uint8)
    visual[~mask] = 40
    cv2.imwrite(str(directory / "normal.png"), visual[..., ::-1])
    cv2.imwrite(str(directory / "source.png"), image)


def _write_stats(path: Path, stats: Dict[str, Any]) -> None:
    """The step's `splat_stats` as JSON, numpy folded to plain Python.

    Beside the images because a placement question is answered from the
    numbers — `source_camera`, `placement.depth_m`, `depth_alignment` —
    and the log prints a summary of them, not the record.
    """
    import json

    def jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"{type(value).__name__} is not JSON serialisable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=jsonable)


# -- a Camera, and the two routes between it and the pointmap frame ---------
def camera_pose(camera: Any) -> Tuple[np.ndarray, np.ndarray]:
    """`(R_c2w, position)` in float64, with the rotation hygiene.

    `Camera.rotation` is float32 built from cross products in
    `look_at_matrix`. Upcast once, check the handedness, and
    re-orthonormalise: `mat_to_quat_wxyz`'s final normalise would otherwise
    silently absorb a residual scale rather than complain about it.
    """
    rotation = np.asarray(camera.rotation, dtype=np.float64).reshape(3, 3)
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        raise ValueError(
            f"pointmap_splat: a camera's rotation has determinant "
            f"{determinant:.6f}, i.e. it is not a right-handed rotation. Every "
            f"Gaussian's orientation is carried through it, so this would "
            f"mirror the shell rather than move it."
        )
    u, _, vt = np.linalg.svd(rotation)
    return u @ vt, np.asarray(camera.position, dtype=np.float64).reshape(3)


def mesh_in_world(mesh_output: Dict[str, Any]) -> np.ndarray:
    """SAM-3D-Body's vertices in body2colmap's world, in float64.

    `FLIP * (vertices + cam_t)` is exactly `coordinates.sam3d_to_world` —
    its 180-degree rotation about X is this sign pattern — done in float64
    rather than the library's float32.
    """
    vertices = np.asarray(mesh_output["vertices"], dtype=np.float64)
    cam_t = np.asarray(mesh_output["cam_t"], dtype=np.float64).reshape(3)
    return (vertices + cam_t) * FLIP


def vertices_in_camera(mesh_world: np.ndarray, camera: Any) -> np.ndarray:
    """The mesh in one camera's OpenCV frame — what `_build_shell` wants.

    World -> camera-local (`R.T`) -> OpenCV (`FLIP`). **Two** FLIPs
    separated by a rotation: the outer one is in `mesh_in_world` above.
    They cancel at the identity pose, so forgetting the inner one is
    invisible to every identity-pose test and shows up on a pod as a
    nonsense depth scale — see tests/test_elevation_views.py, which checks
    a mirrored mesh is caught by `silhouette.height_ratio`.
    """
    return vertices_in_pose(mesh_world, camera_pose(camera))


def vertices_in_pose(mesh_world: np.ndarray,
                     pose: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """`vertices_in_camera` for a bare `(R_c2w, position)` pair."""
    rotation, position = pose
    return ((mesh_world - position) @ rotation) * FLIP     # (R.T @ x).T == x @ R


def refined_photo_pose(refined: Any, given: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The photograph's camera, moved the way the refinement moved the anchor.

    `given` is the anchor camera as the path built it (SAM-3D-Body's
    position, `look_at`-turned onto the orbit target); `refined` is the same
    camera after `refine_cameras`. The rigid motion between them, in world
    coordinates, is T = refined o given^-1; applied to the identity pose the
    photograph was actually taken from it gives (R_T, p_T) with

        R_T = R_f @ R_g.T
        p_T = p_f - R_T @ p_g          (= p_f when the given anchor is the origin)

    That is what the photograph's rays hang on — not `refined` itself,
    which carries the path's `look_at` tilt on top (see the class
    docstring, "The dataset's anchor camera is NOT the photograph's").
    """
    r_f, p_f = camera_pose(refined)
    r_g, p_g = camera_pose(given)
    rotation = r_f @ r_g.T
    return rotation, p_f - rotation @ p_g


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """The angle of a rotation matrix, in degrees."""
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------
class PointmapSplatStep(Step):
    """Single photo -> a Gaussian-splat shell in SAM-3D-Body's world.

    **The base class, not a registered step.** Two specializations sit at
    the bottom of this module and they are what a workflow names:
    `pointmap_splat` (a whole body, from the full frame) and
    `face_pointmap_splat` (a head, from a crop of it). See
    `_source_intrinsics` for the one thing that genuinely differs between
    them; everything else is a pair of measured defaults.

    Instantiable and complete on its own, deliberately: the full-frame
    behaviour lives here as the default rather than in a subclass, because
    a whole image is the general case and a crop is the special one. That
    is also what keeps tests/test_pointmap_splat.py exercising the base
    directly.

    inputs:  {"image": HxWx3 BGR uint8,          the photo sam3d_body was run on
              "mask": HxW float32 [0,1],         rmbg's matte for that photo
              "normal_map": HxWx3 float32,       sapiens2_lite's normals for it
              "mesh_output": dict,               sam3d_body's outputs (the
                                                 `scene` namespace: vertices,
                                                 cam_t, focal_length)
              "cameras": Optional[List[Camera]], the dataset's poses, when the
                                                 photograph's camera is one of
                                                 them rather than the origin
              "anchor_frame_index": Optional[int], which one; default 0
              "given_camera": Optional[Camera]}  that same anchor camera as the
                                                 path built it, BEFORE any
                                                 refinement (render's
                                                 `image_warp.camera`)
    outputs: {"splat_path": str,                 what render_splat/load_splat take
              "splat_scene": SplatScene,
              "splat_stats": dict}               the diagnostics; read them

    Which camera the photograph is unprojected through
    --------------------------------------------------
    Without `cameras`, SAM-3D-Body's own: the origin, looking down -Z,
    which is where `render`'s override mode puts the anchor frame. With
    them, `cameras[anchor_frame_index]` — the same photograph's rays as the
    dataset now describes them. That exists for ONE reason: `refine_cameras`
    moves the anchor. The pixels do not change, and neither does the mesh,
    so the right shell after the refinement is the photograph's pixels on
    the REFINED camera's rays at the MESH's depth — which is a rebuild, not
    a transform. Carrying the bootstrap's shell across by the anchor's own
    pose delta was tried (2026-08-31 to 2026-09-02) and measured wrong by
    50 mm: the delta a bundle adjustment leaves on a single camera is mostly
    a slide along its own viewing ray, the one direction that camera's
    frame cannot constrain, and a rigid carry moves the shell's depth with
    it while the depth was never the camera's to move — it came from the
    mesh, which stayed put, and the frames triangulated around it agree
    with the mesh, not with the slide. Read through the refined camera the
    depth is re-taken from the mesh and lands where the frames put the
    face; see tests/test_pointmap_splat.py's posed-run test.

    The dataset's anchor camera is NOT the photograph's camera
    ----------------------------------------------------------
    Even before the refinement. `render`'s override mode puts the anchor
    camera at SAM-3D-Body's position and then `look_at`s the orbit target —
    the mesh's bounding-box centre, which sits off the photograph's optical
    axis. On cyber2 that is 31 mm above it at 2.14 m: 0.83 degrees of tilt,
    17 px at the render's focal, 28 mm at the face. The anchor FRAME is
    right regardless, because `generate_firstlast` warps the photograph by
    the homography that includes that tilt. But this step unprojects the
    PHOTOGRAPH's pixels with the photograph's intrinsics, and those rays are
    in the photograph's camera — hang them on the dataset camera's rotation
    and the whole shell turns by the tilt. Run 5e2817 (2026-09-04) measured
    exactly that: the cap 28 mm above the frames' face in every view, the
    cameras themselves right.

    So the pose the rays go on is the photograph's camera moved by the
    refinement's DELTA on the anchor: `refined_photo_pose(cameras[i],
    given_camera)` = (R_f R_g^T, p_f - R_f R_g^T p_g), the world motion that
    took the given anchor to the refined one, applied to the identity pose
    the photograph was taken from. The depth is still re-read from the mesh
    through that pose; only the rotation the rays are hung on changes. The
    given anchor is `image_warp.camera`, the very camera the frame was
    warped for. Without `given_camera` the dataset camera is taken to BE the
    photograph's — right only when the path was not `look_at`-turned, and
    logged as an assumption.

    The intrinsics stay the photograph's throughout (`_source_intrinsics`):
    a dataset camera carries the RENDER's focal length, and the anchor frame
    was made to match the photograph by warping the photograph, not by
    changing what lens it was taken with.
    """

    PARAMS = (
        Param("filepath", str, REQUIRED, "The .ply to write"),
        Param("integration_lambda", float, 0.01,
              "Weight of the pointmap's data term against the normals in the "
              "depth solve. Lower trusts the normals more and drifts further "
              "from the pointmap; 0 disables integration and splats the raw "
              "pointmap depth, which under-predicts relief badly",
              minimum=0.0),
        Param("splat_scale", float, 0.4,
              "Tangential Gaussian radius as a multiple of one input pixel's "
              "world footprint (z/f). Resolution-dependent: 0.4 is the "
              "measured full-body value, 0.5 the head-crop one",
              minimum=0.01),
        Param("cliff_k", float, 8.0,
              "Drop a Gaussian whose distance to a masked 4-neighbour exceeds "
              "this many pixel footprints — the sheets the pointmap "
              "interpolates across self-occlusions. 0 disables the cull",
              minimum=0.0),
        Param("mask_threshold", float, 0.5,
              "Matte value above which a pixel is foreground", minimum=0.0, maximum=1.0),
        Param("checkpoint", str, DEFAULT_CHECKPOINT,
              "HF repo for the pointmap head; the family is 0.4b/0.8b/1b",
              advanced=True),
        Param("dtype", str, "float32", "Inference dtype for the pointmap head. "
              "bfloat16's ulp at the depth a subject sits at terraces the depth "
              "map (measured: 23.0 deg of normal disagreement against 5.1 deg "
              "at float32); float32 falls back to bfloat16 on OOM anyway",
              choices=("float32", "bfloat16"), advanced=True),
        Param("device", str, None, "Torch device; empty means cuda if available",
              advanced=True),
        Param("splat_thickness", float, 0.15,
              "Gaussian extent along the normal, as a fraction of splat_scale — "
              "a flat disc, not a sliver", advanced=True),
        Param("max_stretch", float, 3.0,
              "Cap on how far a Gaussian is widened along its foreshortened "
              "in-plane axis; closes grazing-incidence combing in profile views",
              advanced=True),
        Param("supersample", int, 1,
              "Upsample the depth/normal/colour grid by this factor before "
              "splatting: denser, smaller primitives, no extra detail",
              minimum=1, advanced=True),
        Param("fill_max_frac", float, 0.0,
              "Fill mask holes up to this fraction of the subject area. 0 (the "
              "default) fills nothing: an RMBG matte of a whole body has no "
              "holes to fill, so every one left in it is real background — at "
              "masktest's 0.05 an arm raised over the head enclosed 9741 px of "
              "it, which then hung behind the subject in every novel view",
              advanced=True),
        Param("min_component_frac", float, 0.02,
              "Keep every mask component at least this large relative to the "
              "biggest, so detached hands and feet survive", advanced=True),
        Param("close_iters", int, 4, "binary_closing iterations on the mask",
              advanced=True),
        Param("alpha_erode", int, 3,
              "Pixels of silhouette that keep the matte's soft opacity; inside "
              "that the alpha is pinned to 1", advanced=True),
        Param("grazing_floor", float, 0.05,
              "Clamp |D| in the integration to this fraction of the focal "
              "length, and downweight those edges in proportion", advanced=True),
        Param("depth_prior", str, "pointmap",
              "Where the depth solve's low frequencies come from. 'pointmap' "
              "is masktest's behaviour and the default: Sapiens' own depth, "
              "scaled onto the mesh. 'mesh' takes the coarse depth from the "
              "SAM-3D-Body mesh instead and keeps only the pointmap's fine "
              "structure — EXPERIMENTAL, and it deforms hands at the current "
              "bin size; see mesh_depth_prior's docstring before using it",
              choices=("pointmap", "mesh"), advanced=True),
        Param("align_depth", bool, True,
              "Fit a single scale putting the shell at the SAM-3D-Body mesh's "
              "distance. Off keeps the pointmap's own metric scale, which has "
              "no reason to agree with the body fit's", advanced=True),
        Param("align_bin_px", int, 32,
              "Image-bin size for that fit: big enough to hold enough mesh "
              "vertices to stand in for a z-buffer", minimum=4, advanced=True),
        Param("debug_dir", str, None,
              "Write mask / alpha / depth / normal visualisations here, plus "
              "the step's stats (splat_stats, including which camera the "
              "shell was built through) as stats.json. The workflows point "
              "it under <output_root>/debug/ so it rides into the result .zip",
              advanced=True),
    )

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._checkpoint = None

    # -- model ------------------------------------------------------------
    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoImageProcessor

        try:
            from transformers import AutoModelForPointmapEstimation
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise RuntimeError(
                "pointmap_splat needs a transformers with Sapiens2 pointmap "
                "support (AutoModelForPointmapEstimation). The normal head "
                "sapiens2_lite uses landed in the same series; verified here "
                "against transformers 5.16.1."
            ) from exc

        checkpoint = params["checkpoint"]
        self._device = params["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = AutoModelForPointmapEstimation.from_pretrained(checkpoint)
        self._model.to(self._device).eval()
        self._checkpoint = checkpoint

    def unload(self) -> None:
        self._model = None
        self._processor = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _infer_pointmap(self, image_bgr: np.ndarray, dtype_name: str) -> np.ndarray:
        """BGR uint8 -> HxWx3 metric-ish pointmap on the original pixel grid.

        `post_process_pointmap_estimation` is what does the work masktest
        did by hand: divide by the predicted `scales`, crop away the
        letterbox padding the processor added (`do_pad` is true for this
        head, false for the seg one), and resize back to the source size.
        Everything is then pixel-aligned with the matte and the normal map.
        """
        import torch
        from PIL import Image

        height, width = image_bgr.shape[:2]
        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=[rgb], return_tensors="pt").to(self._device)

        model = self._model
        if dtype_name == "bfloat16" and self._device != "cpu":
            model = model.to(torch.bfloat16)
            inputs = inputs.to(torch.bfloat16)

        if self._device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            outputs = model(**inputs)
        # Back to fp32 before the divide by `scales` and the resize, which
        # is where masktest's measured bf16 damage would otherwise compound.
        outputs.pointmaps = outputs.pointmaps.float()
        if outputs.scales is not None:
            outputs.scales = outputs.scales.float()

        result = self._processor.post_process_pointmap_estimation(
            outputs, source_sizes=[(height, width)], target_sizes=[(height, width)]
        )
        if self._device.startswith("cuda"):
            logger.info("%s: pointmap done, peak VRAM %.2f GB",
                        self.STEP_NAME or "pointmap_splat",
                        torch.cuda.max_memory_allocated() / 1e9)
        return result[0]["pointmap"].permute(1, 2, 0).float().cpu().numpy().astype(np.float32)

    def _pointmap(self, image_bgr: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        try:
            return self._infer_pointmap(image_bgr, params["dtype"])
        except torch.cuda.OutOfMemoryError:
            if params["dtype"] != "float32":
                raise
            logger.warning(
                "%s: out of memory at float32, retrying in bfloat16 "
                "(expect depth terracing of a few mm; see this module's docstring)",
                self.STEP_NAME or "pointmap_splat",
            )
            torch.cuda.empty_cache()
            return self._infer_pointmap(image_bgr, "bfloat16")

    # -- the one seam between the specializations -------------------------
    def _source_intrinsics(self, inputs: Dict[str, Any], params: Dict[str, Any],
                           width: int, height: int) -> Tuple[float, float, float]:
        """The pinhole every pixel of `inputs["image"]` is unprojected through.

        SAM-3D-Body's camera, expressed on *this image's* pixel grid. The
        whole placement argument is that a Gaussian ends up on the ray
        through the full-image pixel it came from, so a step that is handed
        a crop has to say where that crop sits — see
        `FacePointmapSplatStep._source_intrinsics`, which is the only
        override.

        Here, the image IS the full frame: the focal is the fit's own and
        the principal point is the image centre, which is what SAM-3D-Body
        assumes (MoGe-2 returns a centred principal point) and what
        body2colmap assumes for every camera it builds.
        """
        focal = float(inputs["mesh_output"]["focal_length"])
        return focal, width / 2.0, height / 2.0

    # -- which camera the photograph is unprojected through ----------------
    @staticmethod
    def _source_camera(inputs: Dict[str, Any], label: str) -> Tuple[Any, Optional[int], Any]:
        """`(dataset camera, its index, the given anchor camera or None)`, or
        `(None, None, None)` for the origin.

        `given_camera` only means anything beside a dataset camera: it is
        the pose that camera had before the refinement, and the rays are
        hung on the motion between the two (`refined_photo_pose`). On its
        own it is ignored — the origin route IS the photograph's camera.

        `cameras` is optional and empty means absent — with the refinement
        switched off a workflow still wires `dataset.cameras`, and that list
        then holds the given poses, whose anchor IS the origin camera, so
        either route gives the same shell (up to the -0.0s `build_gaussians`'
        docstring mentions). `anchor_frame_index` defaults to 0, where a
        circular path's anchor sits anyway; it is the same optional read
        `pointmap_elevation_views` makes.
        """
        cameras = inputs.get("cameras")
        if cameras is None:
            return None, None, None
        cameras = list(cameras)
        if not cameras:
            return None, None, None
        index = int(inputs.get("anchor_frame_index") or 0)
        if not 0 <= index < len(cameras):
            raise ValueError(
                f"{label}: anchor_frame_index {index} is not a frame of a "
                f"{len(cameras)}-camera path. It names the frame the photograph "
                f"was taken from — the anchor."
            )
        return cameras[index], index, inputs.get("given_camera")

    # -- the shell, with an explicit camera ------------------------------
    def _build_shell(self, image: np.ndarray, matte: np.ndarray,
                     normal_map: np.ndarray, *, focal: float, cx: float, cy: float,
                     vertices_cam: np.ndarray,
                     pose: Optional[Tuple[np.ndarray, np.ndarray]],
                     params: Dict[str, Any], label: str) -> "ShellResult":
        """One frame -> one Gaussian shell, in the world `pose` names.

        Everything from the mask clean through `build_gaussians`, with
        every frame-dependent quantity an explicit argument rather than
        something resolved by dispatch. That is the whole point of the
        extraction: `pointmap_elevation_views` builds seventeen of these,
        one per orbit camera, and "which camera is this shell in" has to be
        visible at the call site. The face splat's `_source_intrinsics`
        override is the cautionary tale — it hid which camera a crop was in
        until a 12.6-degree nod was measured off the result.

        `vertices_cam` is the mesh in THIS camera's OpenCV frame, and it is
        the caller's job because the route differs: at the origin it is
        `vertices + cam_t`, and anywhere else it is
        `FLIP * (R.T @ (sam3d_to_world(...) - position))` — two FLIPs, which
        cancel at the identity pose and so are invisible to every
        identity-pose test. See `vertices_in_camera` above.

        `pose` is `(R_c2w, position)` or None; `None` is what both
        registered steps pass and is byte-identical to what this step did
        before the argument existed (see `build_gaussians`).

        Nothing about the numerics moved out of `run()` — this is a cut, not
        a rewrite. What stayed behind is input parsing, the
        `_source_intrinsics` call, the ply write and the `SplatScene`.
        """
        height, width = image.shape[:2]
        if self._model is None or self._checkpoint != params["checkpoint"]:
            self.load(params)

        mask = clean_mask(matte, params["mask_threshold"], params["fill_max_frac"],
                          params["min_component_frac"], params["close_iters"])
        if mask.sum() < 64:
            raise ValueError(f"{label}: the foreground matte is essentially empty")
        alpha = soft_alpha(matte, mask, params["alpha_erode"])
        logger.info("%s: mask %d px (%.1f%% of frame), %d soft-edge px",
                    label, int(mask.sum()), 100 * mask.mean(),
                    int(((alpha > 0.02) & (alpha < 0.98)).sum()))

        xyz_pointmap = self._pointmap(image, params)
        z_raw = xyz_pointmap[..., 2].astype(np.float64)

        # --- 1. the network's own camera, and the surface as it saw it ----
        # The pointmap is a coherent pinhole projection in the camera the
        # network implicitly committed to (0.5 px RMS on a head crop), and
        # its depth is depth along THAT camera's axis. Everything about the
        # surface's shape — the normals' frame, the integration, the relief
        # — belongs in this camera, exactly as masktest did it.
        f_pointmap, cx_pointmap, cy_pointmap, reproj_rms = fit_intrinsics(xyz_pointmap, mask)
        logger.info(
            "%s: pointmap camera f=%.1f cx=%.1f cy=%.1f (RMS %.2f px); "
            "SAM-3D-Body's on this grid f=%.1f cx=%.1f cy=%.1f (focal ratio %.3f)",
            label, f_pointmap, cx_pointmap, cy_pointmap, reproj_rms, focal, cx, cy,
            focal / f_pointmap if f_pointmap else float("nan"),
        )
        if reproj_rms > 2.0:
            logger.warning(
                "%s: the pointmap is a poor pinhole fit (RMS %.2f px). "
                "masktest measured 3.4 px over a full body, so this is expected "
                "there; well above it means the depth map is not trustworthy",
                label, reproj_rms,
            )

        n_pointmap = camera_frame_normals(normal_map, f_pointmap, cx_pointmap, cy_pointmap)
        lam = params["integration_lambda"]
        if lam > 0:
            z_integrated = integrate_depth(z_raw, n_pointmap, mask, f_pointmap, cx_pointmap,
                                           cy_pointmap, lam=lam,
                                           grazing_floor=params["grazing_floor"])
        else:
            z_integrated = z_raw.astype(np.float32)
        surface = backproject(z_integrated.astype(np.float64), f_pointmap, cx_pointmap, cy_pointmap)
        before = normal_angle_error(backproject(z_raw, f_pointmap, cx_pointmap, cy_pointmap),
                                    n_pointmap, mask)
        after = normal_angle_error(surface, n_pointmap, mask)
        logger.info("%s: normal agreement median %.1f deg -> %.1f deg", label,
                    float(np.median(before)) if before.size else float("nan"),
                    float(np.median(after)) if after.size else float("nan"))

        # --- 2. into SAM-3D-Body's camera: rotate, then re-ray ------------
        # Same optical centre, different rays. The rotation turns the crop
        # camera's axis onto the full frame's (12.6 deg on cyber2_6f's head
        # crop; ~0 for a full-frame body shell); the depth taken AFTER it is
        # depth along the axis every camera downstream shares.
        rotation, rotation_stats = rays_rotation(mask, f_pointmap, cx_pointmap, cy_pointmap,
                                                 focal, cx, cy)
        rotated = surface @ rotation.T
        n_cam = n_pointmap @ rotation.T
        z_rotated = rotated[..., 2]

        # Scale is the one degree of freedom a single view leaves, and it is
        # taken from the mesh: the median per-bin ratio of the mesh's front
        # depth to the network's puts the surface at the mesh's distance.
        bin_px = params["align_bin_px"]
        front = mesh_front_depth(vertices_cam, focal, cx, cy, (height, width), bin_px)
        scale = 1.0
        if params["align_depth"]:
            scale, align_stats = depth_scale_to_mesh(z_rotated, mask, front, bin_px)
            stats_alignment = dict(align_stats, scale=scale)
            logger.info(
                "%s: depth scale to mesh %.4f over %d bins "
                "(p10-p90 of the per-bin ratio %.4f-%.4f)",
                label, scale, align_stats["bins_compared"],
                *align_stats["scale_ratio_p10_p90"],
            )
        else:
            stats_alignment = {"scale": 1.0}

        # The placement. Every Gaussian goes on the ray through its own
        # pixel in SAM-3D-Body's camera — that is what makes the anchor
        # frame exact — at a depth that keeps the network's SHAPE:
        #
        #     z = z_mesh + k * (z_net - z_net_ref),   k = scale * f_net / f_sam
        #
        # k is the ratio of the two cameras' pixel footprints at the two
        # distances, i.e. how much wider the surface becomes when its pixels
        # are re-read through the narrower camera at the mesh's distance.
        # Scaling the relief by the same k keeps relief-to-width what the
        # network said. Scaling it by `scale` alone — the depth ratio, which
        # is what this step did before 2026-08-30 — stretches the relief by
        # f_sam / f_net on top of that: 3.19x on cyber2_6f's face, which came
        # out 184 mm wide and 338 mm deep, its integration made WORSE by the
        # normals (36.5 -> 40.3 deg) because they were read in a frame 12.6
        # deg off the depth's.
        z_ref_net = float(np.median(z_rotated[mask]))
        z_ref_mesh = scale * z_ref_net
        width_ratio = scale * f_pointmap / focal
        z_placed = np.where(mask, z_ref_mesh + width_ratio * (z_rotated - z_ref_net), 0.0)
        z_placed_raw = np.where(
            mask, z_ref_mesh + width_ratio * ((backproject(z_raw, f_pointmap, cx_pointmap,
                                                            cy_pointmap) @ rotation.T)[..., 2]
                                               - z_ref_net), 0.0)

        stats: Dict[str, Any] = {
            "image_size": [width, height],
            "mask_pixels": int(mask.sum()),
            "pointmap_intrinsics": {
                "f": f_pointmap, "cx": cx_pointmap, "cy": cy_pointmap,
                "reproj_rms_px": reproj_rms,
            },
            "sam3d_intrinsics": {"f": focal, "cx": cx, "cy": cy},
            "focal_ratio_sam3d_over_pointmap": (focal / f_pointmap) if f_pointmap else None,
            "silhouette": silhouette_agreement(mask, vertices_cam, focal, cx, cy),
            "depth_alignment": stats_alignment,
            "placement": dict(
                rotation_stats,
                width_ratio_k=width_ratio,
                depth_m=z_ref_mesh,
                relief_mm=float(np.ptp(np.percentile(z_placed[mask], [2, 98]))) * 1000.0,
                width_mm=float(np.ptp(np.nonzero(mask)[1])) * z_ref_mesh / focal * 1000.0,
            ),
            "normal_agreement_deg": {
                "before_median": float(np.median(before)) if before.size else None,
                "after_median": float(np.median(after)) if after.size else None,
            },
        }
        logger.info(
            "%s: placed at %.3f m — rotated %.1f deg onto SAM-3D-Body's rays, width "
            "ratio k=%.3f, relief %.1f mm over %.1f mm of width",
            label, z_ref_mesh, rotation_stats["rotation_deg"], width_ratio,
            stats["placement"]["relief_mm"], stats["placement"]["width_mm"],
        )

        stats["depth_prior"] = params["depth_prior"]
        if params["depth_prior"] == "mesh":
            spread_before = float(np.ptp(np.percentile(z_placed[mask], [2, 98])))
            z_placed = mesh_depth_prior(z_placed, mask, front, bin_px)
            spread_after = float(np.ptp(np.percentile(z_placed[mask], [2, 98])))
            stats["depth_spread_m"] = {"pointmap": spread_before, "mesh_prior": spread_after}
            logger.info(
                "%s: depth prior from the mesh — p2..p98 depth "
                "spread %.3f m (pointmap) -> %.3f m (mesh)",
                label, spread_before, spread_after,
            )

        xyz = backproject(z_placed, focal, cx, cy)
        z_aligned, z_refined = z_placed_raw, z_placed     # for the debug dump

        gaussians = build_gaussians(
            xyz, n_cam, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha, mask, focal,
            k_tangent=params["splat_scale"], k_normal=params["splat_thickness"],
            max_stretch=params["max_stretch"], cliff_k=params["cliff_k"],
            supersample=params["supersample"], pose=pose,
        )

        stats["n_splats"] = int(len(gaussians["means"]))
        lo = gaussians["means"].min(0)
        hi = gaussians["means"].max(0)
        stats["world_bounds"] = [[float(v) for v in lo], [float(v) for v in hi]]
        # The pivot a view-angle cull measures around. body2colmap's
        # `OrbitPipeline.splat_view_angle_deg` uses the splat's own bbox
        # centre for exactly this, and for a head sitting well above a
        # full-body orbit target the two are not interchangeable.
        stats["world_center"] = [float(v) for v in (lo + hi) / 2.0]
        return ShellResult(gaussians=gaussians, stats=stats, mask=mask, alpha=alpha,
                           z_aligned=z_aligned, z_refined=z_refined, n_cam=n_cam,
                           front=front)

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.splat_scene import SplatScene

        label = self.STEP_NAME or type(self).__name__
        image = np.asarray(inputs["image"])
        matte = np.asarray(inputs["mask"], dtype=np.float32)
        normal_map = np.asarray(inputs["normal_map"], dtype=np.float32)
        mesh_output = inputs["mesh_output"]

        height, width = image.shape[:2]
        for name, array in (("mask", matte), ("normal_map", normal_map)):
            if array.shape[:2] != (height, width):
                raise ValueError(
                    f"{label}: '{name}' is {array.shape[1]}x{array.shape[0]} "
                    f"but 'image' is {width}x{height}. All three must be the same "
                    f"photo — this step's whole placement argument rests on it."
                )

        # SAM-3D-Body's camera on this image's pixel grid — the full frame's
        # here, a crop's in the face specialization. See _source_intrinsics.
        # These are the photograph's INTRINSICS and come from the mesh fit
        # whichever way the camera below is chosen: the pixels being
        # unprojected are the photograph's either way.
        focal, cx, cy = self._source_intrinsics(inputs, params, width, height)
        camera, index, given = self._source_camera(inputs, label)
        source_stats: Dict[str, Any]
        if camera is None:
            # The origin: SAM-3D-Body's own camera, which is where the
            # photograph was taken from until something says otherwise.
            pose = None
            vertices_cam = (np.asarray(mesh_output["vertices"], dtype=np.float64)
                            + np.asarray(mesh_output["cam_t"], dtype=np.float64).reshape(3))
            source = "SAM-3D-Body's camera at the origin"
            source_stats = {"frame_index": None, "position": [0.0, 0.0, 0.0]}
        elif given is not None:
            # The photograph's camera moved by the refinement's delta on
            # the anchor — NOT the dataset camera itself, which carries the
            # path's look_at tilt. See the class docstring; the mesh's
            # depth is re-read through the same pose.
            pose = refined_photo_pose(camera, given)
            vertices_cam = vertices_in_pose(mesh_in_world(mesh_output), pose)
            tilt = rotation_angle_deg(camera_pose(given)[0])
            delta_deg = rotation_angle_deg(pose[0])
            delta_mm = float(np.linalg.norm(pose[1] - camera_pose(given)[1])) * 1000.0
            source = (
                f"the photograph's camera moved by the anchor's (frame {index}) "
                f"refinement: {delta_deg:.3f} deg, {delta_mm:.1f} mm, to "
                f"({pose[1][0]:.4f}, {pose[1][1]:.4f}, {pose[1][2]:.4f}); the "
                f"dataset's anchor camera itself is look_at-tilted {tilt:.3f} deg "
                f"off the photograph's, which the rays are NOT turned by"
            )
            source_stats = {
                "frame_index": index, "position": [float(v) for v in pose[1]],
                "given_position": [float(v) for v in camera_pose(given)[1]],
                "refined_position": [float(v) for v in camera_pose(camera)[1]],
                "refinement_delta_deg": delta_deg, "refinement_delta_mm": delta_mm,
                "given_lookat_tilt_deg": tilt,
            }
        else:
            # A dataset camera taken to BE the photograph's: right only for
            # a path that was not look_at-turned at the anchor (a plain
            # origin camera). The workflows pass `given_camera` so this
            # branch is the fallback, and it says so.
            pose = camera_pose(camera)
            vertices_cam = vertices_in_camera(mesh_in_world(mesh_output), camera)
            source = (f"dataset camera {index} at "
                      f"({pose[1][0]:.4f}, {pose[1][1]:.4f}, {pose[1][2]:.4f}), "
                      f"taken to be the photograph's own camera (no given_camera: "
                      f"a look_at tilt at the anchor, if any, turns the shell with it)")
            source_stats = {"frame_index": index, "position": [float(v) for v in pose[1]]}
        logger.info("%s: unprojecting through %s", label, source)

        shell = self._build_shell(
            image, matte, normal_map, focal=focal, cx=cx, cy=cy,
            vertices_cam=vertices_cam, pose=pose, params=params, label=label,
        )
        gaussians, stats = shell.gaussians, shell.stats
        stats["source_camera"] = source_stats

        path = Path(params["filepath"])
        write_ply(path, gaussians)
        logger.info("%s: wrote %d gaussians to %s", label, stats["n_splats"], path)

        if params["debug_dir"]:
            _write_debug(Path(params["debug_dir"]), image, shell.mask, shell.alpha,
                         shell.z_aligned, shell.z_refined, shell.n_cam)
            _write_stats(Path(params["debug_dir"]) / "stats.json", stats)

        scene = SplatScene(
            means=gaussians["means"].astype(np.float32),
            scales=gaussians["log_scales"].astype(np.float32),
            quats=gaussians["quats"].astype(np.float32),
            opacities=gaussians["opacity"].astype(np.float32),
            sh_coeffs=gaussians["sh_dc"].astype(np.float32)[:, None, :],
            sh_degree=0,
        )
        return {"splat_path": str(path), "splat_scene": scene, "splat_stats": stats}


# ---------------------------------------------------------------------------
# The two specializations. Everything above is shared; these are the names a
# workflow writes.
# ---------------------------------------------------------------------------
@register_step("pointmap_splat")
class BodyPointmapSplatStep(PointmapSplatStep):
    """The whole subject, from the whole photograph. `PointmapSplatStep` verbatim.

    Thin on purpose. The base is already the full-frame case (see its
    docstring), so this class exists to carry the registered name and to
    state the input contract that goes with it: the matte is RMBG-2.0's,
    over the same photo `sam3d_body` was fitted to, at that photo's own
    resolution.

    It keeps the name `pointmap_splat` rather than becoming
    `body_pointmap_splat` for symmetry with `face_pointmap_splat`: the name
    is written into workflows/fast_helical_native.yaml, the BOOTSTRAPS table
    in tests/test_workflows.py, pipeline/models.py's prefetch registry,
    docs/runpod.md and two READMEs, and renaming it buys nothing but the
    symmetry.
    """


@register_step("face_pointmap_splat")
class FacePointmapSplatStep(PointmapSplatStep):
    """A head, from a crop of that same photograph.

    Same numerics, same world. The difference is that the image handed to
    it is a *cut* of the frame `sam3d_body` was fitted to, so the camera it
    unprojects through has to be SAM-3D-Body's camera re-expressed on the
    crop's pixel grid — see `_source_intrinsics`. Get that wrong and every
    Gaussian lands on the wrong ray, which is the one failure this whole
    approach has no way to absorb.

    Why a crop at all, given that it is no longer a close-up: framing.
    `crop_to_box` at `padding` 3.5 puts the head against the torso, near
    the middle of Sapiens2's field and at that network's own 768/1024,
    rather than wherever it happened to sit in a full-body frame of some
    other shape. Tightening it further does the opposite of what it looks
    like it should — see that module's docstring and the flat face cap of
    2026-09-02.

    inputs:  the base's four, plus
             {"crop_info": dict} — `crop_to_box`'s output:
                 {"box": (x0, y0, x1, y1),   in FULL-image pixels
                  "full_size": (W, H),       the frame the box indexes into
                  "crop_size": (w, h)}       what was actually emitted
    outputs: the base's three; `splat_stats` additionally carries `crop`.

    Three defaults differ from the body's, all measured:

      * `splat_scale` 0.5 rather than 0.4. The tangential radius is
        resolution-dependent, and a head crop *downsamples* into the
        network's 1024x768 where a full body upsamples out of it, so
        adjacent output pixels sit further apart in world terms and the
        primitives have to be correspondingly larger to close.
      * `fill_max_frac` 0.05 rather than 0. Hole filling is wrong for an
        RMBG matte of a body — see `fill_small_holes` — and *required* for a
        segmentation mask of a head, where eyes, nostrils and lips are
        separate classes and the mask fragments into about a dozen pieces
        without it. The merged eye/mouth hole measures 2.7% of a head mask,
        which is what 0.05 is sized against.
      * `align_bin_px` 8 rather than 32. The base's 32 is a full-frame
        number — "big enough to hold enough mesh vertices to stand in for a
        z-buffer" over a whole body. On a face crop it is far too coarse,
        and it errs in one direction: the chin overhangs the throat, so the
        nearest projected vertex in a 32 px bin reports the mesh's front
        surface too NEAR, and `depth_scale_to_mesh` pulls the splat toward
        the camera by that much.

        Measured on cyber2_6f (2026-08-30) by feeding this step's own
        estimator the mesh as its own pointmap — so a perfect estimator
        must return exactly 1.0 — with the truth from a rasterised triangle
        z-buffer of the head rather than a finer bin, which would flatter
        small bins by construction:

            bin px   32     24     16     12      8      4
            error  -15.3  -10.0   -7.2   -5.5   -3.1   -0.6  mm

        Nothing is paid for it. Over 60 Monte Carlo pointmaps carrying the
        shape error this module's own docstring measures (0.94 mm median
        after a global scale, p99 4.6 mm), the run-to-run spread is ~1 mm
        at EVERY bin size — small bins remove the bias without adding
        noise, and the ordering survives degrading the pointmap 13x. The
        two costs that could have bitten do not: at bin 8 a bin still holds
        9 mesh vertices and the vertex-based front sits 0.9 mm from the
        z-buffer's own minimum (erring far, so slightly cancelling), and
        `min_pixels` still admits 93% of the mask's bins while the usable
        count rises from 28 to 308.

        8 rather than 4 for two reasons, neither of them noise. A 4 px bin
        holds exactly 16 pixels against a `min_pixels` of 16, so a bin needs
        100% mask coverage to count — it works, but it balances on the
        threshold. And the truth it was scored against is a 2x-supersampled
        z-buffer, only ~2x finer than a 4 px bin, so the 4-vs-8 gap is
        inside what that measurement can resolve. The 8-vs-32 gap is not.

        The base default stays at 32: a full frame has thin structures
        where a small bin can hold mesh from a hand and pointmap pixels
        from a sleeve, and none of the above measures that.
    """

    PARAMS = with_defaults(
        PointmapSplatStep.PARAMS,
        splat_scale=0.5,
        fill_max_frac=0.05,
        align_bin_px=8,
    )

    def _source_intrinsics(self, inputs: Dict[str, Any], params: Dict[str, Any],
                           width: int, height: int) -> Tuple[float, float, float]:
        """SAM-3D-Body's pinhole, re-expressed on the crop's pixel grid.

        A full-image pixel `u_full` and the crop pixel `u_c` it became are
        related by `u_full = x0 + r * u_c`, with `r` the crop's resize
        factor (1 when the crop is emitted at native resolution, which is
        what `crop_to_box` does). Substituting that into the full frame's
        pinhole `u_full = f_s * X/Z + W/2` gives

            f  = f_s / r
            cx = (W/2 - x0) / r
            cy = (H/2 - y0) / r

        so unprojecting a crop pixel lands on exactly the ray through the
        full-image pixel it was cut from. This is the same relation
        body2colmap's `splat_anchor.compute_anchor_transform` derives in the
        other direction — it carries a splat built in a crop's own frame
        into the full image's — except that doing it here means the splat is
        written in the mesh's world to begin with and needs no anchoring at
        all.

        A non-uniform resize is refused rather than averaged away: it would
        need two focals, and every camera downstream carries one.

        And the frame `crop_info` indexes into is checked against the one
        `sam3d_body` published its focal for. That is the failure this
        arithmetic cannot see by itself: hand the face branch a resized copy
        of the photograph the mesh was fitted to and every term above stays
        self-consistent — the box is in that copy's pixels, the crop is
        still native, `ratio` is still 1 — while `f_s` belongs to a
        different pixel grid, so the whole splat is scaled off the head. The
        check is skipped when `mesh_output` carries no `image_size`, so a
        mesh assembled by hand (tests, an older workflow) still runs.
        """
        info = inputs["crop_info"]
        try:
            x0, y0, x1, y1 = (float(v) for v in info["box"])
            width_full, height_full = (float(v) for v in info["full_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"face_pointmap_splat: 'crop_info' must carry 'box' "
                f"(x0, y0, x1, y1 in full-image pixels) and 'full_size' "
                f"(width, height), as crop_to_box writes them. Got {info!r}"
            ) from exc

        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"face_pointmap_splat: degenerate crop box {info['box']}")

        ratio_x = (x1 - x0) / width
        ratio_y = (y1 - y0) / height
        if abs(ratio_x - ratio_y) > 1e-3 * max(ratio_x, ratio_y):
            raise ValueError(
                f"face_pointmap_splat: crop box {info['box']} against a "
                f"{width}x{height} crop implies a non-uniform resize "
                f"({ratio_x:.4f} horizontally vs {ratio_y:.4f} vertically). "
                f"The crop must keep the frame's aspect ratio — one camera "
                f"carries one focal length."
            )

        source = inputs["mesh_output"]
        frame = source.get("image_size") if hasattr(source, "get") else None
        if frame is not None:
            frame = (int(frame[0]), int(frame[1]))
            if frame != (int(width_full), int(height_full)):
                raise ValueError(
                    f"face_pointmap_splat: the crop was cut from a "
                    f"{int(width_full)}x{int(height_full)} frame, but "
                    f"sam3d_body fitted the mesh — and measured the focal "
                    f"length {float(source['focal_length']):.1f} px this "
                    f"step unprojects through — on a {frame[0]}x{frame[1]} "
                    f"one. The face branch has been handed a resized copy "
                    f"of the photograph instead of the photograph; feed it "
                    f"the same image sam3d_body read."
                )

        ratio = 0.5 * (ratio_x + ratio_y)
        focal = float(inputs["mesh_output"]["focal_length"]) / ratio
        cx = (width_full / 2.0 - x0) / ratio
        cy = (height_full / 2.0 - y0) / ratio
        logger.info(
            "face_pointmap_splat: crop (%.0f, %.0f)-(%.0f, %.0f) of %.0fx%.0f at "
            "%.3f px/px -> f=%.1f cx=%.1f cy=%.1f",
            x0, y0, x1, y1, width_full, height_full, ratio, focal, cx, cy,
        )
        return focal, cx, cy

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        result = super().run(inputs, params)
        info = inputs["crop_info"]
        result["splat_stats"]["crop"] = {
            "box": [int(v) for v in info["box"]],
            "full_size": [int(v) for v in info["full_size"]],
        }
        return result
