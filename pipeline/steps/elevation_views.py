"""pointmap_elevation_views — supporting views off the stage-1 denoised batch.

The face splat's lever, applied to the body. `render_face_support_views` +
`select_support_views` hand `brush` a cap of renders of the photo-derived
face shell as *supporting views* — evidence the training fits where the
mask says to and ignores everywhere else. This step produces the second
set: for every Nth frame of `denoise_pass1`'s output, run Sapiens2's
pointmap head on that frame, build a Gaussian shell from it placed in that
frame's own orbit camera, and render the shell from **two** cameras — one
`elevation_deg` above the source camera's own elevation, one below, with
**no azimuth change**.

**What kind of evidence this is, said once.** The face cap injects
photographic identity that two denoise passes destroyed. These views
inject Sapiens2's *monocular depth prior*: no new appearance, since the
source frames are the training's own frames. That is the right medicine
for a circular orbit's zero parallax — where every training view shares
one great circle, the fit has nothing to triangulate and smears into
chalky zero-parallax streaks at any novel elevation — and it is also the
change most able to make a deliverable worse rather than better, which is
why it was validated against a captured dataset before landing
(docs/stage1-support-views-implementation.md; the A/B is that file's §7-E).

Only the **stage-2** training reads them. `train_final_splat` deliberately
takes no supporting views: by then the dataset is the helical re-render,
denoised a second time and upscaled, while these renders are the circular
bootstrap's at the pre-upscale resolution.

Two phases, not one loop
------------------------
Every shell is built first with the pointmap head resident, the plys are
written, the head is unloaded; only *then* is each ply rendered. Otherwise
the 6.5 GB fp32 head would sit on the card while `brush-splat-render`
(wgpu/Vulkan) is launched seventeen times — a co-residency nothing else in
this pipeline has. The plys are kept either way, so the split costs
nothing and buys a bad run that can be inspected.

Do the shells agree?
--------------------
Shell N and shell N+1 are built independently, from different frames, by a
network with its own idea of the camera. Three levels, three answers, and
the ones that are not guaranteed are the ones this step *measures*:

  * **Lateral position is exact by construction.** Every Gaussian goes on
    the ray through its own pixel of `cameras[i]`, using that camera's own
    `fx, cx, cy`. The network's fitted intrinsics only integrate the shape
    and are then rotated away by `rays_rotation`.
  * **Gross depth is one shared ruler: the mesh.** `depth_scale_to_mesh`
    puts each shell at the depth of the same 3-D body seen from its own
    camera. The shells agree coarsely because they share one body, not
    because their networks agreed.
  * **Not pinned:** that ruler is read with a view-dependent bias (what the
    naked mesh lacks — clothing in front, hair behind — pushes each shell
    toward its own camera), the network's per-frame relief, and the frames
    themselves (VACE denoised each one separately; nothing guarantees a
    sleeve or a hand is in the same place in frame 10 and frame 15).

So every shell publishes a **mesh residual** (its placed depth against the
mesh's own front surface, per bin, in mm — systematic and same-signed
across shells means estimator bias, one outlier means a bad frame) and
every adjacent pair publishes a **pair residual** (shell i's Gaussians
projected into camera j against shell j's own depth there — literally "do
these two agree"). `max_pair_residual_mm` can drop a shell that fails the
second; it is off by default because on the validation capture nothing
came within an order of magnitude of doing so.

Healthy numbers, from that capture (81 frames, 720x1280, r = 1.80 m):
`silhouette.height_ratio` 0.99-1.02 at every azimuth (this is the number
that catches a mirrored or mis-posed mesh), `depth_alignment.scale`
0.85-1.17 across the ring, pair residual median ~10 mm with per-pair
medians under ~20, mesh residual a few mm and *negative* (shell in front —
clothing and hair, the same sign as the face splat's measured -15 mm),
normals 3-8 degrees, ~2 s and 130-180 k Gaussians per shell.

VERIFICATION STATUS: the mechanism was validated end-to-end against a
captured dataset outside this repo (~/Projects/shelltest, 2026-08-31) —
the shells, their agreement, the renders and a three-arm brush A/B — but
this module has never run inside a pipeline run on a pod.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..registry import register_step
from ..step import REQUIRED, Param, with_defaults
from .pointmap_splat import FLIP, PointmapSplatStep, _write_debug, write_ply

logger = logging.getLogger(__name__)

#: Same default `render_splat` carries, and what the tests stub.
_RENDER_BINARY = "brush-splat-render"


# ---------------------------------------------------------------------------
# geometry helpers — the two routes between a Camera and the pointmap frame
# ---------------------------------------------------------------------------
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
            f"pointmap_elevation_views: a camera's rotation has determinant "
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
    nonsense depth scale — see this module's tests, which check a mirrored
    mesh is caught by `silhouette.height_ratio`.
    """
    rotation, position = camera_pose(camera)
    return ((mesh_world - position) @ rotation) * FLIP     # (R.T @ x).T == x @ R


def project_cv(points_world: np.ndarray, camera: Any) -> Tuple[np.ndarray, np.ndarray]:
    """World points -> (pixels (N,2), OpenCV depth (N,)) through `camera`."""
    rotation, position = camera_pose(camera)
    p_cv = ((np.asarray(points_world, dtype=np.float64) - position) @ rotation) * FLIP
    z = p_cv[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([camera.fx * p_cv[:, 0] / z + camera.cx,
                       camera.fy * p_cv[:, 1] / z + camera.cy], axis=1)
    return uv, z


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def mesh_residual(z_placed: np.ndarray, mask: np.ndarray, front: np.ndarray,
                  bin_px: int, min_pixels: int = 16) -> Dict[str, Any]:
    """Per-bin `median(z_placed) - mesh_front_depth`, in mm.

    Positive is the shell BEHIND the mesh. The expected sign is negative:
    the naked body model has no clothing and no hair, so the surface a
    photograph shows sits in front of it. What matters is the size and
    whether it is systematic — a few mm the same way on every shell is the
    estimator bias §6 of the plan describes, one shell out on its own is a
    bad frame.

    `front` is `mesh_front_depth`'s grid (nearest projected vertex per bin,
    +inf where none landed), so this compares like with like: the bins the
    scale was fitted over.
    """
    bins_y, bins_x = front.shape
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return {"bins": 0, "median_mm": None, "p10_p90_mm": None}

    flat = (rows // bin_px) * bins_x + (cols // bin_px)
    depths = z_placed[mask].astype(np.float64)
    order = np.argsort(flat, kind="stable")
    flat_sorted, depths_sorted = flat[order], depths[order]

    flat_front = front.reshape(-1)
    candidates = np.flatnonzero(np.isfinite(flat_front))
    starts = np.searchsorted(flat_sorted, candidates, side="left")
    ends = np.searchsorted(flat_sorted, candidates, side="right")

    deltas = [
        (float(np.median(depths_sorted[start:end])) - float(flat_front[candidate])) * 1000.0
        for candidate, start, end in zip(candidates, starts, ends)
        if end - start >= min_pixels
    ]
    if not deltas:
        return {"bins": 0, "median_mm": None, "p10_p90_mm": None}
    deltas = np.asarray(deltas)
    return {
        "bins": int(deltas.size),
        "median_mm": float(np.median(deltas)),
        "p10_p90_mm": [float(v) for v in np.percentile(deltas, [10, 90])],
    }


def pair_residual(means_i: np.ndarray, camera_j: Any, z_j: np.ndarray,
                  mask_j: np.ndarray) -> Dict[str, Any]:
    """Shell i's Gaussians seen from camera j, against shell j's own depth.

    The one measurement that is literally "do shell N and shell N+1 agree".
    Everything else compares a shell with the mesh, which both shells were
    aligned to and which therefore cannot report the disagreement the mesh
    itself induces (its view-dependent read, §6 item 1 of the plan).

    Returns medians of `|dz|` in mm over the pixels where shell i lands
    inside shell j's matte, and the signed median, which says which way.
    """
    uv, z = project_cv(means_i, camera_j)
    height, width = mask_j.shape
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    on = inside.copy()
    on[inside] = mask_j[v[inside], u[inside]]
    if not on.any():
        return {"overlap_px": 0, "abs_median_mm": None, "abs_p90_mm": None,
                "signed_median_mm": None}
    dz = (z[on] - z_j[v[on], u[on]]) * 1000.0
    return {
        "overlap_px": int(on.sum()),
        "overlap_frac": float(on.mean()),
        "abs_median_mm": float(np.median(np.abs(dz))),
        "abs_p90_mm": float(np.percentile(np.abs(dz), 90)),
        "signed_median_mm": float(np.median(dz)),
    }


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def select_indices(n_frames: int, every_n: int, anchor_index: int,
                   cameras: Sequence[Any]) -> Tuple[List[int], List[int]]:
    """Which frames become shells, and which duplicates were dropped.

    `range(phase, n, every_n)` with `phase = anchor_index % every_n`, so the
    anchor frame — the one frame of the batch that is a real photograph
    rather than a denoised drawing — is always selected whatever `every_n`
    is.

    Then a de-duplication on camera position. `render`'s `overlap` defaults
    to 1, so a circular `OrbitPath` appends `cameras[0]` again as the last
    camera: frames 0 and 80 are two different denoised frames through ONE
    camera. Two shells there would be two shells at one viewpoint, and
    their ± pair would be four renders of one view.
    """
    if every_n < 1:
        raise ValueError(f"pointmap_elevation_views: every_n must be >= 1, got {every_n}")
    phase = anchor_index % every_n
    kept: List[int] = []
    dropped: List[int] = []
    seen: List[np.ndarray] = []
    for index in range(phase, n_frames, every_n):
        position = np.asarray(cameras[index].position, dtype=np.float64).reshape(3)
        if any(float(np.linalg.norm(position - other)) < 1e-6 for other in seen):
            dropped.append(index)
            continue
        seen.append(position)
        kept.append(index)
    return kept, dropped


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------
@register_step("pointmap_elevation_views")
class PointmapElevationViewsStep(PointmapSplatStep):
    """Every Nth denoised frame -> a shell -> a +/- elevation pair of renders.

    inputs:  {"images": List[HxWx3 BGR uint8],     denoise_pass1's frames
              "masks": List[HxW float32],          rmbg's mattes for them
              "normal_maps": List[HxWx3 float32],  sapiens2_lite's normals
              "cameras": List[Camera],             the orbit these were rendered on
              "mesh_output": dict,                 the `scene` namespace
              "orbit_target": (3,),                the pivot
              "anchor_frame_index": Optional[int]} the photograph's frame
    outputs: {"images": List[HxWx3 BGR uint8],     premultiplied on black
              "masks": List[HxW float32],          the render's alpha
              "cameras": List[Camera],
              "stats": dict}                       per-shell + batch diagnostics

    The four lists are parallel and describe the same frames; the step
    raises rather than guessing if they are not.

    The images come out **premultiplied on black**, which is what
    `select_support_views` requires — it divides the colour back out to the
    straight alpha brush's masked mode wants, and that recovery is only
    valid over black. This is deliberately NOT a confidence render.

    The loop lives inside one `run()` because a workflow cannot iterate.
    """

    # The inherited shell knobs stay at the *body* defaults: these are
    # full-frame body shells, built from the same kind of frame
    # `pointmap_splat` is. `filepath` and `debug_dir` are the two that
    # change meaning — this step names one ply per shell rather than
    # writing one — so `filepath` is dropped outright rather than left as a
    # knob that does nothing, and `debug_dir` becomes a directory of
    # per-shell subdirectories.
    #
    # `align_bin_px` is the one measured departure: 16, not the base's 32.
    # Against a real triangle z-buffer of the mesh the systematic
    # shell-in-front-of-mesh bias is -11.9 mm at bin 32, -4.4 mm at bin 16,
    # and *overshoots* to +4.0 mm with a wider spread at bin 8 — the
    # thin-structure failure the plan worried about, visible on a full
    # frame. The base default stays 32 (nothing re-measured the photo
    # path) and the face's 8 stays 8.
    PARAMS = with_defaults(
        tuple(param for param in PointmapSplatStep.PARAMS if param.name != "filepath"),
        align_bin_px=16,
    ) + (
        Param("every_n", int, 5,
              "Build a shell from every Nth denoised frame, phased so the anchor "
              "frame — the one real photograph in the batch — is always among "
              "them. 81 frames at 5 is 17 shells (16 after the orbit's duplicate "
              "camera is dropped) and so 32 supporting views, about +40% on the "
              "training's view count. This is the cost knob: shells are ~2 s each "
              "to build, one brush-splat-render launch each to render, and the "
              "views they add cost brush ~5 min and ~2.5x the splat count",
              minimum=1),
        Param("elevation_deg", float, 20.0,
              "How far above and below the source camera's OWN elevation the two "
              "renders are taken. Azimuth and radius are untouched, which is the "
              "whole point: a circular orbit's zero parallax is a missing "
              "elevation, not a missing azimuth. The ring is not near zero "
              "elevation — frame 0 sits on the photographer's camera — so this is "
              "e0 +/- M and e0 is read from the source camera. The shell was "
              "measured to hold together to about +/-30 from any azimuth",
              minimum=0.0, maximum=89.0),
        Param("output_dir", str, REQUIRED,
              "Where the per-shell .ply files land, one per selected frame. Kept "
              "after the run so a bad batch can be looked at"),
        Param("render_path", str, _RENDER_BINARY,
              "The brush-splat-render binary; mirrors render_splat's param"),
        Param("max_pair_residual_mm", float, 0.0,
              "Drop a shell whose depth disagreement with BOTH its ring "
              "neighbours exceeds this many mm (median |dz|). 0 disables the "
              "check, which is the default: on the validation capture the worst "
              "per-pair median was 17-20 mm against a batch median of 10, i.e. "
              "nothing near an order of magnitude out. If you ever want it on, "
              "50 is the number that capture supports",
              minimum=0.0, advanced=True),
        Param("mask_erode_px", int, 0,
              "Erode the published alpha by this many pixels. The knob for a "
              "shell's open rim — the silhouette where a 2.5-D shell has an edge "
              "the subject does not. Off by default: the rim is ~9% of the "
              "matte within 4 px and flat in elevation (15 to 30 degrees moves it "
              "under 1%), with no visible rim wall at +/-20 from any azimuth",
              minimum=0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap import coordinates
        from body2colmap.camera import Camera

        from .splat import _rasterize

        label = self.STEP_NAME or type(self).__name__
        images = list(inputs["images"])
        masks = list(inputs["masks"])
        normal_maps = list(inputs["normal_maps"])
        cameras = list(inputs["cameras"])
        mesh_output = inputs["mesh_output"]
        target = np.asarray(inputs["orbit_target"], dtype=np.float64).reshape(3)
        anchor_index = int(inputs.get("anchor_frame_index") or 0)

        counts = {"images": len(images), "masks": len(masks),
                  "normal_maps": len(normal_maps), "cameras": len(cameras)}
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"{label}: the four parallel inputs disagree on how many frames "
                f"there are ({', '.join(f'{k}={v}' for k, v in counts.items())}). "
                f"Each shell is built from one frame's image, matte, normals and "
                f"camera, so they have to arrive together."
            )
        if not images:
            raise ValueError(f"{label}: there are no frames to build shells from")

        every_n = params["every_n"]
        elevation_deg = params["elevation_deg"]
        bin_px = params["align_bin_px"]
        output_dir = Path(params["output_dir"])
        debug_dir = Path(params["debug_dir"]) if params["debug_dir"] else None

        kept, dropped = select_indices(len(images), every_n, anchor_index, cameras)
        logger.info(
            "%s: every_n=%d phased on the anchor frame %d — %d shells%s",
            label, every_n, anchor_index, len(kept),
            f", {len(dropped)} dropped as a duplicate camera ({dropped})" if dropped else "",
        )

        mesh_world = mesh_in_world(mesh_output)

        # -- phase 1: the shells, with the pointmap head resident ---------
        built: Dict[int, Dict[str, Any]] = {}
        # Keyed by the frame index as a STRING: `stats` is a report, and a
        # report with integer keys stops being one the moment anything
        # writes it out as JSON.
        failures: Dict[str, str] = {}
        for index in kept:
            started = time.time()
            camera = cameras[index]
            focal, cx, cy = self._orbit_intrinsics(camera, images[index], index, label)
            try:
                shell = self._build_shell(
                    np.asarray(images[index]),
                    np.asarray(masks[index], dtype=np.float32),
                    np.asarray(normal_maps[index], dtype=np.float32),
                    focal=focal, cx=cx, cy=cy,
                    vertices_cam=vertices_in_camera(mesh_world, camera),
                    pose=camera_pose(camera),
                    params=params, label=f"{label}[{index}]",
                )
            except (ValueError, RuntimeError) as exc:
                # One frame the denoiser lost the subject in must not kill a
                # run forty minutes into a pod. Raise only if every shell
                # failed, below.
                failures[str(index)] = str(exc)
                logger.warning("%s: frame %d produced no shell — %s", label, index, exc)
                continue

            ply_path = output_dir / f"view_{index:05d}.ply"
            write_ply(ply_path, shell.gaussians)
            if debug_dir is not None:
                _write_debug(debug_dir / f"view_{index:05d}", np.asarray(images[index]),
                             shell.mask, shell.alpha, shell.z_aligned, shell.z_refined,
                             shell.n_cam)

            radius, azimuth, elevation = coordinates.cartesian_to_spherical(
                np.asarray(camera.position, dtype=np.float64) - target)
            stats = dict(shell.stats)
            stats["frame_index"] = index
            stats["spherical"] = {"radius_m": radius, "azimuth_deg": azimuth,
                                  "elevation_deg": elevation}
            stats["mesh_residual"] = mesh_residual(shell.z_refined, shell.mask,
                                                   shell.front, bin_px)
            stats["build_seconds"] = time.time() - started

            built[index] = {
                "camera": camera,
                "ply_path": ply_path,
                "stats": stats,
                # Only what phase 2 and the pair residual need, downcast:
                # seventeen shells' worth of float64 intermediates is most
                # of a gigabyte, and the plys on disk are the shells now.
                "means": shell.gaussians["means"].astype(np.float32),
                "z": shell.z_refined.astype(np.float32),
                "mask": shell.mask,
                "radius": radius, "azimuth": azimuth, "elevation": elevation,
            }
            logger.info(
                "%s: frame %d (az %.1f, el %.1f) — %d gaussians in %.1f s; "
                "height_ratio %.3f, depth scale %.3f, mesh residual %s, normals %.1f deg",
                label, index, azimuth, elevation, stats["n_splats"],
                stats["build_seconds"], stats["silhouette"]["height_ratio"],
                stats["depth_alignment"]["scale"],
                _mm(stats["mesh_residual"]["median_mm"]),
                stats["normal_agreement_deg"]["after_median"] or float("nan"),
            )

        self.unload()
        if not built:
            raise ValueError(
                f"{label}: every one of the {len(kept)} selected frames failed to "
                f"produce a shell. First failure: "
                f"{next(iter(failures.values()), 'none recorded')}"
            )

        # -- do they agree? -----------------------------------------------
        pairs, skipped = self._pair_residuals(built, params["max_pair_residual_mm"], label)

        # -- phase 2: the renders, with the head unloaded -----------------
        out_images: List[np.ndarray] = []
        out_masks: List[np.ndarray] = []
        out_cameras: List[Any] = []
        for index in sorted(built):
            if index in skipped:
                continue
            entry = built[index]
            camera = entry["camera"]
            view_cameras = self._elevation_pair(
                camera, target, entry["radius"], entry["azimuth"], entry["elevation"],
                elevation_deg, Camera, coordinates, label, index,
            )
            names = [f"view_{index:05d}_{suffix}" for suffix in ("up", "down")]
            # `scene=None` with a `splat_path` that exists: the ply written
            # in phase 1 IS the shell, and body2colmap's SplatRenderer never
            # touches the scene when it has a file to render (its `ply_path`
            # is documented as "rendered as-is instead of serializing").
            # Keeping seventeen shells' Gaussians in memory to hand back
            # what is already on disk is the cost this avoids.
            frames, alphas = _rasterize(
                scene=None, splat_path=str(entry["ply_path"]), cameras=view_cameras,
                image_names=names, width=camera.width, height=camera.height,
                bg_color=(0.0, 0.0, 0.0), render_path=params["render_path"],
            )
            for frame, alpha, view_camera in zip(frames, alphas, view_cameras):
                out_images.append(frame)
                out_masks.append(_erode_alpha(alpha, params["mask_erode_px"]))
                out_cameras.append(view_camera)

        stats = {
            "every_n": every_n,
            "elevation_deg": elevation_deg,
            "align_bin_px": bin_px,
            "anchor_frame_index": anchor_index,
            "selected": kept,
            "dropped_duplicate_cameras": dropped,
            "failed": failures,
            "skipped_for_pair_residual": sorted(skipped),
            "shells": [built[i]["stats"] for i in sorted(built)],
            "pairs": pairs,
            "views": len(out_images),
        }
        medians = [p["abs_median_mm"] for p in pairs if p["abs_median_mm"] is not None]
        stats["pair_residual_median_mm"] = float(np.median(medians)) if medians else None
        logger.info(
            "%s: %d shells built, %d failed, %d skipped — %d supporting views at "
            "e0 +/- %.1f deg; batch pair residual median %s",
            label, len(built), len(failures), len(skipped), len(out_images),
            elevation_deg, _mm(stats["pair_residual_median_mm"]),
        )
        return {"images": out_images, "masks": out_masks, "cameras": out_cameras,
                "stats": stats}

    # -- the pieces run() is made of --------------------------------------
    def _orbit_intrinsics(self, camera: Any, image: np.ndarray, index: int,
                          label: str) -> Tuple[float, float, float]:
        """This frame's pinhole, checked against the frame it goes with.

        Not `_source_intrinsics`: that one answers "where does this image
        sit in SAM-3D-Body's camera", which is the right question for a
        photograph and its crop and the wrong one here — these frames were
        rendered by an orbit camera that already knows its own intrinsics,
        and it is that camera the shell is placed in.

        Both checks are for silent wrongness. A non-square pixel would
        mis-place every Gaussian laterally while everything still ran, and
        a camera describing a different size than the frame it is paired
        with means the four parallel lists are not parallel after all.
        """
        if abs(camera.fx - camera.fy) > 1e-6 * max(abs(camera.fx), 1.0):
            raise ValueError(
                f"{label}: camera {index} has fx={camera.fx} fy={camera.fy}. This "
                f"step unprojects every pixel through a single focal length, so a "
                f"non-square pixel would silently mis-place the whole shell."
            )
        height, width = np.asarray(image).shape[:2]
        if (camera.width, camera.height) != (width, height):
            raise ValueError(
                f"{label}: camera {index} is {camera.width}x{camera.height} but "
                f"frame {index} is {width}x{height}. The camera and the frame have "
                f"to be the same view."
            )
        return float(camera.fx), float(camera.cx), float(camera.cy)

    def _elevation_pair(self, source: Any, target: np.ndarray, radius: float,
                        azimuth: float, elevation: float, delta_deg: float,
                        Camera: Any, coordinates: Any, label: str,
                        index: int) -> List[Any]:
        """The two cameras, and where "no azimuth change" is written down.

        Same radius, same azimuth, same intrinsics as the source view, same
        `up` as the ring (`WorldCoordinates.UP_AXIS`, which is what every
        camera on the path was built with). `elevation` is whatever the
        ring's elevation is — `override_cam_from_mesh` puts frame 0 on the
        photograph's camera, so for a standing photographer the ring sits
        at +10 to +15 degrees and the pair is e0+M / e0-M, not +M / -M.
        """
        out = []
        for sign in (+1.0, -1.0):
            new_elevation = elevation + sign * delta_deg
            if abs(new_elevation) > 89.0:
                raise ValueError(
                    f"{label}: frame {index} sits at elevation {elevation:.1f} deg, "
                    f"so +/-{delta_deg:.1f} reaches {new_elevation:.1f} — look_at "
                    f"degenerates at the pole. Lower elevation_deg."
                )
            position = target + coordinates.spherical_to_cartesian(
                radius, azimuth, new_elevation)
            camera = Camera(focal_length=(source.fx, source.fy),
                            image_size=(source.width, source.height),
                            principal_point=(source.cx, source.cy),
                            position=position)
            camera.look_at(target, coordinates.WorldCoordinates.UP_AXIS)
            out.append(camera)
        return out

    def _pair_residuals(self, built: Dict[int, Dict[str, Any]], threshold_mm: float,
                        label: str) -> Tuple[List[Dict[str, Any]], set]:
        """Each shell against the next one round the ring, both ways.

        The ring closes: the last selected shell is paired with the first,
        because on a circular orbit they are neighbours.

        `threshold_mm` drops a shell only if it disagrees with **both** of
        its neighbours. One bad pair is ambiguous — either shell could be
        the wrong one — while a shell that agrees with nobody is the one to
        remove.
        """
        order = sorted(built)
        pairs: List[Dict[str, Any]] = []
        bad: Dict[int, int] = {index: 0 for index in order}
        for position, index in enumerate(order):
            if len(order) < 2:
                break
            other = order[(position + 1) % len(order)]
            if other == index:
                break
            forward = pair_residual(built[index]["means"], built[other]["camera"],
                                    built[other]["z"], built[other]["mask"])
            backward = pair_residual(built[other]["means"], built[index]["camera"],
                                     built[index]["z"], built[index]["mask"])
            worst = max((v["abs_median_mm"] for v in (forward, backward)
                         if v["abs_median_mm"] is not None), default=None)
            pairs.append({"from": index, "to": other, "forward": forward,
                          "backward": backward, "abs_median_mm": worst})
            logger.info(
                "%s: pair (%d,%d) |dz| median %s / %s, p90 %s / %s over %d px",
                label, index, other, _mm(forward["abs_median_mm"]),
                _mm(backward["abs_median_mm"]), _mm(forward["abs_p90_mm"]),
                _mm(backward["abs_p90_mm"]), forward["overlap_px"],
            )
            if threshold_mm > 0 and worst is not None and worst > threshold_mm:
                bad[index] += 1
                bad[other] += 1

        skipped = {index for index, count in bad.items() if count >= 2}
        if skipped:
            logger.warning(
                "%s: dropping shell(s) %s — each disagrees with both ring "
                "neighbours by more than max_pair_residual_mm=%.1f mm",
                label, sorted(skipped), threshold_mm,
            )
        return pairs, skipped


def _erode_alpha(alpha: np.ndarray, erode_px: int) -> np.ndarray:
    """Pull the published alpha in from the shell's own open rim.

    A 2.5-D shell has a silhouette the subject does not: the surface simply
    stops where the source camera stopped seeing it, and at a novel
    elevation that edge is a thin wall of end-on Gaussians. Off by default
    (measured at ~9% of the matte within 4 px, and flat in elevation); this
    is the knob for a subject that misbehaves.
    """
    if erode_px <= 0:
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * erode_px + 1, 2 * erode_px + 1))
    return cv2.erode(alpha.astype(np.float32), kernel)


def _mm(value: Optional[float]) -> str:
    """A millimetre reading, or a dash — these are all optional."""
    return "n/a" if value is None else f"{value:+.1f} mm"
