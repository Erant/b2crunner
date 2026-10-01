"""extend_orbit — lengthen the helical orbit with two VACE video extensions.

Three small steps around one idea. The helical re-render is 81 frames over
two turns (`render_subject`: lead-in 30 deg, 2 x 360, lead-out 90 — 840 deg
at 10.37 deg a frame), and pass 2 denoises exactly those, in distribution.
VACE can be handed a video in which some frames are INACTIVE (mask 0: real
pixels, keep them) and others REACTIVE (mask 1: generate), and the anchor
injection has used that from the start for one frame. This uses it to
extend the video temporally, twice, each time with another 81-frame pass:

  * the BEFORE pass: its last 40 frames are pass 2's first 40, inactive;
    its first 41 are new frames on the helix continued backwards, reactive;
  * the AFTER pass: its first 41 frames are pass 2's last 41, inactive; its
    last 40 are new frames on the helix continued forwards, reactive.

Each pass is the same 81 frames the model and its LoRA were calibrated at,
and the overlaps differ by one for the VAE's sake: Wan's temporal VAE
encodes frame 0 alone and then every 4 frames, so a latent frame's edge
falls after frame 1, 5, 9, ... and a mask boundary is latent-aligned only
at an index of 4k+1 — 41 either way, which is 81 - 40 for the BEFORE pass
(whose inactive block ENDS the video) and 41 for the AFTER pass (whose
inactive block STARTS it). What the overlap frames come back as is
discarded — pass 2's own frames stay — so the deliverable is trained on
41 + 81 + 40 = 162 views along a longer orbit rather than 81.

**The path.** `extend_helical_path` continues render_subject's helix at
its own angular step. The helix's rule for what lies outside its loops is
the lead-in and lead-out — flat at the bottom and the top of the elevation
band — so the extension is that rule continued: `lead_in_deg` grows by 41
steps and `lead_out_deg` by 40, and the 162-frame path is solved with the
same anchor solver render_subject used (`compute_helical_anchor_params`,
through steps/splat.py's `_anchored_path`). Because the per-frame step is
unchanged, frame `i` of the source path is frame `i + 41` of the extended
one to floating point — checked, not assumed — and the middle 81 cameras
are the dataset's own, verbatim, with the rigid motion `render_subject`
carried from the refined anchor (`_carry_anchor_refinement`) recovered
from them and applied to the new frames. 41 frames is 425 deg, not 360:
at 10.37 deg a frame a turn is 34.7 frames, and what fixes the count is
the pass length (81) minus the overlap. On a flat lead that lap and the
helix's own slow climb put 32 of the 81 new frames within 5 deg of a view
the orbit already had, so `tilt_deg` ramps them out of the band: with 10
deg the ends are at -40 / +40 and no new frame repeats one. Each pass is
rendered on a path of its own (`pass_elevation_offsets`): the ramp runs
through its inactive frames too, so the pass is one uniform helix rather
than pass 2's cameras with their flat lead-in and lead-out in the middle
of the video. That is free because the inactive frames carry the guide
render, never pass 2's pixels, and what the pass returns for them is
dropped.

**The control videos.** `assemble_extension` builds the two 81-frame
videos from a render of one splat along the two passes' own paths
(`render_splat` on `pass_cameras`, matted by rmbg and laid over 0.5 grey
exactly as pass 2's control is — mask_splat's composite), every frame of
each pass, inactive ones included: the pass sees one coherent render from
end to end, with no texture seam between real frames and a render, and the
overlap it hands back is thrown away anyway. The splat is a new `brush`
fit on pass 2's 81 denoised frames (pre-upscale), the trainer having
enforced across views a consistency the denoised frames themselves lack.
Two alternatives ran against it and were dropped on 2026-09-24 as poorer
output (the user's call): the intermediate splat (pass 1's) rendered
directly, and no guide at all (pass 2's real frames inactive, flat grey
new frames — VACE's plain extension).

**The splice.** `splice_extension` takes the new frames off each pass
(41 and 40) and puts pass 2's 81 between them, on the extended cameras,
with the anchor index moved by 41. The dataset is 81 frames until this
step: the guide training and render read it as pass 2 left it, and
`dataset.splat_path` is still the intermediate splat's.

What each pass sees is dumped to `debug/extension_input/{before,after}/`
like pass 1's control video, and the guide splat to
`debug/extension_splat.ply`.

Cost, and it is the reason this is off by default: two more full-
resolution denoises at pass 2's cost each (the resident worker serves them,
so no reload), a 162-frame splat render, plus a fourth splat training for
`retrained`; and everything after (upscale, camera refinement, the face
fits, the final training) runs on twice the frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

# How far a rebuilt source camera may sit from the dataset's live one, in
# metres, before the two paths are not the same helix. The solver is
# deterministic, so a real mismatch is a parameter that differs from
# render_subject's — a different loop count, amplitude or lead — and that
# is an error to name, not a drift to absorb.
_SAME_PATH_TOLERANCE_M = 1e-4

# Wan's temporal VAE folds 4 frames into one latent frame plus one: a pass
# has to be 4k+1 frames long, or diffusers rounds `num_frames` down with a
# warning and encodes a control video longer than the latents it makes.
_VAE_TEMPORAL = 4


def helix_step_deg(n_frames: int, n_loops: int, lead_in_deg: float, lead_out_deg: float) -> float:
    """Degrees of azimuth between consecutive frames of `OrbitPath.helical`:
    frame i sits at `start + i / n_frames * total`, so the step is
    total / n_frames (not n_frames - 1)."""
    total = float(lead_in_deg) + n_loops * 360.0 + float(lead_out_deg)
    return total / int(n_frames)


def extended_helix_params(params: Dict[str, Any], before: int, after: int) -> Dict[str, Any]:
    """render_subject's helix params with `before` frames added to the
    lead-in and `after` to the lead-out, at the source path's own step.

    The lead-in and lead-out are the helix's rule for the frames outside
    its loops — flat at -amplitude before, +amplitude after — so this is
    that rule continued rather than a new one: same radius, same target,
    same elevation band, the same angular speed."""
    step = helix_step_deg(params["n_frames"], params["n_loops"],
                          params["lead_in_deg"], params["lead_out_deg"])
    return dict(
        n_frames=int(params["n_frames"]) + before + after,
        n_loops=int(params["n_loops"]),
        amplitude_deg=float(params["amplitude_deg"]),
        lead_in_deg=float(params["lead_in_deg"]) + before * step,
        lead_out_deg=float(params["lead_out_deg"]) + after * step,
    )


def pass_elevation_offsets(before: int, overlap_before: int, overlap_after: int,
                           after: int, tilt_deg: float) -> Tuple[List[float], List[float]]:
    """Each extension pass's elevation, frame by frame, as an offset from the
    lead it continues: the BEFORE pass from the lead-in's elevation, the AFTER
    pass from the lead-out's. Each pass is one ramp at a constant rate,
    tilt/before (tilt/after) a frame, end to end: the BEFORE pass climbs
    from -tilt at its first frame through 0 at its first inactive frame
    (pass 2's first camera) and on; the AFTER pass reaches 0 at its last
    inactive frame (pass 2's last camera) and climbs to +tilt.

    Why a tilt: the flat leads alone put every new frame back on pass 2's
    band. At the defaults 32 of the 81 new frames sat within 5 deg of a view
    the orbit already had (measured on e9eb3a's cameras, 2026-09-24): each
    425 deg ring lapped its own first frames 2.6 deg apart, and ran over the
    helix's first and last third-turn, which climbs only 30 deg a turn. 10
    deg of tilt takes that to none (nearest-view median 8.5 deg, pass 2's
    own is 9.6).

    Why the inactive frames follow the ramp rather than pass 2's cameras:
    they carry the guide render, not pass 2's pixels, and what the pass
    returns for them is dropped — so they can sit anywhere, and on the ramp
    the pass is one uniform helix with no bend. On pass 2's cameras it
    would climb, stop on the lead-in or lead-out, and climb again."""
    tilt = float(tilt_deg)
    return ([tilt * (i - before) / before for i in range(before + overlap_before)],
            [tilt * (m - (overlap_after - 1)) / after for m in range(overlap_after + after)])


def _at_elevation(camera: Any, target: np.ndarray, elevation_deg: float):
    """`camera` moved to `elevation_deg` about `target`, at the same radius
    and azimuth, and turned back onto the target — the helix's own
    construction (OrbitPath.helical), at an elevation it does not make."""
    from body2colmap import coordinates
    from body2colmap.camera import Camera

    radius, azimuth, _ = coordinates.cartesian_to_spherical(
        np.asarray(camera.position, dtype=np.float64) - target)
    moved = Camera(
        focal_length=(camera.fx, camera.fy),
        image_size=(camera.width, camera.height),
        principal_point=(camera.cx, camera.cy),
        position=target + coordinates.spherical_to_cartesian(radius, azimuth, elevation_deg),
    )
    moved.look_at(target, coordinates.WorldCoordinates.UP_AXIS)
    return moved


def _elevation_deg(camera: Any, target: np.ndarray) -> float:
    from body2colmap import coordinates

    return coordinates.cartesian_to_spherical(
        np.asarray(camera.position, dtype=np.float64) - target)[2]


def new_frames(phase_frames: int, overlap: int) -> int:
    """How many frames an extension pass adds: its length less the frames
    it shares with pass 2. Refuses a pass length the VAE cannot take and an
    overlap that leaves it nothing to share or nothing to paint."""
    phase_frames, overlap = int(phase_frames), int(overlap)
    if phase_frames % _VAE_TEMPORAL != 1:
        raise ValueError(
            f"extend_orbit: phase_frames={phase_frames} is not 4k+1. Each extension pass is "
            f"one Wan video, and its temporal VAE takes 4k+1 frames; diffusers would round "
            f"the count down and encode a control video longer than its latents"
        )
    if not 1 <= overlap < phase_frames:
        raise ValueError(
            f"extend_orbit: overlap={overlap} must leave an extension pass of {phase_frames} "
            f"frames at least one frame to keep and one to paint"
        )
    return phase_frames - overlap


def latent_aligned(boundary: int) -> bool:
    """Whether a per-frame mask that changes value at frame `boundary` (the
    first frame of the second block) changes on a latent frame's edge.
    Wan's temporal VAE encodes frame 0 alone and then every 4 frames, so
    the edges fall after frames 1, 5, 9, ...: a boundary of 4k+1."""
    return int(boundary) % _VAE_TEMPORAL == 1


def _camera_template(camera: Any):
    from body2colmap.camera import Camera

    return Camera(
        focal_length=(camera.fx, camera.fy),
        image_size=(camera.width, camera.height),
        principal_point=(camera.cx, camera.cy),
    )


def _solve(helix: Dict[str, Any], extras: Dict[str, Any], template, effective_mm: float):
    from body2colmap.path import (
        OrbitPath,
        compute_helical_anchor_params,
        compute_original_camera_orbit_params,
    )

    from .splat import _anchored_path

    params = dict(helix, pattern="helical", overlap=1)
    return _anchored_path(
        pattern="helical", params=params, extras=extras, camera_template=template,
        effective_mm=effective_mm, orbit_path_cls=OrbitPath,
        circular_solver=compute_original_camera_orbit_params,
        helical_solver=compute_helical_anchor_params,
    )


def _worst_gap_m(built: List[Any], live: List[Any]) -> float:
    return max(
        float(np.linalg.norm(np.asarray(a.position, dtype=np.float64)
                             - np.asarray(b.position, dtype=np.float64)))
        for a, b in zip(built, live)
    )


@register_step("extend_helical_path")
class ExtendHelicalPathStep(Step):
    """Continue the dataset's anchored helix by whole frames either side.

    inputs:  {"cameras": List[Camera]  — the dataset's live cameras, the
              path render_subject built (and moved, if the anchor was
              refined), "extras": dict — dataset.extras, for orbit_target /
              original_focal_length / focal_length_mm}
    outputs: {"cameras": the extended path, the source cameras verbatim in
              the middle; "image_names": one per camera; "before", "after":
              the new frames each side (phase_frames less that side's
              overlap); "overlap_before", "overlap_after": as given — the
              four counts assemble_extension and splice_extension cut the
              passes by; "tilt_deg": as given, for the orbit record;
              "pass_cameras": the two passes' own paths, the BEFORE pass's
              phase_frames then the AFTER pass's — what the guide is
              rendered at (pass_elevation_offsets). Their new frames are
              `cameras`' new frames; their inactive frames sit on the ramp,
              beside pass 2's cameras rather than on them}

    The helix params here MUST be render_subject's: the source path is
    rebuilt from them and compared against the live cameras, and a mismatch
    is refused rather than absorbed (see _SAME_PATH_TOLERANCE_M).
    """

    PARAMS = (
        Param("phase_frames", int, 81,
              "Frames in each extension pass — pass 2's own length, the count the "
              "model was calibrated at. Must be 4k+1", minimum=1),
        Param("overlap_before", int, 40,
              "Pass 2's first frames that close the BEFORE pass as its inactive "
              "frames; it paints phase_frames - this new frames ahead of them at the "
              "helix's own angular step (41 at the defaults, 425 deg on "
              "render_subject's helix at 10.37 deg a frame). Latent-aligned when "
              "phase_frames - this is 4k+1", minimum=1),
        Param("overlap_after", int, 41,
              "Pass 2's last frames that open the AFTER pass as its inactive frames; "
              "it paints phase_frames - this new frames after them (40 at the "
              "defaults). Latent-aligned when this is 4k+1 — one more than the "
              "BEFORE pass's, because here the inactive block starts the video "
              "rather than ends it", minimum=1),
        Param("tilt_deg", float, 0.0,
              "How far past the helix's elevation band the outermost new frame "
              "reaches: the BEFORE pass ramps up from -amplitude - this, the AFTER "
              "pass up to +amplitude + this, each at one constant rate "
              "(pass_elevation_offsets). 0 = the flat leads continued, which repeats "
              "views the orbit already has"),
        Param("n_frames", int, 81, "render_subject's frame count", minimum=1),
        Param("n_loops", int, 2, "render_subject's turns", minimum=1),
        Param("amplitude_deg", float, 30.0, "render_subject's elevation swing"),
        Param("lead_in_deg", float, 30.0, "render_subject's lead-in"),
        Param("lead_out_deg", float, 90.0, "render_subject's lead-out"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from .pointmap_splat import refined_photo_pose, rotation_angle_deg
        from .splat import _transform_camera

        live: List[Any] = list(inputs["cameras"])
        extras: Dict[str, Any] = dict(inputs["extras"] or {})
        n = int(params["n_frames"])
        phase = int(params["phase_frames"])
        overlap_before, overlap_after = int(params["overlap_before"]), int(params["overlap_after"])
        before = new_frames(phase, overlap_before)
        after = new_frames(phase, overlap_after)

        if len(live) != n:
            raise ValueError(
                f"extend_helical_path: the dataset has {len(live)} cameras but n_frames says "
                f"{n}. These are render_subject's params and must match its output"
            )
        if max(overlap_before, overlap_after) > n:
            raise ValueError(
                f"extend_helical_path: an overlap of {max(overlap_before, overlap_after)} is "
                f"more frames than pass 2's {n}"
            )
        # The mask boundary of each pass against the VAE's latent edges. A
        # warning, not a refusal: the aligned counts are the defaults, and
        # the misaligned pair is a legitimate thing to measure against them.
        for name, boundary in (("BEFORE", before), ("AFTER", overlap_after)):
            if not latent_aligned(boundary):
                logger.warning(
                    "extend_helical_path: the %s pass's mask changes at frame %d, inside a "
                    "latent frame (the edges fall at 4k+1: %d or %d). One latent then "
                    "straddles inactive and reactive frames",
                    name, boundary, boundary - (boundary - 1) % _VAE_TEMPORAL,
                    boundary - (boundary - 1) % _VAE_TEMPORAL + _VAE_TEMPORAL,
                )

        tilt = float(params["tilt_deg"])
        if tilt < 0.0 or float(params["amplitude_deg"]) + tilt >= 85.0:
            raise ValueError(
                f"extend_helical_path: tilt_deg={tilt} must be >= 0 and keep the outermost "
                f"frames under 85 deg of elevation (amplitude {params['amplitude_deg']}); "
                f"past that look_at's up vector degenerates"
            )

        effective_mm = float(extras.get("focal_length_mm", 0.0) or 0.0)
        template = _camera_template(live[0])
        source = {key: params[key] for key in
                  ("n_frames", "n_loops", "amplitude_deg", "lead_in_deg", "lead_out_deg")}
        as_built, anchor = _solve(source, extras, template, effective_mm)
        extended_params = extended_helix_params(source, before, after)
        extended, extended_anchor = _solve(extended_params, extras, template, effective_mm)

        # The same helix, `before` frames later. Checked before the carry, so
        # what is compared is one solver's answer to two parameter sets.
        middle = extended[before:before + n]
        gap = _worst_gap_m(middle, as_built)
        if gap > _SAME_PATH_TOLERANCE_M or extended_anchor != anchor + before:
            raise ValueError(
                f"extend_helical_path: the extended path does not contain the source path — "
                f"worst camera gap {gap * 1000:.3f} mm, anchor at {extended_anchor} against "
                f"{anchor} + {before}. The extension is only exact at the source helix's own "
                f"step; check the helix params against render_subject's"
            )

        # Where render_subject left its path relative to the solver's: the
        # anchor's refinement, carried rigidly (splat.py's
        # _carry_anchor_refinement), or nothing. Recovered from the anchor
        # frame and checked on every frame, then applied to the new ones.
        rotation, translation = refined_photo_pose(live[anchor], as_built[anchor])
        carried = [_transform_camera(camera, rotation, translation) for camera in as_built]
        # Each pass's path, in the solver's frame (about the orbit target,
        # the world's up) before the carry moves it with the rest: the
        # solver's azimuths, pass_elevation_offsets' ramp from the leads.
        target = np.asarray(extras["orbit_target"], dtype=np.float64)
        lead_in, lead_out = _elevation_deg(extended[0], target), _elevation_deg(extended[-1], target)
        ramp_before, ramp_after = pass_elevation_offsets(
            before, overlap_before, overlap_after, after, tilt)
        after_start = before + n - overlap_after
        before_pass = [_at_elevation(extended[i], target, lead_in + d)
                       for i, d in enumerate(ramp_before)]
        after_pass = [_at_elevation(extended[after_start + m], target, lead_out + d)
                      for m, d in enumerate(ramp_after)]
        drift = _worst_gap_m(carried, live)
        if drift > _SAME_PATH_TOLERANCE_M:
            raise ValueError(
                f"extend_helical_path: the dataset's cameras are not render_subject's helix "
                f"moved rigidly — worst gap {drift * 1000:.3f} mm after carrying the anchor's "
                f"motion. Has something refined or replaced dataset.cameras since the "
                f"re-render? The extension has to hang on the path the frames were made on"
            )
        before_pass = [_transform_camera(c, rotation, translation) for c in before_pass]
        after_pass = [_transform_camera(c, rotation, translation) for c in after_pass]
        cameras = before_pass[:before] + live + after_pass[overlap_after:]
        image_names = [f"frame_{i + 1:05d}_.png" for i in range(len(cameras))]

        step = helix_step_deg(n, params["n_loops"], params["lead_in_deg"], params["lead_out_deg"])
        logger.info(
            "extend_helical_path: %d + %d + %d = %d cameras at %.3f deg a frame — %.1f deg "
            "ahead, %.1f deg after, tilting %.1f deg past the lead-in and lead-out "
            "elevations at the ends (two %d-frame "
            "passes sharing pass 2's first %d and last %d frames); the anchor moves from "
            "frame %d to %d; the source path carried %.3f deg / %.1f mm off the solver's",
            before, n, after, len(cameras), step, before * step, after * step, tilt,
            phase, overlap_before, overlap_after, anchor, anchor + before,
            rotation_angle_deg(rotation), float(np.linalg.norm(translation)) * 1000.0,
        )
        return {"cameras": cameras, "image_names": image_names, "before": before,
                "after": after, "overlap_before": overlap_before, "overlap_after": overlap_after,
                "tilt_deg": tilt, "pass_cameras": before_pass + after_pass}


def _phase_dataset(source, images, masks, cameras, image_names):
    """A Dataset of one extension pass's control video, for the debug dump:
    pipeline/dataset.py's on-disk layout, alpha = the VACE flag."""
    from ..dataset import Dataset

    return Dataset(
        images=list(images), image_names=list(image_names), cameras=list(cameras),
        points_3d=source.points_3d, resolution=source.resolution, masks=list(masks),
        reference_image=source.reference_image, anchor_image=source.anchor_image,
        prompt=source.prompt, extras=dict(source.extras),
    )


@register_step("assemble_extension")
class AssembleExtensionStep(Step):
    """The two control videos the extension passes are handed.

    inputs:  {"dataset": Dataset — pass 2's 81 frames as it left them (its
              resolution and metadata, for the checks and the dump),
              "pass_cameras", "image_names", "before", "after",
              "overlap_before", "overlap_after": extend_helical_path's,
              "guide_images", "guide_masks": the guide splat's render at
              `pass_cameras` and rmbg's matte of it — the BEFORE pass's
              frames then the AFTER pass's;
              optional "novel_images", "novel_masks": a second render of the
              same cameras (and its matte) that the REACTIVE frames take
              instead — the workflow renders it at fewer SH bands than the
              inactive frames' render}
    outputs: {"before_images", "before_masks": the BEFORE pass — `before`
              reactive frames then `overlap_before` inactive ones (beside
              pass 2's first); "after_images", "after_masks": the AFTER pass
              — `overlap_after` inactive frames (beside pass 2's last) then
              `after` reactive ones. Masks are the VACE per-frame flag, 0.0
              inactive / 1.0 reactive}

    Every frame is the matted render over `bg_color`, inactive ones
    included. With `novel_images` the reactive frames are that render's
    instead: the inactive frames sit beside the capture, where every SH band
    the guide carries is supported and is the colour context VACE extends
    from, while the reactive frames look from elevations the guide never saw
    and its higher bands only extrapolate there (b24be4 at tilt 40,
    2026-09-30: inactive SH 3 / reactive SH 1 — uniform SH 1 washed the
    context out, a pale leotard from above). Both passes are phase_frames long by construction (before +
    overlap_before = overlap_after + after). The dataset itself is not
    touched.
    """

    PARAMS = (
        Param("bg_color", list, [0.5, 0.5, 0.5],
              "RGB in [0,1] behind the matted guide. 0.5 is the grey every control "
              "frame in this pipeline ends on, and is zero after the [-1, 1] "
              "normalisation"),
        Param("debug_dir", str, None,
              "Where to dump the two control videos as datasets (before/ and after/, "
              "the VACE flag in the alpha), like pass 1's denoise_pass1_input/. None "
              "writes nothing", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from .mask_splat import _composite_one

        dataset = inputs["dataset"]
        source: List[np.ndarray] = list(dataset.images)
        pass_cameras = list(inputs["pass_cameras"])
        image_names = list(inputs["image_names"])
        before, after = int(inputs["before"]), int(inputs["after"])
        overlap_before, overlap_after = int(inputs["overlap_before"]), int(inputs["overlap_after"])
        n = len(source)
        total = before + n + after
        phase = before + overlap_before
        if (phase != overlap_after + after
                or not 1 <= min(overlap_before, overlap_after)
                or max(overlap_before, overlap_after) > n):
            raise ValueError(
                f"assemble_extension: before={before}, after={after}, overlap_before="
                f"{overlap_before}, overlap_after={overlap_after} on {n} frames is not what "
                f"extend_helical_path publishes (the two passes would differ in length)"
            )
        if len(pass_cameras) != 2 * phase or len(image_names) != total:
            raise ValueError(
                f"assemble_extension: {len(pass_cameras)} pass cameras / {len(image_names)} "
                f"names against two {phase}-frame passes around {n} frames — "
                f"extend_helical_path was run on a different batch than the one arriving here"
            )
        guide_images, guide_masks = inputs["guide_images"], inputs["guide_masks"]
        if len(guide_images) != 2 * phase or len(guide_masks) != 2 * phase:
            raise ValueError(
                f"assemble_extension: the guide render has {len(guide_images)} frames / "
                f"{len(guide_masks)} mattes against the two passes' {2 * phase} cameras"
            )
        novel_images, novel_masks = inputs.get("novel_images"), inputs.get("novel_masks")
        if (novel_images is None) != (novel_masks is None):
            raise ValueError("assemble_extension: novel_images and novel_masks come together or not at all")
        if novel_images is not None and (len(novel_images) != 2 * phase or len(novel_masks) != 2 * phase):
            raise ValueError(
                f"assemble_extension: the novel-view render has {len(novel_images)} frames / "
                f"{len(novel_masks)} mattes against the two passes' {2 * phase} cameras"
            )

        height, width = source[0].shape[:2]
        for frame in source:
            if tuple(frame.shape[:2]) != (height, width):
                raise ValueError("assemble_extension: the source frames are not all one size")
        bg = tuple(float(c) for c in params["bg_color"])
        video = [_composite_one(img, mask, bg) for img, mask in zip(guide_images, guide_masks)]
        if novel_images is not None:
            # The reactive frames: the BEFORE pass's first `before`, the AFTER pass's last `after`.
            for i in list(range(before)) + list(range(phase + overlap_after, 2 * phase)):
                video[i] = _composite_one(novel_images[i], novel_masks[i], bg)
        for frame in video:
            if tuple(frame.shape[:2]) != (height, width):
                raise ValueError(
                    f"assemble_extension: a guide frame is {frame.shape[1]}x{frame.shape[0]} "
                    f"against the source's {width}x{height} — render the guide at the "
                    f"dataset's resolution"
                )

        one = np.ones((height, width), dtype=np.float32)
        zero = np.zeros((height, width), dtype=np.float32)
        before_flags = [one] * before + [zero] * overlap_before
        after_flags = [zero] * overlap_after + [one] * after
        result: Dict[str, Any] = {
            "before_images": video[:phase], "before_masks": before_flags,
            "after_images": video[phase:], "after_masks": after_flags,
        }

        debug_dir = params["debug_dir"]
        if debug_dir:
            # Named as the extended dataset names the views they sit beside.
            names = {"before": image_names[:phase], "after": image_names[total - phase:]}
            for name, cut, flags in (("before", slice(0, phase), before_flags),
                                     ("after", slice(phase, 2 * phase), after_flags)):
                _phase_dataset(dataset, video[cut], flags, pass_cameras[cut], names[name]) \
                    .to_disk(Path(debug_dir) / name)
            logger.info("assemble_extension: both control videos written under %s", debug_dir)

        logger.info(
            "assemble_extension: two %d-frame passes of the guide's matted render over grey — "
            "BEFORE: %d reactive then %d inactive (beside pass 2's first %d); AFTER: %d "
            "inactive (beside pass 2's last %d) then %d reactive; reactive frames from %s",
            phase, before, overlap_before, overlap_before, overlap_after, overlap_after, after,
            "the novel-view render" if novel_images is not None else "the same render",
        )
        return result


# The colour match's guard rails. The factors measured on e9eb3a were
# 0.85-0.88; one outside this band is a broken measurement (a guide that
# missed the subject, a matte of nothing), not a colour cast to undo.
_COLOUR_FACTOR_RANGE = (0.6, 1.4)
# Pixels a frame and its guide render must share before their statistics
# mean anything, per set.
_COLOUR_MIN_PIXELS = 1000


def _lab(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _colour_stats(frames, guides, masks, erode_px: int):
    """Pooled L mean, L std and mean chroma of `frames` and of `guides` over
    the same pixels: each guide's matte, eroded so the frame's own edge (the
    subject never lands on the render to the pixel) stays out."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
    f_px, g_px = [], []
    for frame, guide, mask in zip(frames, guides, masks):
        inside = cv2.erode((np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8), kernel) > 0
        f_px.append(_lab(frame)[inside])
        g_px.append(_lab(guide)[inside])
    f, g = np.concatenate(f_px), np.concatenate(g_px)
    if len(f) < _COLOUR_MIN_PIXELS:
        return None

    def stats(lab):
        return lab[:, 0].mean(), lab[:, 0].std(), np.hypot(lab[:, 1], lab[:, 2]).mean()

    return stats(f), stats(g)


def fit_colour_match(new_frames, new_guides, new_masks,
                     ref_frames, ref_guides, ref_masks, erode_px: int = 4):
    """How to bring one extension pass's new frames to pass 2's colour.

    The pass adds its own contrast and saturation (e9eb3a: chroma x1.15-1.20,
    L spread x1.16-1.27, the same across all its new frames), and the two
    sets never share a view, so neither is compared with the other directly:
    each is compared with the guide render at its own cameras, and the
    frames the pass shared with pass 2 (`ref_*`, pass 2's frames there)
    calibrate what "matching the render" looks like. The render itself is
    flatter at the new views than at the ones its splat was fitted on, so
    matching it outright would dull the new frames. Moment matching, not a
    regression: the frames and the render never register to the pixel, and a
    least-squares fit through that misregistration undershoots (x0.71 where
    the chroma ratio says x0.85).

    Returns {"l_mean", "l_target", "l_scale", "chroma_scale"} for
    `apply_colour_match`, or None when there is not enough of the subject to
    measure."""
    new = _colour_stats(new_frames, new_guides, new_masks, erode_px)
    ref = _colour_stats(ref_frames, ref_guides, ref_masks, erode_px)
    if new is None or ref is None:
        return None
    (nf_mean, nf_std, nf_chroma), (ng_mean, ng_std, ng_chroma) = new
    (rf_mean, rf_std, rf_chroma), (rg_mean, rg_std, rg_chroma) = ref
    return {
        "l_mean": float(nf_mean),
        "l_target": float(ng_mean + (rf_mean - rg_mean)),
        "l_scale": float((rf_std / rg_std) / (nf_std / ng_std)),
        "chroma_scale": float((rf_chroma / rg_chroma) / (nf_chroma / ng_chroma)),
    }


def apply_colour_match(frame_bgr: np.ndarray, match: Dict[str, float],
                       weight: Optional[np.ndarray] = None) -> np.ndarray:
    """One frame through `fit_colour_match`'s correction, in Lab: L spread
    about the pass's mean onto the calibrated target, a and b scaled about
    neutral (saturation, not hue). `weight` (HxW in [0, 1]) blends it in, so
    the grey the pass was drawn on stays that grey for the rmbg after."""
    import cv2

    lab = _lab(frame_bgr)
    out = lab.copy()
    out[..., 0] = match["l_target"] + (lab[..., 0] - match["l_mean"]) * match["l_scale"]
    out[..., 1:] *= match["chroma_scale"]
    if weight is not None:
        w = np.clip(np.asarray(weight, dtype=np.float32), 0.0, 1.0)[..., None]
        out = lab + (out - lab) * w
    bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)


def _feathered(mask: np.ndarray, grow_px: int) -> np.ndarray:
    """A guide matte grown by `grow_px` and softened over as many, the
    weight the correction is blended in with: the subject in the new frame
    is where the render put it, give or take."""
    import cv2

    hard = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2)
    grown = cv2.dilate(hard, kernel).astype(np.float32)
    return cv2.GaussianBlur(grown, (0, 0), max(grow_px / 2.0, 0.5))


@register_step("splice_extension")
class SpliceExtensionStep(Step):
    """The extended dataset: each pass's new frames around pass 2's own.

    inputs:  {"dataset": pass 2's 81 frames, "before_denoised": the BEFORE
              pass's output, "after_denoised": the AFTER pass's, "cameras",
              "image_names", "before", "after", "overlap_before",
              "overlap_after": extend_helical_path's,
              "anchor_frame_index"?: pass 2's,
              "guide_images"?, "guide_masks"?: the guide render at the two
              passes' cameras and its matte (assemble_extension's), which
              the colour match measures against — required with it}
    outputs: {"images": before + n + after frames, "masks": the all-1.0
              VACE batch of that length (what mask_splat leaves; the next
              rmbg replaces it), "cameras", "image_names",
              "anchor_frame_index": moved by `before`}

    The overlap frames each pass returned — its version of pass 2's frames
    — are dropped: pass 2's stay, byte for byte. With `colour_match` and a
    guide, each pass's new frames are brought to pass 2's contrast and
    saturation first (`fit_colour_match`).
    """

    PARAMS = (
        Param("colour_match", bool, True,
              "Bring each pass's new frames to pass 2's contrast and saturation, "
              "measured against the guide render at both sets' cameras. The passes "
              "come back more saturated and contrasty than pass 2 (e9eb3a: chroma "
              "+15-20 %, L spread +16-27 %), the same across all their new frames"),
        Param("colour_grow_px", int, 12,
              "How far past the guide's matte the correction reaches, feathered over "
              "as many pixels; beyond it the pass's grey stays untouched",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        dataset = inputs["dataset"]
        source: List[np.ndarray] = list(dataset.images)
        before_out: List[np.ndarray] = list(inputs["before_denoised"])
        after_out: List[np.ndarray] = list(inputs["after_denoised"])
        cameras = list(inputs["cameras"])
        image_names = list(inputs["image_names"])
        before, after = int(inputs["before"]), int(inputs["after"])
        overlap_before, overlap_after = int(inputs["overlap_before"]), int(inputs["overlap_after"])
        n = len(source)
        total = before + n + after
        if len(before_out) != before + overlap_before or len(after_out) != overlap_after + after:
            raise ValueError(
                f"splice_extension: the BEFORE pass returned {len(before_out)} frames and the "
                f"AFTER pass {len(after_out)}, against {before} + {overlap_before} and "
                f"{overlap_after} + {after} — a pass did not return the batch it was handed"
            )
        if len(cameras) != total or len(image_names) != total:
            raise ValueError(
                f"splice_extension: {len(cameras)} cameras / {len(image_names)} names for "
                f"{total} frames"
            )
        height, width = source[0].shape[:2]
        for frame in before_out[:before] + after_out[overlap_after:]:
            if tuple(frame.shape[:2]) != (height, width):
                raise ValueError(
                    f"splice_extension: an extension frame is {frame.shape[1]}x{frame.shape[0]} "
                    f"against pass 2's {width}x{height}"
                )

        new_before, new_after = before_out[:before], after_out[overlap_after:]
        if params["colour_match"]:
            new_before, new_after = self._colour_match(
                inputs, params, source, new_before, new_after, overlap_before, overlap_after,
            )
        images = new_before + source + new_after
        masks = [np.ones((height, width), dtype=np.float32) for _ in images]
        result: Dict[str, Any] = {
            "images": images, "masks": masks, "cameras": cameras, "image_names": image_names,
        }
        anchor = inputs.get("anchor_frame_index")
        if anchor is not None:
            result["anchor_frame_index"] = int(anchor) + before
        logger.info(
            "splice_extension: %d frames — the BEFORE pass's first %d, pass 2's %d, the AFTER "
            "pass's last %d; the passes' versions of pass 2's frames (%d and %d) are dropped",
            total, before, n, after, overlap_before, overlap_after,
        )
        return result

    @staticmethod
    def _colour_match(inputs, params, source, new_before, new_after, overlap_before, overlap_after):
        guides, guide_masks = inputs.get("guide_images"), inputs.get("guide_masks")
        before, n, after = len(new_before), len(source), len(new_after)
        phase = before + overlap_before
        if guides is None or guide_masks is None:
            raise ValueError(
                "splice_extension: colour_match needs the guide render and its matte "
                "(assemble_extension's guide_images / guide_masks)"
            )
        if len(guides) != 2 * phase or len(guide_masks) != 2 * phase:
            raise ValueError(
                f"splice_extension: the guide render has {len(guides)} frames / "
                f"{len(guide_masks)} mattes against the two passes' {2 * phase} cameras"
            )
        # Indices into the guide render (the BEFORE pass's frames then the
        # AFTER pass's): each pass's new frames, and its inactive frames
        # paired with the pass 2 frames they sit beside. Those sit on the
        # pass's ramp, not on pass 2's cameras — within tilt_deg of them,
        # 0 at the seam — so the calibration compares moments over views a
        # few degrees apart, not the same view.
        phases = (
            ("BEFORE", new_before, range(0, before),
             list(zip(range(before, phase), range(0, overlap_before)))),
            ("AFTER", new_after, range(phase + overlap_after, 2 * phase),
             list(zip(range(phase, phase + overlap_after), range(n - overlap_after, n)))),
        )
        corrected = []
        for name, frames, new_idx, ref_pairs in phases:
            match = fit_colour_match(
                frames, [guides[i] for i in new_idx], [guide_masks[i] for i in new_idx],
                [source[j] for _, j in ref_pairs], [guides[i] for i, _ in ref_pairs],
                [guide_masks[i] for i, _ in ref_pairs],
            )
            low, high = _COLOUR_FACTOR_RANGE
            if match is None or not all(
                    low <= match[k] <= high for k in ("l_scale", "chroma_scale")):
                logger.warning(
                    "splice_extension: the %s pass's colour match is out of bounds (%s) — "
                    "its new frames go in as returned", name, match,
                )
                corrected.append(frames)
                continue
            logger.info(
                "splice_extension: %s pass colour match — L spread x%.3f, chroma x%.3f, "
                "L mean %.1f -> %.1f", name, match["l_scale"], match["chroma_scale"],
                match["l_mean"], match["l_target"],
            )
            corrected.append([
                apply_colour_match(frame, match, _feathered(guide_masks[i], params["colour_grow_px"]))
                for frame, i in zip(frames, new_idx)
            ])
        return corrected[0], corrected[1]
