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
the pass length (81) minus the overlap.

**The control videos.** `assemble_extension` builds the two 81-frame
videos, cut from a render of one splat along the whole 162-frame path
(`render_splat` on the extended cameras, matted by rmbg and laid over 0.5
grey exactly as pass 2's control is — mask_splat's composite). Which
splat is the experiment, the `extend_guide` setting:

  * `intermediate` — the splat already trained after pass 1, the one
    render_subject rendered pass 2's control from, on the longer path and
    used directly: its middle 81 frames are pass 2's control again, the
    81 outside are its novel views;
  * `retrained` — a new `brush` fit on pass 2's 81 denoised frames (pre-
    upscale), rendered on the same path: 81 frames at the cameras it was
    fitted on, 81 novel views, with the trainer having enforced across
    views a consistency the denoised frames themselves lack;
  * `none` — the baseline: the inactive frames are pass 2's own and the
    reactive ones 0.5 grey, which the pipeline's [-1, 1] normalisation
    turns into zeros in the reactive latent (diffusers
    pipeline_wan_vace.py, `prepare_video_latents`: `reactive = video *
    mask`, so a grey pixel and a masked-out one are the same value) —
    VACE's plain extension, the model continuing the orbit from the real
    frames and the prompt alone.

With a render, it fills the INACTIVE frames too (`inactive_source:
guide`, the request's "the inactive frames are the output of a brush
training"): the pass then sees one coherent render from end to end, with
no texture seam at the overlap between real frames and a render, and the
overlap it hands back is thrown away anyway. `inactive_source: frames` is
the hybrid — real frames inactive, render reactive, pass 2's own
arrangement at its anchor frame.

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
from typing import Any, Dict, List, Optional

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
              passes by}

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
        drift = _worst_gap_m(carried, live)
        if drift > _SAME_PATH_TOLERANCE_M:
            raise ValueError(
                f"extend_helical_path: the dataset's cameras are not render_subject's helix "
                f"moved rigidly — worst gap {drift * 1000:.3f} mm after carrying the anchor's "
                f"motion. Has something refined or replaced dataset.cameras since the "
                f"re-render? The extension has to hang on the path the frames were made on"
            )
        cameras = (
            [_transform_camera(camera, rotation, translation) for camera in extended[:before]]
            + live
            + [_transform_camera(camera, rotation, translation) for camera in extended[before + n:]]
        )
        image_names = [f"frame_{i + 1:05d}_.png" for i in range(len(cameras))]

        step = helix_step_deg(n, params["n_loops"], params["lead_in_deg"], params["lead_out_deg"])
        logger.info(
            "extend_helical_path: %d + %d + %d = %d cameras at %.3f deg a frame — %.1f deg "
            "on the lead-in elevation ahead, %.1f deg on the lead-out after (two %d-frame "
            "passes sharing pass 2's first %d and last %d frames); the anchor moves from "
            "frame %d to %d; the source path carried %.3f deg / %.1f mm off the solver's",
            before, n, after, len(cameras), step, before * step, after * step,
            phase, overlap_before, overlap_after, anchor, anchor + before,
            rotation_angle_deg(rotation), float(np.linalg.norm(translation)) * 1000.0,
        )
        return {"cameras": cameras, "image_names": image_names, "before": before,
                "after": after, "overlap_before": overlap_before, "overlap_after": overlap_after}


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

    inputs:  {"dataset": Dataset — pass 2's 81 frames as it left them,
              "cameras", "image_names", "before", "after",
              "overlap_before", "overlap_after": extend_helical_path's,
              "guide_images"?, "guide_masks"?: a render of the extended
              path and rmbg's matte of it, both over the WHOLE path —
              required unless `guide` is `none`, ignored then}
    outputs: {"before_images", "before_masks": the BEFORE pass — `before`
              reactive frames then `overlap_before` inactive ones (pass 2's
              first); "after_images", "after_masks": the AFTER pass —
              `overlap_after` inactive frames (pass 2's last) then `after`
              reactive ones. Masks are the VACE per-frame flag, 0.0
              inactive / 1.0 reactive}

    Both passes are phase_frames long by construction (before +
    overlap_before = overlap_after + after). The dataset itself is not
    touched.
    """

    PARAMS = (
        Param("guide", str, "none",
              "Which splat's render fills the reactive frames (`guide_images`, matted "
              "by `guide_masks` over `bg_color`, as pass 2's control is): the "
              "`intermediate` splat's or a `retrained` one's — the step only needs to "
              "know that one is expected; the workflow's gates decide which was "
              "rendered. `none`: flat `bg_color`, VACE's plain extension, the model "
              "continuing the orbit from the inactive frames alone",
              choices=("none", "intermediate", "retrained")),
        Param("inactive_source", str, "guide",
              "What the inactive frames carry with a guide (`none`: always the frames). "
              "`guide`: the render there too, so each pass sees one coherent render "
              "with no texture seam at the overlap — the inactive frames ARE the brush "
              "output, and what the pass hands back for them is discarded anyway. "
              "`frames`: pass 2's real frames inactive beside the rendered reactive "
              "ones, pass 2's own arrangement at its anchor frame",
              choices=("guide", "frames")),
        Param("bg_color", list, [0.5, 0.5, 0.5],
              "RGB in [0,1] behind the matted guide and filling an unguided "
              "extension. 0.5 is the grey every control frame in this pipeline ends "
              "on, and is zero after the [-1, 1] normalisation"),
        Param("debug_dir", str, None,
              "Where to dump the two control videos as datasets (before/ and after/, "
              "the VACE flag in the alpha), like pass 1's denoise_pass1_input/. None "
              "writes nothing", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from .mask_splat import _composite_one

        dataset = inputs["dataset"]
        source: List[np.ndarray] = list(dataset.images)
        cameras = list(inputs["cameras"])
        image_names = list(inputs["image_names"])
        before, after = int(inputs["before"]), int(inputs["after"])
        overlap_before, overlap_after = int(inputs["overlap_before"]), int(inputs["overlap_after"])
        n = len(source)
        total = before + n + after
        if len(cameras) != total or len(image_names) != total:
            raise ValueError(
                f"assemble_extension: {len(cameras)} cameras / {len(image_names)} names for "
                f"{before} + {n} + {after} = {total} frames — extend_helical_path was run on "
                f"a different batch than the one arriving here"
            )
        if (before + overlap_before != overlap_after + after
                or not 1 <= min(overlap_before, overlap_after)
                or max(overlap_before, overlap_after) > n):
            raise ValueError(
                f"assemble_extension: before={before}, after={after}, overlap_before="
                f"{overlap_before}, overlap_after={overlap_after} on {n} frames is not what "
                f"extend_helical_path publishes (the two passes would differ in length)"
            )

        mode = str(params["guide"]).strip()
        guide = mode != "none"
        inactive_source = params["inactive_source"] if guide else "frames"

        height, width = source[0].shape[:2]
        bg = tuple(float(c) for c in params["bg_color"])
        grey = np.empty((height, width, 3), dtype=np.uint8)
        # cv2 order, and int() as the renderers round it (0.5 -> 127).
        grey[:] = [int(bg[2] * 255), int(bg[1] * 255), int(bg[0] * 255)]

        if guide:
            guide_images = inputs.get("guide_images")
            guide_masks = inputs.get("guide_masks")
            if guide_images is None or guide_masks is None:
                raise ValueError(
                    f"assemble_extension: guide is {mode!r} but no guide_images / "
                    f"guide_masks arrived. The splat render and its matte are gated on "
                    f"the same setting — check extend_render_{mode} / extend_guide_masks "
                    f"ran"
                )
            if len(guide_images) != total or len(guide_masks) != total:
                raise ValueError(
                    f"assemble_extension: the guide render has {len(guide_images)} frames / "
                    f"{len(guide_masks)} mattes against the {total}-camera path"
                )
            fill = [_composite_one(img, mask, bg) for img, mask in zip(guide_images, guide_masks)]
        else:
            fill = [grey] * total

        for frame in source:
            if tuple(frame.shape[:2]) != (height, width):
                raise ValueError("assemble_extension: the source frames are not all one size")
        for frame in fill:
            if tuple(frame.shape[:2]) != (height, width):
                raise ValueError(
                    f"assemble_extension: a guide frame is {frame.shape[1]}x{frame.shape[0]} "
                    f"against the source's {width}x{height} — render the guide at the "
                    f"dataset's resolution"
                )

        # The whole extended video as each pass would see it, then cut: new
        # frames reactive, pass 2's frames inactive (their pixels, or the
        # render's in `guide` mode).
        middle = source if inactive_source == "frames" else fill[before:before + n]
        video = fill[:before] + list(middle) + fill[before + n:]
        one = np.ones((height, width), dtype=np.float32)
        zero = np.zeros((height, width), dtype=np.float32)
        flags = [one] * before + [zero] * n + [one] * after

        before_cut = slice(0, before + overlap_before)
        after_cut = slice(total - (overlap_after + after), total)
        result: Dict[str, Any] = {
            "before_images": video[before_cut], "before_masks": flags[before_cut],
            "after_images": video[after_cut], "after_masks": flags[after_cut],
        }

        debug_dir = params["debug_dir"]
        if debug_dir:
            for name, cut in (("before", before_cut), ("after", after_cut)):
                _phase_dataset(dataset, video[cut], flags[cut], cameras[cut], image_names[cut]) \
                    .to_disk(Path(debug_dir) / name)
            logger.info("assemble_extension: both control videos written under %s", debug_dir)

        logger.info(
            "assemble_extension: two %d-frame passes — BEFORE: %d reactive then %d inactive "
            "(pass 2's first %d); AFTER: %d inactive (pass 2's last %d) then %d reactive. The "
            "reactive frames are %s; the inactive ones are %s",
            before + overlap_before, before, overlap_before, overlap_before,
            overlap_after, overlap_after, after,
            f"the {mode} splat's matted render over grey" if guide else "flat grey",
            "pass 2's frames" if inactive_source == "frames" else "the render as well",
        )
        return result


@register_step("splice_extension")
class SpliceExtensionStep(Step):
    """The extended dataset: each pass's new frames around pass 2's own.

    inputs:  {"dataset": pass 2's 81 frames, "before_denoised": the BEFORE
              pass's output, "after_denoised": the AFTER pass's, "cameras",
              "image_names", "before", "after", "overlap_before",
              "overlap_after": extend_helical_path's,
              "anchor_frame_index"?: pass 2's}
    outputs: {"images": before + n + after frames, "masks": the all-1.0
              VACE batch of that length (what mask_splat leaves; the next
              rmbg replaces it), "cameras", "image_names",
              "anchor_frame_index": moved by `before`}

    The overlap frames each pass returned — its version of pass 2's frames
    — are dropped: pass 2's stay, byte for byte.
    """

    PARAMS = ()

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

        images = before_out[:before] + source + after_out[overlap_after:]
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
