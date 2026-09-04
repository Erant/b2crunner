"""Gaussian-splat steps: load_splat / save_splat / render_splat.

Native ports of `nodes/splat_loader_node.py`, `nodes/save_splat_node.py`
and `nodes/splat_render_node.py`.

`render_splat` is the missing half of this project's core loop. `brush`
trains a splat from a set of views; this step renders that trained splat
back out along a *new* camera path, so the next denoise pass has fresh
views to clean up:

    denoise -> brush (train) -> render_splat (re-render) -> denoise -> ...

Every ComfyUI pipeline YAML is built on that loop —
`workflows/api/resplat_helical.json`, `resplat_tiered.json` and
`outline.json` all call `Body2COLMAP_RenderSplat` — so without this step
`brush`'s output is a dead end.

**Rasterisation is `brush-splat-render`, not gsplat.** gsplat publishes no
wheel past torch 2.4/cu124, so on a modern stack it JIT-compiles its CUDA
kernels on first use and needs nvcc at *runtime* — the reason the image had
to ship a CUDA devel base at all. `brush-splat-render` is a standalone Rust
CLI (`crates/brush-splat-render` in the Erant/brush fork, built into
docker/Dockerfile right alongside the `brush` training binary) that loads a
trained `.ply` and rasterises an explicit camera list via the same
wgpu/Vulkan renderer `brush` itself uses. This step shells out to it the
same way `steps/brush.py` shells out to `brush`. See
`~/Projects/brush/docs/splat-render.md` for the CLI, the `cameras.json`
schema, and — the part that is easy to get wrong silently — the coordinate
conversion between body2colmap's OpenGL-convention `Camera` and brush's
OpenCV-convention one, verified there against a real gsplat oracle (mean
abs error 0.0008-0.0015 on RGB, comfortably under 1/255).

**Confidence gating (`confidence: true`) replaces `mask_splat`.** brush can
now write each Gaussian's multi-view evidence into the .ply it exports (see
steps/brush.py's `export_evidence`), and `brush-splat-render --confidence`
turns that into a per-pixel confidence `C`, gates on it, and hands back
`g = smoothstep(gate_lo, gate_hi, C)` as the frame's alpha. That changes
the output contract, which is the part to be careful with:

  * RGB is `(colour*a + cull*(1-a))*g + cull*(1-g)` — composited over the
    **cull colour** and then blended toward it by the gate. A transparent
    pixel is the cull colour, not black.
  * `bg_color`/`--background` is ignored; `cull_color` is the background.
  * The alpha is the decision, already made. Thresholding it, dilating it
    and bilateral-filtering it — `mask_splat` — is wrong rather than
    redundant afterwards: it would re-composite grey frames over black and
    smear the gate's soft edge. The shipped workflows run `mask_splat` in
    `passthrough` for exactly this reason, keeping only its other job
    (replacing the per-pixel alpha with the per-frame VACE batch).

The decision is made once, in 3-D, from what the training views actually
constrained — where the old pair guessed it per pixel per frame from
accumulated opacity. docs/spatial-reinforcement.md is the before/after.

**Keep it off for any render that feeds `select_support_views`** (the
face-view cap renders in both bootstrap workflows). That step
un-premultiplies its input and enforces premultiplied-over-black, so it
refuses a confidence render outright — loudly, which is the intended
failure. The `+splat` compositing in steps/render.py has the same
requirement and no way to violate it: `render_splat_layers` below passes
the background itself.

**Rasterisation is body2colmap's, and used not to be.** This module
carried a parallel implementation of `SplatRenderer.render_many` — its own
`Popen`, its own cameras.json writer, its own judgement of what a non-zero
exit meant — for three reasons, all of which are now gone:

  * `24fe424` stopped the library wrapping gsplat. It shells out to the
    same `brush-splat-render` binary this module was shelling out to.
  * `d0e3ada` gave it the same exit-code tolerance: a render is judged by
    the files it produced, so the known shutdown SIGSEGV — which lands
    after every frame is on disk — is a warning and not a failure. Its
    message is better than the one that was here, naming the missing files
    and decoding the signal number.
  * The three seams `_rasterize` now uses were added for this: `on_fault`,
    `on_output` and `ply_path`. See below.

**What is left here is the three things that are genuinely this project's.**

*Where a crash report goes.* Everything the binary is handed lives in a
temp directory `render_many` deletes on the way out, exception or not —
which is how one brush-splat-render crash on a pod ended up with nothing
to diagnose. `on_fault` is called while that directory still exists;
`_save_render_crashlog` copies the camera list, a per-frame manifest, the
argv, the output and the Vulkan/driver environment into
`paths.crash_dir()` before it goes. It fires for a render that lost frames
*and* for one that wrote everything and then died anyway, because the
second is only probably the known crash.

*Where the log goes.* `on_output` gets each line as the binary writes it,
and it is handed a `proc.OutputRelay` — the same throttled sink
`stream_command` uses for `brush`, so an 81-frame render reports progress
into the pipeline log instead of going quiet until it is done.

*What a frame is called.* The renderer names its files `f00000.png`;
everything else in a run calls them `frame_00001_.png`. The translation is
this module's, and the crash report reads in the run's terms because of it.

`ply_path` is the fourth seam and needs no policy: a trained splat is
hundreds of megabytes and this step usually follows the training that
wrote it, so an existing .ply is rendered where it lies rather than
serialized back out of `SplatScene`.

`render_splat_layers` below is what steps/render.py's `...+splat` modes
call to get their overlay, and goes through `_rasterize` so it inherits
all of the above.

**Verification status.** The camera-path half — which is where all the
subtle behaviour lives — is verified locally against `cyber_6f`'s real
recorded metadata (see tests/test_splat.py), including that the anchored
override path puts a camera exactly on the recorded `anchor_position`.
PLY load/save is verified by round-trip. `brush-splat-render` itself is
verified against gsplat as described above, but that verification ran
outside this pipeline (a manual comparison against local data); running
*through* this step, end to end, on this box's GPU is still pending — see
docs/docker-build-notes.md. The crash and argv paths are covered against a
stub binary (tests/test_render_exit.py, tests/test_splat.py), which is the
only way to test a crash on purpose.

**Focal-length inheritance** is the one piece here that is easy to get
wrong and silent when wrong. The resolution order is: the step's own
`focal_length_mm` param, else the source dataset's recorded
`focal_length_mm`, else auto. In override mode a framed focal length is
*required* — the anchor frame has to match the framing the mesh render
used, or the reference image warped during that render no longer lines up
with the frame it gets injected into. Passing an explicit
`focal_length_mm` that disagrees with the dataset's is therefore warned
about rather than silently honoured.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..proc import (
    OutputRelay,
    crashlog_note,
    describe_path,
    save_crashlog,
)
from ..registry import register_step
from ..step import REQUIRED, Param, Step
from .backdrop import BACKGROUND_PARAMS, build_background, composite_bgr

logger = logging.getLogger(__name__)

_FULL_FRAME_SENSOR_WIDTH_MM = 36.0

# How many of the frames that did land get copied into a crash directory.
# The last few written are the crash's neighbourhood — the frame it died on
# and the ones just before it — and a full 100-frame orbit at 720x1280 is
# not something to copy onto the volume every time a render fails.
_CRASH_FRAMES_KEPT = 4

# The rasteriser, resolved on PATH unless a step names one explicitly.
# The image builds it to /usr/local/bin (see docker/Dockerfile).
_RENDER_BINARY = "brush-splat-render"

# Rebuilt by every render; never inherited from the source dataset's extras.
_REBUILT_KEYS = {
    "cameras", "image_names", "points_3d", "resolution",
    "focal_length_mm", "splat_path",
}
# A new path invalidates an inherited anchor: the orbit is different, so the
# recorded position need not be on it. Reused cameras keep theirs, which is
# exactly why the position (not the frame index) is the durable key.
_ANCHOR_KEYS = {"anchor_frame_index", "anchor_position"}


def _mm_to_pixels(focal_length_mm: float, image_width: int) -> float:
    return (focal_length_mm / _FULL_FRAME_SENSOR_WIDTH_MM) * image_width


@register_step("load_splat")
class LoadSplatStep(Step):
    """Load a trained Gaussian splat from a PLY file.

    inputs:  {} (or {"splat_path": str} to override the param)
    outputs: {"splat_scene": SplatScene, "splat_path": str}
    """

    PARAMS = (
        Param("filepath", str, None,
              "The .ply to load. A `splat_path` input wins over it"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.splat_scene import SplatScene

        filepath = inputs.get("splat_path") or params["filepath"]
        if not filepath:
            raise ValueError("load_splat needs a 'filepath' param or 'splat_path' input")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Splat PLY not found: {path}")

        scene = SplatScene.from_ply(str(path))
        logger.info(
            "load_splat: %s (%d Gaussians, SH degree %d)",
            path, len(scene), scene.sh_degree,
        )
        return {"splat_scene": scene, "splat_path": str(path)}


@register_step("save_splat")
class SaveSplatStep(Step):
    """Write a Gaussian splat scene to a PLY file.

    inputs:  {"splat_scene": SplatScene}
    outputs: {"splat_path": str}
    """

    PARAMS = (
        Param("filepath", str, REQUIRED, "The .ply to write"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        scene = inputs["splat_scene"]
        filepath = params["filepath"]

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene.to_ply(str(path))
        logger.info("save_splat: wrote %d Gaussians to %s", len(scene), path)
        return {"splat_path": str(path)}


@register_step("render_splat")
class RenderSplatStep(Step):
    """Render a trained Gaussian splat along a camera path.

    inputs:  {"splat_scene": SplatScene} or {"splat_path": str},
             plus optionally {"dataset": Dataset} — the source dataset,
             used for framing bounds, focal-length inheritance, camera
             reuse, point-cloud preservation and extras pass-through
    outputs: same shape as steps/render.py — {"images", "masks",
             "cameras", "image_names", "points_3d", "resolution",
             "focal_length_mm"} plus "anchor_position" /
             "anchor_frame_index" when the path was anchored, and any
             non-rebuilt extras carried through from the source dataset

    With no `pattern`, the source dataset's cameras are reused verbatim —
    that is how `outline.json` re-renders the exact same views from a
    trained splat so `replace_views` can swap them back in.

    The `dataset` input is normally the same subject the splat is of, so its
    framing box is the right one to orbit. When it is not — rendering the
    head-only face splat alongside a body dataset, say — `bounds_source:
    splat` frames on the splat's own box instead. The intrinsics are
    untouched by that choice: only the radius moves, which matters because
    every view in one brush training shares a single COLMAP camera line.

    Rasterisation shells out to the `brush-splat-render` binary (a Rust CLI
    on `PATH`, built into docker/Dockerfile — see the module docstring).
    `render_path` overrides the binary name/path, same convention as
    `steps/brush.py`'s `brush_path`.
    """

    # Same shape as steps/render.py's declaration, with two differences:
    # `pattern` is empty by default (reuse the dataset's cameras verbatim),
    # and `n_frames` therefore cannot be REQUIRED — it is read only when a
    # pattern is set, and _generate_path says so if it is missing then.
    PARAMS = (
        Param("pattern", str, "",
              "Shape of the new camera path. Empty reuses the source dataset's "
              "cameras verbatim, which is how outline.json re-renders the exact "
              "same views for replace_views to swap back in. `cap` is the odd one "
              "out and is not an orbit at all: a disc of views around the "
              "photograph's own view of the splat, for supervising a training "
              "off its denoising path",
              choices=("", "circular", "sinusoidal", "helical", "cap")),
        Param("n_frames", int, None, "Views to render; required when a pattern is set",
              minimum=1),
        Param("width", int, 720, "Render width", minimum=1),
        Param("height", int, 1280, "Render height", minimum=1),
        Param("framing", str, "full",
              "Which of the source render's framing presets to reuse for the "
              "bounds. fast_helical_native threads one `framing` global through "
              "both the mesh `render` and this step, so a non-'full' preset "
              "re-renders the splat on the same re-aimed, tighter-framed orbit "
              "the mesh render used and the splat was trained on",
              choices=("full", "torso", "bust", "head")),
        Param("fill_ratio", float, 0.8, "How much of the frame the subject fills",
              minimum=0.0, maximum=1.0),
        Param("bounds_source", str, "dataset",
              "Which bounding box sizes and aims the new orbit. 'dataset' is the "
              "source render's `framing` box, which is what keeps a re-render framed "
              "identically to the render it replaces. 'splat' ignores that box and "
              "uses the loaded splat's own, for rendering a splat that is not the "
              "one the dataset was framed around — a head-only splat orbited on the "
              "body's box is a smudge in the middle of the frame. Pattern-only, and "
              "incompatible with override_cam_from_mesh (which takes its target from "
              "the dataset's metadata and never looks at a box at all)",
              choices=("dataset", "splat")),
        Param("focal_length_mm", float, 0.0,
              "0 inherits the dataset's, then falls back to one derived from width"),
        Param("override_cam_from_mesh", bool, False,
              "Anchor the new path on the dataset's original camera, as steps/render.py "
              "does, and publish where it landed as anchor_position / "
              "anchor_frame_index. ON for the helical re-render of a dataset that was "
              "rendered in override mode: inject_anchor matches on position, so it can "
              "only re-apply the anchor to a path that actually passes through it. "
              "Requires the marker an override-mode render leaves in the extras "
              "(original_focal_length); a dataset without one has been auto-oriented "
              "and its original camera is no longer at the world origin"),
        Param("bg_color", list, [0.0, 0.0, 0.0],
              "RGB in [0,1]. Black, not the recorded run's 127 grey: black is where "
              "that pipeline ends up after mask_splat, and matching its intermediate "
              "grey would reintroduce the same halo, just dimmer. IGNORED when "
              "`confidence` is on — `cull_color` is the background there"),
        Param("confidence", bool, False,
              "Gate the render on each Gaussian's multi-view evidence instead of "
              "leaving mask_splat to threshold rendered alpha afterwards. The alpha "
              "that comes back is then the gate, not accumulated opacity, and the "
              "RGB is composited over `cull_color` rather than `bg_color`. Needs a "
              ".ply trained with brush's `export_evidence`, or an `evidence_dataset` "
              "to measure against. Off for any render feeding select_support_views: "
              "that step requires premultiplied-over-black and refuses this output"),
        Param("cull_color", list, [0.5, 0.5, 0.5],
              "RGB in [0,1] that culled pixels resolve to in confidence mode, and "
              "the colour the kept ones are composited over. One colour for both is "
              "the point: partial coverage fades toward the same value the gate "
              "rejects to, so there is no halo to filter away afterwards"),
        Param("gate_lo", float, 0.45,
              "Confidence at or below which a pixel is fully culled", minimum=0.0,
              maximum=1.0),
        Param("gate_hi", float, 0.65,
              "Confidence at or above which a pixel is fully kept; between the two "
              "the gate is a smoothstep", minimum=0.0, maximum=1.0),
        Param("confidence_sidecar", bool, False,
              "Also keep each frame's raw per-pixel confidence as <stem>.conf.png. "
              "Tuning only: the frames themselves are unaffected, and the sidecars "
              "are copied out of the render's temp directory into the log dir, since "
              "nothing downstream reads them", advanced=True),
        Param("evidence_dataset", str, None,
              "Training dataset directory to measure evidence against when the .ply "
              "carries none — a run whose splat predates brush's export_evidence. "
              "The export_colmap_intermediate output is exactly what train_splat "
              "saw. Unused when the .ply has its own ev_* block", advanced=True),
        Param("conf_args", list, [],
              "Extra --conf-* flags passed verbatim to brush-splat-render, for "
              "tuning the confidence measure itself (--conf-tau, --conf-min-views, "
              "--conf-angle-margin, ...). Empty leaves the binary's defaults",
              advanced=True),
        Param("override_pointcloud", bool, False,
              "Sample a fresh point cloud off the splat instead of keeping the "
              "dataset's. The mesh render's cloud describes the actual subject "
              "geometry; one sampled from a trained splat inherits its noise"),

        Param("cap_radius_deg", float, 30.0,
              "Cap: angular radius of the disc of views, about the splat's centre. "
              "30 is where body2colmap measured a Face_Neck shell still reading "
              "cleanly — past it a 2.5-D shell is into its own open rim, and a rim "
              "is not supervision", minimum=0.0, maximum=180.0),
        Param("elevation_deg", float, 0.0, "Circular: camera elevation"),
        Param("start_azimuth_deg", float, 0.0, "Where the orbit starts"),
        Param("overlap", int, 1,
              "Circular: 1 makes the first and last frame share a position"),
        Param("amplitude_deg", float, 30.0,
              "Sinusoidal/helical: elevation swing either side of the equator"),
        Param("n_cycles", int, 1, "Sinusoidal: elevation cycles over the orbit"),
        Param("n_loops", int, 2, "Helical: turns around the subject"),
        Param("lead_in_deg", float, 45.0, "Helical: azimuth spent easing in"),
        Param("lead_out_deg", float, 45.0, "Helical: azimuth spent easing out"),

        Param("radius", float, None,
              "Orbit radius; empty derives one from the framing", advanced=True),
        Param("pointcloud_samples", int, 10000,
              "Points to sample when override_pointcloud is on", minimum=1,
              advanced=True),
        Param("splat_path", str, None,
              "A .ply to load when no splat_scene input is wired", advanced=True),
        Param("render_path", str, _RENDER_BINARY,
              "The rasteriser binary, on PATH or as an absolute path", advanced=True),
        # No `device`: rasterisation shells out to brush-splat-render, which
        # picks its own. The workflows used to pass one and it was read by
        # nothing; declaring it would put a control in the UI that does
        # nothing at all.
    ) + BACKGROUND_PARAMS

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.splat_scene import SplatScene

        scene = inputs.get("splat_scene")
        splat_path = inputs.get("splat_path") or params["splat_path"]
        if scene is None:
            if not splat_path:
                raise ValueError(
                    "render_splat needs a 'splat_scene' input, or a 'splat_path' "
                    "to load one from"
                )
            scene = SplatScene.from_ply(str(splat_path))

        dataset = inputs.get("dataset")
        width = params["width"]
        height = params["height"]

        cameras, focal_length, effective_mm, anchor_frame_index = _resolve_cameras(
            scene=scene, dataset=dataset, params=params, width=width, height=height
        )

        # BLACK, not white. Two reasons, and the second is the one that
        # actually bit:
        #
        #   * mask_splat composites this render over black a step later, so
        #     anything the mask drops ends at 0 regardless. Rendering white
        #     just guarantees the largest possible disagreement between the
        #     colour a pixel is rendered and the colour it ends up.
        #   * That disagreement is visible, not academic. A splat render has
        #     soft partial-alpha fringes wherever the Gaussians are uncertain
        #     (thin hair, silhouettes). On white those fringes are BRIGHT, and
        #     mask_splat's bilateral filter smooths them across the silhouette
        #     before anything blacks them out — so a white halo survives into
        #     the frames denoise_pass2 sees. On black there is nothing to
        #     bleed: fringe and background are the same colour.
        #
        # The recorded ComfyUI run rendered this stage on 127 grey with
        # alpha=0 in the background (cyber_6f/splatted), and its masked output
        # is black RGB with alpha=255 everywhere (cyber_6f/masked_splatted).
        # Black matches where that pipeline ENDS up, which is what matters
        # here — matching its intermediate grey would reintroduce the same
        # halo, just dimmer.
        #
        # All of the above is the NON-confidence contract, and it is still
        # live: the face-view cap renders that feed select_support_views
        # depend on it, and that step refuses anything else — as does the
        # `+splat` compositing in steps/render.py, which is why
        # `render_splat_layers` below passes the background itself rather
        # than taking one. In confidence mode
        # `bg_color` is not passed to the binary at all — `cull_color` is
        # both the background and the reject colour, and there is no halo to
        # avoid because partial coverage fades toward the same value the
        # gate rejects to. The history stays here rather than moving,
        # because it is the reason black and not grey is the default.
        #
        # A `background` displaces whichever of the two the binary drew, so
        # with a backdrop on this colour is no longer what any pixel ends up
        # (steps/backdrop.py un-composites it out again). It still has to be
        # right — the un-compositing is told which colour to remove — and the
        # renders that must have NO backdrop at all are exactly the ones this
        # paragraph is about: the face cap and the shells, whose alpha
        # select_support_views divides back out.
        bg_color = tuple(params["bg_color"])
        render_path = params["render_path"]
        image_names = [f"frame_{i + 1:05d}_.png" for i in range(len(cameras))]

        confidence = _confidence_options(params)

        logger.info(
            "render_splat: %d Gaussians (SH degree %d), %d frames at %dx%d via %s%s",
            len(scene), scene.sh_degree, len(cameras), width, height, render_path,
            "" if confidence is None else
            f", confidence-gated on {confidence.cull_color} "
            f"(gate {confidence.gate_lo}-{confidence.gate_hi})",
        )

        images, masks = _rasterize(
            scene=scene,
            splat_path=splat_path,
            cameras=cameras,
            image_names=image_names,
            width=width,
            height=height,
            bg_color=bg_color,
            render_path=render_path,
            confidence=confidence,
        )

        # The environment behind the splat (steps/backdrop.py). Composited
        # here rather than asked of the rasteriser, which draws one flat
        # colour and nothing else — and rather than of body2colmap, which
        # refuses a backdrop on a .ply outright for that reason. What it
        # needs to know is which flat colour it is displacing: `cull_color`
        # in confidence mode, where `bg_color` never reaches the binary at
        # all, and `bg_color` otherwise. `masks` is deliberately not touched
        # — it is the splat's own coverage, and every consumer of it (brush's
        # supporting views, mask_splat, colmap_export) means the subject.
        background = build_background(params, cameras)
        if background is not None:
            images = composite_bgr(
                images, masks,
                background=background,
                cameras=cameras,
                flat_color=(
                    bg_color if confidence is None else confidence.cull_color
                ),
            )

        points_3d = _resolve_pointcloud(scene, dataset, params)

        result: Dict[str, Any] = {
            "images": images,
            "masks": masks,
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": points_3d,
            "resolution": (width, height),
            "focal_length_mm": effective_mm,
        }

        # Carry through everything the source dataset knew that this render
        # does not rebuild, so downstream steps keep working without an
        # allowlist for every key.
        if dataset is not None:
            skip = _REBUILT_KEYS | _ANCHOR_KEYS if params["pattern"] else _REBUILT_KEYS
            for key, value in dataset.extras.items():
                if key not in skip:
                    result[key] = value

        if anchor_frame_index is not None:
            result["anchor_frame_index"] = int(anchor_frame_index)
            result["anchor_position"] = np.asarray(
                cameras[anchor_frame_index].position, dtype=np.float32
            )

        return result


def _confidence_options(params: Dict[str, Any]):
    """`params` read as a `ConfidenceOptions`, or None when the mode is off.

    body2colmap's dataclass, not a local mirror of it: the flags it builds
    (`--confidence`, `--cull-color`, the two gates, the sidecar and dataset
    switches) are the binary's, and there is no version of them that is
    this project's to define. The validation below is the one thing that is
    ours, because it is about the params a workflow can set.
    """
    if not params["confidence"]:
        return None

    from body2colmap.splat_renderer import ConfidenceOptions

    gate_lo = float(params["gate_lo"])
    gate_hi = float(params["gate_hi"])
    if gate_lo > gate_hi:
        raise ValueError(
            f"render_splat: gate_lo ({gate_lo}) is above gate_hi ({gate_hi}). "
            f"The gate is a smoothstep from lo to hi, so an inverted pair does "
            f"not mean 'keep less' — it is undefined."
        )
    return ConfidenceOptions(
        cull_color=tuple(float(c) for c in params["cull_color"]),
        gate_lo=gate_lo,
        gate_hi=gate_hi,
        sidecar=bool(params["confidence_sidecar"]),
        dataset=params["evidence_dataset"] or None,
        extra_args=tuple(str(a) for a in params["conf_args"]),
    )


def _keep_sidecars(maps, image_names: List[str]) -> None:
    """Write the render's per-pixel confidence maps out under `logs/`.

    They are diagnostics, not frames: nothing downstream reads them, and
    body2colmap hands them back in memory (`last_confidence_maps`) rather
    than on disk, because the directory the binary wrote them into is gone
    by the time `render_many` returns. Under `logs/` for the reason
    `paths.crash_dir()` gives: whatever gets copied off a pod to read the
    run log brings them along.
    """
    from ..paths import log_dir

    if not maps:
        logger.warning(
            "render_splat: confidence_sidecar is on but brush-splat-render "
            "produced no confidence maps"
        )
        return
    try:
        dest = log_dir() / "confidence" / time.strftime("%Y%m%d-%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for name, image in zip(image_names, maps):
            cv2.imwrite(str(dest / f"{Path(name).stem}.conf.png"), image)
    except OSError as exc:
        logger.warning("render_splat: could not keep the confidence sidecars: %s", exc)
        return
    logger.info("render_splat: kept %d confidence sidecars in %s", len(maps), dest)


def _rasterize(
    *, scene, splat_path, cameras, image_names, width, height, bg_color, render_path,
    confidence=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Render `cameras` against `scene` via body2colmap's `SplatRenderer`.

    Returns (images, masks): BGR uint8 images and float32 [0,1] foreground
    masks, one per camera, in `cameras` order.

    With `confidence`, the binary's output contract changes and so does what
    those two mean: the RGB is composited over the cull colour instead of
    `bg_color`, and the alpha is the confidence gate rather than accumulated
    opacity. It is still "the splat's per-pixel mask" as far as everything
    downstream is concerned — foreground is still 1 — which is why the
    read-back below is unchanged.

    **This used to be a parallel implementation of `render_many`** — its own
    Popen, its own cameras.json writer, its own exit-code judgement — kept
    because the library's version raised on any non-zero exit, told nobody
    what the binary was saying while it ran, and deleted a crashed run's
    directory before anything could be salvaged from it. All three are gone:
    `d0e3ada` judges a render by the files it produced, and the three seams
    used below carry the rest. What is left here is the three things that
    are genuinely this project's — where the log goes, where a crash report
    goes, and what a frame is called.

    `image_names` no longer names the files on disk (body2colmap picks
    those); it survives as the count and the order, and as the names the
    caller knows these frames by.
    """
    from body2colmap.splat_renderer import SplatRenderer

    logger_relay = OutputRelay(logging.getLogger("proc.brush-splat-render"))
    # An existing .ply is rendered where it lies. The common case: this step
    # follows a brush training, which wrote one, and a trained splat is
    # large enough that serialising the scene back out to render it is a
    # real cost rather than a tidy one.
    ply_path = Path(splat_path) if splat_path and Path(splat_path).exists() else None

    renderer = SplatRenderer(
        scene,
        (width, height),
        binary=render_path,
        confidence=confidence,
        ply_path=None if ply_path is None else str(ply_path),
        on_output=logger_relay,
        on_fault=lambda fault: _save_render_crashlog(fault, image_names),
    )
    try:
        logger.info("$ %s", render_path)
        frames = renderer.render_many(cameras, bg_color=bg_color)
        logger_relay.flush()

        if confidence is not None and confidence.sidecar:
            _keep_sidecars(renderer.last_confidence_maps, image_names)

        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for rgba in frames:
            # body2colmap returns RGBA; this pipeline's convention is BGR
            # plus a separate float mask (see steps/render.py's docstring).
            images.append(np.ascontiguousarray(rgba[..., 2::-1]))
            masks.append(rgba[..., 3].astype(np.float32) / 255.0)
        return images, masks
    finally:
        logger_relay.flush()
        renderer.close()


def render_splat_layers(
    *, scene, splat_path, cameras, width: int, height: int,
    render_path: str = _RENDER_BINARY, min_alpha: float = 0.004,
) -> List[np.ndarray]:
    """Render `scene` from `cameras` as straight-alpha RGBA overlay layers.

    The other half of `steps/render.py`'s `...+splat` render modes. That
    step draws the mesh through body2colmap's `Renderer.render_composite`
    and hands it the splat as a pre-rendered `splat_layer` — pyrender
    cannot rasterise Gaussians, and the binary that can renders a whole
    camera list per invocation, so the layers are batched here and passed
    down one frame at a time.

    **Straight alpha, not premultiplied.** `Renderer._composite_splat`
    blends `layer*a + base*(1-a)`, so the colour it is handed has to be the
    surface's own. The binary renders on black, which is premultiplied, so
    the division back out happens here — the same recovery
    `select_support_views` makes, for the same reason, and the same one
    body2colmap's `SplatRenderer.render_many(bg_color=None)` makes
    internally.

    **Why `_rasterize` and not `SplatRenderer.render_many(bg_color=None)`
    directly, which returns exactly this.** `_rasterize` *is* that call
    now; what it adds is this project's crash report, its log relay and its
    frame naming. Going straight to the library would mean a render started
    by a `+splat` mode is the one render in the pipeline that leaves
    nothing behind when it dies. See this module's docstring.

    Args:
        scene: A `SplatScene`, used only if `splat_path` is not on disk.
        splat_path: An existing .ply, reused rather than re-serialized.
        cameras: The cameras to render, in order.
        width: Render width in pixels; must match the cameras'.
        height: Render height in pixels.
        render_path: The rasteriser binary.
        min_alpha: Alpha at or below which a pixel is treated as fully
            transparent — the colour there is not recoverable by dividing,
            and a splat render's long tail of near-zero alpha would
            otherwise tint the whole frame by a level or two. The default
            is `select_support_views`' own.

    Returns:
        One RGBA uint8 array per camera, in `cameras` order.
    """
    from .anchor_stub import _unpremultiply

    image_names = [f"layer_{i + 1:05d}_.png" for i in range(len(cameras))]
    images, masks = _rasterize(
        scene=scene,
        splat_path=splat_path,
        cameras=cameras,
        image_names=image_names,
        width=width,
        height=height,
        bg_color=(0.0, 0.0, 0.0),
        render_path=render_path,
        confidence=None,
    )

    layers: List[np.ndarray] = []
    for bgr, alpha in zip(images, masks):
        alpha = np.where(alpha < min_alpha, 0.0, alpha).astype(np.float32)
        rgb = _unpremultiply(bgr[..., ::-1], alpha, min_alpha)
        layers.append(np.dstack(
            [rgb, np.clip(alpha * 255.0, 0, 255).astype(np.uint8)]
        ))
    return layers


def _save_render_crashlog(fault, image_names: List[str]) -> None:
    """What a crashed render is worth keeping, for `proc.save_crashlog`.

    Wired as body2colmap's `SplatRenderer(on_fault=...)`, which is called
    while the invocation's temp directory still exists and will not once
    this returns. That is the whole reason the hook is there: everything the
    binary was handed lives in that directory, so a crash on a pod left
    nothing behind but an exit code, and the pod does not outlive the
    investigation. Kept: the cameras.json naming the views it was rendering,
    a per-frame manifest of what did and did not get written, and the last
    few frames that did land (the crash's neighbourhood; a full orbit at
    720x1280 is not something to copy onto the volume every time a render
    fails).

    Called for a render that lost frames *and* for one that wrote everything
    and then died anyway — `fault.complete` separates them, and both are
    worth a report, because the second is only *probably* the known shutdown
    crash. The names in the manifest are this step's own
    (`frame_00001_.png`), not the renderer's `f00000.png`, so the report
    reads in the terms the rest of the run uses.
    """
    written = set(fault.written)
    manifest = "\n".join(
        f"{name}  {path.stat().st_size} bytes" if path in written else f"{name}  missing"
        for name, path in zip(image_names, fault.expected)
    )
    # The renderer names its files f00000.png; everything else in this run
    # calls them frame_00001_.png. Translate, so the report reads in the
    # terms the log and the dataset use rather than in the binary's.
    by_path = dict(zip(fault.expected, image_names))
    first_missing = (None if fault.complete
                     else by_path.get(fault.missing[0], fault.missing[0].name))

    if fault.complete:
        logger.warning(
            "brush-splat-render %s but wrote all %d frames — treating the render "
            "as successful. This is the known shutdown crash if the output ends "
            "after the last frame; anything else is a real failure that happened "
            "to leave a usable set of renders, so the diagnostics are kept either "
            "way — %s.",
            fault.status, len(image_names),
            crashlog_note(_render_crashlog(fault, manifest, first_missing)),
        )
        return

    logger.error(
        "brush-splat-render %s with %d of %d frames missing (first: %s). "
        "Diagnostics %s.",
        fault.status, len(fault.missing), len(fault.expected), first_missing,
        crashlog_note(_render_crashlog(fault, manifest, first_missing)),
    )


def _render_crashlog(fault, manifest: str, first_missing) -> Optional[Path]:
    """The report itself. Split only to keep the two log lines readable."""
    return save_crashlog(
        "brush-splat-render",
        cmd=fault.cmd,
        failure=fault.failure,
        summary=[
            f"frames:  {len(fault.written)} of {len(fault.expected)} written"
            + (" (all present)" if fault.complete
               else f", first missing {first_missing}"),
            f"splat:   {_describe_splat_arg(fault.cmd)}",
            f"exit:    {fault.status}",
        ],
        sections=[("frames", manifest),
                  ("output", fault.output)],
        copy=[
            # The views it was attempting to render — the thing whose
            # absence made the pod crash undiagnosable.
            ("", [fault.cameras_path]),
            ("frames", fault.written[-_CRASH_FRAMES_KEPT:]),
        ],
    )


def _describe_splat_arg(cmd: List[str]) -> str:
    """The .ply the render was reading, and whether it is a plausible one.

    A truncated or empty splat is one way this call fails that has nothing
    to do with the renderer, and the file is the training's, not this
    step's temp copy, so it is still there to check afterwards — but only
    if the report says which one it was.
    """
    try:
        return describe_path(Path(cmd[cmd.index("--splat") + 1]))
    except (ValueError, IndexError):
        return "no --splat in argv"


def _resolve_cameras(
    *, scene, dataset, params: Dict[str, Any], width: int, height: int
) -> Tuple[list, float, float, Optional[int]]:
    """Work out which cameras to render from, and at what focal length.

    Returns (cameras, focal_length_px, effective_focal_length_mm,
    anchor_frame_index). Split out of run() so the whole decision — which
    is where every subtlety in this step lives — is testable without gsplat
    or a GPU.
    """
    from body2colmap.camera import Camera
    from body2colmap.path import (
        OrbitPath,
        compute_helical_anchor_params,
        compute_original_camera_orbit_params,
    )
    from body2colmap.utils import compute_auto_orbit_radius, compute_default_focal_length

    pattern = params["pattern"]
    override_cam_from_mesh = params["override_cam_from_mesh"]
    bounds_source = params["bounds_source"]
    extras = dataset.extras if dataset is not None else {}

    # Both of these would otherwise be silent no-ops: neither branch below
    # that they land in ever computes a bounding box, so a caller asking for
    # the splat's own bounds would get the dataset's orbit anyway and no
    # indication of it. That is the exact failure this param exists to
    # prevent, so it is an error rather than a warning.
    if bounds_source == "splat":
        if not pattern:
            raise ValueError(
                "bounds_source='splat' sizes a NEW orbit from the splat's own "
                "bounding box, so it needs a `pattern`. With no pattern the "
                "dataset's cameras are reused verbatim and no box is consulted."
            )
        if override_cam_from_mesh:
            raise ValueError(
                "bounds_source='splat' and override_cam_from_mesh are mutually "
                "exclusive. The anchored path takes its target and radius from the "
                "source dataset's orbit_target and focal length, and never looks at "
                "a bounding box — pick one."
            )

    # Focal length: explicit param > inherited from the dataset > auto.
    focal_length_mm = float(params["focal_length_mm"] or 0.0)
    effective_mm = focal_length_mm
    if effective_mm <= 0:
        effective_mm = float(extras.get("focal_length_mm", 0.0) or 0.0)

    inherited_mm = float(extras.get("focal_length_mm", 0.0) or 0.0)
    if (
        override_cam_from_mesh
        and focal_length_mm > 0
        and inherited_mm > 0
        and not np.isclose(focal_length_mm, inherited_mm)
    ):
        logger.warning(
            "render_splat: focal_length_mm=%.2f overrides the %.2fmm the source "
            "dataset was framed at. The anchor frame will no longer match the "
            "reference image warped during that render. Set focal_length_mm to 0 "
            "to inherit it.",
            focal_length_mm, inherited_mm,
        )

    focal_length = (
        _mm_to_pixels(effective_mm, width) if effective_mm > 0
        else compute_default_focal_length(width)
    )

    # No pattern — the declared default, an empty string. Reuse the
    # dataset's cameras verbatim. Its anchor keys then
    # still describe this render, so they pass through untouched.
    if not pattern:
        if dataset is None or not dataset.cameras:
            raise ValueError(
                "render_splat needs either a 'pattern' param to build a new "
                "camera path, or a 'dataset' input whose cameras it can reuse."
            )
        if override_cam_from_mesh:
            raise ValueError(
                "override_cam_from_mesh requires a 'pattern' — there is no path to "
                "anchor when cameras are reused verbatim. Either set a pattern, or "
                "drop the option (reused cameras already carry the original "
                "render's anchor)."
            )
        logger.info("render_splat: reusing %d cameras from the dataset", len(dataset.cameras))
        return list(dataset.cameras), focal_length, effective_mm, None

    camera_template = Camera(
        focal_length=(focal_length, focal_length), image_size=(width, height)
    )

    if override_cam_from_mesh:
        cameras, anchor_frame_index = _anchored_path(
            pattern=pattern,
            params=params,
            extras=extras,
            camera_template=camera_template,
            effective_mm=effective_mm,
            orbit_path_cls=OrbitPath,
            circular_solver=compute_original_camera_orbit_params,
            helical_solver=compute_helical_anchor_params,
        )
        return cameras, focal_length, effective_mm, anchor_frame_index

    # Framing bounds from the source render, falling back to the splat's own
    # bounds. Using the mesh render's bounds is what keeps a re-render framed
    # identically to the render it is replacing.
    #
    # The mesh render applies the SAME `framing` preset (fast_helical_native
    # threads one `framing` global through both `render` and this step), so a
    # non-"full" preset here re-aims the orbit at that preset's centre and
    # sizes the radius from that preset's box — the orbit the splat was
    # actually trained on. It must not try to compensate by staying on the
    # full-body orbit: the training cameras are not there.
    #
    # `bounds_source: splat` opts out of all of that. It is for rendering a
    # splat that is NOT the one the dataset was framed around — the face
    # splat being the case in hand, a head-only shell whose world centre sits
    # a long way above a full-body orbit target. Framed on the body's box it
    # comes out as a ~180px smudge in the middle of the frame; framed on its
    # own it fills the frame. The lens does not change either way, only the
    # radius: compute_auto_orbit_radius dollies in on the smaller box, which
    # is what keeps every view sharing one set of intrinsics — COLMAP export
    # writes a single camera line for the whole set (body2colmap's
    # ColmapExporter._export_cameras takes cameras[0] and stamps CAMERA_ID 1
    # on every image), so a per-view focal length would be silently discarded.
    framing = params["framing"]
    framing_bounds = extras.get("framing_bounds") or {}

    raw_bounds = None if bounds_source == "splat" else framing_bounds.get(framing)
    if raw_bounds is not None:
        # A disk round-trip (Dataset.to_disk/from_disk) brings framing_bounds
        # back as nested plain lists rather than ndarrays; normalise so the
        # solvers below get real arrays either way.
        bounds = tuple(np.asarray(corner, dtype=np.float32) for corner in raw_bounds)
        logger.info("render_splat: using '%s' framing bounds from the dataset", framing)
    elif bounds_source == "splat":
        # Asked for, not fallen back to — so no warning, whatever `framing`
        # says. `framing` is simply not consulted in this mode.
        bounds = scene.get_bounds()
        logger.info(
            "render_splat: framing the orbit on the splat's own bounds "
            "(bounds_source='splat'); the dataset's '%s' box is ignored", framing,
        )
    else:
        if framing != "full":
            logger.warning(
                "render_splat: framing preset '%s' not available in the dataset's "
                "metadata; falling back to the splat scene's own bounds.", framing,
            )
        bounds = scene.get_bounds()

    orbit_center = (bounds[0] + bounds[1]) / 2.0
    radius = params["radius"]
    if radius is None:
        radius = compute_auto_orbit_radius(
            bounds=bounds,
            render_size=(width, height),
            focal_length=focal_length,
            fill_ratio=params["fill_ratio"],
        )

    path_gen = OrbitPath(target=orbit_center, radius=radius)
    cameras = _generate_path(
        path_gen, pattern, params, camera_template, extras,
        dataset_cameras=list(dataset.cameras) if dataset is not None and dataset.cameras else None,
    )
    logger.info(
        "render_splat: %s path, %d cameras, radius=%.3f", pattern, len(cameras), radius
    )
    return cameras, focal_length, effective_mm, None


def _cap_directions(n_frames: int, cap_radius_deg: float,
                   axis: np.ndarray) -> List[np.ndarray]:
    """`n_frames` unit directions spread evenly over a disc about `axis`.

    Even by AREA, not by angle: the samples are placed on a sunflower
    spiral, whose polar angle is drawn from the cap's own area measure
    (`cos t` uniform between the axis and the rim) and whose azimuth turns
    by the golden angle each step. That is what "uniformly within the
    circle" has to mean on a sphere — spacing the polar angle evenly
    instead would pile most of the views into the middle, where the
    photograph already is.

    Deterministic, and no sample lands exactly on the axis: the first
    sits half a step in, so the set is symmetric about the centre without
    duplicating the source view itself.
    """
    if n_frames < 1:
        raise ValueError(f"cap: n_frames must be at least 1, got {n_frames}")
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("cap: the cap's axis has no direction to spread about.")
    axis = axis / norm

    # Any two vectors perpendicular to the axis will do; picking the world
    # axis the cap leans on least keeps the cross products well conditioned.
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    right = np.cross(axis, seed)
    right /= np.linalg.norm(right)
    up = np.cross(axis, right)

    cos_radius = float(np.cos(np.radians(cap_radius_deg)))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    directions = []
    for index in range(n_frames):
        # Equal-area in the cap: cos(polar) uniform over [cos R, 1].
        fraction = (index + 0.5) / n_frames
        cos_polar = 1.0 - fraction * (1.0 - cos_radius)
        sin_polar = float(np.sqrt(max(0.0, 1.0 - cos_polar * cos_polar)))
        spin = golden_angle * index
        directions.append(
            axis * cos_polar
            + (right * np.cos(spin) + up * np.sin(spin)) * sin_polar
        )
    return directions


def _cap_axis(path_gen, params: Dict[str, Any],
              extras: Dict[str, Any],
              dataset_cameras: Optional[List[Any]] = None) -> Tuple[np.ndarray, str]:
    """Which way the cap points: at the photograph's view of the splat.

    The whole point of the pattern is to sample around the one view that
    was photographed, so the axis is the direction from the splat's centre
    to the camera the photo was taken from. That camera is read LIVE —
    `dataset.cameras[anchor_frame_index]` — rather than from the
    `anchor_position` the render recorded beside it, because the camera
    list is the truth and the record is only as fresh as the last step
    that wrote it: `refine_cameras` rewrites the cameras in place, and
    although the workflows have it republish the anchor's position too,
    a cap aimed from a record that has slipped would sample around a view
    nobody holds. The recorded position is the fallback when the dataset
    carries no cameras or no index (a cap over a bare splat), and without
    either there is no photograph to sample around and the cap falls back
    to the `start_azimuth_deg`/`elevation_deg` direction, the same
    convention the orbits start on.
    """
    target = np.asarray(path_gen.target, dtype=np.float64).reshape(3)
    anchor, source = None, ""
    index = extras.get("anchor_frame_index")
    if dataset_cameras and index is not None and 0 <= int(index) < len(dataset_cameras):
        anchor = np.asarray(dataset_cameras[int(index)].position, dtype=np.float64)
        source = f"the anchor camera's (frame {int(index)}) view of the splat"
    elif extras.get("anchor_position") is not None:
        anchor = np.asarray(extras["anchor_position"], dtype=np.float64)
        source = "the recorded anchor position's view of the splat"
    if anchor is not None:
        axis = anchor.reshape(3) - target
        if float(np.linalg.norm(axis)) >= 1e-9:
            return axis, source
        raise ValueError(
            "render_splat: the cap's anchor camera sits on the splat's centre, "
            "so there is no view direction to sample around."
        )

    from body2colmap import coordinates

    axis = coordinates.spherical_to_cartesian(
        1.0, params["start_azimuth_deg"], params["elevation_deg"])
    return np.asarray(axis, dtype=np.float64), (
        f"start_azimuth_deg={params['start_azimuth_deg']:.1f}, "
        f"elevation_deg={params['elevation_deg']:.1f} (no anchor_position in the "
        f"dataset's extras)")


def _cap_path(path_gen, params: Dict[str, Any], camera_template,
              extras: Dict[str, Any], dataset_cameras: Optional[List[Any]] = None):
    """A disc of views around the photograph's own view of the splat.

    Not an orbit. `circular`/`helical` sweep the whole subject for a
    training set; this samples the neighbourhood of ONE view, because that
    is the only part of the sphere a splat built from one photograph can
    speak for. It exists for `select_support_views`: supervision has to
    come from off the training's denoising path (a view on the path already
    has a denoised frame of its own), and a disc around the anchor is where
    the off-path views that the splat still saw actually are.
    """
    axis, source = _cap_axis(path_gen, params, extras, dataset_cameras)
    radius_deg = params["cap_radius_deg"]
    target = np.asarray(path_gen.target, dtype=np.float64).reshape(3)

    cameras = []
    for direction in _cap_directions(params["n_frames"], radius_deg, axis):
        camera = path_gen._create_camera(
            (target + direction * path_gen.radius).astype(np.float32),
            camera_template)
        camera.look_at(path_gen.target, path_gen.up_vector)
        cameras.append(camera)

    logger.info(
        "render_splat: cap of %d views within %.1f deg of %s",
        len(cameras), radius_deg, source,
    )
    return cameras


def _generate_path(path_gen, pattern: str, params: Dict[str, Any], camera_template,
                   extras: Optional[Dict[str, Any]] = None,
                   dataset_cameras: Optional[List[Any]] = None):
    if params["n_frames"] is None:
        raise ValueError(
            f"render_splat: pattern '{pattern}' builds a new camera path, so it needs "
            "an n_frames param. Leave `pattern` empty to reuse the source dataset's "
            "cameras instead."
        )
    if pattern == "cap":
        return _cap_path(path_gen, params, camera_template, extras or {}, dataset_cameras)
    if pattern == "circular":
        return path_gen.circular(
            n_frames=params["n_frames"],
            elevation_deg=params["elevation_deg"],
            start_azimuth_deg=params["start_azimuth_deg"],
            overlap=params["overlap"],
            camera_template=camera_template,
        )
    if pattern == "sinusoidal":
        return path_gen.sinusoidal(
            n_frames=params["n_frames"],
            amplitude_deg=params["amplitude_deg"],
            n_cycles=params["n_cycles"],
            start_azimuth_deg=params["start_azimuth_deg"],
            camera_template=camera_template,
        )
    if pattern == "helical":
        return path_gen.helical(
            n_frames=params["n_frames"],
            n_loops=params["n_loops"],
            amplitude_deg=params["amplitude_deg"],
            lead_in_deg=params["lead_in_deg"],
            lead_out_deg=params["lead_out_deg"],
            start_azimuth_deg=params["start_azimuth_deg"],
            camera_template=camera_template,
        )
    raise ValueError(f"Unknown path pattern: {pattern!r}")


def _anchored_path(
    *, pattern, params, extras, camera_template, effective_mm,
    orbit_path_cls, circular_solver, helical_solver,
):
    """Build a path anchored to the original SAM-3D-Body camera.

    Mirrors steps/render.py's override mode, but takes the orbit target and
    framed focal length from the source dataset's metadata rather than
    recomputing them from geometry — that is what keeps this render's anchor
    frame identical to the mesh render's, so the reference image warped
    during that render stays valid here without being re-warped.

    Anchoring assumes the original camera sits at the world origin, which
    only holds for a dataset rendered in override mode (a normal render
    auto-orients the scene and breaks it). `original_focal_length` is
    written only in override mode, so its presence is the marker.
    """
    if pattern not in ("circular", "helical"):
        raise ValueError(
            f"override_cam_from_mesh only works with a circular or helical "
            f"pattern, got {pattern!r}"
        )
    if effective_mm <= 0:
        raise ValueError(
            "override_cam_from_mesh needs a framed focal length, but neither the "
            "focal_length_mm param nor the dataset's metadata provides one. "
            "Re-render the source dataset with a render step that records the "
            "framed focal length it actually used."
        )
    if "orbit_target" not in extras:
        raise ValueError(
            "override_cam_from_mesh needs extras['orbit_target'] from the source "
            "render's metadata."
        )
    if "original_focal_length" not in extras:
        raise ValueError(
            "override_cam_from_mesh needs a dataset rendered with "
            "override_cam_from_mesh itself — the marker for that is "
            "extras['original_focal_length'], which this dataset lacks. A "
            "normally-rendered dataset has been auto-oriented, so the original "
            "camera is no longer at the world origin and the anchor would be wrong."
        )

    # Save -> Load turns the ndarray into a plain list; normalise.
    orbit_center = np.asarray(extras["orbit_target"], dtype=np.float32)

    if pattern == "circular":
        orbit_params = circular_solver(orbit_center)
        radius = float(orbit_params["radius"])
        path_gen = orbit_path_cls(target=orbit_center, radius=radius)
        cameras = path_gen.circular(
            n_frames=params["n_frames"],
            elevation_deg=orbit_params["elevation_deg"],
            start_azimuth_deg=orbit_params["start_azimuth_deg"],
            overlap=params["overlap"],
            camera_template=camera_template,
        )
        return cameras, 0

    helix_params = dict(
        n_frames=params["n_frames"],
        n_loops=params["n_loops"],
        amplitude_deg=params["amplitude_deg"],
        lead_in_deg=params["lead_in_deg"],
        lead_out_deg=params["lead_out_deg"],
    )
    anchor_info = helical_solver(target=orbit_center, **helix_params)
    path_gen = orbit_path_cls(target=orbit_center, radius=float(anchor_info["radius"]))
    cameras = path_gen.helical(
        start_azimuth_deg=anchor_info["start_azimuth_deg"],
        elevation_offset_deg=anchor_info["elevation_offset_deg"],
        camera_template=camera_template,
        **helix_params,
    )
    return cameras, int(anchor_info["anchor_frame_index"])


def _resolve_pointcloud(scene, dataset, params: Dict[str, Any]):
    """Sample a fresh point cloud, or keep the dataset's existing one.

    Preserving it is the default: the mesh render's point cloud describes
    the actual subject geometry, while one sampled from a trained splat
    inherits whatever noise that training left behind.
    """
    override = params["override_pointcloud"]
    if not override and dataset is not None and dataset.points_3d is not None:
        logger.info("render_splat: preserving the dataset's point cloud")
        return dataset.points_3d

    n_samples = params["pointcloud_samples"]
    logger.info("render_splat: sampling %d points from the splat scene", n_samples)
    return scene.get_point_cloud(n_samples=n_samples)
