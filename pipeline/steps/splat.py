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

**Keep it off for any render that feeds `composite_splat_views`** (the
face-view renders in fast_helical_shell.yaml). That step composites
premultiplied colour and enforces premultiplied-over-black, so it refuses a
confidence render outright — loudly, which is the intended failure.

**A non-zero exit is not automatically a failed render**, exactly as in
steps/brush.py — same binary's other half, same known shutdown SIGSEGV.
If every expected frame is on disk and non-empty, the render is treated as
successful and the failure logged at WARNING; a missing frame still raises.
Because `_rasterize` renders into a fresh temp directory each call, there
is no stale-artefact question to answer here the way brush's reused
`export_dir` forces one.

**And a crash saves more than its exit code.** Everything a render is
holding — the cameras.json naming the views, the frames written so far —
lives in that temp directory and goes away with it, which is how one
brush-splat-render crash on a pod ended up with nothing to diagnose. Any
non-zero exit, and any clean exit that quietly wrote no frames, now copies
the camera list, a per-frame manifest, the argv, the output tail and the
Vulkan/driver environment into `paths.crash_dir()` first.

Everything *except* rasterisation still belongs to body2colmap
(`SplatScene`, `OrbitPath`, the `path`/`utils` helpers); `SplatRenderer`
(the gsplat wrapper) is no longer used here at all.

**Verification status.** The camera-path half — which is where all the
subtle behaviour lives — is verified locally against `cyber_6f`'s real
recorded metadata (see tests/test_splat.py), including that the anchored
override path puts a camera exactly on the recorded `anchor_position`.
PLY load/save is verified by round-trip. `brush-splat-render` itself is
verified against gsplat as described above, but that verification ran
outside this pipeline (a manual comparison against local data); running
*through* this step, end to end, on this box's GPU is still pending — see
docs/docker-build-notes.md.

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

import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..proc import (
    ProcessFailed,
    crashlog_note,
    describe_path,
    save_crashlog,
    stream_command,
)
from ..registry import register_step
from ..step import REQUIRED, Param, Step

logger = logging.getLogger(__name__)

_FULL_FRAME_SENSOR_WIDTH_MM = 36.0

# How many of the frames that did land get copied into a crash directory.
# The last few written are the crash's neighbourhood — the frame it died on
# and the ones just before it — and a full 100-frame orbit at 720x1280 is
# not something to copy onto the volume every time a render fails.
_CRASH_FRAMES_KEPT = 4

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
              "same views for replace_views to swap back in",
              choices=("", "circular", "sinusoidal", "helical")),
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
              "does. Off here in the helical re-render: that is a fresh, longer orbit "
              "framed from the splat's own bounds, and inject_anchor re-applies the "
              "anchor afterwards by matching on position"),
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
              "to measure against. Off for any render feeding composite_splat_views: "
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
        Param("render_path", str, "brush-splat-render",
              "The rasteriser binary, on PATH or as an absolute path", advanced=True),
        # No `device`: rasterisation shells out to brush-splat-render, which
        # picks its own. The workflows used to pass one and it was read by
        # nothing; declaring it would put a control in the UI that does
        # nothing at all.
    )

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
        # live: the face-view renders that feed composite_splat_views depend
        # on it, and that step refuses anything else. In confidence mode
        # `bg_color` is not passed to the binary at all — `cull_color` is
        # both the background and the reject colour, and there is no halo to
        # avoid because partial coverage fades toward the same value the
        # gate rejects to. The history stays here rather than moving,
        # because it is the reason black and not grey is the default.
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


@dataclass(frozen=True)
class _Confidence:
    """The `--confidence` half of a `brush-splat-render` call, as one value.

    Grouped rather than spread across seven more `_rasterize` keywords
    because they are meaningless individually: `confidence: false` makes
    every one of them dead, which is exactly what `None` says here.
    """

    cull_color: Tuple[float, ...]
    gate_lo: float
    gate_hi: float
    sidecar: bool
    dataset: Optional[str]
    extra_args: Tuple[str, ...]


def _confidence_options(params: Dict[str, Any]) -> Optional[_Confidence]:
    """`params` read as a `_Confidence`, or None when the mode is off."""
    if not params["confidence"]:
        return None
    gate_lo = float(params["gate_lo"])
    gate_hi = float(params["gate_hi"])
    if gate_lo > gate_hi:
        raise ValueError(
            f"render_splat: gate_lo ({gate_lo}) is above gate_hi ({gate_hi}). "
            f"The gate is a smoothstep from lo to hi, so an inverted pair does "
            f"not mean 'keep less' — it is undefined."
        )
    return _Confidence(
        cull_color=tuple(float(c) for c in params["cull_color"]),
        gate_lo=gate_lo,
        gate_hi=gate_hi,
        sidecar=bool(params["confidence_sidecar"]),
        dataset=params["evidence_dataset"] or None,
        extra_args=tuple(str(a) for a in params["conf_args"]),
    )


def _keep_sidecars(output_dir: Path, image_names: List[str]) -> None:
    """Copy `<stem>.conf.png` out of the render's temp directory.

    They are diagnostics, not frames: nothing downstream reads them, and
    `_rasterize` deletes the directory they were written into on the way
    out — the same way a crash used to take its own evidence with it. Under
    `logs/` for the reason `paths.crash_dir()` gives: whatever gets copied
    off a pod to read the run log brings them along.
    """
    from ..paths import log_dir

    try:
        dest = log_dir() / "confidence" / time.strftime("%Y%m%d-%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        kept = 0
        for name in image_names:
            sidecar = output_dir / f"{Path(name).stem}.conf.png"
            if sidecar.exists():
                shutil.copy2(sidecar, dest / sidecar.name)
                kept += 1
    except OSError as exc:
        logger.warning("render_splat: could not keep the confidence sidecars: %s", exc)
        return
    if kept:
        logger.info("render_splat: kept %d confidence sidecars in %s", kept, dest)
    else:
        logger.warning(
            "render_splat: confidence_sidecar is on but brush-splat-render wrote "
            "no .conf.png beside the frames in %s", output_dir,
        )


def _rasterize(
    *, scene, splat_path, cameras, image_names, width, height, bg_color, render_path,
    confidence: Optional[_Confidence] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Render `cameras` against `scene` via the `brush-splat-render` CLI.

    Returns (images, masks): BGR uint8 images and float32 [0,1] foreground
    masks, one per camera, in `cameras` order.

    With `confidence`, the binary's output contract changes and so does what
    those two mean: the RGB is composited over the cull colour instead of
    `bg_color`, and the alpha is the confidence gate rather than accumulated
    opacity. It is still "the splat's per-pixel mask" as far as everything
    downstream is concerned — foreground is still 1 — which is why the
    read-back below is unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="b2c_render_splat_") as tmp:
        tmp_dir = Path(tmp)

        # Reuse an existing PLY on disk rather than re-serializing through
        # SplatScene when we already have one (the common case: this step
        # usually follows `brush` training, which writes a PLY directly).
        ply_path = Path(splat_path) if splat_path and Path(splat_path).exists() else None
        if ply_path is None:
            ply_path = tmp_dir / "scene.ply"
            scene.to_ply(str(ply_path))

        cameras_path = tmp_dir / "cameras.json"
        _write_cameras_json(cameras, image_names, width, height, cameras_path)

        output_dir = tmp_dir / "renders"
        cmd = [
            render_path,
            "--splat", str(ply_path),
            "--cameras", str(cameras_path),
            "--output-dir", str(output_dir),
        ]
        if confidence is None:
            cmd += ["--background", ",".join(f"{c:.6f}" for c in bg_color)]
        else:
            # --background is ignored by the binary in this mode, so it is
            # not passed at all: an argv that carries a background nothing
            # reads is the kind of thing that gets tuned for a run and then
            # blamed for the result.
            cmd += [
                "--confidence",
                "--cull-color", ",".join(f"{c:.6f}" for c in confidence.cull_color),
                "--gate-lo", str(confidence.gate_lo),
                "--gate-hi", str(confidence.gate_hi),
            ]
            if confidence.sidecar:
                cmd.append("--confidence-sidecar")
            if confidence.dataset:
                # Only for a .ply that predates brush's export_evidence. The
                # dataset options have to match what training saw — the
                # alpha mode decides whether ground truth is premultiplied
                # at load, so getting it wrong measures evidence against
                # pixels training never saw. Matching now means passing NO
                # --alpha-mode: steps/brush.py stopped forcing one so that a
                # run can mix masked and transparent views, and brush reads
                # the mode per view from the dataset's own layout. An export
                # of RGBA frames with no masks/ sidecar — which is what
                # colmap_export writes — still loads as transparent, exactly
                # as the forced flag used to make it.
                cmd += ["--dataset", str(confidence.dataset)]
            cmd += list(confidence.extra_args)
        _run_render(
            cmd,
            output_dir=output_dir,
            cameras_path=cameras_path,
            image_names=list(image_names),
        )
        if confidence is not None and confidence.sidecar:
            _keep_sidecars(output_dir, list(image_names))

        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        for name in image_names:
            png_path = output_dir / name
            rgba = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                raise RuntimeError(
                    f"render_splat: brush-splat-render did not produce {png_path}\n"
                    f"Command: {' '.join(cmd)}"
                )
            images.append(rgba[..., :3])
            masks.append(rgba[..., 3].astype(np.float32) / 255.0)

    return images, masks


def _write_cameras_json(cameras, image_names, width: int, height: int, path: Path) -> None:
    """Write brush-splat-render's cameras.json — see
    ~/Projects/brush/docs/splat-render.md for the schema. `camera.rotation`
    is already in the exact row-major, OpenGL-convention c2w form the
    binary expects; the OpenGL->OpenCV axis conversion happens Rust-side.
    """
    payload = {
        "width": width,
        "height": height,
        "cameras": [
            {
                "name": name,
                "fx": float(camera.fx),
                "fy": float(camera.fy),
                "cx": float(camera.cx),
                "cy": float(camera.cy),
                "position": [float(v) for v in camera.position],
                "rotation": [[float(v) for v in row] for row in camera.rotation],
            }
            for camera, name in zip(cameras, image_names)
        ],
    }
    path.write_text(json.dumps(payload))


def _run_render(
    cmd: List[str],
    *,
    output_dir: Path,
    cameras_path: Path,
    image_names: List[str],
) -> None:
    """Run the rasteriser, judging a failed exit against the frames it wrote.

    Same shape as steps/brush.py's `_run_brush`, for the same reason: this
    is the other half of the same Rust binary, and it has been seen failing
    on a pod in what looks like brush's known shutdown SIGSEGV — a crash
    *after* the work is on disk. The artefact is the better witness than
    the exit code, so a non-zero exit that nonetheless left every expected
    frame behind is logged at WARNING and treated as a successful render;
    one with a frame missing still raises.

    Where brush has to prove the .ply is *this* run's (its `export_dir` is
    reused across runs, so a stale export could stand in for a crashed
    one), there is no equivalent check to make here: `_rasterize` renders
    into a fresh `TemporaryDirectory` every call, so anything in
    `output_dir` was written by the process that just exited.

    Either way — and also on the reverse case, a clean exit that silently
    produced nothing — the diagnostics are copied out to `crash_dir()`
    before the temp directory that holds them is deleted.
    """
    try:
        # Same live relay as the brush training step — see pipeline/proc.py.
        stream_command(
            cmd,
            log_name="brush-splat-render",
            not_found_hint=(
                "It is built into the image at /usr/local/bin/brush-splat-render, "
                "alongside `brush`, from the same Erant/brush clone."
            ),
        )
    except ProcessFailed as exc:
        missing = _missing_renders(output_dir, image_names)
        saved = _save_render_crashlog(
            cmd=cmd, cameras_path=cameras_path, output_dir=output_dir,
            image_names=image_names, missing=missing, failure=str(exc),
        )
        if missing:
            logger.error(
                "brush-splat-render failed with %d of %d frames missing (first: %s). "
                "Diagnostics %s.",
                len(missing), len(image_names), missing[0], crashlog_note(saved),
            )
            raise
        logger.warning(
            "brush-splat-render exited non-zero but all %d frames are complete — "
            "treating the render as successful. This is the known shutdown crash "
            "if the output below ends after the last frame; anything else here is "
            "a real failure that happened to leave a usable set of renders, so the "
            "diagnostics are kept either way — %s. Suppressed failure follows.\n%s",
            len(image_names), crashlog_note(saved), exc,
        )
        return

    # A clean exit with frames missing is the failure that used to surface
    # several lines later as `cv2.imread` returning None, by which point the
    # temp directory holding the evidence was already gone.
    missing = _missing_renders(output_dir, image_names)
    if missing:
        saved = _save_render_crashlog(
            cmd=cmd, cameras_path=cameras_path, output_dir=output_dir,
            image_names=image_names, missing=missing,
            failure="brush-splat-render exited 0 without writing every frame.",
        )
        raise RuntimeError(
            f"brush-splat-render exited 0 but wrote {len(image_names) - len(missing)} "
            f"of {len(image_names)} frames (first missing: {missing[0]}).\n"
            f"Diagnostics {crashlog_note(saved)}."
        )


def _missing_renders(output_dir: Path, image_names: List[str]) -> List[str]:
    """The expected frames that are absent or empty, in render order.

    Zero-byte counts as missing for the same reason it does for brush's
    .ply: that is a crash caught mid-write, not a frame.
    """
    missing = []
    for name in image_names:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            missing.append(name)
    return missing


def _save_render_crashlog(
    *,
    cmd: List[str],
    cameras_path: Path,
    output_dir: Path,
    image_names: List[str],
    missing: List[str],
    failure: str,
) -> Optional[Path]:
    """What a crashed render is worth keeping, for `proc.save_crashlog`.

    Everything this step hands the binary lives in a `TemporaryDirectory`
    that is deleted on the way out of `_rasterize`, exception or not — so a
    crash on a pod left nothing behind but an exit code, and the pod itself
    does not outlive the investigation. Kept: the cameras.json naming the
    views it was rendering, a per-frame manifest of what did and did not
    get written, and the last few frames that did land (the crash's
    neighbourhood; a full orbit at 720x1280 is not something to copy onto
    the volume every time a render fails).
    """
    written = [name for name in image_names if name not in set(missing)]
    manifest = "\n".join(
        f"{name}  {(output_dir / name).stat().st_size} bytes"
        if (output_dir / name).exists() else f"{name}  missing"
        for name in image_names
    )
    return save_crashlog(
        "brush-splat-render",
        cmd=cmd,
        failure=failure,
        summary=[
            f"frames:  {len(written)} of {len(image_names)} written"
            + (f", first missing {missing[0]}" if missing else " (all present)"),
            f"splat:   {_describe_splat_arg(cmd)}",
        ],
        sections=[("frames", manifest)],
        copy=[
            # The views it was attempting to render — the thing whose
            # absence made the pod crash undiagnosable.
            ("", [cameras_path]),
            ("frames", [output_dir / name for name in written[-_CRASH_FRAMES_KEPT:]]),
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
    cameras = _generate_path(path_gen, pattern, params, camera_template)
    logger.info(
        "render_splat: %s path, %d cameras, radius=%.3f", pattern, len(cameras), radius
    )
    return cameras, focal_length, effective_mm, None


def _generate_path(path_gen, pattern: str, params: Dict[str, Any], camera_template):
    if params["n_frames"] is None:
        raise ValueError(
            f"render_splat: pattern '{pattern}' builds a new camera path, so it needs "
            "an n_frames param. Leave `pattern` empty to reuse the source dataset's "
            "cameras instead."
        )
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
