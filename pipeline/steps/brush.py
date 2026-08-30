"""Gaussian-splat training via the Erant/brush CLI (a Rust binary on `PATH`,
never a Python binding — built into docker/Dockerfile).

Dispatch: `in_process`, not `docker` — targeting RunPod specifically, where
a pod is a single container with no nested Docker daemon to run a separate
brush image in (confirmed on a real pod: no /var/run/docker.sock, no
`docker` binary at all). This Step's own Python code has no conflicting
dependencies either way — subprocess.Popen'ing a CLI binary doesn't need
venv isolation — so `in_process` is fine for the Python side; what's
*unresolved* is brush's OS-level Vulkan/graphics requirement, since a
default RunPod pod's `NVIDIA_DRIVER_CAPABILITIES` only exposed
`compute,utility` on a real pod tested this session (`vulkaninfo` failed
with `ERROR_INCOMPATIBLE_DRIVER` even with the driver's Vulkan libraries
physically present). docker/Dockerfile bakes
`compute,utility,graphics,display` into the image itself, which is
necessary but not yet confirmed sufficient for how RunPod provisions a pod
from a custom image — see that file's comment and docs/docker.md. If this
never gets resolved for RunPod, `dispatch: docker` (this pipeline still
supports it — `pipeline/dispatch/docker.py`'s own docstring names brush as
its motivating case) is the fallback for any other target that does expose
a Docker daemon.

Port of nodes/brush_node.py's Body2COLMAP_RunBrush, minus everything that
was only there to unwrap ComfyUI's list-batched inputs (this pipeline's
Steps already take plain lists). UNVERIFIED end to end: brush itself was
never actually built or run in this session (see docs/docker.md's "Open
items" — the Vulkan/system-deps list there is researched, not confirmed by
a build), so treat this module the same way as the pod-untested steps
(sam3d_body, seedvr2) even though its logic is a close port of code that
does run in production via ComfyUI.

**A non-zero exit is not automatically a failed training.** brush has been
seen taking SIGSEGV (exit code -11) during shutdown, *after* it has already
written the export — the .ply on disk is complete and the training is done.
So a failed exit is checked against the artefact rather than trusted on its
own: if the export exists, is non-empty, and was written by this run (its
mtime changed, so a stale .ply left in an `export_dir` from a previous run
cannot stand in for a crashed one), the run is treated as successful and the
whole failure — exit code and output tail — is logged at WARNING. Any other
non-zero exit still raises, as does one that left no export behind.

**And a crash saves more than its exit code.** The COLMAP export brush
trains from is built into a `TemporaryDirectory` and deleted on the way out
of `run()`, exception or not — so a training that died on a pod left
nothing to look at but a return code, which is exactly how one
brush-splat-render crash became undiagnosable (see steps/splat.py). Any
non-zero exit, tolerated or not, and any clean exit that wrote no export at
all, now writes the argv, the output tail, the graphics environment, a
description of the export it was training on and the COLMAP model's own
.txt files to `paths.crash_dir()` first. Not the training frames: those are
several hundred MB and they are the dataset's, still on the volume after
the temp directory is gone.

Normal-map supervision: per the original node's behavior, a normal map
that already carries an alpha channel keeps it; otherwise the RGB frame's
own foreground mask (rmbg's output) is reused as the normal map's alpha,
since that mask is what the loss should be restricted to. Brush
auto-detects a `normals/` directory beside `images/` in the COLMAP export
— its absence (masks/normal_maps not passed) just leaves normal
supervision inactive, not an error.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..masks import mask_to_alpha_u8
from ..proc import (
    ProcessFailed,
    crashlog_note,
    describe_path,
    save_crashlog,
    stream_command,
)
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

# The COLMAP model itself — small, and the only record of what brush was
# asked to train on. Its siblings (images/, normals/) are the bulk and are
# deliberately not copied; see the module docstring.
_COLMAP_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")


@register_step("brush")
class BrushStep(Step):
    """Train a 3D Gaussian Splat using the brush CLI tool.

    inputs: {"cameras": List[Camera], "image_names": List[str],
             "points_3d": Tuple[np.ndarray, np.ndarray],
             "images": List[np.ndarray] BGR(A),
             "masks": Optional[List[np.ndarray]] float32 [0,1], foreground=1,
             "normal_maps": Optional[List[np.ndarray]] HxWx3 float32 [-1,1]}
    outputs: {"splat_path": str}

    `output_dir` puts the run under `<output_dir>/brush/training_<ms>/`,
    which is what an intermediate training wants: several of them in one
    workflow can't collide, and which is which is recoverable from the
    timestamps. `export_dir` instead names the directory to export straight
    into, for the one training whose .ply is a deliverable and therefore
    needs a predictable path — it is the whole reason the final splat can
    be `ply/scene.ply` and not `brush/training_1756042129481/export.ply`.
    Set one or the other; `export_dir` wins if both are given.
    """

    PARAMS = (
        Param("total_steps", int, 30000, "Training iterations", minimum=1),
        Param("sh_degree", int, 3, "Spherical-harmonic degree", minimum=0, maximum=4),
        Param("max_resolution", int, 1920, "Longest edge brush trains at", minimum=1),
        Param("max_splats", int, 10_000_000, "Cap on the number of Gaussians", minimum=1),
        Param("refine_every", int, 200, "Densify/prune interval, in steps", minimum=1),
        Param("alpha_mode", str, "transparent",
              "How brush reads the frames' alpha channel", choices=("transparent", "ignore")),
        Param("normal_loss_strength", float, 0.05,
              "Weight on the normal-map supervision loss; 0 disables it", minimum=0.0),
        Param("normal_loss_step_start", int, 5000,
              "Step at which normal supervision switches on", minimum=0),
        Param("normal_loss_every", int, 1,
              "Evaluate the normal loss every Nth step instead of every step. brush "
              "scales the sampled loss by N, so the expected gradient is unchanged "
              "and only the extra normal render in between is skipped; 1 is brush's "
              "own default", minimum=1),
        Param("output_dir", str, None,
              "Puts this training under <output_dir>/brush/training_<ms>/ — what an "
              "intermediate training wants. Empty falls back to the system temp dir, "
              "where the .ply only survives because Dataset.to_disk copies it at the "
              "end of the run"),
        Param("export_dir", str, None,
              "Export straight into this directory instead, for a training whose .ply "
              "is a deliverable and needs a predictable path. Wins over output_dir"),
        Param("export_name", str, "export.ply", "Filename of the exported .ply"),
        Param("brush_path", str, "brush",
              "The brush binary, on PATH or as an absolute path", advanced=True),
        Param("with_viewer", bool, False,
              "Let brush open its interactive viewer window; needs a display",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.exporter import ColmapExporter

        cameras = inputs["cameras"]
        image_names = inputs["image_names"]
        points_3d = inputs.get("points_3d")
        images = inputs["images"]
        masks = inputs.get("masks")
        normal_maps = inputs.get("normal_maps")

        if len(images) != len(image_names):
            raise ValueError(f"images ({len(images)}) and image_names ({len(image_names)}) length mismatch")
        if normal_maps is not None and len(normal_maps) != len(images):
            raise ValueError(
                f"Normal map count ({len(normal_maps)}) does not match image count "
                f"({len(images)}). Every training view needs a matching normal map."
            )

        brush_path = params["brush_path"]
        total_steps = params["total_steps"]
        sh_degree = params["sh_degree"]
        max_resolution = params["max_resolution"]
        max_splats = params["max_splats"]
        refine_every = params["refine_every"]
        alpha_mode = params["alpha_mode"]
        normal_loss_strength = params["normal_loss_strength"]
        normal_loss_step_start = params["normal_loss_step_start"]
        normal_loss_every = params["normal_loss_every"]
        with_viewer = params["with_viewer"]

        export_dir = params["export_dir"]
        if export_dir:
            out_root = Path(export_dir)
        else:
            timestamp = int(time.time() * 1000)
            out_root = (
                Path(params["output_dir"] or tempfile.gettempdir())
                / "brush" / f"training_{timestamp}"
            )
        out_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="b2c_colmap_") as temp_dir:
            colmap_dir = Path(temp_dir)

            ColmapExporter(cameras=cameras, image_names=image_names, points_3d=points_3d).export(
                output_dir=colmap_dir
            )

            images_dir = colmap_dir / "images"
            images_dir.mkdir(exist_ok=True)

            alpha_channel = None
            if masks is not None:
                # mask_to_alpha_u8, not an inline np.clip(m * 255.0, ...):
                # a mask that came from disk is uint8 [0,255], and scaling
                # that by 255 saturates every non-zero value to opaque,
                # throwing away exactly the soft silhouette edge normal
                # supervision cares about. pipeline/masks.py's docstring
                # names this line as the bug it exists to prevent;
                # colmap_export was fixed and this was missed.
                alpha_channel = [mask_to_alpha_u8(m) for m in masks]

            for i, (img, filename) in enumerate(zip(images, image_names)):
                if alpha_channel is not None:
                    alpha = alpha_channel[i]
                    if img.shape[-1] == 4:
                        rgba = img.copy()
                        rgba[..., 3] = alpha
                    elif img.shape[-1] == 3:
                        rgba = np.dstack([img, alpha])
                    else:
                        raise ValueError(f"Unexpected image channels: {img.shape[-1]} (expected 3 or 4)")
                    cv2.imwrite(str(images_dir / filename), rgba)
                else:
                    cv2.imwrite(str(images_dir / filename), img)

            if normal_maps is not None:
                normals_dir = colmap_dir / "normals"
                normals_dir.mkdir(exist_ok=True)
                for i, (normal, filename) in enumerate(zip(normal_maps, image_names)):
                    # normal is HxWx3 float32 in [-1, 1] (sapiens2's output convention) ->
                    # BGR uint8 [0, 255] for disk, matching how images are stored here.
                    normal_bgr = np.clip((normal[..., ::-1] + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
                    if alpha_channel is not None:
                        out = np.dstack([normal_bgr, alpha_channel[i]])
                    else:
                        out = normal_bgr
                    normal_path = normals_dir / Path(filename).with_suffix(".png").name
                    cv2.imwrite(str(normal_path), out)

            ply_output_name = params["export_name"]
            ply_path = out_root / ply_output_name
            cmd = [
                brush_path,
                str(colmap_dir),
                "--total-train-iters", str(total_steps),
                "--sh-degree", str(sh_degree),
                "--export-path", str(out_root.absolute()),
                "--export-name", ply_output_name,
                "--export-every", str(total_steps),
                "--max-resolution", str(max_resolution),
                "--max-splats", str(max_splats),
                "--refine-every", str(refine_every),
            ]
            if with_viewer:
                cmd.append("--with-viewer")
            if masks is not None:
                cmd.extend(["--alpha-mode", alpha_mode])
            if normal_maps is not None:
                cmd.extend([
                    "--normal-loss-weight", str(normal_loss_strength),
                    "--normal-loss-start-iter", str(normal_loss_step_start),
                    "--normal-loss-every", str(normal_loss_every),
                ])

            self._run_brush(cmd, ply_path, colmap_dir=colmap_dir)

        if not ply_path.exists():
            raise RuntimeError(
                f"Expected output PLY file not found: {ply_path}\nBrush may not have exported successfully."
            )

        return {"splat_path": str(ply_path.absolute())}

    def _run_brush(
        self, cmd: List[str], ply_path: Path, colmap_dir: Optional[Path] = None
    ) -> None:
        """Run the training, judging a failed exit against the export.

        `colmap_dir` is the export brush is training from — it exists only
        for the duration of the call, so it is passed in to be described
        and copied into a crash directory before it is deleted. Optional
        only because the exit-code tests drive this with a stand-in binary
        and no COLMAP export at all.
        """
        # What "this run wrote it" means, captured before launching: an
        # export_dir is reused across runs (${output_root}/ply), so a .ply
        # sitting there already is a previous training's, and accepting it
        # after a crash would hand back a stale splat as if it were new.
        before = ply_path.stat().st_mtime_ns if ply_path.exists() else None

        # Training output is relayed to the log line by line as it arrives
        # (see pipeline/proc.py). This used to buffer everything and show
        # it only on failure, which made a 30,000-iteration run — the
        # longest single thing in the pipeline — completely silent.
        try:
            stream_command(
                cmd,
                log_name="brush",
                not_found_hint=(
                    "It is built into the image at /usr/local/bin/brush; on a bare "
                    "machine, build it from Erant/brush's normal-map-supervision "
                    "branch or point the step's brush_path param at it."
                ),
            )
        except ProcessFailed as exc:
            # brush segfaults on shutdown sometimes, with the export already
            # complete on disk (see the module docstring). The artefact is
            # the better witness than the exit code — but only the artefact
            # this run produced.
            exported = _exported_this_run(ply_path, before)
            saved = _save_brush_crashlog(
                cmd=cmd, ply_path=ply_path, colmap_dir=colmap_dir,
                exported=exported, failure=str(exc),
            )
            if not exported:
                logger.error(
                    "brush failed and left no export from this run at %s. "
                    "Diagnostics %s.",
                    ply_path, crashlog_note(saved),
                )
                raise
            logger.warning(
                "brush exited non-zero but %s is complete (%d bytes, written by "
                "this run) — treating the training as successful. This is the "
                "known shutdown crash if the output below ends after the export; "
                "anything else here is a real failure that happened to leave a "
                "usable .ply, so the diagnostics are kept either way — %s. "
                "Suppressed failure follows.\n%s",
                ply_path, ply_path.stat().st_size, crashlog_note(saved), exc,
            )
            return

        # A clean exit that exported nothing is the failure that used to
        # surface back in run(), several lines and one deleted temp
        # directory later, as "Expected output PLY file not found".
        if not ply_path.exists() or ply_path.stat().st_size == 0:
            saved = _save_brush_crashlog(
                cmd=cmd, ply_path=ply_path, colmap_dir=colmap_dir,
                exported=False, failure="brush exited 0 without writing an export.",
            )
            raise RuntimeError(
                f"brush exited 0 but wrote no usable export: {describe_path(ply_path)}\n"
                f"Diagnostics {crashlog_note(saved)}."
            )
        if not _exported_this_run(ply_path, before):
            # Deliberately not fatal, unlike the same condition after a
            # crash: brush said it succeeded, and the only evidence against
            # it is an unchanged mtime, which a filesystem with coarse
            # timestamps could produce for a real overwrite. Worth saying
            # out loud, not worth failing a finished training over.
            logger.warning(
                "brush exited 0 but %s has the same mtime it had before the run — "
                "this may be a previous training's export rather than this one's.",
                ply_path,
            )


def _save_brush_crashlog(
    *,
    cmd: List[str],
    ply_path: Path,
    colmap_dir: Optional[Path],
    exported: bool,
    failure: str,
) -> Optional[Path]:
    """What a crashed training is worth keeping, for `proc.save_crashlog`.

    The COLMAP export is deleted the moment `run()` leaves its
    `TemporaryDirectory`, so the model files that say what brush was
    training on go to the crash directory while they exist. The frames
    beside them do not: they are hundreds of MB, and unlike the model they
    are the dataset's, still on the volume afterwards. What they were is
    recorded instead.
    """
    return save_crashlog(
        "brush",
        cmd=cmd,
        failure=failure,
        summary=[
            f"export:  {describe_path(ply_path)}"
            + (" — written by this run" if exported else " — NOT written by this run"),
        ],
        sections=[("training data", _describe_colmap_export(colmap_dir))],
        copy=[("colmap", [colmap_dir / name for name in _COLMAP_MODEL_FILES])]
        if colmap_dir is not None else [],
    )


def _describe_colmap_export(colmap_dir: Optional[Path]) -> str:
    """The export's shape in text: what brush was handed, and how much of it.

    An empty or half-written `images/` is a failure of this step rather
    than of brush, and telling the two apart afterwards needs the counts,
    since the directory itself is gone by then.
    """
    if colmap_dir is None:
        return "<not recorded>"
    lines = [describe_path(colmap_dir)]
    for name in _COLMAP_MODEL_FILES:
        lines.append(f"  {describe_path(colmap_dir / name)}")
    for name in ("images", "normals"):
        sub = colmap_dir / name
        if not sub.is_dir():
            lines.append(f"  {name}/ absent")
            continue
        files = sorted(f for f in sub.iterdir() if f.is_file())
        total = sum(f.stat().st_size for f in files)
        lines.append(f"  {name}/ {len(files)} files, {total} bytes")
        lines += [f"    {f.name}  {f.stat().st_size} bytes" for f in files[:3]]
        if len(files) > 3:
            lines.append(f"    ... and {len(files) - 3} more")
    return "\n".join(lines)


def _exported_this_run(ply_path: Path, before_mtime_ns: Optional[int]) -> bool:
    """Whether `ply_path` is an export the just-finished brush call wrote.

    Non-empty and newer than whatever was there before it started. Both
    halves matter: a zero-byte file is a crash mid-write, and an unchanged
    mtime is a previous run's .ply that this one never got as far as
    overwriting.
    """
    if not ply_path.exists():
        return False
    stat = ply_path.stat()
    if stat.st_size == 0:
        return False
    return before_mtime_ns is None or stat.st_mtime_ns != before_mtime_ns
