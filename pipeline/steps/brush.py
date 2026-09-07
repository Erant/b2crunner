"""Gaussian-splat training via a trainer CLI on `PATH` (a binary, never a
Python binding — built into docker/Dockerfile).

The binary is **b2ctrain** (Erant/b2ctrain, and `brush_path`'s default
since 2026-09-07): a C++/CUDA trainer that takes brush's argv, brush's
dataset layout (`init.ply`, `masks/`, `normals/`, `weights/`) and writes
brush's .ply with the same `ev_*` evidence block, at roughly a quarter of
the wall time and the same quality and splat count (its docs/STATUS.md has
the measurements against the fork). It REPLACED the Erant/brush fork this
module was written against, which is no longer built into the image at
all — so everything below that says "brush" is the CLI contract rather
than the implementation: the argv, the polish, the crash handling and the
names (`brush_path`, `brush-splat-render`, this module) are unchanged
because that contract is exactly what the new trainer answers to. Pointing
`brush_path` back at a brush binary still works if one is on PATH.

The one thing that is NOT a drop-in is the **alignment loop**. b2ctrain
carries it in-process (`--align-iters`), so with `align_backend: auto`
this step passes the loop's settings on the cold run's argv and makes one
invocation instead of one render, one flow and one re-invocation per
iteration — see `_use_trainer_alignment`. The loop below is the same
alignment done from here; it is what runs under `align_backend: pipeline`,
against any trainer, and it is the reference the loop's settings were
measured on.

Dispatch: `in_process`, not `docker` — targeting RunPod specifically, where
a pod is a single container with no nested Docker daemon to run a separate
brush image in (confirmed on a real pod: no /var/run/docker.sock, no
`docker` binary at all). This Step's own Python code has no conflicting
dependencies either way — subprocess.Popen'ing a CLI binary doesn't need
venv isolation — so `in_process` is fine for the Python side. The
OS-level requirement that used to sit here went with brush: that trainer
was wgpu/Vulkan, and a default RunPod pod's `NVIDIA_DRIVER_CAPABILITIES`
exposes `compute,utility` only, which is why docker/Dockerfile bakes
`compute,utility,graphics,display` into the image and why
`pipeline.cli doctor` had a `vulkan` check. b2ctrain is CUDA: it needs
nothing from the driver that the denoise steps do not already need (a
>= 580 driver for its CUDA 13 build), and neither does the rasteriser the
alignment loop calls, which is the same binary. `dispatch: docker` remains
supported (`pipeline/dispatch/docker.py`'s own docstring names brush as its
motivating case) for any target that does expose a Docker daemon.

Port of nodes/brush_node.py's Body2COLMAP_RunBrush, minus everything that
was only there to unwrap ComfyUI's list-batched inputs (this pipeline's
Steps already take plain lists). What HAS been run: both of the shipped
workflow's trainings, with their real argv and dataset shapes, driven
through this very Step class on a 4070 Ti (bench/b2crunner_step.py in the
b2ctrain repo). What has not: any of it on a pod, or from the image.

**A non-zero exit is not automatically a failed training.** brush has been
seen taking SIGSEGV (exit code -11) during shutdown, *after* it has already
written the export — the .ply on disk is complete and the training is done.
So a failed exit is checked against the artefact rather than trusted on its
own: if the export exists, is non-empty, and was written by this run (its
mtime changed, so a stale .ply left in an `export_dir` from a previous run
cannot stand in for a crashed one), the run is treated as successful and the
whole failure — exit code and output tail — is logged at WARNING. Any other
non-zero exit still raises, as does one that left no export behind. (That
shutdown crash was brush's; b2ctrain has not been seen doing it. The
tolerance stays because judging a training by the artefact it wrote is the
right rule either way, and it is what keeps an hour of GPU when a trainer
dies on the way out.)

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

**Multi-view evidence** (`export_evidence`, on by default) is measured
after the last training step and written into the exported .ply as seven
extra vertex properties (`ev_w_in`, `ev_w_all`, `ev_err`, `ev_views`,
`ev_dir_0..2`): for each Gaussian, how much of its rendered weight landed
inside the training masks, how many views actually supported it, how badly
it disagreed with them, and from which direction it was seen. That is the
per-Gaussian record of "the training views constrained this", and it is
what `render_splat`'s `confidence` mode reads to decide, in 3-D and once,
what `mask_splat` used to guess per pixel per frame from rendered alpha
alone (see docs/spatial-reinforcement.md). Every other .ply reader ignores
the extra properties, and the measurement costs seconds, so it is on for
both trainings — an intermediate splat that carries its evidence needs no
second pass over the dataset to be gated, and the final .ply is a
deliverable that is more useful with it than without.

**The polish** (`polish_steps`, off by default) is a second invocation of
brush on the same export, warm-started from the first run's .ply through an
`init.ply` symlink, with growth off (`--growth-stop-iter 0`, `--refine-every`
past the run's length) and the normal loss on from step 0. It exports over
the first .ply, so nothing downstream has to know it happened. What it bought
was not iterations: measured on the intermediate splat (2026-09-05,
docs/intermediate-splat-guide.md) 9000 of these moved band-limited face
sharpness from 143 to 161 where 9000 more iterations of one cold run reached
137 — the restart at full mean learning rate was the effect. It costs one
dataset reload and, on a 4070 Ti, about 2 minutes.

Past tense on purpose: **no shipped workflow polishes any more.** Every one
of those figures is brush's, and re-measured on b2ctrain (2026-09-07, both
trainings) the polish did not improve quality — so `fast_helical_native`
sets `polish_steps: 0` on both. The machinery stays because the finding is
about a trainer rather than about the idea: a trainer whose schedule leaves
headroom at the end of a cold run can still be worth restarting, and this is
how you would find out.

**The alignment loop** (`align_iters`, 4 by default) is the answer to a
measurement that says the trained splat is softer than the frames it was
fitted to *because of the fit*: the generated views disagree with each
other about where skin, hair and finger texture sits by 1.7-3.6 px mean
(p90 up to 8), and a photometric loss averages that into a blurred
consensus. So between training invocations the frames are pulled onto the
splat's own consensus — render the current .ply at the training cameras,
measure the optical flow from each frame to its render, smooth and cap it
(`align_flow_sigma`/`align_flow_cap`, one entry per iteration or one for
all of them), and Lanczos-warp the frame by it (pipeline/align.py) — and
the training is resumed on the aligned set with growth off, exactly the way
the polish resumes. Measured (docs/final-splat-alignment-guide.md): band-limited face
sharpness 21.1 -> 23.8 over four iterations, +1.2/+0.6/+0.5/+0.4, and
saturating there; novel views gain in the same ratio and fidelity RISES
with sharpness (27.64 -> 28.41 dB), which is the signature of recovered
rather than invented detail. It is worth as much as running brush's dense
growth and costs a fifth of the .ply size.

Three things about it are load-bearing:

- **Every iteration warps the pristine originals**, never a warp of a warp.
  That makes the loop a many-to-one contraction onto one consensus, anchored
  because the splat must still explain the mean image; iterating
  warps-of-warps is pairwise merging, which drifts without bound. The
  originals are the `images`/`masks` inputs, still in memory, and each
  iteration overwrites `images/` from them — nothing on disk is ever warped
  twice.
- **The gain lives in the splat, not in the images.** Each iteration's frame
  set is only ever "originals warped once", so there is no aligned dataset
  anywhere that carries four iterations of improvement, and none of this can
  be exported as a better set of frames.
- **The alignment invocations pass normal weight 0** regardless of
  `normal_loss_strength`. The warped frames no longer agree with the
  `normals/` sidecar beside them, and normal supervision measured as a
  straight loss on the deliverable anyway (guide §1: -18% sharpness *and*
  -1.8 dB fidelity).

What a run leaves behind is a fourth thing worth naming, because these
runs cost an hour of GPU and everything the loop touches is transient. Each
iteration logs the disagreement it measured, and the loop closes with the
whole trajectory on one line — read it against the reference loop's
1.02 -> 1.18 -> 1.26 -> 1.31 px, since a RISING, decelerating measurement is
what a working loop looks like (a sharper render gives the flow more to lock
onto) and not the drift it reads like. A p90 at or past the cap warns, which
is quiet on a healthy run. `align_debug_dir` keeps the rest: every view's
own figures as JSON, plus one warped frame and the render it was warped onto
per iteration, which is the only way to see a tear after the fact.

`export_evidence` then measures against the *warped* frames, since that is
what the finished splat was fitted to — deliberate, because `render_splat`'s
confidence mode gates on those `ev_*` properties. Supporting views are not
warped: they are renders of a splat rather than generated frames, they are
not what the flow was measured on, and the final training takes none.

**Supporting views** (`support_*` inputs) are views the training should
fit where they can be trusted and *ignore* everywhere else — the
confidence-gated splat re-renders are the case this exists for. Their
background is the cull colour, not emptiness, so they must not be allowed
to carve the silhouette: a frame whose alpha says "ignore this" is
brush's **masked** mode, and one whose alpha says "nothing is here" is
**transparent**. Brush resolves that per view from the export's layout —
a `masks/<name>` sidecar means masked, an alpha channel embedded in the
frame means transparent — so this step writes the training views as RGBA
exactly as it always did and the supporting views as RGB plus a sidecar,
and passes no `--alpha-mode`. That flag is a *global force*: passing it
flattens the mix, which is why it is now emitted only when a caller
explicitly asks for one. See brush's docs/mixed-alpha-modes.md.

Two things about that are sharp enough to name. An RGBA frame whose alpha
is really a mask, with no sidecar, loads as transparency and is
premultiplied at load — which destroys the RGB underneath — so intent has
to come from the layout and cannot be sniffed from the pixels. And brush
matches a sidecar to a frame by *stem* as well as by full name, so two
views whose names differ only by extension would share one mask; the
export refuses that rather than silently flipping a training view to
masked.

`--normalize-masked-loss` follows from the same mix. The loss kernel
weights each pixel by the frame's alpha but the trainer averages over the
whole frame, so a masked view whose mask covers a fifth of the frame
contributes about a fifth of the gradient of a transparent view of the
same subject. In a run that is all one mode that is a harmless rescale;
in a mixed one the supporting views quietly count for less, so
`normalize_masked_loss: auto` turns it on exactly when the export
actually carries both.

**Loss weights** (`weights`, optional, one per training view) are the
third thing a view can carry and the only one that is not about its alpha:
a float32 [0,1] map written as a greyscale `weights/<name>.png` sidecar,
which brush multiplies into that view's loss map pixel by pixel on top of
whatever its alpha mode does (brush's docs/loss-weights.md). A transparent
view has no other weight channel — its alpha is a target — so this is how
`face_priority_weights` fades the denoised frames out over the face while
they keep carving the silhouette. Absent means 1 everywhere, and brush's
evidence pass honours the same map, so a silenced region does not count
as evidence against what the supporting views put there.

`export_colmap_intermediate` writes the same export: `colmap_export` takes
the same `support_*` inputs, reads them with this module's `_SupportViews`
and writes them with the same code, so wiring the two steps to the same
context paths gives a debug dataset that is a record of what brush saw
rather than a near-miss of it. (`render_splat`'s `evidence_dataset`
fallback measures evidence against that dataset, and so now measures
against the supporting views too.)

Normal-map supervision: per the original node's behavior, a normal map
that already carries an alpha channel keeps it; otherwise the RGB frame's
own foreground mask (rmbg's output) is reused as the normal map's alpha,
since that mask is what the loss should be restricted to. Brush
auto-detects a `normals/` directory beside `images/` in the COLMAP export
— its absence (masks/normal_maps not passed) just leaves normal
supervision inactive, not an error.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..align import BACKGROUND as ALIGN_BACKGROUND, align_views
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
# The rasteriser the alignment loop renders the current splat with — the
# same binary, the same default and the same crash reporting steps/splat.py
# uses, rather than a second Popen of it here. Module level because it is a
# param default; steps/splat.py imports nothing from this module, so there
# is no cycle to defer around.
from .splat import _RENDER_BINARY, _rasterize

logger = logging.getLogger(__name__)

# The COLMAP model itself — small, and the only record of what brush was
# asked to train on. Its siblings (images/, normals/) are the bulk and are
# deliberately not copied; see the module docstring.
_COLMAP_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")

# Only used when the caller supplies no names of its own. Kept distinct
# from the `frame_NNNNN_` the renderers produce so that a glance at a crash
# directory's images/ says which views were supporting ones.
_SUPPORT_NAME = "support_{:05d}.png"

_NORMALIZE_CHOICES = ("auto", "on", "off")

# What `--refine-every` means when the polish run wants no refinement at all.
# brush's clap declaration takes a u32 in 1.. — there is no 0 for "never" —
# so the interval is simply put past the run's own length.
_NO_REFINE = 1_000_000

# brush's own enum, and the whole list clap will accept. `ignore` was
# declared here for a long time and is not one of them: it never ran only
# because nothing ever set it.
_ALPHA_MODES = ("transparent", "masked")

# Where the alignment loop runs. `trainer` is b2ctrain's in-process loop
# (--align-iters): the renders, the flow and the warps happen on the GPU
# against the frames it already holds, and the refits continue in the same
# process — no .ply export/reload, no frames rewritten to disk, no
# brush-splat-render invocation per iteration. `pipeline` is this step's own
# loop below (pipeline/align.py, one invocation per iteration), which works
# against any trainer with this CLI and is the A/B reference the loop's
# settings were measured on — the two agree on PSNR and on the shape of the
# trajectory, and it is the in-trainer loop that has NOT been through the
# band-limited sharpness metric the settings were tuned with
# (docs/final-splat-alignment-guide.md). `auto` asks the binary, which is
# also what keeps an image whose trainer predates --align-iters working.
_ALIGN_BACKENDS = ("auto", "trainer", "pipeline")

#: Whether a trainer binary carries the in-process alignment loop, by path.
#: Probed once per path: a --help is cheap but not free, and run() is called
#: twice per workflow.
_ALIGN_PROBE: Dict[str, bool] = {}


def _trainer_aligns(brush_path: str) -> bool:
    """True if `brush_path --help` lists --align-iters, else False.

    A binary that cannot be run at all is a False here rather than an error:
    the missing-binary failure belongs to `_run_brush`, with its hint, and
    not to a probe that only decides which of two working paths to take.
    """
    cached = _ALIGN_PROBE.get(brush_path)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [brush_path, "--help"], capture_output=True, text=True, timeout=30,
        )
        supported = "--align-iters" in (result.stdout + result.stderr)
    except (OSError, subprocess.SubprocessError):
        supported = False
    _ALIGN_PROBE[brush_path] = supported
    return supported


def _use_trainer_alignment(backend: str, brush_path: str) -> bool:
    if backend not in _ALIGN_BACKENDS:
        raise ValueError(
            f"align_backend must be one of {', '.join(_ALIGN_BACKENDS)}, not {backend!r}"
        )
    if backend == "trainer":
        return True
    if backend == "pipeline":
        return False
    return _trainer_aligns(brush_path)


def _forced_alpha_mode(setting: Optional[str]) -> Optional[str]:
    """The mode to force on every view, or None to let brush decide per view.

    `auto` (and an empty value, for a workflow that clears the param) is the
    None case: the flag is a global force, so *not passing it* is what lets
    the export's own layout — a masks/ sidecar here, an embedded alpha there
    — resolve the mode view by view.
    """
    if setting is None or setting in ("", "auto"):
        return None
    if setting not in _ALPHA_MODES:
        raise ValueError(
            f"alpha_mode must be auto, {' or '.join(_ALPHA_MODES)}, got {setting!r}. "
            f"brush accepts only its own two modes and would reject the invocation."
        )
    return setting


def _normalize_masked_loss(setting: str, *, mixed: bool) -> bool:
    """Whether to pass `--normalize-masked-loss`, from the param and the run.

    `auto` is the interesting one: the flag corrects a weighting that only
    becomes a bias when the two alpha modes are in the same run (see the
    module docstring), so "on when the export actually carries both" is
    what it should mean, not "on when there are masks".
    """
    if setting not in _NORMALIZE_CHOICES:
        raise ValueError(
            f"normalize_masked_loss must be one of {', '.join(_NORMALIZE_CHOICES)}, "
            f"got {setting!r}"
        )
    if setting == "auto":
        return mixed
    return setting == "on"


@dataclass(frozen=True)
class _SupportViews:
    """The masked half of a mixed training run: views to fit where their
    mask says to and ignore everywhere else.

    Held together as one object because the four lists have to stay
    parallel and because the naming rules below are what keep brush's
    sidecar matching unambiguous — validating that in `run()` would put it
    two hundred lines from the code that writes the files.

    Empty is the normal case, and falsy: a workflow that wires none of the
    `support_*` inputs builds exactly the export this step always built.
    """

    cameras: List[Any]
    image_names: List[str]
    images: List[np.ndarray]
    masks: List[np.ndarray]
    normal_maps: Optional[List[np.ndarray]]

    def __bool__(self) -> bool:
        return bool(self.image_names)

    @classmethod
    def empty(cls) -> "_SupportViews":
        return cls(cameras=[], image_names=[], images=[], masks=[], normal_maps=None)

    @classmethod
    def from_inputs(
        cls, inputs: Dict[str, Any], train_names: Sequence[str]
    ) -> "_SupportViews":
        """Read and validate the `support_*` inputs against the training ones.

        Every failure here is one that would otherwise surface as a
        confusing training rather than an error: a supporting view with no
        mask is a full-weight view fitting the cull colour as if it were
        the subject, and a name that collides — by stem, not just in full,
        because that is how brush matches a sidecar — silently flips a
        training view to masked and stops it carving the silhouette.
        """
        cameras = inputs.get("support_cameras")
        images = inputs.get("support_images")
        masks = inputs.get("support_masks")
        names = inputs.get("support_image_names")
        normal_maps = inputs.get("support_normal_maps")

        supplied = {
            key: value
            for key, value in (
                ("support_cameras", cameras),
                ("support_images", images),
                ("support_masks", masks),
                ("support_image_names", names),
                ("support_normal_maps", normal_maps),
            )
            if value is not None and len(value) > 0
        }
        if not supplied:
            return cls.empty()

        for required in ("support_cameras", "support_images", "support_masks"):
            if required not in supplied:
                raise ValueError(
                    f"{', '.join(sorted(supplied))} given without {required}. A "
                    f"supporting view needs a camera, a frame and a mask: the mask is "
                    f"the whole difference between 'fit this where I trust it' and a "
                    f"view that fits its background too."
                )

        # Re-read from `supplied` so an empty list means the same as an
        # absent one everywhere below — a workflow that wires an input to a
        # context path holding [] is asking for no supporting views, not for
        # a set of them with no names.
        cameras = supplied["support_cameras"]
        images = supplied["support_images"]
        masks = supplied["support_masks"]
        names = supplied.get("support_image_names")
        normal_maps = supplied.get("support_normal_maps")

        count = len(images)
        for key in ("support_cameras", "support_masks", "support_image_names",
                    "support_normal_maps"):
            value = supplied.get(key)
            if value is not None and len(value) != count:
                raise ValueError(
                    f"{key} has {len(value)} entries but support_images has {count}. "
                    f"Supporting views move together."
                )

        names = list(names) if names is not None else [
            _SUPPORT_NAME.format(i + 1) for i in range(count)
        ]
        _check_names(names, train_names)

        return cls(
            cameras=list(cameras),
            image_names=names,
            images=list(images),
            masks=list(masks),
            normal_maps=list(normal_maps) if normal_maps is not None else None,
        )

    def check_intrinsics(self, train_cameras: Sequence[Any]) -> None:
        """Warn if a supporting view's lens is not the training views'.

        `ColmapExporter._export_cameras` writes ONE camera line — read off
        `cameras[0]`, which is a training view — and stamps `CAMERA_ID 1` on
        every image, supporting views included. A supporting view rendered
        through a different focal length or at a different size is therefore
        exported as though it had the training lens, and brush places it
        wherever that lie puts it: the view lands in the wrong part of the
        model, and nothing in the export says so.

        Nothing in the shipped wiring should trip this — `render_splat`'s
        `bounds_source: splat` dollies in rather than zooming, and
        `pointmap_elevation_views` copies the source camera's intrinsics
        outright, both for exactly this reason. It is here because the
        failure is invisible in the output and cheap to name here.

        A warning rather than a raise: this is a fact about the exporter,
        not about the caller, and a run that has already spent an hour of
        GPU should say so and finish rather than die on it.
        """
        if not self or not train_cameras:
            return
        reference = train_cameras[0]
        expected = (float(reference.fx), float(reference.fy),
                    float(reference.cx), float(reference.cy),
                    int(reference.width), int(reference.height))
        for name, camera in zip(self.image_names, self.cameras):
            actual = (float(camera.fx), float(camera.fy),
                      float(camera.cx), float(camera.cy),
                      int(camera.width), int(camera.height))
            if not np.allclose(actual[:4], expected[:4], rtol=1e-5, atol=1e-3) \
                    or actual[4:] != expected[4:]:
                logger.warning(
                    "brush: supporting view %s has intrinsics "
                    "fx/fy/cx/cy=%.3f/%.3f/%.3f/%.3f at %dx%d but the training "
                    "views are %.3f/%.3f/%.3f/%.3f at %dx%d. COLMAP export "
                    "writes a single camera line taken from the training views "
                    "and stamps it on every image, so this view will be trained "
                    "as though it had the training lens and will land in the "
                    "wrong place.", name, *actual, *expected,
                )
                return

    def write(self, colmap_dir: Path) -> None:
        """Write the supporting frames into an already-exported COLMAP model.

        RGB into `images/` and the mask beside it in `masks/` — never RGBA,
        which is the layout that means *transparent* and would have brush
        premultiply the frame and learn empty space outside the mask.
        """
        if not self:
            return
        images_dir = colmap_dir / "images"
        masks_dir = colmap_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)

        for i, (img, filename) in enumerate(zip(self.images, self.image_names)):
            if img.shape[-1] == 4:
                img = img[..., :3]
            elif img.shape[-1] != 3:
                raise ValueError(
                    f"Unexpected support image channels: {img.shape[-1]} (expected 3 or 4)"
                )
            cv2.imwrite(str(images_dir / filename), img)
            cv2.imwrite(str(masks_dir / _sidecar_name(filename)),
                        mask_to_alpha_u8(self.masks[i]))

        if self.normal_maps is not None:
            normals_dir = colmap_dir / "normals"
            normals_dir.mkdir(exist_ok=True)
            for i, (normal, filename) in enumerate(zip(self.normal_maps, self.image_names)):
                normal_bgr = np.clip(
                    (normal[..., ::-1] + 1.0) / 2.0 * 255.0, 0, 255
                ).astype(np.uint8)
                out = np.dstack([normal_bgr, mask_to_alpha_u8(self.masks[i])])
                cv2.imwrite(str(normals_dir / _sidecar_name(filename)), out)

        logger.info(
            "brush: %d supporting view(s) written as RGB + a masks/ sidecar, so brush "
            "reads them as masked; the training views keep their embedded alpha and "
            "stay transparent",
            len(self.image_names),
        )


def _loss_weights(inputs: Dict[str, Any], count: int) -> Optional[List[np.ndarray]]:
    """The `weights` input, checked against the training views.

    Empty and absent both mean "no sidecars" — an optional workflow read of
    a path nothing wrote is None, and a step that publishes weights for a
    batch of zero views publishes []. A count that disagrees is refused: a
    weight map is matched to a view by name, and there is no right way to
    match five maps to four frames.
    """
    weights = inputs.get("weights")
    if weights is None or len(weights) == 0:
        return None
    if len(weights) != count:
        raise ValueError(
            f"weights has {len(weights)} entries but there are {count} training "
            f"views. A loss-weight map belongs to one view; either give every "
            f"view one or none of them."
        )
    return list(weights)


def write_loss_weights(colmap_dir: Path, image_names: Sequence[str],
                       weights: Optional[Sequence[np.ndarray]]) -> None:
    """Write the training views' loss weights as brush's `weights/` sidecar.

    Greyscale PNGs, 255 = full weight, named the way the `masks/` sidecar is
    so brush matches them to their frames by stem. Nothing is written when
    there are none: the directory's absence is what "every view at weight
    1" looks like to brush.
    """
    if not weights:
        return
    weights_dir = colmap_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    for weight, filename in zip(weights, image_names):
        cv2.imwrite(str(weights_dir / _sidecar_name(filename)), mask_to_alpha_u8(weight))
    logger.info(
        "brush: %d training view(s) carry a weights/ sidecar (per-pixel loss "
        "weight); brush multiplies it into each view's loss on top of its alpha "
        "mode", len(weights),
    )


def _write_align_debug(
    directory: Path, history: List[Dict[str, Any]], image_names: Sequence[str],
    warped: np.ndarray, render: np.ndarray, sample: str,
) -> None:
    """Keep what an alignment iteration did, for a run nobody can repeat cheaply.

    A 30,000-iteration training plus four alignment passes is an hour of
    GPU, and everything the loop touches is transient by construction: the
    warped frames live in the COLMAP `TemporaryDirectory` and go with it,
    and each iteration exports over the same .ply. So a splat that comes
    back torn or soft has, by default, nothing behind it but log lines —
    the same hole this module's crash reports exist to close, multiplied by
    the number of iterations.

    Two things land here. `alignment.json` is the whole history, rewritten
    (not appended to) after every iteration so it is complete even if the
    next one dies: per iteration the settings in force, the batch figures,
    and every view's own — which is what makes ONE bad frame findable
    behind a batch average that looks fine. Beside it, one view's warped
    frame and the render it was warped onto, per iteration: a tear, a
    doubled edge or a warp that ran away are visible in that pair and in
    nothing else. The same view every time, so the iterations compare.

    Cheap on purpose — a few hundred KB of JSON and two PNGs per iteration
    against the 84 MB the .ply would cost. Written under `align_debug_dir`,
    which the workflow points into the `debug/` bundle the result .zip
    already carries.
    """
    directory.mkdir(parents=True, exist_ok=True)
    iteration = history[-1]["iteration"]
    payload = {"views": list(image_names), "sample_view": sample,
               "iterations": history}
    (directory / "alignment.json").write_text(json.dumps(payload, indent=1))
    stem = Path(sample).stem
    cv2.imwrite(str(directory / f"iter{iteration}_{stem}_warped.png"), warped)
    cv2.imwrite(str(directory / f"iter{iteration}_{stem}_render.png"), render)


def _flow_schedule(values: Sequence[float], name: str, iters: int) -> List[float]:
    """One flow setting per alignment iteration, first to last.

    A single entry holds the setting constant for the whole loop, which is
    what both of these default to; a full-length list schedules it. The case
    that motivates scheduling is the measured one: aligning against a
    converged render tolerates — and rewards — a finer, longer-reaching
    field than aligning against the blurry cold start does, so
    `align_flow_sigma: [6, 6, 3, 3]` with `align_flow_cap: [6, 6, 12, 12]`
    is the shape to try (FINDINGS: sigma 3 / cap 12 from an already-aligned
    splat reads face 27.0 against 26.7, at unchanged fidelity).

    Any other length is refused, and refused up front: the alternative is
    an alignment that runs three iterations and dies on the fourth, an hour
    of training later.
    """
    if len(values) not in (1, iters):
        raise ValueError(
            f"{name} has {len(values)} entries, but align_iters is {iters} — "
            f"give one value per alignment iteration, or a single one to hold "
            f"it constant for the whole loop."
        )
    schedule = [float(value) for value in values]
    return schedule * iters if len(schedule) == 1 else schedule


def _check_align_sizes(images: Sequence[np.ndarray], cameras: Sequence[Any]) -> None:
    """Refuse an alignment whose frames and cameras describe different images.

    The loop measures the flow between a training frame and a render made
    from that frame's camera, so the two have to be the same size — and if
    they are not, the export handed to brush was already describing its own
    frames wrongly (the upscale rescaling the intrinsics is exactly this
    hazard; see the workflow's stage 5). Cheap here, and invisible
    afterwards: a mismatch would come back as an alignment that quietly made
    the splat worse.
    """
    for image, camera in zip(images, cameras):
        height, width = image.shape[:2]
        if (width, height) != (int(camera.width), int(camera.height)):
            raise ValueError(
                f"align_iters is set, but a training frame is {width}x{height} "
                f"where its camera describes {int(camera.width)}x"
                f"{int(camera.height)}. The alignment renders each camera and "
                f"measures the flow to its frame, which needs the two to agree."
            )


def _link_init_ply(colmap_dir: Path, ply_path: Path) -> None:
    """Point the export's `init.ply` at a trained splat, for a warm start.

    brush initialises from the `init.ply` of the dataset it is handed (its
    `formats/mod.rs` prefers that name over any other .ply present), so
    this — not a flag — is how a second invocation resumes from the first
    one's export. A symlink rather than a copy: the .ply is hundreds of MB,
    it is read once at load and the polish run's own export goes to
    `export_dir`, not here.
    """
    init = colmap_dir / "init.ply"
    if init.is_symlink() or init.exists():
        init.unlink()
    init.symlink_to(ply_path.absolute())


def _sidecar_name(filename: str) -> str:
    """The `masks/` (or `normals/`) name brush will match to `filename`.

    Always .png: these are written by this step, and a mask has no business
    going through a lossy codec.
    """
    return Path(filename).with_suffix(".png").name


def _check_names(support_names: Sequence[str], train_names: Sequence[str]) -> None:
    """Refuse names that would make a sidecar ambiguous.

    brush matches `masks/x.*` to an image whose *stem* is `x` as well as to
    one whose full name is `x`, so `support.jpg` and `support.png` in the
    same export would share one mask — and if the collision is with a
    training view, that view silently becomes masked and stops carving the
    silhouette. Cheap to check here, invisible in a trained splat.
    """
    seen: Dict[str, str] = {}
    for name in list(train_names) + list(support_names):
        if not name:
            raise ValueError("A view name is empty; brush resolves frames by name.")
        stem = Path(name).stem
        if stem in seen:
            raise ValueError(
                f"View names {seen[stem]!r} and {name!r} share the stem {stem!r}. "
                f"brush matches a masks/ sidecar by stem as well as by full name, so "
                f"the two would share one mask — rename one of them."
            )
        seen[stem] = name


@register_step("brush")
class BrushStep(Step):
    """Train a 3D Gaussian Splat using the brush CLI tool.

    inputs: {"cameras": List[Camera], "image_names": List[str],
             "points_3d": Tuple[np.ndarray, np.ndarray],
             "images": List[np.ndarray] BGR(A),
             "masks": Optional[List[np.ndarray]] float32 [0,1], foreground=1,
             "normal_maps": Optional[List[np.ndarray]] HxWx3 float32 [-1,1],
             "weights": Optional[List[np.ndarray]] float32 [0,1], a per-pixel
                        loss weight per training view (weights/ sidecar),
             "support_cameras": Optional[List[Camera]],
             "support_images": Optional[List[np.ndarray]] BGR(A),
             "support_masks": Optional[List[np.ndarray]] float32 [0,1],
             "support_image_names": Optional[List[str]],
             "support_normal_maps": Optional[List[np.ndarray]]}
    outputs: {"splat_path": str}

    The `support_*` inputs are the masked half of a mixed run — extra
    views trained on only where their mask says to (see the module
    docstring). They are optional and independent of the training views:
    a run without them builds byte-identical training data to before.

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
        Param("polish_steps", int, 0,
              "Iterations of a second, growth-off warm start after the main training, "
              "exported over the same .ply. Measured 2026-09-05 on the intermediate "
              "splat: 9000 of these moved band-limited face sharpness 143 -> 161 and "
              "every other part with it, where 9000 extra iterations of ONE cold run "
              "gave a third of that — the restart at full mean-LR is the effect, not "
              "the iteration count. 0 is off, which is what a training nobody has "
              "measured it on should stay at (docs/intermediate-splat-guide.md)",
              minimum=0),
        Param("align_iters", int, 4,
              "Alignment iterations after the main training: render the splat at "
              "the training cameras, warp each ORIGINAL frame onto its own render "
              "by the smoothed optical flow between them, and resume training on "
              "the aligned set with growth off. This is the answer to a fit that "
              "blurs its own training data by averaging views that disagree about "
              "where texture sits (see the module docstring). Measured 2026-09-06 "
              "on the deliverable splat: band-limited face sharpness 21.1 -> 23.8 "
              "over four iterations (+1.2, +0.6, +0.5, +0.4) with fidelity rising "
              "27.64 -> 28.41 dB, and a fifth and sixth worth +0.2 each — 4 is the "
              "saturation point. Measured on a 4070 Ti at 81 views of 1080x1920: "
              "9 s to render, 10 s to measure and apply the flow, and the "
              "fine-tune on top — about a minute an iteration. 0 is off "
              "(docs/final-splat-alignment-guide.md)",
              minimum=0),
        Param("align_steps", int, 3000,
              "Iterations of the growth-off fine-tune each alignment pass runs. "
              "1000 measured identical AT THE FIXED POINT (47 s -> 15 s per "
              "iteration), but every iteration still climbing was measured at "
              "3000, so this is what the trajectory above was made of",
              minimum=1, advanced=True),
        Param("align_flow_sigma", list, [6.0],
              "Gaussian smoothing of the flow field, in pixels — what makes the "
              "warp a texture correction rather than a per-pixel scramble. One "
              "entry per alignment iteration, first to last; a single entry (the "
              "default) holds it at 6 for the whole loop, which is what the "
              "trajectory above was measured with. Schedule it to sharpen the "
              "field as the render it is measured against converges: [6, 6, 3, 3] "
              "beside a cap of [6, 6, 12, 12]",
              advanced=True),
        Param("align_flow_cap", list, [6.0],
              "Largest displacement the alignment applies, in pixels; beyond it "
              "the field is scaled down with its direction kept. Same per-iteration "
              "shape as align_flow_sigma. Raising it to 12 on its own was measured "
              "to change nothing — the residual it would reach is views disagreeing "
              "about hand POSE, which no image warp fixes — but 12 alongside a "
              "sigma of 3 is the one combination that read better",
              advanced=True),
        Param("align_backend", str, "auto",
              "Where the alignment loop runs: `trainer` (b2ctrain's in-process loop — "
              "one invocation; renders, flow and warps on the GPU against the frames it "
              "already holds, refits in the same process; ~0.7 s per pass instead of a "
              "render process, a Python flow and a re-invocation), `pipeline` (this "
              "step's loop: brush-splat-render + pipeline/align.py + one invocation per "
              "iteration — trainer-agnostic, and the reference the loop's settings were "
              "measured on), or `auto` (trainer when the binary's --help lists "
              "--align-iters)", advanced=True),
        Param("align_debug_dir", str, None,
              "Keep each alignment iteration's evidence here: alignment.json (the "
              "settings in force, the batch figures and EVERY view's own, rewritten "
              "after each iteration so it survives a crash in the next one) plus one "
              "view's warped frame and the render it was warped onto. Everything the "
              "loop touches is otherwise transient — the warped frames go with the "
              "COLMAP temp directory and each iteration exports over the same .ply — "
              "so without this a torn or soft result has nothing behind it but log "
              "lines, on a run that costs an hour of GPU. A few hundred KB and two "
              "PNGs an iteration; empty writes nothing",
              advanced=True),
        Param("match_alpha_weight", float, 0.1,
              "Weight of brush's L1 loss on a transparent view's alpha — how hard the "
              "silhouette is fitted to the mask. 0.1 is brush's own default; 0.5 "
              "measured 17% fewer dark wedges (dark splats in the concave gaps a flat "
              "orbit never sees into) at no cost in sharpness and -0.003 IoU",
              minimum=0.0, advanced=True),
        Param("alpha_mode", str, "auto",
              "Force brush to read EVERY view's alpha channel this way, flattening any "
              "mix. auto (the default) lets brush decide per view from the export's "
              "layout — a masks/ sidecar means masked ('ignore outside it'), an alpha "
              "channel in the frame itself means transparent ('nothing is there') — "
              "which is what lets supporting views train alongside the rendered ones. "
              "The training views this step writes are RGBA either way, so auto is what "
              "the old forced 'transparent' did on a run with no support_* views",
              choices=("auto", "transparent", "masked"), advanced=True),
        Param("normalize_masked_loss", str, "auto",
              "Divide a masked view's loss by its mask coverage, so it is not weighted "
              "down by the fraction of the frame its mask covers (brush's "
              "--normalize-masked-loss). auto: on exactly when the export carries both "
              "alpha modes, which is the run where the weighting is a systematic bias "
              "rather than a harmless rescale. Exact for a binary mask, approximate for "
              "a soft one",
              choices=("auto", "on", "off"), advanced=True),
        Param("growth_grad_threshold", float, None,
              "brush's densification threshold — lower grows faster. Empty leaves "
              "brush's own 0.0025. Together with growth_select_fraction 0.4 and "
              "growth_stop_iter 24000 this is the 'dense growth' setting measured "
              "2026-09-06 (0.0012): +40% face sharpness on its own, +2.5 s1 on top "
              "of the alignment loop — and 1.68M splats against 356k, a 424 MB .ply "
              "against 84. A deliberate quality-for-size purchase, which is why it "
              "is off (docs/final-splat-alignment-guide.md §2)",
              minimum=0.0, advanced=True),
        Param("growth_select_fraction", float, None,
              "Fraction of the splats above the threshold that actually grow. "
              "Empty leaves brush's own 0.25; the dense-growth setting is 0.4",
              minimum=0.0, maximum=1.0, advanced=True),
        Param("growth_stop_iter", int, None,
              "Step at which growth stops. Empty leaves brush's own 15000; the "
              "dense-growth setting is 24000. Growth belongs to the COLD START, "
              "which is why the alignment and polish invocations force it to 0 "
              "whatever this says. It does stack with alignment — an earlier "
              "reading that it did not (21.7 with growth against 22.1 without) "
              "turned out to be a property of the TARGET the flow was measured "
              "against, not of resampling: growing on frames aligned to a blurry "
              "cold-start render amplifies the noise in that flow, where frames "
              "aligned to a converged one give 24.7 against 22.5 "
              "(docs/final-splat-alignment-guide.md §2)",
              minimum=0, advanced=True),
        Param("normal_loss_strength", float, 0.05,
              "Weight on the normal-map supervision loss; 0 disables it", minimum=0.0),
        Param("normal_loss_step_start", int, 5000,
              "Step at which normal supervision switches on", minimum=0),
        Param("normal_loss_every", int, 1,
              "Evaluate the normal loss every Nth step instead of every step. brush "
              "scales the sampled loss by N, so the expected gradient is unchanged "
              "and only the extra normal render in between is skipped; 1 is brush's "
              "own default", minimum=1),
        Param("export_evidence", bool, True,
              "Measure each splat's multi-view evidence against every training view "
              "after the last step and write it into the exported .ply as ev_* vertex "
              "properties. That is what render_splat's `confidence` mode reads, and "
              "having it in the .ply is what lets that render need no dataset. Costs "
              "seconds (~2s for 100k splats x 81 views) and every other .ply reader "
              "ignores the extra properties, so it is on for both trainings"),
        Param("evidence_prune_inmask", float, None,
              "Drop splats whose in-mask contribution fraction is below this, and "
              "those no view supported at all, before the export. Implies the "
              "evidence pass. 0.1-0.3 are sane values; empty (the default) prunes "
              "nothing, because this has not been looked at on a real run yet and a "
              "splat dropped here is gone from the deliverable .ply, not merely "
              "hidden in one render", advanced=True),
        Param("evidence_normal_weight", float, 0.0,
              "Fold w * the normal-map residual into the evidence residual, for a "
              "dataset that has normals/. Costs one extra render per view and is "
              "untuned; 0 leaves the residual photometric", minimum=0.0,
              advanced=True),
        Param("output_dir", str, None,
              "Puts this training under <output_dir>/brush/training_<ms>/ — what an "
              "intermediate training wants. Empty falls back to the system temp dir, "
              "where the .ply only survives because Dataset.to_disk copies it at the "
              "end of the run"),
        Param("export_dir", str, None,
              "Export straight into this directory instead, for a training whose .ply "
              "is a deliverable and needs a predictable path. Wins over output_dir"),
        Param("export_name", str, "export.ply", "Filename of the exported .ply"),
        Param("brush_path", str, "b2ctrain",
              "The trainer binary, on PATH or as an absolute path. `b2ctrain` is "
              "what the image ships and this default names; any binary with "
              "brush's CLI works, including the Erant/brush fork it replaced",
              advanced=True),
        Param("render_path", str, _RENDER_BINARY,
              "The rasteriser the alignment loop renders the current splat with, "
              "on PATH or as an absolute path. Same convention as brush_path, and "
              "unused when align_iters is 0 or the loop runs inside the trainer, "
              "which renders with its own", advanced=True),
        Param("with_viewer", bool, False,
              "Pass --with-viewer, which opened brush's interactive viewer window "
              "and needs a display. b2ctrain accepts the flag and ignores it — it "
              "has no viewer — so this does nothing on the shipped trainer",
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
        weights = _loss_weights(inputs, len(images))
        support = _SupportViews.from_inputs(inputs, image_names)

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
        polish_steps = params["polish_steps"]
        align_iters = params["align_iters"]
        align_steps = params["align_steps"]
        align_flow_sigma = params["align_flow_sigma"]
        align_flow_cap = params["align_flow_cap"]
        align_debug_dir = params["align_debug_dir"]
        align_backend = params["align_backend"]
        growth_grad_threshold = params["growth_grad_threshold"]
        growth_select_fraction = params["growth_select_fraction"]
        growth_stop_iter = params["growth_stop_iter"]
        match_alpha_weight = params["match_alpha_weight"]
        alpha_mode = _forced_alpha_mode(params["alpha_mode"])
        # Resolved here rather than at the argv, so a mistyped setting is a
        # refusal before several hundred MB of frames are written out.
        normalize_masked_loss = _normalize_masked_loss(
            params["normalize_masked_loss"], mixed=bool(support) and not alpha_mode
        )
        normal_loss_strength = params["normal_loss_strength"]
        normal_loss_step_start = params["normal_loss_step_start"]
        normal_loss_every = params["normal_loss_every"]
        export_evidence = params["export_evidence"]
        evidence_prune_inmask = params["evidence_prune_inmask"]
        evidence_normal_weight = params["evidence_normal_weight"]
        with_viewer = params["with_viewer"]
        render_path = params["render_path"]

        if align_iters > 0:
            # Both resolved here, with the size check, rather than at the
            # iteration that reads them: a mistyped schedule should be a
            # refusal now and not after an hour of cold-start training.
            align_flow_sigma = _flow_schedule(
                align_flow_sigma, "align_flow_sigma", align_iters)
            align_flow_cap = _flow_schedule(
                align_flow_cap, "align_flow_cap", align_iters)
            _check_align_sizes(images, cameras)

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

            # The supporting views are part of the same COLMAP model — one
            # cameras.txt/images.txt covering both — and differ only in how
            # their frames are written below.
            support.check_intrinsics(cameras)
            ColmapExporter(
                cameras=list(cameras) + support.cameras,
                image_names=list(image_names) + support.image_names,
                points_3d=points_3d,
            ).export(output_dir=colmap_dir)

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

            def training_frames() -> List[np.ndarray]:
                """The training views exactly as they go to disk, BGR(A) uint8.

                Built from the step's own inputs every time it is called, so
                it is always the PRISTINE set — which is what the alignment
                loop below warps from, iteration after iteration. Nothing is
                cached: at 81 frames of 1080x1920 the RGBA copies are ~670 MB,
                and `images` and `masks` are already in memory to build them
                from.
                """
                frames = []
                for i, img in enumerate(images):
                    if alpha_channel is None:
                        frames.append(img)
                        continue
                    if img.shape[-1] == 4:
                        rgba = img.copy()
                        rgba[..., 3] = alpha_channel[i]
                    elif img.shape[-1] == 3:
                        rgba = np.dstack([img, alpha_channel[i]])
                    else:
                        raise ValueError(f"Unexpected image channels: {img.shape[-1]} (expected 3 or 4)")
                    frames.append(rgba)
                return frames

            def write_frames(frames: Sequence[np.ndarray]) -> None:
                for frame, filename in zip(frames, image_names):
                    cv2.imwrite(str(images_dir / filename), frame)

            write_frames(training_frames())

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

            write_loss_weights(colmap_dir, image_names, weights)
            support.write(colmap_dir)

            ply_output_name = params["export_name"]
            ply_path = out_root / ply_output_name

            def command(*, total: int, refine: int, normal_start: int,
                        normal_weight: Optional[float] = None,
                        growth_stop: Optional[int] = None,
                        align: Optional[List[str]] = None) -> List[str]:
                """One brush invocation's argv.

                A function rather than a literal because the polish and
                alignment runs below are the same command with a handful of
                values changed, and every one of them is a flag brush would
                REFUSE TWICE — clap rejects a repeated `--refine-every`, so
                an argv built by appending overrides to this list would not
                run at all.

                `normal_weight` overrides `normal_loss_strength` for the run
                being built. The alignment iterations pass 0: their frames
                have been resampled and no longer line up with the
                `normals/` sidecar beside them, and unlike `normal_start`
                there is no value of the start iteration that turns the term
                off for a run that is only 3000 steps long.
                """
                cmd = [
                    brush_path,
                    str(colmap_dir),
                    "--total-train-iters", str(total),
                    "--sh-degree", str(sh_degree),
                    "--export-path", str(out_root.absolute()),
                    "--export-name", ply_output_name,
                    "--export-every", str(total),
                    "--max-resolution", str(max_resolution),
                    "--max-splats", str(max_splats),
                    "--refine-every", str(refine),
                    "--match-alpha-weight", str(match_alpha_weight),
                ]
                # The growth block. `growth_stop` is the caller's override —
                # the polish and alignment runs force 0 — and the params
                # below it are brush's own defaults unless a workflow buys
                # dense growth (see their help).
                stop = growth_stop if growth_stop is not None else growth_stop_iter
                if stop is not None:
                    cmd.extend(["--growth-stop-iter", str(stop)])
                if growth_grad_threshold is not None:
                    cmd.extend(["--growth-grad-threshold", str(growth_grad_threshold)])
                if growth_select_fraction is not None:
                    cmd.extend(["--growth-select-fraction", str(growth_select_fraction)])
                if with_viewer:
                    cmd.append("--with-viewer")
                # Not passed unless a caller explicitly asked for one:
                # --alpha-mode is a global force, so passing it is what
                # *prevents* the mixed run the layout above sets up. What the
                # old unconditional `--alpha-mode transparent` bought was
                # nothing — an RGBA frame with no sidecar already loads as
                # transparent — which is why dropping it leaves every shipped
                # workflow training on byte-identical data.
                if alpha_mode:
                    cmd.extend(["--alpha-mode", alpha_mode])
                if normalize_masked_loss:
                    cmd.append("--normalize-masked-loss")
                if normal_maps is not None:
                    weight = (normal_loss_strength if normal_weight is None
                              else normal_weight)
                    cmd.extend([
                        "--normal-loss-weight", str(weight),
                        "--normal-loss-start-iter", str(normal_start),
                        "--normal-loss-every", str(normal_loss_every),
                    ])
                # The evidence block. Only the LOD-0 final export carries it,
                # which is this step's case (no --lod-levels is passed, so brush
                # exports one level). --evidence-prune-inmask implies the
                # measurement, but --export-evidence is passed anyway when both
                # are set: the flag is what says the properties end up IN the
                # .ply, and the two are independent on the brush side.
                if export_evidence:
                    cmd.append("--export-evidence")
                if evidence_prune_inmask is not None:
                    cmd.extend(["--evidence-prune-inmask", str(evidence_prune_inmask)])
                if evidence_normal_weight > 0:
                    cmd.extend(["--evidence-normal-weight", str(evidence_normal_weight)])
                # The in-trainer alignment loop (b2ctrain): the same
                # iterations, steps and flow schedule the loop below would
                # run, carried on the cold run's own argv.
                if align:
                    cmd.extend(align)
                return cmd

            if alpha_mode and support:
                logger.warning(
                    "brush: alpha_mode=%s forces all %d views to that mode, "
                    "including the %d supporting view(s) whose masks/ sidecars "
                    "would otherwise have made them masked. Leave alpha_mode "
                    "at auto to train on the mix.",
                    alpha_mode, len(image_names) + len(support.image_names),
                    len(support.image_names),
                )

            in_trainer = align_iters > 0 and _use_trainer_alignment(align_backend, brush_path)
            align_flags: Optional[List[str]] = None
            if in_trainer:
                align_flags = [
                    "--align-iters", str(align_iters),
                    "--align-steps", str(align_steps),
                    "--align-flow-sigma", ",".join(str(v) for v in align_flow_sigma),
                    "--align-flow-cap", ",".join(str(v) for v in align_flow_cap),
                ]
                if align_debug_dir:
                    align_flags.extend(["--align-debug-dir", str(align_debug_dir)])
                logger.info(
                    "brush: the alignment loop (%d iteration(s) of %d steps) runs "
                    "inside the trainer — it renders, flows and warps the views it "
                    "already holds and refits in the same process; alignment.json "
                    "and the sample frames%s are written by the trainer",
                    align_iters, align_steps,
                    f" under {align_debug_dir}" if align_debug_dir else "",
                )
                if polish_steps > 0:
                    # The one place the two backends genuinely differ, and
                    # it is silent: the pipeline loop leaves its last
                    # iteration's warped frames in the export's `images/`,
                    # so the polish below resumes on the frames the splat
                    # was aligned to. The trainer's warps never touch disk
                    # — they live in its GPU views — so a polish after it
                    # is a resume on the PRISTINE originals, which pulls
                    # the fit back toward the disagreement the alignment
                    # just removed. No shipped workflow asks for both
                    # (stage 2 polishes and does not align, stage 5 aligns
                    # and does not polish); if one ever does, run the loop
                    # with `align_backend: pipeline` or polish first.
                    logger.warning(
                        "brush: polish_steps is %d and the alignment loop runs "
                        "inside the trainer, so the polish will resume on the "
                        "UNWARPED original frames — the trainer's warps are never "
                        "written to the dataset. Set align_backend: pipeline if the "
                        "polish should see the aligned frames.",
                        polish_steps,
                    )
            self._run_brush(
                command(total=total_steps, refine=refine_every,
                        normal_start=normal_loss_step_start, align=align_flags),
                ply_path, colmap_dir=colmap_dir,
            )

            # The alignment loop. Each iteration renders the splat as it
            # stands at the training cameras, warps the ORIGINAL frames onto
            # those renders (pipeline/align.py, and see the module
            # docstring's invariant), and resumes training on the aligned
            # set with growth off — the same warm start the polish makes,
            # for a different reason.
            #
            # In here rather than in the workflow because the COLMAP export
            # is this method's TemporaryDirectory: a workflow-level loop
            # would re-export several hundred MB of frames per iteration and
            # would have no way to hand brush an init.ply at all.
            if align_iters > 0 and not in_trainer:
                if support:
                    logger.info(
                        "brush: the %d supporting view(s) are left as they are; "
                        "the alignment warps the %d training views only",
                        len(support.image_names), len(image_names),
                    )
                history: List[Dict[str, Any]] = []
                for iteration in range(1, align_iters + 1):
                    sigma = align_flow_sigma[iteration - 1]
                    cap = align_flow_cap[iteration - 1]
                    renders = self._render_training_views(
                        ply_path, cameras, image_names, render_path=render_path,
                    )
                    frames, stats = align_views(
                        training_frames(), renders, sigma=sigma, cap=cap,
                    )
                    write_frames(frames)
                    history.append({
                        "iteration": iteration, "sigma": sigma, "cap": cap,
                        "align_steps": align_steps,
                        "mean": stats.mean, "p90": stats.p90,
                        "per_view": [
                            {"name": name, "mean": view.mean, "p90": view.p90}
                            for name, view in zip(image_names, stats.views)
                        ],
                    })
                    if align_debug_dir:
                        _write_align_debug(
                            Path(align_debug_dir), history, image_names,
                            frames[0], renders[0], image_names[0],
                        )
                    # Both sets are on disk now, and at 81 views of
                    # 1080x1920 they are well over a gigabyte between them —
                    # not something to hold through a training run for no
                    # reason, on a box that also has to fit the trainer.
                    del renders, frames
                    logger.info(
                        "brush: alignment %d/%d — the training views disagreed with "
                        "their own renders by %.2f px mean, %.2f px p90 (smoothed "
                        "at sigma %.1f, capped at %.1f); refitting for %d steps, "
                        "growth off",
                        iteration, align_iters, stats.mean, stats.p90,
                        sigma, cap, align_steps,
                    )
                    if stats.p90 >= cap:
                        # Mechanical, and quiet on a healthy run: the
                        # reference loop measured a p90 of 2.5-3.3 px
                        # against a cap of 6. At or above it, a tenth of
                        # the average frame wanted to move further than
                        # the clamp allows, so what the iteration applied
                        # is the cap rather than the measurement.
                        logger.warning(
                            "brush: alignment %d/%d is CAP-BOUND — the measured "
                            "disagreement (p90 %.2f px) is at or past the %.1f px "
                            "cap, so the warp is limited by the clamp and not by "
                            "the data. Raising the cap alone was measured not to "
                            "help hands (their residual is pose, not texture); a "
                            "p90 this high on the BODY is worth looking at, and "
                            "align_debug_dir keeps the frame to look at it with.",
                            iteration, align_iters, stats.p90, cap,
                        )
                    _link_init_ply(colmap_dir, ply_path)
                    self._run_brush(
                        command(total=align_steps, refine=_NO_REFINE,
                                normal_start=0, normal_weight=0.0, growth_stop=0),
                        ply_path, colmap_dir=colmap_dir,
                    )

                # The trajectory in one line, because the per-iteration
                # lines are scattered through several thousand lines of
                # training output and the SHAPE is the thing worth seeing.
                # Read it against the reference loop, which went
                # 1.02 -> 1.18 -> 1.26 -> 1.31 px while sharpness rose
                # 21.1 -> 23.8: a rising, DECELERATING measurement is what
                # a working loop looks like — a sharper render gives DIS
                # more to lock onto, so it resolves a displacement the
                # blurry cold start under-reports. What that shape does not
                # do is accelerate, and it does not run at the cap.
                logger.info(
                    "brush: alignment finished — measured disagreement %s px "
                    "across %d iteration(s) (reference: 1.02 -> 1.18 -> 1.26 -> "
                    "1.31, rising and decelerating, while sharpness rose)%s",
                    " -> ".join(f"{entry['mean']:.2f}" for entry in history),
                    align_iters,
                    f"; per-view figures and a sample frame in {align_debug_dir}"
                    if align_debug_dir else "",
                )

            # The polish: the same training resumed from its own export with
            # growth off, at full mean-LR rather than the decayed tail of the
            # first run — which is the whole effect (a plain longer cold run
            # gives a third of it). brush picks the initial splats up from an
            # `init.ply` sitting in the dataset directory, so the link below
            # is the entire handover; the export path is unchanged, so the
            # polished .ply lands over the first one and every reader
            # downstream — including the evidence the confidence render gates
            # on — sees only the finished splat.
            if polish_steps > 0:
                _link_init_ply(colmap_dir, ply_path)
                logger.info(
                    "brush: polishing for %d more iterations from %s, growth off",
                    polish_steps, ply_path.name,
                )
                self._run_brush(
                    command(total=polish_steps, refine=_NO_REFINE,
                            normal_start=0, growth_stop=0),
                    ply_path, colmap_dir=colmap_dir,
                )

        if not ply_path.exists():
            raise RuntimeError(
                f"Expected output PLY file not found: {ply_path}\nBrush may not have exported successfully."
            )

        return {"splat_path": str(ply_path.absolute())}

    def _render_training_views(
        self, ply_path: Path, cameras: Sequence[Any], image_names: Sequence[str],
        *, render_path: str,
    ) -> List[np.ndarray]:
        """The splat as it stands, rendered at the training views' own cameras.

        BGR uint8 composited over `align.BACKGROUND` — the same grey the
        frames are flattened onto before the flow is measured, so the
        silhouette edge contributes the same thing on both sides of it.

        `scene=None` is safe and deliberate: `_rasterize` only serialises a
        scene when there is no .ply on disk to render, and here there always
        is one — the training that just finished wrote it. Loading a
        hundreds-of-MB export into Python to hand it straight back to the
        rasteriser would be pure cost.
        """
        images, _ = _rasterize(
            scene=None,
            splat_path=str(ply_path),
            cameras=list(cameras),
            image_names=list(image_names),
            width=int(cameras[0].width),
            height=int(cameras[0].height),
            bg_color=ALIGN_BACKGROUND,
            render_path=render_path,
            confidence=None,
        )
        return images

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
                    "The trainer is built into the image at /usr/local/bin/b2ctrain; "
                    "on a bare machine, build Erant/b2ctrain (cmake -S . -B build "
                    "-DCMAKE_BUILD_TYPE=Release && cmake --build build) or point the "
                    "step's brush_path param at a binary with brush's CLI."
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
    for name in ("images", "masks", "normals", "weights"):
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
