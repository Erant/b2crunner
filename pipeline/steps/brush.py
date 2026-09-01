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

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

# Only used when the caller supplies no names of its own. Kept distinct
# from the `frame_NNNNN_` the renderers produce so that a glance at a crash
# directory's images/ says which views were supporting ones.
_SUPPORT_NAME = "support_{:05d}.png"

_NORMALIZE_CHOICES = ("auto", "on", "off")

# brush's own enum, and the whole list clap will accept. `ignore` was
# declared here for a long time and is not one of them: it never ran only
# because nothing ever set it.
_ALPHA_MODES = ("transparent", "masked")


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

            support.write(colmap_dir)

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
            # Not passed unless a caller explicitly asked for one:
            # --alpha-mode is a global force, so passing it is what
            # *prevents* the mixed run the layout above sets up. What the
            # old unconditional `--alpha-mode transparent` bought was
            # nothing — an RGBA frame with no sidecar already loads as
            # transparent — which is why dropping it leaves every shipped
            # workflow training on byte-identical data.
            if alpha_mode:
                cmd.extend(["--alpha-mode", alpha_mode])
                if support:
                    logger.warning(
                        "brush: alpha_mode=%s forces all %d views to that mode, "
                        "including the %d supporting view(s) whose masks/ sidecars "
                        "would otherwise have made them masked. Leave alpha_mode "
                        "at auto to train on the mix.",
                        alpha_mode, len(image_names) + len(support.image_names),
                        len(support.image_names),
                    )
            if normalize_masked_loss:
                cmd.append("--normalize-masked-loss")
            if normal_maps is not None:
                cmd.extend([
                    "--normal-loss-weight", str(normal_loss_strength),
                    "--normal-loss-start-iter", str(normal_loss_step_start),
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
    for name in ("images", "masks", "normals"):
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
