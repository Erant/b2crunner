"""refine_cameras — bundle-adjust the generated orbit before it is trained on.

The poses this pipeline hands `brush` were never measured. Every camera in
a helical or circular run is *generated* — an exactly ideal orbit, angular
step 4.500 +- 0.039 deg, camera height y = 0 with std exactly 0, radius
1.8036 +- 0.0158 — and the frames were then produced by a diffusion model
that had its own opinion about where the subject was. So there is
something real to correct, and correcting it is worth **+0.63 PSNR / +0.0045
SSIM** on held-out views (30k iters, three seeds per condition, seed spread
+-0.003 on the baseline). The measurement, and everything else this module
is a port of, is docs/camera-pose-refinement.md.

Do not confuse this with `steps/pose_refine.py`, which is about *body*
pose — re-posing a SAM-3D-Body fit so its mesh agrees with the shell. This
is about *camera* pose. The two share nothing but the word.

`rebase_cameras`, the second step in this module, is the other half of
correcting poses under a running pipeline: the correction below moves the
frames' cameras, and anything already BUILT from one of those cameras — the
face splat's supporting views, which are renders unprojected from the
anchor photograph — has to be carried across with it or it goes on
describing a world nothing else believes in. See that class's docstring.

The recipe, in six parts
------------------------
1.  **Alpha -> COLMAP masks.** Threshold the foreground matte at 127, one
    PNG per frame. COLMAP ignores black pixels (`--ImageReader.mask_path`).
2.  **An input model from the given poses** — cameras.txt and images.txt as
    the dataset holds them, plus an *empty* points3D.txt, which is exactly
    what `point_triangulator` wants.
3.  **ALIKED_N32 + ALIKED_LIGHTGLUE, exhaustive.** A white cyclorama and
    bare skin are low-texture and SIFT thins out on them; ALIKED holds up
    (~465 inliers on adjacent pairs, 1365 verified pairs at 81 frames).
4.  **Triangulate, then N x (bundle adjust -> retriangulate).** One BA pass
    strands observations; a fresh triangulation between passes lets tracks
    reform against the corrected poses. Converges by the third round.
5.  **Put the gauge back** — a Sim(3) (Umeyama, with scale) from the refined
    camera centres onto the given ones. Non-negotiable; see trap 1.
6.  Rebuild `Camera` objects from the aligned poses, intrinsics untouched.

**Foreground-only, deliberately.** Letting features land on the backdrop
too measured 0.089 PSNR better, but a fourth run of the same foreground
pipeline with a different RANSAC draw moved 0.075 on nondeterminism alone —
the two regions are not distinguishable at that sample size, while
refining *at all* is worth ten times the difference between them. And the
background is the half that does not travel: here it is a static studio
backdrop that happens to be rigid and supplies only ~15% of the keypoints
anyway, but one that moves, or is generated inconsistently per frame,
would quietly poison the solve with nothing in the output to say so.

Trap 1 — BA inflates the model by 15-26%, silently
--------------------------------------------------
`bundle_adjuster` fixes the gauge with `TWO_CAMS_FROM_WORLD`: the first
camera's pose plus part of the second's. Global scale therefore hangs off
one cam1<->cam2 baseline and the rest of the reconstruction is free to grow
around it — +15.1%, +21.6%, +19.3%, +23.5% across four runs of the scratch
script this ports, and +21.7%, +23.2%, +26.4% across three of this step.
That last one is outside the range the first four suggested, so treat the
figure as unbounded rather than as 20%-ish: what matters is that step 5
removes it exactly, not that it stays small. Nothing warns you: the reconstruction stays self-consistent,
reprojection error looks fine, `model_analyzer` is happy, and what you get
is a splat a fifth too large in a frame whose `points3D.txt` init no longer
matches it. Step 5 is the fix, and `_check_scale` below is the assertion
that it happened.

Trap 2 — does not apply here, by construction
---------------------------------------------
The scratch script this ports wrote its intermediates into the output
dataset, brush scanned that directory recursively for `cameras.txt`, found
an un-aligned +19% model and trained on it: PSNR 4.45, exit code 0, no
warning anywhere. This step publishes `List[Camera]` into the run context
and does its work in a `TemporaryDirectory` that is not a dataset and is
not under one, so there is nothing for a loader to find. `work_dir` keeps
that scratch for inspection and is likewise not a dataset directory.

Trap 3 — do not let intrinsics float
------------------------------------
Unfreezing focal length on this data drove fx=1105.48 / fy=837.93 from a
given 1066.17 / 1066.17 — a 24% pixel aspect no real camera has — bought
for 0.017 px of mean reprojection error. BA absorbs multi-view
inconsistency into the camera model and warps geometry to do it. Every
intrinsic is frozen here and there is no param to unfreeze one.

What the ceiling is
-------------------
Mean reprojection error 1.630 -> 1.579 px, median 1.485 -> 1.416. It bottoms
out near 1.4 px however many BA rounds are spent, where a well-conditioned
photographic capture sits nearer 0.5. That gap is the frames disagreeing
with each other, which is expected for generated views and is the same
thing showing up in trap 3. So this corrects a wrong *assumption* against
partly unreliable *evidence*, +0.63 PSNR is the right size of prize for
that, and the headroom above it is in the frames rather than in the poses —
i.e. do this once and move on rather than tuning it.

Measured on one dataset with one subject, at 81 frames. It has not been
re-measured on the helical deliverable.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..masks import mask_to_alpha_u8
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

DEFAULT_COLMAP_BINARY = "colmap"

# The ONNX weights ALIKED and LightGlue run on, keyed by the COLMAP option
# that names each one. COLMAP fetches these itself on first use, into
# `$HOME/.cache/colmap/` — which on a pod is the container's writable
# layer, so they come down again after every restart and a run with no
# network cannot start at all. Same problem MediaPipe's .task files had and
# the same answer (steps/face_landmarks.py): cache them on the volume under
# `models_dir()` and hand COLMAP explicit paths, which
# `MaybeDownloadAndCacheFile` passes straight through when the string is a
# local file. `pipeline/models.py` pulls them at pod start off this table.
#
# URLs, names and digests are COLMAP 4.2's own (src/colmap/feature/
# resources.h); the digest is checked here for the same reason COLMAP
# checks it — a truncated download is otherwise an ONNX parse error two
# steps later.
_ONNX_RELEASE = "https://github.com/colmap/colmap/releases/download/3.13.0/"
ONNX_MODELS: Dict[str, Tuple[str, str, str]] = {
    # option name -> (filename, url, sha256)
    "AlikedExtraction.n32_model_path": (
        "aliked-n32.onnx", _ONNX_RELEASE + "aliked-n32.onnx",
        "a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c",
    ),
    "AlikedExtraction.n16rot_model_path": (
        "aliked-n16rot.onnx", _ONNX_RELEASE + "aliked-n16rot.onnx",
        "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    ),
    "AlikedMatching.lightglue_model_path": (
        "aliked-lightglue.onnx", _ONNX_RELEASE + "aliked-lightglue.onnx",
        "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    ),
    "SiftMatching.lightglue_model_path": (
        "sift-lightglue.onnx", _ONNX_RELEASE + "sift-lightglue.onnx",
        "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
    ),
}

# Which of the above a given extractor/matcher type actually loads. A type
# absent from these maps (SIFT, whose extractor is not an ONNX model) needs
# no weights and gets no `--...model_path`.
_EXTRACTOR_MODELS = {
    "ALIKED_N32": ("AlikedExtraction.n32_model_path",),
    "ALIKED_N16ROT": ("AlikedExtraction.n16rot_model_path",),
}
_MATCHER_MODELS = {
    "ALIKED_LIGHTGLUE": ("AlikedMatching.lightglue_model_path",),
    "SIFT_LIGHTGLUE": ("SiftMatching.lightglue_model_path",),
}

FEATURE_TYPES = ("ALIKED_N32", "ALIKED_N16ROT", "SIFT")
MATCHER_TYPES = ("ALIKED_LIGHTGLUE", "SIFT_LIGHTGLUE", "SIFT")
ON_CHECK_FAILURE = ("keep_given", "raise")


class BadSolve(RuntimeError):
    """COLMAP ran and produced something unusable.

    Distinct from COLMAP being *missing*, which `stream_command` raises as a
    plain RuntimeError and which nothing here catches: a broken image is not
    a bad dataset, and falling back to the given poses would hide it on
    every run rather than on one.
    """


def _model_path(filename: str) -> Path:
    """Where COLMAP's ONNX weights are cached — on the volume, not ~/.cache.

    Resolved lazily rather than at import time, like face_landmarks'
    twin: every step module is imported in every isolated venv, and a
    module-level call would mkdir on the volume just to answer "which step
    class is this".
    """
    from ..paths import models_dir

    return models_dir() / "colmap" / filename


def ensure_onnx_model(option: str) -> Path:
    """The cached weights for one COLMAP model option, downloading if absent.

    Shared with `pipeline/models.py`, which calls it at pod start so a run
    never blocks on it mid-pipeline.
    """
    import hashlib

    filename, url, sha256 = ONNX_MODELS[option]
    path = _model_path(filename)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("refine_cameras: downloading %s", filename)
    # Into a sibling temp file first: a download interrupted halfway
    # through leaves a plausible-looking file at the real path otherwise,
    # and every later run then skips the fetch and fails inside ONNX.
    scratch = path.with_suffix(path.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, str(scratch))
        digest = hashlib.sha256(scratch.read_bytes()).hexdigest()
        if digest != sha256:
            raise RuntimeError(
                f"{filename} downloaded from {url} has sha256 {digest}, "
                f"not the expected {sha256}"
            )
        scratch.replace(path)
    finally:
        scratch.unlink(missing_ok=True)
    return path


def onnx_options_for(feature_type: str, matcher_type: str) -> List[str]:
    """The COLMAP model options a given extractor/matcher pair needs."""
    return list(_EXTRACTOR_MODELS.get(feature_type, ())) + \
        list(_MATCHER_MODELS.get(matcher_type, ()))


@register_step("refine_cameras")
class RefineCamerasStep(Step):
    """Bundle-adjust a dataset's camera poses, in place, gauge preserved.

    inputs: {"cameras": List[Camera], "image_names": List[str],
             "images": List[np.ndarray] BGR(A),
             "masks": Optional[List[np.ndarray]] float32 [0,1], foreground=1}
    outputs: {"cameras": List[Camera], "stats": dict,
              "given_cameras": List[Camera] — the poses this step was
              handed, republished untouched}

    `given_cameras` is the input list, straight back out. Publishing it
    costs nothing and is the only way anything downstream can know what
    this step changed: `cameras` overwrites `dataset.cameras` in place, so
    once this has run the poses the rest of the run was BUILT on are gone.
    `rebase_cameras` below needs both halves — see its docstring for the
    consumer that was left standing on the old ones.
    """

    PARAMS = (
        Param("colmap_path", str, DEFAULT_COLMAP_BINARY,
              "The COLMAP binary. Built into docker/Dockerfile with CUDA on, "
              "which is what puts ALIKED and LightGlue on the GPU"),
        Param("iterations", int, 3,
              "How many (bundle adjust -> retriangulate) rounds follow the "
              "first triangulation. Converged by the third on the measured "
              "dataset; the reprojection error is logged per round, so a "
              "value that is doing nothing is visible in the log",
              minimum=1),
        Param("feature_type", str, "ALIKED_N32",
              "COLMAP's --FeatureExtraction.type. ALIKED holds up on a white "
              "cyclorama and bare skin where SIFT thins out",
              choices=FEATURE_TYPES),
        Param("matcher_type", str, "ALIKED_LIGHTGLUE",
              "COLMAP's --FeatureMatching.type; must be the one that goes "
              "with `feature_type`",
              choices=MATCHER_TYPES),
        Param("max_num_features", int, 4096,
              "--AlikedExtraction.max_num_features. 4096 gave ~747 features "
              "per image unmasked and 633 masked on the measured dataset, so "
              "this is a ceiling rather than a target",
              minimum=1, advanced=True),
        Param("use_gpu", bool, True,
              "Run ALIKED and LightGlue on the ONNX CUDA execution provider. "
              "A COLMAP built without CUDA silently uses the CPU provider "
              "instead (~3 min for 81 frames), so this is safe to leave on "
              "everywhere"),
        Param("foreground_only", bool, True,
              "Mask features to the subject. See the module docstring: it is "
              "0.089 PSNR behind letting them land on the backdrop, which is "
              "inside this pipeline's own nondeterminism, and it is the only "
              "variant whose assumption holds on a capture whose background "
              "moves or is generated per frame"),
        Param("max_scale_drift", float, 0.01,
              "Refuse a result whose mean camera radius about the scene "
              "centroid moved by more than this fraction — the assertion for "
              "trap 1, which is a 15-26% miss. 1% rather than the 0.1% the "
              "doc observed, and the difference is not slack: a correction "
              "that is not itself a similarity moves the radius by roughly "
              "the SQUARE of its size (a mean shift of 5% of the radius "
              "measures 0.14%), so a tighter gate would start refusing "
              "large-but-honest corrections under trap 1's name while the "
              "centre-shift check below is the one that actually describes "
              "them. The observed drift is logged either way",
              minimum=0.0, advanced=True),
        Param("max_centre_shift", float, 0.03,
              "Refuse a result whose MEAN camera centre moved more than this "
              "fraction of the scene radius. Measured at 0.0082; much beyond "
              "a few percent means BA re-solved the scene rather than "
              "refining it, so the matches are the thing to suspect",
              minimum=0.0, advanced=True),
        Param("on_check_failure", str, "keep_given",
              "What a failed check does. `keep_given` logs the failure and "
              "publishes the poses unchanged — the behaviour the pipeline had "
              "before this step existed, and the safe end of a 40-minute pod "
              "run; `raise` stops the run. Neither ever publishes poses that "
              "failed a check",
              choices=ON_CHECK_FAILURE),
        Param("work_dir", str, "",
              "Keep the COLMAP scratch (database, per-round models, logs) "
              "here instead of in a temporary directory that is deleted on "
              "the way out. NOT a dataset directory and not under one — see "
              "trap 2 in the module docstring",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cameras = list(inputs["cameras"])
        image_names = list(inputs["image_names"])
        images = list(inputs["images"])
        masks = inputs.get("masks")
        label = self.STEP_NAME or type(self).__name__

        if len(images) != len(image_names) or len(cameras) != len(image_names):
            raise ValueError(
                f"{label}: cameras ({len(cameras)}), image_names "
                f"({len(image_names)}) and images ({len(images)}) must "
                f"describe the same frames"
            )
        if len(cameras) < 3:
            raise ValueError(
                f"{label}: {len(cameras)} frame(s) is not something to bundle "
                f"adjust — the gauge alone costs two cameras"
            )

        work_dir = params["work_dir"]
        if work_dir:
            root = Path(work_dir)
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            return self._refine(root, cameras, image_names, images, masks, params, label)
        with tempfile.TemporaryDirectory(prefix="b2c_refine_") as temp_dir:
            return self._refine(Path(temp_dir), cameras, image_names, images,
                                masks, params, label)

    # ------------------------------------------------------------------
    # the pipeline itself
    # ------------------------------------------------------------------

    def _refine(
        self,
        work: Path,
        cameras: List[Any],
        image_names: List[str],
        images: List[np.ndarray],
        masks: Optional[List[np.ndarray]],
        params: Dict[str, Any],
        label: str,
    ) -> Dict[str, Any]:
        from ..proc import ProcessFailed

        colmap = params["colmap_path"]
        images_dir = work / "images"
        sparse_in = work / "sparse_in"
        images_dir.mkdir()
        sparse_in.mkdir()

        self._warn_on_mixed_intrinsics(cameras, label)
        self._write_images(images, image_names, images_dir)

        mask_dir: Optional[Path] = None
        if params["foreground_only"]:
            if masks is None:
                raise ValueError(
                    f"{label}: foreground_only is on but no masks were wired "
                    f"in. Either pass the foreground mattes or set "
                    f"foreground_only: false and accept a solve that trusts "
                    f"the backdrop"
                )
            mask_dir = work / "masks"
            mask_dir.mkdir()
            self._write_masks(masks, image_names, mask_dir)

        self._write_input_model(cameras, image_names, sparse_in)

        try:
            self._extract(colmap, work, images_dir, mask_dir, cameras[0], params)
            self._match(colmap, work, params)
            model, reprojection = self._triangulate_and_adjust(
                colmap, work, images_dir, params, label)
            refined = self._read_poses(model / "images.txt", image_names, cameras, label)
        except (ProcessFailed, BadSolve) as exc:
            # A COLMAP that ran and failed, or one that came back with a
            # model this dataset cannot be read out of, is the same thing to
            # do about as a failed check: the poses this step was handed are
            # the ones the pipeline used before it existed. A COLMAP that is
            # *missing* raises a plain RuntimeError out of stream_command
            # instead, and is deliberately not caught — see BadSolve.
            return self._give_up(cameras, params, label, f"COLMAP failed: {exc}")

        aligned, transform = _align_to(refined, cameras)
        stats = _movement(aligned, cameras, transform)
        stats["reprojection_px"] = reprojection
        stats["frames"] = len(cameras)

        logger.info(
            "%s: BA scale drift removed %+.2f%% (s=%.6f), leaving %.3f%% of "
            "radius drift; centre movement mean %.4f median %.4f max %.4f "
            "(%.2f%% / %.2f%% of scene radius %.3f); rotation mean %.3f deg "
            "max %.3f deg",
            label, stats["ba_scale_inflation"] * 100.0, transform[0],
            stats["radius_drift"] * 100.0,
            stats["centre_shift"]["mean"], stats["centre_shift"]["median"],
            stats["centre_shift"]["max"], stats["centre_shift"]["mean_frac"] * 100.0,
            stats["centre_shift"]["max_frac"] * 100.0, stats["scene_radius"],
            stats["rotation_deg"]["mean"], stats["rotation_deg"]["max"],
        )

        failure = _check(stats, params)
        if failure:
            stats["accepted"] = False
            stats["failure"] = failure
            return self._give_up(cameras, params, label, failure, stats)

        stats["accepted"] = True
        return {"cameras": aligned, "stats": stats, "given_cameras": cameras}

    def _give_up(
        self,
        cameras: List[Any],
        params: Dict[str, Any],
        label: str,
        why: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        message = (
            f"{label}: refusing the refined poses and keeping the given ones — {why}"
        )
        if params["on_check_failure"] == "raise":
            raise RuntimeError(message)
        logger.error("%s", message)
        out = dict(stats or {})
        out.update({"accepted": False, "failure": why})
        # `given_cameras` is published on this path too, and is the same
        # list as `cameras` here. That is what makes a refusal a no-op
        # downstream rather than a second thing to handle: `rebase_cameras`
        # measures a zero correction and passes its views through.
        return {"cameras": cameras, "stats": out, "given_cameras": cameras}

    # ------------------------------------------------------------------
    # writing what COLMAP reads
    # ------------------------------------------------------------------

    @staticmethod
    def _warn_on_mixed_intrinsics(cameras: Sequence[Any], label: str) -> None:
        """One camera model covers the whole dataset, here and in the export.

        `ColmapExporter` writes a single cameras.txt line from `cameras[0]`
        and gives every image CAMERA_ID 1, and the solve below is set up the
        same way (`--ImageReader.single_camera 1`). Every orbit this
        pipeline generates satisfies that. A dataset that did not would
        already be exporting a cameras.txt disagreeing with its own
        images — a pre-existing problem, not this step's to enforce — but
        it would be refined against the wrong intrinsics here, so say so.
        """
        first = cameras[0]
        key = (first.fx, first.fy, first.cx, first.cy, first.width, first.height)
        odd = [i for i, cam in enumerate(cameras)
               if (cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height) != key]
        if odd:
            logger.warning(
                "%s: %d of %d frames do not share the first camera's intrinsics "
                "(first differs at frame %d). The solve treats the batch as one "
                "camera, so those frames are being refined against the wrong "
                "focal length.", label, len(odd), len(cameras), odd[0],
            )

    @staticmethod
    def _write_images(images, image_names, images_dir: Path) -> None:
        """The frames, exactly as brush's own export writes them."""
        for image, filename in zip(images, image_names):
            cv2.imwrite(str(images_dir / filename), image)

    @staticmethod
    def _write_masks(masks, image_names, mask_dir: Path) -> None:
        """Alpha -> a COLMAP mask: black is what COLMAP ignores.

        Written as `<stem>.png`, which is the alternative name COLMAP tries
        when `<image name>.png` is absent (controllers/image_reader.cc) —
        our names already end in .png, so the literal form would be
        `frame_00001_.png.png`.
        """
        if len(masks) != len(image_names):
            raise ValueError(
                f"masks ({len(masks)}) and image_names ({len(image_names)}) "
                f"length mismatch"
            )
        for mask, filename in zip(masks, image_names):
            binary = (mask_to_alpha_u8(mask) > 127).astype(np.uint8) * 255
            cv2.imwrite(str(mask_dir / (Path(filename).stem + ".png")), binary)

    @staticmethod
    def _write_input_model(cameras, image_names, sparse_in: Path) -> None:
        """The given poses as a COLMAP model with no points.

        `points_3d=None` so the exporter writes no points3D.txt, then an
        empty one: `point_triangulator` wants a model carrying poses and no
        observations, which is exactly what the pipeline's own export is.
        """
        from body2colmap.exporter import ColmapExporter

        ColmapExporter(cameras=list(cameras), image_names=list(image_names),
                       points_3d=None).export(output_dir=sparse_in)
        (sparse_in / "points3D.txt").write_text("")

    # ------------------------------------------------------------------
    # driving COLMAP
    # ------------------------------------------------------------------

    @staticmethod
    def _model_flags(which: Dict[str, Tuple[str, ...]], type_name: str) -> List[str]:
        """`--<option> <cached path>` for every ONNX model a type loads.

        Nothing for a type that has none — SIFT's extractor is not an ONNX
        model — which leaves COLMAP's own default in place.
        """
        flags: List[str] = []
        for option in which.get(type_name, ()):
            flags += [f"--{option}", str(ensure_onnx_model(option))]
        return flags

    def _extract(self, colmap: str, work: Path, images_dir: Path,
                 mask_dir: Optional[Path], camera, params: Dict[str, Any]) -> None:
        model, intrinsics = camera.get_colmap_intrinsics()
        cmd = [
            colmap, "feature_extractor",
            "--database_path", str(work / "database.db"),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", model,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", ",".join(f"{p:.6f}" for p in intrinsics),
            "--FeatureExtraction.type", params["feature_type"],
            "--FeatureExtraction.use_gpu", _flag(params["use_gpu"]),
            "--AlikedExtraction.max_num_features", str(params["max_num_features"]),
        ]
        if mask_dir is not None:
            cmd += ["--ImageReader.mask_path", str(mask_dir)]
        cmd += self._model_flags(_EXTRACTOR_MODELS, params["feature_type"])
        self._run(cmd, "colmap.feature_extractor")

    def _match(self, colmap: str, work: Path, params: Dict[str, Any]) -> None:
        cmd = [
            colmap, "exhaustive_matcher",
            "--database_path", str(work / "database.db"),
            "--FeatureMatching.type", params["matcher_type"],
            "--FeatureMatching.use_gpu", _flag(params["use_gpu"]),
        ] + self._model_flags(_MATCHER_MODELS, params["matcher_type"])
        self._run(cmd, "colmap.exhaustive_matcher")

    def _triangulate_and_adjust(
        self, colmap: str, work: Path, images_dir: Path,
        params: Dict[str, Any], label: str,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Triangulate, then N x (bundle adjust -> retriangulate).

        Returns the directory holding the final BA's poses, and the mean
        reprojection error before and after. The *poses* come off the last
        `bundle_adjuster` rather than the retriangulation behind it — the
        two are identical, since triangulating does not move a camera — but
        the reprojection error has to come off the retriangulation, which
        is the model whose tracks were built against those poses.
        """
        def triangulate(source: Path, destination: Path) -> Dict[str, float]:
            destination.mkdir(parents=True, exist_ok=True)
            self._run([
                colmap, "point_triangulator",
                "--database_path", str(work / "database.db"),
                "--image_path", str(images_dir),
                "--input_path", str(source),
                "--output_path", str(destination),
                "--clear_points", "1",
                "--refine_intrinsics", "0",
            ], "colmap.point_triangulator")
            return self._analyze(colmap, destination)

        current = work / "tri"
        before = triangulate(work / "sparse_in", current)
        logger.info("%s: given poses — mean reprojection error %s px",
                    label, _px(before.get("reprojection_error")))

        poses = work / "sparse_in"
        after = before
        for round_index in range(1, params["iterations"] + 1):
            poses = work / f"ba{round_index}"
            poses.mkdir(parents=True, exist_ok=True)
            self._run([
                colmap, "bundle_adjuster",
                "--input_path", str(current),
                "--output_path", str(poses),
                # Trap 3. Frozen, with no param to unfreeze them.
                "--BundleAdjustment.refine_focal_length", "0",
                "--BundleAdjustment.refine_principal_point", "0",
                "--BundleAdjustment.refine_extra_params", "0",
                "--BundleAdjustment.refine_sensor_from_rig", "0",
            ], "colmap.bundle_adjuster")
            current = work / f"tri{round_index}"
            after = triangulate(poses, current)
            logger.info("%s: round %d/%d — mean reprojection error %s px",
                        label, round_index, params["iterations"],
                        _px(after.get("reprojection_error")))

        # The refined model is binary; the reader below wants text.
        text = work / "refined_txt"
        text.mkdir(parents=True, exist_ok=True)
        self._run([colmap, "model_converter", "--input_path", str(poses),
                   "--output_path", str(text), "--output_type", "TXT"],
                  "colmap.model_converter")
        return text, {"given": before.get("reprojection_error"),
                      "refined": after.get("reprojection_error")}

    def _analyze(self, colmap: str, model: Path) -> Dict[str, float]:
        """`model_analyzer`'s numbers, or an empty dict if it said nothing.

        Parsed with a regex rather than by splitting on the colon: these
        arrive through glog, so every line carries an `I0831 12:00:00.000000
        1 model.cc:456]` prefix with colons of its own.

        Diagnostic, not a gate. The reprojection error bottoms out near
        1.4 px on generated frames however many rounds are spent, so "it is
        high" does not distinguish bad poses from inconsistent frames — see
        the module docstring's ceiling.
        """
        patterns = {
            "reprojection_error": r"Mean reprojection error:\s*([0-9.eE+-]+)px",
            "registered_images": r"Registered images:\s*([0-9]+)",
            "points": r"Points:\s*([0-9]+)",
            "mean_track_length": r"Mean track length:\s*([0-9.eE+-]+)",
        }
        text = "".join(self._run([colmap, "model_analyzer", "--path", str(model)],
                                 "colmap.model_analyzer", throttle=False))
        out: Dict[str, float] = {}
        for name, pattern in patterns.items():
            found = re.search(pattern, text)
            if found:
                out[name] = float(found.group(1))
        return out

    @staticmethod
    def _run(cmd: Sequence[str], log_name: str, throttle: bool = True) -> List[str]:
        from ..proc import stream_command

        return stream_command(
            cmd, log_name, throttle=throttle,
            not_found_hint=(
                "COLMAP is built into docker/Dockerfile's colmap-builder stage; "
                "on a workstation, build it from https://github.com/colmap/colmap "
                "or point `colmap_path` at an existing binary."
            ),
        )

    # ------------------------------------------------------------------
    # reading what COLMAP wrote
    # ------------------------------------------------------------------

    @staticmethod
    def _read_poses(images_txt: Path, image_names: Sequence[str],
                    cameras: Sequence[Any], label: str) -> List[Any]:
        """images.txt -> `Camera`s in world coords, intrinsics carried across.

        The inverse of body2colmap's `world_to_colmap_camera`: that writes
        `R_w2c = G @ R_c2w.T` and `t = -R_w2c @ position` with
        `G = diag(1, -1, -1)`, so `R_c2w = R_w2c.T @ G` and
        `position = -R_w2c.T @ t`.
        """
        from body2colmap.camera import Camera

        poses = _read_images_txt(images_txt)
        missing = [name for name in image_names if name not in poses]
        if missing:
            raise BadSolve(
                f"{label}: the refined model is missing {len(missing)} of "
                f"{len(image_names)} images (first: {missing[0]}). Every frame "
                f"has to survive the solve for the result to describe this "
                f"dataset."
            )

        gl_from_cv = np.diag([1.0, -1.0, -1.0])
        out = []
        for name, given in zip(image_names, cameras):
            quaternion, translation = poses[name]
            r_w2c = _quaternion_to_rotation(quaternion)
            out.append(Camera(
                focal_length=(given.fx, given.fy),
                image_size=(given.width, given.height),
                principal_point=(given.cx, given.cy),
                position=-r_w2c.T @ translation,
                rotation=r_w2c.T @ gl_from_cv,
            ))
        return out


@register_step("rebase_cameras")
class RebaseCamerasStep(Step):
    """Carry a render's cameras across the correction `refine_cameras` made.

    inputs: {"cameras": List[Camera] — the views to move,
             "from_cameras": Optional[List[Camera]] — the dataset's poses
             before the refinement, i.e. its `given_cameras`,
             "to_cameras": Optional[List[Camera]] — the same poses after it,
             "reference_index": Optional[int] — which of the two lists'
             frames defines the correction; the anchor frame}
    outputs: {"cameras": List[Camera], "stats": dict}

    The problem this exists for
    ---------------------------
    The face splat is built out of ONE frame — the reference photograph at
    the anchor camera — and it is built by unprojecting that photograph
    through that camera's pose. Its supporting views (`render_splat`'s cap,
    `select_support_views`) are renders of it, and they are made in the
    bootstrap, long before `refine_cameras` runs.

    Then `refine_cameras` runs, and the anchor camera moves. The frames do
    not: they are the same pixels, now declared to have been taken from
    somewhere slightly else. So the world content that photograph depicts
    moves with the pose — the face is wherever the anchor's ray bundle now
    points — while the splat, and the cap cameras rendered around it, stay
    on the old pose. `brush` is then handed supporting views that put the
    face a centimetre or two off the head every training frame agrees on,
    and weights them by the face's own alpha, which is precisely where it
    hurts. Nothing raises; the face just lands wrong.

    The fix, and why it is exactly a rigid transform
    ------------------------------------------------
    Write the anchor's pose as `P = [R_c2w | position]`. The splat sits at
    `P_old @ x_cam` for camera-frame points `x_cam` read off the
    photograph. The refinement's claim is that those same pixels were
    taken from `P_new`, so the content they depict is at `P_new @ x_cam` —
    the splat, moved by

        D = P_new @ P_old^-1     (R = R_new @ R_old^T, t = p_new - R @ p_old)

    and that is not an approximation of the correction, it *is* it.

    Which is why only the CAMERAS are touched here. A render is a function
    of the pose of the camera relative to the splat, so moving both by the
    same `D` leaves every pixel identical — the images published by
    `select_support_views` stay exactly what they were, and this step
    re-expresses the cameras they were taken from in the world the refined
    poses describe. The .ply itself is never re-read after this point
    (`render` and `render_splat` both consume it in the bootstrap), so
    there is nothing else to move.

    Why the ANCHOR's correction and not some average of the orbit's
    ---------------------------------------------------------------
    Because the splat has exactly one source frame, and D above is written
    in that frame's pose alone. Averaging over the neighbouring cameras
    would be the right move for a shell built from several of them, and it
    is the wrong one here: it would answer "where did the orbit go" when
    the question is "where did THIS photograph's rays go".

    The residual — that the other 80 frames moved by their own deltas and
    so vote for a face very slightly elsewhere — is the refinement's own
    non-rigidity, and is the size of the disagreement between frames
    (`refine_cameras`' `centre_shift` stats), not of the correction.

    What happens when there is nothing to do
    ----------------------------------------
    Three cases, all of them ordinary wiring rather than failures, and all
    of them a pass-through:

      * `refine_cameras` is switched off, so nothing writes its
        `given_cameras` and `from_cameras` reads as None;
      * it ran and REFUSED its solve, in which case it published the given
        poses as both halves and the correction measures exactly zero;
      * this workflow's face branch is off, so there are no cameras to
        move (the step is gated with it, so it does not run at all).

    `max_delta_deg` is the fourth: a correction larger than the whole
    refinement is ever expected to make is more likely a mis-wire — the
    wrong reference index, or two lists that are not the same frames — than
    a real one, and applying it would move the face further than leaving
    it. That publishes the cameras unchanged and says so, the same trade
    `refine_cameras`' own `keep_given` makes.
    """

    PARAMS = (
        Param("max_delta_deg", float, 5.0,
              "Refuse the rebase, and publish the cameras unchanged, if the "
              "reference camera's own correction rotated by more than this. "
              "The refinement's measured maximum is 1.911 deg across a whole "
              "81-frame orbit (docs/camera-pose-refinement.md), so this is a "
              "mis-wire detector rather than a tuning knob. 0 disables it",
              minimum=0.0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cameras = list(inputs["cameras"])
        before = inputs.get("from_cameras")
        after = inputs.get("to_cameras")
        label = self.STEP_NAME or type(self).__name__

        stats: Dict[str, Any] = {"applied": False, "views": len(cameras)}

        if not before or not after:
            # The refinement is switched off. Not a warning: this is the
            # pipeline as it was before that step existed, and the views
            # are correct against the poses that are actually in use.
            logger.info(
                "%s: no refinement to carry across (from_cameras=%s, "
                "to_cameras=%s), so the %d view(s) pass through unchanged",
                label, "absent" if not before else "present",
                "absent" if not after else "present", len(cameras),
            )
            stats["reason"] = "no refinement was wired in"
            return {"cameras": cameras, "stats": stats}

        before, after = list(before), list(after)
        if len(before) != len(after):
            raise ValueError(
                f"{label}: from_cameras has {len(before)} poses and "
                f"to_cameras {len(after)}. They are the same frames before "
                f"and after a refinement and have to arrive together."
            )

        index = inputs.get("reference_index")
        if index is None:
            # Where a circular path's anchor sits anyway — the same
            # fallback pointmap_elevation_views makes for the same reason.
            index = 0
            logger.info(
                "%s: no reference_index wired in; taking frame 0 as the "
                "frame the views were built from", label,
            )
        index = int(index)
        if not 0 <= index < len(before):
            raise ValueError(
                f"{label}: reference_index {index} is not a frame of a "
                f"{len(before)}-camera path. It names the frame these views "
                f"were built from — the anchor."
            )

        rotation, translation = _pose_delta(before[index], after[index])
        moved = float(np.linalg.norm(
            np.asarray(after[index].position, dtype=np.float64).reshape(3)
            - np.asarray(before[index].position, dtype=np.float64).reshape(3)))
        degrees = float(np.degrees(np.arccos(
            np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
        stats.update({
            "reference_index": index,
            "rotation_deg": degrees,
            "reference_shift": moved,
        })

        limit = params["max_delta_deg"]
        if limit > 0.0 and degrees > limit:
            logger.error(
                "%s: refusing to rebase — frame %d's correction rotates by "
                "%.3f deg, past the %.3f deg this step will carry. That is "
                "larger than a refinement of this orbit makes, so suspect the "
                "wiring (a reference_index naming the wrong frame, or two "
                "camera lists that are not the same path) before believing "
                "it. The %d view(s) are published unchanged.",
                label, index, degrees, limit, len(cameras),
            )
            stats["reason"] = (
                f"the reference camera's correction is {degrees:.3f} deg, "
                f"past max_delta_deg={limit:.3f}"
            )
            return {"cameras": cameras, "stats": stats}

        rebased = [_transformed(camera, rotation, translation) for camera in cameras]
        stats["applied"] = True
        logger.info(
            "%s: carried frame %d's correction (%.3f deg, centre moved "
            "%.4f) onto %d supporting view(s)",
            label, index, degrees, moved, len(cameras),
        )
        return {"cameras": rebased, "stats": stats}


def _pose_delta(before: Any, after: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The world transform that takes one camera's old pose to its new one.

    `Camera.rotation` is camera-to-world (body2colmap/camera.py), so a
    world transform is applied to a camera as `R @ rotation` and
    `R @ position + t` — which is exactly what `_align_to` does with the
    Sim(3) it fits, one scale factor aside.
    """
    r_before = np.asarray(before.rotation, dtype=np.float64)
    r_after = np.asarray(after.rotation, dtype=np.float64)
    p_before = np.asarray(before.position, dtype=np.float64).reshape(3)
    p_after = np.asarray(after.position, dtype=np.float64).reshape(3)

    rotation = r_after @ r_before.T
    return rotation, p_after - rotation @ p_before


def _transformed(camera: Any, rotation: np.ndarray, translation: np.ndarray) -> Any:
    """One camera through a world rigid transform, intrinsics untouched."""
    from body2colmap.camera import Camera

    return Camera(
        focal_length=(camera.fx, camera.fy),
        image_size=(camera.width, camera.height),
        principal_point=(camera.cx, camera.cy),
        position=rotation @ np.asarray(camera.position, dtype=np.float64).reshape(3)
        + translation,
        rotation=rotation @ np.asarray(camera.rotation, dtype=np.float64),
    )


# --------------------------------------------------------------------------
# the gauge (step 5), and the checks that say it happened
# --------------------------------------------------------------------------

def _align_to(refined: Sequence[Any], given: Sequence[Any]) -> Tuple[List[Any], Tuple[float, np.ndarray, np.ndarray]]:
    """Put the given model's rotation, translation and scale back.

    Umeyama, with scale, fitted from the refined camera centres onto the
    given ones and applied to the refined poses. Every relative correction
    BA made is kept; the global frame is restored exactly (measured:
    orbit radius 1.8036 -> 1.8034, centroid preserved). This is trap 1's
    only fix — without it the model comes back 15-26% larger, in a frame
    whose points3D.txt init no longer matches it, and nothing downstream
    would say so.
    """
    from body2colmap.camera import Camera

    source = np.array([np.asarray(c.position, dtype=np.float64) for c in refined])
    target = np.array([np.asarray(c.position, dtype=np.float64) for c in given])
    scale, rotation, translation = _umeyama(source, target)

    aligned = []
    for camera, template in zip(refined, given):
        position = scale * (rotation @ np.asarray(camera.position, dtype=np.float64)) + translation
        aligned.append(Camera(
            focal_length=(template.fx, template.fy),
            image_size=(template.width, template.height),
            principal_point=(template.cx, template.cy),
            position=position,
            rotation=rotation @ np.asarray(camera.rotation, dtype=np.float64),
        ))
    return aligned, (scale, rotation, translation)


def _umeyama(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising ||s*R@source_i + t - target_i||."""
    mu_source, mu_target = source.mean(0), target.mean(0)
    centred_source, centred_target = source - mu_source, target - mu_target
    u, singular, vt = np.linalg.svd(centred_target.T @ centred_source / len(source))
    flip = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        flip[2, 2] = -1.0
    rotation = u @ flip @ vt
    variance = (centred_source ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(singular) @ flip) / variance)
    return scale, rotation, mu_target - scale * rotation @ mu_source


def _movement(refined: Sequence[Any], given: Sequence[Any],
              transform: Tuple[float, np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """How far the cameras actually moved, with the gauge already removed.

    i.e. genuine correction rather than a frame change. On the measured
    dataset: mean 0.0148, median 0.0129, max 0.0691 against a 1.804 orbit
    radius (0.82% / 3.83%), rotation mean 0.587 deg and max 1.911 — with
    the vertical component the largest, which is the expected shape, since
    height was the axis the generated orbit pinned hardest (std exactly 0).
    """
    refined_centres = np.array([np.asarray(c.position, dtype=np.float64) for c in refined])
    given_centres = np.array([np.asarray(c.position, dtype=np.float64) for c in given])
    centroid = given_centres.mean(0)

    shift = np.linalg.norm(refined_centres - given_centres, axis=1)
    given_radius = float(np.linalg.norm(given_centres - centroid, axis=1).mean())
    refined_radius = float(np.linalg.norm(refined_centres - centroid, axis=1).mean())

    angles = []
    for a, b in zip(refined, given):
        relative = np.asarray(a.rotation, dtype=np.float64).T @ np.asarray(b.rotation, dtype=np.float64)
        angles.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))

    return {
        "scene_radius": given_radius,
        "scene_radius_refined": refined_radius,
        # Positive = how much BA had grown the model before the Sim(3) put
        # it back. 15-26% across seven runs, and not bounded by that;
        # see trap 1.
        "ba_scale_inflation": 1.0 / transform[0] - 1.0,
        "radius_drift": abs(refined_radius - given_radius) / given_radius if given_radius else 0.0,
        "centre_shift": {
            "mean": float(shift.mean()), "median": float(np.median(shift)),
            "max": float(shift.max()),
            "mean_frac": float(shift.mean() / given_radius) if given_radius else 0.0,
            "max_frac": float(shift.max() / given_radius) if given_radius else 0.0,
        },
        "rotation_deg": {"mean": float(np.mean(angles)), "max": float(np.max(angles))},
    }


def _check(stats: Dict[str, Any], params: Dict[str, Any]) -> str:
    """The doc's check list, as assertions. Empty string means it passed."""
    if stats["radius_drift"] > params["max_scale_drift"]:
        return (
            f"the mean camera radius moved {stats['radius_drift'] * 100:.3f}%, "
            f"over the {params['max_scale_drift'] * 100:.3f}% allowed. The "
            f"Sim(3) restores scale exactly, so this means it did not fit — "
            f"which is trap 1, and a splat "
            f"{stats['ba_scale_inflation'] * 100:+.1f}% off scale in a frame "
            f"its points3D.txt init no longer matches"
        )
    if stats["centre_shift"]["mean_frac"] > params["max_centre_shift"]:
        return (
            f"the mean camera centre moved "
            f"{stats['centre_shift']['mean_frac'] * 100:.2f}% of the scene "
            f"radius, over the {params['max_centre_shift'] * 100:.2f}% "
            f"allowed. That is BA re-solving the scene rather than refining "
            f"it; suspect the matches"
        )
    return ""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _flag(value: bool) -> str:
    return "1" if value else "0"


def _px(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.3f}"


def _read_images_txt(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """name -> (quaternion wxyz, translation), ignoring the POINTS2D lines."""
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        poses[fields[9]] = (
            np.array([float(v) for v in fields[1:5]]),
            np.array([float(v) for v in fields[5:8]]),
        )
        index += 2  # the second line of the pair is POINTS2D
    return poses


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """COLMAP's (w, x, y, z) -> a 3x3 rotation. Note the order: scipy and
    pyquaternion both use (x, y, z, w)."""
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])
