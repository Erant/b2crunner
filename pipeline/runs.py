"""What a run *is*, with no front end attached: what a submission means,
how it is queued, what a finished one left on the volume, and how that is
packaged for download.

All of this used to live in `pipeline.webui`, behind a top-level
`import gradio`, which made it reachable only from the browser. It is now
its own module for one reason: `pipeline.api` and `pipeline.webui` both
drive the same lifecycle, and the two must not be able to drift on what a
submission means. `submit_runs` is the single place a `RunJob` is built —
overrides validated, `requires:` applied, the run named, the fan-out
decided — so an HTTP client and a button press queue exactly the same work.

Nothing here imports gradio, so it also works in a headless install, and
its tests no longer skip when the UI's dependency is absent.

The user-facing refusals raise `SubmitError`, whose message is a sentence
meant to be read: the UI shows it in a toast, the API returns it as a 400.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import steps  # noqa: F401  registers every Step; the registry must be whole
from .gpu_scheduler import GpuScheduler
from .logging_setup import timestamped_run_name
from .paths import log_dir, output_dir, run_jobs_dir, upload_dir
from .run_state import RunJob, RunState
from .step import Param
from .workflow import Output, WorkflowSpec, apply_ui_overrides

logger = logging.getLogger(__name__)

# There is no workflow picker: every submission — an image, or a zip of
# image/prompt pairs — runs this one workflow. The `workflow` argument
# exists so a caller can be explicit, not so it can pick something else.
WORKFLOW_NATIVE = "fast_helical_native"


class SubmitError(Exception):
    """A submission this pipeline refuses, with a sentence saying why.

    A plain exception rather than `gr.Error` because both front ends raise
    it: the UI re-raises it as a toast, the API returns it as a 400. Its
    message is the whole payload — write it as something a person can act
    on ("Pick at least one output", not "invalid state").
    """


# --------------------------------------------------------------------------
# what a workflow declares
# --------------------------------------------------------------------------

def workflow_outputs(name: str) -> List["Output"]:
    """The deliverables this workflow declares, in the order it declares them."""
    from .cli import resolve_workflow

    return WorkflowSpec.from_yaml(resolve_workflow(name)).outputs


def resolve_outputs(spec: WorkflowSpec) -> Dict[str, bool]:
    """The output switches `spec.globals` should carry, `requires:` applied.

    An output whose `requires:` setting is switched off is forced off here
    rather than left to produce something under a name that no longer means
    what it says — the pre-upscale COLMAP export with the upscale off would
    be the ordinary `colmap/` twice. The UI also disables that checkbox, so
    this is the second half of one rule, for the paths that do not go
    through a checkbox at all (`--param`, a stale panel).

    Raises `SubmitError` if nothing would be exported: a run that produces no
    deliverable leaves nothing to download, and it is an hour of GPU either
    way.

    The debug bundle does not count towards that. It is declared as an
    output so it draws as a checkbox and travels as a switch, but it is not
    something a run *produces* — `_write_run_members` refuses to build an
    archive out of it alone, exactly as it refuses to build one out of
    `log.txt`. Counting it would let "colmap off, ply off, debug on" past
    this check and then hand back nothing at the end of an hour.
    """
    resolved = spec.apply_output_requirements()
    deliverables = {
        name: wanted for name, wanted in resolved.items()
        if name not in _debug_output_names(spec.name)
    }
    if deliverables and not any(deliverables.values()):
        raise SubmitError(
            "Pick at least one output — a run that exports nothing leaves "
            "nothing to download. (The debug bundle is not one on its own.)"
        )
    return resolved



# Exactly the fields `Dataset.from_reference_image` cannot fill in: a
# photo gives you `reference_image`, `prompt` and a resolution, and nothing
# else. A workflow that reads one of these before a step has written it is
# a workflow that needs a real dataset handed to it.
_NEEDS_A_REAL_DATASET = {
    "dataset.images", "dataset.image_names", "dataset.cameras", "dataset.points_3d",
}


def workflow_needs_a_dataset(name: str) -> bool:
    """Whether this workflow reads frames or cameras it does not produce.

    A from-a-photo workflow renders its own views first (sam3d_body ->
    render), so nothing reads `dataset.images` until something has written
    it. One that starts from a dataset reads it immediately. Walking the
    steps in order and asking which comes first answers "can this run from
    a photo?" from the steps themselves, rather than from a flag in the
    YAML that could disagree with them.

    A bare `dataset` read (what `save_dataset` takes) is deliberately not a
    trip: by the time a checkpoint runs, the earlier steps have populated
    the object it is handed.
    """
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    written = set()
    for step in spec.steps:
        if any(path in _NEEDS_A_REAL_DATASET - written for path in step.inputs.values()):
            return True
        written.update(step.outputs.values())
    return False


def workflow_param_panel(
    name: str,
) -> tuple[WorkflowSpec, List[Param], List["Output"], List[Dict[str, Any]]]:
    """Everything the Run tab needs to draw itself, for one workflow.

    Also what `GET /api/v1/workflows/{name}` publishes, minus the widgets:
    a client discovering which `settings` keys a submission may carry is
    asking the same question the panel asks, and the two must not be able
    to answer it differently.

    Returns (spec, settings, outputs, steps):

      * `settings` — the workflow's declared knobs, as `Param`s, in file
        order. The panel draws these; a plain `globals:` entry gets no
        control at all, which is what keeps `output_root` off the form.
        That used to be a HIDDEN_GLOBALS denylist, and it mattered: `start()`
        only repoints `output_root` at the run's own timestamped directory
        when the submitted overrides do not contain the key, so a panel that
        round-tripped its literal default permanently defeated the repoint —
        colmap_export and the final training wrote under the process's cwd
        and the Results tab reported that the run had produced nothing.
        Undeclared means undrawable now, so that cannot come back.
      * `outputs` — the deliverables, for the Outputs box.
      * `steps` — one entry per step, `{"id", "step", "params": [Param],
        "overrides": {...}, "global_refs": {...}}`, for the per-step fold.
        `overrides` is what this workflow set on top, already
        template-expanded so a field shows the value the run would really
        use, and keyed by the step's `id:` — which is the whole point: the
        two `brush` trainings in fast_helical_native get their own section
        and their own controls.
    """
    from .cli import resolve_workflow
    from .registry import get_step_class
    from .templating import global_ref, resolve

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))

    scope = {"globals": spec.globals}
    steps: List[Dict[str, Any]] = []
    for step in spec.steps:
        declared = get_step_class(step.step).declared_params()
        if not declared:
            continue
        steps.append({
            "id": step.id,
            "step": step.step,
            "params": list(declared.values()),
            "overrides": resolve(step.params, scope),
            # Params this step reads verbatim from a workflow global
            # ({param: "resolution"} etc.). The panel omits these entirely —
            # their one home is the Settings control. A read-only mirror was
            # worse: it did not track edits to the setting, so it just read as
            # a stale, contradictory second value (f4dca77).
            "global_refs": {
                pname: ref
                for pname, raw in step.params.items()
                if pname in declared and (ref := global_ref(raw)) is not None
            },
        })
    return spec, list(spec.settings), list(spec.outputs), steps



# --------------------------------------------------------------------------
# what an upload is, and the runs it fans out to
# --------------------------------------------------------------------------

# Image extensions a submission understands, bare or inside a zip.
# Kept in step with what `cv2.imread` (via `Dataset.from_reference_image`)
# can decode.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# One (reference_image, prompt) per run a submission fans out to.
RunPlan = List["tuple[Path, str]"]


def _guarded_extract(zip_path: str, target: Path) -> Path:
    """Unpack `zip_path` under `target`, refusing absolute or `..` members.

    This unpacks onto the mounted volume, next to real data, so it cannot
    trust the member names the archive carries.
    """
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            resolved = (target / member).resolve()
            # `is_relative_to`, not a string prefix: `/x/upload-1-evil/f`
            # startswith `/x/upload-1`, so a prefix test lets a member
            # escape into a sibling directory whose name merely extends the
            # target's.
            if not resolved.is_relative_to(target.resolve()):
                raise SubmitError(f"Refusing to extract {member!r}: it escapes the target directory.")
        archive.extractall(target)
    return target


def _iter_files(root: Path):
    """Every real file under `root`, skipping a macOS `__MACOSX/` sidecar."""
    return (p for p in root.rglob("*") if p.is_file() and "__MACOSX" not in p.parts)


def pair_images_with_prompts(root: Path, fallback_prompt: str = "") -> List["tuple[Path, str]"]:
    """[(image_path, prompt), ...] for every image under an already-extracted zip.

    Pairing is by stem within a directory — `image1.jpg` takes `image1.txt`
    beside it — so a zip made by selecting files and one made by zipping a
    folder both work.

    Two shapes are accepted. If the archive carries any `.txt` it is treated
    as image/prompt pairs and every image must have its match (a missing one
    is a slip, not an intent to run promptless). If it carries none, each
    image is a bare reference sheet and `fallback_prompt` — what the Subject
    box holds — is the prompt for all of them.
    """
    images = sorted(p for p in _iter_files(root) if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SubmitError(
            "Nothing usable in the zip: no images (reference sheets, or "
            "`image1.jpg` + `image1.txt` pairs)."
        )
    have_prompts = any(p.suffix.lower() == ".txt" for p in _iter_files(root))

    pairs: List["tuple[Path, str]"] = []
    missing: List[str] = []
    for image in images:
        prompt_file = next(
            (image.with_suffix(s) for s in (".txt", ".TXT") if image.with_suffix(s).exists()),
            None,
        )
        if prompt_file is None:
            (missing.append(image.name) if have_prompts
             else pairs.append((image, fallback_prompt)))
            continue
        pairs.append((image, prompt_file.read_text(encoding="utf-8").strip() or fallback_prompt))

    if missing:
        raise SubmitError(
            "No matching .txt prompt file for: " + ", ".join(missing)
            + ". Give every image a same-named .txt beside it, or remove all the "
            ".txt files to use the Subject box for each."
        )
    return pairs


def resolve_upload(upload_path: str, prompt: str) -> RunPlan:
    """Work out what one upload is, and the runs it fans out to.

    Content decides — there is no input picker any more:

      * a bare image file        -> one reference-sheet run
      * a .zip of images (+ .txt) -> one run per image
    """
    suffix = Path(upload_path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [(save_upload(upload_path, "reference"), prompt)]
    if suffix != ".zip":
        raise SubmitError(
            f"Don't know what to do with a {suffix or 'no-extension'} file — upload "
            "a .zip of image/prompt pairs or a single reference image."
        )

    # Same random suffix, and it matters more here: `_guarded_extract` is
    # `mkdir(exist_ok=True)`, so a second zip landing in the same second
    # would unpack *into* the first one's tree and `pair_images_with_prompts`
    # would then fan out over both archives — re-running someone else's
    # subjects at an hour of GPU each.
    root = _guarded_extract(
        upload_path,
        upload_dir() / f"upload-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}",
    )
    return list(pair_images_with_prompts(root, prompt))


def save_upload(path: str, prefix: str) -> Path:
    """Copy an upload onto the volume; a request's own temp file is not durable.

    The name carries a random suffix as well as a timestamp, for the same
    reason `timestamped_run_name` does: seconds are not fine-grained enough
    to tell two submissions apart. Without it a scripted batch — a loop over
    `POST /api/v1/runs`, which is the whole point of having an API —
    silently overwrites the first sheet with the second, and the run queued
    for subject A starts, an hour later, on subject B's photograph, with
    nothing in any log to say so.
    """
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    destination = upload_dir() / f"{prefix}-{stamp}{Path(path).suffix}"
    shutil.copy(path, destination)
    return destination



# --------------------------------------------------------------------------
# queueing them
# --------------------------------------------------------------------------

def _refuse_unknown_overrides(
    spec: WorkflowSpec,
    global_overrides: Dict[str, Any],
    step_overrides: Dict[str, Dict[str, Any]],
) -> None:
    """Refuse an override this workflow does not declare, naming what it does.

    `apply_ui_overrides` drops these instead, on purpose: a browser's param
    panel can go stale between being drawn and being submitted, and losing a
    control that is no longer on screen beats failing the run over it.

    A programmatic caller has no stale panel. It has a typo — and a typo'd
    override that silently does nothing is the worst outcome available here,
    because the run completes, at the wrong settings, after hours of GPU,
    looking exactly like one that honoured you. Same rule, and much the same
    wording, as `pipeline.cli`'s `--param`.

    Measured against the workflow's *declared* globals, not every global it
    has. The difference is `output_root`, which has no declaration and
    therefore no control anywhere — the UI cannot draw it, and this cannot
    accept it. That is not only tidiness: `pipeline.run_worker` repoints
    `output_root` at the run's own directory *unless the submission pinned
    it*, so an accepted `output_root` would send several GB of splat and
    COLMAP export wherever the caller named, outside the run directory the
    Results tab and `/result` look in.
    """
    from .registry import get_step_class

    declared = spec.declared_globals()
    unknown = sorted(set(global_overrides) - set(declared))
    if unknown:
        raise SubmitError(
            f"Not settings of '{spec.name}': {', '.join(unknown)}. "
            f"It declares: {', '.join(sorted(declared))}. "
            f"For a step's own param, use step_params."
        )
    by_id = {step.id: step for step in spec.steps}
    missing = sorted(set(step_overrides) - set(by_id))
    if missing:
        raise SubmitError(
            f"No such step(s) in '{spec.name}': {', '.join(missing)}. "
            f"Its steps are: {', '.join(by_id)}"
        )
    for step_id, values in step_overrides.items():
        declared = get_step_class(by_id[step_id].step).declared_params()
        bad = sorted(set(values) - set(declared))
        if bad:
            raise SubmitError(
                f"Step '{step_id}' ({by_id[step_id].step}) does not declare "
                f"{', '.join(bad)}. It accepts: {', '.join(declared)}"
            )


def check_submission(
    global_overrides: Optional[Dict[str, Any]] = None,
    step_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    workflow: str = WORKFLOW_NATIVE,
    strict: bool = False,
) -> None:
    """Raise `SubmitError` for anything `submit_runs` would refuse.

    Everything `submit_runs` checks except the plan, which it does not
    need: whether the overrides name real settings, whether their values
    fit, and whether the outputs they select leave a deliverable.

    It exists so a caller can ask *before* putting the submission's input
    on the volume. `resolve_upload` copies a sheet there (or extracts a
    whole zip), and a refusal after that leaves the copy behind with
    nothing that will ever read it.
    """
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(workflow))
    global_overrides = dict(global_overrides or {})
    step_overrides = {k: dict(v) for k, v in (step_overrides or {}).items()}
    if strict:
        _refuse_unknown_overrides(spec, global_overrides, step_overrides)
    try:
        apply_ui_overrides(spec, global_overrides, step_overrides)
        spec.globals.update(resolve_outputs(spec))
        spec.validate()
    except ValueError as exc:
        raise SubmitError(str(exc)) from None


def submit_runs(
    scheduler: GpuScheduler,
    plan: RunPlan,
    global_overrides: Optional[Dict[str, Any]] = None,
    step_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    envs_path: str = "",
    workflow: str = WORKFLOW_NATIVE,
    strict: bool = False,
) -> List[str]:
    """Queue one run per (reference sheet, prompt) in `plan`; return their names.

    The one place a `RunJob` is built. Both front ends come through here, so
    a browser submission and an HTTP one cannot mean different things — the
    same overrides are applied, the same `requires:` rule forces the same
    outputs off, and the same workflow is validated before anything is
    queued.

    Validated *here*, before the first job is queued, rather than inside the
    worker: a bad override should fail while the caller is still listening,
    not forty minutes into a run a `pipeline.run_worker` process reports as
    merely "failed". The overrides are identical across a fanned-out batch,
    so one check covers every job it produces.

    `strict` decides what an override this workflow does not declare means:
    a typo to refuse (an API client) or a stale control to drop (a browser
    whose panel was drawn before the workflow changed under it). See
    `_refuse_unknown_overrides`.

    Returns the run names in plan order; `submit()` never blocks, so this
    returns as soon as the jobs are queued, whether or not a GPU was free.
    """
    from .cli import resolve_workflow

    if not plan:
        raise SubmitError(
            "Nothing to run: no reference sheet in this submission."
        )

    # Copied rather than mutated in place: `resolve_outputs` writes the
    # forced switches back into the overrides, and a caller's dict (the
    # UI's `param_state`) must not silently acquire them.
    global_overrides = dict(global_overrides or {})
    step_overrides = {k: dict(v) for k, v in (step_overrides or {}).items()}

    workflow_path = resolve_workflow(workflow)
    spec = WorkflowSpec.from_yaml(workflow_path)
    if strict:
        _refuse_unknown_overrides(spec, global_overrides, step_overrides)
    try:
        apply_ui_overrides(spec, global_overrides, step_overrides)
        # The deliverable switches, with every `requires:` applied and
        # "exports nothing" refused. Written back into both the spec (so
        # `validate` and the step list see the run that will actually
        # happen) and the overrides (so the worker, which re-reads the
        # pristine YAML, reaches the same answer).
        forced = resolve_outputs(spec)
        global_overrides.update(forced)
        spec.globals.update(forced)
        spec.validate()
    except ValueError as exc:
        # `ParamError` (a ValueError) for a value of the wrong type for the
        # setting it names, plain ValueError for a workflow this
        # submission's overrides have made invalid. Either way it is
        # something the caller can fix, and the exception's own message
        # already says what — so it becomes the refusal rather than being
        # restated. Without this the UI shows a traceback and the API
        # answers 500 for what is plainly a bad request. `SubmitError` is
        # not a ValueError, so `resolve_outputs`' own refusal passes
        # through with its wording intact.
        raise SubmitError(str(exc)) from None

    fanned_out = len(plan) > 1
    names: List[str] = []
    for reference_image, run_prompt in plan:
        # A fanned-out batch names each run after its image so the picker
        # reads `fast_helical_native-image1-...`; a single run keeps the bare
        # workflow prefix it always had. The stem is squeezed to
        # filename-safe chars — it becomes a directory and a log-file name.
        prefix = spec.name
        if fanned_out:
            stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(reference_image).stem).strip("-")
            prefix = f"{spec.name}-{stem}" if stem else spec.name
        run_name = timestamped_run_name(prefix)
        scheduler.submit(RunJob(
            run_name=run_name,
            workflow_name=spec.name,
            workflow_path=str(workflow_path),
            output_dir=str(output_dir() / run_name),
            envs_path=envs_path,
            global_overrides=global_overrides,
            step_overrides=step_overrides,
            reference_image=str(reference_image),
            prompt=run_prompt,
        ))
        names.append(run_name)
    logger.info(
        "queued %d run(s) of %s: %s", len(names), spec.name, ", ".join(names)
    )
    return names


# --------------------------------------------------------------------------
# what a finished run left behind
# --------------------------------------------------------------------------

def result_subdirs(workflow: str = "") -> List[str]:
    """The `dir:` of every output the named workflow declares.

    With no name — an older run whose status.json predates this, or one
    whose workflow file has since been renamed — the union across every
    shipped workflow, so a finished run is still packaged rather than
    reported as empty.

    A name that no longer resolves takes that same fallback. It has to be
    caught explicitly: `resolve_workflow` refuses with `SystemExit`, which
    is a BaseException and so passes straight through the `except` below
    and through a request handler's too — a run recorded against a
    workflow file that has since been renamed or removed (or one submitted
    by path, which is not under the workflow directory at all) turned
    packaging into a 500 rather than the documented fallback.
    """
    from .cli import available_workflows, resolve_workflow

    owned = set(DEBUG_SUBDIRS.values()) | set(DEBUG_SUBDIRS)
    return [
        directory for directory in _output_dirs(workflow).values()
        if directory not in owned
    ]


def _output_dirs(workflow: str = "") -> Dict[str, str]:
    """{output name -> its `dir:`}, for one workflow or the union of all."""
    from .cli import available_workflows, resolve_workflow

    paths = available_workflows()
    if workflow:
        try:
            paths = [resolve_workflow(workflow)]
        except SystemExit:
            logger.info(
                "run recorded workflow %r, which no longer resolves; packaging "
                "against every shipped workflow's deliverables", workflow,
            )
    dirs: Dict[str, str] = {}
    for path in paths:
        try:
            spec = WorkflowSpec.from_yaml(path)
        except (OSError, ValueError, KeyError):
            continue
        for name, directory in spec.output_dirs().items():
            dirs.setdefault(name, directory)
    return dirs


def _debug_output_names(workflow: str = "") -> List[str]:
    """Declared outputs whose `dir:` is one `debug_dirs` already carries.

    `export_debug` declares `dir: debug` so it draws as a checkbox and
    reads as an output like any other — but `debug/` is packaged by the
    debug branch, which remaps `face/` under it and deflates the text.
    Without this the archive would carry every debug member twice.
    """
    owned = set(DEBUG_SUBDIRS.values()) | set(DEBUG_SUBDIRS)
    return [name for name, directory in _output_dirs(workflow).items() if directory in owned]


def result_dirs(run_dir: Optional[Path], workflow: str = "") -> Dict[str, Path]:
    """The deliverable subdirectories this run actually produced.

    Keyed by the name they take inside the archive, which is the same name
    they have on disk — `colmap/`, `ply/` and the debug exports beside them,
    written there by the workflow's own export steps via `output_root`. The
    names come from the workflow's `outputs:` block, so a workflow that adds
    a deliverable is packaged correctly without touching this module.
    """
    if not run_dir:
        return {}
    root = Path(run_dir)
    found = {}
    for name in result_subdirs(workflow):
        candidate = root / name
        if candidate.is_dir() and any(candidate.rglob("*")):
            found[name] = candidate
    return found


# Run-directory subdirectories that ride into the archive under `debug/`,
# keyed by the name they take there. `debug/` is where the workflows point
# every step's `debug_dir` (refine_cameras' given-vs-refined camera dumps,
# the face splat's stats and depth visualisations) and, since 2026-09-05,
# where the FIRST brush training exports `intermediate_splat.ply` — the
# splat that drives the helical re-render, and so the first thing to look
# at when the re-render comes out wrong. `face/` is where the workflow
# writes the face splat .ply files, the one artefact a face-placement
# question cannot be answered without. None of these is a deliverable, so
# they are carried beside `result_dirs`' rather than declared in a
# workflow's `outputs:` block.
#
# Most of it is small — camera dumps, a ~10k-Gaussian face cap — but the
# intermediate splat is not: a 30,000-iteration training is hundreds of MB,
# and it is the reason a result .zip is now noticeably bigger than the
# deliverables alone.
DEBUG_SUBDIRS: Dict[str, str] = {"debug": "debug", "face": "debug/face"}
_DEBUG_TEXT_SUFFIXES = {".txt", ".json", ".log", ".csv"}


def debug_dirs(run_dir: Optional[Path]) -> Dict[str, Path]:
    """The debug subdirectories this run produced: {archive name -> path}.

    The same shape as `result_dirs`, and the same rule about emptiness — a
    directory a step created and then failed to fill is not offered.
    """
    if not run_dir:
        return {}
    root = Path(run_dir)
    found = {}
    for name, arcname in DEBUG_SUBDIRS.items():
        candidate = root / name
        if candidate.is_dir() and any(p.is_file() for p in candidate.rglob("*")):
            found[arcname] = candidate
    return found


def run_log_path(run_dir: Optional[Path], log_path: Optional[Path] = None) -> Optional[Path]:
    """The log file that produced `run_dir`, if it is still on the volume.

    `log_path` is what the run itself recorded in its `status.json`; the
    fallback reconstructs the name `setup_logging` would have chosen, which
    covers a run whose status file predates that field, was never written,
    or belongs to a CLI run nobody registered with the scheduler.
    """
    if log_path and Path(log_path).is_file():
        return Path(log_path)
    if run_dir:
        guess = log_dir() / f"{Path(run_dir).name}.log"
        if guess.is_file():
            return guess
    return None


def _run_members(
    run_dir: Path,
    workflow: str = "",
    log_path: Optional[Path] = None,
    prefix: str = "",
    debug: bool = True,
) -> List["tuple[str, Path, int]"]:
    """Every (arcname, source, compression) one run contributes, in order.

    One list, two readers: `_write_run_members` writes it, and
    `archive_is_current` compares its names against what a cached archive
    actually holds. Deriving both from here is what keeps "is this archive
    still right" honest — a separate reimplementation of the naming would
    drift from the writer and answer for an archive nobody builds.

    Empty for a run with no deliverables, which is what makes "a log alone
    does not make an archive" true of every caller.
    """
    directories = result_dirs(run_dir, workflow)
    if not directories:
        return []

    base = Path(prefix)
    members: List["tuple[str, Path, int]"] = []
    for name, directory in sorted(directories.items()):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                members.append((
                    str(base / name / path.relative_to(directory)),
                    path, zipfile.ZIP_STORED,
                ))
    for name, directory in sorted(debug_dirs(run_dir).items() if debug else {}.items()):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                # Text (COLMAP models, stats) deflates ~10x; the .ply
                # and .png members are the same story as the deliverables.
                text = path.suffix in _DEBUG_TEXT_SUFFIXES
                members.append((
                    str(base / name / path.relative_to(directory)), path,
                    zipfile.ZIP_DEFLATED if text else zipfile.ZIP_STORED,
                ))
    log = run_log_path(run_dir, log_path)
    if log:
        # DEFLATE for this member alone: a log is text, it shrinks by
        # ~10x, and it is small enough for the CPU cost to be nothing.
        members.append((str(base / "log.txt"), log, zipfile.ZIP_DEFLATED))
    return members


def _write_run_members(
    bundle: zipfile.ZipFile,
    run_dir: Path,
    workflow: str = "",
    log_path: Optional[Path] = None,
    prefix: str = "",
    debug: bool = True,
) -> bool:
    """Write one run's deliverables into an already-open archive.

    Shared by the single-run download and the all-runs bundle, so the two
    can never drift into carrying different things. Returns False, having
    written nothing at all, for a run with no deliverables.
    """
    members = _run_members(run_dir, workflow, log_path, prefix, debug)
    for arcname, path, compression in members:
        bundle.write(path, arcname=arcname, compress_type=compression)
    return bool(members)


# Stamped into every archive this module writes, and checked before one is
# reused. Mtimes answer "have the sources changed since"; they cannot
# answer "was this built by code that packaged the same things" — and that
# has already changed twice (log.txt, then debug/). Without the stamp, a
# run packaged before `debug/` was carried keeps handing back its old
# contents forever, while the docs say `debug/` is in there. **Bump this
# whenever `_write_run_members` changes what it writes.**
ARCHIVE_FORMAT = b"b2c-result-1: outputs + debug/ + log.txt"


def archive_is_current(
    archive: Path,
    run_dir: Path,
    workflow: str = "",
    log_path: Optional[Path] = None,
    debug: bool = True,
) -> bool:
    """True when `archive` already holds this run exactly as it stands.

    What makes rescanning a volume full of finished runs cheap: a run that
    has reached a terminal status is not going to write to its directory
    again, and the archive beside it is usually the one the Results tab
    built the first time it was looked at. Without this, the All results
    tab would re-copy every gigabyte on the volume on every press.

    Three checks, cheapest first: the `ARCHIVE_FORMAT` stamp (was this
    built by code that packaged the same things), the member list (does it
    still hold exactly what it would hold now), and modification times
    (has any surviving source changed since). Never contents — the sources
    are hundreds of megabytes of PNG, and the only writer is a run that
    has already finished.
    """
    if not archive.is_file():
        return False
    expected = [
        name for name, _path, _c in _run_members(run_dir, workflow, log_path, debug=debug)
    ]
    try:
        with zipfile.ZipFile(archive) as bundle:
            if bundle.comment != ARCHIVE_FORMAT:
                return False
            # What it holds, against what it would hold now. Mtimes cannot
            # answer this: deleting a file updates its parent's mtime and
            # nothing else, and deleting the last file in a deliverable
            # directory drops that directory out of `result_dirs` so even
            # the parent stops being looked at. Prune `ply/` off a full
            # volume and every later download kept handing back an archive
            # that still contained it.
            if bundle.namelist() != expected:
                return False
    except (OSError, zipfile.BadZipFile):
        return False  # truncated, or not an archive this module wrote
    stamp = archive.stat().st_mtime
    sources = list(result_dirs(run_dir, workflow).values())
    if debug:
        sources += list(debug_dirs(run_dir).values())
    log = run_log_path(run_dir, log_path)
    if log:
        sources.append(log)
    for source in sources:
        # Directories as well as files, and the source root itself:
        # removing a file updates its parent's mtime and nothing else, so
        # a files-only walk cannot see a deletion at all. Prune `debug/`
        # off a full volume and every later download would keep handing
        # back the cached archive that still contains it.
        paths = [source] if source.is_file() else [source, *source.rglob("*")]
        for path in paths:
            try:
                if path.stat().st_mtime > stamp:
                    return False
            except OSError:  # pruned between the walk and the stat
                continue
    return True


# One lock per archive path, so two downloads of *different* runs still
# build in parallel while two of the same one do not. Created under
# `_ARCHIVE_LOCKS_GUARD` because a plain `setdefault` on a dict from two
# threads can still hand out two different locks for one key.
_ARCHIVE_LOCKS: Dict[str, threading.Lock] = {}
_ARCHIVE_LOCKS_GUARD = threading.Lock()


def _archive_lock(archive: Path) -> threading.Lock:
    with _ARCHIVE_LOCKS_GUARD:
        return _ARCHIVE_LOCKS.setdefault(str(archive), threading.Lock())


def build_result_zip(
    run_dir: Optional[Path],
    workflow: str = "",
    log_path: Optional[Path] = None,
    reuse: bool = False,
    debug: bool = True,
) -> Optional[str]:
    """One archive holding only the deliverables: colmap/ and/or ply/.

    Not `shutil.make_archive` over the whole run directory, which is what
    this used to be. A run directory also holds the final Dataset's 81
    full-resolution frames and its pointcloud, and — for any brush training
    given an `output_dir` rather than an `export_dir` — a
    `brush/training_<ms>/` of scaffolding. Zipping all of it produced a
    multi-gigabyte download whose top level was a b2c dataset rather than
    anything COLMAP-shaped, and left the person on the other end to work
    out which parts mattered.

    The run's log rides along as `log.txt` at the top level. It is the one
    thing outside the deliverables worth carrying: an archive that comes
    back wrong is debugged from the log that produced it, and that log
    lives under B2C_LOG_DIR — a different directory on the pod, and one
    that dies with the pod, so by the time the .zip is being looked at the
    log is usually already unreachable.

    So does `debug/` — see `debug_dirs`. The face splats and the camera
    dumps in there are small, and they are exactly what a misplaced face
    was diagnosed from on 2026-09-04, when the pod that held them had
    already been released and the diagnosis had to be triangulated back
    out of the exported renders instead. Neither the log nor `debug/`
    makes an archive on its own: a run that produced no deliverable has
    nothing to download.

    Built under a temporary name and moved into place, under a per-run
    lock. The archive path is fixed — one `<run>-result.zip` per run — and
    there are now three callers that can reach the same one at once: the
    Results tab, the All-results scan, and `GET /api/v1/runs/{name}/result`
    (a sync route, so several run in the threadpool together). Two of them
    opening it `"w"` at the same time interleave into a corrupt archive,
    and `FileResponse` streaming a file another thread has just truncated
    hands a client a short .zip with a 200 on it. `os.replace` is atomic,
    so a reader either gets the old whole archive or the new one.
    """
    if not result_dirs(run_dir, workflow):
        return None

    archive = output_dir() / f"{Path(run_dir).name}-result.zip"
    with _archive_lock(archive):
        if reuse and archive_is_current(archive, Path(run_dir), workflow, log_path, debug):
            return str(archive)
        # Beside the destination, not in TMPDIR: os.replace is only atomic
        # within one filesystem, and on a pod those are different mounts.
        staging = archive.with_suffix(f".zip.{secrets.token_hex(4)}.part")
        try:
            # ZIP_STORED, not DEFLATE: PNGs and .ply files are already
            # compressed or nearly incompressible, and deflating ~2 GB of
            # them on a pod's CPU buys a couple of percent for minutes of
            # wall clock.
            with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED) as bundle:
                bundle.comment = ARCHIVE_FORMAT
                _write_run_members(bundle, Path(run_dir), workflow, log_path, debug=debug)
            os.replace(staging, archive)
        finally:
            staging.unlink(missing_ok=True)
    return str(archive)


# The combined download's name. Fixed, not timestamped: it is a copy of
# every deliverable on the volume, and a new one per press would quietly
# fill the disk the runs themselves need.
BUNDLE_NAME = "all-results.zip"


def build_bundle_zip(states: List[RunState]) -> Optional[str]:
    """Every run's deliverables in one archive, each under its own run name.

    Built from the run directories rather than by zipping up the per-run
    archives: a zip of zips would mean a second full copy of deliverables
    the volume is already holding twice.
    """
    archive = output_dir() / BUNDLE_NAME
    written = 0
    # Staged and moved into place under the same lock as a per-run archive,
    # and for the same reason — this path is fixed, so two presses of the
    # button write the same file. See `build_result_zip`.
    with _archive_lock(archive):
        staging = archive.with_suffix(f".zip.{secrets.token_hex(4)}.part")
        try:
            with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED) as bundle:
                bundle.comment = ARCHIVE_FORMAT
                for state in states:
                    if not state.output_dir:
                        continue
                    if _write_run_members(
                        bundle, Path(state.output_dir), state.workflow, state.log_path,
                        prefix=state.name, debug=wants_debug(state),
                    ):
                        written += 1
            if not written:
                archive.unlink(missing_ok=True)
                return None
            os.replace(staging, archive)
        finally:
            staging.unlink(missing_ok=True)
    return str(archive)


def run_recency(state: RunState) -> float:
    """When a run last mattered, for ordering the list newest-first."""
    if state.finished or state.started:
        return state.finished or state.started
    try:
        return Path(state.output_dir).stat().st_mtime if state.output_dir else 0.0
    except OSError:
        return 0.0


def discover_runs() -> List[RunState]:
    """Every run on this volume, newest first — not just this session's.

    `GpuScheduler` holds its runs in memory, so the **Active run** picker
    empties every time the UI process restarts while the runs themselves
    are still sitting in the output directory. This reads the volume
    instead, from two sources in descending order of what they know:

      * `run_jobs/<name>.status.json` — what the run itself published as
        it went: status, workflow, timings, step list, log path. The
        worker writes it by atomic replace, and it outlives both the
        worker and the UI process that spawned it.
      * a directory in the output root nobody published a status for — a
        `pipeline.cli` run, or one whose status file has been pruned.
        Nothing is known about it beyond what is on disk, so it is
        reported as `unknown` and packaged against the union of every
        shipped workflow's deliverables (see `result_subdirs`).
    """
    states: Dict[str, RunState] = {}
    for path in sorted(run_jobs_dir().glob("*.status.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # caught a worker mid-write, or a truncated file
        state = RunState.from_dict(data)
        if state.name:
            states[state.name] = state
    for directory in sorted(p for p in output_dir().iterdir() if p.is_dir()):
        states.setdefault(directory.name, RunState(
            name=directory.name, status="unknown", output_dir=directory,
        ))
    return sorted(states.values(), key=run_recency, reverse=True)


def completed_runs() -> List[RunState]:
    """The runs on the volume that actually produced deliverables.

    "Completed" is judged by what is on disk, not by the status a run
    published: a run cancelled after its COLMAP export still has a COLMAP
    export, and a run whose status file was pruned still has its output.
    A run still in flight is left out — its exports are the last steps,
    and packaging one mid-write hands back half a dataset.
    """
    return [
        state for state in discover_runs()
        if state.status not in ("queued", "running")
        and result_dirs(state.output_dir, state.workflow)
    ]


def wants_debug(state: RunState) -> bool:
    """Whether this run asked for its `debug/` directory in the archive.

    From what the run published, not from what is on disk: every step
    writes its debug dumps regardless, because they are a side effect of
    steps the run needs anyway — so the directory being there says nothing
    about whether it was wanted.

    A run that published no `outputs` at all predates the switch. It gets
    the debug bundle, which is what those runs were packaged with, rather
    than silently losing content on an upgrade.
    """
    for name in _debug_output_names(state.workflow):
        if name in state.outputs:
            return bool(state.outputs[name])
    return True


def run_contents(state: RunState) -> "tuple[str, int]":
    """One run's deliverables as a summary line and a total size in bytes."""
    parts, total = [], 0
    debug = wants_debug(state)
    for name, path in sorted(result_dirs(state.output_dir, state.workflow).items()):
        files = [f for f in path.rglob("*") if f.is_file()]
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:  # pruned between the walk and the stat
                continue
        parts.append(f"{name}/ ({len(files)})")
    if debug and debug_dirs(state.output_dir):
        parts.append("debug/")
    if run_log_path(state.output_dir, state.log_path):
        parts.append("log.txt")
    return ", ".join(parts), total


def merged_runs(scheduler: GpuScheduler) -> List[RunState]:
    """Every run this box knows about: the scheduler's, then the volume's.

    Neither source is complete on its own. `GpuScheduler` holds its runs in
    memory and so knows things no file records — a job still sitting in the
    queue, and which physical GPU a running one landed on — but forgets
    everything when the server restarts. `discover_runs` reads the volume,
    which outlives the process but has nothing to say about a run that has
    not published a status file yet.

    So: the volume first, the scheduler's own states written over the top.
    A live run is described by the process watching it; everything older is
    described by what it left behind.

    Ordered in flight first, then most recent. `run_recency` alone would
    put a just-queued run *last*: it has no `started`, no `finished` and no
    output directory to take an mtime from, so it scores 0. Which is the
    opposite of what a caller reading the top of this list wants — the run
    it is waiting on.
    """
    states = {state.name: state for state in discover_runs()}
    for state in scheduler.list_runs():
        if state.name:
            states[state.name] = state
    return sorted(
        states.values(),
        key=lambda state: (state.status in ("queued", "running"), run_recency(state)),
        reverse=True,
    )


def find_run(scheduler: GpuScheduler, name: str) -> Optional[RunState]:
    """One run by name, from whichever source knows about it."""
    live = scheduler.snapshot(name)
    if live.name == name:
        return live
    return next((state for state in discover_runs() if state.name == name), None)
