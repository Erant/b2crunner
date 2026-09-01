"""Gradio front end: submit work, watch it run, pull the results back.

Why this exists at all: the pipeline's real home is a rented GPU pod, where
the only things you have are a log pane and, if you set it up in advance,
an HTTP port. A run takes tens of minutes to hours, so "start it and watch
stdout over SSH" means a dropped connection loses the view of a run that is
still going. And the image is meant to be built once and not rebuilt, so
anything you might want to do on the pod has to already be in it.

One upload box, and what it holds decides what runs — there is no input
picker (`resolve_upload`). Two shapes are understood:

  * **a single reference-sheet image** — the from-scratch path, via
    `Dataset.from_reference_image` and `fast_helical_native.yaml`. One
    square image with the subject facing front on the left and seen from
    behind on the right, as a diffusion model generates it; the workflow's
    first step halves it (see steps/reference_sheet.py). One run.
  * **a .zip of image/prompt pairs** — `image1.jpg` + `image1.txt`,
    `image2.png` + `image2.txt`, ... Each image is submitted as its own
    reference-sheet run with the text file as its prompt, and `GpuScheduler`
    fans them across every GPU on the box. The "twelve subjects, four cards"
    path — one upload instead of twelve. A zip of images with no `.txt`
    files works too: each is a reference sheet and the Subject box is the
    prompt for all of them.

Both shapes run the same workflow, `fast_helical_native`, so there is no
workflow picker either.

**The form is the pipeline's, not this module's.** A workflow declares what
it exposes — a `settings:` block of typed, labelled, documented knobs and an
`outputs:` block of deliverables (see pipeline/workflow.py) — and the Run tab
draws exactly that: a Settings box, an Outputs box, and nothing else in
front. The per-step params are still all there, editable, behind one
"Per-step settings" fold in the second column, which is where they belong:
there are ~300 of them and on any given run you touch none.

This module used to hold five tables naming the globals of that one file
(OUTPUT_PARAMS, RESULT_SUBDIRS, HIDDEN_GLOBALS, RESOLUTION_CHOICES,
GLOBAL_CHOICES), each with a comment asking whoever came next to keep it in
step with the YAML. Promoting a step knob to the form is now an edit to the
workflow file alone.

The reference-sheet path is the least proven part of the pipeline — its
front half (split / render / generate_firstlast / inject_anchor) has never
executed end to end.

**Each run is its own OS process, the UI only ever polls it.** A Gradio
generator holds an SSE connection for as long as it yields, and a browser
tab surviving a three-hour run over a pod proxy is not something to design
around. So a `pipeline.run_worker` child owns the run, the UI reads a
snapshot published as JSON, and closing the tab does nothing to the work.
Reopening it and picking the run back up from the dropdown picks the view
back up.

One run per idle GPU, automatically: `GpuScheduler` (pipeline/gpu_scheduler.py)
spawns one `pipeline.run_worker` process per submitted run, `CUDA_VISIBLE_DEVICES`
-pinned to whichever physical GPU is free, and queues the rest. Two runs on
two different cards do not contend for either's VRAM — CUDA context
isolation is a property of the OS process boundary, not something this
process negotiates.

**What a run hands back** is a choice made before it starts, not after: each
Outputs checkbox writes the workflow global its export steps read through
`when:`, and a step switched off there never runs (see pipeline/workflow.py).
That matters because the .ply is a full 30,000-iteration brush training — an
hour of GPU you do not want to spend discovering you only wanted the COLMAP
dataset. The Results tab then offers exactly those: one .zip holding the
`dir:` of every output that actually got written, and nothing else.
"""

from __future__ import annotations

import logging
import re
import shutil
import signal
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import yaml

from . import steps  # noqa: F401  registers every Step; the UI is an entrypoint
from .gpu_scheduler import GpuScheduler, detect_gpu_count
from .logging_setup import timestamped_run_name
from .models import registry
from .paths import data_dir, output_dir, run_jobs_dir, upload_dir
from .run_state import PREVIEW_FRAMES, RunJob, RunState
from .step import Param
from .workflow import (
    Output, WorkflowSpec, apply_ui_overrides, load_envs, truthy,
)

logger = logging.getLogger(__name__)

# The gallery is a sanity check ("did it render a person or grey mush"),
# not a contact sheet — 81 full-resolution PNGs into a browser over a pod
# proxy is slow enough to look broken.
_GALLERY_MAX = 24

PREVIEW_ALL = "All steps"

# There is no workflow picker in the UI: every upload — an image or a zip of
# image/prompt pairs — runs this one workflow. The only pipeline knob left
# is the "Upscale dataset" toggle, which used to be the fast_helical /
# fast_helical_full split and is now one `run_upscale` global.
WORKFLOW_NATIVE = "fast_helical_native"

# Nothing about a particular workflow's settings or deliverables lives in
# this module any more: the Outputs box, the Settings box and the download's
# contents are all read off the workflow's own `settings:` / `outputs:`
# blocks (see pipeline/workflow.py). What used to be here — OUTPUT_PARAMS,
# RESULT_SUBDIRS, HIDDEN_GLOBALS, RESOLUTION_CHOICES, GLOBAL_CHOICES — was
# five tables naming the globals of one file, each with its own comment
# asking the next person to keep it in step with that file.


# --------------------------------------------------------------------------
# helpers the UI calls
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

    Raises `gr.Error` if nothing would be exported: a run that produces no
    deliverable leaves nothing to download, and it is an hour of GPU either
    way.
    """
    resolved = spec.apply_output_requirements()
    if spec.outputs and not any(resolved.values()):
        raise gr.Error(
            "Pick at least one output — a run that exports nothing leaves "
            "nothing to download."
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


# A string default long enough (or with a line break in it) that a one-line
# box is unusable — the denoise prompts are the case this exists for.
_MULTILINE_AT = 80


def _choice_label(value: Any) -> str:
    """How one entry of a param's `choices` reads in a dropdown.

    Gradio's choices have to be scalars, and a declared choice may not be:
    `resolution` picks between [width, height] pairs. Rendering a sequence
    as "720 x 1280" and mapping the label back to the value it came from
    (`_choice_map`) keeps that generic — this used to be a hardcoded
    RESOLUTION_CHOICES list plus a parser for its own label format, which
    only worked for the one setting it was written for.
    """
    if isinstance(value, (list, tuple)):
        return " x ".join(str(item) for item in value)
    return str(value)


def _choice_map(param: Param) -> Dict[str, Any]:
    """{label -> the value that label stands for} for a param's choices."""
    return {_choice_label(choice): choice for choice in param.choices}


def _widget_for(param: Param, value: Any, label: str, key: str):
    """One Gradio control for one declared param, prefilled with `value`.

    Everything the widget needs — type, bounds, choices, help — comes off
    the Param, which is why declaring params was worth doing: the UI has no
    per-step knowledge in it at all.

    `key` is a stable identity for this widget within the `@gr.render` block.
    Gradio uses it to reuse the same DOM element across re-renders instead of
    destroying and recreating it, which also preserves a value the user typed
    but has not yet committed. The keys embed the workflow name, so switching
    workflows changes every key and forces a full redraw — matching
    `on_workflow_change`, which clears the override state at the same moment.
    """
    info = param.help or None
    # `interactive=True` on every widget: these are created inside a
    # `@gr.render` block, where Gradio does not reliably infer that a
    # component wired as an event input should be editable. Constructed with
    # a `value=`, it otherwise defaults to display-only and the whole panel
    # renders read-only.
    if param.type is bool:
        return gr.Checkbox(value=bool(value), label=label, info=info,
                           interactive=True, key=key)
    if param.type in (int, float):
        if param.minimum is not None and param.maximum is not None:
            return gr.Slider(
                minimum=param.minimum, maximum=param.maximum,
                step=1 if param.type is int else None,
                value=value, label=label, info=info, interactive=True, key=key,
            )
        return gr.Number(
            value=value, label=label, info=info,
            precision=0 if param.type is int else None, interactive=True, key=key,
        )
    # Choices are checked before `type`, so a list-typed setting that
    # declares them (resolution) gets a dropdown rather than a YAML box a
    # user can type an unsupported shape into.
    if param.choices:
        labels = list(_choice_map(param))
        current = _choice_label(value) if value is not None else None
        if current is not None and current not in labels:
            # A workflow set something outside its own choice list. Show it
            # rather than silently snapping the control to another value.
            labels = [current, *labels]
        return gr.Dropdown(
            choices=labels, value=current, label=label, info=info,
            # A scalar choice list is a suggestion — a step param can take a
            # value nobody thought to list. A non-scalar one is not: there is
            # no sane way to type "[720, 1280]" into a dropdown.
            allow_custom_value=param.type is not list,
            interactive=True, key=key,
        )
    if param.type in (list, dict):
        shape = "YAML list" if param.type is list else "YAML mapping"
        return gr.Textbox(
            value=yaml.safe_dump(value, default_flow_style=True).strip(),
            label=label, info=(info + f" ({shape})") if info else shape,
            interactive=True, key=key,
        )
    text = value if isinstance(value, str) else ""
    multiline = "\n" in text or len(text) > _MULTILINE_AT
    return gr.Textbox(
        value=text, label=label, info=info, lines=6 if multiline else 1,
        interactive=True, key=key,
    )


def _widget_value(param: Param, raw: Any) -> Any:
    """Bring a widget's value back to the param's declared type.

    Two cases need real work: a dropdown hands back the label it drew, not
    the value behind it, and a list arrives as YAML text. The rest is left
    to `Step.resolve_params` and `WorkflowSpec.coerce_global`, which coerce
    at run time anyway; doing it twice would just mean two places to
    disagree.
    """
    if param.choices:
        mapping = _choice_map(param)
        if raw in mapping:
            return mapping[raw]
        # `allow_custom_value` lets someone type a value that is not on the
        # list. Fall through to the type handling below so it is still
        # brought to the right shape.
    if param.type in (list, dict):
        try:
            parsed = yaml.safe_load(raw) if isinstance(raw, str) else raw
        except yaml.YAMLError as exc:
            raise gr.Error(f"{param.name}: not valid YAML ({exc})")
        if param.type is dict:
            # An empty box is an empty mapping, not a refusal: `{}` is the
            # declared default of every dict param there is, and yaml reads
            # a blank string as None.
            if parsed is None:
                return {}
            if not isinstance(parsed, dict):
                raise gr.Error(f"{param.name}: expected a YAML mapping, got {raw!r}")
            return dict(parsed)
        if not isinstance(parsed, (list, tuple)):
            raise gr.Error(f"{param.name}: expected a YAML list, got {raw!r}")
        return list(parsed)
    return raw


def workflow_summary(name: str) -> str:
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    rows = [
        f"| {i} | `{s.id}` | `{s.step}` | {s.dispatch}{':' + s.env if s.env else ''} |"
        for i, s in enumerate(spec.steps, start=1)
    ]
    return (
        f"**{spec.name}** — {spec.description.strip()}\n\n"
        "| # | id | step | dispatch |\n|---|---|---|---|\n" + "\n".join(rows)
    )


# Image extensions the single upload box understands, bare or inside a zip.
# Kept in step with what `cv2.imread` (via `Dataset.from_reference_image`)
# can decode.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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
            if not str(resolved).startswith(str(target.resolve())):
                raise gr.Error(f"Refusing to extract {member!r}: it escapes the target directory.")
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
    images = sorted(p for p in _iter_files(root) if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        raise gr.Error(
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
        raise gr.Error(
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
    if suffix in _IMAGE_SUFFIXES:
        return [(save_upload(upload_path, "reference"), prompt)]
    if suffix != ".zip":
        raise gr.Error(
            f"Don't know what to do with a {suffix or 'no-extension'} file — upload "
            "a .zip of image/prompt pairs or a single reference image."
        )

    root = _guarded_extract(
        upload_path, upload_dir() / f"upload-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    return list(pair_images_with_prompts(root, prompt))


def save_upload(path: str, prefix: str) -> Path:
    """Copy a Gradio upload onto the volume; its own temp dir is not durable."""
    destination = upload_dir() / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}{Path(path).suffix}"
    shutil.copy(path, destination)
    return destination


def result_subdirs(workflow: str = "") -> List[str]:
    """The `dir:` of every output the named workflow declares.

    With no name — an older run whose status.json predates this, or one
    whose workflow file has since been renamed — the union across every
    shipped workflow, so a finished run is still packaged rather than
    reported as empty.
    """
    from .cli import available_workflows, resolve_workflow

    paths = [resolve_workflow(workflow)] if workflow else available_workflows()
    names: List[str] = []
    for path in paths:
        try:
            spec = WorkflowSpec.from_yaml(path)
        except (OSError, ValueError, KeyError):
            continue
        for directory in spec.output_dirs().values():
            if directory not in names:
                names.append(directory)
    return names


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


def build_result_zip(run_dir: Optional[Path], workflow: str = "") -> Optional[str]:
    """One archive holding only the deliverables: colmap/ and/or ply/.

    Not `shutil.make_archive` over the whole run directory, which is what
    this used to be. A run directory also holds the final Dataset's 81
    full-resolution frames, its pointcloud, and — when the workflow trained
    an intermediate splat — a `brush/training_<ms>/` that is scaffolding,
    not output. Zipping all of it produced a multi-gigabyte download whose
    top level was a b2c dataset rather than anything COLMAP-shaped, and
    left the person on the other end to work out which parts mattered.
    """
    directories = result_dirs(run_dir, workflow)
    if not directories:
        return None

    archive = output_dir() / f"{Path(run_dir).name}-result.zip"
    # ZIP_STORED, not DEFLATE: PNGs and .ply files are already compressed or
    # nearly incompressible, and deflating ~2 GB of them on a pod's CPU
    # buys a couple of percent for minutes of wall clock.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, directory in sorted(directories.items()):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=str(Path(name) / path.relative_to(directory)))
    return str(archive)


def gallery_images(directory: Optional[Path]) -> List[str]:
    if not directory or not Path(directory).exists():
        return []
    frames = sorted(Path(directory).glob("frame_*.png"))
    if not frames:
        frames = sorted(Path(directory).glob("*.png"))
    if len(frames) <= _GALLERY_MAX:
        return [str(p) for p in frames]
    stride = max(1, len(frames) // _GALLERY_MAX)
    return [str(p) for p in frames[::stride][:_GALLERY_MAX]]


def preview_gallery(state: RunState, step_filter: str = "") -> List[Any]:
    """The per-step contact sheet, as (path, caption) pairs in run order.

    Captions carry the step's position and id, so at `columns=8` each row
    is one step and the sheet reads top to bottom as the run did.
    """
    items: List[Any] = []
    for record in state.steps:
        label = f"{record.index:02d} {record.step_id}"
        if step_filter and step_filter != PREVIEW_ALL and label != step_filter:
            continue
        for path in record.previews:
            items.append((path, f"{label} · {Path(path).stem.split('_', 1)[-1]}"))
    return items


def preview_step_choices(state: RunState) -> List[str]:
    return [PREVIEW_ALL] + [
        f"{record.index:02d} {record.step_id}"
        for record in state.steps if record.previews
    ]


def _format_status(state: RunState) -> str:
    if state.status == "idle":
        return "### idle\nPick a run above, or start one on the **Run** tab."
    if state.status == "queued":
        return f"### 🕓 queued — `{state.name}`\nWaiting for a free GPU."

    elapsed = (state.finished or time.time()) - state.started if state.started else 0.0
    icon = {"running": "⏳", "done": "✅", "failed": "❌", "cancelled": "⛔"}.get(state.status, "")
    lines = [
        f"### {icon} {state.status} — `{state.name}`"
        + (f" · **GPU** {state.gpu_index}" if state.gpu_index is not None else ""),
        f"**workflow** `{state.workflow}` · **elapsed** {elapsed:.0f}s "
        f"· **step** {state.current}/{state.total}",
    ]
    if state.message:
        lines.append(f"\n{state.message}")
    if state.output_dir:
        lines.append(f"\n**output** `{state.output_dir}`")
    if state.log_path:
        lines.append(f"**log** `{state.log_path}`")
    return "\n\n".join(lines)


def _fleet_status(scheduler: GpuScheduler) -> str:
    slots = scheduler.gpu_status()
    busy = sum(1 for slot in slots if slot["busy"])
    queued = scheduler.queued_count()
    parts = [f"**{busy} of {len(slots)}** GPU{'s' if len(slots) != 1 else ''} busy"]
    if queued:
        parts.append(f"**{queued}** queued")
    detail = ", ".join(
        f"gpu{slot['gpu']}: {slot['run'] if slot['busy'] else 'idle'}" for slot in slots
    )
    return " · ".join(parts) + (f"  \n_{detail}_" if detail else "")


def _run_choices(scheduler: GpuScheduler) -> List[tuple[str, str]]:
    icons = {"queued": "🕓", "running": "⏳", "done": "✅", "failed": "❌", "cancelled": "⛔"}
    return [
        (
            f"{icons.get(s.status, '?')} {s.name}"
            + (f" (gpu {s.gpu_index})" if s.gpu_index is not None else ""),
            s.name,
        )
        for s in reversed(scheduler.list_runs())
    ]


def _step_rows(state: RunState) -> List[List[Any]]:
    icons = {"pending": "·", "running": "▶", "done": "✓", "failed": "✗", "skipped": "–"}
    return [
        [record.index, icons.get(record.status, "?"), record.step_id, record.step_name,
         f"{record.elapsed:.1f}s" if record.elapsed else ""]
        for record in state.steps
    ]


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

def build_app(envs_path: str, gpu_count: Optional[int] = None) -> gr.Blocks:
    scheduler = GpuScheduler(
        gpu_count=gpu_count if gpu_count is not None else detect_gpu_count(),
        work_dir=run_jobs_dir(),
    )
    default_workflow = WORKFLOW_NATIVE

    with gr.Blocks(title="b2c_runner", analytics_enabled=False) as app:
        gr.Markdown("# b2c_runner\nBody2COLMAP pipeline — submit a run, watch it, collect the output.")

        # Shared across every tab: which run the Progress/Results controls
        # below are currently looking at. A run submitted on the Run tab
        # selects itself here automatically; picking a different one from
        # the dropdown re-points every other tab at it.
        with gr.Row():
            run_picker = gr.Dropdown(
                choices=[], value=None, label="Active run", scale=3,
                info="Every run this session has seen, most recent first.",
            )
            refresh_runs_btn = gr.Button("Refresh run list", size="sm", scale=1)
        fleet_out = gr.Markdown(_fleet_status(scheduler))

        # Keeps the app object usable as a handle onto its own scheduler —
        # `launch()` needs it to forward a shutdown signal to every worker
        # this process spawned.
        app._gpu_scheduler = scheduler  # noqa: SLF001

        with gr.Tab("Run"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_in = gr.File(
                        label="Input",
                        file_types=[".zip", *sorted(_IMAGE_SUFFIXES)],
                        file_count="single",
                    )
                    gr.Markdown(
                        "_Upload **one** of:_\n"
                        "- _a **`.zip` of `image1.jpg` + `image1.txt` pairs** — one "
                        "run per pair, fanned across every GPU;_\n"
                        "- _a single **reference-sheet image** — one run._\n\n"
                        "_Either shape runs `fast_helical_native`._"
                    )
                    # No picker: there is only one shipped pipeline.
                    workflow_in = gr.Dropdown(
                        [WORKFLOW_NATIVE], value=default_workflow,
                        label="Pipeline", interactive=False,
                        info="The params panel and Outputs below follow it.",
                    )
                    prompt_in = gr.Textbox(
                        label="Subject description",
                        placeholder="a woman in a red jacket",
                        info="Fills $SUBJECT_DESC$ in the denoise prompt. The "
                             "fallback when a pair's .txt is missing or empty.",
                    )
                    # Only what the user actually changed, in the two
                    # namespaces a run resolves: {"globals": {...},
                    # "steps": {step_id: {...}}}. Everything untouched stays
                    # owned by the workflow file and the step defaults,
                    # which is what keeps output_root's repoint working and
                    # what makes "reset" mean something.
                    param_state = gr.State({"globals": {}, "steps": {}})

                    def _record(scope: str, key: str, param, baseline, step_id=""):
                        """A .change() handler that files one widget's value.

                        A value equal to what the panel was drawn with is
                        *removed* rather than stored, so nudging a control
                        and putting it back leaves nothing pinned.
                        """
                        def handler(raw, state):
                            value = _widget_value(param, raw)
                            state = {
                                "globals": dict(state.get("globals", {})),
                                "steps": {k: dict(v) for k, v in state.get("steps", {}).items()},
                            }
                            bucket = (
                                state["globals"] if scope == "globals"
                                else state["steps"].setdefault(step_id, {})
                            )
                            if value == baseline:
                                bucket.pop(key, None)
                            else:
                                bucket[key] = value
                            return state
                        return handler

                    def _control(param, value, key, scope, step_id=""):
                        """Draw one param and wire it to `param_state`."""
                        widget = _widget_for(param, value, param.title, key=key)
                        widget.change(
                            _record(scope, param.name, param, value, step_id),
                            inputs=[widget, param_state], outputs=[param_state],
                        )
                        return widget

                    # The two boxes a person actually uses, both drawn from
                    # the workflow's own `settings:` and `outputs:` blocks.
                    # The 300-odd per-step knobs live in the other column,
                    # behind one fold.
                    @gr.render(inputs=[workflow_in])
                    def render_settings(name):
                        spec, settings, outputs, _steps = workflow_param_panel(name)
                        values = spec.globals

                        in_outputs = [s for s in settings if s.group == "outputs"]
                        plain = [s for s in settings
                                 if s.group != "outputs" and not s.advanced]
                        advanced = [s for s in settings
                                    if s.group != "outputs" and s.advanced]

                        def draw(param):
                            return _control(
                                param, values.get(param.name, param.default),
                                f"{name}:globals:{param.name}", "globals",
                            )

                        with gr.Accordion("Settings", open=True):
                            if not plain and not advanced:
                                gr.Markdown("_This pipeline declares no settings._")
                            for param in plain:
                                draw(param)
                            if advanced:
                                with gr.Accordion("More settings", open=False):
                                    for param in advanced:
                                        draw(param)

                        with gr.Accordion("Outputs", open=True):
                            if not outputs:
                                gr.Markdown("_This pipeline declares no outputs._")
                            # A setting in the outputs group is drawn first:
                            # it modifies the deliverables rather than the
                            # run (the SeedVR2 upscale is the case), and the
                            # checkboxes below can depend on it.
                            widgets = {param.name: draw(param) for param in in_outputs}
                            for output in outputs:
                                param = output.as_param()
                                required = (
                                    values.get(output.requires)
                                    if output.requires else True
                                )
                                widgets[output.name] = _control(
                                    param, values.get(output.name, output.default),
                                    f"{name}:globals:{output.name}", "globals",
                                )
                                if output.requires:
                                    widgets[output.name].interactive = truthy(required)

                            # `requires:` in the UI: a deliverable that is
                            # only meaningful alongside another setting
                            # follows that setting's control. The run-side
                            # half of the same rule is `resolve_outputs`,
                            # which forces it off whatever the panel says.
                            for output in outputs:
                                source = widgets.get(output.requires)
                                if source is None:
                                    continue
                                source.change(
                                    lambda on: gr.update(interactive=truthy(on)),
                                    inputs=[source], outputs=[widgets[output.name]],
                                )

                            gr.Markdown(
                                "_What the run produces, and what the Results tab's "
                                ".zip contains. Each box switches its export steps "
                                "off entirely — unticking the .ply skips a whole "
                                "30,000-iteration brush training._"
                            )

                    with gr.Row():
                        start_btn = gr.Button("Start run", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                with gr.Column(scale=1):
                    summary_out = gr.Markdown(workflow_summary(default_workflow))

                    # Everything the pipeline did not promote to a setting:
                    # one accordion per step, each holding that step's own
                    # declared params. Folded away by default and folded
                    # again inside — this is ~300 controls, and the reason
                    # the Settings and Outputs boxes exist.
                    with gr.Accordion("Per-step settings", open=False):
                        gr.Markdown(
                            "_The knobs the pipeline does not expose. A dot marks "
                            "one the workflow sets; the rest show the step's own "
                            "default. A param wired to a pipeline setting is not "
                            "shown here — its one home is the Settings box. "
                            "Advanced holds the knobs that exist because the "
                            "underlying library has them._"
                        )

                        @gr.render(inputs=[workflow_in])
                        def render_steps(name):
                            _spec, _settings, _outputs, step_entries = (
                                workflow_param_panel(name)
                            )
                            for entry in step_entries:
                                label = f"{entry['id']}  ({entry['step']})"
                                with gr.Accordion(label, open=False):
                                    # Params wired to a setting are dropped, not
                                    # drawn read-only: the mirror never tracked
                                    # the Settings control, so it only ever added
                                    # a stale, confusing second value.
                                    refs = entry["global_refs"]
                                    plain = [p for p in entry["params"]
                                             if not p.advanced and p.name not in refs]
                                    advanced = [p for p in entry["params"]
                                                if p.advanced and p.name not in refs]

                                    def draw(param, entry=entry, name=name):
                                        overridden = param.name in entry["overrides"]
                                        value = (
                                            entry["overrides"][param.name] if overridden
                                            else param.default
                                        )
                                        widget = _widget_for(
                                            param, value,
                                            param.title + (" •" if overridden else ""),
                                            key=f"{name}:steps:{entry['id']}:{param.name}",
                                        )
                                        widget.change(
                                            _record("steps", param.name, param, value,
                                                    entry["id"]),
                                            inputs=[widget, param_state],
                                            outputs=[param_state],
                                        )

                                    for param in plain:
                                        draw(param)
                                    if advanced:
                                        with gr.Accordion("Advanced", open=False):
                                            for param in advanced:
                                                draw(param)

        with gr.Tab("Progress"):
            status_out = gr.Markdown(_format_status(RunState()))
            progress_out = gr.Slider(
                minimum=0, maximum=1, value=0, label="Progress", interactive=False,
            )
            steps_out = gr.Dataframe(
                headers=["#", "", "id", "step", "time"],
                datatype=["number", "str", "str", "str", "str"],
                interactive=False,
                label="Steps",
            )
            log_out = gr.Textbox(
                label="Log", lines=24, max_lines=24, autoscroll=True, interactive=False,
            )
            attach_btn = gr.Button(
                "Attach / refresh",
                variant="secondary",
            )
            gr.Markdown(
                "_Each run is its own process, pinned to whichever GPU picked it up. "
                "Closing this tab does not stop it — reopen the page, pick it from "
                "**Active run** above, and press **Attach / refresh** to pick the view "
                "back up._"
            )

        with gr.Tab("Results"):
            results_refresh = gr.Button("Load latest results", variant="primary")
            results_info = gr.Markdown()
            results_zip = gr.File(label="Result (.zip)")
            results_gallery = gr.Gallery(label="Final frames", columns=6, height=400)

            gr.Markdown(
                f"### Per-step frames\n_{PREVIEW_FRAMES} frames spaced evenly "
                "through the batch, captured after every step, so a step that "
                "breaks the output can be identified by looking rather than by "
                "reading the log. One row per step, in run order. These appear "
                "while the run is going — the button above keeps refreshing "
                "until it ends._"
            )
            preview_step_in = gr.Dropdown(
                choices=[PREVIEW_ALL], value=PREVIEW_ALL, label="Step",
                info="Narrow the sheet to one step to see its frames larger.",
            )
            preview_gallery_out = gr.Gallery(
                label="Per-step frames", columns=PREVIEW_FRAMES, height=600,
                object_fit="contain",
            )

        with gr.Tab("Models"):
            gr.Markdown(
                "Checkpoints are pulled at pod start and cached on the volume, so a "
                "reused network volume only pays for this once. A run blocks until the "
                "ones **its own workflow** needs are present — it will not stall "
                "mid-pipeline waiting for a download."
            )
            models_refresh = gr.Button("Refresh", variant="primary")
            models_out = gr.Dataframe(
                headers=["", "model", "size", "needed by", "detail"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
            )

        with gr.Tab("Doctor"):
            gr.Markdown(
                "Checks the things that break a run 40 minutes in: Vulkan (brush), EGL "
                "(render), the per-step venvs, the brush binaries' fork-specific flags, "
                "the HF token's access to the two gated checkpoints, and free space."
            )
            doctor_btn = gr.Button("Run checks", variant="primary")
            doctor_out = gr.Code(label="Report", lines=30)

        # -- wiring --------------------------------------------------------
        def on_start(upload_file, prompt, params):
            params = params or {}
            global_overrides = dict(params.get("globals", {}))
            step_overrides = {k: dict(v) for k, v in params.get("steps", {}).items()}

            if not upload_file:
                raise gr.Error(
                    "Upload something: a .zip of image/prompt pairs, or a "
                    "single reference-sheet image."
                )
            path = upload_file if isinstance(upload_file, str) else upload_file.name
            plan = resolve_upload(path, prompt or "")
            workflow = WORKFLOW_NATIVE

            from .cli import resolve_workflow

            workflow_path = resolve_workflow(workflow)
            spec = WorkflowSpec.from_yaml(workflow_path)
            # Validated here, before any job is queued, so a bad override
            # fails at submit time — in a button handler the user is looking
            # at — instead of forty minutes into a queued run a
            # `pipeline.run_worker` process reports as merely "failed". The
            # overrides are identical across a fanned-out batch, so one
            # check covers every job it produces.
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

            fanned_out = len(plan) > 1

            def _submit(reference_image: Path, run_prompt: str) -> str:
                # A fanned-out batch names each run after its image so the
                # picker reads `fast_helical_native-image1-...`; a single run
                # keeps the bare workflow prefix it always had. The stem is
                # squeezed to filename-safe chars — it becomes a directory
                # and a log-file name.
                prefix = spec.name
                if fanned_out:
                    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", reference_image.stem).strip("-")
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
                return run_name

            first = None
            for reference_image, run_prompt in plan:
                name = _submit(reference_image, run_prompt)
                first = first or name
            # Point the shared picker at the first run; the rest are one
            # dropdown-hop away and already fanning out across the other GPUs.
            return gr.update(choices=_run_choices(scheduler), value=first)

        def stream(run_name: Optional[str]):
            """Poll the selected run until it ends, then one final update.

            A generator rather than a Timer so it works the same on every
            Gradio 4.x/5.x, and it deliberately reads shared state (a
            `RunState` the scheduler last read off that run's `status.json`)
            instead of driving the run: if this connection dies, the
            `pipeline.run_worker` process it is watching does not.
            """
            if not run_name:
                state = RunState()
                yield (_format_status(state), gr.update(value=0), _step_rows(state),
                       "", _fleet_status(scheduler))
                return
            while True:
                state = scheduler.snapshot(run_name)
                fraction = (state.current / state.total) if state.total else 0
                yield (
                    _format_status(state),
                    gr.update(value=fraction),
                    _step_rows(state),
                    scheduler.log_text(run_name),
                    _fleet_status(scheduler),
                )
                if state.status not in ("queued", "running"):
                    break
                time.sleep(1.0)

        progress_outputs = [status_out, progress_out, steps_out, log_out, fleet_out]

        start_btn.click(
            on_start,
            inputs=[upload_in, prompt_in, param_state],
            outputs=[run_picker],
        ).then(stream, inputs=[run_picker], outputs=progress_outputs)

        attach_btn.click(stream, inputs=[run_picker], outputs=progress_outputs)
        refresh_runs_btn.click(
            lambda: gr.update(choices=_run_choices(scheduler)), outputs=[run_picker]
        )
        cancel_btn.click(
            lambda run_name: scheduler.cancel(run_name) if run_name else None,
            inputs=[run_picker], outputs=[],
        )

        def on_results(run_name: Optional[str]):
            state = scheduler.snapshot(run_name) if run_name else RunState()
            directory = state.output_dir
            if not directory or not Path(directory).exists():
                return "No output directory yet — pick or start a run first.", [], None

            directories = result_dirs(Path(directory), state.workflow)
            if not directories:
                # Distinguish "still running" from "ran and produced nothing":
                # the second means the export steps were switched off or
                # failed, and looking in the run directory is the next move.
                note = (
                    "still running — the exports are the last steps"
                    if state.status in ("queued", "running")
                    else "the run produced neither; check the Progress tab's step list"
                )
                wanted = ", ".join(
                    f"`{sub}/`" for sub in result_subdirs(state.workflow)
                ) or "any deliverable"
                return (
                    f"### `{directory}`\n\nNo {wanted} here yet — {note}.",
                    gallery_images(directory),
                    None,
                )

            lines = []
            for name, path in sorted(directories.items()):
                files = [f for f in path.rglob("*") if f.is_file()]
                size = sum(f.stat().st_size for f in files)
                lines.append(f"- **`{name}/`** — {len(files)} files, {size / 1e9:.2f} GB")

            archive = build_result_zip(Path(directory), state.workflow)
            info = (
                f"### `{directory}`\n\nThe .zip below contains:\n\n"
                + "\n".join(lines)
                + "\n\n_The full run directory — the final dataset's frames, the "
                "point cloud, any intermediate brush training — stays on the volume "
                "at the path above; only the deliverables are packaged._"
            )
            return info, gallery_images(directory), archive

        def stream_results(run_name: Optional[str], step_filter: str):
            """Poll the Results tab for as long as the selected run is going.

            Same shape as `stream()` on the Progress tab and for the same
            reason: the per-step frames are worth watching *during* a
            two-hour run, and a generator that reads shared state cannot
            take the run down with it if the connection drops.

            Only the previews change while a run is in flight — the archive
            is not built until there is something to put in it — so the
            expensive half is skipped until the end.
            """
            if not run_name:
                yield ("Pick a run above.", [], None,
                       gr.update(choices=[PREVIEW_ALL]), [])
                return
            while True:
                state = scheduler.snapshot(run_name)
                active = state.status in ("queued", "running")
                if active:
                    finished = (gr.update(), gr.update(), gr.update())
                else:
                    finished = on_results(run_name)
                yield (
                    *finished,
                    gr.update(choices=preview_step_choices(state)),
                    preview_gallery(state, step_filter),
                )
                if not active:
                    break
                time.sleep(3.0)

        results_outputs = [results_info, results_gallery, results_zip,
                           preview_step_in, preview_gallery_out]

        results_refresh.click(
            stream_results, inputs=[run_picker, preview_step_in], outputs=results_outputs
        )

        def on_preview_filter(run_name: Optional[str], step_filter: str):
            state = scheduler.snapshot(run_name) if run_name else RunState()
            return preview_gallery(state, step_filter)

        preview_step_in.change(
            on_preview_filter, inputs=[run_picker, preview_step_in],
            outputs=preview_gallery_out,
        )

        def on_models():
            from .models import read_status

            icons = {"ready": "✓", "fetching": "⏳", "pending": "·", "failed": "✗"}
            known = registry()
            return [
                [icons.get(str(entry["status"]), "?"), key,
                 f"{known[key].approx_gb:.1f} GB",
                 ", ".join(known[key].steps),
                 str(entry.get("detail", ""))[:80]]
                for key, entry in sorted(read_status().items())
            ]

        models_refresh.click(on_models, outputs=models_out)

        def on_doctor():
            from .doctor import format_report, run_checks

            return format_report(run_checks(load_envs(envs_path)), verbose=True)

        doctor_btn.click(on_doctor, outputs=doctor_out)

    return app


def launch(
    host: str = "0.0.0.0", port: int = 7860, envs_path: str = "", share: bool = False,
    gpu_count: Optional[int] = None,
) -> None:
    from .cli import DEFAULT_ENVS

    app = build_app(envs_path or DEFAULT_ENVS, gpu_count=gpu_count)
    scheduler: GpuScheduler = app._gpu_scheduler  # noqa: SLF001
    logger.info("web UI on http://%s:%d (%d GPU worker slot(s))", host, port, scheduler.gpu_count)

    # Nothing forwards SIGTERM/SIGINT to a child this process spawned with
    # `subprocess.Popen` on its own — Python does not do that for you. On a
    # pod stop, without this, every in-flight `pipeline.run_worker` would be
    # orphaned holding a CUDA context until the container's whole cgroup is
    # torn down.
    def _shutdown(signum, frame) -> None:
        logger.info("received signal %s; stopping %d GPU worker(s)", signum, scheduler.gpu_count)
        scheduler.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # queue() is what makes the streaming generator above work at all. The
    # concurrency limit is UI-event concurrency (how many browser
    # connections Gradio itself services at once), unrelated to how many
    # GPU workers the scheduler runs — that is bounded by `gpu_count`.
    app.queue(default_concurrency_limit=4).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        # Uploads and result downloads live on the volume, which is outside
        # Gradio's default allowlist (its own temp dir plus the cwd).
        allowed_paths=[str(data_dir()), str(output_dir()), str(upload_dir())],
    )
