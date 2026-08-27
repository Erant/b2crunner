"""Gradio front end: submit work, watch it run, pull the results back.

Why this exists at all: the pipeline's real home is a rented GPU pod, where
the only things you have are a log pane and, if you set it up in advance,
an HTTP port. A run takes tens of minutes to hours, so "start it and watch
stdout over SSH" means a dropped connection loses the view of a run that is
still going. And the image is meant to be built once and not rebuilt, so
anything you might want to do on the pod has to already be in it.

Three shapes of input, per what the pipeline can start from:

  * **an existing dataset directory** on the mounted volume — the path every
    verification run so far has used (cyber_6f/initial and friends);
  * **an uploaded .zip** of such a directory — the same thing when the
    dataset lives on your laptop rather than the pod;
  * **a front/back reference sheet** — the from-scratch path, via
    `Dataset.from_reference_image` and `fast_helical_native.yaml`. One
    square image with the subject facing front on the left and seen from
    behind on the right, as a diffusion model generates it; the workflow's
    first step halves it (see steps/reference_sheet.py).

Only that last workflow takes a sheet; both `fast_helical` files begin from
a complete dataset. Picking the sheet input against one of those is refused
at submit time (see `workflow_needs_a_dataset`, which reads it off the
steps rather than off a flag) instead of failing on a bare KeyError one
step in.

The sheet path is the least proven of the three — its front half (split /
render / generate_firstlast / inject_anchor) has never executed end to end
— which is exactly why the other two are here: if it falls over, the rest
of the pipeline is still exercisable from a zip without touching the
image.

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

**What a run hands back** is a choice made before it starts, not after: the
Outputs checkboxes set the workflow's `export_colmap` / `export_ply` params,
and a step switched off there never runs (see `when:` in
pipeline/workflow.py). That matters because the .ply is a full
30,000-iteration brush training — an hour of GPU you do not want to spend
discovering you only wanted the COLMAP dataset. The Results tab then offers
exactly those: one .zip holding `colmap/` and/or `ply/`, and nothing else.
"""

from __future__ import annotations

import logging
import shutil
import signal
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import yaml

from . import steps  # noqa: F401  registers every Step; the UI is an entrypoint
from .dataset import find_dataset_root
from .gpu_scheduler import GpuScheduler, detect_gpu_count
from .logging_setup import timestamped_run_name
from .models import registry
from .paths import data_dir, output_dir, run_jobs_dir, upload_dir
from .run_state import PREVIEW_FRAMES, RunJob, RunState
from .step import Param
from .workflow import WorkflowSpec, apply_ui_overrides, load_envs

logger = logging.getLogger(__name__)

# The gallery is a sanity check ("did it render a person or grey mush"),
# not a contact sheet — 81 full-resolution PNGs into a browser over a pod
# proxy is slow enough to look broken.
_GALLERY_MAX = 24

SOURCE_DIRECTORY = "Dataset directory on this machine"
SOURCE_ZIP = "Upload a dataset .zip"
SOURCE_PHOTO = "Front/back reference sheet"

PREVIEW_ALL = "All steps"

# The two things a finished run can hand back, as a checkbox label -> the
# workflow param that switches it on, and the subdirectory of the run it
# lands in. A workflow opts in by declaring those params (fast_helical and
# fast_helical_full both do); one that declares neither just doesn't show
# the control.
OUTPUT_COLMAP = "COLMAP dataset"
OUTPUT_PLY = "Trained .ply (normal-supervised)"
OUTPUT_PARAMS = {OUTPUT_COLMAP: "export_colmap", OUTPUT_PLY: "export_ply"}
OUTPUT_SUBDIRS = {OUTPUT_COLMAP: "colmap", OUTPUT_PLY: "ply"}


# --------------------------------------------------------------------------
# helpers the UI calls
# --------------------------------------------------------------------------

def discover_datasets(root: Optional[Path] = None, max_depth: int = 3) -> List[str]:
    """Directories under the volume that look like b2c datasets.

    Bounded depth because the volume also holds the HF cache, which is tens
    of thousands of files and has no metadata.json anywhere in it.
    """
    base = root or data_dir()
    found: List[str] = []
    if not base.exists():
        return found
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + ["metadata.json"]) if depth else "metadata.json"
        for match in base.glob(pattern):
            parent = str(match.parent)
            if parent not in found:
                found.append(parent)
    return sorted(found)


def workflow_choices() -> List[str]:
    from .cli import available_workflows

    return [p.stem for p in available_workflows()]


def workflow_outputs(name: str) -> List[str]:
    """Which of the Outputs checkboxes this workflow actually understands."""
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    return [label for label, param in OUTPUT_PARAMS.items() if param in spec.globals]


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


# Globals the generated panel deliberately does not draw a control for.
#
# The output switches have a dedicated control (the Outputs checkboxes), and
# showing them here too would give the same setting two editable homes that
# disagree the moment someone touches one.
#
# output_root is excluded for a sharper reason: `start()` only repoints it at
# the run's own timestamped directory when the submitted overrides do NOT
# contain the key (mirroring `pipeline.cli run`'s `--param output_root=...`).
# When the panel was a YAML box showing the workflow's literal default
# (`output/fast_helical`, relative to the process's cwd), every run
# round-tripped it back whether the box was touched or not, permanently
# defeating that repoint — colmap_export and the final brush training wrote
# under the cwd instead of the run directory, and the Results tab reported
# "the run produced neither" even though the run had completed and written
# real output, just not where anything was looking for it. Submitting only
# what the user actually changed means the same thing cannot happen again,
# but leaving it off the panel keeps it impossible rather than merely
# unlikely.
HIDDEN_GLOBALS = set(OUTPUT_PARAMS.values()) | {"output_root"}


def workflow_param_panel(name: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Everything the Run tab needs to draw a control per param.

    Returns (globals, steps), where each step entry is
    {"id", "step", "params": [Param], "overrides": {name: value}} — the
    Param objects carry type/default/help/advanced (pipeline/step.py), and
    `overrides` is what this workflow set on top, already template-expanded
    so a field shows the value the run would really use.

    A step's overrides are keyed by its `id:`, which is the whole point: the
    two `brush` trainings in fast_helical_full get their own section and
    their own controls.
    """
    from .cli import resolve_workflow
    from .registry import get_step_class
    from .templating import resolve

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    globals_shown = {k: v for k, v in spec.globals.items() if k not in HIDDEN_GLOBALS}

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
        })
    return globals_shown, steps


# A string default long enough (or with a line break in it) that a one-line
# box is unusable — the denoise prompts are the case this exists for.
_MULTILINE_AT = 80


def _widget_for(param: Param, value: Any, label: str):
    """One Gradio control for one declared param, prefilled with `value`.

    Everything the widget needs — type, bounds, choices, help — comes off
    the Param, which is why declaring params was worth doing: the UI has no
    per-step knowledge in it at all.
    """
    info = param.help or None
    if param.type is bool:
        return gr.Checkbox(value=bool(value), label=label, info=info)
    if param.type in (int, float):
        if param.minimum is not None and param.maximum is not None:
            return gr.Slider(
                minimum=param.minimum, maximum=param.maximum,
                step=1 if param.type is int else None,
                value=value, label=label, info=info,
            )
        return gr.Number(
            value=value, label=label, info=info,
            precision=0 if param.type is int else None,
        )
    if param.type is list:
        return gr.Textbox(
            value=yaml.safe_dump(value, default_flow_style=True).strip(),
            label=label, info=(info + " (YAML list)") if info else "YAML list",
        )
    if param.choices:
        return gr.Dropdown(
            choices=list(param.choices), value=value, label=label, info=info,
            allow_custom_value=True,
        )
    text = value if isinstance(value, str) else ""
    multiline = "\n" in text or len(text) > _MULTILINE_AT
    return gr.Textbox(
        value=text, label=label, info=info, lines=6 if multiline else 1,
    )


def _widget_value(param: Param, raw: Any) -> Any:
    """Bring a widget's value back to the param's declared type.

    Only the list case needs real work — a list arrives as YAML text. The
    rest is left to `Step.resolve_params`, which coerces at run time
    anyway; doing it twice would just mean two places to disagree.
    """
    if param.type is list:
        try:
            parsed = yaml.safe_load(raw) if isinstance(raw, str) else raw
        except yaml.YAMLError as exc:
            raise gr.Error(f"{param.name}: not valid YAML ({exc})")
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


def extract_dataset_zip(zip_path: str) -> Path:
    """Unpack an uploaded archive into the volume and return its dataset root."""
    target = upload_dir() / f"dataset-{time.strftime('%Y%m%d-%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        # Refuse absolute paths and `..` traversal rather than trusting the
        # archive: this unpacks onto the volume, next to real data.
        for member in archive.namelist():
            resolved = (target / member).resolve()
            if not str(resolved).startswith(str(target.resolve())):
                raise gr.Error(f"Refusing to extract {member!r}: it escapes the target directory.")
        archive.extractall(target)
    return find_dataset_root(target)


def save_upload(path: str, prefix: str) -> Path:
    """Copy a Gradio upload onto the volume; its own temp dir is not durable."""
    destination = upload_dir() / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}{Path(path).suffix}"
    shutil.copy(path, destination)
    return destination


def result_dirs(run_dir: Optional[Path]) -> Dict[str, Path]:
    """The deliverable subdirectories this run actually produced.

    Keyed by the name they take inside the archive, which is the same name
    they have on disk — `colmap/` and `ply/`, written there by the
    workflow's own export steps via `output_root`.
    """
    if not run_dir:
        return {}
    root = Path(run_dir)
    found = {}
    for name in OUTPUT_SUBDIRS.values():
        candidate = root / name
        if candidate.is_dir() and any(candidate.rglob("*")):
            found[name] = candidate
    return found


def build_result_zip(run_dir: Optional[Path]) -> Optional[str]:
    """One archive holding only the deliverables: colmap/ and/or ply/.

    Not `shutil.make_archive` over the whole run directory, which is what
    this used to be. A run directory also holds the final Dataset's 81
    full-resolution frames, its pointcloud, and — when the workflow trained
    an intermediate splat — a `brush/training_<ms>/` that is scaffolding,
    not output. Zipping all of it produced a multi-gigabyte download whose
    top level was a b2c dataset rather than anything COLMAP-shaped, and
    left the person on the other end to work out which parts mattered.
    """
    directories = result_dirs(run_dir)
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
    workflows = workflow_choices()
    default_workflow = "fast_helical_full" if "fast_helical_full" in workflows else workflows[0]

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
                    workflow_in = gr.Dropdown(
                        workflows, value=default_workflow, label="Workflow",
                    )
                    source_in = gr.Radio(
                        [SOURCE_DIRECTORY, SOURCE_ZIP, SOURCE_PHOTO],
                        value=SOURCE_DIRECTORY,
                        label="Input",
                    )
                    dataset_in = gr.Dropdown(
                        discover_datasets(), label="Dataset directory",
                        info=f"b2c datasets found under {data_dir()}",
                        allow_custom_value=True,
                    )
                    rescan_btn = gr.Button("Rescan", size="sm")
                    zip_in = gr.File(
                        label="Dataset .zip", file_types=[".zip"], visible=False,
                    )
                    photo_in = gr.Image(
                        # Not a photo of the subject: the two-panel image a
                        # diffusion model generates, front view left, back
                        # view right. fast_helical_native splits it.
                        label="Front/back reference sheet", type="filepath",
                        visible=False,
                    )
                    prompt_in = gr.Textbox(
                        label="Subject description",
                        placeholder="a woman in a red jacket",
                        info="Fills $SUBJECT_DESC$ in the denoise prompt. "
                             "Optional for an existing dataset that already has one.",
                    )
                    default_outputs = workflow_outputs(default_workflow)
                    outputs_in = gr.CheckboxGroup(
                        choices=default_outputs,
                        value=default_outputs,
                        label="Outputs",
                        info="What the run produces, and what the Results tab's "
                             ".zip contains. Unchecking the .ply skips a whole "
                             "30,000-iteration brush training.",
                        visible=bool(default_outputs),
                    )
                    with gr.Row():
                        start_btn = gr.Button("Start run", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                with gr.Column(scale=1):
                    summary_out = gr.Markdown(workflow_summary(default_workflow))

                    # Only what the user actually changed, in the two
                    # namespaces a run resolves: {"globals": {...},
                    # "steps": {step_id: {...}}}. Everything untouched stays
                    # owned by the workflow file and the step defaults,
                    # which is what keeps output_root's repoint working (see
                    # HIDDEN_GLOBALS) and what makes "reset" mean something.
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

                    @gr.render(inputs=[workflow_in])
                    def render_params(name):
                        globals_shown, step_entries = workflow_param_panel(name)

                        with gr.Accordion("Globals", open=True):
                            if not globals_shown:
                                gr.Markdown("_This workflow declares no globals._")
                            for key, value in globals_shown.items():
                                # A workflow's globals are free-form values,
                                # not declared Params, so the widget is
                                # picked from the value's own Python type.
                                pseudo = Param(name=key, type=type(value), default=value)
                                widget = _widget_for(pseudo, value, key)
                                widget.change(
                                    _record("globals", key, pseudo, value),
                                    inputs=[widget, param_state], outputs=[param_state],
                                )

                        for entry in step_entries:
                            label = f"{entry['id']}  ({entry['step']})"
                            with gr.Accordion(label, open=False):
                                plain = [p for p in entry["params"] if not p.advanced]
                                advanced = [p for p in entry["params"] if p.advanced]

                                def draw(param):
                                    overridden = param.name in entry["overrides"]
                                    value = (
                                        entry["overrides"][param.name] if overridden
                                        else param.default
                                    )
                                    mark = " •" if overridden else ""
                                    widget = _widget_for(param, value, param.name + mark)
                                    widget.change(
                                        _record("steps", param.name, param, value, entry["id"]),
                                        inputs=[widget, param_state], outputs=[param_state],
                                    )

                                for param in plain:
                                    draw(param)
                                if advanced:
                                    with gr.Accordion("Advanced", open=False):
                                        for param in advanced:
                                            draw(param)

                        gr.Markdown(
                            "_A dot marks a param the workflow sets; the rest show the "
                            "step's own default. Advanced holds the knobs that exist "
                            "because the underlying library has them._"
                        )

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
        def on_workflow_change(name: str):
            # The param panel redraws itself off `workflow_in` (see
            # gr.render above); this clears the state that went with the
            # old workflow so a stale override cannot survive the switch.
            choices = workflow_outputs(name)
            return (
                workflow_summary(name),
                {"globals": {}, "steps": {}},
                gr.update(choices=choices, value=choices, visible=bool(choices)),
            )

        workflow_in.change(
            on_workflow_change, inputs=workflow_in,
            outputs=[summary_out, param_state, outputs_in],
        )

        def on_source_change(source: str):
            return (
                gr.update(visible=source == SOURCE_DIRECTORY),
                gr.update(visible=source == SOURCE_DIRECTORY),
                gr.update(visible=source == SOURCE_ZIP),
                gr.update(visible=source == SOURCE_PHOTO),
            )

        source_in.change(
            on_source_change, inputs=source_in,
            outputs=[dataset_in, rescan_btn, zip_in, photo_in],
        )
        rescan_btn.click(lambda: gr.update(choices=discover_datasets()), outputs=dataset_in)

        def on_start(workflow, source, dataset_path, zip_file, photo, prompt,
                     params, selected_outputs):
            params = params or {}
            global_overrides = dict(params.get("globals", {}))
            step_overrides = {k: dict(v) for k, v in params.get("steps", {}).items()}

            supported = workflow_outputs(workflow)
            if supported:
                chosen = set(selected_outputs or [])
                if not chosen:
                    raise gr.Error(
                        "Pick at least one output — a run that exports neither a "
                        "COLMAP dataset nor a .ply leaves nothing to download."
                    )
                # Set both, not just the checked ones: a workflow defaults
                # them to true, so an unchecked box has to actively say false.
                for label in supported:
                    global_overrides[OUTPUT_PARAMS[label]] = label in chosen

            dataset_dir = reference_image = None
            if source == SOURCE_DIRECTORY:
                if not dataset_path:
                    raise gr.Error("Pick a dataset directory (or Rescan if the list is empty).")
                dataset_dir = find_dataset_root(dataset_path)
            elif source == SOURCE_ZIP:
                if not zip_file:
                    raise gr.Error("Upload a .zip of a dataset directory.")
                path = zip_file if isinstance(zip_file, str) else zip_file.name
                dataset_dir = extract_dataset_zip(path)
            else:
                if not photo:
                    raise gr.Error(
                        "Upload the front/back reference sheet (front view on "
                        "the left, back view on the right)."
                    )
                if workflow_needs_a_dataset(workflow):
                    raise gr.Error(
                        f"'{workflow}' starts from an existing dataset — it reads "
                        "frames and cameras it does not render. Use "
                        "fast_helical_native, which builds its own views from a "
                        "reference sheet, or pick a dataset directory / upload a .zip."
                    )
                reference_image = save_upload(photo, "reference")

            from .cli import resolve_workflow

            workflow_path = resolve_workflow(workflow)
            spec = WorkflowSpec.from_yaml(workflow_path)
            # Validated here, before the job is ever queued, so a bad
            # override fails at submit time — in a button handler the user
            # is looking at — instead of forty minutes into a queued run
            # a `pipeline.run_worker` process reports as merely "failed".
            apply_ui_overrides(spec, global_overrides, step_overrides)
            spec.validate()

            run_name = timestamped_run_name(spec.name)
            out = output_dir() / run_name
            job = RunJob(
                run_name=run_name,
                workflow_name=spec.name,
                workflow_path=str(workflow_path),
                output_dir=str(out),
                envs_path=envs_path,
                global_overrides=global_overrides,
                step_overrides=step_overrides,
                dataset_dir=str(dataset_dir) if dataset_dir else None,
                reference_image=str(reference_image) if reference_image else None,
                prompt=prompt or "",
            )
            scheduler.submit(job)
            return gr.update(choices=_run_choices(scheduler), value=run_name)

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
            inputs=[workflow_in, source_in, dataset_in, zip_in, photo_in, prompt_in,
                    param_state, outputs_in],
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

            directories = result_dirs(Path(directory))
            if not directories:
                # Distinguish "still running" from "ran and produced nothing":
                # the second means the export steps were switched off or
                # failed, and looking in the run directory is the next move.
                note = (
                    "still running — the exports are the last steps"
                    if state.status in ("queued", "running")
                    else "the run produced neither; check the Progress tab's step list"
                )
                return (
                    f"### `{directory}`\n\nNo `colmap/` or `ply/` here yet — {note}.",
                    gallery_images(directory),
                    None,
                )

            lines = []
            for name, path in sorted(directories.items()):
                files = [f for f in path.rglob("*") if f.is_file()]
                size = sum(f.stat().st_size for f in files)
                lines.append(f"- **`{name}/`** — {len(files)} files, {size / 1e9:.2f} GB")

            archive = build_result_zip(Path(directory))
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
