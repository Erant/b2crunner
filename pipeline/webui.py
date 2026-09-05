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
`dir:` of every output that actually got written, plus the `log.txt` the run
wrote, and nothing else.

**All results** is that same archive for every run at once, and it reads the
*volume* rather than this process: `GpuScheduler` keeps its runs in memory,
so the Active-run picker empties on every UI restart while the runs sit in
the output directory untouched. `discover_runs` picks them back up out of
the status files the workers published — plus any output directory nobody
published a status for at all, which is what a `pipeline.cli` run leaves.

**This module is the browser's view and nothing else.** What a submission
means, how it is queued, and how a finished run is packaged all live in
pipeline/runs.py, which pipeline/api.py drives over HTTP against the same
scheduler — so a curl and a button press queue identical work. `build_server`
is where the two are put on one port: the API first, the UI mounted under it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import yaml

from . import steps  # noqa: F401  registers every Step; the UI is an entrypoint
from .gpu_scheduler import GpuScheduler, detect_gpu_count
from .models import registry
from .paths import data_dir, output_dir, run_jobs_dir, upload_dir
from .run_state import PREVIEW_FRAMES, RunState
from .runs import (
    BUNDLE_NAME, IMAGE_SUFFIXES, WORKFLOW_NATIVE, SubmitError,
    build_bundle_zip, build_result_zip, completed_runs, resolve_upload,
    result_dirs, result_subdirs, run_contents, run_log_path, run_recency,
    submit_runs, wants_debug, workflow_param_panel,
)
from .step import Param
from .workflow import WorkflowSpec, load_envs, truthy

logger = logging.getLogger(__name__)

# The gallery is a sanity check ("did it render a person or grey mush"),
# not a contact sheet — 81 full-resolution PNGs into a browser over a pod
# proxy is slow enough to look broken.
_GALLERY_MAX = 24

PREVIEW_ALL = "All steps"

# Nothing about a particular workflow's settings or deliverables lives in
# this module any more: the Outputs box, the Settings box and the download's
# contents are all read off the workflow's own `settings:` / `outputs:`
# blocks (see pipeline/workflow.py). What used to be here — OUTPUT_PARAMS,
# RESULT_SUBDIRS, HIDDEN_GLOBALS, RESOLUTION_CHOICES, GLOBAL_CHOICES — was
# five tables naming the globals of one file, each with its own comment
# asking the next person to keep it in step with that file.
#
# Nor does the lifecycle: what an upload means, how a run is queued, and how
# a finished one is packaged all live in pipeline/runs.py, which the HTTP
# API drives too. This module is the browser's view of that and nothing else.


# --------------------------------------------------------------------------
# drawing one param
# --------------------------------------------------------------------------


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
    # The empty string is a real choice on several params — no backdrop, no
    # camera path — and it draws as a blank, unclickable-looking row unless
    # it is given something to read. `_choice_map` maps the label back, so
    # the value the step sees is still "".
    if value == "":
        return "(none)"
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
        # None is "the step decides" (see Param.default) and has to draw as an
        # EMPTY box, not as the word `null`: the box is round-tripped, and a
        # literal "null" is neither a list nor something a person would type.
        text = (
            "" if value is None
            else yaml.safe_dump(value, default_flow_style=True).strip()
        )
        return gr.Textbox(
            value=text,
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
        if parsed is None and param.default is None:
            # A list param that declares None as its default has an off
            # position — `background_base_color` empty means "leave the
            # texture generator its own wall colour" — so clearing the box is
            # a real answer rather than a malformed list. Only for those: on a
            # param with a real list default, an empty box is still a mistake,
            # and returning None there would hand the step a None where it
            # expects three numbers.
            return None
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
                        file_types=[".zip", *sorted(IMAGE_SUFFIXES)],
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

        with gr.Tab("All results"):
            gr.Markdown(
                "**Every run on this volume that produced deliverables**, "
                "packaged in one press — including runs from before this UI "
                "process started, which the **Active run** picker above has "
                "never heard of (it only holds what this process submitted). "
                "One .zip per run, the same contents the Results tab hands "
                "back for a single run.\n\n"
                "_An archive that is already up to date is reused rather than "
                "rebuilt, so a rescan after one new run costs one run's worth "
                "of copying, not the whole volume's._"
            )
            with gr.Row():
                all_refresh = gr.Button(
                    "Scan and package everything", variant="primary", scale=2)
                bundle_in = gr.Checkbox(
                    value=False, label="Also build one combined .zip", scale=1,
                    info="Every run under its own directory in a single "
                         "archive — one download instead of N. It is a second "
                         "full copy of the deliverables on the volume, so it "
                         "is off unless you ask.",
                )
            all_info = gr.Markdown()
            all_table = gr.Dataframe(
                headers=["", "run", "finished", "contents", "size"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False, label="Completed runs",
            )
            all_files = gr.File(
                label="Result archives — one per run", file_count="multiple")
            bundle_out = gr.File(label="Everything in one .zip", visible=False)
            all_gallery = gr.Gallery(
                label="One frame per run", columns=6, height=300,
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
            if not upload_file:
                raise gr.Error(
                    "Upload something: a .zip of image/prompt pairs, or a "
                    "single reference-sheet image."
                )
            path = upload_file if isinstance(upload_file, str) else upload_file.name
            # Everything from here down is `pipeline.runs`, which the HTTP
            # API calls the same way. Its refusals are `SubmitError`s whose
            # message is already a sentence to show someone; this is the one
            # place they become a toast.
            try:
                plan = resolve_upload(path, prompt or "")
                names = submit_runs(
                    scheduler, plan,
                    params.get("globals"), params.get("steps"),
                    envs_path=envs_path,
                )
            except SubmitError as exc:
                raise gr.Error(str(exc)) from None
            # Point the shared picker at the first run; the rest are one
            # dropdown-hop away and already fanning out across the other GPUs.
            return gr.update(choices=_run_choices(scheduler), value=names[0])

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

            # `reuse=True` like every other caller: `archive_is_current`
            # compares mtimes, so a run still writing is correctly seen as
            # stale and rebuilt — while pressing Refresh again on a
            # finished 2 GB run stops re-copying 2 GB to answer the same
            # question.
            archive = build_result_zip(
                Path(directory), state.workflow, state.log_path, reuse=True,
                debug=wants_debug(state),
            )
            if run_log_path(Path(directory), state.log_path):
                lines.append("- **`log.txt`** — the log this run wrote")
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

        def on_all_results(bundle: bool):
            """Package every finished run on the volume, streaming as it goes.

            A generator for the same reason the other two tabs are: the
            first press on a volume holding a dozen runs copies tens of
            gigabytes, and a button that returns nothing for four minutes
            is indistinguishable from a hung one. Each run appears in the
            table as its archive lands.
            """
            icons = {"done": "✅", "failed": "❌", "cancelled": "⛔",
                     "unknown": "•"}
            runs = completed_runs()
            if not runs:
                yield ("No finished run on this volume has produced "
                       "deliverables yet.", [], None,
                       gr.update(visible=False, value=None), [])
                return

            rows: List[List[Any]] = []
            archives: List[str] = []
            thumbs: List[Any] = []
            for index, state in enumerate(runs, 1):
                yield (
                    f"Packaging **{index} of {len(runs)}** — `{state.name}`…",
                    rows, archives, gr.update(), thumbs,
                )
                contents, size = run_contents(state)
                archive = build_result_zip(
                    Path(state.output_dir), state.workflow, state.log_path,
                    reuse=True, debug=wants_debug(state),
                )
                if archive:
                    archives.append(archive)
                when = run_recency(state)
                rows.append([
                    icons.get(state.status, "?"), state.name,
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(when)) if when else "",
                    contents, f"{size / 1e9:.2f} GB",
                ])
                # One frame is enough to tell which subject a run was; the
                # whole point of this tab is not opening each one to find out.
                frames = gallery_images(state.output_dir)
                if frames:
                    thumbs.append((frames[0], state.name))

            packaged = sum(Path(a).stat().st_size for a in archives)
            info = [
                f"**{len(archives)} of {len(runs)}** runs packaged · "
                f"{packaged / 1e9:.2f} GB of archives in `{output_dir()}`."
            ]
            combined = gr.update(visible=False, value=None)
            if bundle:
                yield ("Building the combined .zip…", rows, archives,
                       gr.update(), thumbs)
                path = build_bundle_zip(runs)
                if path:
                    combined = gr.update(visible=True, value=path)
                    info.append(
                        f"`{BUNDLE_NAME}` holds all {len(archives)} under "
                        "`<run name>/`.")
            yield "\n\n".join(info), rows, archives, combined, thumbs

        all_refresh.click(
            on_all_results, inputs=[bundle_in],
            outputs=[all_info, all_table, all_files, bundle_out, all_gallery],
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


def build_server(
    envs_path: str = "", gpu_count: Optional[int] = None, serve_api: bool = True,
) -> "fastapi.FastAPI":
    """The whole server: the HTTP API, then the Gradio UI mounted under it.

    One process on one port, sharing one `GpuScheduler` — which is the
    point. A second process with a scheduler of its own would start a run on
    a card this one already had busy, and neither would know.

    **Order matters.** `gr.mount_gradio_app` ends in `app.mount("/", ...)`,
    a Starlette mount that matches every path there is; Starlette walks its
    routes in registration order, so the API has to be attached first or it
    is unreachable — with no error, just the UI's 404 page where JSON should
    be. `tests/test_api.py` pins that.

    Auth is one shared token, `B2C_API_TOKEN`. The API requires it as a
    bearer header; the UI takes it as the password on Gradio's own login
    form (username `b2c`), which is a browser-shaped way to present the same
    secret. With no token set, neither is guarded and the API is not served
    at all — a dev box needs no ceremony, and a pod that forgets the
    variable serves nothing new rather than serving everything openly.
    """
    import contextlib

    import fastapi

    from .api import (
        API_PREFIX, SHUTDOWN_COMMAND_ENV, TOKEN_ENV, ShutdownController,
        add_schema_route, api_token, build_router, shutdown_command,
    )
    from .cli import DEFAULT_ENVS

    envs_path = envs_path or DEFAULT_ENVS
    blocks = build_app(envs_path, gpu_count=gpu_count)
    scheduler: GpuScheduler = blocks._gpu_scheduler  # noqa: SLF001
    token = api_token()

    # queue() is what makes the streaming generators above work at all. The
    # concurrency limit is UI-event concurrency (how many browser
    # connections Gradio itself services at once), unrelated to how many
    # GPU workers the scheduler runs — that is bounded by `gpu_count`.
    blocks.queue(default_concurrency_limit=4)

    # Nothing forwards SIGTERM/SIGINT to a child this process spawned with
    # `subprocess.Popen` on its own — Python does not do that for you. On a
    # pod stop, without this, every in-flight `pipeline.run_worker` would be
    # orphaned holding a CUDA context until the container's whole cgroup is
    # torn down. It hangs off the server's lifespan rather than off
    # `signal.signal` because uvicorn installs handlers of its own the
    # moment it starts, and the last one to register wins.
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        logger.info("shutting down; stopping %d GPU worker slot(s)", scheduler.gpu_count)
        scheduler.shutdown()

    server = fastapi.FastAPI(
        title="b2c_runner",
        description="Submit a Body2COLMAP run, watch it, collect the output.",
        # Both off: FastAPI would serve them from the app, outside the
        # router's token check, advertising every route of a guarded API on
        # a public pod URL. `add_schema_route` puts the schema back behind
        # the guard.
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    shutdown = ShutdownController(scheduler)
    if not serve_api:
        logger.info("HTTP API disabled by --no-api; serving the web UI alone")
    elif token:
        server.include_router(
            build_router(scheduler, envs_path, shutdown), prefix=API_PREFIX,
        )
        add_schema_route(server, API_PREFIX)
        logger.info("HTTP API on %s (bearer %s), schema at %s/openapi.json",
                    API_PREFIX, TOKEN_ENV, API_PREFIX)
        command = shutdown_command()
        logger.info(
            "%s: %s", SHUTDOWN_COMMAND_ENV,
            command or "(unset — POST /shutdown stops this container and nothing else)",
        )
    else:
        logger.warning(
            "%s is not set: the HTTP API is not being served and the web UI is "
            "unauthenticated. Set it to enable both.", TOKEN_ENV,
        )

    # Keeps the app usable as a handle onto the scheduler it drives, the
    # way `build_app` does for the Blocks — `launch` needs it to stop the
    # workers before uvicorn starts waiting on connections.
    server.state.gpu_scheduler = scheduler
    server.state.shutdown = shutdown

    return gr.mount_gradio_app(
        server, blocks, "/",
        auth=("b2c", token) if token else None,
        auth_message="Sign in with the pod's B2C_API_TOKEN as the password.",
        show_error=True,
        # Uploads and result downloads live on the volume, which is outside
        # Gradio's default allowlist (its own temp dir plus the cwd).
        allowed_paths=[str(data_dir()), str(output_dir()), str(upload_dir())],
    )


def launch(
    host: str = "0.0.0.0", port: int = 7860, envs_path: str = "", share: bool = False,
    gpu_count: Optional[int] = None, serve_api: bool = True,
) -> None:
    """Serve until stopped, and take the GPU workers down first when it is.

    The signal handlers are ours rather than uvicorn's, and they run
    `scheduler.shutdown()` *before* asking uvicorn to wind down. That order
    is the whole point. A shutdown hook on the ASGI lifespan is not enough:
    uvicorn runs lifespan shutdown only after its graceful phase, which
    loops while any connection is still open — and a browser left on the
    Progress or Results tab is holding a Gradio SSE stream for the length
    of the run. So a pod stop with one tab attached would sit in graceful
    shutdown until SIGKILL, and every in-flight `pipeline.run_worker` would
    be orphaned holding a CUDA context: exactly the failure the handler
    exists to prevent, reintroduced by moving it. The lifespan hook stays
    as well, for the shutdowns no signal announces.

    `timeout_graceful_shutdown` bounds the wait in any case — those streams
    poll forever by design and will not close on their own.
    """
    import signal

    import uvicorn

    server = build_server(envs_path, gpu_count=gpu_count, serve_api=serve_api)
    scheduler: GpuScheduler = server.state.gpu_scheduler

    config = uvicorn.Config(
        server, host=host, port=port, timeout_graceful_shutdown=10,
    )
    running = uvicorn.Server(config)
    # The one thing `POST /api/v1/shutdown` cannot work out for itself.
    # Same `should_exit` the signal handler below sets, so a requested stop
    # and a SIGTERM wind down through exactly one path.
    server.state.shutdown.bind(lambda: setattr(running, "should_exit", True))
    # Uvicorn installs its own through `loop.add_signal_handler`, which
    # would replace anything set here; telling it not to is the supported
    # way to embed it.
    running.install_signal_handlers = lambda: None

    def _stop(signum, _frame) -> None:
        logger.info("received signal %s; stopping %d GPU worker slot(s)",
                    signum, scheduler.gpu_count)
        scheduler.shutdown()
        running.should_exit = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logger.info("serving on http://%s:%d", host, port)
    if share:
        _open_tunnel(host, port)
    running.run()


def _open_tunnel(host: str, port: int) -> None:
    """`--share`'s gradio.live tunnel, which `mount_gradio_app` has no flag for.

    A development convenience — on a pod the access path is the provider's
    own proxy — so a tunnel that cannot be set up is logged and stepped
    over. Reaching past `Blocks.launch()` for it means reaching into
    Gradio's internals; failing to serve at all because that moved would be
    a much worse trade than losing the tunnel.
    """
    import secrets

    try:
        from gradio import networking

        url = networking.setup_tunnel(host, port, secrets.token_urlsafe(32), None, None)
    except Exception:
        logger.warning("could not open a share tunnel; serving locally only", exc_info=True)
        return
    logger.info("share tunnel: %s", url)
