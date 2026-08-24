"""Gradio front end: submit work, watch it run, pull the results back.

Why this exists at all: the pipeline's real home is a rented GPU pod, where
the only things you have are a log pane and, if you set it up in advance,
an HTTP port. A run takes tens of minutes to hours, so "start it and watch
stdout over SSH" means a dropped connection loses the view of a run that is
still going. And the image is meant to be built once and not rebuilt, so
anything you might want to do on the pod has to already be in it.

Three shapes of input, per what the pipeline can actually start from:

  * **an existing dataset directory** on the mounted volume — the path every
    verification run so far has used (cyber_6f/initial and friends);
  * **an uploaded .zip** of such a directory — the same thing when the
    dataset lives on your laptop rather than the pod;
  * **a single reference photo** — the from-scratch path, which needs
    `Dataset.from_reference_image` and runs `fast_helical_native.yaml`.

The photo path is the least proven of the three (its front half — render /
generate_firstlast / inject_anchor — has never executed), which is exactly
why the other two are here: if it falls over, the rest of the pipeline is
still exercisable from a zip without touching the image.

**The run is a background thread, the UI only ever polls it.** A Gradio
generator holds an SSE connection for as long as it yields, and a browser
tab surviving a three-hour run over a pod proxy is not something to design
around. So the thread owns the run, the UI reads a snapshot of shared
state, and closing the tab does nothing to the work. Reopening it and
hitting Attach picks the view back up.

One run at a time, deliberately: there is one GPU, and two concurrent
workflows just means both OOM.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import yaml

from . import steps  # noqa: F401  registers every Step; the UI is an entrypoint
from .dataset import Dataset, find_dataset_root
from .logging_setup import DATE_FORMAT, FORMAT, QueueLogHandler, timestamped_run_name
from .paths import data_dir, log_dir, output_dir, upload_dir
from .runner import RunCancelled, RunEvent, WorkflowRunner
from .workflow import WorkflowSpec, load_envs

logger = logging.getLogger(__name__)

# Enough scrollback to cover a whole step's chatter without letting a
# runaway progress bar grow the process's memory without bound.
_LOG_BUFFER_LINES = 4000

# The gallery is a sanity check ("did it render a person or grey mush"),
# not a contact sheet — 81 full-resolution PNGs into a browser over a pod
# proxy is slow enough to look broken.
_GALLERY_MAX = 24

SOURCE_DIRECTORY = "Dataset directory on this machine"
SOURCE_ZIP = "Upload a dataset .zip"
SOURCE_PHOTO = "Single reference photo"


@dataclass
class StepRecord:
    index: int
    step_id: str
    step_name: str
    status: str = "running"
    elapsed: float = 0.0


@dataclass
class RunState:
    name: str = ""
    workflow: str = ""
    status: str = "idle"  # idle | running | done | failed | cancelled
    message: str = ""
    started: float = 0.0
    finished: float = 0.0
    current: int = 0
    total: int = 0
    steps: List[StepRecord] = field(default_factory=list)
    output_dir: Optional[Path] = None
    log_path: Optional[Path] = None
    error: str = ""


class RunManager:
    """Owns the one background run and the state the UI reads off it."""

    def __init__(self, envs_path: str) -> None:
        self.envs_path = envs_path
        self._lock = threading.Lock()
        self._log: deque = deque(maxlen=_LOG_BUFFER_LINES)
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self.state = RunState()

    # -- state ------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> tuple[RunState, str]:
        with self._lock:
            return self.state, "\n".join(self._log)

    def _put_log(self, line: str) -> None:
        with self._lock:
            self._log.append(line)

    def cancel(self) -> None:
        self._cancel.set()
        self._put_log(">>> cancellation requested; the run stops after the current step")

    # -- running ----------------------------------------------------------
    def start(
        self,
        workflow_path: Path,
        params: Dict[str, Any],
        dataset_dir: Optional[Path],
        reference_image: Optional[Path],
        prompt: str,
    ) -> str:
        if self.is_running:
            raise gr.Error("A run is already in progress. Cancel it first, or wait for it.")

        spec = WorkflowSpec.from_yaml(workflow_path)
        spec.params.update(params)
        run_name = timestamped_run_name(spec.name)
        out = output_dir() / run_name
        if "output_root" in spec.params and "output_root" not in params:
            spec.params["output_root"] = str(out)

        with self._lock:
            self._log.clear()
            self.state = RunState(
                name=run_name,
                workflow=spec.name,
                status="running",
                started=time.time(),
                total=len(spec.steps),
                steps=[
                    StepRecord(i, s.id, s.step, status="pending")
                    for i, s in enumerate(spec.steps, start=1)
                ],
                output_dir=out,
            )
        self._cancel.clear()

        self._thread = threading.Thread(
            target=self._run,
            args=(spec, run_name, out, dataset_dir, reference_image, prompt),
            name=f"b2c-run-{run_name}",
            daemon=True,
        )
        self._thread.start()
        return run_name

    def _on_event(self, event: RunEvent) -> None:
        with self._lock:
            state = self.state
            if event.kind == "step_start":
                state.current = event.index
                state.message = f"[{event.index}/{event.total}] {event.step_id}"
                state.steps[event.index - 1].status = "running"
            elif event.kind == "step_end":
                record = state.steps[event.index - 1]
                record.status = "done"
                record.elapsed = event.elapsed
            elif event.kind == "step_error":
                record = state.steps[event.index - 1]
                record.status = "failed"
                record.elapsed = event.elapsed
        # Checked on entry to each step rather than inside one: a step is a
        # single opaque call (often a subprocess holding the GPU), and
        # tearing one down mid-flight risks leaving the card in a state the
        # next run inherits.
        if event.kind == "step_start" and self._cancel.is_set():
            raise RunCancelled(f"cancelled before step {event.index} ({event.step_id})")

    def _run(
        self,
        spec: WorkflowSpec,
        run_name: str,
        out: Path,
        dataset_dir: Optional[Path],
        reference_image: Optional[Path],
        prompt: str,
    ) -> None:
        # Mirror this run's log into the UI buffer. Attached to the root
        # logger so it also picks up the relayed output of subprocess steps
        # (which arrive as `step.<name>` records), not just this module's.
        root = logging.getLogger()
        # A handler below the logger's own level receives nothing, and the
        # root default is WARNING. `setup_logging` normally lowers it, but
        # the UI can also be driven from an embedding process that never
        # called it — in which case the log pane stays silently empty,
        # which reads as "the run is stuck".
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)

        handler = QueueLogHandler(self._put_log)
        handler.setLevel(logging.INFO)
        root.addHandler(handler)

        log_path = log_dir() / f"{run_name}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(file_handler)
        with self._lock:
            self.state.log_path = log_path

        try:
            from .doctor import log_machine_banner

            logger.info("run '%s' (%s)", run_name, spec.name)
            logger.info("output: %s", out)
            log_machine_banner()

            if reference_image is not None:
                dataset = Dataset.from_reference_image(reference_image, prompt=prompt or None)
                logger.info("starting from reference photo %s (%dx%d)",
                            reference_image, *dataset.resolution)
            else:
                dataset = Dataset.from_disk(dataset_dir)
                logger.info("loaded %s: %d frames, %d cameras, %d points",
                            dataset_dir, len(dataset.images), len(dataset.cameras),
                            len(dataset.points_3d[0]))
                if prompt:
                    dataset.prompt = prompt

            envs = load_envs(self.envs_path)
            runner = WorkflowRunner(spec, envs=envs, on_event=self._on_event)
            ctx = runner.run({"dataset": dataset})

            final: Dataset = ctx.get("dataset")
            saved = final.to_disk(out)
            logger.info("saved final dataset to %s (%d frames)", saved, len(final.images))

            with self._lock:
                self.state.status = "done"
                self.state.message = f"complete — {len(final.images)} frames in {saved}"
        except RunCancelled as exc:
            logger.warning("run cancelled: %s", exc)
            with self._lock:
                self.state.status = "cancelled"
                self.state.message = str(exc)
        except Exception as exc:
            logger.error("run failed: %s", exc)
            logger.debug("%s", traceback.format_exc())
            self._put_log(traceback.format_exc())
            with self._lock:
                self.state.status = "failed"
                self.state.message = f"{type(exc).__name__}: {exc}"
                self.state.error = traceback.format_exc()
        finally:
            with self._lock:
                self.state.finished = time.time()
            root.removeHandler(handler)
            root.removeHandler(file_handler)
            file_handler.close()


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


def workflow_params_yaml(name: str) -> str:
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    if not spec.params:
        return "# this workflow declares no params\n"
    return yaml.safe_dump(spec.params, sort_keys=False, allow_unicode=True, width=100)


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


def zip_results(directory: Path) -> Optional[str]:
    if not directory or not Path(directory).exists():
        return None
    archive = Path(shutil.make_archive(
        str(output_dir() / f"{Path(directory).name}-results"), "zip", root_dir=str(directory)
    ))
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


def _format_status(state: RunState) -> str:
    if state.status == "idle":
        return "### idle\nNothing running. Configure a run on the **Run** tab."

    elapsed = (state.finished or time.time()) - state.started
    icon = {"running": "⏳", "done": "✅", "failed": "❌", "cancelled": "⛔"}.get(state.status, "")
    lines = [
        f"### {icon} {state.status} — `{state.name}`",
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


def _step_rows(state: RunState) -> List[List[Any]]:
    icons = {"pending": "·", "running": "▶", "done": "✓", "failed": "✗"}
    return [
        [record.index, icons.get(record.status, "?"), record.step_id, record.step_name,
         f"{record.elapsed:.1f}s" if record.elapsed else ""]
        for record in state.steps
    ]


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

def build_app(envs_path: str) -> gr.Blocks:
    manager = RunManager(envs_path)
    workflows = workflow_choices()
    default_workflow = "fast_helical_full" if "fast_helical_full" in workflows else workflows[0]

    with gr.Blocks(title="b2c_runner", analytics_enabled=False) as app:
        gr.Markdown("# b2c_runner\nBody2COLMAP pipeline — submit a run, watch it, collect the output.")

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
                        label="Reference photo", type="filepath", visible=False,
                    )
                    prompt_in = gr.Textbox(
                        label="Subject description",
                        placeholder="a woman in a red jacket",
                        info="Fills $SUBJECT_DESC$ in the denoise prompt. "
                             "Optional for an existing dataset that already has one.",
                    )
                    with gr.Row():
                        start_btn = gr.Button("Start run", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                with gr.Column(scale=1):
                    summary_out = gr.Markdown(workflow_summary(default_workflow))
                    params_in = gr.Code(
                        value=workflow_params_yaml(default_workflow),
                        language="yaml",
                        label="Params (edited copy of the workflow's own params block)",
                        lines=20,
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
                "_The run happens in a background thread. Closing this tab does not stop it —"
                " reopen the page and press **Attach / refresh** to pick the view back up._"
            )

        with gr.Tab("Results"):
            results_refresh = gr.Button("Load latest results", variant="primary")
            results_info = gr.Markdown()
            results_gallery = gr.Gallery(label="Frames", columns=6, height=400)
            results_zip = gr.File(label="Download everything (.zip)")

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
            return workflow_summary(name), workflow_params_yaml(name)

        workflow_in.change(
            on_workflow_change, inputs=workflow_in, outputs=[summary_out, params_in]
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

        def on_start(workflow, source, dataset_path, zip_file, photo, prompt, params_text):
            try:
                params = yaml.safe_load(params_text) or {}
            except yaml.YAMLError as exc:
                raise gr.Error(f"Params are not valid YAML: {exc}")
            if not isinstance(params, dict):
                raise gr.Error("Params must be a YAML mapping.")

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
                    raise gr.Error("Upload a reference photo.")
                reference_image = save_upload(photo, "reference")

            from .cli import resolve_workflow

            manager.start(
                workflow_path=resolve_workflow(workflow),
                params=params,
                dataset_dir=dataset_dir,
                reference_image=reference_image,
                prompt=prompt or "",
            )

        def stream():
            """Poll the manager until the run ends, then one final update.

            A generator rather than a Timer so it works the same on every
            Gradio 4.x/5.x, and it deliberately reads shared state instead
            of driving the run: if this connection dies, the thread it is
            watching does not.
            """
            while True:
                state, log_text = manager.snapshot()
                fraction = (state.current / state.total) if state.total else 0
                yield (
                    _format_status(state),
                    gr.update(value=fraction),
                    _step_rows(state),
                    log_text,
                )
                if not manager.is_running:
                    break
                time.sleep(1.0)
            state, log_text = manager.snapshot()
            fraction = (state.current / state.total) if state.total else 0
            yield (
                _format_status(state),
                gr.update(value=fraction),
                _step_rows(state),
                log_text,
            )

        progress_outputs = [status_out, progress_out, steps_out, log_out]

        start_btn.click(
            on_start,
            inputs=[workflow_in, source_in, dataset_in, zip_in, photo_in, prompt_in, params_in],
            outputs=[],
        ).success(stream, outputs=progress_outputs)

        attach_btn.click(stream, outputs=progress_outputs)
        cancel_btn.click(lambda: manager.cancel(), outputs=[])

        def on_results():
            state, _ = manager.snapshot()
            directory = state.output_dir
            if not directory or not Path(directory).exists():
                return "No output directory yet — start a run first.", [], None
            files = sorted(Path(directory).rglob("*"))
            size = sum(f.stat().st_size for f in files if f.is_file())
            listing = "\n".join(
                f"- `{f.relative_to(directory)}` ({f.stat().st_size / 1e6:.1f} MB)"
                for f in files if f.is_file() and f.stat().st_size > 1e6
            )
            info = (
                f"### `{directory}`\n\n{len(files)} files, {size / 1e9:.2f} GB total.\n\n"
                f"**Larger than 1 MB:**\n{listing or '_none_'}"
            )
            return info, gallery_images(directory), zip_results(Path(directory))

        results_refresh.click(
            on_results, outputs=[results_info, results_gallery, results_zip]
        )

        def on_doctor():
            from .doctor import format_report, run_checks

            return format_report(run_checks(load_envs(envs_path)), verbose=True)

        doctor_btn.click(on_doctor, outputs=doctor_out)

    return app


def launch(host: str = "0.0.0.0", port: int = 7860, envs_path: str = "", share: bool = False) -> None:
    from .cli import DEFAULT_ENVS

    app = build_app(envs_path or DEFAULT_ENVS)
    logger.info("web UI on http://%s:%d", host, port)
    # queue() is what makes the streaming generator above work at all, and
    # the concurrency limit matches the one-run-at-a-time rule.
    app.queue(default_concurrency_limit=4).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        # Uploads and result downloads live on the volume, which is outside
        # Gradio's default allowlist (its own temp dir plus the cwd).
        allowed_paths=[str(data_dir()), str(output_dir()), str(upload_dir())],
    )
