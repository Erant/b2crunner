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
  * **a single reference photo** — the from-scratch path, via
    `Dataset.from_reference_image`.

No shipped workflow currently takes the third: both `fast_helical` files
begin from a complete dataset, and the one that rendered its own views
(`fast_helical_native.yaml`) was dropped as irrelevant to the pod image.
The control stays, because the plumbing behind it is the expensive half and
adding such a workflow back is a YAML file — but picking it against a
workflow that cannot use it is refused at submit time (see
`workflow_needs_a_dataset`) rather than failing on a bare KeyError one step
in.

**The run is a background thread, the UI only ever polls it.** A Gradio
generator holds an SSE connection for as long as it yields, and a browser
tab surviving a three-hour run over a pod proxy is not something to design
around. So the thread owns the run, the UI reads a snapshot of shared
state, and closing the tab does nothing to the work. Reopening it and
hitting Attach picks the view back up.

One run at a time, deliberately: there is one GPU, and two concurrent
workflows just means both OOM.

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
from .models import is_ready, registry, required_for_steps, wait_until_ready
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

# The per-step contact sheet: eight frames evenly spaced through the batch,
# written after every step. Eight because an 81-frame orbit strides by ten —
# frames 1, 11, 21, ... — which is a quarter turn between neighbours, enough
# to see a step break one side of the subject and not the other.
#
# Downscaled JPEGs, not the frames themselves: a run has around fifteen
# steps, so this is ~120 images, and at full resolution that is a gigabyte
# of PNG nobody can load over a pod's HTTP proxy.
_PREVIEW_FRAMES = 8
_PREVIEW_WIDTH = 480
_PREVIEW_QUALITY = 82
_PREVIEW_DIRNAME = "_previews"

# What a masked-out pixel is drawn over. Mid-grey rather than black or
# white so it reads as "nothing here" against both a dark jacket and a
# blown-out background — the two cases where a mask step going wrong is
# otherwise invisible.
_PREVIEW_BACKDROP = 128

SOURCE_DIRECTORY = "Dataset directory on this machine"
SOURCE_ZIP = "Upload a dataset .zip"
SOURCE_PHOTO = "Single reference photo"

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


@dataclass
class StepRecord:
    index: int
    step_id: str
    step_name: str
    status: str = "running"
    elapsed: float = 0.0
    # Paths to this step's preview frames, filled in as it finishes.
    previews: List[str] = field(default_factory=list)


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


def preview_indices(count: int, wanted: int = _PREVIEW_FRAMES) -> List[int]:
    """Evenly spaced frame indices: 0, 10, 20, ... for an 81-frame batch.

    Strides rather than slicing the front, because the interesting failures
    are positional — a denoise pass that holds up at the front of the orbit
    and falls apart at the back looks perfect in the first eight frames.
    """
    if count <= 0:
        return []
    if count <= wanted:
        return list(range(count))
    stride = count // wanted
    return [i * stride for i in range(wanted)]


def write_previews(images, masks, names, destination: Path) -> List[str]:
    """Write this step's sampled frames as small JPEGs; return their paths.

    Masks are composited in rather than dropped: `rmbg` and `mask_splat`
    change nothing else about a frame, so without this their previews are
    indistinguishable from the step before them — which is exactly when you
    are looking at this gallery.
    """
    import cv2
    import numpy as np

    from .masks import normalize_mask

    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for i in preview_indices(len(images)):
        frame = np.asarray(images[i])
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        alpha = None
        if frame.shape[-1] == 4:
            alpha = normalize_mask(frame[:, :, 3])
            frame = frame[:, :, :3]
        if masks is not None and i < len(masks):
            alpha = normalize_mask(masks[i])

        frame = frame.astype(np.float32)
        if alpha is not None:
            if alpha.shape != frame.shape[:2]:
                alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            frame = frame * alpha[..., None] + _PREVIEW_BACKDROP * (1.0 - alpha[..., None])

        height, width = frame.shape[:2]
        if width > _PREVIEW_WIDTH:
            scale = _PREVIEW_WIDTH / width
            frame = cv2.resize(
                frame, (_PREVIEW_WIDTH, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        name = names[i] if i < len(names) else f"frame_{i + 1:05d}_"
        path = destination / f"{i:05d}_{Path(name).stem}.jpg"
        cv2.imwrite(
            str(path), np.clip(frame, 0, 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), _PREVIEW_QUALITY],
        )
        written.append(str(path))
    return written


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

    def _capture_previews(self, event: RunEvent) -> List[str]:
        """Snapshot a few of this step's output frames, off the Context.

        Runs on the run thread, between steps — a handful of downscaled
        JPEG writes, next to steps measured in minutes. Every failure here
        is swallowed: a debugging aid must never be the thing that kills a
        two-hour run.
        """
        if event.context is None or not self.state.output_dir:
            return []
        try:
            images = event.context.get("dataset.images")
        except (KeyError, AttributeError, TypeError):
            return []
        if not images:
            return []
        try:
            masks = event.context.get("dataset.masks")
        except (KeyError, AttributeError, TypeError):
            masks = None
        try:
            names = event.context.get("dataset.image_names") or []
        except (KeyError, AttributeError, TypeError):
            names = []

        destination = (
            Path(self.state.output_dir) / _PREVIEW_DIRNAME
            / f"{event.index:02d}_{event.step_id}"
        )
        try:
            return write_previews(images, masks, names, destination)
        except Exception:  # noqa: BLE001 - see the docstring
            logger.warning("could not write previews for step %s", event.step_id, exc_info=True)
            return []

    def _on_event(self, event: RunEvent) -> None:
        # Outside the lock: this writes eight JPEGs, and the UI polls the
        # same lock once a second.
        previews = self._capture_previews(event) if event.kind == "step_end" else []

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
                record.previews = previews
            elif event.kind == "step_error":
                record = state.steps[event.index - 1]
                record.status = "failed"
                record.elapsed = event.elapsed
            elif event.kind == "step_skipped":
                # Advance the progress bar past it, but leave it visible in
                # the step table: "why is there no ply/" is answered by
                # seeing the step sitting there marked skipped.
                state.current = event.index
                state.steps[event.index - 1].status = "skipped"
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

            # Block here, not at submit time: the wait belongs on the
            # Progress tab where it is visible and cancellable, not in a
            # button handler that would just appear to hang. Scoped to the
            # steps this run will actually execute — fast_helical must not
            # wait on SeedVR2's 6 GB for an upscale it does not do.
            needed = required_for_steps(step.step for step in spec.enabled_steps())
            if needed and not all(is_ready(key) for key in needed):
                def report(missing):
                    with self._lock:
                        self.state.message = (
                            f"waiting for model download: {', '.join(missing)} "
                            f"(~{sum(registry()[k].approx_gb for k in missing):.0f} GB)"
                        )

                logger.info("models this workflow needs: %s", ", ".join(needed))
                report(needed)
                wait_until_ready(needed, on_wait=report)
                with self._lock:
                    self.state.message = ""
                logger.info("all required models present")

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


def workflow_outputs(name: str) -> List[str]:
    """Which of the Outputs checkboxes this workflow actually understands."""
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    return [label for label, param in OUTPUT_PARAMS.items() if param in spec.params]


def workflow_needs_a_dataset(name: str) -> bool:
    """Whether this workflow reads frames it does not itself produce.

    A workflow that starts from a photo renders its own views first
    (sam3d_body -> render), so nothing reads `dataset.images` until
    something has written it. One that starts from a dataset reads it
    immediately. Walking the steps in order and asking which comes first
    answers "can this run from a photo?" without a flag in the YAML that
    could disagree with the steps under it.
    """
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    written = set()
    for step in spec.steps:
        for path in step.inputs.values():
            root = path.split(".")[0]
            if root == "dataset" and path not in written and path not in (
                "dataset.reference_image", "dataset.prompt", "dataset.anchor_image",
            ):
                return True
        written.update(step.outputs.values())
    return False


def workflow_params_yaml(name: str) -> str:
    from .cli import resolve_workflow

    spec = WorkflowSpec.from_yaml(resolve_workflow(name))
    # The output switches are deliberately left out: they have a dedicated
    # control, and showing them here too would give the same setting two
    # editable homes that disagree the moment someone touches one.
    params = {k: v for k, v in spec.params.items() if k not in OUTPUT_PARAMS.values()}
    if not params:
        return "# this workflow declares no params\n"
    return yaml.safe_dump(params, sort_keys=False, allow_unicode=True, width=100)


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
    icons = {"pending": "·", "running": "▶", "done": "✓", "failed": "✗", "skipped": "–"}
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
            results_zip = gr.File(label="Result (.zip)")
            results_gallery = gr.Gallery(label="Final frames", columns=6, height=400)

            gr.Markdown(
                f"### Per-step frames\n_{_PREVIEW_FRAMES} frames spaced evenly "
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
                label="Per-step frames", columns=_PREVIEW_FRAMES, height=600,
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
            choices = workflow_outputs(name)
            return (
                workflow_summary(name),
                workflow_params_yaml(name),
                gr.update(choices=choices, value=choices, visible=bool(choices)),
            )

        workflow_in.change(
            on_workflow_change, inputs=workflow_in,
            outputs=[summary_out, params_in, outputs_in],
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
                     params_text, selected_outputs):
            try:
                params = yaml.safe_load(params_text) or {}
            except yaml.YAMLError as exc:
                raise gr.Error(f"Params are not valid YAML: {exc}")
            if not isinstance(params, dict):
                raise gr.Error("Params must be a YAML mapping.")

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
                    params[OUTPUT_PARAMS[label]] = label in chosen

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
                if workflow_needs_a_dataset(workflow):
                    raise gr.Error(
                        f"'{workflow}' starts from an existing dataset — its first "
                        "step reads frames it does not render. A photo needs a "
                        "workflow that begins with sam3d_body + render; none is "
                        "shipped. Pick a dataset directory or upload a .zip."
                    )
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
            inputs=[workflow_in, source_in, dataset_in, zip_in, photo_in, prompt_in,
                    params_in, outputs_in],
            outputs=[],
        ).success(stream, outputs=progress_outputs)

        attach_btn.click(stream, outputs=progress_outputs)
        cancel_btn.click(lambda: manager.cancel(), outputs=[])

        def on_results():
            state, _ = manager.snapshot()
            directory = state.output_dir
            if not directory or not Path(directory).exists():
                return "No output directory yet — start a run first.", [], None

            directories = result_dirs(Path(directory))
            if not directories:
                # Distinguish "still running" from "ran and produced nothing":
                # the second means the export steps were switched off or
                # failed, and looking in the run directory is the next move.
                note = (
                    "still running — the exports are the last steps"
                    if state.status == "running"
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

        def stream_results(step_filter: str):
            """Poll the Results tab for as long as the run is going.

            Same shape as `stream()` on the Progress tab and for the same
            reason: the per-step frames are worth watching *during* a
            two-hour run, and a generator that reads shared state cannot
            take the run down with it if the connection drops.

            Only the previews change while a run is in flight — the archive
            is not built until there is something to put in it — so the
            expensive half is skipped until the end.
            """
            while True:
                state, _ = manager.snapshot()
                running = manager.is_running
                if running:
                    finished = (gr.update(), gr.update(), gr.update())
                else:
                    finished = on_results()
                yield (
                    *finished,
                    gr.update(choices=preview_step_choices(state)),
                    preview_gallery(state, step_filter),
                )
                if not running:
                    break
                time.sleep(3.0)

        results_outputs = [results_info, results_gallery, results_zip,
                           preview_step_in, preview_gallery_out]

        results_refresh.click(
            stream_results, inputs=preview_step_in, outputs=results_outputs
        )

        def on_preview_filter(step_filter: str):
            state, _ = manager.snapshot()
            return preview_gallery(state, step_filter)

        preview_step_in.change(
            on_preview_filter, inputs=preview_step_in, outputs=preview_gallery_out
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
