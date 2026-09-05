"""What a run looks like from the outside, shared by every process that
needs to describe or read one: `pipeline.run_worker` (which builds a
`RunState` as it goes and publishes it as JSON), the scheduler in
`pipeline.webui` (which reads that JSON back), and the UI itself.

Deliberately dependency-light — no `gradio` import — so `run_worker.py` can
import it without dragging the UI framework into a plain workflow-execution
process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# The per-step contact sheet: eight frames evenly spaced through the batch,
# written after every step. Eight because an 81-frame orbit strides by ten —
# frames 1, 11, 21, ... — which is a quarter turn between neighbours, enough
# to see a step break one side of the subject and not the other.
#
# Downscaled JPEGs, not the frames themselves: a run has around fifteen
# steps, so this is ~120 images, and at full resolution that is a gigabyte
# of PNG nobody can load over a pod's HTTP proxy.
PREVIEW_FRAMES = 8
PREVIEW_WIDTH = 480
PREVIEW_QUALITY = 82
PREVIEW_DIRNAME = "_previews"

# What a masked-out pixel is drawn over. Mid-grey rather than black or
# white so it reads as "nothing here" against both a dark jacket and a
# blown-out background — the two cases where a mask step going wrong is
# otherwise invisible.
PREVIEW_BACKDROP = 128


@dataclass
class StepRecord:
    index: int
    step_id: str
    step_name: str
    status: str = "running"
    elapsed: float = 0.0
    # Paths to this step's preview frames, filled in as it finishes.
    previews: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "step_id": self.step_id, "step_name": self.step_name,
            "status": self.status, "elapsed": self.elapsed, "previews": list(self.previews),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepRecord":
        return cls(
            index=data["index"], step_id=data["step_id"], step_name=data["step_name"],
            status=data.get("status", "pending"), elapsed=data.get("elapsed", 0.0),
            previews=list(data.get("previews", [])),
        )


@dataclass
class RunState:
    name: str = ""
    workflow: str = ""
    # queued | running | done | failed | cancelled
    status: str = "idle"
    message: str = ""
    started: float = 0.0
    finished: float = 0.0
    current: int = 0
    total: int = 0
    steps: List[StepRecord] = field(default_factory=list)
    output_dir: Optional[Path] = None
    log_path: Optional[Path] = None
    error: str = ""
    # Which physical GPU this run landed on — filled in by the scheduler,
    # not by the worker itself (the worker only ever sees its own pinned
    # view, where it is always device 0).
    gpu_index: Optional[int] = None
    # The output switches this run resolved, by name. Published because
    # packaging cannot otherwise know them: most outputs gate a step, so
    # their directory is simply absent when they are off — but `debug/` is
    # written as a side effect of steps a run needs anyway, so whether to
    # carry it into the archive is a decision only the run itself made.
    # Empty for a run that predates this, which `runs.py` reads as "carry
    # everything", the behaviour those runs were packaged with.
    outputs: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "workflow": self.workflow, "status": self.status,
            "message": self.message, "started": self.started, "finished": self.finished,
            "current": self.current, "total": self.total,
            "steps": [s.to_dict() for s in self.steps],
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "log_path": str(self.log_path) if self.log_path else None,
            "error": self.error, "gpu_index": self.gpu_index,
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        return cls(
            name=data.get("name", ""), workflow=data.get("workflow", ""),
            status=data.get("status", "idle"), message=data.get("message", ""),
            started=data.get("started", 0.0), finished=data.get("finished", 0.0),
            current=data.get("current", 0), total=data.get("total", 0),
            steps=[StepRecord.from_dict(s) for s in data.get("steps", [])],
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            log_path=Path(data["log_path"]) if data.get("log_path") else None,
            error=data.get("error", ""), gpu_index=data.get("gpu_index"),
            outputs=dict(data.get("outputs") or {}),
        )


@dataclass
class RunJob:
    """Everything `pipeline.run_worker` needs to execute one run, crossing
    the process boundary as one JSON file the scheduler writes and the
    worker reads once at startup."""

    run_name: str
    workflow_name: str
    workflow_path: str
    output_dir: str
    envs_path: str = ""
    global_overrides: Dict[str, Any] = field(default_factory=dict)
    step_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dataset_dir: Optional[str] = None
    reference_image: Optional[str] = None
    prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name, "workflow_name": self.workflow_name,
            "workflow_path": self.workflow_path, "output_dir": self.output_dir,
            "envs_path": self.envs_path, "global_overrides": self.global_overrides,
            "step_overrides": self.step_overrides, "dataset_dir": self.dataset_dir,
            "reference_image": self.reference_image, "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunJob":
        return cls(
            run_name=data["run_name"], workflow_name=data["workflow_name"],
            workflow_path=data["workflow_path"], output_dir=data["output_dir"],
            envs_path=data.get("envs_path", ""),
            global_overrides=dict(data.get("global_overrides", {})),
            step_overrides={k: dict(v) for k, v in data.get("step_overrides", {}).items()},
            dataset_dir=data.get("dataset_dir"), reference_image=data.get("reference_image"),
            prompt=data.get("prompt", ""),
        )


def preview_indices(count: int, wanted: int = PREVIEW_FRAMES) -> List[int]:
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
            frame = frame * alpha[..., None] + PREVIEW_BACKDROP * (1.0 - alpha[..., None])

        height, width = frame.shape[:2]
        if width > PREVIEW_WIDTH:
            scale = PREVIEW_WIDTH / width
            frame = cv2.resize(
                frame, (PREVIEW_WIDTH, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        name = names[i] if i < len(names) else f"frame_{i + 1:05d}_"
        path = destination / f"{i:05d}_{Path(name).stem}.jpg"
        cv2.imwrite(
            str(path), np.clip(frame, 0, 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_QUALITY],
        )
        written.append(str(path))
    return written


def tail_lines(path: Path, max_lines: int = 4000, max_bytes: int = 2_000_000) -> str:
    """The last `max_lines` lines of a (possibly still-growing) log file.

    Bounded-byte read from the end rather than reading the whole file: a
    multi-hour run's log is read once a second by every viewer watching it,
    and re-reading a file that only grows would make each poll more
    expensive than the last.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # drop a partial first line
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
