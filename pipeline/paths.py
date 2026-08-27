"""Where this process is allowed to write, resolved once from the environment.

Every path a run produces — the final dataset, brush's .ply, the COLMAP
export, subprocess IPC pickles, log files — has to land on the *mounted
volume*, not the container's writable layer. On a rented pod the writable
layer is a small overlay that a single 81-frame batch of pickles can fill,
and anything written there is gone when the pod is recycled. That failure
mode is not obvious from the error you get (`Disk quota exceeded`, or a
silently vanished output directory after a restart), which is why this is
one module read by everything rather than a convention each caller
remembers.

Defaults are the container's layout; every one is overridable by env var so
the same code runs unchanged on a laptop with no /data at all:

    B2C_DATA_DIR    /data           the mounted volume itself
    B2C_OUTPUT_DIR  $DATA/output    workflow outputs (splats, COLMAP, frames)
    B2C_LOG_DIR     $DATA/logs      one log file per run
    B2C_UPLOAD_DIR  $DATA/uploads   what the web UI receives
    B2C_MODELS_DIR  $DATA/models    weights whose downloader ignores HF_HOME
    TMPDIR          $DATA/tmp       dispatcher IPC pickles (stdlib reads this)

If B2C_DATA_DIR doesn't exist and can't be created (a dev box with no
volume mounted), everything falls back under ./output relative to the
repo — a local `python -m pipeline.cli run ...` keeps working exactly as
it did before this module existed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATA_DIR = "/data"


def _usable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK)


def data_dir() -> Path:
    """The mounted volume, or a repo-local fallback when there isn't one."""
    candidate = Path(os.environ.get("B2C_DATA_DIR", _DEFAULT_DATA_DIR))
    if _usable(candidate):
        return candidate
    return REPO_ROOT / "output" / "_local_data"


def _sub(env_var: str, name: str) -> Path:
    override = os.environ.get(env_var)
    path = Path(override) if override else data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    return _sub("B2C_OUTPUT_DIR", "output")


def log_dir() -> Path:
    return _sub("B2C_LOG_DIR", "logs")


def upload_dir() -> Path:
    return _sub("B2C_UPLOAD_DIR", "uploads")


def models_dir() -> Path:
    """Weights that do NOT come down through huggingface_hub.

    Most of the pipeline's checkpoints are `from_pretrained`/
    `snapshot_download` calls, which honour HF_HOME and so land in
    $DATA/hf_cache without anyone thinking about it. Two do not:

      * seedvr2 hands its vendored downloader an explicit `model_dir` that
        bypasses HF_HOME entirely, and it defaulted to the RELATIVE path
        "models/SEEDVR2" — which resolves against the worker subprocess's
        working directory, i.e. inside the container. Several GB of DiT
        and VAE weights into a pod's 20 GB container disk, re-downloaded
        from scratch every time the pod is recycled.
      * face_landmarks fetches MediaPipe's .task/.tflite files straight
        from a Google Storage URL into ~/.cache, with the same problem in
        miniature.

    Both now default under here.
    """
    return _sub("B2C_MODELS_DIR", "models")


def run_jobs_dir() -> Path:
    """Scratch for the `RunJob`/`RunState` JSON the web UI's GPU scheduler
    and `pipeline.run_worker` exchange across the process boundary.

    Not TMPDIR: a status file needs to survive exactly as long as the
    worker process reading and writing it — the same lifetime as everything
    else under the volume — and keeping it out of the dispatchers' own
    per-step IPC scratch means `ls`ing one directory is not noisy with the
    other's churn.
    """
    return _sub("B2C_RUN_JOBS_DIR", "run_jobs")


def configure_tmpdir() -> Path:
    """Point the stdlib's temp machinery at the volume.

    SubprocessPythonDispatcher round-trips every step's inputs and outputs
    through `tempfile.TemporaryDirectory()`. For an 81-frame 720x1280 batch
    that is roughly 220MB in each direction, per step — enough to exhaust a
    container's default /tmp. Setting TMPDIR *and* tempfile.tempdir covers
    both the child processes (which read the env var) and this process
    (which cached its answer at import time).
    """
    tmp = _sub("TMPDIR", "tmp")
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = str(tmp)
    return tmp
