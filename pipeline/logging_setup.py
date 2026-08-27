"""One place that decides what a run's output looks like.

The pipeline's previous logging was `logging.basicConfig(level=INFO if -v
else WARNING)` — no timestamps, no file, and every message from a step
indistinguishable from every other. That is survivable when you are sitting
in front of the box. It is not survivable on a rented pod where the only
thing you can see is a log pane, the interesting steps run for tens of
minutes, and a rebuild to add a print statement costs an hour.

So: timestamps and elapsed time on every line, a per-run file on the mounted
volume that outlives the container's stdout buffer, and unbuffered stdout so
`docker logs` shows a line the moment it happens instead of when the 8KB
pipe buffer fills.

`elapsed` is in the format on purpose. Wall-clock timestamps answer "when",
but the question you actually ask of a stuck run is "how long has it been
doing this", and subtracting two ISO timestamps in your head at 3am is how
you misread a 40-minute hang as a 4-minute one.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Optional

from .paths import log_dir

_START = time.time()

FORMAT = "%(asctime)s %(elapsed)9s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


class _ElapsedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        seconds = int(record.created - _START)
        record.elapsed = f"+{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        return True


def setup_logging(
    verbose: bool = True,
    log_file: Optional[str | Path] = None,
    run_name: str = "run",
) -> Optional[Path]:
    """Configure root logging for a CLI run or a UI-driven run.

    Returns the log file's path, or None if file logging was disabled.
    `log_file` may be an explicit path; otherwise one is named after
    `run_name` under B2C_LOG_DIR. Pass log_file="" to skip the file.
    """
    # Line-buffered stdout: without this, Python block-buffers when its
    # stdout is a pipe (which it always is under `docker logs`), so a long
    # step looks frozen and then dumps everything at once. PYTHONUNBUFFERED
    # in the image covers the same ground; this covers invocations that
    # don't come through the image.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # not a real stream (pytest capture)
        pass

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if os.environ.get("B2C_DEBUG") else logging.INFO)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)
    elapsed = _ElapsedFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(formatter)
    console.addFilter(elapsed)
    root.addHandler(console)

    resolved: Optional[Path] = None
    if log_file != "":
        resolved = Path(log_file) if log_file else log_dir() / f"{run_name}.log"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # The file always gets INFO even when the console is quiet: the
        # whole point of it is being able to answer "what happened" after
        # the fact, and a quiet console is a display preference, not a
        # statement about what's worth recording.
        file_handler = logging.FileHandler(resolved, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG if os.environ.get("B2C_DEBUG") else logging.INFO)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(elapsed)
        root.addHandler(file_handler)

    # Third-party libraries that are chatty enough to bury the pipeline's
    # own output. Raised individually rather than by setting the root
    # higher, which would also silence the steps.
    for noisy in ("urllib3", "filelock", "matplotlib", "PIL", "httpx", "asyncio",
                  "httpcore", "OpenGL", "huggingface_hub", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.captureWarnings(True)
    return resolved


def timestamped_run_name(prefix: str = "run") -> str:
    # The timestamp alone is only second-resolution: two runs started within
    # the same second (routine once several GPUs can start runs at once)
    # would otherwise share an output dir and a log file. The suffix isn't
    # for humans, just for uniqueness.
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


class QueueLogHandler(logging.Handler):
    """Feeds formatted log lines into a caller-supplied `put(str)`.

    Used by the web UI to mirror a run's log into the browser without
    re-reading the file. Deliberately a handler rather than a tail on the
    log file: a tail has to guess at partial lines and re-open on rotation,
    and this needs neither.
    """

    def __init__(self, put) -> None:
        super().__init__()
        self._put = put
        self.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))
        self.addFilter(_ElapsedFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._put(self.format(record))
        except Exception:  # a UI consumer going away must not break the run
            pass


# The format used inside a `pipeline.worker` subprocess. Deliberately
# terser than FORMAT: the parent stamps every relayed line with its own
# time, elapsed and step name already, so repeating that here would produce
# lines that are half prefix. Level and logger name are kept because those
# are the parts the parent cannot know.
WORKER_FORMAT = "%(levelname)s %(name)s: %(message)s"


def setup_worker_logging() -> None:
    """Configure logging inside an isolated venv's `pipeline.worker` process.

    Without this, a step's `logger.info(...)` in a subprocess-dispatched env
    went nowhere at all: nothing configured a handler there, so logging's
    last-resort handler dropped everything below WARNING. Half the
    diagnostics the steps already write were invisible precisely in the
    environments hardest to debug.
    """
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("B2C_DEBUG") else logging.INFO,
        format=WORKER_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("urllib3", "filelock", "matplotlib", "PIL", "httpx", "httpcore",
                  "OpenGL", "huggingface_hub", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
