"""Run an external binary and relay its output to the log as it happens.

`brush` and `brush-splat-render` each had their own copy of the same
Popen-plus-reader-thread block, and both did the same unhelpful thing:
accumulate every line into a list, show none of it, and quote the last 50
only if the process exited non-zero. A 30,000-iteration brush training run
is tens of minutes of total silence under that scheme, which is precisely
the step you most want to watch on a rented pod.

Two things this adds beyond "log the lines":

**Carriage-return progress bars become lines.** Popen in text mode
translates a bare `\\r` to `\\n`, so indicatif/tqdm-style bars arrive here as
one line per repaint rather than one enormous line at the end. Good for
liveness, catastrophic for a log file — hence:

**A token bucket.** A burst of output (a config dump at startup, a
traceback) passes through in full, because that is where the information
is. Sustained output faster than the refill rate — a progress bar
repainting several times a second for half an hour — is sampled instead,
with a note saying how many repaints were skipped. Without this, one brush
run writes several hundred thousand lines to the volume.

**And a crash keeps more than its exit code.** `save_crashlog` writes what
a failing binary was doing into `paths.crash_dir()`. Both callers hand
their binary a `TemporaryDirectory` — brush its COLMAP export, the
rasteriser its camera list and its frames — and that directory is deleted
on the way out, exception or not. On a rented pod, that left a crash with
nothing behind it but an exit code, and the pod does not outlive the
investigation. It lives here rather than in either step because the two
binaries are the same Rust project, fail the same way, and want the same
environment recorded.

Stdlib-only, like everything else a step module is allowed to import at
module scope (see tests/test_import_discipline.py).
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .paths import crash_dir

# Burst allowance and steady-state rate for relayed output. 60 lines of
# burst covers a startup banner or a Python traceback intact; 0.5 lines/sec
# is roughly one progress repaint every two seconds once a bar gets going.
_BUCKET_CAPACITY = 60.0
_BUCKET_REFILL_PER_SEC = 0.5

_ERROR_TAIL_LINES = 60

# Frozen into every crash report. Both binaries are wgpu/Vulkan, and their
# one known-unresolved deployment question is whether a RunPod pod exposes
# the graphics capability at all (see steps/brush.py's docstring and
# docker/Dockerfile) — so the environment that decides that is the first
# thing you want to read off a crash, and the last thing you can recover
# once the pod is gone.
_CRASH_ENV_VARS = (
    "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_LOADER_LAYERS_ENABLE",
    "WGPU_BACKEND", "LD_LIBRARY_PATH", "DISPLAY",
)


class ProcessFailed(RuntimeError):
    """Non-zero exit, carrying the tail of the output for the traceback."""


def stream_command(
    cmd: Sequence[str],
    log_name: str,
    not_found_hint: str = "",
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    throttle: bool = True,
) -> List[str]:
    """Run `cmd`, logging its combined output live. Returns the tail lines.

    Raises ProcessFailed on a non-zero exit, and RuntimeError with
    `not_found_hint` appended if the binary isn't on PATH at all — the two
    failures worth telling apart, since one is a bad argv and the other is
    a bad image.
    """
    logger = logging.getLogger(f"proc.{log_name}")
    logger.info("$ %s", " ".join(cmd))

    try:
        process = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"{cmd[0]!r} not found on PATH.{(' ' + not_found_hint) if not_found_hint else ''}"
        ) from None

    tail: deque = deque(maxlen=_ERROR_TAIL_LINES)
    tokens = _BUCKET_CAPACITY
    last_refill = time.monotonic()
    skipped = 0
    started = time.monotonic()

    # `with process` rather than a bare wait(): Popen's context manager
    # closes stdout on the way out. Without it every external-binary call
    # leaks a pipe file descriptor, and this runs once per brush training,
    # once per render, and once per frame batch.
    assert process.stdout is not None
    with process:
        for raw in process.stdout:
            line = raw.rstrip()
            if not line:
                continue
            tail.append(raw)

            if not throttle:
                logger.info("%s", line)
                continue

            now = time.monotonic()
            tokens = min(_BUCKET_CAPACITY, tokens + (now - last_refill) * _BUCKET_REFILL_PER_SEC)
            last_refill = now

            if tokens >= 1.0:
                tokens -= 1.0
                if skipped:
                    logger.info("... (%d progress lines skipped)", skipped)
                    skipped = 0
                logger.info("%s", line)
            else:
                skipped += 1

    returncode = process.returncode
    elapsed = time.monotonic() - started
    if skipped:
        logger.info("... (%d progress lines skipped)", skipped)

    if returncode != 0:
        raise ProcessFailed(
            f"{log_name} failed with exit code {returncode} after {elapsed:.1f}s.\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- last {len(tail)} lines of output ---\n" + "".join(tail)
        )

    logger.info("%s finished in %.1fs", log_name, elapsed)
    return list(tail)


def describe_path(path: Optional[Path]) -> str:
    """`<path> (N bytes)`, or what is wrong with it instead.

    Crash reports are read after the machine is gone, so "the file was
    there and it was 43 bytes" and "the file was never written" have to be
    distinguishable in the text itself.
    """
    if path is None:
        return "<not recorded>"
    if not path.exists():
        return f"{path} (MISSING)"
    if path.is_dir():
        entries = sum(1 for _ in path.iterdir())
        return f"{path}/ (directory, {entries} entries)"
    return f"{path} ({path.stat().st_size} bytes)"


def save_crashlog(
    binary: str,
    *,
    cmd: Sequence[str],
    failure: str,
    summary: Sequence[str] = (),
    sections: Sequence[Tuple[str, str]] = (),
    copy: Sequence[Tuple[str, Sequence[Path]]] = (),
) -> Optional[Path]:
    """Freeze what a crashed binary was doing into a directory that survives.

    Writes `report.txt` — the caller's `summary` lines, the argv, the
    graphics environment, the caller's `sections`, and the `failure` text
    (which for a `ProcessFailed` carries the exit code and the tail of the
    binary's own output) — then copies the files in `copy`, each under the
    subdirectory it is paired with (`""` for the crash directory itself).

    The caller chooses what to copy, and the choice is always the same one:
    small things that name what the run was doing (a camera list, a COLMAP
    model) rather than the bulk it was doing it to (a hundred frames), and
    never anything that still exists elsewhere once the temp directory is
    gone.

    Never raises. This runs on the failure path, and a diagnostics problem
    must not replace the failure it was trying to describe. Returns the
    directory, or None if it could not be written.
    """
    logger = logging.getLogger(f"proc.{binary}")
    try:
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        dest = crash_dir() / f"{binary}-{stamp}"
        dest.mkdir(parents=True)

        report = [f"{binary} crash report", f"written: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        report += list(summary)
        report += ["", "--- command ---", " ".join(cmd), "", "--- environment ---"]
        report += [f"{var}={os.environ.get(var, '<unset>')}" for var in _CRASH_ENV_VARS]
        for title, body in sections:
            report += ["", f"--- {title} ---", body]
        # Last, because it is the longest: a ProcessFailed carries the tail
        # of the binary's output, and scrolling past that to reach the
        # single line saying which frame was missing is the wrong order.
        report += ["", "--- failure ---", failure]
        (dest / "report.txt").write_text("\n".join(report) + "\n")

        for subdir, files in copy:
            target = dest / subdir if subdir else dest
            target.mkdir(parents=True, exist_ok=True)
            for source in files:
                if source.exists():
                    shutil.copy2(source, target / source.name)

        return dest
    except OSError as exc:
        logger.warning("could not save %s diagnostics: %s", binary, exc)
        return None


def crashlog_note(saved: Optional[Path]) -> str:
    """How a log line refers to a crash directory that may not exist.

    `save_crashlog` swallows its own failures, so every caller has to read
    sensibly when it hands back nothing.
    """
    return f"saved to {saved}" if saved else "could not be saved (see the warning above)"
