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

Stdlib-only, like everything else a step module is allowed to import at
module scope (see tests/test_import_discipline.py).
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

# Burst allowance and steady-state rate for relayed output. 60 lines of
# burst covers a startup banner or a Python traceback intact; 0.5 lines/sec
# is roughly one progress repaint every two seconds once a bar gets going.
_BUCKET_CAPACITY = 60.0
_BUCKET_REFILL_PER_SEC = 0.5

_ERROR_TAIL_LINES = 60


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
