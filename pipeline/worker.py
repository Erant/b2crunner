"""Entry point run *inside* an isolated venv by SubprocessDispatcher.

Deliberately tiny and dependency-free (stdlib only) so it can be invoked by
any isolated environment's interpreter without needing this repo's full
dependency set installed there — only the specific step module's own deps
(e.g. sam-3d-body's env needs detectron2, not diffusers) plus this `pipeline`
package installed in editable mode. `pipeline.logging_setup` and
`pipeline.paths` are imported here and are stdlib-only for exactly that
reason; keep them that way.

Protocol: argv is [step_name, input_pickle_path, params_json_path,
output_pickle_path]. Inputs/outputs are pickled dicts of plain
numpy/str/dict data — keep step signatures to picklable types, no live GPU
tensors crossing the process boundary.

Everything this process writes to stdout is relayed line by line into the
parent's log by SubprocessPythonDispatcher, so `logger.info` here reaches
the console, the run's log file and the web UI. Before that relay existed
these steps ran silently for tens of minutes; don't reach for
`capture_output` again.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
import traceback


def _describe(value) -> str:
    """A one-line shape summary for a pickled input, without importing numpy."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"array{tuple(shape)} {getattr(value, 'dtype', '')}".strip()
    if isinstance(value, (list, tuple)):
        inner = _describe(value[0]) if value else "empty"
        return f"{type(value).__name__}[{len(value)}] of {inner}"
    if isinstance(value, dict):
        return f"dict({', '.join(sorted(value)[:6])}{'...' if len(value) > 6 else ''})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def main() -> None:
    step_name, input_path, params_path, output_path = sys.argv[1:5]

    from .logging_setup import setup_worker_logging

    setup_worker_logging()
    logger = logging.getLogger("pipeline.worker")

    from . import steps  # noqa: F401  registers all Step subclasses
    from .registry import get_step_class

    with open(input_path, "rb") as f:
        inputs = pickle.load(f)
    with open(params_path, "r") as f:
        params = json.load(f)

    # Logged rather than assumed: a step failing on an input it didn't
    # expect is the most common subprocess failure, and the pickle is gone
    # by the time you read the traceback (the parent's TemporaryDirectory
    # is cleaned up on the way out).
    logger.info("running '%s' in %s", step_name, sys.executable)
    for name, value in inputs.items():
        logger.info("  input  %s = %s", name, _describe(value))
    for name, value in sorted(params.items()):
        logger.info("  param  %s = %s", name, _describe(value))

    started = time.time()
    try:
        step = get_step_class(step_name)()
        step.load(params)
        try:
            outputs = step.run(inputs, params)
        finally:
            step.unload()
    except Exception:
        logger.error("'%s' raised after %.1fs", step_name, time.time() - started)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

    for name, value in outputs.items():
        logger.info("  output %s = %s", name, _describe(value))
    logger.info("'%s' finished in %.1fs", step_name, time.time() - started)

    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()
