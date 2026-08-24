"""Entry point run *inside* an isolated venv by SubprocessDispatcher.

Deliberately tiny and dependency-free (stdlib only) so it can be invoked by
any isolated environment's interpreter without needing this repo's full
dependency set installed there — only the specific step module's own deps
(e.g. sam-3d-body's env needs detectron2, not diffusers) plus this `pipeline`
package installed in editable mode. `pipeline.logging_setup` and
`pipeline.paths` are imported here and are stdlib-only for exactly that
reason; keep them that way.

Two modes, both driven by SubprocessPythonDispatcher:

**One-shot** (the default, `keep_loaded: false`) — argv is [step_name,
input_pickle_path, params_json_path, output_pickle_path]. Load the step, run
it, unload, exit. One process per invocation.

**Serve** (`--serve`, selected by `keep_loaded: true`) — the same work, but
in a loop, one request per line of JSON on stdin, with the Step instance
kept alive between jobs. This exists because `pipeline/workflows/
fast_helical_full.yaml` calls `wan22_vace_denoise` twice with six other
steps in between, so the two calls cannot be merged; under the one-shot mode
that is ~47 GB of weights read off a RunPod network volume *twice*. The pod
has the DRAM to hold them; the network drive should be paid once.

Residency is DRAM residency, not VRAM residency — the distinction is the
whole design. Between jobs the worker partially evicts: the Step hands the
card back (Step.release_vram) and the allocator is emptied, while the
weights stay in host RAM. It has to, because one of the steps between those
two denoise passes is `brush`, which trains a Gaussian splat on the same
GPU; a worker holding ~35 GB of Wan experts in VRAM across that gap would
OOM it on any card, which would be worse than the reloading it replaced.
Full eviction — unload(), instance dropped, host RAM freed — happens at
shutdown. See `release_vram()` and `serve()` below for the request kinds.

Inputs/outputs are pickled dicts of plain numpy/str/dict data — keep step
signatures to picklable types, no live GPU tensors crossing the process
boundary. That is true in both modes: serve mode changes who holds the
model, not what crosses the pipe.

Everything this process writes to stdout is relayed line by line into the
parent's log by SubprocessPythonDispatcher, so `logger.info` here reaches
the console, the run's log file and the web UI. Before that relay existed
these steps ran silently for tens of minutes; don't reach for
`capture_output` again. The one line that is *not* log output is the
SERVE_MARKER completion line below — the parent strips it and keeps
everything else.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
import traceback

# Prefix of the single non-log line serve mode writes per job, carrying a
# JSON status object: {"ok": true} or {"ok": false, "error": "..."}.
#
# Why a marker in the stdout stream rather than a second pipe: the parent
# already has exactly one relay loop reading this stream line by line, and
# it must stay that way — a step's own output and its completion signal
# arriving on different fds would need either a reader thread or a select()
# loop, and the failure mode of getting that wrong (a completion seen
# before the traceback that explains it) is precisely the kind of thing
# that is invisible until a 40-minute step fails on a rented pod. One
# ordered stream, one reader, no interleaving question to get wrong.
#
# The prefix is deliberately ugly so no step's log line can be mistaken for
# it. `pipeline/dispatch/subprocess_python.py` imports this constant rather
# than repeating the literal.
SERVE_MARKER = "@@b2c-worker-status@@"


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


def _import_steps() -> None:
    """Populate the step registry.

    `B2C_EXTRA_STEP_MODULES` (comma-separated importable module names) is
    the worker's only extension point: it imports those modules after
    pipeline's own, so a step defined outside this repo — or a stub step
    defined in tests/ — can be dispatched without being added to
    pipeline/steps/__init__.py. Imported after, not before, so an
    out-of-tree module can never shadow a shipped step's registration
    (register_step raises on a duplicate name).
    """
    from . import steps  # noqa: F401  registers all Step subclasses

    extra = os.environ.get("B2C_EXTRA_STEP_MODULES", "").strip()
    if not extra:
        return
    import importlib

    for name in (part.strip() for part in extra.split(",")):
        if name:
            importlib.import_module(name)


def _read_job(input_path: str, params_path: str):
    with open(input_path, "rb") as f:
        inputs = pickle.load(f)
    with open(params_path, "r") as f:
        params = json.load(f)
    return inputs, params


def _describe_job(logger, step_name: str, inputs, params) -> None:
    # Logged rather than assumed: a step failing on an input it didn't
    # expect is the most common subprocess failure, and the pickle is gone
    # by the time you read the traceback (the parent's TemporaryDirectory
    # is cleaned up on the way out).
    logger.info("running '%s' in %s", step_name, sys.executable)
    for name, value in inputs.items():
        logger.info("  input  %s = %s", name, _describe(value))
    for name, value in sorted(params.items()):
        logger.info("  param  %s = %s", name, _describe(value))


def _write_outputs(logger, step_name: str, outputs, output_path: str, started: float) -> None:
    for name, value in outputs.items():
        logger.info("  output %s = %s", name, _describe(value))
    logger.info("'%s' finished in %.1fs", step_name, time.time() - started)

    with open(output_path, "wb") as f:
        pickle.dump(outputs, f)


# --------------------------------------------------------------------------
# One-shot mode
# --------------------------------------------------------------------------


def run_once(argv) -> None:
    step_name, input_path, params_path, output_path = argv

    logger = logging.getLogger("pipeline.worker")
    from .registry import get_step_class

    inputs, params = _read_job(input_path, params_path)
    _describe_job(logger, step_name, inputs, params)

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

    _write_outputs(logger, step_name, outputs, output_path, started)


# --------------------------------------------------------------------------
# Serve mode
# --------------------------------------------------------------------------


def load_signature(step_class, params):
    """The part of `params` that a reused instance's `load()` depends on.

    THE RELOAD RULE, in one place, because getting it wrong is either a
    silent correctness bug (pass 2 runs against pass 1's checkpoint) or the
    loss of the entire feature (reloading 47 GB because `strength` changed
    from 1.0 to 0.8).

    `Step.load(params)` and `Step.run(inputs, params)` are handed the *same*
    dict, and nothing in the interface says which keys belong to which. So
    a Step class may declare `LOAD_PARAMS`, an iterable of the param names
    its `load()` actually reads; the resident instance is reused only while
    the values under those names are unchanged, and rebuilt when they are
    not.

    When a Step declares nothing (`LOAD_PARAMS` absent) this returns None,
    meaning "reuse unconditionally". That is the deliberate default, for
    two reasons:

      * It matches what `keep_loaded: true` already means for
        InProcessDispatcher, which keys its cached instance on the step
        name alone and has never compared params.
      * The alternative default — reload whenever any param differs —
        would reload on `strength: 1.0` -> `strength: 0.8` in
        fast_helical_full.yaml, i.e. it would do nothing at all for the
        only workflow this feature exists for. A default that silently
        no-ops the feature is worse than one that trusts an opt-in flag.

    `keep_loaded` is opt-in, per step, written by whoever can see all of
    that step's call sites in the workflow. The changed params are logged
    at each reuse (see `_serve`) so a mistake is visible in the run log
    rather than only in the output.
    """
    keys = getattr(step_class, "LOAD_PARAMS", None)
    if keys is None:
        return None
    return {key: params.get(key) for key in keys}


def _emit_status(payload: dict) -> None:
    """Write the one non-log line per job, after everything it explains.

    Both streams are flushed first: the parent merges stderr into stdout at
    the fd level, and `traceback.print_exc()` writes to stderr, so without
    this a failure's traceback could land *after* the status line that
    reports it and be read as part of the next job.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.write(f"{SERVE_MARKER} {json.dumps(payload)}\n")
    sys.stdout.flush()


def _empty_cuda_cache() -> None:
    """Hand the caching allocator's free blocks back to the driver.

    Guarded import because this module runs in every isolated env, and not
    all of them have torch (nor should this file grow a hard dependency on
    it — see the module docstring). Failures are swallowed: a driver
    hiccup here must not fail a step that has already produced its output.
    """
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()


def release_vram(logger, step) -> None:
    """Partial eviction: VRAM back to the driver, weights left in host RAM.

    Called after EVERY job a resident worker serves, which is the whole
    reason `keep_loaded` is safe to turn on. fast_helical_full.yaml puts
    `brush` — a GPU program — between its two `wan22_vace_denoise` passes;
    a worker that sat on ~35 GB of Wan experts across that gap would OOM
    brush on any card, i.e. would be a regression over the reload-every-
    time behaviour it replaces. What `keep_loaded` is buying is skipping
    the ~47 GB *network* read, and DRAM residency alone buys that. The next
    job's re-upload is a PCIe copy.

    Both halves are needed and in this order: the Step moves its modules
    off the card, then the allocator is told to release what that freed —
    PyTorch does not return cached blocks to the driver just because the
    tensors in them moved (see dispatch/in_process.py, where that cost a
    real OOM). `empty_cache()` runs even if the step's own hook raised or
    is the default no-op, because the allocator may still be holding
    activations from the run that just finished.
    """
    if step is not None:
        try:
            step.release_vram()
        except Exception:
            # Non-fatal on purpose: the job's outputs are already written
            # and the weights are still in DRAM. Worst case the next GPU
            # step OOMs, which is a far better failure than discarding a
            # completed 40-minute denoise.
            logger.error("release_vram() raised; the model may still be on the GPU")
            traceback.print_exc()
    _empty_cuda_cache()


def serve() -> int:
    """Serve requests from stdin until EOF, keeping the loaded Step alive.

    One JSON object per line. `kind` selects what to do, defaulting to
    "run" so a request is just the job:

      {"kind": "run", "step", "inputs", "params", "outputs"}
          The last three are file paths — exactly the four argv values
          one-shot mode takes. Reuses the resident Step per the rule in
          load_signature(), then partially evicts (see release_vram) before
          reporting, so the GPU is free by the time the parent's run()
          returns and the next step in the workflow can have the card.
      {"kind": "release_vram"}
          Partial eviction on demand, for a caller that needs the card back
          at a moment that isn't a job boundary.
      {"kind": "shutdown"}
          Full eviction — unload(), drop the instance, exit 0. Closing
          stdin does the same thing; having both means a clean end-of-run
          shutdown is distinguishable in the log from the parent dying.

    One SERVE_MARKER status line back per request. Strictly synchronous —
    the parent writes one request and blocks until its status arrives, so
    there is no request id, no queue and nothing to correlate.
    """
    logger = logging.getLogger("pipeline.worker")
    from .registry import get_step_class

    logger.info("resident worker ready (pid %d, %s)", os.getpid(), sys.executable)

    resident_name: str | None = None
    resident_signature = None
    resident_step = None
    failed = False

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                # Parent closed stdin: end of run, or the parent died.
                logger.info("stdin closed; shutting down")
                break
            line = line.strip()
            if not line:
                continue

            job = json.loads(line)
            kind = job.get("kind", "run")

            if kind == "shutdown":
                logger.info("shutdown requested")
                break

            if kind == "release_vram":
                release_vram(logger, resident_step)
                _emit_status({"ok": True, "released": True})
                continue

            if kind != "run":
                _emit_status({"ok": False, "error": f"unknown request kind {kind!r}"})
                failed = True
                break

            step_name = job["step"]
            started = time.time()
            try:
                inputs, params = _read_job(job["inputs"], job["params"])
                _describe_job(logger, step_name, inputs, params)

                step_class = get_step_class(step_name)
                signature = load_signature(step_class, params)

                if resident_step is not None:
                    reason = ""
                    if resident_name != step_name:
                        reason = f"different step (was '{resident_name}')"
                    elif signature is not None and signature != resident_signature:
                        changed = sorted(
                            key for key in set(signature) | set(resident_signature or {})
                            if signature.get(key) != (resident_signature or {}).get(key)
                        )
                        reason = f"load params changed: {', '.join(changed)}"
                    if reason:
                        logger.info("unloading resident '%s': %s", resident_name, reason)
                        resident_step.unload()
                        _empty_cuda_cache()
                        resident_step = None
                        resident_name = None
                        resident_signature = None

                if resident_step is None:
                    resident_step = step_class()
                    resident_step.load(params)
                    resident_name = step_name
                    resident_signature = signature
                else:
                    logger.info(
                        "reusing loaded '%s' (no reload: %s)",
                        step_name,
                        "LOAD_PARAMS unchanged" if signature is not None
                        else "step declares no LOAD_PARAMS",
                    )

                outputs = resident_step.run(inputs, params)
                _write_outputs(logger, step_name, outputs, job["outputs"], started)
            except Exception as exc:
                logger.error("'%s' raised after %.1fs", step_name, time.time() - started)
                traceback.print_exc()
                _emit_status({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                failed = True
                # A failed job ENDS the resident worker rather than leaving
                # it serving. Two reasons, in order of how much they cost:
                #
                #  1. The exception left the Step mid-run() with unknown
                #     internal state. For the step this feature exists for
                #     that means a diffusers pipeline part-way through a
                #     denoise loop with modules scattered between CPU and
                #     GPU by accelerate's offload hooks; a CUDA OOM in a
                #     forward pass can also leave the context itself
                #     unusable, so the *next* job fails with the same error
                #     and no relation to its own inputs. Reusing 47 GB of
                #     weights owned by a process that just took an unknown
                #     fault is not the trade this feature was asking for.
                #  2. There is nothing to save. WorkflowRunner re-raises a
                #     step failure and aborts the run (runner.py's
                #     _run_one), so a worker that survived the error would
                #     be shut down by close() moments later without ever
                #     serving another job.
                break
            else:
                # Partial eviction BEFORE the status line, not after: the
                # parent treats the status as "this job is done and the
                # card is yours", and the very next thing the runner does
                # may be an in-process GPU step (brush, render_splat).
                # Releasing after would be a race the parent cannot see.
                release_vram(logger, resident_step)
                _emit_status({"ok": True, "elapsed": time.time() - started})
    finally:
        if resident_step is not None:
            # Full eviction: drop the instance so its host RAM goes too,
            # then empty the allocator once more for whatever unload()
            # freed on the card.
            logger.info("unloading resident '%s' (full eviction)", resident_name)
            try:
                resident_step.unload()
            except Exception:  # a bad unload must not mask the real result
                traceback.print_exc()
            resident_step = None
            _empty_cuda_cache()
        sys.stdout.flush()

    return 1 if failed else 0


def main() -> None:
    from .logging_setup import setup_worker_logging

    setup_worker_logging()
    _import_steps()

    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        sys.exit(serve())

    run_once(sys.argv[1:5])


if __name__ == "__main__":
    main()
