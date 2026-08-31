"""`python -m pipeline.run_worker --job J --status S` — run one workflow to
completion as its own OS process, publishing progress as it goes.

This is `pipeline.cli run_workflow`'s orchestration (load spec -> apply
overrides -> wait on required models -> load dataset -> run -> save),
adapted for a caller that isn't watching stdout: the scheduler in
`pipeline.webui` spawns one of these per submitted run, `CUDA_VISIBLE_DEVICES`
-pinned to one physical GPU in the parent's `Popen` call (see
`GpuScheduler`), and needs to read this run's status back without blocking
on it.

There is deliberately no queue, socket, or request/response protocol: the
whole of the IPC is one small JSON file, atomically replaced after every
`RunEvent`, mirroring the marker-file pattern `pipeline.models` already uses
for prefetch status. A crashed or OOM-killed worker leaves whatever it last
published; the scheduler treats a dead process with a non-terminal status as
failed.

Nothing here sets or reads `CUDA_VISIBLE_DEVICES` — by the time this
interpreter starts, the parent has already put it in the environment, so
every step (`Param("device", "cuda", ...)`), subprocess dispatch, and
resident worker underneath sees the one physical GPU this process was
pinned to as device 0. That's the entire multi-GPU mechanism; nothing in
`pipeline/steps/*` needs to know a second GPU exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from . import steps  # noqa: F401  registers every Step; this is an entrypoint
from .dataset import Dataset
from .logging_setup import setup_logging
from .run_state import PREVIEW_DIRNAME, RunJob, RunState, StepRecord, write_previews
from .runner import RunCancelled, RunEvent, WorkflowRunner
from .workflow import WorkflowSpec, apply_ui_overrides, load_envs

logger = logging.getLogger(__name__)

# Flipped by the SIGTERM handler; checked at step boundaries only — a step
# is a single opaque call (often a subprocess holding the GPU), and tearing
# one down mid-flight risks leaving the card in a state the next run on this
# slot inherits.
_cancelled = False


def _handle_sigterm(signum, frame) -> None:
    global _cancelled
    _cancelled = True


class _StatusWriter:
    """Builds a `RunState` off `RunEvent`s and atomically publishes it."""

    def __init__(self, state: RunState, status_path: Path) -> None:
        self.state = state
        self.status_path = status_path

    def publish(self) -> None:
        tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state.to_dict()))
        os.replace(tmp, self.status_path)

    def message(self, text: str) -> None:
        self.state.message = text
        self.publish()

    def _capture_previews(self, event: RunEvent) -> list[str]:
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
            Path(self.state.output_dir) / PREVIEW_DIRNAME
            / f"{event.index:02d}_{event.step_id}"
        )
        try:
            return write_previews(images, masks, names, destination)
        except Exception:  # a debugging aid must never take down the run
            logger.warning("could not write previews for step %s", event.step_id, exc_info=True)
            return []

    def __call__(self, event: RunEvent) -> None:
        previews = self._capture_previews(event) if event.kind == "step_end" else []

        if event.kind == "step_start":
            self.state.current = event.index
            self.state.message = f"[{event.index}/{event.total}] {event.step_id}"
            self.state.steps[event.index - 1].status = "running"
        elif event.kind == "step_end":
            record = self.state.steps[event.index - 1]
            record.status = "done"
            record.elapsed = event.elapsed
            record.previews = previews
        elif event.kind == "step_error":
            record = self.state.steps[event.index - 1]
            record.status = "failed"
            record.elapsed = event.elapsed
        elif event.kind == "step_skipped":
            self.state.current = event.index
            self.state.steps[event.index - 1].status = "skipped"
        self.publish()

        if event.kind == "step_start" and _cancelled:
            raise RunCancelled(f"cancelled before step {event.index} ({event.step_id})")

    def finish(self, status: str, message: str = "", error: str = "") -> None:
        self.state.status = status
        self.state.finished = time.time()
        if message:
            self.state.message = message
        if error:
            self.state.error = error
        self.publish()


def _run(job: RunJob, status_path: Path) -> int:
    from .doctor import log_machine_banner
    from .models import is_ready, registry, required_for_steps, wait_until_ready

    log_path = setup_logging(run_name=job.run_name)

    writer = _StatusWriter(
        RunState(
            name=job.run_name, workflow=job.workflow_name, status="running",
            started=time.time(), output_dir=Path(job.output_dir), log_path=log_path,
        ),
        status_path,
    )
    writer.publish()

    try:
        spec = WorkflowSpec.from_yaml(Path(job.workflow_path))
        apply_ui_overrides(spec, job.global_overrides, job.step_overrides)
        # The submitter already applied these — this is the same rule stated
        # where the run actually happens, so a job JSON that reached here by
        # some other route cannot start an export whose `requires:` is off.
        spec.apply_output_requirements()
        spec.validate()

        if "output_root" in spec.globals and "output_root" not in job.global_overrides:
            spec.globals["output_root"] = job.output_dir

        writer.state.total = len(spec.steps)
        writer.state.steps = [
            StepRecord(i, s.id, s.step, status="pending")
            for i, s in enumerate(spec.steps, start=1)
        ]
        writer.publish()

        logger.info(
            "run '%s' (%s) on CUDA_VISIBLE_DEVICES=%s",
            job.run_name, spec.name, os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        )
        logger.info("output: %s", job.output_dir)
        log_machine_banner()

        # Scoped to the steps this run will actually execute — a run with
        # export_ply switched off must not block on a checkpoint only the
        # skipped step needs.
        needed = required_for_steps(step.step for step in spec.enabled_steps())
        if needed and not all(is_ready(key) for key in needed):
            def report(missing: list[str]) -> None:
                writer.message(
                    f"waiting for model download: {', '.join(missing)} "
                    f"(~{sum(registry()[k].approx_gb for k in missing):.0f} GB)"
                )

            logger.info("models this workflow needs: %s", ", ".join(needed))
            report(needed)
            wait_until_ready(needed, on_wait=report)
            writer.message("")
            logger.info("all required models present")

        if job.reference_image:
            dataset = Dataset.from_reference_image(job.reference_image, prompt=job.prompt or None)
            logger.info(
                "starting from reference sheet %s (%dx%d)",
                job.reference_image, *dataset.resolution,
            )
        else:
            dataset = Dataset.from_disk(job.dataset_dir)
            logger.info(
                "loaded %s: %d frames, %d cameras, %d points",
                job.dataset_dir, len(dataset.images), len(dataset.cameras),
                len(dataset.points_3d[0]),
            )
            if job.prompt:
                dataset.prompt = job.prompt

        envs = load_envs(job.envs_path)
        runner = WorkflowRunner(spec, envs=envs, on_event=writer)
        ctx = runner.run({"dataset": dataset})

        final: Dataset = ctx.get("dataset")
        saved = final.to_disk(Path(job.output_dir))
        logger.info("saved final dataset to %s (%d frames)", saved, len(final.images))

        writer.finish("done", message=f"complete — {len(final.images)} frames in {saved}")
        return 0
    except RunCancelled as exc:
        logger.warning("run cancelled: %s", exc)
        writer.finish("cancelled", message=str(exc))
        return 3
    except Exception as exc:
        logger.error("run failed: %s", exc)
        logger.debug("%s", traceback.format_exc())
        writer.finish(
            "failed", message=f"{type(exc).__name__}: {exc}", error=traceback.format_exc(),
        )
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to the RunJob JSON")
    parser.add_argument("--status", required=True, help="Path to publish RunState JSON to")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    job = RunJob.from_dict(json.loads(Path(args.job).read_text()))
    return _run(job, Path(args.status))


if __name__ == "__main__":
    sys.exit(main())
