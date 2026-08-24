"""Executes a WorkflowSpec against an initial Context.

This is the direct replacement for submit.py's queue_prompt/wait_for_completion
loop: instead of building a ComfyUI API-format graph and polling a server, it
walks the YAML step list and calls each step through its resolved Dispatcher.

Beyond running the steps, this is the one place that knows how far along a
run is, so it is also where progress reporting lives: an `on_event` callback
receives a `RunEvent` at each boundary. The CLI ignores it (the log lines are
enough there); the web UI uses it to drive a progress bar without parsing
log text. Nothing about a step or a dispatcher changes to support this —
they still just return outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .context import Context
from .dispatch import Dispatcher, build_dispatcher
from .templating import resolve
from .workflow import StepSpec, WorkflowSpec

logger = logging.getLogger(__name__)


class RunCancelled(Exception):
    """Raised by an `on_event` observer to stop a run at the next boundary.

    The only exception `_emit` lets through. Everything else an observer
    raises is its own problem — a UI that fails to update must not take
    down a two-hour workflow — but a deliberate stop request has to
    propagate, so it gets its own type rather than relying on the observer
    picking an exception the runner happens not to swallow.
    """


@dataclass
class RunEvent:
    """One boundary in a run. `kind` is the only field always meaningful."""

    kind: str  # workflow_start | step_start | step_end | step_error | workflow_end
    workflow: str
    index: int = 0          # 1-based position of the step, 0 for workflow events
    total: int = 0
    step_id: str = ""
    step_name: str = ""
    elapsed: float = 0.0
    error: str = ""


EventCallback = Callable[[RunEvent], None]


def gpu_memory_summary() -> str:
    """'allocated/reserved/total GB' for cuda:0, or '' if there's no GPU.

    Cheap enough to call after every step and worth having in the log: the
    two bugs the first full local run turned up were both memory-shaped,
    and the failure ('tried to allocate 4.53 GB') tells you nothing about
    which earlier step was still holding the card.
    """
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    try:
        free, total = torch.cuda.mem_get_info()
        return (
            f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f} allocated / "
            f"{torch.cuda.memory_reserved() / 1e9:.2f} reserved / "
            f"{(total - free) / 1e9:.2f} used of {total / 1e9:.2f} GB"
        )
    except Exception:  # driver hiccup must never take down a run
        return ""


class WorkflowRunner:
    def __init__(
        self,
        spec: WorkflowSpec,
        envs: Optional[Dict[str, Dict[str, Any]]] = None,
        on_event: Optional[EventCallback] = None,
    ):
        self.spec = spec
        self.envs = envs or {}
        self.on_event = on_event
        self._dispatchers: Dict[tuple, Dispatcher] = {}

    def _emit(self, event: RunEvent) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except RunCancelled:
            raise
        except Exception:  # a broken observer must not fail the run
            logger.exception("on_event callback raised; continuing")

    def run(self, initial_context: Dict[str, Any]) -> Context:
        ctx = Context(initial_context)
        template_scope = {"params": self.spec.params}
        total = len(self.spec.steps)
        started = time.time()

        logger.info("=" * 72)
        logger.info("workflow '%s': %d steps", self.spec.name, total)
        for index, step_spec in enumerate(self.spec.steps, start=1):
            where = f"{step_spec.dispatch}" + (f":{step_spec.env}" if step_spec.env else "")
            logger.info("  %2d. %-24s %-22s [%s]", index, step_spec.id, step_spec.step, where)
        logger.info("=" * 72)
        self._emit(RunEvent(kind="workflow_start", workflow=self.spec.name, total=total))

        try:
            for index, step_spec in enumerate(self.spec.steps, start=1):
                self._run_one(step_spec, index, total, ctx, template_scope)
        finally:
            for dispatcher in self._dispatchers.values():
                dispatcher.close()

        elapsed = time.time() - started
        logger.info("workflow '%s' complete in %s", self.spec.name, _duration(elapsed))
        self._emit(
            RunEvent(kind="workflow_end", workflow=self.spec.name, total=total, elapsed=elapsed)
        )
        return ctx

    def _run_one(
        self,
        step_spec: StepSpec,
        index: int,
        total: int,
        ctx: Context,
        template_scope: Dict[str, Any],
    ) -> None:
        logger.info(
            "--- [%d/%d] %s (%s) ---------------------------------------",
            index, total, step_spec.id, step_spec.step,
        )
        self._emit(
            RunEvent(
                kind="step_start", workflow=self.spec.name, index=index, total=total,
                step_id=step_spec.id, step_name=step_spec.step,
            )
        )

        started = time.time()
        try:
            self._run_step(step_spec, ctx, template_scope)
        except Exception as exc:
            elapsed = time.time() - started
            logger.error(
                "[%d/%d] %s FAILED after %s: %s",
                index, total, step_spec.id, _duration(elapsed), exc,
            )
            self._emit(
                RunEvent(
                    kind="step_error", workflow=self.spec.name, index=index, total=total,
                    step_id=step_spec.id, step_name=step_spec.step,
                    elapsed=elapsed, error=str(exc),
                )
            )
            raise

        elapsed = time.time() - started
        memory = gpu_memory_summary()
        logger.info(
            "[%d/%d] %s done in %s%s",
            index, total, step_spec.id, _duration(elapsed), f" | {memory}" if memory else "",
        )
        self._emit(
            RunEvent(
                kind="step_end", workflow=self.spec.name, index=index, total=total,
                step_id=step_spec.id, step_name=step_spec.step, elapsed=elapsed,
            )
        )

    def _run_step(self, step_spec: StepSpec, ctx: Context, template_scope: Dict[str, Any]) -> None:
        dispatcher = self._get_dispatcher(step_spec)

        inputs = {}
        for name, path in step_spec.inputs.items():
            try:
                inputs[name] = ctx.get(path)
            except (KeyError, AttributeError, IndexError, TypeError) as exc:
                # Naming the step and the path beats a bare KeyError from
                # three frames down: an unresolvable input almost always
                # means an earlier step didn't write where this one reads,
                # and the two names together identify the wiring bug.
                raise KeyError(
                    f"Step '{step_spec.id}' ({step_spec.step}) input '{name}' reads "
                    f"context path '{path}', which isn't available: {exc}. "
                    f"Context currently holds: {sorted(ctx.as_dict())}"
                ) from exc

        params = resolve(step_spec.params, template_scope)
        outputs = dispatcher.run(step_spec.step, inputs, params)

        for name, path in step_spec.outputs.items():
            if name not in outputs:
                raise KeyError(
                    f"Step '{step_spec.id}' ({step_spec.step}) did not return output '{name}'; "
                    f"it returned: {sorted(outputs)}"
                )
            ctx.set(path, outputs[name])

    def _get_dispatcher(self, step_spec: StepSpec) -> Dispatcher:
        key = (step_spec.dispatch, step_spec.env)
        if key not in self._dispatchers:
            env_config = self.envs.get(step_spec.env, {}) if step_spec.env else {}
            if step_spec.env and not env_config and step_spec.dispatch != "in_process":
                # Silently building a dispatcher with no config produces a
                # confusing failure inside the dispatcher instead of here,
                # where the actual mistake (an envs.yaml that doesn't
                # describe this machine) is visible.
                raise ValueError(
                    f"Step '{step_spec.id}' dispatches to env '{step_spec.env}', which the "
                    f"envs registry doesn't define. Known envs: {sorted(self.envs) or 'none'}. "
                    f"Point --envs at the right registry for this machine "
                    f"(docker/envs.docker.yaml inside the image)."
                )
            self._dispatchers[key] = build_dispatcher(
                step_spec.dispatch, env_config, keep_loaded=step_spec.keep_loaded
            )
        return self._dispatchers[key]


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"
