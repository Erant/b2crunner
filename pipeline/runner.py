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

import ctypes
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .context import Context
from .dispatch import Dispatcher, build_dispatcher
from .registry import get_step_class
from .templating import resolve
from .workflow import StepSpec, WorkflowSpec, step_enabled

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

    kind: str  # workflow_start | step_start | step_end | step_error |
               # step_skipped | workflow_end
    workflow: str
    index: int = 0          # 1-based position of the step, 0 for workflow events
    total: int = 0
    step_id: str = ""
    step_name: str = ""
    elapsed: float = 0.0
    error: str = ""

    # The live Context, on step_end only. A reference, not a copy — an
    # observer must read what it needs and not hold on to it.
    #
    # It is here so the web UI can snapshot a few frames after every step
    # without the runner knowing anything about previews, and without the
    # workflow needing a save_dataset checkpoint between every pair of
    # steps (which would defeat the in-memory design outright). An observer
    # that ignores it costs nothing.
    context: Optional[Any] = None


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


def host_memory_summary() -> str:
    """'RAM x.xx GB (peak y.yy)' for this process, or '' off Linux.

    The host-side twin of `gpu_memory_summary`. On 2026-09-29 two local
    runs were OOM-killed at the final training with the run worker itself
    at 23.8 GB resident, and nothing in the log said so.
    """
    try:
        with open("/proc/self/status") as status:
            fields = dict(line.split(":", 1) for line in status if ":" in line)
        rss = int(fields["VmRSS"].split()[0]) * 1024
        peak = int(fields["VmHWM"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        return ""
    return f"RAM {rss / 1e9:.2f} GB (peak {peak / 1e9:.2f})"


def _payload_bytes(value: Any, depth: int = 0) -> int:
    """Array bytes held by a context entry: ndarrays and tensors, through
    lists, tuples, dicts and plain objects. An estimate for the log, not
    an accounting — shared buffers count once per reference."""
    if depth > 6:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if hasattr(value, "element_size") and hasattr(value, "nelement"):
        try:
            return int(value.element_size() * value.nelement())
        except Exception:
            return 0
    if isinstance(value, dict):
        return sum(_payload_bytes(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_bytes(v, depth + 1) for v in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return sum(_payload_bytes(v, depth + 1) for v in vars(value).values())
    return 0


def _trim_heap() -> None:
    """Hand freed heap back to the OS.

    Frames of a few MB sit under glibc's dynamic mmap threshold once it has
    risen, so they are carved out of the heap, and freeing them leaves the
    pages resident until something trims them. Harmless where there is no
    glibc.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _related(a: str, b: str) -> bool:
    """Whether reading one context path sees the other: the same path, or
    one inside the other (`mesh_output: scene` reads every `scene.*`)."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


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

    def run(
        self, initial_context: Dict[str, Any], keep: Optional[Iterable[str]] = None,
    ) -> Context:
        """Run every enabled step in order and return the context.

        `keep` names the context paths the caller reads once the run is
        over. Given, the runner releases everything else as soon as no
        later enabled step reads it — without it, every intermediate a
        workflow writes stays in the worker's RAM to the end, and helical's
        run worker was OOM-killed at 23.8 GB resident when the final
        training started. None keeps the whole context, for callers (and
        tests) that inspect a step's outputs afterwards.
        """
        ctx = Context(initial_context)
        template_scope = {"globals": self.spec.globals}
        total = len(self.spec.steps)
        started = time.time()

        # Both checked once, up front, for the same reason: a `when:` naming
        # a global that doesn't exist, or a step override naming a param the
        # step doesn't declare, should stop the run here — not forty minutes
        # in when the step it belongs to is finally reached.
        self.spec.validate()
        enabled = [step_enabled(step_spec, self.spec.globals) for step_spec in self.spec.steps]

        logger.info("=" * 72)
        logger.info(
            "workflow '%s': %d steps%s",
            self.spec.name, total,
            f" ({enabled.count(False)} skipped by `when:`)" if not all(enabled) else "",
        )
        for index, step_spec in enumerate(self.spec.steps, start=1):
            where = f"{step_spec.dispatch}" + (f":{step_spec.env}" if step_spec.env else "")
            logger.info(
                "  %2d. %-24s %-22s [%s]%s",
                index, step_spec.id, step_spec.step, where,
                "" if enabled[index - 1] else "  SKIPPED",
            )
        logger.info("=" * 72)
        self._emit(RunEvent(kind="workflow_start", workflow=self.spec.name, total=total))

        # The last enabled step on each dispatcher. A resident worker used to
        # live until the end of the run, so helical's ~10 GB Wan worker sat
        # idle in host RAM through the steps after the last denoise and the
        # upscale's worker was OOM-killed beside it on a 32 GB box.
        last_use = {
            self._dispatcher_key(step_spec): index
            for index, step_spec in enumerate(self.spec.steps, start=1)
            if enabled[index - 1]
        }

        # Every input path of every enabled step, with the step's index, for
        # the release below. Optional reads count: the value is there to be
        # read when the branch that writes it ran.
        reads: List[Tuple[int, str]] = [
            (index, path.rstrip("?"))
            for index, step_spec in enumerate(self.spec.steps, start=1)
            if enabled[index - 1]
            for path in step_spec.inputs.values()
        ]
        keep_paths = None if keep is None else list(keep)
        written: List[str] = []

        try:
            for index, step_spec in enumerate(self.spec.steps, start=1):
                if not enabled[index - 1]:
                    logger.info(
                        "--- [%d/%d] %s skipped (when: %r) ------------------",
                        index, total, step_spec.id, step_spec.when,
                    )
                    self._emit(
                        RunEvent(
                            kind="step_skipped", workflow=self.spec.name, index=index,
                            total=total, step_id=step_spec.id, step_name=step_spec.step,
                        )
                    )
                    continue
                self._run_one(step_spec, index, total, ctx, template_scope)
                if keep_paths is not None:
                    for path in step_spec.outputs.values():
                        if path not in written:
                            written.append(path)
                    self._release(ctx, index, written, reads, keep_paths)
                key = self._dispatcher_key(step_spec)
                if last_use[key] == index:
                    self._close_dispatcher(key)
        finally:
            for dispatcher in self._dispatchers.values():
                dispatcher.close()

        elapsed = time.time() - started
        logger.info("workflow '%s' complete in %s", self.spec.name, _duration(elapsed))
        self._emit(
            RunEvent(kind="workflow_end", workflow=self.spec.name, total=total, elapsed=elapsed)
        )
        return ctx

    @staticmethod
    def _release(
        ctx: Context,
        index: int,
        written: List[str],
        reads: List[Tuple[int, str]],
        keep: List[str],
    ) -> None:
        """Drop every path a step wrote that no step after `index` reads.

        Only step outputs are candidates, and never one inside a `keep`
        path (`dataset.splat_path` is the caller's). A path read later —
        itself, a namespace holding it, or something inside it — stays.
        """
        released = []
        for path in list(written):
            if any(_related(path, kept) for kept in keep):
                continue
            if any(later > index and _related(path, read) for later, read in reads):
                continue
            written.remove(path)
            try:
                value = ctx.get(path)
            except (KeyError, AttributeError, IndexError, TypeError):
                continue
            size = _payload_bytes(value)
            del value
            if ctx.delete(path):
                released.append((size, path))
        if not released:
            return
        gc.collect()
        _trim_heap()
        released.sort(reverse=True)
        total = sum(size for size, _ in released)
        largest = ", ".join(f"{path} {size / 1e9:.2f}" for size, path in released[:4] if size)
        logger.info(
            "released %d context entr%s no later step reads, %.2f GB of arrays%s | %s",
            len(released), "y" if len(released) == 1 else "ies", total / 1e9,
            f" (largest: {largest})" if largest else "", host_memory_summary(),
        )

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
        memory = " | ".join(part for part in (gpu_memory_summary(), host_memory_summary()) if part)
        logger.info(
            "[%d/%d] %s done in %s%s",
            index, total, step_spec.id, _duration(elapsed), f" | {memory}" if memory else "",
        )
        self._emit(
            RunEvent(
                kind="step_end", workflow=self.spec.name, index=index, total=total,
                step_id=step_spec.id, step_name=step_spec.step, elapsed=elapsed,
                context=ctx,
            )
        )

    def _run_step(self, step_spec: StepSpec, ctx: Context, template_scope: Dict[str, Any]) -> None:
        dispatcher = self._get_dispatcher(step_spec)

        inputs = {}
        for name, path in step_spec.inputs.items():
            # A trailing `?` marks an OPTIONAL read: absent means None
            # rather than a failed run. That is the only way to wire a
            # `when:`-gated branch into a step downstream of it — the
            # branch's outputs simply do not exist when it is switched
            # off, and the consumer is a step that runs either way. The
            # shipped case is the face splat's supporting views, which
            # `train_splat` takes when the face branch built them and
            # trains without when `face_splat: false` turns the branch
            # off. Steps that accept one must treat None as "not given",
            # which is what every optional input already means to them.
            optional = path.endswith("?")
            path = path[:-1] if optional else path
            try:
                inputs[name] = ctx.get(path)
            except (KeyError, AttributeError, IndexError, TypeError) as exc:
                if optional:
                    inputs[name] = None
                    continue
                # Naming the step and the path beats a bare KeyError from
                # three frames down: an unresolvable input almost always
                # means an earlier step didn't write where this one reads,
                # and the two names together identify the wiring bug.
                raise KeyError(
                    f"Step '{step_spec.id}' ({step_spec.step}) input '{name}' reads "
                    f"context path '{path}', which isn't available: {exc}. "
                    f"Context currently holds: {sorted(ctx.as_dict())}"
                ) from exc

        # Two stages, in this order: expand `${globals.x}` in whatever the
        # workflow overrode, then merge that onto the step's own declared
        # defaults. The merged dict is what crosses the dispatcher boundary,
        # so every dispatch mode and both worker modes see a complete param
        # set — including `worker.load_signature`, which now compares real
        # defaults instead of None for anything the workflow left out.
        overrides = resolve(step_spec.params, template_scope)
        params = get_step_class(step_spec.step).resolve_params(overrides)
        outputs = dispatcher.run(step_spec.step, inputs, params)

        for name, path in step_spec.outputs.items():
            if name not in outputs:
                raise KeyError(
                    f"Step '{step_spec.id}' ({step_spec.step}) did not return output '{name}'; "
                    f"it returned: {sorted(outputs)}"
                )
            ctx.set(path, outputs[name])

    def _get_dispatcher(self, step_spec: StepSpec) -> Dispatcher:
        # keep_loaded is part of the key, not just an argument to the first
        # build: two steps sharing an env is exactly how residency happens
        # (helical's denoise_pass1 and denoise_pass2 are both
        # subprocess/wan22, so they share one dispatcher and therefore one
        # resident worker). Without keep_loaded in the key, a third step on
        # the same env that did *not* ask for residency would inherit — or
        # deny — it purely by declaration order, which is invisible in the
        # YAML and would show up as a mystery 47 GB reload.
        key = self._dispatcher_key(step_spec)
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

    @staticmethod
    def _dispatcher_key(step_spec: StepSpec) -> tuple:
        return (step_spec.dispatch, step_spec.env, step_spec.keep_loaded)

    def _close_dispatcher(self, key: tuple) -> None:
        dispatcher = self._dispatchers.pop(key, None)
        if dispatcher is None:
            return
        logger.info("no later step uses %s; closing its dispatcher", ":".join(str(k) for k in key[:2] if k))
        try:
            dispatcher.close()
        except Exception:  # a finished dispatcher must not fail the run
            logger.exception("closing the %s dispatcher failed; continuing", key)


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"
