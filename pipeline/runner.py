"""Executes a WorkflowSpec against an initial Context.

This is the direct replacement for submit.py's queue_prompt/wait_for_completion
loop: instead of building a ComfyUI API-format graph and polling a server, it
walks the YAML step list and calls each step through its resolved Dispatcher.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .context import Context
from .dispatch import Dispatcher, build_dispatcher
from .templating import resolve
from .workflow import StepSpec, WorkflowSpec

logger = logging.getLogger(__name__)


class WorkflowRunner:
    def __init__(self, spec: WorkflowSpec, envs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.spec = spec
        self.envs = envs or {}
        self._dispatchers: Dict[tuple, Dispatcher] = {}

    def run(self, initial_context: Dict[str, Any]) -> Context:
        ctx = Context(initial_context)
        template_scope = {"params": self.spec.params}

        try:
            for step_spec in self.spec.steps:
                logger.info("[%s] running step '%s' (%s)", self.spec.name, step_spec.id, step_spec.step)
                self._run_step(step_spec, ctx, template_scope)
        finally:
            for dispatcher in self._dispatchers.values():
                dispatcher.close()

        return ctx

    def _run_step(self, step_spec: StepSpec, ctx: Context, template_scope: Dict[str, Any]) -> None:
        dispatcher = self._get_dispatcher(step_spec)

        inputs = {name: ctx.get(path) for name, path in step_spec.inputs.items()}
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
            self._dispatchers[key] = build_dispatcher(
                step_spec.dispatch, env_config, keep_loaded=step_spec.keep_loaded
            )
        return self._dispatchers[key]
