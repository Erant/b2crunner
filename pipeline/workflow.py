"""Workflow YAML schema and loader.

Two parameter namespaces, and the split is the point:

    name: fast_helical
    globals:                          # affects the whole flow
      resolution: [720, 1280]
      diffusion_steps: 6
    steps:
      - id: denoise
        step: wan22_vace_denoise      # registered Step name
        dispatch: subprocess          # in_process | subprocess | service | docker
        env: wan22                    # key into envs.yaml, ignored for in_process
        inputs:
          control_video: dataset.images
          reference_image: dataset.reference_image
        params:                       # overrides on THIS step's own defaults
          steps: ${globals.diffusion_steps}
          width: ${globals.resolution.0}
        outputs:
          denoised: dataset.images    # written back into the shared Context
        when: ${globals.run_denoise}  # optional; skip the step when falsy

`globals:` is for what two or more steps must agree on — the frame size, the
seed, the prompts, the output root — and is the only thing `${...}` resolves
against. Everything else belongs in the step that consumes it, under that
step's own `params:`, where it overrides the default the Step class declares
(see `Step.PARAMS` in pipeline/step.py). A step's params are therefore
namespaced by its `id:`, which is what lets one workflow call the same step
twice and configure the two calls apart — `fast_helical_full.yaml` trains
`brush` twice and denoises twice.

Only a step whose Step subclass actually writes to disk (e.g. `save_dataset`)
touches disk — everything else stays in the in-memory Context between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .step import FALSE_STRINGS


@dataclass
class StepSpec:
    id: str
    step: str
    dispatch: str = "in_process"
    env: Optional[str] = None

    # Reuse one loaded Step across every call to this step in the run,
    # instead of building and loading it again each time. Off by default:
    # most steps are called once, and for those it only costs a model
    # sitting in memory for the rest of the run.
    #
    # Turn it on when the same `step:` appears more than once in a workflow
    # and its load() is expensive — fast_helical_full.yaml's two
    # `wan22_vace_denoise` passes are the case this exists for, where a
    # reload is ~47 GB re-read off a pod's network volume.
    #
    # Honoured by `dispatch: in_process` (one Step instance, see
    # dispatch/in_process.py) and `dispatch: subprocess` (one long-lived
    # `pipeline.worker --serve` child, see dispatch/subprocess_python.py).
    # `service` needs nothing — a model server is already warm — and
    # `docker` ignores it.
    #
    # The reused instance is NOT reloaded when params change, unless the
    # Step class declares `LOAD_PARAMS` naming the params its load() reads;
    # see `pipeline.worker.load_signature` for the full rule and why the
    # default is what it is. Two steps that set this and share a
    # `dispatch:`/`env:` pair share the loaded model; a step that doesn't
    # set it gets its own dispatcher either way (runner.py's
    # _get_dispatcher).
    keep_loaded: bool = False

    # Run this step only if this resolves truthy. Anything `params:` accepts
    # works — a literal `false`, or a `${globals.x}` reference, which is the
    # case it exists for: a workflow that ends in several optional exports
    # (fast_helical.yaml's COLMAP dataset and trained .ply) needs the caller
    # to pick which ones to pay for, and a 30,000-iteration brush run is not
    # something to start and throw away.
    #
    # A skipped step still occupies its slot in the run — the runner reports
    # it as skipped rather than renumbering around it, so a step list in the
    # UI matches the YAML.
    when: Any = True

    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)

    # Overrides on the Step class's own declared defaults, NOT the complete
    # param set — the runner merges these onto `Step.PARAMS` before dispatch.
    # A param nobody overrides never has to appear here at all.
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepSpec":
        return cls(
            id=data["id"],
            step=data["step"],
            dispatch=data.get("dispatch", "in_process"),
            env=data.get("env"),
            keep_loaded=data.get("keep_loaded", False),
            when=data.get("when", True),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            params=data.get("params", {}),
        )


@dataclass
class WorkflowSpec:
    name: str
    description: str = ""
    globals: Dict[str, Any] = field(default_factory=dict)
    steps: List[StepSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowSpec":
        data = yaml.safe_load(Path(path).read_text())
        if "params" in data:
            # The pre-namespacing shape. Refuse it by name rather than
            # ignoring it: a workflow-level `params:` block used to be where
            # every knob lived, so silently dropping it would run the whole
            # pipeline at the step defaults and look like it had worked.
            raise ValueError(
                f"{path}: top-level `params:` is no longer a workflow key. "
                "Move flow-wide values to `globals:` and per-step values "
                "into that step's own `params:` block, which now holds "
                "overrides on the step's declared defaults."
            )
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            globals=data.get("globals", {}),
            steps=[StepSpec.from_dict(s) for s in data["steps"]],
        )

    def enabled_steps(self) -> List[StepSpec]:
        """The steps this spec's current globals actually select.

        The runner skips the rest as it walks the list; this is for the
        callers that need to know *before* the run starts — chiefly model
        prefetching, which must not block on SeedVR2's 6 GB for a workflow
        whose upscale is switched off.
        """
        return [step for step in self.steps if step_enabled(step, self.globals)]

    def validate(self) -> None:
        """Check every step's overrides against what that step declares.

        Cheap, and it turns the two mistakes that are otherwise invisible
        until runtime — a step name that isn't registered, and a param name
        that no longer exists on the step — into a failure at second zero
        rather than forty minutes into a pod run. `WorkflowRunner.run` calls
        this before the first step, for exactly that reason.
        """
        from .registry import get_step_class

        for step in self.steps:
            step_class = get_step_class(step.step)
            declared = step_class.declared_params()
            if not declared:
                continue
            unknown = sorted(set(step.params) - set(declared))
            if unknown:
                raise ValueError(
                    f"Workflow '{self.name}' step '{step.id}' ({step.step}) sets params "
                    f"that step does not declare: {', '.join(unknown)}. "
                    f"It accepts: {', '.join(declared)}"
                )


# A `when:` that resolves to a string is almost always a `${globals.x}`
# pointing at a value someone typed into a text box, so "false" has to mean
# false — bool("false") is True, and silently running a step the caller
# switched off is the one failure mode this whole mechanism exists to
# prevent. Everything else goes through plain truthiness. `FALSE_STRINGS`
# lives in step.py, which needs the identical rule to coerce a `bool` param.
def step_enabled(step: StepSpec, workflow_globals: Dict[str, Any]) -> bool:
    """Whether `step`'s `when:` selects it, given a workflow's globals."""
    from .templating import resolve

    try:
        value = resolve(step.when, {"globals": workflow_globals})
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        raise KeyError(
            f"Step '{step.id}' has a `when:` of {step.when!r}, which does not "
            f"resolve: {exc}. Workflow globals: {sorted(workflow_globals)}"
        ) from exc
    if isinstance(value, str):
        return value.strip().lower() not in FALSE_STRINGS
    return bool(value)


def load_envs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Loads the per-machine environment registry (python_bin/image/base_url
    per env name). Returns {} if the file doesn't exist yet — fine for
    workflows that only use in_process steps."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data.get("envs", {})
