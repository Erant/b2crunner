"""Workflow YAML schema and loader.

A workflow declares what a person can set and what they can ask for, and
wires the rest itself:

    name: fast_helical
    settings:                         # the knobs the UI draws, in order
      - name: resolution
        label: Resolution
        type: list
        default: [720, 1280]
        choices: [[720, 1280], [600, 1040]]
        help: Frame size, width x height.
    outputs:                          # the deliverables, and their switches
      - name: export_ply
        label: Trained .ply
        dir: ply
        default: true
    globals:                          # plumbing with no control of its own
      output_root: output/fast_helical
    steps:
      - id: denoise
        step: wan22_vace_denoise      # registered Step name
        dispatch: subprocess          # in_process | subprocess | service | docker
        env: wan22                    # key into envs.yaml, ignored for in_process
        inputs:
          control_video: dataset.images
          reference_image: dataset.reference_image
          style_hint: scene.style?    # optional; None when nothing wrote it
        params:                       # overrides on THIS step's own defaults
          steps: 6                    # this step's own knob, a literal
          width: ${globals.resolution.0}
        outputs:
          denoised: dataset.images    # written back into the shared Context
        when: ${globals.run_denoise}  # optional; skip the step when falsy

**One flat namespace, three ways to declare into it.** A `settings:` entry,
an `outputs:` entry and a bare `globals:` key all land in `spec.globals`,
which is the only scope `${...}` resolves against and the only thing
`--param x=y` and the UI's overrides address. A name declared twice is
refused at load: a setting has exactly one home.

The three differ only in what they tell the UI:

  * `settings:` is a declared knob — a `Param` (pipeline/step.py), with the
    same `type`/`default`/`help`/`choices`/`minimum`/`maximum`/`advanced`
    vocabulary a step param has, plus a `label:` and a `group:`. The UI draws
    these, through the same widget code it draws step params with. This is
    where `resolution` and `framing` live: what more than one step must agree
    on AND what somebody actually wants to change.
  * `outputs:` is a deliverable — a switch its export steps read through
    `when:`, plus the `dir:` it lands in under `output_root` (so a finished
    run can be packaged without a hardcoded list of subdirectory names) and
    an optional `requires:` naming a setting it is only meaningful with.
  * `globals:` is what has no control: `output_root`, which the CLI and the
    run worker repoint at the run's own directory.

Everything else belongs in the step that consumes it, under that step's own
`params:`, where it overrides the default the Step class declares (see
`Step.PARAMS` in pipeline/step.py). A step's params are therefore namespaced
by its `id:`, which is what lets one workflow call the same step twice and
configure the two calls apart — `fast_helical_native.yaml` trains `brush`
twice and denoises twice.

A declared setting reaches the steps the way it always has, as
`${globals.<name>}` at each step that reads it. That the reference is written
where it is consumed is the point — it is greppable from the step end, and
`templating.global_ref` uses it to drop the step-level duplicate from the
panel so the setting keeps one editable home. The risk that buys is a
setting nothing reads, i.e. a control that silently does nothing, and
`validate()` refuses that.

An input path ending in `?` is **optional**: if nothing has written it, the
step is handed `None` instead of the run failing. That exists for one shape
of wiring — a `when:`-gated branch feeding a step that runs either way, e.g.
the face splat's supporting views into `brush` — since a gated step's
outputs are simply not in the Context when it is switched off, and there is
otherwise no way to say "take these if they were built". Everything else
stays required, which is what makes a typo'd path a failure at that step
rather than a silently missing input.

Only a step whose Step subclass actually writes to disk (e.g. `save_dataset`)
touches disk — everything else stays in the in-memory Context between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .step import FALSE_STRINGS, Param, ParamError, coerce_param
from .templating import referenced_globals


#: The `type:` names a `settings:` entry may use, and the Python type each
#: one means. Deliberately the same six `Param` accepts: a pipeline setting
#: IS a `Param`, so it is drawn by the widget code the step panel already
#: uses and coerced by `coerce_param`, rather than by a second mechanism
#: that would have to be kept in step with the first.
PARAM_TYPES: Dict[str, type] = {
    "str": str, "int": int, "float": float, "bool": bool, "list": list,
    "dict": dict,
}

_SETTING_KEYS = frozenset({
    "name", "label", "type", "default", "help", "choices",
    "minimum", "maximum", "advanced", "group",
})

_OUTPUT_KEYS = frozenset({"name", "label", "dir", "default", "help", "requires"})


def setting_from_dict(data: Dict[str, Any]) -> Param:
    """One `settings:` entry as a `Param`.

    `type:` is a name from `PARAM_TYPES`; leave it out and it is inferred
    from the default's own Python type, which is what the UI used to guess
    for an undeclared global. A null default carries no type, so that is the
    one case where `type:` is mandatory.

    There is no `REQUIRED` here, unlike a step param: a pipeline setting is a
    control on a form, and a form control has a value in it before anybody
    touches it. A `settings:` entry with no `default` is a mistake, not a
    demand that the caller supply one.
    """
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError(f"settings: every entry needs a `name`; got {data!r}")
    name = data["name"]
    unknown = sorted(set(data) - _SETTING_KEYS)
    if unknown:
        raise ValueError(
            f"setting {name!r}: unknown key(s) {', '.join(unknown)}. "
            f"A setting accepts: {', '.join(sorted(_SETTING_KEYS))}."
        )
    if "default" not in data:
        raise ValueError(
            f"setting {name!r}: needs a `default`. A pipeline setting is a "
            "control that is already showing a value; there is no REQUIRED "
            "sentinel at this level."
        )
    default = data["default"]
    declared = data.get("type")
    if declared is None:
        if default is None:
            raise ValueError(
                f"setting {name!r}: a null default carries no type, so say "
                f"which it is — type: {' | '.join(PARAM_TYPES)}."
            )
        param_type = type(default)
        if param_type not in PARAM_TYPES.values():
            param_type = str
    elif declared not in PARAM_TYPES:
        raise ValueError(
            f"setting {name!r}: unknown type {declared!r}. "
            f"Use one of: {', '.join(PARAM_TYPES)}."
        )
    else:
        param_type = PARAM_TYPES[declared]
    return Param(
        name=name,
        type=param_type,
        default=default,
        help=data.get("help", ""),
        choices=tuple(data.get("choices") or ()),
        minimum=data.get("minimum"),
        maximum=data.get("maximum"),
        advanced=bool(data.get("advanced", False)),
        label=data.get("label", ""),
        group=data.get("group", ""),
    )


@dataclass
class Output:
    """One deliverable the workflow can be asked to produce.

    `name` is the workflow global its checkbox writes — the same name the
    `when:` on the steps that build it reads, so the switch is greppable
    from either end. `directory` (`dir:` in YAML) is where those steps land
    it under `output_root`, which is what lets the UI package a finished run
    without a hardcoded list of subdirectory names.

    `requires` names another setting this output is only meaningful with.
    The pre-upscale COLMAP export is the case: with the upscale off it would
    be byte-identical to the ordinary one, so its checkbox is disabled and
    the global forced false rather than quietly exporting something else.
    """

    name: str
    label: str = ""
    directory: str = ""
    default: bool = True
    help: str = ""
    requires: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Output":
        if not isinstance(data, dict) or "name" not in data:
            raise ValueError(f"outputs: every entry needs a `name`; got {data!r}")
        unknown = sorted(set(data) - _OUTPUT_KEYS)
        if unknown:
            raise ValueError(
                f"output {data['name']!r}: unknown key(s) {', '.join(unknown)}. "
                f"An output accepts: {', '.join(sorted(_OUTPUT_KEYS))}."
            )
        if not data.get("dir"):
            raise ValueError(
                f"output {data['name']!r}: needs a `dir` — the subdirectory "
                "under output_root its export steps write, which is how a "
                "finished run is packaged."
            )
        return cls(
            name=data["name"],
            label=data.get("label", ""),
            directory=data["dir"],
            default=bool(data.get("default", True)),
            help=data.get("help", ""),
            requires=data.get("requires", ""),
        )

    @property
    def title(self) -> str:
        return self.label or self.name

    def as_param(self) -> Param:
        """This switch as the `Param` the UI draws and coercion goes through."""
        return Param(
            name=self.name, type=bool, default=self.default,
            help=self.help, label=self.title,
        )


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
    # and its load() is expensive — fast_helical_native.yaml's two
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
    # (fast_helical_native.yaml's COLMAP dataset and trained .ply) needs the
    # caller to pick which ones to pay for, and a 30,000-iteration brush run
    # is not something to start and throw away.
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

    # What a person sees and sets, declared by the workflow itself.
    # `settings` are knobs, `outputs` are deliverables; both are ordered,
    # because the order is the order the form draws them in.
    settings: List[Param] = field(default_factory=list)
    outputs: List["Output"] = field(default_factory=list)

    # Every setting's and every output's current value lands in here, on top
    # of whatever the literal `globals:` block holds. That is the whole
    # integration: `${globals.x}`, `when:`, `--param x=y` and
    # `apply_ui_overrides` all keep reading one flat namespace and do not
    # need to know a value came from a declaration rather than a bare
    # mapping. `globals:` itself is left holding only what has no control —
    # `output_root`, which the CLI and the run worker repoint.
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
        settings = [setting_from_dict(s) for s in data.get("settings") or ()]
        outputs = [Output.from_dict(o) for o in data.get("outputs") or ()]

        # Three passes onto one namespace, in declaration order, refusing a
        # name that appears twice. A setting with two homes is the failure
        # this guards: two controls for one value, disagreeing the moment
        # somebody touches either.
        merged: Dict[str, Any] = {}
        for source, entries in (
            ("settings", [(s.name, s.default) for s in settings]),
            ("outputs", [(o.name, o.default) for o in outputs]),
            ("globals", list((data.get("globals") or {}).items())),
        ):
            for key, value in entries:
                if key in merged:
                    raise ValueError(
                        f"{path}: {source} declares {key!r}, which is already "
                        "declared earlier in the file. A setting has exactly "
                        "one home."
                    )
                merged[key] = value

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            settings=settings,
            outputs=outputs,
            globals=merged,
            steps=[StepSpec.from_dict(s) for s in data["steps"]],
        )

    def declared_globals(self) -> Dict[str, Param]:
        """Every declared knob, keyed by name — settings and output switches.

        The UI draws from this and `coerce_global` types through it. A bare
        `globals:` entry is deliberately absent: it has no declaration, which
        is exactly why it gets no control.
        """
        declared = {setting.name: setting for setting in self.settings}
        declared.update({output.name: output.as_param() for output in self.outputs})
        return declared

    def coerce_global(self, name: str, value: Any) -> Any:
        """`value` brought to the type `name` is declared with, if it is.

        Gives an override naming a setting the same lenient coercion a step
        param already gets, so `--param run_upscale=false` and a text box
        holding `"720"` mean what they look like. An undeclared global is
        passed through untouched: there is no type to bring it to.
        """
        param = self.declared_globals().get(name)
        if param is None:
            return value
        return coerce_param(value, param, f"workflow '{self.name}'")

    def output_dirs(self) -> Dict[str, str]:
        """{global name -> subdirectory under output_root} for every output."""
        return {output.name: output.directory for output in self.outputs}

    def apply_output_requirements(self) -> Dict[str, bool]:
        """Force off every output whose `requires:` setting is off, in place.

        Returns the resulting {output name -> bool}, which the web UI folds
        back into the overrides it sends the worker.

        This lives on the spec rather than in the UI so that a `--param`
        reaches the same answer a checkbox does: `requires:` says the two
        exports would be the same frames under different names, and that is
        true however the switch got set.
        """
        resolved: Dict[str, bool] = {}
        for output in self.outputs:
            wanted = truthy(self.globals.get(output.name, output.default))
            if output.requires and not truthy(self.globals.get(output.requires)):
                wanted = False
            resolved[output.name] = wanted
        self.globals.update(resolved)
        return resolved

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

        self._validate_declarations()

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

    def _validate_declarations(self) -> None:
        """Check the `settings:` and `outputs:` blocks against the steps.

        Same bargain as the step-param check above: every one of these is a
        mistake that otherwise shows up as a control that silently does
        nothing, which is worse than a control that isn't there — it looks
        like it worked.
        """
        for param in self.settings:
            try:
                value = coerce_param(param.default, param, f"workflow '{self.name}'")
            except ParamError as exc:
                raise ValueError(
                    f"Workflow '{self.name}' setting '{param.name}': its own "
                    f"default does not fit its declared type. {exc}"
                ) from None
            if param.choices and not _among(value, param.choices):
                raise ValueError(
                    f"Workflow '{self.name}' setting '{param.name}': default "
                    f"{param.default!r} is not one of its choices "
                    f"{list(param.choices)!r}."
                )

        declared = set(self.declared_globals())
        for output in self.outputs:
            if output.requires and output.requires not in declared:
                raise ValueError(
                    f"Workflow '{self.name}' output '{output.name}' requires "
                    f"'{output.requires}', which it does not declare. "
                    f"It declares: {', '.join(sorted(declared))}."
                )

        # A setting nothing reads is a dead control: it draws, it records an
        # override, and the run ignores it. Cheap to catch, and the one new
        # way this layer can go wrong that the old bare `globals:` could not
        # (there, an unread global simply had no control either).
        read: set = set()
        for step in self.steps:
            read |= referenced_globals(step.params)
            read |= referenced_globals(step.when)
        read |= {output.requires for output in self.outputs if output.requires}
        orphans = sorted(
            param.name for param in self.settings if param.name not in read
        )
        if orphans:
            raise ValueError(
                f"Workflow '{self.name}' declares setting(s) nothing reads: "
                f"{', '.join(orphans)}. A setting reaches a run through a step's "
                "`${globals.<name>}`, a `when:`, or an output's `requires:` — "
                "wire it up or drop it."
            )


def _among(value: Any, choices: Any) -> bool:
    """`value in choices`, tolerating a list value against a tuple choice."""
    for choice in choices:
        if choice == value:
            return True
        if isinstance(choice, (list, tuple)) and isinstance(value, (list, tuple)):
            if list(choice) == list(value):
                return True
    return False


def apply_ui_overrides(
    spec: WorkflowSpec,
    global_overrides: Dict[str, Any],
    step_overrides: Dict[str, Dict[str, Any]],
) -> None:
    """Apply a UI submission's overrides to `spec`, dropping anything stale.

    Deliberately not `pipeline.cli.apply_param_overrides`'s strict version,
    which raises on an unknown key: a `--param` typo should fail loudly, but
    a UI submission's param panel is drawn from a specific workflow and can
    go stale the moment the user switches the workflow dropdown and submits
    before the redraw lands. Dropping a key that is no longer part of this
    workflow beats failing the run over a control that is no longer on
    screen, or worse, inventing a global nothing reads.

    Used both by the web UI (to validate before a run is even queued) and by
    `pipeline.run_worker` (to apply the same overrides again when it loads
    its own copy of the spec) — the two must agree on what an override means.

    A value naming a declared setting or output is coerced to its type on
    the way in, the same way a step param's is: a checkbox and a text box
    both hand back strings, and `bool("false")` is True.
    """
    spec.globals.update({
        key: spec.coerce_global(key, value)
        for key, value in global_overrides.items()
        if key in spec.globals
    })
    by_id = {step.id: step for step in spec.steps}
    for step_id, values in step_overrides.items():
        if step_id in by_id:
            by_id[step_id].params.update(values)


# A `when:` that resolves to a string is almost always a `${globals.x}`
# pointing at a value someone typed into a text box, so "false" has to mean
# false — bool("false") is True, and silently running a step the caller
# switched off is the one failure mode this whole mechanism exists to
# prevent. Everything else goes through plain truthiness. `FALSE_STRINGS`
# lives in step.py, which needs the identical rule to coerce a `bool` param.
def truthy(value: Any) -> bool:
    """Workflow truthiness, i.e. `when:`'s: the string "false" is False.

    A `when:` or a switch usually resolves through a value somebody typed,
    and `bool("false")` is True. Shared so the UI's disabled-checkbox rule
    and the runner's skip-this-step rule cannot drift apart.
    """
    if isinstance(value, str):
        return value.strip().lower() not in FALSE_STRINGS
    return bool(value)


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
    return truthy(value)


def load_envs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Loads the per-machine environment registry (python_bin/image/base_url
    per env name). Returns {} if the file doesn't exist yet — fine for
    workflows that only use in_process steps."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data.get("envs", {})
