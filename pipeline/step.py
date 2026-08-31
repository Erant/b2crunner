"""Step interface — the unit of work a workflow YAML wires together.

A Step knows how to do one thing (denoise a video, detect landmarks, export
COLMAP). It does NOT know or care whether it's about to be called directly in
this process, spawned in an isolated venv's subprocess, or hit over HTTP on a
warm model server — that's entirely the Dispatcher's problem. This split is
what makes "run this step in isolation instead" a one-line YAML edit instead
of a code change.

Subclass this, decorate with @register_step("name"), and reference "name"
from a workflow YAML's `step:` field.

**A step declares the params it accepts**, as a `PARAMS` tuple of `Param`
(below). That declaration is the single source of truth for the defaults:
`WorkflowRunner` merges a workflow's overrides onto them before dispatch, so
a `run()` body reads `params["filter_size"]` and never
`params.get("filter_size", 6)`. Keeping the defaults in one declarative place
is what lets the web UI enumerate a step's knobs and render a control per
param — with the source of a value, and its help text, still visible.

Stdlib only in this module, deliberately: `pipeline.worker` imports it inside
every isolated venv (see that module's docstring), most of which have no
numpy, no torch, and nothing else from this repo's dependency set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


class _Required:
    """Sentinel for a param the workflow must supply; there is no default."""

    _instance: Optional["_Required"] = None

    def __new__(cls) -> "_Required":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REQUIRED"

    def __bool__(self) -> bool:
        return False


REQUIRED = _Required()


# A param value that arrives as a string and is declared `bool` almost always
# came from a text box or a `--param` on the command line, so "false" has to
# mean false — `bool("false")` is True. Shared with `when:` handling in
# pipeline/workflow.py, which needs the identical rule for the identical
# reason.
FALSE_STRINGS = frozenset({"", "0", "false", "no", "off", "none", "null"})


@dataclass(frozen=True)
class Param:
    """One knob a Step accepts, with everything a UI needs to draw it.

    `type` is a plain Python type — `int`, `float`, `bool`, `str` or `list` —
    used both to coerce whatever a workflow or a text box supplied and to
    pick a widget. `default` is what the step uses when nothing overrides it;
    `REQUIRED` means there is no sensible default and a workflow that omits
    it is an error (an output directory, say). `None` is a legitimate default
    and means something different: "the step computes this at runtime" —
    `device`, which resolves to cuda-if-available inside `run()`, is the
    shape that has.

    `advanced` keeps a param out of the UI's main list and behind a fold. The
    rule for setting it: a knob that exists because the underlying library
    has one, rather than because this pipeline tunes it, is advanced.

    `label` and `group` are for the other producer of `Param`s: a workflow's
    `settings:` block (pipeline/workflow.py), whose knobs are the ones a
    person actually sees. A step param is labelled by its `name` — that is
    the name you would type after `--param` — but a pipeline setting is a
    control on a form and gets a written label. `group` names the box the UI
    draws it in; the only value that means anything today is `outputs`,
    which puts a setting in the Outputs box beside the deliverable
    checkboxes it modifies (the SeedVR2 upscale is the case).
    """

    name: str
    type: type = str
    default: Any = REQUIRED
    help: str = ""
    choices: Tuple[Any, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    advanced: bool = False
    label: str = ""
    group: str = ""

    @property
    def title(self) -> str:
        """What to write on this param's control."""
        return self.label or self.name


class ParamError(ValueError):
    """A param that is missing, unknown, or the wrong type for its step."""


def coerce_param(value: Any, param: "Param", where: str) -> Any:
    """Bring `value` to `param.type`, or explain why it can't be.

    Lenient by design: YAML types most values correctly, but a `--param` or a
    UI text box hands over strings, and refusing `"6"` for an int would make
    both unusable. `None` passes through untouched — see `Param.default`.
    """
    if value is None:
        return None
    target = param.type
    try:
        if target is bool:
            if isinstance(value, str):
                return value.strip().lower() not in FALSE_STRINGS
            return bool(value)
        if target is int:
            # Via float first so "1e4" and 1.0 both work; a fractional value
            # is a mistake worth reporting rather than silently truncating.
            number = float(value)
            if number != int(number):
                raise ValueError(f"{value!r} is not a whole number")
            return int(number)
        if target is float:
            return float(value)
        if target is str:
            return value if isinstance(value, str) else str(value)
        if target is list:
            if isinstance(value, (list, tuple)):
                return list(value)
            raise ValueError(f"expected a list, got {type(value).__name__}")
    except (TypeError, ValueError) as exc:
        raise ParamError(
            f"{where}: param '{param.name}' expects {target.__name__}, "
            f"got {value!r} ({exc})"
        ) from None
    return value


def with_defaults(params: Tuple["Param", ...], **overrides: Any) -> Tuple["Param", ...]:
    """`params` with some defaults replaced — a specialization's declaration.

    Two steps that share their numerics but not their tuning are the case
    this exists for: `pointmap_splat` (a whole body, RMBG's matte) and
    `face_pointmap_splat` (a head crop, a segmentation mask) run identical
    code and disagree on two measured constants. Re-typing the base's whole
    PARAMS tuple to change them would put every *other* default in two
    places, where they drift silently.

    `Param` is a frozen dataclass, so this is `dataclasses.replace` over the
    tuple, order preserved. An override naming a param the base does not
    declare raises — same reasoning as `Step.resolve_params`: a typo that
    quietly does nothing is worse than a refusal.
    """
    from dataclasses import replace

    declared = {param.name for param in params}
    unknown = sorted(set(overrides) - declared)
    if unknown:
        raise ParamError(
            f"with_defaults: no such param(s) {', '.join(unknown)}. "
            f"The base declares: {', '.join(param.name for param in params)}"
        )
    return tuple(
        replace(param, default=overrides[param.name]) if param.name in overrides else param
        for param in params
    )


class Step(ABC):
    # The params this step accepts. Empty means "undeclared": the runner
    # passes a workflow's overrides through untouched and the UI has nothing
    # to draw. That is the escape hatch for a step still being written, not
    # the norm — everything in pipeline/steps/ declares its own.
    PARAMS: Tuple[Param, ...] = ()

    # Set by @register_step, so an error message can name the step the way a
    # workflow YAML does rather than by its class name.
    STEP_NAME: str = ""

    @classmethod
    def declared_params(cls) -> Dict[str, Param]:
        """This step's params, keyed by name, in declaration order."""
        return {param.name: param for param in cls.PARAMS}

    @classmethod
    def resolve_params(cls, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge `overrides` onto the declared defaults and coerce the result.

        The one place a step's effective params are decided. Called by
        `WorkflowRunner` before dispatch — so every dispatcher and both
        worker modes see a complete dict — and again by `pipeline.worker`,
        which is safe because this is idempotent: re-merging an
        already-complete dict changes nothing.

        Raises `ParamError` for an override naming a param the step doesn't
        declare (a typo that silently does nothing is worse than a refusal:
        the run completes, at the wrong settings, looking like it honoured
        you) and for a REQUIRED param nothing supplied.
        """
        overrides = dict(overrides or {})
        declared = cls.declared_params()
        if not declared:
            return overrides

        label = cls.STEP_NAME or cls.__name__
        unknown = sorted(set(overrides) - set(declared))
        if unknown:
            raise ParamError(
                f"Step '{label}' was given params it does not declare: "
                f"{', '.join(unknown)}. It accepts: {', '.join(declared)}"
            )

        resolved: Dict[str, Any] = {}
        for name, param in declared.items():
            if name in overrides:
                resolved[name] = coerce_param(overrides[name], param, f"step '{label}'")
                continue
            if param.default is REQUIRED:
                raise ParamError(
                    f"Step '{label}' needs param '{name}'"
                    + (f" ({param.help})" if param.help else "")
                    + ", and neither the step nor the workflow supplies a value."
                )
            resolved[name] = param.default
        return resolved

    @abstractmethod
    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step. Must be side-effect-free w.r.t. disk unless the
        step's whole purpose is I/O (e.g. SaveDatasetStep).

        `params` arrives complete — every name in `PARAMS`, already coerced
        to its declared type. Read it with `params["name"]`.
        """
        raise NotImplementedError

    def load(self, params: Dict[str, Any]) -> None:
        """Optional: acquire expensive state (load model weights onto GPU).

        Only meaningful for dispatchers that keep a Step instance alive
        across multiple `run()` calls (e.g. ServiceDispatcher). Dispatchers
        that instantiate fresh per call can skip calling this.
        """

    def unload(self) -> None:
        """Optional: release state acquired in load() (free VRAM)."""

    def release_vram(self) -> None:
        """Optional: give the GPU back but KEEP the weights in host RAM.

        The middle ground between "still loaded" and `unload()`, and the
        only reason `keep_loaded` is safe on a shared card. Two evictions,
        two costs:

          * release_vram() — VRAM freed, weights stay in DRAM. The next
            run() re-uploads over PCIe: seconds.
          * unload() — both freed. The next run() re-reads the checkpoint:
            on a RunPod network volume, ~47 GB and tens of minutes for
            wan22_vace_denoise.

        `keep_loaded` exists to skip the *network* read, not to squat on
        the card. fast_helical_native.yaml runs `brush` — a GPU program —
        between its two `wan22_vace_denoise` passes, so a resident worker
        that held ~35 GB of Wan experts in VRAM across that gap would OOM
        brush on any card in existence: strictly worse than the
        reload-every-time behaviour it replaced. So a resident worker calls
        this after every job (see pipeline/worker.py's serve loop) and
        calls unload() only at the end of the run.

        Default is a no-op, which is correct for every step that holds no
        GPU state — most of them — and merely leaves the old behaviour for
        one that does. Override it if this step keeps torch modules alive
        between run() calls.

        Implementing it: move the modules to CPU, then
        `torch.cuda.empty_cache()` — the caching allocator does not hand
        memory back to the driver just because a tensor moved, so without
        the second half the first half buys nothing (see
        dispatch/in_process.py for where that was learned).

        **If the step used accelerate's offload hooks, do not `.to("cpu")`
        it.** `enable_model_cpu_offload()` installs hooks that move modules
        on and off the card per forward pass and track where each one
        belongs; a manual `.to()` underneath them desynchronises that
        bookkeeping. diffusers exposes `pipe.maybe_free_model_hooks()` for
        exactly this — it returns everything to CPU and leaves the hooks
        able to bring it back. A plain `pipe.to(device)` load has no hooks,
        so it *does* need the manual `.to("cpu")`, and its run() then has
        to put the pipeline back on the device before using it. Which of
        the two applies is step-specific knowledge, which is why this hook
        is here rather than being guessed at by the dispatcher.

        Must be idempotent: it is called after every job, including one
        that changed nothing, and may be called when nothing is loaded.
        """
