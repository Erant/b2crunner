"""Global registry mapping a workflow YAML's `step:` name to a Step class."""

from __future__ import annotations

from typing import Dict, Type

from .step import Step

STEP_REGISTRY: Dict[str, Type[Step]] = {}


def register_step(name: str):
    def decorator(cls: Type[Step]) -> Type[Step]:
        if name in STEP_REGISTRY and STEP_REGISTRY[name] is not cls:
            raise ValueError(f"Step '{name}' already registered to {STEP_REGISTRY[name]!r}")
        # Hand the class the name a workflow YAML calls it by, so param
        # errors from Step.resolve_params can say "step 'mask_splat'" rather
        # than "step 'MaskSplatStep'".
        cls.STEP_NAME = name
        STEP_REGISTRY[name] = cls
        return cls

    return decorator


def get_step_class(name: str) -> Type[Step]:
    try:
        return STEP_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown step '{name}'. Registered steps: {sorted(STEP_REGISTRY)}. "
            "Make sure the module defining it has been imported (see pipeline/steps/__init__.py)."
        ) from None
