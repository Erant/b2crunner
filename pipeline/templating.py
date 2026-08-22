"""Minimal `${a.b.c}` substitution for workflow YAML param values.

Not a general templating language on purpose — this is a research pipeline's
config file, not user-facing software. Just enough to let a step's params
reference the workflow's top-level `params:` block (resolution, step counts,
etc.) without repeating literal values at every step.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


def resolve(value: Any, scope: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = _PATTERN.match(value.strip())
        if match:
            return _lookup(match.group(1), scope)
        return value
    if isinstance(value, dict):
        return {k: resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, scope) for v in value]
    return value


def _lookup(path: str, scope: Dict[str, Any]) -> Any:
    parts = path.split(".")
    obj: Any = scope
    for part in parts:
        if isinstance(obj, dict):
            obj = obj[part]
        elif isinstance(obj, (list, tuple)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj
