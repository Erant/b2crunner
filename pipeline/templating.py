"""Minimal `${a.b.c}` substitution for workflow YAML param values.

Not a general templating language on purpose — this is a research pipeline's
config file, not user-facing software. Just enough to let a step's params
reference the workflow's top-level `params:` block (resolution, step counts,
etc.) without repeating literal values at every step.

Two forms, and the difference matters:

    steps: ${params.diffusion_steps}       -> the value itself, typed
    output_dir: ${params.output_root}/colmap  -> string interpolation

A whole-string reference keeps the value's type, so `${params.resolution}`
stays a list and `${params.seed}` stays an int. Anything with text around
it can only be a string, so the referenced value is stringified and spliced
in. The second form exists so a workflow can derive its output paths from
one `output_root` param that the CLI repoints at the run's own directory —
without it, every disk-writing step had a hardcoded relative path, and on a
pod those resolve into the container's writable layer rather than the
mounted volume.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_WHOLE = re.compile(r"^\$\{([^}]+)\}$")
_INLINE = re.compile(r"\$\{([^}]+)\}")


def resolve(value: Any, scope: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        match = _WHOLE.match(stripped)
        if match:
            return _lookup(match.group(1), scope)
        if "${" in value:
            return _INLINE.sub(lambda m: str(_lookup(m.group(1), scope)), value)
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
