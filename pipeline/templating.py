"""Minimal `${a.b.c}` substitution for workflow YAML param values.

Not a general templating language on purpose — this is a research pipeline's
config file, not user-facing software. Just enough to let a step's params
reference the workflow's `globals:` block (the frame size, the output root)
without repeating literal values at every step. `globals` is the only scope
a workflow resolves against: a step's own params are namespaced under its
`id:` and are not addressable from elsewhere in the file.

Two forms, and the difference matters:

    width: ${globals.resolution.0}             -> the value itself, typed
    output_dir: ${globals.output_root}/colmap  -> string interpolation

A whole-string reference keeps the value's type, so `${globals.resolution}`
stays a list and `${globals.resolution.0}` stays an int. Anything with text around
it can only be a string, so the referenced value is stringified and spliced
in. The second form exists so a workflow can derive its output paths from
one `output_root` global that the CLI repoints at the run's own directory —
without it, every disk-writing step had a hardcoded relative path, and on a
pod those resolve into the container's writable layer rather than the
mounted volume.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_WHOLE = re.compile(r"^\$\{([^}]+)\}$")
_INLINE = re.compile(r"\$\{([^}]+)\}")


def global_ref(value: Any) -> "str | None":
    """If `value` is a whole-string `${globals.X}` reference, return `X`.

    Only the whole-value form (`resolution: ${globals.resolution}`), not the
    inline path form (`${globals.output_root}/colmap`) — the caller wants
    the params a step reads verbatim from a global, so the UI can show them
    as linked-to-the-global rather than give one setting a second editable
    home. Returns None for anything else.
    """
    if not isinstance(value, str):
        return None
    match = _WHOLE.match(value.strip())
    if match and match.group(1).startswith("globals."):
        return match.group(1)[len("globals."):]
    return None


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


def referenced_globals(value: Any) -> "set[str]":
    """Every `globals.X` name any `${...}` in `value` reads, recursively.

    Both forms count and only the first path segment is kept, so
    `${globals.resolution.0}` and `${globals.output_root}/colmap` both report
    `resolution` / `output_root`. A reference to any other scope is ignored —
    there is no other scope today, but a typo'd one is not a global.

    `WorkflowSpec.validate` uses this to refuse a declared setting that
    nothing in the workflow reads: a control on the form that changes
    nothing is worse than no control, because it looks like it worked.
    """
    names: "set[str]" = set()
    if isinstance(value, str):
        for path in _INLINE.findall(value):
            head, _, rest = path.partition(".")
            if head == "globals" and rest:
                names.add(rest.split(".")[0])
    elif isinstance(value, dict):
        for item in value.values():
            names |= referenced_globals(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            names |= referenced_globals(item)
    return names
