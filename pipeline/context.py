"""Shared in-memory state a workflow's steps read from / write into.

A step's `inputs`/`outputs` in YAML are dotted paths into this context, e.g.
`dataset.images` or `denoised.images`. `dataset` (a Dataset instance) is
seeded by the runner before the first step; steps write new named objects
into the context as they run. Nothing here ever touches disk — that's the
entire reason a plain dict-of-objects is enough.
"""

from __future__ import annotations

from typing import Any, Dict


class Context:
    def __init__(self, initial: Dict[str, Any] | None = None):
        self._data: Dict[str, Any] = dict(initial or {})

    def get(self, path: str) -> Any:
        parts = path.split(".")
        obj = self._data.get(parts[0])
        if obj is None and parts[0] not in self._data:
            raise KeyError(f"Context has no top-level key '{parts[0]}' (looking up '{path}')")
        for part in parts[1:]:
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]
        return obj

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        if len(parts) == 1:
            self._data[parts[0]] = value
            return
        obj = self._data[parts[0]]
        for part in parts[1:-1]:
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]
        last = parts[-1]
        if hasattr(obj, last) and not isinstance(obj, dict):
            setattr(obj, last, value)
        else:
            obj[last] = value

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)
