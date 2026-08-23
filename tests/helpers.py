"""Shared test helpers.

The golden-data tests below run against `cyber_6f/`, a real completed run of
the original ComfyUI pipeline (stage directories initial/ -> circular/ ->
splatted/ -> masked_splatted/ -> helical/ -> upscaled/ -> colmap/). That
directory is gitignored local reference data, so every test that needs it
skips cleanly when it isn't present rather than failing.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CYBER_6F = REPO_ROOT / "cyber_6f"


def require_stage(*names: str) -> Path | tuple[Path, ...]:
    """Skip the calling test unless every named cyber_6f stage dir exists."""
    paths = []
    for name in names:
        p = CYBER_6F / name
        if not (p / "metadata.json").exists() and not p.is_dir():
            raise unittest.SkipTest(f"reference data missing: {p}")
        if not p.is_dir():
            raise unittest.SkipTest(f"reference data missing: {p}")
        paths.append(p)
    return paths[0] if len(paths) == 1 else tuple(paths)
