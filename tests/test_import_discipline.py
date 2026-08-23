"""Every step module must be importable with only the core deps installed.

pipeline/steps/__init__.py imports all step modules unconditionally so the
registry is complete, including inside an isolated venv that has exactly one
step's dependencies installed (see pipeline/README.md, "Import discipline").
A top-level `import torch` / `from PIL import Image` in any step module
therefore breaks `python -m pipeline.worker <other_step>` in every other env.

This is a static check rather than an import test: it stays meaningful even
in a dev venv that happens to have torch and PIL installed, which is exactly
where such a regression would otherwise go unnoticed.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

STEPS_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "steps"

# Guaranteed present anywhere pipeline itself is installed (see
# pyproject.toml's dependencies) — everything else must be deferred.
ALLOWED_THIRD_PARTY = {"numpy", "cv2", "yaml", "requests", "body2colmap"}

# The real stdlib list rather than a hand-maintained one — a missing entry
# here is a false positive that looks exactly like a real finding (urllib
# was one).
STDLIB_OK = set(sys.stdlib_module_names)


def _top_level_imports(tree: ast.Module):
    """Module names imported at module scope (not inside a def/class)."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import within the package
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


class TestStepImportDiscipline(unittest.TestCase):
    def test_no_heavy_top_level_imports(self):
        offenders = []
        for path in sorted(STEPS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for module, lineno in _top_level_imports(tree):
                if module in STDLIB_OK or module in ALLOWED_THIRD_PARTY:
                    continue
                offenders.append(f"{path.name}:{lineno}: import {module}")

        self.assertEqual(
            offenders,
            [],
            "step modules import non-core dependencies at module scope; defer "
            "these into load()/run() or pipeline.worker breaks in every "
            "isolated venv that lacks them:\n  " + "\n  ".join(offenders),
        )

    def test_registry_is_complete_with_core_deps_only(self):
        import pipeline.steps  # noqa: F401
        from pipeline.registry import STEP_REGISTRY

        expected = {
            "brush", "colmap_export", "generate_firstlast", "inject_anchor",
            "load_dataset", "render", "rmbg", "sam3d_body", "sapiens2_lite",
            "save_dataset", "seedvr2", "wan22_vace_denoise",
        }
        self.assertTrue(expected.issubset(set(STEP_REGISTRY)))


if __name__ == "__main__":
    unittest.main()
