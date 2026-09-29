"""tools/: the launcher's registry matches the tool files and envs.yaml, and it resolves interpreters."""
from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run  # noqa: E402


class TestTools(unittest.TestCase):
    def test_every_tool_is_registered_and_compiles(self):
        files = {p.stem for p in (ROOT / "tools").glob("*.py")} - {"run"}
        self.assertEqual(files, set(run.TOOLS))
        for name in files:
            py_compile.compile(str(ROOT / "tools" / f"{name}.py"), doraise=True)

    def test_every_environment_is_in_envs_yaml(self):
        self.assertTrue(set(run.TOOLS.values()) <= set(run.env_pythons()))

    def test_interpreter_override(self):
        os.environ["B2CRUNNER_PYTHON_WAN22"] = "/some/where/python"
        try:
            self.assertEqual(run.python_for("wan22"), Path("/some/where/python"))
        finally:
            del os.environ["B2CRUNNER_PYTHON_WAN22"]
        self.assertEqual(run.python_for("wan22"), ROOT / run.env_pythons()["wan22"])

    def test_list_and_unknown_tool(self):
        out = subprocess.run([sys.executable, str(ROOT / "tools" / "run.py"), "--list"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        self.assertEqual({line.split()[0] for line in out.stdout.splitlines()}, set(run.TOOLS))
        bad = subprocess.run([sys.executable, str(ROOT / "tools" / "run.py"), "no_such_tool"], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unknown tool", bad.stderr)


if __name__ == "__main__":
    unittest.main()
