"""Everything a run writes has to land on the mounted volume.

On a pod the container's writable layer is a small overlay: an 81-frame
batch of dispatcher IPC pickles is ~220 MB in each direction, a trained
splat is a few GB, and anything written there vanishes when the pod is
recycled. These are the defaults and overrides that keep that from
happening — and the fallback that keeps a laptop with no /data working.
"""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.saved = {
            key: os.environ.get(key)
            for key in ("B2C_DATA_DIR", "B2C_OUTPUT_DIR", "B2C_LOG_DIR",
                        "B2C_UPLOAD_DIR", "TMPDIR")
        }
        self.addCleanup(self._restore)
        for key in self.saved:
            os.environ.pop(key, None)

    def _restore(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_subdirectories_hang_off_the_data_dir(self):
        os.environ["B2C_DATA_DIR"] = self.tmp.name
        from pipeline import paths

        self.assertEqual(paths.data_dir(), Path(self.tmp.name))
        self.assertEqual(paths.output_dir(), Path(self.tmp.name) / "output")
        self.assertEqual(paths.log_dir(), Path(self.tmp.name) / "logs")
        self.assertEqual(paths.upload_dir(), Path(self.tmp.name) / "uploads")
        for directory in (paths.output_dir(), paths.log_dir(), paths.upload_dir()):
            self.assertTrue(directory.is_dir())

    def test_each_subdirectory_can_be_overridden_on_its_own(self):
        os.environ["B2C_DATA_DIR"] = self.tmp.name
        elsewhere = Path(self.tmp.name) / "somewhere-else"
        os.environ["B2C_OUTPUT_DIR"] = str(elsewhere)
        from pipeline import paths

        self.assertEqual(paths.output_dir(), elsewhere)
        self.assertEqual(paths.log_dir(), Path(self.tmp.name) / "logs")

    def test_an_unusable_data_dir_falls_back_instead_of_raising(self):
        """A dev box with no /data must still run, not crash on import."""
        os.environ["B2C_DATA_DIR"] = "/proc/definitely/not/writable"
        from pipeline import paths

        importlib.reload(paths)
        resolved = paths.data_dir()
        self.assertTrue(str(resolved).endswith("_local_data"))

    def test_configure_tmpdir_sets_both_the_env_var_and_tempfile(self):
        """The env var is for child processes; tempfile cached its own answer."""
        os.environ["B2C_DATA_DIR"] = self.tmp.name
        from pipeline import paths

        resolved = paths.configure_tmpdir()
        self.assertEqual(os.environ["TMPDIR"], str(resolved))
        self.assertEqual(tempfile.tempdir, str(resolved))
        self.assertTrue(resolved.is_dir())
        tempfile.tempdir = None  # leave the module as we found it


if __name__ == "__main__":
    unittest.main()


class TestModelDownloadDestinations(unittest.TestCase):
    """No step may cache weights inside the container.

    Most checkpoints come down through huggingface_hub and land in HF_HOME
    without anyone thinking about it. The exceptions are the ones that bit:
    seedvr2 hands its vendored downloader an explicit `model_dir` that
    ignores HF_HOME (it defaulted to the relative "models/SEEDVR2", i.e.
    inside the container, several GB of it), and face_landmarks fetches
    MediaPipe's files from a URL into what used to be ~/.cache. Both are
    lost on every pod restart and both eat the container disk.

    A static check, because reaching either code path for real needs a GPU
    and a licence-accepted HF token.
    """

    def test_seedvr2_model_dir_defaults_onto_the_volume(self):
        import inspect
        import os
        import tempfile

        from pipeline.paths import models_dir
        from pipeline.steps.seedvr2 import SeedVR2Step

        # Comment lines stripped first: the code carries an explanation of
        # what the old default was and why it moved, which would otherwise
        # trip this check on its own prose.
        source = inspect.getsource(SeedVR2Step.load)
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            '"models/SEEDVR2"', code,
            "seedvr2 is back to a relative model_dir, which resolves inside the container",
        )
        self.assertIn("models_dir()", code)

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("B2C_DATA_DIR")
            os.environ["B2C_DATA_DIR"] = tmp
            try:
                self.assertTrue(str(models_dir()).startswith(tmp))
            finally:
                if previous is None:
                    os.environ.pop("B2C_DATA_DIR", None)
                else:
                    os.environ["B2C_DATA_DIR"] = previous

    def test_face_landmarks_does_not_cache_in_home(self):
        from pipeline.steps import face_landmarks

        source = Path(face_landmarks.__file__).read_text()
        self.assertNotIn(
            "Path.home()", source,
            "face_landmarks caches MediaPipe models under ~/.cache, which is inside "
            "the container on a pod",
        )

    def test_no_step_module_names_a_relative_models_directory(self):
        import re

        steps_dir = Path(__file__).resolve().parent.parent / "pipeline" / "steps"
        offenders = []
        for path in sorted(steps_dir.glob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if line.lstrip().startswith("#"):
                    continue
                if re.search(r'["\'](?!/)[\w./-]*models?/[\w./-]+["\']', line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "a step names a relative model directory; it resolves against the worker "
            "subprocess's cwd, which on a pod is inside the container:\n  "
            + "\n  ".join(offenders),
        )
