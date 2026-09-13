"""The `versions:` line every run's log opens with names all three commits.

pipeline/provenance.py asks each moving part for its commit: git for this
checkout (or the ENV the image bakes from its build args), pip's PEP 610
record for body2colmap, `b2ctrain --version` for the trainer (with the
image's pin as the fallback for a binary that predates b49480f). These
tests pin the parsing and the fallbacks without a git, a pip or a binary
of their own.
"""

from __future__ import annotations

import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from pipeline import provenance
from pipeline.doctor import OK, WARN, check_versions, log_machine_banner


class _Dist:
    def __init__(self, version: str, direct_url):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name: str):
        assert name == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


def _fake_binary(directory: Path, output: str) -> Path:
    path = directory / "b2ctrain"
    path.write_text(f"#!/bin/sh\necho '{output}'\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class B2crunnerRevision(unittest.TestCase):
    def test_a_checkout_answers_with_git_and_marks_a_dirty_tree(self):
        with mock.patch.object(provenance, "_git", side_effect=["abc123def456", " M x.py"]):
            with mock.patch.object(Path, "exists", return_value=True):
                self.assertEqual(provenance.git_revision(Path("/repo")), "abc123def456-dirty")
        with mock.patch.object(provenance, "_git", side_effect=["abc123def456", ""]):
            with mock.patch.object(Path, "exists", return_value=True):
                self.assertEqual(provenance.git_revision(Path("/repo")), "abc123def456")

    def test_without_a_checkout_the_image_s_env_stands_in(self):
        with mock.patch.object(provenance, "git_revision", return_value=None):
            with mock.patch.dict(os.environ, {"B2C_GIT_REVISION": "a81d396f1c2d"}):
                self.assertEqual(provenance.b2crunner_revision(), "a81d396f1c2d")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(provenance.b2crunner_revision(), "unknown")


class Body2colmapRevision(unittest.TestCase):
    def _with(self, dist):
        return mock.patch("importlib.metadata.distribution", return_value=dist)

    def test_a_git_install_reports_pip_s_resolved_commit(self):
        record = {"url": "https://github.com/Erant/body2colmap.git",
                  "vcs_info": {"vcs": "git", "commit_id": "339a5983d9c36e31e5dd0265a8de905eb8d4c9c4",
                               "requested_revision": "339a5983d9c36e31e5dd0265a8de905eb8d4c9c4"}}
        with self._with(_Dist("0.2.0", record)):
            self.assertEqual(provenance.body2colmap_revision(), "339a5983d9c3")

    def test_an_editable_install_asks_its_checkout(self):
        record = {"url": "file:///home/x/body2colmap", "dir_info": {"editable": True}}
        with self._with(_Dist("0.2.0", record)):
            with mock.patch.object(provenance, "git_revision", return_value="76a74bc12345-dirty") as git:
                with mock.patch.object(Path, "exists", return_value=True):
                    self.assertEqual(provenance.body2colmap_revision(), "76a74bc12345-dirty")
        git.assert_called_once_with(Path("/home/x/body2colmap"))

    def test_a_pypi_install_or_no_install_is_unknown_not_an_error(self):
        with self._with(_Dist("0.2.0", None)):
            self.assertEqual(provenance.body2colmap_revision(), "unknown (pypi 0.2.0)")
        with mock.patch("importlib.metadata.distribution", side_effect=Exception("no dist")):
            self.assertEqual(provenance.body2colmap_revision(), "unknown (not installed)")


class B2ctrainRevision(unittest.TestCase):
    def test_a_binary_that_names_its_commit_is_believed(self):
        with TemporaryDirectory() as tmp:
            _fake_binary(Path(tmp), "b2ctrain 0.1.0 (b49480f8739c)")
            with mock.patch.dict(os.environ, {"PATH": tmp, "B2CTRAIN_REF": "deadbeef"}):
                self.assertEqual(provenance.b2ctrain_revision(), "b49480f8739c")
            _fake_binary(Path(tmp), "b2ctrain 0.1.0 (b49480f8739c-dirty)")
            with mock.patch.dict(os.environ, {"PATH": tmp}):
                self.assertEqual(provenance.b2ctrain_revision(), "b49480f8739c-dirty")

    def test_an_older_binary_falls_back_to_the_image_s_pin_and_says_so(self):
        with TemporaryDirectory() as tmp:
            _fake_binary(Path(tmp), "b2ctrain 0.1.0")
            env = {"PATH": tmp, "B2CTRAIN_REF": "2977f0e88aa49c7208d793d823cfab28b44c990d"}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(
                    provenance.b2ctrain_revision(),
                    "2977f0e88aa4 (the image's pin; the binary reports only 'b2ctrain 0.1.0')",
                )
            with mock.patch.dict(os.environ, {"PATH": tmp}, clear=True):
                self.assertEqual(provenance.b2ctrain_revision(), "unknown ('b2ctrain 0.1.0')")

    def test_no_binary_is_unknown(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"PATH": tmp}, clear=True):
                self.assertEqual(provenance.b2ctrain_revision(), "unknown (not on PATH)")


class TheDoctorAndTheBanner(unittest.TestCase):
    def test_describe_is_one_line_in_a_fixed_order(self):
        found = {"b2crunner": "a", "body2colmap": "b", "b2ctrain": "c"}
        self.assertEqual(provenance.describe(found), "b2crunner a, body2colmap b, b2ctrain c")

    def test_the_doctor_warns_on_an_unknown_and_never_fails(self):
        with mock.patch.object(provenance, "versions",
                               return_value={"b2crunner": "a", "body2colmap": "b", "b2ctrain": "c"}):
            self.assertEqual(check_versions().status, OK)
        with mock.patch.object(provenance, "versions",
                               return_value={"b2crunner": "a", "body2colmap": "unknown (pypi 0.2.0)",
                                             "b2ctrain": "c"}):
            check = check_versions()
        self.assertEqual(check.status, WARN)
        self.assertEqual(check.detail, "b2crunner a, body2colmap unknown (pypi 0.2.0), b2ctrain c")

    def test_the_banner_opens_with_the_versions_line(self):
        found = {"b2crunner": "a81d396f1c2d", "body2colmap": "339a5983d9c3", "b2ctrain": "b49480f8739c"}
        with mock.patch.object(provenance, "versions", return_value=found):
            with self.assertLogs("pipeline.doctor", level="INFO") as logs:
                log_machine_banner()
        self.assertEqual(
            logs.output[0],
            "INFO:pipeline.doctor:versions: b2crunner a81d396f1c2d, body2colmap 339a5983d9c3, "
            "b2ctrain b49480f8739c",
        )


class TheImageBakesTheFallbacks(unittest.TestCase):
    """The ENVs provenance.py falls back on, and where they sit in the Dockerfile."""

    @classmethod
    def setUpClass(cls):
        cls.text = (Path(__file__).resolve().parent.parent / "docker" / "Dockerfile").read_text()

    def test_the_revision_label_is_also_an_env(self):
        self.assertIn('ENV B2C_GIT_REVISION="${GIT_SHA}${GIT_DIRTY}"', self.text)
        self.assertIn('B2CTRAIN_REF="${B2CTRAIN_REF}"', self.text)

    def test_the_trainer_pin_is_one_global_arg_both_stages_redeclare(self):
        first_from = self.text.index("\nFROM ")
        pin = self.text.index("\nARG B2CTRAIN_REF=")
        self.assertLess(pin, first_from, "the pin must be declared before the first FROM to reach two stages")
        self.assertEqual(self.text.count("\nARG B2CTRAIN_REF="), 1, "one pin, bumped in one place")
        self.assertEqual(self.text.count("\nARG B2CTRAIN_REF\n"), 2, "the builder and the runtime stage each redeclare it")

    def test_the_provenance_envs_sit_last_with_the_label(self):
        env = self.text.index("ENV B2C_GIT_REVISION=")
        self.assertGreater(env, self.text.index("LABEL org.opencontainers.image.title"))
        self.assertNotIn("\nRUN ", self.text[env:], "a RUN after the provenance ENV would re-run on every sha")
        self.assertNotIn("\nCOPY ", self.text[env:])


if __name__ == "__main__":
    unittest.main()
