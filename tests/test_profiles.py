"""Quality profiles (`profiles:` in the workflows, pipeline/workflow.py `Profile`)."""
import tempfile
import unittest
from pathlib import Path

import yaml

import pipeline.steps  # noqa: F401
from pipeline.templating import resolve
from pipeline.workflow import WorkflowSpec, truthy, when_truthy

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / "pipeline" / "workflows").glob("helical*.yaml"))


def _live(spec):
    return {s.id for s in spec.steps if when_truthy(resolve(s.when, {"globals": spec.globals}))}


class TestShippedProfiles(unittest.TestCase):
    def test_medium_high(self):
        for path in WORKFLOWS:
            spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                self.assertEqual([p.name for p in spec.profiles], ["medium", "high"])
                self.assertEqual([p.title for p in spec.profiles], ["Medium", "High"])

    def test_each_profile_runs_what_it_says(self):
        # (extend, reupscale) — every one relit, upscaled and trained
        expect = {"medium": (False, False), "high": (True, False)}
        for path in WORKFLOWS:
            for profile in WorkflowSpec.from_yaml(str(path)).profiles:
                with self.subTest(workflow=path.name, profile=profile.name):
                    spec = WorkflowSpec.from_yaml(str(path))
                    spec.globals.update(profile.settings)
                    spec.validate()
                    live = _live(spec)
                    extend, reupscale = expect[profile.name]
                    self.assertEqual(truthy(spec.globals["extend_orbit"]), extend)
                    self.assertEqual(spec.globals["lighting_correction"], "prepass")
                    self.assertIn("upscale", live)
                    self.assertIn("train_final_splat", live)
                    self.assertEqual("reupscale_train" in live, reupscale)


class TestProfileDeclarations(unittest.TestCase):
    BASE = {
        "name": "t",
        "settings": [{"name": "a", "type": "bool", "default": False},
                     {"name": "b", "type": "bool", "default": False},
                     {"name": "c", "default": "x", "choices": ["x", "y"]}],
        "steps": [{"id": "s", "step": "_test_noop", "when": ["${globals.a}", "${globals.b}", "${globals.c}"]}],
    }

    def _spec(self, profiles):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.yaml"
            path.write_text(yaml.safe_dump({**self.BASE, "profiles": profiles}))
            return WorkflowSpec.from_yaml(str(path))

    def _refused(self, profiles):
        with self.assertRaises(ValueError):
            self._spec(profiles)._validate_declarations()

    def test_a_good_profile(self):
        self._spec([{"name": "p", "settings": {"a": True, "c": "y"}}])._validate_declarations()

    def test_refusals(self):
        self._refused([{"name": "p", "settings": {"nope": True}}])
        self._refused([{"name": "p", "settings": {"c": "z"}}])
        self._refused([{"name": "p", "settings": {"a": True}}, {"name": "p", "settings": {"a": False}}])
        with self.assertRaises(ValueError):
            self._spec([{"name": "Custom", "settings": {"a": True}}])
        with self.assertRaises(ValueError):
            self._spec([{"name": "p", "settings": {}}])
        with self.assertRaises(ValueError):
            self._spec([{"name": "p", "settings": {"a": True}, "colour": "red"}])
