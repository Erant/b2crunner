"""Static validation of every workflow YAML in pipeline/workflows/.

None of these can be executed without a GPU, but most of the ways a
workflow file goes wrong are visible without running it: a step name that
isn't registered, a `${params.x}` that doesn't resolve, an `env:` with no
entry in envs.yaml, a dispatch mode that doesn't exist. This catches those
so a pod run fails on the model, not on a typo.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from pipeline.registry import STEP_REGISTRY
from pipeline.templating import resolve
from pipeline.workflow import WorkflowSpec

import pipeline.steps  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / "pipeline" / "workflows"
ENVS_FILE = REPO_ROOT / "pipeline" / "envs" / "envs.yaml"
DOCKER_ENVS_FILE = REPO_ROOT / "docker" / "envs.docker.yaml"

VALID_DISPATCH = {"in_process", "subprocess", "service", "docker"}


def _workflows():
    return sorted(WORKFLOW_DIR.glob("*.yaml"))


class TestWorkflowFiles(unittest.TestCase):
    def test_there_are_workflows_to_check(self):
        self.assertTrue(_workflows())

    def test_each_workflow_loads(self):
        for path in _workflows():
            with self.subTest(workflow=path.name):
                spec = WorkflowSpec.from_yaml(str(path))
                self.assertTrue(spec.name)
                self.assertTrue(spec.steps)

    def test_step_names_are_registered(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(
                        step.step, STEP_REGISTRY,
                        f"{path.name}: step '{step.id}' references unregistered "
                        f"step '{step.step}'",
                    )

    def test_step_ids_are_unique(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            ids = [s.id for s in spec.steps]
            with self.subTest(workflow=path.name):
                self.assertEqual(len(ids), len(set(ids)))

    def test_dispatch_modes_are_valid(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(step.dispatch, VALID_DISPATCH)

    def test_envs_referenced_exist(self):
        envs = yaml.safe_load(ENVS_FILE.read_text())["envs"]
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                if step.dispatch == "in_process" or not step.env:
                    continue
                with self.subTest(workflow=path.name, step=step.id):
                    self.assertIn(
                        step.env, envs,
                        f"{path.name}: step '{step.id}' uses env '{step.env}' "
                        f"which envs.yaml does not define",
                    )

    def test_non_in_process_steps_name_an_env(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                if step.dispatch in ("subprocess", "docker", "service"):
                    with self.subTest(workflow=path.name, step=step.id):
                        self.assertTrue(
                            step.env,
                            f"{path.name}: '{step.id}' dispatches "
                            f"{step.dispatch} but names no env",
                        )

    def test_docker_env_registry_matches_the_repo_one(self):
        """docker/envs.docker.yaml is copied over pipeline/envs/envs.yaml
        inside the image, so it has to define the same env names.

        These two drifted before: the Dockerfile built /workspace/venv_*
        while envs.yaml named pipeline/envs/*/venv/bin/python, and every
        subprocess step in the image would have failed to find an
        interpreter. Names here, not paths — the paths differ on purpose
        (bare pod vs container).
        """
        repo_envs = set(yaml.safe_load(ENVS_FILE.read_text())["envs"])
        docker_envs = set(yaml.safe_load(DOCKER_ENVS_FILE.read_text())["envs"])
        self.assertEqual(
            repo_envs, docker_envs,
            "pipeline/envs/envs.yaml and docker/envs.docker.yaml define "
            "different env names",
        )

    def test_docker_env_paths_are_absolute(self):
        """A relative python_bin resolves against the process working
        directory, which is what made the container paths wrong before."""
        envs = yaml.safe_load(DOCKER_ENVS_FILE.read_text())["envs"]
        for name, cfg in envs.items():
            with self.subTest(env=name):
                self.assertTrue(
                    cfg["python_bin"].startswith("/"),
                    f"{name}: python_bin must be absolute inside the image",
                )

    def test_all_param_templates_resolve(self):
        """A ${params.x} naming a param that doesn't exist is the single
        easiest mistake to make in these files, and it would otherwise only
        surface once that step is reached on a GPU."""
        for path in _workflows():
            raw = yaml.safe_load(path.read_text())
            params = raw.get("params", {})
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    try:
                        resolve(step.params, {"params": params})
                    except Exception as exc:  # noqa: BLE001
                        self.fail(
                            f"{path.name}: step '{step.id}' has an unresolvable "
                            f"template: {exc}"
                        )

    def test_when_conditions_resolve(self):
        """Same reasoning as the params check above, and more load-bearing:
        an unresolvable `when:` guards a step, and the whole point of
        resolving them up front in the runner is that this fails at the
        start of a run rather than an hour into one."""
        from pipeline.workflow import step_enabled

        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                with self.subTest(workflow=path.name, step=step.id):
                    try:
                        step_enabled(step, spec.params)
                    except Exception as exc:  # noqa: BLE001
                        self.fail(f"{path.name}: step '{step.id}': {exc}")

    def test_the_two_fast_helical_files_differ_only_by_the_upscale(self):
        """They are maintained as copies — there is no include mechanism —
        so the thing worth checking is that they have not drifted into two
        different pipelines that happen to share a name."""
        full = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical_full.yaml"))
        short = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / "fast_helical.yaml"))

        upscale_only = {"upscale", "rescale_cameras"}
        self.assertEqual(
            [s.id for s in full.steps if s.id not in upscale_only],
            [s.id for s in short.steps],
        )
        self.assertEqual(
            set(full.params) - set(short.params), {"upscale_resolution"},
            "the only param either file should have to itself is the upscale's",
        )
        for step in short.steps:
            twin = next(s for s in full.steps if s.id == step.id)
            with self.subTest(step=step.id):
                self.assertEqual(step.step, twin.step)
                self.assertEqual(step.dispatch, twin.dispatch)
                self.assertEqual(step.env, twin.env)
                self.assertEqual(step.inputs, twin.inputs)
                self.assertEqual(step.outputs, twin.outputs)
                self.assertEqual(step.when, twin.when)
                self.assertEqual(step.keep_loaded, twin.keep_loaded)

    def test_every_workflow_declares_the_output_switches(self):
        """A shipped workflow that silently ignores the UI's Outputs
        checkboxes would run for an hour and hand back the wrong thing."""
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            with self.subTest(workflow=path.name):
                self.assertIn("export_colmap", spec.params)
                self.assertIn("export_ply", spec.params)

    def test_context_paths_look_like_paths(self):
        for path in _workflows():
            spec = WorkflowSpec.from_yaml(str(path))
            for step in spec.steps:
                for name, ctx_path in list(step.inputs.items()) + list(step.outputs.items()):
                    with self.subTest(workflow=path.name, step=step.id, name=name):
                        self.assertIsInstance(ctx_path, str)
                        self.assertTrue(ctx_path)
                        self.assertNotIn(" ", ctx_path)
                        self.assertFalse(ctx_path.startswith("."))
                        self.assertFalse(ctx_path.endswith("."))


if __name__ == "__main__":
    unittest.main()
