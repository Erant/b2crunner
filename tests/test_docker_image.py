"""Static checks on the image definition.

The image takes the better part of an hour to build and is meant to be
built once per verification round, so a mistake in it is expensive in a way
a mistake in a Python module is not. Everything checkable without building
is checked here.

Each of these corresponds to a specific way this file has gone wrong, or
would have: the venv paths in envs.docker.yaml drifting from the venvs the
Dockerfile actually creates (which happened, and would have broken every
subprocess step in the image), the application directory sitting under
RunPod's default volume mount path, an entrypoint that exits immediately on
a pod, and unbuffered stdout.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DOCKER_ENVS = REPO_ROOT / "docker" / "envs.docker.yaml"

APP_DIR = "/opt/b2c_runner"


class TestDockerfile(unittest.TestCase):
    def setUp(self):
        self.text = DOCKERFILE.read_text()

    def test_the_app_does_not_live_under_the_default_runpod_volume_mount(self):
        """RunPod's pod template defaults its volume mount path to /workspace.

        A volume mounted there shadows everything beneath it. With the
        application at /workspace/b2c_runner — where it used to be — a pod
        left on the default setting boots into "No module named pipeline"
        while looking perfectly configured.
        """
        offenders = [
            line for line in self.text.splitlines()
            if "/workspace/b2c_runner" in line and not line.strip().startswith("#")
        ]
        self.assertEqual(
            offenders, [],
            "the image still puts the application under /workspace, which a RunPod "
            "volume mounted at its default path will shadow:\n  " + "\n  ".join(offenders),
        )

    def test_workdir_and_copy_agree_on_the_app_directory(self):
        self.assertIn(f"WORKDIR {APP_DIR}", self.text)
        self.assertIn(f"COPY . {APP_DIR}", self.text)

    def test_stdout_is_unbuffered(self):
        """Otherwise a step that runs for twenty minutes prints nothing until it ends."""
        self.assertIn("ENV PYTHONUNBUFFERED=1", self.text)

    def test_every_write_path_points_at_the_volume(self):
        for name in ("B2C_DATA_DIR", "B2C_OUTPUT_DIR", "B2C_LOG_DIR", "B2C_UPLOAD_DIR",
                     "TMPDIR", "HF_HOME"):
            match = re.search(rf"^ENV {name}=(\S+)", self.text, re.MULTILINE)
            with self.subTest(env=name):
                self.assertIsNotNone(match, f"{name} is not set in the runtime stage")
                self.assertTrue(
                    match.group(1).startswith("/data"),
                    f"{name}={match.group(1)} is not on the mounted volume; on a pod that "
                    f"is the container's small, impermanent writable layer",
                )

    def test_the_entrypoint_is_the_script_and_the_default_mode_stays_alive(self):
        """A pod whose container exits is a dead pod with nothing to attach to."""
        self.assertIn('ENTRYPOINT ["/usr/local/bin/b2c-entrypoint"]', self.text)
        self.assertIn('CMD ["ui"]', self.text)
        self.assertIn("COPY docker/entrypoint.sh /usr/local/bin/b2c-entrypoint", self.text)

    def test_the_ui_port_is_exposed(self):
        port = re.search(r"^ENV B2C_PORT=(\d+)", self.text, re.MULTILINE)
        self.assertIsNotNone(port)
        expose = re.search(r"^EXPOSE (.+)$", self.text, re.MULTILINE)
        self.assertIsNotNone(expose)
        self.assertIn(port.group(1), expose.group(1).split())

    def test_gradio_is_installed_into_the_venv_the_entrypoint_runs(self):
        """The entrypoint serves the UI with /opt/venv_main's interpreter.

        Not a check that it appears anywhere in the file — gradio in
        venv_base or one of the isolated step venvs would satisfy that and
        still leave `pipeline.cli ui` unable to import it.
        """
        self.assertRegex(
            self.text,
            r"/opt/venv_main/bin/pip install[^\n]*(\\\n[^\n]*)*gradio",
            "no `pip install ... gradio` into /opt/venv_main, which is the venv "
            "docker/entrypoint.sh runs the UI with",
        )

    def test_debugging_tools_are_present(self):
        """The image is built once; anything needed to diagnose a run must be in it."""
        for tool in ("git", "curl", "procps", "less", "rsync", "openssh-server", "unzip", "jq"):
            with self.subTest(tool=tool):
                self.assertRegex(
                    self.text, rf"(?m)^\s+.*\b{re.escape(tool)}\b",
                    f"{tool} is not installed in the runtime stage",
                )

    def test_every_env_in_the_registry_has_a_venv_built_and_copied(self):
        """envs.docker.yaml naming a venv the Dockerfile never creates would
        break that step only when a workflow first reached it."""
        envs = yaml.safe_load(DOCKER_ENVS.read_text())["envs"]
        for name, config in envs.items():
            venv_root = str(Path(config["python_bin"]).parent.parent)
            with self.subTest(env=name):
                self.assertIn(
                    f"make-child-venv {venv_root}", self.text,
                    f"env '{name}' points at {venv_root}, which the Dockerfile never creates",
                )
                self.assertIn(
                    f"COPY --from=python-builder {venv_root} {venv_root}", self.text,
                    f"env '{name}'s venv is built but never copied into the runtime stage",
                )

    def test_the_orchestrator_venv_is_also_built_and_copied(self):
        self.assertIn("make-child-venv /opt/venv_main", self.text)
        self.assertIn("COPY --from=python-builder /opt/venv_main /opt/venv_main", self.text)


class TestEntrypoint(unittest.TestCase):
    def setUp(self):
        self.text = ENTRYPOINT.read_text()

    def test_it_is_executable(self):
        self.assertTrue(ENTRYPOINT.stat().st_mode & 0o111, "docker/entrypoint.sh is not +x")

    def test_it_is_valid_bash(self):
        result = subprocess.run(["bash", "-n", str(ENTRYPOINT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_cli_subcommands_it_forwards_all_exist(self):
        from pipeline.cli import build_parser

        known = set(build_parser()._subparsers._group_actions[0].choices)
        for command in ("run", "doctor", "steps", "workflows", "ui"):
            with self.subTest(command=command):
                self.assertIn(command, known)
                self.assertIn(command, self.text)

    def test_it_uses_the_venv_that_has_gradio(self):
        self.assertIn("PYTHON=/opt/venv_main/bin/python", self.text)


class TestCompose(unittest.TestCase):
    def test_it_publishes_the_ui_port_the_image_exposes(self):
        compose = yaml.safe_load(COMPOSE.read_text())
        ports = compose["services"]["pipeline"].get("ports", [])
        port = re.search(r"^ENV B2C_PORT=(\d+)", DOCKERFILE.read_text(), re.MULTILINE).group(1)
        self.assertTrue(
            any(str(port) in str(entry) for entry in ports),
            f"docker-compose.yml does not publish port {port}",
        )

    def test_it_grants_the_graphics_driver_capability(self):
        """Without `graphics`, vulkaninfo finds no driver and brush cannot run."""
        compose = yaml.safe_load(COMPOSE.read_text())
        environment = compose["services"]["pipeline"]["environment"]
        capabilities = next(
            (e for e in environment if e.startswith("NVIDIA_DRIVER_CAPABILITIES=")), ""
        )
        self.assertIn("graphics", capabilities)


if __name__ == "__main__":
    unittest.main()
