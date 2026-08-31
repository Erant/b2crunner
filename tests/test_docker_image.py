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

    def test_colmap_is_built_with_cuda_and_reachable(self):
        """The camera refinement's binary, and the two ways it goes wrong.

        CUDA off is the silent one: COLMAP compiles and runs, its ONNX
        execution provider drops to CPU (`SelectONNXExecutionProvider`), and
        the only symptom is a refinement that takes minutes instead of
        seconds. There is nothing at runtime to fail on, so the build asserts
        it instead — and this asserts that the build asserts it.

        Reachability is the other: the binary is installed under its own
        prefix so its bundled ONNX Runtime stays off the global loader path,
        which means the prefix has to be copied whole (the RPATH is
        $ORIGIN/../lib) and put on PATH. steps/refine_cameras.py's default
        `colmap_path` is the bare name `colmap`.
        """
        self.assertIn("AS colmap-builder", self.text)
        self.assertIn("-DCUDA_ENABLED=ON", self.text)
        self.assertIn('grep -q "with CUDA"', self.text)
        self.assertIn("COPY --from=colmap-builder /opt/colmap /opt/colmap", self.text)
        self.assertIn("ENV PATH=/opt/colmap/bin:${PATH}", self.text)

    def test_colmap_s_apt_layer_sits_below_the_venv_copies(self):
        """COLMAP's runtime packages must not live in the big shared apt
        layer at the top of the runtime stage.

        They did, once, and it cost a 15-minute push. Layer invalidation
        cascades forward within a stage, so adding one package up there
        re-runs all six `COPY --from=python-builder /opt/venv_*`, the
        body2colmap layer and the application copy behind them. The COPYs
        dedupe at the registry (unchanged bytes, same blob digest) but the
        re-executed RUN layers cannot — their output is not
        byte-reproducible — so a 1.25 GB apt layer goes up the wire again.

        And it has to sit ABOVE the brush COPYs, which is the other half:
        below them, a bumped BRUSH_REF would invalidate this apt RUN and
        push its ~450 MB with every brush bump.
        """
        colmap_pkg = "libceres4t64"
        venv_base = "COPY --from=python-builder /opt/venv_base"
        brush_copy = "COPY --from=brush-builder /out-brush /usr/local/bin/brush"

        for marker in (colmap_pkg, venv_base, brush_copy):
            self.assertIn(marker, self.text)

        self.assertGreater(
            self.text.index(colmap_pkg), self.text.index(venv_base),
            "COLMAP's apt packages are above the venv copies again — a change "
            "to them now re-runs every layer behind them and re-pushes the "
            "runtime stage's RUN layers",
        )
        self.assertLess(
            self.text.index(colmap_pkg), self.text.index(brush_copy),
            "COLMAP's apt packages moved below the brush COPYs — a BRUSH_REF "
            "bump would now invalidate them and re-push ~450 MB",
        )

    def test_the_onnx_runtime_colmap_gets_matches_the_image_s_cuda(self):
        """COLMAP's own FETCH_ONNX takes the gpu_cuda12 build when CUDA is
        on. This image is CUDA 13, and the CUDA 12 libraries are in neither
        the base image nor torch's bundled nvidia-* wheels — ORT throws when
        the provider will not load, with no CPU fallback outside CoreML, so
        the mismatch would be a crash inside the step on the pod. The
        override has to name cuda13, and the base images have to agree with
        it.
        """
        self.assertIn("gpu_cuda13", self.text)
        self.assertIn("FETCHCONTENT_SOURCE_DIR_ONNXRUNTIME", self.text)

        bases = re.findall(r"^FROM nvidia/cuda:(\S+?)-", self.text, re.M)
        self.assertTrue(bases, "no nvidia/cuda base images found")
        for base in bases:
            self.assertTrue(
                base.startswith("13."),
                f"a CUDA {base} base image alongside a gpu_cuda13 ONNX Runtime: "
                f"the provider links libcudart.so.13 and will not load",
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

    def test_prefetch_defaults_off_locally_but_on_in_the_image(self):
        """The one deliberate divergence between this file and a pod.

        `docker compose up` on a workstation is normally "let me look at the
        UI"; pulling ~65 GB of checkpoints as a side effect of that is a
        poor trade. On a pod it stays on, so the entrypoint must not have
        its own default that disagrees.
        """
        compose = yaml.safe_load(COMPOSE.read_text())
        environment = compose["services"]["pipeline"]["environment"]
        self.assertTrue(
            any(e.startswith("B2C_PREFETCH=") for e in environment),
            "docker-compose.yml does not pin B2C_PREFETCH, so a local `up` would "
            "start a ~65 GB download",
        )
        self.assertIn(
            '"${B2C_PREFETCH:-1}"', ENTRYPOINT.read_text(),
            "the entrypoint no longer defaults prefetching ON, which is what a pod needs",
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
