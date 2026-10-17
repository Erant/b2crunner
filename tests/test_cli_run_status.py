"""A `pipeline.cli run` publishes a status file the way a web-UI run's
worker does, and names its process so a reader can tell a killed run from
a live one.

Without one, a CLI run was a bare `unknown` directory to `discover_runs`,
which counts as over — and Clear runs deleted it while it worked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from pipeline import run_state
from pipeline.run_state import HEARTBEAT_STALE, RunState, owner_alive, process_token, run_owner_alive
from tests.helpers import REPO_ROOT, orbit_dataset
from tests.test_run_worker import ENVS_PATH, _write_workflow


class TestCliPublishesAStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "initial"
        orbit_dataset(n_frames=5, n_points=20).to_disk(self.dataset)
        self.workflow = self.root / "save.yaml"
        _write_workflow(self.workflow, self.root / "checkpoint")

    def tearDown(self):
        self.tmp.cleanup()

    def _cli(self, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
        env = {
            **os.environ,
            "B2C_DATA_DIR": str(self.root / "data"),
            "B2C_LOG_DIR": str(self.root / "logs"),
        }
        for var in ("B2C_RUN_JOBS_DIR", "B2C_OUTPUT_DIR"):
            env.pop(var, None)
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cli", "run", str(self.workflow),
             "--dataset", str(self.dataset), "--envs", ENVS_PATH,
             "--run-name", "from-the-cli", "--no-wait-for-models", *extra],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        status = self.root / "data" / "run_jobs" / "from-the-cli.status.json"
        self.assertTrue(status.exists(), result.stdout + result.stderr)
        return result, json.loads(status.read_text())

    def test_a_finished_run_says_done_with_its_steps(self):
        result, state = self._cli()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(state["status"], "done")
        self.assertEqual([s["status"] for s in state["steps"]], ["done"])
        self.assertEqual(state["output_dir"], str(self.root / "data" / "output" / "from-the-cli"))
        self.assertTrue(state["owner"])
        self.assertGreater(state["finished"], 0)

    def test_a_run_that_never_starts_still_ends_failed(self):
        result, state = self._cli("--param", "no_such_setting=1")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state["status"], "failed")
        self.assertIn("no_such_setting", state["message"])


class TestOwnerAlive(unittest.TestCase):
    def test_this_process_is_alive(self):
        self.assertIs(owner_alive(process_token()), True)

    def test_a_process_that_exited_is_not(self):
        child = subprocess.run(
            [sys.executable, "-c", "from pipeline.run_state import process_token; print(process_token())"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        self.assertIs(owner_alive(child.stdout.strip()), False)

    def test_a_reused_pid_is_not_the_same_process(self):
        boot, namespace, pid, ticks = process_token().split("|")
        self.assertIs(owner_alive("|".join((boot, namespace, pid, str(int(ticks) - 1)))), False)

    def test_another_boot_is_gone(self):
        _, namespace, pid, ticks = process_token().split("|")
        self.assertIs(owner_alive("|".join(("another-boot", namespace, pid, ticks))), False)

    def test_another_pid_namespace_cannot_be_told(self):
        boot, _, pid, ticks = process_token().split("|")
        self.assertIsNone(owner_alive("|".join((boot, "pid:[1]", pid, ticks))))

    def test_no_owner_cannot_be_told(self):
        self.assertIsNone(owner_alive(""))

    def test_without_proc_the_pid_is_all_there_is(self):
        with unittest.mock.patch.object(run_state, "_read", lambda path: ""), \
                unittest.mock.patch.object(run_state, "_pid_namespace", lambda: ""), \
                unittest.mock.patch.object(run_state, "_start_ticks", lambda pid: ""):
            self.assertIs(owner_alive(process_token()), True)



class TestHeartbeat(unittest.TestCase):
    """A CLI run in another container cannot be checked by pid, so its status
    file's heartbeat decides: two such runs sat at `running` for days after
    their containers were gone (2026-10-16), unclearable."""

    def _foreign(self) -> str:
        boot, _, pid, ticks = process_token().split("|")
        return "|".join((boot, "pid:[1]", pid, ticks))

    def test_a_fresh_heartbeat_cannot_be_told(self):
        self.assertIsNone(run_owner_alive(self._foreign(), 1000.0, now=1000.0 + HEARTBEAT_STALE - 1))

    def test_a_stale_heartbeat_is_gone(self):
        self.assertIs(run_owner_alive(self._foreign(), 1000.0, now=1000.0 + HEARTBEAT_STALE + 1), False)

    def test_a_checkable_owner_ignores_the_heartbeat(self):
        self.assertIs(run_owner_alive(process_token(), 0.0, now=1e12), True)

    def test_no_owner_cannot_be_told(self):
        self.assertIsNone(run_owner_alive("", 0.0, now=1e12))

    def test_the_writer_touches_the_file_until_it_finishes(self):
        from pipeline.run_worker import _StatusWriter
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.status.json"
            writer = _StatusWriter(RunState(name="r", status="running"), path)
            writer.publish()
            os.utime(path, (1000.0, 1000.0))
            writer.start_heartbeat(interval=0.02)
            deadline = time.time() + 5
            while path.stat().st_mtime == 1000.0 and time.time() < deadline:
                time.sleep(0.01)
            self.assertGreater(path.stat().st_mtime, 1000.0)
            writer.finish("done")
            self.assertTrue(writer._heartbeat.is_set())


if __name__ == "__main__":
    unittest.main()
