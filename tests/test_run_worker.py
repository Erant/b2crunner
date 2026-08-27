"""`pipeline.run_worker` end to end: one real workflow, run as a real OS
subprocess, watched through the same status.json/log-file mechanism
`pipeline.gpu_scheduler.GpuScheduler` uses to poll it.

This is not `tests/test_runner_events.py` again — that already covers
`WorkflowRunner`'s own event ordering with in-process fakes. What only a
real subprocess invocation can prove is the IPC boundary itself: a job
crosses in as JSON, a terminal status crosses back out as JSON, and a log
file lands where it says it did. `save_dataset` is a plain disk write with
no model or GPU dependency, so this exercises the whole `run_worker.main()`
path (spec load, override application, the model-prefetch gate finding
nothing to wait for, dataset load, `WorkflowRunner`, status publishing, log
setup) without needing a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.run_state import RunJob

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "cyber_6f" / "initial"
# `RunJob.envs_path` defaults to "" only as a dataclass placeholder — every
# real caller (webui.py's on_start, cli.py's --envs default) always supplies
# a real path, same as here. `save_dataset` reads no env config itself
# (dispatch: in_process), but load_envs() still needs a real file to parse.
ENVS_PATH = str(REPO_ROOT / "pipeline" / "envs" / "envs.yaml")


def _write_workflow(path: Path, checkpoint_dir: Path) -> None:
    path.write_text(
        "name: _test_save_only\n"
        "globals: {}\n"
        "steps:\n"
        "  - id: checkpoint\n"
        "    step: save_dataset\n"
        "    dispatch: in_process\n"
        "    inputs:\n"
        "      dataset: dataset\n"
        "    params:\n"
        f"      directory: {checkpoint_dir.as_posix()!r}\n"
        "    outputs:\n"
        "      directory_path: out.directory_path\n"
    )


@unittest.skipUnless(DATASET.exists(), "cyber_6f/initial fixture not present")
class TestRunWorkerSubprocess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, run_name: str, gpu_index: str) -> dict:
        out_dir = self.root / f"{run_name}-out"
        workflow_path = self.root / f"{run_name}.yaml"
        _write_workflow(workflow_path, out_dir / "checkpoint")

        job = RunJob(
            run_name=run_name, workflow_name=run_name,
            workflow_path=str(workflow_path), output_dir=str(out_dir),
            envs_path=ENVS_PATH, dataset_dir=str(DATASET),
        )
        job_path = self.root / f"{run_name}.job.json"
        status_path = self.root / f"{run_name}.status.json"
        job_path.write_text(json.dumps(job.to_dict()))

        # A stand-in for what GpuScheduler's Popen call does: pin the GPU
        # index in the child's environment, and keep every path this run
        # touches under an isolated volume rather than the real one.
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu_index,
            "B2C_DATA_DIR": str(self.root / "data"),
            "B2C_LOG_DIR": str(self.root / "logs"),
        }
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.run_worker",
             "--job", str(job_path), "--status", str(status_path)],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(status_path.exists())
        return json.loads(status_path.read_text())

    def test_a_run_completes_and_publishes_a_terminal_status(self):
        state = self._run("run-a", gpu_index="0")

        self.assertEqual(state["status"], "done")
        self.assertEqual(state["total"], 1)
        self.assertEqual(len(state["steps"]), 1)
        self.assertEqual(state["steps"][0]["status"], "done")
        self.assertTrue(state["log_path"])
        self.assertTrue(Path(state["log_path"]).exists())

    def test_two_runs_pinned_to_different_gpu_indices_do_not_collide(self):
        """Not proof of CUDA isolation itself — that needs real hardware,
        see the plan's verification section — only that two runs shaped
        like GpuScheduler's concurrent slots never share a path."""
        state_a = self._run("run-a", gpu_index="0")
        state_b = self._run("run-b", gpu_index="1")

        self.assertNotEqual(state_a["output_dir"], state_b["output_dir"])
        self.assertNotEqual(state_a["log_path"], state_b["log_path"])
        self.assertEqual(state_a["status"], "done")
        self.assertEqual(state_b["status"], "done")

    def test_a_bad_override_fails_the_run_not_the_worker(self):
        """`spec.validate()` runs inside the worker too (webui.py validates
        again before ever queuing the job, but the worker must not trust
        that it always will be the one submitting)."""
        out_dir = self.root / "run-bad-out"
        workflow_path = self.root / "run-bad.yaml"
        _write_workflow(workflow_path, out_dir / "checkpoint")

        job = RunJob(
            run_name="run-bad", workflow_name="run-bad",
            workflow_path=str(workflow_path), output_dir=str(out_dir),
            envs_path=ENVS_PATH, dataset_dir=str(DATASET),
            step_overrides={"checkpoint": {"not_a_real_param": 1}},
        )
        job_path = self.root / "run-bad.job.json"
        status_path = self.root / "run-bad.status.json"
        job_path.write_text(json.dumps(job.to_dict()))

        env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": "0",
            "B2C_DATA_DIR": str(self.root / "data"), "B2C_LOG_DIR": str(self.root / "logs"),
        }
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.run_worker",
             "--job", str(job_path), "--status", str(status_path)],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
        )

        # apply_ui_overrides only filters unknown *global* overrides (a
        # stale param panel after switching workflows); a step-level
        # override is applied unconditionally and then caught by
        # spec.validate(), same as a bad `--param` would be for the CLI.
        # The point of this test is that the failure is reported cleanly —
        # a non-zero exit and a terminal "failed" status naming the bad
        # param — not a hang or an unhandled crash.
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        state = json.loads(status_path.read_text())
        self.assertEqual(state["status"], "failed")
        self.assertIn("not_a_real_param", state["error"])


if __name__ == "__main__":
    unittest.main()
