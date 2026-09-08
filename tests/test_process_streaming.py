"""Subprocess output reaches the log while the process is still running.

This is the behaviour the RunPod round depends on: `wan22_vace_denoise`,
`seedvr2` and `sam3d_body` dispatch into their own venvs, and `brush` and
`brush-splat-render` are external binaries. All five used to buffer their
output and reveal it only on failure, so the longest steps in the pipeline
were also the only silent ones.
"""

from __future__ import annotations

import logging
import os
import sys
import unittest

from pipeline.dispatch.subprocess_python import (
    SubprocessPythonDispatcher,
    _exit_description,
)
from pipeline.proc import ProcessFailed, stream_command


class CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class _CaptureLogs:
    """Attach a handler to the root logger at INFO for the duration."""

    def __enter__(self):
        self.handler = CapturingHandler()
        self.root = logging.getLogger()
        self.previous = self.root.level
        self.root.setLevel(logging.INFO)
        self.root.addHandler(self.handler)
        return self.handler

    def __exit__(self, *exc):
        self.root.removeHandler(self.handler)
        self.root.setLevel(self.previous)
        return False


class TestStreamCommand(unittest.TestCase):
    def test_output_is_logged_not_swallowed(self):
        with _CaptureLogs() as handler:
            stream_command(
                [sys.executable, "-c", "print('alpha'); print('beta')"], log_name="probe"
            )
        self.assertIn("alpha", handler.messages)
        self.assertIn("beta", handler.messages)

    def test_nonzero_exit_raises_with_the_tail(self):
        with _CaptureLogs():
            with self.assertRaises(ProcessFailed) as caught:
                stream_command(
                    [sys.executable, "-c", "print('last words'); raise SystemExit(3)"],
                    log_name="probe",
                )
        message = str(caught.exception)
        self.assertIn("exit code 3", message)
        self.assertIn("last words", message)

    def test_missing_binary_says_so_and_includes_the_hint(self):
        with self.assertRaises(RuntimeError) as caught:
            stream_command(
                ["definitely-not-a-real-binary-xyz"],
                log_name="probe",
                not_found_hint="Build it from the brush repo.",
            )
        message = str(caught.exception)
        self.assertIn("not found on PATH", message)
        self.assertIn("Build it from the brush repo", message)

    def test_a_flood_of_output_is_throttled_but_reported(self):
        """A progress bar repainting for half an hour must not fill the volume."""
        with _CaptureLogs() as handler:
            stream_command(
                [sys.executable, "-c", "for i in range(2000): print('tick', i)"],
                log_name="probe",
            )
        self.assertLess(
            len(handler.messages), 2000,
            "2000 lines of progress output were relayed verbatim; the token bucket "
            "in pipeline/proc.py is not limiting anything",
        )
        self.assertTrue(
            any("progress lines skipped" in m for m in handler.messages),
            "output was dropped without saying so",
        )

    def test_throttling_can_be_turned_off(self):
        with _CaptureLogs() as handler:
            stream_command(
                [sys.executable, "-c", "for i in range(300): print('tick', i)"],
                log_name="probe",
                throttle=False,
            )
        ticks = [m for m in handler.messages if m.startswith("tick")]
        self.assertEqual(len(ticks), 300)


class TestChildEnvironment(unittest.TestCase):
    def test_env_config_is_layered_onto_the_real_environment(self):
        """An envs.yaml `env:` used to REPLACE the environment wholesale.

        That silently dropped PATH, HF_HOME, HF_TOKEN and TMPDIR, and the
        resulting failure (a gated model 401ing, a download filling the
        container's overlay) pointed nowhere near the config that caused it.
        """
        os.environ["B2C_TEST_SENTINEL"] = "from-parent"
        try:
            dispatcher = SubprocessPythonDispatcher(
                python_bin=sys.executable, env={"B2C_TEST_EXTRA": "from-config"}
            )
            env = dispatcher._child_env()
        finally:
            del os.environ["B2C_TEST_SENTINEL"]

        self.assertEqual(env["B2C_TEST_SENTINEL"], "from-parent")
        self.assertEqual(env["B2C_TEST_EXTRA"], "from-config")
        self.assertIn("PATH", env)

    def test_env_config_overrides_win(self):
        os.environ["B2C_TEST_SENTINEL"] = "from-parent"
        try:
            dispatcher = SubprocessPythonDispatcher(
                python_bin=sys.executable, env={"B2C_TEST_SENTINEL": "from-config"}
            )
            self.assertEqual(dispatcher._child_env()["B2C_TEST_SENTINEL"], "from-config")
        finally:
            del os.environ["B2C_TEST_SENTINEL"]

    def test_missing_interpreter_names_the_envs_registry(self):
        dispatcher = SubprocessPythonDispatcher(python_bin="/nonexistent/bin/python")
        with self.assertRaises(RuntimeError) as caught:
            dispatcher.run("save_dataset", {}, {})
        self.assertIn("envs.yaml", str(caught.exception))


class TestExitDescription(unittest.TestCase):
    """`exit -9` names nothing; the signal is the diagnosis.

    A child killed by a signal runs no cleanup, so the last line of its
    output is often `multiprocessing.resource_tracker`'s leaked-semaphore
    warning — written by a helper process that outlived it — and that line
    reads like a cause. Naming the signal is what stops the next reader
    chasing the warning.
    """

    def test_a_normal_exit_code_is_left_alone(self):
        self.assertEqual(_exit_description(0), "exit 0")
        self.assertEqual(_exit_description(1), "exit 1")

    def test_sigkill_is_named_and_points_at_the_oom_killer(self):
        message = _exit_description(-9)
        self.assertIn("SIGKILL", message)
        self.assertIn("OOM", message)
        self.assertIn("dmesg", message)

    def test_a_native_crash_is_named(self):
        self.assertIn("SIGSEGV", _exit_description(-11))
        self.assertIn("SIGABRT", _exit_description(-6))

    def test_an_unknown_signal_still_says_it_was_a_signal(self):
        self.assertIn("signal", _exit_description(-999))

    def test_never_started_has_a_word_for_it(self):
        self.assertEqual(_exit_description(None), "never started")


if __name__ == "__main__":
    unittest.main()
