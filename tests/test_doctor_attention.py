"""doctor's attention check — the one answer that silently degrades.

`wan22_vace_denoise` asks diffusers for a SageAttention backend and, if
that raises for any of three separate reasons, logs a warning and carries
on with PyTorch native SDPA. A run that took the slow path is
indistinguishable from one that took the fast path except by wall clock,
which is why "which kernel is this actually going to use" is worth a line
in the report rather than a grep through a run log.

The probe runs in the wan22 venv (that is where SageAttention lives), so
these drive it through a stub interpreter rather than needing one.
"""

from __future__ import annotations

import json
import subprocess
import unittest
import unittest.mock

from pipeline import doctor


def _probe_result(payload: dict, returncode: int = 0, stderr: str = ""):
    """What `_run` would return for an interpreter that printed `payload`."""
    return subprocess.CompletedProcess(
        args=["python", "-c", "..."],
        returncode=returncode,
        stdout=(
            "some unrelated import noise on stdout\n"
            "B2C_ATTENTION " + json.dumps(payload) + "\n"
        ),
        stderr=stderr,
    )


HEALTHY = {
    "torch": "2.13.0",
    "gpu": "NVIDIA L40S",
    "sm": "sm_89",
    "sageattention": "2.2.0",
    "triton": "3.2.0",
    "diffusers": "0.40.0",
    "selected": "_sage_qk_int8_pv_fp16_triton",
}

ENVS = {"wan22": {"python_bin": "/opt/venv_wan22/bin/python"}}


class TestCheckAttention(unittest.TestCase):
    def _check(self, result, envs=ENVS, exists=True):
        with unittest.mock.patch.object(doctor, "_run", return_value=result), \
             unittest.mock.patch.object(doctor.Path, "exists", lambda self: exists):
            return doctor.check_attention(envs)

    def test_it_reports_the_selected_kernel(self):
        check = self._check(_probe_result(HEALTHY))

        self.assertEqual(check.status, doctor.OK)
        self.assertEqual(check.detail, "_sage_qk_int8_pv_fp16_triton")
        joined = "\n".join(check.lines)
        self.assertIn("NVIDIA L40S sm_89", joined)
        self.assertIn("sageattention: 2.2.0", joined)
        self.assertIn("triton: 3.2.0", joined)

    def test_it_also_names_the_upscalers_backend(self):
        """seedvr2 has no auto-selection at all, and someone reading this
        report is asking about the whole run, not just the denoise."""
        self.assertIn(
            "seedvr2", "\n".join(self._check(_probe_result(HEALTHY)).lines)
        )

    def test_sdpa_is_a_perfectly_good_answer(self):
        payload = dict(HEALTHY, sm="sm_75", selected="none (PyTorch native SDPA)")
        check = self._check(_probe_result(payload))

        self.assertEqual(check.status, doctor.OK)
        self.assertIn("SDPA", check.detail)

    def test_sage_selected_but_not_installed_warns(self):
        """The case this check exists for: the step will fall back at load
        time, and nothing above WARNING says so."""
        payload = dict(HEALTHY, sageattention="MISSING (ModuleNotFoundError)")
        check = self._check(_probe_result(payload))

        self.assertEqual(check.status, doctor.WARN)
        self.assertIn("fall back", "\n".join(check.lines))

    def test_a_triton_kernel_with_no_triton_warns(self):
        payload = dict(HEALTHY, triton="MISSING (ModuleNotFoundError)")
        check = self._check(_probe_result(payload))

        self.assertEqual(check.status, doctor.WARN)
        self.assertIn("Triton", "\n".join(check.lines))

    def test_a_venv_that_cannot_answer_warns_rather_than_failing_twice(self):
        """check_step_venvs already FAILs on a wan22 venv that cannot import
        torch; reporting the same broken venv as a second failure buries
        the one finding under two."""
        payload = dict(HEALTHY, selected="could not resolve: No module named 'torch'")
        self.assertEqual(self._check(_probe_result(payload)).status, doctor.WARN)

    def test_a_probe_that_prints_nothing_usable_fails(self):
        result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Traceback...\nImportError: boom",
        )
        check = self._check(result)

        self.assertEqual(check.status, doctor.FAIL)
        self.assertIn("ImportError: boom", "\n".join(check.lines))

    def test_no_wan22_env_skips(self):
        self.assertEqual(
            doctor.check_attention({}).status, doctor.SKIP
        )

    def test_a_missing_interpreter_fails(self):
        self.assertEqual(
            self._check(_probe_result(HEALTHY), exists=False).status, doctor.FAIL
        )


class TestItIsWiredIn(unittest.TestCase):
    def test_run_checks_includes_it(self):
        """A check nothing calls reports nothing."""
        names = [c.name for c in doctor.run_checks({})]
        self.assertIn("attention", names)


if __name__ == "__main__":
    unittest.main()
