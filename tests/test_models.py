"""Prefetching checkpoints, and deciding what a run has to wait for.

The point of the module under test is that a run never stalls mid-pipeline
on a download. Two things have to hold for that to be true, and both are
easy to get subtly wrong:

  * "already present" must be answered by asking the loader, not by a
    marker this code wrote — otherwise a network volume warmed by an
    earlier pod, or by hand, looks completely cold;
  * blocking must be scoped to the workflow, or a local-smoke run waits on
    52 GB of Wan2.2 it never touches.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import models
from pipeline.cli import resolve_workflow
from pipeline.workflow import WorkflowSpec


class _OnAVolume:
    """Point B2C_DATA_DIR at a scratch directory for the test's duration."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("B2C_DATA_DIR")
        os.environ["B2C_DATA_DIR"] = self.tmp.name
        return Path(self.tmp.name)

    def __exit__(self, *exc):
        if self.previous is None:
            os.environ.pop("B2C_DATA_DIR", None)
        else:
            os.environ["B2C_DATA_DIR"] = self.previous
        self.tmp.cleanup()
        return False


class TestRegistry(unittest.TestCase):
    def test_every_source_is_wired_to_a_real_step(self):
        from pipeline.registry import STEP_REGISTRY

        import pipeline.steps  # noqa: F401

        for key, source in models.registry().items():
            with self.subTest(model=key):
                self.assertTrue(source.steps, f"{key} names no steps")
                for step in source.steps:
                    self.assertIn(
                        step, STEP_REGISTRY,
                        f"model '{key}' claims to be needed by unregistered step '{step}'",
                    )

    def test_every_model_step_has_a_source(self):
        """A step that downloads weights but isn't in the registry would be
        skipped by the prefetch and stall a run mid-pipeline — the exact
        failure this module exists to prevent."""
        covered = {step for source in models.registry().values() for step in source.steps}
        for step in ("rmbg", "sapiens2_lite", "sam3d_body", "wan22_vace_denoise",
                     "seedvr2", "detect_face_landmarks"):
            with self.subTest(step=step):
                self.assertIn(step, covered)

    def test_sources_agree_with_the_steps_own_constants(self):
        """The registry must not carry its own copy of a checkpoint name."""
        from pipeline.steps.rmbg import DEFAULT_CHECKPOINT as RMBG
        from pipeline.steps.wan22_vace_denoise import DEFAULT_CHECKPOINT as WAN22

        sources = models.registry()
        self.assertIn(RMBG, sources["rmbg"].description)
        self.assertIn(WAN22, sources["wan22"].description)


class TestRequiredForSteps(unittest.TestCase):
    def test_blocking_is_scoped_to_the_workflow(self):
        cases = {
            "roundtrip_example": set(),
            "fast_helical_local_smoke": {"rmbg", "sapiens2"},
            "fast_helical_full": {"rmbg", "sapiens2", "wan22", "wan22_lora", "seedvr2"},
        }
        for workflow, expected in cases.items():
            with self.subTest(workflow=workflow):
                spec = WorkflowSpec.from_yaml(resolve_workflow(workflow))
                self.assertEqual(
                    set(models.required_for_steps(s.step for s in spec.steps)), expected
                )

    def test_the_smoke_workflow_does_not_wait_on_wan22(self):
        """It skips both denoise passes on purpose; waiting on 52 GB it
        never touches would defeat the point of having a smoke test."""
        spec = WorkflowSpec.from_yaml(resolve_workflow("fast_helical_local_smoke"))
        self.assertNotIn("wan22", models.required_for_steps(s.step for s in spec.steps))


class TestReadiness(unittest.TestCase):
    def test_a_warm_volume_with_no_markers_is_detected(self):
        """The case a marker-only check gets wrong: weights already on the
        volume from an earlier pod, an older image, or a hand-run
        envs/wan22/setup.sh."""
        with _OnAVolume():
            source = models.registry()["rmbg"]
            self.assertFalse(models.is_ready("rmbg"))

            with mock.patch.object(models, "registry",
                                   return_value={"rmbg": source.__class__(
                                       **{**source.__dict__, "probe": lambda: True})}):
                self.assertTrue(models.is_ready("rmbg"))
                # ...and the expensive probe is not repeated afterwards.
                self.assertTrue((models._ready_dir() / "rmbg.json").exists())

    def test_a_probe_that_raises_is_treated_as_absent(self):
        """A probe must never be the thing that fails a run."""
        with _OnAVolume():
            source = models.registry()["rmbg"]

            def boom():
                raise RuntimeError("network on fire")

            with mock.patch.object(models, "registry",
                                   return_value={"rmbg": source.__class__(
                                       **{**source.__dict__, "probe": boom})}):
                self.assertFalse(models.is_ready("rmbg"))

    def test_marker_short_circuits_the_probe(self):
        with _OnAVolume():
            probed = []
            source = models.registry()["mediapipe"]

            def counting_probe():
                probed.append(1)
                return False

            models.mark_ready("mediapipe", "somewhere")
            with mock.patch.object(models, "registry",
                                   return_value={"mediapipe": source.__class__(
                                       **{**source.__dict__, "probe": counting_probe})}):
                self.assertTrue(models.is_ready("mediapipe"))
            self.assertEqual(probed, [], "the marker should have short-circuited the probe")


class TestPrefetch(unittest.TestCase):
    def test_one_failure_does_not_abort_the_rest(self):
        """A pod that cannot reach a gated repo should still pull the
        others and still come up."""
        with _OnAVolume():
            calls = []

            def good():
                calls.append("good")
                return "/somewhere"

            def bad():
                calls.append("bad")
                raise RuntimeError("401")

            template = models.registry()["rmbg"]
            fake = {
                "bad": template.__class__(**{**template.__dict__, "key": "bad",
                                             "fetch": bad, "probe": lambda: False}),
                "good": template.__class__(**{**template.__dict__, "key": "good",
                                              "fetch": good, "probe": lambda: False}),
            }
            with mock.patch.object(models, "registry", return_value=fake):
                status = models.prefetch(["bad", "good"])

            self.assertEqual(status["bad"]["status"], models.FAILED)
            self.assertEqual(status["good"]["status"], models.READY)
            self.assertIn("good", calls)

    def test_unknown_key_is_refused_up_front(self):
        with _OnAVolume():
            with self.assertRaises(KeyError):
                models.prefetch(["not_a_model"])

    def test_wait_returns_immediately_when_everything_is_present(self):
        with _OnAVolume():
            with mock.patch.object(models, "is_ready", return_value=True):
                models.wait_until_ready(["wan22"], timeout=1.0)

    def test_wait_raises_a_clear_error_when_a_model_failed(self):
        with _OnAVolume():
            models._write_status({"wan22": {"status": models.FAILED, "detail": "gated"}})
            with mock.patch.object(models, "is_ready", return_value=False):
                with self.assertRaises(models.ModelsUnavailable) as caught:
                    models.wait_until_ready(["wan22"], timeout=1.0, poll=0.05)
            self.assertIn("gated", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
