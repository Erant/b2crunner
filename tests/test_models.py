"""Prefetching checkpoints, and deciding what a run has to wait for.

The point of the module under test is that a run never stalls mid-pipeline
on a download. Two things have to hold for that to be true, and both are
easy to get subtly wrong:

  * "already present" must be answered by asking the loader, not by a
    marker this code wrote — otherwise a network volume warmed by an
    earlier pod, or by hand, looks completely cold;
  * blocking must be scoped to the workflow, or a local-smoke run waits on
    the ~47 GB of Wan2.2 weights it never touches.
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
        from pipeline.steps.sam3d_body import DEFAULT_FOV_CHECKPOINT_REPO as MOGE
        from pipeline.steps.wan22_vace_denoise import DEFAULT_CHECKPOINT as WAN22
        from pipeline.steps.wan22_vace_denoise import DEFAULT_FP8_REPO

        sources = models.registry()
        self.assertIn(RMBG, sources["rmbg"].description)
        self.assertIn(WAN22, sources["wan22"].description)
        self.assertIn(DEFAULT_FP8_REPO, sources["wan22_fp8"].description)
        self.assertIn(MOGE, sources["moge2"].description)


class TestWan22WeightSplit(unittest.TestCase):
    """The denoise step's weights come from two repos, and only two repos.

    The transformers are pre-quantized fp8 from
    silveroxides/Wan_2.2-fp8_scaled_hybrid; the base diffusers repo
    supplies everything else. The bf16 transformers there — 34.68 GB EACH,
    measured — must never be pulled again: diffusers skips downloading a
    component passed to `from_pretrained` directly, so they would be 69 GB
    of pod download for weights nothing opens.
    """

    def test_the_base_repo_no_longer_supplies_transformers(self):
        self.assertNotIn("transformer/*", models.WAN22_ALLOW_PATTERNS)
        self.assertNotIn("transformer_2/*", models.WAN22_ALLOW_PATTERNS)

    def test_but_the_transformer_config_is_still_fetched(self):
        """pipeline/wan_fp8.py's load_config needs the model geometry to
        instantiate WanVACETransformer3DModel, and gets it from the base
        repo's transformer/config.json — kilobytes. Dropping it with the
        rest of transformer/* would break every load."""
        self.assertIn("transformer/config.json", models.WAN22_ALLOW_PATTERNS)

    def test_the_fp8_experts_have_their_own_source(self):
        source = models.registry()["wan22_fp8"]
        self.assertEqual(source.steps, ("wan22_vace_denoise",))

    def test_the_denoise_total_reflects_the_fp8_download(self):
        """approx_gb is what the prefetch tells a fresh pod it is about to
        download, so it has to track what the patterns actually pull:
        17.58 x 2 fp8 + 11.89 base + 1.2 LoRA, not the old 81 GB."""
        registry = models.registry()
        total = sum(registry[key].approx_gb
                    for key in models.required_for_steps(["wan22_vace_denoise"]))
        self.assertAlmostEqual(total, 48.3, delta=1.5)

    def test_the_registry_fetches_the_files_the_step_loads(self):
        """Same rule as the checkpoint names above: one source of truth for
        which two files these are, or the prefetch warms a cache the step
        then misses."""
        from pipeline.steps import wan22_vace_denoise as step

        asked = []
        with mock.patch.object(step, "resolve_fp8_checkpoint",
                               side_effect=lambda name, *a, **kw: asked.append((name, a, kw)) or "/cache/f"):
            models._fetch_wan22_fp8()
            self.assertTrue(models._probe_wan22_fp8())

        self.assertEqual([name for name, _, _ in asked],
                         [step.DEFAULT_FP8_HIGH, step.DEFAULT_FP8_LOW] * 2)
        self.assertEqual([a[0] for _, a, _ in asked], [step.DEFAULT_FP8_REPO] * 4)
        # The probe must not touch the network; the fetch must.
        self.assertEqual([kw.get("local_files_only", False) for _, _, kw in asked],
                         [False, False, True, True])

    def test_a_probe_that_cannot_find_the_files_says_so(self):
        from pipeline.steps import wan22_vace_denoise as step

        def missing(*a, **kw):
            raise OSError("not in the local cache")

        with mock.patch.object(step, "resolve_fp8_checkpoint", side_effect=missing):
            self.assertFalse(models._probe_wan22_fp8())


class TestRequiredForSteps(unittest.TestCase):
    def test_blocking_is_scoped_to_the_workflow(self):
        cases = {
            # sapiens2_pointmap joined this set when the bootstrap started
            # building a photo-to-splat shell (pointmap_splat): 6.5 GB more
            # that a fresh pod blocks on before the run starts, on top of
            # the 6.2 the normal head already costs. Both are Sapiens2 1b
            # repos and neither substitutes for the other.
            "fast_helical_native": {"rmbg", "sapiens2", "sapiens2_pointmap",
                                    "sam3dbody", "moge2",
                                    "wan22", "wan22_fp8", "wan22_lora",
                                    "seedvr2", "mediapipe"},
            "fast_helical_full": {"rmbg", "sapiens2", "wan22", "wan22_fp8",
                                  "wan22_lora", "seedvr2"},
        }
        for workflow, expected in cases.items():
            with self.subTest(workflow=workflow):
                spec = WorkflowSpec.from_yaml(resolve_workflow(workflow))
                self.assertEqual(
                    set(models.required_for_steps(s.step for s in spec.steps)), expected
                )

    def test_run_upscale_false_does_not_wait_on_seedvr2(self):
        """`run_upscale: false` is what fast_helical.yaml used to be. The
        prefetch scans enabled_steps(), so with the upscale gated off,
        blocking on the upscaler's 6 GB before the run starts would defeat
        the point of turning it off."""
        spec = WorkflowSpec.from_yaml(resolve_workflow("fast_helical_full"))
        spec.globals["run_upscale"] = False
        self.assertNotIn(
            "seedvr2",
            models.required_for_steps(s.step for s in spec.enabled_steps()),
        )

    def test_a_when_skipped_step_is_not_waited_on(self):
        """The prefetch reads enabled_steps(), so switching an output off
        also drops whatever only that output needed."""
        spec = WorkflowSpec.from_yaml(resolve_workflow("fast_helical_full"))
        spec.globals["export_colmap"] = False
        spec.globals["export_ply"] = False
        # export_normals is the last sapiens2 use outside stage 2, so this
        # only proves the plumbing; the assertion that matters is that the
        # skipped steps are gone from what gets scanned at all.
        enabled = {s.id for s in spec.enabled_steps()}
        self.assertNotIn("export_colmap", enabled)
        self.assertNotIn("train_final_splat", enabled)
        self.assertEqual(
            set(models.required_for_steps(s.step for s in spec.enabled_steps())),
            {"rmbg", "sapiens2", "wan22", "wan22_fp8", "wan22_lora", "seedvr2"},
        )


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
