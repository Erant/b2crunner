"""The polish run — a second, growth-off brush invocation on the same export.

Measured on the intermediate splat (2026-09-05, docs/intermediate-splat-guide.md):
9000 iterations warm-started from the first run's own .ply moved band-limited
face sharpness from 143 to 161, where 9000 extra iterations of ONE cold run
reached 137. The restart at full mean learning rate is the effect, not the
iteration count, which is why this is a second invocation rather than a bigger
`total_steps`.

Three things carry the weight here. brush resumes from an `init.ply` sitting
in the dataset directory — there is no flag for it — so the link is the whole
handover. The overrides (`--refine-every`, `--growth-stop-iter`,
`--normal-loss-start-iter`) have to be BUILT INTO the argv rather than
appended to the first run's, because clap refuses a repeated flag and the
polish would simply not run. And the export path is unchanged, so the polished
splat lands over the first one and nothing downstream — including the evidence
`render_splat`'s confidence mode gates on — sees anything else.

These check the argv `run()` builds, with the training itself stubbed out.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401


def _inputs(with_normals: bool = False):
    """The smallest batch ColmapExporter will accept: two 8x8 views."""
    from body2colmap.camera import Camera

    cameras = [
        Camera(
            focal_length=(8.0, 8.0),
            image_size=(8, 8),
            principal_point=(4.0, 4.0),
            position=np.array([0.0, 0.0, float(i + 1)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(2)
    ]
    inputs = {
        "cameras": cameras,
        "image_names": ["frame_00001_.png", "frame_00002_.png"],
        "points_3d": (
            np.zeros((4, 3), dtype=np.float32),
            np.zeros((4, 3), dtype=np.uint8),
        ),
        "images": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
        "masks": [np.ones((8, 8), dtype=np.float32) for _ in range(2)],
    }
    if with_normals:
        inputs["normal_maps"] = [
            np.zeros((8, 8, 3), dtype=np.float32) for _ in range(2)
        ]
    return inputs


def _runs(with_normals: bool = False, **overrides):
    """Every brush invocation `run()` makes, and what init.ply was at the time.

    The link is checked from inside the fake because the COLMAP export is a
    TemporaryDirectory `run()` deletes on the way out — by the time this
    returns there is nothing left to look at.
    """
    step_class = get_step_class("brush")
    step = step_class()
    calls = []

    def fake_run_brush(cmd, ply_path, colmap_dir=None):
        init = Path(colmap_dir) / "init.ply"
        calls.append({
            "cmd": list(cmd),
            "init": init.resolve() if init.is_symlink() else None,
        })
        Path(ply_path).write_text(f"ply {len(calls)}\n")

    step._run_brush = fake_run_brush
    with tempfile.TemporaryDirectory() as tmp:
        params = step_class.resolve_params({"export_dir": tmp, **overrides})
        step.run(_inputs(with_normals), params)
    return calls


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class TestThePolishIsOptional(unittest.TestCase):
    def test_it_is_off_by_default(self):
        """A training nobody has measured it on — `train_final_splat` — is
        the case the default is for."""
        self.assertEqual(get_step_class("brush").declared_params()["polish_steps"].default, 0)
        calls = _runs()
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["init"])

    def test_the_first_run_never_stops_growth(self):
        """--growth-stop-iter is the polish's flag alone; the main training
        keeps brush's own schedule."""
        self.assertNotIn("--growth-stop-iter", _runs()[0]["cmd"])
        self.assertNotIn("--growth-stop-iter", _runs(polish_steps=9000)[0]["cmd"])


class TestThePolishRun(unittest.TestCase):
    def setUp(self):
        self.calls = _runs(with_normals=True, polish_steps=9000, total_steps=30000)
        self.first, self.second = (call["cmd"] for call in self.calls)

    def test_it_is_a_second_invocation(self):
        self.assertEqual(len(self.calls), 2)

    def test_it_warm_starts_from_the_first_run_s_export(self):
        """brush has no resume flag: it initialises from the `init.ply` of
        the dataset it is handed, so this link is the handover."""
        self.assertIsNone(self.calls[0]["init"])
        self.assertEqual(
            self.calls[1]["init"],
            Path(_value(self.first, "--export-path")).resolve()
            / _value(self.first, "--export-name"),
        )

    def test_it_runs_for_the_iterations_asked_for_and_exports_them(self):
        self.assertEqual(_value(self.first, "--total-train-iters"), "30000")
        self.assertEqual(_value(self.second, "--total-train-iters"), "9000")
        self.assertEqual(_value(self.second, "--export-every"), "9000")

    def test_it_grows_nothing(self):
        """The point of the restart is more optimisation of the splats that
        are there, not more splats."""
        self.assertEqual(_value(self.second, "--growth-stop-iter"), "0")
        self.assertGreater(int(_value(self.second, "--refine-every")), 9000)

    def test_the_normal_loss_is_on_from_the_first_step(self):
        """Its warm-up exists so rough geometry can form before it is
        penalised for normals; by now the geometry is 30000 steps old."""
        self.assertEqual(_value(self.first, "--normal-loss-start-iter"), "5000")
        self.assertEqual(_value(self.second, "--normal-loss-start-iter"), "0")

    def test_it_exports_over_the_first_ply(self):
        """Nothing downstream is told the polish happened, so it must not be
        able to tell."""
        for flag in ("--export-path", "--export-name"):
            self.assertEqual(_value(self.first, flag), _value(self.second, flag))

    def test_it_carries_the_evidence_the_confidence_render_reads(self):
        """The .ply the re-render gates on is this run's, so the ev_* block
        has to be measured here — the first run's is overwritten."""
        self.assertIn("--export-evidence", self.second)

    def test_neither_argv_repeats_a_flag(self):
        """clap refuses a repeated argument, so a polish built by appending
        overrides to the first run's argv would not run at all — which is
        why the two are built rather than patched."""
        for cmd in (self.first, self.second):
            flags = [word for word in cmd if word.startswith("--")]
            self.assertEqual(len(flags), len(set(flags)), cmd)


class TestMatchAlphaWeight(unittest.TestCase):
    """How hard the silhouette is fitted to the mask. 0.5 measured 17% fewer
    dark wedges at no cost in sharpness; brush's own default is 0.1."""

    def test_brush_s_own_default_is_what_is_passed(self):
        self.assertEqual(
            get_step_class("brush").declared_params()["match_alpha_weight"].default, 0.1
        )
        self.assertEqual(_value(_runs()[0]["cmd"], "--match-alpha-weight"), "0.1")

    def test_it_reaches_both_runs(self):
        for call in _runs(polish_steps=1000, match_alpha_weight=0.5):
            self.assertEqual(_value(call["cmd"], "--match-alpha-weight"), "0.5")


if __name__ == "__main__":
    unittest.main()
