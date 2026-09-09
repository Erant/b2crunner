"""The alignment loop: warp the training views onto the splat's own consensus.

Measured on the deliverable splat (2026-09-06,
docs/final-splat-alignment-guide.md): the fit is what destroys the detail,
not the upscaler — the generated views disagree with each other about where
texture sits by 1.7-3.6 px, and a photometric loss averages that into a
blurred consensus. Four iterations of render -> flow -> warp -> refit moved
band-limited face sharpness 21.1 -> 23.8 with fidelity RISING alongside it.

Two halves are tested here. The warp itself (pipeline/align.py) is checked
against a synthetic shift, because the properties that matter are geometric:
it must move the frame the way the flow says, it must not damage a frame
that already agrees with its render, and it must respect the cap. The loop
in `steps/brush.py` is checked with the training and the rasteriser stubbed
out, because what can go wrong there is bookkeeping — the argv, the warm
start, and above all WHICH frames each iteration warps: warping a warp turns
a contraction onto one consensus into pairwise merging, which drifts without
bound and is the one bug the guide says to look for.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipeline import align
from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401


def _texture(size: int = 96, seed: int = 0) -> np.ndarray:
    """A frame with enough high-frequency content for DIS to match on."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    smooth = cv2.GaussianBlur(noise, (5, 5), 1.5)
    bgr = np.dstack([smooth, np.roll(smooth, 3, axis=0), np.roll(smooth, 3, axis=1)])
    alpha = np.full((size, size), 255, dtype=np.uint8)
    return np.dstack([bgr, alpha])


class TestTheWarp(unittest.TestCase):
    def test_a_frame_that_agrees_with_its_render_is_untouched(self):
        """The fixed point has to be exactly still. Every iteration
        resamples, and a resample that moves pixels it has no reason to move
        is the bilinear bug the guide names (a single 0.5 px bilinear
        resample of an undamaged face crop drops its score from 738 to 134):
        it would show up as detail bleeding away with the iteration count."""
        frame = _texture()
        warped, stats = align.align_view(
            frame, align._flatten(frame), sigma=6.0, cap=6.0
        )
        np.testing.assert_array_equal(warped, frame)
        self.assertAlmostEqual(stats.mean, 0.0, places=3)

    def test_it_moves_the_frame_onto_the_render(self):
        """The whole point: a frame whose texture sits two pixels off its
        render comes back sitting on it."""
        frame = _texture()
        render = align._flatten(np.roll(frame, 2, axis=1))

        warped, stats = align.align_view(frame, render, sigma=6.0, cap=6.0)

        before = np.abs(align._flatten(frame).astype(float) - render).mean()
        after = np.abs(align._flatten(warped).astype(float) - render).mean()
        self.assertLess(after, before / 2)
        # The measured disagreement is the shift it was given.
        self.assertAlmostEqual(stats.mean, 2.0, delta=0.6)

    def test_the_cap_bounds_the_displacement(self):
        """A bad match must not be allowed to tear the frame: past the cap
        the field is scaled down with its direction kept."""
        frame = _texture()
        render = align._flatten(np.roll(frame, 5, axis=1))

        loose, _ = align.align_view(frame, render, sigma=6.0, cap=6.0)
        capped, stats = align.align_view(frame, render, sigma=6.0, cap=1.0)

        # Both move toward the render; the capped one gets a fifth of the way.
        difference = np.abs(align._flatten(capped).astype(float)
                            - align._flatten(loose).astype(float)).mean()
        self.assertGreater(difference, 1.0)
        # The stats report what was MEASURED, not what was applied, so they
        # are the same either way — that is what makes a p90 pinned at the
        # cap readable as "the cap is binding".
        self.assertAlmostEqual(stats.mean, 5.0, delta=1.0)

    def test_the_background_is_not_dragged_in_over_the_silhouette(self):
        """Outside the matte the two images are the same flat grey, so any
        flow there is noise fitted to nothing."""
        frame = _texture()
        frame[..., 3] = 0
        frame[32:64, 32:64, 3] = 255
        render = align._flatten(np.roll(frame, 3, axis=0))

        warped, _ = align.align_view(frame, render, sigma=6.0, cap=6.0)

        # Far from the subject, nothing moved.
        np.testing.assert_array_equal(warped[:16], frame[:16])

    def test_a_batch_keeps_its_order_and_averages_its_stats(self):
        frames = [_texture(seed=i) for i in range(4)]
        renders = [align._flatten(np.roll(f, 2, axis=1)) for f in frames]

        batch, stats = align.align_views(frames, renders, sigma=6.0, cap=6.0)

        self.assertEqual(len(batch), 4)
        for warped, frame, render in zip(batch, frames, renders):
            one, _ = align.align_view(frame, render, sigma=6.0, cap=6.0)
            np.testing.assert_array_equal(warped, one)
        self.assertAlmostEqual(stats.mean, 2.0, delta=0.6)

    def test_a_render_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            align.align_view(_texture(size=64), np.zeros((32, 32, 3), np.uint8),
                             sigma=6.0, cap=6.0)


def _cameras(count: int, size: int = 8):
    from body2colmap.camera import Camera

    return [
        Camera(
            focal_length=(float(size), float(size)),
            image_size=(size, size),
            principal_point=(size / 2.0, size / 2.0),
            position=np.array([0.0, 0.0, float(i + 1)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(count)
    ]


def _inputs(count: int = 2, size: int = 8, with_normals: bool = False):
    inputs = {
        "cameras": _cameras(count, size),
        "image_names": [f"frame_{i + 1:05d}_.png" for i in range(count)],
        "points_3d": (np.zeros((4, 3), dtype=np.float32),
                      np.zeros((4, 3), dtype=np.uint8)),
        # Distinct per view, so "which frames were warped" is answerable.
        "images": [np.full((size, size, 3), 10 * (i + 1), dtype=np.uint8)
                   for i in range(count)],
        "masks": [np.ones((size, size), dtype=np.float32) for _ in range(count)],
    }
    if with_normals:
        inputs["normal_maps"] = [np.zeros((size, size, 3), dtype=np.float32)
                                 for _ in range(count)]
    return inputs


class _Loop:
    """One stubbed `run()` with the alignment loop wired to fakes.

    Records every brush invocation (its argv, whether an init.ply was
    linked, and the frames sitting in the export at the time), every render
    request, and every batch handed to the warp.
    """

    def __init__(self, inputs=None, **overrides):
        # `pipeline` unless a test says otherwise: the shipped default is
        # `auto`, which asks the trainer binary whether it carries the loop
        # — a real b2ctrain on PATH would then take half these tests down
        # the other branch, and which loop runs would depend on the machine
        # the suite runs on. The backends are covered explicitly in
        # TestTheTrainerBackend.
        overrides.setdefault("align_backend", "pipeline")
        step_class = get_step_class("brush")
        step = step_class()
        self.runs = []
        self.renders = []
        self.aligned = []
        inputs = _inputs() if inputs is None else inputs
        names = inputs["image_names"]

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            init = Path(colmap_dir) / "init.ply"
            images = Path(colmap_dir) / "images"
            self.runs.append({
                "cmd": list(cmd),
                "init": init.is_symlink(),
                "frames": [cv2.imread(str(images / name), cv2.IMREAD_UNCHANGED)
                           for name in names],
            })
            Path(ply_path).write_text(f"ply {len(self.runs)}\n")

        def fake_render(ply_path, cameras, image_names, *, render_path):
            self.renders.append({"ply": Path(ply_path).read_text(),
                                 "cameras": len(cameras),
                                 "render_path": render_path})
            return [np.full((c.height, c.width, 3), 128, dtype=np.uint8)
                    for c in cameras]

        def fake_align(frames, renders, *, sigma, cap):
            """Stamps the iteration number into the frames it returns, so
            the export can be told apart from the originals — and warping a
            warp would be visible as a stamp coming back in."""
            self.aligned.append({"frames": [f.copy() for f in frames],
                                 "sigma": sigma, "cap": cap})
            stamped = []
            for frame in frames:
                out = frame.copy()
                out[0, 0, 0] = 100 + len(self.aligned)
                stamped.append(out)
            views = [align.AlignStats(1.0 + i / 10, 2.0 + i / 10)
                     for i in range(len(frames))]
            return stamped, align.BatchStats(1.0, 2.0, views)

        step._run_brush = fake_run_brush
        step._render_training_views = fake_render
        with mock.patch("pipeline.steps.brush.align_views", fake_align):
            with tempfile.TemporaryDirectory() as tmp:
                params = step_class.resolve_params({"export_dir": tmp, **overrides})
                step.run(inputs, params)


def _value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class TestItIsOnByDefault(unittest.TestCase):
    def test_four_iterations_is_the_measured_saturation_point(self):
        """+1.2, +0.6, +0.5, +0.4, and +0.2 for a fifth and a sixth."""
        self.assertEqual(
            get_step_class("brush").declared_params()["align_iters"].default, 4
        )

    def test_it_is_one_training_plus_one_run_per_iteration(self):
        loop = _Loop(align_iters=3)
        self.assertEqual(len(loop.runs), 4)
        self.assertEqual(len(loop.renders), 3)

    def test_zero_is_off(self):
        loop = _Loop(align_iters=0)
        self.assertEqual(len(loop.runs), 1)
        self.assertEqual(loop.renders, [])
        self.assertEqual(loop.aligned, [])


class TestEachIterationWarpsTheOriginals(unittest.TestCase):
    """The invariant the guide puts above everything else. Every iteration
    warps the pristine frames toward the current render, which is a
    many-to-one contraction onto one consensus; iterating warps-of-warps is
    pairwise merging and drifts without bound."""

    def setUp(self):
        self.inputs = _inputs()
        self.loop = _Loop(self.inputs, align_iters=3)

    def test_the_frames_handed_to_the_warp_are_the_inputs_every_time(self):
        originals = [np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
                     for image in self.inputs["images"]]
        self.assertEqual(len(self.loop.aligned), 3)
        for iteration in self.loop.aligned:
            for handed, original in zip(iteration["frames"], originals):
                np.testing.assert_array_equal(handed, original)

    def test_the_export_carries_the_warped_frames_though(self):
        """It is the dataset that is rewritten each time, not the source:
        brush must train on the aligned set, and only the loop keeps the
        originals."""
        stamps = [run["frames"][0][0, 0, 0] for run in self.loop.runs]
        self.assertEqual(stamps, [10, 101, 102, 103])


class TestTheAlignmentInvocation(unittest.TestCase):
    def setUp(self):
        self.loop = _Loop(_inputs(with_normals=True), align_iters=2,
                          align_steps=1500, total_steps=30000)
        self.main, self.first, self.second = (run["cmd"] for run in self.loop.runs)

    def test_it_warm_starts_from_the_splat_as_it_stands(self):
        """brush resumes from an init.ply in the dataset directory — there
        is no flag for it — and each iteration refits the previous
        iteration's export, which is where the accumulated gain lives."""
        self.assertFalse(self.loop.runs[0]["init"])
        self.assertTrue(self.loop.runs[1]["init"])
        self.assertTrue(self.loop.runs[2]["init"])
        self.assertEqual([render["ply"] for render in self.loop.renders],
                         ["ply 1\n", "ply 2\n"])

    def test_it_runs_align_steps_iterations(self):
        self.assertEqual(_value(self.main, "--total-train-iters"), "30000")
        self.assertEqual(_value(self.first, "--total-train-iters"), "1500")
        self.assertEqual(_value(self.second, "--total-train-iters"), "1500")

    def test_growth_and_refinement_are_off(self):
        """Growth belongs in the cold start, which is the run above these:
        it grows on the gradients of the frames as they came in, before any
        flow has been measured against a half-trained splat."""
        self.assertNotIn("--growth-stop-iter", self.main)
        self.assertEqual(_value(self.first, "--growth-stop-iter"), "0")
        self.assertEqual(int(_value(self.first, "--refine-every")), 1_000_000)

    def test_normal_supervision_is_off_for_the_iterations(self):
        """The warped frames no longer line up with the normals/ sidecar
        beside them, and there is no start iteration that turns the term off
        for a run this short — so the weight itself goes to zero, whatever
        the step is configured with."""
        self.assertEqual(float(_value(self.main, "--normal-loss-weight")), 0.05)
        self.assertEqual(float(_value(self.first, "--normal-loss-weight")), 0.0)
        self.assertEqual(float(_value(self.second, "--normal-loss-weight")), 0.0)

    def test_it_exports_over_the_same_ply(self):
        """Nothing downstream has to know the loop happened."""
        for cmd in (self.main, self.first, self.second):
            self.assertEqual(_value(cmd, "--export-name"), "export.ply")

    def test_the_flow_settings_reach_the_warp(self):
        loop = _Loop(align_iters=1, align_flow_sigma=[3.0], align_flow_cap=[12.0])
        self.assertEqual(loop.aligned[0]["sigma"], 3.0)
        self.assertEqual(loop.aligned[0]["cap"], 12.0)


class TestTheFlowSettingsAreSchedulable(unittest.TestCase):
    """One sigma and one cap per iteration, first to last — because the
    render the flow is measured against is not the same thing at iteration
    4 that it was at iteration 1, and a converged one takes a finer,
    longer-reaching field (FINDINGS: sigma 3 / cap 12 from an aligned splat
    reads 27.0 against 26.7 at unchanged fidelity)."""

    def test_the_default_is_six_and_six_throughout(self):
        params = get_step_class("brush").declared_params()
        self.assertEqual(params["align_flow_sigma"].default, [6.0])
        self.assertEqual(params["align_flow_cap"].default, [6.0])

    def test_a_single_entry_holds_for_the_whole_loop(self):
        loop = _Loop(align_iters=3)
        self.assertEqual([one["sigma"] for one in loop.aligned], [6.0, 6.0, 6.0])
        self.assertEqual([one["cap"] for one in loop.aligned], [6.0, 6.0, 6.0])

    def test_a_full_schedule_is_read_in_order(self):
        loop = _Loop(align_iters=4, align_flow_sigma=[6, 6, 3, 3],
                     align_flow_cap=[6, 6, 12, 12])
        self.assertEqual([one["sigma"] for one in loop.aligned], [6.0, 6.0, 3.0, 3.0])
        self.assertEqual([one["cap"] for one in loop.aligned], [6.0, 6.0, 12.0, 12.0])

    def test_a_schedule_of_the_wrong_length_is_refused_before_training(self):
        """An hour into the cold start is the wrong place to find out, so
        this is checked in `run()` before brush is invoked at all."""
        step_class = get_step_class("brush")
        for overrides in ({"align_flow_sigma": [6, 3]}, {"align_flow_cap": []}):
            with self.subTest(**overrides):
                step, runs = step_class(), []
                step._run_brush = lambda cmd, ply, colmap_dir=None: runs.append(cmd)
                with tempfile.TemporaryDirectory() as tmp:
                    params = step_class.resolve_params(
                        {"export_dir": tmp, "align_iters": 4, **overrides})
                    with self.assertRaises(ValueError) as caught:
                        step.run(_inputs(), params)
                self.assertIn("align_iters is 4", str(caught.exception))
                self.assertEqual(runs, [])


class TestItComesBeforeThePolish(unittest.TestCase):
    def test_the_polish_finishes_the_splat_the_loop_produced(self):
        """Both are warm starts with growth off; the polish is the last word
        on the .ply, so it runs on the aligned dataset the loop left behind."""
        loop = _Loop(align_iters=2, align_steps=1500, polish_steps=9000)
        lengths = [_value(run["cmd"], "--total-train-iters") for run in loop.runs]
        self.assertEqual(lengths, ["30000", "1500", "1500", "9000"])
        self.assertEqual(loop.runs[-1]["frames"][0][0, 0, 0], 102)


class TestFramesAndCamerasMustAgree(unittest.TestCase):
    def test_a_size_mismatch_is_refused_before_anything_is_written(self):
        """The loop renders each camera and measures the flow to its frame.
        If the two describe different images the export was already wrong —
        the upscale rescaling the intrinsics is exactly this hazard — and
        the alignment would quietly make the splat worse rather than fail."""
        inputs = _inputs(size=8)
        inputs["cameras"] = _cameras(2, size=16)
        with self.assertRaises(ValueError) as caught:
            _Loop(inputs, align_iters=1)
        self.assertIn("8x8", str(caught.exception))

    def test_it_is_only_checked_when_the_loop_will_run(self):
        inputs = _inputs(size=8)
        inputs["cameras"] = _cameras(2, size=16)
        _Loop(inputs, align_iters=0)


class TestTheGrowthKnobs(unittest.TestCase):
    """Exposed but off: dense growth buys 2.5 s1 on top of the alignment
    loop and costs 4.6x the .ply (424 MB against 84), which is a deliberate
    purchase rather than a default."""

    def test_they_are_not_passed_unless_set(self):
        cmd = _Loop(align_iters=0).runs[0]["cmd"]
        self.assertNotIn("--growth-grad-threshold", cmd)
        self.assertNotIn("--growth-select-fraction", cmd)
        self.assertNotIn("--growth-stop-iter", cmd)

    def test_they_reach_the_cold_start(self):
        cmd = _Loop(align_iters=0, growth_grad_threshold=0.0012,
                    growth_select_fraction=0.4, growth_stop_iter=24000
                    ).runs[0]["cmd"]
        self.assertEqual(_value(cmd, "--growth-grad-threshold"), "0.0012")
        self.assertEqual(_value(cmd, "--growth-select-fraction"), "0.4")
        self.assertEqual(_value(cmd, "--growth-stop-iter"), "24000")

    def test_the_alignment_iterations_still_stop_growth(self):
        """Whatever the cold start was told to do, an alignment iteration
        refits what is already there."""
        runs = _Loop(align_iters=1, growth_stop_iter=24000).runs
        self.assertEqual(_value(runs[0]["cmd"], "--growth-stop-iter"), "24000")
        self.assertEqual(_value(runs[1]["cmd"], "--growth-stop-iter"), "0")


class TestWhatTheRunLeavesBehind(unittest.TestCase):
    """A 30,000-iteration training plus four alignment passes is an hour of
    GPU, and everything the loop touches is transient: the warped frames go
    with the COLMAP temp directory and each iteration exports over the same
    .ply. What is logged and what is kept is therefore all a bad result has
    behind it."""

    def _logs(self, **overrides):
        with self.assertLogs("pipeline.steps.brush", level="INFO") as caught:
            loop = _Loop(**overrides)
        return loop, caught.output

    def test_every_iteration_reports_what_it_measured(self):
        _, logs = self._logs(align_iters=2, align_steps=1500)
        lines = [line for line in logs if "alignment 1/2" in line
                 or "alignment 2/2" in line]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertIn("1.00 px mean", line)
            self.assertIn("2.00 px p90", line)
            self.assertIn("sigma 6.0", line)
            self.assertIn("1500 steps", line)

    def test_the_loop_closes_with_the_whole_trajectory(self):
        """The shape is the thing worth seeing, and the per-iteration lines
        are scattered through thousands of lines of training output."""
        _, logs = self._logs(align_iters=3)
        closing = [line for line in logs if "alignment finished" in line]
        self.assertEqual(len(closing), 1)
        self.assertIn("1.00 -> 1.00 -> 1.00", closing[0])
        # The reference trajectory rides along, since a RISING measurement
        # is what a working loop looks like and reads like drift otherwise.
        self.assertIn("1.02 -> 1.18 -> 1.26 -> 1.31", closing[0])

    def test_a_cap_bound_iteration_warns(self):
        """Mechanical and quiet on a healthy run: the reference loop
        measured a p90 of 2.5-3.3 px against a cap of 6."""
        with self.assertLogs("pipeline.steps.brush", level="WARNING") as caught:
            _Loop(align_iters=1, align_flow_cap=[2.0])
        self.assertIn("CAP-BOUND", caught.output[0])
        self.assertIn("2.0 px cap", caught.output[0])

    def test_a_healthy_run_does_not_warn(self):
        _, logs = self._logs(align_iters=2)
        self.assertEqual([line for line in logs if "WARNING" in line], [])

    def test_nothing_is_kept_unless_a_directory_is_named(self):
        self.assertIsNone(
            get_step_class("brush").declared_params()["align_debug_dir"].default)


class TestTheAlignmentDebugDirectory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "alignment"
        self.loop = _Loop(align_iters=3, align_debug_dir=str(self.dir))
        self.addCleanup(self.tmp.cleanup)

    def test_it_carries_every_view_s_own_figures(self):
        """A batch average that looks fine can hide one bad frame."""
        import json

        record = json.loads((self.dir / "alignment.json").read_text())
        self.assertEqual(record["views"], ["frame_00001_.png", "frame_00002_.png"])
        self.assertEqual(len(record["iterations"]), 3)
        first = record["iterations"][0]
        self.assertEqual(first["sigma"], 6.0)
        self.assertEqual(first["cap"], 6.0)
        self.assertEqual([view["name"] for view in first["per_view"]],
                         ["frame_00001_.png", "frame_00002_.png"])
        self.assertEqual([view["mean"] for view in first["per_view"]], [1.0, 1.1])

    def test_it_is_rewritten_each_iteration_so_a_crash_keeps_the_rest(self):
        """Complete after every iteration, not appended at the end — the
        iteration that dies is the one worth having the history for."""
        import json

        step_class = get_step_class("brush")
        directory = Path(self.tmp.name) / "partial"
        step = step_class()
        calls = []

        def die_on_the_second(cmd, ply_path, colmap_dir=None):
            calls.append(cmd)
            Path(ply_path).write_text("ply\n")
            if len(calls) == 3:
                raise RuntimeError("brush fell over")

        step._run_brush = die_on_the_second
        step._render_training_views = lambda ply, cams, names, *, render_path: [
            np.full((c.height, c.width, 3), 128, np.uint8) for c in cams]
        with tempfile.TemporaryDirectory() as export:
            params = step_class.resolve_params({
                "export_dir": export, "align_iters": 4,
                "align_debug_dir": str(directory)})
            with self.assertRaises(RuntimeError):
                # The REAL warp here, not the fake the other tests patch in,
                # so 16 px frames: DIS refuses anything under 12.
                step.run(_inputs(size=16), params)

        record = json.loads((directory / "alignment.json").read_text())
        self.assertEqual([one["iteration"] for one in record["iterations"]], [1, 2])

    def test_it_keeps_a_warped_frame_beside_the_render_it_was_warped_onto(self):
        """The pair is the only way a tear or a runaway warp is visible
        after the fact; the same view every iteration, so they compare."""
        for iteration in (1, 2, 3):
            warped = self.dir / f"iter{iteration}_frame_00001__warped.png"
            render = self.dir / f"iter{iteration}_frame_00001__render.png"
            self.assertTrue(warped.is_file(), warped)
            self.assertTrue(render.is_file(), render)
        # The warped frame is the one that went to the export, not a copy of
        # the original: the stamp the fake writes is iteration-specific.
        kept = cv2.imread(str(self.dir / "iter2_frame_00001__warped.png"),
                          cv2.IMREAD_UNCHANGED)
        self.assertEqual(kept[0, 0, 0], 102)


class TestTheTrainerBackend(unittest.TestCase):
    """The trainer carries the loop itself (--align-iters): the renders, the
    flow and the warps happen on the GPU against the frames it already
    holds, and the refits continue in the same process. From here that is
    ONE invocation with the loop's settings on its argv, and none of the
    round trips the pipeline loop makes — no re-invocation, no render, no
    warp, no init.ply."""

    def test_it_is_one_invocation_carrying_the_loop_settings(self):
        loop = _Loop(align_iters=3, align_backend="trainer",
                     align_flow_sigma=[6, 6, 3], align_flow_cap=[6, 6, 12])
        self.assertEqual(len(loop.runs), 1)
        self.assertEqual(loop.renders, [])
        self.assertEqual(loop.aligned, [])
        cmd = loop.runs[0]["cmd"]
        self.assertEqual(_value(cmd, "--align-iters"), "3")
        self.assertEqual(_value(cmd, "--align-steps"), "3000")
        self.assertEqual(_value(cmd, "--align-flow-sigma"), "6.0,6.0,3.0")
        self.assertEqual(_value(cmd, "--align-flow-cap"), "6.0,6.0,12.0")
        self.assertNotIn("--align-debug-dir", cmd)
        # The cold run is still the cold run: growth is the workflow's, and
        # nothing was warm-started.
        self.assertFalse(loop.runs[0]["init"])
        self.assertEqual(_value(cmd, "--total-train-iters"), "30000")

    def test_the_debug_dir_reaches_the_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = _Loop(align_iters=2, align_backend="trainer",
                         align_debug_dir=tmp)
        self.assertEqual(_value(loop.runs[0]["cmd"], "--align-debug-dir"), tmp)

    def test_zero_iterations_passes_nothing(self):
        loop = _Loop(align_iters=0, align_backend="trainer")
        self.assertEqual(len(loop.runs), 1)
        self.assertNotIn("--align-iters", loop.runs[0]["cmd"])

    def test_the_pipeline_backend_keeps_the_loop_here(self):
        loop = _Loop(align_iters=3, align_backend="pipeline")
        self.assertEqual(len(loop.runs), 4)
        self.assertEqual(len(loop.renders), 3)
        for run in loop.runs:
            self.assertNotIn("--align-iters", run["cmd"])

    def test_auto_is_the_default_and_asks_the_binary(self):
        params = get_step_class("brush").declared_params()
        self.assertEqual(params["align_backend"].default, "auto")
        # And the binary it asks is the trainer the image ships, so on a
        # real run `auto` resolves to the in-trainer loop.
        self.assertEqual(params["brush_path"].default, "b2ctrain")
        with mock.patch("pipeline.steps.brush._trainer_aligns", return_value=True):
            self.assertEqual(len(_Loop(align_iters=3, align_backend="auto").runs), 1)
        with mock.patch("pipeline.steps.brush._trainer_aligns", return_value=False):
            self.assertEqual(len(_Loop(align_iters=3, align_backend="auto").runs), 4)

    def test_a_binary_that_cannot_run_is_the_pipeline_loop_not_an_error(self):
        from pipeline.steps import brush as brush_step
        brush_step._HELP_PROBE.pop("/nonexistent/trainer", None)
        self.assertFalse(brush_step._trainer_aligns("/nonexistent/trainer"))

    def test_an_unknown_backend_is_refused(self):
        with self.assertRaises(ValueError):
            _Loop(align_iters=1, align_backend="somewhere")

    def test_a_polish_after_a_trainer_aligned_run_says_what_it_resumes_on(self):
        """The one place the backends differ. The pipeline loop leaves its
        last warp in the export's `images/`, so a polish resumes on the
        frames the splat was aligned to; the trainer's warps never reach
        disk, so the same polish resumes on the pristine originals and
        pulls the fit back toward the disagreement the loop just removed.
        No shipped workflow asks for both, and if one ever does this has to
        be in the log rather than in the wall time."""
        with self.assertLogs("pipeline.steps.brush", level="WARNING") as logs:
            loop = _Loop(align_iters=2, align_backend="trainer", polish_steps=500)
        self.assertEqual(len(loop.runs), 2)  # the aligned cold run, then the polish
        self.assertTrue(
            any("UNWARPED" in line for line in logs.output),
            f"no warning about what the polish resumes on: {logs.output}",
        )


if __name__ == "__main__":
    unittest.main()
