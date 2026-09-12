"""The per-splat body-part labels: `sapiens2_seg`'s batched class maps reach
the trainer as a `labels/` sidecar of raw class ids, `--export-labels` is
passed exactly when the trainer knows it, and the workflow wires the three
steps together behind `splat_labels` (steps/brush.py, steps/sapiens2.py,
b2ctrain 2977f0e)."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipeline.registry import get_step_class
import pipeline.steps  # noqa: F401
import pipeline.steps.brush as brush_mod

from .test_brush_evidence import _inputs


def _labels(value=3):
    return [np.full((8, 8), value, dtype=np.uint8) for _ in range(2)]


class TestBrushLabelsSidecar(unittest.TestCase):
    def _run(self, inputs, help_text="--align-iters --export-labels", **overrides):
        step_class = get_step_class("brush")
        step = step_class()
        seen = {}

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            seen.setdefault("cmds", []).append(list(cmd))
            labels = Path(colmap_dir) / "labels" if colmap_dir else None
            seen["labels"] = {}
            if labels and labels.is_dir():
                for path in sorted(labels.iterdir()):
                    seen["labels"][path.name] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        brush_mod._HELP_PROBE["brush"] = help_text
        try:
            with tempfile.TemporaryDirectory() as tmp:
                params = step_class.resolve_params({"export_dir": tmp, "align_iters": 0, "brush_path": "brush", **overrides})
                step.run(inputs, params)
        finally:
            brush_mod._HELP_PROBE.pop("brush", None)
        return seen

    def test_no_labels_means_no_sidecar_and_no_flag(self):
        seen = self._run(_inputs())
        self.assertEqual(seen["labels"], {})
        self.assertNotIn("--export-labels", seen["cmds"][0])

    def test_class_ids_are_written_raw_and_the_flag_is_passed(self):
        seen = self._run({**_inputs(), "labels": _labels(3)})
        self.assertIn("--export-labels", seen["cmds"][0])
        # Named like the masks/ sidecar, one per frame, and class 3 is 3 — not
        # a [0, 1] map scaled to 255.
        self.assertEqual(sorted(seen["labels"]), ["frame_00001_.png", "frame_00002_.png"])
        for image in seen["labels"].values():
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape, (8, 8))
            self.assertTrue((image == 3).all())

    def test_an_older_trainer_gets_the_sidecar_but_not_the_flag(self):
        seen = self._run({**_inputs(), "labels": _labels()}, help_text="--align-iters")
        self.assertEqual(len(seen["labels"]), 2)
        self.assertNotIn("--export-labels", seen["cmds"][0])

    def test_a_count_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            self._run({**_inputs(), "labels": _labels()[:1]})


class TestColmapExportLabels(unittest.TestCase):
    def test_the_bundle_carries_the_sidecar_in_the_brush_layout(self):
        step_class = get_step_class("colmap_export")
        step = step_class()
        with tempfile.TemporaryDirectory() as tmp:
            params = step_class.resolve_params({"output_dir": tmp, "layout": "brush"})
            step.run({**_inputs(), "labels": _labels(13)}, params)
            written = sorted(p.name for p in (Path(tmp) / "labels").iterdir())
            self.assertEqual(written, ["frame_00001_.png", "frame_00002_.png"])
            self.assertTrue((cv2.imread(str(Path(tmp) / "labels" / written[0]), cv2.IMREAD_UNCHANGED) == 13).all())
            params = step_class.resolve_params({"output_dir": tmp, "layout": "flat"})
            with self.assertRaises(ValueError):
                step.run({**_inputs(), "labels": _labels()}, params)


class TestSapiens2SegBatchedPath(unittest.TestCase):
    """`images` in, one uint8 class map per frame out, each frame through
    the model on its own; the single-image path is untouched."""

    def test_labels_per_frame(self):
        step_class = get_step_class("sapiens2_seg")
        step = step_class()
        params = step_class.resolve_params({})
        self.assertEqual(params["dtype"], "float32", "the face branch's measured setting")
        calls = []

        class Seg:  # the tensor's two calls, without torch in this venv
            def __init__(self, a): self.a = a
            def cpu(self): return self
            def numpy(self): return self.a

        class Processor:
            def post_process_semantic_segmentation(self, outputs, target_sizes):
                h, w = target_sizes[0]
                return [Seg(np.full((h, w), outputs, dtype=np.int64))]

        step._ready = lambda p: None
        step._processor = Processor()
        step._forward = lambda image: calls.append(image.shape) or len(calls)
        frames = [np.zeros((6, 4, 4), dtype=np.uint8), np.zeros((5, 3, 3), dtype=np.uint8)]
        out = step.run({"images": frames}, params)
        self.assertEqual(list(out), ["labels"])
        self.assertEqual([l.shape for l in out["labels"]], [(6, 4), (5, 3)])
        self.assertEqual([l.dtype for l in out["labels"]], [np.uint8, np.uint8])
        self.assertEqual([int(l[0, 0]) for l in out["labels"]], [1, 2])
        self.assertEqual(calls, [(6, 4, 4), (5, 3, 3)])


class TestTheWorkflowVotesLabelsOntoTheFinalSplat(unittest.TestCase):
    def _spec(self):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        return WorkflowSpec.from_yaml(resolve_workflow("fast_helical_native"))

    def _step(self, spec, step_id):
        return next(s for s in spec.steps if s.id == step_id)

    def test_segment_views_feeds_the_final_training_and_the_bundle(self):
        spec = self._spec()
        setting = next(p for p in spec.settings if p.name == "splat_labels")
        self.assertIs(setting.default, True)
        seg = self._step(spec, "segment_views")
        self.assertEqual(seg.step, "sapiens2_seg")
        self.assertEqual(seg.when, "${globals.splat_labels}")
        self.assertEqual(seg.inputs, {"images": "dataset.images"})
        self.assertEqual(seg.params.get("dtype"), "bfloat16")
        self.assertEqual(seg.outputs, {"labels": "scene.seg_labels"})
        order = [s.id for s in spec.steps]
        # After everything that rewrites the frames, before anything that reads them out.
        self.assertGreater(order.index("segment_views"), order.index("paste_eyes"))
        self.assertLess(order.index("segment_views"), order.index("export_colmap"))
        self.assertLess(order.index("segment_views"), order.index("train_final_splat"))
        for step_id in ("train_final_splat", "export_colmap"):
            self.assertEqual(self._step(spec, step_id).inputs.get("labels"), "scene.seg_labels?", step_id)
        # The intermediate training does not vote: its splat drives the re-render and is not kept.
        self.assertNotIn("labels", self._step(spec, "train_splat").inputs)

    def test_the_face_branch_keeps_its_float32_matte(self):
        spec = self._spec()
        for s in spec.steps:
            if s.step == "sapiens2_seg" and s.id != "segment_views":
                self.assertNotIn("dtype", s.params, s.id)


if __name__ == "__main__":
    unittest.main()
