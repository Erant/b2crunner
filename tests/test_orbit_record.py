"""The orbit record in the delivered .ply (pipeline/orbit_record.py,
pipeline/steps/orbit_record.py).

What matters is that a record read back from scene.ply is enough to extend
the orbit after the fact. So the path is checked against cyber_6f's real
cameras: the record of an extended run, read back, has to be a path that
extend_helical_path accepts and continues again. The format itself is
checked on a small synthetic .ply: it round-trips, it leaves the b2c.mhr.*
body alone, and it pins the images beside it by checksum.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import ply_meta
from pipeline.dataset import Dataset
from pipeline.orbit_record import (
    PREFIX, cameras_from_arrays, orbit_comments, parse_orbit_comments, read_orbit_record,
)
from pipeline.registry import get_step_class
from pipeline.steps.splat import _resolve_cameras, _transform_camera
from tests.helpers import require_stage, run_step

import pipeline.steps  # noqa: F401

HELIX = dict(n_frames=81, n_loops=2, amplitude_deg=30.0, lead_in_deg=30.0, lead_out_deg=90.0)
PROMPT = "a woman in a silver jacket, 银色夹克"


def _write_ply(path: Path, n: int = 3) -> bytes:
    """A minimal binary splat-shaped .ply with an exporter comment and a
    b2c.mhr.* line; returns the body bytes."""
    body = np.arange(n * 3, dtype="<f4").tobytes()
    header = ("ply\nformat binary_little_endian 1.0\ncomment Created by b2ctrain\n"
              "comment b2c.mhr.version 1\n"
              f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
              "end_header\n")
    path.write_bytes(header.encode("ascii") + body)
    return body


def _dataset(cameras, extras):
    return Dataset(
        images=[], image_names=[], cameras=list(cameras),
        points_3d=(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)),
        resolution=(720, 1280), prompt=PROMPT, extras=dict(extras),
        reference_image=np.full((16, 8, 3), 90, np.uint8),
        anchor_image=np.full((16, 8, 3), 200, np.uint8),
    )


class _Cyber6f(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("initial"))
        params = get_step_class("render_splat").resolve_params(
            dict(HELIX, pattern="helical", override_cam_from_mesh=True))
        source, _, _, anchor = _resolve_cameras(
            scene=None, dataset=cls.ds, params=params, width=720, height=1280)
        # Carried, as render_subject carries the path by the anchor's refinement.
        import cv2
        rotation, _ = cv2.Rodrigues(np.radians(np.array([0.4, 1.2, -0.2])))
        cls.source = [_transform_camera(c, rotation, np.array([0.041, -0.034, 0.021]))
                      for c in source]
        cls.extras = dict(cls.ds.extras, anchor_frame_index=anchor)

    def _roundtrip(self, record, tmp):
        ply = Path(tmp) / "scene.ply"
        _write_ply(ply)
        run_step("embed_orbit_record",
                 {"splat_path": str(ply), "record": record, "cameras": self.source[:3]})
        return read_orbit_record(ply)


class TestARecordExtendsAfterTheFact(_Cyber6f):
    def test_an_unextended_run_s_record_is_extended_as_the_run_would_have_been(self):
        record = run_step("snapshot_orbit", {"dataset": _dataset(self.source, self.extras)},
                          dict(HELIX))["record"]
        with tempfile.TemporaryDirectory() as tmp:
            back = self._roundtrip(record, tmp)
        self.assertEqual(back["extension"], {"before": 0, "after": 0, "overlap_before": 0,
                                             "overlap_after": 0})
        cameras = cameras_from_arrays(back["orbit_cameras"])
        live = run_step("extend_helical_path", {"cameras": self.source, "extras": self.extras},
                        dict(back["helix"]))
        after = run_step("extend_helical_path", {"cameras": cameras, "extras": back["extras"]},
                         dict(back["helix"]))
        self.assertEqual(len(after["cameras"]), 162)
        for got, want in zip(after["cameras"], live["cameras"]):
            np.testing.assert_allclose(got.position, want.position, atol=1e-5)
            np.testing.assert_allclose(got.rotation, want.rotation, atol=1e-5)

    def test_an_extended_run_s_record_is_the_extended_helix_and_extends_again(self):
        first = run_step("extend_helical_path", {"cameras": self.source, "extras": self.extras},
                         dict(HELIX))
        counts = {k: first[k] for k in ("before", "after", "overlap_before", "overlap_after")}
        extras = dict(self.extras, anchor_frame_index=self.extras["anchor_frame_index"] + 41)
        record = run_step("snapshot_orbit",
                          dict({"dataset": _dataset(first["cameras"], extras)}, **counts),
                          dict(HELIX))["record"]
        with tempfile.TemporaryDirectory() as tmp:
            back = self._roundtrip(record, tmp)
        self.assertEqual(back["helix"]["n_frames"], 162)
        self.assertAlmostEqual(back["helix"]["lead_in_deg"], 30.0 + 41 * 840.0 / 81, places=6)
        self.assertEqual(back["extension"]["before"], 41)
        self.assertEqual(back["anchor_frame_index"], self.extras["anchor_frame_index"] + 41)
        # extend_helical_path rebuilds the 162-frame path from the record's
        # params and refuses one it cannot reproduce: this is the check.
        again = run_step("extend_helical_path",
                         {"cameras": cameras_from_arrays(back["orbit_cameras"]),
                          "extras": back["extras"]},
                         dict(back["helix"], overlap_before=40, overlap_after=41))
        self.assertEqual(len(again["cameras"]), 243)

    def test_a_snapshot_off_the_helix_it_names_is_refused(self):
        with self.assertRaisesRegex(ValueError, "81 of the helix"):
            run_step("snapshot_orbit", {"dataset": _dataset(self.source[:-1], self.extras)},
                     dict(HELIX))
        with self.assertRaisesRegex(ValueError, "come as a pair"):
            run_step("snapshot_orbit", {"dataset": _dataset(self.source, self.extras),
                                        "before": 41}, dict(HELIX))


class TestTheHeader(_Cyber6f):
    def _record(self):
        record = run_step("snapshot_orbit",
                          {"dataset": _dataset(self.source, self.extras),
                           "front_image": np.full((16, 8, 3), 30, np.uint8)},
                          dict(HELIX, run_seed=7, framing="full"))["record"]
        return record

    def test_it_round_trips_beside_the_body_and_the_vertices_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "scene.ply"
            body = _write_ply(ply)
            out = run_step("embed_orbit_record", {"splat_path": str(ply), "record": self._record(),
                                                  "cameras": self.source})
            self.assertEqual(set(out["files"]), {"reference", "anchor", "front"})
            for name in ("reference.png", "anchor.png", "front.png"):
                self.assertTrue((Path(tmp) / name).exists())
            back = read_orbit_record(ply)
            comments = ply_meta.read_comments(ply)
            self.assertEqual(ply.read_bytes()[-len(body):], body)
        self.assertIn("b2c.mhr.version 1", comments)
        self.assertEqual(comments[0], "Created by b2ctrain")
        self.assertEqual(back["version"], 1)
        self.assertEqual(back["prompt"], PROMPT)
        self.assertEqual(back["settings"], {"seed": 7, "resolution": [720, 1280],
                                            "framing": "full"})
        self.assertEqual(back["helix"], HELIX)
        self.assertEqual(back["pass_frames"], 81)
        np.testing.assert_allclose(back["extras"]["orbit_target"], self.extras["orbit_target"])
        for group in ("orbit_cameras", "final_cameras"):
            cameras = cameras_from_arrays(back[group])
            self.assertEqual(len(cameras), 81)
            for got, want in zip(cameras, self.source):
                np.testing.assert_allclose(got.position, want.position, atol=0)
                np.testing.assert_allclose(got.rotation, want.rotation, atol=0)
                self.assertEqual((got.fx, got.cy, got.width, got.height),
                                 (want.fx, want.cy, want.width, want.height))

    def test_embedding_twice_leaves_one_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "scene.ply"
            _write_ply(ply)
            for _ in range(2):
                run_step("embed_orbit_record", {"splat_path": str(ply), "record": self._record(),
                                                "cameras": self.source})
            comments = ply_meta.read_comments(ply)
        self.assertEqual(sum(c.startswith(PREFIX + "version") for c in comments), 1)
        self.assertEqual(len([c for c in comments if c.startswith(PREFIX)]),
                         len(orbit_comments(dict(self._record(), final_cameras=self.source,
                                                 images={}))))

    def test_a_missing_or_changed_image_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "scene.ply"
            _write_ply(ply)
            run_step("embed_orbit_record", {"splat_path": str(ply), "record": self._record(),
                                            "cameras": self.source})
            import cv2
            cv2.imwrite(str(Path(tmp) / "anchor.png"), np.zeros((16, 8, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_orbit_record(ply)
            (Path(tmp) / "anchor.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "anchor.png"):
                read_orbit_record(ply)

    def test_the_header_stays_ascii(self):
        lines = orbit_comments(dict(self._record(), final_cameras=None, images={}))
        for line in lines:
            line.encode("ascii")
        self.assertEqual(parse_orbit_comments(lines)["prompt"], PROMPT)

    def test_a_ply_without_a_record_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "scene.ply"
            _write_ply(ply)
            with self.assertRaisesRegex(ValueError, "no b2c.orbit"):
                read_orbit_record(ply)


class TestTheWiring(unittest.TestCase):
    def test_snapshot_after_the_splice_embed_after_the_training_both_behind_export_ply(self):
        from pipeline.workflow import WorkflowSpec
        from tests.test_workflows import WORKFLOW_DIR

        for name in ("helical.yaml", "helical_shell.yaml"):
            with self.subTest(workflow=name):
                spec = WorkflowSpec.from_yaml(str(WORKFLOW_DIR / name))
                steps = {s.id: s for s in spec.steps}
                order = [s.id for s in spec.steps]
                self.assertEqual(order.index("snapshot_orbit"), order.index("extend_splice") + 1)
                self.assertLess(order.index("snapshot_orbit"), order.index("upscale"))
                self.assertLess(order.index("snapshot_orbit"), order.index("refine_cameras_final"))
                self.assertEqual(order[-1], "embed_orbit_record")
                self.assertEqual(order[-2], "train_final_splat")
                for step_id in ("snapshot_orbit", "embed_orbit_record"):
                    self.assertEqual(steps[step_id].when, "${globals.export_ply}")
                snapshot, subject = steps["snapshot_orbit"], steps["render_subject"]
                for key in HELIX:
                    self.assertEqual(snapshot.params[key], subject.params[key])
                self.assertEqual(snapshot.params["pass_frames"], subject.params["n_frames"])
                self.assertEqual(snapshot.params["run_seed"], "${globals.seed}")
                self.assertEqual(steps["embed_orbit_record"].inputs["splat_path"],
                                 steps["train_final_splat"].outputs["splat_path"])


if __name__ == "__main__":
    unittest.main()
