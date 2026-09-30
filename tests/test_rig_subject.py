"""rig_subject (steps/rig_subject.py): runs b2crig's entry point and puts its files where the Results tab and the
viewer look. b2crig itself is replaced by a stand-in `tools/rig_subject.py` that records its argv and writes what the
real one writes (or fails), so this tests the step's contract, not the rig.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.proc import ProcessFailed
from pipeline.steps import rig_subject as RS
from tests.helpers import run_step

import pipeline.steps  # noqa: F401

FAKE = r'''
import json, sys
from pathlib import Path
glb, capture, out = map(Path, sys.argv[1:4])
work = Path(sys.argv[sys.argv.index("--work") + 1])
clip = sys.argv[sys.argv.index("--clip") + 1]
(work / "argv.json").write_text(json.dumps({"glb": str(glb), "capture": str(capture), "input": glb.read_text()}))
if (capture / "FAIL").exists():
    print("rig_subject: cage_train"); sys.exit(3)
out.write_text("rigged " + glb.read_text())
if clip != "none":
    (out.parent / f"{clip}.clip.glb").write_text("clip")
pc = work / "subject" / "clips" / "poseset_romh16"; pc.mkdir(parents=True)
(pc / "poses.json").write_text("{}")
print(json.dumps({"stages": {"rig": 1.0}, "rig": {"splats": 7, "unbound": 0, "cage_vertices": 3,
                  "layers": ["body"]}, "seconds": 1.0}))
'''


class RigSubjectStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.b2crig = self.tmp / "b2crig"
        (self.b2crig / "tools").mkdir(parents=True)
        (self.b2crig / "tools" / "rig_subject.py").write_text(FAKE)
        self.ply = self.tmp / "run" / "ply"
        self.ply.mkdir(parents=True)
        self.subject = self.ply / "scene.glb"
        self.subject.write_text("unrigged")
        self.capture = self.tmp / "run" / "colmap"
        self.capture.mkdir()
        self.debug = self.tmp / "run" / "debug" / "rig"

    def params(self, **extra):
        return {"capture_dir": str(self.capture), "debug_dir": str(self.debug), "b2crig_dir": str(self.b2crig),
                "python": sys.executable, **extra}

    def test_rigs_in_place_and_keeps_the_unrigged_file(self):
        out = run_step("rig_subject", {"subject_path": str(self.subject)}, self.params())
        self.assertEqual(out["subject_path"], str(self.subject))
        self.assertEqual(self.subject.read_text(), "rigged unrigged")
        self.assertEqual((self.debug / "subject_unrigged.glb").read_text(), "unrigged")
        self.assertEqual(out["clip_path"], str(self.ply / "rom_tour.clip.glb"))
        self.assertEqual(json.loads((self.debug / "report.json").read_text())["rig"]["splats"], 7)
        self.assertTrue((self.debug / "poseset.json").is_file())
        self.assertFalse((self.debug / "work").exists())

    def test_no_clip(self):
        out = run_step("rig_subject", {"subject_path": str(self.subject)}, self.params(clip="none"))
        self.assertIsNone(out["clip_path"])
        self.assertEqual(sorted(p.name for p in self.ply.iterdir()), ["scene.glb"])

    def test_keep_work(self):
        run_step("rig_subject", {"subject_path": str(self.subject)}, self.params(keep_work=True))
        argv = json.loads((self.debug / "work" / "argv.json").read_text())
        self.assertEqual(argv["glb"], str(self.debug / "subject_unrigged.glb"))
        self.assertEqual(argv["capture"], str(self.capture))

    def test_a_failure_puts_the_unrigged_file_back(self):
        (self.capture / "FAIL").touch()
        with self.assertRaises(ProcessFailed):
            run_step("rig_subject", {"subject_path": str(self.subject)}, self.params())
        self.assertEqual(self.subject.read_text(), "unrigged")
        self.assertFalse((self.debug / "subject_unrigged.glb").exists())

    def test_no_checkout_is_refused_by_name(self):
        with mock.patch.dict(os.environ, {RS.B2CRIG_ENV: ""}), \
                mock.patch.object(RS, "IMAGE_B2CRIG_DIR", str(self.tmp / "missing")), \
                mock.patch.object(RS.Path, "home", return_value=self.tmp):
            with self.assertRaisesRegex(RuntimeError, "no b2crig checkout"):
                run_step("rig_subject", {"subject_path": str(self.subject)}, self.params(b2crig_dir=""))
        self.assertEqual(self.subject.read_text(), "unrigged")

    def test_checkout_lookup(self):
        with mock.patch.dict(os.environ, {RS.B2CRIG_PYTHON_ENV: ""}):
            self.assertEqual(RS.b2crig_dir(str(self.b2crig)), self.b2crig)
            self.assertEqual(RS.b2crig_python(self.b2crig, "/x/python"), "/x/python")
            self.assertEqual(RS.b2crig_python(self.b2crig), sys.executable)   # no .venv in the stand-in

if __name__ == "__main__":
    unittest.main()
