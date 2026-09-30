"""rig_subject: rig the delivered subject file with b2crig, so the splat can be posed (the "Rig splat" output).

b2crig (Erant/b2crig) owns every rig decision (its BOUNDARY.md); this step only runs its pipeline entry point,
`tools/rig_subject.py SUBJECT.glb CAPTURE_DIR OUT.glb`, and puts the results where the Results tab and the viewer
look. What b2crig does there, in order: the MHR body and splat out of the subject file, the default garment / hair /
face cage layers, the synthetic range-of-motion pose library and a 16-pose containment set chosen from it, the
knuckle split of hand splats, a pose containment fine-tune of the split splat against the capture views, the rig
itself (cage + b2ctrain's binding, b2cgltf SPEC 5) and one clip that tours the library's named poses (SPEC 6).

Files: `ply/scene.glb` becomes the rigged subject file (its scene 0 still shows the splat as delivered), the clip
lands beside it as `<clip>.clip.glb`, and the unrigged file moves to `debug/rig/subject_unrigged.glb`, the input to
re-rig from. A failure puts the unrigged file back before it raises, so the run still delivers it.

Where b2crig is: `b2crig_dir`, else $B2CRIG_DIR, else /opt/b2crig (the image's), else ~/Projects/b2crig. Its python:
`python`, else $B2CRIG_PYTHON, else the checkout's own .venv when it has one, else this interpreter (the image
installs b2crig's few dependencies beside the pipeline's).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..registry import register_step
from ..step import REQUIRED, Param, Step

logger = logging.getLogger(__name__)

B2CRIG_ENV = "B2CRIG_DIR"
B2CRIG_PYTHON_ENV = "B2CRIG_PYTHON"
IMAGE_B2CRIG_DIR = "/opt/b2crig"
ENTRY = Path("tools") / "rig_subject.py"


def b2crig_dir(explicit: str = "") -> Path | None:
    """The b2crig checkout the step runs, or None when there is none."""
    for cand in (explicit, os.environ.get(B2CRIG_ENV, ""), IMAGE_B2CRIG_DIR, str(Path.home() / "Projects" / "b2crig")):
        if cand and (Path(cand) / ENTRY).is_file():
            return Path(cand)
    return None


def b2crig_python(root: Path, explicit: str = "") -> str:
    if explicit or os.environ.get(B2CRIG_PYTHON_ENV):
        return explicit or os.environ[B2CRIG_PYTHON_ENV]
    venv = root / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


@register_step("rig_subject")
class RigSubjectStep(Step):
    """The subject file rigged by b2crig, plus a clip to pose it with.

    inputs:  {"subject_path": export_subject's .glb}
    outputs: {"subject_path": the rigged .glb (the same path), "clip_path": the clip file or None}
    """

    PARAMS = (
        Param("capture_dir", str, REQUIRED,
              "The run's COLMAP dataset with its labels/ (export_colmap's): the views the containment fine-tune "
              "keeps the splat faithful to"),
        Param("debug_dir", str, REQUIRED, "Where the unrigged subject file, the pose set and the training log go"),
        Param("clip", str, "rom_tour", "The clip written beside the subject file ('none': no clip)",
              choices=["rom_tour", "none"]),
        Param("b2crig_dir", str, "", "The b2crig checkout (empty: $B2CRIG_DIR, /opt/b2crig, ~/Projects/b2crig)",
              advanced=True),
        Param("python", str, "", "The interpreter that runs b2crig (empty: $B2CRIG_PYTHON, the checkout's .venv, "
              "this one)", advanced=True),
        Param("keep_work", bool, False, "Keep b2crig's work directory (pose library cages, training dataset) under "
              "debug_dir", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from ..proc import stream_command

        root = b2crig_dir(params["b2crig_dir"])
        if root is None:
            raise RuntimeError(f"rig_subject: no b2crig checkout (set {B2CRIG_ENV}, or clone Erant/b2crig to "
                               f"{IMAGE_B2CRIG_DIR} or ~/Projects/b2crig)")
        subject = Path(inputs["subject_path"])
        capture = Path(params["capture_dir"])
        debug = Path(params["debug_dir"])
        debug.mkdir(parents=True, exist_ok=True)
        unrigged = debug / "subject_unrigged.glb"
        shutil.move(subject, unrigged)
        work = Path(tempfile.mkdtemp(prefix="b2c_rig_"))
        cmd = [b2crig_python(root, params["python"]), str(root / ENTRY), str(unrigged), str(capture), str(subject),
               "--work", str(work), "--clip", params["clip"]]
        try:
            tail = stream_command(cmd, "b2crig", cwd=str(root), throttle=False)
        except BaseException:
            if not subject.exists():
                shutil.move(unrigged, subject)
            logger.error("rig_subject: failed; the unrigged subject file is back at %s, b2crig's work directory "
                         "left at %s", subject, work)
            raise
        report = next((json.loads(line) for line in reversed(tail) if line.startswith("{")), {})
        (debug / "report.json").write_text(json.dumps(report, indent=1))
        S = work / "subject"
        for src, name in ((S / "clips" / "poseset_romh16" / "poses.json", "poseset.json"),
                          (S / "train" / "contain" / "train.log", "containment_train.log")):
            if src.is_file():
                shutil.copy(src, debug / name)
        if params["keep_work"]:
            shutil.rmtree(debug / "work", ignore_errors=True)
            shutil.move(work, debug / "work")
        else:
            shutil.rmtree(work, ignore_errors=True)
        clip = subject.parent / f"{params['clip']}.clip.glb"
        rig = report.get("rig", {})
        logger.info("rig_subject: %s (%.1f MB): %s splats (%s unbound), %s cage vertices, layers %s; clip %s; %ss",
                    subject.name, subject.stat().st_size / 1e6, rig.get("splats"), rig.get("unbound"),
                    rig.get("cage_vertices"), ", ".join(rig.get("layers", [])),
                    clip.name if clip.is_file() else "none", report.get("seconds"))
        return {"subject_path": str(subject), "clip_path": str(clip) if clip.is_file() else None}
