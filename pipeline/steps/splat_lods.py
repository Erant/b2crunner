"""splat_lods: the delivered splat's LOD levels, for an unrigged delivery (b2cgltf SPEC 4.7).

Each level is the delivered splat merged down to a count by b2ctrain (`lod-merge`: splat-transform's KNN
moment-matched merging, never across seg_label classes), then fine-tuned against the delivered splat's own renders
(`lod-distill`: warm, growth off, the teacher's alpha). export_subject writes them into the subject file as
b2c_splat_lod1..n behind b2c_splat. The level sizes are fixed (user, 2026-10-30: always written, no switch; levels
can be stripped later): 100k splats for up to 960 px of screen height, 50k for 480 px. A level that would not hold
fewer splats than the delivered one is left out (the spec needs strictly fewer).

Measured on the six crew-* subjects (186k-386k splats), PSNR against the delivered splat at a quarter of the capture's
size: 40.1-41.6 dB at 100k, 37.4-38.8 dB at 50k (the merge alone, untuned, is ~2 dB lower; b2ctrain's
--sparsify pruning was 33 / 31 dB and looked broken).

A rigged delivery skips this step: b2crig makes its own levels from the splat it retrains (rig_subject).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from ..registry import register_step
from ..step import REQUIRED, Param, Step

logger = logging.getLogger(__name__)

COUNTS = (100_000, 50_000)


def _ply_count(path: Path) -> int:
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    raise ValueError(f"{path}: no vertex count")


@register_step("splat_lods")
class SplatLodsStep(Step):
    """The delivered splat merged to 100k and 50k splats and fine-tuned, for export_subject.

    inputs:  {"splat_path": the delivered .ply}
    outputs: {"levels": [{"path": the level .ply, "count", "method"}], most splats first; [] when the splat is
              already at or under every count}
    """

    PARAMS = (
        # Only the cameras are read: lod-distill trains on the teacher's renders.
        Param("views_dir", str, REQUIRED, "A COLMAP dataset whose cameras the levels are fine-tuned at"),
        Param("work_dir", str, REQUIRED, "Where the levels and their logs go"),
        Param("steps", int, 6000, "Fine-tune steps per level", advanced=True),
        Param("brush_path", str, "b2ctrain", "Trainer binary, on PATH or an absolute path", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        teacher = Path(inputs["splat_path"])
        work = Path(params["work_dir"])
        work.mkdir(parents=True, exist_ok=True)
        n0 = _ply_count(teacher)
        trainer = params["brush_path"]
        levels = []
        for count in COUNTS:
            if count >= n0:
                logger.info("splat_lods: %d splats, no %d level", n0, count)
                continue
            out = work / f"lod_{count}.ply"
            log = work / f"lod_{count}.log"
            with open(log, "w") as fh:
                for cmd in ([trainer, "lod-merge", "--splat", str(teacher), "--count", str(count), "--out", str(out)],
                            [trainer, "lod-distill", "--splat", str(out), "--teacher", str(teacher), "--dataset",
                             params["views_dir"], "--out", str(out), "--steps", str(params["steps"])]):
                    fh.write(" ".join(cmd) + "\n")
                    fh.flush()
                    if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode != 0:
                        raise RuntimeError(f"splat_lods: {cmd[1]} failed for the {count} level (see {log})")
            got = _ply_count(out)
            if got != count:
                raise RuntimeError(f"splat_lods: the {count} level holds {got} splats")
            levels.append({"path": str(out), "count": got,
                           "method": f"b2ctrain lod-merge + lod-distill ({params['steps']} steps)"})
            logger.info("splat_lods: %d -> %d splats", n0, got)
        return {"levels": levels}
