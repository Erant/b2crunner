"""Sapiens2 pointmaps of a clip's WAN frames and of its LBS control render (b2crunner's pointmap head).

Runs in the wan22 environment::

    python3 tools/run.py pointmap_clip CLIP_DIR [--src wan control]

Writes <clip>/pointmap_<src>.npy: float16 [T, H, W, 3], camera-frame points on the source pixel grid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout
import pipeline.steps  # noqa: E402,F401
from pipeline.registry import get_step_class  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--src", nargs="+", default=["wan", "control"])
    a = ap.parse_args()
    cls = get_step_class("pointmap_splat")
    params = cls.resolve_params({"dtype": "bfloat16", "filepath": "/dev/null"})  # filepath: unused, the .ply this step would write
    step = cls()
    step.load(params)
    for src in a.src:
        frames = sorted(f for f in (a.clip / src).glob("*.png") if f.stem.isdigit())
        pm = np.stack([step._infer_pointmap(cv2.imread(str(f), cv2.IMREAD_COLOR), "bfloat16") for f in frames])
        np.save(a.clip / f"pointmap_{src}.npy", pm.astype(np.float16))
        print(f"pointmap_clip: {src} {pm.shape}")


if __name__ == "__main__":
    main()
