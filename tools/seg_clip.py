"""Sapiens2 segmentation of a clip's WAN frames (b2crunner's sapiens2_seg, batched path).

Runs in the wan22 environment (transformers is there)::

    python3 tools/run.py seg_clip CLIP_DIR [SRC OUT]

Reads <SRC>/NNNN.png (default wan), writes <OUT>/NNNN.png (default seg; uint8 Goliath class ids).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout
import pipeline.steps  # noqa: E402,F401
from pipeline.registry import get_step_class  # noqa: E402


def main() -> None:
    clip = Path(sys.argv[1])
    src, out = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("wan", "seg")
    frames = sorted(f for f in (clip / src).glob("*.png") if f.stem.isdigit())
    images = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in frames]
    cls = get_step_class("sapiens2_seg")
    params = cls.resolve_params({"dtype": "bfloat16"})
    step = cls()
    step.load(params)
    labels = step.run({"images": images}, params)["labels"]
    (clip / out).mkdir(exist_ok=True)
    for f, lab in zip(frames, labels):
        cv2.imwrite(str(clip / out / f.name), lab)
    print(f"seg_clip: {len(frames)} frames segmented")


if __name__ == "__main__":
    main()
