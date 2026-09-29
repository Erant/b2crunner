"""A face-crop identity reference for close-up clips: the Sapiens2 face/hair box of a front view (a run's
ply/front.png), squared with a margin and upscaled (Lanczos) to 768 px. Runs in the wan22 environment::

    python3 tools/run.py face_reference <run>/ply/front.png OUT.png [--size 768]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout
import pipeline.steps  # noqa: E402,F401
from pipeline.registry import get_step_class  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--size", type=int, default=768)
a = ap.parse_args()
img = cv2.imread(str(a.image), cv2.IMREAD_COLOR)
if img is None:
    raise SystemExit(f"face_reference: cannot read {a.image}")
cls = get_step_class("sapiens2_seg"); params = cls.resolve_params({"dtype": "bfloat16"}); step = cls(); step.load(params)
lab = step.run({"images": [img]}, params)["labels"][0]
ys, xs = np.nonzero(np.isin(lab, (3, 4, 24, 25)))          # face, hair, lips
face_y = np.nonzero(np.isin(lab, (3,)))[0]
y0, y1 = ys.min(), face_y.max()                              # hair top to the chin
x0, x1 = xs.min(), xs.max()
side = int(max(y1 - y0, x1 - x0) * 1.35); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
top, left = max(cy - side // 2, 0), max(cx - side // 2, 0)
crop = img[top:top + side, left:left + side]
ref = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(str(a.out), ref)
print(f"face_reference: {side}px box at ({left},{top}) -> {a.out}")
