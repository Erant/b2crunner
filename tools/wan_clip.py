"""Run b2crunner's wan22_vace_denoise step on one clip directory.

Runs in the wan22 environment (diffusers and the step's fp8 loader)::

    python3 tools/run.py wan_clip CLIP_DIR

CLIP_DIR holds:
    control/NNNN.png   RGBA renders (b2ctrain render --cage); alpha is composited on `background`
    mask/NNNN.png      optional grey VACE masks, 0 = keep the control pixel, 255 = generate
    reference.png      identity reference
    wan.json           {"params": {...step params...}, "background": [r, g, b]}
Writes wan/NNNN.png and wan/timing.json.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout
import pipeline.steps  # noqa: E402,F401  (registers every step)
from pipeline.registry import get_step_class  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    clip = Path(sys.argv[1])
    cfg = json.loads((clip / "wan.json").read_text())
    bg = np.asarray(cfg.get("background", [0.5, 0.5, 0.5]), np.float32) * 255
    frames = sorted((clip / "control").glob("*.png"))
    frames = [f for f in frames if f.name.count(".") == 1]   # skip <stem>.mask.png / .labels.png sidecars
    video = []
    for f in frames:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32)
        a = im[..., 3:4] / 255.0
        video.append(np.clip(im[..., :3] * a + bg[::-1] * (1 - a), 0, 255).astype(np.uint8))   # BGR
    masks = []
    for f in frames:
        m = clip / "mask" / f.name
        masks.append(cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) if m.exists()
                     else np.full(video[0].shape[:2], 255, np.uint8))
    ref = cv2.imread(str(clip / "reference.png"), cv2.IMREAD_COLOR)

    cls = get_step_class("wan22_vace_denoise")
    params = cls.resolve_params(cfg["params"])
    step = cls()
    t0 = time.time()
    step.load(params)
    t1 = time.time()
    out = step.run({"control_video": video, "control_masks": masks, "reference_image": ref,
                    "subject_desc": cfg.get("subject_desc")}, params)
    t2 = time.time()
    (clip / "wan").mkdir(exist_ok=True)
    for f, im in zip(frames, out["images"]):
        cv2.imwrite(str(clip / "wan" / f.name), im)
    (clip / "wan" / "timing.json").write_text(json.dumps(
        {"load_s": t1 - t0, "run_s": t2 - t1, "frames": len(frames),
         "width": params["width"], "height": params["height"]}))
    print(f"wan_clip: {len(frames)} frames {params['width']}x{params['height']}: load {t1 - t0:.0f}s, run {t2 - t1:.0f}s")


if __name__ == "__main__":
    main()
