#!/usr/bin/env python3
"""Measure how much of the VACE control skeleton survives into a run's frames.

Usage:
    scripts/skeleton_leak.py <result-dir> [<result-dir> ...]
    scripts/skeleton_leak.py --iou --by-hue <result-dir> [...]

A result dir is an unpacked run archive: it needs `debug/denoise_pass1_input/`
(the control video handed to stage 1, written when `export_debug` is on) and
`colmap_intermediate/images/` (the denoised frames the intermediate splat
trains on, written when `export_colmap_intermediate` is on).

What is being measured, and why this shape. The failure is skeleton ink: the
control's limb sticks arriving in the output as painted stripes rather than
as pose. Naively differencing the two images measures mostly the denoise
itself, and thresholding on saturation in the output flags the subject's own
maroon trousers. So instead, for each stick pixel:

  * the control says which pixels are stick (the sticks are near-pure hues
    over a flat grey silhouette, so an 8-bit chroma threshold isolates them
    cleanly — the warped anchor photo, the one genuinely colourful control
    frame, is dropped by SPLAT_MAX_FRAC),
  * the output's own pixels in an annulus just off the stick say what should
    have been there,
  * and the leftover is projected onto that stick's colour direction, with
    luminance removed.

The projection is what makes the number specific. A subject who happens to
be maroon does not score; maroon that lands exactly where a red stick was
drawn, and is not present a few pixels to either side, scores. Units are
8-bit chroma: ~15 is plainly visible ink, ~2 is invisible, 0 is clean. The
sign is meaningful — a run that scores slightly negative has no leak, not
an inverted one.

`--iou` adds the other half of the trade-off: silhouette agreement between
the control's mesh figure and the denoised frame's own alpha matte, on the
same camera. Leak falls whenever the control pushes less hard, so a leak
number alone cannot say whether a setting is better — this says what the
frames gave up to get there. It does not reach 1.0 even when nothing is
wrong: the mesh is a nude body, the output has hair, clothing and shoes, so
the honest reading is the spread across runs (0.85-0.87 over the 2026-09-08
sweep), not the absolute value. It needs the flat backdrop the shipped
workflow renders (`background: ""`) — against `background: grid` the figure
cannot be told from the walls and those frames are skipped, so a grid run
reports `iou_frames: 0`.

`--by-hue` splits the same measurement by the stick's own hue, which in
DWPose's palette is the same thing as splitting by limb: hue 0-15 is the
neck/shoulder yoke, 35-85 the legs, and so on. Measured 2026-09-08, the
yoke leaks 3-4x the rest and is the last artefact to go.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

CHROMA_MIN = 110       # 8-bit; sticks are near-pure hues, skin and hair sit below
CORE_ERODE = 1         # step in off the stick's antialiased edge
NEAR_DILATE = 3        # ... skip this halo when reading the background ...
FAR_DILATE = 9         # ... and read it out to here
SPLAT_MAX_FRAC = 0.03  # a broadly colourful control frame is the anchor photo
MIN_PIXELS = 500       # below this a frame says nothing; the sticks are ~12k px
VISIBLE = 8.0          # `cover` counts stick pixels leaking more than this

CTL_FILL = 114         # the silhouette fill render.py lays down under the sticks
CTL_FILL_TOL = 4       # ... and how far the antialiased edge of it strays
OUT_ALPHA = 128        # the denoised frame's matte, thresholded
MAX_SIL_FRAC = 0.5     # above this the figure was not what got found

_HUE_BANDS = [
    ("neck/shoulder (red)", 0, 15),
    ("arms (orange-yellow)", 15, 35),
    ("legs (green)", 35, 85),
    ("torso (cyan)", 85, 100),
    ("shins (blue)", 100, 135),
    ("head/hands (magenta)", 135, 180),
]


def chroma(img: np.ndarray) -> np.ndarray:
    return img.max(2).astype(np.float32) - img.min(2).astype(np.float32)


def hue_dir(img: np.ndarray) -> np.ndarray:
    """Unit colour direction per pixel, luminance removed."""
    f = img.astype(np.float32)
    f = f - f.mean(2, keepdims=True)
    return f / np.maximum(np.linalg.norm(f, axis=2, keepdims=True), 1e-6)


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def sticks(ctl: np.ndarray):
    """The control's stick pixels, or None if this frame has no drawing on it.

    The frames the anchor is injected into are a warped photograph and carry
    neither sticks nor a mesh silhouette; a broadly colourful frame is one of
    those. Both metrics below skip a frame on this one test, so both report
    on the same set.
    """
    mask = chroma(ctl) > CHROMA_MIN
    if mask.mean() > SPLAT_MAX_FRAC or mask.sum() < MIN_PIXELS:
        return None
    return mask


def frame_leak(ctl: np.ndarray, out: np.ndarray):
    """Per-pixel leak projection plus the mask it is valid on, or None."""
    stick = sticks(ctl)
    if stick is None:
        return None
    stick8 = stick.astype(np.uint8)
    core = cv2.erode(stick8, _disc(CORE_ERODE)).astype(bool)
    if core.sum() < MIN_PIXELS:
        core = stick
    ring = (
        cv2.dilate(stick8, _disc(FAR_DILATE)).astype(bool)
        & ~cv2.dilate(stick8, _disc(NEAR_DILATE)).astype(bool)
    ).astype(np.float32)

    # Background under each stick: the output's own annulus pixels, box-summed
    # over a window wide enough to always contain some of them.
    o = out.astype(np.float32)
    box = (2 * FAR_DILATE + 1,) * 2
    total = cv2.boxFilter(o * ring[..., None], -1, box, normalize=False)
    count = cv2.boxFilter(ring, -1, box, normalize=False)
    enough = count > 20
    background = total / np.maximum(count, 1e-6)[..., None]

    residual = o - background
    residual -= residual.mean(2, keepdims=True)
    valid = core & enough
    if valid.sum() < MIN_PIXELS:
        return None
    return (residual * hue_dir(ctl)).sum(2), valid


def control_silhouette(ctl: np.ndarray) -> np.ndarray:
    """The mesh figure in a control frame, sticks included — or None.

    The fill is a single flat grey against a flat wall, so a value window
    finds it; the sticks are drawn over it and have to be added back by
    chroma and closed over, or the figure comes out perforated. Taking the
    largest component last drops the soft shadow the backdrop casts on the
    wall, which is the one other thing in frame that is not the wall.
    """
    grey = cv2.cvtColor(ctl, cv2.COLOR_BGR2GRAY).astype(np.int32)
    body = (np.abs(grey - CTL_FILL) <= CTL_FILL_TOL) | (chroma(ctl) > CHROMA_MIN)
    closed = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, _disc(4))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    figure = (
        closed.astype(bool) if count < 2
        else labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    )
    # A `background: grid` render puts the fill grey on the walls too, and
    # the largest component is then the room. Measured on 2026-09-08's
    # cd7482: 0.87 of the frame against 0.149 for the same subject on a flat
    # wall. Refuse rather than report the room's IoU as the subject's.
    if figure.mean() > MAX_SIL_FRAC:
        return None
    return figure


def silhouette_iou(run_dir: str) -> dict:
    """Control-mesh vs denoised-matte silhouette IoU, averaged over frames."""
    ctl_dir = os.path.join(run_dir, "debug/denoise_pass1_input")
    out_dir = os.path.join(run_dir, "colmap_intermediate/images")
    scores = []
    for name in sorted(os.listdir(ctl_dir)):
        if not name.startswith("frame_"):
            continue
        out_path = os.path.join(out_dir, name)
        if not os.path.exists(out_path):
            continue
        ctl = cv2.imread(os.path.join(ctl_dir, name), cv2.IMREAD_COLOR)
        out = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
        if ctl is None or out is None or out.ndim != 3 or out.shape[2] != 4:
            continue
        if sticks(ctl) is None:
            continue
        mesh = control_silhouette(ctl)
        if mesh is None:
            continue
        matte = out[..., 3] > OUT_ALPHA
        union = (mesh | matte).sum()
        if union:
            scores.append(float((mesh & matte).sum()) / float(union))
    return {"iou_frames": len(scores),
            "silhouette_iou": round(float(np.mean(scores)), 4) if scores else None}


def _frames(run_dir: str):
    ctl_dir = os.path.join(run_dir, "debug/denoise_pass1_input")
    out_dir = os.path.join(run_dir, "colmap_intermediate/images")
    for name in sorted(os.listdir(ctl_dir)):
        if not name.startswith("frame_"):
            continue
        out_path = os.path.join(out_dir, name)
        if not os.path.exists(out_path):
            continue
        ctl = cv2.imread(os.path.join(ctl_dir, name), cv2.IMREAD_COLOR)
        out = cv2.imread(out_path, cv2.IMREAD_COLOR)
        if ctl is None or out is None or ctl.shape != out.shape:
            continue
        measured = frame_leak(ctl, out)
        if measured is not None:
            yield name, ctl, measured


def measure(run_dir: str) -> dict:
    means, p90s, covers = [], [], []
    for _name, _ctl, (proj, valid) in _frames(run_dir):
        p = proj[valid]
        means.append(float(p.mean()))
        p90s.append(float(np.percentile(p, 90)))
        covers.append(float((p > VISIBLE).mean()))
    if not means:
        return {"run": os.path.basename(run_dir), "frames": 0}
    return {
        "run": os.path.basename(run_dir),
        "frames": len(means),
        "leak_mean": round(float(np.mean(means)), 2),
        "leak_p90": round(float(np.mean(p90s)), 2),
        "cover": round(float(np.mean(covers)), 4),
    }


def measure_by_hue(run_dir: str) -> dict:
    acc = {label: [] for label, _lo, _hi in _HUE_BANDS}
    for _name, ctl, (proj, valid) in _frames(run_dir):
        hue = cv2.cvtColor(ctl, cv2.COLOR_BGR2HSV)[..., 0]
        for label, lo, hi in _HUE_BANDS:
            band = valid & (hue >= lo) & (hue < hi)
            if band.sum() > 200:
                acc[label].append(float(proj[band].mean()))
    return {
        label: (round(float(np.mean(v)), 2) if v else None) for label, v in acc.items()
    }


def main(argv: list[str]) -> int:
    by_hue = "--by-hue" in argv
    want_iou = "--iou" in argv
    dirs = [a for a in argv if not a.startswith("--")]
    if not dirs:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    for run_dir in dirs:
        row = measure(run_dir)
        if want_iou:
            row.update(silhouette_iou(run_dir))
        if by_hue and row["frames"]:
            row["by_hue"] = measure_by_hue(run_dir)
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
