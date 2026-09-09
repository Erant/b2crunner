#!/usr/bin/env python3
"""Score a run's FINAL deliverable, and the pass-2 step that made its frames.

Usage:
    scripts/final_splat_quality.py <result-dir> [<result-dir> ...]

Needs a b2ctrain checkout built locally (`~/Projects/b2ctrain/build/b2ctrain`,
or $B2CTRAIN) — the same rasteriser the pod uses as `brush-splat-render` —
and a result dir unpacked from a run archive with `export_debug` and
`export_ply` on. Renders go under $FSQ_WORK (default: a `.fsq/` beside the
result dir), every 4th camera, so a run scores in about half a minute.

Two renders, two halves of the question (2026-09-08):

  * `ply/scene.ply` at the FINAL training cameras (`colmap/`), on 0.5 grey —
    the deliverable, at the views it was fitted to. `splat_s1` and
    `splat_head_s1` are band-limited sharpness (Laplacian variance after a
    sigma-1 blur — the grain-insensitive metric of
    docs/final-splat-alignment-guide.md) inside the frame's own matte, eroded
    5 px so the silhouette edge does not score; `head` is the top 16% of the
    subject's extent. `psnr` is the render against the frame in the same
    matte — the guide's `fidelity`, the number that RISES when detail is
    recovered and FALLS when frames merely disagree more. `frame_*` is the
    same sharpness on the training frames themselves: what the splat had to
    work with.

  * `debug/intermediate_splat.ply` at the PRE-UPSCALE cameras
    (`colmap_preupscale/`), with the confidence gate and the cull colour
    `rerender_splat` uses (tau 0.3, angle margin 45, 0.5 grey) — a
    reconstruction of pass 2's CONTROL video, since the archive does not
    carry it. `ctl_*` is its sharpness and `p2_*` the sharpness of pass 2's
    output (`colmap_preupscale/images`), both measured ONLY where the
    control is fully kept: the gate's culled holes have hard grey edges
    that score as detail, and with them in the control's head read 44.5
    against the output's 36.1 on E4 when the two are in fact level.
    `p2_psnr_vs_control` is how far pass 2 moved from what it was handed,
    on the same kept pixels.

Plus what the archive already says: `flow` is `debug/alignment/alignment.json`'s
iteration-1 mean (per-pixel view disagreement, pixels, lower is better) and
`ba_inflation` is refine_cameras_final's `ba_scale_inflation` (a run whose
final refinement was refused reports `refine: refused`).

Reading it, from the 2026-09-08 archives: pass 2 at flat 0.8 / shift 2.5
neither adds nor removes head sharpness where the control had detail — it
fills the gate's holes and adds body texture (control 26-35 -> frames
32-35). ef13a7, whose pass 2 let go to 0 on the last step, made the sharpest
frames measured (body 53.9) and a final splat no sharper than anyone's, on
the worst flow (1.376): frames that disagree do not survive the fit.
Sharpness and fidelity have to move together to mean anything.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np

B2CTRAIN = os.environ.get("B2CTRAIN", os.path.expanduser("~/Projects/b2ctrain/build/b2ctrain"))
EVERY = 4
ERODE = 5
HEAD_FRAC = 0.16
# rerender_splat's confidence flags (pipeline/workflows/fast_helical_native.yaml).
CONTROL_FLAGS = ["--confidence", "--cull-color", "0.5,0.5,0.5",
                 "--conf-tau", "0.3", "--conf-angle-margin", "45"]


def _quat_to_r(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def colmap_to_cameras(colmap_dir: str, out_json: str, every: int) -> list:
    """A COLMAP text model as b2ctrain's cameras.json (OpenGL camera-to-world).

    The same arithmetic as b2ctrain's bench/colmap_to_cameras.py, inlined so
    this script needs only the binary.
    """
    cams = {}
    for line in open(os.path.join(colmap_dir, "cameras.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = (t[1], int(t[2]), int(t[3]), [float(v) for v in t[4:]])
    images = []
    for line in open(os.path.join(colmap_dir, "images.txt")):
        t = line.split()
        if not t or t[0].startswith("#") or len(t) < 10:
            continue
        images.append((t[9], [float(v) for v in t[1:8]], int(t[8])))
    images.sort(key=lambda x: x[0])
    images = images[::every]
    model, width, height, params = cams[images[0][2]]
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise SystemExit(f"unsupported camera model {model}")
    out = {"width": width, "height": height, "cameras": []}
    for name, p, _cid in images:
        qw, qx, qy, qz, tx, ty, tz = p
        r = _quat_to_r(qw, qx, qy, qz)
        pos = -r.T @ np.array([tx, ty, tz])
        out["cameras"].append({"name": name, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                               "position": pos.tolist(),
                               "rotation": (r.T @ np.diag([1, -1, -1])).tolist()})
    json.dump(out, open(out_json, "w"), indent=1)
    return [c["name"] for c in out["cameras"]]


def render(ply: str, cameras_json: str, out_dir: str, extra: list) -> None:
    subprocess.run([B2CTRAIN, "render", "--splat", ply, "--cameras", cameras_json,
                    "--output-dir", out_dir, *extra], check=True, capture_output=True)


def _s1(gray: np.ndarray, mask: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return float(cv2.Laplacian(blurred, cv2.CV_64F)[mask].var())


def _matte(frame: np.ndarray):
    """The frame's own alpha, eroded, and its head band; None if too small."""
    if frame.ndim < 3 or frame.shape[2] < 4:
        return None
    alpha = (frame[:, :, 3] > 128).astype(np.uint8)
    alpha = cv2.erode(alpha, np.ones((ERODE, ERODE), np.uint8)).astype(bool)
    if alpha.sum() < 2000:
        return None
    ys = np.where(alpha)[0]
    head = alpha.copy()
    head[int(ys.min() + HEAD_FRAC * (ys.max() - ys.min())):] = False
    return alpha, head


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)


def compare(frames_dir: str, renders_dir: str, names: list, prefix: str, other: str,
            kept_only: bool = False) -> dict:
    """Sharpness of frames and renders inside the frame's matte, and their PSNR.

    With `kept_only` the matte is also cut to where the RENDER's alpha is
    fully kept (eroded 7 px). That is for the confidence-gated control: its
    culled holes have hard grey edges that score as sharpness, and measured
    both ways on E4 (2026-09-08) the control's head went from 44.5 to level
    with pass 2's output once those edges were excluded.
    """
    rows = []
    for name in names:
        frame = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_UNCHANGED)
        rendered = cv2.imread(os.path.join(renders_dir, os.path.splitext(name)[0] + ".png"),
                              cv2.IMREAD_UNCHANGED)
        if frame is None or rendered is None:
            continue
        m = _matte(frame)
        if m is None:
            continue
        alpha, head = m
        if kept_only:
            kept = (rendered[:, :, 3] > 250).astype(np.uint8)
            kept = cv2.erode(kept, np.ones((7, 7), np.uint8)).astype(bool)
            alpha, head = alpha & kept, head & kept
            if alpha.sum() < 2000 or head.sum() < 300:
                continue
        gf, gr = _gray(frame), _gray(rendered)
        mse = float(((gf - gr) ** 2)[alpha].mean())
        rows.append({f"{prefix}_s1": _s1(gf, alpha), f"{prefix}_head_s1": _s1(gf, head),
                     f"{other}_s1": _s1(gr, alpha), f"{other}_head_s1": _s1(gr, head),
                     "psnr": 10 * np.log10(255 ** 2 / max(mse, 1e-6))})
    if not rows:
        return {}
    return {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0]} | {"views": len(rows)}


def archive_stats(run: str) -> dict:
    out = {}
    try:
        it = json.load(open(os.path.join(run, "debug/alignment/alignment.json")))["iterations"]
        out["flow"] = round(it[0]["mean"], 3)
    except (OSError, KeyError, IndexError, ValueError):
        out["flow"] = None
    try:
        st = json.load(open(os.path.join(run, "debug/refine_cameras_final/stats.json")))["stats"]
        out["ba_inflation"] = round(st["ba_scale_inflation"], 3) if st.get("accepted") else "refine: refused"
    except (OSError, KeyError, ValueError):
        out["ba_inflation"] = None
    return out


def score(run: str, work: str) -> dict:
    os.makedirs(work, exist_ok=True)
    result = {"run": os.path.basename(run)}
    # 1. the deliverable at its own training views
    final_cams = os.path.join(work, "cameras_final.json")
    names = colmap_to_cameras(os.path.join(run, "colmap"), final_cams, EVERY)
    final_dir = os.path.join(work, "final")
    render(os.path.join(run, "ply/scene.ply"), final_cams, final_dir, ["--background", "0.5,0.5,0.5"])
    final = compare(os.path.join(run, "colmap/images"), final_dir, names, "frame", "splat")
    result.update({k if k != "views" else "final_views": v for k, v in final.items()})
    # 2. pass 2's output against a reconstruction of its control
    inter_ply = os.path.join(run, "debug/intermediate_splat.ply")
    pre_dir = os.path.join(run, "colmap_preupscale")
    if os.path.exists(inter_ply) and os.path.isdir(pre_dir):
        pre_cams = os.path.join(work, "cameras_preupscale.json")
        names = colmap_to_cameras(pre_dir, pre_cams, EVERY)
        ctl_dir = os.path.join(work, "control")
        render(inter_ply, pre_cams, ctl_dir, CONTROL_FLAGS)
        p2 = compare(os.path.join(pre_dir, "images"), ctl_dir, names, "p2", "ctl", kept_only=True)
        p2["p2_psnr_vs_control"] = p2.pop("psnr", None)
        result.update({k if k != "views" else "p2_views": v for k, v in p2.items()})
    result.update(archive_stats(run))
    return result


def main(argv: list) -> int:
    if not argv or not os.path.exists(B2CTRAIN):
        print(__doc__ if not argv else f"no rasteriser at {B2CTRAIN} (set $B2CTRAIN)")
        return 2
    for run in argv:
        run = os.path.abspath(run.rstrip("/"))
        work = os.path.join(os.environ.get("FSQ_WORK", os.path.join(os.path.dirname(run), ".fsq")),
                            os.path.basename(run))
        try:
            print(json.dumps(score(run, work)))
        except subprocess.CalledProcessError as exc:
            print(json.dumps({"run": os.path.basename(run), "error": exc.stderr.decode()[-500:]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
