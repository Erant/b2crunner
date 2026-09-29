"""Recover a run's MHR body fit when its scene.ply lacks the `b2c.mhr.*` header (the pipeline died before embedding it).

    python3 tools/run.py recover_body <run> OUT.ply [--views 3] [--work DIR]

Replays b2crunner's own steps on what the run left behind (colmap/ + ply/scene.ply):
1. splat_surface: `b2ctrain probe --depth` over the colmap cameras -> oriented surface points (tau 0.5, as the run).
2. sam3d_body on the `--views` most frontal colmap images, with the colmap intrinsics as `cam_int` (so the fit is in
   that camera's metric frame); each fit is placed in the world by its camera, then rigidly ICP'd onto the surface;
   the fit with the smallest surface distance wins.
3. refit_body_to_splat with the run's parameters (b2crunner defaults).
Writes OUT.ply = the splat with the `b2c.mhr.*` comments (pipeline/ply_meta.body_comments), ready for
tools/export_mhr_subject.py. b2ctrain is `--trainer` ($B2CTRAIN, default `b2ctrain` on PATH, as the pipeline). On a run that HAS the header, `--compare` prints joint distances to the original.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout
B2CTRAIN = os.environ.get("B2CTRAIN", "b2ctrain")
GL = np.diag([1.0, -1.0, -1.0])


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def colmap_cameras(ds: Path) -> list:
    """body2colmap-style cameras: OpenGL camera-to-world `rotation`, `position`, intrinsics, plus the image name."""
    p = [ln.split() for ln in (ds / "cameras.txt").read_text().splitlines() if ln and not ln.startswith("#")][0]
    W, H, fx, fy, cx, cy = int(p[2]), int(p[3]), *map(float, p[4:8])
    cams = []
    for ln in (ds / "images.txt").read_text().splitlines():
        f = ln.split()
        if ln.startswith("#") or len(f) < 10:
            continue
        Rw2c, t = qvec_to_R(list(map(float, f[1:5]))), np.array(list(map(float, f[5:8])))
        cams.append(SimpleNamespace(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy, name=f[9],
                                    rotation=Rw2c.T @ GL, position=-Rw2c.T @ t, Rw2c=Rw2c, t=t))
    return cams


def surface(splat: Path, cams: list, work: Path, every: int, trainer: str = B2CTRAIN) -> tuple[np.ndarray, np.ndarray]:
    from pipeline.steps.body_refit import cameras_json, decode_depth_png, unproject_depth
    import cv2
    work.mkdir(parents=True, exist_ok=True)
    (work / "cameras.json").write_text(json.dumps(cameras_json(cams)))
    subprocess.run([str(trainer), "probe", "--splat", str(splat), "--cameras", str(work / "cameras.json"),
                    "--output-dir", str(work), "--depth", "--tau", "0.5", "--every", str(every)],
                   check=True, capture_output=True)
    P, N = [], []
    for f in sorted(work.glob("frame_*.zfirst.png")):
        p, n = unproject_depth(decode_depth_png(cv2.imread(str(f), cv2.IMREAD_UNCHANGED)),
                               cams[int(f.name[6:11])], 8, 0.02)
        P.append(p); N.append(n)
    P, N = np.concatenate(P), np.concatenate(N)
    if len(P) > 300000:
        k = np.sort(np.random.RandomState(0).choice(len(P), 300000, replace=False)); P, N = P[k], N[k]
    return P, N


def rigid_icp(V: np.ndarray, P: np.ndarray, iters: int = 30) -> tuple[np.ndarray, float]:
    """V (body mesh, world) rigidly onto the surface points P: trimmed nearest-point Procrustes, both directions."""
    from scipy.spatial import cKDTree
    from pipeline.steps.body_refit import umeyama
    tp = cKDTree(P)
    T = np.eye(4)
    for _ in range(iters):
        W = V @ T[:3, :3].T + T[:3, 3]
        d, i = tp.query(W)
        keep = d < np.percentile(d, 80)
        _, R, t = umeyama(W[keep], P[i[keep]], with_scale=False)
        step = np.eye(4); step[:3, :3] = R; step[:3, 3] = t
        T = step @ T
    W = V @ T[:3, :3].T + T[:3, 3]
    return T, float(np.median(tp.query(W)[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--splat", type=Path, help="default <run>/ply/scene.ply")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--every", type=int, default=2, help="probe every N-th camera")
    ap.add_argument("--work", type=Path)
    ap.add_argument("--compare", action="store_true", help="the splat has a header: compare the recovered fit")
    ap.add_argument("--trainer", default=B2CTRAIN, help="the b2ctrain binary")
    a = ap.parse_args()
    import cv2
    import torch
    from pipeline import ply_meta
    from pipeline.steps.body_refit import RefitBodyToSplatStep
    from pipeline.steps.sam3d_body import SAM3DBodyStep

    splat = a.splat or a.run / "ply" / "scene.ply"
    work = a.work or Path(tempfile.mkdtemp(prefix="recover_body_"))
    cams = colmap_cameras(a.run / "colmap")
    P, N = surface(splat, cams, work / "probe", a.every, a.trainer)
    print(f"surface: {len(P)} points from {len(cams)} cameras (every {a.every})")

    # The subject faces +Z (b2crunner's world): the most frontal cameras sit along +Z from the surface centroid, level.
    c = np.median(P, 0)
    dirs = np.array([(cam.position - c) / np.linalg.norm(cam.position - c) for cam in cams])
    score = dirs[:, 2] - 0.5 * np.abs(dirs[:, 1])
    order = np.argsort(-score)[:a.views]

    step = SAM3DBodyStep()
    sp = {p.name: p.default for p in step.PARAMS}
    sp["fov_estimator"] = ""          # the colmap intrinsics are given as cam_int
    step.load(sp)
    est = step._estimator
    best = None
    for vi in order:
        cam = cams[vi]
        img = a.run / "colmap" / "images" / cam.name
        K = torch.tensor([[[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]]], dtype=torch.float32)
        outs = est.process_one_image(str(img), cam_int=K, bbox_thr=sp["bbox_thr"])
        if not outs:
            print(f"view {cam.name}: no person"); continue
        o = outs[0]
        raw = np.asarray(o["pred_vertices"], np.float64)
        # raw + cam_t is the OpenCV camera frame of this view (y down, z forward)
        Xc = raw + np.asarray(o["pred_cam_t"], np.float64)
        Vw = Xc @ cam.Rw2c + cam.position           # = Rw2c^T Xc + C
        T, med = rigid_icp(Vw, P)
        print(f"view {cam.name} (score {score[vi]:.2f}): placed, surface median after rigid ICP {med * 1000:.1f} mm, "
              f"ICP moved {np.linalg.norm(T[:3, 3]) * 1000:.0f} mm / "
              f"{np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))):.1f} deg")
        if best is None or med < best[0]:
            best = (med, cam.name, o, Vw @ T[:3, :3].T + T[:3, 3])
    step.unload()
    med, name, o, Vw = best
    print(f"using {name}")

    mesh_output = {"vertices": np.asarray(o["pred_vertices"]), "faces": np.asarray(est.faces),
                   "keypoints_3d": np.asarray(o["pred_keypoints_3d"]),
                   "pose_params": {k: np.asarray(o[k]) for k in ("global_rot", "body_pose_params", "hand_pose_params",
                                                                  "scale_params", "shape_params", "expr_params")}}
    refit = RefitBodyToSplatStep()
    rp = {p.name: p.default for p in refit.PARAMS}
    rp["debug_dir"] = str(work / "refit")
    out = refit.run({"mesh_output": mesh_output, "mesh_world": (Vw.astype(np.float32), np.asarray(est.faces)),
                     "surface": {"points": P, "normals": N, "tau": 0.5}}, rp)
    s = out["body_refit_stats"]
    print("refit: surface-to-body median %.2f -> %.2f cm, p90 %.2f -> %.2f, inside>5mm %.1f%% -> %.1f%%, pose %.1f deg rms"
          % (s["before"]["surface_to_body_cm"]["median_abs"], s["after"]["surface_to_body_cm"]["median_abs"],
             s["before"]["surface_to_body_cm"]["p90_abs"], s["after"]["surface_to_body_cm"]["p90_abs"],
             100 * s["before"]["surface_to_body_cm"]["inside_fraction_5mm"],
             100 * s["after"]["surface_to_body_cm"]["inside_fraction_5mm"], s["pose_delta_deg"]["rms"]))

    bp = out["body_params"]
    lines = ply_meta.body_comments(bp["pose_params"], bp["world_from_raw"], joints=bp["joints"],
                                   global_rots=bp["global_rots"], joint_parents=bp["joint_parents"], model=bp["model"])
    if a.compare:
        ref = ply_meta.parse_body_comments(ply_meta.read_comments(splat))
        new = ply_meta.parse_body_comments(lines)
        d = np.linalg.norm(ref["joints"].reshape(-1, 3) - new["joints"].reshape(-1, 3), axis=1) * 1000   # world
        print(f"compare to the original header (world joints): median {np.median(d):.1f} mm, p90 "
              f"{np.percentile(d, 90):.1f}, max {d.max():.1f}")
    shutil.copyfile(splat, a.out)
    ply_meta.embed_comments(a.out, lines)
    print(f"wrote {a.out} ({len(lines)} b2c.mhr comments); work in {work}")


if __name__ == "__main__":
    main()
