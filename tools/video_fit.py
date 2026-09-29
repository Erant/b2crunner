"""SAM-3D-Body on every frame of a clip: per-frame MHR parameters, camera-frame joints and root rotation.

Runs in the sam3dbody environment (sam_3d_body and head_fit's imports)::

    python3 tools/run.py video_fit CLIP_DIR [--frames wan]

The person box comes from the clip's Sapiens2 seg (seg/), the intrinsics from its cameras.json, so neither the
detector nor MoGe is needed. Writes CLIP_DIR/video_fit.npz: model_params [T, 204] (the TorchScript row, hand PCA
expanded, as tools/export_mhr_subject.py), cam_t [T, 3], joints [T, J, 3] (camera frame, + cam_t), global_rots
[T, J, 3, 3], expr [T, 72].
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout

from pipeline.steps.head_fit import build_mhr_head  # noqa: E402

REPO = "facebook/sam-3d-body-dinov3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--frames", default="wan")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    ck = Path(snapshot_download(REPO))
    model, cfg = load_sam_3d_body(str(ck / "model.ckpt"), device="cuda", mhr_path=str(ck / "assets" / "mhr_model.pt"))
    est = SAM3DBodyEstimator(model, cfg)
    head = build_mhr_head(REPO, None, None, "cuda")
    cams = json.loads((a.clip / "cameras.json").read_text())
    frames = sorted(f for f in (a.clip / a.frames).glob("*.png") if f.stem.isdigit())
    out = {k: [] for k in ("model_params", "cam_t", "joints", "global_rots", "expr", "bbox")}
    n_scales = int(head.scale_mean.shape[0])
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            c = cams["cameras"][min(i, len(cams["cameras"]) - 1)]
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float32)
            seg = cv2.imread(str(a.clip / "seg" / f.name), 0)
            ys, xs = np.nonzero(seg > 0)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            pad = 0.05 * max(x1 - x0, y1 - y0)
            box = np.array([[x0 - pad, y0 - pad, x1 + pad, y1 + pad]], np.float32)
            p = Path(tmp) / "f.png"
            cv2.imwrite(str(p), cv2.imread(str(f)))
            res = est.process_one_image(str(p), bboxes=box, cam_int=torch.as_tensor(K)[None])
            if not res:
                raise SystemExit(f"{f.name}: no person")
            r = res[0]

            def t(v, n=None):
                x = torch.as_tensor(np.asarray(v, np.float32), device="cuda")
                return x[None] if x.ndim == 1 else x
            with torch.no_grad():
                _, _, mp, _ = head.mhr_forward(
                    global_trans=torch.zeros(1, 3, device="cuda"), global_rot=t(r["global_rot"]),
                    body_pose_params=t(r["body_pose_params"]), hand_pose_params=t(r["hand_pose_params"]),
                    scale_params=t(r["scale_params"]), shape_params=t(r["shape_params"]), expr_params=t(r["expr_params"]),
                    scale_offsets=torch.zeros(1, n_scales, device="cuda"),
                    return_joint_coords=True, return_model_params=True, return_joint_rotations=True)
            out["model_params"].append(mp[0].cpu().numpy())
            out["cam_t"].append(np.asarray(r["pred_cam_t"], np.float32))
            out["joints"].append(np.asarray(r["pred_joint_coords"], np.float32) + np.asarray(r["pred_cam_t"], np.float32))
            out["global_rots"].append(np.asarray(r["pred_global_rots"], np.float32))
            out["expr"].append(np.asarray(r["expr_params"], np.float32))
            out["bbox"].append(box[0])
            if i % 10 == 0:
                print(f"video_fit: {f.name}", flush=True)
    np.savez(a.clip / "video_fit.npz", **{k: np.stack(v) for k, v in out.items()})
    print(f"video_fit: {len(frames)} frames -> {a.clip / 'video_fit.npz'}")


if __name__ == "__main__":
    main()
