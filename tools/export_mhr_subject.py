"""Export a delivered splat's MHR body as mhr.npz, so a consumer (b2crig) can pose it without SAM-3D-Body.

Runs once per subject, in the sam3dbody environment (it needs `sam_3d_body` for
the head's scale/hand buffers)::

    python3 tools/run.py export_mhr_subject <run>/ply/scene.ply OUT/mhr.npz

It replays the header's pose parameters (`b2c.mhr.*`, pipeline/ply_meta.py)
through `head_fit.build_mhr_head`, and keeps the one tensor the TorchScript
`mhr_model.pt` actually consumes: the full `model_params` row (global trans x10,
global rot, 130 body params with the hand PCA already expanded into their
slots, then the 68 scales). b2crig then drives `mhr_model.pt` directly by
editing the body slots of that row and leaving the hand slots alone
(`hand_idx`), which is the same thing `mhr_forward` does.

Checks: the replay must reproduce the header's world joints to < 1 mm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout

from pipeline import ply_meta  # noqa: E402
from pipeline.steps.head_fit import FLIP, build_mhr_head, rig_binding_data  # noqa: E402

REPO = "facebook/sam-3d-body-dinov3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    body = ply_meta.parse_body_comments(ply_meta.read_comments(args.ply))
    if not body:
        raise SystemExit(f"{args.ply} carries no b2c.mhr.* body record")
    pose = body["pose_params"]
    wfr = body["world_from_raw"]
    scale = float(np.asarray(wfr["scale"]).reshape(-1)[0])
    rot = np.asarray(wfr["rotation"], np.float64).reshape(3, 3)
    trans = np.asarray(wfr["translation"], np.float64).reshape(3)

    head = build_mhr_head(REPO, None, None, args.device)
    dev = args.device

    def t(key, default_len=None):
        v = pose.get(key)
        if v is None:
            return torch.zeros(1, default_len, device=dev)
        out = torch.as_tensor(np.asarray(v, np.float32), device=dev)
        return out[None] if out.ndim == 1 else out

    n_scales = int(head.scale_mean.shape[0])
    with torch.no_grad():
        verts, joints, model_params, rots = head.mhr_forward(
            global_trans=t("global_trans", 3), global_rot=t("global_rot"),
            body_pose_params=t("body_pose_params"), hand_pose_params=t("hand_pose_params"),
            scale_params=t("scale_params"), shape_params=t("shape_params"),
            expr_params=t("expr_params"), scale_offsets=t("scale_offsets", n_scales),
            return_joint_coords=True, return_model_params=True, return_joint_rotations=True,
        )
    flip = np.asarray(FLIP, np.float64)

    def to_world(raw):  # raw: mhr_forward output (metres, before FLIP)
        return scale * (raw * flip) @ rot.T + trans

    verts_w = to_world(verts[0].double().cpu().numpy())
    joints_w = to_world(joints[0].double().cpu().numpy())
    err = np.abs(joints_w - np.asarray(body["joints"], np.float64).reshape(-1, 3)).max()
    print(f"replay vs header joints: max {err * 1000:.3f} mm")
    if err > 1e-3:
        raise SystemExit("replay does not reproduce the header's joints; wrong model or frame")

    rig = rig_binding_data(head.mhr)
    np.savez(
        args.out,
        model_params=model_params[0].cpu().numpy().astype(np.float32),
        shape_params=t("shape_params")[0].cpu().numpy(),
        expr_params=t("expr_params")[0].cpu().numpy(),
        hand_idx=np.concatenate([head.hand_joint_idxs_left.cpu().numpy(),
                                 head.hand_joint_idxs_right.cpu().numpy()]).astype(np.int64),
        wfr_scale=np.float64(scale), wfr_rotation=rot, wfr_translation=trans, flip=flip,
        verts_world=verts_w.astype(np.float32), joints_world=joints_w.astype(np.float32),
        global_rots_raw=rots[0].cpu().numpy(),
        faces=rig["faces"].astype(np.int32), joint_parents=rig["joint_parents"].astype(np.int32),
        skin_vertex=rig["skin_vertex"], skin_joint=rig["skin_joint"], skin_weight=rig["skin_weight"],
        mhr_model=str(Path(head.model_data_dir).resolve()),
    )
    print(f"wrote {args.out}: {len(verts_w)} verts, {len(joints_w)} joints, "
          f"model_params {tuple(model_params.shape)}")


if __name__ == "__main__":
    main()
