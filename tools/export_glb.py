"""Write a run's subject file (b2cgltf SPEC.md section 4): one .glb with the splat, the MHR body and the capture.

    python3 tools/run.py export_glb RUN OUT.glb [--ply PLY] [--device cuda] [--no-view-cameras]

RUN is a delivered run directory (`ply/scene.ply`, the images beside it, `colmap/`). This is the migration path of
SPEC 4.6: it is lenient about what older runs lack and records it.
- A version-1 body record has its rotations converted (`sourceRecordVersion: 1`).
- Without an orbit record it writes `B2C_orbit` with `"migrated": true` and only the final cameras, read from
  `colmap/`.
The body record itself is required (a run without one needs tools/recover_body.py first).

The body is replayed through MHR (tools/export_mhr_subject.replay, the same < 1 mm check), so this runs in the
sam3dbody environment, and b2cgltf must be importable there (B2CRUNNER_PATH_SAM3DBODY, or installed).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this b2crunner checkout
sys.path.insert(0, str(Path(__file__).resolve().parent))       # the sibling tools

from b2cgltf.b2crunner import convert, subject  # noqa: E402
from pipeline import orbit_record, ply_meta  # noqa: E402
from export_mhr_subject import REPO, replay  # noqa: E402
from recover_body import colmap_cameras  # noqa: E402

CAMERA_TOL = 1e-4   # metres / rotation entries: the record's final cameras against colmap/


def colmap_final(colmap_dir: Path) -> subject.Cameras:
    cams = colmap_cameras(colmap_dir)
    return subject.Cameras(
        rotation=np.array([c.rotation for c in cams]), position=np.array([c.position for c in cams]),
        intrinsics=np.array([[c.fx, c.fy, c.cx, c.cy] for c in cams]),
        image_size=(cams[0].width, cams[0].height), names=[c.name for c in cams], dataset="../colmap/")


def record_cameras(group: dict, names=None, dataset=None) -> subject.Cameras:
    return subject.Cameras(rotation=np.asarray(group["rotation"]), position=np.asarray(group["position"]),
                           intrinsics=np.asarray(group["intrinsics"]),
                           image_size=tuple(int(v) for v in np.asarray(group["image_size"]).reshape(-1)),
                           names=names, dataset=dataset)


def name_final_cameras(final: subject.Cameras, colmap: subject.Cameras) -> subject.Cameras:
    """The record's final cameras carry no image names: take them from colmap/, which must hold the same cameras
    in the same order (SPEC 4.4)."""
    if len(final.position) != len(colmap.position):
        raise SystemExit(f"final_cameras has {len(final.position)} cameras, colmap/ {len(colmap.position)}")
    dp = np.abs(np.asarray(final.position) - colmap.position).max()
    dr = np.abs(np.asarray(final.rotation) - colmap.rotation).max()
    if dp > CAMERA_TOL or dr > CAMERA_TOL:
        raise SystemExit(f"final_cameras and colmap/ disagree (position {dp:.3g}, rotation {dr:.3g}); "
                         "not naming the cameras by order")
    final.names, final.dataset = colmap.names, colmap.dataset
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ply", type=Path, help="the splat (default RUN/ply/scene.ply)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-view-cameras", action="store_true", help="skip the glTF camera nodes (viewing only)")
    args = ap.parse_args()
    ply = args.ply or args.run / "ply" / "scene.ply"

    fields, comments = convert.read_trainer_ply(ply)
    r = replay(ply, args.device)
    record = r.body
    version = int(record.get("version", 1))
    rots = np.asarray(record["global_rots"], np.float64).reshape(-1, 3, 3)
    if version == 1:
        rots = convert.global_rots_v1_to_v2(r.rot, rots)
    elif version != 2:
        raise SystemExit(f"body record version {version} is not one this tool knows")
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download(REPO, local_files_only=True)) / subject.MHR_FILE
    rig = r.rig
    body = subject.Body(
        verts=r.verts_w, faces=rig["faces"], skin_vertex=rig["skin_vertex"], skin_joint=rig["skin_joint"],
        skin_weight=rig["skin_weight"], joint_parents=np.asarray(record["joint_parents"]),
        joints=np.asarray(record["joints"]), global_rots=rots, model_sha256=orbit_record.sha256_file(model),
        world_from_raw={"scale": r.scale, "rotation": r.rot, "translation": r.trans},
        pose_params=r.pose_params, model_params=r.model_params, hand_idx=r.hand_idx,
        source_record_version=version, joint_names=list(r.head.mhr.character_torch.skeleton.joint_names))
    if not np.array_equal(np.asarray(record["joint_parents"]).reshape(-1), rig["joint_parents"].reshape(-1)):
        raise SystemExit("the record's joint_parents differ from the MHR model's")

    colmap = colmap_final(args.run / "colmap") if (args.run / "colmap" / "images.txt").exists() else None
    orbit = orbit_record.parse_orbit_comments(comments)
    if orbit:
        record_images = orbit_record.read_orbit_record(ply).get("images") or {}   # checks the sha256s
        if "final_cameras" not in orbit:
            raise SystemExit("the orbit record has no final_cameras")
        if colmap is None:
            raise SystemExit("naming final_cameras needs RUN/colmap/images.txt")
        capture = subject.Capture(
            final=name_final_cameras(record_cameras(orbit["final_cameras"]), colmap),
            orbit=record_cameras(orbit["orbit_cameras"]) if "orbit_cameras" in orbit else None,
            helix=orbit.get("helix"), extension=orbit.get("extension"), pass_frames=orbit.get("pass_frames"),
            anchor_frame_index=orbit.get("anchor_frame_index"), extras=orbit.get("extras"),
            prompt=orbit.get("prompt"), settings=orbit.get("settings"),
            images={k: Path(v["path"]).read_bytes() for k, v in record_images.items()})
    else:
        if colmap is None:
            raise SystemExit(f"{ply} has no orbit record and RUN has no colmap/: no final cameras to write")
        images = {k: (ply.parent / f).read_bytes() for k, f in orbit_record.IMAGE_FILES.items()
                  if (ply.parent / f).exists()}
        capture = subject.Capture(final=colmap, images=images, migrated=True)
        print(f"no orbit record: migrated B2C_orbit with colmap's {len(colmap.position)} final cameras"
              f"{' and ' + ', '.join(images) if images else ''}")

    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    path = subject.write_subject(args.out, fields, body, capture, tool="b2crunner export_glb", commit=commit,
                                 view_cameras=not args.no_view_cameras)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB): {len(fields['x'])} splats, "
          f"{len(body.verts)} body vertices, {len(body.joints)} joints, body record v{version}, "
          f"{'orbit record' if orbit else 'migrated capture'}")


if __name__ == "__main__":
    main()
