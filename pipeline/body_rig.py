"""The body rig b2ctrain deforms the splat with (`--body-rig`, its gpu/deform.h).

Per training view the trainer learns a small rotation per joint of the refit
MHR body and renders every splat at its skinned position, so the frames'
per-segment limb motion (the generated video swings the arms by centimetres
between parts of the orbit) is explained per view instead of averaged into a
double limb. The file it reads carries, all in the dataset's world frame:

- a subsample of the refit mesh's vertices with their top-4 skinning joints
  and weights (the trainer binds each splat to its nearest one),
- the joint tree (parents, parent-first order) and the canonical joint
  positions (the rotation pivots),
- the ACTIVE joints: the ones that get a per-view rotation. Measured
  (b2ctrain docs/STATUS.md, "Per-view arm rotations"): every joint whose
  subtree is skinned to at least `min_subtree` vertices EXCEPT the root
  chain — joints whose subtree is more than `max_subtree_fraction` of the
  body (root, pelvis, spine). A rotation there moves the whole body per view
  and was measured to destroy sharpness everywhere; with them excluded every
  region sharpened and the legs and torso, which do not move, stayed put.
- the training view names, to match the trainer's frames.

Binary layout (little-endian), version 2::

    "B2CRIG2\\0"  int32[6] {n_verts, n_joints, n_views, 4, 64, n_active}
    float32[n_verts*3] verts  int32[n_verts*4] joints  float32[n_verts*4] weights
    char[n_views*64] names (NUL padded)
    int32[n_joints] parents  float32[n_joints*3] joint_positions  int32[n_active] active
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

MAGIC = b"B2CRIG2\0"
NAME_LEN = 64
K = 4


def build_body_rig(vertices: np.ndarray, joint_positions: np.ndarray, parents: np.ndarray,
                   skin_vertex: np.ndarray, skin_joint: np.ndarray, skin_weight: np.ndarray, *,
                   min_subtree: int = 30, max_subtree_fraction: float = 0.5, vertex_stride: int = 4) -> Dict[str, Any]:
    """The rig for `write_body_rig`, from the refit body.

    `vertices` (N,3) and `joint_positions` (J,3) are the posed body in the
    world frame; `parents` (J,) the joint tree (-1 at the root, parents
    before children); the `skin_*` triplets the model's linear blend
    skinning (`body_refit.rig_binding_data`).
    """
    vertices = np.asarray(vertices, np.float64)
    joint_positions = np.asarray(joint_positions, np.float64)
    parents = np.asarray(parents, np.int64).reshape(-1)
    n_v, n_j = len(vertices), len(parents)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or n_v < K:
        raise ValueError(f"build_body_rig: vertices must be (N,3) with N >= {K}, got {vertices.shape}")
    if joint_positions.shape != (n_j, 3):
        raise ValueError(f"build_body_rig: joint_positions {joint_positions.shape} does not match {n_j} joints")
    if parents[0] != -1 or np.any(parents[1:] >= np.arange(1, n_j)) or np.any(parents[1:] < 0):
        raise ValueError("build_body_rig: parents must be -1 at joint 0 and parent-first (parent index < joint index)")
    sv, sj, sw = (np.asarray(a).reshape(-1) for a in (skin_vertex, skin_joint, skin_weight))
    if not (len(sv) == len(sj) == len(sw)) or sv.max() >= n_v or sj.max() >= n_j or sv.min() < 0 or sj.min() < 0:
        raise ValueError("build_body_rig: skin_vertex / skin_joint / skin_weight do not index the vertices and joints")
    if not 0 < max_subtree_fraction <= 1.0 or vertex_stride < 1 or min_subtree < 1:
        raise ValueError("build_body_rig: max_subtree_fraction in (0, 1], vertex_stride >= 1, min_subtree >= 1")
    weights = np.zeros((n_v, n_j), np.float64)
    np.add.at(weights, (sv.astype(np.int64), sj.astype(np.int64)), sw.astype(np.float64))
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-9)
    dominant = weights.argmax(1)
    subtree = np.bincount(dominant, minlength=n_j).astype(np.int64)
    for j in range(n_j - 1, 0, -1):
        subtree[parents[j]] += subtree[j]
    active = [j for j in range(n_j) if min_subtree <= subtree[j] <= max_subtree_fraction * n_v]
    if not active:
        raise ValueError("build_body_rig: no joint qualifies as active; check min_subtree / max_subtree_fraction")
    idx = np.arange(0, n_v, vertex_stride)
    top = np.argsort(-weights[idx], axis=1)[:, :K]
    tw = np.take_along_axis(weights[idx], top, 1)
    tw /= np.maximum(tw.sum(1, keepdims=True), 1e-9)
    excluded = [j for j in range(n_j) if subtree[j] > max_subtree_fraction * n_v]
    return {
        "verts": vertices[idx].astype(np.float32),
        "joints": top.astype(np.int32),
        "weights": tw.astype(np.float32),
        "parents": parents.astype(np.int32),
        "joint_positions": joint_positions.astype(np.float32),
        "active": np.asarray(active, np.int32),
        "subtree": subtree,
        "excluded_root_chain": np.asarray(excluded, np.int32),
    }


def write_body_rig(path: str | Path, rig: Dict[str, Any], image_names: Sequence[str]) -> None:
    """Write `rig` (from `build_body_rig`) for the training views `image_names` (the frame file names)."""
    names = list(image_names)
    if not names:
        raise ValueError("write_body_rig: no training views")
    for n in names:
        if len(n.encode()) >= NAME_LEN:
            raise ValueError(f"write_body_rig: view name {n!r} is longer than {NAME_LEN - 1} bytes")
    verts, joints, weights = rig["verts"], rig["joints"], rig["weights"]
    parents, jpos, active = rig["parents"], rig["joint_positions"], rig["active"]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(np.array([len(verts), len(parents), len(names), K, NAME_LEN, len(active)], np.int32).tobytes())
        f.write(np.ascontiguousarray(verts, np.float32).tobytes())
        f.write(np.ascontiguousarray(joints, np.int32).tobytes())
        f.write(np.ascontiguousarray(weights, np.float32).tobytes())
        f.write(b"".join(n.encode().ljust(NAME_LEN, b"\0") for n in names))
        f.write(np.ascontiguousarray(parents, np.int32).tobytes())
        f.write(np.ascontiguousarray(jpos, np.float32).tobytes())
        f.write(np.ascontiguousarray(active, np.int32).tobytes())


def read_body_rig(path: str | Path) -> Dict[str, Any]:
    """The file back as arrays (tests, and inspection)."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path} is not a b2ctrain body rig v2")
    hdr = np.frombuffer(data[8:32], np.int32)
    n_v, n_j, n_views, k, name_len, n_active = (int(v) for v in hdr)
    if k != K or name_len != NAME_LEN:
        raise ValueError(f"{path}: unsupported layout")
    off = 32

    def take(dtype, count):
        nonlocal off
        arr = np.frombuffer(data[off:off + count * np.dtype(dtype).itemsize], dtype)
        off += count * np.dtype(dtype).itemsize
        return arr.copy()

    verts = take(np.float32, n_v * 3).reshape(n_v, 3)
    joints = take(np.int32, n_v * K).reshape(n_v, K)
    weights = take(np.float32, n_v * K).reshape(n_v, K)
    names = [data[off + i * NAME_LEN:off + (i + 1) * NAME_LEN].split(b"\0", 1)[0].decode() for i in range(n_views)]
    off += n_views * NAME_LEN
    parents = take(np.int32, n_j)
    jpos = take(np.float32, n_j * 3).reshape(n_j, 3)
    active = take(np.int32, n_active)
    return {"verts": verts, "joints": joints, "weights": weights, "names": names,
            "parents": parents, "joint_positions": jpos, "active": active}
