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

Version 3 (`B2CRIG3\\0`) is the same file followed by::

    float32[n_views*n_verts*3] deltas

a per-view displacement of every rig vertex in the canonical frame, which
the trainer adds to a bound point BEFORE the skinning blend. It is what the
per-view head fit produces (steps/face_views.py, `build_face_rig`): the
MHR expression and neck/head pose that explain each frame's face, so the
face splats render where that frame's face is and the canonical face
converges to one expression instead of the average of 81. A rig dict with a
`view_deltas` entry — {view name: (n_verts, 3)} — is written as v3; views
without an entry get zeros (`assign_view_deltas` fills the gaps between
fitted views first).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

MAGIC = b"B2CRIG2\0"
MAGIC_V3 = b"B2CRIG3\0"
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
        "vertex_index": idx.astype(np.int32),
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
    view_deltas = rig.get("view_deltas")
    deltas = None
    if view_deltas:
        deltas = np.zeros((len(names), len(verts), 3), np.float32)
        for i, n in enumerate(names):
            d = view_deltas.get(n)
            if d is None:
                continue
            d = np.asarray(d, np.float32)
            if d.shape != (len(verts), 3):
                raise ValueError(f"write_body_rig: view_deltas[{n!r}] is {d.shape}, not ({len(verts)}, 3)")
            deltas[i] = d
    with open(path, "wb") as f:
        f.write(MAGIC_V3 if deltas is not None else MAGIC)
        f.write(np.array([len(verts), len(parents), len(names), K, NAME_LEN, len(active)], np.int32).tobytes())
        f.write(np.ascontiguousarray(verts, np.float32).tobytes())
        f.write(np.ascontiguousarray(joints, np.int32).tobytes())
        f.write(np.ascontiguousarray(weights, np.float32).tobytes())
        f.write(b"".join(n.encode().ljust(NAME_LEN, b"\0") for n in names))
        f.write(np.ascontiguousarray(parents, np.int32).tobytes())
        f.write(np.ascontiguousarray(jpos, np.float32).tobytes())
        f.write(np.ascontiguousarray(active, np.int32).tobytes())
        if deltas is not None:
            f.write(deltas.tobytes())


def read_body_rig(path: str | Path) -> Dict[str, Any]:
    """The file back as arrays (tests, and inspection)."""
    data = Path(path).read_bytes()
    if data[:8] not in (MAGIC, MAGIC_V3):
        raise ValueError(f"{path} is not a b2ctrain body rig v2/v3")
    v3 = data[:8] == MAGIC_V3
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
    out = {"verts": verts, "joints": joints, "weights": weights, "names": names,
           "parents": parents, "joint_positions": jpos, "active": active}
    if v3:
        out["deltas"] = take(np.float32, n_views * n_v * 3).reshape(n_views, n_v, 3)
        if off != len(data):
            raise ValueError(f"{path}: {len(data) - off} bytes after the v3 delta block")
    return out


def assign_view_deltas(fitted: Dict[int, np.ndarray], n_views: int, n_verts: int, *,
                       gap: int = 4, hold: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Every training view's delta from the fitted views' (keyed by view index).

    The head fit covers the views that face the camera (about 35 of 81 on
    a helical orbit); the rest need something that does not jump. A view
    between two fitted ones at most `gap` unfitted views apart takes the
    linear interpolation; a view within `hold` of a fitted neighbour on one
    side takes that neighbour's delta unchanged (measured better than
    fading it to zero: swim 3.4 -> 2.4 px on the second subject, b2ctrain
    docs/STATUS.md, "The face per view"); anything further is canonical.
    Returns the (n_views, n_verts, 3) deltas and, per view, how it was set:
    "fit", "interp", "hold" or "zero".
    """
    deltas = np.zeros((n_views, n_verts, 3), np.float32)
    how: List[str] = []
    keys = sorted(fitted)
    for v in range(n_views):
        if v in fitted:
            deltas[v] = fitted[v]
            how.append("fit")
            continue
        lo = max((k for k in keys if k < v), default=None)
        hi = min((k for k in keys if k > v), default=None)
        if lo is not None and hi is not None and hi - lo <= gap + 1:
            w = (v - lo) / (hi - lo)
            deltas[v] = (1 - w) * fitted[lo] + w * fitted[hi]
            how.append("interp")
        elif lo is not None and v - lo <= hold and (hi is None or v - lo <= hi - v):
            deltas[v] = fitted[lo]
            how.append("hold")
        elif hi is not None and hi - v <= hold:
            deltas[v] = fitted[hi]
            how.append("hold")
        else:
            how.append("zero")
    return deltas, how


def face_only_weights(rig_verts: np.ndarray, face_verts: np.ndarray, fade: float) -> np.ndarray:
    """Per rig vertex, 1 on the face core, fading to 0 `fade` metres away from it.

    The deltas of the whole head were measured to blur the hair on three of
    four subjects (the frames' hair does not follow the face fit, and the
    back views pin it canonical, so it was supervised in two places);
    restricted to the face core they cost nothing anywhere.
    """
    from scipy.spatial import cKDTree

    rig_verts = np.asarray(rig_verts, np.float64)
    face_verts = np.asarray(face_verts, np.float64)
    if len(face_verts) == 0:
        raise ValueError("face_only_weights: no face vertices")
    if fade <= 0:
        raise ValueError("face_only_weights: fade must be positive")
    d, _ = cKDTree(face_verts).query(rig_verts)
    return np.clip(1.0 - d / fade, 0.0, 1.0).astype(np.float32)
