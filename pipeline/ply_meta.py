"""The delivered splat carries its body: MHR parameters as PLY header comments.

A PLY header may hold any number of `comment` lines and every reader skips
them — b2ctrain's own loader, brush, plyfile, SuperSplat — so they are the one
place a per-file record survives that no consumer has to know about. A
separate PLY element would not: b2ctrain refuses non-vertex elements, and
most viewers parse every element they find.

The record is the refitted body (`refit_body_to_splat`): the MHR pose, shape
and scale parameters that replay to the fitted mesh through the model file,
the similarity that takes SAM-3D-Body's raw metres into the splat's world
frame, and the posed skeleton — joint positions and global rotations —
already in that world frame, so a consumer can bind splats to bones without
running the model at all. The skinning weights are the model's, not the
fit's, and are read from `mhr_model.pt` (`body_refit.rig_binding_data`).

Line format, one per key::

    comment b2c.mhr.<key> <shape> <values...>

`<shape>` is the array's shape as `d0xd1x...` (`1` for a scalar), values are
`%.9g` floats (float32 round-trips exactly) or `%d` integers; string keys (`model`, `frame`) carry text.
docs/ply-header-records.md is the format's contract for readers outside b2crunner.

Version 2 (2026-09-29) writes `global_rots` in the same world frame as `joints`; version 1 headers wrote
`rotation @ rots` without the raw frame's flip (docs/design-notes.md#mhr-body-record-frames-and-conventions).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PREFIX = "b2c.mhr."
VERSION = 2

# SAM-3D-Body's raw (OpenCV) frame from MHR's own: positions come out of the model with y and z negated.
FLIP = np.diag([1.0, -1.0, -1.0])

#: pose_params entries a replay must pass to the MHR forward, in this order.
POSE_KEYS = ("global_rot", "body_pose_params", "hand_pose_params", "scale_params",
             "shape_params", "expr_params", "global_trans", "scale_offsets")

_FRAME_NOTE = ("world = world_from_raw.scale * raw @ world_from_raw.rotation.T + world_from_raw.translation; "
               "pose_params are SAM-3D-Body raw (global_trans in metres); joints and global_rots are world")


def read_header(path: str | Path) -> List[str]:
    """The header lines of a PLY file, `ply` through `end_header`."""
    lines: List[str] = []
    with open(path, "rb") as f:
        first = f.readline()
        if not first.startswith(b"ply"):
            raise ValueError(f"{path} is not a PLY file")
        lines.append(first.decode("ascii", "replace").rstrip("\r\n"))
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"{path}: header has no end_header")
            line = raw.decode("ascii", "replace").rstrip("\r\n")
            lines.append(line)
            if line == "end_header":
                return lines


def read_comments(path: str | Path) -> List[str]:
    """Every `comment` line's text, in file order."""
    return [line[len("comment "):] for line in read_header(path) if line.startswith("comment ")]


def embed_comments(path: str | Path, comments: Sequence[str], *, replace_prefix: Optional[str] = PREFIX) -> int:
    """Rewrite the PLY at `path` with `comments` added to its header.

    Existing comments that start with `replace_prefix` are dropped first, so
    embedding twice leaves one record; every other header line and the whole
    body are copied byte for byte. The comments go after the header's existing
    comments, before the first element. Returns the number of comment lines
    in the new header.
    """
    path = Path(path)
    header = read_header(path)
    header_bytes = sum(len(line.encode("ascii", "replace")) + 1 for line in header)
    for text in comments:
        if "\n" in text or "\r" in text:
            raise ValueError("a PLY comment is one line")
    kept = [line for line in header
            if not (replace_prefix and line.startswith("comment " + replace_prefix))]
    fmt = next((i for i, line in enumerate(kept) if line.startswith("format ")), None)
    if fmt is None:
        raise ValueError(f"{path}: header has no format line")
    # After the exporter's own comments (they stay first), before the first element.
    at = fmt + 1
    for i, line in enumerate(kept):
        if line.startswith("element "):
            break
        if line.startswith("comment ") and i > fmt:
            at = i + 1
    new_header = kept[:at] + [f"comment {text}" for text in comments] + kept[at:]
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            out.write(("\n".join(new_header) + "\n").encode("ascii"))
            src.seek(header_bytes)
            shutil.copyfileobj(src, out, 16 * 1024 * 1024)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return sum(1 for line in new_header if line.startswith("comment "))


def _fmt(name: str, arr: Any, prefix: str = PREFIX, float_fmt: str = "%.9g") -> str:
    a = np.asarray(arr)
    shape = "x".join(str(d) for d in a.shape) if a.ndim else "1"
    flat = a.reshape(-1)
    if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
        body = " ".join("%d" % int(v) for v in flat)
    else:
        if not np.all(np.isfinite(flat.astype(np.float64))):
            raise ValueError(f"{name} has non-finite values")
        body = " ".join(float_fmt % float(v) for v in flat)
    return f"{prefix}{name} {shape} {body}"


def body_comments(pose_params: Dict[str, Any], world_from_raw: Dict[str, Any], *,
                  joints: Optional[Any] = None, global_rots: Optional[Any] = None,
                  joint_parents: Optional[Any] = None, model: str = "") -> List[str]:
    """The comment lines for a refitted body (see the module docstring).

    `joints` (J,3) and `global_rots` (J,3,3) are as `refit_body_to_splat`
    publishes them, SAM-3D-Body's convention: the joints are in the raw
    (OpenCV) frame, the rotations in MHR's own frame, because SAM-3D-Body
    (and the refit after it) flips the positions but not the rotations. Both
    are written in the WORLD frame: `rotation @ FLIP @ global_rots`, so a
    joint's rotation and position agree. `world_from_raw` is {"scale",
    "rotation" (3,3), "translation" (3,)}.
    """
    missing = [k for k in POSE_KEYS if pose_params.get(k) is None]
    if missing:
        raise ValueError(f"pose_params is missing {missing}; a replay needs every one of {POSE_KEYS}")
    scale = float(world_from_raw["scale"])
    rot = np.asarray(world_from_raw["rotation"], np.float64).reshape(3, 3)
    trans = np.asarray(world_from_raw["translation"], np.float64).reshape(3)
    lines = [f"{PREFIX}version {VERSION}", f"{PREFIX}frame {_FRAME_NOTE}"]
    if model:
        lines.append(f"{PREFIX}model {model}")
    lines.append(_fmt("world_from_raw.scale", np.float64(scale)))
    lines.append(_fmt("world_from_raw.rotation", rot))
    lines.append(_fmt("world_from_raw.translation", trans))
    for key in POSE_KEYS:
        lines.append(_fmt("pose_params." + key, np.asarray(pose_params[key], np.float32)))
    if joint_parents is not None:
        lines.append(_fmt("joint_parents", np.asarray(joint_parents, np.int64)))
    joints_w, rots_w = body_world(world_from_raw, joints, global_rots)
    if joints_w is not None:
        lines.append(_fmt("joints", joints_w))
    if rots_w is not None:
        lines.append(_fmt("global_rots", rots_w))
    return lines


def body_world(world_from_raw: Dict[str, Any], joints: Optional[Any] = None,
               global_rots: Optional[Any] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """`refit_body_to_splat`'s joints (raw frame) and rotations (MHR's frame) in the splat's world frame, where a
    joint's rotation and position agree (record version 2; b2cgltf's joint nodes): `scale * j @ R.T + t` and
    `R @ FLIP @ rots`. Either may be None."""
    scale = float(world_from_raw["scale"])
    rot = np.asarray(world_from_raw["rotation"], np.float64).reshape(3, 3)
    trans = np.asarray(world_from_raw["translation"], np.float64).reshape(3)
    joints_w = None if joints is None else scale * np.asarray(joints, np.float64).reshape(-1, 3) @ rot.T + trans
    rots_w = None if global_rots is None else np.einsum(
        "ij,njk->nik", rot @ FLIP, np.asarray(global_rots, np.float64).reshape(-1, 3, 3))
    return joints_w, rots_w


def parse_body_comments(comments: Iterable[str]) -> Dict[str, Any]:
    """The record back as a dict: arrays under their keys (`pose_params` as a
    nested dict, `world_from_raw` likewise), strings for `model` and `frame`.
    Empty when there is no record."""
    out: Dict[str, Any] = {}
    for text in comments:
        if not text.startswith(PREFIX):
            continue
        rest = text[len(PREFIX):]
        key, _, payload = rest.partition(" ")
        if key in ("version", "model", "frame"):
            out[key] = payload
            continue
        shape_s, _, values = payload.partition(" ")
        shape = tuple(int(d) for d in shape_s.split("x")) if shape_s != "1" else ()
        if key == "joint_parents":
            arr = np.array([int(v) for v in values.split()], np.int64).reshape(shape)
        else:
            arr = np.array([float(v) for v in values.split()], np.float64).reshape(shape)
            if key.startswith("pose_params."):
                arr = arr.astype(np.float32)
        group, _, sub = key.partition(".")
        if sub:
            out.setdefault(group, {})[sub] = arr
        else:
            out[key] = arr
    return out
