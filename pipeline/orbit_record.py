"""The delivered splat carries its orbit: `b2c.orbit.*` PLY header comments.

The orbit extension (steps/extend_orbit.py) costs ~10 minutes a run, and is
only worth paying on a splat that came out well. This record is what lets
it run after the fact, from `ply/scene.ply` and the images beside it,
instead of inside the run that made the splat. The splat itself is the
guide: it is a fit on pass 2's frames, which is what the `retrained` guide
is, and a render of it at a training camera stands in for that camera's
frame. What cannot be recovered from the splat is the path the frames
were made on, the conditioning the denoise used, and the photograph. The
path and the conditioning go here. The images go beside the .ply
(`embed_orbit_record`: reference.png, anchor.png, front.png), since a PLY
header is text and b2ctrain refuses any element but `vertex`
(ply_meta.py's docstring).

What the record holds, one comment line per key:

  * `helix.*`: the helical path's params, n_frames, n_loops,
    amplitude_deg, lead_in_deg, lead_out_deg. If the run extended its
    orbit, these are the EXTENDED helix (`extended_helix_params`), so that
    `extend_helical_path` can rebuild the path the frames sit on and
    continue it again. `extension.*` says what was added (0 and 0 if
    nothing was) and `pass_frames` gives one denoise pass's length.
  * `anchor_frame_index`: the photograph's frame on that path.
  * `orbit_cameras.*`: every camera of that path verbatim, as the frames
    were denoised on it: before `refine_cameras_final`, at the render
    resolution. `rotation` (N,3,3) is camera-to-world and `position` (N,3)
    the centre in the splat's world, as body2colmap.Camera holds them.
    `intrinsics` (N,4) is fx fy cx cy in pixels at `image_size` (w h).
  * `final_cameras.*`: the same for the cameras the splat was trained on,
    refined and at the deliverable's resolution.
  * `extras`: the dataset's extras, JSON: orbit_target,
    original_focal_length and focal_length_mm (the anchor solver's
    inputs), anchor_position, and whatever else serialises.
  * `prompt`: the subject description the denoise prompts substitute for
    $SUBJECT_DESC$, JSON (the header is ASCII).
  * `settings`: seed, resolution and framing as the run had them, JSON.
  * `images`: each image written beside the .ply, with its sha256, JSON.

The array lines are ply_meta's format (`<key> <shape> <values...>`), with
floats at %.17g so float64 cameras round-trip exactly. A
JSON line has `json` where the shape would be.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import ply_meta

PREFIX = "b2c.orbit."
VERSION = 1

HELIX_KEYS = ("n_frames", "n_loops", "amplitude_deg", "lead_in_deg", "lead_out_deg")
EXTENSION_KEYS = ("before", "after", "overlap_before", "overlap_after")
_INT_KEYS = {"helix.n_frames", "helix.n_loops", "pass_frames", "anchor_frame_index",
             "orbit_cameras.image_size", "final_cameras.image_size",
             *(f"extension.{k}" for k in EXTENSION_KEYS)}
_JSON_KEYS = ("extras", "prompt", "settings", "images")

#: The images the record names, and the file each is written to.
IMAGE_FILES = {"reference": "reference.png", "anchor": "anchor.png", "front": "front.png"}


def camera_arrays(cameras: Sequence[Any]) -> Dict[str, np.ndarray]:
    """A camera list as the record's four arrays. One image size for all."""
    if not cameras:
        raise ValueError("orbit record: no cameras")
    sizes = {(int(c.width), int(c.height)) for c in cameras}
    if len(sizes) != 1:
        raise ValueError(f"orbit record: the cameras are not one size: {sorted(sizes)}")
    return {
        "rotation": np.stack([np.asarray(c.rotation, np.float64).reshape(3, 3) for c in cameras]),
        "position": np.stack([np.asarray(c.position, np.float64).reshape(3) for c in cameras]),
        "intrinsics": np.array([[c.fx, c.fy, c.cx, c.cy] for c in cameras], np.float64),
        "image_size": np.array(sizes.pop(), np.int64),
    }


def cameras_from_arrays(arrays: Dict[str, np.ndarray]) -> List[Any]:
    """The record's four arrays back as body2colmap cameras."""
    from body2colmap.camera import Camera

    width, height = (int(v) for v in arrays["image_size"])
    return [
        Camera(focal_length=(float(k[0]), float(k[1])), image_size=(width, height),
               principal_point=(float(k[2]), float(k[3])),
               position=np.asarray(p, np.float32), rotation=np.asarray(r, np.float32))
        for r, p, k in zip(arrays["rotation"], arrays["position"], arrays["intrinsics"])
    ]


def _fmt(name: str, arr: Any) -> str:
    # %.17g, not ply_meta's %.9g: the cameras are float64 and come back exact.
    return ply_meta._fmt(name, arr, PREFIX, "%.17g")


def _json_line(key: str, value: Any) -> str:
    # ensure_ascii: the header is written as ASCII, and a prompt may not be.
    return f"{PREFIX}{key} json {json.dumps(value, ensure_ascii=True, separators=(',', ':'))}"


def orbit_comments(record: Dict[str, Any]) -> List[str]:
    """The header lines for a record (see the module docstring for the keys).

    `record` has `helix` (HELIX_KEYS), `extension` (EXTENSION_KEYS),
    `pass_frames`, `anchor_frame_index` (optional), `orbit_cameras` and
    `final_cameras` (camera lists), and the JSON values `extras`,
    `prompt`, `settings`, `images`."""
    lines = [f"{PREFIX}version {VERSION}"]
    for key in HELIX_KEYS:
        lines.append(_fmt(f"helix.{key}", _typed(f"helix.{key}", record["helix"][key])))
    for key in EXTENSION_KEYS:
        lines.append(_fmt(f"extension.{key}", np.int64(record["extension"][key])))
    lines.append(_fmt("pass_frames", np.int64(record["pass_frames"])))
    if record.get("anchor_frame_index") is not None:
        lines.append(_fmt("anchor_frame_index", np.int64(record["anchor_frame_index"])))
    for group in ("orbit_cameras", "final_cameras"):
        if record.get(group) is None:
            continue
        for name, arr in camera_arrays(record[group]).items():
            lines.append(_fmt(f"{group}.{name}", arr))
    for key in _JSON_KEYS:
        if record.get(key) is not None:
            lines.append(_json_line(key, record[key]))
    return lines


def _typed(key: str, value: Any) -> np.ndarray:
    return np.int64(value) if key in _INT_KEYS else np.float64(value)


def parse_orbit_comments(comments: Iterable[str]) -> Dict[str, Any]:
    """The record back as a dict: groups (`helix`, `extension`,
    `orbit_cameras`, `final_cameras`) as nested dicts, scalars as Python
    numbers, the JSON keys decoded, `version` an int. The camera groups
    stay arrays; `cameras_from_arrays` turns one into cameras. Empty when
    there is no record."""
    out: Dict[str, Any] = {}
    for text in comments:
        if not text.startswith(PREFIX):
            continue
        key, _, payload = text[len(PREFIX):].partition(" ")
        if key == "version":
            out[key] = int(payload)
            continue
        shape_s, _, values = payload.partition(" ")
        if shape_s == "json":
            out[key] = json.loads(values)
            continue
        shape = tuple(int(d) for d in shape_s.split("x")) if shape_s != "1" else ()
        if key in _INT_KEYS:
            arr = np.array([int(v) for v in values.split()], np.int64).reshape(shape)
        else:
            arr = np.array([float(v) for v in values.split()], np.float64).reshape(shape)
        value: Any = arr.item() if arr.ndim == 0 else arr
        group, _, sub = key.partition(".")
        if sub:
            out.setdefault(group, {})[sub] = value
        else:
            out[key] = value
    return out


def read_orbit_record(ply_path: str | Path) -> Dict[str, Any]:
    """`parse_orbit_comments` of a .ply's header, with each named image's
    path resolved beside the .ply and its checksum verified. Raises if the
    file has no record, or an image is missing or does not match."""
    ply_path = Path(ply_path)
    record = parse_orbit_comments(ply_meta.read_comments(ply_path))
    if not record:
        raise ValueError(f"{ply_path} carries no {PREFIX}* record")
    for name, entry in (record.get("images") or {}).items():
        path = ply_path.parent / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"{ply_path.name}'s orbit record names {entry['file']}, "
                                    f"which is not beside it")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match the checksum {ply_path.name} recorded")
        entry["path"] = str(path)
    return record


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embed(ply_path: str | Path, record: Dict[str, Any]) -> int:
    """Write the record into the .ply's header, replacing any earlier
    `b2c.orbit.*` record and leaving every other line (the `b2c.mhr.*`
    body among them) as it was. Returns the number of lines written."""
    lines = orbit_comments(record)
    ply_meta.embed_comments(ply_path, lines, replace_prefix=PREFIX)
    return len(lines)


def json_safe_extras(extras: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The extras that serialise, as Dataset.to_disk keeps them."""
    from .dataset import _json_safe

    out: Dict[str, Any] = {}
    for key, value in (extras or {}).items():
        try:
            safe = _json_safe(value)
            json.dumps(safe)
        except (TypeError, ValueError):
            continue
        out[key] = safe
    return out
