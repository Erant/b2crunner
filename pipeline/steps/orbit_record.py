"""snapshot_orbit / embed_orbit_record: the deliverable carries its orbit.

Two steps around pipeline/orbit_record.py, so that the orbit extension
can run after the fact on a splat that turned out worth extending. The
snapshot is taken when the orbit is final (after `extend_splice`, if one
ran) and before stage 5 and 6 change what the dataset holds:
`refine_cameras_final` moves the cameras, and the upscale rescales their
intrinsics. The embed runs after `train_final_splat`, since each brush
export rewrites the .ply from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


@register_step("snapshot_orbit")
class SnapshotOrbitStep(Step):
    """What the orbit extension needs from this run, taken while it is still
    pass 2's.

    inputs:  {"dataset": the denoised orbit (cameras as the frames were made
              on, extras, prompt, reference and anchor images),
              "before"?, "after"?, "overlap_before"?, "overlap_after"?:
              extend_helical_path's counts, present only if the orbit was
              extended, "front_image"?: the photograph's front panel}
    outputs: {"record": the orbit record without its final cameras (see
              pipeline/orbit_record.py), with the images under `_images`}

    The helix params here MUST be render_subject's, as extend_path's are:
    the record says which helix the cameras sit on, and a later extension
    rebuilds it from these params and refuses a path it cannot reproduce.
    """

    PARAMS = (
        Param("n_frames", int, 81, "render_subject's frame count", minimum=1),
        Param("n_loops", int, 2, "render_subject's turns", minimum=1),
        Param("amplitude_deg", float, 30.0, "render_subject's elevation swing"),
        Param("lead_in_deg", float, 30.0, "render_subject's lead-in"),
        Param("lead_out_deg", float, 90.0, "render_subject's lead-out"),
        Param("pass_frames", int, 81,
              "One denoise pass's length, the length an extension pass has to match",
              minimum=1),
        Param("run_seed", int, 0, "The run's seed, recorded for the extension passes"),
        Param("resolution", list, [720, 1280], "The run's render resolution, recorded"),
        Param("framing", str, "", "The run's framing preset, recorded"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from ..orbit_record import HELIX_KEYS, json_safe_extras
        from .extend_orbit import extended_helix_params

        dataset = inputs["dataset"]
        cameras = list(dataset.cameras)
        source = {key: params[key] for key in HELIX_KEYS}
        counts = {key: inputs.get(key) for key in
                  ("before", "after", "overlap_before", "overlap_after")}
        extended = counts["before"] is not None
        if extended != (counts["after"] is not None):
            raise ValueError("snapshot_orbit: `before` and `after` come as a pair, from "
                             "extend_helical_path")
        extension = {key: int(value or 0) for key, value in counts.items()}
        helix = (extended_helix_params(source, extension["before"], extension["after"])
                 if extended else dict(source))
        if len(cameras) != int(helix["n_frames"]):
            raise ValueError(
                f"snapshot_orbit: the dataset has {len(cameras)} cameras against the "
                f"{helix['n_frames']} of the helix it records"
                f"{' (extended)' if extended else ''}. It has to run on the orbit the "
                f"frames were made on, after extend_splice and before anything moves it"
            )

        extras = dict(dataset.extras or {})
        anchor = extras.get("anchor_frame_index")
        images = {"reference": dataset.reference_image, "anchor": dataset.anchor_image,
                  "front": inputs.get("front_image")}
        record = {
            "helix": helix,
            "extension": extension,
            "pass_frames": int(params["pass_frames"]),
            "anchor_frame_index": None if anchor is None else int(anchor),
            "orbit_cameras": cameras,
            "extras": json_safe_extras(extras),
            "prompt": dataset.prompt or "",
            "settings": {"seed": int(params["run_seed"]),
                         "resolution": [int(v) for v in params["resolution"]],
                         "framing": str(params["framing"])},
            "_images": {name: np.asarray(img) for name, img in images.items() if img is not None},
        }
        logger.info(
            "snapshot_orbit: %d cameras on the %shelix (lead-in %.1f / lead-out %.1f deg), "
            "anchor at frame %s; images kept for the deliverable: %s",
            len(cameras), "extended " if extended else "", helix["lead_in_deg"],
            helix["lead_out_deg"], anchor, ", ".join(sorted(record["_images"])) or "none",
        )
        return {"record": record}


@register_step("embed_orbit_record")
class EmbedOrbitRecordStep(Step):
    """The orbit record into the deliverable .ply's header, and its images
    beside it.

    inputs:  {"splat_path": the trained .ply, "record": snapshot_orbit's,
              "cameras": the cameras the splat was trained on}
    outputs: {"files": the images written, by name}

    Runs after the last training has exported, since each export rewrites
    the file. The images go in the .ply's own directory (`ply/` in the
    result .zip): reference.png is what the denoise conditioned on,
    anchor.png the photograph as it was injected at the anchor frame,
    front.png the photograph's front panel the face fits read. The header
    names each with its sha256.
    """

    PARAMS = ()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        from ..orbit_record import IMAGE_FILES, embed, sha256_file

        ply_path = Path(inputs["splat_path"])
        record = dict(inputs["record"])
        images = record.pop("_images", {}) or {}
        record["final_cameras"] = list(inputs["cameras"])

        files: Dict[str, str] = {}
        entries: Dict[str, Dict[str, str]] = {}
        for name, image in images.items():
            path = ply_path.parent / IMAGE_FILES[name]
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"embed_orbit_record: could not write {path}")
            files[name] = str(path)
            entries[name] = {"file": path.name, "sha256": sha256_file(path)}
        record["images"] = entries

        count = embed(ply_path, record)
        logger.info(
            "embed_orbit_record: %d b2c.orbit.* comments in %s's header (%d orbit cameras, "
            "%d final); beside it: %s",
            count, ply_path.name, len(record["orbit_cameras"]), len(record["final_cameras"]),
            ", ".join(Path(p).name for p in files.values()) or "no images",
        )
        return {"files": files}
