"""export_subject: the run's deliverable as one glTF file (b2cgltf SPEC.md section 4).

The subject file holds the splat, the refitted MHR body skinned to its named skeleton (`B2C_mhr`), and the capture
(`B2C_orbit`: the orbit and final cameras, the orbit record, and the reference / anchor / front images), so nothing
rides beside the .ply any more: `scene.ply` stays the bare trained splat, and the Results tab hands out either one.

Everything comes from the scene: the refit's `body_params` (with the model row, joint names and model hash it now
publishes), its `mesh_world` and `rig_binding`, and snapshot_orbit's record. No MHR replay, no header parsing.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

_ORBIT_KEYS = ("helix", "extension", "pass_frames", "anchor_frame_index", "extras", "prompt", "settings")


def _plain(value: Any) -> Any:
    """JSON-ready: numpy scalars and arrays to Python numbers and lists."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cameras(cameras: Sequence[Any], names=None, dataset=None):
    from b2cgltf.b2crunner.subject import Cameras

    from ..orbit_record import camera_arrays

    arrays = camera_arrays(list(cameras))
    return Cameras(rotation=arrays["rotation"], position=arrays["position"], intrinsics=arrays["intrinsics"],
                   image_size=tuple(int(v) for v in arrays["image_size"]),
                   names=None if names is None else [str(n) for n in names], dataset=dataset)


def _commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@register_step("export_subject")
class ExportSubjectStep(Step):
    """The subject file beside the trained splat.

    inputs:  {"splat_path": the trained .ply, "cameras" / "image_names": what it was trained on,
              "body_params" / "mesh_world" / "rig_binding": refit_body_to_splat's,
              "record": snapshot_orbit's (optional: without it B2C_orbit holds only the final cameras)}
    outputs: {"subject_path": the .glb}
    """

    PARAMS = (
        Param("name", str, "scene.glb", "The subject file's name, beside the .ply"),
        Param("dataset_uri", str, "../colmap/",
              "B2C_orbit.final_cameras.dataset: where the training dataset sits relative to the file (a hint)",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2
        from b2cgltf.b2crunner import convert
        from b2cgltf.b2crunner.subject import Body, Capture, write_subject

        from .. import ply_meta

        ply_path = Path(inputs["splat_path"])
        body_params = inputs["body_params"]
        rig = inputs["rig_binding"]
        verts, faces = inputs["mesh_world"]
        missing = [k for k in ("model_params", "hand_idx", "joint_names", "model_sha256") if k not in body_params]
        if missing:
            raise ValueError(f"export_subject: body_params lacks {missing}; refit_body_to_splat publishes them")
        wfr = body_params["world_from_raw"]
        joints_w, rots_w = ply_meta.body_world(wfr, body_params["joints"], body_params["global_rots"])
        body = Body(
            verts=np.asarray(verts), faces=np.asarray(faces), skin_vertex=rig["skin_vertex"],
            skin_joint=rig["skin_joint"], skin_weight=rig["skin_weight"],
            joint_parents=np.asarray(body_params["joint_parents"]), joints=joints_w, global_rots=rots_w,
            model_sha256=body_params["model_sha256"], world_from_raw=wfr,
            pose_params={k: np.asarray(v) for k, v in body_params["pose_params"].items()},
            model_params=body_params["model_params"], hand_idx=body_params["hand_idx"],
            joint_names=list(body_params["joint_names"]))
        if len(body.verts) != len(rig["rest_vertices"]):
            raise ValueError(f"export_subject: the refit mesh has {len(body.verts)} vertices, the MHR skin "
                             f"{len(rig['rest_vertices'])}")

        record = dict(inputs.get("record") or {})
        images = {}
        for name, image in (record.pop("_images", None) or {}).items():
            ok, png = cv2.imencode(".png", np.asarray(image))
            if not ok:
                raise RuntimeError(f"export_subject: could not encode the {name} image")
            images[name] = png.tobytes()
        capture = Capture(
            final=_cameras(inputs["cameras"], names=inputs["image_names"], dataset=params["dataset_uri"]),
            orbit=_cameras(record["orbit_cameras"]) if record.get("orbit_cameras") is not None else None,
            images=images, **{k: _plain(record.get(k)) for k in _ORBIT_KEYS})

        fields, _ = convert.read_trainer_ply(ply_path)
        path = write_subject(ply_path.parent / params["name"], fields, body, capture, tool="b2crunner",
                             commit=_commit(), log=logger.info)
        logger.info(
            "export_subject: %s (%.1f MB): %d splats, %d body vertices, %d joints, %d final + %d orbit cameras, "
            "images: %s", path.name, path.stat().st_size / 1e6, len(fields["x"]), len(body.verts),
            len(body.joints), len(capture.final.position),
            0 if capture.orbit is None else len(capture.orbit.position), ", ".join(sorted(images)) or "none")
        return {"subject_path": str(path)}
