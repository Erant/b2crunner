"""`build_body_rig`: the per-view deformation rig for the final training, from the refit body.

Numpy only, in the main environment: it takes what `refit_body_to_splat`
publishes (the world mesh, the joints in SAM-3D-Body's raw space with the
similarity into the world frame, and the model's skinning) and produces the
geometry half of the rig; the brush step adds the training view names and
writes the file `b2ctrain --body-rig` reads. See pipeline/body_rig.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..body_rig import build_body_rig
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


@register_step("build_body_rig")
class BuildBodyRigStep(Step):
    """The deformation rig of the refit body.

    inputs:  {"mesh_world": (vertices (N,3), faces) — the refit mesh in the world frame,
              "joints": (J,3) — refit_body_to_splat's joints (raw space),
              "world_from_raw": {"scale", "rotation", "translation"},
              "rig_binding": refit_body_to_splat's rig_binding (joint_parents, skin_*)}
    outputs: {"body_rig": dict for pipeline.body_rig.write_body_rig,
              "body_rig_stats": {...}}
    """

    PARAMS = (
        Param("min_subtree", int, 30,
              "A joint is active (gets a per-view rotation) when its subtree is skinned to at least this many "
              "mesh vertices", minimum=1),
        Param("max_subtree_fraction", float, 0.5,
              "...and to at most this fraction of the body: the root chain (root, pelvis, spine) is excluded, a "
              "rotation there moves the whole body per view and was measured to destroy sharpness everywhere",
              minimum=0.01, maximum=1.0),
        Param("vertex_stride", int, 4, "Keep every n-th mesh vertex for the splat binding", minimum=1),
        Param("debug_dir", str, "", "Write body_rig.json (active joints, subtree sizes) here"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        mesh = inputs.get("mesh_world")
        if not isinstance(mesh, (tuple, list)) or len(mesh) != 2:
            raise ValueError("build_body_rig needs 'mesh_world' as (vertices, faces) — wire refit_body_to_splat's mesh_world")
        joints = np.asarray(inputs.get("joints"), np.float64) if inputs.get("joints") is not None else None
        wfr = inputs.get("world_from_raw")
        rig = inputs.get("rig_binding")
        if joints is None or joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("build_body_rig needs 'joints' (J,3) from refit_body_to_splat")
        if not isinstance(wfr, dict) or any(k not in wfr for k in ("scale", "rotation", "translation")):
            raise ValueError("build_body_rig needs 'world_from_raw' {scale, rotation, translation} from refit_body_to_splat")
        if not isinstance(rig, dict) or any(k not in rig for k in ("joint_parents", "skin_vertex", "skin_joint", "skin_weight")):
            raise ValueError("build_body_rig needs 'rig_binding' from refit_body_to_splat")
        scale = float(wfr["scale"]); rot = np.asarray(wfr["rotation"], np.float64).reshape(3, 3); trans = np.asarray(wfr["translation"], np.float64).reshape(3)
        joints_world = scale * joints @ rot.T + trans
        vertices = np.asarray(mesh[0], np.float64)
        out = build_body_rig(vertices, joints_world, np.asarray(rig["joint_parents"]), rig["skin_vertex"], rig["skin_joint"], rig["skin_weight"],
                             min_subtree=params["min_subtree"], max_subtree_fraction=params["max_subtree_fraction"], vertex_stride=params["vertex_stride"])
        stats = {
            "joints": int(len(out["parents"])), "active": int(len(out["active"])),
            "excluded_root_chain": [int(j) for j in out["excluded_root_chain"]],
            "rig_vertices": int(len(out["verts"])), "mesh_vertices": int(len(vertices)),
            "min_subtree": params["min_subtree"], "max_subtree_fraction": params["max_subtree_fraction"],
        }
        logger.info("build_body_rig: %d of %d joints active (root chain %s excluded), %d rig vertices of %d",
                    stats["active"], stats["joints"], stats["excluded_root_chain"], stats["rig_vertices"], stats["mesh_vertices"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"]); debug.mkdir(parents=True, exist_ok=True)
            (debug / "body_rig.json").write_text(json.dumps({**stats, "active_joints": [int(j) for j in out["active"]],
                                                              "subtree": [int(v) for v in out["subtree"]]}, indent=1))
        rig_out = {k: v for k, v in out.items() if k in ("verts", "joints", "weights", "parents", "joint_positions", "active")}
        return {"body_rig": rig_out, "body_rig_stats": stats}
