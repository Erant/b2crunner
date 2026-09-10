"""`build_body_rig`: the per-view deformation rig, from the refit body or from the initial one.

Numpy only, in the main environment: it takes a body mesh in the world frame,
that body's joints, and the model's skinning, and produces the geometry half
of the rig; the brush step adds the training view names and writes the file
`b2ctrain --body-rig` reads. See pipeline/body_rig.py.

Two callers, one step:

- the FINAL training rigs `refit_body_to_splat`'s body, which publishes the
  joints in SAM-3D-Body's raw space together with the `world_from_raw`
  similarity that places them;
- the INTERMEDIATE training rigs the INITIAL body — SAM-3D-Body's fit after
  `fit_head_to_face` — because at that point no splat exists to refit
  against. There is no `world_from_raw` to wire, so the step recovers it the
  way `refit_body_to_splat` does: `mesh_raw` (the same mesh in raw space,
  scene.vertices) against `mesh_world` gives the similarity, and the joints
  ride through it. Measured (b2ctrain docs/STATUS.md, "The rig at the
  INTERMEDIATE stage"): the initial body rigs the intermediate splat as well
  as the refit body does — hands s1 107.1 against 105.6 — so the second pass
  a refit would need buys nothing there.
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
from .body_refit import umeyama

logger = logging.getLogger(__name__)


@register_step("build_body_rig")
class BuildBodyRigStep(Step):
    """The deformation rig of a body: the refit one, or the initial one.

    inputs:  {"mesh_world": (vertices (N,3), faces) — the body mesh in the world frame,
              "joints": (J,3) — that body's joints, in SAM-3D-Body's raw space,
              "world_from_raw": {"scale", "rotation", "translation"} — from
                     refit_body_to_splat; OMIT it for the initial body and wire
                     "mesh_raw" instead,
              "mesh_raw": (N,3) — the same mesh in raw space (scene.vertices),
                     from which the similarity into the world frame is recovered,
              "rig_binding": the model's skeleton and skinning (joint_parents,
                     skin_*), from refit_body_to_splat or fit_head_to_face}
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
            raise ValueError("build_body_rig needs 'mesh_world' as (vertices, faces) — wire refit_body_to_splat's mesh_world, or render's for the initial body")
        joints = np.asarray(inputs.get("joints"), np.float64) if inputs.get("joints") is not None else None
        wfr = inputs.get("world_from_raw")
        mesh_raw = inputs.get("mesh_raw")
        rig = inputs.get("rig_binding")
        if joints is None or joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("build_body_rig needs 'joints' (J,3) — refit_body_to_splat's, or "
                             "fit_head_to_face's for the initial body")
        if not isinstance(rig, dict) or any(k not in rig for k in ("joint_parents", "skin_vertex", "skin_joint", "skin_weight")):
            raise ValueError("build_body_rig needs 'rig_binding' from refit_body_to_splat or fit_head_to_face")
        vertices = np.asarray(mesh[0], np.float64)
        if isinstance(wfr, dict) and all(k in wfr for k in ("scale", "rotation", "translation")):
            source = "refit"
            scale = float(wfr["scale"])
            rot = np.asarray(wfr["rotation"], np.float64).reshape(3, 3)
            trans = np.asarray(wfr["translation"], np.float64).reshape(3)
        elif mesh_raw is not None:
            source = "initial"
            # The initial body: recover world = s R raw + t from the mesh
            # itself, exactly as refit_body_to_splat does, and refuse if the
            # two meshes are not the same one rigidly placed — a wrong frame
            # would put every rotation pivot centimetres off the body.
            raw = np.asarray(mesh_raw, np.float64)
            if raw.ndim != 2 or raw.shape != vertices.shape:
                raise ValueError(f"build_body_rig: mesh_raw {raw.shape} and mesh_world "
                                 f"{vertices.shape} must be the same mesh in two frames")
            scale, rot, trans = umeyama(raw, vertices)
            residual = float(np.abs(scale * raw @ rot.T + trans - vertices).max())
            if residual > 2e-3:
                raise ValueError(f"build_body_rig: mesh_world is not a rigid placement of mesh_raw "
                                 f"({residual * 1000:.1f} mm off after the best similarity)")
        else:
            raise ValueError("build_body_rig needs either 'world_from_raw' (refit_body_to_splat) "
                             "or 'mesh_raw' (the initial body's raw vertices, scene.vertices)")
        joints_world = scale * joints @ rot.T + trans
        out = build_body_rig(vertices, joints_world, np.asarray(rig["joint_parents"]), rig["skin_vertex"], rig["skin_joint"], rig["skin_weight"],
                             min_subtree=params["min_subtree"], max_subtree_fraction=params["max_subtree_fraction"], vertex_stride=params["vertex_stride"])
        stats = {
            "joints": int(len(out["parents"])), "active": int(len(out["active"])),
            "excluded_root_chain": [int(j) for j in out["excluded_root_chain"]],
            "rig_vertices": int(len(out["verts"])), "mesh_vertices": int(len(vertices)),
            "min_subtree": params["min_subtree"], "max_subtree_fraction": params["max_subtree_fraction"],
            "body": source,
            "world_from_raw_scale": float(scale),
        }
        logger.info("build_body_rig: the %s body — %d of %d joints active (root chain %s excluded), "
                    "%d rig vertices of %d", stats["body"], stats["active"], stats["joints"],
                    stats["excluded_root_chain"], stats["rig_vertices"], stats["mesh_vertices"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"]); debug.mkdir(parents=True, exist_ok=True)
            (debug / "body_rig.json").write_text(json.dumps({**stats, "active_joints": [int(j) for j in out["active"]],
                                                              "subtree": [int(v) for v in out["subtree"]]}, indent=1))
        rig_out = {k: v for k, v in out.items() if k in ("verts", "joints", "weights", "parents", "joint_positions", "active")}
        return {"body_rig": rig_out, "body_rig_stats": stats}
