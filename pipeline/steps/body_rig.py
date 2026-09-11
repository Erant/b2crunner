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
from typing import Any, Dict, List

import numpy as np

from ..body_rig import assign_view_deltas, build_body_rig, face_only_weights
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
        rig_out = {k: v for k, v in out.items() if k in ("verts", "vertex_index", "joints", "weights", "parents", "joint_positions", "active")}
        return {"body_rig": rig_out, "body_rig_stats": stats}


@register_step("build_face_rig")
class BuildFaceRigStep(Step):
    """The per-view head fit as the rig's per-view vertex displacements (rig v3).

    inputs:  {"body_rig": build_body_rig's rig (the refit body's),
              "head_fit_views": fit_head_per_view's output,
              "image_names": the training frames, in order}
    outputs: {"body_rig": the same rig with "view_deltas" — {frame name: (n_rig_verts, 3)} in the
                          canonical frame, for every frame that gets one,
              "face_rig_stats": {...}}

    Each fitted view's displacement is fitted head - canonical head at the
    rig's vertices. Frames the fit does not cover take the interpolation
    between their fitted neighbours (gaps of `gap` frames or less), hold the
    nearest fitted frame's displacement within `hold` frames of it, and are
    canonical beyond that. The displacement is restricted to the FACE CORE
    (`face_motion_cm`: the vertices the expression basis moves by at least
    this much, fading to nothing `face_fade_cm` away): measured on four
    subjects, whole-head deltas blur the hair 2-5% on three of them (the
    frames' hair does not follow the face fit, and the back views pin it
    canonical, so it was supervised in two places) while face-only deltas
    cost nothing anywhere, swim least and keep the eye gain. 0 = the whole
    head, the option when a subject's eyes need it.
    """

    PARAMS = (
        Param("face_motion_cm", float, 0.5,
              "The face core: vertices the expression basis can move by at least this many centimetres. 0 = no "
              "restriction (whole-head deltas). Anything above 0.01 that is not around 0.5 is the whole head too — "
              "the basis touches every head vertex by a hair", minimum=0.0),
        Param("face_fade_cm", float, 3.0, "Fade the deltas to zero this far outside the face core", minimum=0.1),
        Param("gap", int, 4, "Interpolate across runs of at most this many unfitted frames between fitted ones", minimum=0),
        Param("hold", int, 3, "An unfitted frame within this many frames of a fitted one holds its deltas", minimum=0),
        Param("debug_dir", str, "", "Write face_rig.json (how each frame got its deltas) here"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        rig = inputs.get("body_rig")
        if not isinstance(rig, dict) or "verts" not in rig:
            raise ValueError("build_face_rig needs 'body_rig' from build_body_rig")
        fit = inputs.get("head_fit_views")
        if not isinstance(fit, dict) or "verts_world" not in fit:
            raise ValueError("build_face_rig needs 'head_fit_views' from fit_head_per_view")
        names: List[str] = list(inputs["image_names"])
        verts0 = np.asarray(fit["verts0_world"], np.float64)
        idx = rig.get("vertex_index")
        if idx is None:
            raise ValueError("build_face_rig: the rig carries no vertex_index (built by an older build_body_rig?)")
        idx = np.asarray(idx, np.int64)
        rig_verts = np.asarray(rig["verts"], np.float64)
        if idx.max() >= len(verts0) or np.abs(verts0[idx] - rig_verts).max() > 1e-3:
            raise ValueError("build_face_rig: the rig's vertices are not the head fit's canonical body at vertex_index; "
                             "the rig and the fit are from different bodies")
        index = {n: i for i, n in enumerate(names)}
        fitted = {}
        for k, n in enumerate(fit["names"]):
            if n not in index:
                raise ValueError(f"build_face_rig: fitted view {n!r} is not a training frame")
            fitted[index[n]] = (np.asarray(fit["verts_world"][k], np.float64)[idx] - rig_verts).astype(np.float32)
        if not fitted:
            raise ValueError("build_face_rig: the head fit covers no view")
        deltas, how = assign_view_deltas(fitted, len(names), len(idx), gap=int(params["gap"]), hold=int(params["hold"]))
        weights = None
        if params["face_motion_cm"] > 0:
            motion = np.asarray(fit["expression_motion"], np.float64)
            face = verts0[motion > params["face_motion_cm"]]
            if len(face) == 0:
                raise ValueError(f"build_face_rig: no vertex moves more than {params['face_motion_cm']} cm under the expression basis")
            weights = face_only_weights(rig_verts, face, params["face_fade_cm"] / 100.0)
            deltas *= weights[None, :, None]
        view_deltas = {n: deltas[i] for i, n in enumerate(names) if how[i] != "zero"}
        mag = np.linalg.norm(deltas, axis=2)
        counts = {k: how.count(k) for k in ("fit", "interp", "hold", "zero")}
        stats = {
            "frames": len(names), **{f"frames_{k}": v for k, v in counts.items()},
            "rig_vertices": int(len(idx)),
            "rig_vertices_moved": int((mag.max(0) > 1e-4).sum()),
            "face_only": {"motion_cm": params["face_motion_cm"], "fade_cm": params["face_fade_cm"],
                          "full_weight": int((weights >= 1).sum()) if weights is not None else int(len(idx)),
                          "faded": int(((weights > 0) & (weights < 1)).sum()) if weights is not None else 0},
            "max_delta_mm": float(mag.max() * 1000), "gap": params["gap"], "hold": params["hold"],
        }
        logger.info("build_face_rig: %d frames fitted, %d interpolated, %d held, %d canonical; %d of %d rig vertices move "
                    "(face core %d at full weight, %d faded), max %.1f mm", counts["fit"], counts["interp"], counts["hold"],
                    counts["zero"], stats["rig_vertices_moved"], stats["rig_vertices"], stats["face_only"]["full_weight"],
                    stats["face_only"]["faded"], stats["max_delta_mm"])
        if params["debug_dir"]:
            debug = Path(params["debug_dir"]); debug.mkdir(parents=True, exist_ok=True)
            (debug / "face_rig.json").write_text(json.dumps({**stats, "frames_how": dict(zip(names, how))}, indent=1))
        return {"body_rig": {**rig, "view_deltas": view_deltas}, "face_rig_stats": stats}
