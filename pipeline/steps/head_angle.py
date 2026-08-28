"""head_angle_fix — bend a craned-forward head back over the shoulders.

SAM-3D-Body, fitted from a straight-on frontal photo (which the native path
always feeds it), reliably places the head too far forward: the neck-to-head
vector comes out tilted ~30-45 deg off the torso's own up axis, so the
rendered skeleton — and the mesh silhouette the diffusion pass conditions on
— shows the subject with their neck craned forward. cyber_6f already has a
mild case of this. The real fix is in how the body model is fitted /
regularised upstream; this step is the stopgap the pipeline runs in the
meantime, between `reconstruct_body` and `render_initial_views`.

**What it does.** A single rigid nod, applied about the inter-shoulder axis
through the neck joint (MHR70 index 69), graded from 0 at the neck to full at
the crown by a smoothstep on height along the torso-up axis — so the head and
jaw rotate as one piece, the throat bends, and the shoulders and everything
below them do not move at all. Both the mesh vertices and the MHR70
`keypoints_3d` are bent by the identical transform, so the skeleton overlay
still lines up with the silhouette. Everything is done in SAM-3D-Body's raw
output space (the space `pred_vertices` and `pred_keypoints_3d` share before
body2colmap's `sam3d_to_world`), with the pivot, the up axis and the nod axis
all derived from the skeleton itself — no dependence on which way is "up" in
that space.

**auto vs fixed.** `mode="auto"` (the default) measures the current forward
lean — the signed angle, in the sagittal plane, between the neck-to-head
vector and the torso-up axis (neck minus mid-hip) — and rotates back by
however much it takes to leave `target_lean_deg` of lean, capped at
`max_correction_deg`. A head that is already upright enough is left alone.
`mode="fixed"` ignores the measurement and rotates back by exactly
`pitch_deg` (negative leans it further forward).

**Known limitations** (it is a stopgap): the anchor camera / `cam_t` and
`focal_length` are untouched, so the warped reference photo at the anchor
frame still frames the head where the photo has it — a few degrees of nod
does not visibly break the warp, but a large correction will. The 2D face
landmarks from `detect_face` are likewise not re-projected. `scene.joints`
(the 127-joint rig) and `scene.global_rots` are passed through unchanged;
nothing downstream of here consumes them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

# MHR70 joint indices this step reads (body2colmap.skeleton.MHR70_JOINTS).
_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR = 0, 1, 2, 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 9, 10
_NECK = 69
_HEAD_POINTS = (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("degenerate vector")
    return v / n


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate rows of `v` (..,3) about unit `axis` (3,) by per-row `angle`.

    `angle` broadcasts against `v`'s leading shape; a scalar rotates the lot.
    """
    angle = np.asarray(angle, dtype=np.float64)
    cos = np.cos(angle)[..., None]
    sin = np.sin(angle)[..., None]
    k = axis[None, :] if v.ndim > 1 else axis
    k_dot_v = (v * axis).sum(axis=-1, keepdims=True)
    k_cross_v = np.cross(np.broadcast_to(axis, v.shape), v)
    return v * cos + k_cross_v * sin + k * k_dot_v * (1.0 - cos)


@register_step("head_angle_fix")
class HeadAngleFixStep(Step):
    """Nod a craned-forward head back toward the torso axis.

    inputs:  {"vertices": (N, 3) float — SAM-3D-Body `pred_vertices`,
              "keypoints_3d": (70, 3) float — MHR70 `pred_keypoints_3d`}
    outputs: {"vertices": (N, 3) float, "keypoints_3d": (70, 3) float}

    Both arrive in SAM-3D-Body's raw output space and are returned in it,
    bent by one weighted rotation. Wire the outputs back over the same
    `scene.vertices` / `scene.keypoints_3d` the reconstruction wrote.
    """

    PARAMS = (
        Param("mode", str, "auto",
              "'auto' measures the head's forward lean and straightens it to "
              "target_lean_deg; 'fixed' rotates back by exactly pitch_deg",
              choices=("auto", "fixed")),
        Param("target_lean_deg", float, 10.0,
              "auto mode: forward lean to leave, in degrees between the "
              "neck-to-head vector and the torso-up axis. The head is nodded "
              "back only by the excess over this",
              minimum=0.0, maximum=90.0),
        Param("max_correction_deg", float, 35.0,
              "auto mode: never nod the head back by more than this, however "
              "far forward the fit put it",
              minimum=0.0, maximum=90.0),
        Param("pitch_deg", float, 20.0,
              "fixed mode: nod the head back by exactly this many degrees "
              "(negative leans it further forward). Ignored in auto mode"),
        Param("blend_lo_frac", float, 0.0,
              "Start of the neck blend band, as a fraction of the neck-to-head "
              "height: 0 begins the bend at the neck joint (shoulders stay "
              "put), a small negative value starts it just below",
              advanced=True),
        Param("blend_hi_frac", float, 0.9,
              "End of the neck blend band, as a fraction of the neck-to-head "
              "height: above this the head rotates as a rigid piece",
              advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        vertices = np.asarray(inputs["vertices"], dtype=np.float64)
        joints = np.asarray(inputs["keypoints_3d"], dtype=np.float64)
        if joints.shape[0] < 70:
            raise ValueError(
                f"head_angle_fix expects an MHR70 (70-joint) skeleton, got "
                f"{joints.shape[0]} joints"
            )

        neck = joints[_NECK]
        mid_hip = 0.5 * (joints[_L_HIP] + joints[_R_HIP])
        head_center = joints[list(_HEAD_POINTS)].mean(axis=0)

        try:
            spine_up = _unit(neck - mid_hip)
            shoulder_axis = joints[_L_SHOULDER] - joints[_R_SHOULDER]
            # Orthogonalise against spine_up so the nod is a pure sagittal
            # rotation and does not also yaw/roll the head.
            shoulder_axis = _unit(
                shoulder_axis - float(shoulder_axis @ spine_up) * spine_up
            )
        except ValueError:
            logger.warning(
                "head_angle_fix: skeleton too degenerate to derive a nod axis "
                "(coincident hip/shoulder/neck joints); passing geometry through"
            )
            return self._passthrough(inputs)

        head_vec = head_center - neck
        up_comp = float(head_vec @ spine_up)
        if up_comp <= 1e-6:
            logger.warning(
                "head_angle_fix: head center is not above the neck along the "
                "torso axis (up_comp=%.4f); passing geometry through", up_comp
            )
            return self._passthrough(inputs)

        # "forward" = the direction the head actually leans, so the measured
        # lean is >= 0 and a positive correction always nods it back.
        forward = np.cross(shoulder_axis, spine_up)
        forward = _unit(forward - float(forward @ spine_up) * spine_up)
        fwd_comp = float(head_vec @ forward)
        if fwd_comp < 0.0:
            forward = -forward
            fwd_comp = -fwd_comp
        lean_deg = float(np.degrees(np.arctan2(fwd_comp, up_comp)))

        mode = params["mode"]
        if mode == "auto":
            correction_deg = min(
                max(lean_deg - params["target_lean_deg"], 0.0),
                params["max_correction_deg"],
            )
        else:
            correction_deg = params["pitch_deg"]

        if abs(correction_deg) < 0.5:
            logger.info(
                "head_angle_fix: head leans %.1f deg forward, within tolerance "
                "(mode=%s) — leaving geometry unchanged", lean_deg, mode
            )
            return self._passthrough(inputs)

        # Sign: which way about shoulder_axis reduces the forward component.
        probe = _rodrigues(head_vec, shoulder_axis, 1e-3)
        sign = 1.0 if float(probe @ forward) < fwd_comp else -1.0
        theta_full = sign * np.radians(correction_deg)

        span = (params["blend_hi_frac"] - params["blend_lo_frac"]) * up_comp
        if span <= 1e-9:
            raise ValueError(
                "head_angle_fix: blend_hi_frac must be greater than "
                "blend_lo_frac"
            )
        lo = params["blend_lo_frac"] * up_comp

        def _bend(points: np.ndarray) -> np.ndarray:
            rel = points - neck
            height = rel @ spine_up
            t = np.clip((height - lo) / span, 0.0, 1.0)
            weight = t * t * (3.0 - 2.0 * t)  # smoothstep
            rotated = _rodrigues(rel, shoulder_axis, weight * theta_full)
            return neck + rotated

        new_vertices = _bend(vertices)
        new_joints = _bend(joints)

        logger.info(
            "head_angle_fix: head leaned %.1f deg forward; nodded back %.1f deg "
            "(mode=%s) to ~%.1f deg residual lean",
            lean_deg, correction_deg, mode, lean_deg - correction_deg,
        )

        out_vertices = new_vertices.astype(np.asarray(inputs["vertices"]).dtype, copy=False)
        out_joints = new_joints.astype(np.asarray(inputs["keypoints_3d"]).dtype, copy=False)
        return {"vertices": out_vertices, "keypoints_3d": out_joints}

    @staticmethod
    def _passthrough(inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vertices": np.asarray(inputs["vertices"]),
            "keypoints_3d": np.asarray(inputs["keypoints_3d"]),
        }
