"""head_angle_fix — bend a craned-forward head back over the shoulders.

**Superseded in fast_helical_native by `fit_head_to_face` (head_fit.py),
2026-08-30.** This step turns the head to a fixed lean and has no idea where
the photograph's face looks; measured on cyber2_6f its auto nod turned the
head 30 deg away from it (MediaPipe residual 10.5 -> 20.7 px, where the fit
reaches 2.4). It stays registered and tested for workflows without a photo
to fit to, and is declared incompatible with the fit (INCOMPATIBLE_STEPS):
it edits vertices and leaves the pose parameters describing the old head.

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

**Mutually exclusive with `refine_pose_to_splat`**, and
`WorkflowSpec.validate()` refuses a workflow enabling both (see
INCOMPATIBLE_STEPS in pipeline/workflow.py). This step edits the vertices
and keypoints directly and leaves the MHR pose parameters describing the
*old* geometry; that step replays those parameters through the body model
and rebuilds the mesh from them, so it would discard this nod outright —
and its round-trip gate refuses to run once the two disagree. Nor does it
subsume this one — but not because it ignores the head. It moves the head
centre 33 mm back along the sagittal axis, so it is already acting on the
crane's depth component; it simply answers to the shell, which inherited the
same craned head from the same photograph, and settles somewhere the
anatomical prior does not want. (The measured lean goes 32.4 -> 36.0 deg,
but that metric is relative — the hips came 17 mm forward and the neck 27 mm
back, rotating the torso axis more than the head-neck vector rotated.)

Expressing this nod in pose space instead of as a vertex deformation would
remove the mechanical conflict, but it would NOT make the two independent:
the pose fit would still pull the head toward the shell and partly undo it.
The real answer is to put the anatomical constraint into that step's
objective as a term, so a single optimisation trades shell agreement against
plausibility.

**The reprojection compensation.** A bare rotation preserves the neck's
length, so straightening a lean also lifts the head: by `L*(cos10 - cos32)`
= 0.14 L, which on cyber_6f's 163 mm neck is 23 mm, and at that fit's
f=1731 with the head 2.24 m out is 10.7 px up the frame (plus 6.3 px
sideways, since the subject is not square to the camera). An eighth of a
head. That lift is invisible in isolation but it desynchronises
the mesh from the PHOTOGRAPH, which still shows the head craned: the face
splat is pinned to the photo's rays by construction (`face_pointmap_splat`
puts every Gaussian on the ray through the pixel it came from), so the
splat stays put while the mesh head walks out from under it. Worse, that
step scales the splat's depth by comparing, per image bin, the mesh's front
surface against the pointmap's median depth UNDER THE FACE MATTE — also in
the photo's pixels. Move the mesh head off those pixels and the solve
compares cheek against jaw, biasing the one scalar it produces, and the
face then swings across the head by parallax as the orbit turns.

So the nod is followed by a compensating translation, graded by the same
smoothstep and therefore expressed entirely as a SHORTER NECK: the head
centre is put back on its original ray from the SAM-3D-Body camera, at the
depth the rotation gave it. Same pixel, corrected distance. Since a pinhole
at the origin maps every point on a ray to one pixel, this needs only
`cam_t` — no focal length, no principal point (`focal_length` is optional,
and used solely to report the correction in pixels).

Note what this does to the `auto` residual: the compensation moves the head
along the torso axis, which is the large component of the neck vector, so
the residual LEAN comes out a couple of degrees above `target_lean_deg` —
11.9 rather than 10 on cyber_6f, over a neck 6.4% shorter. The head's
forward DISPLACEMENT is unchanged from the bare rotation; the angle grew
only because the neck did. The step logs both, plus the reprojection it
cancelled, so a run says outright how far the face splat used to be off.

The drift grows with the correction, and near `max_correction_deg` it eats
most of the angular benefit (a capped 20-degree nod on a 58-degree lean
leaves ~48 degrees of measured lean, on a fifth less neck, while moving the
head exactly as far back as the bare nod would). If that ever matters,
the fix is to solve `auto` for the angle whose POST-compensation lean hits
the target — a bisection over the same transform — not to weaken this.

**Known limitations** (it is a stopgap): only the ray is restored, not the
distance, so the head is left slightly smaller in projection than the
photograph's — the nod takes it 59 mm further out on cyber_6f, i.e. ~2.6%
at that camera, and the face matte overhangs the mesh head by that much. The 2D face
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


def _project(points: np.ndarray, focal: float) -> np.ndarray:
    """Pixel offsets from the principal point, for points in the camera frame.

    Only ever used on DIFFERENCES of two projections, so the principal point
    cancels and never has to be known. Nothing else here needs the focal
    length either — see `_ray_shift`.
    """
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return focal * p[:, :2] / p[:, 2:3]


def _ray_shift(before: np.ndarray, after: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """Translation putting `after` back on the camera ray through `before`.

    A pinhole with the camera at the origin maps every point of a ray to one
    pixel, so "same pixel" is "same ray" and neither the focal length nor the
    principal point enters. Of the one-parameter family of points on that ray
    we take the one at `after`'s own depth, which is what keeps the nod's
    depth correction while discarding its image-plane drift. The returned
    vector therefore has no z component in the camera frame.
    """
    h0 = np.asarray(before, dtype=np.float64) + cam_t
    h1 = np.asarray(after, dtype=np.float64) + cam_t
    if h0[2] <= 1e-6 or h1[2] <= 1e-6:
        raise ValueError(
            f"head_angle_fix: the head centre is not in front of the camera "
            f"(z={h0[2]:.4f} before the nod, {h1[2]:.4f} after) — 'cam_t' is "
            f"probably not SAM-3D-Body's `pred_cam_t` for this mesh"
        )
    return (h1[2] / h0[2]) * h0 - h1


@register_step("head_angle_fix")
class HeadAngleFixStep(Step):
    """Nod a craned-forward head back toward the torso axis.

    inputs:  {"vertices": (N, 3) float — SAM-3D-Body `pred_vertices`,
              "keypoints_3d": (70, 3) float — MHR70 `pred_keypoints_3d`,
              "cam_t": (3,) float — SAM-3D-Body `pred_cam_t`, REQUIRED,
              "focal_length": float — optional, only to log pixels}

    outputs: {"vertices": (N, 3) float, "keypoints_3d": (70, 3) float}

    Both arrive in SAM-3D-Body's raw output space and are returned in it,
    bent by one weighted rotation plus the compensating translation the
    module docstring describes. Wire the outputs back over the same
    `scene.vertices` / `scene.keypoints_3d` the reconstruction wrote.

    `cam_t` is required and not optional-with-a-fallback on purpose: without
    it there is no ray to hold the head on, and the silent alternative — a
    bare rotation — is the behaviour this step had when the face splat was
    landing off the mesh head.
    """

    PARAMS = (
        Param("mode", str, "auto",
              "'auto' measures the head's forward lean and straightens it to "
              "target_lean_deg; 'fixed' rotates back by exactly pitch_deg",
              choices=("auto", "fixed")),
        Param("target_lean_deg", float, 10.0,
              "auto mode: forward lean to leave, in degrees between the "
              "neck-to-head vector and the torso-up axis. The head is nodded "
              "back only by the excess over this. The lean measured after the "
              "reprojection compensation lands a couple of degrees above it, "
              "because that shortens the neck — see the module docstring",
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
        # Checked before the passthrough branches below, so a workflow that
        # forgot to wire it fails on every input rather than only on the ones
        # whose head happens to need nodding.
        if inputs.get("cam_t") is None:
            raise KeyError(
                "head_angle_fix requires 'cam_t' (SAM-3D-Body's pred_cam_t) to "
                "hold the head on its original camera ray while it nods — wire "
                "scene.cam_t. See the module docstring."
            )
        cam_t = np.asarray(inputs["cam_t"], dtype=np.float64).reshape(3)
        focal = inputs.get("focal_length")
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

        def _weight(points: np.ndarray) -> np.ndarray:
            t = np.clip(((points - neck) @ spine_up - lo) / span, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)  # smoothstep

        # The compensation is solved for the head CENTRE and then applied
        # graded, so it has to be divided by that point's own weight for the
        # head to receive exactly it. With the default blend band the head is
        # fully weighted and the division is by 1.
        head_weight = float(_weight(head_center[None])[0])
        head_rotated = neck + _rodrigues(
            head_center - neck, shoulder_axis, head_weight * theta_full
        )
        if head_weight < 1e-6:
            logger.warning(
                "head_angle_fix: the blend band leaves the head centre "
                "unweighted (blend_hi_frac=%.2f); nodding without the "
                "reprojection compensation", params["blend_hi_frac"]
            )
            shift = np.zeros(3)
        else:
            shift = _ray_shift(head_center, head_rotated, cam_t) / head_weight

        def _bend(points: np.ndarray) -> np.ndarray:
            rel = points - neck
            weight = _weight(points)
            rotated = _rodrigues(rel, shoulder_axis, weight * theta_full)
            return neck + rotated + weight[..., None] * shift

        new_vertices = _bend(vertices)
        new_joints = _bend(joints)

        head_final = head_rotated + head_weight * shift
        head_vec_final = head_final - neck
        neck_scale = float(np.linalg.norm(head_vec_final) / np.linalg.norm(head_vec))
        residual_lean = float(np.degrees(np.arctan2(
            abs(float(head_vec_final @ forward)), float(head_vec_final @ spine_up),
        )))

        logger.info(
            "head_angle_fix: head leaned %.1f deg forward; nodded back %.1f deg "
            "(mode=%s), then translated %.1f mm to hold its camera ray — neck "
            "%.1f%% shorter, %.1f deg residual lean",
            lean_deg, correction_deg, mode, 1000.0 * float(np.linalg.norm(shift)),
            100.0 * (1.0 - neck_scale), residual_lean,
        )
        if focal is not None:
            base = _project(head_center + cam_t, float(focal))
            uncompensated = _project(head_rotated + cam_t, float(focal)) - base
            residual = _project(head_final + cam_t, float(focal)) - base
            logger.info(
                "head_angle_fix: head centre reprojection — the bare nod would "
                "have moved it (%+.1f, %+.1f) px, compensated to (%+.1f, %+.1f) "
                "px. The face splat is pinned to the photo's pixels, so the "
                "first pair is what it used to be misregistered by.",
                uncompensated[0, 0], uncompensated[0, 1],
                residual[0, 0], residual[0, 1],
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
