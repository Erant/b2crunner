"""refine_pose_to_splat — re-pose a SAM-3D-Body fit so its mesh agrees
with the Gaussian shell in novel views.

The problem this solves
-----------------------
`pointmap_splat` builds a shell that reprojects onto the source photograph
exactly, and `render` puts the mesh on the same anchor camera, so at the
anchor the two agree and the skeleton overlay lands on the subject. Rotate
a few degrees and they separate — worst at the hands, which is where a
monocular body fit is least constrained. Measured on the smoke-test photo:
5.1% of the drawn skeleton's pixels fell on background across the +-19 deg
band, rising to 11.3% at 28 deg.

The cause is a depth disagreement the anchor view cannot see. Projected
into the anchor camera, the fit put the raised right wrist 112 mm *in front
of* the shell's own visible surface — impossible, since a joint is inside
the body — while the left wrist sat 120 mm behind it. Neither error shows
at the anchor, because both joints are on the right rays; they only appear
once the camera moves.

Why re-posing, and not warping
------------------------------
Two ways to make the two agree: move the shell (see `pointmap_splat`'s
`depth_prior="mesh"`) or move the body. Warping the shell's depth toward a
32 px-binned mesh field visibly deformed fingers and shoes, and made the
one independent metric — agreement with the predicted normals — worse. It
also cheats: forcing the shell to copy the mesh and then observing that the
mesh's skeleton agrees with it is circular.

Re-posing has none of those problems, because of one measured fact: the
correction needed moves the wrists 50-110 mm almost entirely **along the
anchor ray**, and changes the arm chain's bone lengths by under 4% — which
is articulation slack, not deformation. So a pose can reach it. And a pose
keeps the mesh a real body: `shape_params` and `scale_params` are frozen
here, so every bone length is preserved and the mesh cannot deform.

What makes it safe is the direction. Depth along the anchor ray is exactly
what one photograph cannot observe — it is the degree of freedom the body
fit had to *guess*. Changing it costs nothing at the anchor frame, which
still has to match the warped reference photo pixel for pixel; the 2D
reprojection penalty below is what holds that.

The target
----------
Not "put the joint on the surface" — a joint is inside the body. Each joint
should sit behind the **shell's** visible surface by exactly as far as it
sits behind the **mesh's own** surface, that offset being the body's real
local half-thickness, measured once at the input pose (median +26 mm on the
smoke-test subject, p10 +5, p90 +78). Self-consistent by construction: if
the shell and the mesh already agreed, the target asks for no change at
all.

An earlier version fitted to a per-joint triangulation from a ring of novel
viewpoints instead. It worked less well (right hand -112 -> -55 mm against
-23 mm here) because that target is a noisy per-joint solve which is not
itself a valid pose, so the articulated fit could only chase it partway.
Going straight at the metric is both simpler and better.

Measured result on the smoke-test photo
---------------------------------------
    right hand, joint behind surface   -112 mm  ->   -23 mm
    left hand                          +120 mm  ->   +54 mm
    all joints, p10                    -115 mm  ->   -25 mm
    skeleton pixels off the splat, mean over +-19 deg   5.1% -> 2.8%
    ...at 28 deg                                       11.3% -> 5.2%
    anchor 2D drift                    1.79 px median, 5.96 px max
    bone lengths over 65 bones         median 0.000%, max 4.72% change
    arm chain specifically             under 1%

The improvement grows with angle and is ~0 at the anchor, which is the
signature a real fix should have.

Caveats, honestly
-----------------
* **It fits the mesh to the shell, so a wrong shell is inherited.** Which
  of the two is actually right is not established — one photograph cannot
  separate "the fit put the hand too near" from "the depth network put the
  surface too far". For this pipeline's purpose that is acceptable: what
  the diffusion pass needs is a mesh render and a splat render that agree,
  and the anchor frame — the one place there is ground truth — is pinned by
  the 2D penalty. It is NOT a claim about the true pose.
* The residual is real: ~29 mm of depth error remains, and the right hand
  is still 23 mm on the wrong side. The articulated model cannot reach the
  target exactly.
* Bone lengths move by up to 4.7% on a few short bones despite frozen
  shape/scale, because the MHR70 keypoints are *regressed* from vertices
  rather than being the rig's own joints, so their spacing is not strictly
  pose-invariant. The arm chain, which is what positions the hands, stays
  under 1%.
* There is a mild chicken-and-egg: the shell's depth was scaled onto the
  mesh, and the mesh is now refined against the shell. One pass is what was
  measured. Re-running `pointmap_splat` afterwards is the natural second
  iteration and has not been tried.

Runs in the sam3dbody venv (`dispatch: subprocess`, `env: sam3dbody`) —
it needs the MHR body model's differentiable forward, which is what makes
"optimise the pose" possible at all. `sam3d_body` publishes the pose
parameters as `pose_params` so this step does not have to re-run a 2.8 GB
inference pass to recover them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

#: OpenCV camera frame <-> body2colmap world, and its own inverse. The same
#: diag(1,-1,-1) serves a second, easily-missed purpose here:
#: `mhr_head.forward` applies it to the vertices and keypoints AFTER calling
#: `mhr_forward`, and it is NOT inside that method — replaying a pose
#: without it puts the whole body ~2.8 m away.
FLIP = np.array([1.0, -1.0, -1.0])


def z_buffer(points: np.ndarray, focal: float, cx: float, cy: float,
             shape: Tuple[int, int], dilate: int) -> np.ndarray:
    """Nearest-surface depth per pixel, from a point set, +inf where empty.

    A real rasteriser would need pyrender and a working EGL, which this
    pipeline cannot count on (see docs/docker-build-notes.md on Blackwell).
    It does not need one: an MHR mesh is ~18k vertices and the shell is
    ~480k Gaussians, so projecting points and taking a per-pixel minimum,
    then a min-filter of radius `dilate` to close the gaps between them, is
    a good enough visible-surface estimate for a depth comparison. At
    `dilate=6` the mesh covers 74% of the subject — against the 32 px bins
    an earlier attempt used, which is what deformed fingers.
    """
    import cv2

    height, width = shape
    z = points[:, 2]
    u = focal * points[:, 0] / np.where(z > 1e-6, z, 1.0) + cx
    v = focal * points[:, 1] / np.where(z > 1e-6, z, 1.0) + cy
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    keep = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & (z > 1e-6)

    buffer = np.full((height, width), np.inf)
    if keep.any():
        np.minimum.at(buffer, (vi[keep], ui[keep]), z[keep])
    filled = np.where(np.isfinite(buffer), buffer, 1e6).astype(np.float32)
    if dilate > 0:
        filled = cv2.erode(filled, np.ones((2 * dilate + 1,) * 2, np.uint8))
    return filled


def project(points: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    z = np.clip(points[:, 2], 1e-6, None)
    return np.stack([focal * points[:, 0] / z + cx, focal * points[:, 1] / z + cy], 1)


def anatomical_offsets(joints_cam: np.ndarray, mesh_buffer: np.ndarray,
                       shell_buffer: np.ndarray, focal: float, cx: float,
                       cy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-joint depth behind the mesh's own surface, and which are usable.

    This is the quantity the optimisation preserves: the body's real local
    half-thickness at each joint. Asking the joint to sit that far behind
    the *shell* is what makes the target self-consistent — a shell that
    already matched the mesh would produce no correction at all.
    """
    height, width = mesh_buffer.shape
    pixels = np.round(project(joints_cam, focal, cx, cy)).astype(np.int64)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
              & (joints_cam[:, 2] > 1e-6))
    u = np.clip(pixels[:, 0], 0, width - 1)
    v = np.clip(pixels[:, 1], 0, height - 1)
    offsets = joints_cam[:, 2] - mesh_buffer[v, u]
    valid = inside & (mesh_buffer[v, u] < 1e5) & (shell_buffer[v, u] < 1e5)
    return offsets, valid


def read_splat_means(path: str) -> np.ndarray:
    """Gaussian centres from a binary 3DGS PLY, in the camera frame.

    `pointmap_splat` writes exactly 14 float32 properties per vertex in a
    known order, and this only ever reads files it wrote, so the header is
    scanned for the vertex count rather than parsed in general.
    """
    raw = open(path, "rb").read()
    marker = b"end_header\n"
    header = raw[: raw.index(marker)].decode("ascii", "replace")
    count = int(next(line for line in header.splitlines()
                     if line.startswith("element vertex")).split()[-1])
    properties = sum(1 for line in header.splitlines()
                     if line.startswith("property float"))
    body = np.frombuffer(raw[raw.index(marker) + len(marker):], np.float32)
    means = body.reshape(count, properties)[:, 0:3].astype(np.float64)
    return means * FLIP        # world -> camera


@register_step("refine_pose_to_splat")
class RefinePoseToSplatStep(Step):
    """Re-pose a SAM-3D-Body fit to agree with a Gaussian shell in depth.

    inputs:  {"mesh_output": dict,   sam3d_body's outputs, incl. pose_params
              "splat_path": str,     the shell to agree with
              "image": HxWx3}        the photo both were built from
    outputs: {"vertices", "keypoints_3d", "cam_t", "refine_stats"}
    """

    PARAMS = (
        Param("iterations", int, 1200,
              "Adam steps over the pose parameters", minimum=1),
        Param("learning_rate", float, 6e-4,
              "Step size on the articulation parameters. Adam on a bilinearly "
              "sampled depth map is a rough landscape — this was raised to "
              "2e-3 once and diverged at step 500", advanced=True),
        Param("reprojection_weight", float, 40.0,
              "How hard to hold the anchor view's 2D projection, per pixel "
              "of drift against the equivalent metres of depth error. The "
              "anchor frame carries the warped reference photo, so drift "
              "there is the one cost that is never acceptable"),
        Param("pose_regularisation", float, 0.5,
              "Pull back toward the input pose, so joints the shell says "
              "nothing about do not wander"),
        Param("mesh_dilate_px", int, 6,
              "Min-filter radius closing the gaps between projected mesh "
              "vertices; 6 px covers ~74% of the subject", advanced=True),
        Param("shell_dilate_px", int, 2,
              "Same for the shell, which is far denser and needs less",
              advanced=True),
        Param("checkpoint_repo", str, "facebook/sam-3d-body-dinov3",
              "HF repo the checkpoint is pulled from", advanced=True),
        Param("checkpoint_dir", str, None,
              "A local snapshot directory to use instead of downloading",
              advanced=True),
        Param("mhr_path", str, None,
              "The mhr_model.pt to load; empty means assets/mhr_model.pt "
              "inside the checkpoint directory", advanced=True),
        Param("device", str, "cuda", "Torch device", advanced=True),
    )

    def __init__(self) -> None:
        self._head = None

    def load(self, params: Dict[str, Any]) -> None:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from sam_3d_body import load_sam_3d_body

        checkpoint_dir = params["checkpoint_dir"] or snapshot_download(
            params["checkpoint_repo"]
        )
        mhr_path = params["mhr_path"] or str(
            Path(checkpoint_dir) / "assets" / "mhr_model.pt"
        )
        model, _ = load_sam_3d_body(
            str(Path(checkpoint_dir) / "model.ckpt"),
            device=params["device"], mhr_path=mhr_path,
        )
        # Only the pose head is needed: `mhr_forward` is the differentiable
        # body model. The image backbone is never run here — this step
        # re-poses an existing fit, it does not make a new one.
        self._head = model.head_pose

    def unload(self) -> None:
        self._head = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate before touching torch or the checkpoint: a missing wire
        # should say so in milliseconds, not after a 2.8 GB model load.
        mesh_output = inputs["mesh_output"]
        pose_params = mesh_output.get("pose_params")
        if not pose_params:
            raise ValueError(
                "refine_pose_to_splat needs sam3d_body's 'pose_params' in "
                "mesh_output — it re-runs the MHR body model's forward with "
                "them as free variables. Wire sam3d_body's pose_params output."
            )

        import torch

        if self._head is None:
            self.load(params)

        image = np.asarray(inputs["image"])
        height, width = image.shape[:2]
        focal = float(mesh_output["focal_length"])
        cx, cy = width / 2.0, height / 2.0
        device = params["device"]
        flip = torch.tensor(FLIP, dtype=torch.float32, device=device)

        def tensor(value, grad=False):
            out = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
            if out.ndim == 1:
                out = out[None]
            return out.requires_grad_(grad)

        global_rot0 = tensor(pose_params["global_rot"])
        body_pose0 = tensor(pose_params["body_pose_params"])
        hand_pose = tensor(pose_params["hand_pose_params"])
        scale = tensor(pose_params["scale_params"])
        shape = tensor(pose_params["shape_params"])
        expr = tensor(pose_params["expr_params"])
        cam_t0 = tensor(mesh_output["cam_t"])
        n_joints = len(np.asarray(mesh_output["keypoints_3d"]))

        head = self._head

        def forward(body_pose, global_rot):
            out = head.mhr_forward(
                global_trans=torch.zeros_like(global_rot), global_rot=global_rot,
                body_pose_params=body_pose, hand_pose_params=hand_pose,
                scale_params=scale, shape_params=shape, expr_params=expr,
                return_keypoints=True,
            )
            return out[0] * flip, out[1][:, :n_joints] * flip

        # Gate: the pose is only a usable handle if replaying it reproduces
        # the fit this step was handed. Everything below assumes it does.
        with torch.no_grad():
            verts0, keypoints0 = forward(body_pose0, global_rot0)
        drift = float(np.abs(verts0[0].cpu().numpy()
                             - np.asarray(mesh_output["vertices"])).max())
        if drift > 1e-3:
            raise RuntimeError(
                f"refine_pose_to_splat: replaying the pose parameters does not "
                f"reproduce the input mesh ({drift * 1000:.2f} mm off), so the "
                f"pose is not a faithful handle on this geometry and optimising "
                f"it would silently discard whatever changed the mesh.\n"
                f"The overwhelmingly likely cause is that `head_angle_fix` ran "
                f"first: it rewrites the vertices and keypoints as a graded "
                f"deformation without updating the pose parameters behind them, "
                f"which is exactly this symptom (a ~22 deg nod moves the crown "
                f"~75 mm). The two steps are mutually exclusive — see "
                f"INCOMPATIBLE_STEPS in pipeline/workflow.py. Otherwise the "
                f"pose params and the mesh are from different fits, or the "
                f"body model's convention has changed."
            )

        # --- the target ---------------------------------------------------
        cam_t_np = np.asarray(mesh_output["cam_t"], np.float64).reshape(3)
        joints_cam = np.asarray(mesh_output["keypoints_3d"], np.float64) + cam_t_np
        mesh_buffer = z_buffer(
            np.asarray(mesh_output["vertices"], np.float64) + cam_t_np,
            focal, cx, cy, (height, width), params["mesh_dilate_px"])
        shell_buffer = z_buffer(
            read_splat_means(inputs.get("splat_path") or mesh_output["splat_path"]),
            focal, cx, cy, (height, width), params["shell_dilate_px"])
        offsets, valid = anatomical_offsets(joints_cam, mesh_buffer, shell_buffer,
                                            focal, cx, cy)
        if valid.sum() < 8:
            raise ValueError(
                f"refine_pose_to_splat: only {int(valid.sum())} joints have both "
                f"a mesh and a shell surface under them — the two are not from "
                f"the same photo, or the shell is empty."
            )
        logger.info(
            "refine_pose_to_splat: %d/%d joints usable, anatomical offset "
            "median %+.0f mm (p10 %+.0f, p90 %+.0f)",
            int(valid.sum()), n_joints, np.median(offsets[valid]) * 1000,
            np.percentile(offsets[valid], 10) * 1000,
            np.percentile(offsets[valid], 90) * 1000,
        )

        shell_t = torch.as_tensor(
            np.where(shell_buffer < 1e5, shell_buffer, 0.0),
            dtype=torch.float32, device=device)[None, None]
        offsets_t = torch.as_tensor(offsets, dtype=torch.float32, device=device)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)

        def sample_shell(pixels):
            grid = torch.stack([pixels[..., 0] / (width - 1) * 2 - 1,
                                pixels[..., 1] / (height - 1) * 2 - 1], dim=-1)
            return torch.nn.functional.grid_sample(
                shell_t, grid[None, None], mode="bilinear",
                align_corners=True, padding_mode="border")[0, 0, 0]

        def to_pixels(cam):
            z = cam[..., 2].clamp(min=1e-3)
            return torch.stack([focal * cam[..., 0] / z + cx,
                                focal * cam[..., 1] / z + cy], dim=-1)

        with torch.no_grad():
            pixels0 = to_pixels(keypoints0[0] + cam_t0)

        # One pixel of lateral drift is about z/f metres at the subject's
        # distance; squaring that puts the 2D term in the same units as the
        # depth term, and `reprojection_weight` then reads as "how many
        # times more we care about a pixel than the equivalent metres".
        metres_per_px = float(np.median(joints_cam[:, 2])) / focal
        px_to_m2 = metres_per_px ** 2

        delta_pose = torch.zeros_like(body_pose0, requires_grad=True)
        delta_rot = torch.zeros_like(global_rot0, requires_grad=True)
        delta_cam = torch.zeros_like(cam_t0, requires_grad=True)
        learning_rate = params["learning_rate"]
        optimiser = torch.optim.Adam([
            {"params": [delta_pose], "lr": learning_rate},
            {"params": [delta_rot], "lr": learning_rate / 1.2},
            {"params": [delta_cam], "lr": learning_rate},
        ])
        iterations = params["iterations"]
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=iterations)
        weight_2d = params["reprojection_weight"]
        weight_reg = params["pose_regularisation"]

        def evaluate():
            _, keypoints = forward(body_pose0 + delta_pose, global_rot0 + delta_rot)
            cam = keypoints[0] + (cam_t0 + delta_cam)
            pixels = to_pixels(cam)
            wanted = sample_shell(pixels) + offsets_t
            depth_loss = torch.nn.functional.huber_loss(
                cam[..., 2][valid_t], wanted[valid_t], delta=0.05)
            drift_loss = ((pixels - pixels0) ** 2).sum(-1).mean()
            total = (depth_loss + weight_2d * px_to_m2 * drift_loss
                     + weight_reg * (delta_pose ** 2).mean())
            return total, cam, pixels, wanted

        best = (float("inf"), None)
        for step in range(iterations + 1):
            optimiser.zero_grad()
            total, cam, pixels, wanted = evaluate()
            if total.item() < best[0]:
                best = (total.item(), (delta_pose.detach().clone(),
                                       delta_rot.detach().clone(),
                                       delta_cam.detach().clone()))
            if step % max(iterations // 4, 1) == 0:
                with torch.no_grad():
                    err = (cam[..., 2] - wanted)[valid_t].abs().mean().item() * 1000
                    drift = (pixels - pixels0).norm(dim=-1).mean().item()
                logger.info("refine_pose_to_splat: step %d/%d  depth err %.1f mm  "
                            "anchor drift %.2f px", step, iterations, err, drift)
            if step == iterations:
                break
            total.backward()
            torch.nn.utils.clip_grad_norm_([delta_pose, delta_rot, delta_cam], 1.0)
            optimiser.step()
            schedule.step()

        # Keep the best iterate, not the last — this landscape does diverge.
        with torch.no_grad():
            delta_pose.copy_(best[1][0])
            delta_rot.copy_(best[1][1])
            delta_cam.copy_(best[1][2])
            total, cam, pixels, wanted = evaluate()
            verts, keypoints = forward(body_pose0 + delta_pose, global_rot0 + delta_rot)
            depth_error = (cam[..., 2] - wanted)[valid_t].abs().mean().item()
            anchor_drift = (pixels - pixels0).norm(dim=-1)

        cam_t = (cam_t0 + delta_cam)[0].detach().cpu().numpy()
        vertices = verts[0].detach().cpu().numpy()
        keypoints_np = keypoints[0].detach().cpu().numpy()

        stats = {
            "depth_error_mm": depth_error * 1000,
            "anchor_drift_px": {
                "median": float(anchor_drift.median()),
                "max": float(anchor_drift.max()),
            },
            "joints_used": int(valid.sum()),
            "cam_t_depth_change_m": float(cam_t[2] - cam_t_np[2]),
            "vertex_shift_mm": {
                "median": float(np.median(np.linalg.norm(
                    (vertices + cam_t) - (np.asarray(mesh_output["vertices"]) + cam_t_np),
                    axis=1)) * 1000),
            },
        }
        logger.info(
            "refine_pose_to_splat: depth err %.1f mm, anchor drift %.2f px "
            "(max %.2f), vertices moved %.1f mm median",
            stats["depth_error_mm"], stats["anchor_drift_px"]["median"],
            stats["anchor_drift_px"]["max"], stats["vertex_shift_mm"]["median"],
        )
        return {
            "vertices": vertices,
            "keypoints_3d": keypoints_np,
            "cam_t": cam_t,
            "refine_stats": stats,
        }
