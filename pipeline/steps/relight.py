"""relight_frames — divide each final frame by its own lighting before the deliverable training.

Why (measured 2026-10-02 on five skin-heavy subjects, scratchpad study in
memory `skin_sh_material`): the frames the final training sees come from
three separately generated passes, and the generator relights the subject
from frame to frame. Exposure swings 10-23 % and the white balance 6-12 %
across the 162 frames. The splat's SH bands 1-3 soak that up: on skin a
per-frame low-order lighting over the surface normal explains 52-68 % of
what the SH carries, and view terms on top of it (n.v, a glossy lobe,
Fresnel) add ~1 %. It is lighting, not material.

What this does: a splat trained briefly on the same frames (the workflow's
`relight_pretrain`, ~5k iterations) gives every surface point a position, a
normal and the frames that see it. The frames are fitted as

    F(p, c) = rho(p) * k_c(n_p)          (linear light, per RGB channel)

with rho a colour per surface point and k_c a degree-2 SH function of the
world normal per frame (27 numbers a frame). The part all frames share is
pinned into rho (per point, the mean of k_c over the frames that see it is
1), so what is divided out is each frame's DEPARTURE from the common
lighting, not the lighting itself. Every frame is then divided by its k_c,
evaluated per pixel at the splat's rendered normal.

Measured on two subjects with retraining (plain trainer flags): the SH
share on skin 23 -> 16 % and 47 -> 28 %, pass 2 vs the extensions as
separate trainings +1.5 / +1.9 dB in agreement, and the colour cast between
the passes (one subject's extensions ~9 % bluer) gone. The DC colour is NOT
a better albedo for it (it shifts 5-7 % with view coverage either way).

The geometry comes from a short training rather than the body mesh because
the mesh does not cover bulky clothing: a body vertex under a coat lands on
a different point of the coat from every angle, and its z-test calls an arm
visible behind a cape. The fit runs on the matte classes of the frames' own
class maps when they are wired (hair, shoes, glasses and the mouth's
insides are left out); without them, on the whole matte.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

#: Goliath classes the fit leaves out: Eyeglass, Hair, the shoes, the teeth and the tongue. Dark, glossy or
#: anisotropic; on these the per-frame multiplicative model measured ~0 % of the view dependence.
DEFAULT_EXCLUDE = (2, 4, 9, 18, 26, 27, 28)

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)


def sh9(n: np.ndarray) -> np.ndarray:
    """Real SH degree 0-2 of unit vectors [..., 3] (the 3DGS basis), scaled so the first function is 1."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    out = np.stack([
        np.full_like(x, _C0),
        -_C1 * y, _C1 * z, -_C1 * x,
        _C2[0] * x * y, _C2[1] * y * z, _C2[2] * (2 * z * z - x * x - y * y), _C2[3] * x * z,
        _C2[4] * (x * x - y * y),
    ], -1)
    return out / _C0


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((np.clip(x, 0, None) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _quat_axes(quats_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrices [N, 3, 3] from w x y z quaternions."""
    q = quats_wxyz / np.maximum(np.linalg.norm(quats_wxyz, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def project(points: np.ndarray, camera: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, depth) of world points in a body2colmap camera (camera-to-world rotation, OpenGL axes)."""
    rot = np.asarray(camera.rotation, np.float64).reshape(3, 3)
    pos = np.asarray(camera.position, np.float64).reshape(3)
    pc = (points - pos) @ rot
    z = -pc[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = float(camera.cx) + float(camera.fx) * pc[:, 0] / zs
    v = float(camera.cy) - float(camera.fy) * pc[:, 1] / zs
    return u, v, z


def visible(points: np.ndarray, occluders: np.ndarray, camera: Any, *, grid: int, tolerance: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Which `points` `camera` sees: in frame, in front, and within `tolerance` of the z-buffer the opaque
    `occluders`' centres make at 1/`grid` resolution. Returns (mask, u, v)."""
    w, h = int(camera.width), int(camera.height)
    gw, gh = max(w // grid, 1), max(h // grid, 1)
    u, v, z = project(occluders, camera)
    gu, gv = (u / grid).astype(np.int64), (v / grid).astype(np.int64)
    ok = (z > 0) & (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh)
    zbuf = np.full(gh * gw, np.inf)
    np.minimum.at(zbuf, gv[ok] * gw + gu[ok], z[ok])
    u, v, z = project(points, camera)
    gu, gv = (u / grid).astype(np.int64), (v / grid).astype(np.int64)
    inb = (z > 0) & (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zmin = np.full(len(points), np.inf)
    zmin[inb] = zbuf[gv[inb] * gw + gu[inb]]
    return inb & (z <= zmin + tolerance), u, v


def fit_frame_lighting(point: np.ndarray, cam: np.ndarray, colour: np.ndarray, normals: np.ndarray, n_cams: int, *,
                       iters: int = 4, kmin: float = 0.5, kmax: float = 2.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit F(p, c) = rho(p) * k_c(n_p) on sparse samples.

    point, cam: [E] int sample -> surface point / frame; colour [E, 3] linear; normals [P, 3] unit world normals.
    Returns beta [n_cams, 9, 3] (k_c(n) = sh9(n) @ beta[c]) and fit stats. Frames with too few samples keep k = 1.
    """
    X = sh9(normals)                                         # [P, 9]
    n_pts = len(normals)
    beta = np.zeros((n_cams, 9, 3))
    beta[:, 0, :] = 1.0
    order = np.argsort(cam, kind="stable")
    bounds = np.searchsorted(cam[order], np.arange(n_cams + 1))
    per_cam = [order[bounds[c]:bounds[c + 1]] for c in range(n_cams)]
    count = np.maximum(np.bincount(point, minlength=n_pts), 1)[:, None]

    def k_of(b):
        return np.clip(np.einsum("ek,ekr->er", X[point], b[cam]), kmin, kmax)

    k = k_of(beta)
    for _ in range(iters):
        rho = np.zeros((n_pts, 3))
        np.add.at(rho, point, colour / k)
        rho /= count
        for c, e in enumerate(per_cam):
            if len(e) < 50:
                continue
            for ch in range(3):
                A = X[point[e]] * rho[point[e], ch:ch + 1]
                beta[c, :, ch] = np.linalg.lstsq(A, colour[e, ch], rcond=None)[0]
        # gauge: per point, the mean of k over the frames that see it -> 1 (the static lighting stays in rho)
        k = np.einsum("ek,ekr->er", X[point], beta[cam])
        mean_k = np.zeros((n_pts, 3))
        np.add.at(mean_k, point, k)
        seen = np.bincount(point, minlength=n_pts) > 0
        mean_k = mean_k[seen] / count[seen]
        q = np.stack([np.linalg.lstsq(X[seen], mean_k[:, ch], rcond=None)[0] for ch in range(3)], 1)
        Q = np.maximum(X @ q, 1e-3)
        for c, e in enumerate(per_cam):
            if len(e) < 50:
                continue
            for ch in range(3):
                beta[c, :, ch] = np.linalg.lstsq(X[point[e]], k[e, ch] / Q[point[e], ch], rcond=None)[0]
        k = k_of(beta)

    rho = np.zeros((n_pts, 3))
    np.add.at(rho, point, colour / k)
    rho /= count
    mean_c = np.zeros((n_pts, 3))
    np.add.at(mean_c, point, colour)
    mean_c /= count
    spread = ((colour - mean_c[point]) ** 2).sum()
    resid = ((colour - rho[point] * k) ** 2).sum()
    stats = {"samples": int(len(point)), "points": int((np.bincount(point, minlength=n_pts) > 0).sum()),
             "explained": float(1 - resid / max(spread, 1e-12)),
             "frames_fitted": int(sum(len(e) >= 50 for e in per_cam))}
    return beta, stats


def correction_map(normal_rgb: np.ndarray, alpha: np.ndarray, beta_c: np.ndarray, *, kmin: float, kmax: float
                   ) -> np.ndarray:
    """Per-pixel k [H, W, 3] from a rendered normal map (RGB uint8, normal = 2 * rgb / 255 - 1), faded to 1
    where the splat's alpha is."""
    n = normal_rgb.astype(np.float64) / 255.0 * 2.0 - 1.0
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    k = np.clip(np.einsum("hwk,kr->hwr", sh9(n), beta_c), kmin, kmax)
    a = np.clip(alpha.astype(np.float64), 0, 1)[..., None]
    return 1.0 + a * (k - 1.0)


def apply_correction(image_bgr: np.ndarray, k_rgb: np.ndarray) -> np.ndarray:
    """Divide a BGR(A) uint8 frame by k in linear light; alpha (if any) untouched."""
    rgb = image_bgr[..., 2::-1].astype(np.float64) / 255.0
    out = linear_to_srgb(srgb_to_linear(rgb) / k_rgb)
    res = image_bgr.copy()
    res[..., :3] = (out[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
    return res


@register_step("relight_frames")
class RelightFramesStep(Step):
    """Divide every frame by its own departure from the common lighting (see the module docstring).

    inputs: {"cameras": List[Camera], "images": List[np.ndarray] BGR(A) uint8,
             "splat_path": str — a splat trained on these frames and cameras (geometry only),
             "masks": Optional[List[np.ndarray]] float32 [0, 1] — the frames' mattes,
             "labels": Optional[List[np.ndarray]] uint8 class ids — restricts the fit to matte classes}
    outputs: {"images": List[np.ndarray] — the corrected frames, same shape and dtype,
              "relight_stats": dict}
    """

    PARAMS = (
        # Opaque, flat splats; more costs time, not accuracy.
        Param("max_points", int, 60000,
              "Surface splats sampled for the lighting fit",
              minimum=1000),
        Param("min_views", int, 12, "A surface point is used when this many frames see it", minimum=2),
        # Default: eyeglass, hair, shoes, teeth, tongue.
        Param("exclude_classes", list, list(DEFAULT_EXCLUDE),
              "Goliath class ids the fit ignores; needs `labels`"),
        Param("k_min", float, 0.5, "Lower clamp of the per-pixel correction factor", minimum=0.05),
        Param("k_max", float, 2.0, "Upper clamp of the per-pixel correction factor", minimum=1.0),
        Param("iterations", int, 4, "Alternating rounds of the fit (it settles in two)", minimum=1),
        Param("grid", int, 8, "Visibility z-buffer cell size, in pixels", minimum=1, advanced=True),
        Param("depth_tolerance", float, 0.015,
              "Depth (m) behind the visible surface a splat may sit and still count as seen", minimum=0.0,
              advanced=True),
        Param("min_samples", int, 20000,
              "Leave the frames untouched below this many (point, frame) samples", minimum=0, advanced=True),
        Param("render_path", str, "brush-splat-render", "The rasteriser binary for the normal maps", advanced=True),
        Param("debug_dir", str, "", "Write stats.json and the per-frame lighting coefficients here"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        cameras = list(inputs["cameras"])
        images = list(inputs["images"])
        masks = inputs.get("masks")
        labels = inputs.get("labels")
        if len(images) != len(cameras):
            raise ValueError(f"relight_frames: {len(images)} images for {len(cameras)} cameras")
        for name, maps in (("masks", masks), ("labels", labels)):
            if maps is not None and len(maps) != len(cameras):
                raise ValueError(f"relight_frames: {len(maps)} {name} for {len(cameras)} cameras")

        from body2colmap.splat_scene import SplatScene

        scene = SplatScene.from_ply(str(inputs["splat_path"]))
        kmin, kmax = float(params["k_min"]), float(params["k_max"])
        means = scene.means.astype(np.float64)
        opacity = 1.0 / (1.0 + np.exp(-scene.opacities.astype(np.float64).reshape(-1)))
        scales = np.exp(scene.scales.astype(np.float64))
        axes = _quat_axes(scene.quats.astype(np.float64))
        shortest = np.argmin(scales, 1)
        normals = axes[np.arange(len(means)), :, shortest]
        srt = np.sort(scales, 1)
        flat = srt[:, 0] / np.maximum(srt[:, 1], 1e-12)
        occluders = means[opacity > 0.3]

        # Visibility of every splat, to orient the normals (toward the frames that see them) and to sample.
        grid, tol = int(params["grid"]), float(params["depth_tolerance"])
        toward = np.zeros_like(means)
        vis_all = []
        for cam in cameras:
            seen, u, v = visible(means, occluders, cam, grid=grid, tolerance=tol)
            d = np.asarray(cam.position, np.float64).reshape(3) - means
            toward[seen] += d[seen] / np.maximum(np.linalg.norm(d[seen], axis=1, keepdims=True), 1e-9)
            vis_all.append((seen, u, v))
        normals *= np.sign((normals * toward).sum(1, keepdims=True) + 1e-12)

        rng = np.random.default_rng(0)
        cand = np.flatnonzero((opacity > 0.5) & (flat < 0.5))
        if len(cand) > int(params["max_points"]):
            cand = np.sort(rng.choice(cand, int(params["max_points"]), replace=False))
        exclude = np.asarray(params["exclude_classes"] or [], np.int64)
        pts, cams, cols = [], [], []
        for c, (seen, u, v) in enumerate(vis_all):
            s = cand[seen[cand]]
            iu = np.clip(np.round(u[s]).astype(np.int64), 0, int(cameras[c].width) - 1)
            iv = np.clip(np.round(v[s]).astype(np.int64), 0, int(cameras[c].height) - 1)
            keep = np.ones(len(s), bool)
            if masks is not None:
                keep &= np.asarray(masks[c])[iv, iu] > 0.5
            if labels is not None and len(exclude):
                keep &= ~np.isin(np.asarray(labels[c])[iv, iu], exclude) & (np.asarray(labels[c])[iv, iu] > 0)
            s, iu, iv = s[keep], iu[keep], iv[keep]
            bgr = np.asarray(images[c])[iv, iu, :3].astype(np.float64) / 255.0
            pts.append(s); cams.append(np.full(len(s), c)); cols.append(srgb_to_linear(bgr[:, ::-1]))
        point, cam, colour = np.concatenate(pts), np.concatenate(cams), np.concatenate(cols)
        # only points seen often enough
        seen_count = np.bincount(point, minlength=len(means))
        ok = seen_count[point] >= int(params["min_views"])
        point, cam, colour = point[ok], cam[ok], colour[ok]
        stats: Dict[str, Any] = {"splats": int(len(means)), "candidates": int(len(cand))}
        if len(point) < int(params["min_samples"]):
            logger.warning("relight_frames: %d samples (< %d): frames pass through untouched",
                           len(point), int(params["min_samples"]))
            stats["skipped"] = "too few samples"
            return {"images": images, "relight_stats": stats}

        uniq, local = np.unique(point, return_inverse=True)
        beta, fit = fit_frame_lighting(local, cam, colour, normals[uniq], len(cameras),
                                       iters=int(params["iterations"]), kmin=kmin, kmax=kmax)
        stats.update(fit)

        # Per-pixel normals: the same splat with its oriented normal as the DC colour, over 0.5 grey (= no normal).
        from .splat import _rasterize

        sh = np.zeros_like(scene.sh_coeffs[:, :1, :])
        sh[:, 0, :] = (normals * 0.5 / _C0).astype(np.float32)
        nscene = SplatScene(means=scene.means, scales=scene.scales, quats=scene.quats, opacities=scene.opacities,
                            sh_coeffs=sh, sh_degree=0)
        width, height = int(cameras[0].width), int(cameras[0].height)
        normal_bgr, normal_alpha = _rasterize(
            scene=nscene, splat_path=None, cameras=cameras, image_names=[f"{i:05d}" for i in range(len(cameras))],
            width=width, height=height, bg_color=(0.5, 0.5, 0.5), render_path=params["render_path"])

        out, per_frame = [], []
        for c, image in enumerate(images):
            k = correction_map(normal_bgr[c][..., ::-1], normal_alpha[c], beta[c], kmin=kmin, kmax=kmax)
            out.append(apply_correction(np.asarray(image), k))
            fg = (np.asarray(masks[c]) > 0.5) if masks is not None else (normal_alpha[c] > 0.5)
            per_frame.append(np.median(k[fg], 0).round(4).tolist() if fg.any() else [1.0, 1.0, 1.0])
        lum = np.asarray(per_frame).mean(1)
        stats.update({"median_k_range": [float(lum.min()), float(lum.max())], "median_k_per_frame": per_frame})
        logger.info("relight_frames: %d samples on %d points, %d/%d frames fitted, %.0f%% of the per-view "
                    "variation explained; median correction per frame x%.3f..x%.3f",
                    stats["samples"], stats["points"], stats["frames_fitted"], len(cameras),
                    100 * stats["explained"], lum.min(), lum.max())
        if params["debug_dir"]:
            debug = Path(params["debug_dir"])
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "stats.json").write_text(json.dumps(stats, indent=1))
            np.save(debug / "lighting_coefficients.npy", beta.astype(np.float32))
        return {"images": out, "relight_stats": stats}
