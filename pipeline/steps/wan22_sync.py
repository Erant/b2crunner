"""In-loop 3D synchronisation for `wan22_vace_denoise` — M1 of
docs/latent-splat-guidance-research-2026-09-19.md.

At a chosen denoise step, under the euler sampler, the model's clean
estimate x0 = x_t - sigma * v is decoded, lifted into a Gaussian splat at
the dataset's cameras (b2ctrain, warm-started from the previous sync's
splat), rendered back at the same cameras, encoded again, and the LOW
radial band of that consistent latent replaces the low band of x0. The
step is then taken from the amended estimate, v' = (x_t - x0'') / sigma,
so each frame keeps its own noise and its own high frequencies
(SyncTweedies' case 2, band-limited).

Why only the low band, and why through pixels rather than a latent
splat: measured on this VAE (section 7 of the note). 95 % of a Wan
latent's energy is in the lowest sixth of its radial spectrum, and that
band is the only one that transforms like an image under sub-latent-
pixel motion (rel. err 0.05 at a half-pixel shift, 0.6-1.0 above it);
the decoder forgives a half-pixel error in that band (-5 dB) far more
than in the rest (-12 dB); and Gaussians carrying latent codes fit their
training views and fail every held-out one. So the render's own high
bands are never used — they are the encoder's per-view re-encoding of
the render, which is what x0 already has of its own frame.

The splat is trained on the decoded frames masked by `sync_masks` (dilated,
so a body painted a few pixels off the drawing is inside them) and the
render is composited over the decoded frame by its alpha: the subject is
made consistent, the background stays whatever the model painted.

Nothing here touches the transformer or its hooks. The only seams are
`pipe.vae` (decode/encode, under the same group offloading as the rest
of the pipeline) and the scheduler's euler step, which `_install_sampler`
routes through `LatentSync.velocity` at the chosen steps.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def radial_lowpass(height: int, width: int, band: float, edge: float = 0.03):
    """A (height, width) FFT-domain mask keeping normalised radial
    frequency below `band` (1.0 = the corner Nyquist, so 1/6 is "band 0"
    of the note's sixths), with a raised-cosine edge `edge` wide."""
    import torch

    fy = torch.fft.fftfreq(height)[:, None]
    fx = torch.fft.fftfreq(width)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2) / math.sqrt(0.5)
    if edge <= 0:
        return (r <= band).float()
    t = ((r - (band - edge)) / (2 * edge)).clamp(0, 1)
    return 0.5 * (1 + torch.cos(math.pi * t))


def blend_bands(x0, projected, mask, mix: float):
    """x0 with its low band moved toward `projected`'s:
    x0 + mix * lowpass(projected - x0). Both (B, C, T, h, w); the FFT runs
    over (h, w) per channel and frame, in float32."""
    import torch

    delta = torch.fft.fft2((projected - x0).float())
    delta = torch.fft.ifft2(delta * mask.to(delta.device)).real
    return (x0.float() + mix * delta).to(x0.dtype)


def band_share(z, mask) -> float:
    """Fraction of z's spectral energy inside `mask` — the note's 0.95."""
    import torch

    f = torch.fft.fft2(z.float())
    p = f.real ** 2 + f.imag ** 2
    return float((p * mask.to(p.device)).sum() / p.sum().clamp_min(1e-12))


def rel_err(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def _scaled_camera(camera, width: int, height: int):
    """`camera` at the denoise resolution: intrinsics scaled with the
    frame, pose untouched. A camera already at that size is returned as is."""
    if (int(camera.width), int(camera.height)) == (width, height):
        return camera
    from body2colmap.camera import Camera

    sx, sy = width / float(camera.width), height / float(camera.height)
    return Camera(
        focal_length=(float(camera.fx) * sx, float(camera.fy) * sy),
        image_size=(width, height),
        principal_point=(float(camera.cx) * sx, float(camera.cy) * sy),
        position=np.asarray(camera.position, np.float32),
        rotation=np.asarray(camera.rotation, np.float32),
    )


def cameras_json(cameras: Sequence[Any], image_names: Sequence[str]) -> Dict[str, Any]:
    """The renderer's cameras.json (OpenGL camera-to-world, as
    body2colmap's SplatRenderer and steps/body_refit.py write it)."""
    width, height = int(cameras[0].width), int(cameras[0].height)
    return {
        "width": width, "height": height,
        "cameras": [{
            "name": name,
            "fx": float(cam.fx), "fy": float(cam.fy), "cx": float(cam.cx), "cy": float(cam.cy),
            "position": [float(v) for v in cam.position],
            "rotation": [[float(v) for v in row] for row in cam.rotation],
        } for cam, name in zip(cameras, image_names)],
    }


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch absent or no card
        pass


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    m = mask
    if m.dtype != np.uint8:
        m = np.clip(np.asarray(m, np.float32) * (255.0 if m.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., -1]
    if px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        m = cv2.dilate(m, k)
    return m


class LatentSync:
    """One pass's synchroniser. Built by `Wan22VaceDenoiseStep.run` when
    `sync_steps` is non-empty, consulted by the euler step at those steps,
    dropped after `pipe()` returns.

    `n_ref` is how many latent frames at the front of the sample are the
    reference image's (pass 2 conditions on one); they are neither decoded
    nor touched.
    """

    def __init__(
        self, *, pipe, cameras: Sequence[Any], image_names: Sequence[str],
        points_3d: Optional[np.ndarray], masks: Optional[Sequence[np.ndarray]],
        width: int, height: int, n_ref: int, steps: Sequence[int], mix: Sequence[float],
        band: float, iters: int, warm_iters: int, max_splats: int, mask_dilate_px: int,
        trainer: str, debug_dir: Optional[str],
    ) -> None:
        if len(cameras) != len(image_names):
            raise ValueError(f"wan22_vace_denoise sync: {len(cameras)} cameras but {len(image_names)} image names")
        if masks is not None and len(masks) != len(cameras):
            raise ValueError(f"wan22_vace_denoise sync: {len(masks)} masks for {len(cameras)} cameras")
        self.pipe = pipe
        self.cameras = [_scaled_camera(c, width, height) for c in cameras]
        self.image_names = [str(n) for n in image_names]
        self.points_3d = points_3d
        self.masks = None if masks is None else [_dilate(m, mask_dilate_px) for m in masks]
        self.width, self.height = int(width), int(height)
        self.n_ref = int(n_ref)
        self.steps = [int(s) for s in steps]
        self.mix = {s: float(m) for s, m in zip(self.steps, mix)}
        self.band = float(band)
        self.iters, self.warm_iters, self.max_splats = int(iters), int(warm_iters), int(max_splats)
        self.trainer = trainer
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.workdir = Path(tempfile.mkdtemp(prefix="b2c_sync_"))
        self.previous_ply: Optional[Path] = None
        self.stats: List[Dict[str, Any]] = []
        self._lowpass = None

    # -- the seam ------------------------------------------------------------

    def wants(self, step_index: Optional[int]) -> bool:
        return step_index is not None and step_index in self.mix

    def velocity(self, model_output, sample, sigma: float, step_index: int):
        """The velocity to take the euler step with: (x_t - x0'') / sigma,
        where x0'' is x0 with its low band synchronised."""
        import torch

        started = time.time()
        x0 = (sample.float() - sigma * model_output.float())
        video = x0[:, :, self.n_ref:]
        projected = self.project(video, step_index)
        if self._lowpass is None:
            self._lowpass = radial_lowpass(video.shape[-2], video.shape[-1], self.band)
        mix = self.mix[step_index]
        synced = blend_bands(video, projected, self._lowpass, mix)
        stats = {
            "step": step_index, "sigma": sigma, "mix": mix,
            "rel_change_full": rel_err(projected, video),
            "rel_change_low_band": rel_err(
                blend_bands(torch.zeros_like(video), projected, self._lowpass, 1.0),
                blend_bands(torch.zeros_like(video), video, self._lowpass, 1.0)),
            "rel_applied": rel_err(synced, video),
            "x0_low_band_energy_share": band_share(video, self._lowpass),
            "seconds": time.time() - started,
        }
        self.stats[-1].update(stats)
        logger.info(
            "  sync step %d (sigma %.3f, mix %.2f): render moves x0 by %.3f (low band %.3f), "
            "applied %.3f; low band holds %.1f%% of x0's energy; %.1fs",
            step_index, sigma, mix, stats["rel_change_full"], stats["rel_change_low_band"],
            stats["rel_applied"], 100 * stats["x0_low_band_energy_share"], stats["seconds"],
        )
        x0_new = x0.clone()
        x0_new[:, :, self.n_ref:] = synced
        return ((sample.float() - x0_new) / sigma).to(model_output.dtype)

    # -- decode -> splat -> render -> encode ----------------------------------

    def project(self, latents, step_index: int):
        """The 3D-consistent version of the video latents (B, C, T', h, w),
        in the same normalised space."""
        self.stats.append({"step": step_index})
        t0 = time.time()
        frames = self.decode(latents)
        t1 = time.time()
        rendered = self.lift(frames, step_index)
        t2 = time.time()
        z = self.encode(rendered, latents.device).to(latents.dtype)
        t3 = time.time()
        self.stats[-1].update({"decode_s": t1 - t0, "lift_s": t2 - t1, "encode_s": t3 - t2})
        logger.info("  sync step %d: decode %.1fs, splat+render %.1fs, encode %.1fs",
                    step_index, t1 - t0, t2 - t1, t3 - t2)
        if self.debug_dir is not None:
            self._dump(step_index, frames, rendered)
        return z

    def _vae_scaling(self):
        import torch

        vae = self.pipe.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return vae, mean, std

    def decode(self, latents) -> List[np.ndarray]:
        """Normalised latents -> BGR uint8 frames at the denoise size, the
        same arithmetic as the pipeline's own decode."""
        import torch

        vae, mean, std = self._vae_scaling()
        raw = (latents.float() * std.to(latents.device) + mean.to(latents.device)).to(vae.dtype)
        with torch.no_grad():
            video = vae.decode(raw, return_dict=False)[0]
        video = video.float().clamp(-1, 1)[0].permute(1, 2, 3, 0).cpu().numpy()  # T,H,W,3 RGB
        frames = ((video + 1) * 127.5).round().astype(np.uint8)[..., ::-1]  # BGR
        return [np.ascontiguousarray(f) for f in frames]

    def encode(self, frames: Sequence[np.ndarray], device):
        """BGR uint8 frames -> normalised latents on `device`, as
        `prepare_video_latents` does it (argmax of the posterior, then
        (z - mean) / std)."""
        import torch

        vae, mean, std = self._vae_scaling()
        arr = np.stack([f[..., ::-1] for f in frames]).astype(np.float32) / 127.5 - 1.0  # T,H,W,3 RGB
        video = torch.from_numpy(arr).permute(3, 0, 1, 2)[None].to(device, vae.dtype)
        with torch.no_grad():
            z = vae.encode(video, return_dict=False)[0].mode()
        return (z.float() - mean.to(z.device)) / std.to(z.device)

    def lift(self, frames: Sequence[np.ndarray], step_index: int) -> List[np.ndarray]:
        """Train the splat on `frames` at the dataset's cameras, render it
        back at them, composite over `frames` by the render's alpha."""
        from body2colmap.exporter import ColmapExporter

        if len(frames) != len(self.cameras):
            raise RuntimeError(
                f"wan22_vace_denoise sync: the decode gave {len(frames)} frames for {len(self.cameras)} cameras"
            )
        work = self.workdir / f"step_{step_index:02d}"
        colmap = work / "colmap"
        colmap.mkdir(parents=True, exist_ok=True)
        ColmapExporter(cameras=self.cameras, image_names=self.image_names, points_3d=self.points_3d).export(output_dir=colmap)
        images_dir = colmap / "images"
        images_dir.mkdir(exist_ok=True)
        for frame, name in zip(frames, self.image_names):
            cv2.imwrite(str(images_dir / name), frame)
        if self.masks is not None:
            masks_dir = colmap / "masks"
            masks_dir.mkdir(exist_ok=True)
            for mask, name in zip(self.masks, self.image_names):
                if mask.shape[:2] != (self.height, self.width):
                    mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(masks_dir / name), mask)
        warm = self.previous_ply is not None and self.previous_ply.exists()
        if warm:
            (colmap / "init.ply").symlink_to(self.previous_ply.resolve())
        # The trainer is another CUDA process on the same card: hand back
        # what the caching allocator is holding but not using (the decode's
        # intermediates, mostly) before it starts, or it starts by failing.
        _release_cuda_cache()
        iters = self.warm_iters if warm else self.iters
        out = work / "splat"
        cmd = [
            self.trainer, str(colmap),
            "--total-train-iters", str(iters),
            "--export-path", str(out), "--export-name", "sync.ply", "--export-every", str(iters),
            "--max-resolution", str(max(self.width, self.height)),
            "--max-splats", str(self.max_splats),
            "--eval-every", "1000000",
        ]
        t0 = time.time()
        self._run(cmd, "train")
        ply = out / "sync.ply"
        if not ply.exists():
            raise RuntimeError(f"wan22_vace_denoise sync: {self.trainer} exited 0 but wrote no {ply}")
        t1 = time.time()
        renders = work / "renders"
        cams = work / "cameras.json"
        cams.write_text(json.dumps(cameras_json(self.cameras, self.image_names)))
        # Rendered over BLACK, which is also what the trainer composited
        # over (its `--background-color` default): inside the mask the
        # render IS the frame as the splat learned it, soft edges and all,
        # so the composite is by the mask alone. Compositing by the render's
        # alpha on top of that counts the background twice — a grey band
        # learned as grey-over-black gets grey-over-grey — and decoded as a
        # bright halo round the subject (measured locally, twice).
        self._run([self.trainer, "render", "--splat", str(ply), "--cameras", str(cams),
                   "--output-dir", str(renders), "--background", "0,0,0"], "render")
        t2 = time.time()
        composited = []
        coverage = []
        for index, (frame, name) in enumerate(zip(frames, self.image_names)):
            rgba = cv2.imread(str(renders / name), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                raise RuntimeError(f"wan22_vace_denoise sync: the render wrote no {renders / name}")
            rgb = rgba[..., :3].astype(np.float32)
            if self.masks is not None:
                # The (dilated) subject, with a soft edge so the seam between
                # the render and the frame's own background is not a step.
                m = self.masks[index]
                if m.shape[:2] != rgb.shape[:2]:
                    m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                weight = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 4)[..., None]
            else:
                weight = np.ones(rgb.shape[:2] + (1,), np.float32)
            coverage.append(float(weight.mean()))
            composited.append(np.clip(rgb * weight + frame.astype(np.float32) * (1 - weight), 0, 255).astype(np.uint8))
        self.previous_ply = ply
        self.stats[-1].update({
            "train_s": t1 - t0, "render_s": t2 - t1, "warm": warm, "iters": iters,
            "render_coverage": float(np.mean(coverage)),
        })
        logger.info("  sync step %d: %s %d iters in %.1fs%s, render %.1fs, render covers %.1f%% of the frame",
                    step_index, self.trainer, iters, t1 - t0, " (warm)" if warm else "", t2 - t1,
                    100 * float(np.mean(coverage)))
        return composited

    def _run(self, cmd: List[str], what: str) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]
            raise RuntimeError(f"wan22_vace_denoise sync: {what} failed ({result.returncode}): {' '.join(cmd)}\n{tail}")

    def _dump(self, step_index: int, frames: Sequence[np.ndarray], rendered: Sequence[np.ndarray]) -> None:
        d = self.debug_dir / f"step_{step_index:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(frames), max(1, len(frames) // 8)):
            cv2.imwrite(str(d / f"{i:03d}_x0.jpg"), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(d / f"{i:03d}_render.jpg"), rendered[i], [cv2.IMWRITE_JPEG_QUALITY, 90])

    def finish(self) -> List[Dict[str, Any]]:
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "sync_stats.json").write_text(json.dumps(self.stats, indent=1))
        return self.stats
