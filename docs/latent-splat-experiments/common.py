"""Shared pieces for the Wan-VAE latent experiments (E0-E2 in
docs/latent-splat-guidance-research-2026-09-19.md).

Runs with only the VAE of linoyts/Wan2.2-VACE-Fun-14B-diffusers (the Wan
2.1 VAE every Wan 2.2 14B model uses), in the masktest venv on the
4070 Ti. Latents are always handled in the *normalised* space the
transformer sees: (z - latents_mean) / latents_std per channel, the
same arithmetic as diffusers' pipeline_wan_vace.py.
"""
import os, glob, math, time
import numpy as np
import torch
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAE_REPO = "linoyts/Wan2.2-VACE-Fun-14B-diffusers"
DEV = torch.device("cuda")


def load_vae(dtype=torch.float32):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae", torch_dtype=dtype).to(DEV)
    vae.eval()
    vae.requires_grad_(False)
    mean = torch.tensor(vae.config.latents_mean, device=DEV).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=DEV).view(1, -1, 1, 1, 1)
    return vae, mean, std


def load_frames(dirname="circular", width=480, height=832, n=81):
    """cyber_6f's frames, resized to the denoise size, as float32 [-1,1]
    tensor (1, 3, T, H, W)."""
    paths = sorted(glob.glob(os.path.join(REPO, "cyber_6f", dirname, "frame_*.png")))[:n]
    frames = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(im)
    arr = np.stack(frames).astype(np.float32) / 127.5 - 1.0  # T,H,W,3
    return torch.from_numpy(arr).permute(3, 0, 1, 2)[None].contiguous()


@torch.no_grad()
def encode(vae, mean, std, video):
    """video (1,3,T,H,W) in [-1,1] -> normalised latent (1,16,T',h,w).
    argmax of the posterior, as the pipeline does (sample_mode='argmax')."""
    video = video.to(DEV, vae.dtype)
    z = vae.encode(video).latent_dist.mode()
    return ((z.float() - mean) / std)


@torch.no_grad()
def decode(vae, mean, std, z):
    """normalised latent -> video (1,3,T,H,W) in [-1,1], float32 on CPU."""
    raw = (z.float() * std + mean).to(vae.dtype)
    out = vae.decode(raw).sample
    return out.float().clamp(-1, 1).cpu()


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * math.log10(4.0 / max(mse, 1e-12))  # range is [-1,1] -> peak 2


def radial_bands(h, w, n_bands=6):
    """Index map of radial frequency bands over an (h, w) FFT grid,
    band 0 = DC..lowest, band n-1 = Nyquist. Bands are equal in radius."""
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2) / math.sqrt(0.5)  # 0..1
    return torch.clamp((r * n_bands).long(), max=n_bands - 1)


def band_energy(x, bands, n_bands=6):
    """x (..., h, w) -> energy per radial band, summed over leading dims."""
    f = torch.fft.fft2(x.float())
    p = (f.real ** 2 + f.imag ** 2)
    p = p.reshape(-1, *x.shape[-2:])
    out = torch.zeros(n_bands, dtype=torch.float64)
    for b in range(n_bands):
        out[b] = p[:, bands == b].sum().item()
    return out


def chunk_frames(j):
    """Frames (0-based) that latent frame j is made of: 0 -> [0], j -> 4j-3..4j."""
    return [0] if j == 0 else list(range(4 * j - 3, 4 * j + 1))
