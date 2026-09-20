"""E1b - what the bands of a video latent carry, seen through the decoder.

zv = E(V) decodes at ~39 dB. Replace ONE part of it with the same part
of a different-but-related code (the chunk stack of steady-state
per-view codes S, which decodes at ~25 dB) and decode:
  keep band 0 of zv, take bands 1-5 from S   -> what the high bands carry
  keep bands 1-5 of zv, take band 0 from S   -> what band 0 carries
and the same with S's band 0 spatially shifted by half a latent pixel,
as the kind of error a 3D projection makes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch.nn.functional as F

E0 = sys.argv[1]
d = torch.load(os.path.join(E0, "e0_latents.pt"))
zv, zss = d["zv"], d["zss"][0]
vae, mean, std = load_vae()
video = load_frames("circular")
T = video.shape[2]
h, w = zv.shape[-2:]
bands = radial_bands(h, w)
S = zv.clone()
for j in range(1, 21):
    S[0, :, j] = zss[:, 4 * j - 1]

def split(z, keep):
    f = torch.fft.fft2(z.float())
    m = torch.zeros(h, w, dtype=torch.bool)
    for b in keep:
        m |= bands == b
    return torch.fft.ifft2(f * m).real

def shift(z, dx):
    c = z.shape[1]
    out = z.clone()
    for j in range(z.shape[2]):
        theta = torch.tensor([[1, 0, -2 * dx / w], [0, 1, 0]], dtype=torch.float32)[None]
        grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        out[0, :, j] = F.grid_sample(z[0, :, j][None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    return out

def score(z, label):
    dec = decode(vae, mean, std, z.to(DEV))
    ps = [psnr(dec[0, :, t], video[0, :, t]) for t in range(1, T)]
    matched = [4 * j - 1 for j in range(1, 21)]
    print(f"{label:60s} PSNR all {np.mean(ps):.2f}  matched frames {np.mean([psnr(dec[0, :, t], video[0, :, t]) for t in matched]):.2f}")
    return dec

torch.cuda.empty_cache()
score(zv, "zv = E(V)")
score(S, "S = chunk stack of E_ss(V)")
score(split(zv, [0]) + split(S, [1, 2, 3, 4, 5]), "band 0 from zv, bands 1-5 from S")
score(split(S, [0]) + split(zv, [1, 2, 3, 4, 5]), "band 0 from S, bands 1-5 from zv")
score(split(zv, [0, 1]) + split(S, [2, 3, 4, 5]), "bands 0-1 from zv, bands 2-5 from S")
score(split(shift(S, 0.5), [0]) + split(zv, [1, 2, 3, 4, 5]), "band 0 from S shifted 0.5 latent px, bands 1-5 from zv")
score(split(zv, [0]) + split(shift(zv, 0.5), [1, 2, 3, 4, 5]), "band 0 from zv, bands 1-5 from zv shifted 0.5 px")
score(split(shift(zv, 0.5), [0]) + split(zv, [1, 2, 3, 4, 5]), "band 0 from zv shifted 0.5 px, bands 1-5 from zv")
score(split(zv, [0]), "band 0 of zv only (bands 1-5 zeroed)")
