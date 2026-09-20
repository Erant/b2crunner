"""E0b - how expensive is the steady-state code, and can E_1 be mapped to it?

(a) convergence: rel err of static-clip latent j against the converged
    latent (j=20), for one frame — how many static chunks E_ss needs.
(b) linear map: fit z_ss ~ A z_1 + b (16x16 + 16, per latent pixel) on
    frames 0..60, test on 61..80 — if the steady-state code were an
    affine function of the first-frame code it would come for free.
(c) the same with a 3x3-conv map (small linear conv), to see whether a
    local linear operator explains it.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *

E0 = sys.argv[1]
d = torch.load(os.path.join(E0, "e0_latents.pt"))
z1, zss, zv = d["z1"][0], d["zss"][0], d["zv"][0]   # 16,81,h,w ; 16,21,h,w
vae, mean, std = load_vae()
video = load_frames("circular")

# (a)
k0 = 40
static = video[:, :, k0:k0 + 1].expand(-1, -1, 81, -1, -1).contiguous()
zs = encode(vae, mean, std, static)[0].cpu()
conv = zs[:, -1]
print("(a) static clip: rel err of latent j vs converged (j=20):",
      " ".join(f"{((zs[:, j] - conv).norm() / conv.norm()).item():.3f}" for j in range(21)))
del static

# (b) affine per-pixel map
X = z1[:, :61].permute(1, 2, 3, 0).reshape(-1, 16)      # N,16
Y = zss[:, :61].permute(1, 2, 3, 0).reshape(-1, 16)
Xa = torch.cat([X, torch.ones(len(X), 1)], 1)
A = torch.linalg.lstsq(Xa, Y).solution                    # 17,16
Xt = z1[:, 61:].permute(1, 2, 3, 0).reshape(-1, 16)
Yt = zss[:, 61:].permute(1, 2, 3, 0).reshape(-1, 16)
pred = torch.cat([Xt, torch.ones(len(Xt), 1)], 1) @ A
print(f"(b) affine 16x16 map E_1 -> E_ss: test rel err {((pred - Yt).norm() / Yt.norm()).item():.3f} "
      f"(baseline: E_1 itself {((Xt - Yt).norm() / Yt.norm()).item():.3f}; per-channel offset only "
      f"{((Xt - Yt - (Y - X).mean(0)).norm() / Yt.norm()).item():.3f})")
predf = pred.reshape(20, *z1.shape[2:], 16).permute(3, 0, 1, 2)
bands = radial_bands(*z1.shape[2:])
eb = (band_energy(predf - zss[:, 61:], bands) / band_energy(zss[:, 61:], bands)).sqrt()
print("    per band:", " ".join(f"{x:.2f}" for x in eb.tolist()))

# (c) 3x3 linear conv map, fitted by least squares on unfolded patches
import torch.nn.functional as F
def unfold(z):  # 16,T,h,w -> N, 16*9
    T = z.shape[1]
    p = F.unfold(z.permute(1, 0, 2, 3), kernel_size=3, padding=1)  # T, 144, h*w
    return p.permute(0, 2, 1).reshape(-1, 144)
Xc = torch.cat([unfold(z1[:, :61]), torch.ones(61 * z1.shape[2] * z1.shape[3], 1)], 1)
Ac = torch.linalg.lstsq(Xc, Y).solution
Xct = torch.cat([unfold(z1[:, 61:]), torch.ones(20 * z1.shape[2] * z1.shape[3], 1)], 1)
predc = Xct @ Ac
print(f"(c) 3x3 linear conv map: test rel err {((predc - Yt).norm() / Yt.norm()).item():.3f}")
predcf = predc.reshape(20, *z1.shape[2:], 16).permute(3, 0, 1, 2)
eb = (band_energy(predcf - zss[:, 61:], bands) / band_energy(zss[:, 61:], bands)).sqrt()
print("    per band:", " ".join(f"{x:.2f}" for x in eb.tolist()))

# (d) and the reverse question for the blend: how far is the VIDEO latent from
# E_ss of the chunk's 3rd frame, per band, vs how far adjacent chunk latents are
# from each other per band (the model's own temporal change).
tgt = torch.stack([zv[:, j] for j in range(1, 21)], 1)
ss3 = torch.stack([zss[:, 4 * j - 1] for j in range(1, 21)], 1)
prev = torch.stack([zv[:, j - 1] for j in range(1, 21)], 1)
print("(d) per band: |zv[j]-E_ss(frame 4j-1)| / |zv[j]| :", " ".join(f"{x:.2f}" for x in (band_energy(tgt - ss3, bands) / band_energy(tgt, bands)).sqrt().tolist()))
print("    per band: |zv[j]-zv[j-1]| / |zv[j]|          :", " ".join(f"{x:.2f}" for x in (band_energy(tgt - prev, bands) / band_energy(tgt, bands)).sqrt().tolist()))
print("    energy fraction of zv per band                :", " ".join(f"{x:.3f}" for x in (band_energy(tgt, bands) / band_energy(tgt, bands).sum()).tolist()))
