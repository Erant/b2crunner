"""E1 - equivariance spectrum of the Wan latent under image-plane motion.

For a frame I and a shift s (px), compare E(shift_s(I)) with the
bilinearly shifted latent shift_{s/8}(E(I)), per radial frequency band
and per channel; rotation and scale likewise. Shifts are made by
cropping a 480x832 window at different offsets out of the 720x1280
source, so there is no border fill. Two codes: E_1 (first-frame path)
and E_ss (steady state of a 33-frame static clip), since E0 showed the
video path's chunk latents live near E_ss, not E_1.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch.nn.functional as F

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
vae, mean, std = load_vae()
W, H = 480, 832
src = cv2.imread(os.path.join(REPO, "cyber_6f/circular/frame_00041_.png"))[:, :, ::-1]
# scale the 720x1280 source so a 480x832 window is the production framing plus margin: 560x996
src = cv2.resize(src, (560, 996), interpolation=cv2.INTER_AREA)
src = torch.from_numpy(src.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)  # 3,H,W

def window(dx=0, dy=0, rot=0.0, scale=1.0):
    """480x832 window from the source, centre-anchored, with a sub-transform."""
    c, h, w = src.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    img = src.permute(1, 2, 0).numpy()
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (h - H) // 2, (w - W) // 2
    out = warped[y0:y0 + H, x0:x0 + W]
    return torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1)[None, :, None]  # 1,3,1,H,W

def E1(img):
    return encode(vae, mean, std, img)[0, :, 0]

def Ess(img, n=33):
    return encode(vae, mean, std, img.expand(-1, -1, n, -1, -1).contiguous())[0, :, -1]

def shift_latent(z, dx, dy):
    """bilinear shift of a (16,h,w) latent by (dx, dy) latent pixels."""
    c, h, w = z.shape
    theta = torch.tensor([[1, 0, -2 * dx / w], [0, 1, -2 * dy / h]], dtype=torch.float32, device=z.device)[None]
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
    return F.grid_sample(z[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]

def bands_for(x):
    return radial_bands(x.shape[-2], x.shape[-1]).cpu()
res = {}
for name, E in [("E1", E1), ("Ess", Ess)]:
    z0 = E(window())
    print(f"== {name}: shift equivariance (rel err of E(shift I) vs shift E(I)), per band lo..hi")
    for s in [1, 2, 3, 4, 6, 8, 16, 24]:
        za = E(window(dx=s))
        zb = shift_latent(z0, s / 8, 0)
        # crop the border column(s) the shift wraps
        m = int(np.ceil(s / 8)) + 1
        a, b = za[:, :, m:-m], zb[:, :, m:-m]
        rel = ((a - b).norm() / a.norm()).item()
        eb = (band_energy((a - b).cpu(), bands_for(a)) / band_energy(a.cpu(), bands_for(a))).sqrt()
        # per-channel rel err
        pc = ((a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)).cpu().numpy()
        # reference: how different is the latent from the same content 8 px away (a whole latent pixel)?
        res[f"{name}_shift_{s}"] = dict(rel=rel, band=eb.tolist(), per_channel=pc.tolist())
        print(f"  shift {s:2d} px ({s/8:.3f} latent px): rel {rel:.3f} | bands {' '.join(f'{x:.2f}' for x in eb.tolist())}"
              f" | channels min/median/max {pc.min():.2f}/{np.median(pc):.2f}/{pc.max():.2f}")
    # a pure integer-latent shift of 8 px should be a roll if the encoder were shift-equivariant at stride
    za = E(window(dx=8)); zb = torch.roll(z0, -1, dims=2)
    a, b = za[:, :, 2:-2], zb[:, :, 2:-2]
    print(f"  8 px vs integer roll: rel {((a-b).norm()/a.norm()).item():.3f}")
    print(f"  content change reference: rel err between this frame and the next frame of the orbit (E1):",
          f"{((E1(window()) - encode(vae, mean, std, load_frames('circular', W, H, 42)[:, :, 41:42])[0, :, 0]).norm() / z0.norm()).item():.3f}")
    print(f"== {name}: rotation / scale (rel err vs the correspondingly warped latent)")
    for rot in [1.0, 2.0, 5.0]:
        za = E(window(rot=rot))
        c, h, w = z0.shape
        th = np.deg2rad(rot)
        theta = torch.tensor([[np.cos(th), -np.sin(th) * h / w, 0], [np.sin(th) * w / h, np.cos(th), 0]], dtype=torch.float32, device=z0.device)[None]
        grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        zb = F.grid_sample(z0[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        a, b = za[:, 6:-6, 6:-6], zb[:, 6:-6, 6:-6]
        eb = (band_energy((a - b).cpu(), bands_for(a)) / band_energy(a.cpu(), bands_for(a))).sqrt()
        print(f"  rot {rot:.0f} deg: rel {((a-b).norm()/a.norm()).item():.3f} | bands {' '.join(f'{x:.2f}' for x in eb.tolist())}")
        res[f"{name}_rot_{rot}"] = dict(rel=((a - b).norm() / a.norm()).item(), band=eb.tolist())
    for sc in [0.97, 1.03, 1.1]:
        za = E(window(scale=sc))
        c, h, w = z0.shape
        theta = torch.tensor([[1 / sc, 0, 0], [0, 1 / sc, 0]], dtype=torch.float32, device=z0.device)[None]
        grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        zb = F.grid_sample(z0[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        a, b = za[:, 6:-6, 6:-6], zb[:, 6:-6, 6:-6]
        eb = (band_energy((a - b).cpu(), bands_for(a)) / band_energy(a.cpu(), bands_for(a))).sqrt()
        print(f"  scale {sc:.2f}: rel {((a-b).norm()/a.norm()).item():.3f} | bands {' '.join(f'{x:.2f}' for x in eb.tolist())}")
        res[f"{name}_scale_{sc}"] = dict(rel=((a - b).norm() / a.norm()).item(), band=eb.tolist())
    # decoder-side check: does a bilinearly shifted latent decode to a shifted image?
    dec = decode(vae, mean, std, shift_latent(z0, 0.5, 0)[None, :, None])[0, :, 0]
    ref = window(dx=4)[0, :, 0]
    ref0 = decode(vae, mean, std, z0[None, :, None])[0, :, 0]
    print(f"  D(shift_0.5 E(I)) vs I shifted 4 px: PSNR {psnr(dec[:, :, 16:-16], ref[:, :, 16:-16]):.2f}  (D(E(I)) vs I: {psnr(ref0, window()[0,:,0]):.2f})")
json.dump(res, open(os.path.join(OUT, "e1.json"), "w"))
