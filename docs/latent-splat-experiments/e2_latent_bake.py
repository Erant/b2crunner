"""E2 - the latent bake: Gaussians carrying Wan latents vs the pixel path.

Geometry comes from an RGB splat trained by b2ctrain on the 81 frames
(stage 0, done by the shell driver). Then:

  pixel path   V_c = render(RGB splat) at the 81 cameras
               -> E(V_c) video code vs E(V); E_ss(V_c[k]) vs E_ss(V[k])
  latent path  six colour-only b2ctrain fits (16 latent channels as
               5x3 + 1 "RGB" images at 60x104, geometry frozen, SH 0)
               on E_ss(V[k]) -> rendered latent L[k] vs E_ss(V[k])

both per radial band, and both decoded through the video decoder as a
chunk stack (position 4j-1) against the frames.
"""
import sys, os, json, time, subprocess, struct
sys.path.insert(0, os.path.dirname(__file__))
from common import *

E0, E2 = sys.argv[1], sys.argv[2]
B2C = os.path.expanduser("~/Projects/b2ctrain/build/b2ctrain")
d = torch.load(os.path.join(E0, "e0_latents.pt"))
zss, zv = d["zss"][0], d["zv"][0]        # 16,81,h,w ; 16,21,h,w
C, T, h, w = zss.shape
SCALE = 8.0   # latent -> png: z/SCALE + 0.5

def to_png(z3):   # 3,h,w normalised latent -> uint8 HxWx3 (RGB)
    return ((z3 / SCALE + 0.5).clamp(0, 1) * 255).round().permute(1, 2, 0).numpy().astype(np.uint8)

def from_png(im):  # uint8 HxWx3 RGB -> 3,h,w
    return (torch.from_numpy(im.astype(np.float32) / 255) - 0.5).permute(2, 0, 1) * SCALE

# ---- DC-only init.ply from the RGB splat --------------------------------
def dc_only_ply(src, dst):
    with open(src, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        body = f.read()
    props = [l.split()[2] for l in header.decode().splitlines() if l.startswith("property")]
    n = int([l for l in header.decode().splitlines() if l.startswith("element vertex")][0].split()[2])
    arr = np.frombuffer(body, dtype=np.float32).reshape(n, len(props))
    keep = [i for i, p in enumerate(props) if not p.startswith("f_rest") and not p.startswith("ev_") and p not in ("seg_label", "seg_conf")]
    out = arr[:, keep].copy()
    for i, p in enumerate([props[i] for i in keep]):
        if p.startswith("f_dc"):
            out[:, i] = 0.0
    hdr = "ply\nformat binary_little_endian 1.0\ncomment SH degree: 0\nelement vertex %d\n" % n
    hdr += "".join("property float %s\n" % props[i] for i in keep) + "end_header\n"
    with open(dst, "wb") as f:
        f.write(hdr.encode()); f.write(out.astype(np.float32).tobytes())
    return n

# ---- stage 1: latent datasets + colour-only fits --------------------------
names = sorted(os.listdir(os.path.join(E2, "rgb", "images")))
assert len(names) == T
for i in range(6):
    ds = os.path.join(E2, f"lat{i}")
    if os.path.exists(os.path.join(E2, f"lat{i}_out", "lat.ply")):
        continue
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    ch = list(range(3 * i, min(3 * i + 3, C)))
    for k, name in enumerate(names):
        z3 = torch.zeros(3, h, w)
        z3[:len(ch)] = zss[ch, k]
        cv2.imwrite(os.path.join(ds, "images", name), to_png(z3)[:, :, ::-1])
    f0 = 1213.916992
    open(os.path.join(ds, "cameras.txt"), "w").write(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n1 PINHOLE %d %d %.6f %.6f %d %d\n" % (w, h, f0 * w / 720, f0 * h / 1280, w // 2, h // 2))
    for fn in ("images.txt", "points3D.txt"):
        os.system(f"cp {os.path.join(E2, 'rgb', fn)} {ds}/")
    dc_only_ply(os.path.join(E2, "rgb_out", "rgb.ply"), os.path.join(ds, "init.ply"))
    t0 = time.time()
    cmd = [B2C, ds, "--total-train-iters", "3000", "--export-path", os.path.join(E2, f"lat{i}_out"), "--export-name", "lat.ply",
           "--export-every", "3000", "--max-resolution", str(h), "--eval-every", "1000000", "--sh-degree", "0",
           # NOT 0: a zero rate turns the schedule's log into NaN positions (84 % NaN, measured); 1e-12 freezes them
           "--lr-mean", "1e-12", "--lr-mean-end", "1e-12", "--lr-scale", "1e-12", "--lr-rotation", "1e-12", "--lr-opac", "1e-12",
           "--mean-noise-weight", "0", "--growth-stop-iter", "0", "--refine-every", "1000000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"lat{i}: {time.time()-t0:.0f}s", r.stdout.strip().splitlines()[-2:], r.stderr.strip().splitlines()[-2:] if r.returncode else "")
    if r.returncode:
        sys.exit(r.returncode)
    r = subprocess.run([B2C, "render", "--splat", os.path.join(E2, f"lat{i}_out", "lat.ply"), "--cameras", os.path.join(E2, "cameras_lat.json"),
                        "--output-dir", os.path.join(E2, f"lat{i}_renders"), "--sh-degree", "0"], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr); sys.exit(1)

# ---- stage 2: assemble the rendered latent, and the pixel path ------------
L = torch.zeros(C, T, h, w)
for i in range(6):
    ch = list(range(3 * i, min(3 * i + 3, C)))
    for k, name in enumerate(names):
        im = cv2.imread(os.path.join(E2, f"lat{i}_renders", name), cv2.IMREAD_UNCHANGED)
        if im.shape[-1] == 4:
            im = im[:, :, :3]
        L[ch, k] = from_png(im[:, :, ::-1])[:len(ch)]
# quantisation floor: zss through the png round trip
Q = torch.zeros_like(zss)
for i in range(6):
    ch = list(range(3 * i, min(3 * i + 3, C)))
    for k in range(T):
        z3 = torch.zeros(3, h, w); z3[:len(ch)] = zss[ch, k]
        Q[ch, k] = from_png(to_png(z3))[:len(ch)]

vae, mean, std = load_vae()
video = load_frames("circular")
Vc = []
for name in names:
    im = cv2.imread(os.path.join(E2, "rgb_renders", name), cv2.IMREAD_UNCHANGED)[:, :, :3][:, :, ::-1]
    Vc.append(im)
Vc = torch.from_numpy(np.stack(Vc).astype(np.float32) / 127.5 - 1).permute(3, 0, 1, 2)[None].contiguous()
print(f"RGB splat render vs frames: PSNR {np.mean([psnr(Vc[0, :, t], video[0, :, t]) for t in range(T)]):.2f}")
zvc = encode(vae, mean, std, Vc)[0].cpu()
p = os.path.join(E2, "zss_render.pt")
if os.path.exists(p):
    zss_c = torch.load(p)
    if zss_c.dim() == 3:   # the first run's mis-stacked (16*81, h, w)
        zss_c = zss_c.reshape(T, C, h, w).permute(1, 0, 2, 3).contiguous()
else:
    t0 = time.time()
    zss_c = torch.stack([encode(vae, mean, std, Vc[:, :, k:k + 1].expand(-1, -1, 49, -1, -1).contiguous())[0, :, -1] for k in range(T)], 1).cpu()
    torch.save(zss_c, p); print(f"E_ss(render) x81 {time.time()-t0:.0f}s")

bands = radial_bands(h, w)
def report(label, a, b):
    """a vs b, both (16,T,h,w): rel err overall and per band, on all frames."""
    eb = (band_energy(a - b, bands) / band_energy(b, bands)).sqrt()
    print(f"{label:58s} rel {((a-b).norm()/b.norm()).item():.3f} | bands {' '.join(f'{x:.2f}' for x in eb.tolist())}")
print("bands = radial sixths of the latent spectrum, lo..hi; energy share of E_ss:",
      " ".join(f"{x:.3f}" for x in (band_energy(zss, bands) / band_energy(zss, bands).sum()).tolist()))
report("png quantisation floor: Q(E_ss(V)) vs E_ss(V)", Q, zss)
report("LATENT path: L (16-ch Gaussians) vs E_ss(V)", L, zss)
report("PIXEL path: E_ss(render) vs E_ss(V)", zss_c, zss)
report("PIXEL path: E(render) video code vs E(V)", zvc, zv)
report("LATENT path vs PIXEL path per-view codes: L vs E_ss(render)", L, zss_c)
# how much of the model-facing latent (band 0) does each path keep?
def lowpass(z, keep=1):
    f = torch.fft.fft2(z.float()); f[..., bands > keep - 1] = 0
    return torch.fft.ifft2(f).real
report("LATENT path, band 0 only: L vs E_ss(V)", lowpass(L), lowpass(zss))
report("PIXEL path, band 0 only: E_ss(render) vs E_ss(V)", lowpass(zss_c), lowpass(zss))
report("LATENT path, bands 0-1: L vs E_ss(V)", lowpass(L, 2), lowpass(zss, 2))
report("PIXEL path, bands 0-1: E_ss(render) vs E_ss(V)", lowpass(zss_c, 2), lowpass(zss, 2))
# view-consistency of the latent path itself: rendered latents should be
# smoother across views than the codes they were fitted on
def adj(z):
    return ((z[:, 1:] - z[:, :-1]).norm() / z[:, 1:].norm()).item()
print(f"adjacent-view rel change: E_ss(V) {adj(zss):.3f}  L {adj(L):.3f}  E_ss(render) {adj(zss_c):.3f}")

# ---- stage 3: decode as a chunk stack -------------------------------------
def stack(zk):   # per-view codes (16,T,h,w) -> video latent (1,16,21,h,w) with the chunk's 3rd frame
    out = zv.clone()[None]
    for j in range(1, 21):
        out[0, :, j] = zk[:, 4 * j - 1]
    return out
torch.cuda.empty_cache()
pos = np.array([(t - 1) % 4 for t in range(1, T)])
for label, zk in [("E_ss(V)", zss), ("L (latent Gaussians)", L), ("E_ss(render)", zss_c)]:
    dec = decode(vae, mean, std, stack(zk).to(DEV))
    ps = np.array([psnr(dec[0, :, t], video[0, :, t]) for t in range(1, T)])
    print(f"decode chunk stack of {label:22s}: PSNR mean {ps.mean():.2f}; by position {np.round([ps[pos == q].mean() for q in range(4)], 2).tolist()}")
    if label.startswith("L"):
        strip = torch.cat([torch.cat([video[0, :, t], Vc[0, :, t], dec[0, :, t]], dim=1) for t in (43, 44)], dim=2)
        cv2.imwrite(os.path.join(E2, "e2_strip.jpg"), ((strip.permute(1, 2, 0).numpy()[:, :, ::-1] + 1) * 127.5).astype(np.uint8))
dec = decode(vae, mean, std, zvc[None].to(DEV))
print(f"decode E(render) video code           : PSNR vs frames {np.mean([psnr(dec[0, :, t], video[0, :, t]) for t in range(T)]):.2f}, vs render {np.mean([psnr(dec[0, :, t], Vc[0, :, t]) for t in range(T)]):.2f}")
torch.save(dict(L=L, zss_c=zss_c, zvc=zvc), os.path.join(E2, "e2_latents.pt"))
