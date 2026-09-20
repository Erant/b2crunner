"""E2b - the latent bake with geometry free and every other view held out.

Same two paths as e2_latent_bake.py, but the RGB splat and the six
latent fits are trained on the odd views only, and every number is
reported on the even (held-out) views. The latent fits start from the
RGB splat's geometry and are free to move it (no growth), which E2
showed is what a Gaussian field needs to represent latent codes at all.
"""
import sys, os, json, time, subprocess, shutil
sys.path.insert(0, os.path.dirname(__file__))
from common import *

E0, E2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FRAMES = "colmap"   # the set cyber_6f/colmap's cameras belong to (the upscaled final frames); `circular` is numbered differently
B2C = os.path.expanduser("~/Projects/b2ctrain/build/b2ctrain")
os.makedirs(OUT, exist_ok=True)
vae, mean, std = load_vae()
video = load_frames(FRAMES)
p = os.path.join(OUT, "codes.pt")
if os.path.exists(p):
    d = torch.load(p); zss, zv = d["zss"], d["zv"]
else:
    T = video.shape[2]
    zv = encode(vae, mean, std, video)[0].cpu()
    zss = torch.stack([encode(vae, mean, std, video[:, :, k:k + 1].expand(-1, -1, 21, -1, -1).contiguous())[0, :, -1] for k in range(T)], 1).cpu()
    torch.save(dict(zss=zss, zv=zv), p)
C, T, h, w = zss.shape
SCALE = 8.0
names = [f"frame_{k+1:05d}_.png" for k in range(T)]
train_idx = list(range(1, T, 2))
test_idx = list(range(0, T, 2))
f0 = 1213.916992

def to_png(z3):
    return ((z3 / SCALE + 0.5).clamp(0, 1) * 255).round().permute(1, 2, 0).numpy().astype(np.uint8)
def from_png(im):
    return (torch.from_numpy(im.astype(np.float32) / 255) - 0.5).permute(2, 0, 1) * SCALE

def write_dataset(ds, W, H, images, init_ply=None):
    """COLMAP text dataset with only the training views' images; RGBA in -> images/ + masks/ (brush's masked view)."""
    os.makedirs(os.path.join(ds, "images"), exist_ok=True); os.makedirs(os.path.join(ds, "masks"), exist_ok=True)
    for k in train_idx:
        cv2.imwrite(os.path.join(ds, "images", names[k]), images[k][:, :, :3])
        cv2.imwrite(os.path.join(ds, "masks", names[k]), images[k][:, :, 3])
    open(os.path.join(ds, "cameras.txt"), "w").write(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n1 PINHOLE %d %d %.6f %.6f %d %d\n" % (W, H, f0 * W / 720, f0 * H / 1280, W // 2, H // 2))
    keep = {names[k] for k in train_idx}
    with open(os.path.join(ds, "images.txt"), "w") as f:
        for line in open(os.path.join(E2, "rgb", "images.txt")):
            t = line.split()
            if line.startswith("#") or (len(t) >= 10 and t[9] in keep):
                f.write(line); f.write("\n")   # empty POINTS2D line
    shutil.copy(os.path.join(E2, "rgb", "points3D.txt"), ds)
    if init_ply:
        shutil.copy(init_ply, os.path.join(ds, "init.ply"))

def train(ds, out, iters, extra):
    cmd = [B2C, ds, "--total-train-iters", str(iters), "--export-path", out, "--export-name", "s.ply", "--export-every", str(iters),
           "--eval-every", "1000000"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
    print("  ", [l for l in r.stdout.splitlines() if "Training took" in l])
    return os.path.join(out, "s.ply")

def render(ply, cams, out, sh=None):
    cmd = [B2C, "render", "--splat", ply, "--cameras", cams, "--output-dir", out] + (["--sh-degree", str(sh)] if sh is not None else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr); sys.exit(1)
    return out

# ---- RGB splat on the odd views ------------------------------------------
def matte(im):
    """The pass-1 frames sit on a flat 126-grey; anything 8 levels off it is subject, closed and feathered."""
    d = np.abs(im.astype(np.float32) - 126).max(-1)
    m = (d > 8).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.GaussianBlur(m, (5, 5), 0)
    return np.dstack([im, m])
rgb_imgs = [matte(cv2.resize(cv2.imread(os.path.join(REPO, "cyber_6f", FRAMES, n)), (480, 832), interpolation=cv2.INTER_AREA)) for n in names]
mattes = [im[:, :, 3] for im in rgb_imgs]
write_dataset(os.path.join(OUT, "rgb"), 480, 832, rgb_imgs)
rgb_ply = train(os.path.join(OUT, "rgb"), os.path.join(OUT, "rgb_out"), 15000, ["--max-resolution", "832", "--max-splats", "1000000", "--background-color", "0.494,0.494,0.494"])
render(rgb_ply, os.path.join(E2, "cameras_480.json"), os.path.join(OUT, "rgb_renders"))

# ---- latent fits on the odd views, free geometry, from the RGB splat -------
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
dc_only_ply(rgb_ply, os.path.join(OUT, "init_dc.ply"))
Ml = torch.from_numpy(np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) for m in mattes]).astype(np.float32) / 255)  # T,h,w
zbg = torch.stack([zss[c][Ml < 0.01].median() for c in range(C)])   # 16: the flat grey's steady-state code
print("background latent code:", np.round(zbg.numpy(), 2).tolist())
L = torch.zeros(C, T, h, w)
for i in range(6):
    ch = list(range(3 * i, min(3 * i + 3, C)))
    imgs = []
    for k in range(T):
        z3 = torch.zeros(3, h, w); z3[:len(ch)] = zss[ch, k]
        m = cv2.resize(mattes[k], (w, h), interpolation=cv2.INTER_AREA)
        imgs.append(np.dstack([to_png(z3)[:, :, ::-1], m]))
    ds = os.path.join(OUT, f"lat{i}")
    write_dataset(ds, w, h, imgs, os.path.join(OUT, "init_dc.ply"))
    zb = torch.zeros(3); zb[:len(ch)] = zbg[ch]
    bg = ",".join(f"{v:.4f}" for v in (zb / SCALE + 0.5).clamp(0, 1).tolist())
    ply = train(ds, os.path.join(OUT, f"lat{i}_out"), 6000, ["--max-resolution", str(h), "--sh-degree", "0", "--lr-coeffs-dc", "1e-2",
                                                             "--growth-stop-iter", "0", "--refine-every", "1000000", "--background-color", bg])
    rd = render(ply, os.path.join(E2, "cameras_lat.json"), os.path.join(OUT, f"lat{i}_renders"), sh=0)
    for k in range(T):
        im = cv2.imread(os.path.join(rd, names[k]), cv2.IMREAD_UNCHANGED)
        alpha = torch.from_numpy(im[:, :, 3].astype(np.float32) / 255)
        L[ch, k] = (from_png(im[:, :, 2::-1])[:len(ch)] * alpha + zb[:len(ch), None, None] * (1 - alpha))

# ---- pixel path codes -------------------------------------------------------
def over_grey(im):
    a = im[:, :, 3:4].astype(np.float32) / 255
    return im[:, :, 2::-1].astype(np.float32) * a + 126.0 * (1 - a)
Vc = torch.from_numpy(np.stack([over_grey(cv2.imread(os.path.join(OUT, "rgb_renders", n), cv2.IMREAD_UNCHANGED)) for n in names]) / 127.5 - 1).permute(3, 0, 1, 2)[None].contiguous().float()
Mt = torch.from_numpy(np.stack(mattes).astype(np.float32) / 255)
ps = [10 * math.log10(4.0 / ((((Vc[0, :, t] - video[0, :, t]) ** 2) * Mt[t]).sum() / (3 * Mt[t].sum())).item()) for t in range(T)]
print(f"RGB splat render vs frames: subject-only PSNR train {np.mean([ps[k] for k in train_idx]):.2f} / held-out {np.mean([ps[k] for k in test_idx]):.2f}")
zvc = encode(vae, mean, std, Vc)[0].cpu()
t0 = time.time()
zss_c = torch.stack([encode(vae, mean, std, Vc[:, :, k:k + 1].expand(-1, -1, 21, -1, -1).contiguous())[0, :, -1] for k in range(T)], 1).cpu()
print(f"E_ss(render) x81 (21-frame clips) {time.time()-t0:.0f}s")
torch.save(dict(L=L, zss_c=zss_c, zvc=zvc), os.path.join(OUT, "e2b_latents.pt"))

bands = radial_bands(h, w)
M = torch.from_numpy(np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) for m in mattes]).astype(np.float32) / 255)  # T,h,w
M = (M > 0.5).float()
def report(label, a, b, idx):
    a, b = a[:, idx] * M[idx], b[:, idx] * M[idx]
    eb = (band_energy(a - b, bands) / band_energy(b, bands)).sqrt()
    print(f"{label:52s} rel {((a-b).norm()/b.norm()).item():.3f} | bands {' '.join(f'{x:.2f}' for x in eb.tolist())}")
for split, idx in (("TRAIN", train_idx), ("HELD-OUT", test_idx)):
    print(f"== {split} views")
    report("LATENT path: L vs E_ss(V)", L, zss, idx)
    report("PIXEL path: E_ss(render) vs E_ss(V)", zss_c, zss, idx)
    report("L vs E_ss(render)", L, zss_c, idx)
def adj(z, idx):
    z = z[:, idx]
    return ((z[:, 1:] - z[:, :-1]).norm() / z[:, 1:].norm()).item()
print(f"adjacent held-out view rel change: E_ss(V) {adj(zss, test_idx):.3f}  L {adj(L, test_idx):.3f}  E_ss(render) {adj(zss_c, test_idx):.3f}")

# decode as chunk stacks (position 4j-1 = frames 3,7,11,... all even -> held out)
def stack(zk):
    out = zv.clone()[None]
    for j in range(1, 21):
        out[0, :, j] = zk[:, 4 * j - 1]
    return out
torch.cuda.empty_cache()
matched = [4 * j - 1 for j in range(1, 21)]
for label, zk in [("E_ss(V)", zss), ("L (latent Gaussians)", L), ("E_ss(render)", zss_c)]:
    dec = decode(vae, mean, std, stack(zk).to(DEV))
    Mt = torch.from_numpy(np.stack(mattes).astype(np.float32) / 255)[None]  # 1,T,H,W
    ps = [10 * math.log10(4.0 / ((((dec[0, :, t] - video[0, :, t]) ** 2) * Mt[0, t]).sum() / (3 * Mt[0, t].sum())).item()) for t in matched]
    print(f"decode chunk stack of {label:22s}: subject-only PSNR at the matched (held-out) frames {np.mean(ps):.2f}")
    if label.startswith("L"):
        strip = torch.cat([torch.cat([video[0, :, t], Vc[0, :, t], dec[0, :, t]], dim=1) for t in (43, 47)], dim=2)
        cv2.imwrite(os.path.join(OUT, "e2b_strip.jpg"), ((strip.permute(1, 2, 0).numpy()[:, :, ::-1] + 1) * 127.5).astype(np.uint8))
dec = decode(vae, mean, std, zvc[None].to(DEV))
print(f"decode E(render) video code: PSNR vs frames held-out {np.mean([psnr(dec[0, :, t], video[0, :, t]) for t in test_idx]):.2f}")
