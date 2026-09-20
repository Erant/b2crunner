"""E2c - the latent fits of e2b again, with geometry frozen (rates 1e-12)
or half-free, evaluated on the same held-out split. Reuses e2b's
datasets, init_dc.ply and codes.pt."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from common import *

E2, OUT = sys.argv[1], sys.argv[2]
MODE = sys.argv[3]   # frozen | slow
B2C = os.path.expanduser("~/Projects/b2ctrain/build/b2ctrain")
d = torch.load(os.path.join(OUT, "codes.pt")); zss = d["zss"]
C, T, h, w = zss.shape
SCALE = 8.0
names = [f"frame_{k+1:05d}_.png" for k in range(T)]
train_idx, test_idx = list(range(1, T, 2)), list(range(0, T, 2))
def from_png(im):
    return (torch.from_numpy(im.astype(np.float32) / 255) - 0.5).permute(2, 0, 1) * SCALE
mattes = [cv2.imread(os.path.join(OUT, "lat0", "masks", n), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(OUT, "lat0", "masks", n)) else None for n in names]
Ml = torch.from_numpy(np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) for m in [mm for mm in mattes if mm is not None]]).astype(np.float32) / 255)
zbg = torch.stack([zss[c][:, :, :][torch.cat([Ml] * 0 + [Ml], 0).mean(0, keepdim=True).expand(T, -1, -1) < 0.01].median() for c in range(C)])
L = torch.zeros(C, T, h, w)
geo = (["--lr-mean", "1e-12", "--lr-mean-end", "1e-12", "--lr-scale", "1e-12", "--lr-rotation", "1e-12", "--lr-opac", "1e-12", "--mean-noise-weight", "0"]
       if MODE == "frozen" else ["--lr-mean", "2e-6", "--lr-mean-end", "2e-8", "--lr-scale", "5e-4", "--lr-rotation", "2e-4", "--lr-opac", "1.2e-3", "--mean-noise-weight", "0"])
for i in range(6):
    ch = list(range(3 * i, min(3 * i + 3, C)))
    zb = torch.zeros(3); zb[:len(ch)] = zbg[ch]
    bg = ",".join(f"{v:.4f}" for v in (zb / SCALE + 0.5).clamp(0, 1).tolist())
    out = os.path.join(OUT, f"{MODE}_lat{i}_out")
    r = subprocess.run([B2C, os.path.join(OUT, f"lat{i}"), "--total-train-iters", "6000", "--export-path", out, "--export-name", "s.ply", "--export-every", "6000",
                        "--eval-every", "1000000", "--max-resolution", str(h), "--sh-degree", "0", "--lr-coeffs-dc", "1e-2",
                        "--growth-stop-iter", "0", "--refine-every", "1000000", "--background-color", bg] + geo, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:]); sys.exit(1)
    rd = os.path.join(OUT, f"{MODE}_lat{i}_renders")
    subprocess.run([B2C, "render", "--splat", os.path.join(out, "s.ply"), "--cameras", os.path.join(E2, "cameras_lat.json"), "--output-dir", rd, "--sh-degree", "0"], capture_output=True)
    for k in range(T):
        im = cv2.imread(os.path.join(rd, names[k]), cv2.IMREAD_UNCHANGED)
        alpha = torch.from_numpy(im[:, :, 3].astype(np.float32) / 255)
        L[ch, k] = from_png(im[:, :, 2::-1])[:len(ch)] * alpha + zb[:len(ch), None, None] * (1 - alpha)
bands = radial_bands(h, w)
mask_all = torch.from_numpy(np.stack([cv2.resize(cv2.imread(os.path.join(OUT, "rgb", "masks", n), cv2.IMREAD_GRAYSCALE) if os.path.exists(os.path.join(OUT, "rgb", "masks", n)) else np.zeros((832, 480), np.uint8), (w, h), interpolation=cv2.INTER_AREA) for n in names]).astype(np.float32) / 255)
# masks exist only for training views; rebuild every view's matte from the frames instead
video = load_frames("colmap")
M = ((video[0].abs().amax(0) - 0) > 0).float()  # placeholder, replaced below
fr = ((video[0] + 1) * 127.5)
M = torch.from_numpy(np.stack([cv2.resize(((np.abs(fr[:, t].permute(1, 2, 0).numpy() - 126).max(-1) > 8) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) for t in range(T)]).astype(np.float32) / 255)
M = (M > 0.5).float()
def report(label, a, b, idx):
    a, b = a[:, idx] * M[idx], b[:, idx] * M[idx]
    eb = (band_energy(a - b, bands) / band_energy(b, bands)).sqrt()
    print(f"{label:52s} rel {((a-b).norm()/b.norm()).item():.3f} | bands {' '.join(f'{x:.2f}' for x in eb.tolist())}")
for split, idx in (("TRAIN", train_idx), ("HELD-OUT", test_idx)):
    report(f"{MODE} geometry, {split}: L vs E_ss(V)", L, zss, idx)
def adj(z, idx):
    z = z[:, idx] * M[idx]
    return ((z[:, 1:] - z[:, :-1]).norm() / z[:, 1:].norm()).item()
print(f"adjacent held-out view rel change: E_ss(V) {adj(zss, test_idx):.3f}  L {adj(L, test_idx):.3f}")
torch.save(L, os.path.join(OUT, f"L_{MODE}.pt"))
