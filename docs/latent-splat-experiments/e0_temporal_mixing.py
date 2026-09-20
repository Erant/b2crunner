"""E0 - temporal mixing: is a Wan latent frame a view?

z_video = E(81 frames); z_1[k] = E(frame k alone) through the causal
VAE's first-frame path. For every latent frame j >= 1 (frames 4j-3..4j)
measure how far z_video[j] is from each z_1 of its chunk, against the
motion scale (how far apart the chunk's own frames are in z_1) and
against the latent's own norm, overall and per radial frequency band.
Then decode a latent stack in which every video latent frame is
replaced by the best single-frame latent of its chunk, and score it
against the frames; and encode a STATIC clip (81 copies of one frame)
to separate the causal-state term from the motion term.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
vae, mean, std = load_vae()
video = load_frames("circular")
T = video.shape[2]
print("frames", tuple(video.shape))

t0 = time.time()
zv = encode(vae, mean, std, video)                         # 1,16,21,h,w
print("E(video)", tuple(zv.shape), f"{time.time()-t0:.1f}s")
t0 = time.time()
z1 = torch.cat([encode(vae, mean, std, video[:, :, k:k + 1]) for k in range(T)], dim=2)  # 1,16,81,h,w
print("E(frame) x81", tuple(z1.shape), f"{time.time()-t0:.1f}s")
h, w = zv.shape[-2:]
bands = radial_bands(h, w).to("cpu")
torch.save(dict(zv=zv.cpu(), z1=z1.cpu()), os.path.join(OUT, "e0_latents.pt"))

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

rows = []
for j in range(1, zv.shape[2]):
    fr = chunk_frames(j)
    target = zv[0, :, j]
    d = [rel(z1[0, :, k], target) for k in fr]
    motion = rel(z1[0, :, fr[-1]], z1[0, :, fr[0]])       # first vs last frame of the chunk
    prev = rel(zv[0, :, j - 1], target)                   # previous video latent (temporal smoothness)
    # best single-frame latent per band
    resid_best = z1[0, :, fr[int(np.argmin(d))]] - target
    eb = band_energy(resid_best.cpu(), bands) / band_energy(target.cpu(), bands)
    rows.append(dict(j=j, frames=fr, rel_to_each=d, best=int(np.argmin(d)), motion=motion,
                     prev_latent=prev, band_rel=eb.sqrt().tolist()))
    print(f"j={j:2d} frames {fr[0]:2d}-{fr[-1]:2d}  rel err vs each frame's own latent: "
          f"{' '.join(f'{x:.3f}' for x in d)}  | first-vs-last frame {motion:.3f} | prev latent {prev:.3f} "
          f"| per band {' '.join(f'{x:.2f}' for x in eb.sqrt().tolist())}")

# summary
best_idx = np.array([r["best"] for r in rows])
print("which frame of the chunk best explains the latent (0=first .. 3=last):",
      np.bincount(best_idx, minlength=4).tolist())
print("mean rel err best/first/last: %.3f / %.3f / %.3f ; mean motion %.3f ; mean prev-latent %.3f" % (
    np.mean([min(r["rel_to_each"]) for r in rows]), np.mean([r["rel_to_each"][0] for r in rows]),
    np.mean([r["rel_to_each"][-1] for r in rows]), np.mean([r["motion"] for r in rows]),
    np.mean([r["prev_latent"] for r in rows])))
print("mean per-band rel err of the best frame:", np.round(np.mean([r["band_rel"] for r in rows], 0), 3).tolist())

# Static clip: 81 copies of frame 40 -> does latent j equal E(frame alone)?
k0 = 40
static = video[:, :, k0:k0 + 1].expand(-1, -1, T, -1, -1).contiguous()
zs = encode(vae, mean, std, static)
d_static = [rel(zs[0, :, j], z1[0, :, k0]) for j in range(zs.shape[2])]
print("STATIC clip: rel err of latent j vs E(frame alone):", " ".join(f"{x:.3f}" for x in d_static))
d_static_prev = [rel(zs[0, :, j], zs[0, :, j - 1]) for j in range(1, zs.shape[2])]
print("STATIC clip: rel err latent j vs j-1:", " ".join(f"{x:.3f}" for x in d_static_prev))
print("STATIC clip: cos(latent j, E(frame alone)):", " ".join(f"{cos(zs[0, :, j], z1[0, :, k0]):.3f}" for j in range(zs.shape[2])))
print("STATIC clip: norm ratio |latent j| / |E(frame alone)|:", " ".join(f"{(zs[0, :, j].norm() / z1[0, :, k0].norm()).item():.3f}" for j in range(zs.shape[2])))
print("STATIC clip: per-channel mean of steady state vs E(frame alone):")
print("  steady:", np.round(zs[0, :, -1].mean((1, 2)).cpu().numpy(), 2).tolist())
print("  single:", np.round(z1[0, :, k0].mean((1, 2)).cpu().numpy(), 2).tolist())

# STEADY-STATE HYPOTHESIS: is the video latent of chunk j close to the steady-state
# code of its frames?  E_ss(I) = last latent of a 49-frame static clip of I.
t0 = time.time()
zss = []
for k in range(T):
    clip = video[:, :, k:k + 1].expand(-1, -1, 49, -1, -1).contiguous()
    zss.append(encode(vae, mean, std, clip)[:, :, -1:])
zss = torch.cat(zss, dim=2)   # 1,16,81,h,w
print(f"E_ss(frame) x81 {time.time()-t0:.1f}s")
torch.save(dict(zv=zv.cpu(), z1=z1.cpu(), zss=zss.cpu()), os.path.join(OUT, "e0_latents.pt"))
rows_ss = []
for j in range(1, zv.shape[2]):
    fr = chunk_frames(j)
    target = zv[0, :, j]
    d = [rel(zss[0, :, k], target) for k in fr]
    c = [cos(zss[0, :, k], target) for k in fr]
    motion = rel(zss[0, :, fr[-1]], zss[0, :, fr[0]])
    resid_best = zss[0, :, fr[int(np.argmin(d))]] - target
    eb = (band_energy(resid_best.cpu(), bands) / band_energy(target.cpu(), bands)).sqrt()
    rows_ss.append(dict(j=j, rel=d, cos=c, motion=motion, band_rel=eb.tolist(), best=int(np.argmin(d))))
    print(f"j={j:2d} STEADY rel err vs each frame: {' '.join(f'{x:.3f}' for x in d)} cos {' '.join(f'{x:.3f}' for x in c)}"
          f" | first-vs-last {motion:.3f} | per band {' '.join(f'{x:.2f}' for x in eb.tolist())}")
print("STEADY best-frame histogram:", np.bincount([r["best"] for r in rows_ss], minlength=4).tolist())
print("STEADY mean rel err best/first/last: %.3f / %.3f / %.3f ; mean motion(ss) %.3f" % (
    np.mean([min(r["rel"]) for r in rows_ss]), np.mean([r["rel"][0] for r in rows_ss]),
    np.mean([r["rel"][-1] for r in rows_ss]), np.mean([r["motion"] for r in rows_ss])))
print("STEADY mean per-band rel err (best frame):", np.round(np.mean([r["band_rel"] for r in rows_ss], 0), 3).tolist())
# and the video latent's own temporal structure: how much of zv[j] is explained by zv[j-1]?
print("VIDEO cos(zv[j], zv[j-1]):", " ".join(f"{cos(zv[0, :, j], zv[0, :, j-1]):.3f}" for j in range(1, zv.shape[2])))
print("VIDEO cos(zv[j], E_ss(last frame)):", " ".join(f"{cos(zv[0, :, j], zss[0, :, chunk_frames(j)[-1]]):.3f}" for j in range(1, zv.shape[2])))
print("VIDEO cos(zv[j], E_1(last frame)):", " ".join(f"{cos(zv[0, :, j], z1[0, :, chunk_frames(j)[-1]]):.3f}" for j in range(1, zv.shape[2])))
del zs
torch.cuda.empty_cache()

# Decodes
torch.cuda.empty_cache()
dec_v = decode(vae, mean, std, zv)
p_v = [psnr(dec_v[0, :, t], video[0, :, t]) for t in range(T)]
print(f"D(E(video)) PSNR mean {np.mean(p_v):.2f} (frame0 {p_v[0]:.2f})")
z_swap = zv.clone()
for j in range(1, zv.shape[2]):
    z_swap[0, :, j] = z1[0, :, chunk_frames(j)[-1]]
dec_s = decode(vae, mean, std, z_swap)
p_s = [psnr(dec_s[0, :, t], video[0, :, t]) for t in range(T)]
pos = np.array([(t - 1) % 4 for t in range(1, T)])
print(f"D(per-frame latents, last frame of chunk) PSNR mean {np.mean(p_s[1:]):.2f}; by position in chunk:",
      [round(float(np.mean(np.array(p_s[1:])[pos == q])), 2) for q in range(4)])
z_swap2 = zv.clone()
for j in range(1, zv.shape[2]):
    z_swap2[0, :, j] = z1[0, :, chunk_frames(j)[1]]   # second frame ~ chunk centre-ish
dec_s2 = decode(vae, mean, std, z_swap2)
p_s2 = [psnr(dec_s2[0, :, t], video[0, :, t]) for t in range(T)]
print(f"D(per-frame latents, 2nd frame of chunk) PSNR mean {np.mean(p_s2[1:]):.2f}; by position:",
      [round(float(np.mean(np.array(p_s2[1:])[pos == q])), 2) for q in range(4)])
z_swap3 = zv.clone()
for j in range(1, zv.shape[2]):
    z_swap3[0, :, j] = zss[0, :, chunk_frames(j)[-1]]
dec_s3 = decode(vae, mean, std, z_swap3)
p_s3 = [psnr(dec_s3[0, :, t], video[0, :, t]) for t in range(T)]
print(f"D(steady-state latents, last frame of chunk) PSNR mean {np.mean(p_s3[1:]):.2f}; by position:",
      [round(float(np.mean(np.array(p_s3[1:])[pos == q])), 2) for q in range(4)])
# what does a per-frame latent decode to when decoded alone (1-frame video)?
p_alone = []
for k in [1, 20, 40, 60, 80]:
    d1 = decode(vae, mean, std, z1[:, :, k:k + 1])
    p_alone.append(psnr(d1[0, :, 0], video[0, :, k]))
print("D(E(frame alone)) PSNR:", np.round(p_alone, 2).tolist())

json.dump(dict(rows=rows, rows_ss=rows_ss, static=d_static, psnr_video=p_v, psnr_swap_last=p_s, psnr_swap_2nd=p_s2,
               psnr_swap_ss=p_s3, psnr_alone=p_alone), open(os.path.join(OUT, "e0.json"), "w"))
# a strip to look at: frame 41..44 real / D(E(video)) / D(swap)
strip = torch.cat([torch.cat([video[0, :, t], dec_v[0, :, t], dec_s[0, :, t], dec_s3[0, :, t]], dim=1) for t in range(41, 45)], dim=2)
cv2.imwrite(os.path.join(OUT, "e0_strip.jpg"), ((strip.permute(1, 2, 0).numpy()[:, :, ::-1] + 1) * 127.5).astype(np.uint8))
