"""E3-local: `LatentSync.velocity` end to end on the 4070 Ti, with E(V) of
cyber_6f/colmap's frames standing in for a denoise step's x0.

Exercises the real seams the pod run will: the Wan VAE decode/encode
under the pipeline's normalisation, the COLMAP export, b2ctrain (cold,
then warm-started) and its renderer, the alpha compositing, the band
blend, and the stats. Prints how far the render moves x0 and what the
synchronised estimate decodes to.
"""
import sys, os, types, time, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from body2colmap.camera import Camera
import importlib.util

spec = importlib.util.spec_from_file_location("wan22_sync", os.path.join(REPO, "pipeline", "steps", "wan22_sync.py"))
wan22_sync = importlib.util.module_from_spec(spec); spec.loader.exec_module(wan22_sync)

E2B, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], np.float64); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

# cameras: the 720x1280 COLMAP model -> body2colmap Cameras (OpenGL c2w), the
# inverse of coordinates.world_to_colmap_camera, as bench/colmap_to_cameras.py does it
cams_txt = [l.split() for l in open(os.path.join(REPO, "cyber_6f/colmap/cameras.txt")) if not l.startswith("#")]
_, _, W0, H0, fx, fy, cx, cy = cams_txt[0][:8]
cameras, names = [], []
for line in open(os.path.join(REPO, "cyber_6f/colmap/images.txt")):
    t = line.split()
    if not t or t[0].startswith("#") or len(t) < 10:
        continue
    qw, qx, qy, qz, tx, ty, tz = [float(v) for v in t[1:8]]
    R = quat_to_R(qw, qx, qy, qz); tr = np.array([tx, ty, tz])
    pos = -R.T @ tr; R_gl = R.T @ np.diag([1, -1, -1])
    cameras.append(Camera(focal_length=(float(fx), float(fy)), image_size=(int(W0), int(H0)),
                          principal_point=(float(cx), float(cy)), position=pos.astype(np.float32), rotation=R_gl.astype(np.float32)))
    names.append(t[9])
order = np.argsort(names); cameras = [cameras[i] for i in order]; names = [names[i] for i in order]
pts = np.array([[float(v) for v in l.split()[1:4]] for l in open(os.path.join(REPO, "cyber_6f/colmap/points3D.txt")) if not l.startswith("#")], np.float32)
points_3d = (pts, np.full((len(pts), 3), 128, np.uint8))

vae, mean, std = load_vae()
video = load_frames("colmap")
T = video.shape[2]
zv = torch.load(os.path.join(E2B, "codes.pt"))["zv"][None].to(DEV)   # 1,16,21,104,60 = E(V)
fr = ((video[0] + 1) * 127.5)
masks = [((np.abs(fr[:, t].permute(1, 2, 0).numpy() - 123).max(-1) > 8) * 255).astype(np.uint8) for t in range(T)]

sync = wan22_sync.LatentSync(
    pipe=types.SimpleNamespace(vae=vae), cameras=cameras, image_names=names, points_3d=points_3d, masks=masks,
    width=480, height=832, n_ref=0, steps=[2, 3, 4], mix=[1.0, 1.0, 0.5], band=1 / 6, iters=6000, warm_iters=1500,
    max_splats=400000, mask_dilate_px=24, trainer=os.path.expanduser("~/Projects/b2ctrain/build/b2ctrain"), debug_dir=OUT)
print("cameras scaled:", sync.cameras[0].width, sync.cameras[0].height, round(float(sync.cameras[0].fx), 2))

torch.manual_seed(0)
for step_index, sigma in ((2, 0.955), (3, 0.889), (4, 0.753)):
    noise = torch.randn_like(zv)
    sample = zv + sigma * noise            # x_t at this sigma, x0 = zv exactly
    v = (sample - zv) / sigma              # the model's velocity, if it were perfect
    t0 = time.time()
    v2 = sync.velocity(v, sample, sigma, step_index)
    x0_new = sample - sigma * v2
    print(f"step {step_index}: velocity() {time.time()-t0:.1f}s; x0 moved by rel {wan22_sync.rel_err(x0_new, zv):.3f}")
    torch.cuda.empty_cache()
    dec = decode(vae, mean, std, x0_new)
    ps = [psnr(dec[0, :, t], video[0, :, t]) for t in range(T)]
    print(f"        decode(x0'') vs frames PSNR {np.mean(ps):.2f} (decode(E(V)) is ~39)")
    if step_index == 2:
        strip = torch.cat([torch.cat([video[0, :, t], dec[0, :, t]], dim=1) for t in (10, 40, 70)], dim=2)
        cv2.imwrite(os.path.join(OUT, "e3_strip.jpg"), ((strip.permute(1, 2, 0).numpy()[:, :, ::-1] + 1) * 127.5).astype(np.uint8))
stats = sync.finish()
print(json.dumps(stats, indent=1))
