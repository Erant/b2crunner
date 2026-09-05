# b2c_runner

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline,
extracted from `ComfyUI-Body2COLMAP/pipeline`. See [pipeline/README.md](pipeline/README.md)
for the full design doc, module map, and current status.

Every node in the ComfyUI pack now has a native counterpart. One workflow
ships:

| workflow | starts from | stages |
|---|---|---|
| `fast_helical_native` | a front/back reference sheet | a bootstrap prologue — split the sheet, reconstruct a body, nod the craned head back, build a Gaussian splat of the subject's face from a crop of the front half and composite it onto a circular orbit of outline+skeleton renders, warp the photo onto the anchor frame — then the full native port of the ComfyUI `fast helical` pipeline: two denoise passes and two brush trainings around a helical re-render. The first of those trainings also gets *supporting views* — a cap of renders of the face splat, and a Gaussian shell built off every Nth denoised frame and rendered from ±10° of elevation, which is what gives a circular orbit something to triangulate. `--param run_upscale=false` drops the SeedVR2 upscale (the old `fast_helical` workflow) to isolate it when output looks wrong |

An alternative bootstrap — the photo-to-splat *shell*, a body-wide
Gaussian shell with a pose refit against it and a band of frames rendered
off it (`fast_helical_shell.yaml`) — was tried alongside and retired on
2026-09-04; it lives in git history.

**`fast_helical_native` has not been run end-to-end on a pod** — its
bootstrap prologue has never executed on real hardware. The
[coverage section](pipeline/README.md#coverage-vs-the-comfyui-node-pack)
tracks what is verified against what.

Both workflows end by producing whichever deliverables you ask for, under
the run's output directory:

```
<run>/colmap/                cameras.txt, images.txt, points3D.txt, images/, normals/
<run>/ply/                   scene.ply — brush, normal-supervised
<run>/colmap_intermediate/   debug: what the first brush training was fed
<run>/colmap_preupscale/     debug: the same, from the pre-upscale frames
```

## Install

```bash
pip install -r requirements.txt
```

The Gaussian-splat steps additionally need `plyfile` for PLY I/O, and
`render_splat` needs the `brush-splat-render` binary on `PATH` (built
alongside `brush` in `docker/Dockerfile`) — see `requirements.txt`.
Face-landmark detection needs `mediapipe` (CPU-only); no shipped workflow
uses it any more — `fast_helical_native`'s face splat replaced it — but the
step and its `render` params are still there. The `pointmap_splat` family
needs `scipy`, which arrives anyway as a transitive dependency of
`body2colmap` (via `pyrender`).

## Quickstart

```bash
# from a front/back reference sheet (subject facing front on the left, seen
# from behind on the right) — the workflow splits it and renders its own views
python -m pipeline.cli run fast_helical_native --reference-image sheet.png \
    --prompt "a woman in a red jacket"

# the same thing without the upscaler
python -m pipeline.cli run fast_helical_native --reference-image sheet.png \
    --param run_upscale=false

# just the COLMAP dataset — skips a 30,000-iteration brush training
python -m pipeline.cli run fast_helical_native --reference-image sheet.png \
    --param export_ply=false

# the form the pipeline declares — its settings and its outputs — then
# every step's own params (add --all for the ones nothing overrides)
python -m pipeline.cli params fast_helical_native

# the COLMAP dataset the first brush training is handed, for when the
# helical re-render comes out wrong and the question is what it trained on
python -m pipeline.cli run fast_helical_native --reference-image sheet.png \
    --param export_colmap_intermediate=true

# what can this machine actually run? (GPU, Vulkan, EGL, venvs, HF access)
python -m pipeline.cli doctor

# the web UI: upload a reference sheet, or a .zip of image/prompt pairs (one
# run per pair, fanned across every GPU); watch progress, pull the result
# back as one .zip. Its Settings and Outputs boxes are the workflow's own
# `settings:` / `outputs:` blocks; the ~300 per-step knobs are still all
# there, behind the "Per-step settings" fold.
#
# The same command also serves an HTTP API at /api/v1 on the same port —
# submit, poll, download, cancel — for everything a browser is the wrong
# tool for. Both need B2C_API_TOKEN set: the API takes it as a bearer
# token, the UI as its login password. Without it neither is guarded and
# the API is not served at all. See docs/runpod.md, "Automating it".
python -m pipeline.cli ui            # needs the `ui` extra (gradio, fastapi, uvicorn)

# ...and a client for it. No gradio, no fastapi, no torch — the plain
# `requirements.txt` install is enough, so it runs from a laptop that
# could not host the pipeline. `api run` does the whole job:
# submit, print each stage as it finishes with what it cost, download the
# result .zip. One subcommand per route besides.
export B2C_API_URL=https://<pod-id>-7860.proxy.runpod.net B2C_API_TOKEN=...
python -m pipeline.cli api run sheet.png --prompt "a woman in a red jacket" \
    --param run_upscale=false -o results/
python -m pipeline.cli api runs
python -m pipeline.cli api follow <run>

python -m pipeline.cli workflows     # what's available
python -m pipeline.cli steps
```

Runs write to `$B2C_OUTPUT_DIR` (default `/data/output`, falling back to a
repo-local directory when there's no volume), and each one leaves a
timestamped log under `$B2C_LOG_DIR`. See [pipeline/paths.py](pipeline/paths.py).

## Deploying

One image holds every step's venv plus the `brush` binaries, and serves the
web UI and the HTTP API by default, on one port. [docs/runpod.md](docs/runpod.md)
has the pod template settings, the `B2C_API_TOKEN` both are guarded by, the
curl recipes under **Automating it**, and the debugging recipes;
[docs/docker.md](docs/docker.md) has the design rationale.

## Tests

Stdlib `unittest`, no pytest dependency:

```bash
python -m unittest discover -s tests -t .
```

Most tests are golden-output tests against `cyber_6f/` — a real completed
run of the original ComfyUI pipeline, kept as local reference data and
gitignored. They skip cleanly when it is absent, so a fresh clone still
runs the suite; with it present they compare ported steps against the
frames and COLMAP files the ComfyUI graphs actually produced.
