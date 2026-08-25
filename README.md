# b2c_runner

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline,
extracted from `ComfyUI-Body2COLMAP/pipeline`. See [pipeline/README.md](pipeline/README.md)
for the full design doc, module map, and current status.

Every node in the ComfyUI pack now has a native counterpart. Two workflows
ship, and they differ in exactly one stage:

| workflow | stages | for |
|---|---|---|
| `fast_helical_full` | six — includes the SeedVR2 upscale | the full port of the ComfyUI `fast helical` pipeline |
| `fast_helical` | five — no upscale | the same run with the upscaler taken out, to isolate it when output looks wrong |

**Neither has been run end-to-end on a pod.** The
[coverage section](pipeline/README.md#coverage-vs-the-comfyui-node-pack)
tracks what is verified against what.

Both end by producing whichever deliverables you ask for, under the run's
output directory:

```
<run>/colmap/   cameras.txt, images.txt, points3D.txt, images/, normals/
<run>/ply/      scene.ply — brush, normal-supervised
```

## Install

```bash
pip install -r requirements.txt
```

The Gaussian-splat steps additionally need `plyfile` for PLY I/O, and
`render_splat` needs the `brush-splat-render` binary on `PATH` (built
alongside `brush` in `docker/Dockerfile`) — see `requirements.txt`.
Face-landmark detection needs `mediapipe` (CPU-only).

## Quickstart

```bash
# the full pipeline against an existing dataset
python -m pipeline.cli run fast_helical_full --dataset path/to/b2c_dataset \
    --prompt "a woman in a red jacket"

# the same thing without the upscaler
python -m pipeline.cli run fast_helical --dataset path/to/b2c_dataset

# just the COLMAP dataset — skips a 30,000-iteration brush training
python -m pipeline.cli run fast_helical --dataset path/to/b2c_dataset \
    --param export_ply=false

# what can this machine actually run? (GPU, Vulkan, EGL, venvs, HF access)
python -m pipeline.cli doctor

# the web UI: submit a dataset or a .zip of one, watch progress, pull the
# result back as one .zip
python -m pipeline.cli ui            # needs `pip install 'gradio>=5.0,<7.0'`

python -m pipeline.cli workflows     # what's available
python -m pipeline.cli steps
```

Runs write to `$B2C_OUTPUT_DIR` (default `/data/output`, falling back to a
repo-local directory when there's no volume), and each one leaves a
timestamped log under `$B2C_LOG_DIR`. See [pipeline/paths.py](pipeline/paths.py).

## Deploying

One image holds every step's venv plus the `brush` binaries, and serves the
web UI by default. [docs/runpod.md](docs/runpod.md) has the pod template
settings and the debugging recipes; [docs/docker.md](docs/docker.md) has the
design rationale.

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
