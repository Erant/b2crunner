# b2c_runner

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline,
extracted from `ComfyUI-Body2COLMAP/pipeline`. See [pipeline/README.md](pipeline/README.md)
for the full design doc, module map, and current status.

Every node in the ComfyUI pack now has a native counterpart. Three
workflows ship:

| workflow | starts from | stages |
|---|---|---|
| `fast_helical_full` | an existing dataset | six — the full port of the ComfyUI `fast helical` pipeline, upscale included |
| `fast_helical` | an existing dataset | five — the same run with the SeedVR2 upscale taken out, to isolate it when output looks wrong |
| `fast_helical_native` | a single photo | one forward pass — reconstruct a body, render its own views, denoise, train a splat |

**None has been run end-to-end on a pod**, and `fast_helical_native` is the
least proven of the three: it is the next piece of work, and predates the
export conventions the other two use. The
[coverage section](pipeline/README.md#coverage-vs-the-comfyui-node-pack)
tracks what is verified against what.

The two `fast_helical` workflows end by producing whichever deliverables
you ask for, under the run's output directory:

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

# or from a single photo — the workflow renders its own views
python -m pipeline.cli run fast_helical_native --reference-image photo.jpg \
    --prompt "a woman in a red jacket"

# what can this machine actually run? (GPU, Vulkan, EGL, venvs, HF access)
python -m pipeline.cli doctor

# the web UI: submit a dataset, a .zip of one, or a photo; watch progress,
# pull the result back as one .zip
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
