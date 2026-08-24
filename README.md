# b2c_runner

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline,
extracted from `ComfyUI-Body2COLMAP/pipeline`. See [pipeline/README.md](pipeline/README.md)
for the full design doc, module map, and current status.

Every node in the ComfyUI pack now has a native counterpart, and
`pipeline/workflows/fast_helical_full.yaml` is a complete port of the
`fast helical` pipeline. **It has never been run end-to-end** — `brush`
and the two rasterisers (`render`, `render_splat`) need a GPU pod. The
[coverage section](pipeline/README.md#coverage-vs-the-comfyui-node-pack)
tracks what is verified against what.

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
# run a workflow against an existing dataset
python -m pipeline.cli run roundtrip_example --dataset path/to/b2c_dataset

# or from a single photo — the workflow renders its own views
python -m pipeline.cli run fast_helical_native --reference-image photo.jpg \
    --prompt "a woman in a red jacket"

# what can this machine actually run? (GPU, Vulkan, EGL, venvs, HF access)
python -m pipeline.cli doctor

# the web UI: submit a dataset, a .zip of one, or a photo; watch progress
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
