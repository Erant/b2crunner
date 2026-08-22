# b2c_runner

Standalone (non-ComfyUI) execution engine for the Body2COLMAP pipeline,
extracted from `ComfyUI-Body2COLMAP/pipeline`. See [pipeline/README.md](pipeline/README.md)
for the full design doc, module map, and current status.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
python -m pipeline.cli run pipeline/workflows/roundtrip_example.yaml \
    --dataset path/to/existing/b2c_dataset -v
```
