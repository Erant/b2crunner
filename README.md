# b2crunner

Turns a front/back reference sheet (or a single photo) of a person into a
Gaussian splat of them, delivered as a glTF subject file (`scene.glb`) plus a
COLMAP dataset.

## Run

Everything ships in one Docker image that serves a web UI and an HTTP API on
port 7860:

```bash
export HF_TOKEN=hf_...
./run.sh [DATA_DIR]        # DATA_DIR is mounted at /data; outputs land in /data/output
```

Or from a Python environment with the step venvs set up:

```bash
python -m pipeline.cli doctor                                # what this machine can run
python -m pipeline.cli run helical --reference-image sheet.png
python -m pipeline.cli ui                                    # web UI + /api/v1
```

Set `B2C_API_TOKEN` to guard the UI (login password) and enable the API.

## Model weights

Weights are prefetched at startup, and a run waits only for the ones its
workflow needs. Hugging Face models go through `huggingface_hub`'s normal
cache, so any existing cache is reused: outside Docker, `HF_HOME` /
`HF_HUB_CACHE` are honoured as usual. In Docker, `B2C_WEIGHTS_DIR` points
the hub cache at a mounted directory, and `./run.sh` already binds your
`~/.cache/huggingface` there read-only (override the host path with
`B2C_WEIGHTS_HOST_DIR`). Read-only means a model missing from that cache
fails instead of downloading; drop the `:ro` in
`docker/docker-compose.weights.yml` to let it download. Weights that aren't
on the Hugging Face hub go to `$B2C_MODELS_DIR` (default `/data/models`). See
[docs/docker.md](docs/docker.md#reusing-weights-you-already-have-2026-09-23).

`helical` needs all of the following, about 79 GB in total. Gated repos need
an `HF_TOKEN` whose account has accepted the model's licence.

| Model | Used for | Download |
|---|---|---|
| `silveroxides/Wan_2.2-fp8_scaled_hybrid` (two VACE experts) | video denoise | 35.2 GB |
| `linoyts/Wan2.2-VACE-Fun-14B-diffusers` (VAE, text encoder, scheduler) | video denoise | 11.9 GB |
| `lightx2v/Wan2.2-Lightning` (distill LoRAs) | video denoise | 1.2 GB |
| `facebook/sapiens2-pointmap-1b` | pointmap / depth | 6.5 GB |
| `facebook/sapiens2-seg-1b` | body-part segmentation | 6.5 GB |
| `facebook/sapiens2-normal-1b` | normal maps | 6.2 GB |
| SeedVR2 3B fp8 DiT + VAE | upscale | 6.0 GB |
| `facebook/sam-3d-body-dinov3` (gated) | body reconstruction | 2.8 GB |
| `Ruicheng/moge-2-vitl-normal` | focal-length estimate | 1.3 GB |
| `briaai/RMBG-2.0` (gated) | background removal | 0.9 GB |
| COLMAP ALIKED-N32 + LightGlue ONNX | camera refinement | 70 MB |
| `facebookresearch/dinov3` (torch.hub source) | SAM 3D Body backbone code | 20 MB |
| MediaPipe face landmarker + person detector | face / figure detection | 20 MB |

## More

- [docs/runpod.md](docs/runpod.md) — deploying on a pod, API usage
- [docs/docker.md](docs/docker.md) — the image
- [pipeline/README.md](pipeline/README.md) — design and module map

## Tests

```bash
python -m unittest discover -s tests -t .
```
