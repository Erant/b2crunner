#!/bin/bash
# Run after torch + this env's requirements.txt are installed (see
# scripts/pod_bootstrap.sh). Builds detectron2 with the exact pin from
# facebookresearch/sam-3d-body's INSTALL.md, installs sam-3d-body itself,
# then pre-downloads the checkpoint — the last step requires HF_TOKEN to
# belong to an account that has accepted the license at
# https://huggingface.co/facebook/sam-3d-body-dinov3 (a human step, can't
# be scripted around).
set -euo pipefail
source /workspace/env.sh

pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install -q git+https://github.com/facebookresearch/sam-3d-body.git

python3 -c "
from huggingface_hub import snapshot_download
try:
    path = snapshot_download('facebook/sam-3d-body-dinov3')
    print('checkpoint at', path)
except Exception as e:
    print('DOWNLOAD FAILED — has this HF account accepted the license at')
    print('https://huggingface.co/facebook/sam-3d-body-dinov3 ?')
    print(e)
    raise
"
