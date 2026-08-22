#!/bin/bash
# Run after this env's requirements.txt is installed (which already pulls
# facebookresearch/sam-3d-body itself via pip git+). This step only
# pre-downloads the checkpoint — requires HF_TOKEN to belong to an account
# that has accepted the license at
# https://huggingface.co/facebook/sam-3d-body-dinov3 (a human step, can't
# be scripted around).
set -euo pipefail
source /workspace/env.sh

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
