#!/bin/bash
# Run after this env's requirements.txt is installed with:
#   pip install numpy cython
#   pip install --no-build-isolation -r pipeline/envs/sam3dbody/requirements.txt
# (see that requirements.txt's own comment for why — xtcocotools needs
# numpy at build time without declaring it as a PEP 517 dependency).
#
# Builds detectron2 with the exact pin from facebookresearch/sam-3d-body's
# INSTALL.md, vendors sam-3d-body itself (confirmed on a real pod: it is
# NOT pip-installable — `pip install git+...sam-3d-body.git` fails with
# "does not appear to be a Python project: neither 'setup.py' nor
# 'pyproject.toml' found". INSTALL.md/the official notebook both assume
# you `sys.path.insert` a cloned checkout instead — same situation as
# seedvr2's vendored ComfyUI-SeedVR2_VideoUpscaler, same fix: clone + a
# .pth file), then pre-downloads the checkpoint — the last step requires
# HF_TOKEN to belong to an account that has accepted the license at
# https://huggingface.co/facebook/sam-3d-body-dinov3 (a human step, can't
# be scripted around).
set -euo pipefail
source /workspace/env.sh

pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

# detectron2 pins iopath<0.1.10 but torchvision's own resolver can pull in
# 0.1.10+ as a side effect of matching torch/torchvision versions —
# confirmed on a real pod this doesn't break the actual import, but if it
# ever does, `pip install 'iopath<0.1.10'` after this line is the fix.

VENDOR_DIR=/workspace/venv_sam3dbody_vendor
if [ ! -d "$VENDOR_DIR" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git "$VENDOR_DIR"
fi
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$VENDOR_DIR" > "$SITE_PACKAGES/sam3dbody_vendor.pth"
python3 -c "from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body; print('sam_3d_body import OK')"

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
