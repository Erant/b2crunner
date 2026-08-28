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

# MoGe — sam3d_body's FOV / focal-length estimator (see
# pipeline/steps/sam3d_body.py, fov_estimator="moge2", and INSTALL.md
# step 5). PINNED to b942f00, the last MoGe-2 commit: `main` is now MoGe-3
# (v3.0.0), whose deps drag in a CUDA extension (flex-gemm), gradio>=6, and
# a git package literally named `pipeline` that would shadow this repo's own
# `pipeline` in this venv. --no-deps for moge itself: the only inference-path
# deps are torch/scipy/numpy/cv2 (already present) + utils3d; the
# `pipeline`/training deps MoGe 2.0.0 also lists are import-time only in
# moge/{train,test}/, never in moge.model.v2, which is all steps/sam3d_body.py
# touches (it re-implements upstream's tools/build_fov_estimator rather than
# importing it — detectron2 ships its own top-level `tools` here that wins on
# sys.path). utils3d IS installed with its deps — only `moderngl` is new, it
# is small, and its .np/.pt math submodules (the ones MoGe touches) don't
# need it.
pip install -q --no-deps 'moge @ git+https://github.com/microsoft/MoGe.git@b942f00'
pip install -q 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
python3 -c "from moge.model.v2 import MoGeModel; print('moge (v2) import OK')"

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
