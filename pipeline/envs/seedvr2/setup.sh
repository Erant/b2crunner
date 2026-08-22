#!/bin/bash
# Run after this env's requirements.txt is installed. Vendors
# numz/ComfyUI-SeedVR2_VideoUpscaler (pipeline/steps/seedvr2.py imports
# `src.core.generation_utils` from it — not a published package) and drops
# a .pth file so it's importable without a manual PYTHONPATH export.
set -euo pipefail
source /workspace/env.sh

VENDOR_DIR=/workspace/venv_seedvr2_vendor
if [ ! -d "$VENDOR_DIR" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler "$VENDOR_DIR"
fi

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$VENDOR_DIR" > "$SITE_PACKAGES/seedvr2_vendor.pth"
echo "vendored at $VENDOR_DIR, added to $SITE_PACKAGES/seedvr2_vendor.pth"

echo "NOTE: flash-attn/apex are NOT installed by requirements.txt — they're"
echo "pinned to a specific torch/CUDA ABI upstream. Check this env's actual"
echo "torch build against ComfyUI-SeedVR2_VideoUpscaler's requirements.txt"
echo "before installing them; if the ABI doesn't match, this step needs"
echo "docker dispatch instead of subprocess (see pipeline README)."
