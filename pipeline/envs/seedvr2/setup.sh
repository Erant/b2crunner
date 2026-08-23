#!/bin/bash
# Run after this env's requirements.txt is installed. Vendors
# numz/ComfyUI-SeedVR2_VideoUpscaler (pipeline/steps/seedvr2.py does
# `import inference_cli` and uses its private `_process_frames_core` —
# the repo's real CLI entrypoint, not a published package, and not
# `src.core.generation_utils` as an earlier version of this file assumed:
# that function actually lives in inference_cli.py itself, confirmed by
# reading the repo's source directly).
set -euo pipefail
source /workspace/env.sh

VENDOR_DIR=/workspace/venv_seedvr2_vendor
if [ ! -d "$VENDOR_DIR" ]; then
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler "$VENDOR_DIR"
fi

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$VENDOR_DIR" > "$SITE_PACKAGES/seedvr2_vendor.pth"
echo "vendored at $VENDOR_DIR, added to $SITE_PACKAGES/seedvr2_vendor.pth"

echo "NOTE: attention_mode defaults to sdpa (pure PyTorch, no extra native"
echo "extension). flash-attn/sageattention are optional accelerators only"
echo "imported if explicitly requested via params.attention_mode — install"
echo "them by hand later if flash_attn_2/flash_attn_3/sageattn_* is wanted."
