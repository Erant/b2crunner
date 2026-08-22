#!/bin/bash
# Run after this env's requirements.txt is installed (see
# scripts/pod_bootstrap.sh, which calls this automatically). Downloads the
# WAN2.2 VACE-Fun-14B diffusers checkpoint (only the components
# wan22_vace_denoise.py actually loads) and the Lightning distill LoRAs.
# Idempotent: huggingface_hub skips files already present in HF_HOME.
set -euo pipefail
source /workspace/env.sh

python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download

path = snapshot_download(
    'linoyts/Wan2.2-VACE-Fun-14B-diffusers',
    allow_patterns=[
        'transformer/*', 'transformer_2/*', 'vae/*',
        'text_encoder/*', 'tokenizer/*', 'scheduler/*',
        'model_index.json',
    ],
)
print('checkpoint at', path)

for f in ['high_noise_model.safetensors', 'low_noise_model.safetensors']:
    p = hf_hub_download(
        'lightx2v/Wan2.2-Lightning', f,
        subfolder='Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1',
    )
    print('lora at', p)
"
