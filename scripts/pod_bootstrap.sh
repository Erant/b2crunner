#!/bin/bash
# One-time bootstrap for a fresh GPU pod (developed against a RunPod
# PyTorch-template pod with an L40S, 46GB VRAM, /workspace as a large
# persistent volume and a small overlay root disk — install everything
# under /workspace, not /root).
#
# Usage (on the pod, as root):
#   export HF_TOKEN=hf_...
#   ./scripts/pod_bootstrap.sh
#
# What this does:
#   1. Writes /workspace/env.sh (HF cache dir + token + hf_transfer) —
#      `source /workspace/env.sh` in any later shell/script.
#   2. Arms a safety auto-shutdown (default 2h) via runpodctl so a hung
#      job doesn't burn money unattended. Override with AUTO_SHUTDOWN_HOURS.
#   3. Builds each step's isolated venv under /workspace/venv_<name> from
#      pipeline/envs/<name>/requirements.txt and runs that env's setup.sh
#      (weight downloads etc.) if present.
#
# Safe to re-run: venv creation and weight downloads are idempotent
# (python -m venv no-ops on an existing dir; huggingface_hub skips files
# already in HF_HOME's cache).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO_SHUTDOWN_HOURS="${AUTO_SHUTDOWN_HOURS:-2}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Set HF_TOKEN before running this script." >&2
    exit 1
fi

echo "=== writing /workspace/env.sh ==="
mkdir -p /workspace/hf_cache
cat > /workspace/env.sh <<ENVEOF
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=${HF_TOKEN}
export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface_hub's Xet backend caches to ~/.cache regardless of HF_HOME —
# on a pod with a small root disk and a large /workspace volume, that fills
# the root disk and downloads die with "Disk quota exceeded" even though
# /workspace has plenty of room. Disable it.
export HF_HUB_DISABLE_XET=1
ENVEOF
source /workspace/env.sh

echo "=== arming ${AUTO_SHUTDOWN_HOURS}h auto-shutdown safety net (pod: ${RUNPOD_POD_ID:-unknown}) ==="
if command -v runpodctl >/dev/null && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
    SECONDS_TOTAL=$((AUTO_SHUTDOWN_HOURS * 3600))
    nohup bash -c "sleep ${SECONDS_TOTAL} && runpodctl stop pod ${RUNPOD_POD_ID}" \
        > /workspace/auto_shutdown.log 2>&1 < /dev/null &
    disown
    echo "auto-shutdown armed: pod stops in ${AUTO_SHUTDOWN_HOURS}h unless this process is killed (pid $!)"
else
    echo "WARNING: runpodctl/RUNPOD_API_KEY/RUNPOD_POD_ID not all available — auto-shutdown NOT armed." >&2
fi

for env_name in rmbg sam3dbody wan22 seedvr2; do
    env_dir="$REPO_ROOT/pipeline/envs/$env_name"
    venv_dir="/workspace/venv_${env_name}"
    [ -f "$env_dir/requirements.txt" ] || continue

    echo "=== building $venv_dir ==="
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install -q --upgrade pip

    # torch first, from the CUDA wheel index, before the rest of each env's
    # requirements.txt (which pin torch>=X but don't specify --index-url).
    pip install -q torch --index-url https://download.pytorch.org/whl/cu124
    pip install -q -r "$env_dir/requirements.txt"
    pip install -q -e "$REPO_ROOT"

    if [ -f "$env_dir/setup.sh" ]; then
        echo "=== running $env_dir/setup.sh ==="
        bash "$env_dir/setup.sh"
    fi

    deactivate
done

echo "=== pod_bootstrap.sh complete ==="
