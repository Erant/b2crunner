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
#   3. Builds ONE shared /workspace/venv_main containing every `in_process`-
#      dispatch step's deps (rmbg, sapiens2 as of this writing) plus this
#      pipeline package itself — InProcessDispatcher runs a step directly in
#      the calling process (pipeline/dispatch/in_process.py), not a
#      subprocess, so those deps have to live in whatever env actually runs
#      `python -m pipeline.cli`, not an isolated venv of their own.
#   4. Builds a separate isolated /workspace/venv_<name> per `subprocess`-
#      dispatch step (wan22, sam3dbody, seedvr2 as of this writing) from
#      pipeline/envs/<name>/requirements.txt, and runs that env's setup.sh
#      (weight downloads etc.) if present — these genuinely need isolation
#      (conflicting/heavy deps), dispatched via SubprocessPythonDispatcher
#      per envs.yaml's python_bin.
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
# Xet, not hf_transfer. huggingface_hub no longer uses HF_HUB_ENABLE_HF_TRANSFER
# as a transfer backend — it reads it only to warn you to set
# HF_XET_HIGH_PERFORMANCE instead. Xet matters here because the big weights
# (wan22's two 17.58 GB fp8 experts) live in Xet-backed repos, and a client
# with Xet disabled gets redirected to the xet-bridge CDN and pulls them as
# a single HTTP stream.
# HF_XET_HIGH_PERFORMANCE is deliberately not set — it buffers the whole
# file in RAM instead of flushing to disk as it goes. See docker/Dockerfile.
# Xet used to cache to ~/.cache regardless of HF_HOME, which on a pod with a
# small root disk and a large /workspace volume filled the root disk and
# killed downloads with "Disk quota exceeded". It now derives its cache from
# HF_HOME; this pins it onto the volume regardless of hub version.
export HF_XET_CACHE=/workspace/hf_cache/xet
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

echo "=== building shared /workspace/venv_main (in_process steps: rmbg, sapiens2) ==="
python3 -m venv /workspace/venv_main
source /workspace/venv_main/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
for env_name in rmbg sapiens2; do
    env_dir="$REPO_ROOT/pipeline/envs/$env_name"
    [ -f "$env_dir/requirements.txt" ] && pip install -q -r "$env_dir/requirements.txt"
done
pip install -q -e "$REPO_ROOT"
deactivate

for env_name in sam3dbody wan22 seedvr2; do
    env_dir="$REPO_ROOT/pipeline/envs/$env_name"
    venv_dir="/workspace/venv_${env_name}"
    [ -f "$env_dir/requirements.txt" ] || continue

    echo "=== building $venv_dir ==="
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install -q --upgrade pip

    if [ "$env_name" = "sam3dbody" ]; then
        # cu128, not cu124: confirmed on a real pod that an unpinned/cu124
        # torch install resolved to a cu130 build while the pod's actual
        # system CUDA toolkit was 12.8 (check `/usr/local/cuda-*/bin/nvcc
        # --version` — it may not be on PATH) — torch itself still ran
        # fine mismatched, but detectron2's native-extension build below
        # failed outright with a CUDA version-mismatch error. Match
        # whatever this specific machine's installed toolkit actually is.
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128

        # numpy/cython must be installed first, and the bulk install must
        # run with --no-build-isolation — confirmed on a real pod that
        # xtcocotools's legacy setup.py needs numpy at build time without
        # declaring it as a PEP 517 build dependency, so pip's normal
        # per-package isolated build env doesn't have it even when this
        # venv already does (see pipeline/envs/sam3dbody/requirements.txt).
        pip install -q numpy cython
        pip install -q --no-build-isolation -r "$env_dir/requirements.txt"

        # Re-pin torch+torchvision together after the bulk install, which
        # can silently upgrade torch via some other package's dependency
        # resolution (pytorch-lightning did, on a real pod) — confirmed
        # this leaves torchvision's compiled extensions ABI-mismatched
        # against the new torch ("RuntimeError: operator torchvision::nms
        # does not exist") unless both are reinstalled together.
        pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    else
        # torch first, from the CUDA wheel index, before the rest of each
        # env's requirements.txt (which pin torch>=X but don't specify
        # --index-url).
        pip install -q torch --index-url https://download.pytorch.org/whl/cu124
        pip install -q -r "$env_dir/requirements.txt"
    fi
    pip install -q -e "$REPO_ROOT"

    if [ -f "$env_dir/setup.sh" ]; then
        echo "=== running $env_dir/setup.sh ==="
        bash "$env_dir/setup.sh"
    fi

    deactivate
done

# brush is intentionally NOT built here. It needs OS-level Vulkan/graphics
# capability — confirmed on a real bare RunPod pod that this isn't present
# by default (NVIDIA_DRIVER_CAPABILITIES only exposed compute,utility) and
# isn't fixable by installing anything, on any dispatch, into an
# already-running pod — it has to be baked into the pod's own image at
# creation time. See docker/Dockerfile (which does build brush) and
# docs/docker.md; this bare-pod bootstrap script can't help here at all.

echo "=== pod_bootstrap.sh complete ==="
