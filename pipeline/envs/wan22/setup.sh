#!/bin/bash
# Run after this env's requirements.txt is installed (see
# scripts/pod_bootstrap.sh, which calls this automatically, and which
# `pip install -e`s the repo just before this — so `pipeline` is importable
# here). Downloads everything wan22_vace_denoise loads.
#
# This used to inline its own snapshot_download with its own allow_patterns,
# and it drifted: when the step moved to the pre-quantized fp8 checkpoint,
# this file kept pulling `transformer/*` and `transformer_2/*` — 69 GB that
# nothing loads any more — and never fetched the fp8 experts the step now
# needs, so a bare-pod bootstrap downloaded the wrong 69 GB and still could
# not run. Now it drives pipeline/models.py's registry, which is the single
# place that knows what this step needs, so the next change to that set
# cannot leave this file behind.
#
# Idempotent: the registry records readiness per model and huggingface_hub
# skips files already present in HF_HOME.
set -euo pipefail
source /workspace/env.sh

python3 -c "
from pipeline.models import prefetch, registry, required_for_steps

keys = required_for_steps(['wan22_vace_denoise'])
known = registry()
print('fetching for wan22_vace_denoise:', keys,
      '(~%.1f GB)' % sum(known[k].approx_gb for k in keys))

status = prefetch(keys)
failed = [k for k, v in status.items() if v.get('status') == 'failed']
for key in keys:
    entry = status.get(key, {})
    print('  %-12s %s %s' % (key, entry.get('status', '?'), entry.get('location', '')))
if failed:
    raise SystemExit('failed to fetch: %s' % failed)
"
