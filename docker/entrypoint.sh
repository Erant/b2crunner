#!/bin/bash
# Container entrypoint. Decides what the container *is* — a long-lived pod
# serving the web UI, a one-shot CLI run, or a shell — from its first
# argument.
#
# The image previously had `ENTRYPOINT ["python", "-m", "pipeline.cli"]` and
# nothing else, which meant a RunPod pod started with no arguments printed
# an argparse usage error and exited. A pod whose container exits is a dead
# pod: no UI, no SSH, nothing to attach to, and the only diagnosis available
# is the same usage error. Hence a real entrypoint.
#
#   (no args) | ui         serve the web UI on $B2C_PORT, stay alive
#   run ...                one workflow run, then exit (the old behaviour;
#                          docker-compose.yml still uses this form)
#   doctor | steps | workflows | prefetch
#                          the corresponding CLI subcommand, then exit
#   bash | sh | shell      a shell; extra arguments are passed through, so
#                          `bash -c '...'` works
#   anything else          exec'd verbatim, so `docker run IMAGE nvidia-smi`
#                          works without --entrypoint
#
# Whatever the mode, this first makes the volume's directory layout exist
# and reports what it found. Those few lines at the top of the log answer
# "was the volume even mounted", which is otherwise a question you only get
# to ask after something has failed for an unrelated-looking reason.

set -euo pipefail

PYTHON=/opt/venv_main/bin/python
B2C_PORT="${B2C_PORT:-7860}"
DATA_DIR="${B2C_DATA_DIR:-/data}"

log() { echo "[entrypoint] $*"; }

# --------------------------------------------------------------------------
# volume layout
# --------------------------------------------------------------------------
prepare_volume() {
    if ! mkdir -p "$DATA_DIR"/{output,logs,uploads,tmp,hf_cache,models} 2>/dev/null; then
        log "WARNING: $DATA_DIR is not writable."
        log "  On RunPod this usually means the template has no volume, or its mount"
        log "  path is not $DATA_DIR. Everything will fall back to the container's"
        log "  writable layer, which is small and does not survive the pod."
        return 0
    fi
    log "volume $DATA_DIR: $(df -h "$DATA_DIR" | awk 'NR==2 {print $4" free of "$2}')"

    # The repo lives at /opt/b2c_runner precisely so that a volume mounted at
    # RunPod's default /workspace cannot shadow it. Say so if someone has
    # pointed the volume at the code anyway.
    if [ "$DATA_DIR" = "/opt/b2c_runner" ]; then
        log "WARNING: the volume is mounted over the application directory."
    fi
}

# --------------------------------------------------------------------------
# ssh, for when the UI is not enough
# --------------------------------------------------------------------------
# RunPod passes the account's public key in $PUBLIC_KEY. Without an sshd in
# the image there is no way into a custom-image pod other than the web UI,
# and "the UI won't start" is exactly when you need a shell.
start_sshd() {
    if [ -z "${PUBLIC_KEY:-}" ]; then
        log "no PUBLIC_KEY in the environment; not starting sshd"
        return 0
    fi
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1
    mkdir -p /run/sshd
    /usr/sbin/sshd
    log "sshd listening on 22"
}

# --------------------------------------------------------------------------
# model prefetch
# --------------------------------------------------------------------------
# In the BACKGROUND, deliberately, even though runs block on it. Doing it in
# the foreground would leave the pod with no UI, no log and no healthcheck
# for the half hour it takes to pull ~65 GB — which looks exactly like a pod
# that failed to start, and on RunPod may well be killed as one. This way
# the UI comes up immediately, shows the download on its Models tab, and any
# run submitted meanwhile waits for the subset it actually needs.
#
# Skips anything already on the volume, so a reused network volume starts
# essentially instantly. B2C_PREFETCH=0 turns it off entirely and restores
# the old lazy behaviour (each step downloads its own, mid-run).
start_prefetch() {
    if [ "${B2C_PREFETCH:-1}" = "0" ]; then
        log "B2C_PREFETCH=0 — not prefetching; steps will download on first use"
        return 0
    fi
    if [ -z "${HF_TOKEN:-}" ]; then
        log "WARNING: no HF_TOKEN — the gated checkpoints (RMBG-2.0,"
        log "  sam-3d-body-dinov3) will fail to download. Prefetching the rest."
    fi

    local log_file="${B2C_LOG_DIR:-$DATA_DIR/logs}/prefetch.log"
    log "prefetching model checkpoints in the background -> $log_file"
    nohup "$PYTHON" -m pipeline.cli prefetch --log-file "$log_file" \
        > /dev/null 2>&1 < /dev/null &
    disown
}

# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
serve_ui() {
    prepare_volume
    start_sshd

    # A summary sweep, not the full one: it costs a couple of seconds and
    # puts the pod's actual capabilities (Vulkan for brush, EGL for render,
    # the venvs, the HF token) in the log *before* anyone starts a run that
    # depends on them. Non-blocking on purpose — a WARN or even a FAIL is
    # still worth having a UI to look at.
    log "preflight:"
    "$PYTHON" -m pipeline.cli doctor --summary || log "doctor reported failures (continuing)"

    start_prefetch

    log "starting the web UI on 0.0.0.0:${B2C_PORT}"
    exec "$PYTHON" -m pipeline.cli ui --host 0.0.0.0 --port "$B2C_PORT"
}

main() {
    case "${1:-ui}" in
        ui)
            shift || true
            serve_ui
            ;;
        run|doctor|steps|workflows|prefetch)
            prepare_volume
            exec "$PYTHON" -m pipeline.cli "$@"
            ;;
        bash|sh|shell)
            prepare_volume
            # `exec "$@"`, not `exec /bin/bash`: the latter drops everything
            # after the first word, so `docker run IMAGE bash -c '...'` — the
            # form you reach for to poke at one thing without an interactive
            # session — silently ran an empty shell instead.
            [ "$1" = "shell" ] && { shift; set -- /bin/bash "$@"; }
            exec "$@"
            ;;
        *)
            exec "$@"
            ;;
    esac
}

main "$@"
