#!/bin/bash
# Builds the Erant/brush fork and installs the binary to /usr/local/bin.
# brush is a Rust CLI, not a Python package — pipeline/steps/brush.py calls
# it as a subprocess (see that module and docs/docker.md). No
# requirements.txt here: the only Python dependency (body2colmap) is
# already part of this pipeline package's own pyproject.toml, not
# env-specific.
#
# UNVERIFIED: this build has not actually been run — see docs/docker.md's
# "Open items" section. Confirm `brush --help` runs headless (no display)
# before trusting this in an automated pipeline run.
set -euo pipefail

if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0
    source "$HOME/.cargo/env"
fi

BUILD_DIR="${BRUSH_BUILD_DIR:-/workspace/brush_src}"
if [ ! -d "$BUILD_DIR" ]; then
    git clone https://github.com/Erant/brush.git "$BUILD_DIR"
fi

cd "$BUILD_DIR"
cargo build --release
install -m 755 target/release/brush /usr/local/bin/brush
brush --help
