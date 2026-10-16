#!/bin/sh
# Usage: ./run.sh [--wan-matmul 8|16] [DATA_DIR]
#
# DATA_DIR is the host directory mounted at /data in the container, where
# b2crunner keeps all its intermediate and final data. Without it, /data is
# the b2c_data named volume.
#
# --wan-matmul is the Wan 2.2 denoise's matmul precision: 8 (fp8 x fp8, the
# default) or 16 (bf16 activations against the fp8 weights). It reaches every
# run the container starts as $B2C_WAN_MATMUL.

set -e

usage() {
    echo "usage: $0 [--wan-matmul 8|16] [DATA_DIR]" >&2
    exit 2
}

wan_matmul=8
while [ "$#" -gt 0 ]; do
    case "$1" in
        --wan-matmul) [ "$#" -ge 2 ] || usage; wan_matmul=$2; shift 2 ;;
        --wan-matmul=*) wan_matmul=${1#--wan-matmul=}; shift ;;
        -*) usage ;;
        *) break ;;
    esac
done
case "$wan_matmul" in
    8|16) ;;
    *) echo "$0: --wan-matmul must be 8 or 16, not '$wan_matmul'" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    usage
fi

cd "$(dirname "$0")"

B2C_WAN_MATMUL=$wan_matmul
export B2C_WAN_MATMUL

files="-f docker/docker-compose.yml -f docker/docker-compose.weights.yml"
if [ "$#" -eq 1 ]; then
    mkdir -p "$1"
    B2C_DATA_HOST_DIR=$(cd "$1" && pwd -P)
    export B2C_DATA_HOST_DIR
    files="$files -f docker/docker-compose.datadir.yml"
fi

# $files is split on purpose: it is a list of flags.
docker compose $files up
