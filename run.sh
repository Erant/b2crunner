#!/bin/sh
# Usage: ./run.sh [DATA_DIR]
#
# DATA_DIR is the host directory mounted at /data in the container, where
# b2crunner keeps all its intermediate and final data. Without it, /data is
# the b2c_data named volume.

set -e

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [DATA_DIR]" >&2
    exit 2
fi

cd "$(dirname "$0")"

files="-f docker/docker-compose.yml -f docker/docker-compose.weights.yml"
if [ "$#" -eq 1 ]; then
    mkdir -p "$1"
    B2C_DATA_HOST_DIR=$(cd "$1" && pwd -P)
    export B2C_DATA_HOST_DIR
    files="$files -f docker/docker-compose.datadir.yml"
fi

# $files is split on purpose: it is a list of flags.
docker compose $files up
