#!/bin/sh
# Create a per-step venv that shares /opt/venv_base's packages.
#
# Usage: make-child-venv /opt/venv_wan22
#
# Why not `python -m venv --system-site-packages`: that resolves the venv's
# `home` to the REAL base interpreter (sys._base_executable), not to the
# venv you invoked it from. A child created that way inherits
# /usr/lib/python3's site-packages and sees nothing of venv_base. Tested
# directly; it does not work.
#
# A .pth file does work, and gives all four properties we need:
#   1. the child imports venv_base's packages
#   2. pip in the child reports them "already satisfied", so no child ever
#      installs its own ~3GB torch
#   3. a package installed locally in the child SHADOWS the base copy,
#      because the child's own site-packages is searched first — the
#      isolation that justifies separate venvs at all
#   4. installing in a child leaves venv_base untouched
#
# The zz_ prefix only affects the order .pth files are processed; paths
# they add are appended to sys.path either way, which is what keeps
# property 3 true.

set -eu

TARGET="$1"
BASE="${BASE_VENV:-/opt/venv_base}"

BASE_SITE_PACKAGES="$("${BASE}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"

python3 -m venv "$TARGET"
"${TARGET}/bin/pip" install --upgrade pip setuptools wheel

TARGET_SITE_PACKAGES="$("${TARGET}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$BASE_SITE_PACKAGES" > "${TARGET_SITE_PACKAGES}/zz_shared_base.pth"

# Fail loudly here rather than three layers later with a confusing
# "No module named torch".
"${TARGET}/bin/python" -c "import torch; print('$TARGET shares torch', torch.__version__)"
