#!/usr/bin/env bash
# Walk brush's Vulkan chain link by link and say which one broke.
#
# `vulkaninfo` answers "is there a GPU" with a single yes/no, and every way
# of failing it looks identical from the outside: it enumerates llvmpipe and
# exits 0. docs/docker-build-notes.md section 1 burned most of a day on that
# — snap Docker, CDI vs legacy hook, --privileged and
# NVIDIA_DRIVER_CAPABILITIES were all tested and eliminated before the real
# cause (a missing libegl1, which the NVIDIA ICD needs at instance-creation
# time even though Vulkan never calls EGL) turned up.
#
# So this prints the state of every link instead of the verdict of the last
# one. Run it inside the container on the pod that is failing:
#
#     bash /opt/b2c_runner/scripts/vulkan_probe.sh
#
# It is read-only and takes a couple of seconds.

set -uo pipefail

section() { printf '\n=== %s ===\n' "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

section "0. the card, and the kernel-side driver"
# The driver version here is the KERNEL module's. Every injected userspace
# .so has to match it exactly; a mismatch is silent and fails closed.
if have nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
else
    echo "nvidia-smi ABSENT — the toolkit injected nothing at all"
fi
KERNEL_DRIVER=$(cat /proc/driver/nvidia/version 2>/dev/null | head -1)
echo "/proc/driver/nvidia/version: ${KERNEL_DRIVER:-(absent)}"

section "1. device nodes"
ls -l /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia* — nothing is going to work"
ls -l /dev/dri/ 2>/dev/null || echo "no /dev/dri (not fatal for NVIDIA's ICD)"

section "2. the ICD manifest (injected by nvidia-container-toolkit)"
# `graphics` in NVIDIA_DRIVER_CAPABILITIES is what makes the toolkit mount
# this. If it is missing, the toolkit either was not asked for graphics or
# the HOST driver install has no graphics userspace to mount (a headless
# datacenter driver, or a .run installed --no-opengl-files). Neither is
# fixable from inside the image.
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-(unset)}"
ICD_FOUND=0
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    for f in "$d"/*nvidia*.json; do
        [ -e "$f" ] || continue
        ICD_FOUND=1
        echo "--- $f"
        cat "$f"
        echo
    done
done
[ "$ICD_FOUND" = 1 ] || echo "NO NVIDIA ICD MANIFEST — stop here, this is link 2 (see above)"
echo "-- everything else the loader will scan:"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null

section "3. does the ICD library resolve, and does it MATCH the kernel module"
# libGLX_nvidia.so.0 is the Vulkan ICD (it backs GLX, EGL and Vulkan all
# three). The versioned GL/EGL core libraries beside it must carry the same
# version string as the kernel module printed in section 0.
ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-(glcore|glsi|eglcore|gpucomp|tls|rtcore)' \
    || echo "none of the NVIDIA graphics libraries are on the ldconfig path"
echo "-- versions actually on disk:"
find /usr/lib /usr/lib64 -maxdepth 3 \
    \( -name 'libnvidia-glcore.so.*' -o -name 'libnvidia-gpucomp.so.*' \
       -o -name 'libGLX_nvidia.so.*' \) 2>/dev/null | sort | sed 's/^/   /'
# libnvidia-gpucomp.so is the shader compiler, split out of glcore in the
# 550+ drivers. libnvidia-container older than 1.17 does not know to inject
# it, which breaks Vulkan on exactly the newer-driver hosts — i.e. the
# Blackwell fleet, while the older Ada hosts keep working. It is dlopened by
# path rather than linked, so it is NOT in the ldconfig cache even when it is
# present: test the filesystem, not `ldconfig -p`.
if find /usr/lib /usr/lib64 -maxdepth 3 -name 'libnvidia-gpucomp.so.*' 2>/dev/null | grep -q .; then
    echo "   libnvidia-gpucomp: present"
else
    echo "   libnvidia-gpucomp: ABSENT — expected on driver >= 550. If the"
    echo "   host driver in section 0 is >= 550, this is a libnvidia-container"
    echo "   too old to know about it, and is very likely the cause."
fi

section "4. the loader-side dependencies the ICD needs at instance creation"
# The Ada trap from docs/docker-build-notes.md: the NVIDIA ICD runs a GLVND
# self-registration during vkCreateInstance that needs libEGL.so.1 to be
# resolvable. Without it vk_icdGetInstanceProcAddr returns NULL for
# vkCreateInstance, with no error, and the loader silently picks llvmpipe.
python3 - <<'PY'
import ctypes
for lib in ("libvulkan.so.1", "libEGL.so.1", "libGLdispatch.so.0",
            "libGL.so.1", "libXext.so.6", "libGLX_nvidia.so.0"):
    try:
        ctypes.CDLL(lib)
        print(f"   {lib:24s} loads")
    except OSError as exc:
        print(f"   {lib:24s} FAILS: {exc}")
PY
if have nm && ldconfig -p | grep -q libGLX_nvidia; then
    SO=$(ldconfig -p | awk '/libGLX_nvidia\.so\.0/ {print $NF; exit}')
    echo "-- $SO exports:"
    nm -D "$SO" 2>/dev/null | grep -c vk_icdGetInstanceProcAddr | sed 's/^/   vk_icdGetInstanceProcAddr symbols: /'
fi

section "5. loader version vs ICD api_version"
have vulkaninfo && vulkaninfo --summary 2>/dev/null | grep -iE 'Vulkan Instance Version' | sed 's/^/   /'
dpkg-query -W -f='   libvulkan1 ${Version}\n' libvulkan1 2>/dev/null

section "6. what the loader actually decides, with its reasoning"
if have vulkaninfo; then
    VK_LOADER_DEBUG=error,warn vulkaninfo --summary 2>&1 \
        | grep -iE 'error|warn|could not|skipping|deviceName|driverName|deviceType|apiVersion' \
        | head -40
else
    echo "vulkaninfo not installed"
fi

section "verdict"
if ! have vulkaninfo; then
    echo "vulkaninfo is not installed here, so the last link is untested."
    echo "This script is meant to run INSIDE the container (the image ships"
    echo "vulkan-tools); on a bare host, install vulkan-tools to finish."
elif vulkaninfo --summary 2>/dev/null | grep -qi 'DISCRETE_GPU'; then
    echo "Vulkan reaches the GPU — brush will run on it."
else
    echo "No discrete GPU in Vulkan. Read the sections above in order; the"
    echo "first one that reports something ABSENT or FAILS is the cause."
    echo "Sections 2 and 3 are host-side (driver install / toolkit version)"
    echo "and cannot be fixed by changing this image. Section 4 can."
fi
