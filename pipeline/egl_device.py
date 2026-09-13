"""Which EGL device pyrender renders on.

pyrender takes EGL device index `EGL_DEVICE_ID` (default 0) and nothing else
in this pipeline ever set it. That index is the NVIDIA driver's own
enumeration order, which ignores `CUDA_VISIBLE_DEVICES` — the only GPU
pinning `run_worker` does — and on a rented pod may not even be a card the
container can open: `eglQueryDevicesEXT` lists every GPU on the host, the
toolkit only mounts ours, and `eglInitialize` on one of the others fails
with EGL_NOT_INITIALIZED (0x3001). That is the doctor failure of
2026-09-13: pods landing in host slot 0 passed, the rest did not.

So this walks the devices once, before pyrender is imported, and picks the
one to use:

1. the NVIDIA device whose `EGL_CUDA_DEVICE_NV` is the CUDA device this
   process renders on (ordinal 0 under the worker's pinning), if it
   initialises;
2. else the first NVIDIA device that initialises;
3. else anything that initialises (Mesa's software device — llvmpipe);
4. else nothing, and the caller falls back to OSMesa.

Every probe is `eglInitialize` + `eglTerminate` on a platform-device
display; EGL displays are process-global and refcounted, so pyrender's own
initialise afterwards is unaffected. `doctor` prints the same walk so the
answer is on the pod's first screen rather than in a probe script.
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_VENDOR = 0x3053
EGL_CUDA_DEVICE_NV = 0x323A  # EGL_NV_device_cuda; the query fails on non-CUDA devices

ENV_DEVICE_ID = "EGL_DEVICE_ID"


@dataclass
class EGLDeviceInfo:
    index: int
    vendor: Optional[str]         # None unless probed and initialised
    cuda_device: Optional[int]    # EGL_CUDA_DEVICE_NV, None when unsupported/hidden
    initialised: Optional[bool]   # None = not probed (an NVIDIA device already answered)

    @property
    def is_nvidia(self) -> bool:
        return bool(self.vendor) and "nvidia" in self.vendor.lower()

    def describe(self) -> str:
        cuda = f" cuda:{self.cuda_device}" if self.cuda_device is not None else ""
        if self.initialised is None:
            return f"egl device[{self.index}]:{cuda} not probed"
        if not self.initialised:
            return f"egl device[{self.index}]:{cuda} failed to initialise (EGL_NOT_INITIALIZED)"
        return f"egl device[{self.index}]: {self.vendor}{cuda}"


class _CtypesDriver:
    """The raw EGL entry points, through libEGL.so.1 (glvnd) — no PyOpenGL,
    so this can run before `PYOPENGL_PLATFORM` matters."""

    def __init__(self) -> None:
        egl = ctypes.CDLL("libEGL.so.1")
        egl.eglGetProcAddress.restype = ctypes.c_void_p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        egl.eglQueryString.restype = ctypes.c_char_p
        egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
        egl.eglInitialize.restype = ctypes.c_uint
        egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int)]
        egl.eglTerminate.restype = ctypes.c_uint
        egl.eglTerminate.argtypes = [ctypes.c_void_p]
        self._egl = egl

        def proc(name: bytes, proto):
            addr = egl.eglGetProcAddress(name)
            return proto(addr) if addr else None

        self._query_devices = proc(
            b"eglQueryDevicesEXT",
            ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                             ctypes.POINTER(ctypes.c_int)))
        self._platform_display = proc(
            b"eglGetPlatformDisplayEXT",
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p))
        self._query_attrib = proc(
            b"eglQueryDeviceAttribEXT",
            ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_ssize_t)))  # EGLAttrib = intptr_t
        if self._query_devices is None or self._platform_display is None:
            raise RuntimeError("EGL_EXT_device_enumeration unavailable")

    def query_devices(self) -> List[int]:
        n = ctypes.c_int(0)
        if not self._query_devices(0, None, ctypes.byref(n)) or n.value < 1:
            return []
        arr = (ctypes.c_void_p * n.value)()
        got = ctypes.c_int(0)
        if not self._query_devices(n.value, arr, ctypes.byref(got)):
            return []
        return [arr[i] or 0 for i in range(got.value)]

    def cuda_device(self, device: int) -> Optional[int]:
        if self._query_attrib is None:
            return None
        value = ctypes.c_ssize_t(-1)
        if not self._query_attrib(ctypes.c_void_p(device), EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
            return None
        return int(value.value)

    def probe(self, device: int) -> Optional[str]:
        """Initialise the device's display; its vendor string, or None."""
        dpy = self._platform_display(EGL_PLATFORM_DEVICE_EXT, ctypes.c_void_p(device), None)
        if not dpy:
            return None
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not self._egl.eglInitialize(ctypes.c_void_p(dpy), ctypes.byref(major), ctypes.byref(minor)):
            return None
        try:
            return (self._egl.eglQueryString(ctypes.c_void_p(dpy), EGL_VENDOR) or b"?").decode()
        finally:
            self._egl.eglTerminate(ctypes.c_void_p(dpy))


def enumerate_devices(driver=None) -> List[EGLDeviceInfo]:
    """Every EGL device the driver lists. The ones carrying a CUDA device
    attribute (NVIDIA's) are probed first; the rest (Mesa's, including its
    own driverless view of the NVIDIA card) are probed only if none of those
    initialised — each Mesa probe prints `libEGL warning` noise and none of
    them is a GPU. Raises if libEGL or the enumeration extension is missing
    (then pyrender's default path is the only option anyway)."""
    driver = driver or _CtypesDriver()
    handles = driver.query_devices()
    infos = [EGLDeviceInfo(index=i, vendor=None, cuda_device=driver.cuda_device(h), initialised=None)
             for i, h in enumerate(handles)]

    def probe(info: EGLDeviceInfo) -> None:
        info.vendor = driver.probe(handles[info.index])
        info.initialised = info.vendor is not None

    cuda_backed = [d for d in infos if d.cuda_device is not None]
    for d in cuda_backed:
        probe(d)
    if not any(d.initialised for d in cuda_backed):
        for d in infos:
            if d.initialised is None:
                probe(d)
    return infos


def select_device(devices: List[EGLDeviceInfo], target_cuda: int = 0) -> Optional[EGLDeviceInfo]:
    live = [d for d in devices if d.initialised]
    nvidia = [d for d in live if d.is_nvidia]
    for d in nvidia:
        if d.cuda_device == target_cuda:
            return d
    if nvidia:
        return nvidia[0]
    return live[0] if live else None


def configure(target_cuda: int = 0) -> Tuple[Optional[List[EGLDeviceInfo]], Optional[EGLDeviceInfo]]:
    """Set `EGL_DEVICE_ID` for pyrender, unless the caller already did.
    Returns `(devices, chosen)`: `devices` is None when enumeration itself is
    unavailable (pyrender's default-display path is then the only option and
    is left alone); `chosen` is None when the listed devices all refuse to
    initialise (the caller decides whether that means OSMesa)."""
    try:
        devices = enumerate_devices()
    except Exception as exc:
        logger.warning("EGL device enumeration unavailable (%s); pyrender keeps its default", exc)
        return None, None
    for d in devices:
        logger.info(d.describe())
    if ENV_DEVICE_ID in os.environ:
        logger.info("%s=%s already set; leaving it", ENV_DEVICE_ID, os.environ[ENV_DEVICE_ID])
        return devices, next((d for d in devices if str(d.index) == os.environ[ENV_DEVICE_ID]), None)
    chosen = select_device(devices, target_cuda)
    if chosen is None:
        logger.warning("no EGL device initialises (%d listed)", len(devices))
        return devices, None
    os.environ[ENV_DEVICE_ID] = str(chosen.index)
    logger.info("pyrender renders on %s", chosen.describe())
    return devices, chosen
