"""Which EGL device `render` uses (pipeline/egl_device.py).

pyrender takes EGL device index 0 unless `EGL_DEVICE_ID` says otherwise, and
nothing in the pipeline ever set it. The driver's index 0 is a host-wide
ordering: on a rented pod it is frequently a GPU the container cannot open,
and `eglInitialize` on it fails with EGL_NOT_INITIALIZED (the doctor failure
of 2026-09-13). The selection here is driven through a fake driver — the
real one needs libEGL and a GPU.
"""

from __future__ import annotations

import os
import unittest
import unittest.mock

from pipeline import egl_device
from pipeline.egl_device import EGLDeviceInfo, enumerate_devices, select_device


class _FakeDriver:
    """`devices` is a list of (vendor-or-None, cuda_device-or-None)."""

    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return list(range(100, 100 + len(self._devices)))

    def cuda_device(self, handle):
        return self._devices[handle - 100][1]

    def probe(self, handle):
        return self._devices[handle - 100][0]


def _infos(*devices) -> list:
    return enumerate_devices(_FakeDriver(list(devices)))


class EnumerateTest(unittest.TestCase):
    def test_dead_device_is_reported_not_raised(self):
        infos = _infos((None, 0), ("NVIDIA", 1))
        self.assertEqual([d.initialised for d in infos], [False, True])
        self.assertIn("failed to initialise", infos[0].describe())
        self.assertEqual(infos[1].describe(), "egl device[1]: NVIDIA cuda:1")

    def test_mesa_devices_are_not_probed_while_an_nvidia_one_works(self):
        # The local 4070 Ti box: NVIDIA, then Mesa's driverless view of the
        # same card (would fail, noisily), then Mesa's software devices.
        infos = _infos(("NVIDIA", 0), (None, None), ("Mesa Project", None), ("Mesa Project", None))
        self.assertEqual([d.initialised for d in infos], [True, None, None, None])
        self.assertIn("not probed", infos[1].describe())

    def test_mesa_devices_are_probed_when_no_nvidia_one_works(self):
        infos = _infos((None, 0), ("Mesa Project", None))
        self.assertEqual([d.initialised for d in infos], [False, True])


class SelectTest(unittest.TestCase):
    def test_pod_in_host_slot_two(self):
        # Host has three GPUs, ours is the third: index 0 and 1 refuse to
        # initialise, index 2 is our card (cuda:0 under the worker pinning),
        # Mesa's software device trails.
        infos = _infos((None, None), (None, None), ("NVIDIA", 0), ("Mesa Project", None))
        self.assertEqual(select_device(infos, target_cuda=0).index, 2)
        infos = _infos((None, 1), (None, 2), ("NVIDIA", 0), ("Mesa Project", None))
        self.assertEqual(select_device(infos, target_cuda=0).index, 2)

    def test_matches_the_cuda_device_over_enumeration_order(self):
        # Two workers on a two-GPU pod: worker pinned to cuda:1 must not
        # render on EGL device 0 just because it comes first.
        infos = _infos(("NVIDIA", 0), ("NVIDIA", 1))
        self.assertEqual(select_device(infos, target_cuda=1).index, 1)
        self.assertEqual(select_device(infos, target_cuda=0).index, 0)

    def test_falls_back_to_any_live_nvidia_device(self):
        # The attribute did not match anything (unsupported, or hidden):
        # take the first NVIDIA device that initialises, not Mesa.
        infos = _infos(("Mesa Project", None), (None, None), ("NVIDIA", None))
        self.assertEqual(select_device(infos, target_cuda=0).index, 2)

    def test_software_only_is_still_something(self):
        infos = _infos((None, None), ("Mesa Project", None))
        self.assertEqual(select_device(infos, target_cuda=0).index, 1)

    def test_nothing_initialises(self):
        self.assertIsNone(select_device(_infos((None, None), (None, None)), target_cuda=0))
        self.assertIsNone(select_device([], target_cuda=0))


class ConfigureTest(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(egl_device.ENV_DEVICE_ID, None)

    def tearDown(self):
        os.environ.pop(egl_device.ENV_DEVICE_ID, None)
        if self._env is not None:
            os.environ[egl_device.ENV_DEVICE_ID] = self._env

    def test_exports_the_choice_for_pyrender(self):
        infos = _infos((None, None), ("NVIDIA", 0))
        with unittest.mock.patch.object(egl_device, "enumerate_devices", return_value=infos):
            devices, chosen = egl_device.configure(target_cuda=0)
        self.assertEqual(chosen.index, 1)
        self.assertEqual(os.environ[egl_device.ENV_DEVICE_ID], "1")

    def test_an_explicit_setting_is_left_alone(self):
        os.environ[egl_device.ENV_DEVICE_ID] = "0"
        infos = _infos(("NVIDIA", 0), ("NVIDIA", 1))
        with unittest.mock.patch.object(egl_device, "enumerate_devices", return_value=infos):
            devices, chosen = egl_device.configure(target_cuda=1)
        self.assertEqual(chosen.index, 0)
        self.assertEqual(os.environ[egl_device.ENV_DEVICE_ID], "0")

    def test_nothing_initialises_sets_nothing(self):
        infos = _infos((None, None))
        with unittest.mock.patch.object(egl_device, "enumerate_devices", return_value=infos):
            devices, chosen = egl_device.configure(target_cuda=0)
        self.assertEqual(len(devices), 1)
        self.assertIsNone(chosen)
        self.assertNotIn(egl_device.ENV_DEVICE_ID, os.environ)

    def test_no_enumeration_is_not_a_dead_driver(self):
        with unittest.mock.patch.object(egl_device, "enumerate_devices",
                                        side_effect=RuntimeError("no extension")):
            devices, chosen = egl_device.configure(target_cuda=0)
        self.assertIsNone(devices)
        self.assertIsNone(chosen)
        self.assertNotIn(egl_device.ENV_DEVICE_ID, os.environ)


if __name__ == "__main__":
    unittest.main()
