"""doctor's `check_torch` on a multi-GPU box.

Before this, the matmul smoke test only ever exercised the process's
current/default CUDA device (the bare `"cuda"` string) and its message
hardcoded `cuda:0` regardless of how many devices `torch` actually saw. On
a pod with several GPUs attached, that meant a bad second or third card
could sit there unreported — `doctor` would say OK because device 0 was
fine. `check_torch` now loops every visible device and fails if any one of
them can't run a real kernel.

`torch` isn't installed in every environment this suite runs in (see
`docker/docker-build-notes.md`), so this drives it through a fake module
injected into `sys.modules` rather than needing the real thing — the same
reason `test_doctor_attention.py` stubs the wan22-venv probe instead of
needing SageAttention installed.
"""

from __future__ import annotations

import sys
import types
import unittest
import unittest.mock

from pipeline import doctor


class _FakeTensor:
    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def __float__(self):
        return 1.0


class _FakeProps:
    def __init__(self, name: str, major: int, minor: int, total_memory: float):
        self.name = name
        self.major = major
        self.minor = minor
        self.total_memory = total_memory


def _fake_torch(device_count: int, failing_indices: frozenset = frozenset()) -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.__version__ = "2.13.0"
    mod.version = types.SimpleNamespace(cuda="13.0")

    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: True
    cuda.get_arch_list = lambda: ["sm_89", "sm_90"]
    cuda.device_count = lambda: device_count
    cuda.get_device_properties = lambda i: _FakeProps(f"FakeGPU{i}", 8, 9, 4e10)

    def randn(*args, device=None, **kwargs):
        index = int(str(device).split(":")[1])
        if index in failing_indices:
            raise RuntimeError(f"no CUDA-capable device detected (device {index})")
        return _FakeTensor()

    cuda.randn = randn
    mod.cuda = cuda
    mod.randn = randn
    return mod


class TestCheckTorchMultiGpu(unittest.TestCase):
    def _check(self, device_count: int, failing_indices: frozenset = frozenset()):
        fake = _fake_torch(device_count, failing_indices)
        with unittest.mock.patch.dict(sys.modules, {"torch": fake}):
            return doctor.check_torch()

    def test_every_visible_device_gets_its_own_matmul(self):
        check = self._check(device_count=3)

        self.assertEqual(check.status, doctor.OK)
        joined = "\n".join(check.lines)
        for index in range(3):
            self.assertIn(f"real matmul on cuda:{index} OK", joined)

    def test_a_single_bad_device_among_several_good_ones_fails_the_check(self):
        """The case a device-0-only test would miss entirely."""
        check = self._check(device_count=3, failing_indices=frozenset({2}))

        self.assertEqual(check.status, doctor.FAIL)
        joined = "\n".join(check.lines)
        self.assertIn("real matmul on cuda:0 OK", joined)
        self.assertIn("real matmul on cuda:1 OK", joined)
        self.assertIn("real matmul on cuda:2 FAILED", joined)

    def test_a_single_gpu_box_still_works(self):
        check = self._check(device_count=1)
        self.assertEqual(check.status, doctor.OK)
        self.assertIn("real matmul on cuda:0 OK", "\n".join(check.lines))


if __name__ == "__main__":
    unittest.main()
