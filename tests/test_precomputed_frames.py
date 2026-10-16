"""seedvr2 and sapiens2_lite taking back frames made earlier in the run.

The orbit extension upscales pass 2's 81 frames and estimates their normal
maps at stage 4a for its guide training; the final upscale and normal maps
take them back by offset and only process the extension's new frames. The
models are stubbed: what is checked is which frames reach them and where
the reused ones land.
"""

from __future__ import annotations

import unittest

import numpy as np
from body2colmap.camera import Camera

from pipeline.steps.sapiens2 import Sapiens2LiteStep
from pipeline.steps.seedvr2 import SeedVR2Step


def _frames(count, size=(8, 4), start=0):
    width, height = size
    return [np.full((height, width, 3), start + i, dtype=np.uint8) for i in range(count)]


def _cameras(count, size=(8, 4)):
    width, height = size
    return [Camera(focal_length=(10.0, 10.0), image_size=(width, height),
                   principal_point=(width / 2.0, height / 2.0),
                   position=np.zeros(3, dtype=np.float32), rotation=np.eye(3, dtype=np.float32))
            for _ in range(count)]


class _StubUpscale(SeedVR2Step):
    """2x nearest-neighbour in place of the model, recording what it saw."""

    def __init__(self):
        super().__init__()
        self._dit_model = "stub"
        self.seen = []

    def _upscale(self, frames, params):
        self.seen.append([int(f[0, 0, 0]) for f in frames])
        return [np.repeat(np.repeat(f, 2, axis=0), 2, axis=1) for f in frames]


class TestSeedVR2Reuse(unittest.TestCase):
    def _params(self):
        return {name: param.default for name, param in
                SeedVR2Step.declared_params().items()}

    def test_only_the_frames_outside_the_block_are_upscaled(self):
        step = _StubUpscale()
        done = [np.full((8, 16, 3), 200 + i, dtype=np.uint8) for i in range(3)]
        out = step.run({"images": _frames(7), "cameras": _cameras(7),
                        "upscaled": done, "upscaled_offset": 2}, self._params())
        self.assertEqual(step.seen, [[0, 1, 5, 6]])
        self.assertEqual([int(f[0, 0, 0]) for f in out["images"]], [0, 1, 200, 201, 202, 5, 6])
        self.assertEqual(out["resolution"], (16, 8))
        self.assertTrue(all(c.width == 16 and c.fx == 20.0 for c in out["cameras"]))

    def test_without_a_block_every_frame_is_upscaled(self):
        step = _StubUpscale()
        out = step.run({"images": _frames(3), "cameras": _cameras(3),
                        "upscaled": None, "upscaled_offset": None}, self._params())
        self.assertEqual(step.seen, [[0, 1, 2]])
        self.assertEqual(len(out["images"]), 3)

    def test_a_block_of_another_size_is_refused(self):
        step = _StubUpscale()
        done = [np.zeros((12, 24, 3), dtype=np.uint8)]
        with self.assertRaisesRegex(ValueError, "not made by the same upscale"):
            step.run({"images": _frames(3), "cameras": _cameras(3),
                      "upscaled": done, "upscaled_offset": 1}, self._params())

    def test_a_block_past_the_end_is_refused(self):
        step = _StubUpscale()
        with self.assertRaisesRegex(ValueError, "do not fit"):
            step.run({"images": _frames(3), "cameras": _cameras(3),
                      "upscaled": _frames(2, (16, 8)), "upscaled_offset": 2}, self._params())


class _StubNormals(Sapiens2LiteStep):
    def __init__(self):
        super().__init__()
        self._model = "stub"
        self.seen = []

    def _run_batch(self, images_bgr):
        self.seen.append([int(f[0, 0, 0]) for f in images_bgr])
        return [np.full(f.shape, float(f[0, 0, 0]), dtype=np.float32) for f in images_bgr]


class TestSapiens2LiteReuse(unittest.TestCase):
    def _params(self, batch_size=2):
        params = {name: param.default for name, param in
                  Sapiens2LiteStep.declared_params().items()}
        params["batch_size"] = batch_size
        return params

    def test_only_the_frames_outside_the_block_are_estimated(self):
        step = _StubNormals()
        done = [np.full((4, 8, 3), -1.0, dtype=np.float32) for _ in range(2)]
        out = step.run({"images": _frames(5), "normal_maps": done, "normal_maps_offset": 1},
                       self._params())
        self.assertEqual(step.seen, [[0, 3], [4]])
        self.assertEqual([float(n[0, 0, 0]) for n in out["normal_maps"]], [0, -1, -1, 3, 4])

    def test_without_a_block_every_frame_is_estimated(self):
        step = _StubNormals()
        out = step.run({"images": _frames(3), "normal_maps": None, "normal_maps_offset": None},
                       self._params())
        self.assertEqual(step.seen, [[0, 1], [2]])
        self.assertEqual(len(out["normal_maps"]), 3)

    def test_a_map_of_another_size_is_refused(self):
        step = _StubNormals()
        with self.assertRaisesRegex(ValueError, "against the frame's"):
            step.run({"images": _frames(3), "normal_maps": [np.zeros((8, 16, 3), np.float32)],
                      "normal_maps_offset": 0}, self._params())


if __name__ == "__main__":
    unittest.main()
