"""inject_anchor against the real anchor recorded in cyber_6f.

cyber_6f/initial carries a real anchor: extras["anchor_position"] is the
world origin (override_cam_from_mesh puts the original SAM-3D camera
there), and anchor.png is the image that was injected. Crucially, frames
1 and 81 of that dataset are both byte-identical to anchor.png — the
`overlap=1` case the module docstring describes, where the orbit closes on
itself and two cameras occupy the same position.

That makes this a golden test of the position-matching logic rather than a
synthetic one: the recorded data independently says which frames the
ComfyUI flow injected into, and the port has to find the same ones from
the camera positions alone.

**generate_firstlast is not covered here.** Its input is the single photo
SAM-3D-Body was run on, and that image is not preserved in the dataset —
reference.png is a two-panel front/back sheet used for Wan-VACE
conditioning, a different image with a different framing (the subject's
bounding box scales by 0.59 horizontally against 0.84 vertically, so no
uniform warp maps one to the other). So the warp itself stays verified
only against synthetic data until a real render runs on a GPU pod.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from tests.helpers import require_stage

import pipeline.steps  # noqa: F401


class TestInjectAnchorAgainstRecordedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.from_disk(require_stage("initial"))
        cls.anchor_position = np.asarray(cls.ds.extras["anchor_position"], dtype=np.float32)

    def _run(self, **overrides):
        inputs = {
            "images": self.ds.images,
            "cameras": self.ds.cameras,
            "anchor_position": self.anchor_position,
            "anchor_image": self.ds.anchor_image,
        }
        inputs.update(overrides)
        return get_step_class("inject_anchor")().run(inputs, {})

    def test_recorded_dataset_really_has_a_duplicated_anchor_frame(self):
        """The premise of the test below: two cameras at the anchor, and both
        of those frames already carry the anchor image."""
        positions = np.stack([c.position for c in self.ds.cameras])
        at_anchor = np.flatnonzero(
            np.linalg.norm(positions - self.anchor_position, axis=1) < 1e-6
        )
        self.assertEqual(at_anchor.tolist(), [0, 80])
        for idx in at_anchor:
            np.testing.assert_array_equal(self.ds.images[idx], self.ds.anchor_image)

    def test_injects_into_every_frame_at_the_anchor(self):
        out = self._run()
        injected = [
            i for i, img in enumerate(out["images"])
            if img is self.ds.anchor_image
        ]
        self.assertEqual(injected, [0, 80])

    def test_injected_frames_are_masked_zero(self):
        """Injected frames are reference material, not something to denoise."""
        out = self._run()
        for i, mask in enumerate(out["masks"]):
            if i in (0, 80):
                self.assertTrue(np.all(mask == 0.0), f"frame {i} should be masked 0")
            else:
                self.assertTrue(np.all(mask == 1.0), f"frame {i} should be masked 1")

    def test_survives_reordering_by_matching_on_position(self):
        """The durable key is the position, not the recorded index — the
        whole reason anchor_frame_index is called informational. After a
        rotate_views the index is meaningless but injection must still land
        on the same two cameras."""
        rotated = get_step_class("rotate_views")().run(
            {"dataset": self.ds}, {"start_azimuth_deg": 137.0}
        )["dataset"]

        out = get_step_class("inject_anchor")().run(
            {
                "images": rotated.images,
                "cameras": rotated.cameras,
                "anchor_position": self.anchor_position,
                "anchor_image": self.ds.anchor_image,
            },
            {},
        )
        injected = [
            i for i, img in enumerate(out["images"]) if img is self.ds.anchor_image
        ]
        self.assertEqual(len(injected), 2)
        for i in injected:
            np.testing.assert_allclose(
                rotated.cameras[i].position, self.anchor_position, atol=1e-6
            )

    def test_no_anchor_passes_through(self):
        out = self._run(anchor_image=None)
        self.assertIs(out["images"], self.ds.images)
        self.assertTrue(all(np.all(m == 1.0) for m in out["masks"]))

        out = self._run(anchor_position=None)
        self.assertIs(out["images"], self.ds.images)

    def test_shape_mismatch_raises(self):
        wrong = np.zeros((64, 64, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self._run(anchor_image=wrong)


if __name__ == "__main__":
    unittest.main()
