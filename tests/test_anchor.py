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

import json
import unittest

import numpy as np

from pipeline.dataset import Dataset
from pipeline.registry import get_step_class
from tests.helpers import require_stage, require_stage2

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

    def test_supplied_masks_survive_the_step(self):
        """A step handed somebody else's masks must not manufacture over them.

        The general form of the bug that cost a run's output quality: this
        step took no `masks` input at all and built an all-1.0 batch on
        every call, so wherever it was placed it destroyed whatever the
        mask field was carrying.

        A gradient, not a constant: an all-1.0 mask would pass a test that
        only checked "masks came out non-manufactured".
        """
        h, w = self.ds.images[0].shape[:2]
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, 0)
        supplied = [ramp * (i / len(self.ds.images)) for i in range(len(self.ds.images))]

        out = self._run(masks=[m.copy() for m in supplied])
        injected = [
            i for i, img in enumerate(out["images"]) if img is self.ds.anchor_image
        ]
        self.assertEqual(len(injected), 2, "premise: two frames sit at the anchor")
        for i, mask in enumerate(out["masks"]):
            if i in injected:
                continue
            np.testing.assert_array_equal(
                mask, supplied[i], f"frame {i}'s mask was not passed through"
            )

    def test_an_injected_frame_is_marked_keep_not_denoise(self):
        """0.0, uniform, whether or not masks were supplied.

        `dataset.masks` is the VACE mask everywhere in this pipeline — 1.0
        "synthetic, denoise this", 0.0 "a real photograph, keep it" — and
        an injected frame is by definition the real photograph. Pinned
        because the ComfyUI graph reads the other way round (its MASK is
        inverted and SaveDataset re-inverts on the way to disk), so
        reasoning from the node source instead of the recorded alpha
        produces 1.0 here and tells VACE to regenerate the one real frame
        in the batch.
        """
        h, w = self.ds.images[0].shape[:2]
        for masks in (None, [np.full((h, w), 0.5, np.float32) for _ in self.ds.images]):
            out = self._run(**({} if masks is None else {"masks": masks}))
            injected = [
                i for i, img in enumerate(out["images"]) if img is self.ds.anchor_image
            ]
            self.assertTrue(injected)
            for i in injected:
                with self.subTest(supplied_masks=masks is not None, frame=i):
                    self.assertEqual(float(out["masks"][i].min()), 0.0)
                    self.assertEqual(float(out["masks"][i].max()), 0.0)


class TestMaskThenInjectAgainstCyber2(unittest.TestCase):
    """The stage-2 -> stage-3 chain against the newer recorded run.

    cyber2_6f is the run with anchor injection live. Its masked_splatted is
    80 frames masked, composited over black and bilateral-filtered at a
    uniform alpha of 255, plus frame_00038_ which is that stage's
    anchor.png byte for byte at a uniform alpha of 0 — not composited, not
    filtered.

    Reproducing that requires mask_splat to run BEFORE inject_anchor. The
    other order was shipped, and it put inject_anchor where dataset.masks
    is carrying the splat render's per-pixel alpha, so the alpha mask_splat
    exists to threshold was overwritten with all-1.0 and the stage silently
    became a bilateral filter. `test_the_shipped_order_does_not_reproduce_it`
    puts a number on that; the YAML-level guard is in test_workflows.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.splatted, cls.expected = require_stage2("splatted", "masked_splatted")

    def _dataset(self):
        import cv2

        md = json.loads((self.splatted / "metadata.json").read_text())
        frames = sorted(self.splatted.glob("frame_*.png"))
        imgs = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in frames]

        class _Cam:
            def __init__(self, position):
                self.position = np.asarray(position, dtype=np.float32)

        return Dataset(
            images=[im[:, :, :3].copy() for im in imgs],
            image_names=[f.name for f in frames],
            cameras=[_Cam(c["extrinsics"]["position"]) for c in md["cameras"]],
            points_3d=None,
            resolution=tuple(md["resolution"]),
            masks=[im[:, :, 3].astype(np.float32) / 255.0 for im in imgs],
            anchor_image=cv2.imread(str(self.splatted / "anchor.png"), cv2.IMREAD_UNCHANGED),
            extras=md["b2c_extras"],
        )

    def _run_chain(self):
        ds = self._dataset()
        ds = get_step_class("mask_splat")().run(
            {"dataset": ds}, {"filter_size": 6, "dilation": 2}
        )["dataset"]
        out = get_step_class("inject_anchor")().run(
            {
                "images": ds.images,
                "cameras": ds.cameras,
                "masks": ds.masks,
                "anchor_position": ds.extras["anchor_position"],
                "anchor_image": ds.anchor_image,
            },
            {"tolerance_pct": 0.1},
        )
        return out["images"], out["masks"]

    def test_reproduces_the_recorded_stage(self):
        import cv2

        images, masks = self._run_chain()
        expected = sorted(self.expected.glob("frame_*.png"))
        self.assertEqual(len(images), len(expected))

        anchor_index = int(json.loads(
            (self.splatted / "metadata.json").read_text())["b2c_extras"]["anchor_frame_index"])

        for i, path in enumerate(expected):
            exp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            with self.subTest(frame=i + 1):
                mae = float(np.abs(images[i].astype(int) - exp[:, :, :3].astype(int)).mean())
                # The residual is the bilateral filter's border handling,
                # documented in steps/mask_splat.py; it never reaches 1/255.
                self.assertLess(mae, 1.0)
                # Uniform VACE mask, matching the recorded alpha exactly.
                self.assertEqual(float(masks[i].min()), float(masks[i].max()))
                self.assertEqual(round(float(masks[i].max()) * 255), int(exp[0, 0, 3]))

        self.assertEqual(anchor_index, 37, "premise: cyber2_6f anchors frame 38")

    def test_the_shipped_order_does_not_reproduce_it(self):
        """The same two steps the wrong way round, to put a number on it.

        inject_anchor first overwrites the splat alpha with an all-1.0
        batch, so mask_splat's keep-test passes on every pixel and nothing
        is ever blacked out. Asserted as a floor rather than an exact
        value: the point is the size of the gap, not its digits.
        """
        import cv2

        ds = self._dataset()
        out = get_step_class("inject_anchor")().run(
            {
                "images": ds.images,
                "cameras": ds.cameras,
                "anchor_position": ds.extras["anchor_position"],
                "anchor_image": ds.anchor_image,
            },
            {"tolerance_pct": 0.1},
        )
        ds.images, ds.masks = out["images"], out["masks"]
        ds = get_step_class("mask_splat")().run(
            {"dataset": ds}, {"filter_size": 6, "dilation": 2}
        )["dataset"]

        expected = sorted(self.expected.glob("frame_*.png"))
        errs, kept = [], []
        for i, path in enumerate(expected):
            exp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, :3]
            errs.append(float(np.abs(ds.images[i].astype(int) - exp.astype(int)).mean()))
            kept.append(float((ds.images[i].max(axis=2) > 8).mean()))

        # The correct order lands under 1/255 on every frame (above).
        self.assertGreater(max(errs), 50.0, "wrong order should diverge grossly")
        # Nothing masked: essentially the whole frame survives, against the
        # ~22% the recorded stage keeps.
        self.assertGreater(min(k for i, k in enumerate(kept) if i != 37), 0.9)

        # And the anchor frame, injected first, gets blacked out entirely:
        # its mask is uniform 0.0, so mask_splat's keep-test fails everywhere.
        self.assertEqual(int(ds.images[37].max()), 0)

    def test_the_anchor_frame_is_the_photo_verbatim(self):
        """Byte-exact, at alpha 0 — the single check that pins both the
        ordering and the mask convention at once. Composite it over black
        or bilateral-filter it and the bytes stop matching; mark it 1.0 and
        denoise_pass2 regenerates the only real frame in the batch."""
        import cv2

        images, masks = self._run_chain()
        anchor = cv2.imread(str(self.splatted / "anchor.png"), cv2.IMREAD_UNCHANGED)
        expected = cv2.imread(
            str(self.expected / "frame_00038_.png"), cv2.IMREAD_UNCHANGED
        )

        np.testing.assert_array_equal(expected[:, :, :3], anchor[:, :, :3])
        np.testing.assert_array_equal(images[37], anchor[:, :, :3])
        self.assertEqual(float(masks[37].max()), 0.0)
        self.assertEqual(int(expected[0, 0, 3]), 0)


if __name__ == "__main__":
    unittest.main()
