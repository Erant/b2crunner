"""pick_rear_view — which frame, in which mode, on what ground.

Synthetic: cameras on a ring around an orbit target with the anchor at a
known index, frames that carry their own index in their pixels, and a
matte that covers a known region. See steps/reference_view.py.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from body2colmap.camera import Camera

from tests.helpers import run_step

import pipeline.steps  # noqa: F401

N = 24
TARGET = np.array([0.1, -0.2, 3.0], dtype=np.float32)


def _ring(anchor_index: int, radius: float = 2.0):
    """N cameras evenly round a ring in the XZ plane about TARGET, the
    anchor sitting at `anchor_index`; returns (cameras, anchor_position)."""
    cameras = []
    for i in range(N):
        theta = 2 * np.pi * (i - anchor_index) / N
        position = TARGET + radius * np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=np.float32)
        cameras.append(Camera((100.0, 100.0), (8, 8), position=position))
    return cameras, cameras[anchor_index].position


def _frames():
    """Frame i is flat at value i+1, with a dark border so a matte can be
    seen to cut something."""
    frames = []
    for i in range(N):
        frame = np.full((8, 8, 3), i + 1, dtype=np.uint8)
        frames.append(frame)
    return frames


def _masks():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    return [mask.copy() for _ in range(N)]


def _run(layout, anchor_index=5, reference=None, **params):
    cameras, anchor = _ring(anchor_index)
    inputs = {
        "images": _frames(), "masks": _masks(), "cameras": cameras,
        "anchor_position": anchor, "orbit_target": TARGET,
        "layout": layout, "reference_image": reference,
    }
    return run_step("pick_rear_view", inputs, params)


class TestSingleMode(unittest.TestCase):
    def test_the_frame_opposite_the_anchor_is_picked(self):
        for anchor_index in (0, 5, 17):
            with self.subTest(anchor=anchor_index):
                out = _run("single", anchor_index=anchor_index, matte=False)
                expected = (anchor_index + N // 2) % N
                self.assertEqual(out["rear_view_index"], expected)
                self.assertTrue(np.all(out["reference_image"] == expected + 1))

    def test_the_pick_is_measured_about_the_orbit_target_not_the_origin(self):
        """On a ring the camera furthest from the anchor in space IS the
        opposite one, so push a side camera far out along its own ray:
        its distance from the anchor now wins, its direction does not."""
        cameras, anchor = _ring(0)
        cameras[3].position = TARGET + 10.0 * (cameras[3].position - TARGET)
        far_in_space = int(np.argmax([np.linalg.norm(c.position - anchor) for c in cameras]))
        self.assertEqual(far_in_space, 3)
        out = run_step("pick_rear_view", {
            "images": _frames(), "masks": _masks(), "cameras": cameras,
            "anchor_position": anchor, "orbit_target": TARGET,
            "layout": "single", "reference_image": None,
        }, {"matte": False})
        self.assertEqual(out["rear_view_index"], N // 2)

    def test_the_matte_lays_the_frame_over_grey(self):
        out = _run("single", anchor_index=0)
        picked = out["reference_image"]
        self.assertEqual(picked.dtype, np.uint8)
        self.assertTrue(np.all(picked[2:6, 2:6] == N // 2 + 1))
        self.assertTrue(np.all(picked[0, 0] == 128), picked[0, 0])
        self.assertTrue(np.all(picked[7, 7] == 128))

    def test_bg_color_is_honoured(self):
        out = _run("single", anchor_index=0, bg_color=[1.0, 0.0, 0.0])
        # BGR out: red is (0, 0, 255).
        np.testing.assert_array_equal(out["reference_image"][0, 0], [0, 0, 255])

    def test_matte_needs_a_mask_per_frame(self):
        cameras, anchor = _ring(0)
        with self.assertRaises(ValueError) as caught:
            run_step("pick_rear_view", {
                "images": _frames(), "masks": None, "cameras": cameras,
                "anchor_position": anchor, "orbit_target": TARGET,
                "layout": "single", "reference_image": None,
            })
        self.assertIn("matte", str(caught.exception))

    def test_a_single_run_needs_the_anchor_and_the_target(self):
        cameras, anchor = _ring(0)
        for missing in ("anchor_position", "orbit_target"):
            with self.subTest(missing=missing):
                inputs = {
                    "images": _frames(), "masks": _masks(), "cameras": cameras,
                    "anchor_position": anchor, "orbit_target": TARGET,
                    "layout": "single", "reference_image": None,
                }
                inputs[missing] = None
                with self.assertRaises(ValueError):
                    run_step("pick_rear_view", inputs)

    def test_debug_dir_records_the_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _run("single", anchor_index=3, debug_dir=tmp)
            self.assertTrue((Path(tmp) / "rear_view.png").is_file())
            meta = json.loads((Path(tmp) / "rear_view.json").read_text())
            self.assertEqual(meta["rear_view_index"], out["rear_view_index"])
            self.assertAlmostEqual(meta["angle_from_anchor_deg"], 180.0, places=3)


class TestSheetMode(unittest.TestCase):
    def test_the_back_panel_passes_through_untouched(self):
        panel = np.full((16, 8, 3), 77, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            out = _run("sheet", reference=panel, debug_dir=tmp)
            self.assertIs(out["reference_image"], panel)
            self.assertIsNone(out["rear_view_index"])
            self.assertEqual(list(Path(tmp).iterdir()) if Path(tmp).exists() else [], [])

    def test_a_sheet_run_without_a_reference_is_a_wiring_bug(self):
        with self.assertRaises(ValueError) as caught:
            _run("sheet", reference=None)
        self.assertIn("back panel", str(caught.exception))

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaises(ValueError):
            _run("stereo", reference=np.zeros((4, 4, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
