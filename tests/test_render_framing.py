"""`override_cam_from_mesh` framing: a non-"full" preset must zoom, not just re-crop.

The bug this pins (identical to one in the ComfyUI-Body2COLMAP render node): the
override branch fed the whole mesh to `compute_original_view_framing`, so the
computed focal length always framed the entire body. Picking `torso`/`bust`/
`head` moved only `orbit_center` — the camera re-aimed, the subject stayed the
same size. `_framing_vertices` restricts the framing input to the preset's box.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.steps.render import _framing_vertices


class TestFramingVertices(unittest.TestCase):
    def setUp(self):
        # A tall thin "body": 0..2 in Y, narrow in X/Z. The top quarter
        # (Y >= 1.5) stands in for a "head" preset.
        rng = np.random.default_rng(0)
        self.verts = np.column_stack([
            rng.uniform(-0.2, 0.2, 400),
            rng.uniform(0.0, 2.0, 400),
            rng.uniform(-0.2, 0.2, 400),
        ]).astype(np.float32)
        self.full_bounds = (self.verts.min(axis=0), self.verts.max(axis=0))
        head = self.verts[self.verts[:, 1] >= 1.5]
        self.head_bounds = (head.min(axis=0), head.max(axis=0))

    def test_full_returns_every_vertex_untouched(self):
        out = _framing_vertices(self.verts, self.head_bounds, "full")
        self.assertIs(out, self.verts)

    def test_preset_selects_only_vertices_in_the_box(self):
        out = _framing_vertices(self.verts, self.head_bounds, "head")
        self.assertLess(len(out), len(self.verts))
        self.assertGreaterEqual(out[:, 1].min(), 1.5 - 1e-6)

    def test_preset_with_full_bounds_is_a_noop(self):
        # A preset whose bounds could not be computed reaches the helper as
        # the full bounds (the caller's `.get(framing, ...["full"])`).
        out = _framing_vertices(self.verts, self.full_bounds, "torso")
        self.assertEqual(len(out), len(self.verts))

    def test_degenerate_box_falls_back_to_the_whole_mesh(self):
        empty = (np.array([9, 9, 9], np.float32), np.array([9, 9, 9], np.float32))
        out = _framing_vertices(self.verts, empty, "head")
        self.assertIs(out, self.verts)


class TestFramingActuallyZooms(unittest.TestCase):
    """The regression itself: tighter preset -> longer focal length."""

    def test_head_preset_lengthens_the_framed_focal_length(self):
        try:
            from body2colmap.utils import compute_original_view_framing
        except Exception as exc:  # pragma: no cover - body2colmap optional here
            self.skipTest(f"body2colmap unavailable: {exc}")

        rng = np.random.default_rng(1)
        # In front of the camera (+Z looking down -Z in this convention would
        # project behind; use a positive depth that projects cleanly).
        verts = np.column_stack([
            rng.uniform(-0.3, 0.3, 500),
            rng.uniform(0.0, 1.8, 500),
            rng.uniform(3.0, 3.6, 500),
        ]).astype(np.float32)
        bounds_full = (verts.min(axis=0), verts.max(axis=0))
        head = verts[verts[:, 1] >= 1.35]
        bounds_head = (head.min(axis=0), head.max(axis=0))

        kw = dict(render_size=(720, 1280), original_focal_length=900.0, fill_ratio=0.8)
        fl_full = compute_original_view_framing(
            vertices=_framing_vertices(verts, bounds_full, "full"), **kw
        )["framed_focal_length"]
        fl_head = compute_original_view_framing(
            vertices=_framing_vertices(verts, bounds_head, "head"), **kw
        )["framed_focal_length"]

        self.assertGreater(fl_head, fl_full * 1.5)


if __name__ == "__main__":
    unittest.main()
