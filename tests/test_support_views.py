"""select_support_views — the face splat's second route into a training.

The composite puts the face-splat renders on the skeleton drawings, where
two diffusion passes then rewrite them. This step hands the SAME renders to
brush as supporting views: training evidence that counts only where the
splat's own alpha says to, and is ignored everywhere else. See
steps/anchor_stub.py's class docstring and steps/brush.py's support_*
inputs.

Two things carry the weight here and are what these check. The frames kept
are the ones `composite_splat_views` kept, because past its cull angle a
2.5-D shell shows its own open rim; and the colour is un-premultiplied,
because brush's masked mode does not premultiply ground truth and a
`colour*a` frame would ask the model to be dark and half-transparent along
the silhouette rather than opaque and the right colour.
"""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.registry import get_step_class

import pipeline.steps  # noqa: F401


def _cameras(count: int):
    from body2colmap.camera import Camera

    return [
        Camera(
            focal_length=(8.0, 8.0),
            image_size=(8, 8),
            principal_point=(4.0, 4.0),
            position=np.array([0.0, 0.0, float(i + 1)], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32),
        )
        for i in range(count)
    ]


def _render(alpha_value: float, colour: int = 200):
    """One splat render on black: colour premultiplied by its own alpha."""
    alpha = np.zeros((8, 8), dtype=np.float32)
    alpha[2:6, 2:6] = alpha_value
    image = (np.full((8, 8, 3), colour, dtype=np.float32) * alpha[..., None])
    return image.astype(np.uint8), alpha


def _roles(*roles):
    return [{"index": i, "angle_from_anchor_deg": 0.0, "role": role}
            for i, role in enumerate(roles)]


def _run(inputs, **params):
    step = get_step_class("select_support_views")()
    return step.run(inputs, get_step_class("select_support_views").resolve_params(params))


def _batch(count=3, alpha_value=1.0):
    images, masks = [], []
    for _ in range(count):
        image, alpha = _render(alpha_value)
        images.append(image)
        masks.append(alpha)
    return {"images": images, "masks": masks, "cameras": _cameras(count)}


class TestRoleSelection(unittest.TestCase):
    def test_only_the_composited_frames_are_kept(self):
        """The cull angle is measured once, by the composite. Re-deriving it
        here would be a second copy of it, free to drift."""
        out = _run({**_batch(3), "view_roles": _roles("base", "composited", "base")})
        self.assertEqual(len(out["images"]), 1)
        self.assertEqual(len(out["masks"]), 1)
        self.assertEqual(len(out["cameras"]), 1)

    def test_the_kept_camera_is_the_kept_frame_s(self):
        batch = _batch(3)
        out = _run({**batch, "view_roles": _roles("base", "composited", "base")})
        self.assertIs(out["cameras"][0], batch["cameras"][1])

    def test_without_roles_every_frame_is_kept(self):
        """Right for a splat that is not a 2.5-D shell, and the reason the
        input is optional rather than required."""
        self.assertEqual(len(_run(_batch(3))["images"]), 3)

    def test_an_empty_role_keeps_every_frame(self):
        out = _run({**_batch(3), "view_roles": _roles("base", "composited", "base")},
                   role="")
        self.assertEqual(len(out["images"]), 3)

    def test_no_match_is_empty_rather_than_an_error(self):
        """brush then trains with no supporting views, exactly as before —
        a warning, not a failed run an hour in."""
        out = _run({**_batch(2), "view_roles": _roles("base", "base")})
        self.assertEqual(out["images"], [])
        self.assertEqual(out["cameras"], [])

    def test_mismatched_roles_are_refused(self):
        with self.assertRaises(ValueError):
            _run({**_batch(3), "view_roles": _roles("composited", "composited")})

    def test_mismatched_batch_lengths_are_refused(self):
        batch = _batch(3)
        batch["cameras"] = _cameras(2)
        with self.assertRaises(ValueError):
            _run(batch)


class TestUnpremultiply(unittest.TestCase):
    def test_a_soft_edge_comes_back_to_its_true_colour(self):
        """The render is colour*a on black; brush's masked mode wants the
        straight colour, with the softness carried by the mask alone."""
        image, alpha = _render(0.5, colour=200)
        self.assertEqual(int(image[4, 4, 0]), 100)
        out = _run({"images": [image], "masks": [alpha], "cameras": _cameras(1)})
        self.assertEqual(int(out["images"][0][4, 4, 0]), 200)

    def test_an_opaque_interior_is_untouched(self):
        image, alpha = _render(1.0, colour=200)
        out = _run({"images": [image], "masks": [alpha], "cameras": _cameras(1)})
        self.assertEqual(int(out["images"][0][4, 4, 0]), 200)

    def test_it_can_be_turned_off(self):
        image, alpha = _render(0.5, colour=200)
        out = _run({"images": [image], "masks": [alpha], "cameras": _cameras(1)},
                   unpremultiply=False)
        self.assertEqual(int(out["images"][0][4, 4, 0]), 100)

    def test_transparent_pixels_stay_black(self):
        """1/255 divided by an alpha of 0.002 is noise amplified 500x, and
        the mask weights those pixels at zero anyway."""
        image, alpha = _render(0.5)
        image[0, 0] = 1
        out = _run({"images": [image], "masks": [alpha], "cameras": _cameras(1)})
        self.assertEqual(int(out["images"][0][0, 0, 0]), 0)
        self.assertEqual(float(out["masks"][0][0, 0]), 0.0)

    def test_a_render_on_a_non_black_background_is_refused(self):
        """Dividing by alpha only recovers the straight colour if the
        render was premultiplied over black — the same requirement
        composite_splat_views has, for a different reason."""
        image, alpha = _render(1.0)
        image[image.sum(axis=2) == 0] = 60
        with self.assertRaises(ValueError) as caught:
            _run({"images": [image], "masks": [alpha], "cameras": _cameras(1)})
        self.assertIn("select_support_views", str(caught.exception))


class TestTheseGoStraightIntoBrush(unittest.TestCase):
    """The output shape is brush's support_* input shape — the point of the
    step is that the two need no adapter between them."""

    def test_a_brush_export_takes_them_as_masked_views(self):
        import tempfile
        from pathlib import Path

        out = _run({**_batch(2), "view_roles": _roles("composited", "composited")})
        seen = {}

        step_class = get_step_class("brush")
        step = step_class()

        def fake_run_brush(cmd, ply_path, colmap_dir=None):
            seen["cmd"] = list(cmd)
            seen["files"] = sorted(
                str(p.relative_to(colmap_dir))
                for p in Path(colmap_dir).rglob("*") if p.is_file()
            )
            Path(ply_path).write_text("ply\n")

        step._run_brush = fake_run_brush
        inputs = {
            "cameras": _cameras(2),
            "image_names": ["frame_00001_.png", "frame_00002_.png"],
            "points_3d": (np.zeros((4, 3), dtype=np.float32),
                          np.zeros((4, 3), dtype=np.uint8)),
            "images": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
            "masks": [np.ones((8, 8), dtype=np.float32) for _ in range(2)],
            "support_images": out["images"],
            "support_masks": out["masks"],
            "support_cameras": out["cameras"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            step.run(inputs, step_class.resolve_params({"export_dir": tmp}))

        self.assertIn("masks/support_00001.png", seen["files"])
        self.assertIn("masks/support_00002.png", seen["files"])
        self.assertNotIn("--alpha-mode", seen["cmd"])
        self.assertIn("--normalize-masked-loss", seen["cmd"])


if __name__ == "__main__":
    unittest.main()
