"""select_support_views — the face splat's second route into a training.

`render` composites the face splat onto the skeleton drawings, where two
diffusion passes then rewrite it. This step hands a SEPARATE render of the
same splat — a 30-degree cap of views the denoising path does not cover —
to brush as supporting views: training evidence that counts only where the
splat's own alpha says to, and is ignored everywhere else. See
steps/anchor_stub.py's class docstring and steps/brush.py's support_*
inputs.

Two things carry the weight here and are what these check. The frames kept
are a band whose inner edge is `min_path_angle_deg`, measured to the
nearest camera on the DENOISING PATH, because every frame on that path is a
denoised view of its own and a supporting view sitting on it competes with
one. (The OUTER edge is the cap's own radius, drawn by the sampler that
rendered these views — `render_splat` with `pattern: cap` — rather than
culled here. It used to be a `max_angle_deg` reading `composite_splat_views`'
per-frame verdict; that step is gone, see
docs/revert-when-body2colmap-drops-gsplat.md.) And the colour is
un-premultiplied, because brush's masked mode does not premultiply ground
truth and a `colour*a` frame would ask the model to be dark and
half-transparent along the silhouette rather than opaque and the right
colour.
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


class TestTheBatch(unittest.TestCase):
    def test_every_frame_is_kept_when_nothing_bands_them(self):
        """With no denoising path wired there is no edge to apply, which is
        right for a splat that is not a 2.5-D shell fed by a denoised orbit
        — and the reason `path_cameras` is optional rather than required."""
        self.assertEqual(len(_run(_batch(3))["images"]), 3)

    def test_mismatched_batch_lengths_are_refused(self):
        batch = _batch(3)
        batch["cameras"] = _cameras(2)
        with self.assertRaises(ValueError):
            _run(batch)


class TestTheBand(unittest.TestCase):
    """The inner edge: the DENOISING PATH.

    A band swept along it rather than a hole punched at the source view.
    The denoised batch covers a whole orbit — for `pattern: circular`, one
    elevation and 360 degrees of azimuth — and every frame on it is a
    denoised view in its own right, so a supporting view sitting on the
    path competes with one wherever along it it sits.

    The outer edge is not here: it is the cap's own radius, drawn by the
    sampler that rendered these views. What used to draw it on this side
    read `composite_splat_views`' per-frame verdict, and went with that
    step.
    """

    PIVOT = (0.0, 0.0, 0.0)

    def _camera_at(self, elevation_deg, azimuth_deg, radius=2.0):
        """A camera on the sphere about PIVOT, Y up — OrbitPath's frame."""
        from body2colmap.camera import Camera

        elevation, azimuth = np.radians([elevation_deg, azimuth_deg])
        position = np.array([
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
            radius * np.cos(elevation) * np.cos(azimuth),
        ], dtype=np.float32)
        return Camera(focal_length=(8.0, 8.0), image_size=(8, 8),
                      principal_point=(4.0, 4.0), position=position,
                      rotation=np.eye(3, dtype=np.float32))

    def _ring(self, elevation_deg=0.0, count=36):
        """The denoising path: a circular orbit at one elevation."""
        return [self._camera_at(elevation_deg, 360.0 * i / count)
                for i in range(count)]

    def _views(self, *frames, path=True, **params):
        """`frames` are (elevation, azimuth) pairs on the sphere."""
        batch = _batch(len(frames))
        batch["cameras"] = [self._camera_at(e, a) for e, a in frames]
        inputs = {**batch, "splat_center": self.PIVOT}
        if path:
            inputs["path_cameras"] = self._ring()
        return _run(inputs, **params)

    def _elevations(self, *elevations, **params):
        """Frames varying only in elevation, spread out in azimuth."""
        return self._views(*((e, 40.0 * i)
                             for i, e in enumerate(elevations)), **params)

    # -- the inner edge: the denoising path ------------------------------
    def test_a_view_on_the_path_is_dropped(self):
        self.assertEqual(self._elevations(0.0, 2.0, -4.9)["images"], [])

    def test_the_whole_path_counts_not_just_the_source_view(self):
        """The point of a band. This view is 140 degrees round the orbit
        from the photograph and nowhere near it — but the denoise ran on
        that azimuth too, at this elevation, so it is not ours to
        supervise."""
        out = self._views((1.0, 140.0))
        self.assertEqual(out["images"], [])

    def test_a_view_off_the_path_is_kept(self):
        self.assertEqual(len(self._elevations(12.0, -20.0)["images"]), 2)

    def test_the_band_is_two_sided(self):
        """Below the path is as much on it as above."""
        self.assertEqual(len(self._elevations(-3.0, -20.0)["images"]), 1)

    def test_the_inner_default_is_five_degrees(self):
        self.assertEqual(len(self._elevations(4.0, 6.0)["images"]), 1)

    def test_the_inner_edge_can_be_widened(self):
        out = self._elevations(4.0, 6.0, 20.0, min_path_angle_deg=10.0)
        self.assertEqual(len(out["images"]), 1)

    def test_zero_keeps_the_frames_on_the_path(self):
        out = self._elevations(0.0, 3.0, min_path_angle_deg=0.0)
        self.assertEqual(len(out["images"]), 2)

    def test_a_render_along_the_path_s_own_cameras_keeps_nothing(self):
        """What both workflows wire today (`render_splat` with
        `pattern: ""`): the support cameras ARE the path, so every one of
        them is zero degrees from it. Documented, logged, and not an
        error — brush trains without supporting views."""
        path = self._ring()
        batch = _batch(len(path))
        batch["cameras"] = list(path)
        out = _run({**batch, "path_cameras": path, "splat_center": self.PIVOT})
        self.assertEqual(out["images"], [])

    def test_a_helical_path_is_measured_the_same_way(self):
        """The distance is to the nearest path camera, so a swept path
        needs no second rule: this view is 3 degrees off the part of the
        helix it is nearest, and 8 degrees off the elevation the helix
        happens to start at."""
        helix = [self._camera_at(-10.0 + 20.0 * i / 35.0, 360.0 * i / 36.0)
                 for i in range(36)]
        batch = _batch(1)
        batch["cameras"] = [self._camera_at(-2.0, 80.0)]
        near_helix = _run({**batch, "path_cameras": helix,
                           "splat_center": self.PIVOT})
        self.assertEqual(near_helix["images"], [])

    def test_without_path_cameras_the_inner_edge_cannot_apply(self):
        out = self._elevations(0.0, 1.0, path=False)
        self.assertEqual(len(out["images"]), 2)

    def test_a_path_with_no_pivot_is_refused(self):
        """Measuring an angle needs something to measure it about, and for
        a head on a full-body orbit the target is the wrong point."""
        batch = _batch(1)
        batch["cameras"] = [self._camera_at(20.0, 0.0)]
        with self.assertRaises(ValueError) as caught:
            _run({**batch, "path_cameras": self._ring()})
        self.assertIn("select_support_views", str(caught.exception))

    def test_the_orbit_target_will_do_as_a_pivot(self):
        batch = _batch(1)
        batch["cameras"] = [self._camera_at(20.0, 0.0)]
        out = _run({**batch, "path_cameras": self._ring(),
                    "orbit_target": np.zeros(3)})
        self.assertEqual(len(out["images"]), 1)

    def test_a_view_off_the_path_survives_a_batch_with_views_on_it(self):
        """The band is a per-frame verdict, not a batch-level one."""
        out = self._views((1.0, 0.0), (15.0, 40.0), (-2.0, 80.0))
        self.assertEqual(len(out["images"]), 1)


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
        steps/render.py's `+splat` compositing has, for a different
        reason."""
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

        out = _run({**_batch(2)})
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


class TestTheDebugExportRecordsThem(unittest.TestCase):
    """`export_colmap_intermediate` has to be a record of what brush saw.

    A supporting view is a render made from a pose nothing re-measures, and
    it is produced earlier in the run than the training frames it
    supervises — so "did it land where its content belongs" is a question
    only this export can answer. It cannot answer it if the view is not in
    it.
    """

    def _export(self, tmp, **extra):
        step_class = get_step_class("colmap_export")
        inputs = {
            "cameras": _cameras(2),
            "image_names": ["frame_00001_.png", "frame_00002_.png"],
            "points_3d": (np.zeros((4, 3), dtype=np.float32),
                          np.zeros((4, 3), dtype=np.uint8)),
            "images": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
            "masks": [np.ones((8, 8), dtype=np.float32) for _ in range(2)],
            **extra,
        }
        return step_class().run(
            inputs,
            step_class.resolve_params({"output_dir": str(tmp), "layout": "brush"}),
        )

    def test_the_supporting_views_get_a_frame_a_matte_and_a_pose(self):
        import tempfile
        from pathlib import Path

        out = _run({**_batch(2)})
        with tempfile.TemporaryDirectory() as tmp:
            self._export(
                tmp,
                support_images=out["images"],
                support_masks=out["masks"],
                support_cameras=out["cameras"],
            )
            root = Path(tmp)
            files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
            self.assertIn("images/support_00001.png", files)
            self.assertIn("masks/support_00001.png", files)
            names = [line.split()[-1] for line in
                     (root / "images.txt").read_text().splitlines()
                     if line.strip() and not line.startswith("#")]
            self.assertEqual(
                names,
                ["frame_00001_.png", "frame_00002_.png",
                 "support_00001.png", "support_00002.png"],
            )

    def test_without_them_the_export_is_what_it_always_was(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            self._export(tmp)
            root = Path(tmp)
            self.assertFalse((root / "masks").exists())
            names = [line.split()[-1] for line in
                     (root / "images.txt").read_text().splitlines()
                     if line.strip() and not line.startswith("#")]
            self.assertEqual(names, ["frame_00001_.png", "frame_00002_.png"])

    def test_a_flat_layout_refuses_them_rather_than_dropping_the_mattes(self):
        import tempfile

        out = _run({**_batch(1)})
        step_class = get_step_class("colmap_export")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                step_class().run(
                    {
                        "cameras": _cameras(1),
                        "image_names": ["frame_00001_.png"],
                        "points_3d": (np.zeros((4, 3), dtype=np.float32),
                                      np.zeros((4, 3), dtype=np.uint8)),
                        "support_images": out["images"],
                        "support_masks": out["masks"],
                        "support_cameras": out["cameras"],
                    },
                    step_class.resolve_params(
                        {"output_dir": str(tmp), "layout": "flat"}),
                )
        self.assertIn("masks/", str(caught.exception))

    def test_a_supporting_view_with_a_different_lens_is_called_out(self):
        """ColmapExporter writes one camera line for the whole model, so a
        supporting view rendered through a different lens is exported as
        though it had the training one — and lands wherever that puts it."""
        import tempfile

        from body2colmap.camera import Camera

        out = _run({**_batch(1)})
        zoomed = [Camera(focal_length=(32.0, 32.0), image_size=(8, 8),
                         principal_point=(4.0, 4.0),
                         position=np.array([0.0, 0.0, 3.0], dtype=np.float32),
                         rotation=np.eye(3, dtype=np.float32))]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("pipeline.steps.brush", level="WARNING") as logs:
                self._export(
                    tmp,
                    support_images=out["images"],
                    support_masks=out["masks"],
                    support_cameras=zoomed,
                )
        self.assertIn("wrong place", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
