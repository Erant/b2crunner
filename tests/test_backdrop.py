"""The world-fixed environment behind both renderers' frames.

What is actually this project's here, and therefore what is pinned: where
the backdrop's surface is put (`orbit_frame`), how a step's params become
one (`build_background`), and how it gets behind frames an external
rasteriser already composited over a flat colour (`composite_bgr`). The
textures, the ray intersection and the parallax are body2colmap's and are
tested there.

Two properties carry most of the weight. The first is that the ALPHA IS
NEVER TOUCHED: this pipeline carries an image and its mask separately, so a
backdrop that filled the mask in would hand every consumer downstream a
subject the size of the frame. The second is that the face-cap render still
comes back premultiplied over black — `select_support_views` divides that
alpha back out, and a room behind the splat would be recovered as the
face's own colour.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.registry import get_step_class
from pipeline.steps.backdrop import (
    BACKGROUND_FADE_PARAMS,
    BACKGROUND_PARAMS,
    build_background,
    build_fade,
    composite_bgr,
    orbit_frame,
)
from tests.helpers import stub_render_binary, run_step

import pipeline.steps  # noqa: F401


TARGET = np.array([0.1, 1.4, -2.0], dtype=np.float32)
RADIUS = 2.5


def _camera_template(width=8, height=8):
    from body2colmap.camera import Camera

    return Camera(focal_length=(float(width), float(width)),
                  image_size=(width, height))


def _path_gen():
    from body2colmap.path import OrbitPath

    return OrbitPath(target=TARGET, radius=RADIUS)


def _rs_params(**overrides):
    """render_splat's declared defaults with `overrides` on top."""
    return get_step_class("render_splat").resolve_params(overrides)


def _r_params(**overrides):
    """render's declared defaults with `overrides` on top.

    `n_frames` has no default — it is the one knob every workflow states —
    so it is supplied here to get at the rest.
    """
    return get_step_class("render").resolve_params({"n_frames": 4, **overrides})


class TestOrbitFrame(unittest.TestCase):
    """Recovering where a camera path looks, from the cameras alone.

    Every path this pipeline builds aims its cameras at one point, so that
    point is recoverable exactly. Doing it this way rather than threading
    the target down from each branch is what makes the two steps agree, and
    what covers `render_splat`'s reuse-the-dataset's-cameras mode, which
    never computes an orbit at all.
    """

    def _check(self, cameras):
        center, radius = orbit_frame(cameras)
        np.testing.assert_allclose(center, TARGET, atol=1e-4)
        self.assertAlmostEqual(radius, RADIUS, places=4)

    def test_a_circular_orbit_gives_back_its_target_and_radius(self):
        self._check(_path_gen().circular(n_frames=12, camera_template=_camera_template()))

    def test_a_helical_orbit_does_too(self):
        self._check(_path_gen().helical(
            n_frames=24, n_loops=2, amplitude_deg=30.0,
            camera_template=_camera_template()))

    def test_a_cap_does_too_though_it_is_not_an_orbit(self):
        """The face-support pattern: a narrow disc of views, not a sweep.

        The rays still converge on the splat's centre, which is the only
        property the recovery needs — an arc a few degrees wide would defeat
        an approach that averaged positions instead.
        """
        from pipeline.steps.splat import _cap_path

        cameras = _cap_path(
            _path_gen(),
            _rs_params(n_frames=16, cap_radius_deg=20.0, pattern="cap"),
            _camera_template(),
            {},
        )
        self._check(cameras)

    def test_no_cameras_is_refused(self):
        with self.assertRaises(ValueError):
            orbit_frame([])

    def test_a_camera_sitting_on_its_own_target_is_refused(self):
        """There is no radius to size a backdrop against, and body2colmap
        would otherwise be handed radius 0 and raise about the wrong thing."""
        from body2colmap.camera import Camera

        camera = Camera(focal_length=(8.0, 8.0), image_size=(8, 8),
                        position=np.zeros(3, dtype=np.float32))
        camera.look_at(np.array([0.0, 0.0, -1.0], dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            orbit_frame([camera, camera])
        self.assertIn("no orbit radius", str(caught.exception))


class TestBuildBackground(unittest.TestCase):
    """A step's resolved params into a `Background`."""

    def setUp(self):
        self.cameras = _path_gen().circular(
            n_frames=8, camera_template=_camera_template())

    def test_an_empty_texture_is_no_backdrop_at_all(self):
        """The off switch, and the one the face-cap render has to use."""
        self.assertIsNone(build_background(_rs_params(background=""), self.cameras))
        self.assertIsNone(build_background(_rs_params(background="  "), self.cameras))

    def test_the_default_is_a_grid_cube_around_the_orbit(self):
        background = build_background(_rs_params(), self.cameras)
        self.assertEqual(background.geometry, "cube")
        self.assertAlmostEqual(background.radius, 3.0 * RADIUS, places=4)
        np.testing.assert_allclose(background.center, TARGET, atol=1e-4)

    def test_the_alpha_channel_is_never_filled_in(self):
        """body2colmap defaults `opaque=True` because its frames ARE the
        deliverable. Here the alpha is the mask every downstream step reads
        as the subject's silhouette, so this must be False for every set of
        params a workflow can write."""
        for params in (_rs_params(), _rs_params(background="checker"),
                       _rs_params(background_geometry="sphere"),
                       _rs_params(background_radius_scale=None)):
            self.assertFalse(build_background(params, self.cameras).opaque)

    def test_an_empty_radius_scale_puts_the_surface_at_infinity(self):
        background = build_background(
            _rs_params(background_radius_scale=None), self.cameras)
        self.assertIsNone(background.radius)

    def test_an_explicit_radius_supersedes_the_defaulted_scale(self):
        """A workflow that writes only `background_radius` never wrote the
        scale, so the two are not a conflict — body2colmap's own config
        loader takes the same view, for the same reason."""
        background = build_background(
            _rs_params(background_radius=40.0), self.cameras)
        self.assertAlmostEqual(background.radius, 40.0, places=4)

    def test_a_radius_that_does_not_enclose_the_orbit_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_background(_rs_params(background_radius=RADIUS / 2.0), self.cameras)
        self.assertIn("enclose", str(caught.exception))

    def test_a_scale_inside_the_orbit_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_background(_rs_params(background_radius_scale=1.0), self.cameras)
        self.assertIn("> 1.0", str(caught.exception))

    def test_a_texture_that_is_neither_a_generator_nor_a_path_is_refused(self):
        with self.assertRaises(ValueError):
            build_background(_rs_params(background="gird"), self.cameras)


class TestBackgroundParams(unittest.TestCase):
    """The backdrop's own appearance — its colours above all.

    Mostly handed to the generator whole rather than unpacked into a param
    each, because the four generators name their colours differently (grid's
    base/line/floor/ceiling, checker's a/b, gradient's top/bottom, the sky's
    zenith/horizon/ground) and each carries non-colour knobs beside them. A
    table of that here would be a second copy of body2colmap's signatures,
    stale the moment one gains an argument — so what is pinned is that the
    values arrive, and that a wrong key is refused by name.

    TWO are promoted out of the mapping and into params of their own: the
    wall and grid's ruling, which are the pair a person tunes and the pair
    that decide whether the silhouette reads against the room. Both fold
    back into the same mapping in `build_background`, so what is pinned for
    them is that they arrive by either spelling, that neither is sent when
    unset (which is what keeps a non-grid generator runnable on its
    defaults), and that setting one twice is refused rather than resolved.
    """

    def setUp(self):
        self.cameras = _path_gen().circular(
            n_frames=4, camera_template=_camera_template(64, 64))

    def _frame(self, **overrides):
        """One rendered view of the environment, as int16 for differencing."""
        background = build_background(_rs_params(**overrides), self.cameras)
        return background.render(self.cameras[0]).astype(np.int16)

    def test_recolouring_the_walls_changes_the_render(self):
        """Measured over the whole frame rather than at a pixel: the texture
        is resampled to suit the camera, so any single pixel may have landed
        on a blurred grid line."""
        def red_excess(frame):
            return float(np.median(frame[..., 0] - frame[..., 2]))

        self.assertLess(red_excess(self._frame()), 0.0)
        self.assertGreater(
            red_excess(self._frame(background_params={"base_color": [0.9, 0.1, 0.1]})),
            150.0)

    def test_the_promoted_wall_colour_lands_where_the_mapping_would(self):
        """`background_base_color` is the same wall, reached without having
        to know it is spelled `base_color` inside a YAML mapping."""
        def red_excess(frame):
            return float(np.median(frame[..., 0] - frame[..., 2]))

        promoted = self._frame(background_base_color=[0.9, 0.1, 0.1])
        mapped = self._frame(background_params={"base_color": [0.9, 0.1, 0.1]})
        self.assertGreater(red_excess(promoted), 150.0)
        np.testing.assert_array_equal(promoted, mapped)

    def test_the_promoted_line_colour_repaints_the_ruling(self):
        """The other half of the pair. Grid lines are the brightest thing in
        the default room, so darkening them to the wall drops the frame's
        maximum to about the wall itself."""
        default_max = int(self._frame().max())
        darkened = self._frame(background_line_color=[0.42, 0.44, 0.48])
        self.assertGreater(default_max, 180)
        # Painted the wall's own colour, the ruling stops being the brightest
        # thing in the room: the frame's maximum drops to the wall.
        self.assertLess(int(darkened.max()), 130)

    def test_an_unset_promoted_colour_is_not_sent_at_all(self):
        """The reason both default to None rather than to grid's own values:
        `base_color` and `line_color` are grid-only, and a key that was
        always sent would refuse every other generator by name. checker has
        to build on nothing but the defaults."""
        for texture in ("checker", "gradient", "blender_sky"):
            with self.subTest(texture=texture):
                background = build_background(
                    _rs_params(background=texture), self.cameras)
                # It built at all, which is the whole claim: passing a
                # grid-only key here is what body2colmap refuses by name.
                self.assertIsNotNone(background)
                self.assertEqual(
                    background.render(self.cameras[0]).shape[-1], 3)

    def test_setting_a_promoted_colour_by_both_spellings_is_refused(self):
        """Silently preferring one would leave the other reading as the
        room's colour in a UI that is not describing the run."""
        for name, key in (("background_base_color", "base_color"),
                          ("background_line_color", "line_color")):
            with self.subTest(param=name):
                with self.assertRaises(ValueError) as caught:
                    build_background(
                        _rs_params(**{name: [0.1, 0.2, 0.3]},
                                   background_params={key: [0.4, 0.5, 0.6]}),
                        self.cameras)
                message = str(caught.exception)
                self.assertIn(key, message)
                self.assertIn(name, message)

    def test_a_promoted_colour_is_still_validated_by_body2colmap(self):
        """Not re-checked here: `_as_rgb` already refuses a wrong length and
        an out-of-range component, by value."""
        for bad in ([0.5, 0.5], [1.5, 0.0, 0.0]):
            with self.subTest(colour=bad):
                with self.assertRaises(ValueError):
                    build_background(
                        _rs_params(background_base_color=bad), self.cameras)

    def test_a_generator_that_names_its_colours_differently_works_too(self):
        """checker takes color_a/color_b, not base_color — the reason these
        are passed through instead of mapped onto names of this project's
        own invention."""
        red_blue = self._frame(
            background="checker",
            background_params={"color_a": [1.0, 0.0, 0.0],
                               "color_b": [0.0, 0.0, 1.0]})
        self.assertEqual(int(red_blue[..., 1].max()), 0)

    def test_a_non_colour_knob_goes_through_the_same_way(self):
        """It is the generator's whole keyword surface, not a colour hatch.
        Grid lines are the brightest thing in the room, so widening them
        lifts the whole frame."""
        self.assertGreater(
            self._frame(background_params={"line_width": 0.4}).mean(),
            self._frame().mean() + 30.0)

    def test_a_key_the_generator_does_not_take_is_refused_by_name(self):
        """body2colmap's own check, kept rather than duplicated: it names
        every argument the generator does accept, which is the message
        somebody who guessed `wall_color` actually needs."""
        with self.assertRaises(ValueError) as caught:
            build_background(
                _rs_params(background_params={"wall_color": [0, 0, 0]}),
                self.cameras)
        message = str(caught.exception)
        self.assertIn("wall_color", message)
        self.assertIn("base_color", message)

    def test_params_alongside_a_loaded_texture_are_refused(self):
        """A file has no parameters to take, so they would silently do
        nothing rather than fail."""
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "equirect.png"
            cv2.imwrite(str(path), np.zeros((32, 64, 3), np.uint8))
            with self.assertRaises(ValueError) as caught:
                build_background(
                    _rs_params(background=str(path),
                               background_geometry="sphere",
                               background_params={"base_color": [0, 0, 0]}),
                    self.cameras)
        self.assertIn("not a generator", str(caught.exception))

    def test_the_default_is_empty_and_leaves_the_generator_alone(self):
        declared = get_step_class("render").declared_params()
        self.assertEqual(declared["background_params"].default, {})
        self.assertIs(declared["background_params"].type, dict)
        # The promoted pair say the same thing with None: not sent at all.
        # They are lists so they draw as `[0.5, 0.5, 0.5]`, the shape every
        # other colour in this pipeline is written in.
        for name in ("background_base_color", "background_line_color"):
            self.assertIsNone(declared[name].default, name)
            self.assertIs(declared[name].type, list, name)


class TestCompositeBgr(unittest.TestCase):
    """Getting a backdrop behind frames that already have a flat one.

    `brush-splat-render` draws one colour and knows nothing of
    environments, so what `render_splat` gets back is `C*a + flat*(1-a)`.
    Recovering `C` and re-compositing cancels to a single add, and the test
    is that it lands on the same pixels a straight composite would.
    """

    def setUp(self):
        self.cameras = _path_gen().circular(
            n_frames=3, camera_template=_camera_template())
        self.background = build_background(_rs_params(), self.cameras)

    def _frames(self, straight_bgr, alpha, flat):
        """`straight_bgr` composited over `flat`, as the binary would."""
        flat_bgr = np.array([c * 255.0 for c in reversed(flat)], dtype=np.float32)
        images, masks = [], []
        for _ in self.cameras:
            a = np.full((8, 8), float(alpha), dtype=np.float32)
            rgb = straight_bgr * a[..., None] + flat_bgr * (1.0 - a[..., None])
            images.append(rgb.round().astype(np.uint8))
            masks.append(a)
        return images, masks

    def _expected(self, straight_bgr, alpha, camera):
        env = self.background.render(camera)[..., ::-1].astype(np.float32)
        return straight_bgr * alpha + env * (1.0 - alpha)

    def test_a_frame_over_black_lands_where_a_direct_composite_would(self):
        straight = np.array([20.0, 180.0, 90.0], dtype=np.float32)
        images, masks = self._frames(straight, 0.4, (0.0, 0.0, 0.0))
        out = composite_bgr(images, masks, background=self.background,
                            cameras=self.cameras, flat_color=(0.0, 0.0, 0.0))
        for frame, camera in zip(out, self.cameras):
            np.testing.assert_allclose(
                frame.astype(np.float32),
                self._expected(straight, 0.4, camera), atol=1.5)

    def test_a_frame_over_the_confidence_cull_grey_does_too(self):
        """Confidence mode never passes `bg_color` to the binary at all —
        `cull_color` is both the background and the reject colour — so the
        colour being displaced is the one this is told about, not the one
        the step declares."""
        straight = np.array([200.0, 30.0, 60.0], dtype=np.float32)
        images, masks = self._frames(straight, 0.25, (0.5, 0.5, 0.5))
        out = composite_bgr(images, masks, background=self.background,
                            cameras=self.cameras, flat_color=(0.5, 0.5, 0.5))
        for frame, camera in zip(out, self.cameras):
            np.testing.assert_allclose(
                frame.astype(np.float32),
                self._expected(straight, 0.25, camera), atol=1.5)

    def test_a_fully_covered_frame_is_left_exactly_alone(self):
        straight = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        images, masks = self._frames(straight, 1.0, (0.0, 0.0, 0.0))
        before = [frame.copy() for frame in images]
        out = composite_bgr(images, masks, background=self.background,
                            cameras=self.cameras, flat_color=(0.0, 0.0, 0.0))
        for frame, original in zip(out, before):
            np.testing.assert_array_equal(frame, original)

    def test_the_masks_come_back_untouched(self):
        """The whole reason `opaque` is False: the mask is the splat's own
        coverage, and brush, mask_splat and colmap_export all read it as
        the subject."""
        images, masks = self._frames(
            np.array([50.0, 50.0, 50.0], dtype=np.float32), 0.3, (0.0, 0.0, 0.0))
        before = [mask.copy() for mask in masks]
        composite_bgr(images, masks, background=self.background,
                      cameras=self.cameras, flat_color=(0.0, 0.0, 0.0))
        for mask, original in zip(masks, before):
            np.testing.assert_array_equal(mask, original)


class TestBothStepsDeclareIt(unittest.TestCase):
    """One declaration, spliced into both renderers.

    `render` draws what the first denoise pass sees and `render_splat` what
    the second one sees. A person tunes one set of controls, and a
    workflow's `${globals.background}` reaches both by the same name.
    """

    def test_render_and_render_splat_declare_the_same_knobs(self):
        declared = {
            name: get_step_class(name).declared_params()
            for name in ("render", "render_splat")
        }
        for param in BACKGROUND_PARAMS:
            for name, params in declared.items():
                self.assertIn(param.name, params, name)
                self.assertIs(params[param.name], param, f"{name}.{param.name}")

    def test_the_default_is_a_grid_box(self):
        params = get_step_class("render").declared_params()
        self.assertEqual(params["background"].default, "grid")
        self.assertEqual(params["background_geometry"].default, "cube")
        self.assertEqual(params["background_radius_scale"].default, 3.0)


class TestRenderStepWiring(unittest.TestCase):
    """`render` hands the backdrop to the Renderer rather than compositing.

    Which matters for one reason: `render_composite` is the only thing that
    knows where the base layer ends and the skeleton overlay begins. The
    overlay writes RGB without touching alpha, so a backdrop composited
    after it would blend the skeleton away everywhere off the mesh.

    Pyrender needs a GPU/headless-GL setup this test environment has not
    got, so the Renderer is replaced by a recorder. That still pins the two
    things that are this step's: which cameras the backdrop is measured
    against, and that a single-mode render (no overlay, so no
    `render_composite`) gets composited too.
    """

    def setUp(self):
        import trimesh

        from pipeline.steps import render as render_module

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        rng = np.random.default_rng(3)
        self.mesh_output = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
            "cam_t": np.array([0.0, 0.0, 3.0], dtype=np.float32),
            "keypoints_3d": (rng.normal(size=(70, 3)) * 0.3).astype(np.float32),
            "focal_length": 1000.0,
        }
        self.render_module = render_module
        self.recorder = _RecordingRenderer.install(self)

    def _run(self, **params):
        step = get_step_class("render")()
        return step.run(
            {"mesh_output": self.mesh_output},
            get_step_class("render").resolve_params(
                {"n_frames": 4, "resolution": [8, 8], **params}),
        )

    def test_the_backdrop_is_measured_against_the_cameras_it_renders(self):
        result = self._run(render_mode="mesh")
        background = self.recorder.background
        self.assertIsNotNone(background)
        center, radius = orbit_frame(result["cameras"])
        np.testing.assert_allclose(background.center, center, atol=1e-4)
        self.assertAlmostEqual(background.radius, 3.0 * radius, places=3)

    def test_an_empty_background_leaves_the_renderer_without_one(self):
        self._run(render_mode="mesh", background="")
        self.assertIsNone(self.recorder.background)

    def test_a_single_mode_render_is_composited_by_the_step(self):
        """`render_composite` draws the backdrop under its own base layer,
        but `mesh`/`depth`/`skeleton` never go through it — so the step has
        to, and a missing call is invisible until someone looks at a frame."""
        self._run(render_mode="mesh")
        self.assertEqual(self.recorder.composited, 4)

    def test_a_composite_mode_is_not_composited_twice(self):
        self._run(render_mode="mesh+skeleton")
        self.assertEqual(self.recorder.composited, 0)


class _RecordingRenderer:
    """A stand-in for body2colmap's `Renderer` that draws nothing."""

    def __init__(self, scene, render_size, background=None):
        self.scene = scene
        self.width, self.height = render_size
        self.background = background
        self.composited = 0
        # Every draw call's kwargs, in order — what the step asked for, which
        # is the only thing a test can see when nothing is rasterized.
        self.calls = []

    @classmethod
    def install(cls, test):
        """Patch the Renderer `render.py` imports inside `run()`, and hand
        the test back the one instance it will build."""
        import body2colmap.renderer as renderer_module

        made = []

        def factory(scene, render_size, background=None):
            instance = cls(scene, render_size, background=background)
            made.append(instance)
            return instance

        original = renderer_module.Renderer
        renderer_module.Renderer = factory
        test.addCleanup(setattr, renderer_module, "Renderer", original)
        return _LazyRenderer(made)

    def _blank(self):
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        image[..., 3] = 255
        return image

    def render_mesh(self, **kwargs):
        return self._blank()

    def render_depth(self, **kwargs):
        return self._blank()

    def render_skeleton(self, **kwargs):
        self.calls.append(("render_skeleton", kwargs))
        return self._blank()

    def render_composite(self, **kwargs):
        # The real one composites the backdrop under its base layer itself;
        # counting it here would hide a double composite rather than catch it.
        self.calls.append(("render_composite", kwargs))
        return self._blank()

    def composite_over_background(self, image, camera):
        self.composited += 1
        return image


class _LazyRenderer:
    """The recorder the step builds, once it has built it."""

    def __init__(self, made):
        self._made = made

    def __getattr__(self, name):
        if not self._made:
            raise AssertionError("the render step never built a Renderer")
        return getattr(self._made[-1], name)


class TestRenderSplatBackdrop(unittest.TestCase):
    """The whole step, against the stub rasteriser.

    The stub is told to render at half alpha, because a fully opaque frame
    hides every background question there is — which is also why the
    default stub could not have caught this.
    """

    def _run(self, *, alpha=128, **params):
        from tests.test_splat import _synthetic_scene

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply = root / "s.ply"
            scene = _synthetic_scene()
            run_step("save_splat", {"splat_scene": scene}, {"filepath": str(ply)})
            binary = stub_render_binary(root, alpha=alpha)
            return run_step(
                "render_splat",
                {"splat_scene": scene, "splat_path": str(ply)},
                {"pattern": "circular", "n_frames": 3, "width": 8, "height": 8,
                 "bounds_source": "splat", "render_path": binary, **params},
            )

    def test_the_default_puts_a_room_behind_the_splat(self):
        """The stub renders one flat colour over black, so anything the
        frames gain over the same run with no backdrop is the room.

        Compared against that run rather than measured within a frame: at
        8x8 the whole view can land inside one cell of the grid, so a
        correct backdrop is a colour shift and not necessarily a texture.
        """
        with_room = self._run()
        without = self._run(background="")
        for a, b in zip(with_room["images"], without["images"]):
            self.assertTrue((a != b).any())

    def test_turning_it_off_gives_back_the_flat_frames(self):
        """What every render feeding select_support_views needs, and what
        every run before this change produced."""
        result = self._run(background="")
        for image in result["images"]:
            np.testing.assert_array_equal(image, np.full_like(image, image[0, 0]))

    def test_the_masks_are_the_same_either_way(self):
        with_room = self._run()
        without = self._run(background="")
        for a, b in zip(with_room["masks"], without["masks"]):
            np.testing.assert_array_equal(a, b)

    def test_a_fully_covered_render_is_unchanged_by_it(self):
        """Alpha 255 everywhere means there is nothing behind the splat to
        see, so the backdrop must cost the frame nothing."""
        with_room = self._run(alpha=255)
        without = self._run(alpha=255, background="")
        for a, b in zip(with_room["images"], without["images"]):
            np.testing.assert_array_equal(a, b)


class TestSubjectFade(unittest.TestCase):
    """The room pulling back around the subject — stage 1 only.

    What is this project's here: that the knobs exist on `render` and NOT on
    `render_splat`, that their defaults are the ones chosen for the drawings
    (smoothstep, margin 2, falloff 1), that the shell is fitted in the world
    frame the cameras live in, and that no backdrop means no fade rather than
    a refusal. The ellipsoid fit, the decay profiles and the plain-texture
    reveal are body2colmap's and are tested there.

    The margin of 2 is the one default worth stating twice: the shell is
    fitted to a NAKED SAM-3D-Body mesh, and the subject the denoise should
    produce is a dressed person with hair. A shell that stopped at the mesh's
    own hull would clear nothing where it matters.
    """

    def setUp(self):
        self.cameras = _path_gen().circular(
            n_frames=4, camera_template=_camera_template(64, 64))
        # A small subject at the point the cameras converge on, so the clear
        # zone lands in the middle of the frame and its band does not reach
        # the corners: radius 0.2 inflated to 0.4 by the default margin and
        # fading out by 0.8, against a half-width of ~1.25 at that distance.
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
        self.vertices = (
            np.asarray(sphere.vertices, dtype=np.float32) + TARGET)

    # -- declaration --------------------------------------------------------

    def test_only_the_mesh_render_declares_the_fade(self):
        """Stage 2 has no fade at all, and this is where that is decided.

        `render_splat` never splices these in, so a workflow that tried to
        set one on `rerender_splat` is refused by name rather than quietly
        rendering without it."""
        fade_names = {param.name for param in BACKGROUND_FADE_PARAMS}
        self.assertTrue(fade_names.isdisjoint(
            {param.name for param in BACKGROUND_PARAMS}))
        self.assertTrue(fade_names.issubset(
            {param.name for param in get_step_class("render").PARAMS}))
        self.assertTrue(fade_names.isdisjoint(
            {param.name for param in get_step_class("render_splat").PARAMS}))

    def test_the_defaults_are_the_ones_the_drawings_were_tuned_for(self):
        params = _r_params()
        self.assertEqual(params["background_fade"], "smoothstep")
        self.assertEqual(params["background_fade_margin"], 2.0)
        self.assertEqual(params["background_fade_falloff"], 1.0)
        # The lines go to the WALL, not to a flat patch or a smear.
        self.assertEqual(params["background_fade_target"], "plain")

    # -- building -----------------------------------------------------------

    def test_the_default_params_build_a_fade(self):
        fade = build_fade(_r_params(), self.vertices)
        self.assertIsNotNone(fade)
        self.assertEqual(fade.profile, "smoothstep")
        self.assertEqual(fade.target, "plain")
        self.assertAlmostEqual(fade.falloff, 1.0)

    def test_an_empty_profile_is_the_off_switch(self):
        self.assertIsNone(build_fade(_r_params(background_fade=""), self.vertices))
        self.assertIsNone(build_fade(_r_params(background_fade="  "), self.vertices))

    def test_a_step_that_declares_no_fade_params_gets_none(self):
        """`render_splat`'s resolved params, which carry no `background_fade`
        key at all — the lookup has to be a miss, not a KeyError."""
        self.assertIsNone(build_fade(_rs_params(), self.vertices))

    def test_a_fade_with_nothing_to_fit_it_to_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_fade(_r_params(), None)
        self.assertIn("vertices", str(caught.exception))

    def test_the_margin_inflates_the_fitted_shell(self):
        tight = build_fade(_r_params(background_fade_margin=1.0), self.vertices)
        wide = build_fade(_r_params(), self.vertices)
        np.testing.assert_allclose(
            wide.ellipsoid.axes, 2.0 * tight.ellipsoid.axes, rtol=1e-6)
        # The tight one is the hull, so the mesh is inside it and the default
        # clears well beyond where the drawing's outline can reach.
        self.assertTrue(tight.ellipsoid.contains(self.vertices).all())
        np.testing.assert_allclose(tight.ellipsoid.axes, 0.2, rtol=0.05)

    def test_a_bad_profile_is_refused_by_body2colmap(self):
        with self.assertRaises(ValueError):
            build_fade(_r_params(background_fade="smoothsteps"), self.vertices)

    # -- through the backdrop -----------------------------------------------

    def test_no_backdrop_is_no_fade_and_no_complaint(self):
        """`background_fade` defaults ON, and the face-cap render turns the
        ROOM off. That has to stay a one-line change, not a refusal telling
        the workflow to turn off a second knob it never turned on."""
        self.assertIsNone(build_background(
            _r_params(background=""), self.cameras, self.vertices))

    def test_the_backdrop_carries_the_fade(self):
        background = build_background(_r_params(), self.cameras, self.vertices)
        self.assertIsNotNone(background.fade)
        background = build_background(
            _r_params(background_fade=""), self.cameras, self.vertices)
        self.assertIsNone(background.fade)

    def test_it_clears_the_middle_of_the_frame_and_leaves_the_corners(self):
        """The whole point, in pixels: the ruling next to the subject goes
        and the far field — which is what carries the rotation cue — does
        not move at all."""
        camera = self.cameras[0]
        faded = build_background(
            _r_params(), self.cameras, self.vertices).render(camera)
        plain = build_background(
            _r_params(background_fade=""), self.cameras, self.vertices
        ).render(camera)

        difference = np.abs(faded.astype(np.int16) - plain.astype(np.int16))
        middle = difference[28:36, 28:36]
        self.assertTrue(np.any(middle > 0),
                        "the fade changed nothing where the subject is")
        for corner in (difference[:8, :8], difference[:8, -8:],
                       difference[-8:, :8], difference[-8:, -8:]):
            np.testing.assert_array_equal(corner, 0)

    def test_a_loaded_texture_cannot_fade_to_a_plain_one(self):
        """There is no pattern-free variant of a photograph to reveal, and
        body2colmap refuses the pair rather than silently blurring. Worth
        pinning because `background` takes a path and `background_fade`
        defaults on, so the two meet without anybody asking for it."""
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "room.png"
            cv2.imwrite(str(path), np.full((32, 64, 3), 90, dtype=np.uint8))
            with self.assertRaises(ValueError) as caught:
                build_background(
                    _r_params(background=str(path), background_geometry="sphere"),
                    self.cameras, self.vertices)
            self.assertIn("plain", str(caught.exception))


class TestFadeStepWiring(unittest.TestCase):
    """`render` fits the shell to the scene it is about to draw.

    The one thing that can silently go wrong here is the FRAME. The
    auto-orient branch turns the scene in place before any camera is built,
    so the mesh the renderer draws is not the mesh that arrived on the step's
    input. Fitting the shell to the input would put the clear zone off to one
    side of the subject — visible only as a room that fails to pull back, on
    a run that otherwise looks correct.
    """

    def setUp(self):
        import trimesh

        # Elongated along X, so turning it about Y is something a shell can
        # be caught not having followed. An icosphere would fit the same
        # ellipsoid either way and pin nothing.
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        vertices[:, 0] *= 3.0
        rng = np.random.default_rng(3)
        self.mesh_output = {
            "vertices": vertices,
            "faces": np.asarray(mesh.faces, dtype=np.int32),
            "cam_t": np.array([0.0, 0.0, 3.0], dtype=np.float32),
            "keypoints_3d": (rng.normal(size=(70, 3)) * 0.3).astype(np.float32),
            "focal_length": 1000.0,
        }
        self.recorder = _RecordingRenderer.install(self)

    def _run(self, **params):
        step = get_step_class("render")()
        return step.run(
            {"mesh_output": self.mesh_output},
            get_step_class("render").resolve_params(
                {"n_frames": 4, "resolution": [8, 8], **params}),
        )

    def test_the_shell_encloses_the_mesh_the_renderer_was_handed(self):
        self._run(render_mode="mesh", override_cam_from_mesh=False,
                  initial_rotation=90.0, background_fade_margin=1.0)
        ellipsoid = self.recorder.background.fade.ellipsoid
        self.assertTrue(ellipsoid.contains(self.recorder.scene.vertices).all())
        # And not the mesh as it arrived: a quarter turn takes the long axis
        # right out of a hull fitted after it.
        self.assertFalse(
            ellipsoid.contains(self.mesh_output["vertices"]).all())

    def test_the_override_path_fits_the_scene_too(self):
        """`override_cam_from_mesh` skips the auto-orient, but the step's
        input is still not the world frame: `Scene.from_sam3d_output` is
        where SAM-3D-Body coordinates become world ones, and it puts the mesh
        `cam_t` away from where it arrived. So there is no path on which the
        raw input is the right array to fit."""
        self._run(render_mode="mesh", background_fade_margin=1.0)
        ellipsoid = self.recorder.background.fade.ellipsoid
        self.assertTrue(ellipsoid.contains(self.recorder.scene.vertices).all())
        self.assertFalse(
            ellipsoid.contains(self.mesh_output["vertices"]).all())

    def test_turning_the_fade_off_leaves_the_room(self):
        self._run(render_mode="mesh", background_fade="")
        self.assertIsNotNone(self.recorder.background)
        self.assertIsNone(self.recorder.background.fade)


if __name__ == "__main__":
    unittest.main()
