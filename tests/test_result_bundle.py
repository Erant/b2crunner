"""What the web UI's Results tab packages, and what it deliberately omits.

The download used to be `shutil.make_archive` over the whole run
directory: several gigabytes whose top level was a b2c dataset —
metadata.json, 81 full-resolution frames, pointcloud.npz — with the COLMAP
export tucked in a subdirectory and an intermediate `brush/training_<ms>/`
alongside it. These pin the replacement to the two things a run is for.

`pipeline.webui` imports gradio, so this skips rather than fails where the
UI's own dependency isn't installed — same rule the golden-data tests use
for cyber_6f.
"""

from __future__ import annotations

import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

try:
    from pipeline import webui
except ImportError as exc:  # pragma: no cover - depends on the local env
    raise unittest.SkipTest(f"the web UI's dependencies are not installed here: {exc}")


def _touch(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _run_dir(root: Path, *, colmap: bool, ply: bool, name: str = "run-20260824-101500") -> Path:
    """A run directory shaped like a real one: deliverables plus noise."""
    run = root / name
    # The final Dataset.to_disk output and an intermediate training — both
    # real, both things the archive must not carry.
    _touch(run / "metadata.json")
    _touch(run / "frame_00001_.png", 4096)
    _touch(run / "pointcloud.npz", 4096)
    _touch(run / "brush" / "training_1756042129481" / "export.ply", 8192)
    if colmap:
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            _touch(run / "colmap" / name)
        _touch(run / "colmap" / "images" / "frame_00001_.png")
        _touch(run / "colmap" / "normals" / "frame_00001_.png")
    if ply:
        _touch(run / "ply" / "scene.ply", 2048)
    return run


class _BundleCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # build_result_zip writes the archive to output_dir(); keep that off
        # the real volume (or the repo) for the duration of the test.
        self._patched = unittest.mock.patch.object(
            webui, "output_dir", lambda: self.root / "archives"
        )
        (self.root / "archives").mkdir()
        self._patched.start()
        # Same for the log directory the archive picks `log.txt` out of:
        # without this the fallback would go looking on the real volume.
        self._patched_logs = unittest.mock.patch.object(
            webui, "log_dir", lambda: self.root / "logs"
        )
        (self.root / "logs").mkdir()
        self._patched_logs.start()

    def tearDown(self):
        self._patched_logs.stop()
        self._patched.stop()
        self.tmp.cleanup()

    def _names(self, archive: str):
        with zipfile.ZipFile(archive) as bundle:
            return sorted(bundle.namelist())


class TestResultBundle(_BundleCase):
    def test_it_holds_the_deliverables_and_only_those(self):
        run = _run_dir(self.root, colmap=True, ply=True)
        names = self._names(webui.build_result_zip(run))

        self.assertEqual(names, [
            "colmap/cameras.txt",
            "colmap/images.txt",
            "colmap/images/frame_00001_.png",
            "colmap/normals/frame_00001_.png",
            "colmap/points3D.txt",
            "ply/scene.ply",
        ])

    def test_the_run_directorys_own_frames_stay_out_of_it(self):
        """The dataset frames and the intermediate splat are the bulk of a
        run directory and none of what the archive is for."""
        run = _run_dir(self.root, colmap=True, ply=True)
        names = self._names(webui.build_result_zip(run))

        self.assertNotIn("frame_00001_.png", names)
        self.assertNotIn("metadata.json", names)
        self.assertFalse([n for n in names if n.startswith("brush/")])

    def test_one_selected_output_gives_a_one_directory_archive(self):
        run = _run_dir(self.root, colmap=False, ply=True, name="ply-only")
        self.assertEqual(self._names(webui.build_result_zip(run)), ["ply/scene.ply"])

        run = _run_dir(self.root, colmap=True, ply=False, name="colmap-only")
        self.assertFalse([n for n in self._names(webui.build_result_zip(run))
                          if n.startswith("ply/")])

    def test_no_deliverables_means_no_archive(self):
        run = _run_dir(self.root, colmap=False, ply=False)
        self.assertIsNone(webui.build_result_zip(run))
        self.assertEqual(webui.result_dirs(run), {})

    def test_an_empty_output_directory_does_not_count(self):
        """A step that created its output directory and then failed leaves
        one behind; offering an archive of it is a lie."""
        run = _run_dir(self.root, colmap=False, ply=False)
        (run / "colmap").mkdir()
        self.assertIsNone(webui.build_result_zip(run))

    def test_no_run_at_all(self):
        self.assertIsNone(webui.build_result_zip(None))
        self.assertEqual(webui.result_dirs(None), {})


class TestTheLogRidesAlong(_BundleCase):
    """A download that comes back wrong is debugged from the log that
    produced it, and that log lives under B2C_LOG_DIR — a directory the
    person holding the .zip does not have, and one that dies with the pod.
    """

    def _log(self, run: Path, text: str = "17:01:02 +0:00:01 INFO run: hello\n") -> Path:
        path = self.root / "logs" / f"{run.name}.log"
        path.write_text(text, encoding="utf-8")
        return path

    def test_the_recorded_log_is_packaged_as_log_txt(self):
        run = _run_dir(self.root, colmap=True, ply=True)
        log = self._log(run)

        archive = webui.build_result_zip(run, log_path=log)

        self.assertIn("log.txt", self._names(archive))
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(bundle.read("log.txt").decode(), log.read_text())

    def test_it_is_found_by_name_when_the_run_recorded_none(self):
        """A status.json written before `log_path` existed, or a CLI run
        the scheduler never saw: the name `setup_logging` would have
        chosen is `<run name>.log`, so look there."""
        run = _run_dir(self.root, colmap=True, ply=False)
        self._log(run)

        self.assertIn("log.txt", self._names(webui.build_result_zip(run)))

    def test_a_recorded_path_that_no_longer_exists_falls_back(self):
        run = _run_dir(self.root, colmap=True, ply=False)
        self._log(run)

        archive = webui.build_result_zip(run, log_path=self.root / "gone.log")

        self.assertIn("log.txt", self._names(archive))

    def test_no_log_is_not_an_error(self):
        """The deliverables are still the deliverables — a run whose log was
        pruned must still be downloadable."""
        run = _run_dir(self.root, colmap=True, ply=True)

        names = self._names(webui.build_result_zip(run))

        self.assertNotIn("log.txt", names)
        self.assertIn("ply/scene.ply", names)

    def test_the_log_alone_is_not_a_deliverable(self):
        """`log.txt` rides along with an archive; it does not conjure one
        for a run that produced nothing."""
        run = _run_dir(self.root, colmap=False, ply=False)
        self._log(run)

        self.assertIsNone(webui.build_result_zip(run))

    def test_run_log_path_without_a_run(self):
        self.assertIsNone(webui.run_log_path(None))


class TestOutputSelection(unittest.TestCase):
    def test_the_shipped_workflow_declares_its_deliverables(self):
        """The Outputs box IS the workflow's `outputs:` block: its labels,
        its order, and the `dir:` each one lands in."""
        outputs = webui.workflow_outputs("fast_helical_native")
        self.assertEqual(
            [(o.name, o.directory) for o in outputs],
            [("export_colmap", "colmap"),
             ("export_ply", "ply"),
             ("export_colmap_intermediate", "colmap_intermediate"),
             ("export_colmap_preupscale", "colmap_preupscale")],
        )
        self.assertTrue(all(o.label and o.help for o in outputs))
        preupscale = outputs[-1]
        self.assertEqual(preupscale.requires, "run_upscale")

    def test_a_workflow_without_an_outputs_block_offers_nothing(self):
        """The box hides itself rather than pretending to switch something
        the workflow has never heard of."""
        import os

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False
        ) as handle:
            handle.write(
                "name: bare\nglobals: {}\nsteps:\n"
                "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
            )
            bare = handle.name
        self.addCleanup(os.unlink, bare)
        self.assertEqual(webui.workflow_outputs(bare), [])
        self.assertEqual(webui.result_subdirs(bare), [])

    def test_which_workflows_can_start_from_a_photo(self):
        """The gate on the UI's photo input. It got this wrong once by
        treating `save_dataset`'s bare `dataset` input as an unmet read of
        the frames — a checkpoint at the end of a from-a-photo workflow is
        handed a dataset the earlier steps built, not one the caller had to
        supply.

        The real shipped file never needs one; the contrasting case is
        synthetic, since no shipped workflow reads a dataset field cold any
        more."""
        import os
        import tempfile

        self.assertFalse(webui.workflow_needs_a_dataset("fast_helical_native"))

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(
                "name: needs-dataset\nglobals: {}\nsteps:\n"
                "  - id: a\n    step: rmbg\n    dispatch: in_process\n"
                "    inputs:\n      image: dataset.images\n"
            )
            needs_dataset = handle.name
        self.addCleanup(os.unlink, needs_dataset)
        self.assertTrue(webui.workflow_needs_a_dataset(needs_dataset))

    def test_the_photo_gate_keys_off_the_fields_a_photo_cannot_fill(self):
        """`Dataset.from_reference_image` leaves exactly these empty, so
        these are the reads that mean 'a real dataset was required'. If that
        constructor ever fills one in, this set is what has to change."""
        from pipeline.dataset import Dataset

        import numpy as np

        seeded = Dataset.from_reference_image(np.zeros((8, 8, 3), np.uint8))
        for path in webui._NEEDS_A_REAL_DATASET:
            field = path.split(".", 1)[1]
            with self.subTest(field=field):
                value = getattr(seeded, field)
                empty = len(value) == 0 if field != "points_3d" else len(value[0]) == 0
                self.assertTrue(empty, f"{field} is no longer empty on a photo-seeded Dataset")

    def test_the_switches_are_not_also_settings(self):
        """Two editable homes for one setting disagree the moment someone
        touches either. A deliverable's switch belongs to the Outputs box;
        `WorkflowSpec.from_yaml` refuses it being declared twice, and the
        Settings box draws `settings:` only."""
        _spec, settings, outputs, _steps = webui.workflow_param_panel(
            "fast_helical_native")
        names = {s.name for s in settings}
        self.assertFalse(names & {o.name for o in outputs})
        self.assertIn("resolution", names)
        self.assertIn("framing", names)

    def test_a_step_param_wired_to_a_global_is_marked_read_only(self):
        """The render step's `resolution` is `${globals.resolution}` in
        fast_helical_native. It has to stay a declared param (that is how the
        value crosses the dispatcher), so the panel is what keeps it from
        being a second editable home for the frame size."""
        *_, steps = webui.workflow_param_panel("fast_helical_native")
        render = next(s for s in steps if s["step"] == "render")
        self.assertEqual(render["global_refs"].get("resolution"), "resolution")

        # Every param whose workflow value is a whole `${globals.x}` is
        # captured, across every step — not just render's resolution.
        for step in steps:
            for pname, ref in step["global_refs"].items():
                self.assertTrue(ref and not ref.startswith("$"))

    def test_output_root_is_not_drawn_at_all(self):
        """Sharper than the switches: `pipeline.run_worker` only repoints
        output_root at the run directory when the submitted overrides do not
        already carry it. A control for it would let a run write under the
        process's cwd instead, which is how the Results tab once reported
        "the run produced neither" for a run that had completed fine.

        It is a bare `globals:` key, not a declared setting, and the panel
        draws declarations only — so this is now structural rather than a
        denylist that could be forgotten."""
        spec, settings, outputs, _steps = webui.workflow_param_panel(
            "fast_helical_native")
        self.assertIn("output_root", spec.globals)
        drawn = {s.name for s in settings} | {o.name for o in outputs}
        self.assertNotIn("output_root", drawn)


class TestSettingWidgets(unittest.TestCase):
    """A declared setting draws and reads back as the value it declared.

    The resolution control used to be a hand-written special case — a
    RESOLUTION_CHOICES list, an f-string label format and a parser for it.
    It is now the generic choices path, so this pins the round trip rather
    than the format.
    """

    def _setting(self, name):
        _spec, settings, _outputs, _steps = webui.workflow_param_panel(
            "fast_helical_native")
        return next(s for s in settings if s.name == name)

    def test_a_list_choice_round_trips_through_its_label(self):
        resolution = self._setting("resolution")
        self.assertEqual(webui._choice_label([720, 1280]), "720 x 1280")
        self.assertEqual(
            webui._widget_value(resolution, "600 x 1040"), [600, 1040]
        )

    def test_a_scalar_choice_is_its_own_label(self):
        framing = self._setting("framing")
        self.assertEqual(webui._widget_value(framing, "bust"), "bust")

    def test_a_choices_setting_draws_a_dropdown_not_a_yaml_box(self):
        """`resolution` is `type: list`, and the list branch used to win —
        a free-form YAML box a shape no step supports can be typed into."""
        import gradio as gr

        resolution = self._setting("resolution")
        with gr.Blocks():
            widget = webui._widget_for(resolution, [720, 1280], "Resolution", "k")
        self.assertIsInstance(widget, gr.Dropdown)
        self.assertEqual([c[0] for c in widget.choices][0], "720 x 1280")
        self.assertFalse(widget.allow_custom_value)

    def test_a_mapping_param_draws_a_yaml_box_and_reads_back_a_dict(self):
        """`background_params` — the generator's own colours and shape,
        handed to it whole. It shares the list branch's YAML box, so what is
        pinned is that it comes back a mapping rather than a list, and that
        an empty box is an empty mapping rather than an error."""
        import gradio as gr

        from pipeline.registry import get_step_class

        param = get_step_class("render").declared_params()["background_params"]
        with gr.Blocks():
            widget = webui._widget_for(param, {}, "Backdrop params", "k")
        self.assertIsInstance(widget, gr.Textbox)
        self.assertIn("YAML mapping", widget.info)
        self.assertEqual(
            webui._widget_value(param, "{base_color: [0.2, 0.21, 0.24]}"),
            {"base_color": [0.2, 0.21, 0.24]},
        )
        self.assertEqual(webui._widget_value(param, ""), {})

    def test_a_list_param_that_may_be_unset_draws_and_reads_back_empty(self):
        """`background_base_color` is a colour with an OFF position — unset
        means "leave the texture generator its own wall". Two halves: it
        must not draw as the word `null`, and clearing the box must come
        back as None rather than as a malformed-list error.

        A list param with a real default keeps the error, which is the point
        of keying this on the declared default: `bg_color` cleared to None
        would hand the renderer a None where it unpacks three floats.
        """
        import gradio as gr

        from pipeline.registry import get_step_class

        declared = get_step_class("render").declared_params()
        colour = declared["background_base_color"]
        with gr.Blocks():
            empty = webui._widget_for(colour, None, "background_base_color", "k")
            filled = webui._widget_for(colour, [0.5, 0.5, 0.5], "bbc", "k2")
        self.assertEqual(empty.value, "")
        self.assertEqual(filled.value, "[0.5, 0.5, 0.5]")
        self.assertIsNone(webui._widget_value(colour, ""))
        self.assertEqual(webui._widget_value(colour, "[0.5, 0.5, 0.5]"),
                         [0.5, 0.5, 0.5])
        with self.assertRaises(gr.Error):
            webui._widget_value(declared["bg_color"], "")

    def test_an_empty_choice_reads_as_none_and_maps_back_to_the_empty_string(self):
        """`background` and `pattern` both offer "" as a real choice — no
        backdrop, no camera path. Gradio draws the raw value, so an empty
        choice is a blank row that reads as a rendering fault; it gets a
        label, and `_widget_value` has to bring that label back to the ""
        the step actually takes.

        The dropdown still allows a custom value, which is what keeps the
        choice list a suggestion: `background` also accepts a path to an
        equirectangular image or a cubemap, and nobody can enumerate those.
        """
        import gradio as gr

        from pipeline.registry import get_step_class

        param = get_step_class("render").declared_params()["background"]
        with gr.Blocks():
            widget = webui._widget_for(param, "", "background", "k")
        self.assertIsInstance(widget, gr.Dropdown)
        self.assertIn("(none)", [c[0] for c in widget.choices])
        self.assertNotIn("", [c[0] for c in widget.choices])
        self.assertEqual(widget.value, "(none)")
        self.assertTrue(widget.allow_custom_value)
        self.assertEqual(webui._widget_value(param, "(none)"), "")
        self.assertEqual(webui._widget_value(param, "grid"), "grid")
        self.assertEqual(
            webui._widget_value(param, "/data/room.exr"), "/data/room.exr"
        )

    def test_a_setting_is_labelled_by_its_label_not_its_name(self):
        self.assertEqual(self._setting("run_upscale").title, "Upscale dataset")
        # A step param has no label and falls back to the name you would
        # type after --param.
        *_, steps = webui.workflow_param_panel("fast_helical_native")
        param = steps[0]["params"][0]
        self.assertEqual(param.title, param.name)


if __name__ == "__main__":
    unittest.main()
