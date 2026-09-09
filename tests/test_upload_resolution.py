"""What the web UI's single upload box accepts, and how a zip of image,
prompt and settings files fans out into one run each.

There is no input picker any more: `runs.resolve_upload` looks at what was
uploaded and decides. These pin the two shapes it understands — a bare
reference-sheet image, and a `.zip` of `image1.jpg` / `image1.txt` pairs —
plus the lenient "images, no `.txt`" case, and the optional `image1.yaml`
sidecar that makes such a zip a sweep: one run per image, each at its own
step settings.

All of this is `pipeline.runs`, which the HTTP API drives too — so a
refusal here is the sentence both a Gradio toast and a 400's `detail`
carry, and none of it needs the UI's dependency installed.
"""

from __future__ import annotations

import unittest
import unittest.mock
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline import runs
from pipeline.runs import SubmitError


def _zip(path: Path, members: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


class UploadResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # resolve_upload extracts into upload_dir() and save_upload copies
        # there; keep both off the real volume for the test.
        (self.tmp / "uploads").mkdir()
        patcher = unittest.mock.patch.object(
            runs, "upload_dir", lambda: self.tmp / "uploads"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    # -- a .zip of image/prompt pairs ------------------------------------

    def test_pairs_become_one_plan_entry_each_with_their_prompt(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {
            "image1.jpg": b"jpgbytes", "image1.txt": b"a woman in red\n",
            "image2.png": b"pngbytes", "image2.txt": b"  a man in blue  ",
        })
        plan = runs.resolve_upload(str(src), "")
        self.assertEqual([run.prompt for run in plan], ["a woman in red", "a man in blue"])
        self.assertEqual(
            [run.reference_image.name for run in plan], ["image1.jpg", "image2.png"]
        )

    def test_an_image_missing_its_txt_is_an_error_naming_it(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {
            "image1.jpg": b"x", "image1.txt": b"ok",
            "image2.png": b"x",  # no image2.txt
        })
        with self.assertRaises(SubmitError) as ctx:
            runs.resolve_upload(str(src), "")
        self.assertIn("image2.png", str(ctx.exception))

    def test_a_zip_of_images_with_no_txt_uses_the_subject_box(self):
        src = self.tmp / "sheets.zip"
        _zip(src, {"a.jpg": b"x", "b.jpg": b"x", "readme.md": b"hi"})
        plan = runs.resolve_upload(str(src), "fallback subject")
        self.assertEqual(
            [run.prompt for run in plan], ["fallback subject", "fallback subject"]
        )

    def test_an_empty_pair_txt_falls_back_to_the_subject_box(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {"a.jpg": b"x", "a.txt": b"   \n"})
        plan = runs.resolve_upload(str(src), "fallback")
        self.assertEqual(plan[0].prompt, "fallback")

    def test_a_macosx_sidecar_and_a_nested_folder_are_handled(self):
        src = self.tmp / "folder.zip"
        _zip(src, {
            "batch/image1.jpg": b"x", "batch/image1.txt": b"one",
            "__MACOSX/batch/._image1.jpg": b"junk",
        })
        plan = runs.resolve_upload(str(src), "")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].prompt, "one")

    def test_a_zip_with_no_images_is_an_error(self):
        src = self.tmp / "empty.zip"
        _zip(src, {"notes.md": b"nothing here"})
        with self.assertRaises(SubmitError):
            runs.resolve_upload(str(src), "")

    def test_zip_slip_is_refused(self):
        src = self.tmp / "evil.zip"
        _zip(src, {"../escape.jpg": b"x", "../escape.txt": b"y"})
        with self.assertRaises(SubmitError):
            runs.resolve_upload(str(src), "")

    # -- a bare image -----------------------------------------------------

    def test_a_bare_image_is_one_reference_run_saved_to_the_volume(self):
        img = self.tmp / "sheet.png"
        img.write_bytes(b"pngbytes")
        plan = runs.resolve_upload(str(img), "a subject")
        self.assertEqual(len(plan), 1)
        saved = plan[0].reference_image
        self.assertEqual(plan[0].prompt, "a subject")
        self.assertIsNone(plan[0].settings_path)
        self.assertTrue(Path(saved).exists())
        self.assertTrue(str(saved).startswith(str(self.tmp / "uploads")))

    def test_an_unknown_file_type_is_a_clear_error(self):
        bad = self.tmp / "notes.rtf"
        bad.write_text("hi")
        with self.assertRaises(SubmitError):
            runs.resolve_upload(str(bad), "")

    def test_a_member_escaping_into_a_sibling_directory_is_refused(self):
        """The containment test is a path check, not a string prefix.

        `/uploads/upload-1-evil/x.png` startswith `/uploads/upload-1`, so a
        prefix comparison lets a member land in a sibling whose name merely
        extends the target's — outside the directory this submission was
        given, next to another submission's data.
        """
        target = self.tmp / "uploads" / "upload-1"
        archive = self.tmp / "escape.zip"
        _zip(archive, {"../upload-1-evil/x.png": b"\x89PNG"})
        with self.assertRaises(SubmitError):
            runs._guarded_extract(str(archive), target)
        self.assertFalse((self.tmp / "uploads" / "upload-1-evil").exists())


class SettingsSidecarTests(unittest.TestCase):
    """`image1.yaml` beside `image1.jpg`: the per-run override set.

    The file's two keys are the API's own field names, so what you drop in
    a zip and what you would have POSTed as `settings`/`step_params` are
    the same document. These pin the reading of it; whether a key names a
    real setting is `submit_runs`' business, and lives in the class below.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.tmp / "uploads").mkdir()
        patcher = unittest.mock.patch.object(
            runs, "upload_dir", lambda: self.tmp / "uploads"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def sidecar(self, text: str, name: str = "image1.yaml") -> Path:
        path = self.tmp / name
        path.write_text(text)
        return path

    def test_both_blocks_are_read(self):
        path = self.sidecar(
            "settings:\n"
            "  seed: 7\n"
            "step_params:\n"
            "  train_final_splat:\n"
            "    total_steps: 12000\n"
        )
        self.assertEqual(
            runs.read_settings_sidecar(path),
            ({"seed": 7}, {"train_final_splat": {"total_steps": 12000}}),
        )

    def test_json_is_read_by_the_same_loader(self):
        path = self.sidecar(
            '{"settings": {"seed": 3}, "step_params": {"upscale": {"seed": 3}}}',
            name="image1.json",
        )
        self.assertEqual(
            runs.read_settings_sidecar(path),
            ({"seed": 3}, {"upscale": {"seed": 3}}),
        )

    def test_an_empty_file_is_no_overrides(self):
        self.assertEqual(runs.read_settings_sidecar(self.sidecar("")), ({}, {}))

    def test_a_key_that_is_neither_block_is_refused_naming_it(self):
        """A bare `seed: 7` at the top level looks right and would do
        nothing at all, which is the failure this whole file guards
        against — a run that completes, hours later, at settings nobody
        chose."""
        with self.assertRaises(SubmitError) as ctx:
            runs.read_settings_sidecar(self.sidecar("seed: 7\n"))
        self.assertIn("seed", str(ctx.exception))
        self.assertIn("image1.yaml", str(ctx.exception))

    def test_a_prompt_key_is_pointed_at_the_txt_file(self):
        with self.assertRaises(SubmitError) as ctx:
            runs.read_settings_sidecar(self.sidecar("prompt: a woman in red\n"))
        self.assertIn(".txt", str(ctx.exception))

    def test_a_step_block_that_is_not_a_mapping_is_refused(self):
        with self.assertRaises(SubmitError) as ctx:
            runs.read_settings_sidecar(
                self.sidecar("step_params:\n  train_final_splat: 12000\n")
            )
        self.assertIn("train_final_splat", str(ctx.exception))

    def test_broken_yaml_names_the_file(self):
        with self.assertRaises(SubmitError) as ctx:
            runs.read_settings_sidecar(self.sidecar("settings: [unclosed\n"))
        self.assertIn("image1.yaml", str(ctx.exception))

    # -- and how one reaches a plan --------------------------------------

    def test_each_image_carries_its_own_sidecar(self):
        src = self.tmp / "sweep.zip"
        _zip(src, {
            "a.jpg": b"x", "a.yaml": b"step_params:\n  train_final_splat:\n    align_iters: 8\n",
            "b.jpg": b"x", "b.yml": b"settings:\n  seed: 5\n",
            "c.jpg": b"x",
        })
        plan = runs.resolve_upload(str(src), "subject")
        self.assertEqual(
            [run.step_overrides for run in plan],
            [{"train_final_splat": {"align_iters": 8}}, {}, {}],
        )
        self.assertEqual([run.global_overrides for run in plan], [{}, {"seed": 5}, {}])
        self.assertEqual(
            [run.settings_path.name if run.settings_path else None for run in plan],
            ["a.yaml", "b.yml", None],
        )

    def test_a_sidecar_matching_no_image_is_refused_naming_it(self):
        src = self.tmp / "typo.zip"
        _zip(src, {"image1.jpg": b"x", "image7.yaml": b"settings:\n  seed: 1\n"})
        with self.assertRaises(SubmitError) as ctx:
            runs.resolve_upload(str(src), "")
        self.assertIn("image7.yaml", str(ctx.exception))

    def test_two_sidecars_for_one_image_are_refused(self):
        src = self.tmp / "both.zip"
        _zip(src, {
            "a.jpg": b"x",
            "a.yaml": b"settings:\n  seed: 1\n",
            "a.json": b'{"settings": {"seed": 2}}',
        })
        with self.assertRaises(SubmitError) as ctx:
            runs.resolve_upload(str(src), "")
        self.assertIn("a.yaml", str(ctx.exception))
        self.assertIn("a.json", str(ctx.exception))


class _CollectingScheduler:
    """`submit_runs` only ever calls `submit`; nothing here needs a GPU."""

    def __init__(self):
        self.jobs = []

    def submit(self, job):
        self.jobs.append(job)


class SidecarSubmissionTests(unittest.TestCase):
    """What a sidecar means once it reaches `submit_runs`.

    Asked of the shipped workflow rather than a stub, for the same reason
    `OutputSwitchTests` is: the merge is only worth anything if the keys
    are the real ones.
    """

    def setUp(self):
        self.scheduler = _CollectingScheduler()

    def submit(self, plan, **kwargs):
        names = runs.submit_runs(self.scheduler, plan, **kwargs)
        self.assertEqual(len(names), len(plan))
        return self.scheduler.jobs

    def planned(self, name="image1.jpg", **kwargs):
        return runs.PlannedRun(reference_image=Path(name), **kwargs)

    def test_a_sidecar_beats_the_submission_and_leaves_the_rest_alone(self):
        jobs = self.submit(
            [
                self.planned(
                    "a.jpg",
                    global_overrides={"seed": 9},
                    step_overrides={"train_final_splat": {"align_iters": 8}},
                    settings_path=Path("a.yaml"),
                ),
                self.planned("b.jpg"),
            ],
            global_overrides={"seed": 1, "framing": "bust"},
            step_overrides={"train_final_splat": {"total_steps": 100}},
        )
        self.assertEqual(jobs[0].global_overrides["seed"], 9)
        self.assertEqual(jobs[1].global_overrides["seed"], 1)
        # Untouched by the sidecar, so both runs still carry the panel's.
        self.assertEqual(
            [job.global_overrides["framing"] for job in jobs], ["bust", "bust"]
        )
        # A step's params merge key by key rather than replacing the block.
        self.assertEqual(
            jobs[0].step_overrides["train_final_splat"],
            {"total_steps": 100, "align_iters": 8},
        )
        self.assertEqual(
            jobs[1].step_overrides["train_final_splat"], {"total_steps": 100}
        )

    def test_the_submission_wide_overrides_are_not_mutated(self):
        """One run's sidecar reaching the next run — or the UI's own
        `param_state` — is the bug this batch shape invites."""
        submitted = {"seed": 1}
        self.submit(
            [self.planned("a.jpg", global_overrides={"seed": 9},
                          settings_path=Path("a.yaml")),
             self.planned("b.jpg")],
            global_overrides=submitted,
        )
        self.assertEqual(submitted, {"seed": 1})

    def test_a_sidecar_naming_nothing_is_refused_even_unstrict(self):
        """A browser's stale param panel is forgiven; a file somebody wrote
        by hand for this run is not. The refusal names the file and the
        image, because there is a zip of them."""
        with self.assertRaises(SubmitError) as ctx:
            self.submit([self.planned(
                "a.jpg", global_overrides={"no_such_setting": 1},
                settings_path=Path("a.yaml"),
            )])
        self.assertIn("a.yaml", str(ctx.exception))
        self.assertIn("no_such_setting", str(ctx.exception))

    def test_a_sidecar_switching_every_output_off_is_refused(self):
        with self.assertRaises(SubmitError) as ctx:
            self.submit([self.planned(
                "a.jpg",
                global_overrides={
                    "export_colmap": False, "export_ply": False,
                    "export_debug": True,
                },
                settings_path=Path("a.yaml"),
            )])
        self.assertIn("a.yaml", str(ctx.exception))

    def test_a_refusal_late_in_the_batch_queues_nothing(self):
        """Half a batch submitted and then refused is the shape that
        wastes GPU: the caller fixes the file and submits again, and the
        runs that got through the first time run twice."""
        with self.assertRaises(SubmitError):
            self.submit([
                self.planned("a.jpg"),
                self.planned("b.jpg", global_overrides={"nope": 1},
                             settings_path=Path("b.yaml")),
            ])
        self.assertEqual(self.scheduler.jobs, [])

    def test_the_output_switches_are_resolved_per_run(self):
        """Every run carries its own resolved switches, so one sidecar
        cannot decide what the run beside it exports. `resolve_outputs` is
        also where an output's `requires:` is applied, and it is applied
        against that run's own settings."""
        jobs = self.submit([
            self.planned("a.jpg",
                         global_overrides={"export_ply": False},
                         settings_path=Path("a.yaml")),
            self.planned("b.jpg"),
        ], global_overrides={"export_debug": False})
        self.assertFalse(jobs[0].global_overrides["export_ply"])
        self.assertTrue(jobs[1].global_overrides["export_ply"])
        self.assertFalse(jobs[1].global_overrides["export_debug"])


class OutputSwitchTests(unittest.TestCase):
    """`resolve_outputs` reads the workflow's own `outputs:` block.

    Nothing in runs.py names these switches any more, so the cases below
    are asked of the shipped workflow rather than of a table beside them.
    """

    def spec(self, **globals_):
        from pipeline.cli import resolve_workflow
        from pipeline.workflow import WorkflowSpec

        spec = WorkflowSpec.from_yaml(resolve_workflow("fast_helical_native"))
        spec.globals.update(globals_)
        return spec

    def test_the_declared_defaults_are_both_deliverables(self):
        self.assertEqual(
            runs.resolve_outputs(self.spec()),
            {"export_colmap": True, "export_ply": True, "export_debug": True},
        )

    def test_the_debug_bundle_is_not_a_deliverable_on_its_own(self):
        """It draws as a checkbox and travels as a switch, but a run that
        exports only `debug/` produces nothing: `_write_run_members`
        refuses to build an archive out of it, the same way it refuses to
        build one out of `log.txt`. Counting it here would let that past
        and hand back nothing after an hour of GPU.

        It is not empty, either — since 2026-09-08 it carries the two debug
        COLMAP datasets that used to be outputs of their own. A run for
        those alone is still a run with nothing to deliver.
        """
        with self.assertRaises(SubmitError):
            runs.resolve_outputs(self.spec(
                export_colmap=False, export_ply=False, export_debug=True,
            ))

    def test_switching_the_debug_bundle_off_leaves_a_run_valid(self):
        resolved = runs.resolve_outputs(self.spec(export_debug=False))
        self.assertIs(resolved["export_debug"], False)
        self.assertIs(resolved["export_colmap"], True)

    def requiring_spec(self, **globals_):
        """The shipped workflow with a `requires:` declared on one output.

        No shipped output declares one: the pre-upscale COLMAP export did
        until 2026-09-08, when it became a member of the debug bundle and
        its `when:` grew the `run_upscale` half instead. The rule below is
        the outputs schema's rather than that export's, and the web UI
        greys a checkbox out on it, so it is tested here against a spec
        that declares one rather than deleted with its last user.
        """
        spec = self.spec(**globals_)
        ply = next(o for o in spec.outputs if o.name == "export_ply")
        ply.requires = "run_upscale"
        return spec

    def test_an_output_is_kept_when_its_requirement_is_on(self):
        got = runs.resolve_outputs(self.requiring_spec(
            run_upscale=True, export_colmap=False, export_ply=True,
        ))
        self.assertTrue(got["export_ply"])
        self.assertFalse(got["export_colmap"])

    def test_an_output_is_forced_off_without_its_requirement(self):
        """With the requirement off the export is refused rather than
        quietly redirected — which is what this used to do."""
        with self.assertRaises(SubmitError):
            runs.resolve_outputs(self.requiring_spec(
                run_upscale=False, export_colmap=False, export_ply=True,
            ))

    def test_a_forced_off_output_does_not_take_the_others_with_it(self):
        got = runs.resolve_outputs(self.requiring_spec(
            run_upscale=False, export_colmap=True, export_ply=True,
        ))
        self.assertEqual((got["export_colmap"], got["export_ply"]), (True, False))

    def test_a_string_requirement_is_read_the_way_when_reads_it(self):
        """`--param run_upscale=false` arrives as a string, and
        `bool("false")` is True."""
        got = runs.resolve_outputs(self.requiring_spec(
            run_upscale="false", export_colmap=True, export_ply=True,
        ))
        self.assertFalse(got["export_ply"])

    def test_nothing_selected_is_an_error(self):
        with self.assertRaises(SubmitError):
            runs.resolve_outputs(self.spec(
                export_colmap=False, export_ply=False, export_debug=False,
            ))


if __name__ == "__main__":
    unittest.main()
