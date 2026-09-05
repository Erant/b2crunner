"""What the web UI's single upload box accepts, and how a zip of image/prompt
pairs fans out into one run each.

There is no input picker any more: `runs.resolve_upload` looks at what was
uploaded and decides. These pin the two shapes it understands — a bare
reference-sheet image, and a `.zip` of `image1.jpg` / `image1.txt` pairs —
plus the lenient "images, no `.txt`" case.

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
        self.assertEqual([prompt for _, prompt in plan], ["a woman in red", "a man in blue"])
        self.assertEqual([Path(img).name for img, _ in plan], ["image1.jpg", "image2.png"])

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
            [prompt for _, prompt in plan], ["fallback subject", "fallback subject"]
        )

    def test_an_empty_pair_txt_falls_back_to_the_subject_box(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {"a.jpg": b"x", "a.txt": b"   \n"})
        plan = runs.resolve_upload(str(src), "fallback")
        self.assertEqual(plan[0][1], "fallback")

    def test_a_macosx_sidecar_and_a_nested_folder_are_handled(self):
        src = self.tmp / "folder.zip"
        _zip(src, {
            "batch/image1.jpg": b"x", "batch/image1.txt": b"one",
            "__MACOSX/batch/._image1.jpg": b"junk",
        })
        plan = runs.resolve_upload(str(src), "")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][1], "one")

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
        saved, prompt = plan[0]
        self.assertEqual(prompt, "a subject")
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
            {"export_colmap": True, "export_ply": True,
             "export_colmap_intermediate": False,
             "export_debug": True,
             "export_colmap_preupscale": False},
        )

    def test_the_debug_bundle_is_not_a_deliverable_on_its_own(self):
        """It draws as a checkbox and travels as a switch, but a run that
        exports only `debug/` produces nothing: `_write_run_members`
        refuses to build an archive out of it, the same way it refuses to
        build one out of `log.txt`. Counting it here would let that past
        and hand back nothing after an hour of GPU."""
        with self.assertRaises(SubmitError):
            runs.resolve_outputs(self.spec(
                export_colmap=False, export_ply=False,
                export_colmap_intermediate=False, export_debug=True,
            ))

    def test_switching_the_debug_bundle_off_leaves_a_run_valid(self):
        resolved = runs.resolve_outputs(self.spec(export_debug=False))
        self.assertIs(resolved["export_debug"], False)
        self.assertIs(resolved["export_colmap"], True)

    def test_the_intermediate_colmap_is_independent_of_the_upscale(self):
        """Unlike the pre-upscale export it declares no `requires:`: the
        frames it writes are the ones the first brush training saw, which
        no other export in the run can stand in for."""
        for run_upscale in (True, False):
            with self.subTest(run_upscale=run_upscale):
                got = runs.resolve_outputs(self.spec(
                    run_upscale=run_upscale, export_colmap=False,
                    export_colmap_intermediate=True,
                ))
                self.assertTrue(got["export_colmap_intermediate"])
                self.assertFalse(got["export_colmap"])

    def test_the_intermediate_colmap_alone_is_a_valid_run(self):
        """It is somebody deliberately asking what the first training was
        fed, so it counts as an output rather than tripping the guard."""
        got = runs.resolve_outputs(self.spec(
            export_colmap=False, export_ply=False,
            export_colmap_intermediate=True,
        ))
        self.assertEqual(
            (got["export_colmap"], got["export_ply"],
             got["export_colmap_preupscale"], got["export_colmap_intermediate"]),
            (False, False, False, True),
        )

    def test_pre_upscale_colmap_is_kept_when_upscaling(self):
        got = runs.resolve_outputs(self.spec(
            run_upscale=True, export_colmap=False, export_colmap_preupscale=True,
        ))
        self.assertTrue(got["export_colmap_preupscale"])
        self.assertFalse(got["export_colmap"])

    def test_pre_upscale_colmap_is_forced_off_without_its_requirement(self):
        """`requires: run_upscale`. With the upscale off it would be the
        ordinary colmap/ under a second name, so it is refused rather than
        quietly redirected — which is what this used to do."""
        with self.assertRaises(SubmitError):
            runs.resolve_outputs(self.spec(
                run_upscale=False, export_colmap=False, export_ply=False,
                export_colmap_preupscale=True,
            ))

    def test_a_forced_off_output_does_not_take_the_others_with_it(self):
        got = runs.resolve_outputs(self.spec(
            run_upscale=False, export_colmap=True, export_colmap_preupscale=True,
        ))
        self.assertEqual(
            (got["export_colmap"], got["export_colmap_preupscale"]), (True, False)
        )

    def test_a_string_requirement_is_read_the_way_when_reads_it(self):
        """`--param run_upscale=false` arrives as a string, and
        `bool("false")` is True."""
        got = runs.resolve_outputs(self.spec(
            run_upscale="false", export_colmap=True, export_colmap_preupscale=True,
        ))
        self.assertFalse(got["export_colmap_preupscale"])

    def test_nothing_selected_is_an_error(self):
        with self.assertRaises(SubmitError):
            runs.resolve_outputs(self.spec(
                export_colmap=False, export_ply=False,
                export_colmap_intermediate=False,
                export_colmap_preupscale=False,
            ))


if __name__ == "__main__":
    unittest.main()
