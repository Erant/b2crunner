"""What the web UI's single upload box accepts, and how a zip of image/prompt
pairs fans out into one run each.

There is no input picker any more: `webui.resolve_upload` looks at what was
uploaded and decides. These pin the three shapes it understands — a dataset
`.zip`, a bare reference-sheet image, and a `.zip` of `image1.jpg` /
`image1.txt` pairs — plus the lenient "images, no `.txt`" case.

`pipeline.webui` imports gradio, so this skips rather than fails where the
UI's own dependency isn't installed — same rule the other UI tests use.
"""

from __future__ import annotations

import unittest
import unittest.mock
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from pipeline import webui
except ImportError as exc:  # pragma: no cover - depends on the local env
    raise unittest.SkipTest(f"the web UI's dependencies are not installed here: {exc}")

import gradio as gr


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
            webui, "upload_dir", lambda: self.tmp / "uploads"
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
        dataset_dir, plan = webui.resolve_upload(str(src), "")
        self.assertIsNone(dataset_dir)
        self.assertEqual([prompt for _, prompt in plan], ["a woman in red", "a man in blue"])
        self.assertEqual([Path(img).name for img, _ in plan], ["image1.jpg", "image2.png"])

    def test_an_image_missing_its_txt_is_an_error_naming_it(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {
            "image1.jpg": b"x", "image1.txt": b"ok",
            "image2.png": b"x",  # no image2.txt
        })
        with self.assertRaises(gr.Error) as ctx:
            webui.resolve_upload(str(src), "")
        self.assertIn("image2.png", str(ctx.exception))

    def test_a_zip_of_images_with_no_txt_uses_the_subject_box(self):
        src = self.tmp / "sheets.zip"
        _zip(src, {"a.jpg": b"x", "b.jpg": b"x", "readme.md": b"hi"})
        _dir, plan = webui.resolve_upload(str(src), "fallback subject")
        self.assertEqual(
            [prompt for _, prompt in plan], ["fallback subject", "fallback subject"]
        )

    def test_an_empty_pair_txt_falls_back_to_the_subject_box(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {"a.jpg": b"x", "a.txt": b"   \n"})
        _dir, plan = webui.resolve_upload(str(src), "fallback")
        self.assertEqual(plan[0][1], "fallback")

    def test_a_macosx_sidecar_and_a_nested_folder_are_handled(self):
        src = self.tmp / "folder.zip"
        _zip(src, {
            "batch/image1.jpg": b"x", "batch/image1.txt": b"one",
            "__MACOSX/batch/._image1.jpg": b"junk",
        })
        _dir, plan = webui.resolve_upload(str(src), "")
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][1], "one")

    def test_a_zip_with_neither_images_nor_metadata_is_an_error(self):
        src = self.tmp / "empty.zip"
        _zip(src, {"notes.md": b"nothing here"})
        with self.assertRaises(gr.Error):
            webui.resolve_upload(str(src), "")

    def test_zip_slip_is_refused(self):
        src = self.tmp / "evil.zip"
        _zip(src, {"../escape.jpg": b"x", "../escape.txt": b"y"})
        with self.assertRaises(gr.Error):
            webui.resolve_upload(str(src), "")

    # -- a dataset .zip ------------------------------------------------------

    def test_a_zip_with_metadata_json_resolves_to_a_dataset_dir(self):
        src = self.tmp / "ds.zip"
        _zip(src, {
            "initial/metadata.json": b'{"resolution": [4, 4]}',
            "initial/frame_00001_.png": b"x",
        })
        dataset_dir, plan = webui.resolve_upload(str(src), "hello")
        self.assertIsNotNone(dataset_dir)
        self.assertEqual(Path(dataset_dir).name, "initial")
        self.assertEqual(plan, [(None, "hello")])

    # -- a bare image -----------------------------------------------------

    def test_a_bare_image_is_one_reference_run_saved_to_the_volume(self):
        img = self.tmp / "sheet.png"
        img.write_bytes(b"pngbytes")
        dataset_dir, plan = webui.resolve_upload(str(img), "a subject")
        self.assertIsNone(dataset_dir)
        self.assertEqual(len(plan), 1)
        saved, prompt = plan[0]
        self.assertEqual(prompt, "a subject")
        self.assertTrue(Path(saved).exists())
        self.assertTrue(str(saved).startswith(str(self.tmp / "uploads")))

    def test_an_unknown_file_type_is_a_clear_error(self):
        bad = self.tmp / "notes.rtf"
        bad.write_text("hi")
        with self.assertRaises(gr.Error):
            webui.resolve_upload(str(bad), "")


class WorkflowFromUploadTests(unittest.TestCase):
    """The upload's format picks the workflow — there is no picker."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_workflow_for_a_dataset_dir_vs_an_image_plan(self):
        self.assertEqual(webui.workflow_for(Path("/x/initial")), webui.WORKFLOW_DATASET)
        self.assertEqual(webui.workflow_for(None), webui.WORKFLOW_NATIVE)

    def test_peek_a_dataset_zip_without_extracting_it(self):
        src = self.tmp / "ds.zip"
        _zip(src, {"initial/metadata.json": b"{}", "initial/x.png": b"x"})
        self.assertEqual(webui.peek_upload_workflow(str(src)), webui.WORKFLOW_DATASET)

    def test_peek_a_pairs_zip_is_the_native_path(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {"a.jpg": b"x", "a.txt": b"p"})
        self.assertEqual(webui.peek_upload_workflow(str(src)), webui.WORKFLOW_NATIVE)

    def test_peek_a_bare_image_and_nothing_attached(self):
        img = self.tmp / "sheet.png"
        img.write_bytes(b"x")
        self.assertEqual(webui.peek_upload_workflow(str(img)), webui.WORKFLOW_NATIVE)
        self.assertEqual(webui.peek_upload_workflow(None), webui.WORKFLOW_DATASET)

    def test_peek_a_macosx_metadata_sidecar_does_not_count_as_a_dataset(self):
        src = self.tmp / "pairs.zip"
        _zip(src, {"a.jpg": b"x", "a.txt": b"p", "__MACOSX/metadata.json": b"junk"})
        self.assertEqual(webui.peek_upload_workflow(str(src)), webui.WORKFLOW_NATIVE)


class OutputGlobalsTests(unittest.TestCase):
    """resolve_output_globals folds the pre-upscale-COLMAP rules in."""

    COLMAP = webui.OUTPUT_COLMAP
    PLY = webui.OUTPUT_PLY

    def test_both_boxes_with_the_upscale_on(self):
        self.assertEqual(
            webui.resolve_output_globals([self.COLMAP, self.PLY], True, False),
            {"run_upscale": True, "export_colmap": True, "export_ply": True,
             "export_colmap_preupscale": False,
             "export_colmap_intermediate": False},
        )

    def test_the_intermediate_colmap_is_independent_of_the_upscale(self):
        """Unlike the pre-upscale export, it has no interaction to fold in:
        the frames it writes are the ones the first brush training saw,
        which no other export in the run can stand in for."""
        for run_upscale in (True, False):
            with self.subTest(run_upscale=run_upscale):
                got = webui.resolve_output_globals(
                    [self.PLY], run_upscale, False, True)
                self.assertTrue(got["export_colmap_intermediate"])
                self.assertFalse(got["export_colmap"])

    def test_the_intermediate_colmap_alone_is_a_valid_run(self):
        """It is somebody deliberately asking what the first training was
        fed, so it counts as an output rather than tripping the guard."""
        got = webui.resolve_output_globals([], True, False, True)
        self.assertEqual(
            (got["export_colmap"], got["export_ply"],
             got["export_colmap_preupscale"], got["export_colmap_intermediate"]),
            (False, False, False, True),
        )

    def test_pre_upscale_colmap_only_makes_a_dir_when_upscaling(self):
        got = webui.resolve_output_globals([], True, True)
        self.assertTrue(got["export_colmap_preupscale"])
        self.assertFalse(got["export_colmap"])

    def test_pre_upscale_colmap_collapses_to_plain_colmap_without_upscale(self):
        got = webui.resolve_output_globals([], False, True)
        self.assertFalse(got["export_colmap_preupscale"])
        self.assertTrue(got["export_colmap"])

    def test_pre_and_post_colmap_without_upscale_is_one_colmap(self):
        got = webui.resolve_output_globals([self.COLMAP], False, True)
        self.assertEqual(
            (got["export_colmap"], got["export_colmap_preupscale"]), (True, False)
        )

    def test_nothing_selected_is_an_error(self):
        with self.assertRaises(gr.Error):
            webui.resolve_output_globals([], True, False)


if __name__ == "__main__":
    unittest.main()
