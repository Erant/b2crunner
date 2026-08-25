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


class TestResultBundle(unittest.TestCase):
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

    def tearDown(self):
        self._patched.stop()
        self.tmp.cleanup()

    def _names(self, archive: str):
        with zipfile.ZipFile(archive) as bundle:
            return sorted(bundle.namelist())

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


class TestOutputSelection(unittest.TestCase):
    def test_the_fast_helical_workflows_offer_both(self):
        for name in ("fast_helical", "fast_helical_full"):
            with self.subTest(workflow=name):
                self.assertEqual(
                    sorted(webui.workflow_outputs(name)),
                    sorted([webui.OUTPUT_COLMAP, webui.OUTPUT_PLY]),
                )

    def test_a_workflow_without_the_params_offers_nothing(self):
        """The control hides itself rather than pretending to switch
        something the workflow has never heard of. fast_helical_native ends
        at a checkpoint, with no COLMAP export and no ply/."""
        self.assertEqual(webui.workflow_outputs("fast_helical_native"), [])

    def test_which_workflows_can_start_from_a_photo(self):
        """The gate on the UI's photo input, pinned to the real shipped
        files. It got this wrong once by treating `save_dataset`'s bare
        `dataset` input as an unmet read of the frames — a checkpoint at
        the end of a from-a-photo workflow is handed a dataset the earlier
        steps built, not one the caller had to supply."""
        self.assertFalse(webui.workflow_needs_a_dataset("fast_helical_native"))
        self.assertTrue(webui.workflow_needs_a_dataset("fast_helical"))
        self.assertTrue(webui.workflow_needs_a_dataset("fast_helical_full"))

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

    def test_the_switches_are_not_also_editable_in_the_params_box(self):
        """Two editable homes for one setting disagree the moment someone
        touches either."""
        import yaml

        params = yaml.safe_load(webui.workflow_params_yaml("fast_helical_full"))
        self.assertNotIn("export_colmap", params)
        self.assertNotIn("export_ply", params)
        self.assertIn("seed", params)


if __name__ == "__main__":
    unittest.main()
