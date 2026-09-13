"""The web UI's event wiring — the part no other test touched.

`tests/test_result_bundle.py` covers the param widgets and `tests/test_api.py`
the mount; nothing exercised what a click or a tick actually does. That is
how the page shipped with per-button polling generators that were never
cancelled: two of them painting different runs into the same outputs on
alternating seconds, a button that silently ignored every click while its
generator ran, and a picker that only knew the runs of one process.

These pin the shape that replaced it: one timer, one view function, sends
only what changed, and a picker read off the volume.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import unittest
import unittest.mock
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import gradio as gr

    from pipeline import webui
except ImportError:  # pragma: no cover - depends on the local env
    gr = webui = None

from pipeline.gpu_scheduler import GpuScheduler
from pipeline.run_state import RunJob


def _no_change(value) -> bool:
    """`gr.update()` with nothing in it — "leave the component alone"."""
    return isinstance(value, dict) and value == {"__type__": "update"}


class _StillRunning:
    """A worker process that never ends and never publishes a status."""

    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass


def _touch(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@unittest.skipIf(webui is None, "the web UI's dependencies are not installed here")
class TestWiring(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name)
        patcher = unittest.mock.patch.dict(os.environ, {"B2C_DATA_DIR": str(self.data)})
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.data / "output").mkdir()
        (self.data / "run_jobs").mkdir()
        # The scheduler the app builds gets a spawn that starts nothing, so
        # a submitted job is a running run with no process behind it.
        scheduler = partial(GpuScheduler, spawn=lambda *_: _StillRunning())
        with unittest.mock.patch.object(webui, "GpuScheduler", scheduler):
            self.app = webui.build_app("", gpu_count=1)
        self.scheduler = self.app._gpu_scheduler  # noqa: SLF001
        self.picker = next(
            b for b in self.app.blocks.values() if getattr(b, "label", None) == "Active run"
        )

    # -- fixtures ------------------------------------------------------------

    def _finished_run(self, name: str, *, deliverables: bool = True) -> Path:
        """A run only the volume knows: a status file and an output directory."""
        run = self.data / "output" / name
        _touch(run / "frame_00001_.png", 4096)
        if deliverables:
            for member in ("cameras.txt", "images.txt", "points3D.txt"):
                _touch(run / "colmap" / member)
        log = self.data / "logs" / f"{name}.log"
        _touch(log, 64)
        (self.data / "run_jobs" / f"{name}.status.json").write_text(json.dumps({
            "name": name, "workflow": "helical", "status": "done",
            "started": 90.0, "finished": 100.0, "current": 3, "total": 3,
            "output_dir": str(run), "log_path": str(log),
        }))
        return run

    def _live_run(self, name: str) -> None:
        self.scheduler.submit(RunJob(
            run_name=name, workflow_name="helical", workflow_path="x.yaml",
            output_dir=str(self.data / "output" / name),
        ))

    def _dep(self, event: str, block=None):
        for fn in self.app.fns.values():
            for target_id, target_event in fn.targets:
                if target_event == event and (block is None or target_id == block._id):
                    if block is None and event == "load" and self.picker not in fn.outputs:
                        continue  # gr.render registers loads of its own
                    return fn
        raise AssertionError(f"no {event!r} dependency")

    # -- the shape -------------------------------------------------------------

    def test_nothing_on_the_page_streams_except_the_volume_scan(self):
        generators = [
            fn.fn.__name__ for fn in self.app.fns.values()
            if fn.fn is not None and inspect.isgeneratorfunction(fn.fn)
        ]
        self.assertEqual(generators, ["on_all_results"])
        self.assertEqual(len(self._dep("tick").outputs), 13)
        labels = [b.value for b in self.app.blocks.values() if isinstance(b, gr.Button)]
        for gone in ("Refresh run list", "Attach / refresh", "Load latest results"):
            self.assertNotIn(gone, labels)

    def test_every_gallery_leaves_fullscreen_when_its_preview_closes(self):
        galleries = [b for b in self.app.blocks.values() if isinstance(b, gr.Gallery)]
        self.assertEqual(len(galleries), 3)
        for gallery in galleries:
            dep = self._dep("preview_close", gallery)
            self.assertIsNone(dep.fn, "a round trip for a frontend-only fix")
            self.assertIn("exitFullscreen", dep.js)

    # -- the picker ------------------------------------------------------------

    def test_a_fresh_page_lists_the_volume_and_selects_the_run_in_flight(self):
        self._finished_run("older")
        self._live_run("live")
        picker, fleet, status, *_rest, memo = self._dep("load").fn(None, {}, webui.PREVIEW_ALL)

        self.assertEqual([value for _, value in picker["choices"]], ["live", "older"])
        self.assertEqual(picker["value"], "live")
        self.assertIn("running", status)
        self.assertIn("**1 of 1** GPU busy", fleet)
        self.assertEqual(memo["sig"][0], "live")

    def test_a_run_started_elsewhere_is_picked_up_without_a_button(self):
        self._dep("load").fn(None, {}, webui.PREVIEW_ALL)
        self._finished_run("from-the-cli")
        picker, *_ = self._dep("tick").fn(None, {"choices": []}, webui.PREVIEW_ALL)
        self.assertEqual([value for _, value in picker["choices"]], ["from-the-cli"])

    # -- the view --------------------------------------------------------------

    def test_a_tick_on_a_finished_run_sends_nothing(self):
        self._finished_run("done")
        load = self._dep("load").fn(None, {}, webui.PREVIEW_ALL)
        memo = load[-1]

        tick = self._dep("tick").fn("done", memo, webui.PREVIEW_ALL)
        picker, *painted, memo = tick
        self.assertTrue(_no_change(picker))
        self.assertTrue(all(_no_change(item) for item in painted), painted)

    def test_a_tick_on_a_running_run_repaints_the_live_parts_only(self):
        self._live_run("live")
        memo = self._dep("load").fn(None, {}, webui.PREVIEW_ALL)[-1]
        _picker, fleet, status, progress, steps, log, log_file, info, frames, zip_, *_, memo = (
            self._dep("tick").fn("live", memo, webui.PREVIEW_ALL)
        )
        self.assertTrue(_no_change(fleet))  # nothing started or stopped
        self.assertIn("running", status)  # elapsed moves every tick
        self.assertFalse(_no_change(progress))
        # The results half is only reread when the run's step or status
        # changes — an rglob of a run directory every two seconds is not.
        self.assertTrue(_no_change(info) and _no_change(frames) and _no_change(zip_))
        self.assertTrue(_no_change(log_file))

    def test_switching_runs_repaints_at_once_from_the_new_run(self):
        first = self._finished_run("first")
        second = self._finished_run("second")
        memo = self._dep("load").fn("first", {}, webui.PREVIEW_ALL)[-1]
        self.assertEqual(memo["sig"][0], "first")

        out = self._dep("change", self.picker).fn("second", memo, webui.PREVIEW_ALL)
        _fleet, status, _progress, _steps, _log, log_file, info, *_rest, memo = out
        self.assertIn("`second`", status)
        self.assertIn(str(second), info)
        self.assertNotIn(str(first), info)
        self.assertEqual(log_file["value"], str(self.data / "logs" / "second.log"))
        self.assertEqual(memo["sig"][0], "second")

    def test_looking_at_a_run_never_packages_it(self):
        run = self._finished_run("done")
        out = self._dep("load").fn("done", {}, webui.PREVIEW_ALL)
        info, _frames, archive = out[7], out[8], out[9]
        self.assertIsNone(archive)
        self.assertIn("Package .zip", info)
        self.assertEqual(list((self.data / "output").glob("*.zip")), [])

        package = next(
            fn for fn in self.app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "on_package"
        )
        info, archive = package.fn("done")
        self.assertTrue(Path(archive).is_file())
        self.assertIn("The .zip contains", info)

        # Now the view finds the archive it did not build.
        out = self._dep("change", self.picker).fn("done", {}, webui.PREVIEW_ALL)
        self.assertEqual(out[8], archive)
        # ...until the run directory changes under it.
        time.sleep(0.01)
        _touch(run / "colmap" / "new.txt")
        out = self._dep("change", self.picker).fn("done", {}, webui.PREVIEW_ALL)
        self.assertIsNone(out[8])

    def test_packaging_a_run_this_server_is_still_running_is_refused(self):
        self._live_run("live")
        package = next(
            fn for fn in self.app.fns.values()
            if fn.fn is not None and fn.fn.__name__ == "on_package"
        )
        with self.assertRaises(gr.Error):
            package.fn("live")

    def test_the_log_box_gets_a_tail_not_the_file(self):
        run = self._finished_run("chatty")
        log = self.data / "logs" / "chatty.log"
        log.write_text("".join(f"line {i}\n" for i in range(5000)))
        out = self._dep("load").fn("chatty", {}, webui.PREVIEW_ALL)
        text = out[5]
        self.assertEqual(text.count("\n") + 1, webui._LOG_TAIL_LINES)
        self.assertTrue(text.endswith("line 4999"))
        self.assertTrue(run.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
