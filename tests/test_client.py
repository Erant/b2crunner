"""`pipeline.client` against a real HTTP server, and its poll loop in isolation.

Split deliberately. `TestAgainstAServer` starts uvicorn on an ephemeral port
and drives the actual routes, because a client tested against a mock only
proves the mock agrees with itself — the things that break here are a
header, a status code, a streamed body, a `Content-Disposition`. It builds
the router alone rather than the whole mounted server, so it needs no
gradio: the UI is `tests/test_api.py`'s business.

`TestFollow` does the opposite, against no server at all. `follow` turns
whole-`RunState` polls into a stream of transitions, and what it has to get
right — a step reported once, in index order, the outcome printed below the
stages that produced it — is about the diffing, not the transport. Driving
that over real HTTP would mean racing a worker to write status files.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import unittest
import unittest.mock
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import uvicorn
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - depends on the local env
    raise unittest.SkipTest(f"the server's dependencies are not installed here: {exc}")

from pipeline import runs
from pipeline.api import (
    API_PREFIX, TOKEN_ENV, ShutdownController, add_schema_route, build_router,
)
from pipeline.client import ApiError, B2CClient
from pipeline.gpu_scheduler import GpuScheduler
from pipeline.run_state import RunState

TOKEN = "client-test-token"


def _png(value: int = 0) -> bytes:
    import cv2

    return cv2.imencode(".png", np.full((64, 128, 3), value, np.uint8))[1].tobytes()


class _StubProcess:
    """A started worker that stops when it is told, like one between steps."""

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


class TestAgainstAServer(unittest.TestCase):
    """A real uvicorn on a real socket, driven by the real client."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name)

        env = unittest.mock.patch.dict(
            os.environ, {"B2C_DATA_DIR": str(self.data), TOKEN_ENV: TOKEN}
        )
        env.start()
        self.addCleanup(env.stop)
        for stale in ("B2C_OUTPUT_DIR", "B2C_UPLOAD_DIR", "B2C_RUN_JOBS_DIR", "B2C_LOG_DIR"):
            os.environ.pop(stale, None)

        self.submitted = []
        self.scheduler = GpuScheduler(
            gpu_count=1, work_dir=self.data / "run_jobs",
            spawn=lambda *_: _StubProcess(),
        )
        queue = self.scheduler.submit
        self.scheduler.submit = lambda job: (self.submitted.append(job), queue(job))[1]
        self.addCleanup(self.scheduler.shutdown)

        # Bound to a recorder rather than to this test's uvicorn: what is
        # being checked here is the round trip, not that the process dies.
        self.stopped = []
        controller = ShutdownController(self.scheduler)
        controller.bind(lambda: self.stopped.append("server"))

        app = FastAPI()
        app.include_router(
            build_router(self.scheduler, "envs.yaml", controller), prefix=API_PREFIX,
        )
        add_schema_route(app, API_PREFIX)
        self.url = self._serve(app)
        self.client = B2CClient(self.url, TOKEN)

    def _serve(self, app) -> str:
        """Start uvicorn on port 0 and return the URL it actually bound."""
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.monotonic() + 20
        while not server.started:
            if time.monotonic() > deadline:  # pragma: no cover
                raise AssertionError("the test server never came up")
            time.sleep(0.02)

        def stop():
            server.should_exit = True
            thread.join(timeout=10)

        self.addCleanup(stop)
        port = server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def _sheet(self, name: str = "sheet.png", value: int = 0) -> Path:
        path = self.data / name
        path.write_bytes(_png(value))
        return path

    def finished_run(self, name="finished-run", deliverables=True) -> Path:
        run_dir = runs.output_dir() / name
        (run_dir / "colmap").mkdir(parents=True, exist_ok=True)
        if deliverables:
            (run_dir / "colmap" / "cameras.txt").write_text("# CAMERAS\n")
        log = runs.log_dir() / f"{name}.log"
        log.write_text("\n".join(f"line {i}" for i in range(200)))
        (runs.run_jobs_dir() / f"{name}.status.json").write_text(json.dumps(RunState(
            name=name, workflow="fast_helical_native", status="done",
            message="complete", output_dir=run_dir, log_path=log,
            started=100.0, finished=700.0,
        ).to_dict()))
        return run_dir

    # -- reading ----------------------------------------------------------

    def test_health_and_workflows(self):
        self.assertEqual(self.client.health()["gpu_count"], 1)
        self.assertIn("fast_helical_native", self.client.workflows()["workflows"])

    def test_the_workflow_schema_names_settings_a_submission_may_carry(self):
        body = self.client.workflow("fast_helical_native")
        names = [setting["name"] for setting in body["settings"]]
        self.assertIn("resolution", names)
        # What the endpoint publishes has to be what `submit` accepts —
        # otherwise a client that reads the schema builds a submission the
        # server's strict check then refuses.
        self.client.submit(
            reference_image=str(self._sheet()),
            settings={setting["name"]: setting["default"] for setting in body["settings"]},
        )
        self.assertEqual(len(self.submitted), 1)

    def test_the_openapi_document_comes_back(self):
        self.assertIn(f"{API_PREFIX}/runs", self.client.schema()["paths"])

    # -- auth --------------------------------------------------------------

    def test_a_wrong_token_is_an_ApiError_carrying_the_servers_sentence(self):
        # 401 with no token at all is `tests/test_api.py`'s: the server
        # reads the variable per request and runs in this process, so
        # clearing it here would disarm the server rather than the client.
        with self.assertRaises(ApiError) as caught:
            B2CClient(self.url, "nope").runs()
        self.assertEqual(caught.exception.status, 401)
        self.assertIn("B2C_API_TOKEN", caught.exception.detail)
        self.assertIn(API_PREFIX, caught.exception.url)

    def test_the_token_comes_from_the_environment_when_not_passed(self):
        # $B2C_API_TOKEN is the same variable the pod is given, so the
        # common case is having it exported already.
        self.assertEqual(B2CClient(self.url).health()["status"], "ok")

    # -- submitting --------------------------------------------------------

    def test_an_uploaded_sheet_reaches_the_worker_as_a_durable_copy(self):
        submitted = self.client.submit(
            reference_image=str(self._sheet()), prompt="a woman in a red jacket",
        )
        self.assertEqual(len(submitted), 1)
        job = self.submitted[0]
        self.assertEqual(job.prompt, "a woman in a red jacket")
        # Not the client's path: the request's copy is gone by the time the
        # worker starts, so the job has to name one on the volume.
        self.assertNotEqual(job.reference_image, str(self.data / "sheet.png"))
        self.assertTrue(Path(job.reference_image).is_file())

    def test_a_zip_fans_out_and_the_client_gets_every_name(self):
        archive = self.data / "batch.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for stem in ("alice", "bob"):
                bundle.writestr(f"{stem}.png", _png())
                bundle.writestr(f"{stem}.txt", f"a portrait of {stem}")
        submitted = self.client.submit(reference_image=str(archive))
        self.assertEqual(len(submitted), 2)
        self.assertEqual(
            sorted(job.prompt for job in self.submitted),
            ["a portrait of alice", "a portrait of bob"],
        )

    def test_a_sheet_already_on_the_pod_is_named_rather_than_uploaded(self):
        """The JSON body reaches the same run without the bytes crossing the wire.

        The server still takes its own durable copy — a path a caller named
        may be somewhere as temporary as `/tmp`, and the worker reads it
        much later — so this is not a copy the pod avoids. It is the upload
        it avoids: pushing hundreds of MB through a pod's HTTP proxy to
        reach a disk they are already on.
        """
        sheet = self._sheet("already-there.png", value=77)
        self.client.submit(remote_path=str(sheet), prompt="on the volume")
        job = self.submitted[0]
        self.assertEqual(job.prompt, "on the volume")
        self.assertEqual(Path(job.reference_image).read_bytes(), sheet.read_bytes())
        self.assertTrue(Path(job.reference_image).is_file())

    def test_settings_and_step_params_travel(self):
        self.client.submit(
            reference_image=str(self._sheet()),
            settings={"run_upscale": False, "seed": 11},
            step_params={"denoise_pass1": {"steps": 3}},
        )
        job = self.submitted[0]
        self.assertIs(job.global_overrides["run_upscale"], False)
        self.assertEqual(job.global_overrides["seed"], 11)
        self.assertEqual(job.step_overrides, {"denoise_pass1": {"steps": 3}})

    def test_a_refusal_arrives_with_the_servers_own_wording(self):
        with self.assertRaises(ApiError) as caught:
            self.client.submit(
                reference_image=str(self._sheet()),
                settings={"export_colmap": False, "export_ply": False},
            )
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("Pick at least one output", caught.exception.detail)
        self.assertEqual(self.submitted, [])

    def test_naming_both_or_neither_source_is_refused_before_any_request(self):
        for kwargs in ({}, {"reference_image": "a.png", "remote_path": "b.png"}):
            with self.assertRaises(ValueError):
                self.client.submit(**kwargs)
        self.assertEqual(self.submitted, [])

    def test_a_local_file_that_is_not_there(self):
        with self.assertRaises(ValueError):
            self.client.submit(reference_image=str(self.data / "missing.png"))

    # -- watching and collecting -------------------------------------------

    def test_runs_lists_the_live_one_first(self):
        self.finished_run(name="older-run")
        name = self.client.submit(reference_image=str(self._sheet()))[0]["name"]
        self.assertEqual([state["name"] for state in self.client.runs()],
                         [name, "older-run"])

    def test_one_run_by_name(self):
        self.finished_run(name="older-run")
        self.assertEqual(self.client.run("older-run")["status"], "done")

    def test_an_unknown_run_is_a_404(self):
        with self.assertRaises(ApiError) as caught:
            self.client.run("nope")
        self.assertEqual(caught.exception.status, 404)

    def test_the_log_comes_back_tailed(self):
        self.finished_run(name="logged")
        self.assertEqual(
            self.client.log("logged", tail=3).splitlines(),
            ["line 197", "line 198", "line 199"],
        )

    def test_the_result_streams_into_a_directory_named_by_the_server(self):
        self.finished_run(name="packaged")
        into = self.data / "downloads"
        path = self.client.download_result("packaged", into)
        self.assertEqual(path.parent, into)
        self.assertEqual(path.name, "packaged-result.zip")
        with zipfile.ZipFile(path) as bundle:
            self.assertIn("colmap/cameras.txt", bundle.namelist())
        # Nothing half-written left behind.
        self.assertEqual([p.name for p in into.glob("*.part")], [])

    def test_a_zip_destination_names_the_file_itself(self):
        self.finished_run(name="packaged")
        path = self.client.download_result("packaged", self.data / "mine.zip")
        self.assertEqual(path.name, "mine.zip")

    def test_an_existing_non_zip_file_as_the_destination_is_refused(self):
        # It fell into `mkdir` and raised a bare FileExistsError out of
        # `dispatch`, which special-cases ApiError/ValueError/transport —
        # a traceback where this module gives sentences everywhere else.
        self.finished_run(name="packaged")
        notes = self.data / "notes"
        notes.write_text("not a zip")
        with self.assertRaises(ValueError) as caught:
            self.client.download_result("packaged", notes)
        self.assertIn("refusing to overwrite", str(caught.exception).lower())
        self.assertEqual(notes.read_text(), "not a zip")

    def test_a_destination_that_does_not_exist_yet_is_a_directory(self):
        # The one shape of this argument that is always a mistake: `-o out`
        # used to write a *file* called `out` holding a zip.
        self.finished_run(name="packaged")
        path = self.client.download_result("packaged", self.data / "out")
        self.assertTrue(path.parent.is_dir())
        self.assertEqual(path.name, "packaged-result.zip")

    def test_progress_is_reported_against_a_known_total(self):
        self.finished_run(name="packaged")
        seen = []
        self.client.download_result(
            "packaged", self.data / "dl", on_progress=lambda w, t: seen.append((w, t)),
        )
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], seen[-1][1], "did not finish at 100%")

    def test_a_run_that_produced_nothing_has_no_result(self):
        self.finished_run(name="empty-run", deliverables=False)
        with self.assertRaises(ApiError) as caught:
            self.client.download_result("empty-run", self.data / "dl")
        self.assertEqual(caught.exception.status, 404)

    def test_the_result_is_refused_while_the_run_is_going(self):
        name = self.client.submit(reference_image=str(self._sheet()))[0]["name"]
        with self.assertRaises(ApiError) as caught:
            self.client.download_result(name, self.data / "dl")
        self.assertEqual(caught.exception.status, 409)

    def test_cancelling_a_running_run(self):
        name = self.client.submit(reference_image=str(self._sheet()))[0]["name"]
        self.assertEqual(self.client.cancel(name)["name"], name)

    def test_cancelling_a_finished_run_is_refused(self):
        self.finished_run(name="older-run")
        with self.assertRaises(ApiError) as caught:
            self.client.cancel("older-run")
        self.assertEqual(caught.exception.status, 409)

    # -- stopping ----------------------------------------------------------

    def _wait_for_stop(self, timeout: float = 5.0) -> None:
        """The reply precedes the work, so wait for the work.

        Deliberately not asserted inline: `POST /shutdown` answers 202 and
        *then* stops, which is what lets a client read the reply at all
        rather than having the connection cut from under it. Against a real
        server that ordering is visible; `fastapi.testclient` hides it by
        running background tasks synchronously.
        """
        deadline = time.monotonic() + timeout
        while not self.stopped and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.stopped, ["server"])

    def test_shutdown_asks_the_server_to_stop(self):
        body = self.client.shutdown()
        self.assertIs(body["stopping"], True)
        self.assertEqual(body["abandoned_runs"], [])
        self._wait_for_stop()

    def test_the_reply_arrives_before_the_server_stops(self):
        # If the work happened inline the client would get a reset
        # connection instead of a body it can read.
        self.assertEqual(self.stopped, [])
        self.assertIs(self.client.shutdown()["stopping"], True)
        self._wait_for_stop()

    def test_shutdown_is_refused_while_a_run_is_going(self):
        name = self.client.submit(reference_image=str(self._sheet()))[0]["name"]
        with self.assertRaises(ApiError) as caught:
            self.client.shutdown()
        self.assertEqual(caught.exception.status, 409)
        self.assertIn(name, caught.exception.detail)
        self.assertEqual(self.stopped, [])

    def test_shutdown_force_stops_regardless(self):
        name = self.client.submit(reference_image=str(self._sheet()))[0]["name"]
        self.assertEqual(self.client.shutdown(force=True)["abandoned_runs"], [name])
        self._wait_for_stop()

    def test_a_workflow_whose_file_no_longer_resolves_still_packages(self):
        """A renamed workflow must not turn packaging into a 500.

        `cli.resolve_workflow` refuses with `SystemExit`, a BaseException
        that passes straight through a request handler — so a run recorded
        against a workflow that has since been renamed (or one submitted by
        path) took `result_subdirs`' documented fallback nowhere and
        answered 500 instead.
        """
        run_dir = runs.output_dir() / "orphaned"
        (run_dir / "colmap").mkdir(parents=True, exist_ok=True)
        (run_dir / "colmap" / "cameras.txt").write_text("# CAMERAS\n")
        (runs.run_jobs_dir() / "orphaned.status.json").write_text(json.dumps(RunState(
            name="orphaned", workflow="a_workflow_that_was_deleted", status="done",
            output_dir=run_dir, started=1.0, finished=2.0,
        ).to_dict()))

        path = self.client.download_result("orphaned", self.data / "dl")
        with zipfile.ZipFile(path) as bundle:
            self.assertIn("colmap/cameras.txt", bundle.namelist())


class _ScriptedClient(B2CClient):
    """A client whose `run()` replays a fixed list of `RunState` dicts."""

    def __init__(self, states):
        super().__init__("http://example.invalid", "token")
        self._states = list(states)

    def run(self, name):
        return self._states.pop(0) if len(self._states) > 1 else self._states[0]


def _state(status, steps=(), **extra):
    return dict(
        {
            "name": "r", "workflow": "w", "status": status, "message": "",
            "current": 0, "total": len(steps), "started": 0.0, "finished": 0.0,
            "steps": [
                {"index": i, "step_id": sid, "step_name": sid, "status": st, "elapsed": el}
                for i, (sid, st, el) in enumerate(steps, start=1)
            ],
        },
        **extra,
    )


class TestFollow(unittest.TestCase):
    """Whole-state polls in, one event per real transition out."""

    def _events(self, states):
        client = _ScriptedClient(states)
        return list(client.follow("r", interval=0, sleep=lambda _: None))

    def test_each_step_is_reported_once_with_what_it_cost(self):
        events = self._events([
            _state("running", [("a", "running", 0.0)]),
            _state("running", [("a", "done", 2.5), ("b", "running", 0.0)]),
            _state("running", [("a", "done", 2.5), ("b", "done", 41.0)]),
            _state("done", [("a", "done", 2.5), ("b", "done", 41.0)]),
        ])
        steps = [(e["step_id"], e["elapsed"]) for e in events if e["kind"] == "step"]
        self.assertEqual(steps, [("a", 2.5), ("b", 41.0)])

    def test_a_poll_that_caught_several_reports_them_in_order(self):
        events = self._events([
            _state("running", [("a", "running", 0.0), ("b", "pending", 0.0)]),
            _state("done", [("a", "done", 1.0), ("b", "done", 2.0)]),
        ])
        self.assertEqual(
            [e["step_id"] for e in events if e["kind"] == "step"], ["a", "b"]
        )

    def test_the_outcome_is_reported_below_the_stages_that_produced_it(self):
        # A run that finishes between two polls hands back its terminal
        # status and its last steps together. Printing the status first
        # would put "done" above the stages it is the outcome of.
        events = self._events([
            _state("running", [("a", "running", 0.0)]),
            _state("done", [("a", "done", 1.0)], message="complete"),
        ])
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds.index("step"), kinds.index("end") - 1)
        self.assertEqual(kinds[-1], "end")
        self.assertEqual(kinds.count("end"), 1)

    def test_a_terminal_status_is_only_ever_the_end_event(self):
        events = self._events([_state("failed", [("a", "failed", 0.5)])])
        self.assertEqual([e["kind"] for e in events], ["step", "end"])
        self.assertEqual(events[-1]["status"], "failed")

    def test_end_reports_how_the_run_ended_not_what_its_last_step_did(self):
        # A `when:`-gated tail is ordinary — an unchecked output switches
        # its export steps off — so the last step of a perfectly good run
        # is often `skipped`. Reporting that as the run's outcome would
        # make `end` disagree with the state it carries.
        events = self._events([
            _state("running", [("a", "done", 1.0), ("b", "pending", 0.0)]),
            _state("done", [("a", "done", 1.0), ("b", "skipped", 0.0)]),
        ])
        end = events[-1]
        self.assertEqual(end["kind"], "end")
        self.assertEqual(end["status"], "done")
        self.assertEqual(end["status"], end["state"]["status"])

    def test_queued_then_running_is_two_status_events(self):
        events = self._events([
            _state("queued"), _state("running"), _state("running"), _state("done"),
        ])
        self.assertEqual(
            [e["status"] for e in events if e["kind"] == "status"], ["queued", "running"]
        )

    def test_a_message_is_reported_once_per_change(self):
        # "waiting for model download: ..." is the one that matters — it is
        # otherwise indistinguishable from a run hung before step one.
        events = self._events([
            _state("queued", message="waiting for model download: wan22"),
            _state("running", message="waiting for model download: wan22"),
            _state("running", message="[1/2] split"),
            _state("done"),
        ])
        self.assertEqual(
            [e["text"] for e in events if e["kind"] == "message"],
            ["waiting for model download: wan22", "[1/2] split"],
        )

    def test_a_skipped_step_is_reported_too(self):
        # `when:`-gated steps are how an output switch takes effect, so a
        # run of 42 steps that skipped 6 should say so rather than appear
        # to have lost them.
        events = self._events([
            _state("running", [("a", "skipped", 0.0)]),
            _state("done", [("a", "skipped", 0.0)]),
        ])
        step = next(e for e in events if e["kind"] == "step")
        self.assertEqual(step["status"], "skipped")

    def test_a_dropped_connection_is_retried_rather_than_raised(self):
        """A watch runs for hours across a rented pod's proxy.

        A reset in the middle is ordinary, and abandoning a run that is
        still going perfectly well on the other side would be the wrong
        answer to it.
        """
        import requests

        states = [_state("running"), _state("done")]
        failures = [2]

        class _Flaky(_ScriptedClient):
            def run(self, name):
                if failures[0]:
                    failures[0] -= 1
                    raise requests.ConnectionError("proxy reset")
                return super().run(name)

        slept = []
        events = list(_Flaky(states).follow(
            "r", interval=0, sleep=slept.append,
        ))
        self.assertEqual(events[-1]["kind"], "end")
        self.assertEqual(events[-1]["status"], "done")
        # Backed off between attempts rather than hammering.
        self.assertEqual(slept[:2], [4.0, 8.0])

    def test_a_gateway_error_is_retried(self):
        # 502/503/504 from a pod proxy is the transport, not the server
        # answering — as ordinary as a reset, and the docstring promises to
        # survive one.
        from pipeline.client import ApiError

        states = [_state("running"), _state("done")]
        failures = [2]

        class _Flaky(_ScriptedClient):
            def run(self, name):
                if failures[0]:
                    failures[0] -= 1
                    raise ApiError(503, "upstream unavailable")
                return super().run(name)

        events = list(_Flaky(states).follow("r", interval=0, sleep=lambda _: None))
        self.assertEqual(events[-1]["status"], "done")

    def test_a_404_is_not_retried(self):
        from pipeline.client import ApiError

        calls = [0]

        class _Gone(_ScriptedClient):
            def run(self, name):
                calls[0] += 1
                raise ApiError(404, "No run named 'r'.")

        with self.assertRaises(ApiError):
            list(_Gone([_state("running")]).follow("r", interval=0, sleep=lambda _: None))
        self.assertEqual(calls[0], 1)

    def test_a_server_that_stays_gone_ends_the_watch(self):
        # Bounded: a pod that has actually been released should stop the
        # watch, with the error it stopped on, not poll forever.
        import requests

        class _Dead(_ScriptedClient):
            def run(self, name):
                raise requests.ConnectionError("gone")

        with self.assertRaises(requests.ConnectionError):
            list(_Dead([_state("running")]).follow("r", interval=0, sleep=lambda _: None))

    def test_an_api_error_is_not_retried(self):
        # The server answered; it will answer the same way next time.
        from pipeline.client import ApiError

        calls = [0]

        class _Refusing(_ScriptedClient):
            def run(self, name):
                calls[0] += 1
                raise ApiError(404, "No run named 'r'.")

        with self.assertRaises(ApiError):
            list(_Refusing([_state("running")]).follow("r", interval=0, sleep=lambda _: None))
        self.assertEqual(calls[0], 1)

    def test_a_run_already_finished_yields_its_steps_and_one_end(self):
        events = self._events([_state("done", [("a", "done", 1.0)])])
        self.assertEqual([e["kind"] for e in events], ["step", "end"])


class TestShutdownWhenDone(unittest.TestCase):
    """`api run --shutdown-when-done` stops only once the .zip is on disk.

    Stopping the container is what makes everything still on its volume
    unreachable, so doing it on a download that failed turns a recoverable
    afternoon into a lost one.
    """

    def _run(self, *, download_ok: bool, flag: bool = True):
        import argparse

        from pipeline import api_cli

        calls = []

        class _Client:
            def submit(self, **kwargs):
                return [{"name": "r"}]

            def follow(self, name, interval=5.0, sleep=None):
                yield {"kind": "end", "status": "done",
                       "state": {"name": "r", "status": "done",
                                 "started": 1.0, "finished": 2.0, "message": "complete"}}

            def download_result(self, name, into, on_progress=None):
                calls.append("download")
                if not download_ok:
                    raise ApiError(404, "produced no deliverables")
                path = Path(into)
                path.mkdir(parents=True, exist_ok=True)
                target = path / "r-result.zip"
                target.write_bytes(b"zip")
                return target

            def shutdown(self, force=False):
                calls.append("shutdown")
                return {"stopping": True, "abandoned_runs": [],
                        "stops_gpu_workers": 1, "host_command": ""}

        args = argparse.Namespace(
            image="sheet.png", prompt="", remote=False, param=None, workflow="",
            output=self._tmp.name, interval=0.0, download_anyway=False,
            shutdown_when_done=flag,
        )
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with unittest.mock.patch.object(api_cli, "_client", lambda _a: _Client()):
            with redirect_stdout(buffer):
                code = api_cli.cmd_run(args)
        return calls, code, buffer.getvalue()

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_a_collected_result_is_followed_by_a_shutdown(self):
        calls, code, _ = self._run(download_ok=True)
        self.assertEqual(calls, ["download", "shutdown"])
        self.assertEqual(code, 0)

    def test_a_failed_download_stops_nothing(self):
        calls, code, output = self._run(download_ok=False)
        self.assertEqual(calls, ["download"])
        self.assertNotEqual(code, 0)
        self.assertIn("not shutting down", output)

    def test_without_the_flag_it_never_shuts_down(self):
        calls, _, _ = self._run(download_ok=True, flag=False)
        self.assertEqual(calls, ["download"])


class TestUnreachableServerMessages(unittest.TestCase):
    """What `dispatch` says when the connection fails, per subcommand."""

    def _dispatch(self, api_command, **extra):
        import argparse
        import io
        from contextlib import redirect_stderr

        import requests

        from pipeline import api_cli

        def boom(_args):
            raise requests.ConnectionError("refused")

        args = argparse.Namespace(api_command=api_command, func=boom, **extra)
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            code = api_cli.dispatch(args)
        return code, buffer.getvalue()

    def test_a_run_command_says_how_to_reattach(self):
        code, output = self._dispatch("follow", name="r")
        self.assertEqual(code, 1)
        self.assertIn("api follow r", output)

    def test_shutdown_does_not_offer_to_reattach_to_a_run(self):
        # The likeliest reason the server is unreachable right after a
        # shutdown is that it did what it was asked.
        code, output = self._dispatch("shutdown")
        self.assertEqual(code, 1)
        self.assertNotIn("api follow", output)
        self.assertIn("stopped server", output)

    def test_a_readonly_command_says_neither(self):
        _code, output = self._dispatch("health")
        self.assertIn("could not reach the server", output)
        self.assertNotIn("api follow", output)


class TestTheSummaryLine(unittest.TestCase):
    """`api run`'s last line, for the states a run can actually end in."""

    def _summary(self, state, log="") -> str:
        import io
        from contextlib import redirect_stdout

        from pipeline.api_cli import _report_end

        class _Client(B2CClient):
            def log(self, name, tail=200):
                return log

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _report_end(_Client("http://example.invalid", "t"), state)
        return buffer.getvalue()

    def test_a_run_cancelled_before_it_started_has_no_wall_clock(self):
        # `GpuScheduler.cancel` stamps `finished` on a queued run and
        # leaves `started` at 0, so subtracting one from the other gives
        # the Unix epoch — half a million hours of "wall clock".
        summary = self._summary({
            "name": "r", "status": "cancelled", "started": 0.0,
            "finished": 1_757_000_000.0, "message": "cancelled before it started",
        })
        self.assertIn("CANCELLED: cancelled before it started", summary)
        self.assertNotIn("wall clock", summary)

    def test_a_worker_that_never_spawned_has_no_wall_clock(self):
        summary = self._summary({
            "name": "r", "status": "failed", "started": 0.0,
            "finished": 1_757_000_000.0, "message": "failed to start a GPU worker process",
        })
        self.assertNotIn("wall clock", summary)

    def test_a_failed_run_carries_its_log_tail(self):
        summary = self._summary(
            {"name": "r", "status": "failed", "started": 1.0, "finished": 5.0,
             "message": "RuntimeError: out of memory"},
            log="CUDA out of memory",
        )
        self.assertIn("FAILED: RuntimeError: out of memory", summary)
        self.assertIn("CUDA out of memory", summary)

    def test_an_unreachable_log_does_not_replace_the_outcome(self):
        # Losing the connection between the last poll and this fetch must
        # not swap the answer for a transport error.
        import io
        from contextlib import redirect_stdout

        import requests

        from pipeline.api_cli import _report_end

        class _Client(B2CClient):
            def log(self, name, tail=200):
                raise requests.ConnectionError("proxy reset")

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _report_end(_Client("http://example.invalid", "t"),
                        {"name": "r", "status": "failed", "started": 1.0,
                         "finished": 5.0, "message": "it broke"})
        summary = buffer.getvalue()
        self.assertIn("FAILED: it broke", summary)
        self.assertIn("could not read the log", summary)

    def test_a_run_that_really_ran_reports_its_wall_clock(self):
        summary = self._summary({
            "name": "r", "status": "done", "started": 1_000.0,
            "finished": 1_000.0 + 7872, "message": "complete — 81 frames",
        })
        self.assertIn("(2h11m wall clock)", summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
