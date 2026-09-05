"""The HTTP API: what it accepts, what it refuses, and what it hands back.

Three things these are really pinning:

  * **A submission means the same thing here as in the browser.** Both go
    through `runs.submit_runs`, so what is checked is that the router hands
    it the right plan and overrides and that the resulting `RunJob` is the
    one a button press would have queued.
  * **It is not reachable without the token, and not served without one
    either.** `TestMountedServer` builds the real server the pod runs and
    asserts the API answers under the Gradio mount — `mount_gradio_app`
    mounts at `/`, which matches every path there is, so an API registered
    after it would 404 silently rather than fail loudly.
  * **A run is not packaged mid-flight.** The exports are a run's last
    steps; a `.zip` built while it is still going looks whole and is not.

No GPU and no worker: `GpuScheduler` takes its `spawn` as an argument, so
the process it would have started is a stub. Runs that need to be *finished*
are written onto the volume as the status files a worker publishes, which
is also the path `discover_runs` takes for a run this process never saw.
"""

from __future__ import annotations

import io
import json
import os
import unittest
import unittest.mock
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline import runs
from pipeline.api import API_PREFIX, TOKEN_ENV, build_router
from pipeline.gpu_scheduler import GpuScheduler
from pipeline.run_state import RunState

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _png(width: int = 128, height: int = 64) -> bytes:
    import cv2

    return cv2.imencode(".png", np.zeros((height, width, 3), np.uint8))[1].tobytes()


class _StubProcess:
    """A worker that was started and is still going.

    `poll()` returning None is what keeps the scheduler's slot busy and the
    run at `running`, which is the state most of these want to submit into.
    """

    returncode = None

    def poll(self):
        return None

    def terminate(self):
        self.returncode = -15


class ApiTestCase(unittest.TestCase):
    """A router over a real scheduler whose worker process is a stub."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name)

        env = unittest.mock.patch.dict(
            os.environ, {"B2C_DATA_DIR": str(self.data), TOKEN_ENV: TOKEN}
        )
        env.start()
        self.addCleanup(env.stop)
        # Everything else (output, uploads, run_jobs, logs) derives from
        # B2C_DATA_DIR and is read at call time, so this one variable
        # redirects the whole module onto the temp volume.
        for stale in ("B2C_OUTPUT_DIR", "B2C_UPLOAD_DIR", "B2C_RUN_JOBS_DIR", "B2C_LOG_DIR"):
            os.environ.pop(stale, None)

        # Two lists, because they answer different questions: `submitted`
        # is every job the router queued (what a submission *meant*),
        # `spawned` only the ones a free GPU slot actually started (what the
        # scheduler *did* with them).
        self.submitted = []
        self.spawned = []
        self.scheduler = GpuScheduler(
            gpu_count=1, work_dir=self.data / "run_jobs", spawn=self._spawn
        )
        queue = self.scheduler.submit
        self.scheduler.submit = lambda job: (self.submitted.append(job), queue(job))[1]
        self.addCleanup(self.scheduler.shutdown)

        app = FastAPI()
        app.include_router(build_router(self.scheduler, "envs.yaml"), prefix=API_PREFIX)
        self.client = TestClient(app, raise_server_exceptions=False)

    def _spawn(self, job, gpu_index, job_path, status_path):
        self.spawned.append(job)
        return _StubProcess()

    # -- helpers ----------------------------------------------------------

    def post(self, **kwargs):
        return self.client.post(f"{API_PREFIX}/runs", headers=AUTH, **kwargs)

    def submit_sheet(self, **data):
        return self.post(files={"file": ("sheet.png", _png(), "image/png")}, data=data)

    def finished_run(
        self, name="fast_helical_native-20260904-101500-abc123",
        status="done", deliverables=True, log=True,
    ) -> Path:
        """A run on the volume as a worker would have left it."""
        run_dir = runs.output_dir() / name
        if deliverables:
            (run_dir / "colmap").mkdir(parents=True, exist_ok=True)
            (run_dir / "colmap" / "cameras.txt").write_text("# CAMERAS\n")
            (run_dir / "colmap" / "images.txt").write_text("# IMAGES\n")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
        log_path = None
        if log:
            log_path = runs.log_dir() / f"{name}.log"
            log_path.write_text("\n".join(f"line {i}" for i in range(500)))
        state = RunState(
            name=name, workflow="fast_helical_native", status=status,
            output_dir=run_dir, log_path=log_path, started=1.0, finished=2.0,
        )
        (runs.run_jobs_dir() / f"{name}.status.json").write_text(
            json.dumps(state.to_dict())
        )
        return run_dir


class TestTheTokenGuardsEverything(ApiTestCase):
    def test_no_authorization_header_is_refused(self):
        self.assertEqual(self.client.get(f"{API_PREFIX}/runs").status_code, 401)

    def test_a_wrong_token_is_refused(self):
        response = self.client.get(
            f"{API_PREFIX}/runs", headers={"Authorization": "Bearer nope"}
        )
        self.assertEqual(response.status_code, 401)

    def test_a_non_ascii_header_is_refused_not_a_server_error(self):
        # `hmac.compare_digest` raises TypeError on str arguments outside
        # ASCII, and Starlette decodes headers as latin-1 — so this used to
        # be a 500 and a traceback from a caller who never authenticated.
        # httpx will not send such a header, so the dependency is called
        # directly.
        from fastapi import HTTPException

        from pipeline.api import _require_token

        class _Request:
            headers = {"authorization": "Bearer \xfc"}

        with self.assertRaises(HTTPException) as caught:
            _require_token(_Request())
        self.assertEqual(caught.exception.status_code, 401)

    def test_the_right_token_in_the_wrong_scheme_is_refused(self):
        response = self.client.get(
            f"{API_PREFIX}/runs", headers={"Authorization": f"Basic {TOKEN}"}
        )
        self.assertEqual(response.status_code, 401)

    def test_submitting_is_guarded_too_not_just_reading(self):
        response = self.client.post(
            f"{API_PREFIX}/runs", files={"file": ("s.png", _png(), "image/png")}
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.submitted, [])


class TestSubmitting(ApiTestCase):
    def test_an_uploaded_sheet_becomes_one_queued_run(self):
        response = self.submit_sheet(prompt="a woman in a red jacket")
        self.assertEqual(response.status_code, 202)

        names = [run["name"] for run in response.json()["runs"]]
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith("fast_helical_native-"))

        self.assertEqual(len(self.submitted), 1)
        job = self.submitted[0]
        self.assertEqual(job.prompt, "a woman in a red jacket")
        self.assertEqual(job.workflow_name, "fast_helical_native")
        self.assertEqual(job.output_dir, str(runs.output_dir() / names[0]))

    def test_the_sheet_is_copied_onto_the_volume_exactly_once(self):
        # The request's own temp file is gone by the time the worker starts,
        # so the job has to name a durable copy — and only one: an upload
        # staged here *and* saved by resolve_upload would leave a second
        # copy of every submission that nothing ever reads.
        self.submit_sheet()
        job = self.submitted[0]
        self.assertTrue(Path(job.reference_image).is_file())
        self.assertEqual(
            [p.name for p in runs.upload_dir().iterdir()],
            [Path(job.reference_image).name],
        )

    def test_two_submissions_in_the_same_second_get_their_own_sheet(self):
        # Seconds are not fine-grained enough to name an upload: without a
        # random suffix the second `shutil.copy` overwrote the first, both
        # jobs named one path, and the run queued for subject A started an
        # hour later on subject B's photograph.
        self.submit_sheet()
        self.submit_sheet()
        sheets = {job.reference_image for job in self.submitted}
        self.assertEqual(len(sheets), 2, "two submissions shared one reference image")
        self.assertEqual(len(list(runs.upload_dir().iterdir())), 2)

    def test_two_zips_in_the_same_second_do_not_share_a_directory(self):
        # `_guarded_extract` is mkdir(exist_ok=True), so a shared name let
        # the second archive unpack into the first one's tree and
        # `pair_images_with_prompts` then fanned out over both.
        for stem in ("x", "y"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(f"{stem}1.png", _png())
            response = self.post(
                files={"file": (f"{stem}.zip", buffer.getvalue(), "application/zip")}
            )
            self.assertEqual(len(response.json()["runs"]), 1, "fanned out over both zips")
        self.assertEqual(
            sorted(Path(job.reference_image).stem for job in self.submitted), ["x1", "y1"]
        )

    def test_a_zip_of_pairs_fans_out_to_one_run_per_image(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("alice", "bob"):
                archive.writestr(f"{name}.png", _png())
                archive.writestr(f"{name}.txt", f"a portrait of {name}")

        response = self.post(files={"file": ("batch.zip", buffer.getvalue(), "application/zip")})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(len(response.json()["runs"]), 2)

        # Each run carries its own .txt as the prompt, and is named after
        # its image so the two are told apart in a picker.
        self.assertEqual(
            sorted((job.prompt, Path(job.reference_image).stem) for job in self.submitted),
            [("a portrait of alice", "alice"), ("a portrait of bob", "bob")],
        )
        for job, stem in zip(self.submitted, ("alice", "bob")):
            self.assertIn(f"fast_helical_native-{stem}-", job.run_name)

    def test_only_one_run_starts_when_there_is_one_gpu(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("a", "b", "c"):
                archive.writestr(f"{name}.png", _png())
        response = self.post(
            files={"file": ("batch.zip", buffer.getvalue(), "application/zip")},
            data={"prompt": "shared"},
        )
        self.assertEqual(len(self.submitted), 3)
        self.assertEqual(len(self.spawned), 1, "a one-GPU box started more than one run")
        self.assertEqual(response.json()["queued"], 2)
        self.assertEqual([job.prompt for job in self.submitted], ["shared"] * 3)

    def test_settings_reach_the_job(self):
        self.submit_sheet(settings=json.dumps({"run_upscale": False, "seed": 7}))
        overrides = self.submitted[0].global_overrides
        self.assertIs(overrides["run_upscale"], False)
        self.assertEqual(overrides["seed"], 7)

    def test_step_params_reach_the_job_under_their_step_id(self):
        self.submit_sheet(step_params=json.dumps({"denoise_pass1": {"steps": 3}}))
        self.assertEqual(self.submitted[0].step_overrides, {"denoise_pass1": {"steps": 3}})

    def test_an_output_whose_requires_is_off_is_forced_off(self):
        # Same rule the UI's greyed-out checkbox states: with the upscale
        # off, colmap_preupscale/ would be colmap/ under a name that says
        # otherwise. The worker re-reads the pristine YAML, so the switch
        # has to travel in the overrides, not only in the spec.
        self.submit_sheet(settings=json.dumps(
            {"run_upscale": False, "export_colmap_preupscale": True}
        ))
        self.assertIs(self.submitted[0].global_overrides["export_colmap_preupscale"], False)

    def test_a_json_body_can_name_a_sheet_already_on_the_volume(self):
        sheet = self.data / "staged.png"
        sheet.write_bytes(_png())
        response = self.post(json={
            "reference_image": str(sheet), "prompt": "already here",
            "settings": {"seed": 3},
        })
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.submitted[0].prompt, "already here")
        self.assertEqual(self.submitted[0].global_overrides["seed"], 3)


class TestWhatItRefuses(ApiTestCase):
    def assertRefused(self, response, fragment):
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(fragment, response.json()["detail"])
        self.assertEqual(self.submitted, [], "a refused submission still queued work")

    def test_a_submission_that_exports_nothing(self):
        self.assertRefused(
            self.submit_sheet(settings=json.dumps({
                "export_colmap": False, "export_ply": False,
            })),
            "Pick at least one output",
        )

    def test_a_file_that_is_neither_an_image_nor_a_zip(self):
        self.assertRefused(
            self.post(files={"file": ("notes.txt", b"hello", "text/plain")}),
            "Don't know what to do with a .txt file",
        )

    def test_settings_that_are_not_json(self):
        self.assertRefused(self.submit_sheet(settings="{oops"), "settings: not valid JSON")

    def test_settings_that_are_json_but_not_an_object(self):
        self.assertRefused(
            self.submit_sheet(settings="[1, 2]"), "settings: expected a JSON object"
        )

    def test_a_setting_this_workflow_does_not_declare(self):
        # Lenient for the browser (a panel can go stale), strict here: an
        # override that silently does nothing costs hours of GPU at the
        # wrong settings and looks like a run that honoured you.
        response = self.submit_sheet(settings=json.dumps({"resolutoin": [720, 1280]}))
        self.assertRefused(response, "resolutoin")
        self.assertIn("resolution", response.json()["detail"])

    def test_output_root_which_has_no_control_anywhere(self):
        # It is a bare `globals:` entry, so the UI cannot draw it — and the
        # worker only repoints it at the run's own directory when the
        # submission did NOT pin it. Accepting one here would put several
        # GB of splat and COLMAP export wherever the caller named, outside
        # the directory /result then looks in.
        self.assertRefused(
            self.submit_sheet(settings=json.dumps({"output_root": "/tmp/elsewhere"})),
            "output_root",
        )

    def test_a_step_that_does_not_exist(self):
        self.assertRefused(
            self.submit_sheet(step_params=json.dumps({"denoise_pass9": {"steps": 3}})),
            "No such step(s)",
        )

    def test_a_param_the_step_does_not_declare(self):
        self.assertRefused(
            self.submit_sheet(step_params=json.dumps({"denoise_pass1": {"stpes": 3}})),
            "does not declare stpes",
        )

    def test_a_setting_of_the_wrong_type(self):
        # ParamError is a ValueError; without the translation in
        # submit_runs this is a 500 for what is plainly a bad request.
        response = self.submit_sheet(settings=json.dumps({"seed": "soon"}))
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.submitted, [])

    def test_a_json_body_naming_a_file_that_is_not_there(self):
        self.assertRefused(
            self.post(json={"reference_image": "/data/nope.png"}), "No such file on this pod"
        )

    def test_a_request_carrying_neither_a_file_nor_a_body(self):
        self.assertRefused(self.post(), "multipart `file`")

    def test_form_fields_with_no_file_at_all(self):
        # Declaring File/Form params makes FastAPI consume the stream, so
        # the JSON fallback's `request.body()` raised "Stream consumed" and
        # this was a 500 — for the plain case of forgetting `-F file=@...`.
        self.assertRefused(self.post(data={"prompt": "forgot the sheet"}), "no `file`")

    def test_a_typod_top_level_body_field(self):
        # The same rule strict overrides apply inside `settings`: a
        # misspelled `promt` accepted here would queue an hour of GPU with
        # an empty prompt and a 202 saying everything was fine.
        self.assertRefused(
            self.post(json={"reference_image": "/x.png", "promt": "a woman"}), "promt"
        )

    def test_a_json_body_missing_the_reference_image(self):
        # The route takes multipart *or* JSON, so the body model is applied
        # by hand — which means its own errors have to become 400s here
        # too, rather than surfacing as a server fault.
        self.assertRefused(self.post(json={"prompt": "no sheet"}), "reference_image")

    def test_a_json_body_whose_settings_are_not_an_object(self):
        self.assertRefused(
            self.post(json={"reference_image": "/x.png", "settings": "nope"}), "settings"
        )

    def test_an_unknown_workflow(self):
        self.assertRefused(self.submit_sheet(workflow="fast_helical_shell"), "No such workflow")


class TestWatching(ApiTestCase):
    def test_a_submitted_run_is_listed_and_readable_by_name(self):
        name = self.submit_sheet().json()["runs"][0]["name"]

        listed = self.client.get(f"{API_PREFIX}/runs", headers=AUTH).json()["runs"]
        self.assertEqual([r["name"] for r in listed], [name])

        one = self.client.get(f"{API_PREFIX}/runs/{name}", headers=AUTH).json()
        self.assertEqual(one["status"], "running")
        self.assertEqual(one["gpu_index"], 0)

    def test_the_volumes_runs_are_listed_beside_this_sessions(self):
        # The scheduler forgets everything on restart; the volume does not.
        self.finished_run(name="older-run")
        live = self.submit_sheet().json()["runs"][0]["name"]
        listed = self.client.get(f"{API_PREFIX}/runs", headers=AUTH).json()["runs"]
        self.assertEqual({r["name"] for r in listed}, {"older-run", live})

    def test_a_run_still_in_flight_is_listed_first(self):
        # `run_recency` scores a just-queued run 0 — no started, no
        # finished, no output directory to take an mtime from — which would
        # bury the one run a caller is actually waiting on at the bottom.
        self.finished_run(name="older-run")
        live = self.submit_sheet().json()["runs"][0]["name"]
        listed = self.client.get(f"{API_PREFIX}/runs", headers=AUTH).json()["runs"]
        self.assertEqual([r["name"] for r in listed], [live, "older-run"])

    def test_a_run_this_process_never_saw_is_still_readable(self):
        self.finished_run(name="older-run")
        response = self.client.get(f"{API_PREFIX}/runs/older-run", headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "done")

    def test_an_unknown_run_is_a_404(self):
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/runs/nope", headers=AUTH).status_code, 404
        )

    def test_the_log_comes_back_tailed_not_whole(self):
        self.finished_run(name="logged")
        response = self.client.get(
            f"{API_PREFIX}/runs/logged/log", headers=AUTH, params={"tail": 5}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["log"].splitlines(),
            [f"line {i}" for i in range(495, 500)],
        )

    def test_a_run_with_no_log_on_the_volume(self):
        self.finished_run(name="unlogged", log=False)
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/runs/unlogged/log", headers=AUTH).status_code, 404
        )

    def test_health_reports_the_slots_and_the_queue(self):
        body = self.client.get(f"{API_PREFIX}/health", headers=AUTH).json()
        self.assertEqual(body["gpu_count"], 1)
        self.assertEqual(body["gpus"], [{"gpu": 0, "busy": False, "run": ""}])
        self.assertEqual(body["queued"], 0)
        # The usual reason a queued run appears to do nothing for half an
        # hour is a cold volume still pulling ~72 GB of weights.
        self.assertIn("models", body)


class TestWhatAWorkflowDeclares(ApiTestCase):
    def test_the_settings_are_published_with_their_types_and_choices(self):
        body = self.client.get(
            f"{API_PREFIX}/workflows/fast_helical_native", headers=AUTH
        ).json()
        by_name = {s["name"]: s for s in body["settings"]}
        self.assertEqual(by_name["seed"]["type"], "int")
        self.assertEqual(by_name["resolution"]["type"], "list")
        # A Python type would not survive JSON, and the choices are lists
        # rather than the tuples the dataclass holds.
        self.assertEqual(by_name["resolution"]["choices"][0], [720, 1280])
        self.assertIs(by_name["face_splat"]["advanced"], True)

    def test_the_outputs_are_published_with_their_directories_and_requires(self):
        body = self.client.get(
            f"{API_PREFIX}/workflows/fast_helical_native", headers=AUTH
        ).json()
        by_name = {o["name"]: o for o in body["outputs"]}
        self.assertEqual(by_name["export_colmap"]["dir"], "colmap")
        self.assertEqual(by_name["export_colmap_preupscale"]["requires"], "run_upscale")

    def test_every_declared_setting_is_a_name_submit_will_accept(self):
        # The point of the endpoint: what it lists is what `settings`
        # takes. A drift here means a client reading the schema builds a
        # submission the strict check then refuses.
        body = self.client.get(
            f"{API_PREFIX}/workflows/fast_helical_native", headers=AUTH
        ).json()
        response = self.submit_sheet(settings=json.dumps(
            {s["name"]: s["default"] for s in body["settings"]}
        ))
        self.assertEqual(response.status_code, 202, response.text)

    def test_the_shipped_workflows_are_listed(self):
        body = self.client.get(f"{API_PREFIX}/workflows", headers=AUTH).json()
        self.assertIn("fast_helical_native", body["workflows"])
        self.assertEqual(body["default"], "fast_helical_native")

    def test_an_unknown_workflow_is_a_404(self):
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/workflows/nope", headers=AUTH).status_code, 404
        )


class TestCollectingTheResult(ApiTestCase):
    def test_a_finished_runs_deliverables_come_back_as_one_zip(self):
        self.finished_run(name="done-run")
        response = self.client.get(f"{API_PREFIX}/runs/done-run/result", headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(
                sorted(archive.namelist()),
                ["colmap/cameras.txt", "colmap/images.txt", "log.txt"],
            )

    def test_it_is_refused_while_the_run_is_still_going(self):
        # Its exports are the last steps: packaging one now hands back half
        # a dataset that looks like a whole one.
        name = self.submit_sheet().json()["runs"][0]["name"]
        response = self.client.get(f"{API_PREFIX}/runs/{name}/result", headers=AUTH)
        self.assertEqual(response.status_code, 409)
        self.assertIn("running", response.json()["detail"])

    def test_a_stale_running_status_does_not_lock_the_result_away(self):
        # A container that died mid-run leaves its status file at
        # `running` for good. The deliverables are on the volume; refusing
        # on the published status alone would make them unreachable
        # through the API forever.
        self.finished_run(name="orphaned-run", status="running")
        response = self.client.get(f"{API_PREFIX}/runs/orphaned-run/result", headers=AUTH)
        self.assertEqual(response.status_code, 200, response.text)

    def test_a_finished_run_that_produced_nothing_has_no_result(self):
        self.finished_run(name="empty-run", deliverables=False)
        response = self.client.get(f"{API_PREFIX}/runs/empty-run/result", headers=AUTH)
        self.assertEqual(response.status_code, 404)
        self.assertIn("no deliverables", response.json()["detail"])

    def test_a_cancelled_run_still_hands_back_what_it_managed_to_export(self):
        self.finished_run(name="stopped-run", status="cancelled")
        self.assertEqual(
            self.client.get(f"{API_PREFIX}/runs/stopped-run/result", headers=AUTH).status_code,
            200,
        )


class TestCancelling(ApiTestCase):
    def test_cancelling_a_running_run_terminates_its_worker(self):
        name = self.submit_sheet().json()["runs"][0]["name"]
        response = self.client.post(f"{API_PREFIX}/runs/{name}/cancel", headers=AUTH)
        self.assertEqual(response.status_code, 200)
        # Not mid-step: the worker is signalled and stops at the next step
        # boundary, so this returns with the run still running.
        self.assertEqual(
            [slot["run"] for slot in self.scheduler.gpu_status()], [name]
        )

    def test_cancelling_a_queued_run_drops_it_before_it_starts(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("a", "b"):
                archive.writestr(f"{name}.png", _png())
        queued = self.post(
            files={"file": ("batch.zip", buffer.getvalue(), "application/zip")}
        ).json()["runs"][1]["name"]

        body = self.client.post(f"{API_PREFIX}/runs/{queued}/cancel", headers=AUTH).json()
        self.assertEqual(body["status"], "cancelled")
        self.assertEqual(self.scheduler.queued_count(), 0)
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(len(self.submitted), 2)

    def test_cancelling_a_finished_run_is_refused(self):
        self.finished_run(name="done-run")
        response = self.client.post(f"{API_PREFIX}/runs/done-run/cancel", headers=AUTH)
        self.assertEqual(response.status_code, 409)

    def test_cancelling_a_run_this_server_did_not_start_is_refused(self):
        # A stale `running` status file — left by a server that restarted,
        # or by a `pipeline.cli run` from an SSH shell. `cancel` matches no
        # queued job and no slot, so this used to answer 200 with a blank
        # RunState, telling the caller it had worked.
        self.finished_run(name="ghost-run", status="running")
        response = self.client.post(f"{API_PREFIX}/runs/ghost-run/cancel", headers=AUTH)
        self.assertEqual(response.status_code, 409)
        self.assertIn("not one this server started", response.json()["detail"])

    def test_cancelling_an_unknown_run_is_a_404(self):
        self.assertEqual(
            self.client.post(f"{API_PREFIX}/runs/nope/cancel", headers=AUTH).status_code, 404
        )


try:
    from pipeline import webui
except ImportError:  # pragma: no cover - depends on the local env
    webui = None


@unittest.skipIf(webui is None, "the web UI's dependencies are not installed here")
class TestMountedServer(unittest.TestCase):
    """The API and the UI on one port, which is the shape the pod serves.

    `gr.mount_gradio_app` finishes with `app.mount("/", ...)`, and Starlette
    walks its routes in registration order — so an API attached after that
    mount is unreachable, with no error anywhere, just the UI's 404 where
    JSON should be. That is the regression a Gradio upgrade could introduce
    silently, and it is what this class exists for.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = unittest.mock.patch.dict(
            os.environ, {"B2C_DATA_DIR": self._tmp.name, TOKEN_ENV: TOKEN}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _client(self, **kwargs):
        return TestClient(
            webui.build_server(gpu_count=1, **kwargs), raise_server_exceptions=False
        )

    def test_the_api_answers_from_under_the_ui_mount(self):
        response = self._client().get(f"{API_PREFIX}/health", headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_the_api_routes_come_before_the_catch_all_mount(self):
        paths = [getattr(route, "path", None) for route in self._client().app.routes]
        self.assertIn("", paths, "the Gradio app is not mounted at all")
        included = next(
            i for i, route in enumerate(self._client().app.routes)
            if type(route).__name__ != "Route" and getattr(route, "path", None) != ""
        )
        self.assertLess(included, paths.index(""))

    def test_the_ui_is_still_served(self):
        # Behind Gradio's own login now, so an unauthenticated browser is
        # redirected to /login rather than handed the app.
        response = self._client().get("/", follow_redirects=False)
        self.assertIn(response.status_code, (200, 302, 307))

    def test_the_schema_is_behind_the_token_too(self):
        # FastAPI's own docs_url/openapi_url hang off the app, outside the
        # router's dependency — which on a public pod URL would advertise
        # every route of a guarded API to anyone who asked.
        client = self._client()
        self.assertEqual(client.get(f"{API_PREFIX}/openapi.json").status_code, 401)
        response = client.get(f"{API_PREFIX}/openapi.json", headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            sorted(response.json()["paths"]),
            [
                f"{API_PREFIX}/health",
                f"{API_PREFIX}/runs",
                f"{API_PREFIX}/runs/{{name}}",
                f"{API_PREFIX}/runs/{{name}}/cancel",
                f"{API_PREFIX}/runs/{{name}}/log",
                f"{API_PREFIX}/runs/{{name}}/result",
                f"{API_PREFIX}/workflows",
                f"{API_PREFIX}/workflows/{{name}}",
            ],
        )

    def test_no_token_means_no_api_rather_than_an_open_one(self):
        with unittest.mock.patch.dict(os.environ, {TOKEN_ENV: ""}):
            response = self._client().get(f"{API_PREFIX}/health", headers=AUTH)
        # Swallowed by the Gradio mount: there is no API to reach.
        self.assertNotEqual(response.status_code, 200)

    def test_no_api_serves_the_ui_alone_even_with_a_token(self):
        response = self._client(serve_api=False).get(f"{API_PREFIX}/health", headers=AUTH)
        self.assertNotEqual(response.status_code, 200)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
