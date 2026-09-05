"""HTTP API: submit a run, watch it, pull the result back — without a browser.

The web UI is a person's view of `pipeline.runs`; this is a script's. Both
mount into the same server process on the same port, and both go through
`runs.submit_runs`, so a curl and a button press queue identical work
against the same `GpuScheduler` — one queue, one set of GPU slots, one
answer to "what is running".

That sharing is the reason this is a router built against a live scheduler
rather than a separate server: a second process with a scheduler of its own
would happily start a run on a GPU the UI already had busy, and neither
would know.

Everything is under `/api/v1` and everything needs `B2C_API_TOKEN` as a
bearer token. With no token in the environment the router is never built —
`pipeline.webui.launch` says so in the log and serves the UI alone. That is
deliberate: the pod's URL is public, an API that queues hours of GPU time
is not something to leave open by accident, and "off unless configured"
fails safe where "open unless configured" does not.

Responses are `RunState.to_dict()` verbatim rather than a second set of
declared models, so the wire format cannot drift from what the UI reads off
the same status files. The generated schema is at `/api/v1/openapi.json`,
behind the same token as everything else.

    curl -H "Authorization: Bearer $B2C_API_TOKEN" \
         -F file=@sheet.png -F prompt='a woman in a red jacket' \
         -F 'settings={"run_upscale": false}' \
         http://pod:7860/api/v1/runs
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .gpu_scheduler import GpuScheduler
from .run_state import tail_lines
from .runs import (
    WORKFLOW_NATIVE, SubmitError, build_result_zip, find_run, merged_runs,
    resolve_upload, run_log_path, submit_runs, workflow_param_panel,
)

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"

# Where the UI's login form and this router's bearer check both read from.
TOKEN_ENV = "B2C_API_TOKEN"

# Content types whose body FastAPI has already consumed by the time a route
# body runs, because the route declares Form/File params.
_FORM_TYPES = {"multipart/form-data", "application/x-www-form-urlencoded"}


def api_token() -> Optional[str]:
    """The shared secret, or None when the deployment has not set one."""
    return os.environ.get(TOKEN_ENV) or None


def _require_token(request: Request) -> None:
    """`Authorization: Bearer <token>`, compared without leaking its length.

    A plain `==` on a secret is a timing oracle; `compare_digest` is the
    stdlib's answer and costs nothing here. The 401 says only that the
    header was wrong — never which part of it.
    """
    expected = api_token()
    if not expected:  # pragma: no cover - the router is not built without one
        raise HTTPException(status_code=503, detail="No API token is configured.")
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    # Bytes, not str: `compare_digest` raises TypeError on str arguments
    # outside ASCII, and Starlette decodes header bytes as latin-1 — so a
    # header with one accented character would be a 500 and a traceback,
    # from a caller who has not authenticated at all.
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        presented.strip().encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Send the pod's B2C_API_TOKEN as `Authorization: Bearer <token>`.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class SubmitBody(BaseModel):
    """A submission whose reference sheet is already on the volume.

    The multipart form is the other way in, and the right one for a sheet
    on your laptop. This one is for a file put there by `rsync`,
    `runpodctl send` or an earlier submission — pushing hundreds of MB back
    through a pod's HTTP proxy to reach a disk it is already on is a slow
    way to achieve nothing.
    """

    # Refused, not ignored — the same rule `submit_runs(strict=True)`
    # applies to the settings inside. A misspelled `promt` accepted here
    # would queue an hour of GPU with an empty prompt and a 202 that says
    # everything went fine.
    model_config = ConfigDict(extra="forbid")

    reference_image: str = Field(
        ..., description="Path on the pod to a reference sheet, or a .zip of image/prompt pairs.",
    )
    prompt: str = ""
    settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow settings and output switches, by name — "
                    "GET /api/v1/workflows/{name} lists them.",
    )
    step_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description='Per-step overrides: {"step_id": {"param": value}}.',
    )
    workflow: str = WORKFLOW_NATIVE


def _parse_json_field(raw: Optional[str], field: str) -> Dict[str, Any]:
    """One multipart field carrying a JSON object, or {} when it is absent.

    Multipart has no types, so `settings` arrives as a string. Refusing a
    malformed one here, by name, beats letting it through as a no-op — an
    override that silently does nothing is the failure mode this pipeline's
    `--param` handling already goes out of its way to prevent.
    """
    if not raw or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field}: not valid JSON ({exc}).")
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=400, detail=f"{field}: expected a JSON object, got {type(value).__name__}."
        )
    return value


def _submitted(scheduler: GpuScheduler, names: List[str]) -> Dict[str, Any]:
    return {
        "runs": [scheduler.snapshot(name).to_dict() for name in names],
        "queued": scheduler.queued_count(),
        "gpu_count": scheduler.gpu_count,
    }


def _param_json(param) -> Dict[str, Any]:
    """One declared setting, as much of it as a client can act on.

    `Param.type` is a real Python type (`int`, `list`, ...); it goes over
    the wire as its name, which is also the spelling a workflow's
    `settings:` block uses.
    """
    return {
        "name": param.name,
        "label": param.title,
        "type": param.type.__name__,
        "default": param.default,
        "choices": list(param.choices),
        "minimum": param.minimum,
        "maximum": param.maximum,
        "advanced": param.advanced,
        "group": param.group,
        "help": " ".join(param.help.split()),
    }


def add_schema_route(app, prefix: str = API_PREFIX) -> None:
    """Publish the OpenAPI schema at `<prefix>/openapi.json`, behind the token.

    FastAPI's own `openapi_url`/`docs_url` put those routes on the *app*,
    outside the router's dependency — so on a public pod URL they would
    advertise every route of an otherwise-guarded API to anyone who asked.
    This is the same schema, on the router's terms.

    No Swagger UI page to go with it, deliberately: that page fetches its
    schema from the browser with no bearer header, so behind this guard it
    would render empty and read as a broken API rather than a protected
    one. `curl -H "Authorization: Bearer ..." .../openapi.json` is the
    whole story, and it feeds a generator directly.
    """

    @app.get(f"{prefix}/openapi.json", include_in_schema=False,
             dependencies=[Depends(_require_token)])
    def openapi_schema() -> Dict[str, Any]:
        return app.openapi()


def build_router(scheduler: GpuScheduler, envs_path: str = "") -> APIRouter:
    """The whole API, bound to the scheduler the UI is already driving."""
    router = APIRouter(dependencies=[Depends(_require_token)])

    # -- the box ----------------------------------------------------------

    @router.get("/health")
    def health() -> Dict[str, Any]:
        """Whether there is capacity, and whether the weights have landed.

        The model status is here because it is the usual reason a submitted
        run sits at `queued` doing nothing visible: a cold volume is pulling
        ~72 GB before the first step can start, and a client that cannot see
        that has no way to tell it apart from a wedged pod.
        """
        from .models import read_status

        return {
            "status": "ok",
            "gpu_count": scheduler.gpu_count,
            "gpus": scheduler.gpu_status(),
            "queued": scheduler.queued_count(),
            "models": read_status(),
        }

    # -- what can be asked for --------------------------------------------

    @router.get("/workflows")
    def list_workflows() -> Dict[str, Any]:
        from .cli import available_workflows

        return {
            "workflows": [p.stem for p in available_workflows()],
            "default": WORKFLOW_NATIVE,
        }

    @router.get("/workflows/{name}")
    def get_workflow(name: str) -> Dict[str, Any]:
        """The `settings:` and `outputs:` blocks the workflow declares.

        This is how a client discovers what `settings` keys are legal, and
        what each one accepts, without reading the YAML off the pod — the
        same declarations the UI draws its two boxes from.
        """
        try:
            spec, settings, outputs, _steps = workflow_param_panel(name)
        except SystemExit as exc:  # cli.resolve_workflow's "no such workflow"
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "name": spec.name,
            "description": spec.description.strip(),
            "settings": [_param_json(p) for p in settings],
            "outputs": [
                {
                    "name": o.name, "label": o.label, "dir": o.directory,
                    "default": o.default, "requires": o.requires, "help": o.help,
                }
                for o in outputs
            ],
        }

    # -- submitting --------------------------------------------------------

    @router.post("/runs", status_code=202)
    async def create_runs(
        request: Request,
        file: Optional[UploadFile] = File(
            None, description="A reference sheet, or a .zip of image/prompt pairs.",
        ),
        prompt: str = Form(""),
        settings: Optional[str] = Form(None, description="A JSON object of workflow settings."),
        step_params: Optional[str] = Form(None, description="A JSON object of per-step overrides."),
        workflow: str = Form(WORKFLOW_NATIVE),
    ) -> Dict[str, Any]:
        """Queue one run, or one per image in an uploaded .zip.

        Returns as soon as the jobs are queued — `submit()` never blocks on
        a free GPU — so a 202 means "accepted", not "started". Poll
        `GET /runs/{name}` for the difference.
        """
        if file is not None and file.filename:
            # In the threadpool, not on the event loop: staging spools the
            # upload to disk and `_queue` extracts and copies it, which for
            # a few hundred MB of zip is seconds of blocking I/O — and the
            # Gradio UI is mounted on this same loop in this same process,
            # so doing it here would freeze every progress stream on the box.
            names = await run_in_threadpool(
                _staged_submit, file, prompt, settings, step_params, workflow,
            )
            return _submitted(scheduler, names)

        # Declaring File/Form params makes FastAPI parse the form on every
        # request, which consumes the stream — so `request.body()` below
        # would raise "Stream consumed" rather than reach `_submit_body`'s
        # refusal. A form body that carried no file is the plain case of
        # forgetting `-F file=@sheet.png`, and deserves that sentence.
        if request.headers.get("content-type", "").split(";")[0].strip() in _FORM_TYPES:
            raise HTTPException(
                status_code=400,
                detail="This submission carried form fields but no `file`. Send the "
                       "reference sheet as multipart `file`, or send a JSON body "
                       "naming one already on the pod as `reference_image`.",
            )

        body = _submit_body(await request.body())
        upload = Path(body.reference_image)
        if not upload.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"No such file on this pod: {body.reference_image}. "
                       "Upload it as multipart `file`, or put it on the volume first.",
            )
        names = await run_in_threadpool(
            _queue, upload, body.prompt, json.dumps(body.settings),
            json.dumps(body.step_params), body.workflow,
        )
        return _submitted(scheduler, names)

    def _staged_submit(file: UploadFile, prompt, settings, step_params, workflow) -> List[str]:
        with _staged(file) as upload:
            return _queue(upload, prompt, settings, step_params, workflow)

    def _queue(upload: Path, prompt: str, settings, step_params, workflow) -> List[str]:
        """Resolve one submission and hand it to `runs.submit_runs`.

        The refusals both front ends share arrive as `SubmitError`, whose
        message is already a sentence written to be shown to someone — so it
        becomes the 400's detail verbatim rather than being restated here.
        """
        try:
            plan = resolve_upload(str(upload), prompt)
            return submit_runs(
                scheduler, plan,
                _parse_json_field(settings, "settings"),
                _parse_json_field(step_params, "step_params"),
                envs_path=envs_path, workflow=workflow,
                # A script has no stale param panel to forgive — an override
                # it misspells is a typo, and one that silently does nothing
                # costs hours of GPU at the wrong settings.
                strict=True,
            )
        except SubmitError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except SystemExit as exc:  # cli.resolve_workflow's "no such workflow"
            raise HTTPException(status_code=400, detail=str(exc)) from None

    # -- watching ----------------------------------------------------------

    @router.get("/runs")
    def list_runs() -> Dict[str, Any]:
        """Every run on this box: this session's and the volume's.

        In flight first, then most recently finished — so the run a caller
        is waiting on is at the top rather than at the bottom, which is
        where a just-queued run's (absent) timestamps would put it.
        """
        return {"runs": [state.to_dict() for state in merged_runs(scheduler)]}

    @router.get("/runs/{name}")
    def get_run(name: str) -> Dict[str, Any]:
        return _lookup(name).to_dict()

    @router.get("/runs/{name}/log")
    def get_log(name: str, tail: int = Query(200, ge=1, le=20000)) -> Dict[str, Any]:
        """The last `tail` lines of the run's log, growing file and all."""
        state = _lookup(name)
        path = run_log_path(state.output_dir, state.log_path)
        if path is None:
            raise HTTPException(
                status_code=404, detail=f"No log on this volume for run {name!r}."
            )
        return {"name": name, "path": str(path), "log": tail_lines(path, max_lines=tail)}

    @router.get("/runs/{name}/result")
    def get_result(name: str) -> FileResponse:
        """The run's deliverables as one .zip — `colmap/`, `ply/`, `debug/`, `log.txt`.

        Refused while the run is still going: the exports are its last
        steps, so packaging one mid-flight hands back half a dataset that
        looks like a whole one.
        """
        state = _lookup(name)
        # `state.status` alone is not enough: a container that died mid-run
        # leaves its status file at `running` for good, so a run whose
        # colmap/ and ply/ are sitting complete on the volume could never
        # be collected. Only this server's own scheduler can say a run is
        # really still going — same test `cancel_run` makes below.
        if state.status in ("queued", "running") and scheduler.snapshot(name).name == name:
            raise HTTPException(
                status_code=409,
                detail=f"Run {name!r} is {state.status}; its exports are the last steps.",
            )
        archive = build_result_zip(
            state.output_dir, state.workflow, state.log_path, reuse=True
        )
        if archive is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run {name!r} produced no deliverables"
                       f"{' (' + state.message + ')' if state.message else ''}.",
            )
        return FileResponse(
            archive, media_type="application/zip", filename=Path(archive).name
        )

    @router.post("/runs/{name}/cancel")
    def cancel_run(name: str) -> Dict[str, Any]:
        """Stop a run at its next step boundary, or drop it from the queue.

        Not mid-step: a step is one opaque call, often a subprocess holding
        the GPU, and tearing one down in flight risks leaving the card in a
        state the next run on that slot inherits. So this returns
        immediately with the run still `running`, and it ends when the step
        it is in does.
        """
        state = _lookup(name)
        if state.status not in ("queued", "running"):
            raise HTTPException(
                status_code=409, detail=f"Run {name!r} has already finished ({state.status})."
            )
        if scheduler.snapshot(name).name != name:
            # Found on the volume, not in this scheduler: a run left at
            # `running` by a server that has since restarted, or one a
            # `pipeline.cli run` started from an SSH shell. `cancel` would
            # match no queued job and no slot, do nothing, and hand back a
            # blank RunState under a 200 — telling the caller it worked.
            raise HTTPException(
                status_code=409,
                detail=f"Run {name!r} is not one this server started, so it cannot be "
                       "cancelled from here. Its status file may be stale — check "
                       "whether the process is still alive on the pod.",
            )
        scheduler.cancel(name)
        return scheduler.snapshot(name).to_dict()

    def _lookup(name: str):
        state = find_run(scheduler, name)
        if state is None:
            raise HTTPException(status_code=404, detail=f"No run named {name!r}.")
        return state

    return router


def _submit_body(raw: bytes) -> SubmitBody:
    """The JSON body of a submission that carried no file.

    Parsed by hand rather than declared as the route's body model, because
    the route accepts multipart *or* JSON and FastAPI will only validate
    one of them for you. Which means the model's own errors have to be
    turned into 400s here too — a body missing `reference_image` is a
    request to fix, not a server fault.
    """
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Send a reference sheet as multipart `file`, or a JSON body "
                   "naming one already on the pod as `reference_image`.",
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Body is not valid JSON ({exc}).")
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")
    try:
        return SubmitBody(**value)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Body: {exc}") from None


@contextmanager
def _staged(file: UploadFile):
    """An upload spooled to a temp file, kept only until `resolve_upload` runs.

    Streamed to disk rather than held in memory — a zip of reference sheets
    is tens of megabytes and there is one of these per request — and
    deliberately *not* written to the volume here. `resolve_upload` is what
    decides where a submission's input lives (a durable copy for a single
    sheet, an extraction directory for a zip); doing it here as well would
    leave a second copy of every upload on the volume that nothing reads.
    """
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "").suffix) as staged:
        shutil.copyfileobj(file.file, staged)
        staged.flush()
        yield Path(staged.name)
