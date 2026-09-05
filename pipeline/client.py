"""A client for the pipeline's HTTP API — `pipeline.api` from the outside.

Why this is in the repo rather than left to curl: an API whose only caller
is documentation drifts from it. Everything under `pipeline.cli api` goes
through `B2CClient`, so the routes have a real caller that a test exercises
against a real server, and the recipes in docs/runpod.md have something to
be checked against.

This module deliberately needs nothing but `requests`, which is already a
hard dependency — no gradio, no fastapi, no torch — so it drops into a
script of your own. Reached through `pipeline.cli api` it costs the rest of
the core install as well (that module imports the step registry), which is
still a plain `pip install -r requirements.txt`: the machine driving a pod
does not have to be able to serve one.

    from pipeline.client import B2CClient

    client = B2CClient("https://<pod-id>-7860.proxy.runpod.net", token)
    name = client.submit(reference_image="sheet.png", prompt="...")[0]["name"]
    for event in client.follow(name):
        print(event)
    client.download_result(name, Path("."))

`follow` is the one worth knowing about: it turns polling into a stream of
things that actually happened — a step finished, and how long it took —
by diffing the `steps` list between polls. The server publishes a whole
`RunState` each time and says nothing about what changed; this is where
that becomes a log you can watch.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:7860"
URL_ENV = "B2C_API_URL"
TOKEN_ENV = "B2C_API_TOKEN"
API_PREFIX = "/api/v1"

# A run is over exactly when its status leaves these two — the same rule
# `runs.completed_runs` uses, and the reason a poll loop needs no other
# termination condition.
LIVE = ("queued", "running")

# How often `follow` asks. The server's own scheduler refreshes a run's
# status about once a second, so anything under that samples the same
# answer twice; anything over it delays a step boundary by up to that
# much. Five seconds costs a handful of requests per hour of run.
POLL_SECONDS = 5.0

# How long `follow` keeps trying after the connection drops, and how it
# backs off. A watch runs for hours across a rented pod's HTTP proxy, so a
# reset somewhere in the middle is ordinary rather than exceptional — and
# giving up on one would abandon a run that is still going perfectly well
# on the other side. Roughly five minutes, which outlasts a proxy blip and
# a pod's own brief unreachability without hiding a pod that has gone.
POLL_RETRIES = 8
POLL_BACKOFF = 4.0


class ApiError(RuntimeError):
    """A refusal from the server, with the sentence it gave.

    `pipeline.api` puts something a person can act on in every `detail` —
    which setting was misspelled, that a run is still running, that the
    submission would export nothing. Carrying it through unchanged is the
    whole job here; `status` is for callers that want to branch on 404 vs.
    409 rather than read the text.
    """

    def __init__(self, status: int, detail: str, url: str = "") -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.url = url


class B2CClient:
    """Every route of `pipeline.api`, one method each."""

    def __init__(
        self,
        url: str = "",
        token: str = "",
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.url = (url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self.token = token or os.environ.get(TOKEN_ENV, "")
        self.timeout = timeout
        self._session = session or requests.Session()
        if self.token:
            self._session.headers["Authorization"] = f"Bearer {self.token}"

    # -- plumbing ---------------------------------------------------------

    def _endpoint(self, path: str) -> str:
        return urljoin(self.url + "/", (API_PREFIX + path).lstrip("/"))

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        endpoint = self._endpoint(path)
        kwargs.setdefault("timeout", self.timeout)
        response = self._session.request(method, endpoint, **kwargs)
        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response), endpoint)
        return response

    def _json(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        return self._request(method, path, **kwargs).json()

    # -- the box ----------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/health")

    def workflows(self) -> Dict[str, Any]:
        return self._json("GET", "/workflows")

    def workflow(self, name: str) -> Dict[str, Any]:
        return self._json("GET", f"/workflows/{name}")

    def schema(self) -> Dict[str, Any]:
        """The generated OpenAPI document."""
        return self._json("GET", "/openapi.json")

    # -- submitting --------------------------------------------------------

    def submit(
        self,
        reference_image: Optional[str] = None,
        remote_path: str = "",
        prompt: str = "",
        settings: Optional[Dict[str, Any]] = None,
        step_params: Optional[Dict[str, Dict[str, Any]]] = None,
        workflow: str = "",
    ) -> List[Dict[str, Any]]:
        """Queue a run per reference sheet; return their `RunState` dicts.

        `reference_image` uploads a local file (an image, or a `.zip` of
        image/prompt pairs). `remote_path` names one already on the pod
        instead — the right choice for anything large, since the upload
        would otherwise cross the pod's HTTP proxy to reach a disk it is
        already on.
        """
        if bool(reference_image) == bool(remote_path):
            raise ValueError("Pass exactly one of reference_image or remote_path.")

        fields: Dict[str, Any] = {"prompt": prompt}
        if settings:
            fields["settings"] = json.dumps(settings)
        if step_params:
            fields["step_params"] = json.dumps(step_params)
        if workflow:
            fields["workflow"] = workflow

        if remote_path:
            body = {"reference_image": remote_path, "prompt": prompt}
            if settings:
                body["settings"] = settings
            if step_params:
                body["step_params"] = step_params
            if workflow:
                body["workflow"] = workflow
            return self._json("POST", "/runs", json=body)["runs"]

        path = Path(reference_image or "")
        if not path.is_file():
            raise ValueError(f"No such file: {path}")
        # Streamed from the handle rather than read into memory: a zip of
        # reference sheets is tens of megabytes and there is no reason for
        # this process to hold one.
        with open(path, "rb") as handle:
            return self._json(
                "POST", "/runs",
                files={"file": (path.name, handle, "application/octet-stream")},
                data=fields,
                # Uploading a batch over a pod proxy takes longer than a
                # status poll ever will.
                timeout=max(self.timeout, 600.0),
            )["runs"]

    # -- watching ----------------------------------------------------------

    def runs(self) -> List[Dict[str, Any]]:
        return self._json("GET", "/runs")["runs"]

    def run(self, name: str) -> Dict[str, Any]:
        return self._json("GET", f"/runs/{name}")

    def log(self, name: str, tail: int = 200) -> str:
        return self._json("GET", f"/runs/{name}/log", params={"tail": tail})["log"]

    def cancel(self, name: str) -> Dict[str, Any]:
        return self._json("POST", f"/runs/{name}/cancel")

    def follow(
        self,
        name: str,
        interval: float = POLL_SECONDS,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Poll one run, yielding what changed rather than what it is.

        The server republishes a whole `RunState` and says nothing about
        which parts are new, so a caller that printed every poll would
        print the same thing hundreds of times. This diffs successive
        snapshots and yields one event per real transition:

            {"kind": "status",  "status": ..., "state": {...}}
            {"kind": "message", "text": ...}          # "waiting for models: ..."
            {"kind": "step",    "index":, "step_id":, "step_name":,
             "status": "done"|"failed"|"skipped", "elapsed": float,
             "total": int}
            {"kind": "end",     "status": ..., "state": {...}}

        Ends when the run's status leaves queued/running, and always yields
        exactly one `end` — which is the only place a terminal status is
        reported, so the stages of a poll always print above the outcome
        they produced. Steps are reported in index order even if a poll
        caught several at once, so the output reads as the run happened.

        A dropped connection is retried rather than raised: watching is not
        driving, the run is a process on the pod, and a proxy resetting
        once during a two-hour watch should not abandon it. Retries are
        bounded — a pod that has actually gone still ends the watch, with
        the connection error it ended on.
        """
        import time

        sleep = sleep or time.sleep
        seen: Dict[int, str] = {}
        last_status = ""
        last_message = ""

        while True:
            state = self._poll(name, sleep)
            status = state.get("status", "")
            over = status not in LIVE

            # `status` and `message` describe a run that is still going;
            # the terminal ones belong to `end`, which carries the whole
            # final state. Emitting them here as well would put "done" and
            # the completion message *above* the stages that produced them
            # whenever a run finishes between two polls — which is every
            # short run, and every long one whose last stage lands just
            # after a poll.
            if not over:
                if status != last_status:
                    last_status = status
                    yield {"kind": "status", "status": status, "state": state}

                message = state.get("message") or ""
                if message != last_message:
                    last_message = message
                    if message:
                        yield {"kind": "message", "text": message}

            for step in sorted(state.get("steps") or [], key=lambda s: s["index"]):
                # `step_status`, not `status`: rebinding the run's status
                # here left `end` reporting whatever the last step had done
                # rather than how the run ended.
                step_status = step.get("status", "pending")
                if (step_status in ("pending", "running")
                        or seen.get(step["index"]) == step_status):
                    continue
                seen[step["index"]] = step_status
                yield {
                    "kind": "step", "index": step["index"], "step_id": step["step_id"],
                    "step_name": step["step_name"], "status": step_status,
                    "elapsed": step.get("elapsed", 0.0), "total": state.get("total", 0),
                }

            if over:
                yield {"kind": "end", "status": status, "state": state}
                return
            sleep(interval)

    def _poll(self, name: str, sleep: Callable[[float], None]) -> Dict[str, Any]:
        """One `GET /runs/{name}`, retried through a transient outage.

        Only the transport is retried. An `ApiError` — the run went away, a
        token stopped working — is the server answering, and answering the
        same way next time.
        """
        for attempt in range(POLL_RETRIES):
            try:
                return self.run(name)
            except requests.RequestException as exc:
                if attempt == POLL_RETRIES - 1:
                    raise
                delay = POLL_BACKOFF * (attempt + 1)
                logger.warning(
                    "lost contact with %s (%s); retrying in %.0fs (%d/%d)",
                    self.url, exc, delay, attempt + 1, POLL_RETRIES - 1,
                )
                sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    # -- collecting --------------------------------------------------------

    def download_result(
        self,
        name: str,
        destination: Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Stream the run's `.zip` to `destination`; return the file written.

        `destination` names a file only when it ends in `.zip`; anything
        else is a directory, created if it does not exist, and the server's
        own filename is used inside it. Without that rule `-o results/` on a
        directory that does not exist yet silently produced a *file* called
        `results` holding a zip — the one shape of this argument that is
        always a mistake.

        Written to a `.part` beside the target and moved into place, so an
        interrupted download cannot be mistaken for a complete archive by
        whatever reads the directory next.
        """
        # (connect, read): a read timeout would kill a legitimate download
        # of a couple of gigabytes over a pod proxy, but a connection that
        # never opens should still give up rather than hang.
        response = self._request(
            "GET", f"/runs/{name}/result", stream=True, timeout=(self.timeout, None),
        )
        target = Path(destination)
        if target.is_dir() or target.suffix.lower() != ".zip":
            target.mkdir(parents=True, exist_ok=True)
            target = target / _filename(response, fallback=f"{name}-result.zip")
        target.parent.mkdir(parents=True, exist_ok=True)

        total = int(response.headers.get("content-length") or 0)
        staging = target.with_name(target.name + ".part")
        written = 0
        try:
            with open(staging, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
                    if on_progress:
                        on_progress(written, total)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        return target


def _detail(response: requests.Response) -> str:
    """The server's own sentence, or something honest when there isn't one."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or response.reason or "").strip()[:500]
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    return json.dumps(detail if detail is not None else body)[:500]


_FILENAME = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?')


def _filename(response: requests.Response, fallback: str) -> str:
    """The name from `Content-Disposition`, stripped of any path in it.

    `Path(...).name` because the header is remote input and this becomes a
    path on the caller's disk: a server answering `filename="../../x"`
    must not decide where the download lands.
    """
    match = _FILENAME.search(response.headers.get("content-disposition", ""))
    name = Path(match.group(1)).name if match else ""
    return name or fallback
