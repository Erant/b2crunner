"""`python -m pipeline.cli api ...` — drive a running pod from the outside.

One subcommand per route, so every path has a caller that is not a curl
recipe in a document, plus one that does the whole job:

    api run       submit, follow the stages as they finish, download the .zip
    api follow    the same watch, attached to a run already going
    api submit    queue it and return; for a batch loop that polls later
    api status | runs | log | cancel | result | health | workflows

`api run` is the one to reach for. A run is tens of minutes to hours of
stages, and the two questions while it goes are "which stage is it on" and
"is anything wrong" — so it prints each stage as it completes with what it
cost, which is also the timing data you would otherwise dig out of the log
afterwards.

Connection comes from `--url`/`$B2C_API_URL` and `--token`/`$B2C_API_TOKEN`.
On the pod itself the defaults already work.

`--param` is spelled exactly as it is for a local `pipeline.cli run` — a
bare name is a workflow setting, a dotted one is that step's own param —
because the same thing meaning two things depending on where you type it
is how a batch ends up at the wrong settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .client import DEFAULT_URL, ApiError, B2CClient

# A finished run's status, in the order that reads best in a summary line.
_SYMBOL = {
    "done": "OK", "failed": "FAILED", "cancelled": "CANCELLED",
    "skipped": "skipped", "unknown": "?",
}


def _client(args: argparse.Namespace) -> B2CClient:
    # getattr: `--url`/`--token` are `SUPPRESS`-defaulted so the two
    # parsers carrying them cannot overwrite each other, which means the
    # attributes are absent unless the flag was given. The client then
    # falls back to $B2C_API_URL / $B2C_API_TOKEN.
    return B2CClient(url=getattr(args, "url", ""), token=getattr(args, "token", ""))


def _emit(value: Any) -> int:
    print(json.dumps(value, indent=2, sort_keys=False, default=str))
    return 0


def _duration(seconds: float) -> str:
    """`41.3s`, `6m12s`, `1h41m` — the scale the number is actually at.

    A pipeline stage runs from two seconds (a sheet split) to an hour (a
    30,000-iteration brush training), and one unit cannot show both without
    making one of them unreadable.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def _size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


# --------------------------------------------------------------------------
# the overarching one
# --------------------------------------------------------------------------

def _follow(client: B2CClient, name: str, interval: float) -> Dict[str, Any]:
    """Print a run's stages as they finish; return its final state.

    Each line is one stage that actually completed and what it cost, which
    makes the output a timing breakdown as well as a progress display —
    a slow run is usually one stage being slow, and this says which without
    reading the log.
    """
    final: Dict[str, Any] = {}
    for event in client.follow(name, interval=interval):
        kind = event["kind"]
        if kind == "status":
            print(f"  {event['status']}", flush=True)
        elif kind == "message":
            # The pod says things worth seeing here — most often that it is
            # waiting on tens of GB of checkpoints, which is otherwise
            # indistinguishable from a run that has hung before step one.
            print(f"  {event['text']}", flush=True)
        elif kind == "step":
            total = event["total"] or 0
            width = len(str(total)) if total else 2
            position = f"[{event['index']:>{width}}/{total}]"
            if event["status"] == "skipped":
                print(f"  {position} {event['step_id']:<28} skipped", flush=True)
            else:
                mark = "" if event["status"] == "done" else "  FAILED"
                print(
                    f"  {position} {event['step_id']:<28} "
                    f"{_duration(event['elapsed']):>8}{mark}",
                    flush=True,
                )
        elif kind == "end":
            final = event["state"]
    return final


def _report_end(client: B2CClient, state: Dict[str, Any]) -> None:
    """The last two lines: how it ended, and — if badly — why."""
    status = state.get("status", "unknown")
    started, finished = state.get("started") or 0, state.get("finished") or 0
    # Both, not just a positive difference: a run cancelled while still
    # queued, and one whose worker failed to spawn, are stamped `finished`
    # with `started` left at 0 — so the "difference" is the Unix epoch and
    # the summary claimed half a million hours of wall clock.
    wall = finished - started if started > 0 and finished > 0 else 0.0
    summary = _SYMBOL.get(status, status)
    print(f"\n{summary}: {state.get('message') or status}"
          + (f"  ({_duration(wall)} wall clock)" if wall > 0 else ""))
    if status == "failed":
        # The message is one line; the log says what led to it, and on a
        # pod it dies with the pod, so it is worth pulling now rather than
        # telling someone to go and look.
        print("\n--- last 40 log lines " + "-" * 40)
        try:
            print(client.log(state["name"], tail=40))
        except ApiError as exc:
            print(f"(could not read the log: {exc.detail})")
        except Exception as exc:
            # Losing the connection between the last poll and this fetch
            # must not replace the outcome that was just printed with a
            # transport error. The log is a convenience here; the summary
            # above it is the answer.
            print(f"(could not read the log: {exc})")


def _download(client: B2CClient, name: str, into: Path) -> int:
    reported = [0]

    def progress(written: int, total: int) -> None:
        # A multi-gigabyte download in silence is indistinguishable from a
        # stalled one. Deciles rather than a live bar: this often runs
        # non-interactively, into a log.
        if not total:
            return
        decile = int(written * 10 / total)
        if decile > reported[0]:
            reported[0] = decile
            print(f"  {decile * 10:>3}%  {_size(written)} of {_size(total)}", flush=True)

    try:
        path = client.download_result(name, into, on_progress=progress)
    except ApiError as exc:
        if exc.status == 404:
            print(f"no deliverables to download: {exc.detail}")
            return 1
        raise
    print(f"saved {path} ({_size(path.stat().st_size)})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Submit, watch every stage, then pull the result back."""
    from .cli import parse_param_overrides

    client = _client(args)
    settings, step_params = parse_param_overrides(args.param)

    submitted = client.submit(
        reference_image=None if args.remote else args.image,
        remote_path=args.image if args.remote else "",
        prompt=args.prompt or "",
        settings=settings, step_params=step_params, workflow=args.workflow or "",
    )
    names = [run["name"] for run in submitted]
    if len(names) > 1:
        # A zip fans out to one run per image across every GPU, so there is
        # no single thing to follow. Watching the first while the rest run
        # would be a lie about what is happening.
        print(f"queued {len(names)} runs:")
        for name in names:
            print(f"  {name}")
        print("\nThey are fanning out across the box's GPUs. Follow one with:\n"
              f"  python -m pipeline.cli api follow {names[0]}")
        return 0

    name = names[0]
    print(f"run {name}")
    final = _follow(client, name, args.interval)
    _report_end(client, final)
    if final.get("status") != "done" and not args.download_anyway:
        # A cancelled run may still have exported something, but a failed
        # one usually has not, and silently downloading nothing reads as a
        # successful collection.
        print("\nnot downloading (the run did not finish); "
              "pass --download-anyway to try regardless")
        return 1
    print()
    return _download(client, name, Path(args.output))


def cmd_follow(args: argparse.Namespace) -> int:
    client = _client(args)
    print(f"run {args.name}")
    final = _follow(client, args.name, args.interval)
    _report_end(client, final)
    return 0 if final.get("status") == "done" else 1


# --------------------------------------------------------------------------
# one per route
# --------------------------------------------------------------------------

def cmd_health(args: argparse.Namespace) -> int:
    return _emit(_client(args).health())


def cmd_workflows(args: argparse.Namespace) -> int:
    client = _client(args)
    if not args.name:
        return _emit(client.workflows())
    body = client.workflow(args.name)
    if args.json:
        return _emit(body)
    print(f"{body['name']}\n\nsettings:  (--param <name>=<value>)")
    for setting in body["settings"]:
        choices = f"  choices: {setting['choices']}" if setting["choices"] else ""
        flag = " (advanced)" if setting["advanced"] else ""
        print(f"  {setting['name']:<30} {setting['type']:<6} "
              f"default={setting['default']!r}{flag}{choices}")
    print("\noutputs:   (--param <name>=false to skip one)")
    for output in body["outputs"]:
        needs = f"  (needs {output['requires']})" if output["requires"] else ""
        print(f"  {output['name']:<30} -> {output['dir']}/  "
              f"default={output['default']!r}{needs}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    from .cli import parse_param_overrides

    settings, step_params = parse_param_overrides(args.param)
    submitted = _client(args).submit(
        reference_image=None if args.remote else args.image,
        remote_path=args.image if args.remote else "",
        prompt=args.prompt or "",
        settings=settings, step_params=step_params, workflow=args.workflow or "",
    )
    for run in submitted:
        print(run["name"])
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    runs = _client(args).runs()
    if args.json:
        return _emit(runs)
    if not runs:
        print("no runs on this volume")
        return 0
    for state in runs:
        progress = (f"{state['current']}/{state['total']}"
                    if state.get("total") else "")
        gpu = f"gpu{state['gpu_index']}" if state.get("gpu_index") is not None else ""
        print(f"{state['status']:<10} {state['name']:<46} {progress:>7} {gpu:<5} "
              f"{state.get('message', '')[:60]}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = _client(args).run(args.name)
    if args.json:
        return _emit(state)
    print(f"{state['name']}\n  status   {state['status']}")
    if state.get("total"):
        print(f"  step     {state['current']}/{state['total']}")
    for key in ("message", "workflow", "output_dir", "log_path"):
        if state.get(key):
            print(f"  {key:<8} {state[key]}")
    if state.get("gpu_index") is not None:
        print(f"  gpu      {state['gpu_index']}")
    for step in state.get("steps") or []:
        if step["status"] != "pending":
            print(f"    [{step['index']:>2}] {step['step_id']:<28} "
                  f"{step['status']:<8} {_duration(step['elapsed']):>8}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    print(_client(args).log(args.name, tail=args.tail))
    return 0


def cmd_result(args: argparse.Namespace) -> int:
    return _download(_client(args), args.name, Path(args.output))


def cmd_cancel(args: argparse.Namespace) -> int:
    state = _client(args).cancel(args.name)
    print(f"{state['name']}: {state['status']} — {state.get('message') or 'signalled'}")
    print("A running step finishes first; cancellation takes effect at the "
          "next step boundary.")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    return _emit(_client(args).schema())


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def add_parser(subparsers) -> argparse.ArgumentParser:
    """Attach `api` and its own subcommands to `pipeline.cli`'s parser."""
    # Where and as whom, on a parent parser rather than only on `api`
    # itself — `api runs --url ...` is the ordering people reach for, and
    # attaching them above the subparsers alone makes exactly that spelling
    # an "unrecognized arguments" error while every subcommand's help
    # advertises the flags.
    #
    # `SUPPRESS`, not `""`: a shared parser is applied twice, and a plain
    # default would have the subcommand's copy overwrite a value given
    # before it — `api --url http://pod runs` would silently fall back to
    # localhost. Suppressed, the attribute is set only where the flag was
    # actually typed.
    connection = argparse.ArgumentParser(add_help=False)
    connection.add_argument(
        "--url", default=argparse.SUPPRESS,
        help=f"Base URL of the server (default: ${{B2C_API_URL}}, else {DEFAULT_URL})",
    )
    connection.add_argument(
        "--token", default=argparse.SUPPRESS,
        help="Bearer token (default: $B2C_API_TOKEN, the same value the pod is given)",
    )

    api = subparsers.add_parser(
        "api", help="Drive a running pod over its HTTP API",
        description=__doc__, parents=[connection],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = api.add_subparsers(dest="api_command", required=True)

    def action(name: str, **kwargs) -> argparse.ArgumentParser:
        return actions.add_parser(name, parents=[connection], **kwargs)

    def submission_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "image",
            help="A reference sheet, or a .zip of image/prompt pairs (one run per pair)",
        )
        parser.add_argument("--prompt", default="", help="Subject description")
        parser.add_argument(
            "--remote", action="store_true",
            help="`image` is a path on the POD, not here — for a sheet already put "
                 "on the volume by rsync or runpodctl, which is the sane route for "
                 "anything large",
        )
        parser.add_argument(
            "--param", action="append", metavar="KEY=VALUE",
            help="A workflow setting (seed=7) or one step's own param "
                 "(train_final_splat.total_steps=100); repeatable. Same spelling as "
                 "`pipeline.cli run`. `api workflows <name>` lists the settings",
        )
        parser.add_argument("--workflow", default="", help="Workflow name (defaults to the server's)")

    run_p = action("run", help="Submit, follow every stage, then download the result")
    submission_args(run_p)
    run_p.add_argument("-o", "--output", default=".", help="Where to save the result .zip")
    run_p.add_argument("--interval", type=float, default=5.0, help="Seconds between polls")
    run_p.add_argument(
        "--download-anyway", action="store_true",
        help="Try the download even if the run failed or was cancelled — a run "
             "stopped after its COLMAP export still has one",
    )
    run_p.set_defaults(func=cmd_run)

    follow_p = action("follow", help="Watch a run that is already going")
    follow_p.add_argument("name")
    follow_p.add_argument("--interval", type=float, default=5.0)
    follow_p.set_defaults(func=cmd_follow)

    submit_p = action("submit", help="Queue a run and return its name")
    submission_args(submit_p)
    submit_p.set_defaults(func=cmd_submit)

    health_p = action("health", help="GPU slots, queue depth, model status")
    health_p.set_defaults(func=cmd_health)

    workflows_p = action(
        "workflows", help="What is available, or one workflow's settings and outputs",
    )
    workflows_p.add_argument("name", nargs="?", default="")
    workflows_p.add_argument("--json", action="store_true")
    workflows_p.set_defaults(func=cmd_workflows)

    runs_p = action("runs", help="Every run on the pod, in flight first")
    runs_p.add_argument("--json", action="store_true")
    runs_p.set_defaults(func=cmd_runs)

    status_p = action("status", help="One run's state and step list")
    status_p.add_argument("name")
    status_p.add_argument("--json", action="store_true")
    status_p.set_defaults(func=cmd_status)

    log_p = action("log", help="Tail a run's log")
    log_p.add_argument("name")
    log_p.add_argument("--tail", type=int, default=200)
    log_p.set_defaults(func=cmd_log)

    result_p = action("result", help="Download a finished run's .zip")
    result_p.add_argument("name")
    result_p.add_argument("-o", "--output", default=".")
    result_p.set_defaults(func=cmd_result)

    cancel_p = action("cancel", help="Stop a run at its next step boundary")
    cancel_p.add_argument("name")
    cancel_p.set_defaults(func=cmd_cancel)

    schema_p = action("schema", help="The generated OpenAPI document")
    schema_p.set_defaults(func=cmd_schema)

    return api


def dispatch(args: argparse.Namespace) -> int:
    """Run one `api` subcommand, turning a refusal into a message.

    Every `detail` the server sends is a sentence written to be read, so a
    traceback here would replace something useful with something noisy.
    """
    try:
        return args.func(args)
    except ApiError as exc:
        print(f"error {exc.status}: {exc.detail}", file=sys.stderr)
        if exc.status == 401:
            print("Set --token, or $B2C_API_TOKEN to the value the pod was given.",
                  file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Watching is not driving: the run is a process on the pod and
        # nothing here is holding it up.
        print("\nstopped watching; the run is still going on the pod", file=sys.stderr)
        return 130
    except Exception as exc:  # a connection that never opened, mostly
        import requests

        if isinstance(exc, requests.RequestException):
            print(f"could not reach the server: {exc}", file=sys.stderr)
            # Losing the connection does not stop the run, and `api run`
            # gets this far only after `follow` has already retried for
            # minutes — so say what to do rather than leave it looking like
            # the work was lost.
            name = getattr(args, "name", "") or "<run>"
            print(f"The run is unaffected. Reattach with `api follow {name}`, "
                  f"or collect it later with `api result {name}`.", file=sys.stderr)
            return 1
        raise
