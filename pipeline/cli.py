"""CLI entrypoint for the pipeline.

    python -m pipeline.cli run <workflow> --dataset <dir>
    python -m pipeline.cli run <workflow> --reference-image <photo.jpg>
    python -m pipeline.cli doctor
    python -m pipeline.cli steps | workflows
    python -m pipeline.cli ui

`run` loads a starting Dataset — either a complete on-disk one, or a
bootstrap carrying only a reference photo (see Dataset.from_reference_image)
— runs the workflow, and writes the final in-memory dataset to `--out` so a
run always leaves something on disk to inspect.

`doctor` is the one to reach for on a fresh pod: it answers whether Vulkan,
EGL, the venvs, the brush binaries and the HF token are all actually usable,
which is otherwise only discoverable by starting a 40-minute run and
watching where it dies.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import steps  # noqa: F401  registers all Step subclasses
from .dataset import Dataset
from .logging_setup import setup_logging, timestamped_run_name
from .paths import REPO_ROOT, configure_tmpdir, output_dir
from .runner import WorkflowRunner
from .workflow import WorkflowSpec, load_envs

logger = logging.getLogger("pipeline.cli")

WORKFLOW_DIR = REPO_ROOT / "pipeline" / "workflows"
DEFAULT_ENVS = str(REPO_ROOT / "pipeline" / "envs" / "envs.yaml")


def resolve_workflow(name: str) -> Path:
    """Accept a path, a bare name, or a name without the .yaml suffix."""
    candidate = Path(name)
    if candidate.exists():
        return candidate
    for suffix in ("", ".yaml", ".yml"):
        candidate = WORKFLOW_DIR / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    available = ", ".join(sorted(p.stem for p in WORKFLOW_DIR.glob("*.yaml")))
    raise SystemExit(f"No such workflow: {name!r}. Available: {available}")


def available_workflows() -> List[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yaml"))


def parse_param_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """`--param seed=7 --param resolution=[720,1280]` -> a dict.

    Values go through yaml.safe_load so numbers, lists and booleans arrive
    as themselves rather than as strings — a workflow that does
    `width: ${params.resolution.0}` needs a real list, not "[720,1280]".
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            overrides[key.strip()] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise SystemExit(f"--param {key}: could not parse value {raw!r}: {exc}") from None
    return overrides


def run_workflow(args: argparse.Namespace) -> int:
    from .doctor import log_machine_banner

    workflow_path = resolve_workflow(args.workflow)
    spec = WorkflowSpec.from_yaml(workflow_path)

    run_name = args.run_name or timestamped_run_name(spec.name)
    log_path = setup_logging(verbose=not args.quiet, log_file=args.log_file, run_name=run_name)
    tmp = configure_tmpdir()

    logger.info("run '%s'", run_name)
    logger.info("workflow: %s", workflow_path)
    if log_path:
        logger.info("log file: %s", log_path)
    logger.info("temp dir: %s", tmp)
    log_machine_banner()

    overrides = parse_param_overrides(args.param)
    if overrides:
        unknown = sorted(set(overrides) - set(spec.params))
        if unknown:
            # A typo'd override that silently does nothing is worse than a
            # refusal: the run completes, at the wrong settings, and looks
            # like it honoured you.
            raise SystemExit(
                f"--param names not declared by workflow '{spec.name}': {', '.join(unknown)}. "
                f"It declares: {', '.join(sorted(spec.params))}"
            )
        for key, value in overrides.items():
            logger.info("param override: %s = %r (was %r)", key, value, spec.params[key])
        spec.params.update(overrides)

    envs = load_envs(args.envs)
    logger.info("envs registry: %s (%s)", args.envs, ", ".join(sorted(envs)) or "empty")

    # Block on the models this workflow actually needs, before touching the
    # GPU. Scoped to the workflow rather than to everything: a workflow that
    # skips both denoise passes must not wait on the ~47 GB of Wan2.2
    # weights it never touches.
    if not args.no_wait_for_models:
        from .models import ModelsUnavailable, required_for_steps, wait_until_ready

        # enabled_steps(), not steps: a run with export_ply switched off
        # must not block on a checkpoint only the skipped step needs.
        needed = required_for_steps(step.step for step in spec.enabled_steps())
        if needed:
            logger.info("models this workflow needs: %s", ", ".join(needed))
            try:
                wait_until_ready(needed)
            except ModelsUnavailable as exc:
                logger.error("%s", exc)
                return 1
            logger.info("all required models present")

    if args.reference_image:
        dataset = Dataset.from_reference_image(args.reference_image, prompt=args.prompt)
        logger.info(
            "starting from a reference photo: %s (%dx%d)",
            args.reference_image, *dataset.resolution,
        )
    else:
        dataset = Dataset.from_disk(args.dataset)
        logger.info(
            "loaded dataset %s: %d frames, %d cameras, %d points",
            args.dataset, len(dataset.images), len(dataset.cameras), len(dataset.points_3d[0]),
        )
        if args.prompt:
            dataset.prompt = args.prompt

    out = Path(args.out) if args.out else output_dir() / run_name

    # Repoint the workflow's own disk writes (brush's .ply, the COLMAP
    # export, any save_dataset checkpoint) into this run's output directory.
    # The literal in the YAML is a relative path, which on a pod resolves
    # into the container's writable layer rather than the mounted volume —
    # a few GB of splat written somewhere small and impermanent.
    if "output_root" in spec.params and "output_root" not in overrides:
        spec.params["output_root"] = str(out)
        logger.info("output_root: %s", out)

    runner = WorkflowRunner(spec, envs=envs)

    try:
        ctx = runner.run({"dataset": dataset})
    except Exception:
        logger.exception("run '%s' failed", run_name)
        if log_path:
            logger.error("full log: %s", log_path)
        return 1

    final: Dataset = ctx.get("dataset")
    saved = final.to_disk(out)
    logger.info("saved final dataset to %s (%d frames)", saved, len(final.images))
    if final.splat_path:
        logger.info("splat: %s", final.splat_path)
    if log_path:
        logger.info("log file: %s", log_path)
    return 0


def run_prefetch(args: argparse.Namespace) -> int:
    from .models import FAILED, prefetch, read_status, registry

    setup_logging(verbose=True, log_file=args.log_file, run_name="prefetch")
    keys = [k.strip() for k in args.only.split(",")] if args.only else None

    if args.status:
        known = registry()
        for key, entry in sorted(read_status().items()):
            print(f"{entry['status']:<9} {key:<12} {known[key].approx_gb:>5.1f} GB  {entry.get('detail', '')}")
        return 0

    status = prefetch(keys, force=args.force)
    failures = [k for k, v in status.items() if v["status"] == FAILED]
    if failures:
        logger.error("could not download: %s", ", ".join(failures))
    return 1 if failures else 0


def run_doctor(args: argparse.Namespace) -> int:
    from .doctor import FAIL, format_report, run_checks, worst_status

    setup_logging(verbose=True, log_file=args.log_file, run_name="doctor")
    envs = load_envs(args.envs)
    checks = run_checks(envs)
    print(format_report(checks, verbose=not args.summary))
    return 1 if worst_status(checks) == FAIL else 0


def list_steps(args: argparse.Namespace) -> int:
    from .registry import STEP_REGISTRY

    for name in sorted(STEP_REGISTRY):
        doc = (STEP_REGISTRY[name].__doc__ or "").strip().splitlines()
        print(f"{name:<24} {doc[0] if doc else ''}")
    return 0


def list_workflows(args: argparse.Namespace) -> int:
    for path in available_workflows():
        spec = WorkflowSpec.from_yaml(path)
        print(f"{path.stem:<28} {len(spec.steps):>2} steps  {spec.description.strip().splitlines()[0] if spec.description else ''}")
    return 0


def launch_ui(args: argparse.Namespace) -> int:
    from .webui import launch

    setup_logging(verbose=True, log_file=args.log_file, run_name="webui")
    configure_tmpdir()
    launch(host=args.host, port=args.port, envs_path=args.envs, share=args.share)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--envs", default=DEFAULT_ENVS, help="Path to the envs registry YAML")
        p.add_argument(
            "--log-file", default=None,
            help="Write the run's log here (default: a timestamped file under B2C_LOG_DIR; "
                 "pass an empty string to disable file logging)",
        )

    run_p = sub.add_parser("run", help="Run a workflow")
    run_p.add_argument("workflow", help="Workflow YAML path, or a name from pipeline/workflows/")
    source = run_p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="An existing on-disk b2c dataset directory")
    source.add_argument(
        "--reference-image",
        help="A single photo to start from, for a workflow that renders its own "
             "views (sam3d_body -> render). No shipped workflow does — both "
             "fast_helical files begin from an existing dataset — so this needs "
             "a workflow of your own; see Dataset.from_reference_image",
    )
    run_p.add_argument("--prompt", default=None, help="Subject description used by the denoise steps")
    run_p.add_argument("--out", default=None, help="Directory to save the final dataset to")
    run_p.add_argument("--run-name", default=None, help="Names the log file and default --out")
    run_p.add_argument(
        "--param", action="append", metavar="KEY=VALUE",
        help="Override one of the workflow's params; repeatable",
    )
    run_p.add_argument(
        "--no-wait-for-models", action="store_true",
        help="Start immediately instead of waiting for the workflow's checkpoints; "
             "each step then downloads its own on first use, mid-run",
    )
    run_p.add_argument("-q", "--quiet", action="store_true", help="Warnings only on the console")
    run_p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Accepted for compatibility; INFO is the default now",
    )
    add_common(run_p)
    run_p.set_defaults(func=run_workflow)

    doctor_p = sub.add_parser("doctor", help="Check that this machine can run the pipeline")
    doctor_p.add_argument("--summary", action="store_true", help="One line per check")
    add_common(doctor_p)
    doctor_p.set_defaults(func=run_doctor)

    prefetch_p = sub.add_parser(
        "prefetch", help="Download every model checkpoint up front"
    )
    prefetch_p.add_argument(
        "--only", default=None, metavar="KEYS",
        help="Comma-separated model keys instead of all of them "
             "(rmbg, sapiens2, sam3dbody, wan22, wan22_lora, seedvr2, mediapipe)",
    )
    prefetch_p.add_argument(
        "--force", action="store_true",
        help="Re-verify against the network even for models already marked present",
    )
    prefetch_p.add_argument(
        "--status", action="store_true", help="Report what is present and what is not, then exit"
    )
    add_common(prefetch_p)
    prefetch_p.set_defaults(func=run_prefetch)

    steps_p = sub.add_parser("steps", help="List registered steps")
    steps_p.set_defaults(func=list_steps)

    workflows_p = sub.add_parser("workflows", help="List available workflows")
    workflows_p.set_defaults(func=list_workflows)

    ui_p = sub.add_parser("ui", help="Launch the web UI")
    ui_p.add_argument("--host", default="0.0.0.0")
    ui_p.add_argument("--port", type=int, default=7860)
    ui_p.add_argument("--share", action="store_true", help="Request a public gradio.live tunnel")
    add_common(ui_p)
    ui_p.set_defaults(func=launch_ui)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
