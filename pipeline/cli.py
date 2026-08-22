"""CLI entrypoint: `python -m pipeline.cli run <workflow.yaml> --dataset <dir>`

Loads a dataset from disk into the initial Context (as `dataset`), runs the
workflow, and — if the workflow didn't already checkpoint one itself — writes
the final in-memory dataset to `--out` so a run always leaves something on
disk to inspect.
"""

from __future__ import annotations

import argparse
import logging

from . import steps  # noqa: F401  registers all Step subclasses
from .dataset import Dataset
from .runner import WorkflowRunner
from .workflow import WorkflowSpec, load_envs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("workflow", help="Path to a workflow YAML")
    parser.add_argument("--dataset", required=True, help="Path to an on-disk dataset directory to load")
    parser.add_argument("--envs", default="pipeline/envs/envs.yaml", help="Path to the envs registry YAML")
    parser.add_argument("--out", default=None, help="Directory to save the final dataset to")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    spec = WorkflowSpec.from_yaml(args.workflow)
    envs = load_envs(args.envs)
    dataset = Dataset.from_disk(args.dataset)

    runner = WorkflowRunner(spec, envs=envs)
    ctx = runner.run({"dataset": dataset})

    if args.out:
        final_dataset: Dataset = ctx.get("dataset")
        path = final_dataset.to_disk(args.out)
        print(f"Saved final dataset to {path}")


if __name__ == "__main__":
    main()
