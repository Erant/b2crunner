"""The only two steps that are allowed to touch disk for dataset state.

Everything else in a workflow reads/writes the in-memory Dataset directly via
the Context. A workflow author drops a `save_dataset` step in explicitly when
they want a checkpoint on disk (e.g. to inspect intermediate frames, or to
resume a long pipeline without redoing earlier stages).
"""

from __future__ import annotations

from typing import Any, Dict

from ..dataset import Dataset
from ..registry import register_step
from ..step import REQUIRED, Param, Step


@register_step("save_dataset")
class SaveDatasetStep(Step):
    PARAMS = (
        Param("directory", str, REQUIRED,
              "Where to write the checkpoint. Usually ${globals.output_root}/something."),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        dataset: Dataset = inputs["dataset"]
        directory = params["directory"]
        path = dataset.to_disk(directory)
        return {"directory_path": str(path)}


@register_step("load_dataset")
class LoadDatasetStep(Step):
    PARAMS = (
        Param("directory", str, REQUIRED, "The on-disk dataset directory to read."),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        directory = params["directory"]
        return {"dataset": Dataset.from_disk(directory)}
