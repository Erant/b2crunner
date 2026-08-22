"""Workflow YAML schema and loader.

Example (see pipeline/workflows/example_helical.yaml for a full one):

    name: fast_helical
    params:
      resolution: [512, 512]
      diffusion_steps: 6
    steps:
      - id: denoise
        step: wan22_vace_denoise      # registered Step name
        dispatch: subprocess          # in_process | subprocess | service | docker
        env: wan22                    # key into envs.yaml, ignored for in_process
        inputs:
          control_video: dataset.images
          reference_image: dataset.reference_image
        params:
          steps: ${params.diffusion_steps}
          width: ${params.resolution.0}
        outputs:
          denoised: dataset.images    # written back into the shared Context

Only a step whose Step subclass actually writes to disk (e.g. `save_dataset`)
touches disk — everything else stays in the in-memory Context between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StepSpec:
    id: str
    step: str
    dispatch: str = "in_process"
    env: Optional[str] = None
    keep_loaded: bool = False
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepSpec":
        return cls(
            id=data["id"],
            step=data["step"],
            dispatch=data.get("dispatch", "in_process"),
            env=data.get("env"),
            keep_loaded=data.get("keep_loaded", False),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            params=data.get("params", {}),
        )


@dataclass
class WorkflowSpec:
    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    steps: List[StepSpec] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowSpec":
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            params=data.get("params", {}),
            steps=[StepSpec.from_dict(s) for s in data["steps"]],
        )


def load_envs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Loads the per-machine environment registry (python_bin/image/base_url
    per env name). Returns {} if the file doesn't exist yet — fine for
    workflows that only use in_process steps."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data.get("envs", {})
