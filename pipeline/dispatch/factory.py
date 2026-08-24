"""Builds a Dispatcher from a step's `dispatch:`/`env:` YAML fields.

`env:` names an entry in an environments config (see pipeline/envs/envs.yaml)
that supplies the mechanism-specific details (python_bin / image / base_url).
Keeping that indirection means a workflow YAML says *what kind of isolation*
a step needs conceptually ("subprocess", pointing at env "sam3dbody"), while
*where that env actually lives on this machine* is a separate, per-machine
concern that doesn't belong baked into the workflow file.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import Dispatcher
from .docker import DockerDispatcher
from .in_process import InProcessDispatcher
from .service import ServiceDispatcher
from .subprocess_python import SubprocessPythonDispatcher


def build_dispatcher(
    dispatch: str,
    env_config: Optional[Dict[str, Any]] = None,
    keep_loaded: bool = False,
) -> Dispatcher:
    env_config = env_config or {}

    if dispatch == "in_process":
        return InProcessDispatcher(keep_loaded=keep_loaded)

    if dispatch == "subprocess":
        if "python_bin" not in env_config:
            raise ValueError("dispatch: subprocess requires env config with 'python_bin'")
        # keep_loaded is forwarded here too, not only to InProcessDispatcher:
        # it used to be silently dropped on this branch, which made
        # `keep_loaded: true` on a subprocess step look like it worked and do
        # nothing. That is the branch where it matters most — an in-process
        # step reloading is a Python object; a subprocess step reloading is
        # ~47 GB off a network volume (see subprocess_python.py).
        return SubprocessPythonDispatcher(
            python_bin=env_config["python_bin"],
            cwd=env_config.get("cwd"),
            env=env_config.get("env"),
            keep_loaded=keep_loaded,
        )

    if dispatch == "service":
        if "base_url" not in env_config:
            raise ValueError("dispatch: service requires env config with 'base_url'")
        return ServiceDispatcher(base_url=env_config["base_url"], timeout=env_config.get("timeout", 3600.0))

    if dispatch == "docker":
        if "image" not in env_config:
            raise ValueError("dispatch: docker requires env config with 'image'")
        return DockerDispatcher(
            image=env_config["image"],
            gpus=env_config.get("gpus", "all"),
            extra_args=env_config.get("extra_args"),
        )

    raise ValueError(f"Unknown dispatch kind '{dispatch}' (expected in_process/subprocess/service/docker)")
