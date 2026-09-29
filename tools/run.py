#!/usr/bin/env python3
"""Run a b2crunner tool in the environment it needs (docs/tools.md).

    python3 tools/run.py <tool> [args...]
    python3 tools/run.py --list

The tools run pipeline steps and models outside a workflow, for callers such as b2crig that hand b2crunner files and
get files back. Each tool needs one of the per-model environments (pipeline/envs/envs.yaml); this launcher picks the
interpreter so callers need not know the venv layout. Standard library only: any python3 can start it.

Per host, an environment's interpreter can be overridden with B2CRUNNER_PYTHON_<ENV> (e.g. B2CRUNNER_PYTHON_SAM3DBODY)
and extra import paths prepended with B2CRUNNER_PATH_<ENV> (os.pathsep-separated), e.g. for a SAM-3D-Body source
checkout that is not installed into its venv.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent

#: tool -> the environment (pipeline/envs/envs.yaml key) it runs in
TOOLS = {
    "wan_clip": "wan22",
    "seg_clip": "wan22",
    "pointmap_clip": "wan22",
    "face_reference": "wan22",
    "upscale_clip": "seedvr2",
    "video_fit": "sam3dbody",
    "export_mhr_subject": "sam3dbody",
    "recover_body": "sam3dbody",
    "export_glb": "sam3dbody",
}


def env_pythons() -> dict:
    """envs.yaml's python_bin per environment (the file is simple enough to read without PyYAML)."""
    out, name = {}, None
    for line in (ROOT / "pipeline" / "envs" / "envs.yaml").read_text().splitlines():
        m = re.match(r"^  (\w+):\s*$", line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r"^    python_bin:\s*(\S+)", line)
        if m and name:
            out[name] = m.group(1)
    return out


def python_for(env: str) -> Path:
    override = os.environ.get(f"B2CRUNNER_PYTHON_{env.upper()}")
    if override:
        return Path(override).expanduser()
    rel = env_pythons().get(env)
    if rel is None:
        raise SystemExit(f"run.py: environment {env!r} is not in pipeline/envs/envs.yaml")
    return Path(rel) if Path(rel).is_absolute() else ROOT / rel


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip()); print("\ntools: " + ", ".join(sorted(TOOLS)))
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    if sys.argv[1] == "--list":
        for tool, env in sorted(TOOLS.items()):
            py = python_for(env)
            print(f"{tool:20s} {env:10s} {py}{'' if py.exists() else '   (missing: build the env or set B2CRUNNER_PYTHON_' + env.upper() + ')'}")
        return
    tool, args = sys.argv[1], sys.argv[2:]
    if tool not in TOOLS:
        raise SystemExit(f"run.py: unknown tool {tool!r}; tools: {', '.join(sorted(TOOLS))}")
    env = TOOLS[tool]
    py = python_for(env)
    if not py.exists():
        raise SystemExit(f"run.py: {tool} needs the {env} environment, whose interpreter {py} does not exist; build it "
                         f"(pipeline/envs/{env}) or set B2CRUNNER_PYTHON_{env.upper()}")
    extra = [p for p in os.environ.get(f"B2CRUNNER_PATH_{env.upper()}", "").split(os.pathsep) if p]
    environ = dict(os.environ)
    environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), *(str(Path(p).expanduser()) for p in extra),
                                             *([environ["PYTHONPATH"]] if environ.get("PYTHONPATH") else [])])
    os.execve(str(py), [str(py), str(TOOLS_DIR / f"{tool}.py"), *args], environ)


if __name__ == "__main__":
    main()
