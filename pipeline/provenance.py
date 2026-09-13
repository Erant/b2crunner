"""Which commit of each moving part a run actually ran — one line in every log.

Three repositories change the pixels: this one, body2colmap (the renderer
and the dataset model, a pip install from git) and b2ctrain (the trainer
and rasteriser, a binary). Comparing two result archives starts with
"what code was this", and until 2026-09-13 none of the three was written
anywhere a log could be read back from — the image carries its b2crunner
sha as an OCI label, which `docker inspect` sees and a container does not,
and the other two were pins in the Dockerfile at whatever commit the
Dockerfile was at. `versions()` asks each part directly:

- **b2crunner**: `git rev-parse` in the checkout when there is one (a dev
  machine), else `B2C_GIT_REVISION`, which docker/Dockerfile bakes from
  the same `--build-arg GIT_SHA`/`GIT_DIRTY` the label is made of — the
  image is built from a `git archive`, so there is no .git to ask.
- **body2colmap**: pip's own record of what it installed. A `git+https://`
  install writes `direct_url.json` beside the dist-info (PEP 610) with the
  resolved `commit_id`; an editable install from a checkout names the
  directory, which is then asked with git.
- **b2ctrain**: `b2ctrain --version`, which names its commit since
  b49480f (`b2ctrain 0.1.0 (<sha>)`). An older binary prints the bare
  version, and then `B2CTRAIN_REF` — the pin the image was built at, baked
  as an ENV — stands in, marked as such.

Everything here is best-effort and cheap (one `--version`, two `git`
calls at most, no network): a missing piece reads `unknown` rather than
failing a run over its own bookkeeping.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The short form the log uses; git's own default is 7 and grows with the repo.
SHORT = 12


def _git(path: Path, *args: str) -> Optional[str]:
    """`git -C path args`, stripped, or None when git or the checkout is missing."""
    git = shutil.which("git")
    if not git:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(path), *args], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_revision(path: Path) -> Optional[str]:
    """`<sha>[-dirty]` of the checkout at `path`, or None when it is not one."""
    if not (path / ".git").exists():
        return None
    sha = _git(path, "rev-parse", f"--short={SHORT}", "HEAD")
    if not sha:
        return None
    dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
    return sha + ("-dirty" if dirty else "")


def b2crunner_revision() -> str:
    return git_revision(REPO_ROOT) or os.environ.get("B2C_GIT_REVISION") or "unknown"


def body2colmap_revision() -> str:
    """The installed body2colmap's commit, from pip's PEP 610 record."""
    try:
        from importlib import metadata
        dist = metadata.distribution("body2colmap")
    except Exception:  # noqa: BLE001 — not installed, or no metadata at all
        return "unknown (not installed)"
    version = dist.version
    raw = dist.read_text("direct_url.json")
    if not raw:
        return f"unknown (pypi {version})"
    try:
        info = json.loads(raw)
    except ValueError:
        return f"unknown (pypi {version})"
    vcs = info.get("vcs_info") or {}
    commit = vcs.get("commit_id")
    if commit:
        return commit[:SHORT]
    if (info.get("dir_info") or {}).get("editable") or str(info.get("url", "")).startswith("file:"):
        local = Path(str(info.get("url", "")).replace("file://", "", 1))
        rev = git_revision(local) if local.exists() else None
        return rev or f"unknown (editable {local})"
    return f"unknown ({version})"


_VERSION_SHA = re.compile(r"\(([0-9a-f]{7,40}(?:-dirty)?)\)")


def b2ctrain_revision(binary: str = "b2ctrain") -> str:
    """What `b2ctrain --version` says, or the image's pin when it says nothing."""
    path = shutil.which(binary)
    if not path:
        return "unknown (not on PATH)"
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown (--version failed)"
    text = (result.stdout + result.stderr).strip()
    match = _VERSION_SHA.search(text)
    if match:
        return match.group(1)
    pinned = os.environ.get("B2CTRAIN_REF")
    if pinned:
        return f"{pinned[:SHORT]} (the image's pin; the binary reports only '{text}')"
    return f"unknown ('{text}')"


def versions() -> Dict[str, str]:
    """{"b2crunner": ..., "body2colmap": ..., "b2ctrain": ...}, each a sha or `unknown (...)`."""
    return {
        "b2crunner": b2crunner_revision(),
        "body2colmap": body2colmap_revision(),
        "b2ctrain": b2ctrain_revision(),
    }


def describe(found: Optional[Dict[str, str]] = None) -> str:
    """One line: `b2crunner a81d396f1c2d, body2colmap 339a5983d9c3, b2ctrain b49480f8739c`."""
    found = versions() if found is None else found
    return ", ".join(f"{name} {rev}" for name, rev in found.items())
