"""b2cviewer (Erant/b2cviewer) served beside the web UI, at `/viewer/`.

The viewer is a static WebGL2 page plus two routes: `api/index` (the subject
files it can open) and `data/<path>` (one .glb). Its own `serve.py` answers
those over b2crig's layout (`<subject>/gltf/`); here they answer over this
volume's runs instead, one subject per run's `ply/` directory, named after the
run, so `viewer/?subject=<run>` opens that run. The index itself — which .glb
is a subject file, which clips belong to it — stays b2cviewer's
(`serve.index`), loaded from the checkout, so both hosts list files the same
way.

Where the checkout is: `B2C_VIEWER_DIR` (the image's is /opt/b2cviewer). With
none there nothing is mounted and the Results tab shows no viewer button.

NOT GUARDED: these routes sit on the FastAPI app beside the Gradio mount, so
the UI's login does not cover them — anyone who can reach the server can list
and download every run's subject file. Fine on a dev box; on a public pod it
waits for the server-wide auth rework.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Tuple
from urllib.parse import quote

from .paths import output_dir
from .run_state import RunState
from .runs import SPLAT_DIR

logger = logging.getLogger(__name__)

VIEWER_ENV = "B2C_VIEWER_DIR"
DEFAULT_VIEWER_DIR = "/opt/b2cviewer"
VIEWER_PATH = "/viewer"


def viewer_dir() -> Optional[Path]:
    """The b2cviewer checkout (serve.py + web/), or None when there is none."""
    root = Path(os.environ.get(VIEWER_ENV) or DEFAULT_VIEWER_DIR)
    if (root / "serve.py").is_file() and (root / "web" / "index.html").is_file():
        return root
    return None


def _serve_module(root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("b2cviewer_serve", root / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subject_dirs(runs: Iterable[RunState], root: Path) -> List[Tuple[str, Path]]:
    """(run name, its `ply/`) for every run whose `ply/` is under `root` — the only files `data/` serves."""
    dirs = []
    for state in runs:
        if not state.name or not state.output_dir:
            continue
        directory = Path(state.output_dir) / SPLAT_DIR
        if directory.is_dir() and directory.resolve().is_relative_to(root):
            dirs.append((state.name, directory))
    return dirs


def has_subject_file(run_dir: Optional[Path]) -> bool:
    """The run delivered a subject .glb the viewer can open (a clip file alone is not one)."""
    directory = Path(run_dir) / SPLAT_DIR if run_dir else None
    return bool(directory and directory.is_dir()
                and any(not f.name.endswith(".clip.glb") for f in directory.glob("*.glb")))


def viewer_link(run_name: str) -> str:
    """The viewer on this run, relative to the UI's root (the UI is mounted at `/`)."""
    return f"{VIEWER_PATH.lstrip('/')}/?subject={quote(run_name, safe='')}"


def mount_viewer(server, runs: Callable[[], Iterable[RunState]]) -> bool:
    """Add `/viewer/` to `server`, listing `runs()`. Call before the Gradio mount at `/`, which matches every path.

    False (and nothing added) without a b2cviewer checkout."""
    import fastapi
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    root = viewer_dir()
    if root is None:
        logger.info("no b2cviewer checkout (%s=%s): the 3D viewer is not served",
                    VIEWER_ENV, os.environ.get(VIEWER_ENV) or DEFAULT_VIEWER_DIR)
        return False
    serve = _serve_module(root)
    no_cache = {"Cache-Control": "no-cache"}

    @server.get(VIEWER_PATH, include_in_schema=False)
    def viewer_root():
        return RedirectResponse(f"{VIEWER_PATH}/")

    @server.get(f"{VIEWER_PATH}/api/index", include_in_schema=False)
    def viewer_index():
        base = output_dir().resolve()
        return JSONResponse(serve.index(base, subject_dirs(runs(), base), where=f"<run>/{SPLAT_DIR}/*.glb"),
                            headers=no_cache)

    @server.get(f"{VIEWER_PATH}/data/{{rel:path}}", include_in_schema=False)
    def viewer_data(rel: str):
        path = serve.data_file(output_dir().resolve(), rel)
        if path is None:
            raise fastapi.HTTPException(status_code=404)
        return FileResponse(path, media_type="model/gltf-binary", headers=no_cache)

    server.mount(VIEWER_PATH, StaticFiles(directory=root / "web", html=True), name="b2cviewer")
    logger.info("3D viewer (b2cviewer %s) at %s/ — NOT behind the UI login", root, VIEWER_PATH)
    return True
