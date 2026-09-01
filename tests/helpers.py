"""Shared test helpers.

The golden-data tests below run against real completed runs of the original
ComfyUI pipeline (stage directories initial/ -> circular/ -> splatted/ ->
masked_splatted/ -> helical/ -> upscaled/ -> colmap/). These are gitignored
local reference data, so every test that needs one skips cleanly when it
isn't present rather than failing.

**There are two runs, and which one answers a question matters.**

  * `cyber_6f/` (repo root) is the older one, recorded before anchor
    injection was wired into the ComfyUI graphs. Its `masked_splatted` has
    a uniform alpha of 255 on every frame and no injected anchor anywhere
    in the helical orbit. Fine for the mask/composite arithmetic, and
    actively misleading about the anchor.
  * `cyber2_6f/` (~/Documents by default, `B2C_CYBER2_6F` to override) is
    the newer one, with anchor injection live. Its `splatted` records
    `anchor_frame_index: 37`, and `masked_splatted/frame_00038_.png` is
    byte-identical to that stage's `anchor.png` at a uniform alpha of 0
    while all 80 other frames sit at 255. That is the run to check anything
    involving the anchor or the VACE mask convention against.

The alpha channel in these PNGs is the VACE mask, not a foreground
silhouette: 255 = "synthetic, denoise this frame", 0 = "a real photograph,
keep it". `splatted` is the one exception, and only because that stage is
never fed to VACE — it parks the splat render's per-pixel alpha in the same
channel for `mask_splat` to threshold.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CYBER_6F = REPO_ROOT / "cyber_6f"
CYBER2_6F = Path(
    os.environ.get("B2C_CYBER2_6F", Path.home() / "Documents" / "cyber2_6f")
).expanduser()


def require_stage(*names: str) -> Path | tuple[Path, ...]:
    """Skip the calling test unless every named cyber_6f stage dir exists."""
    paths = []
    for name in names:
        p = CYBER_6F / name
        if not (p / "metadata.json").exists() and not p.is_dir():
            raise unittest.SkipTest(f"reference data missing: {p}")
        if not p.is_dir():
            raise unittest.SkipTest(f"reference data missing: {p}")
        paths.append(p)
    return paths[0] if len(paths) == 1 else tuple(paths)


def require_stage2(*names: str) -> Path | tuple[Path, ...]:
    """Skip the calling test unless every named cyber2_6f stage dir exists.

    The newer recorded run — see the module docstring for why the anchor
    and VACE-mask questions have to be asked of this one and not cyber_6f.
    """
    paths = []
    for name in names:
        p = CYBER2_6F / name
        if not p.is_dir():
            raise unittest.SkipTest(f"reference data missing: {p}")
        paths.append(p)
    return paths[0] if len(paths) == 1 else tuple(paths)


def run_step(name: str, inputs: Dict[str, Any], params: Optional[Dict[str, Any]] = None):
    """Build a registered step and run it the way the runner would.

    The `resolve_params` call is the part that matters: a Step's `run()`
    reads `params["x"]` and relies on the caller having merged in the
    defaults its class declares (pipeline/step.py). WorkflowRunner does
    that before dispatch; a test calling a step directly has to do the
    same, or it is exercising a code path the pipeline never takes.
    """
    from pipeline.registry import get_step_class

    step_class = get_step_class(name)
    return step_class().run(inputs, step_class.resolve_params(params or {}))


def redirect_crash_dir(case: unittest.TestCase) -> Path:
    """Point `paths.crash_dir()` at a temp directory for the duration.

    Both external binaries save diagnostics on a failed exit, and testing
    that means running failures on purpose. Without this the suite writes
    crash directories into the developer's real volume (or, with no volume,
    into the repo's own `output/_local_data`) and leaves them there.

    `paths` resolves B2C_LOG_DIR on every call, so the environment variable
    is all it takes. Returns the `crashes/` directory to look in.
    """
    import tempfile

    tmp = tempfile.TemporaryDirectory(prefix="b2c_crash_test_")
    case.addCleanup(tmp.cleanup)
    logs = Path(tmp.name) / "logs"

    previous = os.environ.get("B2C_LOG_DIR")
    os.environ["B2C_LOG_DIR"] = str(logs)

    def restore():
        if previous is None:
            os.environ.pop("B2C_LOG_DIR", None)
        else:
            os.environ["B2C_LOG_DIR"] = previous

    case.addCleanup(restore)
    return logs / "crashes"


def crash_dirs(crashes: Path) -> list:
    """Every crash directory saved so far, oldest first."""
    return sorted(crashes.iterdir()) if crashes.exists() else []


def stub_render_binary(
    directory,
    *,
    frames: str = "all",
    damage: str = "",
    segfault: bool = False,
    alpha: int = 255,
    record=None,
):
    """An executable stand-in for `brush-splat-render`, as a path.

    The rasterisation is body2colmap's since 2026-08-31 — `_rasterize`
    drives its `SplatRenderer` rather than shelling out itself — so the
    seam this project can still observe is the binary, not an internal
    function. That is also the better place to watch from: what reaches the
    argv and the cameras.json is what the real renderer would see.

    Args:
        directory: Where to write the script.
        frames: ``"all"``, or ``"short"`` to leave the last one unwritten.
        damage: ``"empty"`` writes a zero-byte last frame, ``"truncate"``
            an undecodable one. Both are crashes caught mid-write, and the
            two are checked differently (size, then decode).
        segfault: Die by SIGSEGV once the writing is done — the known
            shutdown crash, which lands after the work is on disk.
        alpha: The frames' alpha channel, 0-255. The default is fully
            opaque, which makes the flat background the renderer composites
            under them invisible; a partial value is what a test of what
            shows THROUGH a splat needs.
        record: A directory to copy the argv (as `argv.json`) and the
            cameras.json into, for a test that wants to read them.
    """
    import sys

    script = Path(directory) / "stub-brush-splat-render.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shutil, signal, sys\n"
        "from pathlib import Path\n"
        "import numpy as np, cv2\n"
        "args = sys.argv[1:]\n"
        "get = lambda n: args[args.index(n) + 1]\n"
        "cams = json.loads(Path(get('--cameras')).read_text())\n"
        "out = Path(get('--output-dir')); out.mkdir(parents=True, exist_ok=True)\n"
        f"record = {(str(record) if record is not None else None)!r}\n"
        "if record:\n"
        "    Path(record).mkdir(parents=True, exist_ok=True)\n"
        "    Path(record, 'argv.json').write_text(json.dumps(sys.argv[1:]))\n"
        "    shutil.copy2(get('--cameras'), Path(record, 'cameras.json'))\n"
        "n = len(cams['cameras'])\n"
        f"write = n - 1 if {frames!r} == 'short' else n\n"
        "sidecar = '--confidence-sidecar' in args\n"
        "for i in range(write):\n"
        "    img = np.zeros((cams['height'], cams['width'], 4), np.uint8)\n"
        "    img[..., :3] = 100\n"
        f"    img[..., 3] = {int(alpha)}\n"
        "    p = out / ('f%05d.png' % i)\n"
        f"    damage = {damage!r}\n"
        "    if damage and i == write - 1:\n"
        "        blob = cv2.imencode('.png', img)[1].tobytes()\n"
        "        p.write_bytes(b'' if damage == 'empty' else blob[:len(blob)//3])\n"
        "    else:\n"
        "        cv2.imwrite(str(p), img)\n"
        "    if sidecar:\n"
        "        cv2.imwrite(str(out / ('f%05d.conf.png' % i)),\n"
        "                    np.zeros((cams['height'], cams['width']), np.uint8))\n"
        "sys.stderr.write('stub wrote %d of %d frames\\n' % (write, n))\n"
        f"if {segfault!r}:\n"
        "    os.kill(os.getpid(), signal.SIGSEGV)\n"
    )
    script.chmod(0o755)
    return str(script)
