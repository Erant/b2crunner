"""Shared test helpers.

Nothing in this suite reads recorded data. Where a test needs a dataset
shaped like a real run, `orbit_dataset` builds one: the cameras come from
body2colmap's own `OrbitPath`, the same solver `render` uses, so the orbit
conventions under test are checked against the renderer's rather than
against a second copy of the arithmetic they implement.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The orbit an `override_cam_from_mesh` render of a standing subject builds
#: at 720x1280: the photograph's camera at the world origin, the subject's
#: centre two metres in front of it, and the framed lens. The numbers are a
#: real render's (an early ComfyUI-era run), so the orbit geometry —
#: radius, the anchor's near-zero elevation, the lens — is a realistic one
#: rather than a round-number special case.
ORBIT_TARGET = (0.006704419851303101, 0.003536224365234375, -2.0372064113616943)
ORBIT_RESOLUTION = (720, 1280)
ORBIT_FOCAL_PX = 1213.9169918936154
ORBIT_FOCAL_MM = 60.69584959468077
ORBIT_ORIGINAL_FOCAL_PX = 1717.3001708984375


def orbit_extras(**overrides) -> Dict[str, Any]:
    """The `b2c_extras` that render publishes for the orbit above."""
    from body2colmap.path import compute_original_camera_orbit_params

    target = np.asarray(ORBIT_TARGET, dtype=np.float32)
    extras: Dict[str, Any] = {
        "focal_length_mm": ORBIT_FOCAL_MM,
        "initial_rotation": 0.0,
        "orbit_target": target,
        "forward_azimuth_deg": float(
            compute_original_camera_orbit_params(target)["start_azimuth_deg"]),
        "anchor_frame_index": 0,
        "anchor_position": np.zeros(3, dtype=np.float32),
        "original_focal_length": ORBIT_ORIGINAL_FOCAL_PX,
    }
    extras.update(overrides)
    return extras


def orbit_dataset(
    *, n_frames: int = 81, frame_size=(18, 32), n_points: int = 1000, seed: int = 0,
):
    """A render-shaped Dataset on the anchored circular orbit above.

    `n_frames` cameras from `OrbitPath.circular` with `overlap=1`, the way
    `render` builds an anchored circular path: frame 0 on the photograph's
    camera at the world origin, and the orbit closing on itself, so the last
    camera is the first one's twin. Both of those frames carry the anchor
    image, as a render's do; every other frame is a distinct flat grey
    (`frame_size` is (width, height): small stand-ins, the cameras keep
    their 720x1280 intrinsics). Masks are all 1.0, the points a small cloud
    around the target.
    """
    from body2colmap.camera import Camera
    from body2colmap.path import OrbitPath, compute_original_camera_orbit_params

    from pipeline.dataset import Dataset

    extras = orbit_extras()
    target = extras["orbit_target"]
    solved = compute_original_camera_orbit_params(target)
    width, height = ORBIT_RESOLUTION
    cameras = OrbitPath(target=target, radius=float(solved["radius"])).circular(
        n_frames=n_frames,
        elevation_deg=solved["elevation_deg"],
        start_azimuth_deg=solved["start_azimuth_deg"],
        overlap=1,
        camera_template=Camera(
            focal_length=(ORBIT_FOCAL_PX, ORBIT_FOCAL_PX), image_size=(width, height)),
    )

    fw, fh = frame_size
    anchor = np.zeros((fh, fw, 3), dtype=np.uint8)
    anchor[:] = (30, 90, 200)
    anchor[: fh // 4, : fw // 4] = 255
    images = [np.full((fh, fw, 3), 20 + (i * 2) % 200, dtype=np.uint8) for i in range(n_frames)]
    images[0] = anchor.copy()
    images[-1] = anchor.copy()

    rng = np.random.default_rng(seed)
    points = (target + rng.normal(scale=0.3, size=(n_points, 3))).astype(np.float32)
    colors = rng.integers(0, 256, size=(n_points, 3)).astype(np.uint8)

    return Dataset(
        images=images,
        image_names=[f"frame_{i + 1:05d}_.png" for i in range(n_frames)],
        cameras=list(cameras),
        points_3d=(points, colors),
        resolution=(width, height),
        masks=[np.ones((fh, fw), dtype=np.float32) for _ in range(n_frames)],
        reference_image=np.full((fh, 2 * fw, 3), 60, dtype=np.uint8),
        anchor_image=anchor,
        prompt="a figure in a jacket",
        extras=extras,
    )


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
