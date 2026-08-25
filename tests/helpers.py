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
