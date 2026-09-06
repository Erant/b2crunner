"""Flow alignment: pull a set of training views onto the splat's own consensus.

The final `.ply` comes out markedly softer than the frames it was trained
on — measured on the refinesplat rig (docs/final-splat-alignment-guide.md):
the trained splat's face scored 52.6 raw Laplacian variance against 400-770
for the SeedVR2 frames it was fitted to, and 75-82 for the *un-upscaled*
720p ones. **The fit is what destroys the detail, not the upscaler**, so no
better upscaler fixes it (one was tried, and made things worse).

The mechanism is local texture disagreement between views. Dense optical
flow from each training frame to the splat's own render at that camera gives
a *whole-region* rigid shift of about 0.5 px — the poses are fine — but a
*per-pixel* flow of 1.7-3.6 px mean, p90 up to 8. Each generated frame puts
skin, hair and finger micro-texture in a slightly different place, and
brush's photometric loss averages the disagreement into a blurred consensus.
The proof that this is the binding constraint: unsharp-masking the splat's
own renders and retraining on them — perfectly view-consistent by
construction, carrying no new information — moved band-limited face
sharpness 91 -> 228. brush absorbs *consistent* detail almost completely.

So the upscale is a texture-registration problem, not a per-frame quality
problem, and this module is the registration half. Per view: DIS optical
flow from the frame to the splat's render at that camera, smoothed (sigma 6
px) so it is a texture correction and not a per-pixel scramble, magnitude
capped (6 px) so a bad match cannot tear the frame, zeroed outside the
subject, and applied to the frame by a Lanczos backward warp. `steps/brush.py`
runs it between training invocations; four iterations is where it saturates.

**Lanczos, never bilinear.** A single 0.5 px bilinear resample of undamaged
face crops drops their raw score from 738 to 134, where Lanczos drops it to
~480 — an early "alignment makes it worse" result was entirely that bug.

**Warp the pristine originals every iteration, never a warp of a warp.**
That invariant belongs to the caller (this module is stateless and warps
whatever it is handed), and it is what makes the loop a many-to-one
contraction onto one consensus rather than pairwise merging with unbounded
drift. See `steps/brush.py`'s alignment block.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: The colour both sides of the comparison are flattened onto before the
#: flow is measured. Only the *agreement* between the two matters, so the
#: value is free — but it must be the same on both sides, which is why the
#: renders are asked for on this grey too (steps/brush.py) rather than on
#: black: compositing a frame's soft matte edge over black and its render
#: over grey would put a difference into the flow that is not in the
#: textures. Mid-grey rather than black also keeps the silhouette edge from
#: dominating the gradient field DIS matches on.
BACKGROUND = (0.5, 0.5, 0.5)

#: Alpha (of 255) above which a pixel counts as subject. The flow is zeroed
#: outside it before smoothing, so the correction fades to nothing at the
#: silhouette instead of dragging background in over it.
_FOREGROUND_ALPHA = 32

#: Per-thread DIS instances. `calc` is not reentrant — it keeps internal
#: scratch buffers — so one per worker thread, created on first use.
_local = threading.local()


@dataclass(frozen=True)
class AlignStats:
    """How far the frames were asked to move, for the run log.

    `mean` and `p90` are of the smoothed field *before* the cap, over the
    subject only — i.e. the disagreement that was measured, not the part of
    it that was applied. That is the quantity the guide's §0 numbers are
    (1.7-3.6 px mean, p90 up to 8), so a run can be compared against them,
    and a p90 pinned at the cap is the sign that the cap is binding.
    """

    mean: float
    p90: float


def _dis() -> "cv2.DISOpticalFlow":
    """This thread's flow estimator.

    MEDIUM preset with the finest scale enabled: the correction wanted here
    is the per-pixel texture disagreement, which lives entirely in the
    finest levels that the default `finest_scale` skips.
    """
    dis = getattr(_local, "dis", None)
    if dis is None:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(0)
        _local.dis = dis
    return dis


def _flatten(frame: np.ndarray) -> np.ndarray:
    """`frame` composited over `BACKGROUND`, as BGR uint8.

    A 3-channel frame is already flat and is returned as-is: the export
    writes RGB only when there are no mattes at all, and inventing an alpha
    for it would be a different image than the one brush trains on.
    """
    if frame.shape[2] == 3:
        return frame
    alpha = frame[..., 3:4].astype(np.float32) / 255.0
    background = np.array(BACKGROUND[::-1], dtype=np.float32) * 255.0
    flat = frame[..., :3].astype(np.float32) * alpha + background * (1.0 - alpha)
    return np.clip(flat, 0, 255).astype(np.uint8)


def align_view(
    frame: np.ndarray,
    render: np.ndarray,
    *,
    sigma: float,
    cap: float,
    grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, AlignStats]:
    """Warp one training frame onto its own render.

    Args:
        frame: The training view as it goes to disk — BGR or BGRA uint8.
            The *pristine* one: see the module docstring.
        render: The splat rendered at that view's camera, BGR uint8,
            composited over `BACKGROUND`.
        sigma: Gaussian smoothing of the flow field, in pixels.
        cap: Largest displacement applied, in pixels. Above it the field is
            scaled down, direction kept.
        grid: A precomputed `(gx, gy)` pixel grid for this frame size, so a
            batch builds it once. Optional; built here when absent.

    Returns:
        `(warped, stats)` — the frame, alpha and all, resampled by the
        capped field with `cv2.INTER_LANCZOS4`.
    """
    if frame.shape[:2] != render.shape[:2]:
        raise ValueError(
            f"Frame {frame.shape[:2]} and render {render.shape[:2]} differ in "
            f"size; the flow between them would not describe either."
        )

    a = cv2.cvtColor(_flatten(frame), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(render if render.shape[2] == 3 else render[..., :3],
                     cv2.COLOR_BGR2GRAY)
    flow = _dis().calc(a, b, None)

    # Outside the subject the two images are the same flat grey, so whatever
    # DIS reports there is noise fitted to nothing. Zeroed BEFORE the blur,
    # which is what makes the applied field fall off across the silhouette
    # rather than stopping at it.
    if frame.shape[2] == 4:
        background = frame[..., 3] <= _FOREGROUND_ALPHA
        flow[background] = 0.0
        foreground = ~background
    else:
        foreground = np.ones(frame.shape[:2], dtype=bool)

    kernel = int(sigma * 3) | 1
    flow = cv2.GaussianBlur(flow, (kernel, kernel), sigma)

    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    flow *= np.minimum(1.0, cap / (magnitude + 1e-6))[..., None]

    if grid is None:
        grid = pixel_grid(frame.shape[:2])
    gx, gy = grid
    # The flow maps frame -> render, and this is a backward map, so the
    # inverse is approximated by its negation. Exact only for a locally
    # constant field, which after a sigma-6 blur is very nearly what this
    # is — and the measured trajectory (guide §3) is of this warp, not of a
    # properly inverted one.
    warped = cv2.remap(
        frame, gx - flow[..., 0], gy - flow[..., 1], cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    measured = magnitude[foreground]
    stats = AlignStats(
        mean=float(measured.mean()) if measured.size else 0.0,
        p90=float(np.percentile(measured, 90)) if measured.size else 0.0,
    )
    return warped, stats


def pixel_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """The `(gx, gy)` identity map `cv2.remap` samples against."""
    height, width = shape
    gy, gx = np.mgrid[0:height, 0:width].astype(np.float32)
    return gx, gy


def align_views(
    frames: Sequence[np.ndarray],
    renders: Sequence[np.ndarray],
    *,
    sigma: float,
    cap: float,
    workers: Optional[int] = None,
) -> Tuple[List[np.ndarray], AlignStats]:
    """`align_view` over a whole batch, in parallel.

    The flow was the dominant cost of an alignment iteration — the guide
    measured about 60 s for 81 frames at 1080x1920 against ~4 s of rendering
    and ~47 s of fine-tuning — and it is embarrassingly parallel across
    frames, so it is run across them here: 9.9 s for the same 81 frames on a
    4070 Ti (measured 2026-09-06 against the recorded run), which takes it
    out of the critical path. Threads rather than processes: DIS, the blur
    and the remap all run inside OpenCV with the GIL released, and the
    frames are hundreds of MB that would otherwise be pickled to and from a
    pool.

    Returns:
        `(warped, stats)` — the warped frames in input order, and the mean
        and p90 of the per-frame means.
    """
    if len(frames) != len(renders):
        raise ValueError(
            f"{len(frames)} frame(s) against {len(renders)} render(s); the "
            f"alignment matches each view to its own camera."
        )
    if not frames:
        return [], AlignStats(0.0, 0.0)

    grid = pixel_grid(frames[0].shape[:2])

    def one(pair):
        frame, render = pair
        # Only frames of the batch's own size share the grid; a batch is one
        # dataset, so the fallback is just correctness insurance.
        return align_view(
            frame, render, sigma=sigma, cap=cap,
            grid=grid if frame.shape[:2] == frames[0].shape[:2] else None,
        )

    count = workers or min(8, os.cpu_count() or 1)
    with ThreadPool(min(count, len(frames))) as pool:
        results = list(pool.map(one, zip(frames, renders)))

    warped = [result[0] for result in results]
    means = [result[1].mean for result in results]
    p90s = [result[1].p90 for result in results]
    return warped, AlignStats(mean=float(np.mean(means)), p90=float(np.mean(p90s)))
