"""mask_splat — blank out everything the trained splat didn't confidently cover.

Native port of `workflows/api/mask_splat.json`, the stage that sits between
re-rendering a trained splat and the denoise pass that cleans it up. In the
ComfyUI flow it is not a node but a subgraph of eight generic nodes; this
module is that subgraph collapsed into one step:

    LoadDataset.masks               alpha of the splat render
      -> ToBinaryMask(threshold=16) keep only near-opaque pixels
      -> InvertMask                 (ComfyUI MASK is inverted; see below)
      -> ImpactDilateMask(dilation) grow the kept region back out a little
      -> AILab_ImageCombiner        composite the frames over black
      -> BilateralFilterImage       edge-preserving smooth of the result
      -> SaveDataset(masks=zeros)   frames saved fully opaque

Why: a splat re-render has soft, low-alpha fringes wherever the Gaussians
are uncertain — thin hair, silhouette edges, anything under-observed. Feeding
those to the next denoise pass propagates the mush. Keeping only near-opaque
interior (plus a small dilated margin) and blacking out the rest gives the
denoiser a clean, hard-edged subject to work from.

**Superseded, and kept anyway.** `render_splat`'s `confidence` mode does
this job properly — it gates on each Gaussian's multi-view evidence, in
3-D and once, rather than on accumulated alpha per pixel per frame — and
every shipped workflow now runs this step as `mode: passthrough` behind
one. Passthrough is not a no-op: it still does the step's *other* job,
replacing the per-pixel splat alpha in `dataset.masks` with the per-frame
all-1.0 VACE batch, which is what `denoise_pass2` reads and what
`inject_anchor` writes its 0.0 into. That is also why the step stays here
rather than being deleted: the ordering it anchors ("inject_anchor must run
AFTER mask_splat", see steps/anchor_stub.py) still holds, and `mode:
threshold` keeps the recorded run reproducible for an A/B. See
docs/spatial-reinforcement.md.

**Mask conventions.** ComfyUI's MASK is inverted (1.0 = background), so the
graph binarises the *background* and inverts it. This pipeline's convention
is foreground = 1 throughout (see steps/rmbg.py), so the equivalent test is
applied directly to the foreground here: a pixel is kept when
`mask >= 1 - threshold/255`. With the default threshold of 16 that means
alpha >= 239/255, i.e. "essentially opaque".

**Verified** against real recorded output: cyber_6f/splatted ->
cyber_6f/masked_splatted, the actual ComfyUI run of this stage at the
`workflows/pipeline/fast helical.yaml` settings (filter_size=6, dilation=2).
Agreement is a mean absolute error of ~0.25/255 per channel with a maximum
of 15 across the frames checked — visually identical but not bit-exact. The
residual sits at mask edges and is consistent with a difference in the
bilateral filter's border handling: the two agree on which pixels survive
for ~99.85% of pixels, and every disagreement is a boundary pixel the
filter left at a value of 1-3 that rounded to black on one side and not the
other. Nothing bright ever lands on the wrong side. See
tests/test_mask_splat.py.

Two semantics worth recording, both established by fitting against that
recorded output rather than by reading the node source (neither is
guessable, and getting either wrong changes which pixels survive):

  * `ToBinaryMask` compares strictly greater-than against `threshold/255`
    with no rounding. Rounding the mask to 0-255 first — the obvious
    reading of an integer threshold — keeps a visibly different pixel set
    and pushed the max error from 15 to 140.
  * `ImpactDilateMask` uses a plain `dilation x dilation` kernel of ones
    (2x2 for dilation=2), not `2*dilation+1` and not an ellipse.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from ..dataset import Dataset
from ..masks import normalize_mask
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)


@register_step("mask_splat")
class MaskSplatStep(Step):
    """Composite frames over black outside the confidently-covered region.

    inputs:  {"dataset": Dataset} — images plus the splat render's alpha
             as dataset.masks (foreground = 1)
    outputs: {"dataset": Dataset} — masked/filtered images, and masks set
             uniformly to 1.0, i.e. VACE "denoise every one of these
             frames" (the next pass must not treat the blacked-out region
             as "already reference material"; the ComfyUI graph says the
             same thing by saving an all-zero MASK, which SaveDataset
             re-inverts to alpha 255). Matches all 80 non-anchor frames of
             cyber2_6f/masked_splatted.

             The anchor frame is NOT this step's business, and must not be:
             in the recorded run frame_00038_ is the real photo verbatim at
             alpha 0, neither composited nor filtered. `inject_anchor` puts
             it back afterwards. Run that step BEFORE this one and it
             overwrites dataset.masks — the splat alpha this step reads —
             with an all-1.0 batch, and everything below silently becomes a
             no-op. See steps/anchor_stub.py.

    The pipeline YAMLs use filter_size/dilation of 6/2 (`fast helical`),
    12/4 (`helical`, `tiered` first pass) and 4/0 (`tiered` second pass) —
    dilation=0 is a valid no-dilate case and is handled.

    `mode: passthrough` skips all of that and emits the frames unchanged,
    for the shipped case where the render upstream was already
    confidence-gated (`render_splat`'s `confidence` param). The masks are
    replaced either way — that half is the step's real remaining job — and
    passthrough does not need `dataset.masks` at all, since it reads no
    alpha.
    """

    # threshold/sigma_* are advanced because they are not free choices: they
    # were fitted against the recorded ComfyUI run this step reproduces (see
    # the module docstring), and moving them breaks that agreement.
    PARAMS = (
        Param("mode", str, "threshold", choices=("threshold", "passthrough"),
              help="threshold: the recorded ComfyUI subgraph — alpha cut, dilate, "
                   "composite over black, bilateral filter. passthrough: leave the "
                   "frames exactly as they arrived, because render_splat already "
                   "gated them on per-Gaussian confidence; the masks are still "
                   "replaced by the all-1.0 VACE batch either way"),
        Param("filter_size", int, 6, "Bilateral filter diameter", minimum=0),
        Param("dilation", int, 2, "Grow the kept region back out by this many pixels; "
              "0 is a valid no-dilate case", minimum=0),
        Param("threshold", int, 16,
              "Opacity cutoff in the inverted ComfyUI sense: a pixel survives at "
              "alpha >= 1 - threshold/255, so 16 means essentially opaque",
              minimum=1, maximum=255, advanced=True),
        Param("sigma_color", float, 0.5, "Bilateral filter colour sigma, in [0,1] image units",
              advanced=True),
        Param("sigma_space", float, 100.0, "Bilateral filter spatial sigma", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        dataset: Dataset = inputs["dataset"]

        mode = params["mode"]
        filter_size = params["filter_size"]
        dilation = params["dilation"]
        threshold = params["threshold"]
        sigma_color = params["sigma_color"]
        sigma_space = params["sigma_space"]

        if mode == "threshold":
            if dataset.masks is None:
                raise ValueError(
                    "mask_splat needs dataset.masks (the splat render's alpha). "
                    "A dataset loaded from RGBA frames has it; one loaded from "
                    "RGB frames does not."
                )
            images: List[np.ndarray] = [
                _mask_one(
                    img, mask,
                    threshold=threshold,
                    dilation=dilation,
                    filter_size=filter_size,
                    sigma_color=sigma_color,
                    sigma_space=sigma_space,
                )
                for img, mask in zip(dataset.images, dataset.masks)
            ]
            logger.info(
                "mask_splat: %d frames (filter_size=%d, dilation=%d, threshold=%d)",
                len(images), filter_size, dilation, threshold,
            )
        else:
            # Not a copy: the frames go downstream untouched, and the arrays
            # are the same ones every other step passes along by reference.
            # `dataset.masks` is not read at all here — a confidence render's
            # alpha is the gate, already applied to the RGB by the rasteriser
            # — so unlike the threshold path this mode does not need one.
            images = list(dataset.images)
            logger.info(
                "mask_splat: %d frames passed through unfiltered; the splat render "
                "gated them already. Masks replaced by the all-1.0 VACE batch.",
                len(images),
            )

        # Fully opaque, matching the all-zero ComfyUI MASK the graph saves.
        h, w = images[0].shape[:2]
        masks = [np.ones((h, w), dtype=np.float32) for _ in images]

        out = Dataset(
            images=images,
            image_names=list(dataset.image_names),
            cameras=list(dataset.cameras),
            points_3d=dataset.points_3d,
            resolution=dataset.resolution,
            masks=masks,
            reference_image=dataset.reference_image,
            anchor_image=dataset.anchor_image,
            prompt=dataset.prompt,
            splat_path=dataset.splat_path,
            extras=dict(dataset.extras),
        )
        return {"dataset": out}


def _mask_one(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: int,
    dilation: int,
    filter_size: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    bgr = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
    fg = normalize_mask(mask)

    # ToBinaryMask + InvertMask, expressed in the foreground-is-1
    # convention: ComfyUI binarises (1 - fg) > threshold/255 and inverts.
    # Strictly greater-than, no rounding — see module docstring.
    keep = ~((1.0 - fg) > (threshold / 255.0))
    keep_u8 = keep.astype(np.uint8) * 255

    if dilation > 0:
        keep_u8 = cv2.dilate(keep_u8, np.ones((dilation, dilation), np.uint8), iterations=1)

    composited = np.clip(
        bgr.astype(np.float32) * (keep_u8.astype(np.float32) / 255.0)[:, :, None], 0, 255
    ).astype(np.uint8)

    if filter_size <= 0:
        return composited

    # sigma_color is expressed in [0,1] image units (the ComfyUI node's
    # scale); cv2 wants it in the same units as the uint8 data.
    return cv2.bilateralFilter(
        composited, filter_size, sigma_color * 255.0, sigma_space
    )
