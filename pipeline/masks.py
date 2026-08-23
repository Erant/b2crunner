"""One place for the mask convention, because there are two of them.

This pipeline's rule is **foreground = 1** everywhere (the inverse of
ComfyUI's MASK, which is 1.0 = background — see steps/mask_splat.py). But
the *range* varies by origin, and both forms circulate freely through a
`Dataset`:

  * `steps/rmbg.py` and `steps/render.py` produce float32 in [0, 1].
  * `Dataset.from_disk()` produces the raw uint8 alpha channel, [0, 255].

Anything consuming `dataset.masks` therefore has to handle both. Doing that
inline is how `colmap_export` ended up with `np.clip(m * 255.0, 0, 255)`,
which silently binarises a uint8 mask (every value >= 1 saturates to 255)
and so throws away the soft edge of a mask that came from disk. The same
mistake was live in `Dataset.to_disk()`, which dropped masks entirely.

Use these helpers instead of re-deriving the rule.
"""

from __future__ import annotations

import numpy as np


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Return a float32 HxW mask in [0, 1] with foreground = 1.

    Accepts float [0,1], uint8 [0,255], and HxWx1 / HxWxC forms (the first
    channel wins). The range is inferred from the data rather than the
    dtype, since a float32 array carrying 0-255 values shows up whenever a
    mask has been through an image codec.
    """
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def mask_to_alpha_u8(mask: np.ndarray) -> np.ndarray:
    """Return a uint8 [0, 255] alpha channel from a mask of either form."""
    return np.clip(normalize_mask(mask) * 255.0, 0, 255).astype(np.uint8)
