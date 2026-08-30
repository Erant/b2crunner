"""Sapiens2 dense prediction: surface normals (`sapiens2_lite`).

Goes through transformers' first-class Sapiens2 support. Two sibling heads
that used to live here or alongside — body-part segmentation
(`sapiens2_seg`) and the pointmap head that fed `pointmap_splat` — were
removed on 2026-08-30 with the rest of the photo-to-splat work; see
pipeline/README.md. Nothing here depends on them and the checkpoints are no
longer prefetched.


Verified against real inference on an L40S pod: both the single-image
(`inputs["image"]`) and batched (`inputs["images"]`) paths run cleanly
against real wan22_vace_denoise output frames, producing correctly-shaped,
properly L2-normalized (min/max within [-1, 1]) normal maps. That run used
the 0.4B checkpoint, which loaded in well under a second — cheap enough to
not bother with a disk-cached/pre-warmed instance. The default is now 1B
(see below), which is ~3.4x those weights; the "don't bother pre-warming"
conclusion has not been re-measured against it.

Uses transformers' first-class Sapiens2 support (added to the library
directly — see the model doc at
https://huggingface.co/docs/transformers/model_doc/sapiens2) rather than
the older facebookresearch/sapiens (v1) "lite" torchscript inference path
this step's name originally referenced. That older path needed its own
minimal-dependency install specifically to avoid the full mmcv/OpenMMLab
stack; transformers' AutoModel path achieves the same "no heavy CV
framework" goal for free, so the "lite" framing in this step's registered
name is about that outcome, not about using the old inference script.

Default checkpoint is the 1B normal-estimation variant —
facebook/sapiens2-normal-1b (2026-08-29; it was 0.8b before that). Pass
params["checkpoint"] for another size: the family is 0.4b/0.8b/1b/5b, all
four confirmed present on the Hub as facebook/sapiens2-normal-<size> (an
earlier version of this docstring listed "0.4b/0.6b/1b/2b/2b" from the
model doc, which is wrong).

1b was chosen when `pointmap_splat` integrated depth from these normals
and its own head only existed at 1b. That consumer is gone; the size is now
just a quality/VRAM decision for the normal maps stage 2 conditions on, and
nothing measures against it any more. 0.8b is the obvious thing to try if
the 6.16 GB below is in the way.

Size is a VRAM decision as well as a quality one: 1B is **6.16 GB** of
weights (measured from the cached blob), against 0.8B's 3.54 GB and 0.4B's
1.81 GB — on top of activations that already needed `batch_size: 2` rather
than the step's default of 8 to fit a 12 GB card at 720x1280 (see
docs/docker-build-notes.md). That headroom is now thinner, so a 12 GB card
running the batched path at upscaled resolution is the case to watch; on a
48 GB L40S or larger it is not a concern. Drop to 0.8b if a small card
starts OOMing here — the step takes the checkpoint as a param precisely so
that is a workflow edit, not a code change.

Output is raw (unnormalized) XYZ normals in camera space, L2-normalized to
[-1, 1] via the image processor's post_process_normal_estimation and
resized back to the input resolution — no background masking applied here;
pipeline/steps/brush.py combines this output with a separate foreground
mask (rmbg's) at the point the normal map actually gets written to disk,
matching nodes/brush_node.py's original design ("a normal map that carries
its own alpha keeps it; otherwise borrow the RGB frame's mask").
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "facebook/sapiens2-normal-1b"

@register_step("sapiens2_lite")
class Sapiens2LiteStep(Step):
    """Sapiens2 normal-map estimation.

    inputs: {"image": np.ndarray BGR} or {"images": List[np.ndarray] BGR}
    outputs: {"normal_map": np.ndarray HxWx3 float32 in [-1,1]}
             or {"normal_maps": List[np.ndarray]} for the batched path
    """

    PARAMS = (
        Param("batch_size", int, 8,
              "Images per forward pass. Worth lowering at upscaled resolution: "
              "1080x1920 float32 normal maps are ~2 GB of host RAM for 81 frames",
              minimum=1),
        Param("checkpoint", str, DEFAULT_CHECKPOINT,
              "HF repo for the normal-estimation model; the family is "
              "0.4b/0.8b/1b/5b. Smaller is the lever if a 12 GB card OOMs",
              advanced=True),
        Param("device", str, None, "Torch device; empty means cuda if available",
              advanced=True),
    )

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None

    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForNormalEstimation

        checkpoint = params["checkpoint"]
        self._device = params["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = AutoModelForNormalEstimation.from_pretrained(checkpoint).to(self._device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        if self._model is None:
            self.load(params)

        if "images" in inputs:
            batch_size = params["batch_size"]
            images = inputs["images"]
            normals = []
            for i in range(0, len(images), batch_size):
                normals.extend(self._run_batch(images[i : i + batch_size]))
            return {"normal_maps": normals}

        return {"normal_map": self._run_batch([inputs["image"]])[0]}

    def _run_batch(self, images_bgr: List[np.ndarray]) -> List[np.ndarray]:
        import torch
        from PIL import Image

        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_bgr]
        sizes = [(img.shape[0], img.shape[1]) for img in images_bgr]

        inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)

        result = self._processor.post_process_normal_estimation(
            outputs, source_sizes=sizes, target_sizes=sizes
        )

        out = []
        for r in result:
            normals = r["normals"]  # (3, H, W), unit vectors in [-1, 1]
            normals_hwc = normals.permute(1, 2, 0).float().cpu().numpy()
            out.append(normals_hwc.astype(np.float32))
        return out
