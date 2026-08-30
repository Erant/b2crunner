"""Sapiens2 dense prediction: surface normals (`sapiens2_lite`) and
body-part segmentation (`sapiens2_seg`).

Both go through transformers' first-class Sapiens2 support and share
this module because they share everything but a head. The pointmap
head lives in steps/pointmap_splat.py instead, with the geometry it
only exists to feed. Everything below is about the normal step unless
it says otherwise; `Sapiens2SegStep` carries its own docstring.


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

1b is what `pointmap_splat` was developed and measured against — its
normals are the relief signal the depth integration is solved from, and
the pointmap head it pairs with only exists at 1b, so matching the two
sizes keeps that pair consistent. Every number in that step's docstring
came from 1b normals.

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

#: The 1B segmentation head, the third Sapiens2 model this pipeline pulls
#: (normals, pointmap, seg). masktest measured its Face_Neck head splat
#: against exactly this checkpoint.
DEFAULT_SEG_CHECKPOINT = "facebook/sapiens2-seg-1b"

#: Goliath body-part class ids, as far as anything here names them. The full
#: taxonomy is 28 foreground classes; these are the ones a `parts` spec
#: refers to by name. From masktest's face_to_splat.py, which took them from
#: the Sapiens docs (docs/SEG.md).
SEG_CLASSES = {
    0: "Background", 1: "Apparel", 2: "Eyeglass", 3: "Face_Neck", 4: "Hair",
    24: "Lower_Lip", 25: "Upper_Lip", 26: "Lower_Teeth", 27: "Upper_Teeth",
    28: "Tongue",
}

#: Every class except Background, minus teeth and tongue: those appear only
#: inside an open mouth, behind the lips, where the pointmap has nothing
#: sensible to say and a Gaussian built there hangs in the middle of a face.
ALL_FOREGROUND = frozenset(range(1, 29)) - {26, 27, 28}

#: Named `parts` selections. `face` is masktest's measured default and the
#: one every number in its SPEC.md was taken with; `head` is the obvious
#: extension nobody has measured (see the `parts` param's help).
PART_PRESETS = {
    "face": frozenset({3}),
    "head": frozenset({2, 3, 4, 24, 25}),
    "body": ALL_FOREGROUND - {4},
    "all": ALL_FOREGROUND,
    "fg": ALL_FOREGROUND,
}


def parse_parts(spec: str) -> "frozenset[int]":
    """A `parts` spec -> the seg class ids it selects.

    Accepts a preset name (`face`, `head`, `body`, `all`) or a comma list of
    class ids (`"2,3,4"`). masktest's grammar, minus nothing.
    """
    text = str(spec).strip().lower()
    if text in PART_PRESETS:
        return PART_PRESETS[text]
    try:
        parts = frozenset(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise ValueError(
            f"sapiens2_seg: cannot read parts={spec!r}. Give a preset "
            f"({', '.join(sorted(PART_PRESETS))}) or a comma-separated list of "
            f"class ids, e.g. '2,3,4'."
        ) from None
    if not parts:
        raise ValueError(f"sapiens2_seg: parts={spec!r} selects no classes")
    return parts


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


@register_step("sapiens2_seg")
class Sapiens2SegStep(Step):
    """Sapiens2 body-part segmentation — a class map, a soft matte, a box.

    The third head in the family, beside `sapiens2_lite`'s normals and
    `pointmap_splat`'s pointmap, and the one masktest built its face splat
    on. It exists here for `face_pointmap_splat`, which needs a matte of a
    *head* — RMBG-2.0 mattes a subject against a background and has no
    notion of where a face stops.

    inputs:  {"image": np.ndarray BGR}
    outputs: {"mask": HxW float32 in [0,1] — the summed class probability
                      of `parts`, which is the soft silhouette alpha,
              "labels": HxW int32 — the argmax class map, all 29 classes,
              "box": (x0, y0, x1, y1) ints — the tight bounding box of the
                      selected region, for `crop_to_box`}

    **`mask` is a probability map, not a binary mask**, and that is what
    makes it a drop-in for `pointmap_splat`'s `mask` input: that step
    thresholds it at `mask_threshold` for the hard support region and keeps
    the soft values at the silhouette for the Gaussians' opacity. Summing
    the selected classes' probabilities and thresholding at 0.5 is
    equivalent to "the union of these classes won the argmax" wherever the
    union is at all confident, so no morphology is duplicated here —
    `clean_mask` and `soft_alpha` in pointmap_splat.py do it once, for both
    the RMBG and the segmentation paths.

    Softmax order matters and matches masktest: the processor interpolates
    the *logits* back to the source resolution and this softmaxes them
    there, rather than softmaxing at 1024x768 and interpolating
    probabilities. The seg config squashes anisotropically with no padding,
    so there is no letterbox to undo.

    Single image only, deliberately: both of its callers segment one
    photograph, and `box` is not a per-batch concept. Ask for the batched
    path when something needs it.

    Loading is not shared between calls unless the workflow says
    `keep_loaded: true`. The face branch runs this step twice — once on the
    full frame to find the head, once on the crop for the mask that is
    actually used — and takes the second checkpoint read (a local HF cache
    hit) rather than holding 6.5 GB of weights in host RAM through two
    81-frame diffusion passes. Flip `keep_loaded` on both steps to trade the
    other way; `release_vram` below is what makes that safe on the card.
    """

    PARAMS = (
        Param("parts", str, "face",
              "Which body parts the mask covers: a preset (face = Face_Neck "
              "only, masktest's measured default; head = + hair, glasses and "
              "lips, plausible but unmeasured — hair is where the pointmap is "
              "least reliable; body; all) or a comma-separated list of Goliath "
              "class ids"),
        Param("checkpoint", str, DEFAULT_SEG_CHECKPOINT,
              "HF repo for the segmentation head; the family is 0.4b/0.8b/1b/5b",
              advanced=True),
        Param("device", str, None, "Torch device; empty means cuda if available",
              advanced=True),
    )

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._checkpoint = None
        # Set by release_vram(), cleared by the next run(). A flag rather
        # than inferring it from `self._device == "cpu"`: that is also what
        # an explicit `device: cpu` looks like, and re-uploading the model
        # to the card would then override the choice the workflow made.
        self._on_cpu_for_eviction = False

    def load(self, params: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        checkpoint = params["checkpoint"]
        self._device = params["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(checkpoint)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        self._model.to(self._device).eval()
        self._checkpoint = checkpoint
        self._on_cpu_for_eviction = False

    def release_vram(self) -> None:
        import torch

        if self._model is not None and self._device != "cpu":
            self._model.to("cpu")
            self._on_cpu_for_eviction = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._checkpoint = None
        self._on_cpu_for_eviction = False
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from PIL import Image

        if self._model is None or self._checkpoint != params["checkpoint"]:
            self.load(params)
        elif self._on_cpu_for_eviction:
            # Came back from a release_vram(); put it on the card again.
            self._model.to(self._device)
            self._on_cpu_for_eviction = False

        parts = parse_parts(params["parts"])
        image_bgr = np.asarray(inputs["image"])
        height, width = image_bgr.shape[:2]

        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        processed = self._processor(images=[rgb], return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**processed)

        result = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)], return_segmentation_scores=True
        )[0]

        labels = result.segmentation.to(torch.int32).cpu().numpy()
        # `segmentation_scores` are the interpolated LOGITS, not
        # probabilities — softmax them here, on the original pixel grid.
        probs = result.segmentation_scores.float().softmax(0)
        selected = sorted(part for part in parts if part < probs.shape[0])
        if not selected:
            raise ValueError(
                f"sapiens2_seg: parts={params['parts']!r} selects no class this "
                f"checkpoint predicts (it has {probs.shape[0]})"
            )
        mask = probs[selected].sum(0).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

        box = _selected_box(np.isin(labels, selected), width, height)
        logger.info(
            "sapiens2_seg: parts=%s (%s), %d px above 0.5 (%.1f%% of frame), "
            "box %s",
            params["parts"], ", ".join(SEG_CLASSES.get(p, str(p)) for p in selected)
            if len(selected) <= 6 else f"{len(selected)} classes",
            int((mask >= 0.5).sum()), 100.0 * float((mask >= 0.5).mean()), box,
        )
        return {"mask": mask, "labels": labels, "box": box}


def _selected_box(hard: np.ndarray, width: int, height: int):
    """Tight box around the largest connected component of `hard`.

    Largest component, not every pixel: a handful of face-classified pixels
    somewhere else in the frame — a hand, a poster on the wall behind the
    subject — would otherwise stretch the box across the image and hand the
    crop back the resolution it was cut to gain.
    """
    from scipy import ndimage

    if not hard.any():
        raise ValueError(
            "sapiens2_seg: the selected classes cover no pixel at all. Either "
            "the subject is not in frame or `parts` names the wrong classes."
        )
    components, count = ndimage.label(hard)
    if count > 1:
        sizes = ndimage.sum(np.ones_like(components), components,
                            index=np.arange(1, count + 1))
        hard = components == (int(np.argmax(sizes)) + 1)

    rows, cols = np.nonzero(hard)
    return (int(cols.min()), int(rows.min()),
            int(cols.max()) + 1, int(rows.max()) + 1)
