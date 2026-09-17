"""refine_texture — sharpen a meshify atlas with FLUX.2 klein: character sheets, or view by view.

**`mode: sheets` (the default, 2026-09-17).** The atlas is re-laid as three
character sheets (pipeline/view_atlas.py): front and back, six oblique
full-body views, four close-up head views — every panel a render of the whole
mesh, so klein always sees a person and its priors apply — and each sheet is
one klein img2img at half its size (2048 px for a 4096 sheet). The edit comes
back to the atlas texel by texel as a delta (b2ctrain out/mesh/view_atlas_m3:
mottled cloth to fabric weave, the petticoat under the hem, the ear and jaw
band that no body view could resolve). Three klein calls, ~90 s each on a
4070 Ti at 8.6 GB, against eleven views before; the face and every texel
`protect_path` names are never repainted and the transfer puts them back
bit-exact. The reserve — surface no panel owns, 40 % of it the TSDF's inner
wall nobody sees — keeps the start texture.

**`mode: views`** is the loop below, kept for the A/B.

The TEXTure / Text2Tex loop of b2ctrain/docs/mesh-plan.md: views of the
textured mesh are rendered (`b2ctrain mesh-render`), the pixels this view
sees better than any earlier one are repainted by klein as img2img inside
that mask (the rest of the render is kept and conditions the repaint), and
the repainted pixels are baked back into the texture where this view's
weight beats the best so far (`b2ctrain mesh-backproject`). The sequential
keep/repaint mask is what keeps the views from blurring each other: every
texel is painted once, by the view that sees it best.

Measured on both local subjects (b2crunner memory uv_texture_klein_refine):
strength 0.8 at 12 steps is the working point (fabric weave, satin sheen,
smears gone, the waistband and buttons stay put to a few px); 0.4-0.65 is a
VAE round trip and 1.0 ignores the structure. The front and back photo
panels go in as SEPARATE reference streams (the pipeline API takes one
`image_reference`; the extra ones are injected through
`prepare_image_latents`, each VAE-encoded on its own with its own index —
how the base pipeline conditions on several images). Init images are
capped at 1 MP by the pipeline, silently, so the views are 704 x 1408 and
1024 x 1024. Rear head close-ups are never rendered: a featureless hair
mask with a prompt naming a person makes klein paint a whole figure into
the hair; the body views cover the back of the head.

`face_policy` decides what klein never touches (mesh-plan.md decision 4;
settled 2026-09-16 on `protect_cap`: the face stays the photograph's
pixels, the other two are kept for an A/B):
  protect_cap   the projected photograph's own footprint (the cap's
                coverage, closed and eroded): identity kept exactly, a tone
                / sharpness step where klein's ring meets the photograph.
  protect_head  the wider face band about the head centre.
  none          klein repaints the face too, head views first, so it is one
                coherent 1024 px pass: seamless, mild identity drift.

One resident klein: this step runs in the wan22 env (diffusers, torchao)
and loads the model once for the whole loop; the two raster halves are
b2ctrain calls. VRAM 6.9 GB peak at 704 x 1408 with two 0.3 MP references;
the text encoder runs on the CPU when the card has less than 9 GB free
(one prompt, a few seconds). ~17 s a view.

Outputs, under `output_dir` beside the atlas: `texture_final.png`,
`mesh_klein.obj` + `.mtl` naming it, `refine_texture.json` (per view: what
it claimed, how long klein took, how much it changed). With `debug_dir`
set, `<i>_<name>/` there keeps every view's render, repaint mask and
repaint (the PNGs; the float maps and the working texture are not kept).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..proc import ProcessFailed, stream_command
from ..registry import register_step
from ..step import Param, Step

logger = logging.getLogger(__name__)

DEFAULT_REPO = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_FP8_REPO = "black-forest-labs/FLUX.2-klein-4b-fp8"
DEFAULT_FP8_FILE = "flux-2-klein-4b-fp8.safetensors"
#: klein-4B's `text_encoder/` is Qwen3-4B byte for byte (checked tensor by
#: tensor, 2026-09-17), so Qwen's own fp8 release stands in for it: 4.8 GB
#: to download instead of 8, dequantised to bf16 at load (see
#: `Klein.load_text_encoder`), the klein output within 1.8/255 of bf16's
#: (b2ctrain out/mesh/view_atlas_m3/te_fp8/README.md).
DEFAULT_TEXT_ENCODER = "Qwen/Qwen3-4B-FP8"

#: Everything the step loads from the bf16 repo: the tokenizer, the VAE and
#: the transformer's config (the transformer's weights come from the fp8
#: file, the text encoder's from DEFAULT_TEXT_ENCODER). `transformer/*.safetensors`
#: is 8 GB nothing opens, `text_encoder/*` 8 GB of bf16 the fp8 file replaces.
KLEIN_ALLOW_PATTERNS = ["model_index.json", "tokenizer/*", "vae/*", "transformer/config.json", "scheduler/*"]
#: The fp8 text encoder: weights, configs and its tokenizer files (a few MB).
TEXT_ENCODER_ALLOW_PATTERNS = ["*.safetensors", "*.json", "merges.txt", "vocab.json"]

DEFAULT_PROMPT_SHEET = (
    "Restore clean, sharp natural surface detail, removing mottled noise and blur. Preserve the exact pixel alignment, pose, "
    "silhouettes, face, clothing design and colors. Do not move, resize, rotate or add anything. {caption}. "
    "Flat even lighting, plain grey background. This is a character reference sheet of one person. ")
SHEET_LAYOUT_PROMPTS = {
    "main": "Preserve its exact layout: large front and back views across the top, small side views at bottom left, "
            "top and bottom views at bottom center, empty grey bottom right.",
    "extra": "Preserve its exact layout: full-body views of the same person from oblique camera angles in a grid.",
    "head": "Preserve its exact layout: close-up views of the same person's head and shoulders from different angles in a grid.",
}

DEFAULT_PROMPT_BODY = (
    "A photograph of {caption}. Keep the pose, the framing and the silhouette exactly as they are; "
    "sharp fabric weave, seams and details, even studio lighting, on a plain grey background.")
DEFAULT_PROMPT_HEAD = (
    "A close-up portrait photograph of {caption}. Keep the pose, the framing and the silhouette exactly as they are; "
    "sharp skin, hair and eyes, even studio lighting, on a plain grey background.")


# -- pure helpers (tests/test_refine_texture.py) -------------------------------

def repaint_mask(alpha: np.ndarray, cos: np.ndarray, dens: np.ndarray, best: np.ndarray, power: float, gain: float,
                 dilate: int) -> Tuple[np.ndarray, np.ndarray]:
    """(the texels this view claims, the dilated mask klein repaints).

    A subject pixel is claimed when this view's weight (facing^power x
    texel density) beats `gain` times the best weight recorded for its
    texel, or the texel was never painted. Protected texels carry +inf and
    are never beaten. The repaint mask is the claim dilated by `dilate`
    pixels (the latent mask is 16 px blocks; the extra ring lets klein
    blend into what is kept) and clipped to the subject.
    """
    import cv2

    subject = alpha > 127
    weight = np.power(np.clip(cos, 0.0, 1.0), power) * dens
    claim = subject & ((weight > gain * best) | (best < 1e-6))
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        painted = cv2.dilate(claim.astype(np.uint8), kernel).astype(bool) & subject
    else:
        painted = claim
    return claim, painted


def feather_repaint(repainted: np.ndarray, render: np.ndarray, mask: np.ndarray, alpha: np.ndarray, feather: int) -> np.ndarray:
    """klein's output composited over the render: inside the mask, with the
    edge softened over `feather` px (an erosion then a blur, so the blend
    lives inside the repaint), and only where the subject is."""
    import cv2

    m = (mask.astype(np.float32) / 255.0) * (alpha.astype(np.float32) / 255.0)
    if feather > 0:
        k = feather // 2 * 2 + 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))
        m = cv2.GaussianBlur(m, (0, 0), feather / 2.0) * (alpha.astype(np.float32) / 255.0)
    m = m[..., None]
    return np.clip(repainted.astype(np.float32) * m + render.astype(np.float32) * (1.0 - m) + 0.5, 0, 255).astype(np.uint8)


def view_order(body_azimuths: Sequence[float], head_azimuths: Sequence[float], head_first: bool) -> List[Tuple[str, float]]:
    """(kind, azimuth) in loop order. Head views first (the whole face in
    one coherent pass) unless told otherwise."""
    head = [("head", float(a)) for a in head_azimuths]
    body = [("body", float(a)) for a in body_azimuths]
    return head + body if head_first else body + head


def fit_reference(image_rgb: np.ndarray, max_pixels: int) -> np.ndarray:
    """A reference panel scaled to at most `max_pixels`, sides a multiple of 16 (the VAE's patch)."""
    import cv2

    h, w = image_rgb.shape[:2]
    s = min(1.0, (max_pixels / float(w * h)) ** 0.5)
    nw, nh = max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def read_f32(path: Path, shape: Tuple[int, ...]) -> np.ndarray:
    data = np.fromfile(path, np.float32)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {data.size} floats, expected {shape}")
    return data.reshape(shape)


# -- klein ----------------------------------------------------------------------

def snapshot_dir(repo: str, allow_patterns: Sequence[str]) -> Path:
    """The local snapshot of an HF repo's files (downloaded if the network is on and they are missing)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=list(allow_patterns)))


def lift_init_pixel_cap(pipe: Any, max_pixels: int) -> None:
    """The klein pipelines silently cap the init image at 1 MP; the sheets need 4. The cap is a literal
    `1024 * 1024` in `__call__` (twice for the init image, twice for the references); the first two are
    rewritten in the pipeline's own source and the class swapped for one carrying the patched method.
    Asserting the count means a diffusers whose pipeline changed shape fails here, loudly, not silently at 1 MP."""
    import importlib
    import inspect
    import textwrap

    cls = type(pipe)
    if getattr(cls, "_b2c_pixel_cap", None) == max_pixels:
        return
    base = cls.__mro__[1] if getattr(cls, "_b2c_pixel_cap", None) else cls
    source = textwrap.dedent(inspect.getsource(base.__call__))
    needle = "1024 * 1024"
    if source.count(needle) != 4:
        raise RuntimeError(f"refine_texture: {base.__name__}.__call__ has {source.count(needle)} pixel caps, expected 4; diffusers changed")
    source = source.replace(needle, f"{max_pixels}", 2)
    namespace = vars(importlib.import_module(base.__module__)).copy()
    exec(compile(source, f"<{base.__name__}.__call__ cap {max_pixels}>", "exec"), namespace)
    pipe.__class__ = type(base.__name__ + "HighRes", (base,), {"__call__": namespace["__call__"], "_b2c_pixel_cap": max_pixels})


def chunk_linears(module: Any, rows: int = 2048) -> None:
    """Every Linear of the transformer runs in row chunks: torchao's dynamic fp8 activation quantisation
    otherwise materialises a float32 copy of the whole token stream per layer (34k tokens at a 2048 sheet)."""
    import types

    import torch

    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and not getattr(m, "_b2c_chunked", False):
            original = m.forward

            def chunked(self_, x, _f=original, _rows=rows):
                if x.numel() // x.shape[-1] <= _rows:
                    return _f(x)
                flat = x.reshape(-1, x.shape[-1])
                out = torch.empty((flat.shape[0], self_.out_features), device=x.device, dtype=x.dtype)
                for start in range(0, len(flat), _rows):
                    out[start:start + _rows] = _f(flat[start:start + _rows])
                return out.reshape(*x.shape[:-1], self_.out_features)

            m.forward = types.MethodType(chunked, m)
            m._b2c_chunked = True  # type: ignore[attr-defined]


class Klein:
    """FLUX.2 klein 4B (fp8 transformer) as an inpainting img2img, resident."""

    def __init__(self, repo: str, fp8_repo: str, fp8_file: str, text_encoder_device: str, text_encoder: str = DEFAULT_TEXT_ENCODER,
                 max_pixels: int = 1024 * 1024) -> None:
        import torch
        from diffusers import AutoencoderKLFlux2, Flux2KleinInpaintPipeline

        from ..flux2_fp8 import load_flux2_fp8_transformer

        self.repo = repo
        self.text_encoder = text_encoder
        self.text_encoder_device = text_encoder_device
        self._embeds: Dict[str, Any] = {}
        transformer = load_flux2_fp8_transformer(repo_id=fp8_repo, filename=fp8_file, config_repo=repo).eval().to("cuda")
        vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", dtype=torch.bfloat16).to("cuda").eval()
        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(repo, text_encoder=None, tokenizer=None, transformer=transformer,
                                                              vae=vae, dtype=torch.bfloat16)
        if max_pixels > 1024 * 1024:
            lift_init_pixel_cap(self.pipe, max_pixels)
            # A 4 MP sheet on a 12 GB card: tiled VAE, and the linears chunked so the fp8 activation
            # quantisation never holds a whole 34k-token activation in float32 at once.
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            chunk_linears(self.pipe.transformer)
        self._orig_prepare = self.pipe.prepare_image_latents
        self.extra_refs: List[Any] = []
        pipe = self.pipe

        def prepare_with_extras(images, *args, **kwargs):
            return self._orig_prepare(list(images) + self.extra_refs, *args, **kwargs)

        pipe.prepare_image_latents = prepare_with_extras
        logger.info("refine_texture: klein loaded, %.1f GB free", torch.cuda.mem_get_info()[0] / 2 ** 30)

    def load_text_encoder(self, device: str):
        """The fp8 Qwen3-4B, dequantised to bf16 at load, on either device.

        The file is the win (4.8 GB instead of 8), not the fp8 matmul: the
        encoder runs four prompts and is dropped, and transformers' fine-grained
        fp8 kernel path wants the `kernels` package at one exact minor (0.16;
        the image has 0.17, the pod died on it) plus a kernel fetched from the
        Hub at run time. `dequantize=True` skips all of that and lands within
        3 % of the original bf16 encoder (closer than the kernel path did).

        transformers 5.16/5.17's quantizer has one more trap: its
        tensor-parallel hook dereferences a table that is None for Qwen3 (no
        tensor parallel here, so the plan is returned untouched).
        """
        import torch
        from transformers import Qwen3ForCausalLM

        try:
            from transformers.quantizers import quantizer_finegrained_fp8 as fp8q

            hook = fp8q.FineGrainedFP8HfQuantizer.update_tp_plan
            if not getattr(hook, "_b2c_guarded", False):
                def guarded(self_, config, _orig=hook):
                    try:
                        return _orig(self_, config)
                    except AttributeError:
                        return config
                guarded._b2c_guarded = True  # type: ignore[attr-defined]
                fp8q.FineGrainedFP8HfQuantizer.update_tp_plan = guarded
        except ImportError:
            pass
        from transformers import FineGrainedFP8Config

        kwargs: Dict[str, Any] = dict(dtype=torch.bfloat16, device_map=device, quantization_config=FineGrainedFP8Config(dequantize=True))
        # Resolved through the cache the way the prefetch probes it (pipeline/models.py), so an offline
        # pod and a warm volume agree on what "present" means; transformers' own subfolder lookups do not.
        return Qwen3ForCausalLM.from_pretrained(snapshot_dir(self.text_encoder, TEXT_ENCODER_ALLOW_PATTERNS), **kwargs).eval()

    def encode(self, prompt: str):
        """Prompt embeddings, cached per prompt; the text encoder is loaded for the call and dropped."""
        if prompt in self._embeds:
            return self._embeds[prompt]
        import torch
        from diffusers import Flux2KleinPipeline
        from transformers import AutoTokenizer

        device = self.text_encoder_device
        if device == "auto":
            device = "cuda" if torch.cuda.mem_get_info()[0] / 2 ** 30 > 9.0 else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(snapshot_dir(self.repo, KLEIN_ALLOW_PATTERNS) / "tokenizer")
        encoder = self.load_text_encoder(device)
        with torch.no_grad():
            emb = Flux2KleinPipeline._get_qwen3_prompt_embeds(text_encoder=encoder, tokenizer=tokenizer, prompt=prompt, dtype=torch.bfloat16,
                                                              device=torch.device(device), max_sequence_length=512, hidden_states_layers=[9, 18, 27])
        del encoder
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._embeds[prompt] = emb.to("cuda")
        return self._embeds[prompt]

    def set_references(self, refs_rgb: Sequence[np.ndarray]) -> None:
        """The first reference goes through the pipeline's own slot, the rest as extra token streams."""
        from PIL import Image

        pil = [Image.fromarray(r) for r in refs_rgb]
        self.primary = pil[0] if pil else None
        self.extra_refs = [self.pipe.image_processor.preprocess(r, r.height, r.width) for r in pil[1:]]

    def repaint(self, render_rgb: np.ndarray, mask: np.ndarray, prompt: str, strength: float, steps: int, guidance: float, seed: int) -> np.ndarray:
        import torch
        from PIL import Image

        h, w = render_rgb.shape[:2]
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.no_grad():
            out = self.pipe(prompt=None, prompt_embeds=self.encode(prompt), image=Image.fromarray(render_rgb), image_reference=self.primary,
                            mask_image=Image.fromarray(mask), height=h, width=w, strength=strength, num_inference_steps=steps,
                            guidance_scale=guidance, generator=generator).images[0]
        if out.size != (w, h):
            logger.warning("refine_texture: klein returned %s for %dx%d; resizing back", out.size, w, h)
            out = out.resize((w, h), Image.LANCZOS)
        # The raster halves run as b2ctrain processes between repaints: hand the inference's transient
        # blocks back to the driver, or a shared card has nothing left for them.
        torch.cuda.empty_cache()
        return np.asarray(out.convert("RGB"))


# -- the step ------------------------------------------------------------------

@register_step("refine_texture")
class RefineTextureStep(Step):
    """The klein texture loop over a meshify atlas (see the module docstring).

    inputs:  {"mesh_dir": str — the atlas directory meshify wrote,
              "mesh_stats"?: dict — meshify's stats (the head centre for the head cameras),
              "front_image": HxWx3 uint8 BGR — the photograph's front panel (reference 1),
              "back_image"?: HxWx3 uint8 BGR or None — the back panel (reference 2),
              "caption"?: str — the subject description the prompts are built from,
              "texture_path"?: str — start from this texture instead of the atlas's (photo_texture's),
              "protect_path"?: str — an R x R mask of texels klein never repaints, on top of face_policy's}
    outputs: {"texture_path": str, "mesh_path": str (mesh_klein.obj), "refine_texture_stats": dict}
    """

    PARAMS = (
        Param("trainer_path", str, "b2ctrain", "The b2ctrain binary (mesh-render / mesh-backproject, views mode)", advanced=True),
        Param("output_dir", str, help="Where the refined texture and the per-view renders go (the run's mesh/)"),
        Param("device", int, 0, "CUDA device index for the raster halves", advanced=True),
        Param("mode", str, "sheets", "sheets: three character sheets, one klein call each (pipeline/view_atlas.py); "
              "views: the per-view TEXTure loop", choices=("sheets", "views")),
        Param("sheet_res", int, 4096, "Side of each sheet in texels", minimum=1024, advanced=True),
        Param("sheet_input", int, 2048, "Side klein sees a sheet at (the init image; 2048 = 4 MP, 8.6 GB on a 4070 Ti)", minimum=512),
        Param("extra_panels", int, 6, "Oblique full-body views on the second sheet (0 = none)", minimum=0, maximum=12),
        Param("head_panels", int, 4, "Close-up head views on the third sheet (0 = none)", minimum=0, maximum=8),
        Param("extra_scope", str, "grazing", "What may move to the oblique views: reserve | small (+ the side/crown/sole panels) | "
              "grazing (+ front/back surface facing its panel below steal_cos)", choices=("reserve", "small", "grazing"), advanced=True),
        Param("steal_cos", float, 0.5, "grazing scope: front/back triangles facing their panel below this cosine may move", minimum=0.0, maximum=1.0, advanced=True),
        Param("head_height", float, 0.32, "The head band the head views frame and may take: metres below the crown", minimum=0.05, advanced=True),
        Param("protect_close", int, 9, "Sheets: pinholes in the protection narrower than this (px at 4096) are closed for klein's mask", minimum=0, advanced=True),
        Param("prompt_sheet", str, DEFAULT_PROMPT_SHEET, "Sheet prompt; {caption} is the subject description, the layout clause is appended per sheet"),
        Param("text_encoder", str, DEFAULT_TEXT_ENCODER, "The Qwen3-4B text encoder repo (Qwen's fp8 release; klein's own is the same weights in bf16)", advanced=True),
        Param("face_policy", str, "protect_cap", "What klein never repaints: protect_cap (the projected photograph's footprint), "
              "protect_head (the face band), none (klein repaints the face too)", choices=("protect_cap", "protect_head", "none")),
        Param("strength", float, 0.8, "img2img strength: 0.8 is the working point, below 0.65 nothing sharpens, 1.0 ignores the render",
              minimum=0.0, maximum=1.0),
        Param("steps", int, 12, "Denoising steps", minimum=1),
        Param("guidance", float, 1.0, "Guidance scale (klein is distilled; 1.0)", advanced=True),
        Param("seed", int, 0, "Noise seed, fixed per view for a reproducible run"),
        Param("power", float, 4.0, "Facing exponent of a view's claim weight (cos^power x texel density)"),
        Param("gain", float, 1.5, "A view claims a texel when its weight beats gain x the best so far"),
        Param("dilate", int, 8, "Repaint ring (px) around the claimed pixels", minimum=0),
        Param("feather", int, 12, "Blend width (px) of the repaint into the kept render", minimum=0),
        Param("erode", int, 4, "Erosion (px) of the view's mask in the back-projection", minimum=0),
        Param("body_size", list, [704, 1408], "Width, height of the body views (the pipeline caps init images at 1 MP)"),
        Param("head_size", int, 1024, "Side of the square head views", minimum=256),
        Param("body_azimuths", list, [0, 180, 90, 270, 45, 135, 225, 315], "Body view azimuths (degrees) in loop order"),
        Param("head_azimuths", list, [0, 45, 315], "Head view azimuths (degrees) in loop order; no rear head views"),
        Param("head_first", bool, True, "Head views before the body views (one coherent face pass)"),
        Param("min_repaint", int, 500, "Skip a view claiming fewer pixels than this", minimum=0, advanced=True),
        Param("ref_max_pixels", int, 300000, "Reference panels are scaled to at most this many pixels each", minimum=16384),
        Param("prompt_body", str, DEFAULT_PROMPT_BODY, "Body view prompt; {caption} is the subject description"),
        Param("prompt_head", str, DEFAULT_PROMPT_HEAD, "Head view prompt; {caption} is the subject description"),
        Param("repo", str, DEFAULT_REPO, "The klein diffusers repo (text encoder, tokenizer, VAE, transformer config)", advanced=True),
        Param("fp8_repo", str, DEFAULT_FP8_REPO, "The fp8 transformer's repo", advanced=True),
        Param("fp8_file", str, DEFAULT_FP8_FILE, "The fp8 transformer's file", advanced=True),
        Param("text_encoder_device", str, "auto", "cuda | cpu | auto (cuda with 9 GB free)", advanced=True),
        Param("debug_dir", str, "", "Keep every view's render / repaint mask / repaint (PNGs) under this directory"),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        trainer = params["trainer_path"]
        if params["mode"] == "views" and shutil.which(trainer) is None and not Path(trainer).is_file():
            raise RuntimeError(f"refine_texture: trainer binary {trainer!r} not found on PATH")
        atlas = Path(str(inputs["mesh_dir"]))
        if not (atlas / "mesh_uv.obj").is_file() or not (atlas / "texture.png").is_file():
            raise FileNotFoundError(f"refine_texture: {atlas} is not a meshify atlas (mesh_uv.obj + texture.png)")
        stats_in = inputs.get("mesh_stats") or {}
        front = inputs.get("front_image")
        if front is None:
            raise ValueError("refine_texture: front_image is required (the photograph's front panel is klein's reference)")
        back = inputs.get("back_image")
        caption = str(inputs.get("caption") or "the person").strip()
        device = str(params["device"])
        out = Path(params["output_dir"])
        # The per-view working files (renders with their float maps, the working texture) live
        # here for the loop and go at the end; the PNGs worth a look are copied to debug_dir.
        views_dir = out / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        debug = Path(params["debug_dir"]) if params["debug_dir"] else None
        t0 = time.time()

        def run(cmd: List[str], name: str) -> None:
            try:
                stream_command(cmd, log_name=f"refine_texture.{name}", throttle=True)
            except ProcessFailed as exc:
                raise RuntimeError(f"refine_texture: `b2ctrain {name}` failed: {exc}") from exc

        # -- the protection: +inf best weight where klein must not paint ------
        mask_png = cv2.imread(str(atlas / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask_png is None:
            raise FileNotFoundError(f"refine_texture: {atlas}/mask.png missing")
        res = mask_png.shape[0]
        best = np.zeros((res, res), np.float32)
        policy = params["face_policy"]
        protect_file = {"protect_cap": "protect_cap.png", "protect_head": "protect_head.png"}.get(policy)
        protected = 0
        if protect_file:
            prot = cv2.imread(str(atlas / protect_file), cv2.IMREAD_GRAYSCALE)
            if prot is None:
                logger.warning("refine_texture: %s has no %s (no cap?); nothing is protected", atlas, protect_file)
            else:
                best[prot > 127] = np.inf
                protected = int((prot > 127).sum())
        extra_protect = inputs.get("protect_path")
        params = dict(params, _protect_path=str(extra_protect) if extra_protect else "")
        if extra_protect:
            prot = cv2.imread(str(extra_protect), cv2.IMREAD_GRAYSCALE)
            if prot is None or prot.shape != (res, res):
                raise FileNotFoundError(f"refine_texture: protect_path {extra_protect} is not an {res} x {res} mask")
            best[prot > 127] = np.inf
            protected = int(np.isinf(best).sum())
        best_path = out / "best.f32"
        best.tofile(best_path)
        texture = out / "texture_cur.png"
        start = Path(str(inputs["texture_path"])) if inputs.get("texture_path") else atlas / "texture.png"
        if not start.is_file():
            raise FileNotFoundError(f"refine_texture: texture_path {start} does not exist")
        shutil.copy(start, texture)

        if params["mode"] == "sheets":
            return self._run_sheets(atlas, out, start, best, protected, policy, front, back, caption, params, debug, t0)

        # -- cameras -------------------------------------------------------------
        bw, bh = int(params["body_size"][0]), int(params["body_size"][1])
        body_json, head_json = out / "cams_body.json", out / "cams_head.json"
        azims = lambda xs: ",".join(str(int(a)) for a in xs)  # noqa: E731
        run([trainer, "mesh-render", "--make-cams", "--atlas", str(atlas), "--output", str(body_json), "--width", str(bw), "--height", str(bh),
             "--azims", azims(params["body_azimuths"]), "--elevs", "0"], "make-cams-body")
        head_cmd = [trainer, "mesh-render", "--make-cams", "--atlas", str(atlas), "--output", str(head_json), "--width", str(params["head_size"]),
                    "--height", str(params["head_size"]), "--azims", azims(params["head_azimuths"]), "--elevs", "0", "--head"]
        centre = stats_in.get("head_centre")
        if centre:
            head_cmd += ["--centre", ",".join(f"{float(v):.6f}" for v in centre)]
        run(head_cmd, "make-cams-head")
        cams = {"body": json.loads(body_json.read_text()), "head": json.loads(head_json.read_text())}
        order = view_order(params["body_azimuths"], params["head_azimuths"], params["head_first"])
        views = []
        for kind, az in order:
            entry = next((c for c in cams[kind]["cameras"] if c["name"].endswith(f"_a{int(az) % 360:03d}.png")), None)
            if entry is None:
                raise RuntimeError(f"refine_texture: no {kind} camera at azimuth {az}")
            views.append((kind, cams[kind]["width"], cams[kind]["height"], entry))

        # -- klein ----------------------------------------------------------------
        klein = Klein(params["repo"], params["fp8_repo"], params["fp8_file"], params["text_encoder_device"])
        refs = [fit_reference(cv2.cvtColor(np.asarray(front), cv2.COLOR_BGR2RGB), params["ref_max_pixels"])]
        if back is not None:
            refs.append(fit_reference(cv2.cvtColor(np.asarray(back), cv2.COLOR_BGR2RGB), params["ref_max_pixels"]))
        klein.set_references(refs)
        prompts = {"body": params["prompt_body"].replace("{caption}", caption), "head": params["prompt_head"].replace("{caption}", caption)}
        for prompt in prompts.values():
            klein.encode(prompt)

        # -- the loop ---------------------------------------------------------------
        log: List[Dict[str, Any]] = []
        for i, (kind, w, h, entry) in enumerate(views):
            stem = Path(entry["name"]).stem
            it = views_dir / f"{i:02d}_{stem}"
            render_dir, refined_dir, bake_mask_dir = it / "render", it / "refined", it / "bake_mask"
            for d in (render_dir, refined_dir, bake_mask_dir):
                d.mkdir(parents=True, exist_ok=True)
            cam_json = it / "cam.json"
            cam_json.write_text(json.dumps({"width": w, "height": h, "cameras": [entry]}))
            run([trainer, "mesh-render", "--atlas", str(atlas), "--cameras", str(cam_json), "--output", str(render_dir), "--texture", str(texture),
                 "--depth", "--aux", str(best_path), "--device", device], f"render-{i}")
            rgba = cv2.imread(str(render_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            alpha = rgba[..., 3]
            cos = read_f32(render_dir / f"{stem}.cos.f32", (h, w))
            dens = read_f32(render_dir / f"{stem}.dens.f32", (h, w))
            prev = read_f32(render_dir / f"{stem}.aux.f32", (h, w))
            claim, painted = repaint_mask(alpha, cos, dens, prev, params["power"], params["gain"], params["dilate"])
            cv2.imwrite(str(bake_mask_dir / f"{stem}.mask.png"), claim.astype(np.uint8) * 255)
            entry_log = {"view": entry["name"], "kind": kind, "claimed_px": int(claim.sum()), "subject_px": int((alpha > 127).sum())}
            logger.info("refine_texture [%d/%d] %s: repaint %.0f%% of the subject", i + 1, len(views), entry["name"],
                        100.0 * claim.sum() / max((alpha > 127).sum(), 1))
            mask_u8 = painted.astype(np.uint8) * 255
            if debug is not None:
                d = debug / it.name
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(render_dir / f"{stem}.png", d / "render.png")
                cv2.imwrite(str(d / "repaint_mask.png"), mask_u8)
            if claim.sum() < params["min_repaint"]:
                entry_log["skipped"] = True
                log.append(entry_log)
                shutil.rmtree(it, ignore_errors=True)
                continue
            render_rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
            t1 = time.time()
            repainted = klein.repaint(render_rgb, mask_u8, prompts[kind], params["strength"], params["steps"], params["guidance"], params["seed"])
            composed = feather_repaint(repainted, render_rgb, mask_u8, alpha, params["feather"])
            refined = np.dstack([cv2.cvtColor(composed, cv2.COLOR_RGB2BGR), alpha])
            cv2.imwrite(str(refined_dir / entry["name"]), refined)
            run([trainer, "mesh-backproject", "--atlas", str(atlas), "--cameras", str(cam_json), "--images", str(refined_dir), "--renders", str(render_dir),
                 "--output", str(texture), "--texture", str(texture), "--power", str(params["power"]), "--mask-dir", str(bake_mask_dir),
                 "--best", str(best_path), "--erode", str(params["erode"]), "--device", device], f"backproject-{i}")
            diff = float(np.abs(composed.astype(np.float32) - render_rgb.astype(np.float32))[alpha > 127].mean())
            entry_log.update({"klein_seconds": round(time.time() - t1, 1), "mean_change": round(diff, 2)})
            log.append(entry_log)
            if debug is not None:
                shutil.copy(refined_dir / entry["name"], debug / it.name / "refined.png")
            shutil.rmtree(it, ignore_errors=True)

        final = out / "texture_final.png"
        shutil.move(str(texture), str(final))
        shutil.rmtree(views_dir, ignore_errors=True)
        for leftover in (best_path, body_json, head_json):
            leftover.unlink(missing_ok=True)
        mesh_obj = out / "mesh_klein.obj"
        obj_text = (atlas / "mesh_uv.obj").read_text().replace("mtllib mesh_uv.mtl", "mtllib mesh_klein.mtl", 1)
        mesh_obj.write_text(obj_text)
        (out / "mesh_klein.mtl").write_text("newmtl tex\nKd 1 1 1\nmap_Kd texture_final.png\n")
        stats = {"face_policy": policy, "protected_texels": protected, "start_texture": str(start), "views": log, "strength": params["strength"], "steps": params["steps"],
                 "seconds": round(time.time() - t0, 1), "caption": caption, "references": len(refs)}
        (out / "refine_texture.json").write_text(json.dumps(stats, indent=1))
        logger.info("refine_texture: %s in %.0fs (%d views, %d repainted)", final, stats["seconds"], len(views), sum(1 for v in log if not v.get("skipped")))
        return {"texture_path": str(final), "mesh_path": str(mesh_obj), "refine_texture_stats": stats}

    def _run_sheets(self, atlas: Path, out: Path, start: Path, best: np.ndarray, protected: int, policy: str, front: Any, back: Any,
                    caption: str, params: Dict[str, Any], debug: Optional[Path], t0: float) -> Dict[str, Any]:
        """The character-sheet mode: layout, one klein call per sheet, the chained transfer back."""
        import cv2

        from ..view_atlas import GREY, apply_sheet, build_layout, read_obj

        device = f"cuda:{int(params['device'])}"
        protect_extra = [Path(str(p)) for p in ([params.get("_protect_path")] if params.get("_protect_path") else [])]
        sheets_dir = out / "sheets"
        t1 = time.time()
        manifest = build_layout(atlas, sheets_dir, texture=start, res=int(params["sheet_res"]), extra=int(params["extra_panels"]),
                                head=int(params["head_panels"]), scope=params["extra_scope"], steal_cos=float(params["steal_cos"]),
                                head_height=float(params["head_height"]), protect=protect_extra, device=device)
        layout_seconds = time.time() - t1
        logger.info("refine_texture: %d sheets laid out in %.0fs", len(manifest["sheets"]), layout_seconds)
        # The protection, in each sheet's coordinates: the face policy's mask and everything protect_path named.
        protect_names = [n for n in (({"protect_cap": "protect_cap.png", "protect_head": "protect_head.png"}.get(policy),) +
                                     tuple(p.name for p in protect_extra)) if n]
        side = int(params["sheet_input"])
        klein = Klein(params["repo"], params["fp8_repo"], params["fp8_file"], params["text_encoder_device"], params["text_encoder"], side * side)
        refs = [fit_reference(cv2.cvtColor(np.asarray(front), cv2.COLOR_BGR2RGB), params["ref_max_pixels"])]
        if back is not None:
            refs.append(fit_reference(cv2.cvtColor(np.asarray(back), cv2.COLOR_BGR2RGB), params["ref_max_pixels"]))
        klein.set_references(refs)
        base_prompt = params["prompt_sheet"].replace("{caption}", caption)
        log: List[Dict[str, Any]] = []
        edited_sheets: List[Tuple[Path, np.ndarray, np.ndarray]] = []
        for sheet in manifest["sheets"]:
            sdir = Path(sheet["dir"])
            kind = sheet["kind"]
            W, H = sheet["width"], sheet["height"]
            diffusion = cv2.imread(str(sdir / "diffusion_texture.png"))
            context = cv2.imread(str(sdir / "context.png"))
            claim = cv2.imread(str(sdir / "edit_mask.png"), cv2.IMREAD_GRAYSCALE)
            protect = np.zeros((H, W), np.uint8)
            for name in protect_names:
                m = cv2.imread(str(sdir / name), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    protect = np.maximum(protect, m)
            # Pinholes in the protection (the photograph's visibility test flickers per texel at grazing angles)
            # are closed for klein's mask only: they stay unpainted, their protected neighbours come back exact,
            # so klein never paints a speckle of contrast into them.
            hole = max(3, int(params["protect_close"] * W / 4096) // 2 * 2 + 1)
            protect = cv2.morphologyEx(protect, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole, hole)))
            # klein's mask is the whole figure (a broad, coherent repaint), less the protection; the sparse
            # ownership only decides what comes back.
            subject = (np.max(np.abs(context.astype(np.int16) - GREY), axis=2) > 5).astype(np.uint8) * 255
            if kind == "main":
                subject[int(.75 * H):, int(.75 * W):] = 0
            paint = cv2.dilate(subject, np.ones((17, 17), np.uint8))
            paint[protect > 127] = 0
            scale = min(1.0, side / float(max(W, H)))
            size = (max(16, int(W * scale) // 16 * 16), max(16, int(H * scale) // 16 * 16))
            inp = cv2.resize(diffusion, size, interpolation=cv2.INTER_AREA)
            m = cv2.resize(paint, size, interpolation=cv2.INTER_NEAREST)
            prompt = base_prompt + SHEET_LAYOUT_PROMPTS.get(kind, "")
            t2 = time.time()
            raw = klein.repaint(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB), m, prompt, params["strength"], params["steps"], params["guidance"], params["seed"])
            klein_seconds = time.time() - t2
            full = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_CUBIC)
            weight = np.clip(cv2.distanceTransform((claim > 0).astype(np.uint8), cv2.DIST_L2, 5) / 8.0, 0, 1)
            weight[protect > 127] = 0
            edited = np.clip(diffusion * (1 - weight[..., None]) + full * weight[..., None], 0, 255).astype(np.uint8)
            cv2.imwrite(str(sdir / "edited.png"), edited)
            change = float(np.abs(edited.astype(np.float32) - diffusion.astype(np.float32))[claim > 0].mean()) if (claim > 0).any() else 0.0
            log.append({"sheet": kind, "size": list(size), "klein_seconds": round(klein_seconds, 1), "mean_change": round(change, 2),
                        "panels": [dict(name=p["name"], triangles=p["triangles"]) for p in json.loads((sdir / "atlas.json").read_text())["panels"]]})
            logger.info("refine_texture [%s]: klein %.0fs at %dx%d, mean change %.1f/255", kind, klein_seconds, size[0], size[1], change)
            if debug is not None:
                d = debug / f"sheet_{kind}"
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / "input.png"), inp)
                cv2.imwrite(str(d / "mask.png"), m)
                cv2.imwrite(str(d / "raw.png"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
                shutil.copy(sdir / "edited.png", d / "edited.png")
                shutil.copy(sdir / "diffusion_texture.png", d / "sheet.png")
            edited_sheets.append((sdir, edited, diffusion))
        # -- back to the atlas, sheet after sheet ----------------------------------
        _, _, olduv = read_obj(atlas / "mesh_uv.obj")
        source_mask = cv2.imread(str(atlas / "mask.png"), cv2.IMREAD_GRAYSCALE)
        start_tex = cv2.imread(str(start))
        cur = start_tex.copy()
        transfer = []
        for sdir, edited, reference in edited_sheets:
            cur, changed, metrics = apply_sheet(sdir, edited, cur, reference, source_mask, olduv, device)
            transfer.append(dict(sheet=sdir.name if sdir != sheets_dir else "main", **metrics))
        keep = np.isinf(best)
        cur[keep] = start_tex[keep]
        final = out / "texture_final.png"
        cv2.imwrite(str(final), cur)
        mesh_obj = out / "mesh_klein.obj"
        mesh_obj.write_text((atlas / "mesh_uv.obj").read_text().replace("mtllib mesh_uv.mtl", "mtllib mesh_klein.mtl", 1))
        (out / "mesh_klein.mtl").write_text("newmtl tex\nKd 1 1 1\nmap_Kd texture_final.png\n")
        # The working sheets are large (three 4096 PNGs and their masks); the edited sheets stay for the eye
        # under debug_dir, the layout manifest for provenance.
        for sdir, _, _ in edited_sheets:
            for name in ("context.png", "texture.png", "source_uv.npy", "edited.png"):
                (sdir / name).unlink(missing_ok=True)
        stats = {"mode": "sheets", "face_policy": policy, "protected_texels": protected, "start_texture": str(start), "sheets": log, "transfer": transfer,
                 "layout": {k: v for k, v in manifest.items() if k != "stats"}, "layout_seconds": round(layout_seconds, 1), "strength": params["strength"],
                 "steps": params["steps"], "seconds": round(time.time() - t0, 1), "caption": caption, "references": len(refs), "text_encoder": params["text_encoder"]}
        (out / "refine_texture.json").write_text(json.dumps(stats, indent=1))
        logger.info("refine_texture: %s in %.0fs (%d sheets, %d texels changed)", final, stats["seconds"], len(log), sum(t["edited_texels"] for t in transfer))
        return {"texture_path": str(final), "mesh_path": str(mesh_obj), "refine_texture_stats": stats}
