"""refine_texture — sharpen a meshify atlas with FLUX.2 klein, view by view.

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

#: Everything the step loads from the bf16 repo: the text encoder, the
#: tokenizer, the VAE and the transformer's config (the weights come from
#: the fp8 file). `transformer/*.safetensors` is 8 GB nothing opens.
KLEIN_ALLOW_PATTERNS = ["model_index.json", "text_encoder/*", "tokenizer/*", "vae/*", "transformer/config.json", "scheduler/*"]

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

class Klein:
    """FLUX.2 klein 4B (fp8 transformer) as an inpainting img2img, resident."""

    def __init__(self, repo: str, fp8_repo: str, fp8_file: str, text_encoder_device: str) -> None:
        import torch
        from diffusers import AutoencoderKLFlux2, Flux2KleinInpaintPipeline

        from ..flux2_fp8 import load_flux2_fp8_transformer

        self.repo = repo
        self.text_encoder_device = text_encoder_device
        self._embeds: Dict[str, Any] = {}
        transformer = load_flux2_fp8_transformer(repo_id=fp8_repo, filename=fp8_file, config_repo=repo).eval().to("cuda")
        vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", dtype=torch.bfloat16).to("cuda").eval()
        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(repo, text_encoder=None, tokenizer=None, transformer=transformer,
                                                              vae=vae, dtype=torch.bfloat16)
        self._orig_prepare = self.pipe.prepare_image_latents
        self.extra_refs: List[Any] = []
        pipe = self.pipe

        def prepare_with_extras(images, *args, **kwargs):
            return self._orig_prepare(list(images) + self.extra_refs, *args, **kwargs)

        pipe.prepare_image_latents = prepare_with_extras
        logger.info("refine_texture: klein loaded, %.1f GB free", torch.cuda.mem_get_info()[0] / 2 ** 30)

    def encode(self, prompt: str):
        """Prompt embeddings, cached per prompt; the text encoder is loaded for the call and dropped."""
        if prompt in self._embeds:
            return self._embeds[prompt]
        import torch
        from diffusers import Flux2KleinPipeline
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        device = self.text_encoder_device
        if device == "auto":
            device = "cuda" if torch.cuda.mem_get_info()[0] / 2 ** 30 > 9.0 else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.repo, subfolder="tokenizer")
        encoder = Qwen3ForCausalLM.from_pretrained(self.repo, subfolder="text_encoder", dtype=torch.bfloat16).to(device).eval()
        with torch.no_grad():
            emb = Flux2KleinPipeline._get_qwen3_prompt_embeds(text_encoder=encoder, tokenizer=tokenizer, prompt=prompt, dtype=torch.bfloat16,
                                                              device=torch.device(device), max_sequence_length=512, hidden_states_layers=[9, 18, 27])
        del encoder
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
              "caption"?: str — the subject description the prompts are built from}
    outputs: {"texture_path": str, "mesh_path": str (mesh_klein.obj), "refine_texture_stats": dict}
    """

    PARAMS = (
        Param("trainer_path", str, "b2ctrain", "The b2ctrain binary (mesh-render / mesh-backproject)", advanced=True),
        Param("output_dir", str, help="Where the refined texture and the per-view renders go (the run's mesh/)"),
        Param("device", int, 0, "CUDA device index for the raster halves", advanced=True),
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
        if shutil.which(trainer) is None and not Path(trainer).is_file():
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
        best_path = out / "best.f32"
        best.tofile(best_path)
        texture = out / "texture_cur.png"
        shutil.copy(atlas / "texture.png", texture)

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
        stats = {"face_policy": policy, "protected_texels": protected, "views": log, "strength": params["strength"], "steps": params["steps"],
                 "seconds": round(time.time() - t0, 1), "caption": caption, "references": len(refs)}
        (out / "refine_texture.json").write_text(json.dumps(stats, indent=1))
        logger.info("refine_texture: %s in %.0fs (%d views, %d repainted)", final, stats["seconds"], len(views), sum(1 for v in log if not v.get("skipped")))
        return {"texture_path": str(final), "mesh_path": str(mesh_obj), "refine_texture_stats": stats}
