# Spatial reinforcement — the black render + `mask_splat` pair

> **Superseded 2026-08-30.** The move this document was written to
> prepare has happened: the decision now lives in the rasteriser, where
> the per-Gaussian evidence is. `brush --export-evidence` writes it into
> the .ply and `brush-splat-render --confidence` gates on it, so
> `rerender_splat` sets `confidence: true` and `mask_splat_fringes` runs
> as `mode: passthrough` in all three workflows. **Sections 1-3 below
> describe the OLD pair**, which is still reachable (`mode: threshold`
> plus `confidence: false`) and still the only thing verified against a
> recorded run. What replaced it, and what changed downstream, is at the
> end under "What the confidence gate does instead".

Between training a splat and handing its re-render to the second denoise
pass, the pipeline throws away every pixel the splat is not confident
about and paints what is left onto pure black. Two steps do it:
`render_splat` (with `bg_color: [0, 0, 0]`) and `mask_splat`. Neither is
clever, and the decision they make together is a per-pixel, per-frame,
2-D one about a 3-D object — which is why this is written down. The
right home for it is almost certainly the splatting/rasterisation side,
where per-Gaussian opacity and coverage are still available; this
document is what has to be reproduced before that move.

## Where it sits

```
denoise_pass1 -> colmap_export -> brush (train, --export-evidence)
              -> rerender_splat   (render_splat, helical, 81 frames,
                                   confidence: true, cull 0.5 grey)
              -> mask_splat_fringes   (mode: passthrough)
              -> reinject_anchor
              -> denoise_pass2   (strength 0.8)
```

Identical in both workflow files (`fast_helical_native.yaml`,
`fast_helical_shell.yaml`); the shell file's copy is kept in sync with
native's by `tests/test_workflows.py`.

Before 2026-08-30 the middle two lines read `BLACK bg` and `filter_size
6, dilation 2`, and that is what sections 1-3 describe.

The stage is a native port of the ComfyUI `mask_splat.json` subgraph
(eight generic nodes — `ToBinaryMask`, `InvertMask`, `ImpactDilateMask`,
`AILab_ImageCombiner`, `BilateralFilterImage`, save), collapsed into one
step in `pipeline/steps/mask_splat.py`.

## What actually happens

### 1. Render on black

`render_splat` shells out to `brush-splat-render`, which writes

```
rgb = colour*alpha + bg*(1 - alpha)
```

With `bg = 0` the RGB it writes is **premultiplied by alpha**, and the
alpha channel comes back as the frame's per-pixel mask
(`dataset.masks`, foreground = 1). The step's `bg_color` default is
already `[0.0, 0.0, 0.0]`.

Black is not cosmetic. A splat render has soft, low-alpha fringes
wherever the Gaussians are uncertain — thin hair, silhouette edges,
anything under-observed. On a white or grey background those fringes are
*bright*, and the bilateral filter in step 2 smears them across the
silhouette before anything blacks them out, so a halo survives into the
frames `denoise_pass2` sees. On black, fringe and background are the same
colour and there is nothing to bleed. The recorded ComfyUI run rendered
this stage on 127 grey with alpha 0 in the background
(`cyber_6f/splatted`) and its masked output is black RGB at alpha 255
everywhere (`cyber_6f/masked_splatted`) — black matches where that
pipeline *ends up*, which is what matters; matching its intermediate grey
would reintroduce the same halo, just dimmer.

(The same premultiplication assumption is relied on, and enforced, by
`select_support_views` in `steps/anchor_stub.py`, which refuses a render
that is more than 8/255 bright where it is fully transparent. The `+splat`
compositing in `steps/render.py` relies on it too, but passes the
background itself, so it has nothing to refuse. Note that since 2026-08-31
`bg_color` is composited in Python by body2colmap rather than passed to the
binary, which always runs on black — the value a transparent pixel ends at
is unchanged, to within the 1/255 of one extra 8-bit round trip.)

### 2. `mask_splat`

Per frame, on the render's own alpha:

| | |
|---|---|
| **threshold** | `keep = alpha >= 1 - threshold/255`, i.e. `>= 239/255` at the default `threshold: 16`. Strictly greater-than, no rounding to 0-255 first. |
| **dilate** | grow `keep` with a plain `dilation x dilation` kernel of ones (2x2 at `dilation: 2`) — not `2*dilation+1`, not an ellipse. `dilation: 0` is valid and skipped. |
| **composite** | `rgb * keep` — everything outside the kept region becomes exactly 0. |
| **bilateral filter** | `cv2.bilateralFilter(d=filter_size, sigmaColor=0.5*255, sigmaSpace=100)` over the composited frame. |
| **masks out** | replaced with an all-1.0 batch. |

The threshold and dilation semantics were fitted against recorded output,
not read off the node source, and neither is guessable: rounding the mask
to 0-255 before comparing pushed the max error from 15 to 140, and a
`2*dilation+1` kernel pushed it to 200.

`filter_size`/`dilation` is 6/2 in all shipping workflows (`fast
helical`); the older `helical` and `tiered` pipelines used 12/4 and 4/0.

### 3. The output mask means something else entirely

`dataset.masks` goes into `mask_splat` as the splat's **per-pixel alpha**
and comes out as an all-1.0 batch, which downstream is the **per-frame**
VACE flag: 1.0 = "synthetic, denoise this frame", 0.0 = "a real
photograph, keep it". Two different kinds of mask share the field, and
this step is where the meaning changes. Consequences:

- The blacked-out region is *not* protected. `denoise_pass2` is told to
  regenerate every one of those frames in full, background included.
- `reinject_anchor` must run **after** `mask_splat`, never before. Before
  it, it would overwrite `dataset.masks` — the splat alpha — with its own
  all-1.0 batch, and every step above would silently become a no-op. The
  anchor frame itself is not this step's business: in the recorded run
  `frame_00038_` is the real photo verbatim at alpha 0, neither
  composited nor filtered.

## The end result

Measured on `cyber_6f/splatted/frame_00020_`:

- 27.2% of the frame carries non-zero splat alpha. Nothing in the frame
  ever reaches alpha 1.0 — the maximum is 0.996, so the 239/255 cutoff is
  close to the top of the actual range, and small changes to `threshold`
  move a lot of pixels.
- 23.5% survives the threshold; dilation grows that to 23.8%.
- So **~13.6% of everything the splat drew is discarded as fringe** (3.7%
  of the frame), and the rest of the frame — about three quarters — is
  hard black.
- The recorded ComfyUI frame is 24.9% non-black, the difference being the
  bilateral filter lifting boundary pixels a value or two off zero.

The port is verified against that recorded stage
(`tests/test_mask_splat.py`): the surviving pixel set agrees on ~99.85%
of pixels with every disagreement a near-black boundary value of 1-3, and
the filtered values differ by a mean of ~0.25/255 with a max of 15,
consistent with a difference in the bilateral filter's border handling.
Visually identical, not bit-exact.

What `denoise_pass2` therefore receives is a hard-edged subject cut out
of black, with no soft matte and no halo, plus a per-frame instruction to
regenerate the whole thing — so it re-invents the background from the
prompt rather than inheriting the splat's guesses about it. Those frames
then go through the upscale and into `train_final_splat`.

## Why it is crude

Worth naming, since fixing it is the point of writing this down:

- **The decision is 2-D and per-frame.** A Gaussian that is well
  observed from one direction and grazing from another is judged
  independently in each frame, so the cut wanders between neighbouring
  views of a smooth 81-frame orbit. Nothing enforces temporal or 3-D
  consistency.
- **Confidence is proxied by rendered alpha.** Accumulated opacity along
  a ray is not the same quantity as "the training views constrained
  this", and the rasteriser is the only thing that ever sees the real
  per-Gaussian evidence.
- **The threshold is brittle by construction.** Alpha tops out at 0.996;
  a cutoff at 0.937 sits inside the noise band rather than safely below
  it.
- **Dilation is a fudge for the threshold being too aggressive** — grow
  back by two pixels what a hard cut just removed, with no relation to
  where the error actually is.
- **The bilateral filter runs on already-composited black**, so it drags
  edge pixels toward the background it was meant to protect them from,
  and is the entire source of the port's residual disagreement with the
  recorded run.
- **The per-pixel alpha is destroyed on the way out**, overwritten by the
  per-frame VACE flag, so nothing downstream can reconsider the decision
  or use a soft version of it.

A fix in the splatting step would decide coverage once, in 3-D, from
per-Gaussian opacity and observation counts, and hand out a soft
confidence channel the denoiser could actually be conditioned on —
replacing the threshold, the dilation and the filter at the same time.

## What the confidence gate does instead

Every bullet in "Why it is crude" is addressed by moving the decision to
where the evidence is, and none of it by a better 2-D filter.

### 1. brush measures the evidence (`export_evidence`, on by default)

After the last training step, brush renders every training view once more
and accumulates, per Gaussian: how much of its rendered weight landed
inside the training masks, how much landed anywhere, how badly it
disagreed with the views, how many views supported it at all, and the
mean direction it was seen from. Those go into the exported `.ply` as
seven extra vertex properties — `ev_w_in`, `ev_w_all`, `ev_err`,
`ev_views`, `ev_dir_0..2` — which every other `.ply` reader ignores. It
costs seconds (~2 s for 100k splats across 81 views) and only the LOD-0
export carries it, which is this pipeline's case.

`evidence_prune_inmask` would drop under-supported splats from the export
itself rather than merely hiding them in one render. It is off, and stays
off until someone has looked at a real run: a pruned splat is gone from
the deliverable. `evidence_normal_weight` folds the normal-map residual
into the evidence residual and is untuned.

### 2. `brush-splat-render --confidence` gates on it

Per pixel it forms a confidence `C` from the evidence of the Gaussians
covering it, and a gate `g = smoothstep(gate_lo, gate_hi, C)` (0.45/0.65
by default). The output contract changes with it, and this is the part to
be careful about:

| | old (still, without `--confidence`) | with `--confidence` |
|---|---|---|
| RGB | `colour*a + background*(1-a)` — premultiplied over `--background` | `(colour*a + cull*(1-a))*g + cull*(1-g)` — over the **cull colour**, then blended toward it by the gate |
| alpha | accumulated splat opacity `a` | the **gate** `g` |
| `--background` | used | ignored; `--cull-color` is the background |

So a rejected pixel is the cull colour (0.5 grey by default), *not*
black, and a fully transparent pixel is not black either. Two
consequences that are wired into the workflows:

- **`mask_splat` must not run its threshold path on this.** Thresholding
  the alpha, dilating and bilateral-filtering would re-composite grey
  frames over black and smear the gate's soft edge. It runs as
  `mode: passthrough`, which keeps only the half that is still needed —
  replacing the per-pixel alpha with the per-frame all-1.0 VACE batch
  (section 3 above is unchanged, and so is the ordering it forces).
- **Keep confidence OFF for the face-view cap renders** that feed
  `select_support_views` in both bootstrap workflows. That step divides the
  colour back out by alpha and enforces premultiplied-over-black
  (`_check_premultiplied` in `steps/anchor_stub.py`), so it refuses a
  confidence render outright. `tests/test_workflows.py` catches the
  wiring before a run does. (Until 2026-08-31 `composite_splat_views` was
  the other step with this requirement; the compositing is
  `render`'s `...+splat` mode now, which passes its own background.)

Evidence source, in order: the `.ply`'s own `ev_*` block, then
`evidence_dataset` (measure it now — point it at
`export_colmap_intermediate`'s output, which is exactly what `train_splat`
saw, for a splat trained before any of this existed), then none — which
warns, trusts every splat, and degenerates the gate back to plain alpha.
With `mask_splat` in passthrough that last case is a stage that does
nothing at all, so both binaries have to come from a build past the
confidence work on `Erant/brush`'s `normal-map-supervision`.

### 3. What `denoise_pass2` sees now

A subject cut out of **0.5 grey** rather than black, with a soft
(≈ two-value) edge from the smoothstep rather than a bilateral-filtered
hard cut, and no halo at all: the render is composited over the same grey
it culls to, so partial coverage fades toward the cull colour instead of
toward a contrasting one. The per-frame VACE mask is unchanged (all 1.0
except the anchor), and `inject_anchor` is untouched — the anchor frame is
still `anchor.png` verbatim at alpha 0. **If the prompt or any negative
prompt mentions a black background, revisit it.**

### 4. Tuning

`confidence_sidecar: true` keeps each frame's raw per-pixel confidence as
`<stem>.conf.png` under the log dir (the render's own temp directory is
deleted on the way out, so they are copied). `conf_args` passes
`--conf-*` flags to the binary verbatim: raise `--conf-tau` if genuine
surface is being culled, lower it if disagreeing splats survive.
`--conf-angle-margin` (default 30°) governs how much of the helix's
elevation extremes is trusted — 15° chewed patches out of a jacket at the
top of the helix, 30° keeps the body and trims only the grazing fringe.
`--conf-facing` and `--evidence-normal-weight` exist, are untuned, and are
left off.

### 5. Known limits

- Evidence view counts use "≥ 1 pixel of in-mask mass at training
  resolution", so they scale with brush's `max_resolution`; the defaults
  were tuned at a 720-1080 px short side.
- Measured along the cyber2_6f helical path, the gated region is slightly
  *less* frame-to-frame stable than a raw alpha cut (kept-region IoU 0.81
  vs 0.85) because it culls more silhouette. What is no longer true is
  that the decision itself wanders per frame — it is one 3-D decision
  rendered from 81 places.
- None of this has been through the pipeline on a pod. The b2crunner side
  is covered at argv level (`tests/test_splat.py`,
  `tests/test_brush_evidence.py`, `tests/test_workflows.py`); the gating
  arithmetic is the renderer's and is verified in the brush repo
  (`docs/splat-confidence.md` there).

