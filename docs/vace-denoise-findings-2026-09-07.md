# The VACE denoise: skeleton bleed, the sampler, and the anchor drift

Findings from 2026-09-07/08, written so the next pod run starts from them.
Three things, in the order they were found. Sections 1 and 3 are
diagnoses; section 2 is a change already in the working tree. Section 4
is the one pod run that settles the open question.

## 1. The skeleton surviving the denoise as ink

**What was seen.** In `fd852e`'s `colmap_intermediate` (the pass-1 output)
the DWPose skeleton is painted into the frames: red neck-to-shoulder
sticks over the hair on back views, blue/purple sticks down the legs.
DWPose limb colours in DWPose places. Pass 2 keeps them: they are baked
into the intermediate splat, rerendered, and denoised at 0.8.

**The invocation is faithful.** `pipeline/steps/wan22_vace_denoise.py`
was read against ComfyUI's `WanVaceToVideo` (comfy_extras/nodes_wan.py)
and the `WAN21_Vace` wrapper: mask polarity, the inactive/reactive split,
the reference latent with its zero half, the mask padding, the latent
normalisation and the trim all match. Not the cause.

**The bleed predates the port.** `cyber_6f/circular` — the ComfyUI output
the port was verified against, strength 1.0 on every step — has it too,
faintly: a magenta line down the jacket's spine and pink hip lines on
frame 41. A skeleton over a grey silhouette is not the black-background
pose map VACE learned, and the model keeps part of it as content.

**The lever the Downloads runs show.** The VACE scale on the low-noise
expert's FIRST step:

| run | pass-1 schedule | bleed |
|---|---|---|
| 9cc643 | high 1.0, low 0.5 on all four low steps | faint |
| fa59e5 | `[1, 1, 1, 0.5, 0.5, 0.5]` | heavy |
| 00ce5a, fd852e | `[1, 1, 0.8, 0.8, 0.8, 0.8]` | heavy |
| 5e2817 | `[1] * 6` | heavy |

fa59e5 differs from 9cc643 only at step 3, and the taper
`[1, 1, 0.75, 0.5, 0.25, 0]` also bleeds (0.75 at step 3). The bleed
turned visibly heavier on 2026-09-01, when the skeleton became the 7 px,
60%-dimmed DWPose style: more ink, darker against the grey ground.

**Two real invocation differences from the reference graph** remained:
the sampler schedule (section 2) and the reference image — diffusers
LETTERBOXES it on white (768x1536 -> 640x1280 with 40 px white bars each
side, `preprocess_conditions` in pipeline_wan_vace.py) where ComfyUI's
`common_upscale(..., "center")` crops to fill at 0.9375. Untested whether
either matters for the bleed.

## 2. The sampler now matches the ComfyUI graph (working tree, untested on a pod)

The reference graph (`ComfyUI-Body2COLMAP/workflows/api/denoise.json`)
sampled `uni_pc` on ComfyUI's `beta` scheduler with no ModelSamplingSD3
node, which leaves a Wan 2.2 model at ComfyUI's default shift of 8.0
(comfy/supported_models.py, `WAN21_T2V.sampling_settings`). The HF
repo's scheduler config is UniPC too, but spaced `linspace` at
`flow_shift: 3.0`, and that is what the step ran until 2026-09-07:

    ComfyUI   beta, shift 8      t = 1000, 988 | 955, 889, 753, 448
    diffusers linspace, shift 3  t = 1000, 938 | 857, 750, 601, 376

`wan22_vace_denoise` gained two per-call params: `sigma_schedule`
(default `beta`) and `sampler_shift` (default 8.0). `linspace` + 3.0 is
the old run. `_comfy_beta_sigmas` reproduces comfy/samplers.py's
`beta_scheduler` with scipy (venv_wan22 sees venv_base's copy);
`_install_sigma_schedule` wraps the scheduler's `set_timesteps` on the
instance because `pipe()` calls it with a step count and nothing else;
`_configure_sampler` writes `flow_shift` through `register_to_config`.
Verified against a real diffusers UniPC scheduler built from the pinned
config (0.41 checkout): beta at shift 8 steps through 999, 987, 954, 889,
753, 447. The pod's diffusers 0.40.0 has the `sigmas` kwarg; 0.36.0 does
not, and the wrapper refuses by signature rather than running linspace
while logging beta. The custom sigmas must be an ndarray (diffusers does
arithmetic on them; a list raises).

Consequence: at shift 8 the checkpoint's `boundary_ratio` 0.875 would put
FOUR of six steps on the high-noise expert (1000, 988, 955, 889 >= 875);
the graph split by count at step 2, and so does `steps_high`/`steps_low`.

**2026-09-08: the solver, the hand-off and the reference image are matched
too**, by four more per-call params, all defaulting to the graph's
behaviour, all with the pre-change behaviour still sayable:

| param | default (= the graph) | the run before |
|---|---|---|
| `solver_variant` | `bh1` | `bh2` |
| `solver_order` | 3 (a cap, see below) | 2 |
| `handoff_reset` | on | off |
| `reference_fit` | `crop` | `letterbox` |

- **Solver.** ComfyUI's `uni_pc` sampler is UniPC's bh1 variant with
  `order = min(3, len(timesteps) - 2)` PER SAMPLER — and each
  `KSamplerAdvanced` owns only its slice of the schedule plus the boundary
  sigma, so the graph's 2-step high-noise sampler ran at order 1 and its
  4-step low-noise one at 3 (`_phase_order`). Per step the graph took
  orders 1, 1 | 1, 2, 2, 1; diffusers left alone takes 1, 2 | 3, 3, 2, 1.
  Both measured on the real scheduler. Two traps found there: raising
  `solver_order` through the config resizes `model_outputs` on the next
  `set_timesteps` but not `timestep_list`, and `step()` then indexes past
  it (IndexError) — the step sizes both itself; and diffusers' bh1 is
  NON-FINITE at the final step because the terminal sigma is 0 and bh1's
  B(h) = h is infinite there. ComfyUI's `sample_unipc` puts 0.001 in
  place of the terminal 0, and so does the step for bh1
  (`COMFY_TERMINAL_SIGMA`); bh2 keeps diffusers' 0 so the old run stays
  byte-identical.
- **Hand-off.** The graph's second KSampler is a NEW sampler: the
  low-noise expert starts with an empty multistep history (its first step
  first-order, no corrector for the step before it), and the first
  sampler ENDS, so its last step is first-order too (`lower_order_final`
  counts the steps its own sampler has left). diffusers runs one loop and
  carries the history across, so its step 3 was corrected by x0
  predictions the HIGH-noise expert made at t=1000 and 988. Two pre-hooks
  reproduce the graph: `_phase_end_hook` on `transformer` drops the order
  to 1 for the last high-noise step; `_handoff_hook` on `transformer_2`,
  at its first forward of a pass, sets the low-noise sampler's order and
  clears `model_outputs`, `timestep_list`, `lower_order_nums`,
  `last_sample` (what `set_timesteps` initialises), keeping `_step_index`.
  Verified on the real scheduler: orders 1, 1, 1, 2, 2, 1 and the corrector
  skipped at step 3, finite throughout.
- **Reference image.** `_fit_reference` is comfy/utils.py's
  `common_upscale(..., "center")`: crop to the frame's aspect about the
  centre (Python `round`), resize to fill with bilinear. A 768x1536 back
  panel loses 85 rows top and bottom; cyber_6f's 1440x1280 sheet loses 360
  columns each side. Compared numerically against ComfyUI's own function:
  identical when no scaling is involved, 0.27 levels mean difference at
  the 768->720 resample. diffusers then finds a reference that already
  fits and pads nothing.

Tests: `tests/test_wan22_conditioning.py` (`TestSamplerSchedule`,
`TestSamplerReachesTheScheduler`, `TestSolverAndHandoff`,
`TestReferenceFit`, `TestDeclaration`), full suite 1154 passed. Two
harness traps recorded there: scipy.stats first imported inside
`patch.dict(sys.modules)` is evicted on exit and refuses a re-import, and
scipy's array-API shim needs `torch.Tensor` on any stubbed torch.

What still differs from the graph, and is left alone: the fp8-scaled
experts with a live LoRA vs the Q8 GGUF quant, the RNG stream, integer
timesteps and the 1e-6 nudge on the first sigma.

Not committed. No image. No pod run.

## 3. The anchor drift on the circular stage — the denoiser, not the warp

**The observation.** In every b2crunner `colmap_intermediate`, a feature
near the frame centre (the trousers button, the top's hem) moves up from
frame 1 to 2 to 3 and then levels off; frames ~65-81 descend gradually
into the anchor at 81. cyber_6f does not do this. Template tracking,
fd852e, button row:

| frame | 81 | 80 | 79 | 78 | 77 | 75 | 72 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| row | 608 | 607 | 605 | 603 | 602 | 600 | 598 | 608 | 606 | 604 | 604 | 603 | 602 |

Every run drifts 5-10 px: 5e2817 (flat strength 1.0), 9cc643 (low 0.5),
the 08-30/31 bikini runs (before the DWPose skeleton and the room
backdrop), fa59e5. `cyber_6f/circular`: a centre patch holds row 588 on
every frame, both sides.

**What is ruled out.** The first 14 steps of the workflow were re-run
locally on `datasets/isolated/00.png` with an extra `save_dataset`
before `warp_reference_to_anchor`, giving the mesh render at the anchor
camera itself (recipe below). At that camera the mesh silhouette's width
matches the warped photo's ROW BY ROW through the torso (rows 450-650
within 1-3 px); the head/feet gaps are hair (mesh scalp 7 px below the
hair top) and the platform soles (mesh feet 22 px above them). The
render is flat from frame to frame (head, feet and skeleton constant over
frames 1-5). Given cameras have uniform height and pitch; `cameras.txt`
carries one model for all frames. Default vs MoGe-2 focal changes the
mesh height by one pixel. So the warp is right, the mesh placement is
right, and the anchor frame is the one correct frame.

**Mechanism.** The denoise pass paints the body 5-10 px higher than both
the photograph and the drawing; the injected anchor frames (VACE mask 0)
pull their neighbours back toward the truth. The transition is
asymmetric because of the Wan VAE's causal chunking: frame 1 is a latent
on its own, so frames 2-5 are the next chunk and the step is sharp;
frames 78-81 share one chunk with the photo at 81, so the descent smears
over the last ten frames.

**Suspects, since the geometry is right.** Everything in the invocation
that differed from the graph is now matched in the working tree (section
2: schedule, shift, solver, hand-off, reference fit) but untested on a
pod. What remains different: the Q8 GGUF vs fp8-scaled quant (rounding
level), and b2crunner's own control content — the outline silhouette, the
face splat composited in, MHR-derived joints drawn as DWPose.

**Earlier stage-2/3 finding, still true but a separate thing**: the
helical re-render anchors at the world origin while the intermediate
splat was trained on `refine_cameras`' moved anchor camera (fd852e: 46 mm,
1.1 deg), so frame 38 of `colmap_preupscale` sits ~5 px low against its
neighbours and the alignment loop's flow peaks there. Fix options: anchor
the helix at the live refined camera (`dataset.cameras[anchor_frame_index]`,
as `_cap_axis` does) and re-warp the anchor image by the residual
rotation, or pin frame 0 in `refine_cameras`. Not implemented.

## 4. The pod run that settles section 3

Run the CURRENT `wan22_vace_denoise` step — the section-2 defaults, so
the invocation matches the graph in everything but the quant and the RNG
— on `cyber_6f/initial`: 81 frames at 720x1280, the anchor photo at
frames 1 and 81 with alpha 0, every render at alpha 255, `reference.png`
(the 1440x1280 whole sheet, which `reference_fit: crop` takes to the
centre 720 columns exactly as ComfyUI did), `prompt.txt`. That is exactly
the control video ComfyUI received for `cyber_6f/circular`.

- Track a centre feature (the jacket's zipper/belt at about x=416,
  y=588 in frame 1) through frames 1-10 and 72-81. In
  `cyber_6f/circular` it holds row 588 on every frame.
- Drift absent and the bleed like `cyber_6f/circular` frame 41's -> the
  invocation is exonerated; it is the control content. Then A/B the
  render on this pipeline's own subject: skeleton on black without the
  silhouette (the canonical pose map), then the face splat off.
- Drift present -> what is left is the quant. Flip the four section-2
  params back one at a time to see which one it was NOT.
- Either way, read the same run for section 1: compare its skeleton bleed
  against `cyber_6f/circular` frame 41.

A single denoise pass is ~9 minutes on the 5090.

## Local reproduction recipe (everything before the Wan denoise)

The first 14 steps run in ~90 s on the 4070 Ti inside
`b2c/pipeline:latest`; only the 14B denoise is out of reach here (29 GB
RAM, 12 GB VRAM). RMBG-2.0 over 81 frames OOMs at 12 GB.

    S=<scratch dir>; R=/home/tristan/Projects/b2crunner
    docker run --rm --gpus all --user $(id -u):$(id -g) \
      -v $HOME/.cache/huggingface:/data/hf_cache \
      -v $HOME/.cache/torch:/data/caches/torch \
      -v $S/data/models:/data/models -v $S/data/output:/data/output \
      -v $S/data/logs:/data/logs -v $S/data/tmp:/data/tmp -v $S/data/uploads:/data/uploads \
      -v $S/data/caches:/data/caches/nv -v $S/data/caches:/data/caches/xdg \
      -v $S/data/caches:/data/caches/triton -v $S/data/caches:/data/caches/mpl \
      -v $R/pipeline:/opt/b2c_runner/pipeline:ro \
      -v $R/docker/envs.docker.yaml:/opt/b2c_runner/pipeline/envs/envs.yaml:ro \
      -v /home/tristan/datasets/isolated:/data/in:ro -v $S:/data/probe \
      b2c/pipeline:latest /opt/venv_main/bin/python -m pipeline.cli run \
        /data/probe/anchor_probe2.yaml --reference-image /data/in/00.png \
        --prompt "$(cat /home/tristan/datasets/isolated/00.txt)" \
        --out /data/output/probe2 --run-name probe --no-wait-for-models

Two traps: mounting `pipeline/` over the image hides the image's env
registry, so `docker/envs.docker.yaml` must be mounted back over
`pipeline/envs/envs.yaml`; and the probe YAML is `fast_helical_native`
truncated after `reinject_anchor_initial` (a copy is at `docs/anchor_probe2.yaml`) with a `save_dataset` step
before the warp (`pre_inject/`) and one after (`initial/`), its
`settings` reduced to the globals those steps read and `outputs: []`.
`--param reconstruct_body.fov_estimator=` gives SAM-3D-Body's default
focal. The saved `pre_inject/frame_00001_.png` alpha is the mesh
coverage at the anchor camera; `initial/frame_00001_.png` is the warped
photo.
