# The quality sweep — six settings sidecars, all on pass 2

Section 6 of `docs/vace-denoise-findings-2026-09-07.md`, as per-image
settings sidecars, in the shape `docs/vace-leak-sweep/README.md` describes:
one reference sheet copied six times under these six stems, one `.yaml`
beside each, zipped flat, **no `.txt` in the zip**, the Subject box holding
the prompt, the param panel EMPTY.

    for f in docs/quality-sweep/F*.yaml; do
      cp <your-sheet>.png "$(dirname "$f")/$(basename "$f" .yaml).png"
    done
    cd docs/quality-sweep && zip ../../quality-sweep.zip F*.png F*.yaml

Use the SAME sheet as the 2026-09-08 sweeps: every number these are read
against is that subject at `seed: 0`.

## What is in it

Base = E4 (run 02ff74): the clean pass 1 the leak sweep recommended, with
the pass 2 every A-family run carried (euler / shift 2.5 / flat 0.8). Each
file's header says what it changes and what the answer looks like.

| # | pass 1 | pass 2 | what it answers |
|---|---|---|---|
| F1 | E4 | `strength: [1]*6` | does holding the control tighter help the final fit? |
| F2 | E4 | `strength: [.6]*6` | ... or does letting go? (0.6 / 0.8 / 1.0 is the curve) |
| F3 | E4 | `sampler_shift: 5` | the leak sweep's strongest lever, on pass 2 |
| F4 | E4 | `sampler_shift: 8` | the step default; B family's pass 2, single-variable |
| F5 | E4 | `sampler_high: uni_pc` | the sampler that hugs the control (E3's), where there is no skeleton to leak |
| F6 | E4 | `steps_high: 3`, `steps_low: 3` at `sampler_shift: 5` | the split that puts each expert on its trained range; reads against F3 |

The workflow's shipped pass 2 (uni_pc high, shift 8, `[0.8, 0.8, 0.6, 0.4,
0.2, 0]`) is not in the batch: those values were never chosen against a
measurement, and a slot spent confirming that is a slot not spent on an
axis. Sampler, shift, strength and the expert split on pass 2 are each covered
instead.

F6 is two changes from E4 and one from F3. It has to be: the high-noise
expert was trained at t >= 875, and at shift 2.5 a third step sits at
t = 868. Shift 5 puts three steps at 1000 / 980 / 929 and three at 833 /
655 / 334, so 3 | 3 there is the split that matches the checkpoint's
boundary — and 2 | 4 at shift 5 (F3) or at the graph's shift 8 hands the
low expert one or two steps above it.

## Reading the results

    scripts/skeleton_leak.py --iou --by-hue <result-dir> ...     # pass 1: all six must equal E4 (-0.53)
    scripts/final_splat_quality.py <result-dir> ...              # the deliverable, and pass 2 against its control

The second needs a local b2ctrain build (`~/Projects/b2ctrain/build/b2ctrain`
or `$B2CTRAIN`) and about half a minute per run. Its docstring says what
each column is. The baselines, all measured 2026-09-08 with it:

| run | frame head s1 | **splat head s1** | splat s1 | **PSNR** | p2 head s1 | ctl head s1 | p2 s1 | ctl s1 | p2 PSNR vs ctl | **flow** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E4 02ff74 (base) | 39.3 | 29.5 | 27.1 | 25.75 | 34.8 | 36.0 | 31.2 | 28.1 | 21.1 | 1.189 |
| E1 f1a7b3 | 42.7 | 32.1 | 27.9 | 25.14 | 37.1 | 35.7 | 28.7 | 25.0 | 21.4 | 1.154 |
| E2 df16ea | 41.4 | 30.7 | 27.3 | 25.61 | 38.1 | 36.6 | 30.2 | 26.3 | 21.2 | 1.165 |
| A 31a008 | 37.9 | 29.3 | 27.4 | 26.55 | 38.7 | 31.5 | 30.2 | 22.8 | 21.5 | 0.986 |
| B 467c17 | 37.5 | 29.3 | 31.1 | 26.67 | 37.4 | 29.1 | 32.5 | 19.5 | 20.5 | 1.039 |
| 0281b9 | 38.3 | 30.0 | 29.3 | 25.61 | 35.3 | 28.2 | 27.8 | 18.5 | 21.3 | 1.047 |
| ef13a7 | 39.4 | 27.9 | 30.9 | 25.17 | 52.0 | 30.0 | 51.9 | 22.4 | 16.4 | 1.376 |
| **F1 11ffc1** `[1]*6` | 37.4 | 28.7 | 26.5 | **26.29** | 35.2 | 37.9 | 28.7 | 28.6 | 21.9 | **1.007** |
| F2 0b6285 `[.6]*6` | 40.2 | 29.2 | 28.7 | 24.77 | 37.9 | 36.6 | 36.2 | 28.9 | 18.8 | 1.389 |
| F3 792cf0 shift 5 | 38.2 | 28.6 | 28.0 | 25.64 | 37.2 | 37.3 | 33.6 | 28.7 | 20.8 | 1.215 |
| F4 5d3064 shift 8 | 36.1 | 26.3 | 26.3 | 25.84 | 36.8 | 35.6 | 34.3 | 28.1 | 20.2 | 1.231 |
| F5 7b5a31 uni_pc | 36.2 | 26.7 | 26.2 | 25.14 | 38.6 | 38.4 | 34.3 | 28.5 | 19.9 | 1.290 |
| F6 778eeb 3/3 @ shift 5 | 38.4 | 28.6 | 27.5 | 25.17 | 34.5 | 39.2 | 29.1 | 28.4 | 21.0 | 1.206 |

Bold columns are the deliverable. A pass-2 setting wins when **splat head
s1 and PSNR rise together**; sharper frames with a falling PSNR and rising
flow is ef13a7 again — detail the fit cannot keep. Body `splat s1` on
B/0281b9/ef13a7 is inflated by skeleton ink and is not comparable to the
clean runs. A/B/0281b9/ef13a7 ran on an image whose `refine_cameras_final`
was refused (trap 5, fixed in `d83499c`), so their final cameras are the
given helix; compare F-runs to E1/E2/E4 first.

## Results (2026-09-08 night)

The F rows above are the batch, measured. Section 6 "Results" of
`docs/vace-denoise-findings-2026-09-07.md` reads them. Short form: pass 1
reproduced on every metric (leak -0.52..-0.59, IoU 0.845-0.847) but not
pixel for pixel (frames 26-28 dB apart at seed 0 — the denoiser amplifies
sub-level control differences into a fresh texture draw), so the seven
identical controls give the noise floor: head s1 sd 1.2, body sd 0.3.
No run raised head s1 and PSNR together. Strength is a monotone dial
(0.6 / 0.8 / 1.0 -> 24.77 / 25.75 / 26.29 dB, flow 1.389 / 1.189 / 1.007)
paid in frame texture; shift 8 and uni_pc each cost ~3 points of head s1;
3/3 buys nothing over F3. Recommended: F1, `strength: [1]*6`. Not applied.
