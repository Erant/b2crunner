# Camera pose refinement — bundle-adjusting the exported orbit before brush

**Status: measured, and wired in as of 2026-08-31.** The measurement below
is an experiment run outside the pipeline, in `~/Projects/colmaptest`,
against one COLMAP dataset (`colmap_intermediate`, 81 helical frames, the
shape `colmap_export` writes with `layout: brush`). Refining the camera
poses with COLMAP before training is worth **+0.63 PSNR** on held-out
views.

It is now `pipeline/steps/refine_cameras.py`, running twice in each of the
two shipped workflows — see "Porting", at the bottom, for what the step
does differently from the scratch scripts and for what is still unverified.

Note the name clash before reading further: `steps/pose_refine.py` is
about *body* pose — re-posing a SAM-3D-Body fit so its mesh agrees with
the shell. This document is about *camera* pose, and the two share
nothing but the word.

## Why there is anything to refine

The poses `colmap_export` writes are the ones the dataset holds, and for
a helical run those are generated, not measured. In `colmap_intermediate`
they are an exactly ideal orbit:

    angular step   4.500 deg +- 0.039, summing to 360.00 over 81 frames
    camera height  y = 0.000000 for every frame, std exactly 0
    orbit radius   1.8036 +- 0.0158
    frames 1, 81   identical pose (the loop closes on itself)

Nothing observed the camera. That is the whole opportunity: bundle
adjustment spends its budget undoing the assumption, and the splat gets
measurably sharper. After refinement the height picks up a 1.4 cm std and
the angular step loosens to 4.504 +- 0.184 deg — BA is not moving the
orbit, it is admitting that the orbit was never perfect.

## Where it sits

In the scratch project it was a filter between two directories:

    colmap_export (layout: brush)
      -> [ camera refinement ]
      -> brush (train)

consuming and producing the same on-disk layout and rewriting `images.txt`
only, with `cameras.txt` and `points3D.txt` carried across byte-for-byte.

In the pipeline it is a filter on `dataset.cameras` instead — same
operation, no dataset directory involved, which is what makes trap 2 below
unreachable. It runs **twice**, because both of this pipeline's training
datasets carry generated poses and neither's corrections describe the
other's:

    denoise_pass1 -> foreground_masks -> normal_maps
      -> [ refine_cameras ]            <- stage 2
      -> stage1_view_splats -> ... -> train_splat

    rerender_splat -> denoise_pass2 -> upscale -> export_masks/normals
      -> [ refine_cameras_final ]      <- stage 6
      -> export_colmap -> train_final_splat

The stage-2 placement is the one with a reason beyond "before brush":
`stage1_view_splats` builds a Gaussian shell **on each frame's own camera
ray** and renders it from a camera derived from that pose, so refining
after it would leave the supporting views built on poses the training no
longer uses.

What is never in the solve, at either site, is a supporting view. The face
cap and the stage-1 shells are renders *made from* the poses being
corrected; they carry no independent evidence about where a camera was.

## What has to be rebuilt after the correction

Not being in the solve is not the same as being unaffected by it. The face
splat is built in the **bootstrap** — unprojected from the reference
photograph through the anchor camera's pose, at the mesh's depth — and its
cap is a render of that splat, so refining the anchor moves the camera out
from under a splat that already exists.

From 2026-08-31 to 2026-09-02 the answer was `rebase_cameras`: carry the
cap's cameras across by the anchor's own pose delta, `D = P_new @ P_old^-1`,
on the argument that the photograph's content moves with its camera and a
splat and its cameras moved by one rigid transform render identically.
Measured on run `fast_helical_native-20260902-143945-e593a8`, that was
wrong by 50 mm. Triangulating MediaPipe landmarks per source in the
exported `colmap_intermediate`, the denoised frames and the stage-1 shells
agreed with each other and with the mesh to within a centimetre, while the
carried cap sat 50 mm *behind* them along the anchor's viewing ray — the
same pixel in the anchor frame, a second face inside the head from the
side, a Janus in the helical re-render. The anchor's delta was 23 mm
toward the subject, 58 mm up and 1.6° of pitch; at the head the vertical
parts cancel and what remains is a slide along the ray, which is the one
direction a single frame's reprojection cannot constrain and so the one
BA is freest to leave there. The carry moved the splat's depth by that
slide, but the depth was never the camera's: it came from the mesh, which
does not move.

So the splat is **rebuilt**: `face_splat_refined` runs `face_pointmap_splat`
again after `refine_cameras`, from the same crop, matte and normals, with
`cameras: dataset.cameras` and `given_camera: scene.image_warp.camera?` —
the photograph's pixels on the photograph's camera moved by the anchor's
refinement delta (T = refined ∘ given⁻¹ applied to the identity pose), the
depth re-taken from the mesh through that pose. Not the refined camera's
own rays: the path's anchor camera is `look_at`-turned onto the orbit
target (the mesh bbox centre, 31 mm off the photograph's axis on cyber2 →
0.83 deg), and the photograph's rays hung on that rotation turn with it.
Run 5e2817 (2026-09-04, the first run after trap 4 was fixed) had the
cameras right and the cap still 17 px / 28 mm above the frames' face for
exactly that reason — the same size as the trap-4 pitch in 9cc643, which
is why the two were mistaken for one. The cap render
and its selection follow it there, and `render_splat` aims the cap along
the live anchor camera (`dataset.cameras[anchor_frame_index]`) rather than
the `anchor_position` the render recorded. That record is kept truthful
too: `refine_cameras` publishes the refined anchor's position
(`anchor_position`, from `anchor_frame_index`), and the first refinement
writes it over `dataset.extras.anchor_position`. `given_cameras` is no
longer published; nothing needs the old poses.

The stage-1 shells always had this wiring: they are built *after* the
refinement, from the poses it produced. The rule is about order, not about
the face, and `tests/test_workflows.py` asserts it that way — and asserts
that the splat a supporting view renders was itself built after the
refinement, through the refined poses.

## The result

brush, 30k iters, `--eval-split-every 8` (70 train / 11 eval), identical
`points3D.txt` init in every condition, three seeds each:

| condition                 | seed 7 | seed 42 | seed 1234 | mean PSNR    | SSIM   |
|---------------------------|--------|---------|-----------|--------------|--------|
| given poses (baseline)    | 33.885 | 33.879  | 33.884    | 33.883 ±.003 | 0.9759 |
| refined, foreground only  | 34.529 | 34.518  | 34.500    | 34.516 ±.012 | 0.9804 |
| refined, all features     | 34.633 | 34.596  | 34.586    | 34.605 ±.020 | 0.9808 |

+0.633 PSNR / +0.0045 SSIM for the foreground variant. Seed spread on the
baseline is ±0.003, so this is not seed noise; the refined runs are ahead
at every eval checkpoint from iteration 2500 onward and never cross back.

**Use the foreground-only variant.** It looks 0.089 PSNR worse than
letting features land on the backdrop too, but a fourth run — the same
foreground pipeline, a different RANSAC draw — came out at 34.591, moving
0.075 on nondeterminism alone. The two feature regions are not
distinguishable at this sample size. What *is* distinguishable is
refining at all, which is worth ten times the difference between them.

And the background is the half you cannot carry to another capture. Here
it is a static studio backdrop that happens to be rigid, and it supplies
only ~15% of the keypoints anyway (747 features per image unmasked, 633
masked — 85% were already on the subject). A backdrop that moves, that is
generated inconsistently per frame, or that is not there at all will
quietly poison an unmasked solve, and nothing in the output will say so.
Masking to alpha costs ~0.09 PSNR that is probably not real, and buys a
pipeline whose assumption holds on every dataset.

## The recipe

Foreground-masked ALIKED + LightGlue, triangulate against the given
poses, then alternate BA with retriangulation, then put the gauge back.

1.  **Alpha -> COLMAP masks.** Threshold alpha at 127, one PNG per frame
    into a scratch dir. COLMAP ignores black pixels
    (`--ImageReader.mask_path`; it accepts `<name>.png` as well as
    `<name>.<ext>.png`).

2.  **Input model from the given poses.** `cameras.txt` and `images.txt`
    as-is, plus an *empty* `points3D.txt`. The exported `images.txt`
    carries no 2D observations, which is exactly what
    `point_triangulator` wants.

3.  **Features and matches.** `ALIKED_N32` + `ALIKED_LIGHTGLUE`, both
    CPU. A white cyclorama and bare skin are low-texture and SIFT thins
    out on them; ALIKED holds up. Intrinsics come from `cameras.txt` via
    `--ImageReader.camera_params` with `single_camera 1`. Exhaustive
    matching at 81 images is ~3 min and gives ~465 inliers on adjacent
    pairs, 1365 verified pairs.

4.  **Triangulate, then 3x (bundle adjust -> retriangulate).** One BA
    pass strands observations; a fresh `point_triangulator` between
    passes lets tracks reform against the corrected poses. Converges by
    the third round. Every intrinsic frozen — see the trap below.

5.  **Put the gauge back.** Fit a similarity transform (Umeyama, with
    scale) from the refined camera centres onto the original ones and
    apply it to the refined poses. Non-negotiable; see the next section.

6.  **Write a dataset that differs only in poses.** `cameras.txt` and the
    original `points3D.txt` copied across untouched.

In the scratch project this is one command:

```
./refine_cameras.sh <colmap_export_dir> <out_dir>
```

`refine_cameras.sh` plus `align_sim3.py` (the step-5 fit, usable
standalone against any pair of `images.txt` files). Both in
`~/Projects/colmaptest`.

## Trap 1 — BA inflates the model by 15-26%, silently

The one that will cost a day if it is not known in advance.
`bundle_adjuster` fixes the gauge with `TWO_CAMS_FROM_WORLD`: the first
camera's pose, plus part of the second's. Global scale therefore hangs
off the single cam-1<->cam-2 baseline, and the rest of the reconstruction
is free to grow around it. Observed across four runs of this pipeline:

    +15.1%   all features
    +21.6%   foreground only
    +19.3%   foreground, script rerun
    +23.5%   foreground, script rerun

and three more from the step that replaced the script, on the same data:

    +21.7%   foreground, CPU
    +23.2%   foreground, CPU, rerun
    +26.4%   foreground, CUDA, inside the built image

The last one is outside the range the first four suggested, which is worth
knowing: this is not a quantity to bound from a handful of samples. What
matters is that it is removed exactly rather than kept small — the Sim(3)
brought that +26.4% back to 0.006% of radius drift like every other run.

Nothing warns you. The reconstruction stays self-consistent, reprojection
error looks fine, and `model_analyzer` is happy. What you get is a splat
a fifth too large, in a frame whose `points3D.txt` init no longer matches
it — and if anything downstream assumes the splat's scale (anchor
reinjection, the mesh/shell agreement `pose_refine` depends on, any
fixed-radius helical re-render) that is a silent, hard-to-attribute
corruption.

The fix is step 5. Fitting the similarity transform on camera centres
keeps every relative correction BA made and restores the global rotation,
translation and scale exactly. Afterwards:

    orbit radius   1.8036 (original) -> 1.8034 (refined+aligned)
    centroid       preserved
    round-trip     4e-10 max error rewriting images.txt

Checked on the trained splats too, not just the cameras: percentile spans
of the exported .ply scatter between -2% and +9% across p0.5/p1/p2/p5/p10
/p25, with no common factor. That is a distribution difference (floaters,
density), not a rescale — a real scale change moves every percentile by
the same ratio.

## Trap 2 — scratch files inside the output dataset get trained on

The first version of the script wrote its intermediates to
`$OUT/work/`. brush scans a dataset directory *recursively* for
`cameras.txt` and picks among all the models it finds
(`select_colmap_model`, most-registered-images wins, ties broken on
path). It found an intermediate — un-aligned, +19% scaled — and trained
on that.

Result: **PSNR 4.45**, exit code 0, no warning anywhere.

Keep scratch outside the dataset tree, and assert exactly one model in
the output before believing any number:

```
[ "$(find "$OUT" -name 'cameras.txt' -o -name 'cameras.bin' | wc -l)" -eq 1 ]
```

The same recursive-scan behaviour is why the generated foreground masks
must not be left in the output dataset either: a `masks/` sidecar flips
brush from `AlphaMode::Transparent` to `AlphaMode::Masked`, which changes
what the eval metric is measuring. Mirror the source layout exactly.

## Trap 3 — do not let intrinsics float

Run as a diagnostic, not as a variant. Unfreezing focal length on this
data drove:

    fx = 1105.48    fy = 837.93     (given: 1066.17 / 1066.17)

A 24% pixel-aspect distortion, which no real camera has, bought for a
0.017 px improvement in mean reprojection error. BA was absorbing
multi-view inconsistency into the camera model and warping geometry to
do it. Keep `refine_focal_length 0`, `refine_principal_point 0`,
`refine_extra_params 0`.

## Trap 4 — the ring's flat valley: every camera pitches, no centre moves

Found on run 9cc643 (2026-09-04), after the step had been in the pipeline
for four days. For a ring of inward-looking cameras, "the subject sits a
little lower" and "every camera pitches up by that angle over the radius"
are the same picture to first order, so BA sees a flat valley along that
direction and stops wherever noise leaves it. The Sim(3) of the recipe's
step 5 is fitted to camera *centres*, which a rotation of each camera
about its own centre does not move — so the valley coordinate passes
straight through it, and through every check above: radius exact, no
centre moved, reprojection error down.

Measured, refined against given, all 81 cameras:

    pitch  mean +0.94 deg  std 0.24  min +0.40  max +1.35   (19 px at f=1166.8)
    yaw    mean -0.04      std 1.05
    roll   mean -0.13      std 0.58
    reprojection 1.540 -> 1.487 px

In the given poses the denoised frames' face sat within 6 px of the mesh
and of the face cap; in the refined ones 19-21 px below both, in every
frame including the photograph's own. Everything the pipeline anchors to
the mesh after this step — the face cap rebuilt through the refined
anchor, the shells' depth ruler, the points3D init, `face_priority`'s
coverage — stayed with the mesh while the training pixels moved, and
brush trained two faces. The two Sep-2 runs carried the same mode at
+0.2 deg; the size is luck.

The given poses are the drawings' poses and the frames follow the
drawings on average, so the refinement's legitimate output is per-camera
jitter about them; anything common to every camera is gauge. The fix is
the missing half of step 5: after the Sim(3), take the chordal mean of
each camera's residual rotation in its own frame out of all of them
(`_remove_common_mode`), positions untouched. On 9cc643's cameras that
removes 0.946 deg and puts the frames' face 0.5-2.4 px from the cap in
every view — tighter than the given poses, since the per-camera
correction now shows. A uniform pitch goes exactly; a single camera's
correction keeps all but its 1/N share; a rigid motion never reaches it.
`max_common_mode_rotation_deg` (3) refuses a mean that says BA lost the
scene rather than drifted along the valley.

## What the ceiling is, and why

Reprojection error over the tracks:

|                        | given poses | refined |
|------------------------|-------------|---------|
| mean                   | 1.630 px    | 1.579 px |
| median                 | 1.485 px    | 1.416 px |
| observations under 1px | 31.0%       | 33.4%    |

It bottoms out near 1.4 px median and stays there however many BA rounds
you spend. A well-conditioned photographic capture sits nearer 0.5. That
gap is the frames disagreeing with each other — expected for generated
views, and the same thing showing up in trap 3, where BA reached for an
impossible camera the moment it was allowed to.

So this is bundle adjustment correcting a wrong *assumption* (the perfect
orbit) against partly unreliable *evidence*. +0.63 PSNR is about the
right size of prize for that. It is also why the recommendation is "do
this once, then move on" rather than "tune this" — the headroom above it
is in the frames, not the poses.

## Camera movement, for reference

After the Sim(3) alignment, i.e. genuine correction with the gauge
removed. Foreground variant:

    centre shift   mean 0.0148   median 0.0129   max 0.0691
                   (0.82% / 3.83% of the 1.804 orbit radius)
    rotation       mean 0.587 deg   max 1.911 deg
    components     radial 0.0036, vertical 0.0097, tangential 0.0086

The subject is ~1.6 units tall, so if a unit is a metre the cameras moved
~1.5 cm on average and 6.9 cm at most. The vertical component being the
largest is the expected shape of the correction: height was the axis the
generated orbit pinned hardest (std exactly 0).

## Checks worth keeping

Cheap, in the order they catch things:

- mean camera radius from the scene centroid, refined vs original —
  **within 0.1%** (catches trap 1; the step gates at 1%, see "Porting")
- count of `cameras.txt`/`cameras.bin` under the output — **exactly 1**
  (catches trap 2; not applicable to the step, which produces no dataset
  directory)
- mean centre shift as a fraction of scene radius — **under ~1%**; much
  beyond a few percent means BA re-solved the scene rather than refining
  it, so suspect the matches
- `model_analyzer` mean reprojection error, before vs after — must fall
- the common-mode rotation the step removed, and the anchor frame's
  residual rotation, both logged in degrees and px at the image centre —
  **a fraction of a degree, a few px** (catches trap 4; 19 px is what run
  9cc643 would have printed. None of the checks above can see it)
- the loader's reported train/eval split — **70 / 11** here

## Porting

Done, on 2026-08-31. `pipeline/steps/refine_cameras.py` is the recipe
above, driving the same four COLMAP binaries in the same order, and the
step's own docstring carries the three traps. What is worth knowing that
the scratch script does not say:

- **It is a filter on `List[Camera]`, not on a directory.** The frames,
  masks and input model are written into a `TemporaryDirectory`, COLMAP
  runs there, and the refined poses come back as `Camera` objects. Nothing
  a loader could scan is produced, so **trap 2 cannot happen** — which is
  also why the check that counted `cameras.txt` files did not survive the
  port. `work_dir` keeps the scratch for inspection and is likewise not a
  dataset directory.
- **Steps 5 and 6 happen in world coordinates.** `align_sim3.py` fits the
  Umeyama transform on COLMAP-convention poses and writes `images.txt`; the
  step reads the refined model back through the inverse of body2colmap's
  `world_to_colmap_camera` and fits in the pipeline's own frame. Same
  transform, one fewer round trip, and the intrinsics are carried across
  from the given cameras rather than re-parsed.
- **The checks are assertions, with `on_check_failure`.** The default,
  `keep_given`, logs the failure at ERROR and publishes the poses
  unchanged — the pipeline's behaviour before the step existed, which is
  the right end of a forty-minute pod run to fail towards. `raise` stops
  the run. Neither publishes poses that failed a check.
- **The scale gate is 1%, not the 0.1% observed above.** A correction that
  is not itself a similarity moves the mean radius by roughly the *square*
  of its size — a mean centre shift of 5% of the radius measures 0.14% —
  so a 0.1% gate would start refusing large-but-honest corrections under
  trap 1's name, while the centre-shift check is the one that actually
  describes them. Trap 1 is a 15-24% miss and clears 1% by a factor of
  fifteen. The observed drift is logged either way.
- **CUDA is on in the image, unlike the experiment's build.** The 4.2.0
  build measured here was `-DCUDA_ENABLED=OFF` because the workstation's
  nvcc rejects gcc 15. In `docker/Dockerfile`'s `colmap-builder` stage it
  is on (Ubuntu 24.04's gcc 13, which CUDA 13's nvcc accepts), which is
  what puts ALIKED and LightGlue on the ONNX **CUDA** execution provider
  instead of the CPU one — the provider is chosen from a compile-time flag,
  not a runtime one. That turns COLMAP's own `FETCH_ONNX` into a hazard:
  with CUDA enabled it fetches the **cuda12** ONNX Runtime, which will not
  load against this image's CUDA 13, so the build overrides it to the
  cuda13 build of the same release. A CPU COLMAP still works and
  `pipeline/doctor.py` WARNs about it.

  What that buys, measured on the built stage against this same 81-frame
  dataset, CPU against the CUDA provider on an RTX 4070 Ti:

  | stage                       | CPU    | CUDA   |       |
  |-----------------------------|--------|--------|-------|
  | ALIKED extraction           | 32.6 s | 2.8 s  | 11.6x |
  | exhaustive LightGlue match  | 176 s  | 24.3 s | 7.3x  |
  | features + matching         | 209 s  | 27 s   | 7.7x  |

  (The CPU column also confirms this document's own "~3 min" estimate for
  the matching: 2.941 min.) The bundle adjustment behind it is Ceres on the
  CPU in both columns and is not affected.
- **The ONNX weights are prefetched.** `pipeline/models.py`'s
  `colmap_onnx` entry pulls ALIKED-N32 and LightGlue (~65 MB) at pod start
  into `$B2C_MODELS_DIR/colmap/`, and the step passes explicit
  `--...model_path` flags. COLMAP's own cache is `$HOME/.cache/colmap`,
  i.e. the container's writable layer, re-downloaded after every restart.
- **Whether to refine is a workflow setting** (`refine_cameras`, on by
  default) rather than something detected. Nothing here detects it: the
  1.4 px floor means "reprojection error is high" does not distinguish bad
  poses from inconsistent frames.

### Verified, and not

The step was run end to end against **this document's own dataset**
(`~/Projects/colmaptest/colmap_intermediate`, 81 frames, the CPU 4.2.0
build), foreground-masked, three BA rounds, twice. It reproduces the
experiment, and the spread between its own two draws is the same
nondeterminism this document already reports:

| quantity                   | this document  | the step, two draws     |
|----------------------------|----------------|-------------------------|
| BA scale inflation         | +21.6%         | +21.7% / +23.2%         |
| radius drift after Sim(3)  | 0.011%         | 0.004% / 0.005%         |
| centre shift, mean         | 0.0148         | 0.0138 / 0.0148         |
| centre shift, median       | 0.0129         | 0.0117 / 0.0123         |
| centre shift, max          | 0.0691         | 0.0642 / 0.0666         |
| mean as a fraction of radius | 0.82%        | 0.76% / 0.82%           |
| rotation, mean (deg)       | 0.587          | 0.590 / 0.652           |
| reprojection error, px     | 1.630 -> 1.579 | 1.571 -> 1.532          |

Both draws passed every check. Against the scratch script's own
`refined_fg` output they disagree by a mean of 4.4 and 4.6 mm and a max of
9.5 and 9.8 mm (0.25% / 0.55% of the orbit radius) — the same solve, drawn
three times.

The **image is verified too**, as of the same day, and by the end of it the
whole chain had been run rather than reasoned about:

- the `colmap-builder` stage builds, and its own guard (`colmap version |
  grep -q "with CUDA"`) passes;
- ALIKED runs through the **CUDA execution provider** on a real GPU against
  these frames — 730-742 features per image against the ~747 this document
  measured on CPU. This is the part that could not be inferred: ONNX
  Runtime throws rather than falling back when a provider will not load, so
  a cuda12 runtime against this CUDA 13 image would have raised here, not
  run slowly;
- `doctor` reports the binary OK in the shipped image;
- and the **step itself ran end to end inside `b2c/pipeline:latest` on the
  GPU**, over all 81 frames: reprojection 1.570 -> 1.529 px, +26.4% of BA
  inflation removed to 0.006% of radius drift, mean centre shift 0.0137
  (0.76% of radius), every check passed, 4.9 mm mean from the scratch
  script's own output.

Two things the image build cost that no amount of reading would have found:
`openimageio-tools` (a *-dev* package's CMake config imports binaries that
ship in a different package, so `find_package` fails outright without it)
and `libglew2.2` in the **runtime** stage (`colmap` links libGLEW even with
GUI_ENABLED=OFF, and the builder stage cannot show it — it has libglew-dev,
so the binary works right up until it is copied somewhere that does not).
`doctor`'s colmap check is what caught the second, on the first build.

**Not verified:** the step inside a full pipeline run, the step on a pod,
and the +0.63 PSNR on the helical deliverable. Measured on one
dataset with one subject; the +0.63 should be re-measured before it is
assumed to hold for a different capture — the machinery to do it is three
seeds per condition at ~5 min a run on a 4070 Ti, which is cheap enough
that single-run comparisons are not worth trusting.
