# Revert list — when body2colmap switches to the brush renderer

**Status: DONE, 2026-08-31.** body2colmap `24fe424` ("Replace gsplat with
the brush splat renderer") landed on its main, and everything below has
been carried out. The file is kept as the record of what the detour was and
why, because the *reason* the b2crunner steps existed is not visible in the
diff that removed them.

What the revert actually did, against the checklist below:

- `CompositeSplatViewsStep` and `_composite_result` are gone.
  `_composite_pivot` and `_check_premultiplied` **stayed** — item 1 below
  is wrong about those two, because `select_support_views` (which did not
  exist when this file was written) shares both.
- `pipeline/steps/render.py` gained `splat_scene`/`splat_path` and
  `anchor_position` inputs, a `splat_max_angle_deg` param, and three
  `*+skeleton+splat` render modes. It renders the layer for the frames that
  survive the cull and passes it to the `render_composite` call it already
  made.
- The rasterisation moved to body2colmap's `SplatRenderer` as well — but
  not on the first pass, and the way it got there is the useful part of
  this record. It was held back because the library raised on any non-zero
  exit and kept nothing when a render died, so `steps/splat.py` carried a
  parallel `_rasterize`/`_run_render`/`_write_cameras_json` driving the
  same binary. body2colmap's `d0e3ada`, later the same day, took the exit
  code half away. What was left was one behaviour — a crashed run's temp
  directory is deleted before anyone can copy anything out of it — holding
  up ~250 lines of duplicate.

  **The fix went into body2colmap rather than being worked around here**,
  as three seams:

  - `on_fault`: called with a `RenderFault` while the run directory still
    exists, for a caller that wants to save a crash report. Fires for a
    lost-frame run *and* for one that wrote everything and then died.
  - `on_output`: each line as the binary writes it, so `proc.OutputRelay`
    (extracted from `stream_command` for this) can throttle it into the
    pipeline log. `render_many` drives the binary with `Popen` and a line
    loop now instead of `subprocess.run`.
  - `ply_path`: render an existing `.ply` rather than serializing the
    scene back out. A trained splat is hundreds of megabytes and this step
    usually follows the training that wrote it.

  `_rasterize` is now a wrapper over `render_many` holding the three
  things that are genuinely this project's — where a crash report goes,
  where the log goes, and what a frame is called (`frame_00001_.png`, not
  the renderer's `f00000.png`). `_run_render`, `_write_cameras_json`,
  `_missing_renders` and the local `_Confidence` mirror of
  `ConfidenceOptions` are gone.

  Two behaviour changes worth knowing, both body2colmap's call: the binary
  is now always invoked with `--background 0,0,0` and `bg_color` is
  composited in Python (so `--background` appears even in confidence mode,
  where it is ignored), and that adds one extra 8-bit round trip, worth at
  most 1/255 on a non-black background. `render_shell_views` is the only
  shipped render with one.

- `select_support_views` lost its `view_roles` input and the `role` /
  `max_angle_deg` params with it: they read the composite step's per-frame
  verdict, nothing else can produce one, and both workflows already had
  them switched off because the cap render draws that edge.
- The cull angle was passed **explicitly** as 60 in both workflows, so it
  did not silently inherit body2colmap's 45 — see the last section.
- `docker/Dockerfile`'s body2colmap pull-forward pin moved from `a3a498d`
  to `24fe424`, with an assert on `render_composite`'s `splat_layer`
  parameter so an older library fails the build rather than the run.

## What was decided, and why

body2colmap `c65a7f7` ("Composite a Gaussian-splat face onto the skeleton")
added a `splat` overlay layer: `skeleton+splat` puts a real Gaussian splat
of the subject's face on the skeleton rig, in place of the synthetic
landmarks of `skeleton+face`. That is exactly what this project's face
branch wants, and the geometry and the compositing rule are taken from it.
The cull angle was too, until this project widened it from the measured 45
to 60 — see below.

What could **not** be taken is the rasterisation. body2colmap renders the
overlay through `SplatRenderer`, which is gsplat, and this project
deliberately has no gsplat: `render_splat` shells out to the
`brush-splat-render` binary instead, and dropping gsplat is what removed
b2crunner's last need for a CUDA toolchain at runtime (see
`pyproject.toml`'s `splat` extra and `docs/docker-build-notes.md`'s
"gsplat: dropped"). Reintroducing it to use one render mode would undo
that, for a build-time cost measured in tens of minutes per image.

So the layer is rendered by a separate `render_splat` step and composited
by a b2crunner step, rather than inside body2colmap's `render_composite`.
That is a duplication of body2colmap's `Renderer._composite_splat`, and it
is temporary.

**body2colmap is expected to move to the brush renderer.** When it does,
this whole detour collapses into two render-mode strings and everything
below comes back out.

## The precondition (met)

Do not start until body2colmap can rasterise a splat overlay without
gsplat — i.e. `OrbitPipeline.attach_splat_overlay` /
`render_splat_layer` no longer import it. Check `body2colmap/splat_renderer.py`
and the `splat` extra in its `pyproject.toml`. Nothing here is worth doing
while the swap would still drag a CUDA toolchain into the image.

Note the second precondition, which is about b2crunner rather than
body2colmap: `pipeline/steps/render.py` does **not** use `OrbitPipeline`.
It builds cameras itself and calls `Renderer` directly (see that module's
docstring on why). So "use the render mode" means passing a `splat_layer`
into `Renderer.render_composite`, which is already that method's public
signature, and the overlay's anchoring has to come from somewhere. It does
not need `body2colmap.splat_anchor`: see below.

## What came out

Everything in this list is b2crunner-side. Nothing in body2colmap needs
reverting — this project only ever added to it.

1. **`pipeline/steps/anchor_stub.py` — `CompositeSplatViewsStep`**
   (registered `composite_splat_views`) and its three helpers
   `_composite_result` / `_composite_pivot` / `_check_premultiplied`.
   Delete. `Renderer.render_composite`'s own `splat_layer` path replaces
   it, `Renderer._composite_splat` replaces the blend, and
   `OrbitPipeline.splat_view_angle_deg` replaces `_composite_pivot`.
   `_check_premultiplied` has no successor and needs none: body2colmap's
   renderer emits straight alpha via `bg_color=None`, so the
   black-background requirement disappears with it.

2. **`pipeline/steps/render.py`** — gains a `splat_scene` / `splat_path`
   input and a `splat_max_angle_deg` param, renders the layer per camera,
   and passes it to the `render_composite` call it already makes. The
   `render_mode` param gains `outline+skeleton+splat`.

3. **`pipeline/workflows/fast_helical_native.yaml`** *and*
   **`fast_helical_shell.yaml`** — both carry the face branch, and in both
   the `render_face_views` and `composite_face` steps come out, with
   `render_initial_views` taking `splat_path: scene.face_splat_path` plus
   `render_mode: outline+skeleton+splat` instead. The `face_splat` global
   stays; it still gates the branch that builds the splat.

4. **`tests/test_face_splat.py`** — `TestCompositeSplatViews` comes out
   with the step. **Everything else in that file stays**: the crop
   bookkeeping, the intrinsics, and above all
   `TestCropIsANoOpOnTheRays`, which is the gate on the part of this that
   is not going anywhere.

5. **`tests/test_workflows.py`** — both `BOOTSTRAPS` entries lose
   `render_face_views` and `composite_face`.

## What stays, and is not part of this

The face branch itself is unaffected by any of it. `sapiens2_seg`,
`crop_to_box`, `FacePointmapSplatStep` and the base/specialization split in
`pipeline/steps/pointmap_splat.py` all produce a `.ply`, and the question
this file is about is only who draws it.

In particular, **`body2colmap.splat_anchor` stays unused even then**, and
that is not an oversight. It exists to carry a splat built in masktest's
own frame — its own fitted focal, its own centroid recentring — into the
mesh's world. `FacePointmapSplatStep` writes the splat in the mesh's world
to begin with: it discards the pointmap's focal and re-unprojects through
SAM-3D-Body's, which is the same correction `compute_anchor_transform`
applies, done at construction. The anchor transform for these files is the
identity, and `attach_splat_overlay` would have to be handed a
`splat_meta.json` saying so. Passing the already-anchored `SplatScene`
straight to the renderer is what item 2 above should do.

The measurement inherited from `c65a7f7` — "reads cleanly to 30, rim
flares by 45, mostly edge by 60" — carries across unchanged; it is a
property of a 2.5-D shell, not of a rasteriser. **The cull angle taken from
it does not.** b2crunner runs `composite_splat_views` at 60, the far end of
that band rather than its clean limit: out there the alternative is a
drawing with no face at all, and a composited frame is an input to two
denoise passes that can rewrite a flared rim. A revert that adopts
body2colmap's render mode inherits its 45 unless `splat_max_angle_deg` is
passed 60 explicitly, and that is a behaviour change to make on purpose
rather than by omission.

`select_support_views` is unaffected by that choice and stays behind after
the revert either way — it consumes a render of its own
(`render_face_support_views`, `pattern: cap`), not the composite. That cap
is sampled at 30 degrees, the measurement's clean limit, because nothing
rewrites a supporting view between the render and the geometry brush
fits.
