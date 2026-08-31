# Revert list — when body2colmap switches to the brush renderer

**Status: pending. Nothing here has been done yet.** This file exists so
the decision behind the face-splat compositing is recoverable, and so the
work of undoing it is a checklist rather than an archaeology exercise.

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

## The precondition

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

## What comes out

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
