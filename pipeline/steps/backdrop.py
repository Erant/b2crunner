"""The world-fixed environment both renderers draw their frames in front of.

Why it exists is body2colmap's finding (`958fd3b`), not this project's: with
a blank background a video diffusion model reads an orbit as the SUBJECT
turning on a turntable, and the prompt is not strong enough to talk it out of
that. A backdrop that sweeps past as the camera moves is the cue that says
otherwise. Which texture supplies it is not a free choice — a sky is
azimuthally symmetric apart from its sun and so barely changes over an orbit;
ruled walls meeting at corners change constantly. Hence the default: a `grid`
cube at 3x the orbit radius, which is the arrangement body2colmap measured as
carrying the cue most strongly.

**Both denoise passes are fed by a renderer, so both renderers need one.**
`render` (pyrender, steps/render.py) draws the frames pass 1 sees;
`render_splat` (brush-splat-render, steps/splat.py) draws the ones pass 2
sees. The knobs, the geometry and the compositing live here rather than in
either, so the two cannot drift.

**Always `opaque=False`, which is not body2colmap's default.** Its CLI forces
alpha to 255 because its frames are the deliverable. This pipeline carries an
image and its mask as separate arrays (steps/render.py's docstring), so the
alpha channel is the silhouette and steps downstream read it as one —
`select_support_views` weights training evidence by it, `mask_splat` consumes
it, `colmap_export` writes it. A backdrop that filled it in would hand every
one of them a subject the size of the frame. The backdrop therefore only ever
reaches the colour, never the mask.

**It is not exported and it is not fitted.** These are conditioning frames.
The backdrop never enters the point cloud or a COLMAP export, and it never
survives into a splat: `rmbg` re-derives the training matte from the denoised
frames (`foreground_masks`, `export_masks`), so what brush fits is the subject
cut out of the room, exactly as it was cut out of the flat grey before.

**The backdrop is fed to both passes; the FADE is stage 1 only.** A room
whose lines run right up to the drawing's outline reads to a video model as a
hard occluding edge, and the outline it is being read off is the BARE MESH's —
so hair and clothing get squashed back onto a naked body's silhouette. The
subject fade (`BACKGROUND_FADE_PARAMS`, `build_fade`) clears a shell around
the subject to leave room to grow into, and `render` declares it while
`render_splat` does not: by stage 2 the silhouette on offer is a splat fitted
to the DENOISED subject, already the right shape, so a clear zone there would
spend rotation cue on nothing. The near reason is the same boundary — the
shell is fitted to mesh vertices, and a .ply has none.

**Where it must stay off**, and why the two step defaults are not the whole
answer: a render whose frames feed `select_support_views` has to stay
premultiplied over black, because that step divides the alpha back out to
recover a straight colour. A backdrop makes that division recover the room.
`render_face_support_views` therefore sets `background: ""` explicitly, next
to the `bg_color: [0, 0, 0]` it already carries for the same reason.

**Every render that wants one names it itself.** These were a `background`
workflow SETTING briefly (2026-09-01), on the reasoning that frames sharing a
diffusion batch have to agree about the room. They still do — but a setting is
a control on the run form, and it put a knob nobody turns per run above the
per-step fold while hiding the colours (`background_params`) that are the
thing actually worth tuning. So the setting is gone and each render names its
own room — `background`, plus `background_base_color` and
`background_line_color`, the two colours promoted out of `background_params`
because they are the pair anybody actually turns — which is where a person
changing one already has the rest of that render's knobs in front of them. The
agreement is now a thing to keep by hand: the renders whose frames reach a
denoise pass are `render_initial_views` and `rerender_splat` (plus the shell
file's `render_shell_views`), and they must match.

**Both shipped workflows currently break that**, deliberately and since
2026-09-04: `render_initial_views` (and, in step with it,
`render_shell_views`) carries `background: ""` while `rerender_splat` still
carries `grid`. The first denoise pass was taken back to a blank backdrop to
get a comparison against the room, the fade and the fade's margin — three
things that landed in three days and were never measured against their own
absence — and the second pass was left alone because that is what was asked
for. It is a knowing exception, not the rule going soft: the note above each
of those `background:` lines says which way it should be resolved, and the
denoise prompt's 场景 slot still describes the ruled room either way.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..step import Param

logger = logging.getLogger(__name__)


#: The knobs a step splices into its own `PARAMS` to accept a backdrop. Shared
#: rather than typed out twice so `render` and `render_splat` agree on names,
#: defaults and help by construction — a person tuning a run reads one set of
#: controls, and a workflow sets the same names on either.
BACKGROUND_PARAMS: Tuple[Param, ...] = (
    # Over a blank background a video model reads an orbit as the subject
    # turning on a turntable; a backdrop sweeping past is the counter-cue.
    # `grid` (ruled walls meeting at corners, floor and ceiling distinct) is
    # what body2colmap measured as carrying it most strongly (958fd3b);
    # `checker` has more raw azimuthal signal but self-similar cells say the
    # view turned, not how far; `blender_sky` is symmetric bar its sun;
    # `gradient` is the no-cue control. (none) is REQUIRED of any render
    # feeding select_support_views, which divides the alpha back out and would
    # recover the room as the subject's colour. Never exported: `rmbg`
    # re-derives the training matte from the denoised frames.
    Param(
        "background", str, "grid",
        "Room drawn behind the subject (or a path to an equirect/cubemap image); "
        "empty for flat grey",
        choices=("grid", "checker", "blender_sky", "gradient", ""),
    ),
    # A cube at a finite radius has corners to pass, which carries the
    # rotation cue; a sphere has none and is for a sky.
    Param(
        "background_geometry", str, "cube",
        "Surface the backdrop texture is mapped onto",
        choices=("cube", "sphere"),
    ),
    # The colour the silhouette must stand out against: the drawings' fill
    # lands on #6F6F6F (111), and grid's own [0.42, 0.44, 0.48] renders
    # 107/112/122, close enough to swallow it. Applies to any generator taking
    # a `base_color` (grid, checker, gradient). Empty is the only value safe
    # for every generator. Setting `base_color` in background_params as well
    # is refused.
    Param(
        "background_base_color", list, None,
        "Wall colour as RGB in [0,1]; empty keeps the generator's own",
    ),
    # Grid's own ruling is [0.88, 0.89, 0.92] (224/226/234). Setting
    # `line_color` in background_params as well is refused.
    Param(
        "background_line_color", list, None,
        "Grid line colour as RGB in [0,1] (grid only); empty keeps the default",
    ),
    # Passed to the generator whole, as a YAML mapping, e.g.
    # {floor_color: [0.2, 0.2, 0.2]}; a key the generator does not accept is
    # rejected by name, which is why nothing grid-only is defaulted here.
    # grid: floor_color, ceiling_color, n_per_face (default 6), line_width
    # (fraction of a cell, default 0.035); checker: color_a, color_b,
    # n_per_face; gradient: top_color, bottom_color; blender_sky:
    # zenith_color, horizon_color, ground_color, sun_* angles. Colours are RGB
    # in [0,1]. Must be empty when background is a path.
    Param(
        "background_params", dict, {},
        "Extra generator options as a YAML mapping (colours, divisions, line width)",
    ),
    # Relative so it still fits when the orbit is auto-framed. Empty puts the
    # surface at infinity: it tracks camera rotation but not translation, so
    # no parallax against the subject and no corners on a cube.
    Param(
        "background_radius_scale", float, 3.0,
        "Backdrop radius as a multiple of the orbit radius (must exceed 1.0)",
    ),
    Param(
        "background_radius", float, None,
        "Backdrop radius in world units; overrides the scale when set",
        advanced=True,
    ),
    Param(
        "background_rotation_deg", float, 0.0,
        "Turn the backdrop about the vertical axis, in degrees",
        advanced=True,
    ),
    Param(
        "background_resolution", int, 1024,
        "Generated texture size in pixels (ignored for a loaded image)",
        advanced=True,
    ),
)


#: The subject fade's knobs — kept OUT of `BACKGROUND_PARAMS` and spliced in
#: by `render` alone, because the fade is a STAGE 1 control and stage 2 has
#: no fade at all.
#:
#: Two reasons it cannot simply be shared. The near one is that the shell is
#: fitted to a MESH: `Ellipsoid.fit` runs on the body's vertices, and
#: `render_splat` has a .ply and no vertices to fit — body2colmap's own
#: `configure_background_fade` raises `RuntimeError` on a splat scene for
#: exactly this reason. The far one is what the fade is FOR. It exists to
#: stop the backdrop reading as a hard occlusion boundary at the silhouette,
#: which is a failure of the drawings that condition denoise_pass1: there the
#: silhouette is the bare mesh's, and the model has to be free to paint hair
#: and clothing outside it. By stage 2 that has already happened — the frames
#: `rerender_splat` draws come off a splat FIT to the denoised subject, so
#: its silhouette is the real one and there is nothing left to expand into.
#: Clearing the room around it there would only throw away rotation cue.
BACKGROUND_FADE_PARAMS: Tuple[Param, ...] = (
    # With grid lines running right up to the outline, a video model takes
    # the outline for an occluding edge and won't paint past it, squashing
    # hair and bulky clothing onto the BARE MESH's shape. Clearing a band
    # next to the subject keeps the cue in the far field. `smoothstep` is flat
    # at both ends, so neither edge leaves a visible ring; `linear` leaves a
    # slope discontinuity, `cosine` is steeper mid-band, `step` is the
    # hard-edged control; `exponential`/`gaussian`/`inverse_square` take
    # background_fade_rate and trail off (inverse_square washes the whole
    # frame). Ignored quietly by a render with no backdrop.
    Param(
        "background_fade", str, "smoothstep",
        "Fade profile clearing the room around the subject; empty turns the fade off",
        choices=("smoothstep", "linear", "cosine", "step", "exponential",
                 "gaussian", "inverse_square", ""),
    ),
    # The shell is fitted to a NAKED SAM-3D-Body mesh, smaller than the
    # dressed subject, which argues for > 1.0. The cost, measured at `full`
    # framing 720x1280 as the share of the room's ruling surviving in frame:
    #
    #     margin   frontal view   three-quarter view
    #     1.0          33%            66-72%
    #     1.5          22%            48-49%
    #     2.0          22%            44%
    #
    # Hence 1.0, which still clears a band at the silhouette. `plain` removes
    # only the pattern, so the room's shading and corners keep cueing rotation
    # at any margin. Raise it if hair and clothing come back pinned to the
    # bare mesh's outline.
    Param(
        "background_fade_margin", float, 1.0,
        "Inflate the subject's fitted shell by this factor before fading (> 0)",
        minimum=0.0,
    ),
    # A multiple rather than pixels because the orbit is auto-framed, so the
    # subject holds its size in frame and a scale-free band holds its look.
    # 1.0 reaches full backdrop at twice the margin-inflated extent. For a
    # hard-edged clear zone use background_fade: step rather than winding
    # this down.
    Param(
        "background_fade_falloff", float, 1.0,
        "Width of the fade band, as a multiple of the subject's radius (> 0)",
        minimum=0.0,
    ),
    # `plain` re-renders the room with its pattern suppressed (walls, floor,
    # ceiling and shading kept, no ruling): the only target that removes a
    # line rather than moving it, but it needs a generated texture (a loaded
    # image cannot be split, and body2colmap refuses the pair). `color` lays
    # one flat mean colour, which reads as a patch across a floor/wall seam.
    # `blur` spreads a bright line into a grey band; the fallback for a loaded
    # texture, not the fix.
    Param(
        "background_fade_target", str, "plain",
        "What the room fades to: pattern-free room, flat colour, or blur",
        choices=("plain", "color", "blur"), advanced=True,
    ),
    # The compact profiles, smoothstep included, reach zero at the end of the
    # band by construction and ignore this.
    Param(
        "background_fade_rate", float, 4.0,
        "Tail tightness for exponential/gaussian/inverse_square fades; larger is tighter",
        minimum=0.0, advanced=True,
    ),
)


def orbit_frame(cameras: Sequence[Any]) -> Tuple[np.ndarray, float]:
    """Where a camera path looks, and how far out it sits.

    Both callers hand this the cameras they are about to render rather than
    the target and radius they were built from, because not every path has
    those to hand: `render_splat` with an empty `pattern` reuses a dataset's
    cameras and never computes an orbit at all, and both steps' anchored paths
    derive their radius inside a branch that keeps it local. What every path
    in this pipeline does share is that its cameras look at ONE point —
    `OrbitPath` calls `Camera.look_at(target)` on each, the cap pattern
    included — so that point is recoverable from the cameras themselves,
    exactly, as the least-squares intersection of their view rays.

    Args:
        cameras: The cameras to be rendered. At least one.

    Returns:
        (center, radius): the world point the views converge on, and the
        distance from it to the furthest camera. `radius` is what a backdrop
        scale is a multiple of, and a backdrop smaller than it would put a
        camera outside its own surface.

    Raises:
        ValueError: If `cameras` is empty, or if every camera sits on the
            point they look at, leaving no radius to scale.
    """
    if not len(cameras):
        raise ValueError("orbit_frame needs at least one camera")

    positions = np.array(
        [np.asarray(camera.position, dtype=np.float64).reshape(3) for camera in cameras]
    )
    directions = np.array(
        [np.asarray(camera.get_forward_vector(), dtype=np.float64).reshape(3)
         for camera in cameras]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Least-squares point closest to every view ray: sum of the projectors
    # onto each ray's orthogonal complement. Exact when the rays converge,
    # which they do, and `lstsq` rather than `solve` so a degenerate set (one
    # camera, or a path collapsed to a point) returns something instead of
    # raising a LinAlgError nobody can act on.
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    center = np.linalg.lstsq(
        projectors.sum(axis=0),
        np.einsum("nij,nj->i", projectors, positions),
        rcond=None,
    )[0]

    radius = float(np.max(np.linalg.norm(positions - center, axis=1)))
    if radius <= 0.0:
        raise ValueError(
            "orbit_frame: every camera sits on the point it looks at, so there "
            "is no orbit radius for a backdrop to be sized against"
        )
    return center.astype(np.float32), radius


def build_fade(params: Dict[str, Any], vertices: Optional[np.ndarray]):
    """The `SubjectFade` a step's params ask for, or None for no fade.

    The shell is an ellipsoid fitted to `vertices` — not to any one frame's
    silhouette, which is the point. An ellipsoid enclosing the mesh encloses
    its outline from EVERY viewpoint, so the clear zone cannot slip inside the
    drawing partway round the orbit; it is one fixed object in the world that
    the camera moves around, rather than a screen-space effect that would swim
    from frame to frame.

    Fitted on the vertices rather than on the mesh's bounding box on purpose,
    which is body2colmap's own choice for the same reason: the ellipsoid
    circumscribing a box has to clear the box's corners, and that pushes every
    semi-axis out by sqrt(3) — a third again on top of whatever
    `background_fade_margin` already asked for.

    Args:
        params: A step's resolved params, carrying `BACKGROUND_FADE_PARAMS`.
        vertices: The subject's (N, 3) mesh vertices, in the same world frame
            as the cameras — so after any `rotate_around_y`, not before.

    Returns:
        A `body2colmap.fade.SubjectFade`, or None when `background_fade` is
        empty or the step declares no fade params at all.

    Raises:
        ValueError: When a fade is asked for with no vertices to fit it to, or
            on a profile, target, margin or falloff body2colmap refuses.
    """
    profile = (params.get("background_fade") or "").strip()
    if not profile:
        return None

    if vertices is None:
        raise ValueError(
            "background_fade needs the subject's mesh vertices to fit its "
            "shell to, and this render has none. Only a mesh render can fade "
            "the room around the subject — a splat has no vertices, which is "
            "why `render_splat` does not declare these params."
        )

    from body2colmap.fade import Ellipsoid, SubjectFade

    ellipsoid = Ellipsoid.fit(
        np.asarray(vertices, dtype=np.float64).reshape(-1, 3),
        margin=params["background_fade_margin"],
    )
    fade = SubjectFade(
        ellipsoid,
        profile=profile,
        falloff=params["background_fade_falloff"],
        rate=params["background_fade_rate"],
        target=params["background_fade_target"],
    )
    logger.info(
        "background fade: %s, margin %g, falloff %g, to %s; subject "
        "semi-axes %s",
        profile, params["background_fade_margin"],
        params["background_fade_falloff"], params["background_fade_target"],
        np.array2string(ellipsoid.axes, precision=3),
    )
    return fade


def build_background(
    params: Dict[str, Any],
    cameras: Sequence[Any],
    vertices: Optional[np.ndarray] = None,
):
    """The `Background` a step's params ask for, or None for no backdrop.

    Args:
        params: A step's resolved params, carrying `BACKGROUND_PARAMS`, and
            `BACKGROUND_FADE_PARAMS` too if the step declares them.
        cameras: The cameras about to be rendered — the backdrop is centred
            and sized against them (see `orbit_frame`).
        vertices: The subject's mesh vertices, for the subject fade (see
            `build_fade`). None from a step that declares no fade params, and
            required by one that does.

    Returns:
        A `body2colmap.background.Background`, or None when `background` is
        empty.

    Raises:
        ValueError: On a texture that names neither a generator nor a path, a
            radius scale that would leave the camera outside the surface, a
            world-unit radius smaller than the orbit it has to enclose, a
            `background_params` key the chosen generator does not accept, or a
            fade body2colmap refuses — `background_fade_target: plain` over a
            texture loaded from a path is the one to expect, since a
            photograph cannot be split into pattern and shading.
    """
    texture = (params["background"] or "").strip()
    if not texture:
        # No room, so nothing to fade — and quietly, not as an error, because
        # `background_fade` defaults ON. The renders that turn the backdrop
        # off do it for `select_support_views` (see the module docstring) and
        # would otherwise have to turn the fade off in a second line that says
        # nothing a reader of the first does not already know.
        return None

    from body2colmap.background import Background

    radius = params["background_radius"]
    radius_scale = params["background_radius_scale"]

    center = None
    if radius is not None or radius_scale is not None:
        center, orbit_radius = orbit_frame(cameras)
        if radius is None:
            # An explicit radius SUPERSEDES the scale rather than conflicting
            # with it. body2colmap's Python API raises on the pair because it
            # can tell an explicit `radius_scale` from its own default; a
            # resolved param dict cannot, so a workflow that sets only
            # `background_radius` would otherwise be refused for a default it
            # never wrote. This is the rule its config-file loader uses, for
            # exactly that reason.
            if radius_scale <= 1.0:
                raise ValueError(
                    f"background_radius_scale must be > 1.0 so the camera stays "
                    f"inside the backdrop, got {radius_scale}"
                )
            radius = float(radius_scale) * orbit_radius
        elif radius <= orbit_radius:
            raise ValueError(
                f"background_radius {radius:g} does not enclose the camera path, "
                f"which reaches {orbit_radius:g} from its centre. Give it more "
                f"than that, or set background_radius_scale instead and let it "
                f"be measured against the orbit."
            )

    # The two promoted colours, folded back into the mapping the generator is
    # actually called with. They are separate params because they are the two
    # a person tunes and a mapping box is a bad place to tune anything — but
    # there is only one channel into the generator, so this is where the two
    # spellings meet. Neither is passed when it is None, which is what keeps a
    # `checker` or a `blender_sky` runnable on nothing but its defaults: a
    # grid-only key that was always sent would refuse those outright.
    generator_params = dict(params["background_params"] or {})
    for name, key in (("background_base_color", "base_color"),
                      ("background_line_color", "line_color")):
        value = params[name]
        if value is None:
            continue
        if key in generator_params:
            # Both spellings set, disagreeing or not. Silently preferring one
            # would leave the other reading as the room's colour in a UI that
            # is not describing the run.
            raise ValueError(
                f"{key} is set twice: as the `{name}` param and inside "
                f"`background_params`. Set one of them — `{name}` is the one "
                f"with its own control."
            )
        generator_params[key] = list(value)

    background = Background.create(
        texture=texture,
        geometry=params["background_geometry"],
        resolution=params["background_resolution"],
        center=center,
        radius=radius,
        rotation_deg=params["background_rotation_deg"],
        # The clear zone around the subject, or None. Handed to the
        # Background rather than applied afterwards because it belongs to the
        # BACKDROP's own render, before the base layer goes over it — which
        # is what keeps it off the subject: it can only ever lighten the room,
        # never the drawing standing in it.
        fade=build_fade(params, vertices),
        # Mostly unread — see `generator_params`. body2colmap checks these
        # against the generator's own signature and, on a miss, names every
        # argument it does accept, which is a better error than this project
        # could write and one that cannot go stale when a generator gains a
        # knob. The RGB triples are its `_as_rgb` to validate too: it already
        # refuses a wrong length and an out-of-range component by value.
        params=generator_params or None,
        # Never True. See this module's docstring: the alpha channel is this
        # pipeline's mask, and filling it in would hand every step downstream
        # a subject the size of the frame.
        opaque=False,
    )
    logger.info("background: %s %s", texture, background.describe())
    return background


def composite_bgr(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    *,
    background,
    cameras: Sequence[Any],
    flat_color: Tuple[float, float, float],
) -> List[np.ndarray]:
    """Put `background` behind frames already composited over a flat colour.

    For `render_splat`, whose frames arrive from an external rasteriser with
    a background already in them — `bg_color`, or `cull_color` in confidence
    mode — rather than as a layer with a hole in it. Un-compositing that flat
    fill and re-compositing over the environment cancels to one add:

        C*a + flat*(1-a)  ->  C*a + env*(1-a)  =  rgb + (env - flat)*(1-a)

    which needs no division, so there is nothing to guard against a small
    alpha the way `_unpremultiply` has to.

    **Exact in the ordinary mode, approximate under `confidence`.** There the
    binary returns `C*(ag) + cull*(1-ag)` while the alpha it hands back is the
    gate `g` alone, so `1-g` under-corrects wherever the splat's own coverage
    `a` is partial. The culled region — where the backdrop actually matters,
    and where `g` is 0 — is corrected exactly; what is left is a whisker of
    cull colour in the soft edge of a kept silhouette, which is the colour
    that edge was deliberately faded toward in the first place.

    Args:
        images: BGR uint8 frames, modified in place.
        masks: Matching float32 [0,1] alpha, one per frame. Not modified —
            the mask is the silhouette and the backdrop is not part of it.
        background: A `body2colmap.background.Background`.
        cameras: The camera each frame was rendered from, in order.
        flat_color: The RGB in [0,1] the frames were composited over.

    Returns:
        `images`.
    """
    flat_bgr = np.array(
        [float(c) * 255.0 for c in reversed(tuple(flat_color))], dtype=np.float32
    )
    for image, alpha, camera in zip(images, masks, cameras):
        env_bgr = background.render(camera)[..., ::-1].astype(np.float32)
        gap = (1.0 - alpha.astype(np.float32))[..., None]
        image[...] = np.clip(
            image.astype(np.float32) + (env_bgr - flat_bgr) * gap, 0, 255
        ).astype(np.uint8)
    return images
