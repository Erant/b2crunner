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
    Param(
        "background", str, "grid",
        "The room this render draws the subject in. It exists to break one "
        "failure: over a blank background a video diffusion model reads an "
        "orbit as the SUBJECT turning on a turntable, and the prompt is not "
        "strong enough to talk it out of that — a backdrop that sweeps past as "
        "the camera moves is the cue that says otherwise. `grid` is ruled walls "
        "meeting at corners over a floor and ceiling that read apart, which is "
        "the arrangement body2colmap measured as carrying that cue most "
        "strongly (958fd3b); `checker` carries more raw azimuthal signal but "
        "its cells are self-similar, so it says the view turned without saying "
        "how far; `blender_sky` is symmetric about the vertical axis apart from "
        "its sun; `gradient` is the control with no cue at all. A path to an "
        "equirectangular image, a packed cubemap or a directory of six cube "
        "faces works instead of a generator. (none) is the flat grey every run "
        "before 2026-09-01 used, and is REQUIRED of any render feeding "
        "select_support_views, which divides the alpha back out and would "
        "recover the room as the subject's own colour. Nothing exports it: "
        "`rmbg` re-derives the training matte from the denoised frames, so what "
        "brush fits is still the subject cut out of the room",
        choices=("grid", "checker", "blender_sky", "gradient", ""),
    ),
    Param(
        "background_geometry", str, "cube",
        "Surface the texture is mapped onto. A cube at a finite radius is a room "
        "with corners to pass, which is what carries the rotation cue; a sphere "
        "has none, and is for a sky",
        choices=("cube", "sphere"),
    ),
    Param(
        "background_base_color", list, None,
        "The WALL colour, as RGB in [0,1] — the one backdrop knob this "
        "pipeline actually tunes, so it gets its own control instead of "
        "living in `background_params` behind a key name. It is the colour "
        "the silhouette has to stand out against: the drawings' fill lands on "
        "#6F6F6F (111), and grid's own [0.42, 0.44, 0.48] renders 107/112/122, "
        "close enough to swallow it. Also `checker`'s and `gradient`'s wall — "
        "any generator that takes a `base_color`. EMPTY leaves the generator's "
        "own, which is the only value that is safe for every generator",
    ),
    Param(
        "background_line_color", list, None,
        "The colour of grid's RULING, as RGB in [0,1]. Its own control for the "
        "same reason as background_base_color: it and the wall are the pair "
        "that decide whether the room reads. EMPTY leaves grid's own "
        "[0.88, 0.89, 0.92] (224/226/234). Only `grid` takes it",
    ),
    Param(
        "background_params", dict, {},
        "Colours and shape for the chosen generator, passed to it whole — the "
        "backdrop's own appearance, as opposed to where its surface sits. EVERY "
        "colour is configurable, none is baked in. Each generator names its "
        "own: grid takes floor_color and ceiling_color, plus n_per_face "
        "(divisions per face, default 6) and line_width (a fraction of one "
        "cell, default 0.035); checker takes color_a/color_b and n_per_face; "
        "gradient takes top_color/bottom_color; blender_sky takes zenith_color, "
        "horizon_color, ground_color and the sun_* angles. The two that ARE "
        "tuned — the wall and grid's ruling — have their own controls above; "
        "setting one of those here as well is refused rather than silently "
        "resolved. Every colour is RGB in [0,1], the same as bg_color and "
        "mesh_color. Written as a YAML mapping — {floor_color: [0.2, 0.2, "
        "0.2]} — and rejected, by name, if the generator does not accept a "
        "key, which is why nothing grid-only is defaulted here: a leftover key "
        "would refuse the run the moment somebody picked checker. Must be "
        "empty when background is a path: a loaded image has no parameters to "
        "take",
    ),
    Param(
        "background_radius_scale", float, 3.0,
        "Backdrop radius as a multiple of the orbit radius, so it still fits when "
        "the orbit is auto-framed. Must exceed 1.0 — the camera has to end up "
        "inside. Empty puts the surface at infinity instead, where it tracks "
        "camera rotation but not translation: no parallax against the subject, "
        "and no corners on a cube. Superseded by background_radius",
    ),
    Param(
        "background_radius", float, None,
        "Backdrop radius in world units. Supersedes background_radius_scale when "
        "set; empty leaves the scale in charge", advanced=True,
    ),
    Param(
        "background_rotation_deg", float, 0.0,
        "Turn the environment about the vertical axis — aims a sky's sun, or "
        "turns a cube's walls relative to the subject", advanced=True,
    ),
    Param(
        "background_resolution", int, 1024,
        "Generated texture size: equirect height, or cube face size. Ignored for "
        "a texture loaded from a path, which keeps its own", advanced=True,
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
    Param(
        "background_fade", str, "smoothstep",
        "Fade the room out in a shell around the subject, so the drawing does "
        "not read as a hard occlusion boundary. The backdrop that carries the "
        "rotation cue costs something at the silhouette: with grid lines "
        "running right up to the outline, a video diffusion model takes the "
        "outline for an occluding edge and refuses to paint past it — bulky "
        "clothing and hair get squashed back onto the shape of the BARE MESH, "
        "which is the one thing these frames must not pin down. Clearing a "
        "band next to the subject keeps the cue in the far field and leaves "
        "room to expand into. This names the profile the room comes back "
        "over: `smoothstep` is flat at both ends, so neither the onset nor "
        "the outer edge leaves a visible ring; `linear` leaves a slope "
        "discontinuity, `cosine` is steeper through the middle, `step` is the "
        "hard-edged control, and `exponential`/`gaussian`/`inverse_square` "
        "take `background_fade_rate` and trail off instead of ending "
        "(inverse_square's tail is a wash over the whole frame). (none) turns "
        "the fade off. Ignored, quietly, by a render with no backdrop — "
        "there is nothing to fade",
        choices=("smoothstep", "linear", "cosine", "step", "exponential",
                 "gaussian", "inverse_square", ""),
    ),
    Param(
        "background_fade_margin", float, 1.0,
        "Inflate the fitted shell by this before the fade is measured. 1.0 is "
        "the hull that just encloses the mesh, and the argument for going "
        "past it is real: the ellipsoid is fitted to a NAKED SAM-3D-Body "
        "mesh, and the subject the denoise is meant to produce is a dressed "
        "person with hair, bigger than the thing measured in every direction. "
        "The argument against is what it costs, which was MEASURED on a real "
        "body at the shipped framing — `full`, 720x1280 — by counting how "
        "much of the room's ruling survives in frame:\n"
        "\n"
        "    margin   frontal view   three-quarter view\n"
        "    1.0          33%            66-72%\n"
        "    1.5          22%            48-49%\n"
        "    2.0          22%            44%\n"
        "\n"
        "At 2.0 the ruling is gone from the whole half of the frame around "
        "the figure and only the far wall keeps any. The backdrop exists to "
        "carry the rotation cue, so that is a lot to spend on headroom the "
        "denoise may not need — hence 1.0, which still clears a band against "
        "the silhouette. What survives at ANY margin is the room's shading: "
        "`plain` takes out the pattern only, so the lighter ceiling, the dark "
        "floor and the corners between walls stay put and keep cueing "
        "rotation where the lines have gone. Raise it if the frames come back "
        "with hair and clothing pinned to the bare mesh's outline. Must be "
        "> 0",
        minimum=0.0,
    ),
    Param(
        "background_fade_falloff", float, 1.0,
        "Width of the band the room fades back in over, as a multiple of the "
        "subject's own radius in that direction — so 1.0 reaches full "
        "backdrop at twice the (margin-inflated) subject extent. A multiple "
        "rather than a pixel count on purpose: the orbit is auto-framed, so "
        "the subject holds its size IN FRAME and a scale-free band holds its "
        "look with it. Must be > 0; use `background_fade: step` for a "
        "hard-edged clear zone instead of winding this down",
        minimum=0.0,
    ),
    Param(
        "background_fade_target", str, "plain",
        "What the room fades TO. `plain` re-renders the same room with its "
        "PATTERN suppressed — grid's walls, floor and ceiling in their own "
        "colours and no ruling — so the lines fade out while the wall, its "
        "shading and the room's corners stay put; it is the only one that "
        "removes a line rather than moving it, and it needs a generated "
        "texture (a loaded image cannot be split into pattern and shading, "
        "and body2colmap refuses the pair outright). `color` lays down one "
        "flat colour, the texture's mean, which reads as a patch wherever the "
        "zone crosses a floor/wall seam. `blur` area-averages the room into "
        "itself, which SPREADS a bright line into a grey band rather than "
        "removing it — the fallback for a loaded texture, not the fix",
        choices=("plain", "color", "blur"), advanced=True,
    ),
    Param(
        "background_fade_rate", float, 4.0,
        "Shape constant for the three profiles that trail off — "
        "`exponential`, `gaussian` and `inverse_square`. Larger is tighter. "
        "Ignored by the compact profiles, `smoothstep` included, which reach "
        "zero at the end of the band by construction", minimum=0.0,
        advanced=True,
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
