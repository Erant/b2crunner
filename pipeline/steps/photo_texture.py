"""photo_texture — the photograph projected straight into the meshify atlas.

Between `meshify` and `refine_texture`. The bake gives the face the
photograph's colours through a detour (b2ctrain out/mesh/integration,
2026-09-16): the cap's Gaussians are rasterised at the 720 x 1280 training
camera, where a 155-row face becomes a 92 x 116 px image, that image is
projected onto the mesh's VERTICES (2 mm apart), the unwrap interpolates
the vertex colours into a 4096 px atlas, and the seam levelling swaps the
photograph's broad colour for the splat bake's. This step skips all of it:
every covered texel that the photograph's own camera sees is sampled from
the photograph at its native resolution, and the bake around it is levelled
TO the photograph rather than the other way round.

The camera is the photograph's (SAM-3D-Body's focal, the frame's centre)
moved by the refinement's delta on the anchor — `refined_photo_pose`, the
same pose `face_splat_refined` hangs the cap on — with the half-pixel shift
between the fit's integer pixel centres and the renderer's. The texels come
from the atlas maps (`position.f32`, `normal.f32`); visibility is a depth
render of the atlas at that camera (`b2ctrain mesh-render --depth`).

What the photograph owns is a confidence field on the surface, not a mask:
  * the FACE (Sapiens2's Face_Neck, lips, glasses; `labels`) at full
    confidence wherever it is seen at all — those texels are the photograph's
    pixels exactly;
  * everything else the photograph sees fades in with the surface's facing
    angle (smoothed normals on a world grid, so chart cuts do not show),
    hair more strictly than skin and cloth (a strand seen at a grazing angle
    is a smear once it is on the side of the head);
  * a soft foreground from the labels (the grey studio background must never
    reach the mesh: the crown and the silhouette project onto it);
  * a region: the head and neck about `mesh_stats.head_centre` (what was
    measured) or the whole subject.
Where the photograph is partial the bake underneath is shifted by a smooth
offset field fitted where both are reliable and diffused outward on the
world grid: the cheek runs into the side of the neck at one tone instead of
the bake's, and it fades to nothing a few centimetres from the last texel
the photograph owns. Only the low band moves; the bake keeps its detail.

`protect_photo.png` is what klein never repaints: every texel the
photograph owns at full confidence (`protect: photo`, the default) or the
face core alone (`face`). Measured on 00307 (b2ctrain out/mesh/photo,
klein on top of each): whole-subject sharpness 39.5 (the bake's cap) ->
46.6 (face protected) -> 51.5 (photo protected), the cap-border colour jump
19.4 -> 14.1 -> 13.7; klein at strength 0.8 over the photograph's own
pixels only drifts them (the jacket's patches moved), so it paints the
sides and the back and leaves the front alone.

Outputs beside the atlas: `texture_photo.png` (what refine_texture starts
from), `protect_photo.png` (what it never repaints), `photo_conf.png` (the
field, for the eye) and `photo_texture.json`.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..proc import ProcessFailed, stream_command
from ..registry import register_step
from ..step import Param, Step
from .pointmap_splat import camera_pose, refined_photo_pose

logger = logging.getLogger(__name__)

#: Goliath class ids (steps/sapiens2.py SEG_CLASSES): what the face is, and what hair is.
FACE_CLASSES = (2, 3, 24, 25)
HAIR_CLASS = 4
FLIP = np.diag([1.0, -1.0, -1.0])


# -- pure helpers (tests/test_photo_texture.py) --------------------------------

def ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smoothstep from 0 at `lo` to 1 at `hi`."""
    t = np.clip((np.asarray(x, np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def photo_camera(refined: Any, given: Optional[Any], focal: float, width: int, height: int,
                 name: str = "photo.png") -> Dict[str, Any]:
    """The photograph's camera as a cameras.json entry the trainer renders.

    Pose: `refined_photo_pose(refined, given)` — the photograph's own rays
    moved by the refinement's delta on the anchor; without `given` the
    refined camera's pose itself (a path that was never look_at-turned).
    Intrinsics: the fit's focal and the frame's centre, +0.5: SAM-3D-Body
    centres pixel (i, j) at (i, j), the renderer at (i + 0.5, j + 0.5).
    """
    if given is not None:
        rotation, position = refined_photo_pose(refined, given)
    else:
        rotation, position = camera_pose(refined)
    return {"name": name, "fx": float(focal), "fy": float(focal), "cx": width / 2.0 + 0.5, "cy": height / 2.0 + 0.5,
            "position": [float(v) for v in position], "rotation": [[float(v) for v in row] for row in rotation]}


def project(points: np.ndarray, entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (u, v, z) in a cameras.json camera (OpenGL c2w rotation; z along the view)."""
    rotation = np.asarray(entry["rotation"], np.float64)
    x = (np.asarray(points, np.float64) - np.asarray(entry["position"], np.float64)) @ (rotation @ FLIP)
    z = x[:, 2]
    safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
    return entry["fx"] * x[:, 0] / safe + entry["cx"], entry["fy"] * x[:, 1] / safe + entry["cy"], z


def sample(image: np.ndarray, u: np.ndarray, v: np.ndarray, cubic: bool = False) -> np.ndarray:
    """`image` at the renderer-grid coordinates (u, v), bilinear (or bicubic), in strips (cv2.remap's 32766 limit)."""
    import cv2

    interp = cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR
    pieces = []
    for start in range(0, len(u), 16000):
        s = slice(start, start + 16000)
        pieces.append(cv2.remap(image, (u[s] - 0.5).astype(np.float32)[None], (v[s] - 0.5).astype(np.float32)[None], interp,
                                borderMode=cv2.BORDER_REPLICATE)[0])
    return np.concatenate(pieces) if pieces else np.zeros((0,) + image.shape[2:], image.dtype)


def grid_smooth(points: np.ndarray, values: np.ndarray, weights: np.ndarray, cell: float, passes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted values diffused on a world-space voxel grid, read back per point.

    The points' weights and weighted values are binned into cells; each pass
    replaces a cell by the mean of itself and its 6 neighbours, numerator
    and denominator alike, so the result is a weighted average over a ball
    that grows with sqrt(passes) x cell — a surface-space blur that ignores
    UV chart cuts. The read-back is trilinear (a cell's value read as-is
    would print the voxel grid on the surface). Returns (values / mass,
    mass) per point, mass being the diffused weight: 0 where nothing
    reliable is near.
    """
    p = np.asarray(points, np.float64)
    lo = p.min(0) - 2 * cell
    fractional = (p - lo) / cell - 0.5
    base = np.floor(fractional).astype(np.int64)
    frac = fractional - base
    shape = tuple(int(v) for v in base.max(0) + 3)
    ids = np.ravel_multi_index(np.floor((p - lo) / cell).astype(np.int64).T, shape)
    n = int(np.prod(shape))
    w = np.asarray(weights, np.float64)
    vals = np.asarray(values, np.float64).reshape(len(p), -1)
    lanes = [np.bincount(ids, weights=w, minlength=n).reshape(shape)]
    lanes += [np.bincount(ids, weights=w * vals[:, k], minlength=n).reshape(shape) for k in range(vals.shape[1])]
    for _ in range(passes):
        for k, f in enumerate(lanes):
            pad = np.pad(f, 1)
            lanes[k] = (pad[1:-1, 1:-1, 1:-1] + pad[:-2, 1:-1, 1:-1] + pad[2:, 1:-1, 1:-1] + pad[1:-1, :-2, 1:-1]
                        + pad[1:-1, 2:, 1:-1] + pad[1:-1, 1:-1, :-2] + pad[1:-1, 1:-1, 2:]) / 7.0
    flat = [f.ravel() for f in lanes]
    read = [np.zeros(len(p)) for _ in lanes]
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wt = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1]) * (frac[:, 2] if dz else 1 - frac[:, 2])
                corner = np.ravel_multi_index((base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz), shape)
                for k, f in enumerate(flat):
                    read[k] += wt * f[corner]
    mass = read[0]
    out = np.stack(read[1:], 1) / np.maximum(mass, 1e-12)[:, None]
    return out.reshape(vals.shape[0], -1).squeeze(), mass


def smooth_normals(points: np.ndarray, normals: np.ndarray, cell: float = 0.008, passes: int = 4) -> np.ndarray:
    """Unit normals averaged on the world grid: the TSDF's normals are noisy at the texel scale."""
    n, _ = grid_smooth(points, normals, np.ones(len(points)), cell, passes)
    n = n.reshape(len(points), 3)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)


def soft_masks(labels: np.ndarray, erode_px: float, face_erode_px: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(foreground, face, hair) as soft float32 maps on the photograph's grid.

    Foreground and face ramp up over `erode_px` / `face_erode_px` pixels
    inside their boundary (Sapiens2's edge is a pixel or two wide and the
    background beyond it is the studio's); hair is the plain class map,
    dilated by a pixel so the ramp at its edge is the strict one.
    """
    import cv2

    lab = np.asarray(labels)
    fg = (lab != 0).astype(np.uint8)
    face = np.isin(lab, FACE_CLASSES).astype(np.uint8)
    hair = (lab == HAIR_CLASS).astype(np.uint8)
    fg_d = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    face_d = cv2.distanceTransform(face, cv2.DIST_L2, 5)
    hair_soft = cv2.dilate(hair, np.ones((3, 3), np.uint8)).astype(np.float32)
    return ramp(fg_d, 0.5, 0.5 + erode_px), ramp(face_d, 0.5, 0.5 + face_erode_px), hair_soft


def head_region(points: np.ndarray, centre: Sequence[float], radius: float, neck_radius: float, neck_drop: float) -> np.ndarray:
    """1 on the head (a ball about `centre`) and the neck (a cylinder below it), fading out over ~15 % of the radius."""
    p = np.asarray(points, np.float64)
    c = np.asarray(centre, np.float64)
    ball = 1.0 - ramp(np.linalg.norm(p - c, axis=1) / radius, 0.88, 1.04)
    lateral = np.sqrt((p[:, 0] - c[0]) ** 2 + (p[:, 2] - c[2]) ** 2) / neck_radius
    below = c[1] - p[:, 1]
    neck = (1.0 - ramp(lateral, 0.82, 1.05)) * (1.0 - ramp(below / neck_drop, 0.9, 1.05)) * ramp(below, -0.02, 0.0)
    return np.maximum(ball, neck).astype(np.float32)


def confidence(facing: np.ndarray, visible: np.ndarray, fg: np.ndarray, face: np.ndarray, hair: np.ndarray, region: np.ndarray,
               facing_lo: float, facing_hi: float, hair_lo: float, hair_hi: float, face_lo: float, face_hi: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(the photograph's confidence per texel, the face core) — see the module docstring."""
    general = ramp(facing, facing_lo, facing_hi) * (1.0 - hair) + ramp(facing, hair_lo, hair_hi) * hair
    conf = general * visible * fg * region
    core = face * visible * ramp(facing, face_lo, face_hi) * region
    return np.maximum(conf, core).astype(np.float32), core.astype(np.float32)


def level_bake(bake: np.ndarray, photo: np.ndarray, conf: np.ndarray, hair: np.ndarray, points: np.ndarray, region: np.ndarray,
               cell: float, passes: int, reliable: float, fade_lo: float, fade_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """The bake shifted toward the photograph's tone where the photograph is partial or absent.

    The offset (photo - bake) is fitted on the texels the photograph owns
    with at least `reliable` confidence and diffused on the world grid —
    as two fields, the hair's and everything else's, because the skin's
    offset diffused into the hair behind the ear would brighten it by the
    cheek's correction. A texel takes the field whose voters' bake colour
    (diffused alongside) is nearer its own, softly. The offset is applied
    with a fade in the diffused occupancy (a distance from the reliable
    texels, in effect) so it dies out away from them. Returns (levelled
    bake, the fade).
    """
    bake = bake.astype(np.float32)
    w_all = np.where(conf >= reliable, conf, 0.0).astype(np.float64) * (region > 0)
    if w_all.sum() <= 0:
        return bake, np.zeros(len(bake), np.float32)
    offset = photo.astype(np.float64) - bake
    _, total = grid_smooth(points, np.zeros(len(points)), np.ones(len(points)), cell, passes)
    fields, fades = [], []
    for material in (hair >= 0.5, hair < 0.5):
        w = w_all * material
        if w.sum() <= 0:
            fields.append(np.zeros((len(bake), 3)))
            fades.append(np.zeros(len(bake)))
            continue
        lanes, mass = grid_smooth(points, np.concatenate([offset, bake], 1), w, cell, passes)
        lanes = lanes.reshape(len(bake), 6)
        occupancy = mass / np.maximum(total, 1e-12)
        fade = ramp(occupancy, fade_lo, fade_hi)
        # How far this field's voters look from the texel, in the bake: the nearer field wins, softly.
        distance = np.linalg.norm(lanes[:, 3:] - bake, axis=1)
        fields.append((lanes[:, :3], distance, fade))
        fades.append(fade)
    picked = np.zeros((len(bake), 3))
    fade_out = np.zeros(len(bake))
    live = [f for f in fields if isinstance(f, tuple)]
    if len(live) == 1:
        picked, fade_out = live[0][0], live[0][2]
    else:
        (o0, d0, f0), (o1, d1, f1) = live
        # Soft choice by colour distance, over a 24/255 scale; a field with no reach here loses regardless.
        s = ramp(d1 - d0, -24.0, 24.0) * (f0 > 0) + (f1 <= 0) * (f0 > 0)
        s = np.clip(s, 0.0, 1.0)
        picked = o0 * s[:, None] + o1 * (1.0 - s)[:, None]
        fade_out = f0 * s + f1 * (1.0 - s)
    fade_out = (fade_out * region).astype(np.float32)
    return bake + picked.astype(np.float32) * fade_out[:, None], fade_out


def read_f32(path: Path, shape: Tuple[int, ...]) -> np.ndarray:
    data = np.fromfile(path, np.float32)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {data.size} floats, expected {shape}")
    return data.reshape(shape)


# -- the step ------------------------------------------------------------------

def occluder_edges(depth: np.ndarray, gap: float, margin: float) -> np.ndarray:
    """A soft band (float32 in [0,1]) within `margin` px of every depth discontinuity of a render, the
    silhouette included: 1 on the edge, fading to 0 at `margin`. `depth` is 0 where nothing was hit."""
    import cv2

    if margin <= 0:
        return np.zeros(depth.shape, np.float32)
    valid = depth > 0
    far = np.where(valid, depth, np.inf).astype(np.float32)
    near = cv2.erode(far, np.ones((3, 3), np.uint8))
    far_d = cv2.dilate(np.where(valid, depth, 0).astype(np.float32), np.ones((3, 3), np.uint8))
    jump = (far_d - near > gap) & np.isfinite(near)
    jump |= valid & ~cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # the silhouette
    dist = cv2.distanceTransform((~jump).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - dist / margin, 0.0, 1.0).astype(np.float32)


def coherent_protect(u: np.ndarray, v: np.ndarray, conf: np.ndarray, width: int, height: int, clean: int) -> np.ndarray:
    """Which texels klein never repaints: the photograph's own at full confidence, as one coherent region.

    The confidence follows the mesh normal, which the TSDF leaves noisy, so a plain
    `conf >= 0.999` has a fractal boundary and thousands of one-texel holes where the
    facing hovers at the threshold. On a character sheet klein then paints around
    speckles and the transfer puts every speck back. So the mask is drawn in the
    photograph's plane, where the region is one silhouette: opened (specks go) and
    closed (pinholes fill) with a `clean` px kernel; a texel is protected when its
    pixel lies in the cleaned region and the photograph colours it at all (the
    blend at a low confidence is still the photograph's smooth fade, not klein's).
    """
    import cv2

    full = conf >= 0.999
    if clean <= 0 or not full.any():
        return full
    ui = np.clip(u.astype(int), 0, width - 1)
    vi = np.clip(v.astype(int), 0, height - 1)
    plane = np.zeros((height, width), np.uint8)
    plane[vi[full], ui[full]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean, clean))
    plane = cv2.morphologyEx(cv2.morphologyEx(plane, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return (plane[vi, ui] > 0) & (conf > 0.05)


@register_step("photo_texture")
class PhotoTextureStep(Step):
    """The photograph into the atlas (see the module docstring).

    inputs:  {"mesh_dir": str — the atlas directory meshify wrote,
              "mesh_stats"?: dict — meshify's stats (the head centre),
              "image": HxWx3 uint8 BGR — the photograph (the front panel sam3d_body ran on),
              "labels"?: HxW int — Sapiens2's class map of that photograph (locate_face's),
              "mask"?: HxW float [0,1] — a matte of it, when there are no labels (no face classes then),
              "mesh_output": dict — sam3d_body's outputs (focal_length),
              "cameras": List[Camera] — the dataset's cameras (the anchor's refined pose),
              "anchor_frame_index"?: int, "given_camera"?: Camera — as face_splat_refined takes them}
    outputs: {"texture_path": str, "protect_path": str, "photo_texture_stats": dict}
    """

    PARAMS = (
        Param("trainer_path", str, "b2ctrain", "The b2ctrain binary (mesh-render for the visibility)", advanced=True),
        Param("device", int, 0, "CUDA device index for the render", advanced=True),
        Param("region", str, "all", "What the photograph may own: all (everything it sees) or head (the head and neck "
              "about the head centre)", choices=("all", "head")),
        Param("head_radius", float, 0.16, "Radius of the head ball about mesh_stats.head_centre (metres)", minimum=0.05),
        Param("neck_radius", float, 0.10, "Radius of the neck cylinder below the head centre (metres)", minimum=0.02),
        Param("neck_drop", float, 0.18, "How far below the head centre the neck cylinder reaches (metres)", minimum=0.0),
        Param("facing_lo", float, 0.20, "Skin and cloth: the photograph fades in from this cosine to the camera ..."),
        Param("facing_hi", float, 0.50, "... to full at this one"),
        Param("hair_lo", float, 0.45, "Hair: the stricter fade-in start"),
        Param("hair_hi", float, 0.85, "Hair: full at this cosine"),
        Param("face_lo", float, 0.10, "The face core is the photograph's pixels from this cosine ..."),
        Param("face_hi", float, 0.30, "... fully at this one"),
        Param("depth_tol", float, 0.004, "Visibility: a texel deeper than the render by more than this is hidden (metres)", minimum=0.0),
        Param("edge_gap", float, 0.01, "An occluder's edge: a depth jump of more than this between neighbouring photo pixels (metres)", minimum=0.0, advanced=True),
        Param("edge_margin", float, 0.0, "Texels within this many photo pixels of an occluder's edge (or the silhouette) are not the "
              "photograph's: the mesh and the photograph disagree there by a few pixels and the occluder's colour leaks. OFF (0) "
              "by default: measured on the pod run of 2026-09-18, the texels it releases fall back to the bake, which leaks the "
              "same white under the hem (pale thigh 2.6 -> 3.4 %); worth turning on once the bake is fixed", minimum=0.0, advanced=True),
        Param("fg_erode", float, 3.0, "Foreground ramps in over this many photo pixels inside Sapiens2's silhouette", minimum=0.0),
        Param("face_erode", float, 4.0, "The face core ramps in over this many pixels inside the face classes", minimum=0.0),
        Param("normal_cell", float, 0.008, "World grid cell (metres) the normals are smoothed on", minimum=0.001, advanced=True),
        Param("normal_passes", int, 4, "Diffusion passes of the normals on that grid", minimum=0, advanced=True),
        Param("level", bool, True, "Shift the bake's low band toward the photograph's tone around what it owns"),
        Param("level_cell", float, 0.01, "World grid cell (metres) of the tone offset field", minimum=0.001, advanced=True),
        Param("level_passes", int, 40, "Diffusion passes of the offset field (its reach grows with the square root)", minimum=1),
        Param("level_reliable", float, 0.6, "Confidence a texel needs to vote for the offset", minimum=0.0, maximum=1.0),
        Param("level_fade", list, [0.01, 0.08], "The offset's fade, in diffused occupancy: [where it starts, where it is full]"),
        Param("cubic", bool, True, "Sample the photograph bicubically (bilinear otherwise)", advanced=True),
        Param("protect", str, "photo", "What klein never repaints: photo (every texel the photograph owns at full confidence; "
              "measured best on every metric) or face (the face core only, klein may retouch the rest of the photograph)",
              choices=("photo", "face")),
        Param("protect_clean", int, 9, "The protection made coherent in the photograph's plane: specks and pinholes smaller than "
              "this many photo pixels go (0 = the raw per-texel threshold)", minimum=0, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        t0 = time.time()
        atlas = Path(str(inputs["mesh_dir"]))
        for name in ("mesh_uv.obj", "texture.png", "position.f32", "normal.f32", "mask_dilated.png"):
            if not (atlas / name).is_file():
                raise FileNotFoundError(f"photo_texture: {atlas} is not a meshify atlas ({name} missing)")
        image = np.asarray(inputs["image"])
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("photo_texture: image must be HxWx3 BGR")
        image = np.ascontiguousarray(image[..., :3])
        height, width = image.shape[:2]
        labels = inputs.get("labels")
        matte = inputs.get("mask")
        if labels is None and matte is None:
            raise ValueError("photo_texture: neither labels (Sapiens2's class map of the photograph) nor mask (a matte of it) was wired")
        cameras = list(inputs["cameras"])
        anchor = int(inputs.get("anchor_frame_index") or 0)
        if not 0 <= anchor < len(cameras):
            raise ValueError(f"photo_texture: anchor_frame_index {anchor} out of range for {len(cameras)} cameras")
        focal = float(inputs["mesh_output"]["focal_length"])
        given = inputs.get("given_camera")
        if given is None:
            logger.warning("photo_texture: no given_camera; the photograph's rays are hung on the refined anchor camera itself")
        stats_in = inputs.get("mesh_stats") or {}
        centre = stats_in.get("head_centre")
        if params["region"] == "head" and not centre:
            raise ValueError("photo_texture: region head needs mesh_stats.head_centre (meshify's; the face branch was off?)")
        trainer = params["trainer_path"]
        device = str(params["device"])

        # -- the camera and its depth -------------------------------------------
        entry = photo_camera(cameras[anchor], given, focal, width, height)
        cams_json = atlas / "cams_photo.json"
        cams_json.write_text(json.dumps({"width": width, "height": height, "cameras": [entry]}, indent=1))
        render_dir = atlas / "photo_render"
        render_dir.mkdir(exist_ok=True)
        try:
            stream_command([trainer, "mesh-render", "--atlas", str(atlas), "--cameras", str(cams_json), "--output", str(render_dir),
                            "--depth", "--device", device], log_name="photo_texture.render", throttle=True)
        except ProcessFailed as exc:
            raise RuntimeError(f"photo_texture: `b2ctrain mesh-render` failed: {exc}") from exc
        depth = read_f32(render_dir / "photo.depth.f32", (height, width))
        seen = cv2.erode((depth > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        # The farthest surface within a pixel of each pixel: at a grazing angle the depth under one pixel spans
        # more than the tolerance and a texel tested against its own pixel's depth flickers in stripes; a texel
        # hidden behind the head is centimetres behind every neighbour and still fails.
        farthest = cv2.dilate(depth, np.ones((3, 3), np.uint8))
        # An occluder's edge in the photograph (the hem over the thigh, hair over the face, the arm over the
        # torso) is where the mesh and the photograph disagree by a few pixels; a texel just behind it that
        # the depth test passes then samples the occluder's colour — the petticoat's white on the thigh that
        # the second denoise grew into patches (2026-09-18). Nothing within `edge_margin` px of a depth jump
        # of `edge_gap` metres is the photograph's.
        edge = occluder_edges(depth, params["edge_gap"], params["edge_margin"])

        # -- the texels ---------------------------------------------------------
        texture = cv2.imread(str(atlas / "texture.png"), cv2.IMREAD_COLOR)
        res = texture.shape[0]
        covered = cv2.imread(str(atlas / "mask_dilated.png"), cv2.IMREAD_GRAYSCALE).ravel() > 0
        positions = np.memmap(atlas / "position.f32", np.float32, mode="r", shape=(res * res, 3))
        normals = np.memmap(atlas / "normal.f32", np.float32, mode="r", shape=(res * res, 3))
        ids = np.flatnonzero(covered)
        p = np.asarray(positions[ids], np.float64)
        u, v, z = project(p, entry)
        inside = (z > 0) & (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
        # Only what the camera can see at all goes further: the region and the rest of the maths on a fraction of the atlas.
        if params["region"] == "head":
            region_all = head_region(p, centre, params["head_radius"], params["neck_radius"], params["neck_drop"])
        else:
            region_all = np.ones(len(ids), np.float32)
        keep = inside & (region_all > 0)
        ids, p, u, v, z, region = ids[keep], p[keep], u[keep], v[keep], z[keep], region_all[keep]
        n = smooth_normals(p, np.asarray(normals[ids], np.float64), params["normal_cell"], params["normal_passes"])
        direction = np.asarray(entry["position"], np.float64) - p
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        facing = (n * direction).sum(1)
        ui, vi = np.clip(u.astype(int), 0, width - 1), np.clip(v.astype(int), 0, height - 1)
        behind = z - farthest[vi, ui]
        tol = params["depth_tol"]
        visible = (1.0 - ramp(behind, 0.5 * tol, tol)) * seen[vi, ui] * (1.0 - edge[vi, ui])

        # -- the photograph's masks ---------------------------------------------
        if labels is not None:
            lab = np.asarray(labels)
            if lab.shape != (height, width):
                raise ValueError(f"photo_texture: labels are {lab.shape}, the photograph {(height, width)}")
            fg_map, face_map, hair_map = soft_masks(lab, params["fg_erode"], params["face_erode"])
        else:
            m = np.asarray(matte, np.float32)
            if m.shape != (height, width):
                raise ValueError(f"photo_texture: mask is {m.shape}, the photograph {(height, width)}")
            fg_map = ramp(cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5), 0.5, 0.5 + params["fg_erode"])
            face_map = np.zeros((height, width), np.float32)
            hair_map = np.zeros((height, width), np.float32)
        fg = sample(fg_map, u, v)
        face = sample(face_map, u, v)
        hair = sample(hair_map, u, v)
        conf, core = confidence(facing, visible, fg, face, hair, region, params["facing_lo"], params["facing_hi"],
                                params["hair_lo"], params["hair_hi"], params["face_lo"], params["face_hi"])

        # -- the colours ----------------------------------------------------------
        photo = sample(image, u, v, cubic=params["cubic"]).astype(np.float32)
        flat = texture.reshape(-1, 3)
        bake = flat[ids].astype(np.float32)
        fade = np.zeros(len(ids), np.float32)
        if params["level"]:
            lo, hi = float(params["level_fade"][0]), float(params["level_fade"][1])
            bake, fade = level_bake(bake, photo, conf, hair, p, region, params["level_cell"], params["level_passes"],
                                    params["level_reliable"], lo, hi)
        out = photo * conf[:, None] + bake * (1.0 - conf[:, None])
        flat[ids] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        texture_path = atlas / "texture_photo.png"
        cv2.imwrite(str(texture_path), texture)

        protect = np.zeros(res * res, np.uint8)
        owned = coherent_protect(u, v, (core if params["protect"] == "face" else conf), width, height, params["protect_clean"])
        protect[ids[owned]] = 255
        protect_path = atlas / "protect_photo.png"
        cv2.imwrite(str(protect_path), protect.reshape(res, res))
        conf_map = np.zeros(res * res, np.float32)
        conf_map[ids] = conf
        cv2.imwrite(str(atlas / "photo_conf.png"), (conf_map.reshape(res, res) * 255 + 0.5).astype(np.uint8))
        fade_map = np.zeros(res * res, np.float32)
        fade_map[ids] = fade
        cv2.imwrite(str(atlas / "photo_level.png"), (fade_map.reshape(res, res) * 255 + 0.5).astype(np.uint8))

        stats = {"camera": entry, "region": params["region"], "protect": params["protect"], "texels_in_view": int(len(ids)),
                 "texels_owned": int((conf > 0.01).sum()), "texels_full": int((conf >= 0.999).sum()),
                 "texels_face": int((core >= 0.999).sum()), "texels_protected": int(owned.sum()),
                 "texels_levelled": int((fade > 0.01).sum()), "seconds": round(time.time() - t0, 1)}
        (atlas / "photo_texture.json").write_text(json.dumps(stats, indent=1))
        logger.info("photo_texture: %s in %.0fs (%d texels owned, %d the face's, %d protected, %d levelled)", texture_path,
                    stats["seconds"], stats["texels_owned"], stats["texels_face"], stats["texels_protected"], stats["texels_levelled"])
        return {"texture_path": str(texture_path), "protect_path": str(protect_path), "photo_texture_stats": stats}
