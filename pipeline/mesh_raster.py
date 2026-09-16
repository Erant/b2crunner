"""mesh_raster — a z-buffer of the body mesh at a pinhole camera, in numpy.

Two consumers, both about the face cap and both wanting the same thing —
"what does the SAM-3D-Body mesh look like from this camera, per pixel":

  * `face_pointmap_splat` in its `mesh_surface` mode puts every cap
    Gaussian ON the body model's head along the photograph's ray through
    its pixel (`hit_surface`), so the cap's shape is the head's and not
    the Sapiens pointmap's relief;
  * `cull_behind_mesh` drops, per view, the cap Gaussians the head hides
    (the far cheek seen through the temple, where the cap has no
    Gaussians of its own to occlude it), so a support view or a coverage
    render of the cap only shows what a camera could see.

Why not pyrender: it returns a depth buffer and nothing else, needs a GL
context, and a face id has to be smuggled through a flat-colour render.
An MHR body is 36k triangles and a frame is a megapixel; rasterising that
in numpy, bucketed by triangle size so the small ones go through one
vectorised pass, takes a fraction of a second, is deterministic, and runs
in a test. `steps/pointmap_splat.py`'s `mesh_front_depth` is the coarse
per-bin version of this for the depth-SCALE fit; this is the per-pixel one.

Conventions (the pipeline's, see `steps/pointmap_splat.py`): the mesh
arrives in the camera's OpenCV frame (X right, Y down, Z into the scene —
`vertices_in_pose`), a pixel (row v, column u) samples the ray through
`((u - cx) / fx, (v - cy) / fy, 1)` with no half-pixel offset, exactly as
`backproject` inverts it, and depth is z along the axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

#: OpenCV camera frame -> body2colmap world, as `steps/pointmap_splat.py`.
_FLIP = np.array([1.0, -1.0, -1.0], np.float64)


@dataclass
class Raster:
    """One view of the mesh: `depth` is z per pixel (+inf where no triangle),
    `face` the winning triangle's index (-1), `normal` its unit normal in
    the camera frame, flipped to face the camera where it does — so a
    front-facing surface has `normal[..., 2] < 0` with `facing` True, and
    a back face seen through a gap has `facing` False."""

    depth: np.ndarray
    face: np.ndarray
    normal: np.ndarray
    facing: np.ndarray

    @property
    def hit(self) -> np.ndarray:
        return self.face >= 0


def rasterize(vertices_cam: np.ndarray, faces: np.ndarray, *, fx: float, fy: float,
              cx: float, cy: float, width: int, height: int, near: float = 1e-4) -> Raster:
    """Z-buffer `faces` of `vertices_cam` (OpenCV camera frame) over a
    width x height grid. Triangles with a vertex at or behind the camera
    are dropped rather than clipped — a body the camera is inside is not
    a case this pipeline has."""
    v = np.asarray(vertices_cam, np.float64).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    depth = np.full(height * width, np.inf)
    face = np.full(height * width, -1, np.int64)
    if len(f) == 0 or len(v) == 0:
        return _finish(depth, face, v, f, width, height)

    tri = v[f]                                       # (T, 3, 3)
    z = tri[..., 2]
    ok = (z > near).all(1)
    u = fx * tri[..., 0] / np.where(ok[:, None], z, 1.0) + cx
    w = fy * tri[..., 1] / np.where(ok[:, None], z, 1.0) + cy
    x0 = np.ceil(u.min(1)); x1 = np.floor(u.max(1))
    y0 = np.ceil(w.min(1)); y1 = np.floor(w.max(1))
    x0 = np.maximum(x0, 0); y0 = np.maximum(y0, 0)
    x1 = np.minimum(x1, width - 1); y1 = np.minimum(y1, height - 1)
    ok &= (x1 >= x0) & (y1 >= y0)
    # Signed doubled area in screen space: a degenerate (edge-on) triangle
    # covers no pixel centre by itself.
    area = (u[:, 1] - u[:, 0]) * (w[:, 2] - w[:, 0]) - (u[:, 2] - u[:, 0]) * (w[:, 1] - w[:, 0])
    ok &= np.abs(area) > 1e-12
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return _finish(depth, face, v, f, width, height)

    bw = (x1 - x0 + 1).astype(np.int64)
    bh = (y1 - y0 + 1).astype(np.int64)
    side = np.maximum(bw, bh)
    # Small triangles by the thousand through one vectorised pass per size
    # class; the few large ones (a torso panel seen close) one at a time.
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64)):
        sel = idx[(side[idx] > lo) & (side[idx] <= hi)]
        if sel.size:
            _scatter(depth, face, sel, hi, x0, y0, u, w, z, area, width, height)
    for t in idx[side[idx] > 64]:
        _scatter(depth, face, np.array([t]), int(side[t]), x0, y0, u, w, z, area, width, height)
    return _finish(depth, face, v, f, width, height)


def _scatter(depth, face, sel, size, x0, y0, u, w, z, area, width, height) -> None:
    """Every pixel of a size x size window from each selected triangle's
    bbox corner: screen-space barycentrics, the inside test, and the depth
    by interpolating 1/z (which is affine in screen space; z is not)."""
    n = len(sel)
    oy, ox = np.mgrid[0:size, 0:size]
    px = x0[sel][:, None] + ox.reshape(1, -1)          # (n, size*size)
    py = y0[sel][:, None] + oy.reshape(1, -1)
    ua, ub, uc = (u[sel, k][:, None] for k in range(3))
    wa, wb, wc = (w[sel, k][:, None] for k in range(3))
    inv = 1.0 / area[sel][:, None]
    l0 = ((ub - px) * (wc - py) - (uc - px) * (wb - py)) * inv
    l1 = ((uc - px) * (wa - py) - (ua - px) * (wc - py)) * inv
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
    inside &= (px < width) & (py < height)
    if not inside.any():
        return
    iz = l0 / z[sel, 0][:, None] + l1 / z[sel, 1][:, None] + l2 / z[sel, 2][:, None]
    zz = 1.0 / iz[inside]
    flat = (py[inside].astype(np.int64) * width + px[inside].astype(np.int64))
    tri = np.broadcast_to(sel[:, None], px.shape)[inside]
    np.minimum.at(depth, flat, zz)
    win = zz <= depth[flat]
    face[flat[win]] = tri[win]


def _finish(depth, face, v, f, width, height) -> Raster:
    depth = depth.reshape(height, width)
    face = face.reshape(height, width)
    normal = np.zeros((height, width, 3))
    facing = np.zeros((height, width), bool)
    hit = face >= 0
    if hit.any():
        tri = v[f[face[hit]]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        # The point itself, for the facing test: front-facing means the
        # normal points back toward the camera along the ray.
        p = tri.mean(1)
        toward = (n * p).sum(1) < 0
        facing[hit] = toward
        n = np.where(toward[:, None], n, -n)
        normal[hit] = n
    return Raster(depth=depth, face=face, normal=normal, facing=facing)


def hit_surface(raster: Raster, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth and normal for every masked pixel: the mesh's where the ray
    hits a front face, else the nearest such pixel's (in pixel space), so
    a hair-rim pixel past the head's silhouette stays attached to the head
    instead of flying to the background. Returns (z, normal, on_mesh)."""
    from scipy.spatial import cKDTree

    on = mask & raster.hit & raster.facing
    if not on.any():
        raise ValueError("mesh_raster: not one masked pixel's ray hits the mesh's front — "
                         "the mask and the mesh are not of the same view")
    z = np.where(on, raster.depth, 0.0)
    normal = np.where(on[..., None], raster.normal, 0.0)
    off = mask & ~on
    if off.any():
        rows, cols = np.nonzero(on)
        tree = cKDTree(np.stack([rows, cols], 1))
        r_off, c_off = np.nonzero(off)
        _, j = tree.query(np.stack([r_off, c_off], 1))
        z[off] = raster.depth[rows[j], cols[j]]
        normal[off] = raster.normal[rows[j], cols[j]]
    return z, normal, on


def vertices_in_camera(mesh_world: np.ndarray, camera: Any) -> np.ndarray:
    """World -> the camera's OpenCV frame, for a body2colmap `Camera`
    (`rotation` camera-to-world, OpenGL axes)."""
    rotation = np.asarray(camera.rotation, np.float64).reshape(3, 3)
    position = np.asarray(camera.position, np.float64).reshape(3)
    return ((np.asarray(mesh_world, np.float64) - position) @ rotation) * _FLIP


def depth_at_camera(mesh_world: Tuple[np.ndarray, np.ndarray], camera: Any) -> np.ndarray:
    """The body's z-buffer from `camera`, at the camera's own size."""
    vertices, faces = mesh_world
    return rasterize(vertices_in_camera(vertices, camera), faces,
                     fx=float(camera.fx), fy=float(camera.fy), cx=float(camera.cx), cy=float(camera.cy),
                     width=int(camera.width), height=int(camera.height)).depth


def behind_mesh(points_world: np.ndarray, mesh_world: Tuple[np.ndarray, np.ndarray], camera: Any,
                margin: float) -> np.ndarray:
    """Which of `points_world` `camera` cannot see for the body: further
    than `margin` behind the body's surface along their pixel's ray, or
    behind the camera. A point off the frame, or in front of the body, or
    on a pixel the body does not cover, is visible."""
    p = ((np.asarray(points_world, np.float64) - np.asarray(camera.position, np.float64))
         @ np.asarray(camera.rotation, np.float64).reshape(3, 3)) * _FLIP
    z = p[:, 2]
    ahead = z > 1e-6
    out = ~ahead
    zs = np.where(ahead, z, 1.0)
    u = np.rint(float(camera.fx) * p[:, 0] / zs + float(camera.cx)).astype(np.int64)
    v = np.rint(float(camera.fy) * p[:, 1] / zs + float(camera.cy)).astype(np.int64)
    width, height = int(camera.width), int(camera.height)
    inside = ahead & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if inside.any():
        depth = depth_at_camera(mesh_world, camera)
        body = depth[v[inside], u[inside]]
        out[inside] = np.isfinite(body) & (z[inside] > body + margin)
    return out


def cull_behind_mesh(scene: Any, mesh_world: Tuple[np.ndarray, np.ndarray], camera: Any,
                     margin: float = 0.015) -> Any:
    """`scene` (a body2colmap SplatScene) without the Gaussians the body
    hides from `camera`. Returns the scene itself when nothing is hidden."""
    from body2colmap.splat_scene import SplatScene

    hidden = behind_mesh(scene.means, mesh_world, camera, margin)
    if not hidden.any():
        return scene
    keep = ~hidden
    extras: Optional[dict] = None
    if getattr(scene, "extras", None):
        extras = {k: np.asarray(v)[keep] for k, v in scene.extras.items()}
    return SplatScene(
        means=scene.means[keep], scales=scene.scales[keep], quats=scene.quats[keep],
        opacities=scene.opacities[keep], sh_coeffs=scene.sh_coeffs[keep],
        sh_degree=scene.sh_degree, extras=extras,
    )
