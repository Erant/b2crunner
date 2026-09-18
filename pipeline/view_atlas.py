"""view_atlas — the mesh texture re-laid as character sheets, so one image model pass sees a person.

A meshify atlas (`mesh-unwrap`: xatlas charts, ~14k islands) is unreadable to an
image model: klein blurs the islands. This lays the SAME triangles out again as
orthographic views of the body — the sheet is a picture of the person — and
transfers the model's edit back to the original atlas texel by texel. The
geometry never changes; only where a triangle's texels live while klein looks
at them.

Three sheets (b2ctrain docs/view-atlas.md, out/mesh/view_atlas_m3/README.md, 00307):

  main    front and back, large, plus the four small axis panels (left, right,
          crown, soles) that own almost nothing once the other sheets exist;
          the reserve keeps its source UVs in the lower-right square.
  extra   N oblique full-body views (6): the reserve a camera can see, what the
          four small panels owned, and front/back surface facing its panel
          below `steal_cos` (the silhouettes, under the hem, the sides of the
          legs — 15 % of the visible pixels, foreshortened where klein could
          never resolve them).
  head    M close-up views of the top `head_height` metres (4): the ear, the
          jaw and the hair, ~200 px in the front panel and unfixable there.

Directions are picked greedily from the four small axes plus a Fibonacci
sphere by the movable surface they IMPROVE; a triangle moves only to a view
that gives it more effective texels ((px/m)^2 x facing cosine). Every panel's
context is a render of the whole mesh from that direction, so the sheet is
always a coherent person; ownership only decides which pixels come back.
38.8 % of 00307's surface is the TSDF's inner wall, never visible from
anywhere: it stays in the reserve and needs nothing.

Rasterisation is a torch z-buffer (id, depth, affine barycentrics), the
orthographic cousin of `pipeline/mesh_raster.py`: per-triangle visibility is
a depth-map test of seven interior points, the panel context and the sheet's
own texel map are id-buffer lookups. No ray tracer, no open3d. On CUDA the
whole layout takes ~30 s for 300k triangles; the CPU path exists for tests.

The transfer (`apply_sheet`) is a direct barycentric lookup, sheet texel ->
atlas texel, with the delta form (edited minus the unedited sheet, added onto
the texture) so that resampling costs nothing where klein changed nothing and
the mapped/reserved boundary stays where it was.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INF_KEY = (1 << 63) - 1
GREY = 180  # the sheet's ground, the grey the experiments' prompts name


# -- geometry ---------------------------------------------------------------------

def read_obj(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`mesh-unwrap`'s mesh_uv.obj -> (vertices (n,3), faces (m,3), per-corner uv (m,3,2)). Triangles only, one vt per corner."""
    v: List[List[float]] = []
    vt: List[List[float]] = []
    fv: List[List[int]] = []
    ft: List[List[int]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                vt.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                corners = [c.split("/") for c in line.split()[1:]]
                if len(corners) != 3:
                    raise ValueError(f"{path}: face with {len(corners)} corners; triangles only")
                if any(len(c) < 2 or not c[1] for c in corners):
                    raise ValueError(f"{path}: a face without texture coordinates")
                fv.append([int(c[0]) - 1 for c in corners])
                ft.append([int(c[1]) - 1 for c in corners])
    V = np.asarray(v, np.float64)
    F = np.asarray(fv, np.int64)
    T = np.asarray(vt, np.float64)[np.asarray(ft, np.int64)]
    return V, F, T


def write_obj(path: Path, V: np.ndarray, F: np.ndarray, uv: np.ndarray, mtl: str, groups: Sequence[Tuple[str, np.ndarray]]) -> None:
    """An OBJ with one `usemtl` group per (material, face selection); the uv is per corner (m,3,2)."""
    with open(path, "w") as out:
        out.write(f"mtllib {mtl}\n")
        for p in V:
            out.write("v %.9g %.9g %.9g\n" % tuple(p))
        for p in uv.reshape(-1, 2):
            out.write("vt %.9g %.9g\n" % tuple(p))
        for material, selected in groups:
            out.write(f"usemtl {material}\n")
            for i in np.flatnonzero(selected):
                out.write("f " + " ".join(f"{vi + 1}/{i * 3 + j + 1}" for j, vi in enumerate(F[i])) + "\n")


def triangle_geometry(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(corners (m,3,3), area (m,), unit normal (m,3), seven interior sample points (m,7,3))."""
    t = V[F]
    cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / np.maximum(2 * area[:, None], 1e-20)
    weights = np.array([[1 / 3] * 3, [.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.49, .49, .02], [.02, .49, .49], [.49, .02, .49]])
    points = np.einsum("sk,nkc->nsc", weights, t)
    return t, area, normal, points


def sphere_directions(n: int) -> np.ndarray:
    """A Fibonacci sphere, deterministic; the candidate oblique directions."""
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** .5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)


def panel_basis(d: np.ndarray, front: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Right/up image axes for a view direction: world up until the view is nearly vertical, so steep views stay upright."""
    u0 = up if abs(d @ up) < .99 else front * np.sign(d @ up)
    u = u0 - (u0 @ d) * d
    u /= np.linalg.norm(u)
    return np.cross(u, d), u


def panel_scale(proj: np.ndarray, box: Sequence[int], pad: int) -> float:
    """px per metre of a projection fitted into a pixel box."""
    x0, y0, x1, y1 = box
    return min((x1 - x0 - 2 * pad) / max(np.ptp(proj[:, 0]), 1e-9), (y1 - y0 - 2 * pad) / max(np.ptp(proj[:, 1]), 1e-9))


def grid_boxes(n: int, rows: int, region: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    """n equal pixel boxes, `rows` rows, filling the pixel region (x0,y0,x1,y1)."""
    if n <= 0:
        return []
    cols = -(-n // rows)
    X0, Y0, X1, Y1 = region
    w = (X1 - X0) / cols
    h = (Y1 - Y0) / rows
    return [(round(X0 + c * w), round(Y0 + r * h), round(X0 + (c + 1) * w), round(Y0 + (r + 1) * h)) for r in range(rows) for c in range(cols)][:n]


def overlap_pairs(tri: np.ndarray) -> np.ndarray:
    """Strict positive-area intersections among 2D triangles; shared edges are allowed.

    Spatial bins enumerate the bounding-box candidates, then a separating-axis
    test checks the actual triangles: continuous UVs, not just texel centres.
    """
    lo, hi = tri.min(1), tri.max(1)
    cell = max(float(np.median(np.max(hi - lo, axis=1))) * 3, 1e-4)
    origin = lo.min(0)
    low = np.floor((lo - origin) / cell).astype(int)
    high = np.floor((hi - origin) / cell).astype(int)
    bins: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(zip(low, high)):
        for x in range(a[0], b[0] + 1):
            for y in range(a[1], b[1] + 1):
                bins.setdefault((x, y), []).append(i)
    found = []
    batch: List[np.ndarray] = []

    def check(pairs: List[np.ndarray]) -> None:
        p = np.concatenate(pairs)
        a, b = p.T
        good = np.all(np.minimum(hi[a], hi[b]) - np.maximum(lo[a], lo[b]) > 1e-10, axis=1)
        p = p[good]
        if not len(p):
            return
        ta, tb = tri[p[:, 0]], tri[p[:, 1]]
        edges = np.concatenate([np.roll(ta, -1, axis=1) - ta, np.roll(tb, -1, axis=1) - tb], 1)
        axes = np.stack([-edges[..., 1], edges[..., 0]], -1)
        pa = np.einsum("nvc,nac->nav", ta, axes)
        pb = np.einsum("nvc,nac->nav", tb, axes)
        overlap = np.minimum(pa.max(2), pb.max(2)) - np.maximum(pa.min(2), pb.min(2))
        good = np.all(overlap > 1e-10 * np.linalg.norm(axes, axis=2), axis=1)
        if good.any():
            found.append(p[good])

    n = 0
    for ids in bins.values():
        if len(ids) < 2:
            continue
        ii, jj = np.triu_indices(len(ids), 1)
        batch.append(np.asarray(ids)[np.stack([ii, jj], 1)])
        n += len(ii)
        if n > 200000:
            check(batch)
            batch, n = [], 0
    if batch:
        check(batch)
    return np.unique(np.concatenate(found), axis=0) if found else np.empty((0, 2), int)


def sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of an image at UV coordinates (v up, the OBJ convention); edges replicate."""
    import cv2

    return cv2.remap(image, (uv[..., 0] * image.shape[1] - .5).astype(np.float32),
                     ((1 - uv[..., 1]) * image.shape[0] - .5).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# -- the rasteriser ------------------------------------------------------------------

class Raster:
    """A z-buffer of triangles given in pixel coordinates: id, depth and affine barycentrics per pixel.

    Every triangle is expanded to the pixels of its screen bbox in chunks of
    `budget` candidates; the inside ones scatter-min a packed (depth bits, id)
    key. Affine interpolation, which is exact for orthographic views and for the
    sheet's own UV space (the only two projections here).
    """

    def __init__(self, F: np.ndarray, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = device
        self.F = torch.as_tensor(np.asarray(F), dtype=torch.long, device=device)

    def run(self, P: Any, Z: Any, W: int, H: int, budget: int = 8_000_000) -> Dict[str, Any]:
        """P (n,2) pixel coordinates (centre of pixel i at i + 0.5), Z (n,) depth (smaller = nearer) -> tri (H,W) long (-1 miss), z, bary (H,W,3)."""
        torch = self.torch
        F = self.F
        P = torch.as_tensor(P, dtype=torch.float32, device=self.device)
        Z = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        z0, z1, z2 = Z[F[:, 0]], Z[F[:, 1]], Z[F[:, 2]]
        area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        umin = torch.minimum(torch.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])
        umax = torch.maximum(torch.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])
        vmin = torch.minimum(torch.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])
        vmax = torch.maximum(torch.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])
        x0 = torch.ceil(umin - 0.5)
        x1 = torch.floor(umax - 0.5)
        y0 = torch.ceil(vmin - 0.5)
        y1 = torch.floor(vmax - 0.5)
        valid = (area.abs() > 1e-12) & (x1 >= 0) & (y1 >= 0) & (x0 <= W - 1) & (y0 <= H - 1)
        x0 = x0.clamp(min=0).long()
        y0 = y0.clamp(min=0).long()
        x1 = x1.clamp(max=W - 1).long()
        y1 = y1.clamp(max=H - 1).long()
        valid &= (x1 >= x0) & (y1 >= y0)
        tris = torch.nonzero(valid)[:, 0]
        keys = torch.full((H * W,), INF_KEY, dtype=torch.long, device=self.device)
        if len(tris):
            nw = (x1 - x0 + 1)[tris]
            cnt = nw * (y1 - y0 + 1)[tris]
            ends = torch.cumsum(cnt, 0)
            starts = ends - cnt
            total = int(ends[-1])
            bounds = torch.searchsorted(ends, torch.arange(0, total, budget, device=self.device)).tolist() + [len(tris)]
            eps = 1e-6
            for c in range(len(bounds) - 1):
                lo, hi = bounds[c], bounds[c + 1]
                if hi <= lo:
                    continue
                sel = tris[lo:hi]
                n_sel = cnt[lo:hi]
                base = starts[lo:hi] - starts[lo]
                ids = torch.repeat_interleave(sel, n_sel)
                off = torch.arange(int(n_sel.sum()), device=self.device) - torch.repeat_interleave(base, n_sel)
                w = torch.repeat_interleave(nw[lo:hi], n_sel)
                px = x0[ids] + off % w
                py = y0[ids] + off // w
                cx = px.float() + 0.5
                cy = py.float() + 0.5
                a0, a1, a2 = p0[ids], p1[ids], p2[ids]
                A = area[ids]
                b0 = _edge(a1, a2, cx, cy) / A
                b1 = _edge(a2, a0, cx, cy) / A
                b2 = _edge(a0, a1, cx, cy) / A
                inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)
                zp = b0 * z0[ids] + b1 * z1[ids] + b2 * z2[ids]
                key = (zp.view(torch.int32).long() << 32) | ids
                keys.scatter_reduce_(0, (py * W + px)[inside], key[inside], "amin")
        hit = keys < INF_KEY
        tri = torch.where(hit, keys & 0xFFFFFFFF, torch.full_like(keys, -1))
        zb = torch.where(hit, (keys >> 32).to(torch.int32).view(torch.float32), torch.zeros((), device=self.device))
        pix = torch.nonzero(hit)[:, 0]
        t = tri[pix]
        cx = (pix % W).float() + 0.5
        cy = (pix // W).float() + 0.5
        A = area[t]
        b0 = _edge(p1[t], p2[t], cx, cy) / A
        b1 = _edge(p2[t], p0[t], cx, cy) / A
        bary = torch.zeros((H * W, 3), device=self.device)
        bary[pix] = torch.stack([b0, b1, 1 - b0 - b1], 1)
        return dict(tri=tri.view(H, W), z=zb.view(H, W), bary=bary.view(H, W, 3), hit=hit.view(H, W))


def _edge(a, b, cx, cy):
    return (b[:, 0] - a[:, 0]) * (cy - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cx - a[:, 0])


class OrthoView:
    """An orthographic camera: image right `r`, image up `u`, the view direction `d` pointing from the mesh TOWARD the viewer."""

    def __init__(self, d: np.ndarray, r: np.ndarray, u: np.ndarray, scale: float, mid: np.ndarray, centre: Tuple[float, float], zfar: float) -> None:
        self.d, self.r, self.u = d, r, u
        self.scale, self.mid, self.centre, self.zfar = scale, mid, centre, zfar

    def project(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(pixel coordinates (n,2), depth (n,)) — depth grows away from the viewer, always positive."""
        px = (X @ self.r - self.mid[0]) * self.scale + self.centre[0]
        py = -(X @ self.u - self.mid[1]) * self.scale + self.centre[1]
        return np.stack([px, py], 1), self.zfar - X @ self.d


def visible_from(raster: Raster, V: np.ndarray, view: OrthoView, points: np.ndarray, W: int, H: int, tol: float) -> np.ndarray:
    """Which triangles are wholly visible from the view: all seven sample points pass the depth-map test at the raster's pixels."""
    P, Z = view.project(V)
    buf = raster.run(P, Z, W, H)
    z = buf["z"].cpu().numpy()
    hit = buf["hit"].cpu().numpy()
    m, s, _ = points.shape
    Pp, Zp = view.project(points.reshape(-1, 3))
    px = np.clip(np.floor(Pp[:, 0]).astype(int), 0, W - 1)
    py = np.clip(np.floor(Pp[:, 1]).astype(int), 0, H - 1)
    ok = hit[py, px] & (Zp <= z[py, px] + tol)
    return ok.reshape(m, s).all(1)


# -- panel selection --------------------------------------------------------------------

def facing_score(visible: np.ndarray, normal: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per triangle: its facing cosine to d when visible from d, else -1."""
    cos = normal @ d
    return np.where(visible & (cos > .08), cos, -1.)


def drop_overlaps(owner: np.ndarray, specs: Sequence[Tuple], V: np.ndarray, F: np.ndarray, t: np.ndarray) -> int:
    """Positive-area overlaps within a panel: the farther triangle goes back to the reserve."""
    removed = 0
    for p, spec in enumerate(specs):
        name, d, r, u = spec[:4]
        ids = np.flatnonzero(owner == p)
        if not len(ids):
            continue
        proj = np.stack([V @ r, V @ u], 1)
        pairs = overlap_pairs(proj[F[ids]])
        if len(pairs):
            depth = t[ids].mean(1) @ d
            loser = np.unique(np.where(depth[pairs[:, 0]] < depth[pairs[:, 1]], pairs[:, 0], pairs[:, 1]))
            owner[ids[loser]] = -1
            removed += len(loser)
        logger.debug("view_atlas: %s overlap pairs %d, owned %d", name, len(pairs), int((owner == p).sum()))
    return removed


class Geometry:
    """Everything the panel picker needs about the mesh, computed once."""

    def __init__(self, V: np.ndarray, F: np.ndarray, front_yaw: float, device: str, vis_res: int) -> None:
        self.V, self.F = V, F
        self.t, self.area, self.normal, self.points = triangle_geometry(V, F)
        yaw = np.deg2rad(front_yaw)
        self.front = np.array([np.sin(yaw), 0, np.cos(yaw)])
        self.up = np.array([0., 1, 0])
        self.right = np.cross(self.up, self.front)
        self.centre = (V.min(0) + V.max(0)) / 2
        self.extent = float(np.linalg.norm(np.ptp(V, axis=0)))
        self.raster = Raster(F, device)
        self.vis_res = vis_res
        self._visible: Dict[Tuple[float, float, float], np.ndarray] = {}

    def view(self, d: np.ndarray, r: np.ndarray, u: np.ndarray, box: Sequence[int], pad: int, fit: Optional[np.ndarray] = None) -> OrthoView:
        """The orthographic view whose fitted extent (the whole mesh, or `fit`'s vertices) fills the pixel box."""
        proj = np.stack([self.V @ r, self.V @ u], 1)
        sel = proj if fit is None else proj[fit]
        x0, y0, x1, y1 = box
        scale = panel_scale(sel, box, pad)
        mid = (sel.min(0) + sel.max(0)) / 2
        return OrthoView(d, r, u, scale, mid, ((x0 + x1) / 2, (y0 + y1) / 2), self.centre @ d + self.extent * 2)

    def visible(self, d: np.ndarray) -> np.ndarray:
        """Triangles wholly visible from direction d, tested against the whole mesh at `vis_res` pixels over the body."""
        key = tuple(np.round(d, 6))
        if key not in self._visible:
            r, u = panel_basis(d, self.front, self.up)
            view = self.view(d, r, u, (0, 0, self.vis_res, self.vis_res), 2)
            tol = 1.5 / view.scale + 5e-4  # a pixel and a half of depth slop, plus half a millimetre
            self._visible[key] = visible_from(self.raster, self.V, view, self.points, self.vis_res, self.vis_res, tol)
        return self._visible[key]


def pick_panels(geo: Geometry, cands: np.ndarray, movable: np.ndarray, own_eff: np.ndarray, fit: Optional[np.ndarray], box: Sequence[int],
                K: int, prefix: str, pad: int) -> Tuple[List[Tuple], np.ndarray]:
    """Greedy view selection for one panel group.

    cands: candidate directions; movable: triangles this group may take; own_eff: their
    current effective density; fit: vertex mask the panel frames (None = whole mesh);
    box: the pixel box every panel of the group gets. Returns specs (name,d,r,u,fit) and
    the panel index per triangle (-1 = not taken): a view takes a triangle only if it
    gives it more texels than it has.
    """
    ids = np.flatnonzero(movable)
    if not len(ids) or K <= 0:
        return [], np.full(len(geo.F), -1)
    bases = [panel_basis(d, geo.front, geo.up) for d in cands]
    sel = slice(None) if fit is None else fit
    cscale = np.array([panel_scale(np.stack([geo.V[sel] @ r, geo.V[sel] @ u], 1), box, pad) for r, u in bases])
    cscore = np.stack([facing_score(geo.visible(d)[ids], geo.normal[ids], d) for d in cands], 1)
    better = (cscore > 0) & (cscale[None, :] ** 2 * cscore > own_eff[ids][:, None])
    covered = np.zeros(len(ids), bool)
    picks: List[int] = []
    area = geo.area[ids]
    for k in range(K):
        gain = ((better & ~covered[:, None]) * area[:, None]).sum(0)
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        covered |= better[:, j]
        picks.append(j)
        logger.info("view_atlas: %s panel %d direction %s adds %.1f%% of the surface", prefix, k, np.round(cands[j], 2), 100 * gain[j] / geo.area.sum())
    specs = []
    for k, j in enumerate(picks):
        d = cands[j]
        r, u = bases[j]
        el = np.degrees(np.arcsin(np.clip(d @ geo.up, -1, 1)))
        az = np.degrees(np.arctan2(d @ geo.right, d @ geo.front))
        specs.append((f"{prefix}{k}_az{az:+.0f}_el{el:+.0f}", d, r, u, fit))
    sub = np.where(better[:, picks], cscore[:, picks], -1.)
    owner = np.full(len(geo.F), -1)
    take = sub.max(1) > 0
    owner[ids[take]] = sub.argmax(1)[take]
    return specs, owner


# -- sheets --------------------------------------------------------------------------

def build_sheet(out: Path, W: int, H: int, pad: int, specs: Sequence[Tuple], owner: np.ndarray, fallback: Optional[Sequence[int]], geo: Geometry,
                olduv: np.ndarray, source: np.ndarray, protect_sources: Sequence[Path], extra_stats: Dict[str, Any], write_maps: bool) -> Tuple[Dict[str, Any], np.ndarray]:
    """Lay the owned triangles of each panel into its pixel box, render the context, rasterize the sheet.

    specs: (name, d, r, u, fit, box). With `fallback` (a pixel box) the unowned triangles keep their source
    UVs inside it and `mesh_uv.obj` addresses the whole mesh; without it only owned triangles are rasterized.
    """
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    V, F, t, area = geo.V, geo.F, geo.t, geo.area
    uv = np.zeros_like(olduv)
    context = np.full((H, W, 3), GREY, np.uint8)
    depth_img = np.zeros((H, W), np.uint8)  # a depth map in the sheet's own layout (near = bright, per panel), for a structure-following model
    seen = np.zeros((H, W), bool)
    panels = []
    for p, (name, d, r, u, fit, box) in enumerate(specs):
        x0, y0, x1, y1 = box
        view = geo.view(d, r, u, box, pad, fit)
        pix, depth = view.project(V)
        ids = owner == p
        uv[ids] = pix[F[ids]] / [W, H]
        uv[ids, :, 1] = 1 - uv[ids, :, 1]
        # The context: the whole mesh from this direction, cropped to the box.
        buf = geo.raster.run(pix - [x0, y0], depth, x1 - x0, y1 - y0)
        tri = buf["tri"].cpu().numpy()
        bary = buf["bary"].cpu().numpy()
        hit = tri >= 0
        suv = (olduv[np.maximum(tri, 0)] * bary[..., None]).sum(-2)
        colors = sample(source, suv)
        context[y0:y1, x0:x1][hit] = colors[hit]
        seen[y0:y1, x0:x1] = hit
        z = buf["z"].cpu().numpy()
        if hit.any():
            lo, hi = float(z[hit].min()), float(z[hit].max())
            rel = 1.0 - (z - lo) / max(hi - lo, 1e-6)  # near = 1
            depth_img[y0:y1, x0:x1][hit] = np.clip(rel[hit] * 215 + 40, 0, 255).astype(np.uint8)
        panels.append(dict(name=name, box_pixels=[int(x0), int(y0), int(x1), int(y1)], direction=[float(z) for z in d], px_per_m=float(view.scale),
                           head=fit is not None, context_only=name.startswith("ctx_"), triangles=int(ids.sum()), area_fraction=float(area[ids].sum() / area.sum())))
    if fallback is not None:
        x0, y0, x1, y1 = fallback
        uv[owner < 0] = olduv[owner < 0] * [(x1 - x0 - 2 * pad) / W, (y1 - y0 - 2 * pad) / H] + [(x0 + pad) / W, (H - y1 + pad) / H]
        panels.append(dict(name="fallback", box_pixels=[int(x0), int(y0), int(x1), int(y1)], triangles=int((owner < 0).sum()),
                           area_fraction=float(area[owner < 0].sum() / area.sum())))
        tri_ids = np.arange(len(F))
    else:
        tri_ids = np.flatnonzero(owner >= 0)
    # The sheet's own texel map: which triangle, at which barycentrics, so the source texture transfers exactly.
    sub = Raster(np.arange(len(tri_ids) * 3).reshape(-1, 3), geo.raster.device)
    P = uv[tri_ids].reshape(-1, 2) * [W, H]
    P[:, 1] = H - P[:, 1]
    Z = np.repeat(np.arange(len(tri_ids), dtype=np.float32) * 1e-3, 3)  # no overlaps by construction; a deterministic tie-break
    buf = sub.run(P, Z, W, H)
    tri = buf["tri"].cpu().numpy()
    bary = buf["bary"].cpu().numpy()
    hit = tri >= 0
    pid = tri_ids[np.maximum(tri, 0)]
    suv = (olduv[pid] * bary[..., None]).sum(-2)
    texture = context.copy()
    color = sample(source, suv)
    texture[hit] = color[hit]
    mask = hit.astype(np.uint8) * 255
    remap = np.full((H, W, 2), -1, np.float32)
    remap[hit] = suv[hit]
    if write_maps:
        vn = vertex_normals(V, F)
        pos = (t[pid] * bary[..., None]).sum(-2)
        nn = (vn[F[pid]] * bary[..., None]).sum(-2)
        nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-20)
        np.where(hit[..., None], pos, 0).astype(np.float32).tofile(out / "position.f32")
        np.where(hit[..., None], nn, 0).astype(np.float32).tofile(out / "normal.f32")
    # Gutters outside the silhouette take the owned surface colour; inside it the context already IS the surface.
    dist, labels = cv2.distanceTransformWithLabels(255 - mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max() + 1, np.int64)
    lut[labels[mask > 0]] = np.flatnonzero(mask)
    nearest = lut[labels]
    gutter = (mask == 0) & (dist <= pad) & ~seen
    texture[gutter] = texture.reshape(-1, 3)[nearest[gutter]]
    cv2.imwrite(str(out / "texture.png"), texture)
    cv2.imwrite(str(out / "context.png"), context)
    cv2.imwrite(str(out / "depth.png"), depth_img)
    cv2.imwrite(str(out / "mask.png"), mask)
    np.save(out / "source_uv.npy", remap)
    np.save(out / "face_panel.npy", owner.astype(np.int8))
    np.save(out / "face_uv.npy", uv)
    diffusion = texture.copy()
    edit_mask = mask.copy()
    if fallback is not None:
        diffusion[y0:y1, x0:x1] = GREY
        edit_mask[y0:y1, x0:x1] = 0
    cv2.imwrite(str(out / "diffusion_texture.png"), diffusion)
    cv2.imwrite(str(out / "edit_mask.png"), edit_mask)
    for path in protect_sources:
        src = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if src is None:
            continue
        o = sample(src, remap)
        o[mask == 0] = 0
        cv2.imwrite(str(out / path.name), o)
    stats = dict(width=W, height=H, pad=pad, verts=len(V), tris=len(F), covered=int(hit.sum()), panels=panels, **extra_stats)
    (out / "atlas.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats, uv


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.zeros_like(V)
    t = V[F]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def build_layout(atlas: Path, output: Path, *, texture: Optional[Path] = None, res: int = 4096, extra: int = 6, head: int = 4,
                 candidates: int = 64, scope: str = "grazing", steal_cos: float = .5, head_height: float = .32, front_yaw: float = 0.,
                 front_min_cos: float = .25, pad: int = 8, protect: Sequence[Path] = (), device: str = "cuda", vis_res: int = 2048,
                 write_maps: bool = False) -> Dict[str, Any]:
    """The three sheets for a meshify atlas. Returns the layout manifest (also written as `<output>/layout.json`).

    `protect`: extra R x R masks in the atlas's texel space to carry into every sheet (the atlas's own
    `protect*.png` are carried automatically).
    """
    import cv2

    V, F, olduv = read_obj(atlas / "mesh_uv.obj")
    source_texture = Path(texture) if texture else atlas / "texture.png"
    source = cv2.imread(str(source_texture))
    if source is None:
        raise FileNotFoundError(f"view_atlas: cannot read {source_texture}")
    geo = Geometry(V, F, front_yaw, device, vis_res)
    front, up, right = geo.front, geo.up, geo.right
    R = res
    main_boxes = [tuple(round(z * R) for z in b) for b in [(0, 0, .5, .72), (.5, 0, 1, .72), (0, .72, .25, 1), (.25, .72, .5, 1), (.5, .72, .75, .86), (.5, .86, .75, 1)]]
    fallback = tuple(int(z) for z in np.rint(np.array([.75, .75, 1., 1.]) * R))
    extra_boxes = grid_boxes(extra, 2 if extra > 3 else 1, (0, 0, R, R))
    # The head sheet: close-ups in the left two thirds, the whole person (the front view) on the right as
    # context only. A sheet of nothing but head-and-shoulder crops on grey is where klein invents a
    # figure or a cap (the pod run of 2026-09-18, out/mesh/view_atlas_m3/README.md): with the body
    # beside them it knows whose head it is drawing.
    head_split = round(R * 2 / 3)
    head_boxes = grid_boxes(head, 2 if head > 2 else 1, (0, 0, head_split, R))
    head_context_box = (head_split, 0, R, R)
    axes = [("front", front, right, up), ("back", -front, -right, up), ("left", right, -front, up), ("right", -right, front, up),
            ("top", up, right, -front), ("bottom", -up, right, front)]
    scores = np.stack([facing_score(geo.visible(d), geo.normal, d) for name, d, r, u in axes], 1)
    owner = scores.argmax(1)
    owner[scores.max(1) < 0] = -1
    for p in (0, 1):
        owner[scores[:, p] >= front_min_cos] = p
    main_specs = [(name, d, r, u, None, box) for (name, d, r, u), box in zip(axes, main_boxes)]
    overlap_removed = drop_overlaps(owner, main_specs, V, F, geo.t)
    scale_main = np.array([panel_scale(np.stack([V @ r, V @ u], 1), box, pad) for n, d, r, u, _, box in main_specs])
    own_cos = np.where(owner >= 0, scores[np.arange(len(F)), np.maximum(owner, 0)], -1.)
    own_eff = np.where(owner >= 0, scale_main[np.maximum(owner, 0)] ** 2 * np.maximum(own_cos, 0), 0.)
    cands = np.concatenate([np.array([d for n, d, r, u in axes[2:]], float), sphere_directions(candidates)])
    groups: List[Tuple[str, List[Tuple], np.ndarray]] = []
    if extra:
        movable = (owner < 0) if scope == "reserve" else (owner != 0) & (owner != 1)
        if scope == "grazing":
            movable |= own_cos < steal_cos
        specs_x, owner_x = pick_panels(geo, cands, movable, own_eff, None, extra_boxes[0], extra, "e", pad)
        specs_x = [(n, d, r, u, fit, box) for (n, d, r, u, fit), box in zip(specs_x, extra_boxes)]
        drop_overlaps(owner_x, specs_x, V, F, geo.t)
        taken = owner_x >= 0
        owner[taken] = -1
        if specs_x:
            sx = np.array([panel_scale(np.stack([V @ r, V @ u], 1), box, pad) for n, d, r, u, _, box in specs_x])
            dirs = np.array([s[1] for s in specs_x])
            own_eff[taken] = sx[owner_x[taken]] ** 2 * np.einsum("ij,ij->i", geo.normal[taken], dirs[owner_x[taken]])
        groups.append(("extra", specs_x, owner_x))
    if head:
        band_v = V[:, 1] > V[:, 1].max() - head_height
        band_t = geo.t[:, :, 1].mean(1) > V[:, 1].max() - head_height
        specs_h, owner_h = pick_panels(geo, cands, band_t, own_eff, band_v, head_boxes[0], head, "h", pad)
        specs_h = [(n, d, r, u, fit, box) for (n, d, r, u, fit), box in zip(specs_h, head_boxes)]
        drop_overlaps(owner_h, specs_h, V, F, geo.t)
        if specs_h:
            specs_h.append(("ctx_front", front, right, up, None, head_context_box))  # context only: no triangle is ever assigned to it
        taken = owner_h >= 0
        owner[taken] = -1
        for g in groups:
            g[2][taken] = -1
        groups.append(("head", specs_h, owner_h))
    protect_sources = sorted(atlas.glob("protect*.png")) + [Path(p) for p in protect]
    common = dict(source=str(atlas), source_texture=str(source_texture), front_yaw=front_yaw)
    stats, uv = build_sheet(output, R, R, pad, main_specs, owner, fallback, geo, olduv, source, protect_sources,
                            dict(overlap_removed=overlap_removed, kind="main", **common), write_maps)
    sheets = [dict(kind="main", dir=str(output), width=R, height=R)]
    layers = [("diffusion", owner, uv, "diffusion_texture.png")]
    for gname, specs_g, owner_g in groups:
        if not specs_g:
            continue
        gstats, guv = build_sheet(output / gname, R, R, pad, specs_g, owner_g, None, geo, olduv, source, protect_sources,
                                  dict(parent=str(output), kind=gname, **common), write_maps)
        stats[gname] = dict(panels=gstats["panels"], owned_area_fraction=float(geo.area[owner_g >= 0].sum() / geo.area.sum()))
        sheets.append(dict(kind=gname, dir=str(output / gname), width=R, height=R))
        layers.append((gname, owner_g, guv, f"{gname}/diffusion_texture.png"))
    unowned = np.ones(len(F), bool)
    direct_uv = olduv.copy()
    for name, own, luv, image in layers:
        direct_uv[own >= 0] = luv[own >= 0]
        unowned &= own < 0
    write_obj(output / "mesh_diffusion.obj", V, F, direct_uv, "mesh_diffusion.mtl", [(n, o >= 0) for n, o, _, _ in layers] + [("reserve", unowned)])
    (output / "mesh_diffusion.mtl").write_text("".join(f"newmtl {n}\nKd 1 1 1\nmap_Kd {img}\n" for n, _, _, img in layers) + "newmtl reserve\nKd 1 1 1\nmap_Kd reserve_texture.png\n")
    cv2.imwrite(str(output / "reserve_texture.png"), source)
    reserve_area = float(geo.area[unowned].sum() / geo.area.sum())
    manifest = dict(sheets=sheets, reserve_area_fraction=reserve_area, stats=stats)
    (output / "layout.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "atlas.json").write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("view_atlas: %d sheets; front/back %.1f%%, extra %.1f%%, head %.1f%%, reserve %.1f%% of the surface", len(sheets),
                100 * float(geo.area[(owner == 0) | (owner == 1)].sum() / geo.area.sum()), 100 * stats.get("extra", {}).get("owned_area_fraction", 0),
                100 * stats.get("head", {}).get("owned_area_fraction", 0), 100 * reserve_area)
    return manifest


# -- the transfer ---------------------------------------------------------------------

def apply_sheet(layout: Path, edited: np.ndarray, source_texture: np.ndarray, reference: Optional[np.ndarray], source_mask: np.ndarray,
                olduv: np.ndarray, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """The sheet's edit brought back into the atlas: (texture, changed mask, metrics).

    A direct barycentric lookup sheet texel -> atlas texel for the triangles this
    sheet owns; with `reference` (the unedited sheet) the edited-minus-reference
    delta is added onto the texture instead, so unchanged pixels are exact.
    Gutters whose nearest surface texel changed are refreshed.
    """
    import cv2

    meta = json.loads((layout / "atlas.json").read_text())
    W, H = meta["width"], meta["height"]
    if edited.shape[:2] != (H, W):
        raise ValueError(f"view_atlas: the edited sheet is {edited.shape[1]}x{edited.shape[0]}, the layout {W}x{H}")
    new = np.load(layout / "face_uv.npy")
    owner = np.load(layout / "face_panel.npy")
    h, w = source_texture.shape[:2]
    result = source_texture.copy()
    changed = np.zeros((h, w), bool)
    ids = np.flatnonzero(owner >= 0)
    if len(ids):
        sub = Raster(np.arange(len(ids) * 3).reshape(-1, 3), device)
        P = olduv[ids].reshape(-1, 2) * [w, h]
        P[:, 1] = h - P[:, 1]
        Z = np.repeat(np.arange(len(ids), dtype=np.float32) * 1e-3, 3)
        buf = sub.run(P, Z, w, h)
        tri = buf["tri"].cpu().numpy()
        bary = buf["bary"].cpu().numpy()
        hit = tri >= 0
        pid = ids[np.maximum(tri, 0)]
        uv = (new[pid] * bary[..., None]).sum(-2)
        color = sample(edited, uv)
        if reference is not None:
            color = np.clip(source_texture.astype(np.int16) + color.astype(np.int16) - sample(reference, uv).astype(np.int16), 0, 255).astype(np.uint8)
            hit &= np.any(color != source_texture, axis=-1)
        result[hit] = color[hit]
        changed = hit
    dist, labels = cv2.distanceTransformWithLabels(255 - source_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max() + 1, np.int64)
    lut[labels[source_mask > 0]] = np.flatnonzero(source_mask)
    nearest = lut[labels]
    gutter = (source_mask == 0) & (dist <= meta["pad"]) & changed.reshape(-1)[nearest]
    result[gutter] = result.reshape(-1, 3)[nearest[gutter]]
    error = np.abs(result.astype(float) - source_texture.astype(float))
    metrics = dict(edited_texels=int(changed.sum()), owned_mean_change=float(error[changed].mean()) if changed.any() else 0.0,
                   unowned_surface_max_error=float(error[(source_mask > 0) & ~changed].max(initial=0)))
    return result, changed.astype(np.uint8) * 255, metrics
