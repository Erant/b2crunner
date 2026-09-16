"""meshify — a textured mesh from a trained splat, through `b2ctrain mesh-*`.

The chain of b2ctrain/docs/mesh-plan.md, orchestrated: the trained splat is
probed for depth and colour at the training cameras and at an orbit around
the subject (`b2ctrain probe --depth --images`), the depths are fused into
a TSDF with the body mesh as the prior for what no view saw (`mesh-fuse`),
the surface is refined to the Sapiens normal maps of the training frames
(`mesh-refine`), coloured from the splat's own renders with the face cap's
photograph projected onto the face (`mesh-bake`), and unwrapped into a
texture atlas with the position / normal / mask maps and the protection
masks the texture refinement (steps/refine_texture.py) reads (`mesh-unwrap`).

Why the splat's own renders and not the frames: the generated frames
disagree with each other by pixels (a limb per orbit segment, a head pose
per view) and any texture baked from them is their blurry mean; the splat
already resolved that disagreement into its view-dependent bands, and a
render of it is consistent with the fused geometry by construction. The
frames' bake bought ~10 % on a sharpness metric after the diffusion pass on
one subject and nothing on the face (mesh-plan.md, decision 1).

The face's geometry is the body model's, not the splat's (`face_geometry:
prior`, the sgc3 recipe): within 3 cm of the cap no depth view is fused,
the prior's signed distance shapes the face and a band blends the two.
Measured on both subjects: the cap's own relief has the eyes proud and the
nose flat, and the splat's depth at the profile is a recess behind the cap
sheet, so either fused alone gives a concave profile; the body model's head
gives a nose. The cap stays the authority for COLOUR: its Gaussians are one
per photo pixel, rasterised at the photograph's camera they are the
photograph's face, and every vertex that camera sees inside their coverage
takes the colour under its own pixel, seam-levelled against the bake.

What comes out, under `output_dir` (the `mesh/` deliverable):
  cams/           the camera lists the probes ran at
  probes/<set>/   the splat's depth (mm, 16-bit) and RGBA renders per camera
  mesh_raw.ply, mesh_geom.ply, mesh_col.ply   fused / refined / coloured
  atlas/          mesh_uv.obj + .mtl, texture.png, position.f32, normal.f32,
                  mask.png, mask_dilated.png, protect_cap.png, protect_head.png,
                  atlas.json
  meshify.json    what ran, with every command line
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..proc import ProcessFailed, stream_command
from ..registry import register_step
from ..step import Param, Step
from .body_refit import cameras_json

logger = logging.getLogger(__name__)


# -- pure helpers (tests/test_meshify.py) --------------------------------------

def look_at_c2w(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenGL camera-to-world rotation (columns right, up, back) looking from
    `position` at `target` with +y up — the cameras.json convention."""
    f = np.asarray(target, np.float64) - np.asarray(position, np.float64)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def orbit_cameras(target: Sequence[float], radius: float, fx: float, width: int, height: int,
                  elevations: Sequence[float], azimuth_step: float, prefix: str = "orb") -> Dict[str, Any]:
    """Flat circular orbits about `target`, one ring per elevation, in the
    trainer's cameras.json layout. Azimuth 0 is the +z side (the front of a
    subject the pipeline renders), positive elevation is above."""
    target = np.asarray(target, np.float64)
    entries = []
    for el in elevations:
        for az in np.arange(0.0, 360.0, azimuth_step):
            a, e = np.radians(az), np.radians(el)
            d = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
            pos = target + radius * d
            entries.append({
                "name": f"{prefix}_e{int(round(el)):+03d}_a{int(round(az)) % 360:03d}.png",
                "fx": float(fx), "fy": float(fx), "cx": width / 2.0, "cy": height / 2.0,
                "position": [float(v) for v in pos],
                "rotation": look_at_c2w(pos, target).tolist(),
            })
    return {"width": int(width), "height": int(height), "cameras": entries}


def subject_frame(mesh_vertices: np.ndarray, cameras: Sequence[Any]) -> Tuple[np.ndarray, float, float]:
    """The orbit's centre (the body mesh's bounds centre), its radius (the
    mean distance of the training cameras to that centre) and the focal to
    use (the first camera's). What f3_c.sh regenerated orbit4 from."""
    v = np.asarray(mesh_vertices, np.float64)
    centre = (v.min(0) + v.max(0)) / 2
    positions = np.asarray([[float(x) for x in cam.position] for cam in cameras], np.float64)
    radius = float(np.linalg.norm(positions - centre, axis=1).mean())
    return centre, radius, float(cameras[0].fx)


def head_centre_from_cap(cap_points: np.ndarray, facing: Sequence[float]) -> np.ndarray:
    """The face cap's Gaussians are the front of the head; their centroid
    pushed 5 cm back along `facing` (the direction the face looks, toward
    the photograph's camera) is a head centre good to a few centimetres,
    which is all the face band (`protect_head`) needs."""
    p = np.asarray(cap_points, np.float64)
    f = np.asarray(facing, np.float64)
    f = f / max(np.linalg.norm(f), 1e-9)
    return p.mean(0) - 0.05 * f


def camera_facing(camera: Any) -> np.ndarray:
    """The direction a face photographed by `camera` looks along: from the
    subject toward the camera, i.e. the camera's +z (back) axis in the
    world (OpenGL camera-to-world columns: right, up, back)."""
    c2w = np.asarray(camera.rotation, np.float64)
    return c2w[:, 2] / max(np.linalg.norm(c2w[:, 2]), 1e-9)


def encode_normal_png(normal: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """A Sapiens normal map (HxWx3 float32 in [-1, 1]) as the RGBA uint8 the
    trainer and mesh-refine read: (n + 1) / 2 per channel, RGB order, the
    subject mask in alpha (255 everywhere without one)."""
    rgb = np.clip((np.asarray(normal, np.float32) + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    if alpha is None:
        a = np.full(rgb.shape[:2], 255, np.uint8)
    else:
        a = np.asarray(alpha)
        if a.dtype != np.uint8:
            a = np.clip(a.astype(np.float32) * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.dstack([rgb, a])


def read_ply_xyz(path: Path) -> np.ndarray:
    """The x/y/z of a binary little-endian vertex-only ply (the face cap)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace").splitlines()
    n = 0
    props: List[Tuple[str, str]] = []
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2", "char": "i1"}
    in_vertex = False
    for line in header:
        words = line.split()
        if not words:
            continue
        if words[0] == "element":
            in_vertex = words[1] == "vertex"
            if in_vertex:
                n = int(words[2])
        elif words[0] == "property" and in_vertex:
            props.append((words[2], types[words[1]]))
    dtype = np.dtype(props)
    rec = np.frombuffer(raw[end:end + n * dtype.itemsize], dtype)
    return np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """A binary little-endian triangle mesh (float xyz, `list uchar uint`), the layout every reader here agrees on."""
    v = np.ascontiguousarray(np.asarray(vertices, np.float32)).reshape(-1, 3)
    f = np.ascontiguousarray(np.asarray(faces, np.uint32)).reshape(-1, 3)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(f)}\nproperty list uchar uint vertex_indices\nend_header\n")
    rec = np.empty(len(f), dtype=[("n", "u1"), ("i", "<u4", (3,))])
    rec["n"] = 3
    rec["i"] = f
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(v.tobytes())
        fh.write(rec.tobytes())


def trainer_has_mesh(trainer: str) -> bool:
    """Does this b2ctrain carry the mesh-* subcommands (docs/mesh.md)?"""
    try:
        result = subprocess.run([trainer, "mesh-fuse", "--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "--prior" in result.stdout + result.stderr


# -- the step ------------------------------------------------------------------

@register_step("meshify")
class MeshifyStep(Step):
    """The trained splat as a textured mesh (see the module docstring).

    inputs:  {"splat_path": str — the trained .ply,
              "cameras": List[Camera] — the training cameras (body2colmap),
              "mesh_world": (vertices, faces) — the body mesh in the world frame (the fusion prior),
              "cap_path"?: str — the face cap's .ply (face_splat_refined): the face's region and colours,
              "normal_maps"?: List[HxWx3 float32] — Sapiens normals of the training frames, for the refine,
              "masks"?: List[HxW] — the training frames' subject masks (the normal maps' alpha),
              "anchor_frame_index"?: int — the training camera the photograph is at (the cap's projection camera),
              "extra_cameras"?: List[Camera] — more cameras to probe and bake from (a helix)}
    outputs: {"mesh_dir": str — the atlas directory (mesh_uv.obj, texture.png, the maps),
              "mesh_stats": dict}
    """

    PARAMS = (
        Param("trainer_path", str, "b2ctrain", "The b2ctrain binary (probe and the mesh-* subcommands)", advanced=True),
        Param("output_dir", str, help="Where the mesh deliverable is written (the run's mesh/)"),
        Param("device", int, 0, "CUDA device index", advanced=True),
        Param("orbit_elevations", list, [-30, -10, 10, 30], "Elevations (degrees) of the orbit rings the splat is probed along"),
        Param("orbit_azimuth_step", float, 5.0, "Azimuth step (degrees) of the orbit rings: 72 cameras a ring at 5", minimum=1.0),
        Param("probe_tau", float, 0.5, "Accumulated alpha marking the splat's surface in the depth probe", minimum=0.01, maximum=0.99),
        Param("voxel", float, 0.002, "TSDF voxel size (metres)", minimum=0.0005),
        Param("bbox_margin", float, 0.12, "Grid margin around the body mesh's bounds (metres): hair and clothing beyond the model", minimum=0.0),
        Param("prior_offset", float, -0.005, "The body prior's surface shifted inward by this (metres) where it fills"),
        Param("carve_min", int, 3, "Views a voxel must be outside of before it is carved", minimum=1),
        Param("min_component", float, 0.05, "Drop mesh components below this fraction of the largest one's area"),
        Param("face_geometry", str, "prior",
              "prior: the body model's head shapes the face (no depth view fused within face_radius of the cap; the sgc3 recipe). "
              "splat: the fused depth as everywhere else", choices=("prior", "splat")),
        Param("face_radius", float, 0.03, "Reach of the face protection about the cap's points (metres)"),
        Param("face_band", float, 0.02, "Blend width from the protected face into the fused surface (metres)"),
        Param("refine_iters", int, 400, "Normal-refine iterations (0 = skip the refine)", minimum=0),
        Param("refine_lam_pos", float, 10.0, "Refine: pull toward the fused positions"),
        Param("refine_lam_lap", float, 2.0, "Refine: Laplacian smoothness"),
        Param("bake_power", float, 6.0, "Bake: facing weight exponent"),
        Param("bake_trim", float, 0.12, "Bake: colour distance of the trimmed mean about the per-vertex median"),
        Param("bake_smooth", int, 1, "Bake: 1-ring median passes on the colours", minimum=0),
        Param("bake_training_views", bool, False,
              "Bake from the training cameras' probes as well as the orbit's (the reference baked from the orbit and the helix only)"),
        Param("cap_lowpass", int, 300, "Cap projection: seam-levelling rounds (0 = paste the photograph's colours as they are)", minimum=0),
        Param("atlas_tris", int, 300000, "Decimation target before the unwrap", minimum=1000),
        Param("atlas_res", int, 4096, "Texture atlas resolution", minimum=256),
        Param("atlas_pad", int, 6, "Chart padding (texels)", minimum=0),
        Param("chart_smooth", int, 2, "Laplacian passes on the charting copy of the decimated mesh (xatlas time and chart count)", minimum=0),
        Param("head_radius", float, 0.12, "The protect_head band's radius about the head centre (metres)"),
        Param("keep_probes", bool, False, "Keep the probe renders under output_dir (a few hundred MB) instead of deleting them", advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        import cv2

        trainer = params["trainer_path"]
        if shutil.which(trainer) is None and not Path(trainer).is_file():
            raise RuntimeError(f"meshify: trainer binary {trainer!r} not found on PATH")
        if not trainer_has_mesh(trainer):
            raise RuntimeError(f"meshify: {trainer!r} has no `mesh-fuse`; a b2ctrain from 2026-09-16 or later is needed")
        splat_path = Path(str(inputs["splat_path"]))
        if not splat_path.is_file():
            raise FileNotFoundError(f"meshify: no splat at {splat_path}")
        cameras = list(inputs["cameras"])
        if not cameras:
            raise ValueError("meshify: no cameras")
        mesh_world = inputs.get("mesh_world")
        if not isinstance(mesh_world, (tuple, list)) or len(mesh_world) != 2:
            raise ValueError("meshify: mesh_world must be the (vertices, faces) pair scene.mesh_world holds")
        body_v, body_f = np.asarray(mesh_world[0], np.float64), np.asarray(mesh_world[1], np.int64)
        cap_path = inputs.get("cap_path")
        cap = Path(str(cap_path)) if cap_path else None
        if cap is not None and not cap.is_file():
            logger.warning("meshify: cap %s does not exist; the face keeps the fused geometry and the bake's colours", cap)
            cap = None
        normal_maps = inputs.get("normal_maps")
        masks = inputs.get("masks")
        anchor = int(inputs.get("anchor_frame_index") or 0)
        if not 0 <= anchor < len(cameras):
            raise ValueError(f"meshify: anchor_frame_index {anchor} out of range for {len(cameras)} cameras")
        extra = list(inputs.get("extra_cameras") or [])
        device = str(params["device"])

        out = Path(params["output_dir"])
        cams_dir, probes_dir, atlas_dir = out / "cams", out / "probes", out / "atlas"
        for d in (cams_dir, probes_dir):
            d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        record: Dict[str, Any] = {"splat": str(splat_path), "commands": []}

        def run(cmd: List[str], name: str) -> None:
            record["commands"].append(" ".join(cmd))
            logger.info("meshify %s: %s", name, " ".join(cmd))
            try:
                stream_command(cmd, log_name=f"meshify.{name}", throttle=True)
            except ProcessFailed as exc:
                raise RuntimeError(f"meshify: `b2ctrain {name}` failed: {exc}") from exc

        # -- cameras: the training views, the orbit, the extras ------------
        width, height = int(cameras[0].width), int(cameras[0].height)
        train_json = cams_dir / "train.json"
        train_json.write_text(json.dumps(cameras_json(cameras)))
        centre, radius, fx = subject_frame(body_v, cameras)
        orbit = orbit_cameras(centre, radius, fx, width, height, [float(e) for e in params["orbit_elevations"]],
                              float(params["orbit_azimuth_step"]))
        orbit_json = cams_dir / "orbit.json"
        orbit_json.write_text(json.dumps(orbit))
        sets = [("train", train_json), ("orbit", orbit_json)]
        if extra:
            extra_json = cams_dir / "extra.json"
            extra_json.write_text(json.dumps(cameras_json(extra)))
            sets.append(("extra", extra_json))
        record["orbit"] = {"centre": centre.tolist(), "radius": radius, "fx": fx, "cameras": len(orbit["cameras"])}

        # -- probes: depth + RGBA render of the splat at every set ----------
        for name, cams in sets:
            run([trainer, "probe", "--splat", str(splat_path), "--cameras", str(cams), "--output-dir", str(probes_dir / name),
                 "--tau", str(params["probe_tau"]), "--depth", "--images", "--device", device], f"probe-{name}")

        # -- fuse -----------------------------------------------------------
        body_ply = out / "body.ply"
        write_mesh_ply(body_ply, body_v, body_f)
        fuse = [trainer, "mesh-fuse", "--output", str(out / "mesh_raw.ply"), "--device", device]
        for name, cams in sets:
            fuse += ["--views", str(cams), str(probes_dir / name)]
        for name, cams in sets:
            fuse += ["--carve", str(cams), str(probes_dir / name)]
        fuse += ["--carve-min", str(params["carve_min"]), "--carve-dilate", "2",
                 "--bbox-from", str(body_ply), "--bbox-margin", str(params["bbox_margin"]),
                 "--voxel", str(params["voxel"]), "--prior", str(body_ply), "--prior-mode", "fill",
                 "--prior-offset", str(params["prior_offset"]), "--min-comp", str(params["min_component"])]
        protect_face = cap is not None and params["face_geometry"] == "prior"
        if protect_face:
            fuse += ["--protect", str(cap), str(params["face_radius"]), "--protect-groups", "none",
                     "--protect-band", str(params["face_band"])]
        run(fuse, "mesh-fuse")

        # -- refine to the normal maps ---------------------------------------
        geom = out / "mesh_raw.ply"
        if params["refine_iters"] > 0 and normal_maps:
            if len(normal_maps) != len(cameras):
                raise ValueError(f"meshify: {len(normal_maps)} normal maps for {len(cameras)} cameras")
            normals_dir = out / "normals"
            normals_dir.mkdir(exist_ok=True)
            for i, normal in enumerate(normal_maps):
                alpha = masks[i] if masks is not None and i < len(masks) else None
                rgba = encode_normal_png(normal, alpha)
                cv2.imwrite(str(normals_dir / f"frame_{i:05d}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            refine = [trainer, "mesh-refine", "--input", str(geom), "--output", str(out / "mesh_geom.ply"),
                      "--views", str(train_json), str(normals_dir), "--iters", str(params["refine_iters"]),
                      "--lam-pos", str(params["refine_lam_pos"]), "--lam-lap", str(params["refine_lam_lap"]), "--device", device]
            if protect_face:
                refine += ["--keep", str(cap), str(params["face_radius"])]
            run(refine, "mesh-refine")
            geom = out / "mesh_geom.ply"
            shutil.rmtree(normals_dir, ignore_errors=True)
        else:
            logger.info("meshify: no normal maps (or refine_iters 0); the fused surface is used as it is")

        # -- bake -------------------------------------------------------------
        bake = [trainer, "mesh-bake", "--input", str(geom), "--output", str(out / "mesh_col.ply"), "--device", device,
                "--mode", "trimmed", "--trim", str(params["bake_trim"]), "--power", str(params["bake_power"]),
                "--alpha-min", "128", "--smooth", str(params["bake_smooth"])]
        for name, cams in sets:
            if name == "train" and not params["bake_training_views"]:
                continue
            bake += ["--views", str(cams), str(probes_dir / name)]
        if cap is not None:
            bake += ["--project", str(train_json), str(anchor), "--cap", str(cap), "--lowpass", str(params["cap_lowpass"])]
            if params["cap_lowpass"] <= 0:
                bake += ["--no-level"]
        run(bake, "mesh-bake")

        # -- unwrap -----------------------------------------------------------
        unwrap = [trainer, "mesh-unwrap", "--input", str(out / "mesh_col.ply"), "--output", str(atlas_dir), "--device", device,
                  "--tris", str(params["atlas_tris"]), "--res", str(params["atlas_res"]), "--pad", str(params["atlas_pad"]),
                  "--chart-smooth", str(params["chart_smooth"]), "--protect-head", "--head-radius", str(params["head_radius"])]
        facing = camera_facing(cameras[anchor])
        unwrap += ["--facing", ",".join(f"{v:.6f}" for v in facing)]
        head_centre: Optional[np.ndarray] = None
        if cap is not None:
            unwrap += ["--cap", str(cap)]
            try:
                head_centre = head_centre_from_cap(read_ply_xyz(cap), facing)
            except Exception as exc:  # noqa: BLE001 - a cap the reader cannot parse only loses the centre estimate
                logger.warning("meshify: could not read the cap for the head centre (%s)", exc)
        if head_centre is not None:
            unwrap += ["--centre", ",".join(f"{v:.6f}" for v in head_centre)]
        run(unwrap, "mesh-unwrap")

        if not params["keep_probes"]:
            shutil.rmtree(probes_dir, ignore_errors=True)
        body_ply.unlink(missing_ok=True)
        atlas_meta = json.loads((atlas_dir / "atlas.json").read_text()) if (atlas_dir / "atlas.json").is_file() else {}
        stats = {
            "cameras": {name: (len(cameras) if name == "train" else len(orbit["cameras"]) if name == "orbit" else len(extra)) for name, _ in sets},
            "cap": str(cap) if cap else None, "face_geometry": params["face_geometry"] if cap else "splat",
            "refined": geom.name == "mesh_geom.ply", "anchor_frame_index": anchor,
            "head_centre": head_centre.tolist() if head_centre is not None else None, "facing": facing.tolist(),
            "atlas": atlas_meta, "seconds": round(time.time() - t0, 1),
        }
        record.update(stats)
        (out / "meshify.json").write_text(json.dumps(record, indent=1))
        logger.info("meshify: %s in %.0fs (atlas %s tris, %s charts covered %s texels)", atlas_dir, stats["seconds"],
                    atlas_meta.get("tris"), atlas_meta.get("uv_verts"), atlas_meta.get("covered"))
        return {"mesh_dir": str(atlas_dir), "mesh_stats": stats}
