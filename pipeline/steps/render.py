"""Multi-view mesh/skeleton rendering + camera-path generation from a
SAM-3D-Body reconstruction — the step that turns sam3d_body's output into
the images/cameras/point-cloud a Dataset needs.

Ported from nodes/render_node.py in the original ComfyUI-Body2COLMAP repo,
minus the ComfyUI-specific bits (tensor conversion, batching, progress bar,
face-landmark overlay — RMBG/MediaPipe face-landmark porting is separately
tracked, see pipeline/README.md's "Not yet started" list). The actual
geometry work — camera path generation, mesh/skeleton rasterization, point
cloud sampling — is body2colmap's, not reimplemented here; this module is a
thin adapter, same as sam3d_body.py/brush.py are for their libraries.
RASTERISATION UNTESTED — pyrender needs a real GPU/headless-GL setup, so
the render loop itself has never executed; leave that for the next pod with
graphics support confirmed. The metadata this step publishes *is* verified
indirectly: steps/views.py and steps/splat.py consume orbit_target /
forward_azimuth_deg / focal_length_mm / anchor_position, and their tests
check those against cyber_6f's real recorded values (e.g. the recorded
focal_length_mm reproduces the recorded camera fx exactly).

**Key finding, not obvious from any docstring**: body2colmap's own
`Scene.from_sam3d_output()` expects the RAW SAM-3D-Body field names
(`pred_vertices`, `pred_cam_t`, `faces`, `pred_keypoints_3d`), and
`pred_keypoints_3d` specifically — NOT `pred_joint_coords`. Confirmed by
reading facebookresearch/sam-3d-body's mhr_head.py directly:
`j3d = j3d[:, :70]  # 308 --> 70 keypoints` is what becomes
`pred_keypoints_3d`, and body2colmap's `Scene._infer_skeleton_format`
maps a 70-joint array to `"mhr70"` by count. `pred_joint_coords` (127
joints in this project's own verified sam3d_body run — see
sam3d_body.py's docstring) is a *different*, larger MHR rig body2colmap
does not consume for skeleton rendering. So this step's adapter reads
`inputs["mesh_output"]["keypoints_3d"]` (sam3d_body.py's own output key
for `pred_keypoints_3d`) — not `inputs["mesh_output"]["joints"]`, which
would silently hand Scene the wrong array. The ComfyUI-Body2COLMAP repo's
own `core/sam3d_adapter.py` does its own hand-rolled version of this same
conversion (and calls it "joints" too, adding to the trap) — this module
skips that file entirely and uses `Scene.from_sam3d_output` directly,
since body2colmap already does this correctly and is the single source of
truth for the SAM-3D-Body -> Scene mapping.

**Headless rendering**: pyrender needs `PYOPENGL_PLATFORM` set to `egl` or
`osmesa` *before pyrender is ever imported* — checked once at process
start, so this happens at this module's own import time (cheap, just an
env var) rather than deferred into load()/run() like the actual pyrender
import. Mirrors ComfyUI-Body2COLMAP's __init__.py: prefer EGL (GPU) if
`libEGL.so.1` loads, else fall back to OSMesa (software, slow but always
works). Left alone if the caller already set PYOPENGL_PLATFORM.

**Output convention**: this pipeline keeps images and masks as separate
lists (cv2 BGR uint8 images, float32 [0,1] masks with foreground=1) rather
than baking alpha into a BGRA image — matches rmbg.py's own convention,
which brush.py's mask input docstring also assumes (masks: "float32 [0,1],
foreground=1"). pyrender's own alpha channel is uint8 [0,255] with the same
polarity (opaque content = high), so this step just rescales it — no
inversion needed, unlike ComfyUI's own MASK convention (1.0=background),
which the original render_node.py had to account for and this port does
not. (Dataset.to_disk/from_disk bakes masks into a 4th image channel when
*persisting* to disk, but that's a serialization detail, not the in-memory
contract between steps.)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

from ..registry import register_step
from ..step import REQUIRED, Param, Step

# Must run before pyrender is ever imported anywhere in this process —
# pyrender/OpenGL check PYOPENGL_PLATFORM at import time, not render time.
if sys.platform.startswith("linux") and "PYOPENGL_PLATFORM" not in os.environ:
    try:
        import ctypes

        ctypes.CDLL("libEGL.so.1")
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    except OSError:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

_FULL_FRAME_SENSOR_WIDTH_MM = 36.0


def _focal_length_mm_to_pixels(focal_length_mm: float, image_width: int) -> float:
    return (focal_length_mm / _FULL_FRAME_SENSOR_WIDTH_MM) * image_width


def _focal_length_pixels_to_mm(focal_length_px: float, image_width: int) -> float:
    return (focal_length_px / image_width) * _FULL_FRAME_SENSOR_WIDTH_MM


# "outline+skeleton" render mode: a flat two-tone silhouette base layer on a
# fixed mid-grey ground, with the skeleton drawn over it exactly as in the
# other "*+skeleton" modes. The silhouette fill is a single grey the
# `outline_strength` percentage picks on a linear ramp from the background
# (0% -> #7F7F7F, silhouette invisible) to black (100% -> #000000).
_OUTLINE_BG_VALUE = 0x7F  # #7F7F7F — always the background, never configurable
_OUTLINE_FULL_STRENGTH_VALUE = 0x00  # #000000 — outline_strength 100%
# body2colmap render_outline's own `blur` default, tracked here so an
# un-set workflow renders exactly as the library would.
_OUTLINE_DEFAULT_BLUR = 4
# The default strength lands the outline exactly on #6F6F6F.
_DEFAULT_OUTLINE_STRENGTH = (
    100.0
    * (_OUTLINE_BG_VALUE - 0x6F)
    / (_OUTLINE_BG_VALUE - _OUTLINE_FULL_STRENGTH_VALUE)
)


def _outline_grey(strength: float) -> float:
    """The [0,1] grey an `outline_strength` percentage maps to.

    Quantised to 8 bits, then nudged by half a level so body2colmap's
    `render_outline` — which does a plain ``int(c * 255)`` — recovers the
    intended byte exactly rather than losing one to float truncation.
    """
    frac = min(max(strength, 0.0), 100.0) / 100.0
    value = round(
        _OUTLINE_BG_VALUE * (1.0 - frac) + _OUTLINE_FULL_STRENGTH_VALUE * frac
    )
    return (value + 0.5) / 255.0


@register_step("render")
class RenderStep(Step):
    """Render a camera-path orbit of a SAM-3D-Body mesh/skeleton.

    inputs: {"mesh_output": dict — sam3d_body.py's raw run() output
             (vertices, faces, cam_t, keypoints_3d, focal_length, ...),
             "face_landmarks": Optional[dict] — steps/face_landmarks.py's
             output, drawn on the skeleton render modes}

    See body2colmap.path.OrbitPath / body2colmap.renderer.Renderer for the
    exact semantics of each param below — this step is a thin pass-through.

    outputs: {"images": List[np.ndarray] (BGR uint8), "masks":
             List[np.ndarray] (float32 [0,1], foreground=1 — see module
             docstring), "cameras": List[Camera],
             "image_names": List[str], "points_3d": (positions, colors),
             "resolution": (width, height), "orbit_target" (np.ndarray(3,)),
             "forward_azimuth_deg" (float), "focal_length_mm" (float),
             "framing_bounds" (dict), "initial_rotation" (float)} plus, when
             override_cam_from_mesh: "anchor_position" (np.ndarray(3,)),
             "anchor_frame_index" (int), "original_focal_length" (float),
             "image_warp" (dict for generate_firstlast — see
             anchor_stub.py).
    """

    # Pattern-specific params all carry defaults rather than being REQUIRED:
    # only one pattern's set is read per call, so requiring n_loops would
    # make every circular render declare a helical param. The defaults are
    # the values the shipped workflows use.
    PARAMS = (
        Param("pattern", str, "circular", "Shape of the camera path",
              choices=("circular", "sinusoidal", "helical")),
        Param("n_frames", int, REQUIRED, "How many views to render", minimum=1),
        Param("resolution", list, [720, 1280],
              "Render size as [width, height]. The shipped workflow passes "
              "${globals.resolution} straight through, so one global fixes "
              "the frame size for every stage — there is no separate "
              "width/height knob on this step to drift from it."),
        Param("render_mode", str, "depth+skeleton", "What each frame draws",
              choices=("mesh", "depth", "skeleton", "mesh+skeleton", "depth+skeleton",
                       "outline+skeleton")),
        Param("outline_strength", float, _DEFAULT_OUTLINE_STRENGTH,
              "outline+skeleton mode only: how dark the flat silhouette fill "
              "is, as a percentage. 0 matches the fixed #7F7F7F background "
              "(the silhouette disappears), 100 is solid black. The default "
              "lands the fill on #6F6F6F. Ignored by every other render_mode",
              minimum=0.0, maximum=100.0),
        Param("outline_blur", int, _OUTLINE_DEFAULT_BLUR,
              "outline+skeleton mode only: Gaussian softening of the "
              "silhouette edge, in pixels (0 leaves a hard two-tone edge). "
              "This is body2colmap's own render_outline `blur`; the skeleton "
              "overlay is composited on top afterwards and stays sharp. "
              "Ignored by every other render_mode",
              minimum=0, advanced=True),
        Param("framing", str, "full", "How much of the body fills the frame",
              choices=("full", "torso", "bust", "head")),
        Param("eye_style", str, "shape",
              "How the face overlay draws the eyes: 'shape' fills each eye as a "
              "flat sclera with a pupil disc (a stronger gaze cue at the "
              "resolutions the diffusion pass conditions on), 'dots' is the "
              "older landmark-dot rendering. Only takes effect with a "
              "face_landmarks input",
              choices=("shape", "dots")),
        Param("eye_color", list, [1.0, 1.0, 1.0],
              "RGB in [0,1] for the filled eye shape (sclera). eye_style "
              "'shape' only"),
        Param("pupil_color", list, [0.0, 0.0, 0.0],
              "RGB in [0,1] for the pupil disc. eye_style 'shape' only"),
        Param("pupil_scale", float, 0.75,
              "Pupil diameter as a fraction of the eye height measured at the "
              "pupil; 1.0 is a disc touching both lids. eye_style 'shape' only",
              minimum=0.0, maximum=1.0),
        Param("override_cam_from_mesh", bool, False,
              "Anchor one frame exactly at the original SAM-3D-Body camera, so a "
              "reference photo can be warped onto it. Circular or helical only, and "
              "it bypasses focal_length_mm/radius/start_azimuth_deg in favour of the "
              "orbit derived from that camera"),
        Param("fill_ratio", float, 0.8, "How much of the frame the subject fills",
              minimum=0.0, maximum=1.0),
        Param("focal_length_mm", float, 0.0,
              "0 means derive one from the render width. Ignored under "
              "override_cam_from_mesh"),
        Param("initial_rotation", float, 0.0,
              "Extra rotation applied after auto-orienting the body toward the "
              "camera. Ignored under override_cam_from_mesh, which must keep the "
              "mesh where the original camera saw it"),
        Param("bg_color", list, [1.0, 1.0, 1.0],
              "RGB in [0,1]. Note this does NOT paint the depth render's background, "
              "nor the outline+skeleton one (that is always the fixed #7F7F7F "
              "ground): its only other use is being published as "
              "image_warp[\"bg_color\"], the border colour generate_firstlast fills "
              "around the warped reference"),

        Param("elevation_deg", float, 0.0, "Circular: camera elevation"),
        Param("start_azimuth_deg", float, 0.0, "Where the orbit starts"),
        Param("overlap", int, 1,
              "Circular: 1 makes the first and last frame share a position"),
        Param("amplitude_deg", float, 30.0,
              "Sinusoidal/helical: elevation swing either side of the equator"),
        Param("n_cycles", int, 1, "Sinusoidal: elevation cycles over the orbit"),
        Param("n_loops", int, 2, "Helical: turns around the subject"),
        Param("lead_in_deg", float, 45.0, "Helical: azimuth spent easing in"),
        Param("lead_out_deg", float, 45.0, "Helical: azimuth spent easing out"),

        Param("radius", float, None,
              "Orbit radius; empty derives one from the framing", advanced=True),
        Param("mesh_color", list, [0.65, 0.74, 0.86], "RGB in [0,1]", advanced=True),
        Param("depth_colormap", str, "grayscale", "Depth render colour map",
              advanced=True),
        Param("skeleton_format", str, "openpose_body25_hands", "Skeleton topology",
              advanced=True),
        Param("joint_radius", float, 0.006, "Skeleton joint size", advanced=True),
        Param("bone_radius", float, 0.003, "Skeleton bone thickness", advanced=True),
        Param("face_mode", str, "full",
              "Face overlay: points plus connectivity lines, points alone, or none. "
              "Only meaningful with a face_landmarks input",
              choices=("full", "points", "none"), advanced=True),
        Param("face_max_angle", float, 90.0,
              "Skip the face overlay past this angle between the face normal and the "
              "camera: 90 is the full hemisphere, 45 near-frontal only",
              minimum=0.0, maximum=180.0, advanced=True),
        Param("pointcloud_samples", int, 10000,
              "Points sampled off the mesh for points3D.txt", minimum=1, advanced=True),
    )

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        from body2colmap.camera import Camera
        from body2colmap.path import (
            OrbitPath,
            compute_helical_anchor_params,
            compute_original_camera_orbit_params,
        )
        from body2colmap.renderer import Renderer
        from body2colmap.scene import Scene
        from body2colmap.utils import (
            compute_auto_orbit_radius,
            compute_default_focal_length,
            compute_original_view_framing,
        )

        mesh_output = inputs["mesh_output"]
        # See module docstring: pred_keypoints_3d (this step's "keypoints_3d"
        # key), NOT pred_joint_coords ("joints"), is what body2colmap's
        # skeleton rendering actually consumes.
        sam3d_dict = {
            "pred_vertices": mesh_output["vertices"],
            "pred_cam_t": mesh_output["cam_t"],
            "faces": mesh_output["faces"],
            "pred_keypoints_3d": mesh_output["keypoints_3d"],
        }
        scene = Scene.from_sam3d_output(sam3d_dict, include_skeleton=True)

        pattern = params["pattern"]
        framing = params["framing"]
        override_cam_from_mesh = params["override_cam_from_mesh"]
        resolution = params["resolution"]
        if len(resolution) != 2:
            raise ValueError(
                f"resolution must be [width, height], got {resolution!r}"
            )
        width, height = (int(v) for v in resolution)
        if min(width, height) < 1:
            raise ValueError(f"resolution must be positive, got {width}x{height}")
        render_mode = params["render_mode"]
        fill_ratio = params["fill_ratio"]

        if override_cam_from_mesh and pattern not in ("circular", "helical"):
            raise ValueError(
                f"override_cam_from_mesh only works with circular or helical "
                f"pattern, got '{pattern}'"
            )

        if not override_cam_from_mesh:
            # Auto-orient: rotate the body to face the camera at frame 0,
            # then apply any user offset. Skipped in override mode, which
            # needs the mesh's real position relative to the original
            # (origin) camera preserved.
            initial_rotation = params["initial_rotation"]
            facing = scene.compute_torso_facing_direction()
            if facing is not None:
                current_angle = float(np.arctan2(facing[0], facing[2]))
                target_angle = float(np.arctan2(0.0, -1.0))  # face -Z (toward camera)
                correction_deg = float(np.degrees(target_angle - current_angle))
            else:
                correction_deg = 0.0
            scene.rotate_around_y(correction_deg + initial_rotation)

        all_framing_bounds = {"full": scene.get_bounds()}
        if scene.skeleton_joints is not None:
            for preset in ("torso", "bust", "head"):
                try:
                    all_framing_bounds[preset] = scene.get_framing_bounds(preset=preset)
                except (ValueError, AttributeError):
                    pass
        current_bounds = all_framing_bounds.get(framing, all_framing_bounds["full"])
        orbit_center = (current_bounds[0] + current_bounds[1]) / 2.0

        image_warp = None
        anchor_frame_index = None
        original_focal_length = float(mesh_output["focal_length"]) if override_cam_from_mesh else None

        if override_cam_from_mesh:
            framing_info = compute_original_view_framing(
                vertices=scene.vertices,
                render_size=(width, height),
                original_focal_length=original_focal_length,
                fill_ratio=fill_ratio,
            )
            framed_fl = framing_info["framed_focal_length"]
            camera_template = Camera(focal_length=(framed_fl, framed_fl), image_size=(width, height))

            if pattern == "circular":
                orbit_params = compute_original_camera_orbit_params(orbit_center)
                derived_radius = float(orbit_params["radius"])
                anchor_azimuth = float(orbit_params["start_azimuth_deg"])
                derived_elevation = orbit_params["elevation_deg"]
                anchor_frame_index = 0

                path_gen = OrbitPath(target=orbit_center, radius=derived_radius)
                cameras = path_gen.circular(
                    n_frames=params["n_frames"],
                    elevation_deg=derived_elevation,
                    start_azimuth_deg=anchor_azimuth,
                    overlap=params["overlap"],
                    camera_template=camera_template,
                )
            else:  # helical
                helix_params = dict(
                    n_frames=params["n_frames"],
                    n_loops=params["n_loops"],
                    amplitude_deg=params["amplitude_deg"],
                    lead_in_deg=params["lead_in_deg"],
                    lead_out_deg=params["lead_out_deg"],
                )
                anchor_info = compute_helical_anchor_params(target=orbit_center, **helix_params)
                derived_radius = float(anchor_info["radius"])
                anchor_frame_index = int(anchor_info["anchor_frame_index"])
                anchor_azimuth = float(anchor_info["anchor_azimuth_deg"])

                path_gen = OrbitPath(target=orbit_center, radius=derived_radius)
                cameras = path_gen.helical(
                    start_azimuth_deg=anchor_info["start_azimuth_deg"],
                    elevation_offset_deg=anchor_info["elevation_offset_deg"],
                    camera_template=camera_template,
                    **helix_params,
                )

            image_warp = {
                "camera": cameras[anchor_frame_index],
                "original_focal_length": original_focal_length,
                "render_size": (width, height),
            }
            focal_length = framed_fl
        else:
            focal_length_mm = params["focal_length_mm"]
            if focal_length_mm <= 0:
                focal_length = compute_default_focal_length(width)
            else:
                focal_length = _focal_length_mm_to_pixels(focal_length_mm, width)

            radius = params["radius"]
            if radius is None:
                radius = compute_auto_orbit_radius(
                    bounds=current_bounds,
                    render_size=(width, height),
                    focal_length=focal_length,
                    fill_ratio=fill_ratio,
                )

            camera_template = Camera(focal_length=(focal_length, focal_length), image_size=(width, height))
            path_gen = OrbitPath(target=orbit_center, radius=radius)

            if pattern == "circular":
                cameras = path_gen.circular(
                    n_frames=params["n_frames"],
                    elevation_deg=params["elevation_deg"],
                    start_azimuth_deg=params["start_azimuth_deg"],
                    overlap=params["overlap"],
                    camera_template=camera_template,
                )
            elif pattern == "sinusoidal":
                cameras = path_gen.sinusoidal(
                    n_frames=params["n_frames"],
                    amplitude_deg=params["amplitude_deg"],
                    n_cycles=params["n_cycles"],
                    start_azimuth_deg=params["start_azimuth_deg"],
                    camera_template=camera_template,
                )
            elif pattern == "helical":
                cameras = path_gen.helical(
                    n_frames=params["n_frames"],
                    n_loops=params["n_loops"],
                    amplitude_deg=params["amplitude_deg"],
                    lead_in_deg=params["lead_in_deg"],
                    lead_out_deg=params["lead_out_deg"],
                    start_azimuth_deg=params["start_azimuth_deg"],
                    camera_template=camera_template,
                )
            else:
                raise ValueError(f"Unknown path pattern: {pattern}")

        mesh_color = tuple(params["mesh_color"])
        bg_color = tuple(params["bg_color"])
        if image_warp is not None:
            image_warp["bg_color"] = bg_color

        depth_colormap = params["depth_colormap"]
        depth_cmap = None if depth_colormap == "grayscale" else depth_colormap

        skeleton_format = params["skeleton_format"]
        joint_radius = params["joint_radius"]
        bone_radius = params["bone_radius"]

        # outline+skeleton: the flat fill grey the strength percentage picks,
        # and the fixed #7F7F7F ground it always sits on (which is exactly the
        # 0%-strength grey — so the two agree by construction).
        outline_fg_color = (_outline_grey(params["outline_strength"]),) * 3
        outline_bg_color = (_outline_grey(0.0),) * 3
        outline_blur = params["outline_blur"]

        # Eye appearance for the face overlay. body2colmap ignores these
        # unless a face_landmarks input makes the face visible, so they are
        # always passed rather than gated here. eye_style "shape" fills each
        # eye as a sclera plus a pupil disc; "dots" restores body2colmap's
        # older landmark-dot rendering.
        eye_opts = {
            "eye_style": params["eye_style"],
            "eye_color": tuple(params["eye_color"]),
            "pupil_color": tuple(params["pupil_color"]),
            "pupil_scale": params["pupil_scale"],
        }

        # Optional face-landmark overlay, from steps/face_landmarks.py.
        # MediaPipe's raw points are converted to OpenPose Face 70 here
        # rather than in that step, because the conversion needs the image
        # size the landmarks were normalized against — which travels with
        # them in the dict.
        openpose_face_70 = None
        face_mode = params["face_mode"]
        face_landmarks = inputs.get("face_landmarks")
        if face_landmarks is not None and face_mode != "none":
            from body2colmap.face import FaceLandmarkIngest

            source = face_landmarks["source"]
            if source != "mediapipe":
                raise ValueError(
                    f"Unsupported face landmark source: {source!r}. Supported: "
                    "'mediapipe' (see steps/face_landmarks.py)."
                )
            openpose_face_70 = FaceLandmarkIngest.from_mediapipe(
                face_landmarks["landmarks"],
                image_size=face_landmarks["image_size"],
            )
        else:
            face_mode = None
        face_max_angle = params["face_max_angle"]

        renderer = Renderer(scene=scene, render_size=(width, height))

        rendered_images = []
        for camera in cameras:
            if render_mode == "mesh":
                img = renderer.render_mesh(camera=camera, mesh_color=mesh_color, bg_color=bg_color)
            elif render_mode == "depth":
                img = renderer.render_depth(camera=camera, colormap=depth_cmap)
            elif render_mode == "skeleton":
                img = renderer.render_skeleton(
                    camera=camera,
                    target_format=skeleton_format,
                    joint_radius=joint_radius,
                    bone_radius=bone_radius,
                    bg_color=bg_color,
                    face_mode=face_mode,
                    face_landmarks=openpose_face_70,
                    face_max_angle=face_max_angle,
                    **eye_opts,
                )
            elif render_mode in ("mesh+skeleton", "depth+skeleton", "outline+skeleton"):
                composite_modes: Dict[str, Any] = {
                    "skeleton": {
                        "target_format": skeleton_format,
                        "joint_radius": joint_radius,
                        "bone_radius": bone_radius,
                    }
                }
                if render_mode == "mesh+skeleton":
                    composite_modes["mesh"] = {"color": mesh_color, "bg_color": bg_color}
                elif render_mode == "depth+skeleton":
                    composite_modes["depth"] = {"colormap": depth_cmap}
                else:  # outline+skeleton — skeleton overlay unchanged, flat grey base
                    composite_modes["outline"] = {
                        "fg_color": outline_fg_color,
                        "bg_color": outline_bg_color,
                        "blur": outline_blur,
                    }
                if face_mode is not None:
                    composite_modes["face"] = {
                        "face_mode": face_mode,
                        "face_landmarks": openpose_face_70,
                        "face_max_angle": face_max_angle,
                        **eye_opts,
                    }
                img = renderer.render_composite(camera=camera, modes=composite_modes)
            else:
                raise ValueError(f"Unknown render_mode: {render_mode}")
            rendered_images.append(img)

        points, colors = scene.get_point_cloud(n_samples=params["pointcloud_samples"])

        n = len(cameras)
        image_names = [f"frame_{i + 1:05d}_.png" for i in range(n)]
        images = [rgba[..., [2, 1, 0]] for rgba in rendered_images]  # RGB -> BGR
        masks = [rgba[..., 3].astype(np.float32) / 255.0 for rgba in rendered_images]

        # The orbit azimuth that corresponds to the front of the skeleton.
        # In override mode that is wherever the original camera ended up;
        # otherwise auto-orient has already turned the skeleton to face -Z
        # and OrbitPath's azimuth 0 is -Z, so front is 0 by construction
        # (independent of start_azimuth_deg).
        forward_azimuth_deg = float(anchor_azimuth) if override_cam_from_mesh else 0.0

        # In override mode the focal_length_mm param is bypassed entirely
        # (the render used the auto-framed focal length), so publishing the
        # raw param would make a downstream re-render silently reframe.
        effective_focal_length_mm = (
            _focal_length_pixels_to_mm(focal_length, width)
            if override_cam_from_mesh
            else params["focal_length_mm"]
        )

        result: Dict[str, Any] = {
            "images": images,
            "masks": masks,
            "cameras": cameras,
            "image_names": image_names,
            "points_3d": (points, colors),
            "resolution": (width, height),
            # Orbit metadata, matching nodes/render_node.py's b2c_data. Not
            # decoration: filter_fov and rotate_views both hard-error
            # without orbit_target/forward_azimuth_deg, and the splat
            # renderer reuses framing_bounds/initial_rotation to keep a
            # re-render framed identically. framing_bounds is in-memory
            # only — Dataset.to_disk()'s JSON filter drops it, exactly as
            # the ComfyUI save node does (see cyber_6f's b2c_extras).
            "orbit_target": np.asarray(orbit_center, dtype=np.float32),
            "forward_azimuth_deg": forward_azimuth_deg,
            "focal_length_mm": effective_focal_length_mm,
            "framing_bounds": all_framing_bounds,
            "initial_rotation": params["initial_rotation"],
        }

        if override_cam_from_mesh:
            result["anchor_position"] = np.asarray(cameras[anchor_frame_index].position, dtype=np.float32)
            result["anchor_frame_index"] = int(anchor_frame_index)
            result["original_focal_length"] = original_focal_length
            result["image_warp"] = image_warp

        return result
