"""View-manipulation steps: drop / filter / rotate / replace / merge.

Native ports of five ComfyUI nodes that do pure dataset surgery — no models,
no GPU, no graphics context:

    nodes/drop_views_node.py      -> drop_views
    nodes/filter_fov_node.py      -> filter_fov
    nodes/rotate_views_node.py    -> rotate_views
    nodes/replace_views_node.py   -> replace_views
    nodes/merge_dataset_node.py   -> merge_datasets

`workflows/api/outline.json` chains filter_fov -> rotate_views ->
replace_views, and `resplat_tiered.json` needs merge_datasets to combine
three circular orbits rendered at different framings, so none of the
ComfyUI pipeline YAMLs can be reproduced without these.

Unlike every other step here, these take and return a whole `Dataset`
rather than loose arrays: each one has to keep cameras, images, masks and
image_names moving together, and splitting that across four Context paths
would make it trivially easy to desynchronise them in a workflow YAML.
`save_dataset` already takes a whole Dataset the same way.

**Azimuth convention** (shared by filter_fov and rotate_views, and the
thing most likely to be got wrong if these are ever re-derived): angles are
measured relative to the *skeleton's front*, not to world axes and not to
the first frame. The orbit azimuth of a camera is
`degrees(arctan2(dx, dz))` about `extras["orbit_target"]` — arctan2 of x
over z, matching body2colmap's `cartesian_to_spherical`, so azimuth 0 is
the +Z direction — and the skeleton-relative azimuth subtracts
`extras["forward_azimuth_deg"]` from it. Net result: 0 = front, +90 =
right, -90 = left, ±180 = back.

Both fields come from the `render` step (and are persisted through
`Dataset.to_disk()` in `b2c_extras`). A dataset produced by
`merge_datasets` deliberately carries neither — cameras from different
orbits have no single orbit center — so filter_fov/rotate_views raise on
merged datasets rather than silently computing nonsense, matching the
ComfyUI nodes' behaviour.

Verified locally against `cyber_6f`'s real 81-camera helical orbit (no pod
needed) — see tests/test_views.py.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dataset import Dataset
from ..registry import register_step
from ..step import Step

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _normalize_angle(deg: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return ((deg + 180.0) % 360.0) - 180.0


def _require_orbit_metadata(dataset: Dataset, step_name: str) -> Tuple[np.ndarray, float]:
    """Pull orbit_target / forward_azimuth_deg out of a Dataset's extras.

    Raises with an actionable message when absent, which is the expected
    outcome for a merged dataset (merge_datasets drops both by design).
    """
    orbit_target = dataset.extras.get("orbit_target")
    if orbit_target is None:
        raise ValueError(
            f"{step_name} requires extras['orbit_target'], which this dataset "
            "does not have. It is written by the render step and survives "
            "Dataset.to_disk() in b2c_extras; merge_datasets drops it "
            "deliberately, so a merged dataset cannot be filtered or rotated."
        )
    forward_azimuth_deg = float(dataset.extras.get("forward_azimuth_deg", 0.0))
    return np.asarray(orbit_target, dtype=np.float64), forward_azimuth_deg


def _relative_azimuths(
    cameras: Sequence[Any], orbit_target: np.ndarray, forward_azimuth_deg: float
) -> List[float]:
    """Each camera's azimuth in degrees relative to the skeleton's front.

    See the module docstring for the convention. Returns values in
    (-180, 180], where 0 = front.
    """
    azimuths = []
    for cam in cameras:
        dx = float(cam.position[0]) - float(orbit_target[0])
        dz = float(cam.position[2]) - float(orbit_target[2])
        orbit_az = float(np.degrees(np.arctan2(dx, dz)))
        azimuths.append(_normalize_angle(orbit_az - forward_azimuth_deg))
    return azimuths


def _reindex(dataset: Dataset, order: Sequence[int], renumber: bool) -> Dataset:
    """Build a new Dataset from `dataset` keeping only `order`, in that order.

    `renumber` regenerates frame_NNNNN_.png names sequentially (what the
    rotate/merge nodes do, since position in the sequence is what those
    change); dropping/filtering keeps the original names so a view stays
    traceable to the frame it came from, matching the ComfyUI nodes.
    """
    images = [dataset.images[i] for i in order]
    cameras = [dataset.cameras[i] for i in order]
    masks = [dataset.masks[i] for i in order] if dataset.masks is not None else None

    if renumber:
        image_names = [f"frame_{j + 1:05d}_.png" for j in range(len(order))]
    else:
        image_names = [dataset.image_names[i] for i in order]

    return Dataset(
        images=images,
        image_names=image_names,
        cameras=cameras,
        points_3d=dataset.points_3d,
        resolution=dataset.resolution,
        masks=masks,
        reference_image=dataset.reference_image,
        anchor_image=dataset.anchor_image,
        prompt=dataset.prompt,
        splat_path=dataset.splat_path,
        extras=dict(dataset.extras),
    )


def parse_view_indices(spec: str) -> set:
    """Parse a 1-based view spec into a set of indices.

        "1,2,3"      -> {1, 2, 3}
        "9-40"       -> {9, ..., 40}
        "1,2,3,9-40" -> {1, 2, 3, 9, ..., 40}
    """
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}': start ({start}) is greater than end ({end})"
                )
            indices.update(range(start, end + 1))
        elif re.fullmatch(r"\d+", part):
            indices.add(int(part))
        else:
            raise ValueError(
                f"Invalid view specification '{part}'. Use comma-separated "
                "integers and ranges, e.g. '1,2,3,9-40'"
            )
    return indices


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

@register_step("drop_views")
class DropViewsStep(Step):
    """Remove views by 1-based index.

    inputs:  {"dataset": Dataset}
    params:  views_to_drop (str, e.g. "1,2,3,9-40")
    outputs: {"dataset": Dataset}
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        dataset: Dataset = inputs["dataset"]
        spec = str(params.get("views_to_drop", "") or "")

        drop_indices = parse_view_indices(spec)
        if not drop_indices:
            logger.info("drop_views: no views specified, passing through unchanged")
            return {"dataset": dataset}

        n_views = len(dataset.cameras)
        out_of_range = {i for i in drop_indices if i < 1 or i > n_views}
        if out_of_range:
            raise ValueError(
                f"View indices out of range (dataset has {n_views} views): "
                f"{sorted(out_of_range)}"
            )

        keep = [i for i in range(n_views) if (i + 1) not in drop_indices]
        if not keep:
            raise ValueError(
                f"Cannot drop all {n_views} views — at least one must remain"
            )

        logger.info(
            "drop_views: dropping %d of %d views, %d remaining",
            n_views - len(keep), n_views, len(keep),
        )
        return {"dataset": _reindex(dataset, keep, renumber=False)}


@register_step("filter_fov")
class FilterFoVStep(Step):
    """Keep only views whose azimuth falls inside a field-of-view cone.

    inputs:  {"dataset": Dataset}
    params:  azimuth_deg (center, default 0 = front),
             fov_deg (total width, default 180 — views within ±fov/2 are kept)
    outputs: {"dataset": Dataset}

    Requires extras["orbit_target"]; see the module docstring on the
    azimuth convention and on why merged datasets are rejected.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        dataset: Dataset = inputs["dataset"]
        azimuth_deg = float(params.get("azimuth_deg", 0.0))
        fov_deg = float(params.get("fov_deg", 180.0))

        orbit_target, forward_azimuth_deg = _require_orbit_metadata(dataset, "filter_fov")
        rel_azimuths = _relative_azimuths(dataset.cameras, orbit_target, forward_azimuth_deg)

        half_fov = fov_deg / 2.0
        center = _normalize_angle(azimuth_deg)
        keep = [
            i for i, az in enumerate(rel_azimuths)
            if abs(_normalize_angle(az - center)) <= half_fov
        ]

        if not keep:
            raise ValueError(
                f"No views fall within azimuth={azimuth_deg}° ± {half_fov}°. "
                f"Camera azimuths range from {min(rel_azimuths):.1f}° to "
                f"{max(rel_azimuths):.1f}° (relative to skeleton front)."
            )

        logger.info(
            "filter_fov: azimuth=%.1f° fov=%.1f° -> keeping %d/%d views",
            center, fov_deg, len(keep), len(dataset.cameras),
        )
        return {"dataset": _reindex(dataset, keep, renumber=False)}


@register_step("rotate_views")
class RotateViewsStep(Step):
    """Cyclically reorder views so frame_00001 sits at a given azimuth.

    inputs:  {"dataset": Dataset}
    params:  start_azimuth_deg (absolute, relative to skeleton front)
    outputs: {"dataset": Dataset}

    The parameter is an *absolute* azimuth, not an offset, so applying this
    twice with the same value is idempotent. Camera/image pairing is
    preserved and image names are renumbered sequentially.

    Twin handling: when a second camera sits within 0.75 of the angular
    step of the target azimuth (the `overlap=1` circular case, where the
    first and last frame share a position), the pair is split to
    frame_00001 and frame_N so they are as far apart as possible in the
    sequence — which is what a FirstLast diffusion pass wants.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        dataset: Dataset = inputs["dataset"]
        start_azimuth_deg = float(params.get("start_azimuth_deg", 0.0))

        n_views = len(dataset.cameras)
        if n_views == 0:
            raise ValueError("Cannot rotate an empty dataset")

        orbit_target, forward_azimuth_deg = _require_orbit_metadata(dataset, "rotate_views")
        azimuths = _relative_azimuths(dataset.cameras, orbit_target, forward_azimuth_deg)

        target = _normalize_angle(start_azimuth_deg)
        diffs = [abs(_normalize_angle(az - target)) for az in azimuths]
        ranked = sorted(range(n_views), key=lambda i: diffs[i])
        best_idx = ranked[0]

        angular_step = 360.0 / n_views
        twin_idx = None
        if n_views >= 2 and diffs[ranked[1]] <= angular_step * 0.75:
            twin_idx = ranked[1]

        order = [(best_idx + i) % n_views for i in range(n_views)]
        if twin_idx is not None:
            order.remove(twin_idx)
            order.append(twin_idx)
            logger.info(
                "rotate_views: start_azimuth=%.1f° -> view %d (az %.1f°) first, "
                "twin view %d (az %.1f°) last (split for FirstLast)",
                start_azimuth_deg, best_idx + 1, azimuths[best_idx],
                twin_idx + 1, azimuths[twin_idx],
            )
        elif order == list(range(n_views)):
            logger.info(
                "rotate_views: start_azimuth=%.1f° -> view 1 (az %.1f°) already "
                "first, no change", start_azimuth_deg, azimuths[0],
            )
            return {"dataset": dataset}
        else:
            logger.info(
                "rotate_views: start_azimuth=%.1f° -> view %d (az %.1f°) becomes "
                "frame_00001", start_azimuth_deg, best_idx + 1, azimuths[best_idx],
            )

        return {"dataset": _reindex(dataset, order, renumber=True)}


@register_step("replace_views")
class ReplaceViewsStep(Step):
    """Swap in views from a second dataset wherever the cameras coincide.

    inputs:  {"dataset": Dataset (base), "replacement": Dataset}
    params:  tolerance_pct (default 0.1 — percent of scene scale)
    outputs: {"dataset": Dataset}

    For every base camera the nearest replacement camera is found; if the
    distance is within `tolerance_pct` percent of the scene scale (the
    bounding-box diagonal of the *base* camera positions), that view's
    camera, image and mask are replaced. Output is always the base set's
    size and order. Multiple base cameras at the same location each match
    independently, so an `overlap=1` orbit's duplicated first/last frame
    both get replaced.

    Typical use (`workflows/api/outline.json`): re-render a subset of views
    from a trained splat, then merge them back into the original orbit.
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        base: Dataset = inputs["dataset"]
        repl: Dataset = inputs["replacement"]
        tolerance_pct = float(params.get("tolerance_pct", 0.1))

        base_pos = np.stack([np.asarray(c.position, dtype=np.float64) for c in base.cameras])
        repl_pos = np.stack([np.asarray(c.position, dtype=np.float64) for c in repl.cameras])

        scale = _scene_scale(base_pos)
        threshold = (tolerance_pct / 100.0) * scale

        matches: Dict[int, Tuple[int, float]] = {}
        for b_idx in range(len(base.cameras)):
            dists = np.linalg.norm(repl_pos - base_pos[b_idx], axis=1)
            r_idx = int(np.argmin(dists))
            d = float(dists[r_idx])
            if d <= threshold:
                matches[b_idx] = (r_idx, d)

        if not matches:
            logger.warning(
                "replace_views: no cameras matched within tolerance %.4f%% "
                "(threshold=%.6f, scale=%.6f); returning base unchanged",
                tolerance_pct, threshold, scale,
            )
            return {"dataset": base}

        logger.info(
            "replace_views: replacing %d/%d views (tolerance=%.4f%%, threshold=%.6f)",
            len(matches), len(base.cameras), tolerance_pct, threshold,
        )

        images = list(base.images)
        cameras = list(base.cameras)
        masks = list(base.masks) if base.masks is not None else None

        for b_idx, (r_idx, _d) in matches.items():
            images[b_idx] = repl.images[r_idx]
            cameras[b_idx] = repl.cameras[r_idx]
            if masks is not None and repl.masks is not None:
                masks[b_idx] = repl.masks[r_idx]

        out = Dataset(
            images=images,
            image_names=list(base.image_names),
            cameras=cameras,
            points_3d=base.points_3d,
            resolution=base.resolution,
            masks=masks,
            reference_image=base.reference_image,
            anchor_image=base.anchor_image,
            prompt=base.prompt,
            splat_path=base.splat_path,
            extras=dict(base.extras),
        )
        return {"dataset": out}


@register_step("merge_datasets")
class MergeDatasetsStep(Step):
    """Concatenate two or more datasets into one.

    inputs:  {"datasets": List[Dataset]} or {"dataset_1": ..., "dataset_2": ...}
    params:  pointcloud_mode ("first" | "merge" | "resample"),
             pointcloud_samples (int, resample only)
    outputs: {"dataset": Dataset}

    Frames are renumbered sequentially across the whole merged set. All
    datasets must share a resolution.

    The merged dataset intentionally drops orbit_target /
    forward_azimuth_deg / framing_bounds: cameras now come from several
    orbits and there is no single orbit center that would make an azimuth
    meaningful. filter_fov and rotate_views therefore refuse to run on the
    result, which is the ComfyUI behaviour too (and the reason those nodes'
    error messages mention merged datasets).
    """

    def run(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        datasets = _collect_datasets(inputs)
        if len(datasets) < 2:
            raise ValueError(
                f"merge_datasets needs at least 2 datasets, got {len(datasets)}"
            )

        pointcloud_mode = str(params.get("pointcloud_mode", "first"))
        pointcloud_samples = int(params.get("pointcloud_samples", 10000))

        first_resolution = tuple(datasets[0].resolution)
        for idx, ds in enumerate(datasets):
            if tuple(ds.resolution) != first_resolution:
                raise ValueError(
                    f"Dataset {idx + 1} has resolution {tuple(ds.resolution)}, but "
                    f"the first has {first_resolution}. All must match."
                )

        images: List[np.ndarray] = []
        cameras: List[Any] = []
        masks: Optional[List[np.ndarray]] = [] if all(d.masks is not None for d in datasets) else None
        for ds in datasets:
            images.extend(ds.images)
            cameras.extend(ds.cameras)
            if masks is not None:
                masks.extend(ds.masks)

        image_names = [f"frame_{i + 1:05d}_.png" for i in range(len(images))]
        points_3d = _merge_pointclouds(datasets, pointcloud_mode, pointcloud_samples)

        extras = dict(datasets[0].extras)
        for key in ("orbit_target", "forward_azimuth_deg", "framing_bounds",
                    "anchor_frame_index"):
            extras.pop(key, None)

        logger.info(
            "merge_datasets: %d datasets -> %d views, %d points (mode=%s)",
            len(datasets), len(images), len(points_3d[0]), pointcloud_mode,
        )

        out = Dataset(
            images=images,
            image_names=image_names,
            cameras=cameras,
            points_3d=points_3d,
            resolution=first_resolution,
            masks=masks,
            reference_image=datasets[0].reference_image,
            anchor_image=datasets[0].anchor_image,
            prompt=datasets[0].prompt,
            splat_path=datasets[0].splat_path,
            extras=extras,
        )
        return {"dataset": out}


def _scene_scale(positions: np.ndarray) -> float:
    """Characteristic scale: bounding-box diagonal of the camera positions."""
    if len(positions) < 2:
        return 1.0
    diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    return diag if diag > 0 else 1.0


def _collect_datasets(inputs: Dict[str, Any]) -> List[Dataset]:
    """Accept either a list under "datasets" or dataset_1..dataset_N keys.

    The numbered form mirrors the ComfyUI node's dynamic inputs (see
    web/merge_dataset.js); the list form is what a workflow YAML would
    naturally wire from a single Context path.
    """
    if "datasets" in inputs:
        return list(inputs["datasets"])

    numbered = []
    i = 1
    while f"dataset_{i}" in inputs:
        numbered.append(inputs[f"dataset_{i}"])
        i += 1
    if numbered:
        return numbered
    raise KeyError(
        "merge_datasets expects either inputs['datasets'] (a list) or "
        "inputs['dataset_1'], inputs['dataset_2'], ..."
    )


def _merge_pointclouds(
    datasets: Sequence[Dataset], mode: str, n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "first":
        return datasets[0].points_3d

    positions = np.concatenate([ds.points_3d[0] for ds in datasets], axis=0)
    colors = np.concatenate([ds.points_3d[1] for ds in datasets], axis=0)

    if mode == "merge":
        return positions, colors

    if mode == "resample":
        total = len(positions)
        if total <= n_samples:
            return positions, colors
        indices = np.random.choice(total, size=n_samples, replace=False)
        return positions[indices], colors[indices]

    raise ValueError(
        f"Unknown pointcloud_mode: {mode!r} (expected 'first', 'merge' or 'resample')"
    )
