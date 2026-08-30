"""Fetch every checkpoint up front, and let a run block until they arrive.

The pipeline's default is lazy: each step downloads what it needs inside
`Step.load()`, the first time a run reaches it. That is fine on a
workstation and poor on a rented pod, where it means a run that appears to
be progressing stalls for twenty minutes in the middle of stage 4 while
Wan2.2 comes down — and if the download fails (a gated repo the token can't
reach, a full disk), you find out an hour into a run that has already
burned an hour of GPU time on the stages before it.

So: pull everything at pod start, and gate `run` on the subset the chosen
workflow actually needs.

Two rules this module is built around.

**The prefetch uses the same code path as the step.** Every entry below
calls the same downloader, against the same cache location, that the step
itself will call — `snapshot_download` into `HF_HOME` for the
huggingface_hub ones, seedvr2's own vendored `download_weight` into the
same `model_dir`, `face_landmarks._ensure_model` for MediaPipe's files. A
prefetch that fetched the same bytes by a different route would prove
nothing about whether the step can find them, and "ready" has to mean the
step will not download anything.

**Blocking is scoped to the workflow, prefetching is not.** Waiting for the
~47 GB of Wan2.2 weights before a run whose denoise passes are switched
off would be actively wrong. The prefetch is greedy by default;
`required_for_steps()` decides what a given run must actually wait on —
fed from `WorkflowSpec.enabled_steps()`, so a `when:`-skipped step's
checkpoint is not waited on either.

Readiness is a marker file per model under `$B2C_MODELS_DIR/.ready/`,
written only after a fetch returns successfully. It is what makes a warm
volume start instantly instead of re-walking every repo's revisions, and it
is why a network volume reused across pods only pays for this once. Pass
`force=True` (or `B2C_PREFETCH_FORCE=1`) to re-verify against the network
if you suspect a cache was half-deleted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .paths import models_dir

logger = logging.getLogger(__name__)

PENDING = "pending"
FETCHING = "fetching"
READY = "ready"
FAILED = "failed"


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSource:
    key: str
    description: str
    steps: tuple           # step names that cannot run without this
    fetch: Callable[[], str]
    #: Is it already on the volume? Must be purely local — no network, no
    #: GPU, cheap enough to call on every `run`. See `is_ready`.
    probe: Callable[[], bool]
    approx_gb: float = 0.0
    gated: bool = False


def _hf_snapshot(repo: str, allow_patterns: Optional[Sequence[str]] = None):
    """A (fetch, probe) pair for one HF repo, both via snapshot_download.

    The probe is the same call as the fetch with `local_files_only=True`,
    which is huggingface_hub's own answer to "is every file of this repo
    already in the cache" — it raises if anything is missing. Asking the
    loader rather than counting files by hand means the probe cannot drift
    from what the step will actually need at load time.
    """
    patterns = list(allow_patterns) if allow_patterns else None

    def fetch() -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo, allow_patterns=patterns)

    def probe() -> bool:
        from huggingface_hub import snapshot_download

        try:
            snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
            return True
        except Exception:
            return False

    return fetch, probe


def _fetch_wan22_loras() -> str:
    from huggingface_hub import hf_hub_download

    from .steps.wan22_vace_denoise import (
        DEFAULT_LORA_HIGH, DEFAULT_LORA_LOW, DEFAULT_LORA_REPO, DEFAULT_LORA_SUBFOLDER,
    )

    paths = [
        hf_hub_download(DEFAULT_LORA_REPO, name, subfolder=DEFAULT_LORA_SUBFOLDER)
        for name in (DEFAULT_LORA_HIGH, DEFAULT_LORA_LOW)
    ]
    return str(Path(paths[0]).parent)


def _probe_wan22_loras() -> bool:
    from huggingface_hub import hf_hub_download

    from .steps.wan22_vace_denoise import (
        DEFAULT_LORA_HIGH, DEFAULT_LORA_LOW, DEFAULT_LORA_REPO, DEFAULT_LORA_SUBFOLDER,
    )

    try:
        for name in (DEFAULT_LORA_HIGH, DEFAULT_LORA_LOW):
            hf_hub_download(
                DEFAULT_LORA_REPO, name,
                subfolder=DEFAULT_LORA_SUBFOLDER, local_files_only=True,
            )
        return True
    except Exception:
        return False


def _fetch_wan22_fp8() -> str:
    """The two pre-quantized fp8 experts, 17.58 GB each.

    Through the step's own `resolve_fp8_checkpoint`, so the prefetch cannot
    disagree with the step about which repo, which filenames, or which
    cache — the rule this module is built on. These are plain
    `hf_hub_download`s of two files rather than a snapshot: the repo holds
    the whole Wan 2.2 fp8 family (T2V, I2V, VACE, several variants of
    each), and we want exactly two of them.
    """
    from .steps.wan22_vace_denoise import (
        DEFAULT_FP8_HIGH, DEFAULT_FP8_LOW, DEFAULT_FP8_REPO, resolve_fp8_checkpoint,
    )

    paths = [
        resolve_fp8_checkpoint(name, DEFAULT_FP8_REPO)
        for name in (DEFAULT_FP8_HIGH, DEFAULT_FP8_LOW)
    ]
    return str(Path(paths[0]).parent)


def _probe_wan22_fp8() -> bool:
    from .steps.wan22_vace_denoise import (
        DEFAULT_FP8_HIGH, DEFAULT_FP8_LOW, DEFAULT_FP8_REPO, resolve_fp8_checkpoint,
    )

    try:
        for name in (DEFAULT_FP8_HIGH, DEFAULT_FP8_LOW):
            resolve_fp8_checkpoint(name, DEFAULT_FP8_REPO, local_files_only=True)
        return True
    except Exception:
        return False


def _fetch_mediapipe() -> str:
    from .steps.face_landmarks import (
        DETECTOR_MODEL_NAME, DETECTOR_MODEL_URL, LANDMARKER_MODEL_NAME,
        LANDMARKER_MODEL_URL, _ensure_model, _model_path,
    )

    _ensure_model(LANDMARKER_MODEL_URL, _model_path(LANDMARKER_MODEL_NAME))
    _ensure_model(DETECTOR_MODEL_URL, _model_path(DETECTOR_MODEL_NAME))
    return str(_model_path(LANDMARKER_MODEL_NAME).parent)


def _probe_mediapipe() -> bool:
    from .steps.face_landmarks import (
        DETECTOR_MODEL_NAME, LANDMARKER_MODEL_NAME, _model_path,
    )

    return all(
        _model_path(name).exists()
        for name in (LANDMARKER_MODEL_NAME, DETECTOR_MODEL_NAME)
    )


def _fetch_moge() -> str:
    """The single `model.pt` MoGe-2's `from_pretrained` pulls for sam3d_body.

    `MoGeModel.from_pretrained(repo)` (moge.model.v2) does exactly one
    `hf_hub_download(repo, "model.pt")` — not a snapshot — so the prefetch
    does the same call against the same cache, and "ready" means
    sam3d_body's `_MoGeFOVEstimator` will not touch the network.
    """
    from huggingface_hub import hf_hub_download

    from .steps.sam3d_body import DEFAULT_FOV_CHECKPOINT_REPO

    return hf_hub_download(DEFAULT_FOV_CHECKPOINT_REPO, "model.pt")


def _probe_moge() -> bool:
    from huggingface_hub import hf_hub_download

    from .steps.sam3d_body import DEFAULT_FOV_CHECKPOINT_REPO

    try:
        hf_hub_download(DEFAULT_FOV_CHECKPOINT_REPO, "model.pt", local_files_only=True)
        return True
    except Exception:
        return False


def _probe_seedvr2() -> bool:
    """Glob rather than import the vendored registry for the filenames.

    Importing `src.utils.model_registry` to learn DEFAULT_DIT/DEFAULT_VAE
    costs a subprocess into venv_seedvr2 (it runs a GPU-backend check on
    import), and this is called on every `run`. A glob is a heuristic, but
    only for the marker-less case — `download_weight` re-verifies a sha256
    per file when the fetch actually runs, so a wrong answer here costs a
    revalidation, not a corrupt model.
    """
    target = models_dir() / "SEEDVR2"
    return (
        any(target.glob("seedvr2_ema_*.safetensors"))
        and any(target.glob("ema_vae_*.safetensors"))
    )


def _fetch_seedvr2() -> str:
    """Runs in venv_seedvr2, because that is where the step will run it.

    The vendored downloader validates a sha256 per file and knows the
    filename->repo mapping; reimplementing either here would be a second
    source of truth for what "downloaded correctly" means.
    """
    from .proc import stream_command

    target = models_dir() / "SEEDVR2"
    script = (
        "from src.utils.downloads import download_weight\n"
        "from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE\n"
        f"ok = download_weight(dit_model=DEFAULT_DIT, vae_model=DEFAULT_VAE, model_dir={str(target)!r})\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    stream_command(
        ["/opt/venv_seedvr2/bin/python", "-c", script],
        log_name="prefetch.seedvr2",
        not_found_hint="Expected inside the container image; run this there.",
    )
    return str(target)


#: Everything `wan22_vace_denoise` loads from the base diffusers repo —
#: which is everything EXCEPT the two transformers. Those now come
#: pre-quantized from a different repo (see `_fetch_wan22_fp8`), and
#: diffusers does not download a component that is passed to
#: `from_pretrained` directly, so `transformer/*` and `transformer_2/*`
#: (34.68 GB EACH) must not be listed here: they would be 69 GB of pod
#: download and volume for weights nothing opens.
#:
#: `transformer/config.json` is the deliberate exception and is NOT
#: redundant with dropping `transformer/*`: pipeline/wan_fp8.py needs the
#: model geometry to instantiate WanVACETransformer3DModel, and fetches it
#: with `hf_hub_download(repo, "config.json", subfolder="transformer")` —
#: the same cache layout snapshot_download writes, so this pattern is what
#: makes that call a cache hit instead of a network round trip. Kilobytes.
WAN22_ALLOW_PATTERNS = [
    "transformer/config.json", "vae/*",
    "text_encoder/*", "tokenizer/*", "scheduler/*", "model_index.json",
]


def _registry() -> List[ModelSource]:
    """Built lazily so importing this module never imports a step module."""
    from .steps.rmbg import DEFAULT_CHECKPOINT as RMBG
    from .steps.sam3d_body import DEFAULT_CHECKPOINT_REPO as SAM3D
    from .steps.sam3d_body import DEFAULT_FOV_CHECKPOINT_REPO as MOGE
    from .steps.sapiens2 import DEFAULT_CHECKPOINT as SAPIENS
    from .steps.wan22_vace_denoise import DEFAULT_CHECKPOINT as WAN22

    # allow_patterns, or these pull the whole repo including formats the
    # pipeline never loads. Measured against the live repos:
    #   RMBG-2.0 is 5.37 GB unfiltered but 0.88 GB of it is used — the rest
    #     is eight ONNX variants and a duplicate pytorch_model.bin. The
    #     `*.py` are NOT optional: the step loads it with
    #     trust_remote_code=True, which needs birefnet.py/BiRefNet_config.py.
    #   sapiens2-normal-* ships the same weights twice (model.safetensors
    #     and sapiens2_<size>_<task>.safetensors, 6.16 GB each on the 1b
    #     default); only the first is what from_pretrained loads.
    # 6.3 GB of pod download and volume, for nothing, on every fresh pod.
    #
    # The pointmap and segmentation heads used to be pulled here too — a
    # third and fourth Sapiens2 repo, ~19 GB of Sapiens2 before anything
    # else on a fresh pod. Both went with the photo-to-splat work on
    # 2026-08-30; nothing left in the pipeline loads either.
    _WEIGHTS_ONLY = ["*.json", "*.py", "model.safetensors"]
    rmbg_fetch, rmbg_probe = _hf_snapshot(RMBG, _WEIGHTS_ONLY)
    sapiens_fetch, sapiens_probe = _hf_snapshot(SAPIENS, _WEIGHTS_ONLY)
    sam3d_fetch, sam3d_probe = _hf_snapshot(SAM3D)
    wan22_fetch, wan22_probe = _hf_snapshot(WAN22, WAN22_ALLOW_PATTERNS)

    return [
        ModelSource(
            "rmbg", f"{RMBG} (background removal)", ("rmbg",),
            rmbg_fetch, rmbg_probe, approx_gb=0.9, gated=True,  # accurate once filtered
        ),
        ModelSource(
            # 6.16 GB at the 1b default, measured from the cached blob with
            # the `model.safetensors`-only filter above (the repo ships the
            # same weights twice). Was 3.5 at the 0.8b default.
            "sapiens2", f"{SAPIENS} (normal maps)", ("sapiens2_lite",),
            sapiens_fetch, sapiens_probe, approx_gb=6.2,
        ),
        ModelSource(
            "sam3dbody", f"{SAM3D} (body reconstruction)",
            ("sam3d_body",),
            sam3d_fetch, sam3d_probe, approx_gb=2.8, gated=True,
        ),
        ModelSource(
            # sam3d_body's fov_estimator defaults to "moge2", so a run with
            # that step now blocks on this too — a single model.pt, ~1.3 GB
            # for the ViT-L variant (approximate; not yet measured on a pod).
            "moge2", f"{MOGE} (sam3d_body FOV / focal-length estimation)",
            ("sam3d_body",), _fetch_moge, _probe_moge, approx_gb=1.3,
        ),
        ModelSource(
            "wan22", f"{WAN22} (denoise: vae, text encoder, scheduler)",
            ("wan22_vace_denoise",),
            # 11.89 GB, measured from the repo with exactly the
            # allow_patterns above: text_encoder 11.36 + vae 0.51 + the
            # tokenizer/scheduler/config JSONs. It said 81 while the bf16
            # transformers (34.68 each) were still being pulled from here;
            # they are not any more. This number is what the prefetch UI
            # tells you a fresh pod is about to download, so it has to
            # track the patterns.
            wan22_fetch, wan22_probe, approx_gb=11.9,
        ),
        ModelSource(
            "wan22_fp8", "silveroxides/Wan_2.2-fp8_scaled_hybrid VACE experts (fp8)",
            ("wan22_vace_denoise",),
            # 17.58 GB per expert, measured. Together with `wan22` above
            # that is ~47 GB for a denoise-capable pod, against 81 GB when
            # the transformers came down in bf16 to be quantized on load.
            _fetch_wan22_fp8, _probe_wan22_fp8, approx_gb=35.2,
        ),
        ModelSource(
            "wan22_lora", "lightx2v/Wan2.2-Lightning distill LoRAs",
            ("wan22_vace_denoise",), _fetch_wan22_loras, _probe_wan22_loras, approx_gb=1.2,
        ),
        ModelSource(
            "seedvr2", "SeedVR2 3B fp8 DiT + VAE (upscale)", ("seedvr2",),
            _fetch_seedvr2, _probe_seedvr2, approx_gb=6.0,
        ),
        ModelSource(
            "mediapipe", "MediaPipe face landmarker + detector",
            ("detect_face_landmarks",), _fetch_mediapipe, _probe_mediapipe, approx_gb=0.01,
        ),
    ]


def registry() -> Dict[str, ModelSource]:
    return {source.key: source for source in _registry()}


def required_for_steps(step_names: Iterable[str]) -> List[str]:
    """Which model keys a workflow made of `step_names` cannot run without."""
    wanted = set(step_names)
    return [key for key, source in registry().items() if wanted & set(source.steps)]


# --------------------------------------------------------------------------
# readiness state, on the volume
# --------------------------------------------------------------------------

def _ready_dir() -> Path:
    path = models_dir() / ".ready"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_path() -> Path:
    return models_dir() / ".prefetch.json"


def is_ready(key: str) -> bool:
    """Is this model already on the volume?

    Two layers, and the second is the one that matters for a reused
    network volume:

    1. A marker file written by a previous successful fetch. Cheap, and
       the common case once a pod has run once.
    2. Failing that, **ask the loader**. `probe()` is the step's own
       loading call restricted to local files, so it answers for weights
       this code never fetched — pulled by an older image, by
       `pipeline/envs/wan22/setup.sh` by hand, or by a previous pod that
       predates the marker files entirely. Without this, a warm volume
       with no markers would look completely cold.

    A successful probe writes the marker, so the expensive path runs at
    most once per model per volume.
    """
    if (_ready_dir() / f"{key}.json").exists():
        return True

    source = registry().get(key)
    if source is None:
        return False
    try:
        if not source.probe():
            return False
    except Exception:  # a probe must never be the thing that fails a run
        logger.debug("probe for %s raised; treating as not present", key, exc_info=True)
        return False

    logger.info("%s: already on the volume (found by probe, not by this pod)", key)
    mark_ready(key, "pre-existing on the volume")
    return True


def mark_ready(key: str, location: str) -> None:
    (_ready_dir() / f"{key}.json").write_text(
        json.dumps({"key": key, "location": location, "fetched_at": time.time()}, indent=2)
    )


def clear_ready(key: str) -> None:
    (_ready_dir() / f"{key}.json").unlink(missing_ok=True)


def read_status() -> Dict[str, Dict[str, object]]:
    """The live view a UI or a waiting run polls.

    Read from a file rather than shared memory because the prefetch is a
    separate process started by the entrypoint — so it survives a UI
    restart, and a UI that starts later can still see what is happening.
    """
    known = registry()
    status = {key: {"status": READY if is_ready(key) else PENDING, "detail": ""} for key in known}
    try:
        recorded = json.loads(_status_path().read_text())
    except (OSError, ValueError):
        return status
    for key, entry in recorded.items():
        if key in status and not (status[key]["status"] == READY and entry.get("status") != FAILED):
            status[key] = entry
        elif key in status and entry.get("status") == FAILED:
            status[key] = entry
    return status


def _write_status(status: Dict[str, Dict[str, object]]) -> None:
    try:
        _status_path().write_text(json.dumps(status, indent=2))
    except OSError:
        logger.debug("could not write prefetch status to %s", _status_path())


def summary_line() -> str:
    status = read_status()
    counts: Dict[str, int] = {}
    for entry in status.values():
        counts[str(entry["status"])] = counts.get(str(entry["status"]), 0) + 1
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))


# --------------------------------------------------------------------------
# fetching, and reporting progress while it happens
# --------------------------------------------------------------------------

#: Seconds between progress lines while a model downloads.
PROGRESS_INTERVAL = float(os.environ.get("B2C_PREFETCH_PROGRESS_INTERVAL", "30"))


def _cache_roots() -> List[Path]:
    """Where downloaded bytes actually land, for measuring growth.

    The HF hub cache and $B2C_MODELS_DIR, NOT $HF_HOME wholesale: the Xet
    chunk cache lives under HF_HOME too, and chunks are written there as
    well as into the blob, so including it would double-count every byte.
    """
    roots = [Path(models_dir())]
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        roots.append(Path(HF_HUB_CACHE))
    except Exception:
        pass
    # NOT filtered on exists(): on a cold pod neither directory has been
    # created yet when the first fetch starts, so filtering here would leave
    # nothing to measure and every interval would report 0 MB/s forever.
    # _bytes_under tolerates a missing root instead.
    return roots


def _bytes_under(roots: Sequence[Path]) -> int:
    total = 0
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _: None):
            for name in filenames:
                try:
                    total += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    # A blob being written can vanish between walk and stat.
                    continue
    return total


def _size(num_bytes: float) -> str:
    """GB once there is a GB to show, MB below that — mediapipe is 10 MB and
    `0.0 GB` says nothing about whether it is moving."""
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.1f} GB"
    return f"{num_bytes / 1e6:.0f} MB"


def _humanize(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class _DownloadProgress:
    """Log "how far along, how fast" every PROGRESS_INTERVAL while a fetch runs.

    Measures growth of the cache directories rather than hooking a
    downloader, because the three fetchers in this module do not share one:
    huggingface_hub has its own tqdm, seedvr2's vendored `download_weight`
    has another, and MediaPipe's is a plain urlretrieve. Directory growth is
    the one signal all three produce.

    Two things that surprised us and are worth expecting in the output:

      * Xet writes in bursts, not a steady stream — it holds chunks in
        memory and flushes them, so an interval can legitimately report
        ~0 MB/s and the next one report double. The average since start is
        the number to trust; the instantaneous one is for spotting a stall.
      * `approx_gb` is a registry constant, not a content-length. Percentages
        can drift past 100 or stop short; they are for orientation, not
        accounting.
    """

    def __init__(self, key: str, approx_gb: float, interval: float = PROGRESS_INTERVAL):
        self._key = key
        self._approx_bytes = int(approx_gb * 1e9)
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_DownloadProgress":
        if self._interval <= 0:
            return self
        self._roots = _cache_roots()
        self._baseline = _bytes_under(self._roots)
        self._started = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        last_bytes, last_time = self._baseline, self._started
        while not self._stop.wait(self._interval):
            now = time.time()
            try:
                current = _bytes_under(self._roots)
            except Exception:
                continue
            fetched = max(0, current - self._baseline)
            recent = (current - last_bytes) / max(1e-6, now - last_time)
            average = fetched / max(1e-6, now - self._started)
            last_bytes, last_time = current, now

            line = f"{self._key}: {_size(fetched)}"
            if self._approx_bytes:
                line += f" of ~{_size(self._approx_bytes)} ({100 * fetched / self._approx_bytes:.0f}%)"
            line += f" — {recent / 1e6:.0f} MB/s now, {average / 1e6:.0f} MB/s avg"
            if self._approx_bytes and average > 1e6:
                remaining = (self._approx_bytes - fetched) / average
                if remaining > 0:
                    line += f", ETA {_humanize(remaining)}"
            logger.info("%s", line)


def prefetch(
    keys: Optional[Sequence[str]] = None,
    force: bool = False,
    stop_on_failure: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Download the named models (default: all), recording progress as it goes.

    Never raises for a single model's failure by default: a pod that cannot
    reach the gated SAM-3D-Body repo should still finish pulling the four it
    can, and still come up. The failure is recorded, and any run that needs
    that model reports it at submit time rather than an hour in.
    """
    known = registry()
    selected = list(keys) if keys else list(known)
    unknown = [key for key in selected if key not in known]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}. Known: {sorted(known)}")

    force = force or os.environ.get("B2C_PREFETCH_FORCE") == "1"
    status = read_status()
    total_gb = sum(known[key].approx_gb for key in selected)
    logger.info(
        "prefetching %d model(s), ~%.0f GB, into %s and %s",
        len(selected), total_gb, os.environ.get("HF_HOME", "the HF cache"), models_dir(),
    )

    for key in selected:
        source = known[key]
        if is_ready(key) and not force:
            logger.info("%s: already present (%s)", key, source.description)
            status[key] = {"status": READY, "detail": "cached"}
            _write_status(status)
            continue

        logger.info("%s: fetching %s (~%.1f GB)%s", key, source.description,
                    source.approx_gb, " [gated]" if source.gated else "")
        status[key] = {"status": FETCHING, "detail": source.description, "started": time.time()}
        _write_status(status)

        started = time.time()
        try:
            with _DownloadProgress(key, source.approx_gb):
                location = source.fetch()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if source.gated:
                detail += (
                    " — this repo is gated; the HF_TOKEN's account must accept the "
                    f"licence at https://huggingface.co/{source.description.split()[0]}"
                )
            logger.error("%s: FAILED after %.0fs — %s", key, time.time() - started, detail)
            status[key] = {"status": FAILED, "detail": detail}
            _write_status(status)
            if stop_on_failure:
                raise
            continue

        elapsed = time.time() - started
        if source.approx_gb and elapsed > 1:
            logger.info("%s: ready in %s (~%.0f MB/s avg) (%s)", key,
                        _humanize(elapsed), source.approx_gb * 1e9 / elapsed / 1e6, location)
        else:
            logger.info("%s: ready in %.0fs (%s)", key, elapsed, location)
        mark_ready(key, location)
        status[key] = {"status": READY, "detail": location, "seconds": round(elapsed, 1)}
        _write_status(status)

    return status


class ModelsUnavailable(RuntimeError):
    """A run needs a model that is not present and could not be fetched."""


def wait_until_ready(
    keys: Sequence[str],
    timeout: float = 7200.0,
    poll: float = 5.0,
    on_wait: Optional[Callable[[List[str]], None]] = None,
) -> None:
    """Block until every key is ready, fetching any that nobody else is.

    Three cases, and they are all reachable in normal use:

      * everything is already there — returns immediately, which is the
        warm-volume case and by far the most common;
      * the entrypoint's background prefetch is still working on them —
        waits for it, reporting through `on_wait` so a caller can say so
        rather than appearing hung;
      * nothing is fetching them and they are absent — fetches them here,
        synchronously. A run must not start without them; falling back to
        the lazy path would put the download back in the middle of the run,
        which is the thing this module exists to prevent.
    """
    missing = [key for key in keys if not is_ready(key)]
    if not missing:
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = read_status()

        failed = [k for k in missing if status.get(k, {}).get("status") == FAILED]
        if failed:
            details = "; ".join(f"{k}: {status[k].get('detail')}" for k in failed)
            raise ModelsUnavailable(f"required model(s) could not be downloaded — {details}")

        in_flight = [k for k in missing if status.get(k, {}).get("status") == FETCHING]
        still_missing = [k for k in missing if not is_ready(k)]
        if not still_missing:
            return

        if not in_flight:
            # Nobody is fetching these. Do it here rather than waiting for
            # a prefetch that is not coming.
            logger.info("required model(s) not present and not being fetched: %s",
                        ", ".join(still_missing))
            prefetch(still_missing, stop_on_failure=True)
            missing = [key for key in keys if not is_ready(key)]
            if not missing:
                return
            continue

        if on_wait:
            on_wait(still_missing)
        logger.info("waiting on model download: %s", ", ".join(in_flight))
        time.sleep(poll)

    raise ModelsUnavailable(
        f"timed out after {timeout:.0f}s waiting for: {', '.join(k for k in keys if not is_ready(k))}"
    )
