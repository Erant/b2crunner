"""`python -m pipeline.cli doctor` — answer "is this pod actually able to run
the pipeline" without running the pipeline.

Every check here corresponds to something that has already gone wrong once,
on a real machine, in a way that cost an hour to diagnose from the failure
alone:

  * `vulkaninfo` finding no driver because NVIDIA_DRIVER_CAPABILITIES lacked
    `graphics` — surfaces as brush exiting mid-run, 40 minutes in.
  * `libEGL.so.1` missing, so pyrender silently fell back to OSMesa (or to
    nothing) — surfaces as `render` producing black frames or dying.
  * `brush` built from the wrong branch, missing `--normal-loss-weight` —
    surfaces as brush rejecting its argv, and only when a workflow that
    passes normal supervision runs.
  * a child venv that can't `import torch` because its .pth into venv_base
    didn't survive a stage copy.
  * a gated checkpoint 401ing because HF_TOKEN belongs to an account that
    never accepted the licence.
  * SageAttention selected but not importable, so every denoise silently
    ran on PyTorch native SDPA — the same output, an hour slower, and
    nothing in the log said so above WARNING.
  * downloads dying with "Disk quota exceeded" because HF cached to the
    container's overlay instead of the volume.

Cheap enough to run at every container start, which the entrypoint does.
Exit code is non-zero if anything FAILs, so it also works as a smoke test in
CI or a pod's start command.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_ICON = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    lines: List[str] = field(default_factory=list)


def _run(cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------

def check_environment() -> Check:
    interesting = [
        "B2C_DATA_DIR", "B2C_OUTPUT_DIR", "B2C_LOG_DIR", "B2C_MODELS_DIR", "TMPDIR",
        "HF_HOME", "HF_HUB_DISABLE_XET", "HF_XET_CACHE", "HF_XET_HIGH_PERFORMANCE",
        "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH", "PYOPENGL_PLATFORM",
    ]
    lines = [f"python {platform.python_version()} at {sys.executable}", f"host {platform.node()}"]
    for name in interesting:
        value = os.environ.get(name)
        # HF_TOKEN is deliberately absent from `interesting`; it is reported
        # as present/absent by check_huggingface and never echoed.
        lines.append(f"{name}={value}" if value else f"{name} (unset)")
    return Check("environment", OK, f"{platform.python_version()} on {platform.node()}", lines)


def check_disk() -> Check:
    from .paths import data_dir, log_dir, models_dir, output_dir

    targets = {
        "/": Path("/"),
        "data": data_dir(),
        "output": output_dir(),
        "logs": log_dir(),
        "models": models_dir(),
        "tmp": Path(os.environ.get("TMPDIR", "/tmp")),
    }
    lines, status = [], OK
    for label, path in targets.items():
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            lines.append(f"{label:7s} {path}: unreadable ({exc})")
            status = FAIL
            continue
        free_gb = usage.free / 1e9
        writable = os.access(path, os.W_OK)
        lines.append(
            f"{label:7s} {path}: {free_gb:.1f} GB free of {usage.total / 1e9:.1f} GB"
            f"{'' if writable else '  NOT WRITABLE'}"
        )
        if not writable and label != "/":
            status = FAIL
        # A brush .ply plus a COLMAP export plus a checkpointed dataset runs
        # to a few GB; under 10 free is where runs start dying halfway.
        elif free_gb < 10 and label in ("data", "output", "tmp", "models"):
            status = WARN if status == OK else status
    return Check("disk", status, "", lines)


def check_nvidia_smi() -> Check:
    if not shutil.which("nvidia-smi"):
        return Check("nvidia-smi", FAIL, "not on PATH — no GPU visible to this container")
    result = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
        "--format=csv,noheader",
    ])
    if result.returncode != 0:
        return Check("nvidia-smi", FAIL, "failed", (result.stderr or result.stdout).splitlines())
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return Check("nvidia-smi", OK, lines[0] if lines else "", lines)


def check_torch() -> Check:
    try:
        import torch
    except ImportError as exc:
        return Check("torch", FAIL, f"not importable: {exc}")

    lines = [
        f"torch {torch.__version__}, built against CUDA {torch.version.cuda}",
        f"arch list: {', '.join(torch.cuda.get_arch_list()) or 'none'}",
    ]
    if not torch.cuda.is_available():
        return Check("torch", FAIL, "torch.cuda.is_available() is False", lines)

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        lines.append(
            f"cuda:{index} {props.name} sm_{props.major}{props.minor} "
            f"{props.total_memory / 1e9:.1f} GB"
        )

    # An actual kernel launch per visible device, not just is_available():
    # the arch-list vs. real-hardware mismatch noted in
    # docs/docker-build-notes.md means the list alone is not a reliable
    # yes/no for "will this run", and on a multi-GPU pod one bad card must
    # not hide behind a matmul that only ever tried device 0.
    status = OK
    for index in range(torch.cuda.device_count()):
        try:
            a = torch.randn(64, 64, device=f"cuda:{index}")
            result = float((a @ a).sum())
            lines.append(f"real matmul on cuda:{index} OK (sum={result:.1f})")
        except Exception as exc:
            lines.append(f"real matmul on cuda:{index} FAILED: {exc}")
            status = FAIL

    if status == FAIL:
        return Check("torch", FAIL, "matmul failed on at least one device", lines)
    return Check("torch", OK, f"{torch.__version__} / cu{torch.version.cuda}", lines)


def _vulkan_chain() -> List[str]:
    """Why Vulkan failed, link by link. See scripts/vulkan_probe.sh for the
    full version — this is the subset cheap enough to run at every start."""
    import ctypes
    import glob
    import re

    lines = []

    # Link 1: did nvidia-container-toolkit mount an ICD manifest at all? If
    # not, either `graphics` was missing from the capabilities OR the host's
    # driver install has no graphics userspace to mount. Neither is fixable
    # from inside the image.
    manifests = sorted(
        glob.glob("/usr/share/vulkan/icd.d/*nvidia*.json")
        + glob.glob("/etc/vulkan/icd.d/*nvidia*.json")
    )
    if manifests:
        lines.append(f"ICD manifest: {', '.join(manifests)}")
    else:
        lines.append("ICD manifest: ABSENT — the toolkit mounted no NVIDIA ICD.")
        lines.append(f"  NVIDIA_DRIVER_CAPABILITIES={os.environ.get('NVIDIA_DRIVER_CAPABILITIES', '(unset)')}")
        lines.append("  It must include 'graphics', and it is read at container-CREATION")
        lines.append("  time — setting it inside a running pod does nothing. If it is")
        lines.append("  already set, the HOST driver has no graphics userspace to inject.")

    # Link 2: the driver's own libraries. libnvidia-gpucomp is the shader
    # compiler, split out of glcore in the 550+ drivers; a libnvidia-container
    # older than 1.17 does not know to inject it, which breaks Vulkan on
    # newer-driver hosts only.
    for pattern, label in (
        ("libGLX_nvidia.so*", "libGLX_nvidia (the ICD itself)"),
        ("libnvidia-glcore.so.*", "libnvidia-glcore"),
        ("libnvidia-gpucomp.so*", "libnvidia-gpucomp (driver >= 550)"),
    ):
        found = glob.glob(f"/usr/lib/x86_64-linux-gnu/{pattern}") + glob.glob(f"/usr/lib64/{pattern}")
        lines.append(f"{label}: {', '.join(sorted(found)) if found else 'ABSENT'}")

    # The version string's position in this line moves between driver
    # branches (the open-kernel-module builds insert "Open" and "for"), so
    # match the number rather than a field index.
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
        match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", text)
        if match:
            lines.append(f"kernel module driver version: {match.group(1)}")
            lines.append("  every injected .so above must carry this exact version")
    except OSError:
        pass

    # Link 3: the dependency that has already caught this project once. The
    # NVIDIA ICD runs a GLVND self-registration during vkCreateInstance that
    # needs libEGL.so.1 resolvable, even though Vulkan never calls EGL.
    # Without it vk_icdGetInstanceProcAddr returns NULL for vkCreateInstance,
    # with no error, and the loader falls back to llvmpipe.
    for lib in ("libEGL.so.1", "libGLdispatch.so.0", "libXext.so.6"):
        try:
            ctypes.CDLL(lib)
            lines.append(f"{lib}: loads")
        except OSError as exc:
            lines.append(f"{lib}: FAILS TO LOAD — {exc}")

    return lines


def check_vulkan() -> Check:
    """Gates `brush`. See docs/docker.md's NVIDIA_DRIVER_CAPABILITIES note."""
    if not shutil.which("vulkaninfo"):
        return Check("vulkan", WARN, "vulkaninfo not installed; cannot verify brush's backend")
    try:
        result = _run(["vulkaninfo", "--summary"], timeout=60)
    except subprocess.TimeoutExpired:
        return Check("vulkan", FAIL, "vulkaninfo timed out")

    output = result.stdout + result.stderr
    lines = [
        line.rstrip() for line in output.splitlines()
        if "deviceName" in line or "driverName" in line or "deviceType" in line
    ]

    # A software fallback is the failure mode this check exists to catch, and
    # it is NOT a non-zero exit: when the NVIDIA ICD declines to create an
    # instance the loader quietly enumerates llvmpipe and vulkaninfo succeeds.
    # Matching on "a deviceName line exists" would pass that. Require a real
    # GPU device type instead.
    gpu = "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" in output or "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" in output
    software = "llvmpipe" in output.lower() or "PHYSICAL_DEVICE_TYPE_CPU" in output

    if result.returncode != 0 or "ERROR_INCOMPATIBLE_DRIVER" in output or not lines:
        return Check(
            "vulkan", FAIL, "no Vulkan device at all — brush cannot run",
            [*_vulkan_chain(), "", "full walk: bash scripts/vulkan_probe.sh", *output.splitlines()[:10]],
        )
    if not gpu:
        detail = "Vulkan found only a software rasteriser — brush would run on the CPU" if software \
            else "Vulkan found no GPU device — brush would run on the CPU"
        return Check(
            "vulkan", FAIL, detail,
            [*lines, "", *_vulkan_chain(), "", "full walk: bash scripts/vulkan_probe.sh"],
        )
    return Check("vulkan", OK, lines[0].strip(), lines)


def check_egl() -> Check:
    """Gates `render` (pyrender). EGL is the GPU path; OSMesa is the slow one."""
    import ctypes

    lines = []
    try:
        ctypes.CDLL("libEGL.so.1")
        lines.append("libEGL.so.1 loads")
    except OSError as exc:
        return Check("egl", FAIL, "libEGL.so.1 missing — `render` falls back to software or dies", [str(exc)])

    try:
        import pyrender  # noqa: F401
    except ImportError:
        return Check("egl", WARN, "libEGL present; pyrender not installed in this venv", lines)

    try:
        import numpy as np
        import pyrender
        from OpenGL import GL

        renderer = pyrender.OffscreenRenderer(64, 64)
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        device = GL.glGetString(GL.GL_RENDERER).decode()
        renderer.delete()
        lines += [f"GL_VENDOR: {vendor}", f"GL_RENDERER: {device}"]
        if "llvmpipe" in device.lower() or "software" in device.lower():
            return Check("egl", WARN, f"software rasteriser ({device}) — `render` will be very slow", lines)
        return Check("egl", OK, device, lines)
    except Exception as exc:
        return Check("egl", FAIL, f"could not create an offscreen GL context: {exc}", lines)


def check_brush_binaries() -> Check:
    """Both binaries present, and `brush` carrying the fork's own flags.

    The flag diff is the check docker/Dockerfile's brush stage describes in
    prose: Erant/brush's `main` merely tracks upstream, so a clone of the
    default branch builds a binary that runs fine and then rejects the argv
    steps/brush.py constructs. Verified here instead of at minute 40 of a
    training run.

    It catches a stale fork build too, not just a `main` one:
    `--normal-loss-every` only landed on normal-map-supervision on
    2026-08-25, `--export-evidence` and `--normalize-masked-loss` on
    2026-08-30, and the Dockerfile's `git clone` is cached on the RUN text
    rather than on remote git state. Rebuilding does not reliably shake
    that loose — `--no-cache-filter brush-builder` re-runs the stage but
    the runtime stage's `COPY --from=brush-builder` still matches its old
    cache record, so the fresh binary is built and discarded; the
    consuming stage has to be named too. See docs/docker-build-notes.md's
    2026-08-25 update.
    """
    required_flags = [
        "--normal-loss-weight", "--normal-loss-start-iter",
        "--normal-loss-every",
        "--export-evidence", "--normalize-masked-loss",
        "--alpha-mode", "--export-name", "--total-train-iters",
    ]
    lines, status = [], OK

    for binary in ("brush", "brush-splat-render"):
        path = shutil.which(binary)
        if not path:
            lines.append(f"{binary}: NOT FOUND on PATH")
            status = FAIL
            continue
        try:
            result = _run([path, "--help"], timeout=30)
        except (subprocess.TimeoutExpired, OSError) as exc:
            lines.append(f"{binary}: {path} — could not run --help ({exc})")
            status = FAIL
            continue
        lines.append(f"{binary}: {path}")
        if binary == "brush":
            help_text = result.stdout + result.stderr
            missing = [flag for flag in required_flags if flag not in help_text]
            if missing:
                lines.append(
                    f"  MISSING FLAGS {', '.join(missing)} — this binary was built from "
                    f"Erant/brush's `main`, or from a normal-map-supervision checkout "
                    f"older than the flags above. Rebuild with "
                    f"`--no-cache-filter brush-builder,runtime`: naming brush-builder "
                    f"alone rebuilds the binary but the runtime stage's COPY keeps "
                    f"serving the cached one."
                )
                status = FAIL
            else:
                lines.append(f"  all {len(required_flags)} fork-specific flags present")

    return Check("brush binaries", status, "", lines)


def check_step_venvs(envs: Dict[str, Dict[str, Any]]) -> Check:
    """Each subprocess-dispatch env's interpreter exists and can import torch."""
    if not envs:
        return Check("step venvs", WARN, "no envs registry loaded — subprocess steps cannot run")

    lines, status = [], OK
    for name, config in sorted(envs.items()):
        python_bin = config.get("python_bin")
        if not python_bin:
            continue
        if not Path(python_bin).exists():
            lines.append(f"{name}: {python_bin} DOES NOT EXIST")
            status = FAIL
            continue
        result = _run(
            [python_bin, "-c", "import torch, pipeline; print(torch.__version__)"], timeout=180
        )
        if result.returncode != 0:
            lines.append(f"{name}: {python_bin} — import failed")
            lines += ["    " + line for line in (result.stderr or "").splitlines()[-5:]]
            status = FAIL
        else:
            lines.append(f"{name}: torch {result.stdout.strip()}")
    return Check("step venvs", status, f"{len(envs)} configured", lines)


# Run inside the wan22 venv, not here: SageAttention, Triton and diffusers
# live in that venv, and the main one has none of them. Prints one `key=value`
# per line so the parent does not have to care about import noise on stdout.
_ATTENTION_PROBE = r"""
import json, sys
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        out["gpu"] = torch.cuda.get_device_name(0)
        out["sm"] = f"sm_{major}{minor}"
except Exception as exc:
    out["error"] = f"torch: {exc}"

for module, key in (("sageattention", "sageattention"), ("triton", "triton"),
                    ("diffusers", "diffusers")):
    try:
        mod = __import__(module)
        out[key] = getattr(mod, "__version__", "present")
    except Exception as exc:
        out[key] = f"MISSING ({type(exc).__name__})"

try:
    from pipeline.steps.wan22_vace_denoise import _select_sage_backend
    out["selected"] = _select_sage_backend() or "none (PyTorch native SDPA)"
except Exception as exc:
    out["selected"] = f"could not resolve: {exc}"

print("B2C_ATTENTION " + json.dumps(out))
"""


def check_attention(envs: Dict[str, Dict[str, Any]]) -> Check:
    """Which attention kernel the denoise step will actually use.

    Worth a check of its own because the answer is arrived at in three
    hops and silently degrades at each one: `_select_sage_backend()` picks
    a backend from the GPU's compute capability, diffusers'
    `set_attention_backend` is asked for it, and if that raises — a
    SageAttention that did not compile for this arch, a missing Triton, a
    diffusers too old to know the name — the step logs a warning and
    carries on with PyTorch native SDPA. A run that quietly took the slow
    path looks exactly like one that took the fast path, only longer, and
    the image builds SageAttention from source specifically so it does not
    have to.

    The probe runs inside the wan22 venv, since that is where those
    packages are; the main venv has none of them.
    """
    import json

    config = (envs or {}).get("wan22", {})
    python_bin = config.get("python_bin")
    if not python_bin:
        return Check("attention", SKIP, "no wan22 env in the registry")
    if not Path(python_bin).exists():
        return Check("attention", FAIL, f"{python_bin} does not exist")

    result = _run([python_bin, "-c", _ATTENTION_PROBE], timeout=180)
    marker = "B2C_ATTENTION "
    line = next(
        (l for l in (result.stdout or "").splitlines() if l.startswith(marker)), None
    )
    if line is None:
        detail = "probe failed inside the wan22 venv"
        return Check("attention", FAIL, detail,
                     [f"    {l}" for l in (result.stderr or "").splitlines()[-5:]])

    found = json.loads(line[len(marker):])
    selected = str(found.get("selected", "unknown"))
    lines = [
        f"gpu: {found.get('gpu', 'none')} {found.get('sm', '')}".rstrip(),
        f"wan22 denoise will request: {selected}",
        f"sageattention: {found.get('sageattention')}",
        f"triton: {found.get('triton')}",
        f"diffusers: {found.get('diffusers')}",
        # Not a guess: this is seedvr2's own default, and unlike wan22 it
        # has no auto-selection — flash-attn/apex are optional accelerators
        # behind non-default attention_mode values. See steps/seedvr2.py.
        "seedvr2 upscale: sdpa (PyTorch native; its attention_mode default)",
    ]

    # A sage backend that was asked for but whose package is missing means
    # the step falls back to SDPA at load time, having said nothing until
    # then. That is the case this check exists to make visible.
    status = OK
    if selected.startswith("could not resolve"):
        # check_step_venvs already FAILs on a wan22 venv that cannot import
        # torch, so this stays a WARN rather than reporting the same broken
        # venv twice.
        status = WARN
    if selected.startswith(("sage", "_sage")) and "MISSING" in str(found.get("sageattention")):
        status = WARN
        lines.append(
            "SageAttention is selected for this GPU but not importable — the "
            "step will fall back to PyTorch native SDPA at load time"
        )
    if selected.startswith("_sage") and "MISSING" in str(found.get("triton")):
        status = WARN
        lines.append("the selected kernel is Triton-based, and Triton is missing")

    return Check("attention", status, selected, lines)


def check_huggingface() -> Check:
    """A token that works, and access to the two gated repos the pipeline needs."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    lines = [f"HF_HOME={os.environ.get('HF_HOME', '(unset — defaults to ~/.cache)')}"]
    if not token:
        return Check(
            "huggingface", WARN,
            "HF_TOKEN not set — gated checkpoints (RMBG-2.0, sam-3d-body-dinov3) will 401",
            lines,
        )
    lines.append(f"HF_TOKEN set ({len(token)} chars)")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return Check("huggingface", WARN, "huggingface_hub not importable here", lines)

    api = HfApi(token=token)
    try:
        lines.append(f"authenticated as {api.whoami()['name']}")
    except Exception as exc:
        return Check("huggingface", FAIL, f"token rejected: {exc}", lines)

    status = OK
    for repo in ("briaai/RMBG-2.0", "facebook/sam-3d-body-dinov3"):
        try:
            api.model_info(repo)
            lines.append(f"{repo}: accessible")
        except Exception as exc:
            lines.append(
                f"{repo}: NOT accessible ({type(exc).__name__}) — accept the licence on "
                f"https://huggingface.co/{repo} with this token's account"
            )
            status = WARN
    return Check("huggingface", status, "", lines)


def check_model_caches() -> Check:
    """How much of the model set is already on the volume.

    The pipeline downloads every checkpoint lazily, on the first run that
    reaches the step needing it — so a fresh pod's first
    `fast_helical_full` spends its opening half-hour downloading, and it is
    worth being able to see that state rather than inferring it from a
    stalled progress bar.
    """
    from .paths import data_dir, models_dir

    def _size(path: Path) -> float:
        if not path.exists():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lines = [
        f"HF cache  {hf_home}: {_size(hf_home):.1f} GB",
        f"models    {models_dir()}: {_size(models_dir()):.1f} GB",
    ]

    hub = hf_home / "hub"
    if hub.exists():
        for repo in sorted(hub.glob("models--*")):
            lines.append(f"  {repo.name.replace('models--', '').replace('--', '/')}"
                         f" ({_size(repo):.1f} GB)")

    # Not a FAIL when empty: an empty cache is the correct state of a fresh
    # pod, not a fault. It is just slow, and worth knowing about in advance.
    inside_container = not str(hf_home).startswith(str(data_dir()))
    return Check(
        "model caches",
        WARN if inside_container else OK,
        "HF_HOME is NOT on the volume — downloads will fill the container disk"
        if inside_container else "",
        lines,
    )


def check_host_ram() -> Check:
    """Enough DRAM to hold a resident model set, which is a NEW requirement.

    `keep_loaded: true` (fast_helical_full's two denoise passes) keeps the
    Wan pipeline alive in host RAM between invocations instead of re-reading
    ~47 GB off the network volume. That is the whole point — but it means
    the pod must actually have the RAM. A pod that doesn't gets its worker
    OOM-killed somewhere in the middle of a long run, or swaps, and neither
    failure names the cause: you see a worker that died, not "you sized the
    pod wrong". Cheap to check at start, expensive to diagnose at hour two.

    WARN rather than FAIL because it depends on the workflow — a run that
    never touches wan22_vace_denoise needs none of this.
    """
    from .models import registry

    def _meminfo_gb() -> tuple[float, float]:
        try:
            import psutil

            vm = psutil.virtual_memory()
            return vm.total / 1e9, vm.available / 1e9
        except ImportError:
            pass
        values = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    values[key] = int(parts[0]) * 1024 / 1e9
        return values.get("MemTotal", 0.0), values.get("MemAvailable", 0.0)

    try:
        total, available = _meminfo_gb()
    except Exception as exc:  # noqa: BLE001 - a check blowing up is a finding
        return Check("host RAM", WARN, f"could not read memory info: {exc}")

    # What a resident wan22_vace_denoise actually holds: the fp8 experts
    # plus the base repo's text_encoder/VAE. Read from the registry so it
    # tracks the real download set rather than drifting into a stale number.
    known = registry()
    resident = sum(known[k].approx_gb for k in ("wan22", "wan22_fp8") if k in known)
    # Headroom for frame batches, activations, the CUDA host allocator and
    # the OS. 1.35x is judgement, not measurement — revise it once a real
    # pod run reports its peak RSS.
    floor = resident * 1.35

    lines = [
        f"total {total:.0f} GB, available {available:.0f} GB",
        f"resident model set for keep_loaded wan22: ~{resident:.0f} GB "
        f"(suggested floor ~{floor:.0f} GB total)",
    ]
    if total and total < floor:
        return Check(
            "host RAM", WARN,
            f"{total:.0f} GB may be too small to hold ~{resident:.0f} GB resident — "
            "a keep_loaded denoise can be OOM-killed mid-run; size the pod up "
            "or drop keep_loaded from the workflow",
            lines,
        )
    return Check("host RAM", OK, "", lines)


def check_ephemeral_caches() -> Check:
    """Caches that default into $HOME, i.e. the pod's ephemeral disk.

    The container sets each of these to a path on the volume. If one is
    unset — running outside the image, or an ENV dropped from the Dockerfile
    — the library silently falls back to $HOME, and the work it caches is
    redone after every pod restart. TRITON_CACHE_DIR is the one that bites:
    every SageAttention backend JIT-compiles through Triton on first use, in
    the middle of the most expensive step in the pipeline.

    WARN, not FAIL: none of this stops a run, and on a dev box with no
    volume the fallback is correct.
    """
    from .paths import data_dir

    root = str(data_dir())
    # (env var, what it caches, whether losing it costs real time)
    watched = [
        ("TRITON_CACHE_DIR", "Triton JIT kernels (SageAttention)", True),
        ("CUDA_CACHE_PATH", "CUDA driver PTX JIT cache", True),
        ("TORCH_HOME", "torch.hub checkpoints", False),
        ("XDG_CACHE_HOME", "generic library caches", False),
        ("MPLCONFIGDIR", "matplotlib font cache", False),
    ]
    lines, offenders, costly = [], [], []
    for var, what, expensive in watched:
        value = os.environ.get(var)
        if not value:
            lines.append(f"{var}: UNSET -> $HOME (ephemeral) — {what}")
            offenders.append(var)
            if expensive:
                costly.append(var)
        elif not value.startswith(root):
            lines.append(f"{var}: {value} — NOT under {root} — {what}")
            offenders.append(var)
            if expensive:
                costly.append(var)
        else:
            lines.append(f"{var}: {value}")

    if not offenders:
        return Check("ephemeral caches", OK, "", lines)
    detail = ", ".join(offenders)
    return Check(
        "ephemeral caches", WARN,
        f"{detail} not on the volume"
        + (" — Triton/CUDA kernels will re-JIT on every pod start" if costly else ""),
        lines,
    )


def check_step_registry() -> Check:
    try:
        from . import steps  # noqa: F401
        from .registry import STEP_REGISTRY
    except Exception as exc:
        return Check("step registry", FAIL, f"could not import pipeline.steps: {exc}")
    return Check(
        "step registry", OK, f"{len(STEP_REGISTRY)} steps registered",
        [", ".join(sorted(STEP_REGISTRY))],
    )


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if not path:
        return Check("ffmpeg", WARN, "not on PATH")
    result = _run([path, "-version"], timeout=20)
    first = result.stdout.splitlines()[0] if result.stdout else path
    return Check("ffmpeg", OK, first)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_checks(envs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Check]:
    """Run every check, never raising — a crashed check is itself a FAIL."""
    checks: List[tuple[str, Callable[[], Check]]] = [
        ("environment", check_environment),
        ("disk", check_disk),
        ("nvidia-smi", check_nvidia_smi),
        ("torch", check_torch),
        ("host RAM", check_host_ram),
        ("vulkan", check_vulkan),
        ("egl", check_egl),
        ("brush binaries", check_brush_binaries),
        ("step venvs", lambda: check_step_venvs(envs or {})),
        ("attention", lambda: check_attention(envs or {})),
        ("huggingface", check_huggingface),
        ("model caches", check_model_caches),
        ("ephemeral caches", check_ephemeral_caches),
        ("step registry", check_step_registry),
        ("ffmpeg", check_ffmpeg),
    ]
    results = []
    for name, fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # a check itself blowing up is a real finding
            results.append(Check(name, FAIL, f"check raised {type(exc).__name__}: {exc}"))
    return results


def format_report(checks: List[Check], verbose: bool = True) -> str:
    width = max(len(c.name) for c in checks) + 2
    out = ["", "=" * 72, "b2c_runner doctor", "=" * 72]
    for check in checks:
        out.append(f"{_ICON[check.status]} {check.status:<5} {check.name:<{width}} {check.detail}")
        if verbose:
            out += [f"          {line}" for line in check.lines]
    counts = {status: sum(1 for c in checks if c.status == status) for status in (OK, WARN, FAIL)}
    out += [
        "-" * 72,
        f"{counts[OK]} ok, {counts[WARN]} warnings, {counts[FAIL]} failures",
        "=" * 72,
        "",
    ]
    return "\n".join(out)


def worst_status(checks: List[Check]) -> str:
    if any(c.status == FAIL for c in checks):
        return FAIL
    if any(c.status == WARN for c in checks):
        return WARN
    return OK


def log_machine_banner() -> None:
    """Three cheap lines at the top of every run's log saying what it ran on.

    Deliberately not the full `doctor` sweep — that shells out to three
    venvs and hits the network. This is GPU, torch and free space only, and
    it exists because "which pod was this, and how much VRAM did it have"
    is the first question asked of any log file after the fact, and the
    answer is otherwise nowhere in it.
    """
    import logging

    log = logging.getLogger("pipeline.doctor")
    for check in (check_nvidia_smi(), check_torch()):
        log.info("%s: %s", check.name, check.detail or check.status)
    disk = check_disk()
    for line in disk.lines:
        log.info("disk: %s", line)
