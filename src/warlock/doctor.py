"""Preflight checks: what's missing before you waste two minutes on a GPU job."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import guidance, models, native, rigging, vram, winjob
from .config import Config

MIN_FREE_DISK_GB = 5.0

# Importing bpy costs seconds and its answer cannot change while this process
# lives, so the probe runs once. /api/health is polled by the UI.
_blender: Check | None = None


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool


def run_checks(config: Config, *, trellis_running: bool = False) -> list[Check]:
    """``trellis_running`` says the port is *ours*.

    Without it the port check reports a permanent false warning for the whole
    life of a warm process: /api/health runs these while trellis-server is
    resident and holding the port it is supposed to hold.
    """
    return [
        _exe_check(config),
        _gguf_check(config),
        _birefnet_check(config),
        _gltfpack_check(config),
        _warlockc_check(),
        _cuda_check(),
        _vram_check(config),
        _job_object_check(),
        _disk_check(config),
        _port_check(config, trellis_running),
        *_t2i_checks(config),
        *_matting_checks(config),
        *_pose_checks(config),
        blender_check(),
    ]


BPY_PROBE_TIMEOUT = 120.0
BPY_INSTALL_HINT = "rigging unavailable; install with: uv sync --extra rig"


def blender_check() -> Check:
    """Can we rig? Probed in a subprocess, for the same reason rigging is.

    Non-fatal by design: bpy is an optional extra with cp313-only wheels, and
    a machine without it should generate meshes exactly as before with the
    rig/pose controls hidden -- the same way a missing image model degrades.
    """
    global _blender
    if _blender is None:
        _blender = _probe_blender()
    return _blender


def _probe_blender() -> Check:
    # Any template at all, not a hardcoded pair: templates are files, adding one
    # is the supported way to add a skeleton, and naming two of them here made
    # renaming or removing either a silent rigging outage.
    if not rigging.templates():
        return Check(
            "Blender (rigging)", False,
            f"no skeleton templates found in {rigging.TEMPLATE_DIR}",
            fatal=False,
        )
    try:
        # winjob.run, not subprocess.run: this fires during Runtime._start and
        # `import bpy` takes seconds, so killing Warlock while it starts used
        # to strand a python.exe mid-import.
        proc = winjob.run(
            [sys.executable, "-c", "import bpy; print(bpy.app.version_string)"],
            capture_output=True, text=True, timeout=BPY_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("Blender (rigging)", False, f"bpy probe failed: {exc}", fatal=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["import failed"]
        return Check("Blender (rigging)", False, f"{detail[0]} -- {BPY_INSTALL_HINT}", fatal=False)
    return Check("Blender (rigging)", True, f"bpy {proc.stdout.strip()}", fatal=False)


def _exe_check(config: Config) -> Check:
    ok = config.trellis_server_exe.exists()
    detail = str(config.trellis_server_exe) if ok else f"not found at {config.trellis_server_exe}"
    return Check("trellis-server.exe", ok, detail, fatal=True)


def _gguf_check(config: Config) -> Check:
    ok = config.trellis_models_dir.exists() and any(config.trellis_models_dir.glob("*.gguf"))
    detail = (
        str(config.trellis_models_dir)
        if ok
        else f"no *.gguf found in {config.trellis_models_dir}"
    )
    return Check("TRELLIS GGUF weights", ok, detail, fatal=True)


def _birefnet_check(config: Config) -> Check:
    # The filename from guidance, not a second copy of it: guidance gates the
    # bg_removal default on this exact file, so a drifted spelling here would
    # report the weights missing while the app quietly kept asking for them.
    path = config.trellis_models_dir / guidance.BIREFNET_WEIGHTS
    ok = path.exists()
    detail = (
        str(path)
        if ok
        else f"missing at {path} -- background matting falls back to a threshold cutout"
    )
    # Named for the process that loads it: there is a second BiRefNet on the
    # host now (see _matting_checks) and the two are different downloads.
    return Check("trellis: birefnet.gguf (background removal)", ok, detail, fatal=False)


def _gltfpack_check(config: Config) -> Check:
    ok = config.gltfpack_exe.exists()
    detail = (
        str(config.gltfpack_exe)
        if ok
        else f"not found at {config.gltfpack_exe} -- meshes ship at full reconstruction density"
    )
    return Check("gltfpack (mesh optimizer)", ok, detail, fatal=False)


def _warlockc_check() -> Check:
    """The native kernels: built locally, optional, and *visibly* optional.

    Non-fatal for the reason gltfpack is: what it buys is speed, and every
    caller has a numpy path it falls back to. Worth a row anyway -- vendor/ is
    gitignored, so "the audit got slower after I moved machines" has exactly
    one cause and no other way to see it.
    """
    ok, detail = native.status()
    return Check("warlockc (native kernels)", ok, detail, fatal=False)


def _cuda_check() -> Check:
    try:
        import torch
    except ImportError:
        return Check(
            "CUDA", False, "torch not installed (uv sync --extra text2image)", fatal=False
        )
    ok = torch.cuda.is_available()
    detail = "available" if ok else "torch.cuda.is_available() is False"
    return Check("CUDA", ok, detail, fatal=False)


def _vram_check(config: Config) -> Check:
    """How large the card is, which mode that chose, and what a job may ask for.

    Fatal only when the budget cannot hold even a lone trellis run: every 3D
    job on such a host fails anyway, and it is better to say so at startup than
    two minutes into the first reconstruction.
    """
    device = vram.probe()
    resolved = vram.plan(
        exclusive=config.vram_exclusive,
        budget_gib=config.vram_budget_gib,
        total_gib=config.vram_total_gib,
        device=device,
        explicit=config.vram_exclusive_explicit,
    )
    if not resolved.enforced:
        return Check("VRAM budget", True, resolved.reason, fatal=False)
    # Only when the plan is actually about this device: with WARLOCK_VRAM_TOTAL
    # standing in for a card, the real card's free figure describes something
    # else entirely and reads as a contradiction.
    measured = device is not None and resolved.total_gib == device.total_gib
    free = f", {device.free_gib:.1f} GiB free now" if measured else ""  # type: ignore[union-attr]
    ok = resolved.budget_gib is not None and resolved.budget_gib >= vram.TRELLIS_GIB
    detail = resolved.reason + free
    if not ok:
        detail += f" -- below the {vram.TRELLIS_GIB:.0f} GiB a reconstruction needs"
    return Check("VRAM budget", ok, detail, fatal=not ok)


def _job_object_check() -> Check:
    """Is kill-on-close actually armed?

    Non-fatal -- a host without job objects is the status quo ante -- but it
    must be *visible*: the whole point of the guarantee is that GPU children
    die with the app, and silently not having it looks identical to having it
    right up until a crash strands trellis-server on port 17971.
    """
    ok = winjob.armed()
    detail = (
        "child processes die with Warlock (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)"
        if ok
        else "unavailable -- trellis-server/Blender can outlive a crash of this process"
    )
    return Check("subprocess cleanup", ok, detail, fatal=False)


def _disk_check(config: Config) -> Check:
    free_gb = shutil.disk_usage(config.data_dir).free / (1024**3)
    ok = free_gb >= MIN_FREE_DISK_GB
    return Check("free disk space", ok, f"{free_gb:.1f} GB free in {config.data_dir}", fatal=False)


def _port_check(config: Config, trellis_running: bool = False) -> Check:
    if trellis_running:
        return Check(
            "trellis port", True, f"port {config.trellis_port} held by trellis-server", fatal=False
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", config.trellis_port))
            ok, detail = True, f"port {config.trellis_port} is free"
        except OSError as exc:
            ok, detail = False, f"port {config.trellis_port} unavailable: {exc}"
    return Check("trellis port", ok, detail, fatal=False)


def _base_model_dir(config: Config, spec: models.BaseModel) -> Path:
    if spec.key == models.DEFAULT_BASE_MODEL and config.t2i_turbo_dir is not None:
        return config.t2i_turbo_dir
    return config.t2i_model_root / spec.dir_name


def _t2i_checks(config: Config) -> list[Check]:
    """One row per registry entry, all non-fatal.

    Every image model is an optional manual download and only the job that
    picks it cares, so a missing one is a note with its exact command rather
    than something that blocks startup. Listing them individually is the point:
    a single 'weights' row can't tell you *which* of five downloads you skipped.
    """
    checks: list[Check] = []
    for spec in models.BASE_MODELS.values():
        path = _base_model_dir(config, spec)
        # model_index.json is the diffusers layout marker; the unet shard is the
        # biggest file and the one a partial/wrong-variant download would miss.
        #
        # A spec may name its own probe files instead, because that formula is
        # an SDXL fact: FLUX.2 klein has no unet/ at all, and its text encoder
        # is half the download -- so a probe that looked only at the transformer
        # would call a half-fetched model present.
        variant = f".{spec.variant}" if spec.variant else ""
        wanted = spec.probe or (f"unet/diffusion_pytorch_model{variant}.safetensors",)
        ok = (path / "model_index.json").exists() and all(
            (path / rel).exists() for rel in wanted
        )
        # The step-distillation LoRA counts as part of the checkpoint, because
        # _load_loras *raises* on a missing one where a style LoRA is merely
        # skipped. Without this the row called the model ready and the job then
        # failed at load time with the checkpoint already in VRAM -- which is
        # the one thing this file exists to find out about first.
        missing_lora = None
        if ok and spec.base_lora:
            lora_path = config.t2i_model_root / "loras" / spec.base_lora
            if not lora_path.exists():
                ok = False
                missing_lora = lora_path
        if ok:
            detail = str(path)
        elif missing_lora is not None:
            detail = (
                f"weights present, but {spec.base_lora} is missing at "
                f"{missing_lora} -- this model cannot run without it; "
                f"download with:\n  {spec.download}"
            )
        else:
            detail = f"not found at {path} -- unavailable; download with:\n  {spec.download}"
        checks.append(Check(f"image model: {spec.label}", ok, detail, fatal=False))
    for lora in models.STYLE_LORAS.values():
        path = config.t2i_model_root / "loras" / lora.filename
        ok = path.exists()
        detail = (
            str(path)
            if ok
            else f"not found at {path} -- style unavailable; download with:\n  {lora.download}"
        )
        checks.append(Check(f"style LoRA: {lora.label}", ok, detail, fatal=False))
    for adapter in models.IP_ADAPTERS.values():
        root = config.t2i_model_root / adapter.dir_name
        weights = root / adapter.subfolder / adapter.weight_name
        # Both halves, deliberately: weights without the CLIP vision encoder
        # load fine and then fail at the first call, which is not a failure a
        # user can read back to a missing download.
        encoder = root / adapter.image_encoder_dir / "config.json"
        ok = weights.exists() and encoder.exists()
        if ok:
            detail = str(root)
        else:
            missing = "weights" if not weights.exists() else "CLIP vision encoder"
            detail = (
                f"{missing} not found under {root} -- conditioning unavailable; "
                f"download with:\n  {adapter.download}"
            )
        checks.append(Check(f"IP-Adapter: {adapter.label}", ok, detail, fatal=False))
    for cn in models.CONTROLNETS.values():
        path = config.t2i_model_root / cn.dir_name
        variant = f".{cn.variant}" if cn.variant else ""
        ok = (path / "config.json").exists() and (
            path / f"diffusion_pytorch_model{variant}.safetensors"
        ).exists()
        detail = (
            str(path)
            if ok
            else f"not found at {path} -- control unavailable; download with:\n  {cn.download}"
        )
        checks.append(Check(f"ControlNet: {cn.label}", ok, detail, fatal=False))
    checks.extend(_metric_checks(config))
    return checks


def _metric_checks(config: Config) -> list[Check]:
    """One row per measurement model, all non-fatal.

    These are only ever used by `python -m warlock.bench`; a missing one costs
    a metric, not a job, which is why they are reported here rather than
    failing anything.
    """
    checks: list[Check] = []
    for spec in models.METRIC_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = (path / "config.json").exists()
        detail = (
            str(path)
            if ok
            else f"not found at {path} -- benchmark metric unavailable; download with:\n"
            f"  {spec.download}"
        )
        checks.append(Check(f"metric model: {spec.label}", ok, detail, fatal=False))
    return checks


def _pose_checks(config: Config) -> list[Check]:
    """The rig's joint-placement weights, non-fatal -- and only the weights.

    Missing, every humanoid rig still happens: ``rigging.fit_template`` scales
    the template onto the mesh bounding box, which is what every rig did before
    this model existed and is still right for a reference standing in a T-pose.
    What is lost is joint placement on the ones that are *not*, and that is a
    quality difference with no other visible cause -- the skeleton is simply in
    the wrong place inside the limbs, which looks like a bad rig rather than a
    missing download.

    A green row claims less than "informed fitting is on", the same honesty
    ``_matting_checks`` keeps: all this has looked at is a directory, and
    whether a detection then clears ``pose2d``'s sanity gates is a property of
    each reference image.
    """
    checks: list[Check] = []
    for spec in models.POSE_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = (path / "config.json").exists()
        detail = (
            f"weights present at {path} -- rig joints are read off the reference image "
            "when the detection is confident, and fall back to the bbox fit when it is not"
            if ok
            else f"not found at {path} -- joint placement falls back to the "
            f"bbox-proportional fit; download with:\n  {spec.download}"
        )
        checks.append(Check(f"pose model: {spec.label}", ok, detail, fatal=False))
    return checks


def _matting_checks(config: Config) -> list[Check]:
    """The host-side matting *weights*, non-fatal -- and only the weights.

    Missing, every 2D export still works -- the corner flood fill in
    pipelines/reference.py produces the alpha instead, with visibly rougher
    edges on anything that is not on a plain background. That is a quality
    difference the user should be able to see the cause of, which is what this
    row is for.

    A green row deliberately claims less than "matting is ready". All it has
    looked at is a directory: whether the model then loads depends on imports
    inside the repo's own modelling code, which are that repo's business and
    not packages this project declares. Saying "ready" on the strength of a
    stat would turn a real failure into a silent fall-back to the flood fill
    with a green row above it, which is the worst of both answers.
    """
    checks: list[Check] = []
    for spec in models.MATTING_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = (path / "config.json").exists()
        if ok:
            detail = f"weights present at {path} -- not checked: whether the model loads"
            if spec.remote_code:
                detail += (
                    "; loading it executes third-party Python from this directory "
                    "in this process (transformers runs the repo's own modelling code)"
                )
        else:
            detail = (
                f"not found at {path} -- 2D exports fall back to the corner fill; "
                f"download with:\n  {spec.download}"
            )
        # "host matting", against _birefnet_check's "trellis": two rows, two
        # different BiRefNets -- one GGUF inside trellis-server, this one on
        # the host for 2D exports -- and a user with rough edges has to be able
        # to tell which download the row is asking for.
        checks.append(Check(f"host matting: {spec.label}", ok, detail, fatal=False))
    return checks
