"""Preflight checks: what's missing before you waste two minutes on a GPU job."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import models, rigging
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
        _cuda_check(),
        _disk_check(config),
        _port_check(config, trellis_running),
        *_t2i_checks(config),
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
        proc = subprocess.run(
            [sys.executable, "-c", "import bpy; print(bpy.app.version_string)"],
            capture_output=True, text=True, timeout=BPY_PROBE_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
    path = config.trellis_models_dir / "birefnet.gguf"
    ok = path.exists()
    detail = (
        str(path)
        if ok
        else f"missing at {path} -- background matting falls back to a threshold cutout"
    )
    return Check("birefnet.gguf (background removal)", ok, detail, fatal=False)


def _gltfpack_check(config: Config) -> Check:
    ok = config.gltfpack_exe.exists()
    detail = (
        str(config.gltfpack_exe)
        if ok
        else f"not found at {config.gltfpack_exe} -- meshes ship at full reconstruction density"
    )
    return Check("gltfpack (mesh optimizer)", ok, detail, fatal=False)


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
        variant = f".{spec.variant}" if spec.variant else ""
        ok = (path / "model_index.json").exists() and (
            path / "unet" / f"diffusion_pytorch_model{variant}.safetensors"
        ).exists()
        detail = (
            str(path)
            if ok
            else f"not found at {path} -- unavailable; download with:\n  {spec.download}"
        )
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
