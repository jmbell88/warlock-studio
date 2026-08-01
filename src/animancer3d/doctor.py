"""Preflight checks: what's missing before you waste two minutes on a GPU job."""

from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

from .config import Config

MIN_FREE_DISK_GB = 5.0


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool


def run_checks(config: Config) -> list[Check]:
    return [
        _exe_check(config),
        _gguf_check(config),
        _birefnet_check(config),
        _cuda_check(),
        _disk_check(config),
        _port_check(config),
        _sdxl_cache_check(config),
    ]


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


def _port_check(config: Config) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", config.trellis_port))
            ok, detail = True, f"port {config.trellis_port} is free"
        except OSError as exc:
            ok, detail = False, f"port {config.trellis_port} unavailable: {exc}"
    return Check("trellis port", ok, detail, fatal=False)


def _sdxl_cache_check(config: Config) -> Check:
    home = os.environ.get("HF_HOME")
    hub = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"
    ok = (hub / f"models--{config.t2i_model_id.replace('/', '--')}").exists()
    detail = "cached" if ok else "not cached yet -- first text job downloads ~7 GB"
    return Check("SDXL-Turbo cache", ok, detail, fatal=False)
