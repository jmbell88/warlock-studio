"""Opt-in Hunyuan3D multi-view worker boundary.

The app process never imports Hunyuan's Python package.  This module only
defines the JSON-lines protocol and admission/license rules; an installed
Hunyuan runtime can be launched as a separate Python 3.10 process by a host
integrator.  Keeping the boundary small also makes missing weights and
cancellation testable without a CUDA installation.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

from .. import winjob

BACKEND = "hunyuan3d_multiview"
LABEL = "Hunyuan3D — multi-view (experimental)"
LICENSE = "Tencent Hunyuan3D license"
LICENSE_ACK = "hunyuan3d-license-acknowledged"
SHAPE_VRAM_GIB = 10.0
TEXTURE_VRAM_GIB = 21.0
COMBINED_VRAM_GIB = 29.0
REQUIRED_VIEWS = ("front", "left", "right", "back")
TEXTURE_MODES = ("geometry", "pbr")


@dataclass(frozen=True, slots=True)
class HunyuanRequest:
    request_id: str
    views: dict[str, str]
    texture_mode: str = "pbr"
    seed: int = 0
    output_path: str = ""
    weights_path: str = ""
    protocol_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def validate_views(views: Mapping[str, str], *, require_all: bool = True) -> list[str]:
    errors: list[str] = []
    unknown = sorted(set(views) - set(REQUIRED_VIEWS))
    if unknown:
        errors.append(f"unknown view(s): {', '.join(unknown)}")
    missing = [name for name in REQUIRED_VIEWS if not views.get(name)]
    if require_all and missing:
        errors.append(f"missing required views: {', '.join(missing)}")
    return errors


def validate_install(
    executable: Path | str | None,
    weights: Path | str | None,
    *,
    license_acknowledged: bool,
    available_vram_gib: float | None = None,
    texture_mode: str = "pbr",
) -> list[str]:
    errors: list[str] = []
    if not license_acknowledged:
        errors.append("Hunyuan3D requires explicit license acknowledgement before installation or use.")
    if executable is None or not Path(executable).is_file():
        errors.append("Hunyuan3D's isolated Python 3.10 worker is not installed.")
    if weights is None or not Path(weights).exists():
        errors.append("Hunyuan3D multi-view weights are not installed.")
    if texture_mode not in TEXTURE_MODES:
        errors.append(f"texture_mode must be one of {TEXTURE_MODES}")
    required = SHAPE_VRAM_GIB if texture_mode == "geometry" else COMBINED_VRAM_GIB
    if available_vram_gib is not None and available_vram_gib < required:
        errors.append(f"Hunyuan3D needs approximately {required:g} GiB VRAM for this run (available: {available_vram_gib:g} GiB).")
    return errors


def request(
    request_id: str,
    views: Mapping[str, str],
    *,
    output_path: Path | str = "",
    weights_path: Path | str = "",
    texture_mode: str = "pbr",
    seed: int = 0,
    license_acknowledged: bool = False,
) -> HunyuanRequest:
    errors = validate_views(views)
    if texture_mode not in TEXTURE_MODES:
        errors.append(f"texture_mode must be one of {TEXTURE_MODES}")
    if not license_acknowledged:
        errors.append("license acknowledgement is required")
    if errors:
        raise ValueError("; ".join(errors))
    return HunyuanRequest(
        request_id,
        dict(views),
        texture_mode,
        int(seed),
        str(output_path),
        str(weights_path),
        1,
    )


def publish_glb(staged: Path | str, destination: Path | str) -> Path:
    """Atomically publish a worker result after it has fully closed the file."""
    staged_path = Path(staged)
    destination_path = Path(destination)
    if not staged_path.is_file() or staged_path.stat().st_size == 0:
        raise ValueError("Hunyuan3D worker did not produce a non-empty GLB")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.replace(destination_path)
    return destination_path


def run_worker(
    executable: Path | str,
    payload: HunyuanRequest,
    *,
    cwd: Path | str | None = None,
    cancel_event: Any | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run the isolated worker and consume its JSON-lines progress protocol."""
    # ``winjob.assign`` follows immediately, as it must: this child loads a 3D
    # generation model and is the largest thing this module spawns, and the
    # window between Popen and that call is the only one in which a parent
    # crash can still orphan it. Tracked as well as assigned, like every other
    # long-lived child: the job object kills it when *this* process dies, and
    # the registry is what lets a shutdown that leaves the interpreter alive
    # stop it too.
    proc = subprocess.Popen(
        [os.fspath(executable)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.fspath(cwd) if cwd else None,
        text=True,
    )
    winjob.assign(proc.pid)
    winjob.track(proc.pid, "hunyuan3d worker")
    result: dict[str, Any] = {}
    try:
        # The handshake is inside the try so that a child which dies before it
        # can be written to is killed and untracked by the finally below, the
        # same as one that fails half way through the protocol.
        assert proc.stdin is not None
        proc.stdin.write(payload.to_json() + "\n")
        proc.stdin.close()
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                raise RuntimeError("Hunyuan3D generation cancelled")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "result":
                result = message
            elif message.get("type") == "error":
                raise RuntimeError(str(message.get("message") or "Hunyuan3D worker failed"))
        code = proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        # Unconditional: a child that ignored the kill is being abandoned to
        # the kill-on-close job either way, and leaving a dead pid in the
        # registry would have shutdown try to terminate a recycled one.
        winjob.untrack(proc.pid)
    if code != 0:
        stderr = (proc.stderr.read() if proc.stderr is not None else "").strip()
        raise RuntimeError(stderr or f"Hunyuan3D worker exited with code {code}")
    return result
