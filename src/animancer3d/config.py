"""Central configuration. Every path/port is env-overridable with ANIMANCER3D_* vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).resolve()


@dataclass(slots=True)
class Config:
    data_dir: Path = field(
        default_factory=lambda: _env_path("ANIMANCER3D_DATA_DIR", PROJECT_ROOT / "assets")
    )
    db_path: Path = field(
        default_factory=lambda: _env_path("ANIMANCER3D_DB", PROJECT_ROOT / "assets" / "jobs.sqlite")
    )
    trellis_server_exe: Path = field(
        default_factory=lambda: _env_path(
            "ANIMANCER3D_TRELLIS_EXE",
            PROJECT_ROOT / "vendor" / "trellis" / "trellis-server.exe",
        )
    )
    trellis_models_dir: Path = field(
        default_factory=lambda: _env_path(
            "ANIMANCER3D_TRELLIS_MODELS", PROJECT_ROOT / "models" / "trellis2-gguf"
        )
    )
    trellis_port: int = field(
        default_factory=lambda: int(os.environ.get("ANIMANCER3D_TRELLIS_PORT", "17971"))
    )
    # Seconds of queue inactivity before the trellis server is stopped to free VRAM.
    trellis_idle_timeout: float = field(
        default_factory=lambda: float(os.environ.get("ANIMANCER3D_TRELLIS_IDLE", "600"))
    )
    # Ungated default; FLUX.1-schnell is better but gated (needs HF auth).
    t2i_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "ANIMANCER3D_T2I_MODEL", "stabilityai/sdxl-turbo"
        )
    )
    # SDXL-Turbo is trained at 512; raise for models that handle 1024 natively.
    t2i_image_size: int = field(
        default_factory=lambda: int(os.environ.get("ANIMANCER3D_T2I_SIZE", "512"))
    )
    # trellis-server.exe's WebP textures declare EXT_texture_webp as required,
    # which Godot's glTF importer does not implement (it refuses the file
    # rather than skip the extension). Off is the correct default.
    trellis_webp: bool = field(
        default_factory=lambda: os.environ.get("ANIMANCER3D_TRELLIS_WEBP", "off").lower()
        in ("1", "true", "on")
    )

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
        _config.data_dir.mkdir(parents=True, exist_ok=True)
    return _config
