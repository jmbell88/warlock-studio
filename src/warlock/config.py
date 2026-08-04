"""Central configuration. Every path/port is env-overridable with WARLOCK_* vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import DEFAULT_BASE_MODEL as DEFAULT_T2I_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# None on purpose, and confirmed by measurement -- see Config.trellis_band.
DEFAULT_TRELLIS_BAND: int | None = None


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).resolve()


def _env_opt_int(name: str, default: int | None) -> int | None:
    """Like int(os.environ[name]) but with an explicit "leave it to the exe" value.

    Empty or "auto" means None, i.e. omit the flag entirely rather than passing
    a number, so trellis-server.exe applies its own heuristic.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return None if raw in ("", "auto") else int(raw)


@dataclass(slots=True)
class Config:
    data_dir: Path = field(
        default_factory=lambda: _env_path("WARLOCK_DATA_DIR", PROJECT_ROOT / "assets")
    )
    db_path: Path = field(
        default_factory=lambda: _env_path("WARLOCK_DB", PROJECT_ROOT / "assets" / "jobs.sqlite")
    )
    trellis_server_exe: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_TRELLIS_EXE",
            PROJECT_ROOT / "vendor" / "trellis" / "trellis-server.exe",
        )
    )
    # Optional: a project folder assets can be copied straight into (e.g. a
    # Godot project's assets/). Unset means the feature is off and its routes
    # 404 -- writing outside data_dir is opt-in, never a default.
    export_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("WARLOCK_EXPORT_DIR", PROJECT_ROOT)
            if os.environ.get("WARLOCK_EXPORT_DIR")
            else None
        )
    )
    # Vendored like trellis-server.exe: a pinned native binary, never downloaded
    # at runtime. Missing it costs you the triangle budgets, not the app --
    # jobs then ship the raw reconstruction, which is what they did before.
    gltfpack_exe: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_GLTFPACK", PROJECT_ROOT / "vendor" / "gltfpack" / "gltfpack.exe"
        )
    )
    # Default triangle profile for a new job. See pipelines/optimize.PROFILES.
    #
    # "raw" and not a named tier: every other tier needs gltfpack, which is
    # vendored but not yet present, so the default used to name a profile that
    # can only fail -- silently in the worker (which logs and ships the
    # reconstruction) and with a 500 on POST /optimize. It is also the only
    # tier the UI offers, because none of the others has been qualified yet.
    # Set WARLOCK_MESH_PROFILE=standard once the binary is in place.
    mesh_profile: str = field(
        default_factory=lambda: os.environ.get("WARLOCK_MESH_PROFILE", "raw")
    )
    # Where `python -m warlock.bench` writes its runs. Outside data_dir on
    # purpose: a benchmark run copies its artifacts rather than referencing
    # them, precisely so it survives prune_jobs.
    bench_dir: Path = field(
        default_factory=lambda: _env_path("WARLOCK_BENCH_DIR", PROJECT_ROOT / "bench")
    )
    trellis_models_dir: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_TRELLIS_MODELS", PROJECT_ROOT / "models" / "trellis2-gguf"
        )
    )
    trellis_port: int = field(
        default_factory=lambda: int(os.environ.get("WARLOCK_TRELLIS_PORT", "17971"))
    )
    # Seconds of queue inactivity before the trellis server is stopped to free VRAM.
    trellis_idle_timeout: float = field(
        default_factory=lambda: float(os.environ.get("WARLOCK_TRELLIS_IDLE", "600"))
    )
    # Where every image model lives: models.BASE_MODELS[k].dir_name resolves
    # against this, and style LoRAs against its loras/ subdirectory. All
    # downloaded once by hand (see README) -- the app never downloads, and
    # loads are local_files_only.
    t2i_model_root: Path = field(
        default_factory=lambda: _env_path("WARLOCK_T2I_ROOT", PROJECT_ROOT / "models")
    )
    # Pre-registry override, still honoured: points the *turbo* entry at an
    # arbitrary local diffusers dir so existing setups keep working. Other base
    # models always resolve under t2i_model_root.
    t2i_turbo_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("WARLOCK_T2I_DIR", PROJECT_ROOT)
            if os.environ.get("WARLOCK_T2I_DIR")
            else None
        )
    )
    # Base model used when a job doesn't name one. Sampler settings are not
    # configurable here on purpose -- they belong to the checkpoint (models.py).
    t2i_model: str = field(
        default_factory=lambda: os.environ.get(
            "WARLOCK_T2I_MODEL", DEFAULT_T2I_MODEL
        )
    )
    # trellis-server.exe's WebP textures declare EXT_texture_webp as required,
    # which Godot's glTF importer does not implement (it refuses the file
    # rather than skip the extension). Off is the correct default.
    trellis_webp: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_TRELLIS_WEBP", "off").lower()
        in ("1", "true", "on")
    )
    # trellis-server.exe's "auto" texture PBR resolution bakes visible per-texel
    # noise into the baseColor atlas at --res 1024/1536 (reproduced via
    # trellis-cli.exe: default auto is noise, explicit --tex-res 512 is clean).
    # Pin it to 512 until upstream fixes the auto heuristic.
    trellis_tex_res: int = field(
        default_factory=lambda: int(os.environ.get("WARLOCK_TRELLIS_TEX_RES", "512"))
    )
    # Width of the narrow band the DC remesh runs over. The exe defaults it to
    # res/512 when the flag is absent, which is what None gives you.
    #
    # Measured 2026-08-01 (`warlock sweep`, one reference image, seed 42,
    # res 1024, hole_fraction @ 1024) -- worst-view see-through fraction:
    #
    #     auto  0.0077   267,360 faces   123 s   (res/512 == band 2 here)
    #     2     0.0077   266,632 faces   143 s
    #     4     0.0167   290,774 faces   124 s
    #     8     0.0110   297,898 faces   136 s
    #     16    0.0125   289,586 faces   193 s
    #
    # Two conclusions, both against the earlier guess that a wider band would
    # help: the heuristic is already the best of the ladder, and widening makes
    # the surface *more* perforated while adding faces and time. So the flag
    # stays off. The run also puts a floor under what counts as a real
    # difference: auto and 2 are the same setting, and still disagreed by 728
    # faces, so anything under ~0.3% is noise.
    #
    # This supersedes an earlier note claiming the default left "~1300
    # disconnected plates, 7-31% of the silhouette". Nothing in this sweep
    # measured worse than 1.7%, so whatever produced those numbers was not
    # this exe at these settings. Re-measure before acting on that claim.
    trellis_band: int | None = field(
        default_factory=lambda: _env_opt_int("WARLOCK_TRELLIS_BAND", DEFAULT_TRELLIS_BAND)
    )
    # Default: SDXL-Turbo (~7 GB) and trellis-server (~16 GB) stay resident
    # together on a 32 GB card. Set to 1/true/on to restore the old
    # stop-trellis -> run-SDXL -> unload -> restart handoff for OOM situations
    # (resolution 1536, smaller GPUs, a resident Flux). Read once at startup;
    # changing the env var mid-run has no effect until restart.
    vram_exclusive: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_VRAM_EXCLUSIVE", "off").lower()
        in ("1", "true", "on")
    )
    # Skeleton template a rig request falls back to when it doesn't name one.
    # Validated against rigging.templates() at request time, not here -- config
    # is imported by everything and must not pull the template registry in.
    rig_template: str = field(
        default_factory=lambda: os.environ.get("WARLOCK_RIG_TEMPLATE", "humanoid")
    )
    # Wall-clock ceiling for one Blender subprocess. Automatic weights on a
    # 300k-face mesh are minutes of CPU; anything past this is a hang, and a
    # hung bpy holds the single-worker queue against every other job.
    rig_timeout: float = field(
        default_factory=lambda: float(os.environ.get("WARLOCK_RIG_TIMEOUT", "1800"))
    )
    # Baking a pose is an import, a handful of quaternion assignments and an
    # export -- seconds, not minutes. It gets its own much tighter ceiling
    # because it runs inline in a request rather than on the queue.
    pose_timeout: float = field(
        default_factory=lambda: float(os.environ.get("WARLOCK_POSE_TIMEOUT", "300"))
    )
    # A sheet is one EEVEE render per cell -- 8 yaws times however many poses.
    # Generous because the cell count is user-chosen, but still bounded: this
    # runs on the serial queue and a hang would block every later job.
    sheet_timeout: float = field(
        default_factory=lambda: float(os.environ.get("WARLOCK_SHEET_TIMEOUT", "1800"))
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
