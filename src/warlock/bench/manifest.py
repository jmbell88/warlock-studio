"""What a run was measured against, recorded so a later run can be compared
to it honestly -- or refused.

The manifest is the reason ``--resume`` is safe. A run half-measured against
one checkpoint and half against another is worse than no run at all, because
nothing in the numbers says which half is which. ``assert_resumable`` is the
only thing standing between that and a plausible-looking scores.json.

Pure and torch-free: ``provenance.versions()`` reads sys.modules and
distribution metadata, never the libraries themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import provenance

MANIFEST_VERSION = 1
FILENAME = "manifest.json"

# Config fields that actually change what comes out. Deliberately not the
# whole Config: data_dir and ports move between machines without changing a
# single pixel, and including them would make every manifest incomparable.
CONFIG_FIELDS = (
    "trellis_band",
    "trellis_tex_res",
    "trellis_gss",
    "trellis_gsh",
    "trellis_max_tokens",
    "trellis_decim",
    "trellis_atlas",
    "trellis_webp",
    "mesh_profile",
    "vram_exclusive",
)


def build_manifest(
    config: Any,
    suite: Any,
    recipe: Any,
    *,
    stage: str,
    items: list[str],
    seeds: list[int],
    started: str,
) -> dict[str, Any]:
    """Everything that decided this run's numbers, in one JSON-safe dict."""
    return {
        "version": MANIFEST_VERSION,
        "started": started,
        "stage": stage,
        "suite": {
            "key": suite.key,
            "file": provenance.file_fingerprint(suite.path),
            "seeds": list(suite.seeds),
        },
        "recipe": {
            "key": recipe.key,
            "file": provenance.file_fingerprint(recipe.path) if recipe.path else "inline",
        },
        "items": list(items),
        "seeds": list(seeds),
        "config": {name: _plain(getattr(config, name, None)) for name in CONFIG_FIELDS},
        "models": _model_fingerprints(config, recipe),
        "versions": provenance.versions(),
    }


def _plain(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def _model_fingerprints(config: Any, recipe: Any) -> dict[str, str]:
    from .. import models

    paths: dict[str, Path | None] = {
        "trellis_server_exe": Path(config.trellis_server_exe),
        "trellis_models_dir": Path(config.trellis_models_dir),
    }
    gltfpack = Path(config.gltfpack_exe)
    if gltfpack.exists():
        paths["gltfpack_exe"] = gltfpack
    base_key = recipe.guidance.get("base_model") or models.DEFAULT_BASE_MODEL
    spec = models.BASE_MODELS.get(base_key)
    if spec is not None:
        paths["base_model"] = config.t2i_model_root / spec.dir_name
        if spec.base_lora:
            paths["base_lora"] = config.t2i_model_root / "loras" / spec.base_lora
    lora_key = recipe.guidance.get("style_lora")
    lora = models.STYLE_LORAS.get(lora_key or "")
    if lora is not None:
        paths["style_lora"] = config.t2i_model_root / "loras" / lora.filename
    return provenance.model_fingerprints(paths)


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / FILENAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / FILENAME).read_text("utf-8"))


def compare_manifests(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Every way the two runs disagree about what they measured.

    Returns human-readable sentences, empty when the two are comparable.
    ``started`` and the item/seed lists are excluded: a resume legitimately
    covers a subset, and the timestamp always differs.
    """
    out: list[str] = []
    if old.get("stage") != new.get("stage"):
        out.append(f"stage: {old.get('stage')} -> {new.get('stage')}")
    for section in ("suite", "recipe"):
        a, b = old.get(section) or {}, new.get(section) or {}
        for key in ("key", "file"):
            if a.get(key) != b.get(key):
                out.append(f"{section}.{key}: {a.get(key)} -> {b.get(key)}")
    for name in CONFIG_FIELDS:
        a, b = (old.get("config") or {}).get(name), (new.get("config") or {}).get(name)
        if a != b:
            out.append(f"config.{name}: {a} -> {b}")
    old_models, new_models = old.get("models") or {}, new.get("models") or {}
    for name in sorted(set(old_models) | set(new_models)):
        if old_models.get(name) != new_models.get(name):
            out.append(f"model {name} changed")
    # Only the versions that can move a pixel; a patch bump in an unrelated
    # library is not worth refusing a resume over.
    for name in ("torch", "diffusers", "prompt"):
        a, b = (old.get("versions") or {}).get(name), (new.get("versions") or {}).get(name)
        if a != b:
            out.append(f"{name}: {a} -> {b}")
    return out


class NotResumable(RuntimeError):
    """The run on disk was measured against something this process is not."""


def assert_resumable(old: dict[str, Any], new: dict[str, Any]) -> None:
    drift = compare_manifests(old, new)
    if drift:
        raise NotResumable(
            "this run was measured against a different setup:\n  "
            + "\n  ".join(drift)
            + "\nStart a new run rather than mixing the two."
        )
