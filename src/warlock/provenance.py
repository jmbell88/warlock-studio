"""What produced this artifact: checkpoint fingerprints, library versions and
the locked recipe.

Top-level rather than under pipelines/ because it describes a *job*, not a
pipeline -- the trellis half and the SDXL half both record through here.

Torch-free by construction: versions() reads ``sys.modules`` and never
imports anything, the same trick queue.vram_gib uses, so this module imports
cleanly without the text2image extra installed.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Bytes hashed from each end of a file. Full sha256 over a 7 GB checkpoint is
# ~30 s of disk per job; head+tail+size catches every realistic case (a
# different download, a truncated file, a swapped variant) without that cost.
HASH_BYTES = 1 << 20

RECIPE_VERSION = 1

# (path, st_size, st_mtime_ns) -> fingerprint
_cache: dict[tuple[str, int, int], str] = {}

# Libraries whose version can change a generated image. Read from sys.modules
# only -- an entry absent from the dict means the module was never imported in
# this process, which is itself the truthful answer.
_TRACKED = (
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "PIL",
    "numpy",
    "cv2",
    "trimesh",
)


def file_fingerprint(path: Path) -> str:
    """``s{size}:{sha256(head||tail)[:32]}``, or "missing" when absent.

    Cached on (path, size, mtime): a run over 160 benchmark items fingerprints
    the same checkpoint 160 times.
    """
    try:
        st = path.stat()
    except OSError:
        return "missing"
    if path.is_dir():
        return _dir_fingerprint(path)

    key = (str(path), st.st_size, st.st_mtime_ns)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(HASH_BYTES))
        if st.st_size > 2 * HASH_BYTES:
            fh.seek(-HASH_BYTES, 2)
            h.update(fh.read(HASH_BYTES))
    out = f"s{st.st_size}:{h.hexdigest()[:32]}"
    _cache[key] = out
    return out


def _dir_fingerprint(path: Path) -> str:
    """A directory's fingerprint is the fingerprint of its sorted (relative
    path, size, per-file fingerprint) listing -- so a swapped safetensors or a
    changed config.json moves it, and mtime alone does not.
    """
    parts: list[str] = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        parts.append(f"{child.relative_to(path).as_posix()}={file_fingerprint(child)}")
    h = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"d{len(parts)}:{h[:32]}"


def model_fingerprints(paths: Mapping[str, Path | None]) -> dict[str, str]:
    """Fingerprint a labelled set of model paths, skipping the None entries."""
    return {
        name: file_fingerprint(path) for name, path in paths.items() if path is not None
    }


def versions() -> dict[str, str]:
    """Versions of the already-imported libraries that can change output.

    Includes prompt.PROMPT_VERSION, which is the whole reason this is not just
    a pip freeze: a change to PROMPT_TEMPLATE or chunk() silently invalidates
    every benchmark comparison, and nothing in a dependency version records it.
    """
    out: dict[str, str] = {}
    for name in _TRACKED:
        mod = sys.modules.get(name)
        version = getattr(mod, "__version__", None) if mod is not None else None
        if version:
            out[name] = str(version)
    out["python"] = sys.version.split()[0]
    prompt_mod = sys.modules.get("warlock.pipelines.prompt")
    if prompt_mod is not None:
        out["prompt"] = str(getattr(prompt_mod, "PROMPT_VERSION", ""))
    out["recipe"] = str(RECIPE_VERSION)
    return out


def trellis_recipe(config: Any, params: Mapping[str, Any], *, mesh_seed: int) -> dict[str, Any]:
    """Everything that decided the mesh, in one json.dumps-able dict."""
    return {
        "version": RECIPE_VERSION,
        "seed": mesh_seed,
        "band": config.trellis_band,
        "tex_res": config.trellis_tex_res,
        "webp": config.trellis_webp,
        "mesh_profile": params.get("mesh_profile", config.mesh_profile),
        "platform": params.get("platform"),
        "size_m": params.get("size_m"),
        "models": model_fingerprints(
            {
                "trellis_server_exe": Path(config.trellis_server_exe),
                "trellis_models_dir": Path(config.trellis_models_dir),
            }
        ),
        "versions": versions(),
    }
