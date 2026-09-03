"""What produced this artifact: checkpoint fingerprints, library versions and
the locked recipe.

Top-level rather than under pipelines/ because it describes a *job*, not a
pipeline -- the trellis half and the SDXL half both record through here.

Torch-free by construction: ``versions()`` reads ``sys.modules`` and, for a
library that is installed but not yet imported, its *distribution metadata* --
never the library itself. So this module imports cleanly without the
text2image extra, and reading a version costs neither the seconds nor the
gigabytes importing torch would.

(It does import one thing: ``pipelines.prompt``, whose own docstring is the
contract that it stays torch-free. That is a deliberate exception, not a lapse
in the rule -- see ``versions``.)
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Bytes hashed from each end of a file. Full sha256 over a 7 GB checkpoint is
# ~30 s of disk per job; head+tail+size catches every realistic case (a
# different download, a truncated file, a swapped variant) without that cost.
HASH_BYTES = 1 << 20

RECIPE_VERSION = 1

# (path, st_size, st_mtime_ns) -> fingerprint
_cache: dict[tuple[str, int, int], str] = {}

# Libraries whose version can change a generated image. Read from sys.modules
# when already imported and from distribution metadata when not -- reading only
# sys.modules made the guard vacuous, because the manifest is built on the
# bench CLI's cold path, before torch or diffusers have ever been touched.
# An entry still absent after both means the library genuinely is not installed.
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


# Distribution names for the tracked modules that are not importable cheaply.
# Reading a version from installed metadata costs a file read; importing torch
# to ask it costs seconds and gigabytes, which is why the bench manifest is
# written before anything heavy has loaded.
_DIST_NAMES = {
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
}


def versions() -> dict[str, str]:
    """Versions of the libraries that can change output.

    Includes prompt.PROMPT_VERSION, which is the whole reason this is not just
    a pip freeze: a change to PROMPT_TEMPLATE, TILE_TEMPLATE or
    chunk() silently invalidates every benchmark comparison, and nothing in a
    dependency version records it.

    An already-imported module answers from its ``__version__``; anything else
    is read from installed distribution metadata rather than imported. That
    distinction is the difference between a real drift guard and a vacuous one:
    the bench manifest is built before torch or diffusers have ever been
    imported, so reading only ``sys.modules`` recorded neither -- and
    ``compare_manifests`` then compared None to None and declared every run
    comparable to every other.
    """
    out: dict[str, str] = {}
    for name in _TRACKED:
        version = _module_version(name)
        if version:
            out[name] = version
    out["python"] = sys.version.split()[0]
    # Imported outright: pipelines.prompt is deliberately torch-free (its own
    # docstring is the contract), so this stays as cheap as a metadata read.
    try:
        from .pipelines import prompt as prompt_mod

        out["prompt"] = str(getattr(prompt_mod, "PROMPT_VERSION", ""))
    except Exception:  # pragma: no cover - the package is always importable
        log.debug("could not read the prompt version", exc_info=True)
    out["recipe"] = str(RECIPE_VERSION)
    return out


def _module_version(name: str) -> str | None:
    mod = sys.modules.get(name)
    version = getattr(mod, "__version__", None) if mod is not None else None
    if version:
        return str(version)
    try:
        from importlib import metadata

        return str(metadata.version(_DIST_NAMES.get(name, name)))
    except Exception:
        # Not installed, or installed under a name we do not know. Absent beats
        # wrong: a missing entry compares equal across two runs that both lack
        # it, which is the honest answer.
        return None


def trellis_recipe(config: Any, params: Mapping[str, Any], *, mesh_seed: int) -> dict[str, Any]:
    """Everything that decided the mesh, in one json.dumps-able dict.

    The server settings are read from the job's own params when it pinned
    them, and only otherwise from the config: a sweep unit that restarts
    trellis-server with ``--band 8`` and then recorded the config's ``None``
    would describe a server that did not run.

    The optimise tier is one of those, and it read the wrong key: a job stores
    it as ``params["profile"]`` (``jobs.create_job``, and what ``queue`` itself
    reads), so ``params["mesh_profile"]`` was never present and every recipe
    recorded the *config default* -- a bench recipe carrying "standard"
    recorded "raw". The two keys beside it were right, which is what made it
    look like a name rather than a bug.
    """
    return {
        "version": RECIPE_VERSION,
        "seed": mesh_seed,
        "band": params.get("trellis_band", config.trellis_band),
        "tex_res": params.get("trellis_tex_res", config.trellis_tex_res),
        "gss": params.get("trellis_gss", config.trellis_gss),
        "gsh": params.get("trellis_gsh", config.trellis_gsh),
        "max_tokens": params.get("trellis_max_tokens", config.trellis_max_tokens),
        "webp": config.trellis_webp,
        "mesh_profile": params.get("profile") or config.mesh_profile,
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
