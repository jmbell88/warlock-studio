"""The preset library: recipes shipped as data beside this module.

A preset is a recipe JSON file under ``presets/``. Nothing here knows what
any of them contains -- ``names()`` is the directory listing and ``load()``
is ``recipe.loads`` -- so adding an effect is adding a file, and the test
that renders every preset in both modes is what keeps them honest. A load
hands back fresh uids (``bump_uids``), because the same preset inserted twice
must not share a layer's identity with its first copy.
"""

from __future__ import annotations

from pathlib import Path

from . import recipe as R

PRESET_DIR = Path(__file__).with_name("presets")


def names() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def path_of(name: str) -> Path:
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"not a preset name: {name!r}")
    path = PRESET_DIR / f"{name}.json"
    if not path.is_file():
        raise KeyError(name)
    return path


def load(name: str) -> R.Recipe:
    """A clamped recipe with fresh layer uids."""
    return R.bump_uids(R.loads(path_of(name).read_text(encoding="utf-8")))


def label(name: str) -> str:
    """``ice_nova`` -> ``Ice nova``: what a menu shows."""
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]
