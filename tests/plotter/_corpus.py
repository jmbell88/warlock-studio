"""Loading the Tiled fixture corpus, without giving the engine a filesystem.

``read_tmx`` and ``read_tmj`` take an ``image_loader`` and a ``tsx_loader``
rather than opening anything themselves, which is what keeps the engine
package pure -- it never learns where a file lives. This module is the host
side of that arrangement for tests: it resolves a relative reference against
the fixture directory, and, when a test is re-reading our own export, against
an in-memory mapping of what that export produced.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from warlock.studio.plotter import tsx
from warlock.studio.plotter.tileset import Tileset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiled"

# The fixture stems the corpus gate requires, each present as both a ``.tmx``
# and a ``.tmj``. Empty until the fixtures are authored in Tiled 1.12.2 --
# see ``fixtures/tiled/FIXTURES.md``. Adding a file means adding its stem
# here in the same commit, or nothing tests it.
MANIFEST: tuple[str, ...] = ()


def pairs() -> list[str]:
    """Stems in the fixture directory having both spellings, sorted."""
    if not FIXTURE_DIR.is_dir():
        return []
    stems = {path.stem for path in FIXTURE_DIR.glob("*.tmx")}
    return sorted(stems & {path.stem for path in FIXTURE_DIR.glob("*.tmj")})


def _read(source: str, directory: Path, extra: dict[str, bytes]) -> bytes:
    """One reference, resolved. ``extra`` wins, and is how a test re-reads an
    export that was never written to disk.

    Both the bare name and a ``tilesets/`` prefix are tried, because that is
    the layout ``tmx_export`` writes and a fixture authored in Tiled keeps its
    tilesets beside the map.
    """
    for key in (source, f"tilesets/{source}", Path(source).name):
        if key in extra:
            return extra[key]
    candidate = directory / source
    if not candidate.is_file():
        candidate = directory / Path(source).name
    return candidate.read_bytes()


def loaders_for(
    directory: Path = FIXTURE_DIR, *, extra: dict[str, bytes] | None = None
) -> dict[str, Callable[[str], Any]]:
    """The keyword pair both readers take.

    ``read_tsx`` takes a ``.tsx``'s bytes *and its decoded image*, not a
    loader -- so resolving the nested reference is this side's job:
    ``tsx.tsx_source`` reports the ``<image source=...>`` path the tileset
    names, and that gets decoded and handed in. ``tests/plotter/test_tmx.py``
    does the same thing against an in-memory export.

    Pillow is imported inside the loader rather than at module scope, matching
    the engine's own rule: nothing should pay for a PNG decoder by importing a
    test helper.
    """
    files = dict(extra or {})

    def image_loader(source: str) -> np.ndarray:
        from PIL import Image

        with Image.open(io.BytesIO(_read(source, directory, files))) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)

    def tsx_loader(source: str) -> Tileset:
        raw = _read(source, directory, files)
        return tsx.read_tsx(raw, image_loader(tsx.tsx_source(raw)))

    return {"image_loader": image_loader, "tsx_loader": tsx_loader}
