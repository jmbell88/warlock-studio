"""Plotter -- the orthogonal tile-map editor's engine.

Pure in the way ``studio/inker/`` and ``studio/clay/`` are: stdlib plus numpy
plus a lazy Pillow, no imgui, no moderngl, no pygame, no ``service``. Every rule
about where a tile lands, what a gid means and what a ``.tmx`` may contain is
therefore assertable headlessly, which is the whole reason the split exists.

The one outward import is :mod:`warlock.studio.undo`, the history engine the
raster editor and Clay already share -- pinned by ``tests/plotter/
test_imports.py`` along with everything else this package reaches for, so the
next outward import is a decision rather than a discovery.

**The word "tile" is overloaded in this repo** -- ``stage == "tile"`` is a
seamless *texture*, which is a different thing entirely -- so no module here is
named for it bare: :mod:`.tilemap` and :mod:`.tileset`, never ``tile``.
"""

from __future__ import annotations
