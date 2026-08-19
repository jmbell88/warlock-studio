"""The shared tile vocabulary: the gid word, the sliced atlas, the blob collapse.

The second shared leaf after ``studio/undo.py``: plotter, packwright and inker
all import it, none owns it, and it imports nothing under ``warlock``.
"""
from __future__ import annotations

from . import blob, gid, roles, slicing
from .tileset import (
    COLLISION_SHAPES,
    FILL_MODES,
    OBJECT_ALIGNMENTS,
    RENDER_SIZES,
    RGBA,
    TerrainSpec,
    TileEllipse,
    TileFrame,
    TileMeta,
    TilePolygon,
    TileRect,
    Tileset,
    TilesetRef,
    colour_text,
    frozen_rgba,
    repolish,
    rgba_colour,
)

__all__ = ["COLLISION_SHAPES", "FILL_MODES", "OBJECT_ALIGNMENTS", "RENDER_SIZES",
           "RGBA", "TerrainSpec", "TileEllipse", "TileFrame",
           "TileMeta", "TilePolygon", "TileRect", "Tileset", "TilesetRef", "blob",
           "colour_text", "frozen_rgba", "gid", "repolish", "rgba_colour", "roles",
           "slicing"]
