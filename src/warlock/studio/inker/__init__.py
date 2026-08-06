"""The Inker engine: layered raster documents, without a window.

Pure Python -- Pillow for codecs and rasterising, numpy for arithmetic, and
nothing else. Nothing under this package imports imgui, moderngl, pygame or the
service layer, which is what makes every rule the editor has about pixels
assertable in a headless test.

The modules, bottom up:

``composite``  blend-mode arithmetic and the region compositor
``layers``     a canvas-sized RGBA plane and the stack that orders them
``undo``       typed edits -- dirty-rect patches, structural changes, replays
``selection``  8-bit masks, the magic wand, floating pixels, the clipboard
``brush``      cached coverage stamps and the spacing walk of one stroke
``gradient``   linear and radial ramps
``transform``  flip, rotate, scale, crop, canvas resize
``ora``        OpenRaster read and write
``document``   the one type that knows about all of the above

Everything a caller needs is re-exported here, so ``from .. import inker`` and
``inker.Document`` keep working exactly as they did when this was one file.
"""

from __future__ import annotations

from .brush import (
    DEFAULT_SPACING,
    MAX_BRUSH,
    MIN_BRUSH,
    MODES,
    SYMMETRY,
    StrokeState,
    clamp_brush,
    make_stamp,
)
from .composite import BLEND_MODES
from .document import (
    OPAQUE_WHITE,
    RGBA,
    SHAPES,
    TRANSPARENT,
    Document,
    matte_for,
    normalise_rect,
)
from .gradient import KINDS as GRADIENT_KINDS
from .layers import Layer, LayerStack
from .ora import ora_bytes, read_ora, write_ora
from .selection import COMBINE_OPS, Clipboard, FloatingBuffer, SelectionMask, magic_wand
from .undo import UNDO_BYTES, UNDO_MAX_DEPTH, UNDO_MIN_DEPTH, UndoStack

__all__ = [
    "BLEND_MODES",
    "COMBINE_OPS",
    "Clipboard",
    "DEFAULT_SPACING",
    "Document",
    "FloatingBuffer",
    "GRADIENT_KINDS",
    "Layer",
    "LayerStack",
    "MAX_BRUSH",
    "MIN_BRUSH",
    "MODES",
    "OPAQUE_WHITE",
    "RGBA",
    "SHAPES",
    "SYMMETRY",
    "SelectionMask",
    "StrokeState",
    "TRANSPARENT",
    "UNDO_BYTES",
    "UNDO_MAX_DEPTH",
    "UNDO_MIN_DEPTH",
    "UndoStack",
    "clamp_brush",
    "magic_wand",
    "make_stamp",
    "matte_for",
    "normalise_rect",
    "ora_bytes",
    "read_ora",
    "write_ora",
]
