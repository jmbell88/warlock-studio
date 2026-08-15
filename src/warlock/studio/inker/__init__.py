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
``indexed``    the palette constraint: nearest-swatch snap, exact remap
``dither``     palette conversion -- nearest, error diffusion, ordered matrices
``transform``  flip, rotate, scale, crop, canvas resize
``animation``  the frames-by-tracks grid, and the sparse cel map in it
``anim_edits`` undo steps for that grid
``ora``        OpenRaster read and write
``document``   the one type that knows about all of the above

``Document`` is a dataclass in ``document.py``, which holds the fields, the
composite and flatten caches, and the cross-cutting write paths every concern
goes through. Its concern blocks are method-only mixins it inherits, one per
sibling module: ``_doc_anim``, ``_doc_paint``, ``_doc_history``,
``_doc_selection``, ``_doc_layers``, ``_doc_geometry``, ``_doc_indexed``. They
are private to the package and nothing outside it names them.

Everything a caller needs is re-exported here, so ``from .. import inker`` and
``inker.Document`` keep working exactly as they did when this was one file.
"""

from __future__ import annotations

from .animation import (
    DEFAULT_DURATION_MS,
    MAX_DURATION_MS,
    MIN_DURATION_MS,
    Animation,
    Frame,
    Tag,
    Track,
)
from .brush import (
    DEFAULT_SPACING,
    MAX_BRUSH,
    MIN_BRUSH,
    MODES,
    NIBS,
    PIXEL_NIBS,
    SYMMETRY,
    StrokeState,
    clamp_brush,
    make_stamp,
)
from .composite import BLEND_MODES
from .dither import METHODS as DITHER_METHODS
from .dither import ORDERED as DITHER_ORDERED
from .document import (
    OPAQUE_WHITE,
    RGBA,
    SHAPES,
    TRANSPARENT,
    Document,
    matte_for,
    normalise_rect,
)
from .filters import FILTERS
from .gradient import KINDS as GRADIENT_KINDS
from .indexed import SORT_KEYS as PALETTE_SORT_KEYS
from .indexed import shade_ramp
from .layers import Layer, LayerStack
from .ora import ora_bytes, read_ora, write_ora
from .selection import COMBINE_OPS, Clipboard, FloatingBuffer, SelectionMask, magic_wand
from .undo import UNDO_BYTES, UNDO_MAX_DEPTH, UNDO_MIN_DEPTH, UndoStack

__all__ = [
    "BLEND_MODES",
    "FILTERS",
    "COMBINE_OPS",
    "DEFAULT_DURATION_MS",
    "MAX_DURATION_MS",
    "MIN_DURATION_MS",
    "Animation",
    "Clipboard",
    "DEFAULT_SPACING",
    "DITHER_METHODS",
    "DITHER_ORDERED",
    "Document",
    "FloatingBuffer",
    "Frame",
    "GRADIENT_KINDS",
    "Layer",
    "LayerStack",
    "MAX_BRUSH",
    "MIN_BRUSH",
    "MODES",
    "NIBS",
    "OPAQUE_WHITE",
    "PALETTE_SORT_KEYS",
    "PIXEL_NIBS",
    "RGBA",
    "SHAPES",
    "SYMMETRY",
    "SelectionMask",
    "StrokeState",
    "TRANSPARENT",
    "Tag",
    "Track",
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
    "shade_ramp",
    "write_ora",
]
