"""Multi-document state for Plotter, without imgui.

The ``inker_state`` split, for the ``inker_state`` reasons: which map is open,
which is dirty, where the view is and which tool is in hand would all still make
sense if the app were driven by a script, so none of it needs a window to be
tested.

**The view type is imported, not reimplemented.** ``inker_state.PaintView`` and
its five functions -- fit, centre, to_image, to_screen, zoom_about -- are a
zoomable, pannable 2D viewport and nothing about them is about pixels. A second
copy would be a second set of clamping rules and a second Ctrl+0, drifting the
first time either is touched.

**Tool settings belong to the app; the view belongs to the document.** Switching
tabs must not change which tool is in your hand, and a tab must remember where
it was scrolled to. Both conventions come from the raster editor beside it,
which is the point -- the two modes should not feel like different applications.

**The stamp brush is a small array, not a tile id.** A rectangular drag across
the tileset palette picks an n-by-m block, and a single tile is that block with
n = m = 1 -- so there is one code path rather than a scalar case and a block
case that have to agree about flags, clipping and the no-op rule.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import docmodes
from .inker_state import PaintView

# What Plotter can open and what each suffix means. ``wmap`` is the project
# file; the other two are Tiled's, and opening one is an *import* -- the
# document remembers the format so Ctrl+S writes back what was opened.
WMAP_SUFFIX = ".wmap"
TMX_SUFFIX = ".tmx"
TMJ_SUFFIX = ".tmj"
MAP_SUFFIXES = (WMAP_SUFFIX, TMX_SUFFIX, TMJ_SUFFIX)

#: What the two empty states tell the user they may drop, derived from the list
#: above rather than written out. Two hand-written copies existed and they
#: already disagreed -- the bridge's named two suffixes and the canvas's named
#: three, so the pane holding the Open button was the one claiming ``.tmj`` was
#: not droppable. ``filetypes.describe`` is the same argument one layer up.
MAP_SUFFIX_TEXT = " / ".join(MAP_SUFFIXES)

# (key, label, letter), and the order is the order of the button grid.
#
# **The letters are Tiled's, not the raster editor's.** They used to match
# Inker's B/E/G/R on the argument that a user arriving from there already had
# that map in their hand -- but this is a *tile map* editor, and the tool a user
# reaches for it with is Tiled, where the same four letters mean something else
# in two cases. Following both was never possible; following the neighbour that
# shares this editor's file formats is the one that pays.
#
# Four letters moved together, and they had to: R is Tiled's rectangular select,
# so Shape had to vacate it before Select could take it, which is why this is
# one change rather than four. Fill goes G -> F (Ctrl+G still toggles the grid,
# a chord, so nothing collides) and Objects goes O -> S, plain, because Ctrl+S
# is save and a chord does not take the bare letter.
TOOLS = (
    ("stamp", "Stamp", "B"),
    ("erase", "Erase", "E"),
    ("fill", "Fill", "F"),
    ("terrain", "Terrain", "T"),
    ("shape", "Shape", "P"),
    ("select", "Select", "R"),
    ("pick", "Pick", "I"),
    ("object", "Objects", "S"),
)

TOOL_KEYS = {letter.lower(): key for key, _label, letter in TOOLS}

_uids = itertools.count(1)


@dataclass
class PlotterDoc:
    """One tab.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two files called ``level.tmx`` one tab, and
    would move a tab's identity every time a Save As renamed it.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    # "wmap" | "tmx" | "tmj" -- what Ctrl+S writes. A map imported from Tiled
    # saves back to Tiled rather than silently becoming a ``.wmap`` the user
    # cannot open in the tool they came from.
    file_format: str = "wmap"
    uid: str = field(default_factory=lambda: f"pl{next(_uids)}")
    view: PaintView = field(default_factory=PaintView)
    saved_head: int = 0
    saving: bool = False

    # Crash-safety, owned by :mod:`studio.journal` (UX-05). Inker's three
    # fields, verbatim, because they are the same three questions: which file
    # this tab owns under the autosave directory (minted on the first copy, so
    # an untouched tab litters nothing), the history position that copy
    # captured (an undo back to it is not a new edit), and the debounce.
    journal_name: str = ""
    journal_head: int | None = None
    journal_at: float = 0.0

    @property
    def busy(self) -> bool:
        """Whether the document may be restructured right now.

        One question with one answer behind it today, spelled as a property
        anyway for the reason ``InkerDoc.busy`` is: a second reason can then be
        added in one place rather than in nine panes that each remember to
        ``or`` a new flag in.
        """
        return self.saving

    @property
    def dirty(self) -> bool:
        return self.doc.dirty

    @property
    def label(self) -> str:
        """What imgui draws. The id after ### is what it *matches* on, so the
        visible part is free to change without moving the tab."""
        return f"{self.title}###{self.uid}"

    def mark_saved(self, head: int | None = None) -> None:
        """Record which history position is now on disk.

        The head is captured when the *encode* starts, not when it finishes: an
        edit made while the file was being written is genuinely not in it.
        """
        self.doc.mark_saved(head)
        self.saved_head = self.doc.saved_head
        self.saving = False


@dataclass
class PlotterState:
    docs: list[PlotterDoc] = field(default_factory=list)
    active_uid: str = ""

    # Tool settings: app-level, shared across documents on purpose.
    tool: str = "stamp"
    # What the Shape tool fills: "rect" or "ellipse". App-level for the same
    # reason ``tool`` is -- it is a preference about how you work, not a fact
    # about a map. Deliberately *not* a pair of tools: they are one gesture with
    # one preview and one undo step, and Tiled's Shape Fill is one button too.
    shape_mode: str = "rect"
    # Geometry placed by the Object tool. Kept with the other tool settings so
    # switching maps does not unexpectedly switch the insertion tool.
    object_shape: str = "rect"
    # Which tileset the palette is showing, by index into ``doc.tilesets``.
    # An index rather than a firstgid because tilesets are append-only and the
    # palette is a view of a list -- and because index 0 is a meaningful
    # default where a firstgid of 0 is not a tileset at all.
    tileset_index: int = 0
    # The block the palette last selected, as encoded gids. ``None`` means
    # nothing is picked, which is what an empty document starts as.
    brush: np.ndarray | None = None
    # Tiled-compatible random mode chooses one non-empty tile from the current
    # palette stamp for each placement instead of stamping the whole block.
    random_mode: bool = False
    # The palette's own drag, in local tile coordinates of the active tileset.
    palette_anchor: tuple[int, int] | None = None
    # Which terrain the Terrain tool lays down, as ``(tileset index, terrain)``.
    # ``None`` means none is picked, which is what a map with no terrain set is.
    # A pair rather than a bare index because a map may carry more than one
    # terrain set and "terrain 2" alone does not say whose.
    terrain: tuple[int, int] | None = None

    # A "New map" was asked for and the setup dialog has not opened yet.
    #
    # A flag rather than each door opening the dialog itself, because a popup
    # belongs to the window that begins it (the rule ``inker_canvas``'s
    # new-canvas popup is written around) and three of the five doors -- the
    # command palette, Home, and Ctrl+N -- are not Plotter panes at all. They
    # all say *that* a map was asked for; the canvas pane is the one window that
    # can ask what it should be, and it clears this when it opens.
    setup_pending: bool = False

    grid: bool = True
    show_objects: bool = True
    minimap: bool = True
    # The cell under the pointer, for the status line. View state, recomputed
    # every frame and never persisted; ``None`` when the pointer is elsewhere.
    hover_cell: tuple[int, int] | None = None
    # The object under the cursor's attention, by uid. View state: not
    # undoable and not persisted, exactly as Clay's element selection is.
    selected_object: int | None = None
    # The marquee, as a normalized *inclusive* cell rect ``(x0, y0, x1, y1)``.
    #
    # View state on the mode, beside ``brush`` and ``selected_object``, and
    # deliberately **not** an undoable document edit: Tiled's selections are not
    # undoable either, and making one an Edit would mark a saved map dirty
    # because somebody dragged a marquee across it. It is also not clamped when
    # stored -- a resize can shrink the map under it -- so every use clamps
    # against the map's current size instead.
    select: tuple[int, int, int, int] | None = None
    # The copied block, and the uid of the document it came from.
    #
    # App-level rather than per-tab so the *refusal* can name its source: gids
    # are numbered against a map's own firstgids, so the same number means a
    # different tile in another map and pasting across would silently redraw the
    # block. Kept rather than dropped on a tab switch precisely so the paste can
    # say which map it came from instead of behaving as though nothing was
    # copied.
    clipboard: Any = None
    clipboard_doc: str = ""

    # Canvas drag state, decided on press because several tools start the same
    # way. ``drag_anchor`` is in *tile* coordinates -- the rectangle tool needs
    # the cell the drag began in, not the pixel.
    # "" | paint | rect | select | object. No "pan": panning is read from
    # ``space_held`` and the middle button on the frame, and never claimed a
    # drag kind -- the value was advertised here for a long time and assigned
    # nowhere, which is the kind of comment that outlives the code it described.
    drag_kind: str = ""
    drag_anchor: tuple[int, int] | None = None
    drag_object: tuple[float, float] | None = None
    # Which corner a resize is pulling ("nw" / "ne" / "se" / "sw"), or "". The
    # *opposite* corner is what stays pinned, so this names the moving end.
    drag_handle: str = ""
    space_held: bool = False
    # The last cell a stamp actually landed on, which is where Shift+click draws
    # a line *from* -- Tiled's model. Document-scoped view state: it names a cell
    # in a particular map, so it is dropped when the tab changes.
    last_paint: tuple[int, int] | None = None
    # The cell the open drag last visited, so a fast drag that skips cells can be
    # filled in. Belongs to the gesture rather than the document, and so is
    # cleared with the rest of the drag.
    drag_last_cell: tuple[int, int] | None = None

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> PlotterDoc | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, doc: PlotterDoc) -> PlotterDoc:
        self.docs.append(doc)
        self.active_uid = doc.uid
        self.clear_drag()
        # Through the same reset ``activate`` does, and for the same reason:
        # arriving at a different document leaves the palette index and the
        # object selection naming things the new map does not have. Adding a
        # tab *is* arriving at one.
        self._forget_document_state()
        return doc

    def get(self, uid: str) -> PlotterDoc | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def close(self, uid: str) -> bool:
        doc = self.get(uid)
        if doc is None:
            return False
        index = self.docs.index(doc)
        self.docs.remove(doc)
        if self.active_uid == uid:
            # The neighbour, not the first: closing a tab should leave you next
            # to where you were rather than at the far end of the bar.
            self.active_uid = self.docs[min(index, len(self.docs) - 1)].uid if self.docs else ""
        self.clear_drag()
        return True

    def activate(self, uid: str) -> None:
        if uid != self.active_uid:
            self.active_uid = uid
            self.clear_drag()
            self._forget_document_state()

    def _forget_document_state(self) -> None:
        """Drop everything that names a *particular* document's contents.

        The palette index and the object selection describe the previous
        document's tilesets and layers; carrying either across would point them
        at things the new one does not have.
        """
        self.tileset_index = 0
        self.brush = None
        self.terrain = None
        self.selected_object = None
        # Names a cell in the document being left, so a Shift+click in the new
        # one would draw a line from somewhere the user never clicked.
        self.last_paint = None
        # Likewise a rectangle of cells: carried across, it would constrain
        # painting in a map the user never drew it on.
        self.select = None

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> PlotterDoc | None:
        """An already-open tab for this file, so opening twice focuses rather
        than forking -- two tabs over one path would race on save.

        **Resolved and case-folded, because on Windows one file has many
        spellings.** ``Level.WMAP`` and ``level.wmap`` are the same file and
        ``Path.__eq__`` says they are not, so the same document opened from the
        recents list and from a drop used to fork into two tabs that then raced
        on save -- exactly what this method exists to prevent. ``resolve`` folds
        the other spellings (a relative path, a symlink, an 8.3 short name);
        ``normcase`` is what makes the comparison case-insensitive *where the
        filesystem is*, and a no-op where it is not.

        Clay and Packwright carry the same shape and the same bug; fixing them
        is deliberately not this change, because each has its own tab list.
        """
        probe = os.path.normcase(str(Path(path).resolve()))
        for doc in self.docs:
            if doc.path is None:
                continue
            if os.path.normcase(str(Path(doc.path).resolve())) == probe:
                return doc
        return None

    # -- drag ---------------------------------------------------------------

    def selection_in(self, doc: Any) -> tuple[int, int, int, int] | None:
        """The marquee clamped to one map's current size, or ``None``.

        Clamped here rather than when stored, because the map can change size
        under a selection: a resize is undoable and the marquee is not, so a
        rect trimmed at store time could not grow back when the resize was
        undone. ``None`` when the map has shrunk out from under it entirely.
        """
        if self.select is None:
            return None
        x0, y0, x1, y1 = self.select
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(int(doc.width) - 1, int(x1)), min(int(doc.height) - 1, int(y1))
        if x1 < x0 or y1 < y0:
            return None
        return x0, y0, x1, y1

    def clear_drag(self) -> None:
        self.drag_kind = ""
        self.drag_anchor = None
        self.drag_object = None
        self.drag_handle = ""
        self.palette_anchor = None
        # Not ``last_paint``: that outlives the gesture on purpose, because the
        # whole point of it is to be there on the *next* click.
        self.drag_last_cell = None



def ensure(ctx: Any) -> PlotterState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Plotter should not pay for it, and
    because ``AppState`` deliberately knows nothing about it.

    Here rather than in ``plotter_mode`` because this and :func:`active` touch
    exactly one thing -- ``ctx.state.plotter`` -- which is this module's whole
    charter, and neither knows a job or a task thread exists.
    """
    state = ctx.state.plotter
    if state is None:
        state = PlotterState()
        ctx.state.plotter = state
    return state


def active(ctx: Any) -> PlotterDoc | None:
    """The focused tab, or ``None``. Deliberately *not* through :func:`ensure`:
    asking which map is open must not create the state that says none is."""
    state = ctx.state.plotter
    return state.active if state is not None else None


# The same answer in three of the four modes; Clay's is on ``stem`` on purpose.
title_for = docmodes.title_for


def format_for(path: Path | None) -> str:
    """Which writer a path implies. ``wmap`` for anything unrecognised, since
    that is the only format that can hold everything the editor models."""
    if path is None:
        return "wmap"
    suffix = path.suffix.lower()
    if suffix == TMX_SUFFIX:
        return "tmx"
    if suffix == TMJ_SUFFIX:
        return "tmj"
    return "wmap"
