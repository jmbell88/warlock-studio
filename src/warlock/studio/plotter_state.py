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
    # Tiled's own letter, and unused here. Its Ctrl+click folds in Tiled's
    # *Select Same Tile* because ``S`` is already Objects.
    ("wand", "Wand", "W"),
    ("pick", "Pick", "I"),
    ("object", "Objects", "S"),
)

#: The tools a **tile** layer can host, and the ones an **object** layer can.
#:
#: Tiled's own split, and the reason to have it: half the toolbox does nothing
#: on the layer you are standing on, and a toolbox that offers a Fill on an
#: object layer is one that has to refuse the click afterwards. Six letters
#: mean two things, which is Tiled's arrangement exactly -- the letter belongs
#: to the *gesture*, and which gesture it is depends on what you are painting.
#:
#: **Insert Template is deliberately absent.** ``docs/COMPAT.md`` states object
#: templates are a non-goal, and offering the tool would be the offer-then-
#: refuse shape this codebase rejects.
TILE_TOOLS = (
    ("stamp", "Stamp", "B"),
    ("erase", "Erase", "E"),
    ("fill", "Fill", "F"),
    ("terrain", "Terrain", "T"),
    ("shape", "Shape", "P"),
    ("select", "Select", "R"),
    ("wand", "Wand", "W"),
    ("pick", "Pick", "I"),
    ("object", "Objects", "S"),
)

OBJECT_TOOLS = (
    ("object", "Select objects", "S"),
    ("object_rect", "Insert rectangle", "R"),
    ("object_point", "Insert point", "I"),
    ("object_ellipse", "Insert ellipse", "E"),
    ("object_polygon", "Insert polygon", "P"),
    ("object_polyline", "Insert polyline", "L"),
    ("object_tile", "Insert tile", "T"),
    ("object_text", "Insert text", "X"),
)

#: Which ``state.object_shape`` each insert tool means. The combo that used to
#: sit in the layers pane *was* this table with one control in front of it;
#: making the tools the control is Tiled's arrangement and one fewer place to
#: look. ``object`` is the pointer and picks nothing.
OBJECT_SHAPES = {
    "object_rect": "rect",
    "object_point": "point",
    "object_ellipse": "ellipse",
    "object_polygon": "polygon",
    "object_polyline": "polyline",
    "object_tile": "tile",
    "object_text": "text",
}


def layer_kind(layer: Any) -> str:
    """"tile", "object", or "" for a layer nothing can be drawn on.

    A group and an image layer answer ``""``: neither hosts a gesture, and the
    tools pane draws its grid disabled with a reason over them, which is the
    house pattern (greyed, never hidden).
    """
    if layer is None:
        return ""
    name = type(layer).__name__
    if name == "ObjectLayer":
        return "object"
    if name == "TileLayer":
        return "tile"
    return ""


def tools_for_kind(kind: str) -> tuple[tuple[str, str, str], ...]:
    """The palette for a layer *kind*, by name."""

    if kind == "object":
        return OBJECT_TOOLS
    if kind == "tile":
        return TILE_TOOLS
    return ()


def tools_for(layer: Any) -> tuple[tuple[str, str, str], ...]:
    """The palette for the layer in hand. Empty for a group or an image."""

    return tools_for_kind(layer_kind(layer))


def tool_keys(layer: Any) -> dict[str, str]:
    """Letter -> tool, for the palette this layer hosts.

    Per layer rather than one global table, because six letters mean two
    things: ``R`` is the rectangular *select* on a tile layer and *insert
    rectangle* on an object one, exactly as in Tiled.
    """

    return {letter.lower(): key for key, _label, letter in tools_for(layer)}


def sync_tool(state: Any, layer: Any) -> str:
    """Put a legal tool in hand for this layer, and remember the last one.

    **Idempotent, and called at the top of both the canvas's draw and the key
    handler**, so no frame can act with a tool the layer cannot host -- which
    is the one way a layer-driven palette goes wrong, and it goes wrong on the
    frame after a layer switch rather than at the switch.
    """

    kind = layer_kind(layer)
    if not kind:
        return state.tool
    palette = [key for key, _label, _letter in tools_for(layer)]
    if state.tool in palette:
        state.remembered[kind] = state.tool
    else:
        # Record it under the kind it *does* belong to before replacing it:
        # the tool in hand at a layer switch is the one the user will want back
        # when they switch back, and it is about to stop being reachable.
        for other in ("tile", "object"):
            if state.tool in {key for key, _l, _k in tools_for_kind(other)}:
                state.remembered[other] = state.tool
        state.tool = state.remembered.get(kind) or palette[0]
        if state.tool not in palette:  # pragma: no cover - a retired tool name
            state.tool = palette[0]
    shape = OBJECT_SHAPES.get(state.tool)
    if shape:
        state.object_shape = shape
    return state.tool


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
    #: Nine numbered stamps, by slot (1-9). **Per tab, and view state**: a
    #: stamp is an array of gids, and gids are numbered against one map's
    #: firstgids -- which is exactly why ``plotter_mode._paste`` refuses a
    #: cross-document tile paste by name. Not undoable, not serialized,
    #: dropped with the tab; ``.wmap`` persistence is a deliberate later step.
    stamps: dict[int, Any] = field(default_factory=dict)
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
    #: The tool last used on each kind of layer, so switching to an object
    #: layer and back finds the brush you left in your hand rather than the
    #: first button of the palette.
    remembered: dict[str, str] = field(default_factory=dict)
    # Which tileset the palette is showing, by index into ``doc.tilesets``.
    # An index rather than a firstgid because tilesets are append-only and the
    # palette is a view of a list -- and because index 0 is a meaningful
    # default where a firstgid of 0 is not a tileset at all.
    tileset_index: int = 0
    # Dim every layer but the active one on the canvas.
    #
    # **A canvas setting and nothing more.** ``render.py`` composites exports off
    # the same resolver the canvas draws from, so a dim living in
    # ``scene.resolve`` would export dimmed -- which is the one way this can be
    # wrong and the reason it is a pane concern.
    highlight: bool = False
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
    #: A menu row has asked for the resize dialog / the map-settings dialog.
    #: Flags rather than calls, ``setup_pending``'s own pattern and its reason:
    #: a popup belongs to the window that begins it, and a menu is not that
    #: window. Cleared by the pane that opens the popup.
    resize_pending: bool = False
    map_settings_pending: bool = False
    #: Which tileset the editor sheet is open on, by index, or None for "the
    #: map is on screen". The branch ``_review_workspace`` already takes.
    editing_tileset: int | None = None
    #: Which tile the sheet is editing, and which of its three tabs is up.
    #: View state: nothing here is written to the document.
    editing_tile: int = 0
    tileset_tab: str = "Tiles"
    # A library asset waiting for the map that ``setup_pending`` is asking
    # about. The Library's *Add to Plotter as a tileset* used to be drawn for
    # any asset with an ``input.png`` and then refused with an error toast when
    # no map happened to be open -- an action offered and then taken back, and
    # six navigation steps for the user to put right. It starts the map now
    # instead, which means the asset has to survive the dialog: the same
    # one-shot shape as the flag above it, cleared by whichever of Create or
    # Cancel gets there. The whole job row rather than its id, because the
    # tileset is named after the asset.
    pending_job_tileset: Any = None

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
    # The non-rectangular half of the selection: a bool array over the whole map
    # saying which cells inside ``select`` the selection actually holds.
    #
    # **``None`` means "a solid rectangle", not "no selection"**, and that is
    # what keeps today's marquee at today's cost: the common case allocates
    # nothing and every door short-circuits on it. ``select`` stays the bounding
    # rect in both cases, so everything that only needs bounds is unchanged.
    #
    # View state beside ``select`` and dropped with it: not undoable, not saved,
    # and not clamped when stored -- ``selection_mask_in`` clamps at use, exactly
    # as ``selection_in`` does, and for the same reason.
    select_mask: Any = None
    # What the selection was when a Shift/Alt marquee drag began, so the release
    # can combine with *that* rather than with the rectangle being drawn.
    # Gesture state, cleared with the rest of the drag.
    select_before: Any = None
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

    # A sheet parked at the add-tileset door, waiting for the user to confirm
    # what to do with it: ``(uid, name, source, pixels, grid)``, where ``uid``
    # names the tab the import is *for* and ``grid`` is what the detector found
    # (a ``slicing.SheetGrid``, a mismatch record, or a ``roles.SheetRoles``,
    # depending on which door parked it).
    #
    # There is no size field here on purpose: the target is always the map's own
    # tile size, which the tab already knows. Adding one would let the popup and
    # the document disagree about what the answer means.
    sheet_import: tuple[str, str, str, np.ndarray, Any] | None = None
    sheet_import_open: bool = False

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
    # Which vertex of a polygon or polyline a drag is moving, or ``None``.
    # Gesture state beside ``drag_handle``, and cleared with it.
    drag_vertex: int | None = None
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
        self.select_mask = None
        # A parked sheet is several megabytes held for *one* tab's popup. It
        # names that tab by uid, so once the tab is gone -- or the user has
        # walked away from it -- nothing will ever answer the question, and the
        # pixels would sit in memory until the app closed.
        self.sheet_import = None
        self.sheet_import_open = False

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

    def selection_mask_in(self, doc: Any) -> Any:
        """The selection's per-cell mask for one map, or ``None`` for a solid rect.

        ``None`` when there is no selection *and* when the selection is a plain
        rectangle -- the two are distinguished by :meth:`selection_in`, which is
        the bounds. Every door pairs the two, so a rectangle costs nothing.

        A mask whose shape no longer matches the map is dropped rather than
        resized: a resize is undoable and the mask is not, so a mask trimmed to
        fit could not grow back, and a wrongly-shaped one would mask the wrong
        cells. Falling back to the bounding rect is the safe direction -- it can
        only ever allow *more* than the mask did, inside a rect the user drew.
        """
        mask = self.select_mask
        if mask is None or self.select is None:
            return None
        if tuple(mask.shape) != (int(doc.height), int(doc.width)):
            return None
        return mask

    def set_selection(
        self, rect: tuple[int, int, int, int] | None, mask: Any = None
    ) -> None:
        """The one door onto the selection, so the rect and the mask cannot
        disagree about which cells are in it.

        A mask is stored only with a rect, and clearing the rect clears both --
        a mask with no bounds is a selection nothing can draw and nothing can
        clamp.
        """
        if rect is None:
            self.select, self.select_mask = None, None
            return
        self.select = rect
        self.select_mask = mask

    def shift_cells(self, dx: int, dy: int, size: tuple[int, int]) -> None:
        """Move every cell-space view field, after an infinite map grew.

        A growth slides the stored window under content that did not move, so a
        cell's *index* changes while its true coordinate does not -- and every
        number this class holds in cell space is an index. Left alone they all
        point one growth to the left of where the user put them: a marquee jumps
        off the selection, a Shift+click draws its line from the wrong cell, and
        an open drag interpolates from a cell the pointer was never in.

        One door for the same reason :meth:`set_selection` is one: these fields
        have to move together or they disagree about which cell is which.

        The mask is *padded* rather than dropped, unlike the resize case
        :meth:`selection_mask_in` guards against. A growth is not a user resize
        -- nothing was cropped and nothing has to grow back -- so the mask can
        be placed into the new window exactly, and dropping it would silently
        widen the selection to its bounding rect mid-stroke.
        """
        if not (dx or dy):
            return
        for name in ("hover_cell", "drag_anchor", "last_paint", "drag_last_cell"):
            cell = getattr(self, name)
            if cell is not None:
                setattr(self, name, (cell[0] + dx, cell[1] + dy))
        for name in ("select", "select_before"):
            rect = getattr(self, name)
            if rect is not None:
                setattr(
                    self,
                    name,
                    (rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy),
                )
        if self.select_mask is not None:
            import numpy as np

            width, height = int(size[0]), int(size[1])
            rows, columns = self.select_mask.shape
            if dy + rows <= height and dx + columns <= width and dx >= 0 and dy >= 0:
                grown = np.zeros((height, width), dtype=bool)
                grown[dy : dy + rows, dx : dx + columns] = self.select_mask
                self.select_mask = grown
            else:
                # Cannot be placed, so the rect stands alone -- the same safe
                # direction ``selection_mask_in`` takes.
                self.select_mask = None

    def clear_drag(self) -> None:
        self.drag_kind = ""
        self.drag_anchor = None
        self.drag_object = None
        self.drag_handle = ""
        self.drag_vertex = None
        self.palette_anchor = None
        # Belongs to the gesture, not the selection: what was selected before a
        # marquee drag started has no meaning once the drag is over.
        self.select_before = None
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
