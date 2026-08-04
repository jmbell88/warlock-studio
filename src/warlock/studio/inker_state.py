"""Multi-document state for Paint mode, without imgui.

The same split ``state.py`` makes, for the same reason: which document is open,
which is dirty, where the view is and what the tool settings are would all still
make sense if the app were driven by a script, so none of it needs a window to
be tested.

Two conventions worth stating, both borrowed from Aseprite because they are
what a user coming from a paint program expects. **Tool settings belong to the
app, not to the document** -- switching tabs must not silently change your brush
size. And **the view belongs to the document** -- a tab remembers where it was
panned to, because scrolling back to where you were working is not something
the user should have to redo per tab switch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIN_ZOOM = 0.05
MAX_ZOOM = 32.0
ZOOM_STEP = 1.15

# How many files the "recent" list keeps. Ten is one screenful of a menu.
MAX_RECENT = 10

# The swatch row's own capacity. Not a palette editor -- the eyedropper is how
# a user gets the colours actually in their image; this only has to hold the
# handful they keep coming back to.
MAX_SWATCHES = 24

TOOLS = (
    ("brush", "Brush", "B"),
    ("eraser", "Eraser", "E"),
    ("fill", "Fill", "G"),
    ("gradient", "Gradient", "U"),
    ("blur", "Blur", "R"),
    ("smudge", "Smudge", "N"),
    ("line", "Line", "P"),
    ("rect", "Rect", "K"),
    ("ellipse", "Ellipse", "J"),
    ("select", "Marquee", "M"),
    ("select_ellipse", "Ellipse select", "S"),
    ("lasso", "Lasso", "Q"),
    ("wand", "Wand", "W"),
    ("move", "Move", "V"),
    ("eyedropper", "Pick", "I"),
)

# Tools whose drag paints into the layer.
PAINT_TOOLS = frozenset({"brush", "eraser", "blur", "smudge"})
SHAPE_TOOLS = frozenset({"line", "rect", "ellipse"})
SELECT_TOOLS = frozenset({"select", "select_ellipse", "lasso", "wand"})

# What each tool asks the brush engine for. Kept here rather than branched at
# the call site so a new mode is one row.
BRUSH_MODES = {"brush": "paint", "eraser": "erase", "blur": "blur", "smudge": "smudge"}

DEFAULT_SWATCHES: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 255),
    (255, 255, 255, 255),
    (128, 128, 128, 255),
    (200, 40, 40, 255),
    (230, 140, 40, 255),
    (230, 210, 60, 255),
    (60, 170, 80, 255),
    (60, 130, 220, 255),
    (140, 70, 200, 255),
    (110, 70, 40, 255),
)

_uids = itertools.count(1)


# --- the view ---------------------------------------------------------------


@dataclass
class PaintView:
    """Where the canvas sits in its pane. Per document, so a tab switch does
    not lose your place."""

    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)
    # Whether the view has been framed yet. False asks the canvas to fit on the
    # next frame it draws, which is the only moment it knows how big the pane
    # is -- the state layer never does.
    fitted: bool = False
    # A zoom to snap to on the next frame, for the same reason: "100%, centred"
    # needs the pane's size, and a keypress does not have it.
    pending_zoom: float | None = None


def clamp_zoom(zoom: float) -> float:
    return max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))


def fit(view: PaintView, size: tuple[int, int], region: tuple[float, float]) -> None:
    """Scale to show the whole document, centred."""
    width, height = size
    zoom = clamp_zoom(min(region[0] / max(width, 1), region[1] / max(height, 1)))
    view.zoom = zoom
    view.pan = (
        (region[0] - width * zoom) * 0.5,
        (region[1] - height * zoom) * 0.5,
    )
    view.fitted = True


def centre(
    view: PaintView, size: tuple[int, int], region: tuple[float, float], zoom: float
) -> None:
    """Set an explicit zoom and re-centre -- what Ctrl+1 (100%) does."""
    width, height = size
    view.zoom = clamp_zoom(zoom)
    view.pan = (
        (region[0] - width * view.zoom) * 0.5,
        (region[1] - height * view.zoom) * 0.5,
    )
    view.fitted = True


def to_image(view: PaintView, origin: tuple[float, float], sx: float, sy: float):
    """Screen -> image coordinates, as floats.

    Floats, not ints: the brush walks sub-pixel positions, and rounding here
    would quantise every stroke to the zoom level it was drawn at.
    """
    return (
        (sx - origin[0] - view.pan[0]) / view.zoom,
        (sy - origin[1] - view.pan[1]) / view.zoom,
    )


def to_screen(view: PaintView, origin: tuple[float, float], x: float, y: float):
    return (
        origin[0] + view.pan[0] + x * view.zoom,
        origin[1] + view.pan[1] + y * view.zoom,
    )


def zoom_about(
    view: PaintView, origin: tuple[float, float], mouse: tuple[float, float], steps: float
) -> None:
    """Zoom keeping whatever pixel is under the cursor under the cursor."""
    before = view.zoom
    after = clamp_zoom(before * (ZOOM_STEP**steps))
    if after == before:
        return
    local = (mouse[0] - origin[0], mouse[1] - origin[1])
    ratio = after / before
    view.zoom = after
    view.pan = (
        local[0] - (local[0] - view.pan[0]) * ratio,
        local[1] - (local[1] - view.pan[1]) * ratio,
    )


# --- one open document ------------------------------------------------------


@dataclass
class InkerDoc:
    """One tab.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two files called "input.png" the same tab,
    and would move a tab's identity every time it was renamed by a Save As.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    file_format: str = "png"  # ora | png
    uid: str = field(default_factory=lambda: f"pd{next(_uids)}")
    view: PaintView = field(default_factory=PaintView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
    saved_head: int = 0
    saving: bool = False
    # The job this document writes back into, if any. ``link_kind`` says what
    # kind of write that is; empty means the document is a plain file.
    job_id: str = ""
    link_kind: str = ""  # "" | "reference-edit"
    has_original: bool = False

    @property
    def dirty(self) -> bool:
        return self.doc.history.head != self.saved_head

    @property
    def linked(self) -> bool:
        return bool(self.job_id)

    @property
    def label(self) -> str:
        """What imgui draws on the tab. The id after ### is what it *matches*
        on, so the visible part is free to change without moving the tab."""
        return f"{self.title}###{self.uid}"

    def mark_saved(self, head: int | None = None) -> None:
        """Record which history position is now on disk.

        Captured when the *encode* starts, not when it finishes: an edit made
        while the file was being written is genuinely not in it, and clearing a
        flag here would call it saved.
        """
        self.saved_head = self.doc.history.head if head is None else head
        self.saving = False


def title_for(path: Path | None) -> str:
    return path.name if path is not None else "Untitled"


# --- everything Paint mode remembers ----------------------------------------


@dataclass
class InkerState:
    docs: list[InkerDoc] = field(default_factory=list)
    active_uid: str = ""
    recent: list[str] = field(default_factory=list)

    # Tool settings: shared across documents on purpose.
    tool: str = "brush"
    brush_size: int = 12
    hardness: float = 0.85
    opacity: float = 1.0
    strength: float = 0.5
    spacing: float = 0.1
    shape_filled: bool = False
    wand_tolerance: int = 32
    wand_contiguous: bool = True
    feather_radius: float = 2.0
    gradient_kind: str = "linear"
    gradient_to_transparent: bool = False
    symmetry: str = "none"
    grid: bool = False
    grid_size: int = 16
    fg: tuple[int, int, int, int] = (0, 0, 0, 255)
    bg: tuple[int, int, int, int] = (255, 255, 255, 255)
    swatches: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_SWATCHES)
    )

    # Drag state, decided on press because several tools start the same way.
    drag_kind: str = ""  # "" | paint | shape | marquee | lasso | move | gradient | pan
    drag_anchor: tuple[float, float] | None = None
    last_point: tuple[float, float] | None = None
    lasso: list[tuple[float, float]] = field(default_factory=list)
    combine: str = "replace"
    space_held: bool = False

    # Free transform is a *state*, not a tool: it takes over the canvas until
    # it is committed or cancelled, and every other tool is unavailable while
    # it is on -- which is exactly what "modal" means and why it cannot live in
    # the tool list beside brush and fill.
    transforming: bool = False
    # What the handle was grabbed at, so a drag is measured against the press
    # rather than against the previous frame.
    transform_ref: tuple[float, float, float, float] | None = None

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> InkerDoc | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, doc: InkerDoc) -> InkerDoc:
        self.docs.append(doc)
        self.active_uid = doc.uid
        self.clear_drag()
        return doc

    def get(self, uid: str) -> InkerDoc | None:
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

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> InkerDoc | None:
        """An already-open tab for this file, so opening twice focuses rather
        than forking -- two tabs over one path would race on save."""
        for doc in self.docs:
            if doc.path is not None and doc.path == path:
                return doc
        return None

    def find_job(self, job_id: str) -> InkerDoc | None:
        for doc in self.docs:
            if doc.job_id == job_id:
                return doc
        return None

    # -- drag ---------------------------------------------------------------

    def clear_drag(self) -> None:
        self.drag_kind = ""
        self.drag_anchor = None
        self.last_point = None
        self.lasso = []
        self.transform_ref = None

    # -- colours ------------------------------------------------------------

    def swap_colours(self) -> None:
        self.fg, self.bg = self.bg, self.fg

    def add_swatch(self, colour: tuple[int, int, int, int]) -> None:
        colour = tuple(int(c) for c in colour)  # type: ignore[assignment]
        if colour in self.swatches:
            return
        self.swatches.append(colour)
        del self.swatches[:-MAX_SWATCHES]

    # -- recent files -------------------------------------------------------

    def remember(self, path: Path | None) -> None:
        """Most recent first, deduplicated, bounded -- as the prompt history
        does, and for the same reason."""
        if path is None:
            return
        text = str(path)
        self.recent = [text] + [p for p in self.recent if p != text]
        del self.recent[MAX_RECENT:]

    def forget(self, path: str) -> None:
        """Drop a path that turned out not to open -- a recent list that keeps
        offering a moved file is worse than a short one."""
        self.recent = [p for p in self.recent if p != path]


# --- brush size stepping ----------------------------------------------------


def step_size(size: int, delta: int) -> int:
    """``[`` and ``]``, accelerating.

    A flat +/-1 makes going from 8 to 200 a chore; scaling the step with the
    size keeps every press feel proportional, which is what every paint program
    does.
    """
    from .inker import clamp_brush

    step = max(1, int(abs(size) * 0.12))
    return clamp_brush(size + (step if delta > 0 else -step))
