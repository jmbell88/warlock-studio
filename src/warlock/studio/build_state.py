"""Multi-document state for Build mode, without imgui.

The same split the raster editor makes, for the same reason: which document is
open, which is dirty, where the camera is and what the snap settings are would
all still make sense if the app were driven by a script, so none of it needs a
window to be tested.

The two conventions travel across from the raster editor unchanged, because
they are what a user arrives expecting. **Tool and snap settings belong to the
app, not to the document** -- switching tabs must not silently change your grid
size. And **the view belongs to the document** -- a tab remembers where its
camera was, because orbiting back to where you were working is not something
the user should have to redo on every tab switch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# How many files the "recent" list keeps. Ten is one screenful of a menu, the
# same number the raster editor settled on.
MAX_RECENT = 10

# Name, label, and the key that selects it. The primitive tools mirror the
# generator registry; the transform tools mirror the three gizmos.
TOOLS = (
    ("select", "Select", "Q"),
    ("move", "Move", "W"),
    ("rotate", "Rotate", "E"),
    ("scale", "Scale", "R"),
)

# The snap increments a modelling package opens on: an eighth of a unit, and
# fifteen degrees. Both are *app* settings, and both are switched off by
# setting them to zero -- which is why every snap function treats zero as the
# identity rather than as a degenerate case to guard.
DEFAULT_SNAP_TRANSLATE = 0.125
DEFAULT_SNAP_ROTATE = 15.0

WBLK_SUFFIX = ".wblk"

_uids = itertools.count(1)


@dataclass
class BuildView:
    """Where the camera sits for one document. Per document, so a tab switch
    does not lose your place."""

    yaw: float = 0.6
    pitch: float = 0.5
    distance: float = 4.0
    target: tuple[float, float, float] = (0.0, 0.5, 0.0)
    # Whether the view has been framed yet. False asks the viewport to fit on
    # the next frame it draws, which is the only moment it knows how big the
    # pane is -- the state layer never does.
    fitted: bool = False


@dataclass
class BuildTab:
    """One open document.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two documents called "Untitled" the same
    tab, and would move a tab's identity every time a Save As renamed it.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    uid: str = field(default_factory=lambda: f"bd{next(_uids)}")
    view: BuildView = field(default_factory=BuildView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
    saved_head: int = 0
    saving: bool = False
    # The asset this document was last exported to, if any. Not a link in the
    # raster editor's sense: a built asset is a *snapshot*, and editing the
    # document afterwards does not change the mesh already on disk.
    job_id: str = ""

    @property
    def dirty(self) -> bool:
        return self.doc.history.head != self.saved_head

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
    return path.stem if path is not None else "Untitled"


@dataclass
class BuildState:
    """Everything Build mode remembers across frames."""

    docs: list[BuildTab] = field(default_factory=list)
    active_uid: str = ""
    recent: list[str] = field(default_factory=list)

    # Tool and snap settings: shared across documents on purpose.
    tool: str = "select"
    snap: bool = False
    snap_translate: float = DEFAULT_SNAP_TRANSLATE
    snap_rotate: float = DEFAULT_SNAP_ROTATE
    grid: bool = True
    wireframe: bool = False

    # What the properties panel offers when the user adds something.
    generator: str = "box"
    # The object whose name is being edited in the outliner, or 0. A uid rather
    # than an index, for the reason every address in this package is one: the
    # outliner reorders and a rename in flight must not follow the position.
    renaming: int = 0

    # Drag state. ``ref`` is what the handle was grabbed at, so a drag is
    # measured against the press rather than against the previous frame --
    # which is also what feeds ``set_transform``'s ``was`` argument, and the
    # reason the gizmo itself has to remember nothing about the object.
    drag_kind: str = ""  # "" | move | rotate | scale | orbit | pan
    drag_axis: str = ""
    ref: dict[str, Any] = field(default_factory=dict)

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> BuildTab | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, tab: BuildTab) -> BuildTab:
        self.docs.append(tab)
        self.active_uid = tab.uid
        self.clear_drag()
        return tab

    def get(self, uid: str) -> BuildTab | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def close(self, uid: str) -> bool:
        tab = self.get(uid)
        if tab is None:
            return False
        index = self.docs.index(tab)
        self.docs.remove(tab)
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

    def find_path(self, path: Path) -> BuildTab | None:
        """An already-open tab for this file, so opening twice focuses rather
        than forking -- two tabs over one path would race on save."""
        for doc in self.docs:
            if doc.path is not None and doc.path == path:
                return doc
        return None

    # -- drag ---------------------------------------------------------------

    def clear_drag(self) -> None:
        self.drag_kind = ""
        self.drag_axis = ""
        self.ref = {}

    # -- recent files -------------------------------------------------------

    def remember(self, path: Path | None) -> None:
        """Most recent first, deduplicated, bounded -- as the raster editor's
        list does, and for the same reason."""
        if path is None:
            return
        text = str(path)
        self.recent = [text] + [p for p in self.recent if p != text]
        del self.recent[MAX_RECENT:]

    def forget(self, path: str) -> None:
        """Drop a path that turned out not to open -- a recent list that keeps
        offering a moved file is worse than a short one."""
        self.recent = [p for p in self.recent if p != path]
