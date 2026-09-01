"""Multi-document state for Clay, without imgui.

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


@dataclass
class LastOp:
    """The op that ran last, and everything needed to run it again differently.

    What an *adjust last operation* card is: a modeller runs Bevel, sees the
    result, and changes the width -- and what must happen then is that the
    first bevel is *undone* and a new one run, not that a second bevel is laid
    over the first. So the record has to carry the state the op ran against,
    not only its numbers.

    ``depth_before`` is where the undo stack stood; ``head_after`` is the serial
    the stack ended on, and it is the guard rather than a convenience: any other
    edit -- a gizmo drag, a rename, a second op -- moves the head, and re-running
    against a stack that has moved on would undo somebody else's work. A card
    whose ``head_after`` no longer matches is stale and hides itself.

    The selection is carried because ops read it and several of them change it:
    Extrude leaves the new faces selected, so re-running from the *current*
    selection would extrude the extrusion.

    Plain data with no methods, in the state module rather than in ``clay_ops``,
    because ``clay_ops`` may not import the pane layer and this is the shape
    both of them pass around.
    """

    name: str
    params: dict[str, float] = field(default_factory=dict)
    depth_before: int = 0
    head_after: int = 0
    element_mode: str = "object"
    element_sel: dict[int, Any] = field(default_factory=dict)
    selection: set[int] = field(default_factory=set)

_uids = itertools.count(1)


@dataclass
class CameraView:
    """Where the camera sits for one document. Per document, so a tab switch
    does not lose your place.

    Named for the camera rather than for Clay because ``ClayView`` is already
    the GL viewport class in ``studio/clay_view.py``, and two unrelated things
    under one name in one package is how a reader comes to believe the tab is
    holding the viewport. This one is plain data: four numbers and a target."""

    yaw: float = 0.6
    pitch: float = 0.5
    distance: float = 4.0
    target: tuple[float, float, float] = (0.0, 0.5, 0.0)
    # Whether the view has been framed yet. False asks the viewport to fit on
    # the next frame it draws, which is the only moment it knows how big the
    # pane is -- the state layer never does. A document opened from a file that
    # carried a camera arrives already True, which is the whole point of storing
    # one: framing over it would throw away the answer just read off disk.
    fitted: bool = False

    def read_from(self, camera: Any) -> None:
        """Take the live camera's angles. ``theta``/``phi`` are its names for
        yaw and pitch; the two spellings meet here and nowhere else.

        The **goals** are read rather than the current values. The camera damps
        toward them, so mid-ease the two differ -- and what a user means by
        "where I left the camera" is where they pointed it, not the frame the
        tab switch happened to interrupt.
        """
        self.yaw = float(getattr(camera, "_goal_theta", camera.theta))
        self.pitch = float(getattr(camera, "_goal_phi", camera.phi))
        self.distance = float(getattr(camera, "_goal_distance", camera.distance))
        self.target = tuple(float(v) for v in getattr(camera, "_goal_target", camera.target))

    def write_to(self, camera: Any) -> None:
        """Put these angles back on the live camera, goals included.

        Both halves, or the camera eases straight back to wherever it was: the
        angles are what the frame draws and the goals are what it converges to,
        and setting one without the other is a restore that undoes itself over
        the next few frames.
        """
        camera.theta = camera._goal_theta = self.yaw
        camera.phi = camera._goal_phi = self.pitch
        camera.distance = camera._goal_distance = self.distance
        camera.set_target(self.target)


@dataclass
class ClayTab:
    """One open document.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two documents called "Untitled" the same
    tab, and would move a tab's identity every time a Save As renamed it.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    uid: str = field(default_factory=lambda: f"bd{next(_uids)}")
    view: CameraView = field(default_factory=CameraView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
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
class ClayState:
    """Everything Clay remembers across frames."""

    docs: list[ClayTab] = field(default_factory=list)
    active_uid: str = ""
    #: ``F`` has been pressed and the viewport has not framed yet.
    #:
    #: A flag rather than a call, the house pattern (``plotter_state``'s
    #: ``setup_pending``, Inker's ``pending_dialog``): framing needs the
    #: viewport, which is a thing ``main`` owns and ``clay_mode`` may not
    #: import, so the mode records the *intent* and the pane consumes it. What
    #: it replaces is a branch in ``App._shortcut`` that reached past
    #: ``clay_mode.handle_key`` to do the same thing -- the one Clay binding
    #: that did not live with the others.
    frame_pending: bool = False

    # Tool and snap settings: shared across documents on purpose.
    tool: str = "select"
    snap: bool = False
    snap_translate: float = DEFAULT_SNAP_TRANSLATE
    snap_rotate: float = DEFAULT_SNAP_ROTATE
    # Snap a move onto the vertex under the cursor, in preference to the grid.
    # A *separate* switch rather than a mode of ``snap``, because the two answer
    # different questions -- "put it on round numbers" and "put it exactly
    # there" -- and a user aligning two parts wants the second without giving up
    # the first everywhere else. Off by default: it changes what a plain drag
    # does, and a viewport that silently jumps is worse than one that does not.
    snap_vertex: bool = False

    # Proportional editing: an element drag carries the geometry around the
    # selection with it, fading out over ``proportional_radius`` metres of world
    # space. Off by default and radius-driven rather than count-driven, because
    # a radius is the thing the user can see -- a "how many rings" control means
    # nothing on an imported mesh whose density varies across it.
    proportional: bool = False
    proportional_radius: float = 0.5
    grid: bool = True

    # How the surface itself is drawn: "solid", "material" or "wireframe".
    #
    # Three modes rather than the one wireframe toggle this replaced, and the
    # addition that matters is **Solid** -- the albedo with no lighting, which
    # is what a modeller works in: it shows silhouette and topology without a
    # specular highlight sitting on the vertex being dragged. Material is the
    # lit render and is what the object will look like; wireframe replaces the
    # fill entirely.
    shading: str = "material"

    # The see-through pass. Off by default because it changes what a click
    # picks as well as what is drawn -- an element behind the surface becomes
    # reachable, which is the whole point and is also a surprise if it happens
    # without being asked for.
    xray: bool = False

    # What the viewport draws *over* the model, by name. A dict rather than a
    # field apiece because the header's popover is a loop over
    # ``clay_header.OVERLAY_ROWS`` and a sixth overlay should be one line there
    # rather than one line in four files.
    #
    # ``grid`` is deliberately **not** in it: it is wired straight to
    # ``ClayView.show_grid`` and has been since the viewport existed, and a
    # second home for one switch is two places that can disagree about it.
    overlays: dict[str, bool] = field(default_factory=lambda: {"wire": False})

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

    # The parameterised op whose popup is open, by name, and the values every
    # such op was last run with. Remembered per op rather than per invocation:
    # a user beveling six edges in turn wants the same width each time, and
    # retyping it is the whole reason a modeller keeps the last value.
    pending_op: str = ""
    op_params: dict[str, dict[str, float]] = field(default_factory=dict)
    # Set when something outside the pane wants that popup *opened*, because
    # ``imgui.open_popup`` only takes effect inside the window whose id stack
    # is current -- the keyboard path runs in the event layer, where there is
    # no window at all. Without it a bare-letter key bound to a parameterised
    # op set ``pending_op`` and nothing ever opened the popup, leaving the mode
    # holding a request it could not act on. Cleared by the pane that opens it.
    open_op_popup: bool = False

    # Where a Shift+click range in the outliner is measured from. A uid, for the
    # reason every address in this package is one: the list reorders, and an
    # anchor that was an index would silently point at a different row.
    outliner_anchor: int = 0

    # The last manifold check, per object: the ``Mesh`` it measured and the rows
    # it produced. Held here rather than recomputed because ``check_manifold``
    # builds a whole adjacency -- O(corners), and not something to run sixty
    # times a second to redraw a line that has not changed.
    #
    # **Keyed on the mesh object, not on a revision or an id.** A ``Mesh`` is
    # immutable and every op replaces it, so ``obj.mesh is measured`` is exactly
    # "this result is still about what is on screen"; an ``id()`` would be
    # recycled by the allocator onto a different mesh and silently report last
    # edit's holes. Keeping the mesh alive is the price, and it is one mesh per
    # object the user has actually asked about.
    manifold: dict[int, tuple[Any, list[Any]]] = field(default_factory=dict)

    # The op that ran last, or None. Recorded by ``clay_ops.run`` and read by
    # the adjust-last-operation card and by Repeat. Session state and never
    # serialized: it names a position in an undo stack, which is the one thing
    # that cannot survive a reopen.
    last_op: LastOp | None = None

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> ClayTab | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, tab: ClayTab) -> ClayTab:
        self.docs.append(tab)
        self.active_uid = tab.uid
        self.clear_drag()
        return tab

    def get(self, uid: str) -> ClayTab | None:
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

    def find_path(self, path: Path) -> ClayTab | None:
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

