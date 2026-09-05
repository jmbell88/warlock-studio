"""Clay's controller: opening, saving, exporting, guarding and keys.

Everything here is *about* documents rather than geometry -- the engine under
``clay/`` has no idea a job or a task thread exists, and this is the layer that
knows about both. The panes draw; this decides.

The one rule that shapes the whole file is the raster editor's: **no file dialog
and no encode ever runs on the frame thread.** A native picker is modal to the
OS and blocks until dismissed, and a document of any size is a zip to build.
Both go through ``ctx.submit`` and come back through :func:`on_task_done`, which
is why saving is a *state* (``ClayTab.saving``) rather than a function call
that returns.

Two consequences follow, and both were bugs in the raster editor before they
were rules here.

**A failed save must clear that state.** ``saving`` gates every control that
changes the document, so without :func:`on_task_failed` one failed write leaves
the tab permanently read-only with no way back short of closing it. The same
applies to a submit the runner *refuses* -- a second save while one is in flight
-- which is why :func:`_start` unsets the flag on a False return.

**The head a save records is read after the document settles**, at exactly one
place. A head captured before whatever the save itself pushes saves the document
against a head one behind it, and dirty -- being a comparison against that head
-- then stays true however many times the user saves.

Every task key carries the ``clay-`` prefix, because the app claims results by
prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from . import atomic, clay_state, dialogs, docmodes, journal, sizeguard
from .clay_state import ClayState, ClayTab

log = logging.getLogger(__name__)

WBLK_FILTER = ["Warlock Clay document (*.wblk)", "*.wblk"]


def _path_key(path: Path) -> str:
    """A short, stable id for a path, safe to fold into a task key.

    ``hash(str(path))`` is salted per process (``PYTHONHASHSEED``) and, at 64
    bits, two different paths in the same session can still land on the same
    ``abs()`` value -- a collision silently drops the second open rather than
    submitting it. sha1 has neither problem: same input, same digest, always.
    """
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def ensure(ctx: Any) -> ClayState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Clay should not pay for it,
    and because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.clay
    if state is None:
        state = ClayState()
        ctx.state.clay = state
    return state


# The three recents wrappers every document mode carries, over the one
# list Home's Resume rows are built from (``docmodes.recents_for``).
remember_path, forget_path, recent_paths = docmodes.recents_for("clay")


def persist(ctx: Any) -> None:
    """Nothing to write any more: the recent list moved to :mod:`.recents`,
    which persists itself on every write. Kept as a no-op because it is called
    from a dozen places after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""



def active(ctx: Any) -> ClayTab | None:
    state = ctx.state.clay
    return state.active if state is not None else None


def _enter_clay(ctx: Any) -> None:
    """Adoption switches modes through ``state.set_mode``, never by assignment.

    A bare assignment to ``state.mode`` skips the ``previous_mode`` /
    ``mode_observed`` pair that function maintains -- the drift its own
    docstring names -- so Esc out of the next pass-through mode would go back
    to wherever a *keypress* last was rather than to Clay. (Worded to stay out
    of ``tests/test_mode_writes.py``'s line scan, which cannot tell prose from
    code.)
    """
    from .state import set_mode

    set_mode(ctx.state, "clay")


# --- opening ----------------------------------------------------------------


def adopt(
    ctx: Any,
    doc: Any,
    *,
    path: Path | None = None,
    title: str | None = None,
    view: dict[str, Any] | None = None,
) -> ClayTab:
    state = ensure(ctx)
    tab = ClayTab(
        doc=doc,
        title=title or clay_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    if view:
        tab.view.yaw = view["yaw"]
        tab.view.pitch = view["pitch"]
        tab.view.distance = view["distance"]
        tab.view.target = view["target"]
        # Already framed, which is the whole point of having stored one: an
        # auto-fit over the top would throw away the answer just read off disk.
        tab.view.fitted = True
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> ClayTab:
    from .clay import document as bd

    return adopt(ctx, bd.ClayDoc(), title="Untitled")


def _within_ceiling(path: Path) -> Path:
    """Refuse a document too big to open, before a byte of it is read.

    Clay had **no size ceiling anywhere**, though
    ``service.files.MAX_CLAY_SOURCE_BYTES`` has existed since the format did --
    applied at exactly one place, the *upload*, and at neither of the two doors
    a user reaches. The ceiling is that same number rather than a second one
    invented here, for ``plotter_io``'s reason: "how big may a clay document
    be" has one answer, and two would drift the first time either moved.
    """
    from ..service.files import MAX_CLAY_SOURCE_BYTES

    return sizeguard.within_ceiling(path, MAX_CLAY_SOURCE_BYTES)


def _within_mesh_ceiling(path: Path) -> Path:
    """The same question about a GLB, which is a different number.

    ``MAX_MESH_BYTES`` and not the document ceiling: an imported mesh is what
    the service already accepts as an *uploaded* mesh, and a hundred-thousand
    triangle ``model.glb`` is the ordinary case rather than the extreme one.
    """
    from ..service.validation import MAX_MESH_BYTES

    return sizeguard.within_ceiling(path, MAX_MESH_BYTES)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from .clay import serialize

    data = _within_ceiling(Path(path)).read_bytes()
    doc = serialize.read_wblk(data)
    return {
        "doc": doc,
        "path": str(path),
        "title": clay_state.title_for(Path(path)),
        # A second read of the same bytes, deliberately: see ``read_view``'s own
        # docstring for why the camera is not a second return value from
        # ``read_wblk``. It parses one small JSON member of an in-memory zip.
        "view": serialize.read_view(data),
    }


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open Clay document", WBLK_FILTER)
        return None if path is None else _load(path)

    ctx.submit("clay-open", run)


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"clay-open:{_path_key(path)}", _load, path)


# --- importing --------------------------------------------------------------

# Above this an import gets a confirm dialog first. Not a refusal -- Clay can
# edit it, and a user who has just asked to edit their own asset should be
# allowed to -- but a rebuild-per-edit at this scale is a visible pause, and
# finding that out by pressing Extrude is worse than being told.
SLOW_TRIANGLES = 200_000


def import_glb_path(ctx: Any, path: Path) -> None:
    """Parse a GLB on a task thread and adopt it as a document.

    The parse and the merge are both O(triangles) and a ``model.glb`` is
    routinely a hundred thousand of them, so neither runs on the frame thread --
    the same rule every dialog and every encode in this module follows.
    """
    ensure(ctx)
    path = Path(path)

    def run() -> dict[str, Any]:
        return _parse_glb(_within_mesh_ceiling(path).read_bytes(), path.stem)

    ctx.submit(f"clay-import:{_path_key(path)}", run)


def edit_asset_in_clay(ctx: Any, job: Any) -> None:
    """Open a library asset in Clay: its authored document if it has one.

    The ``build.wblk`` sidecar is preferred whenever it is there, and that is
    the point of the whole feature -- it is the document the user actually
    authored, with its objects, its names and its generator parameters intact,
    and until now it was a file written and never read back. Failing that, the
    *optimized* ``model.glb`` is imported: it is the mesh that is served,
    grounded and exported, and ``source.glb`` is the raw reconstruction that
    nothing downstream uses.
    """
    ensure(ctx)
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") if isinstance(job, dict) else "") or "Asset"

    def run() -> dict[str, Any]:
        from ..service import files as svc_files

        sidecar = svc_files.clay_source_path(ctx.svc, job_id)
        if sidecar.exists():
            return _load(sidecar)
        mesh = ctx.svc.config.job_dir(job_id) / "model.glb"
        if not mesh.exists():
            raise FileNotFoundError(f"{job_id} has no mesh to edit")
        return _parse_glb(_within_mesh_ceiling(mesh).read_bytes(), name)

    ctx.submit(f"clay-import:{job_id}", run)


def _parse_glb(data: bytes, name: str) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from .clay import glbimport

    doc = glbimport.glb_to_claydoc(data, name=name)
    triangles = sum(
        max(len(obj.mesh.starts) - 1, 0) for obj in doc.objects
    )
    return {"doc": doc, "title": name, "triangles": triangles}


def _adopt_import(ctx: Any, result: dict[str, Any]) -> None:
    """Adopt a parsed import, asking first when it is big enough to be slow."""
    doc, title = result["doc"], result.get("title") or "Imported"
    # A ``.wblk`` sidecar carries a camera; a GLB does not, and ``None`` is
    # simply "frame it", which is what an import has always done.
    view = result.get("view")
    if int(result.get("triangles", 0)) <= SLOW_TRIANGLES:
        adopt(ctx, doc, title=title, view=view)
        _enter_clay(ctx)
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Edit this mesh in Clay?",
            message=(
                f"{int(result['triangles']):,} faces. Editing will be slow -- "
                "every edit rebuilds the whole mesh, and the undo history holds "
                "two copies per step."
            ),
            confirm_label="Edit anyway",
            cancel_label="Cancel",
            on_confirm=lambda: _adopt_now(ctx, doc, title, view),
        )
    )


def _adopt_now(ctx: Any, doc: Any, title: str, view: dict[str, Any] | None = None) -> None:
    adopt(ctx, doc, title=title, view=view)
    _enter_clay(ctx)


# --- saving -----------------------------------------------------------------


def camera_of(ctx: Any, tab: ClayTab) -> Any:
    """The tab's stored camera, refreshed from the live viewport first.

    Called on every path that writes the document, because ``tab.view`` is only
    brought up to date when the tab is switched away from -- so saving the tab
    you are looking at would otherwise store wherever the camera was when you
    last left it, which is the one case where the answer is visibly wrong.
    """
    view = getattr(ctx, "clay_view", None)
    if view is not None and getattr(view, "camera", None) is not None:
        tab.view.read_from(view.camera)
    return tab.view


def remember_camera(ctx: Any, tab: ClayTab | None) -> None:
    """Snapshot the live camera onto a tab that is being switched away from."""
    if tab is not None:
        camera_of(ctx, tab)


def apply_camera(ctx: Any, tab: ClayTab) -> None:
    """Put a tab's camera back on the viewport, or frame it if it has none.

    Framing here rather than in ``ClayView`` because this is the layer that
    knows a *tab* exists: the viewport has one camera and no idea that the thing
    it is drawing changed identity.
    """
    view = getattr(ctx, "clay_view", None)
    if view is None or getattr(view, "camera", None) is None:
        return
    if tab.view.fitted:
        tab.view.write_to(view.camera)
        return
    view.frame_selection(tab.doc)
    tab.view.read_from(view.camera)
    tab.view.fitted = True


# The submit-or-unlock helper is one rule for all four document modes and lives
# in :mod:`.docmodes`; bound here as an assignment rather than wrapped, because
# every call site in this file reaches for one object.
_start = docmodes.start_save


def save_to(ctx: Any, tab: ClayTab, path: Path) -> None:
    """Write the document to a known path.

    The head is read *here*, before the submit and after the document is in
    whatever state the save will encode -- one place, so the two halves of the
    dirty comparison cannot drift apart.
    """
    from .clay import serialize

    path = Path(path)
    doc = tab.doc
    rev = doc.history.head
    data = serialize.wblk_bytes(doc, view=camera_of(ctx, tab))

    def run() -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic.write_bytes(path, data)
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"clay-save:{tab.uid}", run)


def save(ctx: Any, tab: ClayTab | None = None) -> None:
    tab = tab or active(ctx)
    docmodes.save(
        tab, save_as=lambda: save_as(ctx, tab), save_to=lambda: save_to(ctx, tab, tab.path)
    )


def save_as(ctx: Any, tab: ClayTab | None = None) -> None:
    """The picker and the encode on one task thread.

    The bytes are built on the frame thread and the *picker* is what runs on
    the task thread, which is the opposite of what it looks like it should be:
    serialising reads the live document, and doing that after an unbounded
    modal dialog would encode whatever the user did while it was open.
    """
    from .clay import serialize

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    rev = doc.history.head
    data = serialize.wblk_bytes(doc, view=camera_of(ctx, tab))

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save Clay document", f"{title}{clay_state.WBLK_SUFFIX}", WBLK_FILTER
        )
        if path is None:
            return None
        path = path.with_suffix(clay_state.WBLK_SUFFIX)
        # Staged, as ``save_to`` is: a picker aimed at an existing document is
        # the ordinary way to overwrite one, and a write that dies partway
        # through would leave that file truncated with no copy of it anywhere.
        # No mkdir -- the picker returns a directory that exists.
        atomic.write_bytes(path, data)
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"clay-saveas:{tab.uid}", run)


# --- export -----------------------------------------------------------------


def export_asset(ctx: Any, tab: ClayTab | None = None) -> None:
    """Mint an ordinary asset from the document: the point of Clay.

    What comes out is a ``done`` model row, so rigging, posing, sprite sheets,
    the triangle retarget and every mesh export work on it with none of those
    paths learning that Clay exists.

    The mesh is written first and the ``.wblk`` sidecar second, so a crash
    between them leaves the sidecar absent rather than lying about a mesh it
    did not produce. Both the GLB and the document are built on the frame
    thread for the reason ``save_as`` states, and only the service calls go to
    the task thread.
    """
    from .clay import document as bd
    from .clay import serialize
    from .viewer import glbwrite

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not any(obj.visible for obj in doc.objects):
        # Refused here rather than at the service door, so no job directory is
        # ever created for it: check_glb would refuse the same bytes, but only
        # after this had told the user a build was under way.
        ctx.toast("There is nothing visible to export.", "error")
        return

    glb = glbwrite.write_glb(bd.to_model(doc))
    wblk = serialize.wblk_bytes(doc, view=camera_of(ctx, tab))

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from ..service import jobs as svc_jobs

        result = svc_jobs.import_mesh(ctx.svc, glb, name=title, prompt=title)
        job_id = result["id"]
        svc_files.save_clay_source(ctx.svc, job_id, wblk)
        return {"job_id": job_id, "exported": True}

    _start(ctx, tab, f"clay-export:{tab.uid}", run)


# --- task results -----------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``clay-`` key."""
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "clay-open":
        if isinstance(result, dict):
            adopt(
                ctx,
                result["doc"],
                path=Path(result["path"]),
                title=result.get("title"),
                view=result.get("view"),
            )
            _enter_clay(ctx)
        return

    if name == "clay-recover":
        if result is None:
            journal.adopt_failed(ctx, "model")
        if isinstance(result, dict):
            tab = adopt(ctx, result["doc"], path=None, title=result.get("title"))
            docmodes.mark_recovered(tab, result["autosave"])
            _enter_clay(ctx)
        return

    if name == "clay-import":
        # No path: an imported document has no file of its own, so Ctrl+S asks
        # where to put it rather than overwriting the asset it came from.
        if isinstance(result, dict):
            _adopt_import(ctx, result)
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        ctx.cache.invalidate()
        return
    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported"):
        tab.job_id = result["job_id"]
        ctx.cache.invalidate()
        ctx.toast("Exported as an asset.")
        return

    tab.mark_saved(result.get("rev"))
    # See ``inker_mode``: saved is the moment the crash copy stops
    # describing anything at risk (UX-05).
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = clay_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked.

    ``saving`` disables every editing control, so without this a single failed
    write makes the tab permanently read-only with no way back short of
    closing it.
    """
    state = ctx.state.clay
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


# --- the guard --------------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them: ``ConfirmQueue`` holds a single pending
    question, so asking per dirty document would silently drop all but the
    first. Only quitting and closing a tab are destructive -- switching modes
    is not, because Clay is a mode rather than a takeover and its tabs are
    still there when you come back.
    """
    return docmodes.guard(ctx, "clay", "document", "documents", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """Close one document, asking first if it has unsaved work.

    ``ClayState.close`` has been here since the multi-document work and had no
    caller at all: Clay could open documents and never shut one, which also
    meant ``guard``'s "3 documents have unsaved changes" named documents the
    user had no way to reach -- Ctrl+Tab cycled them with nothing on screen
    saying so.

    ``docmodes.close_tab`` asks the question and refuses a tab mid-save; what
    is Clay's is the release below.

    What a Clay document owns in the single GL context is the part worth
    stating. The per-document camera is plain data on the tab
    (``clay_state.CameraView``); the GPU buffers live in ``ctx.clay_view``, whose
    ``_cache`` is keyed on *object* uid and holds whichever document was last
    synced. Dropping the document those buffers were built from without
    releasing them leaves the renderer one frame away from drawing a mesh that
    no longer belongs to anything -- so the cache is cleared, which costs one
    frame of rebuild and is what ``sync`` does on every tab switch anyway.
    """
    state = ensure(ctx)

    def release(tab: ClayTab) -> None:
        view = getattr(ctx, "clay_view", None)
        if view is not None and tab.uid == state.active_uid:
            if getattr(view, "dragging", False):
                # Before the clear, and against the document it was started on:
                # a drag holds the mesh it is moving.
                view.cancel_drag(tab.doc)
            view.clear()
        # Keyed on object uid, so only this document's entries go: the uid
        # counter is process-wide and never rewinds, so nothing can collide
        # with a stale entry, and clearing the whole table made every other
        # open tab redo its (adjacency-building) checks on the next draw.
        for obj in tab.doc.objects:
            state.manifold.pop(obj.uid, None)

    docmodes.close_tab(ctx, state, uid, release)


# --- keys -------------------------------------------------------------------

# Q/W/E/R, which is where a user coming from Blender or Unity puts their left
# hand. Held here rather than in the pane so the mapping is testable.
TOOL_KEYS = {
    "q": "select",
    "w": "move",
    "e": "rotate",
    "r": "scale",
}

# The element modes, on the number row. **Not Tab**, which imgui's keyboard
# navigation owns and which would move focus out of the viewport as well as
# changing the mode; and **not b**, because the hand that is about to press
# Ctrl+Z is already on the number row. 4 is object mode rather than a separate
# key, so the four modes are one contiguous run under four fingers.
ELEMENT_KEYS = {"1": "vertex", "2": "edge", "3": "face", "4": "object"}

# Ctrl+digit axis views, on the numbers a modeller's hand already knows from
# Blender's numpad. **Bound here rather than in ``App._shortcut``**: a global
# binding is checked above the workspace modes and takes its key from them
# permanently, which is the whole reason the mode switch moved to Alt. These
# keys belong to Clay and only Clay.
AXIS_VIEW_KEYS = {"1": "front", "3": "right", "7": "top"}

def axis_view_key(camera: Any, name: str, shift: bool) -> bool:
    """One Ctrl+digit view key, on any camera. -> whether ``name`` was one.

    Shift is the opposite view, as Blender's numpad does it -- Ctrl+1 is the
    front and Ctrl+Shift+1 the back, so six views cost three keys; Ctrl+5
    toggles orthographic. **Shared with Poser rather than restated there**, so
    the two 3-D viewports cannot come to disagree about which number is the
    front -- and so that both require Ctrl. Poser's copy tested the bare
    digit, so a 1 typed into nothing snapped its camera while Clay's did not.
    """
    if name in AXIS_VIEW_KEYS:
        wanted = AXIS_VIEW_KEYS[name]
        if shift:
            wanted = {"front": "back", "right": "left", "top": "bottom"}[wanted]
        camera.look_along(wanted)
        return True
    if name == "5":
        camera.orthographic = not camera.orthographic
        return True
    return False


# Ctrl-shortcuts that change the document. Serialising reads the live document
# on a task thread, so anything that restructures it or moves the history head
# the save captured waits for the save, exactly as a gizmo drag does.
_MUTATING_CTRL = docmodes.WRITE_CHORDS | frozenset({"a", "i", "j", "m"})
#: The chords a live drag swallows -- history and the tab; see ``_ctrl_key``.
#: Deliberately *not* the whole of ``_MUTATING_CTRL``: Ctrl+J under a drag is a
#: pinned behaviour (``test_ctrl_chords_still_reach_their_ops_during_a_drag``).
_DRAG_BLOCKED_CTRL = frozenset({"z", "y", "n", "o", "tab", "w", "s"})


# --- history ------------------------------------------------------------------
#
# One call per direction, rather than two lines under the key handler, because
# the bridge panel draws the same Undo/Redo pair Inker's does. Clay, Plotter and
# Packwright each had a full undo stack and no on-screen control at all, so the
# feature existed only for a user who already knew the chord -- and every
# side effect a step has (nothing, here) belongs to *undoing*, not to the
# keyboard.


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it."""
    tab.doc.undo()



def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump the document to a position in its undo stack. -> whether it moved.

    The history panel's door, and the *third* surface onto the same stack --
    which is why it is here beside the other two rather than in the pane, and
    why the pane will not call ``doc.step_history`` itself. ``plotter_mode``
    has the same three, for the same reason written out there.

    A jump can undo the op the adjust card is offering to re-run, so the record
    of it is dropped: the card's own guard is that the head has not moved, and
    leaving a stale ``last_op`` behind would make Repeat replay an op against a
    document that no longer has what it ran on.
    """

    moved = tab.doc.step_history(index)
    if moved:
        ensure(ctx).last_op = None
    return moved



def handle_key(ctx: Any, event: Any) -> bool:
    """Clay's shortcuts. -> whether the key was consumed.

    **False with nothing open**, which the caller's fall-through depends on:
    Clay owns a viewport, and with no document the viewport's own
    shortcuts must still work. With a document open the key is consumed
    unconditionally, because falling through would let it act on a pane Clay
    mode has replaced.
    """
    import pygame

    if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == "f" and not (
        event.mod & (pygame.KMOD_CTRL | pygame.KMOD_ALT)
    ):
        # Recorded, not done: see ``ClayState.frame_pending``.
        ensure(ctx).frame_pending = True
        return True

    state = ctx.state.clay
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.type != pygame.KEYDOWN:
        return True

    # Off ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (UX-12): ``mod`` is the modifier state at the instant this key was
    # pressed, where ``get_mods()`` is the state *now*, after the event batch
    # drained, so a Ctrl released between the two made a fast chord fall
    # through as the bare letter.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key)

    # A live gizmo drag owns the bare keys, and it has to be asked *first*: the
    # number row is bound to the element modes, so a "1" typed into a drag would
    # otherwise jump into vertex mode halfway through moving something. Esc
    # cancels the drag rather than falling through to the staged clear, and
    # Enter commits it -- neither means anything else while one is under way.
    view = getattr(ctx, "clay_view", None)
    if not ctrl and view is not None and getattr(view, "dragging", False):
        if event.key == pygame.K_ESCAPE:
            view.cancel_drag(doc)
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            view._release_drag(doc)
            return True
        view.drag_key(doc, name)
        # Consumed whether or not the drag wanted it. Falling through here put
        # every unclaimed bare key into the op registry below, so ``E`` typed
        # mid-``G`` ran Extrude against the mesh the drag was still moving --
        # and the drag's own commit then measured from ``_drag_start``, the
        # pre-drag baseline, and reverted it. The rule the comment above states
        # is only a rule if it holds for the keys the drag does *not* know.
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, shift=shift)

    if name in ELEMENT_KEYS and not shift:
        if not tab.saving:
            doc.set_element_mode(ELEMENT_KEYS[name])
    elif not shift and (
        # The registry first, and the ``or`` short-circuits, so a letter an
        # element mode has claimed still fires its op rather than starting a
        # drag -- which is the ordering ``DRAG_KEYS`` records the reason for.
        _registry_key(ctx, tab, doc, name)
        or _keyboard_drag(ctx, view, tab, doc, name)
    ):
        pass
    elif name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif event.key == pygame.K_DELETE:
        if not tab.saving:
            _delete(ctx, doc)
    elif event.key == pygame.K_ESCAPE:
        _escape(state, tab, doc)
    return True



#: The two letters that start a transform with no handle grabbed, and what each
#: starts. **G and S only** -- not R, and the omission is the one interesting
#: thing about the table.
#:
#: ``R`` is the Scale *tool*'s letter and ``E`` is Rotate's, both taken long
#: before this and both in ``clay_state.TOOLS``; taking either back for a drag
#: would move a binding a user already has. What is free is ``G``, which every
#: modelling package uses for grab, and ``S``, which every one of them uses for
#: scale. Rotate is reached mid-drag instead -- ``G`` then ``R`` -- which is a
#: gesture Blender has anyway and which costs nothing here, because switching
#: transforms mid-drag had to work regardless.
#:
#: Checked *after* the op registry, so a letter an element mode has claimed
#: still fires its op: ``S`` is nothing in the registry today, and the ordering
#: is what keeps that from being a thing to remember if it ever is.
DRAG_KEYS = {"g": "move", "s": "scale"}


def _keyboard_drag(ctx: Any, view: Any, tab: ClayTab, doc: Any, name: str) -> bool:
    """Start a keyboard transform. -> whether the key was one.

    Refused while the tab is saving, like every control that changes the
    document -- and refused with nothing selected, where it would be a drag with
    nothing to drag and would swallow a keystroke that means nothing else.
    """

    kind = DRAG_KEYS.get(name)
    if kind is None or view is None or tab.saving:
        return False
    return bool(view.begin_keyboard_drag(doc, kind))


def _registry_key(ctx: Any, tab: ClayTab, doc: Any, name: str) -> bool:
    """Fire the registry op bound to a bare letter, if there is one.

    Checked *before* the tool keys so an element mode can claim a letter the
    transform tools also use -- E is Extrude with faces selected and Rotate
    without -- and checked through ``clay_ops.menu`` so the binding shown in the
    context menu and the binding that fires are one value.
    """
    from . import clay_ops

    if tab.saving or doc.element_mode == "object":
        return False
    op = clay_ops.by_key(doc.element_mode, name.upper())
    if op is None or not op.enabled(doc):
        return False
    return _fire_op(ctx, doc, op)


def _fire_op(ctx: Any, doc: Any, op: Any) -> bool:
    """Run a registry op from the event layer, popping its dialog if it has one.

    Shared by the bare-letter path and the Ctrl-shortcut path so a
    parameterised op bound to either kind of key behaves the same way.
    """
    from . import clay_ops

    state = ensure(ctx)
    if op.params:
        state.pending_op = op.name
        state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
        # Asked for rather than opened: ``imgui.open_popup`` only takes effect
        # inside the window whose id stack is current, and this runs in the
        # event layer.
        state.open_op_popup = True
        return True
    return clay_ops.run(ctx, doc, op)


def _delete(ctx: Any, doc: Any) -> None:
    """``clay.selection.delete_selected``, with its refusals shown as toasts.

    The rule and the reasoning are that function's; what belongs to this layer
    is turning a returned sentence into something the user sees.
    """
    from .clay import selection

    for message in selection.delete_selected(doc):
        _toast(ctx, message)


def _escape(state: ClayState, tab: ClayTab, doc: Any) -> None:
    """Esc, staged: the elements, then the mode, then the objects.

    One key that undoes the last thing the user got into, in the order they got
    into it. It **never leaves Clay mode**: Esc means "drop what I am doing",
    and losing a workspace full of tabs to a stray keypress is not that.
    """
    if not tab.saving:
        if doc.element_mode != "object":
            if doc.element_sel:
                doc.clear_element_sel()
            else:
                doc.set_element_mode("object")
        else:
            doc.select([])
    # Always: abandoning a half-finished drag touches the pane's own state and
    # never the document, so it is safe mid-save.
    state.clear_drag()


def _toast(ctx: Any, message: str) -> None:
    """``docmodes.refuse``: the one refusal door the non-Inker modes share."""
    docmodes.refuse(ctx, message)


def _ctrl_key(
    ctx: Any, state: ClayState, tab: ClayTab, doc: Any, name: str, *, shift: bool
) -> bool:
    if docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL):
        return True
    view = getattr(ctx, "clay_view", None)
    if getattr(view, "dragging", False) and name in _DRAG_BLOCKED_CTRL:
        # A live gizmo drag mutates the geometry in place and commits on
        # release. Ctrl+Z under it undid a step the release then re-applied;
        # Ctrl+N/Tab/O/W swapped the tab out from under the grab and left the
        # moved vertices with no edit and ``dirty`` false. The chords that
        # change history or the tab wait for the release; the view chords and
        # the object ops go on working.
        return True
    if name in AXIS_VIEW_KEYS or name == "5":
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            axis_view_key(view.camera, name, shift)
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "o":
        ask_open(ctx)
    elif name == "n":
        new_document(ctx)
    elif name == "w":
        # Beside its siblings, and the same key Inker, Plotter and Packwright
        # already close a document with.
        close_tab(ctx, tab.uid)
    elif name == "e" and not shift:
        # Not the shifted half: Ctrl+Shift+E was a silent alias of Ctrl+E here,
        # which is the one thing a printed binding exists to prevent
        # (``inker_keys``'s rule). Clay's file export *is* its library export,
        # so the chord every other mode gives the file export has nothing
        # distinct to do here and does nothing.
        export_asset(ctx, tab)
    elif name == "z":
        redo(ctx, tab) if shift else undo(ctx, tab)
    elif name == "y":
        redo(ctx, tab)
    elif name == "a":
        _select_all(doc)
    elif name == "i" and shift:
        _invert(doc)
    elif name == "m" and doc.element_mode == "object":
        # **Merge is Ctrl+M, and Duplicate is Ctrl+J.** Clay used to be the
        # editor that disagreed with the other three about both keys: Ctrl+D
        # duplicated here and deselects in Inker and Plotter, and Ctrl+J merged
        # here and duplicates in Plotter (whose comment names the raster
        # editor's "copy this to its own layer" as the same idea). Two chords
        # meaning two different things in two workspaces of one app is a user
        # pressing the one they learned and getting the other verb.
        #
        # Object mode only, for the reason Ctrl+J is: a merge is about whole
        # objects, and there is no element-mode reading of it to fall back on.
        # Shift picks the union rather than the weld -- one predicate gates
        # both, so the shift never changes whether the key does anything, only
        # which of the two answers it gives.
        from . import clay_ops

        op = clay_ops.get("union" if shift else "join")
        if op.enabled(doc):
            _fire_op(ctx, doc, op)
    elif name == "j" and doc.element_mode == "object":
        # Object mode only: duplicating a *face* selection is a different
        # operation with a different name, and doing the object one instead
        # would silently double a mesh the user is mid-edit on.
        _duplicate_selection(ctx, state, doc)
    elif name == "d" and not tab.saving:
        # Deselect, which is what it does in Inker and in Plotter. Not gated on
        # the element mode: clearing a selection means something in all four.
        # Unlike Esc it is *only* the deselect -- no mode step, no drag cancel
        # -- because a chord a user reaches for deliberately should do one
        # thing, and the staged key already exists for the other reading.
        if doc.element_mode != "object":
            doc.clear_element_sel()
        else:
            doc.select([])
    elif name in GROW_KEYS:
        # Ctrl+plus and Ctrl+minus, on both the number row and the keypad.
        # Four names for two verbs, because the two rows report different key
        # names for the same glyph and a user pressing the one under their hand
        # should not have to know which.
        from . import clay_ops

        op = clay_ops.get(GROW_KEYS[name])
        if op.enabled(doc) and doc.element_mode in op.modes:
            _fire_op(ctx, doc, op)
    elif name == "tab":
        state.cycle(-1 if shift else 1)
    return True


#: The two selection-size chords, by the key names pygame reports. ``=`` is the
#: unshifted key that carries ``+``, which is what a user presses; ``[+]`` and
#: ``[-]`` are the keypad's own names.
GROW_KEYS = {
    "=": "select-more",
    "+": "select-more",
    "[+]": "select-more",
    "-": "select-less",
    "[-]": "select-less",
}


# The three below and ``_delete`` above are thin wrappers on
# ``clay.selection``, kept at their old names and signatures because the panes
# and the tests call them by those names. The behaviour and the reasoning are
# in that module.


def _select_all(doc: Any) -> None:
    """``clay.selection.select_all``: everything visible, in the current mode."""
    from .clay import selection

    selection.select_all(doc)


def _invert(doc: Any) -> None:
    """``clay.selection.invert``: Ctrl+Shift+I, in the current mode."""
    from .clay import selection

    selection.invert(doc)


def _duplicate_selection(ctx: Any, state: ClayState, doc: Any) -> None:
    """``clay.selection.duplicate_selected``.

    ``ctx`` and ``state`` are taken and dropped: neither was ever read, and the
    callers pass them, so the parameters stay rather than becoming a rename.
    """
    from .clay import selection

    del ctx, state
    selection.duplicate_selected(doc)


# --- crash recovery (UX-05) ---------------------------------------------------
#
# The mechanism is :mod:`studio.journal`'s; these are the four answers that are
# about *models*. See that module for the loop, the debounce, the head gate and
# the completion gate, and for the rule they all serve: a journal entry is never
# a save.


def _journal_slots(ctx: Any) -> list[Any]:
    """Dirty tabs that are not mid-write. ``saving`` for ``write_wblk``'s
    reason: it walks the object list, and an edit landing mid-encode produces
    an archive whose parts disagree about what is in the document."""
    state = getattr(ctx.state, "clay", None)
    if state is None:
        return []
    return [tab for tab in state.docs if tab.dirty and not tab.saving]


def _journal_encode(tab: Any) -> bytes:
    from .clay import serialize

    # The camera goes in for the same reason a save carries it: a recovered
    # model that framed itself somewhere else is a recovered model the user has
    # to find their way back around.
    return serialize.wblk_bytes(tab.doc, view=tab.view)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen one recovered ``.wblk`` as an *untitled, dirty* document.

    Untitled for Inker's reason: the file it was copied from may still be on
    disk with its own contents, and adopting the path would arm Ctrl+S to
    overwrite something the user has not looked at.
    """
    ensure(ctx)
    # Read and parsed on a task, ``inker_mode``'s ``inker-recover`` shape: a
    # recovered model is read on the first frame of the session, which is
    # the frame the Home pane is meant to appear on, and a ``.wblk`` is as
    # large as the model it holds. True means "submitted", the same answer the
    # Inker provider gives; ``on_task_done`` does the adopting.
    ctx.submit(f"clay-recover:{_path_key(path)}", _load_recovery, Path(path), dict(meta))
    return True


def _load_recovery(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """The task-thread half of a crash recovery: bytes to document."""
    from .clay import serialize

    try:
        doc = serialize.read_wblk(_within_ceiling(path).read_bytes())
    except Exception:
        # ``None`` rather than a raise: the landing turns it into the one
        # sentence every provider says (``journal.adopt_failed``), where a
        # raise arrived as an *error* toast that no other mode's copy raised.
        log.exception("could not reopen the recovered model at %s", path)
        return None
    return {
        "doc": doc,
        "title": f"{meta.get('title') or path.stem} (recovered)",
        "autosave": str(path),
    }


JOURNAL = journal.register(
    journal.Provider(
        kind="clay",
        ext=".wblk",
        label="model",
        slots=_journal_slots,
        uid_of=lambda tab: tab.uid,
        title_of=lambda tab: tab.title,
        head_of=lambda tab: tab.doc.history.head,
        encode=_journal_encode,
        adopt=_journal_adopt,
    )
)
