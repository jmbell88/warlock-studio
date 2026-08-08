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

import logging
from pathlib import Path
from typing import Any

from . import clay_state, dialogs
from .clay_state import ClayState, ClayTab

log = logging.getLogger(__name__)

WBLK_FILTER = ["Warlock Clay document (*.wblk)", "*.wblk"]


def ensure(ctx: Any) -> ClayState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Clay should not pay for it,
    and because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.clay
    if state is None:
        state = ClayState()
        stored = ctx.settings.get("clay") or {}
        state.recent = [p for p in (stored.get("recent") or []) if isinstance(p, str)]
        ctx.state.clay = state
    return state


def persist(ctx: Any) -> None:
    state = ctx.state.clay
    if state is not None:
        ctx.settings.set("clay", {"recent": state.recent})


def active(ctx: Any) -> ClayTab | None:
    state = ctx.state.clay
    return state.active if state is not None else None


# --- opening ----------------------------------------------------------------


def adopt(ctx: Any, doc: Any, *, path: Path | None = None, title: str | None = None) -> ClayTab:
    state = ensure(ctx)
    tab = ClayTab(
        doc=doc,
        title=title or clay_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    state.add(tab)
    state.remember(path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> ClayTab:
    from .clay import document as bd

    return adopt(ctx, bd.ClayDoc(), title="Untitled")


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from .clay import serialize

    doc = serialize.read_wblk(Path(path).read_bytes())
    return {"doc": doc, "path": str(path), "title": clay_state.title_for(Path(path))}


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
    ctx.submit(f"clay-open:{abs(hash(str(path)))}", _load, path)


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
        return _parse_glb(path.read_bytes(), path.stem)

    ctx.submit(f"clay-import:{abs(hash(str(path)))}", run)


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
        return _parse_glb(mesh.read_bytes(), name)

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
    if int(result.get("triangles", 0)) <= SLOW_TRIANGLES:
        adopt(ctx, doc, title=title)
        ctx.state.mode = "clay"
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
            on_confirm=lambda: _adopt_now(ctx, doc, title),
        )
    )


def _adopt_now(ctx: Any, doc: Any, title: str) -> None:
    adopt(ctx, doc, title=title)
    ctx.state.mode = "clay"


# --- saving -----------------------------------------------------------------


def _start(ctx: Any, tab: ClayTab, key: str, run: Any) -> None:
    tab.saving = True
    if not ctx.submit(key, run):
        # The runner refuses a key that is already in flight. Leaving the flag
        # set here is what makes a tab read-only forever after a double press.
        tab.saving = False


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
    data = serialize.wblk_bytes(doc)

    def run() -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return {"rev": rev, "path": str(path), "retitle": True}

    _start(ctx, tab, f"clay-save:{tab.uid}", run)


def save(ctx: Any, tab: ClayTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    save_to(ctx, tab, tab.path)


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
    data = serialize.wblk_bytes(doc)

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save Clay document", f"{title}{clay_state.WBLK_SUFFIX}", WBLK_FILTER
        )
        if path is None:
            return None
        path = path.with_suffix(clay_state.WBLK_SUFFIX)
        path.write_bytes(data)
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
    wblk = serialize.wblk_bytes(doc)

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
            adopt(ctx, result["doc"], path=Path(result["path"]), title=result.get("title"))
            ctx.state.mode = "clay"
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
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = clay_state.title_for(tab.path)
        state.remember(tab.path)
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
    state = ctx.state.clay
    if state is None or not state.any_dirty:
        proceed()
        return True
    count = sum(1 for doc in state.docs if doc.dirty)
    what = "one document has" if count == 1 else f"{count} documents have"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved work?",
            message=f"{what[0].upper()}{what[1:]} unsaved changes, which will be lost"
            f" if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


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

# Ctrl-shortcuts that change the document. Serialising reads the live document
# on a task thread, so anything that restructures it or moves the history head
# the save captured waits for the save, exactly as a gizmo drag does.
_MUTATING_CTRL = frozenset({"z", "y", "d", "a", "i", "j"})


def handle_key(ctx: Any, event: Any) -> bool:
    """Clay's shortcuts. -> whether the key was consumed.

    **False with nothing open**, which the caller's fall-through depends on:
    Clay owns a viewport, and with no document the viewport's own
    shortcuts must still work. With a document open the key is consumed
    unconditionally, because falling through would let it act on a pane Clay
    mode has replaced.
    """
    import pygame

    state = ctx.state.clay
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.type != pygame.KEYDOWN:
        return True

    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key)

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, shift=shift)

    if name in ELEMENT_KEYS and not shift:
        if not tab.saving:
            doc.set_element_mode(ELEMENT_KEYS[name])
    elif not shift and _registry_key(ctx, tab, doc, name):
        pass
    elif name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif event.key == pygame.K_DELETE:
        if not tab.saving:
            _delete(ctx, doc)
    elif event.key == pygame.K_ESCAPE:
        _escape(state, tab, doc)
    return True


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
    """Delete what is selected, in whatever sense the current mode means it.

    In an element mode this **never** falls through to removing objects. A user
    in face mode pressing Delete means "get rid of these faces", and quietly
    deleting the whole object instead is the most destructive possible
    misreading of one keystroke -- the more so because the object selection in
    an element mode is derived from the element selection, so every object with
    anything selected inside it would go.
    """
    from .clay import elements as el
    from .clay import ops_topo

    if doc.element_mode == "object":
        for uid in list(doc.selection):
            doc.remove_object(uid)
        return
    for uid in list(doc.element_sel):
        obj = doc.by_uid(uid)
        faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face")
        if el.is_empty(faces):
            continue
        try:
            mesh, sel = ops_topo.delete_faces(obj.mesh, faces)
        except el.OpError as error:
            _toast(ctx, str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)


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
    """See ``clay_ops.toast``: the attribute this reached for never existed."""
    ctx.toast(message, "error")


def _ctrl_key(
    ctx: Any, state: ClayState, tab: ClayTab, doc: Any, name: str, *, shift: bool
) -> bool:
    if tab.saving and name in _MUTATING_CTRL:
        return True
    if name in AXIS_VIEW_KEYS:
        # Shift is the opposite view, as Blender's numpad does it -- Ctrl+1 is
        # the front and Ctrl+Shift+1 the back, so six views cost three keys.
        wanted = AXIS_VIEW_KEYS[name]
        if shift:
            wanted = {"front": "back", "right": "left", "top": "bottom"}[wanted]
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            view.camera.look_along(wanted)
    elif name == "5":
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            view.camera.orthographic = not view.camera.orthographic
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "o":
        ask_open(ctx)
    elif name == "n":
        new_document(ctx)
    elif name == "e":
        export_asset(ctx, tab)
    elif name == "z":
        doc.redo() if shift else doc.undo()
    elif name == "y":
        doc.redo()
    elif name == "a":
        _select_all(doc)
    elif name == "i" and shift:
        _invert(doc)
    elif name == "j" and doc.element_mode == "object":
        # Object mode only, for the reason Ctrl+D is: a merge is about whole
        # objects, and there is no element-mode reading of it to fall back on.
        from . import clay_ops

        op = clay_ops.get("join")
        if op.enabled(doc):
            _fire_op(ctx, doc, op)
    elif name == "d" and doc.element_mode == "object":
        # Object mode only: duplicating a *face* selection is a different
        # operation with a different name, and doing the object one instead
        # would silently double a mesh the user is mid-edit on.
        _duplicate_selection(ctx, state, doc)
    elif name == "tab":
        state.cycle(-1 if shift else 1)
    return True


def _select_all(doc: Any) -> None:
    """Everything *visible*, in the current mode's sense of everything.

    The object branch used to take every object and the element branch below it
    has always skipped the invisible ones, which is the asymmetry rather than a
    choice: ``visible=False`` is documented to mean an object does not render,
    does not export and **cannot be picked**, and Ctrl+A was picking them. It
    only showed once merge existed -- Ctrl+A then Ctrl+J pulled geometry the
    user could not see into the survivor -- but Delete and Ctrl+D were reading
    the same selection all along.
    """
    from .clay import elements as el

    if doc.element_mode == "object":
        doc.select([obj.uid for obj in doc.objects if obj.visible])
        return
    for obj in doc.objects:
        if obj.visible:
            doc.set_element_sel(obj.uid, el.select_all(obj.mesh, doc.element_mode))


def _invert(doc: Any) -> None:
    """Ctrl+Shift+I. In object mode it inverts which objects are selected."""
    from .clay import elements as el

    if doc.element_mode == "object":
        # Visible only, for ``_select_all``'s reason: inverting into a hidden
        # object selects something the user cannot see or click back off.
        doc.select([o.uid for o in doc.objects if o.visible and o.uid not in doc.selection])
        return
    for obj in doc.objects:
        if not obj.visible:
            continue
        doc.set_element_sel(
            obj.uid,
            el.invert(obj.mesh, doc.element_sel_of(obj.uid), doc.element_mode),
        )


def _duplicate_selection(ctx: Any, state: ClayState, doc: Any) -> None:
    from .clay import document as bd
    from .clay import ops

    taken = [obj.name for obj in doc.objects]
    fresh = []
    for uid in list(doc.selection):
        copy = ops.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
        taken.append(copy.name)
        doc.add_object(copy)
        fresh.append(copy.uid)
    if fresh:
        doc.select(fresh)
