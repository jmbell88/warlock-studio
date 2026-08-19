"""Plotter's controller: opening, saving, exporting, guarding and keys.

Everything here is *about* documents rather than tiles -- the engine under
``plotter/`` has no idea a job or a task thread exists, and this is the layer
that knows about both. The panes draw; this decides.

The rule that shapes the file is the one Clay and the raster editor already
follow: **no file dialog and no encode ever runs on the frame thread.** A native
picker is modal to the OS and blocks until dismissed; a document of any size is
a zip to build, and a ``.tmx`` export is a zip's worth of PNG encoding besides.
Both go through ``ctx.submit``, which is why saving is a *state*
(``PlotterDoc.saving``) rather than a call that returns.

Two consequences, both of which were bugs elsewhere before they were rules here.
**A failed save must clear that state**, or ``busy`` leaves the tab read-only
until it is closed -- which is what :func:`on_task_failed` is for, and why
:func:`_start` unsets the flag when the runner *refuses* a duplicate key. And
**the head a save records is captured before the submit**, at exactly one place:
a head read after an unbounded modal dialog describes whatever the user did
while it was open.

**A ``.tmx`` export is a mapping of paths, not a file.** TMX has no portable way
to embed an image, so a map is ``map.tmx`` plus one ``.tsx`` and one ``.png`` per
tileset. ``tmx.tmx_export`` returns the mapping and this decides where it lands:
beside the file the user picked, which is what makes the relative paths inside
the ``.tmx`` resolve.

Every task key carries the ``plotter-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import dialogs, docmodes, journal, plotter_state, recents

# ``ensure`` and ``active`` live in :mod:`.plotter_state` -- they touch nothing
# but ``ctx.state.plotter`` -- and the file layer lives in :mod:`.plotter_io`.
# Both are re-exported here, as **plain imports rather than wrappers**, because
# every pane, every key binding and every test says ``plotter_mode.save(ctx)``:
# a wrapper would be a second object where the callers reach for one, and
# ``main.App`` names ``"plotter_mode.persist"`` as a literal besides.
from .plotter_io import (  # noqa: F401
    MAP_FILTER,
    TMJ_FILTER,
    TMX_FILTER,
    WMAP_FILTER,
    _decode,
    _resolve_source,
    _start,
    _within_ceiling,
    ask_open,
    export_map,
    open_path,
    save,
    save_as,
    save_to,
)
from .plotter_state import (  # noqa: F401
    PlotterDoc,
    PlotterState,
    active,
    ensure,
)

# The second facade block, same argument as the first: ``plotter_tilesets`` owns
# every way a tileset arrives, and the panes reach all of them through this
# module. ``TILESET_FILTER`` is re-exported as the *same list object*, which a
# wiring test parametrizes over by identity.
from .plotter_tilesets import (  # noqa: F401
    TILESET_FILTER,
    add_tileset_path,
    ask_add_tileset,
    polish_in_inker,
    tileset_from_inker,
    use_as_tileset,
    use_inker_tileset,
)
from .state import set_mode

log = logging.getLogger(__name__)

DEFAULT_MAP = (32, 32, 32, 32)  # width, height, tile width, tile height


def remember_path(ctx: Any, path: Any) -> None:
    """Put ``path`` at the front of the merged recent list.

    Through :mod:`.recents` rather than onto a field of this mode's own state:
    the four document modes kept four independent ``recent`` lists, and Home's
    single Resume list cannot be built from them at all -- four bare path lists
    carry no ordering *between* them. There is one list now, and this is how
    plotter writes to it.
    """
    recents.remember(ctx.settings, "plotter", path)


def forget_path(ctx: Any, path: Any) -> None:
    """Drop a path that turned out not to open -- :mod:`.recents`' own rule,
    named here so a caller does not have to know this mode's kind string."""
    recents.forget(ctx.settings, "plotter", path)


def recent_paths(ctx: Any) -> list[str]:
    """This mode's recent files, newest first. What its own panel draws."""
    return recents.paths(ctx.settings, "plotter")


def persist(ctx: Any) -> None:
    """Nothing to write any more: the recent list moved to :mod:`.recents`,
    which persists itself on every write. Kept as a no-op because it is called
    from a dozen places after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""





# --- opening ------------------------------------------------------------------


def adopt(
    ctx: Any,
    doc: Any,
    *,
    path: Path | None = None,
    title: str | None = None,
    file_format: str | None = None,
) -> PlotterDoc:
    state = ensure(ctx)
    tab = PlotterDoc(
        doc=doc,
        title=title or plotter_state.title_for(path),
        path=path,
        file_format=file_format or plotter_state.format_for(path),
        saved_head=doc.history.head,
    )
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def ask_new_document(ctx: Any) -> None:
    """The door every "New map" now goes through: ask, then make.

    It does not make a map. Two of the four numbers a map is built from cannot
    be taken back later -- see :mod:`.plotter_setup` -- so the answer is the
    user's rather than a default nobody was offered. Switching modes here is
    what lets Home and the command palette use the one door: the dialog is
    Plotter's, so the caller has to be in Plotter to see it.
    """
    set_mode(ctx.state, "plotter")
    ensure(ctx).setup_pending = True


def new_document(
    ctx: Any,
    size: tuple[int, int, int, int] = DEFAULT_MAP,
    *,
    projection: str | None = None,
) -> PlotterDoc:
    """A blank map. ``projection`` defaults to whatever ``MapDoc`` defaults to.

    Keyword-only and defaulted rather than a fifth member of ``size``, because
    ``size`` is four numbers in two units and a string is neither -- and because
    every existing caller passes the tuple positionally.
    """
    from .plotter.tilemap import MapDoc

    doc = MapDoc(*size) if projection is None else MapDoc(*size, projection=projection)
    doc.add_tile_layer("Ground")
    # A blank document with no layer has nothing to paint into and no row in the
    # layers panel, which reads as broken rather than as empty. The layer is
    # added before the history is cleared so the document still opens clean.
    doc.history.clear()
    doc.mark_saved()
    return adopt(ctx, doc, title="Untitled")


# --- the library --------------------------------------------------------------


def export_library(ctx: Any, tab: PlotterDoc | None = None) -> None:
    """Mint an ordinary asset from the map: a flat render, plus the source.

    The ``clay_mode.export_asset`` shape, and the same payoff -- what comes out
    is a ``done`` reference row, so the library, the inspector, the 2D pipeline
    and every image export work on it without any of them learning that Plotter
    exists. The PNG is written first and ``map.wmap`` second, so a crash between
    them leaves the source absent rather than describing a picture that is not
    there.

    **The encode is on the frame thread and the render is not.** Serialising
    reads the live document, so ``wmap_bytes`` has to run here or a later edit
    lands in the file; compositing the whole map does not, and a 512-square map
    at 32 pixels is a 268-megapixel blend between ``new_frame`` and ``render``.
    So the task reparses the bytes it was handed and renders *those*: the picture
    is a function of the source beside it by construction rather than by two
    reads of one document happening to agree.
    """
    from .plotter import wmap as wmaplib

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not doc.tilesets:
        ctx.toast("There is nothing to render -- add a tileset first.", "error")
        return

    try:
        source = wmaplib.wmap_bytes(doc)
    except wmaplib.WmapUnstorable as exc:
        # The ``.wmap`` writer door, on the frame thread. Version 3 stores the
        # tree, the pictures and the decorations version 2 refused, so what is
        # left behind this door is a layer kind the container has no entry for
        # -- and chunked storage (M5) joins it. An unguarded raise here would
        # take the window down: ``plotter_io._encoded`` already guards the save
        # path for the same reason, and this is the one other frame-thread
        # encode.
        # By name, not by ``ValueError``: anything else out of this encoder is a
        # defect and belongs in a traceback rather than in a toast.
        ctx.toast(f"Nothing was exported. {exc}", "error")
        return

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from ..service import jobs as svc_jobs
        from .plotter.pngio import png_bytes
        from .plotter.render import render_map

        pixels = render_map(wmaplib.read_wmap(source))
        result = svc_jobs.import_reference(
            ctx.svc, png_bytes(pixels), name=title, prompt=title, authored="plotter"
        )
        job_id = result["id"]
        svc_files.save_plotter_source(ctx.svc, job_id, source)
        return {"job_id": job_id, "exported_asset": True}

    _start(ctx, tab, f"plotter-library:{tab.uid}", run)


def edit_asset_in_plotter(ctx: Any, job: Any) -> None:
    """Reopen the ``map.wmap`` beside a library asset."""
    job_id = job["id"] if isinstance(job, dict) else str(job)
    ensure(ctx)

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from ..service.errors import invalid_from
        from .plotter import wmap as wmaplib

        path = svc_files.plotter_source_path(ctx.svc, job_id)
        try:
            doc = wmaplib.read_wmap(Path(path).read_bytes())
        except ValueError as exc:
            raise invalid_from(exc, "This map could not be reopened", field="file") from exc
        return {"doc": doc, "path": "", "title": "Map", "format": "wmap"}

    ctx.submit(f"plotter-open:{job_id}", run)


# --- task results -------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``plotter-`` key."""
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "plotter-open":
        if isinstance(result, dict):
            adopt(
                ctx,
                result["doc"],
                path=Path(result["path"]) if result.get("path") else None,
                title=result.get("title"),
                file_format=result.get("format"),
            )
            set_mode(ctx.state, "plotter")
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return

    if name == "plotter-tileset":
        # Not a save, so ``saving`` was never set and must not be cleared here.
        if isinstance(result, dict) and result.get("tileset") is not None:
            tileset = result["tileset"]
            want = result.get("projection")
            if want and want != tab.doc.projection:
                # One step, not two. A tileset that arrived while the projection
                # did not would leave a map painting the wrong lattice, one
                # Ctrl+Z away from a state nobody asked for.
                #
                # No door has set ``projection`` since the ground generator was
                # deleted on 2026-08-18 -- the new-map dialog owns the lattice
                # now -- so this arm is currently reached only from tests. Kept
                # for ``set_projection``'s own reason: it is the correct shape
                # for a door that carries a lattice with its art, and re-adding
                # it later would mean re-deriving the one-undo-step argument.
                tab.doc.set_projection(want, adding=tileset, source=result.get("source", ""))
            else:
                tab.doc.add_tileset(tileset, source=result.get("source", ""))
            state.tileset_index = len(tab.doc.tilesets) - 1
            state.brush = None
            if tileset.terrains:
                # The thing that just arrived is the thing in your hand, which
                # is exactly what setting ``tileset_index`` says for a palette.
                state.terrain = (state.tileset_index, 0)
            # An isometric map's pixel extent is not width times tile width, so
            # a fit computed against the old projection is the wrong frame.
            tab.view.fitted = False
            ctx.toast("Tileset added.")
        return

    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported_asset"):
        ctx.cache.invalidate()
        ctx.toast("Exported to the library.")
        return
    if result.get("exported"):
        ctx.toast(f"Exported to {result['exported']}")
        return

    tab.mark_saved(result.get("head"))
    # See ``inker_mode``: saved is the moment the crash copy stops
    # describing anything at risk (UX-05).
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = plotter_state.title_for(tab.path)
        tab.file_format = result.get("format") or tab.file_format
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked."""
    state = ctx.state.plotter
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


# --- the guard ----------------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them, the ``clay_mode.guard`` shape. Only quitting
    and closing a tab are destructive: switching modes is not, because Plotter
    is a mode rather than a takeover and its tabs are still there on the way
    back.
    """
    return docmodes.guard(ctx, "plotter", "map", "maps", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    state = ensure(ctx)
    tab = state.get(uid)
    if tab is None:
        return

    def drop() -> None:
        # The document is on disk under a name the user chose, or is gone
        # from the session: either way the crash copy describes work that
        # is no longer at risk, and one left behind is exactly the file
        # that gets offered back after a clean session and confuses
        # somebody (UX-05).
        journal.drop(ctx, tab)
        from .panes import plotter_canvas, plotter_textures

        plotter_textures.release_doc(ctx, uid)
        plotter_canvas.forget_doc(uid)
        state.close(uid)

    if not tab.dirty:
        drop()
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Close without saving?",
            message=f"{tab.title} has unsaved changes.",
            on_confirm=drop,
        )
    )


def release_all(ctx: Any) -> None:
    from .panes import plotter_canvas, plotter_textures

    plotter_textures.release_all(ctx)
    plotter_canvas.forget_all()


# --- keys ---------------------------------------------------------------------

TOOL_KEYS = plotter_state.TOOL_KEYS

# Ctrl bindings that change the document, and are therefore refused while the
# tab is busy. The ``inker_mode._MUTATING_CTRL`` idiom: one list, so a control
# cannot be added to the keyboard and forgotten in the gate.
# Cut is here and copy and paste are not: copy reads, and paste only sets the
# brush and the tool, both of which are view state. Cut is the one that writes.
_MUTATING_CTRL = frozenset({"z", "y", "x"})


def _flipped_h(brush: Any, _shift: bool) -> Any:
    from .plotter import tools as plotter_tools

    return plotter_tools.flip_brush_h(brush)


def _flipped_v(brush: Any, _shift: bool) -> Any:
    from .plotter import tools as plotter_tools

    return plotter_tools.flip_brush_v(brush)


def _turned(brush: Any, shift: bool) -> Any:
    """Z is a quarter turn clockwise; Shift+Z is the same turn three times.

    Three clockwise turns rather than a counter-clockwise routine of its own,
    because a second implementation of the flag algebra is a second thing to
    keep in agreement with ``render.orient`` -- and four turns being the
    identity is already pinned.
    """
    from .plotter import tools as plotter_tools

    for _ in range(3 if shift else 1):
        brush = plotter_tools.rotate_brush_cw(brush)
    return brush


# Tiled's brush transforms, as a table rather than three branches: the letters
# taken are then one thing to read rather than three to keep in agreement.
_BRUSH_TRANSFORMS: dict[str, Any] = {"x": _flipped_h, "y": _flipped_v, "z": _turned}


def handle_key(ctx: Any, event: Any) -> bool:
    """Plotter's keyboard. Returns whether the key was consumed.

    The app calls this unconditionally while the mode is Plotter and returns
    afterwards *whether or not* it consumed the key -- the rule Inker, Clay and
    Review already follow, because letting a key fall through would let F/W/S
    act on a viewport this mode has replaced.
    """
    import pygame

    state = ensure(ctx)
    # Read before the Space branch, because that branch now has to know whether
    # a modifier is down -- see below.
    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    if event.key == pygame.K_SPACE:
        # Seen on both edges, the way ``inker_mode.handle_key`` sees it:
        # space-to-pan is a hold, not a toggle. Filtering KEYUP out above this
        # meant the flag latched on for the rest of the session and every
        # left-drag panned instead of drawing.
        #
        # The *release* is honoured whatever is held, and only the press asks
        # about modifiers. Guarding both would reintroduce the latch by another
        # route: hold Space, then press Ctrl, then let Space go, and the KEYUP
        # arrives with Ctrl set -- ignored, and the flag never clears.
        if event.type != pygame.KEYDOWN:
            state.space_held = False
            return True
        if not (ctrl or shift):
            state.space_held = True
            return True
        # Ctrl+Space and Shift+Space are chords, not a pan. They used to arm
        # pan on their way past, so the next left-drag panned the canvas
        # instead of drawing on it.
    if event.type != pygame.KEYDOWN:
        # Not consumed, and deliberately unlike ``inker_mode.handle_key``,
        # which returns True here. The Space comment above says the two mirror
        # each other and it means the *hold*, which they do; the return is
        # where they part, and ``test_a_key_release_is_never_consumed`` pins
        # this half. Nothing downstream acts on a bare KEYUP, so there is
        # nothing for a release to fall through *into* -- the risk the mode's
        # own docstring names is a KEYDOWN reaching the viewport, and that is
        # answered by the branches below, not by this one.
        return False
    tab = state.active
    name = pygame.key.name(event.key).lower()

    if ctrl:
        if tab is not None and tab.busy and name in _MUTATING_CTRL:
            return True
        return _ctrl_key(ctx, state, tab, name, shift=shift)

    if event.key == pygame.K_ESCAPE:
        # In-mode only: Esc never leaves the mode, which is what every work mode
        # does. **Staged**, the raster editor's idiom, rather than clearing
        # everything at once: one press undoes one thing, outermost first, so a
        # user who pressed it to abandon a drag does not also lose the marquee
        # they spent a gesture placing. Consumed either way -- Esc belongs to
        # this mode even when there is nothing left for it to drop.
        if state.drag_kind:
            state.clear_drag()
        elif state.selected_object is not None:
            state.selected_object = None
        else:
            state.select = None
        return True
    if event.key == pygame.K_DELETE:
        # Guarded inline rather than through ``_MUTATING_CTRL``, which only
        # covers the ctrl table; this writes, so it is refused while saving.
        if tab is not None and not tab.busy:
            _delete(ctx, state, tab)
        return True
    if name in _BRUSH_TRANSFORMS and state.brush is not None:
        # Tiled's X / Y / Z. Deliberately outside the busy gate: the brush is
        # view state, so transforming it writes nothing to the document and
        # pushes no step -- the same reason ``tool`` is unguarded.
        state.brush = _BRUSH_TRANSFORMS[name](state.brush, shift)
        return True
    if name in TOOL_KEYS:
        state.tool = TOOL_KEYS[name]
        return True
    return False


def _selected_tiles(ctx: Any, state: PlotterState, tab: PlotterDoc, *, writing: bool):
    """The active tile layer and the clamped selection, or ``(None, None)``.

    Every refusal is named rather than silent, because "nothing happened" is
    indistinguishable from a broken key.

    ``writing`` is what separates copy from cut: a lock blocks *content* edits,
    and reading a locked layer changes nothing. Refusing a copy would make the
    lock a reason you cannot get at your own tiles, which is not what anybody
    locks a layer for.
    """
    from .plotter.tilemap import TileLayer

    rect = state.selection_in(tab.doc)
    if rect is None:
        ctx.toast("Select some cells first.", "error")
        return None, None
    layer = tab.doc.active()
    if not isinstance(layer, TileLayer):
        ctx.toast("Pick a tile layer first.", "error")
        return None, None
    if writing and layer.locked:
        ctx.toast(f"{layer.name} is locked.", "error")
        return None, None
    return layer, rect


def _copy(ctx: Any, state: PlotterState, tab: PlotterDoc, *, cut: bool) -> None:
    """Take the selected cells, optionally clearing them in one step."""
    import numpy as np

    from .tilegrid import gid as gidlib

    layer, rect = _selected_tiles(ctx, state, tab, writing=cut)
    if layer is None:
        return
    x0, y0, x1, y1 = rect
    # An owned copy, not a view: the layer goes on being painted, and a
    # clipboard that tracked it would paste whatever the map looks like now.
    state.clipboard = np.array(layer.data[y0 : y1 + 1, x0 : x1 + 1], dtype=gidlib.DTYPE)
    state.clipboard_doc = tab.uid
    if cut:
        # One write for the whole rectangle, so a cut is one undo step.
        tab.doc.write_region(
            layer.uid, x0, y0, np.zeros_like(state.clipboard, dtype=gidlib.DTYPE)
        )


def _paste(ctx: Any, state: PlotterState, tab: PlotterDoc) -> None:
    """Paste by loading the clipboard into the brush -- Tiled's model.

    There is no paste *edit*: the block becomes the brush and the tool becomes
    Stamp, so the user places it where they want it and every rule the stamp
    already has -- clipping at the edges, one undo step per stroke, the
    selection constraining it -- applies with nothing new to keep honest.

    A consequence worth stating: our stamp replaces wholesale, so pasted empty
    cells erase what they land on rather than letting it show through.
    """
    if state.clipboard is None:
        ctx.toast("Nothing has been copied.", "error")
        return
    if state.clipboard_doc != tab.uid:
        # Named rather than attempted. Gids are numbered against a map's own
        # firstgids, so the same number is a different tile here -- the paste
        # would land silently wrong rather than fail.
        source = state.get(state.clipboard_doc)
        where = f" from {source.title}" if source is not None else ""
        ctx.toast(f"That copy{where} belongs to another map.", "error")
        return
    state.brush = state.clipboard.copy()
    state.tool = "stamp"


def _delete(ctx: Any, state: PlotterState, tab: PlotterDoc) -> None:
    """Delete clears the selection's cells, or removes the selected object.

    Which one is decided by the tool in hand rather than by what happens to be
    set: with Objects held, Delete is about the object, and a marquee left over
    from earlier must not quietly erase tiles instead.
    """
    import numpy as np

    from .tilegrid import gid as gidlib

    if state.tool == "object" and state.selected_object is not None:
        layer = tab.doc.active()
        if layer is None:
            return
        if getattr(layer, "locked", False):
            ctx.toast(f"{layer.name} is locked.", "error")
            return
        tab.doc.remove_object(layer.uid, state.selected_object)
        state.selected_object = None
        return
    layer, rect = _selected_tiles(ctx, state, tab, writing=True)
    if layer is None:
        return
    x0, y0, x1, y1 = rect
    # One write, so a delete is one undo step whatever it covers.
    tab.doc.write_region(
        layer.uid, x0, y0, np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=gidlib.DTYPE)
    )


def _ctrl_key(
    ctx: Any, state: PlotterState, tab: PlotterDoc | None, name: str, *, shift: bool
) -> bool:
    if name == "n":
        ask_new_document(ctx)
        return True
    if name == "o":
        ask_open(ctx)
        return True
    if tab is None:
        return False
    if name == "w":
        close_tab(ctx, tab.uid)
        return True
    if name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
        return True
    if name == "e":
        export_map(ctx, "tmx", tab) if shift else export_library(ctx, tab)
        return True
    if name == "z":
        # Ctrl+Shift+Z redoes as well, which is what Inker and Clay accept and
        # what a user arriving from either already has in their hand. Ctrl+Y
        # keeps working: this adds a spelling rather than replacing one.
        tab.doc.redo() if shift else tab.doc.undo()
        state.selected_object = None
        return True
    if name == "y":
        tab.doc.redo()
        state.selected_object = None
        return True
    if name == "g":
        state.grid = not state.grid
        return True
    if name == "a":
        # Ctrl+Shift+A deselects, which is Inker's spelling; Ctrl+D is Tiled's.
        # Both, because this editor is reached from both, and neither is taken.
        # View state, so outside the busy gate: a marquee writes nothing.
        state.select = None if shift else (0, 0, tab.doc.width - 1, tab.doc.height - 1)
        return True
    if name == "d":
        state.select = None
        return True
    if name in ("c", "x"):
        _copy(ctx, state, tab, cut=name == "x")
        return True
    if name == "v":
        _paste(ctx, state, tab)
        return True
    if name == "tab":
        state.cycle(-1 if shift else 1)
        return True
    if name == "0":
        tab.view.fitted = False
        return True
    if name == "1":
        # Deferred to the canvas, which is the only thing that knows how big
        # the pane is -- the same reason ``PaintView.pending_zoom`` exists.
        tab.view.pending_zoom = 1.0
        return True
    return False


# --- crash recovery (UX-05) ---------------------------------------------------
#
# ``clay_mode``'s four answers, for maps. See :mod:`studio.journal`.


def _journal_slots(ctx: Any) -> list[Any]:
    state = getattr(ctx.state, "plotter", None)
    if state is None:
        return []
    return [tab for tab in state.docs if tab.dirty and not tab.busy]


def _journal_encode(tab: Any) -> bytes:
    from .plotter import wmap as wmaplib

    return wmaplib.wmap_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    from .plotter import wmap as wmaplib

    ensure(ctx)
    try:
        doc = wmaplib.read_wmap(Path(path).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered map at %s", path)
        ctx.toast("A recovered map could not be reopened.", "warn", action="log")
        return False
    title = f"{meta.get('title') or Path(path).stem} (recovered)"
    tab = adopt(ctx, doc, path=None, title=title, file_format="wmap")
    tab.saved_head = -1
    tab.journal_name = Path(path).name
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="plotter",
        ext=".wmap",
        label="map",
        slots=_journal_slots,
        uid_of=lambda tab: tab.uid,
        title_of=lambda tab: tab.title,
        head_of=lambda tab: tab.doc.history.head,
        encode=_journal_encode,
        adopt=_journal_adopt,
    )
)
