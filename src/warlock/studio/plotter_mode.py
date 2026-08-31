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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import dialogs, docmodes, journal, plotter_state, recents
from .plotter._map_model import ObjectLayer

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
    BLOB_LAYOUT,
    TILESET_FILTER,
    SheetLattice,
    SheetMismatch,
    SheetRecord,
    SheetTerrain,
    add_tileset_path,
    ask_add_tileset,
    ask_replace_tileset,
    choose_layer_image,
    clear_sheet_import,
    import_detected_sheet,
    import_sheet_blind,
    import_sheet_terrain,
    land_layer_image,
    land_tileset,
    land_tileset_image,
    polish_in_inker,
    repaint_tileset,
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
    infinite: bool = False,
) -> PlotterDoc:
    """A blank map. ``projection`` defaults to whatever ``MapDoc`` defaults to.

    Keyword-only and defaulted rather than members of ``size``, because ``size``
    is four numbers in two units and neither a string nor a flag is one -- and
    because every existing caller passes the tuple positionally.
    """
    from .plotter.tilemap import MapDoc

    extra: dict[str, Any] = {"infinite": True} if infinite else {}
    doc = (
        MapDoc(*size, **extra)
        if projection is None
        else MapDoc(*size, projection=projection, **extra)
    )
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
        # tree, the pictures and the decorations version 2 refused and version
        # 10 stores an infinite map, so what is left behind this door is a layer
        # kind the container has no entry for. An unguarded raise here would
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
            doc = wmaplib.read_wmap(_within_ceiling(Path(path)).read_bytes())
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

    if name == "plotter-layer-image":
        if isinstance(result, dict) and result.get("image") is not None:
            land_layer_image(ctx, tab, result)
        return

    if name == "plotter-tileset-image":
        # A *reload*, not an arrival: it replaces art under an existing tileset
        # rather than appending one, which is why it does not share the
        # ``plotter-tileset`` key. Not a save either, so ``saving`` is untouched.
        if isinstance(result, dict) and result.get("replace") is not None:
            land_tileset_image(ctx, tab, result)
        return

    if name == "plotter-tileset":
        # Not a save, so ``saving`` was never set and must not be cleared here.
        if isinstance(result, dict) and result.get("sheet") is not None:
            # A ruled sheet: park it and let the popup ask. Nothing is added to
            # the document here -- ``slicing``'s rule is that the detection is a
            # suggestion, so the only thing that has happened is a measurement.
            state.sheet_import = result["sheet"]
            state.sheet_import_open = False
            return
        if isinstance(result, dict) and result.get("tileset") is not None:
            land_tileset(ctx, state, tab, result)
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


#: ``ctx.state.preview`` keys the Plotter panes mint per tab, per layer and
#: per object. Tab uids are never reused, so an entry left behind is held for
#: the life of the process -- and the minimap's is a whole-map RGBA composite,
#: which is ``plotter_textures.PREFIX``'s own reason for existing. Listed here
#: rather than found by prefix because the layer and object keys carry *their*
#: uids, not the tab's.
_TAB_PREVIEW_PREFIXES = (
    "plotter_minimap:",
    "plotter_resize:",
    "plotter_offset:",
    "plotter_tile_px:",
    "tilemeta:",
    "tsedit:",
)


def _forget_preview(ctx: Any, tab: Any) -> None:
    preview = getattr(ctx.state, "preview", None)
    if not isinstance(preview, dict):
        return
    layer_uids = {layer.uid for layer in tab.doc.all_layers()}
    object_uids = {
        obj.uid for layer in tab.doc.all_layers() for obj in getattr(layer, "objects", ())
    }
    for key in list(preview):
        head, _, rest = key.partition(":")
        if (
            any(key.startswith(f"{p}{tab.uid}") for p in _TAB_PREVIEW_PREFIXES)
            or (head == "plotter_layer_prop" and rest.isdigit() and int(rest) in layer_uids)
            or (head == "plotter_prop" and rest.isdigit() and int(rest) in object_uids)
        ):
            preview.pop(key, None)


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
        _forget_preview(ctx, tab)
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
# Copy is the one that stays out: it only reads. Cut removes, Ctrl+J adds a
# duplicate, and paste is here since ``ObjectClip`` -- pasting an *object*
# writes the document, where a tile paste only sets the brush and the tool.
# Gating the tile case alongside costs a busy tab nothing it could have used.
_MUTATING_CTRL = frozenset({"z", "y", "x", "v", "j"})


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


# --- history ------------------------------------------------------------------
#
# One call per direction, rather than two lines under the key handler, because
# the bridge panel draws the same Undo/Redo pair Inker's does. Clay, Plotter and
# Packwright each had a full undo stack and no on-screen control at all, so the
# feature existed only for a user who already knew the chord -- and every
# side effect a step has (the object selection, which may name a shape the
# step removed) belongs to *undoing*, not to the keyboard.


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it."""
    tab.doc.undo()
    ensure(ctx).select_object(None)



def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()
    ensure(ctx).select_object(None)


def go_to_cell(ctx: Any, tab: Any, column: int, row: int) -> tuple[int, int]:
    """Ask the canvas to centre on one cell. -> the cell it will actually use.

    **Clamped rather than refused.** A coordinate past the edge is a typo or a
    number remembered from a bigger map, and the useful answer to both is the
    nearest cell that exists -- a refusal would put a dialog in the way of a
    navigation. The clamped pair is returned so the dialog can show where it
    landed instead of silently disagreeing with what was typed.

    In the *stored grid's* coordinates, which is what everything else in this
    mode means by a cell: the status bar's ``cell x, y`` readout, the marquee
    and ``last_paint`` all count from the window's own origin, and an infinite
    map slides that window under them together through
    :meth:`PlotterState.shift_cells`. A second coordinate space here would make
    the readout and the dialog disagree on exactly the maps that need them most.

    A flag rather than a pan, ``palette_zoom_rung``'s pattern: the pan is
    measured against a viewport only the canvas has.
    """
    doc = tab.doc
    cell = (
        max(0, min(int(column), int(doc.width) - 1)),
        max(0, min(int(row), int(doc.height) - 1)),
    )
    ensure(ctx).goto_cell = cell
    return cell


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump the document to a position in its undo stack.

    The history panel's door, and the *third* surface onto the same stack --
    which is why it is here beside the other two rather than in the pane. An
    object selection names a uid that a jump can undo out of existence, so it is
    dropped exactly as :func:`undo` drops it; a panel calling
    ``doc.step_history`` directly would leave the Properties pane pointing at an
    object no layer holds.
    """
    moved = tab.doc.step_history(index)
    ensure(ctx).select_object(None)
    return moved



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
    # a modifier is down -- see below. Off ``event.mod``, never
    # ``pygame.key.get_mods()`` -- ``main._shortcut``'s rule (UX-12): ``mod``
    # is the modifier state at the instant this key was pressed, where
    # ``get_mods()`` is the state *now*, after the event batch drained, so a
    # Ctrl released between the two made a fast chord fall through as the
    # bare letter.
    mods = event.mod
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
        elif state.selected_objects:
            state.select_object(None)
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
    # **The letters belong to the layer** (W3.2): six of them mean two things,
    # which is Tiled's arrangement -- ``R`` is the rectangular select on a tile
    # layer and insert-rectangle on an object one. ``sync_tool`` first, and
    # idempotently, so a key pressed on the frame after a layer switch cannot
    # act with a tool the layer does not host.
    zoom_in, zoom_out = _zoom_keys()
    if tab is not None and event.key in zoom_in:
        # Whole rungs, and ``=`` unshifted answers too: it is the same physical
        # key on every layout this ships to. Inker's own pair, spelled the same.
        tab.view.pending_zoom = _stepped(tab.view.zoom, 1)
        return True
    if tab is not None and event.key in zoom_out:
        tab.view.pending_zoom = _stepped(tab.view.zoom, -1)
        return True
    if tab is not None and name == "h":
        # Tiled's Highlight Current Layer. A canvas setting and nothing more --
        # ``render.py`` composites exports off the same resolver, so a dim
        # living in ``scene.resolve`` would export dimmed.
        state.highlight = not state.highlight
        return True
    if tab is not None and name.isdigit() and name != "0":
        # Bare 1-9 recall. Tiled binds these to *tools*; this editor's tools
        # already have letters, and a numbered stamp is the thing a map-painting
        # hand reaches for by number. The divergence is recorded in
        # ``docs/COMPAT.md`` beside the Ctrl+D one.
        return recall_stamp(ctx, state, tab, int(name))
    layer = None if tab is None else tab.doc.active()
    plotter_state.sync_tool(state, layer)
    keys = plotter_state.tool_keys(layer)
    if name in keys:
        state.tool = keys[name]
        shape = plotter_state.OBJECT_SHAPES.get(state.tool)
        if shape:
            state.object_shape = shape
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

    if state.tool == "object" and state.selected_object is not None:
        # With Objects in hand, Ctrl+C is about the object -- a marquee left
        # over from earlier must not quietly copy tiles instead, which is the
        # rule ``_delete`` already follows.
        _copy_object(ctx, state, tab, cut=cut)
        return
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
    if isinstance(state.clipboard, ObjectClip):
        _paste_object(ctx, state, tab)
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


@dataclass(frozen=True)
class ObjectClip:
    """A copied object, tagged so the one clipboard can hold two kinds.

    Kind-tagged rather than a second field beside ``clipboard``, because "what
    does Ctrl+V do" then has one answer read off one value -- two clipboards
    would need a rule for which wins, and that rule is exactly what a user
    cannot see.
    """

    obj: Any


def _copy_object(
    ctx: Any, state: PlotterState, tab: PlotterDoc, *, cut: bool
) -> None:
    """Ctrl+C / Ctrl+X with Objects in hand.

    The object is stored by reference in an :class:`ObjectClip`; the paste makes
    the copy. Storing the live object is safe because every edit to one replaces
    it wholesale rather than writing through -- and a cut has already taken it
    out of the layer, so there is nothing left to write through to.
    """
    layer = tab.doc.active()
    if not isinstance(layer, ObjectLayer):
        return
    # Copying from a locked layer is allowed while cutting is not (the lock
    # blocks content edits and nothing else) -- and a refused cut copies
    # nothing, because a cut that quietly became a copy would leave the user
    # pasting what they believe they moved.
    if cut and getattr(layer, "locked", False):
        ctx.toast(f"{layer.name} is locked.", "error")
        return
    found = next((o for o in layer.objects if o.uid == state.selected_object), None)
    if found is None:
        return
    state.clipboard = ObjectClip(obj=found)
    state.clipboard_doc = tab.uid
    if cut:
        tab.doc.remove_object(layer.uid, found.uid)
        state.select_object(None)


def _paste_object(ctx: Any, state: PlotterState, tab: PlotterDoc) -> None:
    """Paste a copied object onto the active object layer.

    **Cross-map paste is refused whole**, by the same rule the tile clipboard's
    is and for two reasons rather than one: a tile object carries a gid, which
    is numbered against a map's own firstgids, and an object property may name
    an object id, which means something else in another map. Neither can be
    fixed up without guessing.
    """
    import dataclasses

    from .plotter.tilemap import new_uid

    if state.clipboard_doc != tab.uid:
        source = state.get(state.clipboard_doc)
        where = f" from {source.title}" if source is not None else ""
        ctx.toast(f"That object{where} belongs to another map.", "error")
        return
    layer = tab.doc.active()
    if not isinstance(layer, ObjectLayer):
        ctx.toast("Pick an object layer to paste onto.", "error")
        return
    # A paste adds an object, which is a content edit -- the same door
    # ``_delete`` holds, because the lock is enforced at the studio layer.
    if getattr(layer, "locked", False):
        ctx.toast(f"{layer.name} is locked.", "error")
        return
    copy = dataclasses.replace(
        state.clipboard.obj,
        uid=new_uid(),
        id=0,
        x=state.clipboard.obj.x + tab.doc.tile_w,
        y=state.clipboard.obj.y + tab.doc.tile_h,
    )
    tab.doc.add_object(layer.uid, copy)
    state.select_object(copy.uid)


def _duplicate_object(ctx: Any, state: PlotterState, tab: PlotterDoc) -> None:
    """Ctrl+J: copy the selected object one cell down-right.

    Down-right by one cell rather than in place, for the reason every editor
    that offsets a duplicate has: a copy exactly on top of its original is
    indistinguishable from nothing having happened, and the next click selects
    whichever one the hit test reaches first.

    The copy takes a **fresh persistent id** from the monotone counter -- ids are
    never reused -- and its object-typed properties keep pointing where the
    original's pointed, which is Tiled's rule and the only one that does not
    silently rewrite somebody's reference.
    """
    import dataclasses

    from .plotter.tilemap import new_uid

    if state.tool != "object" or state.selected_object is None:
        ctx.toast("Select an object first.", "error")
        return
    layer = tab.doc.active()
    if not isinstance(layer, ObjectLayer):
        return
    # A duplicate adds an object, which is a content edit -- the same door
    # ``_delete`` holds, because the lock is enforced at the studio layer.
    if getattr(layer, "locked", False):
        ctx.toast(f"{layer.name} is locked.", "error")
        return
    found = next((o for o in layer.objects if o.uid == state.selected_object), None)
    if found is None:
        return
    copy = dataclasses.replace(
        found,
        uid=new_uid(),
        # ``0`` makes ``add_object`` mint the next one; setting it here would be
        # a second place that knows how ids are assigned.
        id=0,
        x=found.x + tab.doc.tile_w,
        y=found.y + tab.doc.tile_h,
    )
    tab.doc.add_object(layer.uid, copy)
    state.select_object(copy.uid)
    ctx.toast(f"Duplicated {found.name or 'the object'}.")


def _delete(ctx: Any, state: PlotterState, tab: PlotterDoc) -> None:
    """Delete clears the selection's cells, or removes the selected object.

    Which one is decided by the tool in hand rather than by what happens to be
    set: with Objects held, Delete is about the object, and a marquee left over
    from earlier must not quietly erase tiles instead.
    """
    import numpy as np

    from .tilegrid import gid as gidlib

    if state.tool == "object" and state.selected_objects:
        layer = tab.doc.active()
        if layer is None:
            return
        if getattr(layer, "locked", False):
            ctx.toast(f"{layer.name} is locked.", "error")
            return
        # Every selected object in one step. ``remove_objects`` is the group
        # twin of ``remove_object`` -- one ``compound`` of the same
        # ``ObjectRemoveEdit`` -- so a Delete over five objects is one Ctrl+Z
        # rather than five.
        tab.doc.remove_objects(layer.uid, state.selected_objects)
        state.select_object(None)
        return
    layer, rect = _selected_tiles(ctx, state, tab, writing=True)
    if layer is None:
        return
    x0, y0, x1, y1 = rect
    # One write, so a delete is one undo step whatever it covers.
    tab.doc.write_region(
        layer.uid, x0, y0, np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=gidlib.DTYPE)
    )


#: The zoom ladder the keyboard steps through, as scales. Whole numbers above
#: 1:1 and halving below it -- the pixel-art rule, and the same ladder Inker's
#: keyboard uses, so a hand that learned one editor's + and - has learned both.
ZOOM_RUNGS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _zoom_keys():
    """The two key sets, built lazily: pygame is imported inside the handler."""
    import pygame

    return (
        {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS},
        {pygame.K_MINUS, pygame.K_KP_MINUS},
    )


def _stepped(zoom: float, direction: int) -> float:
    """The next rung up or down from wherever the wheel has left the zoom."""
    rungs = list(ZOOM_RUNGS)
    if direction > 0:
        return next((rung for rung in rungs if rung > zoom + 1e-6), rungs[-1])
    return next((rung for rung in reversed(rungs) if rung < zoom - 1e-6), rungs[0])


def _shift_layer(tab: PlotterDoc, delta: int) -> bool:
    """Move the active layer one place within its own parent. -> whether it moved."""
    doc = tab.doc
    uid = doc.active_layer
    if uid is None:
        return False
    found = doc._locate(uid)
    if found is None:
        return False
    _layer, parent_uid, index = found
    siblings = doc.children_of(parent_uid)
    wanted = index + int(delta)
    if not 0 <= wanted < len(siblings):
        return False
    doc.move_layer(uid, wanted)
    return True


def store_stamp(ctx: Any, state: PlotterState, tab: PlotterDoc, slot: int) -> bool:
    """Ctrl+Shift+N: put the brush in hand into a numbered slot.

    Nine slots, on the number row, because that is where a hand already
    reaches -- and **bare digits recall while the chord stores**, which is the
    way round that matters: recall happens hundreds of times a session and
    storing happens nine times, so the cheap gesture goes to the frequent one.
    """
    if state.brush is None:
        ctx.toast("There is no stamp in hand to store.", "warn")
        return False
    tab.stamps[int(slot)] = state.brush.copy()
    ctx.toast(f"Stored stamp {slot}.")
    return True


def recall_stamp(ctx: Any, state: PlotterState, tab: PlotterDoc, slot: int) -> bool:
    """A bare digit: take a numbered stamp back into the hand.

    Switches to Stamp, which is the paste precedent: a brush loaded with no way
    to put it down is a gesture that stops halfway.
    """
    block = tab.stamps.get(int(slot))
    if block is None:
        ctx.toast(f"Stamp {slot} is empty -- Ctrl+Shift+{slot} stores one.", "warn")
        return False
    state.brush = block.copy()
    state.tool = "stamp"
    return True


def _invert_selection(state: PlotterState, tab: PlotterDoc) -> None:
    """Select every cell the selection does not hold.

    Through ``set_selection``, the one door, so the rect and the mask cannot
    disagree about which cells are in it -- and with a mask always, because the
    inverse of a rectangle is not a rectangle.
    """
    from .plotter import tools as tools_lib

    doc = tab.doc
    mask, rect = tools_lib.invert_selection(
        state.select_mask, state.select, doc.width, doc.height
    )
    state.set_selection(rect, mask)


def grow_selection(state: PlotterState, tab: PlotterDoc, steps: int = 1) -> None:
    """Tiled's Expand Selection, and its Contract on a negative step."""
    import numpy as np

    from .plotter import tools as tools_lib

    doc = tab.doc
    if state.select is None:
        return
    mask = state.select_mask
    if mask is None:
        mask = np.zeros((doc.height, doc.width), dtype=bool)
        x0, y0, x1, y1 = state.select
        mask[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] = True
    grown = (
        tools_lib.grow_selection(mask, steps)
        if steps > 0
        else tools_lib.shrink_selection(mask, -steps)
    )
    state.set_selection(tools_lib._bounds_of(grown, doc.width, doc.height), grown)


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
    if shift and name in ("up", "down"):
        # Tiled's Raise/Lower Layer. Ctrl+Shift rather than Ctrl+arrow: the
        # plain arrows nudge and Ctrl+arrow is free but reads as "a bigger
        # nudge", which is what Inker's Shift+arrow already means.
        _shift_layer(tab, 1 if name == "up" else -1)
        return True
    if shift and name.isdigit() and name != "0":
        return store_stamp(ctx, state, tab, int(name))
    if name == "z":
        # Ctrl+Shift+Z redoes as well, which is what Inker and Clay accept and
        # what a user arriving from either already has in their hand. Ctrl+Y
        # keeps working: this adds a spelling rather than replacing one.
        redo(ctx, tab) if shift else undo(ctx, tab)
        return True
    if name == "y":
        redo(ctx, tab)
        return True
    if name == "g":
        if shift:
            # Tiled's Ctrl+Shift+G, Snap to Grid, as a straight toggle between
            # the grid and off -- it does not disturb Pixel, which has its own
            # chord below, so the pair behave like Tiled's two checkable rows
            # rather than like one three-way cycle nobody can aim.
            state.snap = "off" if state.snap == "grid" else "grid"
            return True
        state.grid = not state.grid
        return True
    if name == "p" and shift:
        # Tiled's Ctrl+Shift+P, Snap to Pixels.
        state.snap = "off" if state.snap == "pixel" else "pixel"
        return True
    if name == "r" and not shift:
        state.rulers = not state.rulers
        return True
    if name == "i" and shift:
        # Ctrl+Shift+I, the chord Inker and Clay both bind for the same verb.
        _invert_selection(state, tab)
        return True
    if name == "a":
        # Ctrl+Shift+A deselects, which is Inker's spelling; Ctrl+D is Tiled's.
        # Both, because this editor is reached from both, and neither is taken.
        # View state, so outside the busy gate: a marquee writes nothing.
        state.select = None if shift else (0, 0, tab.doc.width - 1, tab.doc.height - 1)
        return True
    if name == "d":
        # **Ctrl+D stays deselect here**, a kept divergence from Tiled, where it
        # duplicates. Every other editor in this app deselects on it and the
        # duplicate moves to Ctrl+J instead -- which is where the raster editor
        # already puts "copy this to its own layer".
        state.select = None
        return True
    if name == "j":
        _duplicate_object(ctx, state, tab)
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
        doc = wmaplib.read_wmap(_within_ceiling(Path(path)).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered map at %s", path)
        ctx.toast("A recovered map could not be reopened.", "warn", action="log")
        return False
    title = f"{meta.get('title') or Path(path).stem} (recovered)"
    tab = adopt(ctx, doc, path=None, title=title, file_format="wmap")
    # A recovered document must read dirty until the user saves it somewhere:
    # ``read_wmap`` hands back a document already marked saved, and a clean
    # recovered tab closes without a confirm -- taking the journal copy, the
    # only surviving copy of the work, with it. ``PlotterDoc.dirty`` delegates
    # to the document, so the never-matching head goes on the *document*; the
    # tab's mirror follows so ``mark_saved`` keeps them in step.
    doc.saved_head = -1
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
