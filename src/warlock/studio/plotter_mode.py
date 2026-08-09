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

import numpy as np

from . import dialogs, filetypes, plotter_state, recents
from .plotter_state import PlotterDoc, PlotterState

log = logging.getLogger(__name__)

MAP_FILTER = [
    "Map documents (*.wmap *.tmx *.tmj)",
    "*.wmap",
    "*.tmx",
    "*.tmj",
]
WMAP_FILTER = ["Warlock map (*.wmap)", "*.wmap"]
TMX_FILTER = ["Tiled map (*.tmx)", "*.tmx"]
TMJ_FILTER = ["Tiled map (*.tmj)", "*.tmj"]
# A ``.tsx`` carries its own slicing; anything else is an image sliced at the
# map's tile size. Both halves of the entry are derived, because the label used
# to read "(*.tsx *.png)" over a pattern list that also accepted .jpg, .jpeg,
# .webp and .bmp -- a dialog disclaiming four formats it would have opened.
_TILESET_SUFFIXES = (".tsx", *filetypes.IMAGE_SUFFIXES)
TILESET_FILTER = [
    filetypes.describe("Tilesets and images", _TILESET_SUFFIXES),
    *filetypes.globs(_TILESET_SUFFIXES),
]

DEFAULT_MAP = (32, 32, 32, 32)  # width, height, tile width, tile height


def ensure(ctx: Any) -> PlotterState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Plotter should not pay for it, and
    because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.plotter
    if state is None:
        state = PlotterState()
        ctx.state.plotter = state
    return state


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



def active(ctx: Any) -> PlotterDoc | None:
    state = ctx.state.plotter
    return state.active if state is not None else None


# --- image loading ------------------------------------------------------------


def _decode(path: Path) -> np.ndarray:
    """One image file as RGBA. Task thread only."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _loaders(base: Path):
    """The two callbacks :mod:`~warlock.studio.plotter.tmx` needs.

    Resolving a relative path means touching a filesystem, which the engine
    deliberately cannot do -- so path resolution lives here, where it can be
    anchored to the file being read.
    """
    from .plotter import tsx as tsxlib

    def image_loader(source: str) -> np.ndarray:
        return _decode(base / source)

    def tsx_loader(source: str) -> Any:
        data = (base / source).read_bytes()
        image = tsxlib.tsx_source(data)
        # Relative to the .tsx, not to the map: a tileset folder is the normal
        # Tiled layout and resolving from the map would miss by one directory.
        return tsxlib.read_tsx(data, _decode((base / source).parent / image))

    return {"image_loader": image_loader, "tsx_loader": tsx_loader}


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


def new_document(ctx: Any, size: tuple[int, int, int, int] = DEFAULT_MAP) -> PlotterDoc:
    from .plotter.tilemap import MapDoc

    doc = MapDoc(*size)
    doc.add_tile_layer("Ground")
    # A blank document with no layer has nothing to paint into and no row in the
    # layers panel, which reads as broken rather than as empty. The layer is
    # added before the history is cleared so the document still opens clean.
    doc.history.clear()
    doc.mark_saved()
    return adopt(ctx, doc, title="Untitled")


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from .plotter import tmx as tmxlib
    from .plotter import wmap as wmaplib

    path = Path(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == plotter_state.TMX_SUFFIX:
        doc = tmxlib.read_tmx(data, **_loaders(path.parent))
    elif suffix == plotter_state.TMJ_SUFFIX:
        doc = tmxlib.read_tmj(data, **_loaders(path.parent))
    else:
        doc = wmaplib.read_wmap(data)
    return {
        "doc": doc,
        "path": str(path),
        "title": plotter_state.title_for(path),
        "format": plotter_state.format_for(path),
    }


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open a map", MAP_FILTER)
        return None if path is None else _load(path)

    ctx.submit("plotter-open", run)


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"plotter-open:{abs(hash(str(path)))}", _load, path)


# --- tilesets -----------------------------------------------------------------


def add_tileset_path(
    ctx: Any, path: Path, *, tile_w: int | None = None, tile_h: int | None = None
) -> None:
    """Add a ``.tsx`` or a grid-sliced image to the open map.

    An image is sliced at the *map's* tile size by default, which is right far
    more often than not and is the only default that needs no dialog. A ``.tsx``
    carries its own slicing and ignores both arguments.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    path = Path(path)
    width = int(tile_w or tab.doc.tile_w)
    height = int(tile_h or tab.doc.tile_h)

    def run() -> dict[str, Any]:
        from .plotter import tsx as tsxlib
        from .plotter.tileset import Tileset

        if path.suffix.lower() == ".tsx":
            data = path.read_bytes()
            image = tsxlib.tsx_source(data)
            tileset = tsxlib.read_tsx(data, _decode(path.parent / image))
        else:
            tileset = Tileset(
                name=path.stem, pixels=_decode(path), tile_w=width, tile_h=height
            )
        return {"tileset": tileset, "source": str(path), "uid": tab.uid}

    ctx.submit(f"plotter-tileset:{tab.uid}", run)


def ask_add_tileset(ctx: Any) -> None:
    """The picker and the decode on one task thread.

    One task rather than a pick task and an add task: the pane would otherwise
    have to route a bare path back through the frame thread only to submit a
    second job with it, and the intermediate result has nowhere sensible to
    live while it waits.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    width, height = int(tab.doc.tile_w), int(tab.doc.tile_h)
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        from .plotter import tsx as tsxlib
        from .plotter.tileset import Tileset

        path = dialogs.open_file("Add a tileset", TILESET_FILTER)
        if path is None:
            return None
        if path.suffix.lower() == ".tsx":
            data = path.read_bytes()
            image = tsxlib.tsx_source(data)
            tileset = tsxlib.read_tsx(data, _decode(path.parent / image))
        else:
            tileset = Tileset(
                name=path.stem, pixels=_decode(path), tile_w=width, tile_h=height
            )
        return {"tileset": tileset, "source": str(path), "uid": uid}

    ctx.submit(f"plotter-tileset:{uid}", run)


def use_as_tileset(ctx: Any, job: Any) -> None:
    """A library asset's reference image, sliced as a tileset.

    The bytes are read on the task thread: an ``input.png`` is routinely several
    megabytes and decoding one between ``new_frame`` and ``render`` is the sort
    of stall the whole task layer exists to avoid.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") or job_id) if isinstance(job, dict) else job_id
    width, height = int(tab.doc.tile_w), int(tab.doc.tile_h)

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from .plotter.tileset import Tileset

        path = svc_files.job_dir_file(ctx.svc, job_id, "input.png")
        tileset = Tileset(
            name=str(name), pixels=_decode(Path(path)), tile_w=width, tile_h=height
        )
        return {"tileset": tileset, "source": "", "uid": tab.uid}

    ctx.submit(f"plotter-tileset:{tab.uid}", run)


# --- saving -------------------------------------------------------------------


def _start(ctx: Any, tab: PlotterDoc, key: str, run: Any) -> None:
    tab.saving = True
    if not ctx.submit(key, run):
        # The runner refuses a key already in flight. Leaving the flag set is
        # what makes a tab read-only forever after a double press.
        tab.saving = False


def _encode(doc: Any, file_format: str) -> dict[str, bytes]:
    """The document as the file (or files) that format needs.

    Built on the frame thread by every caller, which is the opposite of what it
    looks like it should be: serialising *reads the live document*, and doing
    that after an unbounded modal dialog would encode whatever the user did
    while it was open.
    """
    from .plotter import tmx as tmxlib
    from .plotter import wmap as wmaplib

    if file_format == "tmx":
        return tmxlib.tmx_export(doc)
    if file_format == "tmj":
        return tmxlib.tmj_export(doc)
    return {"map.wmap": wmaplib.wmap_bytes(doc)}


def _write(files: dict[str, bytes], path: Path) -> None:
    """Write an encoded map beside ``path``.

    The main document takes the user's chosen name; everything else keeps the
    relative path the exporter chose, because those are exactly the paths
    written *inside* the ``.tmx``. Each file goes through a temporary and a
    replace, so a crash mid-write cannot leave a half-written map where a whole
    one was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        target = path if Path(name).name.startswith("map.") else path.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(target)


def save_to(ctx: Any, tab: PlotterDoc, path: Path, file_format: str) -> None:
    path = Path(path)
    head = tab.doc.history.head
    files = _encode(tab.doc, file_format)

    def run() -> dict[str, Any]:
        _write(files, path)
        return {"head": head, "path": str(path), "format": file_format, "retitle": True}

    _start(ctx, tab, f"plotter-save:{tab.uid}", run)


def save(ctx: Any, tab: PlotterDoc | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    save_to(ctx, tab, tab.path, tab.file_format)


def save_as(ctx: Any, tab: PlotterDoc | None = None, *, file_format: str | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    fmt = file_format or tab.file_format
    suffix = {"tmx": plotter_state.TMX_SUFFIX, "tmj": plotter_state.TMJ_SUFFIX}.get(
        fmt, plotter_state.WMAP_SUFFIX
    )
    filters = {"tmx": TMX_FILTER, "tmj": TMJ_FILTER}.get(fmt, WMAP_FILTER)
    title, head = tab.title, tab.doc.history.head
    files = _encode(tab.doc, fmt)
    stem = Path(title).stem or "map"

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Save the map", f"{stem}{suffix}", filters)
        if path is None:
            return None
        path = path.with_suffix(suffix)
        _write(files, path)
        return {"head": head, "path": str(path), "format": fmt, "retitle": True}

    _start(ctx, tab, f"plotter-saveas:{tab.uid}", run)


def export_map(ctx: Any, file_format: str, tab: PlotterDoc | None = None) -> None:
    """Write a Tiled copy without changing what the tab is.

    Deliberately separate from ``save_as``: exporting to TMX to open a map in
    Tiled should not silently retarget Ctrl+S, because the ``.wmap`` holds
    things the ``.tmx`` cannot (an embedded tileset image, for one).
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if not tab.doc.tilesets:
        ctx.toast("A Tiled map needs at least one tileset.", "error")
        return
    suffix = plotter_state.TMX_SUFFIX if file_format == "tmx" else plotter_state.TMJ_SUFFIX
    filters = TMX_FILTER if file_format == "tmx" else TMJ_FILTER
    stem = Path(tab.title).stem or "map"
    files = _encode(tab.doc, file_format)

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Export for Tiled", f"{stem}{suffix}", filters)
        if path is None:
            return None
        path = path.with_suffix(suffix)
        _write(files, path)
        return {"exported": str(path)}

    _start(ctx, tab, f"plotter-export:{tab.uid}", run)


# --- the library --------------------------------------------------------------


def export_library(ctx: Any, tab: PlotterDoc | None = None) -> None:
    """Mint an ordinary asset from the map: a flat render, plus the source.

    The ``clay_mode.export_asset`` shape, and the same payoff -- what comes out
    is a ``done`` reference row, so the library, the inspector, the 2D pipeline
    and every image export work on it without any of them learning that Plotter
    exists. The PNG is written first and ``map.wmap`` second, so a crash between
    them leaves the source absent rather than describing a picture that is not
    there.
    """
    from .plotter import wmap as wmaplib

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, title = tab.doc, tab.title
    if not doc.tilesets:
        ctx.toast("There is nothing to render -- add a tileset first.", "error")
        return

    from .plotter.render import render_map

    pixels = render_map(doc)
    source = wmaplib.wmap_bytes(doc)

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from ..service import jobs as svc_jobs
        from .packwright.compose import png_bytes

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
        from .plotter import wmap as wmaplib

        path = svc_files.plotter_source_path(ctx.svc, job_id)
        doc = wmaplib.read_wmap(Path(path).read_bytes())
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
            ctx.state.mode = "plotter"
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return

    if name == "plotter-tileset":
        # Not a save, so ``saving`` was never set and must not be cleared here.
        if isinstance(result, dict) and result.get("tileset") is not None:
            tab.doc.add_tileset(result["tileset"], source=result.get("source", ""))
            state.tileset_index = len(tab.doc.tilesets) - 1
            state.brush = None
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
    state = ctx.state.plotter
    if state is None or not state.any_dirty:
        proceed()
        return True
    count = sum(1 for doc in state.docs if doc.dirty)
    what = "one map has" if count == 1 else f"{count} maps have"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved work?",
            message=f"{what[0].upper()}{what[1:]} unsaved changes, which will be lost"
            f" if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


def close_tab(ctx: Any, uid: str) -> None:
    state = ensure(ctx)
    tab = state.get(uid)
    if tab is None:
        return

    def drop() -> None:
        from .panes import plotter_textures

        plotter_textures.release_doc(ctx, uid)
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
    from .panes import plotter_textures

    plotter_textures.release_all(ctx)


# --- keys ---------------------------------------------------------------------

TOOL_KEYS = plotter_state.TOOL_KEYS

# Ctrl bindings that change the document, and are therefore refused while the
# tab is busy. The ``inker_mode._MUTATING_CTRL`` idiom: one list, so a control
# cannot be added to the keyboard and forgotten in the gate.
_MUTATING_CTRL = frozenset({"z", "y"})


def handle_key(ctx: Any, event: Any) -> bool:
    """Plotter's keyboard. Returns whether the key was consumed.

    The app calls this unconditionally while the mode is Plotter and returns
    afterwards *whether or not* it consumed the key -- the rule Inker, Clay and
    Review already follow, because letting a key fall through would let F/W/S
    act on a viewport this mode has replaced.
    """
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    tab = state.active
    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key).lower()

    if ctrl:
        if tab is not None and tab.busy and name in _MUTATING_CTRL:
            return True
        return _ctrl_key(ctx, state, tab, name, shift=shift)

    if event.key == pygame.K_ESCAPE:
        # In-mode only: Esc drops a drag or an object selection and never
        # leaves the mode, which is what every work mode does.
        state.clear_drag()
        state.selected_object = None
        return True
    if name in TOOL_KEYS:
        state.tool = TOOL_KEYS[name]
        return True
    if event.key == pygame.K_SPACE:
        state.space_held = True
        return True
    return False


def _ctrl_key(
    ctx: Any, state: PlotterState, tab: PlotterDoc | None, name: str, *, shift: bool
) -> bool:
    if name == "n":
        new_document(ctx)
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
        tab.doc.undo()
        state.selected_object = None
        return True
    if name == "y":
        tab.doc.redo()
        state.selected_object = None
        return True
    if name == "g":
        state.grid = not state.grid
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
