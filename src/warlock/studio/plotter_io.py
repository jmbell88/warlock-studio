"""Plotter's file layer: what a map is read from and written to.

Split out of ``plotter_mode`` because it is the half that touches a disk, and
the rules that shape it are all about that. **No file dialog and no encode ever
runs on the frame thread** -- a native picker is modal to the OS and blocks until
dismissed, and a document of any size is a zip to build -- so both go through
``ctx.submit``, which is why saving is a *state* (``PlotterDoc.saving``) rather
than a call that returns. **The head a save records is captured before the
submit**, at exactly one place: a head read after an unbounded modal dialog
describes whatever the user did while it was open. And **every write is staged**
through a dotfile and an ``os.replace``, cleaned up in a ``finally``.

**A ``.tmx`` export is a mapping of paths, not a file.** TMX has no portable way
to embed an image, so a map is ``map.tmx`` plus one ``.tsx`` and one ``.png`` per
tileset. ``tmx.tmx_export`` returns the mapping and :func:`_write` decides where
it lands: beside the file the user picked, which is what makes the relative paths
inside the ``.tmx`` resolve.

**The engine cannot open a file and this module is why it does not have to.**
``plotter/`` takes loader *callbacks*; :func:`_loaders` supplies them, anchored
to the document being read, and :func:`_resolve_source` and
:func:`_within_ceiling` are the two questions asked of every path before it
becomes a read: may this be followed at all, and is it small enough to hold.

Every task key carries the ``plotter-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from . import dialogs, docmodes, plotter_state
from .plotter_state import PlotterDoc, active, ensure

# One row, all three patterns on it. Written as four entries once, which
# portable-file-dialogs pairs as ("Map documents (...)", "*.wmap") and
# ("*.tmx", "*.tmj") -- so "Open a map" could not see a Tiled file at all, and
# the second row of the dropdown was titled ``*.tmx``.
MAP_FILTER = [
    "Map documents (*.wmap *.tmx *.tmj)",
    "*.wmap *.tmx *.tmj",
]
WMAP_FILTER = ["Warlock map (*.wmap)", "*.wmap"]
TMX_FILTER = ["Tiled map (*.tmx)", "*.tmx"]
TMJ_FILTER = ["Tiled map (*.tmj)", "*.tmj"]


# --- image loading ------------------------------------------------------------


# The lazy-PIL RGBA decode is one rule for every mode that opens an image;
# ``_within_ceiling`` below is *not*, because it reads the map document's own
# ceiling.
_decode = docmodes.decode_rgba


def _within_ceiling(path: Path) -> Path:
    """Refuse a file too big to open, before a byte of it is read.

    The ceiling is ``service.files.MAX_MAP_SOURCE_BYTES`` -- the same number the
    service already refuses an *uploaded* ``map.wmap`` at -- rather than a second
    one invented here, because "how big may a map document be" has one answer and
    two would drift the first time either moved. Imported inside the function so
    this module keeps paying nothing for the service layer until it opens a file,
    and read at call time so a test lowers it rather than building 250 MB.

    A ``ServiceError`` rather than a ``ValueError``: this is the *mode's* refusal
    about a file the user picked, not the engine's about a document's contents,
    and its text reaches the user verbatim through the task classifier.
    ``_decode`` needs no companion ceiling -- Pillow's own ``MAX_IMAGE_PIXELS``
    is the decompression bomb guard for an image, and a second one here would be
    a second answer to that question too.
    """
    from ..service.errors import TooLarge
    from ..service.files import MAX_MAP_SOURCE_BYTES

    if path.stat().st_size > MAX_MAP_SOURCE_BYTES:
        raise TooLarge(
            f"{path.name} is past the {MAX_MAP_SOURCE_BYTES} bytes this build will open",
            field="file",
        )
    return path


def _resolve_source(base: Path, source: str) -> Path:
    """One path named *inside* a map or tileset file, anchored to that file.

    A ``.tmx`` and a ``.tsx`` both name their dependencies by path, and those
    paths come from a file rather than from the user -- which makes them
    untrusted input that this layer turns into a filesystem read.

    **Absolute, drive-qualified and UNC sources are refused; ``..`` is allowed.**
    The asymmetry is deliberate and it is Tiled's own convention that decides it:
    a tileset folder beside a maps folder is the *normal* layout, so
    ``../tilesets/grass.tsx`` is what a legitimate file says and containment to
    the map's own directory would refuse ordinary projects. What relative
    traversal cannot do is reach somewhere the user did not point this at: every
    such path is still rooted at the file they opened. An absolute one is not --
    ``C:\\Windows\\...`` reads whatever it names -- and a UNC path is worse than
    absolute: ``\\\\host\\share`` is a *network* read from a build whose first
    invariant is that it never goes online, issued because a file said so.
    """
    text = str(source)
    pure = PureWindowsPath(text)
    if pure.is_absolute() or pure.drive or pure.root or text.startswith(("\\\\", "//")):
        raise ValueError(
            f"this map names an absolute path ({text}), which Plotter does not follow"
        )
    return base / text


def _json(data: bytes) -> dict:
    """One tileset file's JSON body, for the collection probe.

    Parsed twice on that path -- once here and once inside ``read_tsj`` -- which
    is a few microseconds against a decode per tile, and buys the engine not
    having to hand back a "what do you need" answer before it can be asked for
    the tileset itself.
    """
    import json as jsonlib

    entry = jsonlib.loads(data.decode("utf-8"))
    return entry if isinstance(entry, dict) else {}



def _loaders(base: Path):
    """The two callbacks :mod:`~warlock.studio.plotter.tmx` needs.

    Resolving a relative path means touching a filesystem, which the engine
    deliberately cannot do -- so path resolution lives here, where it can be
    anchored to the file being read, and where :func:`_resolve_source` can say
    which paths are followable at all.
    """
    from .plotter import tsx as tsxlib

    def image_loader(source: str) -> np.ndarray:
        return _decode(_resolve_source(base, source))

    def tsx_loader(source: str) -> Any:
        target = _within_ceiling(_resolve_source(base, source))
        data = target.read_bytes()
        # **The host decides which spelling this is**, because the host is the
        # only thing that read the bytes. The engine hands over a reference and
        # has no way to tell a ``.tsj`` from a ``.tsx`` that is not the very
        # extension it is passing.
        json_tileset = target.suffix.lower() == ".tsj"
        read = tsxlib.read_tsj if json_tileset else tsxlib.read_tsx
        # Relative to the tileset file, not to the map: a tileset folder is the
        # normal Tiled layout and resolving from the map would miss by one
        # directory.
        sources = (
            tsxlib.collection_sources_json(_json(data))
            if json_tileset
            else tsxlib.collection_sources(tsxlib.xml_root(data, "tileset"))
        )
        if sources:
            # An image collection: every tile is its own file, so the host
            # fetches each and the engine composes. One decode per tile is the
            # whole cost, and it is what the format asks for.
            return read(
                data,
                {
                    local: _decode(_resolve_source(target.parent, source))
                    for local, source in sources.items()
                },
            )
        image = (tsxlib.tsj_source if json_tileset else tsxlib.tsx_source)(data)
        return read(data, _decode(_resolve_source(target.parent, image)))

    return {"image_loader": image_loader, "tsx_loader": tsx_loader}


# --- opening ------------------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab.

    **The engine's refusal is framed rather than forwarded.** ``read_tmx`` says
    *this file uses group layers, which Plotter does not support* -- precise, and
    a sentence with no subject in front of a user who pressed Open. ``invalid_from``
    puts one there and keeps the detail, which is the only part that says which
    file and which feature. ``TiledUnsupported`` is a ``ValueError`` subclass, so
    every named refusal flows through this one clause.
    """
    from ..service.errors import invalid_from
    from .plotter import tmx as tmxlib
    from .plotter import wmap as wmaplib

    path = _within_ceiling(Path(path))
    data = path.read_bytes()
    suffix = path.suffix.lower()
    try:
        if suffix == plotter_state.TMX_SUFFIX:
            doc = tmxlib.read_tmx(data, **_loaders(path.parent))
        elif suffix == plotter_state.TMJ_SUFFIX:
            doc = tmxlib.read_tmj(data, **_loaders(path.parent))
        else:
            doc = wmaplib.read_wmap(data)
    except ValueError as exc:
        raise invalid_from(exc, "This map could not be opened", field="file") from exc
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


# --- saving -------------------------------------------------------------------


# One rule for all four document modes: see :func:`docmodes.start_save` for why
# a refused submit has to clear the flag.
_start = docmodes.start_save


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


def _encoded(ctx: Any, doc: Any, file_format: str) -> dict[str, bytes] | None:
    """:func:`_encode`, with a writer-door refusal turned into a toast.

    Encoding used to be the one step in a save that could not fail, so every
    caller ran it bare on the frame thread. It can now: the Tiled exporters
    refuse by name what a document uses and they cannot yet spell, and an
    exception raised on the frame thread takes the window with it.

    ``.wmap``'s own door is quieter than it was and is still named here: its
    version 3 container stores everything version 2 refused, version 4 adds the
    1.12-era fields on top, and version 10 stores an infinite map, so nothing an
    ordinary document holds reaches ``WmapUnstorable`` today. The handler stays
    because the door does -- a layer kind the container has no entry for lands
    there -- and removing the plumbing in order to re-add it next wave would
    leave the intervening builds crashing the frame thread instead of
    toasting.

    **Both refusal types by name, and nothing wider.** ``TiledUnsupported``
    covers the two exporters and ``WmapUnstorable`` covers ours -- a door that
    caught only the Tiled spelling would let a ``.wmap`` save crash the window
    while a ``.tmx`` save of the same document toasted politely, and one that
    caught bare ``ValueError`` would be worse in the other direction: every
    genuine defect in an encoder raises one, and each would reach the user
    dressed as a polite refusal they are meant to act on, with the traceback
    swallowed.

    Toasted rather than framed through ``invalid_from`` like :func:`_load`'s
    refusals, because the two are different sentences: opening is an operation
    on a file the user chose. The frame in front of the refusal is the outcome
    rather than the action -- what somebody looking at a failed save needs first
    is that nothing was written, and every one of these refusals happens before
    a single byte reaches the disk.
    """
    from .plotter.tsx import TiledUnsupported
    from .plotter.wmap import WmapUnstorable

    try:
        return _encode(doc, file_format)
    except (TiledUnsupported, WmapUnstorable) as exc:
        ctx.toast(f"Nothing was written. {exc}", "error")
        return None


def _write(files: dict[str, bytes], path: Path) -> None:
    """Write an encoded map beside ``path``.

    The main document takes the user's chosen name; everything else keeps the
    relative path the exporter chose, because those are exactly the paths
    written *inside* the ``.tmx``. Each file goes through a temporary and an
    ``os.replace``, so a crash mid-write cannot leave a half-written map where a
    whole one was.

    **The staging name is a dotfile and the cleanup is a ``finally``.** The
    temporary used to be ``level.wmap.tmp`` -- visible in the folder the user
    picked, sorted right beside the file it is a fragment of -- and it was
    abandoned there by any failing write, because nothing removed it when the
    replace never happened. A leading dot keeps it out of the way while it
    exists, and ``finally`` means the only file left behind by a failure is the
    one that was already there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        # The main artifact by its exact exporter-chosen name, not by prefix:
        # ``startswith("map.")`` also matched an image layer whose source
        # happened to be ``map.png``, which then overwrote the map at the
        # user's chosen path with the picture (or the other way round,
        # depending on dict order).
        target = (
            path if name in ("map.wmap", "map.tmx", "map.tmj") else path.parent / name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        try:
            tmp.write_bytes(blob)
            os.replace(tmp, target)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


def save_to(ctx: Any, tab: PlotterDoc, path: Path, file_format: str) -> None:
    path = Path(path)
    head = tab.doc.history.head
    files = _encoded(ctx, tab.doc, file_format)
    if files is None:
        return

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
    files = _encoded(ctx, tab.doc, fmt)
    if files is None:
        return
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
    files = _encoded(ctx, tab.doc, file_format)
    if files is None:
        return

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Export for Tiled", f"{stem}{suffix}", filters)
        if path is None:
            return None
        path = path.with_suffix(suffix)
        _write(files, path)
        return {"exported": str(path)}

    _start(ctx, tab, f"plotter-export:{tab.uid}", run)
