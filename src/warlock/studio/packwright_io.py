"""Packwright's file layer: what an atlas document is read from and written to.

Split out of ``packwright_mode`` for the reason ``plotter_io`` was: it is the
half that touches a disk, and the rules that shape it are all about that. **No
file dialog and no encode ever runs on the frame thread** -- a native picker is
modal to the OS and blocks until dismissed, and a document of any size is a zip
of PNGs to build -- so both go through ``ctx.submit``, which is why saving is a
*state* (``PackTab.saving``) rather than a call that returns. **The head a save
records is captured before the submit**: a head read after an unbounded modal
dialog describes whatever the user did while it was open.

**Every write is staged**, through :func:`_write`: the whole set of files is
built in dotfile temporaries first and only then replaced into place, so an
export whose third file cannot be encoded leaves the previous three where they
were rather than half-overwriting them.

**The engine's refusals are framed rather than forwarded.** Only a
``ServiceError``'s text survives ``tasks.py``'s classifier; a bare ``ValueError``
out of ``read_wpack`` reaches the user as "Something went wrong; see the log for
details", which names neither the file nor the reason. So every refusal this
layer can raise is wrapped -- ``invalid_from`` keeps the engine's own sentence,
which is the half that says what to do about it.

Every task key carries the ``packwright-`` prefix, because the app claims
results by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from . import dialogs, docmodes, filetypes, packwright_state, sizeguard
from .packwright_state import PackTab, active, ensure

WPACK_FILTER = ["Warlock atlas (*.wpack)", "*.wpack"]
# Label *and* patterns from ``filetypes``, which is the whole point: the label
# was hand-written and named three of the five suffixes the pattern list
# accepted, so the dialog told the user two of the formats it would have opened
# were unsupported. ``pattern()`` rather than a splatted ``globs()``, because
# portable-file-dialogs pairs a filter list off two at a time -- splatting left
# this row filtering on ``*.png`` alone while advertising five formats.
IMAGE_FILTER = [filetypes.describe("Images"), filetypes.pattern()]
PNG_FILTER = ["PNG image (*.png)", "*.png"]

# The lazy-PIL RGBA decode and the submit-or-unlock helper are one rule for
# every document mode: see :mod:`.docmodes`.
_decode = docmodes.decode_rgba
_start = docmodes.start_save


# --- opening ------------------------------------------------------------------


def _within_ceiling(path: Path) -> Path:
    """Refuse a file too big to open, before a byte of it is read.

    The ceiling is ``service.files.MAX_PACK_SOURCE_BYTES`` -- the same number
    the service already refuses an *uploaded* ``pack.wpack`` at -- rather than a
    second one invented here, because "how big may an atlas document be" has one
    answer and two would drift the first time either moved. ``plotter_io``'s
    rule, applied to the other zip. Imported inside the function so this module
    keeps paying nothing for the service layer until it opens a file, and read
    at call time so a test lowers it rather than building half a gigabyte.

    A ``ServiceError`` rather than a ``ValueError``: this is the *mode's*
    refusal about a file the user picked, not the engine's about a document's
    contents, and its text reaches the user verbatim through the task
    classifier. ``read_wpack``'s own decompressed-bytes ceiling is the second
    door, on what the archive claims rather than on what it weighs.
    """
    from ..service.files import MAX_PACK_SOURCE_BYTES

    return sizeguard.within_ceiling(path, MAX_PACK_SOURCE_BYTES)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab.

    The engine's refusal is framed rather than forwarded: ``read_wpack`` says
    *this atlas document holds 's3' twice* -- precise, and a sentence with no
    subject in front of a user who pressed Open. ``invalid_from`` puts one there
    and keeps the detail, which is the only part that says which file and why.
    """
    from ..service.errors import invalid_from
    from .packwright import wpack

    path = _within_ceiling(Path(path))
    try:
        doc = wpack.read_wpack(path.read_bytes())
    except ValueError as exc:
        raise invalid_from(exc, "This atlas could not be opened", field="file") from exc
    return {
        "doc": doc,
        "path": str(path),
        "title": packwright_state.title_for(path),
    }


def ask_open(ctx: Any) -> None:
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open an atlas document", WPACK_FILTER)
        return None if path is None else _load(path)

    ctx.submit("packwright-open", run)


#: The key prefix an open-by-path carries, and the claim on everything after
#: the first colon. The path itself is the rest of the key rather than a hash of
#: it, so a *failure* can name the file that did not open and drop it off Home's
#: Resume list -- which a hash cannot, and a recent list that keeps offering a
#: file that does not open is worse than a short one. Split on the first colon
#: only, which leaves a drive letter intact.
OPEN_PREFIX = "packwright-open:"


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"{OPEN_PREFIX}{path}", _load, path)


# --- writing ------------------------------------------------------------------


def _write(files: dict[Path, bytes]) -> None:
    """Write a whole set of files: stage all of them, then replace each.

    ``plotter_io._write``'s rule -- a dotfile temporary and an ``os.replace``,
    cleaned up in a ``finally`` -- with one thing added. **Every file is staged
    before any of them is replaced.** An atlas export is up to three files, and
    an encode that raised on the third used to leave the first two already
    written on top of the previous export: half of one export and half of
    another, under names that say they belong together.

    What that does not close is the window *between* the replaces, and it is
    stated rather than papered over: two of three can land and the process die
    before the third. Closing it needs a directory-level transaction no
    filesystem here offers, and the alternative -- staging into a temporary
    directory and renaming that -- would take the user's chosen name off the
    file they picked in the dialog.

    The staging name is a dotfile so a temporary that does survive is out of the
    way rather than sorted right beside the file it is a fragment of, and the
    ``finally`` means the only file a failure leaves behind is the one that was
    already there.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for target, blob in files.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp")
            tmp.write_bytes(blob)
            staged.append((tmp, target))
        for tmp, target in staged:
            os.replace(tmp, target)
    finally:
        for tmp, _target in staged:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


# --- saving -------------------------------------------------------------------


def save_to(ctx: Any, tab: PackTab, path: Path) -> None:
    from .packwright import wpack

    path = Path(path)
    head = tab.doc.history.head
    data = wpack.wpack_bytes(tab.doc)

    def run() -> dict[str, Any]:
        _write({path: data})
        return {"head": head, "path": str(path), "retitle": True}

    _start(ctx, tab, f"packwright-save:{tab.uid}", run)


def save(ctx: Any, tab: PackTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    save_to(ctx, tab, tab.path)


def save_as(ctx: Any, tab: PackTab | None = None) -> None:
    from .packwright import wpack

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    head = tab.doc.history.head
    data = wpack.wpack_bytes(tab.doc)
    stem = Path(tab.title).stem or "atlas"

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save the atlas document", f"{stem}{packwright_state.WPACK_SUFFIX}", WPACK_FILTER
        )
        if path is None:
            return None
        path = path.with_suffix(packwright_state.WPACK_SUFFIX)
        _write({path: data})
        return {"head": head, "path": str(path), "retitle": True}

    _start(ctx, tab, f"packwright-saveas:{tab.uid}", run)


# --- exporting ----------------------------------------------------------------


def export_files(ctx: Any, tab: PackTab | None = None) -> None:
    """The atlas PNG, the JSON sidecar, and a ``.tsx`` when the pack is a grid.

    One picker for the PNG; the others take its stem and sit beside it, which is
    the layout every consumer expects and the only one where the sidecar's
    ``meta.image`` resolves. The PNG and the JSON sidecar are encoded before
    either is written, so *that* pair lands together or not at all.

    **The ``.tsx`` is the one encode step that does not abort the export.**
    ``tsxout.grid_tileset`` refuses by name when an explicit ``columns``
    count's power-of-two rounding has left the image geometry disagreeing
    with what Tiled would derive (see ``layout.grid_layout``) -- and that is
    a statement about the *tileset*, not about the PNG or the JSON, which
    describe exactly what was packed regardless. Aborting the whole export
    over a file that was never going to be trustworthy would throw away two
    files that are; instead the ``.tsx`` is skipped and the result carries
    why, for :func:`~.packwright_mode.on_task_done` to toast.
    """
    from .packwright import compose as composelib
    from .packwright import texturepacker, tsxout

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.layout is None or tab.atlas is None:
        ctx.toast("Nothing packed yet.", "error")
        return
    layout, atlas = tab.layout, tab.atlas
    # Snapshotted here, with ``layout``/``atlas`` above, for the same reason
    # ``request_pack`` snapshots settings before its own task closure: a
    # setting read at run time reflects whatever the user changed while the
    # save dialog was open (unbounded, modal), not what was packed.
    schema = tab.doc.settings.json_schema
    stem = Path(tab.title).stem or "atlas"

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Export the atlas", f"{stem}.png", PNG_FILTER)
        if path is None:
            return None
        path = path.with_suffix(".png")
        files = {
            path: composelib.png_bytes(atlas),
            path.with_suffix(".json"): texturepacker.tp_bytes(
                layout, image_name=path.name, schema=schema
            ),
        }
        tsx_skipped = None
        if layout.is_grid:
            try:
                files[path.with_suffix(".tsx")] = tsxout.grid_tsx(
                    layout, atlas, name=stem, image_name=path.name
                )
            except ValueError as exc:
                tsx_skipped = str(exc)
        _write(files)
        result: dict[str, Any] = {"exported": str(path), "files": len(files)}
        if tsx_skipped is not None:
            result["tsx_skipped"] = tsx_skipped
        return result

    _start(ctx, tab, f"packwright-export:{tab.uid}", run)


def export_library(ctx: Any, tab: PackTab | None = None) -> None:
    """Mint an ordinary asset from the atlas, with ``pack.wpack`` beside it."""
    from .packwright import compose as composelib
    from .packwright import wpack

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.atlas is None:
        ctx.toast("Nothing packed yet.", "error")
        return
    # Unlike :func:`export_files`, whose PNG and sidecar both derive from the
    # *landed* pack and so agree with each other, this pairs the landed atlas
    # with the **current** document's bytes -- so while an edit is unpacked
    # (``pack_dirty``) or a repack is in flight, the picture and the document
    # beside it would describe two different atlases. Refused rather than
    # snapshotted, because the atlas that matches the current document does
    # not exist yet; the preview pane's pump repacks within a frame.
    if tab.pack_dirty or tab.packing:
        ctx.toast("Still packing your latest edits -- try again in a moment.", "error")
        return
    png = composelib.png_bytes(tab.atlas)
    source = wpack.wpack_bytes(tab.doc)
    title = tab.title

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from ..service import jobs as svc_jobs

        result = svc_jobs.import_reference(
            ctx.svc, png, name=title, prompt=title, authored="packwright"
        )
        job_id = result["id"]
        svc_files.save_packwright_source(ctx.svc, job_id, source)
        return {"job_id": job_id, "exported_asset": True}

    _start(ctx, tab, f"packwright-library:{tab.uid}", run)


def edit_asset_in_packwright(ctx: Any, job: Any) -> None:
    """Reopen the ``pack.wpack`` kept beside a library asset.

    Both ways this fails are framed. An asset minted before Packwright existed,
    or one imported from anywhere else, has no atlas document beside it at all
    -- a raw ``FileNotFoundError`` for that reaches the user as the generic
    log-pointer, which is the wrong answer to a question with a plain one. The
    other is the file being unreadable, which is ``_load``'s clause again.
    """
    from ..service.errors import Invalid, invalid_from

    job_id = job["id"] if isinstance(job, dict) else str(job)
    ensure(ctx)

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from .packwright import wpack

        path = Path(svc_files.packwright_source_path(ctx.svc, job_id))
        if not path.exists():
            raise Invalid("This asset has no atlas document beside it.", field="file")
        try:
            doc = wpack.read_wpack(_within_ceiling(path).read_bytes())
        except ValueError as exc:
            raise invalid_from(exc, "This atlas could not be opened", field="file") from exc
        return {"doc": doc, "path": "", "title": "Atlas"}

    ctx.submit(f"{OPEN_PREFIX}{job_id}", run)
