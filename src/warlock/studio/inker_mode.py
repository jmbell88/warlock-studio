"""Inker mode's controller: opening, saving, guarding, keys, and the bridge.

Everything here is *about* documents rather than pixels -- the engine under
``inker/`` has no idea a job or a task thread exists, and this is the layer that
knows about both. The panes draw; this decides.

The one rule that shapes the whole file: **no file dialog and no encode ever
runs on the frame thread.** A native picker is modal to the OS and blocks until
dismissed, and a 4096-square ORA is a second of zlib. Both go through
``ctx.submit`` and come back through ``on_task_done``, which is why saving is a
state (``InkerDoc.saving``) rather than a function call that returns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pipelines import sheet as sheetlib
from . import dialogs, docmodes, filetypes, inker_state, journal, recents
from .inker import animation
from .inker_state import InkerDoc, InkerState
from .state import set_mode

log = logging.getLogger(__name__)

ORA_FILTER = ["OpenRaster (*.ora)", "*.ora"]
PNG_FILTER = ["PNG image (*.png)", "*.png"]
GIF_FILTER = ["Animated GIF (*.gif)", "*.gif"]

# The layered format plus every image the app accepts anywhere -- the tuple
# from ``filetypes``, not a copy of it, so the picker and the suffix check can
# never disagree with each other or with what a drop accepts.
OPENABLE = (".ora", *filetypes.IMAGE_SUFFIXES)
OPEN_FILTER = ["Images and layered files", filetypes.pattern(OPENABLE)]

NEW_PRESETS = ((512, 512), (1024, 1024), (2048, 2048))

# The largest canvas the New dialog will make. Not a limit of the engine, which
# is happy with anything numpy can allocate -- it is a limit on what a *typed*
# number may do: the fields step and accept free text, so one stray digit turns
# 2048 into 20480, which is a 1.7 GiB layer allocated on the frame thread. The
# resize popup is deliberately not capped the same way, because there the
# document already exists and shrinking it is the usual reason to open it.
NEW_MAX = 8192


def clamp_canvas(width: Any, height: Any) -> tuple[int, int]:
    """A typed size, made safe. Clamped rather than refused, the snap rule:
    the fields are being *typed into*, and there is nothing useful for a
    refusal to show halfway through a number."""
    def one(value: Any) -> int:
        try:
            return max(1, min(int(value), NEW_MAX))
        except (TypeError, ValueError):
            return 1

    return one(width), one(height)


def ensure(ctx: Any) -> InkerState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Paint should not pay for its
    swatches, and because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.inker
    if state is None:
        state = InkerState()
        stored = ctx.settings.get("inker") or {}
        swatches = stored.get("swatches")
        if isinstance(swatches, list) and swatches:
            state.swatches = [
                tuple(int(c) for c in s)  # type: ignore[misc]
                for s in swatches
                if isinstance(s, list | tuple) and len(s) == 4
            ] or list(inker_state.DEFAULT_SWATCHES)
        ctx.state.inker = state
    return state


def remember_path(ctx: Any, path: Any) -> None:
    """Put ``path`` at the front of the merged recent list.

    Through :mod:`.recents` rather than onto a field of this mode's own state:
    the four document modes kept four independent ``recent`` lists, and Home's
    single Resume list cannot be built from them at all -- four bare path lists
    carry no ordering *between* them. There is one list now, and this is how
    inker writes to it.
    """
    recents.remember(ctx.settings, "inker", path)


def forget_path(ctx: Any, path: Any) -> None:
    """Drop a path that turned out not to open -- :mod:`.recents`' own rule,
    named here so a caller does not have to know this mode's kind string."""
    recents.forget(ctx.settings, "inker", path)


def recent_paths(ctx: Any) -> list[str]:
    """This mode's recent files, newest first. What its own panel draws."""
    return recents.paths(ctx.settings, "inker")


def persist(ctx: Any) -> None:
    """The swatches, and only those: the recent list moved to :mod:`.recents`,
    which persists itself on every write."""
    state = ctx.state.inker
    if state is None:
        return
    # Merged into whatever is stored rather than replacing it, so the legacy
    # ``recent`` key survives untouched: ``recents`` folds the four per-mode
    # lists in on *its* first read, which may well be after this has run.
    stored = ctx.settings.get("inker")
    block = dict(stored) if isinstance(stored, dict) else {}
    block["swatches"] = [list(s) for s in state.swatches]
    ctx.settings.set("inker", block)


def active(ctx: Any) -> InkerDoc | None:
    state = ctx.state.inker
    return state.active if state is not None else None


# --- opening ----------------------------------------------------------------


def new_document(ctx: Any, width: int, height: int) -> InkerDoc:
    from . import inker

    state = ensure(ctx)
    width, height = clamp_canvas(width, height)
    doc = inker.Document.blank(width, height)
    return _adopt(ctx, state, doc, path=None, title="Untitled", file_format="ora")


def _adopt(
    ctx: Any,
    state: InkerState,
    doc: Any,
    *,
    path: Path | None,
    title: str | None = None,
    file_format: str = "png",
    job_id: str = "",
    link_kind: str = "",
    has_original: bool = False,
    saved_head: int | None = None,
) -> InkerDoc:
    tab = InkerDoc(
        doc=doc,
        title=title or inker_state.title_for(path),
        path=path,
        file_format=file_format,
        # Normally the document as opened *is* what is on disk. The one caller
        # that passes this is the matte hand-off, which opens a document with
        # an edit already applied: recording the head after that edit would
        # call the cutout saved, and closing the tab would discard it without
        # asking.
        saved_head=doc.history.head if saved_head is None else saved_head,
        job_id=job_id,
        link_kind=link_kind,
        has_original=has_original,
    )
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open image", OPEN_FILTER)
        return None if path is None else _load(path)

    ctx.submit("inker-open", run)


def open_path(ctx: Any, path: Path) -> None:
    """Open a known path -- a drop, or a click in the recent list."""
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one file would race on save.
        state.activate(existing.uid)
        return
    if path.suffix.lower() not in OPENABLE:
        ctx.toast("Inker opens images and .ora files.", "error")
        return
    ctx.submit(f"inker-open:{abs(hash(str(path)))}", _load, path)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only."""
    from . import inker

    path = Path(path)
    doc = inker.Document.load(path)
    return {"doc": doc, "path": path, "format": doc.file_format}


def open_pixels(ctx: Any, pixels: Any, *, title: str = "Untitled") -> None:
    """Open an in-memory RGBA array as an ordinary, unlinked document.

    Plotter's polish round trip comes through here. Unlinked deliberately, for
    ``open_sprite_draft``'s reason: it carries no ``job_id`` and no
    ``link_kind``, so saving it cannot write back over anything -- the way back
    to the map is Plotter pulling the finished document in, not this pushing.

    Routed on the ``inker-open`` prefix so ``on_task_done`` adopts it with no
    routing change, and the copy happens on the task thread because the caller's
    array is routinely a tileset's frozen pixels, which nothing may write into.
    """
    import numpy as np

    ensure(ctx)
    set_mode(ctx.state, "inker")
    array = np.array(pixels, dtype=np.uint8)

    def run() -> dict[str, Any]:
        from . import inker

        return {"doc": inker.Document.from_pixels(array, name="Atlas"), "title": title}

    ctx.submit(f"inker-open:pixels:{title}", run)


def ask_import_sheet(ctx: Any) -> None:
    """Pick a sprite sheet off disk. The grid is asked for afterwards.

    Two steps rather than one, and in this order deliberately: the popup that
    asks for the cell size shows how many frames those numbers actually
    produce, which it can only do once the image's size is known. Asking first
    and opening second would mean typing a grid blind and finding out by
    looking at the result.
    """
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        import numpy as np
        from PIL import Image

        path = dialogs.open_file("Import sprite sheet", OPEN_FILTER)
        if path is None:
            return None
        with Image.open(path) as opened:
            opened.load()
            atlas = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
        return {"atlas": atlas, "title": Path(path).stem}

    ctx.submit("inker-sheetin", run)


def import_sheet(ctx: Any) -> bool:
    """Slice the pending atlas on the typed grid and open it. Frame thread.

    Cheap enough to stay here: it is a handful of array copies, and every
    refusal is a message the user has to see beside the fields that caused it
    rather than a toast arriving from a task some frames later.

    Returns whether a document was opened, so the popup can stay up on a
    refusal. Closing it either way would strand the atlas: the popup only
    reopens when a *new* file is picked, so a rejected grid would mean choosing
    the same file again to correct one number.
    """
    from .inker import sheetin

    state = ensure(ctx)
    pending = state.sheet_import
    if pending is None:
        return False
    atlas, title = pending
    try:
        doc = sheetin.document_from_grid(
            atlas,
            state.sheet_cell,
            state.sheet_offset,
            state.sheet_padding,
            state.sheet_count or None,
        )
    except ValueError as exc:
        ctx.toast(f"Cannot import: {exc}.", "warn")
        return False
    state.sheet_import = None
    state.sheet_import_open = False
    _adopt(ctx, state, doc, path=None, title=title, file_format="ora")
    set_mode(ctx.state, "inker")
    return True


def open_sprite_draft(ctx: Any, job_id: str, draft_id: str, candidate: str) -> None:
    """Open one candidate of a sprite draft as an editable animation.

    Routed through the ``inker-open`` key prefix, so ``on_task_done`` adopts
    the result with no routing change at all -- the whole of what is new here
    is what ``_load_sprite_draft`` builds.

    Deliberately *not* a linked document: it carries no ``job_id`` and no
    ``link_kind``, so it is not on the reference-edit write-back path. Saving
    it must not overwrite the draft, because a draft is one of a pair the user
    is choosing between and its sibling is still on disk beside it -- so the
    first save is a Save As, wherever the user wants the sheet to live.

    Opening the same candidate twice makes two tabs. Accepted for now: they are
    genuinely two independent copies of the atlas, and neither can clobber the
    other precisely because neither owns a file.
    """
    ensure(ctx)
    set_mode(ctx.state, "inker")
    ctx.submit(
        f"inker-open:sprite:{draft_id}:{candidate}",
        _load_sprite_draft,
        ctx.svc,
        job_id,
        draft_id,
        candidate,
    )


def _load_sprite_draft(
    svc: Any, job_id: str, draft_id: str, candidate: str
) -> dict[str, Any]:
    """Blocking; task thread only."""
    import numpy as np
    from PIL import Image

    from ..service import sprites as svc_sprites
    from .inker import sheetin

    record = svc_sprites.get_sprite_draft(svc, job_id, draft_id)
    png = svc_sprites.sprite_draft_png(svc, job_id, draft_id, candidate)
    with Image.open(png) as opened:
        opened.load()
        atlas = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
    doc = sheetin.document_from_atlas(
        atlas, record["cells"], str(record.get("sheet_type") or "")
    )
    return {
        "doc": doc,
        "path": None,
        "format": "ora",
        "title": f"{record.get('sheet_type', 'sprite')} {draft_id[:6]}{candidate}",
    }


# --- the job bridge ---------------------------------------------------------


def can_edit_job(ctx: Any, job: Any) -> bool:
    """Whether the "Open in Inker" button belongs on this job's toolbar.

    From the cached row alone -- no filesystem calls, because the toolbar asks
    this every frame.

    Both image stages, and the service's ``files.EDITABLE_STAGES`` is the
    authority rather than a second list here: this is a *permission*, and a
    pane that offered a button the service refuses is the drift
    ``derived_2d_for`` exists to prevent one artifact set over. A tile's albedo
    is the asset, its whole material set derives from it, and the derived maps
    re-derive against its mtime for free.
    """
    from ..service import files as svc_files

    return bool(
        job
        and job.get("stage") in svc_files.EDITABLE_STAGES
        and job.get("status") == "done"
        and "input.png" in (job.get("files") or [])
    )


def open_job_reference(ctx: Any, job: Any, *, matte: bool = False) -> None:
    """Open a reference's image as a linked document.

    Prefers the layered working file when there is a fresh one, so layers
    survive between sessions; falls back to the flat input.png, which is also
    what happens after a revert or a regenerate rewrites the reference behind
    the working file's back.

    ``matte`` is the promote preview's "Fix matte": the document opens with the
    host's cutout already folded into its alpha as one undoable step, so the
    eraser and the brush edit the matte directly. A tab that is *already* open
    for this job is focused rather than re-opened even then -- re-cutting a
    document the user has been editing would apply a matte measured off
    ``input.png`` to pixels that have moved on from it, and the alpha is right
    there to paint.
    """
    state = ensure(ctx)
    job_id = job["id"]
    existing = state.find_job(job_id)
    if existing is not None:
        state.activate(existing.uid)
        set_mode(ctx.state, "inker")
        return
    set_mode(ctx.state, "inker")
    ctx.submit(f"inker-open:{job_id}", _load_job, ctx.svc, job_id, matte=matte)


def _load_job(svc: Any, job_id: str, *, matte: bool = False) -> dict[str, Any]:
    """Blocking; task thread only."""
    from ..service import files as svc_files
    from . import inker

    flat = svc.job_dir(job_id) / "input.png"
    working = svc_files.inker_working_path(svc, job_id)
    status = svc_files.inker_working_status(svc, job_id)
    doc = inker.Document.load(working if status["fresh"] else flat)
    # The document is *about* input.png whichever file it was decoded from:
    # the title, the dedupe and the save all key on the reference.
    doc.path = flat
    edit = svc_files.reference_edit_status(svc, job_id)
    # The tab says which kind of image it holds, because the two are edited for
    # different reasons and a tile's tab called "reference" reads as the wrong
    # asset opened. Blocking thread, so the extra row read is free here.
    stage = (svc.store.get(job_id) or {}).get("stage")
    out = {
        "doc": doc,
        "path": flat,
        "format": "ora",
        "job_id": job_id,
        "link_kind": "reference-edit",
        "has_original": bool(edit.get("has_original")),
        "title": f"{job_id[:8]} {'tile' if stage == 'tile' else 'reference'}",
    }
    if matte:
        # Captured before the cut, and handed back as the tab's saved head: the
        # cutout is an unsaved edit, because nothing has written it to disk.
        out["saved_head"] = doc.history.head
        # Recorded, not merely done. ``_cut_matte`` log-and-swallows, which is
        # right -- a failed matte must not cost the user the reference they
        # asked for -- but "Fix matte" is a command whose *whole content* is
        # the matte, and swallowing it there meant the menu item opened a tab
        # that looked exactly like the one Edit opens and said nothing. The
        # completion branch turns the pair into a toast; see ``on_task_done``.
        out["matte_requested"] = True
        out["matte_applied"] = _cut_matte(svc, job_id, doc)
    return out


def _cut_matte(svc: Any, job_id: str, doc: Any) -> bool:
    """Fold the host's cutout into ``doc``'s alpha. Blocking; task thread only.

    Log-and-swallow, like every other improvement the user did not ask for:
    the thing they asked for is the reference open in the editor, and BiRefNet
    failing (or a working file whose canvas has been resized away from the
    reference's) must not cost them that.
    """
    from ..service import matte as svc_matte

    try:
        alpha, _source = svc_matte.alpha_plane(svc, job_id)
        return bool(doc.apply_matte(alpha))
    except Exception:
        log.exception("could not apply the matte to %s", job_id)
        return False


# --- saving -----------------------------------------------------------------


def save(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Ctrl+S. Save As when the document has never been written anywhere."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.linked:
        _save_linked(ctx, tab)
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    _submit_write(ctx, tab, f"inker-save:{tab.uid}", tab.path, tab.file_format)


def save_as(ctx: Any, tab: InkerDoc | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    doc.commit_floating()  # before the head is read; see _save_linked
    rev = doc.history.head
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Save layered document", f"{suggested}.ora", ORA_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".ora":
            dest = dest.with_suffix(".ora")
        _write(doc, dest, "ora")
        return {"path": dest, "rev": rev, "format": "ora", "retitle": True}

    _start(ctx, tab, f"inker-saveas:{tab.uid}", run)


def export_png(ctx: Any, tab: InkerDoc | None = None) -> None:
    """A flattened PNG. Not a save: it does not change what the tab points at,
    so the document stays dirty against its own file."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    # Not a save, but the same rule about what is on the canvas: the composite
    # a floating buffer draws into is the pane's, not the document's, so an
    # export would otherwise be missing pixels the user is looking at.
    doc.commit_floating()
    suggested = tab.path.stem if tab.path else "untitled"
    state = ctx.state.inker
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Export flattened PNG", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        dest.write_bytes(doc.png_bytes(scale=scale))
        return {"exported": dest}

    _start(ctx, tab, f"inker-export:{tab.uid}", run)


def export_sheet(ctx: Any, tab: InkerDoc | None = None) -> None:
    """An animated document as a packed PNG plus its JSON sidecar.

    Mirrors ``export_png`` exactly -- gated, floating buffer committed first, one
    task under the same key so the two can never run at once, and
    ``{"exported": path}`` back so the existing completion branch toasts it
    unchanged. What differs is only what gets written.

    With one addition the other exports do not need: the frames are read off the
    document on the **frame thread**, because ``_write`` gets away with encoding
    the live document (the encoders only read) and flattening a clip does not --
    it fills and evicts the document's frame cache and copies track properties
    down onto cels, the same structures the onion-skin draw is walking sixty
    times a second.

    That read used to happen inline, in this call. A sixty-frame clip is sixty
    flattens on the frame the user clicked the button, which is a freeze; and it
    cannot move to a task thread for the reason above. So it is spread instead:
    the tab is locked (``saving``, which already refuses every mutation), a
    stepper is parked on the mode state, and :func:`pump_export` flattens one
    frame per app frame until the work list is done. Then, and only then, the
    encode is submitted. This is ``viewer/sheet.StripRender``'s answer to
    exactly the same problem -- sixteen GPU readbacks in one frame versus
    sixteen frames of one -- at a different layer.
    """
    _begin_export(ctx, tab, "sheet")


def export_gif(ctx: Any, tab: InkerDoc | None = None) -> None:
    """An animated document as a GIF anyone can open.

    ``export_sheet``'s shape exactly, down to sharing its task key -- the two
    read the same frames off the same document and must not run at once -- and
    the same frame-spread read, through the same stepper.

    A GIF loops forever rather than honouring a tag's loop flag, and that is the
    honest reading rather than a shortcut: a tag names a *span* of the timeline
    and the export is the whole timeline, so there is no one tag whose flag this
    could be. Exporting a single tag is a different feature and would need to say
    which one.
    """
    _begin_export(ctx, tab, "gif")


def export_pngs(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Every frame as its own numbered PNG, through the same stepper.

    The plainest export there is, and the one an engine with its own importer
    asks for: no atlas to slice, no sidecar to parse. The spread is untouched --
    the frames are read exactly as the sheet and the GIF read them, and only
    the write differs.
    """
    _begin_export(ctx, tab, "pngs")


@dataclass
class _Export:
    """One export's frame-by-frame read of the document.

    Lives on ``InkerState`` rather than on the tab because it is not a property
    of the document -- it is one in-flight operation, and there is one at a time
    by construction (both exports share a task key, and the tab is locked while
    it runs).
    """

    tab: InkerDoc
    kind: str  # "sheet" | "gif" | "pngs"
    suggested: str
    uids: list[str]
    frames: list[Any] = field(default_factory=list)
    #: The inclusive frame range being exported, or None for the whole
    #: timeline. Sliced **at begin**, and ``timing`` is sliced at submit with
    #: this same pair -- safe because the tab has been locked (``saving``) for
    #: the whole spread, so the frame count cannot have moved between them.
    span: tuple[int, int] | None = None
    #: What a GIF's loop block should say: True forever, False once, or a
    #: repeat count. See ``gifout.loop_option``.
    loop: bool | int = True

    @property
    def done(self) -> bool:
        return len(self.frames) >= len(self.uids)

    @property
    def total(self) -> int:
        return len(self.uids)


def export_range(
    ctx: Any,
    tab: InkerDoc | None,
    kind: str,
    span: tuple[int, int],
    *,
    loop: bool | int = True,
) -> None:
    """Export part of the timeline -- a tag, or a marquee'd range.

    The same three exports over fewer frames, so it is the same entry point
    with a span rather than a second pipeline: the frames are read by the same
    stepper, the durations and tags by the same ``timing``, and the sheet is
    written by the same ``sheet.sidecar``. What a span changes is only which
    frames go in, that the tags come back renumbered, and that a directional
    layout is dropped -- all of which ``sheetout`` decides, not this.
    """
    _begin_export(ctx, tab, kind, span=span, loop=loop)


def export_tag(ctx: Any, tab: InkerDoc | None, kind: str, index: int) -> None:
    """One tag, as a sheet or a GIF, with its own looping honoured.

    ``tag.repeat or tag.loop`` is the whole of the difference from a range
    export: a repeat count is the more specific answer to "how many times does
    this play", and 0 means the flag decides -- exactly the rule playback
    follows, spelled once here so the file and the editor cannot disagree.
    """
    tab = tab or active(ctx)
    anim = None if tab is None else tab.doc.anim
    if tab is None or anim is None or not 0 <= index < len(anim.tags):
        return
    tag = anim.tags[index]
    last = len(anim.frames) - 1
    span = (max(0, min(int(tag.start), last)), max(0, min(int(tag.end), last)))
    _begin_export(ctx, tab, kind, span=span, loop=tag.repeat or tag.loop)


def _begin_export(
    ctx: Any,
    tab: InkerDoc | None,
    kind: str,
    *,
    span: tuple[int, int] | None = None,
    loop: bool | int = True,
) -> None:
    """Lock the tab and park the stepper. The click-frame half of an export."""
    from .inker import sheetout

    tab = tab or active(ctx)
    state = ctx.state.inker
    if tab is None or tab.busy or tab.doc.anim is None or state is None:
        return
    if state.export is not None:
        return
    tab.doc.commit_floating()
    try:
        uids = sheetout.frame_uids(tab.doc, span)
    except ValueError as exc:
        ctx.toast(f"Cannot export: {exc}.", "warn")
        return
    # Locked before the first flatten, not at submit time: the whole point of
    # spreading the read is that frames go by between here and the encode, and
    # an edit landing in one of them would put half of two documents in the
    # sheet. ``saving`` is the flag ``busy`` already refuses mutation on.
    tab.saving = True
    state.export = _Export(
        tab=tab,
        kind=kind,
        suggested=tab.path.stem if tab.path else "untitled",
        uids=uids,
        span=span,
        loop=loop,
    )


def pump_export(ctx: Any) -> None:
    """One frame of an in-flight export's read. Called once a frame by the app.

    Beside ``journal.pump`` and in every mode for the same reason: a user who
    started an export and switched to the library must still get their file.
    """
    state = getattr(ctx.state, "inker", None)
    export = None if state is None else state.export
    if export is None:
        return
    from .inker import sheetout

    tab = export.tab
    if tab not in state.docs or tab.doc.anim is None:
        # The tab was closed under the export. Nothing has been written and the
        # lock goes with the tab, so there is nothing to undo.
        state.export = None
        return
    try:
        export.frames.append(
            sheetout.flatten_one(tab.doc, export.uids[len(export.frames)])
        )
    except (ValueError, IndexError, KeyError):
        state.export = None
        tab.saving = False
        ctx.toast("Export failed: a frame could not be flattened.", "warn")
        return
    if not export.done:
        return
    state.export = None
    _submit_export(ctx, export)


def _submit_export(ctx: Any, export: _Export) -> None:
    """The work list is read; hand it to a task. Frame thread."""
    from .inker import gifout, sheetout
    from .inker.transform import upscale

    tab, frames, suggested = export.tab, export.frames, export.suggested
    doc = tab.doc
    state = ctx.state.inker
    # Read here, on the frame thread, with the frames: an app-level setting the
    # user could change while the encode is in flight would otherwise decide
    # the file's size halfway through writing it.
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))
    durations, tags, layout = sheetout.timing(doc, export.span)
    if export.kind == "sheet" and layout is not None and len(frames) != layout.frame_count:
        # Refused on the frame thread, before the file dialog: the engine raises
        # the same ValueError as a backstop, but by then the user has picked a
        # filename and the failure arrives as a task error with no obvious
        # cause. A frame added to (or removed from) a sprite sheet is an
        # ordinary edit, so the fix is to say which count is wrong.
        tab.saving = False
        ctx.toast(
            f"This is a {layout.kind} sheet of {layout.frame_count} frames and "
            f"the document has {len(frames)}.",
            "warn",
        )
        return

    def run_sheet() -> dict[str, Any] | None:
        import json

        dest = dialogs.save_file("Export sprite sheet", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        # Upscaled *before* ``compose``, so the plan is built on the scaled
        # frame size and the cells, the trims and the sidecar all describe the
        # atlas that is actually written. Scaling the finished atlas instead
        # would leave every rectangle in the sidecar naming the wrong pixels.
        # ``sheet.py`` stays the sole writer of the format; none of this is new
        # code in it.
        image, plan, extra = sheetout.compose(
            [upscale(plane, scale) for plane in frames],
            durations,
            tags,
            layout,
            name=suggested,
        )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest, "PNG")
        finally:
            image.close()
        meta = sheetlib.sidecar(
            plan,
            sheet_id=dest.stem,
            source_job=tab.job_id,
            image=dest.name,
            created=time.time(),
            name=suggested,
            trims=extra["trims"],
            animation=extra["animation"],
        )
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"exported": dest}

    # The document's own table when it has one, so an indexed clip exports the
    # colours that were authored rather than a per-frame quantise of them. Read
    # on the frame thread here, with the frames, not inside the task.
    palette = list(doc.palette) if doc.palette else None

    def run_gif() -> dict[str, Any] | None:
        dest = dialogs.save_file("Export animated GIF", f"{suggested}.gif", GIF_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".gif":
            dest = dest.with_suffix(".gif")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Upscaled before the quantiser, not after: a GIF holds palette
        # indices, so there is no "after" -- magnifying the indexed image would
        # be magnifying a palette lookup rather than a picture.
        gifout.write_gif(
            dest,
            [upscale(plane, scale) for plane in frames],
            durations,
            loop=export.loop,
            palette=palette,
        )
        return {"exported": dest}

    def run_pngs() -> dict[str, Any] | None:
        """One PNG per frame, numbered. The plainest thing an engine can eat.

        Numbered from the chosen filename's stem rather than asking for a
        directory: every tool that consumes a sequence wants ``name_0000.png``
        beside its siblings, and a save dialog is the one place a user is
        already picking both the folder and the name.
        """
        from PIL import Image

        dest = dialogs.save_file("Export PNG sequence", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        stem = dest.stem
        dest.parent.mkdir(parents=True, exist_ok=True)
        first = dest
        for index, plane in enumerate(frames):
            out = dest.parent / f"{stem}_{index:04d}.png"
            Image.fromarray(upscale(plane, scale), "RGBA").save(out, "PNG")
            if index == 0:
                first = out
        return {"exported": first}

    runners = {"sheet": run_sheet, "gif": run_gif, "pngs": run_pngs}
    # ``start_save`` rather than a bare submit, so a refused key clears the lock
    # this function did not set -- the tab has been locked since the click.
    _start(ctx, tab, f"inker-export:{tab.uid}", runners.get(export.kind, run_gif))


def _submit_write(ctx: Any, tab: InkerDoc, key: str, path: Path, file_format: str) -> None:
    doc = tab.doc
    # A floating buffer lives in no layer, and the encoders walk the layer
    # stack -- so without this the file omits the pasted pixels while the
    # canvas still shows them, and ``mark_saved`` then calls the document
    # clean. Closing the tab discarded the paste with no prompt.
    doc.commit_floating()
    rev = doc.history.head

    def run() -> dict[str, Any]:
        _write(doc, path, file_format)
        return {"path": path, "rev": rev, "format": file_format}

    _start(ctx, tab, key, run)


def _write(doc: Any, path: Path, file_format: str) -> None:
    """Blocking; task thread only.

    Encodes the *live* document rather than a copy of it. That is safe because
    the encoders only read, and the frame thread only ever appends to a layer's
    pixels in place -- so the worst case is that the file catches a stroke that
    was mid-flight, which is precisely why the revision is captured before the
    submit rather than after the write.
    """
    from . import inker

    if file_format == "ora":
        inker.write_ora(doc, path)
    else:
        path.write_bytes(doc.png_bytes())


# One rule for all four document modes: see :func:`docmodes.start_save` for why
# a refused submit has to clear the flag.
_start = docmodes.start_save


def _save_linked(ctx: Any, tab: InkerDoc) -> None:
    """Write both halves of a reference edit: the flat PNG, then the layers.

    The flat write goes through the untouched ``save_edited_image``, so the
    original backup, the ``hand_edited`` param, the reference re-measure and
    the staged replace all still happen exactly as they did for the old inline
    editor.

    The order is the whole correctness argument, and it is the opposite of what
    it looks like. Freshness is ``paint.ora`` being *newer* than ``input.png``,
    so writing the layers first guarantees the reference is newer than them the
    moment the save completes -- every save would mark its own layers stale and
    the next open would flatten them away. Writing the flat half first leaves
    the sidecar newer on success, and on a crash between the two leaves the
    layers older than the reference, which is exactly the stale verdict a
    half-finished save deserves.
    """
    from ..service import files as svc_files

    doc, job_id = tab.doc, tab.job_id
    # Committed *before* the head is read: the commit pushes a step of its own,
    # so recording the head first saves a document against a head one behind
    # it -- and dirty, being a comparison against that head, then stays true
    # forever however many times the user saves.
    doc.commit_floating()
    rev = doc.history.head

    def run() -> dict[str, Any]:
        from . import inker

        svc_files.save_edited_image(ctx.svc, job_id, doc.png_bytes())
        svc_files.save_inker_working(ctx.svc, job_id, inker.ora_bytes(doc))
        return {"rev": rev, "job_id": job_id, "linked": True}

    _start(ctx, tab, f"inker-save:{tab.uid}", run)


# --- the other direction: Inker -> the pipeline ------------------------------


def save_as_reference(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Mint a new reference job from what is on the canvas, and link to it."""
    from ..service import jobs as svc_jobs

    tab = tab or active(ctx)
    if tab is None or tab.saving or tab.linked:
        return
    doc, title = tab.doc, tab.title
    doc.commit_floating()  # before the head is read; see _save_linked
    rev = doc.history.head

    def run() -> dict[str, Any]:
        result = svc_jobs.import_reference(ctx.svc, doc.png_bytes(), name=title)
        job_id = result["id"]
        from . import inker

        # Linked immediately, so the next Ctrl+S saves in place rather than
        # minting a second job from the same pixels.
        svc_files_save(ctx, job_id, inker.ora_bytes(doc))
        return {"rev": rev, "job_id": job_id, "link": True}

    _start(ctx, tab, f"inker-save:{tab.uid}", run)


def svc_files_save(ctx: Any, job_id: str, data: bytes) -> None:
    from ..service import files as svc_files

    svc_files.save_inker_working(ctx.svc, job_id, data)


def send_to_3d(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Take the flattened canvas into the mesh stage.

    A linked document promotes the reference it already is; an unlinked one
    becomes an ordinary image job, which is the same call the 3D pane's upload
    button makes.
    """
    from ..service import jobs as svc_jobs

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    doc.commit_floating()
    if tab.linked:
        if tab.dirty:
            ctx.toast("Save first, so the mesh is made from what you see.", "error")
            return
        _promote(ctx, tab.job_id)
        return

    def run() -> dict[str, Any]:
        return svc_jobs.create_job(
            ctx.svc, kind="image", output="model", image=doc.png_bytes()
        )

    # Gated like every other encode: ``png_bytes`` walks the layer stack on a
    # task thread, and an undo, a crop or a rotate landing mid-walk restructures
    # it underneath. Keyed by tab so ``on_task_done``/``on_task_failed`` can
    # find the document to unlock, the same shape the saves use.
    _start(ctx, tab, f"inker-send:{tab.uid}", run)
    if tab.saving:
        ctx.toast("Queued a mesh from the drawn image.")


def _promote(ctx: Any, job_id: str) -> None:
    from ..service import errors as svc_errors
    from ..service import jobs as svc_jobs

    def run(force: bool = False) -> dict[str, Any]:
        return svc_jobs.promote_to_model(ctx.svc, job_id, force=force)

    def go(force: bool) -> None:
        if ctx.submit(f"inker-promote:{job_id}", run, force):
            ctx.toast("Queued a mesh from this reference.")

    # The quality gate is a heuristic about composition, not a fact, so it is
    # offered as a confirm rather than a refusal -- the same bargain the 3D
    # pane strikes.
    try:
        report = (ctx.svc.require_job(job_id).get("params") or {}).get("reference_report") or {}
    except svc_errors.ServiceError:
        report = {}
    if report.get("ok") is False:
        reasons = " ".join(report.get("reasons") or ["This image may not reconstruct well."])
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Make a mesh anyway?",
                message=reasons,
                confirm_label="Make it anyway",
                cancel_label="Keep drawing",
                on_confirm=lambda: go(True),
            )
        )
        return
    go(False)


def revert(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Put the generated image back, and drop the layers that described the
    edit -- they are about pixels that will no longer exist."""
    from ..service import files as svc_files

    tab = tab or active(ctx)
    if tab is None or not tab.linked or not tab.has_original or tab.saving:
        return
    job_id = tab.job_id

    def run() -> dict[str, Any]:
        svc_files.revert_reference(ctx.svc, job_id)
        svc_files.discard_inker_working(ctx.svc, job_id)
        return {"reverted": True, "job_id": job_id}

    def go() -> None:
        _start(ctx, tab, f"inker-revert:{tab.uid}", run)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Revert to the original?",
            # Names the *layers* rather than "every edit" (S139). The two are
            # not the same claim: what ``run`` deletes is ``paint.ora``, so a
            # revert loses the layered document itself and not merely the
            # flattened pixels -- reopening this asset afterwards gives back one
            # layer. Stated before the button, the retarget panel's rule.
            message=(
                "The generated image comes back. The layered document is deleted "
                "with it, so reopening this asset gives back a single flat layer."
            ),
            confirm_label="Revert",
            cancel_label="Keep editing",
            on_confirm=go,
        )
    )


# --- task results -----------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from App._on_task_done for every ``inker-`` key."""
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name in ("inker-open",):
        if isinstance(result, dict):
            _adopt(
                ctx,
                state,
                result["doc"],
                path=result.get("path"),
                title=result.get("title"),
                file_format=result.get("format", "png"),
                job_id=result.get("job_id", ""),
                link_kind=result.get("link_kind", ""),
                has_original=bool(result.get("has_original")),
                saved_head=result.get("saved_head"),
            )
            set_mode(ctx.state, "inker")
            if result.get("matte_requested") and not result.get("matte_applied"):
                # "Fix matte" is the one command whose entire content is the
                # cutout, so the swallow in ``_cut_matte`` -- correct for Edit,
                # where the matte is a bonus -- left this menu item opening a
                # tab indistinguishable from the ordinary one and saying
                # nothing. ``action="log"`` because the exception is already
                # there and the log is the only place the reason lives.
                ctx.toast(
                    "Could not apply the cutout; the reference is open as it was.",
                    "warn",
                    action="log",
                )
        return

    if name == "inker-sheetin":
        # The picture only. The grid comes from the popup the bridge panel
        # opens on the next frame, which is why nothing is adopted here.
        if isinstance(result, dict):
            state.sheet_import = (result["atlas"], result.get("title") or "Sheet")
            state.sheet_import_open = False
            set_mode(ctx.state, "inker")
        return

    if name == "inker-recover":
        if isinstance(result, dict):
            tab = _adopt(
                ctx,
                state,
                result["doc"],
                path=None,
                title=result.get("title"),
                file_format="ora",
            )
            # Dirty from the moment it opens, and it owns the file it came
            # from: saving or closing it is what clears the crash copy, and
            # until one of those happens the copy has to stay.
            tab.saved_head = -1
            tab.journal_name = Path(result["autosave"]).name
            set_mode(ctx.state, "inker")
        return

    if name == "inker-autosave":
        return

    if name == "inker-palette":
        # A list of colours, or None for a cancelled picker. Appended rather
        # than replacing: an import is "add these to what I have", and a user
        # who wanted the old ones gone can right-click them away -- where an
        # import that silently wiped a session's palette has no way back.
        if result:
            for colour in result:
                state.add_swatch(colour)
            persist(ctx)
            ctx.toast(f"Added {len(result)} colour(s).", "success")
        return

    if name == "inker-index":
        # The picker came back with a table for a *document*. Resolved through
        # the uid rather than through ``active``: a native picker is unbounded,
        # and the user may well have switched tabs while it was up -- indexing
        # whichever document happens to be in front now would rewrite the wrong
        # file's pixels.
        if result:
            index_to(ctx, state.get(key.split(":", 1)[1]), result)
        return

    if name in ("inker-send", "inker-promote"):
        ctx.cache.invalidate()
        # ``inker-send`` locks its tab while the flatten runs off-thread;
        # ``inker-promote`` has no tab of its own to unlock.
        sent = state.get(key.split(":", 1)[1]) if name == "inker-send" and ":" in key else None
        if sent is not None:
            sent.saving = False
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        ctx.cache.invalidate()
        return
    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported"):
        ctx.toast(f"Exported to {result['exported']}")
        return
    if result.get("reverted"):
        _reload_linked(ctx, tab)
        return

    tab.mark_saved(result.get("rev"))
    # The document is on disk under a name the user chose, so the crash copy is
    # describing work that is no longer at risk. Dropped here rather than on a
    # timer: an autosave that outlived its document is exactly the file that
    # gets offered back after a clean session and confuses somebody.
    drop_autosave(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = inker_state.title_for(tab.path)
        tab.file_format = result.get("format", "ora")
        remember_path(ctx, tab.path)
        persist(ctx)
    linked_just_now = bool(result.get("link"))
    if linked_just_now:
        tab.job_id = result["job_id"]
        tab.link_kind = "reference-edit"
    if tab.linked:
        # "Save as reference" mints a brand-new reference from the drawn
        # pixels, so there is no input.orig.png behind it and Revert has
        # nothing to revert to. The unconditional True here overwrote the
        # False set a line earlier and offered a button that could only fail.
        # Every *other* linked save goes through save_edited_image, which
        # writes the backup.
        tab.has_original = not linked_just_now
        ctx.cache.invalidate()
        _nudge_viewer(ctx, tab)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked.

    ``saving`` disables every editing control, so without this a single failed
    write makes the tab permanently read-only with no way back short of
    closing it.
    """
    state = ctx.state.inker
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


def _reload_linked(ctx: Any, tab: InkerDoc) -> None:
    """Re-decode a linked document after a revert replaced its file."""
    from . import inker

    if tab.path is None:
        return
    try:
        tab.doc = inker.Document.load(tab.path)
    except Exception as exc:
        ctx.toast(f"Reverted, but the image could not be reopened ({exc}).", "error")
        return
    tab.saved_head = tab.doc.history.head
    tab.has_original = False
    tab.view.fitted = False
    ctx.cache.invalidate()
    ctx.toast("Back to the original image.")
    _nudge_viewer(ctx, tab)


def _nudge_viewer(ctx: Any, tab: InkerDoc) -> None:
    """``_sync_viewer`` short-circuits when the path has not changed, so an
    in-place rewrite of input.png would otherwise leave 2D mode showing the
    texture from before the edit."""
    viewer = ctx.viewer
    if viewer is not None and tab.path is not None and viewer.path == tab.path:
        viewer.clear()
        viewer.load_reference(tab.path)


# --- closing and guarding ---------------------------------------------------


def request_close(ctx: Any, tab: InkerDoc) -> None:
    state = ensure(ctx)

    def go() -> None:
        from .panes import inker_textures

        inker_textures.release_doc(ctx, tab.uid)
        drop_autosave(ctx, tab)
        state.close(tab.uid)

    if not tab.dirty:
        go()
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Close without saving?",
            message=f"The changes to {tab.title} will be lost.",
            confirm_label="Close",
            cancel_label="Keep editing",
            on_confirm=go,
        )
    )


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved pixels. -> whether it went ahead now.

    One question for all of them: ``ConfirmQueue`` holds a single pending
    question, so asking per dirty document would silently drop all but the
    first. Only quitting and closing a tab are destructive -- switching modes
    and going Home are not, because Paint is a mode rather than a takeover and
    its tabs are still there when you come back.
    """
    return docmodes.guard(ctx, "inker", "drawing", "drawings", verb, proceed)


# --- free transform ---------------------------------------------------------


def begin_transform(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Ctrl+T. Lifts the selection (or the whole layer) and goes modal."""
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or tab.saving or state.transforming:
        return
    if tab.doc.begin_transform():
        state.transforming = True
        state.clear_drag()


def end_transform(ctx: Any, *, commit: bool) -> None:
    state = ensure(ctx)
    tab = state.active
    state.transforming = False
    state.clear_drag()
    if tab is None:
        return
    if commit:
        tab.doc.commit_floating()
    else:
        tab.doc.cancel_floating()


# --- the OS clipboard -------------------------------------------------------


def paste_from_os(ctx: Any, tab: InkerDoc | None = None) -> bool:
    """Try the system clipboard, then fall back to our own.

    Ours carries a mask and the OS's cannot, which is why the two are separate
    and why this only *seeds* ours: everything downstream of a paste expects to
    know which pixels were selected.

    ``grabclipboard`` is a few milliseconds for a screenshot-sized image and is
    only reached on an explicit Ctrl+V, so it stays on the frame thread rather
    than earning a task key it would spend most of its life idle in.
    """
    tab = tab or active(ctx)
    if tab is None:
        return False
    try:
        import numpy as np
        from PIL import ImageGrab

        grabbed = ImageGrab.grabclipboard()
    except Exception:
        # Unsupported platform, no display, or a clipboard holding something
        # that is not an image. None of those is an error worth a toast.
        log.debug("no image on the system clipboard", exc_info=True)
        return False
    if grabbed is None or isinstance(grabbed, list):
        # A list is a *file* copy rather than an image; opening one would be a
        # silent second way to open files, with none of the checks open_path
        # does.
        return False
    tab.doc.put_clipboard(np.asarray(grabbed.convert("RGBA"), dtype=np.uint8).copy())
    return True


# --- keys -------------------------------------------------------------------

# Aseprite's letters where they exist, because that is the muscle memory a user
# arrives with. Held here rather than in the pane so the mapping is testable.
TOOL_KEYS = {
    "b": "brush",
    "e": "eraser",
    "g": "fill",
    "u": "gradient",
    "r": "blur",
    "n": "smudge",
    "p": "line",
    "k": "rect",
    "j": "ellipse",
    "m": "select",
    "s": "select_ellipse",
    "q": "lasso",
    "w": "wand",
    "v": "move",
    "i": "eyedropper",
}


# --- playback ----------------------------------------------------------------

#: A tick longer than this is treated as a stall rather than as elapsed time.
#: Without the clamp, a two-second hitch (a dialog, a texture upload storm)
#: fast-forwards the clip through twenty frames at once, which reads as a
#: glitch rather than as the catch-up it technically is.
MAX_TICK_MS = 250.0


def toggle_play(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Start or stop playback. Refused while a save is encoding."""
    tab = tab or active(ctx)
    if tab is None or tab.doc.anim is None or tab.saving:
        return
    if tab.playing:
        stop_play(tab)
        return
    # The float is committed *before* playback rather than when it ends: while
    # playing, the canvas draws cached frame flattens, and a floating buffer is
    # in no layer and therefore in no flatten -- it would simply vanish for the
    # duration and reappear at the end.
    tab.doc.commit_floating()
    tab.playing = True
    tab.play_index = tab.doc.anim.current
    tab.play_accum_ms = 0.0
    # Every play starts on the outward leg. A ping-pong stopped halfway back and
    # resumed would otherwise carry on inwards from wherever it was, which reads
    # as the clip playing backwards for no reason anyone watching can see.
    tab.play_forward = True
    # Every play starts the repeat count over. A tag set to play three times
    # and stopped halfway must play three times again when it is started, not
    # remember that it already finished once -- the same argument
    # ``play_forward`` above makes about a ping-pong's leg.
    tab.play_cycles = 0


def stop_play(tab: InkerDoc) -> None:
    """Stop, and leave the playhead where the eye last saw it."""
    if not tab.playing:
        return
    tab.playing = False
    tab.doc.set_current_frame(tab.play_index)
    tab.play_accum_ms = 0.0


def tick_playback(tab: InkerDoc, dt_ms: float) -> None:
    """One frame's worth of time.

    Deliberately does *not* call ``set_current_frame``: that re-materialises the
    stack and recomposites the whole canvas, sixty times a second, to show a
    picture the frame cache already has. The playhead on the document stays put
    and the canvas draws ``play_index``'s cached flatten instead; stopping is
    the one moment the document catches up.
    """
    anim = tab.doc.anim
    if not tab.playing or anim is None:
        return
    durations = [frame.duration_ms for frame in anim.frames]
    index, accum, playing, forward, cycles = animation.advance(
        durations,
        tab.play_index,
        tab.play_accum_ms,
        min(float(dt_ms), MAX_TICK_MS),
        anim.loop_range(tab.play_index),
        direction=anim.play_direction(tab.play_index),
        forward=tab.play_forward,
        repeat=anim.play_repeat(tab.play_index),
        cycles=tab.play_cycles,
    )
    tab.play_index, tab.play_accum_ms, tab.play_forward = index, accum, forward
    tab.play_cycles = cycles
    if not playing:
        stop_play(tab)


# --- the preview pane's second playhead --------------------------------------

#: Bounds on the preview's speed multiplier. A ceiling because past ×4 a clip
#: is a flicker rather than a preview, and a floor because a multiplier that
#: reaches zero is a stopped clip pretending to play.
MIN_PREVIEW_SPEED = 0.25
MAX_PREVIEW_SPEED = 4.0

#: What the preview can play: the whole timeline, or the tag under its own
#: index. Per-tab preview state rather than a document playback mode -- see the
#: divergence list.
PREVIEW_SCOPES = ("clip", "tag")


def toggle_preview(tab: InkerDoc) -> None:
    """Start or stop the preview. Refused for nothing at all.

    Deliberately not gated on ``busy``: the preview neither edits the document
    nor moves its playhead, so there is nothing for a save or for canvas
    playback to be protected from -- and being able to watch the clip while
    drawing on it is the whole feature.
    """
    if tab.doc.anim is None:
        return
    if tab.preview_playing:
        tab.preview_playing = False
        return
    tab.preview_playing = True
    tab.preview_accum_ms = 0.0
    tab.preview_forward = True
    tab.preview_cycles = 0


def tick_preview(tab: InkerDoc, dt_ms: float) -> None:
    """One frame's worth of time for the *preview*'s playhead.

    A clone of :func:`tick_playback` rather than a share, and the difference is
    the point: this one **never touches ``tab.playing`` or ``tab.saving``** and
    never calls ``set_current_frame``. It reads ``anim`` and writes four fields
    on the tab, so a preview running while the user paints needs no gating
    change anywhere -- the canvas draws the document, the preview draws
    ``frame_flat``, which is the same read onion skinning already makes and is
    safe even during a save (``sheetout.snapshot``'s argument).

    The speed multiplier scales time **after** the stall clamp, so a two-second
    hitch is still treated as a stall at ×4 rather than as eight seconds of
    animation.
    """
    anim = tab.doc.anim
    if not tab.preview_playing or anim is None or not anim.frames:
        return
    last = len(anim.frames) - 1
    index = max(0, min(int(tab.preview_index), last))
    if tab.preview_scope == "tag":
        span = anim.loop_range(index)
        direction = anim.play_direction(index)
        repeat = anim.play_repeat(index)
    else:
        # The whole clip, looping, whatever tags happen to cover it. A preview
        # scoped to the clip that stopped at a non-looping tag's end would be
        # answering a question the scope switch just said no to.
        span, direction, repeat = (0, last, True), "forward", 0
    speed = max(MIN_PREVIEW_SPEED, min(float(tab.preview_speed), MAX_PREVIEW_SPEED))
    index, accum, playing, forward, cycles = animation.advance(
        [frame.duration_ms for frame in anim.frames],
        index,
        tab.preview_accum_ms,
        min(float(dt_ms), MAX_TICK_MS) * speed,
        span,
        direction=direction,
        forward=tab.preview_forward,
        repeat=repeat,
        cycles=tab.preview_cycles,
    )
    tab.preview_index, tab.preview_accum_ms = index, accum
    tab.preview_forward, tab.preview_cycles = forward, cycles
    if not playing:
        tab.preview_playing = False


def step_frame(ctx: Any, delta: int, tab: InkerDoc | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.doc.anim is None or tab.busy:
        return
    anim = tab.doc.anim
    tab.doc.set_current_frame((anim.current + delta) % len(anim.frames))


def animate(ctx: Any, tab: InkerDoc | None = None) -> None:
    """The entry point: turn a still document into a two-frame animation."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    tab.doc.add_frame()


def handle_key(ctx: Any, event: Any) -> bool:
    """Inker's shortcuts. -> whether the key was consumed.

    Consumed unconditionally while a document is open, exactly as the old
    inline editor did: F, W and S would otherwise frame and wireframe a
    viewport that is not on screen.
    """
    import pygame

    state = ctx.state.inker
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.key == pygame.K_SPACE:
        # Seen on both edges: space-to-pan is a hold, not a toggle.
        state.space_held = event.type == pygame.KEYDOWN
        return True
    if event.type != pygame.KEYDOWN:
        return True

    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key)

    if state.transforming:
        # Modal: Enter applies, Escape cancels, and nothing else may change the
        # tool out from under a half-finished transform.
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            end_transform(ctx, commit=True)
        elif event.key == pygame.K_ESCAPE or (ctrl and name == "z"):
            # Ctrl+Z during a transform means "undo the transform", which is
            # cancelling it -- not stepping back through the history behind it.
            end_transform(ctx, commit=False)
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, event, shift=shift)

    if name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif name == "x":
        state.swap_colours()
    elif event.key == pygame.K_LEFTBRACKET:
        if shift:
            state.hardness = max(0.0, state.hardness - 0.05)
        else:
            state.brush_size = inker_state.step_size(state.brush_size, -1)
    elif event.key == pygame.K_RIGHTBRACKET:
        if shift:
            state.hardness = min(1.0, state.hardness + 0.05)
        else:
            state.brush_size = inker_state.step_size(state.brush_size, +1)
    elif doc.anim is not None and event.key in (pygame.K_COMMA, pygame.K_PERIOD):
        # Matched on the key constant, not on ``pygame.key.name``: the spelling
        # of a punctuation key comes from SDL and has changed between versions.
        step_frame(ctx, -1 if event.key == pygame.K_COMMA else 1, tab)
    elif doc.anim is not None and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        # Reachable only past the transform branch above, which consumes Enter
        # first and returns -- applying a half-finished transform must not be
        # ambiguous with starting playback.
        toggle_play(ctx, tab)
    elif event.key == pygame.K_DELETE:
        if not tab.busy:
            doc.delete_selection()
    elif event.key == pygame.K_ESCAPE:
        # Never leaves the mode: Esc means "drop what I am doing", and losing a
        # workspace full of tabs to a stray keypress is not that.
        if tab.playing:
            stop_play(tab)
        elif not tab.saving:
            if doc.floating is not None:
                doc.cancel_floating()
            elif doc.mask is not None:
                doc.deselect()
        # Always: abandoning a half-finished drag is safe mid-save, because it
        # touches the pane's own state and never the document.
        state.clear_drag()
    return True


# Ctrl-shortcuts that change the document. A save encodes the *live* document
# on a task thread; that is safe only for a stroke landing mid-write, because
# pixels are written in place. Everything here restructures the layer stack or
# moves the history head the save captured, so it waits for the save the same
# way a brush stroke on the canvas already does.
# ``e`` joins them because plain Ctrl+E now writes the document into the
# library: it flattens the layer stack, which is the same read a save makes and
# is just as wrong to take while one is in flight.
_MUTATING_CTRL = frozenset({"z", "y", "a", "d", "x", "v", "i", "t", "e"})


def _ctrl_key(
    ctx: Any, state: InkerState, tab: InkerDoc, doc: Any, name: str, event, *, shift: bool
):
    import pygame

    # ``busy``, not ``saving``: playback is the second reason the document may
    # not be restructured, and it is the same list of keys for the same reason.
    if tab.busy and name in _MUTATING_CTRL:
        return True

    if name == "z":
        doc.redo() if shift else doc.undo()
    elif name == "y":
        doc.redo()
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "e":
        # Plain Ctrl+E is "put this in the library" in every other document
        # mode -- Clay's ``export_asset``, Plotter's and Packwright's
        # ``export_library`` -- and Shift is the file-on-disk export. Inker had
        # only the Shift half, so the one chord a user carries between the four
        # editors did nothing here.
        #
        # A linked document is already *in* the library (it is somebody's
        # reference, opened for editing), so there is nothing to add and Ctrl+S
        # is the write it wants. Silence rather than a second copy of the asset.
        if shift:
            export_png(ctx, tab)
        elif not tab.linked:
            save_as_reference(ctx, tab)
    elif name == "n":
        new_document(ctx, 1024, 1024)
    elif name == "o":
        ask_open(ctx)
    elif name == "w":
        request_close(ctx, tab)
    elif name == "a":
        doc.select_all()
    elif name == "d":
        doc.deselect()
    elif name == "c":
        doc.copy()
    elif name == "x":
        doc.cut()
    elif name == "v":
        paste_from_os(ctx, tab)
        if shift:
            doc.paste_as_layer()
        else:
            doc.paste()
            state.tool = "move"
    elif name == "i" and shift:
        doc.invert_selection()
    elif name == "t":
        begin_transform(ctx, tab)
    elif event.key == pygame.K_TAB:
        state.cycle(-1 if shift else 1)
    elif name == "0":
        tab.view.fitted = False
    elif name == "1":
        # Applied by the canvas, which is the only thing that knows how big the
        # pane is; a keypress cannot centre on its own.
        tab.view.pending_zoom = 1.0
    elif name == "4":
        # Ctrl+4 / Ctrl+Shift+4 turn the page, Ctrl+5 mirrors it. Both are
        # *view* state -- no pixels move, nothing is pushed, nothing is saved --
        # which is why they sit here beside the zoom keys rather than among the
        # mutating shortcuts.
        inker_state.rotate_view(tab.view, -1 if shift else 1)
    elif name == "5":
        inker_state.flip_view(tab.view)
    return True


def release_all(ctx: Any) -> None:
    from .panes import inker_textures

    inker_textures.release_all(ctx)


# --- palette files ----------------------------------------------------------
#
# The picker and the file are both blocking, so both go to a task thread -- the
# rule every dialog and every encode in this module follows. The *bytes* for an
# export are built on the frame thread, for ``save_as``'s reason: they read live
# state, and doing that after an unbounded modal would write whatever the user
# changed while it was open.

GPL_FILTER = ["GIMP palette (*.gpl)", "*.gpl"]


def import_palette(ctx: Any) -> None:
    from .inker import gpl

    ensure(ctx)

    def run() -> list[tuple[int, int, int, int]] | None:
        path = dialogs.open_file("Import a palette", GPL_FILTER)
        if path is None:
            return None
        return gpl.parse(path.read_text(encoding="utf-8", errors="replace"))

    ctx.submit("inker-palette", run)


def export_palette(ctx: Any) -> None:
    from .inker import gpl

    state = ensure(ctx)
    text = gpl.dumps(list(state.swatches))

    def run() -> None:
        path = dialogs.save_file("Export the palette", "palette.gpl", GPL_FILTER)
        if path is not None:
            path.with_suffix(".gpl").write_text(text, encoding="utf-8")

    ctx.submit("inker-palette-export", run)


# --- indexed colour -----------------------------------------------------------
#
# The palette belongs to the *document* -- it is saved with the file and it is
# what every write snaps to -- so everything here takes a tab and goes through
# ``Document``. The swatch row above is a different thing and stays one: a
# session's favourite colours, persisted in settings, no bearing on any file.
#
# All of these run **inline on the frame thread**, gated on ``tab.busy``, which
# is exactly what the canvas geometry ops in ``panes/inker_bridge`` do and for
# the same reason: they rebind whole layer planes, so one landing mid-save
# writes an archive whose parts disagree. The cost is the same class as a
# rotate, and ``indexed.snap`` works over the region's *distinct* colours
# rather than its pixels, which is what keeps a 40-frame clip inside a frame.


def index_to(ctx: Any, tab: Any, colours: Any) -> bool:
    """Make *tab* indexed against *colours*, or plain RGBA with ``None``."""
    if tab is None or tab.busy:
        return False
    state = ensure(ctx)
    if not tab.doc.set_palette(colours):
        return False
    state.palette_slot = 0
    state.palette_usage = None
    if colours:
        ctx.toast(f"Indexed to {len(list(colours))} colour(s).", "success")
    else:
        # Worth saying, because nothing on the canvas changes: leaving indexed
        # mode lifts the constraint and repaints nothing.
        ctx.toast("Indexed colour off. The pixels are unchanged.")
    return True


def import_document_palette(ctx: Any) -> None:
    """Open a ``.gpl`` and index the active document to it.

    A second task key from ``import_palette``'s, because they are different
    acts on different subjects -- one adds to the session's swatch row, the
    other rewrites every pixel of a file -- and sharing a key would let the
    landing handler guess wrong about which one came back.
    """
    from .inker import gpl

    ensure(ctx)
    tab = active(ctx)
    if tab is None or tab.busy:
        return

    def run() -> list[tuple[int, int, int, int]] | None:
        path = dialogs.open_file("Index to a palette", GPL_FILTER)
        if path is None:
            return None
        return gpl.parse(path.read_text(encoding="utf-8", errors="replace"))

    ctx.submit(f"inker-index:{tab.uid}", run)


def export_document_palette(ctx: Any) -> None:
    """Write the *document's* table out as a ``.gpl``."""
    tab = active(ctx)
    if tab is None or not tab.doc.palette:
        return
    from .inker import gpl

    text = gpl.dumps(list(tab.doc.palette))
    name = f"{tab.path.stem}.gpl" if tab.path else "palette.gpl"

    def run() -> None:
        path = dialogs.save_file("Export the document palette", name, GPL_FILTER)
        if path is not None:
            path.with_suffix(".gpl").write_text(text, encoding="utf-8")

    ctx.submit("inker-palette-export", run)


# --- crash recovery -----------------------------------------------------------
#
# The mechanism moved to :mod:`studio.journal` (UX-05) and this is what is left:
# the four answers that are about *drawings* rather than about journalling.
# Inker had the only crash-safe autosave in the app and every part of its loop
# was load-bearing; what was wrong was that it was written into one mode. See
# that module for the loop, the debounce, the head gate and the completion
# gate.
#
# The rule it turns on is unchanged and is restated here because it is Inker's
# rule as much as the journal's: **a journal entry is never a save.** It does
# not clear ``dirty``, does not move ``saved_head``, does not retitle the tab
# and does not touch the linked job.


def _journal_slots(ctx: Any) -> list[InkerDoc]:
    """The tabs worth copying: dirty, and not mid-write.

    ``busy`` for the reason every other structural control is gated --
    ``write_ora`` walks the layer stack, and a rotate landing mid-write
    produces an archive whose parts disagree about the canvas size.
    """
    state = getattr(ctx.state, "inker", None)
    if state is None:
        return []
    return [tab for tab in state.docs if tab.dirty and not tab.busy]


def _journal_encode(tab: InkerDoc) -> bytes:
    from .inker import ora

    return ora.ora_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen one recovered ``.ora`` as an *untitled, dirty* document.

    Untitled on purpose: the file it was copied from may still be on disk with
    its own contents, and adopting the path would arm Ctrl+S to overwrite it
    with a document the user has not looked at yet. Dirty for the same reason
    -- there is unsaved work here, and the tab must say so.
    """
    ensure(ctx)
    ctx.submit(f"inker-recover:{abs(hash(str(path)))}", _load_recovery, path, meta)
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="inker",
        ext=".ora",
        label="drawing",
        slots=_journal_slots,
        uid_of=lambda tab: tab.uid,
        title_of=lambda tab: tab.title,
        head_of=lambda tab: tab.doc.history.head,
        encode=_journal_encode,
        adopt=_journal_adopt,
    )
)


def drop_autosave(ctx: Any, tab: InkerDoc) -> None:
    """Forget a tab's crash copy. Kept as a name because nine call sites use
    it and they are all saying "this document is somewhere the user chose"."""
    journal.drop(ctx, tab)


def _load_recovery(path: Path, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Blocking; task thread only."""
    from .inker import Document

    doc = Document.load(Path(path))
    doc.path = None
    title = (meta or {}).get("title") or Path(path).stem.rsplit("-", 1)[0]
    return {
        "doc": doc,
        "title": f"{title} (recovered)",
        "format": "ora",
        "autosave": str(path),
    }
