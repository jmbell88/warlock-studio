"""Every way a drawing gets into Inker, and the decode behind each.

A picker, a path, a pixel array, a sprite sheet, an .aseprite, a sprite draft,
a rendered character sheet, a job's reference image -- eight doors, and each is
a frame-thread half that opens the picker and a task-thread half that decodes.
The split is the rule, not a habit: a native picker is modal to the OS and an
image is as large as anything this app opens.

Lifted out of ``studio/inker_mode`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed, so the
move is code motion over tested behaviour rather than a rewrite.

``inker_mode`` is imported as a *module* and never ``from``-imported: every
attribute is resolved at call time, so this file and its parent may be
imported in either order. The parent serves these names back through a PEP
562 ``__getattr__``, which is what keeps ``inker_mode.export_png`` and the
rest working for every caller and every test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import dialogs, inker_mode, sizeguard
from .inker_state import InkerDoc
from .state import set_mode

log = logging.getLogger(__name__)


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    inker_mode.ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open image", inker_mode.OPEN_FILTER)
        return None if path is None else _load(path)

    ctx.submit("inker-open", run)


def open_path(ctx: Any, path: Path) -> None:
    """Open a known path -- a drop, or a click in the recent list."""
    state = inker_mode.ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one file would race on save.
        state.activate(existing.uid)
        return
    if path.suffix.lower() in inker_mode.ASEPRITE_SUFFIXES:
        # Before the ``inker_mode.OPENABLE`` check rather than inside it: a drop of an
        # Aseprite file is an import, and refusing it with "Inker opens images
        # and .ora files" would be telling the user the app cannot do something
        # it can. Nothing is remembered for it -- the tab owns no file, so a
        # recent entry would offer to reopen a document that is never *this*
        # document again.
        import_aseprite_path(ctx, path)
        return
    if path.suffix.lower() not in inker_mode.OPENABLE:
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

    inker_mode.ensure(ctx)
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
    inker_mode.ensure(ctx)

    def run() -> dict[str, Any] | None:
        from . import pixelguard

        path = dialogs.open_file("Import sprite sheet", inker_mode.OPEN_FILTER)
        if path is None:
            return None
        # Through the same door ``Document.open`` uses. A file picker is the
        # one place an *arbitrary* image reaches this mode -- nothing upstream
        # has bounded it the way ``service.files.to_png`` bounds an upload.
        atlas = pixelguard.decode_rgba(path, Path(path).name)
        return {"atlas": atlas, "title": Path(path).stem, "suggest": _suggest_grid(atlas)}

    ctx.submit("inker-sheetin", run)


def _suggest_grid(atlas: Any) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """``(cell, offset, padding)`` for a sheet with visible separator lines.

    Task thread. The Inker has the same blind-grid problem the Plotter's
    add-tileset door has, and the same detector answers it -- ``tilegrid`` is a
    shared leaf and is deliberately importable from here. What differs is what
    is *done* with the answer: the Plotter recomposes irregular cells onto a
    uniform atlas, and this door cannot, because the popup's model is one cell
    size plus one offset plus one padding and an irregular grid has no such
    spelling. So a detection is used only when its segments really are uniform
    on both axes, and anything else is left to the user exactly as today.

    ``None`` for "nothing to suggest", and the caller must treat that as *keep
    what is there*: the three fields deliberately persist across imports as a
    convenience, so resetting them on a failed detection would throw away a
    number the user typed for the previous sheet.
    """
    from .tilegrid import slicing

    grid = slicing.detect_grid(atlas)
    if grid is None:
        return None
    heights = {end - start + 1 for start, end in grid.rows}
    widths = {end - start + 1 for start, end in grid.cols}
    if len(heights) != 1 or len(widths) != 1:
        return None
    cell = (int(next(iter(widths))), int(next(iter(heights))))
    offset = (int(grid.cols[0][0]), int(grid.rows[0][0]))
    # The gap between one segment's end and the next one's start: the rule the
    # cells are separated by, which is exactly what the popup calls padding.
    pad_x = int(grid.cols[1][0] - grid.cols[0][1] - 1) if len(grid.cols) > 1 else 0
    pad_y = int(grid.rows[1][0] - grid.rows[0][1] - 1) if len(grid.rows) > 1 else 0
    return cell, offset, (max(0, pad_x), max(0, pad_y))


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

    state = inker_mode.ensure(ctx)
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
    inker_mode._adopt(ctx, state, doc, path=None, title=title, file_format="ora")
    set_mode(ctx.state, "inker")
    return True


def ask_import_aseprite(ctx: Any) -> None:
    """Pick an ``.aseprite`` off disk and open what it holds.

    One step where the sheet import is two, and for the reason that one is
    two: a sheet has to be told how to cut, and an Aseprite file already says
    where every layer, cel, frame and tag is. So there is nothing to ask, and
    the result goes straight down the ``inker-open`` road with no routing of
    its own.
    """
    inker_mode.ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Import Aseprite file", inker_mode.ASEPRITE_FILTER)
        return None if path is None else _load_aseprite(Path(path))

    ctx.submit("inker-open:aseprite", run)


def import_aseprite_path(ctx: Any, path: Path) -> None:
    """The same import for a path already in hand -- a drop onto the window."""
    inker_mode.ensure(ctx)
    set_mode(ctx.state, "inker")
    ctx.submit(
        f"inker-open:aseprite:{abs(hash(str(path)))}", _load_aseprite, Path(path)
    )


def _load_aseprite(path: Path) -> dict[str, Any]:
    """Blocking; task thread only.

    ``path=None`` in the result, and that stays true even though the app can
    now *write* this format (``save_as`` -> ``aseout.write_aseprite``): what
    ``path=None`` guarantees is that an import never points at the file it
    came from, not that nothing can write it. Pointing the tab at the source
    would arm the first Ctrl+S to overwrite it -- with ORA bytes if the
    document is untouched, or silently if a WRITABLE_SUFFIXES-widening ever
    added this suffix to it -- and an import is exactly the moment nothing has
    been decided yet. The document is an unsaved ORA -- the format that can
    hold layers, a timeline, links and slices without asking the user to pick
    a destination first -- and the first Ctrl+S asks where to put it, Aseprite
    included if that is what is chosen there.
    """
    from ..service.files import MAX_INKER_BYTES
    from .inker import asein

    path = Path(path)
    doc, warnings = asein.document_from_aseprite(
        sizeguard.within_ceiling(path, MAX_INKER_BYTES).read_bytes()
    )
    return {
        "doc": doc,
        "path": None,
        "format": "ora",
        "title": path.stem,
        "warnings": warnings,
    }


def _report_import_warnings(ctx: Any, warnings: Any) -> None:
    """Say what an import dropped: one toast, every line in the log.

    A toast per warning would be a stack of them for one file, and the reader
    only ever wanted to know *whether* something was lost and roughly what --
    so the first line goes on screen with a count beside it and ``action="log"``
    puts the rest one click away. Silence is the wrong alternative: an Aseprite
    file that quietly lost its per-cel opacities is one the user finds out
    about by noticing the drawing looks wrong.
    """
    lines = [str(line) for line in (warnings or [])]
    if not lines:
        return
    for line in lines:
        log.warning("aseprite import: %s", line)
    head = lines[0].rstrip(".")
    text = head if len(lines) == 1 else f"{head} (+{len(lines) - 1} more)"
    ctx.toast(f"Imported: {text}.", "warn", action="log")


def duplicate_document(ctx: Any, tab: InkerDoc | None = None) -> bool:
    """Aseprite's *Duplicate Sprite*: the whole document, as a second tab.

    Through ``open_pixels`` for ``new_from_selection``'s reason -- the adoption
    is the one that already exists -- which makes this a **flattened** copy
    rather than a layered one, and the tab's title says so. A layered duplicate
    would have to deep-copy every cel, every group, the palette, the tilesets
    and the history budget, and the honest name for that is *save-as*.
    """
    state = inker_mode.ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return False
    open_pixels(ctx, tab.doc.flatten(matte=False), title=f"{tab.title} copy")
    return True


def new_from_selection(ctx: Any, tab: InkerDoc | None = None) -> bool:
    """Aseprite's *New Sprite From Selection*. -> whether one was made.

    Through ``open_pixels``, which is the door every other "here are some
    pixels, make a document of them" path already uses (a sheet import, a
    sprite draft, a rendered sheet) -- so the new tab is adopted, titled,
    journalled and made active by the same code, and this function is the
    crop and nothing else.
    """
    state = inker_mode.ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return False
    pixels = tab.doc.selection_pixels()
    if pixels is None or not pixels.size:
        state.say("Select something first -- this makes a document of it.")
        return False
    open_pixels(ctx, pixels, title=f"{tab.title} crop")
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
    inker_mode.ensure(ctx)
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
    from ..service import sprites as svc_sprites
    from . import pixelguard
    from .inker import sheetin

    record = svc_sprites.get_sprite_draft(svc, job_id, draft_id)
    png = svc_sprites.sprite_draft_png(svc, job_id, draft_id, candidate)
    atlas = pixelguard.decode_rgba(png, png.name)
    doc = sheetin.document_from_atlas(
        atlas, record["cells"], str(record.get("sheet_type") or "")
    )
    return {
        "doc": doc,
        "path": None,
        "format": "ora",
        "title": f"{record.get('sheet_type', 'sprite')} {draft_id[:6]}{candidate}",
    }


def open_rendered_sheet(
    ctx: Any, job_id: str, sheet_id: str, *, pixel: bool = False
) -> None:
    """Open a rendered 8-direction sheet (or its pixel restyle) in the Inker.

    :func:`open_sprite_draft`'s shape, and unlinked for exactly its reason: it
    carries no ``job_id`` and no ``link_kind``, so the first Ctrl+S is a Save
    As. The sheet on disk is where the document *came from*, not its file --
    saving over the render would destroy the thing the document is derived from.

    **Sliced by** :func:`sheetin.document_from_grid`, **not**
    ``document_from_atlas``. The render's sidecar fully describes a uniform grid
    -- columns, rows, frame size, cells -- but it is not a
    ``DirectionalLayout`` *kind*: ``animation.SHEET_KINDS`` names the two legacy
    sprite atlases and the planned ``f"{action}{directions}"`` sheets, and a
    Blender render is none of them, so ``DirectionalLayout.of()`` would return
    ``None`` and the whole door would refuse. The grid slicer wants only the
    geometry, which the sidecar has.
    """
    inker_mode.ensure(ctx)
    set_mode(ctx.state, "inker")
    ctx.submit(
        f"inker-open:sheet:{sheet_id}:{'pixel' if pixel else 'render'}",
        _load_rendered_sheet,
        ctx.svc,
        job_id,
        sheet_id,
        pixel,
    )


def open_pixel_artifact(
    ctx: Any,
    job_id: str,
    name: str,
    *,
    title: str,
    pixel_colors: int,
    pixel_palette: str | None,
    pixel_dither: bool,
) -> None:
    """Open a job's derived ``pixel_NNN.png`` as an ordinary document.

    :func:`open_sprite_draft`'s shape, and **unlinked for its reason**: no
    ``job_id`` and no ``link_kind``, so the first Ctrl+S is a Save As.
    ``open_job_reference`` would be wrong here rather than merely different --
    it makes a linked tab whose Ctrl+S writes back, and this artifact is
    *derived*: ``derive.get_file`` rebuilds it whenever the knobs say the copy
    on disk is stale, so the edit would be silently thrown away.

    The three pixel preferences are passed in rather than read here, because
    this body runs on the task thread and they live in settings on the frame
    thread -- the rule ``Ctx.save_artifact`` states. It is also what makes the
    preview, the export and this open describe one file.

    ``get_file`` derives the artifact if it is absent, so this works before
    "Preview pixels" has ever been pressed.
    """
    inker_mode.ensure(ctx)
    set_mode(ctx.state, "inker")

    def run() -> dict[str, Any]:
        """Blocking; task thread only."""
        from ..service import derive as svc_derive
        from . import inker, pixelguard

        path = svc_derive.get_file(
            ctx.svc,
            job_id,
            name,
            pixel_colors=pixel_colors,
            pixel_palette=pixel_palette,
            pixel_dither=pixel_dither,
        )
        array = pixelguard.decode_rgba(path, Path(path).name)
        return {"doc": inker.Document.from_pixels(array, name="Pixels"), "title": title}

    # The ``inker-open`` prefix is what makes ``on_task_done`` adopt this with
    # no routing change; the ``pixel:`` segment stops it colliding with
    # ``open_job_reference``'s key for a different document of the same job.
    ctx.submit(f"inker-open:pixel:{job_id}:{name}", run)


def sheet_grid(record: dict[str, Any]) -> tuple[tuple[int, int], int]:
    """``(cell, count)`` from a rendered sheet's sidecar.

    ``frame_size`` is the square case and is written as **0** on a non-square
    plan, where ``frame_w``/``frame_h`` carry the truth -- so this reads the
    pair first and falls back to the square. ``cells`` is the authority on how
    many there are; ``columns * rows`` would count a padded final row.
    """
    width = int(record.get("frame_w") or 0)
    height = int(record.get("frame_h") or 0)
    if not width or not height:
        square = int(record.get("frame_size") or 0)
        width = height = square
    if width < 1 or height < 1:
        raise ValueError("that sheet's sidecar records no frame size")
    count = len(record.get("cells") or []) or int(
        int(record.get("columns") or 0) * int(record.get("rows") or 0)
    )
    if count < 1:
        raise ValueError("that sheet's sidecar records no cells")
    return (width, height), count


def _load_rendered_sheet(
    svc: Any, job_id: str, sheet_id: str, pixel: bool
) -> dict[str, Any]:
    """Blocking; task thread only."""
    from ..service import sheets as svc_sheets
    from . import pixelguard
    from .inker import sheetin

    if pixel:
        record = svc_sheets.get_pixel_sheet(svc, job_id, sheet_id)
        png = svc_sheets.sheet_pixel_png(svc, job_id, sheet_id)
    else:
        record = svc_sheets.get_sheet(svc, job_id, sheet_id)
        png = svc_sheets.sheet_png(svc, job_id, sheet_id)
    cell, count = sheet_grid(record)
    atlas = pixelguard.decode_rgba(png, png.name)
    cells = record.get("cells") or []
    animation = record.get("animation")
    if cells and animation:
        # The sidecar knows what its own frames mean, so the document opens
        # with them: one tag per animation and direction, and each frame at its
        # own ``duration_ms`` -- a six-per-second idle and a twelve-per-second
        # run in one timeline, which is the whole reason those were written
        # into the sidecar.
        #
        # ``document_from_sheet`` was added for exactly this and then never
        # called: this line said ``document_from_grid``, which is geometry
        # only, so a Troupe sheet opened as 256 untagged frames at one default
        # duration -- while the button's tooltip and the manual both promised
        # the tags. ``document_from_grid`` stays as the fallback for the sheets
        # that have no ``animation`` block: every one written before
        # ``charsheet.animation_block`` existed.
        # ``source`` is what lets the document remember which sheet these
        # pixels came from, so a re-render of the same character can be found
        # and merged later. Only on this branch: the fallback below has no
        # ``animation`` block and therefore no run vocabulary to re-render in.
        doc = sheetin.document_from_sheet(
            atlas, cells, animation, source={"job": job_id, "sheet": sheet_id}
        )
    else:
        doc = sheetin.document_from_grid(atlas, cell, count=count)
    name = str(record.get("name") or sheet_id)
    return {
        "doc": doc,
        "path": None,
        "format": "ora",
        "title": f"{name} (pixel)" if pixel else name,
    }


def newest_sheet_after(svc: Any, job_id: str, sheet_id: str) -> str:
    """The most recent character sheet of one job newer than ``sheet_id``.

    What "the re-render" means without asking: a re-render publishes a new
    sheet in the same job's directory, so the newest one that is not the
    document's own base is the one to merge. Returns "" when there is none.

    Older sheets are deliberately not offered. Merging one would run the
    three-way comparison backwards -- the "incoming" render would be the older
    picture -- and every cell the user has since had re-rendered would read as
    a conflict.
    """
    from .. import rigging

    try:
        records = rigging.list_sheets(svc.job_dir(job_id))
    except OSError:
        return ""
    current = next((r for r in records if str(r.get("id") or "") == sheet_id), None)
    since = float((current or {}).get("created") or 0.0)
    newer = [
        record
        for record in records
        if (record.get("animation") or {}).get("tags")
        and str(record.get("id") or "") != sheet_id
        and float(record.get("created") or 0.0) > since
    ]
    newer.sort(key=lambda r: float(r.get("created") or 0.0), reverse=True)
    return str(newer[0].get("id") or "") if newer else ""


def load_sheet_cells(svc: Any, job_id: str, sheet_id: str) -> list[Any]:
    """One re-rendered sheet's cells, in frame order. Blocking; task thread only.

    ``_load_rendered_sheet``'s first half, for the merge -- which wants the
    pixels and not a document. Refused here rather than inside the funnel when
    the geometry disagrees, so the sentence names the *sheet* the user picked
    instead of surfacing from three layers down.
    """
    from ..service import sheets as svc_sheets
    from . import pixelguard

    record = svc_sheets.get_sheet(svc, job_id, sheet_id)
    png = svc_sheets.sheet_png(svc, job_id, sheet_id)
    atlas = pixelguard.decode_rgba(png, png.name)
    cells = record.get("cells") or []
    if not cells:
        raise ValueError("that sheet's sidecar lists no cells")
    out = []
    for cell in cells:
        x, y = int(cell.get("x", 0)), int(cell.get("y", 0))
        w, h = int(cell.get("w", 0)), int(cell.get("h", 0))
        if w <= 0 or h <= 0 or y + h > atlas.shape[0] or x + w > atlas.shape[1]:
            raise ValueError("that sheet's cells do not fit its own atlas")
        out.append(atlas[y : y + h, x : x + w].copy())
    return out


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
    state = inker_mode.ensure(ctx)
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
