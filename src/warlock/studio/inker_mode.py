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
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pipelines import sheet as sheetlib
from . import dialogs, docmodes, filetypes, fonts, inker_ops, inker_state, journal, recents
from .inker import animation
from .inker.asein import ASEPRITE_SUFFIXES
from .inker_state import InkerDoc, InkerState
from .state import set_mode

log = logging.getLogger(__name__)

ORA_FILTER = ["OpenRaster (*.ora)", "*.ora"]
PNG_FILTER = ["PNG image (*.png)", "*.png"]
GIF_FILTER = ["Animated GIF (*.gif)", "*.gif"]
# ``plotter_io.TMX_FILTER``'s own shape, for the one other place a Tiled
# document is opened or written: a tileset a document holds *is* a
# ``tilegrid.Tileset`` (Chunk 3.1), so exporting or importing one reaches for
# ``plotter.tsx`` directly rather than converting through anything.
TSX_FILTER = ["Tiled tileset (*.tsx)", "*.tsx"]

# The layered format plus every image the app accepts anywhere -- the tuple
# from ``filetypes``, not a copy of it, so the picker and the suffix check can
# never disagree with each other or with what a drop accepts.
OPENABLE = (".ora", *filetypes.IMAGE_SUFFIXES)
OPEN_FILTER = ["Images and layered files", filetypes.pattern(OPENABLE)]

# Deliberately **not** in ``OPENABLE``, still. ``aseout.write_aseprite`` means
# the app can now write this format -- through an explicit Save As, see
# ``save_as`` below -- but that is a capability, not a route: ``OPENABLE`` is
# what a plain *open* or a drop resolves through ``Document.load``, and an
# Aseprite file still arrives through the import door of its own
# (``ask_import_aseprite``/``_load_aseprite``) as an unsaved ORA document whose
# first Ctrl+S asks where to put it. Routing it through ``Document.load``
# instead would leave a tab pointing at the file it was opened from, and one
# Ctrl+S would put ORA bytes over it -- the exact overwrite the import door
# exists to prevent, and the reason ``WRITABLE_SUFFIXES`` below still excludes
# this format even after a Save As has written one: the export is lossy
# (alpha_lock, group opacity, matte, palette-constrained mode all fall out of
# the round trip), and a lossy in-place write on a keystroke that means "keep
# what I have" is exactly what that gate refuses. The suffix pair itself is
# ``asein.ASEPRITE_SUFFIXES`` -- the reader's own constant, imported above
# rather than restated, so the two lists cannot drift.
ASEPRITE_FILTER = ["Aseprite files", filetypes.pattern(ASEPRITE_SUFFIXES)]
# ``save_as``'s picker: ORA first (the default, unsaved suggestion stays
# ``.ora``) with the Aseprite pair beside it -- one dialog, two writers, and
# the *suffix the user actually typed or picked* decides which of them runs.
SAVE_AS_FILTER = ORA_FILTER + ASEPRITE_FILTER

# The two suffixes an in-place save may write, and the reason ``OPENABLE`` is
# not that list. ``Document.load`` stamps every non-ORA input ``file_format =
# "png"`` and ``_write`` dispatches on the *format*, never on the path -- so
# before this gate, opening ``foo.jpg`` and pressing Ctrl+S put PNG bytes into
# a file still named ``.jpg``: a file no viewer reads by its extension and no
# second open recovers, written with no prompt.
#
# Re-encoding to JPEG instead would be the other kind of silence -- a lossy
# write the user did not ask for, over their original, on a keystroke that
# means "keep what I have". So the refusal is the honest half of the same
# argument the Aseprite comment above makes: what cannot be written back is
# saved *somewhere else*, deliberately, through the Save As the user sees.
WRITABLE_SUFFIXES = (".ora", ".png")

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
            # Element types validated too, the module's doctrine for a
            # hand-editable settings file (``_restore_presets``): a swatch
            # entry holding a string would raise out of ``int`` on the first
            # frame Paint mode is opened.
            state.swatches = [
                tuple(int(c) for c in s)  # type: ignore[misc]
                for s in swatches
                if isinstance(s, list | tuple)
                and len(s) == 4
                and all(isinstance(c, int | float) and not isinstance(c, bool) for c in s)
            ] or list(inker_state.DEFAULT_SWATCHES)
        _restore_presets(state, stored.get("presets"))
        _restore_canvas(state, stored.get("canvas"))
        _restore_export(state, stored.get("export"))
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
    """The swatches and the tool presets: the recent list moved to
    :mod:`.recents`, which persists itself on every write.

    The presets ride here rather than in a file of their own for the swatches'
    reason -- a handful of session settings, not a library -- and the block is
    written whole every time, which is what makes deleting one persist without
    a second door.
    """
    state = ctx.state.inker
    if state is None:
        return
    # Merged into whatever is stored rather than replacing it, so the legacy
    # ``recent`` key survives untouched: ``recents`` folds the four per-mode
    # lists in on *its* first read, which may well be after this has run.
    stored = ctx.settings.get("inker")
    block = dict(stored) if isinstance(stored, dict) else {}
    block["swatches"] = [list(s) for s in state.swatches]
    block["presets"] = {
        name: {"tool": saved["tool"], "options": dict(saved["options"])}
        for name, saved in state.presets.items()
    }
    # The canvas furniture: how the user likes to see, which -- like the
    # swatches -- is a property of the person rather than of any document.
    block["canvas"] = {
        "grid": bool(state.grid),
        "grid_size": int(state.grid_size),
        "grid_snap": bool(state.grid_snap),
        "rulers": bool(state.rulers),
        # The other three ``_toggle`` reaches for. Its docstring promises "a
        # preference that resets on the next launch is a control they have to
        # rediscover" and it calls ``persist`` for all six, but only the four
        # above were ever written or read back -- so the pixel grid, the layer
        # edges and the tile numbers paid a full settings write on every toggle
        # and reset every launch anyway.
        "pixel_grid": bool(state.pixel_grid),
        "layer_edges": bool(state.layer_edges),
        "tile_numbers": bool(state.tile_numbers),
    }
    # The last-used export controls -- app-level and shared across tabs, like
    # the canvas furniture beside it, not a per-document ``InkerDoc.
    # export_options`` (those are session-only and never leave the tab). This
    # is what seeds a brand-new tab's controls next session, through
    # ``_restore_export`` below.
    block["export"] = state.export_options_snapshot()
    ctx.settings.set("inker", block)


def _restore_canvas(state: InkerState, stored: Any) -> None:
    """The grid and ruler preferences back off disk, validated not trusted --
    the same doctrine as ``_restore_presets``, for the same hand-editable file."""
    if not isinstance(stored, dict):
        return
    state.grid = bool(stored.get("grid", state.grid))
    size = stored.get("grid_size")
    if isinstance(size, int) and not isinstance(size, bool):
        state.grid_size = max(2, min(512, size))
    state.grid_snap = bool(stored.get("grid_snap", state.grid_snap))
    state.rulers = bool(stored.get("rulers", state.rulers))
    state.pixel_grid = bool(stored.get("pixel_grid", state.pixel_grid))
    state.layer_edges = bool(stored.get("layer_edges", state.layer_edges))
    state.tile_numbers = bool(stored.get("tile_numbers", state.tile_numbers))


def _restore_export(state: InkerState, stored: Any) -> None:
    """The last-used export controls back off disk, over the built-in
    defaults, through :meth:`InkerState.apply_export_options` -- the one door
    that also validates a hand-edited or an older-build settings file, so this
    is a one-line forward rather than a second copy of that doctrine."""
    state.apply_export_options(stored)


def _restore_presets(state: InkerState, stored: Any) -> None:
    """Load the stored preset block, dropping anything that is not one.

    Validated rather than trusted: this is a JSON file a user may edit by hand
    and a build may have changed the option table under, so a preset that
    arrives as a string or names a tool that no longer exists is ignored rather
    than crashing the first frame the panel draws. ``apply_preset`` filters the
    option keys again on the way out, so an entry that survives here is still
    safe if the table moves later.
    """
    if not isinstance(stored, dict):
        return
    tools = {key for key, _label, _short in inker_state.TOOLS}
    for name, saved in list(stored.items())[: inker_state.MAX_PRESETS]:
        if not isinstance(name, str) or not isinstance(saved, dict):
            continue
        options = saved.get("options")
        if saved.get("tool") not in tools or not isinstance(options, dict):
            continue
        state.presets[name[: inker_state.MAX_PRESET_NAME]] = {
            "tool": saved["tool"],
            "options": {
                key: value
                for key, value in options.items()
                if key in inker_state.TOOL_OPTION_DEFAULTS
            },
        }


# --- the image brush --------------------------------------------------------


def capture_brush(ctx: Any) -> bool:
    """Make the selection into the brush tip. -> whether one was captured.

    The whole gesture rather than the engine call alone, because the gesture is
    three things and the middle one is what makes it usable: capture, **pick the
    brush up**, and turn the image tip on. Capturing into a tool the user then
    has to go and find is a feature that looks like it did nothing -- and the
    tool in their hand at that moment is by definition a selection tool, since
    they have just made a selection.

    The two refusals are told apart and both are said out loud: the engine
    answers None to either, and a silent no is indistinguishable from a bug.
    """
    from . import inker

    state = ensure(ctx)
    tab = state.active
    if tab is None:
        return False
    if tab.doc.mask is None:
        ctx.toast("Select something first -- that is what becomes the tip.", "warn")
        return False
    stamp = tab.doc.capture_stamp()
    if stamp is None:
        ctx.toast(
            "That selection is too big for a brush tip -- "
            f"{inker.MAX_STAMP} pixels a side at most.",
            "warn",
        )
        return False
    state.stamp = stamp
    # Through ``set_tool``, like every other way of picking one.
    state.set_tool("brush")
    state.use_stamp = True
    return True


def clear_brush(ctx: Any) -> None:
    """Drop the captured tip and go back to a round one everywhere.

    Every stamping tool's flag, not just the one in hand: the tip is app-level
    and this is the button that says "forget it", so leaving the eraser ticked
    would make a later capture arrive on a tool the user is not looking at.
    """
    state = ensure(ctx)
    state.stamp = None
    for tool in inker_state.STAMP_TOOLS:
        state.options_for(tool)["use_stamp"] = False


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
    if not job_id:
        # A linked document's path is the job's *served* ``input.png``, and a
        # recents entry would offer it back as a plain file: reopening it that
        # way makes Ctrl+S write PNG bytes over the served file in place,
        # bypassing ``save_edited_image``'s backup and staged replace. The way
        # back to a job document is the job's own "Open in Inker", which
        # re-links it.
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
    if path.suffix.lower() in ASEPRITE_SUFFIXES:
        # Before the ``OPENABLE`` check rather than inside it: a drop of an
        # Aseprite file is an import, and refusing it with "Inker opens images
        # and .ora files" would be telling the user the app cannot do something
        # it can. Nothing is remembered for it -- the tab owns no file, so a
        # recent entry would offer to reopen a document that is never *this*
        # document again.
        import_aseprite_path(ctx, path)
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


def ask_import_aseprite(ctx: Any) -> None:
    """Pick an ``.aseprite`` off disk and open what it holds.

    One step where the sheet import is two, and for the reason that one is
    two: a sheet has to be told how to cut, and an Aseprite file already says
    where every layer, cel, frame and tag is. So there is nothing to ask, and
    the result goes straight down the ``inker-open`` road with no routing of
    its own.
    """
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Import Aseprite file", ASEPRITE_FILTER)
        return None if path is None else _load_aseprite(Path(path))

    ctx.submit("inker-open:aseprite", run)


def import_aseprite_path(ctx: Any, path: Path) -> None:
    """The same import for a path already in hand -- a drop onto the window."""
    ensure(ctx)
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
    from .inker import asein

    path = Path(path)
    doc, warnings = asein.document_from_aseprite(path.read_bytes())
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
    state = ensure(ctx)
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
    state = ensure(ctx)
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
    ``DirectionalLayout`` *kind*: ``animation.SHEET_KINDS`` is ``turnaround``
    and ``walk`` only, so ``DirectionalLayout.of()`` would return ``None`` and
    the whole door would refuse. The grid slicer wants only the geometry, which
    the sidecar has.
    """
    ensure(ctx)
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
    ensure(ctx)
    set_mode(ctx.state, "inker")

    def run() -> dict[str, Any]:
        """Blocking; task thread only."""
        import numpy as np
        from PIL import Image

        from ..service import derive as svc_derive
        from . import inker

        path = svc_derive.get_file(
            ctx.svc,
            job_id,
            name,
            pixel_colors=pixel_colors,
            pixel_palette=pixel_palette,
            pixel_dither=pixel_dither,
        )
        with Image.open(path) as opened:
            opened.load()
            array = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
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
    import numpy as np
    from PIL import Image

    from ..service import sheets as svc_sheets
    from .inker import sheetin

    if pixel:
        record = svc_sheets.get_pixel_sheet(svc, job_id, sheet_id)
        png = svc_sheets.sheet_pixel_png(svc, job_id, sheet_id)
    else:
        record = svc_sheets.get_sheet(svc, job_id, sheet_id)
        png = svc_sheets.sheet_png(svc, job_id, sheet_id)
    cell, count = sheet_grid(record)
    with Image.open(png) as opened:
        opened.load()
        atlas = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
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
        doc = sheetin.document_from_sheet(atlas, cells, animation)
    else:
        doc = sheetin.document_from_grid(atlas, cell, count=count)
    name = str(record.get("name") or sheet_id)
    return {
        "doc": doc,
        "path": None,
        "format": "ora",
        "title": f"{name} (pixel)" if pixel else name,
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


def end_convert_session(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Close an open palette-conversion preview, on the document that owns it.

    **Resolved by uid, never through ``active``.** The session lives on one
    ``Document``; ``InkerState`` is one object shared by every tab; and the
    palette pane draws whichever tab is in front. Reaching for ``active`` here
    is exactly how a tab switch came to cancel the conversion on the wrong
    document -- restoring planes that were never previewed while the previewed
    ones kept a dither nobody had approved, with no hook left to take it back.

    With *tab*, this is "settle **this** document if the session is its own",
    and another tab's session is left alone. Without one, it is "settle whatever
    is open", which is what a pane that has stopped drawing the popup needs.

    A uid that names no tab any more -- it was closed under the popup -- clears
    the flag and ends nothing: the document went with it.
    """
    state = ctx.state.inker
    if state is None or not state.convert_uid:
        return
    owner = state.get(state.convert_uid)
    if tab is not None and owner is not None and owner.uid != tab.uid:
        return
    if owner is not None:
        owner.doc.end_convert()
    state.convert_uid = ""


def end_filter_session(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Close an open filter preview, on the document that owns it.

    ``end_convert_session``'s twin, resolved by uid for its reasons -- the
    session lives on one ``Document``, ``InkerState`` is shared by every tab,
    and the popup is drawn for whichever tab is in front.

    It is a *cancel*: ``Document.end_filter``'s rule, that an unanswered
    question is not a yes. Until this existed the pixels a preview had written
    were live in the layer with nothing on the undo stack, so a save serialised
    a filter the user had not approved and ``mark_saved`` called the tab clean
    against it.
    """
    state = ctx.state.inker
    if state is None or not state.filter_uid:
        return
    owner = state.get(state.filter_uid)
    if tab is not None and owner is not None and owner.uid != tab.uid:
        return
    if owner is not None:
        owner.doc.end_filter()
    state.filter_uid = ""


def _settle(ctx: Any, tab: InkerDoc) -> None:
    """Fold what the canvas is showing into what the document is holding.

    At the top of every save and every export, and it is two calls rather than
    one because there are two transient states a serialise could catch -- and
    they are opposite mistakes.

    A **floating buffer** lives in no layer, so the encoders would omit pixels
    the user can see, and ``mark_saved`` would then call the document clean:
    which is how a paste came to be discarded with no prompt. It is *committed*.

    A **conversion preview** is the other way round: the planes are already
    carrying pixels the user has not said yes to, so the file would hold a
    dither nobody asked for and the tab would go clean against it. It is
    *cancelled*, for ``end_filter``'s reason -- an unanswered question is not a
    yes.

    A **filter preview** is the conversion preview's twin and was missed here
    for as long as this function existed: ``preview_filter`` writes into the
    layer every frame the popup is up, so the same sentence applies word for
    word. Cancelling it is the whole of what stopped Ctrl+S with a filter popup
    open from writing an unapproved filter to disk.
    """
    end_convert_session(ctx, tab)
    end_filter_session(ctx, tab)
    tab.doc.commit_floating()


def save(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Ctrl+S. Save As when the document has never been written anywhere."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    # Playback stopped *before* the capture, synchronously: an encode walks
    # the live layer stack on a task thread, and playback's own stop --
    # Escape, or the clip reaching a non-looping tag's end -- rebuilds that
    # stack through ``set_current_frame`` mid-write. Stopping here on the
    # frame thread settles the document first; ``stop_play`` is a no-op on a
    # still tab.
    stop_play(tab)
    if tab.linked:
        _save_linked(ctx, tab)
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    # ``tab.path`` *is* the record of where the document came from -- Save As
    # replaces it with the ``.ora`` it wrote -- so the source suffix needs no
    # field of its own. See ``WRITABLE_SUFFIXES`` for why a JPG cannot be
    # saved in place.
    suffix = tab.path.suffix.lower()
    if suffix not in WRITABLE_SUFFIXES:
        if suffix in ASEPRITE_SUFFIXES:
            # This file did not arrive from outside -- it was *written* here,
            # by an explicit Save As -- so "came from a {SUFFIX} file" would
            # be backwards as well as ungrammatical. The remedy is the same
            # (Save As again), but the reason is its own.
            ctx.toast(
                "This file was written by Save As; Inker does not overwrite"
                " .aseprite in place. Save a copy or an .ora.",
                "info",
            )
        else:
            ctx.toast(
                f"This drawing came from a {suffix.lstrip('.').upper()} file, which "
                "Inker cannot write. Choose where to save the layered copy.",
                "info",
            )
        save_as(ctx, tab)
        return
    _submit_write(ctx, tab, f"inker-save:{tab.uid}", tab.path, tab.file_format)


def save_as(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Two writers behind one dialog. The suggested name and the default
    filter row are still ORA -- nothing about the ordinary flow changes -- but
    a suffix the user chose or typed as ``.aseprite``/``.ase`` routes to
    :func:`aseout.write_aseprite` instead, in :func:`_write`. Anything else,
    including no suffix at all, still lands as ``.ora``: only the two Aseprite
    suffixes opt out of that default, the same way ``dest.with_suffix`` always
    corrected a wrong or missing one.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    stop_play(tab)  # settle the stack before capturing; see save()
    doc = tab.doc
    _settle(ctx, tab)  # before the head is read; see _save_linked
    rev = doc.history.head
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Save layered document", f"{suggested}.ora", SAVE_AS_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() in ASEPRITE_SUFFIXES:
            _write(doc, dest, "aseprite")
            return {"path": dest, "rev": rev, "format": "aseprite", "retitle": True}
        if dest.suffix.lower() != ".ora":
            dest = dest.with_suffix(".ora")
        _write(doc, dest, "ora")
        return {"path": dest, "rev": rev, "format": "ora", "retitle": True}

    _start(ctx, tab, f"inker-saveas:{tab.uid}", run)


def export_png(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """A flattened PNG. Not a save: it does not change what the tab points at,
    so the document stays dirty against its own file.

    ``repeat`` writes straight to the recorded destination and opens no dialog
    -- Repeat Last Export (6.9). The *destination* is what makes that safe to
    do silently: it is a path this user chose for this document, and the toast
    afterwards names it.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    stop_play(tab)  # settle the stack before capturing; see save()
    doc = tab.doc
    # Not a save, but the same rule about what is on the canvas: the composite
    # a floating buffer draws into is the pane's, not the document's, so an
    # export would otherwise be missing pixels the user is looking at.
    _settle(ctx, tab)
    suggested = tab.path.stem if tab.path else "untitled"
    state = ctx.state.inker
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))

    recorded = tab.export_dest if repeat else None

    def run() -> dict[str, Any] | None:
        dest = recorded or dialogs.save_file(
            "Export flattened PNG", f"{suggested}.png", PNG_FILTER
        )
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        dest.write_bytes(doc.png_bytes(scale=scale))
        # ``dest`` and ``export_kind`` so the *next* repeat has something to
        # repeat -- ``on_task_done`` records both.
        return {"exported": dest, "dest": dest, "export_kind": "png"}

    _start(ctx, tab, f"inker-export:{tab.uid}", run)


#: A filesystem-safe stem: the same character set ``plotter.tmx._stem`` allows,
#: because both are a slice's or a tileset's *name* headed for a filename and
#: there is no reason for the two rules to disagree.
_SLICE_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slice_filenames(entries: list[Any]) -> list[str]:
    """Sanitised, collision-free file stems, one per slice, in document order.

    S1 does not pin slice names unique -- two slices called "Hitbox" are a
    legitimate authoring choice -- so this is where the collision is caught: the
    first "Hitbox" keeps its name and the second becomes "Hitbox_2", the same
    shape ``tmx._stem`` uses for two tilesets that share a name, with a trailing
    counter rather than a leading index because a human picks these files off a
    folder listing instead of an engine matching them by position.

    Every candidate is checked against **every name already handed out**, not
    against other occurrences of the same sanitised base -- a counter kept per
    base independently of the others can mint the same bumped name twice (a
    third "Hitbox" landing on "Hitbox_2" a second slice already claimed, or a
    literal "a_2" colliding with what a repeated "a" bumps to), which is a
    silent overwrite in ``run()`` rather than a name a user ever sees.

    **This module holds two collision policies and the split is deliberate.**
    :func:`_split_stems` *refuses* where this bumps, and the deciding question
    is whether anything downstream addresses the file **by name**. Nothing
    addresses a slice PNG by name -- a human picks it off a folder listing, and
    "Hitbox_2.png" is a name they can read and live with. A tag or a layer *is*
    addressed by name by whatever consumes the sheet, so a second "walk"
    quietly becoming "walk_2.png" would be a file claiming to be a clip that
    does not exist. Bumping is friendly where a human disambiguates and
    dishonest where a machine does. A third naming helper answers that same
    question before it picks a side, and ``tests/inker/test_slice_export.py``
    pins both halves against each other so neither can drift onto the other's
    policy unnoticed.
    """
    taken: set[str] = set()
    out = []
    for entry in entries:
        base = _SLICE_SAFE.sub("-", entry.name).strip("-") or "slice"
        candidate = base
        counter = 2
        while candidate in taken:
            candidate = f"{base}_{counter}"
            counter += 1
        taken.add(candidate)
        out.append(candidate)
    return out


def export_slices(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """Every slice as its own PNG, cropped from the current frame's flatten.

    Each slice resolves ``at(current_frame_uid)`` -- so a keyed slice exports
    the rectangle the panel beside it is showing right now, on a still document
    exactly as on an animated one. A per-frame *matrix* of one crop per slice
    per frame is a different export and stays out of scope here; it is
    Packwright's job.

    Not spread through the stepper the animated exports use: this reads one
    flatten, not one per frame, so there is nothing to spend across app frames.
    The geometry -- names and bounds -- is resolved here, on the frame thread,
    for ``_submit_export``'s reason about ``slices_snapshot``: the tab is
    locked (``saving``) for the rest of the call, so "now" and "inside the
    task" would answer the same question, and every other read in this
    function already happens here.
    """
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    doc = tab.doc
    if not doc.slices:
        return
    _settle(ctx, tab)
    suggested = tab.path.stem if tab.path else "untitled"
    state = ctx.state.inker
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))
    frame_uid = tab.frame_uid
    names = _slice_filenames(doc.slices)
    crops = [
        (name, entry.at(frame_uid).bounds)
        for name, entry in zip(names, doc.slices, strict=True)
    ]

    def run() -> dict[str, Any] | None:
        from PIL import Image

        from .inker.transform import upscale

        dest = dialogs.save_file("Export slices as PNGs", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Read here, inside the task, for ``_write``'s reason: the encoders only
        # read, and the frame thread only ever appends to a layer's pixels in
        # place, so the worst this catches is a stroke that was mid-flight.
        flat = doc.flatten()
        first = None
        for name, (x0, y0, x1, y1) in crops:
            crop = upscale(flat[y0:y1, x0:x1], scale)
            out = dest.parent / f"{name}.png"
            Image.fromarray(crop, "RGBA").save(out, "PNG")
            if first is None:
                first = out
        return {"exported": first}

    _start(ctx, tab, f"inker-export:{tab.uid}", run)


def export_sheet(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
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
    _begin_export(ctx, tab, "sheet", repeat=repeat)


def export_gif(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
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
    _begin_export(ctx, tab, "gif", repeat=repeat)


def export_pngs(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """Every frame as its own numbered PNG, through the same stepper.

    The plainest export there is, and the one an engine with its own importer
    asks for: no atlas to slice, no sidecar to parse. The spread is untouched --
    the frames are read exactly as the sheet and the GIF read them, and only
    the write differs.
    """
    _begin_export(ctx, tab, "pngs", repeat=repeat)


@dataclass
class _Leg:
    """One *output file's* worth of an export: which frames, read how.

    An ordinary export has exactly one of these and its ``label`` is empty. A
    split has one per tag or per layer, and the label is what the filename is
    built from -- see :func:`_split_stems`.
    """

    uids: list[str]
    #: What this output is called after the stem, or "" for the single-file
    #: export that is named by the dialog alone.
    label: str = ""
    #: The inclusive frame range being exported, or None for the whole
    #: timeline. Sliced **at begin**, and ``timing`` is sliced at submit with
    #: this same pair -- safe because the tab has been locked (``saving``) for
    #: the whole spread, so the frame count cannot have moved between them.
    span: tuple[int, int] | None = None
    #: What a GIF's loop block should say: True forever, False once, or a
    #: repeat count. See ``gifout.loop_option``.
    loop: bool | int = True
    #: The tracks this leg composites, or None for the whole stack. A split by
    #: layer is the only caller that sets it, and it is what sends the flatten
    #: through ``sheetout.flatten_subset`` -- which stays out of the document's
    #: frame cache, because that cache is keyed on the frame uid alone.
    track_uids: tuple[int, ...] | None = None
    #: "" | "tag" | "layer" -- which of ``sheetout.filename_for``'s two keys
    #: ``label`` fills, so ``_split_stems`` knows whether to build each stem's
    #: default template from :data:`sheetout.DEFAULT_TAG_TEMPLATE` or
    #: :data:`sheetout.DEFAULT_LAYER_TEMPLATE`. Empty for the ordinary,
    #: unsplit leg, whose label is empty too.
    split_kind: str = ""
    frames: list[Any] = field(default_factory=list)
    #: One exact index plane per read frame, or None where the frame's flatten
    #: is not a cel's own materialisation. Parallel to ``frames`` and appended
    #: in the same step, so the two cannot come apart. Only a GIF reads it --
    #: see ``sheetout.index_plane_one``.
    planes: list[Any] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.frames) >= len(self.uids)


@dataclass
class _Export:
    """One export's frame-by-frame read of the document.

    Lives on ``InkerState`` rather than on the tab because it is not a property
    of the document -- it is one in-flight operation, and there is one at a time
    by construction (both exports share a task key, and the tab is locked while
    it runs).

    **A split is one export, not several.** One lock, one stepper, one flatten
    per pump, one task at the end -- the legs are read back to back inside the
    machinery that already guarantees all four. N exports racing each other for
    the same task key, with the tab locked N times over, is the shape this
    deliberately does not take.

    ``uids``/``frames``/``planes``/``span``/``loop`` read the leg being flattened
    now, so every caller written before splits existed still sees the export it
    always saw.
    """

    tab: InkerDoc
    kind: str  # "sheet" | "gif" | "pngs"
    suggested: str
    legs: list[_Leg]
    #: The destination a **repeat** is writing to, or None for the ordinary
    #: export that asks. Carried on the export rather than read off the tab at
    #: the dialog, because by then the tab is locked and the answer has to be
    #: the one the click was made with (6.9).
    recorded: Path | None = None
    #: Which leg the stepper is on. Never rewound: a finished leg's frames stay
    #: on it until the submit reads them.
    at: int = 0

    @property
    def leg(self) -> _Leg:
        return self.legs[self.at]

    @property
    def uids(self) -> list[str]:
        return self.leg.uids

    @property
    def frames(self) -> list[Any]:
        return self.leg.frames

    @property
    def planes(self) -> list[Any]:
        return self.leg.planes

    @property
    def span(self) -> tuple[int, int] | None:
        return self.leg.span

    @property
    def loop(self) -> bool | int:
        return self.leg.loop

    @property
    def read(self) -> int:
        """Frames flattened so far, across every leg. One per pump, always."""
        return sum(len(leg.frames) for leg in self.legs)


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
    from .inker import sheetout

    tag = anim.tags[index]
    _begin_export(
        ctx,
        tab,
        kind,
        span=sheetout.tag_span(anim, tag),
        loop=tag.repeat or tag.loop,
    )


def export_per_tag(ctx: Any, tab: InkerDoc | None = None, kind: str = "sheet") -> None:
    """One file per tag, in one export.

    Each output is exactly what :func:`export_tag` writes for that tag on its
    own -- same span through ``sheetout.tag_span``, same looping, same rebased
    tags in the sidecar -- so a batch and a one-at-a-time sweep produce the same
    files. That is the whole reason the span logic is shared rather than
    repeated here.
    """
    from .inker import sheetout

    tab = tab or active(ctx)
    anim = None if tab is None else tab.doc.anim
    if tab is None or anim is None:
        return
    if not anim.tags:
        # Reachable even though the menu item is disabled: the verb is engine
        # API, and a refusal that says why beats one that does nothing.
        ctx.toast("This document has no tags to split by.", "warn")
        return
    _begin_export(
        ctx,
        tab,
        kind,
        legs=[
            _Leg(
                uids=[],
                label=tag.name,
                span=sheetout.tag_span(anim, tag),
                loop=tag.repeat or tag.loop,
                split_kind="tag",
            )
            for tag in anim.tags
        ],
    )


def export_per_layer(ctx: Any, tab: InkerDoc | None = None, kind: str = "sheet") -> None:
    """One file per top-level layer row, in one export.

    ``sheetout.layer_splits`` decides what a "layer" is here -- a track, or a
    whole group as the one row the panel shows -- and each leg composites only
    its own tracks. The frames are the same frames; what differs is how much of
    the stack goes into each of them.
    """
    from .inker import sheetout

    tab = tab or active(ctx)
    if tab is None or tab.doc.anim is None:
        return
    splits = sheetout.layer_splits(tab.doc)
    if not splits:
        ctx.toast("Every layer is hidden; there is nothing to split.", "warn")
        return
    _begin_export(
        ctx,
        tab,
        kind,
        legs=[
            _Leg(uids=[], label=name, track_uids=uids, split_kind="layer")
            for name, uids in splits
        ],
    )


#: Which function repeats which recorded export. A table rather than a chain
#: of ifs, so a seventh export kind is one row and cannot be forgotten by the
#: repeat path alone -- which is exactly how a "repeat" command goes stale.
REPEATABLE: dict[str, str] = {
    "png": "export_png",
    "sheet": "export_sheet",
    "gif": "export_gif",
    "pngs": "export_pngs",
    "slices": "export_slices",
}


def repeat_export(ctx: Any, tab: InkerDoc | None = None) -> bool:
    """Ctrl+Shift+X: run the last export again, with no dialog.

    **The hot-path escape valve**: configure once, then one key forever. It is
    the whole reason the per-document destination memory exists, and until now
    that memory only seeded the *dialog* -- so the user still had to answer it.

    Refused out loud when this document has never been exported, because "the
    last export" is not a thing yet and a silent key is one the user cannot
    tell from a broken one.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return False
    kind = getattr(tab, "export_kind", "")
    verb = REPEATABLE.get(kind)
    if not verb or tab.export_dest is None:
        state.say(
            "Nothing to repeat yet -- export once and this runs the same one "
            "again."
        )
        return False
    globals()[verb](ctx, tab, repeat=True)
    return True


def _begin_export(
    ctx: Any,
    tab: InkerDoc | None,
    kind: str,
    *,
    span: tuple[int, int] | None = None,
    loop: bool | int = True,
    legs: list[_Leg] | None = None,
    repeat: bool = False,
) -> None:
    """Lock the tab and park the stepper. The click-frame half of an export.

    ``legs`` is the split form: one entry per output file, each carrying its own
    span and its own tracks, and ``None`` is the ordinary single-file export
    (one unlabelled leg over ``span``). Everything after this point -- the
    stepper, the lock, the submit -- is the same code for both.
    """
    from .inker import sheetout

    tab = tab or active(ctx)
    state = ctx.state.inker
    if tab is None or tab.busy or tab.doc.anim is None or state is None:
        return
    if state.export is not None:
        return
    # Seeded once per tab, not on every click: a fresh tab's ``export_options``
    # is empty and this is a no-op, but a tab exported once already restores
    # what *it* last used -- over whatever the previous tab exporting left on
    # the shared controls -- the first time it exports again. Guarded on
    # ``export_seed_uid`` so a user who tweaks a control and clicks Export a
    # second time for the *same* tab keeps that edit rather than having it
    # silently put back; only a switch away and back re-suggests.
    if state.export_seed_uid != tab.uid:
        state.apply_export_options(tab.export_options)
        state.export_seed_uid = tab.uid
    suggested = tab.path.stem if tab.path else "untitled"
    if legs is None:
        legs = [_Leg(uids=[], span=span, loop=loop)]
    leg_kind = legs[0].split_kind if legs else ""
    template = str(getattr(state, "export_template", "") or "").strip() or None
    try:
        # The work lists are filled *here* rather than by the callers, so a span
        # that holds no frames refuses before the lock -- and so "frame 3 of 60"
        # is read once, for every leg, on the frame the button was pressed.
        for leg in legs:
            leg.uids = sheetout.frame_uids(tab.doc, leg.span)
        # Checked here rather than in the runner, where the stem the user picked
        # is finally known: a collision is a property of the labels alone, so it
        # can be refused before the tab is locked and before somebody names a
        # batch that was never going to be written.
        _split_stems(
            suggested, [leg.label for leg in legs], kind=leg_kind, template=template
        )
    except ValueError as exc:
        ctx.toast(f"Cannot export: {exc}.", "warn")
        return
    # **After the refusals, not before them.** Settling commits the floating
    # buffer and cancels the filter and conversion previews, all of which change
    # the document -- so running it above the two checks meant an export that
    # went on to say "cannot export" had already folded a paste into the layers
    # on its way to refusing. Nothing either check reads is affected by it: they
    # ask which frames a span holds and whether the labels collide.
    _settle(ctx, tab)
    # Locked before the first flatten, not at submit time: the whole point of
    # spreading the read is that frames go by between here and the encode, and
    # an edit landing in one of them would put half of two documents in the
    # sheet. ``saving`` is the flag ``busy`` already refuses mutation on.
    tab.saving = True
    state.export = _Export(
        tab=tab,
        kind=kind,
        suggested=suggested,
        legs=legs,
        recorded=tab.export_dest if repeat else None,
    )


def _suggested_dialog_name(tab: InkerDoc, suggested: str, ext: str) -> str:
    """The ``default_name`` an export's save dialog opens with.

    A bare filename ordinarily, exactly what every export always passed. Once
    this tab has exported before, ``export_dest``'s *folder* is folded in
    too -- a native picker reads a path's directory half as where to open --
    so picking this tab back up suggests the folder it was last written into
    rather than wherever the picker happened to be left by the tab exported in
    between.
    """
    dest = tab.export_dest
    if dest is None:
        return f"{suggested}{ext}"
    return str(dest.parent / f"{suggested}{ext}")


def _split_stems(
    stem: str, labels: Sequence[str], *, kind: str = "", template: str | None = None
) -> list[str]:
    """One filename stem per output, through :func:`sheetout.filename_for`.

    **The one place a split's filenames are decided**, which is what makes
    Task 5's filename templates a single edit rather than a sweep through
    three runners. An empty label is the unsplit export and keeps the stem
    the dialog was given, byte for byte -- no template involved, because
    there is nothing here for one to distinguish -- but that shortcut only
    applies when ``kind`` is also empty. A split leg's label can *itself* be
    empty (a loaded ``.ase``/ORA may carry a tag or a track with no name),
    and that empty label is a real, if badly named, split output -- not the
    unsplit sentinel. Collapsing it onto the bare stem would write a file
    indistinguishable from a whole-document export, so a falsy label under a
    non-empty ``kind`` falls back to the literal word ``"tag"``/``"layer"``
    instead, and still goes through the same template and collision check as
    every other label.

    ``kind`` is ``"tag"`` or ``"layer"``, and it picks both which of
    ``filename_for``'s two keys a non-empty label fills and which default
    template applies when ``template`` is None -- :data:`sheetout.
    DEFAULT_TAG_TEMPLATE` or :data:`sheetout.DEFAULT_LAYER_TEMPLATE`, the
    exact ``f"{stem}_{safe}"`` this always wrote before templates existed.

    A collision is **refused**, where ``_slice_filenames`` bumps: a slice is a
    rectangle a person picks off a folder listing, and "Hitbox_2.png" is a name
    they can live with -- but a tag and a layer are addressed *by name* by
    whatever consumes the sheet, and a second "walk" quietly becoming
    "walk_2.png" is a file claiming to be a clip called walk_2. Refusing is the
    only answer that cannot silently be believed. A template that renders two
    labels the same collides here for the identical reason.
    """
    from .inker import sheetout

    default = (
        sheetout.DEFAULT_LAYER_TEMPLATE
        if kind == "layer"
        else sheetout.DEFAULT_TAG_TEMPLATE
    )
    tmpl = template or default
    out: list[str] = []
    for label in labels:
        if not label:
            if not kind:
                out.append(stem)
                continue
            label = "layer" if kind == "layer" else "tag"
        out.append(
            sheetout.filename_for(
                tmpl,
                title=stem,
                tag=None if kind == "layer" else label,
                layer=label if kind == "layer" else None,
            )
        )
    sheetout.require_distinct_names(out)
    return out


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
    leg = export.leg
    try:
        uid = leg.uids[len(leg.frames)]
        if leg.track_uids is None:
            plane = sheetout.flatten_one(tab.doc, uid)
            # Read here, beside the flatten it describes and on the same frame:
            # taken later it would describe a document the user has since
            # edited, and the two have to be a matched pair or the GIF is drawn
            # with one frame's slots and another frame's colours.
            leg.planes.append(sheetout.index_plane_one(tab.doc, uid))
        else:
            plane = sheetout.flatten_subset(tab.doc, uid, leg.track_uids)
            # None rather than a subset index plane: ``index_plane_one`` decides
            # by comparing a candidate cel against the *whole* frame's flatten,
            # which a subset is not. A GIF of one layer therefore quantises from
            # the colours, as every GIF did before index planes existed --
            # correct, just not slot-stable.
            leg.planes.append(None)
        leg.frames.append(plane)
    except (ValueError, IndexError, KeyError):
        state.export = None
        tab.saving = False
        ctx.toast("Export failed: a frame could not be flattened.", "warn")
        return
    if not leg.done:
        return
    if export.at + 1 < len(export.legs):
        # Exactly one flatten has happened this pump, so the next leg starts on
        # the next one: a batch that ran the legs back to back here would be the
        # freeze the stepper exists to prevent, N times over.
        export.at += 1
        return
    state.export = None
    _submit_export(ctx, export)


@dataclass
class _Payload:
    """One output file, as the runners need it: pixels plus what describes them.

    Built on the frame thread by :func:`_submit_export` and read on the task
    thread, which is why it holds values rather than a document to ask.
    """

    label: str
    frames: list[Any]
    planes: list[Any]
    durations: list[int]
    tags: list[Any]
    layout: Any
    slices: list[Any]
    loop: bool | int


def _submit_export(ctx: Any, export: _Export) -> None:
    """The work list is read; hand it to a task. Frame thread."""
    from .inker import gifout, sheetout
    from .inker.transform import upscale

    tab, suggested = export.tab, export.suggested
    doc = tab.doc
    state = ctx.state.inker
    # Read here, on the frame thread, with the frames: an app-level setting the
    # user could change while the encode is in flight would otherwise decide
    # the file's size halfway through writing it.
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))
    # Same reason as ``scale`` beside it: a setting the user could change
    # mid-encode must not decide, halfway through, how this file is packed.
    arrange = getattr(state, "export_arrange", None)
    wrap = max(1, int(getattr(state, "export_wrap", 1) or 1))
    merge = bool(getattr(state, "export_merge", False))
    skip_empty = bool(getattr(state, "export_skip_empty", False))
    trim = bool(getattr(state, "export_trim", False))
    padding = max(0, int(getattr(state, "export_padding", 0) or 0))
    extrude = max(0, int(getattr(state, "export_extrude", 0) or 0))
    template = str(getattr(state, "export_template", "") or "").strip() or None
    # Which of ``sheetout``'s two split templates applies -- see ``_Leg.
    # split_kind``. Every leg of one export shares it, since a batch is either
    # a tag split, a layer split or the ordinary unsplit export, never a mix.
    split_kind = export.legs[0].split_kind if export.legs else ""
    # The whole option set this export is about to run with, captured once
    # here rather than re-read per runner: what a completed export records
    # onto its tab (``on_task_done``) has to be the settings that actually
    # produced the file, not whatever the controls hold by the time the task
    # finishes.
    export_options = state.export_options_snapshot()
    if export.kind == "sheet" and padding < extrude * 2:
        # Refused here, before the file dialog, for the same reason the
        # arrange/layout and merge/layout conflicts below are: ``sheetout.build``
        # would raise this itself, but by then the user has already picked a
        # filename and the failure arrives as an opaque task error.
        tab.saving = False
        ctx.toast(
            f"Padding must be at least twice Extrude to give every sprite "
            f"room ({extrude} x 2 = {extrude * 2}, padding is {padding}).",
            "warn",
        )
        return
    # One payload per output file. A single-file export has exactly one and
    # every line below reads the same as it did before splits existed; a split
    # has one per tag or per layer, each carrying its own timing and its own
    # slices, because a sidecar has to be self-consistent with the file it is
    # beside rather than with the document the batch came from.
    loads: list[_Payload] = []
    for leg in export.legs:
        durations, tags, layout = sheetout.timing(doc, leg.span)
        # Read here, with the timing, and for its reason: it walks the document,
        # so it belongs on the frame thread beside the flatten rather than
        # inside the task. Cheap -- a handful of rectangles -- so there is
        # nothing to spread.
        #
        # Sliced by the *same* span as the frames and the timing, because
        # ``slices_block`` keys by cell index: a span export's third cell is the
        # third frame of the span, and a whole-timeline snapshot here would hang
        # frame 0's rectangles on it.
        # **The JSON meta switches** (6.9), read here with everything else the
        # sidecar is made of: a setting the user could change mid-encode must
        # not decide, halfway through, what the file says about itself.
        slices = (
            sheetout.slices_snapshot(doc, leg.span)
            if getattr(state, "export_meta_slices", True)
            else []
        )
        frames = leg.frames
        if export.kind == "sheet" and layout is not None:
            if len(frames) != layout.frame_count:
                # Refused on the frame thread, before the file dialog: the
                # engine raises the same ValueError as a backstop, but by then
                # the user has picked a filename and the failure arrives as a
                # task error with no obvious cause. A frame added to (or removed
                # from) a sprite sheet is an ordinary edit, so the fix is to say
                # which count is wrong.
                tab.saving = False
                ctx.toast(
                    f"This is a {layout.kind} sheet of {layout.frame_count} "
                    f"frames and the document has {len(frames)}.",
                    "warn",
                )
                return
            if arrange is not None:
                # The same early-refusal shape as the count mismatch above, for
                # the same reason: ``plan_frames`` would raise this itself, but
                # by then the user has already picked a filename and the failure
                # arrives as an opaque task error. A document with its own
                # directional grid keeps it -- Grid is the only Arrange choice
                # such a document has.
                tab.saving = False
                ctx.toast(
                    f"This is a {layout.kind} sheet, which keeps its own fixed "
                    "grid; set Arrange back to Grid to export it.",
                    "warn",
                )
                return
            if merge or skip_empty:
                # Same shape and same reason as the arrange/layout refusal
                # above: a directional grid's cells are poses by yaws, so there
                # is nothing for Merge or Skip empty to act on, and letting the
                # request through would have ``sheetout.compose`` raise it as an
                # opaque task error instead.
                tab.saving = False
                ctx.toast(
                    f"This is a {layout.kind} sheet, which keeps its own fixed "
                    "grid; turn Merge and Skip empty off to export it.",
                    "warn",
                )
                return
        loads.append(
            _Payload(
                label=leg.label,
                frames=frames,
                planes=leg.planes,
                durations=durations,
                tags=tags if getattr(state, "export_meta_tags", True) else [],
                layout=layout,
                slices=slices,
                loop=leg.loop,
            )
        )

    def run_sheet() -> dict[str, Any] | None:
        """Every leg composed, then every file written. Task thread.

        **Two loops rather than one, and that is the whole point.** A split is
        one export producing N files, and ``compose`` can refuse a leg the
        others are fine with -- ``skip_empty`` over a tag with nothing drawn in
        it is the reachable case, and the atlas ceiling and the padding rule are
        two more. Written inside a single loop, a refusal on leg k left legs
        0..k-1 on disk under names the user has every reason to believe, with
        the rest missing and only a toast to say so.

        The seam is the *runner's* own start rather than a pre-dialog check on
        the frame thread, deliberately. The all-empty case alone could be
        checked from the flattens ``_submit_export`` already holds -- but only
        that one: the ceiling and the padding refusals need the plan, which
        needs the compose, so a frame-thread door would leave the same
        half-written batch reachable two other ways while carrying a *second*
        copy of the emptiness rule (a second opinion about what "empty" means is
        exactly the drift ``sheetout`` centralises to avoid). Composing first
        catches every refusal ``compose`` has, present and future, with no rule
        duplicated. The precedent is ``packwright_io._write(files: dict[Path,
        bytes])`` -- encode all, then write all -- for the same reason.

        The cost is honest and bounded: N atlases live at once instead of one.
        A split is per tag or per top-level layer, so N is single digits on any
        real document, and each atlas is the sheet that leg was going to write
        anyway.
        """
        import json

        dest = export.recorded or dialogs.save_file(
            "Export sprite sheet", _suggested_dialog_name(tab, suggested, ".png"), PNG_FILTER
        )
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        stems = _split_stems(
            dest.stem, [load.label for load in loads], kind=split_kind, template=template
        )
        composed: list[tuple[str, Any, dict[str, Any]]] = []
        try:
            for stem, load in zip(stems, loads, strict=True):
                # Upscaled *before* ``compose``, so the plan is built on the
                # scaled frame size and the cells, the trims and the sidecar all
                # describe the atlas that is actually written. Scaling the
                # finished atlas instead would leave every rectangle in the
                # sidecar naming the wrong pixels. ``sheet.py`` stays the sole
                # writer of the format; none of this is new code in it.
                try:
                    image, plan, extra = sheetout.compose(
                        [upscale(plane, scale) for plane in load.frames],
                        load.durations,
                        load.tags,
                        load.layout,
                        # The slice geometry through the same magnification, or
                        # the sidecar describes a canvas that is not the atlas
                        # beside it.
                        sheetout.scale_slices(load.slices, scale),
                        name=suggested,
                        arrange=arrange,
                        wrap=wrap if arrange in ("rows", "columns") else None,
                        merge=merge,
                        skip_empty=skip_empty,
                        trim=trim,
                        padding=padding,
                        extrude=extrude,
                    )
                except ValueError as exc:
                    if not split_kind:
                        raise
                    # Which leg, by name. One file's refusal arriving as the
                    # batch's bare reason ("every frame is empty") says nothing
                    # about *which* tag or layer the user has to go and look at,
                    # and a split is exactly the export where that is the whole
                    # question.
                    raise ValueError(
                        f"{split_kind} {load.label or split_kind!r}: {exc}"
                    ) from exc
                composed.append(
                    (
                        stem,
                        image,
                        sheetlib.sidecar(
                            plan,
                            sheet_id=stem,
                            source_job=tab.job_id,
                            image=f"{stem}.png",
                            created=time.time(),
                            name=suggested,
                            trims=extra["trims"],
                            animation=extra["animation"],
                            pivots=extra["pivots"],
                            slices=extra["slices"],
                            slices_conflict=extra["slices_conflict"],
                        ),
                    )
                )
        except BaseException:
            for _stem, image, _meta in composed:
                image.close()
            raise

        dest.parent.mkdir(parents=True, exist_ok=True)
        first: Path | None = None
        for stem, image, meta in composed:
            out = dest.with_name(f"{stem}.png")
            try:
                image.save(out, "PNG")
            finally:
                image.close()
            # ``with_name`` rather than ``with_suffix``: it spells the sidecar's
            # filename directly from ``stem``, the same way ``out`` itself was
            # just built two lines up, rather than leaning on ``with_suffix`` to
            # rederive that same name by parsing it back out of ``out``'s own
            # name.
            out.with_name(f"{stem}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            if first is None:
                first = out
        return {
            "exported": first,
            "dest": dest,
            "options": dict(export_options),
            "export_kind": export.kind,
        }

    # The document's own table when it has one, so an indexed clip exports the
    # colours that were authored rather than a per-frame quantise of them. Read
    # on the frame thread here, with the frames, not inside the task.
    palette = list(doc.palette) if doc.palette else None

    def run_gif() -> dict[str, Any] | None:
        dest = export.recorded or dialogs.save_file(
            "Export animated GIF", _suggested_dialog_name(tab, suggested, ".gif"), GIF_FILTER
        )
        if dest is None:
            return None
        if dest.suffix.lower() != ".gif":
            dest = dest.with_suffix(".gif")
        stems = _split_stems(
            dest.stem, [load.label for load in loads], kind=split_kind, template=template
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        first: Path | None = None
        for stem, load in zip(stems, loads, strict=True):
            out = dest.with_name(f"{stem}.gif")
            # Upscaled before the quantiser, not after: a GIF holds palette
            # indices, so there is no "after" -- magnifying the indexed image
            # would be magnifying a palette lookup rather than a picture.
            gifout.write_gif(
                out,
                [upscale(plane, scale) for plane in load.frames],
                load.durations,
                loop=load.loop,
                palette=palette,
                # Magnified by the same whole number as the pixels, which is
                # exact on an index plane in a way it is on nothing else:
                # ``upscale`` repeats each element, so a magnified slot is still
                # that slot.
                indices=[
                    None if slots is None else upscale(slots, scale)
                    for slots in load.planes
                ],
            )
            if first is None:
                first = out
        return {
            "exported": first,
            "dest": dest,
            "options": dict(export_options),
            "export_kind": export.kind,
        }

    def run_pngs() -> dict[str, Any] | None:
        """One PNG per frame, numbered. The plainest thing an engine can eat.

        Numbered from the chosen filename's stem rather than asking for a
        directory: every tool that consumes a sequence wants ``name_0000.png``
        beside its siblings, and a save dialog is the one place a user is
        already picking both the folder and the name.

        The per-frame name goes through :func:`sheetout.filename_for` a
        second time, on top of the stem ``_split_stems`` already built: a
        split's own template (``{title}_{tag}``/``{title}_{layer}``, or a
        custom one) decides how the *outputs* of a batch differ from each
        other, and this decides how the *frames inside one output* differ --
        two questions, so a split PNG sequence always numbers its frames with
        :data:`sheetout.DEFAULT_FRAME_TEMPLATE` rather than reading the same
        custom template twice for two different things.
        """
        from PIL import Image

        dest = dialogs.save_file(
            "Export PNG sequence", _suggested_dialog_name(tab, suggested, ".png"), PNG_FILTER
        )
        if dest is None:
            return None
        stems = _split_stems(
            dest.stem, [load.label for load in loads], kind=split_kind, template=template
        )
        frame_template = (
            sheetout.DEFAULT_FRAME_TEMPLATE
            if split_kind
            else (template or sheetout.DEFAULT_FRAME_TEMPLATE)
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        first = dest
        for stem, load in zip(stems, loads, strict=True):
            names = [
                sheetout.filename_for(frame_template, title=stem, frame=index)
                for index in range(len(load.frames))
            ]
            sheetout.require_distinct_names(names)
            for name, plane in zip(names, load.frames, strict=True):
                out = dest.parent / f"{name}.png"
                Image.fromarray(upscale(plane, scale), "RGBA").save(out, "PNG")
                if first is dest:
                    first = out
        return {
            "exported": first,
            "dest": dest,
            "options": dict(export_options),
            "export_kind": export.kind,
        }

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
    _settle(ctx, tab)
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

    The ``.aseprite`` branch alone is wrapped: ``aseout`` refuses a *broken*
    document by name -- a tilemap cel bound to a tileset it does not have, a
    strip pixel an indexed palette has no slot for -- and a bare ``ValueError``
    from a task lands in ``tasks.py``'s generic branch, whose sentence points
    at the log rather than at what was actually wrong (the log still gets the
    traceback either way). ``write_ora`` never refuses by name this way, so it
    is left alone -- wrapping it would only ever catch a bug, not a named
    refusal, and ``invalid_from`` exists for the latter. Same idiom as
    ``import_tileset``'s own wrap, just on the write side of the same format.
    """
    from . import inker

    if file_format == "ora":
        inker.write_ora(doc, path)
    elif file_format == "aseprite":
        from ..service.errors import invalid_from

        try:
            inker.write_aseprite(doc, path)
        except ValueError as exc:
            raise invalid_from(exc, "This drawing could not be saved as .aseprite") from exc
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
    _settle(ctx, tab)
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
    stop_play(tab)  # settle the stack before capturing; see save()
    doc, title = tab.doc, tab.title
    _settle(ctx, tab)  # before the head is read; see _save_linked
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
    stop_play(tab)  # settle the stack before capturing; see save()
    doc = tab.doc
    _settle(ctx, tab)
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
            _report_import_warnings(ctx, result.get("warnings"))
        return

    if name == "inker-sheetin":
        # The picture only. The grid comes from the popup the bridge panel
        # opens on the next frame, which is why nothing is adopted here.
        if isinstance(result, dict):
            state.sheet_import = (result["atlas"], result.get("title") or "Sheet")
            state.sheet_import_open = False
            suggested = result.get("suggest")
            if suggested is not None:
                # Only when a suggestion actually fired. The three fields persist
                # across imports on purpose -- a folder of sheets cut the same way
                # is typed once -- so a ``None`` detection must keep the last
                # values rather than resetting them to the defaults.
                state.sheet_cell, state.sheet_offset, state.sheet_padding = suggested
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

    if name in ("inker-palette-export", "inker-palette-export-doc"):
        # Said out loud. Neither key had a branch here at all, so a palette
        # export fell through to the uid-keyed tail below, found no ``:`` in its
        # key and returned -- reporting neither success nor failure. A write to
        # a path the user chose is exactly the kind of thing that has to answer.
        if result:
            ctx.toast(f"Palette written to {Path(result).name}.", "success")
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

    if name == "inker-palimg":
        # A palette read out of an image. Resolved through the uid for
        # ``inker-index``'s reason, and a key of its own because the *count*
        # comes back with it: a median cut is a loss, and one that happened
        # silently is one the user finds out about by noticing their drawing
        # looks wrong.
        if isinstance(result, dict) and result.get("colours"):
            colours = result["colours"]
            if index_to(ctx, state.get(key.split(":", 1)[1]), colours):
                distinct = int(result.get("distinct", 0))
                if distinct > len(colours):
                    ctx.toast(
                        f"{distinct} colours reduced to {len(colours)}.", "warn"
                    )
        return

    if name == "inker-tileset-import":
        # A tileset for a *document*, resolved through the uid rather than
        # through ``active`` -- ``inker-index``'s reason: a native picker is
        # unbounded, and the user may well have switched tabs while it was up.
        if isinstance(result, dict) and result.get("tileset") is not None:
            target = state.get(key.split(":", 1)[1])
            # ``busy`` re-checked at completion, the way the sibling
            # ``inker-index``/``inker-palimg`` branches land through
            # ``index_to``'s own gate: the picker is unbounded, and a save or
            # playback may have started on this tab while it was up --
            # ``add_tileset`` pushes a history step into a stack an encode is
            # walking.
            if target is not None and not target.busy:
                slot = target.doc.add_tileset(result["tileset"])
                ctx.toast(f"{slot.tileset.name} added.", "success")
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
        # Only the sheet/GIF/PNG-sequence runners hand back "dest"/"options"
        # (``export_png``'s flat export does not) -- present, this is what
        # ``InkerState.apply_export_options`` reads back on this tab's next
        # export, and what ``_suggested_dialog_name`` reads back for its
        # folder.
        dest = result.get("dest")
        if dest is not None:
            tab.export_dest = dest
            # Which *kind* of export wrote it, so Repeat Last Export (6.9) can
            # run the same one again. Recorded here rather than at the click
            # because a click that ended in a cancelled file dialog is not an
            # export to repeat.
            kind = result.get("export_kind")
            if kind:
                tab.export_kind = str(kind)
        options = result.get("options")
        if isinstance(options, dict):
            tab.export_options = dict(options)
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
        doc = inker.Document.load(tab.path)
    except Exception as exc:
        ctx.toast(f"Reverted, but the image could not be reopened ({exc}).", "error")
        return
    from .panes import inker_textures

    # The old document's textures go with it -- ``request_close``'s own pair
    # of calls, minus the tab removal. Without this the swap leaked every
    # layer texture until the tab was closed; the new document re-uploads
    # lazily on its first draw.
    inker_textures.release_doc(ctx, tab.uid)
    tab.doc = doc
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
        # The owner, by uid -- ``convert_uid``'s pattern. The modal lives on
        # this document, and Enter/Escape must land on it and no other; see
        # ``InkerState._settle_transform`` for the switch-away half.
        state.transform_uid = tab.uid
        state.clear_drag()
        _warn_rotsprite(ctx, state, tab)
    elif tab.doc.write_locked():
        # The one refusal worth saying out loud from here: a transform lifts,
        # and a lift is a cut. Every other way of reaching a locked layer goes
        # through the canvas, which raises its own toast on the press.
        ctx.toast(LOCKED_LAYER, "warn")


def _warn_rotsprite(ctx: Any, state: Any, tab: InkerDoc) -> None:
    """Say once, at the start of the gesture, that RotSprite will not be used.

    Once and here rather than at the engine's fallback, because the fallback is
    reached on every mouse-move of a rotate drag: a toast per frame would be
    the loudest bug in the editor. The engine falls back silently for exactly
    that reason -- see ``transform.ROTSPRITE_MAX_PIXELS``.
    """
    from .inker import transform

    buf = tab.doc.floating
    if state.resample != "rotsprite" or buf is None:
        return
    if transform.rotsprite_fits(buf.size):
        return
    ctx.toast(
        "Too big for RotSprite -- rotating with nearest neighbour instead.", "warn"
    )


def end_transform(ctx: Any, *, commit: bool) -> None:
    """Enter, or a click outside: land the transform, or take it back.

    **The commit is where a timeline range takes effect.** The preview only
    ever showed the active cel -- the buffer holds that cel's pixels -- so
    Aseprite's timeline-target behaviour has to happen here, and the visible
    range outline is what tells the user how far it will reach.
    ``commit_floating_range`` falls back to the plain commit whenever a range
    means nothing (a paste, a still document, a rect off the grid), so this
    call site does not have to know which case it is in.

    A **cancel** is deliberately not ranged. Nothing but the active cel was
    ever written, so there is nothing else to put back.
    """
    state = ensure(ctx)
    # The owner, never ``active``: the two are the same tab whenever the modal
    # is genuinely open (``_settle_transform`` cancels on every switch), but
    # resolving by uid is what makes that a fact rather than a hope --
    # committing ``active`` here is exactly how a mid-transform switch came to
    # land one tab's Enter on another tab's floating buffer.
    tab = state.get(state.transform_uid) or state.active
    state.transforming = False
    state.transform_uid = ""
    state.clear_drag()
    if tab is None:
        return
    if not commit:
        tab.doc.cancel_floating()
        return
    rect = tab.range_sel
    if rect is None:
        tab.doc.commit_floating()
    else:
        tab.doc.commit_floating_range(*rect)


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


# --- the multi-click gesture (C4) --------------------------------------------
#
# The vertices live on the state (``InkerState.gesture_pts``) and the clicks are
# collected by the canvas; what is here is the landing, because Enter finishes a
# polygon and a keypress cannot reach into the pane.


def polygon_select(doc: Any, points: Any, op: str = "replace") -> bool:
    """Turn a run of image-space vertices into the document's selection.

    The lasso's landing, extracted so the freehand drag and the polygonal
    gesture cannot come to disagree about it: they differ in how the vertices
    are *collected* and in nothing else, and a second copy of "three points, a
    polygon, one ``select``" is exactly where a rasteriser change lands half
    applied.

    Three vertices is the floor because two are a line -- the rasteriser would
    return an empty mask and ``select`` would push a step that selects nothing.
    -> whether a polygon was committed.

    One ``select`` and therefore exactly one undo step, however many clicks or
    mouse-moves the vertices cost to draw.
    """
    from .inker import SelectionMask

    points = list(points)
    if len(points) < 3:
        return False
    doc.select(SelectionMask.from_polygon(doc.size, points), op)
    return True


def paint_path(state: InkerState, tab: InkerDoc, tool: str, points: Any) -> bool:
    """The multi-click gesture's *other* landing: a stroked path (Q-c).

    ``polygon_select``'s opposite number. The poly lasso's clicks become a
    selection; the polyline, polygon and curve tools' become **paint**, through
    the same ``PaintOps`` machinery the dragged shapes go through -- so the
    selection clip, the alpha lock, the indexed snap, tiled wrapping and the
    single undo step are all the ones the rest of the toolbox already has, and
    none of them is reimplemented here.

    The floor is the poly lasso's, and for its reason: two clicks make a
    polygon that is a line, and a tool asked to close a shape that does not
    enclose anything commits nothing rather than leaving a mark the user has to
    find and undo. An open path needs only its two points.

    Never the background colour: a right press on a shape tool is inert and
    never opens a gesture at all (see ``BG_BUTTON_TOOLS``).
    -> whether anything was painted.
    """
    points = list(points)
    if len(points) < (3 if tool == "polygon" else 2):
        return False
    return bool(
        tab.doc.shape_path(
            tool,
            points,
            state.fg,
            state.brush_size,
            filled=state.shape_filled,
            wrap=tab.tiled,
        )
    )


def commit_gesture(state: InkerState, tab: InkerDoc) -> bool:
    """Land the open multi-click gesture and close it. -> whether it landed.

    Three callers, all of which mean "finish it": the canvas's double-click and
    its click near the first vertex, and Enter. The gesture is closed either
    way -- a polygon of two vertices has nothing to select, and cancelling is
    the honest answer where deselecting would throw away a selection the user
    never asked to lose.

    ``busy`` cancels rather than commits, for the reason the canvas refuses
    input outright while a save is encoding the live document. That is the
    whole of the busy story for the painting shapes too (Q-c): a commit that
    arrived mid-save would put half a polygon in the file, so it does not
    arrive, and the vertices go with it rather than waiting to be finished
    against a document that has moved on.

    Which landing is decided by the tool **at the commit**, and that is safe
    precisely because ``set_tool`` clears the gesture: the tool holding the
    vertices is always the tool that placed them.
    """
    points, op = state.gesture_pts, state.gesture_combine
    tool = state.tool
    state.clear_gesture()
    if tab.busy or not points:
        return False
    tab.doc.commit_floating()
    if tool in inker_state.PATH_SHAPE_TOOLS:
        return paint_path(state, tab, tool, points)
    return polygon_select(tab.doc, points, op)


# --- the text tool ----------------------------------------------------------
#
# The controller half of C14: which fonts there are, and turning what the popup
# holds into a floating buffer. The rasteriser is ``inker/textstamp.py`` and
# knows nothing about any of this -- it takes a path.

#: Where Windows keeps its faces. A literal rather than a lookup through
#: ``%WINDIR%`` because it is a *fallback list*, not a resource the app needs:
#: on a machine where this directory is not there (a test box, another OS) the
#: scan comes back with the vendored face alone and the tool still works.
SYSTEM_FONT_DIR = Path("C:/Windows/Fonts")

#: What the scan will offer. ``.ttc`` collections are included and loaded at
#: face 0, which is the regular weight in every collection Windows ships.
FONT_SUFFIXES = (".ttf", ".otf", ".ttc")

#: The scan's answer, or None before the first one. Cached because the
#: directory holds several hundred files and the popup that reads it is drawn
#: every frame it is up; refreshed only when a caller asks, since a font
#: installed mid-session is rare and a stale list costs one restart.
_font_choices: list[tuple[str, str]] | None = None


def font_choices(*, refresh: bool = False) -> list[tuple[str, str]]:
    """``(path, label)`` for every font the text popup offers.

    The vendored Inter face is first and is the default, so the tool works
    identically on a machine with no fonts installed and on one with four
    hundred -- and so the *documented* default is a file that ships in the
    wheel rather than whatever a directory listing happens to sort first. It is
    checked for rather than assumed: it is a vendored asset and a missing one
    should degrade to the system list, not to a traceback.
    """
    global _font_choices
    if _font_choices is not None and not refresh:
        return _font_choices
    found: list[tuple[str, str]] = []
    vendored = fonts.FONT_DIR / "Inter-Regular.ttf"
    if vendored.exists():
        found.append((str(vendored), "Inter (vendored)"))
    try:
        system = sorted(
            (
                (str(path), path.stem)
                for path in SYSTEM_FONT_DIR.iterdir()
                if path.suffix.lower() in FONT_SUFFIXES
            ),
            key=lambda entry: entry[1].lower(),
        )
    except OSError:
        # No such directory, or no permission to read it. Either way the
        # vendored face is still there and the tool is still usable.
        system = []
    found.extend(system)
    _font_choices = found
    return found


def font_path(state: Any) -> str:
    """The path the stamp will be rendered from. Empty means the default.

    The stored option is a path and not an index, because an index into a
    directory listing means something different the day a font is installed.
    """
    if state.font:
        return str(state.font)
    choices = font_choices()
    return choices[0][0] if choices else ""


def stamp_text(ctx: Any, state: Any, tab: InkerDoc) -> bool:
    """Rasterise what the popup holds and float it. -> whether it landed.

    Two refusals, each with its own sentence, because they are different
    problems: nothing came out of the rasteriser (empty text, a font file the
    system lists but Pillow cannot read, a size the face has no outline at), or
    the layer is locked. The second is the canvas's own wording, because it is
    the same lock and a user should not have to learn that two messages mean
    one thing.

    On success the tool becomes Move -- the Ctrl+V precedent. A stamp arrives
    floating, and the first thing anybody does with a word they have just
    placed is drag it into position; leaving the text tool in hand would mean
    that drag opens a second popup.
    """
    from .inker import textstamp

    if tab.busy:
        return False
    if getattr(state, "text_uid", "") and state.text_uid != tab.uid:
        # The press that set ``text_at`` belongs to another document. The popup
        # closes itself on a tab switch, so this is the belt to that braces: a
        # stamp is a write, and a write at coordinates from a different picture
        # is the kind of thing that must be refused at the door rather than
        # relied on being unreachable.
        return False
    pixels = textstamp.text_stamp(
        state.text_buffer,
        font_path(state),
        int(state.text_size),
        state.fg,
        antialias=bool(state.aa),
    )
    if pixels is None:
        ctx.toast("Nothing to stamp -- check the text and the font.", "warn")
        return False
    if not tab.doc.float_pixels(pixels, state.text_at):
        ctx.toast(LOCKED_LAYER, "warn")
        return False
    # Through ``set_tool``: the one door, so the stamp cannot leave a
    # half-drawn poly-lasso gesture open behind the Move tool (C4).
    state.set_tool("move")
    return True


# --- keys -------------------------------------------------------------------

# Aseprite's letters where they exist, because that is the muscle memory a user
# arrives with. Held here rather than in the pane so the mapping is testable.
#: Letter to tool, **derived from the toolbox** rather than written out again.
#:
#: It was written out again, and that is a table of twenty-three entries kept in
#: step with another table of twenty-three entries by hand -- the arrangement
#: ``plotter_state`` had already replaced with this one line. Every letter and
#: the reasoning for it now lives in exactly one place, ``inker_state.TOOLS``,
#: where the tooltip and the manual read it from too.
TOOL_KEYS = {letter.lower(): key for key, _label, letter in inker_state.TOOLS}

#: Aseprite's *slot-mates*, as a second binding rather than a replacement.
#:
#: Aseprite files several tools two-to-a-slot and cycles them with Shift: the
#: gradient sits on the paint bucket's, the ellipse on the rectangle's, the
#: elliptical marquee on the rectangular one's. Inker gives every tool a letter
#: of its own (one per ``TOOLS`` row, no cycling), so these
#: are added *beside* the plain letters, not instead of them -- a hand trained
#: on Aseprite finds what it reaches for, and a hand trained here keeps what it
#: had. Reconstructed from Aseprite's defaults; if one is wrong it is wrong in
#: one dict.
SHIFT_TOOL_KEYS = {
    "g": "gradient",
    "u": "ellipse",
    "m": "select_ellipse",
}

#: How the toolbox writes a second binding, tool to the chord.
ALT_TOOL_CHORDS = {
    tool: f"Shift+{letter.upper()}" for letter, tool in SHIFT_TOOL_KEYS.items()
}

#: How far a Shift+arrow nudge moves the active layer, in pixels. Eight rather
#: than a grid multiple: a nudge is about the *drawing*, not about the grid
#: overlay, and eight is the step every editor with this shortcut uses.
NUDGE_STEP = 8


def nudge(state: Any, tab: InkerDoc, dx: int, dy: int) -> bool:
    """Move by a whole pixel from the keyboard. -> whether anything moved.

    A floating buffer first, because that is what the arrow keys visibly point
    at while one is up; otherwise the move tool's third arm, opened, previewed
    and committed inline so a press is exactly one undo step rather than a
    session left half open waiting for a release that never comes.

    Gated on the move tool (or a float) rather than global: the arrows are the
    only keys left for a document pane to give away, and quietly translating a
    layer because somebody pressed Right with the brush selected is not a
    trade worth making.
    """
    doc = tab.doc
    if tab.busy:
        return False
    if doc.floating is not None:
        doc.move_floating(dx, dy)
        return True
    if state.tool != "move" or not doc.begin_layer_move():
        return False
    doc.preview_layer_move(dx, dy)
    return doc.commit_layer_move()


# --- playback ----------------------------------------------------------------

#: The one sentence **every** door says. It was written out at the canvas press
#: and nowhere else, which is how the keyboard's copies of the same refusal came
#: to be silent: there was nothing to reuse and no reason to notice.
#:
#: ``begin_transform`` and ``stamp_text`` used to spell their own -- "Unlock it
#: in the layers panel" -- which was two problems at once. It was a second
#: wording for one lock, which ``stamp_text``'s own docstring says it must not
#: be ("a user should not have to learn that two messages mean one thing"), and
#: it named a pane that does not exist: ``panes/inker_timeline`` opens with
#: "There is no layers panel." The padlock is on the timeline row.
LOCKED_LAYER = "That layer is locked. Its padlock is on its timeline row."

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
    if not tab.saving:
        # ``set_current_frame`` rebuilds the layer stack, which is exactly the
        # structure an in-flight encode is walking -- the ``_MUTATING_CTRL``
        # rule, applied to the one mutation stopping playback itself makes.
        # Unreachable while the saves below stop playback before capturing and
        # ``toggle_play`` refuses to start during one; kept as the backstop,
        # at the cost of the playhead resting where play began.
        tab.doc.set_current_frame(tab.play_index)
    tab.play_accum_ms = 0.0


def frame_durations(tab: InkerDoc, anim: Any) -> list[int]:
    """The durations a playhead steps by, Constant Frame Rate included.

    **Constant Frame Rate** (6.7) is Aseprite's own playback switch: play every
    frame at one rate rather than at the durations the document stores. It is a
    *preview* setting and it does not touch the frames -- what an animator is
    asking is "what does this look like at 12 fps", not "make every frame
    83 ms", and answering the second would be an undoable edit to every frame.

    One function because there are two playheads. ``tick_preview`` is a clone of
    ``tick_playback`` rather than a share, deliberately -- it must never touch
    ``playing``, ``saving`` or ``set_current_frame`` -- but the clone had copied
    the plain duration list and not the switch above it, so turning Constant
    Frame Rate on left the timeline playing at 12 fps and the preview pane
    playing at the stored durations. Two playheads disagreeing about one clip is
    the drift a clone invites, so the one part they genuinely share lives here.
    """
    durations = [frame.duration_ms for frame in anim.frames]
    rate = getattr(tab, "constant_rate", 0)
    if rate:
        held = max(1, round(1000.0 / float(rate)))
        return [held] * len(durations)
    return durations


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
    index, accum, playing, forward, cycles = animation.advance(
        frame_durations(tab, anim),
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
        frame_durations(tab, anim),
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
    state = ctx.state.inker
    if state is not None and (state.drag_kind or state.gesture_pts):
        # ``set_current_frame`` rebuilds the layer stack, and an open paint
        # drag holds a ``StrokeState`` addressed into the stack it began on --
        # the next ``stroke_to``/``end_stroke`` raises out of ``by_uid``. A
        # multi-click gesture's vertices likewise belong to the frame they
        # were placed on. Refused like the ``_MUTATING_CTRL`` set: the gesture
        # finishes first.
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
    if state is None:
        return False

    if event.key == pygame.K_SPACE:
        # Seen on both edges: space-to-pan is a hold, not a toggle.
        #
        # **Above the "is there a document" returns below**, which is the whole
        # of the fix: they used to sit in front of this, so holding Space and
        # closing the last tab dropped the release and left the flag on for the
        # rest of the session -- every left-drag panned instead of painting, and
        # every tool press was suppressed. A release is honoured whether or not
        # there is anything to pan, because the flag outlives the document.
        # ``plotter_mode.handle_key`` learned the same lesson at its own door.
        state.space_held = event.type == pygame.KEYDOWN
        return bool(state.docs)
    if not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc
    if event.type != pygame.KEYDOWN:
        return True

    # ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (main.py:2340), and Inker was the last mode still breaking it.
    # ``mod`` is the modifier state at the moment this key was *pressed*;
    # ``get_mods()`` is the state now, after the event batch drained, so a
    # Ctrl released between the press and this call read as never held.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    alt = bool(mods & pygame.KMOD_ALT)
    name = pygame.key.name(event.key)
    # Built here rather than at module scope: pygame is imported lazily in this
    # function, so a module-level table would drag it into every import of the
    # mode. Four entries costs nothing per keypress.
    arrows = {
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
    }

    # **The modal arms are a table now** (W2.8). Which situation the keyboard is
    # in is ``inker_state.key_context``, first-match-wins over one tuple, so the
    # contexts are mutually exclusive by construction instead of by three
    # branches each remembering the other two -- which is how Enter came to mean
    # "apply the transform", "close the polygon" and "play" in one function with
    # the order of the ifs as the only thing keeping them apart.
    context = inker_state.key_context(state, tab)
    if _modal(ctx, state, tab, context, name, event, ctrl=ctrl):
        return True

    # **The registry answers first.** Every op that carries a key is bound
    # here, once, from the same field the menu row prints -- so a chord cannot
    # be advertised in one place and implemented in another, which is what
    # eleven of these branches used to be. What is left below is the bindings
    # that are not ops: the tool letters, the sizes, the nudges, the modeless
    # view keys.
    op = inker_ops.by_key(chord_of(event, ctrl=ctrl, shift=shift, alt=alt), context)
    if op is not None:
        if not (tab.busy and ctrl and name in _MUTATING_CTRL):
            inker_ops.run(ctx, op)
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, event, shift=shift)

    if name in SHIFT_TOOL_KEYS and shift:
        # Ahead of the plain branch, and gated on shift there, so the two
        # cannot both fire on one press.
        state.set_tool(SHIFT_TOOL_KEYS[name])
    elif name in TOOL_KEYS and not shift:
        # Through ``cycle_in_group``: the first press is the binding this
        # letter has always had, and a second press moves along the group.
        state.set_tool(inker_state.cycle_in_group(state.tool, TOOL_KEYS[name]))
    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        # Aseprite's zoom in, and ``=`` unshifted answers it too because that
        # is the same physical key on every layout this ships to.
        tab.view.pending_zoom_rung = 1
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        tab.view.pending_zoom_rung = -1
    elif event.key == pygame.K_TAB:
        # Aseprite's binding, and the reason the timeline can be hidden at all
        # now that it holds the layers: one key, always the same one, whether
        # the document is animated or not. Ctrl+Tab still cycles documents --
        # that branch is in ``_ctrl_key`` and cannot be reached from here.
        from .panes import inker_timeline

        inker_timeline.toggle(state)
    elif name == "x":
        state.swap_colours()
    elif alt and name.isdigit() and name != "0":
        # Aseprite's Alt+1..9. Shift stores, plain recalls -- Plotter's stamp
        # slots' rule and its reason: recall happens hundreds of times a
        # session and storing nine times, so the cheap gesture goes to the
        # frequent one.
        slot = int(name)
        if shift:
            if not state.store_stamp(slot):
                state.say("There is no captured brush to store -- Ctrl+B captures one.")
        elif not state.recall_stamp(slot):
            state.say(f"Brush {slot} is empty -- Alt+Shift+{slot} stores one.")
    elif not shift and name.isdigit():
        # **The number row was entirely unbound in Inker**, and Aseprite's
        # answer to the same spare keys is the same one: opacity in tenths,
        # with 0 meaning full rather than nothing -- a key that made the brush
        # invisible would be one nobody could tell from a broken tool.
        digit = int(name)
        state.opacity = 1.0 if digit == 0 else digit / 10.0
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
    elif event.key in arrows:
        step = NUDGE_STEP if shift else 1
        dx, dy = arrows[event.key]
        # **The return value is read now.** A nudge onto a locked layer came
        # back False and the answer was thrown away, so the arrows did nothing
        # and said nothing -- while the *same* refusal reached by a mouse press
        # raised a toast. The two doors gave two different answers to one
        # question, and the quiet one is the one a user meets by accident.
        # Only when the lock is what actually refused it, though: ``nudge``
        # also declines for a busy tab and for a tool the arrows do not serve
        # (no floating buffer, not the move tool), and blaming the lock for
        # either would be a toast naming the wrong problem.
        if (
            not nudge(state, tab, dx * step, dy * step)
            and not tab.busy
            and (state.tool == "move" or doc.floating is not None)
            and doc.write_locked()
        ):
            state.say(LOCKED_LAYER, remedy="layer_properties", remedy_label="Unlock")
    elif event.key == pygame.K_DELETE:
        if not tab.busy and not doc.delete_selection() and doc.write_locked():
            state.say(LOCKED_LAYER, remedy="layer_properties", remedy_label="Unlock")
    elif event.key == pygame.K_ESCAPE:
        # Never leaves the mode: Esc means "drop what I am doing", and losing a
        # workspace full of tabs to a stray keypress is not that.
        # The move session goes back **unconditionally**, beside ``clear_drag``
        # below and for a stronger version of its reason. It is the one open
        # gesture that has already *written* previewed pixels into the layer
        # with no undo step behind them, so dropping the drag state without it
        # leaves the layer moved, clean and unrecoverable -- and mid-save those
        # pixels are exactly what the encoder is reading off the live document,
        # so they reach the file. Cancelling puts back only what this session
        # itself wrote, which makes it as safe mid-save as abandoning the drag.
        moved = doc.cancel_layer_move()
        if tab.playing:
            stop_play(tab)
        elif not tab.saving and not moved:
            # Only when the move did not already answer the keypress: Esc means
            # "drop the one thing I am doing", not "unwind everything at once".
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
# ``j`` joins them for the ordinary reason: layer-from-selection adds a layer
# (a track, on an animated document) and may cut pixels out of another.
_MUTATING_CTRL = frozenset({"z", "y", "a", "d", "x", "v", "i", "t", "e", "j"})


#: The keys whose chord label is not simply the character on them.
#:
#: Written out rather than derived from ``pygame.key.name``: the names it
#: returns are lowercase and platform-shaped ("return", "left", "escape"), and
#: the label is what a menu row and the shortcut sheet *print*, so the two have
#: to be the same string. ``Op.key`` is both the binding and the label for
#: exactly this reason.
_CHORD_NAMES: dict[str, str] = {
    "return": "Enter",
    "enter": "Enter",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "tab": "Tab",
    "escape": "Esc",
    "delete": "Delete",
    "backspace": "Backspace",
    "space": "Space",
    "[": "[",
    "]": "]",
    ",": ",",
    ".": ".",
}


def chord_of(event: Any, *, ctrl: bool, shift: bool, alt: bool = False) -> str:
    """The chord this key press *is*, in ``Op.key``'s spelling, or ``""``.

    One spelling, used by the binding and by the printed label, so a menu row
    can never advertise a chord the keyboard does not answer -- which is the
    whole reason ``Op.key`` is one field rather than two.
    """
    import pygame

    name = pygame.key.name(event.key).lower()
    label = _CHORD_NAMES.get(name)
    if label is None and len(name) == 1:
        label = name.upper()
    if label is None:
        return ""
    parts = []
    if ctrl:
        parts.append("Ctrl")
    if alt:
        parts.append("Alt")
    if shift:
        parts.append("Shift")
    parts.append(label)
    return "+".join(parts)


def _modal(
    ctx: Any, state: InkerState, tab: InkerDoc, context: str, name: str, event, *, ctrl: bool
) -> bool:
    """Enter and Escape, answered by the context they are pressed in.

    -> whether the key was consumed *and* the rest of ``handle_key`` skipped.

    Only the three modal contexts appear here; every other context falls
    through to the ordinary bindings, which is what "modal" means. A
    transformation swallows every key, not just these two: nothing may change
    the tool out from under a half-finished one.
    """
    import pygame

    enter = event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
    escape = event.key == pygame.K_ESCAPE
    if context == "Transformation":
        if enter:
            end_transform(ctx, commit=True)
        elif escape or (ctrl and name == "z"):
            # Ctrl+Z during a transform means "undo the transform", which is
            # cancelling it -- not stepping back through the history behind it.
            end_transform(ctx, commit=False)
        return True
    if context == "Gesture":
        # An open multi-click gesture answers Enter and Escape before anything
        # else does, and consumes them: Enter would otherwise start playback on
        # an animated document, and Escape would drop the *previous* selection
        # while leaving the half-drawn polygon on screen. Ahead of the tool
        # letters as well, so neither can be reached with a gesture open.
        if enter:
            commit_gesture(state, tab)
            return True
        if escape:
            state.clear_gesture()
            state.clear_drag()
            return True
    return False


def _ctrl_key(
    ctx: Any, state: InkerState, tab: InkerDoc, doc: Any, name: str, event, *, shift: bool
):
    import pygame

    # ``busy``, not ``saving``: playback is the second reason the document may
    # not be restructured, and it is the same list of keys for the same reason.
    if tab.busy and name in _MUTATING_CTRL:
        return True

    if name == "z" and shift:
        # Ctrl+Shift+Z is redo's second spelling, which the registry does not
        # carry: an op has one key, and Ctrl+Y is the one the menu prints.
        doc.redo()
    elif event.key == pygame.K_TAB:
        state.cycle(-1 if shift else 1)
    # Ctrl+Shift+E, Ctrl+Shift+D and Ctrl+Shift+J used to have branches here.
    # All three are ops (``export_png``, ``reselect``, ``move_to_layer``), and
    # the registry is consulted before this function is reached -- so the
    # branches had been unreachable since those ops gained their keys, and a
    # reader would have had to check the registry to know it.
    #
    # Ctrl+4 and Ctrl+5 went the same way. Their *shifted* halves were the
    # awkward part: Ctrl+Shift+4 rotated the other way and Ctrl+Shift+5 was a
    # silent alias for Ctrl+5, neither advertised by any ``Op.key`` -- which is
    # the one thing that field exists to prevent. The reverse rotation is now
    # ``rotate_view_back``, printed on its menu row like every other binding,
    # and the alias is gone.
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

# Both text palette formats behind one filter, the combined pattern first so the
# picker opens on it. ``.pal`` is JASC's text form only -- see ``gpl.parse_jasc``
# for why the other two things called ``.pal`` are refused rather than guessed.
PALETTE_FILTER = [
    "Palettes (*.gpl *.pal)",
    "*.gpl *.pal",
    "GIMP palette (*.gpl)",
    "*.gpl",
    "JASC palette (*.pal)",
    "*.pal",
]


def _write_palette(path: Any, colours: list[tuple[int, int, int, int]], name: str) -> None:
    """Write *colours* in the format the chosen filename asks for.

    The suffix decides, and a name with no suffix at all gets ``.gpl`` -- the
    picker's filter list does not tell us which entry was selected, and the
    filename is the only thing the user actually said.

    ``newline=""`` is load-bearing on Windows, not tidiness. Python's text mode
    defaults to ``newline=None``, which rewrites every line feed it is handed as
    ``os.linesep`` -- so ``dumps_jasc``'s already-correct CRLF reached the disk
    as CR CR LF. Our own ``parse_jasc`` reads that file back perfectly, because
    ``splitlines`` shrugs at anything, which is precisely the trap: the only
    reason to write this format at all is the strict third-party readers that
    will not. Line endings are the serialiser's decision and are made once, in
    ``gpl``; this call writes exactly what it was handed.
    """
    from .inker import gpl

    dest = Path(path)
    if dest.suffix.lower() not in (".gpl", ".pal"):
        dest = dest.with_suffix(".gpl")
    dest.write_text(
        gpl.dumps_for(dest.suffix, colours, name), encoding="utf-8", newline=""
    )


def import_palette(ctx: Any) -> None:
    from .inker import gpl

    ensure(ctx)

    def run() -> list[tuple[int, int, int, int]] | None:
        path = dialogs.open_file("Import a palette", PALETTE_FILTER)
        if path is None:
            return None
        return gpl.parse_any(path.read_text(encoding="utf-8", errors="replace"))

    ctx.submit("inker-palette", run)


def export_palette(ctx: Any) -> None:
    state = ensure(ctx)
    colours = list(state.swatches)

    def run() -> str | None:
        path = dialogs.save_file("Export the palette", "palette.gpl", PALETTE_FILTER)
        if path is None:
            return None
        _write_palette(path, colours, "Warlock")
        return str(path)

    # A key of its own. It used to share ``inker-palette-export`` with
    # ``export_document_palette``, and ``tasks.submit`` refuses a duplicate key
    # -- so whichever picker was already up made the other command do nothing at
    # all, silently, because neither call site reads the bool it answers.
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


#: What the mode picker offers, in the order it draws them.
COLOR_MODES = ("rgb", "indexed", "grayscale")

#: How each mode is written on a button.
COLOR_MODE_LABELS = {"rgb": "RGB", "indexed": "Indexed", "grayscale": "Grayscale"}


def set_color_mode(ctx: Any, tab: Any, mode: str, *, max_colours: int = 32) -> bool:
    """Move a document between RGB, true indexed and grayscale. -> whether it moved.

    One door for all three, because they are one question and because the
    refusals belong together: each conversion is a whole-document rewrite, one
    undo step, and inline on the frame thread for ``index_to``'s reason (a
    partial rewrite landing mid-save writes an archive whose parts disagree).

    Entering **indexed** with no palette builds one from the drawing's own
    colours, exactly as ``palette_from_document`` does for constrained mode --
    two published operations and no third one. Entering it *with* a palette
    keeps the table the user authored.
    """
    if tab is None or tab.busy or mode not in COLOR_MODES:
        return False
    state = ensure(ctx)
    doc = tab.doc
    if doc.color_mode == mode:
        return False
    try:
        if mode == "indexed":
            moved = doc.convert_to_indexed(
                doc.palette or None, "nearest", max_colours=max_colours
            )
        elif mode == "grayscale":
            moved = doc.convert_to_grayscale()
        else:
            moved = doc.convert_to_rgb()
    except ValueError as exc:
        # By name, and with the attempt in front of it: every refusal this can
        # raise is about the palette the user can see (too many colours, a
        # transparent index naming no slot), and a silent False would leave a
        # button that does nothing. The frame is the house rule -- library text
        # with no subject makes the reader work out what was being tried.
        ctx.toast(f"Cannot switch to {COLOR_MODE_LABELS[mode]}: {exc}.", "warn")
        return False
    if not moved:
        return False
    state.palette_slot = 0
    state.palette_slots = []
    state.palette_usage = None
    state.fg_slot = None
    if mode == "indexed":
        ctx.toast(
            f"Indexed: {len(doc.palette)} colours, slot {doc.transparent_index}"
            " is transparent.",
            "success",
        )
    elif mode == "grayscale":
        ctx.toast("Grayscale. Every write lands on a grey from here.", "success")
    else:
        # Worth saying, because the pixels do not move: leaving a mode lifts a
        # constraint, and the drawing looks exactly as it did a moment ago.
        ctx.toast("RGB colour. The pixels are unchanged.")
    return True


def set_transparent_slot(ctx: Any, tab: Any, index: int) -> bool:
    """Move which palette slot means "hole". Indexed documents only."""
    if tab is None or tab.busy or not tab.doc.set_transparent_index(index):
        return False
    ensure(ctx).palette_usage = None
    ctx.toast(f"Slot {index} is transparent now.", "success")
    return True


def index_to(ctx: Any, tab: Any, colours: Any) -> bool:
    """Make *tab* indexed against *colours*, or plain RGBA with ``None``."""
    if tab is None or tab.busy:
        return False
    state = ensure(ctx)
    if not tab.doc.set_palette(colours):
        return False
    state.palette_slot = 0
    state.palette_slots = []
    state.palette_usage = None
    # ``set_color_mode`` clears it too, and for the same reason: the slot the
    # brush was claiming indexes a table that has just been replaced, so left
    # standing it would land the next stroke in whatever colour inherited
    # that number.
    state.fg_slot = None
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
        path = dialogs.open_file("Index to a palette", PALETTE_FILTER)
        if path is None:
            return None
        return gpl.parse_any(path.read_text(encoding="utf-8", errors="replace"))

    ctx.submit(f"inker-index:{tab.uid}", run)


#: The ceiling on a palette read out of an image. The GIF colour table's own
#: limit and the largest number of swatches any of this is useful at -- a
#: photograph has tens of thousands of distinct colours, so an image import
#: *always* median-cuts unless it was pixel art already.
IMAGE_PALETTE_MAX = 256


def palette_from_image(ctx: Any) -> None:
    """Read a palette out of any image and index the active document to it.

    Never refuses on colour count: an image with more colours than the ceiling
    is median-cut down to it and the toast says so. Refusing would mean the
    command works on pixel art and fails on every photograph, which is the half
    of its input the user is least able to predict.

    The decode is on the task thread with the picker, for the reason every
    dialog in this module is: a native picker is modal to the OS, and a JPEG the
    size of a phone photo is not a frame's worth of work either.
    """
    from .inker import dither

    ensure(ctx)
    tab = active(ctx)
    if tab is None or tab.busy:
        return

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Palette from an image", OPEN_FILTER)
        if path is None:
            return None
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGBA"))
        # Counted through ``np.unique`` on a packed uint32 rather than through a
        # set of tuples: a phone photo is twelve million pixels, and the set
        # costs a gigabyte to answer one number.
        rgb = pixels[..., :3][pixels[..., 3] > 0]
        packed = (
            rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
        )
        distinct = int(np.unique(packed).size)
        return {
            "colours": dither.build_palette([pixels], IMAGE_PALETTE_MAX),
            "distinct": distinct,
        }

    ctx.submit(f"inker-palimg:{tab.uid}", run)


def export_document_palette(ctx: Any) -> None:
    """Write the *document's* table out as a ``.gpl`` or a JASC ``.pal``."""
    tab = active(ctx)
    if tab is None or not tab.doc.palette:
        return
    colours = [tuple(c) for c in tab.doc.palette]
    stem = tab.path.stem if tab.path else "palette"

    def run() -> str | None:
        path = dialogs.save_file(
            "Export the document palette", f"{stem}.gpl", PALETTE_FILTER
        )
        if path is None:
            return None
        _write_palette(path, colours, stem)
        return str(path)

    # Not ``inker-palette-export``: see the sibling above for why sharing it
    # made one of the two commands inert whenever the other was open.
    ctx.submit("inker-palette-export-doc", run)


# --- tileset export/import (Wave 3, Chunk 3.6) --------------------------------
#
# An Inker tileset IS a ``tilegrid.Tileset`` -- ``doc.tilesets`` holds a
# ``TilesetSlot`` over one, Chunk 3.1's whole point -- so there is no
# conversion on either side of this door: exporting is the same ``.tsx``/
# ``.png`` pair Packwright's grid packer already writes through
# ``plotter.tsx``/``plotter.pngio``, and importing is the same reader
# Plotter's own tileset-add path (``plotter_tilesets.add_tileset_path``) uses.
# Legal ground for both, same as every other export in this module: this file
# sits above the engine pin.


def new_tilemap_layer(ctx: Any, tab: InkerDoc, tileset_uid: int | None) -> None:
    """A new tilemap layer bound to *tileset_uid*, above the active row.

    Synchronous, ``add_layer``'s own shape: it is one history step over data
    already in memory, so a task thread would only add a frame of latency to a
    button press. The three verbs here are the panel's doors and the *engine*
    owns every refusal; what this adds is the sentence, which is the half a
    ``bool`` return cannot carry.
    """
    if tileset_uid is None:
        ctx.toast("This document has no tileset yet.", "error")
        return
    try:
        layer = tab.doc.add_tilemap_layer(int(tileset_uid))
    except (KeyError, ValueError) as exc:
        ctx.toast(f"The layer was not added: {exc}.", "error")
        return
    ctx.toast(f"{layer.name} added.", "success")


def convert_to_tilemap(ctx: Any, tab: InkerDoc, tile_w: int, tile_h: int) -> None:
    """Cut the active layer into tiles and bind it to the new tileset.

    The whole *track* on an animated document -- see
    ``Document.convert_layer_to_tilemap``, whose ``False`` covers the three
    ordinary "nothing to do" answers this turns into one sentence each.
    """
    doc = tab.doc
    if not len(doc.stack):
        ctx.toast("There is no layer to convert.", "error")
        return
    layer_uid = doc.stack.active.uid
    try:
        done = doc.convert_layer_to_tilemap(layer_uid, int(tile_w), int(tile_h))
    except ValueError as exc:
        ctx.toast(f"The layer was not converted: {exc}.", "error")
        return
    if not done:
        ctx.toast(
            "That layer was not converted -- it is already a tilemap, or it is "
            "locked.",
            "warn",
        )
        return
    ctx.toast(f"Cut into {tile_w} x {tile_h} tiles.", "success")


def convert_to_raster(ctx: Any, tab: InkerDoc) -> None:
    """Turn the active tilemap layer back into an ordinary one.

    Lossless by construction -- the picture is already the materialization --
    and the tileset stays in the document, which is what makes converting back
    and forth to reach the pixel tools free.
    """
    doc = tab.doc
    if not len(doc.stack):
        ctx.toast("There is no layer to convert.", "error")
        return
    if not doc.convert_layer_to_raster(doc.stack.active.uid):
        ctx.toast(
            "That layer was not converted -- it is not a tilemap, or it is locked.",
            "warn",
        )
        return
    ctx.toast("Converted to a plain layer.", "success")


def export_tileset(ctx: Any, tab: InkerDoc | None = None, *, index: int) -> None:
    """One tileset's atlas as a Tiled ``.tsx`` plus its PNG, side by side.

    Not a save -- ``export_png``'s own reason: it does not change what the tab
    points at, so the document stays dirty against its own file. The tileset
    is read here, before the picker, the same rule every export in this module
    follows: serialising after an unbounded modal would write whatever the
    tileset list looked like when the dialog happened to close.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        ctx.toast("Open a drawing first.", "error")
        return
    if index < 0 or index >= len(tab.doc.tilesets):
        return
    tileset = tab.doc.tilesets[index].tileset
    suggested = tileset.name or "tileset"

    def run() -> dict[str, Any] | None:
        from .plotter import pngio
        from .plotter import tsx as tsxlib

        dest = dialogs.save_file("Export tileset", f"{suggested}.tsx", TSX_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".tsx":
            dest = dest.with_suffix(".tsx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        png_path = dest.with_suffix(".png")
        # The ``.tsx`` names the PNG by the name it is written under here, so
        # the pair is consistent by construction rather than by two writers
        # agreeing on a convention.
        dest.write_bytes(tsxlib.tsx_bytes(tileset, image_name=png_path.name))
        png_path.write_bytes(pngio.png_bytes(tileset.pixels))
        return {"exported": dest}

    _start(ctx, tab, f"inker-export:{tab.uid}", run)


def import_tileset(ctx: Any, tab: InkerDoc | None = None) -> None:
    """A Tiled ``.tsx`` plus its image, added to the document's tileset list.

    Grid geometry and terrain sets travel intact -- ``plotter.tsx.read_tsx``
    builds the same ``tilegrid.Tileset`` a ``.tsx`` reaches Plotter's map
    through, and the shared type is what carries ``terrains``/``phases``. The
    picker and the decode both run on the task thread, ``ask_add_tileset``'s
    own reason: a native picker is modal to the OS, and the image behind a
    ``.tsx`` is routinely as large as anything else this module opens.
    """
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        ctx.toast("Open a drawing first.", "error")
        return
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        from ..service.errors import invalid_from
        from .plotter import tsx as tsxlib
        from .plotter_io import _resolve_source, _within_ceiling

        path = dialogs.open_file("Import tileset", TSX_FILTER)
        if path is None:
            return None
        try:
            data = _within_ceiling(path).read_bytes()
            image_name = tsxlib.tsx_source(data)
            image = docmodes.decode_rgba(_resolve_source(path.parent, image_name))
            tileset = tsxlib.read_tsx(data, image)
        except ValueError as exc:
            raise invalid_from(
                exc, "This tileset could not be imported", field="file"
            ) from exc
        return {"tileset": tileset}

    ctx.submit(f"inker-tileset-import:{uid}", run)


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
