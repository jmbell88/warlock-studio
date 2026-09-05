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

import functools
import logging
import math
from pathlib import Path
from typing import Any

from . import (
    atomic,
    dialogs,
    docmodes,
    filetypes,
    fonts,
    # The two split-out modules this one still *calls* rather than merely
    # serves: the playhead (stopped before every capture) and the colour-mode
    # door (an indexed conversion). Both import this module back as a module
    # object, so the pair may be imported in either order.
    inker_open,
    inker_ops,
    inker_palette_io,
    inker_playback,
    inker_state,
    journal,
    recents,
)
from . import settings as settings_mod
from .inker import aseout
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
# 2048 into 20480, which is a 1.7 GiB layer allocated on the frame thread.
NEW_MAX = 8192


def _under(root: Path, name: str) -> Path:
    """``root / name``, refusing anything that escapes ``root``.

    The same rule ``sirens_io._under`` states for its own exports, spelled
    again here rather than reached across for: a sanitiser is one regex away
    from letting a separator through, and the consequence is a file written
    outside the folder the save dialog handed back. Checked rather than argued
    from, at the one place a path is built out of a composed name.
    """
    path = root / name
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f"{name!r} is not a name inside the export folder")
    return path


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


def clamp_resize(current: tuple[int, int], width: Any, height: Any) -> tuple[int, int]:
    """A typed size for the *resize* popup: growth capped, shrinking free.

    The popup stored ``(max(1, w), max(1, h))`` -- a floor and no ceiling --
    and fed it straight to ``doc.scale`` or ``doc.resize_canvas``, neither of
    which has one either. Typing ``100000`` asked for 40 GB per layer, on the
    frame thread, holding a document that has not been saved.

    ``clamp_canvas`` was not simply reused, and the comment that used to sit on
    ``NEW_MAX`` explained why: this popup exists over a document that already
    exists, and **shrinking it is the usual reason to open it**. So the two
    directions are treated differently -- a value at or below what the axis
    already is passes untouched, whatever it is, and only growth meets a
    ceiling. That ceiling is ``NEW_MAX`` *or the document's own size*,
    whichever is larger, which is what keeps a 12,000-pixel canvas somebody
    imported resizable at all rather than snapping to 8192 the moment its
    owner opens the popup to crop it.
    """
    def one(value: Any, now: int) -> int:
        try:
            asked = max(1, int(value))
        except (TypeError, ValueError):
            return max(1, int(now))
        return min(asked, max(int(now), NEW_MAX))

    return one(width, current[0]), one(height, current[1])


def ensure(ctx: Any) -> InkerState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Paint should not pay for its
    swatches, and because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.inker
    if state is None:
        state = InkerState()
        stored = settings_mod.as_dict(ctx.settings.get("inker"))
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
        try:
            state.shortcut_overrides = inker_ops.parse_shortcuts(
                {"version": 1, "overrides": stored.get("shortcuts", {})}
            )
        except (TypeError, ValueError):
            state.shortcut_overrides = {}
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
        # **The symmetry, which for a long time was the same bug one row
        # down.** ``inker_context._symmetry_hit`` called ``persist`` on every
        # press and said in a comment that symmetry rode in this block; it did
        # not, so a mirror set to draw one thing was gone by the next launch.
        # It belongs *here*, with the grid and the swatches, because it is a
        # property of the person rather than of any document -- which is also
        # why it stays out of ``TOOL_OPTION_DEFAULTS`` and out of a saved
        # preset. A preset that dragged the mirrors along would turn "my
        # inking pen" into "my inking pen, and also mirror everything".
        "symmetry": str(state.symmetry),
        # A list rather than a tuple: settings round-trip through JSON and a
        # tuple comes back a list anyway. Saying so here is what stops the read
        # from having to guess which it is looking at.
        "symmetry_axis": (
            None
            if state.symmetry_axis is None
            else [float(state.symmetry_axis[0]), float(state.symmetry_axis[1])]
        ),
        "radial_count": int(state.radial_count),
    }
    # The last-used export controls -- app-level and shared across tabs, like
    # the canvas furniture beside it, not a per-document ``InkerDoc.
    # export_options`` (those are session-only and never leave the tab). This
    # is what seeds a brand-new tab's controls next session, through
    # ``_restore_export`` below.
    block["export"] = state.export_options_snapshot()
    # Only overrides, never a copy of all defaults: a future compatibility
    # update can improve an untouched binding while an explicit user choice is
    # stable.  Round-trip through the registry to keep hand-edited junk out.
    block["shortcuts"] = inker_ops.parse_shortcuts(
        inker_ops.shortcuts_json(state.shortcut_overrides)
    )
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
    # **Through ``axes_of`` then ``compose``, never assigned raw.** That is the
    # rule every *reader* of this field follows, and it is what makes the round
    # trip safe in both directions at once: a legacy ``"xy"`` normalises to
    # ``"x+y"``, an axis this build no longer knows is dropped rather than kept
    # as a word the engine will not match, and a hand-edited ``"sideways"`` or
    # ``null`` comes out as no symmetry rather than raising on the first frame.
    from .inker import brush

    state.symmetry = brush.compose(
        brush.axes_of(stored.get("symmetry", state.symmetry))
    )
    axis = stored.get("symmetry_axis")
    if isinstance(axis, list | tuple) and len(axis) == 2 and all(
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in axis
    ):
        state.symmetry_axis = (float(axis[0]), float(axis[1]))
    count = stored.get("radial_count")
    if isinstance(count, int) and not isinstance(count, bool):
        state.radial_count = max(brush.MIN_RADIAL, min(brush.MAX_RADIAL, count))


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
    # Through ``set_tool``, like every other way of picking one -- unless the
    # bucket is already in hand, in which case it stays. A capture made *from*
    # the fill tool is somebody building a pattern to pour, and switching them
    # to the brush would take the tool away mid-gesture for the sake of a rule
    # about which tool a capture usually arrives on.
    if state.tool not in inker_state.PATTERN_TOOLS:
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
    for tool in inker_state.STAMP_TOOLS | inker_state.PATTERN_TOOLS:
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


def _and_list(items: Any) -> str:
    """``"a", "b" and "c"`` -- for a sentence naming what a format dropped."""

    parts = [str(item) for item in items]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


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
    inker_playback.stop_play(tab)
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
    inker_playback.stop_play(tab)  # settle the stack before capturing; see save()
    doc = tab.doc
    _settle(ctx, tab)  # before the head is read; see _save_linked
    rev = doc.history.head
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Save layered document", f"{suggested}.ora", SAVE_AS_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() in ASEPRITE_SUFFIXES:
            # Measured before the write, on the document being written.
            lossy = aseout.dropped_by_aseprite(doc)
            _write(doc, dest, "aseprite")
            return {
                "path": dest,
                "rev": rev,
                "format": "aseprite",
                "retitle": True,
                "lossy": lossy,
            }
        if dest.suffix.lower() != ".ora":
            dest = dest.with_suffix(".ora")
        _write(doc, dest, "ora")
        return {"path": dest, "rev": rev, "format": "ora", "retitle": True}

    _start(ctx, tab, f"inker-saveas:{tab.uid}", run)




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
        atomic.write_bytes(path, doc.png_bytes())


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
    inker_playback.stop_play(tab)  # settle the stack before capturing; see save()
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
    inker_playback.stop_play(tab)  # settle the stack before capturing; see save()
    doc = tab.doc
    # **Refused before anything is settled.** ``_settle`` commits a floating
    # buffer and cancels a preview, both of which are visible changes to the
    # document; running them and *then* refusing left the user's paste landed
    # and their dither thrown away for a job that never happened. The float is
    # part of the dirty test rather than folded in by settling first: a pending
    # paste is exactly "what you see", which is what the sentence is about.
    if tab.linked and (tab.dirty or doc.floating is not None):
        ctx.toast("Save first, so the mesh is made from what you see.", "error")
        return
    _settle(ctx, tab)
    if tab.linked:
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
    # The toast is on the landing, not here: the *job* is created inside
    # ``run`` on the task thread, so a toast at the press announced a queue
    # entry that did not exist yet -- and still said so when the encode raised.
    _start(ctx, tab, f"inker-send:{tab.uid}", run)


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
    path = tab.path

    def run() -> dict[str, Any]:
        from . import inker

        svc_files.revert_reference(ctx.svc, job_id)
        svc_files.discard_inker_working(ctx.svc, job_id)
        # Decoded here rather than in ``_reload_linked``: the revert has
        # already happened on disk by this line, and a decode that failed
        # must still report the revert -- so it is a field, not a raise.
        result: dict[str, Any] = {"reverted": True, "job_id": job_id, "doc": None}
        if path is not None:
            try:
                result["doc"] = inker.Document.load(path)
            except Exception as exc:
                result["load_error"] = str(exc)
        return result

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


def _done_inpaint_land(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
    # The frame-thread half of ``inker_bridge.poll_inpaint``: the read,
    # the decode and the resize happened on the task thread.
    from .panes import inker_bridge

    if isinstance(result, dict):
        inker_bridge.land_inpaint(ctx, result["pending"], result["pixels"])
    else:
        ctx.toast("The regeneration produced no picture.", "warn")



def _done_flourish_render(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_flourish

    inker_flourish.land(ctx, state, done, now=inker_flourish.clock())



def _done_flourish_texture_queued(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_flourish

    inker_flourish.on_texture_queued(ctx, state, done)



def _done_flourish_texture(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_flourish

    inker_flourish.land_texture(ctx, state, done)



def _done_flourish_prompt(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_flourish

    inker_flourish.land_prompt(ctx, state, done, now=inker_flourish.clock())



def _done_flourish_restyle_queued(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_flourish

    inker_flourish.on_restyle_queued(ctx, state, done)



def _done_flourish_restyle(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_flourish

    inker_flourish.land_restyle(ctx, state, done)


def _done_open(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
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
        inker_open._report_import_warnings(ctx, result.get("warnings"))


def _done_merge(ctx: Any, state: Any, done: Any) -> None:
    from . import inker_sheet

    inker_sheet.land_merge(ctx, state, done)


def _done_inpaint(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
    from .panes import inker_bridge

    inker_bridge.on_inpaint_queued(ctx, result)


def _done_sheetin(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
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


def _done_recover(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
    if result is None:
        journal.adopt_failed(ctx, "drawing")
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


def _done_autosave(ctx: Any, state: Any, done: Any) -> None:
    """Nothing. A journal write is not a save and reports nothing to the user
    -- the entry exists so the key does not fall through to the tail, which
    would clear ``saving`` on a tab no save was running for."""


def _done_palette(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
    # A list of colours, or None for a cancelled picker. Appended rather
    # than replacing: an import is "add these to what I have", and a user
    # who wanted the old ones gone can right-click them away -- where an
    # import that silently wiped a session's palette has no way back.
    if result:
        for colour in result:
            state.add_swatch(colour)
        persist(ctx)
        ctx.toast(f"Added {len(result)} colour(s).", "success")


def _done_palette_export(ctx: Any, state: Any, done: Any) -> None:
    result = done.result
    # Said out loud. Neither key had a branch here at all, so a palette
    # export fell through to the uid-keyed tail below, found no ``:`` in its
    # key and returned -- reporting neither success nor failure. A write to
    # a path the user chose is exactly the kind of thing that has to answer.
    if result:
        ctx.toast(f"Palette written to {Path(result).name}.", "success")


def _done_index(ctx: Any, state: Any, done: Any) -> None:
    key = done.key
    result = done.result
    # The picker came back with a table for a *document*. Resolved through
    # the uid rather than through ``active``: a native picker is unbounded,
    # and the user may well have switched tabs while it was up -- indexing
    # whichever document happens to be in front now would rewrite the wrong
    # file's pixels.
    if result:
        inker_palette_io.index_to(ctx, state.get(key.split(":", 1)[1]), result)


def _done_palimg(ctx: Any, state: Any, done: Any) -> None:
    key = done.key
    result = done.result
    # A palette read out of an image. Resolved through the uid for
    # ``inker-index``'s reason, and a key of its own because the *count*
    # comes back with it: a median cut is a loss, and one that happened
    # silently is one the user finds out about by noticing their drawing
    # looks wrong.
    if isinstance(result, dict) and result.get("colours"):
        colours = result["colours"]
        if inker_palette_io.index_to(ctx, state.get(key.split(":", 1)[1]), colours):
            distinct = int(result.get("distinct", 0))
            if distinct > len(colours):
                ctx.toast(
                    f"{distinct} colours reduced to {len(colours)}.", "warn"
                )


def _done_tileset_import(ctx: Any, state: Any, done: Any) -> None:
    key = done.key
    result = done.result
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



def _done_send(ctx: Any, state: Any, done: Any) -> None:

    key = done.key
    result = done.result
    name = done.key.split(":", 1)[0]
    ctx.cache.invalidate()
    if name == "inker-send" and isinstance(result, dict) and result.get("id"):
        ctx.toast("Queued a mesh from the drawn image.")
    # ``inker-send`` locks its tab while the flatten runs off-thread;
    # ``inker-promote`` has no tab of its own to unlock.
    sent = state.get(key.split(":", 1)[1]) if name == "inker-send" and ":" in key else None
    if sent is not None:
        sent.saving = False




# --- what answers for which task key ------------------------------------------


@functools.cache
def _TASK_HANDLERS() -> dict[str, Any]:
    """``{key prefix: handler}`` for :func:`on_task_done`.

    Built once and cached: the Flourish keys are constants on another module,
    so this cannot be a literal at import time without ``inker_flourish``
    becoming a module-scope import of a module that imports this one back.
    """

    from . import inker_flourish  # noqa: F401 -- named by the table below

    return {
        "inker-inpaint-land": _done_inpaint_land,
        inker_flourish.RENDER_KEY: _done_flourish_render,
        inker_flourish.INSERT_KEY: _done_flourish_render,
        inker_flourish.TEXTURE_KEY: _done_flourish_texture_queued,
        inker_flourish.TEXTURE_LAND_KEY: _done_flourish_texture,
        inker_flourish.PROMPT_KEY: _done_flourish_prompt,
        inker_flourish.RESTYLE_KEY: _done_flourish_restyle_queued,
        inker_flourish.RESTYLE_LAND_KEY: _done_flourish_restyle,
        "inker-open": _done_open,
        "inker-merge": _done_merge,
        "inker-inpaint": _done_inpaint,
        "inker-sheetin": _done_sheetin,
        "inker-recover": _done_recover,
        "inker-autosave": _done_autosave,
        "inker-palette": _done_palette,
        "inker-palette-export": _done_palette_export,
        "inker-palette-export-doc": _done_palette_export,
        "inker-palette-export-image": _done_palette_export,
        "inker-index": _done_index,
        "inker-palimg": _done_palimg,
        "inker-tileset-import": _done_tileset_import,
        "inker-send": _done_send,
        "inker-promote": _done_send,
    }




def on_task_done(ctx: Any, done: Any) -> None:
    """Called from App._on_task_done for every ``inker-`` key.

    **A table, not a chain.** This was fifteen ``if name ==`` arms in a row --
    the fifteenth read fourteen comparisons to get to, a new key was appended
    wherever the last one happened to end, and two of them (the palette
    exports) had been *forgotten* entirely and fell through to the tail, which
    found no uid in the key and returned having reported neither success nor
    failure. A dict says what answers for what in one screen, and a key with no
    entry lands on the tail deliberately rather than by omission.

    The tail is the default and is the ordinary case: a save or an export
    landing on the tab named by the key's own uid.
    """
    state = ensure(ctx)
    key, result = done.key, done.result
    handler = _TASK_HANDLERS().get(key.split(":", 1)[0])
    if handler is not None:
        handler(ctx, state, done)
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
        _reload_linked(ctx, tab, result.get("doc"), result.get("load_error"))
        return

    # **A lossy write is not a save.** ``.aseprite`` has no chunk for a
    # Flourish recipe, the flatten matte, alpha lock or an empty group
    # (``docs/COMPAT.md``), so marking the tab clean over one told the user
    # their work was on disk when part of it was not -- and dropping the
    # journal took the crash copy that still had it. Asked of *this* document
    # rather than of the format, because a plain drawing loses nothing here and
    # is a real save.
    lost = result.get("lossy") or ()
    if lost:
        ctx.toast(
            "Written, but .aseprite has no place for " + _and_list(lost) + ". "
            "The drawing is still unsaved as a layered file.",
            "warn",
        )
    else:
        tab.mark_saved(result.get("rev"))
        # The document is on disk under a name the user chose, so the crash
        # copy is describing work that is no longer at risk. Dropped here
        # rather than on a timer: an autosave that outlived its document is
        # exactly the file that gets offered back after a clean session and
        # confuses somebody.
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
    if not lost:
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


def _reload_linked(ctx: Any, tab: InkerDoc, doc: Any, load_error: str | None = None) -> None:
    """Swap in the re-decoded document after a revert replaced its file.

    ``doc`` was decoded by the revert task; this is the frame-thread half,
    which owns the textures and the tab.
    """
    if tab.path is None:
        return
    if doc is None:
        ctx.toast(f"Reverted, but the image could not be reopened ({load_error}).", "error")
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
    from . import viewer_embed

    viewer = ctx.viewer
    if viewer is not None and tab.path is not None and viewer.path == tab.path:
        viewer.clear()
        # Off the frame thread: this runs on the frame a revert lands, and a
        # 1024-square PNG decode there is a visible hitch at exactly the moment
        # something was pressed. ``_sync_viewer``'s own split; the upload is
        # adopted by ``App._adopt_model`` when it arrives.
        viewer_embed.request_reference(ctx, tab.path)


# --- closing and guarding ---------------------------------------------------


def tool_label(ctx: Any) -> str:
    """The active tool's name, for the status bar's shared branch."""
    state = getattr(ctx.state, "inker", None)
    return inker_state.tool_label(state.tool) if state is not None else ""


def request_close(ctx: Any, tab: InkerDoc) -> None:
    """``docmodes.close_tab``; what is Inker's is the release."""
    state = ensure(ctx)

    def release(tab: InkerDoc) -> None:
        from .panes import inker_textures

        inker_textures.release_doc(ctx, tab.uid)

    docmodes.close_tab(ctx, state, tab.uid, release)


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

#: How the toolbox writes a second binding, tool to the chord.
#:
#: **Read out of the registry, not written out beside it.** This was a hand
#: copy of seven shifted letters checked against ``inker_state.TOOLS`` -- which
#: is the table of *tools*, not the table of *bindings* -- so a rebound tool
#: left the Ctrl+/ sheet advertising a chord that no longer did anything, and
#: a shifted binding added to the registry never reached the sheet at all (the
#: 2026-09-02 review, section 5). ``TOOL_KEYS`` above already learned this
#: lesson; this is the same lesson one table over.
ALT_TOOL_CHORDS = {
    binding.target: binding.chord
    for binding in inker_ops.BINDINGS
    if binding.kind == "tool" and binding.chord.startswith("Shift+")
}

#: Letter to tool for the shifted chords, which is ``ALT_TOOL_CHORDS`` read the
#: other way round. Kept for the callers that ask "what does Shift+B do".
SHIFT_TOOL_KEYS = {
    chord.removeprefix("Shift+").lower(): tool for tool, chord in ALT_TOOL_CHORDS.items()
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






# --- palette files ----------------------------------------------------------
#
# The picker and the file are both blocking, so both go to a task thread -- the
# rule every dialog and every encode in this module follows. The *bytes* for an
# export are built on the frame thread, for ``save_as``'s reason: they read live
# state, and doing that after an unbounded modal would write whatever the user
# changed while it was open.

GPL_FILTER = ["GIMP palette (*.gpl)", "*.gpl"]

# Every text palette format behind one filter, the combined pattern first so the
# picker opens on it. ``.pal`` is JASC's text form only -- see ``gpl.parse_jasc``
# for why the other two things called ``.pal`` are refused rather than guessed.
#
# The four match ``service.palettes.SUFFIXES`` and ``gpl``'s own writers, which
# is asserted rather than assumed: a filter that offers a suffix no writer
# handles produces a file in the wrong format under the right name, and one
# that omits a suffix the reader takes hides files the user already owns.
PALETTE_FILTER = [
    "Palettes (*.gpl *.pal *.hex *.txt)",
    "*.gpl *.pal *.hex *.txt",
    "GIMP palette (*.gpl)",
    "*.gpl",
    "JASC palette (*.pal)",
    "*.pal",
    "Lospec palette (*.hex)",
    "*.hex",
    "Paint.NET palette (*.txt)",
    "*.txt",
]

#: The suffixes ``_write_palette`` will write under, and the set the filter
#: above offers. ``.gpl`` is the fallback for anything else.
PALETTE_SUFFIXES = (".gpl", ".pal", ".hex", ".txt")




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


def _no_document_reason(tab: InkerDoc | None) -> str:
    """Why a document verb is refused -- *this* tab's reason, not one sentence.

    "Open a drawing first." is a lie when a drawing is open and being written;
    a refusal that names the wrong cause is worse than no refusal, because the
    user goes looking for the tab they already have.
    """

    if tab is None:
        return "Open a drawing first."
    return "This drawing is being written; try again when the save lands."


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
        ctx.toast(_no_document_reason(tab), "error")
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
        atomic.write_bytes(dest, tsxlib.tsx_bytes(tileset, image_name=png_path.name))
        atomic.write_bytes(png_path, pngio.png_bytes(tileset.pixels))
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
        ctx.toast(_no_document_reason(tab), "error")
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


def _load_recovery(path: Path, meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Blocking; task thread only. ``None`` when the copy will not reopen --
    a corrupt ``.ora`` arrived as the generic "That did not work" here, where
    every other provider warns (``journal.adopt_failed``, 2026-09-05)."""
    from .inker import Document

    try:
        doc = Document.load(Path(path))
    except Exception:
        log.exception("could not reopen the recovered drawing at %s", path)
        return None
    doc.path = None
    title = (meta or {}).get("title") or Path(path).stem.rsplit("-", 1)[0]
    return {
        "doc": doc,
        "title": f"{title} (recovered)",
        "format": "ora",
        "autosave": str(path),
    }


# -- Flourish -----------------------------------------------------------------------


def flourish_insert(
    ctx: Any,
    tab: Any,
    *,
    preset: str = "fireball",
    mode: str = "painterly",
    directions: int = 1,
    **_: Any,
) -> bool:
    """Bake a preset off-thread and land it as a new effect group.

    The bake runs in a task -- a 128px effect is a few seconds of numpy -- and
    ``on_task_done`` puts it on the document through ``insert_flourish`` as
    one undo step. The recipe is sized to the document when the document is
    smaller than the preset's canvas, so a 64px sprite gets a 64px fireball.
    """
    from dataclasses import replace as _replace

    from . import inker_flourish
    from .inker.flourish import presets

    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return False
    try:
        recipe = presets.load(preset)
    except (KeyError, ValueError) as exc:
        ctx.toast(f"No such effect: {exc}", "error")
        return False
    width, height = tab.doc.size
    side = min(recipe.width, width, recipe.height, height)
    if side < recipe.width or side < recipe.height:
        scale = side / max(recipe.width, recipe.height)
        recipe = _scaled(recipe, scale)
    recipe = _replace(
        recipe,
        mode=mode if mode in ("painterly", "pixel") else "painterly",
        directions=max(1, int(directions)),
    )
    if not inker_flourish.submit_insert(ctx, tab, recipe):
        ctx.toast("An effect is already being inserted into this document.", "info")
        return False
    ctx.toast(f"Rendering {recipe.name}...")
    return True


def _scaled(recipe: Any, scale: float) -> Any:
    """The recipe on a smaller canvas, its geometry scaled with it."""
    from dataclasses import replace as _replace

    from .inker.flourish import curves as flourish_curves
    from .inker.flourish import prims
    from .inker.flourish import recipe as flourish_recipe

    spatial = {"x", "y", "radius", "width", "height", "size", "spawn_radius", "speed",
               "gravity", "thickness", "scale", "noise_scale", "strength", "rise", "drift"}
    layers = []
    for layer in recipe.layers:
        params = dict(layer.params)
        for name, spec in prims.params_of(layer.kind).items():
            if name not in spatial or name not in params:
                continue
            value = params[name]
            if spec.kind == "float":
                params[name] = float(value) * scale
            elif spec.kind == "curve":
                curve = flourish_curves.Curve.from_json(value)
                params[name] = flourish_curves.Curve(
                    tuple((t, v * scale) for t, v in curve.keys), curve.easing
                ).to_json()
        layers.append(_replace(layer, params=params))
    return flourish_recipe.clamp(
        _replace(
            recipe,
            width=max(8, int(round(recipe.width * scale))),
            height=max(8, int(round(recipe.height * scale))),
            layers=tuple(layers),
        )
    )


def flourish_regenerate(ctx: Any, tab: Any, *, force: bool = False, **_: Any) -> bool:
    """Render the active effect again, now, with whatever the inspector holds."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return False
    group = inker_flourish.active_group(state, tab)
    if group is None:
        state.say(inker_flourish.NO_EFFECT)
        return False
    recipe = inker_flourish.current_recipe(state, tab, group)
    if recipe is None:
        return False
    state.flourish_due.pop(group, None)
    if not inker_flourish.submit_render(ctx, tab, group, recipe, force=force):
        state.say(inker_flourish.RENDERING)
        return False
    return True


def flourish_keep_edits(ctx: Any, tab: Any, **_: Any) -> bool:
    """Clear every conflict flag on the active effect: the paint stands."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None:
        return False
    group = inker_flourish.active_group(state, tab)
    if group is None:
        return False
    flagged = tab.doc.flourish_conflicts(group)
    if not flagged:
        state.say(inker_flourish.NO_CONFLICTS)
        return False
    return tab.doc.resolve_flourish(group, flagged)


def flourish_detach(ctx: Any, tab: Any, **_: Any) -> bool:
    """Forget the recipe; the layers stay as ordinary layers."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None:
        return False
    group = inker_flourish.active_group(state, tab)
    if group is None:
        return False
    state.flourish_pending.pop(group, None)
    state.flourish_due.pop(group, None)
    if tab.doc.detach_flourish(group):
        ctx.toast("Detached: the layers are yours, the recipe is gone.", "info")
        return True
    return False


def flourish_export(ctx: Any, tab: Any, **_: Any) -> bool:
    """One sheet per phase: the per-tag export, which is exactly what a phase
    is. Nothing Flourish-specific is written -- a phase sheet is the file
    exporting that tag on its own would write, sidecar and all."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if not inker_flourish.can_export(state, tab):
        state.say(inker_flourish.export_reason(state, tab))
        return False
    from . import inker_export

    inker_export.export_per_tag(ctx, tab, "sheet")
    return True


def flourish_texture_selection(ctx: Any, tab: Any, **_: Any) -> bool:
    """The selection becomes a texture of the active effect."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None:
        return False
    return inker_flourish.texture_from_selection(ctx, state, tab) is not None


def flourish_texture_generate(ctx: Any, tab: Any, *, subject: str = "", **_: Any) -> bool:
    """Queue a texture for the active effect from a few words."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None:
        return False
    return inker_flourish.submit_texture(ctx, state, tab, subject)


def flourish_prompt(ctx: Any, tab: Any, *, text: str | None = None, **_: Any) -> bool:
    """The words in the inspector's field (or ``text``) become a recipe change."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None:
        return False
    words = state.flourish_prompt_text if text is None else text
    return inker_flourish.submit_prompt(ctx, state, tab, words)


def flourish_restyle(
    ctx: Any,
    tab: Any,
    *,
    phase: str = "",
    subject: str = "",
    strength: float = 0.55,
    anchors: int = 3,
    **_: Any,
) -> bool:
    """A few frames of one phase through the image model; the rest interpolated."""
    from . import inker_flourish

    state = ensure(ctx)
    tab = tab or active(ctx)
    if tab is None:
        return False
    names = inker_flourish.phase_names(state, tab)
    if not names:
        return False
    return inker_flourish.submit_restyle(
        ctx,
        state,
        tab,
        phase=phase or names[0],
        subject=subject,
        strength=strength,
        anchors=anchors,
    )


# --- what moved out, and the one door back ------------------------------------
#
# ``inker_export``, ``inker_keys``, ``inker_open``, ``inker_palette_io`` and ``inker_playback`` --
# the modules T7 split this one into (the 2026-09-02 review). Every name they
# define is still reachable as ``inker_mode.<name>``, because that is what the
# panes, the ops registry, the shortcut tables and a good deal of the test
# suite name.
#
# A PEP 562 ``__getattr__`` over a table rather than imports at the bottom of
# this file: each moved module imports *this* one (as a module, so its own
# attribute lookups happen at call time), and a bottom ``from .inker_export
# import ...`` here would fail whenever something imported the pair the other
# way round. Resolving on demand has no order at all.
#
# **A name appears exactly once**, which is what makes the table the record of
# where each thing went rather than a second place to keep in step.
_MOVED: dict[str, str] = {
    "COLOR_MODES": "inker_palette_io",
    "COLOR_MODE_LABELS": "inker_palette_io",
    "IMAGE_PALETTE_MAX": "inker_palette_io",
    "MAX_PREVIEW_SPEED": "inker_playback",
    "MIN_PREVIEW_SPEED": "inker_playback",
    "PALETTE_STRIP_CELL": "inker_palette_io",
    "REPEATABLE": "inker_export",
    "_CHORD_NAMES": "inker_keys",
    "_Export": "inker_export",
    "_Leg": "inker_export",
    "_MUTATING_CTRL": "inker_keys",
    "_Payload": "inker_export",
    "_SLICE_SAFE": "inker_export",
    "_begin_export": "inker_export",
    "_ctrl_key": "inker_keys",
    "_cut_matte": "inker_open",
    "_frame_palette": "inker_export",
    "_load": "inker_open",
    "_load_aseprite": "inker_open",
    "_load_job": "inker_open",
    "_load_rendered_sheet": "inker_open",
    "_load_sprite_draft": "inker_open",
    "_modal": "inker_keys",
    "_palette_text": "inker_palette_io",
    "_report_import_warnings": "inker_open",
    "_slice_filenames": "inker_export",
    "_split_stems": "inker_export",
    "_submit_export": "inker_export",
    "_suggest_grid": "inker_open",
    "_suggested_dialog_name": "inker_export",
    "_write_palette": "inker_palette_io",
    "animate": "inker_playback",
    "ask_import_aseprite": "inker_open",
    "ask_import_sheet": "inker_open",
    "ask_open": "inker_open",
    "can_edit_job": "inker_open",
    "chord_of": "inker_keys",
    "duplicate_document": "inker_open",
    "export_document_palette": "inker_palette_io",
    "export_gif": "inker_export",
    "export_palette": "inker_palette_io",
    "export_palette_image": "inker_palette_io",
    "export_per_layer": "inker_export",
    "export_per_tag": "inker_export",
    "export_png": "inker_export",
    "export_pngs": "inker_export",
    "export_range": "inker_export",
    "export_sheet": "inker_export",
    "export_slices": "inker_export",
    "export_tag": "inker_export",
    "frame_durations": "inker_playback",
    "handle_key": "inker_keys",
    "import_aseprite_path": "inker_open",
    "import_document_palette": "inker_palette_io",
    "import_palette": "inker_palette_io",
    "import_sheet": "inker_open",
    "index_to": "inker_palette_io",
    "load_sheet_cells": "inker_open",
    "new_from_selection": "inker_open",
    "newest_sheet_after": "inker_open",
    "open_job_reference": "inker_open",
    "open_path": "inker_open",
    "open_pixel_artifact": "inker_open",
    "open_pixels": "inker_open",
    "open_rendered_sheet": "inker_open",
    "open_sprite_draft": "inker_open",
    "palette_from_image": "inker_palette_io",
    "palette_strip": "inker_palette_io",
    "pump_export": "inker_export",
    "pump_undo_trim": "inker_export",
    "release_all": "inker_keys",
    "repeat_export": "inker_export",
    "set_color_mode": "inker_palette_io",
    "set_transparent_slot": "inker_palette_io",
    "sheet_grid": "inker_open",
    "step_frame": "inker_playback",
    "stop_play": "inker_playback",
    "tick_playback": "inker_playback",
    "tick_preview": "inker_playback",
    "toggle_play": "inker_playback",
    "toggle_preview": "inker_playback",
}


def __getattr__(name: str) -> Any:
    module = _MOVED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __package__), name)


def __dir__() -> list[str]:
    """The moved names included, so ``dir`` and tab completion still find them."""
    return sorted({*globals(), *_MOVED})
