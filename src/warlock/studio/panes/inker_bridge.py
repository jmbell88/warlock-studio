"""What connects a painting to the rest of the app.

Paint is a third mode rather than a separate program, and this is the panel
that makes that true: a document can become a reference the pipeline accepts,
or go straight to the mesh stage, and a document that came *from* a job can be
saved back into it.

The distinction the whole panel turns on is whether the document is *linked*.
A linked one writes back into a job's input.png (with the layered source kept
beside it); an unlinked one is a plain file that has never been part of a job.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker_mode, theme, widgets
from ..inker import transform
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_colors


def _busy_why(tab: Any) -> str:
    """Why every button on this panel is out, when one of them is.

    ``tab.busy`` is deliberately one question with two answers behind it -- a
    save is encoding off-thread, or playback is running -- and a user reading
    "Saving..." while the clip is looping would go and look for a save. So the
    sentence separates them here, once, and the panel's six buttons share it:
    the ``_VIEWPORT_WHY`` pattern.
    """
    if getattr(tab, "playing", False):
        return "Playback is running. Stop it to edit the document."
    return "This document is being written; the buttons come back when it lands."


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    tab = state.active
    # Most-touched first: Canvas, Animation, Pipeline, then Document and File
    # with the recent list trailing -- so with the Layers pane above, the column
    # reads Layers, Canvas, Animation, Pipeline, Document, File, Recent. The
    # first three only exist with a document open; the help button rides the
    # first heading actually drawn, whichever that is.
    if tab is not None:
        widgets.section("Canvas")
        manual_render.help_button(ctx, "inker-bridge")
        _canvas_ops(ctx, tab)
        imgui.dummy((0, 8))
        _animation(ctx, tab)
        imgui.dummy((0, 8))
        _pipeline(ctx, tab)
        imgui.dummy((0, 8))

    widgets.section("Document")
    if tab is None:
        manual_render.help_button(ctx, "inker-bridge")
    # Above the tab check: importing a sheet *makes* a document, so it has to
    # be reachable when there is none open, which is exactly the moment a user
    # is most likely to want it.
    _sheet_import(ctx)
    if tab is None:
        widgets.muted("Nothing open.")
    else:
        doc = tab.doc
        width, height = doc.size
        widgets.muted(f"{width} x {height} - {len(doc.stack)} layer(s)")
        widgets.muted(f"{tab.view.zoom * 100:.0f}%  -  {tab.file_format.upper()}")
        if tab.path is not None:
            imgui.text_wrapped(str(tab.path))
        if tab.linked:
            widgets.text_colored(theme.OK, f"linked to job {tab.job_id[:8]}")
        else:
            widgets.muted("not part of a job")

    imgui.dummy((0, 8))
    widgets.section("File")
    _file(ctx, state, tab)


def _file_why(state: Any, tab: Any) -> str:
    """Why the five file buttons are out, when they are.

    A save commits the floating buffer, so saving mid-transform would land the
    transform with no confirm and leave the mode pointing at nothing -- which
    is a different sentence from "a save is already running", and the row used
    to say neither.
    """
    if state.transforming:
        return (
            "A transform is still floating. Apply or cancel it first -- saving "
            "would commit it with no confirm."
        )
    return _busy_why(tab)


def _file(ctx: Any, state: Any, tab: Any) -> None:
    """New/Open/Save/Save As/Export PNG, and the recent list.

    Moved here from the canvas's own row in the UI redesign, wave 4.2, and shaped
    like ``plotter_bridge``'s file block on purpose: Inker was the one document
    mode whose file actions did not live in its bridge panel, and the row they
    were on was the app's worst clipping case -- eight labelled buttons plus a
    combo plus a status word chained with ``same_line``, losing "Export PNG"
    off the right edge at 150 %. A full-width button in a fixed-width sidebar
    cannot clip, which is the other half of why they came here.
    """
    from pathlib import Path

    from . import inker_canvas

    width = widgets.grid_width(2)
    if controls.button(f"{icons.PLUS} New", (width, 0)):
        imgui.open_popup("new-canvas")
    imgui.same_line()
    if controls.button(f"{icons.FOLDER_OPEN} Open...", (width, 0)):
        inker_mode.ask_open(ctx)
    # This pane's own registration of the shared popup: a popup belongs to the
    # window that begins it, so the canvas's copy is not reachable from here.
    inker_canvas.new_popup(ctx)

    if tab is not None:
        ready = not tab.busy and not state.transforming
        why = _file_why(state, tab)
        imgui.dummy((0, 4))
        if widgets.disabled_button(
            f"{icons.SAVE} Save (Ctrl+S)", ready, (width, 0), reason=why
        ):
            inker_mode.save(ctx, tab)
        imgui.same_line()
        if widgets.disabled_button("Save As...", ready, (width, 0), reason=why):
            inker_mode.save_as(ctx, tab)
        if widgets.disabled_button(
            f"{icons.UPLOAD} Export PNG", ready, (-1, 0), reason=why
        ):
            inker_mode.export_png(ctx, tab)

        # Minimal tileset doors -- Task 8 owns the picker pane that shows the
        # list itself; until then, "Export tileset..." reaches the first one
        # a document has, the same way every button above reaches the one
        # document a tab holds.
        has_tileset = bool(tab.doc.tilesets)
        if widgets.disabled_button(
            "Export tileset...",
            ready and has_tileset,
            (width, 0),
            reason=why if not ready else "This document has no tileset yet.",
        ):
            inker_mode.export_tileset(ctx, tab, index=0)
        imgui.same_line()
        if widgets.disabled_button(
            "Import tileset (.tsx)...", ready, (width, 0), reason=why
        ):
            inker_mode.import_tileset(ctx, tab)

    widgets.recent_files(
        inker_mode.recent_paths(ctx),
        lambda path: inker_mode.open_path(ctx, Path(path)),
    )


def _animation(ctx: Any, tab: Any) -> None:
    """The one door into animating a document.

    Deliberately a button rather than a mode or a checkbox: animating is an
    *edit* -- one undo step that turns the layers into the first frame's cels --
    so it belongs on the same footing as adding a layer, and Ctrl+Z takes it
    back. Once the document is animated this row goes quiet and the timeline
    strip owns everything else.
    """
    widgets.section("Animation")
    if tab.doc.anim is None:
        # The help rides the button as ``tooltip=`` rather than trailing it as
        # a ``help_marker``: after a ``(-1, 0)`` button there is exactly zero
        # room on the line, so ``same_line_or_wrap`` -- correctly -- dropped
        # the glyph onto the next control's row, where it read as that one's.
        # ``field_label``'s docstring names the defect class; a button has no
        # label row, so its own hover is where the sentence goes.
        if widgets.disabled_button(
            "Animate",
            not tab.busy,
            (-1, 0),
            reason=_busy_why(tab),
            tooltip=(
                "Turns this drawing into frame one of an animation and adds a"
                " second frame. The layers become tracks; Ctrl+Z undoes the"
                " whole thing."
            ),
        ):
            inker_mode.animate(ctx, tab)
        return
    anim = tab.doc.anim
    widgets.muted(f"{len(anim.frames)} frames, {len(anim.tracks)} tracks")
    widgets.muted(f"{anim.duration_ms()} ms total")


def _pipeline(ctx: Any, tab: Any) -> None:
    widgets.section("Pipeline")
    busy = tab.busy
    why = _busy_why(tab)
    # Help as ``tooltip=`` on each full-width button, not a trailing
    # ``help_marker`` -- the Animation row above says why.
    if not tab.linked and widgets.disabled_button(
        "Save as reference",
        not busy,
        (-1, 0),
        reason=why,
        tooltip=(
            "Adds this image to the library as a finished reference, so it"
            " can be meshed, promoted and rerun like a generated one."
        ),
    ):
        inker_mode.save_as_reference(ctx, tab)
    # "Make 3D", not "Send to 3D": there is no 3D to send anything *to* since
    # wave 5 folded 2D and 3D into Create's stages, and this is the same act
    # the Mesh stage's own button performs on a reference -- so it is the same
    # words. The function keeps its name; it is not what anybody reads.
    if widgets.disabled_button(
        "Make 3D",
        not busy,
        (-1, 0),
        reason=why,
        tooltip="Queues the mesh stage from the flattened image.",
    ):
        inker_mode.send_to_3d(ctx, tab)
    if tab.linked and widgets.disabled_button(
        "Revert to original",
        tab.has_original and not busy,
        (-1, 0),
        reason=why
        if busy
        else "There is no original kept for this reference: it has never been edited.",
    ):
        inker_mode.revert(ctx, tab)


def _canvas_ops(ctx: Any, tab: Any) -> None:
    """The body only: ``draw`` owns the "Canvas" heading, because the help
    button has to ride the pane's first heading row."""
    doc = tab.doc
    # Every control below either rebinds a layer's pixels (the geometry ops,
    # via _map_planes) or rebinds the stack wholesale (undo, via
    # restore_snapshot). ``write_ora`` flattens, writes stack.xml and then
    # walks the stack writing one PNG per layer, so either one landing
    # mid-save produces an archive whose parts disagree about the canvas size.
    # The canvas, the layers panel and the keyboard path all gate on this
    # flag; this panel was the hole.
    imgui.begin_disabled(tab.busy)
    if controls.button("Flip H"):
        doc.flip("horizontal")
    imgui.same_line()
    if controls.button("Flip V"):
        doc.flip("vertical")
    imgui.same_line()
    if controls.button("Rotate"):
        doc.rotate90()
    if controls.button("Resize..."):
        imgui.open_popup("inker-resize")
    imgui.same_line()
    if controls.button("Fit view"):
        tab.view.fitted = False
    if controls.button("Filter..."):
        _open_filter(ctx, tab)

    imgui.dummy((0, 6))
    if widgets.disabled_button(
        "Undo", doc.history.can_undo, reason="Nothing to undo yet."
    ):
        doc.undo()
    imgui.same_line()
    if widgets.disabled_button(
        "Redo", doc.history.can_redo, reason="Nothing to redo: this is the newest step."
    ):
        doc.redo()
    imgui.same_line()
    widgets.muted(f"{len(doc.history)} step(s)")
    imgui.end_disabled()

    # Outside the disabled scope: a popup is its own window, and imgui's
    # disabled state is not meant to span a Begin/End pair. It carries the
    # same gate on the two buttons that actually resample the document -- a
    # popup already open when a save starts would otherwise still fire them.
    _resize_popup(ctx, tab)
    _filter_popup(ctx, tab)


def _resize_popup(ctx: Any, tab: Any) -> None:
    if not imgui.begin_popup("inker-resize"):
        return
    widgets.popup_chrome(_imgui=imgui)
    key = f"inker_resize:{tab.uid}"
    width, height = ctx.state.preview.get(key) or tab.doc.size
    imgui.set_next_item_width(sp(90))
    changed_w, width = controls.input_int("W", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed_h, height = controls.input_int("H", int(height), 0)
    if changed_w or changed_h:
        ctx.state.preview[key] = (max(1, width), max(1, height))
    state = inker_mode.ensure(ctx)
    state.resample = widgets.labeled_combo(
        "Resample",
        state.resample,
        [(k, k) for k in transform.RESAMPLES],
        help_text=(
            "Nearest copies each source pixel whole, which is what pixel art needs "
            "-- a filtered scale of a 32x32 sprite comes back blurred and with "
            "thousands of colours in it. Smooth is right for everything else."
        ),
    )
    anchor = _anchor_grid(ctx, tab)
    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if controls.button("Scale image", (sp(180), 0)):
        tab.doc.scale((max(1, width), max(1, height)), resample=state.resample)
        tab.view.fitted = False
        imgui.close_current_popup()
    # Two different operations that a single "resize" would conflate: one
    # resamples the picture, the other changes how much room it has. The anchor
    # belongs to the second: scaling has nowhere to put slack.
    if controls.button("Resize canvas", (sp(180), 0)):
        tab.doc.resize_canvas((max(1, width), max(1, height)), anchor=anchor)
        tab.view.fitted = False
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.end_popup()


# The grid, drawn in reading order. Taken from ``transform.ANCHORS`` rather
# than written again here: the pane decides the layout and the engine owns what
# each name means, which is why that table is written out rather than derived
# from a 3x3 index.
ANCHOR_ROWS = (
    ("top-left", "top", "top-right"),
    ("left", "centre", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)


def _anchor_grid(ctx: Any, tab: Any) -> str:
    """Nine buttons saying where the old image sits in the new canvas.

    Remembered per tab in the preview dictionary beside the width and height,
    so reopening the popup after a mistake offers the same answer rather than
    silently going back to the corner.
    """
    from ..inker import transform as tf

    key = f"inker_anchor:{tab.uid}"
    current = ctx.state.preview.get(key) or "top-left"
    widgets.field_label("anchor")
    for row in ANCHOR_ROWS:
        for name in row:
            selected = name == current
            if selected:
                imgui.push_style_color(
                    imgui.Col_.button.value,
                    imgui.get_style().color_(imgui.Col_.button_active.value),
                )
            if controls.button(f" ##anchor{name}", (sp(28), sp(24))):
                ctx.state.preview[key] = name
                current = name
            if selected:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(name)
            if name != row[-1]:
                imgui.same_line()
        imgui.new_line()
    return current if current in tf.ANCHORS else "top-left"


# --- filters ----------------------------------------------------------------
#
# A live preview rather than an apply-and-look: every one of these is a value
# nobody can predict, and a filter you have to undo to judge is a filter you
# stop using. The document owns the session (``begin_filter`` takes the copy
# every preview recomputes from) so nothing here holds pixels, and the whole
# thing is one undo step however many times a slider moved.


FILTER_POPUP = "inker-filter"


def _open_filter(ctx: Any, tab: Any) -> None:
    from ..inker import filters

    state = inker_mode.ensure(ctx)
    if tab.busy:
        return
    if not state.filter_name:
        state.filter_name = next(iter(filters.FILTERS))
    if tab.doc.begin_filter() is None:
        ctx.toast("There is nothing to filter.", "warn")
        return
    state.filter_open = True
    imgui.open_popup(FILTER_POPUP)


def _filter_values(state: Any, name: str) -> dict[str, Any]:
    from ..inker import filters

    got = state.filter_params.get(name)
    if got is None:
        got = filters.popup_values(name)
        state.filter_params[name] = got
    return got


# A parameter whose *name* is not what to call it on screen. ``replace colour``
# takes ``old`` and ``new`` because ``from`` is a keyword and cannot be a keyword
# argument -- which is a fact about Python and not something a user should have
# to read the source to translate. The manual says From and To, so the popup
# does too.
_PARAM_LABELS = {"old": "From", "new": "To"}


def _param_label(key: str) -> str:
    return _PARAM_LABELS.get(key, key)


def _filter_control(state: Any, values: dict[str, Any], key: str) -> None:
    """One parameter row, drawn by the kind the registry says it is.

    Four kinds rather than a slider and a special case, because the FX staples
    brought parameters a slider cannot hold: a colour, an on/off, and a choice
    between two numbers that has nothing in between. Which kind a name is lives
    in ``filters`` beside the filter that declares it -- see ``COLOUR_PARAMS``.

    Every id carries the *parameter* name rather than the label, so what a
    control is called and what it is are free to differ. (The choice combos go
    through ``labeled_combo``, whose id is its label -- which is the same string
    for both of them, neither being relabelled.)
    """
    from ..inker import filters

    label = _param_label(key)
    if key in filters.COLOUR_PARAMS:
        # ``inker_colors``' own conversions rather than a second pair here: the
        # rounding between imgui's floats and the 8-bit tuple the engine writes
        # with is a rule, and two copies of a rule are one disagreement waiting.
        changed, value = controls.color_edit4(
            f"{label}##{key}", inker_colors._vec(tuple(values[key])), inker_colors.FLAGS
        )
        if changed:
            values[key] = inker_colors._to_rgba(value)
        imgui.same_line()
        # The colour a user wants is nearly always the one they are painting
        # with, and picking it twice in two widgets is the friction this button
        # exists to remove.
        if controls.button(f"use FG##fg{key}"):
            values[key] = tuple(state.fg)
        return
    if key in filters.TOGGLE_PARAMS:
        # Stored as 0.0/1.0, not as a bool: the registry holds one kind of value
        # and ``apply_named`` passes it straight through.
        changed, on = controls.checkbox(f"{label}##{key}", bool(values[key]))
        if changed:
            values[key] = 1.0 if on else 0.0
        return
    choices = filters.CHOICE_PARAMS.get(key)
    if choices is not None:
        # ``labeled_combo`` and not ``combo``: imgui draws a combo's label to its
        # *right* and the default width is -1, so a named combo puts its name
        # past the content region, where same_line clips rather than wraps and
        # the name is simply not drawn. ``widgets.combo``'s docstring is where
        # that rule is written down.
        picked = widgets.labeled_combo(
            label, str(values[key]), [(str(choice), str(choice)) for choice in choices]
        )
        if picked != str(values[key]):
            values[key] = next(c for c in choices if str(c) == picked)
        return
    low, high = filters.RANGES.get(key, (0.0, 1.0))
    imgui.set_next_item_width(sp(160))
    changed, value = controls.slider_float(f"{label}##{key}", float(values[key]), low, high)
    if changed:
        values[key] = float(value)


def _filter_popup(ctx: Any, tab: Any) -> None:
    from ..inker import filters

    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(FILTER_POPUP):
        # imgui closes a popup on a click outside, and the user did not answer
        # the question -- so the pixels on screen are a preview nobody
        # approved. Cancel, never commit.
        if state.filter_open:
            state.filter_open = False
            tab.doc.cancel_filter()
        return
    widgets.popup_chrome(_imgui=imgui)

    name = widgets.labeled_combo(
        "Filter", state.filter_name, [(key, key) for key in filters.FILTERS]
    )
    if name != state.filter_name:
        state.filter_name = name
    values = _filter_values(state, state.filter_name)
    for key in filters.FILTERS[state.filter_name][0]:
        _filter_control(state, values, key)
    if controls.button("Reset##filterreset"):
        # Back to what opening the popup gave, not to the identity defaults --
        # Reset on Invert that unticked all three channels would be a button
        # that turns the filter off.
        state.filter_params[state.filter_name] = filters.popup_values(state.filter_name)

    # Every frame, not only on a change: the combo above can switch filters,
    # and a preview that only ran on a slider move would leave the last
    # filter's pixels under the new filter's controls.
    tab.doc.preview_filter(state.filter_name, **_filter_values(state, state.filter_name))

    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if controls.button("Apply", (sp(90), 0)):
        tab.doc.commit_filter()
        state.filter_open = False
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    _apply_to_range(ctx, tab)
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss -- the trap the params popup in Clay documents.
    if controls.button("Cancel", (sp(90), 0)):
        tab.doc.cancel_filter()
        state.filter_open = False
        imgui.close_current_popup()
    imgui.end_popup()


def _apply_to_range(ctx: Any, tab: Any) -> None:
    """Run the filter over every cel of the timeline's range, in one step.

    **Cancels the preview session first**, which is the whole of what makes
    this safe beside Apply: the session has already written its preview into
    the cel on screen, and running the range filter over that cel would filter
    an already-filtered plane -- the compounding ``preview_filter`` exists to
    avoid, arriving by a different door. Cancelling puts the pixels back, and
    ``filter_range`` then reads every cel including this one exactly once.

    Disabled with no range rather than hidden, the rule the timeline's own menu
    follows: a button that appears and disappears is one the user has to
    rediscover.
    """
    state = inker_mode.ensure(ctx)
    rect = tab.range_sel
    imgui.begin_disabled(tab.busy or rect is None or tab.doc.anim is None)
    if controls.button("Apply to range", (sp(120), 0)):
        values = dict(_filter_values(state, state.filter_name))
        tab.doc.cancel_filter()
        state.filter_open = False
        tab.doc.filter_range(state.filter_name, values, *rect)
        imgui.close_current_popup()
    imgui.end_disabled()
    if rect is None:
        widgets.help_marker(
            "Drag across the timeline to select a range of cels first. Every"
            " distinct cel in it is filtered once, so a linked cel is filtered"
            " once however many frames it appears on."
        )


# --- importing a sprite sheet -----------------------------------------------

SHEET_IMPORT_POPUP = "inker-sheet-import"


def _sheet_import(ctx: Any) -> None:
    """The Import sheet button, and the grid popup once a file is chosen."""
    state = inker_mode.ensure(ctx)
    # ``tooltip=`` rather than a trailing ``help_marker`` -- the Animation row
    # says why a full-width button cannot be followed by one.
    if controls.button(
        "Import sprite sheet...",
        (-1, 0),
        tooltip=(
            "Slices any image into one frame per cell, row by row. For a sheet"
            " this app generated, opening the draft from the library carries"
            " its directions and tags as well; this is for a sheet from"
            " anywhere else."
        ),
    ):
        inker_mode.ask_import_sheet(ctx)
    if state.sheet_import is not None and not state.sheet_import_open:
        state.sheet_import_open = True
        imgui.open_popup(SHEET_IMPORT_POPUP)
    _sheet_import_popup(ctx, state)
    _aseprite_import(ctx)


def _aseprite_import(ctx: Any) -> None:
    """The other import, and no popup at all.

    Beside the sheet import because the two answer the same question -- "this
    drawing was made somewhere else" -- and it needs no popup for the reason
    that one does: a sheet has to be told how to cut and an Aseprite file
    already says where everything is.
    """
    if controls.button(
        "Import Aseprite file...",
        (-1, 0),
        tooltip=(
            "Reads an .aseprite or .ase file: layers, groups, the timeline,"
            " tags, slices, and cels shared between frames as shared cels here"
            " too. Reading only -- the import opens as an unsaved document, so"
            " saving it writes an .ora and never back over the file it came"
            " from. Anything dropped on the way in is named in a message."
        ),
    ):
        inker_mode.ask_import_aseprite(ctx)


def _pair(label: str, value: tuple[int, int], low: int = 0) -> tuple[int, int]:
    """Two small integer fields on one row. -> the pair, floored at ``low``."""
    imgui.set_next_item_width(sp(70))
    _changed_x, x = controls.input_int(f"##{label}x", int(value[0]), 1, 8)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    _changed_y, y = controls.input_int(f"##{label}y", int(value[1]), 1, 8)
    imgui.same_line()
    widgets.muted(label)
    return (max(low, int(x)), max(low, int(y)))


def _sheet_import_popup(ctx: Any, state: Any) -> None:
    from ..inker import sheetin

    if not imgui.begin_popup(SHEET_IMPORT_POPUP):
        # imgui closes a popup on a click outside, and the picture is a
        # megabyte or two: dropping it here is what keeps a cancelled import
        # from pinning the atlas for the rest of the session.
        if state.sheet_import_open:
            state.sheet_import_open = False
            state.sheet_import = None
        return
    widgets.popup_chrome(_imgui=imgui)
    if state.sheet_import is None:
        imgui.end_popup()
        return
    atlas, title = state.sheet_import
    height, width = atlas.shape[:2]
    widgets.muted(f"{title} - {width} x {height}")

    state.sheet_cell = _pair("cell", state.sheet_cell, low=1)
    state.sheet_offset = _pair("offset", state.sheet_offset)
    state.sheet_padding = _pair("padding", state.sheet_padding)
    imgui.set_next_item_width(sp(70))
    _changed, count = controls.input_int("frames (0 = all)", int(state.sheet_count), 1, 8)
    state.sheet_count = max(0, int(count))

    # The count the numbers above actually produce, computed every frame from
    # the same function the import runs -- so what the popup promises and what
    # the import does cannot disagree, and a mistyped cell size says so here
    # rather than in a document twenty frames long.
    try:
        rects = sheetin.grid_rects(
            (int(width), int(height)),
            state.sheet_cell,
            state.sheet_offset,
            state.sheet_padding,
            state.sheet_count or None,
        )
        problem = ""
    except ValueError as exc:
        rects, problem = [], str(exc)
    if problem:
        widgets.text_colored(theme.WARN, problem)
    else:
        widgets.muted(f"{len(rects)} frames")

    imgui.dummy((0, 4))
    imgui.begin_disabled(not rects)
    if controls.button("Import", (sp(90), 0)) and inker_mode.import_sheet(ctx):
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if controls.button("Cancel##sheetin", (sp(90), 0)):
        state.sheet_import = None
        state.sheet_import_open = False
        imgui.close_current_popup()
    imgui.end_popup()


# --- palette conversion -----------------------------------------------------
#
# The filter popup's mechanism against a different session, and for the same
# reason: nobody can predict what Floyd-Steinberg does to *their* drawing on
# *their* palette, and a conversion you have to undo to judge is one you stop
# trying. The document owns the session, so nothing here holds pixels, and
# committing is the ordinary one-undo ``convert_to_palette``.
#
# Opened and drawn from ``panes/inker_colors`` rather than from ``_canvas_ops``
# below, even though it is written here beside its twin: an imgui popup is
# matched by an id computed off the current id stack, and the colours pane and
# this one are different child windows -- ``open_popup`` here and
# ``begin_popup`` there would never meet. The palette section is where the
# controls belong anyway.

CONVERT_POPUP = "inker-convert"


def _convert_table(state: Any, doc: Any) -> list[tuple[int, int, int, int]]:
    """The table the conversion would use: the document's own, or a built one.

    No third choice and no source selector. A document that *has* a palette is
    being re-dithered onto the palette it has -- offering to replace it here
    would put "change my colours" and "change how my pixels reach them" behind
    one button. A document that has none has nothing else to convert to but its
    own pixels, and the swatch row is a session's favourites rather than a
    statement about this file (see ``inker_colors._indexed``).
    """
    if doc.palette:
        return [tuple(c) for c in doc.palette]
    return doc.built_palette(state.convert_max)


def open_convert(ctx: Any, tab: Any) -> None:
    from ..inker import dither

    state = inker_mode.ensure(ctx)
    if tab.busy:
        return
    if state.convert_method not in dither.METHODS:
        state.convert_method = dither.METHODS[0]
    if not tab.doc.begin_convert():
        ctx.toast("There is nothing to convert.", "warn")
        return
    # After ``begin_convert``, which is what makes the built table read the
    # session's snapshot rather than a preview of itself.
    state.convert_table = _convert_table(state, tab.doc)
    state.convert_uid = tab.uid
    imgui.open_popup(CONVERT_POPUP)


def convert_popup(ctx: Any, tab: Any) -> None:
    """Draw the open session's popup, or settle a session nothing will answer.

    Called unconditionally from ``inker_colors.draw`` -- including with no tab
    at all -- because this is the only per-frame hook the session has, and every
    way it can be stranded is a frame where the popup does not get drawn.

    Two of those ways are handled below and neither may act on ``tab``:

    * ``begin_popup`` says no. The user clicked outside, or the pane stopped
      being submitted (leaving Inker mode), and imgui closed the popup. An
      unanswered question is not a yes, so the session is cancelled.
    * the popup is up but this pane is drawing a *different* document. The user
      switched tabs. The session belongs to the tab it was opened on and nothing
      about the new one has anything to do with it.

    In both, ``end_convert_session`` resolves the owner by uid. Reaching for
    ``tab`` here -- which the filter popup this was cloned from does -- is
    exactly how a tab switch came to restore planes that were never previewed
    while the previewed document kept a dither nobody approved, with no hook
    left to take it back.
    """
    from ..inker import dither

    state = inker_mode.ensure(ctx)
    owner = state.get(state.convert_uid) if state.convert_uid else None
    if not imgui.begin_popup(CONVERT_POPUP):
        inker_mode.end_convert_session(ctx)
        return
    widgets.popup_chrome(_imgui=imgui)
    if owner is None or tab is None or owner.uid != tab.uid:
        inker_mode.end_convert_session(ctx)
        imgui.close_current_popup()
        imgui.end_popup()
        return

    state.convert_method = widgets.labeled_combo(
        "Dither", state.convert_method, [(key, key) for key in dither.METHODS]
    )
    if not tab.doc.palette:
        imgui.set_next_item_width(sp(160))
        changed, value = controls.slider_int("Colours", int(state.convert_max), 2, 64)
        if changed:
            state.convert_max = int(value)
        if imgui.is_item_deactivated_after_edit():
            # On release, not on every frame of the drag: building a table is a
            # pass over every plane in the document followed by a median cut,
            # and a slider dragged across its range would ask for sixty of them.
            # This is the only control that changes what the table would be.
            state.convert_table = _convert_table(state, tab.doc)
    widgets.muted(f"{len(state.convert_table)} colour(s); this frame is previewed")
    widgets.help_marker(
        "The preview shows the current frame. Applying converts the whole "
        "document -- every layer and every frame -- as one undo step, because "
        "the palette it installs constrains every write afterwards."
    )

    # Every frame, not only on a change: the combo and the slider can both move
    # the answer, and a preview that only ran on a change would leave the last
    # method's pixels under the new method's controls.
    tab.doc.preview_convert(state.convert_table, state.convert_method)

    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if controls.button("Apply##convert", (sp(90), 0)):
        table = list(state.convert_table)
        if tab.doc.commit_convert(table, state.convert_method):
            state.palette_slot = 0
            state.palette_slots = []
            state.palette_usage = None
            ctx.toast(f"Converted to {len(table)} colour(s).", "success")
        state.convert_uid = ""
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss.
    if controls.button("Cancel##convert", (sp(90), 0)):
        tab.doc.cancel_convert()
        state.convert_uid = ""
        imgui.close_current_popup()
    imgui.end_popup()
