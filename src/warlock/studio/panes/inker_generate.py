"""Where a drawing leaves the Inker: the four service-backed verbs, as buttons.

They have always existed -- ``Make 3D``, ``Save as reference``, ``Add to
Packwright`` and ``Revert to original`` are registered ops in
:mod:`~warlock.studio.inker_ops` -- but they were rows in the File menu and
nowhere else. A user who has drawn something and wants a mesh out of it looks
at the panel column, not at a menu, and every *other* workspace answers that
look: Clay has its bridge pane, Plotter has its files pane, Packwright has its
export pane. This is the Inker's.

**No new pipeline work.** Each button dispatches the same op the menu row does,
through :func:`inker_menu.activate`, so a parameterised op still opens its
sheet and ``pending_dialog`` still works.

Generating *into* a layer was called "a separate programme, deliberately not
started here" when this pane was written. It shipped on 2026-08-30 and it is
not one of these buttons: masked regeneration lives on the Edit menu and in
:mod:`~warlock.studio.inker.inpaint`, because it acts on a *selection* inside
the open document rather than sending the document somewhere. The four verbs
below are still the ones that hand a drawing to another workspace, which is
what this pane is for.

Every button is a :func:`widgets.disabled_button` carrying
``inker_ops.reason_for``: this pane's whole reason for existing is that a verb
should be visible before it is available, and a greyed button with no sentence
is worse than a menu row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from .. import anchors, controls, inker_export, inker_mode, inker_ops, tokens, widgets
from ..inker import sheetout
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_menu

#: The least this pane may be squeezed to, in design px: the file header's
#: two rows and status line, the Export block's five doors and its collapsed
#: options header, the exits heading, four full-width buttons and the link line.
#: Raised from 270 when the five exports moved here from the timeline row
#: they were overflowing out of (2026-09-05).
GENERATE_FLOOR = 360.0

#: The ops, in the order the File menu registers them, with the one-line note
#: that says what each is *for*. The names are checked against the registry at
#: import time by :func:`ops`, so a rename cannot leave a dead button here.
OPS: tuple[tuple[str, str], ...] = (
    ("send_to_3d", "Reconstruct a mesh from this drawing."),
    ("save_as_reference", "Put it in the library as a reference image."),
    ("add_to_packwright", "Send it to the atlas packer as a source."),
    ("revert", "Throw away the edits and reopen what was there before."),
)


def ops() -> list[Any]:
    """The four ops, resolved. ``KeyError`` here rather than a dead button."""

    return [inker_ops.get(name) for name, _note in OPS]


def draw(ctx: Any) -> None:
    anchors.mark_window("inker/generate")
    state = inker_mode.ensure(ctx)
    tab = state.active
    if tab is not None:
        # The shared document header every workspace carries (2026-09-05).
        # Inker was the one mode with no file block in any pane -- its verbs
        # were menu rows only -- so a user who had just met Save in Clay's
        # pane looked here and found nothing.
        widgets.section("Drawing file")
        widgets.document_header(
            tab,
            new=lambda: inker_mode.new_document(ctx, 1024, 1024),
            open_=lambda: inker_mode.ask_open(ctx),
            save=lambda: inker_mode.save(ctx, tab),
            save_as=lambda: inker_mode.save_as(ctx, tab),
            saving=tab.busy,
        )
        imgui.dummy((0, sp(tokens.SP_2)))
        # The Undo/Redo pair and the history popover every other bridge
        # draws. Four bridges' comments said "while Inker drew the same pair
        # twice"; it drew it nowhere (2026-09-05).
        widgets.history_block(
            ctx,
            tab,
            key="inker",
            undo=lambda: tab.doc.undo(),
            redo=lambda: tab.doc.redo(),
            step=lambda index: inker_mode.step_history(ctx, tab, index),
        )
        imgui.dummy((0, sp(tokens.SP_2)))
    # The five doors, before the exits, in the shape ``packwright_bridge``
    # draws its own Export block: what writes files for another application is
    # its own heading, above the verbs that move a drawing *inside* the app.
    _exports(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    # Inker's exits -- Make 3D, Save as reference, Add to Packwright -- under
    # the heading every workspace puts over the same verbs.
    widgets.exits()
    # After the heading, never before it: ``help_button`` is a ``same_line``.
    manual_render.help_button(ctx, "inker-generate")

    for op, (_name, note) in zip(ops(), OPS, strict=True):
        enabled = bool(op.enabled(state, tab))
        if widgets.disabled_button(
            f"{op.label}##inkgen/{op.name}",
            enabled,
            (-1, 0),
            reason=inker_ops.reason_for(op, state, tab),
            tooltip=note,
        ):
            inker_menu.activate(ctx, op)
        widgets.muted(note)
        imgui.dummy((0, sp(4)))

    _link(tab)
    if tab is None:
        # The recent list, which every other bridge offers and this one did
        # not; ``inker_mode.recent_paths`` existed with no pane reading it.
        widgets.recent_files(
            inker_mode.recent_paths(ctx),
            lambda path: inker_mode.open_path(ctx, Path(path)),
        )


def _exports(ctx: Any, tab: Any) -> None:
    """The five doors, drawn once from :mod:`~warlock.studio.inker_export`.

    Every label, tooltip and refusal sentence is that module's, not this
    pane's: the same records the File menu reads, so a door cannot be live in
    one presentation and grey in the other, and a grey one always carries a
    sentence (``door_state`` never returns ``(False, "")``).

    Drawn whether or not a document is open -- refused with *No drawing is
    open.* when there is none -- for the reason this whole pane exists: a verb
    should be visible before it is available.
    """
    widgets.section("Export")
    for door in inker_export.doors():
        enabled, reason = inker_export.door_state(door, tab)
        if widgets.disabled_button(
            f"{door.icon} {door.label}##inkexp/{door.key}",
            enabled,
            (-1, 0),
            reason=reason,
            tooltip=door.tooltip,
        ):
            inker_export.open_door(ctx, tab, door.key)
    if tab is not None and controls.collapsing_header("Sheet options##inkexpopts"):
        # Collapsed by default, and that is the point: the five doors are what
        # this block is for, and nine knobs above the exits pushed Make 3D and
        # its three neighbours off the bottom of the pane (the harness called
        # them *clipped*). Every one of them keeps its remembered value while
        # the header is shut -- closing it changes nothing that is written.
        _export_options(ctx)


def _export_options(ctx: Any) -> None:
    """The knobs the sheet doors read, and nothing else reads.

    They were the trailing of the timeline's export row; they came with the
    exports because each is an argument to the file that is written rather
    than to the clip that is played. App-level, on ``ctx.state.inker``, which
    is where they already lived.
    """
    state = ctx.state.inker
    scale = widgets.combo(
        "Scale##inkerscale",
        str(int(state.export_scale)),
        list(inker_export.EXPORT_SCALES),
        sp(72),
    )
    state.export_scale = max(1, int(scale))
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Magnifies every export by a whole number, nearest neighbour -- "
            "each pixel drawn N times and nothing resampled. The sheet "
            "sidecar is built on the scaled size, so its cells and trims "
            "describe the file that is written; sidecars bound for "
            "Packwright are not scaled."
        )
    arrange_key = state.export_arrange or "grid"
    chosen = widgets.combo(
        "Arrange##inkerarrange",
        arrange_key,
        list(inker_export.ARRANGE_OPTIONS),
        sp(120),
    )
    state.export_arrange = None if chosen == "grid" else chosen
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "How frames pack into the sheet. Grid wraps at the atlas "
            "ceiling, same as it always has. Horizontal/Vertical strip is "
            "one row or column. Rows.../Columns... fixes that side's count "
            "and wraps the rest. A document with its own directional "
            "layout (turnaround, walk) keeps that fixed grid instead."
        )
    if state.export_arrange in sheetout.COUNTED_ARRANGES:
        imgui.set_next_item_width(sp(72))
        changed, value = controls.input_int("Count##inkerwrap", state.export_wrap, 1, 1)
        if changed:
            state.export_wrap = max(1, int(value))
    changed, value = widgets.toggle("Merge", state.export_merge, tag="inker-export-merge")
    if changed:
        state.export_merge = value
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Duplicate frames -- byte-identical, which a linked cel is for "
            "free -- share one cell instead of one each. Each frame still "
            "gets its own duration in the sidecar; only the pixels are "
            "shared. Refused for a document with its own directional "
            "layout, whose cells are poses by yaws rather than frames -- "
            "turn it off to export one."
        )
    imgui.same_line()
    changed, value = widgets.toggle(
        "Skip empty", state.export_skip_empty, tag="inker-export-skip-empty"
    )
    if changed:
        state.export_skip_empty = value
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "A fully-transparent frame gets no cell at all, and the "
            "sidecar names which frames it dropped -- and renumbers the "
            "tags onto what survived. Refused for a document with its own "
            "directional layout, for the same reason Merge is."
        )
    imgui.same_line()
    changed, value = widgets.toggle("Trim", state.export_trim, tag="inker-export-trim")
    if changed:
        state.export_trim = value
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Each cell shrinks to the largest trimmed frame's size, with "
            "every frame's own trimmed pixels placed flush in its "
            "corner. The sidecar's per-cell trim rectangle still names "
            "where that came from in the full frame, for an importer "
            "that wants to put it back."
        )
    # Real captions here, not the ``##`` hidden ids the toolbar row needed: a
    # column can afford a label, where the row could not and lost Ext and the
    # help button off its end for it.
    imgui.set_next_item_width(sp(72))
    changed, value = controls.drag_int("Padding", state.export_padding, 1, 0, 64)
    if changed:
        state.export_padding = max(0, int(value))
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "A border around the atlas and a gutter between every cell, in "
            "pixels. Zero is the sheet this always packed."
        )
    imgui.set_next_item_width(sp(72))
    changed, value = controls.drag_int("Extrude", state.export_extrude, 1, 0, 32)
    if changed:
        state.export_extrude = max(0, int(value))
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Repeats each cell's own border pixels outward into its gutter, "
            "so a filtered texture sampling just past a sprite's edge finds "
            "that sprite's own colour rather than its neighbour's. Padding "
            "must be at least twice this, so two neighbours extruding into "
            "one gutter cannot meet."
        )
    imgui.set_next_item_width(-1)
    state.export_template = widgets.input_text(
        "##inkertemplate",
        state.export_template,
        max_length=80,
        hint=inker_export.EXPORT_TEMPLATE_HINT,
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Filename template. Empty is the plain frame numbering "
            "(PNG sequence) or the tag/layer name (a split export). "
            "{title}, {tag}, {frame} (0000) and {layer} are the whole "
            "vocabulary; a PNG sequence reads {title} and {frame}, a "
            "per-tag or per-layer split reads {title} and {tag}/{layer}."
        )


def link_line(tab: Any) -> str:
    """What this document is attached to, in one sentence.

    Pure and public because it is the *explanation* of a greyed Revert, and an
    explanation is worth an assertion rather than a screenshot.
    """
    if tab is None:
        return "No drawing is open."
    if not tab.linked:
        return "Not linked to a library asset -- Save as reference makes one."
    kind = tab.link_kind or "job"
    original = "with the original kept" if tab.has_original else "no original kept yet"
    return f"Linked to {kind} {tab.job_id} ({original})."


def _link(tab: Any) -> None:
    widgets.muted(link_line(tab))
