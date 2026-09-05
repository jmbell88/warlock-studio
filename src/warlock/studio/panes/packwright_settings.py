"""Packwright's left-bottom pane: the pack settings.

**Sole owner of ``PackSettings``.** Every control here goes through
``packwright_mode.set_settings``, which is the one place that both pushes the
undo step and marks the pack dirty -- a pane writing ``doc.settings`` directly
would change the atlas silently and leave the preview describing the old one.

``PackSettings`` validates on construction, so an impossible combination (an
extrude wider than half the padding) is refused with the reason rather than
clamped. Clamping would silently produce an atlas that bleeds at some zoom
levels, which is exactly the failure the rule exists to prevent.
"""

from __future__ import annotations

from typing import Any

from ...pipelines.sheet import MAX_ATLAS_PX
from .. import controls, icons, packwright_mode, tokens, widgets
from ..manual import render as manual_render
from ..packwright.layout import MODES
from ..tokens import sp

SIZES = (256, 512, 1024, 2048, 4096, 8192)

# The drag's own ceiling. Not ``MAX_SPRITES``: a tileset author picking a
# column count is choosing a *shape*, and a shape past a few dozen columns
# wide is not one anybody is choosing on purpose -- ``PackSettings`` itself
# stays the real limit for a hand-edited manifest past this.
MAX_COLUMNS = 256

_MODE_NOTES = {
    "grid": "Uniform cells, sized to the largest sprite. Exports a .tsx as well, "
    "so the result can be used as a tileset in Plotter or Tiled.",
    "maxrects": "Packs tightly and irregularly. Smaller atlas, but the cells are "
    "not a grid -- an importer has to read the JSON.",
}

_SCHEMA_OPTIONS = [
    ("array", "Array"),
    ("hash", "Hash"),
]

_SCHEMA_NOTES = {
    "array": "TexturePacker's JSON (Array) schema. Every engine and framework "
    "that reads this format already has a loader for it.",
    "hash": "TexturePacker's JSON (Hash) schema: the same frames, keyed by "
    "filename instead of listed in order, for a loader that looks one up by "
    "name. Two sprites sharing a name refuse this schema rather than lose one.",
}


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("Packing")
    manual_render.help_button(ctx, "packwright-settings")

    if tab is None:
        # The heading and nothing else. One voice for one empty state:
        # the canvas's ``nothing_open`` is it, and four panels each
        # repeating it reads as four separate problems.
        return

    settings = tab.doc.settings
    editable = not tab.busy

    # Greyed rather than live-and-discarded while the tab is saving: a slider
    # that moves and snaps back is a control lying about what it did, which is
    # the anti-pattern ``plotter_layers`` and ``clay_tools`` already fixed. The
    # ``editable`` guards below stay as a belt -- ``begin_disabled`` is imgui's
    # answer to *input*, and nothing here should depend on that alone to decide
    # whether a document changes.
    imgui.begin_disabled(not editable)

    mode = widgets.labeled_combo("Mode", settings.mode, [(m, m.title()) for m in MODES])
    if editable and mode != settings.mode:
        # An explicit ``columns`` only means something for a grid pack.
        # ``PackDoc.set_settings`` clears a stale one on a mode switch away
        # from grid (unless this same call also names ``columns``), so this
        # is the ordinary single-keyword call -- see its docstring for why.
        packwright_mode.set_settings(ctx, tab, mode=mode)
    widgets.muted_wrapped(_MODE_NOTES.get(settings.mode, ""))

    if settings.mode == "grid":
        imgui.dummy((0, sp(tokens.SP_2)))
        changed, columns = widgets.labeled_drag_int(
            "Columns", settings.columns or 0, 0, MAX_COLUMNS, speed=0.1
        )
        # One gesture, one step: a drag reports on every frame the pointer
        # moves, and ``set_settings``'s unconditional push (document.py:410)
        # turns one drag into dozens without this (2026-09-05 audit).
        controls.fold_undo(tab.doc.history)
        if changed and editable:
            packwright_mode.set_settings(ctx, tab, columns=int(columns) or None)
        widgets.muted_wrapped(
            "Zero packs the near-square grid the sprite count fits best. Set "
            "one to fix the column count for a tileset you index by it -- a "
            "power-of-two atlas may then carry dead space past the last "
            "column, and a .tsx export refuses rather than misread it."
        )

    imgui.dummy((0, sp(tokens.SP_2)))
    changed, trim = widgets.toggle("Trim transparent edges", settings.trim)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, trim=trim)
    if settings.mode == "grid":
        # Said rather than left to be discovered from the numbers: a grid pack
        # ignores this, because a tile moved to its own bounding box no longer
        # sits where the arithmetic slicing the tileset says it does. The
        # setting is kept, not overwritten, so switching to MaxRects brings the
        # user's answer back.
        widgets.muted_wrapped(
            "A grid pack does not trim: its cells are sliced by arithmetic, so "
            "every tile has to keep the position it was drawn at. This applies "
            "on MaxRects."
        )

    changed, pot = widgets.toggle("Power-of-two atlas", settings.power_of_two)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, power_of_two=pot)

    imgui.dummy((0, sp(tokens.SP_2)))
    changed, padding = widgets.labeled_slider_int("Padding", settings.padding, 0, 16)
    # One gesture, one step -- see the Columns drag above.
    controls.fold_undo(tab.doc.history)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, padding=int(padding))

    changed, extrude = widgets.labeled_slider_int("Extrude", settings.extrude, 0, 8)
    controls.fold_undo(tab.doc.history)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, extrude=int(extrude))
    if settings.extrude:
        widgets.muted_wrapped(
            "Border pixels are repeated into the gutter, so a filtered texture "
            "cannot sample the sprite next door."
        )

    imgui.dummy((0, sp(tokens.SP_2)))
    sizes = [(str(s), f"{s} px") for s in SIZES if s <= MAX_ATLAS_PX]
    picked = widgets.labeled_combo("Max size", str(settings.max_size), sizes)
    if editable and picked != str(settings.max_size):
        packwright_mode.set_settings(ctx, tab, max_size=int(picked))

    imgui.dummy((0, sp(tokens.SP_2)))
    schema = widgets.labeled_combo("Sidecar schema", settings.json_schema, _SCHEMA_OPTIONS)
    if editable and schema != settings.json_schema:
        packwright_mode.set_settings(ctx, tab, json_schema=schema)
    widgets.muted_wrapped(_SCHEMA_NOTES.get(settings.json_schema, ""))

    imgui.end_disabled()

    # **The one visible trigger.** Packing is automatic -- the centre pane's
    # pump repacks whenever ``pack_dirty`` is set -- and the only way to ask
    # for it *now* was a bare ``R`` nothing on screen mentioned. Two panes
    # meanwhile told the user to "press Pack", a button that has never
    # existed. So: the button exists, it names its key, and it lives in the
    # pane that owns ``PackSettings`` rather than on a bridge, because asking
    # for a repack is what you want after changing one of these.
    imgui.dummy((0, sp(tokens.SP_2)))
    sources = bool(tab.doc.sources)
    if widgets.disabled_button(
        f"{icons.REFRESH} Repack (R)",
        editable and sources,
        (-1, 0),
        reason=(
            "This atlas is being written; it comes back when it lands."
            if not editable
            else "Add an image first -- there is nothing to pack."
        ),
    ):
        packwright_mode.request_repack(ctx, tab)

    # Outside the disabled block: it is the *explanation* of why everything
    # above is greyed, and a greyed explanation reads as one more dead control.
    if not editable:
        imgui.dummy((0, sp(tokens.SP_2)))
        widgets.muted("Saving...")
