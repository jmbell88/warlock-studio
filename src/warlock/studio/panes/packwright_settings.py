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
from .. import packwright_mode, widgets
from ..manual import render as manual_render
from ..packwright.layout import MODES

SIZES = (256, 512, 1024, 2048, 4096, 8192)

_MODE_NOTES = {
    "grid": "Uniform cells, sized to the largest sprite. Exports a .tsx as well, "
    "so the result can be used as a tileset in Plotter or Tiled.",
    "maxrects": "Packs tightly and irregularly. Smaller atlas, but the cells are "
    "not a grid -- an importer has to read the JSON.",
}


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("packing")
    manual_render.help_button(ctx, "packwright-settings")

    if tab is None:
        widgets.muted("Start or open an atlas first.")
        return

    settings = tab.doc.settings
    editable = not tab.busy

    mode = widgets.labeled_combo("Mode", settings.mode, [(m, m.title()) for m in MODES])
    if editable and mode != settings.mode:
        packwright_mode.set_settings(ctx, tab, mode=mode)
    widgets.muted_wrapped(_MODE_NOTES.get(settings.mode, ""))

    imgui.dummy((0, 6))
    changed, trim = widgets.toggle("Trim transparent edges", settings.trim)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, trim=trim)

    changed, pot = widgets.toggle("Power-of-two atlas", settings.power_of_two)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, power_of_two=pot)

    imgui.dummy((0, 6))
    changed, padding = widgets.labeled_slider_int("Padding", settings.padding, 0, 16)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, padding=int(padding))

    changed, extrude = widgets.labeled_slider_int("Extrude", settings.extrude, 0, 8)
    if changed and editable:
        packwright_mode.set_settings(ctx, tab, extrude=int(extrude))
    if settings.extrude:
        widgets.muted_wrapped(
            "Border pixels are repeated into the gutter, so a filtered texture "
            "cannot sample the sprite next door."
        )

    imgui.dummy((0, 6))
    sizes = [(str(s), f"{s} px") for s in SIZES if s <= MAX_ATLAS_PX]
    picked = widgets.labeled_combo("Max size", str(settings.max_size), sizes)
    if editable and picked != str(settings.max_size):
        packwright_mode.set_settings(ctx, tab, max_size=int(picked))

    if not editable:
        imgui.dummy((0, 6))
        widgets.muted("Saving...")
