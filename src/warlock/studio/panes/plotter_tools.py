"""Plotter's left-top pane: the tools and the map's own properties.

The tools are what a click on the canvas *means*; the tileset pane owns which
tile it means, and the layers pane owns which layer it lands on. One control,
one owner -- the rule the two generate panes already follow for ``platform``.
"""

from __future__ import annotations

from typing import Any

from .. import icons, plotter_mode, plotter_state, widgets
from ..manual import render as manual_render

# The tool letters, drawn on the buttons. From ``plotter_state.TOOLS`` rather
# than restated, so a tool added there cannot get a button with no key or a key
# with no button.
_ICONS = {
    "stamp": icons.BRUSH,
    "erase": icons.ERASER,
    "fill": icons.PAINT_BUCKET,
    "rect": icons.SQUARE,
    "pick": icons.PIPETTE,
    "object": icons.FLAG,
}


def _tool_grid(state: Any) -> None:
    from imgui_bundle import imgui

    # Width from the style rather than a literal gap: ``theme.apply`` sets
    # item_spacing through ``sp()``, so a grid that subtracted a hard-coded 8
    # was right at UI scale 1.0 and short by five pixels per gap at 1.5 --
    # which is what dropped the raster editor's fifth toolbox column.
    width = widgets.grid_width(3)
    for index, (key, label, letter) in enumerate(plotter_state.TOOLS):
        if index % 3:
            imgui.same_line()
        active = state.tool == key
        if active:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        if imgui.button(f"{_ICONS.get(key, icons.SQUARE)}##tool-{key}", (width, 0)):
            state.tool = key
        if active:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label} ({letter})")


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("tools")
    manual_render.help_button(ctx, "plotter-tools")
    _tool_grid(state)
    imgui.dummy((0, 6))

    if tab is None:
        widgets.muted("Open or start a map to draw on.")
        return

    doc = tab.doc
    _, state.grid = widgets.toggle("Grid (Ctrl+G)", state.grid)
    _, state.show_objects = widgets.toggle("Show objects", state.show_objects)

    imgui.dummy((0, 6))
    widgets.section("map")
    widgets.muted(f"{doc.width} x {doc.height} tiles, {doc.tile_w} x {doc.tile_h} px")
    widgets.muted(f"{doc.pixel_width} x {doc.pixel_height} px overall")

    if tab.busy:
        widgets.muted("Saving...")
        return

    imgui.dummy((0, 6))
    if widgets.header("Resize", default_open=False, persist_key="plotter/resize"):
        _resize_form(ctx, tab)


def _resize_form(ctx: Any, tab: Any) -> None:
    """Grow or crop the grid, anchoring the old content by an offset.

    Cached under ``state.preview`` per tab, so typing a width does not fight
    with the document and switching tabs does not carry a half-typed number
    onto a different map.
    """
    from imgui_bundle import imgui

    key = f"plotter_resize:{tab.uid}"
    form = ctx.state.preview.get(key)
    if form is None or form.get("for") != (tab.doc.width, tab.doc.height):
        form = {
            "for": (tab.doc.width, tab.doc.height),
            "w": tab.doc.width,
            "h": tab.doc.height,
            "dx": 0,
            "dy": 0,
        }
        ctx.state.preview[key] = form

    _, form["w"] = widgets.labeled_slider_int("Width", int(form["w"]), 1, 512)
    _, form["h"] = widgets.labeled_slider_int("Height", int(form["h"]), 1, 512)
    _, form["dx"] = widgets.labeled_slider_int("Offset X", int(form["dx"]), -64, 64)
    _, form["dy"] = widgets.labeled_slider_int("Offset Y", int(form["dy"]), -64, 64)
    imgui.dummy((0, 4))
    if widgets.primary_button("Resize", (-1, 0)):
        try:
            tab.doc.resize(
                int(form["w"]), int(form["h"]),
                offset_x=int(form["dx"]), offset_y=int(form["dy"]),
            )
        except ValueError as exc:
            ctx.toast(str(exc), "error")
            return
        ctx.state.preview.pop(key, None)
        tab.view.fitted = False
