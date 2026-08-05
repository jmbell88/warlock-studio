"""The object list: what is in the document, and what is selected.

Top of the list is the *last* object added, matching every 3D package and
matching the raster editor's layers panel -- the document stores objects in the
order they were placed, so this is the one place that reverses, and it does so
in one expression rather than by keeping a second ordering around.

Selection here is the same selection the viewport shows, and it is deliberately
**not undoable**: clicking an object is not laborious to redo the way a lasso
is, an undoable selection would move ``history.head``, and a document would
then ask to be saved because the user looked at a different object.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, icons, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp

# Design pixels: every use goes through sp(), or the row keeps its 1.0x height
# while the glyph inside it grows with the UI scale.
ROW_HEIGHT = 24.0


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("outliner")
    manual_render.help_button(ctx, "clay-outliner")
    if tab is None:
        widgets.muted("Nothing open.")
        return
    doc = tab.doc
    if not doc.objects:
        widgets.empty_state(icons.LIST, "Empty document", "Add a primitive to start.")
        return

    # Visibility and selection do not change the document's *history*, but
    # renaming and deleting do -- and a save is encoding the document on a task
    # thread. The whole panel is disabled rather than half of it: a list where
    # two of four controls respond is more confusing than one that does not.
    imgui.begin_disabled(tab.saving)
    for index in range(len(doc.objects) - 1, -1, -1):
        _row(state, doc, doc.objects[index], index)
    imgui.end_disabled()


def _row(state: Any, doc: Any, obj: Any, index: int) -> None:
    selected = obj.uid in doc.selection
    imgui.push_id(str(obj.uid))

    eye = icons.EYE if obj.visible else icons.EYE_OFF
    if imgui.button(f"{eye}##vis", (sp(28), sp(ROW_HEIGHT))):
        doc.set_props(obj.uid, visible=not obj.visible)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Hidden objects do not render, export or pick.")
    imgui.same_line()

    width = imgui.get_content_region_avail().x - 32
    if state.renaming == obj.uid:
        imgui.set_next_item_width(width)
        name = widgets.input_text("##rename", obj.name, max_length=120)
        if name != obj.name:
            doc.set_props(obj.uid, name=name)
        if imgui.is_item_deactivated():
            state.renaming = 0
    else:
        label = obj.name or f"object {index}"
        if not obj.visible:
            widgets.text_colored(theme.MUTED, "")
        if imgui.selectable(f"{label}##row", selected, imgui.SelectableFlags_.none, (width, 0))[0]:
            doc.select([obj.uid])
        if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
            state.renaming = obj.uid

    imgui.same_line()
    if imgui.button(f"{icons.TRASH}##del", (sp(28), sp(ROW_HEIGHT))):
        doc.remove_object(obj.uid)
    imgui.pop_id()
