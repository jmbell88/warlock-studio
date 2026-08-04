"""The layers panel: the stack, top first, with its per-layer controls.

Top first because that is how every editor shows it and how ORA stores it --
the engine's list is bottom-first (painter's order), so this is the one place
that reverses, and it does so in one expression rather than by keeping a second
ordering around.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import paint, paint_mode, theme, widgets
from . import paint_textures

THUMB = 40.0


def draw(ctx: Any) -> None:
    state = paint_mode.ensure(ctx)
    tab = state.active
    widgets.section("layers")
    if tab is None:
        widgets.muted("Nothing open.")
        return
    doc = tab.doc
    _actions(ctx, doc)
    imgui.dummy((0, 4))

    # Reversed: the engine's list is painter's order, the panel is not.
    for index in range(len(doc.stack) - 1, -1, -1):
        _row(ctx, tab, doc, index)


def _actions(ctx: Any, doc: Any) -> None:
    if imgui.button("Add"):
        doc.add_layer()
    imgui.same_line()
    if imgui.button("Copy"):
        doc.duplicate_layer()
    imgui.same_line()
    if widgets.disabled_button("Delete", len(doc.stack) > 1):
        doc.remove_layer()
    if widgets.disabled_button("Merge down", doc.stack.active_index > 0):
        doc.merge_down()
    imgui.same_line()
    if widgets.disabled_button("Flatten", len(doc.stack) > 1):
        doc.flatten_layers()

    layer = doc.stack.active
    imgui.set_next_item_width(-1)
    changed, value = imgui.slider_float("Opacity", layer.opacity, 0.0, 1.0)
    if changed:
        # Live while dragging, but only one undo step: set the value directly
        # for the preview and record the step when the drag is released.
        layer.opacity = value
        doc.invalidate_all()
    if imgui.is_item_deactivated_after_edit():
        doc.set_layer_props(opacity=layer.opacity)
    blend = widgets.combo("Blend", layer.blend, [(m, m) for m in paint.BLEND_MODES])
    if blend != layer.blend:
        doc.set_layer_props(blend=blend)


def _row(ctx: Any, tab: Any, doc: Any, index: int) -> None:
    layer = doc.stack[index]
    imgui.push_id(f"layer{layer.uid}")
    active = index == doc.stack.active_index

    changed, visible = imgui.checkbox("##visible", layer.visible)
    if changed:
        doc.set_layer_props(index, visible=visible)
    imgui.same_line()

    texture = paint_textures.layer_thumb(ctx, tab, index)
    if texture is not None:
        imgui.image(widgets.texture_ref(texture), (THUMB, THUMB))
        imgui.same_line()

    imgui.begin_group()
    label = layer.name if layer.visible else f"{layer.name} (hidden)"
    if imgui.selectable(f"{label}##pick", active, 0, (0, THUMB * 0.5))[0]:
        doc.set_active_layer(index)
    _reorder(doc, index)
    imgui.text_colored(
        imgui.ImVec4(*theme.rgba(theme.MUTED)),
        f"{layer.blend}  {layer.opacity * 100:.0f}%",
    )
    imgui.end_group()

    if imgui.begin_popup_context_item("layer-menu"):
        if imgui.selectable("Rename", False)[0]:
            _ask_rename(ctx, doc, index)
        if imgui.selectable("Move up", False)[0]:
            doc.move_layer(index, index + 1)
        if imgui.selectable("Move down", False)[0]:
            doc.move_layer(index, index - 1)
        imgui.end_popup()
    imgui.pop_id()


def _reorder(doc: Any, index: int) -> None:
    """Drag a layer's name onto another row to move it there.

    imgui's drag-and-drop payload carries the *index*, and the drop reads the
    stack again -- so a reorder mid-drag cannot make the drop land on a
    different layer than the one under the cursor.
    """
    if imgui.begin_drag_drop_source(imgui.DragDropFlags_.source_no_hold_to_open_others.value):
        imgui.set_drag_drop_payload_py_id("paint-layer", index)
        imgui.text(doc.stack[index].name)
        imgui.end_drag_drop_source()
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id("paint-layer")
        if payload is not None:
            source = int(payload.data_id)
            if 0 <= source < len(doc.stack) and source != index:
                doc.move_layer(source, index)
        imgui.end_drag_drop_target()


def _ask_rename(ctx: Any, doc: Any, index: int) -> None:
    from .. import dialogs

    layer = doc.stack[index]
    ctx.prompts.ask(
        dialogs.Prompt(
            title="Rename layer",
            label="Name",
            value=layer.name,
            on_accept=lambda text: doc.set_layer_props(index, name=text[:60]),
        )
    )
