"""The layers panel: the stack, top first, with its per-layer controls.

Top first because that is how every editor shows it and how ORA stores it --
the engine's list is bottom-first (painter's order), so this is the one place
that reverses, and it does so in one expression rather than by keeping a second
ordering around.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, inker, inker_mode, theme, widgets
from ..manual import render as manual_render
from . import inker_textures

THUMB = 40.0

# Opacity the active layer held when its slider drag began, keyed by layer uid.
# Only ever holds an entry while a drag is in flight.
_opacity_drag: dict[int, float] = {}


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    tab = state.active
    widgets.section("layers")
    manual_render.help_button(ctx, "inker-layers")
    if tab is None:
        widgets.empty_state(
            icons.LAYERS,
            "No drawing open",
            "Ctrl+N starts one, Ctrl+O opens a file.",
        )
        return
    doc = tab.doc
    # Every control below restructures the stack, and a save is encoding that
    # stack on a task thread. The canvas already refuses strokes mid-save for
    # the same reason; disabling is the panel's version of that, and it says so
    # on screen rather than swallowing clicks.
    imgui.begin_disabled(tab.busy)
    _actions(ctx, doc)
    imgui.dummy((0, 4))

    # J86, and inside the disable for the reason everything else here is.
    needle = widgets.list_filter(ctx, "inker-layers", len(doc.stack))
    # Reversed: the engine's list is painter's order, the panel is not.
    for index in range(len(doc.stack) - 1, -1, -1):
        if needle and needle not in (doc.stack[index].name or "").lower():
            continue
        _row(ctx, tab, doc, index)
    imgui.end_disabled()


def _actions(ctx: Any, doc: Any) -> None:
    if imgui.button("Add"):
        doc.add_layer()
    imgui.same_line()
    if imgui.button("Copy"):
        doc.duplicate_layer()
    imgui.same_line()
    if widgets.disabled_button("Delete", len(doc.stack) > 1):
        doc.remove_layer()
    # Both are refused outright on an animated document -- they are defined over
    # one stack and an animated document has one per frame. Disabling says so
    # before the click rather than after it.
    restructure = doc.can_restructure
    if widgets.disabled_button("Merge down", restructure and doc.stack.active_index > 0):
        doc.merge_down()
    imgui.same_line()
    if widgets.disabled_button("Flatten", restructure and len(doc.stack) > 1):
        doc.flatten_layers()
    if not restructure:
        widgets.muted("Merge and flatten are unavailable while animated.")

    layer = doc.stack.active
    changed, value = widgets.labeled_slider_float("Opacity", layer.opacity, 0.0, 1.0)
    widgets.help_marker(
        "The active layer's opacity. Dragging previews it live and records one "
        "undo step when you let go."
    )
    if changed:
        # Live while dragging, but only one undo step: set the value directly
        # for the preview and record the step when the drag is released. The
        # value it started from has to be remembered here -- by release the
        # layer already holds the new one, and asking the document to diff it
        # against itself records nothing at all.
        _opacity_drag.setdefault(layer.uid, layer.opacity)
        layer.opacity = value
        doc.invalidate_all()
    if imgui.is_item_deactivated_after_edit():
        was = _opacity_drag.pop(layer.uid, None)
        if was is not None:
            doc.set_layer_props(opacity=layer.opacity, was={"opacity": was})
    blend = widgets.combo("Blend", layer.blend, [(m, m) for m in inker.BLEND_MODES])
    widgets.help_marker(
        "How this layer combines with everything under it. Saved into the .ora "
        "so other editors read it the same way."
    )
    if blend != layer.blend:
        doc.set_layer_props(blend=blend)
    changed, locked = imgui.checkbox("Lock alpha", layer.alpha_lock)
    widgets.help_marker(
        "Paints inside what is already on this layer and never past its edge: "
        "colours change, transparency does not. The eraser does nothing on a "
        "locked layer, because erasing is changing transparency."
    )
    if changed:
        doc.set_layer_props(alpha_lock=locked)


def _row(ctx: Any, tab: Any, doc: Any, index: int) -> None:
    layer = doc.stack[index]
    imgui.push_id(f"layer{layer.uid}")
    active = index == doc.stack.active_index

    changed, visible = imgui.checkbox("##visible", layer.visible)
    if changed:
        doc.set_layer_props(index, visible=visible)
    imgui.same_line()

    texture = inker_textures.layer_thumb(ctx, tab, index)
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
        f"{layer.blend}  {layer.opacity * 100:.0f}%"
        + ("  locked" if layer.alpha_lock else ""),
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
        imgui.set_drag_drop_payload_py_id("inker-layer", index)
        imgui.text(doc.stack[index].name)
        imgui.end_drag_drop_source()
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id("inker-layer")
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
