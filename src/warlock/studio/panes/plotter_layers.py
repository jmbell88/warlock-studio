"""Plotter's right-top pane: the layer stack, and an object's properties.

**This pane is the sole owner of object metadata.** A rectangle or a point is
placed on the canvas and named, classed and given typed properties here -- one
control, one owner, so there is never a second place a property can be set from.

The list is drawn top-first, which is the opposite of the document's order.
Every layered editor does it that way (the topmost layer is at the top of the
list) and a user arriving from one would read a bottom-first list as inverted.
"""

from __future__ import annotations

from typing import Any

from .. import icons, plotter_mode, widgets
from ..manual import render as manual_render
from ..plotter.tilemap import MapObject, ObjectLayer, TileLayer, new_uid
from ..plotter.tsx import PROPERTY_TYPES, Prop
from ..tokens import sp


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("layers")
    manual_render.help_button(ctx, "plotter-layers")

    if tab is None:
        widgets.muted("Open or start a map first.")
        return

    doc = tab.doc
    editable = not tab.busy
    width = widgets.grid_width(2)
    if widgets.disabled_button(f"{icons.PLUS} Tiles", editable, (width, 0)):
        doc.add_tile_layer()
    imgui.same_line()
    if widgets.disabled_button(f"{icons.FLAG} Objects", editable, (width, 0)):
        doc.add_object_layer()

    imgui.dummy((0, 4))
    for layer in reversed(doc.layers):
        _row(ctx, doc, state, layer, editable)

    selected = _selected_object(doc, state)
    if selected is not None:
        imgui.dummy((0, 8))
        _object_form(ctx, doc, state, selected, editable)


def _row(ctx: Any, doc: Any, state: Any, layer: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    imgui.push_id(str(layer.uid))
    eye = icons.EYE if layer.visible else icons.EYE_OFF
    # Greyed rather than live-and-discarded. Every one of these drew at full
    # contrast while the tab was saving and then threw the click away, which is
    # the "clickable lie" the house pattern names -- ``clay_tools`` and
    # ``clay_menu`` already wrap theirs. The *selectable* stays live on purpose:
    # choosing which layer you are looking at changes no document and pushes no
    # step, which ``test_choosing_a_layer_pushes_no_step`` pins.
    imgui.begin_disabled(not editable)
    if widgets.small_icon_button(eye, "Show / hide"):
        doc.set_layer_props(layer.uid, visible=not layer.visible)
    imgui.end_disabled()
    imgui.same_line()
    kind = icons.GRID if isinstance(layer, TileLayer) else icons.FLAG
    active = doc.active_layer == layer.uid
    if imgui.selectable(f"{kind} {layer.name or '(unnamed)'}", active)[0]:
        doc.set_active_layer(layer.uid)
        state.selected_object = None
    if imgui.begin_popup_context_item(f"layer-menu-{layer.uid}"):
        imgui.begin_disabled(not editable)
        if imgui.menu_item_simple("Move up"):
            doc.move_layer(layer.uid, doc.index_of(layer.uid) + 1)
        if imgui.menu_item_simple("Move down"):
            doc.move_layer(layer.uid, doc.index_of(layer.uid) - 1)
        imgui.separator()
        if imgui.menu_item_simple("Delete"):
            _delete_layer(ctx, doc, layer)
        imgui.end_disabled()
        imgui.end_popup()
    if active and editable:
        changed, opacity = widgets.labeled_slider_float("Opacity", layer.opacity, 0.0, 1.0)
        if changed:
            doc.set_layer_props(layer.uid, opacity=float(opacity))
        name = widgets.input_text("##name", layer.name, max_length=64)
        if name != layer.name:
            doc.set_layer_props(layer.uid, name=name)
    imgui.pop_id()


def _delete_layer(ctx: Any, doc: Any, layer: Any) -> None:
    from .. import dialogs

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete this layer?",
            message=f"{layer.name or 'The layer'} and everything on it will be removed. "
            "Ctrl+Z brings it back.",
            on_confirm=lambda: doc.remove_layer(layer.uid),
        )
    )


def _selected_object(doc: Any, state: Any) -> tuple[Any, MapObject] | None:
    if state.selected_object is None:
        return None
    for layer in doc.layers:
        if isinstance(layer, ObjectLayer):
            for obj in layer.objects:
                if obj.uid == state.selected_object:
                    return layer, obj
    # The object was deleted or undone away; forgetting it here keeps the form
    # from drawing against a uid nothing answers to.
    state.selected_object = None
    return None


def _object_form(
    ctx: Any, doc: Any, state: Any, found: tuple[Any, MapObject], editable: bool
) -> None:
    from imgui_bundle import imgui

    layer, obj = found
    widgets.section("object")
    if not editable:
        widgets.muted("Saving...")
        return

    name = widgets.input_text("##obj-name", obj.name, max_length=64, hint="name")
    if name != obj.name:
        doc.set_object(layer.uid, obj.uid, name=name)
    obj_class = widgets.input_text("##obj-class", obj.obj_class, max_length=64, hint="class")
    if obj_class != obj.obj_class:
        doc.set_object(layer.uid, obj.uid, obj_class=obj_class)

    widgets.muted(
        f"{obj.kind} at {obj.x:.0f}, {obj.y:.0f}"
        + (f" -- {obj.w:.0f} x {obj.h:.0f}" if obj.kind == "rect" else "")
    )
    changed, visible = widgets.toggle("Visible", obj.visible)
    if changed:
        doc.set_object(layer.uid, obj.uid, visible=visible)

    imgui.dummy((0, 4))
    widgets.section("properties")
    _properties(ctx, doc, layer, obj)

    imgui.dummy((0, 6))
    if widgets.destructive_button(f"{icons.TRASH} Delete object", (-1, 0)):
        doc.remove_object(layer.uid, obj.uid)
        state.selected_object = None


def _properties(ctx: Any, doc: Any, layer: Any, obj: MapObject) -> None:
    """The typed key/value editor.

    Edits go through ``set_object`` with a *whole* replacement dict rather than
    a mutation of the live one, because ``ObjectPropsEdit`` snapshots what it is
    given -- writing into the object's own dict would leave undo restoring the
    value it had already been changed to.
    """
    from imgui_bundle import imgui

    for key in sorted(obj.properties):
        prop = obj.properties[key]
        imgui.push_id(f"prop-{key}")
        widgets.muted(key)
        imgui.same_line()
        value = _value_editor(prop)
        if value is not None and value != prop.value:
            replacement = dict(obj.properties)
            replacement[key] = Prop(type=prop.type, value=value)
            doc.set_object(layer.uid, obj.uid, properties=replacement)
        imgui.same_line()
        if widgets.small_icon_button(icons.X, "Remove"):
            replacement = dict(obj.properties)
            replacement.pop(key, None)
            doc.set_object(layer.uid, obj.uid, properties=replacement)
        imgui.pop_id()

    form_key = f"plotter_prop:{obj.uid}"
    form = ctx.state.preview.setdefault(form_key, {"name": "", "type": "string"})
    form["name"] = widgets.input_text("##prop-name", form["name"], max_length=48, hint="new key")
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    form["type"] = widgets.combo(
        "##prop-type", form["type"], [(t, t) for t in PROPERTY_TYPES], width=sp(90)
    )
    imgui.same_line()
    if widgets.disabled_button(f"{icons.PLUS}##add-prop", bool(form["name"].strip())):
        replacement = dict(obj.properties)
        replacement[form["name"].strip()] = Prop(
            type=form["type"], value=_blank_value(form["type"])
        )
        doc.set_object(layer.uid, obj.uid, properties=replacement)
        form["name"] = ""


def _blank_value(kind: str) -> Any:
    return {"int": 0, "float": 0.0, "bool": False}.get(kind, "")


def _value_editor(prop: Prop) -> Any:
    from imgui_bundle import imgui

    imgui.set_next_item_width(sp(110))
    if prop.type == "bool":
        changed, value = imgui.checkbox("##v", bool(prop.value))
        return value if changed else None
    if prop.type == "int":
        changed, value = imgui.input_int("##v", int(prop.value or 0))
        return value if changed else None
    if prop.type == "float":
        changed, value = imgui.input_float("##v", float(prop.value or 0.0))
        return value if changed else None
    text = widgets.input_text("##v", str(prop.value or ""), max_length=200)
    return text if text != str(prop.value or "") else None


def add_object(
    doc: Any, layer: Any, kind: str, x: float, y: float, w: float, h: float
) -> MapObject:
    """Place a new object. Here rather than in the canvas so the default name
    and the uid mint have one home."""
    count = len(layer.objects) + 1
    return doc.add_object(
        layer.uid,
        MapObject(
            uid=new_uid(),
            name=f"{kind}{count}",
            kind=kind,
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
        ),
    )
