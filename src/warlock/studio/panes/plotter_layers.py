"""Plotter's right-top pane: the layer stack, and an object's properties.

**This pane is the sole owner of object metadata.** Objects are placed on the
canvas and named, classed and given typed properties here -- one control, one
owner, so there is never a second place a property can be set from.

The list is drawn top-first, which is the opposite of the document's order.
Every layered editor does it that way (the topmost layer is at the top of the
list) and a user arriving from one would read a bottom-first list as inverted.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import controls, icons, plotter_mode, widgets
from ..manual import render as manual_render
from ..plotter.props import CONTAINER_TYPES, PROPERTY_TYPES, Prop
from ..plotter.tilemap import (
    BLEND_MODES,
    Capsule,
    Ellipse,
    GroupLayer,
    ImageLayer,
    MapObject,
    ObjectLayer,
    Point,
    Polygon,
    Polyline,
    Rect,
    Text,
    TileLayer,
    TileShape,
    new_uid,
)
from ..tilegrid import gid as gidlib
from ..tokens import sp

#: The glyph each layer kind is listed under. A dict rather than a chain of
#: ``isinstance``, so a fifth kind is one line and cannot be forgotten in the
#: middle of a draw function.
_KIND_ICONS = {
    TileLayer: icons.GRID,
    ObjectLayer: icons.FLAG,
    GroupLayer: icons.FOLDER_OPEN,
    ImageLayer: icons.IMAGE,
}


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Layers")
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
    if widgets.disabled_button(f"{icons.FOLDER_OPEN} Group", editable, (width, 0)):
        doc.add_group_layer()
    imgui.same_line()
    if widgets.disabled_button(f"{icons.IMAGE} Image", editable, (width, 0)):
        doc.add_image_layer()

    if isinstance(doc.active(), ObjectLayer):
        state.object_shape = widgets.labeled_combo(
            "Insert",
            state.object_shape,
            [
                (kind, kind.title())
                for kind in (
                    "rect",
                    "point",
                    "ellipse",
                    "capsule",
                    "polygon",
                    "polyline",
                    "tile",
                    "text",
                )
            ],
        )

    imgui.dummy((0, 4))
    for layer in reversed(doc.layers):
        _row(ctx, doc, state, layer, editable)

    selected = _selected_object(doc, state)
    if selected is not None:
        imgui.dummy((0, 8))
        _object_form(ctx, doc, state, selected, editable)


def _subtree_uids(layer: Any) -> set[int]:
    """Every uid at or under ``layer``.

    The pane's half of ``_map_layers._contains``, written here rather than
    imported because that one is a private helper of the document's own mixin
    and this needs the whole *set* once, not one membership test per candidate.
    The two must agree about what "underneath" means, which they do for the
    only reason they can: both walk ``children`` and nothing else.
    """
    out = {int(layer.uid)}
    for child in getattr(layer, "children", ()) or ():
        out |= _subtree_uids(child)
    return out


def _row(ctx: Any, doc: Any, state: Any, layer: Any, editable: bool) -> None:
    """One layer's row, and -- if it is a group -- its children indented under it.

    Groups are authored from this pane and the row menu can move a layer into a
    group or back to the root. The recursive rendering is also what keeps a
    group's contents visible in the one pane that lists what a map is made of.

    A group's eye and padlock are its own flags, and they reach the subtree
    through :func:`~..plotter.scene.resolve` rather than by being written onto
    each child -- so unhiding the group restores exactly what was showing, and
    a child that was hidden on its own stays hidden.
    """
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
    imgui.same_line()
    # Beside the eye because they are the same kind of switch: both say what
    # this layer will let you do, neither is about its contents. Greyed with the
    # eye while saving, for the clickable-lie reason above.
    padlock = icons.LOCK if layer.locked else icons.LOCK_OPEN
    if widgets.small_icon_button(padlock, "Lock / unlock painting"):
        doc.set_layer_props(layer.uid, locked=not layer.locked)
    imgui.end_disabled()
    imgui.same_line()
    kind = _KIND_ICONS.get(type(layer), icons.LAYERS)
    active = doc.active_layer == layer.uid
    if controls.selectable(f"{kind} {layer.name or '(unnamed)'}", active)[0]:
        doc.set_active_layer(layer.uid)
        state.selected_object = None
    if imgui.begin_popup_context_item(f"layer-menu-{layer.uid}"):
        widgets.popup_chrome(_imgui=imgui)
        imgui.begin_disabled(not editable)
        if controls.menu_item_simple("Move up"):
            doc.move_layer(layer.uid, doc.index_of(layer.uid) + 1)
        if controls.menu_item_simple("Move down"):
            doc.move_layer(layer.uid, doc.index_of(layer.uid) - 1)
        # Every group that is not this layer *and not underneath it*. The
        # filter used to be ``group.uid != layer.uid``, which is only the first
        # half of what ``move_layer`` refuses: a group moved into one of its own
        # descendants is the same cycle, and it raises
        # ``ValueError("a group cannot be moved inside itself")``. Nothing wraps
        # a pane draw, so choosing that item took the window down -- and the
        # rule the rest of this file follows is that a menu never offers an
        # action the document will refuse.
        blocked = _subtree_uids(layer)
        groups = [
            entry
            for entry in doc.all_layers()
            if isinstance(entry, GroupLayer) and entry.uid not in blocked
        ]
        if groups and imgui.begin_menu("Move into"):
            if controls.menu_item_simple("Map root"):
                doc.move_layer(layer.uid, len(doc.layers), parent_uid=None)
            for group in groups:
                # Scoped by uid, not by name: two groups called "Terrain" are
                # ordinary, and sharing one imgui id makes the second one
                # unclickable.
                imgui.push_id(str(group.uid))
                if controls.menu_item_simple(group.name or "Group"):
                    doc.move_layer(layer.uid, len(group.children), parent_uid=group.uid)
                imgui.pop_id()
            imgui.end_menu()
        imgui.separator()
        if controls.menu_item_simple("Duplicate"):
            doc.duplicate_layer(layer.uid)
        if controls.menu_item_simple("Merge down"):
            try:
                doc.merge_down(layer.uid)
            except ValueError as exc:
                # Framed rather than forwarded: the engine's sentence says what
                # was wrong and nothing about what was being attempted.
                ctx.toast(f"Not merged: {exc}.", "error")
        if controls.menu_item_simple("Delete"):
            _delete_layer(ctx, doc, layer)
        imgui.end_disabled()
        imgui.end_popup()
    if active and editable:
        changed, opacity = widgets.labeled_slider_float("Opacity", layer.opacity, 0.0, 1.0)
        if changed:
            doc.set_layer_props(layer.uid, opacity=float(opacity))
        # Labels live above their fields throughout the Studio.  Keeping them
        # out of the ImGui ids also gives the narrow inspector the whole row
        # for the value instead of clipping labels into the controls.
        widgets.field_label("Name")
        name = widgets.input_text("##layer-name", layer.name, max_length=64)
        if name != layer.name:
            doc.set_layer_props(layer.uid, name=name)
        widgets.field_label("Class")
        class_name = widgets.input_text(
            "##layer-class", layer.class_name, max_length=64, hint="Optional class"
        )
        if class_name != layer.class_name:
            doc.set_layer_props(layer.uid, class_name=class_name)
        widgets.field_label("Blend")
        blend_mode = widgets.combo(
            "##layer-blend", layer.blend_mode, [(mode, mode) for mode in BLEND_MODES]
        )
        if blend_mode != layer.blend_mode:
            doc.set_layer_props(layer.uid, blend_mode=blend_mode)
        widgets.field_label("Tint")
        imgui.set_next_item_width(-1)
        changed, tint = controls.color_edit4(
            "##layer-tint", [channel / 255.0 for channel in layer.tint]
        )
        if changed:
            doc.set_layer_props(
                layer.uid,
                tint=tuple(
                    max(0, min(255, int(round(float(channel) * 255))))
                    for channel in tint
                ),
            )
        widgets.field_label("Offset")
        imgui.set_next_item_width(-1)
        changed, offset = controls.input_float2(
            "##layer-offset", [float(layer.offset_x), float(layer.offset_y)]
        )
        if changed:
            doc.set_layer_props(
                layer.uid, offset_x=float(offset[0]), offset_y=float(offset[1])
            )
        widgets.field_label("Parallax")
        imgui.set_next_item_width(-1)
        changed, parallax = controls.input_float2(
            "##layer-parallax", [float(layer.parallax_x), float(layer.parallax_y)]
        )
        if changed:
            doc.set_layer_props(
                layer.uid,
                parallax_x=float(parallax[0]),
                parallax_y=float(parallax[1]),
            )
        if isinstance(layer, ObjectLayer):
            widgets.field_label("Draw order")
            draworder = widgets.combo(
                "##object-layer-draw-order",
                layer.draworder,
                [("topdown", "Top-down"), ("index", "Manual")],
            )
            if draworder != layer.draworder:
                doc.set_layer_props(layer.uid, draworder=draworder)
            widgets.field_label("Outline colour")
            color = widgets.input_text(
                "##object-layer-color",
                layer.color or "",
                max_length=9,
                hint="#RRGGBB",
            )
            if color != (layer.color or ""):
                doc.set_layer_props(layer.uid, color=color or None)
        elif isinstance(layer, ImageLayer):
            if layer.pixels is None:
                widgets.muted_wrapped(
                    "This image layer has no picture yet."
                )
            if widgets.disabled_button(
                f"{icons.PLUS} Choose image...##img-{layer.uid}", editable, (-1, 0)
            ):
                plotter_mode.choose_layer_image(ctx, layer.uid)
            changed_x, repeat_x = widgets.toggle("Repeat X", layer.repeat_x)
            changed_y, repeat_y = widgets.toggle("Repeat Y", layer.repeat_y)
            if changed_x or changed_y:
                doc.set_layer_props(
                    layer.uid,
                    repeat_x=repeat_x if changed_x else layer.repeat_x,
                    repeat_y=repeat_y if changed_y else layer.repeat_y,
                )
        # Collapsed by default: most maps carry none, and an always-open form
        # for an empty mapping is a row of controls that explain nothing. The
        # model has supported these since the format did; only the way in was
        # missing. Rides ``LayerPropsEdit``, so it undoes with the rest.
        if widgets.header("Properties", default_open=False, persist_key="plotter/layer-props"):
            imgui.begin_disabled(layer.locked)
            property_editor(
                ctx,
                f"plotter_layer_prop:{layer.uid}",
                layer.properties,
                lambda values: doc.set_layer_props(layer.uid, properties=values),
                object_options=object_options(doc),
            )
            imgui.end_disabled()
    imgui.pop_id()
    if isinstance(layer, GroupLayer):
        # After ``pop_id``, so a child's own id scope is the one it pushes --
        # nesting them would make the same layer's controls answer to a
        # different string depending on how deep it happens to sit, and imgui
        # keys open popups and active items on exactly that.
        imgui.indent(sp(12))
        for child in reversed(layer.children):
            _row(ctx, doc, state, child, editable)
        imgui.unindent(sp(12))


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
    # ``all_layers`` rather than ``doc.layers``: an object layer inside a group
    # is still an object layer, and a selection that could not be found again
    # would be silently forgotten one frame after it was made.
    for layer in doc.all_layers():
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
    widgets.section("Object")
    if not editable:
        widgets.muted("Saving...")
        return
    if layer.locked:
        # Read-only rather than hidden: the form is how you *look* at an
        # object's properties, and a lock is not a reason to stop seeing them.
        # Said in words as well as greyed, because a pane full of dead controls
        # with no explanation reads as broken.
        widgets.muted(f"{layer.name} is locked.")
        imgui.begin_disabled(True)
        _object_fields(ctx, doc, state, layer, obj)
        imgui.end_disabled()
        return
    _object_fields(ctx, doc, state, layer, obj)


def _object_fields(ctx: Any, doc: Any, state: Any, layer: Any, obj: MapObject) -> None:
    from imgui_bundle import imgui

    name = widgets.input_text("##obj-name", obj.name, max_length=64, hint="name")
    if name != obj.name:
        doc.set_object(layer.uid, obj.uid, name=name)
    obj_class = widgets.input_text("##obj-class", obj.obj_class, max_length=64, hint="class")
    if obj_class != obj.obj_class:
        doc.set_object(layer.uid, obj.uid, obj_class=obj_class)

    widgets.muted(
        f"{obj.kind} at {obj.x:.0f}, {obj.y:.0f}"
        + (f" -- {obj.w:.0f} x {obj.h:.0f}" if hasattr(obj.shape, "w") else "")
    )
    changed, visible = widgets.toggle("Visible", obj.visible)
    if changed:
        doc.set_object(layer.uid, obj.uid, visible=visible)
    changed, opacity = widgets.labeled_slider_float("Opacity", obj.opacity, 0.0, 1.0)
    if changed:
        doc.set_object(layer.uid, obj.uid, opacity=float(opacity))
    changed, rotation = controls.input_float("Rotation", float(obj.rotation))
    if changed:
        doc.set_object(layer.uid, obj.uid, rotation=float(rotation))
    _shape_fields(doc, layer, obj)

    imgui.dummy((0, 4))
    widgets.section("Properties")
    _properties(ctx, doc, layer, obj)

    imgui.dummy((0, 6))
    if widgets.destructive_button(f"{icons.TRASH} Delete object", (-1, 0)):
        doc.remove_object(layer.uid, obj.uid)
        state.selected_object = None


def _shape_fields(doc: Any, layer: Any, obj: MapObject) -> None:
    """Fields carried by tile and text geometry, replacing the frozen shape."""
    shape = obj.shape
    if isinstance(shape, TileShape):
        tile_id, flip_h, flip_v, flip_d = gidlib.decompose(shape.gid)
        changed, value = controls.input_int("Tile gid", tile_id, 1)
        changed_h, next_h = widgets.toggle("Flip horizontal", flip_h)
        changed_v, next_v = widgets.toggle("Flip vertical", flip_v)
        changed_d, next_d = widgets.toggle("Flip diagonal", flip_d)
        if changed or changed_h or changed_v or changed_d:
            try:
                encoded = gidlib.compose(
                    max(0, int(value)) if changed else tile_id,
                    flip_h=next_h if changed_h else flip_h,
                    flip_v=next_v if changed_v else flip_v,
                    flip_d=next_d if changed_d else flip_d,
                )
            except ValueError:
                return
            doc.set_object(layer.uid, obj.uid, shape=replace(shape, gid=encoded))
        return
    if not isinstance(shape, Text):
        return

    text = widgets.input_text("##text-value", shape.text, max_length=2048, hint="text")
    family = widgets.input_text(
        "##text-family", shape.family, max_length=128, hint="font family"
    )
    changed_size, pixel_size = controls.input_int("Pixel size", shape.pixel_size, 1)
    color = widgets.input_text("##text-color", shape.color, max_length=9, hint="#RRGGBB")
    halign = widgets.labeled_combo(
        "Horizontal",
        shape.halign,
        [(value, value.title()) for value in ("left", "center", "right", "justify")],
    )
    valign = widgets.labeled_combo(
        "Vertical",
        shape.valign,
        [(value, value.title()) for value in ("top", "center", "bottom")],
    )
    changed_wrap, wrap = widgets.toggle("Wrap", shape.wrap)
    changed_bold, bold = widgets.toggle("Bold", shape.bold)
    changed_italic, italic = widgets.toggle("Italic", shape.italic)
    changed_underline, underline = widgets.toggle("Underline", shape.underline)
    changed_strike, strikeout = widgets.toggle("Strikeout", shape.strikeout)
    changed_kerning, kerning = widgets.toggle("Kerning", shape.kerning)
    values = {
        "text": text,
        "family": family,
        "pixel_size": max(1, int(pixel_size)) if changed_size else shape.pixel_size,
        "color": color,
        "halign": halign,
        "valign": valign,
        "wrap": wrap if changed_wrap else shape.wrap,
        "bold": bold if changed_bold else shape.bold,
        "italic": italic if changed_italic else shape.italic,
        "underline": underline if changed_underline else shape.underline,
        "strikeout": strikeout if changed_strike else shape.strikeout,
        "kerning": kerning if changed_kerning else shape.kerning,
    }
    if any(getattr(shape, key) != value for key, value in values.items()):
        doc.set_object(layer.uid, obj.uid, shape=replace(shape, **values))


def object_options(doc: Any) -> list[tuple[str, str]]:
    """Persistent object ids as combo entries, with zero as Tiled's ``none``."""
    entries = [("0", "None")]
    for layer in doc.all_layers():
        if not isinstance(layer, ObjectLayer):
            continue
        for obj in layer.objects:
            if obj.id:
                label = obj.name or f"{obj.kind} {obj.id}"
                entries.append((str(obj.id), f"{label} ({obj.id})"))
    return entries


def property_editor(
    ctx: Any,
    form_key: str,
    props: dict[str, Prop],
    on_change: Any,
    *,
    object_options: list[tuple[str, str]] | None = None,
) -> None:
    """The typed key/value editor, for anything that carries custom properties.

    ``on_change`` is handed a **whole replacement dict** and never a mutation of
    the live one, which is the rule that makes the three callers correct rather
    than a convention they each have to remember: the props edits snapshot what
    they are given, so writing into the owner's own dict would leave undo
    restoring the value it had already been changed to.

    ``form_key`` scopes the half-typed new-key row in ``ctx.state.preview``, so
    two editors drawn in one frame -- a layer's and the map's -- do not share
    the name being typed into one of them.

    It lives here rather than in ``widgets`` because this pane owns it and the
    other two reach for it; a cross-pane import is house-normal, and moving it
    would put a Tiled-shaped control in the generic widget set.
    """
    from imgui_bundle import imgui

    for key in sorted(props):
        prop = props[key]
        imgui.push_id(f"prop-{key}")
        widgets.muted(key)
        imgui.same_line()
        value = _value_editor(prop, object_options=object_options)
        if value is not None and value != prop.value:
            replacement = dict(props)
            # ``propertytype`` travels with the value: it names a class or an
            # enum declared in a Tiled project, and editing a value is not a
            # reason to forget which type the user's value belongs to.
            replacement[key] = Prop(
                type=prop.type, value=value, propertytype=prop.propertytype
            )
            on_change(replacement)
        imgui.same_line()
        if widgets.small_icon_button(icons.X, "Remove"):
            replacement = dict(props)
            replacement.pop(key, None)
            on_change(replacement)

        # The unfolded editors stay *inside* ``prop-{key}``. The pop used to be
        # here, which put three separate collisions one line apart:
        # ``##property-class`` was shared by every class property at the same
        # level, and the recursive calls below re-entered this function outside
        # any scope at all -- so a nested editor's ``##prop-name``,
        # ``##prop-type`` and ``##add-prop`` were the *same* imgui ids as its
        # parent's, and typing a new key into the inner row drove the outer one.
        # ``form_key`` scopes the half-typed state in ``state.preview``, which
        # is a different question from what imgui thinks the widget is.
        if prop.type == "class" and widgets.header(
            f"Edit {key}", default_open=False, persist_key=f"{form_key}/{key}/class"
        ):
            class_name = widgets.input_text(
                "##property-class", prop.propertytype, max_length=64, hint="class name"
            )
            if class_name != prop.propertytype:
                replacement = dict(props)
                replacement[key] = Prop("class", prop.value, propertytype=class_name)
                on_change(replacement)
            property_editor(
                ctx,
                f"{form_key}:{key}",
                prop.value,
                lambda values, key=key, prop=prop: on_change(
                    {
                        **props,
                        key: Prop("class", values, propertytype=prop.propertytype),
                    }
                ),
                object_options=object_options,
            )
        elif prop.type == "list" and widgets.header(
            f"Edit {key}", default_open=False, persist_key=f"{form_key}/{key}/list"
        ):
            _list_editor(
                ctx,
                f"{form_key}:{key}",
                prop.value,
                lambda values, key=key, prop=prop: on_change(
                    {
                        **props,
                        key: Prop("list", values, propertytype=prop.propertytype),
                    }
                ),
                object_options=object_options,
            )
        imgui.pop_id()

    form = ctx.state.preview.setdefault(form_key, {"name": "", "type": "string"})
    form["name"] = widgets.input_text("##prop-name", form["name"], max_length=48, hint="new key")
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    form["type"] = widgets.combo(
        "##prop-type", form["type"], [(t, t) for t in AUTHORABLE_TYPES], width=sp(90)
    )
    imgui.same_line()
    if widgets.disabled_button(f"{icons.PLUS}##add-prop", bool(form["name"].strip())):
        replacement = dict(props)
        replacement[form["name"].strip()] = Prop(
            type=form["type"], value=_blank_value(form["type"])
        )
        on_change(replacement)
        form["name"] = ""


def _properties(ctx: Any, doc: Any, layer: Any, obj: MapObject) -> None:
    property_editor(
        ctx,
        f"plotter_prop:{obj.uid}",
        obj.properties,
        lambda values: doc.set_object(layer.uid, obj.uid, properties=values),
        object_options=object_options(doc),
    )


#: What the new-property row offers. Container values start empty and unfold
#: into the same editor recursively after they are added.
AUTHORABLE_TYPES = PROPERTY_TYPES


def _blank_value(kind: str) -> Any:
    return {
        "int": 0,
        "float": 0.0,
        "bool": False,
        "object": 0,
        "class": {},
        "list": [],
    }.get(kind, "")


def _summary(prop: Prop) -> str:
    """A container property on one line, for the read-only row.

    Counts rather than contents: the point is that the property is *there* and
    carries something, which is what stops a class arriving from Tiled looking
    like an empty string until the recursive editor lands.
    """
    count = len(prop.value)
    if prop.type == "class":
        name = prop.propertytype or "class"
        return f"{name} ({count} member{'' if count == 1 else 's'})"
    return f"list ({count} item{'' if count == 1 else 's'})"


def _value_editor(
    prop: Prop, *, object_options: list[tuple[str, str]] | None = None
) -> Any:
    from imgui_bundle import imgui

    imgui.set_next_item_width(sp(110))
    if prop.type == "bool":
        changed, value = controls.checkbox("##v", bool(prop.value))
        return value if changed else None
    if prop.type == "object" and object_options:
        value = widgets.combo("##v", str(int(prop.value or 0)), object_options)
        return int(value) if value != str(int(prop.value or 0)) else None
    if prop.type in ("int", "object"):
        changed, value = controls.input_int("##v", int(prop.value or 0))
        return value if changed else None
    if prop.type == "float":
        changed, value = controls.input_float("##v", float(prop.value or 0.0))
        return value if changed else None
    if prop.type in CONTAINER_TYPES:
        # Read-only, and shown rather than hidden: editing a class member or a
        # list item is the recursive editor's job, and returning ``None`` here
        # is what says "nothing changed" to the caller.
        widgets.muted(_summary(prop))
        return None
    # ``file`` lands here with ``string`` and ``color``: the path is text this
    # editor never resolves, so a text row is the whole control.
    text = widgets.input_text("##v", str(prop.value or ""), max_length=200)
    return text if text != str(prop.value or "") else None


def _list_editor(
    ctx: Any,
    form_key: str,
    items: list[Prop],
    on_change: Any,
    *,
    object_options: list[tuple[str, str]] | None = None,
) -> None:
    """A typed list property, recursively and without mutating its live list."""
    from imgui_bundle import imgui

    for index, prop in enumerate(items):
        imgui.push_id(f"item-{index}")
        widgets.muted(f"[{index}] {prop.type}")
        imgui.same_line()
        value = _value_editor(prop, object_options=object_options)
        if value is not None and value != prop.value:
            replacement = list(items)
            replacement[index] = Prop(prop.type, value, propertytype=prop.propertytype)
            on_change(replacement)
        imgui.same_line()
        if widgets.small_icon_button(icons.X, "Remove item"):
            replacement = list(items)
            replacement.pop(index)
            on_change(replacement)

        # Inside ``item-{index}`` for ``property_editor``'s reason, and this
        # one has the extra sting: a nested list's ``##item-type`` combo and
        # its bare PLUS button were the parent list's ids exactly, so adding an
        # item to an inner list added it to the outer one.
        if prop.type == "class" and widgets.header(
            f"Edit item {index}",
            default_open=False,
            persist_key=f"{form_key}/{index}/class",
        ):
            property_editor(
                ctx,
                f"{form_key}:{index}",
                prop.value,
                lambda values, index=index, prop=prop: on_change(
                    [
                        *items[:index],
                        Prop("class", values, propertytype=prop.propertytype),
                        *items[index + 1 :],
                    ]
                ),
                object_options=object_options,
            )
        elif prop.type == "list" and widgets.header(
            f"Edit item {index}",
            default_open=False,
            persist_key=f"{form_key}/{index}/list",
        ):
            _list_editor(
                ctx,
                f"{form_key}:{index}",
                prop.value,
                lambda values, index=index, prop=prop: on_change(
                    [
                        *items[:index],
                        Prop("list", values, propertytype=prop.propertytype),
                        *items[index + 1 :],
                    ]
                ),
                object_options=object_options,
            )
        imgui.pop_id()

    form = ctx.state.preview.setdefault(f"{form_key}:new-item", {"type": "string"})
    form["type"] = widgets.combo(
        "##item-type",
        form["type"],
        [(kind, kind) for kind in AUTHORABLE_TYPES],
        width=sp(110),
    )
    imgui.same_line()
    if widgets.small_icon_button(icons.PLUS, "Add item"):
        on_change([*items, Prop(form["type"], _blank_value(form["type"]))])


def add_object(
    doc: Any,
    layer: Any,
    kind: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    gid: int = 0,
) -> MapObject:
    """Place a new object. Here rather than in the canvas so the default name
    and the uid mint have one home."""
    count = len(layer.objects) + 1
    width, height = float(w), float(h)
    shape = {
        "rect": lambda: Rect(width, height),
        "point": Point,
        "ellipse": lambda: Ellipse(width, height),
        "capsule": lambda: Capsule(width, height),
        "polygon": lambda: Polygon(
            ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height))
        ),
        "polyline": lambda: Polyline(((0.0, 0.0), (width, height))),
        "tile": lambda: TileShape(gid=gid, w=width, h=height),
        "text": lambda: Text("Text", width, height),
    }[kind]()
    return doc.add_object(
        layer.uid,
        MapObject(
            uid=new_uid(),
            name=f"{kind}{count}",
            shape=shape,
            x=float(x),
            y=float(y),
        ),
    )
