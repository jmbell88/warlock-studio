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


#: Every button in this pane that can grey out does so for this one cause, and
#: hoisting it is the ``_VIEWPORT_WHY`` pattern: four dead controls explaining
#: themselves in four different sentences read as four separate problems.
_BUSY_WHY = "This map is being written; the buttons come back when it lands."

#: How far one level of nesting steps a row in, in design pixels. The same
#: number ``inker_timeline`` uses, because they are the same list drawn twice.
GROUP_INDENT = 8.0

#: What each of the two panes refuses to shrink past, in design pixels. Neither
#: slot declared a floor before, which is how a short window could squeeze the
#: layer list down to its own heading with the stack invisible underneath it.
LAYERS_FLOOR = 140.0
PROPERTIES_FLOOR = 160.0


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Layers")
    manual_render.help_button(ctx, "plotter-layers")

    if tab is None:
        # The heading and nothing else. One voice for one empty state:
        # the canvas's ``nothing_open`` is it, and four panels each
        # repeating it reads as four separate problems.
        return

    doc = tab.doc
    editable = not tab.busy
    width = widgets.grid_width(2)
    if widgets.disabled_button(f"{icons.PLUS} Tiles", editable, (width, 0), reason=_BUSY_WHY):
        doc.add_tile_layer()
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.FLAG} Objects", editable, (width, 0), reason=_BUSY_WHY
    ):
        doc.add_object_layer()
    if widgets.disabled_button(
        f"{icons.FOLDER_OPEN} Group", editable, (width, 0), reason=_BUSY_WHY
    ):
        doc.add_group_layer()
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.IMAGE} Image", editable, (width, 0), reason=_BUSY_WHY
    ):
        doc.add_image_layer()

    # The "Insert" combo that stood here is the object palette's tools now
    # (W3.2): eight kinds behind a dropdown, on a pane that is about layers,
    # when Tiled makes each of them a tool. ``state.object_shape`` is still the
    # stored answer -- ``sync_tool`` writes it from the tool in hand.
    imgui.dummy((0, 4))
    for layer in reversed(doc.layers):
        _row(ctx, doc, state, tab, layer, editable)


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


def _row(
    ctx: Any,
    doc: Any,
    state: Any,
    tab: Any,
    layer: Any,
    editable: bool,
    depth: int = 0,
) -> None:
    """One layer's row, and -- if it folds -- its contents indented under it.

    Groups are authored from this pane and the row menu can move a layer into a
    group or back to the root. The recursive rendering is also what keeps a
    group's contents visible in the one pane that lists what a map is made of.

    A group's eye and padlock are its own flags, and they reach the subtree
    through :func:`~..plotter.scene.resolve` rather than by being written onto
    each child -- so unhiding the group restores exactly what was showing, and
    a child that was hidden on its own stays hidden.

    **One row shape for every kind.** An object layer used to be an imgui tree
    node while everything else was a selectable, so a list of four layers
    carried two different row geometries and the eye sat in a different place
    depending on the kind -- half of why this panel was hard to read down. Both
    fold through the same chevron now, and the row itself is
    ``widgets.list_row``, whose hit target spans the whole width so a click in
    the gap between two glyphs selects the layer instead of doing nothing.
    """
    from imgui_bundle import imgui

    from ..plotter import layer_rows

    active = doc.active_layer == layer.uid
    foldable = layer_rows.can_fold(layer)
    folded = int(layer.uid) in tab.collapsed_rows
    indent = sp(GROUP_INDENT) * depth

    imgui.push_id(str(layer.uid))
    with widgets.list_row(
        f"plotter-layer/{layer.uid}", selected=active, indent=indent
    ) as clicked:
        if clicked:
            # Live while the tab is saving, on purpose: choosing which layer
            # you are looking at changes no document and pushes no step, which
            # ``test_choosing_a_layer_pushes_no_step`` pins.
            doc.set_active_layer(layer.uid)
            state.select_object(None)
        if foldable:
            # **Not gated on ``editable``.** A fold changes nothing about the
            # map, and greying it while a save runs is the panel refusing to
            # scroll -- ``inker_timeline`` makes the same call for its groups.
            chevron = icons.CHEVRON_RIGHT if folded else icons.CHEVRON_DOWN
            if widgets.small_icon_button(
                chevron, "Fold / unfold this row", borderless=True
            ):
                if folded:
                    tab.collapsed_rows.discard(int(layer.uid))
                else:
                    tab.collapsed_rows.add(int(layer.uid))
        else:
            # **``get_text_line_height`` and not ``get_frame_height``**: that
            # is the side ``small_icon_button`` uses, and the wider one put
            # every unfoldable row's eye a few pixels right of every foldable
            # one's -- a ragged column, which is precisely the thing that makes
            # a list hard to scan down.
            imgui.dummy((imgui.get_text_line_height(), 0))
        imgui.same_line()
        eye = icons.EYE if layer.visible else icons.EYE_OFF
        # Greyed rather than live-and-discarded. Every one of these drew at
        # full contrast while the tab was saving and then threw the click away,
        # which is the "clickable lie" the house pattern names.
        imgui.begin_disabled(not editable)
        if widgets.small_icon_button(eye, "Show / hide", borderless=True):
            doc.set_layer_props(layer.uid, visible=not layer.visible)
        imgui.same_line()
        # Beside the eye because they are the same kind of switch: both say
        # what this layer will let you do, neither is about its contents.
        padlock = icons.LOCK if layer.locked else icons.LOCK_OPEN
        if widgets.small_icon_button(
            padlock, "Lock / unlock painting", borderless=True
        ):
            doc.set_layer_props(layer.uid, locked=not layer.locked)
        imgui.end_disabled()
        imgui.same_line()
        _row_label(layer, active)
    _row_menu(ctx, doc, layer, editable)
    _reorder(ctx, doc, layer, editable)
    imgui.pop_id()
    if not folded:
        # After ``pop_id``, so a child's own id scope is the one it pushes --
        # nesting them would make the same layer's controls answer to a
        # different string depending on how deep it happens to sit, and imgui
        # keys open popups and active items on exactly that.
        if getattr(layer, "objects", None):
            _object_rows(ctx, doc, state, layer, editable, depth + 1)
        for child in reversed(getattr(layer, "children", ()) or ()):
            _row(ctx, doc, state, tab, child, editable, depth + 1)


def _row_label(layer: Any, active: bool) -> None:
    """The kind glyph, the name, and the opacity when it is not full.

    **Coloured by state**, which the old row was not: a hidden layer drew
    identically to a visible one, so the eye was the only thing saying so and
    a list of eight layers had to be read glyph by glyph to find the hidden
    one. Muted for hidden, accent for the layer in hand.
    """
    from imgui_bundle import imgui

    from .. import theme

    kind = _KIND_ICONS.get(type(layer), icons.LAYERS)
    name = layer.name or "(unnamed)"
    # The padlock is repeated into the label rather than being only a button,
    # so a locked layer still reads as locked at a glance once the eye has
    # moved on down the list. ``inker_timeline`` does the same.
    locked = f" {icons.LOCK}" if layer.locked else ""
    text = f"{kind} {name}{locked}"
    if not layer.visible:
        widgets.muted(text)
    elif active:
        widgets.text_colored(theme.ACCENT, text)
    else:
        imgui.text(text)

    opacity = float(getattr(layer, "opacity", 1.0))
    if opacity < 0.999:
        # Only when it is not 1.0: a column of "100%" down every row is noise,
        # and the number is worth seeing precisely when it explains why a layer
        # looks wrong. Right-aligned so the names stay a readable column.
        label = f"{opacity * 100:.0f}%"
        room = imgui.get_content_region_avail().x
        width = imgui.calc_text_size(label).x
        if room > width:
            imgui.same_line(imgui.get_cursor_pos().x + room - width, 0.0)
            widgets.muted(label)


def _row_menu(ctx: Any, doc: Any, layer: Any, editable: bool) -> None:
    """The row's right-click menu. Extracted whole; nothing in it changed."""
    from imgui_bundle import imgui

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


def _reorder(ctx: Any, doc: Any, layer: Any, editable: bool) -> None:
    """Drag a layer's row onto another to move it there.

    **The payload is a uid, not an index.** ``inker_timeline`` carries a stack
    index because its stack is flat; this is a tree, addressed by uid
    everywhere -- ``move_layer``, ``index_of`` ("within its own parent's list")
    and every ``LayerMoveEdit`` -- and an index alone does not even say which
    parent it is an index into.

    ``source_allow_null_id`` is **not** optional. The drag source is the row's
    own last item, which imgui may give id 0, and ``BeginDragDropSource``
    asserts outright on a null id; the exception unwinds past ``layout.pane``'s
    ``end_child`` and surfaces as a "Missing PopID()" naming neither the row
    nor the reason. ``inker_timeline`` records the same trap.

    Refused outright while the tab is busy rather than wrapped in
    ``begin_disabled``, which does not reliably stop a drag source.
    """
    from imgui_bundle import imgui

    from ..plotter import layer_rows

    if not editable:
        return
    flags = (
        imgui.DragDropFlags_.source_no_hold_to_open_others.value
        | imgui.DragDropFlags_.source_allow_null_id.value
    )
    if imgui.begin_drag_drop_source(flags):
        imgui.set_drag_drop_payload_py_id("plotter-layer", int(layer.uid))
        widgets.muted(f"{layer.name or '(unnamed)'}")
        imgui.end_drag_drop_source()
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id("plotter-layer")
        if payload is not None:
            source = int(payload.data_id)
            # A group is two possible landings and only the pointer knows
            # which: onto a group's row means *into* it, which is the same
            # answer "Move into" gives from the menu. Anything else reorders
            # beside the row it was dropped on.
            if isinstance(layer, GroupLayer):
                landing = layer_rows.drop_into_group(doc, source, int(layer.uid))
            else:
                landing = layer_rows.drop_target(doc, source, int(layer.uid))
            if landing is not None:
                parent, index = landing
                doc.move_layer(source, index, parent_uid=parent)
        imgui.end_drag_drop_target()


def draw_properties(ctx: Any) -> None:
    """The selected layer's own settings, in a pane of their own.

    **This used to be drawn inside the list, between two sibling rows**, and
    that -- rather than any want of a background -- is what made the layers
    panel hard to read: choosing a layer expanded it into a hundred and fifty
    lines of form, so two adjacent layers were never adjacent on screen and
    the eye had no column of names to run down. Tiled keeps the same fields in
    a Properties panel below the stack, and so does this now.

    Kept in this module rather than given one of its own: they are one
    subject, ``tests/manual/test_coverage.py`` wants a help button per pane
    file, and a second file would need a second exemption argument for no gain.
    """
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Properties")
    manual_render.help_button(ctx, "plotter-properties")
    if tab is None:
        return
    doc = tab.doc
    editable = not tab.busy
    layer = doc.layer(doc.active_layer) if doc.active_layer is not None else None
    if layer is None:
        widgets.muted_wrapped("Choose a layer to see what it carries.")
        return
    if not editable:
        widgets.muted_wrapped(_BUSY_WHY)
        return

    # **What these fields are about**, said on screen rather than inferred from
    # which of them happen to be drawn. The pane shows a layer, one object or a
    # multi-selection, and until the toolbar moved it it sat directly under the
    # layer list, where "the thing above" was answer enough. On the far side of
    # the window from that list it is not: a user reading a Name field has to
    # be told whose name it is.
    widgets.muted(_subject(doc, state, layer))
    if len(state.selected_objects) > 1:
        _group_summary(ctx, doc, state)
        return
    selected = _selected_object(doc, state)
    if selected is not None:
        _object_form(ctx, doc, state, selected, editable)
        return

    changed, opacity = widgets.labeled_slider_float(
        "Opacity",
        layer.opacity,
        0.0,
        1.0,
        help_text="How strongly this layer draws over the ones below it.",
    )
    if changed:
        doc.set_layer_props(layer.uid, opacity=float(opacity))
    # Labels live above their fields throughout the Studio.  Keeping them
    # out of the ImGui ids also gives the narrow inspector the whole row
    # for the value instead of clipping labels into the controls.
    widgets.field_label(
        "Name",
        "What this layer is called in the map, in Tiled, and in the exported "
        ".tmx. Two layers may share a name; the map addresses them by uid.",
    )
    name = widgets.input_text("##layer-name", layer.name, max_length=64)
    if name != layer.name:
        doc.set_layer_props(layer.uid, name=name)
    widgets.field_label(
        "Class",
        "Tiled's per-layer class string. Round-trips through export and import "
        "and means nothing to Plotter itself -- it is for whatever reads the "
        "map afterwards.",
    )
    class_name = widgets.input_text(
        "##layer-class", layer.class_name, max_length=64, hint="Optional class"
    )
    if class_name != layer.class_name:
        doc.set_layer_props(layer.uid, class_name=class_name)
    widgets.field_label(
        "Blend",
        "How this layer composites onto the ones below it. Tiled has no blend "
        "mode, so anything but normal is dropped on .tmx export.",
    )
    blend_mode = widgets.combo(
        "##layer-blend", layer.blend_mode, [(mode, mode) for mode in BLEND_MODES]
    )
    if blend_mode != layer.blend_mode:
        doc.set_layer_props(layer.uid, blend_mode=blend_mode)
    widgets.field_label(
        "Tint",
        "Multiplied into every tile this layer draws. Alpha multiplies the "
        "layer opacity above.",
    )
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
    widgets.field_label(
        "Offset",
        "Shifts the whole layer by this many pixels when drawn, without moving "
        "any tile in the grid.",
    )
    imgui.set_next_item_width(-1)
    changed, offset = controls.input_float2(
        "##layer-offset", [float(layer.offset_x), float(layer.offset_y)]
    )
    if changed:
        doc.set_layer_props(
            layer.uid, offset_x=float(offset[0]), offset_y=float(offset[1])
        )
    widgets.field_label(
        "Parallax",
        "How fast this layer scrolls against the camera. 1 is locked to the "
        "map, 0.5 is half speed -- a distant background.",
    )
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
        widgets.field_label(
            "Draw order",
            "Top-down sorts the objects by their y position each frame, so one "
            "in front overlaps one behind. Manual keeps the order they were "
            "added in.",
        )
        draworder = widgets.combo(
            "##object-layer-draw-order",
            layer.draworder,
            [("topdown", "Top-down"), ("index", "Manual")],
        )
        if draworder != layer.draworder:
            doc.set_layer_props(layer.uid, draworder=draworder)
        widgets.field_label(
            "Outline colour",
            "The colour Tiled draws this layer's object outlines in. It is "
            "editor chrome, not part of the map.",
        )
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
            f"{icons.PLUS} Choose image...##img-{layer.uid}",
            editable,
            (-1, 0),
            reason=_BUSY_WHY,
        ):
            plotter_mode.choose_layer_image(ctx, layer.uid)
        changed_x, repeat_x = widgets.toggle(
            "Repeat X",
            layer.repeat_x,
            tooltip="Tile the picture across the map horizontally.",
        )
        changed_y, repeat_y = widgets.toggle(
            "Repeat Y",
            layer.repeat_y,
            tooltip="Tile the picture down the map vertically.",
        )
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


def _object_rows(
    ctx: Any, doc: Any, state: Any, layer: Any, editable: bool, depth: int = 0
) -> None:
    """One row per object: the same selectable-and-eye shape a layer row has.

    **No per-object lock.** ``MapObject`` has none, adding one is a document
    field plus a ``warlock-dialect`` row in ``COMPAT.md``, and the layer's own
    padlock already gates every object edit -- so the control would be a second
    switch for a state the first one already decides.

    Bottom-up like the canvas draws them: the last object in the list is the
    one on top, so it reads down the page the way it stacks on screen.
    """
    from imgui_bundle import imgui

    indent = sp(GROUP_INDENT) * depth
    for obj in reversed(layer.objects):
        imgui.push_id(str(obj.uid))
        picked = obj.uid in state.selected_objects
        with widgets.list_row(
            f"plotter-object/{obj.uid}", selected=picked, indent=indent
        ) as clicked:
            if clicked:
                doc.set_active_layer(layer.uid)
                state.select_object(obj.uid)
            # No chevron: an object has nothing under it. The gap keeps its
            # eye in the same column as every layer's, so the list still reads
            # as one column of switches rather than two.
            imgui.dummy((imgui.get_text_line_height(), 0))
            imgui.same_line()
            imgui.begin_disabled(not editable)
            if widgets.small_icon_button(
                icons.EYE if obj.visible else icons.EYE_OFF,
                "Show / hide this object",
                borderless=True,
            ):
                doc.set_object(layer.uid, obj.uid, visible=not obj.visible)
            imgui.end_disabled()
            imgui.same_line()
            label = obj.name or f"({obj.kind})"
            if obj.visible:
                imgui.text(label)
            else:
                widgets.muted(label)
        if imgui.begin_popup_context_item(f"obj-menu-{obj.uid}"):
            widgets.popup_chrome(_imgui=imgui)
            imgui.begin_disabled(not editable)
            if controls.menu_item_simple("Raise"):
                doc.reorder_object(layer.uid, obj.uid, 1)
            if controls.menu_item_simple("Lower"):
                doc.reorder_object(layer.uid, obj.uid, -1)
            imgui.separator()
            if controls.menu_item_simple("Delete"):
                doc.remove_object(layer.uid, obj.uid)
                state.selected_objects.discard(obj.uid)
            imgui.end_disabled()
            imgui.end_popup()
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


def _subject(doc: Any, state: Any, layer: Any) -> str:
    """Whose fields these are, in one line. Pure, so it is asserted directly.

    The three cases the pane already branches on, named rather than left to be
    inferred: a layer, one object, or a set of them. The object line carries
    its uid because two objects may share a name and the map addresses them by
    number -- the same reason the layer line does not, since the pane can only
    ever be showing the *active* one.
    """

    count = len(state.selected_objects)
    if count > 1:
        return f"{count} objects"
    found = _selected_object(doc, state)
    if found is not None:
        _layer, obj = found
        name = obj.name or "Object"
        return f"Object: {name} (#{obj.uid})"
    return f"Layer: {layer.name or 'Untitled'}"


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
    # from drawing against a uid nothing answers to. The *whole* set is pruned
    # rather than only the primary, because an undo takes a group away together
    # and half a remembered selection is a group drag over objects that are no
    # longer there.
    live = {
        obj.uid
        for layer in doc.all_layers()
        if isinstance(layer, ObjectLayer)
        for obj in layer.objects
    }
    state.select_objects(state.selected_objects & live)
    return None


def _group_summary(ctx: Any, doc: Any, state: Any) -> None:
    """What the Properties pane says about a multi-selection.

    A count and the two verbs that mean something to a group -- **a summary,
    not a bulk editor**. Editing one field across a set is a real feature with
    a real design question behind it (does an empty field mean "unchanged" or
    "cleared"?), and a form that silently wrote the first object's values onto
    the other four would answer it the worst way. Deselecting down to one
    object is how you edit properties, and the pane says so rather than leaving
    the user to find out.
    """
    from imgui_bundle import imgui

    count = len(state.selected_objects)
    widgets.section(f"{count} objects selected")
    widgets.muted_wrapped(
        "Drag any of them to move the whole set, or press Delete to remove "
        "them. Click one object on its own to edit its properties."
    )
    imgui.dummy((0, 6))
    layer = doc.layer(doc.active_layer) if doc.active_layer is not None else None
    locked = bool(getattr(layer, "locked", False))
    if locked:
        widgets.muted(f"{layer.name} is locked.")
        return
    if widgets.destructive_button(f"{icons.TRASH} Delete {count} objects", (-1, 0)):
        if layer is not None:
            doc.remove_objects(layer.uid, state.selected_objects)
        state.select_object(None)


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
    changed, visible = widgets.toggle(
        "Visible",
        obj.visible,
        tag="obj-visible",
        tooltip="Hides the object without removing it. Tiled carries the flag too.",
    )
    if changed:
        doc.set_object(layer.uid, obj.uid, visible=visible)
    changed, opacity = widgets.labeled_slider_float(
        "Opacity",
        obj.opacity,
        0.0,
        1.0,
        help_text="This one object's opacity, multiplied into its layer's.",
    )
    if changed:
        doc.set_object(layer.uid, obj.uid, opacity=float(opacity))
    changed, rotation = controls.input_float(
        "Rotation",
        float(obj.rotation),
        tooltip="Degrees clockwise about the object's own origin.",
    )
    if changed:
        doc.set_object(layer.uid, obj.uid, rotation=float(rotation))
    _shape_fields(doc, layer, obj)

    imgui.dummy((0, 4))
    widgets.section("Properties")
    _properties(ctx, doc, layer, obj)

    imgui.dummy((0, 6))
    if widgets.destructive_button(f"{icons.TRASH} Delete object", (-1, 0)):
        doc.remove_object(layer.uid, obj.uid)
        state.select_object(None)


def _shape_fields(doc: Any, layer: Any, obj: MapObject) -> None:
    """Fields carried by tile and text geometry, replacing the frozen shape."""
    shape = obj.shape
    if isinstance(shape, TileShape):
        tile_id, flip_h, flip_v, flip_d = gidlib.decompose(shape.gid)
        changed, value = controls.input_int(
            "Tile gid",
            tile_id,
            1,
            tooltip=(
                "Which tile this object draws, as a global id across every "
                "tileset in the map. The three flip flags are stored in the "
                "high bits of the same number."
            ),
        )
        changed_h, next_h = widgets.toggle(
            "Flip horizontal", flip_h, tooltip="Mirror the tile across."
        )
        changed_v, next_v = widgets.toggle(
            "Flip vertical", flip_v, tooltip="Mirror the tile down."
        )
        changed_d, next_d = widgets.toggle(
            "Flip diagonal",
            flip_d,
            tooltip="Transpose the tile. With the other two it makes the quarter turns.",
        )
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
    if widgets.disabled_button(
        f"{icons.PLUS}##add-prop",
        bool(form["name"].strip()),
        reason="Give the property a name first.",
    ):
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
