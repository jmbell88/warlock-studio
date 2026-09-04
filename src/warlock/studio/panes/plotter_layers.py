"""Plotter's right-top pane: the layer stack, and an object's properties.

**This pane is the sole owner of object metadata.** Objects are placed on the
canvas and named, classed and given typed properties here -- one control, one
owner, so there is never a second place a property can be set from.

The list is drawn top-first, which is the opposite of the document's order.
Every layered editor does it that way (the topmost layer is at the top of the
list) and a user arriving from one would read a bottom-first list as inverted.
"""

from __future__ import annotations

from contextlib import contextmanager
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


#: The kinds a layer can be added as, in the order the ``+`` menu offers them.
#: Tiled's own order and Tiled's own four. A table because the menu and the
#: Layer menu both name them, and two lists would come to hold three each.
ADD_KINDS: tuple[tuple[str, str, str], ...] = (
    ("tile", "Tile layer", "add_tile_layer"),
    ("object", "Object layer", "add_object_layer"),
    ("group", "Group", "add_group_layer"),
    ("image", "Image layer", "add_image_layer"),
)

ADD_POPUP = "plotter-layer-add"


#: How wide the name column is, in design pixels. Fixed rather than stretched:
#: a two-column table whose *name* column stretches puts the values at a
#: different x on every form, and a column of values that does not line up is
#: the thing a table is for.
NAME_COLUMN = 104.0


@contextmanager
def _table(table_id: str):
    """Tiled's property table: a name column and a value column, ruled.

    A ``begin_table`` rather than the label-above-field stack this app uses
    everywhere else, and the difference is not decoration. A stack spends two
    rows on every field -- a label line and a full-width control -- so a layer's
    fourteen fields are twenty-eight rows of a 300 px pane, and the *names* are
    a column the eye cannot run down because each one is followed by a control
    the width of the panel. The stack is right for a form somebody fills in
    once; this is a form somebody reads.

    Yields whether the table opened, which is imgui's contract and must be
    honoured: ``end_table`` may only be called when ``begin_table`` returned
    true, and an unbalanced pair takes the frame down.
    """
    from imgui_bundle import imgui

    flags = (
        imgui.TableFlags_.borders_inner_v.value
        | imgui.TableFlags_.row_bg.value
        | imgui.TableFlags_.sizing_stretch_prop.value
    )
    opened = bool(imgui.begin_table(table_id, 2, flags))
    if opened:
        imgui.table_setup_column(
            "name", imgui.TableColumnFlags_.width_fixed.value, sp(NAME_COLUMN)
        )
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_stretch.value)
    try:
        yield opened
    finally:
        if opened:
            imgui.end_table()


def _row_named(name: str, help_text: str = "", *, indent: float = 0.0) -> None:
    """Open a row, write the name into the left cell, and leave the cursor in
    the right one with the item width already set to fill it.

    The help text is a **tooltip on the name** rather than a marker beside it.
    ``help_marker`` after a full-width control is drawn on a line of its own,
    where it reads as the next field's -- the failure ``labeled_combo`` exists
    to avoid -- and in a table there is no room for it at all. The name is the
    thing a reader hovers when they do not know what a field is.
    """
    from imgui_bundle import imgui

    imgui.table_next_row()
    imgui.table_next_column()
    if indent:
        imgui.indent(indent)
    imgui.align_text_to_frame_padding()
    widgets.muted(name)
    if help_text and imgui.is_item_hovered():
        imgui.set_tooltip(help_text)
    if indent:
        imgui.unindent(indent)
    imgui.table_next_column()
    imgui.set_next_item_width(-1)


def draw(ctx: Any) -> None:
    """The layer stack: the active layer's opacity, the list, and the bar.

    **Tiled's three parts, in Tiled's order.** The four *Add* buttons used to
    be a 2x2 grid across the top of the pane -- 68 px of the panel spent on the
    thing done once per map, above the list that is read on every click -- and
    the opacity of the layer in hand was a slider in the Properties pane on the
    other side of the window. Both are wrong the same way round: the frequent
    control was below the rare one.

    So the opacity leads, the list takes everything that is left, and the verbs
    are a footer bar under it. The bar is the ``settings_2d`` footer pattern --
    ``begin_child`` at ``(0, -bar_h)`` so the list scrolls and the bar stays
    put, which is what stops a map with thirty layers pushing Delete off the
    bottom of the pane.
    """
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
    layer = doc.layer(doc.active_layer) if doc.active_layer is not None else None
    _opacity_row(doc, layer, editable)

    # The "Insert" combo that stood here is the object palette's tools now
    # (W3.2): eight kinds behind a dropdown, on a pane that is about layers,
    # when Tiled makes each of them a tool. ``state.object_shape`` is still the
    # stored answer -- ``sync_tool`` writes it from the tool in hand.
    bar_h = imgui.get_frame_height() + imgui.get_style().item_spacing.y
    if imgui.begin_child("##plotter-layer-list", (0.0, -bar_h)):
        for entry in reversed(doc.layers):
            _row(ctx, doc, state, tab, entry, editable)
    imgui.end_child()
    _layer_bar(ctx, state, doc, layer, editable)


def _opacity_row(doc: Any, layer: Any, editable: bool) -> None:
    """The active layer's opacity, over the list rather than across the window.

    Tiled's placement, and the argument is the gesture: opacity is reached for
    *while looking at the stack* -- "which of these is washing the others out"
    -- and the Properties pane is now on the far side of the map. It stays in
    Properties too, because that pane is the full account of what a layer
    carries and a field missing from it would be a field with no home.
    """

    if layer is None:
        widgets.muted("No layer selected")
        return
    changed, opacity = widgets.labeled_slider_float(
        "Opacity",
        float(getattr(layer, "opacity", 1.0)),
        0.0,
        1.0,
        help_text="How strongly this layer draws over the ones below it.",
    )
    if changed and editable:
        doc.set_layer_props(layer.uid, opacity=float(opacity))


def _layer_bar(ctx: Any, state: Any, doc: Any, layer: Any, editable: bool) -> None:
    """Add, duplicate, raise, lower, lock, delete -- the strip under the list.

    Glyphs through ``widgets.small_icon_button`` would be invisible to
    ``probe``; these are ``controls.button`` and ``widgets.disabled_button``,
    which is the same rule the toolbar's tool pill follows and for the same
    reason -- a bar button that stopped working could not otherwise be caught
    by a test that presses it.

    Every button that can refuse says why. "A map keeps at least one layer" is
    Delete's, and it is the sentence the Layer menu already used: one refusal,
    one wording, wherever it is met.
    """
    from imgui_bundle import imgui

    uid = None if layer is None else layer.uid
    many = len(doc.layers) > 1
    if controls.button(f"{icons.PLUS}##plotter-layer-add", tooltip="Add a layer"):
        imgui.open_popup(ADD_POPUP)
    _add_popup(ctx, doc, editable)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.COPY}##plotter-layer-dup",
        editable and uid is not None,
        reason=_BUSY_WHY if not editable else "Select a layer first.",
        tooltip="Duplicate this layer",
    ):
        doc.duplicate_layer(uid)
    imgui.same_line()
    for glyph, delta, verb in (
        (icons.ARROW_UP, 1, "Raise"),
        (icons.ARROW_DOWN, -1, "Lower"),
    ):
        can = editable and plotter_mode.can_shift_layer(doc, uid, delta)
        if widgets.disabled_button(
            f"{glyph}##plotter-layer-{verb.lower()}",
            can,
            reason=_BUSY_WHY
            if not editable
            else "This layer is already at the end of its group.",
            tooltip=f"{verb} this layer",
        ):
            plotter_mode.shift_layer(doc, uid, delta)
        imgui.same_line()
    locked = bool(getattr(layer, "locked", False))
    if widgets.disabled_button(
        f"{icons.LOCK if locked else icons.LOCK_OPEN}##plotter-layer-lock",
        editable and uid is not None,
        reason=_BUSY_WHY if not editable else "Select a layer first.",
        tooltip="Lock / unlock painting on this layer",
    ):
        doc.set_layer_props(uid, locked=not locked)
    imgui.same_line()
    # ``controls.button`` rather than ``widgets.disabled_button`` for this one
    # alone: it is the destructive verb on the bar and wants the role paint,
    # which ``disabled_button`` has no argument for. Same chokepoint either way,
    # so ``probe`` sees it.
    if controls.button(
        f"{icons.TRASH}##plotter-layer-del",
        role=controls.ButtonRole.DESTRUCTIVE,
        enabled=editable and uid is not None and many,
        reason=_BUSY_WHY
        if not editable
        else (
            "A map keeps at least one layer."
            if uid is not None
            else "Select a layer first."
        ),
        tooltip="Delete this layer",
    ):
        _delete_layer(ctx, doc, layer)
    imgui.same_line()
    # The one switch on this bar that is about the *canvas* rather than about a
    # layer, and it earns its place beside them: "which of these am I painting
    # on" is the question the list answers and this is the answer drawn on the
    # map. Also a row in both View surfaces -- see ``plotter_tools``.
    if controls.button(
        f"{icons.EYE}##plotter-layer-highlight",
        selected=bool(state.highlight),
        tooltip="Dim every layer but this one (H)",
    ):
        state.highlight = not state.highlight


def _add_popup(ctx: Any, doc: Any, editable: bool) -> None:
    """The four kinds, behind the ``+``. A menu rather than four buttons: they
    are one decision with four answers, and four glyphs would have been four
    pictures to learn for a thing done once per map."""

    with controls.menu_popup(ADD_POPUP) as opened:
        if not opened:
            return
        for key, label, method in ADD_KINDS:
            hit = controls.menu_item(
                f"{label}##plotter-add/{key}",
                "",
                False,
                editable,
                reason=_BUSY_WHY,
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                getattr(doc, method)()


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
        if state.renaming_layer == int(layer.uid) and editable:
            _rename_field(ctx, doc, state, layer)
        else:
            _row_label(layer, active)
            # **Double-click renames**, which is what every layered editor
            # does and what this list did not: the only way to change a name
            # was to select the layer and cross the window to the Properties
            # pane -- and after wave A that pane is on the *other side* of the
            # map. Read after the label rather than after the row's hit
            # target, so a double-click in the empty gap to the right of a
            # short name still selects rather than opening a field the pointer
            # is nowhere near.
            if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                state.renaming_layer = int(layer.uid)
    _row_menu(ctx, doc, layer, editable)
    _reorder(ctx, doc, layer, editable)
    imgui.pop_id()
    if not folded:
        # After ``pop_id``, so a child's own id scope is the one it pushes --
        # nesting them would make the same layer's controls answer to a
        # different string depending on how deep it happens to sit, and imgui
        # keys open popups and active items on exactly that.
        #
        # **Layers only.** An object layer's contents used to be indented here,
        # which put the stack sixty rows down the pane on a map with sixty
        # triggers. They are ``plotter_objects`` now -- a list you search rather
        # than a tree you unfold.
        for child in reversed(getattr(layer, "children", ()) or ()):
            _row(ctx, doc, state, tab, child, editable, depth + 1)


def _rename_field(ctx: Any, doc: Any, state: Any, layer: Any) -> None:
    """The name, in place, while it is being typed.

    Committed on **deactivation** rather than per keystroke, which is the one
    thing that separates this from the Properties pane's field: that one is a
    read-modify-write on every frame the caret is in it, which is right for a
    form and wrong here -- a rename pushes an undo step, and typing "Ground"
    would push six.

    Escape leaves the name alone; Enter and a click elsewhere both commit,
    because both are how a person says they are finished. The field takes
    keyboard focus on the frame it appears, or the user would have to click the
    thing they just double-clicked.

    The half-typed value lives in ``ctx.state.preview``, keyed by uid, which is
    the pattern the resize and offset forms already use: it is a *draft*, so it
    must not be on the document, and it must not be on the pane either or a
    scroll that culls the row would lose it.
    """
    from imgui_bundle import imgui

    key = f"plotter_rename:{layer.uid}"
    typed = ctx.state.preview.get(key)
    if typed is None:
        typed = layer.name
        ctx.state.preview[key] = typed
        imgui.set_keyboard_focus_here()
    imgui.set_next_item_width(-1)
    ctx.state.preview[key] = widgets.input_text(
        f"##plotter-rename-{layer.uid}", str(typed), max_length=64
    )
    if imgui.is_key_pressed(imgui.Key.escape):
        state.renaming_layer = 0
        ctx.state.preview.pop(key, None)
        return
    if imgui.is_item_deactivated():
        wanted = str(ctx.state.preview.pop(key, layer.name))
        state.renaming_layer = 0
        if wanted != layer.name:
            doc.set_layer_props(layer.uid, name=wanted)


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
        # **The map's own fields**, rather than a sentence telling the user to
        # go and select something. There is always a map when this pane draws,
        # so there is always something true to show -- and the map's size,
        # projection and custom properties are exactly what a Properties pane
        # with nothing selected is asked about. The sentence it replaces was a
        # pane that went blank the moment you clicked off a layer.
        widgets.muted("Map")
        map_rows(ctx, doc, editable)
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

    _layer_table(ctx, doc, layer, editable)
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



def map_rows(ctx: Any, doc: Any, editable: bool) -> None:
    """The map's own facts and its custom properties, as name/value rows.

    Two surfaces read this: the Properties pane with no layer selected, and
    Map > Map properties, which is the dialog that owns the *editable*
    metadata. What is drawn here is deliberately the read-only half plus the
    property table -- the size, the projection and the tile size are set in the
    Resize dialog, and a second set of fields for them is a second place they
    can be typed differently.
    """
    from imgui_bundle import imgui

    with _table("##plotter-map-table") as opened:
        if not opened:
            return
        _row_named("Size", "How many cells the map is, and how big a cell is.")
        imgui.align_text_to_frame_padding()
        widgets.muted(
            f"{doc.width} x {doc.height} tiles, {doc.tile_w} x {doc.tile_h} px"
        )
        _row_named("Pixels", "The whole map, in pixels.")
        imgui.align_text_to_frame_padding()
        widgets.muted(f"{doc.pixel_width} x {doc.pixel_height}")
        _row_named("Projection", "The lattice. Fixed once anything is painted.")
        imgui.align_text_to_frame_padding()
        widgets.muted(str(doc.projection))
        _row_named("Class", "Tiled's per-map class string.")
        imgui.align_text_to_frame_padding()
        widgets.muted(doc.class_name or "-")

    imgui.dummy((0, 4))
    widgets.section("Custom properties")
    imgui.begin_disabled(not editable)
    property_editor(
        ctx,
        f"plotter_map_prop:{id(doc)}",
        doc.properties,
        doc.set_map_properties,
        object_options=object_options(doc),
    )
    imgui.end_disabled()


def _layer_table(ctx: Any, doc: Any, layer: Any, editable: bool) -> None:
    """Everything a layer carries, as name/value rows.

    Fourteen fields were twenty-eight rows before this: a label line and a
    full-width control apiece, in a 300 px pane. Every value starts at the same
    x now, which is what makes the column readable and is why Tiled draws it as
    a table.

    The kind-specific rows sit inside the same table rather than after it, so an
    object layer's draw order and an image layer's repeat flags line up with the
    name and the opacity above them instead of restarting the form.
    """
    from imgui_bundle import imgui

    with _table("##plotter-layer-table") as opened:
        if not opened:
            return
        _row_named(
            "Name",
            "What this layer is called in the map, in Tiled, and in the "
            "exported .tmx. Two layers may share a name; the map addresses "
            "them by uid.",
        )
        name = widgets.input_text("##layer-name", layer.name, max_length=64)
        if name != layer.name:
            doc.set_layer_props(layer.uid, name=name)

        _row_named(
            "Class",
            "Tiled's per-layer class string. Round-trips through export and "
            "import and means nothing to Plotter itself -- it is for whatever "
            "reads the map afterwards.",
        )
        class_name = widgets.input_text(
            "##layer-class", layer.class_name, max_length=64, hint="Optional class"
        )
        if class_name != layer.class_name:
            doc.set_layer_props(layer.uid, class_name=class_name)

        _row_named("Opacity", "How strongly this layer draws over the ones below it.")
        changed, opacity = controls.slider_float(
            "##layer-opacity", float(layer.opacity), 0.0, 1.0
        )
        if changed:
            doc.set_layer_props(layer.uid, opacity=float(opacity))

        _row_named(
            "Blend",
            "How this layer composites onto the ones below it. Tiled has no "
            "blend mode, so anything but normal is dropped on .tmx export.",
        )
        blend_mode = widgets.combo(
            "##layer-blend", layer.blend_mode, [(mode, mode) for mode in BLEND_MODES]
        )
        if blend_mode != layer.blend_mode:
            doc.set_layer_props(layer.uid, blend_mode=blend_mode)

        _row_named(
            "Tint",
            "Multiplied into every tile this layer draws. Alpha multiplies the "
            "layer opacity above.",
        )
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

        _row_named(
            "Offset",
            "Shifts the whole layer by this many pixels when drawn, without "
            "moving any tile in the grid.",
        )
        changed, offset = controls.input_float2(
            "##layer-offset", [float(layer.offset_x), float(layer.offset_y)]
        )
        if changed:
            doc.set_layer_props(
                layer.uid, offset_x=float(offset[0]), offset_y=float(offset[1])
            )

        _row_named(
            "Parallax",
            "How fast this layer scrolls against the camera. 1 is locked to "
            "the map, 0.5 is half speed -- a distant background.",
        )
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
            _row_named(
                "Draw order",
                "Top-down sorts the objects by their y position each frame, so "
                "one in front overlaps one behind. Manual keeps the order they "
                "were added in.",
            )
            draworder = widgets.combo(
                "##object-layer-draw-order",
                layer.draworder,
                [("topdown", "Top-down"), ("index", "Manual")],
            )
            if draworder != layer.draworder:
                doc.set_layer_props(layer.uid, draworder=draworder)

            _row_named(
                "Outline",
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
            _row_named(
                "Picture",
                "The image this layer draws. It is stored in the .wmap rather "
                "than referenced, so moving the file does not break the map.",
            )
            if widgets.disabled_button(
                f"{icons.PLUS} Choose...##img-{layer.uid}",
                editable,
                (-1, 0),
                reason=_BUSY_WHY,
            ):
                plotter_mode.choose_layer_image(ctx, layer.uid)
            if layer.pixels is None:
                _row_named("", "")
                widgets.muted("No picture yet")

            _row_named("Repeat", "Tile the picture across and down the map.")
            changed_x, repeat_x = controls.checkbox(
                "X##img-repeat-x",
                layer.repeat_x,
                tooltip="Tile the picture across the map horizontally.",
            )
            imgui.same_line()
            changed_y, repeat_y = controls.checkbox(
                "Y##img-repeat-y",
                layer.repeat_y,
                tooltip="Tile the picture down the map vertically.",
            )
            if changed_x or changed_y:
                doc.set_layer_props(
                    layer.uid,
                    repeat_x=repeat_x if changed_x else layer.repeat_x,
                    repeat_y=repeat_y if changed_y else layer.repeat_y,
                )


def _delete_layer(ctx: Any, doc: Any, layer: Any) -> None:
    """No confirm. **Undo is the confirmation** (J91), as everywhere else.

    This asked one, and its own message gave the reason not to: "Ctrl+Z brings
    it back". It was the only undoable in-document delete in the app that
    stopped to ask -- Inker, Clay, Packwright and Sirens all remove a layer, an
    object, a page or a pattern outright -- and it asked with ``Confirm``'s
    *unsaved-work* labels on top of that, so a reader was offered [Discard] and
    [Keep editing] for a step that discards nothing and ends no edit. The two
    places that really are irreversible go on asking through ``ask_delete``.
    """
    doc.remove_layer(layer.uid)
    # See ``plotter_menu``'s Delete row: the object selection outlives the
    # layer that held it unless it is pruned. Resolved off the state rather
    # than passed in, because this helper takes the *document*.
    tab = plotter_mode.ensure(ctx).active
    if tab is not None:
        plotter_mode._prune_object_selection(ctx, tab)


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
    """One object's own fields, in the same table the layer form uses.

    The kind and the position used to be a muted sentence -- "rect at 128, 64
    -- 64 x 32" -- which is a *reading* rather than a field: the numbers were on
    screen and could not be typed. They are rows now, and the shape's own
    geometry with them, so an object can be placed exactly rather than dragged
    until the sentence says the right thing.
    """
    from imgui_bundle import imgui

    with _table("##plotter-object-table") as opened:
        if not opened:
            return
        _row_named(
            "Name",
            "What this object is called. Two objects may share a name; the "
            "map addresses them by id.",
        )
        name = widgets.input_text("##obj-name", obj.name, max_length=64, hint="name")
        if name != obj.name:
            doc.set_object(layer.uid, obj.uid, name=name)

        _row_named(
            "Class",
            "Tiled's per-object class string. Round-trips through export and "
            "means nothing to Plotter itself.",
        )
        obj_class = widgets.input_text(
            "##obj-class", obj.obj_class, max_length=64, hint="class"
        )
        if obj_class != obj.obj_class:
            doc.set_object(layer.uid, obj.uid, obj_class=obj_class)

        _row_named("Kind", "What shape this object is. Set by the tool that drew it.")
        imgui.align_text_to_frame_padding()
        widgets.muted(str(obj.kind))

        _row_named(
            "Position",
            "Where the object sits, in map pixels from the top-left corner.",
        )
        changed, position = controls.input_float2(
            "##obj-position", [float(obj.x), float(obj.y)]
        )
        if changed:
            doc.set_object(
                layer.uid, obj.uid, x=float(position[0]), y=float(position[1])
            )

        if hasattr(obj.shape, "w"):
            _row_named("Size", "The object's width and height, in map pixels.")
            changed, size = controls.input_float2(
                "##obj-size", [float(obj.w), float(obj.h)]
            )
            if changed:
                doc.set_object(
                    layer.uid,
                    obj.uid,
                    shape=replace(
                        obj.shape, w=float(size[0]), h=float(size[1])
                    ),
                )

        _row_named(
            "Rotation", "Degrees clockwise about the object's own origin."
        )
        changed, rotation = controls.input_float("##obj-rotation", float(obj.rotation))
        if changed:
            doc.set_object(layer.uid, obj.uid, rotation=float(rotation))

        _row_named(
            "Visible",
            "Hides the object without removing it. Tiled carries the flag too.",
        )
        changed, visible = controls.checkbox("##obj-visible", bool(obj.visible))
        if changed:
            doc.set_object(layer.uid, obj.uid, visible=visible)

        _row_named(
            "Opacity", "This one object's opacity, multiplied into its layer's."
        )
        changed, opacity = controls.slider_float(
            "##obj-opacity", float(obj.opacity), 0.0, 1.0
        )
        if changed:
            doc.set_object(layer.uid, obj.uid, opacity=float(opacity))

        _shape_fields(doc, layer, obj)

    imgui.dummy((0, 4))
    widgets.section("Properties")
    _properties(ctx, doc, layer, obj)

    imgui.dummy((0, 6))
    if widgets.destructive_button(f"{icons.TRASH} Delete object", (-1, 0)):
        doc.remove_object(layer.uid, obj.uid)
        state.select_object(None)


#: The text shape's boolean run, as one table. Six toggles that were six rows
#: are one row of six checkboxes, which is what they are: a set of independent
#: switches on one font, not six separate questions.
TEXT_FLAGS: tuple[tuple[str, str], ...] = (
    ("bold", "B"),
    ("italic", "I"),
    ("underline", "U"),
    ("strikeout", "S"),
    ("wrap", "Wrap"),
    ("kerning", "Kern"),
)


def _shape_fields(doc: Any, layer: Any, obj: MapObject) -> None:
    """Fields carried by tile and text geometry, replacing the frozen shape.

    Drawn **inside the caller's table**, which is why it opens none of its own:
    a tile object's gid belongs in the same column of values as its position,
    and a second table under the first would restart the alignment three rows
    from the end of the form.
    """
    from imgui_bundle import imgui

    shape = obj.shape
    if isinstance(shape, TileShape):
        tile_id, flip_h, flip_v, flip_d = gidlib.decompose(shape.gid)
        _row_named(
            "Tile",
            "Which tile this object draws, as a global id across every tileset "
            "in the map. The three flip flags are stored in the high bits of "
            "the same number.",
        )
        changed, value = controls.input_int("##obj-gid", tile_id, 1)
        _row_named(
            "Flip",
            "Across, down, and transposed. The three together make the quarter "
            "turns.",
        )
        changed_h, next_h = controls.checkbox(
            "H##obj-flip-h", flip_h, tooltip="Mirror the tile across."
        )
        imgui.same_line()
        changed_v, next_v = controls.checkbox(
            "V##obj-flip-v", flip_v, tooltip="Mirror the tile down."
        )
        imgui.same_line()
        changed_d, next_d = controls.checkbox(
            "D##obj-flip-d",
            flip_d,
            tooltip="Transpose the tile. With the other two it makes the "
            "quarter turns.",
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

    _row_named("Text", "What the label reads.")
    text = widgets.input_text("##text-value", shape.text, max_length=2048, hint="text")
    _row_named("Font", "The family name. Tiled resolves it; Plotter stores it.")
    family = widgets.input_text(
        "##text-family", shape.family, max_length=128, hint="font family"
    )
    _row_named("Size", "The font's pixel size.")
    changed_size, pixel_size = controls.input_int("##text-size", shape.pixel_size, 1)
    _row_named("Colour", "The colour the text is drawn in.")
    color = widgets.input_text("##text-color", shape.color, max_length=9, hint="#RRGGBB")
    _row_named("Align", "Horizontal, then vertical, within the object's box.")
    halign = widgets.combo(
        "##text-halign",
        shape.halign,
        [(value, value.title()) for value in ("left", "center", "right", "justify")],
    )
    _row_named("", "")
    valign = widgets.combo(
        "##text-valign",
        shape.valign,
        [(value, value.title()) for value in ("top", "center", "bottom")],
    )
    _row_named("Style", "Bold, italic, underline, strikeout, wrapping, kerning.")
    flags: dict[str, bool] = {}
    for index, (key, label) in enumerate(TEXT_FLAGS):
        if index:
            imgui.same_line()
        changed_flag, value_flag = controls.checkbox(
            f"{label}##text-{key}", bool(getattr(shape, key))
        )
        flags[key] = value_flag if changed_flag else getattr(shape, key)

    values = {
        "text": text,
        "family": family,
        "pixel_size": max(1, int(pixel_size)) if changed_size else shape.pixel_size,
        "color": color,
        "halign": halign,
        "valign": valign,
        **flags,
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


def _prop_form(ctx: Any, form_key: str) -> dict:
    """The half-typed new-key row and the selected row, scoped to one editor.

    In ``ctx.state.preview`` rather than on the pane, which is the pattern the
    resize and offset forms already follow: a draft must not be on the document,
    and it must not be on the pane either or scrolling the row out of view would
    lose it. ``form_key`` is what stops two editors drawn in one frame -- a
    layer's and the map's -- sharing the name being typed into one of them.
    """

    form = ctx.state.preview.setdefault(
        form_key, {"name": "", "type": "string", "selected": "", "folded": set()}
    )
    # Older sessions parked a two-key dict here; fill the rest in rather than
    # replacing it, so a name half-typed across the upgrade survives.
    form.setdefault("selected", "")
    form.setdefault("folded", set())
    return form


def _prop_row_head(
    name: str,
    *,
    depth: int = 0,
    foldable: bool = False,
    folded: bool = False,
    selected: bool = False,
) -> tuple[bool, bool]:
    """Open a property row and draw its name cell. -> (picked, folded toggled).

    The name is a **selectable** rather than muted text, which is what makes the
    ``-`` in the footer possible: a remove button per row was a third control on
    every line of a two-column table, and in a 300 px pane that is the column of
    values gone. One selection and one button says the same thing in the space
    of a name.
    """
    from imgui_bundle import imgui

    imgui.table_next_row()
    imgui.table_next_column()
    indent = sp(GROUP_INDENT) * depth
    if indent:
        imgui.indent(indent)
    toggled = False
    if foldable:
        glyph = icons.CHEVRON_RIGHT if folded else icons.CHEVRON_DOWN
        toggled = widgets.small_icon_button(
            glyph, "Fold / unfold this property", borderless=True
        )
    else:
        # The same width the chevron takes, so every name in the column starts
        # at the same x whether or not it folds. ``get_text_line_height`` is the
        # side ``small_icon_button`` uses -- ``get_frame_height`` is wider and
        # would ripple the column.
        imgui.dummy((imgui.get_text_line_height(), 0))
    imgui.same_line()
    picked = bool(controls.selectable(f"{name}##prop-name", selected)[0])
    if indent:
        imgui.unindent(indent)
    imgui.table_next_column()
    imgui.set_next_item_width(-1)
    return picked, toggled


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

    Drawn as Tiled's name/value table since wave C, with the ``+`` and ``-``
    under it rather than a remove button on every row. It lives here rather than
    in ``widgets`` because this pane owns it and the other two reach for it; a
    cross-pane import is house-normal, and moving it would put a Tiled-shaped
    control in the generic widget set.
    """
    from imgui_bundle import imgui

    form = _prop_form(ctx, form_key)
    with _table(f"##{form_key}/table") as opened:
        if opened:
            _prop_rows(
                ctx, form_key, form, props, on_change, object_options=object_options
            )
    _prop_footer(ctx, form, props, on_change)
    imgui.dummy((0, 2))


def _prop_rows(
    ctx: Any,
    form_key: str,
    form: dict,
    props: dict[str, Prop],
    on_change: Any,
    *,
    object_options: list[tuple[str, str]] | None = None,
    depth: int = 0,
    path: str = "",
) -> None:
    """One row per property, with a container's members indented under it.

    Recursion stays **inside one table**, so a class member's value lines up
    with its parent's rather than restarting the column. The imgui id scope is
    pushed per row and popped after the members, which is what stops a nested
    editor's ``##prop-name`` being the same id as its parent's -- the collision
    that used to make typing into an inner row drive the outer one.
    """
    from imgui_bundle import imgui

    folded = form["folded"]
    for key in sorted(props):
        prop = props[key]
        here = f"{path}/{key}"
        imgui.push_id(f"prop-{here}")
        container = prop.type in CONTAINER_TYPES
        is_folded = here in folded
        picked, toggled = _prop_row_head(
            key,
            depth=depth,
            foldable=container,
            folded=is_folded,
            selected=form["selected"] == here,
        )
        if picked:
            form["selected"] = here
        if toggled:
            folded.discard(here) if is_folded else folded.add(here)
        value = _value_editor(prop, object_options=object_options, ctx=ctx)
        if value is not None and value != prop.value:
            replacement = dict(props)
            # ``propertytype`` travels with the value: it names a class or an
            # enum declared in a Tiled project, and editing a value is not a
            # reason to forget which type the user's value belongs to.
            replacement[key] = Prop(
                type=prop.type, value=value, propertytype=prop.propertytype
            )
            on_change(replacement)
        if container and not is_folded:
            _prop_children(
                ctx,
                form_key,
                form,
                prop,
                key,
                props,
                on_change,
                object_options=object_options,
                depth=depth,
                path=here,
            )
        imgui.pop_id()


def _prop_children(
    ctx: Any,
    form_key: str,
    form: dict,
    prop: Prop,
    key: str,
    props: dict[str, Prop],
    on_change: Any,
    *,
    object_options: list[tuple[str, str]] | None = None,
    depth: int,
    path: str,
) -> None:
    """A class's members or a list's items, one level in."""

    def replace_value(value: Any) -> None:
        replacement = dict(props)
        replacement[key] = Prop(
            type=prop.type, value=value, propertytype=prop.propertytype
        )
        on_change(replacement)

    if prop.type == "class":
        _row_named("class", "The Tiled class this member set belongs to.",
                   indent=sp(GROUP_INDENT) * (depth + 1))
        class_name = widgets.input_text(
            "##property-class", prop.propertytype, max_length=64, hint="class name"
        )
        if class_name != prop.propertytype:
            replacement = dict(props)
            replacement[key] = Prop(
                type=prop.type, value=prop.value, propertytype=class_name
            )
            on_change(replacement)
        _prop_rows(
            ctx,
            form_key,
            form,
            dict(prop.value),
            replace_value,
            object_options=object_options,
            depth=depth + 1,
            path=path,
        )
        return
    for index, item in enumerate(list(prop.value)):
        _row_named(
            f"[{index}] {item.type}",
            "",
            indent=sp(GROUP_INDENT) * (depth + 1),
        )
        value = _value_editor(item, object_options=object_options, ctx=ctx)
        if value is not None and value != item.value:
            items = list(prop.value)
            items[index] = Prop(
                type=item.type, value=value, propertytype=item.propertytype
            )
            replace_value(items)


def _prop_footer(ctx: Any, form: dict, props: dict[str, Prop], on_change: Any) -> None:
    """The new-key row and the remove button, under the table.

    ``-`` acts on the *selected* row rather than carrying its own per-row
    button, and says so when nothing is selected. That is what buys the table
    its value column back in a 300 px pane.

    Nested selections are deliberately not removable from here: the path names a
    member of a class or an item of a list, and removing one is a different
    edit on a different container. The button refuses by name rather than
    silently doing nothing.
    """
    from imgui_bundle import imgui

    selected = str(form.get("selected") or "")
    top_level = selected.startswith("/") and selected.count("/") == 1
    name = selected[1:] if top_level else ""

    form["name"] = widgets.input_text(
        "##prop-name", form["name"], max_length=48, hint="new key"
    )
    imgui.same_line()
    imgui.set_next_item_width(sp(80))
    form["type"] = widgets.combo(
        "##prop-type", form["type"], [(t, t) for t in AUTHORABLE_TYPES], width=sp(80)
    )
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.PLUS}##add-prop",
        bool(form["name"].strip()),
        reason="Give the property a name first.",
        tooltip="Add a property with this name and type",
    ):
        replacement = dict(props)
        replacement[form["name"].strip()] = Prop(
            type=form["type"], value=_blank_value(form["type"])
        )
        on_change(replacement)
        form["name"] = ""
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.MINUS}##remove-prop",
        bool(name) and name in props,
        reason=(
            "Select a property to remove it."
            if not selected
            else "Only a top-level property can be removed here."
        ),
        tooltip=f"Remove {name}" if name else "Remove the selected property",
    ):
        replacement = dict(props)
        replacement.pop(name, None)
        form["selected"] = ""
        on_change(replacement)


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



def _goto_arrow(ctx: Any, value: Any) -> None:
    """The arrow beside an object reference. Drawn only where it can act.

    ``ctx`` is optional because ``_value_editor`` is also reached from the
    tileset editor's own property table, which has no Plotter tab behind it.
    Without one the arrow is not drawn at all, which is right: an arrow that
    cannot jump is worse than no arrow.

    The tab is resolved here rather than threaded through three call sites,
    because there is only ever one answer -- every editor that draws an object
    reference is drawing it against the map that is open.
    """

    if ctx is None:
        return
    from . import plotter_objects

    tab = plotter_mode.ensure(ctx).active
    if tab is None:
        return
    plotter_objects.go_to_arrow(ctx, tab, int(value or 0))


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
    prop: Prop,
    *,
    object_options: list[tuple[str, str]] | None = None,
    ctx: Any = None,
) -> Any:
    from imgui_bundle import imgui

    if prop.type == "bool":
        changed, value = controls.checkbox("##v", bool(prop.value))
        return value if changed else None
    if prop.type == "object" and object_options:
        # Narrowed to leave room for the arrow beside it. ``-1`` would take the
        # whole cell and ``same_line`` past the content edge draws a control
        # nowhere, which is the trap ``same_line_or_wrap`` exists for.
        imgui.set_next_item_width(-sp(26))
        value = widgets.combo("##v", str(int(prop.value or 0)), object_options)
        _goto_arrow(ctx, prop.value)
        return int(value) if value != str(int(prop.value or 0)) else None
    if prop.type == "object":
        imgui.set_next_item_width(-sp(26))
        changed, value = controls.input_int("##v", int(prop.value or 0))
        _goto_arrow(ctx, prop.value)
        return value if changed else None
    if prop.type == "int":
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
