"""Plotter's Objects dock: everything on the map's object layers, as a list.

Objects were rows *inside* the layer list, indented under the layer that held
them. That is Tiled's arrangement for layers and not for objects, and the cost
is exactly what a folded list costs: a map with four object layers holding sixty
triggers between them put the layer stack sixty rows further down the pane, and
the only way to find one object was to know which layer it was on and unfold
that layer.

So the layer list lists **layers**, and this lists objects: name, class and id,
grouped by the layer they sit on, with a search box over the lot. Tiled's own
split, and the reason for it is the question each answers -- "what is this map
made of" is a stack, "where is the door I named" is a list you search.

**A row acts on the object wherever it is.** Clicking one activates its layer
and selects it, because a selection on a layer no tool can reach is a selection
that cannot be edited; double-clicking centres the view on it. The right-click
verbs are ``plotter_canvas.object_menu_rows``, shared rather than copied -- a
second copy of that menu is how the canvas came to label Duplicate ``Ctrl+D``
for a binding that is ``Ctrl+J``.

The pane appears only when the map has an object layer (``skeletons.plotter``'s
``when``), because on a tile-only map it is a heading over nothing.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, plotter_mode, widgets
from ..manual import render as manual_render
from ..plotter.tilemap import ObjectLayer
from ..tokens import sp

#: What this pane refuses to shrink past, in design pixels: a heading, the
#: search box and three rows. Below that it is a scrollbar with a name in it.
OBJECTS_FLOOR = 120.0

#: The search box's key on ``AppState.list_filters``.
FILTER_TAG = "plotter-objects"

MENU = "plotter-objects-menu"


def has_object_layer(ctx: Any) -> bool:
    """Whether this map has anywhere for an object to be. The slot's ``when``.

    Cheap enough per frame -- it is a walk over the layer tree, which is tens of
    entries -- and derived rather than cached for the reason every other derived
    answer in this package is: a cached flag is one more thing that can disagree
    with the document, and this one would go stale the moment a layer was added.
    """

    state = getattr(ctx.state, "plotter", None)
    tab = None if state is None else state.active
    if tab is None:
        return False
    return any(isinstance(layer, ObjectLayer) for layer in tab.doc.all_layers())


def rows(doc: Any) -> list[tuple[Any, list[Any]]]:
    """``(layer, objects)`` for every object layer on the map, top-first.

    Pure, so what the list *contains* is a plain assertion rather than a
    rendered frame. Top-first because that is how the layer stack reads, and a
    dock that ordered its groups the other way round would put the same two
    layers in opposite orders on one screen.

    Through ``all_layers`` and so through groups: a group is not a place objects
    stop existing, and on any map organised into folders that is where they are.
    """

    out = []
    for layer in doc.all_layers():
        if isinstance(layer, ObjectLayer):
            # Bottom-up like the canvas draws them: the last object in the list
            # is the one on top, so the dock reads down the page the way the
            # map stacks on screen.
            out.append((layer, list(reversed(layer.objects))))
    out.reverse()
    return out


def matches(obj: Any, needle: str) -> bool:
    """Whether one object answers the search box.

    Name, class **and id**, because all three are how a person refers to an
    object: the name is what they called it, the class is what it is, and the id
    is what an object-typed property in another object holds -- so pasting a
    number from a property field into the box is a way of asking "what is this a
    reference to".
    """

    if not needle:
        return True
    return needle in (
        f"{obj.name} {obj.obj_class} {obj.id} {obj.kind}".lower()
    )


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Objects")
    manual_render.help_button(ctx, "plotter-objects")
    if tab is None:
        # The heading and nothing else. One voice for one empty state: the
        # canvas's ``nothing_open`` is it.
        return

    doc = tab.doc
    groups = rows(doc)
    total = sum(len(objects) for _layer, objects in groups)
    needle = widgets.list_filter(ctx, FILTER_TAG, total)
    if not total:
        widgets.muted_wrapped(
            "No objects yet. Pick an object layer and one of the insert tools "
            "on the toolbar, then drag on the map."
        )
        return

    shown = 0
    for layer, objects in groups:
        visible = [obj for obj in objects if matches(obj, needle)]
        if not visible:
            continue
        # The layer's own row is a heading rather than a selectable: it is here
        # to say which set the objects under it belong to, and a click on it
        # would be a second, quieter way of choosing a layer -- which the layer
        # list on the same screen already does.
        imgui.push_id(str(layer.uid))
        widgets.muted(f"{icons.FLAG} {layer.name or 'Objects'}")
        for obj in visible:
            shown += 1
            _row(ctx, state, tab, layer, obj)
        imgui.pop_id()
    widgets.no_matches(needle, shown)
    _menu(ctx, state, tab)


def _row(ctx: Any, state: Any, tab: Any, layer: Any, obj: Any) -> None:
    """One object: its name, its class and its id.

    Three columns' worth of information on one row rather than an imgui table,
    because the id is short and the class is usually empty -- a table would
    spend a fixed column on a field most maps do not use, in a 300 px pane. The
    id is right-aligned, which is what makes it a column without one.
    """
    from imgui_bundle import imgui

    picked = obj.uid in state.selected_objects
    with widgets.list_row(f"plotter-object/{obj.uid}", selected=picked) as clicked:
        if clicked:
            # Shift extends, which is the same gesture the canvas marquee makes
            # and the outliner's rule too.
            if imgui.get_io().key_shift:
                state.select_objects(state.selected_objects | {obj.uid})
                state.selected_object = obj.uid
            else:
                state.select_object(obj.uid)
            # The layer too: a selection on a layer no tool can reach is a
            # selection that cannot be edited.
            tab.doc.set_active_layer(layer.uid)
        label = obj.name or f"({obj.kind})"
        if picked:
            from .. import theme

            widgets.text_colored(theme.ACCENT, label)
        else:
            imgui.text(label)
        if obj.obj_class:
            imgui.same_line()
            widgets.muted(obj.obj_class)
        _trailing_id(obj)
    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
        plotter_mode.go_to_object(ctx, tab, layer.uid, obj.uid)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
        state.select_object(obj.uid)
        tab.doc.set_active_layer(layer.uid)
        imgui.open_popup(MENU)


def _trailing_id(obj: Any) -> None:
    """The persistent id, right-aligned, so the names stay a readable column."""
    from imgui_bundle import imgui

    label = f"#{obj.id}"
    room = imgui.get_content_region_avail().x
    width = imgui.calc_text_size(label).x
    if room > width:
        imgui.same_line(imgui.get_cursor_pos().x + room - width, 0.0)
        widgets.muted(label)


def _menu(ctx: Any, state: Any, tab: Any) -> None:
    """The right-click verbs, drawn once for the pane rather than per row.

    A popup belongs to the window that begins it and imgui keys an open one by
    name, so one popup and one selection is the shape that works: a popup per
    row would be as many popups as objects, all but one of them shut.
    """
    from . import plotter_canvas

    layer = tab.doc.active()
    if not isinstance(layer, ObjectLayer):
        return
    with controls.menu_popup(MENU) as opened:
        if opened:
            plotter_canvas.object_menu_rows(ctx, state, tab, layer)


def go_to_arrow(ctx: Any, tab: Any, id: int) -> None:
    """The small arrow beside an ``object``-typed property value.

    Three states, and the third is the reason it exists at all: **unset** (0,
    Tiled's spelling of no reference) draws nothing; **live** draws an arrow
    that jumps to the object; **dangling** draws it greyed, saying the id is not
    on this map. Object ids are monotone and never reused, so a property goes on
    naming its object after the object is deleted -- and until this, a reference
    field showed a bare number with no way to tell those two apart.
    """
    from imgui_bundle import imgui

    wanted = int(id or 0)
    if not wanted:
        return
    found = tab is not None and tab.doc.find_object(wanted) is not None
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.ARROW_RIGHT}##goto-obj-{wanted}",
        bool(found),
        (sp(22), 0),
        reason=f"No object on this map has id {wanted} any more.",
        tooltip=f"Show object #{wanted}",
    ):
        plotter_mode.go_to_object_id(ctx, tab, wanted)
