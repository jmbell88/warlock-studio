"""Clay's tool column: what to add, how to transform it, and snapping.

The same shape the raster editor's tool panel takes -- an icon grid, then the
options for whatever is selected rather than every option at once -- for the
same reason: a panel that shows all of them is unreadable, and a rotation snap
means nothing while the select tool is active.

**Every control that changes the document is disabled while a save is in
flight**, exactly as the layers panel is. Serialising reads the live document
on a task thread, so a control that restructured it mid-encode would write a
file describing a document that never existed. Disabling says so on screen
rather than swallowing the click.

**The action buttons come from the ops registry**, not from a list here. There
were three lists of what Clay can do -- this pane, the key handler and now the
context menu -- and this is the one that stopped being one. Duplicate, Bake,
Mirror and Delete still look exactly as they did; they are just rows of
``clay_ops.menu(mode)`` now, so a button cannot offer an op the menu greys out.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, clay_ops, clay_state, icons, widgets
from ..clay import document as bd
from ..clay import ops
from ..clay import primitives as bp
from ..manual import render as manual_render
from ..tokens import sp

COLUMNS = 4

TOOL_ICONS = {
    "select": icons.SQUARE_DASHED,
    "move": icons.MOVE,
    "rotate": icons.ROTATE_CW,
    "scale": icons.SCALING,
}

# One icon per generator in the registry, with a fallback so a seventh
# primitive appears the day it is added rather than the day someone remembers
# to come back here.
PRIMITIVE_ICONS = {
    "box": icons.BOX,
    "plane": icons.RECTANGLE,
    "cylinder": icons.CROP,
    "cone": icons.TRIANGLE_ALERT,
    "uv_sphere": icons.CIRCLE,
    "torus": icons.CIRCLE,
}

AXES = (("x", "X"), ("y", "Y"), ("z", "Z"))

# The four element modes and the keys that switch them. Held as data beside
# ``TOOL_ICONS`` so the row and ``clay_mode.ELEMENT_KEYS`` are one edit apart
# rather than two files apart.
MODE_BUTTONS = (
    ("object", "Object", "4"),
    ("vertex", "Verts", "1"),
    ("edge", "Edges", "2"),
    ("face", "Faces", "3"),
)


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("tools")
    manual_render.help_button(ctx, "clay-tools")
    _tool_grid(state)
    imgui.dummy((0, 6))
    if tab is None:
        widgets.muted("Open or start a document to build in.")
        return

    from . import clay_menu

    imgui.begin_disabled(tab.saving)
    _mode_row(tab.doc)
    imgui.dummy((0, 6))
    _add(ctx, state, tab.doc)
    imgui.dummy((0, 6))
    _actions(ctx, state, tab.doc)
    clay_menu.params_popup(ctx, state, tab)
    imgui.end_disabled()
    imgui.dummy((0, 6))
    # Snapping and the display toggles change nothing about the document, so
    # they stay live while a save runs.
    _snapping(state)
    imgui.dummy((0, 6))
    _display(ctx, state)


def _tool_grid(state: Any) -> None:
    width = (imgui.get_content_region_avail().x - 8 * (COLUMNS - 1)) / COLUMNS
    for index, (key, label, shortcut) in enumerate(clay_state.TOOLS):
        selected = state.tool == key
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        icon = TOOL_ICONS.get(key) or label[:1]
        if imgui.button(f"{icon}##buildtool{key}", (width, sp(30))):
            state.tool = key
        if selected:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label}  ({shortcut})")
        if index % COLUMNS != COLUMNS - 1:
            imgui.same_line()
    imgui.new_line()


def _mode_row(doc: Any) -> None:
    """Object / Verts / Edges / Faces, highlighting the document's own mode.

    The mode lives on the *document* rather than on ``ClayState``: it is the
    interpretation key for a selection, and an app-level mode would reinterpret
    every other tab's selection on a tab switch. So this reads ``doc`` and the
    state is not involved at all.
    """
    widgets.field_label("mode")
    width = (imgui.get_content_region_avail().x - 8 * (COLUMNS - 1)) / COLUMNS
    for index, (mode, label, key) in enumerate(MODE_BUTTONS):
        selected = doc.element_mode == mode
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value,
                imgui.get_style().color_(imgui.Col_.button_active.value),
            )
        if imgui.button(f"{label}##claymode{mode}", (width, sp(26))):
            doc.set_element_mode(mode)
        if selected:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label} mode  ({key})")
        if index % COLUMNS != COLUMNS - 1:
            imgui.same_line()
    imgui.new_line()


def _add(ctx: Any, state: Any, doc: Any) -> None:
    """One button per generator, straight off the registry.

    Enumerated rather than listed, so a seventh primitive is a new entry in
    ``primitives.GENERATORS`` and no edit here at all -- which is the whole
    reason that registry is data.
    """
    widgets.field_label("add")
    width = (imgui.get_content_region_avail().x - 8 * (COLUMNS - 1)) / COLUMNS
    for index, name in enumerate(sorted(bp.GENERATORS)):
        icon = PRIMITIVE_ICONS.get(name, icons.BOX)
        if imgui.button(f"{icon}##add{name}", (width, sp(28))):
            add_primitive(ctx, doc, name)
        if imgui.is_item_hovered():
            imgui.set_tooltip(name.replace("_", " "))
        if index % COLUMNS != COLUMNS - 1:
            imgui.same_line()
    imgui.new_line()
    del state


def add_primitive(ctx: Any, doc: Any, name: str) -> Any:
    """Place one primitive at its defaults, selected and ready to move.

    The generator name and the parameters it was built with are recorded on the
    object, so the properties panel can offer them and a change regenerates the
    mesh as one step -- until the first element op edits its topology, at which
    point ``clay_ops`` clears the field and the panel switches to counts.
    """
    defaults, build = bp.GENERATORS[name]
    obj = bd.Obj(
        uid=bd.new_uid(),
        name=_unique_name(doc, name.replace("_", " ").title().replace(" ", "")),
        mesh=build(**defaults),
        generator=name,
        params=dict(defaults),
    )
    doc.add_object(obj)
    doc.select([obj.uid])
    del ctx
    return obj


def _unique_name(doc: Any, base: str) -> str:
    taken = {obj.name for obj in doc.objects}
    if base not in taken:
        return base
    # The same counting-up rule ``ops.duplicate`` uses, so two objects never
    # wear one name whichever way they arrived.
    return ops._next_name(base, taken)


def _actions(ctx: Any, state: Any, doc: Any) -> None:
    """One button per registry op that applies in the current mode.

    Two columns, in registration order, with the same enablement predicate the
    menu row uses -- so a greyed-out button and a greyed-out menu row are
    literally the same call. Delete keeps its destructive styling because it is
    the one row a misclick cannot be shrugged off.
    """
    widgets.field_label("actions")
    del state
    ops_here = [op for op in clay_ops.menu(doc.element_mode) if not op.name.startswith("select-")]
    for index, op in enumerate(ops_here):
        enabled = op.enabled(doc)
        label = op.label.rstrip(".")
        if op.name == "delete":
            imgui.new_line()
            if widgets.destructive_button(f"{icons.TRASH} {label}") and enabled:
                clay_ops.run(ctx, doc, op)
            continue
        if widgets.disabled_button(f"{label}##clayop{op.name}", enabled):
            _invoke(ctx, doc, op)
        if imgui.is_item_hovered() and op.key:
            imgui.set_tooltip(op.key)
        if index % 2 == 0:
            imgui.same_line()
        else:
            imgui.new_line()


def _invoke(ctx: Any, doc: Any, op: Any) -> None:
    """Run an op, or hand a parameterised one to the popup the menu also uses."""
    from . import clay_menu

    if not op.params:
        clay_ops.run(ctx, doc, op)
        return
    state = clay_mode.ensure(ctx)
    state.pending_op = op.name
    state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
    imgui.open_popup(clay_menu.PARAM_POPUP)


def _snapping(state: Any) -> None:
    widgets.field_label("snap")
    changed, value = widgets.toggle(f"{icons.MAGNET} Snap", state.snap)
    if changed:
        state.snap = value
    imgui.begin_disabled(not state.snap)
    _, state.snap_translate = imgui.input_float("grid (m)##snapt", state.snap_translate, 0.0625)
    _, state.snap_rotate = imgui.input_float("angle (deg)##snapr", state.snap_rotate, 5.0)
    imgui.end_disabled()
    # Clamped rather than validated: zero is the off switch every snap function
    # already treats as the identity, and a negative grid is meaningless.
    state.snap_translate = max(0.0, float(state.snap_translate))
    state.snap_rotate = max(0.0, float(state.snap_rotate))


def _display(ctx: Any, state: Any) -> None:
    widgets.field_label("view")
    changed, value = widgets.toggle(f"{icons.GRID} Grid", state.grid)
    if changed:
        state.grid = value
    changed, value = widgets.toggle("Wireframe", state.wireframe)
    if changed:
        state.wireframe = value
        view = getattr(ctx, "clay_view", None)
        if view is not None:
            view.wireframe = value
