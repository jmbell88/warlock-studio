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
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, clay_state, icons, widgets
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

    imgui.begin_disabled(tab.saving)
    _add(ctx, state, tab.doc)
    imgui.dummy((0, 6))
    _object_actions(ctx, state, tab.doc)
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
    mesh as one step. Phase 1 never freezes an object; the field exists from day
    one so Phase 2 adds a line rather than a migration.
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


def _object_actions(ctx: Any, state: Any, doc: Any) -> None:
    widgets.field_label("selected")
    has = bool(doc.selection)
    if widgets.disabled_button(f"{icons.COPY} Duplicate", has):
        clay_mode._duplicate_selection(ctx, state, doc)
    imgui.same_line()
    if widgets.disabled_button(f"{icons.MAXIMIZE} Bake", has):
        _bake(doc)

    widgets.field_label("mirror")
    for index, (axis, label) in enumerate(AXES):
        if widgets.disabled_button(f"{icons.FLIP_HORIZONTAL} {label}##mirror", has):
            _mirror(doc, index)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Mirror across {axis.upper()}. Baked into the mesh, never a negative "
                "scale -- glTF readers disagree about whether that flips the winding."
            )
        if index < len(AXES) - 1:
            imgui.same_line()

    if widgets.destructive_button(f"{icons.TRASH} Delete") and has:
        for uid in list(doc.selection):
            doc.remove_object(uid)


def _mirror(doc: Any, axis: int) -> None:
    for uid in list(doc.selection):
        doc.set_mesh(uid, ops.mirror(doc.by_uid(uid), axis).mesh)


def _bake(doc: Any) -> None:
    """Fold each selected object's transform into its geometry.

    Two steps rather than one compound: the mesh and the transform are separate
    edits in this document's vocabulary, and keeping them separate means an
    undo of a bake is two presses rather than a compound type that exists for
    one button.
    """
    for uid in list(doc.selection):
        baked = ops.bake_transform(doc.by_uid(uid))
        doc.set_mesh(uid, baked.mesh)
        doc.set_transform(
            uid,
            translation=baked.translation,
            rotation=baked.rotation,
            scale=baked.scale,
        )


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
