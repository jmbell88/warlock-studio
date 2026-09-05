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

from .. import clay_mode, clay_ops, controls, icons, tokens, widgets
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
    "grid": icons.GRID,
    "capsule": icons.EGG,
    "icosphere": icons.STAR,
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
    """This pane's headings, on tinted blocks.

    The blocks are opened *here* rather than in :func:`layout.pane`, which is
    flat: a pane on a wide canvas wants no tint, and this is one of the four
    narrow sidebars the grouping was written for (see
    ``tests/test_section_blocks.py`` for the report it came from). Wrapping
    ``_body`` rather than inlining the ``with`` keeps every early return inside
    the scope, and the scope closes its last block on the way out.
    """
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    """What you can *add* and what you can *do*, and nothing else.

    Half of what this pane held has gone to the viewport header: the tool grid,
    the mode row, snapping, proportional editing and the view aids. Every one of
    them is a setting changed between clicks in the viewport, and this sidebar
    is on the far side of the window from it.

    What is left is what a sidebar is right for -- two lists that want the
    height, and that are read down rather than flicked between.
    """

    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Tools")
    manual_render.help_button(ctx, "clay-tools")
    if tab is None:
        widgets.muted("Open or start a document to build in.")
        return

    from . import clay_menu

    imgui.begin_disabled(tab.saving)
    _add(ctx, state, tab.doc)
    imgui.dummy((0, sp(tokens.SP_2)))
    _actions(ctx, state, tab.doc)
    imgui.end_disabled()
    # *Outside* the disabled block: the popup greys its own Apply against
    # tab.saving and its Cancel must stay live, or a save that starts while it
    # is open leaves a modal the user cannot dismiss -- the exact trap
    # inker_bridge documents.
    clay_menu.params_popup(ctx, state, tab)


def _add(ctx: Any, state: Any, doc: Any) -> None:
    """One button per generator, straight off the registry.

    Enumerated rather than listed, so a seventh primitive is a new entry in
    ``primitives.GENERATORS`` and no edit here at all -- which is the whole
    reason that registry is data.
    """
    widgets.field_label("add")
    width = widgets.grid_width(COLUMNS)
    for index, name in enumerate(sorted(bp.GENERATORS)):
        icon = PRIMITIVE_ICONS.get(name, icons.BOX)
        if controls.button(f"{icon}##add{name}", (width, sp(28))):
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
    return ops.next_name(base, taken)


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
            # Greyed like every other row here. Checking ``enabled`` *after* the
            # click drew a live red button that did nothing -- and this is the
            # one button where "nothing happened" is hardest to tell apart from
            # "something irreversible happened".
            if widgets.destructive_button(f"{icons.TRASH} {label}", enabled=enabled):
                clay_ops.run(ctx, doc, op)
            continue
        if widgets.disabled_button(f"{label}##clayop{op.name}", enabled):
            _invoke(ctx, doc, op)
        # The key *and* the sentence. ``Op.hint`` was written for the dialog a
        # parameterised op opens, which means the explanation of what an op is
        # for was reachable only by pressing the button -- and the two ops it
        # most has to tell apart sit side by side here.
        tip = "\n".join(part for part in (op.key, op.hint) if part)
        if imgui.is_item_hovered() and tip:
            imgui.set_tooltip(tip)
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
