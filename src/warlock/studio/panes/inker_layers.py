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
from ..tokens import sp
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
    shown = 0
    open_groups: list[int] = []
    for index in range(len(doc.stack) - 1, -1, -1):
        # The group headers are emitted *between* rows rather than as a nested
        # widget tree: the flat stack is authoritative, so the panel walks it
        # exactly as it always did and the folders are punctuation. That is what
        # keeps the filter, the drag reorder and the row ids unchanged.
        hidden = _headers(ctx, tab, doc, index, open_groups)
        if hidden:
            continue
        if needle and needle not in (doc.stack[index].name or "").lower():
            continue
        shown += 1
        _row(ctx, tab, doc, index, len(open_groups))
    widgets.no_matches(needle, shown)
    imgui.end_disabled()


INDENT = 14.0


def _headers(ctx: Any, tab: Any, doc: Any, index: int, open_groups: list[int]) -> bool:
    """Open and close the group headers around row ``index``. Collapsed?

    Called once per row on the way down the panel (top-first), and it keeps
    ``open_groups`` as the outermost-first chain the row sits in -- which is
    both the indent depth and the answer to "is this row inside something the
    user has folded shut".

    A header is drawn the first time a group appears, which given the
    contiguity invariant is exactly once: a group's rows are a run, so the run
    begins at one row and the header belongs above it.
    """
    from ..inker import groups as gp

    if not doc.groups:
        return False
    chain = list(reversed(gp.ancestry(doc.group_of, doc.member_uids()[index])))
    shared = 0
    while (
        shared < len(open_groups)
        and shared < len(chain)
        and open_groups[shared] == chain[shared]
    ):
        shared += 1
    del open_groups[shared:]
    collapsed = False
    for depth, guid in enumerate(chain[shared:], start=shared):
        node = doc.groups.get(guid)
        if node is None:
            continue
        if not collapsed:
            _group_row(ctx, tab, doc, node, depth)
        open_groups.append(guid)
        collapsed = collapsed or guid in tab.collapsed_groups
    return collapsed or any(guid in tab.collapsed_groups for guid in open_groups)


def _group_row(ctx: Any, tab: Any, doc: Any, node: Any, depth: int) -> None:
    """One folder: a fold arrow, an eye, a name, and a drop target."""
    imgui.push_id(f"group{node.uid}")
    if depth:
        imgui.indent(INDENT * depth)
    folded = node.uid in tab.collapsed_groups
    if imgui.small_button("+" if folded else "-"):
        tab.collapsed_groups.discard(node.uid) if folded else tab.collapsed_groups.add(
            node.uid
        )
    imgui.same_line()
    changed, visible = imgui.checkbox("##groupvis", node.visible)
    if changed:
        doc.set_group_props(node.uid, visible=visible)
    imgui.same_line()
    label = f"{icons.LAYERS}  {node.name}"
    if node.opacity < 1.0:
        label += f"  {node.opacity * 100:.0f}%"
    if node.locked:
        label += "  locked"
    imgui.selectable(f"{label}##grouphead", False)
    _drop_onto_group(doc, node.uid)
    if imgui.begin_popup_context_item("group-menu"):
        if imgui.selectable("Ungroup", False)[0]:
            doc.ungroup(node.uid)
        if imgui.selectable("Rename", False)[0]:
            _ask_group_rename(ctx, doc, node)
        if imgui.selectable("Lock", False)[0]:
            doc.set_group_props(node.uid, locked=not node.locked)
        imgui.end_popup()
    imgui.set_next_item_width(sp(90))
    changed, value = imgui.slider_float(
        "##groupopacity", float(node.opacity), 0.0, 1.0, "%.2f"
    )
    if changed:
        doc.set_group_props(node.uid, opacity=value)
    if depth:
        imgui.unindent(INDENT * depth)
    imgui.pop_id()


def _drop_onto_group(doc: Any, group_uid: int | None) -> None:
    """Accept a dragged layer onto a header, moving it into that group.

    The same payload the row-to-row reorder uses, so one drag can be dropped on
    either -- and the engine's ``move_into_group`` does the stack move that
    keeps the group contiguous, in the same undo step as the membership.
    """
    if not imgui.begin_drag_drop_target():
        return
    payload = imgui.accept_drag_drop_payload_py_id("inker-layer")
    if payload is not None:
        source = int(payload.data_id)
        if 0 <= source < len(doc.stack):
            doc.move_into_group(source, group_uid)
    imgui.end_drag_drop_target()


def _ask_group_rename(ctx: Any, doc: Any, node: Any) -> None:
    from .. import dialogs

    ctx.prompts.ask(
        dialogs.Prompt(
            title="Rename group",
            label="Name",
            value=node.name,
            on_accept=lambda text: doc.set_group_props(node.uid, name=text[:60]),
        )
    )


def _can_merge(doc: Any) -> bool:
    """Whether Merge down would be accepted, so the button says so first.

    The engine refuses a merge when *either* participant is content-locked, and
    a button that looks live and does nothing is worse than a greyed one.
    """
    index = doc.stack.active_index
    if index <= 0:
        return False
    return not (doc.write_locked(doc.stack[index]) or doc.write_locked(doc.stack[index - 1]))


def _actions(ctx: Any, doc: Any) -> None:
    if imgui.button("Add"):
        doc.add_layer()
    imgui.same_line()
    if imgui.button("Copy"):
        doc.duplicate_layer()
    imgui.same_line()
    if widgets.disabled_button("Delete", len(doc.stack) > 1):
        doc.remove_layer()
    # Both work on an animated document now, across every frame at once: a
    # merge is memoised on the pair of cels it consumes, so frames that shared
    # a drawing go on sharing the merged one.
    if widgets.disabled_button("Merge down", _can_merge(doc)):
        doc.merge_down()
    imgui.same_line()
    if widgets.disabled_button("Flatten", len(doc.stack) > 1):
        doc.flatten_layers()
    if widgets.disabled_button("Group", len(doc.stack) > 0):
        doc.group_layers()
    widgets.help_marker(
        "Wraps the active layer in a folder. Folders fold visibility, opacity "
        "and the lock down onto what is inside them, and drag a layer onto a "
        "folder's header to move it in. A folder is made around layers that "
        "are already next to each other -- there are no empty ones."
    )
    if doc.anim is not None:
        widgets.muted("Merge and flatten apply to every frame.")

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
    blend = widgets.labeled_combo("Blend", layer.blend, [(m, m) for m in inker.BLEND_MODES])
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
    changed, content = imgui.checkbox("Lock layer", layer.locked)
    widgets.help_marker(
        "Refuses every tool: no strokes, fills, gradients, filters, lifts or "
        "pastes land on it. Renaming, hiding, reordering and deleting still "
        "work, and so do whole-document changes like a rotate or a crop -- the "
        "lock is about what gets painted, not about managing the layer."
    )
    if changed:
        doc.set_layer_props(locked=content)


def _row(ctx: Any, tab: Any, doc: Any, index: int, depth: int = 0) -> None:
    layer = doc.stack[index]
    imgui.push_id(f"layer{layer.uid}")
    if depth:
        imgui.indent(INDENT * depth)
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
        + ("  alpha" if layer.alpha_lock else "")
        + ("  locked" if layer.locked else ""),
    )
    imgui.end_group()

    if imgui.begin_popup_context_item("layer-menu"):
        if imgui.selectable("Rename", False)[0]:
            _ask_rename(ctx, doc, index)
        if imgui.selectable("Move up", False)[0]:
            doc.move_layer(index, index + 1)
        if imgui.selectable("Move down", False)[0]:
            doc.move_layer(index, index - 1)
        imgui.separator()
        if imgui.selectable("Group", False)[0]:
            doc.group_layers([index])
        if doc.group_of.get(_member_uid(doc, index)) is not None and imgui.selectable(
            "Take out of group", False
        )[0]:
            doc.move_into_group(index, None)
        imgui.end_popup()
    if depth:
        imgui.unindent(INDENT * depth)
    imgui.pop_id()


def _member_uid(doc: Any, index: int) -> int:
    """What the group tree knows this row by -- the *track* uid on an animated
    document, because a materialised empty cel carries a placeholder uid of its
    own. ``Document.member_uids`` owns the argument."""
    order = doc.member_uids()
    return order[index] if 0 <= index < len(order) else -1


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
