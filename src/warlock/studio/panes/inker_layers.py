"""The layers panel: the stack, top first, with its per-layer controls.

Top first because that is how every editor shows it and how ORA stores it --
the engine's list is bottom-first (painter's order), so this is the one place
that reverses, and it does so in one expression rather than by keeping a second
ordering around.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker, inker_mode, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_textures

# Design pixels, routed through ``sp()`` at every use site (the K97 rule --
# ``inker_timeline`` does the same with its own constants): raw, a thumbnail
# and an indent stayed physical-pixel sized at 150% DPI.
THUMB = 40.0

# Opacity the active layer held when its slider drag began, keyed by layer uid.
# Only ever holds an entry while a drag is in flight.
_opacity_drag: dict[int, float] = {}


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    tab = state.active
    widgets.section("Layers")
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
    # The Photoshop/Krita shape: what applies to the *active* layer -- blend,
    # opacity, the locks -- reads as a header above the stack, the stack is
    # rows of eye/thumbnail/name, and the verbs sit in an icon bar underneath.
    _header_controls(ctx, doc)
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
    imgui.dummy((0, 4))
    _actions_bar(ctx, doc)
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
        imgui.indent(sp(INDENT) * depth)
    folded = node.uid in tab.collapsed_groups
    if controls.small_button("+" if folded else "-"):
        tab.collapsed_groups.discard(node.uid) if folded else tab.collapsed_groups.add(
            node.uid
        )
    imgui.same_line()
    changed, visible = controls.checkbox("##groupvis", node.visible)
    if changed:
        doc.set_group_props(node.uid, visible=visible)
    imgui.same_line()
    label = f"{icons.LAYERS}  {node.name}"
    if node.opacity < 1.0:
        label += f"  {node.opacity * 100:.0f}%"
    if node.locked:
        label += "  locked"
    controls.selectable(f"{label}##grouphead", False)
    _drop_onto_group(doc, node.uid)
    if imgui.begin_popup_context_item("group-menu"):
        widgets.popup_chrome(_imgui=imgui)
        if controls.selectable("Ungroup", False)[0]:
            doc.ungroup(node.uid)
        if controls.selectable("Rename", False)[0]:
            _ask_group_rename(ctx, doc, node)
        if controls.selectable("Lock", False)[0]:
            doc.set_group_props(node.uid, locked=not node.locked)
        imgui.end_popup()
    imgui.set_next_item_width(sp(90))
    changed, value = controls.slider_float(
        "##groupopacity", float(node.opacity), 0.0, 1.0, "%.2f"
    )
    if changed:
        doc.set_group_props(node.uid, opacity=value)
    if depth:
        imgui.unindent(sp(INDENT) * depth)
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


def _header_controls(ctx: Any, doc: Any) -> None:
    """The active layer's blend, opacity and locks -- the Photoshop header.

    Blend first, then opacity, which is the order every layers panel since
    Photoshop has used and therefore the one a user's eye already knows.
    """
    layer = doc.stack.active
    blend = widgets.labeled_combo(
        "Blend",
        layer.blend,
        [(m, m) for m in inker.BLEND_MODES],
        help_text=(
            "How this layer combines with everything under it. Saved into the .ora "
            "so other editors read it the same way."
        ),
    )
    if blend != layer.blend:
        doc.set_layer_props(blend=blend)
    changed, value = widgets.labeled_slider_float(
        "Opacity",
        layer.opacity,
        0.0,
        1.0,
        help_text=(
            "The active layer's opacity. Dragging previews it live and records one "
            "undo step when you let go."
        ),
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

    widgets.muted("Lock:")
    imgui.same_line()
    if _toggle_icon(
        f"{icons.SQUARE_DASHED}##lockalpha",
        layer.alpha_lock,
        "Lock alpha: paints inside what is already on this layer and never "
        "past its edge -- colours change, transparency does not. The eraser "
        "does nothing here, because erasing is changing transparency.",
    ):
        doc.set_layer_props(alpha_lock=not layer.alpha_lock)
    imgui.same_line()
    if _toggle_icon(
        f"{icons.LOCK}##locklayer",
        layer.locked,
        "Lock layer: refuses every tool -- no strokes, fills, gradients, "
        "filters, lifts or pastes land on it. Renaming, hiding, reordering "
        "and deleting still work, and so do whole-document changes like a "
        "rotate or a crop.",
    ):
        doc.set_layer_props(locked=not layer.locked)
    imgui.same_line()
    # Its own label, not a third button under "Lock:". It is a statement about
    # what a *new cel* starts as, and filing it with the two locks would have
    # the panel claiming it refuses something -- which is the one thing it does
    # not do. Disabled, never hidden, on a still document: the engine refuses
    # it there by name and a greyed button says the setting exists.
    widgets.muted("  Cels:")
    imgui.same_line()
    continuous = doc.anim is not None and doc.anim.tracks[doc.stack.active_index].continuous
    imgui.begin_disabled(doc.anim is None)
    if _toggle_icon(
        f"{icons.COPY}##continuous",
        continuous,
        "Continuous: drawing on an empty frame of this layer starts from a "
        "copy of the last drawing on it rather than from nothing. A copy, not "
        "a link -- editing it does not change the frame it came from.",
    ):
        doc.set_layer_props(continuous=not continuous)
    imgui.end_disabled()


def _toggle_icon(icon: str, engaged: bool, tooltip: str) -> bool:
    """An icon button that shows its on state -- the anchor grid's idiom."""
    if engaged:
        imgui.push_style_color(
            imgui.Col_.button.value,
            imgui.get_style().color_(imgui.Col_.button_active.value),
        )
    clicked = widgets.icon_button(icon, tooltip)
    if engaged:
        imgui.pop_style_color()
    return clicked


def _actions_bar(ctx: Any, doc: Any) -> None:
    """The verbs, as the icon strip under the stack -- where Photoshop and
    Krita both keep them, and where they stop pushing the rows off screen."""
    if widgets.icon_button(f"{icons.PLUS}##addlayer", "Add a layer"):
        doc.add_layer()
    imgui.same_line()
    if widgets.icon_button(
        f"{icons.COPY}##duplayer", "Duplicate the active layer"
    ):
        doc.duplicate_layer()
    imgui.same_line()
    if widgets.icon_button(
        f"{icons.LAYERS}##grouplayer",
        "Group the active layer. Folders fold visibility, opacity and the "
        "lock down onto what is inside them; drag a layer onto a folder's "
        "header to move it in.",
        enabled=len(doc.stack) > 0,
    ):
        doc.group_layers()
    imgui.same_line()
    # Both work on an animated document, across every frame at once: a merge
    # is memoised on the pair of cels it consumes, so frames that shared a
    # drawing go on sharing the merged one.
    if widgets.icon_button(
        f"{icons.CHEVRON_DOWN}##mergedown",
        "Merge down: the active layer into the one under it",
        enabled=_can_merge(doc),
    ):
        doc.merge_down()
    imgui.same_line()
    if widgets.icon_button(
        f"{icons.SQUARE}##flatten",
        "Flatten every layer into one",
        enabled=len(doc.stack) > 1,
    ):
        doc.flatten_layers()
    imgui.same_line()
    if widgets.icon_button(
        f"{icons.TRASH}##dellayer",
        "Delete the active layer",
        danger=True,
        enabled=len(doc.stack) > 1,
    ):
        doc.remove_layer()
    if doc.anim is not None:
        widgets.muted_wrapped("Merge and flatten apply to every frame.")


def track_range(tab: Any, doc: Any) -> tuple[int, int] | None:
    """The timeline range's *track* span, or None. One reader, two panes.

    The range is the timeline's selection and the layers panel is the other
    view of the same axis, so a row inside it draws highlighted here and its
    controls act on the whole span. Clamped at use like every other reader of
    ``range_sel`` -- it is stored unclamped on purpose.
    """
    rect = getattr(tab, "range_sel", None)
    if rect is None or doc.anim is None:
        return None
    low, high = sorted((int(rect[0]), int(rect[1])))
    low, high = max(0, low), min(high, len(doc.anim.tracks) - 1)
    return None if low > high else (low, high)


def extend_range(tab: Any, doc: Any, index: int) -> bool:
    """Shift-click: stretch the timeline range from the active row to this one.

    The anchor is the row that *was* active, read before the click moves it --
    which is the same thing the timeline's own drag anchors on, so the two
    axes of one selection behave the same way. The frame span is left alone
    when there already is one: the user is widening the track side of an
    existing range, not starting a new one.
    """
    if doc.anim is None or not imgui.get_io().key_shift:
        return False
    anchor = doc.stack.active_index
    rect = getattr(tab, "range_sel", None)
    frames = (rect[2], rect[3]) if rect is not None else (0, len(doc.anim.frames) - 1)
    tab.range_sel = (min(anchor, index), max(anchor, index), *frames)
    return True


def _row(ctx: Any, tab: Any, doc: Any, index: int, depth: int = 0) -> None:
    layer = doc.stack[index]
    imgui.push_id(f"layer{layer.uid}")
    if depth:
        imgui.indent(sp(INDENT) * depth)
    active = index == doc.stack.active_index
    span = track_range(tab, doc)
    in_range = span is not None and span[0] <= index <= span[1]

    # The eye, not a checkbox: it is what every layers panel draws there, and
    # the off state is a different glyph rather than an empty box -- an empty
    # box beside a thumbnail reads as "unselected", which visibility is not.
    if widgets.icon_button(
        f"{icons.EYE if layer.visible else icons.EYE_OFF}##visible",
        "Hide this layer" if layer.visible else "Show this layer",
        borderless=True,
    ):
        # A row inside a multi-track range toggles the whole range, as one
        # step: the user selected a block and clicked its eye, and hiding one
        # row of it would be an answer to a question nobody asked.
        if in_range and span[1] > span[0]:
            doc.set_range_props(span[0], span[1], visible=not layer.visible)
        else:
            doc.set_layer_props(index, visible=not layer.visible)
    imgui.same_line()

    texture = inker_textures.layer_thumb(ctx, tab, index)
    if texture is not None:
        imgui.image(widgets.texture_ref(texture), (sp(THUMB), sp(THUMB)))
        imgui.same_line()

    # Name only, the full thumbnail tall: the blend/opacity subtitle moved to
    # the header above (it always described the active layer's controls) and
    # into this row's tooltip, which is where a compact panel keeps detail.
    label = layer.name
    if layer.locked or layer.alpha_lock:
        label += f"  {icons.LOCK}"
    if controls.selectable(f"{label}##pick", active or in_range, 0, (0, sp(THUMB)))[0]:
        extend_range(tab, doc, index)
        doc.set_active_layer(index)
    _reorder(doc, index)
    if imgui.is_item_hovered():
        detail = f"{layer.blend}  {layer.opacity * 100:.0f}%"
        if not layer.visible:
            detail += "  hidden"
        if layer.alpha_lock:
            detail += "  alpha locked"
        if layer.locked:
            detail += "  locked"
        imgui.set_tooltip(detail)

    if imgui.begin_popup_context_item("layer-menu"):
        widgets.popup_chrome(_imgui=imgui)
        if controls.selectable("Rename", False)[0]:
            _ask_rename(ctx, doc, index)
        if controls.selectable("Move up", False)[0]:
            doc.move_layer(index, index + 1)
        if controls.selectable("Move down", False)[0]:
            doc.move_layer(index, index - 1)
        imgui.separator()
        if controls.selectable("Group", False)[0]:
            doc.group_layers([index])
        if doc.group_of.get(_member_uid(doc, index)) is not None and controls.selectable(
            "Take out of group", False
        )[0]:
            doc.move_into_group(index, None)
        imgui.end_popup()
    if depth:
        imgui.unindent(sp(INDENT) * depth)
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
