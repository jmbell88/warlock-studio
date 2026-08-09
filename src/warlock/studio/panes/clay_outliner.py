"""The object list: what is in the document, and what is selected.

Top of the list is the *last* object added, matching every 3D package and
matching the raster editor's layers panel -- the document stores objects in the
order they were placed, so this is the one place that reverses, and it does so
in one expression rather than by keeping a second ordering around.

Selection here is the same selection the viewport shows, and it is deliberately
**not undoable**: clicking an object is not laborious to redo the way a lasso
is, an undoable selection would move ``history.head``, and a document would
then ask to be saved because the user looked at a different object.

**Ctrl toggles and Shift extends**, the convention every file list and every
outliner shares. The range anchor is a *uid* on ``ClayState`` rather than a row
index, for the reason every address in this package is one: the list reorders,
and an index anchor would quietly measure the range from a different object.
An anchor that has been deleted falls back to the clicked row, which is the
same thing a plain click does.
"""

from __future__ import annotations

import contextlib
from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, icons, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp

# Design pixels: every use goes through sp(), or the row keeps its 1.0x height
# while the glyph inside it grows with the UI scale.
ROW_HEIGHT = 24.0

# The drag-and-drop payload name. One constant, because the source and the
# target have to agree and two spellings of it fail silently -- the row simply
# refuses every drop, with nothing to say why.
_DRAG_OBJECT = "clay-object"


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("outliner")
    manual_render.help_button(ctx, "clay-outliner")
    if tab is None:
        widgets.muted("Nothing open.")
        return
    doc = tab.doc
    if not doc.objects:
        widgets.empty_state(icons.LIST, "Empty document", "Add a primitive to start.")
        return

    # Visibility and selection do not change the document's *history*, but
    # renaming and deleting do -- and a save is encoding the document on a task
    # thread. The whole panel is disabled rather than half of it: a list where
    # two of four controls respond is more confusing than one that does not.
    imgui.begin_disabled(tab.saving)
    # J86. Above the rows and inside the disable, because a search that worked
    # while a save was running would let the user narrow the list to one object
    # and then find every control on it refusing the click.
    needle = widgets.list_filter(ctx, "clay-outliner", len(doc.objects))
    _visibility_row(doc)
    for index in range(len(doc.objects) - 1, -1, -1):
        obj = doc.objects[index]
        if needle and needle not in (obj.name or "").lower():
            continue
        _row(ctx, state, doc, obj, index, filtered=bool(needle))
    imgui.end_disabled()


def _visibility_row(doc: Any) -> None:
    """Solo and Show all, above the list.

    Isolating is one click where nine eye toggles were, and it is the pair of
    them that makes it usable: a solo you cannot undo in one action is a solo
    you have to remember the previous state of. Both are single undo steps, so
    Ctrl+Z is the third way back.
    """
    hidden = sum(1 for obj in doc.objects if not obj.visible)
    if widgets.disabled_button(f"{icons.EYE} Solo##claysolo", bool(doc.selection)):
        doc.isolate(doc.selection)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Show only the selected objects")
    imgui.same_line()
    if widgets.disabled_button(f"{icons.EYE_OFF} Show all##clayshowall", hidden > 0):
        doc.show_all()
    if hidden:
        widgets.muted(f"{hidden} hidden")


def _click(state: Any, doc: Any, obj: Any) -> None:
    """Apply one row click, honouring Ctrl (toggle) and Shift (range)."""
    io = imgui.get_io()
    if io.key_shift and state.outliner_anchor:
        range_uids = _range(doc, state.outliner_anchor, obj.uid)
        doc.select(range_uids)
        return
    if io.key_ctrl:
        chosen_uids = set(doc.selection) ^ {obj.uid}
        doc.select(chosen_uids)
    else:
        doc.select([obj.uid])
    state.outliner_anchor = obj.uid


def _range(doc: Any, anchor: int, uid: int) -> list[int]:
    """Every object between two uids in document order, inclusive."""
    order = [o.uid for o in doc.objects]
    try:
        lo, hi = sorted((order.index(anchor), order.index(uid)))
    except ValueError:
        return [uid]
    return order[lo : hi + 1]


def _reorder(ctx: Any, doc: Any, obj: Any, index: int, *, filtered: bool) -> None:
    """Drag one row onto another to move it in the list.

    Display order is the order ``to_model`` and ``write_glb`` emit nodes in, so
    this is an edit rather than a view setting -- and it goes through
    ``move_object``, which pushes one step.

    **Disabled while the list is filtered**, which is the one rule here that is
    not obvious. The rows on screen are then a subset, so dropping between two
    of them names a position in the *filtered* list and there is no honest
    answer for where that is in the real one -- a drop that silently landed
    somewhere else would be a reorder the user cannot see.
    """
    if filtered:
        return
    # The payload carries the *uid* rather than a row index, for the reason
    # every address in this package is one: the drop reads the list again, so a
    # reorder in between cannot make it move a different object.
    if imgui.begin_drag_drop_source(imgui.DragDropFlags_.source_no_hold_to_open_others.value):
        imgui.set_drag_drop_payload_py_id(_DRAG_OBJECT, obj.uid)
        imgui.text(obj.name or "object")
        imgui.end_drag_drop_source()
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(_DRAG_OBJECT)
        if payload is not None:
            # Suppressed rather than checked: the object can be deleted between
            # the pick-up and the drop, and there is nothing for a half-finished
            # drag to report.
            with contextlib.suppress(KeyError):
                doc.move_object(int(payload.data_id), index)
        imgui.end_drag_drop_target()


def _context_menu(ctx: Any, state: Any, doc: Any, obj: Any) -> None:
    """Rename, duplicate and delete on the row itself.

    Duplicate is here rather than only on Ctrl+D because the keyboard version
    acts on the *selection*, and the thing a user right-clicks is one row --
    which is why it selects the row first: a menu item that operated on
    something other than what was clicked would be the sharpest possible
    misreading of a right-click.
    """
    if not imgui.begin_popup_context_item(f"clayrow{obj.uid}"):
        return
    if obj.uid not in doc.selection:
        doc.select([obj.uid])
    if imgui.menu_item(f"{icons.PENCIL} Rename", "", False)[0]:
        state.renaming = obj.uid
    if imgui.menu_item(f"{icons.COPY} Duplicate", "Ctrl+D", False)[0]:
        from .. import clay_mode

        clay_mode._duplicate_selection(ctx, state, doc)
    if imgui.menu_item(f"{icons.EYE} Solo", "", False)[0]:
        doc.isolate([obj.uid])
    imgui.separator()
    if imgui.menu_item(f"{icons.TRASH} Delete", "Del", False)[0]:
        doc.remove_object(obj.uid)
    imgui.end_popup()


def _row(ctx: Any, state: Any, doc: Any, obj: Any, index: int, *, filtered: bool) -> None:
    selected = obj.uid in doc.selection
    imgui.push_id(str(obj.uid))

    eye = icons.EYE if obj.visible else icons.EYE_OFF
    if imgui.button(f"{eye}##vis", (sp(28), sp(ROW_HEIGHT))):
        doc.set_props(obj.uid, visible=not obj.visible)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Hidden objects do not render, export or pick.")
    imgui.same_line()

    # What is still to come on this line: the delete button and the gap before
    # it. Measured rather than written out -- the literal 32 this replaces was
    # already a pixel short of sp(28) + spacing at scale 1.0, and at 1.6 it
    # reserved 32 for 58 and clipped the trash button off the pane.
    width = imgui.get_content_region_avail().x - sp(28) - imgui.get_style().item_spacing.x
    if state.renaming == obj.uid:
        imgui.set_next_item_width(width)
        name = widgets.input_text("##rename", obj.name, max_length=120)
        if name != obj.name:
            doc.set_props(obj.uid, name=name)
        if imgui.is_item_deactivated():
            state.renaming = 0
    else:
        label = obj.name or f"object {index}"
        # A hidden object's name is drawn muted. It used to be a
        # ``text_colored(theme.MUTED, "")`` above the selectable, which coloured
        # nothing -- and, being an item rather than a style push, put the name
        # on the *next* line, so hiding an object silently doubled the height of
        # its row. The colour has to be pushed around the selectable, which is
        # what draws the text.
        hidden = not obj.visible
        if hidden:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.MUTED)))
        if imgui.selectable(f"{label}##row", selected, imgui.SelectableFlags_.none, (width, 0))[0]:
            _click(state, doc, obj)
        if hidden:
            imgui.pop_style_color()
        # Both hang off the selectable, which is the item the user aims at --
        # the eye and the trash button are their own targets and must keep
        # doing only what they say.
        _reorder(ctx, doc, obj, index, filtered=filtered)
        _context_menu(ctx, state, doc, obj)
        if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
            state.renaming = obj.uid

    imgui.same_line()
    if imgui.button(f"{icons.TRASH}##del", (sp(28), sp(ROW_HEIGHT))):
        doc.remove_object(obj.uid)
    imgui.pop_id()
