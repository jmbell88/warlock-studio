"""Where a dragged layer lands, and which rows the panel draws.

The layers panel draws its list **top-first**, which is the opposite of the
document's order -- every layered editor does, and a bottom-first list reads as
inverted to anyone arriving from one. That inversion is the whole reason this
module exists: "dropped above the target on screen" means "after the target in
the document", and an off-by-one in that translation is a layer that lands one
place from where it was let go, every time, in a gesture whose entire feedback
is where it landed.

Pure, and headless like the rest of this package: no imgui, no ``service``, and
uid-addressed throughout. The pane owns the dragging; this owns the arithmetic.
"""

from __future__ import annotations

from typing import Any

from .tilemap import GroupLayer


def subtree_uids(layer: Any) -> set[int]:
    """Every uid at or under ``layer``.

    The pane's half of ``_map_layers._contains``: that one is a private helper
    of the document's own mixin and answers one membership test at a time,
    where a drop guard and a menu filter each want the whole set once. The two
    must agree about what "underneath" means, which they do for the only reason
    they can -- both walk ``children`` and nothing else.
    """
    out = {int(layer.uid)}
    for child in getattr(layer, "children", ()) or ():
        out |= subtree_uids(child)
    return out


def visible_rows(
    layers: list[Any], collapsed: set[int], depth: int = 0
) -> list[tuple[Any, int]]:
    """``(layer, depth)`` for every row the panel draws, top-first.

    Flattened here rather than recursed in the draw so the pane can ask "which
    row is above this one" -- which a drop needs and a recursive draw cannot
    answer without threading state through itself.

    A collapsed group contributes its own row and none of its children's. It is
    still a *group*, so it remains a drop target: folding a group to get it out
    of the way must not also stop things being put into it.
    """
    out: list[tuple[Any, int]] = []
    for layer in reversed(layers):
        out.append((layer, depth))
        children = getattr(layer, "children", None)
        if children and int(layer.uid) not in collapsed:
            out.extend(visible_rows(children, collapsed, depth + 1))
    return out


def can_fold(layer: Any) -> bool:
    """Whether this row has anything to fold open. **Groups only.**

    Object layers folded too until 2026-09-01, and were given the *same*
    chevron on purpose: one used to be an imgui tree node while every other
    kind was a selectable, so one list carried two row shapes and the eye
    landed in a different place depending on the kind. One shape, one fold.

    That argument is answered rather than abandoned: there is still one row
    shape, and an object layer is now a row with nothing under it -- like a tile
    layer, which is what it is from the stack's point of view. The objects moved
    to their own dock, because folding sixty triggers into the layer list put
    the stack sixty rows down the pane and left "where is the door I named"
    answerable only by knowing which layer it was on.
    """
    return bool(getattr(layer, "children", None))


def drop_target(doc: Any, source_uid: int, target_uid: int) -> tuple[Any, int] | None:
    """``(parent_uid, to_index)`` for dropping ``source`` onto ``target``'s row.

    ``None`` when the drop is a no-op or a cycle, so the caller can decline
    without a ``try``: ``move_layer`` raises ``ValueError`` on a group moved
    inside itself, and nothing wraps a pane draw.

    **Why it is simply the target's own index, in both directions.**
    ``move_layer`` removes the layer and *then* inserts at the index given
    (``_relocate``, and the ``len(siblings) - 1`` clamp for a same-parent move
    is the same fact stated in the document's own code). Within one parent,
    inserting at the target's original index lands the source exactly on the
    target's screen row and pushes the target one row the other way -- which
    holds whichever direction the drag went, because removing the source
    shifts the target by one precisely when the source was below it.

    Worked, on ``[a, b, c]`` (screen, top-first: ``c b a``):

    * drag ``a`` onto ``c`` -- ``t = 2``; after removal ``[b, c]``; insert at 2
      gives ``[b, c, a]``, screen ``a c b``. ``a`` took the top row.
    * drag ``c`` onto ``a`` -- ``t = 0``; after removal ``[a, b]``; insert at 0
      gives ``[c, a, b]``, screen ``b a c``. ``c`` took the bottom row.

    Across parents the target's list is untouched by the removal, so the same
    index inserts beside it there.
    """
    if int(source_uid) == int(target_uid):
        return None
    source = doc.layer(int(source_uid))
    target = doc.layer(int(target_uid))
    if source is None or target is None:
        return None
    # A group may not be dropped into its own subtree, and the target *row*
    # counts: dropping onto a child of the dragged group is the same cycle as
    # dropping onto the group.
    if int(target_uid) in subtree_uids(source):
        return None

    parent = doc.parent_uid_of(int(target_uid))
    index = doc.index_of(int(target_uid))
    if doc.parent_uid_of(int(source_uid)) == parent and (
        doc.index_of(int(source_uid)) == index
    ):
        return None
    return parent, index


def drop_into_group(doc: Any, source_uid: int, group_uid: int) -> tuple[Any, int] | None:
    """``(group_uid, index)`` for dropping ``source`` *into* a group.

    Distinct from :func:`drop_target` because dropping onto a group's row is
    genuinely two possible gestures -- reorder beside it, or reparent into it
    -- and only the pane knows which one the pointer asked for. Appends, which
    is the same landing ``Move into`` already gives from the row menu.
    """
    if int(source_uid) == int(group_uid):
        return None
    group = doc.layer(int(group_uid))
    if not isinstance(group, GroupLayer):
        return None
    source = doc.layer(int(source_uid))
    if source is None or int(group_uid) in subtree_uids(source):
        return None
    return int(group_uid), len(group.children)
