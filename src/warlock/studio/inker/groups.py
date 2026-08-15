"""Layer groups: a parallel tree over the flat stack.

The stack stays authoritative. That is the whole design, and it is a deliberate
refusal of the obvious alternative -- a real tree of layers, with the
compositor walking it. Everything in this package that orders pixels is written
against a flat bottom-first list: ``composite_region``'s slicing, the
``_below`` cache's index arithmetic, "a track index and a stack index are the
same number", ``_materialize_frame``, and the ``stack_region`` native kernel's
contract. Making the stack a tree would have rewritten all of them, and every
one of those rewrites is a place a rendering bug can hide, in exchange for a
feature that is about *organising* layers rather than about painting them.

So a group is a set of members and a set of properties, kept beside the stack:

* ``Document.groups`` maps a group uid to its :class:`GroupNode`.
* ``Document.group_of`` maps a *member* uid to the group it is in. Absent means
  root. Members are layers (tracks, on an animated document) and other groups,
  which is how nesting is expressed with one dictionary rather than two.

One invariant makes the parallel structure sound: **a group's leaves are
contiguous in stack order, and spans nest.** That is exactly what makes a
group's members a slice of the painter's order and therefore something a user
can reason about -- a group whose layers were scattered through the stack could
be hidden but never *composited* as a unit. The document's ops maintain it and
:func:`check` asserts it; there is no runtime enforcement in the compositor,
because the compositor never asks.

Compositing is **pass-through**: :func:`resolve` folds a member's ancestry into
one ``(visible, opacity, locked)`` triple and the stack applies it per layer, so
a group at 50% makes each of its layers 50%, blending with everything below as
it always did. *Isolated* group compositing -- rendering the group to its own
buffer and blending that result once, which is what makes a group-level blend
mode meaningful -- is deliberately **not** implemented in v1, and is named here
so it is a known gap rather than a surprise: it needs a second compositing pass
with its own buffer and it changes what every blend mode inside the group means.

The undo steps live here too, and all four cost nothing: a group is a name,
three properties and a dictionary entry, so the byte budget has nothing to say
about them. They address by uid, like every other edit in this package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .layers import new_uid
from .undo import Edit

__all__ = [
    "GroupAddEdit",
    "GroupDissolveEdit",
    "GroupNode",
    "GroupPropsEdit",
    "MembershipEdit",
    "ancestry",
    "check",
    "copy_tree",
    "descends_from",
    "leaves_of",
    "resolve",
]


@dataclass
class GroupNode:
    """One folder in the tree. Four properties and an identity.

    The properties are the ones that *fold* -- see :func:`resolve`. A group has
    deliberately no blend mode, because a blend mode on a pass-through group
    would have nothing to blend: the members are still composited one at a time
    against everything beneath them, so there is no group result for a mode to
    act on. That is the isolated-compositing gap the module docstring names.
    """

    name: str = "Group"
    visible: bool = True
    opacity: float = 1.0
    locked: bool = False
    uid: int = field(default_factory=new_uid)


def ancestry(group_of: Mapping[int, int], uid: int) -> list[int]:
    """The groups above ``uid``, innermost first. ``[]`` at the root.

    Cycle-tolerant rather than cycle-checking: this is called on every
    composite of every layer, and a malformed tree arriving from a file must
    cost a wrong picture at worst, never a hang. :func:`check` is where a cycle
    is *reported*.
    """
    out: list[int] = []
    seen: set[int] = set()
    at = group_of.get(uid)
    while at is not None and at not in seen:
        seen.add(at)
        out.append(at)
        at = group_of.get(at)
    return out


def resolve(
    groups: Mapping[int, GroupNode], group_of: Mapping[int, int], uid: int
) -> tuple[bool, float, bool]:
    """A member's inherited ``(visible, opacity, locked)``.

    Visibility ANDs, opacity multiplies, the lock ORs, and each of the three is
    the only answer that composes. A hidden group with a visible layer in it is
    hidden (or the group's eye does nothing); a group at 50% holding a layer at
    50% shows it at 25% (or nesting two half-opacity groups would be a no-op);
    and a lock is a refusal, so any lock in the ancestry is a refusal.

    A member naming a group that is not in ``groups`` is skipped rather than
    refused -- an ORA can arrive with a dangling reference and the layer is
    still a layer.
    """
    visible, opacity, locked = True, 1.0, False
    for guid in ancestry(group_of, uid):
        node = groups.get(guid)
        if node is None:
            continue
        visible = visible and bool(node.visible)
        opacity *= float(node.opacity)
        locked = locked or bool(node.locked)
    return visible, opacity, locked


def descends_from(group_of: Mapping[int, int], uid: int, ancestor: int) -> bool:
    """Whether ``ancestor`` is ``uid`` or is above it.

    The test a move has to make: dropping a group into its own subtree would
    make a cycle, and a cycle is a group that contains itself -- which is not a
    state any op should be able to reach, so it is refused by name.
    """
    return uid == ancestor or ancestor in ancestry(group_of, uid)


def leaves_of(
    group_of: Mapping[int, int], order: Iterable[int], guid: int
) -> list[int]:
    """The stack members inside ``guid``, in stack order, at any depth."""
    return [uid for uid in order if guid in ancestry(group_of, uid)]


def copy_tree(
    groups: Mapping[int, GroupNode], group_of: Mapping[int, int]
) -> tuple[dict[int, GroupNode], dict[int, int]]:
    """A snapshot of the tree that later edits cannot write through.

    The nodes are *copied*, not merely the dictionary: a ``GroupPropsEdit``
    sets attributes on the live node, and a snapshot holding the same object
    would quietly follow it -- the same trap ``_set_tags`` names one model over.
    """
    return ({uid: replace(node) for uid, node in groups.items()}, dict(group_of))


def check(
    groups: Mapping[int, GroupNode],
    group_of: Mapping[int, int],
    order: list[int],
) -> None:
    """Raise ``ValueError`` unless the tree is well formed against ``order``.

    ``order`` is the stack's member uids, bottom-first. Pure, and used by the
    tests rather than by the ops: the ops are what *maintain* the invariant, and
    a check running inside them would be asserting the thing they just did while
    costing a walk of the stack on every composite.
    """
    positions = {uid: i for i, uid in enumerate(order)}
    known = set(groups)

    for member, parent in group_of.items():
        if parent not in known:
            raise ValueError(f"member {member} names group {parent}, which is gone")
        if member not in positions and member not in known:
            raise ValueError(f"member {member} is neither a stack row nor a group")

    for uid in list(group_of) + list(known):
        # ``ancestry`` stops at a repeat rather than looping, so a cycle shows
        # as an ancestry that comes back round to where it started.
        chain = ancestry(group_of, uid)
        if uid in chain:
            raise ValueError(f"group {uid} is inside itself")

    for guid in known:
        leaves = leaves_of(group_of, order, guid)
        if not leaves:
            raise ValueError(f"group {guid} is empty")
        indices = [positions[uid] for uid in leaves]
        if indices != list(range(min(indices), max(indices) + 1)):
            raise ValueError(f"group {guid} is not contiguous in stack order")


# --- undo steps --------------------------------------------------------------
#
# All four cost zero. A group is a name, three properties and dictionary
# entries; the byte budget exists to stop the history pinning pixels the
# document has let go of, and there are none here.


@dataclass
class GroupAddEdit(Edit):
    """A group created around members that were already in the stack.

    The node object is *held* rather than rebuilt, for ``LayerAddEdit``'s
    reason: re-inserting the same object is what keeps its uid, and the uid is
    what every later membership edit is addressed to.
    """

    node: GroupNode
    members: tuple[int, ...]
    parent: int | None = None

    def undo(self, doc: Any) -> None:
        doc._drop_group(self.node.uid)

    def redo(self, doc: Any) -> None:
        doc._put_group(self.node, self.members, self.parent)


@dataclass
class GroupDissolveEdit(Edit):
    """The same change read the other way, and a separate type because the two
    read differently at the call site: an add is undone by dissolving, and a
    dissolve is undone by putting the group back exactly as it was."""

    node: GroupNode
    members: tuple[int, ...]
    parent: int | None = None

    def undo(self, doc: Any) -> None:
        doc._put_group(self.node, self.members, self.parent)

    def redo(self, doc: Any) -> None:
        doc._drop_group(self.node.uid)


@dataclass
class GroupPropsEdit(Edit):
    """Name, visibility, opacity, lock -- the layer-property edit, one level up."""

    group_uid: int
    before: dict
    after: dict

    def undo(self, doc: Any) -> None:
        doc._set_group_props(self.group_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._set_group_props(self.group_uid, self.after)


@dataclass
class MembershipEdit(Edit):
    """One member's parent, before and after. ``None`` is the root.

    It travels in a ``CompoundEdit`` with whatever stack move keeps the group
    contiguous, because the two are one gesture: undoing the membership without
    the move would leave a group whose leaves are scattered.
    """

    member_uid: int
    before: int | None
    after: int | None

    def undo(self, doc: Any) -> None:
        doc._set_membership(self.member_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._set_membership(self.member_uid, self.after)
