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

Compositing is **pass-through by default**: :func:`resolve` folds a member's
ancestry into one ``(visible, opacity, locked)`` triple and the stack applies it
per layer, so a group at 50% makes each of its layers 50%, blending with
everything below as it always did.

A group that is :func:`isolated` is the other case, and it is the *contiguity
invariant above* that makes it affordable rather than a rewrite. Its leaves are
a slice of the painter's order, so the isolated groups over a stack are a
well-formed bracket sequence: :func:`spans` reports each one's slice and
``LayerStack._entries`` recurses over them, rendering a group's members onto
transparency and handing the result up as one ordinary entry. The flat-stack
model every other part of this package is written against never learns that a
second pass happened.

Two things change inside an isolated group, and both are the point rather than
side effects. A member's own blend mode acts on its *siblings* instead of on
the document beneath -- which is what "isolated" means -- and the group's
opacity applies **once, to the result**, instead of multiplying into every
leaf. Those give different pictures (two overlapping opaque layers in a 50%
group show through each other pass-through, and do not when isolated), so
:func:`resolve` stops accumulating opacity at the innermost isolated ancestor
and :func:`spans` carries it instead. Every document written before this
existed is unaffected, because both fields default to the pass-through answer.

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
    "Span",
    "ancestry",
    "check",
    "copy_tree",
    "descends_from",
    "isolated",
    "leaves_of",
    "resolve",
    "spans",
    "top_level",
]


@dataclass
class GroupNode:
    """One folder in the tree. Six properties and an identity.

    Four of them *fold* into a member's own values -- see :func:`resolve`. The
    other two decide whether the group is a compositing stage at all:

    ``blend``
        The mode the group's own result blends at. Meaningful only on an
        isolated group, because a pass-through group has no result for a mode
        to act on -- which is why setting one *implies* isolation rather than
        being refused without it (see :func:`isolated`).
    ``isolate``
        Whether to composite the members onto transparency and blend that once.

    The two are separate fields rather than one, and that is not tidiness. A
    blend mode implies isolation, but isolation is independently meaningful at
    ``normal`` and full opacity: it is what confines a *member's* Multiply to
    its siblings. Deriving either from the other makes that case -- the one an
    artist reaches for most -- inexpressible. It is also what lets ORA
    round-trip honestly, ``composite-op`` and ``isolation`` being two
    attributes there.
    """

    name: str = "Group"
    visible: bool = True
    opacity: float = 1.0
    locked: bool = False
    blend: str = "normal"
    isolate: bool = False
    uid: int = field(default_factory=new_uid)


@dataclass(frozen=True)
class Span:
    """One isolated group's slice of the stack, and what its result blends at.

    ``lo``/``hi`` are half-open indices into the stack order the span was built
    against -- ``spans`` is rebuilt by ``Document.invalidate_all`` beside the
    fold, so they are never stale against the stack that holds them.

    ``opacity`` is the group's own, multiplied by whatever *pass-through*
    ancestors sit above it: a pass-through group applies its opacity to each
    member individually, and an isolated group is one of its members. The walk
    stops at the next isolated ancestor, whose own ``Span`` carries the rest.
    """

    guid: int
    lo: int
    hi: int
    opacity: float
    blend: str


def isolated(node: GroupNode) -> bool:
    """Whether this group composites as a unit.

    A blend mode implies it, because a mode needs a result to act on and only
    an isolated group has one. So a group is a compositing stage when it was
    asked to be *or* when it was given a mode -- one rule, applied everywhere,
    rather than a refusal the caller has to know about.
    """
    return bool(node.isolate) or node.blend != "normal"


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

    **Opacity stops at the innermost isolated ancestor; visibility and the lock
    do not.** An isolated group's opacity applies once to its finished result,
    which is :class:`Span`'s job, so folding it into the leaves as well would
    apply it twice. Visibility and the lock have no such split -- they are the
    same answer wherever they are asked, and the two readers that ask about
    them (``layer_at`` and the canvas's ``_group_shown``) want the complete
    one. A stack with nothing isolated takes exactly the walk it always did.

    A member naming a group that is not in ``groups`` is skipped rather than
    refused -- an ORA can arrive with a dangling reference and the layer is
    still a layer.
    """
    visible, opacity, locked = True, 1.0, False
    counting = True
    for guid in ancestry(group_of, uid):
        node = groups.get(guid)
        if node is None:
            continue
        visible = visible and bool(node.visible)
        locked = locked or bool(node.locked)
        # Tested *before* the multiply, so an isolated group's own opacity is
        # the first one left out rather than the last one included.
        if isolated(node):
            counting = False
        if counting:
            opacity *= float(node.opacity)
    return visible, opacity, locked


def spans(
    groups: Mapping[int, GroupNode],
    group_of: Mapping[int, int],
    order: Iterable[int],
) -> list[Span] | None:
    """Every isolated group's slice of ``order``, outermost-first.

    ``None`` -- not an empty list -- when nothing is isolated, which is every
    document until somebody sets a group blend mode. That is ``group_fold``'s
    rule and it is load bearing for the same two reasons: ``_entries`` takes
    its original path by identity rather than walking an empty list, and
    ``LayerStack.cuts_isolated`` can answer with one ``is None`` test on the
    hot path of every stroke.

    The bounds come from one walk of the stack rather than one walk per group:
    a row extends the bounds of every isolated ancestor it has. The contiguity
    invariant is what makes ``min``/``max`` sufficient -- a group's leaves have
    no gaps, so its extent *is* its membership. :func:`check` is where that is
    asserted; a malformed tree costs a wrong picture here, never a hang.

    Sorted by ``(lo, -hi)``, which puts a container before everything it
    contains and is what :func:`top_level` walks.
    """
    bounds: dict[int, tuple[int, int]] = {}
    for index, uid in enumerate(order):
        for guid in ancestry(group_of, uid):
            node = groups.get(guid)
            if node is None or not isolated(node):
                continue
            lo, hi = bounds.get(guid, (index, index))
            bounds[guid] = (min(lo, index), max(hi, index))
    if not bounds:
        return None
    out = [
        Span(
            guid=guid,
            lo=lo,
            hi=hi + 1,
            opacity=float(groups[guid].opacity) * _carry(groups, group_of, guid),
            blend=groups[guid].blend,
        )
        for guid, (lo, hi) in bounds.items()
    ]
    out.sort(key=lambda span: (span.lo, -span.hi))
    return out


def _carry(
    groups: Mapping[int, GroupNode], group_of: Mapping[int, int], uid: int
) -> float:
    """The opacity a member inherits from its *pass-through* ancestry.

    :func:`resolve`'s opacity half, factored out because a group needs the same
    answer a layer does: an isolated group is a member of whatever holds it,
    and a pass-through holder dims each member individually.
    """
    out = 1.0
    for guid in ancestry(group_of, uid):
        node = groups.get(guid)
        if node is None:
            continue
        if isolated(node):
            break
        out *= float(node.opacity)
    return out


def top_level(
    spans_in: Iterable[Span], lo: int, hi: int, exclude: int | None = None
) -> list[Span]:
    """The spans within ``[lo, hi)`` that no other span in the list contains.

    ``exclude`` is the group whose *inside* is being asked about, and it is not
    an optimisation: a span is contained in its own range, so a recursion that
    descended into one and asked this question again would be handed the same
    span back forever. Naming the group being entered is what makes the walk
    terminate, and it is exact -- a nested group with the same bounds as its
    parent is a different uid and still comes back.

    One level of the bracket sequence, which is what a compositing pass needs:
    the spans it blends itself, with everything nested inside left for the
    recursion those spans start.

    A span that is only *partly* inside ``[lo, hi)`` is left out entirely. That
    is a range cutting a group in half, which has no correct isolated
    rendering -- ``LayerStack.cuts_isolated`` exists so the callers never ask,
    and leaving it out degrades to pass-through rather than raising out of the
    middle of a draw, which is the choice ``_entries`` already makes about a
    fold caught mid-rebuild.

    Requires the ``(lo, -hi)`` order :func:`spans` returns.
    """
    out: list[Span] = []
    end = lo
    for span in spans_in:
        if span.guid == exclude:
            continue
        if span.lo < lo or span.hi > hi or span.lo < end:
            continue
        out.append(span)
        end = span.hi
    return out


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
