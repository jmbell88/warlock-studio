"""Undo steps for the animation grid.

Every rule ``inker/undo.py`` follows applies here unchanged, and two of them are
worth restating because the grid is where they are easiest to get wrong.

**Address by uid, never by position.** A frame is named by its uid and a track
by its uid, so an undo issued after the timeline has been reordered still lands
on the frame the edit was made to. Indices appear only where the *position* is
the thing being restored -- where in the list a removed frame goes back.

**Hold the object, do not copy it.** ``FrameRemoveEdit`` keeps the removed
``Frame`` and its cels, exactly as ``LayerRemoveEdit`` keeps its layer, because
re-inserting the same objects is what keeps their uids and keeping their uids is
what lets a patch recorded before the removal still apply after the redo.

The one genuinely new question is what a step *costs*. The byte budget exists to
stop the history pinning memory the document has let go of, so the measurement
is "would undoing this leave the history holding the only reference to these
pixels". For a linked cel the answer is no -- the layer is alive in the frames it
is still linked into, and charging for it again would evict real history to make
room for a number that describes nothing. The document ops know which case they
are in and say so through ``pinned``; guessing here from the grid's state would
be wrong for adds, whose layers are in the grid at the moment the edit is built.

``pinned`` is therefore not simply a bool. A removal is routinely *mixed* -- one
cel the grid has let go of beside another still linked into the frame next door
-- so it also takes the ``id()``s of the layers that really are pinned, and
``charged`` sums those alone. A bool in that case is wrong whichever way it is
set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..undo import Edit

__all__ = [
    "AnimateEdit",
    "CelSetEdit",
    "FrameAddEdit",
    "FrameDurationEdit",
    "FrameMoveEdit",
    "FrameOrderEdit",
    "FrameRemoveEdit",
    "TagsEdit",
    "TrackAddEdit",
    "TrackMoveEdit",
    "TrackPropsEdit",
    "TrackRemoveEdit",
]


def pixel_bytes(layers: Iterable[Any]) -> int:
    """Bytes held by these layers, counting a shared object once."""
    seen: dict[int, int] = {}
    for layer in layers:
        if layer is not None:
            seen[id(layer)] = int(layer.pixels.nbytes)
    return sum(seen.values())


def charged(layers: Iterable[Any], pinned: Any) -> int:
    """What this step costs the byte budget.

    ``pinned`` is True or False when the whole set is in one case, and a
    *collection of ``id()``s* when it is not -- which a removal routinely is,
    because a frame can hold one cel the grid is letting go of beside another
    that is still linked into the frame next door. Charging for the linked one
    too evicts real history to make room for a number describing no memory,
    and the alternative reading (charge for neither) loses the budget's grip on
    the one that genuinely is pinned.
    """
    if pinned is True:
        return pixel_bytes(layers)
    if pinned is False:
        return 0
    keep = set(pinned)
    return pixel_bytes(layer for layer in layers if id(layer) in keep)


@dataclass
class AnimateEdit(Edit):
    """Still document <-> animated document.

    The ``Animation`` object is carried rather than rebuilt, for the reason
    ``LayerAddEdit`` carries its layer: ``ensure_animation`` mints a frame uid,
    and a redo that minted a fresh one would strand every cel key, stamp and
    later edit that names the old one. Cost is zero because the cels *are* the
    document's existing layers -- animating copies no pixels at all.
    """

    anim: Any

    def undo(self, doc: Any) -> None:
        doc._set_animation(None)

    def redo(self, doc: Any) -> None:
        doc._set_animation(self.anim)


@dataclass
class FrameAddEdit(Edit):
    index: int
    frame: Any
    cels: dict[int, Any] = field(default_factory=dict)
    pinned: Any = True

    def __post_init__(self) -> None:
        self.cost = charged(self.cels.values(), self.pinned)

    def undo(self, doc: Any) -> None:
        doc._drop_frame(self.frame)

    def redo(self, doc: Any) -> None:
        doc._put_frame(self.index, self.frame, self.cels)


@dataclass
class FrameRemoveEdit(Edit):
    index: int
    frame: Any
    cels: dict[int, Any] = field(default_factory=dict)
    pinned: Any = True

    def __post_init__(self) -> None:
        self.cost = charged(self.cels.values(), self.pinned)

    def undo(self, doc: Any) -> None:
        doc._put_frame(self.index, self.frame, self.cels)

    def redo(self, doc: Any) -> None:
        doc._drop_frame(self.frame)


@dataclass
class FrameMoveEdit(Edit):
    frame_uid: int
    index: int
    to: int

    def undo(self, doc: Any) -> None:
        doc._move_frame(self.frame_uid, self.index)

    def redo(self, doc: Any) -> None:
        doc._move_frame(self.frame_uid, self.to)


@dataclass
class FrameOrderEdit(Edit):
    """The whole frame order, before and after, as uid sequences.

    A reorder of a *range* is expressible as a chain of ``FrameMoveEdit``s and
    that is what it must not be. Each of those ends in ``_move_frame``, which
    ends in ``_anim_changed``, which stamps every frame and recomposites the
    canvas -- so reversing a fifty-frame clip would rebuild the view fifty
    times for one gesture, and a partial failure halfway through would leave an
    order the user never asked for on screen.

    Uids rather than indices, for the reason the module docstring gives: the
    positions are what is being restored, but the *things* being positioned
    have to survive an unrelated insert or delete underneath this step. Cost is
    zero because a permutation moves no pixels at all -- the cels are keyed by
    frame uid, so links survive a reorder without anything being said about
    them.
    """

    before: list[int]
    after: list[int]

    def undo(self, doc: Any) -> None:
        doc._set_frame_order(self.before)

    def redo(self, doc: Any) -> None:
        doc._set_frame_order(self.after)


@dataclass
class FrameDurationEdit(Edit):
    frame_uid: int
    before: int
    after: int

    def undo(self, doc: Any) -> None:
        doc._set_duration(self.frame_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._set_duration(self.frame_uid, self.after)


@dataclass
class TrackAddEdit(Edit):
    index: int
    track: Any
    cels: dict[int, Any] = field(default_factory=dict)
    pinned: Any = True

    def __post_init__(self) -> None:
        self.cost = charged(self.cels.values(), self.pinned)

    def undo(self, doc: Any) -> None:
        doc._drop_track(self.track)

    def redo(self, doc: Any) -> None:
        doc._put_track(self.index, self.track, self.cels)


@dataclass
class TrackRemoveEdit(Edit):
    index: int
    track: Any
    cels: dict[int, Any] = field(default_factory=dict)
    pinned: Any = True

    def __post_init__(self) -> None:
        self.cost = charged(self.cels.values(), self.pinned)

    def undo(self, doc: Any) -> None:
        doc._put_track(self.index, self.track, self.cels)

    def redo(self, doc: Any) -> None:
        doc._drop_track(self.track)


@dataclass
class TrackMoveEdit(Edit):
    track_uid: int
    index: int
    to: int

    def undo(self, doc: Any) -> None:
        doc._move_track(self.track_uid, self.index)

    def redo(self, doc: Any) -> None:
        doc._move_track(self.track_uid, self.to)


@dataclass
class TrackPropsEdit(Edit):
    """Name, opacity, visibility, blend -- the layer-property edit, one row up.

    It is a *track* edit rather than a layer one because the track is
    authoritative: writing the property onto the materialised layer instead
    would last exactly until the next time that frame was rebuilt.
    """

    track_uid: int
    before: dict
    after: dict

    def undo(self, doc: Any) -> None:
        doc._set_track_props(self.track_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._set_track_props(self.track_uid, self.after)


@dataclass
class TagsEdit(Edit):
    """The whole tag list, before and after.

    One type for create, rename, retime, loop and delete, where the grid gets a
    type per operation -- and the difference is that a tag has no uid to address
    it by. Its identity *is* its position in the list, so "tag 2's end moved"
    and "tag 1 was deleted" are the same kind of change to the same small list,
    and an edit that named a tag by index would be reversing the wrong one after
    any delete. Snapshotting is affordable here for the reason it is not for a
    layer: a tag is a name and three numbers.

    Cost is zero for the same reason. The lists hold no pixels, so the byte
    budget has nothing to say about them.
    """

    before: list
    after: list

    def undo(self, doc: Any) -> None:
        doc._set_tags(self.before)

    def redo(self, doc: Any) -> None:
        doc._set_tags(self.after)


@dataclass
class CelSetEdit(Edit):
    """One slot of the grid, before and after.

    Four operations are the same operation seen through this one type, which is
    why there is one type: drawing on an empty frame (``before`` None), clearing
    a cel (``after`` None), linking a frame to another's cel (``after`` is that
    existing object) and unlinking (``after`` is a fresh copy). The unlink case
    is the one with a rule attached -- its copy, and therefore its uid, is made
    **once, at op time**, and the edit holds it. A redo that copied again would
    hand back a layer with a new identity and strand every patch recorded
    against the first one.
    """

    track_uid: int
    frame_uid: int
    before: Any
    after: Any
    #: True or False when both sides are in the same case, and a *collection of
    #: ``id()``s* when they are not -- the same three-valued spelling the
    #: removals use, and for the same reason one step up. A range op sets many
    #: slots at once and routinely mixes the cases inside one gesture: unlinking
    #: a span mints a fresh copy per slot (each of which the history alone will
    #: hold once undone) while the *original* they came from is one object that
    #: must be charged once across all of them, not once per slot. A bool is
    #: wrong whichever way it is set, and setting it True per slot would charge
    #: a 50-frame linked background fifty times and evict the whole history.
    #: An animated merge-down is the same shape from the other direction: it
    #: pushes one of these per affected slot and several of them legitimately
    #: name *one* merged cel (that is how the link is preserved), so charging
    #: every one of them would evict real history to make room for a number
    #: describing pixels that were counted already.
    pinned: Any = True

    def __post_init__(self) -> None:
        # Only the direction that can be *left holding* pixels is charged for.
        # Undoing a set restores ``before``, so what the history pins while the
        # step is on the done stack is ``after``, and vice versa -- but the two
        # stacks together are what ``UndoStack.bytes`` sums, so charge both.
        self.cost = charged([self.before, self.after], self.pinned)

    def _put(self, doc: Any, layer: Any) -> None:
        doc._set_cel(self.track_uid, self.frame_uid, layer)

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)
