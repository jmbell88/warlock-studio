"""Range operations over the animation grid: a rectangle of cells at a time.

Everything here is the per-cel op in ``_doc_anim`` with a rectangle in front of
it, and the three rules that turns into are what this module exists to hold.

**The engine never sees a selection.** Every op takes explicit indices, so the
whole of what "the user dragged a marquee over the timeline" means to the
document is four integers. The selection itself is view state on the tab
(``InkerDoc.range_sel``) and it is *indices, not uids* -- the same choice
``Tag`` makes and for the same reason: a range names a region of the timeline,
so inserting a frame inside it should widen it. It is clamped **at use, never
at store**, which is Plotter's selection rule: a range that survived a delete
by being trimmed at store time would silently shrink under the user, and one
that was never trimmed at all would index off the end.

**One gesture is one Ctrl+Z.** Each op mutates through the raw grid hooks the
per-cel ops already use and pushes exactly one step -- a ``CompoundEdit`` when
there is more than one thing to reverse, the bare edit when there is not, which
is ``add_frame``'s own idiom. An op that changes nothing pushes nothing and
returns False, because ``dirty`` is a comparison against ``history.head`` and a
step that changed nothing would make a saved document ask to be saved.

**A shared cel is charged once.** The byte budget measures what the history
would be left holding, so a range op that touches a linked cel through five
slots must charge for its pixels once -- and only when the grid has genuinely
let go of them. Every op below that can strand pixels therefore computes
``_released`` *after* its mutation and hands the first slot that mentions each
released object the whole bill. See ``anim_edits.charged``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import composite as cp
from . import filters
from . import transform as tf
from .anim_edits import (
    CelSetEdit,
    FrameAddEdit,
    FrameDurationEdit,
    FrameOrderEdit,
    FrameRemoveEdit,
    TrackPropsEdit,
)
from .animation import TRACK_PROPS, Frame, clamp_duration
from .undo import CompoundEdit, IndexPatchEdit, PatchEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document

__all__ = ["CelClip", "RangeOps", "masked_apply"]


def clamp_span(a: Any, b: Any, count: int) -> tuple[int, int] | None:
    """``(low, high)`` inside ``0..count-1``, or None when there is no overlap.

    Pure, and the only place a range is trimmed. The ends are sorted here
    rather than at the call sites because a marquee dragged up and to the left
    hands over a rect whose "start" is past its "end", and every op below would
    otherwise have to remember that.
    """
    if count < 1:
        return None
    low, high = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
    low = max(0, low)
    high = min(high, count - 1)
    if low > high:
        return None
    return low, high


def masked_apply(
    before: np.ndarray,
    filtered: np.ndarray,
    weight: np.ndarray | None,
    *,
    alpha_lock: bool = False,
) -> np.ndarray:
    """``filtered`` faded into ``before`` by a selection weight, as uint8.

    Lifted out of ``preview_filter`` rather than restated, because the rule it
    encodes -- the selection is a *weight* and not a rectangle, so a feathered
    edge fades a filter in -- is the one thing every write in this editor does
    the same way, and two copies of it is how "feathering" comes to mean two
    things. ``weight`` is None for "no selection", which is the whole region at
    full strength.

    The alpha lock is restored after the blend rather than folded into it, for
    ``write_colour``'s reason: "preserve transparency" is exactly *the alpha
    does not change*, so putting the channel back is the definition of it.

    With no selection and no lock the filtered array is handed straight back
    rather than copied. That is not a micro-optimisation: ``preview_filter``
    calls this every frame the popup is up over a memoised array, and a copy
    here would be a full-canvas one per frame on top of the write the caller
    already makes -- exactly the cost the memo exists to remove. Nothing is
    ever written into the argument, so the only obligation on a caller is not
    to mutate what comes back, which is what "apply" means.
    """
    if weight is None:
        out = filtered
    else:
        fade = weight.astype(np.float32)[..., None] / 255.0
        base = before.astype(np.float32)
        out = cp.to_uint8_255(base + (filtered.astype(np.float32) - base) * fade)
    if alpha_lock:
        # Copy only if the array is still the caller's: the blended one above
        # is ours to write into, the memoised one is not.
        out = out.copy() if out is filtered else out
        out[..., 3] = before[..., 3]
    return out


@dataclass
class CelClip:
    """A rectangle of cels, lifted out of the grid and owned outright.

    The planes are *deduped by distinct cel* and the slots index into them,
    which is ``ReplayEdit.grid["slots"]``'s trick at a different scale and for
    the same reason: two slots holding one index come back as two slots holding
    one object, so a clip cut from three frames sharing one background pastes
    as three frames sharing one background. Recording a layer per slot would
    paste three equal copies and quietly break the link, and the user would
    only find out on the next stroke.

    The pixels are copied at *copy* time, not referenced: a clip outlives the
    edits made after it was taken, and a clipboard that showed later strokes
    would be a clipboard nobody could reason about.
    """

    size: tuple[int, int]
    tracks: int
    frames: int
    planes: list[Any] = field(default_factory=list)
    #: ``(track offset, frame offset) -> index into planes``.
    slots: dict[tuple[int, int], int] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.slots)


class RangeOps:
    """Timeline-range edits, mixed into :class:`~.document.Document`."""

    # -- shared plumbing ----------------------------------------------------

    def _range(
        self: Document, t0: Any, t1: Any, f0: Any, f1: Any
    ) -> tuple[int, int, int, int] | None:
        """A cell rect clamped onto the grid, or None if it misses entirely."""
        anim = self.anim
        if anim is None:
            return None
        tracks = clamp_span(t0, t1, len(anim.tracks))
        frames = clamp_span(f0, f1, len(anim.frames))
        if tracks is None or frames is None:
            return None
        return (tracks[0], tracks[1], frames[0], frames[1])

    def _frames(self: Document, f0: Any, f1: Any) -> tuple[int, int] | None:
        anim = self.anim
        return None if anim is None else clamp_span(f0, f1, len(anim.frames))

    def _push_range(self: Document, edits: list[Any]) -> bool:
        """One step for the whole gesture, or nothing at all."""
        if not edits:
            return False
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        return True

    def _set_frame_order(self: Document, uids: list[int]) -> None:
        """Install a permutation of the frames. The ``FrameOrderEdit`` hook.

        The playhead is re-derived **by uid**: it is a position, and the whole
        point of the op is that positions moved, so keeping the index would
        leave the user looking at a different drawing than the one they were
        on. A uid no longer in the grid (a frame deleted after this step was
        recorded, then this step redone) falls back to a clamp rather than
        raising -- there is no frame to go back to and refusing to reorder
        would be the worse answer.
        """
        anim = self.anim
        assert anim is not None
        by_uid = {frame.uid: frame for frame in anim.frames}
        current = anim.frames[max(0, min(anim.current, len(anim.frames) - 1))].uid
        named = set(uids)
        ordered = [by_uid[uid] for uid in uids if uid in by_uid]
        # Anything the caller did not name keeps its place at the end, so a
        # stale order can never *drop* a frame -- the one outcome that would
        # lose pixels rather than merely arrange them oddly.
        ordered += [frame for frame in anim.frames if frame.uid not in named]
        anim.frames = ordered
        try:
            anim.current = anim.frame_index(current)
        except KeyError:  # pragma: no cover - the frame outlives the order
            anim.current = max(0, min(anim.current, len(anim.frames) - 1))
        self._anim_changed()

    def _reorder(self: Document, after: list[int]) -> bool:
        anim = self.anim
        if anim is None:
            return False
        before = [frame.uid for frame in anim.frames]
        if after == before:
            return False
        self.commit_floating()
        self._set_frame_order(after)
        self.history.push(FrameOrderEdit(before, after))
        return True

    # -- frames -------------------------------------------------------------

    def remove_range(self: Document, f0: int, f1: int) -> bool:
        """Delete a span of frames as one step. Refuses the whole timeline.

        Dropped **descending**, which is what makes the undo correct rather
        than merely reversible: ``CompoundEdit`` undoes its children in reverse
        order, so the re-inserts run low index first and each one's recorded
        index is still the position that frame belongs at. Ascending removal
        would record indices that had already shifted under themselves.
        """
        anim = self.anim
        span = self._frames(f0, f1)
        if anim is None or span is None:
            return False
        f0, f1 = span
        if f0 == 0 and f1 == len(anim.frames) - 1:
            # The grid keeps at least one frame -- ``_drop_frame`` asserts it --
            # and "delete every frame" is a request to un-animate the document,
            # which is what undoing the first ``add_frame`` is for.
            return False
        self.commit_floating()
        removed: list[tuple[int, Any, dict[int, Any]]] = []
        for index in range(f1, f0 - 1, -1):
            frame = anim.frames[index]
            cels = {
                track.uid: anim.cels[(track.uid, frame.uid)]
                for track in anim.tracks
                if (track.uid, frame.uid) in anim.cels
            }
            self._drop_frame(frame)
            removed.append((index, frame, cels))
        released = self._released(
            [layer for _index, _frame, cels in removed for layer in cels.values()]
        )
        seen: set[int] = set()
        edits: list[Any] = []
        for index, frame, cels in removed:
            pinned = set()
            for layer in cels.values():
                if id(layer) in released and id(layer) not in seen:
                    seen.add(id(layer))
                    pinned.add(id(layer))
            edits.append(FrameRemoveEdit(index, frame, cels, pinned=frozenset(pinned)))
        return self._push_range(edits)

    def duplicate_range(
        self: Document, f0: int, f1: int, at: int | None = None, *, link: bool = False
    ) -> bool:
        """Copy a span of frames in, after it by default.

        The copy is made **per distinct object**, not per slot: a background
        linked across the whole span becomes *one* new layer shared by the new
        frames, so the copy has the same internal link structure as the
        original and no link back to it. Per-slot copying would turn one
        drawing into five, and the user would find out when editing the
        duplicate's background changed only one frame of it.

        ``link=True`` shares the source objects instead, and charges nothing --
        the same trade ``add_frame(link=True)`` makes, for the same reason: the
        layers are alive in the frames they came from whatever the history does
        with this step.
        """
        anim = self.anim
        span = self._frames(f0, f1)
        if anim is None or span is None:
            return False
        f0, f1 = span
        self.commit_floating()
        at = f1 + 1 if at is None else max(0, min(int(at), len(anim.frames)))
        # Read before any insert: inserting at or below ``f0`` moves the very
        # frames still being read.
        sources = [
            (
                anim.frames[index],
                {
                    track.uid: anim.cels[(track.uid, anim.frames[index].uid)]
                    for track in anim.tracks
                    if (track.uid, anim.frames[index].uid) in anim.cels
                },
            )
            for index in range(f0, f1 + 1)
        ]
        minted: dict[int, Any] = {}
        charged: set[int] = set()
        edits: list[Any] = []
        for offset, (source, cels) in enumerate(sources):
            frame = Frame(duration_ms=source.duration_ms)
            fresh: dict[int, Any] = {}
            pinned: set[int] = set()
            for track_uid, cel in cels.items():
                if link:
                    fresh[track_uid] = cel
                    continue
                copy = minted.get(id(cel))
                if copy is None:
                    copy = cel.copy(name=cel.name)
                    minted[id(cel)] = copy
                fresh[track_uid] = copy
                if id(copy) not in charged:
                    charged.add(id(copy))
                    pinned.add(id(copy))
            index = at + offset
            self._put_frame(index, frame, fresh)
            edits.append(
                FrameAddEdit(
                    index, frame, fresh, pinned=False if link else frozenset(pinned)
                )
            )
        return self._push_range(edits)

    def move_range(self: Document, f0: int, f1: int, to: int) -> bool:
        """Move a span so its first frame lands at index ``to``.

        ``to`` is read against the timeline *without* the span in it, which is
        the only reading that makes "move frames 3-5 to 0" mean what it says
        however far the block is being dragged. One ``FrameOrderEdit``, so
        links survive trivially -- cels are keyed by frame uid and a
        permutation touches no key.
        """
        anim = self.anim
        span = self._frames(f0, f1)
        if anim is None or span is None:
            return False
        f0, f1 = span
        uids = [frame.uid for frame in anim.frames]
        block = uids[f0 : f1 + 1]
        rest = uids[:f0] + uids[f1 + 1 :]
        at = max(0, min(int(to), len(rest)))
        return self._reorder(rest[:at] + block + rest[at:])

    def reverse_range(self: Document, f0: int, f1: int) -> bool:
        anim = self.anim
        span = self._frames(f0, f1)
        if anim is None or span is None:
            return False
        f0, f1 = span
        uids = [frame.uid for frame in anim.frames]
        return self._reorder(uids[:f0] + uids[f0 : f1 + 1][::-1] + uids[f1 + 1 :])

    def set_range_duration(self: Document, f0: int, f1: int, ms: int) -> bool:
        """One duration across a span, as one step.

        A ``FrameDurationEdit`` per frame that *actually changes*: retiming a
        span most of which is already at that duration must not push a step per
        frame that agrees, or undo walks back through a dozen no-ops.
        """
        anim = self.anim
        span = self._frames(f0, f1)
        if anim is None or span is None:
            return False
        f0, f1 = span
        # Nothing here can be *affected* by a floating buffer -- a duration is
        # not pixels -- but the pattern is the pattern: every op in this module
        # commits before it mutates, and one exception is how the next reader
        # learns that some of them do not have to.
        self.commit_floating()
        after = clamp_duration(ms)
        edits: list[Any] = []
        for frame in anim.frames[f0 : f1 + 1]:
            before = frame.duration_ms
            if before == after:
                continue
            self._set_duration(frame.uid, after)
            edits.append(FrameDurationEdit(frame.uid, before, after))
        return self._push_range(edits)

    # -- track properties ----------------------------------------------------

    def set_range_props(self: Document, t0: int, t1: int, **props: Any) -> bool:
        """One property change across a span of tracks, as one step.

        ``set_range_duration``'s rule one axis over: a ``TrackPropsEdit`` per
        track that *actually changes*, so hiding a span most of which is
        already hidden does not make undo walk back through a row of no-ops.

        The keys are checked against a list rather than trusted, because this
        writes with ``setattr``: an unknown one would otherwise mint a new
        attribute on the track, silently, and be lost at the next save.
        """
        anim = self.anim
        if anim is None:
            return False
        span = clamp_span(t0, t1, len(anim.tracks))
        if span is None:
            return False
        unknown = set(props) - TRACK_PROPS
        if unknown:
            raise ValueError(f"unknown track property: {sorted(unknown)[0]}")
        self.commit_floating()
        edits: list[Any] = []
        for index in range(span[0], span[1] + 1):
            track = anim.tracks[index]
            before = {key: getattr(track, key) for key in props}
            if before == props:
                continue
            for key, value in props.items():
                setattr(track, key, value)
            edits.append(TrackPropsEdit(track.uid, before, dict(props)))
        if not edits:
            return False
        pushed = self._push_range(edits)
        self._anim_changed()
        return pushed

    # -- cels ---------------------------------------------------------------

    def _slots_in(self: Document, rect: tuple[int, int, int, int]) -> list[tuple[Any, Any]]:
        """Every ``(track, frame)`` of a cell rect, frame-major.

        Frame-major so first-occurrence bookkeeping runs in the same order
        ``unique_cel_layers`` walks the grid in -- two orders would make "which
        slot is charged for this cel" depend on which helper asked.
        """
        anim = self.anim
        assert anim is not None
        t0, t1, f0, f1 = rect
        return [
            (track, frame)
            for frame in anim.frames[f0 : f1 + 1]
            for track in anim.tracks[t0 : t1 + 1]
        ]

    def clear_range(self: Document, t0: int, t1: int, f0: int, f1: int) -> bool:
        """Empty every occupied cell of the rect."""
        anim = self.anim
        rect = self._range(t0, t1, f0, f1)
        if anim is None or rect is None:
            return False
        # Committed *before* the slots are read, which is the order the whole
        # module follows and the one that matters here: committing a floating
        # buffer autovivifies the cel it lands in, so a set read first would
        # miss a slot that became occupied a line later and leave a cel behind
        # in the middle of a cleared rect.
        self.commit_floating()
        occupied = [
            (track, frame, anim.cels[(track.uid, frame.uid)])
            for track, frame in self._slots_in(rect)
            if (track.uid, frame.uid) in anim.cels
        ]
        if not occupied:
            return False
        for track, frame, _layer in occupied:
            self._set_cel(track.uid, frame.uid, None)
        return self._push_range(
            self._cel_edits([(track, frame, layer, None) for track, frame, layer in occupied])
        )

    def _cel_edits(
        self: Document,
        changes: list[tuple[Any, Any, Any, Any]],
        *,
        minted: Any = (),
    ) -> list[Any]:
        """``CelSetEdit``s for slots already mutated, each object charged once.

        Two questions, and they are not the same one.

        What a *displaced* cel costs is answered by ``_released``, and only
        after the mutation: the history pins pixels exactly where nothing else
        does, so a cel still linked into a frame outside the range costs
        nothing however this step is undone.

        What a *minted* one costs cannot be answered that way at all -- a fresh
        copy or a pasted plane is in the grid the moment it is put there, so
        ``_released`` says no; but undoing the step takes it back out and
        leaves the edit holding it alone. So the caller names the objects it
        created, and they are charged whatever the grid currently says.

        Either way each object is charged to the **first** slot that mentions
        it, which is what makes a shared cel cost one plane and not one per
        slot.
        """
        released = self._released(
            [layer for _t, _f, before, after in changes for layer in (before, after)]
        )
        new = set(minted)
        seen: set[int] = set()
        edits: list[Any] = []
        for track, frame, before, after in changes:
            pinned: set[int] = set()
            for layer in (before, after):
                if layer is None or id(layer) in seen:
                    continue
                if id(layer) in released or id(layer) in new:
                    seen.add(id(layer))
                    pinned.add(id(layer))
            edits.append(
                CelSetEdit(track.uid, frame.uid, before, after, pinned=frozenset(pinned))
            )
        return edits

    def link_range(self: Document, t0: int, t1: int, f0: int, f1: int) -> bool:
        """Make every cell of a track's span show that track's earliest cel.

        Earliest *occupied*, and the empty slots before it join in too: the
        user selected a block and asked for one drawing across it, and leaving
        a hole at the front because nothing had been drawn there yet would be a
        different answer to a question nobody asked.
        """
        anim = self.anim
        rect = self._range(t0, t1, f0, f1)
        if anim is None or rect is None:
            return False
        # Before the grid is read, as everywhere else here: a floating buffer
        # committing into an empty slot inside the rect autovivifies a cel, and
        # a "which slot holds the earliest drawing" answer taken beforehand
        # would be one drawing out of date.
        self.commit_floating()
        t0, t1, f0, f1 = rect
        frames = anim.frames[f0 : f1 + 1]
        changes: list[tuple[Any, Any, Any, Any]] = []
        for track in anim.tracks[t0 : t1 + 1]:
            source = next(
                (
                    anim.cels[(track.uid, frame.uid)]
                    for frame in frames
                    if (track.uid, frame.uid) in anim.cels
                ),
                None,
            )
            if source is None:
                continue
            for frame in frames:
                before = anim.cels.get((track.uid, frame.uid))
                if before is source:
                    continue
                changes.append((track, frame, before, source))
        if not changes:
            return False
        for track, frame, _before, after in changes:
            self._set_cel(track.uid, frame.uid, after)
        return self._push_range(self._cel_edits(changes))

    def unlink_range(self: Document, t0: int, t1: int, f0: int, f1: int) -> bool:
        """Give every shared cel in the rect a private copy.

        The copies -- and therefore their uids -- are minted **once, here**,
        and the edits hold them. That is ``unlink_cel``'s rule and it is not
        negotiable: a redo that copied again would hand back layers with new
        identities and strand every patch recorded against the first ones.
        """
        anim = self.anim
        rect = self._range(t0, t1, f0, f1)
        if anim is None or rect is None:
            return False
        # The float goes down first, so a cel it just brought into existence is
        # in the set below rather than missed by it.
        self.commit_floating()
        # Which slots are shared is then read before anything *this op* writes:
        # unlinking the first slot of a two-slot link makes the second stop
        # reporting as linked, and it still wants its own copy.
        shared = [
            (track, frame, anim.cels[(track.uid, frame.uid)])
            for track, frame in self._slots_in(rect)
            if anim.is_linked(track.uid, frame.uid)
        ]
        if not shared:
            return False
        changes: list[tuple[Any, Any, Any, Any]] = []
        for track, frame, before in shared:
            copy = before.copy(name=before.name)
            self._set_cel(track.uid, frame.uid, copy)
            changes.append((track, frame, before, copy))
        # Every copy is new memory the history alone holds once this is undone,
        # so each is charged; the original they came from is charged only if
        # nothing else is left pointing at it.
        return self._push_range(
            self._cel_edits(changes, minted={id(after) for *_r, after in changes})
        )

    # -- the cel clipboard --------------------------------------------------

    def copy_cels(self: Document, t0: int, t1: int, f0: int, f1: int) -> CelClip | None:
        """A rectangle of cels, taken. A pure read: nothing is pushed.

        The clip lives app-level (``InkerState.cel_clip``) rather than on the
        document, so a range copied in one tab pastes into another -- which is
        the only reason a user would reach for this over duplicate.
        """
        anim = self.anim
        rect = self._range(t0, t1, f0, f1)
        if anim is None or rect is None:
            return None
        t0, t1, f0, f1 = rect
        planes: list[Any] = []
        index_of: dict[int, int] = {}
        slots: dict[tuple[int, int], int] = {}
        for frame_offset, frame in enumerate(anim.frames[f0 : f1 + 1]):
            for track_offset, track in enumerate(anim.tracks[t0 : t1 + 1]):
                cel = anim.cels.get((track.uid, frame.uid))
                if cel is None:
                    continue
                index = index_of.get(id(cel))
                if index is None:
                    index = len(planes)
                    index_of[id(cel)] = index
                    planes.append(cel.copy(name=cel.name))
                slots[(track_offset, frame_offset)] = index
        return CelClip(
            size=self.size,
            tracks=t1 - t0 + 1,
            frames=f1 - f0 + 1,
            planes=planes,
            slots=slots,
        )

    def paste_cels(self: Document, clip: CelClip | None, t0: int, f0: int) -> bool:
        """Put a clip down with its top-left cell at ``(t0, f0)``.

        Refuses a size mismatch outright rather than scaling or cropping: a cel
        is a full-canvas plane and a document that held one of another size
        would raise on the next flatten, somewhere with no clue about where the
        pixels came from.

        Pasting past the end **conjures blank frames inside the same
        ``CompoundEdit``** -- the autovivify pattern one level up -- so one
        Ctrl+Z takes away the cels and the frames they needed together, rather
        than leaving a tail of empty columns nobody asked for.
        """
        anim = self.anim
        if anim is None or clip is None or not clip.slots:
            return False
        if tuple(clip.size) != tuple(self.size):
            return False
        t0 = max(0, min(int(t0), len(anim.tracks) - 1))
        f0 = max(0, min(int(f0), len(anim.frames) - 1))
        landing = [
            (t0 + track_offset, f0 + frame_offset, index)
            for (track_offset, frame_offset), index in sorted(
                clip.slots.items(), key=lambda item: (item[0][1], item[0][0])
            )
            if t0 + track_offset < len(anim.tracks)
        ]
        if not landing:
            return False
        self.commit_floating()
        edits: list[Any] = []
        while len(anim.frames) < f0 + clip.frames:
            frame = Frame()
            index = len(anim.frames)
            self._put_frame(index, frame, {})
            edits.append(FrameAddEdit(index, frame, {}, pinned=False))
        # One fresh object per *plane*, not per slot: the clip's internal links
        # are the whole reason it stores indices, and minting per slot here
        # would throw them away at the last step.
        fresh = {index: plane.copy(name=plane.name) for index, plane in enumerate(clip.planes)}
        changes: list[tuple[Any, Any, Any, Any]] = []
        for track_index, frame_index, index in landing:
            track = anim.tracks[track_index]
            frame = anim.frames[frame_index]
            layer = fresh[index]
            before = anim.cels.get((track.uid, frame.uid))
            if before is layer:
                continue
            self._set_cel(track.uid, frame.uid, layer)
            changes.append((track, frame, before, layer))
        # The pasted planes are new memory; whatever they displaced is charged
        # only where the grid has genuinely let go of it.
        edits += self._cel_edits(
            changes, minted={id(plane) for plane in fresh.values()}
        )
        return self._push_range(edits)

    # -- permutations --------------------------------------------------------

    def _cels_in(
        self: Document, rect: tuple[int, int, int, int]
    ) -> list[tuple[Any, Any]]:
        """Every *distinct* ``(track, cel)`` of a rect, deduped by ``id()``.

        The one piece of bookkeeping every cel-wise range op needs and the
        easiest thing here to get wrong: a background linked across ten frames
        is one object, and an op that walked the slots would flip it ten times
        -- which for a flip is not ten times the flip, it is the identity on an
        even span and therefore an op that silently did nothing. A cel is keyed
        by ``(track, frame)``, so each distinct one belongs to exactly one
        track and there is no ambiguity about which is returned beside it.

        **The track comes back with it because the track is authoritative.**
        Properties are copied down onto a cel only when its frame is
        materialised, so a cel two frames along carries whatever the lock said
        the last time the playhead was on it -- and a range op reading
        ``cel.alpha_lock`` painted straight through "preserve transparency" on
        every frame but the one on screen. ``layers_for`` states the rule; this
        is the door it has to hold at.
        """
        anim = self.anim
        assert anim is not None
        seen: set[int] = set()
        cels: list[tuple[Any, Any]] = []
        for track, frame in self._slots_in(rect):
            cel = anim.cels.get((track.uid, frame.uid))
            if cel is None or id(cel) in seen:
                continue
            seen.add(id(cel))
            cels.append((track, cel))
        return cels

    def _permute_range(
        self: Document, rect: tuple[int, int, int, int] | None, pix_fn: Any, index_fn: Any
    ) -> bool:
        """One exact, shape-preserving permutation over every cel of a rect.

        Shared by flip, rotate and shift, and what it shares with them is
        ``_map_planes``' index discipline applied one cel at a time: given an
        indexed document the *index plane* is permuted and the pixels re-derived
        from it, never re-resolved from the mapped colours. Two palette slots
        holding one colour therefore come out the other side still two slots.
        Re-resolving would collapse them, and nothing on screen would change --
        which is what would make it silent.

        Whole cels, with no mask: the selection is a *weight*, and a weighted
        permutation would have to invent what goes in the part it did not move.
        The ops that do honour it -- ``fill_range``, ``filter_range`` -- are the
        ones where fading between before and after means something.

        Written in place rather than by rebinding ``layer.pixels``: the flatten
        cache, the texture uploader and an open floating buffer may all be
        holding that array. ``_map_planes`` may rebind because it rebuilds the
        stack afterwards; this does not.
        """
        anim = self.anim
        if anim is None or rect is None:
            return False
        # Committed before the slots are read, as everywhere else in this
        # module: a commit autovivifies the cel it lands in, and a target set
        # taken first would miss it.
        self.commit_floating()
        targets = self._cels_in(rect)
        if not targets:
            return False
        width, height = self.size
        box = (0, 0, width, height)
        edits: list[Any] = []
        for _track, layer in targets:
            if self.color_mode == "indexed" and layer.indices is not None:
                before = layer.indices.copy()
                after = index_fn(layer.indices)
                if np.array_equal(before, after):
                    continue
                layer.indices[...] = after
                self._rematerialize(layer, notify=False)
                edits.append(IndexPatchEdit(layer.uid, box, before, after))
            else:
                before = layer.pixels.copy()
                after = pix_fn(layer.pixels)
                if np.array_equal(before, after):
                    continue
                layer.pixels[...] = after
                edits.append(PatchEdit(layer.uid, box, before, after))
            self._stamp_layer(layer.uid)
        if not edits:
            return False
        pushed = self._push_range(edits)
        self.invalidate_all()
        return pushed

    def flip_range(
        self: Document, axis: str, t0: int, t1: int, f0: int, f1: int
    ) -> bool:
        """Mirror every distinct cel of a rect, as one step.

        ``axis`` is one of ``transform.FLIPS`` and is validated there rather
        than here, which is that tuple's whole reason for existing: a third
        axis cannot be offered by a menu and refused by the function.
        """
        fn = lambda plane: tf.flip(plane, axis)  # noqa: E731
        return self._permute_range(self._range(t0, t1, f0, f1), fn, fn)

    def rotate_range(
        self: Document, quarters: int, t0: int, t1: int, f0: int, f1: int
    ) -> bool:
        """Turn every distinct cel of a rect, counter-clockwise, as one step.

        **Refused on a non-square canvas for an odd number of quarters**, by
        name. A quarter turn transposes the plane and a cel is a full-canvas
        array, so turning the cels without turning the canvas would leave the
        grid holding planes of two different shapes -- which the next flatten
        would raise on, somewhere with no clue about where it came from.
        Turning the canvas instead is ``GeometryOps.rotate90``, and it is a
        different request: it turns the whole document, not a range of it.
        """
        quarters = int(quarters) % 4
        if quarters == 0:
            return False
        width, height = self.size
        if quarters % 2 == 1 and width != height:
            raise ValueError("a 90-degree rotation of a cel range needs a square canvas")
        fn = lambda plane: tf.rotate90(plane, quarters)  # noqa: E731
        return self._permute_range(self._range(t0, t1, f0, f1), fn, fn)

    def shift_range(
        self: Document, dx: int, dy: int, wrap: bool, t0: int, t1: int, f0: int, f1: int
    ) -> bool:
        """Shift every distinct cel of a rect by whole pixels, as one step.

        ``wrap`` carries content round the far edge -- Aseprite's own
        range-shift behaviour, and the only variant that is an exact
        permutation, so an indexed cel keeps its duplicate slots. Without it
        the vacated pixels are transparent, which for an index plane means the
        *transparent index* and not slot zero: ``resize_canvas``'s precedent,
        and the same trap.
        """
        if int(dx) == 0 and int(dy) == 0:
            return False
        return self._permute_range(
            self._range(t0, t1, f0, f1),
            lambda plane: tf.translate(plane, dx, dy, wrap=wrap),
            lambda plane: tf.translate(
                plane, dx, dy, wrap=wrap, fill=self.transparent_index
            ),
        )

    # -- fill -----------------------------------------------------------------

    def fill_range(
        self: Document,
        colour: tuple[int, int, int, int],
        t0: int,
        t1: int,
        f0: int,
        f1: int,
    ) -> bool:
        """Flood every distinct cel of a rect with one colour, as one step.

        ``filter_range``'s shape with a solid plane where the filter was, and
        that is the point: the selection is a **weight** here too, so a
        feathered edge fades the colour in rather than stopping it at a
        rectangle. The permutations above cannot honour it and say so; this
        can, because fading between what was there and one flat colour is
        something a partial coverage can mean.

        Like a filter and unlike a stroke, it never autovivifies: a fill over
        an empty cel is a no-op, not a fresh cel full of colour.
        """
        rect = self._range(t0, t1, f0, f1)
        if rect is None:
            return False
        self.commit_floating()
        width, height = self.size
        bounds = self.mask.bounds if self.mask is not None else None
        box = self.clip(bounds or (0, 0, width, height))
        if box is None:
            return False
        x0, y0, x1, y1 = box
        weight = None if self.mask is None else self.mask.mask[y0:y1, x0:x1]
        targets = self._cels_in(rect)
        if not targets:
            return False
        # Minted once and never written into: ``masked_apply`` may hand this
        # very array back when there is no weight and no lock, and the
        # assignment below copies out of it.
        solid = np.full((y1 - y0, x1 - x0, 4), colour, dtype=np.uint8)
        edits: list[Any] = []
        for track, layer in targets:
            before = layer.pixels[y0:y1, x0:x1].copy()
            layer.pixels[y0:y1, x0:x1] = masked_apply(
                before, solid, weight, alpha_lock=track.alpha_lock
            )
            edit = self._patch_edit_for(layer, box, before)
            if edit is None:
                continue
            self._stamp_layer(layer.uid)
            edits.append(edit)
        if not edits:
            return False
        pushed = self._push_range(edits)
        self.invalidate_all()
        return pushed

    # -- filters ------------------------------------------------------------

    def filter_range(
        self: Document,
        name: str,
        params: dict[str, Any],
        t0: int,
        t1: int,
        f0: int,
        f1: int,
    ) -> bool:
        """Run one filter over every distinct cel of a rect, as one step.

        **Deliberately parallel to the filter session rather than sharing it.**
        A session is a live preview over *one* layer: it takes a snapshot,
        recomputes from it every frame while a slider moves, and commits one
        patch. This is a bounded, mask-weighted write over many cels with no
        preview at all -- the contrast is the whole design. Routing this
        through ``begin_filter``/``commit_filter`` would mean one session per
        cel, each of them stamping and recompositing, to show a preview of a
        frame that is not on screen.

        A cel is filtered **once however many slots hold it** -- see
        ``_cels_in``, which is where that dedupe lives now that three other ops
        need it.

        It never autovivifies. A filter over an empty cel is a no-op, not a
        fresh transparent cel with a filtered nothing in it -- the write paths
        conjure cels because a *stroke* has pixels to put down, and this has
        none.
        """
        anim = self.anim
        rect = self._range(t0, t1, f0, f1)
        if anim is None or rect is None:
            return False
        # Before the mask, the box and the cels are read: a floating buffer is
        # pixels the user can see and no layer holds, so filtering around it
        # would filter a picture that is not the one on screen -- and the cel
        # its commit conjures has to be in the target set, not missed by it.
        self.commit_floating()
        width, height = self.size
        bounds = self.mask.bounds if self.mask is not None else None
        box = self.clip(bounds or (0, 0, width, height))
        if box is None:
            return False
        x0, y0, x1, y1 = box
        weight = None if self.mask is None else self.mask.mask[y0:y1, x0:x1]
        targets = self._cels_in(rect)
        if not targets:
            return False
        edits: list[Any] = []
        for track, layer in targets:
            before = layer.pixels[y0:y1, x0:x1].copy()
            filtered = filters.apply_named(name, before, **params)
            layer.pixels[y0:y1, x0:x1] = masked_apply(
                before, filtered, weight, alpha_lock=track.alpha_lock
            )
            # Through ``_patch_edit_for`` rather than straight into a
            # ``PatchEdit``, and that is the whole of what makes this op
            # mode-aware: an indexed cel filtered with a raw RGBA write left
            # ``layer.indices`` describing the picture from before it, and the
            # document went on looking right until it was saved or undone.
            # It also subsumes the old no-op test -- ``None`` *is* "nothing
            # changed", reached by whichever arithmetic the mode uses.
            edit = self._patch_edit_for(layer, box, before)
            if edit is None:
                continue
            # Per touched layer, never ``_stamp_all``: the frames a cel appears
            # on are exactly the flattens this write invalidates, and stamping
            # the whole timeline would throw away every other frame's cache for
            # a write that did not touch it. After the no-op test, so a write
            # the mode resolved back to nothing stamps nothing either.
            self._stamp_layer(layer.uid)
            edits.append(edit)
        if not edits:
            return False
        pushed = self._push_range(edits)
        self.invalidate_all()
        return pushed
