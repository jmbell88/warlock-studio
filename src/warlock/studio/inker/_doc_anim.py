"""The animation grid: frames, tracks, cels and tags.

The grid is the second model in the editor and this is all of it that lives on
the document. ``_materialize_frame`` is the join between the two -- panes,
tools, the floating buffer, the selection and the compositor all go on seeing an
ordinary ``LayerStack``, because that is genuinely what they are handed.

Two neighbours deliberately stay in ``document.py``. The **flatten cache**
(``frame_stamp``/``frame_flat``/``_evict_frames`` and the ``_stamp*`` family)
does, because ``_evict_frames`` reads ``FRAME_CACHE_BYTES`` out of its own
module globals and the ceiling is monkeypatched by uid there. So does **cel
autovivify** (``_ensure_cel_for`` and friends), which is a write path rather
than a grid edit: every method that puts pixels down calls it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .anim_edits import (
    AnimateEdit,
    CelSetEdit,
    FrameAddEdit,
    FrameDurationEdit,
    FrameMoveEdit,
    FrameRemoveEdit,
    TagsEdit,
)
from .animation import Animation, Frame, Tag, Track, clamp_duration
from .layers import Layer, LayerStack
from .undo import one_step

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


class AnimOps:
    """The frames-by-tracks grid, mixed into :class:`~.document.Document`."""

    @property
    def animated(self: Document) -> bool:
        return self.anim is not None

    def _released(self: Document, layers: Any) -> frozenset[int]:
        """Which of these the grid no longer holds, by ``id()``.

        Called *after* a removal, and it is the answer the byte budget wants:
        the history pins pixels only where nothing else does, so a cel still
        linked into another frame costs the budget nothing however the step is
        undone. See ``anim_edits.charged``.
        """
        live = (
            set()
            if self.anim is None
            else {id(layer) for layer in self.anim.cels.values()}
        )
        return frozenset(
            id(layer) for layer in layers if layer is not None and id(layer) not in live
        )

    def ensure_animation(self: Document) -> Animation:
        """Turn a still document into a one-frame animation, in place.

        Nothing is copied and nothing is renumbered: each existing ``Layer``
        *becomes* the first frame's cel and each ``Track`` takes that layer's
        uid, so every patch already on the undo stack goes on addressing the
        same pixels. That is what lets the first "Add frame" be undone back to a
        plain document without the history underneath it noticing.
        """
        if self.anim is not None:
            return self.anim
        frame = Frame()
        tracks = [Track.of(layer) for layer in self.stack]
        self.anim = Animation(
            tracks=tracks,
            frames=[frame],
            cels={
                (track.uid, frame.uid): layer
                for track, layer in zip(tracks, self.stack, strict=True)
            },
            current=0,
        )
        return self.anim

    def drop_animation(self: Document) -> None:
        """Back to a still document showing whatever frame is current.

        The undo hook for ``AnimateEdit``. The stack is left exactly as it
        stands, which is right because ``ensure_animation`` did not build it --
        on the only path that matters (animate, then undo) it is still the
        original layer objects.
        """
        self.anim = None
        self._forget_all_frames()

    def _materialize_frame(self: Document, *, active: int | None = None) -> None:
        """Rebuild ``stack`` as the view of the current frame.

        This is the join between the two models, and the reason the rest of the
        editor needed almost no changes: panes, tools, the floating buffer, the
        selection, the ``_below`` cache and the compositor all go on seeing an
        ordinary ``LayerStack``, because that is genuinely what they are handed.
        """
        if self.anim is None or not self.anim.frames:
            return
        layers = self.anim.layers_for(self.anim.frame, self.size)
        if not layers:
            # A grid with no tracks. ``LayerStack`` has no empty form, so the
            # previous frame's list stays -- which is only ever reached while an
            # edit is midway through rebuilding the rows, and the edit ends with
            # another ``_anim_changed``.
            return
        index = self.stack.active_index if active is None else active
        self.stack = LayerStack(layers, max(0, min(index, len(layers) - 1)))

    def set_current_frame(self: Document, index: int) -> None:
        """Move the playhead. Not undoable -- it is view state, like the
        active layer, and a document that asked to be saved because the user
        looked at another frame would make ``dirty`` a lie."""
        if self.anim is None:
            return
        index = max(0, min(int(index), len(self.anim.frames) - 1))
        if index == self.anim.current:
            return
        self.commit_floating()
        self.anim.current = index
        self._materialize_frame()
        self.invalidate_all()

    def _anim_changed(self: Document, *, active: int | None = None) -> None:
        """What every grid edit ends with: rebuild the view, recomposite.

        Unconditional rather than "when the current frame is affected". Working
        out whether it was means asking whether a track property, a reorder or a
        link touched the frame on screen, and the cost of being wrong is a
        viewport showing pixels the document no longer has -- against a full
        recomposite on a click, which ``invalidate_all`` already argues is
        affordable.

        Every frame is stamped, for the reason ``invalidate_all`` deliberately
        does not: a track is authoritative over every frame it appears in, so a
        hide or an opacity change really does alter every cached flatten in the
        document.
        """
        self._stamp_all()
        self._materialize_frame(active=active)
        self.invalidate_all()

    # -- raw grid mutation, for the edits to call --------------------------

    def _set_animation(self: Document, anim: Animation | None) -> None:
        self.anim = anim
        if anim is None:
            self._forget_all_frames()
        self._anim_changed()

    def _put_frame(self: Document, index: int, frame: Frame, cels: dict[int, Layer]) -> None:
        anim = self.anim
        assert anim is not None
        anim.frames.insert(max(0, min(index, len(anim.frames))), frame)
        for track_uid, layer in cels.items():
            anim.cels[(track_uid, frame.uid)] = layer
        self._anim_changed()

    def _drop_frame(self: Document, frame: Frame) -> None:
        anim = self.anim
        assert anim is not None
        # The floor is the public wrappers' -- ``remove_frame`` refuses the last
        # frame, and the only other caller is ``FrameAddEdit.undo``, whose add
        # was made against a grid that already had one. Stated here because
        # everything below assumes ``anim.frame`` still answers.
        assert len(anim.frames) > 1, "a grid keeps at least one frame"
        anim.frames.pop(anim.frame_index(frame.uid))
        for key in [key for key in anim.cels if key[1] == frame.uid]:
            del anim.cels[key]
        anim.forget_placeholders(frame_uid=frame.uid)
        self._forget_frame(frame.uid)
        anim.current = max(0, min(anim.current, len(anim.frames) - 1))
        self._anim_changed()

    def _move_frame(self: Document, frame_uid: int, to: int) -> None:
        anim = self.anim
        assert anim is not None
        frame = anim.frames.pop(anim.frame_index(frame_uid))
        anim.frames.insert(max(0, min(to, len(anim.frames))), frame)
        anim.current = anim.frame_index(frame_uid)
        self._anim_changed()

    def _set_duration(self: Document, frame_uid: int, ms: int) -> None:
        anim = self.anim
        assert anim is not None
        anim.frames[anim.frame_index(frame_uid)].duration_ms = clamp_duration(ms)
        self.rev += 1

    def _put_track(self: Document, index: int, track: Track, cels: dict[int, Layer]) -> None:
        anim = self.anim
        assert anim is not None
        index = max(0, min(index, len(anim.tracks)))
        anim.tracks.insert(index, track)
        for frame_uid, layer in cels.items():
            anim.cels[(track.uid, frame_uid)] = layer
        self._anim_changed(active=index)

    def _drop_track(self: Document, track: Track) -> None:
        anim = self.anim
        assert anim is not None
        # As in ``_drop_frame``: ``remove_layer`` refuses the last row and
        # ``TrackAddEdit.undo`` can only reverse an add made above one.
        assert len(anim.tracks) > 1, "a grid keeps at least one track"
        index = anim.track_index(track.uid)
        anim.tracks.pop(index)
        for key in [key for key in anim.cels if key[0] == track.uid]:
            del anim.cels[key]
        anim.forget_placeholders(track_uid=track.uid)
        self._anim_changed(active=min(index, len(anim.tracks) - 1))

    def _move_track(self: Document, track_uid: int, to: int) -> None:
        anim = self.anim
        assert anim is not None
        track = anim.tracks.pop(anim.track_index(track_uid))
        to = max(0, min(to, len(anim.tracks)))
        anim.tracks.insert(to, track)
        self._anim_changed(active=to)

    def _set_track_props(self: Document, track_uid: int, props: dict) -> None:
        anim = self.anim
        assert anim is not None
        track = anim.tracks[anim.track_index(track_uid)]
        for key, value in props.items():
            setattr(track, key, value)
        self._anim_changed()

    def _set_cel(self: Document, track_uid: int, frame_uid: int, layer: Layer | None) -> None:
        anim = self.anim
        assert anim is not None
        if layer is None:
            anim.cels.pop((track_uid, frame_uid), None)
        else:
            anim.cels[(track_uid, frame_uid)] = layer
        # No targeted stamp: ``_anim_changed`` stamps every frame, this one
        # included, and a second bump of the same counter buys nothing.
        self._anim_changed()

    # -- frames -------------------------------------------------------------

    def add_frame(
        self: Document, index: int | None = None, *, link: bool = False, copy: bool = False
    ) -> Frame:
        """Append or insert a frame, optionally carrying the current one's cels.

        Three shapes in one entry point because they differ only in what goes
        into ``cels``: blank (nothing), duplicate-linked (the same objects, so
        editing either frame edits both) and duplicate-copied (fresh copies).

        The first call on a still document is one ``CompoundEdit`` of
        ``AnimateEdit`` and this ``FrameAddEdit``, so a single Ctrl+Z goes all
        the way back to a plain document rather than leaving a one-frame
        animation nobody asked for.
        """
        self.commit_floating()
        edits: list[Any] = []
        if self.anim is None:
            edits.append(AnimateEdit(self.ensure_animation()))
        anim = self.anim
        assert anim is not None
        source = anim.frame
        at = len(anim.frames) if index is None else max(0, min(int(index), len(anim.frames)))
        frame = Frame(duration_ms=source.duration_ms)
        cels: dict[int, Layer] = {}
        if link or copy:
            for track in anim.tracks:
                cel = anim.cels.get((track.uid, source.uid))
                if cel is not None:
                    cels[track.uid] = cel if link else cel.copy(name=cel.name)
        # The playhead moves *before* the insert, so ``_put_frame``'s own
        # ``_anim_changed`` materialises the new frame -- rather than after it,
        # which meant rebuilding the whole view twice for one click.
        anim.current = at
        self._put_frame(at, frame, cels)
        # ``pinned`` only when this op allocated pixels of its own. A linked
        # duplicate holds objects the grid is keeping alive anyway, and charging
        # for them again would evict real history to make room for a number that
        # describes no memory.
        edits.append(FrameAddEdit(at, frame, cels, pinned=copy))
        self.history.push(one_step(edits))
        return frame

    def remove_frame(self: Document, index: int | None = None) -> bool:
        anim = self.anim
        if anim is None or len(anim.frames) <= 1:
            return False
        self.commit_floating()
        index = anim.current if index is None else max(0, min(int(index), len(anim.frames) - 1))
        frame = anim.frames[index]
        cels = {
            track.uid: anim.cels[(track.uid, frame.uid)]
            for track in anim.tracks
            if (track.uid, frame.uid) in anim.cels
        }
        self._drop_frame(frame)
        self.history.push(
            FrameRemoveEdit(index, frame, cels, pinned=self._released(cels.values()))
        )
        return True

    def move_frame(self: Document, index: int, to: int) -> bool:
        anim = self.anim
        if anim is None:
            return False
        to = max(0, min(int(to), len(anim.frames) - 1))
        if to == index or not 0 <= index < len(anim.frames):
            return False
        self.commit_floating()
        uid = anim.frames[index].uid
        self._move_frame(uid, to)
        self.history.push(FrameMoveEdit(uid, index, to))
        return True

    def set_frame_duration(self: Document, index: int, ms: int) -> bool:
        anim = self.anim
        if anim is None or not 0 <= index < len(anim.frames):
            return False
        frame = anim.frames[index]
        before, after = frame.duration_ms, clamp_duration(ms)
        if before == after:
            return False
        self._set_duration(frame.uid, after)
        self.history.push(FrameDurationEdit(frame.uid, before, after))
        return True

    # -- cels ---------------------------------------------------------------

    def _slot(
        self: Document, track_index: int | None, frame_index: int | None
    ) -> tuple[Track, Frame] | None:
        anim = self.anim
        if anim is None:
            return None
        ti = self.stack.active_index if track_index is None else track_index
        fi = anim.current if frame_index is None else frame_index
        if not (0 <= ti < len(anim.tracks) and 0 <= fi < len(anim.frames)):
            return None
        return anim.tracks[ti], anim.frames[fi]

    def clear_cel(
        self: Document, track_index: int | None = None, frame_index: int | None = None
    ) -> bool:
        slot = self._slot(track_index, frame_index)
        if slot is None:
            return False
        track, frame = slot
        assert self.anim is not None
        before = self.anim.cels.get((track.uid, frame.uid))
        if before is None:
            return False
        self.commit_floating()
        self._set_cel(track.uid, frame.uid, None)
        # Clearing one slot of a linked cel releases nothing: the object is
        # alive in the frames it is still linked into.
        self.history.push(
            CelSetEdit(
                track.uid, frame.uid, before, None, pinned=bool(self._released([before]))
            )
        )
        return True

    def link_cel(
        self: Document,
        source_frame: int,
        track_index: int | None = None,
        frame_index: int | None = None,
    ) -> bool:
        """Point a slot at another frame's cel, so the two share one object."""
        slot = self._slot(track_index, frame_index)
        anim = self.anim
        if slot is None or anim is None or not 0 <= source_frame < len(anim.frames):
            return False
        track, frame = slot
        source = anim.cels.get((track.uid, anim.frames[source_frame].uid))
        before = anim.cels.get((track.uid, frame.uid))
        if source is None or source is before:
            return False
        self.commit_floating()
        self._set_cel(track.uid, frame.uid, source)
        # ``pinned=False``: the shared object is alive in the frame it came
        # from whatever the history does with this step.
        self.history.push(CelSetEdit(track.uid, frame.uid, before, source, pinned=False))
        return True

    def unlink_cel(
        self: Document, track_index: int | None = None, frame_index: int | None = None
    ) -> bool:
        """Give a linked slot a private copy, so editing it stops editing the
        other frames.

        The copy -- and therefore its uid -- is made **once, here**, and the
        edit holds it. A redo that copied again would mint a new identity and
        strand every patch recorded against the first one, which is the same
        rule ``flatten_layers`` follows when it draws its uid up front.
        """
        slot = self._slot(track_index, frame_index)
        anim = self.anim
        if slot is None or anim is None:
            return False
        track, frame = slot
        if not anim.is_linked(track.uid, frame.uid):
            return False
        before = anim.cels[(track.uid, frame.uid)]
        self.commit_floating()
        copy = before.copy(name=before.name)
        self._set_cel(track.uid, frame.uid, copy)
        self.history.push(CelSetEdit(track.uid, frame.uid, before, copy, pinned=True))
        return True

    # -- tags ---------------------------------------------------------------

    def _set_tags(self: Document, tags: list[Tag]) -> None:
        """Undo hook for every tag op. Installs *copies*.

        The edit holds the lists it was given and undo/redo may run any number
        of times, so handing the document the same ``Tag`` objects would let the
        next rename write through into the step that is meant to reverse it.
        """
        anim = self.anim
        if anim is None:
            return
        anim.tags = [replace(tag) for tag in tags]
        self.rev += 1

    def _clamped_tag(self: Document, tag: Tag) -> Tag:
        """A tag with its span inside the timeline and its ends the right way up."""
        anim = self.anim
        assert anim is not None
        last = len(anim.frames) - 1
        start = max(0, min(int(tag.start), last))
        end = max(0, min(int(tag.end), last))
        return replace(
            tag,
            name=(tag.name.strip() or "tag"),
            start=min(start, end),
            end=max(start, end),
            loop=bool(tag.loop),
            repeat=max(0, int(tag.repeat)),
        )

    def _push_tags(self: Document, tags: list[Tag]) -> bool:
        """Install a new tag list as one undo step, or nothing if it is the same.

        The no-op check is the rule every other op here follows: dirty is a
        comparison against ``history.head``, so a step that changes nothing
        makes a saved document ask to be saved. ``Tag`` is a plain dataclass, so
        ``==`` is the value comparison this wants.
        """
        anim = self.anim
        if anim is None:
            return False
        before = [replace(tag) for tag in anim.tags]
        after = [self._clamped_tag(tag) for tag in tags]
        if before == after:
            return False
        self._set_tags(after)
        self.history.push(TagsEdit(before, after))
        return True

    def add_tag(
        self: Document, name: str, start: int, end: int | None = None, *, loop: bool = True
    ) -> bool:
        """A named span of frames. Overlaps are allowed -- ``active_tag`` picks
        the innermost, which is what makes a short "hit" inside a long "combat"
        the useful arrangement rather than an ambiguous one."""
        if self.anim is None:
            return False
        span = Tag(name=name, start=start, end=start if end is None else end, loop=loop)
        return self._push_tags([*self.anim.tags, span])

    def remove_tag(self: Document, index: int) -> bool:
        anim = self.anim
        if anim is None or not 0 <= index < len(anim.tags):
            return False
        return self._push_tags([t for i, t in enumerate(anim.tags) if i != index])

    def set_tag(self: Document, index: int, **props: Any) -> bool:
        """Rename, retime or re-loop one tag. Unnamed keys are left alone."""
        anim = self.anim
        if anim is None or not 0 <= index < len(anim.tags):
            return False
        edited = replace(anim.tags[index], **props)
        return self._push_tags(
            [edited if i == index else t for i, t in enumerate(anim.tags)]
        )
